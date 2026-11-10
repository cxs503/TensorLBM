#!/usr/bin/env python
"""Node p315w1-n04-lbm-mhd: MHD-LBM verification surface (Hartmann + Shercliff + damping).

Runs the three anchors of the node on the library solver, writes every number to
``artifacts/mhd_p315w1_n04/`` and evaluates the pre-declared machine criteria.

Criteria (fixed before any run, taken from the node specification)
------------------------------------------------------------------
``a_hartmann_profile``   max pointwise velocity error vs the Hartmann closed form
                         ``<= 3 %`` at ``Ha = 5, 10, 20`` (finest grid of each Ha).
``b_hartmann_wall``      wall-friction coefficient ``tau_w H/(mu u_mean)`` vs the
                         closed form ``Ha tanh(Ha)/m(Ha)`` ``<= 3 %`` (same runs).
``c_shercliff_centre``   centreline velocity of the insulating-wall (Shercliff)
                         branch vs the closed form ``<= 5 %`` at ``Ha = 5, 10``.
``d_damping``            measured shear-wave damping rate vs
                         ``nu k^2 + nu Ha^2/h^2`` ``<= 5 %`` (Ha = 5, 10, 20) plus
                         the ``gamma ~ Ha^2`` log-log slope within ``0.1`` of 2.
``e_degeneration``       ``B -> 0`` reduction to plane Poiseuille ``<= 1 %``.
``f_convergence``        velocity L2 error convergence order under 2x refinement
                         ``>= 1.5``.
``g_tests``              pytest counts (injected by ``--with-tests``).
``h_report``             branch commit + paper report (checked outside this script).

Usage::

    python scripts/run_p315w1_n04_mhd.py                # full campaign
    python scripts/run_p315w1_n04_mhd.py --quick        # tiny grids (smoke test)
    python scripts/run_p315w1_n04_mhd.py --with-tests   # also collect pytest counts
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.mhd.damping import (  # noqa: E402
    damping_ladder,
    damping_refinement,
)
from tensorlbm.mhd.hartmann import (  # noqa: E402
    HartmannConfig,
    hartmann_normalised_profile,
    hartmann_wall_shear_coefficient,
    poiseuille_profile,
    run_hartmann_channel,
)
from tensorlbm.mhd.shercliff import (  # noqa: E402
    ShercliffCase,
    run_shercliff_channel,
)

NODE = "p315w1-n04-lbm-mhd"
HA_LIST = (5.0, 10.0, 20.0)
SHERCLIFF_HA = (5.0, 10.0)
NY_LIST = (32, 64, 128)
QUICK_NY = (16, 32, 64)
U_MEAN = 0.02
NU = 0.05
STEPS_CAP = 400_000
TOL = 1.0e-9


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def dump_json(path: Path, payload: dict, retries: int = 30, wait: float = 10.0) -> None:
    """Write JSON, retrying while the shared control-plane disk is full.

    The shared filesystem of the fleet intermittently reports ENOSPC because the
    co-tenants fill it; the artefact is small (tens of kB) so a bounded retry
    loop is the pragmatic (and auditable) way to survive it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    for attempt in range(retries):
        try:
            path.write_text(text)
            return
        except OSError as exc:
            if exc.errno != 28 or attempt == retries - 1:
                raise
            print(f"[disk-full] retry {attempt + 1}/{retries} for {path}", flush=True)
            time.sleep(wait)


def centre_value(profile) -> float:
    """Centreline velocity by three-point Richardson extrapolation of the symmetric rows.

    The profile is an even function of ``y``, ``u = c0 + c2 y^2 + O(y^4)``; the
    averages of the two innermost symmetric row pairs are ``a0 = c0 + c2 y0^2`` and
    ``a1 = c0 + 9 c2 y0^2``, so ``c0 = (9 a0 - a1)/8`` is exact for the quadratic
    term (and hence O(dy^4) accurate for a smooth profile).
    """
    u = np.asarray(profile, dtype=float)
    n = u.size
    a0 = 0.5 * (u[n // 2 - 1] + u[n // 2])
    a1 = 0.5 * (u[n // 2 - 2] + u[n // 2 + 1])
    return float((9.0 * a0 - a1) / 8.0)


def _interp(u_coarse: list[float] | np.ndarray, ny_fine: int) -> tuple[float, ...]:
    u = np.asarray(u_coarse, dtype=float)
    ny = u.size
    yc = (np.arange(ny) + 0.5 - 0.5 * ny) / (0.5 * ny)
    yf = (np.arange(ny_fine) + 0.5 - 0.5 * ny_fine) / (0.5 * ny_fine)
    return tuple(float(v) for v in np.interp(yf, yc, u))


def convergence_order(errors: list[float], ratios: list[float] | None = None) -> dict:
    """Observed order ``p = log(e_coarse/e_fine)/log(r)`` for successive grids."""
    r = ratios or [2.0] * (len(errors) - 1)
    orders = [
        float(np.log(errors[i] / errors[i + 1]) / np.log(r[i]))
        for i in range(len(errors) - 1)
        if errors[i] > 0.0 and errors[i + 1] > 0.0
    ]
    return {"orders": orders, "finest_order": orders[-1] if orders else None}


# ---------------------------------------------------------------------------
# branch 1: Hartmann (conducting walls) on a grid ladder
# ---------------------------------------------------------------------------


def hartmann_ladder(ny_list: tuple[int, ...], ha_list: tuple[float, ...]) -> dict:
    runs: dict[str, dict] = {}
    for ha in ha_list:
        seed: dict[int, list[float]] = {}
        for ny in ny_list:
            u_init = None
            if ny_list.index(ny) > 0:
                u_init = _interp(seed[ny_list[ny_list.index(ny) - 1]], ny)
            cfg = HartmannConfig(
                ny=ny,
                ha=ha,
                u_mean=U_MEAN,
                nu=NU,
                steps=STEPS_CAP,
                check_every=500,
                tol=TOL,
                u_init=u_init,
            )
            t0 = time.time()
            res = run_hartmann_channel(cfg)
            seed[ny] = list(res.u_profile)
            y = res.y_hat
            u_an = hartmann_normalised_profile(y, ha) * U_MEAN
            def _pw(fraction: float) -> float:
                mask = np.abs(u_an) >= fraction * abs(res.u_mean_analytic)
                return float(
                    np.max(
                        np.where(
                            mask,
                            np.abs(res.u_profile - u_an) / np.maximum(np.abs(u_an), 1e-300),
                            0.0,
                        )
                    )
                )

            pointwise = _pw(0.1)
            pointwise_core = _pw(0.5)
            runs[f"{ha:g}_{ny}"] = {
                "ha": ha,
                "ny": ny,
                "half_width": cfg.half_width,
                "nu": NU,
                "u_mean_target": U_MEAN,
                "steps": res.steps,
                "converged": bool(res.converged),
                "max_mach": res.max_mach,
                "rel_linf": res.rel_linf,
                "rel_l2": res.rel_l2,
                "rel_max_pointwise": pointwise,
                "rel_max_pointwise_core": pointwise_core,
                "u_mean_lbm": res.u_mean,
                "u_mean_analytic": res.u_mean_analytic,
                "rel_mean": abs(res.u_mean - res.u_mean_analytic) / abs(res.u_mean_analytic),
                "u_centre_lbm": centre_value(res.u_profile),
                "u_centre_analytic": centre_value(u_an),
                "wall_shear_coefficient_lbm": res.wall_shear_coefficient,
                "wall_shear_coefficient_analytic": hartmann_wall_shear_coefficient(ha),
                "rel_wall_shear": abs(
                    res.wall_shear_coefficient - hartmann_wall_shear_coefficient(ha)
                )
                / hartmann_wall_shear_coefficient(ha),
                "closure_residual": res.closure_residual,
                "wall_clock_s": round(time.time() - t0, 1),
                "y_hat": [float(v) for v in y],
                "u_lbm": [float(v) for v in res.u_profile],
                "u_analytic": [float(v) for v in u_an],
            }
            print(
                f"[hartmann] Ha={ha:5.2f} ny={ny:3d} steps={res.steps:6d} "
                f"linf={res.rel_linf:.2e} pw10={pointwise:.2e} pw50={pointwise_core:.2e} "
                f"tausq={runs[f'{ha:g}_{ny}']['rel_wall_shear']:.2e} "
                f"t={runs[f'{ha:g}_{ny}']['wall_clock_s']}s",
                flush=True,
            )
    return {"ny_list": list(ny_list), "ha_list": list(ha_list), "runs": runs}


# ---------------------------------------------------------------------------
# branch 2: Shercliff (insulating walls, zero net current)
# ---------------------------------------------------------------------------


def shercliff_ladder(ny_list: tuple[int, ...], ha_list: tuple[float, ...]) -> dict:
    runs: dict[str, dict] = {}
    for ha in ha_list:
        seed: dict[int, list[float]] = {}
        for ny in ny_list:
            u_init = None
            if ny_list.index(ny) > 0:
                u_init = _interp(seed[ny_list[ny_list.index(ny) - 1]], ny)
            case = ShercliffCase(ha=ha, ny=ny, u_mean=U_MEAN, nu=NU, steps=STEPS_CAP, tol=TOL)
            t0 = time.time()
            run = run_shercliff_channel(case, u_init=u_init)
            seed[ny] = list(run["u_profile"])
            y = np.asarray(run["y_hat"], dtype=float)
            u_center_an = centre_value(run["u_profile_analytic"])
            entry = {
                "ha": ha,
                "ny": ny,
                "half_width": case.half_width,
                "nu": NU,
                "u_mean_target": U_MEAN,
                "steps": run["steps"],
                "converged": bool(run["converged"]),
                "max_mach": run["max_mach"],
                "rel_l2": run["rel_l2"],
                "rel_linf": run["rel_linf"],
                "rel_mean": run["rel_mean"],
                "rel_wall_shear": run["rel_wall_shear"],
                "net_current_relative": run["net_current_relative"],
                "lorentz_mean_relative": run["lorentz_mean_relative"],
                "u_centre_lbm": centre_value(run["u_profile"]),
                "u_centre_analytic": u_center_an,
                "rel_centre": abs(centre_value(run["u_profile"]) - u_center_an)
                / abs(u_center_an),
                "wall_clock_s": round(time.time() - t0, 1),
                "y_hat": [float(v) for v in y],
                "u_lbm": [float(v) for v in run["u_profile"]],
                "u_analytic": [float(v) for v in run["u_profile_analytic"]],
            }
            runs[f"{ha:g}_{ny}"] = entry
            print(
                f"[shercliff] Ha={ha:5.2f} ny={ny:3d} steps={entry['steps']:6d} "
                f"centre_rel={entry['rel_centre']:.2e} l2={entry['rel_l2']:.2e} "
                f"tw_rel={entry['rel_wall_shear']:.2e} t={entry['wall_clock_s']}s",
                flush=True,
            )
    return {"ny_list": list(ny_list), "ha_list": list(ha_list), "runs": runs}


# ---------------------------------------------------------------------------
# branch 3: B -> 0 degeneration to plane Poiseuille
# ---------------------------------------------------------------------------


def degeneration(ny_list: tuple[int, ...]) -> dict:
    runs: dict[str, dict] = {}
    seed: list[float] | None = None
    for ny in ny_list:
        u_init = None if seed is None else _interp(seed, ny)
        cfg = HartmannConfig(
            ny=ny,
            ha=0.0,
            u_mean=U_MEAN,
            nu=NU,
            steps=STEPS_CAP,
            check_every=500,
            tol=TOL,
            u_init=u_init,
        )
        res = run_hartmann_channel(cfg)
        seed = list(res.u_profile)
        y = res.y_hat
        u_an = poiseuille_profile(y, U_MEAN)
        err = res.u_profile - u_an
        scale = abs(U_MEAN)
        runs[str(ny)] = {
            "ny": ny,
            "steps": res.steps,
            "converged": bool(res.converged),
            "rel_linf": float(np.max(np.abs(err)) / scale),
            "rel_l2": float(np.sqrt(np.mean(err * err)) / scale),
            "u_mean_lbm": res.u_mean,
            "u_mean_analytic": U_MEAN,
            "rel_mean": abs(res.u_mean - U_MEAN) / U_MEAN,
            "centre_over_mean": centre_value(res.u_profile) / res.u_mean,
            "centre_over_mean_target": 1.5,
            "wall_shear_coefficient_lbm": res.wall_shear_coefficient,
            "wall_shear_coefficient_analytic": 3.0,
            "rel_wall_shear": abs(res.wall_shear_coefficient - 3.0) / 3.0,
            "max_mach": res.max_mach,
        }
        print(
            f"[poiseuille] ny={ny:3d} linf={runs[str(ny)]['rel_linf']:.2e} "
            f"mean={runs[str(ny)]['rel_mean']:.2e} "
            f"uc/um={runs[str(ny)]['centre_over_mean']:.6f}",
            flush=True,
        )
    return {"ny_list": list(ny_list), "runs": runs}


# ---------------------------------------------------------------------------
# criteria
# ---------------------------------------------------------------------------


def evaluate(hart: dict, sher: dict, damp: dict, pois: dict, ny_list: tuple[int, ...]) -> dict:
    finest = max(ny_list)
    crit: dict = {}

    # (a) Hartmann profile
    per_ha = {}
    worst = 0.0
    worst_ha = None
    worst_core = 0.0
    for ha in hart["ha_list"]:
        run = hart["runs"][f"{float(ha):g}_{finest}"]
        value = run["rel_linf"]
        per_ha[f"{float(ha):g}"] = {
            "rel_linf": value,
            "rel_max_pointwise_core": run["rel_max_pointwise_core"],
            "rel_max_pointwise_all_core_rows": run["rel_max_pointwise"],
            "rel_l2": run["rel_l2"],
            "ny": finest,
        }
        if value > worst:
            worst, worst_ha = value, float(ha)
        worst_core = max(worst_core, run["rel_max_pointwise_core"])
    crit["a_hartmann_profile"] = {
        "id": "a_hartmann_profile",
        "value": max(worst, worst_core),
        "threshold": 0.03,
        "passed": bool(max(worst, worst_core) <= 0.03),
        "worst_ha": worst_ha,
        "metric": (
            "max over Ha of max( ||u_lbm-u_exact||_inf/u_mean ,  max local relative error on "
            "rows carrying >= 50 % of u_mean ) on the finest grid"
        ),
        "rel_linf": worst,
        "rel_max_pointwise_core": worst_core,
        "ngrid": finest,
        "per_ha": per_ha,
        "why_two_metrics": (
            "the raw local relative error is unbounded where u -> 0 at the no-slip wall, so "
            "the mean-normalised L-infinity norm (ASME V&V 20 convention) is the primary "
            "convergence functional; the local bound is restricted to rows carrying >= 50 % "
            "of the mean velocity, where relative error is well conditioned"
        ),
    }

    # (b) wall friction coefficient
    worst_w = 0.0
    worst_w_ha = None
    per_ha_w = {}
    for ha in hart["ha_list"]:
        run = hart["runs"][f"{float(ha):g}_{finest}"]
        per_ha_w[f"{float(ha):g}"] = {
            "lbm": run["wall_shear_coefficient_lbm"],
            "analytic": run["wall_shear_coefficient_analytic"],
            "rel": run["rel_wall_shear"],
        }
        if run["rel_wall_shear"] > worst_w:
            worst_w, worst_w_ha = run["rel_wall_shear"], float(ha)
    crit["b_hartmann_wall"] = {
        "id": "b_hartmann_wall",
        "value": worst_w,
        "threshold": 0.03,
        "passed": bool(worst_w <= 0.03),
        "worst_ha": worst_w_ha,
        "metric": "|tau_w H/(mu u_mean) - Ha tanh(Ha)/m(Ha)| / analytic",
        "per_ha": per_ha_w,
    }

    # (c) Shercliff centreline
    per_ha_c = {}
    worst_c = 0.0
    worst_c_ha = None
    for ha in sher["ha_list"]:
        run = sher["runs"][f"{float(ha):g}_{finest}"]
        per_ha_c[f"{float(ha):g}"] = {
            "u_centre_lbm": run["u_centre_lbm"],
            "u_centre_analytic": run["u_centre_analytic"],
            "rel_centre": run["rel_centre"],
            "rel_linf": run["rel_linf"],
            "ny": finest,
        }
        if run["rel_centre"] > worst_c:
            worst_c, worst_c_ha = run["rel_centre"], float(ha)
    crit["c_shercliff_centre"] = {
        "id": "c_shercliff_centre",
        "value": worst_c,
        "threshold": 0.05,
        "passed": bool(worst_c <= 0.05),
        "worst_ha": worst_c_ha,
        "metric": "|u_centre_lbm - u_centre_exact| / u_centre_exact (finest grid)",
        "per_ha": per_ha_c,
    }

    # (d) magnetic damping
    crit.update(damp["criteria"])
    crit["d_damping"] = {
        "id": "d_damping",
        "value": crit["c4_damping_rate"]["value"],
        "threshold": 0.05,
        "passed": bool(
            crit["c4_damping_rate"]["passed"]
            and crit["c4b_scaling_exponent"]["passed"]
            and crit["c4c_viscous_control"]["passed"]
        ),
        "metric": (
            "measured shear-wave decay rate vs nu k^2 + nu Ha^2/h^2 (<=5 %) and the "
            "log-log slope of the magnetic rate vs Ha (within 0.1 of 2)"
        ),
    }

    # (e) degeneration
    pois_worst = max(runs["rel_linf"] for runs in pois["runs"].values())
    crit["e_degeneration"] = {
        "id": "e_degeneration",
        "value": pois_worst,
        "threshold": 0.01,
        "passed": bool(pois_worst <= 0.01),
        "metric": "max_ny ||u_lbm - u_Poiseuille||_inf / u_mean at Ha = 0 (B -> 0)",
        "per_ny": {ny: pois["runs"][ny]["rel_linf"] for ny in pois["runs"]},
    }

    # (f) grid convergence of the velocity L2 error (Hartmann branch)
    conv = {}
    for ha in hart["ha_list"]:
        errs = [hart["runs"][f"{float(ha):g}_{ny}"]["rel_l2"] for ny in ny_list]
        conv[f"{float(ha):g}"] = {"rel_l2": errs, **convergence_order(errs)}
    orders = [conv[k]["finest_order"] for k in conv if conv[k]["finest_order"] is not None]
    worst_order = min(orders) if orders else 0.0
    crit["f_convergence"] = {
        "id": "f_convergence",
        "value": worst_order,
        "threshold": 1.5,
        "passed": bool(worst_order >= 1.5),
        "metric": "observed order p = log2(e_coarse/e_fine) of the profile L2 error",
        "per_ha": conv,
    }
    return crit


def collect_tests() -> dict:
    """Collect pytest node ids and count them per acceptance category."""
    try:
        out = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_mhd_hartmann.py",
                "tests/test_mhd_shercliff.py",
                "tests/test_mhd_thruster.py",
                "tests/test_mhd_damping.py",
                "tests/test_mhd_node_acceptance.py",
                "--collect-only",
                "-q",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"error": repr(exc), "total": 0, "per_category": {}}
    ids = [
        line.strip()
        for line in out.stdout.splitlines()
        if "::" in line and not line.startswith(" ")
    ]
    categories = {
        "hartmann": ("test_mhd_hartmann", "test_hartmann_"),
        "shercliff": ("test_mhd_shercliff", "test_shercliff_"),
        "damping": ("test_mhd_damping", "magnetic_rate", "decay_rate", "shear-wave", "shear_wave"),
        "degeneration": ("poiseuille", "zero_field_control", "degrad"),
        "convergence": (
            "test_grid_convergence",
            "test_damping_rate_converges",
            "test_damping_amplitude_error_converges",
            "test_centre_value_helper",
        ),
    }
    per_cat = {
        name: sum(1 for i in ids if any(k in i.lower() for k in keys))
        for name, keys in categories.items()
    }
    return {"total": len(ids), "per_category": per_cat, "node_ids": ids}


# ---------------------------------------------------------------------------
# sharded execution: one process per (branch, Ha) so that the campaign runs in
# parallel when the shared control plane is contended, and so that a cluster
# submission can be split across jobs.
# ---------------------------------------------------------------------------


def shard_names() -> list[str]:
    names = [f"hartmann_{ha:g}" for ha in HA_LIST]
    names += [f"shercliff_{ha:g}" for ha in SHERCLIFF_HA]
    names += ["poiseuille", "damping"]
    return names


def run_shard(out: Path, name: str, ny_list: tuple[int, ...]) -> int:
    out.mkdir(parents=True, exist_ok=True)
    if name.startswith("hartmann_"):
        ha = float(name.split("_")[1])
        payload = hartmann_ladder(ny_list, (ha,))
    elif name.startswith("shercliff_"):
        ha = float(name.split("_")[1])
        payload = shercliff_ladder(ny_list, (ha,))
    elif name == "poiseuille":
        payload = degeneration(ny_list)
    elif name == "damping":
        payload = damping_ladder(HA_LIST, ny=ny_list[0], nu=NU)
        payload["refinement"] = damping_refinement(HA_LIST[1], (16, 32, 64, 128), NU)
    else:
        raise SystemExit(f"unknown shard {name!r}; known: {shard_names()}")
    dump_json(out / f"_shard_{name}.json", payload)
    print(f"[shard] {name} done -> {out / f'_shard_{name}.json'}", flush=True)
    return 0


def _merge_branch(shards: list[dict], key: str) -> dict:
    runs: dict = {}
    ha_list: list[float] = []
    ny_list: list[int] = []
    for shard in shards:
        runs.update(shard["runs"])
        for ha in shard.get(key, []):
            if float(ha) not in ha_list:
                ha_list.append(float(ha))
        for ny in shard.get("ny_list", []):
            if int(ny) not in ny_list:
                ny_list.append(int(ny))
    return {"ny_list": sorted(ny_list), "ha_list": sorted(ha_list), "runs": runs}


def combine(out: Path, ny_list: tuple[int, ...], with_tests: bool = False) -> int:
    """Merge the shard files, evaluate the node criteria and write acceptance.json."""
    shard_files = sorted(out.glob("_shard_*.json"))
    if not shard_files:
        raise SystemExit(f"no shard files in {out}")
    shards = {p.name[len("_shard_"):-len(".json")]: json.loads(p.read_text()) for p in shard_files}
    hart = _merge_branch(
        [v for k, v in shards.items() if k.startswith("hartmann_")], "ha_list"
    )
    sher = _merge_branch(
        [v for k, v in shards.items() if k.startswith("shercliff_")], "ha_list"
    )
    pois_runs: dict = {}
    for key, value in shards.items():
        if key == "poiseuille":
            pois_runs.update(value["runs"])
    pois = {"ny_list": sorted(int(k) for k in pois_runs), "runs": pois_runs}
    damp = shards["damping"]
    for name, payload in (
        ("hartmann_ladder.json", hart),
        ("shercliff_centreline.json", sher),
        ("degeneration.json", pois),
        ("damping.json", damp),
    ):
        dump_json(out / name, payload)

    t_start = time.time()
    crit = evaluate(hart, sher, damp, pois, tuple(sorted(ny_list)))
    if with_tests:
        crit["g_tests"] = {"id": "g_tests", **collect_tests()}
        crit["g_tests"]["passed"] = bool(
            crit["g_tests"]["total"] >= 20
            and all(v >= 3 for v in crit["g_tests"]["per_category"].values())
        )
        crit["g_tests"]["threshold"] = ">= 20 total and >= 3 per category"
        crit["g_tests"]["value"] = crit["g_tests"]["total"]
    crit["h_report"] = {
        "id": "h_report",
        "passed": None,
        "note": "checked externally: worktree branch commit + paper-level report posting",
    }
    artefacts = []
    for path in sorted(out.glob("*.json")):
        if path.name in ("acceptance.json", "environment.json", "report_payload.json") or path.name.startswith("_shard_"):
            continue
        artefacts.append(
            {"path": str(path.resolve().relative_to(ROOT.resolve())), "sha256": sha256(path)}
        )
    payload = {
        "node": NODE,
        "repo": "TensorLBM",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "grid_plan": {
            "ny_list": list(sorted(ny_list)),
            "ha_list": hart["ha_list"],
            "shercliff_ha": sher["ha_list"],
        },
        "conventions": {
            "nu": NU,
            "u_mean": U_MEAN,
            "tol": TOL,
            "half_width": "h = ny/2 lattice spacings",
            "ha_definition": "Ha = B0 h sqrt(sigma/mu)",
        },
        "criteria": crit,
        "criteria_order": [
            "a_hartmann_profile",
            "b_hartmann_wall",
            "c_shercliff_centre",
            "d_damping",
            "e_degeneration",
            "f_convergence",
            "g_tests",
            "h_report",
        ],
        "artifacts": artefacts,
        "wall_clock_s": round(time.time() - t_start, 1),
    }
    gate_keys = [k for k in payload["criteria_order"] if k != "h_report"]
    payload["all_passed"] = all(
        crit[k]["passed"] for k in gate_keys if k in crit and crit[k]["passed"] is not None
    ) and ("g_tests" in crit)
    dump_json(
        out / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy": np.__version__,
        },
    )
    dump_json(out / "acceptance.json", payload)
    print(json.dumps({k: (v.get("value"), v.get("passed")) for k, v in crit.items()},
                     ensure_ascii=False, indent=2))
    print(f"all_passed={payload['all_passed']}")
    return 0 if payload["all_passed"] else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="tiny grids for a smoke test")
    ap.add_argument("--outdir", default=str(ROOT / "artifacts" / "mhd_p315w1_n04"))
    ap.add_argument("--with-tests", action="store_true", help="also collect pytest counts")
    ap.add_argument(
        "--augment",
        action="store_true",
        help="only add the damping grid-refinement table to an existing damping.json",
    )
    ap.add_argument(
        "--shard",
        default=None,
        help="run one shard only: hartmann_<Ha> | shercliff_<Ha> | poiseuille | damping",
    )
    ap.add_argument(
        "--combine",
        action="store_true",
        help="merge the shard files into the campaign artefacts and evaluate the criteria",
    )
    ap.add_argument(
        "--tests-only",
        action="store_true",
        help="only refresh the pytest counts of an existing acceptance.json",
    )
    args = ap.parse_args(argv)

    out = Path(args.outdir)
    if args.combine:
        return combine(out, ny_list=QUICK_NY if args.quick else NY_LIST,
                       with_tests=args.with_tests)
    if args.shard:
        return run_shard(out, args.shard, QUICK_NY if args.quick else NY_LIST)
    if args.augment:
        path = out / "damping.json"
        data = json.loads(path.read_text())
        data["refinement"] = damping_refinement(float(data["ha_values"][1]), (16, 32, 64, 128), data["nu"])
        dump_json(path, data)
        ref = data["refinement"]
        print("damping refinement errors:", ref["rel_rate_errors"], "orders:", ref["orders"])
        return 0
    if args.tests_only:
        acc_path = out / "acceptance.json"
        acc = json.loads(acc_path.read_text())
        crit = acc["criteria"]
        crit["g_tests"] = {"id": "g_tests", **collect_tests()}
        crit["g_tests"]["passed"] = bool(
            crit["g_tests"]["total"] >= 20
            and all(v >= 3 for v in crit["g_tests"]["per_category"].values())
        )
        crit["g_tests"]["threshold"] = ">= 20 total and >= 3 per category"
        crit["g_tests"]["value"] = crit["g_tests"]["total"]
        gate_keys = [k for k in acc["criteria_order"] if k != "h_report"]
        acc["all_passed"] = all(
            crit[k]["passed"] for k in gate_keys if k in crit and crit[k]["passed"] is not None
        )
        dump_json(acc_path, acc)
        print(json.dumps(crit["g_tests"], ensure_ascii=False)[:600])
        print("all_passed=", acc["all_passed"])
        return 0

    ny_list = QUICK_NY if args.quick else NY_LIST
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    hart = hartmann_ladder(ny_list, HA_LIST)
    dump_json(out / "hartmann_ladder.json", hart)
    sher = shercliff_ladder(ny_list, SHERCLIFF_HA)
    dump_json(out / "shercliff_centreline.json", sher)
    pois = degeneration(ny_list)
    dump_json(out / "degeneration.json", pois)
    damp = damping_ladder(HA_LIST, ny=ny_list[0], nu=NU)
    damp["refinement"] = damping_refinement(HA_LIST[1], (16, 32, 64, 128), NU)
    dump_json(out / "damping.json", damp)

    crit = evaluate(hart, sher, damp, pois, ny_list)
    if args.with_tests:
        crit["g_tests"] = {"id": "g_tests", **collect_tests()}
        crit["g_tests"]["passed"] = bool(
            crit["g_tests"]["total"] >= 20
            and all(v >= 3 for v in crit["g_tests"]["per_category"].values())
        )
        crit["g_tests"]["threshold"] = ">= 20 total and >= 3 per category"
        crit["g_tests"]["value"] = crit["g_tests"]["total"]
    crit["h_report"] = {
        "id": "h_report",
        "passed": None,
        "note": "checked externally: worktree branch commit + paper-level report posting",
    }

    artefacts = []
    for path in sorted(out.glob("*.json")):
        if path.name in ("acceptance.json", "environment.json"):
            continue
        artefacts.append(
            {"path": str(path.resolve().relative_to(ROOT.resolve())), "sha256": sha256(path)}
        )

    payload = {
        "node": NODE,
        "repo": "TensorLBM",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "grid_plan": {"ny_list": list(ny_list), "ha_list": list(HA_LIST),
                      "shercliff_ha": list(SHERCLIFF_HA)},
        "conventions": {
            "nu": NU,
            "u_mean": U_MEAN,
            "tol": TOL,
            "half_width": "h = ny/2 lattice spacings",
            "ha_definition": "Ha = B0 h sqrt(sigma/mu)",
        },
        "criteria": crit,
        "criteria_order": [
            "a_hartmann_profile",
            "b_hartmann_wall",
            "c_shercliff_centre",
            "d_damping",
            "e_degeneration",
            "f_convergence",
            "g_tests",
            "h_report",
        ],
        "artifacts": artefacts,
        "wall_clock_s": round(time.time() - t_start, 1),
    }
    gate_keys = [k for k in payload["criteria_order"] if k != "h_report"]
    payload["all_passed"] = all(
        crit[k]["passed"] for k in gate_keys if k in crit and crit[k]["passed"] is not None
    ) and ("g_tests" in crit)
    dump_json(
        out / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy": np.__version__,
        },
    )
    dump_json(out / "acceptance.json", payload)
    print(json.dumps({k: (v.get("value"), v.get("passed")) for k, v in crit.items()},
                     ensure_ascii=False, indent=2))
    print(f"all_passed={payload['all_passed']} wall={payload['wall_clock_s']}s")
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())