"""XVAL — Young-Laplace analytic vs droplet calibration on the SOY free-energy LBM.

Anchor: ``XVAL-LBM-YOUNG-LAPLACE-ANALYTIC-PLUS-DROPLET-CALI``.

Kernel under test
-----------------
``src/tensorlbm/multiphase_soy_dropin.py`` (D2Q9 Swift-Oono-Yeomans free-energy
binary solver, the sailfish ``fingering_periodic`` port) driven through
``src/tensorlbm/multiphase_static_droplet.py::soy_free_energy_droplet``.

Two halves of the anchor
------------------------
**ANALYTIC**: the closed-form tension of the square-gradient free energy,
``sigma = sqrt(8 kappa A^3 / (9 B^2)) == (4/3) kappa psi0^2 / xi`` (both forms
are returned by ``xval_lib.soy_constants`` and their identity is pinned), and the
2D Young-Laplace law ``dp = sigma / R``.

**DROPLET-CALI**: ``sigma_eff = dp * R_eq`` measured from droplets at three radii,
with ``R_eq`` measured from the phase-field area (never the initial radius) and
``dp`` measured from the solver's own pressure field ``p = rho/3 + psi mu``.
Calibration is then cross-validated: leave-one-radius-out prediction of ``dp``.

Arms
----
``radius``      R = 10 / 15 / 20 in boxes n = 64 / 96 / 128 (n >= 6R)
``sigma``       sigma_target = 0.005 / 0.010 / 0.020 at R = 15
``xi``          interface width xi = 2 / 3 / 4 at R = 15 (discretisation error)
``resolution``  n = 96 vs 128 at R = 15
``box``         n = 64 at R = 15 (n = 4.3R: periodic-image contamination control)

Every number in ``artifacts/summary.json`` is produced here from the populations;
nothing is hand-typed.  ``--quick`` halves the step budget for smoke runs and is
*flagged* in the artifacts so a quick run can never be quoted as a full one.

Usage::

    python xval/XVAL-LBM-YOUNG-LAPLACE-ANALYTIC-PLUS-DROPLET-CALI/run_xval.py --jobs 6
    python .../run_xval.py --quick --jobs 4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from multiprocessing import Pool

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _HERE)

import xval_lib as XL  # noqa: E402

OUT_DIR = os.path.join(_HERE, "artifacts")

# ---------------------------------------------------------------------------
# Protocol (frozen before any run - see predeclared.json)
# ---------------------------------------------------------------------------

TOL_PCT = 15.0          # owner accuracy standard
TIGHT_PCT = 5.0         # self-set diagnostic band (radius independence)
SELF_REST_TOL = 1.0e-3  # max|u| / cs for "at rest"
MASS_TOL = 1.0e-6       # |rho_mean - rho0| / rho0
WARMUP_STEPS = 3500
N_SAMPLES = 20
SAMPLE_EVERY = 25


@dataclass(frozen=True)
class Case:
    key: str
    n: int
    radius: float
    sigma: float
    xi: float
    arm: str

    @property
    def steps(self) -> int:
        return WARMUP_STEPS + N_SAMPLES * SAMPLE_EVERY


CASES: tuple[Case, ...] = (
    Case("radius_R10_n64", 64, 10.0, 0.010, 3.0, "radius"),
    Case("radius_R15_n96", 96, 15.0, 0.010, 3.0, "radius"),
    Case("radius_R20_n128", 128, 20.0, 0.010, 3.0, "radius"),
    Case("sigma_s005_n96", 96, 15.0, 0.005, 3.0, "sigma"),
    Case("sigma_s020_n96", 96, 15.0, 0.020, 3.0, "sigma"),
    Case("xi_2_n96", 96, 15.0, 0.010, 2.0, "xi"),
    Case("xi_4_n96", 96, 15.0, 0.010, 4.0, "xi"),
    Case("res_n128_R15", 128, 15.0, 0.010, 3.0, "resolution"),
    Case("box_n64_R15", 64, 15.0, 0.010, 3.0, "box"),
)


# Fallback case set, added because the control-plane shell cap (120 s) cannot run
# the n=96/128 boxes locally and the cluster queue (cj2037/cj2045) had not
# dispatched: same question, same thresholds, same 4000-step budget, small boxes.
# Recorded as a protocol deviation in predeclared.json before it was run.
CASES_SMALL_BOX: tuple[Case, ...] = (
    Case("radius_R8_n64", 64, 8.0, 0.010, 3.0, "radius"),
    Case("radius_R10_n64", 64, 10.0, 0.010, 3.0, "radius"),
    Case("radius_R13_n64", 64, 13.0, 0.010, 3.0, "radius"),
    Case("sigma_s005_n64", 64, 10.0, 0.005, 3.0, "sigma"),
    Case("sigma_s020_n64", 64, 10.0, 0.020, 3.0, "sigma"),
    Case("xi_2_n64", 64, 10.0, 0.010, 2.0, "xi"),
    Case("xi_4_n64", 64, 10.0, 0.010, 4.0, "xi"),
)


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------


def run_case(case: Case, *, quick: bool = False, threads: int = 1) -> dict:
    """Simulate one droplet and measure everything that does not need the answer."""
    import torch

    torch.set_num_threads(max(1, threads))
    from tensorlbm.multiphase_soy_dropin import SOYConfig, SOYFreeEnergyD2Q9

    const = XL.soy_constants(case.sigma, case.xi)
    cfg = SOYConfig(
        nx=case.n,
        ny=case.n,
        radius=case.radius,
        sigma=case.sigma,
        xi=case.xi,
        backend="torch",
        dtype="float64",
    )
    solver = SOYFreeEnergyD2Q9(cfg)
    solver.init_droplet()
    rho0 = cfg.rho0
    psi0_sum = float(solver.b.to_numpy(solver.psi).sum())

    warmup = WARMUP_STEPS // 2 if quick else WARMUP_STEPS
    n_samples = 5 if quick else N_SAMPLES
    sample_every = SAMPLE_EVERY
    t0 = time.time()
    history: list[list[float]] = []
    samples: list[dict[str, float]] = []
    for step in range(1, warmup + n_samples * sample_every + 1):
        solver.step()
        if step > warmup and (step - warmup) % sample_every == 0:
            samples.append(_snapshot(solver, case, const))
            history.append([float(step), samples[-1]["dp_eq"], samples[-1]["u_max"]])

    names = [k for k in samples[0] if k != "n_inside"] if samples else []
    agg = {k: float(np.mean([s[k] for s in samples])) for k in names}
    agg.update({f"{k}_std": float(np.std([s[k] for s in samples])) for k in names})
    agg["n_samples"] = len(samples)
    agg["n_inside"] = float(samples[-1]["n_inside"]) if samples else 0.0
    agg["n_outside"] = float(samples[-1]["n_outside"]) if samples else 0.0

    # the fields of the last sample are kept for the planted-defect analysis
    fields = _fields(solver, case, agg)
    result = {
        "case_id": case.key,
        "arm": case.arm,
        "params": {
            "n": case.n,
            "radius": case.radius,
            "sigma_target": case.sigma,
            "xi": case.xi,
            "steps": warmup + n_samples * sample_every,
            "warmup": warmup,
            "n_samples": n_samples,
            "sample_every": sample_every,
            "radius_over_n": case.radius / case.n,
            "backend": "torch",
            "dtype": "float64",
            "pressure_form": cfg.pressure_form,
            "tau_g": cfg.tau_g,
            "gamma": cfg.gamma,
        },
        "analytic": const,
        "measures": agg,
        "ref": {},
        "errors": {},
        "history": history,
        "wall_s": time.time() - t0,
        "_fields": fields,
    }
    dp_ref = XL.laplace_reference_2d(const["sigma_analytic"], agg["R_eq"])
    result["ref"] = {
        "dp_laplace": dp_ref,
        "dp_laplace_initial_radius": XL.laplace_reference_2d(const["sigma_analytic"], case.radius),
        "sigma_analytic": const["sigma_analytic"],
        "sigma_eff": agg["dp_eq"] * agg["R_eq"],
    }
    result["errors"] = {
        "dp_rel_pct": 100.0 * (agg["dp_eq"] - dp_ref) / dp_ref,
        "sigma_eff_vs_analytic_pct": 100.0
        * (result["ref"]["sigma_eff"] - const["sigma_analytic"])
        / const["sigma_analytic"],
        "sigma_cap_vs_analytic_pct": 100.0
        * (agg["sigma_cap"] - const["sigma_analytic"])
        / const["sigma_analytic"],
        "dp_fixed_vs_analytic_initial_pct": 100.0
        * (agg["dp_fixed"] - result["ref"]["dp_laplace_initial_radius"])
        / result["ref"]["dp_laplace_initial_radius"],
        "R_eq_vs_Rinit_pct": 100.0 * (agg["R_eq"] - case.radius) / case.radius,
        "xi_meas_vs_xi_pct": 100.0 * (agg["xi_meas"] - case.xi) / case.xi,
        "mass_rel_drift": abs(agg["rho_mean"] - rho0) / rho0,
        "psi_sum_rel_drift": abs(agg["psi_sum"] - psi0_sum) / abs(psi0_sum),
    }
    return result


def _snapshot(solver, case: Case, const: dict) -> dict[str, float]:
    """Measure one state: everything that does not need the answer."""
    b = solver.b
    psi = b.to_numpy(solver.psi)
    p = b.to_numpy(solver.p)
    ux = b.to_numpy(solver.ux)
    uy = b.to_numpy(solver.uy)
    gx = b.to_numpy(solver.grad_x)
    gy = b.to_numpy(solver.grad_y)
    rho = b.to_numpy(solver.rho)

    R_eq, centre = XL.equivalent_radius(psi)
    r = XL.radial_distance(psi.shape, centre)
    dp_eq, n_in, n_out = XL.pressure_jump(p, r, R_eq, case.xi)
    # library convention reproduction: masks on the *initial* radius / box centre
    r_box, _ = XL._grid(psi.shape)
    dp_fixed, _, _ = XL.pressure_jump(p, r_box, case.radius, case.xi)
    sigma_cap = XL.gradient_energy_sigma(gx, gy, const["kappa"], R_eq)
    rest = XL.spurious_velocity(ux, uy)
    width = XL.interface_width_fit(psi, R_eq, centre)
    return {
        "dp_eq": dp_eq,
        "dp_fixed": dp_fixed,
        "R_eq": R_eq,
        "sigma_cap": sigma_cap,
        "u_max": rest["u_max"],
        "u_rms": rest["u_rms"],
        "u_max_over_cs": rest["u_max_over_cs"],
        "rho_mean": float(rho.mean()),
        "psi_sum": float(psi.sum()),
        "psi_abs_sum": float(np.abs(psi).sum()),
        "xi_meas": width["xi_meas"],
        "R_fit": width["R_fit"],
        "centre_x": centre[0],
        "centre_y": centre[1],
        "n_inside": float(n_in),
        "n_outside": float(n_out),
    }


def _fields(solver, case: Case, agg: dict) -> dict:
    b = solver.b
    psi = b.to_numpy(solver.psi)
    p = b.to_numpy(solver.p)
    R_eq, centre = XL.equivalent_radius(psi)
    return {
        "p": p.astype(np.float32),
        "psi": psi.astype(np.float32),
        "r": XL.radial_distance(psi.shape, centre).astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Aggregation / gates
# ---------------------------------------------------------------------------


def _arms(cases: list[dict]) -> dict:
    by_arm: dict[str, list[dict]] = {}
    for c in cases:
        by_arm.setdefault(c["arm"], []).append(c)
    out: dict = {}
    for arm, rows in by_arm.items():
        rows = sorted(rows, key=lambda c: c["params"]["radius"] if arm == "radius" else c["params"]["sigma_target"])
        out[arm] = {
            "cases": [c["case_id"] for c in rows],
            "max_abs_dp_err_pct": max(abs(c["errors"]["dp_rel_pct"]) for c in rows),
            "max_abs_sigma_eff_err_pct": max(abs(c["errors"]["sigma_eff_vs_analytic_pct"]) for c in rows),
            "max_abs_sigma_cap_err_pct": max(abs(c["errors"]["sigma_cap_vs_analytic_pct"]) for c in rows),
            "rows": [
                {
                    "case_id": c["case_id"],
                    "x": c["params"]["radius"] if arm == "radius" else c["params"]["sigma_target"],
                    "R_eq": c["measures"]["R_eq"],
                    "dp_eq": c["measures"]["dp_eq"],
                    "sigma_eff": c["ref"]["sigma_eff"],
                    "sigma_analytic": c["analytic"]["sigma_analytic"],
                    "dp_rel_pct": c["errors"]["dp_rel_pct"],
                    "sigma_eff_vs_analytic_pct": c["errors"]["sigma_eff_vs_analytic_pct"],
                }
                for c in rows
            ],
        }
    return out


def _calibration(radius_rows: list[dict]) -> dict:
    radii = [c["measures"]["R_eq"] for c in radius_rows]
    dp = [c["measures"]["dp_eq"] for c in radius_rows]
    sig = XL.fit_sigma_through_origin(np.array(dp), 1.0 / np.array(radii))
    free = XL.free_line_fit(np.array(dp), 1.0 / np.array(radii))
    loo = XL.leave_one_radius_out(radii, dp)
    return {
        "radii": radii,
        "dp": dp,
        "sigma_fit_through_origin": sig,
        "free_line": free,
        "leave_one_out": loo,
        "max_abs_loo_err_pct": max((abs(r["err_pct"]) for r in loo), default=float("nan")),
        "radius_independence_max_pct": max(
            abs(100.0 * (dp[i] * radii[i] - sig) / sig) for i in range(len(radii))
        ),
    }


def _gates(cases: list[dict], arms: dict, calib: dict, analytic: dict, defects: dict) -> list[dict]:
    valid = [c for c in cases if c["measures"]["dp_eq"] == c["measures"]["dp_eq"]]
    gates = []

    def add(name: str, thr: float, val: float, kind: str = "<=") -> None:
        ok = val <= thr if kind == "<=" else val >= thr
        gates.append({"name": name, "threshold": thr, "measured": val, "kind": kind, "pass": bool(ok)})

    add(
        "G1 droplet-calibration |sigma_eff - sigma_analytic|/sigma_analytic [%] (radius arm)",
        TOL_PCT,
        arms["radius"]["max_abs_sigma_eff_err_pct"],
    )
    add(
        "G2 per-point Young-Laplace |dp - sigma_analytic/R_eq|/(sigma_analytic/R_eq) [%] (all cases)",
        TOL_PCT,
        max(abs(c["errors"]["dp_rel_pct"]) for c in valid),
    )
    if calib:
        add(
            "G3 radius independence: max |dp_i R_i - sigma_fit|/sigma_fit [%]",
            TOL_PCT,
            calib["radius_independence_max_pct"],
        )
    else:
        gates.append({"name": "G3 radius independence", "threshold": TOL_PCT,
                      "measured": float("nan"), "kind": "<=", "pass": False, "status": "not_run"})
    # G4 sigma-linearity: dp must scale like sigma_target (slope 1 on log-log)
    sig_rows = [c for c in cases if c["arm"] == "sigma"
                or c["case_id"] in ("radius_R15_n96", "radius_R10_n64")]
    if len(sig_rows) >= 3:
        xs = np.log([c["params"]["sigma_target"] for c in sig_rows])
        ys = np.log([c["measures"]["dp_eq"] for c in sig_rows])
        slope = float(np.polyfit(xs, ys, 1)[0])
        add("G4 sigma-linearity exponent |slope - 1| (dp vs sigma_target)", 0.15, abs(slope - 1.0))
    else:
        gates.append({"name": "G4 sigma-linearity", "threshold": 0.15, "measured": float("nan"),
                      "kind": "<=", "pass": False, "status": "not_run"})
    if calib:
        add("G5 leave-one-radius-out prediction max |err| [%]", TOL_PCT, abs(calib["max_abs_loo_err_pct"]))
    else:
        gates.append({"name": "G5 leave-one-radius-out", "threshold": TOL_PCT,
                      "measured": float("nan"), "kind": "<=", "pass": False, "status": "not_run"})
    add(
        "G6a analytic identity |sigma_coef - sigma_grad|/sigma [1e-12]",
        1e-12,
        analytic["sigma_identity_residual"],
    )
    add(
        "G6b static rest: max |u|/c_s over all cases [1e-3]",
        SELF_REST_TOL,
        max(c["measures"]["u_max_over_cs"] for c in valid),
    )
    add(
        "G6c mass conservation |rho_mean - rho0|/rho0 [1e-6]",
        MASS_TOL,
        max(abs(c["errors"]["mass_rel_drift"]) for c in valid),
    )
    # G8 discretisation trend: the xi = 4 error may not be worse than the xi = 2 error
    xi_rows = {c["params"]["xi"]: c for c in cases if c["arm"] == "xi"}
    ref_row = next((c for c in cases if c["case_id"] in ("radius_R15_n96", "radius_R10_n64")), None)
    if ref_row is not None:
        xi_rows[3.0] = ref_row
    if 2.0 in xi_rows and 4.0 in xi_rows:
        e2 = abs(xi_rows[2.0]["errors"]["sigma_eff_vs_analytic_pct"])
        e4 = abs(xi_rows[4.0]["errors"]["sigma_eff_vs_analytic_pct"])
        gates.append(
            {
                "name": "G8 discretisation trend: |err(xi=4)| <= |err(xi=2)| [%]",
                "threshold": e2,
                "measured": e4,
                "kind": "trend",
                "pass": bool(e4 <= e2 + 1e-12),
            }
        )
    # G7 falsifiability is reported separately (honest, not part of gate_pass)
    caught = sum(1 for v in defects.values() if abs(v["max_abs_err_pct"]) > TOL_PCT)
    gates.append(
        {
            "name": "G7 falsifiability: planted defects caught by the 15% band (of 5)",
            "threshold": 3.0,
            "measured": float(caught),
            "kind": ">=",
            "pass": bool(caught >= 3),
        }
    )
    return gates


def _env(git_sha: str) -> dict:
    import torch

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "git_sha": git_sha,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _run_star(payload) -> dict:
    case, opts = payload
    return safe_run_case(case, **opts)


def safe_run_case(case: "Case", **opts) -> dict:
    """Run one case, converting a diverged run into an honest NaN case.

    A case that blows up must not take the campaign down and must not be silently
    dropped: it is reported with ``status='failed'`` so the arms that depend on it
    show up as not_run instead of shrinking the sample.
    """
    try:
        res = run_case(case, **opts)
    except Exception as exc:  # noqa: BLE001 - the failure itself is the datum
        nan = float("nan")
        res = {
            "case_id": case.key,
            "arm": case.arm,
            "params": {"n": case.n, "radius": case.radius, "sigma_target": case.sigma, "xi": case.xi},
            "analytic": XL.soy_constants(case.sigma, case.xi),
            "measures": {},
            "ref": {},
            "errors": {},
            "history": [],
            "wall_s": nan,
            "status": "failed",
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "_fields": {"p": np.zeros((1, 1), dtype=np.float32),
                        "psi": np.zeros((1, 1), dtype=np.float32),
                        "r": np.zeros((1, 1), dtype=np.float32)},
        }
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--only", default="", help="comma separated case ids")
    ap.add_argument("--caseset", default="declared", choices=["declared", "small_box"])
    args = ap.parse_args()

    pool_cases = CASES if args.caseset == "declared" else CASES_SMALL_BOX
    selected = [c for c in pool_cases if not args.only or c.key in args.only.split(",")]
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    if args.jobs > 1:
        with Pool(min(args.jobs, len(selected))) as pool:
            cases = pool.map(_run_star, [(c, {"quick": args.quick, "threads": args.threads}) for c in selected])
    else:
        cases = [safe_run_case(c, quick=args.quick, threads=args.threads) for c in selected]
    wall = time.time() - t0

    analytic = XL.soy_constants(0.010, 3.0)
    ok_cases = [c for c in cases if c.get("status") != "failed"]
    if len(ok_cases) < len(cases):
        print(f"[xval] {len(cases) - len(ok_cases)} case(s) failed and are reported as status=failed")
    arms = _arms(ok_cases)
    radius_rows = sorted([c for c in ok_cases if c["arm"] == "radius"], key=lambda c: c["measures"]["R_eq"])
    calib = _calibration(radius_rows) if len(radius_rows) >= 3 else {}
    defects = XL.planted_defects(ok_cases)
    gates = _gates(ok_cases, arms, calib, analytic, defects)
    for c in cases:
        if c.get("status") == "failed":
            gates.append({"name": f"{c['case_id']}: case failed", "threshold": 0.0, "measured": float("nan"),
                          "kind": "<=", "pass": False, "status": "failed",
                          "reason": c.get("failure_reason")})

    git_sha = os.environ.get("GIT_SHA", "")
    if not git_sha:
        try:
            import subprocess

            git_sha = (
                subprocess.check_output(["git", "-C", _ROOT, "rev-parse", "--short", "HEAD"], text=True)
                .strip()
            )
        except Exception:
            git_sha = "unknown"

    summary = {
        "anchor": "XVAL-LBM-YOUNG-LAPLACE-ANALYTIC-PLUS-DROPLET-CALI",
        "repo": "TensorLBM",
        "model": "D2Q9 Swift-Oono-Yeomans free-energy binary LBM (SOY), matched density, Guo forcing",
        "kernel": {
            "solver": "src/tensorlbm/multiphase_soy_dropin.py::SOYFreeEnergyD2Q9",
            "driver": "src/tensorlbm/multiphase_static_droplet.py::soy_free_energy_droplet",
            "pressure_field": "p = rho/3 + psi*mu (pressure_form='oss')",
            "note": "the runner here drives SOYFreeEnergyD2Q9 directly so the fields can be measured",
        },
        "reference": {
            "laplace_2d": "dp = sigma/R (cylinder; sphere would be 2 sigma/R)",
            "sigma_coef": "sigma = sqrt(8 kappa A^3 / (9 B^2))",
            "sigma_grad": "sigma = (4/3) kappa psi0^2 / xi",
            "psi0": "sqrt(A/B) = 1",
            "xi": "sqrt(2 kappa/A)",
            "sigma_identity_residual": analytic["sigma_identity_residual"],
            "sigma_analytic_at_reference": analytic["sigma_analytic"],
        },
        "protocol": {
            "tol_pct": TOL_PCT,
            "tight_diagnostic_pct": TIGHT_PCT,
            "rest_tol_u_over_cs": SELF_REST_TOL,
            "mass_tol": MASS_TOL,
            "warmup_steps": WARMUP_STEPS // 2 if args.quick else WARMUP_STEPS,
            "n_samples": 5 if args.quick else N_SAMPLES,
            "sample_every": SAMPLE_EVERY,
            "quick": bool(args.quick),
            "caseset": args.caseset,
        },
        "analytic_block": analytic,
        "cases": XL.strip_fields(cases),
        "arms": arms,
        "calibration": calib,
        "falsifiability": defects,
        "gates": gates,
        "gate_pass": bool(all(g["pass"] for g in gates if not g["name"].startswith("G7"))),
        "falsifiability_pass": bool(next(g["pass"] for g in gates if g["name"].startswith("G7"))),
        "wall_s": wall,
        "env": _env(git_sha),
    }

    path = os.path.join(args.out, "summary.json")
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=1, default=str)

    # compact field bundle for the hero radius case (reproducible post-processing)
    hero = next((c for c in cases if c["case_id"] in ("radius_R15_n96", "radius_R10_n64")), cases[0])
    if hero.get("status") == "failed":
        hero = next((c for c in cases if c.get("status") != "failed"), cases[0])
    np.savez_compressed(
        os.path.join(args.out, "hero_fields.npz"),
        psi=hero["_fields"]["psi"],
        p=hero["_fields"]["p"],
        r=hero["_fields"]["r"],
        meta=np.array([hero["params"]["n"], hero["measures"]["R_eq"], hero["params"]["xi"]]),
    )

    print(f"[xval] wrote {path}  wall={wall:.1f}s  gate_pass={summary['gate_pass']}")
    for g in gates:
        print(f"  {'PASS' if g['pass'] else 'FAIL'}  {g['name']}: {g['measured']} (thr {g['threshold']})")
    for c in cases:
        if c.get("status") == "failed":
            print(f"  {c['case_id']:<16} FAILED {c.get('failure_reason')}")
            continue
        print(
            f"  {c['case_id']:<16} R_eq={c['measures']['R_eq']:6.2f} dp={c['measures']['dp_eq']:.5e} "
            f"dp_err={c['errors']['dp_rel_pct']:+6.2f}% sig_eff={c['ref']['sigma_eff']:.5f} "
            f"sig_cap_err={c['errors']['sigma_cap_vs_analytic_pct']:+6.2f}% umax/cs={c['measures']['u_max_over_cs']:.1e} "
            f"wall={c['wall_s']:.0f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())