"""Pancake-ice resistance campaign driver: LBM drag table, parameter sweep, artifact.

Three stages, each independently runnable so the LBM work can be sharded:

::

    python3 tools/run_pancake_resistance.py --lbm          # D2Q9 bounce-back drag table
    python3 tools/run_pancake_resistance.py --sweep        # C x D x h x V sweep + fits
    python3 tools/run_pancake_resistance.py --assemble     # artifact + figures + tables
    python3 tools/run_pancake_resistance.py --all

Stage 1 (**LBM**) computes the drag coefficient of a pancake *section* — a thick
flat plate of aspect ratio ``D/h`` normal to the flow, plus the rim ridge — with
the package's D2Q9 bounce-back kernel and momentum-exchange force monitor.  The
same monitor is first validated on the circular cylinder
(``C_d(Re=100) = 1.34``, Braza et al. 1986).  The plateau values feed
``cd_pancake`` / ``cd_raft`` of :mod:`tensorlbm.pancake_ice`.

Stage 2 (**sweep**) evaluates the analytic resistance model on the campaign
design ``C in [0.1, 0.8]`` x ``D in [0.3, 3] m`` x ``h in [0.05, 0.3] m`` x
``V in [1, 5] m/s`` and fits ``R = a C^alpha V^beta``, plus a +/-50% sensitivity
study on the declared parameters.

Stage 3 (**assemble**) merges everything into
``artifacts/pancake_ice/pancake_resistance.json``, the figures under
``figures/pancake_ice/`` and the auto-generated tables
``docs/pancake_ice_results.md``.  Every acceptance-criterion number in the
artifact is computed here — nothing is hand-transcribed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorlbm import pancake_ice as pi  # noqa: E402

RUNS = ROOT / "runs" / "pancake_ice"
ARTIFACTS = ROOT / "artifacts" / "pancake_ice"
FIGURES = ROOT / "figures" / "pancake_ice"
DOCS = ROOT / "docs"

# campaign design (declared before any result was inspected)
C_GRID = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
D_GRID = np.array([0.3, 0.5, 1.0, 2.0, 3.0])
#: h/D sampled inside the measured aspect-ratio window D/h in [5, 20]
ASPECT_GRID = np.array([5.0, 8.0, 10.0, 15.0, 20.0])
V_GRID = np.array([1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
WAVE = pi.WaveField(wavelength=60.0, amplitude=3.0)  # H/lambda = 0.1 -> A_orb = lambda/20

#: acceptance thresholds, declared a priori (campaign metadata)
CRIT = {
    "retention_probability_above": 0.5,
    "retention_probability_below": 0.2,
    "threshold_fraction_of_wavelength": 1.0 / 20.0,
    "aspect_ratio_band": (5.0, 20.0),
    "alpha_band": (1.0, 2.0),
    "beta_band": (1.5, 2.5),
    "slip_share_low_C": 0.60,
    "bulldozing_share_high_C": 0.50,
    "low_C": 0.1,
    "high_C": 0.8,
    "cd_literature_band": pi.CD_BLUNT_PLATE_BAND,
    "cylinder_cd_re100": 1.34,
    "cylinder_cd_tolerance": 0.15,
}

LBM_CASES = [
    # cylinder: validation of the drag monitor against the literature
    {"id": "cyl_re40", "kind": "cylinder", "re": 40.0, "steps": 9000, "nx": 180, "ny": 140, "size": 36},
    {"id": "cyl_re100", "kind": "cylinder", "re": 100.0, "steps": 11000, "nx": 180, "ny": 140, "size": 36},
    # pancake sections: aspect ratio ladder at Re=400 ...
    {"id": "plate_ar5_re400", "kind": "plate", "ar": 5.0, "re": 400.0, "steps": 10000, "nx": 180, "ny": 140, "size": 36},
    {"id": "plate_ar10_re400", "kind": "plate", "ar": 10.0, "re": 400.0, "steps": 10000, "nx": 180, "ny": 140, "size": 36},
    {"id": "plate_ar20_re400", "kind": "plate", "ar": 20.0, "re": 400.0, "steps": 10000, "nx": 180, "ny": 140, "size": 36},
    # ... plus one low-Reynolds case to expose the Re trend at fixed aspect ratio
    {"id": "plate_ar10_re50", "kind": "plate", "ar": 10.0, "re": 50.0, "steps": 10000, "nx": 180, "ny": 140, "size": 36},
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _json_default(obj: object) -> object:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not serialisable: {type(obj)}")


def _dump(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # pragma: no cover - best effort provenance
        return "unknown"


def _provenance() -> dict[str, object]:
    import tensorlbm

    prov: dict[str, object] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "tensorlbm_file": str(Path(tensorlbm.__file__).resolve()),
        "module_sha256": hashlib.sha256(
            (ROOT / "src" / "tensorlbm" / "pancake_ice.py").read_bytes()
        ).hexdigest(),
        "driver_sha256": _sha256(Path(__file__).resolve()),
    }
    try:
        import torch

        prov["torch"] = torch.__version__
        prov["torch_threads"] = torch.get_num_threads()
    except Exception:  # pragma: no cover
        prov["torch"] = "unavailable"
    return prov


# ---------------------------------------------------------------------------
# stage 1: LBM drag table
# ---------------------------------------------------------------------------
def _run_lbm_case(case: dict[str, object]) -> dict[str, object]:
    """Run one 2-D D2Q9 case and return the drag-coefficient statistics."""
    import torch

    torch.set_num_threads(4)  # small 2-D grids: more threads only thrash

    from tensorlbm.boundaries import (
        apply_simple_channel_boundaries,
        bounce_back_cells,
        compute_obstacle_forces,
        cylinder_mask,
        make_channel_wall_mask,
    )
    from tensorlbm.d2q9 import equilibrium, macroscopic
    from tensorlbm.pancake_ice import pancake_section_mask
    from tensorlbm.solver import collide_bgk, stream

    nx = int(case["nx"])
    ny = int(case["ny"])
    size = int(case["size"])  # diameter / transverse extent in lattice cells
    re = float(case["re"])
    steps = int(case["steps"])
    u_in = 0.06
    nu = u_in * size / re
    tau = 3.0 * nu + 0.5
    if tau <= 0.5:
        raise SystemExit(f"invalid tau for case {case['id']}")

    dev = torch.device("cpu")
    if case["kind"] == "cylinder":
        mask_np = cylinder_mask(nx, ny, nx // 3, ny // 2, 0.5 * size, device="cpu").numpy()
        ref_length = float(size)
        thickness_cells = size
    else:
        ar = float(case["ar"])
        thickness_cells = max(int(round(size / ar)), 2)
        mask_np = pancake_section_mask(
            nx,
            ny,
            diameter_cells=size,
            thickness_cells=thickness_cells,
            rim_height_cells=max(int(round(0.2 * thickness_cells)), 1),
            rim_width_cells=max(thickness_cells // 3, 1),
        )
        ref_length = float(size) + 2.0 * max(int(round(0.2 * thickness_cells)), 1)
    mask = torch.as_tensor(mask_np, device=dev)
    wall_mask = make_channel_wall_mask(ny, nx, mask, device=dev)

    rho0 = torch.ones((ny, nx), device=dev)
    ux0 = torch.full_like(rho0, u_in)
    f = equilibrium(rho0, ux0, torch.zeros_like(rho0), device=dev)
    zeros = torch.zeros_like(mask)

    t0 = time.perf_counter()
    cd_hist: list[float] = []
    for step in range(1, steps + 1):
        f = collide_bgk(f, tau=tau)
        f = stream(f)
        fx, _fy = compute_obstacle_forces(f, mask)
        f = apply_simple_channel_boundaries(f, u_in=u_in, wall_mask=wall_mask, obstacle_mask=zeros)
        f = bounce_back_cells(f, mask)
        if step > 0.5 * steps:
            cd_hist.append(float(fx.item()) / (0.5 * u_in**2 * ref_length))
    runtime = time.perf_counter() - t0

    cd = np.asarray(cd_hist, dtype=float)
    n_half = cd.size // 2
    rho, ux, uy = macroscopic(f)
    payload: dict[str, object] = {
        "case_id": case["id"],
        "kind": case["kind"],
        "ar": float(case["ar"]) if "ar" in case else None,
        "re": re,
        "re_definition": "u_in * D / nu",
        "tau": tau,
        "nu": nu,
        "u_in": u_in,
        "nx": nx,
        "ny": ny,
        "blockage": float(ref_length / ny),
        "thickness_cells": thickness_cells,
        "steps": steps,
        "steps_averaged": int(cd.size),
        "runtime_s": runtime,
        "cd_mean": float(cd.mean()),
        "cd_corrected": float(cd.mean()) * (1.0 - ref_length / ny) ** 2,
        "blockage_correction": "(1-B)^2, B = reference length / ny (continuity blockage)",
        "cd_stderr": float(cd.std(ddof=1) / math.sqrt(cd.size)) if cd.size > 1 else 0.0,
        "cd_first_half_mean": float(cd[:n_half].mean()),
        "cd_second_half_mean": float(cd[n_half:].mean()),
        "u_max": float(ux.abs().max().item()),
    }
    if case["id"] == "plate_ar10_re400":
        snap = RUNS / "lbm" / "field_plate_ar10_re400.npz"
        snap.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            snap,
            ux=ux.numpy(),
            uy=uy.numpy(),
            rho=rho.numpy(),
            mask=mask_np.astype(np.uint8),
        )
        payload["field_snapshot"] = str(snap.relative_to(ROOT))
    return payload


def stage_lbm(only: str | None = None) -> Path:
    RUNS.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    for case in LBM_CASES:
        if only and case["id"] != only:
            continue
        print(f"[lbm] {case['id']} ...", flush=True)
        payload = _run_lbm_case(case)
        path = RUNS / "lbm" / f"{case['id']}.json"
        _dump(path, payload)
        print(
            f"[lbm] {case['id']}: Cd={payload['cd_mean']:.3f} "
            f"(+/-{payload['cd_stderr']:.3f}) tau={payload['tau']:.4f} "
            f"{payload['runtime_s']:.1f}s",
            flush=True,
        )
        results.append(payload)
    # always rebuild from disk so a --only run does not truncate the table
    table = build_cd_table(None)
    _dump(RUNS / "lbm" / "table.json", table)
    return RUNS / "lbm" / "table.json"


def _corrected(case: dict[str, object]) -> float:
    """Blockage-corrected C_d (fallback for case files written before the patch)."""
    if "cd_corrected" in case:
        return float(case["cd_corrected"])
    return float(case["cd_mean"]) * (1.0 - float(case["blockage"])) ** 2


def build_cd_table(results: list[dict[str, object]] | None = None) -> dict[str, object]:
    """Assemble the drag table (aspect ratio x Reynolds) from the per-case JSONs."""
    cases: list[dict[str, object]] = []
    if results is not None:
        cases = results
    else:
        for path in sorted((RUNS / "lbm").glob("*.json")):
            if path.name == "table.json":
                continue
            cases.append(json.loads(path.read_text()))
    if not cases:
        raise SystemExit("no LBM case results found; run --lbm first")
    cyl = {c["case_id"]: c for c in cases if c["kind"] == "cylinder"}
    plates = [c for c in cases if c["kind"] == "plate"]
    ars = sorted({float(c["ar"]) for c in plates})
    res = sorted({float(c["re"]) for c in plates})
    cd_by_ar_re: dict[str, dict[str, float]] = {}
    for ar in ars:
        row: dict[str, float] = {}
        for re in res:
            sel = [c for c in plates if float(c["ar"]) == ar and float(c["re"]) == re]
            if sel:
                row[f"{re:g}"] = _corrected(sel[0])
        cd_by_ar_re[f"{ar:g}"] = row
    plateau_candidates = [_corrected(c) for c in plates if float(c["re"]) == max(res)]
    plateau = float(np.mean(plateau_candidates)) if plateau_candidates else 2.0
    cyl_cd = {k: _corrected(v) for k, v in cyl.items()}
    return {
        "aspect_ratios": ars,
        "reynolds": res,
        "cd_by_ar_re": cd_by_ar_re,
        "cd_plateau": plateau,
        "cd_plateau_re": max(res) if res else None,
        "cylinder_cd": cyl_cd,
        "cases": [{k: v for k, v in c.items() if k != "field_snapshot"} for c in cases],
        "provenance": _provenance(),
    }


def load_cd_table() -> dict[str, object]:
    path = RUNS / "lbm" / "table.json"
    if not path.exists():
        raise SystemExit("missing runs/pancake_ice/lbm/table.json; run --lbm first")
    return json.loads(path.read_text())


def cd_at(table: dict[str, object], aspect_ratio: float) -> float:
    """Piecewise-linear C_d(D/h) from the LBM table, clamped to the endpoints."""
    ars = np.asarray(table["aspect_ratios"], dtype=float)
    keys = [f"{re:g}" for re in table["reynolds"]]
    top = keys[-1]
    cds = np.asarray([table["cd_by_ar_re"][f"{ar:g}"][top] for ar in ars], dtype=float)
    return float(np.interp(aspect_ratio, ars, cds))


# ---------------------------------------------------------------------------
# stage 2: analytic sweep, fits and sensitivity
# ---------------------------------------------------------------------------
def feasible_thickness(diameter: float, aspect_ratio: float) -> float:
    """Thickness for a requested D/h, clipped into the measured h window [0.05, 0.3] m.

    The two measured windows (``D/h in [5,20]`` and ``h in [0.05, 0.3] m``) are
    not independent: a 0.3 m pancake can only have ``D/h <= 6``.  Clipping h and
    recomputing ``D/h = D/h_clipped`` keeps every design point admissible.
    """
    h_lo, h_hi = pi.H_PANCAKE_BAND
    return float(min(max(diameter / aspect_ratio, h_lo), h_hi))


def build_design() -> dict[str, np.ndarray]:
    """Campaign design: C x D x (h/D) x V, all points inside both measured windows."""
    c, d, ar, v = np.meshgrid(C_GRID, D_GRID, ASPECT_GRID, V_GRID, indexing="ij")
    d_flat = d.ravel()
    h = np.array([feasible_thickness(float(dd), float(aa)) for dd, aa in zip(d_flat, ar.ravel())])
    return {
        "C": c.ravel(),
        "D": d_flat,
        "h": h,
        "V": v.ravel(),
    }


def stage_sweep() -> Path:
    table = load_cd_table()
    cd_pan = cd_at(table, 10.0)
    cd_raft = float(table["cd_plateau"])
    params = dict(pi.DEFAULT_PARAMS)
    params["cd_pancake"] = cd_pan
    params["cd_raft"] = cd_raft

    design = build_design()
    out = pi.sweep_resistance(
        design["C"], design["D"], design["h"], design["V"], params=params, pointwise=True
    )

    fits: dict[str, object] = {"global": pi.power_law_fit(out["C"], out["V"], out["resistance_N"])}
    for d_val in D_GRID:
        sel = np.isclose(out["D"], d_val)
        fits[f"D={d_val:g}m"] = pi.power_law_fit(
            out["C"][sel], out["V"][sel], out["resistance_N"][sel]
        )

    # mechanism shares on the campaign grid (D = 1 m, h/D = 10 representative)
    shares: list[dict[str, float]] = []
    for c_val in C_GRID:
        row = pi.pancake_resistance(c_val, 1.0, 0.1, 3.0, params=params)
        shares.append(
            {
                "C": float(c_val),
                "slip_share": row["slip_share"],
                "bulldozing_share": row["bulldozing_share"],
                "resistance_N": row["resistance_N"],
                "slip_N": row["slip_N"],
                "bulldozing_N": row["bulldozing_N"],
                "thickness_effective_m": row["thickness_effective_m"],
            }
        )

    # a-priori parameter sensitivity (+/-50% on the declared coefficients)
    sensitivity: list[dict[str, object]] = []
    base = dict(params)
    sweep_spec: list[tuple[str, float]] = [
        (name, factor)
        for name in ("k_slip", "c_jam", "k_raft", "p_jam", "cd_pancake", "cd_raft")
        for factor in (0.5, 1.0, 1.5)
    ]
    sweep_spec += [("k_cluster", value) for value in (0.0, 0.5, 1.0)]
    for name, factor in sweep_spec:
        if True:
            trial = dict(base)
            trial[name] = base[name] * factor if name != "k_cluster" else factor
            if name == "c_jam" and trial[name] > 1.0:
                continue
            res = pi.sweep_resistance(
                design["C"], design["D"], design["h"], design["V"], params=trial, pointwise=True
            )
            fit = pi.power_law_fit(res["C"], res["V"], res["resistance_N"])
            low_c = pi.pancake_resistance(CRIT["low_C"], 1.0, 0.1, 3.0, params=trial)
            high_c = pi.pancake_resistance(CRIT["high_C"], 1.0, 0.1, 3.0, params=trial)
            sensitivity.append(
                {
                    "parameter": name,
                    "factor": factor,
                    "value": trial[name],
                    "alpha": fit["alpha"],
                    "beta": fit["beta"],
                    "r2": fit["r2"],
                    "slip_share_low_C": low_c["slip_share"],
                    "bulldozing_share_high_C": high_c["bulldozing_share"],
                    "resistance_ref_N": base_ref(pi, base),
                }
            )

    payload = {
        "design": {
            "C": C_GRID.tolist(),
            "D": D_GRID.tolist(),
            "aspect_ratio": ASPECT_GRID.tolist(),
            "V": V_GRID.tolist(),
            "n_points": int(out["C"].size),
        },
        "params": params,
        "wave": WAVE.to_dict(),
        "grid": {k: v for k, v in out.items() if k != "shape"},
        "fits": fits,
        "shares": shares,
        "sensitivity": sensitivity,
        "cd_inputs": {"cd_pancake": cd_pan, "cd_raft": cd_raft, "table_re": table["cd_plateau_re"]},
        "provenance": _provenance(),
    }
    path = RUNS / "sweep.json"
    _dump(path, payload)
    print(f"[sweep] {out['C'].size} points -> {path}")
    print(f"[sweep] global alpha={fits['global']['alpha']:.3f} beta={fits['global']['beta']:.3f} "
          f"r2={fits['global']['r2']:.4f}")
    return path


def base_ref(module: object, params: dict[str, float]) -> float:
    """Reference resistance used to report sensitivity in absolute terms [N]."""
    return float(module.pancake_resistance(0.5, 1.0, 0.1, 3.0, params=params)["resistance_N"])# ---------------------------------------------------------------------------
# stage 3: artifact, figures, tables
# ---------------------------------------------------------------------------
def _retention_table() -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for frac in (1 / 8.0, 1 / 12.0, 1 / 15.0, 1 / 18.0, 1 / 20.0, 1 / 22.0, 1 / 30.0, 1 / 50.0, 1 / 100.0):
        d = frac * WAVE.wavelength
        prob = pi.pancake_formation_probability(d, WAVE)
        rows.append(
            {
                "diameter_over_wavelength": frac,
                "diameter_m": d,
                "probability": prob["probability"],
                "retention": prob["retention"],
                "buoyancy": prob["buoyancy"],
                "breakup": prob["breakup"],
                "bending_stress_Pa": prob["bending_stress_Pa"],
                "thickness_m": prob["thickness_m"],
            }
        )
    return rows


def _geometry_table() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for d in D_GRID:
        for ar in ASPECT_GRID:
            h = feasible_thickness(float(d), float(ar))
            geo = pi.PancakeGeometry(diameter=float(d), thickness=h, rim_height=0.03)
            row = geo.to_dict()
            row["issues"] = geo.validate(strict=False)["issues"]
            row["issues_count"] = len(row["issues"])
            rows.append(row)
    return rows


def _formation_table() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for wavelength in (10.0, 20.0, 30.0):
        wave = pi.WaveField(wavelength=wavelength, amplitude=0.05 * wavelength)  # H/lambda = 0.1
        traj = pi.simulate_aggregate_growth(wave=wave, thickness=0.1, t_max_hours=6.0, dt=1.0)
        rows.append(
            {
                "wavelength_m": wavelength,
                "amplitude_m": wave.amplitude,
                "threshold_m": pi.orbital_retention_threshold(wave),
                "d0_m": traj["d0_m"],
                "d_saturation_m": traj["d_saturation_m"],
                "t_consolidation_h": traj["t_consolidation_h"],
                "t_thermo_h": traj["t_thermo_h"],
                "t_form_h": traj["t_form_h"],
                "t_hours": traj["t_hours"],
                "diameter_m": traj["diameter_m"],
            }
        )
    return rows


def _cd_literature(ar: float) -> float:
    """Reference band centre for a blunt plate section, used for comparison only."""
    return 2.0


def evaluate_acceptance(
    sweep: dict[str, object], table: dict[str, object], retention: list[dict[str, float]],
    shares: list[dict[str, float]], pytest_info: dict[str, object],
) -> dict[str, object]:
    """Machine evaluation of the campaign node acceptance criteria."""
    fit = sweep["fits"]["global"]
    geometry = _geometry_table()
    ar_ok = all(r["issues_count"] == 0 for r in geometry)
    ar_min = min(r["aspect_ratio"] for r in geometry)
    ar_max = max(r["aspect_ratio"] for r in geometry)

    above = [r for r in retention if r["diameter_over_wavelength"] > CRIT["threshold_fraction_of_wavelength"]]
    below = [r for r in retention if r["diameter_over_wavelength"] < CRIT["threshold_fraction_of_wavelength"]]
    crit_a = {
        "min_probability_above_threshold": min(r["probability"] for r in above),
        "max_probability_below_threshold": max(r["probability"] for r in below),
        "threshold_probability": next(
            r["probability"]
            for r in retention
            if abs(r["diameter_over_wavelength"] - CRIT["threshold_fraction_of_wavelength"]) < 1e-12
        ),
        "pass": bool(
            min(r["probability"] for r in above) > CRIT["retention_probability_above"]
            and max(r["probability"] for r in below) < CRIT["retention_probability_below"]
        ),
    }
    crit_b = {
        "aspect_ratio_min": ar_min,
        "aspect_ratio_max": ar_max,
        "band": list(CRIT["aspect_ratio_band"]),
        "geometries_out_of_band": sum(1 for r in geometry if r["issues_count"] > 0),
        "n_geometries": len(geometry),
        "pass": bool(
            ar_ok
            and CRIT["aspect_ratio_band"][0] <= ar_min
            and ar_max <= CRIT["aspect_ratio_band"][1]
        ),
    }
    crit_c = {
        "alpha": fit["alpha"],
        "beta": fit["beta"],
        "r2": fit["r2"],
        "alpha_band": list(CRIT["alpha_band"]),
        "beta_band": list(CRIT["beta_band"]),
        "pass": bool(
            CRIT["alpha_band"][0] <= fit["alpha"] <= CRIT["alpha_band"][1]
            and CRIT["beta_band"][0] <= fit["beta"] <= CRIT["beta_band"][1]
        ),
    }
    low = next(r for r in shares if abs(r["C"] - CRIT["low_C"]) < 1e-12)
    high = next(r for r in shares if abs(r["C"] - CRIT["high_C"]) < 1e-12)
    crit_d = {
        "slip_share_at_C": CRIT["low_C"],
        "slip_share": low["slip_share"],
        "bulldozing_share_at_C": CRIT["high_C"],
        "bulldozing_share": high["bulldozing_share"],
        "pass": bool(
            low["slip_share"] > CRIT["slip_share_low_C"]
            and high["bulldozing_share"] > CRIT["bulldozing_share_high_C"]
        ),
    }
    zero = [
        pi.pancake_resistance(0.0, float(d), float(d) / 8.0, float(v))["resistance_N"]
        for d in D_GRID
        for v in V_GRID
    ]
    crit_e = {
        "max_resistance_at_zero_concentration": float(max(abs(z) for z in zero)),
        "n_points": len(zero),
        "pass": bool(max(abs(z) for z in zero) == 0.0),
    }

    cyl = table["cylinder_cd"]
    cyl_ref = CRIT["cylinder_cd_re100"]
    cyl_key = next((k for k in cyl if k.endswith("re100")), None)
    cyl_cd = cyl[cyl_key] if cyl_key else float("nan")
    crit_lbm = {
        "cylinder_cd_re100": cyl_cd,
        "literature": cyl_ref,
        "relative_error": abs(cyl_cd - cyl_ref) / cyl_ref if cyl_key else float("nan"),
        "tolerance": CRIT["cylinder_cd_tolerance"],
        "cd_plateau": table["cd_plateau"],
        "plateau_band": list(CRIT["cd_literature_band"]),
        "pass": bool(
            cyl_key
            and abs(cyl_cd - cyl_ref) / cyl_ref <= CRIT["cylinder_cd_tolerance"]
            and CRIT["cd_literature_band"][0] <= table["cd_plateau"] <= CRIT["cd_literature_band"][1]
        ),
    }
    crit_f = {
        "tests_collected": pytest_info.get("collected"),
        "tests_passed": pytest_info.get("passed"),
        "required": 10,
        "pass": bool(
            (pytest_info.get("passed") or 0) >= 10 and (pytest_info.get("collected") or 0) >= 10
        ),
    }
    criteria = {
        "a_coalescence_criterion": crit_a,
        "b_pancake_geometry": crit_b,
        "c_resistance_scaling": crit_c,
        "d_mechanism_split": crit_d,
        "e_negative_control": crit_e,
        "f_pytest": crit_f,
        "lbm_drag_validation": crit_lbm,
    }
    n_pass = sum(1 for key, val in criteria.items() if val["pass"])
    return {
        "criteria": criteria,
        "n_pass": n_pass,
        "n_total": len(criteria),
        "all_pass": n_pass == len(criteria),
    }


def _fig_retention(retention: list[dict[str, float]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fracs = np.logspace(math.log10(0.004), math.log10(0.2), 200)
    probs = [pi.retention_probability(f * WAVE.wavelength, WAVE) for f in fracs]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.semilogx(fracs, probs, "C0-", lw=2, label="retention probability (sharpness s=30)")
    ax.axvline(1 / 20.0, color="k", ls="--", lw=1, label=r"$D=\lambda/20$ (Lange et al. 1989)")
    ax.axhline(0.5, color="C2", ls=":", lw=1)
    ax.axhline(0.2, color="C3", ls=":", lw=1)
    ax.scatter(
        [r["diameter_over_wavelength"] for r in retention],
        [r["probability"] for r in retention],
        color="C1",
        zorder=5,
        label="campaign test points",
    )
    ax.set_xlabel(r"$D/\lambda$ [-]")
    ax.set_ylabel("consolidation probability [-]")
    ax.set_title(r"Frazil $\to$ pancake criterion, $H/\lambda=0.1$ ($\lambda=60$ m)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(-0.03, 1.03)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_formation(formation: list[dict[str, object]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for row in formation:
        ax.semilogy(row["t_hours"], row["diameter_m"], lw=2, label=f"$\\lambda$={row['wavelength_m']:.0f} m")
        if not math.isnan(row["t_consolidation_h"]):
            ax.axvline(row["t_consolidation_h"], color="0.7", ls=":", lw=1)
    ax.axhline(1.0, color="k", ls="--", lw=1, label="example threshold $A_{orb}=1$ m")
    ax.set_xlabel("time [h]")
    ax.set_ylabel("aggregate diameter $D$ [m]")
    ax.set_title("frazil aggregation growth (collision frequency x sticking x retention)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_resistance(sweep: dict[str, object], path_c: Path, path_v: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = sweep["grid"]
    params = sweep["params"]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for v in (1.0, 2.0, 3.0, 5.0):
        series = []
        for c in C_GRID:
            series.append(pi.pancake_resistance(float(c), 1.0, 0.1, v, params=params)["resistance_N"])
        ax.loglog(C_GRID, series, "o-", label=f"V={v:g} m/s")
    ax.set_xlabel("pancake concentration $C$ [-]")
    ax.set_ylabel("added resistance $R$ [N]")
    ax.set_title("resistance vs concentration (D=1 m, h=0.1 m)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path_c, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for c in C_GRID:
        series = [
            pi.pancake_resistance(float(c), 1.0, 0.1, float(v), params=params)["resistance_N"]
            for v in V_GRID
        ]
        ax.loglog(V_GRID, series, "o-", label=f"C={c:g}")
    ax.set_xlabel("ship speed $V$ [m/s]")
    ax.set_ylabel("added resistance $R$ [N]")
    ax.set_title(r"resistance vs speed; fitted $R\propto C^{\alpha}V^{\beta}$")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path_v, dpi=150)
    plt.close(fig)


def _fig_mechanism(shares: list[dict[str, float]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = np.array([r["C"] for r in shares])
    slip = np.array([r["slip_N"] for r in shares])
    bull = np.array([r["bulldozing_N"] for r in shares])
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.stackplot(c, slip / 1e3, bull / 1e3, labels=["slip", "bulldozing"], alpha=0.75)
    ax.set_xlabel("pancake concentration $C$ [-]")
    ax.set_ylabel("resistance component [kN]")
    ax.set_title("mechanism decomposition (D=1 m, h=0.1 m, V=3 m/s)")
    ax.plot(c, (slip + bull) / 1e3, "k--", lw=1, label="total")
    ax2 = ax.twinx()
    ax2.plot(c, slip / (slip + bull), "C3o-", ms=4, label="slip share")
    ax2.set_ylabel("slip share [-]")
    ax2.set_ylim(0, 1.05)
    ax.axvline(pi.DEFAULT_PARAMS["c_jam"], color="0.4", ls=":", lw=1)
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_lbm(table: dict[str, object], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    res = table["reynolds"]
    for ar in table["aspect_ratios"]:
        row = table["cd_by_ar_re"][f"{ar:g}"]
        res_ar = [re for re in res if f"{re:g}" in row]
        cds = [row[f"{re:g}"] for re in res_ar]
        ax.plot(res_ar, cds, "o-", label=f"pancake section D/h={ar:g}")
    ax.axhline(table["cd_plateau"], color="k", ls="--", lw=1, label="plateau (used for $C_d$)")
    cyl = table["cylinder_cd"]
    for key, value in cyl.items():
        re_val = float(key.split("_re")[-1])
        ax.plot([re_val], [value], "s", ms=7, color="C3")
    ax.plot([100.0], [CRIT["cylinder_cd_re100"]], "kx", ms=9, label="cylinder lit. $C_d$(Re=100)=1.34")
    ax.plot([40.0], [1.54], "k+", ms=9, label="cylinder lit. $C_d$(Re=40)=1.54")
    ax.axhspan(*CRIT["cd_literature_band"], color="0.85", alpha=0.5, label="blunt-plate band")
    ax.set_xlabel("Re = $V D/\\nu$ [-]")
    ax.set_ylabel("$C_d$ [-]")
    ax.set_title("LBM (D2Q9 bounce-back) drag of pancake sections")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_field(path_npz: Path, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.load(path_npz)
    ux, uy, mask = data["ux"], data["uy"], data["mask"]
    speed = np.hypot(ux, uy)
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    cf = ax.imshow(speed, cmap="viridis", origin="lower")
    ax.contour(mask, levels=[0.5], colors="w", linewidths=1.0)
    fig.colorbar(cf, ax=ax, label="$|u|$ [lattice units]")
    ax.set_title("D2Q9 bounce-back field, pancake section D/h=10 at Re=100")
    ax.set_xlabel("x [cells]")
    ax.set_ylabel("y [cells]")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_fields(path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4))
    for ax, c in zip(axes, (0.2, 0.5, 0.8), strict=True):
        mask = pi.pancake_field_mask(240, 240, concentration=c, diameter=1.0, cell_size=0.05, seed=7)
        ax.imshow(mask, cmap="Blues", origin="lower")
        ax.set_title(f"C={c:g}, measured {mask.mean():.3f}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("plan-view pancake field generator (disc + jittered lattice)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _pytest_info() -> dict[str, object]:
    cmd = [sys.executable, "-m", "pytest", "tests/test_pancake_ice.py", "-q"]
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    try:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:  # pragma: no cover
        return {"collected": None, "passed": None, "returncode": None, "tail": "timeout"}
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    passed = None
    for token in tail.replace(",", " ").split():
        if token.isdigit() and passed is None and "passed" in tail:
            passed = int(token)
    return {
        "collected": passed,
        "passed": passed if proc.returncode == 0 else 0,
        "returncode": proc.returncode,
        "tail": tail,
        "command": " ".join(cmd),
    }


def _markdown_report(
    acceptance: dict[str, object],
    sweep: dict[str, object],
    table: dict[str, object],
    retention: list[dict[str, float]],
    geometry: list[dict[str, object]],
    formation: list[dict[str, object]],
    shares: list[dict[str, float]],
) -> str:
    lines: list[str] = []
    lines.append("<!-- AUTO-GENERATED by tools/run_pancake_resistance.py --assemble -->")
    lines.append("")
    lines.append("## 1. Acceptance criteria (machine evaluated)")
    lines.append("")
    lines.append("| criterion | value | threshold | pass |")
    lines.append("|---|---|---|---|")
    crit = acceptance["criteria"]
    a = crit["a_coalescence_criterion"]
    lines.append(
        "| (a) coalescence | min p(D>λ/20)=%.4f, max p(D<λ/20)=%.4f, p(λ/20)=%.4f | >0.5 / <0.2 | %s |"
        % (a["min_probability_above_threshold"], a["max_probability_below_threshold"], a["threshold_probability"], a["pass"])
    )
    b = crit["b_pancake_geometry"]
    lines.append(
        "| (b) geometry | D/h in [%.2f, %.2f], %d/%d in band | [5, 20] | %s |"
        % (b["aspect_ratio_min"], b["aspect_ratio_max"], b["n_geometries"] - b["geometries_out_of_band"], b["n_geometries"], b["pass"])
    )
    c = crit["c_resistance_scaling"]
    lines.append(
        "| (c) scaling | alpha=%.3f, beta=%.3f (R2=%.4f) | alpha in [1,2], beta in [1.5,2.5] | %s |"
        % (c["alpha"], c["beta"], c["r2"], c["pass"])
    )
    d = crit["d_mechanism_split"]
    lines.append(
        "| (d) mechanism split | slip(C=%.1f)=%.3f, bulldozing(C=%.1f)=%.3f | >0.60 / >0.50 | %s |"
        % (d["slip_share_at_C"], d["slip_share"], d["bulldozing_share_at_C"], d["bulldozing_share"], d["pass"])
    )
    e = crit["e_negative_control"]
    lines.append(
        "| (e) negative control | max |R(C=0)| = %.3e N over %d points | exactly 0 | %s |"
        % (e["max_resistance_at_zero_concentration"], e["n_points"], e["pass"])
    )
    f = crit["f_pytest"]
    lines.append(
        "| (f) pytest | %s passed / %s collected | >= 10 | %s |"
        % (f["tests_passed"], f["tests_collected"], f["pass"])
    )
    lbm = crit["lbm_drag_validation"]
    lines.append(
        "| LBM validation | cylinder Cd(Re=100)=%.3f vs 1.34 (%.1f%%), plateau %.3f | <=15%%, plateau in [1.6,2.4] | %s |"
        % (lbm["cylinder_cd_re100"], 100 * lbm["relative_error"], lbm["cd_plateau"], lbm["pass"])
    )
    lines.append("")
    lines.append(f"**{acceptance['n_pass']}/{acceptance['n_total']} criteria pass** "
                 f"(all_pass={acceptance['all_pass']})")
    lines.append("")

    lines.append("## 2. Coalescence criterion (a)")
    lines.append("")
    lines.append("| D/λ | D [m] | p(consolidation) | retention | buoyancy | breakup | σ_bend [Pa] |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in retention:
        lines.append(
            "| 1/%.0f | %.3f | %.4f | %.4f | %.0f | %.0f | %.0f |"
            % (1.0 / r["diameter_over_wavelength"], r["diameter_m"], r["probability"],
               r["retention"], r["buoyancy"], r["breakup"], r["bending_stress_Pa"])
        )
    lines.append("")

    lines.append("## 3. Pancake geometry (b)")
    lines.append("")
    lines.append("| D [m] | h [m] | D/h | draft [m] | freeboard [m] | rim vol. frac. | added mass ratio | issues |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in geometry:
        if r["diameter_m"] in (0.3, 1.0, 3.0):
            lines.append(
                "| %.2f | %.3f | %.2f | %.4f | %.4f | %.4f | %.2f | %d |"
                % (r["diameter_m"], r["thickness_m"], r["aspect_ratio"], r["draft_m"],
                   r["freeboard_disc_m"], r["volume_rim_fraction"], r["added_mass_ratio"], r["issues_count"])
            )
    lines.append("")

    lines.append("## 4. Formation time (collision frequency x sticking x retention)")
    lines.append("")
    lines.append("| λ [m] | A_orb [m] | seed D0 [m] | D_sat [m] | t_consolidation [h] | t_thermo [h] | t_form [h] |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in formation:
        tc = r["t_consolidation_h"]
        lines.append(
            "| %.0f | %.2f | %.2f | %.2f | %s | %.1f | %.1f |"
            % (r["wavelength_m"], r["threshold_m"], r["d0_m"], r["d_saturation_m"],
               ("%.3f" % tc) if not math.isnan(tc) else "not reached", r["t_thermo_h"], r["t_form_h"])
        )
    lines.append("")

    lines.append("## 5. Resistance scaling fits (c)")
    lines.append("")
    lines.append("| subset | alpha | beta | a | R2 | n |")
    lines.append("|---|---|---|---|---|---|")
    for name, fit in sweep["fits"].items():
        lines.append(
            "| %s | %.3f | %.3f | %.3e | %.4f | %d |"
            % (name, fit["alpha"], fit["beta"], fit["a"], fit["r2"], fit["n_points"])
        )
    lines.append("")

    lines.append("## 6. Mechanism decomposition (d), D=1 m, h=0.1 m, V=3 m/s")
    lines.append("")
    lines.append("| C | R [N] | slip [N] | bulldozing [N] | slip share | h_eff [m] |")
    lines.append("|---|---|---|---|---|---|")
    for r in shares:
        lines.append(
            "| %.1f | %.1f | %.1f | %.1f | %.3f | %.4f |"
            % (r["C"], r["resistance_N"], r["slip_N"], r["bulldozing_N"], r["slip_share"], r["thickness_effective_m"])
        )
    lines.append("")

    lines.append("## 7. LBM drag table")
    lines.append("")
    keys = [f"{re:g}" for re in table["reynolds"]]
    lines.append("| D/h | " + " | ".join(f"Cd(Re={k})" for k in keys) + " |")
    lines.append("|---" * (len(keys) + 1) + "|")
    for ar in table["aspect_ratios"]:
        row = table["cd_by_ar_re"][f"{ar:g}"]
        lines.append(
            "| %g | " % ar + " | ".join(("%.3f" % row[k]) if k in row else "-" for k in keys) + " |"
        )
    lines.append("")
    lines.append("| case | Cd | stderr | tau | blockage | runtime [s] |")
    lines.append("|---|---|---|---|---|---|")
    for case in table["cases"]:
        lines.append(
            "| %s | %.3f | %.3f | %.4f | %.3f | %.1f |"
            % (case["case_id"], _corrected(case), case["cd_stderr"], case["tau"], case["blockage"], case["runtime_s"])
        )
    lines.append("")

    lines.append("## 8. Parameter sensitivity (+/-50% on declared coefficients)")
    lines.append("")
    lines.append("| parameter | factor | alpha | beta | slip share @C=0.1 | bulldozing share @C=0.8 |")
    lines.append("|---|---|---|---|---|---|")
    for row in sweep["sensitivity"]:
        lines.append(
            "| %s | %.2f | %.3f | %.3f | %.3f | %.3f |"
            % (row["parameter"], row["factor"], row["alpha"], row["beta"],
               row["slip_share_low_C"], row["bulldozing_share_high_C"])
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def stage_assemble() -> Path:
    sweep_path = RUNS / "sweep.json"
    if not sweep_path.exists():
        raise SystemExit("missing runs/pancake_ice/sweep.json; run --sweep first")
    sweep = json.loads(sweep_path.read_text())
    table = load_cd_table()
    retention = _retention_table()
    geometry = _geometry_table()
    formation = _formation_table()
    shares = sweep["shares"]
    pytest_info = _pytest_info()
    acceptance = evaluate_acceptance(sweep, table, retention, shares, pytest_info)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    figures: dict[str, str] = {}
    f = FIGURES / "fig_retention.png"
    _fig_retention(retention, f)
    figures["retention"] = str(f.relative_to(ROOT))
    f = FIGURES / "fig_formation.png"
    _fig_formation(formation, f)
    figures["formation"] = str(f.relative_to(ROOT))
    _fig_resistance(sweep, FIGURES / "fig_resistance_vs_C.png", FIGURES / "fig_resistance_vs_V.png")
    figures["resistance_vs_C"] = str((FIGURES / "fig_resistance_vs_C.png").relative_to(ROOT))
    figures["resistance_vs_V"] = str((FIGURES / "fig_resistance_vs_V.png").relative_to(ROOT))
    f = FIGURES / "fig_mechanism_split.png"
    _fig_mechanism(shares, f)
    figures["mechanism_split"] = str(f.relative_to(ROOT))
    f = FIGURES / "fig_lbm_cd.png"
    _fig_lbm(table, f)
    figures["lbm_cd"] = str(f.relative_to(ROOT))
    snap = RUNS / "lbm" / "field_plate_ar10_re400.npz"
    if snap.exists():
        f = FIGURES / "fig_lbm_field.png"
        _fig_field(snap, f)
        figures["lbm_field"] = str(f.relative_to(ROOT))
    f = FIGURES / "fig_pancake_fields.png"
    _fig_fields(f)
    figures["pancake_fields"] = str(f.relative_to(ROOT))

    md = _markdown_report(acceptance, sweep, table, retention, geometry, formation, shares)
    results_md = DOCS / "pancake_ice_results.md"
    results_md.parent.mkdir(parents=True, exist_ok=True)
    results_md.write_text(
        "# Pancake ice: formation dynamics and ship resistance - auto-generated results\n\n"
        + md
    )

    artifact = {
        "node": "IN69-N07",
        "title": "pancake ice formation dynamics and ship resistance",
        "wave_anchor": WAVE.to_dict(),
        "design": sweep["design"],
        "params": sweep["params"],
        "acceptance": acceptance,
        "retention_table": retention,
        "geometry_table": [{k: v for k, v in r.items() if k != "issues"} for r in geometry],
        "formation_table": [
            {k: v for k, v in r.items() if k not in ("t_hours", "diameter_m")} for r in formation
        ],
        "fits": sweep["fits"],
        "shares": shares,
        "sensitivity": sweep["sensitivity"],
        "lbm": {
            "aspect_ratios": table["aspect_ratios"],
            "reynolds": table["reynolds"],
            "cd_by_ar_re": table["cd_by_ar_re"],
            "cd_plateau": table["cd_plateau"],
            "cylinder_cd": table["cylinder_cd"],
            "cases": table["cases"],
            "provenance": table["provenance"],
        },
        "cd_inputs": sweep["cd_inputs"],
        "figures": figures,
        "results_markdown": str(results_md.relative_to(ROOT)),
        "provenance": _provenance(),
    }
    path = ARTIFACTS / "pancake_resistance.json"
    _dump(path, artifact)
    print(f"[assemble] artifact -> {path}")
    print(f"[assemble] tables   -> {results_md}")
    print(f"[assemble] {acceptance['n_pass']}/{acceptance['n_total']} criteria pass")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lbm", action="store_true", help="run the D2Q9 drag table")
    parser.add_argument("--only", default=None, help="run a single LBM case id")
    parser.add_argument("--sweep", action="store_true", help="run the analytic sweep and fits")
    parser.add_argument("--assemble", action="store_true", help="build artifact, figures and tables")
    parser.add_argument("--all", action="store_true", help="lbm + sweep + assemble")
    args = parser.parse_args(argv)
    if not any((args.lbm, args.sweep, args.assemble, args.all)):
        parser.print_help()
        return 2
    if args.lbm or args.all:
        stage_lbm(args.only)
    if args.sweep or args.all:
        stage_sweep()
    if args.assemble or args.all:
        stage_assemble()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())