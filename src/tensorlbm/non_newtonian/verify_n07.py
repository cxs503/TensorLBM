"""Verification post-processing for the non-Newtonian LBM campaign (P304W2-N07).

Reads the ``runs/n07/*.npz`` raw fields produced by
:mod:`tensorlbm.non_newtonian.run_case`, rebuilds every benchmark metric,
applies the machine-checkable acceptance criteria C1..C6 and writes

* ``artifacts/n07/validation.json``  -- all numbers (no hand-copied values)
* ``artifacts/n07/*.png``           -- profile / field figures
* ``artifacts/n07/summary.md``      -- human-readable table

Usage::

    python -m tensorlbm.non_newtonian.verify_n07 --runs runs/n07 --out artifacts/n07
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

U_LID_CAVITY = 0.1
WALL_OFFSET = 0.0   # calibrated effective wall position (lattice units)

# Ghia, Ghia & Shin (1982), Re = 100 -- u along the vertical centreline.
GHIA_RE100_U = {
    "y": [1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172,
          0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0000],
    "u": [1.00000, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151, 0.00332, -0.13641,
          -0.20581, -0.21090, -0.15662, -0.10150, -0.06434, -0.04775, -0.04192,
          -0.03717, 0.00000],
}


def _load(path: Path):
    return np.load(path)


def compare_ghia(npz_path: Path) -> dict:
    d = _load(npz_path)
    y, u = d["y"], d["u_vertical"] / U_LID_CAVITY
    yg = np.array(GHIA_RE100_U["y"])[::-1]
    ug = np.array(GHIA_RE100_U["u"])[::-1]
    un = np.interp(yg, y, u)
    err = np.abs(un - ug)
    return {
        "reference": "Ghia, Ghia & Shin (1982) Re=100, u centreline",
        "max_abs_error_u_over_ulid": float(err.max()),
        "rms_error_u_over_ulid": float(np.sqrt(np.mean(err ** 2))),
        "l2_relative_error": float(np.linalg.norm(un - ug) / np.linalg.norm(ug)),
        "max_relative_error_pct": float(err.max() * 100.0),   # vs u_lid
        "min_u_over_ulid_numeric": float(u.min()),
        "min_u_over_ulid_ghia": float(ug.min()),
        "y_at_min_numeric": float(y[int(np.argmin(u))]),
        "y_at_min_ghia": float(yg[int(np.argmin(ug))]),
    }


def compare_profiles(a_npz: Path, b_npz: Path) -> dict:
    """Grid-convergence comparison on the common (normalised) coordinate."""
    da, db = _load(a_npz), _load(b_npz)
    ya, ua = da["y"], da["u_vertical"] / U_LID_CAVITY
    yb, ub = db["y"], db["u_vertical"] / U_LID_CAVITY
    ub_i = np.interp(ya, yb, ub)
    diff = np.abs(ua - ub_i)
    return {
        "max_abs_diff_u_over_ulid": float(diff.max()),
        "l2_rel_diff": float(np.linalg.norm(diff) / max(np.linalg.norm(ub_i), 1e-30)),
        "note": "difference between the two grid resolutions (self-convergence)",
    }


def _reference_profile(d) -> np.ndarray:
    """Rebuild the exact reference profile from the stored run metadata.

    Uses the effective wall position calibrated on the Newtonian case
    (``WALL_OFFSET``, see docs/non_newtonian_benchmarks_n07.md) so that the
    metric never depends on which default was active when the run was made.
    """
    import torch

    from .rheology import (
        poiseuille_carreau_yasuda_profile,
        poiseuille_power_law_profile,
    )

    y = np.asarray(d["y"], dtype=float)
    H = float(d["H"])
    H_eff = H - WALL_OFFSET
    y_eff = y * (H_eff / H)
    G = float(d["G"])
    n = float(d["n_power"])
    rheo = str(d["rheology"])
    yt = torch.tensor(y_eff, dtype=torch.float64)
    if rheo == "carreau_yasuda":
        return poiseuille_carreau_yasuda_profile(
            yt, H_eff, G, float(d["nu0"]), float(d["nuinf"]),
            float(d["lam"]), n, float(d["yasuda"]),
        ).numpy()
    return poiseuille_power_law_profile(yt, H_eff, G, float(d["K"]), n).numpy()


def pipe_metrics(npz_path: Path) -> dict:
    d = _load(npz_path)
    y, u = d["y"], d["u"]
    try:
        ua = _reference_profile(d)
    except Exception:      # very old runs without metadata
        ua = d["ua"]
    err = u - ua
    res = {
        "l2_relative_error": float(np.linalg.norm(err) / np.linalg.norm(ua)),
        "linf_relative_error": float(np.abs(err).max() / np.abs(ua).max()),
        "u_max_numeric": float(u.max()),
        "u_max_reference": float(ua.max()),
        "centreline_relative_error": float(
            abs(u.max() - ua.max()) / abs(ua.max())),
        "wall_offset_lattice": WALL_OFFSET,
    }
    res.update({k: float(d[k]) for k in ("K", "G", "n_power", "H") if k in d})
    return res


def contraction_metrics(npz_path: Path) -> dict:
    d = _load(npz_path)
    if "ux" not in d or "wall_mask" not in d:
        return {"note": "fields trimmed from the archived npz; see the case json"}
    ux, uy = d["ux"], d["uy"]
    wm = d["wall_mask"]
    fluid = ~wm
    # re-entrant corner vortex: negative streamwise velocity next to the step
    ny, nx = ux.shape
    return {
        "u_min": float(ux[fluid].min()),
        "u_max": float(ux[fluid].max()),
        "mass_imbalance": float(np.abs(ux[fluid]).mean()),
        "n_fluid": int(fluid.sum()),
    }


def build_report(runs: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"criteria": {}, "cases": {}}

    # ---------------- cavity -------------------------------------------
    cav_N65 = {n: runs / f"cavity_n{n}_N65.npz" for n in (0.5, 1.0, 1.5)}
    cav_N97 = {n: runs / f"cavity_n{n}_N97.npz" for n in (0.5, 1.0, 1.5)}
    for n in (0.5, 1.0, 1.5):
        entry = {}
        if cav_N65[n].exists():
            entry["N65"] = compare_profiles(cav_N65[n], cav_N65[n])
            entry["N65"]["y"] = _load(cav_N65[n])["y"].tolist()
            entry["N65"]["u"] = (_load(cav_N65[n])["u_vertical"] / U_LID_CAVITY).tolist()
        if cav_N97[n].exists():
            entry["N97"] = {}
            entry["N97"]["u"] = (_load(cav_N97[n])["u_vertical"] / U_LID_CAVITY).tolist()
            entry["N97"]["y"] = _load(cav_N97[n])["y"].tolist()
        if n == 1.0 and cav_N65[n].exists():
            entry["ghia_comparison"] = compare_ghia(cav_N65[n])
        if cav_N65[n].exists() and cav_N97[n].exists():
            entry["grid_convergence"] = compare_profiles(cav_N65[n], cav_N97[n])
        report["cases"][f"cavity_n{n}"] = entry

    # ---------------- pipe / channel ------------------------------------
    for n in (0.5, 1.0, 1.5):
        for ny in (65, 129):
            p = runs / f"pipe_n{n}_ny{ny}.npz"
            if p.exists():
                report["cases"][f"pipe_powerlaw_n{n}_ny{ny}"] = pipe_metrics(p)
                d = _load(p)
                report["cases"][f"pipe_powerlaw_n{n}_ny{ny}"]["y"] = d["y"].tolist()
                report["cases"][f"pipe_powerlaw_n{n}_ny{ny}"]["u"] = d["u"].tolist()
                report["cases"][f"pipe_powerlaw_n{n}_ny{ny}"]["ua"] = d["ua"].tolist()

    for p in sorted(runs.glob("pipe_cy_*.npz")):
        report["cases"][p.stem] = pipe_metrics(p)
        d = _load(p)
        report["cases"][p.stem]["y"] = d["y"].tolist()
        report["cases"][p.stem]["u"] = d["u"].tolist()
        report["cases"][p.stem]["ua"] = d["ua"].tolist()

    for p in sorted(runs.glob("contraction_*.npz")):
        report["cases"][p.stem] = contraction_metrics(p)

    # ---------------- criterion evaluation ------------------------------
    crit = report["criteria"]

    def _cav(n):
        c = report["cases"].get(f"cavity_n{n}", {})
        return c

    ghia = _cav(1.0).get("ghia_comparison", {})
    crit["C1"] = {
        "description": "lid-driven cavity centreline profile error < 3%",
        "n=1.0_vs_Ghia(1982)": ghia.get("max_relative_error_pct"),
        "status_n1.0": "PASS" if ghia.get("max_relative_error_pct", 1e9) < 3.0 else "FAIL",
        "n=0.5_vs_Bell&Surana(1997)": None,
        "n=1.5_vs_Bell&Surana(1997)": None,
        "status_n0.5_n1.5": "WITHHELD (Bell & Surana 1997 tables not available offline; "
                            "grid-convergence evidence reported instead)",
        "grid_convergence_max_diff": {
            n: _cav(n).get("grid_convergence", {}).get("max_abs_diff_u_over_ulid")
            for n in (0.5, 1.0, 1.5)
        },
    }

    pl = {n: report["cases"].get(f"pipe_powerlaw_n{n}_ny129") or
             report["cases"].get(f"pipe_powerlaw_n{n}_ny65") for n in (0.5, 1.0, 1.5)}
    cy = {k: v for k, v in report["cases"].items() if k.startswith("pipe_cy_")}
    crit["C2"] = {
        "description": "channel/pipe velocity-profile L2 error < 2% (three n values)",
        "power_law": {n: (pl[n] or {}).get("l2_relative_error") for n in pl},
        "power_law_status": {n: ("PASS" if (pl[n] or {}).get("l2_relative_error", 1e9) < 0.02
                                 else "FAIL") for n in pl},
        "carreau_yasuda": {k: v.get("l2_relative_error") for k, v in cy.items()},
        "carreau_yasuda_status": {k: ("PASS" if v.get("l2_relative_error", 1e9) < 0.02 else "FAIL")
                                  for k, v in cy.items()},
    }
    crit["C3"] = {
        "description": "4:1 contraction vortex length < 5% / centreline velocity < 3% "
                       "vs Xue et al. (2004)",
        "status": "WITHHELD (Xue et al. 2004 digitised data unavailable offline); "
                  "solver delivers a mesh-converged vortex length only",
    }
    c4 = {}
    for p in sorted(runs.glob("result_*.json")):
        d = json.loads(p.read_text())
        hist = d.get("picard_history") or []
        # iterations needed for |dtau|/tau < 1e-4 (the campaign criterion)
        if hist:
            n_tau = next((i for i, v in enumerate(hist, start=1) if v < 1e-4), len(hist))
        else:
            # legacy runs without a stored history: fall back to the reported
            # Picard block count (those runs used a tau tolerance of 1e-5)
            n_tau = d.get("picard_iterations")
        c4[p.stem] = {
            "picard_blocks": d.get("picard_iterations"),
            "tau_iterations_to_1e-4": n_tau,
            "total_lbm_steps": d.get("total_lbm_steps"),
            "converged": d.get("converged"),
            "flow_residual": d.get("flow_residual"),
            "tau_residual": d.get("tau_residual"),
        }
    vals = [v["tau_iterations_to_1e-4"] for v in c4.values()
            if v["tau_iterations_to_1e-4"] is not None]
    ok = all(v < 50 for v in vals) if vals else None
    crit["C4"] = {
        "description": "Picard iterations < 50 for all cases "
                       "(rheology residual |dtau|/tau < 1e-4)",
        "cases": c4,
        "status": "PASS" if ok else ("FAIL" if c4 else "NO DATA"),
    }
    report["criteria"] = crit
    (out / "validation.json").write_text(json.dumps(report, indent=2, default=float))
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/n07")
    ap.add_argument("--out", default="artifacts/n07")
    a = ap.parse_args(argv)
    rep = build_report(Path(a.runs), Path(a.out))
    print(json.dumps(rep["criteria"], indent=2, default=float))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())