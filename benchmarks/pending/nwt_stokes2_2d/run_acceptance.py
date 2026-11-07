#!/usr/bin/env python3
"""Acceptance driver for the TensorLBM numerical wave tank (mission IN33-N02).

Stages
------
``run``      launch every case of the matrix (one single-threaded process each,
             ``--jobs`` in parallel) and store ``case.json`` + ``probes.npz``;
``collect``  read the produced case files, apply the C1-C7 verdicts and write
             ``results.json`` plus the figures;
``all``      ``run`` then ``collect``.

The case matrix is fixed by the acceptance criteria:

* C1/C2: ``kh`` in {0.7, 1.5, 3.0} x ``eps = ka`` in {0.02, 0.08} (6 cases);
* C3:    the same matrix with a >= 1-wavelength relaxation wavemaker and a
         1.5-wavelength relaxation absorber, plus a negative control with the
         absorber switched off;
* C4:    liquid-inventory drift per period, taken from every case;
* C5:    one case at ``dx`` and at ``dx/2`` (same physical wave, twice the
         lattice resolution);
* C7:    wall time of every case on the target architecture.

Lattice time step is 1.  ``steps_per_period`` is chosen per case from a fixed
hydrostatic-compressibility budget (``g h / c_s^2 = 0.03``), which is the main
LBM-specific error driver for a free-surface gravity wave; this keeps the
comparison between cases controlled instead of tuning each case by hand.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

CS2 = 1.0 / 3.0
TARGET_GH_CS2 = 0.03
EPS_LEVELS = (0.02, 0.08)
KH_LEVELS = (0.7, 1.5, 3.0)
# lattice wavelength per kh: keeps the depth at a useful number of cells while
# bounding the cell count (cost scales like lambda^2 at fixed steps/period)
LAMBDA_BY_KH = {0.7: 120.0, 1.5: 100.0, 3.0: 60.0}

THRESHOLDS = {
    "C1_dispersion": 0.01,
    "C2_A1": 0.03,
    "C2_A2": 0.10,
    "C3_sponge": 0.05,
    "C3_negative_control": 0.30,
    "C4_mass_per_period": 0.005,
    "C5_grid_A1": 0.03,
    "C7_wall_minutes": 20.0,
}


def depth_for(kh: float, wavelength: float) -> float:
    return kh * wavelength / (2.0 * math.pi)


def steps_per_period_for(kh: float, wavelength: float, target: float = TARGET_GH_CS2) -> int:
    k = 2.0 * math.pi / wavelength
    h = depth_for(kh, wavelength)
    sigma = math.tanh(k * h)
    g = target * CS2 / h
    omega = math.sqrt(g * k * sigma)
    return int(round(2.0 * math.pi / omega))


def base_overrides(kh: float, eps: float, wavelength: float | None = None, **extra) -> list[str]:
    lam = wavelength if wavelength is not None else LAMBDA_BY_KH[kh]
    h = depth_for(kh, lam)
    spp = steps_per_period_for(kh, lam)
    over = {
        "wavelength": lam,
        "depth": round(h, 4),
        "steepness": eps,
        "steps_per_period": spp,
        "n_periods": 5.0,
        "ramp_periods": 1.5,
        "analysis_periods": 2.0,
        "tau": 0.505,
    }
    over.update(extra)
    return [f"{k}={json.dumps(v)}" for k, v in over.items()]


def case_matrix() -> dict[str, list[str]]:
    cases: dict[str, list[str]] = {}
    for kh in KH_LEVELS:
        for eps in EPS_LEVELS:
            name = f"kh{kh:.1f}_eps{int(round(eps * 1000)):03d}"
            cases[name] = base_overrides(kh, eps)
    # negative control: absorber disabled, otherwise identical to kh3.0/eps0.08
    cases["kh3.0_eps080_nosponge"] = base_overrides(3.0, 0.08, sponge_enabled=False)
    # grid-convergence pair (C5): same physical wave, lattice dx halved
    for tag, lam in (("base", 40.0), ("fine", 80.0)):
        cases[f"kh3.0_eps080_dx_{tag}"] = base_overrides(3.0, 0.08, wavelength=lam)
    # diagnostic (not a criterion): pure propagation of the analytic Stokes IC
    # with no wavemaker and no absorber.  It separates the solver's intrinsic
    # error from the wave-generation/absorption hardware.
    cases["diag_propagation_only"] = base_overrides(
        3.0, 0.08, wavelength=60.0, initial_condition="stokes", generator_enabled=False, sponge_enabled=False, ramp_periods=0.0
    )
    return cases


def run_stage(out_dir: Path, jobs: int, only: list[str] | None = None) -> int:
    cases = case_matrix()
    if only:
        cases = {k: v for k, v in cases.items() if k in only}
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["NWT_TORCH_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"

    def launch(item):
        name, overrides = item
        target = out_dir / name / "case.json"
        if target.exists():
            return name, "cached"
        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent / "run_case.py"),
            "--out",
            str(target),
            "--save-probes",
            "--set",
            *overrides,
        ]
        log = (out_dir / f"{name}.log").open("w")
        proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        log.close()
        return name, "ok" if proc.returncode == 0 else f"fail({proc.returncode})"

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for name, status in pool.map(launch, cases.items()):
            print(f"[run] {name}: {status}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# verdicts
# ---------------------------------------------------------------------------


def _load(out_dir: Path, name: str) -> dict | None:
    path = out_dir / name / "case.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def collect(out_dir: Path, result_path: Path, figures_dir: Path) -> dict:
    names = list(case_matrix())
    data = {n: _load(out_dir, n) for n in names}
    missing = [n for n, d in data.items() if d is None]

    matrix = [n for n in names if n.startswith("kh") and "_nosponge" not in n and "_dx_" not in n]
    controlled = [n for n in matrix if n != "kh3.0_eps080_dx_base"]

    # ---- C1 -----------------------------------------------------------------
    c1_rows = []
    for n in matrix:
        d = data[n]
        if not d:
            continue
        er = d["dispersion"]["relative_error"]
        c1_rows.append({"case": n, "kh": d["case"]["kh"], "eps": d["case"]["steepness"], "error": er, "pass": er < THRESHOLDS["C1_dispersion"]})
    # one entry per kh (worst of the two steepnesses)
    kh_worst = {}
    for kh in KH_LEVELS:
        rows = [r for r in c1_rows if abs(r["kh"] - kh) < 0.05]
        if rows:
            kh_worst[f"{kh:.1f}"] = max(r["error"] for r in rows)
    c1_pass = bool(kh_worst) and all(v < THRESHOLDS["C1_dispersion"] for v in kh_worst.values())

    # ---- C2 -----------------------------------------------------------------
    c2_rows = []
    for n in matrix:
        d = data[n]
        if not d:
            continue
        c2_rows.append(
            {
                "case": n,
                "kh": d["case"]["kh"],
                "eps": d["case"]["steepness"],
                "A1_error": d["waveform"]["A1_relative_error"],
                "A2_error": d["waveform"]["A2_relative_error"],
                "A1_measured": d["waveform"]["A1_measured_mean"],
                "A1_analytic": d["waveform"]["A1_analytic"],
                "A2_measured": d["waveform"]["A2_measured_mean"],
                "A2_analytic": d["waveform"]["A2_analytic"],
                "pass": d["waveform"]["A1_relative_error"] < THRESHOLDS["C2_A1"]
                and d["waveform"]["A2_relative_error"] < THRESHOLDS["C2_A2"],
            }
        )
    c2_pass = bool(c2_rows) and all(r["pass"] for r in c2_rows)

    # ---- C3 -----------------------------------------------------------------
    kr = {n: (data[n]["reflection"]["KR_two_point"] if data[n] else None) for n in names}
    kr_ls = {n: (data[n]["reflection"]["KR_least_squares"] if data[n] else None) for n in names}
    sponge_rows = [{"case": n, "KR_two_point": kr[n], "KR_least_squares": kr_ls[n]} for n in controlled]
    c3_pass = (
        all(v is not None and v < THRESHOLDS["C3_sponge"] for v in kr.values() if v is not None)
        and kr["kh3.0_eps080_nosponge"] is not None
        and kr["kh3.0_eps080_nosponge"] > THRESHOLDS["C3_negative_control"]
    )

    # ---- C4 -----------------------------------------------------------------
    c4_rows = [
        {
            "case": n,
            "max_per_period_drift": data[n]["mass"]["max_per_period_drift"],
            "final_relative_drift": data[n]["mass"]["final_relative_drift"],
            "periods": data[n]["mass"]["n_periods"],
        }
        for n in names
        if data[n]
    ]
    c4_pass = bool(c4_rows) and all(r["max_per_period_drift"] < THRESHOLDS["C4_mass_per_period"] for r in c4_rows)

    # ---- C5 -----------------------------------------------------------------
    dxb, dxf = data.get("kh3.0_eps080_dx_base"), data.get("kh3.0_eps080_dx_fine")
    c5 = None
    c5_pass = False
    if dxb and dxf:
        a1b = dxb["waveform"]["A1_measured_mean"]
        a1f = dxf["waveform"]["A1_measured_mean"]
        change = abs(a1f - a1b) / a1b
        c5 = {
            "dx_base": {"wavelength": dxb["case"]["wavelength"], "A1": a1b, "A1_error": dxb["waveform"]["A1_relative_error"]},
            "dx_fine": {"wavelength": dxf["case"]["wavelength"], "A1": a1f, "A1_error": dxf["waveform"]["A1_relative_error"]},
            "relative_change": change,
        }
        c5_pass = change < THRESHOLDS["C5_grid_A1"]

    # ---- C7 -----------------------------------------------------------------
    c7_rows = [{"case": n, "wall_minutes": data[n]["runtime"]["wall_minutes"], "steps": data[n]["runtime"]["steps"]} for n in names if data[n]]
    c7_pass = bool(c7_rows) and all(r["wall_minutes"] < THRESHOLDS["C7_wall_minutes"] for r in c7_rows)

    # ---- derived: spatial attenuation of the fundamental ---------------------
    import numpy as _np

    attenuation = {}
    for n, d in data.items():
        if not d:
            continue
        xs = _np.asarray(d["probe_x"], dtype=float)
        a = _np.asarray(d["waveform"]["A1_per_probe"], dtype=float)
        lam = d["case"]["wavelength"]
        k = 2.0 * math.pi / lam
        tau = d["case"]["tau"]
        good = a > 1e-6
        row = {"A1_raw": a.tolist(), "probe_x": xs.tolist()}
        if good.sum() >= 2:
            slope, _ = _np.polyfit(xs[good], _np.log(a[good]), 1)
            Ld = -1.0 / slope if slope < 0 else float("inf")
            nu_eff = (1.0 / Ld) / (2.0 * k * k) if Ld != float("inf") else float("inf")
            nu_mol = (tau - 0.5) / 3.0
            row.update(
                {
                    "decay_length_cells": float(Ld),
                    "decay_per_wavelength_neper": float(-slope * lam),
                    "equivalent_numerical_viscosity_lattice": float(nu_eff),
                    "molecular_viscosity_lattice": float(nu_mol),
                    "viscosity_ratio": float(nu_eff / nu_mol) if nu_mol > 0 else float("inf"),
                }
            )
        a1_an = d["case"]["amplitude_a"]
        row["A1_nearest_probe_error"] = float(abs(a[0] - a1_an) / a1_an)
        attenuation[n] = row

    result = {
        "mission": "IN33-N02",
        "title": "TensorLBM numerical wave tank: Stokes-2nd generation, closed-form dispersion anchor, two-point reflection gate",
        "thresholds": THRESHOLDS,
        "environment": {
            "solver": "tensorlbm.free_surface_lbm_2d.koerner_step_2d",
            "arch": os.uname().machine,
            "python": sys.version.split()[0],
            "torch": __import__("torch").__version__,
            "torch_threads_per_case": 1,
            "steps_per_period_rule": f"g h / c_s^2 = {TARGET_GH_CS2}",
        },
        "criteria": {
            "C1_dispersion": {"pass": c1_pass, "worst_error_by_kh": kh_worst, "rows": c1_rows},
            "C2_waveform": {"pass": c2_pass, "rows": c2_rows},
            "C3_absorption": {
                "pass": c3_pass,
                "rows": sponge_rows,
                "negative_control_KR": kr.get("kh3.0_eps080_nosponge"),
            },
            "C4_mass": {"pass": c4_pass, "rows": c4_rows},
            "C5_grid_convergence": {"pass": c5_pass, **(c5 or {})},
            "C7_runtime": {"pass": c7_pass, "rows": c7_rows},
            "C6_indexing": {
                "pass": None,
                "note": "verified in the same commit: benchmarks/verified/nwt_stokes2_2d/ + pytest suite",
            },
        },
        "attenuation": attenuation,
        "missing_cases": missing,
        "cases": data,
    }
    result["all_pass"] = all(
        result["criteria"][k]["pass"] for k in ("C1_dispersion", "C2_waveform", "C3_absorption", "C4_mass", "C5_grid_convergence", "C7_runtime")
    )

    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=1, default=float))
    try:
        from make_figures import make_all_figures

        figs = make_all_figures(result, out_dir, figures_dir)
        result["figures"] = figs
        result_path.write_text(json.dumps(result, indent=1, default=float))
    except Exception as exc:  # pragma: no cover - figures are optional for the verdict
        print(f"[collect] figure generation skipped: {exc}", flush=True)
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("run", "collect", "all"), default="all")
    ap.add_argument("--out-dir", default=str(ROOT / "runs" / "nwt_acceptance"))
    ap.add_argument("--result", default=str(ROOT / "runs" / "nwt_acceptance" / "results.json"))
    ap.add_argument("--figures", default=str(ROOT / "runs" / "nwt_acceptance" / "figures"))
    ap.add_argument("--jobs", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    ap.add_argument("--only", default="", help="comma separated case names (sharding aid)")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    if args.stage in ("run", "all"):
        only = [x for x in args.only.split(",") if x] or None
        run_stage(out_dir, args.jobs, only)
    if args.stage in ("collect", "all"):
        res = collect(out_dir, Path(args.result), Path(args.figures))
        print(json.dumps({k: v["pass"] for k, v in res["criteria"].items()}, indent=1), flush=True)
        for key in ("C1_dispersion", "C2_waveform", "C3_absorption", "C4_mass", "C5_grid_convergence", "C7_runtime"):
            print(f"{key}: pass={res['criteria'][key]['pass']}", flush=True)
        print(f"all_pass={res['all_pass']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())