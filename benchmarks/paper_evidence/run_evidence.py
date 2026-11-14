#!/usr/bin/env python3
"""Paper-evidence bundle runner (GAP-DIFFLBM-PAPER-EVIDENCE).

One command per benchmark item; every item writes a self-describing JSON that
binds a *reference value* (analytic closed form or a literature table stored in
this checkout together with its citation) to the *value this code actually
produced*, plus the per-quantity relative error and the 3 % verdict.

Items (see ``ITEMS`` below)
---------------------------
analytic   A1 2-D plane Poiseuille           Boussinesq parabola
           A2 3-D square-duct Poiseuille     Boussinesq series + Shah-London fRe
           A3 shear-wave viscosity recovery  nu = (tau-1/2)/3
literature L1 lid-driven cavity Re=100        Ghia, Ghia & Shin (1982)
           L2 lid-driven cavity Re=400        Ghia, Ghia & Shin (1982)
           L3 periodic cylinder array        Sangani & Acrivos (1982)
cross      X1 LBM vs FVM vs FEM, same cavity  three independent discretisations

Design rules kept deliberately strict:

* nothing is retyped by hand -- every number in the emitted JSON is read from a
  solver return value or from the in-repo reference table at run time;
* every item is measured at two resolutions (or two parameter points) so a
  convergence / sensitivity statement is possible;
* the analytic items reuse the *asserted* case functions of
  ``tests/test_lbm_verification.py`` (imported by path) instead of a copy, so
  the published numbers cannot drift from the CI gate;
* the literature items reuse the already-verified runners
  (``benchmarks/verified/...``) as subprocesses -- this bundle only arbitrates
  and tabulates, it does not re-implement them.

Usage
-----
    python benchmarks/paper_evidence/run_evidence.py --list
    python benchmarks/paper_evidence/run_evidence.py --item A1_poiseuille_2d
    python benchmarks/paper_evidence/run_evidence.py --item-index 3   # 1-based
    python benchmarks/paper_evidence/run_evidence.py --all
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

THRESHOLD_PCT = 3.0  # task book: >15% unacceptable, <=3% is the paper claim
OUT_DIR_DEFAULT = REPO / "runs" / "paper_evidence"

# --------------------------------------------------------------------------- #
# item registry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Item:
    key: str
    category: str          # analytic | literature | cross_solver
    title: str
    repro_cmd: str


ITEMS: tuple[Item, ...] = (
    Item(
        "A1_poiseuille_2d",
        "analytic",
        "2-D plane channel Poiseuille flow vs the parabolic closed form",
        "python benchmarks/paper_evidence/run_evidence.py --item A1_poiseuille_2d",
    ),
    Item(
        "A2_square_duct_3d",
        "analytic",
        "3-D square-duct Poiseuille flow vs the Boussinesq series and f*Re = 56.91",
        "python benchmarks/paper_evidence/run_evidence.py --item A2_square_duct_3d",
    ),
    Item(
        "A3_shear_wave_viscosity",
        "analytic",
        "shear-wave amplitude decay vs nu = (tau - 1/2)/3",
        "python benchmarks/paper_evidence/run_evidence.py --item A3_shear_wave_viscosity",
    ),
    Item(
        "L1_cavity_ghia_re100",
        "literature",
        "lid-driven cavity Re=100 vs Ghia, Ghia & Shin (1982) centreline tables",
        "python benchmarks/paper_evidence/run_evidence.py --item L1_cavity_ghia_re100",
    ),
    Item(
        "L2_cavity_ghia_re400",
        "literature",
        "lid-driven cavity Re=400 vs Ghia, Ghia & Shin (1982) centreline tables",
        "python benchmarks/paper_evidence/run_evidence.py --item L2_cavity_ghia_re400",
    ),
    Item(
        "L3_cylinder_array_sangani",
        "literature",
        "2-D periodic cylinder array permeability vs Sangani & Acrivos (1982)",
        "python benchmarks/paper_evidence/run_evidence.py --item L3_cylinder_array_sangani",
    ),
    Item(
        "X1_cross_solver_cavity",
        "cross_solver",
        "LBM / FVM(psi-omega) / FEM(Q2-Q1) three-way agreement + MMS orders",
        "python benchmarks/paper_evidence/run_evidence.py --item X1_cross_solver_cavity",
    ),
)
ITEM_BY_KEY = {it.key: it for it in ITEMS}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:  # pragma: no cover - git not available
        return "unknown"


def _env() -> dict:
    import torch

    return {
        "git_sha": _git_sha(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": __import__("numpy").__version__,
        "platform": platform.platform(),
        "torch_threads": torch.get_num_threads(),
        "default_dtype": str(torch.get_default_dtype()).replace("torch.", ""),
    }


def row(quantity: str, ref, got, err_pct: float, threshold_pct: float = THRESHOLD_PCT,
        unit: str = "", note: str = "") -> dict:
    """One line of the per-item error table (ref -> got -> error -> verdict)."""
    return {
        "quantity": quantity,
        "unit": unit,
        "ref": ref,
        "got": got,
        "err_pct": float(err_pct),
        "threshold_pct": float(threshold_pct),
        "pass": bool(float(err_pct) < float(threshold_pct)),
        "note": note,
    }


def _order(e_coarse: float, e_fine: float, r: float) -> float | None:
    """Observed order from two errors separated by refinement ratio r."""
    if e_fine <= 0 or e_coarse <= 0 or r <= 1:
        return None
    return float(math.log(e_coarse / e_fine) / math.log(r))


def _write(out_dir: Path, key: str, payload: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{key}.json"
    payload["generated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    return out


def _load_analytic_module():
    """Import the asserted analytic ladder by path (single source of truth)."""
    path = REPO / "tests" / "test_lbm_verification.py"
    spec = importlib.util.spec_from_file_location("tlbm_analytic_ladder", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _run(cmd: list[str], log_prefix: str) -> None:
    print(f"[{log_prefix}] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(REPO), check=True, env={**os.environ, "PYTHONPATH": str(SRC)})


# --------------------------------------------------------------------------- #
# A1 -- plane Poiseuille
# --------------------------------------------------------------------------- #


def item_A1(out_dir: Path) -> dict:
    import torch

    mod = _load_analytic_module()
    rows, per_res = [], []
    for ny in (32, 64):
        r = mod.poiseuille_case(torch.device("cpu"), ny=ny, steps=4000)
        per_res.append(
            {
                "resolution": ny,
                "height_cells": r["height"],
                "steps": r["steps"],
                "tau": r["tau"],
                "nu": r["nu"],
                "body_force": r["body_force"],
                "u_max_num": r["u_max_num"],
                "u_max_exact": r["u_max_exact"],
                "err_profile_pct": 100.0 * r["rel_err"],
                "u_num": r["u_num"],
                "u_exact": r["u_exact"],
                "y": r["y"],
            }
        )
        rows.append(
            row(
                f"u_max at H={r['height']} vs g*H^2/(8 nu)", r["u_max_exact"], r["u_max_num"],
                100.0 * abs(r["u_max_num"] - r["u_max_exact"]) / r["u_max_exact"],
            )
        )
        rows.append(
            row(
                f"profile max|u-u_exact|/u_max at H={r['height']}", 0.0, r["u_max_num"],
                100.0 * r["rel_err"],
                note="normalised by the analytic peak",
            )
        )
    order = _order(per_res[0]["err_profile_pct"], per_res[1]["err_profile_pct"], 2.0)

    # noise floor: the same tightest case in float32 vs float64 shows whether the
    # reported error is discretisation or arithmetic.
    floors = {}
    for dt in ("float64", "float32"):
        torch.set_default_dtype(getattr(torch, dt))
        try:
            rr = mod.poiseuille_case(torch.device("cpu"), ny=64, steps=4000)
            floors[dt] = 100.0 * rr["rel_err"]
        finally:
            torch.set_default_dtype(torch.float64)

    return {
        "item": "A1_poiseuille_2d",
        "category": "analytic",
        "title": ITEM_BY_KEY["A1_poiseuille_2d"].title,
        "anchor": {
            "kind": "analytic",
            "source": "steady plane Poiseuille flow, u(y)=g/(2 nu)(y-1/2)(H+1/2-y) "
                      "for half-way bounce-back walls at y=0.5 and y=H+0.5",
            "citation": "Boussinesq (1868) parabolic channel solution; lattice form as in "
                        "Kruger et al., The Lattice Boltzmann Method (2017), ch. 6",
            "pointers": ["tests/test_lbm_verification.py::poiseuille_case (Step C)"],
        },
        "settings": {
            "resolutions": [p["resolution"] for p in per_res],
            "steps": 4000, "nx": 80, "nz": 4, "tau": per_res[0]["tau"],
            "u_max_target": 0.05,
        },
        "per_resolution": per_res,
        "rows": rows,
        "observed_order": order,
        "noise_floor_pct": floors,
        "repro_cmd": ITEM_BY_KEY["A1_poiseuille_2d"].repro_cmd,
        "env": _env(),
    }


# --------------------------------------------------------------------------- #
# A2 -- square duct
# --------------------------------------------------------------------------- #


def item_A2(out_dir: Path) -> dict:
    import torch

    mod = _load_analytic_module()
    rows, per_res = [], []
    for length in (10, 14):
        r = mod.duct_poiseuille_case(torch.device("cpu"), length=length, steps=4000)
        per_res.append(
            {
                "resolution": length,
                "steps": r["steps"],
                "tau": r["tau"],
                "nu": r["nu"],
                "body_force": r["body_force"],
                "err_profile_pct": 100.0 * r["rel_err"],
                "err_l2_pct": 100.0 * r["err_l2"],
                "err_center_pct": 100.0 * r["err_center"],
                "u_max_num": r["u_max_num"],
                "u_max_exact": r["u_max_exact"],
                "resistance_num": r["resistance_num"],
                "resistance_exact_nodal": r["resistance_exact_nodal"],
                "resistance_literature": r["resistance_literature"],
                "resistance_err_nodal_pct": 100.0 * r["resistance_err_nodal"],
                "resistance_err_literature_pct": 100.0 * r["resistance_err_literature"],
                "diag_coord": r["diag_coord"],
                "diag_num": r["diag_num"],
                "diag_exact": r["diag_exact"],
            }
        )
        rows.append(
            row(f"profile max err / u_max, L={length}", 0.0, r["u_max_num"],
                100.0 * r["rel_err"], note="vs Boussinesq series on the same nodes")
        )
        rows.append(
            row(f"centre-line L2 err, L={length}", 0.0, r["u_max_num"],
                100.0 * r["err_l2"])
        )
        rows.append(
            row(
                f"resistance R=gL^2/(nu u_mean) vs series, L={length}",
                r["resistance_exact_nodal"], r["resistance_num"],
                100.0 * r["resistance_err_nodal"],
                note="same-node average, quadrature difference excluded",
            )
        )
        rows.append(
            row(
                f"resistance R vs Shah-London f*Re=56.91, L={length}",
                r["resistance_literature"], r["resistance_num"],
                100.0 * r["resistance_err_literature"],
                note="literature integral quantity, nodal-average quadrature bias included",
            )
        )
    order = _order(per_res[0]["err_profile_pct"], per_res[1]["err_profile_pct"], 14 / 10)

    return {
        "item": "A2_square_duct_3d",
        "category": "analytic",
        "title": ITEM_BY_KEY["A2_square_duct_3d"].title,
        "anchor": {
            "kind": "analytic+literature",
            "source": "3-D duct: Boussinesq double-series solution; integral check against "
                      "the laminar friction-constant table f*Re = 56.91 (R = 28.455)",
            "citation": "Boussinesq (1868); Shah & London (1978) Laminar Flow Forced "
                        "Convection in Ducts, Adv. Heat Transfer Suppl. 1",
            "pointers": ["tests/test_lbm_verification.py::duct_poiseuille_case (Step E)",
                         "tests/test_lbm_verification.py::duct_series"],
        },
        "settings": {
            "resolutions": [p["resolution"] for p in per_res],
            "steps": 4000, "nx_periodic": 4, "tau": per_res[0]["tau"],
            "u_nominal": 0.05,
        },
        "per_resolution": per_res,
        "rows": rows,
        "observed_order": order,
        "repro_cmd": ITEM_BY_KEY["A2_square_duct_3d"].repro_cmd,
        "env": _env(),
    }


# --------------------------------------------------------------------------- #
# A3 -- shear wave
# --------------------------------------------------------------------------- #


def item_A3(out_dir: Path) -> dict:
    import torch

    mod = _load_analytic_module()
    rows, per_res = [], []
    for nx in (32, 64):
        r = mod.shear_wave_case(torch.device("cpu"), nx=nx, steps=100, tau=1.0)
        per_res.append(
            {
                "resolution": nx, "steps": r["steps"], "tau": r["tau"],
                "nu_expected": r["nu_expected"], "nu_measured": r["nu_measured"],
                "amplitude_init": r["amplitude_init"],
                "amplitude_final": r["amplitude_final"],
                "err_nu_pct": 100.0 * r["rel_err"],
            }
        )
        rows.append(
            row(f"nu from exp(-nu k^2 t), nx={nx}", r["nu_expected"], r["nu_measured"],
                100.0 * r["rel_err"], unit="lattice")
        )
    order = _order(per_res[0]["err_nu_pct"], per_res[1]["err_nu_pct"], 2.0)
    return {
        "item": "A3_shear_wave_viscosity",
        "category": "analytic",
        "title": ITEM_BY_KEY["A3_shear_wave_viscosity"].title,
        "anchor": {
            "kind": "analytic",
            "source": "u_y(x,t) = A0 cos(kx) exp(-nu k^2 t) with nu = (tau-1/2)/3",
            "citation": "atmospheric/Stokes shear-wave decay; lattice form Kruger et al. "
                        "(2017) sec. 5.2",
            "pointers": ["tests/test_lbm_verification.py::shear_wave_case (Step A)"],
        },
        "settings": {"resolutions": [p["resolution"] for p in per_res],
                     "steps": 100, "tau": 1.0, "amplitude": 0.01},
        "per_resolution": per_res,
        "rows": rows,
        "observed_order": order,
        "repro_cmd": ITEM_BY_KEY["A3_shear_wave_viscosity"].repro_cmd,
        "env": _env(),
    }


# --------------------------------------------------------------------------- #
# L1 / L2 -- lid-driven cavity vs Ghia
# --------------------------------------------------------------------------- #


def _cavity_item(key: str, re: float, runner: str, out_dir: Path) -> dict:
    raw = out_dir / f"{key}_raw.json"
    _run(
        [sys.executable, runner, "--nx", "128", "192", "--re", str(re),
         "--steps", "100000", "--device", "cpu", "--out", str(raw)],
        key,
    )
    data = json.loads(raw.read_text())
    grids = {k: v for k, v in data.items() if k not in ("convergence", "verified",
                                                       "verified_date", "case",
                                                       "description", "key_fix")}
    rows, per_res = [], []
    for nx_str, g in sorted(grids.items(), key=lambda kv: int(kv[0])):
        per_res.append({"resolution": int(nx_str), **g})
        rows.append(row(f"u(0.5,0.5)/U_lid vs Ghia, n={nx_str}", g["u_mid_ghia"],
                        g["u_mid"], 100.0 * abs(g["u_mid"] - g["u_mid_ghia"])
                        / abs(g["u_mid_ghia"])))
        rows.append(row(f"vortex x_c vs Ghia, n={nx_str}", g["vortex_ghia"][0],
                        g["vortex"][0], 100.0 * abs(g["vortex"][0] - g["vortex_ghia"][0])
                        / g["vortex_ghia"][0]))
        rows.append(row(f"vortex y_c vs Ghia, n={nx_str}", g["vortex_ghia"][1],
                        g["vortex"][1], 100.0 * abs(g["vortex"][1] - g["vortex_ghia"][1])
                        / g["vortex_ghia"][1]))
        rows.append(row(f"max centreline deviation (u,v) vs Ghia, n={nx_str}", 0.0,
                        g["max_abs_dev_pct"], g["max_abs_dev_pct"],
                        note="runner-reported max % over the two centreline profiles"))
    dev = [p["max_abs_dev_pct"] for p in per_res]
    return {
        "item": key,
        "category": "literature",
        "title": ITEM_BY_KEY[key].title,
        "anchor": {
            "kind": "literature",
            "source": "Ghia, Ghia & Shin (1982) tabulated centreline u/v profiles and main "
                      "vortex centre, Re=100/400; table transcribed in-repo at "
                      "src/tensorlbm/lid_driven_cavity.py (GHIA_RE100/GHIA_RE400)",
            "citation": "U. Ghia, K.N. Ghia, C.T. Shin, J. Comput. Phys. 48(3):387-411 (1982), "
                        "doi:10.1016/0021-9991(82)90058-4",
            "pointers": ["benchmarks/verified/cavity/re100/run.py (runner reused verbatim)",
                         f"benchmarks/verified/cavity/re{int(re)}/run.py"],
        },
        "settings": {"resolutions": [p["resolution"] for p in per_res], "re": re,
                     "u_lid": per_res[0]["u_lid"], "tau": [p["tau"] for p in per_res],
                     "steps": per_res[0]["steps"], "collision": "mrt",
                     "boundary": "pre-streaming half-way bounce-back + Zou/He moving lid"},
        "per_resolution": per_res,
        "rows": rows,
        "observed_order": _order(dev[0], dev[1], 192 / 128),
        "noise_floor_pct": {"last_resid": [p.get("last_resid") for p in per_res]},
        "repro_cmd": ITEM_BY_KEY[key].repro_cmd,
        "raw_artifact": str(raw.relative_to(REPO)),
        "env": _env(),
    }


def item_L1(out_dir: Path) -> dict:
    return _cavity_item("L1_cavity_ghia_re100", 100.0,
                        "benchmarks/verified/cavity/re100/run.py", out_dir)


def item_L2(out_dir: Path) -> dict:
    return _cavity_item("L2_cavity_ghia_re400", 400.0,
                        "benchmarks/verified/cavity/re400/run.py", out_dir)


# --------------------------------------------------------------------------- #
# L3 -- Sangani & Acrivos permeability
# --------------------------------------------------------------------------- #


def item_L3(out_dir: Path) -> dict:
    runner = "benchmarks/verified/permeability/run.py"
    cases_dir = out_dir / "L3_cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    ladder = {0.3: (26, 52, 104), 0.5: (20, 40, 80)}
    rows, per_res = [], []
    for phi, ds in ladder.items():
        for d in ds:
            _run(
                [sys.executable, runner, "case", "--phi", str(phi), "--d", str(d),
                 "--force-scale", "10", "--device", "cpu", "--out-dir", str(cases_dir)],
                f"L3 phi={phi} d={d}",
            )
            f = cases_dir / f"case_phi{phi}_d{d}_fs10.json"
            rec = json.loads(f.read_text())
            der, run = rec["derived"], rec["run"]
            per_res.append(
                {
                    "phi": phi, "resolution": d, "phi_actual": rec["geometry"]["phi_actual"],
                    "steps_run": run["steps_run"], "wall_s": run["wall_seconds"],
                    "k_sim": der["k_sim"], "k_ref": der["k_ref_nominal"],
                    "err_pct": der["err_pct"], "Re_cell": der["Re_cell_achieved"],
                    "Ma": der["Ma"], "f_table_at_phi": der["f_table_at_phi"],
                }
            )
            rows.append(row(f"k (Darcy) / d^2, phi={phi}, d={d}", der["k_ref_nominal"],
                            der["k_sim"], der["err_pct"],
                            note="ref from Sangani-Acrivos f table, k_ref=d^2/((1-phi) f)"))
    by_phi = {}
    for p in per_res:
        by_phi.setdefault(p["phi"], []).append(p["err_pct"])
    order = {str(phi): _order(e[0], e[-1], max(d for d in ladder[phi]) / min(ladder[phi]))
             for phi, e in by_phi.items()}
    return {
        "item": "L3_cylinder_array_sangani",
        "category": "literature",
        "title": ITEM_BY_KEY["L3_cylinder_array_sangani"].title,
        "anchor": {
            "kind": "literature",
            "source": "Sangani & Acrivos (1982) pressure-form drag table f(phi) for a square "
                      "array of cylinders; k_ref = d^2/((1-phi) f); table stored in-repo at "
                      "benchmarks/verified/permeability/run.py (SANGANI_TABLE)",
            "citation": "A.S. Sangani, A. Acrivos, Int. J. Multiphase Flow 8(2):193-206 (1982), "
                        "doi:10.1016/0301-9322(82)90029-5",
            "pointers": ["benchmarks/verified/permeability/run.py (runner reused verbatim)"],
        },
        "settings": {"resolutions": {str(phi): list(ds) for phi, ds in ladder.items()},
                     "force_scale": 10.0, "tau": 1.0, "collision": "bgk",
                     "protocol": "periodic unit cell, mask body force, steady-state detection"},
        "per_resolution": per_res,
        "rows": rows,
        "observed_order": order,
        "repro_cmd": ITEM_BY_KEY["L3_cylinder_array_sangani"].repro_cmd,
        "env": _env(),
    }


# --------------------------------------------------------------------------- #
# X1 -- cross solver
# --------------------------------------------------------------------------- #


def item_X1(out_dir: Path) -> dict:
    raw = out_dir / "X1_cross_solver_raw.json"
    code = (
        "import json;"
        "from pathlib import Path;"
        "from tensorlbm.cross_solver_verification.runner import BenchmarkConfig, run_benchmark;"
        f"cfg=BenchmarkConfig(mms_sizes_lbm=(16,32), mms_sizes_fvm=(16,32), mms_sizes_fem=(8,16),"
        "cavity_sizes_lbm=(64,128), cavity_sizes_fvm=(64,128), cavity_sizes_fem=(32,64),"
        "reynolds=(100,), reynolds_fvm=(100,), reynolds_fem=(100,),"
        f"cross_n=128, ledger_n=128, out_path=Path(r'{raw}')) ;"
        "run_benchmark(cfg)"
    )
    _run([sys.executable, "-c", code], "X1")
    rep = json.loads(raw.read_text())
    crit = rep["criteria"]
    cs = rep.get("cross_solver", {})
    pairs = cs.get("pairs", {})
    rows = []
    c1 = crit.get("C1_ghia_re100", {})
    for solver, v in (c1.get("per_solver") or {}).items():
        rows.append(
            row(f"centreline u max dev vs Ghia Re=100, {solver} n={v['n']}", 0.0,
                v["u_max_dev_pct"], v["u_max_dev_pct"], threshold_pct=3.0)
        )
    for name, val in (pairs.get("u_rel_l2") or {}).items():
        rows.append(row(f"relative L2 of u: {name} @n={cs.get('n')}", 0.0, 100.0 * val,
                        100.0 * val, threshold_pct=2.0,
                        note="cross-solver consistency, C2 threshold 2%"))
    for name, val in (pairs.get("v_rel_l2") or {}).items():
        rows.append(row(f"relative L2 of v: {name} @n={cs.get('n')}", 0.0, 100.0 * val,
                        100.0 * val, threshold_pct=2.0))
    mms = (rep.get("mms", {}).get("solvers") or {})
    orders = {}
    for solver, node in mms.items():
        o = node.get("order") if isinstance(node, dict) else None
        if isinstance(o, (int, float)):
            orders[solver] = float(o)
    # centreline profiles vs Ghia, straight out of the raw report (no retyping)
    profiles = {}
    for solver in ("lbm", "fvm", "fem"):
        for r in (rep.get("cavity", {}).get(solver) or []):
            if r.get("re") == 100 and "ghia" in r:
                profiles[f"{solver}_n{r['n']}"] = r["ghia"]

    # velocity fields for the contour panel: three independent solvers, same mesh
    fields_npz = out_dir / "X1_fields.npz"
    fields_meta = {}
    try:
        import numpy as np

        from tensorlbm.cross_solver_verification.fem import solve_taylor_hood
        from tensorlbm.cross_solver_verification.fvm import solve_psi_omega
        from tensorlbm.cross_solver_verification.lbm import run_lbm_cavity

        n_f, re_f, u_lid = 64, 100.0, 0.1
        r_l = run_lbm_cavity(n=n_f, re=re_f, u_lid=u_lid)
        r_f = solve_psi_omega(n_f, u_lid / re_f, mode="cavity", u_lid=u_lid)
        r_e = solve_taylor_hood(n_f, u_lid / re_f, bc="walls", u_lid=u_lid)
        np.savez_compressed(
            fields_npz,
            lbm_u=r_l.u, lbm_v=r_l.v, fvm_u=r_f.u, fvm_v=r_f.v,
            fem_u=r_e.u, fem_v=r_e.v, n=n_f, re=re_f, u_lid=u_lid,
        )
        fields_meta = {"n": n_f, "re": re_f, "u_lid": u_lid, "npz": str(fields_npz.relative_to(REPO)),
                       "shape": list(r_l.u.shape),
                       "lbm_steps": int(getattr(r_l, "steps", -1)),
                       "fvm_iterations": int(getattr(r_f, "iterations", -1)),
                       "fem_newton": int(getattr(r_e, "newton_iterations", -1))}
    except Exception as exc:  # noqa: BLE001 - disclosed, not hidden
        fields_meta = {"error": f"{type(exc).__name__}: {exc}"}

    return {
        "item": "X1_cross_solver_cavity",
        "category": "cross_solver",
        "title": ITEM_BY_KEY["X1_cross_solver_cavity"].title,
        "anchor": {
            "kind": "cross_solver+literature+analytic",
            "source": "three independent discretisations of the same lid-driven cavity: "
                      "lattice-BGK, cell-centred psi-omega finite volume (exact discrete "
                      "Poisson), Q2-Q1 Taylor-Hood finite elements; plus a manufactured "
                      "solution for each solver's observed order",
            "citation": "Ghia et al. (1982) for the cavity; Roache (1998) MMS verification; "
                        "package: src/tensorlbm/cross_solver_verification/",
            "pointers": ["src/tensorlbm/cross_solver_verification/{lbm,fvm,fem}.py",
                         "src/tensorlbm/cross_solver_verification/runner.py"],
        },
        "settings": {"cavity_sizes": {"lbm": [64, 128], "fvm": [64, 128], "fem": [32, 64]},
                     "re": 100, "cross_n": cs.get("n"), "mms_sizes_lbm": [16, 32]},
        "criteria": crit,
        "cross_pairs": pairs,
        "cavity_profiles": profiles,
        "fields": fields_meta,
        "mms_orders": orders,
        "ledger": cs.get("ledger", {}),
        "rows": rows,
        "repro_cmd": ITEM_BY_KEY["X1_cross_solver_cavity"].repro_cmd,
        "raw_artifact": str(raw.relative_to(REPO)),
        "env": _env(),
    }


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #

RUNNERS = {
    "A1_poiseuille_2d": item_A1,
    "A2_square_duct_3d": item_A2,
    "A3_shear_wave_viscosity": item_A3,
    "L1_cavity_ghia_re100": item_L1,
    "L2_cavity_ghia_re400": item_L2,
    "L3_cylinder_array_sangani": item_L3,
    "X1_cross_solver_cavity": item_X1,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--item", action="append", default=None,
                    help="item key (repeatable); default = all")
    ap.add_argument("--item-index", type=int, default=None,
                    help="1-based index into ITEMS (for shard submission)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "4")))
    args = ap.parse_args(argv)

    if args.list:
        for i, it in enumerate(ITEMS, 1):
            print(f"{i:2d}  {it.key:28s} [{it.category:12s}] {it.title}")
        return 0

    import torch

    torch.set_num_threads(args.threads)
    torch.set_default_dtype(torch.float64)

    if args.item_index is not None:
        keys = [ITEMS[args.item_index - 1].key]
    elif args.item:
        keys = args.item
    else:
        keys = [it.key for it in ITEMS]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    for key in keys:
        t0 = time.time()
        try:
            payload = RUNNERS[key](args.out_dir)
        except Exception as exc:  # noqa: BLE001 - recorded, never swallowed silently
            import traceback

            traceback.print_exc()
            failures.append({"item": key, "error": f"{type(exc).__name__}: {exc}"})
            continue
        payload["wall_seconds"] = round(time.time() - t0, 1)
        path = _write(args.out_dir, key, payload)
        n_pass = sum(1 for r in payload["rows"] if r["pass"])
        print(f"[{key}] {n_pass}/{len(payload['rows'])} rows <= 3%  -> {path}", flush=True)
    if failures:
        (args.out_dir / "FAILURES.json").write_text(json.dumps(failures, indent=1))
        print(f"FAILED items: {[f['item'] for f in failures]}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())