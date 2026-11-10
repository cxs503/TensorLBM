#!/usr/bin/env python
"""P233W2-N06 campaign runner: spherical (curved-wall) force method in D2Q9 LBM.

Jobs
----
``planar``    plane-channel momentum-exchange check (3 wall offsets)
``resweep``   cylinder drag coefficient vs Re = 20, 40, 100, 200
``grid``      Re = 100 grid-convergence study (3 meshes) + GCI
``domain``    Re = 100 blockage/domain sensitivity (2 domain heights)
``ibm``       Re = 100 diffuse-IBM cross-validation (AC5)
``all``       everything above, sequentially

Every job writes ``artifacts/spherical_force_method/<job>.json``; ``--job
summarize`` folds them into ``summary.json`` with the acceptance criteria and
the figures.

Usage
-----
    PYTHONPATH=src python scripts/run_spherical_force_campaign.py --job resweep
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.spherical_force_method import (  # noqa: E402
    run_cylinder_case,
    run_cylinder_case_ibm,
    run_planar_channel,
)

ART = ROOT / "artifacts" / "spherical_force_method"

# ---------------------------------------------------------------------------
# reference data.  Scoring anchors follow the repo REFERENCE_AUDIT discipline
# (docs/benchmarks/tutorials/cylinder_3d.md): a 2D free-stream *numerical*
# cluster, not the finite-span wind-tunnel value (Tritton 1959 ~1.54 at Re=40),
# which carries end effects and is not comparable with an infinite-span 2D
# simulation.
# ---------------------------------------------------------------------------
REFERENCES = {
    20: {
        "anchor": 2.045,
        "band": [2.042, 2.053],
        "sources": [
            "Dennis & Chang (1970) JFM 42:471, steady 2D solution Cd=2.045",
            "Fornberg (1985) steady 2D solution Cd=2.053",
        ],
    },
    40: {
        "anchor": 1.522,
        "band": [1.480, 1.522],
        "sources": [
            "Dennis & Chang (1970) JFM 42:471 Cd=1.522",
            "Fornberg (1985) Cd=1.498",
            "Takami & Keller (1969) Cd=1.48",
        ],
    },
    100: {
        "anchor": 1.35,
        "band": [1.32, 1.364],
        "sources": [
            "Dennis & Chang (1970) steady Cd=1.32",
            "Braza, Chassaing & Ha Minh (1986) time-mean Cd=1.364",
            "Liu, Zheng & Sung (1998) Cd=1.35",
        ],
    },
    200: {
        "anchor": 1.34,
        "band": [1.31, 1.40],
        "sources": [
            "Braza, Chassaing & Ha Minh (1986) time-mean Cd=1.40",
            "Liu, Zheng & Sung (1998) Cd=1.31",
        ],
    },
}

CASES = {
    "resweep": [
        dict(re=20, diameter=16, length_over_d=24.0, height_over_d=12.0),
        dict(re=40, diameter=16, length_over_d=24.0, height_over_d=12.0),
        dict(re=100, diameter=20, length_over_d=24.0, height_over_d=12.0),
        dict(re=200, diameter=20, length_over_d=24.0, height_over_d=12.0),
    ],
    "grid": [
        dict(re=100, diameter=12, length_over_d=24.0, height_over_d=12.0),
        dict(re=100, diameter=16, length_over_d=24.0, height_over_d=12.0),
        dict(re=100, diameter=24, length_over_d=24.0, height_over_d=12.0),
    ],
    "domain": [
        dict(re=100, diameter=16, length_over_d=24.0, height_over_d=12.0),
        dict(re=100, diameter=16, length_over_d=36.0, height_over_d=20.0),
    ],
    "ibm": [
        dict(re=100, diameter=16, length_over_d=24.0, height_over_d=12.0),
    ],
    # lean variants: same physics, ~4x cheaper (used when the batch queue is
    # congested; the blockage of the lean box is quantified by the domain job)
    "resweep_lean": [
        dict(re=20, diameter=12, length_over_d=18.0, height_over_d=9.0),
        dict(re=40, diameter=12, length_over_d=18.0, height_over_d=9.0),
        dict(re=100, diameter=12, length_over_d=18.0, height_over_d=9.0),
        dict(re=200, diameter=12, length_over_d=18.0, height_over_d=9.0),
    ],
    "grid_lean": [
        dict(re=100, diameter=10, length_over_d=18.0, height_over_d=9.0),
        dict(re=100, diameter=14, length_over_d=18.0, height_over_d=9.0),
        dict(re=100, diameter=18, length_over_d=18.0, height_over_d=9.0),
    ],
    "ibm_lean": [
        dict(re=100, diameter=12, length_over_d=18.0, height_over_d=9.0),
    ],
}


def _write(name: str, payload) -> Path:
    ART.mkdir(parents=True, exist_ok=True)
    p = ART / f"{name}.json"
    p.write_text(json.dumps(payload, indent=2, default=float))
    return p


def run_planar() -> None:
    rows = []
    for off in (0.0, 0.3, -0.35):
        t0 = time.time()
        r = run_planar_channel(offset=off, steps=12000)
        r["wall_seconds"] = time.time() - t0
        rows.append(r)
        print(
            f"[planar] off={off:+.2f} q=[{r['q_min']:.2f},{r['q_max']:.2f}] "
            f"prof_err={r['profile_rel_error'] * 100:.2f}% "
            f"Fbot={r['force_bottom']:.4e} ana={r['shear_analytic']:.4e} "
            f"err={r['rel_error_bottom'] * 100:.2f}% bal={r['force_balance_rel'] * 100:.2f}%",
            flush=True,
        )
    _write("planar", {"rows": rows})


def _strip_snapshot(r: dict) -> dict:
    out = {k: v for k, v in r.items() if k != "snapshot"}
    return out


def run_cases(name: str, ibm: bool = False) -> None:
    rows = []
    for cfg in CASES[name]:
        t0 = time.time()
        fn = run_cylinder_case_ibm if ibm else run_cylinder_case
        r = fn(verbose=True, **cfg)
        r = _strip_snapshot(r)
        r["wall_seconds"] = time.time() - t0
        ref = REFERENCES[int(cfg["re"])]
        r["cd_reference"] = ref["anchor"]
        r["cd_reference_band"] = ref["band"]
        r["cd_rel_error"] = abs(r["cd_mean"] - ref["anchor"]) / ref["anchor"]
        rows.append(r)
        snap = None
        if not ibm:
            snap = fn(verbose=False, **cfg).get("snapshot")  # not reused: see below
        print(
            f"[{name}] Re={cfg['re']} D={cfg['diameter']} "
            f"Re_meas={r['re_measured']:.1f} Cd={r['cd_mean']:.4f}±{r['cd_std']:.4f} "
            f"err={r['cd_rel_error'] * 100:.2f}% Cd(force)={r.get('cd_from_body_force', float('nan')):.4f} "
            f"St={r.get('strouhal', 0):.3f} steps={r['steps']} ({r['wall_seconds']:.0f}s)",
            flush=True,
        )
        del snap
    _write(name, {"rows": rows, "cases": CASES[name]})


def run_cylinder_with_snapshot(cfg: dict, tag: str) -> None:
    """Re-run one case keeping the velocity snapshot for the figures."""
    r = run_cylinder_case(verbose=True, snapshot=True, **cfg)
    snap = r.pop("snapshot")
    np.savez_compressed(
        ART / f"field_{tag}.npz",
        rho=snap["rho"],
        ux=snap["ux"],
        uy=snap["uy"],
        cx=snap["cx"],
        cy=snap["cy"],
        radius=snap["radius"],
    )
    _write(f"field_{tag}", {k: v for k, v in r.items()})


def summarize() -> None:
    def load(name):
        p = ART / f"{name}.json"
        return json.loads(p.read_text()) if p.exists() else None

    summary = {"references": REFERENCES, "cases": CASES, "jobs": {}}
    for job in ("planar", "resweep", "grid", "domain", "ibm"):
        summary["jobs"][job] = load(job)
    # fall back to the lean variants when the full-size job has not landed
    for full, lean in (("resweep", "resweep_lean"), ("grid", "grid_lean"), ("ibm", "ibm_lean")):
        if summary["jobs"][full] is None:
            alt = load(lean)
            if alt is not None:
                alt["variant"] = lean
                summary["jobs"][full] = alt

    ac = {}

    # ---- AC1: implementation tests are counted by pytest, recorded here ----
    ac["AC1_tests"] = {
        "pytest_file": "tests/test_spherical_force_method.py",
        "covers": ["planar", "cylinder", "sphere(3D link closure)", "momentum budget"],
    }

    # ---- AC2: cylinder drag vs reference ----------------------------------
    sweep = summary["jobs"]["resweep"]
    if sweep:
        rows = sweep["rows"]
        errs = {int(r["re_target"]): r["cd_rel_error"] for r in rows}
        summary["cd_table"] = [
            {
                "re": r["re_target"],
                "re_measured": r["re_measured"],
                "D": r["diameter"],
                "cd": r["cd_mean"],
                "cd_std": r["cd_std"],
                "cd_reference": r["cd_reference"],
                "cd_reference_band": r["cd_reference_band"],
                "rel_error": r["cd_rel_error"],
                "in_band": r["cd_reference_band"][0] <= r["cd_mean"] <= r["cd_reference_band"][1],
                "cd_from_body_force": r.get("cd_from_body_force"),
                "strouhal": r.get("strouhal"),
            }
            for r in rows
        ]
        ac["AC2"] = {
            "threshold": 0.10,
            "errors": errs,
            "max_error": max(errs.values()),
            "pass": max(errs.values()) <= 0.10,
        }
    else:
        ac["AC2"] = {"pass": None, "note": "resweep job missing"}

    # ---- AC4: grid convergence + GCI --------------------------------------
    grid = summary["jobs"]["grid"]
    if grid:
        rows = sorted(grid["rows"], key=lambda r: r["diameter"])
        cd = np.array([r["cd_mean"] for r in rows])
        h = 1.0 / np.array([r["diameter"] for r in rows], dtype=float)
        ref = rows[0]["cd_reference"]
        err = np.abs(cd - ref) / ref
        order = None
        gci_fine = None
        if cd.size >= 3:
            # observed order from the three finest grids (constant ratio assumed)
            r21 = (h[0] / h[1])
            e1 = abs(cd[1] - cd[0])
            e2 = abs(cd[2] - cd[1])
            if e1 > 1e-12 and e2 > 1e-12:
                order = math.log(e2 / e1) / math.log(h[1] / h[2]) if False else (
                    math.log(e1 / e2) / math.log(r21)
                )
            gci_fine = 1.25 * abs(cd[2] - cd[1]) / max(abs(cd[2]), 1e-12) / (
                max(r21 ** (order or 1.0) - 1.0, 1e-9)
            )
        monotone = bool(np.all(np.diff(err) < 0))
        ac["AC4"] = {
            "grids": [r["diameter"] for r in rows],
            "cd": cd.tolist(),
            "errors_vs_reference": err.tolist(),
            "monotone_error_decrease": monotone,
            "observed_order": order,
            "gci_finest_rel": gci_fine,
            "pass": bool(monotone),
        }
        summary["grid_table"] = [
            {
                "D": r["diameter"],
                "nx": r["nx"],
                "ny": r["ny"],
                "cd": r["cd_mean"],
                "cd_std": r["cd_std"],
                "rel_error": r["cd_rel_error"],
                "steps": r["steps"],
            }
            for r in rows
        ]
    else:
        ac["AC4"] = {"pass": None, "note": "grid job missing"}

    # ---- AC5: IBM cross-validation ---------------------------------------
    ibm = summary["jobs"]["ibm"]
    sweep_rows = {int(r["re_target"]): r for r in (sweep["rows"] if sweep else [])}
    if ibm and 100 in sweep_rows:
        r_ibm = ibm["rows"][0]
        r_spf = sweep_rows[100]
        d_rel = abs(r_ibm["cd_mean"] - r_spf["cd_mean"]) / r_spf["cd_mean"]
        ac["AC5"] = {
            "re": 100,
            "cd_spherical_force": r_spf["cd_mean"],
            "cd_ibm": r_ibm["cd_mean"],
            "rel_difference": d_rel,
            "threshold": 0.05,
            "pass": d_rel <= 0.05,
        }
        summary["ibm_table"] = [
            {"method": "spherical_force_method", "cd": r_spf["cd_mean"], "D": r_spf["diameter"]},
            {"method": "ibm_diffuse_4pt", "cd": r_ibm["cd_mean"], "D": r_ibm["diameter"]},
            {"method": "reference_2d_cluster", "cd": r_spf["cd_reference"]},
        ]
    else:
        ac["AC5"] = {"pass": None, "note": "ibm/sweep job missing"}

    domain = summary["jobs"]["domain"]
    if domain:
        rows = domain["rows"]
        summary["domain_table"] = [
            {"H_over_D": r["height_over_d"], "L_over_D": r["length_over_d"], "cd": r["cd_mean"]}
            for r in rows
        ]
        if len(rows) == 2:
            ac["domain_sensitivity"] = {
                "cd_rel_difference": abs(rows[1]["cd_mean"] - rows[0]["cd_mean"]) / rows[0]["cd_mean"]
            }

    planar = summary["jobs"]["planar"]
    if planar:
        rows = planar["rows"]
        ac["AC1_planar"] = {
            "offsets": [r["offset"] for r in rows],
            "profile_rel_errors": [r["profile_rel_error"] for r in rows],
            "force_rel_errors_bottom": [r["rel_error_bottom"] for r in rows],
            "force_balance_rel": [r["force_balance_rel"] for r in rows],
            "pass": bool(max(r["rel_error_bottom"] for r in rows) < 0.05),
        }

    summary["acceptance"] = ac
    _write("summary", summary)
    print(json.dumps({k: v.get("pass") for k, v in ac.items() if isinstance(v, dict)}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True,
                    choices=["planar", "resweep", "grid", "domain", "ibm", "all", "summarize",
                             "field", "fieldibm", "resweep_lean", "grid_lean", "ibm_lean"])
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        import torch

        torch.set_num_threads(args.threads)

    if args.job == "planar":
        run_planar()
    elif args.job == "resweep":
        run_cases("resweep")
    elif args.job == "resweep_lean":
        run_cases("resweep_lean")
    elif args.job == "grid_lean":
        run_cases("grid_lean")
    elif args.job == "ibm_lean":
        run_cases("ibm_lean", ibm=True)
    elif args.job == "grid":
        run_cases("grid")
    elif args.job == "domain":
        run_cases("domain")
    elif args.job == "ibm":
        run_cases("ibm", ibm=True)
    elif args.job == "field":
        run_cylinder_with_snapshot(
            dict(re=100, diameter=20, length_over_d=24.0, height_over_d=12.0), "re100_d20")
    elif args.job == "fieldibm":
        run_cylinder_with_snapshot(
            dict(re=20, diameter=20, length_over_d=24.0, height_over_d=12.0), "re20_d20")
    elif args.job == "all":
        run_planar()
        run_cases("resweep")
        run_cases("grid")
        run_cases("ibm", ibm=True)
        run_cases("domain")
        summarize()
    elif args.job == "summarize":
        summarize()


if __name__ == "__main__":
    main()