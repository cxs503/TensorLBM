"""P48-N01 curved-wall boundary calibration sweep runner.

Groups
    stokes    Stokes' second problem (C1)
    couette   plane Couette negative control (C6)
    annulus   Taylor-Couette annulus, exact torque (C3/C4)
    sphere    sphere drag vs Schiller-Naumann / experimental anchor (C2)

Usage
    python scripts/run_curved_boundary_calibration.py --group cheap --threads 8
    python scripts/run_curved_boundary_calibration.py --group sphere --diameters 12 16
    python scripts/run_curved_boundary_calibration.py --evaluate
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.curved_boundary_calibration import (  # noqa: E402
    BOUNDARY_FORMATS,
    CD_ANCHORS,
    convergence_order,
    run_annulus_couette,
    run_body_drag,
    run_plane_couette,
    run_stokes_second,
    schiller_naumann,
)

OUT = ROOT / "artifacts" / "curved_boundary_calibration"


def _dump(obj: dict[str, Any], name: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in obj.items() if k != "fields"}
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, default=float))
    return path


def run_stokes_group(quick: bool, closure: str) -> list[dict[str, Any]]:
    rows = []
    periods, warm = (2, 1) if quick else (3, 2)
    for fmt in BOUNDARY_FORMATS:
        res = run_stokes_second(
            u0=0.05, delta=4.0 if quick else 6.0, tau=0.65, format=fmt,
            periods=periods, warmup_periods=warm,
            samples_per_period=24 if quick else 48,
        )
        res["closure"] = closure
        rows.append(res)
        print(f"  stokes {fmt:18s} amp_err={res['max_amp_rel_err']*100:.3f}% "
              f"phase_err={res['max_phase_err_rad']*1e3:.2f}mrad "
              f"complex={res['max_complex_rel_err']*100:.3f}%", flush=True)
    return rows


def run_couette_group(closure: str) -> list[dict[str, Any]]:
    rows = []
    for (qb, qt) in ((0.5, 0.5), (0.3, 0.7), (0.7, 0.3)):
        for fmt in BOUNDARY_FORMATS:
            res = run_plane_couette(wall_speed=0.05, tau=0.65, format=fmt,
                                    ny=18, nx=2, nz=2, steps=4000,
                                    q_bottom=qb, q_top=qt)
            res["closure"] = closure
            rows.append(res)
            print(f"  couette q={qb}/{qt} {fmt:18s} F_err={res['force_rel_err']*100:.4f}% "
                  f"CV={res['cv_rel_err']*100:.4f}% prof={res['profile_max_rel_err']*100:.3f}%",
                  flush=True)
    return rows


def run_annulus_group(sizes: list[int], formats: list[str], closure: str,
                      steps: int, gap_ratios: list[float]) -> list[dict[str, Any]]:
    rows = []
    for r1 in sizes:
        for fmt in formats:
            res = run_annulus_couette(radius_inner=r1, radius_outer=2 * r1,
                                      format=fmt, steps=steps)
            res["closure"] = closure
            res["group"] = "size_ladder"
            rows.append(res)
            print(f"  annulus R1={r1:3d} {fmt:18s} T_me_err={res['torque_rel_err']*100:.3f}% "
                  f"T_cv_err={res['torque_cv_rel_err']*100:.3f}% "
                  f"prof={res['profile_max_rel_err']*100:.2f}% ({res['wall_seconds']:.0f}s)",
                  flush=True)
    for gr in gap_ratios:
        r1 = 16
        r2 = int(round(r1 * gr))
        for fmt in formats:
            res = run_annulus_couette(radius_inner=r1, radius_outer=r2,
                                      format=fmt, steps=steps)
            res["closure"] = closure
            res["group"] = "gap_ratio_family"
            rows.append(res)
            print(f"  annulus ratio={gr:.2f} {fmt:18s} T_me_err={res['torque_rel_err']*100:.3f}%"
                  f" T_cv_err={res['torque_cv_rel_err']*100:.3f}%", flush=True)
    return rows


def run_sphere_group(cases: list[dict[str, Any]], closure: str) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        t0 = time.perf_counter()
        res = run_body_drag(
            kind=case.get("kind", "sphere"),
            diameter=case["diameter"],
            reynolds=case["reynolds"],
            u_in=case.get("u_in", 0.1),
            format=case.get("format", "bouzidi_quadratic"),
            collision=case.get("collision", "regularized"),
            upstream=case.get("upstream", 2.0),
            downstream=case.get("downstream", 4.0),
            lateral=case.get("lateral", 2.0),
            steps_per_diameter=case.get("steps_per_diameter", 250.0),
            avg_fraction=case.get("avg_fraction", 0.4),
            cv_margin=case.get("cv_margin", 4),
        )
        res["closure"] = closure
        res["group"] = "sphere_factorial"
        rows.append(res)
        print(f"  {res['kind']} D={case['diameter']:2d} Re={case['reynolds']:6.1f} "
              f"{res['format']:18s} Cd={res['cd']:.4f} err={res['cd_rel_err']*100:+.2f}% "
              f"cd_std={res['cd_std']:.4f} CV={res['cd_cv_mean']} ({time.perf_counter()-t0:.0f}s)",
              flush=True)
    return rows


def evaluate(summary: dict[str, Any]) -> dict[str, Any]:
    """Re-derive the C1-C6 verdicts from the committed per-case numbers."""
    crit: dict[str, Any] = {}

    stokes = [r for r in summary.get("stokes", [])]
    if stokes:
        worst = max(stokes, key=lambda r: r["max_complex_rel_err"])
        crit["C1_stokes_second"] = {
            "value": {"max_amp_rel_err": worst["max_amp_rel_err"],
                      "max_phase_err_rad": worst["max_phase_err_rad"],
                      "max_complex_rel_err": worst["max_complex_rel_err"]},
            "threshold": 0.02,
            "passed": bool(worst["max_amp_rel_err"] <= 0.02 and worst["max_phase_err_rad"] <= 0.02
                           and worst["max_complex_rel_err"] <= 0.02),
            "worst_format": worst["format"],
            "caliber": "complex relative error of the Fourier amplitude at omega, sampled band y'<=3 delta",
        }

    sphere = summary.get("sphere", [])
    if sphere:
        per_re: dict[str, Any] = {}
        for re_val in sorted({float(r["reynolds"]) for r in sphere}):
            rows = [r for r in sphere if float(r["reynolds"]) == re_val]
            best = min(rows, key=lambda r: r["cd_rel_err"])
            per_re[f"{re_val:g}"] = {
                "anchor": CD_ANCHORS.get(re_val, schiller_naumann(re_val)),
                "cases": [{"diameter": r["diameter"], "format": r["format"],
                           "cd": r["cd"], "rel_err": r["cd_rel_err"]} for r in rows],
                "best": {"diameter": best["diameter"], "format": best["format"],
                         "cd": best["cd"], "rel_err": best["cd_rel_err"]},
            }
        passed = all(v["best"]["rel_err"] <= 0.05 for v in per_re.values())
        crit["C2_sphere_drag"] = {
            "value": {k: {"best_cd": v["best"]["cd"], "best_rel_err": v["best"]["rel_err"]}
                      for k, v in per_re.items()},
            "threshold": 0.05,
            "passed": bool(passed and len(per_re) >= 2),
            "detail": per_re,
            "caliber": "Cd from link-wise momentum exchange, window mean over the last 40% of steps",
        }

    ann = summary.get("annulus", [])
    ladder = [r for r in ann if r.get("group") == "size_ladder"
              and r["format"] == "bouzidi_quadratic"]
    if len(ladder) >= 2:
        ladder = sorted(ladder, key=lambda r: r["radius_inner"])
        errs = [r["torque_rel_err"] for r in ladder]
        sizes = [r["radius_inner"] for r in ladder]
        conv = convergence_order(sizes, errs)
        crit["C3_quadratic_order"] = {
            "value": conv,
            "threshold": 2.0,
            "passed": bool(conv["order"] >= 2.0),
            "errors": dict(zip([str(s) for s in sizes], errs)),
            "caliber": "observed order of the exact-torque error of the Taylor-Couette annulus "
                       "(bit-exact analytic reference)",
        }
    mt = [r for r in ann if r.get("me_vs_cv_rel_diff") is not None]
    if mt:
        worst = max(mt, key=lambda r: r["me_vs_cv_rel_diff"])
        crit["C4_me_vs_cv"] = {
            "value": {"max_rel_diff": worst["me_vs_cv_rel_diff"],
                      "worst_case": f"R1={worst['radius_inner']} {worst['format']}",
                      "torque_me": worst["torque_me"], "torque_cv": worst["torque_cv"]},
            "threshold": 0.01,
            "passed": bool(worst["me_vs_cv_rel_diff"] <= 0.01),
            "caliber": "momentum-exchange wall torque vs control-surface viscous torque",
        }

    couette = summary.get("couette", [])
    if couette:
        degenerate = [r for r in couette if r["q_bottom"] == 0.5 and r["q_top"] == 0.5]
        worst = max(degenerate, key=lambda r: r["force_rel_err"])
        spread = max(r["force_top_x"] for r in degenerate) - min(r["force_top_x"] for r in degenerate)
        crit["C6_plane_degenerate"] = {
            "value": {"max_force_rel_err": worst["force_rel_err"],
                      "format_spread": spread,
                      "profile_max_rel_err": max(r["profile_max_rel_err"] for r in degenerate)},
            "threshold": 1e-3,
            "passed": bool(worst["force_rel_err"] <= 1e-3),
            "caliber": "q=1/2 plane Couette wall shear mu U / H; all three formats must agree",
        }
    return crit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="cheap",
                    choices=["cheap", "stokes", "couette", "annulus", "sphere", "all", "evaluate"])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--closure", default="proxy")
    ap.add_argument("--sizes", type=int, nargs="*", default=[8, 12, 16])
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--gap-ratios", type=float, nargs="*", default=[1.5, 3.0])
    ap.add_argument("--diameters", type=int, nargs="*", default=[12, 16])
    ap.add_argument("--reynolds", type=float, nargs="*", default=[100.0])
    ap.add_argument("--steps-per-diameter", type=float, default=250.0)
    ap.add_argument("--lateral", type=float, default=2.0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {}
    t0 = time.perf_counter()
    if args.group in ("stokes", "cheap", "all"):
        summary["stokes"] = run_stokes_group(args.quick, args.closure)
    if args.group in ("couette", "cheap", "all"):
        summary["couette"] = run_couette_group(args.closure)
    if args.group in ("annulus", "cheap", "all"):
        summary["annulus"] = run_annulus_group(
            args.sizes, ["bouzidi_quadratic"] if args.quick else list(BOUNDARY_FORMATS),
            args.closure, args.steps, args.gap_ratios)
    if args.group in ("sphere", "all"):
        cases = []
        for d in args.diameters:
            for re_val in args.reynolds:
                cases.append({"diameter": d, "reynolds": re_val,
                              "format": "bouzidi_quadratic",
                              "steps_per_diameter": args.steps_per_diameter,
                              "lateral": args.lateral})
        summary["sphere"] = run_sphere_group(cases, args.closure)
    summary["meta"] = {
        "closure": args.closure,
        "group": args.group,
        "wall_seconds": time.perf_counter() - t0,
        "torch_threads": args.threads,
    }
    stamp = time.strftime("%Y%m%dT%H%M%S")
    name = f"sweep_{args.group}_{args.closure}_{stamp}"
    _dump(summary, name, out_dir)
    print(f"wrote {out_dir / (name + '.json')} in {summary['meta']['wall_seconds']:.0f}s")


if __name__ == "__main__":
    main()
