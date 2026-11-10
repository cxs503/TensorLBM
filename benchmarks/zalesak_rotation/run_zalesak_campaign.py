#!/usr/bin/env python3
"""P189-N06 -- Zalesak slotted-cylinder rotation campaign for TensorLBM.

Runs the pre-declared experimental design of the campaign node
``P189-N06`` and writes machine-readable artifacts:

* ``stage=arms``    -- 2-D operator comparison at N = 128 (control + treatments)
* ``stage=grid``    -- grid-convergence ladder N = 64 / 128 / 256 (selected arm)
* ``stage=three-d`` -- D3Q19 3-D runs (slab-slot and cylindrical-slot) and the
                       cross-section comparison against D2Q9
* ``stage=all``     -- everything above

Every stage appends to ``artifacts/zalesak_p189_n06/zalesak_results.json`` and
never rewrites earlier stages, so partial runs are safe.

Usage
-----
    python benchmarks/zalesak_rotation/run_zalesak_campaign.py --stage arms
    python benchmarks/zalesak_rotation/run_zalesak_campaign.py --stage grid --n 256
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.free_surface_zalesak import (  # noqa: E402
    analytic_volume_2d,
    cross_section_index,
    convergence_order,
    discrete_divergence_max,
    gci,
    interface_thickness,
    liquid_volume,
    rigid_rotation_field,
    run_zalesak_case,
    shape_overlap,
    slotted_cylinder_extruded_phi_3d,
    slotted_cylinder_phi_2d,
    slotted_sphere_phi_3d,
    cross_section,
)

ART = ROOT / "artifacts" / "zalesak_p189_n06"

# Pre-declared acceptance thresholds (campaign node P189-N06 metadata).
THRESHOLDS = {
    "volume_rel_change_max": 5.0e-3,      # (a) |dV|/V < 0.5 % at N = 128
    "interface_thickness_max_cells": 3.0,  # (b) 10-90 interface thickness < 3 cells
    "overlap_min": 0.95,                   # (c) overlap with the initial shape > 95 %
    "convergence_order_min": 0.5,          # (d) observed order of the shape error
    "cross_section_max_dev": 0.05,         # (e) D3Q19 vs D2Q9 cross-section < 5 %
    "divergence_max": 1.0e-10,             # (f) |div u| < 1e-10
}

# Pre-declared arm table (declared before looking at any result).
ARMS = [
    {"name": "A0_upwind_no_comp", "scheme": "upwind", "limiter": "vanleer", "c_comp": 0.0},
    {"name": "A1_muscl_vanleer_no_comp", "scheme": "muscl", "limiter": "vanleer", "c_comp": 0.0},
    {"name": "A2_muscl_vanleer_c050", "scheme": "muscl", "limiter": "vanleer", "c_comp": 0.5},
    {"name": "A3_muscl_vanleer_c100", "scheme": "muscl", "limiter": "vanleer", "c_comp": 1.0},
    {"name": "A4_muscl_vanleer_c150", "scheme": "muscl", "limiter": "vanleer", "c_comp": 1.5},
    {"name": "A5_muscl_superbee_c100", "scheme": "muscl", "limiter": "superbee", "c_comp": 1.0},
]
SELECTED_ARM = "A5_muscl_superbee_c100"  # chosen by the pre-declared rule (min L1 at N=128)


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #


def _git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _load() -> dict:
    path = ART / "zalesak_results.json"
    if path.exists():
        return json.loads(path.read_text())
    return {
        "campaign": "P189-N06",
        "repo": "TensorLBM",
        "git_rev": _git_rev(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "dtype": "float64",
        "thresholds": THRESHOLDS,
        "arms_table": ARMS,
        "selected_arm": SELECTED_ARM,
        "stages": {},
    }


def _save(payload: dict) -> None:
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "zalesak_results.json").write_text(json.dumps(payload, indent=2))


def _criterion_flags(metrics: dict) -> dict:
    """Machine-checked acceptance flags for one 2-D N = 128 run."""
    return {
        "a_volume_rel_change": abs(metrics["volume_rel_change"]) < THRESHOLDS["volume_rel_change_max"],
        "b_interface_thickness": metrics["thickness_mean"] < THRESHOLDS["interface_thickness_max_cells"],
        "c_overlap": metrics["overlap_recall"] > THRESHOLDS["overlap_min"],
        "c_overlap_iou": metrics["overlap_iou"] > THRESHOLDS["overlap_min"],
        "f_divergence": metrics["div_max"] < THRESHOLDS["divergence_max"],
    }


def _slim(metrics: dict) -> dict:
    keep = (
        "n", "dim", "lattice", "scheme", "limiter", "c_comp", "steps", "u_max_lattice",
        "div_max", "volume_initial", "volume_final", "volume_rel_change",
        "thickness_initial_mean", "thickness_mean", "thickness_p95", "thickness_max",
        "overlap", "overlap_recall", "overlap_iou", "hausdorff_cells", "l1_error",
        "boundedness_violations", "phi_min", "phi_max", "wall_s",
    )
    return {k: metrics[k] for k in keep if k in metrics}


def run_arm(arm: dict, n: int, u_max_lat: float = 0.2, sample_every: int | None = None) -> dict:
    t0 = time.time()
    res = run_zalesak_case(
        n, dim=2, u_max_lat=u_max_lat, c_comp=arm["c_comp"],
        scheme=arm["scheme"], limiter=arm["limiter"], sample_every=sample_every,
    )
    metrics = dict(res["metrics"])
    metrics["wall_s"] = round(time.time() - t0, 2)
    metrics["arm"] = arm["name"]
    metrics.update(_criterion_flags(metrics))
    return {"metrics": metrics, "history": res["history"], "phi_final": res["phi_final"],
            "phi_initial": res["phi_initial"]}


# --------------------------------------------------------------------------- #
# stages                                                                      #
# --------------------------------------------------------------------------- #


def stage_arms(n: int = 128, u_max_lat: float = 0.2, only: list[str] | None = None) -> dict:
    payload = _load()
    rows = dict(payload.get("stages", {}).get("arms", {}).get("arms", {}))
    for arm in ARMS:
        if only and arm["name"] not in only:
            continue
        out = run_arm(arm, n, u_max_lat)
        rows[arm["name"]] = {"metrics": _slim(out["metrics"]), "history": out["history"]}
        m = out["metrics"]
        print(
            f"[arms] {arm['name']:<24} n={n} Vrel={m['volume_rel_change']:+.2e} "
            f"th={m['thickness_mean']:.2f} ov_recall={m['overlap_recall']:.4f} "
            f"iou={m['overlap_iou']:.4f} L1={m['l1_error']:.5f} "
            f"hd={m['hausdorff_cells']:.1f} {m['wall_s']}s", flush=True,
        )
    payload["stages"]["arms"] = {
        "n": n, "u_max_lat": u_max_lat, "arms": rows,
        "complete": len(rows) == len(ARMS),
    }
    _save(payload)
    return rows


def stage_grid(ns: list[int], u_max_lat: float = 0.2, arm_name: str | None = None) -> dict:
    payload = _load()
    arm = next(a for a in ARMS if a["name"] == (arm_name or SELECTED_ARM))
    grid_all = payload.setdefault("stages", {}).setdefault("grid", {})
    rows = dict(grid_all.get(arm["name"], {}).get("runs", {}))
    for n in ns:
        if str(n) in rows:
            continue
        out = run_arm(arm, n, u_max_lat)
        rows[str(n)] = {"metrics": _slim(out["metrics"]), "history": out["history"]}
        m = out["metrics"]
        print(
            f"[grid] n={n:<4} steps={m['steps']} Vrel={m['volume_rel_change']:+.2e} "
            f"th={m['thickness_mean']:.2f} ov_recall={m['overlap_recall']:.4f} "
            f"iou={m['overlap_iou']:.4f} L1={m['l1_error']:.6f} {m['wall_s']}s", flush=True,
        )
    # observed order + GCI from the L1 shape error (pre-declared post-processing)
    order = {}
    keys = sorted(rows, key=int)
    for k0, k1 in zip(keys[:-1], keys[1:]):
        e0 = rows[k0]["metrics"]["l1_error"]
        e1 = rows[k1]["metrics"]["l1_error"]
        p = convergence_order(e0, e1)
        order[f"{k0}->{k1}"] = {
            "p_l1": p,
            "gci_fine": gci(e0, e1, p) if math.isfinite(p) else float("nan"),
        }
    vol_order = {}
    for k0, k1 in zip(keys[:-1], keys[1:]):
        v0 = abs(rows[k0]["metrics"]["volume_rel_change"])
        v1 = abs(rows[k1]["metrics"]["volume_rel_change"])
        vol_order[f"{k0}->{k1}"] = {"p_volume": convergence_order(v0, v1)}
    grid_all[arm["name"]] = {
        "u_max_lat": u_max_lat, "arm": arm["name"], "runs": rows,
        "convergence": order, "volume_convergence": vol_order,
    }
    payload["stages"]["grid"] = grid_all
    _save(payload)
    return rows


def stage_three_d(n: int = 64, u_max_lat: float = 0.2) -> dict:
    """D3Q19 runs: kernel equivalence (extruded body), slotted sphere, bore.

    Three 3-D configurations are exercised at the same grid and step count as a
    matched 2-D D2Q9 run:

    * ``kernel_equivalence`` -- the 2-D body *extruded through y* (exactly
      y-invariant): the D3Q19 run must reproduce the D2Q9 run on the
      cross-section to round-off.  This is the rigorous 3-D-vs-2-D kernel test
      (criterion (e)).
    * ``slotted_sphere`` -- the standard 3-D Zalesak body (sphere with the
      planar-faced slot): its cross-section at the sphere centre equals the 2-D
      initial condition, but the interface is a *sphere*, so the (geometry
      dependent) interface-compression flux acquires an out-of-plane component
      and the cross-section drifts from the 2-D solution.  Reported with the
      measured deviation plus the c_comp = 0 reference value.
    * ``slotted_sphere_bore`` -- same sphere with a drilled bore instead of the
      planar cut (a genuinely three-dimensional slot).
    """
    payload = _load()
    arm = next(a for a in ARMS if a["name"] == SELECTED_ARM)
    kw = {"u_max_lat": u_max_lat, "c_comp": arm["c_comp"], "scheme": arm["scheme"],
          "limiter": arm["limiter"], "sample_every": None}

    # ---- 1. kernel equivalence: extruded body (exact 2-D/3-D equivalence) --
    phi3x = slotted_cylinder_extruded_phi_3d(n)
    sec0 = cross_section(phi3x)
    run2d_x = run_zalesak_case(n, dim=2, phi_init=sec0, **kw)
    run3d_x = run_zalesak_case(n, dim=3, phi_init=phi3x, **kw)
    sec_x = cross_section(run3d_x["phi_final"])
    ref_x = run2d_x["phi_final"]
    kern = {
        "n": n,
        "max_abs_diff": float((sec_x - ref_x).abs().max().item()),
        "mean_abs_diff": float((sec_x - ref_x).abs().mean().item()),
        "cross_section_dev": float((sec_x - ref_x).abs().mean().item() / max(ref_x.mean().item(), 1e-30)),
        "volume_rel_change_3d": run3d_x["metrics"]["volume_rel_change"],
        "volume_rel_change_2d": run2d_x["metrics"]["volume_rel_change"],
        "l1_3d_vs_initial": run3d_x["metrics"]["l1_error"],
        "l1_2d_vs_initial": run2d_x["metrics"]["l1_error"],
        "criterion_pass": bool(
            float((sec_x - ref_x).abs().mean().item() / max(ref_x.mean().item(), 1e-30))
            < THRESHOLDS["cross_section_max_dev"]
        ),
    }

    # ---- 2. slotted sphere (standard 3-D Zalesak body) --------------------
    phi3 = slotted_sphere_phi_3d(n)
    m3 = dict(run_zalesak_case(n, dim=3, **kw)["metrics"])
    m3["arm"] = SELECTED_ARM
    # matched 2-D run started from the sphere's own mid-plane slice
    phi3_run = run_zalesak_case(n, dim=3, **kw)
    run2d = run_zalesak_case(n, dim=2, phi_init=cross_section(phi3), **kw)
    sec = cross_section(phi3_run["phi_final"])
    ref = run2d["phi_final"]
    # c_comp = 0 reference: the sharpening-free transport must agree exactly
    kw0 = dict(kw, c_comp=0.0)
    run3d_0 = run_zalesak_case(n, dim=3, phi_init=phi3, **kw0)
    run2d_0 = run_zalesak_case(n, dim=2, phi_init=cross_section(phi3), **kw0)
    sec_0 = cross_section(run3d_0["phi_final"])
    ref_0 = run2d_0["phi_final"]
    areas = [float(phi3[:, k].mean().item()) for k in range(n)]
    big = max(areas)
    contrast = big / max(sorted(a for a in areas if a > 0.01 * big)[0], 1e-30)
    cross = {
        "n": n,
        "y_index": cross_section_index(n),
        "max_abs_diff": float((sec - ref).abs().max().item()),
        "mean_abs_diff": float((sec - ref).abs().mean().item()),
        "cross_section_dev": float((sec - ref).abs().mean().item() / max(ref.mean().item(), 1e-30)),
        "overlap_recall_3d_vs_2d": shape_overlap(sec, ref, mode="recall"),
        "overlap_iou_3d_vs_2d": shape_overlap(sec, ref, mode="jaccard"),
        "area_rel_diff": float(
            (liquid_volume(sec) - liquid_volume(ref)) / max(liquid_volume(ref), 1e-30)),
        "dev_with_compression_off": float(
            (sec_0 - ref_0).abs().mean().item() / max(ref_0.mean().item(), 1e-30)),
        "max_abs_diff_compression_off": float((sec_0 - ref_0).abs().max().item()),
        "slice_area_max_over_min_1pct": contrast,
        "thickness_3d_cross_section": interface_thickness(sec)["mean"],
        "thickness_2d": interface_thickness(ref)["mean"],
        "criterion_pass": bool(
            float((sec - ref).abs().mean().item() / max(ref.mean().item(), 1e-30))
            < THRESHOLDS["cross_section_max_dev"]
        ),
    }

    # ---- 3. genuine 3-D slot: drilled bore --------------------------------
    phi3b = slotted_sphere_phi_3d(n, cylindrical_slot=True)
    run3b = run_zalesak_case(n, dim=3, phi_init=phi3b, **kw)
    m3b = dict(run3b["metrics"])
    m3b["arm"] = SELECTED_ARM

    payload["stages"]["three_d"] = {
        "n": n,
        "u_max_lat": u_max_lat,
        "arm": SELECTED_ARM,
        "kernel_equivalence": kern,
        "slotted_sphere": {**_slim(m3), "cross_section_vs_2d": cross},
        "slotted_sphere_bore": _slim(m3b),
    }
    _save(payload)
    print(f"[3d] n={n} kernel-equiv dev={kern['cross_section_dev']:.2e} (max|d|={kern['max_abs_diff']:.2e}) | "
          f"sphere dev={cross['cross_section_dev']:.3f} (c=0: {cross['dev_with_compression_off']:.2e}) "
          f"ov_recall={m3['overlap_recall']:.4f} th={m3['thickness_mean']:.2f} | "
          f"bore ov_recall={m3b['overlap_recall']:.4f} Vrel={m3b['volume_rel_change']:+.1e}", flush=True)
    return payload["stages"]["three_d"]


def stage_reference() -> dict:
    """Geometry verification: discrete volume vs analytic, centroid, thickness."""
    payload = _load()
    rows = {}
    v_ref = analytic_volume_2d()
    for n in (32, 64, 128, 256):
        phi = slotted_cylinder_phi_2d(n)
        x = (torch.arange(n, dtype=torch.float64) + 0.5) / n
        yy, xx = torch.meshgrid(x, x, indexing="ij")
        v = liquid_volume(phi)
        cx = float((phi * xx).sum() / phi.sum())
        cy = float((phi * yy).sum() / phi.sum())
        rows[str(n)] = {
            "volume_discrete": v,
            "volume_analytic": v_ref,
            "volume_rel_error": (v - v_ref) / v_ref,
            "centroid": [cx, cy],
            "thickness_initial": interface_thickness(phi),
            "n_cells_on_interface": int(
                ((phi > 0.0) & (phi < 1.0)).sum().item()
            ),
        }
    # divergence-free check for the analytic field at every test resolution
    divs = {}
    for n in (64, 128, 256):
        u = rigid_rotation_field(n, dim=2, omega=2.0 * math.pi * n / max(8, round(2 * math.pi * 0.3867 * n / 0.2)))
        divs[str(n)] = discrete_divergence_max(u)
        u3 = rigid_rotation_field(n, dim=3, omega=2.0 * math.pi * n / max(8, round(2 * math.pi * 0.3867 * n / 0.2)))
        divs[f"{n}_3d"] = discrete_divergence_max(u3)
    payload["stages"]["reference"] = {
        "analytic_volume_2d": v_ref,
        "geometry": rows,
        "divergence": divs,
        "divergence_max": max(divs.values()),
    }
    _save(payload)
    return payload["stages"]["reference"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["arms", "grid", "three-d", "reference", "all"])
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--u-max-lat", type=float, default=0.2)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--arm", default=None, help="comma-separated subset of arm names")
    ap.add_argument("--ns", default=None, help="comma-separated grid sizes")
    ap.add_argument("--grid-arm", default=None, help="arm name for the grid stage")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    if args.stage in ("reference", "all"):
        stage_reference()
    if args.stage in ("arms", "all"):
        only = args.arm.split(",") if args.arm else None
        stage_arms(n=args.n or 128, u_max_lat=args.u_max_lat, only=only)
    if args.stage in ("grid", "all"):
        ns = [int(v) for v in args.ns.split(",")] if args.ns else (
            [args.n] if args.n else [64, 128, 256])
        stage_grid(ns=ns, u_max_lat=args.u_max_lat, arm_name=args.grid_arm)
    if args.stage in ("three-d", "all"):
        stage_three_d(n=args.n or 64, u_max_lat=args.u_max_lat)


if __name__ == "__main__":
    main()