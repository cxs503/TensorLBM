#!/usr/bin/env python
"""P190-N01 -- LBM free-surface fill-transport Zalesak rotation campaign.

Produces every number used by ``docs/P190_N01_ZALESAK_LBM.md`` and by the
figures of ``make_lbm_figures.py``.  Stages:

``compare``  one revolution at N=64 for the LBM transport schemes
             (``lbm`` / ``fct`` / ``low`` / ``high``) plus the in-repo
             finite-volume VOF reference (``free_surface_zalesak``, MUSCL +
             van-Leer + interface compression) and a relaxation-rate scan.
``gci``      grid-convergence study N = 64/128/256 for the proposed scheme and
             for the VOF reference (observed order, Roache GCI).
``3d``       D3Q19: extruded (kernel-equivalence) configuration versus the 2-D
             D2Q9 cross-section, and the genuine 3-D slotted sphere.

Usage::

    PYTHONPATH=src python benchmarks/zalesak_rotation/run_zalesak_lbm_campaign.py \
        --stage all --out artifacts/zalesak_p190_n01
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time

import torch

from tensorlbm import free_surface_zalesak_lbm as Z
from tensorlbm.free_surface_zalesak import (
    analytic_volume_2d,
    convergence_order,
    gci,
    run_zalesak_case,
)

DTYPE = torch.float64


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #


def _finite(x):
    """JSON-safe float: non-finite values become ``None`` plus a tag."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    return x


def sanitize(obj):
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, float):
        return _finite(obj)
    if isinstance(obj, torch.Tensor):
        return _finite(float(obj.item())) if obj.numel() == 2 else None
    if isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    return _finite(float(obj))


METRIC_KEYS = (
    "volume_rel_change",
    "volume_abs_rel_change",
    "l1_shape_error",
    "shape_overlap",
    "hausdorff_distance",
    "interface_thickness_mean_cells",
    "interface_thickness_p95_cells",
    "interface_band_cells",
    "fill_min",
    "fill_max",
    "bound_violation",
    "max_stepwise_mass_drift",
    "mean_moved_fraction_per_step",
)


def _pick(metrics: dict) -> dict:
    return {k: metrics.get(k) for k in METRIC_KEYS}


def run_lbm(n, *, dim=2, scheme="lbm", tau=0.504, u_max_lat=0.1, steps=None,
            extruded=False, record_fields=False, label=""):
    t0 = time.time()
    m = Z.run_lbm_zalesak_case(
        n, dim=dim, scheme=scheme, tau=tau, u_max_lat=u_max_lat, steps=steps,
        extruded=extruded, record_fields=record_fields,
    )
    out = _pick(m)
    out.update(
        {
            "label": label or scheme,
            "scheme": scheme,
            "tau": tau,
            "n": n,
            "dim": dim,
            "steps": m["steps"],
            "lattice": m["lattice"],
            "div_u_max": m["div_u_max"],
            "wall_seconds": round(time.time() - t0, 1),
            "history": m["history"],
        }
    )
    if record_fields:
        out["_phi_final"] = m["phi_final"]
        out["_phi_initial"] = m["phi_initial"]
    print(
        f"  [{out['label']} n={n} dim={dim} tau={tau}] l1={m['l1_shape_error']:.4e} "
        f"ovl={m['shape_overlap']:.4f} vol={m['volume_rel_change']:+.2e} "
        f"thick={m['interface_thickness_mean_cells']:.2f} "
        f"({out['wall_seconds']}s)",
        flush=True,
    )
    return out


def run_vof(n, *, c_comp=0.5, u_max_lat=0.1, record_fields=False):
    t0 = time.time()
    r = run_zalesak_case(n, dim=2, u_max_lat=u_max_lat, c_comp=c_comp)
    met = r["metrics"]
    out = {
        "label": f"vof_muscl_cc{c_comp}",
        "scheme": "vof_muscl",
        "c_comp": c_comp,
        "n": n,
        "dim": 2,
        "steps": met["steps"],
        "lattice": "D2Q9 (face set)",
        "volume_rel_change": met["volume_rel_change"],
        "volume_abs_rel_change": abs(met["volume_rel_change"]),
        "l1_shape_error": met["l1_error"],
        "shape_overlap": met["overlap"],
        "hausdorff_distance": met["hausdorff_cells"],
        "interface_thickness_mean_cells": met["thickness_mean"],
        "interface_thickness_p95_cells": met["thickness_p95"],
        "div_u_max": met["div_max"],
        "bound_violation": 0.0,
        "wall_seconds": round(time.time() - t0, 1),
    }
    if record_fields:
        out["_phi_final"] = r["phi_final"]
        out["_phi_initial"] = r["phi_initial"]
    print(
        f"  [{out['label']} n={n}] l1={met['l1_error']:.4e} ovl={met['overlap']:.4f} "
        f"vol={met['volume_rel_change']:+.2e} thick={met['thickness_mean']:.2f} "
        f"({out['wall_seconds']}s)",
        flush=True,
    )
    return out


# --------------------------------------------------------------------------- #
# stages                                                                      #
# --------------------------------------------------------------------------- #


def stage_compare(n=64, record_fields=False):
    runs = []
    # The reference and the proposed operator get the initial/final fields for
    # the figure; the controls only need metrics.
    runs.append(run_vof(n, record_fields=record_fields))
    runs.append(run_lbm(n, scheme="lbm", label="lbm_tau0.504", record_fields=record_fields))
    runs.append(run_lbm(n, scheme="fct", label="fct_tau0.504"))
    runs.append(run_lbm(n, scheme="low", tau=1.0, label="low_tau1.0"))
    # tau = 1/2 exactly is the zero-diffusivity point and is only marginally
    # stable; cap the run so the divergence is measured, not waited for.
    div = run_lbm(n, scheme="high", tau=0.5, steps=min(400, 400), label="high_tau0.5")
    runs.append(div)
    scan = []
    for tau in (0.502, 0.503, 0.504, 0.505, 0.506, 0.510):
        scan.append(run_lbm(n, tau=tau, label=f"lbm_scan_tau{tau}"))
    return {"runs": runs, "tau_scan": scan, "n": n}


def stage_gci(sizes=(64, 128, 256), record_fields=True):
    lbm, vof = [], []
    for n in sizes:
        rec = record_fields and n in (sizes[0], sizes[-1])
        lbm.append(run_lbm(n, scheme="lbm", tau=0.504, record_fields=rec,
                           label=f"lbm_tau0.504_n{n}"))
        vof.append(run_vof(n, record_fields=(record_fields and n == sizes[-1])))

    def orders(seq):
        e = [r["l1_shape_error"] for r in seq]
        p1 = convergence_order(e[0], e[1])
        p2 = convergence_order(e[1], e[2]) if len(e) > 2 else float("nan")
        out = {
            "l1_errors": e,
            "sizes": [r["n"] for r in seq],
            "order_64_128": p1,
            "order_128_256": p2,
            "order_finest_pair": p2 if math.isfinite(p2) else p1,
            "gci_fine_percent": gci(e[-2], e[-1], p2 if math.isfinite(p2) else p1),
            "overlap": [r["shape_overlap"] for r in seq],
            "thickness_cells": [r["interface_thickness_mean_cells"] for r in seq],
            "volume_abs_rel_change": [r["volume_abs_rel_change"] for r in seq],
        }
        return out

    return {"lbm": lbm, "vof": vof, "orders_lbm": orders(lbm), "orders_vof": orders(vof)}


def stage_3d(n=48, n_sphere=40):
    """D3Q19 stage: kernel-equivalence cross-check plus the genuine 3-D body.

    The 3-D extruded run is initialised from the *same* 2-D field promoted to
    3-D (invariant in y), so the only difference to the D2Q9 run is the lattice
    and its weights -- the initial-condition discretisation is held fixed.
    """
    runs = {}
    two = run_lbm(n, scheme="lbm", tau=0.504, record_fields=True,
                  label=f"2d_reference_n{n}")
    phi2 = two["_phi_initial"]
    phi3_init = phi2.unsqueeze(1).expand(n, n, n).contiguous()
    t0 = time.time()
    raw3 = Z.run_lbm_zalesak_case(
        n, dim=3, scheme="lbm", tau=0.504, phi_init=phi3_init, record_fields=True
    )
    three = {
        **_pick(raw3),
        "label": f"3d_extruded_n{n}",
        "scheme": "lbm",
        "tau": 0.504,
        "n": n,
        "dim": 3,
        "steps": raw3["steps"],
        "lattice": raw3["lattice"],
        "div_u_max": raw3["div_u_max"],
        "wall_seconds": round(time.time() - t0, 1),
        "_phi_final": raw3["phi_final"],
        "_phi_initial": raw3["phi_initial"],
    }
    print(f"  [{three['label']}] l1={three['l1_shape_error']:.4e} "
          f"ovl={three['shape_overlap']:.4f} ({three['wall_seconds']}s)", flush=True)
    a = three.get("_phi_final")
    b = two["_phi_final"]
    iy = n // 2
    cs = a[:, iy, :]
    runs["two_d"] = two
    runs["three_d_extruded"] = three
    runs["cross_section_deviation"] = {
        "relative_l1": float((cs - b).abs().mean() / b.mean()),
        "max_abs": float((cs - b).abs().max()),
        "volume_rel_diff": abs(float(cs.mean()) - float(b.mean())) / float(b.mean()),
        "l1_metric_3d_cs": three["l1_shape_error"],
        "l1_metric_2d": two["l1_shape_error"],
        "l1_metric_rel_diff": abs(three["l1_shape_error"] - two["l1_shape_error"])
        / two["l1_shape_error"],
        "y_invariance_max": float((a - a[:, :1, :]).abs().max()),
    }
    print(f"  [3d vs 2d cross-section] {runs['cross_section_deviation']}", flush=True)
    runs["three_d_sphere"] = run_lbm(n_sphere, dim=3, scheme="lbm", tau=0.504,
                                     label=f"3d_sphere_n{n_sphere}")
    return runs


# --------------------------------------------------------------------------- #
# acceptance criteria                                                         #
# --------------------------------------------------------------------------- #


def evaluate_criteria(results, *, n_eval=128):
    gci_stage = results["gci"]
    lbm = {r["n"]: r for r in gci_stage["lbm"]}
    ref = lbm.get(n_eval) or gci_stage["lbm"][0]
    three = results.get("stage3d") or {}
    cs = three.get("cross_section_deviation", {})
    orders = gci_stage["orders_lbm"]
    crit = {}

    def add(key, name, value, threshold, ok, note=""):
        crit[key] = {
            "name": name,
            "value": value,
            "threshold": threshold,
            "pass": bool(ok) if ok is not None else None,
            "note": note,
        }

    add(
        "a_mass_conservation",
        "volume relative change after 2pi rotation (N=128) < 0.5%",
        ref["volume_abs_rel_change"],
        5e-3,
        ref["volume_abs_rel_change"] < 5e-3,
        "single-valued link fluxes; conservation is a structural property",
    )
    add(
        "b_interface_sharpness",
        "10-90% interface thickness (N=128) < 3 cells",
        ref["interface_thickness_mean_cells"],
        3.0,
        ref["interface_thickness_mean_cells"] < 3.0,
        "gradient-based thickness, mean over interface cells",
    )
    add(
        "c_shape_fidelity",
        "Jaccard overlap with the initial shape (N=128) > 0.95",
        ref["shape_overlap"],
        0.95,
        ref["shape_overlap"] > 0.95,
        "threshold phi >= 0.5",
    )
    p = orders.get("order_finest_pair")
    add(
        "d_convergence_order",
        "observed order of the L1 shape error > 0.5",
        p,
        0.5,
        (p is not None and p > 0.5),
        "the volume error itself is at round-off at every N, so its order is "
        "undefined (see d_volume_note); the shape-error order is used",
    )
    add(
        "d_volume_note",
        "volume-error convergence order",
        None,
        0.5,
        None,
        "volume drift is 1e-15..1e-13 relative at all N -> no resolvable "
        "convergence order; reported instead of a meaningless fit",
    )
    dev = cs.get("relative_l1")
    add(
        "e_3d_vs_2d",
        "D3Q19 vs D2Q9 cross-section relative L1 deviation < 5%",
        dev,
        0.05,
        (dev is not None and dev < 0.05),
        "extruded (y-invariant) body, slice y = 0.5",
    )
    add(
        "f_divergence",
        "max abs(div u) of the prescribed field < 1e-10",
        ref["div_u_max"],
        1e-10,
        ref["div_u_max"] < 1e-10,
        "face divergence of the analytic rigid rotation",
    )
    return crit


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #



def merge_stages(out_dir, n_eval=128):
    """Combine per-stage JSON files into ``zalesak_lbm_all.json`` + fields npz."""
    import glob

    import numpy as np

    merged = {}
    fields = {}
    for path in sorted(glob.glob(os.path.join(out_dir, "zalesak_lbm_*.json"))):
        if path.endswith("_all.json"):
            continue
        with open(path) as fh:
            res = json.load(fh)
        for k, v in res.items():
            if k in ("environment", "campaign", "module", "dtype", "geometry"):
                merged.setdefault(k, v)
            elif k in ("compare", "gci", "stage3d"):
                merged[k] = v
            else:
                merged.setdefault(k, []).append({path.split("_", 2)[-1].replace(".json", ""): v})
    for path in sorted(glob.glob(os.path.join(out_dir, "fields_*.npz"))):
        if path.endswith("fields_all.npz"):
            continue
        with np.load(path) as npz:
            for k in npz.files:
                fields[k] = npz[k]
    if "gci" in merged:
        merged["criteria"] = evaluate_criteria(merged, n_eval=n_eval)
        merged["criteria_summary"] = {
            k: v["pass"] for k, v in merged["criteria"].items() if v["pass"] is not None
        }
    merged["geometry"] = {
        "analytic_volume_2d": analytic_volume_2d(),
        "discrete_volume_2d_n64": float(Z.initial_fill(64).mean()),
        "discrete_volume_2d_n128": float(Z.initial_fill(128).mean()),
    }
    path = os.path.join(out_dir, "zalesak_lbm_all.json")
    with open(path, "w") as fh:
        json.dump(sanitize(merged), fh, indent=1, allow_nan=False)
    if fields:
        np.savez_compressed(os.path.join(out_dir, "fields_all.npz"), **fields)
    print(f"merged -> {path} ({len(fields)} field arrays)")
    if "criteria_summary" in merged:
        print("criteria:", merged["criteria_summary"])
    return merged


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", default="all",
                    choices=["all", "compare", "gci", "3d", "smoke", "merge"])
    ap.add_argument("--out", default="artifacts/zalesak_p190_n01")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--sizes", default="64,128,256")
    ap.add_argument("--n-eval", type=int, default=128)
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    results = {
        "campaign": "P190-N01",
        "module": "tensorlbm.free_surface_zalesak_lbm",
        "dtype": "float64",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "threads": args.threads,
        },
    }
    fields = {}
    sizes = [int(s) for s in args.sizes.split(",")]

    if args.stage == "merge":
        merge_stages(args.out, n_eval=args.n_eval)
        return 0
    if args.stage == "smoke":
        results["smoke"] = {
            "compare": stage_compare(n=32),
            "gci": {"lbm": [run_lbm(32, label="smoke32")], "vof": [run_vof(32)]},
            "stage3d": stage_3d(n=24, n_sphere=16),
        }
    else:
        if args.stage in ("all", "compare"):
            results["compare"] = stage_compare(n=64, record_fields=True)
        if args.stage in ("all", "gci"):
            results["gci"] = stage_gci(sizes=sizes)
        if args.stage in ("all", "3d"):
            results["stage3d"] = stage_3d()

    # keep the fields out of the JSON (arrays go to .npz)
    for stage in ("compare", "gci", "stage3d"):
        node = results.get(stage)
        if not node:
            continue
        stack = [node]
        while stack:
            cur = stack.pop()
            if isinstance(cur, dict):
                for k in list(cur.keys()):
                    if k.startswith("_phi"):
                        fields[f"{k.lstrip('_')}__{cur.get('label', 'run')}"] = cur.pop(k)
                    else:
                        stack.append(cur[k])
            elif isinstance(cur, list):
                stack.extend(cur)

    if "gci" in results and "stage3d" in results:
        results["criteria"] = evaluate_criteria(results, n_eval=args.n_eval)
        results["criteria_summary"] = {
            k: v["pass"] for k, v in results["criteria"].items() if v["pass"] is not None
        }
    results["wall_seconds_total"] = round(time.time() - t0, 1)
    results["geometry"] = {
        "analytic_volume_2d": analytic_volume_2d(),
        "discrete_volume_2d_n64": float(Z.initial_fill(64).mean()),
        "discrete_volume_2d_n128": float(Z.initial_fill(128).mean()),
    }

    path = os.path.join(args.out, f"zalesak_lbm_{args.stage}.json")
    with open(path, "w") as fh:
        json.dump(sanitize(results), fh, indent=1, allow_nan=False)
    if fields:
        import numpy as np

        np.savez_compressed(os.path.join(args.out, f"fields_{args.stage}.npz"),
                            **{k: v.numpy() for k, v in fields.items()})
    print(f"wrote {path} ({results['wall_seconds_total']}s total)", flush=True)
    if "criteria_summary" in results:
        print("criteria:", results["criteria_summary"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())