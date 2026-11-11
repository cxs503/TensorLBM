#!/usr/bin/env python
"""Re-evaluate the acceptance criteria from stored case payloads.

Useful when a criterion's tolerance or bookkeeping changes: the (expensive) LBM
runs are read back from ``artifacts/*.json`` instead of being repeated.

Note on the control-volume bookkeeping: payloads written before the sign
convention was fixed store the plane-flux sum ``mom_in - mom_out`` (positive for
a body in *drag*) under the key ``C_T_control_volume``.  This script normalises
the key to the thrust convention (``C_T = mom_out - mom_in``) before evaluating.
"""
from __future__ import annotations

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
sys.path.insert(0, BENCH)
sys.path.insert(0, os.path.join(BENCH, "..", "..", "src"))

import run_sweep as S  # noqa: E402


def _recompute_window(payload: dict) -> None:
    """Re-averaged coefficients over the post-warm-up cycles.

    ``run_sweep`` averaged over *all* sampled steps, i.e. including the
    ``warmup_cycles`` transient.  The stored 64-samples-per-cycle force series
    allows the mean to be recomputed over the last full cycles only, which is
    what the acceptance criterion requires (>= 8 cycles, transient discarded).
    """
    import numpy as np

    cfg = payload["config"]
    per = cfg["steps_per_cycle"]
    warm = cfg["warmup_cycles"]
    series = payload.get("series") or []
    if not series:
        return
    steps = np.array([r["s"] for r in series], dtype=float)
    fx = np.array([r["fx"] for r in series], dtype=float)
    fy = np.array([r["fy"] for r in series], dtype=float)
    w = np.array([r["w"] for r in series], dtype=float)
    keep = steps > warm * per
    if keep.sum() < 16:
        return
    q = 0.5 * cfg["U"] ** 2 * cfg["chord"]
    qp = 0.5 * cfg["U"] ** 3 * cfg["chord"]
    ct = fx[keep] / q
    cp = w[keep] / qp
    st = steps[keep]
    # per-cycle means from the (possibly sparse) samples: bucket by cycle index
    cyc_idx = ((st - warm * per) // per).astype(int)
    cyc_T = np.array([ct[cyc_idx == i].mean() for i in range(cyc_idx.max() + 1)])
    cyc_P = np.array([cp[cyc_idx == i].mean() for i in range(cyc_idx.max() + 1)])
    payload["coeffs_window"] = {
        "C_T": float(ct.mean()),
        "C_L": float((-fy[keep] / q).mean()),
        "C_P": float(cp.mean()),
        "eta": float(ct.mean() / cp.mean()) if cp.mean() > 0 else float("nan"),
        "C_T_cycle_std": float(cyc_T.std(ddof=1)) if cyc_T.size > 1 else 0.0,
        "n_cycles_averaged": float(len(st) / per),
        "n_cycles_full": int(cyc_idx.max() + 1),
        "samples": int(keep.sum()),
    }
    payload["coeffs"] = {**payload["coeffs"], **payload["coeffs_window"]}


def main() -> int:
    outdir = os.path.join(BENCH, "artifacts")
    payloads = {}
    for path in sorted(glob.glob(os.path.join(outdir, "*.json"))):
        name = os.path.basename(path)[:-5]
        if name in {"results", "analytic"} or name.startswith(("ext_", "fields_")):
            continue
        with open(path) as fh:
            d = json.load(fh)
        if "label" not in d or "config" not in d:
            continue
        extra = d.setdefault("extra", {})
        cv_drag = extra.get("C_D_control_volume")
        if cv_drag is None:
            cv_drag = extra.get("C_T_control_volume")
        if cv_drag is not None:
            # stored plane-flux sum is drag-positive; normalise both keys
            extra["C_D_control_volume"] = float(cv_drag)
            extra["C_T_control_volume"] = -float(cv_drag)
        if "config" in d:
            _recompute_window(d)
            d["sha256"] = S.canonical_sha256(d)
        payloads[d["label"]] = d
    with open(os.path.join(outdir, "analytic.json")) as fh:
        analytic = json.load(fh)

    missing = {c["label"] for c in S.CASES} - set(payloads)
    if missing:
        print("missing cases, cannot evaluate:", sorted(missing))
        return 1
    results = S.evaluate(payloads, analytic, outdir)
    with open(os.path.join(outdir, "results.json"), "w") as fh:
        json.dump(results, fh, indent=1)
    print(json.dumps(results["summary"], indent=1))
    for c in results["criteria"]:
        print(f"  {c['id']:4s} {c['verdict']:9s} {c['name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
