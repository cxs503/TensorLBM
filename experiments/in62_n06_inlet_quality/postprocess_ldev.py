#!/usr/bin/env python3
"""Tolerance sweep for the self-preservation length ``L_dev`` (post-processing).

The predeclared 5 % criterion is stricter than the statistical resolution of the
slab statistics (the inlet eddies have a correlation time of ``σ/u_conv ≈ 160``
steps, so a 4000-step run yields only O(20) independent samples per slab).  This
script turns the resulting ``nan`` into a quantitative statement:

* ``L_dev(tol)`` for ``tol = 5 … 30 %`` (same estimator, same data);
* the measured noise floor of the anisotropy shape, ``σ_shape``, from the
  per-slab standard error ``R_stderr`` divided by ``sqrt(N_eff)``.

Nothing is recomputed: the input is ``results_duct_<case>.json``.

Usage::

    PYTHONPATH=src python experiments/in62_n06_inlet_quality/postprocess_ldev.py \
        --run-dir runs/in62_n06_inlet_quality
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from tensorlbm.turbulence.inlet_development import development_length

TOLS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)


def _noise_floor(R: torch.Tensor, R_err: torch.Tensor, x_start: int, x_stop: int,
                 window: int) -> dict:
    """Per-slab shape uncertainty of ``R_ij/R_uu`` and its moving-average level."""
    ruu = R[:, 0, 0].clamp(min=1e-300)
    shape_err = R_err / ruu.view(-1, 1, 1)
    # entry-wise quadrature over the six independent components
    comp = torch.sqrt((shape_err[:, [0, 1, 2, 0], [0, 1, 2, 1]] ** 2).sum(-1))
    slab = float(comp[x_start:x_stop].median())
    return {
        "slab_shape_stderr_median": slab,
        "moving_average_floor": slab / math.sqrt(max(1, window)),
        "window": int(window),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/in62_n06_inlet_quality")
    ap.add_argument("--x-start", type=int, default=2)
    args = ap.parse_args()

    rd = Path(args.run_dir)
    out: dict = {"tols": list(TOLS), "cases": {}}
    for path in sorted(rd.glob("results_duct_*.json")):
        case = path.stem.replace("results_duct_", "")
        rec = json.loads(path.read_text())
        R = torch.tensor(rec["R_profile"], dtype=torch.float64)
        R_err = torch.tensor(rec["R_stderr_profile"], dtype=torch.float64)
        nx = R.shape[0]
        sweep = {}
        for tol in TOLS:
            shape_only = development_length(
                R, tol=tol, x_start=args.x_start, require_rate=False
            )
            strict = development_length(R, tol=tol, x_start=args.x_start)
            sweep[f"{tol:.2f}"] = {
                "L_dev_shape_only": shape_only["L_dev"],
                "L_dev_shape_and_rate": strict["L_dev"],
                "x_stop": shape_only["x_stop"],
                "max_shape_deviation": max(
                    d for d in shape_only["shape_deviation"] if math.isfinite(d)
                ),
            }
        floor = _noise_floor(R, R_err, args.x_start, nx - 1, max(5, nx // 12))
        rec["samples"] = rec.get("samples")
        out["cases"][case] = {
            "samples": rec.get("samples"),
            "predeclared_5pct_shape_only": sweep["0.05"]["L_dev_shape_only"],
            "predeclared_5pct_shape_and_rate": sweep["0.05"]["L_dev_shape_and_rate"],
            "sweep": sweep,
            "noise": floor,
        }
        print(f"[ldev] {case}: L_dev(5%, shape)={sweep['0.05']['L_dev_shape_only']} "
              f"L_dev(15%, shape)={sweep['0.15']['L_dev_shape_only']} "
              f"shape-noise floor={floor['moving_average_floor']:.3f}", flush=True)

    out["gates"] = {
        "G6_L_dev_5pct_shape_only": {
            k: v["predeclared_5pct_shape_only"] for k, v in out["cases"].items()
        },
        "G6r_L_dev_5pct_shape_and_rate": {
            k: v["predeclared_5pct_shape_and_rate"] for k, v in out["cases"].items()
        },
        "G6b_note": (
            "the 5 % criterion is below the statistical resolution of this "
            "sample count; the sweep quantifies the tolerance dependence "
            "instead of hiding it"
        ),
    }
    (rd / "results_ldev.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {rd / 'results_ldev.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())