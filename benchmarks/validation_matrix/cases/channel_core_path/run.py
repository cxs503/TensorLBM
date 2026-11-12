#!/usr/bin/env python3
"""Validation-matrix case ``channel_core_path_ci`` — T2 core path, laminar channel.

The T2 GPU core-path optimisation (``src/tensorlbm/gpu_core_path.py``) is only
worth anything if the optimised time loop still solves the *physics* it claims
to solve.  This case therefore drives the body-force channel through the core
path itself (collision + Guo force + streaming + half-way bounce-back in one
fused step) and judges the result against the closed-form Poiseuille profile

    u(y) = a / (2 nu) * s * (H - s),   s = y - 0.5,   H = ny - 2

at two resolutions (``ny = 18`` and ``ny = 34``), so the artifact carries both
the error level and its resolution trend.

Artifact: ``benchmarks/validation_matrix/cases/channel_core_path/result.json``
in the schema consumed by ``benchmarks/validation_matrix/criteria.py``
(``levels[i].metrics.{u_max_err_pct, profile_l2_pct, mass_drift_pct}``).

Repro::

    python benchmarks/validation_matrix/cases/channel_core_path/run.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from tensorlbm.gpu_core_path import (  # noqa: E402
    GPUCorePathConfig,
    run_channel_case,
)

#: Two resolutions: channel width H = 16 and H = 32 lattice units.
LEVELS = (
    {"level": "ny=18 (H=16)", "ny": 18, "max_steps": 6000},
    {"level": "ny=34 (H=32)", "ny": 34, "max_steps": 20000},
)
TAU = 0.8
U_MAX_TARGET = 0.05


def main() -> int:
    torch.set_num_threads(int(os.environ.get("TL_THREADS", "1")))
    backend = os.environ.get("CORE_PATH_BACKEND", "auto")
    levels = []
    for spec in LEVELS:
        cfg = GPUCorePathConfig(
            ny=spec["ny"],
            tau=TAU,
            u_max_target=U_MAX_TARGET,
            steps=spec["max_steps"],
            backend=backend,  # type: ignore[arg-type]
            precision=os.environ.get("CORE_PATH_PRECISION", "FP64FP64"),
            label=f"channel_ny{spec['ny']}",
        )
        t0 = time.perf_counter()
        row = run_channel_case(cfg, check_every=200, steady_tol=2e-5)
        levels.append(
            {
                "level": spec["level"],
                "metrics": {
                    "u_max_err_pct": 100.0 * row["u_max_rel_err"],
                    "profile_l2_pct": 100.0 * row["profile_rel_l2"],
                    "mass_drift_pct": 100.0 * row["mass_drift"],
                },
                "u_max_ref": row["u_max_ref"],
                "u_max_got": row["u_max_got"],
                "steps": row["steps"],
                "steady_at_step": row["steady_at_step"],
                "elapsed_s": row["elapsed_s"],
                "mlups": row["mlups"],
                "backend": row["backend"],
                "device": row["device"],
                "compute_dtype": row["compute_dtype"],
                "wall_clock_s": time.perf_counter() - t0,
            }
        )
        print(
            f"{spec['level']:14s} u_max ref={row['u_max_ref']:.6f} got={row['u_max_got']:.6f} "
            f"err={levels[-1]['metrics']['u_max_err_pct']:.3f}%  "
            f"L2={levels[-1]['metrics']['profile_l2_pct']:.3f}%  "
            f"steps={row['steps']} steady@{row['steady_at_step']}",
            flush=True,
        )

    payload = {
        "case": "channel_core_path_ci",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reference": "analytic Poiseuille u(y)=a/(2 nu) s (H-s), s=y-0.5, H=ny-2",
        "reference_basis": "analytic",
        "implementation": "src/tensorlbm/gpu_core_path.py::run_channel_case",
        "config": {
            "tau": TAU,
            "nu": (TAU - 0.5) / 3.0,
            "u_max_target": U_MAX_TARGET,
            "init": "rest",
            "steady_tol": 2e-5,
        },
        "levels": levels,
    }
    out = HERE / "result.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())