#!/usr/bin/env python3
"""Canonical sphere Re=100 BFL/control-volume runner for the SDAA host.

Thin wrapper around ``tensorlbm.sphere_bfl_control_volume`` that
(i) boots the SDAA teco-inductor codegen (same shim as benchmarks/compile_route)
before any ``torch.compile`` is attempted, and
(ii) exposes the canonical benchmark with co-scaled domain / time / sponge /
control-volume parameters (the ``R``-scaled family of the admitted
sphere-v9-corrected-bfl evidence).

Usage:
    run_canonical.py --radius 9 --width-over-r 16 --device sdaa:8 --output out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "benchmarks"))

import torch  # noqa: E402

try:
    import torch_sdaa  # noqa: F401,E402
    import torch_sdaa._inductor  # noqa: F401,E402  registers sdaa codegen
except Exception as exc:  # pragma: no cover
    print(f"[warn] torch_sdaa._inductor not imported: {exc}", flush=True)

# Installing the shared inductor defect shims (stale Reduction binding,
# teco scheduling/expand-dim defects) — same module the verified benchmark
# compile path uses; it runs at import time.
try:
    import compile_route  # noqa: F401,E402

    print(f"[shim] compile_route loaded, teco tag={compile_route._SDAA_INDUCTOR_TAG}", flush=True)
except Exception as exc:  # pragma: no cover
    print(f"[warn] compile_route shim not loaded: {exc}", flush=True)

from tensorlbm.sphere_bfl_control_volume import (  # noqa: E402
    SphereBFLControlVolumeConfig,
    run_sphere_bfl_control_volume,
)


def _jsonable(obj):
    """Recursively convert torch / numpy scalars+tensors to JSON-safe types.

    The library result carries a few 0-d torch tensors (``cd_bfl_link``,
    ``observer_difference_pct`` and the ``force_observer_target_met`` bool),
    which ``json.dumps`` refuses.  Sanitise the whole tree instead of
    patching individual keys so future fields cannot reintroduce the crash.
    """
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        np = None
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if np is not None:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
    return obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, required=True)
    ap.add_argument(
        "--width-over-r",
        type=float,
        default=16.0,
        help="transverse half..full width in units of R (16 -> 8D wide)",
    )
    ap.add_argument(
        "--length-over-r",
        type=float,
        default=24.0,
        help="streamwise length in units of R (24 -> 12D)",
    )
    ap.add_argument("--center-x-fraction", type=float, default=0.30)
    ap.add_argument("--steps-over-r", type=float, default=800.0)
    ap.add_argument("--warmup-over-r", type=float, default=533.3333333333334)
    ap.add_argument("--window-over-r", type=float, default=266.6666666666667)
    ap.add_argument("--ramp-over-r", type=float, default=80.0)
    ap.add_argument("--sponge-over-r", type=float, default=2.0)
    ap.add_argument("--sponge-strength", type=float, default=0.2)
    ap.add_argument("--inlet-sponge", action="store_true")
    ap.add_argument("--cv-margin-over-r", type=float, default=2.0 / 3.0)
    ap.add_argument("--lattice-speed", type=float, default=0.06)
    ap.add_argument("--reynolds", type=float, default=100.0)
    ap.add_argument("--collision-model", default="natural_kbc_d3q19")
    ap.add_argument("--collision-chunk-cells", type=int, default=262144)
    ap.add_argument("--compile-natural-kbc", action="store_true")
    ap.add_argument("--far-field-mode", default="non_equilibrium_extrapolation")
    ap.add_argument("--device", default="sdaa:0")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    R = args.radius
    nx = int(round(args.length_over_r * R))
    nw = int(round(args.width_over_r * R))
    cfg = SphereBFLControlVolumeConfig(
        nx=nx,
        ny=nw,
        nz=nw,
        radius=R,
        center_x_fraction=args.center_x_fraction,
        reynolds=args.reynolds,
        lattice_speed=args.lattice_speed,
        steps=int(round(args.steps_over_r * R)),
        warmup_steps=int(round(args.warmup_over_r * R)),
        ramp_steps=int(round(args.ramp_over_r * R)),
        sponge_width=int(round(args.sponge_over_r * R)),
        sponge_strength=args.sponge_strength,
        sponge_inlet=bool(args.inlet_sponge),
        cv_margin=int(round(args.cv_margin_over_r * R)),
        far_field_mode=args.far_field_mode,
        report_interval=max(1, int(round(100 * R))),
        statistics_window_steps=int(round(args.window_over_r * R)),
        collision_model=args.collision_model,
        collision_chunk_cells=args.collision_chunk_cells,
        compile_natural_kbc=bool(args.compile_natural_kbc),
        device=args.device,
    )
    print(
        f"[canonical] R={R} shape_zyx=({cfg.nz},{cfg.ny},{cfg.nx}) "
        f"cells={cfg.nx * cfg.ny * cfg.nz} est_GiB={cfg.nx * cfg.ny * cfg.nz * 1000 / 2**30:.1f} "
        f"steps={cfg.steps} warmup={cfg.warmup_steps} window={cfg.statistics_window_steps} "
        f"nu={cfg.nu:.6g} tau={cfg.tau:.6f}",
        flush=True,
    )
    result = run_sphere_bfl_control_volume(cfg)
    payload = _jsonable(result)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "cd_control_volume": payload["result"]["cd_control_volume"],
                "cd_bfl_link": payload["result"]["cd_bfl_link"],
                "reference_error_pct": payload["result"]["reference_error_pct"],
                "observer_difference_pct": payload["result"]["observer_difference_pct"],
                "numerical_quality_admitted": payload["acceptance"]["numerical_quality_admitted"],
            },
            indent=2,
        ),
        flush=True,
    )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
