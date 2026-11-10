#!/usr/bin/env python3
"""Run ONE LES plane-channel case of the P127W2-N03 campaign.

Geometry follows the campaign specification ``Lx x H x Lz = 2 pi h x 2h x pi h``
in units of the half-height ``h``; the streamwise/spanwise node counts are
derived from ``h`` so the box aspect ratio is exact to one node.

The case writes a single JSON artifact holding

* the full solver result (wall-unit profiles, Reynolds stresses, spectra,
  SGS budget, mass bookkeeping, friction-velocity routes),
* a compact ``summary`` of the headline scalars, and
* the campaign ``verdict`` (per-criterion PASS / FAIL / WITHHELD with the
  numbers behind every boolean).

Nothing in the artifact is hand-entered: the verdict is computed by
:func:`tensorlbm.turbulence.les_channel3d.evaluate_case` from the run itself and
the vendored MKM1999 tables.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.turbulence.les_channel3d import (  # noqa: E402
    LESChannel,
    LESChannelConfig,
    evaluate_case,
    mkm_profile_for,
    summary_metrics,
)


def build_config(args: argparse.Namespace) -> LESChannelConfig:
    h = args.h
    nx = args.nx or max(8, int(round(2.0 * math.pi * h)))
    nz = args.nz or max(8, int(round(math.pi * h)))
    return LESChannelConfig(
        nx=nx,
        nz=nz,
        ny=2 * h + 2,  # two solid rows -> H = 2h
        re_tau=args.re_tau,
        u_tau=args.u_tau,
        steps=args.steps,
        warmup_steps=args.warmup,
        sample_interval=args.sample_interval,
        report_interval=args.report_interval,
        smagorinsky_cs=args.cs,
        strain_source=args.strain_source,
        wall_model=args.wall_model,
        wall_model_interval=args.wall_model_interval,
        wall_model_filter=args.wall_model_filter,
        perturbation_amplitude=args.perturbation,
        dtype=args.dtype,
        wall_origin=0.5,
        wall_scheme="interpolated",
        label=args.label,
        seed=args.seed,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--re-tau", type=float, required=True)
    ap.add_argument("--h", type=int, required=True, help="half-height in lattice nodes")
    ap.add_argument("--nx", type=int, default=0)
    ap.add_argument("--nz", type=int, default=0)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--warmup", type=int, required=True)
    ap.add_argument("--sample-interval", type=int, default=10)
    ap.add_argument("--report-interval", type=int, default=1000)
    ap.add_argument("--cs", type=float, default=0.12)
    ap.add_argument("--u-tau", type=float, default=0.01)
    ap.add_argument("--perturbation", type=float, default=0.35)
    ap.add_argument("--strain-source", default="fneq", choices=["fneq", "velocity_gradient"])
    ap.add_argument("--wall-model", default="none", choices=["none", "spalding"])
    ap.add_argument("--wall-model-interval", type=int, default=10)
    ap.add_argument("--wall-model-filter", type=float, default=0.5)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--seed", type=int, default=20261109)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-verdict", action="store_true")
    args = ap.parse_args(argv)

    torch.set_num_threads(max(1, int(args.threads)))
    cfg = build_config(args)
    t0 = time.time()
    sim = LESChannel(cfg)
    result = sim.run_les(verbose=True)
    profile = mkm_profile_for(args.re_tau, ROOT)
    verdict = (
        {"skipped": True}
        if args.no_verdict
        else evaluate_case(result, dns=profile)
    )
    payload = {
        "mission": "P127W2-N03",
        "label": args.label,
        "re_tau_target": args.re_tau,
        "geometry": {
            "nx": cfg.nx,
            "ny": cfg.ny,
            "nz": cfg.nz,
            "h": cfg.half_height,
            "lx_over_h": cfg.nx / cfg.half_height,
            "lz_over_h": cfg.nz / cfg.half_height,
            "dy_plus": cfg.dy_plus,
            "first_fluid_node_y_plus": cfg.dy_plus * cfg.wall_origin,
            "dy_plus": cfg.dy_plus,
            "wall_model": cfg.wall_model,
        },
        "wall_clock_seconds": time.time() - t0,
        "dns_reference": None
        if profile is None
        else {"source": profile.source, "re_tau": profile.re_tau},
        "summary": summary_metrics(result, profile),
        "verdict": verdict,
        "result": result,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    s = payload["summary"]
    print(
        f"[{args.label}] done in {payload['wall_clock_seconds']:.0f}s "
        f"cells={cfg.nx * cfg.ny * cfg.nz} "
        f"u+max={s.get('u_plus_max')} uv+peak={s.get('uv_peak_plus')} "
        f"sgs@50={s.get('sgs_fraction_y50')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())