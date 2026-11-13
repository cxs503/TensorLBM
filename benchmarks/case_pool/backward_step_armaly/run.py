#!/usr/bin/env python3
"""CASE-BACKWARD-STEP-LBM -- Armaly-caliber backward-facing step, compatibility CLI.

Role of this file (round 3)
---------------------------
Round 1 of this case shipped its own scalar LBM loop here.  That driver and the
library module :mod:`tensorlbm.bfs_armaly` (used by ``run_case.py``, by the
committed-artefact grading test and by ``verdict.py``) were two independent
implementations of the same benchmark: a second collision default, a second
near-wall derivative and a second reattachment caliper.  Worse, the artefact it
wrote could not be judged at all -- ``verdict.py`` reads ``derived.mach_inlet``,
``result.mass_flux_rel_error``, ``result.x1_over_s_reverse_extent`` and
``convergence_history``, none of which the round-1 payload had, so every point it
produced came back as FAIL(missing fields).

This file is therefore now a *thin* command-line shim over the single source of
truth (:func:`tensorlbm.bfs_armaly.run_armaly_case`).  The historical flags are
still accepted, so the reproduction commands quoted in
``docs/CASE_BACKWARD_STEP_LBM.md`` and ``docs/audit_m15924.md`` keep working, and
the artefact written is the canonical one -- machine-judgeable by ``verdict.py``
without any adapter.

Caliber (see ``src/tensorlbm/bfs_armaly.py`` for the full table)
    geometry   inlet channel h_in, step s, downstream H = h_in + s; by default the
               measured Armaly ratio 10.1/5.2 is kept (``--m`` is the step height
               in cells, ``h_in = round(m * 52/49)``).  ``--er2`` restores the
               round-1 lattice with h_in = s = m (ER = 2, +3.0 % vs the experiment).
    inlet      fully developed plane-Poiseuille profile, row mean exactly u_bar
    Re         Re = u_bar * D_h / nu, D_h = 2 h_in (Armaly caliber)
    observable X1/s, lower-wall reattachment = zero crossing of the time-averaged
               wall shear tau_w ~ -2u_1 + 3u_2 - u_3 (exact for a quadratic)

Usage
-----
    PYTHONPATH=src python3 benchmarks/case_pool/backward_step_armaly/run.py \\
        --re 100 --m 16 --steps 12000 --u 0.1
    PYTHONPATH=src python3 benchmarks/case_pool/backward_step_armaly/run.py --er2 --re 400 --m 26
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.bfs_armaly import (  # noqa: E402
    ARMALY_EXPANSION_RATIO,
    ArmalyConfig,
    run_armaly_case,
)

ART = REPO / "artifacts" / "case_backward_step_armaly"
#: Armaly et al. (1983) measured expansion ratio, kept as lattice integers 52:49
_RATIO_NUM, _RATIO_DEN = 52, 49


def config_from_args(args: argparse.Namespace) -> ArmalyConfig:
    """Map the historical CLI onto :class:`ArmalyConfig` (geometry is the only choice)."""
    m = int(args.m)
    if args.er2:
        hi, s = m, m
    else:
        hi, s = int(round(m * _RATIO_NUM / _RATIO_DEN)), m
    tag = args.tag or f"cli_re{int(round(args.re))}_m{m}" + ("_er2" if args.er2 else "")
    return ArmalyConfig(
        re=float(args.re),
        hi=hi,
        s=s,
        u_bar=float(args.u),
        l_over_h=float(args.l_over_h),
        n_steps=int(args.steps),
        avg_from=1.0 - float(args.avg_frac),
        avg_stride=int(args.avg_stride),
        collision="bgk" if args.collision == "bgk" else "trt",
        tag=tag,
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--re", type=float, default=100.0)
    ap.add_argument("--m", type=int, default=16, help="step height in cells (round-1 meaning; "
                                                     "the inlet height follows the Armaly ratio)")
    ap.add_argument("--u", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--avg-frac", type=float, default=0.25, help="last fraction of steps used for the time average")
    ap.add_argument("--avg-stride", type=int, default=200)
    ap.add_argument("--l-over-h", type=float, default=12.0)
    ap.add_argument("--collision", default="trt", choices=["trt", "rlbm", "bgk"],
                    help="'rlbm' is the round-1 name of the TRT collision (now the default)")
    ap.add_argument("--er2", action="store_true", help="round-1 lattice: h_in = s = m (ER = 2)")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--tag", default=None, help="artefact tag (default cli_re<Re>_m<m>)")
    ap.add_argument("--artifacts-dir", default=str(ART),
                    help="where <tag>/{result.json,wall_shear.csv,field.npz} is written")
    ap.add_argument("--out", default=None, help="also write the JSON payload at this path (round-1 habit)")
    args = ap.parse_args(argv)

    torch.set_num_threads(max(1, int(args.threads)))
    cfg = config_from_args(args)
    out_dir = Path(args.artifacts_dir) / cfg.tag
    t0 = time.time()
    payload = run_armaly_case(cfg, out_dir=out_dir)

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=2, default=str))

    res, der = payload["result"], payload["derived"]
    er = der["expansion_ratio"]
    print(f"[Re={cfg.re:.0f} h_in={cfg.hi} s={cfg.s} ER={er:.4f} "
          f"({der['expansion_ratio_deviation_pct']:+.1f}% vs {ARMALY_EXPANSION_RATIO:.4f}) "
          f"tau={der['tau']:.4f}] steps={res['steps_run']} "
          f"X1/S={res['x1_over_s']:.4f} (shear) / {res['x1_over_s_row1']:.4f} (row1) "
          f"t={time.time() - t0:.0f}s", flush=True)
    print(f"RESULT Re={cfg.re:.0f} m={args.m}: X1/S={res['x1_over_s']:.4f} -> {out_dir / 'result.json'}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())