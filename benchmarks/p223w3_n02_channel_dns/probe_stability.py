#!/usr/bin/env python3
"""Stability / cost envelope probe for the native channel kernel.

Records the two numbers the campaign design hinges on:

1. **Collision branch.**  ``--reg 1`` (second-order Hermite projection, the
   pre-fix default) versus ``--reg 0`` (TRT + Smagorinsky) on the same state.
   The projection branch blows up as soon as the flow turns turbulent at the
   relaxation times this kernel runs at.
2. **Wall-normal spacing.**  At fixed ``u_tau = 0.0125`` the relaxation time is
   ``tau0 = 0.5 + 3 u_tau / dy+``, so the accessible ``dy+`` is bounded by the
   stability limit ``tau0 >~ 0.51``.  The sweep below brackets that limit and
   is what the campaign's ``dy+ <= 4`` choice is based on.

Usage::

    python benchmarks/p223w3_n02_channel_dns/probe_stability.py

Writes ``artifacts/p223w3_n02_channel_dns/stability_probe.json``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO / "src"))
OUT = _REPO / "artifacts" / "p223w3_n02_channel_dns"

from tensorlbm.turbulence.les_channel import (  # noqa: E402
    LesChannelConfig,
    run_native,
)


def attempt(name: str, **kw) -> dict:
    cfg = LesChannelConfig(**kw)
    started = time.time()
    try:
        payload = run_native(cfg, OUT / "probe_runs", name, overwrite=True)
        return {"case": name, "diverged": False,
                "u_bulk": payload["bulk_velocity"],
                "mass_drift": payload["diagnostics"]["mass_drift_fraction"],
                "wall_clock_s": round(time.time() - started, 1),
                "tau0": payload["derived"]["tau0"],
                "dy_plus": payload["derived"]["dy_plus"]}
    except RuntimeError as exc:
        return {"case": name, "diverged": True, "error": str(exc).splitlines()[0][:200],
                "wall_clock_s": round(time.time() - started, 1),
                "tau0": 0.5 + 3.0 * cfg.u_tau * cfg.half_height
                / (cfg.re_tau * cfg.half_height / cfg.re_tau) / cfg.half_height,
                "dy_plus": cfg.re_tau / cfg.half_height}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    common = dict(nx=64, ny=34, nz=32, steps=2500, warmup=800, sample=50,
                  re_tau=180.0, u_tau=0.0125, cs=0.1, perturb=0.05,
                  perturb_modes=6, seed=5, threads=16)
    rows = []
    # 1) collision branch on the *same* state, 4000 steps.  ny=50 -> tau0=0.505
    #    is the only band where plain TRT survives long enough to make the
    #    comparison meaningful (below that both branches fail immediately).
    for reg in (0, 1):
        rows.append(attempt(f"reg{reg}_dy7.5", regularized=reg,
                            **{**common, "ny": 50, "steps": 4000,
                               "warmup": 1000}))
    # 2) wall-normal spacing sweep, plain TRT (tau0 = 0.5 + 3 u_tau/dy+)
    for ny in (26, 34, 50, 74, 122):
        rows.append(attempt(f"trt_ny{ny}", regularized=0, **{**common, "ny": ny}))
    payload = {
        "kernel": "src/tensorlbm/turbulence/les_chan.c",
        "u_tau": 0.0125,
        "note": "short-horizon probe (2500 steps, collision pair 4000 steps) from "
                "the same seed/perturbation; 'diverged' = kernel refused to write "
                "a non-finite state.  The long-horizon stability limit is stricter "
                "than this probe shows: campaign runs at dy+ = 4/5/6 (tau0 = "
                "0.5094/0.5075/0.5062) survived 3e4-7e4 steps before diverging.",
        "runs": rows,
    }
    (OUT / "stability_probe.json").write_text(json.dumps(payload, indent=1))
    for r in rows:
        print(f"{r['case']:14s} tau0={r['tau0']:.5f} dy+={r['dy_plus']:5.2f} "
              f"{'DIVERGED' if r['diverged'] else 'ok ubulk=%.4f' % r['u_bulk']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())