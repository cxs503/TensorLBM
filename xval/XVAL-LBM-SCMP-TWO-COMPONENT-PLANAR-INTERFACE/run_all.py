"""Run the pre-registered arm matrix of XVAL-LBM-SCMP-TWO-COMPONENT-PLANAR-INTERFACE.

Usage::

    python3 run_all.py                 # every arm
    python3 run_all.py oss             # OSS (reference) arms only
    python3 run_all.py tl              # TensorLBM arms only
    python3 run_all.py tl_primary oss_primary   # explicit labels

Every arm is written to its own JSON immediately, so an interrupted job keeps
the finished arms.  Nothing is re-tuned: the label -> parameters map below is
the pre-registered matrix (see predeclared.json).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import oss_arm  # noqa: E402
import tl_arm  # noqa: E402

OUT = ROOT / "artifacts" / "xval_scmp_two_component_planar_interface"

# ---------------------------------------------------------------- OSS arms
OSS_BASE = dict(nfluids=2, gA=0.0, gB=0.0, noutput=2000)
OSS_ARMS = {
    "oss_primary": dict(OSS_BASE, nx=64, ny=64, gAB=6.0, nsteps=12000),
    "oss_g4": dict(OSS_BASE, nx=64, ny=64, gAB=4.0, nsteps=12000),
    "oss_n32": dict(OSS_BASE, nx=32, ny=32, gAB=6.0, nsteps=12000, noutput=1000),
    "oss_t8000": dict(OSS_BASE, nx=64, ny=64, gAB=6.0, nsteps=8000),
    "oss_divergence": dict(OSS_BASE, nx=64, ny=64, gA=-4.7, gAB=6.0, nsteps=400, noutput=200),
}

# ------------------------------------------------------------ TensorLBM arms
TL_ARMS = {
    "tl_primary": dict(n=64, g_ab=6.0, steps=12000, variant="faithful"),
    "tl_g4": dict(n=64, g_ab=4.0, steps=12000, variant="faithful"),
    "tl_n32": dict(n=32, g_ab=6.0, steps=12000, variant="faithful"),
    "tl_t8000": dict(n=64, g_ab=6.0, steps=8000, variant="faithful"),
    "tl_linear": dict(n=64, g_ab=6.0, steps=12000, variant="faithful", psi="linear"),
    "tl_native": dict(n=64, g_ab=6.0, steps=12000, variant="native"),
    "tl_divergence": dict(n=64, g_ab=6.0, steps=400, variant="faithful", g_self=(-4.7, 0.0)),
}


def _summary(res: dict) -> str:
    fields = res["fields"]
    a = np.asarray(fields["density_comp0"], dtype=float)
    if not np.all(np.isfinite(a)):
        return "DIVERGED (non-finite fields)"
    prof = a.mean(axis=1)
    umax = float(np.hypot(np.asarray(fields["velx"], float), np.asarray(fields["vely"], float)).max())
    return (f"rhoA bulk [{prof.min():.4f}, {prof.max():.4f}]  u_max={umax:.3e}  "
            f"wall={res['wall_seconds']:.0f}s  step={res.get('step_last', res.get('steps_run'))}")


def main(argv: list[str]) -> int:
    only = set(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log = []
    if not only or "oss" in only or any(a.startswith("oss_") for a in only):
        for label, params in OSS_ARMS.items():
            if only and "oss" not in only and label not in only:
                continue
            t = time.time()
            res = oss_arm.arm(label, params)
            path = oss_arm.save(res, OUT)
            line = f"[OSS] {label:14s} {_summary(res)}  ({time.time()-t:.0f}s)"
            print(line, flush=True)
            log.append(line)
    if not only or "tl" in only or any(a.startswith("tl_") for a in only):
        for label, cfg in TL_ARMS.items():
            if only and "tl" not in only and label not in only:
                continue
            t = time.time()
            res = tl_arm.run_arm(label=label, sample_every=2000, **cfg)
            path = tl_arm.save(res, OUT)
            line = f"[TL ] {label:14s} {_summary(res)}  ({time.time()-t:.0f}s)"
            print(line, flush=True)
            log.append(line)
    (OUT / "run_log.txt").write_text("\n".join(log) + "\n")
    print(f"ALL DONE in {time.time()-t0:.0f}s -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))