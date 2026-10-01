#!/usr/bin/env python3
"""Profile the per-step cost of a suboff_re1000 L=80 run on SDAA.

Wraps general_sim.lbm_step_correct, far_field_bc_3d, GeneralSimEngine._sample_forces
and GeneralSimEngine.step's isfinite guard to attribute the ~3.7 s/step.
Runs a short (default 30-step) job via run.py's own main().
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path("/root/TensorLBM_feat2")
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402
import tensorlbm.general_sim as gs  # noqa: E402
from tensorlbm import boundaries3d  # noqa: E402

S: dict[str, float] = {}
N: dict[str, int] = {}


def _acc(key, dt):
    S[key] = S.get(key, 0.0) + dt
    N[key] = N.get(key, 0) + 1


def wrap(mod, name):
    orig = getattr(mod, name)

    def f(*a, **k):
        t = time.time()
        r = orig(*a, **k)
        torch.sdaa.synchronize() if hasattr(torch, "sdaa") else None
        _acc(name, time.time() - t)
        return r

    setattr(mod, name, f)


for nm in (
    "lbm_step_correct",
    "far_field_bc_3d",
    "bounce_back_cells_3d",
    "correct_mass3d",
):
    if hasattr(gs, nm):
        wrap(gs, nm)
for nm in ("far_field_bc_3d", "bounce_back_cells_3d"):
    if hasattr(boundaries3d, nm):
        wrap(boundaries3d, nm)

_orig_sample = gs.GeneralSimEngine._sample_forces


def _sample(self, dpS, nu_lb):
    t = time.time()
    _orig_sample(self, dpS, nu_lb)
    _acc("_sample_forces", time.time() - t)


gs.GeneralSimEngine._sample_forces = _sample

# also time the isfinite divergence guard by wrapping torch.isfinite
_orig_isfinite = torch.isfinite


def _isfinite(x, *a, **k):
    t = time.time()
    r = _orig_isfinite(x, *a, **k)
    _acc("isfinite", time.time() - t)
    return r


torch.isfinite = _isfinite

sys.argv = [str(REPO / "benchmarks/pending/suboff_re1000/run.py")] + sys.argv[1:]
import runpy  # noqa: E402

t0 = time.time()
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
finally:
    print("\n=== PROFILE (cumulative, wall) ===", flush=True)
    for k in sorted(S, key=lambda x: -S[x]):
        print(f"{k:28s} total={S[k]:8.3f}s  n={N[k]:6d}  avg={S[k]/max(N[k],1)*1000:8.2f} ms", flush=True)
    print(f"script wall {time.time()-t0:.1f}s", flush=True)