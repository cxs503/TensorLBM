"""In-repo D3Q27 step operators under test (torch fp64, layout ``(27,nz,ny,nx)``).

Each arm is a *whole step* ``f^{n+1} = Stream(Collide(f^n))`` assembled only
from the repo's own kernels -- the reference in :mod:`ref27` uses a different
collision (lbmpy symbolic) and a different streaming (numpy definition), so an
agreement at fp64 round-off means the composition is right, not that the same
code was run twice.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
XV = HERE.parent
REPO = XV.parents[1]
_SRC = str(REPO / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import refcore27 as core  # noqa: E402
from tensorlbm.cumulant import (  # noqa: E402
    collide_cumulant_d3q27,
    collide_cumulant_geier_d3q27,
)
from tensorlbm.d3q27 import (  # noqa: E402
    collide_bgk27,
    collide_mrt27,
    collide_trt27,
    stream27_roll,
)

DTYPE = torch.float64


def _to_torch(f: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(f), dtype=DTYPE)


def _to_numpy(f: torch.Tensor) -> np.ndarray:
    return f.detach().cpu().numpy().astype(np.float64, copy=True)


# ---------------------------------------------------------------------------
# collision halves
# ---------------------------------------------------------------------------
def collide_a1_bgk(f: torch.Tensor, tau: float) -> torch.Tensor:
    return collide_bgk27(f, tau)


def collide_a2_trt(f: torch.Tensor, tau: float) -> torch.Tensor:
    return collide_trt27(f, tau)  # lambda_trt defaults to the magic 3/16


def collide_a3_mrt_native(f: torch.Tensor, tau: float) -> torch.Tensor:
    return collide_mrt27(f, tau)  # documented defaults s_e=1.19, s_eps=1.4, s_q=1.2


def collide_a4_cumulant_geier(f: torch.Tensor, tau: float) -> torch.Tensor:
    return collide_cumulant_geier_d3q27(f, tau)


def collide_a5_cumulant_legacy(f: torch.Tensor, tau: float) -> torch.Tensor:
    return collide_cumulant_d3q27(f, tau)


COLLIDERS = {
    "A1_d3q27_bgk": collide_a1_bgk,
    "A2_d3q27_trt": collide_a2_trt,
    "A3_d3q27_mrt_native": collide_a3_mrt_native,
    "A4_d3q27_cumulant_geier": collide_a4_cumulant_geier,
    "A5_d3q27_cumulant_legacy": collide_a5_cumulant_legacy,
}

#: reference model each arm is compared against (see ``ref27.reference_collider``)
REFERENCE_MODEL = {
    "A1_d3q27_bgk": "srt",
    "A2_d3q27_trt": "trt",
    "A3_d3q27_mrt_native": "mrt_repo",
    "A4_d3q27_cumulant_geier": "cumulant_discrete_eq",
    "A5_d3q27_cumulant_legacy": "cumulant",
}

#: arms whose operator algebra is claimed identical to the reference (gated);
#: A5 is a declared *convention probe* (relax f_neq cumulants vs full-cumulant
#: relaxation), reported with its measured gap and amplitude scaling, never
#: re-tuned into a pass.
GATED_ARMS = ("A1_d3q27_bgk", "A2_d3q27_trt", "A3_d3q27_mrt_native",
              "A4_d3q27_cumulant_geier")
PROBE_ARMS = ("A5_d3q27_cumulant_legacy",)


# ---------------------------------------------------------------------------
# streaming half
# ---------------------------------------------------------------------------
def stream(f: np.ndarray) -> np.ndarray:
    """``tensorlbm.d3q27.stream27_roll`` (torch roll streaming)."""
    return _to_numpy(stream27_roll(_to_torch(f)))


def stream_definition(f: np.ndarray) -> np.ndarray:
    """Reference streaming (numpy definition), for the streaming-only pin."""
    return core.definition_stream27(np.asarray(f, dtype=np.float64))


# ---------------------------------------------------------------------------
# whole steps
# ---------------------------------------------------------------------------
def step(arm: str, f: np.ndarray, tau: float) -> np.ndarray:
    """One in-repo step ``Stream_repo(Collide_arm(f))``."""
    post = _to_numpy(COLLIDERS[arm](_to_torch(f), tau))
    return stream(post)


def run_arm(arm: str, f0: np.ndarray, tau: float, steps: int,
            keep: tuple[int, ...] = ()) -> dict[int, np.ndarray]:
    """Return ``{t: f^t}`` for ``t`` in ``keep`` (``steps`` always kept)."""
    want = sorted(set(keep) | {steps})
    f = np.array(f0, dtype=np.float64, copy=True)
    out: dict[int, np.ndarray] = {}
    for t in range(1, int(steps) + 1):
        f = step(arm, f, tau)
        if t in want:
            out[t] = f.copy()
    return {t: out[t] for t in want}