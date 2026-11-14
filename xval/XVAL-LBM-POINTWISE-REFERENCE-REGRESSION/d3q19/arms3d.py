"""In-repo D3Q19 step operators under test (torch fp64, layout ``(19,nz,ny,nx)``).

Each arm is a *whole step* ``f^{n+1} = Stream(Collide(f^n))`` assembled only
from the repo's own kernels -- the reference in :mod:`ref3d` uses a different
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

from tensorlbm import diff3d  # noqa: E402
from tensorlbm.collision_d3q19_advanced import collide_bgk_d3q19  # noqa: E402
from tensorlbm.solver3d import collide_bgk3d, collide_mrt3d  # noqa: E402

DTYPE = torch.float64


def _to_torch(f: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(f), dtype=DTYPE)


def _to_numpy(f: torch.Tensor) -> np.ndarray:
    return f.detach().cpu().numpy().astype(np.float64, copy=True)


def _consts():
    return diff3d.constants(dtype=DTYPE)


def step_solver3d_bgk(f: np.ndarray, tau: float, **_: object) -> np.ndarray:
    """``solver3d.collide_bgk3d`` + ``diff3d.stream3d`` (production 3-D path)."""
    c, _w = _consts()
    g = _to_torch(f)
    return _to_numpy(diff3d.stream3d(collide_bgk3d(g, tau), c))


def step_diff3d(f: np.ndarray, tau: float, **_: object) -> np.ndarray:
    """``diff3d.step3d`` (its own collide + roll streaming)."""
    c, w = _consts()
    return _to_numpy(diff3d.step3d(_to_torch(f), tau, c, w))


def step_adv_bgk(f: np.ndarray, tau: float, **_: object) -> np.ndarray:
    """``collision_d3q19_advanced.collide_bgk_d3q19`` + ``diff3d.stream3d``."""
    c, _w = _consts()
    g = _to_torch(f)
    return _to_numpy(diff3d.stream3d(collide_bgk_d3q19(g, tau), c))


def step_mrt_uniform(f: np.ndarray, tau: float, **_: object) -> np.ndarray:
    """``solver3d.collide_mrt3d`` with every *user-visible* rate at ``1/tau``.

    This is the closest the repo allows to "MRT == BGK": the three cubic rates
    (moment indices 16-18) are hard-wired to 1.0 in ``_mrt3d_s_vec``, so the
    step agrees with BGK only in the limit ``tau -> 1`` -- pinned by arm F1.
    """
    c, _w = _consts()
    g = _to_torch(f)
    rate = 1.0 / tau
    g = collide_mrt3d(g, tau, s_e=rate, s_eps=rate, s_q=rate, s_pi=rate)
    return _to_numpy(diff3d.stream3d(g, c))


def step_mrt_native(f: np.ndarray, tau: float, **_: object) -> np.ndarray:
    """``solver3d.collide_mrt3d`` with its native (s_e, s_eps, s_q, s_pi)."""
    c, _w = _consts()
    g = _to_torch(f)
    return _to_numpy(diff3d.stream3d(collide_mrt3d(g, tau), c))


ARMS = {
    "A1_solver3d_bgk": step_solver3d_bgk,
    "A2_diff3d_step": step_diff3d,
    "A3_adv_bgk_d3q19": step_adv_bgk,
    "A4_mrt_uniform": step_mrt_uniform,
    "A5_mrt_native": step_mrt_native,
}


def run_arm(arm: str, f0: np.ndarray, tau: float, steps: int) -> list[np.ndarray]:
    """Return the fields ``f^1 .. f^steps`` of one arm."""
    fn = ARMS[arm]
    f = np.array(f0, dtype=np.float64, copy=True)
    out = []
    for _ in range(steps):
        f = fn(f, tau)
        out.append(f)
    return out