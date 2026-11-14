"""lbmpy side of the D3Q19 whole-step reference (requires lbmpy 2.0).

The collision operator is built symbolically by lbmpy and evaluated pointwise
(``LbmCollisionRule.lambdify``); the streaming half is the definition operator
from :mod:`refcore3d`.  Nothing here is transcribed from the repo: the moment
basis for the MRT arm is *reconstructed* from the repo's own M matrix by
``lbmpy_pointwise_ref.moment_basis_check``-style symbolic solving, and the
relaxation-rate vector is read from ``solver3d._mrt3d_s_vec``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
XV = HERE.parent
REPO = XV.parents[1]
for _p in (str(REPO / "src"), str(XV)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import lbmpy_pointwise_ref as lpref  # noqa: E402  (round-1 reference module)
import refcore3d as core  # noqa: E402

LATTICE = "D3Q19"


def reference_collider(tau: float, equilibrium: str = "discrete", model: str = "srt", **params):
    """A pointwise lbmpy collision operator for D3Q19."""
    continuous = equilibrium == "continuous"
    if model == "srt":
        return lpref.PointwiseReference(
            LATTICE, "srt", tau=tau, continuous_equilibrium=continuous
        )
    if model in ("mrt_from_matrix", "mrt_uniform"):
        return lpref.PointwiseReference(LATTICE, "mrt_from_matrix", **params)
    raise ValueError(f"unknown model {model!r}")


def repo_mrt_rates(tau: float, *, uniform: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """(M matrix, relaxation-rate vector) exactly as the repo uses them.

    ``uniform=True`` builds the rate vector through the repo's own
    ``_mrt3d_s_vec`` with every user-visible rate at ``1/tau`` -- the closest
    thing to a "degenerate to BGK" setting that the repo's hard-wired cubic
    rates (indices 16-18 are pinned to 1.0) allow.
    """
    import torch  # noqa: PLC0415

    from tensorlbm.solver3d import _M_D3Q19_DATA, _mrt3d_s_vec  # noqa: PLC0415

    if uniform:
        s_vec = _mrt3d_s_vec(1.0 / tau, 1.0 / tau, 1.0 / tau, 1.0 / tau, 1.0 / tau,
                             dtype=torch.float64, device=torch.device("cpu"))
    else:
        s_vec = _mrt3d_s_vec(1.19, 1.4, 1.2, 1.19, 1.0 / tau, dtype=torch.float64,
                             device=torch.device("cpu"))
    return np.asarray(_M_D3Q19_DATA, dtype=np.float64), s_vec.numpy().astype(np.float64)


def whole_step_reference(
    f0: np.ndarray,
    tau: float,
    steps: int,
    *,
    equilibrium: str = "discrete",
    model: str = "srt",
    keep: tuple[int, ...] = (),
    **params,
) -> dict[int, np.ndarray]:
    """Advance the reference step ``S_ref = Stream_definition o Collide_lbmpy``.

    Returns a dict ``{t: f}`` with ``t`` in ``keep`` (``keep`` must be a subset of
    ``1..steps``; ``steps`` is always included).
    """
    want = set(keep) | {steps}
    collider = reference_collider(tau, equilibrium=equilibrium, model=model, tau=tau, **params)
    f = np.array(f0, dtype=np.float64, copy=True)
    states: dict[int, np.ndarray] = {0: f}
    for t in range(1, steps + 1):
        f = core.definition_stream3d(collider.apply(f))
        if t in want:
            states[t] = f
    return {t: states[t] for t in want}


def moment_basis_gap(moment_matrix: np.ndarray) -> float:
    """``max |M_lbmpy - M_repo|`` for the reconstructed D3Q19 moment basis."""
    return lpref.moment_basis_check(moment_matrix, LATTICE)


def provenance() -> dict:
    return lpref.lbmpy_provenance()