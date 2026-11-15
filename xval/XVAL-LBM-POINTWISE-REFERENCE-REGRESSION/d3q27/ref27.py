"""lbmpy side of the D3Q27 whole-step reference (requires lbmpy 2.0).

The collision operator is built symbolically by lbmpy from the Geier 2015 /
Lallemand-Luo definitions and evaluated pointwise (``LbmCollisionRule.lambdify``);
the streaming half is the definition operator from :mod:`refcore27`, which shares
no code with ``tensorlbm.d3q27.stream27_roll``.

Nothing here is transcribed from the repo: the MRT moment *basis* is
reconstructed from the repo's own ``_M_D3Q27_DATA`` by symbolically recovering
the Lagrange polynomials of each row, and the relaxation-rate vector is built
from the documented defaults of ``tensorlbm.d3q27.collide_mrt27``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
XV = HERE.parent
REPO = XV.parents[1]
for _p in (str(REPO / "src"), str(XV), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import lbmpy_pointwise_ref as lpref  # noqa: E402
import refcore27 as core  # noqa: E402

LATTICE = "D3Q27"


# ---------------------------------------------------------------------------
# MRT: basis + relaxation rates exactly as the repo uses them
# ---------------------------------------------------------------------------
def repo_mrt_rates(
    tau: float,
    *,
    s_e: float = 1.19,
    s_eps: float = 1.4,
    s_q: float = 1.2,
    s_pi: float | None = None,
) -> list[float]:
    """The 27-entry rate vector of ``collide_mrt27`` at its documented defaults.

    Layout (from the repo source): ``[0,0,0,0,s_e] + [1/tau]*5 + [s_q]*9 + [s_eps] + [s_pi]*7``
    with ``s_pi = s_e`` when not given.
    """
    if s_pi is None:
        s_pi = s_e
    return (
        [0.0, 0.0, 0.0, 0.0, s_e]
        + [1.0 / tau] * 5
        + [s_q] * 9
        + [s_eps]
        + [s_pi] * 7
    )


def repo_moment_matrix() -> np.ndarray:
    """D3Q27 moment matrix of the repo, ``(27, 27)`` float64."""
    from tensorlbm.d3q27 import _M_D3Q27_DATA  # noqa: PLC0415

    return np.asarray(_M_D3Q27_DATA, dtype=np.float64)


# ---------------------------------------------------------------------------
# reference colliders
# ---------------------------------------------------------------------------
def reference_collider(tau: float, model: str, **params):
    """Pointwise lbmpy collision operator for D3Q27."""
    if model == "srt":
        return lpref.PointwiseReference(
            LATTICE, "srt", tau=tau, continuous_equilibrium=False
        )
    if model == "trt":
        return lpref.PointwiseReference(
            LATTICE, "trt", tau_plus=tau, lambda_trt=params.get("lambda_trt", 3.0 / 16.0)
        )
    if model == "mrt_repo":
        return lpref.PointwiseReference(
            LATTICE,
            "mrt_from_matrix",
            moment_matrix=repo_moment_matrix(),
            relaxation_rates=repo_mrt_rates(
                tau,
                s_e=params.get("s_e", 1.19),
                s_eps=params.get("s_eps", 1.4),
                s_q=params.get("s_q", 1.2),
                s_pi=params.get("s_pi"),
            ),
        )
    if model == "cumulant":
        # Geier 2015 default polynomial cumulant set, continuous Maxwellian
        # equilibrium, every non-shear group relaxed with 1.
        return lpref.PointwiseReference(LATTICE, "cumulant", tau=tau)
    if model == "cumulant_discrete_eq":
        # Same relaxation, discrete 2nd-order Maxwellian equilibrium (the
        # convention of tensorlbm.d3q27.equilibrium27).
        return lpref.PointwiseReference(LATTICE, "cumulant_discrete_eq", tau=tau)
    raise ValueError(f"unknown reference model {model!r}")


# ---------------------------------------------------------------------------
# whole step
# ---------------------------------------------------------------------------
def whole_step(
    f0: np.ndarray,
    collider,
    steps: int,
    keep: tuple[int, ...] = (),
) -> dict[int, np.ndarray]:
    """Advance ``S_ref = Stream_definition o Collide_lbmpy``.

    Returns ``{t: f^t}`` for ``t`` in ``keep`` (``steps`` is always kept).
    """
    want = sorted(set(keep) | {steps})
    f = np.array(f0, dtype=np.float64, copy=True)
    states: dict[int, np.ndarray] = {}
    for t in range(1, int(steps) + 1):
        f = core.definition_stream27(collider.apply(f))
        if t in want:
            states[t] = f.copy()
    return {t: states[t] for t in want}


def moment_basis_gap() -> float:
    """``max |M_lbmpy - M_repo|`` for the reconstructed D3Q27 moment basis."""
    return lpref.moment_basis_check(repo_moment_matrix(), LATTICE)


def provenance() -> dict:
    return lpref.lbmpy_provenance()