"""Minimal lattice definitions used by the turbulence-closure validation face.

Only two lattices are needed here: ``D2Q9`` (planar validation cases: backward
facing step, circular cylinder) and ``D3Q19`` (wall-resolved / wall-modelled
turbulent channel).  The discrete velocity sets are written down explicitly so
that the very same ordering can be reproduced by the C kernels (see
``mrt_d2q9.c`` / ``mrt_channel3d.c``); the ordering is asserted by
``tests/test_turbulence_closure_mrt.py``.

No physics lives in this module - it is a lookup table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["D2Q9", "D3Q19", "Lattice", "get_lattice"]


@dataclass(frozen=True)
class Lattice:
    """A discrete velocity set with lattice weights."""

    name: str
    c: np.ndarray  # (nq, d) integer velocities
    w: np.ndarray  # (nq,) weights summing to 1
    opp: np.ndarray  # (nq,) index of the opposite velocity

    @property
    def nq(self) -> int:
        return int(self.c.shape[0])

    @property
    def dim(self) -> int:
        return int(self.c.shape[1])

    @property
    def cs2(self) -> float:
        """Speed of sound squared, ``1/3`` for both lattices."""
        return 1.0 / 3.0

    def equilibrium(self, rho, u):
        """Second-order Maxwell-Boltzmann equilibrium ``f_eq`` (numpy).

        ``rho`` has shape ``(...,)`` and ``u`` has shape ``(..., dim)``; the
        returned array has shape ``(nq, ...)``.
        """
        rho = np.asarray(rho, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)
        cu = np.tensordot(self.c, u, axes=([1], [u.ndim - 1]))  # (nq, ...)
        usq = np.sum(u * u, axis=-1)[None, ...]
        w = self.w.reshape((self.nq,) + (1,) * (cu.ndim - 1))
        return rho[None, ...] * w * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)


def _build(name, c, w):
    c = np.asarray(c, dtype=np.int64)
    w = np.asarray(w, dtype=np.float64)
    opp = np.empty(len(c), dtype=np.int64)
    for q, v in enumerate(c):
        match = np.where(np.all(c == -v, axis=1))[0]
        if match.size != 1:  # pragma: no cover - definition error
            raise ValueError(f"velocity {v} has no unique opposite")
        opp[q] = int(match[0])
    return Lattice(name=name, c=c, w=w, opp=opp)


D2Q9 = _build(
    "D2Q9",
    c=[
        (0, 0),
        (1, 0),
        (0, 1),
        (-1, 0),
        (0, -1),
        (1, 1),
        (-1, 1),
        (-1, -1),
        (1, -1),
    ],
    w=[4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36],
)

D3Q19 = _build(
    "D3Q19",
    c=[
        (0, 0, 0),
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
        (1, 1, 0),
        (-1, -1, 0),
        (1, -1, 0),
        (-1, 1, 0),
        (1, 0, 1),
        (-1, 0, -1),
        (1, 0, -1),
        (-1, 0, 1),
        (0, 1, 1),
        (0, -1, -1),
        (0, 1, -1),
        (0, -1, 1),
    ],
    w=[
        1 / 3,
        1 / 18,
        1 / 18,
        1 / 18,
        1 / 18,
        1 / 18,
        1 / 18,
        1 / 36,
        1 / 36,
        1 / 36,
        1 / 36,
        1 / 36,
        1 / 36,
        1 / 36,
        1 / 36,
        1 / 36,
        1 / 36,
        1 / 36,
        1 / 36,
    ],
)

_LATTICES = {"D2Q9": D2Q9, "D3Q19": D3Q19}


def get_lattice(name: str) -> Lattice:
    try:
        return _LATTICES[name.upper()]
    except KeyError:  # pragma: no cover - defensive
        raise KeyError(f"unknown lattice {name!r}; known: {sorted(_LATTICES)}") from None