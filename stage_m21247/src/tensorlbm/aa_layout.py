"""Port of the 2b-t/LB-t *A-A layout pattern* for microscopic populations.

Anchor ``XVAL-LBM-NO-MULTIPHASE-LAYOUT-PATTERN-ONLY``.  Upstream
`2b-t/LB-t <https://github.com/2b-t/LB-t>`_ has **no multiphase model** and its
collision operator is a stub (``include/lbt/populations/collision/bgk.hpp``
ends with ``// TODO(tobit): Implement collide-stream``).  The only mechanism
that is actually specified *and* executable upstream is the **population
layout pattern**::

    template <class LT, std::int32_t NP = 1>
    using Population = AaPopulation<LT, NP>;        // population.hpp:19

with one merged array of ``NZ*NY*NX*NP*ND`` entries and the linear index
(``include/lbt/populations/indexing/indexing.hpp``)::

    (((z*NY + y)*NX + x)*NP + p)*ND + n*OFF + d

* ``NP``  — number of populations stored *side by side* inside a cell block
  (the component slot); upstream only ever instantiates ``NP == 1``,
* ``n``   — 0 for the "positive velocity" half, 1 for the "negative" half,
* ``d``   — index inside the half, ``OFF = ND/2``,
* ``ND``  — ``SPEEDS + PAD``: the *padded* population count.  For the upstream
  default D2Q9 (``D2Q9P10``) that is ``ND=10``, ``SPEEDS=9``, ``PAD=1``,
  ``OFF=5``; the padding slot has ``MASK == 0`` and is never read or written,
* parity — the "A-A" (single buffer) access pattern: ``indexRead`` /
  ``indexWrite`` are defined per odd/even time step and differ by a half-index
  flip (``n -> !n``) and by a one-cell offset along the direction of travel.

What upstream actually *pins down* (and this module reproduces bit-exactly,
verified against a program compiled from the unmodified upstream headers, see
``benchmarks/xval_lbt_layout_pattern/lbt_ref_dump.cpp`` and
``artifacts/xval_lbt_layout/lbt_ref_layout.json``):

1. the linear index formula and its inverse ``linearToSpatial``,
2. the parity accessors ``indexRead<Even/Odd>`` and ``indexWrite<Even/Odd>``;
   upstream's own unit test (``test/populations/population_base_test.hpp``)
   asserts the invariant ``read<!TS>(x,n,d) == write<TS>(x,n,d)``, i.e. the
   parity pair is **transport-free**: the one-cell offset present in both
   accessors cancels.  Transport is supposed to live in the never-written
   collide-stream kernel,
3. the NP stacking (component index ``p`` inside the cell block, stride ``ND``).

What upstream does **not** provide, and this module therefore supplies with an
explicit, documented convention (:meth:`AaPopulation.push`): the actual lattice
transport.  The ported pattern is used here as the *storage layer*; the solver
runs :meth:`AaPopulation.step` (BGK collision + Guo body force + push) on top of
it and is validated (a) cell-by-cell against TensorLBM's production dense
D2Q9 kernels and (b) against the analytic Poiseuille parabola.

Array shape
-----------
``A`` has shape ``(NZ, NY, NX, NP, ND)``.  C-order raveling of that shape *is*
the upstream linear index, so the layout equivalence check is an identity, not
an approximation.  Note the layout is *not* performance-motivated: ``p`` sits
inside the cell block (stride ``ND`` between components) and the padding slot
wastes ``PAD/ND`` of the memory.  It is adopted because it is the OSS
convention a cross-validation can be pinned to.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "EVEN",
    "ODD",
    "D2Q9_ND",
    "D2Q9_OFF",
    "D2Q9_DX",
    "D2Q9_DY",
    "D2Q9_W",
    "D2Q9_MASK",
    "TL_TO_SLOT",
    "SLOT_TO_TL",
    "LATTICE_OPPOSITE",
    "AaLayout",
    "AaPopulation",
    "equilibrium_d2q9",
    "flip_half",
]

# ---------------------------------------------------------------------------
# Lattice tables copied verbatim from upstream ``include/lbt/lattices/D2Q9.hpp``
# (``D2Q9P10`` = alias ``D2Q9<T>``).  They are constants *of the OSS file*, not
# numbers invented here; ``tests/test_aa_layout_xval.py`` re-checks them against
# the JSON emitted by the compiled upstream headers.
# ---------------------------------------------------------------------------
D2Q9_SPEEDS = 9
D2Q9_PAD = 1
D2Q9_ND = D2Q9_SPEEDS + D2Q9_PAD  # 10
D2Q9_OFF = D2Q9_ND // 2  # 5

D2Q9_DX = np.array([0, 1, 0, 1, -1, 0, -1, 0, -1, 1], dtype=np.float64)
D2Q9_DY = np.array([0, 0, 1, 1, 1, 0, 0, -1, -1, -1], dtype=np.float64)
D2Q9_W = np.array(
    [
        4.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 36.0,
        1.0 / 36.0,
        4.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 36.0,
        1.0 / 36.0,
    ],
    dtype=np.float64,
)
D2Q9_MASK = np.array([1, 1, 1, 1, 1, 0, 1, 1, 1, 1], dtype=bool)

#: TensorLBM direction order (``tensorlbm.constants.D2Q9.c``) -> upstream slot.
#: TensorLBM: 0:(0,0) 1:(1,0) 2:(0,1) 3:(-1,0) 4:(0,-1) 5:(1,1) 6:(-1,1) 7:(-1,-1) 8:(1,-1)
#: upstream : 0:(0,0) 1:(1,0) 2:(0,1) 3:(1,1) 4:(-1,1) 5:(0,0)pad 6:(-1,0) 7:(0,-1) 8:(-1,-1) 9:(1,-1)
TL_TO_SLOT = np.array([0, 1, 2, 6, 7, 3, 4, 8, 9], dtype=np.int64)

SLOT_TO_TL = np.full(D2Q9_ND, -1, dtype=np.int64)
SLOT_TO_TL[TL_TO_SLOT] = np.arange(len(TL_TO_SLOT))


def _build_opposite() -> np.ndarray:
    c = np.stack([D2Q9_DX[TL_TO_SLOT], D2Q9_DY[TL_TO_SLOT]], axis=1).astype(np.int64)
    table = np.zeros(len(c), dtype=np.int64)
    for i in range(len(c)):
        table[i] = int(np.argmin(((c + c[i]) ** 2).sum(axis=1)))
    return table


#: Opposite direction table in TensorLBM direction order.
LATTICE_OPPOSITE = _build_opposite()

EVEN = 0
ODD = 1


def flip_half(slot: int, off: int = D2Q9_OFF) -> int:
    """``n -> !n`` exactly as upstream does it (``static_cast<std::int32_t>(!n)``)."""
    return slot + off if slot < off else slot - off


@dataclass(frozen=True)
class AaLayout:
    """Value object implementing the upstream index algebra.

    ``index = (((z*NY + y)*NX + x)*NP + p)*ND + n*OFF + d``
    """

    nx: int
    ny: int
    nz: int = 1
    npop: int = 1
    nd: int = D2Q9_ND
    off: int = D2Q9_OFF
    dx: np.ndarray = field(
        default_factory=lambda: D2Q9_DX.copy(), repr=False, compare=False
    )
    dy: np.ndarray = field(
        default_factory=lambda: D2Q9_DY.copy(), repr=False, compare=False
    )
    dz: np.ndarray = field(
        default_factory=lambda: np.zeros(D2Q9_ND, dtype=np.float64), repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.nx <= 0 or self.ny <= 0 or self.nz <= 0:
            raise ValueError(f"grid must be positive (got nx={self.nx}, ny={self.ny}, nz={self.nz})")
        if self.npop <= 0:
            raise ValueError(f"npop must be >= 1 (got {self.npop})")
        if self.nd != 2 * self.off:
            raise ValueError(f"upstream requires ND == 2*OFF (got nd={self.nd}, off={self.off})")

    @property
    def total_slots(self) -> int:
        return self.nz * self.ny * self.nx * self.npop * self.nd

    @property
    def shape(self) -> tuple[int, int, int, int, int]:
        return (self.nz, self.ny, self.nx, self.npop, self.nd)

    # -- scalar index algebra (indexing.hpp) -------------------------------
    def spatial_to_linear(self, x: int, y: int, z: int, n: int, d: int, p: int = 0) -> int:
        return int(
            (
                ((int(z) * self.ny + int(y)) * self.nx + int(x)) * self.npop + int(p)
            )
            * self.nd
            + int(n) * self.off
            + int(d)
        )

    def linear_to_spatial(self, index: int) -> tuple[int, int, int, int, int, int]:
        index = int(index)
        if index < 0 or index >= self.total_slots:
            raise IndexError(f"index {index} outside [0, {self.total_slots})")
        factor = self.nd * self.npop * self.nx * self.ny
        rest = index % factor
        z = index // factor
        factor = self.nd * self.npop * self.nx
        y = rest // factor
        rest %= factor
        factor = self.nd * self.npop
        x = rest // factor
        rest %= factor
        factor = self.nd
        p = rest // factor
        rest %= factor
        n = rest // self.off
        d = rest % self.off
        return int(x), int(y), int(z), int(n), int(d), int(p)

    # -- parity accessors (aa_pattern.hpp) --------------------------------
    def index_read(self, ts: int, x: int, y: int, z: int, n: int, d: int, p: int = 0) -> int:
        if ts == ODD:
            slot = (1 - n) * self.off + d
            xn = (self.nx + x + int(self.dx[slot])) % self.nx
            yn = (self.ny + y + int(self.dy[slot])) % self.ny
            zn = (self.nz + z + int(self.dz[slot])) % self.nz
            return self.spatial_to_linear(xn, yn, zn, n, d, p)
        if ts == EVEN:
            return self.spatial_to_linear(x, y, z, 1 - n, d, p)
        raise ValueError(f"unknown timestep parity {ts!r} (use EVEN or ODD)")

    def index_write(self, ts: int, x: int, y: int, z: int, n: int, d: int, p: int = 0) -> int:
        if ts == ODD:
            return self.spatial_to_linear(x, y, z, 1 - n, d, p)
        if ts == EVEN:
            slot = n * self.off + d
            xn = (self.nx + x - int(self.dx[slot])) % self.nx
            yn = (self.ny + y - int(self.dy[slot])) % self.ny
            zn = (self.nz + z - int(self.dz[slot])) % self.nz
            return self.spatial_to_linear(xn, yn, zn, n, d, p)
        raise ValueError(f"unknown timestep parity {ts!r} (use EVEN or ODD)")


def equilibrium_d2q9(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """Second-order D2Q9 equilibrium, ``rho.shape + (9,)``, TensorLBM direction order."""
    cu = D2Q9_DX[TL_TO_SLOT] * ux[..., None] + D2Q9_DY[TL_TO_SLOT] * uy[..., None]
    usq = (ux * ux + uy * uy)[..., None]
    w = D2Q9_W[TL_TO_SLOT]
    return w * rho[..., None] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)


class AaPopulation:
    """Multi-population store following the LB-t A-A layout pattern.

    Parameters
    ----------
    nx, ny, nz:
        Grid resolution (``nz == 1`` for 2-D).
    npop:
        Populations stacked side by side — the upstream template parameter
        ``NP``.  Upstream never instantiates ``NP > 1``; ``npop=2`` is used by
        the cross-validation case to carry a two-component column.
    """

    def __init__(self, nx: int, ny: int, nz: int = 1, npop: int = 1, dtype=np.float64) -> None:
        self.layout = AaLayout(nx=nx, ny=ny, nz=nz, npop=npop)
        self.dtype = dtype
        self.A = np.zeros(self.layout.shape, dtype=dtype)

    # -- metadata ---------------------------------------------------------
    @property
    def nx(self) -> int:
        return self.layout.nx

    @property
    def ny(self) -> int:
        return self.layout.ny

    @property
    def nz(self) -> int:
        return self.layout.nz

    @property
    def npop(self) -> int:
        return self.layout.npop

    @property
    def nd(self) -> int:
        return self.layout.nd

    # -- upstream scalar read/write (opaque across the whole address space) --
    def read(self, ts: int, x: int, y: int, z: int, n: int, d: int, p: int = 0) -> float:
        """Value at ``indexRead<ts>(x,y,z,n,d,p)`` (upstream semantics)."""
        return float(self.A.reshape(-1)[self.layout.index_read(ts, x, y, z, n, d, p)])

    def write(self, ts: int, x: int, y: int, z: int, n: int, d: int, p: int, value: float) -> None:
        """Store ``value`` at ``indexWrite<ts>(x,y,z,n,d,p)`` (upstream semantics)."""
        self.A.reshape(-1)[self.layout.index_write(ts, x, y, z, n, d, p)] = value

    # -- dense interchange (TensorLBM direction order) ---------------------
    def to_dense(self) -> np.ndarray:
        """Populations as a dense array ``(nz, ny, nx, npop, 9)``.

        Only the nine ``MASK == 1`` slots are exported; the padding slot is
        dropped exactly like a solver would never touch it.
        """
        return self.A[..., TL_TO_SLOT]

    def from_dense(self, f: np.ndarray) -> "AaPopulation":
        f = np.asarray(f, dtype=self.dtype)
        if f.ndim == 3:
            f = f[None, :, :, None, :]
        elif f.ndim == 4:
            f = f[:, :, :, None, :]
        if f.shape != (self.nz, self.ny, self.nx, self.npop, len(TL_TO_SLOT)):
            raise ValueError(
                f"dense shape {f.shape} incompatible with layout "
                f"{(self.nz, self.ny, self.nx, self.npop, len(TL_TO_SLOT))}"
            )
        self.A[..., TL_TO_SLOT] = f
        for slot in np.flatnonzero(~D2Q9_MASK[: self.nd]):
            self.A[..., int(slot)] = 0.0
        return self

    def macroscopic(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``rho, ux, uy`` per population, shape ``(nz, ny, nx, npop)``."""
        f = self.to_dense()
        rho = f.sum(axis=-1)
        ux = (f * D2Q9_DX[TL_TO_SLOT]).sum(axis=-1) / rho
        uy = (f * D2Q9_DY[TL_TO_SLOT]).sum(axis=-1) / rho
        return rho, ux, uy

    # -- the piece upstream leaves unimplemented: transport ---------------
    def push(self, fpost: np.ndarray) -> None:
        """Store post-collision populations *one cell upstream* (push scheme).

        TensorLBM's realization of the A-A pattern: the post-collision
        population of direction ``i`` is deposited at ``x_i - c_i`` so that
        :meth:`to_dense` sees it again at ``x_i`` after the one-cell offset
        encoded in the pattern's ``indexRead<Odd>`` accessor.  Upstream has no
        collide-stream (``collision/bgk.hpp:75`` is a ``TODO``), so this
        transport rule is *our* convention, validated against TensorLBM's
        production dense kernels and the analytic Poiseuille solution.

        ``fpost`` has shape ``(nz, ny, nx, npop, 9)``, TensorLBM direction order.
        """
        fpost = np.asarray(fpost, dtype=self.dtype)
        if fpost.shape != (self.nz, self.ny, self.nx, self.npop, len(TL_TO_SLOT)):
            raise ValueError(
                f"fpost shape {fpost.shape} incompatible with layout "
                f"{(self.nz, self.ny, self.nx, self.npop, len(TL_TO_SLOT))}"
            )
        A = self.A
        for i, slot in enumerate(TL_TO_SLOT):
            shifts = (int(self.layout.dz[int(slot)]), int(self.layout.dy[int(slot)]), int(self.layout.dx[int(slot)]))
            A[..., int(slot)] = np.roll(fpost[..., i], shifts, axis=(0, 1, 2))
        for slot in np.flatnonzero(~D2Q9_MASK[: self.nd]):
            A[..., int(slot)] = 0.0

    def copy(self) -> "AaPopulation":
        out = AaPopulation(self.nx, self.ny, self.nz, self.npop, dtype=self.dtype)
        out.A = self.A.copy()
        return out