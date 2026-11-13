"""List-based (sparse) D2Q9 LBM -- port of the ``lssfau/lbmpy`` ``lbmpy.sparse`` conventions.

Why this module exists
----------------------
TensorLBM's production kernels are *dense*: populations live in ``(Q, nx, ny)``
arrays and every cell of the bounding box is visited.  The OSS reference
(``lbmpy/sparse``, shipped as ``lbmpy/sparse/mapping.py`` in lbmpy 2.0) instead
keeps a **list of active cells** plus a **per-cell, per-direction neighbour
index array**, so that streaming becomes pure indirection in a 1-D index space
and solid cells are never stored at all.  This module ports that data layout
into TensorLBM so the two storage/traversal schemes can be cross-validated
against one another and against analytic solutions.

Reference conventions (taken from the OSS source, not from prose):

* ``lbmpy/sparse/mapping.py:84-104`` ``SparseLbMapper._assemble`` -- fluid cells
  come first in the list (``np.argwhere(flag & fluid_flag)``), optional
  non-fluid boundary cells are appended *after* them; the coordinate array is a
  packed struct of ``uint32`` fields.
* ``lbmpy/sparse/mapping.py:106-154`` ``create_index_array`` -- for every
  non-center direction the array stores a **flat pdf offset**
  ``pdf_index(cell, d) = cell + d * len(self)`` (``mapping.py:110-111``), i.e.
  the stride convention of the direction index is SoA.  The center direction is
  *not* stored, hence ``Q-1`` columns per cell.
* no-slip is resolved by **index aliasing** (``mapping.py:126``): if the pull
  neighbour is solid the entry points at the *own* cell in the opposite
  direction, ``pdf_index(own_cell, inverse_idx(direction))`` -- halfway
  bounce-back without a separate boundary kernel.
* periodicity is a coordinate wrap across the ghost layers
  (``mapping.py:128-140``): ``x_i == ghost-1 -> x_i += n_i - 2*ghost`` and
  ``x_i == n_i-ghost -> x_i -= n_i - 2*ghost``; no-slip is tested *before*
  periodicity.
* the rest direction is read from the own cell
  (``update_rule_sparse.py:86-95`` ``_list_substitutions``: direction ``i``
  reads index column ``i-1``, the rest population reads ``src(0)``).

The collision + streaming *generator* of the OSS module
(``create_lb_update_rule_sparse``) does not build against the pinned
pystencils 2.0 (see ``docs/XVAL-LBM-SPARSE-LIST-BASED-AOS-LBM.md``, section
"failure modes"); the update step below is therefore a NumPy re-implementation
*of the semantics above*, while the index array can be produced either by the
same algorithm as the OSS function (:func:`build_index_array`) or by handing in
the array built by ``lbmpy.sparse.SparseLbMapper`` verbatim.

``dense_collide_stream`` implements the *same discrete scheme* (BGK + Guo
forcing, halfway bounce-back, periodic wrap) with an independent, dense
traversal, so ``sparse`` vs ``dense`` agreement isolates the data-layout change
and nothing else.
"""

from __future__ import annotations

import numpy as np

# D2Q9 in the repo ordering used by ``tensorlbm.d2q9`` (x-first).
C_D2Q9 = np.array(
    [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]],
    dtype=np.int64,
)
# Same lattice in the lbmpy/lssfau ``LBStencil(D2Q9)`` ordering (y-first), used
# when the index array is produced by ``lbmpy.sparse.SparseLbMapper``.
C_D2Q9_LBMPY = np.array(
    [[0, 0], [0, 1], [0, -1], [-1, 0], [1, 0], [-1, 1], [1, 1], [-1, -1], [1, -1]],
    dtype=np.int64,
)

FLUID = 1
WALL = 2
GHOST = 0


def weights_for(stencil: np.ndarray) -> np.ndarray:
    """D2Q9 weights from the squared stencil norm (4/9, 1/9, 1/36)."""
    norms = (stencil**2).sum(axis=1)
    table = {0: 4.0 / 9.0, 1: 1.0 / 9.0, 2: 1.0 / 36.0}
    return np.array([table[int(n)] for n in norms], dtype=np.float64)


def opposite_indices(stencil: np.ndarray) -> np.ndarray:
    """Index of the reverse direction for every direction."""
    st = np.asarray(stencil)
    return np.array([int(np.where((st == -d).all(axis=1))[0][0]) for d in st], dtype=np.int64)


W_D2Q9 = weights_for(C_D2Q9)
OPP_D2Q9 = opposite_indices(C_D2Q9)


# --------------------------------------------------------------------------- #
# index-array construction (port of lbmpy/sparse/mapping.py:84-154)
# --------------------------------------------------------------------------- #
def build_index_array(
    flags: np.ndarray,
    *,
    fluid_flag: int = FLUID,
    no_slip_flag: int = WALL,
    ghost_layers: int = 1,
    layout: str = "soa",
    stencil: np.ndarray | None = None,
):
    """Map a flag grid to ``(coordinates, index_array)``.

    Parameters
    ----------
    flags:
        integer grid; bit ``fluid_flag`` marks active cells, ``no_slip_flag``
        marks solid cells.  Cells carrying neither flag are ghost/dummy cells
        (``ghost_layers`` wide at every periodic border).
    layout:
        ``"soa"`` (default, the OSS convention) stores flat offsets as
        ``cell + d*N``; ``"aos"`` stores ``cell*Q + d`` and is *inconsistent*
        with the OSS index array -- kept only to make that mismatch explicit.
    stencil:
        ``(Q, dim)`` direction table; defaults to the repo ordering.

    Returns
    -------
    coordinates: ``(N, dim)`` uint32.
    index_array: ``(N, Q-1)`` uint32; column ``d-1`` holds the flat offset of
        the population that streams into the cell along direction ``d``.
    """
    stencil = C_D2Q9 if stencil is None else np.asarray(stencil)
    if layout not in ("soa", "aos"):
        raise ValueError("layout must be 'soa' or 'aos'")
    q, dim = stencil.shape
    fluid = (flags & fluid_flag).astype(bool)
    solid = (flags & no_slip_flag).astype(bool)
    # OSS ``fluid_boundary_mask`` = other_boundary_mask | fluid_flag; the OSS
    # caller passes other_boundary_mask=0 (no dedicated boundary kernel), so
    # only active cells are directly addressable.
    domain = fluid

    coords = np.argwhere(fluid).astype(np.uint32)
    n_cells = coords.shape[0]
    shape = np.array(flags.shape, dtype=np.int64)
    lookup = {tuple(int(v) for v in c): i for i, c in enumerate(coords)}

    def flat(cell: int, d: int) -> np.uint32:
        off = cell + d * n_cells if layout == "soa" else cell * q + d
        return np.uint32(off)

    inverse = opposite_indices(stencil)
    index_array = np.zeros((n_cells, q - 1), dtype=np.uint32)
    for d in range(1, q):
        for cell, x in enumerate(coords.astype(np.int64)):
            nb = x - stencil[d]
            inside = bool(np.all(nb >= 0) and np.all(nb < shape))
            if inside and domain[tuple(nb)]:
                index_array[cell, d - 1] = flat(lookup[tuple(int(v) for v in nb)], d)
                continue
            if inside and solid[tuple(nb)]:  # no-slip *before* periodicity
                index_array[cell, d - 1] = flat(cell, inverse[d])
                continue
            wrapped = nb.copy()
            for k in range(dim):
                if wrapped[k] == ghost_layers - 1:
                    wrapped[k] += shape[k] - 2 * ghost_layers
                elif wrapped[k] == shape[k] - ghost_layers:
                    wrapped[k] -= shape[k] - 2 * ghost_layers
            if not (np.all(wrapped >= 0) and np.all(wrapped < shape)):
                raise ValueError(f"could not find neighbour for {x} direction {d}")
            key = tuple(int(v) for v in wrapped)
            if key not in lookup:
                raise ValueError(f"periodic wrap of {x} direction {d} hit a non-fluid cell")
            index_array[cell, d - 1] = flat(lookup[key], d)
    return coords, index_array


# --------------------------------------------------------------------------- #
# shared collision operator (identical in both arms)
# --------------------------------------------------------------------------- #
def equilibrium(rho, ux, uy, stencil: np.ndarray | None = None):
    """Second-order D2Q9 equilibrium, population axis last."""
    st = C_D2Q9 if stencil is None else np.asarray(stencil)
    w = weights_for(st)
    cu = np.multiply.outer(ux, st[:, 0]) + np.multiply.outer(uy, st[:, 1])
    usq = (ux * ux + uy * uy)[..., None]
    return rho[..., None] * w * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)


def _collide_guo(f, tau, force, stencil: np.ndarray | None = None):
    """BGK collision with Guo forcing; ``f`` has shape ``(N, Q)``."""
    st = C_D2Q9 if stencil is None else np.asarray(stencil)
    w = weights_for(st)
    fx, fy = force
    rho = f.sum(axis=-1)
    mx = f @ st[:, 0]
    my = f @ st[:, 1]
    ux = (mx + 0.5 * fx) / rho
    uy = (my + 0.5 * fy) / rho
    feq = equilibrium(rho, ux, uy, st)
    # Guo source term (Guo, Zheng & Shi 2002), full form:
    #   (1 - 1/(2 tau)) w_i [ 3 (c_i - u) . F + 9 (c_i . u)(c_i . F) ] / rho
    # The second (O(Ma F)) term is *not* optional: without it sum_i Lambda_i =
    # -3 (u . F)/rho and the forcing leaks/gains mass every step.
    cuf = st[None, :, 0] * fx + st[None, :, 1] * fy
    cdotu = st[None, :, 0] * ux[..., None] + st[None, :, 1] * uy[..., None]
    force_term = (
        w
        * (3.0 * (cuf - (ux[..., None] * fx + uy[..., None] * fy)) + 9.0 * cdotu * cuf)
        / rho[..., None]
    )
    omega = 1.0 / tau
    return f + omega * (feq - f) + (1.0 - 0.5 * omega) * force_term, rho, ux, uy


def velocity_from(f, force, stencil: np.ndarray | None = None):
    """Density and velocity with the Guo half-force correction."""
    st = C_D2Q9 if stencil is None else np.asarray(stencil)
    rho = f.sum(axis=-1)
    return rho, (f @ st[:, 0] + 0.5 * force[0]) / rho, (f @ st[:, 1] + 0.5 * force[1]) / rho


# --------------------------------------------------------------------------- #
# dense reference arm
# --------------------------------------------------------------------------- #
def dense_collide_stream(f, tau=0.8, force=(0.0, 0.0), periodic=(True, False),
                         stencil: np.ndarray | None = None):
    """One dense step (pull gather, then collide) with halfway bounce-back.

    ``f`` has shape ``(Q, L, H)`` and holds *post-collision* populations (the
    same state convention as :meth:`SparseListLBM.step`, which makes the two
    arms step-by-step comparable).  Walls are the planes ``y < 0`` and
    ``y >= H`` (no-slip), ``x``/``y`` are periodic where ``periodic`` is True.
    """
    st = C_D2Q9 if stencil is None else np.asarray(stencil)
    opp = opposite_indices(st)
    q = f.shape[0]
    length, height = f.shape[1], f.shape[2]
    g = np.empty_like(f)  # streamed-in populations (pre collision)
    g[0] = f[0]
    for d in range(1, q):
        cx, cy = int(st[d, 0]), int(st[d, 1])
        xs = (np.arange(length) - cx) % length if periodic[0] else np.arange(length) - cx
        ys = (np.arange(height) - cy) % height if periodic[1] else np.arange(height) - cy
        okx = (xs >= 0) & (xs < length)
        oky = (ys >= 0) & (ys < height)
        if not periodic[0] and not okx.all():
            raise ValueError("non-periodic x with an outflow neighbour is not supported")
        src = f[d][np.ix_(xs.clip(0, length - 1), ys.clip(0, height - 1))]
        ok = okx[:, None] & oky[None, :]
        # halfway bounce-back == index aliasing: own cell, opposite direction
        g[d] = np.where(ok, src, f[opp[d]])
    return _collide_guo(g.transpose(1, 2, 0), tau, force, st)[0].transpose(2, 0, 1)


# --------------------------------------------------------------------------- #
# sparse list-based AoS arm
# --------------------------------------------------------------------------- #
class SparseListLBM:
    """List-based D2Q9 solver: ``pdf`` is a flat ``Q*N`` array touched via ``index_array``.

    Parameters
    ----------
    flags: flag grid (see :func:`build_index_array`).
    tau, force: BGK relaxation time and constant body force (lattice units).
    layout: ``"soa"`` = OSS stride convention (``cell + d*N``); ``"aos"`` keeps
        one cell's populations contiguous (``cell*Q + d``) and then needs its
        own index array -- mixing the two is a silent addressing bug.
    index_array: optional pre-built array (e.g. from ``lbmpy.sparse``); must
        match ``layout`` and ``stencil``.
    stencil: direction table matching ``index_array``.
    """

    def __init__(
        self,
        flags: np.ndarray,
        tau: float = 0.8,
        force=(0.0, 0.0),
        layout: str = "soa",
        index_array: np.ndarray | None = None,
        stencil: np.ndarray | None = None,
    ):
        self.stencil = C_D2Q9 if stencil is None else np.asarray(stencil)
        self.weights = weights_for(self.stencil)
        self.opposite = opposite_indices(self.stencil)
        self.coords, idx = build_index_array(flags, layout=layout, stencil=self.stencil)
        if index_array is not None:  # allow injecting the OSS-built array
            if index_array.shape != idx.shape:
                raise ValueError("injected index array has the wrong shape")
            idx = np.asarray(index_array, dtype=np.uint32)
        self.index_array = idx
        self.layout = layout
        self.tau = float(tau)
        self.force = (float(force[0]), float(force[1]))
        self.n_cells = self.coords.shape[0]
        self.q = self.stencil.shape[0]
        self._write_offsets = np.stack(
            [self.flat_offsets(d) for d in range(self.q)], axis=1
        )  # (N, Q)
        self.pdf = np.zeros(self.n_cells * self.q, dtype=np.float64)
        self.reset()

    # -- addressing ------------------------------------------------------- #
    def flat_offsets(self, d: int) -> np.ndarray:
        cells = np.arange(self.n_cells)
        if self.layout == "soa":
            return cells + d * self.n_cells
        return cells * self.q + d

    # -- initialisation / IO ---------------------------------------------- #
    def reset(self, rho=1.0, ux=None, uy=None):
        rho_f = np.full(self.n_cells, rho, dtype=np.float64) if np.isscalar(rho) else np.asarray(rho, float)
        ux_f = np.zeros(self.n_cells) if ux is None else np.asarray(ux, float)
        uy_f = np.zeros(self.n_cells) if uy is None else np.asarray(uy, float)
        self.pdf = np.zeros(self.n_cells * self.q, dtype=np.float64)
        self.pdf[self._write_offsets] = equilibrium(rho_f, ux_f, uy_f, self.stencil)

    def populations(self) -> np.ndarray:
        """``(N, Q)`` view of the populations in list order."""
        return self.pdf[self._write_offsets]

    def dense_view(self, shape) -> np.ndarray:
        """Populations scattered onto the dense grid ``(Q, L, H)`` (ghost cells NaN)."""
        out = np.full((self.q,) + tuple(shape), np.nan)
        xs = self.coords[:, 0].astype(int)
        ys = self.coords[:, 1].astype(int)
        out[(slice(None), xs, ys)] = self.populations().T
        return out

    def macroscopic(self):
        return velocity_from(self.populations(), self.force, self.stencil)

    # -- time step --------------------------------------------------------- #
    def step(self):
        n = self.n_cells
        fpre = np.empty((n, self.q), dtype=np.float64)
        # rest direction: own cell (update_rule_sparse.py:95, ``src(0)``)
        fpre[:, 0] = self.pdf[self._write_offsets[:, 0]]
        # direction d reads index column d-1 (the pull source)
        for d in range(1, self.q):
            fpre[:, d] = self.pdf[self.index_array[:, d - 1]]
        fpost = _collide_guo(fpre, self.tau, self.force, self.stencil)[0]
        new = np.empty_like(self.pdf)
        new[self._write_offsets] = fpost
        self.pdf = new
        return self


# --------------------------------------------------------------------------- #
# analytic references
# --------------------------------------------------------------------------- #
def poiseuille_analytic(y, height, force_x, nu):
    """Halfway-wall plane Poiseuille profile; walls sit at ``y = -0.5`` and ``y = H-0.5``."""
    s = np.asarray(y, dtype=np.float64) + 0.5
    return force_x / (2.0 * nu) * s * (height - s)


def taylor_green_analytic(amplitude0, nu, k, t):
    """Decay of the 2-D Taylor-Green vortex: ``u = U0 exp(-2 nu k^2 t)``."""
    return amplitude0 * np.exp(-2.0 * nu * k * k * t)