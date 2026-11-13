"""XVAL-LBM-RASTER-OBSTACLE-CORNER-CLASS-BOUNCE-BACK -- comparison harness.

Reference implementation ("OSS" arm)
------------------------------------
``ebinan92/Fingering_dynamics`` (checkout ``150a573``, read-only at
``/tmp/Fingering_dynamics``), tag ``raster-obstacle-corner-class-bounce-back``
(platform OSS inventory ``docs/oss_inventory/OSS-INV-Fingering_dynamics.md``):

* raster geometry -> boolean mask + a *corner list* of rectangle bounding boxes
  (``lattice_boltzmann/create_block.py`` ``getRectangleblock`` uses
  ``cv2.rectangle``, i.e. both corners inclusive, then ``setblock`` /
  ``binary_fill_holes`` at ``create_block.py:395``);
* obstacle treatment = **corner-class half-way bounce-back**: eight barrier
  classes (N/S/E/W bands of fluid nodes hugging a rectangle + the four diagonal
  corner cells) are painted from the rectangle corners and a hand-written table
  reflects the directions that point into the rectangle
  (``lattice_boltzmann/bounce_back.py:25-86``, ``halfway_bounceback_rec``);
* streaming is a periodic ``np.roll`` sweep (``lattice_boltzmann/fingering.py:393``),
  called at ``fingering.py:551-560`` *after* ``f_behind = copy.deepcopy(f)`` so
  that the reflection reads the pre-stream (post-collision) population at the
  same fluid node;
* the channel walls are handled by ``bottom_top_wall`` (``fingering.py:432``):
  rows 0 and ``ny-1`` reflect directions {2,5,6} / {4,7,8} respectively.

The repository carries no LICENSE file, so nothing is vendored verbatim: the
arithmetic above is re-derived here (band/corner painting + per-direction
reflection table) and cited line-by-line instead.

Our arm ("mask" arm)
--------------------
``tensorlbm.bitflag_neighbour_mask``: a per-direction closed-link bitmask built
once from the raster mask (bit ``i-1`` set iff link ``x -> x - c_i`` is closed),
with streaming + half-way bounce-back driven by those bits
(``stream_mask_vector`` semantics: ``f_i(x) = f_opp(x)`` for a closed link).
The mask is exact: it closes *exactly* the links whose upstream node is solid,
so the eight corner classes fall out of the bit pattern instead of being painted
by hand.

Both arms are driven by the **same** single-phase D2Q9 BGK collision with Guo
second-order body forcing, the same raster geometry and the same measurement
code, so the only difference left is the bounce-back link set.

Layout everywhere: ``f`` has shape ``(9, nx, ny)`` with ``D2Q9_C`` ordering
``1=E, 2=N, 3=W, 4=S, 5=NE, 6=NW, 7=SW, 8=SE`` (identical to the OSS ``e``
ordering in ``fingering.py``; ``D2Q9_C`` of ``tensorlbm.bitflag_neighbour_mask``).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from tensorlbm.bitflag_neighbour_mask import (
    D2Q9_C as _TL_C,
    D2Q9_W as _TL_W,
)

#: D2Q9 stencil in the **OSS direction order** of ``fingering.py`` /
#: ``validation.py`` (``e[1..8] = cos/sin of (i-1)*pi/2`` then the diagonals from
#: ``(i-5)*pi/2 + pi/4``): 1=E, 2=N, 3=W, 4=S, 5=NE, 6=NW, 7=SW, 8=SE.  The
#: bounce-back table below is written in this order so it can be read side by
#: side with ``bounce_back.py:48-86``.
D2Q9_C = np.array(
    [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]],
    dtype=np.int64,
)
D2Q9_W = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4, dtype=np.float64)
_OPP_INDEX = {tuple(int(v) for v in row): i for i, row in enumerate(D2Q9_C.tolist())}
D2Q9_OPP = np.array([_OPP_INDEX[tuple(-int(v) for v in row)] for row in D2Q9_C.tolist()],
                    dtype=np.int64)

#: the stencil is the module stencil with a permutation of indices 1..8 only:
#: same speed set and same weights (checked, not asserted from prose)
assert sorted(map(tuple, D2Q9_C.tolist())) == sorted(map(tuple, _TL_C.tolist()))
assert np.allclose(sorted(D2Q9_W.tolist()), sorted(_TL_W.tolist()))


# ---------------------------------------------------------------------------
# geometry (raster rectangles, OSS pattern)
# ---------------------------------------------------------------------------
def oss_rect_corner_list(nx: int, ny: int, b: int):
    """Staggered rectangle array in the OSS main-case arrangement.

    ``fingering.py:532-545`` builds columns of ``b x b`` blocks with a stride of
    ``2b`` in x, alternating the vertical offset by one block (even columns at
    ``60*(i+1) + i*20``, odd columns at ``60*i + (i+1)*20`` for ``b = 20``, i.e.
    vertical period ``4b`` and offset ``2b``).  The pattern is reproduced here
    parameterised by ``b`` inside an ``nx x ny`` box, keeping at least one
    ``b``-wide fluid margin from every border so no block touches a wall row.
    """
    rects = []
    period = 4 * b
    offset = 2 * b
    k = 0
    while True:
        x0 = b + 2 * b * k
        x1 = x0 + b - 1
        if x1 > nx - b:
            break
        y = (offset if (k % 2 == 1) else offset + b)
        while y + b - 1 <= ny - b:
            rects.append(((x0, y), (x1, y + b - 1)))
            y += period
        k += 1
    return rects


def raster_solid_mask(nx: int, ny: int, rects) -> np.ndarray:
    """Boolean ``(nx, ny)`` solid mask, corners inclusive (OSS ``cv2.rectangle``)."""
    solid = np.zeros((nx, ny), dtype=bool)
    for (x0, y0), (x1, y1) in rects:
        solid[x0 : x1 + 1, y0 : y1 + 1] = True
    return solid


def exact_closed_links(solid: np.ndarray) -> np.ndarray:
    """Exact closed-link bits ``(nx, ny)`` uint8: bit ``i-1`` iff ``x - c_i`` is solid.

    Same rule as ``tensorlbm.bitflag_neighbour_mask.neighbour_mask`` (closed iff
    the upstream cell is not open), with periodic x and closed y: the rows
    outside the domain are the channel walls.
    """
    nx, ny = solid.shape
    mask = np.zeros((nx, ny), dtype=np.uint8)
    for i in range(1, 9):
        cx, cy = int(D2Q9_C[i, 0]), int(D2Q9_C[i, 1])
        xs = (np.arange(nx) - cx) % nx
        ys = np.arange(ny) - cy
        closed = np.ones((nx, ny), dtype=bool)
        inside = (ys >= 0) & (ys < ny)
        closed[:, inside] = solid[np.ix_(xs, ys[inside])]
        mask |= (closed.astype(np.uint8) << np.uint8(i - 1))
    return mask


# ---------------------------------------------------------------------------
# OSS corner-class bounce-back table
# ---------------------------------------------------------------------------
#: reflection table of ``halfway_bounceback_rec`` (``bounce_back.py:48-86``):
#: direction -> [(class, source direction)], in the OSS assignment order.
OSS_TABLE = {
    1: [("e", 3)],
    2: [("n", 4)],
    3: [("w", 1)],
    4: [("s", 2)],
    5: [("ne", 7), ("n", 7), ("e", 7)],
    6: [("nw", 8), ("n", 8), ("w", 8)],
    7: [("sw", 5), ("s", 5), ("w", 5)],
    8: [("se", 6), ("s", 6), ("e", 6)],
}


def corner_class_masks(nx: int, ny: int, rects) -> dict[str, np.ndarray]:
    """Paint the eight OSS barrier classes from the rectangle corner list.

    Transcribes ``bounce_back.py:33-46`` (+1 row above the top edge, -1 row below
    the bottom edge, -1/+1 column beside the left/right edges, and the four
    diagonal corner cells), with the class names of the reflection table.
    Out-of-domain indices are dropped here (the OSS code would wrap them
    silently) -- reported as a porting note, our geometry keeps a ``b``-wide
    margin so no index can be out of range.
    """
    masks = {k: np.zeros((nx, ny), dtype=bool) for k in ("n", "s", "e", "w", "nw", "ne", "sw", "se")}
    dropped = 0
    for (x0, y0), (x1, y1) in rects:
        if not (1 <= x0 - 1 and x1 + 1 <= nx - 2 and 1 <= y0 - 1 and y1 + 1 <= ny - 2):
            dropped += 1
        masks["n"][x0 : x1 + 1, y1 + 1] = True
        masks["s"][x0 : x1 + 1, y0 - 1] = True
        masks["w"][x0 - 1, y0 : y1 + 1] = True
        masks["e"][x1 + 1, y0 : y1 + 1] = True
        masks["nw"][x0 - 1, y1 + 1] = True
        masks["ne"][x1 + 1, y1 + 1] = True
        masks["sw"][x0 - 1, y0 - 1] = True
        masks["se"][x1 + 1, y0 - 1] = True
    masks["_dropped_rects_out_of_range"] = dropped
    return masks


def oss_bounced_links(nx: int, ny: int, rects) -> np.ndarray:
    """Bounced-link bits of the OSS table, same encoding as the exact mask."""
    masks = corner_class_masks(nx, ny, rects)
    bounced = np.zeros((nx, ny), dtype=np.uint8)
    for i, entries in OSS_TABLE.items():
        sel = np.zeros((nx, ny), dtype=bool)
        for cls, _src in entries:
            sel |= masks[cls]
        bounced |= (sel.astype(np.uint8) << np.uint8(i - 1))
    return bounced, masks


def link_difference(solid: np.ndarray, rects):
    """Enumerate the link-level difference between the table and the exact mask.

    Only *obstacle* links are compared: fluid nodes with ``1 <= y <= ny-2`` (the
    first/last rows carry the channel-wall links, which the OSS handles with
    ``bottom_top_wall`` instead of the corner table, and solid nodes own no
    physical link at all).
    """
    nx, ny = solid.shape
    exact = exact_closed_links(solid)
    table, masks = oss_bounced_links(nx, ny, rects)
    relevant = np.zeros((nx, ny), dtype=bool)
    relevant[:, 1 : ny - 1] = True
    fluid = (~solid) & relevant
    spurious = np.zeros_like(table)  # bounced by the table but open upstream
    for i in range(1, 9):
        bit = np.uint8(1 << (i - 1))
        t = (table & bit) != 0
        e = (exact & bit) != 0
        spurious |= ((t & ~e & fluid) << np.uint8(i - 1)).astype(np.uint8)
    missed = np.zeros_like(table)
    for i in range(1, 9):
        bit = np.uint8(1 << (i - 1))
        t = (table & bit) != 0
        e = (exact & bit) != 0
        missed |= ((e & ~t & fluid) << np.uint8(i - 1)).astype(np.uint8)
    def count(m):
        """bit count restricted to the obstacle-link nodes (fluid, interior rows)"""
        return int(sum(bin(int(v)).count("1") for v in m.ravel()))

    def count_fluid(m):
        return int(sum(bin(int(v)).count("1") for v in (m[fluid]).ravel()))

    return {
        "exact_links": count_fluid(exact),
        "table_links": count_fluid(table),
        "spurious_links": count_fluid(spurious),
        "missed_links": count_fluid(missed),
        "spurious_mask": spurious,
        "missed_mask": missed,
        "exact_mask": exact,
        "table_mask": table,
        "class_masks": masks,
    }


# ---------------------------------------------------------------------------
# common stepper (single-phase D2Q9 BGK + Guo body force), arm-specific streaming
# ---------------------------------------------------------------------------
def equilibrium(rho, ux, uy) -> np.ndarray:
    usqr = 1.5 * (ux * ux + uy * uy)
    feq = np.empty((9,) + rho.shape, dtype=np.float64)
    for i in range(9):
        cx, cy = int(D2Q9_C[i, 0]), int(D2Q9_C[i, 1])
        cu = 3.0 * (cx * ux + cy * uy)
        feq[i] = D2Q9_W[i] * rho * (1.0 + cu + 0.5 * cu * cu - usqr)
    return feq


def guo_force(rho, ux, uy, gx, gy, omega) -> np.ndarray:
    out = np.empty((9,) + rho.shape, dtype=np.float64)
    ug = ux * gx + uy * gy
    for i in range(9):
        cx, cy = int(D2Q9_C[i, 0]), int(D2Q9_C[i, 1])
        cg = cx * gx + cy * gy
        out[i] = (1.0 - 0.5 * omega) * D2Q9_W[i] * rho * (
            3.0 * (cg - ug) + 9.0 * (cx * ux + cy * uy) * cg
        )
    return out


@dataclass
class Case:
    nx: int = 132
    ny: int = 144
    b: int = 12
    tau: float = 2.0
    g: float = 1e-3
    steps: int = 20000
    check_every: int = 1000
    u_rel_tol: float = 1e-7
    blocks: bool = True
    seed: int = 0

    @property
    def omega(self) -> float:
        return 1.0 / self.tau

    @property
    def nu(self) -> float:
        return (self.tau - 0.5) / 3.0


class Arm:
    """One bounce-back operator on the shared stepper."""

    def __init__(self, case: Case, kind: str):
        self.c = case
        self.kind = kind
        nx, ny = case.nx, case.ny
        self.rects = oss_rect_corner_list(nx, ny, case.b) if case.blocks else []
        self.solid = raster_solid_mask(nx, ny, self.rects) if case.blocks else np.zeros((nx, ny), bool)
        self.fluid = ~self.solid
        self.exact_mask = exact_closed_links(self.solid)
        self.table_mask, self.class_masks = oss_bounced_links(nx, ny, self.rects)
        self.mask = self.exact_mask if kind == "mask" else self.table_mask
        # wall links: the closed links at the first/last row are the ones whose
        # *upstream* cell is outside the domain -- dirs 2/5/6 at y = 0 and
        # 4/7/8 at y = ny-1 (pull convention; identical to the OSS
        # ``bottom_top_wall`` assignments f2<-f4, f5<-f7, f6<-f8 at row 0,
        # fingering.py:432-447)
        wall = np.zeros((nx, ny), dtype=np.uint8)
        for i in (2, 5, 6):
            wall[:, 0] |= np.uint8(1 << (i - 1))
        for i in (4, 7, 8):
            wall[:, ny - 1] |= np.uint8(1 << (i - 1))
        self.wall_mask = wall
        # consistency: the exact-link constructor already closes the out-of-domain
        # upstream links, so the wall must be a subset of the exact mask
        assert not np.any((wall & ~self.exact_mask) != 0)
        self.mask_all = self.mask | wall
        # index tables for periodic-x / clipped-y pull streaming
        self._xs = [(np.arange(nx) - int(D2Q9_C[i, 0])) % nx for i in range(9)]
        self._ys = [np.clip(np.arange(ny) - int(D2Q9_C[i, 1]), 0, ny - 1) for i in range(9)]
        self._seeded = False

    # -- streaming ---------------------------------------------------------
    def stream(self, f_post: np.ndarray, use_mask: bool) -> np.ndarray:
        c = self.c
        nx, ny = c.nx, c.ny
        if use_mask:
            out = np.empty_like(f_post)
            out[0] = f_post[0]
            for i in range(1, 9):
                pulled = f_post[i][self._xs[i][:, None], self._ys[i][None, :]]
                bounced = f_post[D2Q9_OPP[i]]
                bit = np.uint8(1 << (i - 1))
                closed = (self.mask_all & bit) != 0
                out[i] = np.where(closed, bounced, pulled)
            out[:, self.solid] = f_post[:, self.solid]
            return out
        # OSS path: periodic roll everywhere, then the corner-class table, then walls
        out = np.empty_like(f_post)
        for i in range(9):
            cx, cy = int(D2Q9_C[i, 0]), int(D2Q9_C[i, 1])
            out[i] = np.roll(np.roll(f_post[i], cx, axis=0), cy, axis=1)
        for i, entries in OSS_TABLE.items():
            for cls, src in entries:
                sel = self.class_masks[cls]
                out[i][sel] = f_post[src][sel]
        # bottom_top_wall (applied to every column: our domain is periodic in x,
        # the OSS main run uses the two end columns as inlet/outlet)
        out[2][:, 0] = f_post[4][:, 0]
        out[5][:, 0] = f_post[7][:, 0]
        out[6][:, 0] = f_post[8][:, 0]
        out[4][:, ny - 1] = f_post[2][:, ny - 1]
        out[7][:, ny - 1] = f_post[5][:, ny - 1]
        out[8][:, ny - 1] = f_post[6][:, ny - 1]
        return out

    # -- diagnostics -------------------------------------------------------
    def macros(self, f):
        c = self.c
        rho = f.sum(axis=0)
        mx = np.einsum("i,ixy->xy", D2Q9_C[:, 0].astype(float), f) + 0.5 * c.g * rho
        safe = np.where(rho > 0.0, rho, 1.0)
        return rho, mx / safe

    def obstacle_force(self, f_post: np.ndarray) -> float:
        """Momentum-exchange force on the obstacles (link set of *this* arm).

        F_x = sum over bounced obstacle links of 2 c_ix f_opp (Ladd convention,
        static wall, pre-stream/post-collision populations).
        """
        fx = 0.0
        for i in range(1, 9):
            bit = np.uint8(1 << (i - 1))
            sel = ((self.mask & bit) != 0) & self.fluid
            if not sel.any():
                continue
            cx = int(D2Q9_C[i, 0])
            if cx == 0:
                continue
            fx += cx * 2.0 * float(f_post[D2Q9_OPP[i]][sel].sum())
        return fx

    def init_field(self) -> np.ndarray:
        """Rest state (rho = 1, u = 0) in both arms -- the same deterministic seed.

        Starting from rest keeps the transient physical (no analytic solution is
        injected into the compared fields) and identical for the two operators.
        """
        c = self.c
        zeros = np.zeros((c.nx, c.ny))
        return equilibrium(np.ones((c.nx, c.ny)), zeros, zeros)

    def run(self) -> dict:
        c = self.c
        f = self.init_field()
        history = []
        converged_at = None
        prev_u = None
        u_ref_hist = []
        for step in range(1, c.steps + 1):
            rho, ux = self.macros(f)
            feq = equilibrium(rho, ux, np.zeros_like(ux))
            f_post = f + c.omega * (feq - f)
            f_post = f_post + guo_force(rho, ux, np.zeros_like(ux), c.g, 0.0, c.omega)
            f_post[:, self.solid] = f[:, self.solid]
            f = self.stream(f_post, use_mask=(self.kind == "mask"))
            if step % c.check_every == 0 or step == 1:
                rho, ux = self.macros(f)
                u_bar = float(ux[self.fluid].mean())
                k_lat = c.nu * u_bar / c.g
                fx = self.obstacle_force(f_post)
                u_field = ux.copy()
                change = None
                if prev_u is not None:
                    change = float(np.abs(u_field - prev_u).max() / max(abs(u_bar), 1e-30))
                prev_u = u_field
                history.append(
                    dict(step=step, u_bar=u_bar, k_lattice=k_lat, fx_lattice=fx, rel_change=change)
                )
                if change is not None and change < c.u_rel_tol and converged_at is None:
                    converged_at = step
                    break
        rho, ux = self.macros(f)
        return dict(
            case=asdict(c) | {"omega": c.omega, "nu": c.nu},
            arm=self.kind,
            n_blocks=len(self.rects),
            solid_fraction=float(self.solid.mean()),
            stopped_at=step,
            converged_at=converged_at,
            history=history,
            ux=ux,
            rho=rho,
            mask=self.mask,
            solid=self.solid,
        )


# ---------------------------------------------------------------------------
# analytic anchor: empty channel, link-midpoint walls
# ---------------------------------------------------------------------------
def analytic_channel(case: Case) -> dict:
    """Discrete steady state of the BGK channel with walls at ``y = -1/2``, ``ny-1/2``.

    ``u(y) = g/(2 nu) (y+1/2)(ny-1/2-y)`` and therefore
    ``u_bar = (g/(2 nu)) (ny^2/6 + 1/12)``, ``K = nu u_bar / g = ny^2/12 + 1/24``.
    The closed form is re-evaluated numerically here (no hand-typed numbers).
    """
    ny = case.ny
    y = np.arange(ny) + 0.5
    u = case.g / (2.0 * case.nu) * y * (ny - y)
    u_bar = float(u.mean())
    return dict(u_profile=u, u_bar=u_bar, k_lattice=case.nu * u_bar / case.g)


def save_run(out_dir: Path, res: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "fields.npz",
        ux=res["ux"],
        rho=res["rho"],
        mask=res["mask"],
        solid=res["solid"],
        history=np.array(
            [
                (h["step"], h["u_bar"], h["k_lattice"], h["fx_lattice"],
                 -1.0 if h["rel_change"] is None else h["rel_change"])
                for h in res["history"]
            ]
        ),
    )
    meta = {k: v for k, v in res.items() if k not in ("ux", "rho", "mask", "solid")}
    (out_dir / "metrics.json").write_text(json.dumps(meta, indent=2))