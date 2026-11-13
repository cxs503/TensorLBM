"""Lean NumPy D2Q9 solver whose boundary treatment is driven by the
**bitflag neighbour mask** of :mod:`tensorlbm.bitflag_neighbour_mask`.

Purpose
-------
This is the runnable half of the XVAL against ``Maarten-vd-Sande/lbm``: the same
discrete scheme is advanced twice, once with the OSS-style *scan* boundary
treatment (8 neighbour-type tests per open node, ``lbm/streamcollide.py:63``) and
once with the precomputed per-direction neighbour mask, so the two paths can be
compared bit-for-bit and against physics references.

Discrete scheme (identical in both modes)
-----------------------------------------
* D2Q9, second-order equilibrium, BGK with ``omega = 1/tau``;
* body force through Guo's second-order forcing with the half-force velocity
  correction, so the steady plane-Poiseuille profile is the analytic parabola
  ``u(y) = g (y - y_w) (H - (y - y_w)) / (2 nu)`` with ``y_w = 1/2``;
* half-way bounce-back on closed links (staircase wall at the link midpoint,
  exactly the OSS operator ``f_i <- f_opp(i)`` at the fluid node), with the
  optional moving-wall momentum term;
* inlet: Zou-He velocity closure with the non-equilibrium bounce-back rule
  (own derivation, validated against the analytic inflow-driven channel);
* outlet: Zou-He pressure closure (``rho = 1``) or OSS-faithful zero-gradient
  (``lbm/streamcollide.py:210-217``) -- selectable, because that difference is
  one of the documented porting trade-offs.

Force measurement uses the momentum-exchange method over the closed links.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .bitflag_neighbour_mask import (
    CellFlag,
    D2Q9_C,
    D2Q9_OPP,
    D2Q9_W,
    StreamStats,
    neighbour_mask,
    open_cells,
    stream_mask_vector,
    stream_scan_vector,
)

__all__ = [
    "BitflagD2Q9",
    "make_channel",
    "make_cylinder_channel",
    "poiseuille_profile",
    "couette_profile",
]

MODE_MASK = "mask"
MODE_SCAN = "scan"


def _feq(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """Second-order D2Q9 equilibrium, shape ``(9, nx, ny)``."""
    usqr = 1.5 * (ux * ux + uy * uy)
    out = np.empty((9,) + rho.shape, dtype=np.float64)
    for i in range(9):
        cx, cy = int(D2Q9_C[i, 0]), int(D2Q9_C[i, 1])
        cu = 3.0 * (cx * ux + cy * uy)
        out[i] = D2Q9_W[i] * rho * (1.0 + cu + 0.5 * cu * cu - usqr)
    return out


def _guo_force(rho, ux, uy, gx, gy, omega) -> np.ndarray:
    """Guo (1997) second-order force term, ``(1 - omega/2)`` split included."""
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
class BitflagD2Q9:
    """D2Q9 cell-based solver with a selectable boundary kernel.

    ``cell_type`` holds :class:`CellFlag` values: ``INLET`` cells in column 0 and
    ``OUTLET`` cells in the last column are handled by the closures below;
    ``OBSTACLE`` cells are frozen and never read.
    """

    cell_type: np.ndarray
    tau: float
    mode: str = MODE_MASK
    body_force: tuple[float, float] = (0.0, 0.0)
    inlet_profile: np.ndarray | None = None  # u_x(y) prescribed on INLET cells
    outlet_mode: str = "zou_he"  # "zou_he" | "zero_gradient"
    u_wall_x: np.ndarray | None = None
    rho0: float = 1.0
    f: np.ndarray = field(init=False)
    mask: np.ndarray = field(init=False)
    steps: int = field(default=0, init=False)
    last_stream_stats: StreamStats | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.cell_type = np.asarray(self.cell_type).astype(np.int32)
        if self.cell_type.ndim != 2:
            raise ValueError("cell_type must be 2-D (nx, ny)")
        if self.mode not in (MODE_MASK, MODE_SCAN):
            raise ValueError(f"unknown mode {self.mode!r}")
        if self.outlet_mode not in ("zou_he", "zero_gradient", "zero_gradient_all", "outflow"):
            raise ValueError(f"unknown outlet_mode {self.outlet_mode!r}")
        self.omega = 1.0 / float(self.tau)
        self.mask = neighbour_mask(self.cell_type)
        self.op = open_cells(self.cell_type)
        self.nx, self.ny = self.cell_type.shape
        rho = np.where(self.op, self.rho0, 0.0)
        self.f = _feq(rho, np.zeros_like(rho), np.zeros_like(rho))
        # precompute c_i . u_w for the moving-wall term, shape (9, nx, ny)
        if self.u_wall_x is not None:
            uw = np.asarray(self.u_wall_x, dtype=np.float64)
            self._wall_dot = np.stack(
                [int(D2Q9_C[i, 0]) * uw for i in range(9)]
            )
        else:
            self._wall_dot = None

    # -- diagnostics -------------------------------------------------------
    @property
    def nu(self) -> float:
        return (1.0 / self.omega - 0.5) / 3.0

    def macros(self, f: np.ndarray | None = None):
        """Return ``(rho, ux, uy)`` with the half-force velocity correction."""
        f = self.f if f is None else f
        rho = f.sum(axis=0)
        gx, gy = self.body_force
        safe = np.where(rho > 0.0, rho, 1.0)
        mx = np.einsum("i,ixy->xy", D2Q9_C[:, 0].astype(float), f) + 0.5 * gx * rho
        my = np.einsum("i,ixy->xy", D2Q9_C[:, 1].astype(float), f) + 0.5 * gy * rho
        return rho, mx / safe, my / safe

    # -- closures ----------------------------------------------------------
    def _apply_inlet(self, f: np.ndarray) -> None:
        """Zou-He velocity closure on the inlet column (non-equilibrium bounce-back).

        With the outgoing populations known, the closure ``f_i = f_opp(i) +
        rho * 6 * w_i * (c_i . u)`` for the three ``+x`` directions gives
        ``rho = (sum_known f + f3 + f4 + f5) / (1 - u_x)``; both steps are
        re-derived in ``docs/XVAL-BITFLAG-NEIGHBOUR-MASK.md`` and validated
        against the analytic inflow-driven channel.
        """
        if self.inlet_profile is None:
            return
        x = 0
        ux_y = np.asarray(self.inlet_profile, dtype=np.float64)
        for y in range(self.ny):
            if not (int(self.cell_type[x, y]) & int(CellFlag.INLET)):
                continue
            ux = float(ux_y[min(y, ux_y.size - 1)])
            known = (
                f[0, x, y] + f[1, x, y] + f[2, x, y] + f[3, x, y] + f[4, x, y] + f[5, x, y]
            )
            rho = (known + f[3, x, y] + f[4, x, y] + f[5, x, y]) / (1.0 - ux)
            f[6, x, y] = f[3, x, y] + (2.0 / 3.0) * rho * ux
            f[7, x, y] = f[5, x, y] + (1.0 / 6.0) * rho * ux
            f[8, x, y] = f[4, x, y] + (1.0 / 6.0) * rho * ux

    def _apply_outlet(self, f: np.ndarray) -> None:
        x = self.nx - 1
        rho_out = self.rho0
        for y in range(self.ny):
            if not (int(self.cell_type[x, y]) & int(CellFlag.OUTLET)):
                continue
            if self.outlet_mode == "outflow":
                # Populations whose link leaves the domain are discarded (the
                # push-scheme equivalent of "the population left the domain");
                # bouncing them back instead makes mass pile up at the outlet.
                for i in range(1, 9):
                    if self.mask[x, y] & np.uint8(1 << (i - 1)):
                        f[i, x, y] = 0.0
                for i in (3, 4, 5):
                    f[i, x, y] = f[i, x - 1, y]
                continue
            if self.outlet_mode in ("zero_gradient", "zero_gradient_all"):
                # OSS set_outlet copies the incoming (-x) populations only
                # (lbm/streamcollide.py:210-217); "zero_gradient_all" also copies
                # the outgoing ones, which makes the column flux-conserving.
                dirs = (3, 4, 5) if self.outlet_mode == "zero_gradient" else range(1, 9)
                for i in dirs:
                    f[i, x, y] = f[i, x - 1, y]
                continue
            # density equation with rho prescribed gives the normal velocity:
            # rho*ux = f0+f1+f2 + 2(f6+f7+f8) - rho   (derivation in the docs)
            ux = (
                f[0, x, y] + f[1, x, y] + f[2, x, y]
                + 2.0 * (f[6, x, y] + f[7, x, y] + f[8, x, y])
                - rho_out
            ) / rho_out
            uy = 3.0 * (f[2, x, y] - f[1, x, y]) / (2.0 * rho_out)
            f[3, x, y] = f[6, x, y] - (2.0 / 3.0) * rho_out * ux
            f[4, x, y] = f[8, x, y] - (1.0 / 6.0) * rho_out * (ux + uy)
            f[5, x, y] = f[7, x, y] - (1.0 / 6.0) * rho_out * (ux - uy)

    # -- one lattice step --------------------------------------------------
    def step(self) -> StreamStats:
        rho, ux, uy = self.macros()
        feq = _feq(rho, ux, uy)
        gx, gy = self.body_force
        f_post = self.f + self.omega * (feq - self.f)
        if gx != 0.0 or gy != 0.0:
            f_post = f_post + _guo_force(rho, ux, uy, gx, gy, self.omega)
        # closed cells never stream
        f_post = np.where(self.op[None, :, :], f_post, self.f)
        stats = StreamStats()
        if self.mode == MODE_MASK:
            f_new = stream_mask_vector(
                f_post,
                self.cell_type,
                self.mask,
                wall_velocity=self._wall_dot,
                stats=stats,
            )
        else:
            f_new = stream_scan_vector(
                f_post,
                self.cell_type,
                wall_velocity=self._wall_dot,
                stats=stats,
            )
        f_new = np.where(self.op[None, :, :], f_new, self.f)
        self._apply_inlet(f_new)
        self._apply_outlet(f_new)
        self.f = f_new
        self.steps += 1
        self.last_stream_stats = stats
        return stats

    # -- forces ------------------------------------------------------------
    def momentum_exchange_force(self) -> tuple[float, float]:
        """Momentum-exchange force on all closed links (per unit rho * dx/dt^2)."""
        f = self.f
        fx = 0.0
        fy = 0.0
        for i in range(1, 9):
            closed = ((self.mask & np.uint8(1 << (i - 1))) != 0) & self.op
            if not closed.any():
                continue
            cx, cy = int(D2Q9_C[i, 0]), int(D2Q9_C[i, 1])
            contrib = f[i] + f[D2Q9_OPP[i]]
            fx += cx * float(contrib[closed].sum())
            fy += cy * float(contrib[closed].sum())
        return fx, fy

    def cylinder_force(self, centre, radius: float) -> tuple[float, float]:
        """Momentum-exchange force restricted to links of one obstacle.

        Only closed links whose node is within ``radius + 1`` cells of ``centre``
        contribute, so wall links of the channel are excluded.
        """
        f = self.f
        xs = np.arange(self.nx)[:, None]
        ys = np.arange(self.ny)[None, :]
        near = ((xs - centre[0]) ** 2 + (ys - centre[1]) ** 2) <= (radius + 1.5) ** 2
        fx = fy = 0.0
        for i in range(1, 9):
            closed = ((self.mask & np.uint8(1 << (i - 1))) != 0) & self.op & near
            if not closed.any():
                continue
            cx, cy = int(D2Q9_C[i, 0]), int(D2Q9_C[i, 1])
            contrib = f[i] + f[D2Q9_OPP[i]]
            s = float(contrib[closed].sum())
            fx += cx * s
            fy += cy * s
        return fx, fy


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------
def make_channel(nx: int, ny: int, *, inlet: bool = False, outlet: bool = False) -> np.ndarray:
    """Plane channel: solid rows at ``y = 0`` and ``y = ny - 1``.

    Wall planes therefore sit at ``y = 1/2`` and ``y = ny - 1/2``; the channel
    height in lattice units is ``H = ny - 1``.
    """
    ct = np.full((nx, ny), int(CellFlag.FLUID), dtype=np.int32)
    ct[:, 0] = int(CellFlag.OBSTACLE)
    ct[:, ny - 1] = int(CellFlag.OBSTACLE)
    if inlet:
        ct[0, 1 : ny - 1] = int(CellFlag.INLET)
    if outlet:
        ct[nx - 1, 1 : ny - 1] = int(CellFlag.OUTLET)
    return ct


def make_cylinder_channel(
    *,
    nx: int,
    ny: int,
    centre: tuple[float, float],
    radius: float,
) -> np.ndarray:
    """Channel with a circular obstacle (staircase mask) + inlet/outlet columns.

    The circle is rasterised by centre distance ``< radius`` (the repo convention
    for its validated cylinder case), which leaves the effective diameter within
    one cell of ``2 * radius``.
    """
    ct = make_channel(nx, ny, inlet=True, outlet=True)
    xs = np.arange(nx)[:, None]
    ys = np.arange(ny)[None, :]
    inside = (xs - centre[0]) ** 2 + (ys - centre[1]) ** 2 < radius ** 2
    ct[inside] = int(CellFlag.OBSTACLE)
    return ct


def poiseuille_profile(ny: int, nu: float, body_force: float) -> tuple[np.ndarray, np.ndarray]:
    """Analytic plane-Poiseuille profile for a wall at ``y = 1/2`` (exact OU).

    Returns ``(y, u(y))`` on the fluid cell rows ``1 .. ny-2`` with
    ``u = g (y - 1/2) (ny - 1/2 - y) / (2 nu)``.
    """
    y = np.arange(1, ny - 1, dtype=float)
    h = ny - 1.0
    u = body_force * (y - 0.5) * (h - (y - 0.5)) / (2.0 * nu)
    return y, u


def couette_profile(ny: int, u_wall: float) -> tuple[np.ndarray, np.ndarray]:
    """Analytic Couette profile between ``y = 1/2`` (rest) and ``y = ny - 1/2``."""
    y = np.arange(1, ny - 1, dtype=float)
    h = ny - 1.0
    return y, u_wall * (y - 0.5) / h