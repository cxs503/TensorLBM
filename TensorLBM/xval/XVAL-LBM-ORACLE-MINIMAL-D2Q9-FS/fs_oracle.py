"""Conservation-exact minimal D2Q9 free-surface (VoF fill-fraction) oracle.

Lineage
-------
Algorithm family: Thuerey/Koerner volume-of-fluid free-surface LBM, as written up
in the minimal teaching skeleton ``Maarten-vd-Sande/lbm`` (``lbm/streamcollide.py``,
``lbm/update_types.py``).  The skeleton's *core* D2Q9 operators were already
cross-validated against TensorLBM (see ``run_xval.py``, levels L1-L3).  Its
*free-surface* half was never validated -- and, as shown by the OSS inventory and
by direct execution, it is not merely unvalidated but broken: the liquid mass
diverges (``-5.9e19`` after 200 steps on a 60x60 dam break).

This module is the minimal *corrected* reference implementation.  Every deviation
from the skeleton is a documented fix of a defect from the OSS inventory
(docs/oss-inventory/oss-inv-Maarten-vd-Sande-lbm.md, table "缺陷清单"):

  D1  ``rho_prev = np.copy(mass)`` typo  -> previous-state fields are captured
      explicitly before the update (``prev`` bundle passed to ``update_types``).
  D2  ``fdist`` never reset across steps -> all flux bookkeeping is allocated per
      step; there is no cross-step scratch array at all.
  D3  ``rho`` division without guard      -> ``_RHO_FLOOR`` guard.
  D4  ``cell_type & CT.GAS == 2`` dead branch -> cell types are tested through
      single-bit masks (``classify``), so a GAS cell can never report eps=0.5.
  D5  force term missing ``1/cs^2``       -> ``forcing="guo"`` (correct, second
      order) is the default; ``forcing="raw"`` reproduces the skeleton's
      first-order term (effective acceleration ``-g/3``) for A/B comparison.
  D6  chained out-of-range comparison     -> all shifts are zero-filled and
      domain-exact (no edge replication, no periodic wrap).
  D7  silent velocity clamp at Ma=1.41    -> guard at ``U_GUARD`` with a counted
      ``guard_hits`` diagnostic instead of a silent modification.
  D8  object dtype cell types             -> int8 bit flags.
  D9  per-step print / history retention  -> silent, caller-driven diagnostics.

Conservation by construction
----------------------------
Mass moves through faces.  For every direction ``i`` we build ONE flux array
``Q_i(x) = w(x,i) * f_out_i(x)`` (valid only when both ``x`` and ``x + c_i`` are
liquid) and then apply it twice: ``-Q_i(x)`` on the sender and ``+Q_i(x - c_i)``
on the receiver.  Hence ``sum_x dm(x) == 0`` up to floating-point reordering, and
the volume-weighted face factor ``w = 0.5*(eps(x) + eps(x + c_i))`` is identical
for both sides of the face.  Cell-type conversions move the residual mass with
weights normalised to the residual, so they are conservative as well.

Units: lattice units throughout.  ``rho0 = 1``, ``cs^2 = 1/3``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

# ---------------------------------------------------------------------------
# D2Q9 lattice (identical ordering to tensorlbm.d2q9: verified in test suite)
# ---------------------------------------------------------------------------
CX = np.array([0, 1, 0, -1, 0, 1, 1, -1, -1], dtype=np.int64)
CY = np.array([0, 0, 1, 0, -1, 1, -1, -1, 1], dtype=np.int64)
W = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)
CS2 = 1.0 / 3.0

# ---------------------------------------------------------------------------
# Cell types (bit flags, int8) and constants of the conversion rule
# ---------------------------------------------------------------------------
FLUID = np.int8(1)
INTERFACE = np.int8(2)
GAS = np.int8(4)
OBSTACLE = np.int8(8)
LIQUID = np.int8(3)  # FLUID | INTERFACE

RHO_FLOOR = 1e-12
U_GUARD = 0.30  # stability guard on the macroscopic velocity (was 0.8165 = Ma 1.41)
FILL_OFFSET = 0.01  # eps excursion allowed before a phase change
LONELY_EPS = 0.1  # "lonely" fraction used by the topology rules


def classify(cell_type: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Boolean masks for fluid / interface / gas / obstacle (single-bit tests)."""
    return (
        (cell_type & FLUID) != 0,
        (cell_type & INTERFACE) != 0,
        (cell_type & GAS) != 0,
        (cell_type & OBSTACLE) != 0,
    )


def shift(arr: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """``out[x, y] = arr[x + dx, y + dy]`` with zero fill (no wrap, no replicate)."""
    nx, ny = arr.shape
    out = np.zeros_like(arr)
    xs0, xs1 = max(0, -dx), min(nx, nx - dx)
    ys0, ys1 = max(0, -dy), min(ny, ny - dy)
    if xs0 >= xs1 or ys0 >= ys1:
        return out
    out[xs0:xs1, ys0:ys1] = arr[xs0 + dx : xs1 + dx, ys0 + dy : ys1 + dy]
    return out


def equilibrium(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """D2Q9 second-order equilibrium, shape (9, nx, ny)."""
    usqr = 1.5 * (ux * ux + uy * uy)
    feq = np.empty((9,) + rho.shape, dtype=np.float64)
    for i in range(9):
        vu = 3.0 * (CX[i] * ux + CY[i] * uy)
        feq[i] = rho * W[i] * (1.0 + vu + 0.5 * vu * vu - usqr)
    return feq


def macroscopic(
    f: np.ndarray, fx: np.ndarray = None, fy: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """rho, ux, uy from the populations.

    With a body force the physical velocity carries the half-force correction
    ``u = sum(c f)/rho + F/(2 rho)`` (Guo).  ``guard_hits`` counts the cells whose
    speed exceeded ``U_GUARD``.
    """
    rho = f.sum(axis=0)
    rho_safe = np.where(rho > RHO_FLOOR, rho, RHO_FLOOR)
    ux = np.einsum("i,ijk->jk", CX.astype(np.float64), f) / rho_safe
    uy = np.einsum("i,ijk->jk", CY.astype(np.float64), f) / rho_safe
    if fx is not None:
        ux = ux + 0.5 * fx / rho_safe
        uy = uy + 0.5 * fy / rho_safe
    speed = np.sqrt(ux * ux + uy * uy)
    guard_hits = int(np.count_nonzero(speed > U_GUARD))
    scale = np.where(speed > U_GUARD, U_GUARD / np.maximum(speed, 1e-30), 1.0)
    return rho, ux * scale, uy * scale, guard_hits


def _shifted_alpha(a: np.ndarray, i: int) -> np.ndarray:
    return shift(a, CX[i], CY[i])


# ---------------------------------------------------------------------------
# Free-surface state container
# ---------------------------------------------------------------------------
@dataclass
class FreeSurfaceState:
    f: np.ndarray  # populations, (9, nx, ny)
    rho: np.ndarray
    ux: np.ndarray
    uy: np.ndarray
    mass: np.ndarray  # liquid volume per cell (the VoF ledger)
    ctype: np.ndarray  # cell types
    time: int = 0
    guard_hits: int = 0


@dataclass
class FreeSurfaceParams:
    nx: int = 48
    ny: int = 48
    omega: float = 1.0
    gravity: Tuple[float, float] = (0.0, -1.0e-3)
    forcing: str = "guo"  # "guo" (correct) | "raw" (skeleton defect D5, -g/3)
    rho_l: float = 1.0  # liquid density reference: m = rho_l <=> a full cell
    track_guard: bool = True


class FreeSurfaceSim:
    """Minimal corrected D2Q9 free-surface solver (2D, single phase + void)."""

    def __init__(self, params: FreeSurfaceParams):
        self.p = params
        nx, ny = params.nx, params.ny
        self.ctype = np.zeros((nx, ny), dtype=np.int8)
        self.mass = np.zeros((nx, ny), dtype=np.float64)
        self.rho = np.zeros((nx, ny), dtype=np.float64)
        self.ux = np.zeros((nx, ny), dtype=np.float64)
        self.uy = np.zeros((nx, ny), dtype=np.float64)
        self.f = np.zeros((9, nx, ny), dtype=np.float64)
        self.time = 0
        self.guard_hits = 0
        self.mass0 = 0.0
        self.mass_history: list[float] = []

    # -- initialisation ---------------------------------------------------
    def fill_box_walls(self) -> None:
        """Closed box: one cell thick obstacle frame."""
        self.ctype[0, :] = OBSTACLE
        self.ctype[-1, :] = OBSTACLE
        self.ctype[:, 0] = OBSTACLE
        self.ctype[:, -1] = OBSTACLE

    def set_liquid_from_height(
        self, height_fn, rho_l: float = 1.0, x_lo: int = None, x_hi: int = None
    ) -> None:
        """Initialise liquid columns from a height function ``h(x)`` in cell units.

        ``height_fn`` returns the (possibly fractional) liquid depth counted from
        the bottom obstacle row (``j = 1``).  Cells below ``floor(h)`` are FLUID,
        the cell containing the surface is INTERFACE with ``mass = frac*rho``,
        everything above is GAS.
        """
        nx, ny = self.p.nx, self.p.ny
        x_lo = 1 if x_lo is None else x_lo
        x_hi = nx - 1 if x_hi is None else x_hi
        self.ctype[:] = GAS
        self.fill_box_walls()  # re-applies the obstacle frame
        self.ctype[:x_lo, :] = GAS
        self.ctype[x_hi:, :] = GAS
        self.mass[:] = 0.0
        for ix in range(x_lo, x_hi):
            h = float(height_fn(ix))
            if h <= 0.0:
                continue
            j_lo = 1
            j_iface = j_lo + int(np.floor(h))
            for jy in range(j_lo, min(j_iface, ny - 1)):
                self.ctype[ix, jy] = FLUID
                self.mass[ix, jy] = rho_l
            if j_iface < ny - 1:
                frac = h - np.floor(h)
                if frac > 1e-9:
                    self.ctype[ix, j_iface] = INTERFACE
                    self.mass[ix, j_iface] = frac * rho_l
                else:
                    # surface exactly on a cell face: make the next cell interface
                    if j_iface + 1 < ny - 1:
                        self.ctype[ix, j_iface] = INTERFACE
                        self.mass[ix, j_iface] = rho_l
                        if j_iface + 1 < ny - 1:
                            self.ctype[ix, j_iface + 1] = INTERFACE
                            self.mass[ix, j_iface + 1] = 0.5 * rho_l
        self._finalise_init()

    def set_dam(self, dam_width: int, dam_height: int, rho_l: float = 1.0) -> None:
        """Classic dam-break initial condition (liquid column on the left)."""
        def h(_ix: int):
            return dam_height

        self.ctype[:] = GAS
        self.fill_box_walls()
        nx, ny = self.p.nx, self.p.ny
        j0 = ny - 1 - dam_height
        self.mass[:] = 0.0
        for ix in range(1, dam_width + 1):
            for jy in range(max(j0, 1), ny - 1):
                self.ctype[ix, jy] = FLUID
                self.mass[ix, jy] = rho_l
            if j0 - 1 >= 1:
                self.ctype[ix, j0 - 1] = INTERFACE
                self.mass[ix, j0 - 1] = 0.5 * rho_l
        if dam_width + 1 < nx - 1:
            for jy in range(max(j0, 1), ny - 1):
                self.ctype[dam_width + 1, jy] = INTERFACE
                self.mass[dam_width + 1, jy] = 0.5 * rho_l
        self._finalise_init()

    def _finalise_init(self) -> None:
        fluid, iface, gas, obs = classify(self.ctype)
        self.rho[fluid] = 1.0
        self.rho[iface] = 1.0
        self.rho[gas] = 0.0
        self.ux[:] = 0.0
        self.uy[:] = 0.0
        self.f[:] = equilibrium(self.rho, self.ux, self.uy)
        self.f[:, gas | obs] = 0.0
        self.mass0 = float(self.mass.sum())
        self.mass_history = [self.mass0]
        self.time = 0
        self.guard_hits = 0

    # -- helpers ----------------------------------------------------------
    def eps(self) -> np.ndarray:
        """Fill fraction; FLUID -> 1, GAS/OBSTACLE -> 0, INTERFACE -> m/rho."""
        fluid, iface, gas, obs = classify(self.ctype)
        eps = np.zeros_like(self.rho)
        eps[fluid] = 1.0
        eps[iface] = np.clip(self.mass[iface] / self.p.rho_l, 0.0, 1.0)
        return eps

    def normal(self) -> Tuple[np.ndarray, np.ndarray]:
        """Outward interface normal from the fill-fraction gradient."""
        eps = self.eps()
        nx = 0.5 * (shift(eps, -1, 0) - shift(eps, 1, 0))
        ny = 0.5 * (shift(eps, 0, -1) - shift(eps, 0, 1))
        return nx, ny

    def total_mass(self) -> float:
        return float(self.mass.sum())

    # -- collision --------------------------------------------------------
    def _collide(self, f: np.ndarray, rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
        """BGK collision + body force.  ``forcing`` selects the scheme."""
        p = self.p
        gx, gy = p.gravity
        omega = p.omega
        fx = rho * gx
        fy = rho * gy
        if p.forcing == "guo":
            # Guo: u* = u + F/(2 rho); S_i = w_i (1 - w/2)[...]
            rs = np.where(rho > RHO_FLOOR, rho, RHO_FLOOR)
            usx = ux + 0.5 * fx / rs
            usy = uy + 0.5 * fy / rs
            feq = equilibrium(rho, usx, usy)
            fout = f + omega * (feq - f)
            for i in range(9):
                cu = CX[i] * usx + CY[i] * usy
                cf = CX[i] * fx + CY[i] * fy
                src = W[i] * (1.0 - 0.5 * omega) * (
                    (cf - (usx * fx + usy * fy)) / CS2 + cu * cf / (CS2 * CS2)
                )
                fout[i] += src
        elif p.forcing == "raw":
            # Maarten-vd-Sande skeleton: f -= rho*t*(c.g)  ->  a_eff = -g/3
            feq = equilibrium(rho, ux, uy)
            fout = f + omega * (feq - f)
            for i in range(9):
                fout[i] -= rho * W[i] * (CX[i] * gx + CY[i] * gy)
        else:
            raise ValueError(f"unknown forcing scheme {p.forcing!r}")
        return fout

    # -- one time step ----------------------------------------------------
    def step(self) -> None:
        p = self.p
        fluid, iface, gas, obs = classify(self.ctype)
        liquid = fluid | iface

        # previous-state fields consumed by the conversion rule (defect D1)
        rho_prev = self.rho.copy()
        mass_prev = self.mass.copy()
        ctype_prev = self.ctype.copy()

        fout = self._collide(self.f, self.rho, self.ux, self.uy)
        fout[:, gas | obs] = 0.0

        # Reconstruction target: equilibrium at the *liquid reference* density
        # (the skeleton hard-codes rho=1.0 here).  Using the cell's own sum(f)
        # instead lets rho and m grow together, eps is then pinned and no phase
        # change ever happens -- that was the stalled-flow failure mode found here.
        feq_iface = equilibrium(
            np.full_like(self.rho, p.rho_l), self.ux, self.uy
        )
        nxf, nyf = self.normal()

        eps = self.eps()
        f_new = np.zeros_like(self.f)
        dm = np.zeros_like(self.mass)

        # ---- pass 1: incoming populations -----------------------------------
        # A void-facing population is replaced by the reflected equilibrium
        # (anti-bounce-back).  That reflection is *not* mass neutral away from
        # equilibrium: sum_i(feq_i + feq_opp - f_opp - f_i) > 0 would act as an
        # invisible mass source at the surface (measured: +0.2 mass per step per
        # interface cell, i.e. +12% of the total in 400 steps).  The reflection is
        # therefore mean-corrected per cell so that it transports momentum but not
        # mass.  See REPORT section L5.
        vals = []
        gas_dir = []
        dev_sum = np.zeros_like(self.mass)
        gas_cnt = np.zeros_like(self.mass)
        for i in range(9):
            io = int(OPP[i])
            nbr_src_liq = shift(liquid.astype(np.float64), -CX[i], -CY[i]) > 0.5
            nbr_src_gas = shift(gas.astype(np.float64), -CX[i], -CY[i]) > 0.5
            nbr_src_obs = shift(obs.astype(np.float64), -CX[i], -CY[i]) > 0.5
            pulled = shift(fout[i], -CX[i], -CY[i])
            mirror = feq_iface[i] + feq_iface[io] - fout[io]

            val = np.where(nbr_src_liq, pulled, 0.0)
            val = np.where((~nbr_src_liq) & nbr_src_obs, fout[io], val)
            val = np.where((~nbr_src_liq) & (~nbr_src_obs) & nbr_src_gas, mirror, val)
            m_gas = (~nbr_src_liq) & (~nbr_src_obs) & nbr_src_gas & liquid
            vals.append(val)
            gas_dir.append(m_gas)
            if i > 0:
                dev = np.where(m_gas, mirror - fout[i], 0.0)
                dev_sum += dev
                gas_cnt += m_gas.astype(np.float64)

        # mean-correct the reflection: momentum is kept, mass is not created
        has_gas = gas_cnt > 0.0
        corr = np.where(has_gas, dev_sum / np.where(has_gas, gas_cnt, 1.0), 0.0)

        for i in range(9):
            val = np.where(gas_dir[i] & has_gas, vals[i] - corr, vals[i])
            f_new[i] = np.where(liquid, val, 0.0)

            # ---- pass 2: mass ledger ---------------------------------------
            # liquid-liquid faces: one flux value applied with both signs and a
            # shared volume weight -> exactly antisymmetric (machine precision).
            # void / wall faces: weight 1, and the (corrected) reflection is booked
            # as an incoming flux, so the ledger is exactly what the populations do.
            if i == 0:
                continue
            tgt_liq = shift(liquid.astype(np.float64), CX[i], CY[i]) > 0.5
            src_liq = shift(liquid.astype(np.float64), -CX[i], -CY[i]) > 0.5
            eps_src = shift(eps, -CX[i], -CY[i])
            eps_tgt = shift(eps, CX[i], CY[i])
            w_out = np.where(tgt_liq, 0.5 * (eps + eps_tgt), 1.0)
            w_in = np.where(src_liq, 0.5 * (eps + eps_src), 1.0)
            dm += np.where(liquid, w_in * f_new[i] - w_out * fout[i], 0.0)

        # macroscopic quantities
        fx = self.rho * p.gravity[0] if p.forcing == "guo" else None
        fy = self.rho * p.gravity[1] if p.forcing == "guo" else None
        rho_new, ux_new, uy_new, hits = macroscopic(f_new, fx, fy)
        if p.track_guard:
            self.guard_hits += hits
            ux_new = np.where(~liquid, 0.0, ux_new)
            uy_new = np.where(~liquid, 0.0, uy_new)

        mass_new = self.mass + dm

        self.f = f_new
        self.rho = np.where(liquid, rho_new, 0.0)
        self.ux, self.uy = ux_new, uy_new
        self.mass = mass_new
        self.time += 1

        self._update_types(rho_prev, mass_prev, ctype_prev)
        self.mass_history.append(self.total_mass())

    # -- cell-type update (Koerner conversion rule) -----------------------
    def _update_types(
        self,
        rho_prev: np.ndarray,
        mass_prev: np.ndarray,
        ctype_prev: np.ndarray,
    ) -> None:
        fluid, iface, gas, obs = classify(self.ctype)
        eps = self.mass / self.p.rho_l  # signed, unclipped: drives the phase change

        # neighbourhood existence
        has_fluid = np.zeros_like(iface)
        has_iface = np.zeros_like(iface)
        empty_adj = np.zeros_like(iface)  # adjacent gas cell (void neighbour)
        for i in range(1, 9):
            has_fluid |= shift(fluid.astype(np.float64), CX[i], CY[i]) > 0.5
            has_iface |= shift(iface.astype(np.float64), CX[i], CY[i]) > 0.5
            empty_adj |= shift(gas.astype(np.float64), CX[i], CY[i]) > 0.5

        to_fluid = iface & (
            (eps > 1.0 + FILL_OFFSET) | ((eps >= 1.0 - LONELY_EPS) & ~empty_adj)
        )
        to_gas = iface & (
            (eps < -FILL_OFFSET)
            | ((eps <= LONELY_EPS) & has_fluid & empty_adj)
            | (~has_fluid & ~has_iface)
        )

        if not (to_fluid.any() or to_gas.any()):
            return

        nxf, nyf = self.normal()

        # residual mass to be redistributed (excess for I->F, remainder for I->G)
        # signed residual: I->F gives back (m - rho) -- which can be *negative*
        # for the "lonely" conversion, and must be given back to the neighbours
        # rather than silently topping the cell up to rho (that invented mass);
        # I->G gives back whatever is left, normally a small positive number.
        excess = np.zeros_like(self.mass)
        excess[to_fluid] = self.mass[to_fluid] - self.p.rho_l
        excess[to_gas] = self.mass[to_gas]

        # One weight array per direction.  Receivers that are themselves
        # converting this step are excluded, otherwise the share deposited into
        # them would be wiped by the flush below (that was the mass leak of the
        # first revision: +2.0 per step on a 48x48 dam break).
        receiving = ~(to_fluid | to_gas)
        weights = []
        wsum = np.zeros_like(self.mass)
        for i in range(1, 9):
            eta = CX[i] * nxf + CY[i] * nyf
            eta = np.where(to_fluid, np.maximum(eta, 0.0), np.maximum(-eta, 0.0))
            tgt_iface = shift(iface.astype(np.float64), CX[i], CY[i]) > 0.5
            tgt_liquid = shift((fluid | iface).astype(np.float64), CX[i], CY[i]) > 0.5
            # a converting receiver would wipe the share in the flush below
            tgt_recv = shift(receiving.astype(np.float64), CX[i], CY[i]) > 0.5
            w = np.where(tgt_iface, eta, np.where(tgt_liquid, 1e-12 + 0.05 * eta, 0.0))
            w = np.where(tgt_recv, w, 0.0)
            w = np.where((to_fluid | to_gas) & receiving, w, 0.0)
            weights.append(w)
            wsum += w

        # a converting cell with nowhere to spill must NOT convert: the residual
        # would be stranded in a GAS cell (negative mass!) and then be destroyed
        # when the cell is re-wetted -> that was the remaining mass source.
        orphan = (to_fluid | to_gas) & (wsum <= 0.0)
        if orphan.any():
            to_fluid &= ~orphan
            to_gas &= ~orphan
            excess[orphan] = 0.0

        # weights are normalised per converting cell, so the scattered share sums
        # to the residual exactly -> the conversion is conservative
        scatter = np.zeros_like(self.mass)
        for i, w in enumerate(weights, start=1):
            share = np.where(wsum > 0.0, w * excess / np.where(wsum > 0.0, wsum, 1.0), 0.0)
            scatter += shift(share, -CX[i], -CY[i])
        self.mass += scatter

        # convert and flush the residual
        self.mass[to_fluid] = self.p.rho_l
        self.mass[to_gas] = 0.0
        self.ctype[to_fluid] = FLUID
        self.ctype[to_gas] = GAS

        # newly wetted cells: a void next to a fresh fluid/gas cell becomes interface
        for i in range(1, 9):
            fresh = shift((to_fluid | to_gas).astype(np.float64), -CX[i], -CY[i]) > 0.5
            gas_now = classify(self.ctype)[2]
            new_iface = fresh & gas_now
            if new_iface.any():
                self._init_new_interface(new_iface, rho_prev, mass_prev, ctype_prev)



    def _init_new_interface(
        self,
        mask: np.ndarray,
        rho_prev: np.ndarray,
        mass_prev: np.ndarray,
        ctype_prev: np.ndarray,
    ) -> None:
        """Initialise freshly wetted cells by averaging the liquid neighbours."""
        fluid, iface, gas, obs = classify(self.ctype)
        src = (fluid | iface) & ~mask
        rsum = np.zeros_like(self.rho)
        usum = np.zeros_like(self.rho)
        vsum = np.zeros_like(self.rho)
        cnt = np.zeros_like(self.rho)
        for i in range(1, 9):
            valid = shift(src.astype(np.float64), CX[i], CY[i]) > 0.5
            rsum += np.where(valid, shift(self.rho, CX[i], CY[i]), 0.0)
            usum += np.where(valid, shift(self.ux, CX[i], CY[i]), 0.0)
            vsum += np.where(valid, shift(self.uy, CX[i], CY[i]), 0.0)
            cnt += valid.astype(np.float64)
        ok = mask & (cnt > 0.0)
        if not ok.any():
            return
        rho_new = rsum[ok] / cnt[ok]
        ux_new = usum[ok] / cnt[ok]
        uy_new = vsum[ok] / cnt[ok]
        self.rho[ok] = rho_new
        self.ux[ok] = ux_new
        self.uy[ok] = uy_new
        self.mass[ok] = 0.0
        self.ctype[ok] = INTERFACE
        feq = equilibrium(
            np.where(ok, self.p.rho_l, 0.0), np.where(ok, self.ux, 0.0), np.where(ok, self.uy, 0.0)
        )
        for i in range(9):
            self.f[i] = np.where(ok, feq[i], self.f[i])

    # -- diagnostics ------------------------------------------------------
    def column_height(self, rho_l: float = 1.0) -> np.ndarray:
        """Liquid depth per column in cell units (sum of m/rho_l over the column)."""
        return self.mass.sum(axis=1) / rho_l

    def liquid_mask(self) -> np.ndarray:
        fluid, iface, _, _ = classify(self.ctype)
        return fluid | iface

    def front_position(self) -> float:
        cols = self.liquid_mask().any(axis=1)
        if not cols.any():
            return 0.0
        return float(np.nonzero(cols)[0].max())

    def mass_drift(self) -> float:
        return abs(self.total_mass() - self.mass0) / max(abs(self.mass0), 1e-30)