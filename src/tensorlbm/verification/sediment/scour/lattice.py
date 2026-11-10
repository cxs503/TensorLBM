"""Lean D2Q9 kernels for the sediment-transport / scour verification suite.

Design constraints
------------------
This module is deliberately dependency-light (NumPy only) and self-contained so
that the morphodynamic verifications of :mod:`tensorlbm.verification.sediment.scour`
can run inside the CI image *and* on the loong64 cluster nodes without a
CUDA/torch stack.  It is **not** a replacement for the production solvers in
:mod:`tensorlbm.boundary` / :mod:`tensorlbm.core`; it is the minimal engine the
sediment verification face needs:

* D2Q9, second-order equilibrium, BGK collision;
* Guo body force (gravity / streamwise driving force), split form;
* halfway bounce-back for solids (bed of arbitrary, cell-stepped elevation and
  the impermeable obstacle);
* specular (free-slip) reflection for the rigid-lid free surface;
* Smagorinsky sub-grid closure so that coarse-grid runs at engineering
  Reynolds numbers stay stable instead of blowing up at ``tau -> 0.5``;
* standard log-law bed-shear estimator (the quantity morphodynamics consumes).

Notation
--------
All quantities are lattice units unless a name ends in ``_phys``.  The
conversion used by the callers is ``dx`` (m/cell), ``dt`` (s/step),
``u_phys = u_lat * dx / dt`` and ``nu_lat = nu_phys * dt / dx**2``.

Array layout is ``(9, nx, ny)`` with ``x`` the streamwise and ``y`` the
wall-normal direction; ``y = 0`` is the bed.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# D2Q9 stencil.  Order: rest, E, N, W, S, NE, NW, SW, SE.
# ---------------------------------------------------------------------------
CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int64)
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int64)
W = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)


def _specular_table() -> np.ndarray:
    """Index of the population whose *normal* component is reversed.

    Used for the specular (free-slip) rigid-lid boundary: a population leaving
    the domain through the top is re-injected with ``cy -> -cy`` and ``cx``
    unchanged, which is the discrete statement of ``u_n = 0, du_t/dn = 0``.
    """
    table = np.zeros(9, dtype=np.int64)
    for i in range(9):
        for j in range(9):
            if CX[j] == CX[i] and CY[j] == -CY[i]:
                table[i] = j
                break
    return table


SPEC = _specular_table()


def equilibrium(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """Second-order D2Q9 equilibrium, shape ``(9, nx, ny)``."""
    cu = CX[:, None, None] * ux[None] + CY[:, None, None] * uy[None]
    usq = ux * ux + uy * uy
    return W[:, None, None] * rho[None] * (
        1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq[None]
    )


class D2Q9Channel:
    """Periodic-in-x plane channel with a no-slip bed and a slip rigid lid.

    Parameters
    ----------
    nx, ny : int
        Grid size.  ``x`` is periodic; ``y = 0`` is the bed (bounce-back wall)
        and ``y = ny - 1`` is the rigid lid (specular slip unless
        ``top="wall"``).
    u_lat : float
        Reference lattice velocity used to fix ``dt`` from the physical
        reference velocity (only for bookkeeping / conversions).
    nu_lat : float
        Molecular lattice viscosity; ``tau0 = 3 nu_lat + 0.5``.
    g_lat : float
        Streamwise body force per unit mass in lattice units.
    cs_smag : float
        Smagorinsky constant; ``0`` disables the sub-grid closure.
    solid : np.ndarray[bool] | None
        Initial solid mask ``(nx, ny)``; ``True`` = impermeable.
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        *,
        u_lat: float = 0.05,
        nu_lat: float = 1e-2,
        g_lat: float = 1e-6,
        cs_smag: float = 0.0,
        solid: np.ndarray | None = None,
        top: str = "slip",
    ) -> None:
        self.nx, self.ny = int(nx), int(ny)
        self.u_lat, self.nu_lat, self.g_lat = float(u_lat), float(nu_lat), float(g_lat)
        self.cs_smag = float(cs_smag)
        self.tau0 = 3.0 * self.nu_lat + 0.5
        self.top = top

        if solid is None:
            solid = np.zeros((self.nx, self.ny), dtype=bool)
            solid[:, 0] = True  # bed wall row (no-slip, halfway bounce-back)
        solid[:, -1] = True  # lid row: slip (specular) or no-slip, see ``top``
        self.solid = solid.copy()

        self.rho = np.ones((self.nx, self.ny))
        self.ux = np.zeros((self.nx, self.ny))
        self.uy = np.zeros((self.nx, self.ny))
        self.f = equilibrium(self.rho, self.ux, self.uy)
        self.steps_done = 0
        self._solid_shift = None
        #: quadratic per-cell drag (e.g. depth-averaged bottom friction), used as
        #: ``a_x = g_lat - drag_coeff * |u| * u_x``.  Scalar or ``(nx, ny)``.
        self.drag_coeff: float | np.ndarray = 0.0
        #: Multiplier on the Guo source term.  ``1.0`` injects the source
        #: momentum ``rho*a`` per step, which is what makes the discrete
        #: momentum balance ``rho*g*H = tau_w`` exact (see
        #: ``tests/test_sediment_scour.py::test_momentum_balance``).
        self.force_scale = 1.0
        #: Optional mixing length ``(nx, ny)`` in lattice units; switches the
        #: sub-grid closure from Smagorinsky to ``nu_t = l_m^2 |S|``.
        self.mixing_length: np.ndarray | None = None

    # -- geometry -----------------------------------------------------------
    def set_solid(self, solid: np.ndarray) -> None:
        """Replace the solid mask, re-seeding freshly wetted cells."""
        solid = np.asarray(solid, dtype=bool)
        newly_wet = self.solid & ~solid
        newly_dry = ~self.solid & solid
        self.solid = solid.copy()
        self.f[:, newly_dry] = 0.0
        self._solid_shift = None
        if newly_wet.any():
            rho, ux, uy = self.macros()
            feq = equilibrium(
                np.where(newly_wet, 1.0, rho),
                np.where(newly_wet, ux, 0.0),
                np.where(newly_wet, uy, 0.0),
            )
            self.f[:, newly_wet] = feq[:, newly_wet]

    def _shifts(self):
        if self._solid_shift is None:
            self._solid_shift = [
                np.roll(np.roll(self.solid, int(CX[i]), axis=0), int(CY[i]), axis=1)
                for i in range(9)
            ]
        return self._solid_shift

    def bed_profile(self) -> np.ndarray:
        """Bed elevation (cell index of the topmost solid cell) per column."""
        prof = np.zeros(self.nx, dtype=np.int64)
        for x in range(self.nx):
            col = np.nonzero(self.solid[x])[0]
            prof[x] = col.max() if col.size else -1
        return prof

    # -- basic operators ----------------------------------------------------
    def macros(self):
        f = self.f
        rho = f.sum(axis=0)
        # solid cells never hold a meaningful density (their populations are
        # only used by the reflection); guard the division so they stay finite.
        safe = np.where(rho > 0.0, rho, 1.0)
        ux = (CX[:, None, None] * f).sum(axis=0) / safe
        uy = (CY[:, None, None] * f).sum(axis=0) / safe
        fluid = ~self.solid
        ux = np.where(fluid, ux, 0.0)
        uy = np.where(fluid, uy, 0.0)
        return rho, ux, uy

    def _eddy_viscosity(self, feq) -> np.ndarray:
        """Sub-grid viscosity.

        Two closures are available:

        * ``mixing_length is None`` -> Smagorinsky, ``nu_t = (Cs dx)^2 |S|``.
          Appropriate when the resolved strain is not dominated by a
          wall-normal mixing length (the plan-view scour case);
        * ``mixing_length`` given -> ``nu_t = l_m^2 |S|`` with
          ``l_m ~ kappa * y_wall``.  This is what makes a coarse-grid channel
          reproduce the log law at all: Smagorinsky's fixed length is
          *decreasing* in the outer layer while the log layer needs
          ``nu_t = kappa y u_*``, so a Smagorinsky channel has a wrong mean slope.
        """
        f = self.f
        cxy = (CX * CY)[:, None, None]
        pixx = (((CX * CX)[:, None, None]) * (f - feq)).sum(axis=0)
        piyy = (((CY * CY)[:, None, None]) * (f - feq)).sum(axis=0)
        pixy = (cxy * (f - feq)).sum(axis=0)
        rho = np.maximum(self.f.sum(axis=0), 1e-30)
        if self.mixing_length is not None:
            sxx = -1.5 * pixx / (rho * self.tau0)
            syy = -1.5 * piyy / (rho * self.tau0)
            sxy = -1.5 * pixy / (rho * self.tau0)
            s_mag = np.sqrt(2.0 * (sxx * sxx + 2.0 * sxy * sxy + syy * syy))
            return self.mixing_length**2 * s_mag
        q = np.sqrt(2.0 * (pixx * pixx + 2.0 * pixy * pixy + piyy * piyy))
        tau = 0.5 * (np.sqrt(self.tau0**2 + 18.0 * np.sqrt(2.0) * self.cs_smag**2 * q / rho) - self.tau0)
        return np.maximum(tau - (self.tau0 - 0.5), 0.0) / 3.0

    def step(self, n: int = 1, *, record: bool = False) -> dict:
        """Advance ``n`` lattice steps.  Returns diagnostics if ``record``."""
        for _ in range(n):
            rho, ux, uy = self.macros()
            feq = equilibrium(rho, ux, uy)
            tau = np.full_like(rho, self.tau0)
            if self.cs_smag > 0.0 or self.mixing_length is not None:
                nut = self._eddy_viscosity(feq)
                tau = tau + 3.0 * nut
            tau = np.maximum(tau, 0.501)
            # Guo forcing: a = (g_lat, 0)
            ax = self.g_lat - self.drag_coeff * np.abs(ux) * ux
            ay = -self.drag_coeff * np.abs(ux) * uy
            cu = CX[:, None, None] * ux[None] + CY[:, None, None] * uy[None]
            cxu = CX[:, None, None] * ax + CY[:, None, None] * ay
            src = W[:, None, None] * rho[None] * (
                3.0 * (cxu - (ax * ux + ay * uy)) + 9.0 * cu * cxu
            )
            fpost = self.f - (self.f - feq) / tau + self.force_scale * src

            fnew = np.empty_like(fpost)
            shifts = self._shifts() if not (
                self.top == "slip" and self.steps_done == 0 and self._solid_shift is not None
            ) else self._solid_shift
            shifted_f = [
                None,
                np.roll(fpost[1], 1, axis=0),
                np.roll(fpost[2], 1, axis=1),
                np.roll(fpost[3], -1, axis=0),
                np.roll(fpost[4], -1, axis=1),
                np.roll(np.roll(fpost[5], 1, axis=0), 1, axis=1),
                np.roll(np.roll(fpost[6], -1, axis=0), 1, axis=1),
                np.roll(np.roll(fpost[7], -1, axis=0), -1, axis=1),
                np.roll(np.roll(fpost[8], 1, axis=0), -1, axis=1),
            ]
            fnew[0] = fpost[0]
            slip = self.top == "slip"
            for i in range(1, 9):
                source = shifts[i]
                if slip and i in (4, 7, 8):
                    refl = fpost[SPEC[i]]
                else:
                    refl = fpost[OPP[i]]
                fnew[i] = np.where(source, refl, shifted_f[i])
            self.f = fnew
            self.steps_done += 1
        if not record:
            return {}
        self.rho, self.ux, self.uy = self.macros()
        return self.diagnostics()

    # -- diagnostics --------------------------------------------------------
    def diagnostics(self) -> dict:
        rho, ux, uy = self.macros()
        self.rho, self.ux, self.uy = rho, ux, uy
        fluid = ~self.solid
        return {
            "steps": self.steps_done,
            "mass": float(rho[fluid].sum()),
            "u_max": float(np.abs(ux[fluid]).max()),
            "u_mean_y": float(ux[fluid].mean()),
        }

    def vertical_profile(self, x: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Bed-normal profile ``(y_mid, u_x)`` of the given column (default mid)."""
        if x is None:
            x = self.nx // 2
        col = ~self.solid[x]
        y = np.nonzero(col)[0]
        return y + 0.5, self.ux[x, y]

    def sheets(self) -> list[np.ndarray]:
        """Index arrays of the first fluid cell above the bed for each column."""
        out = []
        for x in range(self.nx):
            col = np.nonzero(~self.solid[x])[0]
            if col.size == 0:
                out.append(np.array([], dtype=np.int64))
            else:
                out.append(np.array([col[0]], dtype=np.int64))
        return out

    # -- bed shear ---------------------------------------------------------
    def bed_shear(self, *, kappa: float = 0.41, z0_fac: float = 1.0 / 9.0) -> np.ndarray:
        """Bed shear stress per column via the log-law wall function.

        The wall-function is the standard estimator used when the viscous
        sublayer is not resolved::

            u_1 / u_* = (1/kappa) * ln(9 * z_1 * u_* / nu)

        solved with a damped fixed point on ``u_*``.  ``z_1 = 0.5`` cell is the
        distance of the first fluid cell centre above the bed surface, so the
        returned stress is located half a cell above the staircase bed (a
        documented ~O(dx) bias for the morphodynamic forcing).
        """
        u1 = np.zeros(self.nx)
        for x in range(self.nx):
            col = np.nonzero(~self.solid[x])[0]
            if col.size:
                u1[x] = abs(self.ux[x, col[0]])
        z1 = 0.5
        # Viscous branch first: if the first cell sits inside the sublayer
        # (y+ < 11.5) the log law does not apply and the linear profile is exact.
        ustar_visc = np.sqrt(self.nu_lat * u1 / z1)
        y_plus = z1 * ustar_visc / self.nu_lat
        log_branch = y_plus >= 11.5
        ustar = 0.1 * u1 + 1e-6
        for _ in range(80):
            arg = np.maximum(9.0 * z1 * ustar / self.nu_lat, 1.01)
            new = kappa * u1 / np.log(arg)
            ustar = 0.5 * (ustar + new)
        ustar = np.where(log_branch, ustar, ustar_visc)
        return ustar**2  # rho = 1 in lattice units