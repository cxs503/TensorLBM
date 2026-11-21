"""D2Q9 solver for (layered, variable-viscosity) plane Poiseuille flow.

This is the compute kernel of the cross-validation case
``XVAL-LBM-LAYERED-POISEUILLE-ANALYTIC-INLET``.  It deliberately stays small and
numpy-only so that the *reference* (exact piecewise-parabolic solution) and the
*numerics* never share code.

Implemented ingredients
-----------------------
* D2Q9, BGK/SRT collision, Guo (2002) forcing -- identical forcing formula to
  the in-repo D3Q19 duct solver (``verification/square_duct_poiseuille``).
* Per-node relaxation time ``tau(y)`` from a layered marker field, i.e. an
  immiscible two-layer channel with a *sharp* viscosity interface.  Nothing
  about the interface is dynamic: this isolates the analytic-inlet mechanism and
  the piecewise-stress-continuity physics (see the predeclared deviation note --
  TCLB couples the same profile to a phase-field/Shan-Chen interface, which per
  ``NEG-M15267-MCMP-POISEUILLE-R4`` cannot reach the interface-spurious
  threshold anyway).
* Half-way bounce-back on the two solid rows.
* Two streamwise closures, selected by ``XBoundary``:

  - ``"periodic"``    : TCLB ``developedFlow`` literal -- analytic full-field
    initial condition + body force + periodic streamwise wrap.  The problem is
    one-dimensional in y, so the domain can be a few columns wide.
  - ``"inlet_outlet"``: extension of the same idea to an open channel: Zou-He
    (1997) velocity inlet carrying the *analytic* profile, zero-gradient
    outflow, body force still on.  This is what "analytic inlet" means when the
    profile is not just an initial guess but a boundary condition.
* Initial condition selected by ``init_mode``: ``"analytic"`` (TCLB
  ``developedFlow``), ``"uniform"`` (zero velocity, the control arm).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

# --------------------------------------------------------------------------- #
# D2Q9 lattice
# --------------------------------------------------------------------------- #
CX = np.array([0, 1, -1, 0, 0, 1, -1, 1, -1], dtype=int)
CY = np.array([0, 0, 0, 1, -1, 1, 1, -1, -1], dtype=int)
W = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4, dtype=float)
OPP = np.array([0, 2, 1, 4, 3, 8, 7, 6, 5], dtype=int)
CS2 = 1.0 / 3.0


def equilibrium(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """D2Q9 second-order equilibrium, shape (9, ny, nx)."""
    cu = CX[:, None, None] * ux[None] + CY[:, None, None] * uy[None]
    u2 = ux * ux + uy * uy
    return W[:, None, None] * rho[None] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2[None])


def macroscopic(f: np.ndarray, force_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """rho, ux, uy with the Guo velocity shift ``u = (sum f c + F/2)/rho``."""
    rho = f.sum(axis=0)
    mx = (f * CX[:, None, None]).sum(axis=0) + 0.5 * force_x
    my = (f * CY[:, None, None]).sum(axis=0)
    safe = np.where(rho > 0.0, rho, 1.0)
    return rho, mx / safe, my / safe


@dataclass
class ChannelConfig:
    """Geometry / physics of one layered channel arm (all lattice units)."""

    height: int = 64                 # wall-to-wall fluid rows (ny = height + 2)
    nx: int = 8                      # columns; ignored for inlet_outlet if nx<0 -> 4*height
    nus: tuple = (0.12, 0.04)        # kinematic viscosity per layer, bottom -> top
    interface_fracs: tuple = (0.5,)  # interior interfaces as fractions of height
    u_peak: float = 0.05             # target peak velocity (set g from it)
    force: Optional[float] = None    # explicit body force (overrides u_peak)
    x_boundary: str = "periodic"     # "periodic" | "inlet_outlet"
    init_mode: str = "analytic"      # "analytic" | "uniform"
    inlet_from_analytic: bool = True # inlet_outlet only: analytic profile vs mass-equivalent plug
    outflow_density: float = 1.0     # Zou-He pressure outlet reference density
    steps: int = 40000
    sample_every: int = 200
    tol_steady: float = 1e-9
    steady_window: int = 200
    label: str = "chan"

    def __post_init__(self) -> None:
        if self.height < 4:
            raise ValueError("height must be >= 4 fluid rows")
        if min(self.nus) <= 0:
            raise ValueError("viscosities must be positive")
        if self.x_boundary not in ("periodic", "inlet_outlet"):
            raise ValueError("x_boundary must be 'periodic' or 'inlet_outlet'")
        if self.init_mode not in ("analytic", "uniform"):
            raise ValueError("init_mode must be 'analytic' or 'uniform'")
        if len(self.nus) != len(self.interface_fracs) + 1:
            raise ValueError("need len(nus) == len(interface_fracs) + 1")
        for nu in self.nus:
            if not (0.5 + 3.0 * nu > 0.505):
                raise ValueError(f"tau={0.5 + 3 * nu:.4f} too close to 0.5 (unstable)")


class LayeredChannel:
    """D2Q9 BGK solver with a layered viscosity field and an analytic inlet."""

    def __init__(self, cfg: ChannelConfig,
                 unit_profile: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        """``unit_profile(y)`` is the exact velocity profile **per unit body force**;
        the initial condition is ``g * unit_profile(y)`` so that the analytic
        initialisation (TCLB ``developedFlow``) starts at the arm's own amplitude."""
        self.cfg = cfg
        ny = cfg.height + 2
        nx = cfg.nx if cfg.x_boundary == "periodic" else (cfg.nx if cfg.nx > 0 else 4 * cfg.height)
        self.ny, self.nx = ny, nx

        # --- geometry: rows 0 and ny-1 solid; wall at y = 0.5 and ny-1.5 ---- #
        self.solid = np.zeros((ny, nx), dtype=bool)
        self.solid[0, :] = True
        self.solid[-1, :] = True

        # wall-normal coordinate of the fluid rows measured from the lower wall
        self.y_wall = np.arange(1, ny - 1, dtype=float) - 0.5     # (ny-2,)
        self.height = float(ny - 2)

        # --- layered viscosity field ---------------------------------------- #
        interfaces = [f * self.height for f in cfg.interface_fracs]
        layer = np.searchsorted(np.array(interfaces, dtype=float), self.y_wall, side="right")
        nu_y = np.asarray(cfg.nus, dtype=float)[layer]             # (ny-2,)
        self.layer = layer
        self.nu_field = np.zeros((ny, nx))
        self.nu_field[1:-1, :] = nu_y[:, None]
        self.tau_field = 0.5 + 3.0 * self.nu_field
        self.omega = np.where(self.solid, 0.0, 1.0 / np.where(self.solid, 1.0, self.tau_field))

        # --- body force from the target peak velocity ----------------------- #
        if cfg.force is not None:
            self.g = float(cfg.force)
        else:
            if unit_profile is None:
                raise ValueError("unit_profile required to derive the force from u_peak")
            unit = unit_profile(self.y_wall)                         # g = 1 profile
            scale = float(np.max(np.abs(unit)))
            if scale <= 0:
                raise ValueError("degenerate analytic profile")
            self.g = cfg.u_peak / scale
        self.fx = np.full((ny, nx), self.g)
        self.fx[self.solid] = 0.0

        # --- streaming source masks (pull convention) ----------------------- #
        self._build_stream_masks()

        # --- initial condition ---------------------------------------------- #
        self.f = self._initial_populations(unit_profile)

        # --- bookkeeping ---------------------------------------------------- #
        self.step_count = 0
        self.history: List[Dict[str, float]] = []
        self._u_prev_for_steady: Optional[np.ndarray] = None
        self._u_steady_ref: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    def _build_stream_masks(self) -> None:
        """Pre-compute, per direction, the solid mask of the upstream source node.

        Pull streaming: ``f_new[i](y, x) = f_post[i](y - cy_i, x - cx_i)``.  Where
        that source is solid, half-way bounce-back replaces the population with
        ``f_post[opp(i)]`` *at the receiving node* (wall exactly half-way between
        the fluid node and the solid node).
        """
        periodic = self.cfg.x_boundary == "periodic"
        self._bb_mask = {0: np.zeros_like(self.solid)}
        for i in range(1, 9):
            ys = np.arange(self.ny) - CY[i]
            xs = np.arange(self.nx) - CX[i]
            if periodic:
                src = self.solid[np.ix_(ys % self.ny, xs % self.nx)]
            else:
                keep_y = (ys >= 0) & (ys < self.ny)
                keep_x = (xs >= 0) & (xs < self.nx)
                src = self.solid[np.ix_(np.clip(ys, 0, self.ny - 1), np.clip(xs, 0, self.nx - 1))].copy()
                # out-of-domain in x is handled by the x boundary condition, not by bounce-back
                src[~keep_y, :] = False
                src[:, ~keep_x] = False
            self._bb_mask[i] = src & ~self.solid

    def _initial_populations(self, unit_profile: Optional[Callable]) -> np.ndarray:
        ny, nx = self.ny, self.nx
        rho = np.ones((ny, nx))
        if self.cfg.init_mode == "analytic" and unit_profile is not None:
            ux = np.zeros((ny, nx))
            ux[1:-1, :] = (self.g * unit_profile(self.y_wall))[:, None]
        else:
            ux = np.zeros((ny, nx))
        uy = np.zeros((ny, nx))
        f = equilibrium(rho, ux, uy)
        f[:, self.solid] = 0.0
        return f

    # ------------------------------------------------------------------ #
    def _collide(self) -> np.ndarray:
        rho, ux, uy = macroscopic(self.f, self.fx)
        feq = equilibrium(rho, ux, uy)
        om = self.omega[None]
        f_post = self.f - om * (self.f - feq)
        # Guo forcing: (1 - omega/2) * w_i * [3 (c_i - u) + 9 (c_i.u) c_i] . F
        cu = CX[:, None, None] * ux[None] + CY[:, None, None] * uy[None]
        guo = W[:, None, None] * (
            3.0 * (CX[:, None, None] - ux[None]) * self.fx[None]
            + 9.0 * cu * (CX[:, None, None] * self.fx[None])
        )
        f_post = f_post + (1.0 - 0.5 * om) * guo
        f_post[:, self.solid] = 0.0
        self._last_rho, self._last_ux, self._last_uy = rho, ux, uy
        return f_post

    def _stream(self, f_post: np.ndarray) -> np.ndarray:
        periodic = self.cfg.x_boundary == "periodic"
        f_new = np.empty_like(f_post)
        for i in range(9):
            if periodic:
                f_new[i] = np.roll(f_post[i], (CY[i], CX[i]), axis=(0, 1))
            else:
                src = f_post[i]
                # zero-gradient extrapolation in x first, then shift in y only
                g = np.empty_like(src)
                g[:, 1:] = src[:, :-1]
                g[:, 0] = src[:, 0]
                if i in (2, 6, 8):        # cx = -1 : take from the right neighbour
                    g[:, :-1] = src[:, 1:]
                    g[:, -1] = src[:, -1]
                if CY[i] != 0:
                    g = np.roll(g, CY[i], axis=0)
                f_new[i] = g
            m = self._bb_mask[i]
            if m.any():
                f_new[i][m] = f_post[OPP[i]][m]
        f_new[:, self.solid] = 0.0
        if not periodic:
            self._apply_inlet_outlet(f_new)
        return f_new

    # ------------------------------------------------------------------ #
    def _apply_inlet_outlet(self, f: np.ndarray) -> None:
        """Zou-He velocity inlet at x=0 + Zou-He pressure outlet at x=nx-1.

        Index sets follow *this* lattice ordering (``c5=(1,1), c6=(-1,1),
        c7=(1,-1), c8=(-1,-1)``): east-going ``E={1,5,7}``, west-going
        ``Wst={2,6,8}``, north ``{3,5,6}``, south ``{4,7,8}``.  Getting these
        sets wrong is the classic Zou-He bug -- it silently rescales the
        momentum flux, see "failure modes" in the case report.

        Inlet (west wall, unknown ``f1,f5,f7``, prescribed ``ux(y)``)::

            rho      = (f0 + f3 + f4 + 2*(f2+f6+f8)) / (1 - ux)      [mass + x-momentum]
            f1       = f2 + (2/3) rho ux                              [non-eq. bounce-back]
            f5 + f7  = rho ux + (f2+f6+f8) - f1
            f5 - f7  = rho uy - (f3-f4) - (f6-f8)

        Outlet (east wall, unknown ``f2,f6,f8``, prescribed ``rho = rho_out``)::

            ux       = (f0 + f3 + f4 + 2*(f1+f5+f7)) / rho - 1
            f2       = f1 - (2/3) rho ux
            f6 + f8  = (f1+f5+f7) - rho ux - f2
            f6 - f8  = -(f3-f4) - (f5-f7)                             [uy = 0]

        A *pressure* outlet (not a copied outflow) is required for the arm to be
        well posed: with an inlet velocity Dirichlet plus a body force and a
        copied outflow, mass cannot leave and the channel keeps accelerating
        (measured before the fix: 2.1x the analytic peak, rho in [0.91, 1.13]).
        """
        ux_in = np.zeros(self.ny)
        if self.cfg.inlet_from_analytic:
            ux_in[1:-1] = self._u_inlet                   # analytic profile (TCLB idea)
        else:
            # control: mass-equivalent PLUG inflow (same flow rate, flat profile)
            ux_in[1:-1] = float(np.mean(self._u_inlet))
        uy_in = np.zeros(self.ny)

        # ---------------- inlet (west) -------------------------------------- #
        f0, f2, f3, f4, f6, f8 = (f[0, :, 0], f[2, :, 0], f[3, :, 0],
                                  f[4, :, 0], f[6, :, 0], f[8, :, 0])
        rho = (f0 + f3 + f4 + 2.0 * (f2 + f6 + f8)) / (1.0 - ux_in)
        f1 = f2 + (2.0 / 3.0) * rho * ux_in
        s_ = rho * ux_in + (f2 + f6 + f8) - f1            # f5 + f7
        d_ = rho * uy_in - (f3 - f4) - (f6 - f8)          # f5 - f7
        f[1, :, 0], f[5, :, 0], f[7, :, 0] = f1, 0.5 * (s_ + d_), 0.5 * (s_ - d_)
        f[[1, 5, 7], 0, 0] = 0.0
        f[[1, 5, 7], -1, 0] = 0.0

        # ---------------- outlet (east, pressure) --------------------------- #
        rho_o = self.cfg.outflow_density
        g0, g1, g3, g4, g5, g7 = (f[0, :, -1], f[1, :, -1], f[3, :, -1],
                                  f[4, :, -1], f[5, :, -1], f[7, :, -1])
        ux_o = (g0 + g3 + g4 + 2.0 * (g1 + g5 + g7)) / rho_o - 1.0
        f2o = g1 - (2.0 / 3.0) * rho_o * ux_o
        s_o = (g1 + g5 + g7) - rho_o * ux_o - f2o        # f6 + f8
        d_o = -(g3 - g4) - (g5 - g7)                     # f6 - f8   (u_y = 0)
        f[2, :, -1], f[6, :, -1], f[8, :, -1] = f2o, 0.5 * (s_o + d_o), 0.5 * (s_o - d_o)
        f[[2, 6, 8], 0, -1] = 0.0
        f[[2, 6, 8], -1, -1] = 0.0
        self._outlet_ux = float(np.mean(ux_o[1:-1]))
        self._outlet_rho = rho_o

    def set_inlet_profile(self, u_inlet: np.ndarray) -> None:
        """Prescribed inlet velocity profile on the interior rows (analytic or plug)."""
        self._u_inlet = np.asarray(u_inlet, dtype=float)
        self._u_plug = float(np.mean(self._u_inlet))

    def step(self) -> None:
        f_post = self._collide()
        self.f = self._stream(f_post)
        self.step_count += 1

    # ------------------------------------------------------------------ #
    def velocity(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rho = self.f.sum(axis=0)
        safe = np.where(rho > 0.0, rho, 1.0)
        ux = ((self.f * CX[:, None, None]).sum(axis=0) + 0.5 * self.fx) / safe
        uy = (self.f * CY[:, None, None]).sum(axis=0) / safe
        return rho, ux, uy

    def profile(self, developed_from: Optional[float] = None) -> np.ndarray:
        """x-averaged u_x over the interior rows (optionally only the developed part)."""
        _, ux, _ = self.velocity()
        x0 = 0 if developed_from is None else int(round(developed_from))
        return ux[1:-1, x0:].mean(axis=1)

    def run(self, monitor: Optional[Callable[[int, "LayeredChannel"], None]] = None) -> Dict[str, float]:
        """Time-march until the predeclared steady-state criterion or ``steps``."""
        cfg = self.cfg
        x0 = None if cfg.x_boundary == "periodic" else int(round(1.5 * self.height))
        converged = False
        for _ in range(cfg.steps):
            self.step()
            if self.step_count % cfg.sample_every == 0:
                prof = self.profile(x0)
                if self._u_steady_ref is None:
                    self._u_steady_ref = prof.copy()
                    last_change = np.inf
                else:
                    last_change = float(
                        np.max(np.abs(prof - self._u_steady_ref)) / max(np.max(np.abs(prof)), 1e-30)
                    )
                    self._u_steady_ref = prof.copy()
                if monitor is not None:
                    monitor(self.step_count, self)
                if last_change < cfg.tol_steady:
                    converged = True
                    break
        return {"steps": float(self.step_count), "converged": float(converged),
                "last_rel_change": float(last_change if self._u_steady_ref is not None else np.nan)}