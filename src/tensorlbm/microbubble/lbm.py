"""Two-phase mixture LBM for a bubbly wall layer (IN62-N04).

Physics and level of description
--------------------------------
A microbubble drag-reduction boundary layer contains bubbles of diameter
``d_b ~ 10..100 um`` in a layer of thickness ``delta_b ~ 100 um .. 1 mm`` while
the boundary layer itself is ``delta ~ 10 mm``.  The scale separation
``delta/d_b ~ 10^2`` forbids resolving individual interfaces over the plate
length (that is the regime of the interface-resolved Shan-Chen work in the
``bubble_dr`` module, which is restricted to single bubbles / small patches).
The correct level of description here is therefore the **Euler-Euler two-phase
mixture (void-fraction) model**:

    rho(x) = rho_l (1 - alpha(x)) ,      mu(x) = mu_l (1 - alpha(x)) ,

i.e. the gas carries no momentum (``rho_g/rho_l ~ 1e-3``) but removes viscous
shear from the region it occupies.  :class:`MixtureChannelLBM` solves the
incompressible mixture momentum equation with a D2Q9 BGK lattice
(variable relaxation time per row) at matched volume flow rate, so that

    DR = 1 - G_bubbly / G_clean        (G = driving pressure gradient)

is a genuine two-phase drag reduction (matched throughput is what makes the
comparison meaningful).

Exact anchor
------------
For a channel of height ``H`` with a wall-normal viscosity profile ``nu(y)`` the
steady momentum balance is closed-form:

    tau(y) = tau_w - G y ,     du/dy = tau(y)/nu(y) ,

with ``(tau_w, G)`` fixed by no-slip at both walls and by the prescribed bulk
velocity.  :func:`exact_two_layer_channel` solves that problem on a fine grid
(independent of the LBM lattice), giving the anchor the lattice run is graded
against.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .loglaw import KAPPA  # re-exported for the report's closure table

CS2 = 1.0 / 3.0
_CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
_CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
_W = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
_OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])


def mixture_viscosity(alpha: float, nu_liquid: float,
                      nu_gas_ratio: float = 20.0) -> float:
    """Euler-Euler mixture viscosity ``nu_m = nu_l (1-alpha) + nu_g alpha``.

    ``nu_g = nu_l / nu_gas_ratio``; the ratio is capped well below the physical
    water/air value (55) because a lattice BGK run with ``tau -> 0.5`` becomes
    numerically stiff.  The cap is a *numerical* limitation and is stated in
    the report; the closure used for the turbulent regime does not depend on
    it (there the void fraction enters only through the mixing length).
    """
    return nu_liquid * (1.0 - alpha) + nu_liquid / nu_gas_ratio * alpha


# ---------------------------------------------------------------------------
# exact anchor
# ---------------------------------------------------------------------------


def exact_two_layer_channel(nu_of_y, H: float, U_bulk: float, n: int = 4001) -> dict:
    """Exact steady solution of the mixture channel problem.

    ``nu_of_y`` maps a wall-normal position ``y in [0, H]`` to the mixture
    kinematic viscosity.  Returns the pressure gradient, the two wall shears
    and the velocity profile, obtained from

        u(y) = tau_w A(y) - G B(y),   A(y) = int_0^y dy'/nu ,  B(y) = int_0^y y' dy'/nu ,

    with ``u(0) = u(H) = 0`` and ``(1/H) int u dy = U_bulk``.
    """
    y = np.linspace(0.0, H, n)
    nu = np.asarray([nu_of_y(float(yy)) for yy in y], dtype=float)
    inv = 1.0 / nu
    A = np.concatenate([[0.0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(y))])
    B = np.concatenate([[0.0],
                        np.cumsum(0.5 * (y[1:] * inv[1:] + y[:-1] * inv[:-1]) * np.diff(y))])
    A_H, B_H = A[-1], B[-1]
    Abar = np.trapezoid(A, y) / H
    Bbar = np.trapezoid(B, y) / H
    # u(H) = 0  ->  tau_w = G B_H / A_H ;  mean velocity -> G
    denom = Abar * B_H / A_H - Bbar
    G = U_bulk / denom
    tau_w = G * B_H / A_H
    u = tau_w * A - G * B
    tau = tau_w - G * y
    return {
        "G": float(G),
        "tau_w_bottom": float(tau[0]),
        "tau_w_top": float(-tau[-1]),
        "u": u,
        "y": y,
        "Cf": float(2.0 * G * H / (2.0 * U_bulk ** 2)),
    }


def two_layer_drag_reduction(H: float, U_bulk: float, nu_liquid: float,
                             alpha_layer: float, h_layer: float,
                             nu_gas_ratio: float = 20.0) -> dict:
    """Exact DR of a bubbly layer of thickness ``h_layer`` on one wall."""
    def nu_clean(_y):
        return nu_liquid

    def nu_bubbly(y):
        in_layer = (y <= h_layer) and h_layer > 0.0
        return mixture_viscosity(alpha_layer if in_layer else 0.0,
                                 nu_liquid, nu_gas_ratio)

    clean = exact_two_layer_channel(nu_clean, H, U_bulk)
    bub = exact_two_layer_channel(nu_bubbly, H, U_bulk)
    return {
        "G_clean": clean["G"],
        "G_bubbly": bub["G"],
        "DR": 1.0 - bub["G"] / clean["G"],
        "clean": clean,
        "bubbly": bub,
    }


# ---------------------------------------------------------------------------
# lattice Boltzmann mixture channel
# ---------------------------------------------------------------------------


@dataclass
class ChannelConfig:
    """Mixture channel run (lattice units)."""

    ny: int = 48
    nx: int = 1
    tau_liquid: float = 0.8
    nu_gas_ratio: float = 20.0
    U_target: float = 0.05
    alpha_layer: float = 0.0
    h_layer: float = 0.0          # in lattice units (wall-normal thickness)
    n_steps: int = 20000
    ctrl_every: int = 20
    G0: float | None = None
    seed: int = 0
    relax: float = 0.5            # controller relaxation exponent

    @property
    def nu_liquid(self) -> float:
        return CS2 * (self.tau_liquid - 0.5)

    @property
    def H(self) -> float:
        """Distance between the two wall faces (lattice units)."""
        return float(self.ny - 2)

    def alpha_of_row(self) -> np.ndarray:
        """Void fraction of each fluid row (0 = bottom wall layer)."""
        rows = np.arange(self.ny - 2)            # fluid rows 1..ny-2
        y = rows + 0.5                           # distance from the bottom face
        return np.where(y <= self.h_layer, self.alpha_layer, 0.0)


class MixtureChannelLBM:
    """D2Q9 BGK channel with a wall-normal void-fraction profile.

    A multiplicative controller holds the bulk velocity at ``U_target`` (for a
    laminar channel ``U_bulk`` is linear in the driving gradient, so the update
    ``G <- G U_target/U_bulk`` is exact after one control step); the converged
    gradient is the drag measure.
    """

    def __init__(self, cfg: ChannelConfig):
        self.cfg = cfg
        self.alpha = cfg.alpha_of_row()
        nu = np.array([mixture_viscosity(float(a), cfg.nu_liquid,
                                        cfg.nu_gas_ratio) for a in self.alpha])
        self.nu_rows = nu
        # relaxation time per fluid row; clip for stability
        self.tau_rows = np.clip(0.5 + nu / CS2, 0.5009, 1.98)
        # laminar estimate G = 12 nu U/H^2 as the controller starting point
        self.G = (cfg.G0 if cfg.G0 is not None
                  else 12.0 * cfg.nu_liquid * cfg.U_target / cfg.H ** 2)
        rho = np.ones((cfg.ny, cfg.nx))
        self.f = np.zeros((9, cfg.ny, cfg.nx))
        # start from the laminar clean-channel profile (steady state of the
        # alpha = 0 problem) so that no long viscous transient is needed
        y = np.arange(cfg.ny) - 0.5
        H = cfg.H
        u0 = 6.0 * cfg.U_target * y * (H - y) / H ** 2
        u0 = u0[:, None] * np.ones((1, cfg.nx))
        for i in range(9):
            cu = _CX[i] * u0
            self.f[i] = _W[i] * rho * (1.0 + cu / CS2 + 0.5 * (cu / CS2) ** 2
                                       - 0.5 * u0 ** 2 / CS2)
        # solid rows carry no populations
        self.f[:, 0, :] = 0.0
        self.f[:, cfg.ny - 1, :] = 0.0
        self.history: list[float] = []
        self.steps_done = 0
        # no control before the viscous time has elapsed, otherwise the
        # controller integrates against a flow that has not responded yet
        self.warmup = max(400, int(1.0 * cfg.H ** 2 / cfg.nu_liquid))

    # -- helpers ------------------------------------------------------------
    def _macroscopic(self):
        f = self.f
        rho = f.sum(axis=0)
        rho = np.where(rho <= 0, 1.0, rho)
        ux = (f[1] - f[3] + f[5] - f[6] - f[7] + f[8]) / rho
        uy = (f[2] - f[4] + f[5] + f[6] - f[7] - f[8]) / rho
        return rho, ux, uy

    def _equilibrium(self, rho, ux, uy, force_x):
        tau = np.zeros((self.cfg.ny, 1))
        tau[1:self.cfg.ny - 1, 0] = self.tau_rows
        ux_eq = ux + tau * force_x / rho      # forcing by velocity shift
        uy_eq = uy
        geq = np.zeros_like(self.f)
        for i in range(9):
            cu = _CX[i] * ux_eq + _CY[i] * uy_eq
            geq[i] = _W[i] * rho * (1.0 + cu / CS2 + 0.5 * (cu / CS2) ** 2
                                    - 0.5 * (ux_eq ** 2 + uy_eq ** 2) / CS2)
        return geq

    # -- one step -----------------------------------------------------------
    def step(self) -> None:
        cfg = self.cfg
        rho, ux, uy = self._macroscopic()
        geq = self._equilibrium(rho, ux, uy, self.G)
        tau_full = np.ones((cfg.ny, cfg.nx))
        tau_full[1:cfg.ny - 1, :] = self.tau_rows[:, None]
        fpost = self.f - (self.f - geq) / tau_full
        ny = cfg.ny
        new = np.zeros_like(fpost)
        for i in range(9):
            new[i] = np.roll(fpost[i], shift=(_CY[i], _CX[i]), axis=(0, 1))
        # bounce-back on both walls (half-way)
        for i in range(9):
            if _CY[i] > 0:
                new[i, 1, :] = fpost[_OPP[i], 1, :]
            elif _CY[i] < 0:
                new[i, ny - 2, :] = fpost[_OPP[i], ny - 2, :]
        new[:, 0, :] = 0.0
        new[:, ny - 1, :] = 0.0
        self.f = new

    def control(self) -> None:
        u_bulk = self.bulk_velocity()
        self.history.append(u_bulk)
        if self.steps_done < self.warmup or u_bulk < 1e-12:
            return
        ratio = self.cfg.U_target / u_bulk
        ratio = min(max(ratio, 0.8), 1.25)             # damped update
        self.G *= ratio ** self.cfg.relax

    def wall_shear_ladd(self) -> float:
        """Wall shear from Ladd momentum exchange on the bottom wall.

        Momentum transferred to the wall per step by the bounce-back on the
        boundary links: ``sum_{c_iy<0} 2 c_ix f_i^post`` (per unit length).
        The tangential (streamwise) component is the wall shear.
        """
        _, ux, _ = self._macroscopic()
        rho, _, _ = self._macroscopic()
        geq = self._equilibrium(rho, ux, np.zeros_like(ux), self.G)
        tau_full = np.ones((self.cfg.ny, self.cfg.nx))
        tau_full[1:self.cfg.ny - 1, :] = self.tau_rows[:, None]
        fpost = self.f - (self.f - geq) / tau_full
        total = 0.0
        for i in range(9):
            if _CY[i] < 0:
                total += 2.0 * _CX[i] * float(fpost[i, 1, :].sum())
        return total / self.cfg.nx

    def run(self, n_steps: int | None = None) -> None:
        n = n_steps or self.cfg.n_steps
        for k in range(n):
            self.step()
            self.steps_done += 1
            if self.steps_done % self.cfg.ctrl_every == 0:
                self.control()

    def bulk_velocity(self) -> float:
        _, ux, _ = self._macroscopic()
        return float(ux[1:-1, :].mean())

    def profile(self) -> np.ndarray:
        _, ux, _ = self._macroscopic()
        return ux[1:-1, :].mean(axis=1)

    @property
    def y(self) -> np.ndarray:
        return np.arange(self.cfg.ny - 2) + 0.5