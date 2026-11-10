"""Void-fraction transport for a microbubble layer (IN62-N04).

Governing equation (as specified for this node):

    d(alpha)/dt + div(alpha u) = div(D grad alpha) ,

with ``u`` the carrier (liquid) velocity and ``D`` the turbulent dispersion
coefficient.  For a thin bubble layer on a flat plate the steady balance is
one-dimensional in the wall-normal direction ``y``:

    D d2(alpha)/dy2 - v_s d(alpha)/dy = 0 ,

``v_s`` being the wall-normal entrainment ("sweep") velocity of the turbulent
boundary layer.  With ``alpha(0) = alpha_w`` and ``alpha(H) = 0``

    alpha(y) = alpha_w sinh(Pe (1 - y/H)) / sinh(Pe),   Pe = v_s H / D .

Three solvers are provided so that the LBM can be checked against something
that is *not* itself an LBM:

* :func:`alpha_profile_analytic` -- the closed form above;
* :class:`TransportFV`  -- finite-volume, conservative (structural mass ledger);
* :class:`TransportLBM` -- D2Q9 advection-diffusion LBM (BGK, anti-bounce-back
  Dirichlet walls), i.e. the lattice solve of the same equation.

Scaling of ``D`` and ``v_s`` (no free constants):

    v_s = c_e u_tau ,        (entrainment velocity, c_e = 0.1 of u_tau)
    D   = kappa u_tau (H/2) / Sc_t ,   (mixing-length eddy diffusivity)

so ``Pe = 2 c_e Sc_t / kappa`` is independent of the scale -- the profile is
self-similar, which is why a single non-dimensional profile can be used for
every ``Re`` in the campaign.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from .loglaw import KAPPA

#: entrainment fraction and turbulent Schmidt number for the bubble layer
C_ENTRAIN = 0.1
SC_T = 0.9
#: bubble-layer thickness as a fraction of the boundary-layer thickness
LAYER_FRACTION = 0.1


@dataclass(frozen=True)
class TransportConfig:
    """One void-fraction transport problem (lattice units unless stated)."""

    ny: int = 64
    H: float = 32.0
    D: float = 0.08
    v_sweep: float = 0.02
    alpha_w: float = 0.2
    capture: float = 0.3
    dt: float = 1.0

    @property
    def dy(self) -> float:
        return self.H / self.ny

    @property
    def peclet(self) -> float:
        return self.v_sweep * self.H / self.D

    def with_flow(self, u_tau: float, delta: float) -> "TransportConfig":
        """Config for a boundary layer of thickness ``delta`` and friction
        velocity ``u_tau`` (same units): sets ``H``, ``D`` and ``v_s``."""
        H = LAYER_FRACTION * delta
        D = KAPPA * u_tau * (0.5 * H) / SC_T
        v_s = C_ENTRAIN * u_tau
        return replace(self, H=H, D=D, v_sweep=v_s)


def alpha_profile_analytic(cfg: TransportConfig, y) -> np.ndarray:
    """Steady profile, exact solution of ``D a'' - v_s a' = 0``.

    With ``alpha(0) = alpha_w`` and ``alpha(H) = 0``:

        alpha(y) = alpha_w (exp(Pe) - exp(Pe y/H)) / (exp(Pe) - 1) ,

    which tends to the linear profile ``alpha_w (1 - y/H)`` for ``Pe -> 0``.
    (Note that ``sinh`` is *not* a solution of this first-order-in-``alpha'``
    balance: the drift makes the equation ``alpha' = c exp(v_s y/D)``.)
    """
    y = np.asarray(y, dtype=float)
    Pe = cfg.peclet
    if abs(Pe) < 1e-8:
        return cfg.alpha_w * np.clip(1.0 - y / cfg.H, 0.0, 1.0)
    return cfg.alpha_w * (math.exp(Pe) - np.exp(Pe * y / cfg.H)) / (math.exp(Pe) - 1.0)


# ---------------------------------------------------------------------------
# finite-volume reference solver
# ---------------------------------------------------------------------------


class TransportFV:
    """Conservative finite-volume solver for the 1-D steady/transient problem.

    Faces are at ``y = i dy`` (``i = 0 .. ny``); cell centres at ``(i+0.5) dy``.
    The discrete balance ``d(inventory)/dt = J_0 - J_ny - capture`` is exact by
    construction, so the mass ledger is structural rather than numerical.
    """

    def __init__(self, cfg: TransportConfig):
        self.cfg = cfg
        # explicit stability of the central-difference FV scheme:
        #   D dt/dy^2 <= 0.4  (diffusion)   and   v_s dt/dy <= 0.4 (advection)
        dt_max = 0.4 * min(cfg.dy ** 2 / max(cfg.D, 1e-30),
                           cfg.dy / max(cfg.v_sweep, 1e-30))
        self.n_sub = max(1, int(math.ceil(cfg.dt / dt_max)))
        self.dt_sub = cfg.dt / self.n_sub
        self.alpha = np.zeros(cfg.ny)
        self.injected = 0.0
        self.captured = 0.0
        self.outlet = 0.0

    def _fluxes(self, a: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        dy = cfg.dy
        J = np.zeros(cfg.ny + 1)
        J[1:-1] = -cfg.D * (a[1:] - a[:-1]) / dy + cfg.v_sweep * 0.5 * (a[1:] + a[:-1])
        J[0] = -cfg.D * (a[0] - cfg.alpha_w) / (0.5 * dy) + \
            cfg.v_sweep * 0.5 * (a[0] + cfg.alpha_w)
        J[-1] = -cfg.D * (0.0 - a[-1]) / (0.5 * dy) + cfg.v_sweep * 0.5 * a[-1]
        return J

    def step(self) -> None:
        cfg = self.cfg
        for _ in range(self.n_sub):
            J = self._fluxes(self.alpha)
            J_in = max(0.0, float(J[0]))
            cap = cfg.capture * J_in
            self.injected += J_in * self.dt_sub
            self.captured += cap * self.dt_sub
            self.outlet += max(0.0, float(J[-1])) * self.dt_sub
            J_net = J.copy()
            J_net[0] = (1.0 - cfg.capture) * J_in
            self.alpha = self.alpha + (self.dt_sub / cfg.dy) * (J_net[:-1] - J_net[1:])

    def run(self, n_steps: int = 20000, tol: float = 1e-12) -> None:
        for _ in range(n_steps):
            before = self.alpha.copy()
            self.step()
            if np.max(np.abs(self.alpha - before)) < tol:
                break

    def profile(self) -> np.ndarray:
        return self.alpha.copy()

    @property
    def y(self) -> np.ndarray:
        return (np.arange(self.cfg.ny) + 0.5) * self.cfg.dy

    def supply(self) -> dict:
        """Steady gas budget per unit wall area (lattice velocity units)."""
        J = self._fluxes(self.alpha)
        J_in = max(0.0, float(J[0]))
        J_out = max(0.0, float(J[-1]))
        J_cap = self.cfg.capture * J_in
        return {
            "supply": J_in,
            "captured": J_cap,
            "outlet": J_out,
            "inventory": float(self.alpha.sum()) * self.cfg.dy,
            "closure_residual": J_in - J_cap - J_out,
        }


# ---------------------------------------------------------------------------
# lattice Boltzmann solver (D2Q9 advection-diffusion)
# ---------------------------------------------------------------------------

_CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
_CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
_W = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9,
               1 / 36, 1 / 36, 1 / 36, 1 / 36])
_OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
CS2 = 1.0 / 3.0


class TransportLBM:
    """D2Q9 BGK advection-diffusion LBM for the void fraction.

    The drift velocity is prescribed (one-way coupling): ``alpha`` is a passive
    scalar in the given turbulent velocity field ``u = (0, v_s)``.  Wall
    conditions are Dirichlet values imposed with the classic anti-bounce-back
    rule ``g_i = -g_opp(i) + 2 w_i alpha_bc``.
    """

    def __init__(self, cfg: TransportConfig, nx: int = 1,
                 alpha_init: float = 0.0):
        self.cfg = cfg
        self.nx = nx
        self.tau_d = 0.5 + cfg.D / CS2
        self.w = _W
        self.alpha = np.full((cfg.ny, nx), alpha_init)
        self.g = np.zeros((9, cfg.ny, nx))
        for i in range(9):
            self.g[i] = _W[i] * self.alpha
        self.steps = 0

    # -- helpers ------------------------------------------------------------
    def _velocity(self) -> np.ndarray:
        """Prescribed drift velocity field ``(ny, nx, 2)`` = ``(0, v_s)``."""
        u = np.zeros((self.cfg.ny, self.nx, 2))
        u[:, :, 1] = self.cfg.v_sweep
        return u

    def _equilibrium(self, alpha: np.ndarray, u: np.ndarray) -> np.ndarray:
        geq = np.zeros((9, self.cfg.ny, self.nx))
        for i in range(9):
            cu = _CX[i] * u[:, :, 0] + _CY[i] * u[:, :, 1]
            geq[i] = _W[i] * alpha * (1.0 + cu / CS2)
        return geq

    # -- solver -------------------------------------------------------------
    def step(self) -> None:
        cfg = self.cfg
        u = self._velocity()
        geq = self._equilibrium(self.alpha, u)
        gp = self.g + (geq - self.g) / self.tau_d      # BGK collision
        ny = cfg.ny
        new = np.zeros_like(gp)
        for i in range(9):
            new[i] = np.roll(gp[i], shift=(_CY[i], _CX[i]), axis=(0, 1))
        # walls (half-way, Dirichlet alpha_w at y=0 and 0 at y=H)
        for i in range(9):
            if _CY[i] > 0:      # entering the domain from the bottom wall
                new[i, 0, :] = -gp[_OPP[i], 0, :] + 2.0 * _W[i] * cfg.alpha_w
            elif _CY[i] < 0:    # entering from the top wall (alpha = 0)
                new[i, ny - 1, :] = -gp[_OPP[i], ny - 1, :]
        self.g = new
        self.alpha = self.g.sum(axis=0)
        self.steps += 1

    def run(self, n_steps: int = 20000, tol: float = 1e-13) -> None:
        for _ in range(n_steps):
            before = self.alpha.copy()
            self.step()
            if np.max(np.abs(self.alpha - before)) < tol:
                break

    @property
    def y(self) -> np.ndarray:
        return (np.arange(self.cfg.ny) + 0.5) * self.cfg.dy

    def profile(self) -> np.ndarray:
        return self.alpha.mean(axis=1)


def alpha_profile(cfg_or_lbm) -> np.ndarray:
    """Void-fraction profile from a solver instance."""
    return cfg_or_lbm.profile()


def mass_ledger(lbm, capture: float = 0.3) -> dict:
    """Mass ledger of a transport solve (per unit wall area).

    ``injected`` = ``alpha_w v_s`` in-flux implied by the steady profile,
    ``outlet`` = out-flux at ``y = H``, ``captured`` = wall sink.  The residual
    is normalised by the in-flux so that it is a dimensionless closure error.
    """
    a = lbm.profile()
    cfg = lbm.cfg
    if hasattr(lbm, "_fluxes"):
        J = lbm._fluxes(a)
        J_in, J_out = max(0.0, float(J[0])), max(0.0, float(J[-1]))
    else:
        # central differences on the lattice profile
        dy = cfg.dy
        J_in = -cfg.D * (a[0] - cfg.alpha_w) / (0.5 * dy) + \
            cfg.v_sweep * 0.5 * (a[0] + cfg.alpha_w)
        J_out = -cfg.D * (0.0 - a[-1]) / (0.5 * dy) + cfg.v_sweep * 0.5 * a[-1]
    cap = capture * J_in
    return {
        "injected": J_in,
        "captured": cap,
        "outlet": J_out,
        "inventory": float(a.sum()) * cfg.dy,
        "residual": J_in - cap - J_out,
        "closure_error": (J_in - cap - J_out) / J_in if J_in > 0 else 0.0,
    }


def analytic_supply(cfg: TransportConfig) -> dict:
    """Gas budget of the steady profile in closed form (per unit wall area).

    The steady flux is uniform: ``J = v_s alpha_w exp(Pe)/(exp(Pe)-1)`` in
    lattice velocity units; the wall sink removes ``capture x J`` and the rest
    leaves through ``y = H``.
    """
    Pe = cfg.peclet
    J = cfg.v_sweep * cfg.alpha_w * (math.exp(Pe) / (math.exp(Pe) - 1.0)
                                    if abs(Pe) > 1e-9 else 1.0 / max(Pe, 1e-30))
    if abs(Pe) < 1e-9:
        J = cfg.v_sweep * cfg.alpha_w
    # J is the uniform interior flux; the wall sink captures a fraction
    # ``capture`` of what is injected, so the *injected* flux at the wall is
    # J/(1-capture) and it is that gas the compressor must supply.
    J_wall = J / max(1.0 - cfg.capture, 1e-12)
    return {
        "supply": J_wall,
        "interior_flux": J,
        "captured": cfg.capture * J_wall,
        "outlet": J,
        "inventory": float(np.trapezoid(
            alpha_profile_analytic(cfg, (np.arange(cfg.ny) + 0.5) * cfg.dy),
            (np.arange(cfg.ny) + 0.5) * cfg.dy)),
    }


def buffer_layer_alpha(cfg: TransportConfig, y_plus_sample: float,
                       u_tau: float, nu: float) -> float:
    """Void fraction at ``y+ = y_plus_sample`` (clipped inside the layer).

    This is ``alpha_buffer`` of the mixing-length closure: the log-law slope is
    set by the void fraction where the turbulent shear stress is produced, not
    by the wall value.
    """
    y = y_plus_sample * nu / u_tau
    y = min(max(y, 0.0), cfg.H)
    return float(alpha_profile_analytic(cfg, [y])[0])