"""Two-component Shan-Chen (MCMP) LBM for layered plane-Poiseuille flow.

Model
-----
Two mass densities ``rho_A``, ``rho_B`` with one distribution function each.
The components interact through the standard Shan-Chen pseudopotential force

    F_sigma(x) = -G * psi_sigma(x) * sum_i w_i psi_sigma'(x + c_i) c_i

with ``psi_sigma(rho) = rho0 * (1 - exp(-rho/rho0))``.  ``G > 0`` is repulsive,
so the two components demix into two layers; each component keeps its own
relaxation time ``tau_sigma`` and therefore its own kinematic viscosity

    nu_sigma = cs^2 * (tau_sigma - 1/2)        (cs^2 = 1/3)

Force injection uses the exact-difference method (EDM, Kupershtokh 2004):

    f_sigma^+ = f_sigma - (f_sigma - f_sigma^eq(rho_sigma, u*)) / tau_sigma
                        + [ f_sigma^eq(rho_sigma, u* + F_sigma/rho_sigma)
                            - f_sigma^eq(rho_sigma, u*) ]

where ``u*`` is the *common* (mixture) velocity of the two components, which is
what makes a layered co-flow possible.  The physical velocity reported by
:meth:`TwoComponentSCSolver.get_macroscopic` includes the half-force shift
``F_total / (2 rho_total)``.

Boundaries: periodic in ``x``; halfway bounce-back no-slip at the first and
last rows of the grid (``y = 0`` and ``y = ny-1`` are solid rows, the fluid
rows are ``1 .. ny-2``).  Solid neighbours of a fluid node contribute
``psi = wall_psi`` (0.0 = neutral wetting) to the interaction force.

Kinematic viscosity of a pure component is independent of its density in this
model, and both components are pushed with the *same* body-force acceleration
``g_x`` per unit mass, so the steady state of the layered case is exactly the
piecewise-quadratic solution of

    nu_j u_j'' = -g_x,   u(0)=u(H)=0,   u and nu*u' continuous at interfaces

computed independently by :mod:`.analytic` (``solve_layered``).

References
----------
Shan & Chen (1993) Phys. Rev. E 47, 1815; Shan & Doolen (1995) J. Stat. Phys.
81, 379; Martys & Chen (1996) Phys. Rev. E 53, 743; Kupershtokh (2004) Comput.
Math. Appl. 48, 1141 (EDM); Guo, Zheng & Shi (2002) Phys. Rev. E 65, 046308.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from .analytic import solve_layered
from ...multiphase import (collide_edm_common, psi_exp, stream_bounce_back,
                           sc_two_component_force)

# --------------------------------------------------------------------------- #
# D2Q9 lattice
# --------------------------------------------------------------------------- #
CS2 = 1.0 / 3.0

_W = [4.0 / 9.0] + [1.0 / 9.0] * 4 + [1.0 / 36.0] * 4
_C = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)]
_OPP = [0, 3, 4, 1, 2, 7, 8, 5, 6]

W = torch.tensor(_W, dtype=torch.float64)
C = torch.tensor(_C, dtype=torch.float64)
OPPOSITE = list(_OPP)


def _dt(dtype: torch.dtype) -> torch.dtype:
    return dtype


# --------------------------------------------------------------------------- #
# Elementary operators (kept importable / unit-testable)
# --------------------------------------------------------------------------- #
def equilibrium(rho: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    """D2Q9 equilibrium ``f^eq = w_i rho [1 + cu/cs^2 + (cu)^2/(2cs^4) - u^2/(2cs^2)]``."""
    from ...multiphase import equilibrium as _eq
    return _eq(rho, ux, uy)


def macroscopic(f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Density and (unshifted) momentum velocity of a single component."""
    from ...multiphase import macroscopic as _macro
    return _macro(f)


def stream_bounce_back_y(
    f: torch.Tensor,
    solid_top: Optional[torch.Tensor] = None,
    solid_bot: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Streaming: periodic in ``x``, halfway bounce-back on the first/last row.

    Delegates to the library kernel :func:`tensorlbm.multiphase.stream_bounce_back`
    (the same one the production SC step :func:`tensorlbm.multiphase.step_sc`
    uses).  Kept as a named alias because the earlier version of this module -
    and the lesson it carries - lived here: it rolled only in ``x``
    (``shifts=(0, -cx)``) and then overwrote the two wall rows, so the vertical
    directions were never advected: a body-forced run gave ``u_max = g*t``
    (ballistic, flat profile) instead of a viscous Poiseuille solution.
    """
    if solid_top is None:
        solid_top = torch.zeros(f.shape[1:], dtype=torch.bool, device=f.device)
        solid_top[0, :] = True
    if solid_bot is None:
        solid_bot = torch.zeros(f.shape[1:], dtype=torch.bool, device=f.device)
        solid_bot[-1, :] = True
    solid = solid_top | solid_bot
    return stream_bounce_back(f, solid)


# --------------------------------------------------------------------------- #
# Configuration / solver
# --------------------------------------------------------------------------- #
@dataclass
class TwoComponentSCConfig:
    """Configuration of a two-component SC channel run.

    All quantities are in lattice units.  ``ny``/``nx`` count *cells including
    the two wall rows*, hence the fluid height is ``H = ny - 2``.
    """

    ny: int
    nx: int
    tau_a: float
    tau_b: float
    G: float
    g_x: float
    g_y: float
    steps: int
    init_rho_a: float = 2.0
    init_rho_b: float = 2.0
    init_layer_frac: float = 0.5  # fraction of the fluid height occupied by A, from the bottom
    rho0_psi: float = 1.0
    forcing: str = "edm"  # "edm" | "guo" | "velocity_shift"
    sample_every: int = 100
    device: str = "cpu"
    seed: int = 20261115
    dtype: str = "float64"
    wall_psi: float = 0.0
    wall_mode: str = "mirror"  # "mirror" = neutral wetting (psi_solid = psi of the mirror fluid node) | "dry" (psi_solid = wall_psi)
    noise: float = 0.0
    interface_width: float = 1.0  # tanh half-width of the initial A/B interface (0 = step)
    g_ramp_steps: int = 0  # ramp G from 0 to its target over this many steps (adiabatic demixing)
    u_eq: str = "viscous"  # equilibrium velocity coupling: "viscous" (Martys & Chen 1996) | "mixture" (density weighted)

    # -- derived ---------------------------------------------------------- #
    @property
    def height(self) -> float:
        return float(self.ny - 2)

    @property
    def nu_a(self) -> float:
        return CS2 * (self.tau_a - 0.5)

    @property
    def nu_b(self) -> float:
        return CS2 * (self.tau_b - 0.5)


class TwoComponentSCSolver:
    """Two-component Shan-Chen solver with pluggable forcing schemes."""

    def __init__(self, cfg: TwoComponentSCConfig):
        self.cfg = cfg
        self.ny, self.nx = cfg.ny, cfg.nx
        self.device = torch.device(cfg.device)
        self.dtype = getattr(torch, cfg.dtype)
        self.H = float(cfg.ny - 2)

        mask = torch.zeros(self.ny, self.nx, dtype=torch.bool, device=self.device)
        mask[0, :] = True
        mask[-1, :] = True
        self.solid = mask
        self.solid_top = mask.clone()
        self.solid_bot = mask.clone()
        self.fluid = ~mask

        gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
        self.f_a = self._initialize_component(cfg.init_rho_a, cfg.init_layer_frac, is_a=True, gen=gen)
        self.f_b = self._initialize_component(cfg.init_rho_b, cfg.init_layer_frac, is_a=False, gen=gen)
        self.step_count = 0

    # ------------------------------------------------------------------ #
    def _initialize_component(self, rho0, layer_frac, is_a, gen) -> torch.Tensor:
        """Rest state: component A fills the bottom ``layer_frac`` of the fluid,
        component B the complementary top part (``layer_frac`` is shared so that
        A and B meet at the same interface)."""
        rho = torch.zeros(self.ny, self.nx, dtype=self.dtype, device=self.device)
        s = torch.arange(1, self.ny - 1, dtype=self.dtype, device=self.device) - 0.5
        xi = (s - layer_frac * self.H) / max(self.cfg.interface_width, 1e-6)
        if is_a:
            frac = 0.5 * (1.0 - torch.tanh(xi))
        else:
            frac = 0.5 * (1.0 + torch.tanh(xi))
        filled = (float(rho0) * frac).view(-1, 1).expand(-1, self.nx).contiguous()
        rho[1:-1, :] = filled
        if self.cfg.noise > 0.0:
            n = torch.randn(rho.shape, generator=gen, dtype=torch.float32)
            rho = rho + self.cfg.noise * n.to(dtype=self.dtype, device=self.device)
        rho = torch.clamp(rho, min=1e-3)
        rho[self.solid] = 0.0
        ux = torch.zeros_like(rho)
        uy = torch.zeros_like(rho)
        return equilibrium(rho, ux, uy)

    @property
    def effective_G(self) -> float:
        """Interaction strength at the current step (0 while the ramp is running)."""
        n = self.cfg.g_ramp_steps
        if n and self.step_count < n:
            return self.cfg.G * (self.step_count / float(n))
        return self.cfg.G

    # ------------------------------------------------------------------ #
    def _interaction_force(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """SC force on A and B + external body force; returns ``(fx_a, fy_a, fx_b, fy_b)``.

        Delegates to :func:`tensorlbm.multiphase.sc_two_component_force` so the
        verification runs the *shipped* two-component SC kernel, with the
        interaction strength ``G`` ramped from 0 (see :attr:`effective_G`).
        """
        rho_a, _, _ = macroscopic(self.f_a)
        rho_b, _, _ = macroscopic(self.f_b)
        return sc_two_component_force(
            rho_a, rho_b, G=self.effective_G, solid_mask=self.solid,
            wall_psi=self.cfg.wall_psi, wall_mode=self.cfg.wall_mode,
            gx=self.cfg.g_x, gy=self.cfg.g_y,
        )

    def _component_states(self):
        rho_a, ux_a, uy_a = macroscopic(self.f_a)
        rho_b, ux_b, uy_b = macroscopic(self.f_b)
        return rho_a, ux_a, uy_a, rho_b, ux_b, uy_b

    def _common_velocity(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Equilibrium velocity ``u*`` shared by both components, and total density.

        ``u_eq="viscous"`` uses the Martys & Chen (1996) viscosity-weighted
        coupling ``u* = sum_sigma (rho_sigma u_sigma / tau_sigma) /
        sum_sigma (rho_sigma / tau_sigma)``; ``u_eq="mixture"`` the plain
        density-weighted mixture velocity.  The viscous coupling is what keeps a
        *layered* co-flow consistent when the two components have different
        relaxation times (each layer then relaxes its own momentum), and it is
        the coupling used by the sibling MCMP stratified-Poiseuille campaign.
        No half-force shift is applied here (that is only done on output).
        """
        rho_a, ux_a, uy_a, rho_b, ux_b, uy_b = self._component_states()
        rho_t = rho_a + rho_b
        if self.cfg.u_eq == "mixture":
            inv = 1.0 / torch.clamp(rho_t, min=1e-12)
            ux = (rho_a * ux_a + rho_b * ux_b) * inv
            uy = (rho_a * uy_a + rho_b * uy_b) * inv
        elif self.cfg.u_eq == "viscous":
            wa = rho_a / self.cfg.tau_a
            wb = rho_b / self.cfg.tau_b
            inv = 1.0 / torch.clamp(wa + wb, min=1e-12)
            ux = (wa * ux_a + wb * ux_b) * inv
            uy = (wa * uy_a + wb * uy_b) * inv
        else:
            raise ValueError(f"Unknown u_eq: {self.cfg.u_eq}")
        return ux, uy, rho_t

    # -- forcing schemes ------------------------------------------------- #
    def _collide_edm(self, f, rho, ux, uy, fx, fy, tau):
        """EDM at the common (viscosity-weighted) velocity -- library kernel."""
        return collide_edm_common(f, rho, ux, uy, fx, fy, tau)

    def _collide_guo(self, f, rho, ux, uy, fx, fy, tau):
        inv_rho = 1.0 / torch.clamp(rho, min=1e-12)
        uxp = ux + 0.5 * fx * inv_rho
        uyp = uy + 0.5 * fy * inv_rho
        feq = equilibrium(rho, uxp, uyp)
        src = torch.empty_like(f)
        uf = uxp * fx + uyp * fy
        for i in range(9):
            cx, cy = _C[i]
            cf = cx * fx + cy * fy
            cu = 3.0 * (cx * uxp + cy * uyp)
            src[i] = _W[i] * (1.0 - 0.5 / tau) * ((cf - uf) * 3.0 + cu * cf / CS2)
        return f - (f - feq) / tau + src

    def _collide_velocity_shift(self, f, rho, ux, uy, fx, fy, tau):
        inv_rho = 1.0 / torch.clamp(rho, min=1e-12)
        feq = equilibrium(rho, ux + tau * fx * inv_rho, uy + tau * fy * inv_rho)
        return f - (f - feq) / tau

    # ------------------------------------------------------------------ #
    def step(self) -> None:
        """One time step: interaction force -> collide -> stream."""
        fx_a, fy_a, fx_b, fy_b = self._interaction_force()
        ux, uy, _ = self._common_velocity()
        rho_a, _, _ = macroscopic(self.f_a)
        rho_b, _, _ = macroscopic(self.f_b)

        if self.cfg.forcing == "edm":
            fn = self._collide_edm
        elif self.cfg.forcing == "guo":
            fn = self._collide_guo
        elif self.cfg.forcing == "velocity_shift":
            fn = self._collide_velocity_shift
        else:
            raise ValueError(f"Unknown forcing: {self.cfg.forcing}")

        fa = fn(self.f_a, rho_a, ux, uy, fx_a, fy_a, self.cfg.tau_a)
        fb = fn(self.f_b, rho_b, ux, uy, fx_b, fy_b, self.cfg.tau_b)

        self.f_a = stream_bounce_back_y(fa, self.solid_top, self.solid_bot)
        self.f_b = stream_bounce_back_y(fb, self.solid_top, self.solid_bot)
        self.f_a[:, self.solid] = 0.0
        self.f_b[:, self.solid] = 0.0
        self.step_count += 1

    # ------------------------------------------------------------------ #
    def get_macroscopic(self) -> Dict[str, torch.Tensor]:
        """Macroscopic fields, including the physical (half-force shifted) velocity."""
        rho_a, ux_a_raw, uy_a_raw = macroscopic(self.f_a)
        rho_b, ux_b_raw, uy_b_raw = macroscopic(self.f_b)
        fx_a, fy_a, fx_b, fy_b = self._interaction_force()
        rho_t = rho_a + rho_b
        inv = 1.0 / torch.clamp(rho_t, min=1e-12)
        ux_t, uy_t, _ = self._common_velocity()
        ux_t = ux_t + 0.5 * (fx_a + fx_b) * inv
        uy_t = uy_t + 0.5 * (fy_a + fy_b) * inv
        return {
            "rho_a": rho_a,
            "rho_b": rho_b,
            "rho_total": rho_t,
            "ux_a": ux_a_raw,
            "ux_b": ux_b_raw,
            "ux_total": ux_t,
            "uy_total": uy_t,
        }

    def get_x_averaged_profile(self) -> Dict[str, np.ndarray]:
        """x-averaged profiles on the fluid rows; ``y = j - 0.5`` (wall distance)."""
        m = self.get_macroscopic()
        j = slice(1, self.ny - 1)
        out = {"y": np.arange(1, self.ny - 1, dtype=float) - 0.5}
        for k in ("rho_a", "rho_b", "rho_total", "ux_a", "ux_b", "ux_total"):
            out[k] = m[k][j, :].mean(dim=1).detach().cpu().numpy()
        return out

    # ------------------------------------------------------------------ #
    def run(self, monitor_every: Optional[int] = None, steady_tol: Optional[float] = None):
        """Time-march ``cfg.steps`` steps; records history and steady-state drift."""
        history = []
        prev = None
        max_drift = 0.0
        for step in range(1, self.cfg.steps + 1):
            self.step()
            every = monitor_every or self.cfg.sample_every
            if every and step % every == 0:
                prof = self.get_x_averaged_profile()
                u = prof["ux_total"]
                u_max = float(np.max(np.abs(u)))
                drift = 0.0 if prev is None else float(np.max(np.abs(u - prev)))
                prev = u
                max_drift = max(max_drift, drift)
                history.append(
                    {
                        "step": step,
                        "rho_a_mean": float(prof["rho_a"].mean()),
                        "rho_b_mean": float(prof["rho_b"].mean()),
                        "ux_max": u_max,
                        "profile_drift": drift,
                    }
                )
                if steady_tol is not None and step > every and drift < steady_tol * max(u_max, 1e-30):
                    break
        return {
            "steps": self.step_count,
            "history": history,
            "final_profile": self.get_x_averaged_profile(),
            "max_profile_drift": max_drift,
        }


# --------------------------------------------------------------------------- #
# Reference (analytic) solution helper used by the validation entry point
# --------------------------------------------------------------------------- #
def layered_reference(cfg: TwoComponentSCConfig, n_sample: int = 0):
    """Exact two-layer Poiseuille reference for a two-component SC configuration.

    Component A occupies the bottom ``init_layer_frac`` of the fluid height with
    viscosity ``nu_a = cs^2 (tau_a - 1/2)``; component B the rest with ``nu_b``.
    """
    if abs(cfg.init_layer_frac - 0.5) > 1e-12:
        raise ValueError("layered_reference supports a single mid-plane interface (init_layer_frac=0.5)")
    H = cfg.height
    iface = cfg.init_layer_frac * H
    sol = solve_layered(height=H, interfaces=[iface], nus=[cfg.nu_a, cfg.nu_b], g=cfg.g_x)
    if n_sample:
        y = np.linspace(0.0, H, n_sample + 2)[1:-1]
        return sol, y, sol.u(y)
    return sol