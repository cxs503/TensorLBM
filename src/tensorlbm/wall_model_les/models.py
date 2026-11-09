"""Equilibrium and non-equilibrium wall models for wall-modelled LES (WMLES).

Both models answer the same question in the same units: given the *resolved*
tangential velocity ``u_e`` sampled at a wall-normal distance ``y_e`` (the
"exchange location"), what is the wall shear stress ``tau_w`` and what
tangential velocity should the wall-adjacent node carry?

Equilibrium model
-----------------
Pure law-of-the-wall inversion ``u_e = u_tau f(y_e u_tau / nu)`` with the
unified Spalding law by default (valid from the sublayer into the log region)
or the pure log law when ``law="log"``.  This is the model used by most
engineering WMLES and it is the reference against which the non-equilibrium
model is measured.

Non-equilibrium model
---------------------
The thin-boundary-layer (ODE) wall model of Wang & Moin (2002) / Duprat et al.
(2011).  Inside the modelled layer the wall-normal integrated momentum
equation is

    d tau / dy = rho * S,      tau(y) = rho u_tau^2 + rho S y,
    tau(y)     = rho (nu + nu_t) du/dy,
    nu_t(y)    = [kappa y (1 - exp(-y+ / A+))]^2 |du/dy|,

with ``S = (1/rho) dp/dx + dU/dt + U dU/dx`` the momentum source carried by the
resolved flow (pressure gradient plus streamwise advection plus unsteady
term).  The nonlinear algebraic relation for ``du/dy`` is solved in closed
form,

    du/dy = 2 tau / ( nu + sqrt(nu^2 + 4 l^2 tau) ),

and the profile is integrated from the wall to ``y_e``; ``u_tau`` is then the
root of ``u(y_e; u_tau) - u_e = 0`` (monotone, solved by bisection).

``S = 0`` recovers the equilibrium mixing-length profile, which is by
construction within a few percent of the log law -- this is asserted in the
unit tests and it is the physical consistency gate of the implementation.
For ``S < 0`` (favourable gradient / accelerating boundary layer) ``tau`` grows
away from the wall and ``u_tau`` drops for the same ``u_e``; for ``S > 0``
(adverse gradient) the opposite happens.  This is the physical mechanism that
a pure log-law model cannot represent and it is what makes the model
"non-equilibrium".
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .laws import (
    B_DEFAULT,
    KAPPA_DEFAULT,
    VAN_DRIEST_A_PLUS,
    law_u_plus,
    law_u_plus_scalar,
    solve_u_tau,
    solve_u_tau_scalar,
)

__all__ = [
    "WallModelResult",
    "EquilibriumWallModel",
    "NonEquilibriumWallModel",
    "source_term_from_resolved",
]


@dataclass(frozen=True)
class WallModelResult:
    """Output of a wall model at one sampling station."""

    u_tau: torch.Tensor
    tau_w: torch.Tensor
    u_wall_node: torch.Tensor
    y_plus_exchange: torch.Tensor
    model: str

    def as_dict(self) -> dict:
        return {
            "u_tau": self.u_tau.detach().cpu(),
            "tau_w": self.tau_w.detach().cpu(),
            "u_wall_node": self.u_wall_node.detach().cpu(),
            "y_plus_exchange": self.y_plus_exchange.detach().cpu(),
            "model": self.model,
        }


@dataclass(frozen=True)
class EquilibriumWallModel:
    """Law-of-the-wall equilibrium wall model."""

    kappa: float = KAPPA_DEFAULT
    b: float = B_DEFAULT
    law: str = "spalding"
    rho: float = 1.0

    def skin_friction(
        self,
        u_exchange,
        y_exchange: float,
        nu: float,
        *,
        source: float = 0.0,
        y_wall_node: float | None = None,
        guess: float | None = None,
    ) -> WallModelResult:
        """Return ``u_tau``, ``tau_w`` and the velocity to impose at ``y_wall_node``.

        ``source`` is accepted for interface parity with
        :class:`NonEquilibriumWallModel` and is ignored: by definition the
        equilibrium model assumes local stress equilibrium, so the momentum
        source cannot enter.  Ignoring it explicitly (instead of silently
        dropping the argument) is what makes the equilibrium/non-equilibrium
        comparison in the channel a controlled one.
        """
        del source
        y_eval = y_exchange if y_wall_node is None else y_wall_node
        if isinstance(u_exchange, (int, float)):
            u_tau_s = solve_u_tau_scalar(
                float(u_exchange),
                y_exchange,
                nu,
                law=self.law,
                kappa=self.kappa,
                b=self.b,
                guess=guess,
            )
            y_plus_s = y_eval * u_tau_s / nu
            u_node_s = u_tau_s * law_u_plus_scalar(y_plus_s, law=self.law, kappa=self.kappa, b=self.b)
            return WallModelResult(
                u_tau=torch.tensor([u_tau_s], dtype=torch.float64),
                tau_w=torch.tensor([self.rho * u_tau_s**2], dtype=torch.float64),
                u_wall_node=torch.tensor([u_node_s], dtype=torch.float64),
                y_plus_exchange=torch.tensor([y_exchange * u_tau_s / nu], dtype=torch.float64),
                model=f"equilibrium:{self.law}",
            )
        u_exchange = torch.as_tensor(u_exchange, dtype=torch.float64)
        u_tau = solve_u_tau(
            u_exchange, y_exchange, nu, law=self.law, kappa=self.kappa, b=self.b
        )
        y_plus = y_eval * u_tau / nu
        u_node = u_tau * law_u_plus(y_plus, law=self.law, kappa=self.kappa, b=self.b)
        return WallModelResult(
            u_tau=u_tau,
            tau_w=self.rho * u_tau.square(),
            u_wall_node=u_node,
            y_plus_exchange=y_exchange * u_tau / nu,
            model=f"equilibrium:{self.law}",
        )


@dataclass(frozen=True)
class NonEquilibriumWallModel:
    """Thin-boundary-layer ODE wall model (pressure gradient + advection aware)."""

    kappa: float = KAPPA_DEFAULT
    b: float = B_DEFAULT
    van_driest_a_plus: float = VAN_DRIEST_A_PLUS
    n_layers: int = 64
    rho: float = 1.0

    # ------------------------------------------------------------------
    def layer_faces(self, y_exchange: float, dtype=torch.float64) -> torch.Tensor:
        """Layer faces with quadratic clustering towards the wall."""
        n = int(self.n_layers)
        s = torch.linspace(0.0, 1.0, n + 1, dtype=dtype)
        return y_exchange * s.square()

    def profile(
        self,
        u_tau,
        y_exchange: float,
        nu: float,
        source: float = 0.0,
        *,
        y_eval: float | None = None,
    ):
        """Integrate the modelled layer.

        Returns ``u`` at ``y_eval`` when given (scalar), otherwise the profile
        array ``(batch, n_layers + 1)`` on the layer faces including ``y = 0``.
        """
        u_tau = torch.as_tensor(u_tau, dtype=torch.float64).reshape(-1, 1)
        faces = self.layer_faces(y_exchange)  # (n+1,)
        dy = torch.diff(faces)
        y_mid = 0.5 * (faces[:-1] + faces[1:])
        tau = (u_tau.square() + source * y_mid).clamp_min(1.0e-30)
        ell = self.kappa * y_mid * (
            1.0 - torch.exp(-(y_mid * u_tau) / (nu * self.van_driest_a_plus))
        )
        ell = ell.clamp_min(1.0e-12)
        # Rationalised closed-form solution of  tau = (nu + ell^2 du/dy) du/dy
        du_dy = 2.0 * tau / (nu + torch.sqrt(nu * nu + 4.0 * ell.square() * tau))
        u = torch.cumsum(du_dy * dy, dim=1)
        u = torch.cat([torch.zeros_like(u[:, :1]), u], dim=1)
        if y_eval is None:
            return u
        y_eval = min(max(float(y_eval), 0.0), float(y_exchange))
        # Exact linear interpolation inside the containing layer.
        idx = int(torch.searchsorted(faces, torch.tensor(y_eval, dtype=torch.float64)))
        idx = max(1, min(idx, faces.numel() - 1))
        y0 = float(faces[idx - 1])
        y1 = float(faces[idx])
        w = (y_eval - y0) / (y1 - y0)
        u_eval = (1.0 - w) * u[:, idx - 1] + w * u[:, idx]
        if y_eval == 0.0:
            u_eval = torch.zeros_like(u[:, 0])
        return u_eval

    def skin_friction(
        self,
        u_exchange,
        y_exchange: float,
        nu: float,
        *,
        source: float = 0.0,
        y_wall_node: float | None = None,
        iterations: int = 50,
    ) -> WallModelResult:
        """Solve for ``u_tau`` such that the modelled profile matches ``u_exchange``."""
        u_exchange = torch.as_tensor(u_exchange, dtype=torch.float64).abs().reshape(-1)
        if u_exchange.numel() == 0:
            raise ValueError("empty exchange velocity")
        base = torch.sqrt(torch.clamp(u_exchange * nu / y_exchange, min=1.0e-24))
        low = 0.25 * base
        high = 8.0 * base + 1.0e-10

        def residual(u_tau: torch.Tensor) -> torch.Tensor:
            return self.profile(u_tau, y_exchange, nu, source, y_eval=y_exchange).reshape(
                -1
            ) - u_exchange

        for _ in range(40):
            need = residual(high) < 0.0
            if not bool(need.any()):
                break
            high = torch.where(need, high * 4.0, high)
        lo = low
        hi = high
        for _ in range(iterations):
            mid = 0.5 * (lo + hi)
            below = residual(mid) < 0.0
            lo = torch.where(below, mid, lo)
            hi = torch.where(below, hi, mid)
        u_tau = 0.5 * (lo + hi)
        y_eval = y_exchange if y_wall_node is None else y_wall_node
        u_node = self.profile(u_tau, y_exchange, nu, source, y_eval=y_eval).reshape(-1)
        return WallModelResult(
            u_tau=u_tau,
            tau_w=self.rho * u_tau.square(),
            u_wall_node=u_node,
            y_plus_exchange=y_exchange * u_tau / nu,
            model="nonequilibrium:ode",
        )


def source_term_from_resolved(
    *,
    dp_dx: float = 0.0,
    rho: float = 1.0,
    dU_dt: float = 0.0,
    U: float = 0.0,
    dU_dx: float = 0.0,
) -> float:
    """Assemble the non-equilibrium momentum source ``S`` (per unit mass)."""
    return dp_dx / rho + dU_dt + U * dU_dx