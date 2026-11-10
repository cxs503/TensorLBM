"""Minimal 2D D2Q9 LBM core used by the three-way LBM-DEM-FSI coupling chain.

Single-phase, weakly compressible, BGK collision with a **Guo body-force
field** (the immersed-boundary hand-off), half-way bounce-back solid walls
with optional wall velocity, and non-equilibrium-stress diagnostics that the
interface-closure gates rely on:

* deviatoric stress ``sigma_ij = -((tau-0.5)/tau) * Pi_ij^neq``
  (Chapman-Enskog, exact for BGK at the Navier-Stokes level),
* viscous dissipation rate ``eps = 2 mu S_ij S_ij``,
* kinetic energy, momentum, wall momentum exchange (for moving walls).

Everything is in lattice units (``dx = dt = 1``, ``c_s^2 = 1/3``).  The core
deliberately stays small (one file, no free surface) so that every number used
in the acceptance gates can be traced to one of the diagnostics above.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

from ..d2q9 import C as _C_INT
from ..d2q9 import OPPOSITE as _OPP_INT
from ..d2q9 import W as _W_F32

CS2 = 1.0 / 3.0

C = _C_INT.to(torch.int64)
OPP = _OPP_INT.to(torch.int64)
W = _W_F32.to(torch.float64)


# ---------------------------------------------------------------------------
# index helpers
# ---------------------------------------------------------------------------


def shift(
    a: torch.Tensor, dx: int, dy: int, pad: float = 0.0, periodic: bool = False
) -> torch.Tensor:
    """``out[..., y, x] = a[..., y - dy, x - dx]`` with ``pad`` outside.

    With ``periodic=True`` the shift wraps around instead of padding.
    """
    if periodic:
        return torch.roll(a, shifts=(dy, dx), dims=(-2, -1))
    ny, nx = a.shape[-2], a.shape[-1]
    out = torch.full_like(a, pad)
    ys = slice(max(0, dy), ny + min(0, dy))
    xs = slice(max(0, dx), nx + min(0, dx))
    if ys.stop <= ys.start or xs.stop <= xs.start:
        return out
    out[..., ys, xs] = a[
        ..., ys.start - dy: ys.stop - dy, xs.start - dx: xs.stop - dx
    ]
    return out


def _equilibrium(rho: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    """D2Q9 equilibrium populations, shape ``(9, ny, nx)``."""
    u2 = ux * ux + uy * uy
    cu = C[:, 0].view(9, 1, 1) * ux.unsqueeze(0) + C[:, 1].view(9, 1, 1) * uy.unsqueeze(0)
    w = W.view(9, 1, 1)
    return w * rho.unsqueeze(0) * (
        1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2.unsqueeze(0)
    )


class Flow2D:
    """A compact D2Q9 BGK solver with a force field and bounce-back walls."""

    def __init__(
        self,
        ny: int,
        nx: int,
        tau: float = 0.8,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ) -> None:
        self.ny, self.nx = int(ny), int(nx)
        self.tau = float(tau)
        self.dtype = dtype
        self.device = torch.device(device)
        self.f = torch.zeros((9, self.ny, self.nx), dtype=dtype, device=self.device)
        self.solid = torch.zeros((self.ny, self.nx), dtype=torch.bool, device=self.device)
        #: per-cell wall velocity used by velocity bounce-back, shape (2, ny, nx)
        self.wall_u: Optional[torch.Tensor] = None
        self._w = W.to(dtype=dtype, device=self.device).view(9, 1, 1)
        self._c = C.to(dtype=dtype, device=self.device)

    # -- setup ------------------------------------------------------------
    @property
    def nu(self) -> float:
        return (self.tau - 0.5) / 3.0

    def set_equilibrium(
        self,
        rho=1.0,
        ux=0.0,
        uy=0.0,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        def _t(v):
            if isinstance(v, torch.Tensor):
                return v.to(dtype=self.dtype, device=self.device)
            return torch.full((self.ny, self.nx), float(v), dtype=self.dtype, device=self.device)

        r, u, v = _t(rho), _t(ux), _t(uy)
        feq = _equilibrium(r, u, v)
        if mask is None:
            self.f = feq
        else:
            m = mask.unsqueeze(0).to(device=self.device)
            self.f = torch.where(m, feq, self.f)

    # -- moment diagnostics ----------------------------------------------
    def macroscopic(self, f: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f = self.f if f is None else f
        rho = f.sum(dim=0).clamp(min=1e-12)
        ux = (f * self._c[:, 0].view(9, 1, 1)).sum(dim=0) / rho
        uy = (f * self._c[:, 1].view(9, 1, 1)).sum(dim=0) / rho
        return rho, ux, uy

    def neq_stress(self, f: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Non-equilibrium momentum flux ``Pi_ij^neq``, shape ``(3, ny, nx)``.

        Components are ordered ``(xx, xy, yy)``.
        """
        f = self.f if f is None else f
        rho, ux, uy = self.macroscopic(f)
        feq = _equilibrium(rho, ux, uy)
        fneq = f - feq
        cx = self._c[:, 0].view(9, 1, 1)
        cy = self._c[:, 1].view(9, 1, 1)
        pxx = (fneq * cx * cx).sum(dim=0)
        pxy = (fneq * cx * cy).sum(dim=0)
        pyy = (fneq * cy * cy).sum(dim=0)
        return torch.stack([pxx, pxy, pyy])

    def deviatoric_stress(self, f: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Viscous stress ``sigma_ij = 2 mu S_ij``, ordering ``(xx, xy, yy)``."""
        pi = self.neq_stress(f)
        return -(self.tau - 0.5) / self.tau * pi

    def kinetic_energy(self, f: Optional[torch.Tensor] = None) -> float:
        rho, ux, uy = self.macroscopic(f)
        return float((0.5 * rho * (ux * ux + uy * uy)).sum().item())

    def momentum(self, f: Optional[torch.Tensor] = None) -> torch.Tensor:
        rho, ux, uy = self.macroscopic(f)
        return torch.tensor(
            [float((rho * ux).sum().item()), float((rho * uy).sum().item())],
            dtype=self.dtype,
        )

    def viscous_dissipation(self, f: Optional[torch.Tensor] = None) -> float:
        """``eps = 2 mu S_ij S_ij`` summed over fluid cells (lattice units)."""
        sig = self.deviatoric_stress(f)
        mu = self.nu  # rho = 1 reference
        sxx = sig[0] / (2.0 * mu)
        sxy = sig[1] / (2.0 * mu)
        syy = sig[2] / (2.0 * mu)
        rate = 2.0 * mu * (sxx * sxx + 2.0 * sxy * sxy + syy * syy)
        fluid = (~self.solid).to(self.dtype)
        return float((rate * fluid).sum().item())

    # -- time step --------------------------------------------------------
    def _neighbour_solid_mask(self, periodic: bool = False) -> torch.Tensor:
        """``(9, ny, nx)`` bool: neighbour in direction i is solid / outside."""
        s = self.solid.to(torch.int8).unsqueeze(0)
        out = torch.zeros((9, self.ny, self.nx), dtype=torch.int8, device=self.device)
        for i in range(1, 9):
            out[i] = shift(
                s, int(C[i, 0]), int(C[i, 1]), pad=1, periodic=periodic
            )[0]
        return out > 0

    def step(
        self,
        fx: Optional[torch.Tensor] = None,
        fy: Optional[torch.Tensor] = None,
        periodic: bool = False,
    ) -> dict:
        """Advance one lattice step with an optional Guo body-force field.

        ``fx``/``fy`` are force *densities* in lattice units (per unit volume).
        Returns a dict with the momentum transferred *to the fluid* by wall
        links (``wall_impulse``) and the integrated force field.
        """
        f = self.f
        zero = torch.zeros((self.ny, self.nx), dtype=self.dtype, device=self.device)
        fxv = zero if fx is None else fx
        fyv = zero if fy is None else fy

        rho = f.sum(dim=0).clamp(min=1e-12)
        cx = self._c[:, 0].view(9, 1, 1)
        cy = self._c[:, 1].view(9, 1, 1)
        mom_x = (f * cx).sum(dim=0) + 0.5 * fxv
        mom_y = (f * cy).sum(dim=0) + 0.5 * fyv
        ux = mom_x / rho
        uy = mom_y / rho

        feq = _equilibrium(rho, ux, uy)
        f_post = f - (f - feq) / self.tau

        # Guo forcing source
        fdotu = fxv * ux + fyv * uy
        src = (1.0 - 0.5 / self.tau) * self._w * 3.0 * (
            (cx - ux.unsqueeze(0)) * fxv.unsqueeze(0)
            + (cy - uy.unsqueeze(0)) * fyv.unsqueeze(0)
            + 3.0 * (cx * ux.unsqueeze(0) + cy * uy.unsqueeze(0)) * fdotu.unsqueeze(0)
        )
        f_post = f_post + src

        nb_solid = self._neighbour_solid_mask(periodic=periodic)
        f_new = torch.zeros_like(f)
        f_new[0] = f_post[0]
        wall_mom = torch.zeros(2, dtype=self.dtype, device=self.device)
        for i in range(1, 9):
            ci = (int(C[i, 0]), int(C[i, 1]))
            streamed = shift(f_post[i], ci[0], ci[1], pad=0.0, periodic=periodic)
            opp = int(OPP[i])
            bounce = f_post[opp]
            if self.wall_u is not None:
                # velocity bounce-back: f_opp(x, t+1) built from f_i* + 6 w rho (c_i.u_w)/cs^2
                uw_x = self.wall_u[0]
                uw_y = self.wall_u[1]
                cu = ci[0] * uw_x + ci[1] * uw_y
                bounce = bounce + 6.0 * W[i].item() * rho * cu / (2.0 * CS2)
                wall_mom[0] += float((6.0 * W[i].item() * rho * cu / (2.0 * CS2)).sum().item()) * ci[0]
                wall_mom[1] += float((6.0 * W[i].item() * rho * cu / (2.0 * CS2)).sum().item()) * ci[1]
            f_new[i] = torch.where(nb_solid[i], bounce, streamed)
            # momentum exchange on the solid: F = sum_links c_i (f_i^* + f_opp^{new})

        solid_mask = self.solid.unsqueeze(0)
        f_new = torch.where(solid_mask, torch.zeros_like(f_new), f_new)
        self.f = f_new
        return {
            "wall_impulse": wall_mom,
            "force_integral": torch.tensor(
                [float(fxv.sum().item()), float(fyv.sum().item())], dtype=self.dtype
            ),
        }


def taylor_green_decay(
    ny: int = 32,
    nx: int = 32,
    u0: float = 0.05,
    nu: float = 0.05,
    n_steps: int = 200,
    tau: Optional[float] = None,
) -> dict:
    """Decaying Taylor-Green vortex: a closed energy-ledger sanity check.

    The analytic solution on ``[0, 2pi)^2`` is ``u = u0 cos x sin y exp(-2 nu t)``,
    ``v = -u0 sin x cos y exp(-2 nu t)``; kinetic energy decays as
    ``exp(-4 nu t)``.  Viscous dissipation integrated over time must equal the
    energy drop.  Used by the test-suite to qualify the core before the gates.
    """
    if tau is None:
        tau = 3.0 * nu + 0.5
    flow = Flow2D(ny, nx, tau=tau)
    x = torch.arange(nx, dtype=torch.float64) * (2 * math.pi / nx)
    y = torch.arange(ny, dtype=torch.float64) * (2 * math.pi / ny)
    X, Y = torch.meshgrid(x, y, indexing="xy")
    flow.set_equilibrium(1.0, u0 * torch.cos(X) * torch.sin(Y), -u0 * torch.sin(X) * torch.cos(Y))
    e0 = flow.kinetic_energy()
    diss = 0.0
    for _ in range(n_steps):
        diss += flow.viscous_dissipation()
        flow.step(periodic=True)
    e1 = flow.kinetic_energy()
    t = float(n_steps)
    return {
        "E0": e0,
        "E1": e1,
        "dissipation_integral": diss,
        "ledger_residual": (e0 - e1 - diss) / e0,
        "analytic_ratio": math.exp(-4.0 * flow.nu * t),
        "measured_ratio": e1 / e0,
    }