"""Free-energy (Cahn-Hilliard) D2Q9 channel solver with two-conservation Zou-He ends.

Two-distribution Swift-Osborn-Yeomans model, same discrete kernels as the OSS reference
``ebinan92/Fingering_dynamics`` (``lattice_boltzmann/fingering.py``) as ported in
``xval/xval_fingering_dynamics.py``:

* chemical potential   ``mu = a psi (1 - psi^2) - kappa lap(psi)``,  ``a = -2 kappa / xi^2``,
  ``kappa = 3/4 sigma xi``
* free-energy pressure ``p = rho/3 + psi mu/3``
* ``f`` equilibrium carries ``p`` and ``rho u``; ``g`` equilibrium carries ``psi`` and the
  chemical-potential diffusion term ``3 gamma mu``
* Guo source ``S_i = (1 - 1/(2 tau_f)) w_i [3 c_i.F + 9 (c_i.u)(c_i.F) - 3 u.F]`` on ``f``
* half-way bounce-back on the top/bottom walls (no-slip for ``f``, zero normal flux for ``g``)
* force-consistent two-conservation Zou-He inlet/outlet (see
  :mod:`tensorlbm.phasefield.zouhe_two_conservation``)

The three boundary routes are exposed through ``bc_mode``:

``"corrected"``  force-consistent reconstruction, flux-consistent ``psi`` (this work)
``"impose"``     force-consistent reconstruction, prescribed ``psi`` split by weights
                 (upstream ``ebinan92`` variant)
``"classic"``    classical Zou-He: half-force term dropped (negative control)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .zouhe_two_conservation import (
    column_invariants,
    reconstruct_inlet,
    reconstruct_outlet,
)

_CX = (0, 1, 0, -1, 0, 1, -1, -1, 1)
_CY = (0, 0, 1, 0, -1, 1, 1, -1, -1)
_W = (4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36)
_OPP = (0, 3, 4, 1, 2, 7, 8, 5, 6)


def _t(values, dtype, device, shape=None):
    out = torch.tensor(values, dtype=dtype, device=device).reshape(-1)
    if shape is not None:
        out = out.reshape(shape)
    return out


def poiseuille_profile(ny: int, u_max: float, *, dtype=torch.float64, device=None) -> torch.Tensor:
    """Analytic no-slip channel profile on fluid rows ``y = 1 .. ny-2``.

    Half-way walls sit at ``y = 0.5`` and ``y = ny - 1.5``, so the channel height is
    ``H = ny - 2`` and ``u(y) = u_max (1 - ((y - y_c)/(H/2))^2)``.
    """
    y = torch.arange(1, ny - 1, dtype=dtype, device=device)
    height = float(ny - 2)
    yc = 0.5 * (0.5 + (ny - 1.5))
    return u_max * (1.0 - ((y - yc) / (height / 2.0)) ** 2)


@dataclass
class ChannelConfig:
    ny: int = 34
    nx: int = 96
    sigma: float = 0.01
    xi: float = 2.0
    tau_f: float = 0.8
    tau_g: float = 0.7
    gamma: float = 0.1
    u_scale: float = 0.04
    psi_mode: str = "flux"
    bc_mode: str = "corrected"
    body_force_x: float = 0.0
    dtype: torch.dtype = torch.float64
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    @property
    def kappa(self) -> float:
        return 0.75 * self.sigma * self.xi

    @property
    def a(self) -> float:
        return -2.0 * self.kappa / (self.xi**2)


class FreeEnergyChannel2D:
    """Driven channel with a flat/planar order-parameter interface."""

    def __init__(self, cfg: ChannelConfig):
        self.cfg = cfg
        dt, dev = cfg.dtype, cfg.device
        self.cx = _t(_CX, dt, dev, (9, 1, 1))
        self.cy = _t(_CY, dt, dev, (9, 1, 1))
        self.w = _t(_W, dt, dev, (9, 1, 1))
        self.w0 = _W[0]
        self.opp = _t(_OPP, dt, dev).long()
        self.f = torch.zeros((9, cfg.ny, cfg.nx), dtype=dt, device=dev)
        self.g = torch.zeros((9, cfg.ny, cfg.nx), dtype=dt, device=dev)
        self.psi = torch.zeros((cfg.ny, cfg.nx), dtype=dt, device=dev)
        self.step_count = 0
        self.history: list[dict[str, float]] = []

    # ------------------------------------------------------------------ setup
    def init_flat_interface(self, y0: float | None = None, rho0: float = 1.0) -> None:
        """Equilibrium start: ``psi = tanh((y - y0)/xi)`` with ``u = 0`` (``mu = 0``)."""
        cfg = self.cfg
        if y0 is None:
            y0 = 0.5 * (cfg.ny - 1)
        y = torch.arange(cfg.ny, dtype=cfg.dtype, device=cfg.device).reshape(cfg.ny, 1)
        self.psi = torch.tanh((y - y0) / cfg.xi).expand(cfg.ny, cfg.nx).contiguous()
        rho = torch.full_like(self.psi, rho0)
        gx, gy, lap = self.gradients(self.psi)
        mu = cfg.a * self.psi * (1.0 - self.psi**2) - cfg.kappa * lap
        p = rho / 3.0 + self.psi * mu / 3.0
        zeros = torch.zeros_like(self.psi)
        self.f = self._feq(rho, zeros, zeros, p)
        self.g = self._geq(self.psi, zeros, zeros, mu)
        self.step_count = 0
        self.history.clear()
        _ = (gx, gy)

    def equilibrate_interface(
        self, y0: float | None = None, n_iter: int = 20000, tol: float = 1e-12, omega: float = 20.0
    ) -> dict[str, float]:
        """Relax a 1D tanh into the *discrete* mu = 0 profile of the D2Q9 stencil.

        The continuum tanh has ``mu = 0`` only in the continuum limit; on the lattice the
        D2Q9 Laplacian leaves a residual ``max|mu| ~ 5e-2`` for ``xi = 3`` that drives a slow
        spurious evolution of the interface.  Relaxing ``psi <- psi - omega mu`` gives the
        discrete equilibrium ``mu ~ 0`` and removes that artefact from the flow tests.
        """
        cfg = self.cfg
        if y0 is None:
            y0 = 0.5 * (cfg.ny - 1)
        y = torch.arange(cfg.ny, dtype=cfg.dtype, device=cfg.device).reshape(cfg.ny, 1)
        col = torch.tanh((y - y0) / cfg.xi)
        mu_max = float("inf")
        for _ in range(n_iter):
            padded = torch.cat([col[:1], col, col[-1:]], dim=0)
            acc = torch.zeros_like(col)
            for i in range(1, 9):
                if _CX[i] != 0:
                    continue
                acc = acc + _W[i] * padded[1 + _CY[i] : 1 + _CY[i] + cfg.ny]
            lap = 6.0 * (acc - (1.0 - self.w0) * col)
            mu = cfg.a * col * (1.0 - col**2) - cfg.kappa * lap
            mu_max = float(mu.abs().max())
            if mu_max < tol:
                break
            col = col - omega * mu
        self.psi = col.expand(cfg.ny, cfg.nx).contiguous()
        return {"mu_max": mu_max, "psi_min": float(col.min()), "psi_max": float(col.max())}

    def init_channel_flow(self, u_max: float | None = None, y0: float | None = None) -> None:
        """Analytic-start channel: Poiseuille velocity, interface at ``y0``, linear pressure.

        The density is initialised with the laminar pressure ramp
        ``rho(x) = 1 + 3 (dp/dx) (x - (nx-1))`` with ``dp/dx = -12 nu u_mean / H^2``
        (plane Poiseuille), so the run only has to relax a small acoustic/transient
        residual instead of the full startup.
        """
        cfg = self.cfg
        if u_max is None:
            u_max = cfg.u_scale
        self.init_flat_interface(y0=y0, rho0=1.0)
        prof = poiseuille_profile(cfg.ny, u_max, dtype=cfg.dtype, device=cfg.device)
        u = torch.cat([prof[:1], prof, prof[-1:]]).reshape(cfg.ny, 1).expand(cfg.ny, cfg.nx)
        u_mean = float(prof.mean())
        nu = (cfg.tau_f - 0.5) / 3.0
        height = float(cfg.ny - 2)
        dpdx = -12.0 * nu * u_mean / height**2
        x = torch.arange(cfg.nx, dtype=cfg.dtype, device=cfg.device).reshape(1, cfg.nx)
        rho = 1.0 + 3.0 * dpdx * (x - (cfg.nx - 1))
        rho = rho.expand(cfg.ny, cfg.nx).contiguous()
        _, _, lap = self.gradients(self.psi)
        mu = self.chemical_potential(self.psi, lap)
        p = rho / 3.0 + self.psi * mu / 3.0
        zeros = torch.zeros_like(self.psi)
        self.f = self._feq(rho, u, zeros, p)
        self.g = self._geq(self.psi, u, zeros, mu)
        self.step_count = 0
        self.history.clear()

    def init_uniform(self, psi0: float = 0.0, rho0: float = 1.0, u0: float = 0.0) -> None:
        cfg = self.cfg
        self.psi = torch.full((cfg.ny, cfg.nx), psi0, dtype=cfg.dtype, device=cfg.device)
        rho = torch.full_like(self.psi, rho0)
        mu = torch.full_like(self.psi, cfg.a * psi0 * (1.0 - psi0**2))
        p = rho / 3.0 + self.psi * mu / 3.0
        u = torch.full_like(self.psi, u0)
        zeros = torch.zeros_like(self.psi)
        self.f = self._feq(rho, u, zeros, p)
        self.g = self._geq(self.psi, u, zeros, mu)
        self.step_count = 0
        self.history.clear()

    # ------------------------------------------------------------- operators
    def gradients(self, psi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Isotropic D2Q9 gradient/Laplacian with replicate (neutral) wall ghosts."""
        padded = torch.cat([psi[:1], psi, psi[-1:]], dim=0)
        padded = torch.cat([padded[:, :1], padded, padded[:, -1:]], dim=1)
        gx = torch.zeros_like(psi)
        gy = torch.zeros_like(psi)
        acc = torch.zeros_like(psi)
        ny, nx = psi.shape
        for i in range(1, 9):
            cx, cy = _CX[i], _CY[i]
            shifted = padded[1 + cy : 1 + cy + ny, 1 + cx : 1 + cx + nx]
            wi = _W[i]
            gx = gx + 3.0 * wi * cx * shifted
            gy = gy + 3.0 * wi * cy * shifted
            acc = acc + wi * shifted
        lap = 6.0 * (acc - (1.0 - self.w0) * psi)
        return gx, gy, lap

    def chemical_potential(self, psi: torch.Tensor, lap: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        return cfg.a * psi * (1.0 - psi**2) - cfg.kappa * lap

    def _feq(self, rho, ux, uy, p):
        cu = self.cx * ux.reshape(1, *ux.shape) + self.cy * uy.reshape(1, *uy.shape)
        usq = (ux**2 + uy**2).reshape(1, *ux.shape)
        A = torch.empty((9, *rho.shape), dtype=rho.dtype, device=rho.device)
        A[0] = (rho - 3.0 * (1.0 - self.w0) * p) / self.w0
        A[1:] = 3.0 * p.unsqueeze(0).expand(8, *p.shape)
        return self.w * (A + rho.unsqueeze(0) * (3.0 * cu + 4.5 * cu**2 - 1.5 * usq))

    def _geq(self, psi, ux, uy, mu):
        cfg = self.cfg
        cu = self.cx * ux.reshape(1, *ux.shape) + self.cy * uy.reshape(1, *uy.shape)
        usq = (ux**2 + uy**2).reshape(1, *ux.shape)
        B = torch.empty((9, *psi.shape), dtype=psi.dtype, device=psi.device)
        B[0] = (psi - 3.0 * (1.0 - self.w0) * cfg.gamma * mu) / self.w0
        B[1:] = 3.0 * cfg.gamma * mu.unsqueeze(0).expand(8, *mu.shape)
        return self.w * (B + psi.unsqueeze(0) * (3.0 * cu + 4.5 * cu**2 - 1.5 * usq))

    # ----------------------------------------------------------------- state
    def macroscopic(self):
        rho = self.f.sum(0)
        gx, gy, lap = self.gradients(self.psi)
        mu = self.chemical_potential(self.psi, lap)
        fx = mu * gx + self.cfg.body_force_x
        fy = mu * gy
        jx = (self.f * self.cx).sum(0)
        jy = (self.f * self.cy).sum(0)
        ux = (jx + 0.5 * fx) / rho
        uy = (jy + 0.5 * fy) / rho
        p = rho / 3.0 + self.psi * mu / 3.0
        return rho, ux, uy, mu, (fx, fy), p

    # ------------------------------------------------------------------ step
    def step(self) -> None:
        cfg = self.cfg
        dt, dev = cfg.dtype, cfg.device
        rho, ux, uy, mu, (fx, fy), p = self.macroscopic()

        feq = self._feq(rho, ux, uy, p)
        cu = self.cx * ux.reshape(1, cfg.ny, cfg.nx) + self.cy * uy.reshape(1, cfg.ny, cfg.nx)
        ciF = self.cx * fx.reshape(1, cfg.ny, cfg.nx) + self.cy * fy.reshape(1, cfg.ny, cfg.nx)
        uF = (ux * fx + uy * fy).reshape(1, cfg.ny, cfg.nx)
        src = (1.0 - 0.5 / cfg.tau_f) * self.w * (
            3.0 * ciF + 9.0 * cu * ciF - 3.0 * uF
        )
        f_post = self.f - (self.f - feq) / cfg.tau_f + src

        geq = self._geq(self.psi, ux, uy, mu)
        g_post = self.g - (self.g - geq) / cfg.tau_g

        f_new = torch.empty_like(f_post)
        g_new = torch.empty_like(g_post)
        f_new[0] = f_post[0]
        g_new[0] = g_post[0]
        for i in range(1, 9):
            dx, dy = _CX[i], _CY[i]
            f_new[i] = torch.roll(torch.roll(f_post[i], dx, dims=1), dy, dims=0)
            g_new[i] = torch.roll(torch.roll(g_post[i], dx, dims=1), dy, dims=0)

        # half-way bounce-back walls (solid rows 0 and ny-1, wall at y=0.5 / ny-1.5)
        for i in (4, 7, 8):
            f_new[i, 1, :] = f_post[_OPP[i], 1, :]
            g_new[i, 1, :] = g_post[_OPP[i], 1, :]
        for i in (2, 5, 6):
            f_new[i, cfg.ny - 2, :] = f_post[_OPP[i], cfg.ny - 2, :]
            g_new[i, cfg.ny - 2, :] = g_post[_OPP[i], cfg.ny - 2, :]

        # open boundaries: use the post-stream order parameter to build the force
        psi_post = g_new.sum(0)
        _, _, lap_post = self.gradients(psi_post)
        mu_post = self.chemical_potential(psi_post, lap_post)
        gx_post, gy_post, _ = self.gradients(psi_post)
        f_col_in = (mu_post[:, 0] * gx_post[:, 0] + cfg.body_force_x, mu_post[:, 0] * gy_post[:, 0])
        f_col_out = (
            mu_post[:, -1] * gx_post[:, -1] + cfg.body_force_x,
            mu_post[:, -1] * gy_post[:, -1],
        )

        u_in = self.inlet_profile()
        psi_mode = "impose" if cfg.bc_mode == "impose" else "flux"
        psi_target = psi_post[:, 0] if cfg.bc_mode == "impose" else None
        f_in, g_in, _ = reconstruct_inlet(
            f_new[:, :, 0],
            g_new[:, :, 0],
            u_in,
            f_col_in,
            psi_mode=psi_mode,
            psi_target=psi_target,
            force_correction=cfg.bc_mode != "classic",
        )
        f_new[:, :, 0] = f_in
        g_new[:, :, 0] = g_in

        f_out, g_out, _ = reconstruct_outlet(
            f_new[:, :, -1],
            g_new[:, :, -1],
            1.0,
            f_col_out,
            psi_out=psi_post[:, -2] if cfg.bc_mode == "impose" else None,
            force_correction=cfg.bc_mode != "classic",
        )
        f_new[:, :, -1] = f_out
        g_new[:, :, -1] = g_out

        self.f, self.g = f_new, g_new
        self.psi = g_new.sum(0)
        self.step_count += 1
        _ = dev

    # ----------------------------------------------------------- diagnostics
    def inlet_profile(self) -> torch.Tensor:
        """Column profile over **all** rows (solid rows carry the edge value)."""
        prof = poiseuille_profile(
            self.cfg.ny, self.cfg.u_scale, dtype=self.cfg.dtype, device=self.cfg.device
        )
        if self._uniform_inflow:
            prof = torch.full_like(prof, self.cfg.u_scale)
        return torch.cat([prof[:1], prof, prof[-1:]])

    _uniform_inflow: bool = False

    def column_state(self, which: str):
        """Post-BC invariants of the inlet/outlet column."""
        col = 0 if which == "inlet" else self.cfg.nx - 1
        _, gx, gy, lap = (None, *self.gradients(self.psi))
        mu = self.chemical_potential(self.psi[:, col], lap[:, col])
        force = (mu * gx[:, col] + self.cfg.body_force_x, mu * gy[:, col])
        return column_invariants(
            self.f[:, :, col],
            self.g[:, :, col],
            force,
            force_correction=self.cfg.bc_mode != "classic",
        )

    def budgets(self) -> dict[str, float]:
        cin = self.column_state("inlet")
        cout = self.column_state("outlet")
        fluid = slice(1, self.cfg.ny - 1)
        rho, ux, uy, mu, (fx, fy), p = self.macroscopic()
        # physically meaningful flux integrals over the fluid rows
        q_in = float((cin.mass_flux[fluid]).sum())
        q_out = float((cout.mass_flux[fluid]).sum())
        qpsi_in = float(cin.psi_flux[fluid].sum())
        qpsi_out = float(cout.psi_flux[fluid].sum())
        return {
            "mass_flux_in": q_in,
            "mass_flux_out": q_out,
            "mass_flux_residual": (q_out - q_in) / max(abs(q_in), abs(q_out), 1e-30),
            "psi_flux_in": qpsi_in,
            "psi_flux_out": qpsi_out,
            "psi_flux_residual": (qpsi_out - qpsi_in)
            / max(abs(qpsi_in), abs(qpsi_out), 1e-30),
            "rho_total": float(self.f[:, fluid, :].sum()),
            "psi_total": float(self.g[:, fluid, :].sum()),
            "max_abs_uy": float(uy[fluid, :].abs().max()),
            "max_abs_ux": float(ux[fluid, :].abs().max()),
            "psi_min": float(self.psi[fluid, :].min()),
            "psi_max": float(self.psi[fluid, :].max()),
        }

    def run(self, n_steps: int, *, record_every: int = 0) -> None:
        for _ in range(n_steps):
            self.step()
            if record_every and self.step_count % record_every == 0:
                self.history.append({"step": self.step_count, **self.budgets()})


__all__ = ["ChannelConfig", "FreeEnergyChannel2D", "poiseuille_profile"]