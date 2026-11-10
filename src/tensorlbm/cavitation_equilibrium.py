"""Equilibrium (constant-pressure) cavitating-airfoil solver — P40W4-N03.

Why a second cavitation solver
------------------------------
:mod:`tensorlbm.cavitation_airfoil` integrates the *kinetic* Schnerr-Sauer
mass-transfer source explicitly on a homogeneous mixture.  On the 2-D
hydrofoil benchmarks that formulation cannot start a cavity at all: the
measured pressure pocket where ``p < p_v`` is 1-3 lattice cells wide at
``cells_per_chord <= 20`` (see ``runs/p40w4n03/instr_*.json``), the volume
fraction there is advected away faster than the Rayleigh-Plesset growth rate
builds it up, and the pressure field therefore never acquires the feedback that
holds a sheet cavity at ``p_v``.  The measured cavity length then collapses onto
the *single-phase* isobar, which is a factor 3-5 shorter than the experimental
value (that is the P41-N02 honest-FAIL, `docs/P41-N02_cavitation_airfoil_validation.md`).

This module implements the stiff (equilibrium) limit of the same physics, which
is what the sheet/cloud-cavity benchmarks actually require:

``alpha``       -- vapour volume fraction, transported and relaxed towards the
                   thermodynamic equilibrium value implied by the *local*
                   pressure:
                   ``alpha -> 1`` where ``p < p_v``, ``0`` where ``p > p_v``;
``p``           -- liquid pressure, solved with the vapour region **pinned at
                   ``p_v``** (Dirichlet), i.e. the constant-pressure-cavity /
                   Riabouchinsky closure that classical sheet-cavity work uses
                   (Brennen 1995, ch. 7: the cavity interior is an isobar);
liquid          -- incompressible, ``rho_v/rho_l -> 0`` so the phase-change
                   volume source is dropped and the liquid stays discretely
                   divergence free.

The cavity therefore becomes a genuine free boundary: its length is not read off
a prescribed isobar of the single-phase field but is the fixed point at which the
``p = p_v`` surface closes.  The mixture pressure feedback that was missing in
the kinetic formulation is now enforced exactly.

Numerics
--------
* staggered MAC grid (velocity on faces, ``p``/``alpha`` at cell centres);
* 2nd-order upwind normal convection + central cross terms, forward Euler;
* Smagorinsky SGS viscosity;
* pressure: ``lap(phi) = div(u*)/dt`` with Dirichlet ``phi = p_v`` on vapour
  cells and ``phi = 0`` on the outlet column (pressure outlet, which fixes the
  physical gauge and hence ``p_inf = 0``), zero normal gradient elsewhere;
  solved by conjugate gradients preconditioned with the constant-coefficient
  FFT Neumann solve (:func:`mixed_poisson`), warm-started from the previous step;
* airfoil: Brinkman penalisation (same ``eta`` convention as the kinetic solver).

Everything is non-dimensional (length/chord, velocity/``U``,
pressure/``rho_l U^2``), so ``St = f`` directly.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch

from .cavitation_airfoil import (
    AirfoilGeometry,
    CavitationFlowConfig,
    _divergence,
    _grad_cell_to_ufaces,
    _grad_cell_to_vfaces,
    _pad_edge,
    _shift,
    build_airfoil_mask,
    cavity_metrics,
    dominant_frequency,
    solve_poisson_neumann,
)

__all__ = [
    "EquilibriumCavityConfig",
    "EquilibriumCavitySolver",
    "equilibrium_alpha_target",
    "mixed_poisson",
]


def equilibrium_alpha_target(
    p: torch.Tensor, p_v: float, switch_width: float = 0.02
) -> torch.Tensor:
    """Stiff-limit vapour fraction implied by the local pressure.

    ``alpha -> 1`` for ``p << p_v``, ``alpha -> 0`` for ``p >> p_v``, with a
    linear transition of width ``switch_width`` centred on ``p_v`` (the width is
    a numerical regularisation of the infinite equilibrium slope, not a physical
    parameter; the report carries its sensitivity).
    """
    half = max(0.5 * float(switch_width), 1e-12)
    return ((p_v - p) / (2.0 * half) + 0.5).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Mixed Dirichlet / Neumann pressure solve
# ---------------------------------------------------------------------------


def _lap_neumann(phi: torch.Tensor, h: float) -> torch.Tensor:
    """5-point Laplacian with zero normal gradient on all four domain faces."""
    return (
        _shift(phi, -1, 0) - 2.0 * phi + _shift(phi, 1, 0)
        + _shift(phi, -1, 1) - 2.0 * phi + _shift(phi, 1, 1)
    ) / (h * h)


def mixed_poisson(
    rhs: torch.Tensor,
    fixed_mask: torch.Tensor,
    fixed_value: float | torch.Tensor,
    h: float,
    x0: torch.Tensor | None = None,
    tol: float = 1.0e-9,
    max_iter: int = 60,
) -> tuple[torch.Tensor, dict]:
    """``lap(phi) = rhs`` on free cells, ``phi = fixed_value`` on ``fixed_mask``.

    Zero normal gradient on the domain faces.  Solved with conjugate gradients
    preconditioned by the constant-coefficient FFT Neumann inverse (the same
    operator with the Dirichlet rows removed, which is a good approximation as
    long as the pinned region is a modest fraction of the domain), warm-started
    from ``x0``.
    """
    if not bool(fixed_mask.any()):
        phi = solve_poisson_neumann(rhs, h)
        return phi, {"iters": 0, "residual": 0.0, "mode": "neumann"}
    free = ~fixed_mask
    value = torch.as_tensor(fixed_value, dtype=rhs.dtype, device=rhs.device)
    b = torch.where(free, rhs, value.expand_as(rhs))
    x = torch.where(free, x0 if x0 is not None else torch.zeros_like(rhs), value.expand_as(rhs))

    def apply(v: torch.Tensor) -> torch.Tensor:
        out = _lap_neumann(v, h)
        return torch.where(free, out, v)

    r = torch.where(free, b - apply(x), torch.zeros_like(b))
    scale = float(b.abs().max()) + 1e-30
    z = _precondition(r, free, h)
    d = z.clone()
    rz = float((r * z).sum())
    it = 0
    res = float(r.abs().max()) / scale
    while res > tol and it < max_iter:
        ad = apply(d)
        denom = float((d * ad).sum())
        if abs(denom) < 1e-300:
            break
        a = rz / denom
        x = x + a * d
        x = torch.where(free, x, value.expand_as(rhs))
        r = r - a * ad
        r = torch.where(free, r, torch.zeros_like(b))
        res = float(r.abs().max()) / scale
        z = _precondition(r, free, h)
        rz_new = float((r * z).sum())
        if abs(rz) < 1e-300:
            break
        d = z + (rz_new / rz) * d
        rz = rz_new
        it += 1
    return x, {"iters": it, "residual": res, "mode": "mixed"}


def _precondition(r: torch.Tensor, free: torch.Tensor, h: float) -> torch.Tensor:
    """FFT Neumann inverse applied to the free-cell residual (zero elsewhere)."""
    z = solve_poisson_neumann(torch.where(free, r, torch.zeros_like(r)), h)
    return torch.where(free, z, torch.zeros_like(z))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class EquilibriumCavityConfig:
    """One 2-D cavitating-airfoil run (non-dimensional, ``c = U = 1``)."""

    airfoil: str = "NACA0015"
    alpha_deg: float = 5.0
    sigma: float = 1.0
    re: float = 1.0e5
    cells_per_chord: int = 50
    domain_x_chords: float = 8.0
    domain_y_chords: float = 4.0
    le_x_chords: float = 2.0
    le_y_chords: float = 2.0
    n_steps: int = 4000
    cfl: float = 0.25
    smagorinsky: float = 0.1
    viscosity_floor: float = 1.0e-4
    cavity_threshold: float = 0.5
    #: half-width of the equilibrium switch in pressure units; the vapour
    #: fraction goes 0 -> 1 across ``p_v +- switch_width/2``.
    switch_width: float = 0.02
    #: relaxation time of ``alpha`` towards equilibrium, in steps (``tau =
    #: relax_steps * dt``).  Must be small enough that the free boundary follows
    #: the pressure field; the sensitivity is measured in the report.
    relax_steps: float = 4.0
    sample_interval: int = 2
    transient_fraction: float = 0.5
    spinup_steps: int = 200
    outlet_pressure_cells: int = 0
    """Pressure-outlet Dirichlet columns.  ``0`` (default) keeps the pure
    Neumann outlet of the kinetic solver, which is stable on this grid; the
    physical gauge then comes from the vapour Dirichlet region (``p = p_v``)
    once a cavity exists, and from the inlet anchor before that."""
    seed_length: float = 0.0
    """Initial vapour patch over ``x/c in [0.02, 0.02+seed_length]`` on the
    suction surface (0 = cold start from liquid)."""
    seed_alpha: float = 0.9
    max_cg_iter: int = 60
    seed: int = 0

    def __post_init__(self) -> None:
        if self.cells_per_chord < 8:
            msg = "cells_per_chord must be >= 8"
            raise ValueError(msg)

    @property
    def h(self) -> float:
        return 1.0 / self.cells_per_chord

    @property
    def nx(self) -> int:
        return int(round(self.domain_x_chords * self.cells_per_chord))

    @property
    def ny(self) -> int:
        return int(round(self.domain_y_chords * self.cells_per_chord))

    @property
    def p_v(self) -> float:
        """Vapour pressure for ``p_inf = 0``: ``sigma = (p_inf - p_v)/q``."""
        return -0.5 * self.sigma

    @property
    def nu(self) -> float:
        return 1.0 / self.re


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


class EquilibriumCavitySolver:
    """Constant-pressure (equilibrium) cavitating flow past a 2-D airfoil."""

    def __init__(self, cfg: EquilibriumCavityConfig, device: str = "cpu",
                 eta: float = 1.0e-3) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.eta = eta
        c = cfg
        dtype = torch.float64
        self.h = c.h
        self.dt = c.cfl * c.h
        nx, ny = c.nx, c.ny
        solid, self.geometry = build_airfoil_mask(
            c.airfoil, nx, ny, c.cells_per_chord, c.alpha_deg,
            c.le_x_chords * c.cells_per_chord, c.le_y_chords * c.cells_per_chord,
        )
        self.solid = solid.to(self.device)
        self.u = torch.ones(nx + 1, ny, dtype=dtype, device=self.device)
        self.v = torch.zeros(nx, ny + 1, dtype=dtype, device=self.device)
        self.p = torch.zeros(nx, ny, dtype=dtype, device=self.device)
        self.alpha = torch.zeros(nx, ny, dtype=dtype, device=self.device)
        self.pen = torch.where(
            self.solid,
            torch.full((nx, ny), c.cfl * c.h / eta, dtype=dtype, device=self.device),
            torch.zeros((nx, ny), dtype=dtype, device=self.device),
        )
        pu = _pad_edge(self.pen, 1, 1, 0)
        pv = _pad_edge(self.pen, 1, 1, 1)
        self.damp_u = 1.0 + 0.5 * (pu[:-1, :] + pu[1:, :])
        self.damp_v = 1.0 + 0.5 * (pv[:, :-1] + pv[:, 1:])
        # pressure-outlet column (Dirichlet p = 0), not inside the body
        self.outlet = torch.zeros(nx, ny, dtype=torch.bool, device=self.device)
        self.outlet[nx - c.outlet_pressure_cells:, :] = True
        self._step = 0
        self.time_history: list[float] = []
        self.cl_history: list[float] = []
        self.cd_history: list[float] = []
        self.lcav_history: list[float] = []
        self.mask_history: list[int] = []
        self.div_residual_max = 0.0
        self.mass_balance_max = 0.0
        self.cg_iters: list[int] = []
        self.cg_residual_max = 0.0
        self.p_inlet_history: list[float] = []
        if c.seed_length > 0.0:
            self._seed_cavity(c.seed_length, c.seed_alpha)

    # -- geometry helpers -------------------------------------------------
    def _seed_cavity(self, length: float, alpha: float) -> None:
        nx = self.alpha.shape[0]
        h = self.h
        x_chord = (torch.arange(nx, dtype=torch.float64) + 0.5) * h - self.cfg.le_x_chords
        band = (x_chord > 0.02) & (x_chord < 0.02 + length)
        for i in torch.nonzero(band, as_tuple=False).flatten().tolist():
            col = torch.nonzero(self.solid[i], as_tuple=False).flatten()
            if col.numel() == 0:
                continue
            j0 = int(col.max()) + 1
            self.alpha[i, j0:j0 + 3] = alpha

    def _inlet(self) -> None:
        self.u[0, :] = 1.0

    # -- momentum ---------------------------------------------------------
    def _advection_u(self) -> torch.Tensor:
        h = self.h
        u = self.u
        ufr = 0.5 * (u + _shift(u, -1, 0))
        up = 1.5 * u - 0.5 * _shift(u, 1, 0)
        un = 1.5 * _shift(u, -1, 0) - 0.5 * _shift(u, -2, 0)
        flux = torch.where(ufr > 0, ufr * up, ufr * un)
        cx = (flux - _shift(flux, 1, 0)) / h
        vpad = _pad_edge(self.v, 1, 1, 0)  # (nx+2, ny+1)
        v_here = 0.25 * (
            vpad[:-1, :-1] + vpad[:-1, 1:] + vpad[1:, :-1] + vpad[1:, 1:]
        )
        # 2nd-order upwind transverse derivative (damps the odd-even mode that a
        # central cross term leaves undamped at Re = 1e5)
        du_dy = torch.where(
            v_here > 0,
            (1.5 * u - 2.0 * _shift(u, 1, 1) + 0.5 * _shift(u, 2, 1)) / h,
            (-1.5 * u + 2.0 * _shift(u, -1, 1) - 0.5 * _shift(u, -2, 1)) / h,
        )
        cy = v_here * du_dy
        return -(cx + cy)


    def _advection_v(self) -> torch.Tensor:
        h = self.h
        v = self.v
        vfr = 0.5 * (v + _shift(v, -1, 1))
        up = 1.5 * v - 0.5 * _shift(v, 1, 1)
        un = 1.5 * _shift(v, -1, 1) - 0.5 * _shift(v, -2, 1)
        flux = torch.where(vfr > 0, vfr * up, vfr * un)
        cy = (flux - _shift(flux, 1, 1)) / h
        upad = _pad_edge(self.u, 1, 1, 1)  # (nx+1, ny+2)
        u_here = 0.25 * (
            upad[:-1, :-1] + upad[1:, :-1] + upad[:-1, 1:] + upad[1:, 1:]
        )
        dv_dx = torch.where(
            u_here > 0,
            (1.5 * v - 2.0 * _shift(v, 1, 0) + 0.5 * _shift(v, 2, 0)) / h,
            (-1.5 * v + 2.0 * _shift(v, -1, 0) - 0.5 * _shift(v, -2, 0)) / h,
        )
        cx = u_here * dv_dx
        return -(cx + cy)


    def _strain_magnitude(self) -> torch.Tensor:
        h = self.h
        ux = (self.u[1:, :] - self.u[:-1, :]) / h
        vy = (self.v[:, 1:] - self.v[:, :-1]) / h
        uc = 0.5 * (self.u[:-1, :] + self.u[1:, :])
        vc = 0.5 * (self.v[:, :-1] + self.v[:, 1:])
        uy = (_shift(uc, -1, 1) - _shift(uc, 1, 1)) / (2 * h)
        vx = (_shift(vc, -1, 0) - _shift(vc, 1, 0)) / (2 * h)
        sxy = 0.5 * (uy + vx)
        return torch.sqrt(2.0 * (ux**2 + vy**2 + 2.0 * sxy**2) + 1e-30)


    def _viscous(self, f: torch.Tensor, nu_field: torch.Tensor) -> torch.Tensor:
        h = self.h
        lap = (_shift(f, -1, 0) - 2 * f + _shift(f, 1, 0)) / h**2
        lap = lap + (_shift(f, -1, 1) - 2 * f + _shift(f, 1, 1)) / h**2
        return nu_field * lap


    def _nu_at_u(self, nu_c: torch.Tensor) -> torch.Tensor:
        """Cell-centred viscosity averaged onto the ``nx+1`` x-faces."""
        pad = _pad_edge(nu_c, 1, 1, 0)
        return 0.5 * (pad[:-1, :] + pad[1:, :])


    def _nu_at_v(self, nu_c: torch.Tensor) -> torch.Tensor:
        """Cell-centred viscosity averaged onto the ``ny+1`` y-faces."""
        pad = _pad_edge(nu_c, 1, 1, 1)
        return 0.5 * (pad[:, :-1] + pad[:, 1:])

    # -- diagnostics ------------------------------------------------------
    def forces(self) -> tuple[float, float]:
        """Penalisation reaction (drag, lift) coefficients, 2-D per unit depth."""
        inv_eta = self.pen / self.dt  # 1/eta (= 0 in the fluid)
        cxp = inv_eta * self.u[:-1, :]  # (nx, ny) u at cell centres
        cyp = inv_eta * 0.5 * (self.v[:, :-1] + self.v[:, 1:])
        scale = 2.0 * self.h**2
        return float(cxp.sum() * scale), float(cyp.sum() * scale)

    # -- forces -----------------------------------------------------------
    def forces(self) -> tuple[float, float]:
        h = self.h
        u_body = 0.5 * (self.u[:-1, :] + self.u[1:, :])
        v_body = 0.5 * (self.v[:, :-1] + self.v[:, 1:])
        fx = float((u_body * self.solid).sum()) * h * h / self.eta
        fy = float((v_body * self.solid).sum()) * h * h / self.eta
        return fx, fy

    def cl(self) -> float:
        """Lift coefficient ``2 F_y`` (forces non-dimensionalised by ``q c``)."""
        return 2.0 * self.forces()[1]

    def cd(self) -> float:
        return 2.0 * self.forces()[0]

    # -- one step ---------------------------------------------------------
    def step(self) -> None:
        c = self.cfg
        h = self.h
        dt = self.dt
        self._inlet()

        strain = self._strain_magnitude()
        nu_t = (c.smagorinsky * h) ** 2 * strain
        nu_c = c.nu + nu_t + c.viscosity_floor
        r_u = self._advection_u() + self._viscous(self.u, self._nu_at_u(nu_c))
        r_v = self._advection_v() + self._viscous(self.v, self._nu_at_v(nu_c))
        u_star = (self.u + dt * r_u) / self.damp_u
        v_star = (self.v + dt * r_v) / self.damp_v
        self._inlet()

        # global flux balance before the projection (incompressible liquid: the
        # outflow absorbs exactly the inflow, so the Neumann system stays
        # compatible -- this must be applied *before* the divergence is taken)
        q_in = float(self.u[0, :].mean())
        if c.outlet_pressure_cells <= 0:
            u_star[-1, :] = u_star[-1, :] + (q_in - float(u_star[-1, :].mean()))
        div_star = _divergence(u_star, v_star, h)
        rhs = div_star / dt
        vap = self.alpha > c.cavity_threshold
        fixed = vap | self.outlet
        phi, info = mixed_poisson(
            rhs, fixed, c.p_v, h, x0=self.p, max_iter=c.max_cg_iter
        )
        if bool(self.outlet.any()):
            phi = torch.where(vap, torch.full_like(phi, c.p_v), phi)
            phi = torch.where(self.outlet & ~vap, torch.zeros_like(phi), phi)
        self.p_offset = 0.0
        if not bool(vap.any()):
            # no cavity yet: the Neumann projection leaves the level free, so
            # anchor it on the inflow plane and use that level for the switch
            self.p_offset = -float(phi[:3, :].mean())
        p_phys = phi + self.p_offset
        self.cg_iters.append(info["iters"])
        self.cg_residual_max = max(self.cg_residual_max, info["residual"])
        self.p = phi
        self.u = u_star - _grad_cell_to_ufaces(phi, h) * dt
        self.v = v_star - _grad_cell_to_vfaces(phi, h) * dt

        # equilibrium vapour fraction: relax alpha towards the pressure switch
        p_c = p_phys
        target = equilibrium_alpha_target(p_c, c.p_v, c.switch_width)
        if self._step < c.spinup_steps:
            target = self.alpha  # hold the cold start during the spin-up
        beta = min(0.5, dt / max(c.relax_steps * dt, 1e-30))
        # bounded upwind transport of the vapour fraction (divergence free
        # liquid: a first-order upwind update is monotone for |u| dt/h < 1)
        u_c = 0.5 * (self.u[:-1, :] + self.u[1:, :])
        v_c = 0.5 * (self.v[:, :-1] + self.v[:, 1:])
        a = self.alpha
        fx = torch.where(u_c > 0, u_c * (a - _shift(a, 1, 0)), u_c * (_shift(a, -1, 0) - a)) / h
        fy = torch.where(v_c > 0, v_c * (a - _shift(a, 1, 1)), v_c * (_shift(a, -1, 1) - a)) / h
        self.alpha = (a - dt * (fx + fy) + beta * (target - a)).clamp(0.0, 1.0)
        self.alpha = self.alpha.masked_fill(self.solid, 0.0)

        # diagnostics: liquid divergence away from the free boundary, and the
        # inlet pressure level (should be ~0 for a correctly posed p_out = 0)
        div = _divergence(self.u, self.v, h)
        band = vap | _shift(vap.long(), 1, 0).bool() | _shift(vap.long(), -1, 0).bool()
        band = band | _shift(vap.long(), 1, 1).bool() | _shift(vap.long(), -1, 1).bool()
        interior = ~band & ~self.solid
        self.div_residual_max = max(
            self.div_residual_max, float(div[interior].abs().max()) * h
        )
        q_in = float(self.u[0, :].mean())
        q_out = float(self.u[-1, :].mean())
        self.mass_balance_max = max(
            self.mass_balance_max, abs(q_out - q_in) / max(abs(q_in), 1e-12)
        )
        self.p_inlet_history.append(float(self.p[:3, :].mean()))
        self._step += 1

    # -- sampling ---------------------------------------------------------
    def cavity_metrics(self) -> dict:
        return cavity_metrics(
            self.alpha, _as_flow_config(self.cfg),
            threshold=self.cfg.cavity_threshold, geometry=self.geometry,
        )

    def sample(self) -> None:
        self.time_history.append(self._step * self.dt)
        self.cl_history.append(self.cl())
        self.cd_history.append(self.cd())
        self.lcav_history.append(self.cavity_metrics()["l_cav"])
        self.mask_history.append(int((self.alpha > self.cfg.cavity_threshold).sum()))

    def run(self, progress_every: int = 0) -> dict:
        for _ in range(self.cfg.n_steps):
            self.step()
            if self._step % self.cfg.sample_interval == 0:
                self.sample()
            if progress_every and self._step % progress_every == 0:
                m = self.cavity_metrics()
                print(
                    f"  step {self._step:6d} t={self._step * self.dt:7.2f} "
                    f"L/c={m['l_cav']:.3f} d/c={m['delta_cav']:.4f} "
                    f"Cl={self.cl_history[-1]:+.3f} Cd={self.cd_history[-1]:.3f} "
                    f"p_in={self.p_inlet_history[-1]:+.4f} cg={self.cg_iters[-1]}",
                    flush=True,
                )
        return self.summary()

    def summary(self) -> dict:
        m = self.cavity_metrics()
        n = len(self.cl_history)
        transient = int(self.cfg.transient_fraction * n)
        t = self.time_history[transient:]
        lc = self.lcav_history[transient:]
        cl = self.cl_history[transient:]
        f_cl, st_cl, spec = dominant_frequency(t, cl) if len(t) > 8 else (0.0, None, {})
        f_lc, st_lc, _ = dominant_frequency(t, lc) if len(t) > 8 else (0.0, None, {})
        return {
            "config": asdict(self.cfg),
            "l_cav": m["l_cav"],
            "delta_cav": m["delta_cav"],
            "detachment_x": m["detachment_x"],
            "closure_x": m["closure_x"],
            "cavity_area": m["area"],
            "l_cav_mean": float(sum(lc) / max(len(lc), 1)) if lc else 0.0,
            "l_cav_min": float(min(lc)) if lc else 0.0,
            "l_cav_max": float(max(lc)) if lc else 0.0,
            "shedding_amplitude": (float(max(lc) - min(lc)) if lc else 0.0),
            "cl_mean": float(sum(cl) / max(len(cl), 1)) if cl else 0.0,
            "cd_mean": float(sum(self.cd_history[transient:]) / max(len(self.cd_history[transient:]), 1)),
            "st_cl": (st_cl[1] if st_cl else 0.0),
            "st_lc": (st_lc[1] if st_lc else 0.0),
            "f_cl": f_cl,
            "f_lc": f_lc,
            "spectrum": spec,
            "div_residual_max": self.div_residual_max,
            "mass_balance_max": self.mass_balance_max,
            "cg_iters_mean": float(sum(self.cg_iters) / max(len(self.cg_iters), 1)),
            "cg_residual_max": self.cg_residual_max,
            "p_inlet_mean": float(sum(self.p_inlet_history[-50:]) / max(len(self.p_inlet_history[-50:]), 1)),
            "n_vapour_cells": int((self.alpha > self.cfg.cavity_threshold).sum()),
        }


def _as_flow_config(cfg: EquilibriumCavityConfig) -> CavitationFlowConfig:
    """Adapter so :func:`cavity_metrics` can be reused unchanged."""
    return CavitationFlowConfig(
        airfoil=cfg.airfoil,
        alpha_deg=cfg.alpha_deg,
        sigma=cfg.sigma,
        re=cfg.re,
        cells_per_chord=cfg.cells_per_chord,
        domain_x_chords=cfg.domain_x_chords,
        domain_y_chords=cfg.domain_y_chords,
        le_x_chords=cfg.le_x_chords,
        le_y_chords=cfg.le_y_chords,
        n_steps=cfg.n_steps,
    )