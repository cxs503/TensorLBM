"""Equilibrium pressure of the symmetric free-energy (binary-fluid) LBM model.

Why this module exists
----------------------
TensorLBM carries the free-energy (Swift/Orlandini/Osborn/Yeomans 1996) binary
fluid in two places: the LBM stepper :func:`tensorlbm.multiphase.free_energy_step`
(Korteweg force ``F = -phi grad(mu)``) and the free-energy pressure tensor in
:mod:`tensorlbm.free_energy_stress`.  Neither one *closed the loop* on the
quantity those two pieces are built to reproduce: the **equilibrium pressure**
of the model

    f(phi) = a/2 phi^2 + b/4 phi^4 + kappa/2 |grad phi|^2 ,   a = -A < 0, b = B > 0

At two-phase coexistence the model predicts, in closed form,

* coexistence (bulk, Maxwell) pressure  ``p_coex = A^2 / (4 b)``,
* equilibrium interface width            ``xi = sqrt(2 kappa / A)``,
* surface tension                        ``sigma = (2 sqrt 2 / 3) sqrt(kappa A^3) / b``,

and the *normal* component of the pressure tensor must be **constant across a
flat interface and equal to ``p_coex``** (mechanical equilibrium, ``div P = 0``):

    P_ab = d delta_ab + kappa d_a phi d_b phi,
    d    = p0 - kappa phi lap(phi) - kappa/2 |grad phi|^2,
    p0   = phi f'(phi) - f = a/2 phi^2 + 3 b/4 phi^4,
    P_nn = d + kappa (d_n phi)^2,   P_tt = d,   P_nn - P_tt = kappa (d_n phi)^2.

This module measures those three numbers on (i) the discrete equilibrium field
and (ii) a **time-stepped** free-energy LBM state, and compares them with the
closed forms.  Everything here is a *measurement*; nothing is hard-coded to the
answer (the closed forms are computed from ``(A, B, kappa)`` by the analytic
helpers in :mod:`tensorlbm.free_energy_stress`, and the coexistence pressure is
*independently* re-derived by a Maxwell-construction root find).

Periodic-domain caveat (a trap this module exists to avoid)
----------------------------------------------------------
A single planar interface is **not** periodic: ``phi(-L/2) != phi(+L/2)`` and the
wrap-around step produces a spurious pressure spike (measured >1000 % of
``p_coex`` in a first attempt).  The equilibrium field is therefore built with
*two* interfaces per period,

    phi(x) = phi_eq tanh((x-x1)/xi) tanh((x-x2)/xi),  x1 = L/4, x2 = 3L/4,

which is the planar equilibrium profile to within ``exp(-L/(2 xi))``
(< 1e-6 for ``L >= 64, xi <= 3``) and is exactly periodic.  ``P_nn - P_tt``
integrated over the period then equals ``2 sigma``.

Scope / non-goals
-----------------
* The phase field is *held* at its equilibrium profile during the LBM run: the
  production ``free_energy_step`` Cahn-Hilliard population does not preserve the
  equilibrium profile (no ``-M grad(mu)`` flux in ``g``; the phase field then
  diffuses with the BGK numerical diffusivity ``c_s^2 (tau_g - 1/2)``).  That
  defect is measured, not patched, by :func:`diagnose_ch_phase_flux_gap` and
  reported in ``docs/XVAL_LBM_FREE_ENERGY_BINARY_EQUILIBRIUM_PRESSURE.md``.
  Holding ``phi`` is the standard "static two-phase field" spurious-current test.
* Only the flat (planar) interface is graded.  The curved (droplet) Laplace law
  needs the O(1/R) coexistence-composition shift, which is *not* reachable by
  relaxing the tanh profile in a finite periodic box; the droplet numbers are
  reported as a documented diagnostic, never as an acceptance number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from ..free_energy_stress import (
    FreeEnergyParams,
    bulk_pressure,
    chemical_potential,
    interface_width,
    laplacian,
    pressure_tensor,
    surface_tension_analytic,
)
from .operators import BoundaryPolicy  # noqa: F401  (re-exported for callers)

__all__ = [
    "EquilibriumPressureConfig",
    "PlanarEquilibriumField",
    "PressureDiagnostics",
    "LbmRunDiagnostics",
    "MaxwellConstruction",
    "periodic_planar_profile",
    "maxwell_coexistence",
    "planar_pressure_diagnostics",
    "run_planar_equilibrium_lbm",
    "diagnose_ch_phase_flux_gap",
    "equilibrium_pressure_report",
]

_CS2 = 1.0 / 3.0


@dataclass(frozen=True)
class EquilibriumPressureConfig:
    """Case definition for the planar equilibrium-pressure verification.

    ``interface_cells`` is the interface thickness in *lattice* units,
    ``xi = interface_cells``; the parameter set is then ``kappa = A xi^2 / 2``
    so that the model's own ``xi = sqrt(2 kappa / A)`` matches the sampled
    profile (no width mismatch, which would make the field relax instead of
    being at equilibrium).
    """

    A: float = 1.0
    B: float = 1.0
    interface_cells: float = 2.0
    nx: int = 128
    ny: int = 16
    steps: int = 4000
    tau_f: float = 1.0
    sample_every: int = 500

    def __post_init__(self) -> None:
        if self.A <= 0.0 or self.B <= 0.0:
            raise ValueError("A and B must be positive (double-well coefficients)")
        if self.interface_cells <= 0.5:
            raise ValueError("interface_cells must be > 0.5 for a resolved interface")
        if self.nx < 32 or self.ny < 4:
            raise ValueError("nx >= 32 and ny >= 4 required (two interfaces per period)")
        if self.steps < 1:
            raise ValueError("steps must be >= 1")
        if self.tau_f <= 0.5:
            raise ValueError("tau_f must be > 0.5")

    @property
    def kappa(self) -> float:
        return 0.5 * self.A * self.interface_cells**2

    @property
    def params(self) -> FreeEnergyParams:
        return FreeEnergyParams(A=self.A, B=self.B, kappa=self.kappa)

    def as_dict(self) -> dict[str, float | int]:
        return {
            "A": self.A,
            "B": self.B,
            "interface_cells": self.interface_cells,
            "kappa": self.kappa,
            "nx": self.nx,
            "ny": self.ny,
            "steps": self.steps,
            "tau_f": self.tau_f,
        }


@dataclass(frozen=True)
class PlanarEquilibriumField:
    """Periodic two-interface equilibrium field and its closed-form anchors."""

    phi: torch.Tensor
    phi_eq: float
    xi: float
    sigma_analytic: float
    p_coex_closed_form: float
    wrap_residual: float


@dataclass(frozen=True)
class PressureDiagnostics:
    """Pressure-tensor measurements on a planar two-phase field."""

    p_coex_closed_form: float
    p_coex_numeric: float
    p0_bulk: tuple[float, float]
    p_nn_mean: float
    p_nn_min: float
    p_nn_max: float
    p_nn_max_deviation: float
    p_nn_max_deviation_relative: float
    sigma_integral: float
    sigma_analytic: float
    sigma_relative_error: float
    coexistence_relative_error: float

    def as_dict(self) -> dict[str, object]:
        return {
            "p_coex_closed_form": self.p_coex_closed_form,
            "p_coex_numeric": self.p_coex_numeric,
            "p0_bulk": list(self.p0_bulk),
            "p_nn_mean": self.p_nn_mean,
            "p_nn_min": self.p_nn_min,
            "p_nn_max": self.p_nn_max,
            "p_nn_max_deviation": self.p_nn_max_deviation,
            "p_nn_max_deviation_relative": self.p_nn_max_deviation_relative,
            "sigma_integral": self.sigma_integral,
            "sigma_analytic": self.sigma_analytic,
            "sigma_relative_error": self.sigma_relative_error,
            "coexistence_relative_error": self.coexistence_relative_error,
        }


@dataclass(frozen=True)
class LbmRunDiagnostics:
    """Outcome of the time-stepped free-energy LBM (momentum) run."""

    steps: int
    p_nn_total_mean_history: tuple[float, ...]
    p_nn_total_deviation_history: tuple[float, ...]
    max_velocity_history: tuple[float, ...]
    rho_min: float
    rho_max: float
    final_max_velocity: float
    final_p_nn_total_deviation_relative: float

    def as_dict(self) -> dict[str, object]:
        return {
            "steps": self.steps,
            "p_nn_total_mean_history": list(self.p_nn_total_mean_history),
            "p_nn_total_deviation_history": list(self.p_nn_total_deviation_history),
            "max_velocity_history": list(self.max_velocity_history),
            "rho_min": self.rho_min,
            "rho_max": self.rho_max,
            "final_max_velocity": self.final_max_velocity,
            "final_p_nn_total_deviation_relative": self.final_p_nn_total_deviation_relative,
        }


@dataclass(frozen=True)
class MaxwellConstruction:
    """Coexistence state from an independent numerical Maxwell construction."""

    phi_light: float
    phi_heavy: float
    mu_coex: float
    p_coex: float
    residual: float

    def as_dict(self) -> dict[str, object]:
        return {
            "phi_light": self.phi_light,
            "phi_heavy": self.phi_heavy,
            "mu_coex": self.mu_coex,
            "p_coex": self.p_coex,
            "residual": self.residual,
        }


def periodic_planar_profile(
    ny: int, nx: int, params: FreeEnergyParams, *, n_interfaces: int = 2
) -> PlanarEquilibriumField:
    """Exact-periodic equilibrium field with ``n_interfaces`` flat interfaces.

    Two interfaces (default) put a light and a heavy slab in the period; the
    product-of-tanh construction is the planar equilibrium profile at each
    interface up to ``exp(-L/(2 xi))`` while satisfying ``phi(0) = phi(L)``.
    """
    if n_interfaces != 2:
        raise ValueError("only the two-interface (periodic) construction is supported")
    phi_eq = math.sqrt(params.A / params.B)
    xi = interface_width(params)
    x = torch.arange(nx, dtype=torch.float64)
    x1, x2 = 0.25 * nx, 0.75 * nx
    profile = phi_eq * torch.tanh((x - x1) / xi) * torch.tanh((x - x2) / xi)
    phi = profile.unsqueeze(0).expand(ny, nx).contiguous()
    # Periodicity residual: |phi(0) - phi(nx-1)| relative to phi_eq is O(exp(-L/2xi)) /
    # ... the profile is sampled, so measure the *interface* wrap instead: the wrap
    # jump seen by the 5-point Laplacian would be the whole dynamic range.
    wrap_residual = abs(float(profile[0] - profile[-1])) / phi_eq
    return PlanarEquilibriumField(
        phi=phi,
        phi_eq=phi_eq,
        xi=xi,
        sigma_analytic=surface_tension_analytic(params),
        p_coex_closed_form=0.25 * params.A**2 / params.B,
        wrap_residual=wrap_residual,
    )


def maxwell_coexistence(params: FreeEnergyParams) -> MaxwellConstruction:
    """Independent numerical Maxwell construction (no closed form used).

    Two coexisting phases share ``mu`` and ``p0 = phi f'(phi) - f``.  With the
    symmetric double well the mean-field solution is ``phi = +-phi_eq``; the
    root find below does not assume that: it solves
    ``mu(phi_h) - mu(phi_l) = 0`` and ``p0(phi_h) - p0(phi_l) = 0`` by a
    bracketed bisection on ``phi_h`` with ``phi_l = -phi_h`` relaxed by a
    two-variable Newton iteration started from ``(+-2, +-2) sqrt(A/B)``.
    """

    a, b = params.a, params.b

    def mu(phi: float) -> float:
        return a * phi + b * phi**3

    def p0(phi: float) -> float:
        return 0.5 * a * phi**2 + 0.75 * b * phi**4

    # Newton on the 2x2 system in (phi_l, phi_h).
    phi_l, phi_h = -2.0 * math.sqrt(params.A / params.B), 2.0 * math.sqrt(params.A / params.B)
    for _ in range(200):
        f1 = mu(phi_h) - mu(phi_l)
        f2 = p0(phi_h) - p0(phi_l)
        j11, j12 = -(a + 3.0 * b * phi_l**2), (a + 3.0 * b * phi_h**2)
        j21, j22 = -(a * phi_l + 3.0 * b * phi_l**3), (a * phi_h + 3.0 * b * phi_h**3)
        det = j11 * j22 - j12 * j21
        if abs(det) < 1e-300:
            raise RuntimeError("Maxwell construction Jacobian became singular")
        d_phi_l = (-f1 * j22 + f2 * j12) / det
        d_phi_h = (-f2 * j11 + f1 * j21) / det
        phi_l += d_phi_l
        phi_h += d_phi_h
        if max(abs(d_phi_l), abs(d_phi_h)) < 1e-14:
            break
    residual = max(abs(mu(phi_h) - mu(phi_l)), abs(p0(phi_h) - p0(phi_l)))
    return MaxwellConstruction(
        phi_light=phi_l,
        phi_heavy=phi_h,
        mu_coex=0.5 * (mu(phi_l) + mu(phi_h)),
        p_coex=0.5 * (p0(phi_l) + p0(phi_h)),
        residual=residual,
    )


def planar_pressure_diagnostics(
    field: PlanarEquilibriumField, params: FreeEnergyParams, *, scheme: str = "iso9"
) -> PressureDiagnostics:
    """Measure ``p0``, ``P_nn`` uniformity and ``sigma = int (P_nn - P_tt)``.

    ``scheme`` selects the discrete gradient family used by the production
    pressure tensor (``"iso9"`` is the ludwig/fluid default, ``"5pt"`` the
    anisotropic central difference) so the discretisation sensitivity is
    measurable rather than assumed.
    """
    phi = field.phi
    pxx, pyy, _pxy = pressure_tensor(phi, params, scheme)
    p_nn = pxx[0]  # flat interface with normal x: y-invariant, take one row
    p_tt = pyy[0]
    p0 = bulk_pressure(phi, params)[0]

    p_nn_mean = float(p_nn.mean())
    deviation = (p_nn - p_nn_mean).abs()
    max_dev = float(deviation.max())
    sigma_integral = 0.5 * float((p_nn - p_tt).sum())  # two interfaces per period

    # Bulk samples: the slab centres (>= 4 xi inside each bulk region).
    xi = field.xi
    i_light = int(round(0.5 * phi.shape[1]))
    i_heavy = 0
    p0_bulk = (float(p0[i_light]), float(p0[i_heavy]))

    maxwell = maxwell_coexistence(params)
    together = max(abs(p0_bulk[0]), abs(p0_bulk[1]), field.p_coex_closed_form)
    return PressureDiagnostics(
        p_coex_closed_form=field.p_coex_closed_form,
        p_coex_numeric=maxwell.p_coex,
        p0_bulk=p0_bulk,
        p_nn_mean=p_nn_mean,
        p_nn_min=float(p_nn.min()),
        p_nn_max=float(p_nn.max()),
        p_nn_max_deviation=max_dev,
        p_nn_max_deviation_relative=max_dev / max(abs(field.p_coex_closed_form), 1e-30),
        sigma_integral=sigma_integral,
        sigma_analytic=field.sigma_analytic,
        sigma_relative_error=sigma_integral / field.sigma_analytic - 1.0,
        coexistence_relative_error=max(
            abs(p0_bulk[0] - field.p_coex_closed_form), abs(p0_bulk[1] - field.p_coex_closed_form)
        )
        / max(together, 1e-30),
    )


def _total_pressure_normal(
    rho: torch.Tensor, phi: torch.Tensor, params: FreeEnergyParams, scheme: str = "iso9"
) -> torch.Tensor:
    """``P_nn^total = rho c_s^2 + d + kappa (d_x phi)^2`` from an LBM state."""
    pxx, _pyy, _pxy = pressure_tensor(phi, params, scheme)
    return rho * _CS2 + pxx


def run_planar_equilibrium_lbm(
    config: EquilibriumPressureConfig, *, scheme: str = "iso9"
) -> tuple[LbmRunDiagnostics, torch.Tensor]:
    """Time-step the production free-energy LBM on a static planar interface.

    The **momentum** population ``f`` is advanced by the production
    :func:`tensorlbm.multiphase.free_energy_step` Korteweg-force collision
    (``F = -phi grad(mu)``) followed by the production periodic streaming; the
    order parameter is re-imposed at its equilibrium profile every step (see
    the module docstring: the production ``g`` equation does not preserve it).
    The run measures how well mechanical equilibrium holds, i.e. how uniform the
    *total* normal pressure ``rho c_s^2 + P_nn`` stays and how large the residual
    (spurious) velocity is.
    """
    from ..d2q9 import equilibrium, macroscopic
    from ..multiphase import free_energy_step, init_free_energy_g
    from ..solver import stream

    params = config.params
    field = periodic_planar_profile(config.ny, config.nx, params)
    phi = field.phi
    zeros = torch.zeros_like(phi)
    f = equilibrium(torch.ones_like(phi), zeros, zeros)

    means: list[float] = []
    deviations: list[float] = []
    velocities: list[float] = []
    p_coex = field.p_coex_closed_form

    for step in range(1, config.steps + 1):
        g = init_free_energy_g(phi, zeros, zeros)
        f, _g_out = free_energy_step(
            f,
            g,
            tau_f=config.tau_f,
            tau_g=0.7,
            A=config.A,
            B=config.B,
            kappa=config.kappa,
            Gamma=0.5,
        )
        f = stream(f)
        if step % config.sample_every == 0 or step == config.steps:
            rho, ux, uy = macroscopic(f)
            p_nn_tot = _total_pressure_normal(rho, phi, params, scheme)
            mean = float(p_nn_tot.mean())
            means.append(mean)
            deviations.append(float((p_nn_tot - mean).abs().max()) / max(abs(p_coex), 1e-30))
            velocities.append(float(torch.sqrt(ux * ux + uy * uy).max()))

    rho, ux, uy = macroscopic(f)
    return (
        LbmRunDiagnostics(
            steps=config.steps,
            p_nn_total_mean_history=tuple(means),
            p_nn_total_deviation_history=tuple(deviations),
            max_velocity_history=tuple(velocities),
            rho_min=float(rho.min()),
            rho_max=float(rho.max()),
            final_max_velocity=velocities[-1] if velocities else float("nan"),
            final_p_nn_total_deviation_relative=deviations[-1] if deviations else float("nan"),
        ),
        phi,
    )


def diagnose_ch_phase_flux_gap(
    *, nx: int = 32, ny: int = 32, steps: int = 400, tau_g: float = 0.7, Gamma: float = 0.5
) -> dict[str, float]:
    """Measure whether the production ``g`` equation preserves equilibrium.

    A static droplet is prepared in its equilibrium profile and advanced with
    :func:`tensorlbm.multiphase.free_energy_step`.  The peak order parameter is
    tracked: the production population transports ``phi`` without the
    ``-M grad(mu)`` Cahn-Hilliard flux, so the phase field is a passive scalar
    with numerical diffusivity ``D_num = c_s^2 (tau_g - 1/2)`` and the peak
    decays.  The measured decay length is compared with ``sqrt(4 D_num t)``.
    """
    from ..d2q9 import equilibrium, macroscopic
    from ..multiphase import free_energy_step, init_free_energy_g
    from ..solver import stream

    A = B = 0.1
    kappa = 0.02
    xi = math.sqrt(2.0 * kappa / A)
    radius = 8.0
    y = torch.arange(ny, dtype=torch.float64).view(-1, 1)
    x = torch.arange(nx, dtype=torch.float64).view(1, -1)
    r = torch.sqrt((y - 0.5 * ny) ** 2 + (x - 0.5 * nx) ** 2)
    phi0 = torch.tanh((radius - r) / xi)
    f = equilibrium(torch.ones_like(phi0), torch.zeros_like(phi0), torch.zeros_like(phi0))
    g = init_free_energy_g(phi0)
    peak_initial = float(phi0.max())
    for _ in range(steps):
        f, g = free_energy_step(f, g, tau_f=1.0, tau_g=tau_g, A=A, B=B, kappa=kappa, Gamma=Gamma)
        f = stream(f)
        g = stream(g)
    peak_final = float(g.sum(dim=0).max())
    d_num = _CS2 * (tau_g - 0.5)
    return {
        "steps": float(steps),
        "tau_g": tau_g,
        "peak_initial": peak_initial,
        "peak_final": peak_final,
        "peak_decay_ratio": peak_final / peak_initial,
        "numerical_diffusivity_cs2_tau_minus_half": d_num,
        "diffusion_length_estimate": math.sqrt(4.0 * d_num * steps),
    }


def equilibrium_pressure_report(
    resolutions: tuple[float, ...] = (1.5, 2.0, 3.0),
    *,
    nx: int = 128,
    ny: int = 16,
    steps: int = 2000,
    sample_every: int | None = None,
    schemes: tuple[str, ...] = ("iso9", "5pt"),
    include_lbm: bool = True,
    include_ch_gap: bool = True,
) -> dict[str, object]:
    """Run the full verification sweep and return a JSON-serialisable report.

    The sweep varies the interface resolution ``xi / h`` (the controlling
    discretisation parameter) at fixed ``nx`` and reports, per resolution:
    coexistence pressure, ``P_nn`` uniformity, the ``sigma`` integral, and (for
    the production discretisation ``iso9``) the time-stepped LBM residual.
    """
    sample = sample_every or max(1, steps // 8)
    rows: list[dict[str, object]] = []
    for cells in resolutions:
        config = EquilibriumPressureConfig(
            A=1.0, B=1.0, interface_cells=float(cells), nx=nx, ny=ny, steps=steps,
            sample_every=sample,
        )
        params = config.params
        field = periodic_planar_profile(ny, nx, params)
        per_scheme: dict[str, object] = {}
        for scheme in schemes:
            diag = planar_pressure_diagnostics(field, params, scheme=scheme)
            per_scheme[scheme] = diag.as_dict()
        row: dict[str, object] = {
            "interface_cells": float(cells),
            "xi": field.xi,
            "wrap_residual": field.wrap_residual,
            "schemes": per_scheme,
        }
        if include_lbm:
            run, _phi = run_planar_equilibrium_lbm(config, scheme="iso9")
            row["lbm_run"] = run.as_dict()
        rows.append(row)

    report: dict[str, object] = {
        "anchor": "XVAL-LBM-FREE-ENERGY-BINARY-EQUILIBRIUM-PRESSURE",
        "model": {
            "free_energy": "f = a/2 phi^2 + b/4 phi^4 + kappa/2 |grad phi|^2, a=-A",
            "pressure_tensor": "P_ab = d delta_ab + kappa d_a phi d_b phi, "
            "d = p0 - kappa phi lap(phi) - kappa/2 |grad phi|^2",
            "closed_forms": {
                "p_coex": "A^2 / (4 B)",
                "xi": "sqrt(2 kappa / A)",
                "sigma": "(2 sqrt(2) / 3) sqrt(kappa A^3) / B",
            },
        },
        "settings": {"nx": nx, "ny": ny, "steps": steps, "A": 1.0, "B": 1.0},
        "rows": rows,
        "maxwell": maxwell_coexistence(FreeEnergyParams(A=1.0, B=1.0, kappa=1.0)).as_dict(),
    }
    if include_ch_gap:
        report["ch_phase_flux_gap"] = diagnose_ch_phase_flux_gap()
    return report