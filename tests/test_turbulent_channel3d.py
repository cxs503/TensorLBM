"""Analytic verification ladder for ``tensorlbm.turbulence.turbulent_channel3d``.

Every anchor below is closed form in lattice units, so the residual is a
discretisation error and not a solver-to-solver comparison:

===== ======================= ==========================================
anchor quantity tested       analytic baseline
===== ======================= ==========================================
A      momentum source        dU/dt = a            (Guo forcing, exact)
B      bulk viscosity         nu_eff = (tau+ - 1/2)/3
C      laminar channel        u(y) = a y (H-y) / (2 nu)
D      wall-origin geometry   measured offset tracks the imposed ``q``
===== ======================= ==========================================

Negative controls (the anchors must be able to fail):

* ``test_guo_sp_force_split_underdrives`` -- the audited ``guo_sp`` variant
  injects ~2.4 % of ``a`` at the campaign ``tau+``; anchor A flags it.
* ``test_legacy_twin_parameterisation_is_hypersonic`` -- running the laminar
  twin at the campaign ``Re_tau`` is hypersonic (``u_max = u_tau Re_tau/2``)
  and diverges; the ``u_max_target`` twin removes that coupling.

All cases run on CPU float32 with one torch thread: the numbers are device
independent and the whole module stays inside the ~25 s PR fast lane.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.turbulence.turbulent_channel3d import (
    TurbulentChannel3D,
    TurbulentChannel3DConfig,
    poiseuille_exact,
    reichardt_u_plus,
    run_laminar_poiseuille,
)

NX, NZ, NY = 8, 8, 18
U_TAU = 0.01
HALF_HEIGHT = (NY - 2) / 2.0


@pytest.fixture(autouse=True)
def _single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _nu_to_config(nu: float, **kwargs) -> TurbulentChannel3DConfig:
    """Config whose ``nu`` is exactly the requested value (u_tau fixed)."""
    return TurbulentChannel3DConfig(
        nx=NX, nz=NZ, ny=NY, u_tau=U_TAU, re_tau=U_TAU * HALF_HEIGHT / nu, **kwargs
    )


def _uniform_solver(config: TurbulentChannel3DConfig) -> TurbulentChannel3D:
    sim = TurbulentChannel3D(config)
    ones = torch.ones(1, sim.N, dtype=sim.dtype)
    zeros = torch.zeros(3, sim.N, dtype=sim.dtype)
    sim.f = sim._equilibrium(ones, zeros).reshape(19, sim.nz, sim.ny, sim.nx)
    return sim


def _interior_ux(sim: TurbulentChannel3D) -> float:
    m = torch.mm(sim.A, sim.f.view(19, sim.N))
    ux = (m[1] / m[0]).view(sim.nz, sim.ny, sim.nx)
    return float(ux[:, 6:12, :].mean())


# ---------------------------------------------------------------------------
# anchor A: the force term must deliver exactly dU/dt = a
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("nu", [0.002, 0.02, 0.04, 0.12, 0.2])
def test_force_impulse_equals_acceleration(nu: float):
    """Interior momentum grows at rate ``a`` for every resolved tau+."""
    accel = 1e-4
    steps = 12
    cfg = _nu_to_config(nu, steps=steps, warmup_steps=0, body_force=accel)
    sim = _uniform_solver(cfg)
    u0 = _interior_ux(sim)
    for _ in range(steps):
        sim._step_once()
    rate = (_interior_ux(sim) - u0) / steps / accel
    assert abs(rate - 1.0) < 5e-3, f"tau+={cfg.tau_plus:.4f}: dU/dt = {rate:.4f} a"


def test_guo_sp_force_split_underdrives():
    """Negative control: the audited ``guo_sp`` split loses the drive at low nu."""
    accel = 1e-4
    steps = 12
    campaign_nu = U_TAU * HALF_HEIGHT / 180.0  # Re_tau = 180 campaign viscosity
    ratios = {}
    for scheme in ("trt_split", "guo_sp"):
        cfg = _nu_to_config(
            campaign_nu, steps=steps, warmup_steps=0, body_force=accel, force_scheme=scheme
        )
        sim = _uniform_solver(cfg)
        u0 = _interior_ux(sim)
        for _ in range(steps):
            sim._step_once()
        ratios[scheme] = (_interior_ux(sim) - u0) / steps / accel
    assert abs(ratios["trt_split"] - 1.0) < 5e-3
    assert ratios["guo_sp"] < 0.05, (
        "guo_sp was expected to underdrive at campaign tau+; measured "
        f"{ratios['guo_sp']:.4f} a -- if this changed, re-audit the forcing split"
    )
    assert ratios["trt_split"] / max(ratios["guo_sp"], 1e-12) > 20.0


# ---------------------------------------------------------------------------
# anchor B: unforced shear mode decays at the configured viscosity
# ---------------------------------------------------------------------------
def test_shear_wave_viscosity_matches_tau_plus():
    """``nu_eff`` from a wall-normal sine mode must equal ``(tau+-1/2)/3``."""
    nu = 0.04
    steps = 2000
    cfg = _nu_to_config(nu, steps=steps, warmup_steps=0, body_force=0.0, wall_scheme="halfway")
    sim = TurbulentChannel3D(cfg)
    height = cfg.height
    y = torch.arange(NY, dtype=torch.float64) - 0.5
    mode = torch.sin(math.pi * y / height)
    mode[0] = 0.0
    mode[-1] = 0.0
    rho = torch.ones(NZ, NY, NX, dtype=sim.dtype)
    ux = (0.01 * mode).to(sim.dtype).view(1, NY, 1).expand(NZ, NY, NX).contiguous()
    zeros = torch.zeros_like(rho)
    sim.f = sim._equilibrium(
        rho.reshape(1, sim.N), torch.stack((ux, zeros, zeros)).reshape(3, sim.N)
    ).reshape(19, NZ, NY, NX)

    def amplitude() -> float:
        m = torch.mm(sim.A, sim.f.view(19, sim.N))
        profile = (m[1] / m[0]).view(NZ, NY, NX).mean(dim=(0, 2)).double()
        return float((profile * mode).sum() / (mode * mode).sum())

    a_early = None
    for step in range(1, steps + 1):
        sim._step_once()
        if step == 500:
            a_early = amplitude()
    a_late = amplitude()
    k2 = (math.pi / height) ** 2
    nu_eff = -math.log(a_late / a_early) / ((steps - 500) * k2)
    assert abs(nu_eff / nu - 1.0) < 0.05, f"nu_eff = {nu_eff:.5f} vs nu = {nu:.5f}"


# ---------------------------------------------------------------------------
# anchor C: laminar twin vs the exact plane-Poiseuille parabola
# ---------------------------------------------------------------------------
def test_laminar_twin_recovers_exact_poiseuille():
    """Campaign ``Re_tau`` inherited, but the twin re-derives a subsonic Mach."""
    cfg = TurbulentChannel3DConfig(nx=NX, nz=NZ, ny=NY, re_tau=180.0, u_tau=U_TAU)
    result = run_laminar_poiseuille(cfg, steps=800)
    twin = result["config"]
    assert twin["u_max_target"] == 0.05
    assert result["u_max_analytic"] < 0.06, "twin must stay subsonic"
    assert twin["tau_plus"] > 0.55, "twin must be well resolved (tau+ away from 1/2)"
    # 800 steps still carries a startup transient: measured L2 = 6.8e-3 and it
    # keeps falling (3.3e-3 at 2000 steps, 9.2e-4 with a 2000-step window), so
    # the ladder asserts a converged bound rather than the asymptotic rate.
    assert result["relative_l2_error"] < 1.5e-2, (
        f"laminar twin L2 error {result['relative_l2_error']:.3e} vs exact parabola"
    )
    ratio = result["u_max_mean"] / result["u_max_analytic"]
    assert abs(ratio - 1.0) < 2e-2, f"peak velocity ratio {ratio:.5f}"
    # halfway bounce-back places the no-slip plane half a cell from the first
    # fluid node; the fit may only recover a small numerical slip on top.
    assert abs(result["wall_origin_shift"]) < 0.05, (
        f"measured wall-origin shift {result['wall_origin_shift']:+.4f} exceeds 0.05"
    )
    # the body-force route is exact by construction and is the reference here
    assert abs(result["u_tau_body_force"] / twin["u_tau"] - 1.0) < 1e-3
    # the profile must be a parabola, not merely of the right amplitude
    analytic = torch.tensor(result["u_analytic"], dtype=torch.float64)
    measured = torch.tensor(result["u_mean"], dtype=torch.float64)
    rel = (measured - analytic).abs() / analytic.max()
    assert float(rel.max()) < 2e-2, f"max pointwise error {float(rel.max()):.4f}"


def test_wall_origin_semantics_is_a_documented_limitation():
    """Anchor D: halfway is recovered, but the fit does *not* track ``q != 0.5``.

    Measured 2026-11-07 on this branch (ny = 18, twin re-parameterised to
    u_max = 0.05): the estimator returns 0.4725 / 0.4800 / 0.4850 at
    800 / 1200 / 2000 steps for ``wall_origin = 0.5`` and 0.4950 / 0.5175 /
    0.5375 for ``wall_origin = 0.375`` with ``wall_scheme="interpolated"``, i.e.
    it stays near the half-way origin instead of following the imposed -0.125
    geometry, and the interpolated profile keeps a 2.5-3.1 % residual that does
    *not* shrink with steps.  Consequence: ``wall_origin != 0.5`` must not be
    used for a ``y+`` calibration until that is resolved -- the campaign runs
    half-way bounce-back.  This test freezes the observation so that a future
    fix fails it and forces a re-audit.
    """
    cfg_half = TurbulentChannel3DConfig(nx=NX, nz=NZ, ny=NY, re_tau=180.0, u_tau=U_TAU)
    cfg_interp = TurbulentChannel3DConfig(
        nx=NX, nz=NZ, ny=NY, re_tau=180.0, u_tau=U_TAU,
        wall_origin=0.375, wall_scheme="interpolated",
    )
    half = run_laminar_poiseuille(cfg_half, steps=800)
    interp = run_laminar_poiseuille(cfg_interp, steps=800)
    assert abs(half["wall_origin_measured"] - 0.5) < 0.05
    assert abs(interp["wall_origin_measured"] - 0.5) < 0.05, (
        "the wall-origin estimator now tracks the imposed q; the y+ calibration "
        "limitation documented in docs/turbulent_channel3d_verification_r1.md "
        "may be lifted -- re-audit before trusting wall_origin != 0.5"
    )
    assert interp["relative_l2_error"] > half["relative_l2_error"], (
        "interpolated bounce-back no longer loses to halfway; re-audit"
    )


# ---------------------------------------------------------------------------
# negative control for the calibration path itself
# ---------------------------------------------------------------------------
def test_legacy_twin_parameterisation_is_hypersonic():
    """``u_max_target=None`` keeps the campaign Re_tau and diverges to NaN."""
    cfg = TurbulentChannel3DConfig(nx=NX, nz=NZ, ny=NY, re_tau=180.0, u_tau=U_TAU)
    legacy = run_laminar_poiseuille(cfg, steps=400, u_max_target=None)
    assert legacy["u_max_analytic"] > 0.5, "legacy twin is hypersonic by construction"
    assert not math.isfinite(legacy["relative_l2_error"]), (
        "legacy twin was expected to diverge; if it now stays finite the "
        "u_max_target parameterisation may be unnecessary -- re-audit"
    )
    fixed = run_laminar_poiseuille(cfg, steps=800)
    assert math.isfinite(fixed["relative_l2_error"])
    assert fixed["relative_l2_error"] < 1.5e-2


# ---------------------------------------------------------------------------
# analytic helpers and fail-closed configuration
# ---------------------------------------------------------------------------
def test_poiseuille_exact_matches_its_own_flux_and_walls():
    height, accel, nu = 16.0, 2e-4, 0.03
    y = torch.linspace(0.0, height, 2001, dtype=torch.float64)
    u = poiseuille_exact(y, height, accel, nu)
    assert float(u[0]) == 0.0 and float(u[-1]) == 0.0
    flux = float(torch.trapezoid(u, y))
    assert abs(flux / (accel * height**3 / (12.0 * nu)) - 1.0) < 1e-4
    peak = float(u.max())
    assert abs(peak / (accel * height**2 / (8.0 * nu)) - 1.0) < 1e-3


def test_reichardt_law_is_consistent_with_the_log_law():
    """Reichardt must approach the von Karman log law in the inertial layer."""
    y_plus = torch.tensor([100.0, 300.0, 1000.0], dtype=torch.float64)
    reichardt = reichardt_u_plus(y_plus)
    log_law = torch.log(y_plus) / 0.41 + 5.2
    # the two correlations agree to a fraction of a u_tau over 100 < y+ < 1000
    assert float((reichardt - log_law).abs().max()) < 1.0
    assert float(reichardt[0]) > float(reichardt_u_plus(torch.tensor([10.0], dtype=torch.float64))[0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ny": 8},
        {"re_tau": 0.0},
        {"u_tau": 0.0},
        {"wall_origin": 0.1},
        {"wall_scheme": "specular"},
        {"force_scheme": "guo"},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        TurbulentChannel3DConfig(**kwargs).validate()