"""Tests for the frazil/grease-ice suspension closure (campaign node IN54W1-N03).

Covers :mod:`tensorlbm.frazil_suspension` (hindered settling, drag transition,
concentration-coupled rheology, effective diffusivity, flow capacity) and the
numerical half :mod:`tensorlbm.frazil_settling_lbm` (quadrature reference,
scalar advection-diffusion, closed settling column, variable-viscosity channel).

The LBM cases are deliberately small (H <= 48, <= 4000 steps) so the file runs in
a few seconds; the production-size runs and the full artifact live in
``benchmarks/frazil_suspension_bench.py``.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.frazil_lbm import poiseuille_analytic_profile
from tensorlbm.frazil_rheology import (
    DEFAULT_PHI_MAX,
    FrazilRheologyError,
    frazil_consistency_index,
    frazil_flow_index,
    relative_viscosity_krieger_dougherty,
)
from tensorlbm.frazil_settling_lbm import (
    _interface_height,
    _profile_error,
    powerlaw_tau_field,
    run_advection_diffusion_check,
    run_powerlaw_channel,
    run_settling_column,
    stratified_powerlaw_reference,
)
from tensorlbm.frazil_suspension import (
    DEFAULT_D_FRAZIL,
    MU_WATER,
    batch_settling_interface,
    concentration_sweep,
    difelice_drag_amplification,
    drag_coefficient,
    drag_coefficient_suspension,
    effective_diffusivity,
    effective_mixture_viscosity,
    flow_capacity_ratio,
    hindered_drag_amplification,
    hindered_settling_velocity,
    richardson_zaki_exponent,
    settling_from_drag,
    single_particle_settling_velocity,
    stokes_settling_velocity,
)

# ---------------------------------------------------------------------------
# Single-particle drag / settling
# ---------------------------------------------------------------------------


def test_drag_coefficient_stokes_limit_and_plateau():
    assert drag_coefficient(1e-4) == pytest.approx(24.0 / 1e-4, rel=3e-4)
    # Schiller-Naumann correction factor is exact by construction
    re = 3.7
    expected = 24.0 / re * (1.0 + 0.15 * re**0.687)
    assert float(drag_coefficient(re)) == pytest.approx(expected, rel=1e-14)
    # Newton plateau
    assert float(drag_coefficient(5.0e3)) == pytest.approx(0.44, rel=1e-14)


def test_drag_coefficient_rejects_non_positive_reynolds():
    with pytest.raises(FrazilRheologyError):
        drag_coefficient(0.0)
    with pytest.raises(FrazilRheologyError):
        drag_coefficient(-1.0)


def test_single_particle_settling_matches_stokes_in_creeping_limit():
    run = single_particle_settling_velocity(5.0e-6)  # 5 um floc -> Re << 1
    assert run["converged"]
    assert run["reynolds"] < 0.02
    assert run["velocity"] == pytest.approx(run["stokes_velocity"], rel=2e-2)
    assert run["drag_ratio"] == pytest.approx(1.0, abs=5e-2)


def test_single_particle_settling_is_slower_than_stokes_for_millimetre_flocs():
    run = single_particle_settling_velocity(DEFAULT_D_FRAZIL)
    assert run["velocity"] < run["stokes_velocity"]  # inertial drag slows it down
    assert 1.0 < run["reynolds"] < 100.0


def test_stokes_settling_velocity_guards():
    with pytest.raises(FrazilRheologyError):
        stokes_settling_velocity(-1.0)
    with pytest.raises(FrazilRheologyError):
        stokes_settling_velocity(1e-3, mu_fluid=0.0)
    with pytest.raises(FrazilRheologyError):
        stokes_settling_velocity(1e-3, rho_particle=1001.0)  # denser than water


# ---------------------------------------------------------------------------
# Richardson-Zaki
# ---------------------------------------------------------------------------


def test_richardson_zaki_exponent_branches():
    assert float(richardson_zaki_exponent(0.1)) == pytest.approx(4.65)
    assert float(richardson_zaki_exponent(0.5)) == pytest.approx(4.4 * 0.5**-0.03)
    assert float(richardson_zaki_exponent(10.0)) == pytest.approx(4.4 * 10.0**-0.1)
    assert float(richardson_zaki_exponent(1.0e3)) == pytest.approx(2.40)
    # within the frazil-relevant branch the exponent decreases monotonically
    re = torch.logspace(0.0, 2.5, 20)
    m = richardson_zaki_exponent(re)
    assert bool((m[1:] < m[:-1]).all())


def test_richardson_zaki_exponent_rejects_non_positive_reynolds():
    with pytest.raises(FrazilRheologyError):
        richardson_zaki_exponent(0.0)


def test_hindered_settling_velocity_limits_and_monotonicity():
    v_s0 = 0.02
    assert float(hindered_settling_velocity(0.0, v_s0, exponent=3.5)) == pytest.approx(v_s0)
    c = torch.linspace(0.0, 0.55, 40)
    v = hindered_settling_velocity(c, v_s0, exponent=3.5)
    assert bool((v[1:] < v[:-1]).all())
    assert float(v[-1]) > 0.0
    # explicit formula
    c_val = 0.3
    assert float(hindered_settling_velocity(c_val, v_s0, exponent=2.0)) == pytest.approx(
        v_s0 * (1.0 - c_val / DEFAULT_PHI_MAX) ** 2.0, rel=1e-14
    )


def test_hindered_settling_velocity_guards():
    with pytest.raises(FrazilRheologyError):
        hindered_settling_velocity(DEFAULT_PHI_MAX, 0.01, exponent=3.0)
    with pytest.raises(FrazilRheologyError):
        hindered_settling_velocity(-0.1, 0.01, exponent=3.0)
    with pytest.raises(FrazilRheologyError):
        hindered_settling_velocity(0.1, 0.01)  # neither exponent nor Re
    with pytest.raises(FrazilRheologyError):
        hindered_settling_velocity(0.1, 0.01, exponent=3.0, re_p=5.0)  # both
    with pytest.raises(FrazilRheologyError):
        hindered_settling_velocity(0.1, -0.01, exponent=3.0)


def test_drag_amplification_identity_and_roundtrip():
    m = 3.55
    c = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    omega = hindered_drag_amplification(c, exponent=m)
    expected = (1.0 - c / DEFAULT_PHI_MAX) ** (-2.0 * m)
    assert torch.allclose(omega, expected, rtol=1e-13)
    assert float(omega[0]) == pytest.approx(1.0)
    rt = settling_from_drag(c, 8.5, 0.015, exponent=m)
    assert float(torch.as_tensor(rt["relative_error"]).max()) < 1e-12


def test_drag_coefficient_suspension_reduces_to_single_particle():
    re = 12.0
    assert float(drag_coefficient_suspension(0.0, re, exponent=3.5)) == pytest.approx(
        float(drag_coefficient(re)), rel=1e-14
    )
    # monotone in concentration
    c = torch.linspace(0.0, 0.4, 9)
    cd = drag_coefficient_suspension(c, re, exponent=3.5)
    assert bool((cd[1:] > cd[:-1]).all())


def test_difelice_amplification_comparison_is_weaker_than_rz():
    single = single_particle_settling_velocity()
    re = single["reynolds"]
    m = float(richardson_zaki_exponent(re))
    c = 0.3
    rz = float(hindered_drag_amplification(c, exponent=m))
    df = float(difelice_drag_amplification(c, re))
    assert df > 1.0
    # documented model spread: RZ with c_max = 0.6 is far stronger than Di Felice
    assert rz > 10.0 * df


# ---------------------------------------------------------------------------
# Concentration-coupled rheology / diffusivity / flow capacity
# ---------------------------------------------------------------------------


def test_effective_mixture_viscosity_clear_water_limit():
    for g in (0.1, 1.0, 100.0):
        out = effective_mixture_viscosity(0.0, g)
        assert float(torch.as_tensor(out["mu_eff"]).reshape(-1)[0]) == pytest.approx(
            MU_WATER, rel=1e-12
        )
        assert float(torch.as_tensor(out["viscosity_ratio"]).reshape(-1)[0]) == pytest.approx(
            1.0, rel=1e-12
        )
        assert float(torch.as_tensor(out["flow_index"]).reshape(-1)[0]) == pytest.approx(1.0)


def test_effective_mixture_viscosity_is_shear_thinning():
    phi = 0.4
    g = torch.tensor([0.1, 1.0, 10.0, 100.0], dtype=torch.float64)
    out = effective_mixture_viscosity(phi, g)
    mu = out["mu_eff"]
    assert bool((mu[1:] < mu[:-1]).all())  # viscosity falls with shear rate
    n = float(torch.as_tensor(out["flow_index"]).reshape(-1)[0])
    # power-law slope in log-log space is exactly n-1
    slope = float(torch.log(mu[1] / mu[0]) / torch.log(g[1] / g[0]))
    assert slope == pytest.approx(n - 1.0, abs=1e-7)  # log-ratio round-off
    # and the KD value is reproduced at gamma_dot = 1
    mu_r = float(relative_viscosity_krieger_dougherty(phi))
    assert float(mu[1]) == pytest.approx(MU_WATER * mu_r, rel=1e-12)


def test_effective_diffusivity_clear_water_and_monotonicity():
    clear = effective_diffusivity(0.0, 10.0)
    assert float(torch.as_tensor(clear["D_eff"]).reshape(-1)[0]) == pytest.approx(1e-9, rel=1e-12)
    assert float(torch.as_tensor(clear["shear_induced"]).reshape(-1)[0]) == pytest.approx(0.0)
    c = torch.linspace(0.0, 0.4, 9)
    d = effective_diffusivity(c, 10.0)["D_eff"]
    assert bool((d[1:] > d[:-1]).all())


def test_effective_diffusivity_guards():
    with pytest.raises(FrazilRheologyError):
        effective_diffusivity(0.1, 1.0, molecular=0.0)
    with pytest.raises(FrazilRheologyError):
        effective_diffusivity(0.1, 1.0, diameter=-1.0)
    with pytest.raises(FrazilRheologyError):
        effective_diffusivity(DEFAULT_PHI_MAX, 1.0)


def test_flow_capacity_ratio_matches_direct_poiseuille_evaluation():
    a, Hh, mu_w = 1.2e-5, 10.0, 0.02
    for phi in (0.0, 0.2, 0.4):
        n = float(frazil_flow_index(phi))
        K = float(frazil_consistency_index(phi, mu_water=mu_w))
        u_max = n / (n + 1.0) * (a / K) ** (1.0 / n) * Hh ** ((n + 1.0) / n)
        q = u_max * (n + 1.0) / (2.0 * n + 1.0)
        q0 = 0.5 * a / mu_w * Hh**2 * (2.0 / 3.0)
        assert float(flow_capacity_ratio(phi)) == pytest.approx(q / q0, rel=1e-12)
    assert float(flow_capacity_ratio(0.0)) == pytest.approx(1.0, rel=1e-14)
    assert float(flow_capacity_ratio(0.4)) < 0.02


def test_batch_settling_interface_is_linear():
    out = batch_settling_interface(0.2, 0.0, v_s0=0.02, exponent=3.5, height_0=1.0)
    assert out["height"] == pytest.approx(1.0)
    v = out["velocity"]
    out2 = batch_settling_interface(0.2, 10.0, v_s0=0.02, exponent=3.5, height_0=1.0)
    assert out2["height"] == pytest.approx(1.0 - v * 10.0)
    with pytest.raises(FrazilRheologyError):
        batch_settling_interface(0.2, -1.0, v_s0=0.02, exponent=3.5)


def test_concentration_sweep_table_is_self_consistent():
    sweep = concentration_sweep((0.0, 0.2, 0.4), (1.0,))
    assert len(sweep["rows"]) == 3
    first = sweep["rows"][0]
    assert first["settling_velocity_ratio"] == pytest.approx(1.0, rel=1e-14)
    assert first["drag_amplification"] == pytest.approx(1.0, rel=1e-14)
    assert first["flow_capacity_ratio"] == pytest.approx(1.0, rel=1e-14)
    ratios = [r["settling_velocity_ratio"] for r in sweep["rows"]]
    assert ratios[0] > ratios[1] > ratios[2]
    assert sweep["rows"][2]["flow_index"] == pytest.approx(
        float(frazil_flow_index(0.4)), rel=1e-14
    )


# ---------------------------------------------------------------------------
# Lattice-Boltzmann side
# ---------------------------------------------------------------------------


def test_powerlaw_tau_field_newtonian_limit():
    ny, nx = 6, 2
    rho = torch.ones((ny, nx), dtype=torch.float64)
    zero = torch.zeros((ny, nx), dtype=torch.float64)
    f = torch.zeros((9, ny, nx), dtype=torch.float64)
    # equilibrium with zero velocity -> B = 0 -> viscosity clipped at nu_max
    from tensorlbm.d2q9 import equilibrium

    f = equilibrium(rho, zero, zero)
    K = torch.full((ny, nx), 0.02, dtype=torch.float64)
    tau = powerlaw_tau_field(f, K, torch.ones((ny, nx), dtype=torch.float64))
    # Newtonian: tau = 1/2 + 3K irrespective of the (zero) stress
    assert float(tau.max()) == pytest.approx(0.5 + 3.0 * 0.02, rel=1e-10)


def test_powerlaw_tau_field_guards():
    ny, nx = 4, 1
    f = torch.zeros((9, ny, nx), dtype=torch.float64)
    K = torch.full((ny, nx), 0.02, dtype=torch.float64)
    with pytest.raises(FrazilRheologyError):
        powerlaw_tau_field(f, K, torch.full((ny, nx), 1.5, dtype=torch.float64))
    with pytest.raises(FrazilRheologyError):
        powerlaw_tau_field(f, -K, torch.ones((ny, nx), dtype=torch.float64))


def test_stratified_reference_reproduces_poiseuille_closed_form():
    H = 20
    for n in (1.0, 0.7, 0.5):
        K = 0.0115
        a = 3.6e-5
        ref = stratified_powerlaw_reference(
            torch.full((H,), K), torch.full((H,), n), a, H
        )
        u_exact, _ = poiseuille_analytic_profile(H, n, K, a)
        err = _profile_error(ref["u"], u_exact)
        assert err["l2_relative"] < 1e-6, (n, err)
        assert abs(ref["u_top_wall"]) < 1e-12  # symmetry -> wall condition exact
    with pytest.raises(FrazilRheologyError):
        stratified_powerlaw_reference(torch.full((5,), 0.01), torch.full((5,), 0.7), 1e-5, 20)


def test_scalar_advection_diffusion_matches_exact_travelling_wave():
    run = run_advection_diffusion_check(u=0.05, D=0.01, nx=32, ny=2, steps=200)
    assert run["l2_relative"] < 0.01
    assert run["advection_speed_error"] < 0.01
    assert run["diffusivity_relative_error"] < 0.05


def test_settling_column_interface_matches_richardson_zaki():
    run = run_settling_column(
        0.2, v_s0=0.08, exponent=3.5, H=48, nx=1, max_steps=4000, check_every=10,
        travel_cells=4.0,
    )
    assert run["relative_error"] < 0.03
    assert run["mass_drift"] < 1e-12
    assert run["steps"] < 4000


def test_settling_column_guards():
    with pytest.raises(FrazilRheologyError):
        run_settling_column(0.2, v_s0=0.08, exponent=3.5, D=0.0, H=16, nx=1, max_steps=10)
    with pytest.raises(FrazilRheologyError):
        run_settling_column(DEFAULT_PHI_MAX, v_s0=0.08, exponent=3.5, H=16, nx=1, max_steps=10)


def test_uniform_powerlaw_channel_matches_exact_profile():
    H = 12
    n, K, a = 0.7, 0.0115, 3.6e-5
    u_exact, _ = poiseuille_analytic_profile(H, n, K, a)
    run = run_powerlaw_channel(
        consistency_index=K,
        flow_index=n,
        ax=a,
        H=H,
        nx=1,
        max_steps=4000,
        check_every=50,
        window=10,
        tol=1e-4,
        avg_steps=100,
        initial_profile=u_exact,
        nu_max=0.5,
    )
    err = _profile_error(run["profile"], u_exact)
    assert err["l2_relative"] < 0.03
    assert run["converged"]


def test_channel_rejects_bad_profiles():
    with pytest.raises(FrazilRheologyError):
        run_powerlaw_channel(
            consistency_index=torch.ones(3) * 0.02,
            flow_index=0.7,
            ax=1e-5,
            H=12,
            nx=1,
            max_steps=5,
        )


def test_interface_height_interpolates():
    profile = torch.tensor([0.4, 0.4, 0.4, 0.2, 0.0, 0.0], dtype=torch.float64)
    # level 0.3 sits half way between index 2 (0.4) and index 3 (0.2)
    assert _interface_height(profile, 0.3) == pytest.approx(2.5 + 0.5)
    assert math.isnan(_interface_height(torch.zeros(5, dtype=torch.float64), 0.1))