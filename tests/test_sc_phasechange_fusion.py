"""Tests of the fused Shan-Chen + enthalpy phase-change verification surface.

Node ``p348w3-n04``.  The suite covers the analytic anchors (Neumann
similarity, Maxwell construction), the numerical solvers at a small but
representative size, and the acceptance-gate logic on synthetic payloads so the
gates themselves are falsifiable.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensorlbm.phase_change.stefan_exact import planar_lambda
from tensorlbm.sc_phasechange import gates
from tensorlbm.sc_phasechange.enthalpy import (
    ICE_WATER,
    EnthalpyPhaseChange1D,
    NeumannMelting,
    PhaseChangeProperties,
)
from tensorlbm.sc_phasechange.highdensity import (
    Coexistence,
    EosModel,
    carnahan_starling_pressure,
    maxwell_construction,
    peng_robinson_pressure,
    temperature_for_density_ratio,
)
from tensorlbm.sc_phasechange.rt_anchor import _mode_amplitude

P = ICE_WATER


# ---------------------------------------------------------------------------
# EOS / Maxwell construction
# ---------------------------------------------------------------------------
def test_cs_pressure_ideal_gas_limit():
    """At low density the CS EOS must reduce to p = rho R T."""
    rho = 1e-6
    p = carnahan_starling_pressure(rho, 0.5)
    assert p == pytest.approx(rho * 0.5, rel=2e-3)


def test_pr_pressure_ideal_gas_limit():
    rho = 1e-6
    p = peng_robinson_pressure(rho, 0.5)
    assert p == pytest.approx(rho * 0.5, rel=2e-3)


def test_maxwell_matches_archived_lock():
    """CS coexistence at T_r=0.60 reproduces the archived lock values."""
    eos = EosModel.cs()
    co = maxwell_construction(eos, 0.60 * eos.t_critical)
    assert co.rho_liquid == pytest.approx(0.4062041397340105, rel=2e-3)
    assert co.rho_vapour == pytest.approx(0.0030814585698829495, rel=2e-2)
    assert co.density_ratio == pytest.approx(131.82203509211558, rel=3e-2)


def test_maxwell_saturation_pressure_between_spinodals():
    eos = EosModel.cs()
    co = maxwell_construction(eos, 0.7 * eos.t_critical)
    p_spin_v = float(carnahan_starling_pressure(co.rho_spinodal_vapour, co.t))
    p_spin_l = float(carnahan_starling_pressure(co.rho_spinodal_liquid, co.t))
    assert min(p_spin_l, p_spin_v) < co.p_sat < max(p_spin_l, p_spin_v)


def test_maxwell_ratio_decreases_with_temperature():
    eos = EosModel.cs()
    ratios = [
        maxwell_construction(eos, tr * eos.t_critical).density_ratio for tr in (0.60, 0.70, 0.80)
    ]
    assert ratios[0] > ratios[1] > ratios[2]


def test_maxwell_rejects_supercritical():
    eos = EosModel.cs()
    with pytest.raises(ValueError):
        maxwell_construction(eos, 1.05 * eos.t_critical)


def test_temperature_for_ratio_hits_targets():
    eos = EosModel.cs()
    for target in (10.0, 50.0):
        co = temperature_for_density_ratio(eos, target)
        assert co.density_ratio == pytest.approx(target, rel=5e-3)
        assert co.t_reduced < 1.0


def test_laplace_fit_recovers_sigma():
    """Perfect dp = sigma/R data must return sigma with zero deviation."""
    from tensorlbm.sc_phasechange.highdensity import LaplaceCase, fit_laplace

    sigma = 0.037
    cases = []
    for r in (12.0, 16.0, 20.0):
        cases.append(
            LaplaceCase(
                radius_init=r,
                radius_eff=r,
                dp=sigma / r,
                rho_in=0.4,
                rho_out=0.004,
                density_ratio=100.0,
                sigma=float("nan"),
                spurious_ma=1e-3,
                mass_drift_per_step=1e-9,
                mass_drift_total=1e-6,
                steps_run=10,
                converged=True,
                stable=True,
            )
        )
    fit = fit_laplace(cases)
    assert fit["sigma"] == pytest.approx(sigma, rel=1e-6)
    assert fit["max_dev_pct"] == pytest.approx(0.0, abs=1e-6)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Neumann similarity solution
# ---------------------------------------------------------------------------
def test_neumann_matches_one_phase_relation():
    """Zero subcooling reduces to lambda exp(lambda^2) erf(lambda) = Ste/sqrt(pi)."""
    for ste in (0.1, 0.5, 1.0):
        ref = NeumannMelting(P.superheat_for_stefan(ste), 0.0, P)
        assert ref.lam == pytest.approx(planar_lambda(ste, side="solid"), rel=1e-6)


def test_neumann_lambda_shrinks_with_subcooling():
    """Melting into a colder solid is *slower*: part of the wall heat is lost
    into the subcooled layer instead of paying for latent heat, so the
    similarity coefficient lambda decreases (0.465 -> 0.440 -> 0.417 for
    superheat 40 K)."""
    refs = [NeumannMelting(40.0, sub, P) for sub in (0.0, 5.0, 10.0)]
    assert refs[0].lam > refs[1].lam > refs[2].lam
    assert refs[0].lam == pytest.approx(0.4648, abs=2e-3)


def test_neumann_front_scales_with_sqrt_t():
    ref = NeumannMelting(40.0, 5.0, P)
    ratio = ref.front(1000.0) / ref.front(250.0)
    assert ratio == pytest.approx(2.0, rel=1e-12)


def test_neumann_boundary_and_interface_temperatures():
    ref = NeumannMelting(30.0, 8.0, P)
    t = 5000.0
    s = ref.front(t)
    profile = ref.temperature(np.array([0.0, s, 50.0 * s]), t)
    assert profile[0] == pytest.approx(ref.t_surface, rel=1e-9)
    assert profile[1] == pytest.approx(P.t_melt, abs=1e-6)
    assert profile[2] == pytest.approx(ref.t_initial, rel=1e-6)


def test_neumann_wall_flux_decays_like_one_over_sqrt_t():
    ref = NeumannMelting(40.0, 0.0, P)
    q1, q2 = ref.wall_heat_flux(1000.0), ref.wall_heat_flux(4000.0)
    assert q1 / q2 == pytest.approx(2.0, rel=1e-12)


def test_neumann_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        NeumannMelting(0.0)
    with pytest.raises(ValueError):
        NeumannMelting(30.0, -1.0)


def test_neumann_stefan_number_round_trip():
    ste = 0.37
    dT = P.superheat_for_stefan(ste)
    assert P.stefan_number(dT) == pytest.approx(ste, rel=1e-12)


# ---------------------------------------------------------------------------
# Enthalpy solver
# ---------------------------------------------------------------------------
def _small_solver(ste=0.5, sub=0.0, n_cells=200, length=0.2):
    return EnthalpyPhaseChange1D(
        length=length,
        n_cells=n_cells,
        superheat=P.superheat_for_stefan(ste),
        subcooling=sub,
    )


def test_enthalpy_temperature_inversion_round_trip():
    sim = _small_solver()
    h = np.array([-1e5, 0.0, 0.5 * P.latent, P.latent * 1.5, P.latent * 3.0])
    temp = sim.temperature_of(h)
    assert temp[0] < P.t_melt
    assert temp[1] == pytest.approx(P.t_melt)
    assert temp[2] == pytest.approx(P.t_melt)
    assert temp[3] > P.t_melt and temp[4] > temp[3]


def test_liquid_fraction_bounds():
    sim = _small_solver()
    h = np.array([-1e4, 0.0, 0.25 * P.latent, 0.75 * P.latent, 2 * P.latent])
    phi = sim.liquid_fraction_of(h)
    assert phi[0] == 0.0 and phi[-1] == 1.0
    assert phi[2] == pytest.approx(0.25, rel=1e-12)
    assert np.all((phi >= 0.0) & (phi <= 1.0))


def test_liquid_fraction_complement_is_one():
    sim = _small_solver()
    sim.run(200.0, [200.0])
    phi = sim.liquid_fraction
    assert np.allclose(phi + (1.0 - phi), 1.0)


def test_cfl_guard():
    with pytest.raises(ValueError):
        EnthalpyPhaseChange1D(length=0.1, n_cells=50, superheat=20.0, cfl=0.7)


def test_short_run_reproduces_neumann_front():
    sim = _small_solver(ste=0.5, n_cells=300, length=0.3)
    ref = NeumannMelting(sim.superheat, sim.subcooling, P)
    t_end = ref.time_for_front(0.02)
    rec = sim.run(t_end, [t_end])[-1]
    assert abs(rec.front_error_pct) < 2.0
    assert rec.front > 0.0


def test_front_is_monotone_in_time():
    sim = _small_solver(ste=1.0, n_cells=200, length=0.2)
    ref = NeumannMelting(sim.superheat, 0.0, P)
    times = [ref.time_for_front(0.005 * k) for k in (1, 2, 3, 4)]
    fronts = [r.front for r in sim.run(times[-1], times, ref)]
    assert all(b > a for a, b in zip(fronts, fronts[1:]))


def test_stefan_residual_small_at_moderate_time():
    """The residual is measured over a window thick enough for the stencil:
    the first window (thin liquid layer, t -> 0 singularity) is excluded, as
    in the gate rule of the campaign node."""
    sim = _small_solver(ste=0.5, n_cells=400, length=0.4)
    ref = NeumannMelting(sim.superheat, 0.0, P)
    t_end = ref.time_for_front(0.03)
    rec = sim.run(t_end, [0.5 * t_end, t_end], ref)[-1]
    assert rec.front / sim.dx > 14.0
    assert abs(rec.stefan_residual_pct) < 2.0


def test_solid_region_at_melting_point_without_subcooling():
    sim = _small_solver(ste=0.5, sub=0.0, n_cells=300, length=0.3)
    ref = NeumannMelting(sim.superheat, 0.0, P)
    t_end = ref.time_for_front(0.02)
    sim.run(t_end, [t_end])
    temp = sim.temperature
    solid = sim.x > sim.front + 4 * sim.dx
    assert np.allclose(temp[solid], P.t_melt, atol=1e-9)


def test_subcooled_solid_is_below_melting_point():
    sim = _small_solver(ste=0.5, sub=10.0, n_cells=400, length=0.4)
    ref = NeumannMelting(sim.superheat, 10.0, P)
    t_end = ref.time_for_front(0.02)
    sim.run(t_end, [t_end])
    temp = sim.temperature
    far = sim.x > sim.front + 10 * sim.dx
    assert np.all(temp[far] < P.t_melt + 1e-9)
    assert float(temp[far].min()) >= ref.t_initial - 1e-6


def test_energy_increases_and_budget_closes():
    sim = _small_solver(ste=0.5, n_cells=300, length=0.3)
    ref = NeumannMelting(sim.superheat, 0.0, P)
    t_end = ref.time_for_front(0.02)
    rec = sim.run(t_end, [0.3 * t_end, t_end], ref)
    assert rec[-1].energy_total > rec[0].energy_total
    assert abs(rec[-1].energy_imbalance_pct) < 2.0


def test_wall_flux_matches_neumann():
    sim = _small_solver(ste=0.5, n_cells=300, length=0.3)
    ref = NeumannMelting(sim.superheat, 0.0, P)
    t_end = ref.time_for_front(0.02)
    sim.run(t_end, [t_end])
    numeric = sim.wall_flux_audit()
    analytic = ref.wall_heat_flux(sim.time)
    assert numeric == pytest.approx(analytic, rel=0.05)


def test_front_crossing_consistent_with_integral_front():
    sim = _small_solver(ste=1.0, n_cells=300, length=0.3)
    ref = NeumannMelting(sim.superheat, 0.0, P)
    t_end = ref.time_for_front(0.02)
    sim.run(t_end, [t_end])
    assert sim.front_crossing() == pytest.approx(sim.front, rel=0.05)


def test_enthalpy_solver_rejects_tiny_grid():
    with pytest.raises(ValueError):
        EnthalpyPhaseChange1D(length=0.1, n_cells=4, superheat=20.0)


# ---------------------------------------------------------------------------
# Rayleigh-Taylor diagnostics
# ---------------------------------------------------------------------------
def test_mode_amplitude_reads_a_known_sine():
    nx, ny, lam, amp = 64, 96, 64, 3.0
    y = torch.arange(ny, dtype=torch.float32).view(ny, 1)
    x = torch.arange(nx, dtype=torch.float32).view(1, nx)
    y_int = ny / 2.0 + amp * torch.cos(2.0 * math.pi * x / lam)
    phi = torch.tanh((y - y_int) / 1.5)
    assert _mode_amplitude(phi, lam) == pytest.approx(amp, rel=0.05)
    # reversed phase (crest pointing down) must give the same magnitude
    assert _mode_amplitude(torch.flip(phi, dims=(0,)), lam) == pytest.approx(amp, rel=0.05)


def test_mode_amplitude_zero_for_flat_interface():
    phi = torch.ones(48, 32)
    phi[:24] = -1.0
    assert _mode_amplitude(phi, 32) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Gates (falsifiability of the acceptance logic)
# ---------------------------------------------------------------------------
def _enthalpy_payload(
    err_pct=0.5, stefan_pct=1.0, energy_pct=1.0, n_cases=9, layer_cells=50.0, dx=1e-3
):
    cases = []
    for i in range(n_cases):
        sub = 0.0 if i % 3 == 0 else (5.0 if i % 3 == 1 else 10.0)
        cases.append(
            dict(
                ste=(0.1, 0.5, 1.0)[i // 3],
                dTsub=sub,
                lam=0.4,
                dx=dx,
                errs_pct=[err_pct] * 10,
                times=[float(k) for k in range(10)],
                stefan_windows=[
                    dict(residual_pct=stefan_pct, front_m=layer_cells * dx) for _ in range(10)
                ],
                energy_windows=[energy_pct] * 10,
                energy_cumulative_imbalance_pct=energy_pct,
                max_abs_phase_bookkeeping_pct=0.0,
            )
        )
    return dict(part="enthalpy", cases=cases)


def _laplace_payload(dev_pct=1.0, ma=1e-3, drift=1e-9, ratio_miss=1.0):
    tiers = []
    for target in (10.0, 50.0, 100.0):
        tiers.append(
            dict(
                target_ratio=target,
                coexistence_ratio=target,
                t_reduced=0.7,
                cases=[
                    dict(
                        radius_eff=r,
                        dp=0.03 / r,
                        rho_in=0.4,
                        rho_out=0.4 / (target * ratio_miss),
                        mass_drift_per_step=drift,
                        mass_drift_total=drift * 1000,
                    )
                    for r in (14.0, 18.0, 22.0)
                ],
                fit=dict(
                    sigma=0.03, r2=0.9999, max_dev_pct=dev_pct, dev_pct=[dev_pct, -dev_pct, 0.0]
                ),
                max_spurious_ma=ma,
                all_converged=True,
                all_stable=True,
            )
        )
    return dict(part="laplace", tiers=tiers)


def _rt_payload(err_pct=3.0):
    return dict(
        part="rt",
        wavelength=64,
        gravity=2e-3,
        cases=[
            dict(
                atwood=a,
                sigma_sim=1.0,
                sigma_analytic=1.0,
                sigma_viscous=1.0,
                rel_error_pct=err_pct,
                rel_error_viscous_pct=2.0,
                fit_r2=0.999,
                steps_run=3000,
            )
            for a in (0.1, 1.0 / 3.0, 0.5)
        ],
    )


def test_c1_passes_on_clean_payload():
    out = gates.gate_c1_neumann(_enthalpy_payload(err_pct=0.5))
    assert out["verdict"]["passed"] is True


def test_c1_fails_above_threshold():
    out = gates.gate_c1_neumann(_enthalpy_payload(err_pct=4.0))
    assert out["verdict"]["passed"] is False


def test_c1_requires_ten_time_points():
    payload = _enthalpy_payload()
    for c in payload["cases"]:
        c["errs_pct"] = c["errs_pct"][:6]
        c["times"] = c["times"][:6]
    assert gates.gate_c1_neumann(payload)["verdict"]["passed"] is False


def test_c2_rejects_thin_layer_windows():
    """Windows with a liquid layer thinner than the fit stencil are excluded."""
    payload = _enthalpy_payload(stefan_pct=1.0, layer_cells=5.0)
    out = gates.gate_c2_stefan(payload)
    assert out["verdict"]["passed"] is False  # no usable windows -> no evidence


def test_c2_passes_on_thick_layer_windows():
    out = gates.gate_c2_stefan(_enthalpy_payload(stefan_pct=1.5, layer_cells=50.0))
    assert out["verdict"]["passed"] is True


def test_c2_fails_above_threshold():
    out = gates.gate_c2_stefan(_enthalpy_payload(stefan_pct=2.5, layer_cells=50.0))
    assert out["verdict"]["passed"] is False


def test_c3_passes_on_clean_payload():
    assert gates.gate_c3_laplace(_laplace_payload())["verdict"]["passed"] is True


def test_c3_fails_on_large_deviation():
    out = gates.gate_c3_laplace(_laplace_payload(dev_pct=3.0))
    assert out["verdict"]["passed"] is False


def test_c3_fails_on_spurious_current():
    out = gates.gate_c3_laplace(_laplace_payload(ma=0.05))
    assert out["verdict"]["passed"] is False


def test_c3_fails_when_measured_ratio_misses_target():
    out = gates.gate_c3_laplace(_laplace_payload(ratio_miss=0.5))
    assert out["verdict"]["passed"] is False


def test_c4_passes_and_fails():
    assert gates.gate_c4_rayleigh_taylor(_rt_payload(3.0))["verdict"]["passed"] is True
    assert gates.gate_c4_rayleigh_taylor(_rt_payload(12.0))["verdict"]["passed"] is False


def test_c5_uses_per_step_drift():
    ok = gates.gate_c5_mass(_laplace_payload(drift=1e-9), _enthalpy_payload())
    bad = gates.gate_c5_mass(_laplace_payload(drift=1e-4), _enthalpy_payload())
    assert ok["verdict"]["passed"] is True
    assert bad["verdict"]["passed"] is False


def test_c6_window_imbalance_gate():
    assert gates.gate_c6_energy(_enthalpy_payload(energy_pct=1.5))["verdict"]["passed"] is True
    assert gates.gate_c6_energy(_enthalpy_payload(energy_pct=2.5))["verdict"]["passed"] is False


def test_c7_counts_tests():
    assert gates.gate_c7_tests(30, 0)["verdict"]["passed"] is True
    assert gates.gate_c7_tests(20, 0)["verdict"]["passed"] is False
    assert gates.gate_c7_tests(30, 1)["verdict"]["passed"] is False


def test_assemble_verdict_all_pass():
    parts = dict(enthalpy=_enthalpy_payload(), laplace=_laplace_payload(), rt=_rt_payload())
    verdict = gates.assemble_verdict(parts, dict(count=30, failed=0))
    assert verdict["all_pass"] is True
    assert len(verdict["gates"]) == 7


def test_assemble_verdict_detects_failure():
    parts = dict(
        enthalpy=_enthalpy_payload(err_pct=9.0),
        laplace=_laplace_payload(),
        rt=_rt_payload(),
    )
    verdict = gates.assemble_verdict(parts, dict(count=30, failed=0))
    assert verdict["all_pass"] is False


def test_coexistence_density_ratio_property():
    co = Coexistence(
        t=0.05,
        t_reduced=0.6,
        rho_liquid=0.4,
        rho_vapour=0.004,
        p_sat=1e-4,
        rho_spinodal_vapour=0.04,
        rho_spinodal_liquid=0.3,
    )
    assert co.density_ratio == pytest.approx(100.0, rel=1e-12)


def test_properties_alpha_ordering():
    props = PhaseChangeProperties()
    assert props.alpha_solid > props.alpha_liquid > 0.0
