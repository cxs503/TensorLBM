"""Tests for the gas-layer / microbubble drag-reduction closure (IN39-N06).

The suite grades the machinery that the campaign relies on, not the campaign
itself (that is the artifact): analytic references, the net-power ledger's
monotonicity, the sha256-anchored literature envelope, mass conservation of the
kernel/wall/source combination, and a short end-to-end single-phase and
two-phase run.  All cases use tiny grids and a few hundred steps so the file runs
in seconds.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.gas_layer import economics, kernels, literature, reference
from tensorlbm.gas_layer.laplace import LaplaceConfig, run_laplace_gate
from tensorlbm.gas_layer.single_phase import SinglePhaseChannel2D, run_single_phase_gate
from tensorlbm.gas_layer.two_phase import TwoPhaseConfig, run_case

# ---------------------------------------------------------------------------
# analytic references
# ---------------------------------------------------------------------------


def test_laminar_channel_identities() -> None:
    """Cf = 12/Re_h, tau_w = rho g h/2 and the u+ law are self-consistent."""
    g, nu, h, rho = 1.0e-4, 0.1, 40.0, 1.0
    U = reference.laminar_mean_velocity(g, nu, h)
    tau = reference.laminar_wall_shear(g, rho, h)
    # momentum balance: tau_w = 6 mu U / h  (equivalently Cf = 12/Re_h)
    assert math.isclose(tau, 6.0 * nu * rho * U / h, rel_tol=1e-12)
    cf_direct = reference.friction_coefficient(U, h, nu, rho)
    cf_from_shear = reference.friction_coefficient_from_shear(tau, U, rho)
    assert math.isclose(cf_direct, cf_from_shear, rel_tol=1e-12)
    # exact laminar law is quadratic and matches the profile at the centre
    y = torch.tensor([0.0, 0.5 * h, h], dtype=torch.float64)
    u_tau = reference.utau(tau, rho)
    re_tau = reference.re_tau(u_tau, 0.5 * h, nu)
    u_plus = reference.laminar_u_plus(y * u_tau / nu, re_tau)
    assert math.isclose(
        float(u_plus[1]) * u_tau,
        reference.laminar_profile(g, nu, h, 0.5 * h),
        rel_tol=1e-12,
    )
    # Cf is monotone decreasing in Reynolds number
    assert reference.friction_coefficient(2 * U, h, nu) < cf_direct


def test_ittc_friction_line_decreases_with_length() -> None:
    cf1 = reference.skin_friction_ittc57(1.0e6)
    cf2 = reference.skin_friction_ittc57(1.0e8)
    assert cf1 > cf2 > 0.0


# ---------------------------------------------------------------------------
# net-power ledger
# ---------------------------------------------------------------------------


def test_compressor_work_and_net_power_signs() -> None:
    w = economics.compressor_specific_work()
    assert w > 0.0
    tau = economics.wall_shear_from_friction(10.0, 100.0)
    # a gas layer that removes nothing cannot pay for its own compressor
    assert economics.net_power_density(0.0, 10.0, tau, 0.01) < 0.0
    # more drag reduction -> more net power, more gas -> less net power
    a = economics.net_power_density(0.3, 10.0, tau, 0.01)
    b = economics.net_power_density(0.4, 10.0, tau, 0.01)
    c = economics.net_power_density(0.3, 10.0, tau, 0.05)
    assert b > a and c < a


def test_net_power_monotone_in_compressor_efficiency() -> None:
    """The sign-error guard: better efficiency can only enlarge the profit domain."""
    tau = economics.wall_shear_from_friction(10.0, 100.0)
    vals = [economics.net_power_density(0.3, 10.0, tau, 0.02, eta_c=e) for e in (0.4, 0.6, 0.8)]
    assert vals[0] < vals[1] < vals[2]
    model = lambda c: 0.30 * float(c > 0.0)  # noqa: E731 - tiny closure
    sens = economics.efficiency_sensitivity(
        model, [0.002, 0.01, 0.05], [4.0, 8.0, 12.0], (0.5, 0.6, 0.7)
    )
    assert sens["monotone_in_eta_c"] is True
    areas = sens["positive_area_fraction"]
    assert areas[0] <= areas[1] <= areas[2]


def test_break_even_efficiency_consistency() -> None:
    w = economics.compressor_specific_work()
    tau = 20.0
    dr, C_q = 0.25, 0.01
    eta_be = economics.required_efficiency(dr, tau, C_q, w)
    assert eta_be > 0.0
    # exactly at break-even the net power vanishes
    net = economics.net_power_density(dr, 10.0, tau, C_q, eta_c=min(eta_be, 1.0), w_c=w)
    assert abs(net) < 1e-9 or eta_be > 1.0


def test_positive_domain_shrinks_for_lower_efficiency() -> None:
    model = lambda c: 0.3  # noqa: E731
    good = economics.positive_net_benefit_boundary(model, [0.002, 0.02, 0.2], [2.0, 6.0, 12.0],
                                                   eta_c=0.7)
    bad = economics.positive_net_benefit_boundary(model, [0.002, 0.02, 0.2], [2.0, 6.0, 12.0],
                                                  eta_c=0.5)
    assert bad["positive_area_fraction"] <= good["positive_area_fraction"]


# ---------------------------------------------------------------------------
# literature envelope
# ---------------------------------------------------------------------------


def test_reference_table_hash_is_stable() -> None:
    assert literature.REFERENCE_TABLE_SHA256 == literature.table_sha256()
    assert len(literature.REFERENCE_TABLE_SHA256) == 64


def test_envelope_verdicts() -> None:
    inside = literature.literature_envelope_check(0.03, 0.30)
    assert inside["verdict"] == "inside_band"
    assert inside["best_entry"]
    outside = literature.literature_envelope_check(0.03, 0.95)
    assert outside["verdict"] == "outside_band"
    missing = literature.literature_envelope_check(10.0, 0.30)
    assert missing["verdict"] == "no_matching_cq"


# ---------------------------------------------------------------------------
# kernels: conservation identities
# ---------------------------------------------------------------------------


def test_streaming_conserves_mass_with_walls() -> None:
    """Half-way bounce-back must conserve mass exactly (no leaking wall rows)."""
    ny, nx = 12, 8
    solid = torch.zeros((ny, nx), dtype=torch.bool)
    solid[0, :] = True
    solid[-1, :] = True
    rho = torch.full((ny, nx), 1.5, dtype=torch.float64)
    u = torch.zeros_like(rho)
    f = kernels.equilibrium(rho, u, u)
    f[:, solid] = 0.0
    mass0 = float(f.sum())
    out = kernels.stream(f, solid=solid)
    assert math.isclose(float(out["f"].sum()), mass0, rel_tol=1e-14)
    # reflected populations must be non-zero at the wall-adjacent rows
    assert float(out["f"][:, :, 1].sum()) > 0.0
    assert abs(float(out["wall_force_x"].sum())) < 1e-12


def test_lattice_weights_sum_to_one() -> None:
    """The float64 weights must sum to 1 exactly, otherwise mass leaks per step."""
    assert math.isclose(float(kernels._W.sum()), 1.0, rel_tol=1e-15)


def test_equilibrium_preserves_density_and_momentum() -> None:
    torch.manual_seed(0)
    rho = 1.0 + 0.1 * torch.rand(8, 6, dtype=torch.float64)
    ux = 0.05 * torch.rand(8, 6, dtype=torch.float64)
    uy = -0.03 * torch.rand(8, 6, dtype=torch.float64)
    f = kernels.equilibrium(rho, ux, uy)
    r, mx, my = kernels.macroscopic(f)
    assert torch.allclose(r, rho, rtol=1e-13, atol=1e-14)
    assert torch.allclose(mx, ux, rtol=1e-12, atol=1e-13)
    assert torch.allclose(my, uy, rtol=1e-12, atol=1e-13)


# ---------------------------------------------------------------------------
# single-phase gate
# ---------------------------------------------------------------------------


def test_single_phase_channel_reproduces_analytic_shear() -> None:
    """Cf from momentum exchange must match 12/Re_h to better than 3 %."""
    out = run_single_phase_gate(re_tau_target=30.0, ny=98, nx=2, n_steps=500, n_warmup=150)
    assert out["cf_rel_error"] <= 0.03
    assert out["profile_rel_error"] <= 0.03
    assert out["gate"]["cf_rel_error_le_3pct"] is True


def test_single_phase_channel_geometry_and_validation() -> None:
    ch = SinglePhaseChannel2D(ny=20, nx=3, nu=0.1, g=1e-4)
    assert ch.height == pytest.approx(19.0)
    assert ch.fluid_rows == (1, 18)
    with pytest.raises(ValueError):
        SinglePhaseChannel2D(ny=4, nx=2, nu=0.1, g=1e-4)
    with pytest.raises(ValueError):
        SinglePhaseChannel2D(ny=20, nx=2, nu=1e-6, g=1e-4)


# ---------------------------------------------------------------------------
# two-phase channel
# ---------------------------------------------------------------------------


def test_closed_two_phase_channel_conserves_mass() -> None:
    """No source, no outlet: the total mass may not drift."""
    cfg = TwoPhaseConfig(
        ny=20, nx=24, U_target=0.04, n_steps=120, sample_every=40,
        initial_layer_thickness=0.0, wall_psi=0.3 * 3.9,
    )
    rec = run_case(cfg)
    assert abs(rec["mass_relative_drift"]) <= 1e-6
    assert abs(rec["ledger_relative_error"]) <= 1e-6
    assert math.isfinite(rec["wall_shear"])
    assert math.isfinite(rec["U_measured"])


def test_gas_source_ledger_closes() -> None:
    """Injected gas mass is booked and the total-inventory ledger closes."""
    cfg = TwoPhaseConfig(
        ny=20,
        nx=24,
        U_target=0.04,
        n_steps=100,
        sample_every=40,
        source=True,
        source_rate=0.2,
        source_x_fraction=(0.1, 0.25),
        wall_psi=0.3 * 3.9,
    )
    rec = run_case(cfg)
    assert rec["mass_injected"] > 0.0
    assert abs(rec["ledger_relative_error"]) <= 1e-6
    # the booked injection must explain the inventory change to within 1e-6
    assert abs(rec["mass_relative_drift"]) <= max(1e-6, 10.0 * rec["mass_injected"]
                                                  / rec["mass_initial"])


def test_two_phase_config_validation() -> None:
    with pytest.raises(ValueError):
        TwoPhaseConfig(nx=8).validate()
    with pytest.raises(ValueError):
        TwoPhaseConfig(x_boundary="open", sponge_cells=0).validate()
    with pytest.raises(ValueError):
        TwoPhaseConfig(source=True, source_rate=0.0).validate()
    with pytest.raises(ValueError):
        TwoPhaseConfig(eos_temperature=0.04, tau=0.5).validate()


def test_gas_fraction_and_coverage_definition() -> None:
    """Coverage is the near-wall gas volume fraction, 0 for pure liquid, 1 for gas."""
    cfg = TwoPhaseConfig(ny=16, nx=16, n_steps=1)
    sim = __import__("tensorlbm.gas_layer.two_phase", fromlist=["GasLayerChannel2D"])
    ch = sim.GasLayerChannel2D(cfg)
    rho_l = torch.full((16, 16), cfg.rho_liquid, dtype=torch.float64)
    rho_g = torch.full((16, 16), cfg.rho_gas, dtype=torch.float64)
    assert ch.coverage(rho_l) == pytest.approx(0.0, abs=1e-12)
    assert ch.coverage(rho_g) == pytest.approx(1.0, abs=1e-12)
    mixed = rho_l.clone()
    mixed[1, :6] = cfg.rho_gas
    assert ch.coverage(mixed) == pytest.approx(6.0 / 16.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Laplace gate (short run: convergence is the campaign's job, finiteness here)
# ---------------------------------------------------------------------------


def test_laplace_gate_returns_positive_pressure_jump() -> None:
    cfg = LaplaceConfig(nx=40, ny=40, radii=(7.0, 9.0), n_steps=250, init_width=4.0)
    out = run_laplace_gate(cfg)
    assert out["sigma_eff"] > 0.0
    assert out["min_density_ratio"] > 1.0
    for rec in out["per_radius"]:
        assert rec["delta_p"] > 0.0
        assert math.isfinite(rec["interface_width"])

# ---------------------------------------------------------------------------
# design-point API
# ---------------------------------------------------------------------------


def test_design_point_closure_extrapolation_bounds() -> None:
    """Plateau and linear sub-range closures bracket the profitable domain."""
    from tensorlbm.gas_layer import api

    plateau = api.load_closure(extrapolation="plateau")
    linear = api.load_closure(extrapolation="linear")
    assert plateau["measured_C_q_range"][0] > 0.0
    # far below the measured range the two extrapolations must disagree
    hi = api.design_point(10.0, 2.0e-4, closure=plateau)
    lo = api.design_point(10.0, 2.0e-4, closure=linear)
    assert hi["drag_reduction"] > lo["drag_reduction"]
    assert hi["inside_measured_C_q_range"] is False
    # inside the measured band both closures agree
    a = api.design_point(10.0, 0.05, closure=plateau)
    b = api.design_point(10.0, 0.05, closure=linear)
    assert a["drag_reduction"] == pytest.approx(b["drag_reduction"])
    assert a["inside_measured_C_q_range"] is True


def test_design_point_rejects_bad_input() -> None:
    from tensorlbm.gas_layer import api

    with pytest.raises(ValueError):
        api.design_point(10.0, -1.0)
    with pytest.raises(ValueError):
        api.design_point(0.0, 0.01)
    with pytest.raises(ValueError):
        api.load_closure(extrapolation="quadratic")
