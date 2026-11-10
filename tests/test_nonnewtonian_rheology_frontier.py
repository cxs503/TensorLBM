"""Tests for the non-Newtonian rheology verification surface (P40W4-N04).

Coverage
--------
* closures: power law / Carreau–Yasuda / Cross / Herschel–Bulkley against
  independently written formulas (not the implementation under test),
* the 1-D anchor: constitutive inversion, quadrature against the closed-form
  Ostwald–de Waele profile (machine precision), plug geometry ``τ_y/(ρa)``,
* the lattice solver: Newtonian control case, power-law channel vs closed form,
  shear-thickening case, mass conservation of the Guo forcing (regression for
  the forcing bug fixed in this node), plug detectors on a H-B channel,
  Carreau–Yasuda viscosity-field consistency,
* runtime guards (invalid input), and the benchmark artifact contract.

All simulations here are deliberately small (h ≤ 24, ≤ 6000 steps) so the file
runs in well under a minute on one core; the production numbers live in
``benchmarks/p40w4_n04_nonnewtonian_rheology/result.json``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(REPO / "src"))

from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.nonnewtonian_frontier import (  # noqa: E402
    carreau_yasuda,
    channel_geometry,
    channel_reference,
    cross,
    herschel_bulkley,
    l2_relative_error,
    newtonian,
    plug_half_width_fit,
    plug_half_width_ideal_from_closure,
    plug_half_width_stress,
    plug_half_width_velocity,
    plug_report,
    power_law,
    run_channel,
    shear_rate_profile,
    symmetrise,
    wall_nonequilibrium_2d,
)
from tensorlbm.nonnewtonian_frontier.reference import (  # noqa: E402
    gamma_of_tau,
    power_law_velocity,
    power_law_wall_shear_rate,
)
from tensorlbm.powerlaw import collide_powerlaw_bgk_forced, guo_force_term  # noqa: E402
from tensorlbm.solver import stream  # noqa: E402

TORCH_THREADS = 4
torch.set_num_threads(TORCH_THREADS)


# ---------------------------------------------------------------------------
# closed-form helpers, written independently of the module under test
# ---------------------------------------------------------------------------

def _cy_formula(g, nu0, nuinf, lam, n, a):
    return nuinf + (nu0 - nuinf) * (1.0 + (lam * g) ** a) ** ((n - 1.0) / a)


def _cross_formula(g, nu0, nuinf, lam, m):
    return nuinf + (nu0 - nuinf) / (1.0 + (lam * g) ** m)


# ---------------------------------------------------------------------------
# closures
# ---------------------------------------------------------------------------

def test_power_law_closure_matches_definition():
    """ν = K γ̇^(n−1) for a thinning and a thickening exponent."""
    g = torch.logspace(-4, -1, 17, dtype=torch.float64)
    for n, K in ((0.7, 0.02), (1.3, 0.5)):
        cl = power_law(K, n)
        got = cl.nu_of_gamma(g)
        want = K * g ** (n - 1.0)
        assert torch.allclose(got, want, rtol=1e-12)


def test_carreau_yasuda_matches_independent_formula_and_limits():
    """Carreau–Yasuda: formula, low- and high-shear asymptotes."""
    cl = carreau_yasuda(nu_0=0.30, nu_inf=0.02, relaxation_time=500.0, flow_index=0.4, yasuda_exponent=2.0)
    g = torch.logspace(-6, 1, 41, dtype=torch.float64)
    want = _cy_formula(g, 0.30, 0.02, 500.0, 0.4, 2.0)
    assert torch.allclose(cl.nu_of_gamma(g), want, rtol=1e-12)
    # asymptotes
    assert abs(float(cl.nu_of_gamma(torch.tensor([0.0], dtype=torch.float64))[0]) - 0.30) < 1e-6
    assert abs(float(cl.nu_of_gamma(torch.tensor([1e6], dtype=torch.float64))[0]) - 0.02) < 1e-3
    # monotone shear thinning
    assert bool(torch.all(torch.diff(cl.nu_of_gamma(g)) < 0.0))


def test_cross_matches_independent_formula_and_exponent_scaling():
    """Cross model: formula plus the λγ̇ ≫ 1 power-law tail ν ∝ γ̇^(−m)."""
    cl = cross(nu_0=0.30, nu_inf=0.02, relaxation_time=500.0, exponent=1.2)
    g = torch.logspace(-6, 1, 33, dtype=torch.float64)
    want = _cross_formula(g, 0.30, 0.02, 500.0, 1.2)
    assert torch.allclose(cl.nu_of_gamma(g), want, rtol=1e-12)
    assert bool(torch.all(torch.diff(cl.nu_of_gamma(g)) < 0.0))
    # with ν∞ → 0 the tail is a pure power law ν ∝ γ̇^(−m); check the log-log slope
    tail = cross(nu_0=0.30, nu_inf=1e-12, relaxation_time=500.0, exponent=1.2)
    lo, hi = torch.tensor([2.0], dtype=torch.float64), torch.tensor([4.0], dtype=torch.float64)
    slope = math.log(float(tail.nu_of_gamma(hi) / tail.nu_of_gamma(lo))) / math.log(2.0)
    assert -1.25 < slope < -1.15


def test_herschel_bulkley_limits_and_plug_viscosity():
    """H-B: power-law branch at high shear, bounded plug viscosity τ_y·m."""
    ty, K, n, m = 3.5e-4, 0.02, 0.6, 1.0e4
    cl = herschel_bulkley(ty, K, n, regularization=m)
    g_hi = torch.tensor([1e-1], dtype=torch.float64)
    expected = ty / float(g_hi) + K * float(g_hi) ** (n - 1.0)
    assert abs(float(cl.nu_of_gamma(g_hi)[0]) / expected - 1.0) < 1e-6
    assert cl.tau_y == pytest.approx(ty)
    # Bingham limit (n = 1): the plug viscosity saturates at τ_y·m
    bingham = herschel_bulkley(ty, 0.02, 1.0, regularization=m)
    g_lo = torch.tensor([1e-12], dtype=torch.float64)
    assert abs(float(bingham.nu_of_gamma(g_lo)[0]) / (ty * m + 0.02) - 1.0) < 1e-6
    # shear-thinning (n < 1): the power-law branch adds K γ̇^(n-1), so the plug
    # viscosity is *not* bounded by τ_y·m — it must stay finite (GAMMA_FLOOR)
    v = float(cl.nu_of_gamma(g_lo)[0])
    assert math.isfinite(v) and v > ty * m


def test_closure_input_validation():
    """Invalid parameters are rejected instead of silently producing NaNs."""
    with pytest.raises(ValueError):
        power_law(-1.0, 0.7)
    with pytest.raises(ValueError):
        power_law(0.01, 0.0)
    with pytest.raises(ValueError):
        carreau_yasuda(0.02, 0.30, 500.0, 0.4)  # nu_inf > nu_0
    with pytest.raises(ValueError):
        cross(0.30, 0.02, -1.0, 1.2)
    with pytest.raises(ValueError):
        herschel_bulkley(-1e-4, 0.02, 0.6)


# ---------------------------------------------------------------------------
# one-dimensional anchors
# ---------------------------------------------------------------------------

def test_constitutive_inversion_inverts_the_closure():
    """γ̇(τ) really inverts τ(γ̇) for all three closures."""
    g = torch.logspace(-6, -1, 25, dtype=torch.float64)
    for cl in (
        power_law(0.02, 0.7),
        carreau_yasuda(0.30, 0.02, 500.0, 0.4),
        cross(0.30, 0.02, 500.0, 1.2),
        herschel_bulkley(3.5e-4, 0.02, 0.6, regularization=1e4),
    ):
        tau = cl.tau_of_gamma(g)
        back = gamma_of_tau(cl, tau)
        assert float(((back - g).abs() / g).max()) < 1e-6


def test_reference_recovers_closed_form_power_law_profile():
    """The quadrature anchor reproduces the analytic Ostwald–de Waele profile."""
    H, n, K, U = 10.0, 0.7, 0.0164, 0.02
    g_wall = U * (n + 1.0) / (n * H)
    K = 0.1 * g_wall ** (1.0 - n)
    accel = K * g_wall ** n / H
    cl = power_law(K, n)
    y = torch.arange(1, int(2 * H), dtype=torch.float64) - H
    ref = channel_reference(cl, accel, H, y=y, n_panels=2000)
    closed = power_law_velocity(y, H, accel, K, n)
    assert float(((ref["u"] - closed).abs()).max() / float(closed.max())) < 1e-9
    assert abs(ref["gamma_dot_wall"] / power_law_wall_shear_rate(H, accel, K, n) - 1.0) < 1e-9


def test_reference_no_slip_and_monotone_profile():
    """Anchor profiles satisfy u(±H) = 0, u_max at the centre, γ̇ > 0 inside."""
    H = 8.0
    cl = carreau_yasuda(0.30, 0.02, 500.0, 0.4)
    ref = channel_reference(cl, 2e-5, H, y=torch.tensor([-H, 0.0, H], dtype=torch.float64))
    assert abs(float(ref["u"][0])) < 1e-14 and abs(float(ref["u"][2])) < 1e-14
    assert float(ref["u"][1]) == pytest.approx(ref["u_max"], rel=1e-6)
    assert ref["gamma_dot_wall"] > 0.0


def test_plug_half_width_ideal_is_tau_y_over_rho_a():
    """r_c = τ_y/(ρa) and the regularised anchor detection agrees to ~1 %."""
    H, n, K, m = 12.0, 0.6, 0.02, 1.0e4
    plug_fraction = 0.35
    p = (n + 1.0) / n
    amp = 0.02 / (H * (1.0 - plug_fraction)) ** p
    accel = K * (amp * (n + 1.0) / n) ** n
    ty = plug_fraction * accel * H
    cl = herschel_bulkley(ty, K, n, regularization=m)
    assert plug_half_width_ideal_from_closure(cl, accel) == pytest.approx(ty / accel)
    y = channel_geometry(int(2 * H))["y_center"]
    ref = channel_reference(cl, accel, H, y=y, n_panels=2000)
    rep = plug_report(cl, accel, y, ref["u"], ref["gamma_dot"], y_ref=y, u_ref=ref["u"],
                      gamma_ref=ref["gamma_dot"], half_width=H)
    assert abs(rep["rel_ref_fit"]) < 0.01          # fit detector: regularisation bias
    assert abs(rep["rel_ref_stress"]) < 1e-6       # stress detector is exact on the anchor


# ---------------------------------------------------------------------------
# lattice solver
# ---------------------------------------------------------------------------

def test_guo_forcing_is_mass_neutral():
    """Σ_i F_i = 0 — regression for the 3× factor that made the force a mass source."""
    ny, nx = 8, 4
    rho = torch.ones((ny, nx))
    ux = torch.full((ny, nx), 0.01)
    uy = torch.zeros((ny, nx))
    f = equilibrium(rho, ux, uy)
    tau = torch.full((ny, nx), 0.8)
    F = guo_force_term(f, 1.0e-4, tau, ux_star=ux + 0.5e-4)
    mass = F.sum(dim=0)
    assert float(mass.abs().max()) < 1e-9
    cx = torch.tensor([0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0]).view(9, 1, 1)
    mom = (F * cx).sum(dim=0)
    # the discrete force carries the (1 - 1/(2τ)) weight; the velocity shift of
    # the equilibrium supplies the other half of ρa per step, which the channel
    # profile tests verify end to end.
    assert float((mom - (1.0 - 0.5 / 0.8) * 1.0e-4).abs().max()) < 1e-9


def test_wall_nonequilibrium_sets_no_slip():
    """The wall condition reproduces a resting wall state (u ≈ 0 on the plates)."""
    geo = channel_geometry(8, nx=3)
    ny = geo["ny"]
    ux = torch.zeros((ny, 3))
    ux[1:-1] = 0.01
    f = equilibrium(torch.ones((ny, 3)), ux, torch.zeros((ny, 3)))
    g = wall_nonequilibrium_2d(f)
    _, uxb, _ = macroscopic(g)
    assert float(uxb[0].abs().max()) < 1e-3
    assert float(uxb[-1].abs().max()) < 1e-3
    # and the fluid rows are untouched
    assert torch.allclose(g[:, 1:-1], f[:, 1:-1])


def test_newtonian_control_matches_hagen_poiseuille():
    """ν = const: the solver must reproduce the parabola (control for the surface)."""
    h, nu, u_max = 16, 0.1, 0.02
    H = h / 2
    accel = 2.0 * nu * u_max / H ** 2
    cl = newtonian(nu)
    y = channel_geometry(h)["y_center"]
    ref = channel_reference(cl, accel, H, y=y)
    out = run_channel(h, accel, cl, nx=1, n_steps=15000, tol=1e-6, check_every=500,
                      u_init=ref["u"])
    err = l2_relative_error(out["u"][1:-1].double(), ref["u"][1:-1])
    assert err < 0.02
    assert out["u_max"] == pytest.approx(float(ref["u_max"]), rel=0.02)


def test_power_law_channel_matches_closed_form():
    """Shear-thinning (n = 0.7) channel profile within the 3 % engineering gate."""
    h, n, nu_wall, u_max = 16, 0.7, 0.1, 0.02
    H = h / 2
    g_wall = u_max * (n + 1.0) / (n * H)
    K = nu_wall * g_wall ** (1.0 - n)
    accel = K * g_wall ** n / H
    cl = power_law(K, n)
    y = channel_geometry(h)["y_center"]
    ref = channel_reference(cl, accel, H, y=y)
    u_init = power_law_velocity(y, H, accel, K, n)
    u_init[0] = 0.0
    u_init[-1] = 0.0
    out = run_channel(h, accel, cl, nx=1, n_steps=15000, tol=1e-6, check_every=500,
                      u_init=u_init)
    closed = power_law_velocity(y[1:-1], H, accel, K, n)
    err = l2_relative_error(out["u"][1:-1].double(), closed)
    assert err < 0.03
    # the measured shear rate must be consistent with the driving stress
    r_meas, g_meas = symmetrise(out["y"], out["gamma_dot"].mean(dim=1))
    r_ref, g_ref = symmetrise(y, ref["gamma_dot"])
    keep = r_meas <= H * (1.0 + 1e-12)      # fluid region only (no solid rows)
    g_m, g_r = g_meas[keep], g_ref[keep]
    # the wall shear rate is the engineering quantity; the centreline shear rate
    # vanishes and is ill-conditioned, so the profile is compared where the
    # shear rate is a non-trivial fraction of its wall value
    assert abs(float(g_m[-1]) / float(g_r[-1]) - 1.0) < 0.20
    # and the whole shear-rate field is accurate to a few % of the wall value
    assert float((g_m - g_r).abs().max()) < 0.05 * float(g_r[-1])


def test_shear_thickening_channel_matches_closed_form():
    """Shear-thickening (n = 1.3) channel profile within 3 %."""
    h, n, nu_wall, u_max = 16, 1.3, 0.1, 0.02
    H = h / 2
    g_wall = u_max * (n + 1.0) / (n * H)
    K = nu_wall * g_wall ** (1.0 - n)
    accel = K * g_wall ** n / H
    cl = power_law(K, n)
    y = channel_geometry(h)["y_center"]
    u_init = power_law_velocity(y, H, accel, K, n)
    u_init[0] = 0.0
    u_init[-1] = 0.0
    out = run_channel(h, accel, cl, nx=1, n_steps=15000, tol=1e-6, check_every=500,
                      u_init=u_init)
    closed = power_law_velocity(y[1:-1], H, accel, K, n)
    assert l2_relative_error(out["u"][1:-1].double(), closed) < 0.03


def test_carreau_yasuda_viscosity_field_follows_the_curve():
    """The viscosity the solver applies equals η(γ̇_measured) of the closure."""
    h, u_max = 16, 0.02
    H = h / 2
    cl = carreau_yasuda(0.30, 0.02, 400.0, 0.4)
    from tensorlbm.nonnewtonian_frontier.reference import calibrate_accel

    accel = calibrate_accel(cl, H, u_max, n_panels=200, n_iter=40)["accel"]
    y = channel_geometry(h)["y_center"]
    ref = channel_reference(cl, accel, H, y=y, n_panels=2000)
    out = run_channel(h, accel, cl, nx=1, n_steps=15000, tol=1e-6, check_every=500,
                      u_init=ref["u"])
    rows = slice(1, h)
    g = out["gamma_dot"][rows].mean(dim=1).double()
    applied = out["nu"][rows].mean(dim=1).double()
    direct = cl.nu_of_gamma(torch.clamp(g, min=1e-30))
    assert float(((applied - direct).abs() / direct).max()) < 1e-6
    assert l2_relative_error(out["u"][1:-1].double(), ref["u"][1:-1]) < 0.05


def test_herschel_bulkley_plug_detectors_on_a_coarse_channel():
    """Ice-slurry plug: fit detector lands within 8 % at a coarse grid (h = 12)."""
    h, n, K, m, plug_fraction, u_max = 12, 0.6, 0.02, 1.0e4, 0.35, 0.02
    H = h / 2
    p = (n + 1.0) / n
    amp = u_max / (H * (1.0 - plug_fraction)) ** p
    accel = K * (amp * (n + 1.0) / n) ** n
    ty = plug_fraction * accel * H
    cl = herschel_bulkley(ty, K, n, regularization=m)
    y = channel_geometry(h)["y_center"]
    ref = channel_reference(cl, accel, H, y=y, n_panels=2000)
    out = run_channel(h, accel, cl, nx=1, n_steps=15000, tol=1e-6, check_every=500,
                      u_init=ref["u"])
    rep = plug_report(cl, accel, out["y"], out["u"], out["gamma_dot"].mean(dim=1),
                      y_ref=y, u_ref=ref["u"], gamma_ref=ref["gamma_dot"], half_width=H)
    assert abs(rep["r_c_lbm_fit"] / rep["r_c_ideal"] - 1.0) < 0.08
    # the fit detector reproduces the ideal plug on the anchor itself
    assert abs(rep["rel_ref_fit"]) < 0.02
    # velocity threshold detector is a coarse diagnostic only
    assert plug_half_width_velocity(*symmetrise(y, ref["u"]), delta=0.01) > 0.0
    assert plug_half_width_stress(*symmetrise(y, ref["gamma_dot"]), cl) == pytest.approx(rep["r_c_ideal"], rel=1e-4)


def test_shear_rate_profile_semantics():
    """The shear-rate stencil is exact for a parabola, including the wall rows."""
    h = 10
    H = h / 2
    y = channel_geometry(h)["y_center"]
    u = (H ** 2 - y ** 2).float()   # parabola
    ux = u.unsqueeze(1).repeat(1, 3)
    g = shear_rate_profile(ux)
    g_ref = (2.0 * y.abs()).float().unsqueeze(1).repeat(1, 3)
    # the wall-adjacent rows use the one-sided quadratic stencil, exact for a
    # parabola as well
    assert float((g[1:-1] - g_ref[1:-1]).abs().max()) < 1e-4


def test_run_channel_validates_arguments():
    """Guard rails: bad drive / relaxation inputs are rejected."""
    cl = newtonian(0.1)
    with pytest.raises(ValueError):
        run_channel(8, 0.0, cl, n_steps=1)
    with pytest.raises(ValueError):
        run_channel(8, 1e-5, cl, omega_nu=0.0, n_steps=1)
    with pytest.raises(ValueError):
        channel_geometry(2)


def test_streaming_preserves_mass_with_body_force():
    """A forced channel conserves mass to float32 accuracy (forcing bug regression)."""
    h, nx = 8, 2
    geo = channel_geometry(h, nx=nx)
    ny = geo["ny"]
    cl = newtonian(0.1)
    accel = 2.0 * cl.params["nu"] * 0.02 / (0.5 * h) ** 2
    f = equilibrium(torch.ones((ny, nx)), torch.zeros((ny, nx)), torch.zeros((ny, nx)))
    m0 = float(f.sum())
    tau = torch.full((ny, nx), 3.0 * cl.params["nu"] + 0.5)
    for _ in range(600):
        f = collide_powerlaw_bgk_forced(f, accel, 1.0, 1.0, tau_field=tau)
        f = wall_nonequilibrium_2d(f)
        f = stream(f)
    drift = abs(float(f.sum()) - m0) / m0
    assert drift < 1e-4, f"mass drift {drift:.2e} — the body force is acting as a mass source"
    _, ux, _ = macroscopic(f)
    assert float(ux[1:-1].abs().max()) > 0.0


def test_benchmark_artifact_contract():
    """If the benchmark has been run, its artifact must satisfy the gate schema."""
    res = REPO / "benchmarks" / "p40w4_n04_nonnewtonian_rheology" / "result.json"
    gates = REPO / "benchmarks" / "p40w4_n04_nonnewtonian_rheology" / "gates.json"
    if not res.exists() or not gates.exists():
        pytest.skip("benchmark artifacts not present in this checkout")
    data = json.loads(res.read_text())
    g = json.loads(gates.read_text())
    ids = {item["id"] for item in g["gates"]}
    assert {"G1_power_law_profile", "G2_carreau_yasuda", "G3_cross",
            "G4_ice_slurry_plug", "G5_grid_convergence_order", "G6_pytest"} <= ids
    for item in g["gates"]:
        assert "threshold" in item and "worst" in item and "passed" in item
    assert data["case"] == "p40w4_n04_nonnewtonian_rheology"
    assert "power_law" in data and "carreau_yasuda" in data and "herschel_bulkley" in data