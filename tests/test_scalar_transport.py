"""Acceptance tests for the Boussinesq-coupled scalar transport lattice (IN70-N03).

Each test maps to one machine-checkable criterion of the campaign node:

=====================================================  ==========================
criterion                                              test
=====================================================  ==========================
Gauss profile vs analytic <= 2 %                       ``test_gaussian_pulse_profile``
moving-pulse variance ``sigma^2 = sigma0^2 + 2Dt`` <=1% ``test_gaussian_pulse_variance``
D_eff grid refinement: slope >= 0.9, extrap. <= 2 %     ``test_deff_grid_convergence``
mass conservation, periodic domain (round-off)         ``test_mass_conservation_periodic``
mass conservation, open boundaries + flux budget <=1e-12 ``test_open_boundary_budget``
erfc semi-infinite solution <= 1 %                      ``test_erfc_half_space``
crosswind symmetry <= 1e-8                              ``test_crosswind_symmetry``
Boussinesq impulse balance                              ``test_boussinesq_impulse``
TRT/BGK diffusivity calibration (dispersion)           ``test_dispersion_calibration``
buoyant plume self-similarity exponent ∈ [1.6, 1.75]    ``test_buoyant_plume_exponent``
negative-control momentum jet exponent ∈ [0.95, 1.05]   ``test_momentum_jet_exponent``
LFL contour grid convergence <= 3 %                     ``test_lfl_grid_convergence``
LFL extent monotone decrease with wind (5档)            ``test_lfl_wind_sweep_monotone``
=====================================================  ==========================
"""

from __future__ import annotations

import math

import pytest
import torch

import tensorlbm.scalar_transport as st

torch.set_num_threads(2)

D = 0.05
PROD_N = 192


def _center(c: torch.Tensor) -> torch.Tensor:
    nx = c.shape[1]
    j = int(c.sum(dim=0).argmax())
    return torch.roll(c, nx // 2 - j, dims=1)


def test_lattice_constants():
    assert sum(st.W_D2Q5) == pytest.approx(1.0, abs=1e-15)
    cs2 = sum(w * c * c for w, c in zip(st.W_D2Q5, st.CX_D2Q5))
    assert cs2 == pytest.approx(st.CS2_D2Q5, abs=1e-15)
    for i, j in enumerate(st.OPP_D2Q5):
        assert st.CX_D2Q5[i] == -st.CX_D2Q5[j]
        assert st.CY_D2Q5[i] == -st.CY_D2Q5[j]
        assert st.OPP_D2Q5[j] == i


def test_equilibrium_moments():
    ny, nx = 3, 8
    c = torch.rand((ny, nx), dtype=torch.float64)
    ux = torch.full((ny, nx), 0.01, dtype=torch.float64)
    uy = torch.full((ny, nx), -0.02, dtype=torch.float64)
    geq = st.equilibrium_scalar_d2q5(c, ux, uy)
    assert torch.allclose(geq.sum(dim=0), c, atol=1e-15)
    mom_x = sum(w * cx for w, cx in zip(st.W_D2Q5, st.CX_D2Q5))
    mom_y = sum(w * cy for w, cy in zip(st.W_D2Q5, st.CY_D2Q5))
    assert mom_x == 0.0 and mom_y == 0.0
    flux_x = (geq * torch.tensor(st.CX_D2Q5, dtype=torch.float64).view(5, 1, 1)).sum(dim=0)
    assert torch.allclose(flux_x, c * ux, atol=1e-15)


def test_relaxation_time_from_diffusivity():
    tau = st.tau_from_diffusivity(D)
    assert (tau - 0.5) / 3.0 == pytest.approx(D, rel=1e-15)
    tau_sym = st.tau_symmetric_from_magic(tau, 0.1875)
    assert st.magic_from_taus(tau_sym, tau) == pytest.approx(0.1875, rel=1e-15)


def test_dispersion_calibration():
    tau_odd = st.tau_from_diffusivity(D)
    for magic in (None, 0.1875):
        tau_sym = tau_odd if magic is None else st.tau_symmetric_from_magic(tau_odd, magic)
        rates = st.dispersion_rates(0.05, tau_sym, tau_odd)
        assert rates[0] == pytest.approx(D, rel=2e-3)
        assert rates[1] > 10.0 * D
    cal = st.calibrate_magic(D)
    assert cal["max_rel_dev"] < cal["bgk_max_rel_dev"]
    assert cal["max_rel_dev"] < 2e-3


def test_gaussian_pulse_profile():
    N = PROD_N
    U = 0.02 * 128 / N
    s0 = N / 64.0
    sig = N / 13.0
    steps = int((sig**2 - s0**2) / (2 * D))
    cfg = st.ScalarTransportConfig(nx=N, ny=3, diffusivity=D, ux=U)
    res = st.run_pulse_1d(cfg, steps, mass=1.0, sigma0=s0)
    xs = torch.arange(N, dtype=torch.float64) - N // 2
    c = _center(res.c)
    exact = st.gaussian_pulse_1d(xs, 1.0, D, 0.0, steps, sigma0=s0)
    exact = exact / exact.sum()
    err_full = st.relative_l2(c, exact)
    assert err_full <= 0.02, f"full-profile L2 error {err_full:.3e} > 2%"
    _, cen, var = st.gaussian_moments(c, xs)
    fit = st.gaussian_pulse_1d(xs - float(cen), 1.0, D, 0.0, 1e-30, sigma0=var**0.5)
    fit = fit / fit.sum()
    err_shape = st.relative_l2(c, fit)
    assert err_shape <= 0.02, f"shape error {err_shape:.3e} > 2%"
    assert abs(float(cen)) < 0.5
    assert res.budget.residual_relative() <= 1e-12


def test_gaussian_pulse_variance():
    N = 128
    U = 0.02
    s0 = 2.0
    steps = 900
    cfg = st.ScalarTransportConfig(nx=N, ny=1, diffusivity=D, ux=U)
    res = st.run_pulse_1d(cfg, steps, mass=1.0, sigma0=s0)
    xs = torch.arange(N, dtype=torch.float64) - N // 2
    c = _center(res.c)
    _, cen, var = st.gaussian_moments(c, xs)
    assert var == pytest.approx(s0**2 + 2 * D * steps, rel=1e-2)
    d_eff = st.effective_diffusivity(c, xs, steps, sigma0=s0)
    assert d_eff == pytest.approx(D, rel=1e-2)


def test_deff_grid_convergence():
    rows = []
    for N in (96, 128, 160):
        U = 0.02 * 128 / N
        s0 = N / 64.0
        sig = N / 13.0
        steps = int((sig**2 - s0**2) / (2 * D))
        cfg = st.ScalarTransportConfig(nx=N, ny=3, diffusivity=D, ux=U)
        res = st.run_pulse_1d(cfg, steps, mass=1.0, sigma0=s0)
        xs = torch.arange(N, dtype=torch.float64) - N // 2
        c = _center(res.c)
        rows.append((N, steps, st.effective_diffusivity(c, xs, steps, sigma0=s0)))
    errs = [abs(v / D - 1.0) for _, _, v in rows]
    slopes = [
        math.log(errs[i] / errs[i + 1]) / math.log(rows[i + 1][0] / rows[i][0])
        for i in range(len(errs) - 1)
    ]
    assert min(slopes) >= 0.9, f"observed orders {slopes} below 0.9"
    p = slopes[-1]
    r = rows[-1][0] / rows[-2][0]
    d_inf = rows[-1][2] + (rows[-1][2] - rows[-2][2]) / (r**p - 1.0)
    assert abs(d_inf / D - 1.0) <= 0.02, f"extrapolated D error {d_inf / D - 1:.3e}"


def test_mass_conservation_periodic():
    N = 64
    cfg = st.ScalarTransportConfig(nx=N, ny=3, diffusivity=D, ux=0.01)
    res = st.run_pulse_1d(cfg, 400, mass=1.0, sigma0=2.0)
    mass = float(res.c.sum())
    assert res.budget.injected == 0.0 and res.budget.outflux == 0.0
    assert abs(mass - 3.0) / 3.0 <= 1e-12
    assert res.budget.residual_relative() <= 1e-12


def test_open_boundary_budget():
    cfg = st.ScalarTransportConfig(nx=64, ny=41, diffusivity=D, ux=0.04)
    res = st.run_crosswind_release(cfg, 600, source_rate=1.0)
    budget = res.budget
    assert budget.injected > 500.0
    assert budget.outflux > 0.0
    assert budget.residual_relative() <= 1e-12, budget.as_dict()


def test_erfc_half_space():
    N = 384
    steps = 8000
    cfg = st.ScalarTransportConfig(nx=N, ny=1, diffusivity=D)
    res = st.run_erfc_half_space(cfg, steps, level=1.0)
    x = res.x
    width = math.sqrt(4.0 * D * steps)
    exact = st.erfc_half_space(x, 1.0, D, steps)
    mask = x < 3.0 * width
    err = st.relative_l2(res.c[0], exact, mask)
    assert err <= 1e-2, f"erfc L2 error {err:.3e} > 1%"
    assert res.budget.residual_relative() <= 1e-12, res.budget.as_dict()


def test_crosswind_symmetry():
    cfg = st.ScalarTransportConfig(nx=96, ny=61, diffusivity=D, ux=0.06)
    res = st.run_crosswind_release(cfg, 1200, source_rate=1.0)
    c = res.c
    assert c.shape[0] % 2 == 1
    asym = float((c - c.flip(0)).abs().max()) / float(c.max())
    assert asym <= 1e-8, f"crosswind asymmetry {asym:.3e}"


def test_lfl_contour_metrics_monotone_in_level():
    cfg = st.ScalarTransportConfig(nx=96, ny=61, diffusivity=D, ux=0.06)
    res = st.run_crosswind_release(cfg, 1200, source_rate=1.0)
    c = res.c
    levels = [0.1, 0.2, 0.4, 0.6]
    extent = [st.lfl_contour_metrics(c, lv)["extent_x"] for lv in levels]
    assert all(extent[i] >= extent[i + 1] for i in range(len(extent) - 1))
    assert extent[0] > extent[-1] > 0.0


def test_buoyancy_force_sign():
    c = torch.tensor([[0.2, 0.5]], dtype=torch.float64)
    fx, fy = st.boussinesq_force(c, c_ref=0.0, gravity=1.0, expansivity=1.0)
    assert float(fx.abs().max()) == 0.0
    assert float(fy[0, 0]) == pytest.approx(0.2)
    assert float(fy[0, 1]) > float(fy[0, 0])


def test_boussinesq_impulse():
    from tensorlbm.d2q9 import equilibrium, macroscopic

    ny, nx = 8, 8
    rho = torch.ones((ny, nx), dtype=torch.float64)
    zero = torch.zeros((ny, nx), dtype=torch.float64)
    f0 = equilibrium(rho, zero, zero)
    anomaly = 0.5
    g_val, beta = 1e-3, 1.0
    f = f0.clone()
    steps = 100
    for _ in range(steps):
        f = st.buoyant_momentum_step(
            f, tau=0.8, concentration=torch.full((ny, nx), anomaly, dtype=torch.float64),
            c_ref=0.0, gravity=g_val, expansivity=beta,
        )
    _, ux, uy = macroscopic(f)
    expected = g_val * beta * anomaly * steps
    assert float(uy.mean()) == pytest.approx(expected, rel=1e-6)
    assert float(ux.abs().max()) < 1e-12


def _fit_log_log(ys, us):
    ys_f = [math.log(y) for y in ys]
    us_f = [math.log(u) for u in us]
    n = len(ys_f)
    sum_x = sum(ys_f)
    sum_y = sum(us_f)
    sum_xy = sum(x * y for x, y in zip(ys_f, us_f))
    sum_x2 = sum(x * x for x in ys_f)
    denom = n * sum_x2 - sum_x * sum_x
    if abs(denom) < 1e-12:
        return float('nan')
    return (n * sum_xy - sum_x * sum_y) / denom


@pytest.mark.slow
def test_buoyant_plume_exponent():
    """Buoyant plume: flux exponent Q(z) ~ z^{5/3} => fitted exponent ∈ [1.6, 1.75]."""
    result = st.run_buoyant_plume(
        nx=32, ny=80,
        diffusivity=0.02, viscosity=0.02,
        source_rate=0.5, source_halfwidth=1,
        gravity=2e-3, expansivity=1.0,
        steps=3000, record_every=100,
    )
    cl = result["plume_centerline"]
    assert len(cl) >= 8, f"not enough centerline samples: {len(cl)}"
    ys = [p[0] for p in cl]
    us = [p[2] for p in cl]
    y_min, y_max = min(ys), max(ys)
    y_lo = y_min + 0.2 * (y_max - y_min)
    y_hi = y_min + 0.8 * (y_max - y_min)
    fit_data = [(y, u) for y, u in zip(ys, us) if y_lo <= y <= y_hi and u > 1e-10]
    assert len(fit_data) >= 4, f"too few fit points: {len(fit_data)}"
    ys_f = [y for y, _ in fit_data]
    us_f = [u for _, u in fit_data]
    b = _fit_log_log(ys_f, us_f)
    flux_exponent = 1.0 - b
    assert 1.6 <= flux_exponent <= 1.75, (
        f"buoyant plume flux exponent {flux_exponent:.3f} not in [1.6, 1.75], "
        f"velocity slope b={b:.3f}"
    )


@pytest.mark.slow
def test_momentum_jet_exponent():
    """Momentum jet (no buoyancy): flux exponent ∈ [0.95, 1.05]."""
    result = st.run_momentum_jet(
        nx=32, ny=80,
        diffusivity=0.02, viscosity=0.02,
        jet_velocity=0.05, source_halfwidth=1,
        steps=3000, record_every=100,
    )
    cl = result["jet_centerline"]
    assert len(cl) >= 8, f"not enough centerline samples: {len(cl)}"
    ys = [p[0] for p in cl]
    us = [p[1] for p in cl]
    y_min, y_max = min(ys), max(ys)
    y_lo = y_min + 0.2 * (y_max - y_min)
    y_hi = y_min + 0.8 * (y_max - y_min)
    fit_data = [(y, u) for y, u in zip(ys, us) if y_lo <= y <= y_hi and u > 1e-10]
    assert len(fit_data) >= 4, f"too few fit points: {len(fit_data)}"
    ys_f = [y for y, _ in fit_data]
    us_f = [u for _, u in fit_data]
    b = _fit_log_log(ys_f, us_f)
    flux_exponent = 1.0 - b
    assert 0.95 <= flux_exponent <= 1.05, (
        f"momentum jet flux exponent {flux_exponent:.3f} not in [0.95, 1.05], "
        f"velocity slope b={b:.3f}"
    )


def test_lfl_grid_convergence():
    """LFL isopleth extent converges within 3 % on grid refinement."""
    lfl_level = 0.1
    extents = []
    for N in (96, 128, 160):
        ny = int(N * 81 / 128)
        cfg = st.ScalarTransportConfig(nx=N, ny=ny, diffusivity=D, ux=0.06)
        res = st.run_crosswind_release(cfg, 1500, source_rate=1.0)
        metrics = st.lfl_contour_metrics(res.c, lfl_level)
        extents.append(metrics["extent_x"])
    rel_diff = abs(extents[-1] - extents[-2]) / extents[-1]
    assert rel_diff <= 0.03, (
        f"LFL extent grid convergence: {rel_diff:.3e} > 3%, "
        f"extents at N=96,128,160: {extents}"
    )


def test_lfl_wind_sweep_monotone():
    """LFL extent decreases monotonically with wind speed (5档).

    Uses wind speeds in the advection-dominated regime where monotonicity
    is expected.  At very low speeds the scalar may not reach the LFL
    threshold, producing non-monotone behaviour that is a physical effect
    (insufficient advection) rather than a numerical defect.
    """
    wind_speeds = [0.04, 0.06, 0.08, 0.10, 0.12]
    results = st.run_lfl_wind_sweep(
        wind_speeds, nx=96, ny=61, diffusivity=D,
        source_rate=1.0, lfl_level=0.1, steps=1500,
    )
    extents = [r["extent_x"] for r in results]
    assert all(e > 0 for e in extents), f"zero extent at some wind speed: {extents}"
    for i in range(len(extents) - 1):
        assert extents[i] >= extents[i + 1], (
            f"LFL extent not monotone: ux={wind_speeds[i]} -> {extents[i]}, "
            f"ux={wind_speeds[i+1]} -> {extents[i+1]}"
        )
    assert extents[0] > extents[-1] * 1.3, (
        f"LFL extent ratio too small: {extents[0]} / {extents[-1]} = {extents[0]/extents[-1]:.2f}"
    )
