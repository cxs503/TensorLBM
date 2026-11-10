"""Tests for the turbulence-closure validation face (MRT + NEEM + three cases).

The tests are ordered from the definition level (moment matrix, collision
operator, viscosity) to the case level (geometry, metrics, reference anchors).
Anything that needs a C compiler is skipped gracefully when ``gcc`` is absent.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from tensorlbm.turbulence_closure import cases, metrics, reference, verification
from tensorlbm.turbulence_closure.lattice import D2Q9, D3Q19
from tensorlbm.turbulence_closure.mrt import (
    MRTModel,
    build_moment_matrix,
    mode_rates,
    stress_mode_indices,
)

HAS_CC = shutil.which("gcc") is not None or shutil.which("cc") is not None
needs_cc = pytest.mark.skipif(not HAS_CC, reason="no C compiler available")


# ---------------------------------------------------------------------------
# collision operator
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lattice", [D2Q9, D3Q19])
def test_moment_matrix_is_orthonormal(lattice):
    M = build_moment_matrix(lattice)
    assert M.shape == (lattice.nq, lattice.nq)
    assert np.abs(M @ M.T - np.eye(lattice.nq)).max() < 1e-12


def test_stress_modes_are_the_traceless_quadratic_subspace():
    assert list(stress_mode_indices(D2Q9)) == [4, 5]
    assert list(stress_mode_indices(D3Q19)) == [5, 6, 7, 8, 9]
    # the energy row of D3Q19 (index 4) must be orthogonal to all stress rows
    M = build_moment_matrix(D3Q19)
    rows = M[stress_mode_indices(D3Q19)]
    assert np.abs(rows @ M[4]).max() < 1e-12


def test_mode_rates_conserve_mass_and_momentum():
    s = mode_rates(build_moment_matrix(D2Q9), D2Q9)
    assert s[0] == 0.0 and s[1] == 0.0 and s[2] == 0.0


def test_mrt_with_equal_rates_reproduces_bgk():
    """With every rate set to 1/tau the MRT operator must be exactly BGK."""
    import torch

    torch.manual_seed(0)
    ny, nx, tau = 6, 4, 0.7
    f = torch.rand(9, ny, nx).double() + 0.1
    model = MRTModel("D2Q9", tau=tau, s_energy=1.0 / tau, s_ghost=1.0 / tau)
    got = model.collide(f)

    rho = f.sum(0)
    c = torch.as_tensor(D2Q9.c).double()
    u = torch.stack([torch.einsum("i,i...->...", c[:, a], f) / rho for a in range(2)])
    cu = torch.einsum("ia,a...->i...", c, u)
    usq = (u * u).sum(0)
    w = torch.as_tensor(D2Q9.w).double().view(9, 1, 1)
    feq = rho * w * (1 + 3 * cu + 4.5 * cu * cu - 1.5 * usq)
    bgk = f - (f - feq) / tau
    assert float((got - bgk).abs().max()) < 1e-12


def test_collision_conserves_density_and_momentum():
    import torch

    torch.manual_seed(1)
    f = torch.rand(9, 5, 4).double() + 0.05
    model = MRTModel("D2Q9", tau=0.9)
    out = model.collide(f)
    assert float((out.sum(0) - f.sum(0)).abs().max()) < 1e-12
    c = torch.as_tensor(D2Q9.c).double()
    for a in range(2):
        before = torch.einsum("i,i...->...", c[:, a], f)
        after = torch.einsum("i,i...->...", c[:, a], out)
        assert float((before - after).abs().max()) < 1e-12


@pytest.mark.parametrize("tau", [0.55, 0.8])
def test_shear_wave_decay_matches_viscosity(tau):
    res = verification.shear_wave_decay(tau, ny=16, steps=400)
    assert res["rel_error"] < 0.05


def test_smagorinsky_increases_effective_viscosity():
    import torch

    torch.manual_seed(3)
    model = MRTModel("D2Q9", tau=0.51, smagorinsky=0.15)
    f = torch.rand(9, 8, 6).double() + 0.2
    tau_field, _ = model.tau_field(f)
    assert float(tau_field.min()) > 0.51


# ---------------------------------------------------------------------------
# native kernels
# ---------------------------------------------------------------------------

@needs_cc
@pytest.mark.parametrize("tau", [0.6, 1.05])
def test_native_kernel_viscosity(tau):
    res = verification.poiseuille_viscosity(tau, ny=24, nx=4)
    assert res["rel_error"] < 0.01


@needs_cc
def test_channel_kernel_matches_laminar_poiseuille():
    """The D3Q19 kernel must reproduce the analytic parabolic profile."""
    from tensorlbm.turbulence_closure import native

    ny, nx, nz, tau = 18, 4, 4, 0.65
    nu = (tau - 0.5) / 3.0
    h = (ny - 2) / 2.0
    g = 1.2e-4
    prefix = "/tmp/tensorlbm_tc_test_lam"
    native.run_channel3d(
        out_prefix=prefix, nx=nx, ny=ny, nz=nz, steps=8000, warmup=6000,
        tau=tau, force=g, cs=0.0, wall_mode=0, sample_every=10, threads=4,
    )
    prof = native.read_profile(prefix)
    y = prof["y"]
    yw, y1 = 0.5, ny - 1.5
    analytic = g / (2 * nu) * (y - yw) * (y1 - y)
    rel = np.abs(prof["u_mean"] - analytic).max() / analytic.max()
    assert rel < 0.02, f"laminar channel profile deviates by {rel:.3%}"


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def test_strouhal_estimator_recovers_known_value():
    u, d, st_true = 0.1, 20, 0.164
    period = d / (st_true * u)
    t = np.arange(0, 40 * period, 1.0)
    fy = np.sin(2 * np.pi * t / period) + 0.05 * np.random.default_rng(0).standard_normal(t.size)
    st = metrics.strouhal_from_lift(t, fy, u_in=u, diameter=d)
    assert abs(st["st_crossings"] - st_true) / st_true < 0.01
    assert abs(st["st_fft"] - st_true) / st_true < 0.03
    assert st["n_cycles"] >= 35


def test_reattachment_length_recovers_known_crossing():
    ny, nx, x_step = 12, 120, 20
    ux = np.ones((ny, nx))
    x = np.arange(nx)
    # reversed flow between the step and x = 60, reattaches at 59.5
    ux[1, x_step:60] = -0.2
    ux[1, 60:] = np.linspace(0.001, 1.0, nx - 60)
    mean = {"ux": ux}
    res = metrics.reattachment_length(mean, x_step=x_step, step_h=10.0, u_in=1.0)
    assert abs(res["x_r_over_h"] - (59.5 - 20) / 10.0) < 0.1


def test_wall_units_force_and_gradient_routes_agree():
    ny, nu, force = 66, 0.002, 1e-6
    h = (ny - 2) / 2.0
    u_tau = np.sqrt(force * h)
    y = np.arange(1, ny - 1, dtype=float)
    # synthetic log profile
    u = u_tau * (np.log(np.maximum((y - 0.5) * u_tau / nu, 1.0)) / 0.41 + 5.2)
    res = metrics.wall_units({"y": y, "u_mean": u, "u_rms": np.zeros_like(y)},
                             ny=ny, nu=nu, force=force)
    assert res["re_tau_force"] == pytest.approx(u_tau * h / nu)
    assert np.isfinite(res["u_plus"]).all()


def test_compare_u_plus_flags_a_shifted_profile():
    from tensorlbm.validation import mkm1999

    dns = mkm1999.load_profile(180)
    y_plus = np.linspace(5, 170, 60)
    case = {"y_plus": y_plus, "u_plus": dns.u_plus_at(y_plus) * 1.2}
    res = metrics.compare_u_plus(case, dns, y_plus_range=(5.0, None))
    assert res["status"] == "ok"
    assert res["max_rel_error"] == pytest.approx(0.2, abs=1e-6)
    case["u_plus"] = dns.u_plus_at(y_plus)
    assert metrics.compare_u_plus(case, dns, y_plus_range=(5.0, None))["max_rel_error"] < 1e-12


# ---------------------------------------------------------------------------
# case geometry / configuration
# ---------------------------------------------------------------------------

def test_step_geometry_matches_armaly():
    cfg = cases.StepConfig(re=100)
    assert cfg.expansion_ratio == pytest.approx(1.9423, abs=1e-4)
    solid, body, x_step = cases.step_masks(cfg)
    assert solid.shape == (cfg.ny, cfg.nx)
    assert solid[0].all() and solid[-1].all()
    assert solid[1 : cfg.step_h + 1, :x_step].all()
    assert not solid[cfg.step_h + 1 : -1, :x_step].any()
    # inlet opening height equals the upstream channel height
    assert int((~solid[:, 0].astype(bool)).sum()) == cfg.upstream_h


def test_step_reynolds_number_definition():
    cfg = cases.StepConfig(re=389, step_h=49, upstream_h=52, u_mean=0.1)
    assert cfg.nu * cfg.re == pytest.approx(cfg.u_mean * 2 * cfg.upstream_h)
    assert cfg.tau < 1.9


def test_cylinder_geometry_and_validation():
    cfg = cases.CylinderConfig(re=100, diameter=24)
    solid, body = cases.cylinder_masks(cfg)
    assert solid[0].all() and solid[-1].all()
    assert body.sum() > 0
    assert not (body[0].any() or body[-1].any())
    cfg.validate()
    with pytest.raises(ValueError):
        cases.CylinderConfig(re=10000, diameter=24).validate()  # tau too close to 1/2


def test_channel_config_self_consistency():
    cfg = cases.ChannelConfig(re_tau=590, ny=242, nu=0.0018)
    h = (242 - 2) / 2
    assert cfg.u_tau == pytest.approx(cfg.re_tau * cfg.nu / h)
    assert np.sqrt(cfg.force * cfg.h) == pytest.approx(cfg.u_tau)
    assert cfg.du == pytest.approx(cfg.re_tau / h)
    cfg.validate()


# ---------------------------------------------------------------------------
# reference anchors
# ---------------------------------------------------------------------------

def test_reference_anchors_are_ordered_and_sourced():
    bfs = [reference.bfs_reference(re) for re in (100, 200, 389)]
    assert [b.value for b in bfs] == sorted(b.value for b in bfs)
    assert all(b.source for b in bfs)
    for re in (100, 200, 300):
        exp = reference.cylinder_reference(re, "experiment")
        dns = reference.cylinder_reference(re, "dns_2d")
        assert abs(exp.value - dns.value) / dns.value < 0.10
        assert all([exp.source, dns.source])


def test_mkm_reference_tables_are_available():
    from tensorlbm.validation import mkm1999

    for re in (180, 395, 590):
        prof = mkm1999.load_profile(re)
        assert prof.u_plus.size > 20
        assert np.all(np.diff(prof.y_plus) > 0)
        assert prof.u_plus[-1] > 15.0