"""Benchmark-grade tests for the non-Newtonian LBM package (P304W2-N07).

Covers the six acceptance criteria of the campaign node at a coarse
(fast) resolution:

* C2/C5 -- channel flow vs the analytical / semi-analytical reference
* C4    -- Picard (rheology) iteration count
* C6    -- this file: > 10 tests of model / strain rate / iteration / benchmarks

The production-resolution numbers live in ``artifacts/n07/validation.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tensorlbm.non_newtonian.cavity import CavityConfig, CavityNonNewtonian
from tensorlbm.non_newtonian.pipe import PipeConfig, PipeNonNewtonian
from tensorlbm.non_newtonian.verify_n07 import _reference_profile
from tensorlbm.non_newtonian.rheology import (
    apparent_viscosity_carreau_yasuda,
    apparent_viscosity_power_law,
    carreau_yasuda_max_velocity,
    carreau_yasuda_shear_rate,
    poiseuille_carreau_yasuda_profile,
    poiseuille_newtonian_profile,
    poiseuille_power_law_profile,
    solve_body_force_for_target_umax_cy,
    strain_rate_magnitude_2d,
)

torch.set_num_threads(1)


# ---------------------------------------------------------------------------
# 1-3: Carreau-Yasuda model limits
# ---------------------------------------------------------------------------

def test_cy_zero_shear_plateau():
    g = torch.tensor([1e-12, 1e-10], dtype=torch.float64)
    nu = apparent_viscosity_carreau_yasuda(
        g, nu_0=0.1, nu_inf=0.005, relaxation_time=1000.0, flow_index=0.5)
    assert torch.allclose(nu, torch.full_like(g, 0.1), rtol=1e-6)


def test_cy_high_shear_plateau():
    # lambda*gamma_dot >> 1 and (nu_0-nu_inf)*(lambda*gamma_dot)^(n-1) << nu_inf
    g = torch.tensor([1e12, 1e14], dtype=torch.float64)
    nu = apparent_viscosity_carreau_yasuda(
        g, nu_0=0.1, nu_inf=0.005, relaxation_time=1e-6, flow_index=0.5)
    assert torch.allclose(nu, torch.full_like(g, 0.005), rtol=2e-2)


def test_cy_newtonian_degeneration_constant():
    """mu_0 == mu_inf  =>  nu = const (Newtonian), independent of shear rate."""
    g = torch.logspace(-8, 8, 33, dtype=torch.float64)
    nu = apparent_viscosity_carreau_yasuda(
        g, nu_0=0.032, nu_inf=0.032, relaxation_time=1.0, flow_index=0.5)
    assert torch.allclose(nu, torch.full_like(g, 0.032), rtol=1e-12)


def test_cy_shear_thinning_monotone():
    g = torch.logspace(-6, 6, 200, dtype=torch.float64)
    nu = apparent_viscosity_carreau_yasuda(
        g, nu_0=0.1, nu_inf=0.005, relaxation_time=1000.0, flow_index=0.5)
    d = (nu[1:] - nu[:-1]).max()
    assert d <= 0.0


# ---------------------------------------------------------------------------
# 4-6: exact Carreau-Yasuda channel reference (stress inversion + quadrature)
# ---------------------------------------------------------------------------

def test_cy_exact_channel_reduces_to_poiseuille():
    H, G, nu = 32.0, 3.125e-6, 0.032
    y = torch.linspace(-H, H, 65, dtype=torch.float64)
    u_cy = poiseuille_carreau_yasuda_profile(y, H, G, nu, nu, 1.0, 1.0)
    u_nw = poiseuille_newtonian_profile(y, H, G, nu)
    assert float((u_cy - u_nw).abs().max()) < 1e-12


def test_cy_stress_inversion_consistency():
    """Gamma * nu(Gamma) must equal the imposed stress G*|s|."""
    s = torch.linspace(0.0, 30.0, 31, dtype=torch.float64)
    G, nu_0, nu_inf, lam, n = 5e-6, 0.1, 0.005, 1000.0, 0.5
    g = carreau_yasuda_shear_rate(s, G, nu_0, nu_inf, lam, n)
    nu = apparent_viscosity_carreau_yasuda(
        g, nu_0=nu_0, nu_inf=nu_inf, relaxation_time=lam,
        flow_index=n, shear_rate_floor=1e-300)
    stress = g * nu
    assert float((stress - G * s).abs().max() / (G * 30.0)) < 1e-8


def test_cy_body_force_solver_roundtrip():
    H = 32.0
    target = 0.05
    G = solve_body_force_for_target_umax_cy(H, target, 0.1, 0.005, 1000.0, 0.5)
    umax = carreau_yasuda_max_velocity(H, G, 0.1, 0.005, 1000.0, 0.5)
    assert abs(umax - target) / target < 1e-3


# ---------------------------------------------------------------------------
# 7: strain-rate magnitude
# ---------------------------------------------------------------------------

def test_strain_rate_magnitude_linear_shear():
    """u = a*y => gamma_dot = |a| exactly (interior nodes)."""
    ny, nx = 33, 4
    a = 0.0123
    # unit lattice spacing: strain_rate_magnitude_2d differentiates on a unit grid
    y = torch.arange(ny, dtype=torch.float64) - (ny - 1) / 2.0
    ux = (a * y).unsqueeze(1).repeat(1, nx)
    uy = torch.zeros_like(ux)
    g = strain_rate_magnitude_2d(ux, uy)
    interior = g[1:-1, :]
    assert float((interior - a).abs().max()) < 1e-12


def test_power_law_newtonian_equals_constant():
    g = torch.logspace(-6, 6, 21, dtype=torch.float64)
    nu = apparent_viscosity_power_law(g, consistency_index=0.032, flow_index=1.0)
    assert torch.allclose(nu, torch.full_like(g, 0.032), rtol=1e-12)


# ---------------------------------------------------------------------------
# 8: wall boundary condition puts zero velocity on the wall nodes
# ---------------------------------------------------------------------------

def test_bounce_back_zero_wall_velocity():
    cfg = PipeConfig(nx=4, ny=33, n_power=1.0, picard_max_iter=1,
                     lbm_steps_per_picard=400, max_steps=400, flow_tol=1e-12,
                     u_init_amplitude=0.04, u_init_shape="flat")
    solver = PipeNonNewtonian(cfg)
    solver.solve()
    wall_u = torch.cat([solver.ux[0, :], solver.ux[-1, :]]).abs().max().item()
    assert wall_u < 1e-12


# ---------------------------------------------------------------------------
# 9: Newtonian limit -- power-law n=1 and CY with mu0=muinf give the same field
# ---------------------------------------------------------------------------

def test_newtonian_limit_power_law_matches_cy():
    base = dict(nx=33, ny=33, re=100.0, u_lid=0.1, n_power=1.0,
                picard_max_iter=6, picard_tol=1e-8, lbm_steps_per_picard=400,
                flow_tol=1e-12, max_steps=2400)
    a = CavityNonNewtonian(CavityConfig(rheology="power_law", **base))
    a.solve()
    nu_ref = a._consistency_K
    b = CavityNonNewtonian(CavityConfig(rheology="carreau_yasuda",
                                        nu_0=nu_ref, nu_inf=nu_ref,
                                        lambda_cy=1.0, **base))
    b.solve()
    diff = (a.ux - b.ux).abs().max().item()
    assert diff < 1e-10 * 0.1


# ---------------------------------------------------------------------------
# 10: channel benchmark at coarse resolution vs the analytical profile (C2/C5)
# ---------------------------------------------------------------------------

RUNS_N07 = Path(__file__).resolve().parents[1] / "runs" / "n07"


@pytest.mark.parametrize(
    "npz,limit",
    [
        ("pipe_n1.0_ny65.npz", 0.02),
        ("pipe_n1.0_ny129.npz", 0.02),
        ("pipe_n1.5_ny129.npz", 0.02),
        ("pipe_cy_n0.5_ny129.npz", 0.02),
        ("pipe_cy_n1.0_ny65.npz", 0.02),
    ],
)
def test_channel_benchmark_artifacts_below_2pct(npz, limit):
    """C2/C5: production channel runs stored in runs/n07 must be < 2 % L2."""
    p = RUNS_N07 / npz
    if not p.exists():
        pytest.skip(f"{npz} not generated in this checkout")
    d = np.load(p)
    u = d["u"]
    ua = _reference_profile(d)          # same convention as verify_n07
    l2 = float(np.linalg.norm(u - ua) / np.linalg.norm(ua))
    assert l2 < limit, (npz, l2)


def test_channel_newtonian_smoke_short_run():
    """Fast end-to-end check: Newtonian channel reaches ~parabolic shape."""
    cfg = PipeConfig(nx=4, ny=17, n_power=1.0, picard_max_iter=6,
                     picard_tol=1e-6, lbm_steps_per_picard=500,
                     flow_tol=1e-4, max_steps=3000, u_max_target=0.05,
                     u_init_amplitude=0.05, u_init_shape="parabolic")
    solver = PipeNonNewtonian(cfg)
    info = solver.solve()
    y, u = solver.get_velocity_profile()
    ua = solver.analytical_profile(y)
    l2 = float(np.linalg.norm(u - ua) / np.linalg.norm(ua))
    assert l2 < 0.05, l2
    assert "flow_residual" in info and "flow_history" in info


# ---------------------------------------------------------------------------
# 11-12: Picard iteration accounting (C4)
# ---------------------------------------------------------------------------

def test_picard_iterations_under_50_cavity_benchmarks():
    """C4: the stored cavity benchmark runs must converge in < 50 Picard blocks."""
    f = RUNS_N07 / "result_cavity_n1.5.json"
    if not f.exists():
        pytest.skip("cavity benchmark json not generated in this checkout")
    d = json.loads(f.read_text())
    assert d["converged"] is True
    assert d["picard_iterations"] < 50


def test_picard_accounting_fields():
    cfg = CavityConfig(nx=17, ny=17, re=100.0, u_lid=0.1, n_power=1.5,
                       picard_max_iter=4, picard_tol=1e-6,
                       lbm_steps_per_picard=200, flow_tol=1e-8, max_steps=800)
    info = CavityNonNewtonian(cfg).solve()
    assert info["picard_iterations"] <= 4
    assert len(info["picard_history"]) == len(info["flow_history"])
    assert info["converged"] is False        # flow not steady in 800 steps


def test_solve_requires_both_flow_and_rheology_convergence():
    """A run that is cut off before the flow is steady must not report converged."""
    cfg = PipeConfig(nx=4, ny=33, n_power=0.5, picard_max_iter=2,
                     picard_tol=1e-6, lbm_steps_per_picard=100,
                     flow_tol=1e-10, max_steps=200, tau_max=5.0)
    info = PipeNonNewtonian(cfg).solve()
    assert info["converged"] is False
    assert info["flow_residual"] > 1e-10


def test_power_law_profile_limits():
    y = torch.tensor([0.0], dtype=torch.float64)
    H, G, K = 32.0, 1e-6, 1e-3
    u = poiseuille_power_law_profile(y, H, G, K, 1.0)
    assert float(u[0]) > 0.0
    # n = 1 must reproduce the Newtonian parabola with nu = K
    y2 = torch.linspace(-H, H, 33, dtype=torch.float64)
    u1 = poiseuille_power_law_profile(y2, H, G, K, 1.0)
    u2 = poiseuille_newtonian_profile(y2, H, G, K)
    assert float((u1 - u2).abs().max()) < 1e-14