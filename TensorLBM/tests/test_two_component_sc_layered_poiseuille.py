"""Tests for the two-component Shan-Chen layered-Poiseuille validation.

The assertions are *physical*: a laminar channel driver must reproduce the exact
Poiseuille profile (that is precisely what the previous delivery got wrong --
its run gave ``u_max = g*t`` because the vertical streaming was a no-op), the
``G > 0`` interaction must demix the two components, and the mass of each
component must be conserved exactly (mass is not a convergence property).
"""

import numpy as np
import pytest
import torch

from tensorlbm.verification.layered_poiseuille.analytic import solve_layered
from tensorlbm.verification.layered_poiseuille.two_component_sc import (
    TwoComponentSCConfig,
    TwoComponentSCSolver,
    equilibrium,
    macroscopic,
    psi_exp,
    stream_bounce_back_y,
)


# --------------------------------------------------------------------------- #
# building blocks
# --------------------------------------------------------------------------- #
def test_streaming_advects_in_both_directions():
    """Regression for the defect that sank m21558: y-advection must happen.

    A distribution that is a delta in the ``+y`` direction must move one cell up
    per time step (periodic x, halfway bounce-back walls on rows 0 / ny-1).
    """
    f = torch.zeros(9, 12, 4)
    i = 2  # c = (0, +1) in the D2Q9 ordering (1:(1,0) 2:(0,1) 3:(-1,0) 4:(0,-1))
    f[i, 5, 2] = 1.0
    out = stream_bounce_back_y(f)
    assert out[i, 6, 2] == pytest.approx(1.0)
    assert float(out[i].sum()) == pytest.approx(1.0)


def test_streaming_x_periodic():
    f = torch.zeros(9, 12, 4)
    f[1, 5, 3] = 1.0  # c = (1, 0)
    out = stream_bounce_back_y(f)
    assert out[1, 5, 0] == pytest.approx(1.0)  # wrapped around in x


def test_pseudopotential_monotone_and_bounded():
    rho = torch.linspace(0.01, 20.0, 200)
    psi = psi_exp(rho)
    assert torch.all(psi[1:] >= psi[:-1])  # float32 saturates at rho0 for large rho
    assert float(psi.max()) <= 1.0 + 1e-6


def test_equilibrium_moments():
    rho = torch.full((6, 3), 2.0)
    ux = torch.full((6, 3), 0.05)
    uy = torch.full((6, 3), -0.02)
    f = equilibrium(rho, ux, uy)
    r, ex, ey = macroscopic(f)
    assert torch.allclose(r, rho, atol=1e-9)
    assert torch.allclose(ex, ux, atol=1e-9)
    assert torch.allclose(ey, uy, atol=1e-9)


# --------------------------------------------------------------------------- #
# precision: single-fluid control (this is the defect detector)
# --------------------------------------------------------------------------- #
def test_single_component_poiseuille_within_3pct():
    """One component, G=0, uniform rho: exact Poiseuille ``u=g y (H-y)/(2 nu)``."""
    cfg = TwoComponentSCConfig(ny=18, nx=4, tau_a=0.8, tau_b=0.8, G=0.0,
                               g_x=1e-4, g_y=0.0, steps=2000,
                               init_rho_a=2.0, init_rho_b=0.0, init_layer_frac=1.0)
    solver = TwoComponentSCSolver(cfg)
    prof = solver.run()["final_profile"]
    sol = solve_layered(cfg.height, [0.5 * cfg.height], [cfg.nu_a, cfg.nu_a], cfg.g_x)
    u_ref = sol.u(prof["y"])
    eps_inf = np.max(np.abs(prof["ux_total"] - u_ref)) / u_ref.max()
    assert eps_inf < 0.03, f"single-fluid Poiseuille error {eps_inf:.4%} > 3%"
    # no ballistic acceleration: g*t after 2000 steps would be 0.2, 6x the viscous peak
    assert np.max(np.abs(prof["ux_total"])) < 0.05


def test_mixture_coupling_reproduces_single_fluid():
    """Two components with equal tau and G=0 must behave as one fluid (0.1 < eps < 3%)."""
    cfg = TwoComponentSCConfig(ny=18, nx=2, tau_a=0.8, tau_b=0.8, G=0.0,
                               g_x=1e-4, g_y=0.0, steps=1500,
                               init_rho_a=2.0, init_rho_b=2.0, init_layer_frac=0.5,
                               interface_width=2.0)
    solver = TwoComponentSCSolver(cfg)
    prof = solver.run()["final_profile"]
    sol = solve_layered(cfg.height, [0.5 * cfg.height], [0.1, 0.1], cfg.g_x)
    u_ref = sol.u(prof["y"])
    eps_inf = np.max(np.abs(prof["ux_total"] - u_ref)) / u_ref.max()
    assert eps_inf < 0.03, f"mixture control error {eps_inf:.4%} > 3%"


# --------------------------------------------------------------------------- #
# demixing, conservation, stability
# --------------------------------------------------------------------------- #
def test_demixing_and_mass_conservation():
    cfg = TwoComponentSCConfig(ny=18, nx=2, tau_a=0.8, tau_b=1.0, G=2.5,
                               g_x=0.0, g_y=0.0, steps=900, init_rho_a=2.1, init_rho_b=2.3,
                               init_layer_frac=0.5, interface_width=2.0, g_ramp_steps=400,
                               u_eq="viscous", wall_mode="mirror")
    solver = TwoComponentSCSolver(cfg)
    m0 = solver.get_macroscopic()
    mass_a0, mass_b0 = float(m0["rho_a"].sum()), float(m0["rho_b"].sum())
    prof = solver.run()["final_profile"]
    assert np.all(np.isfinite(prof["ux_total"]))
    phi = prof["rho_a"] / (prof["rho_a"] + prof["rho_b"] + 1e-12)
    mid = len(phi) // 2
    assert phi[: mid // 2].mean() > 0.6, "component A did not settle at the bottom"
    assert phi[-mid // 2 :].mean() < 0.4, "component B did not settle at the top"
    m1 = solver.get_macroscopic()
    assert abs(float(m1["rho_a"].sum()) - mass_a0) / mass_a0 < 1e-6
    assert abs(float(m1["rho_b"].sum()) - mass_b0) / mass_b0 < 1e-6


def test_static_interface_has_no_spurious_flow():
    cfg = TwoComponentSCConfig(ny=18, nx=2, tau_a=0.8, tau_b=1.0, G=2.5,
                               g_x=0.0, g_y=0.0, steps=600, init_rho_a=2.1, init_rho_b=2.3,
                               init_layer_frac=0.5, interface_width=2.0, g_ramp_steps=300)
    solver = TwoComponentSCSolver(cfg)
    prof = solver.run()["final_profile"]
    driven = solve_layered(cfg.height, [0.5 * cfg.height], [cfg.nu_a, cfg.nu_b], 5e-5)
    ratio = np.max(np.abs(prof["ux_total"])) / driven.u_peak()
    assert ratio < 0.15, f"spurious current {ratio:.3%} of the driven peak"


def test_runs_are_bit_reproducible():
    kwargs = dict(ny=16, nx=2, tau_a=0.8, tau_b=1.0, G=2.5, g_x=5e-5, g_y=0.0, steps=120,
                  init_rho_a=2.1, init_rho_b=2.3, init_layer_frac=0.5, g_ramp_steps=60)
    a = TwoComponentSCSolver(TwoComponentSCConfig(**kwargs)).run()["final_profile"]
    b = TwoComponentSCSolver(TwoComponentSCConfig(**kwargs)).run()["final_profile"]
    assert np.array_equal(a["ux_total"], b["ux_total"])
    assert np.array_equal(a["rho_a"], b["rho_a"])


# --------------------------------------------------------------------------- #
# anchor KPI: layered two-component Poiseuille vs the exact layered solution
# --------------------------------------------------------------------------- #
def test_layered_poiseuille_within_15pct():
    """Two-component layered Poiseuille vs the exact layered solution (owner gate 15%)."""
    cfg = TwoComponentSCConfig(ny=24, nx=1, tau_a=0.8, tau_b=1.0, G=2.5, g_x=5e-5, g_y=0.0,
                               steps=3000, init_rho_a=2.12, init_rho_b=2.36,
                               init_layer_frac=0.5, interface_width=2.0, g_ramp_steps=1000,
                               u_eq="viscous", wall_mode="mirror")
    solver = TwoComponentSCSolver(cfg)
    prof = solver.run()["final_profile"]
    u = prof["ux_total"]
    y = prof["y"]
    H = cfg.height
    nom = solve_layered(H, [0.5 * H], [cfg.nu_a, cfg.nu_b], cfg.g_x)
    phi = prof["rho_a"] / (prof["rho_a"] + prof["rho_b"] + 1e-12)
    below = np.where(phi >= 0.5)[0]
    y_iface = float(y[below[-1]]) if len(below) else 0.5 * H       # measured interface
    matched = solve_layered(H, [y_iface], [cfg.nu_a, cfg.nu_b], cfg.g_x)
    err = {}
    for tag, sol in (("nominal", nom), ("interface_matched", matched)):
        ref = sol.u(y)
        err[tag] = (np.max(np.abs(u - ref)) / ref.max(),
                    np.sqrt(np.mean((u - ref) ** 2)) / np.sqrt(np.mean(ref ** 2)))
    best = min(err.values(), key=lambda e: e[0])
    assert best[0] < 0.15, f"layered profile error {err} exceeds the 15% owner gate"
    # the peak must sit in the low-viscosity (top) layer, i.e. above the interface
    assert np.argmax(u) >= len(u) // 2  # peak at/above the interface (low-viscosity side)
    # viscosity ratio recovered from the curvature of the two layers
    half = 0.5 * H
    lower = (y >= 2.0) & (y <= half - 3.0)
    a2 = np.polyfit(y[lower], u[lower], 2)[0]
    nu_lower = -cfg.g_x / (2 * a2)
    assert abs(nu_lower - cfg.nu_a) / cfg.nu_a < 0.30, f"nu_lower={nu_lower:.4f} vs {cfg.nu_a}"
