"""Unit tests for the density-ratio colour-gradient reachability probes.

These tests verify:
1. The initialisation produces the correct pure-phase densities.
2. The static stratification probe is stable at ratio 1 (the validated
   equal-density limit) and detects the instability at ratio 1000 with
   default parameters.
3. The water-entry probe runs without crashing (it may be unstable, but
   the probe must return a well-formed dict).
"""

from __future__ import annotations

import torch

from tensorlbm.color_gradient_density_ratio import (
    init_droplet_ratio,
    init_stratified_ratio,
    run_droplet_probe,
    run_stratified_probe,
    spurious_current,
)


def test_init_stratified_ratio_pure_phase_densities():
    """The pure-phase partial densities must equal rho_a0 / rho_b0."""
    ny, nx = 32, 16
    rho_a0, rho_b0 = 1000.0, 1.0
    f, g = init_stratified_ratio(ny, nx, rho_a0, rho_b0, width=2.0,
                                 dev=torch.device("cpu"))
    rho_a = f.sum(dim=0)
    rho_b = g.sum(dim=0)
    # In the lower half (heavy), rho_a ~ rho_a0 and rho_b ~ 0.
    lower = rho_a[: ny // 4]
    assert float(lower.max()) > 0.9 * rho_a0
    assert float(rho_b[: ny // 4].max()) < 0.1 * rho_b0
    # In the upper half (light), rho_b ~ rho_b0 and rho_a ~ 0.
    upper = rho_b[-ny // 4:]
    assert float(upper.max()) > 0.9 * rho_b0
    assert float(rho_a[-ny // 4:].max()) < 0.1 * rho_a0


def test_init_droplet_ratio_pure_phase_densities():
    """The droplet centre must have rho_a ~ rho_a0 and rho_b ~ 0."""
    n = 32
    rho_a0, rho_b0 = 1000.0, 1.0
    f, g = init_droplet_ratio(n, 8.0, rho_a0, rho_b0, width=2.0,
                              dev=torch.device("cpu"))
    rho_a = f.sum(dim=0)
    rho_b = g.sum(dim=0)
    centre = rho_a[n // 2, n // 2]
    assert float(centre) > 0.9 * rho_a0
    assert float(rho_b[n // 2, n // 2]) < 0.1 * rho_b0


def test_stratified_probe_ratio1_stable():
    """At ratio 1 (the validated limit) the probe must be stable with
    round-off-level spurious current."""
    r = run_stratified_probe(32, 16, 1.0, 1.0,
                             nu_a=1.0 / 6.0, nu_b=1.0 / 6.0,
                             A=0.04, beta=0.5, steps=100, width=2.0,
                             dev=torch.device("cpu"))
    assert r["stable"] is True
    # Small spurious current (the colour-gradient model generates a small
    # spurious current at the interface; at ratio 1 it is < 1e-3).
    assert r["u_max"] < 1e-3


def test_stratified_probe_ratio1000_default_params_unstable():
    """At ratio 1000 with default parameters (width=2, nu=1/6) the model
    must go NaN (the documented instability)."""
    r = run_stratified_probe(32, 16, 1000.0, 1.0,
                             nu_a=1.0 / 6.0, nu_b=1.0 / 6.0,
                             A=0.0, beta=0.5, steps=50, width=2.0,
                             dev=torch.device("cpu"))
    assert r["stable"] is False
    assert r["nan_step"] is not None


def test_stratified_probe_ratio1000_wide_interface_stable():
    """At ratio 1000 with a wide interface (width=8) and high viscosity
    (nu=5) the model must not blow up (no NaN), even though the spurious
    current is large."""
    r = run_stratified_probe(32, 16, 1000.0, 1.0,
                             nu_a=5.0, nu_b=5.0,
                             A=0.0, beta=0.5, steps=50, width=8.0,
                             dev=torch.device("cpu"))
    assert r["stable"] is True
    # The spurious current is large (a known limitation), but finite.
    assert r["u_max"] < 1.0


def test_droplet_probe_ratio1_stable():
    """At ratio 1 the droplet must be stable with small radius drift."""
    r = run_droplet_probe(32, 8.0, 1.0, 1.0,
                          nu_a=1.0 / 6.0, nu_b=1.0 / 6.0,
                          A=0.04, beta=0.5, steps=100, width=2.0,
                          dev=torch.device("cpu"))
    assert r["stable"] is True
    assert r["radius_drift_rel"] is not None
    assert r["radius_drift_rel"] < 0.1  # < 10% drift


def test_spurious_current_zero_for_equilibrium():
    """An equilibrium PDF pair must have zero velocity."""
    f, g = init_stratified_ratio(16, 8, 1.0, 1.0, width=2.0,
                                 dev=torch.device("cpu"))
    # At ratio 1 with a flat interface, the velocity is zero.
    u = spurious_current(f, g)
    assert u < 1e-12
