"""Tests for the pseudopotential (Shan-Chen) viscosity-ratio decoupling and the
Hadamard-Rybczynski comparison layer (node IN31-N12, part 2).

The production drop-settling sweep lives in
``benchmarks/rb_discrete_stability/run_hr_bench.py``; here only the fast,
deterministic layers are exercised:

* the relaxation-time inversion that decouples ``lam`` from ``rho_d/rho_c``,
* equality of the per-component-gravity collision kernel with the library kernel,
* the *measured* viscosity ratio (periodic shear-wave decay) at fixed density ratio.
"""

from __future__ import annotations

import pytest

from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.hadamard_rybczynski import drag_coefficient_factor, hr_drag, hr_velocity
from tensorlbm.multiphase3d import collide_sc_two_component_3d
from tensorlbm.pseudopotential_hr import (
    collide_two_component_with_gravity,
    decoupled_relaxation,
    verify_decoupling,
    viscosity_from_tau,
)


def test_relaxation_inversion_is_exact_and_decoupled():
    """tau_d from eq. (3) must give back the requested lambda at any density ratio."""
    for lam in (0.2, 0.5, 1.0, 2.0):
        for dr in (0.5, 1.0, 2.0):
            par = decoupled_relaxation(lam, 1.0, dr, tau_c=0.6)
            mu_c = viscosity_from_tau(1.0, par["tau_c"])
            mu_d = viscosity_from_tau(dr, par["tau_d"])
            assert mu_d / mu_c == pytest.approx(lam, rel=1e-12)
            assert par["density_ratio"] == pytest.approx(dr)


def test_relaxation_rejects_unrealisable_requests():
    """A lambda that would need tau > 2 must raise rather than silently clamp."""
    with pytest.raises(ValueError):
        decoupled_relaxation(10.0, 1.0, 1.0, tau_c=1.0)
    with pytest.raises(ValueError):
        decoupled_relaxation(-1.0, 1.0, 1.0, tau_c=1.0)


def test_per_component_gravity_kernel_matches_library():
    """With equal gravities and relaxation times the new kernel is bit-identical
    to ``multiphase3d.collide_sc_two_component_3d``."""
    import torch

    torch.manual_seed(0)
    n = 8
    rho1 = 1.0 + 0.1 * torch.rand(n, n, n, dtype=torch.float64)
    rho2 = 0.3 + 0.1 * torch.rand(n, n, n, dtype=torch.float64)
    zero = torch.zeros(n, n, n, dtype=torch.float64)
    ux = 0.01 * (torch.rand(n, n, n, dtype=torch.float64) - 0.5)
    f1 = equilibrium3d(rho1, ux, zero, zero)
    f2 = equilibrium3d(rho2, ux, zero, zero)
    for u_eq in ("self", "mixture"):
        for tau1, tau2 in ((0.9, 1.2), (0.9, 0.9)):
            a1, a2 = collide_two_component_with_gravity(
                f1, f2, -1.0, tau1, tau2, 3e-4, 3e-4, u_eq=u_eq
            )
            b1, b2 = collide_sc_two_component_3d(
                f1, f2, G_12=-1.0, tau1=tau1, tau2=tau2, gz=3e-4, u_eq=u_eq
            )
            assert float((a1 - b1).abs().max()) == 0.0
            assert float((a2 - b2).abs().max()) == 0.0


def test_measured_viscosity_ratio_tracks_prescription_not_density():
    """Shear-wave measurement: lam is reproduced (<=1%) and does not move with
    the density ratio (decoupled engine)."""
    rows = verify_decoupling([0.5, 1.0], [0.5, 2.0], tau_c=0.6, steps=400)
    for r in rows:
        assert abs(r["lam_rel_err"]) < 0.01
    # at fixed lam, changing the density ratio must not change lam_measured
    for lam in (0.5, 1.0):
        sel = [r for r in rows if r["lam"] == lam]
        vals = [r["lam_measured"] for r in sel]
        assert max(vals) - min(vals) < 0.01


def test_hr_law_limits_and_ordering():
    """The analytic law used as the comparison target (sanity, not a new result)."""
    mu, R, U = 0.1, 5.0, 1e-3
    assert drag_coefficient_factor(0.0) == pytest.approx(2.0)
    assert drag_coefficient_factor(1e9) == pytest.approx(3.0, rel=1e-6)
    f_bubble = hr_drag(mu, R, U, 0.0)
    f_stokes = hr_drag(mu, R, U, 1e9)
    assert f_stokes > f_bubble
    drho, g = 0.7, 1e-4
    u_small = hr_velocity(drho, g, R, mu, 0.01)
    u_large = hr_velocity(drho, g, R, mu, 100.0)
    assert u_large < u_small
    assert u_small == pytest.approx(drho * g * R**2 / (3.0 * mu), rel=1e-2)
    assert u_large == pytest.approx(2.0 * drho * g * R**2 / (9.0 * mu), rel=1e-2)