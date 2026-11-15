"""Unit + regression tests for the Allen-Cahn / CSF force chain (m22329).

The heavy acceptance grading (Young-Laplace ``dp`` vs ``2 sigma/R`` across
density ratios 1:1 / 10:1 / 100:1) is produced by
``scripts/xval_ac_csf_force_chain_m22329.py --matrix`` and stored in
``artifacts/xval_ac_csf_force_chain_m22329/matrix.json``.  The tests here pin
the *pieces* that grading depends on, so a refactor cannot silently change the
meaning of the numbers:

* each force term equals its published formula (checked against an independent
  finite-difference evaluation of the same expression),
* the velocity normalisation is the one the ablation found stable,
* the variable-density Laplace helper does not confuse the bulk density jump
  with the capillary jump (the defect of the pre-existing helper),
* the density-deviation formulation is stable on a 10:1 droplet where the
  pinned-density one is not, and a coarse ``dp`` stays in the right ballpark.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.allen_cahn_lbm import free_energy_parameters, laplace_pressure_jump
from tensorlbm.phasefield.ac_csf_force_chain import (
    ForceChainConfig,
    _equilibrium_density_deviation,
    _equilibrium_full_density,
    ac_csf_step,
    chemical_potential_isotropic,
    density_from_phase_field,
    hydrodynamic_force_chain,
    laplace_pressure_jump_2phase,
    pressure_force,
    relaxation_time_from_phase_field,
    second_moment_neq,
    surface_tension_force,
    viscous_force,
)
from tensorlbm.phasefield.isotropic_fd import isotropic_gradient_3d

torch.set_num_threads(1)


def _droplet(n: int, radius: float, width: float) -> torch.Tensor:
    axis = torch.arange(n, dtype=torch.float64)
    zz, yy, xx = torch.meshgrid(axis, axis, axis, indexing="ij")
    r = torch.sqrt((zz - n / 2 + 0.5) ** 2 + (yy - n / 2 + 0.5) ** 2 + (xx - n / 2 + 0.5) ** 2)
    return torch.tanh((radius - r) / width)


# --------------------------------------------------------------------------- #
# material fields
# --------------------------------------------------------------------------- #
def test_density_interpolation_hits_both_bulk_values():
    phi = torch.tensor([[[-1.0, 0.0, 1.0]]], dtype=torch.float64)
    rho = density_from_phase_field(phi, rho_h=1.0, rho_l=0.1)
    assert rho[0, 0, 0].item() == pytest.approx(0.1)
    assert rho[0, 0, 1].item() == pytest.approx(0.55)
    assert rho[0, 0, 2].item() == pytest.approx(1.0)


def test_relaxation_time_obeys_cs2_equals_one_third():
    """``tau = 1/2 + 3 nu`` so that the recovered viscosity is ``nu``."""
    phi = torch.tensor([[[-1.0, 1.0]]], dtype=torch.float64)
    tau = relaxation_time_from_phase_field(phi, nu_h=0.1, nu_l=0.02)
    for t, nu in ((tau[0, 0, 0].item(), 0.02), (tau[0, 0, 1].item(), 0.1)):
        assert (t - 0.5) / 3.0 == pytest.approx(nu)


# --------------------------------------------------------------------------- #
# the three force terms
# --------------------------------------------------------------------------- #
def test_surface_tension_force_is_mu_grad_phi():
    phi = _droplet(16, 5.0, 2.0)
    mu = chemical_potential_isotropic(phi, beta=1e-3, kappa=1e-2)
    got = surface_tension_force(phi, mu)
    g = isotropic_gradient_3d(phi, boundary="periodic")
    for a in range(3):
        assert torch.allclose(got[a], mu * g[a])


def test_pressure_force_matches_published_formula():
    """``F_p = -(1/3) rho_dev (rho_H - rho_L) grad(phi)``, evaluated independently."""
    phi = _droplet(16, 5.0, 2.0)
    rho_dev = torch.linspace(-1.0, 1.0, phi.numel()).reshape(phi.shape)
    got = pressure_force(phi, rho_dev, rho_h=1.0, rho_l=0.1)
    g = isotropic_gradient_3d(phi, boundary="periodic")
    for a in range(3):
        expected = -(1.0 / 3.0) * rho_dev * 0.9 * g[a]
        assert torch.allclose(got[a], expected)
    # a uniform density deviation on a flat-free field must give a nonzero term
    assert float(torch.stack(got).abs().max()) > 0.0


def test_pressure_force_vanishes_at_equal_density():
    phi = _droplet(12, 4.0, 2.0)
    rho_dev = torch.ones_like(phi)
    got = pressure_force(phi, rho_dev, rho_h=1.0, rho_l=1.0)
    assert max(float(x.abs().max()) for x in got) == 0.0


def test_second_moment_neq_of_a_single_population():
    """One nonzero population in direction ``q`` must give ``c_q (x) c_q``."""
    neq = torch.zeros(19, 3, 4, 5, dtype=torch.float64)
    from tensorlbm.d3q19 import C

    q = 1  # (1, 0, 0)
    neq[q] = 2.5
    pis = second_moment_neq(neq)
    assert pis[0, 0].unique().numel() == 1
    assert pis[0, 0][0, 0, 0].item() == pytest.approx(2.5)
    assert float(pis[1, 1].abs().max()) == 0.0
    assert float(pis[0, 1].abs().max()) == 0.0

    q = 13  # a face-diagonal direction
    neq.zero_()
    neq[q] = 1.0
    pis = second_moment_neq(neq)
    cx, cy, cz = [float(C[q, a]) for a in range(3)]
    assert pis[0, 1][0, 0, 0].item() == pytest.approx(cx * cy)
    assert pis[2, 2][0, 0, 0].item() == pytest.approx(cz * cz)


def test_viscous_force_sign_and_scale():
    """``F_v = (1/2 - tau) Delta rho (Pi^neq . grad phi)``; antisymmetric in tau."""
    phi = _droplet(16, 5.0, 2.0)
    neq = torch.zeros(19, *phi.shape, dtype=torch.float64)
    neq[1] = 1e-3
    pis = second_moment_neq(neq)
    rho_h, rho_l = 1.0, 0.1
    fv_lo = viscous_force(phi, pis, torch.full_like(phi, 0.6), rho_h, rho_l)
    fv_hi = viscous_force(phi, pis, torch.full_like(phi, 0.9), rho_h, rho_l)
    g = isotropic_gradient_3d(phi, boundary="periodic")
    expected_x = (0.5 - 0.6) * 0.9 * pis[0, 0] * g[0]
    assert torch.allclose(fv_lo[0], expected_x)
    # (1/2 - tau) is linear in tau, so scaling tau must scale the force: the
    # ratio is exactly (0.5 - 0.9) / (0.5 - 0.6) = 4
    assert torch.allclose(fv_hi[0], 4.0 * fv_lo[0])
    assert float(fv_hi[0].abs().max()) > float(fv_lo[0].abs().max())


# --------------------------------------------------------------------------- #
# chain assembly
# --------------------------------------------------------------------------- #
def test_chain_switches_add_exactly_the_selected_terms():
    n = 16
    phi = _droplet(n, 5.0, 2.0)
    f = _equilibrium_density_deviation(torch.zeros_like(phi), *(torch.zeros_like(phi),) * 3)
    f = f + 1e-6 * torch.randn_like(f)
    f_eq = _equilibrium_density_deviation(torch.zeros_like(phi), *(torch.zeros_like(phi),) * 3)
    rho_dev = f.sum(dim=0)
    tau = relaxation_time_from_phase_field(phi, 0.1, 0.02)
    kwargs = dict(beta=1e-3, kappa=1e-2, rho_h=1.0, rho_l=0.1, tau=tau, rho_dev=rho_dev)
    st, _ = hydrodynamic_force_chain(
        phi, f, f_eq, config=ForceChainConfig(terms=("st",)), **kwargs
    )
    stp, parts = hydrodynamic_force_chain(
        phi, f, f_eq, config=ForceChainConfig(terms=("st", "p")), **kwargs
    )
    stpv, _ = hydrodynamic_force_chain(
        phi, f, f_eq, config=ForceChainConfig(terms=("st", "p", "v")), **kwargs
    )
    for a in range(3):
        assert torch.allclose(stp[a] - st[a], _pressure_component(a, phi, rho_dev))
        assert float(torch.stack(stpv).abs().max()) >= float(torch.stack(stp).abs().max())
        assert torch.allclose(stpv[a] - stp[a], _viscous_component(a, phi, f - f_eq, tau))


def _pressure_component(a: int, phi: torch.Tensor, rho_dev: torch.Tensor) -> torch.Tensor:
    return pressure_force(phi, rho_dev, 1.0, 0.1)[a]


def _viscous_component(a: int, phi: torch.Tensor, neq: torch.Tensor, tau: torch.Tensor):
    return viscous_force(phi, second_moment_neq(neq), tau, 1.0, 0.1)[a]


def test_unknown_chain_term_is_rejected():
    with pytest.raises(ValueError, match="unknown force chain term"):
        ForceChainConfig(terms=("st", "bogus"))


# --------------------------------------------------------------------------- #
# Laplace grading
# --------------------------------------------------------------------------- #
def test_laplace_helper_sphere_uses_two_curvatures():
    phi = _droplet(24, 8.0, 2.5)
    rho_dev = torch.zeros_like(phi)
    rho_dev[phi > 0.9] = 3.0 * 1e-3
    out = laplace_pressure_jump_2phase(
        None, phi, radius=8.0, sigma=1e-2, rho_h=1.0, rho_l=0.1, rho_dev=rho_dev
    )
    assert out["curvature_coefficient"] == pytest.approx(2.0 / 8.0)
    assert out["dp_laplace"] == pytest.approx(2e-2 / 8.0)
    # dp = cs^2 * rho_dev = 1e-3 exactly, so the reported error is quantifiable
    assert out["dp_measured"] == pytest.approx(1e-3, rel=1e-12)
    cyl = laplace_pressure_jump_2phase(
        None,
        phi,
        radius=8.0,
        sigma=1e-2,
        rho_h=1.0,
        rho_l=0.1,
        rho_dev=rho_dev,
        geometry="cylinder",
    )
    assert cyl["dp_laplace"] == pytest.approx(1e-2 / 8.0)


def test_laplace_helper_does_not_report_the_bulk_density_jump():
    """REG-RATIO: at 10:1 the *total* density jump is ``0.9/3 = 0.3``.

    The pre-existing helper (:func:`tensorlbm.allen_cahn_lbm.laplace_pressure_jump`)
    treats ``p = cs^2 rho`` with the total density, so on a *perfect* 10:1 droplet
    at rest it reports ``dp = 0.3`` where the capillary jump is ``~1e-3``: a
    ~30000% error that has nothing to do with the force chain.  The new helper
    measures the deviation from each phase's own equation of state instead.
    """
    from tensorlbm.d3q19 import equilibrium3d

    phi = _droplet(24, 8.0, 2.5)
    rho = density_from_phase_field(phi, 1.0, 0.1)
    zero = torch.zeros_like(phi)
    f = equilibrium3d(rho, zero, zero, zero).to(torch.float64)

    legacy = laplace_pressure_jump(f, phi, radius=8.0, sigma=1e-2)
    assert legacy["dp_measured"] == pytest.approx((1.0 - 0.1) / 3.0, rel=5e-2)

    modern = laplace_pressure_jump_2phase(
        f, phi, radius=8.0, sigma=1e-2, rho_h=1.0, rho_l=0.1, formulation="full_density"
    )
    # residual is the interface-shell bias of the fixed ``phi > 0.9`` bulk mask
    # (the shell is a large fraction of a small droplet), two orders of
    # magnitude below the 0.3 the legacy helper reports
    assert abs(modern["dp_measured"]) < 2e-2, modern


# --------------------------------------------------------------------------- #
# stepper behaviour
# --------------------------------------------------------------------------- #
def _step_series(grid, radius, width, ratio, steps, formulation, terms=("st", "p", "v")):
    phi = _droplet(grid, radius, width)
    if formulation == "density_deviation":
        f = _equilibrium_density_deviation(phi * 0, phi * 0, phi * 0, phi * 0)
    else:
        rho = density_from_phase_field(phi, 1.0, 1.0 / ratio)
        f = _equilibrium_full_density(rho, phi * 0, phi * 0, phi * 0).to(torch.float64)
    config = ForceChainConfig(terms=terms, formulation=formulation)  # type: ignore[arg-type]
    us = []
    for _ in range(steps):
        f, phi, diag = ac_csf_step(
            f,
            phi,
            rho_h=1.0,
            rho_l=1.0 / ratio,
            nu_h=0.1,
            nu_l=0.02,
            sigma=1e-2,
            W=width,
            mobility=5e-3,
            config=config,
        )
        if not (torch.isfinite(f).all() and torch.isfinite(phi).all()):
            return us, False
        us.append(diag["max_abs_u"])
    return us, True


def test_density_deviation_formulation_is_stable_at_ten_to_one():
    """REG-RATIO root cause: a pinned ``p = cs^2 rho(phi)`` cannot carry ``sigma/R``.

    Measured on 2026-11-09 (m22329): the legacy formulation accumulates momentum
    ``~8e-2`` per step and goes non-finite; the density-deviation formulation
    keeps ``max|u|`` bounded and small over the same number of steps.
    """
    us_dev, ok_dev = _step_series(20, 6.0, 2.0, 10.0, 120, "density_deviation")
    assert ok_dev, "density-deviation formulation must stay finite"
    assert max(us_dev) < 5e-2, f"spurious current too large: {max(us_dev):.3e}"

    us_full, ok_full = _step_series(20, 6.0, 2.0, 10.0, 120, "full_density")
    assert (not ok_full) or max(us_full) > 0.05, (
        "the pinned-density formulation is expected to fail here; if it no longer does, "
        "re-measure the ablation in scripts/xval_ac_csf_force_chain_m22329.py before "
        "relaxing this regression"
    )


def test_physical_velocity_normalisation_is_unstable():
    """FALSIFIED: ``u = (sum c f + F/2)/rho(phi)`` in the deviation formulation.

    Kept as a regression because it is the obvious-looking choice; measured
    non-finite within 13 steps at 10:1 (m22329).  A change that makes it work
    would invalidate the documented reason for ``momentum_density``.
    """
    phi = _droplet(20, 6.0, 2.0)
    f = _equilibrium_density_deviation(phi * 0, phi * 0, phi * 0, phi * 0)
    config = ForceChainConfig(
        terms=("st", "p", "v"), formulation="density_deviation", momentum_density="physical"
    )
    finite = True
    for _ in range(40):
        f, phi, diag = ac_csf_step(
            f,
            phi,
            rho_h=1.0,
            rho_l=0.1,
            nu_h=0.1,
            nu_l=0.02,
            sigma=1e-2,
            W=2.0,
            mobility=5e-3,
            config=config,
        )
        if not torch.isfinite(f).all():
            finite = False
            break
    assert not finite, "the physical normalisation was expected to diverge (see docstring)"


def test_coarse_laplace_jump_is_in_the_right_ballpark():
    """Coarse bound only; the graded acceptance lives in the cluster artifact."""
    grid, radius, width = 24, 8.0, 2.5
    phi = _droplet(grid, radius, width)
    f = _equilibrium_density_deviation(phi * 0, phi * 0, phi * 0, phi * 0)
    config = ForceChainConfig(terms=("st", "p", "v"), formulation="density_deviation")
    for _ in range(150):
        f, phi, _ = ac_csf_step(
            f,
            phi,
            rho_h=1.0,
            rho_l=0.1,
            nu_h=0.1,
            nu_l=0.02,
            sigma=1e-2,
            W=width,
            mobility=5e-3,
            config=config,
        )
    jump = laplace_pressure_jump_2phase(
        f, phi, radius=radius, sigma=1e-2, rho_h=1.0, rho_l=0.1
    )
    assert jump["dp_measured"] > 0.0, "pressure must be higher inside the droplet"
    assert jump["relative_error"] < 0.85, jump


def test_ac_csf_step_rejects_bad_shapes():
    phi = _droplet(8, 2.0, 1.5)
    f = _equilibrium_density_deviation(phi * 0, phi * 0, phi * 0, phi * 0)
    with pytest.raises(ValueError, match="f must have shape"):
        ac_csf_step(torch.zeros(9, 8, 8, 8, dtype=torch.float64), phi)
    with pytest.raises(ValueError, match="phi must have shape"):
        ac_csf_step(f, phi[:4])
    with pytest.raises(ValueError, match="densities must be positive"):
        ac_csf_step(f, phi, rho_l=0.0)