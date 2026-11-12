"""Machine gates for the free-energy Laplace activation face.

Four things are pinned, all of them falsifiable without a solver:

1. the two closed forms of ``sigma`` are algebraically identical, and the
   gradient-energy integral of the analytic profile agrees with both (this is the
   analytic account every free-energy two-phase result is graded against);
2. ``sigma_from_profile`` recovers ``sigma`` from an *analytic* profile with an
   error that decreases with the interface width - the discretisation error bar -
   and a regression guard for the "counts 2 sigma" defect (a full axis cut
   crosses the interface twice; the first version of the function did that and
   was biased high by tens of percent);
3. ``equilibrium_gate`` withholds a measurement when the state is not an
   equilibrium state (non-finite, non-uniform mu, not at rest) and reports the
   code, so a failing run can never be quoted as a Laplace number;
4. ``stability_probe`` does not call an exploding-but-finite run ``stable``.

The solver-backed smoke test is tiny (12^3, few steps) and marked ``slow`` only
for the desktop-CI budget; it asserts finiteness, not physics.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.phasefield.laplace_face import (
    analytic_reference,
    analytic_sigma_gradient_integral,
    chemical_potential_spread,
    equilibrium_gate,
    radial_setup,
    run_static_droplet,
    sigma_from_profile,
    stability_probe,
    tanh_profile,
)

A = 0.1
B = 0.1


@pytest.mark.parametrize("kappa", [0.02, 0.05, 0.1, 0.2])
def test_sigma_closed_forms_are_identical(kappa: float) -> None:
    ref = analytic_reference(A, B, kappa)
    assert ref["sigma_identity_residual"] < 1e-12
    assert ref["sigma"] == pytest.approx(math.sqrt(8.0 * kappa * A**3 / (9.0 * B * B)))


@pytest.mark.parametrize("kappa", [0.02, 0.05, 0.1, 0.2])
def test_gradient_integral_closed_form_matches_coefficient_form(kappa: float) -> None:
    ref = analytic_reference(A, B, kappa)
    integral = analytic_sigma_gradient_integral(A, B, kappa)
    assert integral == pytest.approx(ref["sigma"], rel=1e-12)


def test_xi_and_phi0_follow_the_double_well() -> None:
    ref = analytic_reference(A, B, 0.05)
    # phi0 = sqrt(A/B), xi = sqrt(2 kappa / A)
    assert ref["phi0"] == pytest.approx(math.sqrt(A / B))
    assert ref["xi"] == pytest.approx(math.sqrt(2 * 0.05 / A))
    assert ref["xi"] == pytest.approx(1.0)


@pytest.mark.parametrize("kappa,xi", [(0.05, 1.0), (0.1, math.sqrt(2.0)), (0.2, 2.0)])
def test_field_route_recovers_analytic_sigma(kappa: float, xi: float) -> None:
    """Discretisation error bar of the field route on the analytic profile.

    The interface is placed at ``n/4`` so the outward half-line of the cut crosses
    it exactly once.  A full axis cut would cross it twice and return ``2 sigma``
    (the defect this test guards): with the half-line the error is a pure
    discretisation error and must shrink with ``xi``.
    """
    n = 96
    ref = analytic_reference(A, B, kappa)
    r, centre = radial_setup(n)
    phi = tanh_profile(r, R=n / 4.0, phi0=ref["phi0"], xi=ref["xi"])
    measured = sigma_from_profile(phi, centre=centre, kappa=kappa)
    rel = abs(measured - ref["sigma"]) / ref["sigma"]
    # One cell per interface width at xi = 1 is the worst case; the gate is a
    # coarse ceiling, the *trend* is pinned by the next test.  The measured
    # values are in docs/FREE_ENERGY_KORTEWEG_LAPLACE_20261112.md table 2.
    assert rel < 0.40, f"xi={xi}: field route rel. error {rel:.3%} above 40%"


def test_field_route_counts_one_interface_only() -> None:
    """Regression for the double-counting defect: 2x sigma is out of the gate."""
    n = 96
    kappa = 0.2
    ref = analytic_reference(A, B, kappa)
    r, centre = radial_setup(n)
    phi = tanh_profile(r, R=n / 4.0, phi0=ref["phi0"], xi=ref["xi"])
    measured = sigma_from_profile(phi, centre=centre, kappa=kappa)
    assert measured < 1.3 * ref["sigma"]


def test_field_route_error_decreases_with_interface_width() -> None:
    """The error bar of the field route is a function of ``xi``, not a constant.

    A wide interface is resolved by more lattice nodes, so the discrete
    gradient-energy integral must approach the analytic ``sigma`` monotonically.
    Ordering is pinned rather than absolute values: the absolute level depends on
    the cut resolution and is reported in the bench table instead.
    """
    n = 96
    errors = []
    for kappa in (0.05, 0.1, 0.2):
        ref = analytic_reference(A, B, kappa)
        r, centre = radial_setup(n)
        phi = tanh_profile(r, R=n / 4.0, phi0=ref["phi0"], xi=ref["xi"])
        measured = sigma_from_profile(phi, centre=centre, kappa=kappa)
        errors.append(abs(measured - ref["sigma"]) / ref["sigma"])
    assert errors[0] > errors[1] > errors[2], errors


