"""Gates for the analytic Young-Laplace droplet calibration protocol.

Each test pins one link of the measurement chain (alias ``XVAL-LBM-FREE-ENERGY-LBM-
ANALYTIC-YOUNG-LAPLACE-D``).  The *physics verdict* lives in
``benchmarks/case_pool/free_energy_young_laplace/evidence/m20929/summary.json`` and is
produced by ``run_xval.py``; here we assert the invariants that would otherwise break
silently and turn a physics mismatch into a post-processing artefact:

1. the two closed forms of the free-energy tension are the same number, and the
   gradient-energy form agrees with a *numerical* integral of the analytic tanh
   profile (the closed form is not taken on faith);
2. the diffuse-interface radius is recovered from a synthetic droplet by both the
   ``phi = 0`` contour and the ``atanh`` fit, and the fit returns the model ``xi``;
3. the polar projection of the pressure tensor preserves the trace exactly, and the
   radial anisotropy of the *analytic* pressure tensor integrates to the analytical
   tension (so a later mismatch on a simulated field is about the simulation);
4. the double-count defect (integrating across two interfaces returns ``2 sigma``)
   is *rejected* by the integration guard rather than silently accepted;
5. bulk-band extraction raises instead of returning ``nan`` when a band is empty;
6. the grading logic fails planted defects of known magnitude (20 % tension error,
   runaway spurious current) and passes a 2 % one.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.phasefield.free_energy_pressure import (  # noqa: E402
    grad_lap,
    pressure_tensor_analytic,
)
from tensorlbm.phasefield.laplace_droplet_calibration import (  # noqa: E402
    DropletCase,
    DropletMeasures,
    LaplaceCriteria,
    bulk_pressure_jump,
    droplet_phi0,
    grade_case,
    grade_report,
    polar_pressure_components,
    radial_anisotropy_profile,
    radial_average,
    radius_and_width_from_tanh_fit,
    radius_from_contour,
    run_droplet_case,
    sigma_analytic,
    sigma_from_gradient_energy,
    tension_integrals,
    xi_analytic,
)


# --------------------------------------------------------------------------- #
# 1. analytic anchors
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("A,kappa", [(0.1, 0.2), (0.05, 0.1), (0.2, 0.45), (1.0, 1.0)])
def test_two_closed_forms_and_gradient_energy_integral_agree(A: float, kappa: float) -> None:
    sigma = sigma_analytic(A, kappa)
    xi = xi_analytic(A, kappa)
    # sigma = (2/3) sqrt(2 kappa A) == (4/3) kappa / xi  (algebraic identity)
    assert sigma == pytest.approx((4.0 / 3.0) * kappa / xi, rel=1e-12)
    # ... and the same number from a numerical integral of kappa |grad phi|^2
    assert sigma_from_gradient_energy(A, kappa) == pytest.approx(sigma, rel=1e-6)
    assert xi == pytest.approx(math.sqrt(2.0 * kappa / A), rel=1e-15)


def test_analytic_anchors_reject_invalid_parameters() -> None:
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            sigma_analytic(0.1, bad)
        with pytest.raises(ValueError):
            xi_analytic(bad, 0.2)


# --------------------------------------------------------------------------- #
# 2. radius recovery from a synthetic droplet
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("R,xi", [(10.0, 2.0), (16.0, 3.0), (22.0, 3.5)])
def test_radius_and_width_are_recovered_from_a_synthetic_droplet(R: float, xi: float) -> None:
    N = 96
    phi = droplet_phi0(N, R, xi)
    radius, _dx, _dy = _radius_grid(N)
    r, phi_mean = radial_average(phi, radius, bin_width=0.1)[:2]
    assert radius_from_contour(r, phi_mean) == pytest.approx(R, abs=0.1)
    R_fit, xi_fit = radius_and_width_from_tanh_fit(r, phi_mean, band=0.9)
    assert R_fit == pytest.approx(R, abs=0.1)
    assert xi_fit == pytest.approx(xi, rel=0.02)


def test_radius_from_contour_rejects_profiles_without_a_crossing() -> None:
    r = torch.linspace(0.5, 20.0, 40, dtype=torch.float64)
    with pytest.raises(ValueError):
        radius_from_contour(r, torch.ones_like(r))
    with pytest.raises(ValueError):
        radius_and_width_from_tanh_fit(r, torch.zeros_like(r))


# --------------------------------------------------------------------------- #
# 3. pressure-tensor projection and the analytic tension integral
# --------------------------------------------------------------------------- #
def test_polar_projection_preserves_the_tensor_trace() -> None:
    N = 48
    radius, dx, dy = _radius_grid(N)
    phi = droplet_phi0(N, 12.0, 2.0)
    gx, gy, lap = grad_lap(phi)
    rho = torch.ones_like(phi)
    Pxx, Pyy, Pxy = pressure_tensor_analytic(rho, phi, gx, gy, lap, 0.1, 0.2)
    Prr, Ptt = polar_pressure_components(Pxx, Pyy, Pxy, dx, dy)
    assert torch.allclose(Prr + Ptt, Pxx + Pyy, atol=1e-12, rtol=1e-12)
    # radial anisotropy of the free-energy tensor is kappa |grad phi|^2 for a radial profile
    anis_expected = 0.2 * (gx * gx + gy * gy)
    # exact algebraically; the tolerance absorbs cancellation where gx^2 ~ gy^2
    assert torch.allclose(Prr - Ptt, anis_expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("R,A,kappa", [(20.0, 0.1, 0.2), (26.0, 0.05, 0.225)])
def test_analytic_pressure_tensor_integrates_to_the_analytic_tension(
    R: float, A: float, kappa: float
) -> None:
    # the profile must use the xi that belongs to (A, kappa), otherwise the analytic
    # tensor and the synthetic field describe different free energies
    xi = xi_analytic(A, kappa)
    N = 128
    phi = droplet_phi0(N, R, xi)
    radius, dx, dy = _radius_grid(N)
    gx, gy, lap = grad_lap(phi)
    rho = torch.ones_like(phi)
    Pxx, Pyy, Pxy = pressure_tensor_analytic(rho, phi, gx, gy, lap, A, kappa)
    Prr, Ptt = polar_pressure_components(Pxx, Pyy, Pxy, dx, dy)
    r_max = 0.5 * N - 1.0
    r_bins, anis = radial_anisotropy_profile(Prr, Ptt, radius, bin_width=0.25, r_max=r_max)
    sigma_ref = sigma_analytic(A, kappa)
    out = tension_integrals(
        r_bins, anis, r_lo=R - 4.0 * xi, r_hi=R + 4.0 * xi, bin_width=0.25
    )
    # one interface -> sigma, never 2 sigma.  The nine-point stencil under-resolves a
    # thin interface (xi = 2 is ~4 cells), which costs several percent -- the band is
    # 10 % here and the resolution trend is what the production run grades.
    assert out["sigma_integral"] == pytest.approx(sigma_ref, rel=0.10)
    # the anisotropy kernel's first moment is the surface of tension, within O(xi^2/R)
    assert abs(out["R_surface_of_tension"] - R) < 0.15 * xi
    # the capillary integral reproduces sigma/R to the same order
    assert out["dp_tensor"] == pytest.approx(sigma_ref / R, rel=0.15)


# --------------------------------------------------------------------------- #
# 4. the double-count guard actually rejects double counting
# --------------------------------------------------------------------------- #
def test_tension_integral_rejects_a_window_spanning_two_interfaces() -> None:
    r = torch.arange(0.125, 60.0, 0.25, dtype=torch.float64)
    lobe = torch.exp(-((r - 20.0) ** 2) / 2.0) + torch.exp(-((r - 40.0) ** 2) / 2.0)
    with pytest.raises(ValueError, match="lobe"):
        tension_integrals(r, lobe, r_lo=0.0, r_hi=60.0, bin_width=0.25)
    # one-sided window: exactly one interface
    one = tension_integrals(r, lobe, r_lo=10.0, r_hi=30.0, bin_width=0.25)
    two = one["sigma_integral"]
    assert two == pytest.approx(math.sqrt(2.0 * math.pi), rel=0.1)


def test_tension_integral_rejects_degenerate_inputs() -> None:
    r = torch.arange(0.125, 10.0, 0.25, dtype=torch.float64)
    zeros = torch.zeros_like(r)
    with pytest.raises(ValueError):
        tension_integrals(r, zeros, r_lo=0.0, r_hi=10.0, bin_width=0.25)
    with pytest.raises(ValueError):
        tension_integrals(r, zeros, r_lo=5.0, r_hi=1.0, bin_width=0.25)
    with pytest.raises(ValueError):
        tension_integrals(r[:2], zeros[:2], r_lo=0.0, r_hi=1.0, bin_width=0.25)


# --------------------------------------------------------------------------- #
# 5. bulk-band extraction guards
# --------------------------------------------------------------------------- #
def test_bulk_pressure_jump_recovers_a_synthetic_jump() -> None:
    N = 64
    radius, _dx, _dy = _radius_grid(N)
    R, xi, dp_true = 12.0, 2.0, 1.5e-3
    p_out = 0.31
    field = torch.where(radius <= R, p_out + dp_true, torch.full_like(radius, p_out))
    p_in, p_out_meas, dp = bulk_pressure_jump(field, radius, R, xi, offset_xi=2.0)
    assert p_in == pytest.approx(p_out + dp_true, rel=1e-12)
    assert p_out_meas == pytest.approx(p_out, rel=1e-12)
    assert dp == pytest.approx(dp_true, rel=1e-12)


def test_bulk_pressure_jump_raises_instead_of_returning_nan() -> None:
    N = 48
    radius, _dx, _dy = _radius_grid(N)
    pressure = torch.zeros_like(radius)
    with pytest.raises(ValueError, match="outer bulk band"):
        bulk_pressure_jump(pressure, radius, 20.0, 2.0, offset_xi=3.0, r_max=24.0)  # no outer room
    with pytest.raises(ValueError, match="inner bulk band"):
        bulk_pressure_jump(pressure, radius, 1.0, 2.0, offset_xi=3.0)  # no inner region


# --------------------------------------------------------------------------- #
# 6. grading logic fails planted defects
# --------------------------------------------------------------------------- #
def _measures(sigma_capillary_rel_error: float, *, u_max: float = 1e-4) -> DropletMeasures:
    sigma = 0.13333333333333333
    R = 12.0
    dp = sigma * (1.0 + sigma_capillary_rel_error) / R
    return DropletMeasures(
        R_contour=R - 0.05,
        R_fit=R,
        xi_fit=2.0,
        R_sigma=R + 0.1,
        p_in=0.3,
        p_out=0.3 - dp,
        dp_bulk=dp,
        dp_tensor=sigma / R,
        sigma_integral=sigma,
        u_max=u_max,
        u_rms=u_max * 0.5,
        sigma_analytic=sigma,
        xi_analytic=2.0,
        radius_over_xi=R / 2.0,
        phi_drift=1e-12,
        rho_drift=1e-12,
    )


def test_grading_passes_a_two_percent_defect_and_accepts_the_stretch_band() -> None:
    case = DropletCase(N=64, R=12.0, steps=10)
    row = grade_case(case, _measures(0.02))
    assert row["passed"] is True
    assert row["checks"]["C1_sigma_capillary_vs_analytic"]["within_stretch"] is True
    assert grade_report([row])["verdict"] == "PASS"


def test_grading_fails_a_twenty_percent_tension_error() -> None:
    case = DropletCase(N=64, R=12.0, steps=10)
    row = grade_case(case, _measures(0.20))
    assert row["passed"] is False
    assert row["checks"]["C1_sigma_capillary_vs_analytic"]["within_stretch"] is False
    report = grade_report([row])
    assert report["verdict"] == "FAIL"
    assert report["within_owner_band"] is False
    assert any("C1_sigma_capillary_vs_analytic" in f for f in report["failing_checks"])


def test_grading_fails_a_runaway_spurious_current() -> None:
    case = DropletCase(N=64, R=12.0, steps=10)
    row = grade_case(case, _measures(0.0, u_max=0.1))
    assert row["checks"]["C6_spurious_current"]["passed"] is False
    assert row["passed"] is False


def test_criteria_band_defaults_match_the_owner_standard() -> None:
    criteria = LaplaceCriteria()
    assert criteria.owner_band == 0.15
    assert criteria.stretch_band < criteria.owner_band


# --------------------------------------------------------------------------- #
# 7. end-to-end smoke of the case runner on the production kernel
# --------------------------------------------------------------------------- #
def test_short_droplet_case_runs_and_conserves() -> None:
    case = DropletCase(N=48, R=9.0, steps=24, sample_every=8, tau_phi=1.0, Gamma=0.05)
    measures = run_droplet_case(case)
    assert math.isfinite(measures.dp_bulk)
    assert measures.R_fit == pytest.approx(9.0, abs=0.5)
    assert measures.phi_drift <= 1e-8
    assert measures.rho_drift <= 1e-10
    assert len(measures.history) == 3
    assert measures.profile and len(measures.profile) > 10


def test_case_configuration_validation() -> None:
    with pytest.raises(ValueError):
        DropletCase(N=8, R=4.0)
    with pytest.raises(ValueError):
        DropletCase(N=64, R=-1.0)
    with pytest.raises(ValueError):
        DropletCase(N=64, R=12.0, steps=0)
    with pytest.raises(ValueError):
        DropletCase(N=32, R=12.0, bulk_offset_xi=5.0)  # droplet fills the box


def _radius_grid(N: int, center: float = 0.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from tensorlbm.phasefield.free_energy_pressure import periodic_distance

    c = 0.5 * (N - 1)
    grid = torch.arange(N, dtype=torch.float64)
    dx = periodic_distance(grid.view(1, -1).expand(N, N).contiguous(), c, float(N))
    dy = periodic_distance(grid.view(-1, 1).expand(N, N).contiguous(), c, float(N))
    return torch.sqrt(dx * dx + dy * dy), dx, dy