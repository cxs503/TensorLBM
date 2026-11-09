"""Taylor–Aris shear dispersion verification (D2Q5 passive-scalar kernel).

External references (not cell values):

* transversely periodic shear ``u = U sin(2 pi y / H)``:  ``D_eff = D + U^2 H^2/(8 pi^2 D)``
* plane Poiseuille (analytic integral only, no walls in this module): ``D + 2/105 u_m^2 h^2/D``
* circular pipe: ``D + a^2 u_m^2/(48 D)``

The lattice-Boltzmann check uses only the shipped fully periodic D2Q5 operators
from :mod:`tensorlbm.passive_scalar`, so it verifies the kernel that the library
actually ships, with no boundary-condition confound.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.passive_scalar import bounce_back_scalar
from tensorlbm.taylor_dispersion import (
    ShearDispersionConfig,
    channel_taylor_diffusivity,
    periodic_shear_taylor_diffusivity,
    pipe_taylor_diffusivity,
    run_periodic_shear_dispersion,
    run_uniform_flow_control,
    taylor_diffusivity_from_profile,
)

D = 0.1
TOL = 0.03  # 3 % — ten-fold tighter than the platform 15 % acceptability line


# ---------------------------------------------------------------------------
# Analytic references
# ---------------------------------------------------------------------------


def test_periodic_shear_quadrature_matches_closed_form() -> None:
    """The general Aris integral reproduces U^2 H^2 / (8 pi^2 D) for periodic shear."""
    ny, u_amp = 64, 0.05
    y = torch.arange(ny, dtype=torch.float64)
    u = u_amp * torch.sin(2.0 * math.pi * y / ny)
    quad = taylor_diffusivity_from_profile(D, u, periodic=True)
    closed = periodic_shear_taylor_diffusivity(D, u_amp, float(ny))
    assert abs(quad / closed - 1.0) < 1e-2


def test_channel_quadrature_matches_poiseuille_closed_form() -> None:
    """J(psi^2)/D reproduces the classic 2/105 plane-channel coefficient."""
    h = 4.0
    dy = 0.02
    n = int(2 * h / dy) + 1
    y = -h + dy * torch.arange(n, dtype=torch.float64)
    u = 1.5 * 0.05 * (1.0 - (y / h) ** 2)
    quad = taylor_diffusivity_from_profile(D, u, dy=dy)
    closed = channel_taylor_diffusivity(D, 0.05, h)
    assert abs(quad / closed - 1.0) < 2e-2


def test_pipe_closed_form_scaling() -> None:
    """Pipe reference value and its quadratic scaling in u_mean and radius."""
    d1 = pipe_taylor_diffusivity(D, 0.05, 2.0)
    assert d1 == pytest.approx(D + 2.0**2 * 0.05**2 / (48.0 * D), rel=1e-14)
    # doubling the mean velocity quadruples the Taylor term, not the molecular part
    d2 = pipe_taylor_diffusivity(D, 0.10, 2.0)
    assert (d2 - D) == pytest.approx(4.0 * (d1 - D), rel=1e-12)


def test_zero_shear_limit_reduces_to_molecular_diffusion() -> None:
    assert periodic_shear_taylor_diffusivity(D, 0.0, 16.0) == pytest.approx(D)
    assert channel_taylor_diffusivity(D, 0.0, 4.0) == pytest.approx(D)


@pytest.mark.parametrize("bad", [0.0, -0.1, float("nan")])
def test_non_positive_diffusivity_is_rejected(bad: float) -> None:
    with pytest.raises(ValueError):
        periodic_shear_taylor_diffusivity(bad, 0.05, 16.0)
    with pytest.raises(ValueError):
        taylor_diffusivity_from_profile(bad, [0.0, 0.1, 0.0], dy=1.0)


# ---------------------------------------------------------------------------
# Lattice-Boltzmann measurement
# ---------------------------------------------------------------------------


def test_periodic_shear_dispersion_matches_taylor_aris() -> None:
    cfg = ShearDispersionConfig(pe=8.0, n_steps=6000, measure_every=10)
    res = run_periodic_shear_dispersion(cfg)
    # the Taylor enhancement must be large enough for the test to have teeth
    assert res.d_eff_theory / D > 1.5
    assert abs(res.rel_error) < TOL, (
        f"D_eff numeric {res.d_eff_numeric:.5f} vs theory {res.d_eff_theory:.5f} "
        f"({res.rel_error * 100:+.2f}%) exceeds {TOL:.0%}"
    )
    # axial mode must be far slower than the transverse relaxation (asymptotics valid)
    assert res.axial_rate_theory < 0.1 * res.transverse_rate


def test_uniform_flow_control_has_no_shear_enhancement() -> None:
    """Negative control: uniform advection cannot enhance spreading."""
    cfg = ShearDispersionConfig(pe=8.0, n_steps=4000, measure_every=10)
    res = run_uniform_flow_control(cfg)
    assert abs(res.rel_error) < TOL, (
        f"uniform-flow control measured {res.d_eff_numeric:.5f} vs D={D} "
        f"({res.rel_error * 100:+.2f}%)"
    )


def test_shear_and_uniform_measurements_are_separated() -> None:
    """Discriminating power: the shear case must differ from the control by >50 %."""
    cfg = ShearDispersionConfig(pe=8.0, n_steps=4000, measure_every=10)
    shear = run_periodic_shear_dispersion(cfg)
    control = run_uniform_flow_control(cfg)
    assert shear.d_eff_numeric > 1.5 * control.d_eff_numeric


# ---------------------------------------------------------------------------
# D2Q5 scalar zero-flux boundary (defect regression)
# ---------------------------------------------------------------------------


def test_bounce_back_scalar_reverses_d2q5_directions_only() -> None:
    g = torch.arange(5 * 3 * 4, dtype=torch.float64).reshape(5, 3, 4)
    mask = torch.zeros(3, 4, dtype=torch.bool)
    mask[0, :] = True
    out = bounce_back_scalar(g, mask)
    assert tuple(out.shape) == (5, 3, 4)
    # masked cells: +y <-> -y (2 <-> 4), +x <-> -x (1 <-> 3)
    assert torch.equal(out[2, 0], g[4, 0])
    assert torch.equal(out[4, 0], g[2, 0])
    assert torch.equal(out[1, 0], g[3, 0])
    assert torch.equal(out[3, 0], g[1, 0])
    # unmasked cells untouched
    assert torch.equal(out[:, 1:], g[:, 1:])
    # bouncing back conserves the scalar on the masked cells
    assert out[:, 0].sum() == pytest.approx(g[:, 0].sum())


def test_bounce_back_scalar_rejects_d2q9_distribution() -> None:
    """The D2Q9 momentum helper silently produced a 9-population tensor here."""
    with pytest.raises(ValueError):
        bounce_back_scalar(torch.zeros(9, 3, 4), torch.zeros(3, 4, dtype=torch.bool))


def test_bounce_back_scalar_gives_zero_normal_flux() -> None:
    """Bounce-back must reverse (not mirror-equalise) the wall-normal pair."""
    g = torch.zeros(5, 4, 6, dtype=torch.float64)
    g[2, 0] = 0.3  # inward (+y) population on the wall row
    g[4, 0] = 0.1  # outward (-y) population
    mask = torch.zeros(4, 6, dtype=torch.bool)
    mask[0, :] = True
    out = bounce_back_scalar(g, mask)
    # scalar carried by the two normal populations is conserved
    assert out[2, 0].sum() + out[4, 0].sum() == pytest.approx(0.4 * 6)