"""Tests for :mod:`tensorlbm.maneuvering_cfd` (node IN59-N02).

The suite separates three kinds of evidence:

* geometry / non-dimensionalisation identities (exact algebra),
* numerical-operator tests (uniform flow preserved, momentum-exchange force
  identical to the repository's independent implementation),
* identification tests (synthetic round trip: a known derivative set must be
  recovered from forces the fitter was not shown).

None of these tests claims agreement with experiment; that is what the node
report's honest-FAIL section is for.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.maneuvering_cfd import (
    HullFlowConfig,
    KCS_MODEL_SCALE,
    body_axes_from_lattice,
    bounce_back_and_force,
    control_volume_momentum_balance,
    cross_flow_theory_derivatives,
    forward_mmg_forces,
    from_prime,
    identify_all,
    identify_circular_motion,
    identify_rudder,
    identify_static_drift,
    kcs_hull_mask,
    run_hull_flow,
    synthetic_roundtrip,
    to_prime,
)
from tensorlbm.obstacles import compute_obstacle_forces_3d
import tensorlbm.maneuvering_cfd as mc


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def test_kcs_voxel_geometry_matches_reference():
    hull = kcs_hull_mask((16, 32, 150), length_cells=120.0, midship=(75.0, 16.0), keel_z=4.5)
    v = hull.verification
    assert v.passed, v.as_dict()
    assert v.cb_rel_error < 0.05
    assert abs(v.cb - KCS_MODEL_SCALE["Cb"]) < 0.05 * KCS_MODEL_SCALE["Cb"]
    # Lpp / B / T ratios of the KCS form are respected to the cell size
    assert abs(v.length_cells - 120.0) <= 2.0
    assert abs(v.beam_cells / v.length_cells - KCS_MODEL_SCALE["B"] / KCS_MODEL_SCALE["L"]) < 0.02
    assert abs(v.draft_cells / v.length_cells - KCS_MODEL_SCALE["T"] / KCS_MODEL_SCALE["L"]) < 0.03


def test_geometry_check_fails_closed_on_coarse_grid():
    """A too-coarse hull must be rejected, not silently used for a force claim."""
    coarse = kcs_hull_mask((40, 60, 140), length_cells=48.0, midship=(70.0, 30.0), keel_z=12.0)
    assert coarse.verification.passed is False
    assert coarse.verification.cb_rel_error > 0.05


def test_rudder_plate_adds_solid_cells_and_breaks_port_starboard_symmetry():
    plain = kcs_hull_mask((16, 32, 150), length_cells=120.0, midship=(75.0, 16.0), keel_z=4.5)
    rudder = kcs_hull_mask(
        (16, 32, 150), length_cells=120.0, midship=(75.0, 16.0), keel_z=4.5, rudder_deg=35.0
    )
    assert rudder.n_solid > plain.n_solid
    mirror = kcs_hull_mask(
        (16, 32, 150), length_cells=120.0, midship=(75.0, 16.0), keel_z=4.5, rudder_deg=-35.0
    )
    assert mirror.n_solid == rudder.n_solid
    assert not torch.equal(rudder.mask, mirror.mask)


# --------------------------------------------------------------------------- #
# non-dimensionalisation / frame rotation
# --------------------------------------------------------------------------- #
def test_prime_roundtrip_and_scaling():
    X, Y, N = to_prime(10.0, -4.0, 2.0, rho=1.0, U=0.08, L=120.0, T=5.6)
    back = from_prime(X, Y, N, rho=1.0, U=0.08, L=120.0, T=5.6)
    assert back == pytest.approx((10.0, -4.0, 2.0), rel=1e-12)
    # doubles the inflow -> quadratically smaller primes
    X2, _, _ = to_prime(10.0, 0.0, 0.0, rho=1.0, U=0.16, L=120.0, T=5.6)
    assert X2 == pytest.approx(X / 4.0, rel=1e-12)


def test_body_axes_rotation():
    # zero yaw: identity
    assert body_axes_from_lattice(1.0, 2.0, 0.0) == pytest.approx((1.0, 2.0))
    # a hull yawed 90 deg to starboard sees the stream from the side: the
    # lattice +x force is pure -body-Y ... and body X is +lattice y
    X, Y = body_axes_from_lattice(1.0, 0.0, 90.0)
    assert X == pytest.approx(0.0, abs=1e-12)
    assert Y == pytest.approx(-1.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# numerical operators
# --------------------------------------------------------------------------- #
def test_bounce_back_force_matches_obstacle_module():
    """The in-loop force must be identical to the repository's independent one."""
    torch.manual_seed(0)
    shape = (8, 10, 12)
    f = torch.rand(19, *shape) * 0.1 + 0.01
    solid = torch.zeros(shape, dtype=torch.bool)
    solid[3:5, 4:6, 5:7] = True
    _, fx, fy, fz = bounce_back_and_force(f, solid)
    ref = compute_obstacle_forces_3d(f, solid)
    assert (fx, fy, fz) == pytest.approx(tuple(float(v) for v in ref), rel=1e-12, abs=1e-12)


def test_uniform_flow_is_preserved_by_the_boundary_pipeline():
    """A uniform stream with no hull must stay uniform (catches streaming-sign bugs)."""
    shape = (12, 16, 40)
    nz, ny, nx = shape
    solid = torch.zeros(shape, dtype=torch.bool)
    u_in, tau = 0.05, 0.6
    f = equilibrium3d(
        torch.ones(shape), torch.full(shape, u_in), torch.zeros(shape), torch.zeros(shape)
    )
    inlet = equilibrium3d(
        torch.ones((nz, ny, 1)),
        torch.full((nz, ny, 1), u_in),
        torch.zeros((nz, ny, 1)),
        torch.zeros((nz, ny, 1)),
    )[:, :, :, 0]
    for _ in range(60):
        rho, ux, uy, uz = macroscopic3d(f)
        f = f + (equilibrium3d(rho, ux, uy, uz) - f) / tau
        f = mc._stream(f)
        mc._zero_external_inflow(f)
        f, _, _, _ = bounce_back_and_force(f, solid)
        f[:, :, :, 0] = inlet
        f[:, :, :, -1] = f[:, :, :, -2]
        for axis in (0, 1):
            mc._specular_free_slip(f, axis, 0)
            mc._specular_free_slip(f, axis, 1)
    rho, ux, uy, uz = macroscopic3d(f)
    assert torch.allclose(rho, torch.ones_like(rho), atol=1e-6)
    assert torch.allclose(ux, torch.full_like(ux, u_in), atol=1e-6)
    assert float(uy.abs().max()) < 1e-6


def _short_drift_run(yaw_deg: float, steps: int = 260):
    shape = (14, 26, 70)
    hull = kcs_hull_mask(
        shape, length_cells=44.0, midship=(35.0, 13.0), keel_z=4.5, yaw_deg=yaw_deg
    )
    cfg = HullFlowConfig(
        shape=shape,
        mask=hull.mask,
        u_in=0.06,
        tau=0.6,
        steps=steps,
        converge_window=10_000,   # never early-stop: keeps the test short
        midship=(35.0, 13.0),
        keel_z=4.5,
    )
    return run_hull_flow(cfg)


def test_short_lbm_drift_is_antisymmetric_in_beta():
    """Port/starboard antisymmetry of the LBM forces about beta = 0.

    This is a convention-free physical check (the +beta and -beta geometries are
    exact lattice mirrors): ``X'`` must be even and ``Y'``/``N'`` odd in ``beta``.
    It also exercises the whole solver stack (streaming, bounce-back,
    momentum-exchange force and yaw moment) at a size that fits a fast test.
    """
    plus = _short_drift_run(+10.0)
    minus = _short_drift_run(-10.0)
    for res in (plus, minus):
        assert math.isfinite(res.fx) and math.isfinite(res.fy)
        assert math.isfinite(res.moment_yaw)
        assert res.steps_run == 260

    X_plus, Y_plus = body_axes_from_lattice(plus.fx, plus.fy, +10.0)
    X_minus, Y_minus = body_axes_from_lattice(minus.fx, minus.fy, -10.0)

    def rel(a: float, b: float) -> float:
        return abs(a - b) / max(abs(a), abs(b), 1e-12)

    # drag: even in beta (same sign, similar magnitude)
    assert X_plus > 0.0 and X_minus > 0.0
    assert rel(abs(X_plus), abs(X_minus)) < 0.25
    # side force and yaw moment: odd in beta
    assert Y_plus * Y_minus < 0.0
    assert rel(abs(Y_plus), abs(Y_minus)) < 0.25
    assert plus.moment_yaw * minus.moment_yaw < 0.0
    assert rel(abs(plus.moment_yaw), abs(minus.moment_yaw)) < 0.25


# --------------------------------------------------------------------------- #
# identification
# --------------------------------------------------------------------------- #
def test_static_drift_fit_recovers_polynomial():
    recs = []
    for b in (-20.0, -10.0, 0.0, 10.0, 20.0):
        beta = math.radians(b)
        recs.append(
            {
                "beta_deg": b,
                "X_prime": -0.02 + 0.12 * beta**2 + 0.9 * beta**4,
                "Y_prime": -0.6 * beta - 4.5 * beta**3,
                "N_prime": -0.11 * beta - 0.6 * beta**3,
            }
        )
    fits = identify_static_drift(recs)
    assert fits["Y"].coefficients["Yb"] == pytest.approx(-0.6, rel=1e-9)
    assert fits["Y"].coefficients["Ybbb"] == pytest.approx(-4.5, rel=1e-9)
    assert fits["N"].coefficients["Nb"] == pytest.approx(-0.11, rel=1e-9)
    assert fits["X"].coefficients["Xbbbb"] == pytest.approx(0.9, rel=1e-6)
    assert fits["X"].r2 == pytest.approx(1.0, abs=1e-12)


def test_identification_rejects_degenerate_sweeps():
    with pytest.raises(ValueError):
        identify_static_drift([])
    zero_rudder = [
        {"delta_deg": 0.0, "X_prime": 0.0, "Y_prime": 0.0, "N_prime": 0.0}
    ]
    with pytest.raises(ValueError):
        identify_rudder(zero_rudder)
    zero_circular = [{"r_prime": 0.0, "X_prime": 0.0, "Y_prime": 0.0, "N_prime": 0.0}]
    with pytest.raises(ValueError):
        identify_circular_motion(zero_circular)


def test_synthetic_roundtrip_noiseless_and_noisy():
    clean = synthetic_roundtrip()
    assert clean["max_relative_error"] < 1e-10
    noisy = synthetic_roundtrip(noise=0.02, seed=1)
    assert noisy["max_relative_error"] < 0.10
    # noise must be visible: a 2 % perturbation cannot reproduce the truth exactly
    assert noisy["max_relative_error"] > 1e-6


def test_forward_model_reproduces_the_identified_force_model():
    clean = synthetic_roundtrip()
    table = clean["recovered"]
    X, Y, N = forward_mmg_forces(table, beta_deg=8.0, r_prime=0.2, delta_deg=-15.0)
    assert math.isfinite(X) and math.isfinite(Y) and math.isfinite(N)
    assert Y < 0.0  # beta > 0 -> port-side force
    assert N < 0.0


def test_cross_flow_theory_baseline_is_signed_and_flagged():
    theory = cross_flow_theory_derivatives()
    assert theory["Ybbb"] < 0.0
    assert theory["Ybbb_band"][0] <= theory["Ybbb"] <= theory["Ybbb_band"][1]
    # the analytic KCS half-breadth used here is fore-aft symmetric, so the strip
    # integral cannot produce a bow/stern moment -- documented, not hidden
    assert abs(theory["Nbbb"]) < 1e-9
    assert "assumptions" in theory


def test_control_volume_balance_is_finite():
    shape = (8, 10, 20)
    f = equilibrium3d(
        torch.ones(shape), torch.full(shape, 0.05), torch.zeros(shape), torch.zeros(shape)
    )
    out = control_volume_momentum_balance(f, 0.05)
    assert all(math.isfinite(v) for v in out.values())