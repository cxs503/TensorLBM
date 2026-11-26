"""D3Q19 coverage + measured-order regression for the regularised wall closure.

Anchor: ``XVAL-LBM-REGULARISED-BC-NONEQUILIBRIUM-STRESS-REC`` (xval arm R4).

``tests/test_regularised_bc.py`` pins the closure semantics on D2Q9 and pins the
D3Q19 *tables*; nothing there checks that the D3Q19 path is physically right or
that the "second order" claim is measurable.  The tests below add

* the projection invariants on the D3Q19 velocity set (moments kept, third
  moment of the non-equilibrium part removed, idempotence);
* the preregistered phase metric definition (masked to ``|A_ref| > 0.1 U0``);
* a two-point order-of-accuracy regression on the analytic Stokes layer;
* a 3-D D3Q19 body-force Poiseuille channel against its analytic parabola.

The heavy sweeps (8-point ladder, tau sweep) live in
``development/verification/regularised_bc_noneq_stress/run_xval_r4.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from tensorlbm import regularised_bc as rb

XVAL = Path(__file__).resolve().parents[1] / "development" / "verification" / "regularised_bc_noneq_stress"
D3_NORMALS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def _random_d3(seed: int = 7, shape: tuple[int, ...] = (3, 2, 2)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    f = np.ones((19,) + shape) * rb.D3Q19.w[:, None, None, None]
    return f + 0.02 * rng.standard_normal((19,) + shape)


@pytest.mark.parametrize("normal", D3_NORMALS)
def test_d3q19_projection_keeps_moments_and_stress(normal):
    """f -> feq + w/(2cs^4) Q:S keeps rho, u and S and drops the third moment."""
    spec = rb.D3Q19
    f = _random_d3()
    rho, u = rb.macroscopic(spec, f)
    out = rb.project_second_order(spec, f, rho=rho, u=u)
    rho2, u2 = rb.macroscopic(spec, out)
    assert np.allclose(rho2, rho, atol=1e-14)
    assert np.allclose(u2, u, atol=1e-14)
    s1 = rb.nonequilibrium_stress(spec, f, rho=rho, u=u)
    s2 = rb.nonequilibrium_stress(spec, out, rho=rho2, u=u2)
    assert np.allclose(s2, s1, atol=1e-14)
    neq = out - rb.equilibrium(spec, rho2, u2)
    assert np.max(np.abs(rb.third_moment(spec, neq))) < 1e-14
    # idempotent: a distribution that is already a second-order Hermite
    # polynomial is a fixed point of the projection
    assert np.allclose(rb.project_second_order(spec, out), out, atol=1e-14)


@pytest.mark.parametrize("normal", D3_NORMALS)
def test_d3q19_completed_unknowns_are_antisymmetric(normal):
    """``f_q = f_q^eq + (f_opp - f_opp^eq)`` for c_q.n = +1, known links untouched."""
    spec = rb.D3Q19
    f = _random_d3(seed=3)
    rho = np.ones(f.shape[1:])
    u = np.stack([0.01 * np.ones(f.shape[1:]) for _ in range(3)])
    out = rb.complete_unknowns(spec, f, np.asarray(normal), rho=rho, u=u)
    feq = rb.equilibrium(spec, rho, u)
    cn = spec.c @ np.asarray(normal)
    d = out - feq
    # the unknown direction inherits the *same* non-equilibrium content as its
    # mirror (that is what "anti-symmetric completion" means for the stress:
    # d_q = d_opp while the populations themselves are mirrored)
    for q in np.nonzero(cn == 1)[0]:
        o = spec.opposite[q]
        assert np.allclose(d[q], d[o], atol=1e-15)
    known = cn != 1
    assert np.array_equal(out[known], f[known])


def _oscillating(delta: float, tau: float = 0.8) -> dict:
    nu = (tau - 0.5) / 3.0
    omega = 2.0 * nu / delta**2
    period = 2.0 * np.pi / omega
    cfg = rb.ChannelConfig(
        case="oscillating_wall",
        wall_bc="regularised",
        ny=int(round(max(20.0, 10.0 * delta))) + 2,
        nx=2,
        tau=tau,
        u_wall=0.05,
        omega=omega,
        steps=int(round(4.0 * period)),
        check_every=200,
        tol=1e-10,
        init="rest",
    )
    return rb.run_channel(cfg)


def test_phase_metric_is_masked_to_the_preregistered_support():
    """The far field must not dominate the phase error (|A_ref| < 0.1 U0 excluded).

    ``PREREG.md`` declares ``phase_max_abs`` on ``|A_ref| > 0.1 U0``; the
    unmasked maximum is the phase of two numerical zeros and is O(1) rad at any
    resolution, so a metric without the mask measures nothing.
    """
    rec = _oscillating(2.0)
    assert np.isfinite(rec["phase_max_abs_masked"])
    assert rec["phase_max_abs_masked"] < 0.1
    assert rec["phase_max_abs"] > 3.0 * rec["phase_max_abs_masked"]


def test_stokes_layer_error_is_second_order_in_delta():
    """Two-point order of accuracy of the regularised closure (analytic Stokes layer).

    R4 measured 1.81-2.00 over tau in {0.8, 1.0, 1.5}; the coarse two-point
    version here keeps the regression cheap and only asserts that the error
    really falls like (Delta_x/delta)^2 rather than saturating at a closure bias.
    """
    coarse = _oscillating(3.0)["l2_rel_complex"]
    fine = _oscillating(6.0)["l2_rel_complex"]
    slope = np.log(coarse / fine) / np.log(2.0)
    assert 1.5 < slope < 2.5, f"observed order {slope:.2f} is not second order"
    assert fine < 0.01, f"delta = 6 cells: l2 = {100 * fine:.3f}% is not below 1%"


def test_d3q19_poiseuille_channel_matches_analytic():
    """3-D D3Q19 body-force Poiseuille through the same closure (physical path)."""
    sys.path.insert(0, str(XVAL))
    import run_xval_r4 as R4  # noqa: E402  (xval runner shipped next to the tests)

    nu = (0.8 - 0.5) / 3.0
    ny = 16
    g = 8.0 * nu * 0.05 / (ny - 1.0) ** 2
    err = {}
    for wall_bc in ("regularised", "bounce_back"):
        rec = R4.channel_3d(
            "poiseuille", ny=ny, tau=0.8, wall_bc=wall_bc, body_force=g,
            init="analytic", tol=1e-7, max_steps=6000,
        )
        assert rec["converged"], wall_bc
        err[wall_bc] = rec["l2_rel"]
    for wall_bc, value in err.items():
        assert value < 0.03, f"{wall_bc}: 3-D Poiseuille l2 = {100 * value:.2f}% >= 3%"
    assert abs(rec["mass_drift"]) < 1e-10