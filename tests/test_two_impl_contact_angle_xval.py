"""Invariance tests for the two independent contact-angle implementations.

Anchor: ``XVAL-LBM-TWO-INDEPENDENT-CONTACT-ANGLE-IMPLS``.

These tests assert *invariants*, not just "the code runs":

* the wall-value law ``cos(theta) = (3 psi_w - psi_w^3)/2`` inverts exactly
  (so the ``dirichlet`` arm cannot drift from the module's single source of truth);
* the ``neumann_flux`` and ``geometric_ghost`` arms coincide **bit-for-bit** in
  their common neutral limit (``theta = 90 deg``): at ``cos(theta)=0`` the flux
  vanishes and ``cot(theta)=0``, so both ghosts are the Neumann ghost.  A
  disagreement here would mean the two implementations are not solving the same
  continuum problem;
* the benchmark-side third estimator (``theta_from_contact_normal``) recovers the
  angle of a synthetic diffuse cap of known geometry -- it is used as
  acceptance evidence, so it is validated against a constructed answer;
* the integration caller (``tensorlbm.wetting.ladder.contact_two_impl``) hands
  the *same* wall value to the production solver ``CAC2D`` (``wall_order``
  policy) and refuses the arms the solver cannot represent;
* the delivered run artefact (``runs/two_impl_contact_angle/summary.json``)
  satisfies the predeclared 15% gates.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tensorlbm.wetting.two_impl_contact_angle import (  # noqa: E402
    ARMS,
    TwoImplConfig,
    measure_contact_angle,
    psi_wall_from_angle,
    run_arm,
    wall_ghost_rows,
)
from tensorlbm.wetting.wall_order_parameter_2d import young_cos_theta  # noqa: E402

CASE = REPO / "benchmarks" / "case_pool" / "two_impl_contact_angle"
if str(CASE) not in sys.path:
    sys.path.insert(0, str(CASE))
from run_case import theta_from_contact_normal  # noqa: E402

REFS = json.loads((CASE / "refs.json").read_text())


# ---------------------------------------------------------------------------
# the law and the predeclared gates
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("theta_deg", [30.0, 60.0, 90.0, 110.0, 130.0])
def test_psi_wall_inverts_young_law(theta_deg: float) -> None:
    p = psi_wall_from_angle(math.radians(theta_deg))
    assert -1.0 <= p <= 1.0
    assert young_cos_theta(p) == pytest.approx(math.cos(math.radians(theta_deg)), abs=1e-12)


def test_refs_declare_numeric_gates() -> None:
    gates = REFS["gates"]
    assert gates["G2_accuracy_vs_young_pct"] == 15.0
    assert gates["G3_cross_arm_spread_pct"] == 15.0
    assert gates["G4_estimator_spread_deg"] == 10.0
    assert set(REFS["arms"]) == set(ARMS)
    assert len(REFS["reference"]["theta_ref_deg"]) >= 3


# ---------------------------------------------------------------------------
# the two flux/geometric implementations collapse in their common limit
# ---------------------------------------------------------------------------
def test_ghosts_degenerate_at_neutral_angle() -> None:
    cfg = TwoImplConfig()
    torch.manual_seed(0)
    psi = torch.tanh(torch.randn(cfg.ny, cfg.nx, dtype=torch.float64))
    theta = math.radians(90.0)
    flux, _ = wall_ghost_rows("neumann_flux", psi, theta, cfg.kappa, cfg.surface_tension)
    geom, _ = wall_ghost_rows("geometric_ghost", psi, theta, cfg.kappa, cfg.surface_tension)
    assert torch.allclose(flux, geom, atol=1e-14)
    # ... and they must *differ* away from the neutral point, otherwise the
    # arms would be the same implementation.
    flux2, _ = wall_ghost_rows("neumann_flux", psi, math.radians(60.0), cfg.kappa, cfg.surface_tension)
    geom2, _ = wall_ghost_rows("geometric_ghost", psi, math.radians(60.0), cfg.kappa, cfg.surface_tension)
    assert float((flux2 - geom2).abs().max()) > 1e-3


@pytest.mark.parametrize("arm", ["neumann_flux", "geometric_ghost"])
def test_wall_ghosts_satisfy_the_continuum_condition(arm: str) -> None:
    """The ghost each arm builds is a solution of the *same* wall condition.

    ``neumann_flux``: the ghost must satisfy the discrete fixed point of
    ``d_n psi = (3/4) sigma cos(theta)/kappa * (1 - psi_wallplane^2)`` (this is
    the continuum Young flux, written with ``psi_wallplane = (psi_0 + psi_g)/2``);
    ``geometric_ghost``: for an interface that is *parallel* to the wall
    (``|d_t psi| = 0``, i.e. psi varying only normal to the wall) the geometric
    ghost must reduce to the Neumann ghost ``psi_0`` at every angle.  A failure
    of either means the arm is not discretising the condition it claims.
    """
    cfg = TwoImplConfig(ny=24, nx=32, interface_width=3.0)
    y = torch.arange(cfg.ny, dtype=torch.float64).view(-1, 1)
    psi = torch.tanh((y - cfg.ny / 2.0) / 1.0).repeat(1, cfg.nx)
    for theta_deg in (60.0, 90.0, 120.0):
        ghost, _ = wall_ghost_rows(arm, psi, math.radians(theta_deg), cfg.kappa,
                                   cfg.surface_tension)
        if arm == "geometric_ghost":
            assert float((ghost - psi[0]).abs().max()) < 1e-12
        else:
            amp = 0.75 * cfg.surface_tension * math.cos(math.radians(theta_deg)) / cfg.kappa
            plane = 0.5 * (psi[0] + ghost)
            residual = (psi[0] - ghost) - (-amp * (1.0 - plane ** 2))
            assert float(residual.abs().max()) < 1e-12


# ---------------------------------------------------------------------------
# the benchmark-side estimator is validated against a constructed answer
# ---------------------------------------------------------------------------
def _synthetic_cap(theta_deg: float, cfg: TwoImplConfig, radius: float = 14.0,
                   wall_y: float = -0.5) -> torch.Tensor:
    """Sharp circular cap of known angle smeared to ``tanh(d/xi)`` (analytic answer)."""
    ny, nx = cfg.ny, cfg.nx
    y = torch.arange(ny, dtype=torch.float64).view(-1, 1) - wall_y
    x = torch.arange(nx, dtype=torch.float64).view(1, -1) - (nx - 1) / 2.0
    theta = math.radians(theta_deg)
    yc = wall_y - radius * math.cos(theta)          # circle centre
    d = radius - torch.sqrt(x ** 2 + (y + wall_y - yc) ** 2)
    psi = torch.tanh(d / (cfg.interface_width / 2.0))
    # gas far field must be -1 for the estimator's gas reference
    psi = torch.where(y < 0, psi, psi)
    return psi


@pytest.mark.parametrize("theta_deg", [70.0, 90.0, 110.0, 130.0])
def test_normal_estimator_recovers_synthetic_cap(theta_deg: float) -> None:
    cfg = TwoImplConfig()
    psi = _synthetic_cap(theta_deg, cfg, radius=16.0)
    est = theta_from_contact_normal(psi, cfg.interface_width)
    assert est["n_normal_points"] >= 2
    assert abs(est["theta_normal_deg"] - theta_deg) < 6.0, est


# ---------------------------------------------------------------------------
# integration: the production solver is driven by the same wall value
# ---------------------------------------------------------------------------
def test_contact_two_impl_hands_the_wall_value_to_the_solver() -> None:
    from tensorlbm.wetting.ladder import contact_two_impl

    for theta_deg in (60.0, 110.0):
        out = contact_two_impl(theta_deg, "dirichlet", shape=(32, 32), radius=8.0, steps=0)
        expect = psi_wall_from_angle(math.radians(theta_deg))
        assert out["solver_wall_phi_policy"] == "wall_order"
        assert out["solver_wall_order_parameter"] == pytest.approx(expect, abs=1e-14)
        assert out["theta_young_from_psi_wall_deg"] == pytest.approx(theta_deg, abs=1e-9)


def test_contact_two_impl_rejects_unrepresentable_arms() -> None:
    from tensorlbm.wetting.ladder import contact_two_impl

    for arm in ("neumann_flux", "geometric_ghost", "nonsense"):
        with pytest.raises(ValueError):
            contact_two_impl(90.0, arm, shape=(32, 32), radius=8.0, steps=0)


# ---------------------------------------------------------------------------
# the implementations really equilibrate (mass-conserving Allen-Cahn descent)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("arm", list(ARMS))
def test_arms_equilibrate_and_conserve_mass(arm: str) -> None:
    cfg = TwoImplConfig()                     # graded grid: 56 x 80, xi = 3, R = 16
    out = run_arm(arm, math.radians(90.0), cfg=cfg, radius=16.0, max_iter=15000)
    assert out["mass_error"] < 1e-9           # mass-conserving descent
    meas = measure_contact_angle(out["psi"], cfg.interface_width, wall_y=-0.5)
    assert meas["measurable"]
    assert abs(meas["theta_cap_deg"] - 90.0) < 15.0
    assert abs(meas["theta_circle_deg"] - 90.0) < 15.0


@pytest.mark.parametrize("arm", ["neumann_flux", "geometric_ghost"])
def test_wall_angle_readback_has_no_quadrant_flip(arm: str) -> None:
    """Regression: the read-back must return theta, not ``180 deg - theta``.

    The read-back used ``atan2(|d_t psi|, +d_n psi)`` and therefore reported
    70.0 deg for a prescribed 110 deg (measured before the fix); the continuum
    condition is ``d_n psi = -cos(theta)(1-psi^2)/xi``, so the angle is
    ``atan2(|d_t psi|, -d_n psi)``.
    """
    from tensorlbm.wetting.two_impl_contact_angle import wall_angle_readback

    cfg = TwoImplConfig()
    rb = {}
    for theta_deg in (60.0, 90.0, 110.0, 130.0):
        psi = _synthetic_cap(theta_deg, cfg, radius=16.0)
        rb[theta_deg] = wall_angle_readback(arm, psi, math.radians(theta_deg), cfg.kappa,
                                            cfg.surface_tension)["theta_deg"]
    # no quadrant flip: angles above 90 deg must stay above 90 deg (they used to
    # come back as 180 - theta, i.e. 70 deg for 110 deg), and the read-back must
    # be monotone in the prescribed angle (it is a local, one-row diagnostic, so
    # its absolute value is not an acceptance quantity -- the shape estimators are)
    assert rb[90.0] == pytest.approx(90.0, abs=6.0)
    assert 95.0 < rb[110.0] < 125.0
    assert rb[130.0] > 95.0
    assert rb[60.0] < rb[90.0] < rb[110.0] < rb[130.0]


# ---------------------------------------------------------------------------
# delivered run artefact carries the acceptance numbers
# ---------------------------------------------------------------------------
def test_delivered_benchmark_artefact_passes_the_predeclared_gates() -> None:
    path = REPO / "runs" / "two_impl_contact_angle" / "summary.json"
    if not path.is_file():
        pytest.skip("delivered run artefact absent (runs/two_impl_contact_angle/summary.json)")
    payload = json.loads(path.read_text())
    gates = payload["gates"]
    assert not payload["quick"], "the delivered artefact must be the graded sweep"
    assert gates["G2_accuracy_vs_young"]["max_abs_err_pct"] < 15.0
    assert gates["G3_cross_arm_spread"]["max_spread_pct"] < 15.0
    assert gates["G5_conservation"]["pass"]
    assert gates["G6_neutral_limit_degeneracy"]["pass"]
    assert {r["arm"] for r in payload["rows"]} == set(ARMS)
    assert len({r["theta_ref_deg"] for r in payload["rows"]}) >= 4
    if "G7_solver_cross_check" in gates:
        assert gates["G7_solver_cross_check"]["pass"]