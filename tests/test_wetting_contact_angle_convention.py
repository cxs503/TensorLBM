"""Convention tests for the geometrical (gradient-rotation) wetting BC.

Round 1 of ``XVAL-LBM-GEOMETRIC-WETTING-CONTACT-ANGLE`` could not see the
convention defect because both of its checks were self-referential: one compared
the port with a literal transcription of the reference kernel (faithfulness, not
correctness), the other read the angle back out of the rotated gradient with
``acos(g_hat . n_s)`` -- the very quantity the kernel sets.

The tests below use an *independent* geometric truth: an equilibrium sessile
drop is a spherical cap, so its own outward normal at the contact line is known
in closed form.  See :mod:`tensorlbm.cg_wetting_convention` for the derivation of

    angle(g_hat, n_s) = 180 - theta_physical .

What is pinned here
-------------------
1. the analytic-cap check passes for the physical convention and fails, with
   exactly the mirror error ``|180 - 2 theta|``, for the reference convention;
2. ``theta = 90`` is the degenerate point where both conventions agree (which is
   why a single 90 deg validation arm cannot detect the defect);
3. the solver entry point maps the angle as documented, and keeps the previous
   numeric behaviour available behind ``contact_angle_convention="reference"``.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.cg_wetting_convention import (
    CAP_FLOOR_DEG,
    cap_convention_error,
    physical_to_reference_angle,
)
from tensorlbm.color_gradient2d import color_gradient_two_tau_step
from tensorlbm.cg_wetting_geometric import sessile_drop_initial

ANGLES = (30.0, 60.0, 90.0, 120.0, 150.0)


# ---------------------------------------------------------------------------
# The map itself
# ---------------------------------------------------------------------------
def test_physical_to_reference_is_the_mirror() -> None:
    assert physical_to_reference_angle(60.0) == pytest.approx(120.0)
    assert physical_to_reference_angle(120.0) == pytest.approx(60.0)
    assert physical_to_reference_angle(90.0) == pytest.approx(90.0)
    # reference convention is a raw pass-through
    assert physical_to_reference_angle(60.0, "reference") == pytest.approx(60.0)


def test_physical_to_reference_rejects_out_of_range() -> None:
    for bad in (0.0, 180.0, -5.0, 200.0):
        with pytest.raises(ValueError):
            physical_to_reference_angle(bad)
    with pytest.raises(ValueError):
        physical_to_reference_angle(60.0, "mirror")


# ---------------------------------------------------------------------------
# Analytic cap: independent geometric truth
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("theta", ANGLES)
def test_physical_convention_matches_analytic_cap(theta: float) -> None:
    """The rotated gradient must reproduce the analytic cap normal."""
    c = cap_convention_error(theta, "physical")
    assert c["n_samples"] > 5, c
    # the kernel sets exactly 180 - theta (tight, it is an algebraic identity)
    assert c["kernel_exact_angle_vs_ns_deg"] == pytest.approx(180.0 - theta, abs=1e-6)
    # ... and that agrees with the cap's own normal to discretisation accuracy
    assert c["err_vs_cap_normal_deg_mean"] < CAP_FLOOR_DEG, c
    assert c["physical_ok"] is True


@pytest.mark.parametrize("theta", [30.0, 60.0, 120.0, 150.0])
def test_reference_convention_realises_the_mirror(theta: float) -> None:
    """Round-1 semantics realise ``180 - theta``: mismatch = |180 - 2 theta|."""
    c = cap_convention_error(theta, "reference")
    assert c["kernel_exact_angle_vs_ns_deg"] == pytest.approx(theta, abs=1e-6)
    assert c["err_vs_cap_normal_deg_mean"] == pytest.approx(c["mirror_prediction_deg"], abs=10.0), c
    assert c["physical_ok"] is False


def test_ninety_degrees_is_blind_to_the_convention() -> None:
    """The degenerate point: both conventions coincide, so a 90 deg arm proves nothing."""
    a = cap_convention_error(90.0, "physical")
    b = cap_convention_error(90.0, "reference")
    assert a["theta_arg_deg"] == b["theta_arg_deg"] == pytest.approx(90.0)
    assert a["err_vs_cap_normal_deg_mean"] == pytest.approx(b["err_vs_cap_normal_deg_mean"], abs=1e-9)


# ---------------------------------------------------------------------------
# Solver integration
# ---------------------------------------------------------------------------
def _state(ny: int = 20, nx: int = 24, seed: int = 7):
    torch.manual_seed(seed)
    f_r, f_b = sessile_drop_initial(ny, nx, radius=7.0, theta_deg=60.0)
    solid = torch.zeros(ny, nx, dtype=torch.bool)
    solid[0, :] = True
    return f_r, f_b, solid


def test_solver_physical_is_reference_with_the_mirrored_angle() -> None:
    """``physical 60`` must equal ``reference 120`` bit for bit."""
    f_r, f_b, solid = _state()
    phys = color_gradient_two_tau_step(
        f_r, f_b, solid_mask=solid, wall_contact_angle=60.0,
        contact_angle_convention="physical",
    )
    ref = color_gradient_two_tau_step(
        f_r, f_b, solid_mask=solid, wall_contact_angle=120.0,
        contact_angle_convention="reference",
    )
    assert torch.equal(phys[0], ref[0])
    assert torch.equal(phys[1], ref[1])


def test_solver_direction_map_is_mapped_too() -> None:
    """The per-face override goes through the same map."""
    f_r, f_b, solid = _state()
    phys = color_gradient_two_tau_step(
        f_r, f_b, solid_mask=solid, wall_contact_angle=60.0,
        wall_contact_angle_by_direction={(0, -1): 45.0},
        contact_angle_convention="physical",
    )
    ref = color_gradient_two_tau_step(
        f_r, f_b, solid_mask=solid, wall_contact_angle=120.0,
        wall_contact_angle_by_direction={(0, -1): 135.0},
        contact_angle_convention="reference",
    )
    assert torch.equal(phys[0], ref[0])
    assert torch.equal(phys[1], ref[1])


def test_solver_default_convention_is_physical_and_conserves_mass() -> None:
    f_r, f_b, solid = _state()
    m_r, m_b = float(f_r.sum()), float(f_b.sum())
    default = color_gradient_two_tau_step(f_r, f_b, solid_mask=solid, wall_contact_angle=60.0)
    explicit = color_gradient_two_tau_step(
        f_r, f_b, solid_mask=solid, wall_contact_angle=60.0,
        contact_angle_convention="physical",
    )
    assert torch.equal(default[0], explicit[0]), "default convention is no longer physical"
    assert math.isclose(float(default[0].sum()), m_r, rel_tol=1e-12)
    assert math.isclose(float(default[1].sum()), m_b, rel_tol=1e-12)


def test_solver_rejects_unknown_convention() -> None:
    f_r, f_b, solid = _state()
    with pytest.raises(ValueError):
        color_gradient_two_tau_step(
            f_r, f_b, solid_mask=solid, wall_contact_angle=60.0,
            contact_angle_convention="mirror",
        )