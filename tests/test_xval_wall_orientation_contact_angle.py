"""Invariant tests for the wall-orientation cross-validation of the contact-angle BC.

``development/verification/contact-angle/wall-orientation`` asks a question the
horizontal-plate matrix of ``XVAL-LBM-TWO-INDEPENDENT-CONTACT-ANGLE-IMPLS``
cannot answer: is the discretised Young wall condition of the production solver
(:mod:`tensorlbm.cac_lbm2d`) independent of which side of the box the wall sits
on?  These tests pin the parts that can be pinned without a long relaxation:

* the frame transforms used to measure all four orientations are exact
  permutations and their round trip is the identity;
* the two (wall-at-bottom) estimators recover the angle of an *exact* circular
  cap identically in all four frames;
* **the solver itself is rotation symmetric**: a configuration and its 90 deg
  rotation, run for the same number of steps, produce the same field after the
  frame is mapped back.  A failure here would make the xval rows meaningless.
* the predeclared thresholds were not relaxed after the sweep was inspected.
"""

from __future__ import annotations

import json
import math
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
XVAL = REPO / "development" / "verification" / "contact-angle" / "wall-orientation"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from tensorlbm import cac_lbm2d as cac  # noqa: E402
from tensorlbm.wetting import metrics as M  # noqa: E402

X = import_module("development.verification.contact-angle.wall-orientation.run_xval")

SHAPE = (64, 64)
THICK = 2
WIDTH = 4.0
RADIUS = 12.0
Y_WALL = THICK - 0.5


@pytest.mark.parametrize("side", X.PREDECLARED["orientations"])
def test_frame_transform_roundtrip_and_wall_location(side: str) -> None:
    mask_c = X.wall_mask_canonical(SHAPE, THICK)
    phys = X.from_canonical(mask_c, side)
    ys, xs = np.nonzero(phys)
    if side == "bottom":
        assert ys.min() == 0 and ys.max() == THICK - 1 and len(set(xs.tolist())) == SHAPE[1]
    elif side == "top":
        assert ys.min() == SHAPE[0] - THICK and ys.max() == SHAPE[0] - 1
    elif side == "left":
        assert xs.min() == 0 and xs.max() == THICK - 1 and len(set(ys.tolist())) == SHAPE[0]
    else:
        assert xs.min() == SHAPE[1] - THICK and xs.max() == SHAPE[1] - 1
    back = X.to_canonical(phys, side)
    assert np.array_equal(back, mask_c)
    assert back[:THICK, :].all() and not back[THICK:, :].any()


@pytest.mark.parametrize("side", X.PREDECLARED["orientations"])
@pytest.mark.parametrize("theta", [60.0, 90.0, 120.0])
def test_estimators_are_frame_invariant_on_an_exact_cap(side: str, theta: float) -> None:
    phi_c = X.cap_field(SHAPE, RADIUS, WIDTH, theta, Y_WALL)
    phi = X.from_canonical(phi_c, side)
    circ = M.contact_angle_circle_fit(X.to_canonical(phi, side), Y_WALL)
    cap = M.spherical_cap_inversion(X.to_canonical(phi, side), Y_WALL)
    # the circle fit of an exact cap is the exact angle; the area inversion is a
    # coarser estimator (it discards the profile) with a known few-degree bias.
    assert abs(circ["theta_deg"] - theta) < 1.0
    if theta <= 90.0:
        assert abs(cap["theta_deg"] - theta) < 6.0
    else:
        assert abs(cap["theta_deg"] - theta) < 12.0


def _field_after_steps(side: str, theta: float, steps: int) -> np.ndarray:
    mask = torch.from_numpy(X.from_canonical(X.wall_mask_canonical(SHAPE, THICK), side).copy())
    phi0 = torch.from_numpy(
        X.from_canonical(X.cap_field(SHAPE, RADIUS, WIDTH, theta, Y_WALL), side).copy()
    )
    sim = cac.CAC2D(
        SHAPE,
        sigma=0.02,
        width=WIDTH,
        rho_l=1.0,
        rho_g=0.1,
        mu_l=0.1,
        mu_g=0.01,
        wall_mask=mask,
        wall_phi_policy="contact",
        contact_angle_deg=theta,
        gravity=(0.0, 0.0),
        gravity_mode="rho",
        axes=X.axes_policy(side),
        dtype=torch.float64,
    )
    sim.initialize(phi0)
    cac.run(sim, steps, sample_every=0)
    return X.to_canonical(sim.f.sum(dim=0).detach().cpu().numpy(), side)


@pytest.mark.parametrize("theta", [60.0, 90.0, 120.0])
def test_solver_is_symmetric_under_a_ninety_degree_rotation(theta: float) -> None:
    """Bottom wall vs left wall, same experiment; the mapped-back fields must agree.

    Gravity is zero, so the rotation is an exact symmetry of the discrete problem;
    any deviation is an asymmetry of the wall machinery, which is exactly what the
    xval measures.  Tolerance: float64 round-off accumulated over 12 steps.
    """
    a = _field_after_steps("bottom", theta, 12)
    b = _field_after_steps("left", theta, 12)
    assert np.abs(a - b).max() < 1e-9


@pytest.mark.parametrize("theta", [60.0, 90.0, 120.0])
def test_solver_is_symmetric_under_a_mirror(theta: float) -> None:
    """Bottom wall vs top wall: an exact mirror, so the mapped fields must agree."""
    a = _field_after_steps("bottom", theta, 12)
    b = _field_after_steps("top", theta, 12)
    assert np.abs(a - b).max() < 1e-9


@pytest.mark.parametrize("theta", [60.0, 120.0])
def test_axis_policy_must_rotate_with_the_wall(theta: float) -> None:
    """Negative control: keeping the bottom-wall halo policy on a side wall breaks isotropy.

    The halo policy (zero fill across the walls, periodic along them) is part of
    the geometry, not of the numerics: rotating the experiment by 90 deg without
    rotating the policy is a *different* boundary-value problem, and it shows up
    as an O(1) difference in the wall rows.  This test exists so that a future
    refactor that hard-codes ``axes="wp"`` cannot silently invalidate the xval.
    """
    def field(side: str, axes: str) -> np.ndarray:
        mask = torch.from_numpy(X.from_canonical(X.wall_mask_canonical(SHAPE, THICK), side).copy())
        phi0 = torch.from_numpy(
            X.from_canonical(X.cap_field(SHAPE, RADIUS, WIDTH, theta, Y_WALL), side).copy()
        )
        sim = cac.CAC2D(
            SHAPE, sigma=0.02, width=WIDTH, rho_l=1.0, rho_g=0.1, mu_l=0.1, mu_g=0.01,
            wall_mask=mask, wall_phi_policy="contact", contact_angle_deg=theta,
            gravity=(0.0, 0.0), gravity_mode="rho", axes=axes, dtype=torch.float64,
        )
        sim.initialize(phi0)
        sim.refresh_solid_ghosts()
        return X.to_canonical(sim.f.sum(dim=0).detach().cpu().numpy(), side)

    good = np.abs(field("bottom", "wp") - field("left", "pw")).max()
    bad = np.abs(field("bottom", "wp") - field("left", "wp")).max()
    assert good < 1e-12
    assert bad > 1e-2


def test_predeclared_thresholds_not_relaxed() -> None:
    th = json.loads((XVAL / "predeclared.json").read_text())["predeclared_thresholds"]
    assert th == {
        "accept_pct": 15.0,
        "orientation_pct": 15.0,
        "mirror_deg": 0.5,
        "cross_arm_pct": 15.0,
        "min_apex_over_width": 1.0,
        "circle_rms_over_width": 0.5,
    }


def test_summary_gates_are_machine_evaluated() -> None:
    summary = json.loads((XVAL / "summary.json").read_text())
    assert summary["n_rows"] == len(summary["rows"])
    assert summary["verdict"] in ("PASS", "FAIL")
    for key in ("mirror", "orientation", "cross_arm"):
        for row in summary[key]:
            assert isinstance(row["pass"], bool)
            assert math.isfinite(row.get("diff_deg", row.get("diff_pct")))
    assert summary["anchor_pass"] == all(
        summary["per_arm"][arm]["n_pass"] == summary["per_arm"][arm]["n_measurable"]
        for arm in ("contact", "wall_order")
    )
