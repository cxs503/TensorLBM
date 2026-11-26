"""Sessile-cap geodesic contact-angle estimator: the branch must be falsifiable.

Anchor ``XVAL-LBM-TWO-INDEPENDENT-CONTACT-ANGLE-IMPLS``.  Rounds 3 and 4 measured
the third contact-angle estimator with ``theta = 2 atan(a/h)``, which is the
*supplement* of the contact angle for a sessile cap: the R3 audit reported a
72 % FAIL of that estimator and R4 "fixed" it with a caller-supplied
``lbm_convention`` boolean plus an ``a/h > 1`` conditional.

The exact geometry (round 5, ``development/verification/contact-angle/round-5``)
is ``z_c = -R cos t``, ``h = R (1 - cos t)``, ``a = R sin t`` for **every**
``t in (0, 180)``, so

    a/h = cot(t/2)   =>   theta = 2 atan(h/a) = 180 deg - 2 atan(a/h)

with no branch test.  These tests pin that relation and keep it falsifiable:
the estimator is scored on a cap whose *label* disagrees with its geometry, and
it must follow the geometry.

Fields only -- no solver run, so this file costs milliseconds.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pytest

from tensorlbm.cg_contact_angle import contact_angle_geodesic

Y_WALL = 0.5  # library argument; first cell centre sits at y_wall - 0.5
NX, NY = 77, 66
RADIUS = 18.0
XI = 2.0


def sessile_cap(theta_deg: float, *, mislabelled: bool = False) -> np.ndarray:
    """Diffused cap, ``+1`` inside.

    ``mislabelled=False``: physical template, centre ``wall - R cos theta``, the
    true contact angle is ``theta_deg``.
    ``mislabelled=True``: the round-3 template, centre ``wall + R cos theta``,
    whose true contact angle is ``180 - theta_deg``.
    """
    th = math.radians(theta_deg)
    wall = -0.5
    y_c = wall + RADIUS * math.cos(th) if mislabelled else wall - RADIUS * math.cos(th)
    y = np.arange(NY, dtype=float)[:, None]
    x = np.arange(NX, dtype=float)[None, :]
    r = np.hypot(x - (NX - 1) / 2.0, y - y_c)
    return np.tanh((RADIUS - r) / XI)


@pytest.mark.parametrize("theta_deg", [45.0, 70.0, 90.0, 110.0, 135.0])
def test_physical_cap_recovers_true_angle(theta_deg: float) -> None:
    """No convention flag: the estimator returns the cap's true contact angle."""
    m = contact_angle_geodesic(sessile_cap(theta_deg), XI, y_wall=Y_WALL)
    assert abs(m["theta_deg"] - theta_deg) <= 2.0, m
    # a/h = cot(theta/2) for a sessile cap
    assert m["a_over_h"] == pytest.approx(1.0 / math.tan(math.radians(theta_deg) / 2.0), abs=0.1)
    assert m["convention"] == "physical_sessile"


@pytest.mark.parametrize("label_deg", [45.0, 70.0, 110.0, 135.0])
def test_estimator_follows_geometry_not_label(label_deg: float) -> None:
    """Falsifiability: on the round-3 (mislabelled) template the answer is 180 - label.

    A estimator that a caller can steer with a flag (or with a test on ``a/h``)
    cannot pass this and the physical-cap test at the same time.
    """
    m = contact_angle_geodesic(sessile_cap(label_deg, mislabelled=True), XI, y_wall=Y_WALL)
    assert abs(m["theta_deg"] - (180.0 - label_deg)) <= 2.0, m


def test_supplement_is_the_old_formula() -> None:
    """The retired formula is exactly the mirror: 2 atan(a/h) = 180 - 2 atan(h/a)."""
    for theta_deg in (45.0, 70.0, 90.0, 110.0, 135.0):
        m = contact_angle_geodesic(sessile_cap(theta_deg), XI, y_wall=Y_WALL)
        old = math.degrees(2.0 * math.atan(m["a_over_h"]))
        assert abs(old + m["theta_deg"] - 180.0) <= 1e-9
        # and it is the supplement of the true angle, i.e. wrong by |180 - 2 theta|
        assert old == pytest.approx(180.0 - theta_deg, abs=2.0)


def test_lbm_convention_flag_is_deprecated_and_ignored() -> None:
    psi = sessile_cap(70.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        flagged = contact_angle_geodesic(psi, XI, y_wall=Y_WALL, lbm_convention=True)
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    base = contact_angle_geodesic(psi, XI, y_wall=Y_WALL)
    assert flagged["theta_deg"] == pytest.approx(base["theta_deg"], abs=1e-12)


COMMITTED = (
    Path(__file__).resolve().parents[1]
    / "development"
    / "verification"
    / "contact-angle"
    / "round-3"
    / "runs_r3"
)


@pytest.mark.parametrize(
    "tag,prescribed",
    [
        ("geometric_ghost__th070.0__baseline__xi3", 70.0),
        ("geometric_ghost__th110.0__baseline__xi3", 110.0),
        ("neumann_flux__th070.0__baseline__xi3", 70.0),
        ("neumann_flux__th135.0__baseline__xi3", 135.0),
    ],
)
def test_committed_golden_fields_inside_owner_gate(tag: str, prescribed: float) -> None:
    """The two independent wall closures vs the prescribed Young angle, from the
    committed round-3 fields (no re-simulation): owner gate is 15 %."""
    npz = COMMITTED / f"{tag}.npz"
    if not npz.exists():  # pragma: no cover - evidence bundle missing
        pytest.skip("committed golden field not present")
    psi = np.load(npz)["psi"]
    m = contact_angle_geodesic(psi, 3.0, y_wall=Y_WALL)
    err_pct = abs(m["theta_deg"] - prescribed) / prescribed * 100.0
    assert err_pct <= 15.0, (tag, m, err_pct)