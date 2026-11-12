"""Canonical-answer lock for every GCI / observed-order site (``docs/GCI_CONVENTION_AUDIT.md``).

The repository quotes Roache's grid-convergence index in a large share of its
acceptance verdicts, but the index is re-implemented in more than a dozen
modules.  A conventional test ("the function returns a positive number",
"``gci`` grows with the error") cannot see a missing ``1/(r^p - 1)`` factor or a
missing safety factor, which is exactly how two defects survived here:

* ``flapping_propulsion.gci`` omitted ``1/(r^p - 1)`` (3x too large at
  ``p = 2, r = 2``);
* ``ice_chain3d.reference.gci`` hard-wired ``Fs = 1.0`` (25 % too small).

Every site is therefore evaluated on a ladder whose exact answer is known::

    f(h) = 1 + 0.5 h^2,  r = 2,  h = 1/32, 1/16, 1/8

giving ``p = 2``, ``f_Richardson = 1`` and

    GCI_fine(Fs) = Fs * |(f2 - f1)/f1| / (r^p - 1)

== ``0.061005 %`` at ``Fs = 1.25`` and ``0.048804 %`` at ``Fs = 1.0``.

Files audited: ``scripts/gci_convention_audit.py`` (producer of
``artifacts/gci_convention/audit.json``).
"""

from __future__ import annotations

import math

import pytest

P_TRUE = 2.0
R_TRUE = 2.0
F_EXACT = 1.0
C_TRUE = 0.5
H = [1.0 / 32.0, 1.0 / 16.0, 1.0 / 8.0]
F_VALUES = [F_EXACT + C_TRUE * h**P_TRUE for h in H]
F1, F2, F3 = F_VALUES  # fine, medium, coarse
E21 = F2 - F1
ERR_VALUES = [C_TRUE * h**P_TRUE for h in H]

GCI_FS125 = 1.25 * abs(E21 / F1) / (R_TRUE**P_TRUE - 1.0)
GCI_FS1 = 1.00 * abs(E21 / F1) / (R_TRUE**P_TRUE - 1.0)


def test_canonical_constants_are_self_consistent():
    """The ladder must satisfy the closed form it is used to certify."""
    assert F1 < F2 < F3  # fine -> coarse
    assert F1 > F_EXACT
    assert math.isclose(abs(E21), C_TRUE * H[1] ** P_TRUE - C_TRUE * H[0] ** P_TRUE, rel_tol=1e-15)
    # Richardson: f1 + (f1 - f2)/(r^p - 1) == f_exact
    assert math.isclose(F1 + (F1 - F2) / (R_TRUE**P_TRUE - 1.0), F_EXACT, rel_tol=1e-12)
    assert 100.0 * GCI_FS1 / GCI_FS125 == pytest.approx(80.0, rel=1e-12)


# --------------------------------------------------------------------------- #
# three-grid GCI sites: reported value must equal the closed form
# --------------------------------------------------------------------------- #
def test_airfoil_polar_gci_matches_closed_form():
    from tensorlbm.airfoil_polar import gci_analysis

    out = gci_analysis(F1, F2, F3, r=R_TRUE)
    assert out["gci_fine_pct"] == pytest.approx(100.0 * GCI_FS125, rel=1e-9)
    assert out["p"] == pytest.approx(P_TRUE, rel=1e-9)
    assert out["extrapolated"] == pytest.approx(F_EXACT, rel=1e-9)


def test_ice_keel_drag_gci_matches_closed_form():
    from tensorlbm.ice_keel_drag.criteria import gci

    out = gci(F1, F2, F3, ratio=R_TRUE)
    assert out["ok"] is True
    assert out["p"] == pytest.approx(P_TRUE, rel=1e-9)
    assert out["gci_fine"] == pytest.approx(GCI_FS125, rel=1e-9)
    assert out["f_exact"] == pytest.approx(F_EXACT, rel=1e-9)


def test_icebow_cad_gci_matches_closed_form():
    from tensorlbm.icebow_cad import gci

    out = gci(F1, F2, F3, r=R_TRUE)
    assert out["observed_order"] == pytest.approx(P_TRUE, rel=1e-9)
    assert out["gci_fine_percent"] == pytest.approx(100.0 * GCI_FS125, rel=1e-9)
    assert out["richardson"] == pytest.approx(F_EXACT, rel=1e-9)


def test_free_surface_zalesak_gci_matches_closed_form():
    from tensorlbm.free_surface_zalesak import gci

    out = gci(E21 / F1, E21 / F1, P_TRUE, r=R_TRUE)
    assert out == pytest.approx(100.0 * GCI_FS125, rel=1e-9)


def test_ice_roughness_three_zone_agrees_within_iteration_tolerance():
    """Observed order is solved iteratively here: 0.2 % band, not round-off."""
    from tensorlbm.ice_roughness.three_zone import gci_three_grid

    out = gci_three_grid(F1, F2, F3, ratio=R_TRUE)
    gci_value = out.get("gci", out.get("gci_fine"))
    assert gci_value is not None
    # the site reports a fraction
    assert gci_value == pytest.approx(GCI_FS125, rel=5e-3)


# --------------------------------------------------------------------------- #
# regression: the two repaired sites (defect_fix 2026-11-12)
# --------------------------------------------------------------------------- #
def test_flapping_propulsion_gci_has_the_richardson_denominator():
    """Pre-fix value was ``1.25 |e21|/|f1|`` = (r^p - 1) x too large."""
    from tensorlbm.flapping_propulsion import gci

    out = gci(F1, F2, F3, r=R_TRUE)
    pre_fix = 1.25 * abs(E21) / abs(F1)  # what the first implementation returned
    assert pre_fix / GCI_FS125 == pytest.approx(R_TRUE**P_TRUE - 1.0, rel=1e-12)
    assert out.gci_fine == pytest.approx(GCI_FS125, rel=1e-9)
    assert out.gci_fine == pytest.approx(pre_fix / (R_TRUE**P_TRUE - 1.0), rel=1e-12)
    assert out.p_observed == pytest.approx(P_TRUE, rel=1e-9)
    assert out.f_extrapolated == pytest.approx(F_EXACT, rel=1e-9)


def test_ice_chain3d_gci_carries_the_roache_safety_factor():
    """Pre-fix value was the Fs = 1 index: 25 % below Roache's three-grid GCI."""
    from tensorlbm.ice_chain3d.reference import gci

    out = gci(F1, F2, F3, r=R_TRUE)
    assert out["p_apparent"] == pytest.approx(P_TRUE, rel=1e-9)
    assert out["f_extrapolated"] == pytest.approx(F_EXACT, rel=1e-9)
    assert out["safety_factor"] == pytest.approx(1.25)
    assert out["GCI_fine"] == pytest.approx(GCI_FS125, rel=1e-9)
    assert out["GCI_fine"] == pytest.approx(1.25 * GCI_FS1, rel=1e-9)
    # opt-out stays possible for a reported-and-labelled "no safety factor" run
    bare = gci(F1, F2, F3, r=R_TRUE, safety=1.0)
    assert bare["GCI_fine"] == pytest.approx(GCI_FS1, rel=1e-9)


def test_ice_chain3d_gci_flags_non_monotone_data():
    from tensorlbm.ice_chain3d.reference import gci

    out = gci(F3, F2, F1, r=R_TRUE)  # reversed ladder -> decreasing error
    assert out["valid"] is False or out["p_apparent"] is None or out["GCI_fine"] is not None


# --------------------------------------------------------------------------- #
# observed-order producers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["loglog_leastsq", "h_p_fit", "drift_slope"])
def test_observed_order_recovers_p_two(name):
    if name == "loglog_leastsq":
        from tensorlbm.convergence_mms import observed_order

        out = observed_order(H, ERR_VALUES)
        assert out["p"] == pytest.approx(P_TRUE, rel=1e-9)
        assert out["r2"] == pytest.approx(1.0, abs=1e-9)
    elif name == "h_p_fit":
        from tensorlbm.ibm_sphere_surface_force import observed_order

        out = observed_order(H, F_VALUES)
        assert out["order"] == pytest.approx(P_TRUE, rel=1e-6)
        assert out["extrapolated"] == pytest.approx(F_EXACT, rel=1e-6)
    else:
        from tensorlbm.free_surface_mass_ledger import observed_order

        out = observed_order(
            {str(i): h for i, h in enumerate(H)},
            {str(i): e for i, e in enumerate(ERR_VALUES)},
        )
        assert out["order"] == pytest.approx(P_TRUE, rel=1e-9)