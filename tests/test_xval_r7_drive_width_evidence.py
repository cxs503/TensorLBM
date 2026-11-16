"""Round-7 evidence tests: closure-aware body-force normalisation of the NEBB driver.

Two layers:

1. **Artifact layer** -- the committed run artifacts are re-derived here from the
   stored per-row profiles (the analytic reference is re-evaluated in closed
   form, never read from the file).  These tests fail if a number in
   ``artifacts/r7`` / ``artifacts/r6`` cannot be reproduced.
2. **Live layer** -- a short end-to-end solve of the production core path
   (``wall_bc='nebb'``) proves that the *fix in the tree* delivers the declared
   centre-line velocity, and that the same metric recovers the pre-fix overshoot
   when the drive is normalised with the legacy half-way width.  Without the
   second arm the first assertion would be unfalsifiable.

Anchor: XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "xval" / "XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S"
ART7 = BASE / "artifacts" / "r7" / "drive_width_ab.json"
ART6 = BASE / "artifacts" / "r6" / "r6_driver_consistency.json"

sys.path.insert(0, str(REPO / "benchmarks" / "xval"))
from nebb_bodyforce_evidence_r7 import pointwise_err, ref_profile  # noqa: E402


# --------------------------------------------------------------------------- #
# closed-form reference, independent of the modules under test
# --------------------------------------------------------------------------- #
def test_closed_form_parabola_shape():
    ny, nu, a = 18, 0.2, 1e-2
    u = ref_profile("nebb", ny, nu, a)
    H, y = ny - 1, np.arange(1, ny - 1, dtype=np.float64)
    # exact parabola on the node coordinates, walls on the wall rows
    np.testing.assert_allclose(u, a / (2 * nu) * y * (H - y), atol=0, rtol=1e-15)
    # symmetric about the centre plane, peak bounded by the continuous maximum
    np.testing.assert_allclose(u, u[::-1], atol=1e-15)
    assert u.max() <= a * H * H / (8.0 * nu) * (1 + 1e-15)
    assert u.max() >= a * H * H / (8.0 * nu) * (1 - 1.0 / H)
    # half-way closure: walls half a cell outside -> width ny-2, same construction
    uh = ref_profile("halfway", ny, nu, a)
    s = np.arange(1, ny - 1, dtype=np.float64) - 0.5
    np.testing.assert_allclose(uh, a / (2 * nu) * s * ((ny - 2) - s), atol=0, rtol=1e-15)
    np.testing.assert_allclose(uh, uh[::-1], atol=1e-15)
    # the two closures do not give the same amplitude (H = ny-1 vs ny-2)
    assert uh.max() < u.max()


def test_pointwise_err_rejects_shifted_profile():
    ny = 18
    ref = ref_profile("nebb", ny, 0.2, 1e-3)
    assert pointwise_err(ref, ref) == 0.0
    assert pointwise_err(1.03 * ref, ref) == pytest.approx(0.03, rel=1e-12)
    # the pre-fix widths differ by 1.265 at ny=10 -> the metric must see it
    got = ref_profile("nebb", 10, 0.2, 1e-3 * ((9.0 / 8.0) ** 2))
    assert pointwise_err(got, ref_profile("nebb", 10, 0.2, 1e-3)) > 0.2


# --------------------------------------------------------------------------- #
# artifact layer
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def r7() -> dict:
    assert ART7.exists(), f"missing run artifact {ART7}"
    return json.loads(ART7.read_text())


def _arm(payload, wall_bc, ny, drive):
    """The arm, or a skip when the campaign cell has not been run yet.

    A partially completed campaign (scheduler queue) must not look like a green
    test, and must not look like a broken one either: the missing cell is
    reported as a skip with the cell name in the reason.
    """
    for a in payload["arms"]:
        if a["wall_bc"] == wall_bc and a["ny"] == ny and a["drive"] == drive:
            return a
    pytest.skip(f"campaign cell not run yet: {wall_bc}/ny={ny}/{drive}")


def test_every_arm_recomputes_from_its_profiles(r7):
    declared = float(r7["declared_u_max"])
    for arm in r7["arms"]:
        ny, nu, a = int(arm["ny"]), float(arm["nu"]), float(arm["body_force"])
        raw = np.asarray(arm["profile_raw"])
        phys = np.asarray(arm["profile_phys"])
        # momentum definition: physical velocity == raw + a/2 (Guo 2002)
        np.testing.assert_allclose(phys - raw, 0.5 * a, atol=1e-15)
        # the stored analytic reference must equal the closed form
        np.testing.assert_allclose(arm["profile_ref"],
                                   ref_profile(arm["wall_bc"], ny, nu, a), atol=1e-12)
        # and the quoted numbers must follow from the stored array
        assert float(phys.max()) == pytest.approx(arm["u_max_phys"], abs=1e-15)
        assert float(phys.max() / declared) == pytest.approx(arm["ratio_vs_declared"],
                                                             abs=1e-12)
        assert pointwise_err(phys, ref_profile(arm["wall_bc"], ny, nu, a)) == \
            pytest.approx(arm["err_phys_pointwise"], rel=1e-12, abs=0.0)


def test_fix_hits_declared_velocity_within_truncation(r7):
    """The fixed normalisation delivers the declared velocity up to the
    truncation error of the D3Q19 closure (measured 3.1 % at ny=10, H=9 cells)."""
    for wall_bc in ("nebb", "halfway"):
        for ny in (10, 18):
            arm = _arm(r7, wall_bc, ny, "closure_width")
            assert arm["ratio_vs_declared"] == pytest.approx(1.0, abs=0.05), (wall_bc, ny)
            # and it must beat the pre-fix normalisation on the same cell
            try:
                legacy = _arm(r7, wall_bc, ny, "legacy_halfway_width")
            except Exception:  # cell not run yet
                continue
            assert abs(legacy["ratio_vs_declared"] - 1.0) > \
                abs(arm["ratio_vs_declared"] - 1.0)


@pytest.mark.xfail(strict=True,
                   reason="P1's self-imposed 1 % band is MISSED at ny=10 (3.11 %): that is "
                          "truncation error of the D3Q19 BGK + Guo closure, not a "
                          "normalisation error -- see REPORT_R7.md §5. Kept as an "
                          "executable record of the honest FAIL.")
def test_p1_one_percent_band_is_not_met(r7):
    for wall_bc in ("nebb", "halfway"):
        for ny in (10, 18):
            arm = _arm(r7, wall_bc, ny, "closure_width")
            assert abs(arm["ratio_vs_declared"] - 1.0) < 0.01, (wall_bc, ny)


def test_legacy_width_recovers_the_prefix_overshoot(r7):
    """The metric is sensitive: the legacy normalisation must show the defect."""
    for ny in (10, 18):
        arm = _arm(r7, "nebb", ny, "legacy_halfway_width")
        predicted = ((ny - 1) / (ny - 2)) ** 2
        assert arm["ratio_vs_declared"] == pytest.approx(predicted, rel=0.01)
    # and the defect is material at ny=10 (> 15 % owner threshold)...
    assert _arm(r7, "nebb", 10, "legacy_halfway_width")["ratio_vs_declared"] - 1.0 > 0.15
    # ...but below the threshold at ny=18, i.e. a ny=18-only test would miss it
    assert _arm(r7, "nebb", 18, "legacy_halfway_width")["ratio_vs_declared"] - 1.0 < 0.15


def test_halfway_width_control(r7):
    for ny in (10, 18):
        a = _arm(r7, "halfway", ny, "closure_width")["ratio_vs_declared"]
        b = _arm(r7, "halfway", ny, "legacy_halfway_width")["ratio_vs_declared"]
        assert abs(a - b) < 1e-9


@pytest.mark.xfail(strict=True,
                   reason="P4 falsified for the D3Q19 core path: the residual error is "
                          "truncation, not round-off (Round-6 had inferred exactness from the "
                          "D2Q9 module arm). Kept as an executable record.")
def test_physical_velocity_is_round_off_exact(r7):
    for wall_bc in ("nebb", "halfway"):
        for ny in (10, 18):
            arm = _arm(r7, wall_bc, ny, "closure_width")
            assert arm["err_phys_pointwise"] < 1e-10, (wall_bc, ny)
            assert arm["converged"], (wall_bc, ny, arm["steady_residual"])


def test_public_diagnostic_reports_physical_velocity(r7):
    for wall_bc in ("nebb", "halfway"):
        for ny in (10, 18):
            arm = _arm(r7, wall_bc, ny, "closure_width")
            assert 0.99 <= arm["diagnostic_shift_ratio"] <= 1.01, (wall_bc, ny)


def test_r6_campaign_rows_recompute(r7):
    if not ART6.exists():  # pragma: no cover - artifact is committed
        pytest.skip("r6 campaign artifact absent")
    payload = json.loads(ART6.read_text())
    declared = float(r7["declared_u_max"])
    assert payload["drives"], "r6 campaign stored no drive rows"
    for arm in payload["drives"]:
        phys = np.asarray(arm["profile_phys"])
        ref = ref_profile(arm["wall_bc"], int(arm["ny"]), float(arm["nu"]),
                          float(arm["body_force"]))
        assert float(phys.max() / declared) == pytest.approx(arm["ratio_phys_to_target"],
                                                             abs=1e-12)
        assert pointwise_err(phys, ref) == pytest.approx(arm["err_phys_pointwise"],
                                                         rel=1e-12, abs=0.0)
        # the config's own width stays the half-way width: the mismatch is the
        # documented trigger of the defect, and must be visible in the artifact
        assert arm["height_config"] == pytest.approx(float(arm["ny"]) - 2)
        assert arm["height_matches_closure"] is (arm["wall_bc"] == "halfway")


def test_evidence_builder_recomputes_everything():
    """The builder must re-derive every stored number from the raw profiles and
    report no integrity problem (rc 1 is allowed when a *predeclared* check
    genuinely fails -- P1/P4 above -- but an integrity problem is not)."""
    sys.path.insert(0, str(REPO / "benchmarks" / "xval"))
    import nebb_bodyforce_evidence_r7 as ev

    rc = ev.main(["--strict"])
    assert rc in (0, 1)
    evidence = json.loads((BASE / "artifacts" / "r7" / "evidence_r7.json").read_text())
    assert evidence["detail"]["problems"] == []
    assert evidence["benchmark"]["rows"], "no ref/got rows"
    for row in evidence["benchmark"]["rows"]:
        assert row["ref_u_max"] > 0 and row["got_u_max"] > 0
    assert rc == (0 if evidence["all_checks_pass"] else 1)


# --------------------------------------------------------------------------- #
# live layer (short solve; the full campaign lives in artifacts/)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(os.environ.get("TP_RUN_LIVE") != "1",
                    reason="live core-path solves take tens of seconds per arm; run with "
                           "TP_RUN_LIVE=1 (the committed artifacts carry the same check)")
@pytest.mark.parametrize("ny", [10])
def test_live_solve_fix_vs_legacy(ny, tmp_path):
    """End-to-end: the tree's solver honours the declared velocity, and the
    legacy normalisation reproduces the +26.5 % overshoot at ny=10."""
    from run_xval_r7_drive_width_ab import run_arm

    fixed = run_arm(ny, "nebb", "closure_width")
    legacy = run_arm(ny, "nebb", "legacy_halfway_width")
    assert fixed["ratio_vs_declared"] == pytest.approx(1.0, abs=0.05)
    assert legacy["ratio_vs_declared"] == pytest.approx(((ny - 1) / (ny - 2)) ** 2,
                                                        rel=0.01)
    # the physical velocity diagnostic is the one that makes this measurable
    assert 0.99 <= fixed["diagnostic_shift_ratio"] <= 1.01
    assert fixed["err_phys_pointwise"] < 0.15