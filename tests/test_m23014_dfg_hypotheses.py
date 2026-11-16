"""m23014 — regression locks for the DFG 2D-2 root-cause discriminators (H3, H4).

These tests assert the *mechanism* facts the mission relies on, not the pass/fail
of the case.  They are cheap (masks + frozen series, no solver run):

* H3: the discretised cylinder mask matches pi*r^2 within 1%, the *nominal*
  blockage equals the benchmark spec, and the half-way bounce-back surface puts
  the hydrodynamic diameter at D+1 cells (an O(1/D) bias, quantified here).
* H4: the Cl amplitude estimator is insensitive to the averaging-window length
  (so "too short a sample" is NOT the error source), while the start-up
  transient IS (so the window must exclude it).

If a future change to the geometry or the estimator silently moves these
numbers, the root-cause attribution in docs/m23014/ becomes wrong -- hence the
lock.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np
import pytest
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "docs/m23014"))

from tensorlbm.validation import cylinder_dfg as cdfg  # noqa: E402
from tensorlbm.boundary.boundaries import cylinder_mask  # noqa: E402

import audit_h3_h4 as audit  # noqa: E402

ART = REPO / "benchmarks/verified/cylinder/re100_dfg/artifacts"


@pytest.fixture(scope="module")
def geom():
    return audit.geometry_audit(levels=(20, 28, 40))


@pytest.mark.parametrize("level", ["D20", "D28", "D40"])
def test_mask_area_matches_circle_within_one_percent(geom, level):
    g = geom[level]
    assert abs(g["area_ratio_mask_over_circle"] - 1.0) < 0.01, (
        f"{level}: discretised mask area {g['solid_cells']} vs pi*r^2 "
        f"{g['area_circle_pi_r2']:.1f} (ratio {g['area_ratio_mask_over_circle']:.4f})"
    )


@pytest.mark.parametrize("level,expect", [("D20", 0.24390), ("D40", 0.24390)])
def test_nominal_blockage_equals_benchmark_spec(geom, level, expect):
    """DFG 2D-2 pins D/H = 0.1/0.41; a mismatch would be a geometry bug."""
    assert geom[level]["blockage_nominal_D_over_H"] == pytest.approx(expect, abs=5e-5)


def test_halfway_bounce_back_effective_diameter_is_D_plus_one(geom):
    """Half-way BB puts the wall half a cell outside the outermost solid centre.

    This is the O(1/D) geometric bias the mission quantifies (H3).  It must decay
    with D -- if it ever stops decaying, the mask convention changed.
    """
    biases = []
    for lv in ("D20", "D28", "D40"):
        g = geom[lv]
        assert g["effective_diameter_over_nominal"] == pytest.approx(
            (int(lv[1:]) + 1.0) / int(lv[1:]), rel=1e-12
        )
        assert g["effective_radius_halfway"] == pytest.approx(g["cell_centre_radius"] + 0.5)
        biases.append(g["cd_bias_predicted_pct_from_effective_Da"])
    assert biases[0] > biases[1] > biases[2], f"bias must decay with D, got {biases}"


@pytest.mark.parametrize("level", ["D20", "D28", "D40"])
def test_reynolds_is_exactly_preserved_at_every_level(geom, level):
    """Grid refinement at fixed U_mean and Re must keep Re exact (tau drifts, Re must not)."""
    assert geom[level]["re_check"] == pytest.approx(100.0, rel=1e-9)


def test_probe_columns_expand_outward_from_the_surface(geom):
    """DFG probes sit on the cylinder surface, i.e. on solid cells: the case must
    sample fluid columns and extrapolate.  Assert the helper does that."""
    p = geom["D40"]["probe"]
    # upstream probes step in -x, so "far" is the smaller column index
    assert p["up_far"] < p["up_near"] - 1, "upstream probes must step away from the surface"
    assert p["dn_far"] > p["dn_near"] + 1, "downstream probes must step away from the surface"
    assert p["x_probe_a"] < p["x_probe_b"]


# --------------------------------------------------------------------------- #
# H4
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def sweep():
    return audit.window_sweep(levels=(20, 28, 40), periods=(4, 6, 8, 10, 12))


@pytest.mark.parametrize("level", ["D20", "D28", "D40"])
def test_window_length_does_not_move_the_amplitude(sweep, level):
    """'Too short a sample' would show up as a strong drift with window length.

    The retained window is transient-free, so extending it 4 -> 12 periods must
    change Cl amplitude by well under the 10% mission tolerance.  Observed
    drift is ~1-2% at D40 (comparable to the estimator's own reproducibility).
    """
    amps = [w["cl_amp_fundamental"] for w in sweep[level]["windows"]]
    drift = 100.0 * (max(amps) - min(amps)) / min(amps)
    assert drift < 5.0, f"{level}: window-length drift {drift:.2f}% -> sampling duration matters"


@pytest.mark.parametrize("level,min_share", [("D28", 5.0), ("D40", 5.0)])
def test_startup_transient_contaminates_a_whole_record_peakpick(level, min_share):
    """Including the start-up transient inflates (max-min)/2 by >= several percent.

    This is the effect m16827 found behind the +36% headline; locked so the
    attribution stays reproducible.
    """
    dec = audit.transient_decomposition(levels=(level,))[level]
    assert dec["transient_share_pct_of_value"] >= min_share, (
        f"{level}: transient share {dec['transient_share_pct_of_value']:.2f}% "
        "-> the headline estimator defect would not reproduce"
    )


def test_own_estimator_reproduces_the_frozen_case_estimator():
    """My independent ACF+least-squares estimator must agree with the frozen
    m16827 re-judged values to within the analyzer's stated reproducibility (1%).

    Guards against H4 conclusions being an artifact of my reimplementation.
    """
    frozen = json.loads((ART / "rejudge_cl_amp.json").read_text())
    dec = audit.transient_decomposition(levels=(20, 28, 40))
    worst = max(abs(dec[k]["reproduce_delta_pct"]) for k in dec)
    assert worst < 1.0, f"analyzer disagrees with frozen estimator by {worst:.3f}%"
    for k in dec:
        assert dec[k]["frozen_m16827_cl_amp_new"] == pytest.approx(
            frozen["levels"][k]["cl_amp_new"], rel=1e-9
        )

# --------------------------------------------------------------------------- #
# H2 — tau / viscosity-model sensitivity from frozen data
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def h2():
    return audit.tau_pair_frozen()


def test_frozen_arm_reproduces_the_rejudged_cd(h2):
    """The window-mean Cd of my analyzer must match the frozen re-judged Cd.

    This is what licenses using the analyzer on the u=0.06 arm, which was never
    re-judged by the case code.
    """
    assert abs(h2["cd_selfcheck_delta_pct"]) < 0.05, h2["cd_selfcheck_delta_pct"]


def test_tau_shift_is_only_marginal_at_fixed_geometry(h2):
    """Pre-declared H2 rule: >=5% -> tau dominant, <2% -> not dominant.

    Measured ~4%: marginal.  Locked with a band so a future change that makes the
    solver strongly tau-dependent is caught.
    """
    d = abs(h2["deltas_pct"]["cl_amp"])
    assert 2.0 <= d < 5.0, f"tau sensitivity {d:.2f}% left the pre-declared marginal band"
    assert h2["h2_verdict"] == "tau_model_marginal"


def test_streamwise_metrics_are_less_tau_sensitive_than_the_lift(h2):
    """Cd/dp/St move far less than Cl under the same perturbation -- consistent with
    the case's error pattern (Cd/dp/St are near tolerance while Cl is not)."""
    dl = abs(h2["deltas_pct"]["cl_amp"])
    for m in ("cd", "dp", "st"):
        assert abs(h2["deltas_pct"][m]) < dl, f"{m} is more tau-sensitive than Cl"
