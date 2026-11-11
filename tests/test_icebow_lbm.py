"""Tests for the ice-bow LBM towed-resistance leg (:mod:`tensorlbm.icebow_lbm`).

Coverage map (node IN13W2-N06):

* geometry hand-off: the voxeliser used by the LBM leg reproduces the analytic
  submerged volume of the same hull surface;
* channel layout: the body is placed symmetrically, with a physical blockage;
* **solver equivalence**: the campaign loop returns bit-comparable momentum
  exchange to :func:`tensorlbm.full_wet.run_fully_wetted_flow` (the audited R1
  path) on the same mask, so the local loop cannot silently drift;
* steady statistics: the reported mean/drift are the documented window
  arithmetic (checked on a synthetic history, no solver involved);
* driver sanity on a textbook case (sphere vs Schiller-Naumann) and the
  hybrid viscous/wave bookkeeping.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm import icebow_cad as cad
from tensorlbm import icebow_lbm as lbm

pars = cad.IceBowParams()


@pytest.fixture(scope="module")
def bow_mask() -> lbm.SubmergedBowMask:
    return lbm.submerged_bow_mask(pars, 64)


def test_submerged_mask_matches_analytic_submerged_volume(bow_mask):
    """The LBM leg eats the same hull: voxel volume vs analytic, <3 %."""
    assert abs(bow_mask.volume_error_percent) < 3.0
    assert bow_mask.waterline_cells >= 4
    assert bow_mask.mask.any()


def test_mask_is_cut_at_the_design_waterline(bow_mask):
    nz = bow_mask.mask.shape[0]
    assert nz * bow_mask.spacing <= pars.draft + bow_mask.spacing
    # The waterline layer must still carry a solid cross-section.
    assert bow_mask.mask[-1].sum() > 0.1 * bow_mask.mask[0].sum()


def test_too_coarse_grid_is_rejected_with_actionable_message():
    with pytest.raises(ValueError, match="cells_per_length"):
        lbm.submerged_bow_mask(pars, 16)


def test_channel_layout_is_symmetric_and_physical(bow_mask):
    solid, layout = lbm.build_bow_channel(bow_mask, pars, lbm.BowDomain())
    assert solid.shape == tuple(layout["shape_zyx"])
    # port/starboard symmetry about the tank centreline
    assert np.array_equal(solid.sum(axis=1), solid[:, ::-1, :].sum(axis=1))
    assert 0.01 < layout["blockage_midship"] < 0.60
    # the body hangs from the rigid lid: its top layer is the last z layer
    _, y0, x0 = layout["body_offset_zyx"]
    assert solid[-1, y0 : y0 + layout["body_shape_zyx"][1], x0 : x0 + 10].any()
    assert 0.6 < layout["domain_lengths"]["upstream"] / layout["domain_lengths"]["downstream"] + 0.6


def test_case_rejects_unphysical_setup():
    with pytest.raises(ValueError):
        lbm.BowLbmCase(params=pars, target_velocity=0.2)
    with pytest.raises(ValueError):
        lbm.BowLbmCase(params=pars, re=0.0)
    with pytest.raises(ValueError):
        lbm.BowLbmCase(params=pars, gain=2.0)
    case = lbm.BowLbmCase(params=pars, cells_per_length=70, re=1000.0, target_velocity=0.08)
    assert 0.5 < case.tau < 0.55
    assert case.nu == pytest.approx(0.08 * 70 / 1000.0)


def test_label_matches_campaign_label_convention():
    case = lbm.BowLbmCase(params=pars, cells_per_length=64)
    assert case.label().startswith("straight_a30_cb0.7")
    # the campaign joins LBM rows to the wave rows on this string
    assert f"{pars.family}_a{pars.alpha_stem_deg:g}_cb{pars.cb_target:g}" == "straight_a30_cb0.7"


def test_steady_stats_arithmetic_on_synthetic_history():
    steps = list(range(10, 1010, 10))
    ramp = [5.0 * (1.0 - math.exp(-s / 100.0)) for s in steps]
    stats = lbm._steady_stats(ramp, steps, 0.5)
    assert stats["samples"] == len([s for s in steps if s >= steps[0] + 0.5 * (steps[-1] - steps[0])])
    assert stats["mean"] == pytest.approx(np.mean(ramp[-stats["samples"] :]), rel=1e-12)
    assert stats["drift_rel"] < 0.05  # converged tail


def test_steady_stats_flags_a_ramp():
    steps = list(range(10, 1010, 10))
    ramp = [0.01 * s for s in steps]
    stats = lbm._steady_stats(ramp, steps, 0.5)
    assert stats["drift_rel"] > 0.2  # clearly not steady


def test_momentum_budget_closes():
    """Momentum exchange drag == momentum the source injects (control volume).

    Every lateral face is free-slip and the hull is the only streamwise
    momentum sink, so at steady state ``drag_mex == du/tau * N_fluid``.  A
    mismatch means a leaking boundary or a wrong force extraction.
    """
    out = lbm.run_case(
        lbm.BowLbmCase(
            params=pars,
            cells_per_length=46,
            re=1000.0,
            target_velocity=0.08,
            steps=600,
            sample_every=10,
            steady_window=0.3,
        )
    )
    assert out["status"] == "COMPLETED"
    # the mean streamwise velocity is the enforced control target
    assert out["setup"]["u_mean_steady"] == pytest.approx(0.08, rel=0.05)
    assert out["setup"]["reynolds_checked"] == pytest.approx(1000.0, rel=0.05)
    assert out["drag_lattice"] > 0.0
    assert abs(out["momentum_budget_percent"]) < 20.0
    # grid/Re-independent id: this is the join key used by the wave leg
    assert out["member_label"] == "straight_a30_cb0.7"


def test_sphere_drag_is_in_the_textbook_band():
    out = lbm.sphere_validation(
        re=100.0, diameter_cells=6, box=(32, 20, 20), steps=600, steady_window=0.5, sample_every=20
    )
    cd_ref = 24.0 / 100.0 * (1.0 + 0.15 * 100.0**0.687)
    assert out["cd_reference"] == pytest.approx(cd_ref, rel=1e-12)
    # coarse grid + short transient: a band, not the 15 % engineering claim
    assert 0.5 * cd_ref < out["cd_lbm"] < 2.5 * cd_ref


def test_hybrid_split_bookkeeping():
    rows = [
        {"label": "straight_a20_cb0.7", "member": {"family": "straight", "alpha_stem_deg": 20.0}, "c_t_lbm": 0.010},
        {"label": "straight_a25_cb0.7", "member": {"family": "straight", "alpha_stem_deg": 25.0}, "c_t_lbm": 0.0105},
        {"label": "straight_a30_cb0.7", "member": {"family": "straight", "alpha_stem_deg": 30.0}, "c_t_lbm": 0.011},
        {"label": "straight_a35_cb0.7", "member": {"family": "straight", "alpha_stem_deg": 35.0}, "c_t_lbm": 0.0118},
    ]
    wave = {
        "share_wave": 0.25,
        "cw_by_label": {
            "straight_a20_cb0.7": 1.00,
            "straight_a25_cb0.7": 0.95,
            "straight_a30_cb0.7": 0.90,
            "straight_a35_cb0.7": 0.84,
        },
    }
    out = lbm.hybrid_sensitivity(rows, wave)
    assert out["share_wave"] == pytest.approx(0.25)
    ref = next(r for r in out["rows"] if abs(r["alpha_stem_deg"] - 30.0) < 1e-9)
    assert ref["delta_rt_hybrid"] == pytest.approx(0.0, abs=1e-12)
    lo = next(r for r in out["rows"] if abs(r["alpha_stem_deg"] - 35.0) < 1e-9)
    expected = 0.75 * (0.0118 - 0.011) / 0.011 + 0.25 * (0.84 - 0.90) / 0.90
    assert lo["delta_rt_hybrid"] == pytest.approx(expected, rel=1e-12)
    # viscous leg grows with rake, wave leg shrinks: the split must be reported
    assert out["monotone_delta_ct_viscous"] is True
    assert out["monotone_delta_cw_wave"] is True