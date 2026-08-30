"""Tests for the anchor-row selection layer (``tensorlbm.ai.anchor_selection``).

CPU suite for the wave-16 winner landing: geometric target construction,
the span rule from the wave-15 W2 campaign, and the promote-from-held-out
matching used for the slender-1.30 corpus promotion (350 -> 353 rows).
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorlbm.ai.anchor_selection import (
    MIN_SPAN_DECADES,
    anchor_targets,
    match_anchor_rows,
    span_decades,
    validate_span,
)


class TestAnchorTargets:
    def test_k3_recipe_endpoints_and_geometric_mid(self):
        tgt = anchor_targets(63.2, 654.2, k=3)
        assert tgt.shape == (3,)
        assert tgt[0] == pytest.approx(63.2)
        assert tgt[-1] == pytest.approx(654.2)
        # geometric midpoint, equally spaced in log10
        assert tgt[1] == pytest.approx(np.sqrt(63.2 * 654.2))
        assert np.allclose(np.diff(np.log10(tgt)), np.log10(654.2 / 63.2) / 2)

    def test_targets_sorted_and_inside_range(self):
        for k in (2, 3, 6, 12):
            tgt = anchor_targets(40.0, 900.0, k=k)
            assert len(tgt) == k
            assert np.all(np.diff(tgt) > 0.0)
            assert tgt[0] == pytest.approx(40.0)
            assert tgt[-1] == pytest.approx(900.0)

    def test_k2_is_endpoints_only(self):
        tgt = anchor_targets(63.2, 654.2, k=2)
        assert tgt.tolist() == pytest.approx([63.2, 654.2])

    def test_k1_rejected(self):
        with pytest.raises(ValueError, match="k >= 2"):
            anchor_targets(63.2, 654.2, k=1)

    @pytest.mark.parametrize("lo,hi", [(0.0, 100.0), (-5.0, 100.0), (100.0, 100.0), (200.0, 100.0)])
    def test_bad_range_rejected(self, lo: float, hi: float):
        with pytest.raises(ValueError, match="0 < re_min < re_max"):
            anchor_targets(lo, hi, k=3)


class TestSpanRule:
    def test_span_math(self):
        assert span_decades([63.2, 654.2]) == pytest.approx(np.log10(654.2 / 63.2))
        assert span_decades([100.0, 1000.0, 5000.0]) == pytest.approx(np.log10(50.0))

    def test_campaign_spans_classified(self):
        # measured wave-15 cells: adjacent pair fails, spread sets pass
        ok_adj, span_adj = validate_span([190.2, 205.9])
        assert not ok_adj
        assert span_adj == pytest.approx(0.034, abs=1e-3)
        for ok_set in ([63.2, 654.2], [63.2, 205.9, 654.2], [72.9, 122.2, 151.1]):
            ok, _ = validate_span(ok_set)
            want = span_decades(ok_set) >= MIN_SPAN_DECADES
            assert ok is want
            assert ok == (span_decades(ok_set) >= 0.4)

    def test_narrow_random_draw_fails_the_rule(self):
        # the failing k=3 random draw (0.32 decades) must be flagged
        ok, span = validate_span([72.9, 122.2, 151.1])
        assert not ok
        assert span == pytest.approx(0.317, abs=5e-3)

    def test_single_value_rejected(self):
        with pytest.raises(ValueError, match="at least two"):
            span_decades([100.0])


class TestMatchAnchorRows:
    def test_slender130_promotion(self):
        # the production case: 28 archived slender-1.30 rows, targets
        # min/geo-mid/max of the query range; rows at Re 63.2 / 205.9 /
        # 654.2 are the nearest and promote the 350-row hole corpus.
        rng = np.random.default_rng(0)
        # filler rows live in [240, 600]: far from all three anchor targets
        row_re = np.concatenate(([63.2, 654.2, 205.9], 240.0 + 360.0 * rng.random(25)))
        tgt = anchor_targets(63.2, 654.2, k=3)
        idx, achieved = match_anchor_rows(row_re, tgt)
        assert achieved[0] == pytest.approx(63.2)
        assert achieved[1] == pytest.approx(205.9)
        assert achieved[2] == pytest.approx(654.2)
        assert len(set(idx.tolist())) == 3
        # the promoted set must itself clear the span rule
        ok, span = validate_span(achieved)
        assert ok and span > 1.0

    def test_unsorted_rows_handled(self):
        row_re = np.array([654.2, 100.1, 63.2, 350.0, 205.9])
        idx, achieved = match_anchor_rows(row_re, anchor_targets(63.2, 654.2))
        assert achieved.tolist() == pytest.approx([63.2, 205.9, 654.2])
        assert row_re[idx].tolist() == pytest.approx(achieved.tolist())

    def test_missing_target_raises_with_hint(self):
        # no row near the low anchor -> the axis value needs a scan
        row_re = np.array([200.0, 350.0, 654.2])
        with pytest.raises(ValueError, match="scan this anchor"):
            match_anchor_rows(row_re, anchor_targets(63.2, 654.2), max_log10_distance=0.05)

    def test_duplicate_match_rejected(self):
        # both outer targets snap to the same sparse row
        row_re = np.array([120.0, 121.0, 122.0])
        with pytest.raises(ValueError, match="matched two anchor targets"):
            match_anchor_rows(row_re, anchor_targets(63.2, 654.2), max_log10_distance=2.0)

    def test_tolerance_widens_matching(self):
        row_re = np.array([70.0, 300.0, 600.0])
        with pytest.raises(ValueError, match="scan this anchor"):
            match_anchor_rows(row_re, anchor_targets(63.2, 654.2), max_log10_distance=0.01)
        idx, achieved = match_anchor_rows(
            row_re, anchor_targets(63.2, 654.2), max_log10_distance=0.2
        )
        assert achieved.tolist() == pytest.approx([70.0, 300.0, 600.0])

    def test_empty_rows_rejected(self):
        with pytest.raises(ValueError, match="no archived rows"):
            match_anchor_rows([], anchor_targets(63.2, 654.2))

    def test_nonpositive_re_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            match_anchor_rows([63.2, -1.0, 654.2], anchor_targets(63.2, 654.2))
