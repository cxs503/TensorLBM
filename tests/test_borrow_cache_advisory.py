"""Tests for the 2026-09-08 borrow follow-ups: result cache + distance advisory.

Two additive behaviours on top of the 2026-09-04 borrow path, pinned
against small pure-numpy fixtures (no torch / GPU):

- the per-geometry LRU result cache of :class:`FieldProvider`
  (follow-up #1): a repeat query returns the SAME
  :class:`~tensorlbm.ai.field_provider.BorrowedField` (content-keyed,
  never object identity), hit outputs equal miss outputs byte-for-byte,
  the bound evicts least-recently-used, ``cache_size=0`` disables, and
  validation/pool-safety semantics are unchanged by the cache;
- the out-of-family ``distance_advisory`` of
  :func:`tensorlbm.ai.field_borrow.borrow_serving_field` (follow-up
  #2): in-family / suspect levels against a known threshold, the four
  exact keys, the calibrated default, non-blocking (info only — no
  raise, no warning, same served field), absent for the distance-less
  ``mean`` fallback.

The flag-off byte-identity of the service stays pinned by the untouched
``TestServiceHookCache::test_default_off_is_byte_identical`` of
``tests/test_field_borrow.py``.

Evidence: ``/nfs/wangxi/runs/borrow_cache_20260908/`` (calibration.json
+ evidence.json; the 57.7 ms retrieval cost this cache amortizes was
measured in ``/nfs/wangxi/runs/l2_walkthrough_20260906/``).
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from tensorlbm.ai.field_borrow import borrow_serving_field
from tensorlbm.ai.field_provider import (
    DEFAULT_BORROW_CACHE_SIZE,
    DISTANCE_ADVISORY_THRESHOLD,
    FIELD_CHANNELS,
    FieldProvider,
)

_SDF_SHAPE = (3, 4, 4)


def make_pool(n: int = 4, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(n, 5, 4, 4) field pool + (n, 3, 4, 4) SDF pool, guard-friendly.

    Fields sit tight around a common mean (rel-L2 to the pool mean well
    under the 0.15 in-manifold threshold), like real corpus rows.
    """
    rng = np.random.default_rng(seed)
    fields = rng.uniform(0.09, 0.11, size=(n, FIELD_CHANNELS, 4, 4)).astype(np.float32)
    sdfs = rng.standard_normal((n, *_SDF_SHAPE)).astype(np.float32)
    return fields, sdfs


def shifted(sdf: np.ndarray, delta: float) -> np.ndarray:
    """Query at exact float64 SDF-L2 distance ``delta`` from ``sdf``."""
    return (sdf + np.float32(delta / np.sqrt(sdf.size))).astype(np.float32)


# ---------------------------------------------------------------------------
# Follow-up #1 — the per-geometry borrow-result cache
# ---------------------------------------------------------------------------
class TestBorrowCache:
    def test_repeat_query_returns_the_same_object(self) -> None:
        fields, sdfs = make_pool(n=4)
        provider = FieldProvider(fields, pool_sdfs=sdfs)
        target = shifted(sdfs[2], 0.5)
        a = provider.borrow(target_sdf=target)
        b = provider.borrow(target_sdf=target.copy())  # new array object, same content
        assert b is a  # content key, never object identity
        np.testing.assert_array_equal(b.fields, fields[2])
        assert b.donor_index == 2 == a.donor_index

    def test_hit_outputs_equal_miss_outputs_byte_for_byte(self) -> None:
        fields, sdfs = make_pool(n=4)
        target = shifted(sdfs[1], 0.3)
        cached = FieldProvider(fields, pool_sdfs=sdfs).borrow(target_sdf=target)
        uncached = FieldProvider(fields, pool_sdfs=sdfs, cache_size=0).borrow(target_sdf=target)
        assert cached.donor_index == uncached.donor_index
        assert cached.distance == uncached.distance
        assert cached.guard_ok == uncached.guard_ok is True
        assert cached.guard_rel_l2 == uncached.guard_rel_l2
        assert cached.fields.tobytes() == uncached.fields.tobytes()
        assert cached.provenance == uncached.provenance

    def test_serving_info_is_identical_on_repeat_queries(self) -> None:
        fields, sdfs = make_pool(n=4)
        provider = FieldProvider(fields, pool_sdfs=sdfs)
        a = borrow_serving_field(provider, shifted(sdfs[0], 0.2))
        b = borrow_serving_field(provider, shifted(sdfs[0], 0.2))
        assert a.info == b.info
        assert a.fields.tobytes() == b.fields.tobytes()
        assert a.info["field_borrow"]["donor_index"] == 0

    def test_lru_bound_evicts_least_recently_used(self) -> None:
        fields, sdfs = make_pool(n=4)
        provider = FieldProvider(fields, pool_sdfs=sdfs, cache_size=2)
        a1 = provider.borrow(target_sdf=shifted(sdfs[0], 0.1))
        provider.borrow(target_sdf=shifted(sdfs[1], 0.1))
        provider.borrow(target_sdf=shifted(sdfs[2], 0.1))  # overflows: entry 0 evicted
        a1_again = provider.borrow(target_sdf=shifted(sdfs[0], 0.1))
        assert a1_again is not a1  # recomputed, not replayed from the cache
        assert a1_again.donor_index == a1.donor_index == 0
        assert a1_again.distance == a1.distance
        assert a1_again.fields.tobytes() == a1.fields.tobytes()

    def test_lru_refresh_on_hit_protects_the_hot_entry(self) -> None:
        fields, sdfs = make_pool(n=4)
        provider = FieldProvider(fields, pool_sdfs=sdfs, cache_size=2)
        hot = provider.borrow(target_sdf=shifted(sdfs[0], 0.1))
        provider.borrow(target_sdf=shifted(sdfs[1], 0.1))
        assert provider.borrow(target_sdf=shifted(sdfs[0], 0.1)) is hot  # refresh
        provider.borrow(target_sdf=shifted(sdfs[2], 0.1))  # evicts entry 1, not the hot one
        assert provider.borrow(target_sdf=shifted(sdfs[0], 0.1)) is hot

    def test_cache_size_zero_disables(self) -> None:
        fields, sdfs = make_pool(n=3)
        provider = FieldProvider(fields, pool_sdfs=sdfs, cache_size=0)
        assert provider._borrow_cache is None
        target = shifted(sdfs[2], 0.4)
        a = provider.borrow(target_sdf=target)
        b = provider.borrow(target_sdf=target)
        assert b is not a  # every call recomputes
        assert b.donor_index == a.donor_index
        assert b.fields.tobytes() == a.fields.tobytes()

    def test_distinct_queries_do_not_collide(self) -> None:
        fields, sdfs = make_pool(n=3)
        provider = FieldProvider(fields, pool_sdfs=sdfs)
        a = provider.borrow(target_sdf=shifted(sdfs[0], 0.05))
        b = provider.borrow(target_sdf=shifted(sdfs[1], 0.05))
        assert a.donor_index == 0
        assert b.donor_index == 1
        assert provider.borrow(target_sdf=shifted(sdfs[0], 0.05)).fields.tobytes() == (
            a.fields.tobytes()
        )

    def test_cache_never_bypasses_validation(self) -> None:
        fields, sdfs = make_pool(n=3)
        provider = FieldProvider(fields, pool_sdfs=sdfs)
        good = shifted(sdfs[0], 0.1)
        assert provider.borrow(target_sdf=good).donor_index == 0  # primes the cache
        assert provider.borrow(target_sdf=good).donor_index == 0  # hit path works
        with pytest.raises(ValueError, match="target_sdf must have shape"):
            provider.borrow(target_sdf=np.zeros((5, 5), dtype=np.float32))  # miss + raise
        with pytest.raises(ValueError, match="unknown strategy"):
            provider.borrow(target_sdf=good, strategy="sdf")  # hit cannot happen pre-validation

    def test_pool_is_never_poisoned_by_returned_fields(self) -> None:
        fields, sdfs = make_pool(n=3)
        provider = FieldProvider(fields, pool_sdfs=sdfs)
        got = provider.borrow(target_sdf=sdfs[0])
        got.fields[...] = 0.0  # documented read-only contract broken by the caller
        assert not np.array_equal(fields[0], np.zeros_like(fields[0]))  # pool safe

    def test_bad_cache_size_rejected(self) -> None:
        fields, sdfs = make_pool(n=2)
        for bad in (-1, 1.5, True):
            with pytest.raises(ValueError, match="cache_size"):
                FieldProvider(fields, pool_sdfs=sdfs, cache_size=bad)

    def test_default_is_on_with_the_documented_size(self) -> None:
        fields, sdfs = make_pool(n=2)
        provider = FieldProvider(fields, pool_sdfs=sdfs)
        assert DEFAULT_BORROW_CACHE_SIZE == 16
        assert provider.cache_size == DEFAULT_BORROW_CACHE_SIZE
        assert provider._borrow_cache is not None


# ---------------------------------------------------------------------------
# Follow-up #2 — the out-of-family distance advisory
# ---------------------------------------------------------------------------
class TestDistanceAdvisory:
    def test_in_family_level(self) -> None:
        fields, sdfs = make_pool(n=3)
        provider = FieldProvider(fields, pool_sdfs=sdfs, distance_advisory_threshold=2.0)
        got = borrow_serving_field(provider, shifted(sdfs[0], 1.0))
        adv = got.info["field_borrow"]["distance_advisory"]
        assert adv["level"] == "in_family"
        assert adv["threshold"] == 2.0
        assert adv["reference"] == "corpus_loo_nn"
        assert adv["distance_ratio"] == pytest.approx(0.5, rel=1e-5)

    def test_suspect_level(self) -> None:
        fields, sdfs = make_pool(n=3)
        provider = FieldProvider(fields, pool_sdfs=sdfs, distance_advisory_threshold=2.0)
        got = borrow_serving_field(provider, shifted(sdfs[0], 3.0))
        adv = got.info["field_borrow"]["distance_advisory"]
        assert adv["level"] == "suspect"
        assert adv["distance_ratio"] == pytest.approx(1.5, rel=1e-5)

    def test_exact_keys(self) -> None:
        fields, sdfs = make_pool(n=2)
        provider = FieldProvider(fields, pool_sdfs=sdfs)
        adv = borrow_serving_field(provider, sdfs[0]).info["field_borrow"]["distance_advisory"]
        assert sorted(adv) == ["distance_ratio", "level", "reference", "threshold"]

    def test_calibrated_default_threshold_is_present(self) -> None:
        fields, sdfs = make_pool(n=2)
        provider = FieldProvider(fields, pool_sdfs=sdfs)  # no explicit threshold
        assert provider.distance_advisory_threshold == DISTANCE_ADVISORY_THRESHOLD
        adv = borrow_serving_field(provider, sdfs[0]).info["field_borrow"]["distance_advisory"]
        assert adv["threshold"] == DISTANCE_ADVISORY_THRESHOLD
        # pin the 2026-09-08 calibration itself (corpus design-LOO max, see
        # /nfs/wangxi/runs/borrow_cache_20260908/calibration.json)
        assert DISTANCE_ADVISORY_THRESHOLD == pytest.approx(8.875449208906158, abs=1e-9)
        assert adv["level"] == "in_family"  # distance 0.0 <= threshold

    def test_cond_near_distance_feeds_the_same_rule(self) -> None:
        # borrow_serving_field is sdf-shaped, so cond_near reaches the
        # advisory only via provider.borrow; the rule (level from the
        # distance vs threshold) is the same distance branch.
        fields, _sdfs = make_pool(n=3)
        cond = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 5.0]])
        provider = FieldProvider(fields, pool_cond=cond, distance_advisory_threshold=1.0)
        near = provider.borrow(target_cond=np.array([0.0, 0.0]), strategy="cond_near")
        far = provider.borrow(target_cond=np.array([50.0, 50.0]), strategy="cond_near")
        assert near.distance is not None and far.distance is not None
        assert near.distance <= 1.0 < far.distance
        # mirror of the wrapper rule on these two distances:
        for dist, expected in ((near.distance, "in_family"), (far.distance, "suspect")):
            level = "suspect" if dist > provider.distance_advisory_threshold else "in_family"
            assert level == expected

    def test_mean_strategy_has_no_advisory(self) -> None:
        fields, sdfs = make_pool(n=2)
        provider = FieldProvider(fields, pool_sdfs=sdfs)
        got = borrow_serving_field(provider, sdfs[0], strategy="mean")
        assert got.info["field_borrow"]["distance"] is None
        assert "distance_advisory" not in got.info["field_borrow"]

    def test_advisory_is_non_blocking(self, caplog: pytest.LogCaptureFixture) -> None:
        fields, sdfs = make_pool(n=3)
        provider = FieldProvider(fields, pool_sdfs=sdfs, distance_advisory_threshold=0.5)
        target = shifted(sdfs[0], 3.0)  # distance 3.0 >> 0.5 -> suspect
        with caplog.at_level(logging.WARNING, logger="tensorlbm.ai.field_borrow"):
            got = borrow_serving_field(provider, target)
        adv = got.info["field_borrow"]["distance_advisory"]
        assert adv["level"] == "suspect"
        assert got.info["field_borrow"]["guard_ok"] is True  # field itself is in-manifold
        np.testing.assert_array_equal(got.fields, fields[0])  # still serves the donor field
        assert caplog.text == ""  # info only: no warning, no raise

    def test_bad_threshold_rejected(self) -> None:
        fields, sdfs = make_pool(n=2)
        for bad in (0.0, -1.0, float("inf"), float("nan")):
            with pytest.raises(ValueError, match="distance_advisory_threshold"):
                FieldProvider(fields, pool_sdfs=sdfs, distance_advisory_threshold=bad)
