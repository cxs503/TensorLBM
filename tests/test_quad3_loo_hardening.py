"""Quad3 LOO degeneracy hardening: no NaN std in the served payload.

Regresses the 2026-09-01 UQ-campaign finding (pool381 measured cache:
157 rows / 126 distinct Re): when a design curve carries DUPLICATE cached
Re rows, ``quad3_loo_std`` returns ``None`` because a leave-one-out
pseudo-fit declines on a duplicate nearest-3 — yet the top-level
nearest-3 for an out-of-window query stays non-degenerate, so the point
IS served by quad3.  Before the hardening such a point carried
``std = nan`` (labelled ``unavailable_fewer_than_4_cached_rows``, wrong
on a 157-row curve); the hardened contract keeps the network ensemble
std for the point and records the true cause in the policy info.

Deliberately NOT changed (asserted here): the default-off status, the
quad3 prediction values, and the std semantics when the LOO rms IS
computable.
"""

from __future__ import annotations

import numpy as np
import pytest

# TestModelBackend is aliased to a non-Test name so pytest does not
# re-collect the fixture class into THIS module (it runs in its own file).
from test_inference_service import TEST_GRID, _syn_corpus
from test_inference_service import TestModelBackend as _FixtureService

from tensorlbm.ai.inference_service import (
    RE_POLICY_NETWORK,
    RE_POLICY_QUAD3_FALLBACK,
    DragSurrogateService,
    EnvelopeMahalanobisGuardrail,
    quad3_loo_std,
    quad3_nearest3,
)

#: Extra measured rows appended to the base fixture corpus.  Re-running
#: the SAME design at the SAME Re is exactly how duplicates entered the
#: production cache.  ("full", 1.0, 1.0) gains a second Re=50 row -> its
#: curve is [50, 50, 64, 81, 100] (5 rows, 4 distinct Re): the nearest-3
#: for a query above the window is [64, 81, 100] (non-degenerate, quad3
#: applies) while the leave-one-out pseudo-fit at Re=64 hits the
#: duplicate pair (LOO unavailable — degeneracy, not row count).
_DUP_ROW = ("full", 1.0, 1.0, 50.0)

#: ("bare_hull", 1.0, 1.0) gains Re=81 -> curve [64, 81, 100]: only 3
#: cached rows, so quad3 applies out-of-window but leave-one-out needs
#: at least 4 rows (the honest row-count case of LOO unavailability).
_THIN_ROW = ("bare_hull", 1.0, 1.0, 81.0)


def _svc_with_extra_curve_rows(
    extra: tuple[tuple[str, float, float, float], ...],
) -> DragSurrogateService:
    """Base test service + extra measured rows on given designs.

    Members / guard come from the shared fixture (deterministic seeds, so
    the network std at a point is reproducible call-to-call); only the
    measured-curve index is extended.  Extra C_D rows sit ~3% off the
    base curve — a plausible re-run, never a duplicate C_D.
    """
    base = _FixtureService()._service()  # noqa: SLF001 — fixture reuse
    corpus = _syn_corpus()
    assert base.cache_re is not None and base.cache_cd is None
    u_in = float(corpus["uin"][0])  # the design-key u_in slot (0.1 here)
    designs = list(base.cache_designs)
    designs.extend((h, s, f, u_in) for h, s, f, _ in extra)
    rows_cd = [float(corpus["cd"][0]) * (1.03 + 0.01 * k) for k in range(len(extra))]
    rng = np.random.default_rng(11)
    fields = np.concatenate(
        [
            np.asarray(base.corpus_cache),
            rng.uniform(0.0, 0.2, size=(len(extra), 5, TEST_GRID.ny, TEST_GRID.nx)),
        ]
    )
    cond = np.concatenate([corpus["cond"], corpus["cond"][[0] * len(extra)]])
    return DragSurrogateService(
        base.backend,
        EnvelopeMahalanobisGuardrail(cond),
        corpus_cache=fields,
        grid=base.grid,
        cache_re=np.concatenate([base.cache_re, [r for _, _, _, r in extra]]),
        cache_designs=designs,
        cache_cd=np.concatenate([corpus["cd"], rows_cd]),
    )


def _design_curve(
    svc: DragSurrogateService, key: tuple[str, float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Cached (re, cd) rows of one design of the extended cache, ascending."""
    assert svc.cache_re is not None and svc.cache_designs is not None and svc.cache_cd is not None
    sel = [i for i, d in enumerate(svc.cache_designs) if d[:3] == key]
    order = np.argsort(svc.cache_re[sel], kind="stable")
    return (
        np.asarray(svc.cache_re)[sel][order],
        np.asarray(svc.cache_cd)[sel][order],
    )


class TestQuad3LooHardening:
    """re_policy=quad3_fallback under LOO-unavailable caches."""

    def test_duplicate_re_curve_serves_network_std_not_nan(self) -> None:
        svc = _svc_with_extra_curve_rows((_DUP_ROW,))
        curve_re, curve_cd = _design_curve(svc, ("full", 1.0, 1.0))
        assert curve_re.tolist() == [50.0, 50.0, 64.0, 81.0, 100.0]
        assert quad3_loo_std(curve_re, curve_cd) is None  # degeneracy, NOT row count

        ref = svc.predict("full", 1.0, 1.0, [300.0])  # default: network path
        res = svc.predict("full", 1.0, 1.0, [300.0], re_policy=RE_POLICY_QUAD3_FALLBACK)
        pol = res.info["re_policy"]

        # The quad3 value itself is untouched by the hardening.
        assert pol["method"] == RE_POLICY_QUAD3_FALLBACK
        assert pol["n_quad3_points"] == 1
        val, sel_re = quad3_nearest3(curve_re, curve_cd, 300.0)
        assert sel_re.tolist() == [64.0, 81.0, 100.0]
        assert res.cd[0] == pytest.approx(val, rel=1e-12)

        # Hardened contract: finite std == the network ensemble std of the
        # same service at the same point (kept bit-identical), never NaN.
        assert np.isfinite(ref.std[0]) and ref.std[0] > 0.0
        assert np.isfinite(res.std[0])
        assert res.std[0] == ref.std[0]
        assert pol["loo_rel_rms"] is None
        assert pol["std_source"] == "network_ensemble_std_loo_degenerate"
        assert pol["quad3_loo_degenerate"] is True
        assert pol["quad3_loo_duplicate_re_rows"] == 1
        # No member band exists for a measured curve — lo/hi stay NaN.
        assert np.isnan(res.lo[0]) and np.isnan(res.hi[0])

    def test_three_row_curve_serves_network_std_not_nan(self) -> None:
        """LOO unavailable from row count (<4) must not serve NaN either."""
        svc = _svc_with_extra_curve_rows((_THIN_ROW,))
        curve_re, _ = _design_curve(svc, ("bare_hull", 1.0, 1.0))
        assert curve_re.tolist() == [64.0, 81.0, 100.0]

        ref = svc.predict("bare_hull", 1.0, 1.0, [300.0])
        res = svc.predict("bare_hull", 1.0, 1.0, [300.0], re_policy=RE_POLICY_QUAD3_FALLBACK)
        pol = res.info["re_policy"]
        assert pol["method"] == RE_POLICY_QUAD3_FALLBACK
        assert pol["n_quad3_points"] == 1
        assert np.isfinite(res.std[0])
        assert res.std[0] == ref.std[0]
        assert pol["loo_rel_rms"] is None
        assert pol["std_source"] == "network_ensemble_std_loo_needs_4_cached_rows"
        # The duplicate-Re flag is reserved for actual degeneracy.
        assert "quad3_loo_degenerate" not in pol
        assert "quad3_loo_duplicate_re_rows" not in pol

    def test_clean_cache_behavior_unchanged(self) -> None:
        """LOO computable: std = cd * LOO rms and no degeneracy keys."""
        svc = _svc_with_extra_curve_rows(())
        res = svc.predict("full", 1.0, 1.0, [300.0], re_policy=RE_POLICY_QUAD3_FALLBACK)
        pol = res.info["re_policy"]
        curve_re, curve_cd = _design_curve(svc, ("full", 1.0, 1.0))
        assert curve_re.tolist() == [50.0, 64.0, 81.0, 100.0]
        loo = quad3_loo_std(curve_re, curve_cd)
        assert loo is not None
        val, _ = quad3_nearest3(curve_re, curve_cd, 300.0)
        assert pol["n_quad3_points"] == 1
        assert res.cd[0] == pytest.approx(val, rel=1e-12)
        assert res.std[0] == pytest.approx(val * loo, rel=1e-12)
        assert pol["std_source"] == "quad3_loo_relative_rms_times_cd"
        assert "quad3_loo_degenerate" not in pol
        assert "quad3_loo_duplicate_re_rows" not in pol

    def test_default_off_untouched_even_on_duplicate_cache(self) -> None:
        svc = _svc_with_extra_curve_rows((_DUP_ROW,))
        grid = np.array([64.0, 300.0, 41.0])  # includes triggered points
        a = svc.predict("full", 1.0, 1.0, grid)
        b = svc.predict("full", 1.0, 1.0, grid, re_policy=RE_POLICY_NETWORK)
        for arr in ("cd", "std", "lo", "hi"):
            assert getattr(a, arr).tobytes() == getattr(b, arr).tobytes()
        assert "re_policy" not in a.info
        assert "re_policy" not in b.info
        # With the policy ON, the in-window point keeps the network std
        # bit-identical too (per-point routing); Re=41 declines on the
        # duplicate nearest-3 ([50, 50, 64]) and keeps the network value —
        # so nothing on this grid is NaN.
        on = svc.predict("full", 1.0, 1.0, grid, re_policy=RE_POLICY_QUAD3_FALLBACK)
        assert on.info["re_policy"]["quad3_mask"] == [False, True, False]
        assert any("re=41" in d for d in on.info["re_policy"]["declined_points"])
        assert on.std[0] == a.std[0]
        assert on.std[2] == a.std[2]
        assert np.isfinite(on.std).all()
