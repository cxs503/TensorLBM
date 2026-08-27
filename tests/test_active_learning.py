"""Tests for ``tensorlbm.ai.active_learning`` (B4-P3b) — synthetic, CPU-only.

No GPU and no /nfs paths: the corpus, the family cache (npz + meta json)
and the ensemble checkpoints are fabricated in ``tmp_path``; the CAD
geometry front-end runs on a resolution-64 grid (the smallest grid the
production ARCH modes fit).  The mother-design parity assertions pin the
fit-time exactness contract (bit-identical v3 channels), and the split
test pins the protocol-stability property the incremental retrain relies
on (original rows keep their fit/val/test membership after augmentation).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from tensorlbm.ai.active_learning import (
    ARM_CFG,
    HULLFORM_AXES,
    AcquisitionLabel,
    AcquisitionPoint,
    FlaggedQuery,
    MotherEval,
    ServiceSpec,
    TrendSpec,
    _exclusion_floor,
    _params_config,
    append_fam_fragment,
    augment_corpus,
    axes_envelope,
    corpus_cond_v3,
    corpus_cond_v5,
    corpus_design_keys,
    corpus_param_keys,
    corpus_point_keys,
    default_fresh_re_grid,
    eval_loop,
    fit_stats,
    honest_verdict,
    hullform_component_counts,
    hullform_condition_rows,
    hullform_condition_rows_v5,
    hullform_geo_block,
    labels_from_cache,
    load_fam_fragment,
    point_param_key,
    predict_design,
    propose_acquisition,
    retrain_ensemble,
    spearman_rho,
    split_random,
    trend_stat,
    write_loop_report,
)
from tensorlbm.ai.drag_cond import (
    CondFNODrag,
    SuboffGrid,
    condition_v3,
    geometry_channels,
    suboff_geometry_features,
)
from tensorlbm.ai.inference_service import (
    FLAG_OK,
    FLAG_REJECT,
    FLAG_REVIEW,
    CondDragCheckpoint,
    EnvelopeMahalanobisGuardrail,
    GuardVerdict,
    save_checkpoint,
)
from tensorlbm.suboff_cad import SuboffConfig

# resolution 64 -> (nz, ny, nx) = (32, 32, 64): the smallest grid that fits
# the production ARCH modes (16, 32) (rfft2 keeps nx // 2 + 1 = 33 x-modes).
TEST_GRID = SuboffGrid.from_resolution(64)
MOTHER_ENV = {axis: (1.0, 1.0) for axis in HULLFORM_AXES}

# Real mother designs (hull/scale variety) whose geometry blocks span the
# channel envelope the way the true corpus does.
DESIGNS = [
    ("bare_hull", 1.0, 1.0),
    ("with_sail", 1.0, 1.0),
    ("with_sail", 1.3, 1.0),
    ("full", 1.0, 1.0),
    ("full", 1.0, 1.5),
    ("full", 0.7, 1.2),
]
RES = np.array([80.0, 120.0, 200.0, 300.0, 450.0, 600.0])
HULLS = ("bare_hull", "with_sail", "full")


def _design_geo(hull: str, sail: float, fin: float) -> np.ndarray:
    return np.asarray(geometry_channels(suboff_geometry_features(hull, sail, fin, grid=TEST_GRID)))


def _family_params(shape: dict[str, float]) -> dict[str, Any]:
    params: dict[str, Any] = {"hull_type": "with_sail", "sail_scale": 1.0, "fin_scale": 1.0}
    params.update({a: 1.0 for a in HULLFORM_AXES})
    params.update(shape)
    return params


def make_corpus(n_per_design: int = 5, seed: int = 7) -> dict[str, np.ndarray]:
    """Synthetic cache_v4-layout corpus over the real design envelopes."""
    rng = np.random.default_rng(seed)
    rows = [
        (hull, sail, fin, float(re), di % 3)
        for di, (hull, sail, fin) in enumerate(DESIGNS)
        for re in RES[:n_per_design]
    ]
    n = len(rows)
    return dict(
        x=rng.normal(0.5, 0.1, (n, 5, TEST_GRID.ny, TEST_GRID.nx)).astype(np.float32),
        dsi=np.array([d for _h, _s, _f, _r, d in rows]),
        re=np.array([r for _h, _s, _f, r, _d in rows]),
        uin=np.full(n, 0.1),
        sail=np.array([s for _h, s, _f, _r, _d in rows]),
        fin=np.array([f for _h, _s, f, _r, _d in rows]),
        hull=np.array([HULLS.index(h) for h, _s, _f, _r, _d in rows]),
        step=np.full(n, 4000, dtype=np.int64),
        aproj=np.full(n, 69, dtype=np.int64),
        cd=20.0
        * np.array([r for _h, _s, _f, r, _d in rows]) ** -0.42
        * (1.0 + 0.05 * np.arange(n)),
        geo=np.stack([_design_geo(h, s, f) for h, s, f, _r, _d in rows]),
        aux=rng.normal(0.5, 0.05, (n, 8)),
        mask_bit_eq=np.zeros(n, dtype=bool),
        v_sail=np.full(n, 331, dtype=np.int64),
        v_fin=np.full(n, 36, dtype=np.int64),
        v_solid=np.full(n, 4162, dtype=np.int64),
        aproj_cad=np.full(n, 69, dtype=np.int64),
        aproj_bare=np.full(n, 65, dtype=np.int64),
    )


def write_fam_cache(tmp_path: Path) -> Path:
    """Synthetic B4-fam family cache (npz + meta json) in tmp_path."""
    rng = np.random.default_rng(11)
    fams = [
        ("fam_blunt", 6, {"l_over_d_mult": 0.75}),
        ("fam_slender", 7, {"l_over_d_mult": 1.30}),
    ]
    res = [110.0, 210.0, 330.0, 520.0]
    meta: list[dict[str, object]] = []
    cols: dict[str, list[Any]] = {
        k: []
        for k in (
            "x",
            "dsi",
            "re",
            "uin",
            "sail",
            "fin",
            "hull",
            "step",
            "aproj",
            "cd",
            "geo",
            "aux",
            "mask_bit_eq",
            "fam",
        )
    }
    for fam_name, fam_dsi, axes in fams:
        for re in res:
            base = {a: 1.0 for a in HULLFORM_AXES}
            base.update(axes)
            meta.append(
                {
                    "hull": "with_sail",
                    "sail": 1.0,
                    "fin": 1.0,
                    "u_in": 0.1,
                    "fam": fam_name,
                    "re": re,
                    **base,
                }
            )
            cols["x"].append(
                rng.normal(0.5, 0.1, (5, TEST_GRID.ny, TEST_GRID.nx)).astype(np.float32)
            )
            cols["dsi"].append(np.int64(fam_dsi))
            cols["re"].append(np.float64(re))
            cols["uin"].append(np.float64(0.1))
            cols["sail"].append(np.float64(1.0))
            cols["fin"].append(np.float64(1.0))
            cols["hull"].append(np.int64(1))
            cols["step"].append(np.int64(4000))
            cols["aproj"].append(np.int64(69))
            cols["cd"].append(np.float64(18.0 * re**-0.42))
            cols["geo"].append(rng.normal(0.0, 0.01, 4))
            cols["aux"].append(rng.normal(0.5, 0.05, 8))
            cols["mask_bit_eq"].append(False)
            cols["fam"].append(np.int64(fam_dsi))
    npz = {k: np.stack(v) for k, v in cols.items()}
    path = tmp_path / "cache_fam.npz"
    np.savez(path, **npz)  # type: ignore[arg-type]
    path.with_name("cache_fam_meta.json").write_text(json.dumps(meta))
    return path


def flagged_queries(existing_cond: np.ndarray, grid: SuboffGrid = TEST_GRID) -> list[FlaggedQuery]:
    """Flagged queries carrying their real guard scores (as the service logs them).

    Includes two-axis corners so the flagged score band (the acquisition
    shell) has realistic width.
    """
    shapes = [
        {"l_over_d_mult": 0.75},
        {"l_over_d_mult": 1.30},
        {"nose_len_mult": 1.30},
        {"sail_x_mult": 1.30},
        {"l_over_d_mult": 0.75, "nose_len_mult": 1.30},
        {"l_over_d_mult": 1.30, "sail_x_mult": 1.30},
    ]
    guard = EnvelopeMahalanobisGuardrail(existing_cond)
    out = []
    for i, shape in enumerate(shapes):
        params = _family_params(shape)
        for re in (RES[i % 4], RES[(i + 2) % 4]):
            cond = hullform_condition_rows(params, [float(re)], grid=grid)
            out.append(
                FlaggedQuery(
                    params=params,
                    re=float(re),
                    verdict="review",
                    score=float(guard.row_scores(cond)[0]),
                    member_std=0.05,
                )
            )
    return out


def write_fake_ckpt_dir(
    tmp_path: Path, corpus: dict[str, np.ndarray], tag: str, seed: int = 3
) -> Path:
    """Two tiny CondFNODrag members as serving-format checkpoints."""
    cond = corpus_cond_v3(corpus)
    ylog = np.log10(corpus["cd"])
    st = fit_stats(corpus["x"], cond, ylog, corpus["aux"], list(range(len(ylog))))
    out = tmp_path / f"ckpts_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    for k in range(2):
        torch.manual_seed(seed * 100 + k)
        arch: dict[str, Any] = dict(
            in_ch=5, width=4, n_layers=2, modes=(4, 8), cond_dim=8, aux_dim=0
        )
        net = CondFNODrag(**arch)
        ckpt = CondDragCheckpoint(
            arch=arch,
            state_dict={kk: v.detach().cpu() for kk, v in net.state_dict().items()},
            norm=dict(
                ch_mean=st["ch_mean"],
                ch_std=st["ch_std"],
                p_mean=st["p_mean"],
                p_std=st["p_std"],
                y_mean=st["y_mean"],
                y_std=st["y_std"],
            ),
            meta=dict(arm="C_full", seed=k, member=f"m{k}"),
        )
        save_checkpoint(ckpt, out / f"m{k}.pt")
    return out


# ---------------------------------------------------------------------------
# Geometry front-end
# ---------------------------------------------------------------------------


class TestGeometryFrontend:
    def test_mother_geo_bitwise_parity(self) -> None:
        """Fit-time exactness: mother designs reproduce drag_cond bitwise."""
        for hull, sail, fin in DESIGNS:
            mine = hullform_geo_block(hull, sail, fin, TEST_GRID, None)
            ref = _design_geo(hull, sail, fin)
            assert np.array_equal(mine, ref), (hull, sail, fin)

    def test_variant_moves_channels(self) -> None:
        mother = hullform_geo_block("with_sail", 1.0, 1.0, TEST_GRID, None)
        slender = hullform_geo_block(
            "with_sail", 1.0, 1.0, TEST_GRID, SuboffConfig(l_over_d_mult=1.3)
        )
        assert not np.array_equal(mother, slender)
        assert slender[0] > mother[0]  # thinner bare outline -> larger aproj ratio
        assert slender[3] > mother[3]  # same sail over a smaller hull -> larger solid frac

    def test_condition_rows_parity(self) -> None:
        params = {"hull_type": "with_sail", "sail_scale": 1.0, "fin_scale": 1.0}
        mine = hullform_condition_rows(params, RES[:3], grid=TEST_GRID)
        geo = _design_geo("with_sail", 1.0, 1.0)
        ref = condition_v3(
            RES[:3], np.full(3, 0.1), np.ones(3), np.ones(3), np.broadcast_to(geo, (3, 4))
        )
        assert mine.shape == (3, 8)
        assert np.array_equal(mine, ref)

    def test_component_counts_and_param_validation(self) -> None:
        counts = hullform_component_counts("with_sail", 1.0, 1.0, TEST_GRID, None)
        v_bare, v_sail, v_fin, v_solid, aproj, aproj_bare = counts
        assert v_bare > 0 and v_solid == v_bare + v_sail + v_fin
        assert aproj >= aproj_bare > 0
        with pytest.raises(ValueError, match="unknown design params"):
            _params_config({"banana": 2.0})
        with pytest.raises(ValueError, match="finite and positive"):
            hullform_condition_rows({"hull_type": "with_sail"}, [-1.0], grid=TEST_GRID)


# ---------------------------------------------------------------------------
# Acquisition
# ---------------------------------------------------------------------------


class TestProposeAcquisition:
    def _existing(self) -> np.ndarray:
        return np.asarray(corpus_cond_v3(make_corpus()))

    def test_all_strategies_budget_and_determinism(self) -> None:
        existing = self._existing()
        queries = flagged_queries(existing)

        def std_fn(pts: list[AcquisitionPoint]) -> np.ndarray:
            return np.array([0.1 + 0.8 * ((i * 7) % 11) / 11.0 for i in range(len(pts))])

        for strategy in ("envelope_shell", "max_disagreement", "coverage"):
            kwargs: dict[str, Any] = (
                {"member_std_fn": std_fn} if strategy == "max_disagreement" else {}
            )
            a = propose_acquisition(
                queries,
                strategy=strategy,
                budget=5,
                existing_cond=existing,
                grid=TEST_GRID,
                seed=5,
                n_candidates=96,
                **kwargs,
            )
            b = propose_acquisition(
                queries,
                strategy=strategy,
                budget=5,
                existing_cond=existing,
                grid=TEST_GRID,
                seed=5,
                n_candidates=96,
                **kwargs,
            )
            assert len(a) == 5, f"{strategy}: got {len(a)} points"
            assert [p.key for p in a] == [p.key for p in b], strategy  # bitwise rerun
            assert all(p.strategy == strategy for p in a)

    def test_returned_points_respect_exclusion_floor(self) -> None:
        existing = self._existing()
        queries = flagged_queries(existing)
        guard = EnvelopeMahalanobisGuardrail(existing)
        floor = _exclusion_floor(existing, guard)
        for strategy in ("envelope_shell", "coverage"):
            kwargs: dict[str, Any] = {"n_candidates": 128} if strategy == "envelope_shell" else {}
            pts = propose_acquisition(
                queries,
                strategy=strategy,
                budget=6,
                existing_cond=existing,
                grid=TEST_GRID,
                seed=2,
                **kwargs,
            )
            assert pts, strategy
            for p in pts:
                cond = hullform_condition_rows(p.params, [p.re], grid=TEST_GRID)
                assert float(guard.row_scores(cond)[0]) >= floor, (strategy, p.key)

    def test_existing_cloud_exclusion_drops_in_cloud_candidate(self) -> None:
        queries = flagged_queries(self._existing())
        # dense tight cloud exactly AT the blunt shape / re=200 candidate
        blunt = _family_params({"l_over_d_mult": 0.75})
        cloud_row = hullform_condition_rows(blunt, [200.0], grid=TEST_GRID)[0]
        rng = np.random.default_rng(0)
        cloud = cloud_row[None, :] + rng.normal(0.0, 1e-6, (30, 8))
        pts = propose_acquisition(
            queries, strategy="coverage", budget=6, existing_cond=cloud, grid=TEST_GRID
        )
        assert pts, "the non-blunt corners must survive"
        for p in pts:
            is_blunt_at_cloud = (
                abs(p.params["l_over_d_mult"] - 0.75) < 1e-9 and abs(p.re - 200.0) < 1e-9
            )
            assert not is_blunt_at_cloud, "in-cloud candidate must be excluded"
        assert any(abs(p.params["l_over_d_mult"] - 1.30) < 1e-9 for p in pts)

    def test_coverage_corners_and_re_provenance(self) -> None:
        queries = flagged_queries(self._existing())
        pts = propose_acquisition(
            queries,
            strategy="coverage",
            budget=6,
            existing_cond=self._existing(),
            grid=TEST_GRID,
        )
        flagged_res = {q.re for q in queries}
        for p in pts:
            moved = [a for a in HULLFORM_AXES if abs(p.params[a] - 1.0) > 1e-9]
            assert len(moved) == 1, p.params  # single-axis corner
            assert p.params[moved[0]] in (0.75, 1.30)
            assert p.re in flagged_res  # Re drawn from the flagged region
        assert len({p.key for p in pts}) == len(pts)  # unique keys

    def test_coverage_re_is_family_consistent(self) -> None:
        """Corners pair only with Re values of their own shape family.

        The real B4-fam cache gives every row a distinct Re, so a
        (blunt shape, slender Re) candidate would be unlabelable.
        """
        existing = self._existing()
        queries = [
            FlaggedQuery(
                params=_family_params({"l_over_d_mult": 0.75}),
                re=re,
                verdict="review",
                score=1.5,
                member_std=0.05,
            )
            for re in (100.0, 125.0, 150.0, 175.0)
        ] + [
            FlaggedQuery(
                params=_family_params({"l_over_d_mult": 1.30}),
                re=re,
                verdict="review",
                score=1.5,
                member_std=0.05,
            )
            for re in (400.0, 500.0, 600.0, 700.0)
        ]
        pts = propose_acquisition(
            queries, strategy="coverage", budget=8, existing_cond=existing, grid=TEST_GRID
        )
        assert 5 <= len(pts) <= 8  # pool = 2 corners x 4 own-family Re levels
        blunt = [p for p in pts if abs(p.params["l_over_d_mult"] - 0.75) < 1e-9]
        slender = [p for p in pts if abs(p.params["l_over_d_mult"] - 1.30) < 1e-9]
        assert blunt and slender
        for p in blunt:
            assert p.re in (100.0, 125.0, 150.0, 175.0), p.re
        for p in slender:
            assert p.re in (400.0, 500.0, 600.0, 700.0), p.re

    def test_envelope_shell_inside_design_box(self) -> None:
        existing = self._existing()
        queries = flagged_queries(existing)
        pts = propose_acquisition(
            queries,
            strategy="envelope_shell",
            budget=6,
            existing_cond=existing,
            grid=TEST_GRID,
            seed=1,
            n_candidates=96,
        )
        assert len(pts) == 6
        for p in pts:
            assert 0.75 <= p.params["l_over_d_mult"] <= 1.30
            assert 1.0 <= p.params["nose_len_mult"] <= 1.30
            assert min(q.re for q in queries) <= p.re <= max(q.re for q in queries)

    def test_max_disagreement_picks_high_std(self) -> None:
        queries = flagged_queries(self._existing())

        def std_fn(pts: list[AcquisitionPoint]) -> np.ndarray:
            return np.array(
                [1.0 if abs(p.params["l_over_d_mult"] - 0.75) < 1e-9 else 0.0 for p in pts]
            )

        pts = propose_acquisition(
            queries,
            strategy="max_disagreement",
            budget=4,
            existing_cond=self._existing(),
            grid=TEST_GRID,
            member_std_fn=std_fn,
        )
        assert len(pts) == 4
        assert all(abs(p.params["l_over_d_mult"] - 0.75) < 1e-9 for p in pts)

    def test_rejects_bad_args(self) -> None:
        queries = flagged_queries(self._existing())
        existing = self._existing()
        with pytest.raises(ValueError, match="strategy"):
            propose_acquisition(
                queries, strategy="magic", budget=2, existing_cond=existing, grid=TEST_GRID
            )
        with pytest.raises(ValueError, match="budget"):
            propose_acquisition(
                queries, strategy="coverage", budget=0, existing_cond=existing, grid=TEST_GRID
            )
        with pytest.raises(ValueError, match="member_std_fn"):
            propose_acquisition(
                queries,
                strategy="max_disagreement",
                budget=2,
                existing_cond=existing,
                grid=TEST_GRID,
            )
        with pytest.raises(ValueError, match="flagged queries"):
            propose_acquisition(
                queries[:1],
                strategy="coverage",
                budget=2,
                existing_cond=existing,
                grid=TEST_GRID,
            )


class TestFreshReAcquisition:
    """Fresh-Re mode + exact-key dedup (G1, 2026-08-27 real-label campaign).

    The cached-Re mode can only propose flagged (cached) Re values and the
    Mahalanobis floor over-rejects fresh-Re corner points — 10 of 24
    exact-key-proven-new campaign points scored below it.  Fresh-Re mode
    draws Re from a grid crossed with the same geometry arms, keeps the
    floor advisory (score/floor-pass recorded per point), and
    ``labeled_keys`` excludes already-labeled keys in BOTH modes.
    """

    FRESH_GRID = [137.0, 211.0, 389.0]  # neither flagged nor cached Re values

    def _existing(self) -> np.ndarray:
        return np.asarray(corpus_cond_v3(make_corpus()))

    def test_fresh_re_levels_come_from_grid_not_cache(self) -> None:
        existing = self._existing()
        queries = flagged_queries(existing)
        pts = propose_acquisition(
            queries,
            strategy="coverage",
            budget=6,
            existing_cond=existing,
            grid=TEST_GRID,
            fresh_re=True,
            fresh_re_grid=self.FRESH_GRID,
        )
        assert len(pts) == 6
        assert len({p.key for p in pts}) == len(pts)  # unique keys
        for p in pts:
            assert p.re in self.FRESH_GRID, p.re
            assert p.re not in {q.re for q in queries}  # not the flagged Re set
            assert p.re not in set(RES)  # not a cached corpus Re either
            assert p.guard_score is not None and p.floor_pass is not None

    def test_fresh_re_default_grid_spans_corpus_window(self) -> None:
        corpus = make_corpus()  # cached Re = RES[:5] = 80 .. 450
        existing = np.asarray(corpus_cond_v3(corpus))
        queries = flagged_queries(existing)
        # .9g discipline: the log round-trip recovers the corpus Re bitwise
        assert default_fresh_re_grid(existing, 8) == [
            float(v) for v in np.geomspace(float(corpus["re"].min()), float(corpus["re"].max()), 8)
        ]
        pts = propose_acquisition(
            queries,
            strategy="coverage",
            budget=8,
            existing_cond=existing,
            grid=TEST_GRID,
            fresh_re=True,
        )
        assert pts
        expected = set(default_fresh_re_grid(existing, 8))
        for p in pts:
            assert p.re in expected
        assert any(p.re not in {q.re for q in queries} for p in pts)  # genuinely fresh

    def test_fresh_re_all_strategies_on_grid(self) -> None:
        existing = self._existing()
        queries = flagged_queries(existing)
        shell = propose_acquisition(
            queries,
            strategy="envelope_shell",
            budget=4,
            existing_cond=existing,
            grid=TEST_GRID,
            seed=3,
            n_candidates=96,
            fresh_re=True,
            fresh_re_grid=self.FRESH_GRID,
        )
        assert shell and all(p.re in self.FRESH_GRID for p in shell)

        def std_fn(pts: list[AcquisitionPoint]) -> np.ndarray:
            return np.array([p.re for p in pts])  # favour the highest grid Re

        md = propose_acquisition(
            queries,
            strategy="max_disagreement",
            budget=4,
            existing_cond=existing,
            grid=TEST_GRID,
            member_std_fn=std_fn,
            fresh_re=True,
            fresh_re_grid=self.FRESH_GRID,
        )
        assert len(md) == 4
        assert [p.re for p in md] == [389.0] * 4  # top-std picks all land on 389
        assert len({p.key for p in md}) == 4  # ...at distinct geometry arms

    def test_exact_key_dedup_in_both_modes(self) -> None:
        existing = self._existing()
        queries = flagged_queries(existing)
        common: dict[str, Any] = dict(
            strategy="coverage", budget=5, existing_cond=existing, grid=TEST_GRID
        )
        first = propose_acquisition(queries, **common)
        assert len(first) == 5
        blocked = [p.key for p in first]
        # old mode: dedup is the safety property (the floor already shrank
        # the cached-Re pool, so a full-budget REFILL is not guaranteed —
        # only that no blocked key is ever re-proposed)
        again = propose_acquisition(queries, labeled_keys=blocked, **common)
        assert not ({p.key for p in again} & {p.key for p in first})
        # fresh mode: floor is advisory, so the pool = corners x grid and
        # the refill is exactly budget again, all disjoint from the blocked
        fresh = propose_acquisition(
            queries,
            labeled_keys=blocked,
            fresh_re=True,
            fresh_re_grid=self.FRESH_GRID,
            **common,
        )
        assert len(fresh) == 5
        assert not ({p.key for p in fresh} & {p.key for p in first})

    def test_corpus_point_keys_exclude_corpus_duplicates(self) -> None:
        corpus = make_corpus()
        existing = np.asarray(corpus_cond_v3(corpus))
        queries = flagged_queries(existing)  # flagged Re values ARE corpus Re values
        keys = corpus_point_keys(corpus)
        assert len(keys) == len(corpus["cd"])
        i = 5  # first with_sail(1,1) row of the default 6x5 corpus
        assert corpus["re"][i] == RES[0]
        assert keys[i] == point_param_key(
            {"hull_type": "with_sail", "sail_scale": 1.0, "fin_scale": 1.0},
            float(corpus["re"][i]),
        )
        corpus_keys = set(keys)

        def std_fn(pts: list[AcquisitionPoint]) -> np.ndarray:
            # favour the all-mother combo — exactly the corpus duplicates
            return np.array(
                [
                    1.0 if all(abs(p.params[a] - 1.0) < 1e-9 for a in HULLFORM_AXES) else 0.0
                    for p in pts
                ]
            )

        common: dict[str, Any] = dict(
            strategy="max_disagreement",
            budget=6,
            existing_cond=existing,
            grid=TEST_GRID,
            member_std_fn=std_fn,
        )
        raw = propose_acquisition(queries, **common)
        dedup = propose_acquisition(queries, labeled_keys=keys, **common)
        assert not ({p.key for p in dedup} & corpus_keys)  # safety property
        removed = {p.key for p in raw} - {p.key for p in dedup}
        assert removed <= corpus_keys  # ...and only corpus keys were dropped

    def test_floor_advisory_in_fresh_mode_hard_in_old_mode(self) -> None:
        queries = flagged_queries(self._existing())
        # dense tight cloud exactly AT the blunt shape / re=200 candidate
        blunt = _family_params({"l_over_d_mult": 0.75})
        cloud_row = hullform_condition_rows(blunt, [200.0], grid=TEST_GRID)[0]
        rng = np.random.default_rng(0)
        cloud = cloud_row[None, :] + rng.normal(0.0, 1e-6, (30, 8))
        guard = EnvelopeMahalanobisGuardrail(cloud)
        floor = _exclusion_floor(cloud, guard)

        old_pts = propose_acquisition(
            queries, strategy="coverage", budget=6, existing_cond=cloud, grid=TEST_GRID
        )
        assert old_pts
        assert not any(
            abs(p.params["l_over_d_mult"] - 0.75) < 1e-9 and abs(p.re - 200.0) < 1e-9
            for p in old_pts
        ), "old mode: the in-cloud candidate is hard-dropped"
        assert all(p.floor_pass for p in old_pts)  # survivors all cleared the floor

        fresh_pts = propose_acquisition(
            queries,
            strategy="coverage",
            budget=6,
            existing_cond=cloud,
            grid=TEST_GRID,
            fresh_re=True,
            fresh_re_grid=[150.0, 200.0, 300.0],
        )
        below = [
            p
            for p in fresh_pts
            if abs(p.params["l_over_d_mult"] - 0.75) < 1e-9 and abs(p.re - 200.0) < 1e-9
        ]
        assert below, "fresh mode must RETAIN the below-floor in-cloud candidate"
        for p in below:
            assert p.floor_pass is False
            assert p.guard_score is not None and p.guard_score < floor
        assert any(p.floor_pass for p in fresh_pts)  # advisory, not inverted

    def test_default_path_bitwise_unchanged(self) -> None:
        existing = self._existing()
        queries = flagged_queries(existing)
        common: dict[str, Any] = dict(
            strategy="coverage", budget=5, existing_cond=existing, grid=TEST_GRID
        )
        implicit = propose_acquisition(queries, **common)
        explicit = propose_acquisition(
            queries,
            fresh_re=False,
            fresh_re_grid=None,
            labeled_keys=None,
            **common,
        )
        assert [p.key for p in implicit] == [p.key for p in explicit]
        assert [p.guard_score for p in implicit] == [p.guard_score for p in explicit]
        assert [p.floor_pass for p in implicit] == [p.floor_pass for p in explicit]
        assert implicit == explicit  # dataclass equality: defaults are inert

    def test_fresh_re_arg_validation(self) -> None:
        queries = flagged_queries(self._existing())
        existing = self._existing()
        with pytest.raises(ValueError, match="fresh_re_grid requires fresh_re"):
            propose_acquisition(
                queries,
                strategy="coverage",
                budget=2,
                existing_cond=existing,
                grid=TEST_GRID,
                fresh_re_grid=[100.0],
            )
        with pytest.raises(ValueError, match="finite and positive"):
            propose_acquisition(
                queries,
                strategy="coverage",
                budget=2,
                existing_cond=existing,
                grid=TEST_GRID,
                fresh_re=True,
                fresh_re_grid=[100.0, -1.0],
            )
        # a cloud at a single Re cannot define a window: ask for a grid
        row = hullform_condition_rows(_family_params({}), [200.0], grid=TEST_GRID)[0]
        noise = np.random.default_rng(0).normal(0.0, 1e-6, (30, 8))
        noise[:, 0] = 0.0  # identical log-Re column -> degenerate window
        one_re_cloud = row[None, :] + noise
        with pytest.raises(ValueError, match="degenerate"):
            propose_acquisition(
                queries,
                strategy="coverage",
                budget=2,
                existing_cond=one_re_cloud,
                grid=TEST_GRID,
                fresh_re=True,
            )


# ---------------------------------------------------------------------------
# Oracle labels + corpus augmentation
# ---------------------------------------------------------------------------


class TestLabelsAndAugment:
    def _fam_points(self) -> list[AcquisitionPoint]:
        return [
            AcquisitionPoint(params=_family_params(shape), re=re, strategy="coverage")
            for shape in ({"l_over_d_mult": 0.75}, {"l_over_d_mult": 1.30})
            for re in (110.0, 330.0)
        ]

    def test_labels_exact_and_unmatched(self, tmp_path: Path) -> None:
        cache = write_fam_cache(tmp_path)
        pts = self._fam_points()
        pts.append(
            AcquisitionPoint(
                params=_family_params({"l_over_d_mult": 0.9}),  # interior shape: no label
                re=110.0,
                strategy="envelope_shell",
            )
        )
        labels = labels_from_cache(pts, cache)
        assert [lab.matched for lab in labels] == [True, True, True, True, False]
        z = np.load(cache)
        for lab in labels[:4]:
            assert lab.cd == z["cd"][lab.source_row]
            assert lab.re_delta == 0.0
            assert lab.payload["cd"] == lab.cd
            assert lab.payload["x"].shape == (5, TEST_GRID.ny, TEST_GRID.nx)
            assert lab.source_fam in ("fam_blunt", "fam_slender")
        assert np.isnan(labels[-1].cd)
        assert labels[-1].source_row is None

    def test_labels_re_tolerance(self, tmp_path: Path) -> None:
        cache = write_fam_cache(tmp_path)
        pt = AcquisitionPoint(
            params=_family_params({"l_over_d_mult": 0.75}), re=110.0 * 1.02, strategy="coverage"
        )
        assert not labels_from_cache([pt], cache)[0].matched  # exact: no match
        lab = labels_from_cache([pt], cache, re_tol=0.03)[0]
        assert lab.matched and 0.0 < lab.re_delta <= 0.03

    def test_labels_missing_cache_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            labels_from_cache(self._fam_points()[:1], tmp_path / "nope.npz")

    def test_augment_schema_and_remap(self, tmp_path: Path) -> None:
        cache = write_fam_cache(tmp_path)
        corpus = make_corpus()
        fp = self._fam_points()
        pts = [fp[0], fp[2]]  # one blunt (fam dsi 6) @110, one slender (7) @110
        labels = labels_from_cache(pts, cache)
        assert all(lab.matched for lab in labels)
        out = augment_corpus(corpus, pts, labels, grid=TEST_GRID)
        assert set(out) == set(corpus), "schema identical to the base index"
        n0 = len(corpus["cd"])
        assert len(out["cd"]) == n0 + 2
        assert int(out["dsi"][-2]) == int(corpus["dsi"].max()) + 1  # fam 6 -> max+1
        assert int(out["dsi"][-1]) == int(corpus["dsi"].max()) + 2  # fam 7 -> max+2
        for j, lab in enumerate(labels):
            i = n0 + j
            assert out["cd"][i] == lab.cd
            assert np.array_equal(out["x"][i], lab.payload["x"])
            assert np.array_equal(out["aux"][i], lab.payload["aux"])
            assert out["mask_bit_eq"].dtype == corpus["mask_bit_eq"].dtype
            cfg = SuboffConfig(l_over_d_mult=pts[j].params["l_over_d_mult"])
            assert np.array_equal(
                out["geo"][i], hullform_geo_block("with_sail", 1.0, 1.0, TEST_GRID, cfg)
            )
            counts = hullform_component_counts("with_sail", 1.0, 1.0, TEST_GRID, cfg)
            v_bare, v_sail, v_fin, v_solid, aproj, aproj_bare = counts
            assert int(out["v_sail"][i]) == v_sail
            assert int(out["v_fin"][i]) == v_fin
            assert int(out["v_solid"][i]) == v_solid == v_bare + v_sail + v_fin
            assert int(out["aproj_cad"][i]) == aproj
            assert int(out["aproj_bare"][i]) == aproj_bare

    def test_augment_mother_geo_bitwise(self, tmp_path: Path) -> None:
        """Acquired mother-shape rows reproduce the cached geo convention."""
        cache = write_fam_cache(tmp_path)
        meta_path = cache.with_name("cache_fam_meta.json")
        meta = json.loads(meta_path.read_text())
        meta.append(
            {
                "hull": "with_sail",
                "sail": 1.0,
                "fin": 1.0,
                "u_in": 0.1,
                "fam": "fam_mother",
                "re": 250.0,
                **{a: 1.0 for a in HULLFORM_AXES},
            }
        )
        z = dict(np.load(cache))
        for k, v in z.items():
            pad = np.zeros((1,) + v.shape[1:], dtype=v.dtype)
            z[k] = np.concatenate([v, pad], axis=0)
        z["dsi"][-1] = 8
        z["fam"][-1] = 8
        z["re"][-1] = 250.0
        z["cd"][-1] = 2.0
        np.savez(cache, **z)
        meta_path.write_text(json.dumps(meta))
        pt = AcquisitionPoint(
            params=_family_params({}),  # all axes 1.0 = mother shape
            re=250.0,
            strategy="coverage",
        )
        labels = labels_from_cache([pt], cache)
        assert labels[0].matched
        out = augment_corpus(make_corpus(), [pt], labels, grid=TEST_GRID)
        assert np.array_equal(out["geo"][-1], _design_geo("with_sail", 1.0, 1.0))

    def test_augment_rejects_unmatched(self, tmp_path: Path) -> None:
        cache = write_fam_cache(tmp_path)
        corpus = make_corpus()
        pt = AcquisitionPoint(
            params=_family_params({"l_over_d_mult": 0.9}), re=110.0, strategy="envelope_shell"
        )
        labels = labels_from_cache([pt], cache)
        with pytest.raises(ValueError, match="unmatched"):
            augment_corpus(corpus, [pt], labels, grid=TEST_GRID)

    def test_axes_envelope(self) -> None:
        env = axes_envelope([{"l_over_d_mult": 1.0}, {"l_over_d_mult": 0.75}])
        assert env["l_over_d_mult"] == (0.75, 1.0)
        assert env["nose_len_mult"] == (1.0, 1.0)


# ---------------------------------------------------------------------------
# Honest verdict + trend statistics
# ---------------------------------------------------------------------------


class TestVerdictAndTrend:
    def test_honest_verdict_downgrade_and_unflag(self) -> None:
        ok = GuardVerdict(flag=FLAG_OK, score=1.2, reasons=("mahalanobis=1.2",))
        variant = {"l_over_d_mult": 1.3}
        mother = {"l_over_d_mult": 1.0}
        wide = {axis: (0.75, 1.3) for axis in HULLFORM_AXES}
        down = honest_verdict(ok, variant, MOTHER_ENV)
        assert down.flag == FLAG_REVIEW and down.score == 1.2
        assert "outside the served corpus envelope" in down.reasons[0]
        assert honest_verdict(ok, mother, MOTHER_ENV).flag == FLAG_OK
        assert honest_verdict(ok, variant, wide).flag == FLAG_OK  # the un-flag
        assert honest_verdict(GuardVerdict(FLAG_REJECT, 9.9), variant, MOTHER_ENV).flag == (
            FLAG_REJECT
        )
        assert honest_verdict(GuardVerdict(FLAG_REVIEW, 5.0), variant, MOTHER_ENV).flag == (
            FLAG_REVIEW
        )

    def test_trend_up_down_flat(self) -> None:
        vals = [0.75, 1.0, 1.3]
        up = np.array([[1.0, 2.0], [1.5, 2.5], [2.0, 3.0]])
        down = np.array([[2.0, 3.0], [1.5, 2.5], [1.0, 2.0]])
        flat = np.ones((3, 2))
        assert trend_stat(vals, up).sign == 1
        assert trend_stat(vals, up).mean_rho == 1.0
        assert trend_stat(vals, down).sign == -1
        assert trend_stat(vals, flat).sign == 0
        with pytest.raises(ValueError, match="cd_rows"):
            trend_stat(vals, np.ones(3))

    def test_spearman_ties(self) -> None:
        assert spearman_rho([1, 2, 3], [1, 2, 3]) == 1.0
        assert spearman_rho([1, 2, 3], [3, 2, 1]) == -1.0
        assert spearman_rho([1, 1, 1], [1, 2, 3]) == 0.0
        assert spearman_rho([1, 1, 2], [1, 1, 3]) == 1.0
        with pytest.raises(ValueError):
            spearman_rho([1.0], [1.0])


# ---------------------------------------------------------------------------
# Split discipline + retrain determinism
# ---------------------------------------------------------------------------


def _grouped_corpus(n_groups: int = 15, per_group: int = 2) -> dict[str, np.ndarray]:
    """Corpus with duplicate param rows so split groups are non-trivial."""
    rng = np.random.default_rng(3)
    n = n_groups * per_group
    re = np.tile(np.geomspace(80, 600, n_groups), per_group)
    return dict(
        x=rng.normal(0, 1, (n, 5, TEST_GRID.ny, TEST_GRID.nx)).astype(np.float32),
        dsi=np.repeat(np.arange(3, dtype=np.int64), per_group * 5)[:n],
        re=re,
        uin=np.full(n, 0.1),
        sail=np.full(n, 1.0),
        fin=np.full(n, 1.0),
        hull=np.full(n, 1, dtype=np.int64),
        step=np.full(n, 4000, dtype=np.int64),
        aproj=np.full(n, 69, dtype=np.int64),
        cd=20.0 * re**-0.42,
        geo=rng.normal(0.0, 0.01, (n, 4)),
        aux=rng.normal(0.5, 0.05, (n, 8)),
        mask_bit_eq=np.zeros(n, dtype=bool),
    )


class TestProtocol:
    def test_split_stable_under_augmentation(self) -> None:
        base = _grouped_corpus()
        n0 = len(base["cd"])
        sp0 = split_random(base)
        aug = dict(base)
        m = 8
        for k, v in base.items():
            pad = np.zeros((m,) + v.shape[1:], dtype=v.dtype)
            aug[k] = np.concatenate([v, pad], axis=0)
        aug["dsi"] = np.concatenate([base["dsi"], np.full(m, 99, np.int64)])
        aug["re"] = np.concatenate([base["re"], np.tile(np.geomspace(100, 500, 4), 2)])
        aug["cd"] = np.concatenate([base["cd"], np.full(m, 2.0)])
        sp1 = split_random(aug)
        for part in ("fit", "val", "test", "train"):
            kept = [i for i in sp1[part] if i < n0]
            assert kept == sp0[part], part  # original rows unchanged
        assert any(i >= n0 for i in sp1["test"])  # new dataset carved too

    def test_retrain_deterministic_cpu(self, tmp_path: Path) -> None:
        corpus = _grouped_corpus(n_groups=10, per_group=2)
        a = retrain_ensemble(
            corpus,
            tmp_path / "a",
            seeds=(0,),
            device="cpu",
            hp_overrides=dict(epochs=3, patience=3, batch=8),
        )
        b = retrain_ensemble(
            corpus,
            tmp_path / "b",
            seeds=(0,),
            device="cpu",
            hp_overrides=dict(epochs=3, patience=3, batch=8),
        )
        assert a[0]["mape"] == b[0]["mape"]  # identical metric row
        assert a[0]["best_epoch"] == b[0]["best_epoch"]
        ta = torch.load(tmp_path / "a" / "al_aug_s0.pt", weights_only=False)
        tb = torch.load(tmp_path / "b" / "al_aug_s0.pt", weights_only=False)
        for k in ta["state_dict"]:
            assert torch.equal(ta["state_dict"][k], tb["state_dict"][k]), k
        assert ARM_CFG["sampling"] == "quota"  # protocol untouched by overrides


# ---------------------------------------------------------------------------
# eval_loop end-to-end on fake ensembles
# ---------------------------------------------------------------------------


class TestEvalLoop:
    def _eval_points(self) -> list[AcquisitionPoint]:
        return [
            AcquisitionPoint(params=_family_params(shape), re=re, strategy="coverage")
            for shape in ({"l_over_d_mult": 0.75}, {"l_over_d_mult": 1.30})
            for re in (110.0, 330.0)
        ]

    def test_report_end_to_end(self, tmp_path: Path) -> None:
        corpus = make_corpus()
        cache = write_fam_cache(tmp_path)
        before_dir = write_fake_ckpt_dir(tmp_path, corpus, tag="before", seed=3)
        after_dir = write_fake_ckpt_dir(tmp_path, corpus, tag="after", seed=3)  # same members
        cond = corpus_cond_v3(corpus)
        designs = corpus_design_keys(corpus)

        eval_pts = self._eval_points()
        labels = labels_from_cache(eval_pts, cache)
        labels_cd = {lab.point_key: lab.cd for lab in labels if lab.matched}
        assert len(labels_cd) == 4

        # the after-corpus contains the family rows -> guard fit + envelope widen
        fam_cond = np.stack(
            [hullform_condition_rows(p.params, [p.re], grid=TEST_GRID)[0] for p in eval_pts]
        )
        trend = TrendSpec(
            axis="l_over_d_mult",
            values=(0.75, 1.0, 1.3),
            re_grid=(150.0, 400.0),
            base_params={"hull_type": "with_sail", "sail_scale": 1.0, "fin_scale": 1.0},
            truth_cd=np.array([[8.0, 4.0], [11.0, 5.5], [16.0, 7.0]]),
        )
        report = eval_loop(
            before_dir,
            after_dir,
            eval_pts,
            guard_features_before=cond,
            guard_features_after=np.vstack([cond, fam_cond]),
            axes_env_before=MOTHER_ENV,
            axes_env_after={axis: (0.75, 1.3) for axis in HULLFORM_AXES},
            corpus_cache=corpus["x"],
            cache_re=corpus["re"],
            cache_designs=designs,
            trend=trend,
            mother_eval=MotherEval(x=corpus["x"][:4], cond=cond[:4], cd=corpus["cd"][:4]),
            device="cpu",
            grid=TEST_GRID,
            labels_cd=labels_cd,
        )
        assert report.n_eval_points == 4
        assert report.family_mape_before >= 0.0 and report.family_mape_after >= 0.0
        assert report.trend["truth"]["sign"] == 1
        assert "flipped_to_agree" in report.trend
        assert set(report.verdicts_before) <= {"review", "reject", "ok"}
        # before: outside the mother envelope -> review; after: un-flagged
        assert (
            sum(1 for f in report.verdict_flips if f["before"] == "review" and f["after"] == "ok")
            >= 1
        )
        assert report.mother_mape_before is not None
        assert report.mother_mape_after is not None
        for row in report.per_point:
            assert np.isfinite(row["cd_before"]) and np.isfinite(row["cd_after"])
            assert row["chan_flag_before"] in ("ok", "review", "reject")
        # JSON round trip + disk write
        assert len(json.dumps(report.as_dict())) > 0
        out = write_loop_report(report, tmp_path / "loop_report.json")
        assert Path(out).is_file()
        disk = json.loads(Path(out).read_text())
        assert disk["n_eval_points"] == 4
        assert disk["trend"]["truth"]["sign"] == 1

    def test_eval_loop_requires_labels(self, tmp_path: Path) -> None:
        corpus = make_corpus()
        before = write_fake_ckpt_dir(tmp_path, corpus, tag="b2", seed=3)
        with pytest.raises(ValueError, match="labels_cd"):
            eval_loop(
                before,
                before,
                self._eval_points()[:2],
                guard_features_before=corpus_cond_v3(corpus),
                guard_features_after=corpus_cond_v3(corpus),
                axes_env_before=MOTHER_ENV,
                axes_env_after=MOTHER_ENV,
                corpus_cache=corpus["x"],
                cache_re=corpus["re"],
                cache_designs=corpus_design_keys(corpus),
                device="cpu",
                grid=TEST_GRID,
            )

    def test_service_spec_backend_and_nearest_field(self, tmp_path: Path) -> None:
        corpus = make_corpus()
        ck = write_fake_ckpt_dir(tmp_path, corpus, tag="spec", seed=3)
        spec = ServiceSpec(
            ckpt_dir=ck,
            guard_features=corpus_cond_v3(corpus),
            axes_env=MOTHER_ENV,
            corpus_cache=corpus["x"],
            cache_re=corpus["re"],
            cache_designs=corpus_design_keys(corpus),
        )
        backend = spec.backend()
        assert backend.n_members == 2
        field, row = spec.nearest_field("with_sail", 1.0, 1.0, 250.0, 0.1)
        assert field.shape == (5, TEST_GRID.ny, TEST_GRID.nx)
        assert field.dtype == np.float32
        assert corpus["re"][row] == 300.0  # nearest log-Re of {120, 300} to 250
        with pytest.raises(ValueError, match="not in the attached field cache"):
            spec.nearest_field("bare_hull", 9.9, 1.0, 250.0, 0.1)
        bare = ServiceSpec(ckpt_dir=ck, guard_features=corpus_cond_v3(corpus), axes_env=MOTHER_ENV)
        with pytest.raises(ValueError, match="no corpus field cache"):
            bare.nearest_field("with_sail", 1.0, 1.0, 250.0, 0.1)

    def test_predict_design_shapes(self, tmp_path: Path) -> None:
        corpus = make_corpus()
        ck = write_fake_ckpt_dir(tmp_path, corpus, tag="pred", seed=3)
        spec = ServiceSpec(
            ckpt_dir=ck,
            guard_features=corpus_cond_v3(corpus),
            axes_env=MOTHER_ENV,
            corpus_cache=corpus["x"],
            cache_re=corpus["re"],
            cache_designs=corpus_design_keys(corpus),
        )
        backend = spec.backend()
        mat, mean, std = predict_design(
            backend, spec, _family_params({"l_over_d_mult": 1.3}), RES[:3], grid=TEST_GRID
        )
        assert mat.shape == (2, 3)
        assert mean.shape == (3,) and std.shape == (3,)
        assert np.isfinite(mat).all()

    def test_point_and_corpus_keys(self) -> None:
        corpus = make_corpus(n_per_design=2)
        keys = corpus_param_keys(corpus)
        i = 2 * 1  # first with_sail(1,1) row
        expected = (
            f"{corpus['re'][i]:.6g}|{corpus['uin'][i]:.6g}|{corpus['sail'][i]:.6g}"
            f"|{corpus['fin'][i]:.6g}|with_sail"
        )
        assert keys[i] == expected
        p = AcquisitionPoint(
            params={"hull_type": "with_sail", "l_over_d_mult": 1.3}, re=123.456, strategy="x"
        )
        assert "with_sail" in p.key and "1.3" in p.key and "123.456" in p.key


# ---------------------------------------------------------------------------
# B4 v5 — sail axial-position conditioning (2026-08-27 sail_x campaign fix)
# ---------------------------------------------------------------------------


def _sailx_label(point: AcquisitionPoint, cd: float, fam: int = 9) -> AcquisitionLabel:
    """Hand-built matched label of a sail_x point (payload = cache schema)."""
    rng = np.random.default_rng(42)
    return AcquisitionLabel(
        point_key=point.key,
        matched=True,
        cd=cd,
        payload=dict(
            x=rng.normal(0.5, 0.1, (5, TEST_GRID.ny, TEST_GRID.nx)).astype(np.float32),
            re=point.re,
            uin=0.1,
            sail=1.0,
            fin=1.0,
            hull=1,
            step=4000,
            aproj=69,
            cd=cd,
            aux=rng.normal(0.5, 0.05, 8),
            mask_bit_eq=False,
            fam=fam,
        ),
    )


class TestConditionV5Path:
    """The v5 axis fix: channel exists, prefix is v3, serving varies with it."""

    def test_rows_prefix_and_channel(self) -> None:
        params = _family_params({"sail_x_mult": 0.7})
        v3 = hullform_condition_rows(params, RES[:3], grid=TEST_GRID)
        v5 = hullform_condition_rows_v5(params, RES[:3], grid=TEST_GRID)
        assert v5.shape == (3, 9)
        np.testing.assert_array_equal(v5[:, :8], v3)  # bit-identical prefix
        np.testing.assert_allclose(v5[:, 8], np.log10(0.7))

    def test_default_params_encode_mother_zero(self) -> None:
        rows = hullform_condition_rows_v5(_family_params({}), [150.0], grid=TEST_GRID)
        assert rows[0, 8] == 0.0

    def test_geo_block_invariant_but_rows_v5_vary(self) -> None:
        """P1 pinned: the geo block cannot see the axis, the v5 rows can."""
        geo = {
            m: hullform_geo_block("with_sail", 1.0, 1.0, TEST_GRID, SuboffConfig(sail_x_mult=m))
            for m in (0.7, 1.0, 1.4)
        }
        for m in (0.7, 1.0, 1.4):  # mask-derived block strictly invariant (root cause)
            np.testing.assert_array_equal(geo[m], geo[1.0])
        rows = [
            hullform_condition_rows_v5(_family_params({"sail_x_mult": m}), [150.0], grid=TEST_GRID)[
                0
            ]
            for m in (0.7, 1.0, 1.4)
        ]
        np.testing.assert_array_equal(
            np.stack(rows)[:, :8], np.stack([rows[1][:8]] * 3)
        )  # v3 part flat
        assert len({float(r[8]) for r in rows}) == 3  # the channel separates them

    def test_corpus_cond_v5_mother_and_column(self) -> None:
        corpus = make_corpus(n_per_design=2)
        v5 = corpus_cond_v5(corpus)
        assert v5.shape == (len(corpus["cd"]), 9)
        np.testing.assert_array_equal(v5[:, :8], corpus_cond_v3(corpus))
        np.testing.assert_array_equal(v5[:, 8], np.zeros(len(corpus["cd"])))
        n = len(corpus["cd"])
        with_col = dict(corpus)
        mults = np.ones(n)
        mults[n // 2 :] = 0.7
        with_col["sail_x_mult"] = mults
        out = corpus_cond_v5(with_col)
        np.testing.assert_allclose(out[:, 8], np.log10(with_col["sail_x_mult"]))

    def test_augment_carries_sail_x_mult(self, tmp_path: Path) -> None:
        corpus = make_corpus(n_per_design=2)
        corpus["sail_x_mult"] = np.ones(len(corpus["cd"]))
        pt = AcquisitionPoint(
            params=_family_params({"sail_x_mult": 0.7}), re=150.0, strategy="coverage"
        )
        out = augment_corpus(corpus, [pt], [_sailx_label(pt, cd=3.0)], grid=TEST_GRID)
        assert out["sail_x_mult"][-1] == 0.7
        assert np.all(out["sail_x_mult"][:-1] == 1.0)
        np.testing.assert_array_equal(  # geo row is the invariant mother block
            out["geo"][-1], hullform_geo_block("with_sail", 1.0, 1.0, TEST_GRID)
        )

    def test_retrain_and_serve_v5_varies_with_sail_x(self, tmp_path: Path) -> None:
        """End-to-end v5 path: retrain(cond='v5') -> serve via predict_design."""
        corpus = make_corpus(n_per_design=2)
        n0 = len(corpus["cd"])
        corpus["sail_x_mult"] = np.ones(n0)
        rng = np.random.default_rng(5)
        for m in (0.7, 1.4):  # two sailx rows, own dataset id -> quota-sampled
            for k, v in corpus.items():
                pad = np.zeros((1,) + np.asarray(v).shape[1:], dtype=np.asarray(v).dtype)
                corpus[k] = np.concatenate([np.asarray(v), pad], axis=0)
            corpus["dsi"][-1] = 99
            corpus["re"][-1] = 150.0
            corpus["sail"][-1] = 1.0
            corpus["fin"][-1] = 1.0
            corpus["hull"][-1] = 1
            corpus["cd"][-1] = 18.0 * 150.0**-0.42
            corpus["aux"][-1] = rng.normal(0.5, 0.05, 8)
            corpus["sail_x_mult"][-1] = m
            corpus["geo"][-1] = _design_geo("with_sail", 1.0, 1.0)
            corpus["x"][-1] = rng.normal(0.5, 0.1, (5, TEST_GRID.ny, TEST_GRID.nx)).astype(
                np.float32
            )
        ck = tmp_path / "ckpts_v5"
        retrain_ensemble(
            corpus,
            ck,
            seeds=(0,),
            device="cpu",
            hp_overrides=dict(epochs=2, patience=2, batch=8),
            cond="v5",
        )
        spec = ServiceSpec(
            ckpt_dir=ck,
            guard_features=corpus_cond_v5(corpus),
            axes_env={axis: (0.7 if axis == "sail_x_mult" else 1.0, 1.4) for axis in HULLFORM_AXES},
            corpus_cache=corpus["x"],
            cache_re=corpus["re"],
            cache_designs=corpus_design_keys(corpus),
        )
        backend = spec.backend()
        assert backend.cond_dim == 9
        _mat, mean_lo, _std = predict_design(
            backend,
            spec,
            _family_params({"sail_x_mult": 0.7}),
            RES[:3],
            grid=TEST_GRID,
            cond_version="v5",
        )
        _mat, mean_hi, _std = predict_design(
            backend,
            spec,
            _family_params({"sail_x_mult": 1.4}),
            RES[:3],
            grid=TEST_GRID,
            cond_version="v5",
        )
        assert not np.array_equal(mean_lo, mean_hi)  # axis reaches the prediction
        with pytest.raises(ValueError, match=r"cond must be \(N, 9\)"):
            predict_design(  # v3 rows against a v5 ensemble: width guard fires
                backend, spec, _family_params({}), RES[:3], grid=TEST_GRID
            )
        with pytest.raises(ValueError, match="cond_version"):
            predict_design(
                backend,
                spec,
                _family_params({}),
                RES[:3],
                grid=TEST_GRID,
                cond_version="v2",
            )

    def test_retrain_v5_deterministic_cpu(self, tmp_path: Path) -> None:
        corpus = make_corpus(n_per_design=2)
        corpus["sail_x_mult"] = np.ones(len(corpus["cd"]))
        a = retrain_ensemble(
            corpus,
            tmp_path / "a",
            seeds=(0,),
            device="cpu",
            hp_overrides=dict(epochs=2, patience=2, batch=8),
            cond="v5",
        )
        b = retrain_ensemble(
            corpus,
            tmp_path / "b",
            seeds=(0,),
            device="cpu",
            hp_overrides=dict(epochs=2, patience=2, batch=8),
            cond="v5",
        )
        assert a[0]["mape"] == b[0]["mape"]
        ta = torch.load(tmp_path / "a" / "al_aug_s0.pt", weights_only=False)
        assert ta["meta"]["cond"] == "v5"
        assert ta["arch"]["cond_dim"] == 9


# ---------------------------------------------------------------------------
# B4-fam corpus fragment (v5 fam arms, 2026-08-28)
# ---------------------------------------------------------------------------


def _write_fam_cache_prod(tmp_path: Path) -> Path:
    """Synthetic cache_fam in the PRODUCTION schema (``geom``, no ``geo``).

    Unlike :func:`write_fam_cache` (which predates the fragment path and
    swaps the 6/7 labels), this uses the production family semantics
    6=slender / 7=blunt / 8=long_nose / 9=aft_sail, the generalised
    ``geom`` block instead of ``geo``, and a base block of ``fam = -1``
    rows so the meta (family block only) is genuinely shorter than the
    cache — the alignment contract :func:`load_fam_fragment` pins.
    """
    rng = np.random.default_rng(13)
    fams = [
        ("fam_slender", 6, {"l_over_d_mult": 1.30}),
        ("fam_blunt", 7, {"l_over_d_mult": 0.75}),
        ("fam_long_nose", 8, {"nose_len_mult": 1.30}),
        ("fam_aft_sail", 9, {"sail_x_mult": 1.30}),
    ]
    res = [110.0, 210.0, 330.0, 520.0]
    meta: list[dict[str, object]] = []
    keys = (
        "x",
        "dsi",
        "re",
        "uin",
        "sail",
        "fin",
        "hull",
        "step",
        "aproj",
        "cd",
        "geom",
        "aux",
        "mask_bit_eq",
        "fam",
    )
    cols: dict[str, list[Any]] = {k: [] for k in keys}

    def append_row(re: float, fam_label: int) -> None:
        cols["x"].append(rng.normal(0.5, 0.1, (5, TEST_GRID.ny, TEST_GRID.nx)).astype(np.float32))
        cols["dsi"].append(np.int64(max(fam_label, 0)))
        cols["re"].append(np.float64(re))
        cols["uin"].append(np.float64(0.1))
        cols["sail"].append(np.float64(1.0))
        cols["fin"].append(np.float64(1.0))
        cols["hull"].append(np.int64(1))
        cols["step"].append(np.int64(4000))
        cols["aproj"].append(np.int64(69))
        cols["cd"].append(np.float64(18.0 * re**-0.42))
        cols["geom"].append(rng.normal(0.0, 0.01, 4))
        cols["aux"].append(rng.normal(0.5, 0.05, 8))
        cols["mask_bit_eq"].append(True)
        cols["fam"].append(np.int64(fam_label))

    for re in res[:2]:  # base block (the v2 corpus rows the cache carries)
        append_row(re, -1)
    for fam_name, fam_dsi, axes in fams:
        for re in res:
            base = {a: 1.0 for a in HULLFORM_AXES}
            base.update(axes)
            meta.append(
                {
                    "hull": "with_sail",
                    "sail": 1.0,
                    "fin": 1.0,
                    "u_in": 0.1,
                    "fam": fam_name,
                    "re": re,
                    **base,
                }
            )
            append_row(re, fam_dsi)
    path = tmp_path / "cache_fam.npz"
    np.savez(path, **{k: np.stack(v) for k, v in cols.items()})  # type: ignore[arg-type]
    path.with_name("cache_fam_meta.json").write_text(json.dumps(meta))
    return path


class TestFamFragment:
    def test_aft_sail_geo_bitwise_mother_and_v5_channel(self, tmp_path: Path) -> None:
        """The aft_sail fragment: geo == mother bitwise, axis only in ch9."""
        cache = _write_fam_cache_prod(tmp_path)
        frag = load_fam_fragment(cache, fam_labels=(9,), grid=TEST_GRID)
        assert len(frag["cd"]) == 4
        mother = hullform_geo_block("with_sail", 1.0, 1.0, TEST_GRID)
        for g in frag["geo"]:  # pure sail translation -> block strictly invariant
            np.testing.assert_array_equal(g, mother)
        assert np.all(frag["sail_x_mult"] == 1.3)
        assert np.all(frag["l_over_d_mult"] == 1.0)
        assert np.all(frag["fam"] == 9)

        corpus = make_corpus(n_per_design=2)
        n0 = len(corpus["cd"])
        corpus["sail_x_mult"] = np.ones(n0)
        out = append_fam_fragment(corpus, frag)
        cond = corpus_cond_v5(out)
        np.testing.assert_array_equal(cond[:n0, 8], np.zeros(n0))  # mother -> 0
        np.testing.assert_allclose(cond[n0:, 8], np.log10(1.3))  # aft rows vary
        np.testing.assert_array_equal(corpus_cond_v3(out)[:n0], corpus_cond_v3(corpus))

    def test_fragment_cad_geo_and_counts_all_families(self, tmp_path: Path) -> None:
        cache = _write_fam_cache_prod(tmp_path)
        frag = load_fam_fragment(cache, grid=TEST_GRID)
        assert len(frag["cd"]) == 16
        mother = hullform_geo_block("with_sail", 1.0, 1.0, TEST_GRID)
        for j in range(len(frag["cd"])):
            axes = {a: float(frag[a][j]) for a in HULLFORM_AXES}
            cfg = SuboffConfig(**axes)
            np.testing.assert_array_equal(
                frag["geo"][j], hullform_geo_block("with_sail", 1.0, 1.0, TEST_GRID, cfg)
            )
            v_bare, v_sail, v_fin, v_solid, aproj, aproj_bare = hullform_component_counts(
                "with_sail", 1.0, 1.0, TEST_GRID, cfg
            )
            assert frag["v_sail"][j] == v_sail
            assert frag["v_solid"][j] == v_solid == v_bare + v_sail + v_fin
            assert frag["aproj_cad"][j] == aproj
            assert frag["aproj_bare"][j] == aproj_bare
        slender = frag["geo"][frag["fam"] == 6][0]
        assert not np.array_equal(slender, mother)  # block DOES see hull-form axes
        assert frag["cd"][0] == pytest.approx(18.0 * 110.0**-0.42)  # payload verbatim

    def test_append_remaps_dsi_and_keeps_base_split(self, tmp_path: Path) -> None:
        cache = _write_fam_cache_prod(tmp_path)
        frag = load_fam_fragment(cache, grid=TEST_GRID)
        corpus = make_corpus(n_per_design=2)
        corpus["sail_x_mult"] = np.ones(len(corpus["cd"]))
        out = append_fam_fragment(corpus, frag)
        n0 = len(corpus["cd"])
        assert len(out["cd"]) == n0 + 16
        assert set(out) == set(corpus), "schema identical to the base index (fam dropped)"
        new_dsi = sorted({int(d) for d in out["dsi"][n0:]})
        assert new_dsi == [int(corpus["dsi"].max()) + 1 + i for i in range(4)]
        s0 = split_random(corpus)
        s1 = split_random(out)
        for part in ("fit", "val", "test", "train"):
            assert [i for i in s1[part] if i < n0] == s0[part]

    def test_append_without_sailx_column_refuses_variant_rows(self, tmp_path: Path) -> None:
        cache = _write_fam_cache_prod(tmp_path)
        frag = load_fam_fragment(cache, fam_labels=(9,), grid=TEST_GRID)
        corpus = make_corpus(n_per_design=2)  # no sail_x_mult column
        with pytest.raises(ValueError, match="sail_x_mult"):
            append_fam_fragment(corpus, frag)
        hull_form = load_fam_fragment(cache, fam_labels=(6,), grid=TEST_GRID)
        out = append_fam_fragment(corpus, hull_form)  # all-mother axis: fine
        assert "sail_x_mult" not in out

    def test_meta_drift_and_alignment_raise(self, tmp_path: Path) -> None:
        cache = _write_fam_cache_prod(tmp_path)
        meta_path = cache.with_name("cache_fam_meta.json")
        meta = json.loads(meta_path.read_text())
        drifted = [dict(m) for m in meta]
        drifted[0]["l_over_d_mult"] = 1.10  # family cache drifted
        meta_path.write_text(json.dumps(drifted))
        with pytest.raises(ValueError, match="disagree with canonical"):
            load_fam_fragment(cache, fam_labels=(6,), grid=TEST_GRID)
        meta_path.write_text(json.dumps(meta[:-1]))  # meta no longer lists all rows
        with pytest.raises(ValueError, match="family rows but meta"):
            load_fam_fragment(cache, grid=TEST_GRID)
        mismatch = [dict(m) for m in meta]
        mismatch[0]["re"] = 111.0  # misaligned block
        meta_path.write_text(json.dumps(mismatch))
        with pytest.raises(ValueError, match="does not match cache row"):
            load_fam_fragment(cache, grid=TEST_GRID)
