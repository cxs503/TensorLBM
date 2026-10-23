"""Tests for the field-borrow serving wiring (``field_borrow`` + the hook).

Pins the 2026-09-04 L2 field-borrow path against tiny CPU fixtures:

- the policy resolver: default ``cache``, opt-in ``field_borrow``, typoed
  names raise (the ``re_policy`` convention);
- ``param_cond_rows`` reproduces the e2e ``load_fam`` cond assembly
  (log10 columns, ts2/ts4 slicing);
- ``borrow_serving_field`` surfaces the full provenance block and SERVES a
  failed in-manifold guard — flagged and warning-logged, never raised,
  never re-strategied;
- the ``DragSurrogateService`` hook: default OFF is byte-identical
  (served arrays and ``info``), ON borrows only when the cache cannot
  resolve the field, cached designs / caller fields keep their resolution,
  and the original error survives when the flag cannot help;
- two-stage (per-member) backends serve the param-cond composition with
  the query SDF as a model input;
- real-corpus smoke (skipif absent): a production ts2 ensemble serves one
  corpus design through the borrow path with a same-design donor —
  bit-identical to the manual cached-fields composition, and an honest
  leave-design-out donor stays within the e2e noise floor.

Evidence: ``/nfs/wangxi/runs/l2_e2e_validation_20260904`` (e2e LODO) and
``/nfs/wangxi/runs/l2_field_sensitivity_20260904`` (field-sensitivity).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from tensorlbm.ai.ckpt_bundle import PerMemberEnsembleBackend, load_member_bundle
from tensorlbm.ai.drag_cond import (
    CondFNODrag,
    SuboffGrid,
    geometry_channels,
    suboff_geometry_features,
)
from tensorlbm.ai.field_borrow import (
    CORPUS_COND_KEYS,
    FIELD_POLICY_BORROW,
    FIELD_POLICY_CACHE,
    borrow_conditioning,
    borrow_serving_field,
    param_cond_rows,
    resolve_field_policy,
)
from tensorlbm.ai.field_provider import FIELD_CHANNELS, FieldProvider
from tensorlbm.ai.geom_encoder import SDFEncoderV2
from tensorlbm.ai.inference_service import (
    BackendQueryError,
    CondDragCheckpoint,
    DragSurrogateService,
    ModelEnsembleBackend,
    NullGuardrail,
    ensemble_stats,
)
from tensorlbm.ai.sdf_two_stage import SupervisedSDFEncoder, TwoStageCondFNODrag

TEST_GRID = SuboffGrid.from_resolution(32)  # ny=nz=16, nx=32
NY, NX = TEST_GRID.ny, TEST_GRID.nx
TINY_V3 = dict(in_ch=5, width=8, n_layers=2, modes=(4, 8), cond_dim=8)
SDF_SHAPE = (3, 4, 4)  # retrieval key only on the v3/v4 path

#: Two-stage fixture dims (mirrors tests/test_ckpt_bundle.py: production
#: latent width, everything else shrunk).
LATENT = 32
TS_SDF_SHAPE = (8, 8, 16)

#: Real production corpus files (smoke-skipped where absent, e.g. CI).
_PROD = {
    "fam": "/nfs/wangxi/runs/b4_fam_20260824/cache_fam.npz",
    "ext": "/nfs/wangxi/runs/sdf_slender_20260828/cache_ext56.npz",
    "sdf_fam": "/nfs/wangxi/runs/b4_sdf2_20260825/sdf_fam350.npz",
    "sdf_ext": "/nfs/wangxi/runs/sdf_slender_20260828/sdf_ext2.npz",
}
_PROD_MISSING = [p for p in _PROD.values() if not os.path.isfile(p)]
_BUNDLES = "/nfs/wangxi/runs/ckpt_bundle_pm20260831"
_BUNDLE_MISSING = [
    f"{_BUNDLES}/ts2_s{s}.pt" for s in range(3) if not os.path.isfile(f"{_BUNDLES}/ts2_s{s}.pt")
]

HULL_NAMES = ("bare_hull", "with_sail", "full")


def _syn_corpus() -> dict[str, np.ndarray]:
    """Deterministic synthetic corpus in the cache.npz convention (11 rows)."""
    designs = [
        ("full", 1.0, 1.0, [50.0, 64.0, 81.0, 100.0]),
        ("with_sail", 1.5, 1.0, [50.0, 64.0, 81.0, 100.0]),
        ("bare_hull", 1.0, 1.0, [64.0, 100.0]),
        ("with_sail", 0.6, 1.0, [81.0]),
    ]
    rows: list[dict[str, float]] = []
    rng = np.random.default_rng(0)
    for hull, sail, fin, res in designs:
        geo = geometry_channels(suboff_geometry_features(hull, sail, fin, grid=TEST_GRID))
        for re in res:
            rows.append(
                dict(
                    hull=hull,
                    sail=sail,
                    fin=fin,
                    uin=0.1,
                    re=re,
                    cd=float(20.0 * (re / 50.0) ** -0.45 * (1.0 + 0.05 * rng.standard_normal())),
                    geo=geo,
                )
            )
    return {
        "hull": np.array([HULL_NAMES.index(r["hull"]) for r in rows]),
        "sail": np.array([r["sail"] for r in rows]),
        "fin": np.array([r["fin"] for r in rows]),
        "uin": np.array([r["uin"] for r in rows]),
        "re": np.array([r["re"] for r in rows]),
        "cd": np.array([r["cd"] for r in rows]),
        "geo": np.stack([r["geo"] for r in rows]),
    }


def _pool(n: int, seed: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic (n, 5, ny, nx) field pool + (n, 3, 4, 4) SDF pool.

    Fields are drawn tight around a common mean (rel-L2 to the pool mean
    ~0.06) so the in-manifold guard PASSES — like real corpus rows; use
    :func:`_outlier_pool` for the guard-failure branch.
    """
    rng = np.random.default_rng(seed)
    fields = rng.uniform(0.09, 0.11, size=(n, FIELD_CHANNELS, NY, NX)).astype(np.float32)
    sdfs = rng.standard_normal((n, *SDF_SHAPE)).astype(np.float32)
    return fields, sdfs


def _outlier_pool() -> tuple[np.ndarray, np.ndarray]:
    """Rows 0/1 in-manifold, row 2 at rel-L2 ~1 from the pool mean."""
    fields = np.ones((3, FIELD_CHANNELS, NY, NX), dtype=np.float32)
    fields[1] *= 1.02
    fields[2] *= 4.0  # ||row2 - mean|| / ||mean|| ~= 0.99 -> guard FAILS
    sdfs = (
        np.arange(3 * int(np.prod(SDF_SHAPE)), dtype=np.float64)
        .reshape(3, *SDF_SHAPE)
        .astype(np.float32)
    )
    return fields, sdfs


def _v3_service(
    field_provider: FieldProvider | None = None,
    attach_cache: bool = True,
    n_members: int = 3,
) -> tuple[DragSurrogateService, tuple[np.ndarray, np.ndarray]]:
    """Tiny v3/v4-conditioned service over the synthetic corpus."""
    corpus = _syn_corpus()
    n = len(corpus["cd"])
    ylog = np.log10(corpus["cd"])
    fields, sdfs = _pool(n)
    cond = np.concatenate(
        [
            np.log10(corpus["re"])[:, None],
            np.log10(corpus["uin"])[:, None],
            np.log10(corpus["sail"])[:, None],
            np.log10(corpus["fin"])[:, None],
            corpus["geo"],
        ],
        axis=1,
    )
    ckpts = []
    for s in range(n_members):
        torch.manual_seed(100 + s)
        model = CondFNODrag(**TINY_V3)
        with torch.no_grad():
            for pw in model.pointwise:
                pw.weight.mul_(1.0 + 0.02 * s)
        ckpts.append(
            CondDragCheckpoint(
                arch=dict(TINY_V3),
                state_dict=model.state_dict(),
                norm={
                    "ch_mean": np.zeros(5),
                    "ch_std": np.ones(5),
                    "p_mean": cond.mean(axis=0),
                    "p_std": np.where(cond.std(axis=0) < 1e-6, 1.0, cond.std(axis=0)),
                    "y_mean": float(ylog.mean()),
                    "y_std": float(max(ylog.std(), 1e-8)),
                },
                meta={"member": f"s{s}"},
            )
        )
    designs = [
        (HULL_NAMES[int(h)], float(s), float(f), float(u))
        for h, s, f, u in zip(corpus["hull"], corpus["sail"], corpus["fin"], corpus["uin"])
    ]
    svc = DragSurrogateService(
        ModelEnsembleBackend(ckpts),
        NullGuardrail(),
        corpus_cache=fields if attach_cache else None,
        grid=TEST_GRID,
        cache_re=corpus["re"] if attach_cache else None,
        cache_designs=designs if attach_cache else None,
        field_provider=field_provider,
    )
    return svc, (fields, sdfs)


class _BoomProvider(FieldProvider):
    """Fail loudly if borrowed from — pins the flag-off no-touch contract."""

    def borrow(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        raise AssertionError("FieldProvider.borrow must not run when field_policy is off")


# ---------------------------------------------------------------------------
# Policy resolver + cond assembly
# ---------------------------------------------------------------------------


class TestPolicyResolver:
    def test_default_and_explicit_names(self) -> None:
        assert resolve_field_policy() == FIELD_POLICY_CACHE == "cache"
        assert resolve_field_policy(None) == FIELD_POLICY_CACHE
        assert resolve_field_policy("field_borrow") == FIELD_POLICY_BORROW

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown field_policy"):
            resolve_field_policy("borrow")


class TestParamCondRows:
    def test_matches_load_fam_convention_and_arm_slicing(self) -> None:
        re_grid = np.array([50.0, 64.0, 81.0])
        # The e2e driver assembly: stack log10 of the four corpus columns,
        # then slice to the arm width (ts2=2, ts4=4).
        re_col = np.log10(re_grid)
        uin_col = np.full(3, np.log10(0.1))
        sail_col = np.full(3, np.log10(1.5))
        fin_col = np.full(3, np.log10(0.8))
        full = np.stack([re_col, uin_col, sail_col, fin_col], axis=1)
        np.testing.assert_array_equal(param_cond_rows(re_grid, 0.1, 1.5, 0.8, 4), full)
        np.testing.assert_array_equal(param_cond_rows(re_grid, 0.1, 1.5, 0.8, 2), full[:, :2])
        assert CORPUS_COND_KEYS == ("re", "uin", "sail", "fin")

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            param_cond_rows([], 0.1, 1.0, 1.0, 2)
        with pytest.raises(ValueError, match="finite and positive"):
            param_cond_rows([64.0], -0.1, 1.0, 1.0, 2)
        with pytest.raises(ValueError, match="finite and positive"):
            param_cond_rows([-64.0], 0.1, 1.0, 1.0, 2)
        with pytest.raises(ValueError, match="param_dim"):
            param_cond_rows([64.0], 0.1, 1.0, 1.0, 5)
        with pytest.raises(ValueError, match="param_dim"):
            param_cond_rows([64.0], 0.1, 1.0, 1.0, True)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# borrow_serving_field (provenance + guard honesty)
# ---------------------------------------------------------------------------


class TestBorrowServingField:
    def test_provenance_block_is_complete(self) -> None:
        fields, sdfs = _pool(n=4)
        provider = FieldProvider(fields, pool_sdfs=sdfs)
        got = borrow_serving_field(provider, sdfs[2] + np.float32(1e-3))
        assert got.info["field_source"] == "field_borrow"
        prov = got.info["field_borrow"]
        assert prov["strategy"] == "sdf_near"
        assert prov["donor_index"] == 2
        assert prov["guard_ok"] is True
        assert 0.0 < prov["guard_rel_l2"] <= prov["guard_threshold"] == 0.15
        assert prov["pool_size"] == 4
        assert prov["e2e"] == "/nfs/wangxi/runs/l2_e2e_validation_20260904"
        np.testing.assert_array_equal(got.fields, fields[2])

    def test_guard_failure_serves_flagged_and_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        fields, sdfs = _outlier_pool()
        provider = FieldProvider(fields, pool_sdfs=sdfs)
        with caplog.at_level(logging.WARNING, logger="tensorlbm.ai.field_borrow"):
            got = borrow_serving_field(provider, sdfs[2] + 0.5)
        assert got.info["field_borrow"]["guard_ok"] is False
        assert got.info["field_borrow"]["guard_rel_l2"] > 0.15
        assert got.info["field_borrow"]["strategy"] == "sdf_near"  # NOT re-strategied
        np.testing.assert_array_equal(got.fields, fields[2])  # still served
        assert "guard FAILED" in caplog.text

    def test_guard_pass_does_not_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        fields, sdfs = _pool(n=3)
        provider = FieldProvider(fields, pool_sdfs=sdfs)
        with caplog.at_level(logging.WARNING, logger="tensorlbm.ai.field_borrow"):
            borrow_serving_field(provider, sdfs[0])
        assert caplog.text == ""

    def test_bad_strategy_delegates_to_provider(self) -> None:
        fields, sdfs = _pool(n=2)
        with pytest.raises(ValueError, match="unknown strategy"):
            borrow_serving_field(FieldProvider(fields, pool_sdfs=sdfs), sdfs[0], strategy="sdf")


# ---------------------------------------------------------------------------
# Service hook — v3/v4 model backend
# ---------------------------------------------------------------------------


class TestServiceHookCache:
    def test_default_off_is_byte_identical(self) -> None:
        svc, (fields, _sdfs) = _v3_service()
        re_grid = np.array([50.0, 64.0, 81.0])
        a = svc.predict("full", 1.0, 1.0, re_grid)
        b = svc.predict("full", 1.0, 1.0, re_grid, field_policy=FIELD_POLICY_CACHE)
        # The verbatim pre-flag composition, re-derived by hand:
        cond, _ = svc.condition_rows("full", 1.0, 1.0, re_grid)
        field, field_info = svc._resolve_field("full", 1.0, 1.0, float(re_grid[0]), 0.1, None, None)
        mean, std, lo, hi = ensemble_stats(svc.backend.predict(field, cond))
        ref = {"cd": mean, "std": std, "lo": lo, "hi": hi}
        for arr in ("cd", "std", "lo", "hi"):
            assert getattr(a, arr).tobytes() == getattr(b, arr).tobytes()
            np.testing.assert_array_equal(getattr(a, arr), ref[arr])
        assert a.info == b.info
        assert a.info["field_source"] == "cache_design_nearest_re"
        assert "field_borrow" not in a.info
        assert a.info == {**field_info, "geometry": a.info["geometry"], "uq_temperature": 1.0}
        assert a.guard == b.guard

    def test_off_path_never_touches_the_provider(self) -> None:
        fields, sdfs = _pool(11)
        svc, _ = _v3_service(field_provider=_BoomProvider(fields[:4], pool_sdfs=sdfs[:4]))
        svc.predict("full", 1.0, 1.0, [64.0])  # cached design: no borrow, no boom
        with pytest.raises(BackendQueryError):
            svc.predict("full", 2.5, 2.5, [64.0], sdf=sdfs[0])  # flag off: still raises

    def test_flag_on_borrows_new_geometry(self) -> None:
        fields, sdfs = _pool(11)
        provider = FieldProvider(fields, pool_sdfs=sdfs)
        svc, _ = _v3_service(field_provider=provider, attach_cache=True)
        target = sdfs[9] + np.float32(1e-3)  # nearest donor: row 9
        res = svc.predict(
            "full", 2.5, 2.5, [64.0, 81.0], sdf=target, field_policy=FIELD_POLICY_BORROW
        )
        assert res.info["field_source"] == "field_borrow"
        prov = res.info["field_borrow"]
        assert prov["donor_index"] == 9
        assert prov["strategy"] == "sdf_near"
        assert prov["guard_ok"] is True
        # Served numbers are exactly the manual composition on the borrowed
        # field (same predict call, same inputs -> bit-identical).
        cond, _ = svc.condition_rows("full", 2.5, 2.5, np.array([64.0, 81.0]))
        mean, _, _, _ = ensemble_stats(svc.backend.predict(fields[9], cond))
        np.testing.assert_array_equal(res.cd, mean)

    def test_cached_design_and_caller_fields_are_not_borrowed(self) -> None:
        fields, sdfs = _pool(11)
        provider = FieldProvider(fields, pool_sdfs=sdfs)
        svc, _ = _v3_service(field_provider=provider, attach_cache=True)
        off = svc.predict("full", 1.0, 1.0, [64.0, 100.0])
        cached = svc.predict(
            "full", 1.0, 1.0, [64.0, 100.0], sdf=sdfs[0], field_policy=FIELD_POLICY_BORROW
        )
        assert cached.info["field_source"] == "cache_design_nearest_re"
        assert "field_borrow" not in cached.info
        for arr in ("cd", "std", "lo", "hi"):
            assert getattr(cached, arr).tobytes() == getattr(off, arr).tobytes()
        caller = svc.predict(
            "full",
            2.5,
            2.5,
            [64.0],
            fields=np.zeros((5, NY, NX), dtype=np.float32),
            sdf=sdfs[1],
            field_policy=FIELD_POLICY_BORROW,
        )
        assert caller.info["field_source"] == "caller"
        assert "field_borrow" not in caller.info

    def test_flag_on_cannot_help_reraises_original(self) -> None:
        fields, sdfs = _pool(11)
        svc, _ = _v3_service(field_provider=FieldProvider(fields, pool_sdfs=sdfs))
        # No query SDF: the original cache-miss error survives verbatim.
        with pytest.raises(BackendQueryError, match="not in the attached field cache"):
            svc.predict("full", 2.5, 2.5, [64.0], field_policy=FIELD_POLICY_BORROW)
        # SDF but no provider attached: same.
        svc_no_prov, _ = _v3_service(attach_cache=True)
        with pytest.raises(BackendQueryError, match="not in the attached field cache"):
            svc_no_prov.predict(
                "full", 2.5, 2.5, [64.0], sdf=sdfs[0], field_policy=FIELD_POLICY_BORROW
            )

    def test_unknown_policy_name_raises(self) -> None:
        svc, _ = _v3_service()
        with pytest.raises(ValueError, match="unknown field_policy"):
            svc.predict("full", 1.0, 1.0, [64.0], field_policy="borrow")

    def test_guard_failure_serves_flagged(self, caplog: pytest.LogCaptureFixture) -> None:
        fields, sdfs = _outlier_pool()
        provider = FieldProvider(fields, pool_sdfs=sdfs)
        svc, _ = _v3_service(field_provider=provider, attach_cache=False)
        with caplog.at_level(logging.WARNING, logger="tensorlbm.ai.field_borrow"):
            res = svc.predict(
                "full", 2.5, 2.5, [64.0], sdf=sdfs[2] + 0.5, field_policy=FIELD_POLICY_BORROW
            )
        assert res.info["field_borrow"]["guard_ok"] is False
        assert np.isfinite(res.cd).all()  # served, not suppressed
        assert "guard FAILED" in caplog.text


# ---------------------------------------------------------------------------
# Two-stage (per-member) backends: the e2e-validated composition
# ---------------------------------------------------------------------------


def _ts_member(seed: int) -> tuple[SupervisedSDFEncoder, TwoStageCondFNODrag]:
    torch.manual_seed(seed)
    trunk = SDFEncoderV2(latent_dim=LATENT, base=4)
    sup = SupervisedSDFEncoder(trunk, target_dim=3)
    full = TwoStageCondFNODrag(
        sup.encoder,
        param_dim=2,
        latent_dim=LATENT,
        aux_dim=0,
        in_ch=5,
        width=8,
        n_layers=2,
        modes=(4, 8),
        mlp_hidden=16,
        film_hidden=12,
    )
    full.freeze_encoder()
    sup.eval()
    full.eval()
    return sup, full


def _ts_norm() -> dict[str, Any]:
    rng = np.random.default_rng(11)
    return dict(
        ch_mean=rng.standard_normal(5),
        ch_std=np.abs(rng.standard_normal(5)) + 0.5,
        p_mean=rng.standard_normal(2),
        p_std=np.abs(rng.standard_normal(2)) + 0.5,
        y_mean=0.3,
        y_std=0.2,
    )


def _ts_service(
    pool_fields: np.ndarray, pool_sdfs: np.ndarray, *, attach_cache: bool
) -> tuple[DragSurrogateService, PerMemberEnsembleBackend]:
    pairs = [_ts_member(s) for s in range(2)]
    backend = PerMemberEnsembleBackend(pairs, [_ts_norm() for _ in pairs])
    n = pool_fields.shape[0]
    designs = [("full", 1.0, 1.0, 0.1)] * n
    svc = DragSurrogateService(
        backend,
        NullGuardrail(),
        corpus_cache=pool_fields if attach_cache else None,
        grid=TEST_GRID,
        cache_re=np.linspace(50.0, 100.0, n),
        cache_designs=designs if attach_cache else None,
        field_provider=FieldProvider(pool_fields, pool_sdfs=pool_sdfs),
    )
    return svc, backend


class TestServiceHookTwoStage:
    def test_borrow_serves_param_cond_composition(self) -> None:
        rng = np.random.default_rng(3)
        pool_fields = rng.uniform(0.0, 0.2, (6, FIELD_CHANNELS, NY, NX)).astype(np.float32)
        pool_sdfs = rng.standard_normal((6, *TS_SDF_SHAPE)).astype(np.float32)
        svc, backend = _ts_service(pool_fields, pool_sdfs, attach_cache=False)
        assert backend.cond_dim == 2
        re_grid = np.array([64.0, 81.0, 100.0])
        res = svc.predict(
            "full",
            1.2,
            0.9,
            re_grid,
            u_in=0.1,
            sdf=pool_sdfs[4],
            field_policy=FIELD_POLICY_BORROW,
        )
        prov = res.info["field_borrow"]
        assert prov["donor_index"] == 4 and prov["strategy"] == "sdf_near"
        # ts2 param cond, not condition_v3: same call, same inputs.
        member_cond = param_cond_rows(re_grid, 0.1, 1.2, 0.9, backend.cond_dim)
        member = backend.predict(pool_fields[4], pool_sdfs[4], member_cond)
        mean, std, _, _ = ensemble_stats(member)
        np.testing.assert_array_equal(res.cd, mean)
        np.testing.assert_array_equal(res.std, std)

    def test_two_stage_needs_the_query_sdf(self) -> None:
        rng = np.random.default_rng(4)
        pool_fields = rng.uniform(0.0, 0.2, (3, FIELD_CHANNELS, NY, NX)).astype(np.float32)
        pool_sdfs = rng.standard_normal((3, *TS_SDF_SHAPE)).astype(np.float32)
        svc, _ = _ts_service(pool_fields, pool_sdfs, attach_cache=True)
        with pytest.raises(BackendQueryError, match="pass sdf="):
            svc.predict("full", 1.0, 1.0, [64.0])  # cached design, but no SDF

    def test_cached_design_serves_own_field_with_sdf(self) -> None:
        rng = np.random.default_rng(5)
        pool_fields = rng.uniform(0.0, 0.2, (3, FIELD_CHANNELS, NY, NX)).astype(np.float32)
        pool_sdfs = rng.standard_normal((3, *TS_SDF_SHAPE)).astype(np.float32)
        svc, backend = _ts_service(pool_fields, pool_sdfs, attach_cache=True)
        res = svc.predict("full", 1.0, 1.0, [50.0, 100.0], sdf=pool_sdfs[0])
        assert res.info["field_source"] == "cache_design_nearest_re"
        assert "field_borrow" not in res.info
        member_cond = param_cond_rows([50.0, 100.0], 0.1, 1.0, 1.0, 2)
        nearest = backend.predict(pool_fields[0], pool_sdfs[0], member_cond)
        np.testing.assert_array_equal(res.cd, ensemble_stats(nearest)[0])


class TestBorrowConditioning:
    def test_composition_equals_the_pieces(self) -> None:
        fields, sdfs = _pool(n=3)
        provider = FieldProvider(fields, pool_sdfs=sdfs)
        got = borrow_conditioning(sdfs[1], [64.0, 81.0], 0.1, 1.5, 0.8, provider, param_dim=2)
        assert got.provenance["donor_index"] == 1
        np.testing.assert_array_equal(got.fields, fields[1])
        np.testing.assert_array_equal(got.cond, param_cond_rows([64.0, 81.0], 0.1, 1.5, 0.8, 2))
        assert "sdf" not in got.__dataclass_fields__  # target owns the SDF


# ---------------------------------------------------------------------------
# Real-corpus smoke: production ts2 ensemble through the new path
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    bool(_PROD_MISSING) or bool(_BUNDLE_MISSING),
    reason=f"production corpus/bundles not on this machine (missing: {_PROD_MISSING + _BUNDLE_MISSING})",
)
def test_production_ts2_borrow_matches_cached_fields(tmp_path: Path) -> None:
    """One corpus design served via field_policy=field_borrow.

    Full pool: the sdf_near donor is the design's own first row (bit-exact
    SDF, distance 0), so the served curve must be BIT-IDENTICAL to the
    manual cached-fields composition.  Leave-design-out pool: an honest
    other-design donor serves within the e2e noise floor.
    """
    for name, fname in {
        "fam": "cache_fam.npz",
        "ext": "cache_ext56.npz",
        "sdf_fam": "sdf_fam350.npz",
        "sdf_ext": "sdf_ext2.npz",
    }.items():
        os.symlink(_PROD[name], tmp_path / fname)
    provider = FieldProvider.from_corpus(str(tmp_path))
    assert provider.pool_sdfs is not None

    with np.load(_PROD["fam"]) as z:
        dsi = np.asarray(z["dsi"])
        re = np.asarray(z["re"], dtype=np.float64)
        uin = np.asarray(z["uin"], dtype=np.float64)
        hull_ids = np.asarray(z["hull"])
        sails = np.asarray(z["sail"], dtype=np.float64)
        fins = np.asarray(z["fin"], dtype=np.float64)
    # Target = first design with >= 2 rows whose geometry (SDF) is UNIQUE in
    # the corpus: designs 0-2 share one SDF (Re families of one shape), so
    # leaving one of THEM out would still retrieve a bit-identical twin and
    # prove nothing about cross-geometry borrowing.
    pool_sdfs = np.asarray(provider.pool_sdfs)
    target_dsi: int | None = None
    for g in np.unique(dsi):
        rr = np.where(dsi == g)[0]
        if rr.size < 2:
            continue
        own = set(rr.tolist())
        if not any(
            np.array_equal(pool_sdfs[i], pool_sdfs[rr[0]])
            for i in range(pool_sdfs.shape[0])
            if i not in own
        ):
            target_dsi = int(g)
            break
    assert target_dsi is not None, "no unique-geometry design in the corpus"
    rows = np.where(dsi == target_dsi)[0]
    assert rows.size >= 2
    hull = HULL_NAMES[int(hull_ids[rows[0]])]
    sail = float(sails[rows[0]])
    fin = float(fins[rows[0]])
    pos = np.unique(np.linspace(0, rows.size - 1, min(rows.size, 5)).round().astype(int))
    tgt = rows[pos]
    re_grid = re[tgt]
    u_in = float(uin[tgt[0]])
    assert np.allclose(uin[tgt], u_in)
    sdf0 = np.asarray(provider.pool_sdfs[rows[0]], dtype=np.float32)

    members = [load_member_bundle(f"{_BUNDLES}/ts2_s{s}.pt", device="cpu") for s in range(3)]
    backend = PerMemberEnsembleBackend.from_bundles(members, device="cpu")
    svc = DragSurrogateService(backend, NullGuardrail(), field_provider=provider)

    res = svc.predict(
        hull, sail, fin, re_grid, u_in=u_in, sdf=sdf0, field_policy=FIELD_POLICY_BORROW
    )
    prov = res.info["field_borrow"]
    assert prov["strategy"] == "sdf_near"
    assert prov["donor_index"] == int(rows[0])  # own design, first row
    assert prov["distance"] == pytest.approx(0.0, abs=1e-12)  # bit-exact SDF regen analog
    assert prov["guard_ok"] is True and prov["guard_rel_l2"] <= 0.15
    # Decisive check: bit-identical to the manual cached-fields composition.
    member = backend.predict(
        np.asarray(provider.pool_fields[rows[0]], dtype=np.float32),
        sdf0,
        param_cond_rows(re_grid, u_in, sail, fin, backend.cond_dim),
    )
    mean, std, _, _ = ensemble_stats(member)
    np.testing.assert_array_equal(res.cd, mean)
    np.testing.assert_array_equal(res.std, std)

    # Leave-design-out pool: honest other-design donor, bounded deviation.
    row_set = set(rows.tolist())
    keep = np.array([i for i in range(provider.pool_fields.shape[0]) if i not in row_set])
    lodo_provider = FieldProvider(provider.pool_fields[keep], pool_sdfs=provider.pool_sdfs[keep])
    lodo_svc = DragSurrogateService(backend, NullGuardrail(), field_provider=lodo_provider)
    lodo = lodo_svc.predict(
        hull, sail, fin, re_grid, u_in=u_in, sdf=sdf0, field_policy=FIELD_POLICY_BORROW
    )
    lodo_prov = lodo.info["field_borrow"]
    # donor_index numbers the REDUCED keep-pool; map back to corpus rows.
    assert int(keep[lodo_prov["donor_index"]]) not in row_set
    assert lodo_prov["distance"] > 0.0
    assert lodo_prov["guard_ok"] is True
    rel = np.abs(lodo.cd / res.cd - 1.0)
    assert rel.max() < 0.05, f"LODO donor moved the curve by {rel.max():.4f}"
