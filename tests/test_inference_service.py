"""Tests for the B4 drag-surrogate inference service layer.

Covers the three concerns of ``tensorlbm.ai.inference_service``:

- guardrails (envelope + Mahalanobis, feature-space agnosticism);
- backends (real-checkpoint ensemble with batch invariance and UQ
  statistics; replay over a synthetic v4-layout run directory);
- ONNX export honesty (plain-module blocker recorded; matmul twin parity)
  and the offline UQ / guard-calibration helpers.

All fixtures are synthetic and CPU-only; the production-grid CAD features
are exercised through the small test grid only (self-consistency is what
the service requires — see ``tests/test_drag_cond.py`` for the
production-grid geometry guarantees).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tensorlbm.ai.drag_cond import (
    CondFNODrag,
    SuboffGrid,
    condition_v3,
    geometry_channels,
    suboff_geometry_features,
)
from tensorlbm.ai.fno import SpectralConv2d
from tensorlbm.ai.inference_service import (
    COND_V3_CHANNEL_NAMES,
    ENV_UQ_TEMPERATURE,
    RE_POLICY_NETWORK,
    RE_POLICY_QUAD3_FALLBACK,
    BackendQueryError,
    CondDragCheckpoint,
    DragSurrogateService,
    EnvelopeMahalanobisGuardrail,
    ModelEnsembleBackend,
    NullGuardrail,
    ReplayEnsembleBackend,
    SpectralConv2dMatmul,
    choose_threshold,
    ensemble_picp,
    ensemble_stats,
    error_std_spearman,
    export_cond_fno_onnx,
    guard_threshold_sweep,
    load_checkpoint,
    load_corpus_index,
    quad3_loo_std,
    quad3_nearest3,
    resolve_re_policy,
    resolve_uq_temperature,
    save_checkpoint,
    to_matmul_spectral,
)

TEST_GRID = SuboffGrid.from_resolution(32)  # ny=nz=16, nx=32
TINY_ARCH = dict(in_ch=5, width=8, n_layers=2, modes=(4, 8), cond_dim=8)

#: Synthetic corpus: three designs, one with an Re sweep (for replay
#: interpolation), 238 rows total is unnecessary — 4+4+2 points suffice.
SYN_DESIGNS = [
    ("full", 1.0, 1.0, [50.0, 64.0, 81.0, 100.0]),
    ("with_sail", 1.5, 1.0, [50.0, 64.0, 81.0, 100.0]),
    ("bare_hull", 1.0, 1.0, [64.0, 100.0]),
    ("with_sail", 0.6, 1.0, [81.0]),  # single-Re design: replay single_point mode
]


def _syn_corpus(grid: SuboffGrid = TEST_GRID) -> dict[str, np.ndarray]:
    """Deterministic synthetic corpus in the cache.npz convention."""
    rows: list[dict[str, float]] = []
    rng = np.random.default_rng(0)
    for hull, sail, fin, res in SYN_DESIGNS:
        geo = geometry_channels(suboff_geometry_features(hull, sail, fin, grid=grid))
        for re in res:
            rows.append(
                dict(
                    hull=hull,
                    sail=sail,
                    fin=fin,
                    uin=0.1,
                    re=re,
                    geo=geo,
                    cd=float(20.0 * (re / 50.0) ** -0.45 * (1.0 + 0.05 * rng.standard_normal())),
                )
            )
    out = {
        "hull": np.array([{"bare_hull": 0, "with_sail": 1, "full": 2}[r["hull"]] for r in rows]),
        "sail": np.array([r["sail"] for r in rows]),
        "fin": np.array([r["fin"] for r in rows]),
        "uin": np.array([r["uin"] for r in rows]),
        "re": np.array([r["re"] for r in rows]),
        "dsi": np.zeros(len(rows), dtype=np.int64),
        "cd": np.array([r["cd"] for r in rows]),
        "geo": np.stack([r["geo"] for r in rows]),
    }
    out["cond"] = condition_v3(out["re"], out["uin"], out["sail"], out["fin"], out["geo"])
    return out


def _tiny_member(seed: int) -> CondFNODrag:
    torch.manual_seed(seed)
    return CondFNODrag(**TINY_ARCH)


def _member_norm(cond: np.ndarray, ylog: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "ch_mean": np.zeros(5, dtype=np.float64),
        "ch_std": np.ones(5, dtype=np.float64),
        "p_mean": cond.mean(axis=0),
        "p_std": np.where(cond.std(axis=0) < 1e-6, 1.0, cond.std(axis=0)),
        "y_mean": float(ylog.mean()),
        "y_std": float(max(ylog.std(), 1e-8)),
    }


# ---------------------------------------------------------------------------
# Guardrail
# ---------------------------------------------------------------------------


class TestGuardrail:
    def test_in_distribution_rows_are_ok(self) -> None:
        rng = np.random.default_rng(1)
        feats = rng.normal(size=(200, 8))
        guard = EnvelopeMahalanobisGuardrail(feats)
        verdict = guard.check(feats[:10])
        assert verdict.flag == "ok"
        assert verdict.reasons == ()

    def test_envelope_violation_rejects_and_names_dimension(self) -> None:
        rng = np.random.default_rng(2)
        feats = rng.normal(size=(200, 8)) * [1, 2, 0.5, 1, 1, 1, 1, 1]
        names = tuple(f"f{i}" for i in range(8))
        guard = EnvelopeMahalanobisGuardrail(feats, names, margin=0.0)
        query = feats.mean(axis=0)[None, :].repeat(3, axis=0)
        query[:, 2] = feats[:, 2].max() + 5.0  # far outside one envelope dim
        verdict = guard.check(query)
        assert verdict.flag == "reject"
        assert any("f2" in r for r in verdict.reasons)
        assert any("above envelope" in r for r in verdict.reasons)

    def test_mahalanobis_flags_correlated_shift(self) -> None:
        # An in-envelope but jointly unlikely row (off the correlated ridge).
        rng = np.random.default_rng(3)
        base = rng.normal(size=(400, 4))
        feats = np.concatenate([base, base[:, :1] + base[:, 1:2] * 0.1], axis=1)
        guard = EnvelopeMahalanobisGuardrail(feats, margin=0.0)
        query = feats.mean(axis=0)[None, :]
        query[0, 4] = feats[:, 4].max()  # last dim at its edge while dim0 is mean
        assert guard.check(query).flag in {"review", "reject"}

    def test_margin_widens_envelope(self) -> None:
        rng = np.random.default_rng(4)
        feats = rng.normal(size=(100, 3))
        tight = EnvelopeMahalanobisGuardrail(feats, margin=0.0)
        wide = EnvelopeMahalanobisGuardrail(feats, margin=0.5)
        query = feats.max(axis=0)[None, :] + 0.2
        assert tight.check(query).flag == "reject"
        # The widened envelope admits it unless the Mahalanobis ball rejects.
        if wide.row_scores(query)[0] < wide.mahal_threshold:
            assert wide.check(query).flag == "ok"

    def test_feature_space_agnostic_latent_dropin(self) -> None:
        """The guard accepts a 32-dim SDF-latent-style matrix unchanged."""
        rng = np.random.default_rng(5)
        latents = rng.normal(size=(238, 32)) * rng.uniform(0.1, 2.0, size=32)
        guard = EnvelopeMahalanobisGuardrail(latents, names=tuple(f"z{i}" for i in range(32)))
        assert guard.feature_names[0] == "z0"
        assert guard.check(latents[:5]).flag == "ok"
        far = latents[:1] + 10.0
        assert guard.check(far).flag == "reject"

    def test_invalid_fits_raise(self) -> None:
        with pytest.raises(ValueError, match="N>=2"):
            EnvelopeMahalanobisGuardrail(np.ones((1, 8)))
        with pytest.raises(ValueError, match="non-finite"):
            EnvelopeMahalanobisGuardrail(np.full((10, 8), np.nan))

    def test_null_guard_passes_everything(self) -> None:
        g = NullGuardrail()
        assert g.feature_names == COND_V3_CHANNEL_NAMES
        assert g.n_fit == 0
        verdict = g.check(np.full((3, 8), 1e9))
        assert (verdict.flag, verdict.score, verdict.reasons) == ("ok", 0.0, ())
        np.testing.assert_array_equal(g.row_scores(np.zeros((2, 8))), [0.0, 0.0])


# ---------------------------------------------------------------------------
# Checkpoints + model backend
# ---------------------------------------------------------------------------


class TestCheckpoints:
    def test_roundtrip_preserves_predictions(self, tmp_path: Path) -> None:
        cond = _syn_corpus()["cond"]
        ylog = np.log10(_syn_corpus()["cd"])
        model = _tiny_member(0)
        ckpt = CondDragCheckpoint(
            arch=dict(TINY_ARCH),
            state_dict=model.state_dict(),
            norm=_member_norm(cond, ylog),
            meta={"member": "s0"},
        )
        path = save_checkpoint(ckpt, tmp_path / "member_s0.pt")
        loaded = load_checkpoint(path)
        torch.testing.assert_close(loaded.to_model().spectral[0].weight, model.spectral[0].weight)
        x = torch.randn(3, 5, TEST_GRID.ny, TEST_GRID.nx)
        p = torch.randn(3, 8)
        with torch.no_grad():
            torch.testing.assert_close(loaded.to_model()(x, p), model(x, p))
        assert loaded.meta["member"] == "s0"

    def test_rejects_foreign_files(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.pt"
        torch.save({"something": "else"}, bad)
        with pytest.raises(ValueError, match="not a CondDragCheckpoint"):
            load_checkpoint(bad)


class TestModelBackend:
    def _service(self, n_members: int = 3) -> DragSurrogateService:
        corpus = _syn_corpus()
        ylog = np.log10(corpus["cd"])
        ckpts = []
        for s in range(n_members):
            model = _tiny_member(100 + s)
            # Give each member a distinct deterministic output scale so the
            # ensemble std is non-degenerate.
            with torch.no_grad():
                for pw in model.pointwise:
                    pw.weight.mul_(1.0 + 0.02 * s)
            ckpts.append(
                CondDragCheckpoint(
                    arch=dict(TINY_ARCH),
                    state_dict=model.state_dict(),
                    norm=_member_norm(corpus["cond"], ylog),
                    meta={"member": f"s{s}"},
                )
            )
        rng = np.random.default_rng(7)
        fields = rng.uniform(0.0, 0.2, size=(len(corpus["cd"]), 5, TEST_GRID.ny, TEST_GRID.nx))
        designs = [
            (str(h), float(s), float(f), float(u))
            for h, s, f, u in zip(corpus["hull"], corpus["sail"], corpus["fin"], corpus["uin"])
        ]
        # Numeric hull ids need mapping back to names for the design index.
        names = ["bare_hull", "with_sail", "full"]
        designs = [(names[int(h)], s, f, u) for h, s, f, u in designs]
        guard = EnvelopeMahalanobisGuardrail(corpus["cond"])
        return DragSurrogateService(
            ModelEnsembleBackend(ckpts),
            guard,
            corpus_cache=fields,
            grid=TEST_GRID,
            cache_re=corpus["re"],
            cache_designs=designs,
        )

    def test_predict_curve_shapes_and_uq(self) -> None:
        svc = self._service()
        res = svc.predict("full", 1.0, 1.0, np.geomspace(50.0, 100.0, 64))
        assert res.cd.shape == (64,)
        assert res.lo.shape == res.hi.shape == res.std.shape == (64,)
        assert np.all(res.lo <= res.cd) and np.all(res.cd <= res.hi)
        assert np.all(res.std > 0)  # members were perturbed
        assert res.backend == "model"
        assert res.members == ("s0", "s1", "s2")
        assert res.guard.flag == "ok"
        assert res.info["field_source"] == "cache_design_nearest_re"
        assert res.info["field_row"] in range(10)

    def test_batch_invariance(self) -> None:
        svc = self._service()
        grid = np.geomspace(50.0, 100.0, 8)
        batched = svc.predict("full", 1.0, 1.0, grid)
        for i, re in enumerate(grid):
            single = svc.predict("full", 1.0, 1.0, [re])
            np.testing.assert_allclose(single.cd[0], batched.cd[i], rtol=1e-5, atol=1e-8)

    def test_explicit_fields_and_field_point(self) -> None:
        svc = self._service()
        fields = np.zeros((5, TEST_GRID.ny, TEST_GRID.nx), dtype=np.float32)
        a = svc.predict("full", 1.0, 1.0, [64.0], fields=fields)
        b = svc.predict("full", 1.0, 1.0, [64.0], field_point=0)
        assert a.info["field_source"] == "caller"
        assert b.info["field_source"] == "cache_row"
        np.testing.assert_allclose(a.cd, b.cd, rtol=1e-4)  # field 0 is ~zeros vs cache row

    def test_field_resolution_failure(self) -> None:
        svc = self._service()
        with pytest.raises(BackendQueryError):
            svc.predict("full", 2.5, 2.5, [64.0])  # design absent from cache

    def test_guard_flags_extrapolated_re(self) -> None:
        svc = self._service()
        res = svc.predict("full", 1.0, 1.0, [50.0, 100.0, 5000.0])
        assert res.guard.flag == "reject"
        assert any("log10_re" in r for r in res.guard.reasons)

    def test_input_validation(self) -> None:
        svc = self._service()
        with pytest.raises(ValueError, match="non-empty"):
            svc.predict("full", 1.0, 1.0, [])
        with pytest.raises(ValueError, match="finite and positive"):
            svc.predict("full", 1.0, 1.0, [-10.0])


# ---------------------------------------------------------------------------
# Replay backend
# ---------------------------------------------------------------------------


def _write_syn_run_dir(tmp_path: Path) -> Path:
    """Synthetic run directory in the v3/v4 preds layout."""
    corpus = _syn_corpus()
    n = len(corpus["cd"])
    rng_x = np.random.default_rng(12)
    x = rng_x.standard_normal((n, 5, TEST_GRID.ny, TEST_GRID.nx)).astype(np.float32)
    np.savez(
        tmp_path / "cache.npz",
        x=x,
        **{k: v for k, v in corpus.items() if k != "cond"},
    )
    np.savez(tmp_path / "cache_v3.npz", geo=corpus["geo"])
    preds: dict[str, np.ndarray] = {}
    fold_rows = np.arange(n)  # evaluate every corpus row in the fold
    truth = corpus["cd"].copy()
    rng = np.random.default_rng(11)
    preds["loho::full::C_full::true"] = truth
    preds["loho::full::C_full::idx"] = fold_rows
    for tag in ("", "s1", "s2"):
        member = truth * (1.0 + 0.03 * rng.standard_normal(n))
        key = "loho::full::C_full::pred" if tag == "" else f"loho::full::C_full::{tag}::pred"
        preds[key] = member
        if tag:
            preds[f"loho::full::C_full::{tag}::true"] = truth
            preds[f"loho::full::C_full::{tag}::idx"] = fold_rows
    np.savez(tmp_path / "preds_v4.npz", **preds)
    return tmp_path


class TestReplayBackend:
    def test_member_discovery_and_shapes(self, tmp_path: Path) -> None:
        be = ReplayEnsembleBackend(_write_syn_run_dir(tmp_path))
        assert be.n_members == 3
        assert be.member_labels() == ["s0", "s1", "s2"]
        n_rows = sum(len(d[3]) for d in SYN_DESIGNS)
        assert be.member_matrix().shape == (3, n_rows)

    def test_exact_and_interpolated_service(self, tmp_path: Path) -> None:
        run_dir = _write_syn_run_dir(tmp_path)
        svc = DragSurrogateService.from_run_dir(run_dir, arm="C_full", fold="loho::full")
        res = svc.predict("full", 1.0, 1.0, [50.0, 64.0, 73.0, 100.0])
        assert res.backend == "replay"
        assert res.info["mode"] == "log_re_interp"
        assert res.info["n_exact"] == 3
        assert res.info["n_interpolated"] == 1
        assert res.info["n_extrapolated"] == 0
        # Exact points must equal the archived member predictions at row 0.
        be = ReplayEnsembleBackend(run_dir)
        member = be.member_matrix()
        np.testing.assert_allclose(res.cd[0], member[:, 0].mean(), rtol=1e-12)
        # log-Re interpolation is linear per member between 64 and 81.
        w = (np.log10(73.0) - np.log10(64.0)) / (np.log10(81.0) - np.log10(64.0))
        row_of_64 = 1
        row_of_81 = 2
        expect = member[:, row_of_64] + w * (member[:, row_of_81] - member[:, row_of_64])
        np.testing.assert_allclose(res.cd[2], expect.mean(), rtol=1e-12)

    def test_off_sweep_nearest_fallback(self, tmp_path: Path) -> None:
        run_dir = _write_syn_run_dir(tmp_path)
        svc = DragSurrogateService.from_run_dir(run_dir)
        res = svc.predict("full", 1.0, 1.0, [10.0, 500.0])
        assert res.info["n_extrapolated"] == 2
        # Off-sweep queries clamp to the nearest archived member value on
        # their side of the sweep (re=50 below, re=100 above).
        be = ReplayEnsembleBackend(run_dir)
        member = be.member_matrix()
        np.testing.assert_allclose(res.cd[0], member[:, 0].mean(), rtol=1e-12)
        np.testing.assert_allclose(res.cd[1], member[:, 3].mean(), rtol=1e-12)

    def test_single_point_design_mode(self, tmp_path: Path) -> None:
        svc = DragSurrogateService.from_run_dir(_write_syn_run_dir(tmp_path))
        res = svc.predict("with_sail", 0.6, 1.0, [81.0, 120.0])
        assert res.info["mode"] == "single_point"
        assert res.info["n_exact"] == 1
        assert res.info["n_extrapolated"] == 1
        # Both queries repeat the single archived member value.
        np.testing.assert_allclose(res.cd[0], res.cd[1])

    def test_unknown_design_raises(self, tmp_path: Path) -> None:
        svc = DragSurrogateService.from_run_dir(_write_syn_run_dir(tmp_path))
        with pytest.raises(BackendQueryError, match="not present"):
            svc.predict("full", 9.9, 1.0, [64.0])

    def test_uq_band_matches_members(self, tmp_path: Path) -> None:
        run_dir = _write_syn_run_dir(tmp_path)
        svc = DragSurrogateService.from_run_dir(run_dir)
        res = svc.predict("full", 1.0, 1.0, [64.0])
        be = ReplayEnsembleBackend(run_dir)
        col = be.member_matrix()[:, 1]
        np.testing.assert_allclose(
            [res.lo[0], res.hi[0], res.cd[0]], [col.min(), col.max(), col.mean()], rtol=1e-12
        )
        assert res.std[0] == pytest.approx(col.std(ddof=1))

    def test_default_guard_features_from_caches(self, tmp_path: Path) -> None:
        from tensorlbm.ai.inference_service import default_guard_features

        feats = default_guard_features(_write_syn_run_dir(tmp_path))
        assert feats.shape == (11, 8)
        ref = _syn_corpus()["cond"]
        np.testing.assert_allclose(feats, ref, rtol=1e-12)

    def test_load_corpus_index(self, tmp_path: Path) -> None:
        idx = load_corpus_index(_write_syn_run_dir(tmp_path))
        assert idx.fields.shape == (11, 5, TEST_GRID.ny, TEST_GRID.nx)
        assert idx.fields.dtype == np.float32
        assert idx.re.shape == (11,)
        assert idx.designs[0] == ("full", 1.0, 1.0, 0.1)
        np.testing.assert_allclose(idx.cond, _syn_corpus()["cond"], rtol=1e-12)

    def test_v4_layout_cache_discovery(self, tmp_path: Path) -> None:
        """The real v4 run dir keeps everything in ``cache_v4.npz`` (geo
        inside); the backend + guard-feature loader must find it without
        being told."""
        from tensorlbm.ai.inference_service import default_guard_features

        run_dir = _write_syn_run_dir(tmp_path)
        v4_dir = tmp_path / "v4"
        v4_dir.mkdir()
        combined = dict(np.load(run_dir / "cache.npz"))
        combined["geo"] = np.load(run_dir / "cache_v3.npz")["geo"]
        np.savez(v4_dir / "cache_v4.npz", **combined)
        np.savez(v4_dir / "preds_v4.npz", **dict(np.load(run_dir / "preds_v4.npz")))

        svc = DragSurrogateService.from_run_dir(v4_dir)
        res = svc.predict("full", 1.0, 1.0, [64.0])
        np.testing.assert_allclose(
            res.cd,
            DragSurrogateService.from_run_dir(run_dir).predict("full", 1.0, 1.0, [64.0]).cd,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            default_guard_features(v4_dir), default_guard_features(run_dir), rtol=1e-12
        )

    def test_grid_passthrough_guard_consistency(self, tmp_path: Path) -> None:
        """``from_run_dir(grid=...)`` must evaluate query geometry on the
        grid the guard fit matrix was computed on.  A grid mismatch moves
        the geometry channels (out of the fit feature space), whether or
        not the widened envelope catches it — consistency is load-bearing."""
        run_dir = _write_syn_run_dir(tmp_path)
        matched = DragSurrogateService.from_run_dir(run_dir, grid=TEST_GRID)
        verdict = matched.predict("full", 1.0, 1.0, [64.0]).guard
        assert verdict.flag == "ok", verdict.reasons
        cond_ok, geo_ok = matched.condition_rows("full", 1.0, 1.0, [64.0])
        default = DragSurrogateService.from_run_dir(run_dir)
        cond_def, geo_def = default.condition_rows("full", 1.0, 1.0, [64.0])
        assert not np.allclose(geo_ok["sail_frac"], geo_def["sail_frac"])
        assert not np.allclose(cond_ok, cond_def)


# ---------------------------------------------------------------------------
# UQ temperature knob (#251 landing: reported-sigma scaling only)
# ---------------------------------------------------------------------------


def _resvc(svc: DragSurrogateService, t: float | str | None) -> DragSurrogateService:
    """Re-wrap a built service with a temperature, keeping backend/guard/cache."""
    return DragSurrogateService(
        svc.backend,
        svc.guard,
        corpus_cache=svc.corpus_cache,
        grid=svc.grid,
        cache_re=svc.cache_re,
        cache_designs=svc.cache_designs,
        uq_temperature=t,
    )


class TestUqTemperature:
    """``arg > TENSORLBM_DRAG_UQ_TEMPERATURE > 1.0``; sigma-only semantics."""

    def test_resolver_default_and_blank_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ENV_UQ_TEMPERATURE, raising=False)
        assert resolve_uq_temperature() == 1.0
        assert resolve_uq_temperature(None) == 1.0
        monkeypatch.setenv(ENV_UQ_TEMPERATURE, "  ")
        assert resolve_uq_temperature() == 1.0  # blank env counts as unset

    def test_resolver_env_and_arg_precedence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENV_UQ_TEMPERATURE, "2.3")
        assert resolve_uq_temperature() == 2.3  # env used when arg is None
        assert resolve_uq_temperature("1.0") == 1.0  # arg beats env
        assert resolve_uq_temperature(1.7) == 1.7  # float args pass through
        monkeypatch.setenv(ENV_UQ_TEMPERATURE, "2.5 ")
        assert resolve_uq_temperature() == 2.5  # surrounding whitespace ignored

    def test_resolver_rejects_bad_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for bad in ("abc", "0", "-2", "inf", "nan"):
            with pytest.raises(ValueError, match=ENV_UQ_TEMPERATURE):
                resolve_uq_temperature(bad)
        monkeypatch.setenv(ENV_UQ_TEMPERATURE, "zero")
        with pytest.raises(ValueError, match=ENV_UQ_TEMPERATURE):
            resolve_uq_temperature()

    def test_default_is_bit_identical(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ENV_UQ_TEMPERATURE, raising=False)
        base = TestModelBackend()._service()  # noqa: SLF001 — fixture reuse
        explicit_none = _resvc(base, None)
        explicit_one = _resvc(base, 1.0)
        re_grid = np.geomspace(50.0, 100.0, 16)
        for other in (explicit_none, explicit_one):
            a = base.predict("full", 1.0, 1.0, re_grid)
            b = other.predict("full", 1.0, 1.0, re_grid)
            np.testing.assert_array_equal(a.std, b.std)  # bit-for-bit, not approx
            np.testing.assert_array_equal(a.cd, b.cd)
            np.testing.assert_array_equal(a.lo, b.lo)
            np.testing.assert_array_equal(a.hi, b.hi)
            assert a.guard == b.guard
            assert b.info["uq_temperature"] == 1.0

    def test_temperature_scales_reported_std_exactly(self) -> None:
        base = _resvc(TestModelBackend()._service(), 1.0)  # noqa: SLF001 — fixture reuse
        warm = _resvc(base, 2.3)
        re_grid = np.geomspace(50.0, 100.0, 16)
        a = base.predict("full", 1.0, 1.0, re_grid)
        b = warm.predict("full", 1.0, 1.0, re_grid)
        np.testing.assert_array_equal(b.std, a.std * 2.3)  # exact x2.3
        assert b.uq_dict()["mean_std"] == pytest.approx(float(np.mean(a.std)) * 2.3, rel=1e-12)
        np.testing.assert_array_equal(b.cd, a.cd)  # mean untouched
        np.testing.assert_array_equal(b.lo, a.lo)  # min-max band untouched
        np.testing.assert_array_equal(b.hi, a.hi)
        assert b.members == a.members and b.backend == a.backend
        assert b.info["uq_temperature"] == 2.3

    def test_verdict_invariant_under_temperature(self) -> None:
        base = _resvc(TestModelBackend()._service(), 1.0)  # noqa: SLF001 — fixture reuse
        warm = _resvc(base, 2.3)
        for re_grid in ([64.0], [50.0, 100.0, 5000.0]):  # ok case and reject case
            a = base.predict("full", 1.0, 1.0, re_grid)
            b = warm.predict("full", 1.0, 1.0, re_grid)
            assert b.guard == a.guard  # flag, score and reasons bit-identical
            assert b.guard.flag == a.guard.flag

    def test_env_reaches_service_and_arg_wins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_dir = _write_syn_run_dir(tmp_path)
        monkeypatch.setenv(ENV_UQ_TEMPERATURE, "2.5")
        env_svc = DragSurrogateService.from_run_dir(run_dir)
        arg_svc = DragSurrogateService.from_run_dir(run_dir, uq_temperature=1.0)
        raw = env_svc.predict("full", 1.0, 1.0, [64.0])
        got = arg_svc.predict("full", 1.0, 1.0, [64.0])
        assert env_svc.uq_temperature == 2.5
        np.testing.assert_array_equal(raw.std, got.std * 2.5)
        assert got.info["uq_temperature"] == 1.0
        assert raw.info["uq_temperature"] == 2.5


# ---------------------------------------------------------------------------
# UQ + calibration helpers
# ---------------------------------------------------------------------------


class TestUQHelpers:
    def test_ensemble_stats_single_member(self) -> None:
        mean, std, lo, hi = ensemble_stats(np.array([[1.0, 2.0, 3.0]]))
        np.testing.assert_allclose(mean, [1, 2, 3])
        np.testing.assert_allclose(std, [0, 0, 0])
        np.testing.assert_allclose(lo, [1, 2, 3])
        np.testing.assert_allclose(hi, [1, 2, 3])

    def test_picp(self) -> None:
        member = np.array([[1.0, 2.0], [3.0, 4.0], [2.0, 6.0]])
        assert ensemble_picp(member, np.array([2.0, 5.0])) == 1.0
        assert ensemble_picp(member, np.array([0.5, 5.0])) == 0.5

    def test_spearman(self) -> None:
        std = np.array([1.0, 2.0, 3.0, 4.0])
        err = np.array([0.1, 0.4, 0.2, 0.8])
        assert error_std_spearman(std, err) == pytest.approx(0.8, abs=0.01)
        assert error_std_spearman(err, err) == pytest.approx(1.0)

    def test_threshold_sweep_and_choice(self) -> None:
        scores = np.array([0.5, 1.0, 2.0, 3.5, 8.0, 9.0])
        errors = np.array([0.01, 0.02, 0.05, 0.20, 0.30, 0.40])
        rows = guard_threshold_sweep(scores, errors, large_error=0.10)
        by_t = {round(r.threshold, 3): r for r in rows}
        assert by_t[3.5].n_flagged == 2  # 8.0, 9.0
        assert by_t[3.5].n_large == 3
        assert by_t[3.5].large_captured == 2
        assert by_t[3.5].capture_rate == pytest.approx(2 / 3)
        assert by_t[3.5].precision == pytest.approx(1.0)
        best = choose_threshold(rows, target_capture=2 / 3)
        assert best.threshold == pytest.approx(3.5)
        # An unreachable target (>1 capture) falls back to the last row.
        loose = choose_threshold(rows, target_capture=1.5)
        assert loose is rows[-1]

    def test_sweep_validation(self) -> None:
        with pytest.raises(ValueError, match="same-length"):
            guard_threshold_sweep(np.ones(4), np.ones(5))


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------


def _onnx_available() -> bool:
    try:
        import onnx  # noqa: F401

        return True
    except ImportError:
        return False


class TestOnnxExport:
    def test_spectral_matmul_parity(self) -> None:
        torch.manual_seed(0)
        for ny, nx, my, mx in [(16, 32, 6, 10), (32, 64, 8, 16), (8, 8, 4, 5)]:
            twin = SpectralConv2dMatmul(3, 4, my, mx, ny=ny, nx=nx)
            ref = SpectralConv2d(3, 4, my, mx)
            with torch.no_grad():
                ref.weight.copy_(twin.weight)
            x = torch.randn(2, 3, ny, nx, dtype=torch.float64)
            twin = twin.double()
            got = twin(x)
            want = ref.double()(x)
            rel = (got - want).abs().max() / want.abs().max()
            assert rel < 1e-5, f"parity broke at ({ny},{nx},{my},{mx}): {rel}"

    def test_to_matmul_spectral_model_parity(self) -> None:
        torch.manual_seed(1)
        model = CondFNODrag(**TINY_ARCH)
        twin = to_matmul_spectral(model, ny=TEST_GRID.ny, nx=TEST_GRID.nx)
        assert all(isinstance(s, SpectralConv2dMatmul) for s in twin.spectral)
        x = torch.randn(3, 5, TEST_GRID.ny, TEST_GRID.nx)
        p = torch.randn(3, 8)
        with torch.no_grad():
            rel = (twin(x, p) - model(x, p)).abs().max() / model(x, p).abs().max()
        assert rel < 1e-4

    def test_export_report_structure(self, tmp_path: Path) -> None:
        torch.manual_seed(2)
        model = CondFNODrag(**TINY_ARCH)
        report = export_cond_fno_onnx(
            model, tmp_path / "drag.onnx", ny=TEST_GRID.ny, nx=TEST_GRID.nx
        )
        # The plain module is fft-blocked on every torch we support; if a
        # future torch lifts that, the report just says so.
        assert isinstance(report["plain_blocker"], (str, type(None)))
        if _onnx_available():
            assert report["matmul_export_ok"] or report["matmul_blocker"]
            if report["matmul_export_ok"]:
                assert Path(report["path"]).is_file()
                assert report["matmul_parity_max_abs"] < 1e-3
        else:
            # Without the onnx package the exporter cannot serialise; the
            # blocker must be recorded verbatim, never papered over.
            assert report["matmul_export_ok"] is False
            assert report["matmul_blocker"] and "onnx" in report["matmul_blocker"]
            assert report["checker"].startswith("skipped")

    def test_rejects_oversized_modes(self) -> None:
        with pytest.raises(ValueError, match="exceed rfft2 corner"):
            SpectralConv2dMatmul(2, 2, 4, 10, ny=8, nx=8)  # mx > nx//2+1


# ---------------------------------------------------------------------------
# Real-data fixture: fresh-Re campaign 2026-08-27 (embedded copy)
#
# Cached curves (28 rows per family; extrap_eval.json::curves rows with
# source="cache", i.e. the cache_fam.npz family rows the campaign read)
# and the campaign's stored quad3 predictions
# (extrap_eval.json::per_point[].preds.quad3) for every out-of-window
# scan point of the family.  Generated by gen_fixture.py in the run
# directory from the read-only artifacts; the committed test is
# self-contained (no /nfs dependency).
# ---------------------------------------------------------------------------

CAMPAIGN_QUAD3_FIXTURES: dict[str, dict[str, list]] = {
    "fam_long_nose": {
        "re": [
            65.16140657325276,
            70.71137568379525,
            76.03975245412214,
            83.3413547668938,
            89.28411747958448,
            94.80170276095032,
            110.43008152149676,
            118.3075217516246,
            130.22214081478202,
            138.12646362478029,
            146.38188543266966,
            162.96785252219783,
            182.6968787924799,
            189.4179180446004,
            215.4693041033772,
            226.62714658903823,
            253.2220791012612,
            269.52480138258574,
            297.28494110429665,
            329.61710274954675,
            363.2209602992397,
            402.36935396514656,
            436.8960518360348,
            475.4385113819902,
            536.84437509048,
            549.1093897168645,
            614.1261618276332,
            689.0152018687742,
        ],
        "cd": [
            15.048160489572032,
            14.124587549355951,
            13.356466258250888,
            12.45290389973415,
            11.819186806933928,
            11.29693090316061,
            10.081395201946364,
            9.581514075270865,
            8.931447959900513,
            8.557047004211846,
            8.20553556722525,
            7.597851995761467,
            7.006337104926574,
            6.830350756390045,
            6.242337468883966,
            6.027845510677114,
            5.585347067258537,
            5.352908804859608,
            5.009768406592615,
            4.675475013480611,
            4.384476438879149,
            4.10019518756499,
            3.8871151328801288,
            3.681715009351299,
            3.4086271666296537,
            3.3605357040077033,
            3.133949117530902,
            2.9199130824218464,
        ],
        "checks": [
            [1374.76607, 1.9522656643630625],
            [41.1140681, 21.683339249589327],
            [867.418747, 2.542510800845309],
            [1092.0155, 2.223237221891132],
            [58.0751647, 16.464952881979112],
        ],
    },
    "fam_blunt": {
        "re": [
            63.040232447150814,
            66.68339683586994,
            74.1833375680585,
            84.15222656073834,
            85.43540318304365,
            100.28709881870822,
            108.7660319287255,
            118.79596740771956,
            124.31673918647955,
            142.4167644359085,
            156.95071796321778,
            158.18899199651864,
            172.37957660037605,
            198.03883794203594,
            222.36617404076435,
            243.30552854704766,
            264.3351071843176,
            267.4545105342252,
            311.19858716002994,
            328.93440151384345,
            366.01118174098207,
            384.72475109465796,
            443.89494145728656,
            473.0986470297965,
            510.9887990848751,
            562.014701101921,
            605.2197842624075,
            647.920992538194,
        ],
        "cd": [
            12.232159857105994,
            11.718745011270844,
            10.809938106103798,
            9.836146812220768,
            9.726126977641666,
            8.643256150784769,
            8.148023825488053,
            7.646383717157212,
            7.401964790187775,
            6.722628361312672,
            6.280884080832558,
            6.246647589141502,
            5.886046301124484,
            5.353016278906059,
            4.950271401404669,
            4.661472633181671,
            4.412544131046161,
            4.378572415194607,
            3.966091172104435,
            3.8265865327723856,
            3.5736047428300868,
            3.4622376615880053,
            3.16472048497027,
            3.0420363382602638,
            2.901181516353972,
            2.737826227010893,
            2.6183610160868347,
            2.5139606369027336,
        ],
        "checks": [
            [726.979311, 2.3489695706084426],
            [815.684202, 2.1970912496865362],
            [1292.77234, 1.6992166885681934],
            [56.1846663, 13.365025680831787],
        ],
    },
    "fam_slender": {
        "re": [
            63.23585728669994,
            66.41675391128827,
            72.93530477206203,
            85.02984245221715,
            87.38475268466478,
            99.08747597707938,
            109.59128122462343,
            113.41364956625354,
            122.15344969590552,
            137.53984327540513,
            151.0674997476282,
            161.83076325067154,
            177.55049906514245,
            190.21453315501776,
            205.920406235933,
            234.59027998489364,
            255.02009937026722,
            268.13795281010675,
            303.31595436959464,
            333.63977774449563,
            348.98114199567345,
            409.3701473361305,
            435.9003362625295,
            464.047229179716,
            507.79721870262904,
            558.0874545685491,
            591.197555435092,
            654.214690073743,
        ],
        "cd": [
            23.105146937027925,
            22.232205930298985,
            20.66611611350838,
            18.357408811515636,
            17.977245176166647,
            16.337923475629868,
            15.144225206403501,
            14.760609658745425,
            13.966781016483546,
            12.795851536627325,
            11.947937910223288,
            11.366119847544244,
            10.632309797885162,
            10.121567706089643,
            9.567520726871274,
            8.730278518050907,
            8.23781142410423,
            7.957251165505674,
            7.3130062020694195,
            6.855914208239741,
            6.651751689745909,
            5.981745539188738,
            5.739914143147567,
            5.510234262882952,
            5.198217577167918,
            4.893327895773663,
            4.717670519215364,
            4.426893806306733,
        ],
        "checks": [
            [734.040955, 4.122080928969288],
            [1036.86041, 3.348607572706566],
            [50.2300269, 27.737607267836943],
        ],
    },
}


class TestQuad3RealData:
    def test_reproduces_campaign_quad3_within_1e_9(self) -> None:
        """quad3_nearest3 reproduces the stored campaign preds.

        Same definition, same rows, same float expression — the bar is
        1e-9 RELATIVE (mission gate); the measured worst deviation is
        tracked in ``worst`` and asserted at the end as a single
        summarising check so a failure names the actual number.
        """
        worst = 0.0
        for fam, fx in CAMPAIGN_QUAD3_FIXTURES.items():
            re = np.asarray(fx["re"], dtype=np.float64)
            cd = np.asarray(fx["cd"], dtype=np.float64)
            for re_q, want in fx["checks"]:
                out = quad3_nearest3(re, cd, float(re_q))
                assert out is not None, (fam, re_q)
                got = out[0]
                worst = max(worst, abs(got - float(want)) / abs(float(want)))
                assert got == pytest.approx(float(want), rel=1e-9, abs=1e-12), (
                    fam,
                    re_q,
                    got - float(want),
                )
        assert worst < 1e-9


# ---------------------------------------------------------------------------
# Measured-curve quad3 fallback (re_policy, campaign 2026-08-27)
# ---------------------------------------------------------------------------


def _svc_with_cd() -> DragSurrogateService:
    """TestModelBackend fixture re-wrapped with the measured-curve cache."""
    base = TestModelBackend()._service()  # noqa: SLF001 — fixture reuse
    return DragSurrogateService(
        base.backend,
        base.guard,
        corpus_cache=base.corpus_cache,
        grid=base.grid,
        cache_re=base.cache_re,
        cache_designs=base.cache_designs,
        cache_cd=_syn_corpus()["cd"],
    )


def _design_curve(hull: str, sail: float, fin: float) -> tuple[np.ndarray, np.ndarray]:
    """Cached (re, cd) rows of one design of the synthetic corpus, ascending."""
    corpus = _syn_corpus()
    sel = [
        i
        for i in range(len(corpus["cd"]))
        if corpus["hull"][i] == {"bare_hull": 0, "with_sail": 1, "full": 2}[hull]
        and corpus["sail"][i] == sail
        and corpus["fin"][i] == fin
    ]
    order = np.argsort(corpus["re"][sel])
    return corpus["re"][sel][order], corpus["cd"][sel][order]


class TestQuad3Fallback:
    """re_policy=quad3_fallback: routing, math, std semantics, default OFF."""

    def test_policy_resolver(self) -> None:
        assert resolve_re_policy() == RE_POLICY_NETWORK
        assert resolve_re_policy(None) == RE_POLICY_NETWORK
        assert resolve_re_policy("network") == RE_POLICY_NETWORK
        assert resolve_re_policy("quad3_fallback") == RE_POLICY_QUAD3_FALLBACK
        with pytest.raises(ValueError, match="unknown re_policy"):
            resolve_re_policy("quad4")

    def test_quad3_value_is_exact_polyfit_and_nearest3_selection(self) -> None:
        re6 = np.array([10.0, 20.0, 40.0, 80.0, 160.0, 320.0])
        cd6 = 50.0 * re6**-0.4
        val, sel = quad3_nearest3(re6, cd6, 500.0)  # above the sweep
        assert sel.tolist() == [80.0, 160.0, 320.0]  # top-3 edge levels
        lr = np.log10(sel)
        lc = np.log10(50.0 * sel**-0.4)
        c2, c1, c0 = np.polyfit(lr, lc, 2)
        x = np.log10(500.0)
        assert val == pytest.approx(10.0 ** (c2 * x * x + c1 * x + c0), rel=1e-12)
        _, sel_below = quad3_nearest3(re6, cd6, 5.0)
        assert sel_below.tolist() == [10.0, 20.0, 40.0]  # bottom-3
        _, sel_mid = quad3_nearest3(re6, cd6, 90.0)  # interior: 3 straddling
        assert sel_mid.tolist() == [40.0, 80.0, 160.0]
        rng = np.random.default_rng(3)
        perm = rng.permutation(re6.size)  # cache row order must not matter
        val2, _ = quad3_nearest3(re6[perm], cd6[perm], 500.0)
        assert val2 == val

    def test_quad3_validation_and_degenerate_decline(self) -> None:
        with pytest.raises(ValueError, match=">= 3 cached rows"):
            quad3_nearest3(np.array([1.0, 2.0]), np.array([1.0, 2.0]), 3.0)
        with pytest.raises(ValueError, match="matching 1-D"):
            quad3_nearest3(np.ones(5), np.ones(4), 2.0)
        with pytest.raises(ValueError, match="positive"):
            quad3_nearest3(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), -1.0)
        dup_re = np.array([10.0, 10.0, 20.0, 40.0])
        dup_cd = np.array([5.0, 5.1, 3.0, 2.0])
        # nearest triple to log10(5) is {10, 10, 20}: duplicate log10 Re.
        assert quad3_nearest3(dup_re, dup_cd, 5.0) is None
        assert quad3_loo_std(dup_re, dup_cd) is None  # a pseudo-fit is degenerate

    def test_loo_std_semantics(self) -> None:
        re4 = np.array([50.0, 64.0, 81.0, 100.0])
        cd4 = 20.0 * (re4 / 50.0) ** -0.45
        assert quad3_loo_std(re4[:3], cd4[:3]) is None  # fewer than 4 rows
        x = np.log10(np.array([10.0, 20.0, 40.0, 80.0, 160.0, 320.0]))
        re6 = 10.0**x
        quad = 10.0 ** (0.01 * x * x - 0.5 * x + 1.0)  # exact log-log quadratic
        assert quad3_loo_std(re6, quad) < 1e-12  # quad3 fits it exactly
        cubic = 10.0 ** (0.002 * x**3 - 0.5 * x + 1.0)  # curvature quad3 misses
        assert quad3_loo_std(re6, cubic) > 1e-4

    def test_outside_window_exact_design_serves_quad3(self) -> None:
        svc = _svc_with_cd()
        res = svc.predict("full", 1.0, 1.0, [300.0], re_policy=RE_POLICY_QUAD3_FALLBACK)
        curve_re, curve_cd = _design_curve("full", 1.0, 1.0)
        val, chosen = quad3_nearest3(curve_re, curve_cd, 300.0)
        assert chosen.tolist() == [64.0, 81.0, 100.0]
        assert res.cd[0] == pytest.approx(val, rel=1e-12)
        pol = res.info["re_policy"]
        assert pol["method"] == "quad3_fallback"
        assert pol["n_quad3_points"] == 1
        assert pol["n_cached_rows"] == 4
        assert pol["nearest_cached_re"][0] == [64.0, 81.0, 100.0]
        assert pol["quad3_mask"] == [True]
        loo = quad3_loo_std(curve_re, curve_cd)
        assert res.std[0] == pytest.approx(val * loo, rel=1e-12)
        assert pol["std_source"] == "quad3_loo_relative_rms_times_cd"
        assert np.isnan(res.lo[0]) and np.isnan(res.hi[0])  # no member band
        # Conservative interaction: out-of-window Re still guard-REJECTS.
        assert res.guard.flag == "reject"
        assert any("log10_re" in r for r in res.guard.reasons)

    def test_below_window_and_boundary_rule(self) -> None:
        svc = _svc_with_cd()
        # The window is the corpus window [50, 100]; boundary values are
        # INSIDE (network), strictly-below/above values are outside.
        below = svc.predict("full", 1.0, 1.0, [49.0], re_policy=RE_POLICY_QUAD3_FALLBACK)
        curve_re, curve_cd = _design_curve("full", 1.0, 1.0)
        val, chosen = quad3_nearest3(curve_re, curve_cd, 49.0)
        assert chosen.tolist() == [50.0, 64.0, 81.0]
        assert below.cd[0] == pytest.approx(val, rel=1e-12)
        assert below.info["re_policy"]["corpus_re_window"] == [50.0, 100.0]
        for edge in (50.0, 100.0):
            at = svc.predict("full", 1.0, 1.0, [edge], re_policy=RE_POLICY_QUAD3_FALLBACK)
            assert at.info["re_policy"]["method"] == "network"
            assert at.info["re_policy"]["reason"] == "re_inside_corpus_window"

    def test_in_window_keeps_network_bit_identical(self) -> None:
        svc = _svc_with_cd()
        grid = np.array([50.0, 64.0, 73.0, 100.0])
        off = svc.predict("full", 1.0, 1.0, grid)
        on = svc.predict("full", 1.0, 1.0, grid, re_policy=RE_POLICY_QUAD3_FALLBACK)
        np.testing.assert_array_equal(on.cd, off.cd)
        np.testing.assert_array_equal(on.std, off.std)
        np.testing.assert_array_equal(on.lo, off.lo)
        np.testing.assert_array_equal(on.hi, off.hi)
        assert on.info["re_policy"]["n_quad3_points"] == 0

    def test_mixed_grid_applies_per_point(self) -> None:
        svc = _svc_with_cd()
        grid = np.array([64.0, 300.0, 41.0])
        off = svc.predict("full", 1.0, 1.0, grid)
        on = svc.predict("full", 1.0, 1.0, grid, re_policy=RE_POLICY_QUAD3_FALLBACK)
        assert on.info["re_policy"]["quad3_mask"] == [False, True, True]
        assert on.cd[0] == off.cd[0]  # in-window point bit-identical
        curve_re, curve_cd = _design_curve("full", 1.0, 1.0)
        for i, re_q in ((1, 300.0), (2, 41.0)):
            val, _ = quad3_nearest3(curve_re, curve_cd, re_q)
            assert on.cd[i] == pytest.approx(val, rel=1e-12)
        assert np.isnan(on.lo[1]) and np.isnan(on.hi[2])
        assert np.isfinite(on.lo[0]) and np.isfinite(on.hi[0])

    def test_no_exact_design_match_keeps_network(self) -> None:
        svc = _svc_with_cd()
        fields = np.zeros((5, TEST_GRID.ny, TEST_GRID.nx), dtype=np.float32)
        res = svc.predict(
            "full", 2.5, 2.5, [300.0], fields=fields, re_policy=RE_POLICY_QUAD3_FALLBACK
        )
        pol = res.info["re_policy"]
        assert pol["method"] == "network"
        assert pol["reason"] == "no_exact_design_match"
        ref = svc.predict("full", 2.5, 2.5, [300.0], fields=fields)
        np.testing.assert_array_equal(res.cd, ref.cd)

    def test_insufficient_cached_rows_declines(self) -> None:
        svc = _svc_with_cd()
        # bare_hull 1.0/1.0 has only 2 cached rows (64, 100).
        res = svc.predict("bare_hull", 1.0, 1.0, [300.0], re_policy=RE_POLICY_QUAD3_FALLBACK)
        pol = res.info["re_policy"]
        assert pol["method"] == "network"
        assert pol["reason"] == "insufficient_cached_rows"
        assert pol["n_cached_rows"] == 2
        ref = svc.predict("bare_hull", 1.0, 1.0, [300.0])
        np.testing.assert_array_equal(res.cd, ref.cd)

    def test_service_without_measured_cache_declines(self, tmp_path: Path) -> None:
        svc = TestModelBackend()._service()  # noqa: SLF001 — fixture reuse (no cache_cd)
        res = svc.predict("full", 1.0, 1.0, [300.0], re_policy=RE_POLICY_QUAD3_FALLBACK)
        assert res.info["re_policy"]["reason"] == "no_measured_curve_cache"
        ref = svc.predict("full", 1.0, 1.0, [300.0])
        np.testing.assert_array_equal(res.cd, ref.cd)
        # The replay/run-dir service has no measured cache either.
        rep = DragSurrogateService.from_run_dir(_write_syn_run_dir(tmp_path))
        got = rep.predict("full", 1.0, 1.0, [300.0], re_policy=RE_POLICY_QUAD3_FALLBACK)
        assert got.info["re_policy"]["reason"] == "no_measured_curve_cache"
        np.testing.assert_array_equal(got.cd, rep.predict("full", 1.0, 1.0, [300.0]).cd)

    def test_default_off_is_byte_identical(self) -> None:
        svc = _svc_with_cd()
        no_cd = TestModelBackend()._service()  # noqa: SLF001 — fixture reuse
        grid = np.array([64.0, 100.0, 300.0, 41.0])  # includes out-of-window points
        a = svc.predict("full", 1.0, 1.0, grid)
        b = svc.predict("full", 1.0, 1.0, grid, re_policy=RE_POLICY_NETWORK)
        c = no_cd.predict("full", 1.0, 1.0, grid)
        for arr in ("cd", "std", "lo", "hi"):
            assert getattr(a, arr).tobytes() == getattr(b, arr).tobytes()
            assert getattr(a, arr).tobytes() == getattr(c, arr).tobytes()
        assert "re_policy" not in a.info
        assert "re_policy" not in b.info
        assert a.guard == b.guard == c.guard
        assert a.info == c.info

    def test_policy_is_deterministic(self) -> None:
        svc = _svc_with_cd()
        grid = np.array([300.0, 41.0, 64.0])
        a = svc.predict("full", 1.0, 1.0, grid, re_policy=RE_POLICY_QUAD3_FALLBACK)
        b = svc.predict("full", 1.0, 1.0, grid, re_policy=RE_POLICY_QUAD3_FALLBACK)
        for arr in ("cd", "std", "lo", "hi"):
            assert getattr(a, arr).tobytes() == getattr(b, arr).tobytes()
        assert a.info["re_policy"] == b.info["re_policy"]

    def test_unknown_policy_name_raises(self) -> None:
        svc = _svc_with_cd()
        with pytest.raises(ValueError, match="unknown re_policy"):
            svc.predict("full", 1.0, 1.0, [64.0], re_policy="quad4")

    def test_misaligned_cache_cd_raises(self) -> None:
        base = TestModelBackend()._service()  # noqa: SLF001 — fixture reuse
        corpus = _syn_corpus()
        with pytest.raises(ValueError, match="row-aligned"):
            DragSurrogateService(
                base.backend,
                base.guard,
                corpus_cache=base.corpus_cache,
                grid=base.grid,
                cache_re=corpus["re"],
                cache_designs=base.cache_designs,
                cache_cd=corpus["cd"][:3],
            )

    def test_load_corpus_index_carries_cd(self, tmp_path: Path) -> None:
        idx = load_corpus_index(_write_syn_run_dir(tmp_path))
        assert idx.cd is not None
        np.testing.assert_allclose(idx.cd, _syn_corpus()["cd"], rtol=1e-12)
