"""Tests for ``tensorlbm.ai.active_learning_grad`` (B4-P3c) — synthetic, CPU-only.

No /nfs paths and (except one explicitly skipped CUDA smoke) no GPU: the
corpus, the ensemble checkpoints and the acquisition queries are
fabricated exactly the way ``tests/test_active_learning.py`` fabricates
them.  The load-bearing contracts pinned here:

- ``batch_drag_gradients`` reproduces point-wise ``drag_gradients``
  (STE and soft paths) to float round-off, with bitwise-equal geometry
  channels, and is deterministic for identical arguments;
- ``propose_acquisition_grad`` delegates the three #243 strategies
  BITWISE (identical key lists) and its ``gradient`` strategy draws
  from exactly the coverage pool;
- the honest-axis gate excludes sign-unstable / dead axes and reports
  them instead of averaging them in;
- the trend-margin helpers are exact on analytic slopes and the
  calibration refuses unstable / sign-flipped inputs.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from tensorlbm.ai.active_learning import (
    HULLFORM_AXES,
    AcquisitionPoint,
    FlaggedQuery,
    corpus_cond_v3,
    fit_stats,
    hullform_condition_rows,
    propose_acquisition,
)
from tensorlbm.ai.active_learning_grad import (
    GRADIENT_STRATEGY_NAMES,
    GradBatch,
    apply_trend_calibration,
    axis_stability,
    batch_drag_gradients,
    calibrate_trend_margin,
    gradient_scores,
    margin_ratio,
    propose_acquisition_grad,
    trend_slopes,
)
from tensorlbm.ai.drag_cond import (
    CondFNODrag,
    SuboffGrid,
    geometry_channels,
    suboff_geometry_features,
)
from tensorlbm.ai.inference_service import (
    CondDragCheckpoint,
    EnvelopeMahalanobisGuardrail,
    save_checkpoint,
)
from tensorlbm.diff_voxelize import DIFF_PARAM_NAMES, drag_gradients

# resolution 64 -> (nz, ny, nx) = (32, 32, 64): the smallest grid that fits
# the tiny test ARCH modes (4, 8) (rfft2 keeps nx // 2 + 1 = 33 x-modes).
TEST_GRID = SuboffGrid.from_resolution(64)
DESIGNS = [
    ("bare_hull", 1.0, 1.0),
    ("with_sail", 1.0, 1.0),
    ("with_sail", 1.3, 1.0),
    ("full", 1.0, 1.0),
    ("full", 0.7, 1.2),
]
RES = np.array([80.0, 120.0, 200.0, 300.0, 450.0, 600.0])
HULLS = ("bare_hull", "with_sail", "full")
ARCH: dict[str, Any] = dict(in_ch=5, width=4, n_layers=2, modes=(4, 8), cond_dim=8, aux_dim=0)


def _design_geo(hull: str, sail: float, fin: float) -> np.ndarray:
    return np.asarray(geometry_channels(suboff_geometry_features(hull, sail, fin, grid=TEST_GRID)))


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
        cd=20.0 * np.array([r for _h, _s, _f, r, _d in rows]) ** -0.42,
        geo=np.stack([_design_geo(h, s, f) for h, s, f, _r, _d in rows]),
        aux=rng.normal(0.5, 0.05, (n, 8)),
        mask_bit_eq=np.zeros(n, dtype=bool),
    )


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
        net = CondFNODrag(**ARCH)
        ckpt = CondDragCheckpoint(
            arch=ARCH,
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


def _family_params(shape: dict[str, float]) -> dict[str, Any]:
    params: dict[str, Any] = {"hull_type": "with_sail", "sail_scale": 1.0, "fin_scale": 1.0}
    params.update({a: 1.0 for a in HULLFORM_AXES})
    params.update(shape)
    return params


def flagged_queries(existing_cond: np.ndarray) -> list[FlaggedQuery]:
    """Flagged queries with real guard scores (as the service logs them)."""
    shapes = [
        {"l_over_d_mult": 0.75},
        {"l_over_d_mult": 1.30},
        {"nose_len_mult": 1.30},
        {"sail_x_mult": 1.30},
    ]
    guard = EnvelopeMahalanobisGuardrail(existing_cond)
    out: list[FlaggedQuery] = []
    for i, shape in enumerate(shapes):
        params = _family_params(shape)
        for re in (RES[i % 4], RES[(i + 2) % 4]):
            cond = hullform_condition_rows(params, [float(re)], grid=TEST_GRID)
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


VARIED_DESIGNS: list[dict[str, float]] = [
    {"l_over_d_mult": 0.75, "sail_scale": 1.0, "fin_scale": 1.0},
    {"l_over_d_mult": 1.30, "nose_len_mult": 1.0, "sail_x_mult": 1.0},
    {"nose_len_mult": 1.30},
    {"sail_scale": 1.2, "l_over_d_mult": 1.15, "stern_len_mult": 1.1},
]
DESIGN_RES = [110.0, 210.0, 330.0, 520.0]


def _field_row(corpus: dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray(corpus["x"][1], dtype=np.float32)


# ---------------------------------------------------------------------------
# Batched gradient engine
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def ckpts(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, np.ndarray]]:
    corpus = make_corpus()
    d = write_fake_ckpt_dir(tmp_path_factory.mktemp("ag"), corpus, "grad")
    return d, corpus


class TestBatchDragGradients:
    def _pointwise(
        self, ckpt_dir: Path, corpus: dict[str, np.ndarray]
    ) -> tuple[list[dict[str, Any]], GradBatch]:
        paths = sorted(str(p) for p in ckpt_dir.glob("*.pt"))
        batch = batch_drag_gradients(
            VARIED_DESIGNS,
            paths,
            _field_row(corpus),
            re=DESIGN_RES,
            u_in=0.1,
            grid=TEST_GRID,
            hull_type="with_sail",
            device="cpu",
            chunk=3,
        )
        singles = [
            drag_gradients(
                d,
                paths,
                _field_row(corpus),
                re=float(re),
                u_in=0.1,
                grid=TEST_GRID,
                hull_type="with_sail",
                device="cpu",
            )
            for d, re in zip(VARIED_DESIGNS, DESIGN_RES)
        ]
        return singles, batch

    def test_parity_with_pointwise_ste(self, ckpts: tuple[Path, dict[str, np.ndarray]]) -> None:
        singles, batch = self._pointwise(*ckpts)
        assert batch.grads.shape == (len(VARIED_DESIGNS), len(DIFF_PARAM_NAMES))
        assert batch.member_grads.shape == (2, len(VARIED_DESIGNS), len(DIFF_PARAM_NAMES))
        for i, g in enumerate(singles):
            idx = [j for j, a in enumerate(batch.axis_names) if a in g["grads"]]
            ref = np.array([g["grads"][batch.axis_names[j]] for j in idx])
            np.testing.assert_allclose(batch.grads[i, idx], ref, rtol=1e-6, atol=1e-9)
            mref = np.array(
                [[g["member_grads"][batch.axis_names[j]][m] for j in idx] for m in range(2)]
            )
            np.testing.assert_allclose(batch.member_grads[:, i, idx], mref, rtol=1e-5, atol=1e-9)
            np.testing.assert_allclose(batch.log10_cd[i], g["log10_cd"], rtol=0, atol=1e-8)
            # geometry channels are bitwise equal (hard STE forward)
            np.testing.assert_array_equal(batch.channels[i], np.array(list(g["channels"].values())))

    def test_parity_with_pointwise_soft(self, ckpts: tuple[Path, dict[str, np.ndarray]]) -> None:
        paths = sorted(str(p) for p in ckpts[0].glob("*.pt"))
        corpus = ckpts[1]
        batch = batch_drag_gradients(
            VARIED_DESIGNS[:2],
            paths,
            _field_row(corpus),
            re=DESIGN_RES[:2],
            grid=TEST_GRID,
            hull_type="with_sail",
            ste=False,
            device="cpu",
            chunk=2,
        )
        for i, d in enumerate(VARIED_DESIGNS[:2]):
            g = drag_gradients(
                d,
                paths,
                _field_row(corpus),
                re=float(DESIGN_RES[i]),
                grid=TEST_GRID,
                hull_type="with_sail",
                ste=False,
                device="cpu",
            )
            idx = [j for j, a in enumerate(batch.axis_names) if a in g["grads"]]
            ref = np.array([g["grads"][batch.axis_names[j]] for j in idx])
            np.testing.assert_allclose(batch.grads[i, idx], ref, rtol=1e-6, atol=1e-9)

    def test_chunking_allclose_and_determinism(
        self, ckpts: tuple[Path, dict[str, np.ndarray]]
    ) -> None:
        paths = sorted(str(p) for p in ckpts[0].glob("*.pt"))
        corpus = ckpts[1]
        a = batch_drag_gradients(
            VARIED_DESIGNS,
            paths,
            _field_row(corpus),
            re=DESIGN_RES,
            grid=TEST_GRID,
            hull_type="with_sail",
            chunk=1,
        )
        b = batch_drag_gradients(
            VARIED_DESIGNS,
            paths,
            _field_row(corpus),
            re=DESIGN_RES,
            grid=TEST_GRID,
            hull_type="with_sail",
            chunk=4,
        )
        c = batch_drag_gradients(
            VARIED_DESIGNS,
            paths,
            _field_row(corpus),
            re=DESIGN_RES,
            grid=TEST_GRID,
            hull_type="with_sail",
            chunk=4,
        )
        np.testing.assert_allclose(a.grads, b.grads, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(a.channels, b.channels, rtol=1e-12, atol=0)
        assert np.array_equal(b.grads, c.grads)  # same args -> bitwise
        assert np.array_equal(b.member_grads, c.member_grads)
        assert b.point_keys == c.point_keys

    def test_throughput_smoke(self, ckpts: tuple[Path, dict[str, np.ndarray]]) -> None:
        """One batched call is not slower than B single-design calls.

        A smoke, not a benchmark: the margin is deliberately loose (CI
        timing jitter); the real throughput table lives in the run
        report.  Also re-checks the chunked result equals the unchunked.
        """
        paths = sorted(str(p) for p in ckpts[0].glob("*.pt"))
        corpus = ckpts[1]
        designs = [
            {"l_over_d_mult": 0.8 + 0.05 * i, "nose_len_mult": 1.0, "sail_x_mult": 1.0}
            for i in range(5)
        ]
        res = [150.0 + 20.0 * i for i in range(5)]
        t0 = time.perf_counter()
        single = [
            batch_drag_gradients(
                [d],
                paths,
                _field_row(corpus),
                re=float(r),
                grid=TEST_GRID,
                hull_type="with_sail",
                chunk=1,
            )
            for d, r in zip(designs, res)
        ]
        t_single = time.perf_counter() - t0
        t0 = time.perf_counter()
        batched = batch_drag_gradients(
            designs,
            paths,
            _field_row(corpus),
            re=res,
            grid=TEST_GRID,
            hull_type="with_sail",
            chunk=5,
        )
        t_batch = time.perf_counter() - t0
        np.testing.assert_allclose(
            batched.grads, np.stack([s.grads[0] for s in single]), rtol=1e-4, atol=1e-8
        )
        assert t_batch <= 3.0 * t_single + 1.0, (t_batch, t_single)
        print(f"\nthroughput smoke: 5 singles {t_single:.2f}s vs 1 batch {t_batch:.2f}s")

    def test_input_validation(self, ckpts: tuple[Path, dict[str, np.ndarray]]) -> None:
        paths = sorted(str(p) for p in ckpts[0].glob("*.pt"))
        corpus = ckpts[1]
        with pytest.raises(ValueError, match="non-empty"):
            batch_drag_gradients([], paths, _field_row(corpus))
        with pytest.raises(ValueError, match="length-3"):
            batch_drag_gradients(VARIED_DESIGNS[:3], paths, _field_row(corpus), re=[1.0, 2.0])
        with pytest.raises(ValueError, match="positive"):
            batch_drag_gradients(VARIED_DESIGNS[:1], paths, _field_row(corpus), re=-5.0)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
    def test_cuda_smoke(self, ckpts: tuple[Path, dict[str, np.ndarray]]) -> None:
        paths = sorted(str(p) for p in ckpts[0].glob("*.pt"))
        corpus = ckpts[1]
        gpu = batch_drag_gradients(
            VARIED_DESIGNS[:2],
            paths,
            _field_row(corpus),
            re=DESIGN_RES[:2],
            grid=TEST_GRID,
            hull_type="with_sail",
            device="cuda:0",
            chunk=2,
        )
        cpu = batch_drag_gradients(
            VARIED_DESIGNS[:2],
            paths,
            _field_row(corpus),
            re=DESIGN_RES[:2],
            grid=TEST_GRID,
            hull_type="with_sail",
            device="cpu",
            chunk=2,
        )
        np.testing.assert_allclose(gpu.channels, cpu.channels, rtol=1e-12, atol=1e-12)
        # smoke, not precision: float32 kernel layout differs between
        # devices (measured ~3e-3 rel on the production grid); only the
        # order of magnitude and sign must survive the device move
        np.testing.assert_allclose(gpu.grads, cpu.grads, rtol=5e-3, atol=1e-4)
        np.testing.assert_allclose(gpu.log10_cd, cpu.log10_cd, rtol=0, atol=1e-4)


# ---------------------------------------------------------------------------
# Acquisition: gradient strategy
# ---------------------------------------------------------------------------
def _synthetic_grad_batch(
    points: list[AcquisitionPoint],
    ens: np.ndarray,
    member_scale: float = 0.2,
    seed: int = 0,
) -> GradBatch:
    """GradBatch from crafted ensemble gradients + noisy members."""
    rng = np.random.default_rng(seed)
    n = len(points)
    members = []
    for _ in range(5):
        noisy = ens + member_scale * np.abs(ens) * rng.normal(size=ens.shape)
        members.append(noisy)
    return GradBatch(
        point_keys=tuple(p.key for p in points),
        axis_names=tuple(DIFF_PARAM_NAMES),
        grads=ens,
        member_grads=np.stack(members),
        member_labels=tuple(f"m{i}" for i in range(5)),
        log10_cd=np.zeros(n),
        channels=np.zeros((n, 4)),
    )


@pytest.fixture(scope="session")
def setup() -> dict[str, Any]:
    corpus = make_corpus()
    existing = np.asarray(corpus_cond_v3(corpus))
    queries = flagged_queries(existing)
    pool = propose_acquisition(
        queries,
        strategy="coverage",
        budget=4096,
        existing_cond=existing,
        grid=TEST_GRID,
        seed=0,
    )
    return {"corpus": corpus, "existing": existing, "queries": queries, "pool": pool}


class TestGradientStrategy:
    def _grad_fn(self, points: list[AcquisitionPoint]) -> GradBatch:
        """Steep l_over_d candidates get huge gradients on every axis."""
        n = len(points)
        ens = np.full((n, len(DIFF_PARAM_NAMES)), 0.01)
        for i, p in enumerate(points):
            steep = abs(float(p.params.get("l_over_d_mult", 1.0)) - 1.0)
            ens[i] *= 1.0 + 50.0 * steep
        return _synthetic_grad_batch(points, ens)

    def test_strategy_names_and_validation(self, setup: dict[str, Any]) -> None:
        assert GRADIENT_STRATEGY_NAMES == (
            "envelope_shell",
            "max_disagreement",
            "coverage",
            "gradient",
        )
        with pytest.raises(ValueError, match="strategy must be one of"):
            propose_acquisition_grad(
                setup["queries"], strategy="banana", budget=4, existing_cond=setup["existing"]
            )
        with pytest.raises(ValueError, match="requires grad_fn"):
            propose_acquisition_grad(
                setup["queries"],
                strategy="gradient",
                budget=4,
                existing_cond=setup["existing"],
                grid=TEST_GRID,
            )

    def test_gradient_pool_is_coverage_pool_and_deterministic(self, setup: dict[str, Any]) -> None:
        out = propose_acquisition_grad(
            setup["queries"],
            strategy="gradient",
            budget=3,
            existing_cond=setup["existing"],
            grad_fn=self._grad_fn,
            grid=TEST_GRID,
            seed=0,
        )
        pool_keys = {p.key for p in setup["pool"]}
        assert 0 < len(out) <= 3
        assert all(p.strategy == "gradient" for p in out)
        assert {p.key for p in out} <= pool_keys
        again = propose_acquisition_grad(
            setup["queries"],
            strategy="gradient",
            budget=3,
            existing_cond=setup["existing"],
            grad_fn=self._grad_fn,
            grid=TEST_GRID,
            seed=0,
        )
        assert [p.key for p in out] == [p.key for p in again]

    def test_gradient_prefers_steep_candidates(self, setup: dict[str, Any]) -> None:
        out = propose_acquisition_grad(
            setup["queries"],
            strategy="gradient",
            budget=1,
            existing_cond=setup["existing"],
            grad_fn=self._grad_fn,
            grid=TEST_GRID,
            seed=0,
        )
        assert len(out) == 1
        assert abs(float(out[0].params["l_over_d_mult"]) - 1.0) > 0.2

    def test_budget_at_least_pool_returns_pool(self, setup: dict[str, Any]) -> None:
        pool_len = len(setup["pool"])
        out = propose_acquisition_grad(
            setup["queries"],
            strategy="gradient",
            budget=pool_len + 5,
            existing_cond=setup["existing"],
            grad_fn=self._grad_fn,
            grid=TEST_GRID,
            seed=0,
        )
        assert len(out) == pool_len
        assert [p.key for p in out] == [p.key for p in setup["pool"]]

    def test_delegation_bitwise(self, setup: dict[str, Any]) -> None:
        """The three #243 strategies come out bitwise identical."""
        existing, queries = setup["existing"], setup["queries"]

        def std_fn(pts: list[AcquisitionPoint]) -> np.ndarray:
            return np.array([0.1 + 0.8 * ((i * 7) % 11) / 11.0 for i in range(len(pts))])

        for strategy, kwargs in (
            ("envelope_shell", {"n_candidates": 64}),
            ("max_disagreement", {"member_std_fn": std_fn}),
            ("coverage", {}),
        ):
            mine = propose_acquisition_grad(
                queries,
                strategy=strategy,
                budget=5,
                existing_cond=existing,
                grid=TEST_GRID,
                seed=11,
                **kwargs,
            )
            ref = propose_acquisition(
                queries,
                strategy=strategy,
                budget=5,
                existing_cond=existing,
                grid=TEST_GRID,
                seed=11,
                **kwargs,
            )
            assert mine == ref, strategy  # dataclass equality incl. params dict
            assert [p.strategy for p in mine] == [strategy] * len(ref)

    def test_w_coverage_endpoints(self, setup: dict[str, Any]) -> None:
        """w=0 ranks by gradient, w=1 purely by channel-space Mahalanobis."""
        pure = propose_acquisition_grad(
            setup["queries"],
            strategy="gradient",
            budget=1,
            existing_cond=setup["existing"],
            grad_fn=self._grad_fn,
            grid=TEST_GRID,
        )
        cov = propose_acquisition_grad(
            setup["queries"],
            strategy="gradient",
            budget=1,
            existing_cond=setup["existing"],
            grad_fn=self._grad_fn,
            grid=TEST_GRID,
            w_coverage=1.0,
        )
        table = gradient_scores(
            setup["pool"],
            self._grad_fn(setup["pool"]),
            setup["existing"],
            w_coverage=1.0,
            grid=TEST_GRID,
        )
        best_cov = max(range(len(setup["pool"])), key=lambda i: table.scores_final[i])
        assert cov[0].key == setup["pool"][best_cov].key
        # the synthetic gradient mass is on l_over_d corners; the pure
        # gradient pick is not decided by the Mahalanobis term
        assert abs(float(pure[0].params.get("l_over_d_mult", 1.0)) - 1.0) > 0.2

    def test_axis_exclusion_reported(self, setup: dict[str, Any]) -> None:
        """Sign-unstable and dead axes are excluded, not averaged in."""
        n = len(setup["pool"])
        ens = np.full((n, len(DIFF_PARAM_NAMES)), 0.05)
        members = []
        for m in range(5):
            row = ens.copy()
            if m % 2 == 0:  # half the members flip the fin axis sign
                row[:, DIFF_PARAM_NAMES.index("fin_scale")] *= -1.0
            members.append(row)
        ens[:, DIFF_PARAM_NAMES.index("stern_len_mult")] = 0.0  # dead axis
        batch = GradBatch(
            point_keys=tuple(p.key for p in setup["pool"]),
            axis_names=tuple(DIFF_PARAM_NAMES),
            grads=ens,
            member_grads=np.stack(members),
            member_labels=tuple(f"m{i}" for i in range(5)),
            log10_cd=np.zeros(n),
            channels=np.zeros((n, 4)),
        )
        table = gradient_scores(setup["pool"], batch, setup["existing"], grid=TEST_GRID)
        assert "fin_scale" in table.axes_excluded
        assert "sign-unstable" in table.exclusion_reasons["fin_scale"]
        assert "stern_len_mult" in table.axes_excluded
        assert "dead axis" in table.exclusion_reasons["stern_len_mult"]
        assert "fin_scale" not in table.axes_used
        assert "stern_len_mult" not in table.axes_used
        assert not table.fallback_coverage_order
        assert set(table.axes_used) | set(table.axes_excluded) == set(DIFF_PARAM_NAMES)

    def test_all_axes_dead_falls_back(self, setup: dict[str, Any]) -> None:
        n = len(setup["pool"])
        zeros = np.zeros((n, len(DIFF_PARAM_NAMES)))
        batch = _synthetic_grad_batch(setup["pool"], zeros)
        table = gradient_scores(setup["pool"], batch, setup["existing"], grid=TEST_GRID)
        assert table.fallback_coverage_order
        assert not table.axes_used
        out = propose_acquisition_grad(
            setup["queries"],
            strategy="gradient",
            budget=4,
            existing_cond=setup["existing"],
            grad_fn=lambda pts: _synthetic_grad_batch(
                pts, np.zeros((len(pts), len(DIFF_PARAM_NAMES)))
            ),
            grid=TEST_GRID,
        )
        assert [p.key for p in out] == [p.key for p in setup["pool"][:4]]

    def test_axis_stability_numbers(self) -> None:
        ens = np.array([[1.0, 0.0], [1.0, 0.0]])
        members = np.stack(
            [
                np.full((2, 2), 1.0),  # agrees everywhere
                np.full((2, 2), 1.0),
                np.full((2, 2), 1.0),
                np.full((2, 2), 1.0),
                np.array([[-1.0, 0.0], [-1.0, 0.0]]),  # disagrees everywhere
            ]
        )
        batch = GradBatch(
            point_keys=("a", "b"),
            axis_names=("x", "y"),
            grads=ens,
            member_grads=members,
            member_labels=("m0", "m1", "m2", "m3", "m4"),
            log10_cd=np.zeros(2),
            channels=np.zeros((2, 4)),
        )
        stability, magnitude = axis_stability(batch)
        np.testing.assert_allclose(stability, [0.8, 0.0])
        np.testing.assert_allclose(magnitude, [1.0, 0.0])


# ---------------------------------------------------------------------------
# Trend-margin calibration
# ---------------------------------------------------------------------------
class TestTrendMargin:
    def test_trend_slopes_exact(self) -> None:
        v = np.array([0.75, 1.0, 1.3])
        slope_true = 0.5
        cd = 10.0 ** (slope_true * v[:, None] + 2.0) * np.ones((3, 3))
        np.testing.assert_allclose(trend_slopes(v, cd), slope_true, rtol=1e-12)
        # two-point sweep degenerates to the difference quotient
        cd2 = 10.0 ** (0.7 * np.array([1.0, 1.3])[:, None] + 2.0)
        np.testing.assert_allclose(trend_slopes([1.0, 1.3], cd2), 0.7, rtol=1e-12)

    def test_trend_slopes_validation(self) -> None:
        with pytest.raises(ValueError, match="n_values=3"):
            trend_slopes([0.75, 1.0, 1.3], np.ones((2, 3)))
        with pytest.raises(ValueError, match="constant"):
            trend_slopes([1.0, 1.0, 1.0], np.ones((3, 2)))
        with pytest.raises(ValueError, match="positive"):
            trend_slopes([1.0, 1.3], -np.ones((2, 2)))
        with pytest.raises(ValueError, match="finite"):
            trend_slopes([1.0, 1.3], np.array([[np.nan, 1.0], [1.0, 1.0]]))

    def test_margin_ratio_damped_and_flipped(self) -> None:
        v = [0.75, 1.0, 1.3]
        truth = 10.0 ** (0.5 * np.asarray(v)[:, None] + 2.0)
        damped = 10.0 ** (0.19 * np.asarray(v)[:, None] + 2.0)
        stat = margin_ratio("l_over_d_mult", v, truth, damped)
        np.testing.assert_allclose(stat.ratios, 0.38, rtol=1e-9)
        assert stat.sign_agree
        flipped = 10.0 ** (-0.2 * np.asarray(v)[:, None] + 2.0)
        stat2 = margin_ratio("l_over_d_mult", v, truth, flipped)
        assert stat2.median_ratio < 0.0
        assert not stat2.sign_agree

    def test_calibrate_stable_axis(self) -> None:
        cal = calibrate_trend_margin({"l_over_d_mult": [0.36, 0.38, 0.40, 0.37]})
        assert cal.calibratable["l_over_d_mult"]
        np.testing.assert_allclose(cal.scales["l_over_d_mult"], 1.0 / 0.375, rtol=1e-9)
        np.testing.assert_allclose(cal.fit_dispersion["l_over_d_mult"], 0.0175 / 0.375, rtol=1e-9)
        np.testing.assert_allclose(cal.apply("l_over_d_mult", 0.375), 1.0, rtol=1e-9)
        with pytest.raises(KeyError):
            cal.apply("sail_x_mult", 1.0)

    def test_calibrate_refuses_unstable(self) -> None:
        mixed_sign = calibrate_trend_margin({"a": [0.4, -0.3, 0.5]})
        assert not mixed_sign.calibratable["a"]
        assert "sign" in mixed_sign.notes["a"]
        tight = calibrate_trend_margin({"a": [0.38, 0.39, 0.38]})
        assert tight.calibratable["a"]
        wide = calibrate_trend_margin({"a": [0.1, 0.5, 0.9, 0.3]})
        assert not wide.calibratable["a"]
        assert "dispersion" in wide.notes["a"]
        few = calibrate_trend_margin({"a": [0.4, 0.4]})
        assert not few.calibratable["a"]
        assert "samples" in few.notes["a"]
        empty = calibrate_trend_margin({"a": []})
        assert not empty.calibratable["a"]
        assert empty.notes["a"] == "no finite ratios"

    def test_apply_trend_calibration_band(self) -> None:
        cal = calibrate_trend_margin({"l_over_d_mult": [0.36, 0.38, 0.40, 0.37]})
        out = apply_trend_calibration(cal, {"l_over_d_mult": [0.35, 0.40, 0.37]})
        assert out["l_over_d_mult"]["calibratable"]
        np.testing.assert_allclose(
            out["l_over_d_mult"]["calibrated_median"], 0.37 / 0.375, rtol=1e-9
        )
        assert out["l_over_d_mult"]["in_band_fraction"] == 1.0
        bad = calibrate_trend_margin({"nose_len_mult": [0.4, -0.3]})
        out2 = apply_trend_calibration(bad, {"nose_len_mult": [0.5, 0.6]})
        assert not out2["nose_len_mult"]["calibratable"]
        # uncalibratable axes keep their raw ratios (no silent scaling)
        np.testing.assert_allclose(out2["nose_len_mult"]["calibrated_ratios"], [0.5, 0.6])
