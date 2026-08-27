"""Service-backend factory + protocol tests (TRT echo slice, 2026-08-27).

Three tiers, mirroring ``tests/test_deploy_latency.py``:

- pure tier (every venv, no optional deps): ``resolve_backend_kind``
  resolution order, ``make_backend`` error paths and the degrade-to-torch
  fallback, the :class:`FusedGraphBackend` service protocol against a fake
  runtime (shapes, dtypes, validation messages, chunked row order), and the
  default-path invariance of ``DragSurrogateService.from_checkpoints``;
- ONNX tier (skipped without onnx + onnxruntime): a tiny fused-ensemble
  artifact exported from two random members must serve through
  ``make_backend("onnx")`` at the torch parity floor;
- TensorRT tier (skipped without tensorrt + cuda-python + a CUDA device, or
  without the PR #249 server artifacts): the real ``b4_serve`` plans behind
  ``make_backend("trt")`` must match the torch backend per member at the
  fp32-strict / f16 parity floors measured in PR #249, and the three
  backends must produce identical guard verdicts for the same designs.

Server commands (GPU 2)::

    # pure tier (torch venv or ci-cpu venv)
    PYTHONPATH=src /nfs/wangxi/venvs/tensorlbm/bin/python -m pytest \\
        tests/test_service_backends.py -q --basetemp=/nfs/wangxi/tmp/pt_te

    # TRT + real-artifact tiers (deploy venv borrowing torch + LD_LIBRARY_PATH)
    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src:/nfs/wangxi/venvs/tensorlbm/lib/python3.12/site-packages \\
    LD_LIBRARY_PATH=/nfs/wangxi/venvs/tensorlbm/lib/python3.12/site-packages/nvidia/cudnn/lib \\
        /nfs/wangxi/venvs/deploy/bin/python -m pytest tests/test_service_backends.py \\
        -q --basetemp=/nfs/wangxi/tmp/pt_te
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tensorlbm.ai.drag_cond import (
    CondFNODrag,
    SuboffGrid,
    condition_v3,
    geometry_channels,
    suboff_geometry_features,
)
from tensorlbm.ai.geometry_pipeline import GeometryEchoPipeline
from tensorlbm.ai.inference_service import (
    DragSurrogateService,
    EnvelopeMahalanobisGuardrail,
    ModelEnsembleBackend,
    save_checkpoint,
)
from tensorlbm.ai.service_backends import (
    ENV_BACKEND_FALLBACK,
    ENV_BACKEND_KIND,
    ENV_BACKEND_PLAN,
    FusedGraphBackend,
    TorchEnsembleBackend,
    make_backend,
    resolve_backend_kind,
)

TEST_GRID = SuboffGrid.from_resolution(32)
ARCH_SMALL = dict(
    in_ch=5, width=16, n_layers=2, modes=(8, 16), mlp_hidden=64, film_hidden=32, cond_dim=8
)
_SERVE_CKPTS = Path("/nfs/wangxi/runs/b4_serve_20260824/ckpts")
_PLAN_DIR = Path("/nfs/wangxi/runs/deploy_latency_20260825")
_PLAN_FP32 = _PLAN_DIR / "ensemble_cfull_stacked_trt_fp32.plan"
_PLAN_F16 = _PLAN_DIR / "ensemble_cfull_stacked_trt_f16.plan"
_V4_RUN = Path("/nfs/wangxi/runs/b4_v4_20260824")


def _tiny_checkpoint(seed: int) -> Any:
    import torch

    torch.manual_seed(seed)
    model = CondFNODrag(**ARCH_SMALL)
    from tensorlbm.ai.inference_service import CondDragCheckpoint

    return CondDragCheckpoint(
        arch=dict(ARCH_SMALL),
        state_dict=model.state_dict(),
        norm=dict(
            ch_mean=np.zeros(5, dtype=np.float64),
            ch_std=np.ones(5, dtype=np.float64),
            p_mean=np.zeros(8, dtype=np.float64),
            p_std=np.ones(8, dtype=np.float64),
            y_mean=0.0,
            y_std=1.0,
        ),
        meta=dict(member=f"m{seed}", synthetic="random weights"),
    )


def _guard_features(grid: SuboffGrid) -> np.ndarray:
    rows = []
    for hull in ("bare_hull", "with_sail", "full"):
        for sail in (0.8, 1.0, 1.2):
            geo = geometry_channels(suboff_geometry_features(hull, sail, 1.0, grid=grid))
            rows.append(
                condition_v3(
                    np.array([50.0, 100.0]),
                    np.full(2, 0.1),
                    np.full(2, sail),
                    np.ones(2),
                    np.broadcast_to(geo, (2, 4)),
                )
            )
    return np.concatenate(rows, axis=0)


class _FakeFusedRuntime:
    """Chunk-independent numpy stand-in for a fused-ensemble runtime.

    ``member_cd[m, j] = m * 100 + cond[j, 0]`` — the value depends only on the
    row CONTENT, so chunked and unchunked execution must agree bitwise and
    concatenation/order bugs show up as wrong values.
    """

    def __init__(self, n_members: int = 3) -> None:
        self.n_members = int(n_members)
        self.call_sizes: list[int] = []

    def predict(self, fields: Any, cond: Any) -> np.ndarray:
        fields = np.asarray(fields)
        cond = np.asarray(cond)
        if fields.ndim == 4:
            fields = fields[0]
        assert fields.ndim == 3 and fields.shape[0] == 5
        assert cond.ndim == 2 and cond.shape[1] == 8
        self.call_sizes.append(int(cond.shape[0]))
        base = cond[:, 0].astype(np.float64)
        return np.stack([m * 100.0 + base for m in range(self.n_members)])


def _cond_rows(values: list[float] | np.ndarray) -> np.ndarray:
    """(N, 8) condition block with the row identity in column 0."""
    arr = np.zeros((len(values), 8), dtype=np.float64)
    arr[:, 0] = values
    return arr


# ---------------------------------------------------------------------------
# pure tier: kind resolution + factory error paths
# ---------------------------------------------------------------------------


def test_resolve_backend_kind_defaults_to_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_BACKEND_KIND, raising=False)
    assert resolve_backend_kind() == "torch"
    assert resolve_backend_kind(None) == "torch"


def test_resolve_backend_kind_env_and_param_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_BACKEND_KIND, " trt ")
    assert resolve_backend_kind() == "trt"  # env used when param is None
    assert resolve_backend_kind("onnx") == "onnx"  # param beats env
    monkeypatch.setenv(ENV_BACKEND_KIND, "")
    assert resolve_backend_kind() == "torch"  # empty env = unset


def test_resolve_backend_kind_rejects_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_BACKEND_KIND, raising=False)
    with pytest.raises(ValueError, match="unknown drag backend kind"):
        resolve_backend_kind("tensorrt")
    monkeypatch.setenv(ENV_BACKEND_KIND, "bad")
    with pytest.raises(ValueError, match="TENSORLBM_DRAG_BACKEND"):
        resolve_backend_kind()


def test_make_backend_unknown_kind_lists_valid() -> None:
    with pytest.raises(ValueError, match="torch.*onnx.*trt"):
        make_backend("cuda")


def test_make_backend_torch_returns_tagged_model_backend() -> None:
    backend = make_backend("torch", ckpts=[_tiny_checkpoint(s) for s in range(2)])
    assert isinstance(backend, TorchEnsembleBackend)
    assert isinstance(backend, ModelEnsembleBackend)
    assert backend.backend_kind == "torch"
    assert backend.kind == "model"  # inherited serving kind: default semantics
    assert backend.member_labels() == ["m0", "m1"]


def test_make_backend_torch_requires_ckpts() -> None:
    with pytest.raises(ValueError, match="member checkpoints"):
        make_backend("torch")


def test_make_backend_trt_requires_artifact(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_BACKEND_PLAN, raising=False)
    with pytest.raises(ValueError, match="artifact_path") as exc:
        make_backend("trt", ckpts=[])
    assert ENV_BACKEND_PLAN in str(exc.value)


def test_make_backend_uses_env_artifact(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    plan = tmp_path / "fake.plan"
    plan.write_bytes(b"")
    monkeypatch.setenv(ENV_BACKEND_PLAN, str(plan))
    monkeypatch.setattr(
        "tensorlbm.ai.service_backends._build_trt_runtime",
        lambda *a, **k: _FakeFusedRuntime(2),
    )
    backend = make_backend("trt", ckpts=[])
    assert isinstance(backend, FusedGraphBackend)
    assert backend.init_report["artifact"] == str(plan.resolve())
    assert backend.n_members == 2


def test_make_backend_trt_unavailable_raises_with_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan = tmp_path / "missing.plan"
    monkeypatch.setenv(ENV_BACKEND_PLAN, str(plan))

    def _boom(*_a: Any, **_k: Any) -> None:
        raise ImportError("the tensorrt package is required to build engines")

    monkeypatch.setattr("tensorlbm.ai.service_backends._build_trt_runtime", _boom)
    with pytest.raises(RuntimeError, match="tensorrt package is required"):
        make_backend("trt", ckpts=[])


def test_make_backend_fallback_to_torch_records_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(ENV_BACKEND_PLAN, str(tmp_path / "x.plan"))

    def _boom(*_a: Any, **_k: Any) -> None:
        raise ImportError("the tensorrt package is required")

    monkeypatch.setattr("tensorlbm.ai.service_backends._build_trt_runtime", _boom)
    ckpts = [_tiny_checkpoint(s) for s in range(2)]
    backend = make_backend("trt", ckpts=ckpts, fallback="torch")
    assert isinstance(backend, TorchEnsembleBackend)
    assert backend.backend_kind == "torch"
    assert backend.init_report["fallback_from"] == "trt"
    assert "tensorrt package is required" in backend.init_report["fallback_reason"]
    assert backend.member_labels() == ["m0", "m1"]  # still serves


def test_make_backend_fallback_without_ckpts_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(ENV_BACKEND_PLAN, str(tmp_path / "x.plan"))

    def _boom(*_a: Any, **_k: Any) -> None:
        raise ImportError("no tensorrt here")

    monkeypatch.setattr("tensorlbm.ai.service_backends._build_trt_runtime", _boom)
    with pytest.raises(RuntimeError, match="no member checkpoints"):
        make_backend("trt", ckpts=[], fallback="torch")


# ---------------------------------------------------------------------------
# pure tier: adapter protocol against a fake runtime
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_backend() -> FusedGraphBackend:
    return FusedGraphBackend(
        _FakeFusedRuntime(3),
        kind="trt_fake",
        labels=["m0", "m1", "m2"],
        chunk=2,
    )


def test_fused_predict_protocol(fake_backend: FusedGraphBackend) -> None:
    out = fake_backend.predict(np.zeros((5, 32, 64), np.float32), _cond_rows([1, 2, 3, 4, 5]))
    assert out.shape == (3, 5)
    assert out.dtype == np.float64
    assert out[0].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert out[2].tolist() == [201.0, 202.0, 203.0, 204.0, 205.0]


def test_fused_predict_validation_matches_torch_messages(fake_backend: FusedGraphBackend) -> None:
    with pytest.raises(ValueError, match=r"fields must be \(5, ny, nx\)"):
        fake_backend.predict(np.zeros((4, 32, 64)), np.zeros((2, 8)))
    with pytest.raises(ValueError, match=r"cond must be \(N, 8\)"):
        fake_backend.predict(np.zeros((5, 32, 64)), np.zeros((2, 7)))


def test_fused_predict_batch_protocol(fake_backend: FusedGraphBackend) -> None:
    cond = _cond_rows([1, 2, 3, 4, 5, 6, 7])
    out = fake_backend.predict_batch(np.zeros((2, 5, 32, 64), np.float32), cond, np.array([5, 2]))
    assert out.shape == (3, 7)
    assert out[0].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    with pytest.raises(ValueError, match="counts must be positive"):
        fake_backend.predict_batch(np.zeros((2, 5, 32, 64)), cond, np.array([0, 7]))
    with pytest.raises(ValueError, match="counts sum 6 != condition rows 7"):
        fake_backend.predict_batch(np.zeros((2, 5, 32, 64)), cond, np.array([5, 1]))
    with pytest.raises(ValueError, match=r"fields must be \(G, 5, ny, nx\)"):
        fake_backend.predict_batch(np.zeros((5, 32, 64)), cond, np.array([7]))


def test_fused_chunking_preserves_row_order_bitwise(fake_backend: FusedGraphBackend) -> None:
    cond = _cond_rows([1, 2, 3, 4, 5])
    out = fake_backend.predict(np.zeros((5, 32, 64), np.float32), cond)
    assert fake_backend._runtime.call_sizes == [2, 2, 1]  # noqa: SLF001 — the contract under test
    unchunked = FusedGraphBackend(
        _FakeFusedRuntime(3), kind="x", labels=["m0", "m1", "m2"], chunk=None
    )
    ref = unchunked.predict(np.zeros((5, 32, 64), np.float32), cond)
    assert np.array_equal(out, ref)


def test_fused_backend_serves_in_drag_service(fake_backend: FusedGraphBackend) -> None:
    service = DragSurrogateService(
        fake_backend,
        EnvelopeMahalanobisGuardrail(_guard_features(TEST_GRID)),
        grid=TEST_GRID,
    )
    result = service.predict(
        "full",
        1.0,
        1.0,
        [50.0, 100.0],
        fields=np.zeros((5, TEST_GRID.ny, TEST_GRID.nx), np.float32),
    )
    assert result.backend == "trt_fake"  # isinstance-dispatch routed to the model path
    assert result.members == ("m0", "m1", "m2")
    assert result.cd.shape == (2,)
    assert result.lo.shape == (2,)
    assert result.guard.flag in ("ok", "review", "reject")  # verdict always attached


def test_fused_backend_serves_in_echo_pipeline(fake_backend: FusedGraphBackend) -> None:
    service = DragSurrogateService(
        fake_backend,
        EnvelopeMahalanobisGuardrail(_guard_features(TEST_GRID)),
        grid=TEST_GRID,
    )
    pipeline = GeometryEchoPipeline(service, grid=TEST_GRID, device="cpu")
    res = pipeline.predict_from_params({"hull_type": "full", "sail_scale": 1.0}, [80.0])
    assert res.backend == "trt_fake"
    assert res.cd.shape == (1,)
    assert res.confident or res.guard.flag in ("ok", "review", "reject")


# ---------------------------------------------------------------------------
# pure tier: default-path invariance of the service constructor
# ---------------------------------------------------------------------------


def _write_ckpts(tmp_path: Path, n: int = 2) -> list[Path]:
    paths = []
    for s in range(n):
        p = tmp_path / f"tiny_s{s}.pt"
        save_checkpoint(_tiny_checkpoint(s), p)
        paths.append(p)
    return paths


def test_from_checkpoints_default_builds_plain_model_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(ENV_BACKEND_KIND, raising=False)
    monkeypatch.delenv(ENV_BACKEND_FALLBACK, raising=False)
    paths = _write_ckpts(tmp_path)
    service = DragSurrogateService.from_checkpoints(
        paths, _guard_features(TEST_GRID), grid=TEST_GRID
    )
    assert type(service.backend) is ModelEnsembleBackend  # pre-TRT default, bit-for-bit


def test_from_checkpoints_env_backend_routes_through_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_BACKEND_KIND, "trt")
    monkeypatch.setenv(ENV_BACKEND_PLAN, "/tmp/whatever.plan")
    sentinel = FusedGraphBackend(
        _FakeFusedRuntime(2), kind="sentinel", labels=["m0", "m1"], chunk=8
    )
    calls: dict[str, Any] = {}

    def _fake_make_backend(kind: str, **kwargs: Any) -> FusedGraphBackend:
        calls["kind"] = kind
        calls["artifact_path"] = kwargs.get("artifact_path")
        return sentinel

    monkeypatch.setattr("tensorlbm.ai.service_backends.make_backend", _fake_make_backend)
    paths = _write_ckpts(tmp_path)
    service = DragSurrogateService.from_checkpoints(
        paths, _guard_features(TEST_GRID), grid=TEST_GRID
    )
    assert calls["kind"] == "trt"
    assert calls["artifact_path"] is None  # env plan resolves inside make_backend
    assert service.backend is sentinel


def test_from_checkpoints_bad_env_backend_fails_loudly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_BACKEND_KIND, "nonsense")
    paths = _write_ckpts(tmp_path)
    with pytest.raises(ValueError, match="unknown drag backend kind"):
        DragSurrogateService.from_checkpoints(paths, _guard_features(TEST_GRID), grid=TEST_GRID)


# ---------------------------------------------------------------------------
# ONNX tier: tiny fused artifact behind make_backend
# ---------------------------------------------------------------------------


def test_make_backend_onnx_matches_torch(tmp_path: Path) -> None:
    pytest.importorskip("onnx", reason="ONNX tier needs onnx")
    pytest.importorskip("onnxruntime", reason="ONNX tier needs onnxruntime")
    from tensorlbm.ai.onnx_deploy import export_ensemble_onnx

    ckpts = [_tiny_checkpoint(s) for s in range(2)]
    artifact = tmp_path / "tiny_ensemble.onnx"
    report = export_ensemble_onnx(ckpts, artifact, ny=32, nx=64, member_labels=["a", "b"])
    assert report["export_ok"], report["blocker"]

    backend = make_backend("onnx", ckpts=ckpts, artifact_path=artifact)
    assert isinstance(backend, FusedGraphBackend)
    assert backend.kind == "onnx"
    assert backend.member_labels() == ["a", "b"]  # artifact metadata wins

    rng = np.random.default_rng(7)
    field = rng.standard_normal((5, 32, 64)).astype(np.float32)
    cond = rng.standard_normal((6, 8))
    ref = ModelEnsembleBackend(ckpts, device="cpu").predict(field, cond)
    got = backend.predict(field, cond)
    assert got.shape == ref.shape == (2, 6)
    assert got.dtype == np.float64
    assert np.abs(np.log10(got) - np.log10(ref)).max() < 5e-3
    # batch form composes the same rows
    got_b = backend.predict_batch(field[None], cond, np.array([6]))
    assert np.abs(got_b - got).max() == 0.0


# ---------------------------------------------------------------------------
# TensorRT tier: tiny engine behind make_backend
# ---------------------------------------------------------------------------


def test_make_backend_trt_tiny_engine_matches_ort(tmp_path: Path) -> None:
    pytest.importorskip("onnx", reason="TRT tier builds engines from ONNX")
    ort = pytest.importorskip("onnxruntime", reason="TRT tier reference needs onnxruntime")
    pytest.importorskip("tensorrt", reason="TRT tier needs tensorrt")
    cudart = pytest.importorskip("cuda.bindings.runtime", reason="TRT tier needs cuda-python")
    err, count = cudart.cudaGetDeviceCount()
    if int(err) != 0 or int(count) < 1:
        pytest.skip("no CUDA device")

    from tensorlbm.ai import trt_deploy
    from tensorlbm.ai.onnx_deploy import export_ensemble_onnx

    ckpts = [_tiny_checkpoint(s) for s in range(2)]
    f64 = tmp_path / "tiny_ensemble.onnx"
    report = export_ensemble_onnx(ckpts, f64, ny=32, nx=64)
    assert report["export_ok"], report["blocker"]
    f32 = tmp_path / "tiny_ensemble_f32.onnx"
    trt_deploy.f64_tail_to_f32(f64, f32)
    plan = tmp_path / "tiny_ensemble.plan"
    built = trt_deploy.build_engine(
        f32, plan, min_batch=1, opt_batch=4, max_batch=4, workspace_gb=1.0
    )
    assert built["plan_bytes"] > 0

    backend = make_backend(
        "trt", ckpts=ckpts, artifact_path=plan, trt_max_batch=4, precision="fp32_strict"
    )
    assert backend.kind == "trt_fp32_strict"
    assert backend.chunk_rows == 4  # engine profile max wins
    assert backend.member_labels() == ["m0", "m1"]

    rng = np.random.default_rng(11)
    field = rng.standard_normal((5, 32, 64)).astype(np.float32)
    cond = rng.standard_normal((6, 8))  # forces the chunked path 4 + 2
    got = backend.predict(field, cond)
    assert got.shape == (2, 6) and got.dtype == np.float64

    session = ort.InferenceSession(str(f32), providers=["CPUExecutionProvider"])
    ref = session.run(None, {"field": field[None], "cond": cond.astype(np.float32)})[0]
    assert np.abs(np.log10(got) - np.log10(ref)).max() < 1e-4  # TRT vs same f32 graph

    torch_ref = ModelEnsembleBackend(ckpts, device="cpu").predict(field, cond)
    assert np.abs(np.log10(got) - np.log10(torch_ref)).max() < 5e-3  # graph floor


# ---------------------------------------------------------------------------
# TensorRT tier: the real b4_serve plans + checkpoints (server-gated)
# ---------------------------------------------------------------------------

_HAS_REAL = any(_SERVE_CKPTS.glob("*.pt")) and _PLAN_FP32.is_file() and _PLAN_F16.is_file()


@pytest.mark.skipif(not _HAS_REAL, reason="b4_serve plans/checkpoints not present on this host")
class TestRealPlans:
    """PR #249 plans behind the service factory: parity + verdict consistency."""

    @pytest.fixture()
    def torch_backend(self) -> ModelEnsembleBackend:
        from tensorlbm.ai.inference_service import load_checkpoint

        ckpts = [load_checkpoint(p) for p in sorted(_SERVE_CKPTS.glob("*.pt"))]
        return ModelEnsembleBackend(ckpts, device="cpu")

    @pytest.fixture()
    def corpus(self) -> Any:
        from tensorlbm.ai.inference_service import load_corpus_index

        return load_corpus_index(_V4_RUN)

    @pytest.mark.parametrize(
        ("plan", "precision", "tol"),
        [(_PLAN_FP32, "fp32_strict", 1e-6), (_PLAN_F16, "f16", 2e-3)],
    )
    def test_member_parity_and_batch_composition(
        self,
        torch_backend: ModelEnsembleBackend,
        corpus: Any,
        plan: Path,
        precision: str,
        tol: float,
    ) -> None:
        pytest.importorskip("tensorrt", reason="real-plan tier needs tensorrt")
        pytest.importorskip("cuda.bindings.runtime", reason="real-plan tier needs cuda-python")
        backend = make_backend("trt", artifact_path=plan, precision=precision, trt_max_batch=8)
        assert isinstance(backend, FusedGraphBackend)
        assert backend.n_members == torch_backend.n_members
        assert backend.member_labels() == torch_backend.member_labels()
        assert backend.chunk_rows == 8

        field = corpus.fields[3]
        cond = corpus.cond[[0, 40, 80, 120, 160, 200, 240, 270, 5]]  # 9 rows -> chunks 8 + 1
        ref = torch_backend.predict(field, cond)
        got = backend.predict(field, cond)
        assert got.shape == ref.shape == (5, 9)
        max_diff = float(np.abs(np.log10(got) - np.log10(ref)).max())
        assert max_diff < tol, f"{precision} max |dlog10 C_D| {max_diff:.3e} >= {tol}"
        # multi-geometry batch form == per-geometry predicts, row for row
        # (each geometry on its OWN field; 4- and 5-row calls stay in one chunk)
        got_b = backend.predict_batch(
            np.stack([field, corpus.fields[7]]),
            np.concatenate([cond[:4], cond[4:]]),
            np.array([4, 5]),
        )
        assert np.array_equal(got_b[:, :4], backend.predict(field, cond[:4]))
        assert np.array_equal(got_b[:, 4:], backend.predict(corpus.fields[7], cond[4:]))

    def test_three_backends_same_verdicts_and_close_cd(
        self, torch_backend: ModelEnsembleBackend, corpus: Any
    ) -> None:
        pytest.importorskip("tensorrt", reason="real-plan tier needs tensorrt")
        pytest.importorskip("cuda.bindings.runtime", reason="real-plan tier needs cuda-python")
        from tensorlbm.ai.inference_service import load_checkpoint

        ckpts = [load_checkpoint(p) for p in sorted(_SERVE_CKPTS.glob("*.pt"))]
        guard = EnvelopeMahalanobisGuardrail(corpus.cond)
        # designs that actually live in the field cache (first of each
        # (hull, sail, fin) key), swept over two archived Reynolds numbers
        first_re: dict[tuple[str, float, float], float] = {}
        for key, re in zip(corpus.designs, corpus.re):
            first_re.setdefault(key[:3], float(re))
        assert len(first_re) >= 4
        designs = [(hull, sail, fin, [re, re * 1.5]) for (hull, sail, fin), re in first_re.items()][
            :4
        ]

        def _svc(backend: Any) -> DragSurrogateService:
            return DragSurrogateService(
                backend,
                guard,
                corpus_cache=corpus.fields,
                cache_re=corpus.re,
                cache_designs=list(corpus.designs),
            )

        services = {
            "torch": _svc(torch_backend),
            "trt_fp32_strict": _svc(
                make_backend("trt", ckpts=ckpts, artifact_path=_PLAN_FP32, precision="fp32_strict")
            ),
            "trt_f16": _svc(
                make_backend("trt", ckpts=ckpts, artifact_path=_PLAN_F16, precision="f16")
            ),
        }
        for hull, sail, fin, re_list in designs:
            results = {
                name: svc.predict(hull, sail, fin, re_list) for name, svc in services.items()
            }
            flags = {r.guard.flag for r in results.values()}
            assert len(flags) == 1, (hull, sail, fin, flags)  # guard is backend-independent
            ref = results["torch"]
            for r in results.values():
                assert r.guard.reasons == ref.guard.reasons
                assert np.isclose(r.guard.score, ref.guard.score, rtol=0, atol=1e-9)
                assert r.uq_dict().keys() == ref.uq_dict().keys()
            for name, r in results.items():
                rel = float(np.max(np.abs(r.cd - ref.cd) / ref.cd))
                assert rel < 5e-3, (name, hull, sail, fin, rel)
