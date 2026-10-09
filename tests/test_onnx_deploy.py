"""Tests for the fused-ensemble ONNX deployment layer (B4-P3c).

Two tiers:

- torch-only tests (always run in CI): the fused graph classes are
  validated against ``ModelEnsembleBackend`` / ``ensemble_stats`` before
  any ONNX involvement — the norm folding, the unrolled member path and
  the stacked weight-batching must agree with the served reference in
  plain torch.
- ONNX tests (skipped when onnx/onnxruntime are absent — CI has neither):
  tiny synthetic 2-member ensemble -> fused export -> onnxruntime parity
  on random + edge conditions, the raw-in/raw-out normalisation-folding
  contract, the manifest round-trip and the backend interface.

To run the ONNX tier on the 5090 server, put the private target install
on PYTHONPATH::

    PYTHONPATH=/nfs/wangxi/runs/b4_serve_20260824/pydeps:src \\
        pytest tests/test_onnx_deploy.py -q --basetemp=/nfs/wangxi/tmp/pt_onnx

All fixtures are synthetic; nothing here touches /nfs.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from tensorlbm.ai.drag_cond import CondFNODrag
from tensorlbm.ai.inference_service import (
    CondDragCheckpoint,
    ModelEnsembleBackend,
    ensemble_stats,
    save_checkpoint,
)
from tensorlbm.ai.onnx_deploy import (
    ENSEMBLE_DESIGNS,
    MANIFEST_SCHEMA,
    MANIFEST_SCHEMA_VERSION,
    OnnxEnsembleBackend,
    StackedEnsembleGraph,
    UnrolledEnsembleGraph,
    export_ensemble_onnx,
    load_manifest,
    verify_ensemble_onnx,
    write_manifest,
)

try:  # optional deployment deps; CI has neither and the tier below skips
    import onnx as _onnx  # noqa: F401
    import onnxruntime as _ort  # noqa: F401

    HAVE_ONNX = True
except ImportError:
    HAVE_ONNX = False

requires_onnx = pytest.mark.skipif(not HAVE_ONNX, reason="onnx/onnxruntime not installed")

#: Small but structurally complete CondFNODrag (matches the production
#: body plan: lift / spectral + pointwise + FiLM / pooled head; aux off).
TINY_ARCH: dict[str, Any] = dict(
    in_ch=5, width=8, n_layers=2, modes=(4, 8), cond_dim=8, mlp_hidden=16, film_hidden=12
)
NY, NX = 16, 32


def synth_ckpt(seed: int) -> CondDragCheckpoint:
    """Random-weight member with non-trivial (non-identity) fit stats."""
    torch.manual_seed(seed)
    model = CondFNODrag(**TINY_ARCH)
    rng = np.random.default_rng(seed)
    return CondDragCheckpoint(
        arch=dict(TINY_ARCH),
        state_dict=model.state_dict(),
        norm=dict(
            ch_mean=rng.uniform(0.1, 0.9, 5).astype(np.float32),
            ch_std=rng.uniform(0.3, 1.7, 5).astype(np.float32),
            p_mean=rng.uniform(-1.0, 1.0, 8),
            p_std=rng.uniform(0.4, 1.6, 8),
            y_mean=np.asarray(0.62),
            y_std=np.asarray(0.27),
        ),
        meta=dict(seed=seed),
    )


@pytest.fixture(scope="module")
def members() -> list[CondDragCheckpoint]:
    return [synth_ckpt(0), synth_ckpt(1)]


@pytest.fixture(scope="module")
def raw_query() -> tuple[np.ndarray, np.ndarray]:
    """Random raw field + random and edge condition rows (last is a dup)."""
    rng = np.random.default_rng(7)
    field = rng.standard_normal((5, NY, NX)).astype(np.float32)
    rand = rng.standard_normal((13, 8))
    edges = np.stack([np.zeros(8), np.full(8, 4.0), np.full(8, -4.0), rand[0]])
    return field, np.concatenate([rand, edges], axis=0)


def _graph_inputs(raw_query: tuple[np.ndarray, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    field, cond = raw_query
    return torch.from_numpy(field)[None], torch.from_numpy(cond.astype(np.float32))


# ---------------------------------------------------------------------------
# torch-only tier (runs everywhere, incl. CI without onnx)
# ---------------------------------------------------------------------------


def test_unrolled_graph_matches_reference_backend(
    members: list[CondDragCheckpoint], raw_query: tuple[np.ndarray, np.ndarray]
) -> None:
    """Norm-folded unrolled graph == ModelEnsembleBackend in plain torch."""
    field, cond = raw_query
    want = ModelEnsembleBackend(members, device="cpu").predict(field, cond)
    graph = UnrolledEnsembleGraph(members, ny=NY, nx=NX).eval()
    with torch.no_grad():
        outs = graph(*_graph_inputs(raw_query))
    np.testing.assert_allclose(outs[0].numpy(), want, rtol=1e-9, atol=1e-9)


def test_in_graph_stats_match_ensemble_stats(
    members: list[CondDragCheckpoint], raw_query: tuple[np.ndarray, np.ndarray]
) -> None:
    graph = StackedEnsembleGraph(members, ny=NY, nx=NX).eval()
    with torch.no_grad():
        outs = graph(*_graph_inputs(raw_query))
    # the in-graph statistics must match ensemble_stats applied to the
    # graph's OWN member matrix (tight); member-level agreement with the
    # torch reference backend is pinned by the dedicated tests below
    mean, std, lo, hi = ensemble_stats(outs[0].numpy())
    for got, expect in zip(outs[1:], (mean, std, lo, hi)):
        np.testing.assert_allclose(got.numpy(), expect, rtol=1e-9, atol=1e-9)


def test_stacked_graph_matches_unrolled(
    members: list[CondDragCheckpoint], raw_query: tuple[np.ndarray, np.ndarray]
) -> None:
    """Weight-stacked batching == unrolled members (stacking correctness)."""
    with torch.no_grad():
        unrolled = UnrolledEnsembleGraph(members, ny=NY, nx=NX).eval()(*_graph_inputs(raw_query))
        stacked = StackedEnsembleGraph(members, ny=NY, nx=NX).eval()(*_graph_inputs(raw_query))
    for a, b in zip(unrolled, stacked):
        np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=2e-4, atol=1e-6)


def test_single_member_std_is_zero(raw_query: tuple[np.ndarray, np.ndarray]) -> None:
    solo = [synth_ckpt(3)]
    with torch.no_grad():
        outs = UnrolledEnsembleGraph(solo, ny=NY, nx=NX).eval()(*_graph_inputs(raw_query))
    assert outs[0].shape[0] == 1
    np.testing.assert_array_equal(outs[2].numpy(), 0.0)


# ---------------------------------------------------------------------------
# ONNX tier (skipped without onnx/onnxruntime)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def exported(
    tmp_path_factory: pytest.TempPathFactory, members: list[CondDragCheckpoint]
) -> dict[str, dict[str, Any]]:
    """Both designs exported once for the ONNX-tier tests."""
    reports: dict[str, dict[str, Any]] = {}
    for design in ENSEMBLE_DESIGNS:
        path = tmp_path_factory.mktemp(f"onnx_{design}") / f"tiny_ens_{design}.onnx"
        reports[design] = export_ensemble_onnx(members, path, design=design, ny=NY, nx=NX)
    return reports


@requires_onnx
@pytest.mark.parametrize("design", ENSEMBLE_DESIGNS)
def test_export_reports_ok(exported: dict[str, dict[str, Any]], design: str) -> None:
    report = exported[design]
    assert report["export_ok"], report["blocker"]
    assert report["checker"] == "ok"
    assert report["metadata_embedded"]
    assert report["artifact_bytes"] > 0
    assert report["n_members"] == 2


@requires_onnx
@pytest.mark.parametrize("design", ENSEMBLE_DESIGNS)
def test_onnx_parity_random_and_edges(
    members: list[CondDragCheckpoint],
    raw_query: tuple[np.ndarray, np.ndarray],
    exported: dict[str, dict[str, Any]],
    design: str,
) -> None:
    """Artifact vs torch backend on random + edge rows: raw in, raw out."""
    field, cond = raw_query
    want = ModelEnsembleBackend(members, device="cpu").predict(field, cond)
    backend = OnnxEnsembleBackend(exported[design]["path"])
    got = backend.predict(field, cond)
    assert got.shape == (2, cond.shape[0])
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
    stats = backend.predict_stats(field, cond)
    mean, std, lo, hi = ensemble_stats(want)
    for key, expect in zip(("cd_mean", "cd_std", "cd_min", "cd_max"), (mean, std, lo, hi)):
        np.testing.assert_allclose(stats[key], expect, rtol=1e-5, atol=1e-6)


@requires_onnx
def test_norm_folding_contract(
    members: list[CondDragCheckpoint],
    raw_query: tuple[np.ndarray, np.ndarray],
    exported: dict[str, dict[str, Any]],
) -> None:
    """The artifact consumes RAW rows: feeding pre-normalised condition rows
    must give clearly different outputs (the z-score lives in-graph)."""
    field, cond = raw_query
    ckpt_norm = members[0].norm
    cond_norm = (cond - ckpt_norm["p_mean"]) / ckpt_norm["p_std"]
    backend = OnnxEnsembleBackend(exported["stacked"]["path"])
    raw_out = backend.predict(field, cond)
    norm_out = backend.predict(field, cond_norm)
    assert not np.allclose(raw_out, norm_out, rtol=1e-3), (
        "artifact appears to expect normalised inputs"
    )
    want = ModelEnsembleBackend(members, device="cpu").predict(field, cond)
    np.testing.assert_allclose(raw_out, want, rtol=1e-5, atol=1e-6)


@requires_onnx
def test_verify_report_real_rows(
    members: list[CondDragCheckpoint],
    raw_query: tuple[np.ndarray, np.ndarray],
    exported: dict[str, dict[str, Any]],
) -> None:
    """verify_ensemble_onnx random + real-conditions passes (synthetic rows)."""
    field, cond = raw_query
    rng = np.random.default_rng(3)
    fields = rng.permutation(np.stack([field] * 6))  # row-aligned real-style block
    report = verify_ensemble_onnx(
        exported["stacked"]["path"],
        members,
        n_random=5,
        ny=NY,
        nx=NX,
        real_fields=fields,
        real_cond=cond[:6],
    )
    assert report["random"]["n_rows"] == 5
    assert report["real"]["n_rows"] == 6
    assert max(report["real"]["per_member_max_abs_log10"]) < 1e-5
    assert report["real"]["n_nonfinite_mismatch"] == 0


@requires_onnx
def test_backend_predict_batch_g1_consistency(
    raw_query: tuple[np.ndarray, np.ndarray], exported: dict[str, dict[str, Any]]
) -> None:
    field, cond = raw_query
    backend = OnnxEnsembleBackend(exported["stacked"]["path"])
    assert backend.kind == "onnx"
    assert backend.n_members == 2
    assert backend.member_labels() == ["m0", "m1"]  # embedded metadata
    single = backend.predict(field, cond)
    batched = backend.predict_batch(field[None], cond)
    assert batched.shape == (1, 2, cond.shape[0])
    np.testing.assert_array_equal(batched[0], single)
    multi = backend.predict_batch(np.stack([field, field]), cond[:4])
    assert multi.shape == (2, 2, 4)
    np.testing.assert_array_equal(multi[1], backend.predict(field, cond[:4]))


def test_manifest_round_trip(tmp_path: Path, members: list[CondDragCheckpoint]) -> None:
    """Sidecar is versioned, self-describing and hashes match the files."""
    ckpt_paths = [
        save_checkpoint(ckpt, tmp_path / f"member_{i}.pt") for i, ckpt in enumerate(members)
    ]
    onnx_path = tmp_path / "tiny_ens.onnx"
    onnx_path.write_bytes(b"not-a-real-graph-but-hashed-verbatim")
    parity = {"random": {"n_rows": 16, "per_member_max_abs": [1e-06, 2e-06]}}
    latency = {"rows": [{"backend": "onnx-cpu", "batch": 64, "p50_ms": 683.5}]}
    manifest_path = write_manifest(
        tmp_path / "manifest.json",
        ckpt_paths,
        onnx_path,
        parity,
        latency,
        design="stacked",
        ny=NY,
        nx=NX,
    )
    manifest = load_manifest(manifest_path)
    assert manifest["schema"] == MANIFEST_SCHEMA
    assert manifest["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert manifest["onnx"]["design"] == "stacked"
    assert manifest["onnx"]["ny"] == NY
    assert manifest["onnx"]["n_members"] == 2
    assert [Path(m["path"]).name for m in manifest["members"]] == ["member_0.pt", "member_1.pt"]
    assert manifest["onnx"]["sha256"] == hashlib.sha256(onnx_path.read_bytes()).hexdigest()
    assert manifest["parity"] == parity
    assert manifest["latency"] == latency
    bad = tmp_path / "bad.json"
    bad.write_text('{"schema": "something.else", "schema_version": 1}')
    with pytest.raises(ValueError, match="is not a"):
        load_manifest(bad)
