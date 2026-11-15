"""B4-P3c — fused-ensemble ONNX deployment of the SUBOFF drag surrogate.

Turns a serving checkpoint ensemble (``CondDragCheckpoint`` members of
:mod:`tensorlbm.ai.inference_service`) into ONE torch-free ONNX artifact
for CAD/DCC plugin embedding.  Building on PR #239 (``export_cond_fno_onnx``
+ the FFT-free ``SpectralConv2dMatmul`` twin, whose plain export is blocked
by ``aten::fft_rfft2``), this module fuses the whole ensemble — M members,
their per-member normalisation and the ensemble UQ statistics — into a
single graph.

Artifact contract (schema v1, pinned by ``tests/test_onnx_deploy.py``)
--------------------------------------------------------------------------------
Inputs (both raw — normalisation is folded INTO the graph):

``field``
    float32 ``(1, 5, ny, nx)`` (production ``ny=64, nx=128``) — the raw
    mid-plane channel stack ``[ux/u, uy/u, uz/u, rho, solid_mask]`` of ONE
    geometry, NOT z-scored.  The leading 1 is the geometry axis: all
    condition rows are evaluated against this single field (the served
    query pattern is one geometry swept over Reynolds numbers).
``cond``
    float32 ``(N, 8)`` with dynamic ``N >= 1`` — raw
    ``condition_v3`` rows
    ``[log10_re, log10_u_in, log10_sail_scale, log10_fin_scale,
    log_aproj_ratio, sail_frac, fin_frac, solid_frac]``, NOT z-scored.

Outputs (all float64, linear drag coefficient C_D — the same space
``ModelEnsembleBackend.predict`` + ``ensemble_stats`` operate in):

``member_cd`` ``(M, N)``
    Per-member linear C_D (``10 ** (z * y_std + y_mean)`` with the
    member's fit statistics — the log10 de-normalisation is in-graph).
``cd_mean`` / ``cd_std`` / ``cd_min`` / ``cd_max`` ``(N,)``
    Deep-ensemble mean, sample standard deviation (ddof=1; all zeros when
    ``M == 1``) and min/max band, computed INSIDE the graph so the
    artifact returns UQ, not M raw predictions.

Two fused designs are implemented; both were exported, verified and
benchmarked on the 5-member serving ensemble (see
``docs/onnx_deploy_20260825.md`` for the decision record).  Real-corpus
parity is equal (per-member max|d| 1.1e-05 linear C_D / 4.7e-07 log10 for
both); the stacked graph has 3.5x fewer nodes and is ~5.6x faster on the
onnxruntime CPU EP, so ``stacked`` is the default and ``unrolled`` stays
selectable as the simple-parity fallback:

``stacked`` (default)
    Batch-of-members via weight stacking — grouped 1x1 convolutions for
    lift/pointwise, batched matmuls for the cond/MLP linears and a
    batched contraction for the spectral weights — one forward produces
    ``(M, N)`` directly.
``unrolled``
    M sequential member blocks (each a full matmul-twin ``CondFNODrag``)
    whose outputs are stacked; simple parity, bigger graph.

``onnx`` / ``onnxruntime`` are NOT repository dependencies: every import
is guarded, the export path degrades to honest reports without them, and
the tests skip cleanly when they are absent.  For local work the private
target install is used (``uv pip install --target``)::

    PYTHONPATH=/nfs/wangxi/runs/b4_serve_20260824/pydeps:src
"""

from __future__ import annotations

import hashlib
import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from .inference_service import (
    CondDragCheckpoint,
    ModelEnsembleBackend,
    SpectralConv2dMatmul,
    ensemble_stats,
    to_matmul_spectral,
)

__all__ = [
    "ENSEMBLE_DESIGNS",
    "MANIFEST_SCHEMA",
    "MANIFEST_SCHEMA_VERSION",
    "ONNX_INPUT_NAMES",
    "ONNX_OUTPUT_NAMES",
    "OnnxEnsembleBackend",
    "StackedEnsembleGraph",
    "StackedSpectralMatmul",
    "UnrolledEnsembleGraph",
    "export_ensemble_onnx",
    "load_manifest",
    "verify_ensemble_onnx",
    "write_manifest",
]

#: Fused-graph designs shipped by :func:`export_ensemble_onnx`.
ENSEMBLE_DESIGNS = ("unrolled", "stacked")

#: Graph input/output names (also the ONNX port names).
ONNX_INPUT_NAMES = ("field", "cond")
ONNX_OUTPUT_NAMES = ("member_cd", "cd_mean", "cd_std", "cd_min", "cd_max")

#: Manifest sidecar identity (see :func:`write_manifest`).
MANIFEST_SCHEMA = "tensorlbm.onnx_deploy.ensemble_manifest"
MANIFEST_SCHEMA_VERSION = 1

_REQUIRED_NORM_KEYS = ("ch_mean", "ch_std", "p_mean", "p_std", "y_mean", "y_std")

_INSTALL_HINT = (
    "onnx/onnxruntime are optional deployment dependencies of TensorLBM; "
    "install them into a private target dir, e.g. "
    "`uv pip install --target <dir> onnx onnxruntime` and put <dir> on "
    "PYTHONPATH (server private dir: /nfs/wangxi/runs/b4_serve_20260824/pydeps)"
)


def _require_onnxruntime() -> Any:
    """Import onnxruntime on demand with a clear failure message."""
    try:
        import onnxruntime
    except ImportError as exc:  # pragma: no cover - exercised only without ort
        raise ImportError(f"onnxruntime is not installed. {_INSTALL_HINT}") from exc
    return onnxruntime


def _norm_arrays(ckpt: CondDragCheckpoint) -> dict[str, np.ndarray]:
    missing = [k for k in _REQUIRED_NORM_KEYS if k not in ckpt.norm]
    if missing:
        raise ValueError(f"checkpoint norm missing keys: {missing}")
    return {k: np.asarray(ckpt.norm[k]) for k in _REQUIRED_NORM_KEYS}


def _ensemble_outputs(cd: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Member matrix -> (members, mean, ddof=1 std, min, max), all in-graph."""
    n_members = int(cd.shape[0])
    mean = cd.mean(dim=0)
    if n_members > 1:
        var = ((cd - mean) ** 2).sum(dim=0) / float(n_members - 1)
        std = torch.sqrt(var)
    else:
        std = torch.zeros_like(mean)
    return cd, mean, std, cd.min(dim=0).values, cd.max(dim=0).values


# ---------------------------------------------------------------------------
# Design "unrolled": loop-unrolled member blocks in one graph
# ---------------------------------------------------------------------------


class _NormFoldedMember(nn.Module):
    """One member: RAW field + RAW cond in -> linear C_D out.

    The member's z-score statistics (channel, condition and label fit
    stats) are registered as buffers and applied inside ``forward`` —
    byte-for-byte the arithmetic of ``ModelEnsembleBackend.predict`` for
    that member, including the float64 log10 de-normalisation
    ``10 ** (z * y_std + y_mean)``.
    """

    # Declared buffer types (nn.Module resolves these via __getattr__ at
    # runtime; the annotations pin them for static typing — same pattern
    # as SpectralConv2dMatmul).
    ch_mean: torch.Tensor
    ch_std: torch.Tensor
    p_mean: torch.Tensor
    p_std: torch.Tensor
    y_mean: torch.Tensor
    y_std: torch.Tensor

    def __init__(self, ckpt: CondDragCheckpoint, *, ny: int, nx: int) -> None:
        super().__init__()
        norm = _norm_arrays(ckpt)
        self.net = to_matmul_spectral(ckpt.to_model(), ny=ny, nx=nx)
        self.register_buffer(
            "ch_mean", torch.as_tensor(norm["ch_mean"], dtype=torch.float32).view(1, -1, 1, 1)
        )
        self.register_buffer(
            "ch_std", torch.as_tensor(norm["ch_std"], dtype=torch.float32).view(1, -1, 1, 1)
        )
        self.register_buffer(
            "p_mean", torch.as_tensor(norm["p_mean"], dtype=torch.float32).view(1, -1)
        )
        self.register_buffer(
            "p_std", torch.as_tensor(norm["p_std"], dtype=torch.float32).view(1, -1)
        )
        self.register_buffer("y_mean", torch.tensor(float(norm["y_mean"]), dtype=torch.float64))
        self.register_buffer("y_std", torch.tensor(float(norm["y_std"]), dtype=torch.float64))

    def forward(self, field: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """``field`` (N, 5, ny, nx) raw, ``cond`` (N, 8) raw -> (N,) linear C_D."""
        x = (field - self.ch_mean) / self.ch_std
        p = (cond - self.p_mean) / self.p_std
        z = self.net(x, p)
        return torch.pow(10.0, z.double() * self.y_std + self.y_mean)


class UnrolledEnsembleGraph(nn.Module):
    """Fused ensemble, design (b): M member blocks unrolled in one graph.

    Each member is a full matmul-twin ``CondFNODrag`` with its own folded
    normalisation; the member outputs are stacked and the ensemble UQ
    statistics (mean / ddof=1 std / min-max) are computed in-graph.  No
    weight stacking tricks — the parity path is the per-member path PR
    #239 already pinned, evaluated M times.
    """

    def __init__(self, ckpts: Sequence[CondDragCheckpoint], *, ny: int, nx: int) -> None:
        super().__init__()
        if not ckpts:
            raise ValueError("ensemble needs at least one checkpoint")
        self.members = nn.ModuleList([_NormFoldedMember(c, ny=ny, nx=nx) for c in ckpts])

    def forward(self, field: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """``field`` (1, 5, ny, nx) raw, ``cond`` (N, 8) raw -> 5 outputs."""
        n = cond.shape[0]
        x = field.expand(n, -1, -1, -1)
        outs = torch.stack([m(x, cond) for m in self.members], dim=0)
        return _ensemble_outputs(outs)


# ---------------------------------------------------------------------------
# Design "stacked": batch-of-members via weight stacking
# ---------------------------------------------------------------------------


class StackedSpectralMatmul(nn.Module):
    """Spectral layer with a stacked member-axis weight, (M, B) batched.

    Same operator as :class:`~tensorlbm.ai.inference_service.SpectralConv2dMatmul`
    (real-matmul DFT bases, mirrored Hermitian columns, ortho scaling) but
    the member weights are stacked into one ``(M, in, out, my, mx, 2)``
    parameter and the layer consumes/produces ``(M, B, C, ...)`` tensors:
    the forward-y DFT and the inverse-y DFT broadcast a shared 2-D basis
    over the batch, and the complex weight multiply contracts the channel
    axis with one batched matmul over ``(M, my*mx)`` groups.  The DFT basis
    buffers are shared across members — they are pure functions of
    ``(ny, nx, modes)`` and therefore bit-identical for every member.
    """

    # Declared buffer types (see _NormFoldedMember).
    ay_r: torch.Tensor
    ay_i: torch.Tensor
    bx_r: torch.Tensor
    bx_i: torch.Tensor
    cy: torch.Tensor
    sy: torch.Tensor
    cx_mu: torch.Tensor
    sx_mu: torch.Tensor
    scale: torch.Tensor

    def __init__(self, twins: Sequence[SpectralConv2dMatmul]) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.stack([t.weight.data for t in twins], dim=0))
        ref = twins[0]
        for name in ("ay_r", "ay_i", "bx_r", "bx_i", "cy", "sy", "cx_mu", "sx_mu", "scale"):
            self.register_buffer(name, getattr(ref, name).clone())

    def _apply_weight(self, x: torch.Tensor, part: int) -> torch.Tensor:
        """``x (M, B, I, my, mx)`` x stacked weight part -> ``(M, B, O, my, mx)``.

        One batched matmul over ``(M, my*mx)`` frequency groups, each a
        ``(B, I) @ (I, O)`` channel contraction — the batched equivalent of
        the twin's ``einsum("bimn,iomn->bomn")``.  All reshape dims come
        from the (static) parameter shape, never from the traced batch
        axis, so the exporter sees constants.
        """
        m, _, i, my, mx = (int(s) for s in self.weight.shape[:5])
        o = int(self.weight.shape[2])
        k = my * mx
        xf = x.reshape(m, -1, i, k).permute(0, 3, 1, 2)
        wf = self.weight[..., part].permute(0, 3, 4, 1, 2).reshape(m, k, i, o)
        y = torch.matmul(xf, wf)
        return y.permute(0, 2, 3, 1).reshape(m, -1, o, my, mx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x (M, B, C, ny, nx)`` -> ``(M, B, C, ny, nx)``."""
        xt = x.transpose(-1, -2)
        xr = torch.matmul(xt, self.ay_r.transpose(0, 1)).transpose(-1, -2)
        xi = torch.matmul(xt, self.ay_i.transpose(0, 1)).transpose(-1, -2)
        x_r = torch.matmul(xr, self.bx_r) - torch.matmul(xi, self.bx_i)
        x_i = torch.matmul(xr, self.bx_i) + torch.matmul(xi, self.bx_r)
        y_r = self._apply_weight(x_r, 0) - self._apply_weight(x_i, 1)
        y_i = self._apply_weight(x_r, 1) + self._apply_weight(x_i, 0)
        p = torch.matmul(y_r, self.cx_mu) - torch.matmul(y_i, self.sx_mu)
        q = torch.matmul(y_i, self.cx_mu) + torch.matmul(y_r, self.sx_mu)
        return self.scale * (
            torch.matmul(self.cy.transpose(0, 1), p) - torch.matmul(self.sy.transpose(0, 1), q)
        )


def _grouped_conv1x1(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, h: int, w: int
) -> torch.Tensor:
    """Stacked 1x1 conv over the member axis via ONE grouped Conv2d node.

    ``x`` ``(M, B, C, H, W)``; ``weight`` ``(M, O, C, 1, 1)``; ``bias``
    ``(M, O)`` — the per-member weights are stacked on axis 0 so a single
    ``Conv2d(groups=M)`` over a ``(B, M*C, H, W)`` layout applies member
    ``m``'s kernel to member ``m``'s slice, arithmetic identical to M
    separate 1x1 convolutions.  ``h``/``w`` must be static ints: every
    reshape dim except the traced batch axis is a constant so the exporter
    can prove the convolution channel shapes.
    """
    m = int(weight.shape[0])
    o = int(weight.shape[1])
    c = int(weight.shape[2])
    xb = x.permute(1, 0, 2, 3, 4).reshape(-1, m * c, h, w)
    y = nn.functional.conv2d(xb, weight.reshape(m * o, c, 1, 1), bias.reshape(m * o), groups=m)
    return y.reshape(-1, m, o, h, w).permute(1, 0, 2, 3, 4)


def _batched_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Stacked Linear over the member axis: ``x (M, B, I)``, ``weight (M, O, I)``."""
    return torch.matmul(x, weight.transpose(1, 2)) + bias.unsqueeze(1)


def _linear_at(seq: nn.Sequential | nn.ModuleList, i: int) -> nn.Linear:
    """The i-th Linear of a Sequential/ModuleList (typed stacking accessor)."""
    layer = seq[i]
    assert isinstance(layer, nn.Linear)
    return layer


def _conv_at(modules: nn.ModuleList, i: int) -> nn.Conv2d:
    """The i-th Conv2d of a ModuleList (typed accessor for stacking)."""
    layer = modules[i]
    assert isinstance(layer, nn.Conv2d)
    return layer


def _spectral_at(modules: nn.ModuleList, i: int) -> SpectralConv2dMatmul:
    """The i-th matmul-twin spectral layer of a twin CondFNODrag."""
    layer = modules[i]
    assert isinstance(layer, SpectralConv2dMatmul)
    return layer


def _bias_of(layer: nn.Linear | nn.Conv2d) -> torch.Tensor:
    """Bias of a Linear/Conv2d (always present in the served body plan)."""
    bias = layer.bias
    assert bias is not None
    return bias.data


class StackedEnsembleGraph(nn.Module):
    """Fused ensemble, design (a): batch-of-members via weight stacking.

    Every member tensor (lift/pointwise conv kernels, spectral weights,
    cond/FiLM/MLP linears, normalisation statistics) is stacked on a new
    leading member axis and the whole ensemble runs as one batched forward
    producing ``(M, N)`` member outputs: grouped 1x1 convolutions for the
    spatial mixes, batched matmuls for the condition/MLP paths and
    :class:`StackedSpectralMatmul` for the Fourier mixes.  Normalisation
    folding and the ensemble UQ statistics are identical to the unrolled
    design.
    """

    # Declared buffer types (see _NormFoldedMember).
    ch_mean: torch.Tensor
    ch_std: torch.Tensor
    p_mean: torch.Tensor
    p_std: torch.Tensor
    y_mean: torch.Tensor
    y_std: torch.Tensor

    def __init__(self, ckpts: Sequence[CondDragCheckpoint], *, ny: int, nx: int) -> None:
        super().__init__()
        if not ckpts:
            raise ValueError("ensemble needs at least one checkpoint")
        twins = [to_matmul_spectral(c.to_model(), ny=ny, nx=nx) for c in ckpts]
        ref = twins[0]
        self.n_members = len(twins)
        self.n_layers = len(ref.spectral)
        self.width = int(ref.lift.out_channels)
        self.ny = int(ny)
        self.nx = int(nx)

        self.lift_w = nn.Parameter(torch.stack([t.lift.weight.data for t in twins], dim=0))
        self.lift_b = nn.Parameter(torch.stack([_bias_of(t.lift) for t in twins], dim=0))
        self.spectral = nn.ModuleList(
            [
                StackedSpectralMatmul([_spectral_at(t.spectral, i) for t in twins])
                for i in range(self.n_layers)
            ]
        )
        self.pointwise_w = nn.ParameterList(
            [
                nn.Parameter(
                    torch.stack([_conv_at(t.pointwise, i).weight.data for t in twins], dim=0)
                )
                for i in range(self.n_layers)
            ]
        )
        self.pointwise_b = nn.ParameterList(
            [
                nn.Parameter(
                    torch.stack([_bias_of(_conv_at(t.pointwise, i)) for t in twins], dim=0)
                )
                for i in range(self.n_layers)
            ]
        )
        self.ce0_w = nn.Parameter(
            torch.stack([_linear_at(t.cond_embed, 0).weight.data for t in twins], dim=0)
        )
        self.ce0_b = nn.Parameter(
            torch.stack([_bias_of(_linear_at(t.cond_embed, 0)) for t in twins], dim=0)
        )
        self.ce1_w = nn.Parameter(
            torch.stack([_linear_at(t.cond_embed, 2).weight.data for t in twins], dim=0)
        )
        self.ce1_b = nn.Parameter(
            torch.stack([_bias_of(_linear_at(t.cond_embed, 2)) for t in twins], dim=0)
        )
        self.film_w = nn.ParameterList(
            [
                nn.Parameter(torch.stack([_linear_at(t.film, i).weight.data for t in twins], dim=0))
                for i in range(self.n_layers)
            ]
        )
        self.film_b = nn.ParameterList(
            [
                nn.Parameter(torch.stack([_bias_of(_linear_at(t.film, i)) for t in twins], dim=0))
                for i in range(self.n_layers)
            ]
        )
        self.head0_w = nn.Parameter(
            torch.stack([_linear_at(t.head, 0).weight.data for t in twins], dim=0)
        )
        self.head0_b = nn.Parameter(
            torch.stack([_bias_of(_linear_at(t.head, 0)) for t in twins], dim=0)
        )
        self.head1_w = nn.Parameter(
            torch.stack([_linear_at(t.head, 2).weight.data for t in twins], dim=0)
        )
        self.head1_b = nn.Parameter(
            torch.stack([_bias_of(_linear_at(t.head, 2)) for t in twins], dim=0)
        )

        norms = [_norm_arrays(c) for c in ckpts]
        self.register_buffer(
            "ch_mean",
            torch.as_tensor(np.stack([n["ch_mean"] for n in norms]), dtype=torch.float32).view(
                -1, 1, 5, 1, 1
            ),
        )
        self.register_buffer(
            "ch_std",
            torch.as_tensor(np.stack([n["ch_std"] for n in norms]), dtype=torch.float32).view(
                -1, 1, 5, 1, 1
            ),
        )
        self.register_buffer(
            "p_mean",
            torch.as_tensor(np.stack([n["p_mean"] for n in norms]), dtype=torch.float32).view(
                -1, 1, 8
            ),
        )
        self.register_buffer(
            "p_std",
            torch.as_tensor(np.stack([n["p_std"] for n in norms]), dtype=torch.float32).view(
                -1, 1, 8
            ),
        )
        self.register_buffer(
            "y_mean",
            torch.as_tensor(
                np.array([float(n["y_mean"]) for n in norms]), dtype=torch.float64
            ).view(-1, 1),
        )
        self.register_buffer(
            "y_std",
            torch.as_tensor(np.array([float(n["y_std"]) for n in norms]), dtype=torch.float64).view(
                -1, 1
            ),
        )

    def forward(self, field: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """``field`` (1, 5, ny, nx) raw, ``cond`` (N, 8) raw -> 5 outputs."""
        n = cond.shape[0]
        x = (field.unsqueeze(0) - self.ch_mean) / self.ch_std
        x = x.expand(self.n_members, n, -1, -1, -1)
        p = (cond.unsqueeze(0) - self.p_mean) / self.p_std
        e = nn.functional.gelu(_batched_linear(p, self.ce0_w, self.ce0_b))
        e = _batched_linear(e, self.ce1_w, self.ce1_b)
        x = _grouped_conv1x1(x, self.lift_w, self.lift_b, self.ny, self.nx)
        for i in range(self.n_layers):
            h = self.spectral[i](x) + _grouped_conv1x1(
                x, self.pointwise_w[i], self.pointwise_b[i], self.ny, self.nx
            )
            gamma, beta = _batched_linear(e, self.film_w[i], self.film_b[i]).chunk(2, dim=-1)
            x = nn.functional.gelu(
                gamma.unsqueeze(-1).unsqueeze(-1) * h + beta.unsqueeze(-1).unsqueeze(-1)
            )
        pooled = x.mean(dim=(-2, -1))
        feat = torch.cat([pooled, p], dim=-1)
        z = nn.functional.gelu(_batched_linear(feat, self.head0_w, self.head0_b))
        z = _batched_linear(z, self.head1_w, self.head1_b).squeeze(-1)
        cd = torch.pow(10.0, z.double() * self.y_std + self.y_mean)
        return _ensemble_outputs(cd)


_GRAPH_CLASSES: dict[str, type[nn.Module]] = {
    "unrolled": UnrolledEnsembleGraph,
    "stacked": StackedEnsembleGraph,
}


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_ensemble_onnx(
    ckpts: Sequence[CondDragCheckpoint],
    path: str | Path,
    *,
    opset: int = 17,
    design: str = "stacked",
    ny: int = 64,
    nx: int = 128,
    member_labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Export the whole ensemble (norm folding + UQ stats) to ONE ONNX graph.

    Parameters
    ----------
    ckpts:
        Member checkpoints (``CondDragCheckpoint``); arch must be shared
        across members, per-member normalisation may differ (it is folded
        into the graph per member).
    path:
        Destination ``.onnx`` file (parents created).
    design:
        ``"stacked"`` (batched weights, default — fewer nodes, ~5.6x
        faster on the ORT CPU EP) or ``"unrolled"`` (M member blocks).
    opset:
        ONNX opset (17 pinned; the matmul twin needs nothing newer).
    member_labels:
        Optional labels embedded as ONNX model metadata
        (``tensorlbm.member_labels``) for downstream provenance.

    Returns an honest report dict: export success / blocker string, export
    wall time, artifact size, node count, checker result and a random-batch
    onnxruntime smoke parity vs the torch graph.  ``onnx`` (checker /
    metadata) and ``onnxruntime`` (parity) are optional and their absence
    is recorded, never papered over.
    """
    if design not in ENSEMBLE_DESIGNS:
        raise ValueError(f"design must be one of {ENSEMBLE_DESIGNS}, got {design!r}")
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if member_labels is None:
        member_labels = [str(c.meta.get("member", f"m{i}")) for i, c in enumerate(ckpts)]
    report: dict[str, Any] = {
        "design": design,
        "n_members": len(ckpts),
        "opset": int(opset),
        "ny": int(ny),
        "nx": int(nx),
        "path": None,
        "export_ok": False,
        "blocker": None,
        "export_seconds": None,
        "artifact_bytes": None,
        "graph_nodes": None,
        "metadata_embedded": False,
        "checker": "skipped (onnx package not installed)",
        "runtime_providers": None,
        "runtime_parity": "skipped (onnxruntime not installed)",
        "torch": str(torch.__version__),
    }

    graph = _GRAPH_CLASSES[design](ckpts, ny=ny, nx=nx)
    example_field = torch.randn(1, 5, ny, nx)
    example_cond = torch.randn(4, 8)
    with torch.no_grad():
        graph.eval()
        t0 = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=getattr(torch.jit, "TracerWarning", UserWarning)
                )
                torch.onnx.export(
                    graph,
                    (example_field, example_cond),
                    str(out_path),
                    opset_version=int(opset),
                    input_names=list(ONNX_INPUT_NAMES),
                    output_names=list(ONNX_OUTPUT_NAMES),
                    dynamic_axes={
                        "cond": {0: "n_cond"},
                        "member_cd": {1: "n_cond"},
                        "cd_mean": {0: "n_cond"},
                        "cd_std": {0: "n_cond"},
                        "cd_min": {0: "n_cond"},
                        "cd_max": {0: "n_cond"},
                    },
                    dynamo=False,
                )
        except Exception as exc:  # noqa: BLE001 — the blocker text is the deliverable
            report["blocker"] = f"{type(exc).__name__}: {exc}"
            return report
        report["export_seconds"] = float(time.perf_counter() - t0)
        report["export_ok"] = True
        report["path"] = str(out_path.resolve())
        report["artifact_bytes"] = out_path.stat().st_size
        ref = graph(example_field, example_cond)

    try:
        import onnx as _onnx

        model = _onnx.load(str(out_path))
        report["graph_nodes"] = len(model.graph.node)
        meta_props = {
            "tensorlbm.member_labels": json.dumps(list(member_labels)),
            "tensorlbm.ensemble_contract": json.dumps(
                {
                    "design": design,
                    "opset": int(opset),
                    "field": "float32 (1, 5, ny, nx) raw",
                    "cond": "float32 (N, 8) raw condition_v3",
                    "outputs": {
                        "member_cd": "float64 (M, N) linear C_D",
                        "cd_mean": "float64 (N,) linear C_D",
                        "cd_std": "float64 (N,) sample std ddof=1",
                        "cd_min": "float64 (N,)",
                        "cd_max": "float64 (N,)",
                    },
                    "ny": int(ny),
                    "nx": int(nx),
                }
            ),
        }
        for key, value in meta_props.items():
            entry = model.metadata_props.add()
            entry.key = key
            entry.value = value
        _onnx.save(model, str(out_path))
        report["artifact_bytes"] = out_path.stat().st_size
        report["metadata_embedded"] = True
        try:
            _onnx.checker.check_model(model)
            report["checker"] = "ok"
        except Exception as exc:  # noqa: BLE001
            report["checker"] = f"failed: {type(exc).__name__}: {exc}"
    except ImportError:
        report["metadata_embedded"] = False

    try:
        _ort = _require_onnxruntime()
        session = _ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        report["runtime_providers"] = list(_ort.get_available_providers())
        outs = session.run(
            None,
            {
                "field": example_field.numpy(),
                "cond": example_cond.numpy(),
            },
        )
        parities: dict[str, Any] = {}
        with np.errstate(divide="ignore", invalid="ignore"):
            for i, (name, out) in enumerate(zip(ONNX_OUTPUT_NAMES, outs)):
                got = np.asarray(out, dtype=np.float64)
                want = ref[i].numpy().astype(np.float64)
                if name == "member_cd":
                    # log10 space: random trace inputs are far out of
                    # distribution and the linear-scale |d| is meaningless
                    # there (10**z amplifies float32 noise to 1e+40).
                    d, _ = _max_abs_finite(np.log10(got), np.log10(want))
                else:
                    d, _ = _max_abs_finite(got, want)
                parities[name] = d
        report["runtime_parity"] = {
            "n_cond": int(example_cond.shape[0]),
            "max_abs_per_output": parities,
            "max_abs_overall": max(parities.values()),
        }
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001
        report["runtime_parity"] = f"failed: {type(exc).__name__}: {exc}"
    return report


# ---------------------------------------------------------------------------
# Runtime backend
# ---------------------------------------------------------------------------


class OnnxEnsembleBackend:
    """ORT-backed twin of :class:`ModelEnsembleBackend` over one artifact.

    Same raw-input contract and return space as the torch backend —
    :meth:`predict` takes the raw ``(5, ny, nx)`` field and raw ``(N, 8)``
    condition rows and returns the ``(M, N)`` member matrix of linear C_D
    (guardrails compose outside, exactly as for the torch backend) — but
    evaluates ONE onnxruntime session of the fused-ensemble graph, so the
    host needs no torch.  The graph's in-graph statistics are exposed via
    :meth:`predict_stats`.

    Providers default to CPU; pass ``providers=["CUDAExecutionProvider"]``
    when the runtime build has it (the server private build does not — its
    provider list is Azure/CPU only, recorded in the export report).
    """

    def __init__(
        self,
        onnx_path: str | Path,
        *,
        providers: Sequence[str] | None = None,
        intra_op_threads: int | None = None,
    ) -> None:
        ort = _require_onnxruntime()
        provs = list(providers) if providers else ["CPUExecutionProvider"]
        options = None
        if intra_op_threads is not None:
            options = ort.SessionOptions()
            options.intra_op_num_threads = int(intra_op_threads)
            options.inter_op_num_threads = 1
        kwargs: dict[str, Any] = {} if options is None else {"sess_options": options}
        self._session = ort.InferenceSession(str(onnx_path), providers=provs, **kwargs)
        self.onnx_path = str(Path(onnx_path).resolve())
        self.providers = list(self._session.get_providers())
        in_names = [i.name for i in self._session.get_inputs()]
        out_names = [o.name for o in self._session.get_outputs()]
        if tuple(in_names) != ONNX_INPUT_NAMES or tuple(out_names) != ONNX_OUTPUT_NAMES:
            raise ValueError(
                f"artifact ports do not match the ensemble contract: in={in_names}, out={out_names}"
            )
        member_shape = self._session.get_outputs()[0].shape
        self._n_members = int(member_shape[0])
        meta = self._session.get_modelmeta().custom_metadata_map
        labels = (
            json.loads(meta["tensorlbm.member_labels"])
            if "tensorlbm.member_labels" in meta
            else None
        )
        self._labels = (
            [str(x) for x in labels]
            if labels is not None
            else [f"m{i}" for i in range(self._n_members)]
        )

    @property
    def n_members(self) -> int:
        return self._n_members

    @property
    def kind(self) -> str:
        return "onnx"

    def member_labels(self) -> list[str]:
        return list(self._labels)

    def _run(self, field: np.ndarray, cond: np.ndarray) -> dict[str, np.ndarray]:
        field = np.asarray(field, dtype=np.float32)
        cond = np.asarray(cond, dtype=np.float64).astype(np.float32)
        if field.ndim == 3:
            field = field[None]
        if field.ndim != 4 or field.shape[0] != 1:
            raise ValueError(f"field must be (1, 5, ny, nx) or (5, ny, nx), got {field.shape}")
        if cond.ndim != 2 or cond.shape[1] != 8:
            raise ValueError(f"cond must be (N, 8), got {cond.shape}")
        outs = self._session.run(None, {"field": field, "cond": cond})
        return {name: np.asarray(out) for name, out in zip(ONNX_OUTPUT_NAMES, outs)}

    def predict(self, fields: np.ndarray, cond: np.ndarray) -> np.ndarray:
        """Raw field + raw cond rows -> ``(M, N)`` member linear C_D."""
        return self._run(fields, cond)["member_cd"]

    def predict_stats(self, fields: np.ndarray, cond: np.ndarray) -> dict[str, np.ndarray]:
        """Raw inputs -> in-graph ensemble stats (mean/std/min/max, linear C_D)."""
        outs = self._run(fields, cond)
        return {k: outs[k] for k in ("cd_mean", "cd_std", "cd_min", "cd_max")}

    def predict_batch(self, fields: np.ndarray, cond: np.ndarray) -> np.ndarray:
        """Geometry batch variant: ``(G, M, N)`` member linear C_D.

        ``fields`` is ``(G, 5, ny, nx)`` (one field per geometry; every
        geometry is evaluated against all ``N`` condition rows), ``cond``
        is ``(N, 8)``.  The artifact itself carries ONE geometry axis, so
        ``G`` geometries are G session calls stacked host-side; with
        ``G == 1`` this is exactly :meth:`predict`.
        """
        fields = np.asarray(fields, dtype=np.float32)
        if fields.ndim != 4:
            raise ValueError(f"fields must be (G, 5, ny, nx), got {fields.shape}")
        return np.stack([self.predict(fields[g], cond) for g in range(fields.shape[0])], axis=0)


# ---------------------------------------------------------------------------
# Parity verification
# ---------------------------------------------------------------------------


def _max_abs_finite(got: np.ndarray, ref: np.ndarray) -> tuple[float, int]:
    """Max |got-ref| over positions where BOTH are finite.

    Out-of-distribution stress rows (pure-noise fields) legitimately drive
    ``10 ** z`` to overflow in the torch reference AND the graph; those
    positions are excluded from the max but counted as mismatches whenever
    finiteness disagrees between the two — so a graph that overflows where
    torch does not (or vice versa) is caught by ``n_nonfinite_mismatch``.
    """
    got = np.asarray(got, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    mismatch = int((np.isfinite(ref) != np.isfinite(got)).sum())
    both = np.isfinite(ref) & np.isfinite(got)
    if not both.any():
        return 0.0, mismatch
    return float(np.abs(got[both] - ref[both]).max()), mismatch


def _parity_block(
    session: Any,
    reference: ModelEnsembleBackend,
    field: np.ndarray,
    cond: np.ndarray,
) -> dict[str, Any]:
    """Compare one (field, cond rows) block: graph vs torch reference."""
    torch_members = reference.predict(field, cond)  # (M, N) linear C_D, float64
    outs = session.run(
        None,
        {
            "field": np.asarray(field, dtype=np.float32)[None],
            "cond": np.asarray(cond).astype(np.float32),
        },
    )
    onnx_members = np.asarray(outs[0])
    t_mean, t_std, t_lo, t_hi = ensemble_stats(torch_members)
    per_member, per_member_log = [], []
    with np.errstate(divide="ignore", invalid="ignore"):
        for m in range(torch_members.shape[0]):
            d, _ = _max_abs_finite(onnx_members[m], torch_members[m])
            per_member.append(d)
            d_log, _ = _max_abs_finite(np.log10(onnx_members[m]), np.log10(torch_members[m]))
            per_member_log.append(d_log)
    stats = {}
    for key, ref_arr in (("mean", t_mean), ("std", t_std), ("min", t_lo), ("max", t_hi)):
        name = {"mean": 1, "std": 2, "min": 3, "max": 4}[key]
        d, bad = _max_abs_finite(np.asarray(outs[name]), ref_arr)
        stats[key] = d
        if bad:
            stats[f"{key}_nonfinite_mismatch"] = bad
    _, total_bad = _max_abs_finite(onnx_members, torch_members)
    return {
        "n_rows": int(cond.shape[0]),
        "per_member_max_abs": per_member,
        "per_member_max_abs_log10": per_member_log,
        "n_nonfinite_mismatch": total_bad,
        "stats_max_abs": stats,
    }


def verify_ensemble_onnx(
    onnx_path: str | Path,
    ckpts: Sequence[CondDragCheckpoint],
    *,
    n_random: int = 64,
    seed: int = 0,
    ny: int = 64,
    nx: int = 128,
    real_fields: np.ndarray | None = None,
    real_cond: np.ndarray | None = None,
    max_real_rows: int = 512,
    reference_device: str = "cpu",
) -> dict[str, Any]:
    """Parity report of the fused artifact vs the torch ensemble backend.

    Two passes:

    * random — ``n_random`` standard-normal condition rows against one
      standard-normal field (parity is distribution-free; this pins the
      graph equivalence itself);
    * real — optional row-aligned ``(R, 5, ny, nx)`` fields and ``(R, 8)``
      condition rows (e.g. corpus rows from a B4 ``cache_v4.npz``): each
      row is evaluated with its own field and its own condition row, so
      the artifact is compared on real inputs, not just random ones.

    The reference runs on CPU (``reference_device``) for determinism —
    device kernels differ in reduction order, which would pollute the
    graph-parity signal.  Reports per-member max|d| in linear and log10
    C_D plus max|d| of the in-graph ensemble statistics.
    """
    ort = _require_onnxruntime()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    reference = ModelEnsembleBackend(ckpts, device=reference_device)
    rng = np.random.default_rng(seed)
    rand_field = rng.standard_normal((5, ny, nx)).astype(np.float32)
    rand_cond = rng.standard_normal((n_random, 8))
    report: dict[str, Any] = {
        "onnx_path": str(Path(onnx_path).resolve()),
        "n_members": len(ckpts),
        "reference_device": reference_device,
        "torch": str(torch.__version__),
        "onnxruntime": ort.__version__,
        "seed": int(seed),
        "random": _parity_block(session, reference, rand_field, rand_cond),
        "real": None,
    }
    if real_fields is not None and real_cond is not None:
        real_fields = np.asarray(real_fields, dtype=np.float32)
        real_cond = np.asarray(real_cond, dtype=np.float64)
        if real_fields.shape[0] != real_cond.shape[0]:
            raise ValueError(
                f"real fields/cond must be row-aligned, got {real_fields.shape} vs {real_cond.shape}"
            )
        r = min(real_fields.shape[0], int(max_real_rows))
        per_member = np.zeros((len(ckpts),))
        per_member_log = np.zeros((len(ckpts),))
        stats: dict[str, float] = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        nonfinite_bad = 0
        for i in range(r):
            block = _parity_block(session, reference, real_fields[i], real_cond[i : i + 1])
            per_member = np.maximum(per_member, np.asarray(block["per_member_max_abs"]))
            per_member_log = np.maximum(
                per_member_log, np.asarray(block["per_member_max_abs_log10"])
            )
            for key in stats:
                stats[key] = max(stats[key], block["stats_max_abs"][key])
            nonfinite_bad += int(block["n_nonfinite_mismatch"])
        report["real"] = {
            "n_rows": int(r),
            "n_available": int(real_fields.shape[0]),
            "per_member_max_abs": per_member.tolist(),
            "per_member_max_abs_log10": per_member_log.tolist(),
            "n_nonfinite_mismatch": nonfinite_bad,
            "stats_max_abs": stats,
        }
    return report


# ---------------------------------------------------------------------------
# Deployment manifest
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(
    path: str | Path,
    ckpt_paths: Sequence[str | Path],
    onnx_path: str | Path,
    parity_report: dict[str, Any] | None,
    latency: dict[str, Any] | None,
    *,
    design: str = "stacked",
    opset: int = 17,
    ny: int = 64,
    nx: int = 128,
    extra: dict[str, Any] | None = None,
) -> str:
    """Write the versioned JSON sidecar describing the deployed artifact.

    Downstream tooling (CAD/DCC plugin loaders, artifact registries) reads
    this to know exactly what the ``.onnx`` file is: member checkpoints +
    hashes, artifact hash/size/design/opset, the pinned IO contract, and
    the parity / latency evidence collected at build time.
    """
    onnx_file = Path(onnx_path)
    members = []
    for p in ckpt_paths:
        p = Path(p)
        members.append(
            {
                "path": str(p.resolve()),
                "size_bytes": p.stat().st_size,
                "sha256": _sha256(p),
            }
        )
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": "tensorlbm.ai.onnx_deploy",
        "onnx": {
            "path": str(onnx_file.resolve()),
            "size_bytes": onnx_file.stat().st_size,
            "sha256": _sha256(onnx_file),
            "design": design,
            "opset": int(opset),
            "n_members": len(members),
            "inputs": {
                "field": "float32 (1, 5, ny, nx) raw",
                "cond": "float32 (N, 8) raw condition_v3",
            },
            "outputs": {
                "member_cd": "float64 (M, N) linear C_D",
                "cd_mean": "float64 (N,) linear C_D",
                "cd_std": "float64 (N,) sample std ddof=1",
                "cd_min": "float64 (N,) linear C_D",
                "cd_max": "float64 (N,) linear C_D",
            },
            "ny": int(ny),
            "nx": int(nx),
        },
        "members": members,
        "parity": parity_report,
        "latency": latency,
    }
    if extra:
        manifest["extra"] = dict(extra)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    return str(out.resolve())


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate a :func:`write_manifest` sidecar."""
    with open(Path(path), encoding="utf-8") as fh:
        loaded: dict[str, Any] = json.load(fh)
        manifest = loaded
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"{path} is not a {MANIFEST_SCHEMA} file")
    if int(manifest.get("schema_version", -1)) != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"manifest schema_version {manifest.get('schema_version')!r} != {MANIFEST_SCHEMA_VERSION}"
        )
    return manifest
