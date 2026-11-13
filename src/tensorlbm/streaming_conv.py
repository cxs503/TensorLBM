"""Conv-as-streaming: discrete transport of populations as a *fixed* convolution kernel.

The idea is taken from ``loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow``
(``LatFlow/D2Q9.py`` ``STREAM`` table + ``LatFlow/Domain.py`` ``StreamSC`` /
``CollideSC``, ``utils.simple_conv`` / ``utils.pad_mobius``): populations are
stored channel-last ``[1, *Ndim, q]`` and the whole migration step -- direction
re-ordering plus the neighbour shift -- is a single 3x3 (2D) / 3x3x3 (3D)
constant convolution with *one* nonzero tap per output channel.  Wall bounce is
a second, 1x1 convolution whose kernel is the direction-reversal (BOUNCE)
matrix, blended through a boundary mask.

This module ports that formulation onto TensorLBM's native field layout
``(q, ny, nx)`` / ``(q, nz, ny, nx)`` (i.e. axis order ``y, x`` / ``z, y, x``)
and deliberately **derives** the kernels from the velocity table instead of
copying the upstream tables:

    kernel index along an axis = 1 - c_axis

(the ``1`` is the circular-padding offset: with one ghost cell on each side a
``VALID`` cross-correlation tap at index ``k`` reads the original cell
``k - 1``).  Upstream encodes the same relation by hand into ``STREAM``, which
is how the D3Q19 ``q=3`` tap came to be written as ``(0, 0, 1)`` instead of
``(1, 0, 1)`` -- see :func:`stream_table_mismatches`, the self-check upstream
lacks.

All entry points are torch-based and mirror :func:`tensorlbm.solver.stream`
(push/gather convention: ``out[q, y, x] = f[q, y - c_y, x - c_x]``) so the two
transport formulations are drop-in interchangeable.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "AXIS_COMPONENT",
    "bounce_blend",
    "bounce_kernel_dense",
    "conv_stream_kernel",
    "conv_stream_kernel_dense",
    "describe_stream_table",
    "stream_conv",
    "stream_conv_2d",
    "stream_conv_3d",
    "stream_table_mismatches",
]

#: velocity-component index for each field axis name
AXIS_COMPONENT = {"x": 0, "y": 1, "z": 2}


# --------------------------------------------------------------------- kernels
def _lveloc_array(lveloc: Any) -> np.ndarray:
    """Velocity table as ``(q, 3)`` int array; 2D ``(q, 2)`` tables are zero-padded."""
    lv = lveloc.detach().cpu().numpy() if isinstance(lveloc, torch.Tensor) else np.asarray(lveloc)
    lv = lv.astype(np.int64, copy=False)
    if lv.ndim != 2 or lv.shape[1] not in (2, 3):
        raise ValueError(f"velocity table must have shape (q, 2) or (q, 3), got {lv.shape}")
    if lv.shape[1] == 2:
        lv = np.concatenate([lv, np.zeros((lv.shape[0], 1), dtype=np.int64)], axis=1)
    return lv


def _kernel_index(c: Sequence[int], axes: str) -> tuple[int, ...]:
    """Padding-aware cross-correlation tap index for velocity ``c``."""
    return tuple(1 - int(c[AXIS_COMPONENT[a]]) for a in axes)


def conv_stream_kernel(lveloc: Any, axes: str = "yx") -> np.ndarray:
    """Depthwise kernel of the fixed-kernel streaming operator.

    Returns an array of shape ``(q, 1, 3, 3)`` for ``axes="yx"`` and
    ``(q, 1, 3, 3, 3)`` for ``axes="zyx"`` -- the layout
    ``torch.nn.functional.conv2d`` / ``conv3d`` expects for grouped
    (one group per population) convolution.  Exactly one tap per output
    channel is 1.0, all others are 0.0, so applying it is an exact
    permutation of memory (no arithmetic error).
    """
    lv = _lveloc_array(lveloc)
    q = lv.shape[0]
    weight = np.zeros((q, 1) + (3,) * len(axes), dtype=np.float64)
    for i in range(q):
        weight[(i, 0) + _kernel_index(lv[i], axes)] = 1.0
    return weight


def conv_stream_kernel_dense(lveloc: Any, axes: str = "yx") -> np.ndarray:
    """Same operator in the upstream (channel-last, cross-channel) layout.

    Shape ``(3, 3, q, q)`` (2D) or ``(3, 3, 3, q, q)`` (3D): the tail
    ``[q_in, q_out]`` block is diagonal, which is what makes streaming
    expressible as an ordinary ``conv2d`` on a ``[1, *Ndim, q]`` tensor.
    """
    lv = _lveloc_array(lveloc)
    q = lv.shape[0]
    dense = np.zeros((3,) * len(axes) + (q, q), dtype=np.float64)
    for i in range(q):
        dense[_kernel_index(lv[i], axes) + (i, i)] = 1.0
    return dense


def bounce_kernel_dense(opposite: Any) -> np.ndarray:
    """1x1 bounce kernel ``(1, 1, q, q)``: row ``q`` is one-hot on ``opposite[q]``."""
    opp = (
        opposite.detach().cpu().numpy()
        if isinstance(opposite, torch.Tensor)
        else np.asarray(opposite)
    ).astype(np.int64, copy=False)
    q = opp.shape[0]
    kernel = np.zeros((1, 1, q, q), dtype=np.float64)  # [kh, kw, q_in, q_out]
    for i in range(q):
        kernel[0, 0, int(opp[i]), i] = 1.0  # out[opp[i]] <- in[i]
    return kernel


# ---------------------------------------------------------------- self-checks
def _tap_index(table: np.ndarray, q_in: int) -> tuple[int, ...] | None:
    block = table[..., q_in, q_in]
    if not np.any(block):
        return None
    return tuple(int(v) for v in np.unravel_index(int(np.argmax(block)), block.shape))


def describe_stream_table(lveloc: Any, table: Any, axes: str = "yx") -> list[dict]:
    """Per-direction audit of a hard-coded STREAM table against its velocity table.

    Returns one record per population with the tap index implied by the
    velocity table (``expected``), the tap index actually stored in ``table``
    (``table``) and the verdict flag ``ok``.  ``table`` is in the upstream
    ``(*[3]*d, q_in, q_out)`` layout.
    """
    lv = _lveloc_array(lveloc)
    tab = np.asarray(table, dtype=np.float64)
    expected = conv_stream_kernel_dense(lv, axes)
    if tab.shape != expected.shape:
        raise ValueError(f"table shape {tab.shape} != expected {expected.shape}")
    records = []
    for i in range(lv.shape[0]):
        exp_idx = _tap_index(expected, i)
        got_idx = _tap_index(tab, i)
        records.append(
            {
                "q": i,
                "c": [int(v) for v in lv[i]],
                "expected": list(exp_idx) if exp_idx is not None else None,
                "table": list(got_idx) if got_idx is not None else None,
                "ok": exp_idx == got_idx,
            }
        )
    return records


def stream_table_mismatches(lveloc: Any, table: Any, axes: str = "yx") -> list[dict]:
    """Only the directions whose stored tap disagrees with the velocity table."""
    return [r for r in describe_stream_table(lveloc, table, axes) if not r["ok"]]


# ------------------------------------------------------------------ operators
def _weight_tensor(
    kernel: np.ndarray, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    return torch.as_tensor(kernel, dtype=dtype, device=device)


def stream_conv_2d(
    f: torch.Tensor,
    *,
    lveloc: Any | None = None,
    kernel: np.ndarray | None = None,
    axes: str = "yx",
) -> torch.Tensor:
    """Fixed-kernel streaming for a 2D field ``f`` of shape ``(q, ny, nx)``.

    Periodic (mobius) padding: one wrapped ghost cell per side, then a
    ``VALID`` cross-correlation, which reproduces the periodic neighbour
    gather exactly.  ``kernel`` overrides the depthwise kernel (used by the
    cross-validation arm that feeds upstream tables in verbatim).
    """
    if f.ndim != 3:
        raise ValueError(f"2D conv-as-streaming expects (q, ny, nx), got {tuple(f.shape)}")
    q = f.shape[0]
    if kernel is None:
        if lveloc is None:
            from .d2q9 import C as lveloc  # local import keeps module import-light
        kernel = conv_stream_kernel(lveloc, axes="yx")
    weight = _weight_tensor(kernel, f.dtype, f.device)
    if tuple(weight.shape[:2]) != (q, 1):
        raise ValueError(f"kernel shape {tuple(weight.shape)} does not fit q={q}")
    padded = F.pad(f.unsqueeze(0), (1, 1, 1, 1), mode="circular")
    return F.conv2d(padded, weight, groups=q).squeeze(0)


def stream_conv_3d(
    f: torch.Tensor,
    *,
    lveloc: Any | None = None,
    kernel: np.ndarray | None = None,
    axes: str = "zyx",
) -> torch.Tensor:
    """Fixed-kernel streaming for a 3D field ``f`` of shape ``(q, nz, ny, nx)``."""
    if f.ndim != 4:
        raise ValueError(f"3D conv-as-streaming expects (q, nz, ny, nx), got {tuple(f.shape)}")
    q = f.shape[0]
    if kernel is None:
        if lveloc is None:
            from .d3q19 import C as lveloc
        kernel = conv_stream_kernel(lveloc, axes="zyx")
    weight = _weight_tensor(kernel, f.dtype, f.device)
    if tuple(weight.shape[:2]) != (q, 1):
        raise ValueError(f"kernel shape {tuple(weight.shape)} does not fit q={q}")
    padded = F.pad(f.unsqueeze(0), (1, 1, 1, 1, 1, 1), mode="circular")  # [1, q, nz, ny, nx]
    return F.conv3d(padded, weight, groups=q).squeeze(0)


def stream_conv(f: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Dispatch to :func:`stream_conv_2d` / :func:`stream_conv_3d` by rank."""
    if f.ndim == 3:
        kwargs.setdefault("axes", "yx")
        return stream_conv_2d(f, **kwargs)
    if f.ndim == 4:
        kwargs.setdefault("axes", "zyx")
        return stream_conv_3d(f, **kwargs)
    raise ValueError(f"conv-as-streaming expects rank 3 or 4, got {f.ndim}")


def bounce_blend(
    f_post_collision: torch.Tensor,
    f_pre_collision: torch.Tensor,
    boundary: torch.Tensor,
    *,
    opposite: Any | None = None,
    kernel: np.ndarray | None = None,
) -> torch.Tensor:
    """Upstream ``CollideSC`` boundary step: 1x1 reversal conv blended by a mask.

    ``f = (1 - m) * f_post + m * conv1x1(f_pre * m, BOUNCE)``

    The reflected populations are taken from the *pre*-collision state (that is
    upstream's choice, see ``LatFlow/Domain.py`` ``CollideSC``), and the wall
    sits on the solid cell face -- i.e. a half-way bounce-back.  ``boundary``
    is 1.0 on solid cells and accepts shapes ``(ny, nx)`` or ``(1, ny, nx)``.
    """
    if boundary.dim() == 2:
        mask = boundary.unsqueeze(0)
    elif boundary.dim() == 3:
        mask = boundary
    else:
        raise ValueError(f"boundary mask must be (ny, nx) or (1, ny, nx), got {tuple(boundary.shape)}")
    mask = mask.to(dtype=f_post_collision.dtype, device=f_post_collision.device)
    q = f_post_collision.shape[0]
    if kernel is None:
        if opposite is None:
            from .d2q9 import OPPOSITE as opposite
        kernel = bounce_kernel_dense(opposite)
    weight = _weight_tensor(kernel, f_post_collision.dtype, f_post_collision.device)
    if tuple(weight.shape[:2]) == (1, 1):  # upstream layout [kh, kw, q_in, q_out]
        weight = weight.permute(3, 2, 0, 1)
    if tuple(weight.shape[:2]) != (q, q):
        raise ValueError(f"bounce kernel shape {tuple(weight.shape)} does not fit q={q}")
    # 1x1 cross-channel convolution (upstream's Op), channel-first in torch
    stacked = (f_pre_collision * mask).unsqueeze(0)  # [1, q, ny, nx]
    reflected = F.conv2d(stacked, weight).squeeze(0)  # [q, ny, nx]
    return f_post_collision * (1.0 - mask) + reflected * mask