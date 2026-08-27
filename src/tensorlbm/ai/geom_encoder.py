"""Learned geometry conditioning via a signed-distance encoder (B4 SDF).

The v3 surrogate (:mod:`tensorlbm.ai.drag_cond`) conditions the FiLM-FNO on
four hand-derived geometry channels — functions of the SUBOFF CAD design
parameters that must be re-derived for every new shape family.  This module
replaces the hand block with a *learned* geometry latent: the solid voxel
mask (the artefact a simulation user supplies anyway) is mapped to a signed
distance field, downsampled, and encoded by a small 3-D CNN whose output
condition the drag regressor consumes.  The encoder trains jointly with the
regressor (end-to-end), so the latent is optimised for drag prediction, and
the latent space doubles as a geometry-distance metric for extrapolation
guardrails (nearest-neighbour distance vs per-point error).

Representation conventions (fixed here, depended on by tests):

- **SDF** — ``phi < 0`` inside the solid, ``phi > 0`` outside, in **voxel
  units** on the simulation lattice, with the ``scipy.ndimage``-equivalent
  discretisation ``phi = edt(~mask) - edt(mask)`` (distance to the nearest
  opposite-phase voxel; the exact EDT — see :func:`signed_distance_field`).
- **Clip + pool** — the full-resolution SDF is clipped to
  ``[-SDF_CLIP_VOXELS, +SDF_CLIP_VOXELS]``, scaled to ``[-1, 1]`` and
  mean-pooled with stride ``SDF_POOL_STRIDE`` (production 64x64x128 ->
  32x32x64).  Clip-then-pool keeps the block bounded and preserves the
  zero level set; pooling of an even-shifted SDF commutes bitwise with the
  shift (pinned in ``tests/test_geom_encoder.py``).
- **Latent** — ``tanh``-bounded in ``[-1, 1]`` per dimension, so it enters
  the condition vector un-normalised (the log-parameter/hand blocks are
  still z-scored on fit statistics by the training script, exactly as v3).

Also provides :class:`SDFCondFNODrag`, the joint encoder + FiLM-FNO model:
a thin composition around :class:`tensorlbm.ai.drag_cond.CondFNODrag` with
the geometry block of the condition vector supplied by the encoder.  The
v3 regressor body, parameter-creation order and training protocol are
untouched.

v2 (B4 SDF-2, 2026-08-25, purely additive — every v1 name above keeps its
behaviour) attacks the v1 latent degeneracy.  Post-mortem on the 238-point
corpus (``/nfs/wangxi/runs/b4_sdf_20260824/latents.npz``): the jointly
trained v1 latent is rank-1 in practice — PC1 carries 99.87-99.9% of the
corpus variance (participation ratio 1.00 for ``sdf_joint`` and
``sdf_joint_wide``) along the sail axis and fins are invisible to a linear
probe.  Two causes, two fixes, same input contract (v1/v2 differ only in
the trunk):

- **Multi-scale readout** — v1 pools one feature map globally, averaging
  thin appendages (fins are ~1 voxel thick on the production lattice,
  sub-cell after pooling) into sub-percent contributions.  v2
  (:class:`SDFEncoderV2`) reads residual-block features at three depths
  and pools each scale by channel mean AND channel max (max keeps
  small-support structures alive), ~253k params at ``base=12`` / ~446k at
  ``base=16`` (v1: 46k).
- **Latent-shape penalty** — nothing in the v1 objective rewards spreading
  information across latent dimensions.  :func:`vicreg_latent_penalty`
  (VICReg-style variance hinge + de-correlation) is offered for the
  ``v2_reg`` arm; :func:`latent_spectrum` / :func:`participation_ratio`
  measure the outcome.
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .drag_cond import CondFNODrag

__all__ = [
    "SDF_CLIP_VOXELS",
    "SDFEncoder",
    "SDFCondFNODrag",
    "SDFEncoderV2",
    "SDFCondFNODragV2",
    "SDF_POOL_STRIDE",
    "VAR_EPS",
    "ResidualBlock3d",
    "condition_sdf_params",
    "latent_spectrum",
    "logit_margin_penalty",
    "participation_ratio",
    "sdf_volume",
    "signed_distance_field",
    "vicreg_latent_penalty",
]

#: Signed distances are clipped to +- this many voxels before pooling.
SDF_CLIP_VOXELS = 8.0

#: Stride-``SDF_POOL_STRIDE`` mean pooling of the clipped SDF.
SDF_POOL_STRIDE = 2

# squared-distance brute force chunking (voxels x targets per tile); kept
# small enough for CPU testing, irrelevant for wall-clock on GPU
_VOXEL_CHUNK = 16384
_TARGET_CHUNK = 1024

#: Sentinel for "no opposite-phase voxel exists" (degenerate all-solid /
#: all-empty masks); squared distance, saturates at the clip after sqrt.
#: Any real squared distance is << 10**9 on simulation lattices.
_NO_BOUNDARY_D2 = 10**9

#: Variance epsilon inside the sqrt of :func:`vicreg_latent_penalty`
#: (VICReg convention): bounds the gradient of the std at zero variance.
VAR_EPS = 1e-4


def _phase_boundaries(mask: Tensor) -> tuple[Tensor, Tensor]:
    """26-connected boundary voxels of each phase.

    Returns ``(solid_boundary, complement_boundary)`` boolean volumes with
    the shape of ``mask``.  ``solid_boundary`` = solid voxels with at least
    one non-solid 26-neighbour; ``complement_boundary`` the mirror image.
    """
    if mask.dtype != torch.bool:
        raise TypeError(f"mask must be boolean, got {mask.dtype}")
    m = mask.to(torch.float32).unsqueeze(0).unsqueeze(0)
    dilated = F.max_pool3d(m, kernel_size=3, stride=1, padding=1).bool()
    eroded = ~F.max_pool3d(1.0 - m, kernel_size=3, stride=1, padding=1).bool()
    return mask & ~eroded, ~mask & dilated


def _min_sq_distance(
    coords: Tensor, targets: Tensor, out: Tensor, v_chunk: int, t_chunk: int
) -> None:
    """In-place ``out = min(out, min_w |coords_v - targets_w|^2)`` (int32).

    ``coords`` (V, 3) query voxels, ``targets`` (T, 3) reference voxels,
    both int32; ``out`` (V,) int32 running minimum.  Integer arithmetic
    makes the minimum reduction exact and order-independent — bitwise
    deterministic on any device.
    """
    for v0 in range(0, coords.shape[0], v_chunk):
        cv = coords[v0 : v0 + v_chunk]
        for t0 in range(0, targets.shape[0], t_chunk):
            tt = targets[t0 : t0 + t_chunk]
            diff = cv.unsqueeze(1) - tt.unsqueeze(0)  # (v, t, 3)
            d2 = (diff * diff).sum(dim=2).min(dim=1).values  # (v,)
            torch.minimum(out[v0 : v0 + v_chunk], d2, out=out[v0 : v0 + v_chunk])


def _nearest_sq_distances(coords: Tensor, boundary: Tensor, v_chunk: int, t_chunk: int) -> Tensor:
    """Squared distance from every voxel in ``coords`` to the boundary set.

    Empty boundary (degenerate single-phase mask) yields the
    ``_NO_BOUNDARY_D2`` sentinel everywhere.
    """
    out = torch.full((coords.shape[0],), _NO_BOUNDARY_D2, dtype=torch.int32, device=coords.device)
    if bool(boundary.any()):
        tgt = coords[boundary.reshape(-1)]
        _min_sq_distance(coords, tgt, out, v_chunk, t_chunk)
    return out


def signed_distance_field(
    mask: Tensor,
    *,
    v_chunk: int = _VOXEL_CHUNK,
    t_chunk: int = _TARGET_CHUNK,
) -> Tensor:
    """Exact signed distance field of a boolean voxel mask.

    ``phi[v] = +dist(v, nearest solid voxel)`` for ``v`` outside the solid,
    ``-dist(v, nearest non-solid voxel)`` for ``v`` inside — voxel units,
    the discrete EDT convention of ``scipy.ndimage.distance_transform_edt``
    composed as ``edt(~mask) - edt(mask)``.

    Exact without scipy: the nearest opposite-phase voxel of any query
    voxel always lies on the 26-connected phase boundary (an interior
    opposite-phase voxel always has a same-phase neighbour one step closer
    to the query), so brute-forcing the *boundary* voxels reproduces the
    full-set minimum exactly.  All reductions run on int32 squared
    distances — deterministic, order-independent, bitwise reproducible on
    any device.

    Degenerate masks (all solid / all empty) have an empty boundary on one
    side; those distances saturate at ``sqrt(_NO_BOUNDARY_D2)`` and are
    clipped by :func:`sdf_volume`.
    """
    if mask.ndim != 3:
        raise ValueError(f"mask must be (D, H, W), got {tuple(mask.shape)}")
    solid_b, comp_b = _phase_boundaries(mask)
    device = mask.device
    d, h, w = mask.shape
    grids = torch.meshgrid(
        torch.arange(d, device=device),
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij",
    )
    coords = torch.stack(grids, dim=-1).reshape(-1, 3).to(torch.int32)
    d2_out = _nearest_sq_distances(coords, solid_b, v_chunk, t_chunk)
    d2_in = _nearest_sq_distances(coords, comp_b, v_chunk, t_chunk)
    flat = mask.reshape(-1)
    dist = torch.sqrt(torch.where(flat, d2_in, d2_out).to(torch.float64)).to(torch.float32)
    return torch.where(flat, -dist, dist).reshape(d, h, w)


def sdf_volume(
    mask: Tensor,
    *,
    clip: float = SDF_CLIP_VOXELS,
    pool: int = SDF_POOL_STRIDE,
    v_chunk: int = _VOXEL_CHUNK,
    t_chunk: int = _TARGET_CHUNK,
) -> Tensor:
    """Clip, scale and mean-pool the signed distance field of ``mask``.

    Returns a ``(1, 1, D', H', W')`` float32 volume in ``[-1, 1]`` — the
    ``(N=1, C=1)`` layout :class:`SDFEncoder` consumes.  ``D' = floor(D /
    pool)`` etc.; production 64x64x128 masks give 32x32x64.
    """
    if not (clip > 0.0 and math.isfinite(clip)):
        raise ValueError(f"clip must be finite and positive, got {clip}")
    if pool < 1:
        raise ValueError(f"pool must be >= 1, got {pool}")
    phi = signed_distance_field(mask, v_chunk=v_chunk, t_chunk=t_chunk)
    phi = phi.clamp(-clip, clip) / clip
    return F.avg_pool3d(phi.unsqueeze(0).unsqueeze(0), kernel_size=pool, stride=pool)


class SDFEncoder(nn.Module):
    """Four stride-2 Conv3d layers + global mean pool -> bounded latent.

    Production input ``(B, 1, 32, 32, 64)`` -> four stride-2 3x3x3 convs
    (base channels 8/16/32/32, GELU) -> ``(B, 32, 2, 2, 4)`` -> global mean
    pool -> linear + ``tanh`` -> ``(B, latent_dim)`` latent in ``[-1, 1]``.
    ~49k parameters at ``base=8`` (spec ceiling ~1M; capacity attribution
    via ``base`` if the latent underperforms).
    """

    def __init__(self, latent_dim: int = 32, *, base: int = 8, in_ch: int = 1) -> None:
        super().__init__()
        if latent_dim < 1 or base < 1 or in_ch < 1:
            raise ValueError("latent_dim, base and in_ch must all be >= 1")
        self.latent_dim = int(latent_dim)
        self.body = nn.Sequential(
            nn.Conv3d(in_ch, base, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv3d(base, base * 2, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv3d(base * 2, base * 4, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv3d(base * 4, base * 4, 3, stride=2, padding=1),
            nn.GELU(),
        )
        self.head = nn.Linear(base * 4, self.latent_dim)

    def forward(self, sdf: Tensor) -> Tensor:
        """``(B, in_ch, D, H, W)`` SDF volume -> ``(B, latent_dim)`` latent."""
        feat = self.body(sdf).mean(dim=(2, 3, 4))
        return torch.tanh(self.head(feat))


class SDFCondFNODrag(nn.Module):
    """Joint SDF-encoder + FiLM-FNO drag regressor (B4 SDF arms).

    The condition vector handed to the inner :class:`CondFNODrag` is
    ``[p, encoder(sdf)]`` where ``p`` is the z-scored
    log-parameter/hand-crafted block supplied by the training script
    (``param_dim + hand_dim`` columns) and the latent is appended raw — it
    is already bounded by the encoder's ``tanh`` head, so no (stale) fit
    statistics are applied to it while it co-adapts with the encoder.
    """

    def __init__(
        self,
        *,
        param_dim: int = 4,
        hand_dim: int = 0,
        latent_dim: int = 32,
        aux_dim: int = 0,
        encoder_base: int = 8,
        in_ch: int = 5,
        width: int = 32,
        n_layers: int = 4,
        modes: tuple[int, int] = (16, 32),
        mlp_hidden: int = 128,
        film_hidden: int = 64,
    ) -> None:
        super().__init__()
        if param_dim < 0 or hand_dim < 0:
            raise ValueError("param_dim and hand_dim must be >= 0")
        if param_dim + hand_dim == 0 and latent_dim < 1:
            raise ValueError("need at least one condition column (param or latent)")
        self.param_dim = int(param_dim)
        self.hand_dim = int(hand_dim)
        self.latent_dim = int(latent_dim)
        self.encoder = SDFEncoder(latent_dim=self.latent_dim, base=encoder_base)
        self.fno = CondFNODrag(
            in_ch=in_ch,
            width=width,
            n_layers=n_layers,
            modes=modes,
            cond_dim=self.param_dim + self.hand_dim + self.latent_dim,
            mlp_hidden=mlp_hidden,
            film_hidden=film_hidden,
            aux_dim=aux_dim,
        )

    def forward(
        self, x: Tensor, sdf: Tensor, p: Tensor, return_aux: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        if p.shape[1] != self.param_dim + self.hand_dim:
            raise ValueError(
                f"p must have {self.param_dim + self.hand_dim} columns, got {p.shape[1]}"
            )
        q = torch.cat([p, self.encoder(sdf)], dim=1)
        out = self.fno(x, q, return_aux=return_aux)
        if return_aux:
            y, aux = cast(tuple[Tensor, Tensor], out)
            return y, aux
        return cast(Tensor, out)


def condition_sdf_params(
    re: np.ndarray,
    u_in: np.ndarray,
    sail_scale: np.ndarray,
    fin_scale: np.ndarray,
    geometry: np.ndarray | None = None,
) -> np.ndarray:
    """Pre-normalisation condition block for the SDF arms.

    ``[log10 re, log10 u_in, log10 sail, log10 fin]`` plus, when
    ``geometry`` is given, the v3 4-channel hand-crafted block appended
    (``sdf_plus_hand`` arm).  The training script z-scores these columns on
    fit statistics exactly as v3 does; the learned latent is appended
    inside :class:`SDFCondFNODrag` (bounded, un-normalised).
    """
    re = np.asarray(re, dtype=np.float64)
    u_in = np.asarray(u_in, dtype=np.float64)
    sail_scale = np.asarray(sail_scale, dtype=np.float64)
    fin_scale = np.asarray(fin_scale, dtype=np.float64)
    n = len(re)
    block = np.stack(
        [np.log10(re), np.log10(u_in), np.log10(sail_scale), np.log10(fin_scale)], axis=1
    )
    if geometry is None:
        return block
    geometry = np.asarray(geometry, dtype=np.float64)
    if geometry.shape != (n, 4):
        raise ValueError(f"geometry block must be ({n}, 4), got {geometry.shape}")
    return np.concatenate([block, geometry], axis=1)


# ---------------------------------------------------------------------------
# v2 (B4 SDF-2): multi-scale residual trunk + latent-shape utilities
# ---------------------------------------------------------------------------


class ResidualBlock3d(nn.Module):
    """Stride-``stride`` residual block: ``gelu(conv(gelu(conv(x))) + skip(x))``.

    The skip is the identity when shape and channels are preserved, else a
    1x1x1 conv at the block stride (projection shortcut).
    """

    def __init__(self, in_ch: int, out_ch: int, *, stride: int = 1) -> None:
        super().__init__()
        if in_ch < 1 or out_ch < 1 or stride < 1:
            raise ValueError("in_ch, out_ch and stride must all be >= 1")
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, stride=1, padding=1)
        self.skip: nn.Module = (
            nn.Identity()
            if (stride == 1 and in_ch == out_ch)
            else nn.Conv3d(in_ch, out_ch, 1, stride=stride)
        )

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv2(F.gelu(self.conv1(x)))
        return F.gelu(y + self.skip(x))


class SDFEncoderV2(nn.Module):
    """Multi-scale residual trunk for the pooled SDF (v2 encoder).

    Input contract identical to :class:`SDFEncoder` — ``(B, in_ch, D, H,
    W)`` pooled SDF in ``[-1, 1]`` (production ``(B, 1, 32, 32, 64)``), so
    v1/v2 A/B runs differ only in the trunk.  Differences from v1:

    - residual blocks instead of a bare conv stack, 3 stride-2 stages
      (32x32x64 -> 16x16x32 -> 8x8x16 -> 4x4x8);
    - features read at ALL THREE depths, each pooled per channel by mean
      AND max (max survives structures with tiny spatial support, e.g. the
      ~1-voxel-thick stern fins);
    - wider trunk: ~253k parameters at ``base=12``, ~446k at ``base=16``
      (v1: 46,288 at ``base=8``) — spec ceiling ~500k.

    Output: ``(B, latent_dim)`` in ``[-1, 1]`` (``tanh`` head, like v1).
    """

    def __init__(self, latent_dim: int = 32, *, base: int = 12, in_ch: int = 1) -> None:
        super().__init__()
        if latent_dim < 1 or base < 1 or in_ch < 1:
            raise ValueError("latent_dim, base and in_ch must all be >= 1")
        self.latent_dim = int(latent_dim)
        self.stem = nn.Sequential(nn.Conv3d(in_ch, base, 3, stride=1, padding=1), nn.GELU())
        self.scale1 = ResidualBlock3d(base, base * 2, stride=2)
        self.scale2 = ResidualBlock3d(base * 2, base * 4, stride=2)
        self.scale3 = ResidualBlock3d(base * 4, base * 4, stride=2)
        self.head = nn.Linear((base * 2 + base * 4 + base * 4) * 2, self.latent_dim)

    def scale_features(self, sdf: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Per-scale pooled features ``(B, 4*base)``, ``(B, 8*base)``, ``(B, 8*base)``.

        Each scale contributes the channel mean and the channel max of its
        feature map (mean = coarse hull form, max = strongest local
        response, which thin appendages can still win at shallow depth).
        """
        x1 = self.scale1(self.stem(sdf))
        x2 = self.scale2(x1)
        x3 = self.scale3(x2)

        def pooled(x: Tensor) -> Tensor:
            return torch.cat([x.mean(dim=(2, 3, 4)), x.amax(dim=(2, 3, 4))], dim=1)

        return pooled(x1), pooled(x2), pooled(x3)

    def forward(self, sdf: Tensor) -> Tensor:
        """``(B, in_ch, D, H, W)`` SDF volume -> ``(B, latent_dim)`` latent."""
        z, _ = self.forward_with_logits(sdf)
        return z

    def forward_with_logits(self, sdf: Tensor) -> tuple[Tensor, Tensor]:
        """``(latent, pre-tanh logits)`` — logits exposed for the v2_reg2 arm.

        Joint training on the 350-pt corpus saturates the ``tanh`` head to a
        corpus-CONSTANT +-1 latent (rank-0 collapse; the FiLM-FNO reads the
        geometry from the solid-mask channel of its field input, so nothing
        else in the objective keeps the latent alive, and once saturated the
        ``tanh`` gradient ~ ``1 - z^2`` is too small to recover).  A logit-
        margin penalty on these logits (keep |logits| <= ~2) keeps the head
        in its responsive range; see :func:`logit_margin_penalty`.
        """
        f1, f2, f3 = self.scale_features(sdf)
        logits = self.head(torch.cat([f1, f2, f3], dim=1))
        return torch.tanh(logits), logits


class SDFCondFNODragV2(nn.Module):
    """v2 joint model — :class:`SDFEncoderV2` + the unchanged v3 FiLM-FNO body.

    Same contract and condition assembly as :class:`SDFCondFNODrag`
    (``[z-scored p | raw bounded latent]``); only the encoder trunk differs.
    For the ``v2_reg`` arm the training script adds
    :func:`vicreg_latent_penalty` ``(encoder(sdf))`` to the loss — the
    module itself stays loss-agnostic.
    """

    def __init__(
        self,
        *,
        param_dim: int = 4,
        hand_dim: int = 0,
        latent_dim: int = 32,
        aux_dim: int = 0,
        encoder_base: int = 12,
        in_ch: int = 5,
        width: int = 32,
        n_layers: int = 4,
        modes: tuple[int, int] = (16, 32),
        mlp_hidden: int = 128,
        film_hidden: int = 64,
    ) -> None:
        super().__init__()
        if param_dim < 0 or hand_dim < 0:
            raise ValueError("param_dim and hand_dim must be >= 0")
        if param_dim + hand_dim == 0 and latent_dim < 1:
            raise ValueError("need at least one condition column (param or latent)")
        self.param_dim = int(param_dim)
        self.hand_dim = int(hand_dim)
        self.latent_dim = int(latent_dim)
        self.encoder = SDFEncoderV2(latent_dim=self.latent_dim, base=encoder_base)
        self.fno = CondFNODrag(
            in_ch=in_ch,
            width=width,
            n_layers=n_layers,
            modes=modes,
            cond_dim=self.param_dim + self.hand_dim + self.latent_dim,
            mlp_hidden=mlp_hidden,
            film_hidden=film_hidden,
            aux_dim=aux_dim,
        )

    def forward(
        self, x: Tensor, sdf: Tensor, p: Tensor, return_aux: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        if p.shape[1] != self.param_dim + self.hand_dim:
            raise ValueError(
                f"p must have {self.param_dim + self.hand_dim} columns, got {p.shape[1]}"
            )
        q = torch.cat([p, self.encoder(sdf)], dim=1)
        out = self.fno(x, q, return_aux=return_aux)
        if return_aux:
            y, aux = cast(tuple[Tensor, Tensor], out)
            return y, aux
        return cast(Tensor, out)

    def latent_pair(self, sdf: Tensor) -> tuple[Tensor, Tensor]:
        """``(latent, pre-tanh logits)`` of the v2 encoder (v2_reg2 arm)."""
        return self.encoder.forward_with_logits(sdf)


def logit_margin_penalty(logits: Tensor, *, margin: float = 2.0) -> Tensor:
    """Soft ``|logits| <= margin`` guard for the pre-``tanh`` head output.

    ``mean(relu(|logits| - margin)^2)`` — exactly zero inside the margin,
    quadratic outside.  Keeps ``tanh`` in its responsive range (at
    ``|logit| = 2``, ``tanh' = 1 - tanh(2)^2 ~= 0.07``, vs ~1e-9 at 5)
    so a variance/de-correlation penalty on the latent can still push
    through the head after the rank-0 saturation observed in v2/v2_reg.
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be (B, D), got {tuple(logits.shape)}")
    if margin <= 0.0:
        raise ValueError(f"margin must be positive, got {margin}")
    excess = torch.relu(logits.abs() - margin)
    return (excess * excess).mean()


def vicreg_latent_penalty(z: Tensor, *, var_target: float = 1.0, gamma: float = 1.0) -> Tensor:
    """VICReg-style latent-shape penalty of a batch (variance hinge + de-correlation).

    ``penalty = mean_c relu(var_target - std(z_c)) + gamma * ||corr(z)||_off^2 / d``

    computed per column of the ``(B, d)`` batch ``z``: the hinge keeps every
    latent dimension's batch std at or above ``var_target`` (fights
    dimension collapse), the correlation term penalises redundant
    directions (fights the rank-1 degeneracy observed in v1).  Exactly
    zero for a batch with uncorrelated unit-std columns; large for
    collapsed / rank-1 latents.  Deterministic (fixed reduction order).

    Numerics: the std is ``sqrt(var + VAR_EPS)`` (the VICReg convention).
    A plain ``torch.std`` has an unbounded ``d(std)/dz`` as a column's
    batch variance -> 0, and a fully-saturated ``tanh`` column (every row
    exactly +-1) has variance exactly 0 — the un-eps'd version produced
    inf gradients and NaN weights late in v2_reg training.
    """
    if z.ndim != 2:
        raise ValueError(f"z must be (B, D), got {tuple(z.shape)}")
    if z.shape[0] < 2:
        raise ValueError("need >= 2 rows to compute batch statistics")
    if var_target <= 0.0:
        raise ValueError(f"var_target must be positive, got {var_target}")
    if gamma < 0.0:
        raise ValueError(f"gamma must be >= 0, got {gamma}")
    std = torch.sqrt(z.var(dim=0, unbiased=True) + VAR_EPS)
    var_term = torch.relu(var_target - std).mean()
    zc = z - z.mean(dim=0, keepdim=True)
    d = z.shape[1]
    cov = (zc.T @ zc) / (zc.shape[0] - 1)
    inv_std = 1.0 / torch.clamp(torch.sqrt(torch.diagonal(cov) + VAR_EPS), min=1e-6)
    corr = cov * inv_std.unsqueeze(0) * inv_std.unsqueeze(1)
    off = corr - torch.diag_embed(torch.diagonal(corr))
    cov_term = (off * off).sum() / d
    return var_term + gamma * cov_term


def latent_spectrum(z: np.ndarray, *, n_components: int | None = None) -> np.ndarray:
    """PCA explained-variance ratios of a latent corpus, in descending order.

    ``z`` is an ``(N, D)`` corpus (float, any sign); returns the spectrum
    ``lambda_k / sum(lambda)`` of the singular values of the mean-centred
    corpus, truncated to ``n_components`` when given.  Pure numpy (float64,
    deterministic).
    """
    a = np.asarray(z, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] < 2:
        raise ValueError(f"z must be (N, D) with N >= 2, got {a.shape}")
    if n_components is not None and n_components < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components}")
    zc = a - a.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(zc, compute_uv=False)
    lam = sv * sv
    if lam.sum() <= 0.0:
        raise ValueError("degenerate corpus: zero total variance")
    ev = lam / lam.sum()
    return ev if n_components is None else ev[:n_components]


def participation_ratio(z: np.ndarray) -> float:
    """Participation ratio of a latent corpus: ``(sum lambda)^2 / sum lambda^2``.

    1.0 for a rank-1 corpus, ``D`` for an isotropic ``D``-dimensional one;
    the v2 target is >= 8 effective dimensions on the 350-point corpus
    (v1 measured 1.00).
    """
    ev = latent_spectrum(z)
    return float(1.0 / float((ev * ev).sum()))
