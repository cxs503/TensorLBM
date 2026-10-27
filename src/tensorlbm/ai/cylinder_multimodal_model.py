"""Patchified multimodal flow-force transformer for 2-D cylinder-flow forecasting.

Combines four complementary modalities produced by the cylinder-flow solver:

1. **Physical field modality**: normalized ``(ux*, uy*, drho*)`` plus local
   spatial gradients ``(dux/dx, dux/dy, duy/dx, duy/dy)`` tokenized into 2-D
   patches so grids such as ``160 x 64`` (`640` patches at ``4 x 4``) or
   ``320 x 100`` (`2000` patches) stay well below dense cell-wise attention
   limits.
2. **Geometry modality**: ``(solid_mask, normalized_sdf)`` fused both locally
   inside each spatial patch and globally via a multi-scale 2-D SDF encoder
   with mean + max pooling (preserving small cylinder radii).
3. **Physics & aerodynamic force modality**: dimensionless regime vector
   ``(log10(Re), u_in, tau - 0.5, r / ny, ny / nx, dt / 100)`` and optional
   current force ``(Cd_t, Cl_t)``.
4. **Visual frame modality**: optional rendered visualization
   ``flow_step_XXXXXX.png`` encoded by a lightweight 2-D CNN with modality
   dropout during training and a cross-modal field-vision alignment head,
   allowing zero-PNG inference inside solver loops.

At epoch 0, residual field and force output projections are zero-initialized so
the forecaster starts at the exact persistence baseline ``Y_{t+dt} = Y_t`` and
learns physical time increments governed by fluid-masked field, vorticity,
divergence-consistency, aerodynamic force, and cross-modal alignment losses.
"""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader

from .cylinder_multimodal_data import (
    CylinderMultimodalDataset,
    CylinderNormalization,
    CylinderTrajectoryRecord,
    MultimodalDatasetManifest,
    collate_cylinder_multimodal,
)

if TYPE_CHECKING:
    from tensorlbm.ml.training_job import TrainingJobRegistry

__all__ = [
    "MULTIMODAL_MODEL_FAMILY",
    "MultimodalCylinderArch",
    "MultimodalCylinderOutput",
    "MultimodalCylinderForecaster",
    "MultimodalTrainConfig",
    "MultimodalTrainResult",
    "ForecastEvaluationReport",
    "RolloutStepMetrics",
    "compute_vorticity_batch",
    "compute_divergence_batch",
    "train_multimodal_cylinder_forecaster",
    "evaluate_forecaster_vs_persistence",
    "rollout_trajectory_forecast",
    "save_multimodal_forecaster",
    "load_multimodal_forecaster",
]

MULTIMODAL_MODEL_FAMILY = "cylinder_multimodal_patch_transformer_v1"


# ---------------------------------------------------------------------------
# Differential operators matching cylinder_flow.py conventions
# ---------------------------------------------------------------------------


def compute_vorticity_batch(ux: Tensor, uy: Tensor) -> Tensor:
    """Central-difference z-vorticity ``duy/dx - dux/dy`` on ``(B, ny, nx)``."""
    dux_dy = torch.zeros_like(ux)
    duy_dx = torch.zeros_like(uy)
    dux_dy[:, 1:-1, :] = 0.5 * (ux[:, 2:, :] - ux[:, :-2, :])
    duy_dx[:, :, 1:-1] = 0.5 * (uy[:, :, 2:] - uy[:, :, :-2])
    return duy_dx - dux_dy


def compute_divergence_batch(ux: Tensor, uy: Tensor) -> Tensor:
    """Central-difference velocity divergence ``dux/dx + duy/dy`` on ``(B, ny, nx)``."""
    dux_dx = torch.zeros_like(ux)
    duy_dy = torch.zeros_like(uy)
    dux_dx[:, :, 1:-1] = 0.5 * (ux[:, :, 2:] - ux[:, :, :-2])
    duy_dy[:, 1:-1, :] = 0.5 * (uy[:, 2:, :] - uy[:, :-2, :])
    return dux_dx + duy_dy


def _spatial_velocity_gradients(field: Tensor) -> Tensor:
    """Compute central-difference ``(dux_dx, dux_dy, duy_dx, duy_dy)`` for ``(B, 3, ny, nx)``."""
    ux = field[:, 0]
    uy = field[:, 1]
    dux_dx = torch.zeros_like(ux)
    dux_dy = torch.zeros_like(ux)
    duy_dx = torch.zeros_like(uy)
    duy_dy = torch.zeros_like(uy)
    dux_dx[:, :, 1:-1] = 0.5 * (ux[:, :, 2:] - ux[:, :, :-2])
    dux_dy[:, 1:-1, :] = 0.5 * (ux[:, 2:, :] - ux[:, :-2, :])
    duy_dx[:, :, 1:-1] = 0.5 * (uy[:, :, 2:] - uy[:, :, :-2])
    duy_dy[:, 1:-1, :] = 0.5 * (uy[:, 2:, :] - uy[:, :-2, :])
    return torch.stack([dux_dx, dux_dy, duy_dx, duy_dy], dim=1)


# ---------------------------------------------------------------------------
# Multimodal encoders & AdaLN-Zero patch transformer blocks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MultimodalCylinderArch:
    """Architecture hyper-parameters for :class:`MultimodalCylinderForecaster`."""

    patch_size: tuple[int, int] = (4, 4)
    history_steps: int = 1
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 3
    ffn_dim: int = 192
    geom_latent_dim: int = 16
    vis_latent_dim: int = 16
    cond_dim: int = 64
    align_dim: int = 32
    pos_grid_size: tuple[int, int] = (16, 40)
    dropout: float = 0.0

    def __post_init__(self) -> None:
        py, px = int(self.patch_size[0]), int(self.patch_size[1])
        gy, gx = int(self.pos_grid_size[0]), int(self.pos_grid_size[1])
        hist = int(self.history_steps)
        if py < 1 or px < 1:
            raise ValueError(f"patch_size must be positive, got {self.patch_size}")
        if hist < 1:
            raise ValueError(f"history_steps must be >= 1, got {self.history_steps}")
        if gy < 1 or gx < 1:
            raise ValueError(f"pos_grid_size must be positive, got {self.pos_grid_size}")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        object.__setattr__(self, "patch_size", (py, px))
        object.__setattr__(self, "history_steps", hist)
        object.__setattr__(self, "pos_grid_size", (gy, gx))


class GeometrySDFEncoder2D(nn.Module):
    """Multi-scale 2-D geometry encoder reading ``(solid_mask, sdf_norm)``."""

    def __init__(self, out_dim: int, base_channels: int = 16) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        self.stage1 = nn.Sequential(
            nn.Conv2d(2, c1, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.stage2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(c2, c2, kernel_size=3, padding=1),
            nn.GELU(),
        )
        # Multi-scale mean + max readout preserves small obstacle features
        self.proj = nn.Linear(2 * (c1 + c2), out_dim)

    def forward(self, geom: Tensor) -> Tensor:
        f1 = self.stage1(geom)
        f2 = self.stage2(f1)
        s1 = torch.cat([f1.mean(dim=(-2, -1)), f1.amax(dim=(-2, -1))], dim=-1)
        s2 = torch.cat([f2.mean(dim=(-2, -1)), f2.amax(dim=(-2, -1))], dim=-1)
        return torch.tanh(self.proj(torch.cat([s1, s2], dim=-1)))


class VisionFrameEncoder2D(nn.Module):
    """Lightweight CNN encoder for rendered flow-visualization frames."""

    def __init__(self, out_dim: int, base_channels: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((2, 4)),
            nn.Flatten(),
            nn.Linear(base_channels * 2 * 8, out_dim),
        )

    def forward(self, visual: Tensor) -> Tensor:
        return torch.tanh(self.net(visual))


class _AdaLNTransformerBlock(nn.Module):
    """Pre-LN Transformer block with AdaLN-Zero conditioning from multimodal vector ``c``."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_dim: int,
        cond_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.ada_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * d_model),
        )
        nn.init.zeros_(self.ada_mod[-1].weight)  # type: ignore[arg-type]
        nn.init.zeros_(self.ada_mod[-1].bias)  # type: ignore[arg-type]

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.ada_mod(cond).unsqueeze(1).chunk(6, dim=-1)
        )
        h1 = self.norm1(x) * (1.0 + scale_msa) + shift_msa
        attn_out, _ = self.attn(h1, h1, h1, need_weights=False)
        x = x + (1.0 + gate_msa) * attn_out

        h2 = self.norm2(x) * (1.0 + scale_mlp) + shift_mlp
        x = x + (1.0 + gate_mlp) * self.mlp(h2)
        return x


@dataclass(frozen=True, slots=True)
class MultimodalCylinderOutput:
    """Outputs of :class:`MultimodalCylinderForecaster` for one forward pass."""

    field_next_norm: Tensor
    delta_field_norm: Tensor
    force_next: Tensor
    field_proj: Tensor
    visual_proj: Tensor


class MultimodalCylinderForecaster(nn.Module):
    """Patchified multimodal Transformer forecasting ``(ux*, uy*, drho*)`` and ``(Cd, Cl)``."""

    def __init__(self, arch: MultimodalCylinderArch | None = None) -> None:
        super().__init__()
        self.arch = arch or MultimodalCylinderArch()
        py, px = self.arch.patch_size
        hist = self.arch.history_steps
        d_model = self.arch.d_model

        self.geom_encoder = GeometrySDFEncoder2D(self.arch.geom_latent_dim)
        self.vision_encoder = VisionFrameEncoder2D(self.arch.vis_latent_dim)

        # Physics(6) + GeomLatent + VisLatent + vis_flag(1) + hist_forces(2*H) + force_flag(1)
        raw_cond_in = 6 + self.arch.geom_latent_dim + self.arch.vis_latent_dim + 1 + 2 * hist + 1
        self.cond_mlp = nn.Sequential(
            nn.Linear(raw_cond_in, self.arch.cond_dim),
            nn.SiLU(),
            nn.Linear(self.arch.cond_dim, self.arch.cond_dim),
        )

        # Local conv stem over history fields(3*H) + temporal diffs(3*(H-1)) + geom(2) + grads(4)
        in_ch = 3 * hist + 3 * max(hist - 1, 0) + 2 + 4
        stem_ch = max(d_model // 2, 16)
        self.spatial_stem = nn.Sequential(
            nn.Conv2d(in_ch, stem_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(stem_ch, stem_ch, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.patch_embed = nn.Conv2d(
            stem_ch,
            d_model,
            kernel_size=(py, px),
            stride=(py, px),
        )

        gy, gx = self.arch.pos_grid_size
        self.pos_embed_2d = nn.Parameter(torch.randn(1, d_model, gy, gx) * 0.02)

        self.blocks = nn.ModuleList(
            [
                _AdaLNTransformerBlock(
                    d_model=d_model,
                    n_heads=self.arch.n_heads,
                    ffn_dim=self.arch.ffn_dim,
                    cond_dim=self.arch.cond_dim,
                    dropout=self.arch.dropout,
                )
                for _ in range(self.arch.n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

        # Unpatchify + local 3x3 smoothing + zero-initialized residual field head
        self.unpatch_proj = nn.Linear(d_model, stem_ch * py * px)
        self.refine_conv = nn.Sequential(
            nn.Conv2d(stem_ch + in_ch, stem_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(stem_ch, 3, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.refine_conv[-1].weight)  # type: ignore[arg-type]
        nn.init.zeros_(self.refine_conv[-1].bias)  # type: ignore[arg-type]

        # Force readout: boundary-weighted + global token pooling + conditioning
        force_in = 2 * d_model + self.arch.cond_dim
        self.delta_force_head = nn.Sequential(
            nn.Linear(force_in + 2 * hist, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )
        nn.init.zeros_(self.delta_force_head[-1].weight)  # type: ignore[arg-type]
        nn.init.zeros_(self.delta_force_head[-1].bias)  # type: ignore[arg-type]

        self.direct_force_head = nn.Sequential(
            nn.Linear(force_in, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )

        # Cross-modal alignment heads
        self.field_align_proj = nn.Linear(d_model, self.arch.align_dim)
        self.vis_align_proj = nn.Linear(self.arch.vis_latent_dim, self.arch.align_dim)

    def _interpolated_pos_embed(self, py_count: int, px_count: int) -> Tensor:
        pos: Tensor = self.pos_embed_2d
        if (pos.shape[-2], pos.shape[-1]) != (py_count, px_count):
            pos = F.interpolate(
                pos,
                size=(py_count, px_count),
                mode="bilinear",
                align_corners=False,
            )
        return pos.flatten(2).transpose(1, 2)  # (1, py_count * px_count, d_model)

    def forward(
        self,
        field_t: Tensor,
        geom: Tensor,
        physics_cond: Tensor,
        *,
        history_fields: Tensor | None = None,
        history_forces: Tensor | None = None,
        force_t: Tensor | None = None,
        has_force: Tensor | None = None,
        visual_t: Tensor | None = None,
        has_visual: Tensor | None = None,
    ) -> MultimodalCylinderOutput:
        if field_t.ndim != 4 or field_t.shape[1] != 3:
            raise ValueError(f"Expected field_t (B, 3, ny, nx), got {tuple(field_t.shape)}")
        if geom.ndim != 4 or geom.shape[1] != 2 or geom.shape[2:] != field_t.shape[2:]:
            raise ValueError(
                f"Expected geom (B, 2, ny, nx) matching field_t, got {tuple(geom.shape)}"
            )

        bsz, _, ny, nx = field_t.shape
        hist = self.arch.history_steps
        device = field_t.device
        dtype = field_t.dtype

        if history_fields is None:
            hist_f = field_t.unsqueeze(1).expand(bsz, hist, 3, ny, nx).contiguous()
        else:
            if (
                history_fields.ndim != 5
                or history_fields.shape[2] != 3
                or history_fields.shape[3:] != (ny, nx)
            ):
                raise ValueError(
                    f"Expected history_fields (B, H, 3, ny, nx), got {tuple(history_fields.shape)}"
                )
            if history_fields.shape[1] >= hist:
                hist_f = history_fields[:, -hist:].contiguous()
            else:
                pad_n = hist - int(history_fields.shape[1])
                pad = history_fields[:, :1].expand(bsz, pad_n, 3, ny, nx)
                hist_f = torch.cat([pad, history_fields], dim=1).contiguous()
        ref_field = hist_f[:, -1]

        z_geom = self.geom_encoder(geom)

        if visual_t is not None and has_visual is not None and bool(has_visual.any()):
            vis_mask = has_visual.to(device=device, dtype=dtype).unsqueeze(-1)
            z_vis_raw = self.vision_encoder(visual_t)
            z_vis = z_vis_raw * vis_mask
        else:
            vis_mask = torch.zeros((bsz, 1), device=device, dtype=dtype)
            z_vis = torch.zeros((bsz, self.arch.vis_latent_dim), device=device, dtype=dtype)

        if history_forces is not None or force_t is not None:
            if has_force is None:
                f_mask_bool = torch.ones((bsz,), device=device, dtype=torch.bool)
            else:
                f_mask_bool = has_force.to(device=device, dtype=torch.bool)
            f_mask = f_mask_bool.to(dtype=dtype).unsqueeze(-1)
            if history_forces is not None:
                if history_forces.shape[1] >= hist:
                    raw_hf = history_forces[:, -hist:]
                else:
                    pad_fn = hist - int(history_forces.shape[1])
                    raw_hf = torch.cat(
                        [history_forces[:, :1].expand(bsz, pad_fn, 2), history_forces],
                        dim=1,
                    )
            else:
                assert force_t is not None
                raw_hf = force_t.unsqueeze(1).expand(bsz, hist, 2)
            gated_hf = raw_hf * f_mask.unsqueeze(-1)
        else:
            f_mask_bool = torch.zeros((bsz,), device=device, dtype=torch.bool)
            f_mask = torch.zeros((bsz, 1), device=device, dtype=dtype)
            gated_hf = torch.zeros((bsz, hist, 2), device=device, dtype=dtype)

        gated_hf_asinh = torch.asinh(gated_hf)
        gated_hf_flat = gated_hf_asinh.reshape(bsz, 2 * hist)
        last_force_asinh = gated_hf_asinh[:, -1]
        cond_in = torch.cat(
            [physics_cond, z_geom, z_vis, vis_mask, gated_hf_flat, f_mask],
            dim=-1,
        )
        cond = self.cond_mlp(cond_in)

        # Pad spatial dimensions to multiples of patch_size if needed
        py, px = self.arch.patch_size
        pad_y = (py - (ny % py)) % py
        pad_x = (px - (nx % px)) % px
        grads = _spatial_velocity_gradients(ref_field)
        spatial_parts = [hist_f.reshape(bsz, 3 * hist, ny, nx)]
        if hist > 1:
            temp_diffs = (hist_f[:, 1:] - hist_f[:, :-1]).reshape(bsz, 3 * (hist - 1), ny, nx)
            spatial_parts.append(temp_diffs)
        spatial_parts.extend([geom, grads])
        spatial_in = torch.cat(spatial_parts, dim=1)

        if pad_y > 0 or pad_x > 0:
            spatial_padded = F.pad(spatial_in, (0, pad_x, 0, pad_y), mode="replicate")
            geom_padded = F.pad(geom, (0, pad_x, 0, pad_y), mode="replicate")
        else:
            spatial_padded = spatial_in
            geom_padded = geom

        stem_feat = self.spatial_stem(spatial_padded)
        patch_grid = self.patch_embed(stem_feat)  # (B, d_model, Py, Px)
        py_count, px_count = int(patch_grid.shape[2]), int(patch_grid.shape[3])

        tokens = patch_grid.flatten(2).transpose(1, 2)
        tokens = tokens + self._interpolated_pos_embed(py_count, px_count)

        for block in self.blocks:
            tokens = block(tokens, cond)
        tokens = self.final_norm(tokens)

        # Unpatchify to padded spatial grid, refine with skip connection, and crop
        ny_pad, nx_pad = py_count * py, px_count * px
        stem_ch = stem_feat.shape[1]
        unpatched = self.unpatch_proj(tokens).view(bsz, py_count, px_count, stem_ch, py, px)
        unpatched = (
            unpatched.permute(0, 3, 1, 4, 2, 5).contiguous().view(bsz, stem_ch, ny_pad, nx_pad)
        )
        delta_padded = self.refine_conv(torch.cat([unpatched, spatial_padded], dim=1))
        delta_field = delta_padded[:, :, :ny, :nx]

        # Enforce zero velocity inside solid obstacle cells
        solid_bool = geom[:, 0] > 0.5
        fluid_float = (~solid_bool).to(dtype=dtype)
        field_raw_pred = ref_field + delta_field
        ux_pred = field_raw_pred[:, 0] * fluid_float
        uy_pred = field_raw_pred[:, 1] * fluid_float
        drho_pred = field_raw_pred[:, 2]
        field_next_norm = torch.stack([ux_pred, uy_pred, drho_pred], dim=1)
        effective_delta = field_next_norm - ref_field

        # Near-obstacle fluid pooling for aerodynamic force prediction
        sdf_patches = F.avg_pool2d(geom_padded[:, 1:2], kernel_size=(py, px), stride=(py, px))
        fluid_patches = F.avg_pool2d(
            (geom_padded[:, 0:1] <= 0.5).to(dtype=dtype),
            kernel_size=(py, px),
            stride=(py, px),
        )
        near_wall_w = (torch.exp(-torch.clamp(sdf_patches, min=0.0) * 4.0) * fluid_patches).flatten(
            1
        )
        near_wall_w = near_wall_w / near_wall_w.sum(dim=1, keepdim=True).clamp_min(1e-6)
        near_pool = torch.bmm(near_wall_w.unsqueeze(1), tokens).squeeze(1)
        global_pool = tokens.mean(dim=1)

        force_feat = torch.cat([near_pool, global_pool, cond], dim=-1)
        delta_force_asinh = self.delta_force_head(torch.cat([force_feat, gated_hf_flat], dim=-1))
        direct_force_asinh = self.direct_force_head(force_feat)
        force_next_asinh = torch.where(
            f_mask_bool.unsqueeze(-1),
            last_force_asinh + delta_force_asinh,
            direct_force_asinh,
        )
        force_next = torch.sinh(force_next_asinh.clamp(-10.0, 10.0))

        field_proj = F.normalize(self.field_align_proj(global_pool), dim=-1, eps=1e-6)
        visual_proj = F.normalize(self.vis_align_proj(z_vis), dim=-1, eps=1e-6)

        return MultimodalCylinderOutput(
            field_next_norm=field_next_norm,
            delta_field_norm=effective_delta,
            force_next=force_next,
            field_proj=field_proj,
            visual_proj=visual_proj,
        )


# ---------------------------------------------------------------------------
# Training configuration, physics-aware multi-task loss, and registry hook
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MultimodalTrainConfig:
    """Hyper-parameters for :func:`train_multimodal_cylinder_forecaster`."""

    epochs: int = 30
    batch_size: int = 8
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    rho_weight: float = 0.25
    vorticity_weight: float = 0.25
    divergence_weight: float = 0.1
    force_weight: float = 0.05
    vision_align_weight: float = 0.02
    visual_dropout_prob: float = 0.3
    force_input_dropout_prob: float = 0.2
    pushforward_steps: int = 1
    pushforward_decay: float = 0.7
    pushforward_detach_prob: float = 0.5
    gradient_clip_norm: float | None = 1.0
    seed: int = 0
    device: str = "cpu"


@dataclass(frozen=True, slots=True)
class MultimodalTrainResult:
    """Completed multimodal training summary and loss history."""

    model: MultimodalCylinderForecaster
    train_losses: tuple[float, ...]
    val_losses: tuple[float, ...]
    best_epoch: int
    best_val_loss: float
    train_wall_seconds: float
    job_id: str | None = None
    registered_model_id: int | None = None


def _compute_multimodal_loss(
    out: MultimodalCylinderOutput,
    *,
    target_norm: Tensor,
    solid_mask: Tensor,
    target_force: Tensor,
    active_visual: Tensor,
    cfg: MultimodalTrainConfig,
) -> tuple[Tensor, dict[str, float]]:
    fluid_mask = ~solid_mask
    fluid_f = fluid_mask.to(dtype=target_norm.dtype).unsqueeze(1)
    fluid_cells = fluid_f.sum().clamp_min(1.0)
    vel_sq = (out.field_next_norm[:, :2] - target_norm[:, :2]).pow(2) * fluid_f
    loss_vel = vel_sq.sum() / (fluid_cells * 2.0)

    rho_sq = (out.field_next_norm[:, 2:3] - target_norm[:, 2:3]).pow(2) * fluid_f
    loss_rho = rho_sq.sum() / fluid_cells
    loss_field = loss_vel + cfg.rho_weight * loss_rho

    # Interior fluid mask for central-difference derivatives
    interior_fluid = fluid_mask[:, 1:-1, 1:-1].to(dtype=target_norm.dtype)
    int_denom = interior_fluid.sum().clamp_min(1.0)

    vort_pred = compute_vorticity_batch(out.field_next_norm[:, 0], out.field_next_norm[:, 1])[
        :, 1:-1, 1:-1
    ]
    vort_true = compute_vorticity_batch(target_norm[:, 0], target_norm[:, 1])[:, 1:-1, 1:-1]
    loss_vort = ((vort_pred - vort_true).pow(2) * interior_fluid).sum() / int_denom

    div_pred = compute_divergence_batch(out.field_next_norm[:, 0], out.field_next_norm[:, 1])[
        :, 1:-1, 1:-1
    ]
    div_true = compute_divergence_batch(target_norm[:, 0], target_norm[:, 1])[:, 1:-1, 1:-1]
    loss_div = ((div_pred - div_true).pow(2) * interior_fluid).sum() / int_denom

    # Supervise forces in asinh space: linear near zero (Cl, steady Cd) and
    # logarithmic on impulsive acoustic startup spikes so field gradients stay balanced.
    loss_force = F.mse_loss(torch.asinh(out.force_next), torch.asinh(target_force))

    if bool(active_visual.any()):
        cos_sim = (out.field_proj[active_visual] * out.visual_proj[active_visual]).sum(dim=-1)
        loss_align = (1.0 - cos_sim).mean()
    else:
        loss_align = torch.zeros((), device=target_norm.device, dtype=target_norm.dtype)

    total = (
        loss_field
        + cfg.vorticity_weight * loss_vort
        + cfg.divergence_weight * loss_div
        + cfg.force_weight * loss_force
        + cfg.vision_align_weight * loss_align
    )
    parts = {
        "total": float(total.detach().item()),
        "field": float(loss_field.detach().item()),
        "vorticity": float(loss_vort.detach().item()),
        "divergence": float(loss_div.detach().item()),
        "force": float(loss_force.detach().item()),
        "vision_align": float(loss_align.detach().item()),
    }
    return total, parts


def train_multimodal_cylinder_forecaster(
    train_dataset: CylinderMultimodalDataset,
    val_dataset: CylinderMultimodalDataset | None = None,
    *,
    arch: MultimodalCylinderArch | None = None,
    config: MultimodalTrainConfig | None = None,
    manifest: MultimodalDatasetManifest | None = None,
    job_registry: TrainingJobRegistry | None = None,
    save_weights_path: str | Path | None = None,
    model_name: str = "cylinder_multimodal_forecaster",
) -> MultimodalTrainResult:
    """Train :class:`MultimodalCylinderForecaster` on trajectory-split data."""
    cfg = config or MultimodalTrainConfig()
    effective_arch = arch or MultimodalCylinderArch(history_steps=train_dataset.history_steps)
    if cfg.epochs < 1:
        raise ValueError("epochs must be >= 1")
    if cfg.pushforward_steps < 1:
        raise ValueError("pushforward_steps must be >= 1")
    if cfg.pushforward_steps > train_dataset.rollout_steps:
        raise ValueError(
            f"pushforward_steps ({cfg.pushforward_steps}) exceeds train_dataset.rollout_steps "
            f"({train_dataset.rollout_steps})"
        )
    if len(train_dataset) < 1:
        raise ValueError("train_dataset must contain at least 1 transition")

    torch.manual_seed(cfg.seed)
    rng = torch.Generator()
    rng.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    job_id: str | None = None
    if job_registry is not None:
        job_meta: dict[str, Any] = {
            "family": MULTIMODAL_MODEL_FAMILY,
            "arch": {
                **asdict(effective_arch),
                "patch_size": list(effective_arch.patch_size),
                "pos_grid_size": list(effective_arch.pos_grid_size),
            },
            "train_config": asdict(cfg),
        }
        if manifest is not None:
            job_meta["dataset_fingerprint"] = manifest.fingerprint
            job_meta["train_run_ids"] = list(manifest.train_run_ids)
            job_meta["val_run_ids"] = list(manifest.val_run_ids)
        job = job_registry.create_job(job_meta)
        job_id = job.job_id
        job_registry.update_status(job_id, "running")

    t0 = time.perf_counter()
    try:
        model = MultimodalCylinderForecaster(effective_arch).to(device)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(cfg.epochs, 1),
            eta_min=cfg.learning_rate * 0.05,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=collate_cylinder_multimodal,
            generator=rng,
        )
        val_loader = (
            DataLoader(
                val_dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                collate_fn=collate_cylinder_multimodal,
            )
            if val_dataset is not None and len(val_dataset) > 0
            else None
        )

        train_losses: list[float] = []
        val_losses: list[float] = []
        best_val = float("inf")
        best_epoch = 1
        best_state = copy.deepcopy(model.state_dict())

        pf_steps = int(cfg.pushforward_steps)
        pf_weights = [float(cfg.pushforward_decay**k) for k in range(pf_steps)]
        pf_weight_sum = max(sum(pf_weights), 1e-12)

        for epoch in range(1, cfg.epochs + 1):
            model.train()
            epoch_sum = 0.0
            n_items = 0
            for batch in train_loader:
                bsz = int(batch["field_t"].shape[0])
                hist_fields = batch["history_fields"].to(device)
                hist_forces = batch["history_forces"].to(device)
                target_fields = batch["target_fields"].to(device)
                target_forces = batch["target_forces"].to(device)
                geom = batch["geom"].to(device)
                solid_mask = batch["solid_mask"].to(device)
                physics_cond = batch["physics_cond"].to(device)
                visual_t = batch["visual_t"].to(device)
                has_visual = batch["has_visual"].to(device)

                # Modality dropout for zero-PNG / zero-force inference robustness
                if cfg.visual_dropout_prob > 0.0:
                    keep_vis = (
                        torch.rand((bsz,), generator=rng).to(device) >= cfg.visual_dropout_prob
                    )
                    active_vis = has_visual & keep_vis
                else:
                    active_vis = has_visual

                if cfg.force_input_dropout_prob > 0.0:
                    active_force = (
                        torch.rand((bsz,), generator=rng).to(device) >= cfg.force_input_dropout_prob
                    )
                else:
                    active_force = torch.ones((bsz,), device=device, dtype=torch.bool)

                optimizer.zero_grad(set_to_none=True)
                cur_hf = hist_fields
                cur_hforce = hist_forces
                zero_vis = torch.zeros((bsz,), device=device, dtype=torch.bool)
                total_loss = torch.zeros((), device=device, dtype=hist_fields.dtype)

                for k in range(pf_steps):
                    step_vis_flag = active_vis if k == 0 else zero_vis
                    step_vis = visual_t if k == 0 else None
                    out = model(
                        cur_hf[:, -1],
                        geom,
                        physics_cond,
                        history_fields=cur_hf,
                        history_forces=cur_hforce,
                        force_t=cur_hforce[:, -1],
                        has_force=active_force,
                        visual_t=step_vis,
                        has_visual=step_vis_flag,
                    )
                    step_loss, _ = _compute_multimodal_loss(
                        out,
                        target_norm=target_fields[:, k],
                        solid_mask=solid_mask,
                        target_force=target_forces[:, k],
                        active_visual=step_vis_flag,
                        cfg=cfg,
                    )
                    total_loss = total_loss + pf_weights[k] * step_loss
                    if k + 1 < pf_steps:
                        detach_step = cfg.pushforward_detach_prob > 0.0 and bool(
                            (torch.rand((), generator=rng) < cfg.pushforward_detach_prob).item()
                        )
                        next_f = (
                            out.field_next_norm.detach() if detach_step else out.field_next_norm
                        )
                        next_force = out.force_next.detach() if detach_step else out.force_next
                        if cur_hf.shape[1] > 1:
                            cur_hf = torch.cat([cur_hf[:, 1:], next_f.unsqueeze(1)], dim=1)
                            cur_hforce = torch.cat(
                                [cur_hforce[:, 1:], next_force.unsqueeze(1)],
                                dim=1,
                            )
                        else:
                            cur_hf = next_f.unsqueeze(1)
                            cur_hforce = next_force.unsqueeze(1)

                loss = total_loss / pf_weight_sum
                torch.autograd.backward(loss)
                if cfg.gradient_clip_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_norm)
                optimizer.step()

                epoch_sum += float(loss.detach().item()) * bsz
                n_items += bsz

            scheduler.step()
            mean_train = epoch_sum / max(n_items, 1)
            train_losses.append(mean_train)

            if val_loader is not None:
                model.eval()
                val_sum = 0.0
                val_items = 0
                with torch.no_grad():
                    for batch in val_loader:
                        bsz = int(batch["field_t"].shape[0])
                        hist_fields = batch["history_fields"].to(device)
                        hist_forces = batch["history_forces"].to(device)
                        field_next = batch["field_next"].to(device)
                        geom = batch["geom"].to(device)
                        solid_mask = batch["solid_mask"].to(device)
                        physics_cond = batch["physics_cond"].to(device)
                        force_next = batch["force_next"].to(device)
                        visual_t = batch["visual_t"].to(device)
                        has_visual = batch["has_visual"].to(device)
                        active_force = torch.ones((bsz,), device=device, dtype=torch.bool)
                        out = model(
                            hist_fields[:, -1],
                            geom,
                            physics_cond,
                            history_fields=hist_fields,
                            history_forces=hist_forces,
                            force_t=hist_forces[:, -1],
                            has_force=active_force,
                            visual_t=visual_t,
                            has_visual=has_visual,
                        )
                        v_loss, _ = _compute_multimodal_loss(
                            out,
                            target_norm=field_next,
                            solid_mask=solid_mask,
                            target_force=force_next,
                            active_visual=has_visual,
                            cfg=cfg,
                        )
                        val_sum += float(v_loss.item()) * bsz
                        val_items += bsz
                mean_val = val_sum / max(val_items, 1)
            else:
                # Deterministic no-dropout pass on training set when val_dataset is absent
                model.eval()
                eval_sum = 0.0
                eval_items = 0
                with torch.no_grad():
                    for batch in train_loader:
                        bsz = int(batch["field_t"].shape[0])
                        hist_fields = batch["history_fields"].to(device)
                        hist_forces = batch["history_forces"].to(device)
                        field_next = batch["field_next"].to(device)
                        geom = batch["geom"].to(device)
                        solid_mask = batch["solid_mask"].to(device)
                        physics_cond = batch["physics_cond"].to(device)
                        force_next = batch["force_next"].to(device)
                        visual_t = batch["visual_t"].to(device)
                        has_visual = batch["has_visual"].to(device)
                        active_force = torch.ones((bsz,), device=device, dtype=torch.bool)
                        out = model(
                            hist_fields[:, -1],
                            geom,
                            physics_cond,
                            history_fields=hist_fields,
                            history_forces=hist_forces,
                            force_t=hist_forces[:, -1],
                            has_force=active_force,
                            visual_t=visual_t,
                            has_visual=has_visual,
                        )
                        e_loss, _ = _compute_multimodal_loss(
                            out,
                            target_norm=field_next,
                            solid_mask=solid_mask,
                            target_force=force_next,
                            active_visual=has_visual,
                            cfg=cfg,
                        )
                        eval_sum += float(e_loss.item()) * bsz
                        eval_items += bsz
                mean_val = eval_sum / max(eval_items, 1)

            val_losses.append(mean_val)
            if mean_val <= best_val:
                best_val = mean_val
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())

        model.load_state_dict(best_state)
        model.eval()
        wall_sec = time.perf_counter() - t0

        registered_model_id: int | None = None
        if save_weights_path is not None:
            save_multimodal_forecaster(
                model,
                save_weights_path,
                normalization=train_dataset.normalization,
                manifest_fingerprint=manifest.fingerprint if manifest is not None else None,
            )

        if job_registry is not None and job_id is not None:
            summary_metrics = {
                "final_train_loss": train_losses[-1],
                "best_val_loss": best_val,
                "best_epoch": best_epoch,
                "train_wall_seconds": wall_sec,
            }
            job_registry.record_metrics(job_id, summary_metrics)
            if save_weights_path is not None:
                registered_model_id = job_registry.register_model(
                    job_id,
                    name=model_name,
                    path=Path(save_weights_path).resolve(),
                    arch={
                        **asdict(effective_arch),
                        "patch_size": list(effective_arch.patch_size),
                        "pos_grid_size": list(effective_arch.pos_grid_size),
                    },
                    metrics=summary_metrics,
                )
            job_registry.update_status(job_id, "completed")

        return MultimodalTrainResult(
            model=model,
            train_losses=tuple(train_losses),
            val_losses=tuple(val_losses),
            best_epoch=best_epoch,
            best_val_loss=best_val,
            train_wall_seconds=wall_sec,
            job_id=job_id,
            registered_model_id=registered_model_id,
        )
    except Exception as exc:
        if job_registry is not None and job_id is not None:
            job_registry.update_status(job_id, "failed", error=str(exc))
        raise


# ---------------------------------------------------------------------------
# Persistence-gated holdout evaluation & multi-step rollout
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ForecastEvaluationReport:
    """Quantitative comparison of :class:`MultimodalCylinderForecaster` vs persistence."""

    n_samples: int
    use_visual: bool
    use_force_input: bool
    model_rel_l2_velocity: float
    persistence_rel_l2_velocity: float
    velocity_skill_vs_persistence: float
    model_rel_l2_rho_perturbation: float
    persistence_rel_l2_rho_perturbation: float
    model_rel_l2_vorticity: float
    persistence_rel_l2_vorticity: float
    vorticity_skill_vs_persistence: float
    model_force_rmse: float
    persistence_force_rmse: float
    model_force_mae_cd: float
    persistence_force_mae_cd: float
    model_force_mae_cl: float
    persistence_force_mae_cl: float
    solid_velocity_max: float
    fluid_divergence_rms: float
    target_fluid_divergence_rms: float
    fluid_mass_drift_rel: float
    eval_wall_seconds: float
    ms_per_sample: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _masked_rel_l2(pred: Tensor, target: Tensor, mask: Tensor, eps: float = 1e-12) -> float:
    m = mask.to(dtype=pred.dtype)
    while m.ndim < pred.ndim:
        m = m.unsqueeze(1)
    diff_norm = torch.sqrt(((pred - target).pow(2) * m).sum(dim=tuple(range(1, pred.ndim))))
    ref_norm = torch.sqrt((target.pow(2) * m).sum(dim=tuple(range(1, pred.ndim)))).clamp_min(eps)
    return float((diff_norm / ref_norm).mean().item())


def evaluate_forecaster_vs_persistence(
    model: MultimodalCylinderForecaster,
    dataset: CylinderMultimodalDataset,
    *,
    batch_size: int = 8,
    device: str = "cpu",
    use_visual: bool = True,
    use_force_input: bool = True,
) -> ForecastEvaluationReport:
    """Evaluate one-step forecasts on *dataset* against the persistence baseline."""
    if len(dataset) < 1:
        raise ValueError("dataset must contain at least 1 transition")

    dev = torch.device(device)
    model = model.to(dev)
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_cylinder_multimodal,
    )

    vel_err_m: list[float] = []
    vel_err_p: list[float] = []
    rho_err_m: list[float] = []
    rho_err_p: list[float] = []
    vort_err_m: list[float] = []
    vort_err_p: list[float] = []
    force_sq_m: list[Tensor] = []
    force_sq_p: list[Tensor] = []
    cd_abs_m: list[Tensor] = []
    cd_abs_p: list[Tensor] = []
    cl_abs_m: list[Tensor] = []
    cl_abs_p: list[Tensor] = []
    div_rms_list: list[float] = []
    target_div_rms_list: list[float] = []
    mass_drift_list: list[float] = []
    solid_vel_max = 0.0
    total_samples = 0

    t0 = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            bsz = int(batch["field_t"].shape[0])
            field_t = batch["field_t"].to(dev)
            hist_fields = batch["history_fields"].to(dev)
            hist_forces = batch["history_forces"].to(dev)
            raw_t = batch["raw_t"].to(dev)
            raw_next = batch["raw_next"].to(dev)
            geom = batch["geom"].to(dev)
            solid_mask = batch["solid_mask"].to(dev)
            physics_cond = batch["physics_cond"].to(dev)
            force_t = batch["force_t"].to(dev)
            force_next = batch["force_next"].to(dev)
            visual_t = batch["visual_t"].to(dev)
            has_visual = (
                batch["has_visual"].to(dev)
                if use_visual
                else torch.zeros((bsz,), device=dev, dtype=torch.bool)
            )
            has_force = torch.full((bsz,), use_force_input, device=dev, dtype=torch.bool)

            u_scale = batch["u_scale"].to(dev).view(bsz, 1, 1)
            rho_scale = batch["rho_scale"].to(dev).view(bsz, 1, 1)
            rho_offset = float(batch["rho_offset"])

            out = model(
                field_t,
                geom,
                physics_cond,
                history_fields=hist_fields,
                history_forces=hist_forces,
                force_t=force_t,
                has_force=has_force,
                visual_t=visual_t if use_visual else None,
                has_visual=has_visual,
            )

            ux_pred = out.field_next_norm[:, 0] * u_scale
            uy_pred = out.field_next_norm[:, 1] * u_scale
            rho_pred = rho_offset + out.field_next_norm[:, 2] * rho_scale

            vel_pred = torch.stack([ux_pred, uy_pred], dim=1)
            vel_persist = raw_t[:, :2]
            vel_true = raw_next[:, :2]

            fluid_mask = ~solid_mask
            vel_err_m.append(_masked_rel_l2(vel_pred, vel_true, fluid_mask) * bsz)
            vel_err_p.append(_masked_rel_l2(vel_persist, vel_true, fluid_mask) * bsz)

            drho_pred = (rho_pred - rho_offset).unsqueeze(1)
            drho_persist = (raw_t[:, 2] - rho_offset).unsqueeze(1)
            drho_true = (raw_next[:, 2] - rho_offset).unsqueeze(1)
            rho_err_m.append(_masked_rel_l2(drho_pred, drho_true, fluid_mask) * bsz)
            rho_err_p.append(_masked_rel_l2(drho_persist, drho_true, fluid_mask) * bsz)

            interior_fluid = fluid_mask[:, 1:-1, 1:-1]
            vort_m = compute_vorticity_batch(ux_pred, uy_pred)[:, 1:-1, 1:-1].unsqueeze(1)
            vort_p = compute_vorticity_batch(raw_t[:, 0], raw_t[:, 1])[:, 1:-1, 1:-1].unsqueeze(1)
            vort_t = compute_vorticity_batch(raw_next[:, 0], raw_next[:, 1])[
                :, 1:-1, 1:-1
            ].unsqueeze(1)
            vort_err_m.append(_masked_rel_l2(vort_m, vort_t, interior_fluid) * bsz)
            vort_err_p.append(_masked_rel_l2(vort_p, vort_t, interior_fluid) * bsz)

            force_sq_m.append((out.force_next - force_next).pow(2))
            force_sq_p.append((force_t - force_next).pow(2))
            cd_abs_m.append((out.force_next[:, 0] - force_next[:, 0]).abs())
            cd_abs_p.append((force_t[:, 0] - force_next[:, 0]).abs())
            cl_abs_m.append((out.force_next[:, 1] - force_next[:, 1]).abs())
            cl_abs_p.append((force_t[:, 1] - force_next[:, 1]).abs())

            if bool(solid_mask.any()):
                speed_solid = torch.sqrt(ux_pred.pow(2) + uy_pred.pow(2))[solid_mask]
                solid_vel_max = max(solid_vel_max, float(speed_solid.max().item()))

            int_f = interior_fluid.to(dtype=ux_pred.dtype)
            int_denom = int_f.sum().clamp_min(1.0)
            div_m = compute_divergence_batch(ux_pred, uy_pred)[:, 1:-1, 1:-1]
            div_t = compute_divergence_batch(raw_next[:, 0], raw_next[:, 1])[:, 1:-1, 1:-1]
            div_rms_list.append(
                float(torch.sqrt((div_m.pow(2) * int_f).sum() / int_denom).item()) * bsz
            )
            target_div_rms_list.append(
                float(torch.sqrt((div_t.pow(2) * int_f).sum() / int_denom).item()) * bsz
            )

            fl_f = fluid_mask.to(dtype=rho_pred.dtype)
            mass_pred = (rho_pred * fl_f).sum(dim=(1, 2))
            mass_true = (raw_next[:, 2] * fl_f).sum(dim=(1, 2)).clamp_min(1e-6)
            mass_drift_list.append(
                float(((mass_pred - mass_true).abs() / mass_true).mean().item()) * bsz
            )
            total_samples += bsz

    wall_sec = time.perf_counter() - t0
    denom = float(max(total_samples, 1))

    m_vel = sum(vel_err_m) / denom
    p_vel = sum(vel_err_p) / denom
    m_rho = sum(rho_err_m) / denom
    p_rho = sum(rho_err_p) / denom
    m_vort = sum(vort_err_m) / denom
    p_vort = sum(vort_err_p) / denom

    vel_skill = 1.0 - (m_vel / max(p_vel, 1e-12))
    vort_skill = 1.0 - (m_vort / max(p_vort, 1e-12))

    m_f_rmse = float(torch.sqrt(torch.cat(force_sq_m, dim=0).mean()).item())
    p_f_rmse = float(torch.sqrt(torch.cat(force_sq_p, dim=0).mean()).item())

    return ForecastEvaluationReport(
        n_samples=total_samples,
        use_visual=use_visual,
        use_force_input=use_force_input,
        model_rel_l2_velocity=m_vel,
        persistence_rel_l2_velocity=p_vel,
        velocity_skill_vs_persistence=vel_skill,
        model_rel_l2_rho_perturbation=m_rho,
        persistence_rel_l2_rho_perturbation=p_rho,
        model_rel_l2_vorticity=m_vort,
        persistence_rel_l2_vorticity=p_vort,
        vorticity_skill_vs_persistence=vort_skill,
        model_force_rmse=m_f_rmse,
        persistence_force_rmse=p_f_rmse,
        model_force_mae_cd=float(torch.cat(cd_abs_m, dim=0).mean().item()),
        persistence_force_mae_cd=float(torch.cat(cd_abs_p, dim=0).mean().item()),
        model_force_mae_cl=float(torch.cat(cl_abs_m, dim=0).mean().item()),
        persistence_force_mae_cl=float(torch.cat(cl_abs_p, dim=0).mean().item()),
        solid_velocity_max=solid_vel_max,
        fluid_divergence_rms=sum(div_rms_list) / denom,
        target_fluid_divergence_rms=sum(target_div_rms_list) / denom,
        fluid_mass_drift_rel=sum(mass_drift_list) / denom,
        eval_wall_seconds=wall_sec,
        ms_per_sample=(wall_sec * 1000.0) / denom,
    )


@dataclass(frozen=True, slots=True)
class RolloutStepMetrics:
    """Autoregressive forecast error at rollout horizon step ``k`` vs persistence."""

    horizon_index: int
    solver_step: int
    model_rel_l2_velocity: float
    persistence_rel_l2_velocity: float
    model_rel_l2_vorticity: float
    persistence_rel_l2_vorticity: float
    pred_cd: float
    true_cd: float
    pred_cl: float
    true_cl: float


def rollout_trajectory_forecast(
    model: MultimodalCylinderForecaster,
    trajectory: CylinderTrajectoryRecord,
    *,
    start_index: int = 0,
    horizon: int | None = None,
    step_stride: int = 1,
    normalization: CylinderNormalization | None = None,
    device: str = "cpu",
) -> tuple[RolloutStepMetrics, ...]:
    """Autoregressively roll out *model* in zero-PNG physics mode vs persistence."""
    norm = normalization or CylinderNormalization()
    u_scale, rho_scale = norm.scales_for_run(trajectory.u_in)
    max_steps = (len(trajectory.frames) - 1 - start_index) // step_stride
    if max_steps < 1:
        raise ValueError("Insufficient trajectory frames for rollout")
    n_roll = min(horizon, max_steps) if horizon is not None else max_steps

    dev = torch.device(device)
    model = model.to(dev)
    model.eval()

    solid_mask = torch.from_numpy(trajectory.solid_mask).unsqueeze(0).to(dev)
    fluid_mask = ~solid_mask
    interior_fluid = fluid_mask[:, 1:-1, 1:-1]
    geom = (
        torch.stack(
            [
                torch.from_numpy(trajectory.solid_mask).to(dtype=torch.float32),
                torch.from_numpy(trajectory.sdf_norm),
            ],
            dim=0,
        )
        .unsqueeze(0)
        .to(dev)
    )

    f0 = trajectory.frames[start_index]
    init_raw = (
        torch.stack(
            [torch.from_numpy(f0.ux), torch.from_numpy(f0.uy), torch.from_numpy(f0.rho)],
            dim=0,
        )
        .unsqueeze(0)
        .to(dev)
    )
    hist_len = model.arch.history_steps
    hist_indices = [max(0, start_index - (hist_len - 1 - h) * step_stride) for h in range(hist_len)]
    init_hist_fields: list[Tensor] = []
    init_hist_forces: list[list[float]] = []
    for hi in hist_indices:
        fh = trajectory.frames[hi]
        f_raw = torch.stack(
            [torch.from_numpy(fh.ux), torch.from_numpy(fh.uy), torch.from_numpy(fh.rho)],
            dim=0,
        ).to(dev)
        init_hist_fields.append(
            torch.stack(
                [
                    f_raw[0] / u_scale,
                    f_raw[1] / u_scale,
                    (f_raw[2] - norm.rho_offset) / rho_scale,
                ],
                dim=0,
            )
        )
        init_hist_forces.append([fh.cd, fh.cl])

    cur_hist_fields = torch.stack(init_hist_fields, dim=0).unsqueeze(0)  # (1, H, 3, ny, nx)
    cur_hist_forces = torch.tensor(
        [init_hist_forces],
        dtype=torch.float32,
        device=dev,
    )  # (1, H, 2)
    cur_norm = cur_hist_fields[:, -1]
    cur_force = cur_hist_forces[:, -1]
    has_force = torch.ones((1,), dtype=torch.bool, device=dev)
    has_vis = torch.zeros((1,), dtype=torch.bool, device=dev)

    records: list[RolloutStepMetrics] = []
    with torch.no_grad():
        for k in range(1, n_roll + 1):
            prev_frame = trajectory.frames[start_index + (k - 1) * step_stride]
            true_frame = trajectory.frames[start_index + k * step_stride]
            delta_step = float(true_frame.step - prev_frame.step)
            physics_cond = torch.tensor(
                [
                    [
                        math.log10(max(trajectory.re, 1e-6)),
                        trajectory.u_in,
                        trajectory.tau - 0.5,
                        trajectory.radius / float(trajectory.ny),
                        float(trajectory.ny) / float(trajectory.nx),
                        delta_step / 100.0,
                    ]
                ],
                dtype=torch.float32,
                device=dev,
            )

            out = model(
                cur_norm,
                geom,
                physics_cond,
                history_fields=cur_hist_fields,
                history_forces=cur_hist_forces,
                force_t=cur_force,
                has_force=has_force,
                visual_t=None,
                has_visual=has_vis,
            )
            cur_norm = out.field_next_norm
            cur_force = out.force_next
            if hist_len > 1:
                cur_hist_fields = torch.cat(
                    [cur_hist_fields[:, 1:], cur_norm.unsqueeze(1)],
                    dim=1,
                )
                cur_hist_forces = torch.cat(
                    [cur_hist_forces[:, 1:], cur_force.unsqueeze(1)],
                    dim=1,
                )
            else:
                cur_hist_fields = cur_norm.unsqueeze(1)
                cur_hist_forces = cur_force.unsqueeze(1)

            ux_pred = cur_norm[:, 0] * u_scale
            uy_pred = cur_norm[:, 1] * u_scale
            vel_pred = torch.stack([ux_pred, uy_pred], dim=1)
            vel_true = (
                torch.stack(
                    [torch.from_numpy(true_frame.ux), torch.from_numpy(true_frame.uy)],
                    dim=0,
                )
                .unsqueeze(0)
                .to(dev)
            )
            vort_m = compute_vorticity_batch(ux_pred, uy_pred)[:, 1:-1, 1:-1].unsqueeze(1)
            vort_p = compute_vorticity_batch(init_raw[:, 0], init_raw[:, 1])[
                :, 1:-1, 1:-1
            ].unsqueeze(1)
            vort_t = compute_vorticity_batch(vel_true[:, 0], vel_true[:, 1])[
                :, 1:-1, 1:-1
            ].unsqueeze(1)

            records.append(
                RolloutStepMetrics(
                    horizon_index=k,
                    solver_step=true_frame.step,
                    model_rel_l2_velocity=_masked_rel_l2(vel_pred, vel_true, fluid_mask),
                    persistence_rel_l2_velocity=_masked_rel_l2(
                        init_raw[:, :2], vel_true, fluid_mask
                    ),
                    model_rel_l2_vorticity=_masked_rel_l2(vort_m, vort_t, interior_fluid),
                    persistence_rel_l2_vorticity=_masked_rel_l2(vort_p, vort_t, interior_fluid),
                    pred_cd=float(cur_force[0, 0].item()),
                    true_cd=true_frame.cd,
                    pred_cl=float(cur_force[0, 1].item()),
                    true_cl=true_frame.cl,
                )
            )
    return tuple(records)


# ---------------------------------------------------------------------------
# Checkpoint persistence (weights + JSON sidecar)
# ---------------------------------------------------------------------------


def save_multimodal_forecaster(
    model: MultimodalCylinderForecaster,
    path: str | Path,
    *,
    normalization: CylinderNormalization | None = None,
    manifest_fingerprint: str | None = None,
) -> tuple[Path, Path]:
    """Save weights to *path* and JSON metadata sidecar to ``<path>.json``."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "family": MULTIMODAL_MODEL_FAMILY,
        "arch": {
            **asdict(model.arch),
            "patch_size": list(model.arch.patch_size),
            "pos_grid_size": list(model.arch.pos_grid_size),
        },
        "state_dict": model.state_dict(),
    }
    torch.save(payload, out)
    sidecar = Path(f"{out}.json")
    meta: dict[str, Any] = {
        "family": MULTIMODAL_MODEL_FAMILY,
        "backend": "torch",
        "arch": payload["arch"],
        "normalization": asdict(normalization or CylinderNormalization()),
        "manifest_fingerprint": manifest_fingerprint,
    }
    sidecar.write_text(f"{json.dumps(meta, indent=2, sort_keys=True)}\n", encoding="utf-8")
    return out, sidecar


def load_multimodal_forecaster(
    path: str | Path,
    *,
    device: str = "cpu",
) -> MultimodalCylinderForecaster:
    """Load a :class:`MultimodalCylinderForecaster` saved by :func:`save_multimodal_forecaster`."""
    raw = torch.load(Path(path), map_location=device, weights_only=True)
    if not isinstance(raw, dict) or raw.get("family") != MULTIMODAL_MODEL_FAMILY:
        raise ValueError(f"Checkpoint at {path} is not a {MULTIMODAL_MODEL_FAMILY} bundle")
    arch_dict = dict(raw["arch"])
    arch_dict.setdefault("history_steps", 1)
    arch_dict["patch_size"] = tuple(int(v) for v in arch_dict["patch_size"])
    arch_dict["pos_grid_size"] = tuple(int(v) for v in arch_dict["pos_grid_size"])
    arch = MultimodalCylinderArch(**arch_dict)
    model = MultimodalCylinderForecaster(arch)
    model.load_state_dict(raw["state_dict"])
    model.to(device)
    model.eval()
    return model
