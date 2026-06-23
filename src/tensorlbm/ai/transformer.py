"""Transformer-based self-supervised learning for flow-field snapshots.

This module provides a compact masked-reconstruction transformer that learns
from unlabeled ``(u_x, u_y)`` flow fields.  The trained model can be saved and
later deployed for inference/reconstruction diagnostics.

Pass ``backend="paddle"`` or ``backend="mindspore"`` to
:func:`train_flow_transformer_self_supervised` to use a non-PyTorch framework.
Set ``TENSORLBM_BACKEND`` in the environment for a process-wide default.
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn, optim

from ..backends import get_backend, get_ops

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class FlowTransformerArch:
    """Architecture hyper-parameters for :class:`FlowFieldTransformer`."""

    in_features: int = 2
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 2
    ffn_dim: int = 128
    dropout: float = 0.1
    max_tokens: int = 4096


@dataclass(frozen=True)
class FlowTransformerTrainConfig:
    """Training hyper-parameters for self-supervised transformer learning."""

    epochs: int = 20
    batch_size: int = 8
    learning_rate: float = 1e-3
    val_fraction: float = 0.1
    mask_ratio: float = 0.15
    seed: int = 0
    device: str = "cpu"
    lr_scheduler: str = "none"
    patience: int | None = None
    gradient_clip_norm: float | None = 1.0
    mask_ratio_schedule: str = "none"
    mask_ratio_start: float | None = None


class FlowFieldTransformer(nn.Module):
    """Masked-token reconstruction transformer for 2-D flow-field tokens."""

    def __init__(self, arch: FlowTransformerArch | None = None) -> None:
        super().__init__()
        self.arch = arch or FlowTransformerArch()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.arch.in_features))
        self.input_proj = nn.Linear(self.arch.in_features, self.arch.d_model)
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, self.arch.max_tokens, self.arch.d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.arch.d_model,
            nhead=self.arch.n_heads,
            dim_feedforward=self.arch.ffn_dim,
            dropout=self.arch.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=self.arch.n_layers)
        self.head = nn.Linear(self.arch.d_model, self.arch.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        if x.ndim != 3:
            raise ValueError(f"Expected (B, T, F), got {tuple(x.shape)}")
        if x.shape[-1] != self.arch.in_features:
            raise ValueError(
                f"Expected in_features={self.arch.in_features}, got {x.shape[-1]}",
            )
        n_tokens = int(x.shape[1])
        if n_tokens > self.arch.max_tokens:
            raise ValueError(
                f"Token count {n_tokens} exceeds max_tokens={self.arch.max_tokens}",
            )
        h = self.input_proj(x) + self.pos_embedding[:, :n_tokens, :]
        h = self.encoder(h)
        return self.head(h)

    def apply_mask_token(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Replace masked tokens with a learnable token embedding."""
        if mask.shape != x.shape[:2]:
            raise ValueError(f"Expected mask shape {tuple(x.shape[:2])}, got {tuple(mask.shape)}")
        return torch.where(mask.unsqueeze(-1), self.mask_token.expand_as(x), x)


def flow_snapshot_to_tokens(ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    """Convert one ``(ux, uy)`` snapshot to a token tensor ``(T, 2)``."""
    if ux.shape != uy.shape or ux.ndim != 2:
        raise ValueError(
            f"ux and uy must be 2-D tensors with equal shape, got "
            f"{tuple(ux.shape)} and {tuple(uy.shape)}",
        )
    tokens = torch.stack([ux, uy], dim=-1)  # (ny, nx, 2)
    return tokens.reshape(-1, 2).contiguous()


def build_flow_token_batch(
    snapshots: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Stack snapshots into a batch ``(N, T, 2)`` and return grid shape."""
    if not snapshots:
        raise ValueError("At least one snapshot is required")
    ny, nx = snapshots[0][0].shape
    seqs: list[torch.Tensor] = []
    for ux, uy in snapshots:
        if ux.shape != (ny, nx) or uy.shape != (ny, nx):
            raise ValueError("All snapshots must share the same (ny, nx) shape")
        seqs.append(flow_snapshot_to_tokens(ux, uy))
    return torch.stack(seqs, dim=0), (ny, nx)


def _split_train_val(
    data: torch.Tensor,
    val_fraction: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(data.shape[0])
    if n < 2:
        raise ValueError("Need at least 2 snapshots for train/validation split")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    n_val = max(1, int(round(n * val_fraction)))
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=g)
    idx_val = perm[:n_val]
    idx_train = perm[n_val:]
    if idx_train.numel() == 0:
        idx_train = idx_val[:1]
    return data.index_select(0, idx_train), data.index_select(0, idx_val)


def save_flow_transformer_model(
    model: FlowFieldTransformer,
    path: str | Path,
) -> Path:
    """Persist model weights and architecture using a non-pickle format."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    arrays = {name: tensor.detach().cpu().numpy() for name, tensor in state.items()}
    with p.open("wb") as fh:
        np.savez_compressed(fh, **arrays)
    meta = {
        "arch": asdict(model.arch),
        "format_version": 1,
        "family": "flow_transformer_ssl",
    }
    p.with_suffix(p.suffix + ".json").write_text(json.dumps(meta, indent=2))
    return p


def load_flow_transformer_model(path: str | Path) -> FlowFieldTransformer:
    """Load a saved transformer model."""
    p = Path(path)
    meta_path = p.with_suffix(p.suffix + ".json")
    if not meta_path.exists():
        raise ValueError(f"Metadata file not found: {meta_path}")
    meta = json.loads(meta_path.read_text())
    arch = FlowTransformerArch(**meta.get("arch", {}))
    model = FlowFieldTransformer(arch)
    with p.open("rb") as fh:
        arrays = np.load(fh, allow_pickle=False)
        current = model.state_dict()
        loaded = {
            name: torch.from_numpy(arrays[name]).to(dtype=current[name].dtype)
            for name in current
        }
    model.load_state_dict(loaded)
    model.eval()
    return model


def _build_scheduler(
    optimizer: optim.Optimizer,
    epochs: int,
    scheduler_name: str,
) -> optim.lr_scheduler.LRScheduler | optim.lr_scheduler.ReduceLROnPlateau | None:
    name = scheduler_name.strip().lower()
    if name == "none":
        return None
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(epochs)))
    if name == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    raise ValueError(f"Unsupported lr_scheduler: {scheduler_name!r}")


def _scheduled_mask_ratio(cfg: FlowTransformerTrainConfig, epoch: int) -> float:
    target = float(cfg.mask_ratio)
    schedule = str(cfg.mask_ratio_schedule).strip().lower()
    if schedule == "none":
        return target
    start = float(
        cfg.mask_ratio_start if cfg.mask_ratio_start is not None else max(0.01, target * 0.5),
    )
    start = min(max(start, 0.01), 0.99)
    if schedule == "step":
        return start if int(epoch) == 0 else target
    if schedule == "linear":
        span = max(1, int(cfg.epochs) - 1)
        alpha = min(max(epoch / span, 0.0), 1.0)
        return start + (target - start) * alpha
    raise ValueError(f"Unsupported mask_ratio_schedule: {cfg.mask_ratio_schedule!r}")


def train_flow_transformer_self_supervised(
    snapshots: list[tuple[torch.Tensor, torch.Tensor]],
    out_path: str | Path,
    arch: FlowTransformerArch | None = None,
    config: FlowTransformerTrainConfig | None = None,
    progress_callback: Callable[[dict[str, float]], None] | None = None,
    *,
    backend: str | None = None,
) -> dict[str, Any]:
    """Train a masked-reconstruction transformer on unlabeled flow fields.

    Args:
        snapshots: List of ``(ux, uy)`` velocity snapshot pairs (torch.Tensor).
        out_path: Where to save the trained model (weights + JSON metadata).
        arch: Architecture hyper-parameters.
        config: Training hyper-parameters.
        progress_callback: Optional per-epoch callback ``(metrics_dict) -> None``.
        backend: Computation backend.  Defaults to ``TENSORLBM_BACKEND``
            env var (``"torch"`` if unset).  Valid: ``"torch"``, ``"paddle"``,
            ``"mindspore"``.

    Returns:
        Metadata dict with paths, history, and the ``backend`` key.
    """
    backend_name = backend or get_backend()
    cfg = config or FlowTransformerTrainConfig()
    arch = arch or FlowTransformerArch()

    # Convert snapshots to numpy regardless of backend so the same code path
    # builds the token batch for all backends.
    snapshots_np = [
        (
            ux.detach().cpu().numpy() if hasattr(ux, "detach") else np.array(ux),
            uy.detach().cpu().numpy() if hasattr(uy, "detach") else np.array(uy),
        )
        for ux, uy in snapshots
    ]

    if backend_name != "torch":
        return _train_flow_transformer_backend(
            snapshots_np, arch, cfg, Path(out_path), backend_name, progress_callback
        )

    # ---- original torch path ------------------------------------------------
    batch, grid = build_flow_token_batch(snapshots)
    if batch.shape[1] > int(arch.max_tokens):
        raise ValueError(
            f"Grid token count {batch.shape[1]} exceeds max_tokens={arch.max_tokens}",
        )

    device = torch.device(cfg.device)
    model = FlowFieldTransformer(arch).to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(cfg.learning_rate))
    loss_fn = nn.MSELoss()
    scheduler = _build_scheduler(optimizer, int(cfg.epochs), str(cfg.lr_scheduler))

    has_val_split = int(batch.shape[0]) >= 2
    if has_val_split:
        train_x, val_x = _split_train_val(batch, cfg.val_fraction, cfg.seed)
    else:
        train_x = batch
        val_x = batch

    train_x = train_x.to(device)
    val_x = val_x.to(device)
    g = torch.Generator(device="cpu").manual_seed(int(cfg.seed))
    history: list[dict[str, float]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_val_loss = float("inf")
    epochs_without_improve = 0
    t0 = time.perf_counter()

    for epoch in range(int(cfg.epochs)):
        model.train()
        perm = torch.randperm(train_x.shape[0], generator=g)
        running_loss = 0.0
        n_batches = 0
        bs = max(1, min(int(cfg.batch_size), int(train_x.shape[0])))
        mask_ratio = _scheduled_mask_ratio(cfg, epoch)
        for i in range(0, int(train_x.shape[0]), bs):
            idx = perm[i : i + bs]
            xb = train_x.index_select(0, idx)
            mask = torch.rand(
                xb.shape[0], xb.shape[1],
                device=device,
            ) < float(mask_ratio)
            x_masked = model.apply_mask_token(xb, mask)

            optimizer.zero_grad()
            pred = model(x_masked)
            loss = loss_fn(pred[mask], xb[mask]) if bool(mask.any()) else loss_fn(pred, xb)
            loss.backward()
            if cfg.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=float(cfg.gradient_clip_norm),
                )
            optimizer.step()
            running_loss += float(loss.detach())
            n_batches += 1

        model.eval()
        train_loss = float(running_loss / max(1, n_batches))
        with torch.no_grad():
            if has_val_split:
                val_mask = torch.rand(
                    val_x.shape[0], val_x.shape[1],
                    device=device,
                ) < float(mask_ratio)
                val_in = model.apply_mask_token(val_x, val_mask)
                val_pred = model(val_in)
                if bool(val_mask.any()):
                    val_loss = float(loss_fn(val_pred[val_mask], val_x[val_mask]).detach())
                else:
                    val_loss = float(loss_fn(val_pred, val_x).detach())
            else:
                val_loss = train_loss

        current_lr = float(optimizer.param_groups[0]["lr"])
        metrics = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "lr": float(current_lr),
            "mask_ratio": float(mask_ratio),
        }
        history.append(metrics)
        if progress_callback is not None:
            progress_callback(dict(metrics))
        improved = val_loss < (best_val_loss - 1e-12)
        if improved:
            best_val_loss = float(val_loss)
            best_epoch = int(epoch)
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        elif scheduler is not None:
            scheduler.step()
        if cfg.patience is not None and epochs_without_improve > int(cfg.patience):
            break

    model.load_state_dict(best_state)
    out = Path(out_path)
    save_flow_transformer_model(model, out)
    elapsed = time.perf_counter() - t0
    final = history[best_epoch]
    return {
        "path": str(out),
        "family": "flow_transformer_ssl",
        "arch": asdict(arch),
        "config": asdict(cfg),
        "backend": "torch",
        "n_snapshots": int(batch.shape[0]),
        "n_tokens": int(batch.shape[1]),
        "grid": [int(grid[0]), int(grid[1])],
        "history": history,
        "final_train_loss": float(final["train_loss"]),
        "final_val_loss": float(final["val_loss"]),
        "best_epoch": int(best_epoch),
        "stopped_early": bool(len(history) < int(cfg.epochs)),
        "training_time_s": float(elapsed),
    }


# ---------------------------------------------------------------------------
# Backend-agnostic transformer training (PaddlePaddle / MindSpore)
# ---------------------------------------------------------------------------

def _train_flow_transformer_backend(
    snapshots_np: list[tuple[np.ndarray, np.ndarray]],
    arch: FlowTransformerArch,
    cfg: FlowTransformerTrainConfig,
    out_path: Path,
    backend_name: str,
    progress_callback,
) -> dict[str, Any]:
    """Backend-agnostic transformer training loop."""
    ops = get_ops()

    # Build token batch from numpy snapshots
    if not snapshots_np:
        raise ValueError("At least one snapshot is required")
    ny, nx = snapshots_np[0][0].shape
    n_tokens = ny * nx
    if n_tokens > int(arch.max_tokens):
        raise ValueError(f"Grid token count {n_tokens} exceeds max_tokens={arch.max_tokens}")

    # Stack all snapshots: (N, T, 2) as numpy
    batch_np = np.stack(
        [np.stack([ux.ravel(), uy.ravel()], axis=-1) for ux, uy in snapshots_np],
        axis=0,
    ).astype(np.float32)  # (N, T, 2)
    grid = (ny, nx)
    n_snap = len(batch_np)

    # Split train / val
    has_val_split = n_snap >= 2
    if has_val_split:
        n_val = max(1, int(round(n_snap * cfg.val_fraction)))
        rng = np.random.RandomState(int(cfg.seed))
        perm = rng.permutation(n_snap)
        idx_val   = perm[:n_val]
        idx_train = perm[n_val:] if n_val < n_snap else perm[:1]
        train_np  = batch_np[idx_train]
        val_np    = batch_np[idx_val]
    else:
        train_np = val_np = batch_np

    # Convert to backend tensors
    train_x = ops.to_device(ops.tensor(train_np), cfg.device)
    val_x   = ops.to_device(ops.tensor(val_np), cfg.device)

    # Build model
    ops.manual_seed(cfg.seed)
    model = ops.build_flow_transformer(
        int(arch.in_features), int(arch.d_model), int(arch.n_heads),
        int(arch.n_layers), int(arch.ffn_dim), float(arch.dropout),
        int(arch.max_tokens), device=cfg.device,
    )
    loss_fn   = ops.mse_loss_fn()
    optimizer = ops.adam_optimizer(model, cfg.learning_rate)
    scheduler = None
    sch_name  = str(cfg.lr_scheduler).strip().lower()
    if sch_name == "cosine":
        scheduler = ops.cosine_lr_scheduler(optimizer, int(cfg.epochs))
    elif sch_name == "plateau":
        scheduler = ops.plateau_lr_scheduler(optimizer)

    rng2 = np.random.RandomState(int(cfg.seed))
    n_train_snap = int(train_np.shape[0])
    bs = max(1, min(int(cfg.batch_size), n_train_snap))
    history: list[dict[str, float]] = []
    best_state_np: dict[str, np.ndarray] = {}
    best_epoch = 0
    best_val_loss = float("inf")
    epochs_without_improve = 0
    t0 = time.perf_counter()

    for epoch in range(int(cfg.epochs)):
        # Scheduled mask ratio
        mask_ratio = float(_scheduled_mask_ratio(cfg, epoch))
        ops.train_mode(model)
        perm_ep = rng2.permutation(n_train_snap)
        running_loss = 0.0
        n_batches = 0
        for i in range(0, n_train_snap, bs):
            idx = perm_ep[i: i + bs]
            xb_np = train_np[idx]  # (B, T, 2)
            xb = ops.to_device(ops.tensor(xb_np), cfg.device)
            # Build boolean mask
            mask_np = rng2.random(xb_np.shape[:2]) < mask_ratio  # (B, T)
            mask = ops.to_device(ops.tensor(mask_np.astype(np.float32)), cfg.device)
            # Boolean mask for backend
            mask_bool = mask > 0.5

            # Apply mask token and run forward + loss
            x_masked = model.apply_mask_token(xb, mask_bool)

            def _step_loss():
                pred = model(x_masked)
                if ops.any_true(mask_bool):
                    # only compute loss on masked tokens
                    pred_flat = ops.reshape(pred, [-1, int(arch.in_features)])
                    xb_flat   = ops.reshape(xb,   [-1, int(arch.in_features)])
                    mask_flat = ops.reshape(mask_bool, [-1])
                    p = ops.index_select(pred_flat, 0, ops.tensor(
                        np.where(mask_np.ravel())[0].astype(np.int32), device=cfg.device))
                    t = ops.index_select(xb_flat, 0, ops.tensor(
                        np.where(mask_np.ravel())[0].astype(np.int32), device=cfg.device))
                    return loss_fn(p, t)
                return loss_fn(pred, xb)

            batch_loss = ops.train_step(model, lambda a, b: _step_loss(), optimizer, xb, xb, cfg.gradient_clip_norm)
            running_loss += float(batch_loss)
            n_batches += 1

        train_loss = running_loss / max(1, n_batches)
        ops.eval_mode(model)
        with ops.no_grad():
            # val loss
            mask_val_np = rng2.random(val_np.shape[:2]) < mask_ratio
            mask_val_bool_np = mask_val_np.astype(np.float32)
            val_mask_t = ops.to_device(ops.tensor(mask_val_bool_np), cfg.device)
            val_mask_bool = val_mask_t > 0.5
            val_in = model.apply_mask_token(val_x, val_mask_bool)
            val_pred = model(val_in)
            if ops.any_true(val_mask_bool):
                vp = ops.reshape(val_pred, [-1, int(arch.in_features)])
                vt = ops.reshape(val_x,    [-1, int(arch.in_features)])
                vp2 = ops.index_select(vp, 0, ops.tensor(
                    np.where(mask_val_np.ravel())[0].astype(np.int32), device=cfg.device))
                vt2 = ops.index_select(vt, 0, ops.tensor(
                    np.where(mask_val_np.ravel())[0].astype(np.int32), device=cfg.device))
                val_loss = float(ops.float_scalar(loss_fn(vp2, vt2)))
            else:
                val_loss = float(ops.float_scalar(loss_fn(val_pred, val_x)))

        current_lr = ops.get_lr(optimizer)
        metrics: dict[str, float] = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "lr": float(current_lr),
            "mask_ratio": float(mask_ratio),
        }
        history.append(metrics)
        if progress_callback is not None:
            progress_callback(dict(metrics))
        if val_loss < (best_val_loss - 1e-12):
            best_val_loss = val_loss
            best_epoch = int(epoch)
            best_state_np = ops.get_state_dict_numpy(model)
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
        if scheduler is not None:
            ops.scheduler_step(scheduler, val_loss if ops.is_plateau_scheduler(scheduler) else None)
        if cfg.patience is not None and epochs_without_improve > int(cfg.patience):
            break

    ops.load_state_dict_numpy(model, best_state_np)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_path), **best_state_np)
    meta = {
        "arch": asdict(arch),
        "format_version": 2,
        "family": "flow_transformer_ssl",
        "backend": backend_name,
    }
    out_path.with_suffix(out_path.suffix + ".json").write_text(json.dumps(meta, indent=2))

    elapsed = time.perf_counter() - t0
    final = history[best_epoch] if history else {"train_loss": float("nan"), "val_loss": float("nan")}
    return {
        "path": str(out_path),
        "family": "flow_transformer_ssl",
        "arch": asdict(arch),
        "config": asdict(cfg),
        "backend": backend_name,
        "n_snapshots": n_snap,
        "n_tokens": n_tokens,
        "grid": [int(grid[0]), int(grid[1])],
        "history": history,
        "final_train_loss": float(final.get("train_loss", float("nan"))),
        "final_val_loss": float(final.get("val_loss", float("nan"))),
        "best_epoch": int(best_epoch),
        "stopped_early": bool(len(history) < int(cfg.epochs)),
        "training_time_s": float(elapsed),
    }


def reconstruct_flow_field(
    model: FlowFieldTransformer,
    ux: torch.Tensor,
    uy: torch.Tensor,
) -> dict[str, Any]:
    """Run deployed inference and return reconstructed fields + diagnostics."""
    tokens = flow_snapshot_to_tokens(ux, uy)
    ny, nx = ux.shape
    x = tokens.unsqueeze(0)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        pred = model(x).squeeze(0)
    if was_training:
        model.train()

    ux_rec = pred[:, 0].reshape(ny, nx)
    uy_rec = pred[:, 1].reshape(ny, nx)
    err = pred - tokens
    return {
        "ux_reconstructed": ux_rec,
        "uy_reconstructed": uy_rec,
        "mse": float(torch.mean(err * err)),
        "max_abs_error": float(torch.max(torch.abs(err))),
    }
