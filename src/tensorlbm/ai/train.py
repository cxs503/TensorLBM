"""Training loop for the AI turbulence model.

A deliberately small and self-contained implementation: full-batch (or
mini-batch) Adam + MSE for a configurable number of epochs.  Designed to
run in seconds on CPU so it is usable both inside the platform agent and
in the CI test suite.
"""
from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn, optim

from .dataset import EddyViscosityDataset, load_dataset_pt
from .model import EddyViscosityMLP, ModelArch, save_model


@dataclass(frozen=True)
class TrainConfig:
    """Hyper-parameters of :func:`train_eddy_viscosity_model`."""

    epochs: int = 20
    batch_size: int = 4096
    learning_rate: float = 1e-3
    val_fraction: float = 0.1
    seed: int = 0
    # Architecture options forwarded to :class:`ModelArch`.
    hidden_features: int = 16
    n_hidden_layers: int = 2
    activation: str = "tanh"
    device: str = "cpu"
    lr_scheduler: str = "none"
    patience: int | None = None
    gradient_clip_norm: float | None = 1.0


def _r2_score(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    var = torch.var(y_true, unbiased=False)
    if float(var) <= 0.0:
        return 0.0
    residual = torch.mean((y_pred - y_true) ** 2)
    return float(1.0 - residual / var)


def _iter_minibatches(
    n: int, batch_size: int, generator: torch.Generator,
) -> list[torch.Tensor]:
    perm = torch.randperm(n, generator=generator)
    return [perm[i : i + batch_size] for i in range(0, n, batch_size)]


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


def train_eddy_viscosity_model(
    dataset: EddyViscosityDataset | str | Path,
    out_path: str | Path,
    config: TrainConfig | None = None,
) -> dict[str, Any]:
    """Train an :class:`EddyViscosityMLP` and persist it to ``out_path``.

    Args:
        dataset: Either an in-memory :class:`EddyViscosityDataset` or a
            path to one produced by
            :func:`tensorlbm.ai.save_dataset_pt`.
        out_path: File path where the trained model is saved.
        config: Optional :class:`TrainConfig`.

    Returns:
        A metadata dict containing the loss history, final train / val
        metrics, model architecture and saved file path.
    """
    cfg = config or TrainConfig()
    if not isinstance(dataset, EddyViscosityDataset):
        dataset = load_dataset_pt(dataset)
    if len(dataset) < 4:
        raise ValueError(
            f"Dataset is too small to train ({len(dataset)} samples).",
        )

    device = torch.device(cfg.device)
    train_ds, val_ds = dataset.split(cfg.val_fraction, seed=cfg.seed)
    x_train = train_ds.features.to(device)
    y_train = train_ds.targets.to(device)
    x_val = val_ds.features.to(device)
    y_val = val_ds.targets.to(device)

    torch.manual_seed(int(cfg.seed))
    arch = ModelArch(
        in_features=int(x_train.shape[-1]),
        hidden_features=int(cfg.hidden_features),
        n_hidden_layers=int(cfg.n_hidden_layers),
        activation=str(cfg.activation),
    )
    model = EddyViscosityMLP(arch).to(device)
    feature_mean = x_train.mean(dim=0)
    feature_std = x_train.std(dim=0, unbiased=False).clamp_min(1e-6)
    model.set_feature_stats(feature_mean, feature_std)
    optimizer = optim.Adam(model.parameters(), lr=float(cfg.learning_rate))
    loss_fn = nn.MSELoss()
    scheduler = _build_scheduler(optimizer, int(cfg.epochs), str(cfg.lr_scheduler))

    generator = torch.Generator(device="cpu").manual_seed(int(cfg.seed))
    history: list[dict[str, float]] = []
    n_train = x_train.shape[0]
    batch_size = max(1, min(int(cfg.batch_size), n_train))
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_val_mse = float("inf")
    epochs_without_improve = 0
    t0 = time.perf_counter()

    for epoch in range(int(cfg.epochs)):
        model.train()
        epoch_loss = 0.0
        epoch_abs_error = 0.0
        n_samples = 0
        for idx in _iter_minibatches(n_train, batch_size, generator):
            xb = x_train.index_select(0, idx)
            yb = y_train.index_select(0, idx)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            if cfg.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=float(cfg.gradient_clip_norm),
                )
            optimizer.step()
            epoch_loss += float(loss.detach()) * xb.shape[0]
            epoch_abs_error += float(torch.abs(pred.detach() - yb).sum())
            n_samples += xb.shape[0]
        train_mse = epoch_loss / max(1, n_samples)
        train_mae = epoch_abs_error / max(1, n_samples)

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val)
            val_mse = float(loss_fn(val_pred, y_val).detach())
            val_mae = float(torch.mean(torch.abs(val_pred - y_val)).detach())
            val_r2 = _r2_score(val_pred, y_val)
        current_lr = float(optimizer.param_groups[0]["lr"])
        metrics = {
            "epoch": int(epoch),
            "train_mse": float(train_mse),
            "train_mae": float(train_mae),
            "val_mse": float(val_mse),
            "val_mae": float(val_mae),
            "val_r2": float(val_r2),
            "lr": float(current_lr),
        }
        history.append(metrics)
        improved = val_mse < (best_val_mse - 1e-12)
        if improved:
            best_val_mse = float(val_mse)
            best_epoch = int(epoch)
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_mse)
        elif scheduler is not None:
            scheduler.step()
        if cfg.patience is not None and epochs_without_improve > int(cfg.patience):
            break

    model.load_state_dict(best_state)
    out_path = Path(out_path)
    save_model(model, out_path)

    elapsed = time.perf_counter() - t0
    final = history[best_epoch] if history else {"train_mse": float("nan")}
    return {
        "path": str(out_path),
        "arch": asdict(arch),
        "config": asdict(cfg),
        "n_samples_train": int(n_train),
        "n_samples_val": int(x_val.shape[0]),
        "history": history,
        "final_train_mse": float(final.get("train_mse", float("nan"))),
        "final_train_mae": float(final.get("train_mae", float("nan"))),
        "final_val_mse": float(final.get("val_mse", float("nan"))),
        "final_val_mae": float(final.get("val_mae", float("nan"))),
        "final_val_r2": float(final.get("val_r2", float("nan"))),
        "best_epoch": int(best_epoch),
        "stopped_early": bool(len(history) < int(cfg.epochs)),
        "training_time_s": float(elapsed),
    }
