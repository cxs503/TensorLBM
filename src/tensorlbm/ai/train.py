"""Training loop for the AI turbulence model.

A deliberately small and self-contained implementation: full-batch (or
mini-batch) Adam + MSE for a configurable number of epochs.  Designed to
run in seconds on CPU so it is usable both inside the platform agent and
in the CI test suite.

The :func:`train_eddy_viscosity_model` function accepts an optional
``backend`` keyword argument.  Pass ``backend="paddle"`` or
``backend="mindspore"`` (or set ``TENSORLBM_BACKEND`` in the environment)
to train with a non-PyTorch framework while keeping the same API.
"""
from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn, optim

from ..backends import get_backend, get_ops, using_backend
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


# ---------------------------------------------------------------------------
# Helpers – backend-agnostic
# ---------------------------------------------------------------------------

def _to_numpy(x) -> np.ndarray:
    """Convert any tensor / array to a numpy array."""
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "asnumpy"):          # MindSpore
        return x.asnumpy()
    if hasattr(x, "numpy"):            # torch / paddle
        try:
            return x.detach().cpu().numpy()
        except Exception:
            return x.numpy()
    return np.array(x)


def _split_numpy(
    features: np.ndarray,
    targets: np.ndarray,
    val_fraction: float,
    seed: int,
):
    n = len(features)
    n_val = max(1, int(round(n * val_fraction)))
    rng = np.random.RandomState(int(seed))
    perm = rng.permutation(n)
    idx_val = perm[:n_val]
    idx_train = perm[n_val:] if n_val < n else perm[:1]
    return (features[idx_train], targets[idx_train]), (features[idx_val], targets[idx_val])


def _r2_score_np(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    v = float(np.var(y_true))
    if v <= 0.0:
        return 0.0
    return float(1.0 - np.mean((y_pred - y_true) ** 2) / v)


def _iter_minibatches_np(n: int, batch_size: int, rng: np.random.RandomState):
    perm = rng.permutation(n)
    return [perm[i: i + batch_size] for i in range(0, n, batch_size)]


def _build_scheduler_ops(ops, optimizer, cfg: TrainConfig):
    name = str(cfg.lr_scheduler).strip().lower()
    if name == "none":
        return None
    if name == "cosine":
        return ops.cosine_lr_scheduler(optimizer, int(cfg.epochs))
    if name == "plateau":
        return ops.plateau_lr_scheduler(optimizer)
    raise ValueError(f"Unsupported lr_scheduler: {cfg.lr_scheduler!r}")


# ---------------------------------------------------------------------------
# Original torch-only helpers (kept for internal use by the torch path)
# ---------------------------------------------------------------------------

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
    return [perm[i: i + batch_size] for i in range(0, n, batch_size)]


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


# ---------------------------------------------------------------------------
# Multi-backend training implementation
# ---------------------------------------------------------------------------

def _train_with_backend(
    features_np: np.ndarray,
    targets_np: np.ndarray,
    out_path: Path,
    cfg: TrainConfig,
    backend_name: str,
) -> dict[str, Any]:
    """Backend-agnostic training loop.  Returns the same metadata dict as the
    torch-specific implementation."""
    ops = get_ops()  # ops module for the active backend

    if len(features_np) < 4:
        raise ValueError(f"Dataset too small ({len(features_np)} samples).")

    # --- split ---
    (x_tr_np, y_tr_np), (x_val_np, y_val_np) = _split_numpy(
        features_np, targets_np, cfg.val_fraction, cfg.seed
    )

    # --- convert to backend tensors ---
    x_train = ops.to_device(ops.tensor(x_tr_np), cfg.device)
    y_train = ops.to_device(ops.tensor(y_tr_np), cfg.device)
    x_val   = ops.to_device(ops.tensor(x_val_np), cfg.device)
    y_val   = ops.to_device(ops.tensor(y_val_np), cfg.device)

    # --- feature normalisation stats (backend-specific) ---
    ops.manual_seed(cfg.seed)
    feature_mean_np, feature_std_np = ops.compute_feature_stats(x_train)

    # --- build model ---
    in_features = int(x_tr_np.shape[-1])
    model = ops.build_eddy_viscosity_mlp(
        in_features,
        int(cfg.hidden_features),
        int(cfg.n_hidden_layers),
        str(cfg.activation),
        feature_mean_np,
        feature_std_np,
        device=cfg.device,
    )
    loss_fn   = ops.mse_loss_fn()
    optimizer = ops.adam_optimizer(model, cfg.learning_rate)
    scheduler = _build_scheduler_ops(ops, optimizer, cfg)

    rng = np.random.RandomState(int(cfg.seed))
    n_train   = len(x_tr_np)
    batch_size = max(1, min(int(cfg.batch_size), n_train))
    history: list[dict[str, float]] = []
    best_state_np: dict[str, np.ndarray] = {}
    best_epoch = 0
    best_val_mse = float("inf")
    epochs_without_improve = 0
    t0 = time.perf_counter()

    for epoch in range(int(cfg.epochs)):
        ops.train_mode(model)
        epoch_loss = 0.0
        epoch_abs_err = 0.0
        n_seen = 0
        for idx in _iter_minibatches_np(n_train, batch_size, rng):
            xb = ops.index_select(x_train, 0, ops.tensor(idx.astype(np.int32), device=cfg.device))
            yb = ops.index_select(y_train, 0, ops.tensor(idx.astype(np.int32), device=cfg.device))
            batch_loss = ops.train_step(model, loss_fn, optimizer, xb, yb, cfg.gradient_clip_norm)
            epoch_loss += float(batch_loss) * len(idx)
            # compute abs error for monitoring
            with ops.no_grad():
                pred_np = _to_numpy(model(xb))
            epoch_abs_err += float(np.sum(np.abs(pred_np - _to_numpy(yb))))
            n_seen += len(idx)

        train_mse = epoch_loss / max(1, n_seen)
        train_mae = epoch_abs_err / max(1, n_seen)

        ops.eval_mode(model)
        with ops.no_grad():
            val_pred_np = _to_numpy(model(x_val))
        val_true_np = _to_numpy(y_val)
        val_mse = float(np.mean((val_pred_np - val_true_np) ** 2))
        val_mae = float(np.mean(np.abs(val_pred_np - val_true_np)))
        val_r2  = _r2_score_np(val_pred_np, val_true_np)

        current_lr = ops.get_lr(optimizer)
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
            best_val_mse = val_mse
            best_epoch = int(epoch)
            best_state_np = ops.get_state_dict_numpy(model)
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if scheduler is not None:
            ops.scheduler_step(scheduler, val_mse if ops.is_plateau_scheduler(scheduler) else None)
        if cfg.patience is not None and epochs_without_improve > int(cfg.patience):
            break

    # restore best weights
    ops.load_state_dict_numpy(model, best_state_np)

    # save as portable numpy + JSON
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_path), **best_state_np)
    arch_dict = {
        "in_features": in_features,
        "hidden_features": int(cfg.hidden_features),
        "n_hidden_layers": int(cfg.n_hidden_layers),
        "activation": str(cfg.activation),
    }
    import json
    meta = {
        "arch": arch_dict,
        "normalization": {
            "feature_mean": feature_mean_np.tolist(),
            "feature_std": feature_std_np.tolist(),
        },
        "backend": backend_name,
        "format_version": 3,
    }
    out_path.with_suffix(out_path.suffix + ".json").write_text(json.dumps(meta, indent=2))

    elapsed = time.perf_counter() - t0
    final = history[best_epoch] if history else {}
    return {
        "path": str(out_path),
        "arch": arch_dict,
        "config": asdict(cfg),
        "backend": backend_name,
        "n_samples_train": int(n_train),
        "n_samples_val": int(len(x_val_np)),
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_eddy_viscosity_model(
    dataset: EddyViscosityDataset | str | Path,
    out_path: str | Path,
    config: TrainConfig | None = None,
    *,
    backend: str | None = None,
) -> dict[str, Any]:
    """Train an :class:`EddyViscosityMLP` and persist it to *out_path*.

    Args:
        dataset: Either an in-memory :class:`EddyViscosityDataset` or a
            path to one produced by :func:`tensorlbm.ai.save_dataset_pt`.
        out_path: File path where the trained model is saved.
        config: Optional :class:`TrainConfig`.
        backend: Which computation backend to use.  Defaults to the value
            of the ``TENSORLBM_BACKEND`` environment variable (``"torch"``
            if unset).  Valid values: ``"torch"``, ``"paddle"``,
            ``"mindspore"``.

    Returns:
        A metadata dict with loss history, final metrics, arch, path, and
        the ``backend`` key indicating which framework was used.
    """
    cfg = config or TrainConfig()
    backend_name = backend or get_backend()

    if not isinstance(dataset, EddyViscosityDataset):
        dataset = load_dataset_pt(dataset)
    if len(dataset) < 4:
        raise ValueError(f"Dataset is too small to train ({len(dataset)} samples).")

    features_np = _to_numpy(dataset.features)
    targets_np  = _to_numpy(dataset.targets)
    out_path = Path(out_path)

    with using_backend(backend_name):
        # ------------------------------------------------------------------
        # PyTorch path: keep existing behaviour exactly (uses .pt format)
        # ------------------------------------------------------------------
        if backend_name == "torch":
            device = torch.device(cfg.device)
            train_ds, val_ds = dataset.split(cfg.val_fraction, seed=cfg.seed)
            x_train = train_ds.features.to(device)
            y_train = train_ds.targets.to(device)
            x_val   = val_ds.features.to(device)
            y_val   = val_ds.targets.to(device)

            torch.manual_seed(int(cfg.seed))
            arch = ModelArch(
                in_features=int(x_train.shape[-1]),
                hidden_features=int(cfg.hidden_features),
                n_hidden_layers=int(cfg.n_hidden_layers),
                activation=str(cfg.activation),
            )
            model = EddyViscosityMLP(arch).to(device)
            feature_mean = x_train.mean(dim=0)
            feature_std  = x_train.std(dim=0, unbiased=False).clamp_min(1e-6)
            model.set_feature_stats(feature_mean, feature_std)
            optimizer = optim.Adam(model.parameters(), lr=float(cfg.learning_rate))
            loss_fn   = nn.MSELoss()
            scheduler = _build_scheduler(optimizer, int(cfg.epochs), str(cfg.lr_scheduler))

            generator = torch.Generator(device="cpu").manual_seed(int(cfg.seed))
            history: list[dict[str, float]] = []
            n_train    = x_train.shape[0]
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
                            model.parameters(), max_norm=float(cfg.gradient_clip_norm),
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
                    val_r2  = _r2_score(val_pred, y_val)
                current_lr = float(optimizer.param_groups[0]["lr"])
                met = {
                    "epoch": int(epoch),
                    "train_mse": float(train_mse),
                    "train_mae": float(train_mae),
                    "val_mse": float(val_mse),
                    "val_mae": float(val_mae),
                    "val_r2": float(val_r2),
                    "lr": float(current_lr),
                }
                history.append(met)
                if val_mse < (best_val_mse - 1e-12):
                    best_val_mse = val_mse
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
                "backend": "torch",
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

        # ------------------------------------------------------------------
        # PaddlePaddle / MindSpore path
        # ------------------------------------------------------------------
        return _train_with_backend(features_np, targets_np, out_path, cfg, backend_name)
