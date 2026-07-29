"""SUBOFF 3D flow-field reconstruction training module.

Encapsulates the pretraining (main14) and fine-tuning (main38) workflows
as callable library functions with dataclass-based configuration.
"""
from __future__ import annotations

import csv
import logging
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from einops import repeat

from .suboff_utils import (
    build_suboff_model,
    default_suboff_device,
    ensure_dir,
    get_suboff_coords,
    load_checkpoint,
    pointwise_rel_loss,
    save_checkpoint,
    _move_to_device,
)


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SuboffTrainConfig:
    """Hyper-parameters for SUBOFF pretraining (main14-style)."""
    lr: float = 6e-4
    iters: int = 125_000
    batch_size: int = 4
    ckpt_every: int = 2500
    log_dir: str = "./"
    resume: bool = False
    path_to_resume: str = ""
    # Mask ratio sampling (truncated normal)
    mask_ratio_min: float = 0.49
    mask_ratio_max: float = 0.99
    mask_ratio_mu: float = 0.55
    mask_ratio_std: float = 0.25
    # Data
    data_dir: str = ""          # NPY snapshot directory
    n_train: int = 1250
    n_test: int = 250
    # Device
    device: str = field(default_factory=default_suboff_device)


@dataclass(frozen=True)
class SuboffFinetuneConfig:
    """Hyper-parameters for SUBOFF fine-tuning (main38-style)."""
    lr: float = 6e-4
    iters: int = 31_250
    batch_size: int = 1
    ckpt_every: int = 3125
    log_dir: str = "./"
    resume: bool = True
    path_to_resume: str = ""    # Pretrained checkpoint path
    # Mask ratio sampling
    mask_ratio_min: float = 0.49
    mask_ratio_max: float = 0.99
    mask_ratio_mu: float = 0.55
    mask_ratio_std: float = 0.25
    # Data
    data_dir: str = ""          # Fine-tuning NPY snapshot directory
    n_train: int = 3            # Fine-tuning uses fewer snapshots
    n_test: int = 2
    # Device
    device: str = field(default_factory=default_suboff_device)


# ── Data loading ─────────────────────────────────────────────────────────────

def _load_npy_channel(data_dir: str, channel: str, indices: list[int]) -> torch.Tensor:
    """Load NPY snapshots for one channel at given time indices.

    Args:
        data_dir: Root directory containing p/ux/uy/uz subdirs.
        channel: Channel name (p, ux, uy, uz).
        indices: List of snapshot indices (e.g. [0,1,...,1249] for train).

    Returns:
        Tensor of shape [len(indices), total_points, 1].
    """
    ch_dir = os.path.join(data_dir, channel)
    result = np.empty((len(indices), 500_000), dtype=np.float32)
    for i, idx in enumerate(indices):
        arr = np.load(os.path.join(ch_dir, f"{idx}.npy")).astype(np.float32)
        # Crop to SUBOFF region [49:149, :, 49:149] and flatten
        arr = arr[49:149, :, 49:149].flatten()
        result[i, :] = arr
    return torch.as_tensor(result, dtype=torch.float32).unsqueeze(dim=-1)


def _load_npy_channel_ori27(data_dir: str, channel: str, indices: list[int]) -> torch.Tensor:
    """Load NPY snapshots for ori27 (100 XY slices of 5000 points each)."""
    ch_dir = os.path.join(data_dir, channel)
    result = np.empty((len(indices), 500_000), dtype=np.float32)
    for i, idx in enumerate(indices):
        arr = np.load(os.path.join(ch_dir, f"{idx}.npy")).astype(np.float32)
        arr = arr[49:149, :, 49:149]
        for j in range(100):
            result[i, j * 5000:(j + 1) * 5000] = arr[:, :, j].flatten()
    return torch.as_tensor(result, dtype=torch.float32).unsqueeze(dim=-1)


def _load_npy_channel_ori28_addition(data_dir: str, channel: str, indices: list[int]) -> torch.Tensor:
    """Load NPY snapshots for ori28_addition (50 XZ slices of 10000 points each)."""
    ch_dir = os.path.join(data_dir, channel)
    result = np.empty((len(indices), 500_000), dtype=np.float32)
    for i, idx in enumerate(indices):
        arr = np.load(os.path.join(ch_dir, f"{idx}.npy")).astype(np.float32)
        arr = arr[49:149, :, 49:149]
        for j in range(50):
            result[i, j * 10000:(j + 1) * 10000] = arr[:, j, :].flatten()
    return torch.as_tensor(result, dtype=torch.float32).unsqueeze(dim=-1)


def _load_multi_re_data(data_dir: str, n_train: int, n_test: int,
                        load_funcs: list) -> tuple[torch.Tensor, torch.Tensor]:
    """Load 3 Re-group data and merge into [3, n, 500000, 4] tensors.

    Args:
        data_dir: NPY snapshot root directory.
        n_train: Number of training snapshots (indices 0..n_train-1).
        n_test: Number of test snapshots (indices 1250..1250+n_test-1).
        load_funcs: List of 3 channel-loading functions (ori27, ori28, ori28_addition).

    Returns:
        (train_data, test_data) each of shape [3, n, 500000, 4].
    """
    channels = ("p", "ux", "uy", "uz")
    train_indices = list(range(n_train))
    # Determine total available snapshots from the p directory
    p_dir = os.path.join(data_dir, "p")
    total_snaps = len([f for f in os.listdir(p_dir) if f.endswith(".npy")]) if os.path.isdir(p_dir) else 1500
    # Test indices start after training data, capped by total available
    test_start = min(n_train, total_snaps - n_test)
    test_indices = list(range(test_start, test_start + n_test))
    # Safety: cap indices to available range
    train_indices = [i for i in train_indices if i < total_snaps]
    test_indices = [i for i in test_indices if i < total_snaps]

    all_train, all_test = [], []
    for func in load_funcs:
        train_parts = [func(data_dir, ch, train_indices) for ch in channels]
        train_cat = torch.cat(train_parts, dim=-1)  # [n_train, 500000, 4]
        test_parts = [func(data_dir, ch, test_indices) for ch in channels]
        test_cat = torch.cat(test_parts, dim=-1)    # [n_test, 500000, 4]
        all_train.append(train_cat)
        all_test.append(test_cat)

    train_data = torch.stack(all_train)  # [3, n_train, 500000, 4]
    test_data = torch.stack(all_test)    # [3, n_test, 500000, 4]
    return train_data, test_data


def _load_train_data_pretrain(cfg: SuboffTrainConfig):
    """Load pretraining data (3 Re groups: ori27, ori28, ori28_addition)."""
    from .suboff_dataset import CylinderDatasetMultiRe14

    train_data, test_data = _load_multi_re_data(
        cfg.data_dir, cfg.n_train, cfg.n_test,
        [_load_npy_channel_ori27, _load_npy_channel, _load_npy_channel_ori28_addition],
    )
    tw = 1
    train_dataset = CylinderDatasetMultiRe14(train_data, tw, push_forward=0)
    test_dataset = CylinderDatasetMultiRe14(test_data, tw, push_forward=0)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=True)
    return train_loader, test_loader


def _load_train_data_finetune(cfg: SuboffFinetuneConfig):
    """Load fine-tuning data (3 Re groups: ori73, ori74, ori74_addition)."""
    from .suboff_dataset import CylinderDatasetMultiRe14

    # Fine-tuning uses same data layout but different n_train/n_test
    train_data, test_data = _load_multi_re_data(
        cfg.data_dir, cfg.n_train, cfg.n_test,
        [_load_npy_channel_ori27, _load_npy_channel, _load_npy_channel_ori28_addition],
    )
    tw = 1
    train_dataset = CylinderDatasetMultiRe14(train_data, tw, push_forward=0)
    test_dataset = CylinderDatasetMultiRe14(test_data, tw, push_forward=0)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=True)
    return train_loader, test_loader


# ── Position coordinates ────────────────────────────────────────────────────

def _prepare_positions(batch_size: int, device: torch.device):
    """Prepare 3 sets of position coordinates for the 3 data groups."""
    from .suboff_coord import coord_ori27, coord_ori28, coord_ori28_addition

    pos_all1 = torch.as_tensor(coord_ori27(), dtype=torch.float32)
    pos_all2 = torch.as_tensor(coord_ori28(), dtype=torch.float32)
    pos_all3 = torch.as_tensor(coord_ori28_addition(), dtype=torch.float32)

    pos_all1 = repeat(pos_all1, "n d -> b n d", b=batch_size)
    pos_all2 = repeat(pos_all2, "n d -> b n d", b=batch_size)
    pos_all3 = repeat(pos_all3, "n d -> b n d", b=batch_size)

    pos_all1 = _move_to_device(pos_all1, device)
    pos_all2 = _move_to_device(pos_all2, device)
    pos_all3 = _move_to_device(pos_all3, device)
    return pos_all1, pos_all2, pos_all3


# ── Mask ratio generator ────────────────────────────────────────────────────

def _make_mask_ratio_generator(cfg):
    """Create truncated normal mask ratio sampler."""
    import scipy.stats as stats
    return stats.truncnorm(
        (cfg.mask_ratio_min - cfg.mask_ratio_mu) / cfg.mask_ratio_std,
        (cfg.mask_ratio_max - cfg.mask_ratio_mu) / cfg.mask_ratio_std,
        loc=cfg.mask_ratio_mu,
        scale=cfg.mask_ratio_std,
    )


# ── Shared training loop ────────────────────────────────────────────────────

def _run_training_loop(
    encoder, decoder, enc_optim, enc_scheduler,
    train_loader, test_loader,
    pos_all1, pos_all2, pos_all3,
    mask_ratio_generator,
    cfg, checkpoint_dir, log_prefix, logger,
    progress_callback=None,
) -> dict[str, Any]:
    """Core training loop shared by pretraining and fine-tuning.

    Args:
        encoder/decoder: Model modules.
        enc_optim: Optimizer.
        enc_scheduler: LR scheduler.
        train_loader/test_loader: Data loaders.
        pos_all1/2/3: Position coordinate tensors.
        mask_ratio_generator: Truncated normal sampler.
        cfg: TrainConfig or FinetuneConfig.
        checkpoint_dir: Where to save checkpoints.
        log_prefix: Prefix for CSV/log files (e.g. "main14", "main38").
        logger: Python logger instance.
        progress_callback: Optional callable(dict) invoked each iteration
            with {phase, epoch, total, loss, lr, mse} for live progress.

    Returns:
        Dict with training history and final metrics.
    """
    device = torch.device(cfg.device)

    start_n_iter = 0
    if cfg.resume and cfg.path_to_resume:
        ckpt = load_checkpoint(cfg.path_to_resume, map_location=device)
        encoder.load_state_dict(ckpt["encoder"])
        decoder.load_state_dict(ckpt["decoder"])
        start_n_iter = ckpt.get("n_iter", 0)
        if "enc_optim" in ckpt:
            enc_optim.load_state_dict(ckpt["enc_optim"])
        if "enc_sched" in ckpt:
            enc_scheduler.load_state_dict(ckpt["enc_sched"])
        print("last checkpoint restored")

    # CSV logging
    train_csv = os.path.join(cfg.log_dir, f"{log_prefix}_train.csv")
    test_csv = os.path.join(cfg.log_dir, f"{log_prefix}_test.csv")
    if not cfg.resume:
        with open(train_csv, "w", newline="") as f:
            csv.writer(f).writerow(["batch", "loss1"])
        with open(test_csv, "w", newline="") as f:
            csv.writer(f).writerow(["batch", "loss1"])

    history: list[dict[str, float]] = []
    n_iter = start_n_iter
    best_loss = float("inf")

    train_data_iter = iter(train_loader)
    pbar = tqdm(total=cfg.iters)
    pbar.update(n_iter)

    while True:
        encoder.train()
        decoder.train()

        try:
            data = next(train_data_iter)
        except StopIteration:
            del train_data_iter
            train_data_iter = iter(train_loader)
            data = next(train_data_iter)

        # Data preparation: random sub-sampling from 3 groups
        x1, x2, x3, _ = data
        num1 = random.randint(0, 99)
        num2 = random.randint(0, 99)
        num3 = random.randint(0, 49)
        x1 = x1[:, :, num1 * 5000:(num1 + 1) * 5000, :]
        x2 = x2[:, :, num2 * 5000:(num2 + 1) * 5000, :]
        x3 = x3[:, :, num3 * 10000:(num3 + 1) * 10000, :]
        temp = torch.cat((x1, x2, x3), dim=2)

        # Random mask sampling
        mask_rate = float(mask_ratio_generator.rvs(1)[0])
        num = math.ceil(int(temp.shape[-2]) * (1 - mask_rate))
        index = random.sample(range(0, temp.shape[-2] - 1), num)
        x = temp[:, :, index, :]
        y = temp.squeeze(dim=1)

        pos1 = pos_all1[:, num1 * 5000:(num1 + 1) * 5000, :]
        pos2 = pos_all2[:, num2 * 5000:(num2 + 1) * 5000, :]
        pos3 = pos_all3[:, num3 * 10000:(num3 + 1) * 10000, :]
        pos = torch.cat((pos1, pos2, pos3), dim=1)
        prop_pos = pos
        input_pos = pos[:, index, :]

        x, y = x.to(device), y.to(device)

        z = encoder.forward(x, input_pos)
        pred = decoder.forward(z, prop_pos, input_pos)

        all_loss = pointwise_rel_loss(pred, y, p=2)
        mse_loss_torch = F.mse_loss(pred, y)
        loss = all_loss

        enc_optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 2.0)
        enc_optim.step()
        enc_scheduler.step()

        loss_val = float(loss.item())
        mse_val = float(mse_loss_torch.item())
        history.append({"iter": n_iter, "train_loss": loss_val, "train_mse": mse_val})

        # CSV write
        with open(train_csv, "a", newline="") as f:
            csv.writer(f).writerow([n_iter, loss_val * 1e4])

        pbar.set_description(
            f"loss(1e-4):{loss_val*1e4:.3f}|mse(1e-4):{mse_val*1e4:.3f}|lr:{enc_optim.param_groups[0]['lr']:.3e}|n:{num}")
        pbar.update(1)

        # Live progress callback
        if progress_callback:
            progress_callback({
                "phase": "training",
                "epoch": n_iter,
                "total": cfg.iters,
                "loss": loss_val * 1e4,
                "mse": mse_val * 1e4,
                "lr": enc_optim.param_groups[0]['lr'],
            })

        n_iter += 1

        # Periodic testing + checkpoint
        if (n_iter - 1) % cfg.ckpt_every == 0 or n_iter >= cfg.iters:
            # Set phase to testing BEFORE the test loop so frontend catches it
            if progress_callback:
                progress_callback({
                    "phase": "testing",
                    "epoch": n_iter - 1,
                    "total": cfg.iters,
                    "loss": None,
                    "mse": None,
                    "lr": enc_optim.param_groups[0]['lr'],
                    "best_loss": best_loss,
                })
            encoder.eval()
            decoder.eval()
            all_avg_loss = []

            for data in tqdm(test_loader, desc="Testing"):
                x1, x2, x3, _ = data
                num1 = random.randint(0, 99)
                num2 = random.randint(0, 99)
                num3 = random.randint(0, 49)
                x1 = x1[:, :, num1 * 5000:(num1 + 1) * 5000, :]
                x2 = x2[:, :, num2 * 5000:(num2 + 1) * 5000, :]
                x3 = x3[:, :, num3 * 10000:(num3 + 1) * 10000, :]
                temp = torch.cat((x1, x2, x3), dim=2)
                mask_rate = float(mask_ratio_generator.rvs(1)[0])
                num = math.ceil(int(temp.shape[-2]) * (1 - mask_rate))
                index = random.sample(range(0, temp.shape[-2] - 1), num)
                x = temp[:, :, index, :]
                y = temp.squeeze(dim=1)

                pos1 = pos_all1[:, num1 * 5000:(num1 + 1) * 5000, :]
                pos2 = pos_all2[:, num2 * 5000:(num2 + 1) * 5000, :]
                pos3 = pos_all3[:, num3 * 10000:(num3 + 1) * 10000, :]
                pos = torch.cat((pos1, pos2, pos3), dim=1)
                prop_pos = pos
                input_pos = pos[:, index, :]

                x, y = x.to(device), y.to(device)

                with torch.no_grad():
                    z = encoder.forward(x, input_pos)
                    pred = decoder.forward(z, prop_pos, input_pos)
                    loss = pointwise_rel_loss(pred, y, p=2)
                    all_avg_loss.append(float(loss.item()))

            test_avg = float(np.mean(all_avg_loss)) * 1e4
            with open(test_csv, "a", newline="") as f:
                csv.writer(f).writerow([n_iter - 1, test_avg])

            # Testing phase callback
            if progress_callback:
                progress_callback({
                    "phase": "testing",
                    "epoch": n_iter - 1,
                    "total": cfg.iters,
                    "loss": test_avg,
                    "mse": None,
                    "lr": enc_optim.param_groups[0]['lr'],
                    "best_loss": best_loss,
                })

            if logger:
                logger.info("Testing")
                logger.info(f"Current iteration: {n_iter - 1}")
                logger.info(f"Testing avg loss (1e-4): {test_avg}")

            # Save checkpoint
            ckpt = {
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "n_iter": n_iter,
                "enc_optim": enc_optim.state_dict(),
                "enc_sched": enc_scheduler.state_dict(),
            }
            save_checkpoint(ckpt, os.path.join(checkpoint_dir, f"model_checkpoint{n_iter - 1}.ckpt"))
            del ckpt

            if test_avg < best_loss:
                best_loss = test_avg

            if n_iter >= cfg.iters:
                break

    pbar.close()
    return {
        "config": asdict(cfg),
        "history": history,
        "best_loss_1e4": best_loss,
        "final_iter": n_iter,
        "checkpoint_dir": checkpoint_dir,
    }


# ── Public API ───────────────────────────────────────────────────────────────

def train_suboff(cfg: SuboffTrainConfig | None = None, progress_callback=None) -> dict[str, Any]:
    """Run SUBOFF pretraining (main14-style).

    Args:
        cfg: Training configuration. Uses defaults if None.
        progress_callback: Optional callable(dict) for live progress updates.

    Returns:
        Dict with training history, best loss, checkpoint directory.
    """
    if cfg is None:
        cfg = SuboffTrainConfig()

    device = torch.device(cfg.device)
    encoder, decoder = build_suboff_model(device)

    checkpoint_dir = os.path.join(cfg.log_dir, "model_ckpt14")
    ensure_dir(checkpoint_dir)

    # Logger
    logger = logging.getLogger("LOG")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(cfg.log_dir, "logging_info14.txt"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
    logger.info("=======Option used=======")
    for k, v in asdict(cfg).items():
        logger.info(f"{k}: {v}")

    # Optimizer + scheduler
    enc_optim = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=cfg.lr, weight_decay=1e-4,
    )
    enc_scheduler = OneCycleLR(
        enc_optim, max_lr=cfg.lr, total_steps=cfg.iters,
        div_factor=1e4, pct_start=0.3, final_div_factor=1e4,
    )

    mask_ratio_generator = _make_mask_ratio_generator(cfg)
    train_loader, test_loader = _load_train_data_pretrain(cfg)
    pos_all1, pos_all2, pos_all3 = _prepare_positions(cfg.batch_size, device)

    return _run_training_loop(
        encoder, decoder, enc_optim, enc_scheduler,
        train_loader, test_loader,
        pos_all1, pos_all2, pos_all3,
        mask_ratio_generator, cfg,
        checkpoint_dir, "main14", logger,
        progress_callback=progress_callback,
    )


def finetune_suboff(cfg: SuboffFinetuneConfig | None = None, progress_callback=None) -> dict[str, Any]:
    """Run SUBOFF fine-tuning (main38-style).

    Loads a pretrained checkpoint and fine-tunes on new Re工况 data.

    Args:
        cfg: Fine-tuning configuration. Uses defaults if None.
        progress_callback: Optional callable(dict) for live progress updates.

    Returns:
        Dict with training history, best loss, checkpoint directory.
    """
    if cfg is None:
        cfg = SuboffFinetuneConfig()

    device = torch.device(cfg.device)
    encoder, decoder = build_suboff_model(device)
    # ===================== 新增代码块开始 =====================
    # 1. 手动加载预训练模型权重，只加载网络，不加载优化器/调度器
    pretrain_ckpt_path = cfg.path_to_resume
    if pretrain_ckpt_path:
        ckpt = load_checkpoint(pretrain_ckpt_path, map_location=device)
        encoder.load_state_dict(ckpt["encoder"], strict=False)
        decoder.load_state_dict(ckpt["decoder"], strict=False)
        print(f"[Finetune] Load pretrain weight only: {pretrain_ckpt_path}, skip optimizer & scheduler")

    # 2. 重建冻结dataclass，强制关闭resume，让底层不再恢复优化器
    cfg = SuboffFinetuneConfig(
        lr=cfg.lr,
        iters=cfg.iters,
        batch_size=cfg.batch_size,
        ckpt_every=cfg.ckpt_every,
        log_dir=cfg.log_dir,
        resume=False,       # 关键：关闭续训加载分支
        path_to_resume="",  # 关键：清空路径，不触发resume逻辑
        mask_ratio_min=cfg.mask_ratio_min,
        mask_ratio_max=cfg.mask_ratio_max,
        mask_ratio_mu=cfg.mask_ratio_mu,
        mask_ratio_std=cfg.mask_ratio_std,
        data_dir=cfg.data_dir,
        n_train=cfg.n_train,
        n_test=cfg.n_test,
        device=cfg.device
    )
    # ===================== 新增代码块结束 =====================


    checkpoint_dir = os.path.join(cfg.log_dir, "model_ckpt38")
    ensure_dir(checkpoint_dir)

    # Logger
    logger = logging.getLogger("LOG")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(cfg.log_dir, "logging_info38.txt"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
    logger.info("=======Option used=======")
    for k, v in asdict(cfg).items():
        logger.info(f"{k}: {v}")

    # Optimizer + scheduler
    enc_optim = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=cfg.lr, weight_decay=1e-4,
    )
    enc_scheduler = OneCycleLR(
        enc_optim, max_lr=cfg.lr, total_steps=cfg.iters,
        div_factor=1e4, pct_start=0.3, final_div_factor=1e4,
    )

    mask_ratio_generator = _make_mask_ratio_generator(cfg)
    train_loader, test_loader = _load_train_data_finetune(cfg)
    pos_all1, pos_all2, pos_all3 = _prepare_positions(cfg.batch_size, device)

    return _run_training_loop(
        encoder, decoder, enc_optim, enc_scheduler,
        train_loader, test_loader,
        pos_all1, pos_all2, pos_all3,
        mask_ratio_generator, cfg,
        checkpoint_dir, "main38", logger,
        progress_callback=progress_callback,
    )
