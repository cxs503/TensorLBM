"""SUBOFF DDP (DistributedDataParallel) multi-card training.

Uses TCCL backend for inter-card gradient sync on SDAA.
Each card processes different data batches; gradients are all-reduced.

Usage:
    # 4-card pretraining
    torchrun --nproc_per_node=4 -m tensorlbm.ai.suboff_train_ddp \
        --mode pretrain --iters 125000 --data_dir /root/LBM-Platform/suboff_all/suboff8

    # 8-card fine-tuning
    torchrun --nproc_per_node=8 -m tensorlbm.ai.suboff_train_ddp \
        --mode finetune --iters 31250 --data_dir /path/to/finetune_data

    # 32-card full cluster
    torchrun --nproc_per_node=32 -m tensorlbm.ai.suboff_train_ddp \
        --mode pretrain --iters 125000
"""

from __future__ import annotations

import os
import sys
import math
import random
import logging
import time
import argparse
from pathlib import Path
from dataclasses import asdict

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from .suboff_utils import (
    build_suboff_model,
    ensure_dir,
    load_checkpoint,
    save_checkpoint,
    pointwise_rel_loss,
    _move_to_device,
)
from .suboff_train import (
    SuboffTrainConfig,
    SuboffFinetuneConfig,
    _load_train_data_pretrain,
    _load_train_data_finetune,
    _prepare_positions,
    _make_mask_ratio_generator,
    _load_multi_re_data,
)
from .suboff_dataset import CylinderDatasetMultiRe14


def _setup_ddp():
    """Initialize DDP process group from torchrun env vars."""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        dist.init_process_group(
            backend="tccl",
            rank=rank,
            world_size=world_size,
        )

    device = torch.device(f"sdaa:{local_rank}")
    torch.sdaa.set_device(device)
    return rank, world_size, local_rank, device


def _cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _is_main(rank):
    return rank == 0


def _make_ddp_loader(dataset, batch_size, rank, world_size, shuffle=True):
    """Create a DataLoader with DistributedSampler."""
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        drop_last=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=2,
        pin_memory=False,
    )
    return loader, sampler


def train_ddp(
    mode: str = "pretrain",
    iters: int = 125000,
    lr: float = 6e-4,
    batch_size: int = 4,
    data_dir: str = "",
    n_train: int = 1250,
    n_test: int = 250,
    ckpt_every: int = 2500,
    log_dir: str = "./",
    resume: bool = False,
    path_to_resume: str = "",
):
    """Run DDP training across multiple SDAA cards."""
    rank, world_size, local_rank, device = _setup_ddp()

    if _is_main(rank):
        print(f"{'=' * 60}")
        print(f"DDP Training: {world_size} cards | mode={mode}")
        print(f"{'=' * 60}")

    # ── Config ──
    if mode == "pretrain":
        cfg = SuboffTrainConfig(
            lr=lr,
            iters=iters,
            batch_size=batch_size,
            data_dir=data_dir,
            n_train=n_train,
            n_test=n_test,
            ckpt_every=ckpt_every,
            log_dir=log_dir,
            resume=resume,
            path_to_resume=path_to_resume,
            device=f"sdaa:{local_rank}",
        )
        ckpt_subdir = "model_ckpt14"
        log_prefix = "main14"
    else:
        cfg = SuboffFinetuneConfig(
            lr=lr,
            iters=iters,
            batch_size=batch_size,
            data_dir=data_dir,
            n_train=n_train,
            n_test=n_test,
            ckpt_every=ckpt_every,
            log_dir=log_dir,
            resume=resume,
            path_to_resume=path_to_resume,
            device=f"sdaa:{local_rank}",
        )
        ckpt_subdir = "model_ckpt38"
        log_prefix = "main38"

    # ── Model ──
    encoder, decoder = build_suboff_model(device)

    # Load pretrained weights (for finetune or resume)
    if cfg.resume and cfg.path_to_resume:
        ckpt = load_checkpoint(cfg.path_to_resume, map_location=device)
        encoder.load_state_dict(ckpt["encoder"], strict=False)
        decoder.load_state_dict(ckpt["decoder"], strict=False)
        if _is_main(rank):
            print(f"[Rank 0] Loaded checkpoint: {cfg.path_to_resume}")

    # ── Wrap with DDP ──
    if world_size > 1:
        encoder = DDP(encoder, device_ids=[local_rank])
        decoder = DDP(decoder, device_ids=[local_rank])

    # ── Data loading with DistributedSampler ──
    if mode == "pretrain":
        train_dataset, test_dataset = _load_train_data_pretrain(cfg)
        # _load_train_data_pretrain returns loaders, not datasets
        # We need to rebuild with DistributedSampler
        from .suboff_train import (
            _load_multi_re_data,
            _load_npy_channel_ori27,
            _load_npy_channel,
            _load_npy_channel_ori28_addition,
        )

        train_data, test_data = _load_multi_re_data(
            cfg.data_dir,
            cfg.n_train,
            cfg.n_test,
            [_load_npy_channel_ori27, _load_npy_channel, _load_npy_channel_ori28_addition],
        )
        train_dataset = CylinderDatasetMultiRe14(train_data, tw=1, push_forward=0)
        test_dataset = CylinderDatasetMultiRe14(test_data, tw=1, push_forward=0)
    else:
        train_data, test_data = _load_multi_re_data(
            cfg.data_dir,
            cfg.n_train,
            cfg.n_test,
            [_load_npy_channel_ori27, _load_npy_channel, _load_npy_channel_ori28_addition],
        )
        train_dataset = CylinderDatasetMultiRe14(train_data, tw=1, push_forward=0)
        test_dataset = CylinderDatasetMultiRe14(test_data, tw=1, push_forward=0)

    train_loader, train_sampler = _make_ddp_loader(
        train_dataset, cfg.batch_size, rank, world_size, shuffle=True
    )
    test_loader, _ = _make_ddp_loader(test_dataset, cfg.batch_size, rank, world_size, shuffle=False)

    # ── Positions ──
    pos_all1, pos_all2, pos_all3 = _prepare_positions(cfg.batch_size, device)

    # ── Optimizer + Scheduler ──
    enc_optim = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=cfg.lr,
        weight_decay=1e-4,
    )
    from torch.optim.lr_scheduler import OneCycleLR

    enc_scheduler = OneCycleLR(
        enc_optim,
        max_lr=cfg.lr,
        total_steps=cfg.iters,
        div_factor=1e4,
        pct_start=0.3,
        final_div_factor=1e4,
    )

    mask_ratio_generator = _make_mask_ratio_generator(cfg)

    # ── Logger (rank 0 only) ──
    logger = None
    if _is_main(rank):
        checkpoint_dir = os.path.join(cfg.log_dir, ckpt_subdir)
        ensure_dir(checkpoint_dir)
        logger = logging.getLogger(f"LOG_{log_prefix}")
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(cfg.log_dir, f"logging_info{log_prefix[-2:]}.txt"))
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(fh)
        logger.info(f"DDP training: {world_size} cards, mode={mode}, iters={iters}")
    else:
        checkpoint_dir = os.path.join(cfg.log_dir, ckpt_subdir)

    # ── Training loop ──
    n_iter = 0
    best_loss = float("inf")
    train_data_iter = iter(train_loader)

    pbar = tqdm(total=cfg.iters, disable=not _is_main(rank))
    pbar.update(n_iter)

    while True:
        encoder.train()
        decoder.train()
        train_sampler.set_epoch(n_iter)  # Important for shuffling in DDP

        try:
            data = next(train_data_iter)
        except StopIteration:
            del train_data_iter
            train_data_iter = iter(train_loader)
            data = next(train_data_iter)

        # Data preparation
        x1, x2, x3, _ = data
        num1 = random.randint(0, 99)
        num2 = random.randint(0, 99)
        num3 = random.randint(0, 49)
        x1 = x1[:, :, num1 * 5000 : (num1 + 1) * 5000, :]
        x2 = x2[:, :, num2 * 5000 : (num2 + 1) * 5000, :]
        x3 = x3[:, :, num3 * 10000 : (num3 + 1) * 10000, :]
        temp = torch.cat((x1, x2, x3), dim=2)

        mask_rate = float(mask_ratio_generator.rvs(1)[0])
        num = math.ceil(int(temp.shape[-2]) * (1 - mask_rate))
        index = random.sample(range(0, temp.shape[-2] - 1), num)
        x = temp[:, :, index, :]
        y = temp.squeeze(dim=1)

        pos1 = pos_all1[:, num1 * 5000 : (num1 + 1) * 5000, :]
        pos2 = pos_all2[:, num2 * 5000 : (num2 + 1) * 5000, :]
        pos3 = pos_all3[:, num3 * 10000 : (num3 + 1) * 10000, :]
        pos = torch.cat((pos1, pos2, pos3), dim=1)
        prop_pos = pos
        input_pos = pos[:, index, :]

        x, y = x.to(device), y.to(device)

        # ★ Use __call__ (not .forward) so DDP hooks fire
        z = encoder(x, input_pos)
        pred = decoder(z, prop_pos, input_pos)

        loss = pointwise_rel_loss(pred, y, p=2)
        mse_loss = torch.nn.functional.mse_loss(pred, y)

        enc_optim.zero_grad()
        loss.backward()  # DDP auto-syncs gradients here
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 2.0)
        enc_optim.step()
        enc_scheduler.step()

        loss_val = float(loss.item())
        mse_val = float(mse_loss.item())

        if _is_main(rank):
            pbar.set_description(
                f"loss(1e-4):{loss_val * 1e4:.3f}|mse(1e-4):{mse_val * 1e4:.3f}|"
                f"lr:{enc_optim.param_groups[0]['lr']:.3e}"
            )
            pbar.update(1)

        n_iter += 1

        # Periodic testing + checkpoint (rank 0 only)
        if (n_iter - 1) % cfg.ckpt_every == 0 or n_iter >= cfg.iters:
            # Set phase to testing
            if _is_main(rank):
                print(f"\n[Iter {n_iter}] Testing phase...")

            encoder.eval()
            decoder.eval()
            all_avg_loss = []

            with torch.no_grad():
                for data in tqdm(test_loader, desc="Testing", disable=not _is_main(rank)):
                    x1, x2, x3, _ = data
                    num1 = random.randint(0, 99)
                    num2 = random.randint(0, 99)
                    num3 = random.randint(0, 49)
                    x1 = x1[:, :, num1 * 5000 : (num1 + 1) * 5000, :]
                    x2 = x2[:, :, num2 * 5000 : (num2 + 1) * 5000, :]
                    x3 = x3[:, :, num3 * 10000 : (num3 + 1) * 10000, :]
                    temp = torch.cat((x1, x2, x3), dim=2)
                    mask_rate = float(mask_ratio_generator.rvs(1)[0])
                    num = math.ceil(int(temp.shape[-2]) * (1 - mask_rate))
                    index = random.sample(range(0, temp.shape[-2] - 1), num)
                    x = temp[:, :, index, :]
                    y = temp.squeeze(dim=1)

                    pos1 = pos_all1[:, num1 * 5000 : (num1 + 1) * 5000, :]
                    pos2 = pos_all2[:, num2 * 5000 : (num2 + 1) * 5000, :]
                    pos3 = pos_all3[:, num3 * 10000 : (num3 + 1) * 10000, :]
                    pos = torch.cat((pos1, pos2, pos3), dim=1)
                    prop_pos = pos
                    input_pos = pos[:, index, :]

                    x, y = x.to(device), y.to(device)
                    z = encoder(x, input_pos)
                    pred = decoder(z, prop_pos, input_pos)
                    loss = pointwise_rel_loss(pred, y, p=2)
                    all_avg_loss.append(float(loss.item()))

            # All-reduce test loss across cards
            if world_size > 1:
                loss_tensor = torch.tensor([np.mean(all_avg_loss)], device=device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                test_avg = float(loss_tensor.item()) * 1e4
            else:
                test_avg = float(np.mean(all_avg_loss)) * 1e4

            if _is_main(rank):
                print(f"[Iter {n_iter}] Test loss (1e-4): {test_avg:.3f}")

            if test_avg < best_loss:
                best_loss = test_avg
                if _is_main(rank):
                    # Save checkpoint (unwrap DDP)
                    enc_state = (
                        encoder.module.state_dict() if world_size > 1 else encoder.state_dict()
                    )
                    dec_state = (
                        decoder.module.state_dict() if world_size > 1 else decoder.state_dict()
                    )
                    save_checkpoint(
                        enc_state,
                        dec_state,
                        enc_optim,
                        enc_scheduler,
                        n_iter,
                        best_loss,
                        os.path.join(checkpoint_dir, f"model_checkpoint{n_iter}.ckpt"),
                    )
                    print(
                        f"[Rank 0] Saved checkpoint: model_checkpoint{n_iter}.ckpt (best={best_loss:.3f})"
                    )

            encoder.train()
            decoder.train()

        if n_iter >= cfg.iters:
            break

    pbar.close()

    if _is_main(rank):
        print(f"\n{'=' * 60}")
        print(f"Training complete: {n_iter} iters, best_loss(1e-4)={best_loss:.3f}")
        print(f"Checkpoints: {checkpoint_dir}")
        print(f"{'=' * 60}")

    _cleanup_ddp()
    return {"best_loss_1e4": best_loss, "final_iter": n_iter, "checkpoint_dir": checkpoint_dir}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SUBOFF DDP Multi-Card Training")
    parser.add_argument("--mode", choices=["pretrain", "finetune"], default="pretrain")
    parser.add_argument("--iters", type=int, default=125000)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--data-dir", default="/root/LBM-Platform/suboff_all/suboff8")
    parser.add_argument("--n-train", type=int, default=1250)
    parser.add_argument("--n-test", type=int, default=250)
    parser.add_argument("--ckpt-every", type=int, default=2500)
    parser.add_argument("--log-dir", default="./")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--path-to-resume", default="")
    args = parser.parse_args()

    train_ddp(
        mode=args.mode,
        iters=args.iters,
        lr=args.lr,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        n_train=args.n_train,
        n_test=args.n_test,
        ckpt_every=args.ckpt_every,
        log_dir=args.log_dir,
        resume=args.resume,
        path_to_resume=args.path_to_resume,
    )
