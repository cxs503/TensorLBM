"""SUBOFF reconstruction utility functions.

Checkpoint save/load, device detection, model construction, and loss functions
shared by training, fine-tuning, and inference modules.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import torch


# ── Device helpers ──────────────────────────────────────────────────────────

def default_suboff_device() -> str:
    """Return the best available device string for SUBOFF models.

    Priority: SDAA (LoongArch accelerator) > CUDA > CPU.
    """
    try:
        import torch_sdaa  # noqa: F401
        if torch.sdaa.is_available():
            return "sdaa:0"
    except ImportError:
        pass
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def _move_to_device(obj, device: torch.device):
    """Move tensor/module to device respecting exact device index."""
    return obj.to(device)


# ── Model construction ──────────────────────────────────────────────────────

def build_suboff_model(device: torch.device | str | None = None):
    """Build the SUBOFF Encoder-Decoder reconstruction model.

    Args:
        device: Target device. Defaults to :func:`default_suboff_device`.

    Returns:
        (encoder, decoder) tuple of nn.Module on the specified device.
    """
    from tensorlbm.ai.nn.encoder_module import IrregSTEncoder2D
    from tensorlbm.ai.nn.decoder_module import IrregSTDecoder2D

    if device is None:
        device = torch.device(default_suboff_device())
    else:
        device = torch.device(device)

    encoder = IrregSTEncoder2D(
        input_channels=4,
        time_window=1,
        in_emb_dim=144,
        out_channels=144,
        heads=1,
        depth=4,
        res=200,
        use_ln=True,
        emb_dropout=0.0,
    )
    decoder = IrregSTDecoder2D(
        latent_channels=144,
        out_channels=4,
        res=200,
        scale=2,
        dropout=0.1,
    )
    encoder = _move_to_device(encoder, device)
    decoder = _move_to_device(decoder, device)
    return encoder, decoder


# ── Loss ─────────────────────────────────────────────────────────────────────

def pointwise_rel_loss(x: torch.Tensor, y: torch.Tensor, p: int = 2) -> torch.Tensor:
    """Pointwise relative L-p loss between prediction and target.

    Args:
        x: Predicted tensor [b, n, c].
        y: Target tensor [b, n, c].
        p: L-p norm order (1 or 2).

    Returns:
        Scalar loss value.
    """
    assert x.shape == y.shape
    if p == 1:
        diff = (x - y).abs()
    else:
        diff = (x - y).pow(p)
    # Relative: diff / |y|^p (unnormalized, denominator = 1 in original code)
    diff = diff.sum(dim=-1)   # sum over channels
    diff = diff.mean(dim=1)   # mean over points
    diff = diff.mean()        # mean over batch
    return diff


# ── Checkpoint ───────────────────────────────────────────────────────────────

def save_checkpoint(state: dict, save_path: str, is_best: bool = False, max_keep: int | None = None):
    """Save a training checkpoint.

    Args:
        state: Dict with model/optimizer/scheduler state_dicts and n_iter.
        save_path: Destination file path.
        is_best: If True, also copy to best_model.ckpt.
        max_keep: Max number of checkpoints to keep (oldest deleted).
    """
    torch.save(state, save_path)

    save_dir = os.path.dirname(save_path)
    list_path = os.path.join(save_dir, "latest_checkpoint.txt")

    save_name = os.path.basename(save_path)
    if os.path.exists(list_path):
        with open(list_path) as f:
            ckpt_list = f.readlines()
        ckpt_list = [save_name + "\n"] + ckpt_list
    else:
        ckpt_list = [save_name + "\n"]

    if max_keep is not None:
        for ckpt in ckpt_list[max_keep:]:
            old = os.path.join(save_dir, ckpt[:-1])
            if os.path.exists(old):
                os.remove(old)
        ckpt_list[max_keep:] = []

    with open(list_path, "w") as f:
        f.writelines(ckpt_list)

    if is_best:
        shutil.copyfile(save_path, os.path.join(save_dir, "best_model.ckpt"))


def load_checkpoint(ckpt_path: str, map_location=None) -> dict:
    """Load a training checkpoint.

    Args:
        ckpt_path: Path to checkpoint file or directory.
        map_location: Device mapping for torch.load.

    Returns:
        Checkpoint dict with encoder/decoder/enc_optim/enc_sched/n_iter.
    """
    if os.path.isdir(ckpt_path):
        list_path = os.path.join(ckpt_path, "latest_checkpoint.txt")
        if os.path.exists(list_path):
            with open(list_path) as f:
                name = f.readline().strip()
            ckpt_path = os.path.join(ckpt_path, name)
        else:
            # Fallback: find latest .ckpt by modification time
            ckpts = sorted(Path(ckpt_path).glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
            if ckpts:
                ckpt_path = str(ckpts[0])
            else:
                raise FileNotFoundError(f"No checkpoint found in {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    print(f" [*] Loading checkpoint from {ckpt_path} succeed!")
    return ckpt


def ensure_dir(dir_name: str):
    """Create directory if it does not exist."""
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


# ── Coordinate helpers ───────────────────────────────────────────────────────

def get_suboff_coords(n_points: int | None = None, coord_name: str = "ori27",
                      data_dir: str | None = None) -> torch.Tensor:
    """Get SUBOFF position coordinates.

    Args:
        n_points: Max number of points to return (truncates if needed).
        coord_name: Coordinate function name (ori27, ori28, ori28_addition).
        data_dir: If provided, try loading coords.npy from this directory first.

    Returns:
        Float32 tensor of shape [n_points, 3].
    """
    import numpy as np

    # Try coords.npy first
    if data_dir:
        cp = os.path.join(data_dir, "coords.npy")
        if os.path.exists(cp):
            coords = np.load(cp).astype(np.float32)
            if n_points is not None:
                coords = coords[:n_points]
            return torch.as_tensor(coords)

    # Fallback to built-in coordinate functions
    from tensorlbm.ai.suboff_coord import coord_ori27, coord_ori28, coord_ori28_addition

    coord_funcs = {
        "ori27": coord_ori27,
        "ori28": coord_ori28,
        "ori28_addition": coord_ori28_addition,
    }
    func = coord_funcs.get(coord_name, coord_ori27)
    coords = func().astype(np.float32)
    if n_points is not None:
        coords = coords[:n_points]
    return torch.as_tensor(coords, dtype=torch.float32)
