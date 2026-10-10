"""SUBOFF 3D flow-field reconstruction inference module.

Encapsulates the prediction (test37) and error analysis workflows
as callable library functions.
"""

from __future__ import annotations

import csv
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from einops import repeat

from .suboff_utils import (
    build_suboff_model,
    default_suboff_device,
    get_suboff_coords,
    load_checkpoint,
    pointwise_rel_loss,
    _move_to_device,
)

_DEMO_CHANNEL_TO_INDEX = {"p": 0, "ux": 1, "uy": 2, "uz": 3, "u": 4}
_DEMO_VOLUME_SHAPE = (100, 50, 100)


# ── Configuration ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SuboffPredictConfig:
    """Configuration for SUBOFF inference (test37-style)."""

    checkpoint_path: str = ""  # Path to model checkpoint
    data_dir: str = ""  # NPY snapshot directory (suboff8/p,ux,uy,uz)
    snap_idx: int = 55  # Snapshot index (relative to test set start)
    test_set_offset: int = 1250  # Test set starts at this index
    batch_size: int = 1
    device: str = field(default_factory=default_suboff_device)


@dataclass(frozen=True)
class SuboffErrorConfig:
    """Configuration for SUBOFF error analysis."""

    checkpoint_path: str = ""
    data_dir: str = ""
    n_points: int = 5000
    max_snaps: int = 10  # Max snapshots to evaluate (avoid loading all 1500)
    device: str = field(default_factory=default_suboff_device)


# ── NPY file reading ─────────────────────────────────────────────────────────


def _read_test_file1(path: str) -> np.ndarray:
    """Read NPY file, crop to [49:149, :, 49:149], flatten per XY slice (100 slices of 5000)."""
    file = np.load(path).astype(np.float32)
    file = file[49:149, :, 49:149]
    result = np.empty(500_000, dtype=np.float32)
    for i in range(100):
        result[i * 5000 : (i + 1) * 5000] = file[:, :, i].flatten()
    return result


def _read_test_file2(path: str) -> np.ndarray:
    """Read NPY file, crop to [49:149, :, 49:149], flatten all."""
    file = np.load(path).astype(np.float32)
    file = file[49:149, :, 49:149]
    return file.flatten()


def _read_test_file3(path: str) -> np.ndarray:
    """Read NPY file, crop to [49:149, :, 49:149], flatten per XZ slice (50 slices of 10000)."""
    file = np.load(path).astype(np.float32)
    file = file[49:149, :, 49:149]
    result = np.empty(500_000, dtype=np.float32)
    for i in range(50):
        result[i * 10000 : (i + 1) * 10000] = file[:, i, :].flatten()
    return result


# ── Prediction ───────────────────────────────────────────────────────────────


def predict_suboff(cfg: SuboffPredictConfig | None = None) -> dict[str, Any]:
    """Run SUBOFF flow-field reconstruction prediction (test37-style).

    Loads a checkpoint, reads a test snapshot, runs encoder-decoder inference,
    and returns predicted/true/error fields as numpy arrays.

    Args:
        cfg: Prediction configuration. Uses defaults if None.

    Returns:
        Dict containing:
        - coords: [500000, 3] numpy array (X,Y,Z)
        - real: [500000, 5] numpy array (p,ux,uy,uz,u) -- ground truth
        - pred: [500000, 5] numpy array (p,ux,uy,uz,u) -- prediction
        - error: [500000, 5] numpy array (pred - real)
        - input: [20000, 5] numpy array -- sparse input points
        - mape: float -- mean absolute percentage error on velocity magnitude
        - rel_l2: float -- average relative L2 loss
        - mse: float -- average MSE loss
    """
    if cfg is None:
        cfg = SuboffPredictConfig()

    device = torch.device(cfg.device)

    # Build model and load checkpoint
    encoder, decoder = build_suboff_model(device)
    ckpt = load_checkpoint(cfg.checkpoint_path, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    encoder.eval()
    decoder.eval()

    # Load test snapshot
    root = cfg.data_dir
    path1 = os.path.join(root, "p")
    path2 = os.path.join(root, "ux")
    path3 = os.path.join(root, "uy")
    path4 = os.path.join(root, "uz")
    filename = f"/{cfg.snap_idx + cfg.test_set_offset}.npy"

    # 3 representations of the same snapshot
    press1 = _read_test_file1(path1 + filename)
    ux1 = _read_test_file1(path2 + filename)
    uy1 = _read_test_file1(path3 + filename)
    uz1 = _read_test_file1(path4 + filename)
    test_data1 = torch.cat(
        [
            torch.as_tensor(press1, dtype=torch.float32).unsqueeze(-1),
            torch.as_tensor(ux1, dtype=torch.float32).unsqueeze(-1),
            torch.as_tensor(uy1, dtype=torch.float32).unsqueeze(-1),
            torch.as_tensor(uz1, dtype=torch.float32).unsqueeze(-1),
        ],
        dim=-1,
    ).unsqueeze(0)  # [1, 500000, 4]

    press2 = _read_test_file2(path1 + filename)
    ux2 = _read_test_file2(path2 + filename)
    uy2 = _read_test_file2(path3 + filename)
    uz2 = _read_test_file2(path4 + filename)
    u = np.sqrt(ux2**2 + uy2**2 + uz2**2).reshape(-1, 1)
    result1 = np.concatenate(
        [
            press2.reshape(-1, 1),
            ux2.reshape(-1, 1),
            uy2.reshape(-1, 1),
            uz2.reshape(-1, 1),
            u,
        ],
        axis=1,
    )  # [500000, 5]

    test_data2 = torch.cat(
        [
            torch.as_tensor(press2, dtype=torch.float32).unsqueeze(-1),
            torch.as_tensor(ux2, dtype=torch.float32).unsqueeze(-1),
            torch.as_tensor(uy2, dtype=torch.float32).unsqueeze(-1),
            torch.as_tensor(uz2, dtype=torch.float32).unsqueeze(-1),
        ],
        dim=-1,
    ).unsqueeze(0)  # [1, 500000, 4]

    press3 = _read_test_file3(path1 + filename)
    ux3 = _read_test_file3(path2 + filename)
    uy3 = _read_test_file3(path3 + filename)
    uz3 = _read_test_file3(path4 + filename)
    test_data3 = torch.cat(
        [
            torch.as_tensor(press3, dtype=torch.float32).unsqueeze(-1),
            torch.as_tensor(ux3, dtype=torch.float32).unsqueeze(-1),
            torch.as_tensor(uy3, dtype=torch.float32).unsqueeze(-1),
            torch.as_tensor(uz3, dtype=torch.float32).unsqueeze(-1),
        ],
        dim=-1,
    ).unsqueeze(0)  # [1, 500000, 4]

    test_data = torch.cat(
        [
            test_data1.unsqueeze(0),
            test_data2.unsqueeze(0),
            test_data3.unsqueeze(0),
        ],
        dim=0,
    )  # [3, 1, 500000, 4]

    # Prepare dataset and loader
    from .suboff_dataset import CylinderDatasetMultiRe14

    tw = 1
    test_dataset = CylinderDatasetMultiRe14(test_data, tw, push_forward=0)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=True)

    # Position coordinates
    from .suboff_coord import coord_ori27, coord_ori28, coord_ori28_addition

    pos_all1 = repeat(
        torch.as_tensor(coord_ori27(), dtype=torch.float32), "n d -> b n d", b=cfg.batch_size
    )
    pos_all2 = repeat(
        torch.as_tensor(coord_ori28(), dtype=torch.float32), "n d -> b n d", b=cfg.batch_size
    )
    pos_all3 = repeat(
        torch.as_tensor(coord_ori28_addition(), dtype=torch.float32),
        "n d -> b n d",
        b=cfg.batch_size,
    )

    pos_all1 = _move_to_device(pos_all1, device)
    pos_all2 = _move_to_device(pos_all2, device)
    pos_all3 = _move_to_device(pos_all3, device)

    prop_pos = pos_all2

    # Run inference
    all_avg_loss = []
    torch_avg_loss = []
    pred_result = None
    input_data = None
    pred = None

    for data in tqdm(test_loader, desc="Predicting"):
        x1, x2, x3, y = data
        num1 = random.randint(0, 99)
        num2 = random.randint(0, 99)
        num3 = random.randint(0, 49)
        x1 = x1[:, :, num1 * 5000 : (num1 + 1) * 5000, :]
        x2 = x2[:, :, num2 * 5000 : (num2 + 1) * 5000, :]
        x3 = x3[:, :, num3 * 10000 : (num3 + 1) * 10000, :]
        x = torch.cat((x1, x2, x3), dim=2)

        pos1 = pos_all1[:, num1 * 5000 : (num1 + 1) * 5000, :]
        pos2 = pos_all2[:, num2 * 5000 : (num2 + 1) * 5000, :]
        pos3 = pos_all3[:, num3 * 10000 : (num3 + 1) * 10000, :]
        input_pos = torch.cat((pos1, pos2, pos3), dim=1)

        # Save input data for visualization
        x_visu = x.cpu().numpy()
        input_pos_visu = input_pos.cpu().numpy()
        p_in = x_visu[0, 0, :, 0]
        ux_in = x_visu[0, 0, :, 1]
        uy_in = x_visu[0, 0, :, 2]
        uz_in = x_visu[0, 0, :, 3]
        u_in = np.sqrt(ux_in**2 + uy_in**2 + uz_in**2).reshape(-1, 1)
        input_data = np.concatenate(
            [
                input_pos_visu[0, :, :],
                p_in.reshape(-1, 1),
                ux_in.reshape(-1, 1),
                uy_in.reshape(-1, 1),
                uz_in.reshape(-1, 1),
                u_in,
            ],
            axis=1,
        )  # [20000, 8]

        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            z = encoder.forward(x, input_pos)
            pred = decoder.forward(z, prop_pos, input_pos)  # [1, 500000, 4]

            y_temp = y[:, :, 1:]
            pred_temp = pred[:, :, 1:]
            loss = pointwise_rel_loss(pred_temp, y_temp, p=2)
            mse_loss = F.mse_loss(pred_temp, y_temp)
            all_avg_loss.append(float(loss.item()))
            torch_avg_loss.append(float(mse_loss.item()))

    if pred is None:
        raise RuntimeError("No prediction produced — test_loader may be empty")

    # Convert prediction to numpy
    pred_np = pred.cpu().numpy()  # [1, 500000, 4]
    p_pred = pred_np[0, :, 0]
    ux_pred = pred_np[0, :, 1]
    uy_pred = pred_np[0, :, 2]
    uz_pred = pred_np[0, :, 3]
    u_pred = np.sqrt(ux_pred**2 + uy_pred**2 + uz_pred**2).reshape(-1, 1)
    result2 = np.concatenate(
        [
            p_pred.reshape(-1, 1),
            ux_pred.reshape(-1, 1),
            uy_pred.reshape(-1, 1),
            uz_pred.reshape(-1, 1),
            u_pred,
        ],
        axis=1,
    )  # [500000, 5]

    # Error
    result_error = result2 - result1

    # MAPE on velocity magnitude
    true_values = result1[:, 4]
    predicted_values = result2[:, 4]
    ape = np.abs((true_values - predicted_values) / (true_values + 1e-10)) * 100
    mape = float(np.mean(ape))

    # Coordinates
    from .suboff_coord import coord_ori28

    coords = coord_ori28()

    return {
        "coords": coords,  # [500000, 3]
        "real": result1,  # [500000, 5]
        "pred": result2,  # [500000, 5]
        "error": result_error,  # [500000, 5]
        "input": input_data,  # [20000, 8] or None
        "mape": mape,
        "rel_l2_avg": float(np.mean(all_avg_loss)) * 1e4,
        "mse_avg": float(np.mean(torch_avg_loss)) * 1e4,
        "checkpoint": cfg.checkpoint_path,
        "snap_idx": cfg.snap_idx,
    }


def render_suboff_flowfield_demo(
    inference_result: dict[str, Any],
    output_dir: str | Path,
    *,
    slice_axis: str = "z",
    slice_index: int = 50,
    channels: tuple[str, ...] = ("u", "p"),
    file_prefix: str = "suboff_v03",
) -> dict[str, Any]:
    """Render standard SUBOFF demo flow-field figures from prediction results."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    pred = np.asarray(inference_result["pred"], dtype=np.float32)
    real = np.asarray(inference_result["real"], dtype=np.float32)
    error = np.asarray(inference_result["error"], dtype=np.float32)
    expected_points = int(np.prod(_DEMO_VOLUME_SHAPE))
    if pred.shape != (expected_points, 5):
        raise ValueError(f"pred shape must be ({expected_points}, 5), got {pred.shape}")
    if real.shape != (expected_points, 5):
        raise ValueError(f"real shape must be ({expected_points}, 5), got {real.shape}")
    if error.shape != (expected_points, 5):
        raise ValueError(f"error shape must be ({expected_points}, 5), got {error.shape}")

    axis_to_dim = {"z": 2, "y": 1, "x": 0}
    axis = slice_axis.lower()
    if axis not in axis_to_dim:
        raise ValueError("slice_axis must be one of: x, y, z")
    dim = axis_to_dim[axis]
    max_idx = _DEMO_VOLUME_SHAPE[dim] - 1
    slice_idx = max(0, min(int(slice_index), max_idx))

    requested_channels = [c for c in channels if c in _DEMO_CHANNEL_TO_INDEX]
    if not requested_channels:
        requested_channels = ["u", "p"]

    def _slice_2d(flat_field: np.ndarray, channel: str) -> np.ndarray:
        vol = flat_field[:, _DEMO_CHANNEL_TO_INDEX[channel]].reshape(_DEMO_VOLUME_SHAPE)
        return np.take(vol, slice_idx, axis=dim)

    generated_files: list[str] = []
    for channel in requested_channels:
        real_2d = _slice_2d(real, channel)
        pred_2d = _slice_2d(pred, channel)
        abs_err_2d = np.abs(_slice_2d(error, channel))

        for kind, arr, cmap in (
            ("real", real_2d, "viridis"),
            ("pred", pred_2d, "viridis"),
            ("abs_error", abs_err_2d, "magma"),
        ):
            fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=120)
            im = ax.imshow(arr, origin="lower", cmap=cmap, aspect="auto")
            ax.set_title(f"{channel} {kind} slice {axis}={slice_idx}")
            ax.set_xlabel("i")
            ax.set_ylabel("j")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fname = f"{file_prefix}_{channel}_{kind}_slice-{axis}{slice_idx:03d}.png"
            out_path = output_root / fname
            fig.tight_layout()
            fig.savefig(out_path)
            plt.close(fig)
            generated_files.append(str(out_path.resolve()))

    return {
        "slice_axis": axis,
        "slice_index": slice_idx,
        "channels": requested_channels,
        "files": generated_files,
    }


# ── Error analysis ───────────────────────────────────────────────────────────


def error_analysis_suboff(cfg: SuboffErrorConfig | None = None) -> dict[str, Any]:
    """Run per-snapshot per-channel error analysis.

    Loads a checkpoint, iterates over test snapshots, and computes
    per-channel MAE, RMSE, and relative L2 errors.

    Args:
        cfg: Error analysis configuration.

    Returns:
        Dict containing per-snapshot errors and summary statistics.
    """
    if cfg is None:
        cfg = SuboffErrorConfig()

    device = torch.device(cfg.device)

    # Build model and load checkpoint
    encoder, decoder = build_suboff_model(device)
    ckpt_path = cfg.checkpoint_path
    if not ckpt_path:
        # Find latest checkpoint
        ckpt_dir = Path(__file__).resolve().parent.parent.parent.parent / "checkpoints" / "suboff"
        ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if ckpts:
            ckpt_path = str(ckpts[0])
    if ckpt_path:
        ckpt = load_checkpoint(ckpt_path, map_location=device)
        encoder.load_state_dict(ckpt.get("encoder", {}), strict=False)
        decoder.load_state_dict(ckpt.get("decoder", {}), strict=False)
    encoder.eval()
    decoder.eval()

    # Load coordinates
    coords = get_suboff_coords(cfg.n_points, data_dir=cfg.data_dir)
    n_points = min(cfg.n_points, coords.shape[0])
    pos = coords[:n_points].to(device).unsqueeze(0)

    # Load test data (limited by max_snaps)
    test_data = _load_npy_data(cfg.data_dir, max_snaps=cfg.max_snaps)
    if test_data is None:
        raise ValueError(f"No data found at {cfg.data_dir}")

    ch_names = ["pressure", "vx", "vy", "vz"]
    all_errors: list[dict] = []
    t0 = time.perf_counter()

    for snap_idx in range(test_data.shape[0]):
        snap = test_data[snap_idx].to(device)
        idxs = torch.arange(n_points, device=device)
        x = snap[idxs].unsqueeze(0).unsqueeze(0)  # [1, 1, N, 4]
        with torch.no_grad():
            z = encoder(x, pos)
            pred = decoder(z, pos, pos)
        true = x.cpu().numpy()[0, 0]  # [N, 4]
        pred_np = pred.cpu().numpy()[0]

        ch_errs = {}
        for ci, cn in enumerate(ch_names):
            t_ch = true[:, ci]
            p_ch = pred_np[:, ci]
            mae = float(np.abs(t_ch - p_ch).mean())
            rmse = float(np.sqrt(((t_ch - p_ch) ** 2).mean()))
            rel_l2 = float(np.linalg.norm(t_ch - p_ch) / (np.linalg.norm(t_ch) + 1e-10))
            ch_errs[cn] = {"mae": round(mae, 6), "rmse": round(rmse, 6), "rel_l2": round(rel_l2, 6)}

        all_errors.append({"snapshot": snap_idx, "channels": ch_errs})

    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Summary
    summary = {}
    for cn in ch_names:
        vals = [e["channels"][cn]["rel_l2"] for e in all_errors]
        summary[cn] = {
            "rel_l2_mean": round(float(np.mean(vals)), 6),
            "rel_l2_max": round(float(np.max(vals)), 6),
        }

    return {
        "status": "ok",
        "n_snapshots": test_data.shape[0],
        "n_points": n_points,
        "time_ms": round(elapsed_ms, 1),
        "checkpoint": ckpt_path,
        "summary": summary,
        "per_snapshot": all_errors,
    }


def _load_npy_data(data_dir: str, max_snaps: int | None = None) -> torch.Tensor | None:
    """Load multi-channel NPY snapshot data from {data_dir}/p,ux,uy,uz.

    Args:
        data_dir: Root directory containing p/ux/uy/uz subdirs.
        max_snaps: Max number of snapshots to load (None = all).

    Returns:
        Tensor of shape [n_snaps, total_points, 4] or None.
    """
    import glob

    channels = ("p", "ux", "uy", "uz")
    result: list[torch.Tensor] | None = None

    for ci, ch in enumerate(channels):
        cd = os.path.join(data_dir, ch)
        if not os.path.isdir(cd):
            return None
        files = sorted(
            [f for f in os.listdir(cd) if f.endswith(".npy")],
            key=lambda x: int(x.rsplit(".", 1)[0]),
        )
        if max_snaps is not None:
            files = files[:max_snaps]
        if not files:
            return None
        stacked: list[np.ndarray] = []
        for fn in files:
            arr = np.load(os.path.join(cd, fn)).astype(np.float32)
            stacked.append(arr.flatten())
        t = torch.as_tensor(np.stack(stacked), dtype=torch.float32).unsqueeze(-1)
        if result is None:
            result = [t]
        else:
            result.append(t)

    if result is None or len(result) != 4:
        return None
    return torch.cat(result, dim=-1)
