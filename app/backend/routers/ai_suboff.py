"""
SUBOFF API — 3D Encoder-Decoder 训练/微调/推理/可视化端点

所有核心逻辑已封装为 tensorlbm.ai 库函数，路由仅负责：
  - HTTP 请求解析 → 构造 Config 对象
  - 调用库函数（异步线程包装长时间训练任务）
  - 返回 JSON 结果
"""
from __future__ import annotations

import gc
import math
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from tensorlbm.ai.suboff_utils import (
    build_suboff_model,
    default_suboff_device,
    get_suboff_coords,
    load_checkpoint,
    pointwise_rel_loss,
)
from tensorlbm.ai.suboff_train import (
    SuboffFinetuneConfig,
    SuboffTrainConfig,
    finetune_suboff,
    train_suboff,
)
from tensorlbm.ai.suboff_inference import (
    SuboffErrorConfig,
    SuboffPredictConfig,
    error_analysis_suboff,
    predict_suboff,
)

router = APIRouter()

CKPT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "checkpoints" / "suboff"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

def _best_checkpoint() -> Path | None:
    """Select the checkpoint with the highest iteration count."""
    ckpts = list(CKPT_DIR.glob("model_checkpoint*.ckpt"))
    if not ckpts:
        return None
    # Sort by iteration number extracted from filename (e.g. model_checkpoint31249.ckpt → 31249)
    def _iter_num(p: Path) -> int:
        m = re.search(r"checkpoint(\d+)", p.name)
        return int(m.group(1)) if m else -1
    ckpts.sort(key=_iter_num, reverse=True)
    return ckpts[0]

_training_jobs: dict[str, dict] = {}


# ── Request models ───────────────────────────────────────────────────────────

class SuboffTrainRequest(BaseModel):
    iters: int = Field(125_000, ge=1, le=500_000, description="训练迭代次数")
    batch_size: int = Field(4, ge=1, le=64)
    lr: float = Field(6e-4, gt=0, le=1e-1, description="学习率")
    n_points: int = Field(2000, ge=100, le=50000, description="每样本采样点数")
    data_dir: str | None = Field(None, description="NPY 数据目录 ({dir}/p/{idx}.npy, ...)")
    n_train: int = Field(1250, ge=1, le=1500, description="训练快照数")
    n_test: int = Field(250, ge=1, le=500, description="测试快照数")
    device: str = Field(default_suboff_device())


class SuboffFinetuneRequest(BaseModel):
    iters: int = Field(31_250, ge=1, le=500_000, description="微调迭代次数")
    lr: float = Field(1e-5, gt=0, le=1e-3, description="微调学习率（应低于预训练）")
    n_points: int = Field(2000, ge=100, le=50000)
    data_dir: str = Field(..., description="微调数据目录（不同 Re 工况）")
    checkpoint: str | None = Field(None, description="预训练 checkpoint 路径，默认用最新")
    device: str = Field(default_suboff_device())


class SuboffPredictRequest(BaseModel):
    n_points: int = Field(2000, ge=100, le=20000)
    device: str = Field(default_suboff_device(), description="推理设备 (cpu/sdaa:0/cuda:0)")


class SuboffErrorRequest(BaseModel):
    data_dir: str = Field(..., description="测试数据目录（NPY 快照）")
    n_points: int = Field(5000, ge=100, le=50000, description="评估采样点数")
    max_snaps: int = Field(10, ge=1, le=1500, description="最大评估快照数")
    device: str = Field(default_suboff_device())


# ── Training ─────────────────────────────────────────────────────────────────

@router.post("/train")
def train_suboff_api(req: SuboffTrainRequest):
    """Start a SUBOFF pretraining job (runs in background thread)."""
    import threading

    job_id = f"suboff_train_{int(time.time())}"
    _training_jobs[job_id] = {"status": "preparing", "epoch": 0, "total": req.iters, "loss": None, "phase": "preparing", "lr": None, "mse": None}

    def worker():
        try:
            cfg = SuboffTrainConfig(
                lr=req.lr,
                iters=req.iters,
                batch_size=req.batch_size,
                data_dir=req.data_dir or "",
                n_train=req.n_train,
                n_test=req.n_test,
                device=req.device,
            )
            def on_progress(info):
                # Sanitize non-JSON-compliant floats (inf, nan)
                bl = info.get("best_loss", _training_jobs[job_id].get("best_loss", 0))
                if bl is not None and (not isinstance(bl, (int, float)) or math.isinf(bl) or math.isnan(bl)):
                    bl = None
                loss = info["loss"]
                if loss is not None and (math.isinf(loss) or math.isnan(loss)):
                    loss = None
                mse = info.get("mse")
                if mse is not None and (math.isinf(mse) or math.isnan(mse)):
                    mse = None
                _training_jobs[job_id].update({
                    "status": "running",
                    "epoch": info["epoch"],
                    "total": info["total"],
                    "loss": loss,
                    "phase": info["phase"],
                    "lr": info["lr"],
                    "mse": mse,
                    "best_loss": bl,
                })
            result = train_suboff(cfg, progress_callback=on_progress)
            _training_jobs[job_id] = {
                "status": "completed",
                "best_loss": result.get("best_loss_1e4", 0),
                "final_iter": result.get("final_iter", 0),
                "checkpoint_dir": result.get("checkpoint_dir", ""),
                "phase": "completed",
                "epoch": result.get("final_iter", 0),
                "total": req.iters,
                "loss": result.get("best_loss_1e4", 0),
                "lr": None,
                "mse": None,
            }
        except Exception as e:
            _training_jobs[job_id] = {"status": "failed", "error": str(e), "phase": "failed"}
            import traceback; traceback.print_exc()
        finally:
            gc.collect()
            torch.sdaa.empty_cache()

    threading.Thread(target=worker, daemon=True).start()
    return {"job_id": job_id, "status": "started", "iters": req.iters}


@router.post("/finetune")
def finetune_suboff_api(req: SuboffFinetuneRequest):
    """Start a SUBOFF fine-tuning job (runs in background thread)."""
    import threading

    job_id = f"suboff_ft_{int(time.time())}"
    _training_jobs[job_id] = {"status": "preparing", "epoch": 0, "total": req.iters, "loss": None, "phase": "preparing", "lr": None, "mse": None}

    def worker():
        try:
            ckpt_path = req.checkpoint
            if ckpt_path is None:
                best = _best_checkpoint()
                ckpt_path = str(best) if best else ""

            cfg = SuboffFinetuneConfig(
                lr=req.lr,
                iters=req.iters,
                batch_size=1,
                data_dir=req.data_dir,
                path_to_resume=ckpt_path,
                device=req.device,
            )
            def on_progress(info):
                # Sanitize non-JSON-compliant floats (inf, nan)
                bl = info.get("best_loss", _training_jobs[job_id].get("best_loss", 0))
                if bl is not None and (not isinstance(bl, (int, float)) or math.isinf(bl) or math.isnan(bl)):
                    bl = None
                loss = info["loss"]
                if loss is not None and (math.isinf(loss) or math.isnan(loss)):
                    loss = None
                mse = info.get("mse")
                if mse is not None and (math.isinf(mse) or math.isnan(mse)):
                    mse = None
                _training_jobs[job_id].update({
                    "status": "running",
                    "epoch": info["epoch"],
                    "total": info["total"],
                    "loss": loss,
                    "phase": info["phase"],
                    "lr": info["lr"],
                    "mse": mse,
                    "best_loss": bl,
                })
            result = finetune_suboff(cfg, progress_callback=on_progress)
            _training_jobs[job_id] = {
                "status": "completed",
                "best_loss": result.get("best_loss_1e4", 0),
                "final_iter": result.get("final_iter", 0),
                "checkpoint_dir": result.get("checkpoint_dir", ""),
                "phase": "completed",
                "epoch": result.get("final_iter", 0),
                "total": req.iters,
                "loss": result.get("best_loss_1e4", 0),
                "lr": None,
                "mse": None,
            }
        except Exception as e:
            _training_jobs[job_id] = {"status": "failed", "error": str(e), "phase": "failed"}
            import traceback; traceback.print_exc()
        finally:
            gc.collect()
            torch.sdaa.empty_cache()

    threading.Thread(target=worker, daemon=True).start()
    return {"job_id": job_id, "status": "started", "iters": req.iters}


@router.get("/train/{job_id}")
def train_status(job_id: str):
    if job_id not in _training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    j = _training_jobs[job_id]
    return {
        "job_id": job_id, "status": j.get("status", "?"),
        "epoch": j.get("epoch", 0), "total": j.get("total", 0),
        "loss": j.get("loss"), "best_loss": j.get("best_loss", 0),
        "phase": j.get("phase", "preparing"),
        "lr": j.get("lr"), "mse": j.get("mse"),
        "error": j.get("error"),
    }


@router.get("/train")
def list_training():
    return {"jobs": list(_training_jobs.keys())[-10:]}


# ── Inference ────────────────────────────────────────────────────────────────

_model_cache: tuple | None = None


def _get_model():
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    device = torch.device(default_suboff_device())
    enc, dec = build_suboff_model(device)
    best = _best_checkpoint()
    if best:
        ckpt = load_checkpoint(str(best), map_location=device)
        enc.load_state_dict(ckpt.get("encoder", {}), strict=False)
        dec.load_state_dict(ckpt.get("decoder", {}), strict=False)
    enc.eval(); dec.eval()
    _model_cache = (enc, dec, device, None)
    return _model_cache


@router.get("/status")
def suboff_status():
    ckpts = list(CKPT_DIR.glob("*.ckpt"))
    best = _best_checkpoint()
    return {"checkpoints": len(ckpts), "latest": best.name if best else None, "model_loaded": _model_cache is not None}


@router.post("/predict")
def suboff_predict_api(req: SuboffPredictRequest):
    """Run SUBOFF prediction using library function."""
    try:
        # Find best checkpoint (highest iteration)
        best = _best_checkpoint()
        if not best:
            raise HTTPException(status_code=400, detail="No checkpoint found")

        # Find data directory
        data_dirs = [
            "/root/LBM-Platform/suboff_all/suboff8",
            "/tmp/suboff_600x150",
            "/tmp/suboff_demo",
        ]
        data_dir = None
        for d in data_dirs:
            if os.path.isdir(os.path.join(d, "p")):
                data_dir = d
                break
        if data_dir is None:
            raise HTTPException(status_code=400, detail="No snapshot data found")

        cfg = SuboffPredictConfig(
            checkpoint_path=str(best),
            data_dir=data_dir,
            device=req.device,
        )
        result = predict_suboff(cfg)

        # Extract stats for API response
        pred = result["pred"]   # [500000, 5]
        real = result["real"]   # [500000, 5]
        return {
            "status": "ok",
            "mape": round(result["mape"], 2),
            "rel_l2_avg_1e4": round(result["rel_l2_avg"], 3),
            "mse_avg_1e4": round(result["mse_avg"], 3),
            "stats": {
                "vx": {"min": float(pred[:, 1].min()), "max": float(pred[:, 1].max()), "mean": float(pred[:, 1].mean())},
                "vy": {"min": float(pred[:, 2].min()), "max": float(pred[:, 2].max()), "mean": float(pred[:, 2].mean())},
            },
            "recon_error": {
                "vx_rel_l2": float(np.linalg.norm(pred[:, 1] - real[:, 1]) / (np.linalg.norm(real[:, 1]) + 1e-10)),
                "vy_rel_l2": float(np.linalg.norm(pred[:, 2] - real[:, 2]) / (np.linalg.norm(real[:, 2]) + 1e-10)),
            },
            "checkpoint": str(best),
            "device": cfg.device,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        gc.collect()
        torch.sdaa.empty_cache()


# ── Error analysis ───────────────────────────────────────────────────────────

@router.post("/error")
def suboff_error_api(req: SuboffErrorRequest):
    """Run error analysis using library function."""
    try:
        best = _best_checkpoint()
        ckpt_path = str(best) if best else ""

        cfg = SuboffErrorConfig(
            checkpoint_path=ckpt_path,
            data_dir=req.data_dir,
            n_points=req.n_points,
            max_snaps=req.max_snaps,
            device=req.device,
        )
        result = error_analysis_suboff(cfg)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Data scan ────────────────────────────────────────────────────────────────

@router.get("/data")
def list_data(data_dir: str | None = None):
    """List available NPY snapshot directories."""
    if data_dir is None:
        data_dir = str(CKPT_DIR.parent.parent / "suboff_snapshots")
    p = Path(data_dir)
    if not p.exists():
        return {"data_dir": data_dir, "exists": False}

    def _count_snaps(d: Path) -> dict | None:
        result = {}
        for ch in ("p", "ux", "uy", "uz"):
            chd = d / ch
            if chd.is_dir():
                files = sorted([f.name for f in chd.glob("*.npy")], key=lambda x: int(x.rsplit(".", 1)[0]))
                result[ch] = len(files)
        return result if result else None

    re_dirs = sorted([d for d in p.iterdir() if d.is_dir() and d.name.startswith("Re_")])
    if re_dirs:
        total = 0
        per_re = {}
        for rd in re_dirs:
            cnt = _count_snaps(rd)
            if cnt:
                n = min(cnt.values())
                per_re[rd.name] = n
                total += n
        return {"data_dir": data_dir, "multi_re": True, "re_groups": len(re_dirs),
                "per_re": per_re, "total_snapshots": total}

    cnt = _count_snaps(p)
    return {"data_dir": data_dir, "channels": cnt, "total_snapshots": min(cnt.values()) if cnt else 0}


# ── Animation (time-series slice data, no model needed) ──────────────────────

@router.get("/animate")
def suboff_animate(
    data_dir: str = "/root/LBM-Platform/suboff_all/suboff8",
    snap_start: int = 0,
    snap_end: int = 20,
    slice_axis: str = "z",
    slice_idx: int | None = None,
    channels: str = "p,ux,uy,uz",
    view: str = "full",
):
    """Return slice data for multiple snapshots for frontend animation playback.

    This endpoint only reads raw NPY data and extracts 2D slices — no model
    inference is needed, so it's fast and suitable for time-series animation.

    Two view modes:
      - 'full':  Show the complete tail flow-field data (200, 50, 200) as saved
                 by the LBM simulation. This is the original computational result,
                 NOT the training-cropped region.
      - 'train': Show only the near-suboff training crop region [49:149,:,49:149]
                 → (100, 50, 100). This is what the AI model was trained on.

    Args:
        data_dir: NPY snapshot root directory (with p/ux/uy/uz subdirs).
        snap_start: First snapshot index (0-based, maps to file {snap_start+1250}.npy).
        snap_end: Last snapshot index (exclusive). Max range = 50 frames.
        slice_axis: Slice axis — 'z' (XY plane), 'y' (XZ plane), 'x' (YZ plane).
        slice_idx: Slice position along the axis. Default = center.
        channels: Comma-separated channel names to include (e.g. "p,ux").
        view: 'full' = original LBM tail field (no crop), 'train' = training crop only.

    Returns:
        Dict with frames array, each frame containing 2D slice arrays per channel.
    """
    try:
        ch_list = [c.strip() for c in channels.split(",") if c.strip() in ("p", "ux", "uy", "uz")]
        if not ch_list:
            ch_list = ["p", "ux", "uy", "uz"]

        crop = view == "train"

        # Cap frame range to avoid huge responses
        n_frames = min(snap_end - snap_start, 50)
        if n_frames <= 0:
            n_frames = 20
        snap_end = snap_start + n_frames

        # Determine grid size from first available snapshot
        test_offset = 1250  # test set starts at index 1250
        first_file = os.path.join(data_dir, "p", f"{snap_start + test_offset}.npy")
        if not os.path.isfile(first_file):
            # Try without offset (raw indices)
            first_file = os.path.join(data_dir, "p", f"{snap_start}.npy")
            test_offset = 0
        if not os.path.isfile(first_file):
            raise HTTPException(status_code=400, detail=f"No snapshot data found at {data_dir}")

        sample = np.load(first_file).astype(np.float32)
        full_shape = list(sample.shape)  # e.g. [200, 50, 200]
        if crop:
            sample = sample[49:149, :, 49:149]
        cropped_shape = list(sample.shape)  # e.g. [100, 50, 100]
        C = sample.shape[0]

        if slice_idx is None:
            slice_idx = C // 2
        slice_idx = max(0, min(slice_idx, C - 1))

        axis_map = {"z": (0, "XY"), "y": (1, "XZ"), "x": (2, "YZ")}
        ax_dim, plane_name = axis_map.get(slice_axis, axis_map["z"])

        frames = []
        for snap_idx in range(snap_start, snap_end):
            frame_data = {}
            for ch in ch_list:
                fpath = os.path.join(data_dir, ch, f"{snap_idx + test_offset}.npy")
                if not os.path.isfile(fpath):
                    continue
                arr = np.load(fpath).astype(np.float32)
                if crop:
                    arr = arr[49:149, :, 49:149]
                slice_2d = np.take(arr, slice_idx, axis=ax_dim)
                # Downsample for faster transfer: max 200x200 for full, 100x100 for train
                max_dim = 200 if not crop else 100
                ny, nx = slice_2d.shape
                if ny > max_dim or nx > max_dim:
                    step_y = max(1, ny // max_dim)
                    step_x = max(1, nx // max_dim)
                    slice_2d = slice_2d[::step_y, ::step_x]
                frame_data[ch] = {
                    "data": slice_2d.tolist(),
                    "shape": list(slice_2d.shape),
                }
            frames.append({"snap_idx": snap_idx, "channels": frame_data})

        return {
            "status": "ok",
            "snap_start": snap_start,
            "snap_end": snap_end,
            "n_frames": len(frames),
            "slice_axis": slice_axis,
            "slice_idx": slice_idx,
            "plane_name": plane_name,
            "channels": ch_list,
            "view": view,
            "full_shape": full_shape,
            "cropped_shape": cropped_shape,
            "frames": frames,
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── Visualization ────────────────────────────────────────────────────────────

@router.get("/viz-data")
def suboff_viz_data(
    data_dir: str = "/root/LBM-Platform/suboff_all/suboff8",
    snap_idx: int = 0,
    n_points: int = 50000,
    slice_axis: str = "z",
    slice_idx: int | None = None,
    device: str = "cpu",
):
    """Return raw slice data (true + predicted + error) for frontend Plotly rendering."""
    try:
        # Use cached model instead of rebuilding every request
        enc, dec, dev, _ = _get_model()
        # Override device if user requests different one
        if device != str(dev):
            dev = torch.device(device)
            enc, dec = build_suboff_model(dev)
            best = _best_checkpoint()
            if best:
                ckpt = load_checkpoint(str(best), map_location=dev)
                enc.load_state_dict(ckpt.get("encoder", {}), strict=False)
                dec.load_state_dict(ckpt.get("decoder", {}), strict=False)
            enc.eval(); dec.eval()

        channels = ("p", "ux", "uy", "uz")
        snap_data = {}
        for ch in channels:
            arr = np.load(os.path.join(data_dir, ch, f"{snap_idx + 1250}.npy")).astype(np.float32)
            arr = arr[49:149, :, 49:149]
            snap_data[ch] = arr

        C = snap_data["p"].shape[0]  # 100
        true_3d = np.stack([snap_data[ch] for ch in channels], axis=-1)  # [100, 50, 100, 4]

        # Build sparse input for model
        coords = get_suboff_coords(n_points, data_dir=data_dir).to(dev)
        pos = coords.unsqueeze(0)
        flat = np.stack([snap_data[ch].flatten() for ch in channels], axis=-1)
        x = torch.as_tensor(flat[:n_points], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(dev)

        with torch.no_grad():
            z = enc(x, pos)
            pred_out = dec(z, pos, pos)

        pred_3d = pred_out.cpu().numpy()[0][:500_000, :].reshape(C, 50, C, 4) if n_points >= 500_000 else None

        if slice_idx is None:
            slice_idx = C // 2
        slice_idx = max(0, min(slice_idx or 0, C - 1))

        ch_names = ["pressure", "vx", "vy", "vz"]
        axis_map = {"z": (0, "XY"), "y": (1, "XZ"), "x": (2, "YZ")}
        ax_dim, plane_name = axis_map.get(slice_axis, axis_map["z"])

        slices = {}
        for ci, cn in enumerate(ch_names):
            true_slice = np.take(true_3d[:, :, :, ci], slice_idx, axis=ax_dim)
            entry = {"true": true_slice.tolist(), "shape": list(true_slice.shape)}
            if pred_3d is not None:
                pred_slice = np.take(pred_3d[:, :, :, ci], slice_idx, axis=ax_dim)
                entry["pred"] = pred_slice.tolist()
                entry["error"] = (pred_slice - true_slice).tolist()
            slices[cn] = entry

        best = _best_checkpoint()
        result = {
            "status": "ok", "snapshot": snap_idx,
            "slice_axis": slice_axis, "slice_idx": slice_idx,
            "plane_name": plane_name, "grid_size": C,
            "checkpoint": best.name if best else None,
            "device": str(dev),
            "slices": slices,
        }
        return result
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        gc.collect()
        torch.sdaa.empty_cache()

@router.get("/viz")
def suboff_visualization(
    data_dir: str = "/root/LBM-Platform/suboff_all/suboff8",
    snap_idx: int = 0,
    n_points: int = 50000,
    slice_axis: str = "z",
    slice_idx: int | None = None,
):
    """Generate flow field slice images (true vs predicted) as base64 PNGs."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from io import BytesIO
        import base64
        import glob as _glob

        # Use cached model instead of rebuilding every request
        enc, dec, device, _ = _get_model()

        # Load snapshot data
        channels = ("p", "ux", "uy", "uz")
        snap_data = {}
        for ch in channels:
            arr = np.load(os.path.join(data_dir, ch, f"{snap_idx + 1250}.npy")).astype(np.float32)
            arr = arr[49:149, :, 49:149]
            snap_data[ch] = arr

        # Build true 3D field [Z, Y, X, 4]
        C = snap_data["p"].shape[0]  # 100
        true_3d = np.stack([snap_data[ch] for ch in channels], axis=-1)  # [100, 50, 100, 4]

        # Build sparse input for model
        coords = get_suboff_coords(n_points, data_dir=data_dir).to(device)
        pos = coords.unsqueeze(0)

        # Flatten and build input tensor
        # Use ori28 (full flatten) for visualization
        flat = np.stack([snap_data[ch].flatten() for ch in channels], axis=-1)  # [500000, 4]
        x = torch.as_tensor(flat[:n_points], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            z = enc(x, pos)
            pred_out = dec(z, pos, pos)

        if n_points >= 500_000:
            pred_3d = pred_out.cpu().numpy()[0][:500_000, :].reshape(C, 50, C, 4)
        else:
            pred_3d = None

        if slice_idx is None:
            slice_idx = C // 2
        slice_idx = max(0, min(slice_idx or 0, C - 1))

        ch_names = ["pressure", "vx", "vy", "vz"]
        axis_map = {"z": (0, "XY"), "y": (1, "XZ"), "x": (2, "YZ")}
        ax_dim, plane_name = axis_map.get(slice_axis, axis_map["z"])

        images = {}
        for ci, cn in enumerate(ch_names):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            true_slice = np.take(true_3d[:, :, :, ci], slice_idx, axis=ax_dim)
            im1 = ax1.imshow(true_slice.T, origin="lower", cmap="jet", aspect="auto")
            ax1.set_title(f"True {cn} ({plane_name} slice {slice_idx})")
            plt.colorbar(im1, ax=ax1, fraction=0.046)

            if pred_3d is not None:
                pred_slice = np.take(pred_3d[:, :, :, ci], slice_idx, axis=ax_dim)
                diff = pred_slice - true_slice
                im2 = ax2.imshow(diff.T, origin="lower", cmap="RdBu_r", aspect="auto")
                ax2.set_title(f"Error {cn}")
                plt.colorbar(im2, ax=ax2, fraction=0.046)
            else:
                ax2.text(0.5, 0.5, "need more pts", ha="center", va="center")

            buf = BytesIO()
            plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            plt.close()
            images[cn] = base64.b64encode(buf.getvalue()).decode()

        return {
            "status": "ok", "snapshot": snap_idx,
            "slice_axis": slice_axis, "slice_idx": slice_idx,
            "grid_size": C, "checkpoint": (_best_checkpoint().name if _best_checkpoint() is not None else None),
            "images": images,
        }
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── Multi-dataset slice comparison (no model needed) ─────────────────────────

SUBOFF_DATA_ROOT = Path("/root/LBM-Platform/suboff_all")

@router.get("/datasets")
def suboff_datasets():
    """List available SUBOFF datasets with grid info."""
    datasets = []
    if SUBOFF_DATA_ROOT.exists():
        for d in sorted(SUBOFF_DATA_ROOT.iterdir()):
            if not d.is_dir() or not (d / "p").is_dir():
                continue
            npy_files = sorted((d / "p").glob("*.npy"), key=lambda f: int(f.stem))
            if not npy_files:
                continue
            shape = np.load(str(npy_files[0])).shape
            n_snaps = len(npy_files)
            datasets.append({
                "name": d.name,
                "path": str(d),
                "shape": list(shape),
                "n_snaps": n_snaps,
            })
    return {"datasets": datasets}


@router.get("/slice-compare")
def suboff_slice_compare(
    datasets: str = "suboff6,suboff8,suboff8_base",
    snap_idx: int = 1499,
    slice_axis: str = "z",
    slice_idx: int | None = None,
    velocity_only: bool = True,
):
    """Return mid-section velocity field slices for multiple datasets for comparison.

    Reads raw NPY data only — no AI model needed. Returns 2D slice arrays
    for ux, uy, uz (velocity components) at the specified cross-section position.

    Args:
        datasets: Comma-separated dataset names (e.g. "suboff6,suboff8,suboff8_base").
        snap_idx: Snapshot index (0-based, maps directly to file name).
        slice_axis: 'z' (XY plane), 'y' (XZ plane), 'x' (YZ plane).
        slice_idx: Position along slice axis. Default = center of that axis.
        velocity_only: If true, only return ux/uy/uz (skip pressure).
    """
    try:
        ds_names = [s.strip() for s in datasets.split(",") if s.strip()]
        if not ds_names:
            raise HTTPException(status_code=400, detail="No datasets specified")

        channels = ("ux", "uy", "uz") if velocity_only else ("p", "ux", "uy", "uz")
        axis_map = {"z": (0, "XY"), "y": (1, "XZ"), "x": (2, "YZ")}
        ax_dim, plane_name = axis_map.get(slice_axis, axis_map["z"])

        results = []
        for ds_name in ds_names:
            ds_path = SUBOFF_DATA_ROOT / ds_name
            if not ds_path.exists() or not (ds_path / "p").is_dir():
                results.append({"name": ds_name, "error": f"Dataset not found: {ds_name}"})
                continue

            # Read shape from first npy to determine grid dimensions
            first_npy = sorted((ds_path / "p").glob("*.npy"), key=lambda f: int(f.stem))[0]
            shape = np.load(str(first_npy)).shape

            # Determine slice position
            if slice_idx is None:
                slice_idx_val = shape[ax_dim] // 2
            else:
                slice_idx_val = max(0, min(slice_idx, shape[ax_dim] - 1))

            # Load channels for this snapshot
            snap_data = {}
            for ch in channels:
                fpath = ds_path / ch / f"{snap_idx}.npy"
                if not fpath.exists():
                    fpath = ds_path / ch / f"{snap_idx + 1250}.npy"
                if not fpath.exists():
                    results.append({"name": ds_name, "error": f"Snapshot {snap_idx} not found"})
                    break
                arr = np.load(str(fpath)).astype(np.float32)
                snap_data[ch] = arr

            if len(snap_data) != len(channels):
                continue

            # Compute velocity magnitude
            speed = np.sqrt(snap_data["ux"]**2 + snap_data["uy"]**2 + snap_data["uz"]**2)

            # Extract 2D slices
            slices = {}
            for ch in channels:
                slice_2d = np.take(snap_data[ch], slice_idx_val, axis=ax_dim)
                ny, nx = slice_2d.shape
                max_dim = 200
                if ny > max_dim or nx > max_dim:
                    step_y = max(1, ny // max_dim)
                    step_x = max(1, nx // max_dim)
                    slice_2d = slice_2d[::step_y, ::step_x]
                slices[ch] = {
                    "data": slice_2d.tolist(),
                    "shape": list(slice_2d.shape),
                }

            # Speed slice
            speed_slice = np.take(speed, slice_idx_val, axis=ax_dim)
            ny, nx = speed_slice.shape
            max_dim = 200
            if ny > max_dim or nx > max_dim:
                step_y = max(1, ny // max_dim)
                step_x = max(1, nx // max_dim)
                speed_slice = speed_slice[::step_y, ::step_x]
            slices["speed"] = {
                "data": speed_slice.tolist(),
                "shape": list(speed_slice.shape),
            }

            speed_stats = {
                "min": float(speed_slice.min()),
                "max": float(speed_slice.max()),
                "mean": float(speed_slice.mean()),
            }

            results.append({
                "name": ds_name,
                "shape": list(shape),
                "snap_idx": snap_idx,
                "slice_axis": slice_axis,
                "slice_idx": slice_idx_val,
                "plane_name": plane_name,
                "slices": slices,
                "speed_stats": speed_stats,
            })

        return {"status": "ok", "datasets": results, "channels": list(channels)}
    except HTTPException:
        raise
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
