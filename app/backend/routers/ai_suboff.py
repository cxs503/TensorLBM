"""
SUBOFF API — 3D Encoder-Decoder 训练/推理端点
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()

CKPT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "checkpoints" / "suboff"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

_training_jobs: dict[str, dict] = {}


class SuboffTrainRequest(BaseModel):
    epochs: int = Field(50, ge=1, le=5000, description="训练轮数")
    batch_size: int = Field(8, ge=1, le=64)
    lr: float = Field(6e-4, gt=0, le=1e-1, description="学习率")
    n_points: int = Field(2000, ge=100, le=50000, description="每样本采样点数")
    data_dir: str | None = Field(None, description="NPY 数据目录 ({dir}/p/{idx}.npy, ...)")
    device: str = Field("cuda" if torch.cuda.is_available() else "cpu")


# ── model / loss ──

def build_model(device: torch.device):
    from tensorlbm.ai.nn.encoder_module import IrregSTEncoder2D
    from tensorlbm.ai.nn.decoder_module import IrregSTDecoder2D

    enc = IrregSTEncoder2D(
        input_channels=4, time_window=1, in_emb_dim=144, out_channels=144,
        heads=1, depth=4, res=200, use_ln=True, emb_dropout=0.0,
    ).to(device)
    dec = IrregSTDecoder2D(
        latent_channels=144, out_channels=4, res=200, scale=2, dropout=0.1,
    ).to(device)
    return enc, dec


def pointwise_rel_loss(x: torch.Tensor, y: torch.Tensor, p: int = 2) -> torch.Tensor:
    eps = 1e-8
    y_norm = y.abs() + eps if p == 1 else y.pow(p) + eps
    diff = (x - y).abs() if p == 1 else (x - y).pow(p)
    return (diff / y_norm).sum(dim=-1).mean()


# ── NPY data loading ──

def _load_npy_snapshots(data_dir: str) -> torch.Tensor | None:
    """Load multi-channel NPY snapshot data from {data_dir}/p,ux,uy,uz.
    If data_dir contains Re_* subdirectories, loads all and merges.
    Returns [num_snaps, total_points, 4] tensor or None if not found.
    Channels are normalized to [-1, 1] range."""
    import glob

    def _load_one_dir(d: str) -> torch.Tensor | None:
        channels = ("p", "ux", "uy", "uz")
        result: list[torch.Tensor] | None = None
        for ci, ch in enumerate(channels):
            cd = os.path.join(d, ch)
            if not os.path.isdir(cd):
                return None
            files = sorted(
                [f for f in os.listdir(cd) if f.endswith(".npy")],
                key=lambda x: int(x.rsplit(".", 1)[0]),
            )
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

    # Try direct first, then multi-Re
    data = _load_one_dir(data_dir)
    if data is not None:
        for ci in range(4):
            ch_data = data[..., ci]
            ch_min = ch_data.min(); ch_max = ch_data.max()
            if ch_max - ch_min > 1e-12:
                data[..., ci] = 2.0 * (ch_data - ch_min) / (ch_max - ch_min) - 1.0
        return data

    # Try Re_* subdirectories
    re_dirs = sorted(glob.glob(os.path.join(data_dir, "Re_*")))
    if not re_dirs:
        return None
    all_snaps: list[torch.Tensor] = []
    for rd in re_dirs:
        d = _load_one_dir(rd)
        if d is not None:
            all_snaps.append(d)
    if not all_snaps:
        return None
    data = torch.cat(all_snaps, dim=0)
    # Per-channel min-max normalization to [0, 1] (stable for MSE training)
    for ci in range(4):
        ch_data = data[..., ci]
        ch_min = ch_data.min(); ch_max = ch_data.max()
        if ch_max - ch_min > 1e-12:
            data[..., ci] = (ch_data - ch_min) / (ch_max - ch_min)
    return data


# ── training ──

@router.post("/train")
def train_suboff(req: SuboffTrainRequest):
    import threading

    job_id = f"suboff_train_{int(time.time())}"
    _training_jobs[job_id] = {"status": "preparing", "epoch": 0, "total": req.epochs, "loss": None}

    def worker():
        try:
            device = torch.device(req.device)
            enc, dec = build_model(device)
            opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=req.lr)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=req.lr, total_steps=req.epochs,
                pct_start=0.1, anneal_strategy="cos",
            )

            # Load real data
            real_data: torch.Tensor | None = None
            if req.data_dir:
                real_data = _load_npy_snapshots(req.data_dir)
                if real_data is not None:
                    print(f"[train] Loaded {real_data.shape[0]} snapshots, {real_data.shape[1]} points each from {req.data_dir}")

            # Load coordinates (prefer exported coords.npy)
            coords_path = os.path.join(req.data_dir, "coords.npy") if req.data_dir else None
            if coords_path and os.path.exists(coords_path):
                coords_raw = torch.tensor(np.load(coords_path), dtype=torch.float32)[:req.n_points]
            else:
                from tensorlbm.ai.suboff_coord import coord_ori27
                coords_raw = torch.tensor(coord_ori27(), dtype=torch.float32)[:req.n_points]
            pos = coords_raw.to(device)

            enc.train(); dec.train()
            best_loss = 1e10

            # Fixed contiguous point indices for stable position mapping
            n_pts = min(req.n_points, real_data.shape[1] if real_data is not None else req.n_points)

            for epoch in range(1, req.epochs + 1):
                if real_data is not None:
                    # Take fixed contiguous segment (different snap per epoch)
                    snap = real_data[torch.randint(0, real_data.shape[0], (1,)).item()]
                    x = snap[:n_pts].unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, N, 4]
                    target = x
                else:
                    # Synthetic: noisy zeros
                    noise = 0.05 * torch.randn(1, 1, req.n_points, 4, device=device)
                    x = torch.cat([torch.zeros(1, 1, req.n_points, 1, device=device), pos.unsqueeze(0).unsqueeze(0)], dim=-1)
                    target = x + noise

                opt.zero_grad()
                z = enc(x, pos.unsqueeze(0))
                pred = dec(z, pos.unsqueeze(0), pos.unsqueeze(0))
                loss = nn.functional.mse_loss(pred, target)

                loss.backward()
                nn.utils.clip_grad_norm_(list(enc.parameters()) + list(dec.parameters()), 0.1)
                opt.step()
                scheduler.step()

                loss_val = float(loss.item())
                if not (loss_val == loss_val): loss_val = 999.0
                _training_jobs[job_id] = {"status": "training", "epoch": epoch, "total": req.epochs, "loss": loss_val}

                if loss_val < best_loss:
                    best_loss = loss_val
                    torch.save({
                        "encoder": enc.state_dict(), "decoder": dec.state_dict(),
                        "config": {"model_dim": 144, "n_heads": 1, "n_layers": 4},
                        "epoch": epoch, "loss": loss_val,
                    }, str(CKPT_DIR / "suboff_best.ckpt"))

                if epoch % 50 == 0:
                    torch.save({"encoder": enc.state_dict(), "decoder": dec.state_dict()},
                               str(CKPT_DIR / f"suboff_epoch{epoch}.ckpt"))

            _training_jobs[job_id]["status"] = "completed"
            _training_jobs[job_id]["best_loss"] = best_loss

        except Exception as e:
            _training_jobs[job_id] = {"status": "failed", "error": str(e)}
            import traceback; traceback.print_exc()

    threading.Thread(target=worker, daemon=True).start()
    return {"job_id": job_id, "status": "started", "epochs": req.epochs}


@router.get("/train/{job_id}")
def train_status(job_id: str):
    if job_id not in _training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    j = _training_jobs[job_id]
    return {
        "job_id": job_id, "status": j.get("status", "?"), "epoch": j.get("epoch", 0),
        "total": j.get("total", 0), "loss": j.get("loss"),
        "best_loss": float(j.get("best_loss", 0)) if j.get("best_loss") == j.get("best_loss") else 0.0,
        "error": j.get("error"),
    }


@router.get("/train")
def list_training():
    return {"jobs": list(_training_jobs.keys())[-10:]}


# ── inference ──

_model_cache: tuple | None = None


def _get_model():
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc, dec = build_model(device)
    ckpts = sorted(CKPT_DIR.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if ckpts:
        ckpt = torch.load(str(ckpts[0]), map_location=device, weights_only=False)
        enc.load_state_dict(ckpt.get("encoder", {}), strict=False)
        dec.load_state_dict(ckpt.get("decoder", {}), strict=False)
    enc.eval(); dec.eval()
    _model_cache = (enc, dec, device, None)
    return _model_cache


class SuboffPredictRequest(BaseModel):
    n_points: int = Field(2000, ge=100, le=20000)


@router.get("/status")
def suboff_status():
    ckpts = sorted(CKPT_DIR.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return {"checkpoints": len(ckpts), "latest": ckpts[0].name if ckpts else None, "model_loaded": _model_cache is not None}


def _get_coords(data_dir: str | None, n_points: int) -> torch.Tensor:
    """Get position coordinates — prefer coords.npy, fallback to coord_ori27."""
    if data_dir:
        cp = os.path.join(data_dir, "coords.npy")
        if os.path.exists(cp):
            return torch.tensor(np.load(cp), dtype=torch.float32)[:n_points]
    from tensorlbm.ai.suboff_coord import coord_ori27
    return torch.tensor(coord_ori27(), dtype=torch.float32)[:n_points]


@router.post("/predict")
def suboff_predict(req: SuboffPredictRequest):
    try:
        enc, dec, device, _ = _get_model()
        coords = _get_coords(None, req.n_points).to(device)
        pos = coords.unsqueeze(0)
        x = torch.zeros(1, 1, req.n_points, 4, device=device)
        t0 = time.perf_counter()
        with torch.no_grad():
            z = enc(x, pos)
            pred = dec(z, pos, pos)
        elapsed = (time.perf_counter() - t0) * 1000
        p = pred.cpu().numpy()[0]
        return {
            "status": "ok", "shape": list(pred.shape), "channels": ["pressure", "vx", "vy", "vz"],
            "stats": {
                "vx": {"min": float(p[:, 1].min()), "max": float(p[:, 1].max()), "mean": float(p[:, 1].mean())},
                "vy": {"min": float(p[:, 2].min()), "max": float(p[:, 2].max()), "mean": float(p[:, 2].mean())},
            },
            "time_ms": round(elapsed, 1), "device": str(device),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── data scan ──

@router.get("/data")
def list_data(data_dir: str | None = None):
    """List available NPY snapshot directories."""
    if data_dir is None:
        data_dir = str(CKPT_DIR.parent.parent / "suboff_snapshots")
    p = Path(data_dir)
    if not p.exists():
        return {"data_dir": data_dir, "exists": False}
    result = {}
    for ch in ("p", "ux", "uy", "uz"):
        chd = p / ch
        if chd.is_dir():
            files = sorted([f.name for f in chd.glob("*.npy")], key=lambda x: int(x.rsplit(".", 1)[0]))
            result[ch] = len(files)
    return {"data_dir": data_dir, "channels": result, "total_snapshots": min(result.values()) if result else 0}


# ── fine-tuning ──

class SuboffFinetuneRequest(BaseModel):
    epochs: int = Field(30, ge=1, le=5000)
    lr: float = Field(1e-5, gt=0, le=1e-3, description="微调学习率（应低于预训练）")
    n_points: int = Field(2000, ge=100, le=50000)
    data_dir: str = Field(..., description="微调数据目录（不同 Re 工况）")
    checkpoint: str | None = Field(None, description="预训练 checkpoint 路径，默认用最新")
    device: str = Field("cuda" if torch.cuda.is_available() else "cpu")


@router.post("/finetune")
def finetune_suboff(req: SuboffFinetuneRequest):
    import threading

    job_id = f"suboff_ft_{int(time.time())}"
    _training_jobs[job_id] = {"status": "preparing", "epoch": 0, "total": req.epochs, "loss": None}

    def worker():
        try:
            from tensorlbm.ai.suboff_coord import coord_ori27

            device = torch.device(req.device)
            enc, dec = build_model(device)

            # Load pre-trained checkpoint
            ckpt_path = req.checkpoint
            if ckpt_path is None:
                ckpts = sorted(CKPT_DIR.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
                ckpt_path = str(ckpts[0]) if ckpts else None
            if ckpt_path and os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                enc.load_state_dict(ckpt.get("encoder", {}), strict=False)
                dec.load_state_dict(ckpt.get("decoder", {}), strict=False)
                _training_jobs[job_id]["pretrained_loss"] = ckpt.get("loss", "?")
            else:
                _training_jobs[job_id]["status"] = "failed"
                _training_jobs[job_id]["error"] = "No pre-trained checkpoint found"
                return

            # Load fine-tuning data
            real_data = _load_npy_snapshots(req.data_dir)
            if real_data is None:
                _training_jobs[job_id]["status"] = "failed"
                _training_jobs[job_id]["error"] = f"No data at {req.data_dir}"
                return

            opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=req.lr)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=req.lr, total_steps=req.epochs,
                pct_start=0.1, anneal_strategy="cos",
            )

            coords_raw = torch.tensor(coord_ori27(), dtype=torch.float32)[:req.n_points]
            pos = coords_raw.to(device)
            enc.train(); dec.train()
            best_loss = 1e10

            for epoch in range(1, req.epochs + 1):
                snap = real_data[torch.randint(0, real_data.shape[0], (1,)).item()]
                idxs = torch.randint(0, snap.shape[0], (req.n_points,))
                x = snap[idxs].unsqueeze(0).unsqueeze(0).to(device)

                opt.zero_grad()
                with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    z = enc(x, pos.unsqueeze(0))
                    pred = dec(z, pos.unsqueeze(0), pos.unsqueeze(0))
                    loss = nn.functional.mse_loss(pred, x)
                loss.backward()
                nn.utils.clip_grad_norm_(list(enc.parameters()) + list(dec.parameters()), 0.5)
                opt.step()
                scheduler.step()

                loss_val = float(loss.item())
                if not (loss_val == loss_val): loss_val = 999.0
                _training_jobs[job_id] = {"status": "finetuning", "epoch": epoch, "total": req.epochs, "loss": loss_val}

                if loss_val < best_loss:
                    best_loss = loss_val
                    torch.save({
                        "encoder": enc.state_dict(), "decoder": dec.state_dict(),
                        "config": {"model_dim": 144, "n_heads": 1, "n_layers": 4},
                        "epoch": epoch, "loss": loss_val, "finetuned": True,
                    }, str(CKPT_DIR / "suboff_finetuned.ckpt"))

            _training_jobs[job_id]["status"] = "completed"
            _training_jobs[job_id]["best_loss"] = best_loss

        except Exception as e:
            _training_jobs[job_id] = {"status": "failed", "error": str(e)}
            import traceback; traceback.print_exc()

    threading.Thread(target=worker, daemon=True).start()
    return {"job_id": job_id, "status": "started", "epochs": req.epochs}


# ── error analysis ──

class SuboffErrorRequest(BaseModel):
    data_dir: str = Field(..., description="测试数据目录（NPY 快照）")
    n_points: int = Field(5000, ge=100, le=50000, description="评估采样点数")
    device: str = Field("cuda" if torch.cuda.is_available() else "cpu")


@router.post("/error")
def suboff_error_analysis(req: SuboffErrorRequest):
    try:
        # Load test data
        test_data = _load_npy_snapshots(req.data_dir)
        if test_data is None:
            raise HTTPException(status_code=400, detail=f"No data at {req.data_dir}")

        from tensorlbm.ai.suboff_coord import coord_ori27
        device = torch.device(req.device)
        enc, dec = build_model(device)

        # Load best checkpoint
        ckpts = sorted(CKPT_DIR.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if ckpts:
            ckpt = torch.load(str(ckpts[0]), map_location=device, weights_only=False)
            enc.load_state_dict(ckpt.get("encoder", {}), strict=False)
            dec.load_state_dict(ckpt.get("decoder", {}), strict=False)

        enc.eval(); dec.eval()
        coords = _get_coords(req.data_dir, req.n_points).to(device)
        pos = coords.unsqueeze(0)

        all_errors: list[dict] = []
        t0 = time.perf_counter()

        for snap_idx in range(test_data.shape[0]):
            snap = test_data[snap_idx].to(device)
            idxs = torch.arange(min(req.n_points, snap.shape[0]), device=device)
            x = snap[idxs].unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                z = enc(x, pos)
                pred = dec(z, pos, pos)
            true = x.cpu().numpy()[0, 0]  # [N, 4]
            pred_np = pred.cpu().numpy()[0]

            # Per-channel errors
            ch_names = ["pressure", "vx", "vy", "vz"]
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
        for cn in ["pressure", "vx", "vy", "vz"]:
            vals = [e["channels"][cn]["rel_l2"] for e in all_errors]
            summary[cn] = {"rel_l2_mean": round(float(np.mean(vals)), 6), "rel_l2_max": round(float(np.max(vals)), 6)}

        return {
            "status": "ok", "n_snapshots": test_data.shape[0],
            "n_points": req.n_points, "time_ms": round(elapsed_ms, 1),
            "checkpoint": ckpts[0].name if ckpts else None,
            "summary": summary, "per_snapshot": all_errors,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
