"""
SUBOFF 预训练 API — 3D Encoder-Decoder 训练端点
"""
from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()

CKPT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / "checkpoints" / "suboff"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

_training_jobs: dict[str, dict] = {}


class SuboffTrainRequest(BaseModel):
    epochs: int = Field(50, ge=1, le=5000, description="训练轮数")
    batch_size: int = Field(8, ge=1, le=64)
    lr: float = Field(6e-4, gt=0, le=1e-1, description="学习率")
    n_points: int = Field(2000, ge=100, le=50000, description="每样本坐标点数")
    device: str = Field("cuda" if torch.cuda.is_available() else "cpu")


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


@router.post("/train")
def train_suboff(req: SuboffTrainRequest):
    import threading

    job_id = f"suboff_train_{int(time.time())}"
    _training_jobs[job_id] = {"status": "preparing", "epoch": 0, "total": req.epochs, "loss": None}

    def worker():
        try:
            from tensorlbm.ai.suboff_coord import coord_ori27

            device = torch.device(req.device)
            enc, dec = build_model(device)
            opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=req.lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=req.lr, total_steps=req.epochs,
                pct_start=0.1, anneal_strategy="cos",
            )

            # 合成数据（真实训练需 NPY 文件）
            coords_raw = torch.tensor(coord_ori27(), dtype=torch.float32)[:req.n_points]
            pos = coords_raw.to(device)

            enc.train(); dec.train()
            best_loss = 1e10

            for epoch in range(1, req.epochs + 1):
                # 合成输入：添加噪声模拟流场变化
                noise = 0.05 * torch.randn(1, 1, req.n_points, 4, device=device)
                x = torch.cat([torch.zeros(1, 1, req.n_points, 1, device=device), pos.unsqueeze(0).unsqueeze(0)], dim=-1)
                x = x + noise
                target = x.clone()

                opt.zero_grad()
                with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    z = enc(x, pos.unsqueeze(0))
                    pred = dec(z, pos.unsqueeze(0), pos.unsqueeze(0))
                    loss = pointwise_rel_loss(pred, target)

                loss.backward()
                nn.utils.clip_grad_norm_(list(enc.parameters()) + list(dec.parameters()), 1.0)
                opt.step()
                scheduler.step()

                loss_val = float(loss.item())
                if not (loss_val==loss_val): loss_val=999.0  # sanitize NaN
                _training_jobs[job_id] = {"status": "training", "epoch": epoch, "total": req.epochs, "loss": loss_val}

                if loss_val < best_loss:
                    best_loss = loss_val
                    ckpt_path = CKPT_DIR / f"suboff_best.ckpt"
                    torch.save({
                        "encoder": enc.state_dict(), "decoder": dec.state_dict(),
                        "config": {"model_dim": 144, "n_heads": 1, "n_layers": 4},
                        "epoch": epoch, "loss": loss_val,
                    }, str(ckpt_path))

                if epoch % 10 == 0:
                    ckpt_path = CKPT_DIR / f"suboff_epoch{epoch}.ckpt"
                    torch.save({"encoder": enc.state_dict(), "decoder": dec.state_dict()}, str(ckpt_path))

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
    return {"job_id": job_id, "status": j.get("status","?"), "epoch": j.get("epoch",0),
            "total": j.get("total",0), "loss": j.get("loss"), "best_loss": float(j.get("best_loss",0)) if j.get("best_loss")==j.get("best_loss") else 0.0}


@router.get("/train")
def list_training():
    return {"jobs": list(_training_jobs.keys())[-10:]}


# ---- Inference ----

_model_cache: tuple | None = None


def _get_model():
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    from tensorlbm.ai.suboff_coord import coord_ori27
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc, dec = build_model(device)
    ckpts = sorted(CKPT_DIR.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if ckpts:
        ckpt = torch.load(str(ckpts[0]), map_location=device, weights_only=False)
        enc.load_state_dict(ckpt.get("encoder", {}), strict=False)
        dec.load_state_dict(ckpt.get("decoder", {}), strict=False)
    enc.eval(); dec.eval()
    _model_cache = (enc, dec, device, coord_ori27)
    return _model_cache


class SuboffPredictRequest(BaseModel):
    n_points: int = Field(2000, ge=100, le=20000)


@router.get("/status")
def suboff_status():
    ckpts = sorted(CKPT_DIR.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return {"checkpoints": len(ckpts), "latest": ckpts[0].name if ckpts else None, "model_loaded": _model_cache is not None}


@router.post("/predict")
def suboff_predict(req: SuboffPredictRequest):
    try:
        enc, dec, device, coord_fn = _get_model()
        coords = torch.tensor(coord_fn(), dtype=torch.float32, device=device)[:req.n_points]
        pos = coords.unsqueeze(0)
        x = torch.cat([torch.zeros(1, 1, req.n_points, 1, device=device), pos.unsqueeze(1)], dim=-1)
        t0 = time.perf_counter()
        with torch.no_grad():
            z = enc(x, pos)
            pred = dec(z, pos, pos)
        elapsed = (time.perf_counter() - t0) * 1000
        p = pred.cpu().numpy()[0]
        return {
            "status": "ok", "shape": list(pred.shape),
            "stats": {
                "vx": {"min": float(p[:,1].min()), "max": float(p[:,1].max()), "mean": float(p[:,1].mean())},
                "vy": {"min": float(p[:,2].min()), "max": float(p[:,2].max()), "mean": float(p[:,2].mean())},
            },
            "time_ms": round(elapsed, 1), "device": str(device),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
