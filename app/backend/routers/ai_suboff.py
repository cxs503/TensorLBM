"""
SUBOFF 3D surrogate model — inference API for TensorLBM Platform.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()

CKPT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "checkpoints" / "suboff"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

_model_cache: tuple | None = None


class SuboffPredictRequest(BaseModel):
    n_points: int = Field(2000, ge=100, le=20000, description="坐标点数")
    device: str = Field("cpu", description="推理设备 cpu/cuda")


def _get_model():
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    from tensorlbm.ai.nn.encoder_module import IrregSTEncoder2D
    from tensorlbm.ai.nn.decoder_module import IrregSTDecoder2D
    from tensorlbm.ai.suboff_coord import coord_ori27

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enc = IrregSTEncoder2D(
        input_channels=4, time_window=1, in_emb_dim=144, out_channels=144,
        heads=1, depth=4, res=200, use_ln=True, emb_dropout=0.0,
    ).to(device)
    dec = IrregSTDecoder2D(
        latent_channels=144, out_channels=4, res=200, scale=2, dropout=0.1,
    ).to(device)

    ckpts = sorted(CKPT_DIR.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if ckpts:
        ckpt = torch.load(str(ckpts[0]), map_location=device, weights_only=False)
        enc.load_state_dict(ckpt.get("encoder", {}), strict=False)
        dec.load_state_dict(ckpt.get("decoder", {}), strict=False)

    enc.eval()
    dec.eval()
    _model_cache = (enc, dec, device, coord_ori27)
    return _model_cache


@router.get("/status")
async def suboff_status():
    ckpts = sorted(CKPT_DIR.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return {
        "checkpoints": len(ckpts),
        "latest": ckpts[0].name if ckpts else None,
        "model_loaded": _model_cache is not None,
    }


@router.post("/predict")
async def suboff_predict(req: SuboffPredictRequest):
    try:
        enc, dec, device, coord_fn = _get_model()
        coords = torch.tensor(coord_fn(), dtype=torch.float32, device=device)
        n = min(coords.shape[0], req.n_points)
        coords = coords[:n]
        pos = coords.unsqueeze(0)
        x = torch.cat([torch.zeros(1, 1, n, 1, device=device), pos.unsqueeze(1)], dim=-1)

        t0 = time.perf_counter()
        with torch.no_grad():
            z = enc(x, pos)
            pred = dec(z, pos, pos)
        elapsed = (time.perf_counter() - t0) * 1000

        p = pred.cpu().numpy()[0]
        return {
            "status": "ok",
            "shape": list(pred.shape),
            "channels": ["pressure", "vx", "vy", "vz"],
            "stats": {
                "vx": {"min": float(p[:, 1].min()), "max": float(p[:, 1].max()), "mean": float(p[:, 1].mean())},
                "vy": {"min": float(p[:, 2].min()), "max": float(p[:, 2].max()), "mean": float(p[:, 2].mean())},
            },
            "time_ms": round(elapsed, 1),
            "device": str(device),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
