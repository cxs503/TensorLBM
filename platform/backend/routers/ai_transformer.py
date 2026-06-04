"""Transformer self-supervised flow-model API endpoints."""
from __future__ import annotations

import uuid
from pathlib import Path

import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()

_AI_ROOT = Path("/tmp/tensorlbm_platform/ai")
_AI_ROOT.mkdir(parents=True, exist_ok=True)


class TransformerTrainRequest(BaseModel):
    nx: int = Field(48, ge=16, le=256)
    ny: int = Field(48, ge=16, le=256)
    tau: float = 0.8
    c_s: float = 0.1
    data_steps: int = Field(40, ge=1, le=2000)
    sample_every: int = Field(10, ge=1, le=2000)
    seed: int = 0
    device: str = "cpu"
    epochs: int = Field(20, ge=1, le=500)
    batch_size: int = Field(8, ge=1, le=512)
    learning_rate: float = 1e-3
    mask_ratio: float = Field(0.15, gt=0.0, lt=1.0)
    d_model: int = Field(32, ge=8, le=256)
    n_heads: int = Field(4, ge=1, le=8)
    n_layers: int = Field(2, ge=1, le=8)
    ffn_dim: int = Field(128, ge=16, le=1024)


class TransformerInferRequest(BaseModel):
    model_id: int | None = None
    ux: list[list[float]] | None = None
    uy: list[list[float]] | None = None
    nx: int = Field(48, ge=8, le=256)
    ny: int = Field(48, ge=8, le=256)
    seed: int = 0


def _generate_snapshots(
    nx: int,
    ny: int,
    tau: float,
    c_s: float,
    data_steps: int,
    sample_every: int,
    seed: int,
    device: str,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    from tensorlbm import equilibrium, macroscopic
    from tensorlbm.solver import stream
    from tensorlbm.turbulence import collide_smagorinsky_bgk

    dev = torch.device(device)
    torch.manual_seed(int(seed))
    ys = torch.arange(ny, device=dev).float()
    xs = torch.arange(nx, device=dev).float()
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    kx = 2.0 * torch.pi / max(nx, 1)
    ky = 2.0 * torch.pi / max(ny, 1)
    ux = 0.05 + 0.02 * torch.sin(2.0 * kx * xx) * torch.cos(ky * yy)
    uy = 0.02 * torch.cos(kx * xx) * torch.sin(2.0 * ky * yy)
    f = equilibrium(torch.ones_like(ux), ux, uy)

    out: list[tuple[torch.Tensor, torch.Tensor]] = []
    for step in range(int(data_steps)):
        f = collide_smagorinsky_bgk(f, tau=float(tau), C_s=float(c_s))
        f = stream(f)
        if (step + 1) % int(sample_every) == 0:
            _rho, ux_s, uy_s = macroscopic(f)
            out.append((ux_s.detach().cpu().clone(), uy_s.detach().cpu().clone()))
    if not out:
        _rho, ux_s, uy_s = macroscopic(f)
        out.append((ux_s.detach().cpu().clone(), uy_s.detach().cpu().clone()))
    return out


def _resolve_model_path(model_id: int | None) -> Path:
    from tensorlbm import LBMDatabase

    db_path = _AI_ROOT / "platform.db"
    if not db_path.exists():
        raise HTTPException(status_code=404, detail="AI model database not found")
    db = LBMDatabase.open(db_path)
    try:
        if model_id is None:
            models = db.list_models(limit=100)
            rows = [
                m for m in models
                if isinstance(m.get("arch"), dict)
                and m["arch"].get("model_family") == "flow_transformer_ssl"
            ]
            if not rows:
                raise HTTPException(status_code=404, detail="No transformer model found")
            row = rows[0]
        else:
            row = db.get_model_record(int(model_id))
    finally:
        db.close()
    if row is None:
        raise HTTPException(status_code=404, detail=f"model_id={model_id} not found")
    p = Path(row["path"])
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"model file missing: {p}")
    if _AI_ROOT.resolve() not in p.resolve().parents:
        raise HTTPException(status_code=422, detail="Resolved model path is outside AI workspace")
    return p


@router.post("/transformer/train")
async def train_transformer(req: TransformerTrainRequest) -> dict:
    from tensorlbm import (
        FlowTransformerArch,
        FlowTransformerTrainConfig,
        LBMDatabase,
        train_flow_transformer_self_supervised,
    )

    try:
        snapshots = _generate_snapshots(
            nx=req.nx,
            ny=req.ny,
            tau=req.tau,
            c_s=req.c_s,
            data_steps=req.data_steps,
            sample_every=min(req.sample_every, req.data_steps),
            seed=req.seed,
            device=req.device,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    work = _AI_ROOT / f"transformer_{uuid.uuid4().hex[:8]}"
    work.mkdir(parents=True, exist_ok=True)
    model_path = work / "flow_transformer.pt"

    arch = FlowTransformerArch(
        d_model=req.d_model,
        n_heads=req.n_heads,
        n_layers=req.n_layers,
        ffn_dim=req.ffn_dim,
        max_tokens=max(1024, req.nx * req.ny),
    )
    cfg = FlowTransformerTrainConfig(
        epochs=req.epochs,
        batch_size=req.batch_size,
        learning_rate=req.learning_rate,
        mask_ratio=req.mask_ratio,
        seed=req.seed,
        device=req.device,
    )

    try:
        train_meta = train_flow_transformer_self_supervised(
            snapshots=snapshots,
            out_path=model_path,
            arch=arch,
            config=cfg,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    db_path = _AI_ROOT / "platform.db"
    db = LBMDatabase.open(db_path)
    try:
        run_id = db.insert_run(
            name=f"flow_transformer_ssl_{req.nx}x{req.ny}",
            run_type="ai_flow_transformer_ssl",
            config=req.model_dump(),
            output_dir=str(work),
        )
        model_id = db.insert_model(
            name=f"flow_transformer_ssl_e{req.epochs}",
            path=str(model_path),
            arch={"model_family": "flow_transformer_ssl", **train_meta["arch"]},
            dataset_id=None,
            metrics={
                "final_train_loss": train_meta["final_train_loss"],
                "final_val_loss": train_meta["final_val_loss"],
            },
        )
    finally:
        db.close()

    return {
        "ok": True,
        "model_id": model_id,
        "run_id": run_id,
        "model_path": str(model_path),
        "n_snapshots": len(snapshots),
        "grid": [req.ny, req.nx],
        "final_train_loss": train_meta["final_train_loss"],
        "final_val_loss": train_meta["final_val_loss"],
        "history": train_meta["history"],
    }


@router.get("/transformer/models")
async def list_transformer_models(limit: int = 20) -> dict:
    from tensorlbm import LBMDatabase

    db_path = _AI_ROOT / "platform.db"
    if not db_path.exists():
        return {"count": 0, "models": []}
    db = LBMDatabase.open(db_path)
    try:
        rows = db.list_models(limit=max(1, min(limit, 200)))
    finally:
        db.close()

    models = [
        r for r in rows
        if isinstance(r.get("arch"), dict)
        and r["arch"].get("model_family") == "flow_transformer_ssl"
    ]
    return {"count": len(models), "models": models}


@router.post("/transformer/infer")
async def infer_transformer(req: TransformerInferRequest) -> dict:
    from tensorlbm import load_flow_transformer_model, reconstruct_flow_field

    path = _resolve_model_path(req.model_id)

    if req.ux is not None and req.uy is not None:
        ux = torch.tensor(req.ux, dtype=torch.float32)
        uy = torch.tensor(req.uy, dtype=torch.float32)
        if ux.ndim != 2 or uy.ndim != 2 or ux.shape != uy.shape:
            raise HTTPException(status_code=422, detail="ux/uy must be same-shape 2-D arrays")
    else:
        ys = torch.arange(req.ny).float()
        xs = torch.arange(req.nx).float()
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        kx = 2.0 * torch.pi / max(req.nx, 1)
        ky = 2.0 * torch.pi / max(req.ny, 1)
        ux = 0.05 + 0.02 * torch.sin(2.0 * kx * xx + 0.1 * req.seed) * torch.cos(ky * yy)
        uy = 0.02 * torch.cos(kx * xx) * torch.sin(2.0 * ky * yy + 0.2 * req.seed)

    try:
        model = load_flow_transformer_model(path)
        pred = reconstruct_flow_field(model, ux, uy)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "ok": True,
        "model_path": str(path),
        "grid": [int(ux.shape[0]), int(ux.shape[1])],
        "mse": pred["mse"],
        "max_abs_error": pred["max_abs_error"],
        "ux_mean": float(ux.mean()),
        "uy_mean": float(uy.mean()),
        "ux_rec_mean": float(pred["ux_reconstructed"].mean()),
        "uy_rec_mean": float(pred["uy_reconstructed"].mean()),
    }
