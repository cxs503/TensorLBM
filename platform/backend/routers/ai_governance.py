"""AI governance endpoints for model registry, confidence gates and active learning."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter()

_AI_ROOT = Path("/tmp/tensorlbm_platform/ai")


class ConfidenceGateRequest(BaseModel):
    prediction: float
    baseline: float
    uncertainty: float = Field(0.0, ge=0.0)
    max_relative_error: float = Field(0.15, gt=0.0)
    max_uncertainty: float = Field(0.2, ge=0.0)


class CandidateSample(BaseModel):
    sample_id: str
    uncertainty: float = Field(..., ge=0.0)
    novelty: float = Field(0.0, ge=0.0)
    impact: float = Field(0.0, ge=0.0)


class ActiveLearningRequest(BaseModel):
    candidates: list[CandidateSample] = Field(..., min_length=1, max_length=500)
    top_k: int = Field(10, ge=1, le=200)
    weights: dict[str, float] = Field(
        default_factory=lambda: {"uncertainty": 0.6, "novelty": 0.2, "impact": 0.2},
    )


def _safe_div(num: float, den: float) -> float:
    if den == 0.0:
        return 0.0 if num == 0.0 else float("inf")
    return num / den


@router.get("/registry-summary")
async def registry_summary(limit: int = 200) -> dict[str, Any]:
    """Summarize AI model registry records for governance reporting."""
    from tensorlbm import LBMDatabase

    db_path = _AI_ROOT / "platform.db"
    if not db_path.exists():
        return {"count": 0, "models": [], "quality": {}}

    db = LBMDatabase.open(db_path)
    try:
        models = db.list_models(limit=max(1, min(limit, 2000)))
    finally:
        db.close()

    if not models:
        return {"count": 0, "models": [], "quality": {}}

    train_losses = [
        float((m.get("metrics") or {}).get("final_train_loss"))
        for m in models
        if (m.get("metrics") or {}).get("final_train_loss") is not None
    ]
    val_losses = [
        float((m.get("metrics") or {}).get("final_val_loss"))
        for m in models
        if (m.get("metrics") or {}).get("final_val_loss") is not None
    ]
    val_r2 = [
        float((m.get("metrics") or {}).get("final_val_r2"))
        for m in models
        if (m.get("metrics") or {}).get("final_val_r2") is not None
    ]

    return {
        "count": len(models),
        "latest_model": models[0],
        "models": models,
        "quality": {
            "avg_train_loss": (sum(train_losses) / len(train_losses)) if train_losses else None,
            "avg_val_loss": (sum(val_losses) / len(val_losses)) if val_losses else None,
            "avg_val_r2": (sum(val_r2) / len(val_r2)) if val_r2 else None,
        },
    }


@router.post("/confidence-gate")
async def confidence_gate(req: ConfidenceGateRequest) -> dict[str, Any]:
    """Apply uncertainty + error threshold and decide AI-vs-HPC execution path."""
    relative_error = abs(_safe_div(req.prediction - req.baseline, req.baseline))
    pass_error = relative_error <= req.max_relative_error
    pass_uncertainty = req.uncertainty <= req.max_uncertainty
    accepted = pass_error and pass_uncertainty
    return {
        "accepted": accepted,
        "prediction": req.prediction,
        "baseline": req.baseline,
        "relative_error": relative_error,
        "uncertainty": req.uncertainty,
        "thresholds": {
            "max_relative_error": req.max_relative_error,
            "max_uncertainty": req.max_uncertainty,
        },
        "recommended_action": "accept_ai" if accepted else "fallback_hpc_high_fidelity",
    }


@router.post("/active-learning/prioritize")
async def prioritize_active_learning(req: ActiveLearningRequest) -> dict[str, Any]:
    """Rank candidate samples for HPC re-simulation and incremental retraining."""
    w_u = float(req.weights.get("uncertainty", 0.6))
    w_n = float(req.weights.get("novelty", 0.2))
    w_i = float(req.weights.get("impact", 0.2))

    ranked: list[dict[str, Any]] = []
    for c in req.candidates:
        score = w_u * c.uncertainty + w_n * c.novelty + w_i * c.impact
        ranked.append(
            {
                "sample_id": c.sample_id,
                "uncertainty": c.uncertainty,
                "novelty": c.novelty,
                "impact": c.impact,
                "score": score,
            },
        )
    ranked.sort(key=lambda x: x["score"], reverse=True)
    top_k = min(req.top_k, len(ranked))
    return {
        "selected": ranked[:top_k],
        "count": top_k,
        "weights": {"uncertainty": w_u, "novelty": w_n, "impact": w_i},
        "loop": "select -> hpc_resimulate -> retrain",
    }
