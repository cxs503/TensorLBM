"""AI governance endpoints for model registry, confidence gates and active learning."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter()

_AI_ROOT = Path("/tmp/tensorlbm_platform/ai")
_POLICY_REGISTRY: dict[str, dict[str, Any]] = {}


class ConfidenceGateRequest(BaseModel):
    prediction: float
    baseline: float
    scenario: str = Field(default="default", min_length=1, max_length=128)
    model_id: str = Field(default="default", min_length=1, max_length=128)
    uncertainty: float = Field(0.0, ge=0.0)
    max_relative_error: float = Field(0.15, gt=0.0)
    max_uncertainty: float = Field(0.2, ge=0.0)
    ci_half_width: float = Field(0.0, ge=0.0)


class GovernancePolicyRequest(BaseModel):
    scenario: str = Field(..., min_length=1, max_length=128)
    model_id: str = Field(default="default", min_length=1, max_length=128)
    max_relative_error: float = Field(0.15, gt=0.0)
    max_uncertainty: float = Field(0.2, ge=0.0)
    max_ci_half_width: float = Field(0.1, ge=0.0)
    drift_threshold: float = Field(0.2, ge=0.0)
    require_human_review_error: float = Field(0.3, ge=0.0)
    require_human_review_uncertainty: float = Field(0.25, ge=0.0)


class DriftMonitorRequest(BaseModel):
    scenario: str = Field(default="default", min_length=1, max_length=128)
    model_id: str = Field(default="default", min_length=1, max_length=128)
    baseline_mean: float
    current_mean: float
    baseline_std: float = Field(0.0, ge=0.0)
    current_std: float = Field(0.0, ge=0.0)
    sample_count: int = Field(1, ge=1)


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


def _policy_key(scenario: str, model_id: str) -> str:
    return f"{scenario}:{model_id}"


def _resolve_policy(scenario: str, model_id: str) -> dict[str, Any]:
    default = {
        "max_relative_error": 0.15,
        "max_uncertainty": 0.2,
        "max_ci_half_width": 0.1,
        "drift_threshold": 0.2,
        "require_human_review_error": 0.3,
        "require_human_review_uncertainty": 0.25,
    }
    return {
        **default,
        **_POLICY_REGISTRY.get(_policy_key(scenario, model_id), {}),
    }


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
    policy = _resolve_policy(req.scenario, req.model_id)
    max_relative_error = min(float(req.max_relative_error), float(policy["max_relative_error"]))
    max_uncertainty = min(float(req.max_uncertainty), float(policy["max_uncertainty"]))
    max_ci_half_width = float(policy["max_ci_half_width"])
    relative_error = abs(_safe_div(req.prediction - req.baseline, req.baseline))
    pass_error = relative_error <= max_relative_error
    pass_uncertainty = req.uncertainty <= max_uncertainty
    pass_ci = req.ci_half_width <= max_ci_half_width
    accepted = pass_error and pass_uncertainty and pass_ci
    human_review = (
        relative_error >= float(policy["require_human_review_error"])
        or req.uncertainty >= float(policy["require_human_review_uncertainty"])
    )
    return {
        "accepted": accepted,
        "scenario": req.scenario,
        "model_id": req.model_id,
        "prediction": req.prediction,
        "baseline": req.baseline,
        "relative_error": relative_error,
        "uncertainty": req.uncertainty,
        "ci_half_width": req.ci_half_width,
        "thresholds": {
            "max_relative_error": max_relative_error,
            "max_uncertainty": max_uncertainty,
            "max_ci_half_width": max_ci_half_width,
        },
        "human_review_required": human_review,
        "recommended_action": (
            "manual_review_required"
            if human_review
            else ("accept_ai" if accepted else "fallback_hpc_high_fidelity")
        ),
    }


@router.post("/policies")
async def upsert_policy(req: GovernancePolicyRequest) -> dict[str, Any]:
    """Upsert adaptive governance policy for scenario/model pair."""
    key = _policy_key(req.scenario, req.model_id)
    payload = req.model_dump()
    _POLICY_REGISTRY[key] = {
        "max_relative_error": payload["max_relative_error"],
        "max_uncertainty": payload["max_uncertainty"],
        "max_ci_half_width": payload["max_ci_half_width"],
        "drift_threshold": payload["drift_threshold"],
        "require_human_review_error": payload["require_human_review_error"],
        "require_human_review_uncertainty": payload["require_human_review_uncertainty"],
    }
    return {"policy_key": key, "policy": _POLICY_REGISTRY[key]}


@router.get("/policies")
async def list_policies() -> dict[str, Any]:
    """List configured governance policies."""
    return {"count": len(_POLICY_REGISTRY), "policies": _POLICY_REGISTRY}


@router.post("/drift-monitor")
async def drift_monitor(req: DriftMonitorRequest) -> dict[str, Any]:
    """Compute normalized drift score and decide recalibration action."""
    policy = _resolve_policy(req.scenario, req.model_id)
    baseline_scale = max(abs(req.baseline_mean), req.baseline_std, 1e-12)
    mean_shift = abs(req.current_mean - req.baseline_mean)
    std_shift = abs(req.current_std - req.baseline_std)
    drift_score = (mean_shift + 0.5 * std_shift) / baseline_scale
    threshold = float(policy["drift_threshold"])
    drifted = drift_score >= threshold
    return {
        "scenario": req.scenario,
        "model_id": req.model_id,
        "sample_count": req.sample_count,
        "drift_score": drift_score,
        "drift_threshold": threshold,
        "drifted": drifted,
        "recommended_action": (
            "recalibrate_and_enqueue_hpc_samples" if drifted else "monitor"
        ),
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
