"""Experimental public API surface for TensorLBM.

Symbols in this module may evolve faster than those in ``tensorlbm.api``.
"""
from __future__ import annotations

from .ai import (
    AIPipelineResult,
    EddyViscosityDataset,
    EddyViscosityMLP,
    FlowFieldTransformer,
    FlowTransformerArch,
    FlowTransformerTrainConfig,
    LBMDatabase,
    TrainConfig,
    build_flow_token_batch,
    collide_ai_les_bgk,
    extract_les_samples_2d,
    load_dataset_pt,
    load_flow_transformer_model,
    load_model,
    predict_nu_t_2d,
    predict_tau_eff_2d,
    reconstruct_flow_field,
    run_ai_dns_pipeline,
    run_ai_les_pipeline,
    save_dataset_pt,
    save_flow_transformer_model,
    save_model,
    strain_rate_tensor_2d,
    train_eddy_viscosity_model,
    train_flow_transformer_self_supervised,
)

__all__ = [
    "EddyViscosityDataset",
    "EddyViscosityMLP",
    "LBMDatabase",
    "TrainConfig",
    "AIPipelineResult",
    "extract_les_samples_2d",
    "strain_rate_tensor_2d",
    "save_dataset_pt",
    "load_dataset_pt",
    "save_model",
    "load_model",
    "train_eddy_viscosity_model",
    "predict_nu_t_2d",
    "predict_tau_eff_2d",
    "collide_ai_les_bgk",
    "run_ai_dns_pipeline",
    "run_ai_les_pipeline",
    "FlowTransformerArch",
    "FlowTransformerTrainConfig",
    "FlowFieldTransformer",
    "build_flow_token_batch",
    "train_flow_transformer_self_supervised",
    "save_flow_transformer_model",
    "load_flow_transformer_model",
    "reconstruct_flow_field",
]

