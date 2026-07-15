"""Torch-based AI utilities for TensorLBM flow and turbulence workflows.

The core API covers datasets, turbulence closures, pipelines, and the
self-supervised flow transformer.  SUBOFF reconstruction is optional and is
queried explicitly with :func:`get_suboff_availability`; it never blocks this
package's core imports.
"""
from __future__ import annotations

import importlib

from .database import (
    LBMDatabase,
    connect,
    get_model_record,
    insert_dataset,
    insert_model,
    insert_run,
    list_datasets,
    list_models,
    list_runs,
)
from .dataset import (
    EddyViscosityDataset,
    extract_les_samples_2d,
    load_dataset_pt,
    save_dataset_pt,
    strain_rate_tensor_2d,
)
from .inference import collide_ai_les_bgk, predict_nu_t_2d, predict_tau_eff_2d
from .model import EddyViscosityMLP, load_model, save_model
from .pipeline import AIPipelineResult, run_ai_dns_pipeline, run_ai_les_pipeline
from .train import TrainConfig, train_eddy_viscosity_model
from .transformer import (
    FlowFieldTransformer,
    FlowTransformerArch,
    FlowTransformerTrainConfig,
    build_flow_token_batch,
    load_flow_transformer_model,
    reconstruct_flow_field,
    save_flow_transformer_model,
    train_flow_transformer_self_supervised,
)

__all__ = [
    # dataset
    "EddyViscosityDataset",
    "extract_les_samples_2d",
    "strain_rate_tensor_2d",
    "save_dataset_pt",
    "load_dataset_pt",
    # database
    "LBMDatabase",
    "connect",
    "insert_run",
    "insert_dataset",
    "insert_model",
    "get_model_record",
    "list_runs",
    "list_datasets",
    "list_models",
    # model
    "EddyViscosityMLP",
    "save_model",
    "load_model",
    # train
    "TrainConfig",
    "train_eddy_viscosity_model",
    # inference
    "predict_nu_t_2d",
    "predict_tau_eff_2d",
    "collide_ai_les_bgk",
    # pipeline
    "AIPipelineResult",
    "run_ai_dns_pipeline",
    "run_ai_les_pipeline",
    # transformer self-supervised learning
    "FlowTransformerArch",
    "FlowTransformerTrainConfig",
    "FlowFieldTransformer",
    "build_flow_token_batch",
    "train_flow_transformer_self_supervised",
    "save_flow_transformer_model",
    "load_flow_transformer_model",
    "reconstruct_flow_field",
]

_OPTIONAL_SUBOFF_MODULES = (
    "tensorlbm.ai.suboff_utils",
    "tensorlbm.ai.suboff_train",
    "tensorlbm.ai.suboff_inference",
)


def _load_optional_suboff_api() -> tuple[bool, str]:
    """Check whether all optional SUBOFF modules can be imported.

    Only a missing module that is part of the optional SUBOFF group is
    converted into an unavailable result.  Missing transitive dependencies and
    other import errors remain visible to callers.
    """
    for module_name in _OPTIONAL_SUBOFF_MODULES:
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name == module_name:
                return False, f"Optional SUBOFF module is not installed: {module_name}"
            raise
    return True, "Optional SUBOFF modules are importable."


def get_suboff_availability() -> dict[str, str | bool]:
    """Return the explicit availability state of optional SUBOFF support."""
    available, reason = _load_optional_suboff_api()
    return {
        "available": available,
        "status": "AVAILABLE" if available else "NOT_AVAILABLE",
        "reason": reason,
    }


__all__ += ["get_suboff_availability"]
