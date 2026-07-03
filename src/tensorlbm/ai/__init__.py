"""AI sub-package for HPC + AI demonstration in TensorLBM.

Implements the end-to-end pipeline

    Agent → modelling/solving/post-processing
          → dataset extraction
          → SQLite persistence
          → neural-network turbulence-model training
          → AI-enhanced LBM collision (LES closure)

All components are CPU-friendly and are exposed both as ordinary Python
APIs and as platform agent tools.

Multi-backend support
---------------------
Set the active computation framework before importing (or at any point
before training / inference):

    import tensorlbm.backends as B
    B.set_backend("paddle")    # or "mindspore"

Or via environment variable::

    TENSORLBM_BACKEND=paddle python my_script.py

The LBM solver core still runs on PyTorch; only the AI sub-package uses
the multi-backend dispatch.
"""
from __future__ import annotations

from ..backends import get_backend, set_backend  # re-export for convenience

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
    # backends
    "get_backend",
    "set_backend",
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

# SUBOFF 3D surrogate modules
from tensorlbm.ai.nn import encoder_module, decoder_module, attention_module
from tensorlbm.ai.suboff_coord import coord_ori27, coord_ori28, coord_ori28_addition
from tensorlbm.ai.suboff_dataset import (
    CylinderDatasetMultiRe14,
    read_multi_re_cylinder_data27,
    read_multi_re_cylinder_data28,
    read_multi_re_cylinder_data28_addition,
)
# SUBOFF reconstruction — training, fine-tuning, inference, utilities
from .suboff_utils import (
    build_suboff_model,
    default_suboff_device,
    ensure_dir,
    get_suboff_coords,
    load_checkpoint,
    pointwise_rel_loss,
    save_checkpoint,
)
from .suboff_train import (
    SuboffFinetuneConfig,
    SuboffTrainConfig,
    finetune_suboff,
    train_suboff,
)
from .suboff_inference import (
    SuboffErrorConfig,
    SuboffPredictConfig,
    error_analysis_suboff,
    predict_suboff,
)

__all__ += [
    # SUBOFF reconstruction — utilities
    "build_suboff_model",
    "default_suboff_device",
    "ensure_dir",
    "get_suboff_coords",
    "load_checkpoint",
    "pointwise_rel_loss",
    "save_checkpoint",
    # SUBOFF reconstruction — training
    "SuboffTrainConfig",
    "train_suboff",
    "SuboffFinetuneConfig",
    "finetune_suboff",
    # SUBOFF reconstruction — inference
    "SuboffPredictConfig",
    "predict_suboff",
    "SuboffErrorConfig",
    "error_analysis_suboff",
]
