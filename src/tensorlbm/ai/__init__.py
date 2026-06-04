"""AI sub-package for HPC + AI demonstration in TensorLBM.

Implements the end-to-end pipeline

    Agent → modelling/solving/post-processing
          → dataset extraction
          → SQLite persistence
          → neural-network turbulence-model training
          → AI-enhanced LBM collision (LES closure)

All components are CPU-friendly, depend only on ``torch`` and the Python
standard library (``sqlite3``), and are exposed both as ordinary Python
APIs and as platform agent tools.
"""
from __future__ import annotations

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
    "run_ai_les_pipeline",
    "run_ai_dns_pipeline",
]
