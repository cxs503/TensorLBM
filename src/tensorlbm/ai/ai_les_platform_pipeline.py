"""AI-LES (AI turbulence model) case, integrated on the HPC+AI platform.

Wires the flagship AI-LES demonstration — HPC LBM solving -> strain-rate
sampling -> SQLite persistence -> MLP eddy-viscosity training -> AI-enhanced
LBM validation — into the new platform modules (data catalog / training-job
registry / model serving), mirroring the SUBOFF integration.

The heavy lifting stays in :func:`tensorlbm.ai.pipeline.run_ai_les_pipeline`;
this module wraps it and records every step (run / dataset / job / model) plus
the full lineage in the platform ledger.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from tensorlbm.data.catalog import (
    AssetRecord,
    FieldDataCatalog,
    LineageRecord,
)
from tensorlbm.ml.training_job import TrainingJob, TrainingJobRegistry


class AILesPlatformPipeline:
    """AI-LES full pipeline on the platform: solve -> register -> train -> serve."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self.catalog = FieldDataCatalog.open(self.db_path)
        self.training = TrainingJobRegistry.open(self.db_path)
        from tensorlbm.ml.serving import ModelRegistry

        self.serving = ModelRegistry.open(self.db_path)

    def close(self) -> None:
        self.catalog.close()
        self.training.close()
        self.serving.close()

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(
        self,
        work_dir: str | Path,
        *,
        name_prefix: str = "ai_les",
        pipeline_fn: Callable[..., Any] | None = None,
        pipeline_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run the AI-LES pipeline and record the whole flow on the platform.

        ``pipeline_fn`` defaults to
        :func:`tensorlbm.ai.pipeline.run_ai_les_pipeline`; inject a stub in
        tests to avoid a real LBM run.
        """
        if pipeline_fn is None:
            from tensorlbm.ai.pipeline import run_ai_les_pipeline

            pipeline_fn = run_ai_les_pipeline

        result = pipeline_fn(str(work_dir), **(pipeline_kwargs or {}))

        run_asset_id = self._register_run(result, name_prefix)
        dataset_asset_id = self._register_dataset(result, name_prefix)
        job = self._register_training_job(result, name_prefix)
        model_id = self._register_model(result, name_prefix)

        # lineage: run -> dataset -> job -> model
        self.catalog.add_lineage(
            LineageRecord(
                source_id=run_asset_id,
                target_id=dataset_asset_id,
                relation_type="derived_from",
                resource_type="run",
            )
        )
        job_asset_id = f"{name_prefix}:job:{job.job_id}"
        self.catalog.add_lineage(
            LineageRecord(
                source_id=dataset_asset_id,
                target_id=job_asset_id,
                relation_type="trained_on",
                resource_type="dataset",
            )
        )

        return {
            "run_asset_id": run_asset_id,
            "dataset_asset_id": dataset_asset_id,
            "job_id": job.job_id,
            "model_id": model_id,
            "pipeline_result": _result_to_dict(result),
        }

    # ------------------------------------------------------------------
    # Step registrars
    # ------------------------------------------------------------------

    def _register_run(self, result: Any, name_prefix: str) -> str:
        asset_id = f"{name_prefix}:run:{result.run_id}"
        self.catalog.register_asset(
            AssetRecord(
                asset_id=asset_id,
                name=f"AI-LES data run {result.run_id}",
                kind="run",
                description=f"data_source={result.data_source} snapshots={result.n_snapshots}",
                tags=(name_prefix, "ai-les"),
            )
        )
        return asset_id

    def _register_dataset(self, result: Any, name_prefix: str) -> str:
        asset_id = f"{name_prefix}:dataset:{result.dataset_id}"
        self.catalog.register_asset(
            AssetRecord(
                asset_id=asset_id,
                name=f"AI-LES strain-rate dataset {result.dataset_id}",
                kind="dataset",
                description=f"n_samples={result.n_samples} path={result.dataset_path}",
                tags=(name_prefix, "ai-les"),
            )
        )
        return asset_id

    def _register_training_job(self, result: Any, name_prefix: str) -> TrainingJob:
        cfg = result.to_dict() if hasattr(result, "to_dict") else {}
        # only keep scalar config for the job record
        config = {
            k: v for k, v in cfg.items() if isinstance(v, (str, int, float, bool)) or v is None
        }
        job = self.training.create_job(
            config,
            dataset_id=getattr(result, "dataset_id", None),
        )
        self.training.update_status(job.job_id, "running")
        metrics = {
            k: v
            for k, v in (getattr(result, "training", None) or {}).items()
            if isinstance(v, (int, float))
        }
        if metrics:
            self.training.record_metrics(job.job_id, metrics)
        self.training.update_status(job.job_id, "completed")
        return job

    def _register_model(self, result: Any, name_prefix: str) -> int:
        return self.serving.register_model(
            name=f"{name_prefix}-eddy-mlp",
            path=str(result.model_path),
            arch={"family": "eddy_viscosity_mlp"},
            dataset_id=getattr(result, "dataset_id", None),
            metrics={
                k: v
                for k, v in (getattr(result, "training", None) or {}).items()
                if isinstance(v, (int, float))
            },
            family="eddy_viscosity_mlp",
        )

    def upstream_assets(self, asset_id: str) -> list[str]:
        return self.catalog.upstream(asset_id)


def _result_to_dict(result: Any) -> dict[str, Any]:
    if hasattr(result, "to_dict"):
        return result.to_dict()
    if hasattr(result, "__dataclass_fields__"):
        return asdict(result)
    return dict(vars(result)) if hasattr(result, "__dict__") else {}
