"""Flow Transformer (self-supervised) case, integrated on the HPC+AI platform.

Wires the self-supervised masked-reconstruction flow transformer into the
platform modules (data catalog / training-job registry / model serving),
mirroring the SUBOFF and AI-LES integrations.

The training itself stays in
:func:`tensorlbm.ai.transformer.train_flow_transformer_self_supervised`; this
module records the flow-field data, the training job, and the model, plus the
full lineage (data -> dataset -> job -> model).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from tensorlbm.data.catalog import (
    AssetRecord,
    FieldDataCatalog,
    LineageRecord,
)
from tensorlbm.ml.training_job import TrainingJob, TrainingJobRegistry


class FlowTransformerPlatformPipeline:
    """Flow-transformer pipeline: register snapshots -> train -> serve."""

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
        snapshots: Sequence[tuple[torch.Tensor, torch.Tensor]],
        *,
        name_prefix: str = "flow_tf",
        out_name: str = "flow_transformer.pt",
        train_fn: Callable[..., dict[str, Any]] | None = None,
        arch: Any = None,
        train_config: Any = None,
    ) -> dict[str, Any]:
        """Train a flow transformer and record the whole flow on the platform.

        ``snapshots`` is a sequence of ``(ux, uy)`` velocity-field pairs.
        ``train_fn`` defaults to
        :func:`tensorlbm.ai.transformer.train_flow_transformer_self_supervised`.
        """
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        out_path = work_dir / out_name

        if train_fn is None:
            from tensorlbm.ai.transformer import train_flow_transformer_self_supervised

            train_fn = train_flow_transformer_self_supervised

        data_asset_id, dataset_asset_id, dataset_id = self._register_data(
            snapshots,
            name_prefix,
        )
        result = train_fn(
            list(snapshots),
            str(out_path),
            arch=arch,
            config=train_config,
        )
        job = self._register_training_job(result, dataset_id, name_prefix)
        model_id = self._register_model(result, dataset_id, name_prefix)

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
            "data_asset_id": data_asset_id,
            "dataset_asset_id": dataset_asset_id,
            "job_id": job.job_id,
            "model_id": model_id,
            "training_result": result,
        }

    # ------------------------------------------------------------------
    # Step registrars
    # ------------------------------------------------------------------

    def _register_data(
        self,
        snapshots: Sequence[tuple[torch.Tensor, torch.Tensor]],
        name_prefix: str,
    ) -> tuple[str, str, int]:
        if not snapshots:
            raise ValueError("at least one (ux, uy) snapshot is required")
        ux0, _ = snapshots[0]
        shape = tuple(ux0.shape)

        data_asset_id = f"{name_prefix}:flow-field"
        self.catalog.register_asset(
            AssetRecord(
                asset_id=data_asset_id,
                name="Flow-field velocity snapshots",
                kind="field_product",
                field_name="u",
                units="lu",
                shape=str(shape),
                dtype=str(ux0.dtype),
                tags=(name_prefix, "flow-transformer"),
            )
        )

        dataset_asset_id = f"{name_prefix}:dataset"
        self.catalog.register_asset(
            AssetRecord(
                asset_id=dataset_asset_id,
                name="Flow-transformer snapshot dataset",
                kind="dataset",
                description=f"n_snapshots={len(snapshots)} grid={shape}",
                tags=(name_prefix, "flow-transformer"),
            )
        )
        self.catalog.add_lineage(
            LineageRecord(
                source_id=data_asset_id,
                target_id=dataset_asset_id,
                relation_type="derived_from",
                resource_type="product",
            )
        )

        from tensorlbm.ai.database import insert_dataset

        dataset_id = insert_dataset(
            self.training._conn,
            name=f"{name_prefix}-dataset",
            path="",  # snapshots are in-memory, not on disk
            n_samples=len(snapshots),
            metadata={"grid": list(shape), "task": "flow-reconstruction"},
        )
        return data_asset_id, dataset_asset_id, dataset_id

    def _register_training_job(
        self,
        result: dict[str, Any],
        dataset_id: int | None,
        name_prefix: str,
    ) -> TrainingJob:
        config = {
            k: v
            for k, v in (result.get("config") or {}).items()
            if isinstance(v, (str, int, float, bool)) or v is None
        }
        job = self.training.create_job(config, dataset_id=dataset_id)
        self.training.update_status(job.job_id, "running")
        metrics = {}
        for key in ("final_train_loss", "final_val_loss", "n_tokens", "n_snapshots"):
            val = result.get(key)
            if isinstance(val, (int, float)):
                metrics[key] = val
        if metrics:
            self.training.record_metrics(job.job_id, metrics)
        self.training.update_status(job.job_id, "completed")
        return job

    def _register_model(
        self,
        result: dict[str, Any],
        dataset_id: int | None,
        name_prefix: str,
    ) -> int:
        metrics = {
            k: v
            for k, v in result.items()
            if k in ("final_train_loss", "final_val_loss") and isinstance(v, (int, float))
        }
        return self.serving.register_model(
            name=f"{name_prefix}-flow-transformer",
            path=str(result.get("path", "")),
            arch=dict(result.get("arch") or {}),
            dataset_id=dataset_id,
            metrics=metrics,
            family="flow_transformer_ssl",
        )

    def upstream_assets(self, asset_id: str) -> list[str]:
        return self.catalog.upstream(asset_id)
