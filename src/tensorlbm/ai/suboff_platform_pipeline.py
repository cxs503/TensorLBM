"""SUBOFF case, end-to-end on the HPC+AI platform (clean-room integration).

Wires the pre-existing SUBOFF case (LBM field data -> surrogate training ->
inference) into the new platform modules:

* data management  -> ``tensorlbm.data.catalog.FieldDataCatalog``
* model training   -> ``tensorlbm.ml.training_job.TrainingJobRegistry``
* model serving    -> ``tensorlbm.ml.serving.ModelRegistry`` / ``InferenceService``

The pipeline itself only orchestrates the already-existing pieces; the SUBOFF
training/inference functions are injected (or default to the ``tensorlbm.ai``
implementations) so the whole flow is testable without a real LBM run.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from tensorlbm.data.catalog import (
    AssetRecord,
    FieldDataCatalog,
    LineageRecord,
    QualityCheck,
)
from tensorlbm.ml.training_job import TrainingJob, TrainingJobRegistry

# SUBOFF LBM snapshot channels (data_dir/<channel>/*.npy)
SUBOFF_CHANNELS = ("p", "ux", "uy", "uz")
_CHANNEL_UNITS = {"p": "lu", "ux": "lu", "uy": "lu", "uz": "lu"}


class SuboffPlatformPipeline:
    """SUBOFF full pipeline: register data -> train -> serve, with lineage."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self.catalog = FieldDataCatalog.open(self.db_path)
        self.training = TrainingJobRegistry.open(self.db_path)
        # serving registry shares the same SQLite ledger
        from tensorlbm.ml.serving import ModelRegistry

        self.serving = ModelRegistry.open(self.db_path)

    def close(self) -> None:
        self.catalog.close()
        self.training.close()
        self.serving.close()

    # ------------------------------------------------------------------
    # 1. Data management
    # ------------------------------------------------------------------

    def register_field_data(
        self,
        data_dir: str | Path,
        *,
        name_prefix: str = "suboff",
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Scan ``data_dir/<channel>/*.npy`` and register each channel as a
        field-product asset, run finiteness quality checks, and register a
        dataset asset with ``field -> dataset`` lineage edges.

        Returns ``{"asset_ids": [...], "dataset_asset_id": ..., "dataset_id": ...}``.
        """
        data_dir = Path(data_dir)
        asset_ids: list[str] = []
        shapes: dict[str, tuple[int, ...]] = {}

        for channel in SUBOFF_CHANNELS:
            ch_dir = data_dir / channel
            if not ch_dir.is_dir():
                continue
            npy_files = sorted(f for f in os.listdir(ch_dir) if f.endswith(".npy"))
            if not npy_files:
                continue
            sample = np.load(ch_dir / npy_files[0])
            shapes[channel] = tuple(sample.shape)
            asset_id = f"{name_prefix}:{channel}"
            self.catalog.register_asset(
                AssetRecord(
                    asset_id=asset_id,
                    name=f"SUBOFF {channel}",
                    kind="field_product",
                    field_name=channel,
                    units=_CHANNEL_UNITS.get(channel, "lu"),
                    shape=str(sample.shape),
                    dtype=str(sample.dtype),
                    source_run_id=run_id,
                    tags=(name_prefix, "suboff"),
                )
            )
            finite = bool(np.isfinite(sample).all())
            self.catalog.record_quality(
                asset_id,
                [
                    QualityCheck(
                        "finiteness",
                        finite,
                        f"{channel}: {sample.shape} npy snapshots={len(npy_files)}",
                    ),
                ],
            )
            asset_ids.append(asset_id)

        if not asset_ids:
            raise ValueError(f"no SUBOFF channel npy found under {data_dir}")

        dataset_asset_id = f"{name_prefix}:dataset"
        self.catalog.register_asset(
            AssetRecord(
                asset_id=dataset_asset_id,
                name="SUBOFF dataset",
                kind="dataset",
                description=f"channels={list(shapes)}",
                tags=(name_prefix, "suboff"),
            )
        )
        # lineage: each channel product -> dataset
        for asset_id in asset_ids:
            self.catalog.add_lineage(
                LineageRecord(
                    source_id=asset_id,
                    target_id=dataset_asset_id,
                    relation_type="derived_from",
                    resource_type="product",
                )
            )

        # dataset record in the shared ledger (int key for training jobs)
        from tensorlbm.ai.database import insert_dataset

        dataset_id = insert_dataset(
            self.training._conn,
            name=f"{name_prefix}-dataset",
            path=str(data_dir),
            n_samples=sum(
                len([f for f in os.listdir(data_dir / c) if f.endswith(".npy")]) for c in shapes
            ),
            metadata={"task": "suboff-surrogate", "channels": list(shapes)},
        )
        return {
            "asset_ids": asset_ids,
            "dataset_asset_id": dataset_asset_id,
            "dataset_id": dataset_id,
        }

    # ------------------------------------------------------------------
    # 2. Training
    # ------------------------------------------------------------------

    def run_training(
        self,
        train_cfg: Any,
        *,
        dataset_id: int | None = None,
        dataset_asset_id: str | None = None,
        product_asset_ids: list[str] | None = None,
        train_fn: Callable[[Any], dict[str, Any]] | None = None,
    ) -> tuple[TrainingJob, int]:
        """Run a SUBOFF training job on the platform.

        Wraps the training in a :class:`TrainingJobRegistry` state machine,
        records the metrics, registers the model, and writes the full lineage
        (``product -> dataset -> job``).

        ``train_fn`` defaults to :func:`tensorlbm.ai.suboff_train.train_suboff`.
        """
        if train_fn is None:
            from tensorlbm.ai.suboff_train import train_suboff

            train_fn = train_suboff

        config = (
            asdict(train_cfg)
            if hasattr(train_cfg, "__dataclass_fields__")
            else dict(train_cfg)
            if isinstance(train_cfg, dict)
            else {"cfg": str(train_cfg)}
        )
        job = self.training.create_job(config, dataset_id=dataset_id)
        self.training.update_status(job.job_id, "running")

        try:
            result = train_fn(train_cfg)
        except Exception as exc:  # noqa: BLE001 — mark job failed, re-raise
            self.training.update_status(job.job_id, "failed", error=str(exc))
            raise

        metrics = {
            "best_loss_1e4": result.get("best_loss_1e4"),
            "final_iter": result.get("final_iter"),
        }
        metrics = {k: v for k, v in metrics.items() if v is not None}
        self.training.record_metrics(job.job_id, metrics)

        checkpoint_dir = result.get("checkpoint_dir", "")
        model_id = self.training.register_model(
            job.job_id,
            name="suboff-surrogate",
            path=checkpoint_dir,
            arch={"family": "suboff_surrogate"},
            metrics=metrics,
        )

        # lineage: product -> dataset -> job
        if dataset_asset_id is not None:
            job_asset_id = f"job:{job.job_id}"
            self.catalog.register_asset(
                AssetRecord(
                    asset_id=job_asset_id,
                    name=f"training job {job.job_id}",
                    kind="run",
                )
            )
            for pid in product_asset_ids or []:
                self.catalog.add_lineage(
                    LineageRecord(
                        source_id=pid,
                        target_id=dataset_asset_id,
                        relation_type="derived_from",
                        resource_type="product",
                    )
                )
            self.catalog.add_lineage(
                LineageRecord(
                    source_id=dataset_asset_id,
                    target_id=job_asset_id,
                    relation_type="trained_on",
                    resource_type="dataset",
                )
            )

        self.training.update_status(job.job_id, "completed")
        return job, model_id

    # ------------------------------------------------------------------
    # 3. Serving
    # ------------------------------------------------------------------

    def run_inference(
        self,
        predict_cfg: Any,
        *,
        predict_fn: Callable[[Any], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Run SUBOFF inference (defaults to
        :func:`tensorlbm.ai.suboff_inference.predict_suboff`)."""
        if predict_fn is None:
            from tensorlbm.ai.suboff_inference import predict_suboff

            predict_fn = predict_suboff
        return predict_fn(predict_cfg)

    # ------------------------------------------------------------------
    # 4. End-to-end
    # ------------------------------------------------------------------

    def run_full_pipeline(
        self,
        data_dir: str | Path,
        train_cfg: Any,
        predict_cfg: Any | None = None,
        *,
        train_fn: Callable[[Any], dict[str, Any]] | None = None,
        predict_fn: Callable[[Any], dict[str, Any]] | None = None,
        name_prefix: str = "suboff",
    ) -> dict[str, Any]:
        """Register field data, train, and (optionally) infer — full flow."""
        reg = self.register_field_data(data_dir, name_prefix=name_prefix)
        job, model_id = self.run_training(
            train_cfg,
            dataset_id=reg["dataset_id"],
            dataset_asset_id=reg["dataset_asset_id"],
            product_asset_ids=reg["asset_ids"],
            train_fn=train_fn,
        )
        out: dict[str, Any] = {
            "registered": reg,
            "job_id": job.job_id,
            "model_id": model_id,
        }
        if predict_cfg is not None:
            out["inference"] = self.run_inference(predict_cfg, predict_fn=predict_fn)
        return out

    def upstream_assets(self, asset_id: str) -> list[str]:
        """Transitive upstream assets of a dataset/model/job asset."""
        return self.catalog.upstream(asset_id)
