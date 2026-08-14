"""AI4S application development framework — the platform's extension SDK.

A single abstract base class, :class:`AI4SApplication`, turns "a few hardcoded
case scripts" into "an integrated development platform": a developer implements
five methods (produce data / build model / make dataset / train / infer) and
inherits the full-stack pipeline — HPC data production, data-catalog
registration, training-job lifecycle, model serving, and lineage — for free.

See ``docs/plans/ai4s-integrated-platform-architecture.md`` for the platform
architecture this SDK sits at the top of.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch

from tensorlbm.data.catalog import AssetRecord, FieldDataCatalog, LineageRecord


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DataProduct:
    """Metadata of one HPC-produced field product (the data itself stays on
    disk / in the solver output)."""

    name: str
    field_name: str
    shape: tuple[int, ...]
    dtype: str
    units: str = "lu"
    path: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Outcome of a training run: weights path + metrics + architecture."""

    model_path: str
    metrics: Mapping[str, Any]
    arch: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Prediction:
    """Inference result (tensor/array plus optional metadata)."""

    output: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RunReport:
    """Full-stack run summary with platform identifiers and lineage."""

    name: str
    family: str
    data_asset_id: str
    dataset_asset_id: str
    job_id: str
    model_id: int
    metrics: Mapping[str, Any]
    lineage_upstream: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# The application base class
# ---------------------------------------------------------------------------

class AI4SApplication(ABC):
    """Base class for every AI4S application on the platform.

    Subclasses implement the five abstract methods; :meth:`run` then wires them
    through the platform layer (catalog -> training-job -> serving) with full
    lineage, automatically.
    """

    name: str = "unnamed"
    family: str = "generic"
    version: str = "1.0"

    # ---- developer-implemented interface --------------------------------

    @abstractmethod
    def produce_data(self, cfg: Mapping[str, Any]) -> DataProduct:
        """Produce HPC field data (run the solver) and return its metadata."""

    @abstractmethod
    def build_model(self, arch: Mapping[str, Any]) -> torch.nn.Module:
        """Construct the model (neural operator / PINN / GNN / ...)."""

    @abstractmethod
    def make_dataset(self, product: DataProduct) -> Any:
        """Build a training dataset from a data product."""

    @abstractmethod
    def train(
        self,
        dataset: Any,
        model: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> TrainingResult:
        """Run the training loop; return weights path + metrics."""

    @abstractmethod
    def infer(self, model: torch.nn.Module, sample: Any) -> Prediction:
        """Run inference on a sample."""

    # ---- framework-provided full-stack pipeline ------------------------

    def run(
        self,
        db_path: str | Path,
        produce_cfg: Mapping[str, Any],
        train_cfg: Mapping[str, Any],
        *,
        name_prefix: str | None = None,
        run_id: str | None = None,
    ) -> RunReport:
        """Full-stack run: produce -> register -> train -> serve -> lineage."""
        prefix = name_prefix or self.name
        catalog = FieldDataCatalog.open(db_path)
        training = _open_training(db_path)
        serving = _open_serving(db_path)
        try:
            # 1. data production
            product = self.produce_data(produce_cfg)

            # 2. register field product + quality
            data_asset_id = f"{prefix}:{product.field_name}"
            catalog.register_asset(AssetRecord(
                asset_id=data_asset_id,
                name=product.name,
                kind="field_product",
                field_name=product.field_name,
                units=product.units,
                shape=str(product.shape),
                dtype=product.dtype,
                source_run_id=run_id,
                tags=(prefix, self.name),
            ))
            from tensorlbm.data.catalog import QualityCheck
            catalog.record_quality(data_asset_id, [
                QualityCheck("registered", True, product.field_name),
            ])

            # 3. dataset + lineage (data -> dataset)
            dataset = self.make_dataset(product)
            dataset_asset_id = f"{prefix}:dataset"
            catalog.register_asset(AssetRecord(
                asset_id=dataset_asset_id,
                name=f"{self.name} dataset",
                kind="dataset",
                description=f"field={product.field_name}",
                tags=(prefix, self.name),
            ))
            catalog.add_lineage(LineageRecord(
                source_id=data_asset_id, target_id=dataset_asset_id,
                relation_type="derived_from", resource_type="product",
            ))

            # 4. training job (state machine)
            arch_cfg = dict(train_cfg.get("arch") or {})
            model = self.build_model(arch_cfg)
            job = training.create_job(
                dict(train_cfg), dataset_id=None,
            )
            training.update_status(job.job_id, "running")
            result = self.train(dataset, model, train_cfg)
            numeric_metrics = {
                k: v for k, v in result.metrics.items()
                if isinstance(v, (int, float))
            }
            if numeric_metrics:
                training.record_metrics(job.job_id, numeric_metrics)
            training.update_status(job.job_id, "completed")

            # 5. model serving + lineage (dataset -> job)
            model_id = serving.register_model(
                name=f"{prefix}-{self.name}",
                path=result.model_path,
                arch=dict(result.arch),
                metrics=numeric_metrics,
                family=self.family,
            )
            job_asset_id = f"{prefix}:job:{job.job_id}"
            catalog.register_asset(AssetRecord(
                asset_id=job_asset_id, name=f"training job {job.job_id}",
                kind="run",
            ))
            catalog.add_lineage(LineageRecord(
                source_id=dataset_asset_id, target_id=job_asset_id,
                relation_type="trained_on", resource_type="dataset",
            ))

            return RunReport(
                name=self.name, family=self.family,
                data_asset_id=data_asset_id, dataset_asset_id=dataset_asset_id,
                job_id=job.job_id, model_id=model_id,
                metrics=numeric_metrics,
                lineage_upstream=tuple(catalog.upstream(job_asset_id)),
            )
        finally:
            catalog.close()
            training.close()
            serving.close()


# ---------------------------------------------------------------------------
# Helpers (lazy imports so the SDK stays light)
# ---------------------------------------------------------------------------

def _open_training(db_path: str | Path):
    from tensorlbm.ml.training_job import TrainingJobRegistry
    return TrainingJobRegistry.open(db_path)


def _open_serving(db_path: str | Path):
    from tensorlbm.ml.serving import ModelRegistry
    return ModelRegistry.open(db_path)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ApplicationRegistry:
    """Discovery registry for AI4S applications (name -> class)."""

    def __init__(self) -> None:
        self._apps: dict[str, type[AI4SApplication]] = {}

    def register(self, cls: type[AI4SApplication]) -> type[AI4SApplication]:
        if not issubclass(cls, AI4SApplication):
            raise TypeError(f"{cls!r} is not an AI4SApplication")
        self._apps[cls.name] = cls
        return cls

    def get(self, name: str) -> type[AI4SApplication]:
        try:
            return self._apps[name]
        except KeyError:
            raise KeyError(f"unknown AI4S application {name!r}") from None

    def names(self) -> list[str]:
        return sorted(self._apps)

    def __len__(self) -> int:
        return len(self._apps)


# a process-wide default registry
registry = ApplicationRegistry()
