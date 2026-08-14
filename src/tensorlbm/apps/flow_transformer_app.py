"""Flow Transformer as an :class:`AI4SApplication` framework instance.

Wraps the self-supervised masked-reconstruction flow transformer
(:mod:`tensorlbm.ai.transformer`) behind the five-method AI4S SDK contract
(``produce_data`` / ``build_model`` / ``make_dataset`` / ``train`` /
``infer``).  Once those five methods are implemented, the inherited
:meth:`AI4SApplication.run` wires the case through the full platform stack —
HPC data production, data-catalog registration, training-job lifecycle,
model serving, and lineage — automatically.

This module is the *framework* view of what
:class:`tensorlbm.ai.flow_transformer_platform_pipeline.FlowTransformerPlatformPipeline`
does as a bespoke orchestration script: the same behaviour, but expressed as a
reusable application class registered with the platform's
:class:`~tensorlbm.apps.base.ApplicationRegistry`.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Mapping, cast

import torch

from tensorlbm.ai.transformer import (
    FlowFieldTransformer,
    FlowTransformerArch,
    FlowTransformerTrainConfig,
    build_flow_token_batch,
    reconstruct_flow_field,
    train_flow_transformer_self_supervised,
)
from tensorlbm.apps.base import AI4SApplication, DataProduct, Prediction, TrainingResult


def _filter_fields(mapping: Mapping[str, Any], cls: type) -> dict[str, Any]:
    """Keep only keys that are valid constructor fields of ``cls``."""
    names = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in mapping.items() if k in names}


class FlowTransformerApp(AI4SApplication):
    """Self-supervised flow-field reconstruction transformer as an AI4S app."""

    name: str = "flow_transformer"
    family: str = "flow_transformer_ssl"
    version: str = "1.0"

    def __init__(self, *, train_fn: Callable[..., dict[str, Any]] | None = None) -> None:
        # Injectable training function so tests can substitute a mock and so
        # callers can override the backend without touching the class.
        self._train_fn = train_fn or train_flow_transformer_self_supervised

    # ------------------------------------------------------------------
    # Developer-implemented interface
    # ------------------------------------------------------------------

    def produce_data(self, cfg: Mapping[str, Any]) -> DataProduct:
        """Return the metadata for a batch of ``(ux, uy)`` flow snapshots.

        ``cfg["snapshots"]`` may supply in-memory snapshots directly
        (``list[tuple[Tensor, Tensor]]``); otherwise random snapshots are
        generated from ``cfg`` (``n_snapshots``, ``grid``, ``seed``).
        """
        snapshots = cfg.get("snapshots")
        if snapshots is None:
            snapshots = self._random_snapshots(cfg)
        snapshots = [self._to_tensor_pair(pair) for pair in snapshots]
        if not snapshots:
            raise ValueError("at least one (ux, uy) snapshot is required")

        ux0, uy0 = snapshots[0]
        shape = tuple(int(d) for d in ux0.shape)
        return DataProduct(
            name="flow-field-velocity-snapshots",
            field_name="u",
            shape=shape,
            dtype=str(ux0.dtype),
            units="lu",
            path=str(cfg.get("data_path", "")),
            metadata={
                "snapshots": snapshots,
                "n_snapshots": len(snapshots),
            },
        )

    def build_model(self, arch: Mapping[str, Any]) -> torch.nn.Module:
        """Construct the flow transformer from architecture hyper-parameters."""
        if isinstance(arch, FlowTransformerArch):
            arch_cfg = arch
        else:
            arch_cfg = FlowTransformerArch(
                **_filter_fields(arch or {}, FlowTransformerArch),
            )
        return FlowFieldTransformer(arch_cfg)

    def make_dataset(self, product: DataProduct) -> dict[str, Any]:
        """Convert snapshots into a token batch ``(N, T, 2)`` plus grid shape."""
        snapshots = product.metadata["snapshots"]
        batch, grid = build_flow_token_batch(snapshots)
        return {"snapshots": snapshots, "batch": batch, "grid": grid}

    def train(
        self,
        dataset: Any,
        model: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> TrainingResult:
        """Run the self-supervised training loop and return weights + metrics."""
        snapshots = dataset["snapshots"]
        out_path = str(cfg.get("out_path") or f"{self.name}_model.pt")
        arch = getattr(model, "arch", None) or FlowTransformerArch(
            **_filter_fields(cfg.get("arch") or {}, FlowTransformerArch),
        )
        train_cfg = FlowTransformerTrainConfig(
            **_filter_fields(cfg, FlowTransformerTrainConfig),
        )

        result = cast(
            dict[str, Any],
            self._train_fn(snapshots, out_path, arch=arch, config=train_cfg),
        )

        return TrainingResult(
            model_path=str(result.get("path", out_path)),
            metrics={
                "final_train_loss": float(result.get("final_train_loss", float("nan"))),
                "final_val_loss": float(result.get("final_val_loss", float("nan"))),
            },
            arch=dict(result.get("arch") or dataclasses.asdict(arch)),
        )

    def infer(self, model: torch.nn.Module, sample: Any) -> Prediction:
        """Reconstruct a ``(ux, uy)`` sample through the trained model."""
        ux, uy = sample
        result = reconstruct_flow_field(model, ux, uy)
        return Prediction(
            output=(result["ux_reconstructed"], result["uy_reconstructed"]),
            metadata={
                "mse": float(result["mse"]),
                "max_abs_error": float(result["max_abs_error"]),
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _random_snapshots(self, cfg: Mapping[str, Any]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        n = int(cfg.get("n_snapshots", 8))
        grid = cfg.get("grid", (8, 8))
        if isinstance(grid, int):
            ny = nx = int(grid)
        else:
            ny, nx = (int(grid[0]), int(grid[1]))
        seed = int(cfg.get("seed", 0))
        g = torch.Generator().manual_seed(seed)
        return [
            (torch.rand((ny, nx), generator=g), torch.rand((ny, nx), generator=g))
            for _ in range(n)
        ]

    @staticmethod
    def _to_tensor_pair(pair: Any) -> tuple[torch.Tensor, torch.Tensor]:
        ux, uy = pair
        ux = ux if isinstance(ux, torch.Tensor) else torch.as_tensor(ux, dtype=torch.float32)
        uy = uy if isinstance(uy, torch.Tensor) else torch.as_tensor(uy, dtype=torch.float32)
        return ux, uy
