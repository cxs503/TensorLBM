"""SUBOFF surrogate case as an :class:`AI4SApplication` framework instance.

Wraps the pre-existing SUBOFF reconstruction case
(``tensorlbm.ai.suboff_train`` / ``tensorlbm.ai.suboff_inference`` /
``tensorlbm.ai.suboff_utils``) behind the five-method application contract so
it inherits the platform's full-stack pipeline (HPC data production -> catalog
registration -> training job -> model serving -> lineage) from
:class:`tensorlbm.apps.base.AI4SApplication`.

The heavy pieces are injectable (``train_fn`` / ``predict_fn`` / ``build_fn``)
so the whole flow is testable without a real LBM run or a real training loop.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any, Callable, Mapping

import numpy as np
import torch

from tensorlbm.apps.base import AI4SApplication, DataProduct, Prediction, TrainingResult

# SUBOFF LBM snapshot channels (data_dir/<channel>/*.npy)
SUBOFF_CHANNELS = ("p", "ux", "uy", "uz")
_CHANNEL_UNITS = {"p": "lu", "ux": "lu", "uy": "lu", "uz": "lu"}


def _to_config_fields(cls: type, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Project a plain mapping onto a frozen dataclass's declared fields."""
    if isinstance(cfg, cls):
        return cfg  # type: ignore[return-value]
    valid = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in dict(cfg).items() if k in valid}


class SuboffSurrogateApp(AI4SApplication):
    """SUBOFF 3D flow-field reconstruction surrogate, expressed as an app."""

    name = "suboff_surrogate"
    family = "suboff_surrogate"
    version = "1.0"

    def __init__(
        self,
        train_fn: Callable[[Any], dict[str, Any]] | None = None,
        predict_fn: Callable[[Any], dict[str, Any]] | None = None,
        build_fn: Callable[[Mapping[str, Any]], Any] | None = None,
    ) -> None:
        """Optionally inject the heavy callables (defaults are the real ones).

        Args:
            train_fn: Signature ``(SuboffTrainConfig) -> dict``; defaults to
                :func:`tensorlbm.ai.suboff_train.train_suboff`.
            predict_fn: Signature ``(SuboffPredictConfig) -> dict``; defaults to
                :func:`tensorlbm.ai.suboff_inference.predict_suboff`.
            build_fn: Signature ``(arch mapping) -> model``; defaults to
                :func:`tensorlbm.ai.suboff_utils.build_suboff_model`.
        """
        self._train_fn = train_fn
        self._predict_fn = predict_fn
        self._build_fn = build_fn

    # ------------------------------------------------------------------
    # 1. HPC data production
    # ------------------------------------------------------------------

    def produce_data(self, cfg: Mapping[str, Any]) -> DataProduct:
        """Scan ``cfg["data_dir"]/p,ux,uy,uz/*.npy`` and return the product.

        The four channels are a single logical *flow-field* product; per-channel
        shapes and snapshot counts are recorded in the product metadata.
        """
        data_dir = cfg.get("data_dir") or cfg.get("path") or ""
        if not data_dir:
            raise ValueError("produce_data requires cfg['data_dir']")
        data_dir = str(data_dir)

        shapes: dict[str, tuple[int, ...]] = {}
        n_snapshots: int | None = None
        for channel in SUBOFF_CHANNELS:
            ch_dir = os.path.join(data_dir, channel)
            if not os.path.isdir(ch_dir):
                continue
            files = sorted(f for f in os.listdir(ch_dir) if f.endswith(".npy"))
            if not files:
                continue
            sample = np.load(os.path.join(ch_dir, files[0]))
            shapes[channel] = tuple(sample.shape)
            n_snapshots = len(files) if n_snapshots is None else min(n_snapshots, len(files))

        if not shapes:
            raise ValueError(f"no SUBOFF channel npy found under {data_dir}")

        field_name = str(cfg.get("field_name", "flow_field"))
        # representative shape: prefer the pressure channel, else first found
        rep_shape = shapes.get("p") or next(iter(shapes.values()))
        return DataProduct(
            name=f"SUBOFF {field_name}",
            field_name=field_name,
            shape=rep_shape,
            dtype="float32",
            units=str(cfg.get("units", "lu")),
            path=data_dir,
            metadata={
                "channels": list(shapes),
                "shapes": shapes,
                "n_snapshots": n_snapshots or 0,
            },
        )

    # ------------------------------------------------------------------
    # 2. Model construction
    # ------------------------------------------------------------------

    def build_model(self, arch: Mapping[str, Any]) -> Any:
        """Build the SUBOFF encoder-decoder reconstruction model.

        Returns ``(encoder, decoder)`` from
        :func:`tensorlbm.ai.suboff_utils.build_suboff_model`.
        """
        if self._build_fn is not None:
            return self._build_fn(dict(arch))
        from tensorlbm.ai.suboff_utils import build_suboff_model

        device = arch.get("device") if isinstance(arch, Mapping) else None
        return build_suboff_model(device)

    # ------------------------------------------------------------------
    # 3. Dataset construction
    # ------------------------------------------------------------------

    def make_dataset(self, product: DataProduct) -> Any:
        """Read the channel NPY snapshots under ``product.path`` into one tensor.

        Returns a dict ``{"data": [n_snap, n_points, 4], "channels": [...],
        "n_snapshots": int, "data_dir": str}`` — a light, shape-agnostic
        stand-in for the full ``CylinderDatasetMultiRe14`` used by the real
        training loop (which re-reads the raw NPY files itself).
        """
        data_dir = product.path
        channels = tuple(product.metadata.get("channels", SUBOFF_CHANNELS))
        max_snaps = int(product.metadata.get("n_snapshots") or 1)

        channel_arrays: list[np.ndarray] = []
        for channel in channels:
            ch_dir = os.path.join(data_dir, channel)
            files = sorted(f for f in os.listdir(ch_dir) if f.endswith(".npy"))
            if not files:
                raise ValueError(f"no npy snapshots under {ch_dir}")
            arrs = [
                np.load(os.path.join(ch_dir, f)).astype(np.float32).reshape(-1)
                for f in files[:max_snaps]
            ]
            channel_arrays.append(np.stack(arrs))  # [n_snap, n_points]

        data = torch.from_numpy(np.stack(channel_arrays, axis=-1))  # [n_snap, n_points, 4]
        return {
            "data": data,
            "channels": list(channels),
            "n_snapshots": int(data.shape[0]),
            "data_dir": data_dir,
        }

    # ------------------------------------------------------------------
    # 4. Training
    # ------------------------------------------------------------------

    def train(
        self,
        dataset: Any,
        model: Any,
        cfg: Mapping[str, Any],
    ) -> TrainingResult:
        """Convert ``cfg`` to a ``SuboffTrainConfig`` and run the training loop.

        ``model`` and ``dataset`` are accepted for contract compatibility; the
        default training function re-builds its own model and re-reads data
        from ``cfg.data_dir`` (the injected ``train_fn`` may use them instead).
        """
        from tensorlbm.ai.suboff_train import SuboffTrainConfig, train_suboff

        config = _to_config_fields(SuboffTrainConfig, cfg)
        if not isinstance(config, SuboffTrainConfig):
            config = SuboffTrainConfig(**config)

        train_fn = self._train_fn or train_suboff
        result = train_fn(config)

        metrics = {
            "best_loss_1e4": result.get("best_loss_1e4"),
            "final_iter": result.get("final_iter"),
        }
        metrics = {k: float(v) for k, v in metrics.items() if v is not None}
        return TrainingResult(
            model_path=str(result.get("checkpoint_dir", "")),
            metrics=metrics,
            arch={"family": self.family, "name": self.name},
        )

    # ------------------------------------------------------------------
    # 5. Inference
    # ------------------------------------------------------------------

    def infer(self, model: Any, sample: Any) -> Prediction:
        """Wrap :func:`tensorlbm.ai.suboff_inference.predict_suboff` on ``sample``.

        ``sample`` may be a ``SuboffPredictConfig`` or a mapping of its fields;
        ``model`` is accepted for contract compatibility (the default predictor
        re-loads the checkpoint itself).
        """
        from tensorlbm.ai.suboff_inference import SuboffPredictConfig, predict_suboff

        cfg = (
            _to_config_fields(SuboffPredictConfig, sample)
            if isinstance(sample, Mapping)
            else sample
        )
        if not isinstance(cfg, SuboffPredictConfig):
            cfg = SuboffPredictConfig(**cfg)  # type: ignore[arg-type]

        predict_fn = self._predict_fn or predict_suboff
        result = predict_fn(cfg)

        output = result.get("pred")
        metadata = {k: v for k, v in result.items() if k != "pred"}
        return Prediction(output=output, metadata=metadata)


__all__ = ["SUBOFF_CHANNELS", "SuboffSurrogateApp"]
