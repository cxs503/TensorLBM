"""AI-LES (AI eddy-viscosity turbulence model) as an AI4S framework instance.

This module turns the flagship AI-LES demonstration — HPC LBM solving →
strain-rate sampling → MLP eddy-viscosity training → AI-enhanced inference —
into a concrete :class:`~tensorlbm.apps.base.AI4SApplication`.  Implementing
the five framework methods (``produce_data`` / ``build_model`` / ``make_dataset``
/ ``train`` / ``infer``) is all that is required; the full-stack pipeline
(catalog registration, training-job lifecycle, model serving, lineage) is
inherited from :meth:`AI4SApplication.run`.

The heavy lifting stays in the existing ``tensorlbm.ai`` sub-package:

* data production  : ``tensorlbm.ai.pipeline._init_random_velocity_field`` +
  ``tensorlbm.ai.pipeline._run_les_smoke`` (a small 2-D LES smoke run that
  harvests velocity snapshots);
* dataset          : ``tensorlbm.ai.dataset.extract_les_samples_2d`` +
  ``EddyViscosityDataset`` (strain-rate features → Smagorinsky ν_t target);
* model            : ``tensorlbm.ai.model.EddyViscosityMLP``;
* training         : ``tensorlbm.ai.train.train_eddy_viscosity_model``;
* inference        : ``tensorlbm.ai.inference.predict_nu_t_2d``.

Both the solver step and the training loop can be injected through the
constructor (``run_les_fn`` / ``train_fn``) so tests can exercise the full
:meth:`run` loop with mock data and no real compute.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from tensorlbm.apps.base import (
    AI4SApplication,
    DataProduct,
    Prediction,
    TrainingResult,
)
from tensorlbm.ai.dataset import EddyViscosityDataset, extract_les_samples_2d
from tensorlbm.ai.inference import predict_nu_t_2d
from tensorlbm.ai.model import EddyViscosityMLP, ModelArch
from tensorlbm.ai.pipeline import _run_les_smoke
from tensorlbm.ai.train import TrainConfig, train_eddy_viscosity_model

__all__ = ["AILesApp"]


# ---------------------------------------------------------------------------
# The application
# ---------------------------------------------------------------------------

class AILesApp(AI4SApplication):
    """AI-LES eddy-viscosity MLP as a platform application.

    Attributes:
        name: Registry name of the application.
        family: Serving model family (``"eddy_viscosity_mlp"`` is understood
            by :class:`tensorlbm.ml.serving.InferenceService`).
    """

    name: str = "ai_les"
    family: str = "eddy_viscosity_mlp"
    version: str = "1.0"

    def __init__(
        self,
        *,
        train_fn: Callable[..., Any] | None = None,
        run_les_fn: Callable[..., Any] | None = None,
    ) -> None:
        """Create the app, optionally injecting the solver / training steps.

        Args:
            train_fn: Optional override for the training step.  It is called as
                ``train_fn(dataset, model, cfg)`` and must return a
                :class:`TrainingResult` (or a mapping with ``model_path`` /
                ``metrics`` / ``arch`` keys).  Defaults to
                :func:`tensorlbm.ai.train.train_eddy_viscosity_model`.
            run_les_fn: Optional override for the data-production solver.  It
                is called with the same keyword arguments as
                ``_run_les_smoke`` and must return a list of ``(ux, uy)``
                velocity-field snapshots.  Defaults to ``_run_les_smoke``
                (which initialises the field via ``_init_random_velocity_field``).
        """
        super().__init__()
        self._train_fn = train_fn
        self._run_les_fn = run_les_fn

    # ---- developer-implemented interface ---------------------------------

    def produce_data(self, cfg: Mapping[str, Any]) -> DataProduct:
        """Run a short LES smoke simulation and return 2-D velocity snapshots.

        The actual velocity fields are carried in
        :attr:`DataProduct.metadata["snapshots"]` (in-memory) so the downstream
        ``make_dataset`` step can extract strain-rate samples without touching
        disk.
        """
        nx = int(cfg.get("nx", 64))
        ny = int(cfg.get("ny", 64))
        tau = float(cfg.get("tau", 0.8))
        c_s = float(cfg.get("c_s", 0.1))
        n_steps = int(cfg.get("n_steps", 40))
        sample_every = int(cfg.get("sample_every", 10))
        seed = int(cfg.get("seed", 0))
        device = torch.device(str(cfg.get("device", "cpu")))

        run_les = self._run_les_fn or _run_les_smoke
        snapshots = run_les(
            nx=nx,
            ny=ny,
            tau=tau,
            c_s=c_s,
            n_steps=n_steps,
            sample_every=sample_every,
            seed=seed,
            device=device,
        )
        if not snapshots:
            raise ValueError("LES smoke run produced no velocity snapshots")

        ux0, uy0 = snapshots[0]
        return DataProduct(
            name="AI-LES velocity snapshots",
            field_name="velocity",
            shape=tuple(ux0.shape),
            dtype=str(ux0.dtype),
            units="lu",
            metadata={
                "snapshots": snapshots,
                "c_s": c_s,
                "n_snapshots": len(snapshots),
            },
        )

    def build_model(self, arch: Mapping[str, Any]) -> torch.nn.Module:
        """Construct the MLP eddy-viscosity model from an arch mapping."""
        model_arch = ModelArch(
            in_features=int(arch.get("in_features", 3)),
            hidden_features=int(arch.get("hidden_features", 16)),
            n_hidden_layers=int(arch.get("n_hidden_layers", 2)),
            activation=str(arch.get("activation", "tanh")),
        )
        return EddyViscosityMLP(model_arch)

    def make_dataset(self, product: DataProduct) -> Any:
        """Extract a strain-rate → ν_t regression dataset from a product."""
        snapshots = product.metadata.get("snapshots")
        if not snapshots:
            raise ValueError(
                "DataProduct metadata must carry 'snapshots' (list of (ux, uy))",
            )
        c_s = float(product.metadata.get("c_s", 0.1))

        feats_list: list[torch.Tensor] = []
        targs_list: list[torch.Tensor] = []
        for ux, uy in snapshots:
            feats, targs = extract_les_samples_2d(ux, uy, c_s=c_s)
            feats_list.append(feats)
            targs_list.append(targs)
        features = torch.cat(feats_list, dim=0)
        targets = torch.cat(targs_list, dim=0)
        return EddyViscosityDataset(
            features=features,
            targets=targets,
            c_s=c_s,
            description=(
                f"strain-rate samples from {product.name} "
                f"({len(snapshots)} frames)"
            ),
        )

    def train(
        self,
        dataset: Any,
        model: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> TrainingResult:
        """Train the eddy-viscosity MLP and return weights path + metrics.

        The injectable ``train_fn`` (if provided) is called as
        ``train_fn(dataset, model, cfg)``; otherwise the default
        ``train_eddy_viscosity_model`` runs with a :class:`TrainConfig`
        assembled from ``cfg``.
        """
        out_path = Path(str(cfg.get("out_path") or cfg.get("model_path") or "ai_les_model.pt"))

        if self._train_fn is not None:
            return _coerce_training_result(
                self._train_fn(dataset, model, cfg),
                out_path=out_path,
            )

        train_cfg = _train_config_from(cfg)
        raw = train_eddy_viscosity_model(dataset, out_path, train_cfg)
        return TrainingResult(
            model_path=str(raw.get("path", out_path)),
            metrics={"final_train_mse": raw.get("final_train_mse", float("nan"))},
            arch=dict(raw.get("arch") or {}),
        )

    def infer(self, model: torch.nn.Module, sample: Any) -> Prediction:
        """Predict the eddy-viscosity field for a ``(ux, uy)`` snapshot."""
        ux, uy = sample
        nu_t = predict_nu_t_2d(model, ux, uy)
        return Prediction(
            output=nu_t,
            metadata={
                "field_name": "nu_t",
                "shape": tuple(nu_t.shape),
                "units": "lu",
            },
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _train_config_from(cfg: Mapping[str, Any]) -> TrainConfig:
    """Assemble a :class:`TrainConfig` from a run's training configuration.

    Architecture keys may live either at the top level of ``cfg`` or inside its
    ``"arch"`` sub-mapping (mirroring how :meth:`AI4SApplication.run` splits the
    two).
    """
    arch = dict(cfg.get("arch") or {})

    def _get(*keys: str, default: Any) -> Any:
        for key in keys:
            if key in cfg and cfg[key] is not None:
                return cfg[key]
            if key in arch and arch[key] is not None:
                return arch[key]
        return default

    return TrainConfig(
        epochs=int(_get("epochs", default=20)),
        batch_size=int(_get("batch_size", default=4096)),
        learning_rate=float(_get("learning_rate", default=1e-3)),
        val_fraction=float(_get("val_fraction", default=0.1)),
        seed=int(_get("seed", default=0)),
        hidden_features=int(_get("hidden_features", default=16)),
        n_hidden_layers=int(_get("n_hidden_layers", default=2)),
        activation=str(_get("activation", default="tanh")),
        device=str(_get("device", default="cpu")),
        lr_scheduler=str(_get("lr_scheduler", default="none")),
        patience=_get("patience", default=None),
        gradient_clip_norm=_get("gradient_clip_norm", default=1.0),
    )


def _coerce_training_result(
    result: Any,
    *,
    out_path: Path,
) -> TrainingResult:
    """Normalise a ``train_fn`` return value into a :class:`TrainingResult`."""
    if isinstance(result, TrainingResult):
        return result
    if isinstance(result, Mapping):
        return TrainingResult(
            model_path=str(result.get("model_path") or result.get("path") or out_path),
            metrics={"final_train_mse": result.get("final_train_mse", float("nan"))},
            arch=dict(result.get("arch") or {}),
        )
    raise TypeError(
        "train_fn must return a TrainingResult or a mapping with "
        f"'model_path'/'metrics'/'arch' keys, got {type(result).__name__}",
    )
