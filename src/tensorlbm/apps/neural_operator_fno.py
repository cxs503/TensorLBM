"""Fourier Neural Operator (FNO2d) as an :class:`AI4SApplication` framework instance.

This module turns the pre-existing 2-D Fourier Neural Operator
(:mod:`tensorlbm.ai.fno`, following Li et al. 2021, arXiv:2010.08895) into a
concrete platform application.  Implementing the five framework methods
(``produce_data`` / ``build_model`` / ``make_dataset`` / ``train`` /
``infer``) is all that is required; the full-stack pipeline (catalog
registration, training-job lifecycle, model serving, lineage) is inherited
from :meth:`AI4SApplication.run`.

The learned operator maps a *coarse* velocity field (upsampled to the fine
grid) onto the corresponding *fine* velocity field — a flow-field
super-resolution surrogate that reuses the existing ``FNO2d`` /
``FNO2dArch`` / ``save_fno2d`` / ``load_fno2d`` building blocks without
re-implementing any spectral-convolution machinery.

The heavy pieces are injectable (``run_les_fn`` / ``train_fn``) so the whole
flow is testable without a real LBM run or a real training loop.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorlbm.ai.fno import FNO2d, FNO2dArch, save_fno2d
from tensorlbm.ai.pipeline import _coarsen_mean, _run_les_smoke
from tensorlbm.apps.base import (
    AI4SApplication,
    DataProduct,
    Prediction,
    TrainingResult,
)

__all__ = ["NeuralOperatorFNO"]


def _to_tensor(x: Any) -> torch.Tensor:
    """Coerce a field snapshot into a float32 ``torch.Tensor``."""
    if isinstance(x, torch.Tensor):
        return x.to(torch.float32)
    return torch.as_tensor(x, dtype=torch.float32)


class NeuralOperatorFNO(AI4SApplication):
    """2-D flow-field super-resolution Fourier Neural Operator as an app.

    Attributes:
        name: Registry name of the application.
        family: Serving model family (``"fno2d"`` is understood by
            :class:`tensorlbm.ml.serving.InferenceService`).
    """

    name: str = "neural_operator_fno"
    family: str = "fno2d"
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
                ``metrics`` / ``arch`` keys).  Defaults to the local Adam+MSE
                training loop.
            run_les_fn: Optional override for the data-production solver.  It
                is called with the same keyword arguments as ``_run_les_smoke``
                and must return a list of ``(ux, uy)`` velocity-field
                snapshots.  Defaults to ``_run_les_smoke``.
        """
        super().__init__()
        self._train_fn = train_fn
        self._run_les_fn = run_les_fn

    # ---- developer-implemented interface ---------------------------------

    def produce_data(self, cfg: Mapping[str, Any]) -> DataProduct:
        """Run a short LES smoke simulation and return 2-D velocity snapshots.

        The velocity fields are carried in
        :attr:`DataProduct.metadata["snapshots"]` (in-memory) so the downstream
        ``make_dataset`` step can build coarse→fine sample pairs without
        touching disk.
        """
        nx = int(cfg.get("nx", 32))
        ny = int(cfg.get("ny", 32))
        tau = float(cfg.get("tau", 0.8))
        c_s = float(cfg.get("c_s", 0.1))
        n_steps = int(cfg.get("n_steps", 16))
        sample_every = int(cfg.get("sample_every", 4))
        seed = int(cfg.get("seed", 0))
        device = torch.device(str(cfg.get("device", "cpu")))
        factor = max(1, int(cfg.get("downsample_factor", 2)))

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
            name="FNO 2D velocity snapshots",
            field_name="u",
            shape=tuple(_to_tensor(ux0).shape),
            dtype=str(_to_tensor(ux0).dtype),
            units="lu",
            metadata={
                "snapshots": snapshots,
                "n_snapshots": len(snapshots),
                "downsample_factor": factor,
            },
        )

    def build_model(self, arch: Mapping[str, Any]) -> torch.nn.Module:
        """Construct the :class:`FNO2d` operator from an arch mapping."""
        if isinstance(arch, FNO2dArch):
            arch_cfg = arch
        else:
            arch_cfg = FNO2dArch(
                in_channels=int(arch.get("in_channels", 2)),
                out_channels=int(arch.get("out_channels", 2)),
                width=int(arch.get("width", 16)),
                n_layers=int(arch.get("n_layers", 2)),
                modes_x=int(arch.get("modes_x", 8)),
                modes_y=int(arch.get("modes_y", 8)),
                mlp_hidden=int(arch.get("mlp_hidden", 32)),
                activation=str(arch.get("activation", "gelu")),
            )
        return FNO2d(arch_cfg)

    def make_dataset(self, product: DataProduct) -> dict[str, Any]:
        """Build coarse→fine (super-resolution) sample pairs from snapshots.

        Each snapshot yields one sample: the input is the coarse velocity field
        upsampled back to the fine grid, and the target is the original fine
        velocity field.  Returns a light dict
        ``{"inputs": (N, 2, ny, nx), "targets": (N, 2, ny, nx), ...}``.
        """
        snapshots = product.metadata.get("snapshots")
        if not snapshots:
            raise ValueError(
                "DataProduct metadata must carry 'snapshots' (list of (ux, uy))",
            )
        factor = max(1, int(product.metadata.get("downsample_factor", 2)))

        ux0 = _to_tensor(snapshots[0][0])
        grid = (int(ux0.shape[0]), int(ux0.shape[1]))

        inputs: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        for ux, uy in snapshots:
            ux = _to_tensor(ux)
            uy = _to_tensor(uy)
            ny, nx = ux.shape
            fine = torch.stack([ux, uy], dim=0).unsqueeze(0)  # (1, 2, ny, nx)
            coarse_ux = _coarsen_mean(ux, factor)
            coarse_uy = _coarsen_mean(uy, factor)
            coarse = torch.stack([coarse_ux, coarse_uy], dim=0).unsqueeze(0)  # (1, 2, cy, cx)
            up = F.interpolate(
                coarse, size=(ny, nx), mode="bilinear", align_corners=False,
            )
            inputs.append(up[0])  # (2, ny, nx)
            targets.append(fine[0])  # (2, ny, nx)

        return {
            "inputs": torch.stack(inputs),
            "targets": torch.stack(targets),
            "grid": grid,
            "downsample_factor": factor,
            "n_samples": len(inputs),
        }

    def train(
        self,
        dataset: Any,
        model: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> TrainingResult:
        """Train the FNO operator and return weights path + metrics.

        The injectable ``train_fn`` (if provided) is called as
        ``train_fn(dataset, model, cfg)``; otherwise the local Adam+MSE loop
        runs and the checkpoint is written with :func:`tensorlbm.ai.fno.save_fno2d`.
        """
        out_path = Path(
            str(cfg.get("out_path") or cfg.get("model_path") or "fno2d_model.pt"),
        )

        if self._train_fn is not None:
            return _coerce_training_result(
                self._train_fn(dataset, model, cfg),
                out_path=out_path,
            )

        result = _train_fno2d(
            dataset,
            cast(FNO2d, model),
            out_path,
            epochs=int(cfg.get("epochs", 20)),
            batch_size=int(cfg.get("batch_size", 16)),
            learning_rate=float(cfg.get("learning_rate", 1e-3)),
            seed=int(cfg.get("seed", 0)),
            device=str(cfg.get("device", "cpu")),
        )
        arch = dict(result.get("arch") or {})
        if not arch:
            model_arch = getattr(model, "arch", None)
            if model_arch is not None:
                arch = asdict(model_arch)
        return TrainingResult(
            model_path=str(result.get("path", out_path)),
            metrics={"train_loss": float(result.get("train_loss", float("nan")))},
            arch=arch,
        )

    def infer(self, model: torch.nn.Module, sample: Any) -> Prediction:
        """Map a coarse velocity field through the trained operator.

        ``sample`` may be a ``(ux, uy)`` tuple of 2-D tensors or a single
        ``(C, ny, nx)`` input tensor.
        """
        if isinstance(sample, (tuple, list)):
            ux, uy = sample
            x = torch.stack([_to_tensor(ux), _to_tensor(uy)], dim=0)  # (2, ny, nx)
        else:
            x = _to_tensor(sample)
        model.eval()
        with torch.no_grad():
            out = model(x.unsqueeze(0)).squeeze(0)  # (out_channels, ny, nx)
        return Prediction(
            output=out,
            metadata={
                "field_name": "u",
                "shape": tuple(out.shape),
                "units": "lu",
            },
        )


# ---------------------------------------------------------------------------
# Default training loop
# ---------------------------------------------------------------------------

def _train_fno2d(
    dataset: Mapping[str, Any],
    model: FNO2d,
    out_path: Path,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: str,
) -> dict[str, Any]:
    """Adam + MSE training loop (CPU-friendly) and checkpoint save."""
    X = dataset["inputs"]
    Y = dataset["targets"]
    torch_device = torch.device(device)
    torch.manual_seed(int(seed))
    model.to(torch_device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    loss_fn = nn.MSELoss()
    n = int(X.shape[0])
    batch_size = max(1, int(batch_size))

    final_loss = float("nan")
    for _ in range(max(0, int(epochs))):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb = X[idx].to(torch_device)
            yb = Y[idx].to(torch_device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        final_loss = epoch_loss / n

    model.eval()
    path = save_fno2d(model, out_path)
    return {
        "path": str(path),
        "train_loss": final_loss,
        "arch": asdict(model.arch),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
            metrics={"train_loss": float(result.get("train_loss", result.get("final_train_loss", float("nan"))))},
            arch=dict(result.get("arch") or {}),
        )
    raise TypeError(
        "train_fn must return a TrainingResult or a mapping with "
        f"'model_path'/'metrics'/'arch' keys, got {type(result).__name__}",
    )
