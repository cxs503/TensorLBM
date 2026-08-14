"""Monte-Carlo dropout uncertainty quantification as an :class:`AI4SApplication`.

This module implements the framework's *uncertainty quantification* (UQ)
application: an MLP surrogate (here an eddy-viscosity / strain-rate proxy
trained on 2-D LBM velocity snapshots) whose dropout layers are kept active at
inference time.  Running ``N`` stochastic forward passes (Monte-Carlo dropout,
following Gal & Ghahramani 2016, "Dropout as a Bayesian Approximation:
Representing Model Uncertainty in Deep Learning", arXiv:1506.02142) turns the
point estimate into a predictive distribution, from which the mean prediction
and its per-point standard deviation (a measure of epistemic uncertainty) are
reported.

The implementation is clean-room and independent: the MLP and the sampling
loop are written from scratch, reusing only the existing framework building
blocks (:mod:`tensorlbm.ai.pipeline` for the LES smoke run and
:mod:`tensorlbm.ai.dataset` for strain-rate feature extraction).  The heavy
steps are injectable (``run_les_fn`` / ``train_fn``) so the full closed-loop
pipeline is testable without a real solver run or training loop.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, cast

import torch
import torch.nn as nn

from tensorlbm.ai.dataset import (
    extract_les_samples_2d,
    strain_rate_tensor_2d,
)
from tensorlbm.ai.pipeline import _run_les_smoke
from tensorlbm.apps.base import (
    AI4SApplication,
    DataProduct,
    Prediction,
    TrainingResult,
)

__all__ = [
    "MCDropoutMLP",
    "UncertaintyQuantification",
    "UQMLPArch",
    "load_uq_mlp",
    "save_uq_mlp",
]


# ---------------------------------------------------------------------------
# MLP surrogate architecture (dropout kept active for MC sampling)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UQMLPArch:
    """Hyper-parameters describing an :class:`MCDropoutMLP`."""

    in_features: int = 3  # S_xx, S_yy, S_xy
    hidden_features: int = 32
    n_hidden_layers: int = 2
    dropout_p: float = 0.1
    activation: str = "gelu"  # "gelu" | "relu" | "tanh"
    out_features: int = 1  # eddy viscosity ν_t


def _activation(name: str) -> nn.Module:
    name = str(name).lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name!r}")


class MCDropoutMLP(nn.Module):
    """MLP surrogate with dropout layers used for MC-dropout sampling.

    Input ``(..., in_features)`` strain-rate features map to ``(..., 1)``
    eddy-viscosity values.  Keeping the module in *training* mode during
    inference activates the dropout layers and yields a different sample on
    every forward pass.
    """

    def __init__(self, arch: UQMLPArch | None = None) -> None:
        super().__init__()
        self.arch = arch or UQMLPArch()
        layers: list[nn.Module] = []
        in_dim = int(self.arch.in_features)
        for _ in range(int(self.arch.n_hidden_layers)):
            layers.append(nn.Linear(in_dim, int(self.arch.hidden_features)))
            layers.append(_activation(self.arch.activation))
            layers.append(nn.Dropout(float(self.arch.dropout_p)))
            in_dim = int(self.arch.hidden_features)
        layers.append(nn.Linear(in_dim, int(self.arch.out_features)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def arch_dict(self) -> dict[str, Any]:
        """Return the architecture hyper-parameters as a plain dict."""
        return asdict(self.arch)


# ---------------------------------------------------------------------------
# Persistence helpers (mirror save_model / load_model / save_mesh_gnn)
# ---------------------------------------------------------------------------


def save_uq_mlp(model: MCDropoutMLP, path: str | Path) -> Path:
    """Serialize an :class:`MCDropoutMLP` to a ``.pt`` file plus JSON metadata."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), p)
    meta = {
        "arch": model.arch_dict(),
        "model_class": "MCDropoutMLP",
        "format_version": 1,
    }
    meta_path = p.with_suffix(p.suffix + ".json")
    meta_path.write_text(json.dumps(meta, indent=2))
    return p


def load_uq_mlp(path: str | Path) -> MCDropoutMLP:
    """Load an :class:`MCDropoutMLP` saved by :func:`save_uq_mlp`."""
    p = Path(path)
    blob = torch.load(p, map_location="cpu", weights_only=True)
    meta_path = p.with_suffix(p.suffix + ".json")
    arch_dict: dict[str, Any] = {}
    if meta_path.exists():
        arch_dict = json.loads(meta_path.read_text()).get("arch") or {}
    arch = UQMLPArch(**arch_dict) if arch_dict else UQMLPArch()
    model = MCDropoutMLP(arch)
    state_dict = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported model payload in {p}")
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_tensor(x: Any) -> torch.Tensor:
    """Coerce a field snapshot into a float32 ``torch.Tensor``."""
    if isinstance(x, torch.Tensor):
        return x.to(torch.float32)
    return torch.as_tensor(x, dtype=torch.float32)


def _sample_to_features(sample: Any) -> torch.Tensor:
    """Turn an inference ``sample`` into a ``(M, in_features)`` feature matrix.

    Accepts either a ``(ux, uy)`` tuple of 2-D velocity fields (strain-rate
    features are derived on the fly) or a raw feature tensor of shape
    ``(M, in_features)`` / ``(in_features,)``.
    """
    if isinstance(sample, (tuple, list)):
        ux, uy = sample
        s_xx, s_yy, s_xy = strain_rate_tensor_2d(_to_tensor(ux), _to_tensor(uy))
        return torch.stack([s_xx, s_yy, s_xy], dim=-1).reshape(-1, 3)
    x = _to_tensor(sample)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    return x


def _coerce_training_result(result: Any, *, out_path: Path) -> TrainingResult:
    """Normalise a ``train_fn`` return value into a :class:`TrainingResult`."""
    if isinstance(result, TrainingResult):
        return result
    if isinstance(result, Mapping):
        return TrainingResult(
            model_path=str(result.get("model_path") or result.get("path") or out_path),
            metrics={
                "train_loss": float(
                    result.get("train_loss", result.get("final_train_loss", float("nan")))
                ),
            },
            arch=dict(result.get("arch") or {}),
        )
    raise TypeError(
        "train_fn must return a TrainingResult or a mapping with "
        f"'model_path'/'metrics'/'arch' keys, got {type(result).__name__}",
    )


# ---------------------------------------------------------------------------
# The application
# ---------------------------------------------------------------------------


class UncertaintyQuantification(AI4SApplication):
    """MC-dropout uncertainty quantification over an MLP flow-field surrogate.

    Attributes:
        name: Registry name of the application.
        family: Serving model family (``"uq"``).
        n_mc_samples: Number of stochastic forward passes used by
            :meth:`infer` (may be overridden per instance).
    """

    name: str = "uncertainty_quantification"
    family: str = "uq"
    version: str = "1.0"

    n_mc_samples: int = 20

    def __init__(
        self,
        *,
        train_fn: Callable[..., Any] | None = None,
        run_les_fn: Callable[..., Any] | None = None,
        n_mc_samples: int = 20,
    ) -> None:
        """Create the app, optionally injecting the solver / training steps.

        Args:
            train_fn: Optional override for the training step, called as
                ``train_fn(dataset, model, cfg)`` and returning a
                :class:`TrainingResult` (or a compatible mapping).
            run_les_fn: Optional override for the data-production solver,
                called with the same keyword arguments as ``_run_les_smoke``.
            n_mc_samples: Default number of MC-dropout samples in :meth:`infer`.
        """
        super().__init__()
        self._train_fn = train_fn
        self._run_les_fn = run_les_fn
        self.n_mc_samples = max(1, int(n_mc_samples))

    # ---- developer-implemented interface ---------------------------------

    def produce_data(self, cfg: Mapping[str, Any]) -> DataProduct:
        """Run a short LES smoke simulation and return 2-D velocity snapshots.

        The snapshots are carried in ``DataProduct.metadata["snapshots"]`` so
        the downstream ``make_dataset`` step can build strain-rate → eddy-
        viscosity sample pairs without touching disk.
        """
        nx = int(cfg.get("nx", 32))
        ny = int(cfg.get("ny", 32))
        tau = float(cfg.get("tau", 0.8))
        c_s = float(cfg.get("c_s", 0.1))
        n_steps = int(cfg.get("n_steps", 16))
        sample_every = int(cfg.get("sample_every", 4))
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
            name="UQ 2D velocity snapshots",
            field_name="nu_t",
            shape=tuple(_to_tensor(ux0).shape),
            dtype=str(_to_tensor(ux0).dtype),
            units="lu",
            metadata={
                "snapshots": snapshots,
                "n_snapshots": len(snapshots),
                "c_s": c_s,
            },
        )

    def build_model(self, arch: Mapping[str, Any]) -> torch.nn.Module:
        """Construct the dropout-MLP surrogate from an arch mapping."""
        if isinstance(arch, UQMLPArch):
            return MCDropoutMLP(arch)
        return MCDropoutMLP(
            UQMLPArch(
                in_features=int(arch.get("in_features", 3)),
                hidden_features=int(arch.get("hidden_features", 32)),
                n_hidden_layers=int(arch.get("n_hidden_layers", 2)),
                dropout_p=float(arch.get("dropout_p", 0.1)),
                activation=str(arch.get("activation", "gelu")),
                out_features=int(arch.get("out_features", 1)),
            ),
        )

    def make_dataset(self, product: DataProduct) -> dict[str, Any]:
        """Build strain-rate → eddy-viscosity regression samples from snapshots.

        Returns ``{"inputs": (N, 3), "targets": (N, 1), ...}`` where each row
        is one lattice cell's strain-rate features and its Smagorinsky eddy-
        viscosity label.
        """
        snapshots = product.metadata.get("snapshots")
        if not snapshots:
            raise ValueError(
                "DataProduct metadata must carry 'snapshots' (list of (ux, uy))",
            )
        c_s = float(product.metadata.get("c_s", 0.1))

        feats_list: list[torch.Tensor] = []
        targs_list: list[torch.Tensor] = []
        for ux, uy in snapshots:
            f, t = extract_les_samples_2d(_to_tensor(ux), _to_tensor(uy), c_s=c_s)
            feats_list.append(f)
            targs_list.append(t)

        inputs = torch.cat(feats_list, dim=0)
        targets = torch.cat(targs_list, dim=0)
        return {
            "inputs": inputs,
            "targets": targets,
            "c_s": c_s,
            "n_samples": int(inputs.shape[0]),
            "in_features": int(inputs.shape[1]),
        }

    def train(
        self,
        dataset: Any,
        model: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> TrainingResult:
        """Train the dropout-MLP surrogate and return weights path + metrics."""
        out_path = Path(
            str(cfg.get("out_path") or cfg.get("model_path") or "uq_mlp_model.pt"),
        )

        if self._train_fn is not None:
            return _coerce_training_result(
                self._train_fn(dataset, model, cfg),
                out_path=out_path,
            )

        result = _train_uq_mlp(
            dataset,
            cast(MCDropoutMLP, model),
            out_path,
            epochs=int(cfg.get("epochs", 30)),
            batch_size=int(cfg.get("batch_size", 256)),
            learning_rate=float(cfg.get("learning_rate", 1e-3)),
            seed=int(cfg.get("seed", 0)),
            device=str(cfg.get("device", "cpu")),
        )
        arch = dict(result.get("arch") or {})
        if not arch:
            arch = model.arch_dict() if isinstance(model, MCDropoutMLP) else {}
        return TrainingResult(
            model_path=str(result.get("path", out_path)),
            metrics={"train_loss": float(result.get("train_loss", float("nan")))},
            arch=arch,
        )

    def infer(self, model: torch.nn.Module, sample: Any) -> Prediction:
        """MC-dropout inference: return predictive mean, std, and raw samples.

        The model is placed in training mode (dropout active) and ``N``
        stochastic forward passes are collected; the per-point standard
        deviation across the samples quantifies epistemic uncertainty.
        """
        x = _sample_to_features(sample)  # (M, in_features)
        n = max(1, int(getattr(self, "n_mc_samples", 20)))
        device = next(model.parameters()).device
        x = x.to(device=device)

        model.train()  # keep dropout active (MC-dropout)
        samples: list[torch.Tensor] = []
        with torch.no_grad():
            for _ in range(n):
                samples.append(model(x))
        stacked = torch.stack(samples, dim=0)  # (N, M, out_features)
        mean = stacked.mean(dim=0)  # (M, out_features)
        std = stacked.std(dim=0)  # (M, out_features)
        return Prediction(
            output={
                "mean": mean,
                "std": std,
                "samples": stacked,
            },
            metadata={
                "method": "mc_dropout",
                "n_samples": n,
                "shape": tuple(mean.shape),
                "units": "lu",
            },
        )


# ---------------------------------------------------------------------------
# Default training loop
# ---------------------------------------------------------------------------


def _train_uq_mlp(
    dataset: Mapping[str, Any],
    model: MCDropoutMLP,
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
    path = save_uq_mlp(model, out_path)
    return {
        "path": str(path),
        "train_loss": final_loss,
        "arch": model.arch_dict(),
    }
