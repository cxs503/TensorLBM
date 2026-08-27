"""Inverse problem as an :class:`AI4SApplication` — recover physics from data.

Whereas a PINN *forwards* the physics (the residual is part of the loss), an
inverse problem runs the physics *backwards*: the flow field is observed and
the unknown physical parameters that produced it are reconstructed by
differentiating through a forward model with ``torch.autograd``.

The reference model here is the analytic plane *Couette–Poiseuille* channel
flow,

.. math::

    u(y) = U_w \\cdot \\frac{y}{H} + \\frac{G}{2\\,\\nu}\\, y\\,(H - y)

where :math:`G` is the (known) streamwise pressure gradient, :math:`H` the
(known) channel half-height, :math:`U_w` the wall velocity and :math:`\\nu` the
kinematic viscosity.  :math:`\\nu` and :math:`U_w` are the *unknown* physical
parameters: given noisy point observations of :math:`u(y)`, the application
minimises the mean-squared error between the parametric forward model and the
observations, back-propagating the gradient into ``nu`` / ``u_wall``.  The two
parameters are independently identifiable (``U_w`` is pinned by the value at
``y = H``, ``G/\\nu`` by the parabolic bulge), so this is a well-posed 1- or
2-parameter inversion.

Data production is injectable (``field_fn``) and so is the training loop
(``train_fn``), keeping the full :meth:`run` pipeline testable without any HPC
solver run.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, cast

import torch
import torch.nn as nn

from tensorlbm.apps.base import (
    AI4SApplication,
    DataProduct,
    Prediction,
    TrainingResult,
)

__all__ = [
    "InverseArch",
    "InverseProblem",
    "ParametricChannelFlow",
    "load_inverse_model",
    "save_inverse_model",
]


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InverseArch:
    """Description of a :class:`ParametricChannelFlow` inverse model.

    ``nu_init`` / ``u_wall_init`` are the initial guesses for the unknown
    physical parameters; ``G`` / ``H`` are the (known) flow geometry;
    ``invert`` lists which parameters are unknown and therefore optimised.
    """

    nu_init: float = 0.5
    u_wall_init: float = 0.0
    G: float = 1.0
    H: float = 1.0
    invert: tuple[str, ...] = ("nu", "u_wall")


class ParametricChannelFlow(nn.Module):
    """Differentiable forward model of plane Couette–Poiseuille channel flow.

    The unknown physical parameters are ``nn.Parameter`` s (``nu`` is kept
    positive via an ``exp`` reparameterisation of ``log_nu``); the geometry
    ``G`` / ``H`` is fixed and known.  ``forward`` maps a ``(N, 1)``
    coordinate tensor ``y`` to the ``(N, 1)`` streamwise velocity ``u(y)``.
    """

    def __init__(
        self,
        nu_init: float = 0.5,
        u_wall_init: float = 0.0,
        G: float = 1.0,
        H: float = 1.0,
        invert: tuple[str, ...] = ("nu", "u_wall"),
    ) -> None:
        super().__init__()
        self.G = float(G)
        self.H = float(H)
        self.invert = tuple(invert)
        # nu > 0 is enforced by exp(); store the raw log so the value stays
        # positive under unconstrained gradient descent.
        self.log_nu = nn.Parameter(torch.tensor(math.log(max(float(nu_init), 1e-6))))
        self.u_wall = nn.Parameter(torch.tensor(float(u_wall_init)))

    def forward(self, coords: torch.Tensor) -> torch.Tensor:  # noqa: D401
        y = coords[..., 0]
        nu = torch.exp(self.log_nu)
        u = self.u_wall * (y / self.H) + (self.G / (2.0 * nu)) * y * (self.H - y)
        return u.unsqueeze(-1)

    # -- accessors ---------------------------------------------------------

    def nu(self) -> float:
        """Current (recovered) kinematic viscosity."""
        return float(torch.exp(self.log_nu).detach().item())

    def u_wall_value(self) -> float:
        """Current (recovered) wall velocity."""
        return float(self.u_wall.detach().item())

    def param_values(self) -> dict[str, float]:
        """Current values of the unknown physical parameters."""
        return {"nu": self.nu(), "u_wall": self.u_wall_value()}

    def arch_dict(self) -> dict[str, Any]:
        """Serialisable descriptor of the recovered model (for serving)."""
        return {
            "nu": self.nu(),
            "u_wall": self.u_wall_value(),
            "G": self.G,
            "H": self.H,
            "invert": list(self.invert),
        }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_inverse_model(model: nn.Module, path: str | Path) -> Path:
    """Serialize a :class:`ParametricChannelFlow` to a ``.pt`` + JSON sidecar."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), p)
    meta = {"arch": cast(ParametricChannelFlow, model).arch_dict(), "format_version": 1}
    p.with_suffix(p.suffix + ".json").write_text(json.dumps(meta, indent=2))
    return p


def load_inverse_model(path: str | Path) -> ParametricChannelFlow:
    """Inverse of :func:`save_inverse_model` (used by the serving layer)."""
    p = Path(path)
    state = torch.load(p, map_location="cpu", weights_only=True)
    meta_path = p.with_suffix(p.suffix + ".json")
    arch_dict: dict[str, Any] = {}
    if meta_path.exists():
        arch_dict = json.loads(meta_path.read_text()).get("arch") or {}
    model = ParametricChannelFlow(
        nu_init=float(arch_dict.get("nu", 0.5)),
        u_wall_init=float(arch_dict.get("u_wall", 0.0)),
        G=float(arch_dict.get("G", 1.0)),
        H=float(arch_dict.get("H", 1.0)),
        invert=tuple(arch_dict.get("invert", ("nu", "u_wall"))),
    )
    model.load_state_dict(state)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------


def _couette_poiseuille(
    coords: torch.Tensor,
    *,
    nu: float,
    u_wall: float,
    G: float,
    H: float,
) -> torch.Tensor:
    """Exact plane Couette–Poiseuille profile (the ground-truth generator)."""
    y = coords[..., 0]
    u = u_wall * (y / H) + (G / (2.0 * nu)) * y * (H - y)
    return u.unsqueeze(-1)


# ---------------------------------------------------------------------------
# The application
# ---------------------------------------------------------------------------


class InverseProblem(AI4SApplication):
    """Recover unknown physical parameters (``nu`` / ``U_w``) from flow data.

    Attributes:
        name: Registry name of the application.
        family: Serving model family (``"inverse"`` is understood by
            :class:`tensorlbm.ml.serving.InferenceService`).
    """

    name: str = "inverse_problem"
    family: str = "inverse"
    version: str = "1.0"

    def __init__(
        self,
        *,
        train_fn: Callable[..., Any] | None = None,
        field_fn: Callable[..., Any] | None = None,
    ) -> None:
        """Create the app, optionally injecting the field / training loop.

        Args:
            train_fn: Optional override for the inversion step.  Called as
                ``train_fn(dataset, model, cfg)`` and must return a
                :class:`TrainingResult` (or a mapping with ``model_path`` /
                ``metrics`` / ``arch`` keys).  Defaults to the local
                Adam + observation-MSE gradient-descent inversion.
            field_fn: Optional override for the ground-truth field generator.
                Called as ``field_fn(coords, nu=..., u_wall=..., G=..., H=...)``
                and must return a ``(N, 1)`` velocity tensor.  Defaults to
                :func:`_couette_poiseuille`.
        """
        super().__init__()
        self._train_fn = train_fn
        self._field_fn = field_fn

    # ---- developer-implemented interface ---------------------------------

    def produce_data(self, cfg: Mapping[str, Any]) -> DataProduct:
        """Generate the "true" flow field from known parameters and observe it.

        The observed ``(y, u)`` points, the true parameters and the geometry are
        carried in :attr:`DataProduct.metadata` so ``make_dataset`` needs no
        disk access.
        """
        G = float(cfg.get("G", 1.0))
        H = float(cfg.get("H", 1.0))
        nu_true = float(cfg.get("nu_true", 0.1))
        u_wall_true = float(cfg.get("u_wall_true", 0.0))
        n_points = int(cfg.get("n_points", 64))
        y0 = float(cfg.get("y0", 0.0))
        y1 = float(cfg.get("y1", H))
        noise = float(cfg.get("noise", 0.0))
        seed = int(cfg.get("seed", 0))
        device = torch.device(str(cfg.get("device", "cpu")))

        field_fn = self._field_fn or _couette_poiseuille
        coords = torch.linspace(y0, y1, n_points, device=device, dtype=torch.float32).unsqueeze(-1)
        obs = (
            field_fn(
                coords,
                nu=nu_true,
                u_wall=u_wall_true,
                G=G,
                H=H,
            )
            .detach()
            .clone()
        )
        if noise > 0.0:
            g = torch.Generator(device="cpu").manual_seed(int(seed))
            obs = obs + float(noise) * torch.randn(obs.shape, generator=g).to(obs)

        return DataProduct(
            name="Couette–Poiseuille channel-flow velocity observations",
            field_name="velocity_field",
            shape=tuple(obs.shape),
            dtype=str(obs.dtype),
            units="lu",
            metadata={
                "coords": coords,
                "observations": obs,
                "nu_true": nu_true,
                "u_wall_true": u_wall_true,
                "G": G,
                "H": H,
                "domain": [y0, y1],
                "channels": ["u"],
            },
        )

    def build_model(self, arch: Mapping[str, Any]) -> torch.nn.Module:
        """Construct the parametric forward model from an arch mapping."""
        if isinstance(arch, InverseArch):
            inv_arch = arch
        else:
            inv_arch = InverseArch(
                nu_init=float(arch.get("nu_init", 0.5)),
                u_wall_init=float(arch.get("u_wall_init", 0.0)),
                G=float(arch.get("G", 1.0)),
                H=float(arch.get("H", 1.0)),
                invert=tuple(arch.get("invert", ("nu", "u_wall"))),
            )
        return ParametricChannelFlow(
            nu_init=inv_arch.nu_init,
            u_wall_init=inv_arch.u_wall_init,
            G=inv_arch.G,
            H=inv_arch.H,
            invert=inv_arch.invert,
        )

    def make_dataset(self, product: DataProduct) -> dict[str, Any]:
        """Build the observation set ``{"coords", "observations", ...}``."""
        required = ("coords", "observations")
        missing = [k for k in required if product.metadata.get(k) is None]
        if missing:
            raise ValueError(
                f"DataProduct metadata must carry {required}, missing {missing}",
            )
        return {
            "coords": product.metadata["coords"],
            "observations": product.metadata["observations"],
            "nu_true": float(product.metadata.get("nu_true", 0.1)),
            "u_wall_true": float(product.metadata.get("u_wall_true", 0.0)),
            "G": float(product.metadata.get("G", 1.0)),
            "H": float(product.metadata.get("H", 1.0)),
            "domain": tuple(product.metadata.get("domain", (0.0, 1.0))),
        }

    def train(
        self,
        dataset: Any,
        model: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> TrainingResult:
        """Invert the unknown parameters and return the recovered-parameter path.

        The injectable ``train_fn`` (if provided) is called as
        ``train_fn(dataset, model, cfg)``; otherwise the local Adam loop
        minimises the observation MSE over the parameters in ``model.invert``
        and writes the checkpoint with :func:`save_inverse_model`.
        """
        out_path = Path(
            str(cfg.get("out_path") or cfg.get("model_path") or "inverse_model.pt"),
        )

        if self._train_fn is not None:
            return _coerce_training_result(self._train_fn(dataset, model, cfg), out_path=out_path)

        result = _train_inverse(
            dataset,
            cast(ParametricChannelFlow, model),
            out_path,
            epochs=int(cfg.get("epochs", 300)),
            learning_rate=float(cfg.get("learning_rate", 5e-2)),
            seed=int(cfg.get("seed", 0)),
            device=str(cfg.get("device", "cpu")),
        )
        return TrainingResult(
            model_path=str(result["path"]),
            metrics={
                "final_loss": float(result["final_loss"]),
                "recovered_nu": float(result["recovered_nu"]),
                "recovered_u_wall": float(result["recovered_u_wall"]),
                "nu_error": float(result["nu_error"]),
                "u_wall_error": float(result["u_wall_error"]),
            },
            arch=result["arch"],
        )

    def infer(self, model: torch.nn.Module, sample: Any) -> Prediction:
        """Evaluate the reconstructed field at coordinate sample(s) ``y``.

        ``sample`` may be a ``(N, 1)`` / ``(N,)`` tensor (batch), a scalar, or a
        1-D array of ``y`` coordinates.
        """
        y = _to_coords(sample)
        scalar = y.ndim == 0
        if scalar:
            y = y.reshape(1, 1)
            single = True
        elif y.ndim == 1:
            y = y.unsqueeze(-1)
            single = True
        else:
            single = False
        model.eval()
        with torch.no_grad():
            out = model(y)
        if scalar:
            out = out.reshape(())
        elif single:
            out = out.squeeze(-1)
        meta: dict[str, Any] = {"field_name": "velocity_field", "units": "lu"}
        if isinstance(model, ParametricChannelFlow):
            meta.update(model.param_values())
        return Prediction(output=out, metadata=meta)


# ---------------------------------------------------------------------------
# Default inversion loop
# ---------------------------------------------------------------------------


def _train_inverse(
    dataset: Mapping[str, Any],
    model: ParametricChannelFlow,
    out_path: Path,
    *,
    epochs: int,
    learning_rate: float,
    seed: int,
    device: str,
) -> dict[str, Any]:
    """Gradient-descent inversion: minimise MSE(pred, obs) over the parameters."""
    coords = dataset["coords"]
    obs = dataset["observations"]
    nu_true = float(dataset.get("nu_true", 0.1))
    u_wall_true = float(dataset.get("u_wall_true", 0.0))

    torch_device = torch.device(device)
    torch.manual_seed(int(seed))
    model.to(torch_device)
    model.train()

    # Pin the known geometry onto the model.
    model.G = float(dataset.get("G", model.G))
    model.H = float(dataset.get("H", model.H))

    # Known (non-inverted) parameters are pinned to their true value and frozen.
    invert = set(getattr(model, "invert", ("nu", "u_wall")))
    if "nu" not in invert:
        model.log_nu.data.fill_(math.log(max(nu_true, 1e-6)))
        model.log_nu.requires_grad_(False)
    if "u_wall" not in invert:
        model.u_wall.data.fill_(u_wall_true)
        model.u_wall.requires_grad_(False)

    coords = coords.to(torch_device)
    obs = obs.to(torch_device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=float(learning_rate))
    loss_fn = nn.MSELoss()

    final_loss = float("nan")
    for _ in range(max(0, int(epochs))):
        optimizer.zero_grad()
        pred = model(coords)
        loss = loss_fn(pred, obs)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())

    model.eval()
    recovered = model.param_values()
    path = save_inverse_model(model, out_path)
    return {
        "path": str(path),
        "final_loss": final_loss,
        "recovered_nu": recovered["nu"],
        "recovered_u_wall": recovered["u_wall"],
        "nu_error": abs(recovered["nu"] - nu_true),
        "u_wall_error": abs(recovered["u_wall"] - u_wall_true),
        "arch": model.arch_dict(),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_coords(sample: Any) -> torch.Tensor:
    """Normalise a coordinate sample into a ``float32`` tensor of shape ``(...,)``."""
    if isinstance(sample, torch.Tensor):
        y = sample.to(torch.float32)
    else:
        y = torch.as_tensor(sample, dtype=torch.float32)
    return y


def _coerce_training_result(result: Any, *, out_path: Path) -> TrainingResult:
    """Normalise a ``train_fn`` return value into a :class:`TrainingResult`."""
    if isinstance(result, TrainingResult):
        return result
    if isinstance(result, Mapping):
        metrics = dict(result.get("metrics") or {})
        if not metrics:
            metrics = {
                "final_loss": float(result.get("final_loss", float("nan"))),
                "recovered_nu": float(result.get("recovered_nu", float("nan"))),
                "recovered_u_wall": float(result.get("recovered_u_wall", float("nan"))),
            }
        return TrainingResult(
            model_path=str(result.get("model_path") or result.get("path") or out_path),
            metrics=metrics,
            arch=dict(result.get("arch") or {}),
        )
    raise TypeError(
        "train_fn must return a TrainingResult or a mapping with "
        f"'model_path'/'metrics'/'arch' keys, got {type(result).__name__}",
    )
