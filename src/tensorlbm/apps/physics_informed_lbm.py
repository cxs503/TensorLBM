"""Physics-Informed Neural Network (PINN) as an :class:`AI4SApplication`.

This module turns the Physics-Informed Neural Network idea (Raissi et al.,
2019, Journal of Computational Physics, arXiv:1711.10561) into a concrete
platform application.  A multi-layer perceptron represents a 2-D steady
incompressible flow field ``(x, y) -> (u, v, p)`` and is trained with a loss
that combines a *data* term (match labelled velocity/pressure points) and a
*physics-residual* term (the incompressible Navier-Stokes equations, whose
derivatives are computed with ``torch.autograd``).

The residuals enforced are:

* continuity       : ``∂u/∂x + ∂v/∂y = 0``
* x-momentum       : ``u·∂u/∂x + v·∂u/∂y + ∂p/∂x − ν·∇²u = 0``
* y-momentum       : ``u·∂v/∂x + v·∂v/∂y + ∂p/∂y − ν·∇²v = 0``

The reference field is an *exact* steady solution of the incompressible Euler
equations (``ν = 0``) — the Taylor–Green vortex — so the physics residual is
identically zero for the ground-truth field, giving the network a well-posed
target.  Data production is injectable (``field_fn``) and so is the training
loop (``train_fn``), keeping the full :meth:`run` pipeline testable without any
HPC solver run.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
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
    "PINNMLP",
    "PinnArch",
    "PhysicsInformedLBM",
    "load_pinn_model",
    "save_pinn_model",
]


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PinnArch:
    """Hyper-parameters describing a :class:`PINNMLP`."""

    in_features: int = 2
    hidden_dim: int = 32
    n_layers: int = 3
    out_features: int = 3
    activation: str = "tanh"  # "tanh" | "relu" | "gelu"


def _activation(name: str) -> nn.Module:
    name = str(name).lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name!r}")


class PINNMLP(nn.Module):
    """MLP that maps spatial coordinates ``(x, y)`` to ``(u, v, p)``.

    The network is a stack of ``n_layers`` ``Linear -> activation`` blocks
    followed by a final linear projection to ``out_features`` (3 by default:
    the two velocity components and the pressure).  ``tanh`` is the default
    activation, matching standard PINN practice for smooth flow fields.
    """

    def __init__(self, arch: PinnArch | None = None) -> None:
        super().__init__()
        self.arch = arch or PinnArch()
        layers: list[nn.Module] = []
        in_dim = self.arch.in_features
        for _ in range(self.arch.n_layers):
            layers.append(nn.Linear(in_dim, self.arch.hidden_dim))
            layers.append(_activation(self.arch.activation))
            in_dim = self.arch.hidden_dim
        layers.append(nn.Linear(in_dim, self.arch.out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_pinn_model(model: nn.Module, path: str | Path) -> Path:
    """Serialize a :class:`PINNMLP` to a ``.pt`` state-dict + JSON arch."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), p)
    meta = {"arch": asdict(cast(PINNMLP, model).arch), "format_version": 1}
    p.with_suffix(p.suffix + ".json").write_text(json.dumps(meta, indent=2))
    return p


def load_pinn_model(path: str | Path) -> PINNMLP:
    """Inverse of :func:`save_pinn_model`."""
    p = Path(path)
    state = torch.load(p, map_location="cpu", weights_only=True)
    meta_path = p.with_suffix(p.suffix + ".json")
    arch_dict: dict[str, Any] = {}
    if meta_path.exists():
        arch_dict = json.loads(meta_path.read_text()).get("arch") or {}
    arch = PinnArch(**arch_dict) if arch_dict else PinnArch()
    model = PINNMLP(arch)
    model.load_state_dict(state)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

def _taylor_green(xy: torch.Tensor) -> torch.Tensor:
    """Taylor–Green vortex: an exact steady solution of 2-D incompressible Euler.

    ``u = -cos(x)·sin(y)``, ``v = sin(x)·cos(y)``,
    ``p = -¼·(cos(2x) + cos(2y))``.

    With ``ν = 0`` this field satisfies both the continuity and the momentum
    equations exactly, so it is a clean ground-truth for the PINN residual.
    """
    x = xy[..., 0]
    y = xy[..., 1]
    u = -torch.cos(x) * torch.sin(y)
    v = torch.sin(x) * torch.cos(y)
    p = -0.25 * (torch.cos(2.0 * x) + torch.cos(2.0 * y))
    return torch.stack([u, v, p], dim=-1)


def pde_residuals(
    model: nn.Module,
    xy: torch.Tensor,
    nu: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the (continuity, x-momentum, y-momentum) residuals at ``xy``.

    ``xy`` must be a ``(N, 2)`` coordinate tensor.  Derivatives are obtained
    with ``torch.autograd.grad`` (``create_graph=True``) so the residuals carry
    gradients with respect to the network parameters — the standard PINN
    mechanism.  Returns three ``(N, 1)`` tensors.
    """
    xy = xy.clone().detach().requires_grad_(True)
    pred = model(xy)  # (N, 3)
    u = pred[:, 0:1]
    v = pred[:, 1:2]
    p = pred[:, 2:3]

    ones = torch.ones_like(u)

    du = torch.autograd.grad(u, xy, grad_outputs=ones, create_graph=True)[0]
    dv = torch.autograd.grad(v, xy, grad_outputs=ones, create_graph=True)[0]
    dp = torch.autograd.grad(p, xy, grad_outputs=ones, create_graph=True)[0]

    u_x, u_y = du[:, 0:1], du[:, 1:2]
    v_x, v_y = dv[:, 0:1], dv[:, 1:2]
    p_x, p_y = dp[:, 0:1], dp[:, 1:2]

    u_xx = torch.autograd.grad(u_x, xy, grad_outputs=ones, create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, xy, grad_outputs=ones, create_graph=True)[0][:, 1:2]
    v_xx = torch.autograd.grad(v_x, xy, grad_outputs=ones, create_graph=True)[0][:, 0:1]
    v_yy = torch.autograd.grad(v_y, xy, grad_outputs=ones, create_graph=True)[0][:, 1:2]

    continuity = u_x + v_y
    momentum_x = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    momentum_y = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
    return continuity, momentum_x, momentum_y


def _sample_points(
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    n: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Sample ``n`` random points uniformly in ``[x0, x1] × [y0, y1]``."""
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    x = torch.rand(int(n), generator=g) * (x1 - x0) + x0
    y = torch.rand(int(n), generator=g) * (y1 - y0) + y0
    return torch.stack([x, y], dim=-1).to(device=device, dtype=torch.float32)


# ---------------------------------------------------------------------------
# The application
# ---------------------------------------------------------------------------

class PhysicsInformedLBM(AI4SApplication):
    """2-D steady incompressible-flow PINN as a platform application.

    Attributes:
        name: Registry name of the application.
        family: Serving model family (``"pinn"`` is understood by
            :class:`tensorlbm.ml.serving.InferenceService`).
    """

    name: str = "physics_informed_lbm"
    family: str = "pinn"
    version: str = "1.0"

    def __init__(
        self,
        *,
        train_fn: Callable[..., Any] | None = None,
        field_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        """Create the app, optionally injecting the reference-field / training.

        Args:
            train_fn: Optional override for the training step.  It is called as
                ``train_fn(dataset, model, cfg)`` and must return a
                :class:`TrainingResult` (or a mapping with ``model_path`` /
                ``metrics`` / ``arch`` keys).  Defaults to the local
                Adam + (data + λ·physics) training loop.
            field_fn: Optional override for the reference field.  It maps a
                ``(N, 2)`` coordinate tensor to a ``(N, 3)`` ``(u, v, p)``
                tensor.  Defaults to :func:`_taylor_green`.
        """
        super().__init__()
        self._train_fn = train_fn
        self._field_fn = field_fn

    # ---- developer-implemented interface ---------------------------------

    def produce_data(self, cfg: Mapping[str, Any]) -> DataProduct:
        """Generate a 2-D reference flow field and sample training points.

        The default reference field is the Taylor–Green vortex (an exact
        steady solution of the incompressible Euler equations); an LBM smoke
        run (or any other source) can be substituted via the ``field_fn``
        injection.  The sampled ``(x, y) -> (u, v, p)`` points are carried in
        :attr:`DataProduct.metadata` so ``make_dataset`` needs no disk access.
        """
        x0 = float(cfg.get("x0", 0.0))
        x1 = float(cfg.get("x1", 2.0 * math.pi))
        y0 = float(cfg.get("y0", 0.0))
        y1 = float(cfg.get("y1", 2.0 * math.pi))
        n_points = int(cfg.get("n_points", 256))
        n_collocation = int(cfg.get("n_collocation", n_points))
        seed = int(cfg.get("seed", 0))
        device = torch.device(str(cfg.get("device", "cpu")))
        nu = float(cfg.get("nu", 0.0))

        field_fn = self._field_fn or _taylor_green
        points = _sample_points(x0, x1, y0, y1, n_points, seed, device)
        labels = field_fn(points).detach().clone()
        collocation = _sample_points(
            x0, x1, y0, y1, n_collocation, seed + 1, device,
        )

        return DataProduct(
            name="PINN 2D steady incompressible flow field",
            field_name="flow_field",
            shape=tuple(labels.shape),
            dtype=str(labels.dtype),
            units="lu",
            metadata={
                "points": points,
                "labels": labels,
                "collocation": collocation,
                "domain": [x0, x1, y0, y1],
                "nu": nu,
                "channels": ["u", "v", "p"],
            },
        )

    def build_model(self, arch: Mapping[str, Any]) -> torch.nn.Module:
        """Construct the coordinate-to-field MLP from an arch mapping."""
        if isinstance(arch, PinnArch):
            pinn_arch = arch
        else:
            pinn_arch = PinnArch(
                in_features=int(arch.get("in_features", 2)),
                hidden_dim=int(arch.get("hidden_dim", 32)),
                n_layers=int(arch.get("n_layers", 3)),
                out_features=int(arch.get("out_features", 3)),
                activation=str(arch.get("activation", "tanh")),
            )
        return PINNMLP(pinn_arch)

    def make_dataset(self, product: DataProduct) -> dict[str, Any]:
        """Build the PINN training point set from a data product.

        Returns a light dict ``{"points", "labels", "collocation", "domain",
        "nu"}`` — the labelled data points plus the collocation points on which
        the physics residual is evaluated during training.
        """
        required = ("points", "labels", "collocation")
        missing = [k for k in required if product.metadata.get(k) is None]
        if missing:
            raise ValueError(
                f"DataProduct metadata must carry {required}, missing {missing}",
            )
        return {
            "points": product.metadata["points"],
            "labels": product.metadata["labels"],
            "collocation": product.metadata["collocation"],
            "domain": tuple(product.metadata.get("domain", (0.0, 1.0, 0.0, 1.0))),
            "nu": float(product.metadata.get("nu", 0.0)),
        }

    def train(
        self,
        dataset: Any,
        model: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> TrainingResult:
        """Train the PINN and return weights path + metrics.

        The injectable ``train_fn`` (if provided) is called as
        ``train_fn(dataset, model, cfg)``; otherwise the local Adam loop
        minimises ``data_mse + λ·physics_residual`` and writes the checkpoint
        with :func:`save_pinn_model`.
        """
        out_path = Path(
            str(cfg.get("out_path") or cfg.get("model_path") or "pinn_model.pt"),
        )

        if self._train_fn is not None:
            return _coerce_training_result(
                self._train_fn(dataset, model, cfg),
                out_path=out_path,
            )

        nu = float(dataset.get("nu", float(cfg.get("nu", 0.0))))
        result = _train_pinn(
            dataset,
            model,
            out_path,
            epochs=int(cfg.get("epochs", 100)),
            learning_rate=float(cfg.get("learning_rate", 1e-3)),
            lambda_physics=float(cfg.get("lambda_physics", 1.0)),
            nu=nu,
            collocation_batch=int(cfg.get("collocation_batch", 0)),
            seed=int(cfg.get("seed", 0)),
            device=str(cfg.get("device", "cpu")),
        )
        arch = dict(result.get("arch") or {})
        if not arch and isinstance(model, PINNMLP):
            arch = asdict(model.arch)
        return TrainingResult(
            model_path=str(result.get("path", out_path)),
            metrics={
                "train_loss": float(result.get("train_loss", float("nan"))),
                "physics_loss": float(result.get("physics_loss", float("nan"))),
                "data_loss": float(result.get("data_loss", float("nan"))),
            },
            arch=arch,
        )

    def infer(self, model: torch.nn.Module, sample: Any) -> Prediction:
        """Evaluate the field at coordinate sample(s) ``(x, y)``.

        ``sample`` may be a ``(N, 2)`` tensor (batch), a ``(2,)`` tensor
        (single point), or a ``(x, y)`` pair of scalars / ``(N,)`` arrays.
        """
        xy = _to_coords(sample)
        single = xy.ndim == 1
        if single:
            xy = xy.unsqueeze(0)
        model.eval()
        with torch.no_grad():
            out = model(xy)
        if single:
            out = out.squeeze(0)
        return Prediction(
            output=out,
            metadata={
                "field_name": "flow_field",
                "shape": tuple(out.shape),
                "units": "lu",
                "channels": ["u", "v", "p"],
            },
        )


# ---------------------------------------------------------------------------
# Default training loop
# ---------------------------------------------------------------------------

def _train_pinn(
    dataset: Mapping[str, Any],
    model: nn.Module,
    out_path: Path,
    *,
    epochs: int,
    learning_rate: float,
    lambda_physics: float,
    nu: float,
    collocation_batch: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    """Adam + (data MSE + λ·physics residual) loop, then checkpoint save."""
    points = dataset["points"]
    labels = dataset["labels"]
    collocation = dataset["collocation"]

    torch_device = torch.device(device)
    torch.manual_seed(int(seed))
    model.to(torch_device)
    model.train()

    points = points.to(torch_device)
    labels = labels.to(torch_device)
    collocation = collocation.to(torch_device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    loss_fn = nn.MSELoss()

    n_col = int(collocation.shape[0])
    collocation_batch = max(0, int(collocation_batch))
    if collocation_batch == 0 or collocation_batch >= n_col:
        collocation_batch = n_col

    final_train = float("nan")
    final_physics = float("nan")
    final_data = float("nan")
    for _ in range(max(0, int(epochs))):
        optimizer.zero_grad()
        data_loss = loss_fn(model(points), labels)

        if n_col > 0:
            if collocation_batch < n_col:
                idx = torch.randperm(n_col, device=torch_device)[:collocation_batch]
                cb = collocation[idx]
            else:
                cb = collocation
            cont, mom_x, mom_y = pde_residuals(model, cb, nu=nu)
            physics_loss = (cont ** 2).mean() + (mom_x ** 2).mean() + (mom_y ** 2).mean()
        else:
            physics_loss = torch.zeros((), device=torch_device)

        loss = data_loss + lambda_physics * physics_loss
        loss.backward()
        optimizer.step()

        final_train = float(loss.item())
        final_physics = float(physics_loss.item())
        final_data = float(data_loss.item())

    model.eval()
    path = save_pinn_model(cast(PINNMLP, model), out_path)
    return {
        "path": str(path),
        "train_loss": final_train,
        "physics_loss": final_physics,
        "data_loss": final_data,
        "arch": asdict(cast(PINNMLP, model).arch),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_coords(sample: Any) -> torch.Tensor:
    """Normalise a coordinate sample into a ``float32`` tensor of shape ``(..., 2)``."""
    if isinstance(sample, torch.Tensor):
        xy = sample.to(torch.float32)
    elif isinstance(sample, (tuple, list)) and len(sample) == 2:
        x, y = sample
        x = torch.as_tensor(x, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.float32)
        xy = torch.stack([x, y], dim=-1)
    else:
        raise TypeError(
            "sample must be a (N,2)/(2,) tensor or a (x, y) pair, "
            f"got {type(sample).__name__}",
        )
    return xy


def _coerce_training_result(
    result: Any,
    *,
    out_path: Path,
) -> TrainingResult:
    """Normalise a ``train_fn`` return value into a :class:`TrainingResult`."""
    if isinstance(result, TrainingResult):
        return result
    if isinstance(result, Mapping):
        metrics = dict(result.get("metrics") or {})
        if not metrics:
            metrics = {
                "train_loss": float(result.get("train_loss", float("nan"))),
                "physics_loss": float(result.get("physics_loss", float("nan"))),
                "data_loss": float(result.get("data_loss", float("nan"))),
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
