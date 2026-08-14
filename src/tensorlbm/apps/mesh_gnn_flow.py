"""MeshGraphNet-style message-passing GNN as an :class:`AI4SApplication`.

This module implements a graph neural network for flow-field prediction on
unstructured/grid graphs, following the *MeshGraphNet* blueprint of Pfaff et
al. 2021 (arXiv:2010.03409): node features (velocity) are encoded, a stack of
message-passing layers exchanges information along edges (neighbouring grid
points), and a decoder maps the enriched node embeddings back to a velocity
prediction (the field at the next time step).

``torch_geometric`` is *not* a hard dependency: the message-passing machinery
is written in pure PyTorch using ``index_select`` (gather) for messages and
``index_add_`` (scatter-add) for aggregation, so the whole stack runs on CPU
with only ``torch`` installed.

Implementing the five framework methods (``produce_data`` / ``build_model`` /
``make_dataset`` / ``train`` / ``infer``) is all that is required; the
full-stack pipeline (catalog registration, training-job lifecycle, model
serving, lineage) is inherited from :meth:`AI4SApplication.run`.  The heavy
pieces (``train_fn`` / ``produce_fn``) are injectable for cheap closed-loop
testing.
"""

from __future__ import annotations

import json
import math
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
    "MeshGraphNet",
    "MeshGraphNetLayer",
    "MeshGNNFlow",
    "load_mesh_gnn",
    "save_mesh_gnn",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_tensor(x: Any) -> torch.Tensor:
    """Coerce ``x`` into a float32 ``torch.Tensor``."""
    if isinstance(x, torch.Tensor):
        return x.to(torch.float32)
    return torch.as_tensor(x, dtype=torch.float32)


def _activation_module(name: str) -> nn.Module:
    name = name.lower()
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"unknown activation {name!r}")


def _mlp(in_dim: int, out_dim: int, hidden_dim: int, activation: str) -> nn.Module:
    """Two-layer MLP with a single hidden layer (MeshGraphNet building block)."""
    return nn.Sequential(
        nn.Linear(int(in_dim), int(hidden_dim)),
        _activation_module(activation),
        nn.Linear(int(hidden_dim), int(out_dim)),
    )


# ---------------------------------------------------------------------------
# Message-passing GNN (pure PyTorch)
# ---------------------------------------------------------------------------

class MeshGraphNetLayer(nn.Module):
    """One MeshGraphNet message-passing layer.

    Operates on node embeddings ``h`` and edge embeddings ``e`` (both of width
    ``hidden_dim``) and a ``(2, E)`` ``edge_index`` in which row 0 holds source
    node ids and row 1 holds destination node ids.  The update follows
    MeshGraphNet: edges are updated first, then messages are built from source
    nodes + updated edges, aggregated at destinations (scatter-add), and nodes
    are updated with the aggregated messages (residual).
    """

    def __init__(self, hidden_dim: int, activation: str) -> None:
        super().__init__()
        hd = int(hidden_dim)
        self.edge_mlp = _mlp(3 * hd, hd, hd, activation)
        self.msg_mlp = _mlp(2 * hd, hd, hd, activation)
        self.node_mlp = _mlp(2 * hd, hd, hd, activation)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        e: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src = edge_index[0]
        dst = edge_index[1]

        # 1. edge update (residual)
        e = e + self.edge_mlp(torch.cat([e, h[src], h[dst]], dim=-1))

        # 2. message: (source node, updated edge) -> message
        msg = self.msg_mlp(torch.cat([h[src], e], dim=-1))

        # 3. aggregate at destinations (scatter-add; disjoint graphs stay
        #    separate because there are no cross-graph edges)
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, msg)

        # 4. node update (residual)
        h = h + self.node_mlp(torch.cat([h, agg], dim=-1))

        return h, e


class MeshGraphNet(nn.Module):
    """MeshGraphNet-style encoder-processor-decoder GNN.

    Args:
        node_dim: Node input feature width (e.g. 2 for ``(u, v)`` velocity).
        edge_dim: Edge input feature width (relative displacement + velocity).
        out_dim:  Node output width (e.g. 2 for predicted velocity).
        hidden_dim: Width of the latent node/edge embeddings.
        n_layers: Number of message-passing layers.
        activation: Activation name (``silu`` / ``gelu`` / ``relu``).
    """

    def __init__(
        self,
        node_dim: int = 2,
        edge_dim: int = 5,
        out_dim: int = 2,
        hidden_dim: int = 32,
        n_layers: int = 3,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.node_dim = int(node_dim)
        self.edge_dim = int(edge_dim)
        self.out_dim = int(out_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.activation = str(activation)

        act = self.activation
        self.node_encoder = _mlp(self.node_dim, self.hidden_dim, self.hidden_dim, act)
        self.edge_encoder = _mlp(self.edge_dim, self.hidden_dim, self.hidden_dim, act)
        self.layers = nn.ModuleList(
            [MeshGraphNetLayer(self.hidden_dim, act) for _ in range(self.n_layers)]
        )
        self.node_decoder = _mlp(self.hidden_dim, self.out_dim, self.hidden_dim, act)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Map node features to node predictions over a graph.

        Args:
            x: ``(N, node_dim)`` node features.
            edge_index: ``(2, E)`` long tensor (row 0 = source, row 1 = dest).
            edge_attr: ``(E, edge_dim)`` edge features.

        Returns:
            ``(N, out_dim)`` node predictions.
        """
        h = self.node_encoder(x)
        e = self.edge_encoder(edge_attr)
        for layer in self.layers:
            h, e = layer(h, edge_index, e)
        return self.node_decoder(h)

    def arch_dict(self) -> dict[str, Any]:
        """Return the architecture hyper-parameters as a plain dict."""
        return {
            "node_dim": self.node_dim,
            "edge_dim": self.edge_dim,
            "out_dim": self.out_dim,
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
            "activation": self.activation,
        }


# ---------------------------------------------------------------------------
# Persistence helpers (mirror tensorlbm.ai.fno.save_fno2d / load_fno2d)
# ---------------------------------------------------------------------------

def save_mesh_gnn(model: MeshGraphNet, path: str | Path) -> Path:
    """Serialize a :class:`MeshGraphNet` to a ``.pt`` file plus JSON metadata."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), p)
    meta = {
        "arch": model.arch_dict(),
        "model_class": "MeshGraphNet",
        "format_version": 1,
    }
    meta_path = p.with_suffix(p.suffix + ".json")
    meta_path.write_text(json.dumps(meta, indent=2))
    return p


def load_mesh_gnn(path: str | Path) -> MeshGraphNet:
    """Load a :class:`MeshGraphNet` saved by :func:`save_mesh_gnn`."""
    p = Path(path)
    blob = torch.load(p, map_location="cpu", weights_only=True)
    meta_path = p.with_suffix(p.suffix + ".json")
    arch_dict: dict[str, Any] = {}
    if meta_path.exists():
        arch_dict = json.loads(meta_path.read_text()).get("arch") or {}
    model = MeshGraphNet(**arch_dict) if arch_dict else MeshGraphNet()
    model.load_state_dict(blob)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Synthetic flow-graph data generation
# ---------------------------------------------------------------------------

def _grid_edges(grid_size: int) -> torch.Tensor:
    """Build undirected 4-connectivity edges over a ``grid_size``×``grid_size`` grid."""
    gs = int(grid_size)
    idx = torch.arange(gs * gs).reshape(gs, gs)
    src = torch.cat([idx[:, :-1].reshape(-1), idx[:-1, :].reshape(-1)])
    dst = torch.cat([idx[:, 1:].reshape(-1), idx[1:, :].reshape(-1)])
    return torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)


def _velocity_field(pos: torch.Tensor, t: float, c: float) -> torch.Tensor:
    """Analytic time-varying 2D velocity field (translating vortex sheet).

    The field is defined by position so a *next-time-step* target is a
    well-defined (and non-trivial) function of the current field: recovering
    it requires estimating spatial gradients from neighbour information,
    which is exactly what message passing supplies.
    """
    x = pos[:, 0]
    y = pos[:, 1]
    u = torch.sin(2.0 * math.pi * (x - c * t)) * torch.cos(2.0 * math.pi * y)
    v = torch.cos(2.0 * math.pi * x) * torch.sin(2.0 * math.pi * (y - c * t))
    return torch.stack([u, v], dim=1)


def _edge_features(pos: torch.Tensor, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Build edge features ``[dx, dy, dist, du, dv]`` from positions/velocities."""
    src, dst = edge_index
    rel = pos[dst] - pos[src]
    dist = rel.norm(dim=1, keepdim=True)
    dvel = x[dst] - x[src]
    return torch.cat([rel, dist, dvel], dim=1)


def _make_flow_graph(
    grid_size: int,
    *,
    c: float,
    t: float,
    dt: float,
) -> dict[str, torch.Tensor]:
    """Build one flow-field graph: current velocity -> next-time-step velocity."""
    gs = int(grid_size)
    coords = torch.linspace(0.0, 1.0, gs)
    gy, gx = torch.meshgrid(coords, coords, indexing="ij")
    pos = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)  # (N, 2)

    x = _velocity_field(pos, t, c)               # node velocity features (N, 2)
    y = _velocity_field(pos, t + dt, c)          # next-time-step target  (N, 2)
    edge_index = _grid_edges(gs)                 # (2, E)
    edge_attr = _edge_features(pos, x, edge_index)  # (E, 5)

    return {
        "x": x,
        "pos": pos,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "y": y,
    }


def _collate(graphs: list[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Merge a batch of graphs into one big graph (block-diagonal edge index).

    Message aggregation never crosses graphs because there are no inter-graph
    edges, so no explicit batch pointer is needed downstream.
    """
    x = torch.cat([g["x"] for g in graphs], dim=0)
    y = torch.cat([g["y"] for g in graphs], dim=0)
    edge_attr = torch.cat([g["edge_attr"] for g in graphs], dim=0)

    edges: list[torch.Tensor] = []
    offset = 0
    for g in graphs:
        edges.append(g["edge_index"] + offset)
        offset += int(g["x"].shape[0])
    edge_index = torch.cat(edges, dim=1)
    return {"x": x, "edge_index": edge_index, "edge_attr": edge_attr, "y": y}


# ---------------------------------------------------------------------------
# The application
# ---------------------------------------------------------------------------

class MeshGNNFlow(AI4SApplication):
    """MeshGraphNet-style GNN flow predictor as an :class:`AI4SApplication`.

    Attributes:
        name: Registry name of the application.
        family: Serving model family (``"gnn"``, understood by
            :class:`tensorlbm.ml.serving.InferenceService`).
    """

    name: str = "mesh_gnn_flow"
    family: str = "gnn"
    version: str = "1.0"

    def __init__(
        self,
        *,
        train_fn: Callable[..., Any] | None = None,
        produce_fn: Callable[..., Any] | None = None,
    ) -> None:
        """Create the app, optionally injecting the data / training steps.

        Args:
            train_fn: Optional override for the training step.  Called as
                ``train_fn(dataset, model, cfg)`` and must return a
                :class:`TrainingResult` (or a mapping with ``model_path`` /
                ``metrics`` / ``arch`` keys).  Defaults to the local Adam+MSE
                loop.
            produce_fn: Optional override for data production.  Called as
                ``produce_fn(grid_size=..., n_graphs=..., c=..., dt=..., t0=...)``
                and must return a list of graph dicts.  Defaults to the
                synthetic flow-field generator.
        """
        super().__init__()
        self._train_fn = train_fn
        self._produce_fn = produce_fn

    # ---- developer-implemented interface --------------------------------

    def produce_data(self, cfg: Mapping[str, Any]) -> DataProduct:
        """Generate 2D flow-graph snapshots and return their metadata.

        Each snapshot carries grid-point coordinates (``pos``), node velocity
        features (``x``), a neighbour ``edge_index``, edge features
        (``edge_attr``) and the next-time-step target (``y``).  The graphs are
        carried in :attr:`DataProduct.metadata["graphs"]` (in-memory).
        """
        grid_size = int(cfg.get("grid_size", 8))
        n_graphs = int(cfg.get("n_graphs", 4))
        c = float(cfg.get("c", 0.1))
        dt = float(cfg.get("dt", 0.05))
        t0 = float(cfg.get("t0", 0.0))

        if self._produce_fn is not None:
            graphs = self._produce_fn(
                grid_size=grid_size, n_graphs=n_graphs, c=c, dt=dt, t0=t0,
            )
        else:
            graphs = [
                _make_flow_graph(grid_size, c=c, t=t0 + k * dt, dt=dt)
                for k in range(n_graphs)
            ]
        if not graphs:
            raise ValueError("flow-graph production returned no graphs")

        first = graphs[0]
        return DataProduct(
            name="2D flow graph snapshots",
            field_name="u",
            shape=tuple(_to_tensor(first["x"]).shape),
            dtype=str(_to_tensor(first["x"]).dtype),
            units="lu",
            metadata={
                "graphs": graphs,
                "n_nodes": int(first["x"].shape[0]),
                "n_edges": int(first["edge_index"].shape[1]),
                "grid_size": grid_size,
                "n_graphs": len(graphs),
                "edge_dim": int(first["edge_attr"].shape[1]),
            },
        )

    def build_model(self, arch: Mapping[str, Any]) -> nn.Module:
        """Construct the :class:`MeshGraphNet` from an arch mapping."""
        return MeshGraphNet(
            node_dim=int(arch.get("node_dim", 2)),
            edge_dim=int(arch.get("edge_dim", 5)),
            out_dim=int(arch.get("out_dim", 2)),
            hidden_dim=int(arch.get("hidden_dim", 32)),
            n_layers=int(arch.get("n_layers", 3)),
            activation=str(arch.get("activation", "silu")),
        )

    def make_dataset(self, product: DataProduct) -> dict[str, Any]:
        """Build a graph dataset from a data product.

        Returns ``{"graphs": [ ... ], "node_dim": ..., "edge_dim": ...,
        "out_dim": ..., "n_samples": ...}``; each graph is a dict with
        ``x`` / ``edge_index`` / ``edge_attr`` / ``y`` / ``pos`` keys.
        """
        graphs = product.metadata.get("graphs")
        if not graphs:
            raise ValueError("DataProduct metadata must carry 'graphs'")

        first = graphs[0]
        return {
            "graphs": list(graphs),
            "node_dim": int(first["x"].shape[1]),
            "edge_dim": int(first["edge_attr"].shape[1]),
            "out_dim": int(first["y"].shape[1]),
            "n_samples": len(graphs),
        }

    def train(
        self,
        dataset: Any,
        model: nn.Module,
        cfg: Mapping[str, Any],
    ) -> TrainingResult:
        """Train the GNN and return weights path + metrics.

        The injectable ``train_fn`` (if provided) is called as
        ``train_fn(dataset, model, cfg)``; otherwise the local Adam+MSE loop
        runs and the checkpoint is written with :func:`save_mesh_gnn`.
        """
        out_path = Path(
            str(cfg.get("out_path") or cfg.get("model_path") or "mesh_gnn_model.pt"),
        )

        if self._train_fn is not None:
            return _coerce_training_result(
                self._train_fn(dataset, model, cfg),
                out_path=out_path,
            )

        result = _train_gnn(
            dataset,
            cast(MeshGraphNet, model),
            out_path,
            epochs=int(cfg.get("epochs", 20)),
            batch_size=int(cfg.get("batch_size", 4)),
            learning_rate=float(cfg.get("learning_rate", 1e-3)),
            seed=int(cfg.get("seed", 0)),
            device=str(cfg.get("device", "cpu")),
        )
        arch = dict(result.get("arch") or {})
        if not arch:
            arch = model.arch_dict()
        return TrainingResult(
            model_path=str(result.get("path", out_path)),
            metrics={"train_loss": float(result.get("train_loss", float("nan")))},
            arch=arch,
        )

    def infer(self, model: nn.Module, sample: Any) -> Prediction:
        """Run the GNN over a graph sample.

        ``sample`` may be a dict (``{"x", "edge_index", ["edge_attr"]}``) or a
        ``(x, edge_index[, edge_attr])`` tuple.  Missing ``edge_attr`` is
        replaced with zeros.
        """
        x, edge_index, edge_attr = _unpack_sample(sample, model)
        model.eval()
        with torch.no_grad():
            out = model(x, edge_index, edge_attr)
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

def _train_gnn(
    dataset: Mapping[str, Any],
    model: MeshGraphNet,
    out_path: Path,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: str,
) -> dict[str, Any]:
    """Adam + MSE training loop over graph mini-batches (CPU-friendly)."""
    graphs = dataset["graphs"]
    torch_device = torch.device(device)
    torch.manual_seed(int(seed))
    model.to(torch_device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    loss_fn = nn.MSELoss()
    n = len(graphs)
    batch_size = max(1, int(batch_size))

    final_loss = float("nan")
    for _ in range(max(0, int(epochs))):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        count = 0
        for i in range(0, n, batch_size):
            idxs = perm[i : i + batch_size]
            batch = _collate([graphs[j] for j in idxs.tolist()])
            xb = batch["x"].to(torch_device)
            eb = batch["edge_index"].to(torch_device)
            ab = batch["edge_attr"].to(torch_device)
            yb = batch["y"].to(torch_device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb, eb, ab), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idxs)
            count += len(idxs)
        final_loss = epoch_loss / max(1, count)

    model.eval()
    path = save_mesh_gnn(model, out_path)
    return {
        "path": str(path),
        "train_loss": final_loss,
        "arch": model.arch_dict(),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unpack_sample(
    sample: Any,
    model: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalise an inference sample into ``(x, edge_index, edge_attr)``."""
    edge_dim = int(getattr(model, "edge_dim", 5))

    if isinstance(sample, (tuple, list)):
        x = _to_tensor(sample[0])
        edge_index = sample[1].to(torch.long)
        edge_attr = (
            _to_tensor(sample[2])
            if len(sample) >= 3
            else torch.zeros(int(edge_index.shape[1]), edge_dim)
        )
        return x, edge_index, edge_attr

    if isinstance(sample, Mapping):
        x = _to_tensor(sample.get("x", sample.get("nodes")))
        edge_index = sample["edge_index"].to(torch.long)
        edge_attr = sample.get("edge_attr")
        edge_attr = (
            _to_tensor(edge_attr)
            if edge_attr is not None
            else torch.zeros(int(edge_index.shape[1]), edge_dim)
        )
        return x, edge_index, edge_attr

    raise TypeError(
        "sample must be a graph dict or (x, edge_index[, edge_attr]) tuple, "
        f"got {type(sample).__name__}",
    )


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
                )
            },
            arch=dict(result.get("arch") or {}),
        )
    raise TypeError(
        "train_fn must return a TrainingResult or a mapping with "
        f"'model_path'/'metrics'/'arch' keys, got {type(result).__name__}",
    )
