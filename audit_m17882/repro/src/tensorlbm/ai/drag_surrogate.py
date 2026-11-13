"""Field-to-drag surrogate modelling: predict C_D from LBM plane snapshots.

Pairs a scan campaign's exported field snapshots (``tensorlbm.solver-export/v1``,
one ``fields.h5`` per point) with exact drag labels from the control-volume
observer (``tensorlbm.drag-history/v1`` sidecars / ``status.json:
drag_mean_tail``, see :mod:`tensorlbm.scan_drag`), joined on the Reynolds
number.  The labels are the *measured* C_D of PR #204 — not the wake-survey
estimator.

The surrogate (:class:`FNODragRegressor`) reuses the Fourier layer of
:class:`tensorlbm.ai.fno.SpectralConv2d` with a global-pool scalar head: a
plane snapshot ``(C, ny, nx)`` → C_D.  Two cheap baselines are provided so the
surrogate's value is measured against the strongest prior on a single-parameter
sweep — a power-law fit ``C_D = a·Re^b`` (the physics scaling) and the
train-mean.

With ``N`` scan points per split this is a small-``N`` regime: snapshots from
the same point share one trajectory and one label, so effective sample size is
the number of *points*, not (point, step) rows.  Keep ``PlaneSampleSpec.steps``
short (one step per point is the clean choice) and read metrics accordingly.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .fno import SpectralConv2d, _get_activation

STUDY_SCHEMA = "tensorlbm.drag-surrogate-study/v1"

DEFAULT_CHANNELS: tuple[str, ...] = ("ux", "uy", "rho")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlaneSampleSpec:
    """How to turn a 3-D snapshot into plane samples for the surrogate."""

    plane: int | str = "center"
    """Lateral (z) index of the extracted plane, or ``"center"`` for nz // 2."""

    channels: tuple[str, ...] = DEFAULT_CHANNELS
    """Field datasets read from each snapshot group (e.g. ux, uy, rho)."""

    steps: tuple[int, ...] = (4000,)
    """Snapshot steps used as samples; one (point, step) row per entry."""

    velocity_scale: bool = False
    """Divide velocity channels (ux/uy/uz) by the point's ``u_in``.

    Scale-invariant inputs: at fixed u_in the raw velocity magnitude is a
    perfect Re proxy (the v1 pilot exploited this without saying so), so a
    surrogate trained on raw channels learns the velocity *scale*, not the
    wake *shape*, and fails to transfer across u_in levels (59.6 % MAPE on
    the Re × u_in grid vs 5.4 % after retraining; see
    ``docs/drag_surrogate_fno_20260822.md`` v1.1). Default keeps raw
    channels; labels are unaffected either way.
    """

    param_names: tuple[str, ...] = ()
    """Per-point scalar conditioning parameters read from snapshot attrs.

    Geometry-axis campaigns store e.g. ``sail_scale`` / ``fin_scale`` as
    snapshot attributes; naming them here feeds them to the surrogate head
    as extra scalar inputs (log10 + train-split standardisation, see
    :class:`ParamNorm`).  The wake plane alone may under-determine these
    (similar wakes, different appendage scales at the resolution of the
    plane sample), so conditioning is the explicit route.  Empty (default)
    keeps the plane-only model of B1-ML/B1-3 unchanged.
    """


@dataclass
class DragSplit:
    """A materialised train/val/test split of plane samples."""

    x: np.ndarray  # (N, C, ny, nx) float32
    cd: np.ndarray  # (N,) float64 — exact C_D (per point, repeated per step)
    re: np.ndarray  # (N,) float64
    u_in: np.ndarray | None = None  # (N,) float64, from snapshot attrs
    params: np.ndarray | None = None  # (N, K) raw attr values, columns = spec.param_names
    param_names: tuple[str, ...] = ()  # column labels for ``params``
    point_id: list[str] = field(default_factory=list)
    step: list[int] = field(default_factory=list)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return int(self.x.shape[0])


def iter_point_ids(fields_dir: str | Path) -> tuple[str, ...]:
    """Sorted point ids under ``<fields_dir>/points`` that export fields."""
    root = Path(fields_dir) / "points"
    return tuple(
        p.name for p in sorted(root.iterdir()) if p.is_dir() and (p / "fields.h5").is_file()
    )


def read_plane_snapshot(
    h5_path: str | Path, step: int, channels: tuple[str, ...], plane: int | str
) -> np.ndarray:
    """Read one snapshot's channels at a lateral plane → ``(C, ny, nx)``.

    Raises:
        KeyError: if ``step`` is not an exported snapshot (message lists the
            available steps).
    """
    import h5py

    with h5py.File(h5_path, "r") as f:
        key = f"step_{step:06d}"
        if key not in f:
            available = sorted(int(k.removeprefix("step_")) for k in f.keys())
            raise KeyError(f"no snapshot at step {step} in {h5_path}; have {available}")
        plane_idx = f[key]["ux"].shape[0] // 2 if plane == "center" else int(plane)
        return np.stack(
            [np.asarray(f[key][name][plane_idx], dtype=np.float32) for name in channels]
        )


def load_exact_cd(
    drag_dir: str | Path,
    fields_dir: str | Path,
    *,
    tail_fraction: float = 0.25,
    rho: float = 1.0,
) -> dict[float, float]:
    """Exact C_D per Reynolds number from a drag-observer campaign.

    ``force_x`` tail mean (last ``tail_fraction`` of samples; falls back to
    ``status.json:drag_mean_tail``) normalised by ``rho * u_in^2 * S_proj``.
    ``u_in`` comes from the paired fields campaign's snapshot attributes and
    ``S_proj`` from its solid mask (:func:`tensorlbm.drag_survey.projected_area`),
    matching the benchmark convention.

    Raises:
        ValueError: if two points share a Reynolds number — use
            :func:`load_exact_cd_per_point` for multi-parameter campaigns
            where Re repeats across points.
    """
    per_point = load_exact_cd_per_point(drag_dir, fields_dir, tail_fraction=tail_fraction, rho=rho)
    out: dict[float, float] = {}
    for pd, cd in per_point.items():
        re = _point_re(drag_dir, pd)
        if re in out:
            raise ValueError(f"duplicate Re {re} in {drag_dir}")
        out[re] = cd
    return out


def load_exact_cd_per_point(
    drag_dir: str | Path,
    fields_dir: str | Path,
    *,
    tail_fraction: float = 0.25,
    rho: float = 1.0,
) -> dict[str, float]:
    """Exact C_D per point id — the join that works when Re repeats.

    Same physics as :func:`load_exact_cd`, keyed by ``point_id`` so
    multi-parameter campaigns (e.g. a Re × u_in grid) keep one label per
    point instead of collapsing onto the last point at each Re.
    """
    import h5py

    from ..drag_survey import projected_area

    drag_root = Path(drag_dir) / "points"
    fields_root = Path(fields_dir) / "points"
    out: dict[str, float] = {}
    for pd in sorted(drag_root.iterdir()):
        if not pd.is_dir() or not (pd / "status.json").is_file():
            continue
        status = json.loads((pd / "status.json").read_text())
        history_path = pd / "drag_history.json"
        if history_path.is_file():
            samples = json.loads(history_path.read_text())["samples"]
            fx = np.asarray([s["force_x"] for s in samples], dtype=np.float64)
            tail = fx[int(len(fx) * (1.0 - tail_fraction)) :]
            force = float(tail.mean())
        else:
            force = float(status["drag_mean_tail"])

        fh5 = fields_root / pd.name / "fields.h5"
        with h5py.File(fh5, "r") as f:
            last_key = sorted(f.keys())[-1]
            u_in = float(f[last_key].attrs["u_in"])
            mask = np.asarray(f[last_key]["solid_mask"])
        out[pd.name] = 2.0 * force / (rho * u_in**2 * projected_area(mask))
    return out


def _point_re(drag_dir: str | Path, point_id: str) -> float:
    return float(
        json.loads((Path(drag_dir) / "points" / point_id / "status.json").read_text())["params"][
            "re"
        ]
    )


def build_drag_split(
    fields_dir: str | Path,
    cd_by_re: dict[float, float] | None = None,
    point_ids: tuple[str, ...] | list[str] = (),
    spec: PlaneSampleSpec | None = None,
    *,
    cd_by_point: dict[str, float] | None = None,
) -> DragSplit:
    """Materialise ``(point, step)`` plane samples with exact-C_D labels.

    Labels join either by point id (``cd_by_point`` — required when Re
    repeats across points, e.g. a Re × u_in grid) or by Reynolds number
    (``cd_by_re``, single-parameter campaigns). With
    ``spec.velocity_scale`` the velocity channels are divided by the
    point's ``u_in`` (scale-invariant inputs); ``split.u_in`` carries the
    per-row inflow velocity either way.  ``spec.param_names`` collects the
    named snapshot attrs per point into ``split.params`` (geometry
    conditioning, see :class:`ParamNorm`).
    """
    if cd_by_re is None and cd_by_point is None:
        raise ValueError("provide cd_by_re or cd_by_point")
    spec = spec or PlaneSampleSpec()
    vel_idx = tuple(i for i, ch in enumerate(spec.channels) if ch in ("ux", "uy", "uz"))
    root = Path(fields_dir) / "points"
    xs, cds, res, u_ins, pids, steps, prm = [], [], [], [], [], [], []
    for pid in point_ids:
        h5 = root / pid / "fields.h5"
        with_batch = read_plane_snapshot(h5, spec.steps[0], spec.channels, spec.plane)
        re = float(_snapshot_attr(h5, "re"))
        u_in = float(_snapshot_attr(h5, "u_in"))
        if spec.param_names:
            prm.append([float(_snapshot_attr(h5, name)) for name in spec.param_names])
        if spec.velocity_scale and vel_idx:
            with_batch = with_batch.copy()
            with_batch[list(vel_idx)] /= np.float32(u_in)
        if cd_by_point is not None:
            if pid not in cd_by_point:
                raise KeyError(f"no exact C_D for point {pid}")
            cd = cd_by_point[pid]
        else:
            if re not in cd_by_re:
                raise KeyError(f"no exact C_D for Re={re} (point {pid})")
            cd = cd_by_re[re]
        for step in spec.steps:
            plane = (
                with_batch
                if step == spec.steps[0]
                else read_plane_snapshot(h5, step, spec.channels, spec.plane)
            )
            if spec.velocity_scale and vel_idx and step != spec.steps[0]:
                plane = plane.copy()
                plane[list(vel_idx)] /= np.float32(u_in)
            xs.append(plane)
            cds.append(cd)
            res.append(re)
            u_ins.append(u_in)
            pids.append(pid)
            steps.append(step)
    return DragSplit(
        x=np.stack(xs),
        cd=np.asarray(cds, dtype=np.float64),
        re=np.asarray(res, dtype=np.float64),
        u_in=np.asarray(u_ins, dtype=np.float64),
        params=np.asarray(prm, dtype=np.float64) if prm else None,
        param_names=tuple(spec.param_names),
        point_id=pids,
        step=steps,
    )


def _snapshot_attr(h5_path: str | Path, name: str) -> float:
    import h5py

    with h5py.File(h5_path, "r") as f:
        key = sorted(f.keys())[0]
        return float(f[key].attrs[name])


# ---------------------------------------------------------------------------
# Normalisation + torch dataset
# ---------------------------------------------------------------------------


@dataclass
class DragNorm:
    """Train-split statistics; everything downstream un-normalises through it."""

    channel_mean: list[float]
    channel_std: list[float]
    target_mean: float
    target_std: float
    transform: str = "log10"  # "log10" | "identity"

    def encode_target(self, cd: np.ndarray) -> np.ndarray:
        y = np.log10(cd) if self.transform == "log10" else np.asarray(cd, dtype=np.float64)
        return (y - self.target_mean) / self.target_std

    def decode_target(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64) * self.target_std + self.target_mean
        return 10.0**y if self.transform == "log10" else y


def fit_norm(split: DragSplit, transform: str = "log10") -> DragNorm:
    """Fit channel/target statistics on a (train) split only."""
    c_mean = split.x.reshape(split.x.shape[0], split.x.shape[1], -1).mean(axis=(0, 2))
    c_std = split.x.reshape(split.x.shape[0], split.x.shape[1], -1).std(axis=(0, 2))
    y = np.log10(split.cd) if transform == "log10" else split.cd
    return DragNorm(
        channel_mean=[float(v) for v in c_mean],
        channel_std=[float(max(v, 1e-8)) for v in c_std],
        target_mean=float(y.mean()),
        target_std=float(max(y.std(), 1e-8)),
        transform=transform,
    )


@dataclass
class ParamNorm:
    """Train-split statistics for the scalar conditioning parameters.

    Each column is log10-transformed (all conditioning params are positive
    scale factors — sail/fin scale, u_in, ...; the log map is what makes
    the standardisation uniform across magnitudes) and then standardised.
    """

    mean: list[float]  # per column, of log10(params)
    std: list[float]  # per column, of log10(params)
    names: tuple[str, ...] = ()

    def encode(self, params: np.ndarray) -> np.ndarray:
        p = np.log10(np.asarray(params, dtype=np.float64))
        return (p - np.asarray(self.mean)) / np.asarray(self.std)

    def decode(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64) * np.asarray(self.std) + np.asarray(self.mean)
        return 10.0**z


def fit_param_norm(split: DragSplit) -> ParamNorm | None:
    """Fit :class:`ParamNorm` on a (train) split; ``None`` if no params."""
    if split.params is None:
        return None
    p = np.log10(split.params)
    return ParamNorm(
        mean=[float(v) for v in p.mean(axis=0)],
        std=[float(max(v, 1e-8)) for v in p.std(axis=0)],
        names=tuple(split.param_names),
    )


class DragPlaneDataset(Dataset):
    """Normalised (plane, target) pairs; targets are standardised C_D.

    When ``split.params`` is set (geometry conditioning) items are
    ``(plane, params, target)`` with params standardised by *pnorm*
    (fit on the train split); otherwise the B1-ML/B1-3 two-tuple.
    """

    def __init__(self, split: DragSplit, norm: DragNorm, pnorm: ParamNorm | None = None) -> None:
        mean = torch.as_tensor(norm.channel_mean).view(1, -1, 1, 1)
        std = torch.as_tensor(norm.channel_std).view(1, -1, 1, 1)
        self.x = torch.as_tensor((split.x - mean.numpy()) / std.numpy(), dtype=torch.float32)
        self.y = torch.as_tensor(norm.encode_target(split.cd), dtype=torch.float32)
        self.cd = split.cd
        self.re = split.re
        self.p = (
            torch.as_tensor(pnorm.encode(split.params), dtype=torch.float32)
            if split.params is not None and pnorm is not None
            else None
        )

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        if self.p is not None:
            return self.x[idx], self.p[idx], self.y[idx]
        return self.x[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FNODragArch:
    """Hyper-parameters for :class:`FNODragRegressor`."""

    in_channels: int = 3
    width: int = 32
    n_layers: int = 4
    modes_y: int = 12
    modes_x: int = 12
    mlp_hidden: int = 128
    activation: str = "gelu"
    n_params: int = 0
    """Scalar conditioning inputs concatenated after pooling (B1-7)."""


class FNODragRegressor(nn.Module):
    """Fourier-encoder plane → scalar regressor (FNO2d body + pool head).

    Maps ``(B, in_channels, ny, nx)`` plane snapshots to one scalar per sample
    (standardised C_D) via the FNO2d Fourier-layer stack followed by global
    spatial mean pooling and a two-layer MLP head.  With
    ``arch.n_params > 0`` a ``(B, n_params)`` conditioning vector
    (standardised ``ParamNorm`` output) is concatenated between pooling and
    the head — the geometry-aware variant of B1-7.
    """

    def __init__(self, arch: FNODragArch | None = None) -> None:
        super().__init__()
        self.arch = arch or FNODragArch()
        a = self.arch
        self._act = _get_activation(a.activation)
        self.lift = nn.Conv2d(a.in_channels, a.width, kernel_size=1)
        self.spectral = nn.ModuleList(
            [SpectralConv2d(a.width, a.width, a.modes_y, a.modes_x) for _ in range(a.n_layers)]
        )
        self.pointwise = nn.ModuleList(
            [nn.Conv2d(a.width, a.width, kernel_size=1) for _ in range(a.n_layers)]
        )
        self.head = nn.Sequential(
            nn.Linear(a.width + a.n_params, a.mlp_hidden),
            nn.GELU(),
            nn.Linear(a.mlp_hidden, 1),
        )

    def forward(self, x: torch.Tensor, p: torch.Tensor | None = None) -> torch.Tensor:
        x = self.lift(x)
        for spec, pw in zip(self.spectral, self.pointwise):
            x = self._act(spec(x) + pw(x))
        x = x.mean(dim=(2, 3))  # global pool over the plane
        if self.arch.n_params:
            if p is None:
                raise ValueError("arch.n_params > 0 requires conditioning input p")
            x = torch.cat([x, p.to(x.dtype)], dim=1)
        return self.head(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------


@dataclass
class DragTrainConfig:
    epochs: int = 400
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 60
    seed: int = 0
    device: str | None = None  # None → cuda if available else cpu


@dataclass
class DragTrainResult:
    model: FNODragRegressor
    norm: DragNorm
    history: dict[str, list[float]]
    best_epoch: int
    pnorm: ParamNorm | None = None  # set when geometry conditioning was used


def _resolve_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_drag_surrogate(
    train_split: DragSplit,
    val_split: DragSplit,
    arch: FNODragArch | None = None,
    config: DragTrainConfig | None = None,
    transform: str = "log10",
) -> DragTrainResult:
    """Train the surrogate; normalisation is fitted on ``train_split`` only."""
    arch = arch or FNODragArch(in_channels=train_split.x.shape[1])
    config = config or DragTrainConfig()
    if arch.in_channels != train_split.x.shape[1]:
        raise ValueError(
            f"arch.in_channels={arch.in_channels} != data channels {train_split.x.shape[1]}"
        )
    torch.manual_seed(config.seed)
    norm = fit_norm(train_split, transform)
    pnorm = fit_param_norm(train_split)
    if pnorm is not None and arch.n_params != train_split.params.shape[1]:
        raise ValueError(
            f"arch.n_params={arch.n_params} != conditioning columns {train_split.params.shape[1]}"
        )
    if pnorm is None and arch.n_params:
        raise ValueError("arch.n_params > 0 but the splits carry no params (spec.param_names?)")
    train_ds = DragPlaneDataset(train_split, norm, pnorm)
    val_ds = DragPlaneDataset(val_split, norm, pnorm)
    device = _resolve_device(config.device)
    model = FNODragRegressor(arch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    batch = min(config.batch_size, len(train_ds))
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=min(config.batch_size, len(val_ds)))

    def _batch_loss(batch) -> torch.Tensor:
        xb, pb, yb = batch if len(batch) == 3 else (batch[0], None, batch[1])
        pred = model(xb.to(device), pb.to(device) if pb is not None else None)
        return nn.functional.mse_loss(pred, yb.to(device))

    history: dict[str, list[float]] = {"train": [], "val": []}
    best_val, best_epoch, best_state = float("inf"), -1, None
    for epoch in range(config.epochs):
        model.train()
        losses = []
        for batch in train_loader:
            loss = _batch_loss(batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            val_losses = [float(_batch_loss(batch)) for batch in val_loader]
        history["train"].append(float(np.mean(losses)))
        history["val"].append(float(np.mean(val_losses)))
        if history["val"][-1] < best_val - 1e-12:
            best_val, best_epoch = history["val"][-1], epoch
            best_state = {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= config.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return DragTrainResult(
        model=model, norm=norm, history=history, best_epoch=best_epoch, pnorm=pnorm
    )


@torch.no_grad()
def predict_cd(
    model: FNODragRegressor,
    split: DragSplit,
    norm: DragNorm,
    pnorm: ParamNorm | None = None,
    *,
    device: str | None = None,
    batch_size: int = 64,
) -> np.ndarray:
    """Predict raw (un-normalised) C_D for a split.

    ``pnorm`` is required when the model was trained with conditioning
    (``arch.n_params > 0``); it comes back from :func:`train_drag_surrogate`
    or :func:`load_drag_regressor`.
    """
    device = _resolve_device(device)
    model = model.to(device).eval()
    ds = DragPlaneDataset(split, norm, pnorm)
    loader = DataLoader(ds, batch_size=min(batch_size, len(ds)))
    ys = []
    for batch in loader:
        xb, pb = (batch[0], batch[1]) if len(batch) == 3 else (batch[0], None)
        ys.append(model(xb.to(device), pb.to(device) if pb is not None else None).cpu().numpy())
    return norm.decode_target(np.concatenate(ys))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MAE / RMSE / R² / MAPE between true and predicted C_D values."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "mape": float(np.mean(np.abs(err) / np.abs(y_true)) * 100.0),
        "n": int(y_true.size),
    }


def power_law_fit(re: np.ndarray, cd: np.ndarray) -> tuple[float, float]:
    """Least-squares power law ``C_D = a·Re^b`` → ``(log10_a, b)``."""
    exponent, log10_a = np.polyfit(
        np.log10(np.asarray(re, dtype=np.float64)), np.log10(np.asarray(cd, dtype=np.float64)), 1
    )
    return float(log10_a), float(exponent)


def power_law_predict(coeffs: tuple[float, float], re: np.ndarray) -> np.ndarray:
    """Evaluate a :func:`power_law_fit` result at ``re``."""
    log10_a, exponent = coeffs
    return 10.0 ** (log10_a + exponent * np.log10(np.asarray(re, dtype=np.float64)))


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_drag_regressor(
    model: FNODragRegressor,
    norm: DragNorm,
    path: str | Path,
    pnorm: ParamNorm | None = None,
) -> Path:
    """Serialize model weights + arch + normalisations (``.pt`` + ``.pt.json``)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), p)
    meta = {
        "model_class": "FNODragRegressor",
        "arch": asdict(model.arch),
        "norm": asdict(norm),
        "param_norm": asdict(pnorm) if pnorm is not None else None,
        "format_version": 1,
    }
    p.with_suffix(p.suffix + ".json").write_text(json.dumps(meta, indent=2))
    return p


def load_drag_regressor(path: str | Path) -> tuple[FNODragRegressor, DragNorm, ParamNorm | None]:
    """Load a surrogate saved by :func:`save_drag_regressor` (eval mode).

    Returns ``(model, norm, pnorm)``; ``pnorm`` is ``None`` for artifacts
    without conditioning (the pre-B1-7 format).
    """
    p = Path(path)
    meta = json.loads(p.with_suffix(p.suffix + ".json").read_text())
    model = FNODragRegressor(FNODragArch(**meta["arch"]))
    model.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))
    model.eval()
    raw = meta.get("param_norm")
    pnorm = ParamNorm(**raw) if raw else None
    return model, DragNorm(**meta["norm"]), pnorm


# ---------------------------------------------------------------------------
# Study orchestrator
# ---------------------------------------------------------------------------


def run_drag_surrogate_study(
    fields_dir: str | Path,
    drag_dir: str | Path,
    out_dir: str | Path,
    *,
    spec: PlaneSampleSpec | None = None,
    arch: FNODragArch | None = None,
    config: DragTrainConfig | None = None,
    transform: str = "log10",
) -> dict:
    """End-to-end study: splits → train → FNO vs baselines → artefacts.

    Splits come from ``<fields_dir>/dataset.json:split_points`` (the campaign's
    own point-level split, so no test point is ever seen in training or in the
    normalisation statistics).  Writes ``model.pt(.json)``, ``metrics.json``
    and ``predictions.csv`` under ``out_dir``; returns the metrics summary.
    """
    spec = spec or PlaneSampleSpec()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    split_points = json.loads((Path(fields_dir) / "dataset.json").read_text())["split_points"]
    cd_by_point = load_exact_cd_per_point(drag_dir, fields_dir)
    splits = {
        name: build_drag_split(fields_dir, point_ids=pids, spec=spec, cd_by_point=cd_by_point)
        for name, pids in split_points.items()
    }
    result = train_drag_surrogate(splits["train"], splits["val"], arch, config, transform)
    save_drag_regressor(result.model, result.norm, out / "model.pt", result.pnorm)

    coeffs = power_law_fit(splits["train"].re, splits["train"].cd)
    cd_mean = float(splits["train"].cd.mean())
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    predictions: dict[str, dict[str, np.ndarray]] = {}
    for name, split in splits.items():
        preds = {
            "fno": predict_cd(result.model, split, result.norm, result.pnorm),
            "power_law": power_law_predict(coeffs, split.re),
            "mean": np.full(len(split), cd_mean),
        }
        predictions[name] = preds
        metrics[name] = {k: regression_metrics(split.cd, v) for k, v in preds.items()}

    param_cols = list(spec.param_names)
    with open(out / "predictions.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "split",
                "point_id",
                "step",
                "re",
                *param_cols,
                "cd_true",
                "cd_fno",
                "cd_power_law",
                "cd_mean",
            ]
        )
        for name, split in splits.items():
            for i, pid in enumerate(split.point_id):
                writer.writerow(
                    [
                        name,
                        pid,
                        split.step[i],
                        float(split.re[i]),
                        *(float(split.params[i, j]) for j in range(len(param_cols))),
                        float(split.cd[i]),
                        float(predictions[name]["fno"][i]),
                        float(predictions[name]["power_law"][i]),
                        cd_mean,
                    ]
                )

    summary = {
        "schema": STUDY_SCHEMA,
        "fields_dir": str(fields_dir),
        "drag_dir": str(drag_dir),
        "spec": asdict(spec),
        "arch": asdict(result.model.arch),
        "config": asdict(config or DragTrainConfig()),
        "power_law": {"log10_a": coeffs[0], "exponent": coeffs[1]},
        "param_norm": asdict(result.pnorm) if result.pnorm else None,
        "best_epoch": result.best_epoch,
        "n_points": {k: len(v.point_id) for k, v in splits.items()},
        "metrics": metrics,
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    return summary
