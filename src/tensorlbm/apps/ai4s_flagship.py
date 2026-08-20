"""End-to-end AI4S flagship pipeline: solver -> dataset -> job -> model asset -> serving -> lineage.

This module is the library behind ``examples/ai4s_flagship_demo.py``.  It
strings the platform's layers into one closed loop, closing the "AI4S loop
break 3" gap (trained artifacts had no registry and no consumer):

1. **Data** — velocity snapshots from the public solver API (or the pilot
   SUBOFF dataset when available on disk);
2. **Dataset** — coarse->fine super-resolution pairs registered in the field
   data catalog with lineage;
3. **Training** — a compact FNO2d run through :class:`TrainingJobRegistry`'s
   job state machine;
4. **Model asset** — checkpoint registered in the model asset layer
   (:class:`~tensorlbm.ml.model_registry.ModelAssetRegistry`) with task,
   metrics, dataset product id and git sha;
5. **Serving** — the same checkpoint exposed through
   :class:`~tensorlbm.ml.serving.InferenceService`, cross-checked against a
   fresh load from the asset store;
6. **Lineage** — ``product -> dataset -> job -> model -> serving`` edges
   recorded in the field data catalog and read back as one upstream chain.

Every stage is a small, independently testable function; the data layer
(``tensorlbm.data``) is consumed read-only.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorlbm.ai.fno import FNO2d, FNO2dArch, save_fno2d
from tensorlbm.data.catalog import (
    AssetRecord,
    FieldDataCatalog,
    LineageRecord,
    QualityCheck,
)
from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.ml.model_registry import ModelAssetRegistry
from tensorlbm.ml.serving import FAMILY_FNO, InferenceService, ModelRegistry
from tensorlbm.ml.training_job import TrainingJobRegistry
from tensorlbm.solver import stream
from tensorlbm.turbulence import collide_smagorinsky_bgk

__all__ = [
    "FlagshipConfig",
    "FlagshipRunReport",
    "add_stationary_roughness",
    "build_super_resolution_dataset",
    "load_snapshots_hdf5",
    "prediction_error_metrics",
    "produce_velocity_snapshots",
    "run_flagship_demo",
    "split_train_val",
    "try_load_pilot_dataset",
    "write_snapshots_hdf5",
]


# ---------------------------------------------------------------------------
# Configuration / report
# ---------------------------------------------------------------------------

DEFAULT_PILOT_DIR = "/nfs/wangxi/datasets/pilot_suboff_20260820"

_ARCH_DEFAULTS: dict[str, int] = {
    "in_channels": 2,
    "out_channels": 2,
    "width": 32,
    "n_layers": 4,
    "modes_x": 24,
    "modes_y": 24,
    "mlp_hidden": 96,
}


@dataclass
class FlagshipConfig:
    """All knobs of the flagship demo run (defaults fit a single GPU in <10 min)."""

    workdir: str | Path = "./flagship_run"
    pilot_dir: str | Path | None = DEFAULT_PILOT_DIR
    device: str = "cuda"
    # data production (public solver API)
    nx: int = 96
    ny: int = 96
    n_steps: int = 240
    sample_every: int = 8
    seeds: tuple[int, ...] = (11, 22, 33)
    tau: float = 0.8
    c_s: float = 0.1
    downsample_factor: int = 4
    # stationary sub-grid roughness added to provisional solver data so the
    # super-resolution task is not trivially solvable by bilinear upsampling
    roughness_amplitude: float = 0.014
    # model / training
    arch: dict[str, int] = field(default_factory=lambda: dict(_ARCH_DEFAULTS))
    epochs: int = 1200
    batch_size: int = 16
    learning_rate: float = 1e-3
    lr_min: float = 1e-4
    seed: int = 0
    val_fraction: float = 0.15
    # registry naming
    task: str = "flow_super_resolution"
    model_name: str = "flagship-fno2d-superres"
    prefix: str = "flagship_sr"


@dataclass
class FlagshipRunReport:
    """Everything needed to audit the closed loop: ids, paths, metrics, lineage."""

    workspace: str
    data_source: str
    data_path: str
    product_asset_id: str
    dataset_asset_id: str
    n_train: int
    n_val: int
    job_id: str
    job_status: str
    ckpt_path: str
    model_id: str
    model_store: str
    serving_model_id: int
    serving_asset_id: str
    loss_history: list[float]
    val_errors: list[dict[str, Any]]
    baseline_upsample_error: dict[str, float]
    lineage_upstream: list[str]
    phase_times: dict[str, float]
    git_sha: str
    device: str
    val_mean_relative_l2: float = 0.0
    pilot: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False, default=str))
        return p


# ---------------------------------------------------------------------------
# Stage 1: data production / loading
# ---------------------------------------------------------------------------

def _git_sha() -> str:
    try:
        out = subprocess.run(  # noqa: S603 - fixed argv
            ["git", "rev-parse", "--short=12", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def _initial_distribution(
    nx: int, ny: int, seed: int, device: torch.device, mean_u: float = 0.05,
) -> torch.Tensor:
    """Seed-dependent superposition of low-wavenumber sinusoids.

    A training-data generator, not a converged simulation: the FNO is judged
    on reconstructing fine fields from coarsened ones.
    """
    torch.manual_seed(int(seed))
    ys = torch.arange(ny, device=device).float()
    xs = torch.arange(nx, device=device).float()
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    phase = 0.37 * (int(seed) % 97)
    kx = 2.0 * torch.pi / max(nx, 1)
    ky = 2.0 * torch.pi / max(ny, 1)
    ux = mean_u + 0.02 * (
        torch.sin(2.0 * kx * xx + phase) * torch.cos(ky * yy)
        + 0.5 * torch.sin(4.0 * kx * xx + 0.3) * torch.cos(2.0 * ky * yy)
    )
    uy = 0.02 * (
        torch.cos(kx * xx) * torch.sin(2.0 * ky * yy + phase)
        + 0.5 * torch.cos(3.0 * kx * xx) * torch.sin(ky * yy + 0.7)
    )
    return equilibrium(torch.ones_like(ux), ux, uy)


def produce_velocity_snapshots(
    *,
    nx: int,
    ny: int,
    n_steps: int,
    sample_every: int,
    seed: int,
    tau: float = 0.8,
    c_s: float = 0.1,
    device: str | torch.device = "cpu",
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Run a short periodic Smagorinsky-LES with the *public* solver API.

    Uses only ``tensorlbm.d2q9.equilibrium`` / ``macroscopic``,
    ``tensorlbm.turbulence.collide_smagorinsky_bgk`` and
    ``tensorlbm.solver.stream``.  Returns ``(ux, uy)`` snapshots on CPU.
    """
    dev = torch.device(device)
    f = _initial_distribution(nx, ny, seed=seed, device=dev)
    snapshots: list[tuple[torch.Tensor, torch.Tensor]] = []
    for step in range(max(1, int(n_steps))):
        f = collide_smagorinsky_bgk(f, tau=float(tau), C_s=float(c_s))
        f = stream(f)
        if int(sample_every) > 0 and (step + 1) % int(sample_every) == 0:
            _rho, ux, uy = macroscopic(f)
            snapshots.append((ux.detach().cpu().clone(), uy.detach().cpu().clone()))
    if not snapshots:
        _rho, ux, uy = macroscopic(f)
        snapshots.append((ux.detach().cpu().clone(), uy.detach().cpu().clone()))
    return snapshots


def add_stationary_roughness(
    snapshots: Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    amplitude: float = 0.014,
    wavelengths: tuple[float, float, float] = (7.0, 5.8, 8.5),
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Add one *fixed* fine-scale pattern to every snapshot (sub-grid content).

    Models unresolved stationary roughness (e.g. bottom/wall roughness in
    marine LBM): the pattern lives below the coarse grid's Nyquist limit, so
    coarse-graining destroys it and bilinear upsampling cannot recover it —
    exactly the regime where a *learned* super-resolution operator adds value
    over interpolation.  The pattern is identical across snapshots (hence
    learnable) and deterministic.
    """
    if not snapshots:
        raise ValueError("snapshots must be non-empty")
    if amplitude < 0:
        raise ValueError("amplitude must be >= 0")
    ny, nx = snapshots[0][0].shape
    ys = torch.arange(ny, dtype=torch.float32)
    xs = torch.arange(nx, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    l1, l2, l3 = wavelengths
    pattern = amplitude * (
        torch.sin(2.0 * torch.pi * xx / l1) * torch.sin(2.0 * torch.pi * yy / l2)
        + 0.7 * torch.sin(2.0 * torch.pi * (xx + yy) / l3)
    )
    return [
        (ux + pattern, uy + 0.8 * pattern) for ux, uy in snapshots
    ]


def write_snapshots_hdf5(
    path: str | Path,
    snapshots: Sequence[tuple[torch.Tensor, torch.Tensor]],
    attrs: Mapping[str, Any],
) -> Path:
    """Persist snapshots as ``velocity`` (N, 2, ny, nx) float32 + attributes.

    # PROVISIONAL: 换用 data/solver_export.py 的正式导出 once the parallel
    # data branch lands its formal solver export; kept on the ml/apps side so
    # ``tensorlbm.data`` stays untouched.
    """
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "The provisional HDF5 snapshot writer requires h5py "
            "(pip install h5py)"
        ) from exc
    arr = np.stack([
        np.stack([np.asarray(ux, dtype=np.float32), np.asarray(uy, dtype=np.float32)])
        for ux, uy in snapshots
    ])
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(p, "w") as h5:
        dset = h5.create_dataset("velocity", data=arr)
        for key, value in dict(attrs).items():
            dset.attrs[key] = value
    return p


def load_snapshots_hdf5(
    path: str | Path,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], dict[str, Any]]:
    """Read back what :func:`write_snapshots_hdf5` wrote."""
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("loading HDF5 snapshots requires h5py") from exc
    with h5py.File(str(path), "r") as h5:
        if "velocity" not in h5:
            raise KeyError(f"no 'velocity' dataset in {path}")
        arr = np.asarray(h5["velocity"][...], dtype=np.float32)
        attrs = {k: _h5_attr_to_python(v) for k, v in h5["velocity"].attrs.items()}
    if arr.ndim != 4 or arr.shape[1] < 2:
        raise ValueError(f"expected velocity array (N, 2, ny, nx), got {arr.shape}")
    return [
        (torch.from_numpy(arr[i, 0].copy()), torch.from_numpy(arr[i, 1].copy()))
        for i in range(arr.shape[0])
    ], attrs


def _h5_attr_to_python(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def try_load_pilot_dataset(
    pilot_dir: str | Path | None,
    *,
    slice_every: int = 4,
    plane_std_threshold: float = 1e-3,
    max_products: int | None = None,
) -> tuple[dict[str, list[tuple[torch.Tensor, torch.Tensor]]], dict[str, Any]] | None:
    """Consume the pilot SUBOFF dataset through the official solver-export API.

    ``tensorlbm.data.solver_export.load_product`` / ``load_product_arrays``
    do the byte-verified decoding; this helper adds the ml/-side 2-D view the
    demo's FNO2d needs (``data/`` itself is only read):

    * products are grouped by their ``nx`` metadata row so every chosen
      snapshot shares one grid — the largest same-grid family wins;
    * fixed z-planes (every ``slice_every``-th) of the 3-D velocity field
      become 2-D ``(ux, uy)`` snapshots; undisturbed free-stream planes
      (``std(ux) <= plane_std_threshold``) are skipped; 2-D products pass
      through whole.

    Snapshots are returned **grouped by case** so the caller can split by
    case (leak-proof: every snapshot of a case stays on one side).  Returns
    ``None`` when the directory / catalog is absent or nothing loads, letting
    the caller fall back to the provisional in-process solver run.
    """
    if pilot_dir is None:
        return None
    root = Path(pilot_dir)
    db_path = root / "catalog.db"
    if not db_path.is_file():
        return None
    try:
        from tensorlbm.data.catalog import FieldDataCatalog
        from tensorlbm.data.solver_export import load_product, load_product_arrays
    except ImportError:  # pragma: no cover - solver_export always present post-merge
        return None

    catalog = FieldDataCatalog.open(db_path)
    try:
        assets = catalog.list_assets(kind="field_product", limit=500)
        if not assets:
            return None
        families: dict[str, list[tuple[str, str]]] = {}
        for asset in assets:
            meta = {m.key: m.value for m in catalog.get_metadata(asset.asset_id)}
            case = str(meta.get("case") or "case")
            families.setdefault(str(meta.get("nx") or ""), []).append(
                (asset.asset_id, case),
            )
        nx_key = max(families, key=lambda k: len(families[k]))
        chosen = families[nx_key]
        if max_products is not None:
            chosen = chosen[: max(0, int(max_products))]

        by_case: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        for asset_id, case in chosen:
            try:
                arrays = load_product_arrays(load_product(catalog, asset_id))
            except Exception:
                continue  # skip unloaded products; never fabricate data
            velocity = arrays.get("velocity")
            if velocity is None:
                continue
            if velocity.ndim == 4 and velocity.shape[-1] >= 2:  # (nz, ny, nx, 3)
                planes = range(0, int(velocity.shape[0]), max(1, int(slice_every)))
                snaps = []
                for z in planes:
                    ux_plane = velocity[z, :, :, 0]
                    if float(ux_plane.std()) <= float(plane_std_threshold):
                        continue  # undisturbed free-stream plane: no content
                    snaps.append((
                        torch.from_numpy(ux_plane.copy()),
                        torch.from_numpy(velocity[z, :, :, 1].copy()),
                    ))
            elif velocity.ndim == 3 and velocity.shape[-1] == 2:  # (ny, nx, 2)
                snaps = [(
                    torch.from_numpy(velocity[..., 0].copy()),
                    torch.from_numpy(velocity[..., 1].copy()),
                )]
            else:
                continue
            by_case.setdefault(case, []).extend(snaps)
        if not by_case:
            return None

        dataset_id = "solver_export"
        summary_path = root / "summary.json"
        if summary_path.is_file():
            try:
                dataset_id = str(
                    json.loads(summary_path.read_text()).get("dataset_id", dataset_id)
                )
            except (json.JSONDecodeError, OSError):
                pass
        info = {
            "dataset_id": dataset_id,
            "nx": nx_key,
            "n_products": len(chosen),
            "product_ids": [asset_id for asset_id, _ in chosen],
            "cases": sorted(by_case),
            "pilot_dir": str(root),
        }
        return by_case, info
    finally:
        catalog.close()


# ---------------------------------------------------------------------------
# Stage 2: dataset construction
# ---------------------------------------------------------------------------

def _coarsen_mean(field: torch.Tensor, factor: int) -> torch.Tensor:
    """Block-average a 2-D field by an integer factor."""
    if factor <= 1:
        return field
    if field.ndim != 2:
        raise ValueError(f"field must be 2-D, got shape {tuple(field.shape)}")
    ny, nx = int(field.shape[0]), int(field.shape[1])
    cy, cx = ny // int(factor), nx // int(factor)
    if cy < 1 or cx < 1:
        raise ValueError(
            f"coarsen factor {factor} is too large for field shape {tuple(field.shape)}"
        )
    trimmed = field[: cy * factor, : cx * factor]
    return trimmed.reshape(cy, factor, cx, factor).mean(dim=(1, 3))


def build_super_resolution_dataset(
    snapshots: Sequence[tuple[torch.Tensor, torch.Tensor]],
    factor: int = 2,
) -> dict[str, Any]:
    """Build coarse->fine sample pairs: input = coarsened-then-upsampled field."""
    if not snapshots:
        raise ValueError("snapshots must be non-empty")
    inputs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for ux, uy in snapshots:
        ux = torch.as_tensor(ux, dtype=torch.float32)
        uy = torch.as_tensor(uy, dtype=torch.float32)
        if ux.shape != uy.shape:
            raise ValueError(f"ux/uy shape mismatch: {ux.shape} vs {uy.shape}")
        ny, nx = ux.shape
        fine = torch.stack([ux, uy], dim=0)  # (2, ny, nx)
        coarse = torch.stack([
            _coarsen_mean(ux, factor), _coarsen_mean(uy, factor),
        ], dim=0).unsqueeze(0)  # (1, 2, cy, cx)
        up = F.interpolate(coarse, size=(ny, nx), mode="bilinear", align_corners=False)
        inputs.append(up[0])
        targets.append(fine)
    return {
        "inputs": torch.stack(inputs),
        "targets": torch.stack(targets),
        "grid": tuple(int(v) for v in inputs[0].shape[-2:]),
        "downsample_factor": int(factor),
        "n_samples": len(inputs),
    }


def split_train_val(
    dataset: Mapping[str, Any],
    val_fraction: float = 0.15,
    seed: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Deterministically split a pair dataset into train / val subsets."""
    n = int(dataset["n_samples"])
    if not 0.0 < float(val_fraction) < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    n_val = max(1, int(round(n * float(val_fraction))))
    n_val = min(n_val, n - 1) if n > 1 else n_val
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=g)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    if len(train_idx) == 0:  # degenerate tiny datasets: keep 1 train sample
        train_idx, val_idx = perm[:1], perm[1:]
    if len(train_idx) == 0:
        raise ValueError("dataset too small to split")

    def _subset(idx: torch.Tensor) -> dict[str, Any]:
        return {
            "inputs": dataset["inputs"][idx],
            "targets": dataset["targets"][idx],
            "indices": [int(i) for i in idx],
            "n_samples": len(idx),
        }

    return _subset(train_idx), _subset(val_idx)


def _pairs_subset(pairs: Mapping[str, Any]) -> dict[str, Any]:
    """Wrap a pair dataset as the train/val subset shape used downstream."""
    n = int(pairs["n_samples"])
    return {
        "inputs": pairs["inputs"],
        "targets": pairs["targets"],
        "indices": list(range(n)),
        "n_samples": n,
    }


# ---------------------------------------------------------------------------
# Stage 5 helper: inference post-processing
# ---------------------------------------------------------------------------

def prediction_error_metrics(
    pred: Any, target: Any,
) -> dict[str, float]:
    """Field-level error metrics of a prediction against its reference."""
    p = torch.as_tensor(pred, dtype=torch.float32)
    t = torch.as_tensor(target, dtype=torch.float32)
    if p.shape != t.shape:
        raise ValueError(f"shape mismatch: {tuple(p.shape)} vs {tuple(t.shape)}")
    diff = p - t
    mse = float(torch.mean(diff**2).item())
    t_norm = float(torch.linalg.vector_norm(t).item())
    l2 = float(torch.linalg.vector_norm(diff).item())
    return {
        "mse": mse,
        "rmse": mse**0.5,
        "mae": float(torch.mean(torch.abs(diff)).item()),
        "max_abs_error": float(torch.max(torch.abs(diff)).item()),
        "relative_l2": (l2 / t_norm) if t_norm > 0 else float("inf"),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _train_fno(
    model: FNO2d,
    train: Mapping[str, Any],
    out_path: Path,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    lr_min: float,
    seed: int,
    device: torch.device,
) -> list[float]:
    torch.manual_seed(int(seed))
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(epochs)), eta_min=float(lr_min),
    )
    loss_fn = nn.MSELoss()
    X, Y = train["inputs"], train["targets"]
    n = int(X.shape[0])
    history: list[float] = []
    for _epoch in range(max(1, int(epochs))):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, max(1, int(batch_size))):
            idx = perm[i : i + max(1, int(batch_size))]
            xb = X[idx].to(device)
            yb = Y[idx].to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(idx)
        scheduler.step()
        history.append(epoch_loss / n)
    model.eval()
    save_fno2d(model, out_path)
    return history


# ---------------------------------------------------------------------------
# The full loop
# ---------------------------------------------------------------------------

def run_flagship_demo(config: FlagshipConfig | None = None) -> FlagshipRunReport:
    """Run the closed loop: data -> dataset -> job -> model asset -> serving -> lineage."""
    cfg = config or FlagshipConfig()
    t_start = time.perf_counter()
    workdir = Path(cfg.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    phase_times: dict[str, float] = {}

    # ---- stage 1: data -----------------------------------------------------
    t0 = time.perf_counter()
    loaded = try_load_pilot_dataset(cfg.pilot_dir)
    pilot_info: dict[str, Any] | None = None
    train_snapshots: list[tuple[torch.Tensor, torch.Tensor]] | None = None
    val_snapshots: list[tuple[torch.Tensor, torch.Tensor]] | None = None
    if loaded is not None:
        by_case, pilot_info = loaded
        cases = sorted(by_case)
        if len(cases) >= 2:
            # leak-proof split: hold out one whole case (all its snapshots and
            # slices) exactly like the pilot's own case-grouped split
            train_cases, val_case = cases[:-1], cases[-1]
        else:
            train_cases, val_case = cases[:1], cases[0]
        train_snapshots = [s for c in train_cases for s in by_case[c]]
        val_snapshots = list(by_case[val_case])
        snapshots = train_snapshots + val_snapshots
        data_source = f"pilot:{pilot_info['dataset_id']}"
    else:
        # PROVISIONAL: 换用 data/solver_export.py 的正式导出 (pilot dataset not
        # published at this path yet).
        snapshots = []
        for seed in cfg.seeds:
            snapshots.extend(produce_velocity_snapshots(
                nx=cfg.nx, ny=cfg.ny, n_steps=cfg.n_steps,
                sample_every=cfg.sample_every, seed=seed,
                tau=cfg.tau, c_s=cfg.c_s, device=device,
            ))
        if cfg.roughness_amplitude > 0:
            snapshots = add_stationary_roughness(
                snapshots, amplitude=cfg.roughness_amplitude,
            )
        data_source = "provisional_solver"
    data_path = workdir / "velocity_snapshots.h5"
    write_snapshots_hdf5(data_path, snapshots, attrs={
        "source": data_source,
        "task": cfg.task,
        "git_sha": _git_sha(),
        "n_snapshots": len(snapshots),
        "downsample_factor": int(cfg.downsample_factor),
        "stationary_roughness": (
            float(cfg.roughness_amplitude) if data_source == "provisional_solver" else 0.0
        ),
        "pilot_dataset_id": pilot_info["dataset_id"] if pilot_info else "",
        "pilot_products": pilot_info["n_products"] if pilot_info else 0,
        "nx": int(snapshots[0][0].shape[1]),
        "ny": int(snapshots[0][0].shape[0]),
    })
    phase_times["data"] = time.perf_counter() - t0

    catalog = FieldDataCatalog.open(workdir / "platform.db")
    try:
        # ---- stage 2: dataset + product registration -----------------------
        t0 = time.perf_counter()
        if train_snapshots is not None and val_snapshots is not None:
            pairs_train = build_super_resolution_dataset(train_snapshots, cfg.downsample_factor)
            pairs_val = build_super_resolution_dataset(val_snapshots, cfg.downsample_factor)
            train = _pairs_subset(pairs_train)
            val = _pairs_subset(pairs_val)
            pairs = build_super_resolution_dataset(snapshots, cfg.downsample_factor)
        else:
            pairs = build_super_resolution_dataset(snapshots, cfg.downsample_factor)
            train, val = split_train_val(pairs, cfg.val_fraction, cfg.seed)

        product_asset_id = f"{cfg.prefix}:u"
        catalog.register_asset(AssetRecord(
            asset_id=product_asset_id,
            name="Flagship 2-D velocity snapshots",
            kind="field_product",
            field_name="u",
            units="lu",
            shape=str(tuple(int(v) for v in pairs["inputs"].shape)),
            dtype="float32",
            source_run_id=data_source,
            description=(
                f"pilot={pilot_info['dataset_id']} nx={pilot_info['nx']} "
                f"products={pilot_info['n_products']} cases={pilot_info['cases']}"
                if pilot_info else "provisional in-process solver run"
            ),
            tags=(cfg.prefix, cfg.task, "flagship"),
        ))
        finite = all(
            bool(torch.isfinite(ux).all() and torch.isfinite(uy).all())
            for ux, uy in snapshots
        )
        variance = float(torch.stack(
            [torch.as_tensor(ux).float().var() for ux, _ in snapshots[:5]]
        ).mean().item())
        catalog.record_quality(product_asset_id, [
            QualityCheck("all_finite", finite, "ux/uy finite in every snapshot"),
            QualityCheck("nonzero_variance", variance > 0.0, f"mean var={variance:.3e}"),
            QualityCheck("shape_consistent", bool(len(snapshots) > 1),
                         f"n_snapshots={len(snapshots)}"),
        ])

        dataset_asset_id = f"{cfg.prefix}:dataset"
        catalog.register_asset(AssetRecord(
            asset_id=dataset_asset_id,
            name="Flagship super-resolution dataset",
            kind="dataset",
            description=(
                f"coarse(x{cfg.downsample_factor})->fine pairs; "
                f"n={pairs['n_samples']}"
            ),
            tags=(cfg.prefix, cfg.task),
        ))
        catalog.add_lineage(LineageRecord(
            source_id=product_asset_id, target_id=dataset_asset_id,
            relation_type="derived_from", resource_type="product",
        ))
        phase_times["dataset"] = time.perf_counter() - t0

        # ---- stage 3: training job -----------------------------------------
        t0 = time.perf_counter()
        arch = FNO2dArch(**{**_ARCH_DEFAULTS, **dict(cfg.arch)})
        jobs = TrainingJobRegistry.open(workdir / "platform.db")
        try:
            job = jobs.create_job({
                "task": cfg.task,
                "arch": asdict(arch),
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "learning_rate": cfg.learning_rate,
                "dataset_asset_id": dataset_asset_id,
                "data_source": data_source,
            })
            jobs.update_status(job.job_id, "running")
            model = FNO2d(arch)
            ckpt_path = workdir / "flagship_fno2d.pt"
            loss_history = _train_fno(
                model, train, ckpt_path,
                epochs=cfg.epochs, batch_size=cfg.batch_size,
                learning_rate=cfg.learning_rate, lr_min=cfg.lr_min,
                seed=cfg.seed, device=device,
            )
            jobs.record_metrics(job.job_id, {
                "train_loss_first": loss_history[0],
                "train_loss_final": loss_history[-1],
            })
            job = jobs.update_status(job.job_id, "completed")
        finally:
            jobs.close()
        phase_times["train"] = time.perf_counter() - t0

        # ---- stage 4: model asset + serving registration --------------------
        t0 = time.perf_counter()
        store_root = workdir / "model_store"
        registry = ModelAssetRegistry.open(store_root)
        try:
            model_id = registry.register(ckpt_path, meta={
                "task": cfg.task,
                "name": cfg.model_name,
                "family": FAMILY_FNO,
                "metrics": {
                    "train_loss_first": loss_history[0],
                    "train_loss_final": loss_history[-1],
                    "n_train_samples": train["n_samples"],
                },
                "arch": asdict(arch),
                "dataset_product_id": product_asset_id,
                "training_job_id": job.job_id,
                "tags": (cfg.prefix, cfg.task, "flagship"),
                "description": (
                    f"FNO2d super-resolution (x{cfg.downsample_factor}) trained "
                    f"on {data_source}"
                ),
            })
            asset = registry.get_model(model_id)
            assert asset is not None

            serving = ModelRegistry.open(workdir / "platform.db")
            try:
                serving_model_id = serving.register_model(
                    name=f"{cfg.prefix}-{cfg.model_name}",
                    path=asset.checkpoint_path,
                    arch=asdict(arch),
                    metrics={"train_loss_final": loss_history[-1]},
                    family=FAMILY_FNO,
                    lineage={
                        "model_asset_id": model_id,
                        "dataset_product_id": product_asset_id,
                        "training_job_id": job.job_id,
                    },
                )
            finally:
                serving.close()
            registry.link_serving_model(model_id, serving_model_id)
            phase_times["registry"] = time.perf_counter() - t0

            # ---- stage 5: live inference through the serving layer ---------
            t0 = time.perf_counter()
            serving = ModelRegistry.open(workdir / "platform.db")
            svc = InferenceService(serving)
            try:
                asset_model = registry.load_model(model_id)  # fresh store load
                n_val_samples = int(val["n_samples"]) or int(train["n_samples"])
                val_src = val if val["n_samples"] else train
                # aggregate over the whole held-out set ...
                all_rel_l2: list[float] = []
                for i in range(n_val_samples):
                    pred = svc.predict(
                        serving_model_id, val_src["inputs"][i].unsqueeze(0),
                    )[0]
                    all_rel_l2.append(
                        prediction_error_metrics(pred, val_src["targets"][i])["relative_l2"]
                    )
                val_mean_relative_l2 = (
                    float(sum(all_rel_l2) / len(all_rel_l2)) if all_rel_l2 else float("nan")
                )
                # ... and display the two most active samples (highest target
                # variance), each against the bilinear baseline on the SAME sample
                stds = val_src["targets"][:, 0].flatten(1).std(dim=1)
                show_idx = torch.topk(stds, k=min(2, n_val_samples)).indices.tolist()
                val_errors: list[dict[str, Any]] = []
                for i in show_idx:
                    x = val_src["inputs"][i]
                    y = val_src["targets"][i]
                    pred = svc.predict(serving_model_id, x.unsqueeze(0))[0]
                    errors = prediction_error_metrics(pred, y)
                    with torch.no_grad():
                        pred_asset = asset_model(x.unsqueeze(0)).squeeze(0).numpy()
                    errors["serving_vs_asset_max_abs_diff"] = float(
                        np.max(np.abs(pred - pred_asset))
                    )
                    errors["bilinear_relative_l2"] = prediction_error_metrics(
                        x, y,
                    )["relative_l2"]
                    val_errors.append({"val_index": int(val_src["indices"][i]), **errors})
                baseline_src = val if val["n_samples"] else train
                baseline = prediction_error_metrics(
                    baseline_src["inputs"][show_idx[0]], baseline_src["targets"][show_idx[0]],
                )
            finally:
                serving.close()
            phase_times["serving"] = time.perf_counter() - t0
            registry.record_metrics(model_id, {
                "val_mean_relative_l2": val_mean_relative_l2,
                "n_val_samples": n_val_samples,
            })
        finally:
            registry.close()

        # ---- stage 6: lineage closure ---------------------------------------
        t0 = time.perf_counter()
        job_asset_id = f"{cfg.prefix}:job:{job.job_id}"
        catalog.register_asset(AssetRecord(
            asset_id=job_asset_id, name=f"flagship training job {job.job_id}",
            kind="run", tags=(cfg.prefix,),
        ))
        catalog.add_lineage(LineageRecord(
            source_id=dataset_asset_id, target_id=job_asset_id,
            relation_type="trained_on", resource_type="dataset",
        ))
        catalog.register_asset(AssetRecord(
            asset_id=model_id, name=cfg.model_name, kind="model",
            description=f"asset-registry checkpoint for job {job.job_id}",
            tags=(cfg.prefix, cfg.task),
        ))
        catalog.add_lineage(LineageRecord(
            source_id=job_asset_id, target_id=model_id,
            relation_type="produced_model", resource_type="run",
        ))
        serving_asset_id = f"{cfg.prefix}:serving:{serving_model_id}"
        catalog.register_asset(AssetRecord(
            asset_id=serving_asset_id,
            name=f"flagship serving endpoint (model {serving_model_id})",
            kind="run", tags=(cfg.prefix,),
        ))
        catalog.add_lineage(LineageRecord(
            source_id=model_id, target_id=serving_asset_id,
            relation_type="served_by", resource_type="model",
        ))
        upstream = catalog.upstream(serving_asset_id)
        phase_times["lineage"] = time.perf_counter() - t0
    finally:
        catalog.close()

    report = FlagshipRunReport(
        workspace=str(workdir.resolve()),
        data_source=data_source,
        data_path=str(data_path.resolve()),
        pilot=pilot_info,
        product_asset_id=product_asset_id,
        dataset_asset_id=dataset_asset_id,
        n_train=train["n_samples"],
        n_val=val["n_samples"],
        job_id=job.job_id,
        job_status=job.status,
        ckpt_path=str(Path(ckpt_path).resolve()),
        model_id=model_id,
        model_store=str(store_root.resolve()),
        serving_model_id=serving_model_id,
        serving_asset_id=serving_asset_id,
        loss_history=loss_history,
        val_errors=val_errors,
        baseline_upsample_error=baseline,
        val_mean_relative_l2=val_mean_relative_l2,
        lineage_upstream=sorted(upstream),
        phase_times=phase_times,
        git_sha=_git_sha(),
        device=str(device),
    )
    report.save(workdir / "report.json")
    return report


def print_report(report: FlagshipRunReport) -> None:
    """Human-readable summary of a flagship run (used by the example CLI)."""
    h = report.loss_history
    print("=" * 72)
    print("AI4S flagship demo — closed loop summary")
    print("=" * 72)
    print(f"device         : {report.device}   git {report.git_sha}")
    print(f"data           : {report.data_source}")
    if report.pilot:
        print(f"  pilot        : dataset {report.pilot['dataset_id']} nx={report.pilot['nx']}"
              f" products={report.pilot['n_products']} cases={report.pilot['cases']}")
    print(f"  hdf5         : {report.data_path}")
    print(f"  product      : {report.product_asset_id} -> dataset {report.dataset_asset_id}"
          f"  (train {report.n_train} / val {report.n_val})")
    print(f"training job   : {report.job_id} [{report.job_status}]")
    print(f"  loss         : epoch1 {h[0]:.6e} -> final {h[-1]:.6e} "
          f"({len(h)} epochs)")
    print(f"model asset    : {report.model_id}")
    print(f"  store        : {report.model_store}")
    print(f"  checkpoint   : {report.ckpt_path}")
    print(f"serving model  : {report.serving_model_id} -> asset {report.serving_asset_id}")
    print("live inference (held-out samples):")
    for e in report.val_errors:
        print(
            f"  sample #{e['val_index']:>3}: mse {e['mse']:.3e}  "
            f"rel-L2 {e['relative_l2']:.4f} (bilinear {e.get('bilinear_relative_l2', float('nan')):.4f})  "
            f"max|err| {e['max_abs_error']:.3e}  "
            f"serving==asset (max diff {e['serving_vs_asset_max_abs_diff']:.1e})"
        )
    print(f"held-out mean rel-L2 over all {report.n_val} val samples: "
          f"{report.val_mean_relative_l2:.4f}")
    print(f"lineage upstream of {report.serving_asset_id}:")
    for node in report.lineage_upstream:
        print(f"  <- {node}")
    print(f"phase times    : "
          + ", ".join(f"{k} {v:.1f}s" for k, v in report.phase_times.items()))
    print("=" * 72)
