"""Trajectory-aware multimodal dataset loader and R2 catalog bridge for cylinder flow.

Implements Stage 2 of the cylinder-flow AI dataset pipeline defined in
``docs/benchmarks/tutorials/cylinder_flow_ai_dataset.md``:

1. Enforces all seven Stage-1 numerical/metadata quality gates plus visual
   frame verification on solver output directories.
2. Computes exact 2-D signed distance fields (SDF) from ``solid_mask`` without
   external SciPy dependencies.
3. Enforces trajectory-level train/val/test splits (never splitting adjacent
   frames of one solver trajectory across splits) and writes versioned JSON
   dataset manifests with explicit normalization scales and code revisions.
4. Provides :class:`CylinderMultimodalDataset` pairing physical fields
   ``(ux, uy, rho)``, geometry ``(solid_mask, sdf)``, aerodynamic force history
   ``(cd, cl)``, physical regime metadata ``(Re, u_in, tau, radius, ...)``, and
   rendered flow visualizations ``flow_step_XXXXXX.png``.
5. Bridges validated cylinder-flow runs into the canonical ``tensorlbm.data``
   HDF5 + ``FieldDataProductR2`` catalog contract via
   :func:`export_cylinder_run_to_r2_catalog`.
"""

from __future__ import annotations

import csv
import json
import math
import re
import zipfile
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Sequence

import matplotlib.image as mpimg
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from tensorlbm.data.catalog import FieldDataCatalog
from tensorlbm.data.solver_export import register_product, save_fields_hdf5

__all__ = [
    "DATASET_MANIFEST_SCHEMA",
    "SDF_CLIP_VOXELS_2D",
    "TrajectoryValidationError",
    "CylinderSnapshotFrame",
    "CylinderTrajectoryRecord",
    "CylinderTrajectorySummary",
    "CylinderNormalization",
    "MultimodalDatasetManifest",
    "CylinderMultimodalSample",
    "CylinderMultimodalDataset",
    "signed_distance_field_2d",
    "normalized_sdf_2d",
    "validate_cylinder_trajectory",
    "trim_trajectory_spinup",
    "split_cylinder_trajectories",
    "collate_cylinder_multimodal",
    "export_cylinder_run_to_r2_catalog",
]

DATASET_MANIFEST_SCHEMA = "tensorlbm.cylinder-multimodal-dataset/v1"
SDF_CLIP_VOXELS_2D = 8.0

_REQUIRED_NPZ_KEYS = frozenset({"step", "ux", "uy", "rho", "speed", "vorticity", "solid_mask"})
_FLOAT_NPZ_KEYS = ("ux", "uy", "rho", "speed", "vorticity")
_STEP_FILE_RE = re.compile(r"^fields_step_(\d+)\.npz$")
_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")


class TrajectoryValidationError(ValueError):
    """Raised when a cylinder-flow run directory fails a dataset quality gate."""


# ---------------------------------------------------------------------------
# Exact 2-D Euclidean signed distance field (Felzenszwalb-Huttenlocher)
# ---------------------------------------------------------------------------


def _edt_1d_sq(f: np.ndarray) -> np.ndarray:
    """Exact 1-D squared Euclidean distance transform along the 1-D array *f*."""
    n = int(f.shape[0])
    d = np.empty(n, dtype=np.float64)
    v = np.zeros(n, dtype=np.int64)
    z = np.empty(n + 1, dtype=np.float64)
    k = 0
    v[0] = 0
    z[0] = -np.inf
    z[1] = np.inf

    for q in range(1, n):
        vk = int(v[k])
        s = ((f[q] + q * q) - (f[vk] + vk * vk)) / (2.0 * (q - vk))
        while s <= z[k]:
            k -= 1
            vk = int(v[k])
            s = ((f[q] + q * q) - (f[vk] + vk * vk)) / (2.0 * (q - vk))
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = np.inf

    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        diff = q - int(v[k])
        d[q] = diff * diff + f[int(v[k])]
    return d


def _edt_2d(target: np.ndarray) -> np.ndarray:
    """Exact 2-D Euclidean distance to the nearest True cell in *target*."""
    ny, nx = target.shape
    inf = float((ny + nx) ** 2 * 4 + 1)
    grid = np.where(target, 0.0, inf).astype(np.float64)

    for y in range(ny):
        if not np.all(grid[y] == 0.0):
            grid[y] = _edt_1d_sq(grid[y])
    for x in range(nx):
        col = grid[:, x]
        if not np.all(col == inf):
            grid[:, x] = _edt_1d_sq(col)
    return np.sqrt(np.maximum(grid, 0.0))


def signed_distance_field_2d(
    solid_mask: np.ndarray,
    *,
    degenerate_sentinel: float = 64.0,
) -> np.ndarray:
    """Compute exact 2-D signed distance field (negative inside solid, positive in fluid)."""
    mask = np.asarray(solid_mask, dtype=np.bool_)
    if mask.ndim != 2:
        raise ValueError(f"solid_mask must be 2-D, got shape {mask.shape}")
    if not np.any(mask):
        return np.full(mask.shape, float(degenerate_sentinel), dtype=np.float32)
    if np.all(mask):
        return np.full(mask.shape, -float(degenerate_sentinel), dtype=np.float32)
    dist_to_solid = _edt_2d(mask)
    dist_to_fluid = _edt_2d(~mask)
    sdf = dist_to_solid - dist_to_fluid
    return np.asarray(sdf, dtype=np.float32)


def normalized_sdf_2d(
    solid_mask: np.ndarray,
    *,
    clip_voxels: float = SDF_CLIP_VOXELS_2D,
) -> np.ndarray:
    """Clipped and scaled 2-D SDF in ``[-1, 1]`` matching ``geom_encoder`` conventions."""
    if clip_voxels <= 0.0:
        raise ValueError("clip_voxels must be positive")
    sdf = signed_distance_field_2d(solid_mask, degenerate_sentinel=clip_voxels)
    clipped = np.clip(sdf, -clip_voxels, clip_voxels) / float(clip_voxels)
    return np.asarray(clipped, dtype=np.float32)


# ---------------------------------------------------------------------------
# Validated trajectory records & manifests
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CylinderSnapshotFrame:
    """One validated output step of a cylinder-flow trajectory."""

    step: int
    npz_path: Path
    png_path: Path | None
    ux: np.ndarray
    uy: np.ndarray
    rho: np.ndarray
    speed: np.ndarray
    vorticity: np.ndarray
    cd: float
    cl: float


@dataclass(frozen=True, slots=True)
class CylinderTrajectoryRecord:
    """A complete quality-checked solver trajectory on a fixed 2-D grid."""

    run_id: str
    run_dir: Path
    nx: int
    ny: int
    u_in: float
    re: float
    radius: float
    nu: float
    tau: float
    n_steps: int
    output_interval: int
    collision: str
    code_revision: str
    solid_mask: np.ndarray
    sdf_norm: np.ndarray
    frames: tuple[CylinderSnapshotFrame, ...]
    strouhal: float | None = None


@dataclass(frozen=True, slots=True)
class CylinderNormalization:
    """Explicit invertible normalization rules recorded in the dataset manifest."""

    velocity_mode: str = "inlet_velocity"
    fixed_u_scale: float | None = None
    rho_offset: float = 1.0
    min_u_scale: float = 1e-4
    min_rho_scale: float = 1e-4

    def scales_for_run(self, u_in: float) -> tuple[float, float]:
        if self.velocity_mode == "inlet_velocity":
            u_scale = max(abs(float(u_in)), self.min_u_scale)
        elif self.velocity_mode == "fixed":
            if self.fixed_u_scale is None or self.fixed_u_scale <= 0.0:
                raise ValueError("fixed_u_scale must be > 0 when velocity_mode='fixed'")
            u_scale = max(float(self.fixed_u_scale), self.min_u_scale)
        else:
            raise ValueError(f"Unsupported velocity_mode: {self.velocity_mode!r}")
        # Weakly-compressible D2Q9 acoustic scale: delta_rho ~ u_in / c_s = sqrt(3) * u_in
        rho_scale = max(math.sqrt(3.0) * u_scale, self.min_rho_scale)
        return u_scale, rho_scale


@dataclass(frozen=True, slots=True)
class CylinderTrajectorySummary:
    """Manifest entry summarizing one trajectory in a split."""

    run_id: str
    run_dir: str
    split: str
    nx: int
    ny: int
    re: float
    u_in: float
    radius: float
    tau: float
    n_frames: int
    steps: tuple[int, ...]
    u_scale: float
    rho_scale: float
    code_revision: str


@dataclass(frozen=True, slots=True)
class MultimodalDatasetManifest:
    """Versioned trajectory-level split and normalization manifest."""

    schema: str
    step_stride: int
    normalization: CylinderNormalization
    train_run_ids: tuple[str, ...]
    val_run_ids: tuple[str, ...]
    test_run_ids: tuple[str, ...]
    trajectories: tuple[CylinderTrajectorySummary, ...]
    fingerprint: str
    min_step: int = 0
    history_steps: int = 1
    rollout_steps: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "step_stride": self.step_stride,
            "min_step": self.min_step,
            "history_steps": self.history_steps,
            "rollout_steps": self.rollout_steps,
            "normalization": asdict(self.normalization),
            "train_run_ids": list(self.train_run_ids),
            "val_run_ids": list(self.val_run_ids),
            "test_run_ids": list(self.test_run_ids),
            "trajectories": [{**asdict(t), "steps": list(t.steps)} for t in self.trajectories],
            "fingerprint": self.fingerprint,
        }

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            f"{json.dumps(self.to_dict(), indent=2, sort_keys=True)}\n",
            encoding="utf-8",
        )
        return out

    @classmethod
    def load(cls, path: str | Path) -> MultimodalDatasetManifest:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if raw.get("schema") != DATASET_MANIFEST_SCHEMA:
            raise ValueError(f"Unsupported manifest schema: {raw.get('schema')!r}")
        norm = CylinderNormalization(**raw["normalization"])
        trajs = tuple(
            CylinderTrajectorySummary(
                run_id=str(item["run_id"]),
                run_dir=str(item["run_dir"]),
                split=str(item["split"]),
                nx=int(item["nx"]),
                ny=int(item["ny"]),
                re=float(item["re"]),
                u_in=float(item["u_in"]),
                radius=float(item["radius"]),
                tau=float(item["tau"]),
                n_frames=int(item["n_frames"]),
                steps=tuple(int(s) for s in item["steps"]),
                u_scale=float(item["u_scale"]),
                rho_scale=float(item["rho_scale"]),
                code_revision=str(item["code_revision"]),
            )
            for item in raw["trajectories"]
        )
        return cls(
            schema=str(raw["schema"]),
            step_stride=int(raw["step_stride"]),
            min_step=int(raw.get("min_step", 0)),
            history_steps=int(raw.get("history_steps", 1)),
            rollout_steps=int(raw.get("rollout_steps", 1)),
            normalization=norm,
            train_run_ids=tuple(str(r) for r in raw["train_run_ids"]),
            val_run_ids=tuple(str(r) for r in raw["val_run_ids"]),
            test_run_ids=tuple(str(r) for r in raw["test_run_ids"]),
            trajectories=trajs,
            fingerprint=str(raw["fingerprint"]),
        )


# ---------------------------------------------------------------------------
# Stage-1 quality checks (all 7 rules + visual frame check)
# ---------------------------------------------------------------------------


def _parse_forces_csv(forces_path: Path) -> dict[int, tuple[float, float]]:
    if not forces_path.is_file():
        raise TrajectoryValidationError(f"Missing forces.csv in {forces_path.parent}")
    forces: dict[int, tuple[float, float]] = {}
    try:
        with forces_path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None or {"step", "cd", "cl"} - set(reader.fieldnames):
                raise TrajectoryValidationError(
                    f"forces.csv must have columns step,cd,cl; got {reader.fieldnames}"
                )
            for row in reader:
                step = int(row["step"])
                cd = float(row["cd"])
                cl = float(row["cl"])
                if not (math.isfinite(cd) and math.isfinite(cl)):
                    raise TrajectoryValidationError(
                        f"Non-finite force values at step {step} in {forces_path}"
                    )
                if step in forces:
                    raise TrajectoryValidationError(f"Duplicate step {step} in {forces_path}")
                forces[step] = (cd, cl)
    except (ValueError, KeyError, OSError) as exc:
        if isinstance(exc, TrajectoryValidationError):
            raise
        raise TrajectoryValidationError(f"Malformed forces.csv at {forces_path}: {exc}") from exc
    if not forces:
        raise TrajectoryValidationError(f"Empty forces.csv at {forces_path}")
    return forces


def _validate_step_cadence(
    steps: list[int],
    *,
    output_interval: int,
    n_steps: int,
    resumed: bool,
) -> None:
    if len(steps) < 2:
        raise TrajectoryValidationError(
            f"Trajectory must contain at least 2 snapshots for forecasting; got {len(steps)}"
        )
    for prev, curr in zip(steps, steps[1:]):
        if curr <= prev:
            raise TrajectoryValidationError(f"Snapshot steps are not strictly increasing: {steps}")
    if not resumed:
        expected = list(range(output_interval, n_steps + 1, output_interval))
        if expected and expected[-1] != n_steps:
            expected.append(n_steps)
        elif not expected:
            expected = [n_steps]
        if steps != expected:
            raise TrajectoryValidationError(
                f"Snapshot steps {steps} do not match expected cadence {expected}"
            )
    else:
        for idx, step in enumerate(steps):
            is_last = idx == len(steps) - 1
            if step % output_interval != 0 and not (is_last and step == n_steps):
                raise TrajectoryValidationError(
                    f"Snapshot step {step} violates output_interval={output_interval} "
                    f"(n_steps={n_steps})"
                )


def validate_cylinder_trajectory(
    run_dir: str | Path,
    *,
    run_id: str | None = None,
    require_visual_png: bool = True,
    fallback_code_revision: str | None = None,
    sdf_clip_voxels: float = SDF_CLIP_VOXELS_2D,
) -> CylinderTrajectoryRecord:
    """Validate a cylinder-flow output directory against all Stage-1 quality checks."""
    root = Path(run_dir).resolve()
    if not root.is_dir():
        raise TrajectoryValidationError(f"Run directory does not exist: {root}")

    metadata_path = root / "run_metadata.json"
    if not metadata_path.is_file():
        raise TrajectoryValidationError(f"Missing run_metadata.json in {root}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrajectoryValidationError(f"Invalid run_metadata.json in {root}: {exc}") from exc

    if not isinstance(metadata, dict) or not isinstance(metadata.get("config"), dict):
        raise TrajectoryValidationError(f"run_metadata.json missing 'config' object in {root}")

    cfg = metadata["config"]
    derived = metadata.get("derived", {})
    repro = metadata.get("reproducibility", {})
    diagnostics = metadata.get("diagnostics", [])
    restart = metadata.get("restart", {})

    try:
        nx = int(cfg["nx"])
        ny = int(cfg["ny"])
        u_in = float(cfg["u_in"])
        re_val = float(cfg["re"])
        radius = float(cfg["radius"])
        n_steps = int(cfg["n_steps"])
        output_interval = int(cfg["output_interval"])
        collision = str(cfg.get("collision", "bgk"))
        nu = float(derived.get("nu", u_in * 2.0 * radius / re_val))
        tau = float(derived.get("tau", 3.0 * nu + 0.5))
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise TrajectoryValidationError(f"Incomplete solver config in {metadata_path}") from exc

    if nx <= 0 or ny <= 0 or u_in <= 0.0 or re_val <= 0.0 or radius <= 0.0 or tau <= 0.5:
        raise TrajectoryValidationError(f"Non-physical solver parameters in {metadata_path}")

    # Check 7: code revision retained
    git_commit = repro.get("git_commit") if isinstance(repro, dict) else None
    code_revision = (
        str(git_commit).strip()
        if isinstance(git_commit, str) and git_commit.strip()
        else (fallback_code_revision or "").strip()
    )
    if not code_revision:
        raise TrajectoryValidationError(
            f"Missing code revision (reproducibility.git_commit) in {metadata_path}"
        )

    # Check 6: forces.csv and metadata diagnostics finite
    forces = _parse_forces_csv(root / "forces.csv")
    if not isinstance(diagnostics, list) or not diagnostics:
        raise TrajectoryValidationError(f"Missing or empty diagnostics list in {metadata_path}")
    for diag in diagnostics:
        if not isinstance(diag, dict):
            raise TrajectoryValidationError(f"Invalid diagnostic entry in {metadata_path}")
        for key in ("mass", "mass_drift", "max_speed", "mean_rho", "cd", "cl"):
            val = diag.get(key)
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
                raise TrajectoryValidationError(
                    f"Non-finite diagnostic {key!r}={val!r} in {metadata_path}"
                )

    # Discover fields_step_*.npz files
    candidates: list[tuple[int, Path]] = []
    for child in root.iterdir():
        match = _STEP_FILE_RE.match(child.name)
        if match is not None and child.is_file():
            candidates.append((int(match.group(1)), child))
    candidates.sort(key=lambda pair: pair[0])
    if not candidates:
        raise TrajectoryValidationError(f"No fields_step_XXXXXX.npz files found in {root}")

    frames: list[CylinderSnapshotFrame] = []
    reference_mask: np.ndarray | None = None
    seen_steps: list[int] = []

    for filename_step, npz_path in candidates:
        # Check 1: opens cleanly with allow_pickle=False
        try:
            with np.load(npz_path, allow_pickle=False) as archive:
                keys = set(archive.files)
                if keys != _REQUIRED_NPZ_KEYS:
                    raise TrajectoryValidationError(
                        f"{npz_path.name} keys {sorted(keys)} != {sorted(_REQUIRED_NPZ_KEYS)}"
                    )
                step_arr = archive["step"]
                if step_arr.ndim != 0 or step_arr.dtype.kind not in {"i", "u"}:
                    raise TrajectoryValidationError(
                        f"{npz_path.name} 'step' must be an integer scalar"
                    )
                step = int(step_arr)
                if step != filename_step:
                    raise TrajectoryValidationError(
                        f"{npz_path.name} internal step {step} != filename step {filename_step}"
                    )

                loaded_floats: dict[str, np.ndarray] = {}
                for field_name in _FLOAT_NPZ_KEYS:
                    arr = archive[field_name]
                    # Check 2: exact configured (ny, nx) shape & floating dtype
                    if arr.shape != (ny, nx) or arr.dtype.kind != "f":
                        raise TrajectoryValidationError(
                            f"{npz_path.name} field {field_name!r} has shape={arr.shape}, "
                            f"dtype={arr.dtype}; expected ({ny}, {nx}) float"
                        )
                    # Check 3: strictly finite values
                    if not np.all(np.isfinite(arr)):
                        raise TrajectoryValidationError(
                            f"{npz_path.name} field {field_name!r} contains non-finite values"
                        )
                    loaded_floats[field_name] = np.ascontiguousarray(arr, dtype=np.float32)

                mask_arr = archive["solid_mask"]
                if mask_arr.shape != (ny, nx) or mask_arr.dtype != np.bool_:
                    raise TrajectoryValidationError(
                        f"{npz_path.name} 'solid_mask' has shape={mask_arr.shape}, "
                        f"dtype={mask_arr.dtype}; expected ({ny}, {nx}) bool"
                    )
                mask_copy = np.ascontiguousarray(mask_arr, dtype=np.bool_)
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            if isinstance(exc, TrajectoryValidationError):
                raise
            raise TrajectoryValidationError(
                f"Failed to load {npz_path.name} with allow_pickle=False: {exc}"
            ) from exc

        # Check 5: identical solid mask across fixed-geometry trajectory
        if reference_mask is None:
            if np.all(mask_copy):
                raise TrajectoryValidationError(
                    f"{npz_path.name} solid_mask marks the entire domain as solid"
                )
            reference_mask = mask_copy
        elif not np.array_equal(reference_mask, mask_copy):
            raise TrajectoryValidationError(
                f"{npz_path.name} solid_mask differs from initial snapshot mask"
            )

        if step not in forces:
            raise TrajectoryValidationError(
                f"Snapshot step {step} missing from forces.csv in {root}"
            )
        cd, cl = forces[step]

        png_candidate = root / f"flow_step_{step:06d}.png"
        png_path: Path | None
        if png_candidate.is_file() and png_candidate.stat().st_size > 0:
            png_path = png_candidate
        elif require_visual_png:
            raise TrajectoryValidationError(
                f"Missing or empty visualization {png_candidate.name} in {root}"
            )
        else:
            png_path = None

        seen_steps.append(step)
        frames.append(
            CylinderSnapshotFrame(
                step=step,
                npz_path=npz_path,
                png_path=png_path,
                ux=loaded_floats["ux"],
                uy=loaded_floats["uy"],
                rho=loaded_floats["rho"],
                speed=loaded_floats["speed"],
                vorticity=loaded_floats["vorticity"],
                cd=cd,
                cl=cl,
            )
        )

    # Check 4: strictly increasing steps agreeing with output_interval cadence
    resumed = bool(restart.get("resumed", False)) if isinstance(restart, dict) else False
    _validate_step_cadence(
        seen_steps,
        output_interval=output_interval,
        n_steps=n_steps,
        resumed=resumed,
    )
    if set(forces.keys()) != set(seen_steps):
        raise TrajectoryValidationError(
            f"forces.csv steps {sorted(forces)} do not match snapshot steps {seen_steps}"
        )

    assert reference_mask is not None
    sdf_norm = normalized_sdf_2d(reference_mask, clip_voxels=sdf_clip_voxels)
    st_raw = metadata.get("strouhal")
    strouhal = float(st_raw) if isinstance(st_raw, (int, float)) and math.isfinite(st_raw) else None
    resolved_id = (run_id or cfg.get("run_name") or root.name).strip()

    return CylinderTrajectoryRecord(
        run_id=resolved_id,
        run_dir=root,
        nx=nx,
        ny=ny,
        u_in=u_in,
        re=re_val,
        radius=radius,
        nu=nu,
        tau=tau,
        n_steps=n_steps,
        output_interval=output_interval,
        collision=collision,
        code_revision=code_revision,
        solid_mask=reference_mask,
        sdf_norm=sdf_norm,
        frames=tuple(frames),
        strouhal=strouhal,
    )


# ---------------------------------------------------------------------------
# Leakage-safe trajectory splitting & manifest creation
# ---------------------------------------------------------------------------


def trim_trajectory_spinup(
    trajectory: CylinderTrajectoryRecord,
    *,
    min_step: int = 0,
) -> CylinderTrajectoryRecord:
    """Return a copy of *trajectory* discarding acoustic spin-up frames ``step < min_step``."""
    if min_step <= 0:
        return trajectory
    kept = tuple(frame for frame in trajectory.frames if frame.step >= min_step)
    if len(kept) < 2:
        raise TrajectoryValidationError(
            f"Trajectory {trajectory.run_id!r} has fewer than 2 frames >= min_step={min_step} "
            f"(got {len(kept)})"
        )
    return CylinderTrajectoryRecord(
        run_id=trajectory.run_id,
        run_dir=trajectory.run_dir,
        nx=trajectory.nx,
        ny=trajectory.ny,
        u_in=trajectory.u_in,
        re=trajectory.re,
        radius=trajectory.radius,
        nu=trajectory.nu,
        tau=trajectory.tau,
        n_steps=trajectory.n_steps,
        output_interval=trajectory.output_interval,
        collision=trajectory.collision,
        code_revision=trajectory.code_revision,
        solid_mask=trajectory.solid_mask,
        sdf_norm=trajectory.sdf_norm,
        frames=kept,
        strouhal=trajectory.strouhal,
    )


def _manifest_fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(canonical).hexdigest()


def split_cylinder_trajectories(
    train_runs: Sequence[CylinderTrajectoryRecord],
    val_runs: Sequence[CylinderTrajectoryRecord] = (),
    test_runs: Sequence[CylinderTrajectoryRecord] = (),
    *,
    step_stride: int = 1,
    min_step: int = 0,
    history_steps: int = 1,
    rollout_steps: int = 1,
    normalization: CylinderNormalization | None = None,
) -> MultimodalDatasetManifest:
    """Create a leakage-safe trajectory-level dataset manifest.

    Raises :class:`ValueError` if any trajectory ``run_id`` or directory path
    appears more than once across ``train_runs``, ``val_runs``, and ``test_runs``.
    """
    if step_stride < 1 or history_steps < 1 or rollout_steps < 1 or min_step < 0:
        raise ValueError(
            "step_stride, history_steps, and rollout_steps must be >= 1 and min_step >= 0"
        )
    if not train_runs:
        raise ValueError("train_runs must contain at least one trajectory")

    norm = normalization or CylinderNormalization()
    seen_ids: dict[str, str] = {}
    seen_dirs: dict[Path, str] = {}
    summaries: list[CylinderTrajectorySummary] = []
    min_required = (history_steps + rollout_steps - 1) * step_stride + 1

    for split_name, group in (
        ("train", train_runs),
        ("val", val_runs),
        ("test", test_runs),
    ):
        for raw_traj in group:
            traj = trim_trajectory_spinup(raw_traj, min_step=min_step)
            if traj.run_id in seen_ids:
                raise ValueError(
                    f"Trajectory leakage: run_id {traj.run_id!r} appears in both "
                    f"{seen_ids[traj.run_id]!r} and {split_name!r}"
                )
            resolved_dir = traj.run_dir.resolve()
            if resolved_dir in seen_dirs:
                raise ValueError(
                    f"Trajectory leakage: directory {resolved_dir} appears in both "
                    f"{seen_dirs[resolved_dir]!r} and {split_name!r}"
                )
            if len(traj.frames) < min_required:
                raise ValueError(
                    f"Trajectory {traj.run_id!r} has {len(traj.frames)} kept frames, "
                    f"insufficient for history_steps={history_steps}, "
                    f"rollout_steps={rollout_steps}, step_stride={step_stride}"
                )
            seen_ids[traj.run_id] = split_name
            seen_dirs[resolved_dir] = split_name
            u_scale, rho_scale = norm.scales_for_run(traj.u_in)
            summaries.append(
                CylinderTrajectorySummary(
                    run_id=traj.run_id,
                    run_dir=str(resolved_dir),
                    split=split_name,
                    nx=traj.nx,
                    ny=traj.ny,
                    re=traj.re,
                    u_in=traj.u_in,
                    radius=traj.radius,
                    tau=traj.tau,
                    n_frames=len(traj.frames),
                    steps=tuple(frame.step for frame in traj.frames),
                    u_scale=u_scale,
                    rho_scale=rho_scale,
                    code_revision=traj.code_revision,
                )
            )

    core = {
        "schema": DATASET_MANIFEST_SCHEMA,
        "step_stride": step_stride,
        "min_step": min_step,
        "history_steps": history_steps,
        "rollout_steps": rollout_steps,
        "normalization": asdict(norm),
        "train_run_ids": [t.run_id for t in train_runs],
        "val_run_ids": [t.run_id for t in val_runs],
        "test_run_ids": [t.run_id for t in test_runs],
        "trajectories": [
            {
                "run_id": s.run_id,
                "split": s.split,
                "nx": s.nx,
                "ny": s.ny,
                "re": s.re,
                "u_in": s.u_in,
                "radius": s.radius,
                "tau": s.tau,
                "steps": list(s.steps),
                "u_scale": s.u_scale,
                "rho_scale": s.rho_scale,
                "code_revision": s.code_revision,
            }
            for s in summaries
        ],
    }
    fingerprint = _manifest_fingerprint(core)
    return MultimodalDatasetManifest(
        schema=DATASET_MANIFEST_SCHEMA,
        step_stride=step_stride,
        min_step=min_step,
        history_steps=history_steps,
        rollout_steps=rollout_steps,
        normalization=norm,
        train_run_ids=tuple(t.run_id for t in train_runs),
        val_run_ids=tuple(t.run_id for t in val_runs),
        test_run_ids=tuple(t.run_id for t in test_runs),
        trajectories=tuple(summaries),
        fingerprint=fingerprint,
    )


# ---------------------------------------------------------------------------
# Multimodal PyTorch Dataset
# ---------------------------------------------------------------------------


def _load_resized_png(png_path: Path, image_size: tuple[int, int]) -> torch.Tensor:
    img_np = mpimg.imread(str(png_path))
    if img_np.ndim == 2:
        img_np = np.repeat(img_np[..., None], 3, axis=-1)
    elif img_np.shape[-1] > 3:
        img_np = img_np[..., :3]
    if img_np.dtype == np.uint8:
        img_f = img_np.astype(np.float32) / 255.0
    else:
        img_f = np.clip(img_np.astype(np.float32), 0.0, 1.0)
    chw = torch.from_numpy(np.ascontiguousarray(img_f.transpose(2, 0, 1)))
    if tuple(chw.shape[1:]) != image_size:
        chw = F.interpolate(
            chw.unsqueeze(0),
            size=image_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return chw


@dataclass(frozen=True, slots=True)
class CylinderMultimodalSample:
    """One paired multimodal forecast window ``(t-(H-1)..t) -> (t+1..t+K)``."""

    field_t: torch.Tensor
    field_next: torch.Tensor
    raw_t: torch.Tensor
    raw_next: torch.Tensor
    history_fields: torch.Tensor
    history_forces: torch.Tensor
    target_fields: torch.Tensor
    target_raw: torch.Tensor
    target_forces: torch.Tensor
    geom: torch.Tensor
    solid_mask: torch.Tensor
    physics_cond: torch.Tensor
    force_t: torch.Tensor
    force_next: torch.Tensor
    visual_t: torch.Tensor
    has_visual: bool
    u_scale: float
    rho_scale: float
    rho_offset: float
    run_id: str
    step_t: int
    step_next: int


class CylinderMultimodalDataset(Dataset[CylinderMultimodalSample]):
    """Trajectory-aware multimodal dataset yielding intra-trajectory sequence windows."""

    def __init__(
        self,
        trajectories: Sequence[CylinderTrajectoryRecord],
        *,
        step_stride: int = 1,
        history_steps: int = 1,
        rollout_steps: int = 1,
        min_step: int = 0,
        normalization: CylinderNormalization | None = None,
        load_visual: bool = True,
        image_size: tuple[int, int] = (48, 96),
    ) -> None:
        if step_stride < 1 or history_steps < 1 or rollout_steps < 1 or min_step < 0:
            raise ValueError(
                "step_stride, history_steps, and rollout_steps must be >= 1 and min_step >= 0"
            )
        if not trajectories:
            raise ValueError("trajectories must not be empty")
        shapes = {(t.ny, t.nx) for t in trajectories}
        if len(shapes) != 1:
            raise ValueError(
                f"All trajectories in one dataset batch must share one grid shape; got {shapes}"
            )

        trimmed = tuple(trim_trajectory_spinup(t, min_step=min_step) for t in trajectories)
        self.trajectories = trimmed
        self.step_stride = step_stride
        self.history_steps = history_steps
        self.rollout_steps = rollout_steps
        self.min_step = min_step
        self.normalization = normalization or CylinderNormalization()
        self.load_visual = load_visual
        self.image_size = image_size
        self.grid_shape: tuple[int, int] = next(iter(shapes))

        self._index: list[tuple[int, int]] = []
        self._geom_cache: list[torch.Tensor] = []
        self._mask_cache: list[torch.Tensor] = []
        self._scales: list[tuple[float, float]] = []
        self._visual_cache: dict[tuple[int, int], torch.Tensor] = {}

        history_span = (history_steps - 1) * step_stride
        rollout_span = rollout_steps * step_stride
        min_required = history_span + rollout_span + 1

        for t_idx, traj in enumerate(self.trajectories):
            if len(traj.frames) < min_required:
                raise ValueError(
                    f"Trajectory {traj.run_id!r} has only {len(traj.frames)} kept frames "
                    f"(need >= {min_required} for history_steps={history_steps}, "
                    f"rollout_steps={rollout_steps}, step_stride={step_stride})"
                )
            mask_t = torch.from_numpy(traj.solid_mask)
            geom_t = torch.stack(
                [
                    mask_t.to(dtype=torch.float32),
                    torch.from_numpy(traj.sdf_norm),
                ],
                dim=0,
            )
            self._mask_cache.append(mask_t)
            self._geom_cache.append(geom_t)
            self._scales.append(self.normalization.scales_for_run(traj.u_in))
            # ref_idx is the index of current step t; history precedes it, rollout follows it
            for ref_idx in range(history_span, len(traj.frames) - rollout_span):
                self._index.append((t_idx, ref_idx))

    def __len__(self) -> int:
        return len(self._index)

    def _encode_fields(
        self, frame: CylinderSnapshotFrame, u_scale: float, rho_scale: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = torch.stack(
            [
                torch.from_numpy(frame.ux),
                torch.from_numpy(frame.uy),
                torch.from_numpy(frame.rho),
            ],
            dim=0,
        )
        normed = torch.stack(
            [
                raw[0] / u_scale,
                raw[1] / u_scale,
                (raw[2] - self.normalization.rho_offset) / rho_scale,
            ],
            dim=0,
        )
        return normed, raw

    def __getitem__(self, idx: int) -> CylinderMultimodalSample:
        t_idx, ref_idx = self._index[idx]
        traj = self.trajectories[t_idx]
        u_scale, rho_scale = self._scales[t_idx]

        hist_norm_list: list[torch.Tensor] = []
        hist_raw_list: list[torch.Tensor] = []
        hist_force_list: list[torch.Tensor] = []
        for h in range(self.history_steps - 1, -1, -1):
            h_frame = traj.frames[ref_idx - h * self.step_stride]
            f_norm, f_raw = self._encode_fields(h_frame, u_scale, rho_scale)
            hist_norm_list.append(f_norm)
            hist_raw_list.append(f_raw)
            hist_force_list.append(torch.tensor([h_frame.cd, h_frame.cl], dtype=torch.float32))

        tgt_norm_list: list[torch.Tensor] = []
        tgt_raw_list: list[torch.Tensor] = []
        tgt_force_list: list[torch.Tensor] = []
        for k in range(1, self.rollout_steps + 1):
            k_frame = traj.frames[ref_idx + k * self.step_stride]
            k_norm, k_raw = self._encode_fields(k_frame, u_scale, rho_scale)
            tgt_norm_list.append(k_norm)
            tgt_raw_list.append(k_raw)
            tgt_force_list.append(torch.tensor([k_frame.cd, k_frame.cl], dtype=torch.float32))

        history_fields = torch.stack(hist_norm_list, dim=0)
        history_forces = torch.stack(hist_force_list, dim=0)
        target_fields = torch.stack(tgt_norm_list, dim=0)
        target_raw = torch.stack(tgt_raw_list, dim=0)
        target_forces = torch.stack(tgt_force_list, dim=0)

        frame_t = traj.frames[ref_idx]
        frame_next = traj.frames[ref_idx + self.step_stride]
        field_t = history_fields[-1]
        raw_t = hist_raw_list[-1]
        field_next = target_fields[0]
        raw_next = target_raw[0]
        force_t = history_forces[-1]
        force_next = target_forces[0]

        delta_step = float(frame_next.step - frame_t.step)
        physics_cond = torch.tensor(
            [
                math.log10(max(traj.re, 1e-6)),
                traj.u_in,
                traj.tau - 0.5,
                traj.radius / float(traj.ny),
                float(traj.ny) / float(traj.nx),
                delta_step / 100.0,
            ],
            dtype=torch.float32,
        )

        if self.load_visual and frame_t.png_path is not None:
            key = (t_idx, ref_idx)
            if key not in self._visual_cache:
                self._visual_cache[key] = _load_resized_png(frame_t.png_path, self.image_size)
            visual_t = self._visual_cache[key]
            has_visual = True
        else:
            visual_t = torch.zeros((3, *self.image_size), dtype=torch.float32)
            has_visual = False

        return CylinderMultimodalSample(
            field_t=field_t,
            field_next=field_next,
            raw_t=raw_t,
            raw_next=raw_next,
            history_fields=history_fields,
            history_forces=history_forces,
            target_fields=target_fields,
            target_raw=target_raw,
            target_forces=target_forces,
            geom=self._geom_cache[t_idx],
            solid_mask=self._mask_cache[t_idx],
            physics_cond=physics_cond,
            force_t=force_t,
            force_next=force_next,
            visual_t=visual_t,
            has_visual=has_visual,
            u_scale=u_scale,
            rho_scale=rho_scale,
            rho_offset=self.normalization.rho_offset,
            run_id=traj.run_id,
            step_t=frame_t.step,
            step_next=frame_next.step,
        )


def collate_cylinder_multimodal(
    samples: Sequence[CylinderMultimodalSample],
) -> dict[str, Any]:
    """Collate :class:`CylinderMultimodalSample` items into batched tensors."""
    if not samples:
        raise ValueError("samples must not be empty")
    return {
        "field_t": torch.stack([s.field_t for s in samples], dim=0),
        "field_next": torch.stack([s.field_next for s in samples], dim=0),
        "raw_t": torch.stack([s.raw_t for s in samples], dim=0),
        "raw_next": torch.stack([s.raw_next for s in samples], dim=0),
        "history_fields": torch.stack([s.history_fields for s in samples], dim=0),
        "history_forces": torch.stack([s.history_forces for s in samples], dim=0),
        "target_fields": torch.stack([s.target_fields for s in samples], dim=0),
        "target_raw": torch.stack([s.target_raw for s in samples], dim=0),
        "target_forces": torch.stack([s.target_forces for s in samples], dim=0),
        "geom": torch.stack([s.geom for s in samples], dim=0),
        "solid_mask": torch.stack([s.solid_mask for s in samples], dim=0),
        "physics_cond": torch.stack([s.physics_cond for s in samples], dim=0),
        "force_t": torch.stack([s.force_t for s in samples], dim=0),
        "force_next": torch.stack([s.force_next for s in samples], dim=0),
        "visual_t": torch.stack([s.visual_t for s in samples], dim=0),
        "has_visual": torch.tensor([s.has_visual for s in samples], dtype=torch.bool),
        "u_scale": torch.tensor([s.u_scale for s in samples], dtype=torch.float32),
        "rho_scale": torch.tensor([s.rho_scale for s in samples], dtype=torch.float32),
        "rho_offset": float(samples[0].rho_offset),
        "run_ids": [s.run_id for s in samples],
        "step_t": [s.step_t for s in samples],
        "step_next": [s.step_next for s in samples],
    }


# ---------------------------------------------------------------------------
# Bridge to canonical R2 HDF5 + FieldDataCatalog infrastructure
# ---------------------------------------------------------------------------


def _resolve_40hex_sha(code_revision: str) -> str:
    candidate = code_revision.strip().lower()
    if _HEX40_RE.fullmatch(candidate):
        return candidate
    return sha256(code_revision.encode("utf-8")).hexdigest()[:40]


def export_cylinder_run_to_r2_catalog(
    trajectory: CylinderTrajectoryRecord,
    catalog: FieldDataCatalog,
    *,
    h5_path: str | Path | None = None,
    blob_root: str | Path | None = None,
    mass_tol: float = 1e-2,
) -> tuple[str, ...]:
    """Bridge a validated Stage-1 cylinder run into ``FieldDataCatalog`` R2 products."""
    resolved_h5 = (
        Path(h5_path).resolve()
        if h5_path is not None
        else (trajectory.run_dir / f"{trajectory.run_id}_r2.h5").resolve()
    )
    code_sha = _resolve_40hex_sha(trajectory.code_revision)
    mask_int8 = trajectory.solid_mask.astype(np.int8)

    for frame in trajectory.frames:
        save_fields_hdf5(
            resolved_h5,
            arrays={
                "rho": frame.rho,
                "ux": frame.ux,
                "uy": frame.uy,
                "solid_mask": mask_int8,
                "vorticity": frame.vorticity,
            },
            attrs={
                "step": frame.step,
                "run_id": trajectory.run_id,
                "case": "cylinder_flow_2d",
                "collision": trajectory.collision,
                "re": trajectory.re,
                "u_in": trajectory.u_in,
                "nu": trajectory.nu,
                "tau": trajectory.tau,
            },
        )

    product_ids: list[str] = []
    for frame in trajectory.frames:
        pid = register_product(
            catalog,
            resolved_h5,
            metadata={
                "run_id": trajectory.run_id,
                "case": "cylinder_flow_2d",
                "step": frame.step,
                "code_sha": code_sha,
                "collision": trajectory.collision,
                "lattice": "D2Q9",
                "re": trajectory.re,
                "u_in": trajectory.u_in,
                "nu": trajectory.nu,
                "tau": trajectory.tau,
                "n_steps": trajectory.n_steps,
            },
            blob_root=blob_root,
            mass_tol=mass_tol,
            extra_metrics={"cd": frame.cd, "cl": frame.cl},
        )
        product_ids.append(pid)
    return tuple(product_ids)
