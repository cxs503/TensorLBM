from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib
import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Callable

from .backends import get_ops, using_backend

from .boundaries import (
    apply_simple_channel_boundaries,
    bounce_back_cells,
    compute_obstacle_forces,
    cylinder_mask,
    make_channel_wall_mask,
    nscbc_outlet_2d,
    stabilize_outlet_backflow,
    zou_he_inlet_velocity,
    zou_he_outlet_pressure,
)
from .checkpoint import load_checkpoint, save_checkpoint
from .config_io import load_config_json, save_config_json
from .d2q9 import equilibrium, macroscopic
from .logging_config import configure_logging, logger
from .solver import collide_bgk, correct_mass, stream
from .turbulence_stats import TurbulenceStatsAccumulator, compute_turbulence_intensity
from .utils import (
    DiagnosticPoint,
    configure_cpu_threads,
    get_reproducibility_metadata,
    prepare_run_dir,
    resolve_device,
)

try:
    from tqdm import tqdm as _tqdm

    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_VALID_BACKENDS = frozenset({"torch", "paddle", "mindspore"})
_D2Q9_C = np.array(
    [
        [0, 0],
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, -1],
        [1, 1],
        [-1, 1],
        [-1, -1],
        [1, -1],
    ],
    dtype=np.int64,
)
_D2Q9_W = np.array(
    [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36],
    dtype=np.float32,
)
_D2Q9_OPPOSITE = [0, 3, 4, 1, 2, 7, 8, 5, 6]


def _maybe_compile(fn: Callable[..., Any], use_compile: bool) -> Callable[..., Any]:
    """Optionally wrap *fn* with ``torch.compile``.

    Falls back gracefully when ``torch.compile`` is unavailable (PyTorch <
    2.0) or when *use_compile* is ``False``.

    Args:
        fn: Callable to (optionally) compile.
        use_compile: Whether to attempt compilation.

    Returns:
        Either the compiled or the original callable.
    """
    if not use_compile:
        return fn
    try:
        return torch.compile(fn)
    except AttributeError:
        logger.warning(
            "torch.compile not available (requires PyTorch >= 2.0); running in eager mode"
        )
        return fn


@dataclass(frozen=True)
class CylinderFlowConfig:
    nx: int = 320
    ny: int = 100
    u_in: float = 0.08
    re: float = 100.0
    radius: float = 12.0
    n_steps: int = 1200
    output_interval: int = 200
    output_root: Path = Path("outputs")
    run_name: str | None = None
    seed: int = 0
    device: str = "cpu"
    backend: str = "torch"
    num_threads: int | None = None
    overwrite: bool = False
    resume_checkpoint: Path | None = None
    use_compile: bool = False
    collision: str = "bgk"
    mrt_s_e: float = 1.64
    mrt_s_eps: float = 1.54
    mrt_s_q: float = 1.7

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "device", self.device.lower())
        object.__setattr__(self, "backend", self.backend.lower())
        if self.resume_checkpoint is not None:
            object.__setattr__(self, "resume_checkpoint", Path(self.resume_checkpoint))

    @property
    def nu(self) -> float:
        return self.u_in * 2.0 * self.radius / self.re

    @property
    def tau(self) -> float:
        return 3.0 * self.nu + 0.5

    def validate(self) -> None:
        if self.nx < 16 or self.ny < 8:
            msg = "nx and ny must be at least 16 and 8"
            raise ValueError(msg)
        if self.n_steps < 1:
            msg = "n_steps must be >= 1"
            raise ValueError(msg)
        if self.output_interval < 1:
            msg = "output_interval must be >= 1"
            raise ValueError(msg)
        if self.u_in <= 0.0 or self.re <= 0.0 or self.radius <= 0.0:
            msg = "u_in, re, and radius must be > 0"
            raise ValueError(msg)
        if self.backend not in _VALID_BACKENDS:
            msg = f"Unsupported backend: {self.backend}"
            raise ValueError(msg)
        if self.num_threads is not None and self.num_threads < 1:
            msg = "num_threads must be >= 1"
            raise ValueError(msg)
        if self.tau <= 0.5:
            msg = f"Invalid tau={self.tau:.4f}; increase re or reduce u_in/radius"
            raise ValueError(msg)

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        re_label = str(int(self.re)) if float(self.re).is_integer() else f"{self.re:g}"
        return f"nx{self.nx}_ny{self.ny}_re{re_label}_uin{self.u_in:.3f}_steps{self.n_steps}"

    def save(self, path: str | Path) -> Path:
        """Save this config to a JSON file.

        Args:
            path: Output file path (should end with ``.json``).

        Returns:
            Resolved path to the written file.
        """
        return save_config_json(self, path)

    @classmethod
    def load(cls, path: str | Path) -> CylinderFlowConfig:
        """Load a :class:`CylinderFlowConfig` from a JSON file.

        Args:
            path: Path to a JSON file written by :meth:`save`.

        Returns:
            Reconstructed :class:`CylinderFlowConfig` instance.
        """
        return load_config_json(cls, path)


def compute_vorticity(ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    """Central-difference z-vorticity ∂uy/∂x − ∂ux/∂y (interior cells only)."""
    dux_dy = torch.zeros_like(ux)
    duy_dx = torch.zeros_like(uy)
    dux_dy[1:-1, :] = 0.5 * (ux[2:, :] - ux[:-2, :])
    duy_dx[:, 1:-1] = 0.5 * (uy[:, 2:] - uy[:, :-2])
    return duy_dx - dux_dy


def _to_numpy(x: object) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        try:
            return x.detach().cpu().numpy()
        except Exception:
            pass
    if hasattr(x, "asnumpy"):
        return x.asnumpy()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def _compute_vorticity_np(ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    dux_dy = np.zeros_like(ux)
    duy_dx = np.zeros_like(uy)
    dux_dy[1:-1, :] = 0.5 * (ux[2:, :] - ux[:-2, :])
    duy_dx[:, 1:-1] = 0.5 * (uy[:, 2:] - uy[:, :-2])
    return duy_dx - dux_dy


def _backend_cylinder_mask(
    ops: object,
    nx: int,
    ny: int,
    cx: float,
    cy: float,
    radius: float,
    *,
    device: str,
):
    yy, xx = ops.meshgrid(
        ops.arange(ny, dtype=ops.float32_dtype(), device=device),
        ops.arange(nx, dtype=ops.float32_dtype(), device=device),
        indexing="ij",
    )
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2


def _backend_make_channel_wall_mask(
    ops: object,
    ny: int,
    nx: int,
    obstacle_mask,
    *,
    device: str,
):
    yy, _ = ops.meshgrid(
        ops.arange(ny, dtype=ops.float32_dtype(), device=device),
        ops.arange(nx, dtype=ops.float32_dtype(), device=device),
        indexing="ij",
    )
    return ((yy == 0.0) | (yy == float(ny - 1))) & (~obstacle_mask)


def _backend_equilibrium(ops: object, rho, ux, uy, *, device: str):
    weights = ops.reshape(ops.tensor(_D2Q9_W, device=device), (9, 1, 1))
    c = ops.tensor(_D2Q9_C.astype(np.float32), device=device)
    cx = ops.reshape(c[:, 0], (9, 1, 1))
    cy = ops.reshape(c[:, 1], (9, 1, 1))
    u_sq = ux * ux + uy * uy
    cu = cx * ux + cy * uy
    return weights * ops.unsqueeze(rho, 0) * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * ops.unsqueeze(u_sq, 0))


def _backend_macroscopic(ops: object, f, *, device: str):
    c = ops.tensor(_D2Q9_C.astype(np.float32), device=device)
    cx = ops.reshape(c[:, 0], (9, 1, 1))
    cy = ops.reshape(c[:, 1], (9, 1, 1))
    rho = ops.sum_val(f, dim=0)
    rho_safe = ops.clamp_min(rho, 1e-12)
    ux = ops.sum_val(f * cx, dim=0) / rho_safe
    uy = ops.sum_val(f * cy, dim=0) / rho_safe
    return rho, ux, uy


def _backend_collide_bgk(ops: object, f, tau: float, *, device: str):
    rho, ux, uy = _backend_macroscopic(ops, f, device=device)
    feq = _backend_equilibrium(ops, rho, ux, uy, device=device)
    return f - (f - feq) / tau


def _backend_stream(ops: object, f):
    shifted = [
        ops.roll(f[i], shifts=(int(_D2Q9_C[i, 1]), int(_D2Q9_C[i, 0])), dims=(0, 1))
        for i in range(9)
    ]
    return ops.stack(shifted, dim=0)


def _backend_bounce_back_cells(ops: object, f, mask):
    bounced = ops.stack([f[i] for i in _D2Q9_OPPOSITE], dim=0)
    return ops.where(ops.unsqueeze(mask, 0), bounced, f)


def _backend_apply_simple_channel_boundaries(
    ops: object,
    f,
    *,
    u_in: float,
    wall_mask,
    obstacle_mask,
    device: str,
):
    rho, _, _ = _backend_macroscopic(ops, f, device=device)
    f_new = ops.clone(f)
    u_col = ops.ones((rho.shape[0], 1), device=device) * float(u_in)
    uy_col = ops.zeros((rho.shape[0], 1), device=device)
    feq_in = _backend_equilibrium(ops, rho[:, 1:2], u_col, uy_col, device=device)
    f_new[:, :, 0] = feq_in[:, :, 0]
    f_new[:, :, -1] = f_new[:, :, -2]
    f_new = _backend_bounce_back_cells(ops, f_new, wall_mask)
    f_new = _backend_bounce_back_cells(ops, f_new, obstacle_mask)
    return f_new


def _backend_compute_obstacle_forces(ops: object, f, obstacle_mask, *, device: str):
    c = ops.tensor(_D2Q9_C.astype(np.float32), device=device)
    cx = ops.reshape(c[:, 0], (9, 1, 1))
    cy = ops.reshape(c[:, 1], (9, 1, 1))
    f_solid = f * ops.unsqueeze(obstacle_mask, 0)
    fx = 2.0 * ops.sum_val(cx * f_solid)
    fy = 2.0 * ops.sum_val(cy * f_solid)
    return fx, fy


def _backend_correct_mass(ops: object, f, target_mass: float):
    current = ops.sum_val(f)
    current_scalar = ops.float_scalar(current)
    if abs(current_scalar) < 1e-30:
        return f
    return f * (target_mass / current_scalar)


def _non_torch_feature_enabled(settings: dict[str, object] | None) -> bool:
    return bool(settings and settings.get("enabled", False))


def _run_cylinder_flow_backend(
    config: CylinderFlowConfig,
    *,
    synthetic_inflow: dict[str, object] | None = None,
    sponge_layer: dict[str, object] | None = None,
    outlet_control: dict[str, object] | None = None,
    turbulence_statistics: dict[str, object] | None = None,
    diagnostic_callback: Callable[[dict[str, object]], None] | None = None,
) -> Path:
    configure_logging()
    if config.use_compile:
        raise ValueError("use_compile is only supported on the torch backend")
    if config.collision != "bgk":
        raise ValueError("Only bgk collision is currently supported on non-torch cylinder-flow backends")
    unsupported = {
        "synthetic_inflow": synthetic_inflow,
        "sponge_layer": sponge_layer,
        "outlet_control": outlet_control,
        "turbulence_statistics": turbulence_statistics,
    }
    enabled = [name for name, settings in unsupported.items() if _non_torch_feature_enabled(settings)]
    if enabled:
        names = ", ".join(sorted(enabled))
        raise ValueError(f"{names} are currently only supported on the torch backend")
    if config.device == "mps":
        raise ValueError("mps device is only supported on the torch backend")

    ops = get_ops()
    device = config.device
    run_dir = prepare_run_dir(
        config.output_root,
        "cylinder_flow",
        config.resolved_run_name(),
        config.overwrite,
    )

    ckpt_str = str(config.resume_checkpoint) if config.resume_checkpoint else None
    metadata: dict[str, object] = {
        "config": {
            **asdict(config),
            "output_root": str(config.output_root),
            "resume_checkpoint": ckpt_str,
        },
        "derived": {"nu": config.nu, "tau": config.tau},
        "runtime": {
            "torch_version": torch.__version__,
            "backend": config.backend,
            "device": device,
            "num_threads": config.num_threads,
        },
        "reproducibility": get_reproducibility_metadata(),
        "engineering_closure": {
            "synthetic_inflow": {"enabled": False},
            "sponge_layer": {"enabled": False},
            "outlet_control": {"enabled": False},
            "turbulence_statistics": {"enabled": False},
        },
    }

    cx_obs, cy_obs = config.nx * 0.25, config.ny * 0.5
    obstacle = _backend_cylinder_mask(
        ops,
        config.nx,
        config.ny,
        cx_obs,
        cy_obs,
        config.radius,
        device=device,
    )
    wall_mask = _backend_make_channel_wall_mask(
        ops,
        config.ny,
        config.nx,
        obstacle,
        device=device,
    )

    start_step = 1
    restart_info: dict[str, object] = {"resumed": False}
    if config.resume_checkpoint is not None:
        f_torch, resume_step, ckpt_meta = load_checkpoint(
            config.resume_checkpoint,
            expected_shape=(9, config.ny, config.nx),
            expected_lattice_directions=9,
        )
        if resume_step >= config.n_steps:
            raise ValueError(
                f"resume checkpoint step {resume_step} is not less than n_steps={config.n_steps}"
            )
        f = ops.to_device(ops.tensor(f_torch.detach().cpu().numpy()), device)
        start_step = resume_step + 1
        restart_info = {
            "resumed": True,
            "source_checkpoint": str(config.resume_checkpoint),
            "source_step": resume_step,
            "checkpoint_format_version": ckpt_meta.get("format_version"),
        }
    else:
        rho0 = ops.ones((config.ny, config.nx), device=device)
        ux0 = ops.ones((config.ny, config.nx), device=device) * config.u_in
        ux0 = ops.where(obstacle, ops.zeros((config.ny, config.nx), device=device), ux0)
        uy0 = ops.zeros((config.ny, config.nx), device=device)
        f = _backend_equilibrium(ops, rho0, ux0, uy0, device=device)

    initial_mass = float(config.nx * config.ny)
    diagnostics: list[dict[str, object]] = []
    fy_steps: list[object] = []
    diameter = 2.0 * config.radius
    dyn_pressure = 0.5 * config.u_in**2 * diameter

    logger.info(
        "Running D2Q9 cylinder flow backend=%s device=%s NX=%s NY=%s tau=%.4f steps=%s output_interval=%s",
        config.backend,
        device,
        config.nx,
        config.ny,
        config.tau,
        config.n_steps,
        config.output_interval,
    )
    logger.info("Run directory: %s", run_dir)

    step_range = range(start_step, config.n_steps + 1)
    step_iter = (
        _tqdm(step_range, desc="Cylinder flow", unit="step")
        if _TQDM_AVAILABLE
        else step_range
    )
    for step in step_iter:
        f = _backend_collide_bgk(ops, f, config.tau, device=device)
        f = _backend_stream(ops, f)
        fx, fy = _backend_compute_obstacle_forces(ops, f, obstacle, device=device)
        f = _backend_apply_simple_channel_boundaries(
            ops,
            f,
            u_in=config.u_in,
            wall_mask=wall_mask,
            obstacle_mask=obstacle,
            device=device,
        )
        fy_steps.append(ops.detach(fy))

        if step % config.output_interval == 0:
            f = _backend_correct_mass(ops, f, initial_mass)

        if step % config.output_interval == 0 or step == config.n_steps:
            cd = ops.float_scalar(fx) / dyn_pressure if dyn_pressure != 0.0 else float("nan")
            cl = ops.float_scalar(fy) / dyn_pressure if dyn_pressure != 0.0 else float("nan")

            rho, ux, uy = _backend_macroscopic(ops, f, device=device)
            zeros = ops.zeros((config.ny, config.nx), device=device)
            ux = ops.where(obstacle, zeros, ux)
            uy = ops.where(obstacle, zeros, uy)
            speed = ops.sqrt(ux * ux + uy * uy)
            mass = ops.float_scalar(ops.sum_val(rho))

            point = DiagnosticPoint(
                step=step,
                mass=mass,
                mass_drift=mass - initial_mass,
                max_speed=ops.float_scalar(ops.max_val(speed)),
                mean_rho=ops.float_scalar(ops.mean(rho)),
            )
            diag_entry: dict[str, object] = {**asdict(point), "cd": cd, "cl": cl}
            diagnostics.append(diag_entry)
            logger.info(
                "step=%5d mass=%.6f drift=%+.6f mean_rho=%.6f max|u|=%.6f Cd=%.4f Cl=%.4f",
                point.step,
                point.mass,
                point.mass_drift,
                point.mean_rho,
                point.max_speed,
                cd,
                cl,
            )

            _save_flow_snapshot(
                run_dir,
                step,
                speed,
                _compute_vorticity_np(_to_numpy(ux), _to_numpy(uy)),
                obstacle,
            )
            if diagnostic_callback is not None:
                diagnostic_callback(diag_entry)
            save_checkpoint(torch.from_numpy(_to_numpy(f)).to(dtype=torch.float32), step, run_dir)

    cl_series = [
        ops.float_scalar(fy_t) / dyn_pressure if dyn_pressure != 0.0 else float("nan")
        for fy_t in fy_steps
    ]
    half = len(cl_series) // 2
    st = _strouhal_number(cl_series[half:], 1, config.u_in, diameter)

    metadata["diagnostics"] = diagnostics
    metadata["restart"] = restart_info
    if st is not None:
        metadata["strouhal"] = st
        logger.info("Strouhal number St ≈ %.4f", st)

    forces_csv = run_dir / "forces.csv"
    with forces_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "cd", "cl"])
        for d in diagnostics:
            writer.writerow([d["step"], d["cd"], d["cl"]])

    metadata_path = run_dir / "run_metadata.json"
    metadata_path.write_text(
        f"{json.dumps(metadata, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    logger.info("Saved metadata: %s", metadata_path)
    return run_dir


def _strouhal_number(
    cl_series: list[float], output_interval: int, u_in: float, diameter: float
) -> float | None:
    """Estimate Strouhal number from the dominant frequency of the lift-coefficient series.

    Returns *None* when the series is too short or has no clear spectral peak.
    Uses numpy FFT (O(N log N)) rather than a manual DFT loop.
    """
    import numpy as np

    n = len(cl_series)
    if n < 16:
        return None
    n2 = 1
    while n2 * 2 <= n:
        n2 *= 2
    data = np.array(cl_series[:n2], dtype=np.float64)
    spectrum = np.abs(np.fft.rfft(data))
    best_k = int(np.argmax(spectrum[1:])) + 1
    if best_k <= 0:
        return None
    freq_lbm = best_k / (n2 * output_interval)
    return freq_lbm * diameter / u_in


def _save_flow_snapshot(
    run_dir: Path,
    step: int,
    speed: object,
    vort: object,
    obstacle: object,
) -> None:
    speed_np = _to_numpy(speed)
    vort_np = _to_numpy(vort)
    obs_np = _to_numpy(obstacle)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    im0 = axes[0].imshow(speed_np, origin="lower", cmap="viridis")
    axes[0].contour(obs_np, levels=[0.5], colors="white", linewidths=0.7)
    axes[0].set_title("Velocity magnitude")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(vort_np, origin="lower", cmap="coolwarm")
    axes[1].contour(obs_np, levels=[0.5], colors="black", linewidths=0.7)
    axes[1].set_title("Vorticity")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    out = run_dir / f"flow_step_{step:06d}.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _summarize_turbulence_stats(
    stats: TurbulenceStatsAccumulator,
    *,
    u_ref: float,
) -> dict[str, object]:
    tu = compute_turbulence_intensity(stats.tke, u_ref=max(u_ref, 1e-12))
    return {
        "n_samples": stats.count,
        "domain_mean_u": float(stats.mean_u.mean().item()),
        "domain_mean_v": float(stats.mean_v.mean().item()),
        "uu_mean": float(stats.uu.mean().item()),
        "vv_mean": float(stats.vv.mean().item()),
        "uv_mean": float(stats.uv.mean().item()),
        "tke_mean": float(stats.tke.mean().item()),
        "tu_percent_mean": float(tu.mean().item()),
        "wall_normal_profile": {
            "mean_u": stats.mean_u.mean(dim=1).cpu().tolist(),
            "uu": stats.uu.mean(dim=1).cpu().tolist(),
            "vv": stats.vv.mean(dim=1).cpu().tolist(),
            "uv": stats.uv.mean(dim=1).cpu().tolist(),
            "tke": stats.tke.mean(dim=1).cpu().tolist(),
            "tu_percent": tu.mean(dim=1).cpu().tolist(),
        },
    }


def run_cylinder_flow(
    config: CylinderFlowConfig,
    *,
    synthetic_inflow: dict[str, object] | None = None,
    sponge_layer: dict[str, object] | None = None,
    outlet_control: dict[str, object] | None = None,
    turbulence_statistics: dict[str, object] | None = None,
    diagnostic_callback: Callable[[dict[str, object]], None] | None = None,
) -> Path:
    config.validate()
    with using_backend(config.backend):
        if config.backend != "torch":
            return _run_cylinder_flow_backend(
                config,
                synthetic_inflow=synthetic_inflow,
                sponge_layer=sponge_layer,
                outlet_control=outlet_control,
                turbulence_statistics=turbulence_statistics,
                diagnostic_callback=diagnostic_callback,
            )
    configure_logging()
    configure_logging()
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    device = resolve_device(config.device)
    applied_num_threads = configure_cpu_threads(device, config.num_threads)
    run_dir = prepare_run_dir(
        config.output_root,
        "cylinder_flow",
        config.resolved_run_name(),
        config.overwrite,
    )

    ckpt_str = str(config.resume_checkpoint) if config.resume_checkpoint else None
    metadata: dict[str, object] = {
        "config": {
            **asdict(config),
            "output_root": str(config.output_root),
            "resume_checkpoint": ckpt_str,
        },
        "derived": {"nu": config.nu, "tau": config.tau},
        "runtime": {
            "torch_version": torch.__version__,
            "backend": config.backend,
            "device": str(device),
            "num_threads": applied_num_threads,
        },
        "reproducibility": get_reproducibility_metadata(),
        "engineering_closure": {
            "synthetic_inflow": synthetic_inflow or {"enabled": False},
            "sponge_layer": sponge_layer or {"enabled": False},
            "outlet_control": outlet_control or {"enabled": False},
            "turbulence_statistics": turbulence_statistics or {"enabled": False},
        },
    }

    cx_obs, cy_obs = config.nx * 0.25, config.ny * 0.5
    obstacle = cylinder_mask(config.nx, config.ny, cx_obs, cy_obs, config.radius, device=device)
    wall_mask = make_channel_wall_mask(config.ny, config.nx, obstacle, device=device)

    # Resume from checkpoint or initialise fresh
    start_step = 1
    restart_info: dict[str, object] = {"resumed": False}
    if config.resume_checkpoint is not None:
        f, resume_step, ckpt_meta = load_checkpoint(
            config.resume_checkpoint,
            device=device,
            expected_shape=(9, config.ny, config.nx),
            expected_lattice_directions=9,
        )
        if resume_step >= config.n_steps:
            raise ValueError(
                f"resume checkpoint step {resume_step} is not less than n_steps={config.n_steps}"
            )
        f = f.to(device)
        start_step = resume_step + 1
        logger.info("Resumed from checkpoint %s at step %d", config.resume_checkpoint, resume_step)
        restart_info = {
            "resumed": True,
            "source_checkpoint": str(config.resume_checkpoint),
            "source_step": resume_step,
            "checkpoint_format_version": ckpt_meta.get("format_version"),
        }
    else:
        rho0 = torch.ones((config.ny, config.nx), device=device)
        ux0 = torch.full((config.ny, config.nx), config.u_in, device=device)
        uy0 = torch.zeros((config.ny, config.nx), device=device)
        ux0[obstacle] = 0.0
        f = equilibrium(rho0, ux0, uy0, device=device)

    rho0_mass = torch.ones((config.ny, config.nx), device=device)
    initial_mass = float(rho0_mass.sum().item())
    diagnostics: list[dict[str, object]] = []
    # Accumulate fy as GPU tensors; defer .item() to post-loop to avoid per-step sync
    fy_steps: list[torch.Tensor] = []
    inlet_rms_history: list[float] = []
    sponge_profile_1d: torch.Tensor | None = None
    outlet_mode = "copy"
    outlet_sigma = 0.25
    outlet_rho_target = 1.0
    backflow_stabilization = False
    max_backflow_speed = 0.0
    turbulence_acc: TurbulenceStatsAccumulator | None = None
    turbulence_start_step = 0
    turbulence_sample_every = 1

    if synthetic_inflow and bool(synthetic_inflow.get("enabled", False)):
        from .synthetic_inflow import DFSEMInlet, DigitalFilterInlet

        u_mean = torch.full((config.ny, 1), config.u_in, device=device)
        inflow_kwargs = {
            "ny": config.ny,
            "nz": 1,
            "uu": float(synthetic_inflow.get("uu", 1e-4)),
            "vv": float(synthetic_inflow.get("vv", 1e-4)),
            "ww": float(synthetic_inflow.get("ww", 1e-4)),
            "uv": float(synthetic_inflow.get("uv", 0.0)),
            "uw": float(synthetic_inflow.get("uw", 0.0)),
            "vw": float(synthetic_inflow.get("vw", 0.0)),
            "length_scale": float(synthetic_inflow.get("length_scale", 5.0)),
            "device": device,
            "seed": int(config.seed + int(synthetic_inflow.get("seed_offset", 101))),
        }
        if str(synthetic_inflow.get("method", "dfsem")).lower() == "digital_filter":
            inflow_generator = DigitalFilterInlet(**inflow_kwargs)
        else:
            inflow_generator = DFSEMInlet(
                u_mean=u_mean,
                n_eddies=int(synthetic_inflow.get("n_eddies", 200)),
                **inflow_kwargs,
            )
    else:
        inflow_generator = None

    if sponge_layer and bool(sponge_layer.get("enabled", False)):
        from .sponge_bc import sponge_profile

        sponge_start = int(float(sponge_layer.get("start_fraction", 0.8)) * max(config.nx - 1, 1))
        sponge_profile_1d = sponge_profile(
            nx=config.nx,
            x0=min(max(0, sponge_start), config.nx - 1),
            x1=config.nx - 1,
            amplitude=float(sponge_layer.get("amplitude", 0.35)),
            exponent=float(sponge_layer.get("exponent", 3.0)),
            device=device,
        )
        metadata["engineering_closure"]["sponge_layer"] = {
            **(sponge_layer or {}),
            "x0": sponge_start,
            "x1": config.nx - 1,
        }

    if outlet_control and bool(outlet_control.get("enabled", False)):
        outlet_mode = str(outlet_control.get("mode", "nscbc")).lower()
        outlet_sigma = float(outlet_control.get("nscbc_sigma", 0.25))
        outlet_rho_target = float(outlet_control.get("rho_target", 1.0))
        backflow_stabilization = bool(outlet_control.get("backflow_stabilization", True))
        max_backflow_speed = float(outlet_control.get("max_backflow_speed", 0.0))
        metadata["engineering_closure"]["outlet_control"] = {
            **(outlet_control or {}),
            "mode": outlet_mode,
            "nscbc_sigma": outlet_sigma,
            "rho_target": outlet_rho_target,
            "backflow_stabilization": backflow_stabilization,
            "max_backflow_speed": max_backflow_speed,
        }

    if turbulence_statistics and bool(turbulence_statistics.get("enabled", False)):
        turbulence_acc = TurbulenceStatsAccumulator()
        turbulence_start_step = int(turbulence_statistics.get("start_step", 0))
        turbulence_sample_every = int(turbulence_statistics.get("sample_every", 1))

    diameter = 2.0 * config.radius
    dyn_pressure = 0.5 * config.u_in**2 * diameter

    # Optionally JIT-compile the hot-path kernels
    # Select collision operator: use f, tau=... form (tau captured in lambda)
    if config.collision == "mrt":
        from tensorlbm.solver import collide_mrt

        tau = config.tau

        def _collide_raw(f: torch.Tensor) -> torch.Tensor:
            return collide_mrt(f, tau, config.mrt_s_e, config.mrt_s_eps, config.mrt_s_q)
    else:
        tau = config.tau

        def _collide_raw(f: torch.Tensor) -> torch.Tensor:
            return collide_bgk(f, tau)

    _collide = _maybe_compile(_collide_raw, config.use_compile)
    _stream = _maybe_compile(stream, config.use_compile)

    logger.info(
        "Running D2Q9 cylinder flow device=%s NX=%s NY=%s tau=%.4f "
        "steps=%s output_interval=%s compile=%s num_threads=%s",
        device,
        config.nx,
        config.ny,
        config.tau,
        config.n_steps,
        config.output_interval,
        config.use_compile,
        applied_num_threads,
    )
    logger.info("Run directory: %s", run_dir)

    step_range = range(start_step, config.n_steps + 1)
    step_iter = (
        _tqdm(step_range, desc="Cylinder flow", unit="step")
        if _TQDM_AVAILABLE
        else step_range
    )
    for step in step_iter:
        f = _collide(f)
        f = _stream(f)
        if sponge_profile_1d is not None:
            from .sponge_bc import apply_viscous_sponge_2d

            rho_s, ux_s, uy_s = macroscopic(f)
            f = apply_viscous_sponge_2d(f, rho_s, ux_s, uy_s, tau, sponge_profile_1d)
        fx, fy = compute_obstacle_forces(f, obstacle)
        if inflow_generator is not None:
            _u_fluct, _v_fluct, _ = inflow_generator.sample()
            inlet_u = torch.clamp(
                config.u_in + _u_fluct[:, 0],
                min=max(config.u_in * 0.25, 1e-5),
                max=config.u_in * 2.0,
            )
            inlet_v = torch.clamp(
                _v_fluct[:, 0],
                min=-0.5 * config.u_in,
                max=0.5 * config.u_in,
            )
            inlet_rms_history.append(float((_u_fluct[:, 0].std()).item()))
            f = zou_he_inlet_velocity(f, inlet_u, inlet_v)
            f[:, :, -1] = f[:, :, -2]
            f = bounce_back_cells(f, wall_mask)
            f = bounce_back_cells(f, obstacle)
        else:
            if outlet_mode == "copy":
                f = apply_simple_channel_boundaries(
                    f,
                    u_in=config.u_in,
                    wall_mask=wall_mask,
                    obstacle_mask=obstacle,
                )
            else:
                f = zou_he_inlet_velocity(f, config.u_in)
                if outlet_mode == "zou_he":
                    f = zou_he_outlet_pressure(f, rho_out=outlet_rho_target)
                elif outlet_mode == "nscbc":
                    f = nscbc_outlet_2d(f, rho_target=outlet_rho_target, sigma=outlet_sigma)
                else:
                    raise ValueError(
                        f"Unsupported outlet_control mode '{outlet_mode}'. "
                        "Expected one of: copy, zou_he, nscbc."
                    )
                if backflow_stabilization:
                    f = stabilize_outlet_backflow(
                        f,
                        max_backflow_speed=max_backflow_speed,
                        solid_mask_col=(wall_mask[:, -1] | obstacle[:, -1]),
                    )
                f = bounce_back_cells(f, wall_mask)
                f = bounce_back_cells(f, obstacle)

        # Store fy as a GPU scalar tensor – no .item() sync on every step
        fy_steps.append(fy.detach())

        # Correct mass drift every output_interval steps
        if step % config.output_interval == 0:
            f = correct_mass(f, initial_mass)

        if step % config.output_interval == 0 or step == config.n_steps:
            # Sync only at output intervals (much less frequent)
            cd = float(fx.item()) / dyn_pressure if dyn_pressure != 0.0 else float("nan")
            cl = float(fy.item()) / dyn_pressure if dyn_pressure != 0.0 else float("nan")

            rho, ux, uy = macroscopic(f)
            ux = ux.masked_fill(obstacle, 0.0)
            uy = uy.masked_fill(obstacle, 0.0)
            speed = torch.sqrt(ux * ux + uy * uy)
            mass = float(rho.sum().item())
            if (
                turbulence_acc is not None
                and step >= turbulence_start_step
                and (step - turbulence_start_step) % turbulence_sample_every == 0
            ):
                turbulence_acc.update(ux, uy)

            point = DiagnosticPoint(
                step=step,
                mass=mass,
                mass_drift=mass - initial_mass,
                max_speed=float(speed.max().item()),
                mean_rho=float(rho.mean().item()),
            )
            diag_entry: dict[str, object] = {**asdict(point), "cd": cd, "cl": cl}
            diagnostics.append(diag_entry)
            if turbulence_acc is not None and turbulence_acc.count > 0:
                diag_entry["tke_mean"] = float(turbulence_acc.tke.mean().item())
            if inlet_rms_history:
                diag_entry["inlet_rms_u"] = inlet_rms_history[-1]
            logger.info(
                "step=%5d mass=%.6f drift=%+.6f mean_rho=%.6f max|u|=%.6f Cd=%.4f Cl=%.4f",
                point.step,
                point.mass,
                point.mass_drift,
                point.mean_rho,
                point.max_speed,
                cd,
                cl,
            )

            vort = compute_vorticity(ux, uy)
            _save_flow_snapshot(run_dir, step, speed, vort, obstacle)
            if diagnostic_callback is not None:
                diagnostic_callback(diag_entry)

            # Save checkpoint at every output step
            save_checkpoint(f, step, run_dir)

    # Batch-convert all stored fy values post-loop for Strouhal number computation
    cl_series = [
        float(fy_t.item()) / dyn_pressure if dyn_pressure != 0.0 else float("nan")
        for fy_t in fy_steps
    ]
    half = len(cl_series) // 2
    # cl_series is sampled every step (sample spacing = 1 step), not every
    # output_interval.  Pass spacing=1 so the frequency axis is in cycles/step.
    st = _strouhal_number(cl_series[half:], 1, config.u_in, diameter)

    metadata["diagnostics"] = diagnostics
    metadata["restart"] = restart_info
    if st is not None:
        metadata["strouhal"] = st
        logger.info("Strouhal number St ≈ %.4f", st)
    if inlet_rms_history:
        metadata["engineering_closure"]["synthetic_inflow_runtime"] = {
            "mean_u_rms": sum(inlet_rms_history) / len(inlet_rms_history),
            "last_u_rms": inlet_rms_history[-1],
        }
    if sponge_profile_1d is not None:
        metadata["engineering_closure"]["sponge_layer_runtime"] = {
            "max_strength": float(sponge_profile_1d.max().item()),
            "mean_strength": float(sponge_profile_1d.mean().item()),
        }
    if turbulence_acc is not None and turbulence_acc.count > 0:
        metadata["engineering_closure"]["turbulence_statistics_runtime"] = (
            _summarize_turbulence_stats(
                turbulence_acc,
                u_ref=config.u_in,
            )
        )

    forces_csv = run_dir / "forces.csv"
    with forces_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "cd", "cl"])
        for d in diagnostics:
            writer.writerow([d["step"], d["cd"], d["cl"]])

    metadata_path = run_dir / "run_metadata.json"
    metadata_path.write_text(
        f"{json.dumps(metadata, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    logger.info("Saved metadata: %s", metadata_path)
    return run_dir
