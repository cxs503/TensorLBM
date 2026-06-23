from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib
import torch

if TYPE_CHECKING:
    from collections.abc import Callable

from .boundaries import (
    apply_simple_channel_boundaries,
    bounce_back_cells,
    compute_obstacle_forces,
    cylinder_mask,
    make_channel_wall_mask,
    zou_he_inlet_velocity,
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
    speed: torch.Tensor,
    vort: torch.Tensor,
    obstacle: torch.Tensor,
) -> None:
    speed_np = speed.detach().cpu().numpy()
    vort_np = vort.detach().cpu().numpy()
    obs_np = obstacle.detach().cpu().numpy()

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
    turbulence_statistics: dict[str, object] | None = None,
    diagnostic_callback: Callable[[dict[str, object]], None] | None = None,
) -> Path:
    configure_logging()
    config.validate()
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
            "device": str(device),
            "num_threads": applied_num_threads,
        },
        "reproducibility": get_reproducibility_metadata(),
        "engineering_closure": {
            "synthetic_inflow": synthetic_inflow or {"enabled": False},
            "sponge_layer": sponge_layer or {"enabled": False},
            "turbulence_statistics": turbulence_statistics or {"enabled": False},
        },
    }

    cx_obs, cy_obs = config.nx * 0.25, config.ny * 0.5
    obstacle = cylinder_mask(config.nx, config.ny, cx_obs, cy_obs, config.radius, device=device)
    wall_mask = make_channel_wall_mask(config.ny, config.nx, obstacle, device=device)

    # Resume from checkpoint or initialise fresh
    start_step = 1
    if config.resume_checkpoint is not None:
        f, resume_step, _ckpt_meta = load_checkpoint(config.resume_checkpoint, device=device)
        f = f.to(device)
        start_step = resume_step + 1
        logger.info("Resumed from checkpoint %s at step %d", config.resume_checkpoint, resume_step)
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
            f = apply_simple_channel_boundaries(
                f,
                u_in=config.u_in,
                wall_mask=wall_mask,
                obstacle_mask=obstacle,
            )

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
