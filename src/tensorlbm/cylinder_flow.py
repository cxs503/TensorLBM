from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import torch

from .boundaries import (
    apply_simple_channel_boundaries,
    compute_obstacle_forces,
    cylinder_mask,
    make_channel_wall_mask,
)
from .checkpoint import load_checkpoint, save_checkpoint
from .d2q9 import equilibrium, macroscopic
from .logging_config import configure_logging, logger
from .solver import collide_bgk, stream
from .utils import (
    DiagnosticPoint,
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
    overwrite: bool = False
    resume_checkpoint: Path | None = None

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
        if self.tau <= 0.5:
            msg = f"Invalid tau={self.tau:.4f}; increase re or reduce u_in/radius"
            raise ValueError(msg)

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        re_label = str(int(self.re)) if float(self.re).is_integer() else f"{self.re:g}"
        return f"nx{self.nx}_ny{self.ny}_re{re_label}_uin{self.u_in:.3f}_steps{self.n_steps}"


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


def run_cylinder_flow(config: CylinderFlowConfig) -> Path:
    configure_logging()
    config.validate()
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    device = resolve_device(config.device)
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
        "runtime": {"torch_version": torch.__version__, "device": str(device)},
        "reproducibility": get_reproducibility_metadata(),
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
    cl_series: list[float] = []

    diameter = 2.0 * config.radius
    dyn_pressure = 0.5 * config.u_in**2 * diameter

    logger.info(
        "Running D2Q9 cylinder flow device=%s NX=%s NY=%s tau=%.4f steps=%s output_interval=%s",
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
        f = collide_bgk(f, tau=config.tau)
        f = stream(f)
        fx, fy = compute_obstacle_forces(f, obstacle)
        f = apply_simple_channel_boundaries(
            f,
            u_in=config.u_in,
            wall_mask=wall_mask,
            obstacle_mask=obstacle,
        )

        cd = float(fx) / dyn_pressure if dyn_pressure != 0.0 else float("nan")
        cl = float(fy) / dyn_pressure if dyn_pressure != 0.0 else float("nan")
        cl_series.append(cl)

        if step % config.output_interval == 0 or step == config.n_steps:
            rho, ux, uy = macroscopic(f)
            ux = ux.masked_fill(obstacle, 0.0)
            uy = uy.masked_fill(obstacle, 0.0)
            speed = torch.sqrt(ux * ux + uy * uy)
            mass = float(rho.sum().item())

            point = DiagnosticPoint(
                step=step,
                mass=mass,
                mass_drift=mass - initial_mass,
                max_speed=float(speed.max().item()),
                mean_rho=float(rho.mean().item()),
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

            vort = compute_vorticity(ux, uy)
            _save_flow_snapshot(run_dir, step, speed, vort, obstacle)

            # Save checkpoint at every output step
            save_checkpoint(f, step, run_dir)

    half = len(cl_series) // 2
    st = _strouhal_number(cl_series[half:], config.output_interval, config.u_in, diameter)

    metadata["diagnostics"] = diagnostics
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
