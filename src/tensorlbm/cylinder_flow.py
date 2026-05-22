from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import torch

from .boundaries import apply_simple_channel_boundaries, compute_obstacle_forces, cylinder_mask, make_channel_wall_mask
from .d2q9 import equilibrium, macroscopic
from .solver import collide_bgk, stream
from .utils import DiagnosticPoint, prepare_run_dir, resolve_device

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

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "device", self.device.lower())

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
    dux_dy[1:-1, :] = 0.5 * (ux[2:, :] - ux[:-2, :])   # ∂ux/∂y
    duy_dx[:, 1:-1] = 0.5 * (uy[:, 2:] - uy[:, :-2])   # ∂uy/∂x
    return duy_dx - dux_dy


def _strouhal_number(cl_series: list[float], output_interval: int, u_in: float, diameter: float) -> float | None:
    """Estimate Strouhal number from the dominant frequency of the lift-coefficient series.

    Returns *None* when the series is too short or has no clear spectral peak.
    Uses numpy FFT (O(N log N)) rather than a manual DFT loop.
    """
    import numpy as np

    n = len(cl_series)
    if n < 16:
        return None
    # Zero-pad to next power of two for a clean spectral estimate
    n2 = 1
    while n2 * 2 <= n:
        n2 *= 2
    data = np.array(cl_series[:n2], dtype=np.float64)
    # Real FFT: rfft returns n2//2 + 1 complex bins; bin 0 is DC
    spectrum = np.abs(np.fft.rfft(data))
    # Ignore DC bin (index 0); find the peak among bins 1..n2//2
    best_k = int(np.argmax(spectrum[1:])) + 1
    if best_k <= 0:
        return None
    freq_lbm = best_k / (n2 * output_interval)  # cycles per LBM step
    return freq_lbm * diameter / u_in


def _save_flow_snapshot(run_dir: Path, step: int, speed: torch.Tensor, vort: torch.Tensor, obstacle: torch.Tensor) -> None:
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
    config.validate()
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    device = resolve_device(config.device)
    run_dir = prepare_run_dir(config.output_root, "cylinder_flow", config.resolved_run_name(), config.overwrite)

    metadata: dict[str, object] = {
        "config": {**asdict(config), "output_root": str(config.output_root)},
        "derived": {"nu": config.nu, "tau": config.tau},
        "runtime": {"torch_version": torch.__version__, "device": str(device)},
    }

    cx_obs, cy_obs = config.nx * 0.25, config.ny * 0.5
    obstacle = cylinder_mask(config.nx, config.ny, cx_obs, cy_obs, config.radius, device=device)
    wall_mask = make_channel_wall_mask(config.ny, config.nx, obstacle, device=device)

    rho0 = torch.ones((config.ny, config.nx), device=device)
    ux0 = torch.full((config.ny, config.nx), config.u_in, device=device)
    uy0 = torch.zeros((config.ny, config.nx), device=device)
    ux0[obstacle] = 0.0
    f = equilibrium(rho0, ux0, uy0, device=device)

    initial_mass = float(rho0.sum().item())
    diagnostics: list[dict[str, object]] = []
    cl_series: list[float] = []

    # Reference quantities for non-dimensional force coefficients
    diameter = 2.0 * config.radius
    dyn_pressure = 0.5 * 1.0 * config.u_in ** 2 * diameter  # rho_ref = 1

    print(
        "Running D2Q9 cylinder flow "
        f"device={device} NX={config.nx} NY={config.ny} tau={config.tau:.4f} "
        f"steps={config.n_steps} output_interval={config.output_interval}"
    )
    print(f"Run directory: {run_dir}")

    for step in range(1, config.n_steps + 1):
        f = collide_bgk(f, tau=config.tau)
        f = stream(f)
        # Compute drag/lift via momentum exchange BEFORE bounce-back is applied
        fx, fy = compute_obstacle_forces(f, obstacle)
        f = apply_simple_channel_boundaries(f, u_in=config.u_in, wall_mask=wall_mask, obstacle_mask=obstacle)

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
            print(
                f"step={point.step:5d} mass={point.mass:.6f} "
                f"drift={point.mass_drift:+.6f} mean_rho={point.mean_rho:.6f} "
                f"max|u|={point.max_speed:.6f} Cd={cd:.4f} Cl={cl:.4f}"
            )

            vort = compute_vorticity(ux, uy)
            _save_flow_snapshot(run_dir, step, speed, vort, obstacle)

    # Strouhal number from second half of lift time-series (avoids transient)
    half = len(cl_series) // 2
    st = _strouhal_number(cl_series[half:], config.output_interval, config.u_in, diameter)

    metadata["diagnostics"] = diagnostics
    if st is not None:
        metadata["strouhal"] = st
        print(f"Strouhal number St ≈ {st:.4f}")

    # Save per-step force time-series as CSV for post-processing
    forces_csv = run_dir / "forces.csv"
    with forces_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "cd", "cl"])
        for d in diagnostics:
            writer.writerow([d["step"], d["cd"], d["cl"]])

    metadata_path = run_dir / "run_metadata.json"
    metadata_path.write_text(f"{json.dumps(metadata, indent=2, sort_keys=True)}\n", encoding="utf-8")
    print(f"Saved metadata: {metadata_path}")
    return run_dir
