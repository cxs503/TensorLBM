"""D3Q27 sphere-flow example.

Demonstrates D3Q27 channel flow past a sphere using BGK or MRT collision.
The D3Q27 lattice includes all 8 corner velocity directions and achieves
4th-order isotropy, reducing numerical artefacts at moderate-to-high Re
compared with the standard D3Q19 lattice.

Usage
-----
.. code-block:: bash

    PYTHONPATH=src python examples/sphere_flow_d3q27.py \\
        --nx 60 --ny 30 --nz 30 --radius 4 --n-steps 100 --output-interval 50 \\
        --run-name smoke --overwrite
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import torch

from tensorlbm import (
    DiagnosticPoint,
    configure_logging,
    equilibrium27,
    get_reproducibility_metadata,
    logger,
    macroscopic27,
    prepare_run_dir,
    resolve_device,
    save_checkpoint,
    sphere_mask,
)
from tensorlbm.boundaries_d3q27 import (
    apply_zou_he_channel_boundaries_27,
    make_channel_wall_mask_27,
)
from tensorlbm.d3q27 import collide_bgk27, collide_mrt27, stream27

try:
    from tqdm import tqdm as _tqdm  # type: ignore[import-untyped]

    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

matplotlib.use("Agg")
from dataclasses import asdict

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class SphereFlowD3Q27Config:
    """Configuration for a D3Q27 3-D sphere-flow simulation."""

    nx: int = 120
    ny: int = 60
    nz: int = 60
    u_in: float = 0.06
    re: float = 50.0
    radius: float = 8.0
    n_steps: int = 500
    output_interval: int = 100
    output_root: Path = Path("outputs")
    run_name: str | None = None
    seed: int = 0
    device: str = "cpu"
    overwrite: bool = False
    use_mrt: bool = False

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
        if self.nx < 16 or self.ny < 8 or self.nz < 8:
            raise ValueError("nx, ny, and nz must be at least 16, 8, and 8")
        if self.n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        if self.output_interval < 1:
            raise ValueError("output_interval must be >= 1")
        if self.u_in <= 0.0 or self.re <= 0.0 or self.radius <= 0.0:
            raise ValueError("u_in, re, and radius must be > 0")
        if self.tau <= 0.5:
            raise ValueError(
                f"Invalid tau={self.tau:.4f}; increase re or reduce u_in/radius"
            )

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        re_label = str(int(self.re)) if float(self.re).is_integer() else f"{self.re:g}"
        coll = "mrt" if self.use_mrt else "bgk"
        return (
            f"d3q27_{coll}_nx{self.nx}_ny{self.ny}_nz{self.nz}_re{re_label}"
            f"_uin{self.u_in:.3f}_steps{self.n_steps}"
        )


def _save_snapshot(
    run_dir: Path,
    step: int,
    speed: torch.Tensor,
    obstacle: torch.Tensor,
    nz: int,
) -> None:
    """Save velocity magnitude on mid-z slice as PNG."""
    mid_z = nz // 2
    speed_np = speed[mid_z].detach().cpu().numpy()
    obs_np = obstacle[mid_z].detach().cpu().float().numpy()

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    im = ax.imshow(speed_np, origin="lower", cmap="viridis")
    ax.contour(obs_np, levels=[0.5], colors="white", linewidths=0.7)
    ax.set_title(f"D3Q27 speed – mid-z slice (step {step})")
    plt.colorbar(im, ax=ax, fraction=0.046)

    out = run_dir / f"flow_step_{step:06d}.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)


def run_sphere_flow_d3q27(config: SphereFlowD3Q27Config) -> Path:
    """Run a D3Q27 sphere-flow simulation and save results."""
    configure_logging()
    config.validate()
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    device = resolve_device(config.device)
    run_dir = prepare_run_dir(
        config.output_root,
        "sphere_flow_d3q27",
        config.resolved_run_name(),
        config.overwrite,
    )

    metadata: dict[str, object] = {
        "config": {**asdict(config), "output_root": str(config.output_root)},
        "derived": {"nu": config.nu, "tau": config.tau},
        "runtime": {"torch_version": torch.__version__, "device": str(device)},
        "reproducibility": get_reproducibility_metadata(),
    }

    cx = config.nx * 0.25
    cy = config.ny * 0.5
    cz = config.nz * 0.5
    obstacle = sphere_mask(
        config.nx, config.ny, config.nz, cx, cy, cz, config.radius, device=device
    )
    wall_mask = make_channel_wall_mask_27(config.nz, config.ny, config.nx, obstacle, device=device)

    rho0 = torch.ones((config.nz, config.ny, config.nx), device=device)
    ux0 = torch.full_like(rho0, config.u_in)
    uy0 = torch.zeros_like(rho0)
    uz0 = torch.zeros_like(rho0)
    ux0[obstacle] = 0.0
    f = equilibrium27(rho0, ux0, uy0, uz0, device=device)

    initial_mass = float(rho0.sum().item())
    diagnostics: list[dict[str, float | int]] = []

    logger.info(
        "Running D3Q27 sphere flow device=%s NX=%s NY=%s NZ=%s tau=%.4f steps=%s "
        "output_interval=%s collision=%s",
        device,
        config.nx,
        config.ny,
        config.nz,
        config.tau,
        config.n_steps,
        config.output_interval,
        "MRT" if config.use_mrt else "BGK",
    )
    logger.info("Run directory: %s", run_dir)

    step_range = range(1, config.n_steps + 1)
    step_iter = (
        _tqdm(step_range, desc="D3Q27 sphere flow", unit="step")
        if _TQDM_AVAILABLE
        else step_range
    )
    for step in step_iter:
        f = collide_mrt27(f, tau=config.tau) if config.use_mrt else collide_bgk27(f, tau=config.tau)
        f = stream27(f)
        f = apply_zou_he_channel_boundaries_27(
            f, u_in=config.u_in, wall_mask=wall_mask, obstacle_mask=obstacle
        )

        if step % config.output_interval == 0 or step == config.n_steps:
            rho, ux, uy, uz = macroscopic27(f)
            ux = ux.masked_fill(obstacle, 0.0)
            uy = uy.masked_fill(obstacle, 0.0)
            uz = uz.masked_fill(obstacle, 0.0)
            speed = torch.sqrt(ux * ux + uy * uy + uz * uz)
            mass = float(rho.sum().item())

            point = DiagnosticPoint(
                step=step,
                mass=mass,
                mass_drift=mass - initial_mass,
                max_speed=float(speed.max().item()),
                mean_rho=float(rho.mean().item()),
            )
            diagnostics.append(asdict(point))
            logger.info(
                "step=%5d mass=%.6f drift=%+.6f mean_rho=%.6f max|u|=%.6f",
                point.step,
                point.mass,
                point.mass_drift,
                point.mean_rho,
                point.max_speed,
            )
            _save_snapshot(run_dir, step, speed, obstacle, config.nz)
            save_checkpoint(f, step, run_dir)

    metadata["diagnostics"] = diagnostics
    metadata_path = run_dir / "run_metadata.json"
    metadata_path.write_text(
        f"{json.dumps(metadata, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    logger.info("Saved metadata: %s", metadata_path)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a D3Q27 sphere-flow LBM demonstration."
    )
    parser.add_argument("--nx", type=int, default=120, help="Grid length (x)")
    parser.add_argument("--ny", type=int, default=60, help="Grid height (y)")
    parser.add_argument("--nz", type=int, default=60, help="Grid depth  (z)")
    parser.add_argument(
        "--u-in", dest="u_in", type=float, default=0.06, help="Inlet velocity"
    )
    parser.add_argument("--re", type=float, default=50.0, help="Target Reynolds number")
    parser.add_argument("--radius", type=float, default=8.0, help="Sphere radius")
    parser.add_argument(
        "--n-steps", dest="n_steps", type=int, default=500, help="Simulation steps"
    )
    parser.add_argument(
        "--output-interval",
        type=int,
        default=100,
        help="Diagnostic and image cadence",
    )
    parser.add_argument("--output-root", default="outputs", help="Output root directory")
    parser.add_argument(
        "--run-name", default=None, help="Override deterministic run folder name"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default="cpu", help="Execution device"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output directory if it already exists",
    )
    parser.add_argument(
        "--mrt",
        dest="use_mrt",
        action="store_true",
        help="Use MRT collision operator (default: BGK)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = SphereFlowD3Q27Config(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        u_in=args.u_in,
        re=args.re,
        radius=args.radius,
        n_steps=args.n_steps,
        output_interval=args.output_interval,
        output_root=args.output_root,
        run_name=args.run_name,
        seed=args.seed,
        device=args.device,
        overwrite=args.overwrite,
        use_mrt=args.use_mrt,
    )
    run_sphere_flow_d3q27(config)


if __name__ == "__main__":
    main()
