"""Free-Energy (FE) phase-field water entry benchmark — 2D cylinder.

Uses the Swift et al. binary-fluid formulation with:
  - Phase field φ = Σg (order parameter)
  - Korteweg capillary force + Boussinesq buoyancy
  - Rigid cylinder obstacle (bounce-back), initially above water
  - Gravity drives the cylinder downward
"""
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from tensorlbm.boundaries import bounce_back_cells
from tensorlbm.d2q9 import C as C2
from tensorlbm.d2q9 import equilibrium
from tensorlbm.multiphase import free_energy_step, init_free_energy_g
from tensorlbm.solver import stream
from tensorlbm.utils import flow_step_image_path, prepare_run_dir, resolve_device, write_legacy_snapshot_alias

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class FEWaterEntryConfig:
    nx: int = 200
    ny: int = 160
    radius: float = 12.0
    water_level: int = 80
    clearance: int = 4
    rho_heavy: float = 0.8   # effective density for phi=+1 (water) — must match CG
    rho_light: float = 0.4    # effective density for phi=-1 (air)
    tau_f: float = 1.5        # more viscosity for stability
    tau_g: float = 0.7
    A: float = 0.1
    B: float = 0.1
    kappa: float = 0.02
    Gamma: float = 0.5
    g: float = 2e-5           # lower gravity for stability
    n_steps: int = 3000
    output_interval: int = 300
    output_root: str = "outputs"
    run_name: str = "fe_2d"
    device: str = "cpu"
    overwrite: bool = False

    def __post_init__(self):
        object.__setattr__(self, "output_root", str(Path(self.output_root)))
        object.__setattr__(self, "device", self.device.lower())

    @property
    def sphere_center(self):
        cx = self.nx // 2
        cy = self.water_level + self.clearance + int(self.radius) + 1
        return float(cx), float(cy)


def _circle_mask(ny, nx, cx, cy, r, device):
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2


def _wall_mask(ny, nx, device):
    mask = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = True
    return mask


def _momentum_exchange(f, g, sphere, c_dev):
    """Ladd momentum-exchange on FE f+g distributions."""
    f_total = f + g  # (9, ny, nx)
    cx = c_dev[:, 0].float().view(9, 1, 1)
    cy = c_dev[:, 1].float().view(9, 1, 1)
    mask = sphere.unsqueeze(0)
    f_sol = f_total * mask
    fx = 2.0 * float((cx * f_sol).sum().item())
    fy = 2.0 * float((cy * f_sol).sum().item())
    return fx, fy


def run_fe_water_entry(config: FEWaterEntryConfig):
    device = resolve_device(config.device)
    ny, nx = config.ny, config.nx
    cx, cy = config.sphere_center

    run_dir = prepare_run_dir(
        Path(config.output_root), "fe_water_entry", config.run_name, config.overwrite,
    )
    print(f"FE Water Entry  nx={nx} ny={ny}  R={config.radius}  kappa={config.kappa}")
    print(f"Run dir: {run_dir}")

    # Walls + sphere
    wall = _wall_mask(ny, nx, device)
    sphere = _circle_mask(ny, nx, cx, cy, config.radius, device)
    solid = wall | sphere

    # Phase field: +1 below water_level, -1 above (tanh profile, wider interface)
    y = torch.arange(ny, dtype=torch.float32, device=device)
    w = 6.0  # wider interface for stability
    prof = 0.5 * (1.0 - torch.tanh((y - config.water_level) / w))
    phi = 2.0 * prof - 1.0  # maps [0,1] to [-1, +1]
    phi = phi.view(ny, 1).expand(ny, nx).clone()
    phi[sphere] = 0.0

    # Init FE distributions
    zero = torch.zeros((ny, nx), device=device)
    rho = torch.ones_like(phi)
    f = equilibrium(rho, zero, zero)
    g = init_free_energy_g(phi, zero, zero)

    c_dev = C2.to(device)
    gy = -config.g

    force_series = []

    for step in range(1, config.n_steps + 1):
        # FE collision with Boussinesq buoyancy
        f, g = free_energy_step(
            f, g,
            tau_f=config.tau_f, tau_g=config.tau_g,
            A=config.A, B=config.B, kappa=config.kappa, Gamma=config.Gamma,
            gy=gy,
            rho_heavy=config.rho_heavy, rho_light=config.rho_light,
        )

        # Stream
        f = stream(f)
        g = stream(g)

        # Diagnostics before bounce-back
        if step % config.output_interval == 0 or step == config.n_steps:
            fx_s, fy_s = _momentum_exchange(f, g, sphere, c_dev)
            phi_cur = g.sum(dim=0)
            rho_cur = f.sum(dim=0).clamp(min=1e-12)
            mean_phi = float(phi_cur[~solid].mean().item())

            entry = {"step": step, "fx": round(fx_s, 8), "fy": round(fy_s, 8),
                     "mean_phi": round(mean_phi, 6)}
            force_series.append(entry)
            print(f"step={step:5d}  Fx={fx_s:.4e}  Fy={fy_s:.4e}  mean_φ={mean_phi:.4f}")

            # Snapshot
            phi_img = (phi_cur * 0.5 + 0.5).detach().cpu().numpy()
            obs_np = sphere.detach().cpu().float().numpy()
            fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
            im = ax.imshow(phi_img, origin="lower", cmap="RdBu", vmin=0, vmax=1)
            ax.contour(obs_np, levels=[0.5], colors="black", linewidths=1.5)
            plt.colorbar(im, ax=ax, fraction=0.03, label="Phase field φ")
            ax.set_title(f"FE Water Entry — step {step}")
            out = flow_step_image_path(run_dir, step)
            fig.savefig(out, dpi=120)
            write_legacy_snapshot_alias(run_dir, step)
            plt.close(fig)

        # Bounce-back on f only at solid cells.  For g (order parameter), set to
        # neutral equilibrium (phi=0) inside the obstacle — sharp bounce-back on
        # the phase field creates gradient singularities that crash FE.
        f = bounce_back_cells(f, solid)
        # g: neutral order parameter at solid cells
        g_zero = init_free_energy_g(torch.zeros_like(phi), zero, zero)
        g = torch.where(solid.unsqueeze(0), g_zero, g)

    # Write outputs
    forces_csv = run_dir / "forces.csv"
    with forces_csv.open("w", newline="", encoding="utf-8") as fh:
        if force_series:
            writer = csv.DictWriter(fh, fieldnames=list(force_series[0].keys()))
            writer.writeheader()
            writer.writerows(force_series)

    metadata = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
        "forces": force_series,
        "model": "Free-Energy (Swift et al.)",
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Saved metadata → {run_dir / 'run_metadata.json'}")
    return run_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--kappa", type=float, default=0.02)
    parser.add_argument("--n-steps", type=int, default=3000)
    parser.add_argument("--run-name", default=None, help="Single run name (skips kappa sweep)")
    parser.add_argument("--output-root", default="outputs")
    args = parser.parse_args()

    if args.run_name:
        config = FEWaterEntryConfig(
            kappa=args.kappa, n_steps=args.n_steps,
            device=args.device, run_name=args.run_name,
            output_root=args.output_root, overwrite=True,
        )
        run_fe_water_entry(config)
    else:
        for kappa in [0.01, 0.02, 0.05]:
            config = FEWaterEntryConfig(
                kappa=kappa, n_steps=args.n_steps,
                device=args.device, run_name=f"fe_k{kappa:.2f}",
                output_root=args.output_root, overwrite=True,
            )
            run_fe_water_entry(config)
