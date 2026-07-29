"""Free-Energy 3D sphere water entry — D3Q19 phase-field.

Uses free_energy_step_3d with Boussinesq buoyancy, Korteweg capillary force,
and neutral phase-field (phi=0) at obstacle cells to avoid gradient singularities.
"""
import csv, json, argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")

from tensorlbm.boundaries3d import bounce_back_cells_3d
from tensorlbm.d3q19 import C as C3
from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.multiphase3d import free_energy_step_3d, init_free_energy_g_3d
from tensorlbm.solver3d import stream3d
from tensorlbm.utils import flow_step_image_path, prepare_run_dir, resolve_device, write_legacy_snapshot_alias


@dataclass(frozen=True)
class FE3DConfig:
    nx: int = 64; ny: int = 64; nz: int = 64
    radius: float = 6.0
    water_level: int = 32; clearance: int = 3
    rho_heavy: float = 0.8; rho_light: float = 0.4
    tau_f: float = 1.5; tau_g: float = 0.7
    A: float = 0.1; B: float = 0.1
    kappa: float = 0.02; Gamma: float = 0.5
    g: float = 2e-5
    n_steps: int = 2000; output_interval: int = 200
    output_root: str = "outputs"; run_name: str = "fe_3d"
    device: str = "cpu"; overwrite: bool = False

    def __post_init__(self):
        object.__setattr__(self, "output_root", str(Path(self.output_root)))

    @property
    def sphere_center(self):
        cx = self.nx // 2; cy = self.ny // 2
        cz = self.water_level + self.clearance + int(self.radius) + 1
        return float(cx), float(cy), float(cz)


def run_fe_3d(config: FE3DConfig):
    device = resolve_device(config.device)
    nz, ny, nx = config.nz, config.ny, config.nx
    cx, cy, cz = config.sphere_center

    run_dir = prepare_run_dir(Path(config.output_root), "fe_water_entry_3d", config.run_name, config.overwrite)
    print(f"FE 3D Water Entry  {nx}x{ny}x{nz}  R={config.radius}  k={config.kappa}")

    # Walls
    wall = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    wall[0] = wall[-1] = True; wall[:, 0] = wall[:, -1] = True; wall[:, :, 0] = wall[:, :, -1] = True

    # Sphere
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32), indexing="ij")
    sphere = ((xx - cx)**2 + (yy - cy)**2 + (zz - cz)**2) <= config.radius**2
    solid = wall | sphere

    # Phase field: +1 below water_level (water), -1 above (air)
    z = torch.arange(nz, dtype=torch.float32, device=device)
    prof = 0.5 * (1.0 - torch.tanh((z - config.water_level) / 6.0))
    phi = 2.0 * prof.view(nz, 1, 1).expand(nz, ny, nx) - 1.0
    phi = phi.clone()

    zero = torch.zeros((nz, ny, nx), device=device)
    rho = torch.ones_like(phi)
    f = equilibrium3d(rho, zero, zero, zero)
    g = init_free_energy_g_3d(phi, zero, zero, zero)

    c_dev = C3.to(device)
    gz_body = -config.g
    force_series = []

    for step in range(1, config.n_steps + 1):
        f, g = free_energy_step_3d(
            f, g,
            tau_f=config.tau_f, tau_g=config.tau_g,
            A=config.A, B=config.B, kappa=config.kappa, Gamma=config.Gamma,
            gz=gz_body,
            rho_heavy=config.rho_heavy, rho_light=config.rho_light,
        )
        f = stream3d(f)
        g = stream3d(g)

        if step % config.output_interval == 0 or step == config.n_steps:
            ft = f + g
            cx_d = c_dev[:, 0].float().view(19, 1, 1, 1)
            cy_d = c_dev[:, 1].float().view(19, 1, 1, 1)
            cz_d = c_dev[:, 2].float().view(19, 1, 1, 1)
            fs = ft * sphere.unsqueeze(0)
            fx = 2.0 * float((cx_d * fs).sum())
            fy = 2.0 * float((cy_d * fs).sum())
            fz = 2.0 * float((cz_d * fs).sum())
            phi_cur = g.sum(dim=0)
            mean_phi = float(phi_cur[~solid].mean())

            entry = {"step": step, "fx": round(fx, 8), "fy": round(fy, 8),
                     "fz": round(fz, 8), "mean_phi": round(mean_phi, 6)}
            force_series.append(entry)
            print(f"step={step:5d}  Fx={fx:.4e}  Fy={fy:.4e}  Fz={fz:.4e}  mean_φ={mean_phi:.4f}")

        # f: bounce-back.  g: neutral (phi=0) at obstacle
        f = bounce_back_cells_3d(f, solid)
        g_zero = init_free_energy_g_3d(torch.zeros_like(phi), zero, zero, zero)
        g = torch.where(solid.unsqueeze(0), g_zero, g)

    forces_csv = run_dir / "forces.csv"
    with forces_csv.open("w", newline="", encoding="utf-8") as fh:
        if force_series:
            w = csv.DictWriter(fh, fieldnames=list(force_series[0].keys()))
            w.writeheader(); w.writerows(force_series)

    md = {"config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
          "forces": force_series, "model": "Free-Energy 3D (D3Q19)"}
    (run_dir / "run_metadata.json").write_text(json.dumps(md, indent=2, sort_keys=True) + "\n")
    print(f"Saved → {run_dir / 'run_metadata.json'}")
    return run_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--kappa", type=float, default=0.02)
    p.add_argument("--n-steps", type=int, default=2000)
    p.add_argument("--run-name", default="fe_3d")
    p.add_argument("--output-root", default="outputs")
    p.add_argument("--nx", type=int, default=64)
    p.add_argument("--ny", type=int, default=64)
    p.add_argument("--nz", type=int, default=64)
    p.add_argument("--radius", type=float, default=6.0)
    args = p.parse_args()

    cfg = FE3DConfig(
        nx=args.nx, ny=args.ny, nz=args.nz, radius=args.radius,
        kappa=args.kappa, n_steps=args.n_steps,
        device=args.device, run_name=args.run_name,
        output_root=args.output_root, overwrite=True,
    )
    run_fe_3d(cfg)
