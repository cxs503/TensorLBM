#!/usr/bin/env python3
"""Diagnostic field/geometry snapshot for the DTMB 5415 case (m452).

Runs the case for a short while at one resolution and saves a mid-plane
(x-z at y = centreline) snapshot:

* top panel - the **solid mask** the solver actually integrates.  With the
  waterline-clipped (open) STL this is a pillar hanging from the waterline
  (the defect m438 measured as Cb ~ 18); with the capped STL it is the hull.
* bottom panel - |u| in the same plane, i.e. the flow field the force
  integral is taken from.

Used only for diagnosis/reporting; the benchmark itself stays in ``run.py``.

Usage
-----
  PYTHONPATH=src python _field_snapshot.py --resol 32 --steps 600 \
      --stl geometry/dtmb5415_underwater_closed.stl --out /tmp/field.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_HERE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from run import (  # noqa: E402
    BEAM,
    DRAFT,
    LPP,
    U_PHYS,
    CollisionModel,
    ForceMethod,
    GeneralSimConfig,
    GeneralSimEngine,
    GeometryConfig,
    GeometrySource,
    LatticeModel,
    OutputConfig,
    PhysicsConfig,
    SolverConfig,
    WallTreatment,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stl", default=str(_HERE / "geometry" / "dtmb5415_underwater_closed.stl"))
    ap.add_argument("--resol", type=int, default=32)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--device", default="sdaa:0")
    ap.add_argument("--out", default="/tmp/dtmb5415_field.png")
    args = ap.parse_args()

    L = args.resol
    config = GeneralSimConfig(
        name=f"snapshot_L{L}",
        geometry=GeometryConfig(
            source=GeometrySource.STL_FILE, stl_path=args.stl, stl_units="m"
        ),
        physics=PhysicsConfig(
            density=1000.0,
            viscosity=U_PHYS * LPP / 1000.0,
            inlet_velocity=U_PHYS,
            reference_length=LPP,
        ),
        solver=SolverConfig(
            lattice=LatticeModel.D3Q19,
            collision=CollisionModel.MRT,
            resolution=L,
            domain_padding=(1.0, 4.0, 1.0, 1.0, 1.0, 1.0),
            max_steps=args.steps,
            snapshot_interval=10_000_000,
            force_sample_interval=10,
            device=args.device,
            wall_treatment=WallTreatment.AUTO,
            force_method=ForceMethod.PRESSURE_FRICTION,
            pressure_extrap="none",
            p0_method="near_wall",
            friction_formula="mix50",
            mass_correction=True,
            divergence_check_interval=50,
        ),
        output=OutputConfig(
            directory="/tmp/m454/snapshot", formats=[], save_macroscopic=False, save_forces=True
        ),
    )
    engine = GeneralSimEngine(config)
    engine.setup()
    info = engine.run()
    print("run:", {k: info[k] for k in ("status", "steps") if k in info}, flush=True)

    from tensorlbm.drag_pressure import _macroscopic

    rho, ux, uy, uz = _macroscopic(engine.f)
    speed = torch_sqrt(ux * ux + uy * uy + uz * uz)
    k = engine.solid.shape[1] // 2  # y mid-plane
    solid_sl = engine.solid[:, k, :].cpu().numpy()
    speed_sl = speed[:, k, :].cpu().numpy()
    dx = LPP / L

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True)
    ax = axes[0]
    ax.imshow(solid_sl.T, origin="lower", aspect="auto", cmap="Greys", vmin=0, vmax=1,
              extent=[0, solid_sl.shape[0] * dx, 0, solid_sl.shape[1] * dx])
    ax.set_ylabel("z [m]")
    ax.set_title(
        f"solid mask, mid-plane y=0  ({Path(args.stl).name}, L={L}, "
        f"solid cells={int(np.asarray(solid_sl).sum())} in plane, "
        f"draft={DRAFT} m = {DRAFT / dx:.2f} cells)"
    )
    ax.axhline(DRAFT, color="tab:blue", lw=0.8, ls="--")
    ax = axes[1]
    im = ax.imshow(speed_sl.T, origin="lower", aspect="auto", cmap="viridis",
                   extent=[0, speed_sl.shape[0] * dx, 0, speed_sl.shape[1] * dx])
    ax.contour(solid_sl.T, levels=[0.5], colors="w", linewidths=0.6,
               extent=[0, solid_sl.shape[0] * dx, 0, solid_sl.shape[1] * dx])
    ax.axhline(DRAFT, color="tab:blue", lw=0.8, ls="--")
    ax.set_xlabel("x [m] (flow direction)")
    ax.set_ylabel("z [m]")
    ax.set_title(f"|u| mid-plane after {args.steps} steps (lattice units)")
    fig.colorbar(im, ax=ax, pad=0.01)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print("figure ->", args.out, flush=True)

    np.savez_compressed(
        str(Path(args.out).with_suffix(".npz")),
        solid=solid_sl,
        speed=speed_sl,
        dx=dx,
        stl=str(args.stl),
        L=L,
        steps=args.steps,
    )


def torch_sqrt(t):
    import torch

    return torch.sqrt(t)


if __name__ == "__main__":
    main()