"""Run one 2-D turbulent-channel case and store its wall-normal time series.

This is the data-production half of the ``u+`` error-decomposition study: it
saves the x-averaged streamwise velocity profile ``<u>(y, t)`` sampled at a
fixed cadence, so that every downstream statistic (batch-means confidence
interval, integrated autocorrelation time, domain-length arbitration, SGS
negative control) can be recomputed from the stored array without re-running the
solver.

Usage::

    python benchmarks/turbulence_error_decomposition/run.py \
        --case base --warmup 8000 --samples 40000 --sample-every 2 \
        --out runs/base.npz
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from tensorlbm.boundaries import bounce_back_cells, make_channel_wall_mask
from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.solver import stream
from tensorlbm.turbulence import collide_smagorinsky_bgk


@dataclass(frozen=True)
class ChannelCase:
    """Configuration of one 2-D body-force-driven channel run."""

    name: str
    ny: int = 64
    re_tau: float = 180.0
    u_tau: float = 0.02
    smagorinsky_cs: float = 0.1
    box_factor: float = 2.0
    warmup: int = 8000
    samples: int = 40000
    sample_every: int = 2
    seed: int = 20261108
    noise: float = 0.1

    @property
    def delta(self) -> float:
        """Channel half-height in lattice units."""
        return 0.5 * self.ny

    @property
    def nx(self) -> int:
        return int(round(self.box_factor * math.pi * self.delta))

    @property
    def nu(self) -> float:
        return self.u_tau * self.delta / self.re_tau

    @property
    def tau(self) -> float:
        return 3.0 * self.nu + 0.5

    @property
    def body_force(self) -> float:
        return 2.0 * self.u_tau**2 / self.ny

    def metadata(self) -> dict[str, object]:
        payload = asdict(self)
        payload.update(
            {
                "delta": self.delta,
                "nx": self.nx,
                "nu": self.nu,
                "tau": self.tau,
                "body_force": self.body_force,
                "box_length_lattice": self.nx,
                "box_length_in_delta": self.nx / self.delta,
                "total_steps": self.warmup + self.samples * self.sample_every,
            },
        )
        return payload


def _add_body_force(f: torch.Tensor, acceleration: float) -> torch.Tensor:
    rho, ux, uy = macroscopic(f)
    return equilibrium(rho, ux + acceleration, uy, device=f.device)


def run_case(case: ChannelCase, *, threads: int = 8) -> dict[str, object]:
    """Integrate one case and return the stored profile time series."""
    torch.set_num_threads(threads)
    device = torch.device("cpu")
    ny, nx = case.ny, case.nx
    obstacle = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    wall = make_channel_wall_mask(ny, nx, obstacle, device)
    generator = torch.Generator(device=device).manual_seed(case.seed)
    rho = torch.ones((ny, nx), device=device)
    ux = case.u_tau * case.noise * torch.rand((ny, nx), generator=generator, device=device)
    uy = case.u_tau * case.noise * (torch.rand((ny, nx), generator=generator, device=device) - 0.5)
    ux[wall] = 0.0
    f = equilibrium(rho, ux, uy, device=device)

    started = time.time()
    total = case.warmup + case.samples * case.sample_every
    profiles = np.empty((case.samples, ny), dtype=np.float64)
    wall_shear = np.empty(case.samples, dtype=np.float64)
    index = 0
    for step in range(1, total + 1):
        f = collide_smagorinsky_bgk(f, tau=case.tau, C_s=case.smagorinsky_cs)
        f = stream(f)
        f = _add_body_force(f, case.body_force)
        f = bounce_back_cells(f, wall)
        if step > case.warmup and (step - case.warmup) % case.sample_every == 0:
            _, ux_field, _ = macroscopic(f)
            profiles[index] = ux_field.mean(dim=1).cpu().numpy()
            wall_shear[index] = float(
                (ux_field[1, :] - ux_field[0, :]).abs().mean().item(),
            )
            index += 1
    elapsed = time.time() - started
    y = np.arange(ny, dtype=np.float64)
    wall_distance = np.minimum(y, ny - 1 - y) + 0.5
    return {
        "case": case,
        "profiles": profiles,
        "wall_shear": wall_shear,
        "wall_distance": wall_distance,
        "elapsed_seconds": elapsed,
        "steps_per_second": total / elapsed,
    }


def save_run(payload: dict[str, object], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    case: ChannelCase = payload["case"]  # type: ignore[assignment]
    np.savez_compressed(
        target,
        profiles=payload["profiles"],
        wall_shear=payload["wall_shear"],
        wall_distance=payload["wall_distance"],
        metadata=json.dumps(
            {
                **case.metadata(),
                "elapsed_seconds": payload["elapsed_seconds"],
                "steps_per_second": payload["steps_per_second"],
            },
        ),
    )
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="base")
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--re-tau", type=float, default=180.0)
    parser.add_argument("--u-tau", type=float, default=0.02)
    parser.add_argument("--cs", type=float, default=0.1)
    parser.add_argument("--box-factor", type=float, default=2.0)
    parser.add_argument("--warmup", type=int, default=8000)
    parser.add_argument("--samples", type=int, default=40000)
    parser.add_argument("--sample-every", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    case = ChannelCase(
        name=args.case,
        ny=args.ny,
        re_tau=args.re_tau,
        u_tau=args.u_tau,
        smagorinsky_cs=args.cs,
        box_factor=args.box_factor,
        warmup=args.warmup,
        samples=args.samples,
        sample_every=args.sample_every,
        seed=args.seed,
        noise=args.noise,
    )
    payload = run_case(case, threads=args.threads)
    path = save_run(payload, args.out)
    print(
        json.dumps(
            {
                "out": str(path),
                "case": case.name,
                "nx": case.nx,
                "ny": case.ny,
                "steps_per_second": round(float(payload["steps_per_second"]), 2),
                "elapsed_seconds": round(float(payload["elapsed_seconds"]), 1),
                "mean_shear_velocity": float(payload["wall_shear"].mean()),
            },
        ),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())