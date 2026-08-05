"""Cylinder flow benchmark endpoint — compare CPU/SDAA × f32/bf16 speeds."""

import time
from fastapi import APIRouter
from pydantic import BaseModel

import torch

from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.solver import collide_bgk, stream
from tensorlbm.boundaries import (
    apply_simple_channel_boundaries,
    cylinder_mask,
    make_channel_wall_mask,
)

router = APIRouter()


class BenchRequest(BaseModel):
    nx: int = 200
    ny: int = 80
    u_in: float = 0.08
    re: float = 100.0
    radius: float = 10.0
    warmup: int = 100
    bench_steps: int = 500


@router.post("/run")
def bench_run(req: BenchRequest):
    """Run cylinder flow benchmark on all 4 configs: CPU/f32, CPU/bf16, SDAA/f32, SDAA/bf16."""
    nx, ny = req.nx, req.ny
    u_in, re, radius = req.u_in, req.re, req.radius
    cx_obs = nx * 0.25
    cy_obs = ny * 0.5
    tau = 3.0 * u_in * 2.0 * radius / re + 0.5

    configs = [
        ("cpu", "float32", torch.float32, torch.device("cpu")),
        ("cpu", "bfloat16", torch.bfloat16, torch.device("cpu")),
        ("sdaa", "float32", torch.float32, torch.device("sdaa")),
        ("sdaa", "bfloat16", torch.bfloat16, torch.device("sdaa")),
    ]

    results = []
    for dev_name, dtype_name, dtype, device in configs:
        try:
            obstacle = cylinder_mask(nx, ny, cx_obs, cy_obs, radius, device=device)
            wall_mask = make_channel_wall_mask(ny, nx, obstacle, device=device)

            rho0 = torch.ones((ny, nx), device=device, dtype=dtype)
            ux0 = torch.full((ny, nx), u_in, device=device, dtype=dtype)
            uy0 = torch.zeros((ny, nx), device=device, dtype=dtype)
            ux0[obstacle] = 0.0
            f = equilibrium(rho0, ux0, uy0, device=device)

            # Warmup
            for _ in range(req.warmup):
                f = collide_bgk(f, tau)
                f = stream(f)
                f = apply_simple_channel_boundaries(f, u_in=u_in, wall_mask=wall_mask, obstacle_mask=obstacle)

            # Synchronize before benchmark
            if dev_name == "sdaa":
                torch.sdaa.synchronize()

            t0 = time.perf_counter()
            for _ in range(req.bench_steps):
                f = collide_bgk(f, tau)
                f = stream(f)
                f = apply_simple_channel_boundaries(f, u_in=u_in, wall_mask=wall_mask, obstacle_mask=obstacle)

            if dev_name == "sdaa":
                torch.sdaa.synchronize()
            t1 = time.perf_counter()

            elapsed = t1 - t0
            ms_per_step = elapsed / req.bench_steps * 1000
            steps_per_sec = req.bench_steps / elapsed

            # Check stability
            rho, ux, uy = macroscopic(f)
            rho_min = float(rho.min().item())
            rho_max = float(rho.max().item())
            speed_max = float(torch.sqrt(ux * ux + uy * uy).max().item())

            results.append({
                "device": dev_name,
                "dtype": dtype_name,
                "ms_per_step": round(ms_per_step, 2),
                "steps_per_sec": round(steps_per_sec, 1),
                "elapsed_s": round(elapsed, 3),
                "rho_min": round(rho_min, 4),
                "rho_max": round(rho_max, 4),
                "speed_max": round(speed_max, 6),
                "status": "ok",
            })
        except Exception as e:
            results.append({
                "device": dev_name,
                "dtype": dtype_name,
                "ms_per_step": None,
                "steps_per_sec": None,
                "elapsed_s": None,
                "rho_min": None,
                "rho_max": None,
                "speed_max": None,
                "status": f"error: {e}",
            })

    return {"configs": results, "params": {"nx": nx, "ny": ny, "u_in": u_in, "re": re, "radius": radius, "tau": round(tau, 4), "warmup": req.warmup, "bench_steps": req.bench_steps}}
