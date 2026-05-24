"""Benchmark API endpoints.

Each endpoint runs a tensorlbm benchmark suite in the background via the
job manager and returns a job_id for status polling.
"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from .. import job_manager

router = APIRouter()


# ---------------------------------------------------------------------------
# Marine benchmark suite
# ---------------------------------------------------------------------------

class MarineBenchmarkParams(BaseModel):
    cases: list[Literal["cylinder", "sloshing", "pipeline", "turbulent_channel", "wigley"]] = Field(
        default=["cylinder", "sloshing", "pipeline", "turbulent_channel", "wigley"],
        description="Which benchmark cases to run",
    )
    fast: bool = Field(True, description="Use reduced step counts for quick validation")
    device: str = "cpu"


@router.post("/marine")
async def run_marine(params: MarineBenchmarkParams) -> dict:
    """Run the marine / ship-and-ocean engineering benchmark suite."""

    def _run(job: job_manager.Job) -> dict:
        results: dict[str, object] = {}
        output_root = job.output_dir

        if "cylinder" in params.cases:
            from tensorlbm import CylinderFlowConfig, run_cylinder_flow

            cfg = CylinderFlowConfig(
                nx=200 if params.fast else 320,
                ny=80 if params.fast else 100,
                n_steps=500 if params.fast else 20000,
                output_interval=100 if params.fast else 2000,
                device=params.device,
                output_root=output_root / "cylinder",
                overwrite=True,
            )
            run_cylinder_flow(cfg)
            results["cylinder"] = "ok"

        if "sloshing" in params.cases:
            from tensorlbm import SloshingTankConfig, run_sloshing_tank

            cfg2 = SloshingTankConfig(
                nx=100 if params.fast else 200,
                ny=80 if params.fast else 160,
                water_level=40 if params.fast else 80,
                n_steps=600 if params.fast else 6000,
                output_interval=100 if params.fast else 600,
                device=params.device,
                output_root=output_root / "sloshing",
                overwrite=True,
            )
            run_sloshing_tank(cfg2)
            results["sloshing"] = "ok"

        if "pipeline" in params.cases:
            from tensorlbm import PipelineFlowConfig, run_pipeline_flow

            cfg3 = PipelineFlowConfig(
                nx=200 if params.fast else 400,
                ny=80 if params.fast else 160,
                n_steps=1000 if params.fast else 30000,
                output_interval=200 if params.fast else 5000,
                device=params.device,
                output_root=output_root / "pipeline",
                overwrite=True,
            )
            run_pipeline_flow(cfg3)
            results["pipeline"] = "ok"

        if "turbulent_channel" in params.cases:
            from tensorlbm import TurbulentChannelConfig, run_turbulent_channel

            cfg4 = TurbulentChannelConfig(
                nx=64 if params.fast else 256,
                ny=32 if params.fast else 64,
                n_steps=1000 if params.fast else 50000,
                averaging_start=500 if params.fast else 20000,
                output_interval=200 if params.fast else 5000,
                device=params.device,
                output_root=output_root / "turbulent_channel",
                overwrite=True,
            )
            run_turbulent_channel(cfg4)
            results["turbulent_channel"] = "ok"

        if "wigley" in params.cases:
            from tensorlbm import ShipHullFlowConfig, run_ship_hull_flow

            cfg5 = ShipHullFlowConfig(
                nx=80 if params.fast else 160,
                ny=30 if params.fast else 60,
                nz=20 if params.fast else 40,
                n_steps=200 if params.fast else 2000,
                output_interval=50 if params.fast else 200,
                device=params.device,
                output_root=output_root / "wigley",
                overwrite=True,
            )
            run_ship_hull_flow(cfg5)
            results["wigley"] = "ok"

        return results

    job_id = job_manager.submit(
        name=f"Marine Benchmarks ({'fast' if params.fast else 'full'})",
        job_type="benchmark_marine",
        config=params.model_dump(),
        fn=_run,
    )
    return {"job_id": job_id, "message": "Marine benchmark submitted"}


# ---------------------------------------------------------------------------
# Multiphase benchmark suite
# ---------------------------------------------------------------------------

class MultiphaseBenchmarkParams(BaseModel):
    fast: bool = True
    device: str = "cpu"


@router.post("/multiphase")
async def run_multiphase(params: MultiphaseBenchmarkParams) -> dict:
    """Run the multiphase LBM benchmark suite (static droplet, spinodal, Poiseuille)."""

    def _run(job: job_manager.Job) -> dict:
        from tensorlbm import (
            MultiphaseBenchmarkSuiteConfig,
            run_multiphase_benchmark_suite,
        )

        cfg = MultiphaseBenchmarkSuiteConfig(
            fast=params.fast,
            device=params.device,
            output_root=job.output_dir,
        )
        result = run_multiphase_benchmark_suite(cfg)
        return {"summary": str(result)}

    job_id = job_manager.submit(
        name=f"Multiphase Benchmarks ({'fast' if params.fast else 'full'})",
        job_type="benchmark_multiphase",
        config=params.model_dump(),
        fn=_run,
    )
    return {"job_id": job_id, "message": "Multiphase benchmark submitted"}


# ---------------------------------------------------------------------------
# Lid-driven cavity – Ghia comparison
# ---------------------------------------------------------------------------

class GhiaBenchmarkParams(BaseModel):
    nx: int = Field(64, ge=16, description="Grid size (square)")
    re: Literal[100, 400, 1000] = 100
    n_steps: int = Field(5000, ge=1)
    device: str = "cpu"


@router.post("/ghia")
async def run_ghia(params: GhiaBenchmarkParams) -> dict:
    """Run lid-driven cavity and compare against Ghia et al. (1982) reference."""

    def _run(job: job_manager.Job) -> dict:
        from tensorlbm import (
            GHIA_RE100,
            GHIA_RE400,
            GHIA_RE1000,
            LidDrivenCavityConfig,
            compare_ghia,
            run_lid_driven_cavity,
        )

        cfg = LidDrivenCavityConfig(
            nx=params.nx,
            re=float(params.re),
            n_steps=params.n_steps,
            output_interval=max(params.n_steps // 5, 1),
            device=params.device,
            output_root=job.output_dir,
            overwrite=True,
        )
        run_lid_driven_cavity(cfg)

        # Load the last checkpoint and compare
        from tensorlbm import load_checkpoint, macroscopic

        ckpts = sorted(job.output_dir.rglob("checkpoint_*.pt"), key=lambda p: p.stem)
        if ckpts:
            f, _step = load_checkpoint(ckpts[-1])
            _rho, ux, uy = macroscopic(f)
            ref = {100: GHIA_RE100, 400: GHIA_RE400, 1000: GHIA_RE1000}[params.re]
            err = compare_ghia(ux, uy, ref)
            return {"re": params.re, "ghia_error": err}
        return {"re": params.re, "ghia_error": None}

    job_id = job_manager.submit(
        name=f"Ghia Benchmark Re={params.re}",
        job_type="benchmark_ghia",
        config=params.model_dump(),
        fn=_run,
    )
    return {"job_id": job_id, "message": "Ghia comparison benchmark submitted"}


# ---------------------------------------------------------------------------
# MLUPS performance benchmark
# ---------------------------------------------------------------------------

class MLUPSParams(BaseModel):
    sizes: list[int] = Field(
        default=[128, 256, 512],
        description="Grid sizes to benchmark (nx = ny = size)",
    )
    steps: int = Field(100, ge=10, description="Steps per size")
    device: str = "cpu"


@router.post("/mlups")
async def run_mlups(params: MLUPSParams) -> dict:
    """Measure D2Q9 BGK performance in MLUPS (Million Lattice Updates Per Second)."""

    def _run(job: job_manager.Job) -> dict:
        import time

        import torch

        from tensorlbm import collide_bgk, equilibrium, macroscopic, stream

        device = torch.device(params.device)
        results: list[dict] = []
        for size in params.sizes:
            ny, nx = size, size
            rho = torch.ones((ny, nx), dtype=torch.float32, device=device)
            u = torch.zeros((ny, nx, 2), dtype=torch.float32, device=device)
            f = equilibrium(rho, u)
            tau = 0.6

            # Warm-up
            for _ in range(10):
                rho2, ux2, uy2 = macroscopic(f)
                feq = equilibrium(rho2, torch.stack([ux2, uy2], dim=-1))
                f = collide_bgk(f, feq, tau)
                f = stream(f)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            t0 = time.perf_counter()
            for _ in range(params.steps):
                rho2, ux2, uy2 = macroscopic(f)
                feq = equilibrium(rho2, torch.stack([ux2, uy2], dim=-1))
                f = collide_bgk(f, feq, tau)
                f = stream(f)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - t0

            mlups = (nx * ny * params.steps) / elapsed / 1e6
            results.append({
                "size": size,
                "nx": nx,
                "ny": ny,
                "steps": params.steps,
                "elapsed_s": round(elapsed, 3),
                "mlups": round(mlups, 2),
            })

        import json

        (job.output_dir / "mlups_results.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
        return {"results": results}

    job_id = job_manager.submit(
        name=f"MLUPS Benchmark ({params.device})",
        job_type="benchmark_mlups",
        config=params.model_dump(),
        fn=_run,
    )
    return {"job_id": job_id, "message": "MLUPS benchmark submitted"}


# ---------------------------------------------------------------------------
# Porous media benchmarks
# ---------------------------------------------------------------------------

class PorousBenchmarkParams(BaseModel):
    fast: bool = True
    device: str = "cpu"


@router.post("/porous")
async def run_porous(params: PorousBenchmarkParams) -> dict:
    """Run porous media drainage and capillary invasion benchmarks."""

    def _run(job: job_manager.Job) -> dict:
        results: dict[str, object] = {}

        from tensorlbm import LaplaceTestConfig, run_laplace_test

        laplace_cfg = LaplaceTestConfig(
            fast=params.fast,
            device=params.device,
            output_root=job.output_dir / "laplace",
        )
        run_laplace_test(laplace_cfg)
        results["laplace"] = "ok"

        from tensorlbm import CapillaryInvasionConfig, run_capillary_invasion

        cap_cfg = CapillaryInvasionConfig(
            fast=params.fast,
            device=params.device,
            output_root=job.output_dir / "capillary",
        )
        run_capillary_invasion(cap_cfg)
        results["capillary_invasion"] = "ok"

        return results

    job_id = job_manager.submit(
        name=f"Porous Media Benchmarks ({'fast' if params.fast else 'full'})",
        job_type="benchmark_porous",
        config=params.model_dump(),
        fn=_run,
    )
    return {"job_id": job_id, "message": "Porous media benchmark submitted"}
