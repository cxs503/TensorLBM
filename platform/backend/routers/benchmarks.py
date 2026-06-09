"""Benchmark API endpoints.

Each endpoint runs a tensorlbm benchmark suite in the background via the
job manager and returns a job_id for status polling.
"""
from __future__ import annotations

from fastapi import APIRouter
from .. import job_manager
from ..schemas.benchmarks import (
    AccuracyBenchmarkParams,
    GhiaBenchmarkParams,
    MLUPSParams,
    MarineBenchmarkParams,
    MultiphaseBenchmarkParams,
    PorousBenchmarkParams,
)
from ..services.benchmarks import submit_benchmark

router = APIRouter()


# ---------------------------------------------------------------------------
# Marine benchmark suite
# ---------------------------------------------------------------------------



@router.post("/marine")
async def run_marine(params: MarineBenchmarkParams) -> dict:
    """Run the marine / ship-and-ocean engineering benchmark suite."""

    def _run(job: job_manager.Job) -> dict:
        results: dict[str, object] = {}
        output_root = job.output_dir

        if "cylinder" in params.cases:
            job_manager.raise_if_cancelled(job.job_id)
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
            job_manager.raise_if_cancelled(job.job_id)
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
            job_manager.raise_if_cancelled(job.job_id)
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
            job_manager.raise_if_cancelled(job.job_id)
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
            job_manager.raise_if_cancelled(job.job_id)
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

        if "suboff" in params.cases:
            job_manager.raise_if_cancelled(job.job_id)
            from tensorlbm import SuboffResistanceBenchmarkConfig, run_suboff_resistance_benchmark

            cfg6 = SuboffResistanceBenchmarkConfig(
                hull_type="full",
                max_iterations=3 if params.fast else 4,
                target_error_pct=3.0,
                device=params.device,
            )
            results["suboff"] = run_suboff_resistance_benchmark(cfg6)

        if "geometry_library" in params.cases:
            job_manager.raise_if_cancelled(job.job_id)
            from tensorlbm import (
                ShipHullType,
                SuboffHullType,
                build_ship_hull_mask,
                build_suboff_mask,
            )

            nx, ny, nz = (128, 64, 48) if not params.fast else (80, 40, 30)
            ship_entries: list[dict[str, object]] = []
            for hull_type in (ShipHullType.WIGLEY, ShipHullType.SERIES60, ShipHullType.KCS):
                _mask, stats = build_ship_hull_mask(
                    hull_type=hull_type,
                    nx=nx,
                    ny=ny,
                    nz=nz,
                    length=nx * 0.5,
                    beam=ny * 0.22,
                    draft=nz * 0.25,
                    device=params.device,
                )
                cb_sim = float(stats["Cb_numerical"])
                cb_ref = float(stats["Cb"])
                cb_err = abs(cb_sim - cb_ref) / (abs(cb_ref) + 1e-12) * 100.0
                ship_entries.append({
                    "hull_type": hull_type.value,
                    "cb_sim": cb_sim,
                    "cb_ref": cb_ref,
                    "cb_error_pct": cb_err,
                    "pass": cb_err < 35.0,
                })

            suboff_entries: list[dict[str, object]] = []
            solid_cells: list[int] = []
            for hull_type in (
                SuboffHullType.BARE_HULL,
                SuboffHullType.WITH_SAIL,
                SuboffHullType.FULL,
            ):
                _mask, stats = build_suboff_mask(
                    hull_type=hull_type,
                    nx=nx,
                    ny=ny,
                    nz=nz,
                    length=nx * 0.6,
                    device=params.device,
                )
                solid = int(stats["solid_cells"])
                solid_cells.append(solid)
                suboff_entries.append({
                    "hull_type": hull_type.value,
                    "solid_cells": solid,
                    "l_d_ratio": float(stats["L_D_ratio"]),
                })

            cb_sim_values = [float(item["cb_sim"]) for item in ship_entries]
            cb_order_ok = cb_sim_values[0] < cb_sim_values[1] < cb_sim_values[2]
            ship_ok = all(bool(item["pass"]) for item in ship_entries) and cb_order_ok
            suboff_ok = (
                solid_cells == sorted(solid_cells)
                and len(set(solid_cells)) == len(solid_cells)
            )
            results["geometry_library"] = {
                "name": "marine_geometry_library",
                "ship": ship_entries,
                "suboff": suboff_entries,
                "ship_ok": ship_ok,
                "cb_order_ok": cb_order_ok,
                "suboff_ok": suboff_ok,
                "all_ok": ship_ok and suboff_ok,
            }

        return results

    return submit_benchmark(
        name=f"Marine Benchmarks ({'fast' if params.fast else 'full'})",
        job_type="benchmark_marine",
        params=params,
        runner=_run,
        message="Marine benchmark submitted",
    )


# ---------------------------------------------------------------------------
# Multiphase benchmark suite
# ---------------------------------------------------------------------------



@router.post("/multiphase")
async def run_multiphase(params: MultiphaseBenchmarkParams) -> dict:
    """Run the multiphase LBM benchmark suite including phase-field droplet tests."""

    def _run(job: job_manager.Job) -> dict:
        job_manager.raise_if_cancelled(job.job_id)
        from tensorlbm import (
            MultiphaseBenchmarkSuiteConfig,
            run_multiphase_benchmark_suite,
        )

        if params.fast:
            # Build reduced sub-configs for a quick smoke / CI run.
            from tensorlbm import (  # noqa: I001
                FreeEnergyDropletConfig,
                Spinodal3DConfig,
                SpinodaleConfig,
                StaticDroplet3DConfig,
                StaticDropletConfig,
                TwoPhaseChannelCompareConfig,
            )

            droplet = StaticDropletConfig(
                nx=40, ny=40, radii=(8.0,), n_steps=200, output_interval=100,
            )
            spinodal = SpinodaleConfig(
                nx=32, ny=32, n_steps=200, output_interval=100,
            )
            free_energy = FreeEnergyDropletConfig(
                nx=40, ny=40, radius=8.0, n_steps=200, output_interval=100,
            )
            droplet_3d = StaticDroplet3DConfig(
                nx=20, ny=20, nz=20, radii=(4.0,), n_steps=100, output_interval=100,
            )
            spinodal_3d = Spinodal3DConfig(
                nx=20, ny=20, nz=20, n_steps=120, output_interval=120,
            )
            poiseuille = TwoPhaseChannelCompareConfig()
            # Some installations may have different field names; use only
            # the safe top-level overrides.
            cfg = MultiphaseBenchmarkSuiteConfig(
                droplet=droplet,
                spinodal=spinodal,
                free_energy=free_energy,
                poiseuille=poiseuille,
                droplet_3d=droplet_3d,
                spinodal_3d=spinodal_3d,
                device=params.device,
                output_root=job.output_dir,
                overwrite=True,
            )
        else:
            cfg = MultiphaseBenchmarkSuiteConfig(
                device=params.device,
                output_root=job.output_dir,
                overwrite=True,
            )
        result = run_multiphase_benchmark_suite(cfg)
        return {"summary": str(result)}

    return submit_benchmark(
        name=f"Multiphase Benchmarks ({'fast' if params.fast else 'full'})",
        job_type="benchmark_multiphase",
        params=params,
        runner=_run,
        message="Multiphase benchmark submitted",
    )


# ---------------------------------------------------------------------------
# Lid-driven cavity – Ghia comparison
# ---------------------------------------------------------------------------



@router.post("/ghia")
async def run_ghia(params: GhiaBenchmarkParams) -> dict:
    """Run lid-driven cavity and compare against Ghia et al. (1982) reference."""

    def _run(job: job_manager.Job) -> dict:
        job_manager.raise_if_cancelled(job.job_id)
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

    return submit_benchmark(
        name=f"Ghia Benchmark Re={params.re}",
        job_type="benchmark_ghia",
        params=params,
        runner=_run,
        message="Ghia comparison benchmark submitted",
    )


# ---------------------------------------------------------------------------
# MLUPS performance benchmark
# ---------------------------------------------------------------------------



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
            job_manager.raise_if_cancelled(job.job_id)
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

    return submit_benchmark(
        name=f"MLUPS Benchmark ({params.device})",
        job_type="benchmark_mlups",
        params=params,
        runner=_run,
        message="MLUPS benchmark submitted",
    )


# ---------------------------------------------------------------------------
# Porous media benchmarks
# ---------------------------------------------------------------------------



@router.post("/porous")
async def run_porous(params: PorousBenchmarkParams) -> dict:
    """Run porous media drainage and capillary invasion benchmarks."""

    def _run(job: job_manager.Job) -> dict:
        job_manager.raise_if_cancelled(job.job_id)
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
        job_manager.raise_if_cancelled(job.job_id)
        run_capillary_invasion(cap_cfg)
        results["capillary_invasion"] = "ok"

        return results

    return submit_benchmark(
        name=f"Porous Media Benchmarks ({'fast' if params.fast else 'full'})",
        job_type="benchmark_porous",
        params=params,
        runner=_run,
        message="Porous media benchmark submitted",
    )


# ---------------------------------------------------------------------------
# Single-phase accuracy benchmark suite
# ---------------------------------------------------------------------------



@router.post("/accuracy")
async def run_accuracy(params: AccuracyBenchmarkParams) -> dict:
    """Run the single-phase accuracy benchmark suite.

    Covers lid-driven cavity (Ghia 1982), backward-facing step (Armaly 1983),
    and rotating cylinder / Magnus effect (Mittal & Kumar 2003).
    """

    def _run(job: job_manager.Job) -> dict:
        results: dict[str, object] = {}
        output_root = job.output_dir

        if "cavity" in params.cases:
            job_manager.raise_if_cancelled(job.job_id)
            from tensorlbm import LidDrivenCavityConfig, run_lid_driven_cavity

            nx = 64 if params.fast else 128
            re_cases = [
                (100, 8000 if params.fast else 30000),
                (400, 10000 if params.fast else 40000),
                (1000, 12000 if params.fast else 50000),
            ]
            cavity_results = []
            for re_int, n_steps in re_cases:
                job_manager.raise_if_cancelled(job.job_id)
                cfg = LidDrivenCavityConfig(
                    nx=nx,
                    re=float(re_int),
                    n_steps=n_steps,
                    output_interval=max(n_steps // 4, 1),
                    device=params.device,
                    output_root=output_root / "cavity" / f"re{re_int}",
                    run_name=f"cavity_re{re_int}",
                    overwrite=True,
                )
                run_dir = run_lid_driven_cavity(cfg)

                import json as _json
                meta = _json.loads((run_dir / "run_metadata.json").read_text())
                ghia_errors = meta.get("ghia_errors") or {}
                cavity_results.append({
                    "re": re_int,
                    "rmse_u": ghia_errors.get("rmse_u"),
                    "rmse_v": ghia_errors.get("rmse_v"),
                })
            results["cavity"] = cavity_results

        if "bfs" in params.cases:
            job_manager.raise_if_cancelled(job.job_id)
            from tensorlbm import BackwardFacingStepConfig, run_backward_facing_step

            nx, ny, step_h, x_step = (
                (240, 60, 30, 60) if params.fast else (400, 80, 40, 80)
            )
            bfs_results = []
            for re_int, n_steps in [(100, 15000 if params.fast else 40000),
                                    (200, 15000 if params.fast else 40000)]:
                job_manager.raise_if_cancelled(job.job_id)
                cfg = BackwardFacingStepConfig(
                    nx=nx,
                    ny=ny,
                    step_h=step_h,
                    x_step=x_step,
                    u_in=0.05,
                    re=float(re_int),
                    n_steps=n_steps,
                    output_interval=max(n_steps // 4, 1),
                    device=params.device,
                    output_root=output_root / "bfs" / f"re{re_int}",
                    run_name=f"bfs_re{re_int}",
                    overwrite=True,
                )
                run_dir = run_backward_facing_step(cfg)

                import json as _json
                meta = _json.loads((run_dir / "run_metadata.json").read_text())
                bfs_results.append({
                    "re": re_int,
                    "xr_star": meta.get("final_reattachment_xr_star"),
                })
            results["bfs"] = bfs_results

        if "rotating_cylinder" in params.cases:
            job_manager.raise_if_cancelled(job.job_id)
            from tensorlbm import RotatingCylinderConfig, run_rotating_cylinder

            nx, ny, radius, n_steps = (
                (240, 80, 10.0, 6000) if params.fast else (400, 120, 15.0, 12000)
            )
            rot_results = []
            for alpha in (1.0, 2.0):
                job_manager.raise_if_cancelled(job.job_id)
                cfg = RotatingCylinderConfig(
                    nx=nx,
                    ny=ny,
                    u_in=0.05,
                    re=200.0,
                    radius=radius,
                    spin_ratio=alpha,
                    n_steps=n_steps,
                    output_interval=max(n_steps // 6, 1),
                    device=params.device,
                    output_root=output_root / "rotating_cylinder" / f"alpha{alpha:g}",
                    run_name=f"rotating_re200_alpha{alpha:g}",
                    overwrite=True,
                )
                run_dir = run_rotating_cylinder(cfg)

                import json as _json
                meta = _json.loads((run_dir / "run_metadata.json").read_text())
                rot_results.append({
                    "spin_ratio": alpha,
                    "cl_mean": meta.get("cl_mean"),
                    "cd_mean": meta.get("cd_mean"),
                })
            results["rotating_cylinder"] = rot_results

        return results

    return submit_benchmark(
        name=f"Accuracy Benchmarks ({'fast' if params.fast else 'full'})",
        job_type="benchmark_accuracy",
        params=params,
        runner=_run,
        message="Accuracy benchmark submitted",
    )
