#!/usr/bin/env python3
"""Campaign node ``p387w2-n04``: IBM sphere surface-force three-layer verification.

Each invocation runs **one** case and writes one JSON record, so the whole
matrix can be sharded over the cluster:

    python scripts/p387w2_n04_sphere_force/run_validation.py drag   --re 100 --diameter 12 --out cases/
    python scripts/p387w2_n04_sphere_force/run_validation.py torque --re-omega 1 --diameter 12 --out cases/

Cases
-----
``drag``   flow past a sphere (body-force drive, periodic box): ``C_D``
``torque`` steadily rotating sphere (quiescent box): ``C_M``
``nc``     negative controls (no solver): null space / Galilean / legacy
``assemble`` merge the case records into the acceptance report JSON

All numbers in lattice units; ``dt = dx = rho_0 = 1``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tensorlbm.ibm_sphere_surface_force import (  # noqa: E402
    SphereDragConfig,
    SphereRotationConfig,
    adjudicate_empirical,
    build_sphere_surface,
    legacy_unweighted_marker_force,
    pressure_only_torque,
    run_sphere_drag,
    run_sphere_rotation,
    surface_traction,
)

CASES = Path(__file__).resolve().parents[2] / "artifacts" / "p387w2_n04" / "cases"


def _write(record: dict, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{record['name']}.json"
    path.write_text(json.dumps(record, indent=1, default=float))
    print(f"wrote {path}", flush=True)
    return path


def case_drag(args: argparse.Namespace) -> dict:
    cfg = SphereDragConfig(
        diameter=args.diameter,
        reynolds=args.re,
        lateral_diameters=args.lateral,
        streamwise_diameters=args.streamwise,
        target_speed=args.speed,
        n_theta=args.n_theta,
        n_phi=args.n_phi,
        collision=args.collision,
        guo=not args.no_guo,
        dtype=args.dtype,
        drive_gain=args.drive_gain,
        snapshot_path=args.snapshot,
        max_steps=args.max_steps,
        min_steps=args.min_steps,
        report_interval=args.report_interval,
        steady_tol=args.steady_tol,
    )
    t0 = time.time()
    res = run_sphere_drag(cfg, verbose=True)
    res["name"] = args.name or f"drag_re{args.re:g}_d{args.diameter:g}_{args.lateral:g}D"
    res["wall_seconds"] = time.time() - t0
    res["layer"] = "drag"
    res["torch_num_threads"] = torch.get_num_threads()
    return res


def case_torque(args: argparse.Namespace) -> dict:
    cfg = SphereRotationConfig(
        diameter=args.diameter,
        re_omega=args.re_omega,
        box_diameters=args.box,
        tip_speed=args.speed,
        n_theta=args.n_theta,
        n_phi=args.n_phi,
        collision=args.collision,
        guo=not args.no_guo,
        dtype=args.dtype,
        max_steps=args.max_steps,
        min_steps=args.min_steps,
        report_interval=args.report_interval,
        steady_tol=args.steady_tol,
    )
    t0 = time.time()
    res = run_sphere_rotation(cfg, verbose=True)
    res["name"] = args.name or f"torque_reo{args.re_omega:g}_d{args.diameter:g}_{args.box:g}D"
    res["wall_seconds"] = time.time() - t0
    res["layer"] = "torque"
    res["torch_num_threads"] = torch.get_num_threads()
    return res


def case_nc(_: argparse.Namespace) -> dict:
    """Negative controls that need no solver (exact, round-off level)."""
    n = 24
    radius = 4.0
    surface = build_sphere_surface((n, n, n), (12.0, 12.0, 12.0), radius, n_theta=16, n_phi=32)
    rho = torch.ones((n, n, n), dtype=torch.float64)
    u = torch.zeros((3, n, n, n), dtype=torch.float64)
    u[0] = 0.05
    # NC1: the boundary moves with the fluid -> rigid translation, a null vector
    target = torch.zeros((surface.n_markers, 3), dtype=torch.float64)
    target[:, 0] = 0.05
    quad = surface_traction(surface, u, target, rho)
    # NC2: add a uniform free stream to both the field and the boundary
    shift = torch.tensor([0.021, -0.013, 0.007], dtype=torch.float64)
    shifted = surface_traction(surface, u + shift.view(3, 1, 1, 1), target + shift, rho)
    # Legacy estimators on a *constant traction* (the case they cannot represent)
    tau = 0.02
    target_tau = target.clone()
    target_tau[:, 0] += tau
    quad_tau = surface_traction(surface, u, target_tau, rho, n_correctors=0, n_normal_passes=0)
    legacy_f, legacy_torque = legacy_unweighted_marker_force(surface, u, target_tau, rho)
    pressure_torque = pressure_only_torque(surface, rho, u)
    scale = 0.05 * surface.area_total
    exact_tau_force = tau * surface.area_total
    return {
        "name": "negative_controls",
        "layer": "controls",
        "case": "controls",
        "n_markers": surface.n_markers,
        "area_total": surface.area_total,
        "nc1_uniform_translation_force_rel": float(quad.force.norm()) / scale,
        "nc1_uniform_translation_torque_rel": float(quad.torque.norm()) / (scale * radius),
        "nc2_galilean_force_rel_change": float((shifted.force - quad.force).norm()) / scale,
        "nc2_galilean_torque_rel_change": float((shifted.torque - quad.torque).norm())
        / (scale * radius),
        "quadrature_constant_traction_rel_error": abs(
            float(quad_tau.force.norm()) - exact_tau_force
        )
        / exact_tau_force,
        "legacy_g15_uniform_force_rel": float(legacy_f.norm()) / exact_tau_force,
        "legacy_marker_density_ratio": surface.n_markers / surface.area_total,
        "legacy_uniform_torque_rel": float(legacy_torque.norm()) / (exact_tau_force * radius),
        "g12_pressure_only_torque_rel": float(pressure_torque.norm()) / (scale * radius),
        "band_re100": adjudicate_empirical(100.0),
    }


def case_assemble(args: argparse.Namespace) -> dict:
    files = sorted(args.out.glob("*.json"))
    records = [json.loads(p.read_text()) for p in files]
    drag = {r["name"]: r for r in records if r.get("layer") == "drag"}
    torque = {r["name"]: r for r in records if r.get("layer") == "torque"}
    controls = [r for r in records if r.get("layer") == "controls"]
    summary = {
        "node": "p387w2-n04",
        "repo": "TensorLBM",
        "module": "src/tensorlbm/ibm_sphere_surface_force.py",
        "drag_cases": {
            k: {
                "reynolds_target": v["config"]["reynolds_target"],
                "reynolds_measured": v["reynolds_measured"],
                "cd_mean": v["cd_mean"],
                "cd_std": v["cd_std"],
                "cd_drive_mean": v["cd_drive_mean"],
                "err_sn_pct": v["cd_vs_schiller_naumann_pct"],
                "err_cg_pct": v["cd_vs_clift_gauvin_pct"],
                "band": v["anchor"],
                "diameter": v["config"]["diameter"],
                "lateral_diameters": v["config"]["lateral_diameters"],
                "steps": v["config"]["steps"],
                "steady": v["steady"],
                "max_mass_residual": v["max_mass_residual"],
                "max_closure_residual": v["max_closure_residual"],
            }
            for k, v in drag.items()
        },
        "torque_cases": {
            k: {
                "re_omega": v["config"]["re_omega_target"],
                "cm_mean": v["cm_mean"],
                "cm_std": v["cm_std"],
                "cm_stokes": v["cm_stokes_reference"],
                "err_stokes_pct": v["cm_vs_stokes_pct"],
                "cm_pressure_only": v["cm_pressure_only_mean"],
                "diameter": v["config"]["diameter"],
                "box_diameters": v["config"]["box_diameters"],
                "steps": v["config"]["steps"],
                "max_mass_residual": v["max_mass_residual"],
            }
            for k, v in torque.items()
        },
        "controls": controls[0] if controls else {},
    }
    out = args.out / "assembly.json"
    out.write_text(json.dumps(summary, indent=1, default=float))
    print(f"wrote {out}", flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=["drag", "torque", "nc", "assemble"])
    parser.add_argument("--out", type=Path, default=CASES)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--diameter", type=float, default=12.0)
    parser.add_argument("--re", type=float, default=100.0)
    parser.add_argument("--re-omega", dest="re_omega", type=float, default=1.0)
    parser.add_argument("--lateral", type=float, default=8.0)
    parser.add_argument("--streamwise", type=float, default=8.0)
    parser.add_argument("--box", type=float, default=8.0)
    parser.add_argument("--speed", type=float, default=0.04)
    parser.add_argument("--n-theta", dest="n_theta", type=int, default=24)
    parser.add_argument("--n-phi", dest="n_phi", type=int, default=48)
    parser.add_argument("--collision", choices=["bgk", "trt"], default="bgk")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--snapshot", type=str, default=None, help="save a mid-plane field snapshot (.pt)")
    parser.add_argument("--drive-gain", dest="drive_gain", type=float, default=0.0,
                        help="relative gain of the drive controller; 0 = fixed drive")
    parser.add_argument("--no-guo", action="store_true")
    parser.add_argument("--max-steps", dest="max_steps", type=int, default=40000)
    parser.add_argument("--min-steps", dest="min_steps", type=int, default=6000)
    parser.add_argument("--report-interval", dest="report_interval", type=int, default=200)
    parser.add_argument("--steady-tol", dest="steady_tol", type=float, default=2e-3)
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args(argv)

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)

    handler = {
        "drag": case_drag,
        "torque": case_torque,
        "nc": case_nc,
        "assemble": case_assemble,
    }[args.case]
    record = handler(args)
    if args.case != "assemble":
        _write(record, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())