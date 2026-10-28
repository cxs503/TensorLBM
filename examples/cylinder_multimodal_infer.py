"""Pure condition-to-trajectory AI inference CLI for 2-D cylinder flow (Scheme A).

Takes ONLY scalar simulation conditions ``(nx, ny, u_in, re, radius, n_steps,
output_interval)`` plus a trained ``MultimodalCylinderForecaster`` checkpoint,
analytically constructs the cylinder obstacle mask, signed distance field, and
``step=0`` uniform initial state, and autoregressively predicts the entire flow
trajectory ``(ux, uy, rho, speed, vorticity)`` and aerodynamic force history
``(Cd, Cl)`` without invoking the LBM solver.

Outputs solver-compatible ``fields_step_XXXXXX.npz``, ``flow_step_XXXXXX.png``
(unless ``--no-png`` is passed), ``forces.csv``, and ``run_metadata.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from tensorlbm import CylinderFlowConfig, run_cylinder_flow
from tensorlbm.ai import (
    CylinderConditionSpec,
    forecast_from_conditions,
    validate_cylinder_trajectory,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Directly forecast 2-D cylinder-flow fields and (Cd, Cl) from scalar "
            "conditions using a trained MultimodalCylinderForecaster."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to trained multimodal_forecaster.pt checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/cylinder_ai_forecast"),
        help="Directory to write predicted NPZ/PNG snapshots, forces.csv, and metadata",
    )
    parser.add_argument("--nx", type=int, default=96, help="Grid width")
    parser.add_argument("--ny", type=int, default=48, help="Grid height")
    parser.add_argument("--u-in", dest="u_in", type=float, default=0.06, help="Inlet velocity")
    parser.add_argument("--re", type=float, default=70.0, help="Target Reynolds number")
    parser.add_argument("--radius", type=float, default=5.0, help="Cylinder radius")
    parser.add_argument(
        "--n-steps",
        dest="n_steps",
        type=int,
        default=160,
        help="Target simulation horizon in LBM steps",
    )
    parser.add_argument(
        "--output-interval",
        dest="output_interval",
        type=int,
        default=20,
        help="Step interval per autoregressive hop (should match training interval)",
    )
    parser.add_argument("--device", default="cpu", help="Torch device (cpu or cuda)")
    parser.add_argument(
        "--no-png",
        dest="no_png",
        action="store_true",
        help="Skip Matplotlib PNG rendering for fastest pure-numerical NPZ + CSV output",
    )
    parser.add_argument(
        "--compare-lbm",
        dest="compare_lbm",
        action="store_true",
        help="Optionally run D2Q9 LBM under identical conditions to report accuracy & speedup",
    )
    return parser


def _rel_l2_fluid(pred: np.ndarray, target: np.ndarray, fluid: np.ndarray) -> float:
    diff = np.sqrt(np.sum(((pred - target) ** 2) * fluid))
    ref = max(float(np.sqrt(np.sum((target**2) * fluid))), 1e-12)
    return float(diff / ref)


def run_condition_inference(args: argparse.Namespace) -> dict[str, Any]:
    spec = CylinderConditionSpec(
        nx=args.nx,
        ny=args.ny,
        u_in=args.u_in,
        re=args.re,
        radius=args.radius,
        n_steps=args.n_steps,
        output_interval=args.output_interval,
    )
    result = forecast_from_conditions(
        args.checkpoint,
        spec,
        device=args.device,
        include_initial_step=True,
    )
    out_dir = result.save_artifacts(
        args.output_dir.resolve(),
        save_field_npz=True,
        save_png=not args.no_png,
    )

    last_step = result.steps[-1]
    summary: dict[str, Any] = {
        "output_dir": str(out_dir),
        "conditions": {
            "nx": spec.nx,
            "ny": spec.ny,
            "u_in": spec.u_in,
            "re": spec.re,
            "radius": spec.radius,
            "nu": spec.nu,
            "tau": spec.tau,
            "n_steps": spec.n_steps,
            "output_interval": spec.output_interval,
        },
        "predicted_steps": [s.step for s in result.steps],
        "ai_wall_seconds": result.wall_seconds,
        "ai_ms_per_step": result.ms_per_step,
        "final_step_forces": {
            "step": last_step.step,
            "cd": last_step.cd,
            "cl": last_step.cl,
        },
    }

    if args.compare_lbm:
        import time

        t0 = time.perf_counter()
        lbm_dir = run_cylinder_flow(
            CylinderFlowConfig(
                nx=spec.nx,
                ny=spec.ny,
                u_in=spec.u_in,
                re=spec.re,
                radius=spec.radius,
                n_steps=spec.n_steps,
                output_interval=spec.output_interval,
                output_root=out_dir / "_lbm_reference",
                run_name="lbm_ref",
                device=args.device,
                overwrite=True,
                save_field_npz=True,
                save_initial_snapshot=True,
            )
        )
        lbm_wall = time.perf_counter() - t0
        lbm_traj = validate_cylinder_trajectory(lbm_dir, require_visual_png=False)
        fluid = (~result.solid_mask).astype(np.float32)
        lbm_by_step = {f.step: f for f in lbm_traj.frames}
        step_errors: list[dict[str, float | int]] = []
        for s in result.steps:
            if s.step == 0 or s.step not in lbm_by_step:
                continue
            gt = lbm_by_step[s.step]
            pred_vel = np.stack([s.ux, s.uy], axis=0)
            true_vel = np.stack([gt.ux, gt.uy], axis=0)
            init_vel = np.stack([result.steps[0].ux, result.steps[0].uy], axis=0)
            step_errors.append(
                {
                    "step": s.step,
                    "ai_rel_l2_velocity": _rel_l2_fluid(pred_vel, true_vel, fluid[None]),
                    "uniform_init_rel_l2_velocity": _rel_l2_fluid(init_vel, true_vel, fluid[None]),
                    "pred_cd": s.cd,
                    "true_cd": gt.cd,
                    "pred_cl": s.cl,
                    "true_cl": gt.cl,
                }
            )
        summary["lbm_comparison"] = {
            "lbm_wall_seconds": lbm_wall,
            "speedup_vs_lbm_run": lbm_wall / max(result.wall_seconds, 1e-9),
            "step_metrics": step_errors,
        }

    (out_dir / "inference_summary.json").write_text(
        f"{json.dumps(summary, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = run_condition_inference(args)
    fin = summary["final_step_forces"]
    msg = (
        f"[Scheme-A Pure-AI Cylinder Forecast] "
        f"Re={summary['conditions']['re']:g}, R={summary['conditions']['radius']:g}, "
        f"grid={summary['conditions']['nx']}x{summary['conditions']['ny']}, "
        f"steps={summary['predicted_steps'][-1]} "
        f"in {summary['ai_wall_seconds'] * 1000.0:.2f} ms "
        f"({summary['ai_ms_per_step']:.2f} ms/hop) -> "
        f"final Cd={fin['cd']:.4f}, Cl={fin['cl']:.4f} | saved to {summary['output_dir']}"
    )
    if "lbm_comparison" in summary:
        cmp_info = summary["lbm_comparison"]
        last_err = cmp_info["step_metrics"][-1]
        msg += (
            f" | speedup={cmp_info['speedup_vs_lbm_run']:.1f}x vs LBM, "
            f"final vel rel-L2={last_err['ai_rel_l2_velocity']:.4f} "
            f"(vs uniform-init {last_err['uniform_init_rel_l2_velocity']:.4f})"
        )
    print(msg)


if __name__ == "__main__":
    main()
