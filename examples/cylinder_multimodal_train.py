"""End-to-end Stage-2 multimodal cylinder-flow dataset & forecast training CLI.

Supports two workflows:
1. ``--generate-smoke``: runs real D2Q9 cylinder-flow simulations across
   disjoint Reynolds numbers (e.g., train Re={20, 50, 80}, val Re={35},
   test Re={65}), validates all Stage-1 quality gates, optionally bridges
   snapshots into ``FieldDataCatalog`` (R2 HDF5 products), trains the
   patchified multimodal forecaster with ``TrainingJobRegistry`` tracking,
   and evaluates held-out forecasts against the persistence baseline.
2. Existing solver runs via ``--train-dirs``, ``--val-dirs``, ``--test-dirs``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tensorlbm import CylinderFlowConfig, run_cylinder_flow
from tensorlbm.ai import (
    CylinderMultimodalDataset,
    MultimodalCylinderArch,
    MultimodalTrainConfig,
    evaluate_forecaster_vs_persistence,
    export_cylinder_run_to_r2_catalog,
    rollout_trajectory_forecast,
    split_cylinder_trajectories,
    train_multimodal_cylinder_forecaster,
    trim_trajectory_spinup,
    validate_cylinder_trajectory,
)
from tensorlbm.data import FieldDataCatalog
from tensorlbm.ml.training_job import TrainingJobRegistry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the Stage-2 multimodal cylinder-flow forecaster."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/cylinder_multimodal_stage2"),
        help="Directory for generated runs, manifests, checkpoints, and reports",
    )
    parser.add_argument(
        "--generate-smoke",
        action="store_true",
        help="Generate real multi-Re cylinder-flow trajectories before training",
    )
    parser.add_argument("--nx", type=int, default=96, help="Grid width for generated runs")
    parser.add_argument("--ny", type=int, default=48, help="Grid height for generated runs")
    parser.add_argument("--u-in", dest="u_in", type=float, default=0.06, help="Inlet speed")
    parser.add_argument("--radius", type=float, default=5.0, help="Cylinder radius")
    parser.add_argument("--n-steps", dest="n_steps", type=int, default=160, help="Solver steps")
    parser.add_argument(
        "--output-interval",
        dest="output_interval",
        type=int,
        default=20,
        help="Snapshot interval for generated runs",
    )
    parser.add_argument(
        "--train-re",
        type=float,
        nargs="+",
        default=[20.0, 50.0, 80.0],
        help="Reynolds numbers for generated training trajectories",
    )
    parser.add_argument(
        "--val-re",
        type=float,
        nargs="+",
        default=[35.0],
        help="Unseen Reynolds numbers for generated validation trajectories",
    )
    parser.add_argument(
        "--test-re",
        type=float,
        nargs="+",
        default=[65.0],
        help="Unseen Reynolds numbers for generated test trajectories",
    )
    parser.add_argument(
        "--train-radii",
        type=float,
        nargs="*",
        default=None,
        help="Optional cylinder radii for variable-geometry training trajectories",
    )
    parser.add_argument(
        "--val-radii",
        type=float,
        nargs="*",
        default=None,
        help="Optional cylinder radii for validation trajectories",
    )
    parser.add_argument(
        "--test-radii",
        type=float,
        nargs="*",
        default=None,
        help="Optional unseen cylinder radii for holdout test trajectories",
    )
    parser.add_argument(
        "--min-step",
        dest="min_step",
        type=int,
        default=0,
        help="Discard initial impulsive acoustic spin-up frames strictly below min_step",
    )
    parser.add_argument(
        "--history-steps",
        dest="history_steps",
        type=int,
        default=1,
        help="Number of consecutive historical frames H fed into the forecaster",
    )
    parser.add_argument(
        "--pushforward-steps",
        dest="pushforward_steps",
        type=int,
        default=1,
        help="Number of unrolled autoregressive pushforward training steps K",
    )
    parser.add_argument(
        "--train-dirs",
        type=Path,
        nargs="*",
        default=[],
        help="Existing cylinder-flow run directories for training",
    )
    parser.add_argument(
        "--val-dirs",
        type=Path,
        nargs="*",
        default=[],
        help="Existing cylinder-flow run directories for validation",
    )
    parser.add_argument(
        "--test-dirs",
        type=Path,
        nargs="*",
        default=[],
        help="Existing cylinder-flow run directories for holdout testing",
    )
    parser.add_argument("--patch-y", type=int, default=4, help="Vertical patch size")
    parser.add_argument("--patch-x", type=int, default=4, help="Horizontal patch size")
    parser.add_argument("--d-model", dest="d_model", type=int, default=64, help="Token width")
    parser.add_argument(
        "--n-layers", dest="n_layers", type=int, default=3, help="Transformer depth"
    )
    parser.add_argument("--epochs", type=int, default=35, help="Training epochs")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-3, help="Peak AdamW learning rate")
    parser.add_argument("--device", default="cpu", help="Torch device (cpu or cuda)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--register-r2-catalog",
        action="store_true",
        help="Also export trajectories to HDF5 and register R2 FieldDataCatalog products",
    )
    return parser


def _generate_split_runs(
    re_list: list[float],
    radii_list: list[float],
    split_tag: str,
    args: argparse.Namespace,
    sim_root: Path,
) -> list[Path]:
    dirs: list[Path] = []
    multi_radius = len(radii_list) > 1 or (
        len(radii_list) == 1 and abs(radii_list[0] - float(args.radius)) > 1e-9
    )
    for re_val in re_list:
        re_label = f"{re_val:g}".replace(".", "p")
        for rad in radii_list:
            rad_label = f"{rad:g}".replace(".", "p")
            run_name = (
                f"{split_tag}_re{re_label}_r{rad_label}"
                if multi_radius
                else f"{split_tag}_re{re_label}"
            )
            cfg = CylinderFlowConfig(
                nx=args.nx,
                ny=args.ny,
                u_in=args.u_in,
                re=re_val,
                radius=rad,
                n_steps=args.n_steps,
                output_interval=args.output_interval,
                output_root=sim_root,
                run_name=run_name,
                device=args.device,
                overwrite=True,
                save_field_npz=True,
            )
            dirs.append(run_cylinder_flow(cfg))
    return dirs


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    out_dir: Path = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.generate_smoke:
        sim_root = out_dir / "simulations"
        train_radii = list(args.train_radii) if args.train_radii else [float(args.radius)]
        val_radii = list(args.val_radii) if args.val_radii else [float(args.radius)]
        test_radii = list(args.test_radii) if args.test_radii else [float(args.radius)]
        train_dirs = _generate_split_runs(list(args.train_re), train_radii, "train", args, sim_root)
        val_dirs = _generate_split_runs(list(args.val_re), val_radii, "val", args, sim_root)
        test_dirs = _generate_split_runs(list(args.test_re), test_radii, "test", args, sim_root)
    else:
        train_dirs = list(args.train_dirs)
        val_dirs = list(args.val_dirs)
        test_dirs = list(args.test_dirs)
        if not train_dirs:
            raise ValueError("Provide --train-dirs or pass --generate-smoke")

    raw_train_trajs = [validate_cylinder_trajectory(d) for d in train_dirs]
    raw_val_trajs = [validate_cylinder_trajectory(d) for d in val_dirs]
    raw_test_trajs = [validate_cylinder_trajectory(d) for d in test_dirs]

    min_step = int(args.min_step)
    train_trajs = [trim_trajectory_spinup(t, min_step=min_step) for t in raw_train_trajs]
    val_trajs = [trim_trajectory_spinup(t, min_step=min_step) for t in raw_val_trajs]
    test_trajs = [trim_trajectory_spinup(t, min_step=min_step) for t in raw_test_trajs]

    hist_steps = int(args.history_steps)
    pf_steps = int(args.pushforward_steps)
    manifest = split_cylinder_trajectories(
        train_trajs,
        val_trajs,
        test_trajs,
        min_step=min_step,
        history_steps=hist_steps,
        rollout_steps=pf_steps,
    )
    manifest_path = manifest.save(out_dir / "dataset_manifest.json")

    db_path = out_dir / "governance.sqlite"
    r2_product_count = 0
    if args.register_r2_catalog:
        catalog = FieldDataCatalog.open(db_path)
        try:
            for traj in [*train_trajs, *val_trajs, *test_trajs]:
                pids = export_cylinder_run_to_r2_catalog(
                    traj,
                    catalog,
                    h5_path=out_dir / "r2_hdf5" / f"{traj.run_id}.h5",
                )
                r2_product_count += len(pids)
        finally:
            catalog.close()

    train_ds = CylinderMultimodalDataset(
        train_trajs,
        normalization=manifest.normalization,
        history_steps=hist_steps,
        rollout_steps=pf_steps,
        load_visual=True,
    )
    val_ds = (
        CylinderMultimodalDataset(
            val_trajs,
            normalization=manifest.normalization,
            history_steps=hist_steps,
            rollout_steps=1,
            load_visual=True,
        )
        if val_trajs
        else None
    )

    arch = MultimodalCylinderArch(
        patch_size=(args.patch_y, args.patch_x),
        history_steps=hist_steps,
        d_model=args.d_model,
        n_layers=args.n_layers,
    )
    train_cfg = MultimodalTrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        pushforward_steps=pf_steps,
        seed=args.seed,
        device=args.device,
    )

    weights_path = out_dir / "checkpoints" / "multimodal_forecaster.pt"
    registry = TrainingJobRegistry.open(db_path)
    try:
        train_res = train_multimodal_cylinder_forecaster(
            train_ds,
            val_ds,
            arch=arch,
            config=train_cfg,
            manifest=manifest,
            job_registry=registry,
            save_weights_path=weights_path,
        )

        eval_trajs = test_trajs or val_trajs or train_trajs
        eval_ds = CylinderMultimodalDataset(
            eval_trajs,
            normalization=manifest.normalization,
            history_steps=hist_steps,
            rollout_steps=1,
            load_visual=True,
        )
        multimodal_eval = evaluate_forecaster_vs_persistence(
            train_res.model,
            eval_ds,
            batch_size=args.batch_size,
            device=args.device,
            use_visual=True,
            use_force_input=True,
        )
        solver_mode_eval = evaluate_forecaster_vs_persistence(
            train_res.model,
            eval_ds,
            batch_size=args.batch_size,
            device=args.device,
            use_visual=False,
            use_force_input=True,
        )
        rollout_start = min(max(hist_steps - 1, 0), max(len(eval_trajs[0].frames) - 2, 0))
        rollout = rollout_trajectory_forecast(
            train_res.model,
            eval_trajs[0],
            start_index=rollout_start,
            normalization=manifest.normalization,
            device=args.device,
        )

        if train_res.job_id is not None:
            registry.record_metrics(
                train_res.job_id,
                {
                    "holdout_rel_l2_velocity": multimodal_eval.model_rel_l2_velocity,
                    "persistence_rel_l2_velocity": multimodal_eval.persistence_rel_l2_velocity,
                    "velocity_skill_vs_persistence": multimodal_eval.velocity_skill_vs_persistence,
                    "solver_mode_velocity_skill": solver_mode_eval.velocity_skill_vs_persistence,
                },
            )
    finally:
        registry.close()

    report: dict[str, Any] = {
        "manifest_path": str(manifest_path),
        "dataset_fingerprint": manifest.fingerprint,
        "r2_registered_products": r2_product_count,
        "training": {
            "job_id": train_res.job_id,
            "registered_model_id": train_res.registered_model_id,
            "best_epoch": train_res.best_epoch,
            "best_val_loss": train_res.best_val_loss,
            "final_train_loss": train_res.train_losses[-1],
            "train_wall_seconds": train_res.train_wall_seconds,
        },
        "holdout_multimodal_evaluation": multimodal_eval.to_dict(),
        "holdout_zero_png_solver_evaluation": solver_mode_eval.to_dict(),
        "autoregressive_rollout": [asdict(step) for step in rollout],
    }
    report_path = out_dir / "evaluation_report.json"
    report_path.write_text(f"{json.dumps(report, indent=2, sort_keys=True)}\n", encoding="utf-8")
    return report


def main() -> None:
    args = build_parser().parse_args()
    report = run_pipeline(args)
    mm = report["holdout_multimodal_evaluation"]
    zr = report["holdout_zero_png_solver_evaluation"]
    stage_tag = (
        "Stage-3"
        if (
            args.history_steps > 1
            or args.pushforward_steps > 1
            or args.min_step > 0
            or bool(args.train_radii)
        )
        else "Stage-2"
    )
    print(
        f"[{stage_tag} Multimodal Cylinder] "
        f"Holdout vel rel-L2={mm['model_rel_l2_velocity']:.4f} "
        f"vs persistence={mm['persistence_rel_l2_velocity']:.4f} "
        f"(skill={mm['velocity_skill_vs_persistence']:+.2%}); "
        f"zero-PNG solver skill={zr['velocity_skill_vs_persistence']:+.2%}"
    )


if __name__ == "__main__":
    main()
