"""AI4S flagship demo: one command that closes the whole platform loop.

Runs the end-to-end pipeline of :mod:`tensorlbm.apps.ai4s_flagship`::

    solver (public API) -> HDF5 dataset -> catalog product/dataset
    -> TrainingJobRegistry job -> FNO2d super-resolution training (GPU)
    -> model-asset registry (task/metrics/dataset product_id/git sha)
    -> serving registration + live inference on held-out samples
    -> lineage graph read back as one upstream chain

Optionally (default on) it also exercises the application-SDK skeleton
(:meth:`tensorlbm.apps.base.AI4SApplication.run` via ``NeuralOperatorFNO``)
with a tiny CPU configuration, showing the same chain wired automatically.

Usage (single GPU, fits in minutes)::

    CUDA_VISIBLE_DEVICES=7 python examples/ai4s_flagship_demo.py \
        --workdir /nfs/wangxi/am_flagship_20260820/run

Outputs under ``--workdir``: ``velocity_snapshots.h5``, ``flagship_fno2d.pt``
(+ json), ``platform.db`` (catalog + jobs + serving), ``model_store/``
(asset registry + checkpoint copies + meta.json sidecars), ``report.json``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tensorlbm.apps.ai4s_flagship import (
    DEFAULT_PILOT_DIR,
    FlagshipConfig,
    print_report,
    run_flagship_demo,
)


def run_sdk_skeleton(db_path: Path):
    """Exercise the application SDK's automatic ``run()`` wiring (tiny, CPU)."""
    from tensorlbm.apps.neural_operator_fno import NeuralOperatorFNO

    app = NeuralOperatorFNO()
    return app.run(
        db_path,
        produce_cfg=dict(
            nx=32, ny=32, n_steps=8, sample_every=4,
            device="cpu", downsample_factor=2,
        ),
        train_cfg=dict(
            arch=dict(
                in_channels=2, out_channels=2, width=8, n_layers=2,
                modes_x=6, modes_y=6, mlp_hidden=16,
            ),
            epochs=3, batch_size=4, learning_rate=1e-3, device="cpu",
            out_path=str(db_path.parent / "sdk_skeleton_fno2d.pt"),
        ),
        name_prefix="sdk_skeleton",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="TensorLBM AI4S flagship end-to-end demo",
    )
    parser.add_argument(
        "--workdir", default="./flagship_run",
        help="workspace for data/registry/report artifacts",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="training device (falls back to cpu when CUDA is absent)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="training epochs (defaults to FlagshipConfig.epochs)",
    )
    parser.add_argument(
        "--pilot-dir", default=DEFAULT_PILOT_DIR,
        help="pilot dataset directory; falls back to the in-process solver "
             "when absent (pass an empty string to disable)",
    )
    parser.add_argument(
        "--skip-sdk-demo", action="store_true",
        help="skip the tiny application-SDK run() skeleton demo",
    )
    args = parser.parse_args(argv)

    cfg = FlagshipConfig(
        workdir=args.workdir,
        device=args.device,
        pilot_dir=(args.pilot_dir or None),
    )
    if args.epochs is not None:
        cfg.epochs = int(args.epochs)
    report = run_flagship_demo(cfg)
    print_report(report)

    if not args.skip_sdk_demo:
        print("\napplication-SDK skeleton (AI4SApplication.run, tiny CPU cfg):")
        sdk = run_sdk_skeleton(Path(args.workdir) / "platform.db")
        print(f"  app            : {sdk.name} (family {sdk.family})")
        print(f"  data asset     : {sdk.data_asset_id}")
        print(f"  dataset asset  : {sdk.dataset_asset_id}")
        print(f"  training job   : {sdk.job_id}")
        print(f"  serving model  : {sdk.model_id}  metrics {dict(sdk.metrics)}")
        print(f"  lineage        : {' <- '.join(sdk.lineage_upstream) or '(none)'}")

    print(f"\nfull report written to {Path(args.workdir) / 'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
