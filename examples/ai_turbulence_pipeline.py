"""End-to-end HPC + AI demonstration.

Runs the full pipeline:

    modelling / solving → dataset extraction →
    SQLite persistence → AI turbulence-model training →
    AI-enhanced LBM (LES) validation run

and prints a concise summary suitable for CI logs.  See
``docs/ai_turbulence.md`` for the design overview.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tensorlbm import TrainConfig, run_ai_les_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", default="outputs/ai_les_demo")
    parser.add_argument("--nx", type=int, default=48)
    parser.add_argument("--ny", type=int, default=48)
    parser.add_argument("--tau", type=float, default=0.8)
    parser.add_argument("--c-s", type=float, default=0.1)
    parser.add_argument("--data-steps", type=int, default=60)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--val-steps", type=int, default=40)
    parser.add_argument("--data-source", choices=("les", "dns"), default="les")
    parser.add_argument("--dns-scale", type=int, default=2)
    parser.add_argument("--dns-warmup-steps", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    result = run_ai_les_pipeline(
        work_dir=Path(args.work_dir),
        nx=args.nx, ny=args.ny,
        tau=args.tau, c_s=args.c_s,
        data_steps=args.data_steps, sample_every=args.sample_every,
        val_steps=args.val_steps,
        train_config=TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device=args.device,
        ),
        seed=args.seed,
        device=args.device,
        data_source=args.data_source,
        dns_scale=args.dns_scale,
        dns_warmup_steps=args.dns_warmup_steps,
    )
    summary = {
        "work_dir": str(result.work_dir),
        "db_path": str(result.db_path),
        "dataset_path": str(result.dataset_path),
        "model_path": str(result.model_path),
        "ids": {"run": result.run_id, "dataset": result.dataset_id,
                "model": result.model_id},
        "n_samples": result.n_samples,
        "data_source": result.data_source,
        "n_snapshots": result.n_snapshots,
        "training": {
            "final_train_mse": result.training["final_train_mse"],
            "final_val_mse": result.training["final_val_mse"],
            "final_val_r2": result.training["final_val_r2"],
        },
        "validation": result.validation,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
