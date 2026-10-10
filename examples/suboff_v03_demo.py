"""One-click SUBOFF v0.3 closed-loop demo: checkpoint -> inference -> flow-field maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tensorlbm.ai.suboff_platform_pipeline import SuboffPlatformPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        required=True,
        help="SUBOFF snapshot root: data_dir/{p,ux,uy,uz}/*.npy",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/suboff_v03_demo",
        help="Demo output directory",
    )
    parser.add_argument(
        "--db-path",
        default="outputs/suboff_v03_demo/suboff_platform.db",
        help="SQLite ledger path",
    )
    parser.add_argument("--checkpoint", default="models/suboff_v0.3.pt", help="Checkpoint path")
    parser.add_argument(
        "--snap-idx",
        type=int,
        default=55,
        help="Snapshot index within test-set window",
    )
    parser.add_argument(
        "--test-set-offset",
        type=int,
        default=1250,
        help="Dataset test split offset",
    )
    parser.add_argument("--device", default=None, help="Inference device, e.g. cpu / cuda:0")
    parser.add_argument(
        "--slice-axis",
        choices=("x", "y", "z"),
        default="z",
        help="Flow-field visualization slice axis",
    )
    parser.add_argument(
        "--slice-index",
        type=int,
        default=50,
        help="Flow-field visualization slice index",
    )
    args = parser.parse_args()

    pipeline = SuboffPlatformPipeline(Path(args.db_path))
    try:
        result = pipeline.run_checkpoint_inference_demo(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            checkpoint_path=args.checkpoint,
            snap_idx=args.snap_idx,
            test_set_offset=args.test_set_offset,
            device=args.device,
            slice_axis=args.slice_axis,
            slice_index=args.slice_index,
        )
    finally:
        pipeline.close()

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
