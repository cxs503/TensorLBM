#!/usr/bin/env python3
"""Fresh 2D turbulent-channel grid ladder (Re_tau = 180) on the CPU node.

Purpose: give the GCI machinery of :mod:`tensorlbm.validation` a *newly
produced* three-grid sequence (the 3D GPU runs in
``benchmarks/pending/channel_turb_3d`` are reused as the 3D tier, see their
provenance hashes in the ladder artifact).

Driven entirely by library entry points:

    from tensorlbm.turbulent_channel import TurbulentChannelConfig, run_turbulent_channel

The 2D reduction cannot sustain the full 3D turbulent cascade, so its absolute
agreement with channel DNS is expected to be worse than the 3D runs; what the
ladder uses it for is (i) a fresh, reproducible convergence sequence and
(ii) an explicit demonstration of the 2D-vs-3D modelling gap.

Usage
-----
    python benchmarks/in12_s6_wall_ladder/run_2d_channel_ladder.py --steps 10000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tensorlbm.turbulent_channel import (  # noqa: E402
    TurbulentChannelConfig,
    run_turbulent_channel,
)

LEVELS = {
    "C1": {"nx": 128, "ny": 65},
    "C2": {"nx": 192, "ny": 97},
    "C3": {"nx": 256, "ny": 129},
}


def run_level(name: str, *, steps: int, averaging_start: int, out_root: Path) -> dict:
    level = LEVELS[name]
    config = TurbulentChannelConfig(
        nx=level["nx"],
        ny=level["ny"],
        re_tau=180.0,
        u_tau=0.0025,
        n_steps=steps,
        averaging_start=averaging_start,
        smagorinsky_cs=0.1,
        output_interval=max(steps // 8, 1),
        output_root=out_root,
        run_name=name,
        overwrite=True,
        seed=20261107,
    )
    started = time.time()
    run_dir = run_turbulent_channel(config)
    elapsed = time.time() - started
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    record = {
        "level": name,
        "grid": [level["nx"], level["ny"]],
        "steps": steps,
        "averaging_start": averaging_start,
        "wall_seconds": elapsed,
        "log_law_rms_error": metadata.get("log_law_rms_error"),
        "run_dir": str(run_dir.relative_to(REPO_ROOT)),
        "velocity_profile_csv": str(
            (run_dir / "velocity_profile.csv").relative_to(REPO_ROOT)
        ),
        "nu": metadata["derived"]["nu"],
        "u_tau": config.u_tau,
    }
    (out_root / f"{name}.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"[{name}] {level['nx']}x{level['ny']} steps={steps} "
        f"{elapsed:.1f}s log-law RMS={record['log_law_rms_error']}",
        flush=True,
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--averaging-start", type=int, default=None)
    parser.add_argument(
        "--out-root",
        default=str(REPO_ROOT / "artifacts" / "in12_s6" / "runs2d"),
    )
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(20261107)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    averaging_start = args.averaging_start or args.steps // 2
    records = []
    for name in LEVELS:
        records.append(
            run_level(
                name,
                steps=args.steps,
                averaging_start=averaging_start,
                out_root=out_root,
            )
        )
    (out_root / "ladder_runs.json").write_text(
        json.dumps(
            {"steps": args.steps, "averaging_start": averaging_start, "levels": records},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())