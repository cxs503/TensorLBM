"""Run the P88W1-N01 forced-sloshing campaign and regenerate report + figures.

    python scripts/run_p88w1_campaign.py --workers 10

Steps: campaign (cases in ``tensorlbm.sloshing_forced.campaign.case_table``) ->
figures (``scripts/make_p88w1_figures.py``) -> report + JSON summary
(``scripts/gen_p88w1_report.py``).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--artifacts", type=Path, default=Path("artifacts/p88w1"))
    ap.add_argument("--skip-campaign", action="store_true")
    args = ap.parse_args(argv)

    env = {"PYTHONPATH": str(ROOT / "src")}
    import os

    run_env = dict(os.environ)
    run_env.update(env)

    if not args.skip_campaign:
        subprocess.run(
            [sys.executable, "-m", "tensorlbm.sloshing_forced.campaign",
             "--out", str(args.artifacts), "--workers", str(args.workers)],
            check=True, cwd=ROOT, env=run_env,
        )
    subprocess.run(
        [sys.executable, "scripts/make_p88w1_figures.py", "--artifacts", str(args.artifacts)],
        check=True, cwd=ROOT, env=run_env,
    )
    subprocess.run(
        [sys.executable, "scripts/gen_p88w1_report.py", "--artifacts", str(args.artifacts)],
        check=True, cwd=ROOT, env=run_env,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
