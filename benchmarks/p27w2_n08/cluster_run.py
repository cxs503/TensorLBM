"""Cluster execution script for P27W2-N08.

Submits the 4-experiment campaign to the loong64 cluster via run_cluster.
Each experiment runs independently and can be sharded across nodes.

Usage (from worktree root):
    python benchmarks/p27w2_n08/cluster_run.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def run_on_cluster():
    """Submit experiments to the cluster."""
    experiments = [
        {"domain": 2.0, "cs": 0.0, "label": "domain_2pi_cs0.0"},
        {"domain": 2.0, "cs": 0.1, "label": "domain_2pi_cs0.1"},
        {"domain": 4.0, "cs": 0.0, "label": "domain_4pi_cs0.0"},
        {"domain": 4.0, "cs": 0.1, "label": "domain_4pi_cs0.1"},
    ]

    # For now, run locally with reduced grid for validation
    # Full campaign should be submitted to cluster
    print("P27W2-N08 Cluster Execution Plan")
    print("=" * 50)
    print()
    print("Experiments to run:")
    for exp in experiments:
        print(f"  {exp['label']}: Lx={exp['domain']}πδ, C_s={exp['cs']}")
    print()
    print("Recommended cluster command:")
    print("  run_cluster action=shard repo=TensorLBM "
          "cmd_tpl='PYTHONPATH=src python benchmarks/p27w2_n08/run_p27w2_n08.py "
          "run --domain {domain} --cs {cs} --steps 80000 --seed 42' "
          "shards=4 timeout=7200")
    print()
    print("For local validation (reduced grid):")
    for exp in experiments:
        cmd = (
            f"PYTHONPATH=src python benchmarks/p27w2_n08/run_p27w2_n08.py run "
            f"--domain {exp['domain']} --cs {exp['cs']} --steps 20000 --seed 42"
        )
        print(f"  {cmd}")


if __name__ == "__main__":
    run_on_cluster()
