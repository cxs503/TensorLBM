#!/usr/bin/env python3
"""Turbulent viscoplastic channel campaign runner.

Submits parameter sweep to cluster:
- Yield number Y ∈ {0, 1, 5, 10, 20, 50}
- Flow index n ∈ {0.8, 1.0}
- Grid: D3Q19 (nx=128, ny=48, nz=64)

Each case runs ~30000 steps with statistics collection.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))


def generate_cluster_commands():
    """Generate cluster commands for parameter sweep."""
    # Parameter matrix
    Y_values = [0.0, 1.0, 5.0, 10.0, 20.0]
    n_values = [0.8, 1.0]

    # Fixed parameters
    re_tau = 180.0
    u_tau = 0.006
    rho = 1.0
    K = 0.02

    commands = []
    for Y in Y_values:
        for n in n_values:
            # Compute tau_y from Y
            tau_y = Y * rho * u_tau ** 2

            name = f"Y{Y:.0f}_n{n:.1f}"
            cmd = (
                f"python benchmarks/in45_hb_turb/run_single_case.py "
                f"--name {name} "
                f"--tau-y {tau_y:.6e} "
                f"--K {K} "
                f"--n {n} "
                f"--re-tau {re_tau} "
                f"--u-tau {u_tau} "
                f"--steps 30000 "
                f"--warmup 5000 "
                f"--out-dir benchmarks/in45_hb_turb/artifacts/turb_cases"
            )
            commands.append({"name": name, "Y": Y, "n": n, "tau_y": tau_y, "cmd": cmd})

    return commands


def main():
    """Generate cluster submission script."""
    commands = generate_cluster_commands()

    out_dir = REPO / "benchmarks" / "in45_hb_turb" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save command list
    cmd_path = out_dir / "cluster_commands.json"
    cmd_path.write_text(json.dumps(commands, indent=2))
    print(f"Saved {len(commands)} commands to {cmd_path}")

    # Print summary
    print("\nParameter sweep:")
    print(f"  Y values: {sorted(set(c['Y'] for c in commands))}")
    print(f"  n values: {sorted(set(c['n'] for c in commands))}")
    print(f"  Total cases: {len(commands)}")

    # Generate shell script for cluster
    sh_path = out_dir / "run_cluster.sh"
    with open(sh_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Viscoplastic turbulent channel campaign\n")
        f.write("# Submit with: run_cluster submit --cmd 'bash benchmarks/in45_hb_turb/artifacts/run_cluster.sh'\n\n")
        for cmd in commands:
            f.write(f"echo 'Running {cmd[\"name\"]}...'\n")
            f.write(f"{cmd['cmd']}\n\n")
    sh_path.chmod(0o755)
    print(f"Shell script: {sh_path}")

    return commands


if __name__ == "__main__":
    main()
