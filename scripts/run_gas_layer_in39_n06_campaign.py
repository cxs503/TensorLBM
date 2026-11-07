#!/usr/bin/env python3
"""Run the IN39-N06 gas-layer campaign and print the gate summary.

    PYTHONPATH=src python scripts/run_gas_layer_in39_n06_campaign.py

The lattice work is single-threaded on purpose: the runs are small (40 x 64) and
thread oversubscription on a shared host is far slower than one thread.
"""

from __future__ import annotations

import json
import time

import torch

torch.set_num_threads(1)

from tensorlbm.gas_layer.campaign import run_campaign  # noqa: E402  (thread pinning first)


def main() -> int:
    t0 = time.time()
    report = run_campaign("artifacts/gas_layer_in39_n06")
    print(f"elapsed {time.time() - t0:.1f} s")
    print(json.dumps(report["gates"], indent=1))
    print(f"passed {report['gates_passed']} / {report['gates_total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())