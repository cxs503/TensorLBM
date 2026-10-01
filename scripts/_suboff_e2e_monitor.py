#!/usr/bin/env python3
"""Progress-monitoring launcher for the SUBOFF Re=1000 end-to-end run.

Keeps benchmarks/pending/suboff_re1000/run.py UNMODIFIED: it monkeypatches
GeneralSimEngine.run to spawn a daemon thread that periodically prints the
last-500-sample window mean of the in-memory forces_log, so long (~1-6 h)
convergence runs are diagnosable without touching the benchmark runner or the
solver.  All physics / config / post-processing stays in run.py.

Usage (identical argv to run.py):
  python scripts/_suboff_e2e_monitor.py --resolution 48 --steps 20000 \
      --device sdaa:0 --collision mrt --friction mix50 --out DIR
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks/pending/suboff_re1000"))

import run as runmod  # noqa: E402
from tensorlbm.general_sim import GeneralSimEngine  # noqa: E402

_orig_run = GeneralSimEngine.run

# frontal->wetted rescale for the raw engine-unit window mean (see run.py)
_REScale = None


def _monitor(engine: GeneralSimEngine, interval: float = 60.0) -> None:
    while not getattr(engine, "_mon_stop", False):
        time.sleep(interval)
        log = getattr(engine, "forces_log", None)
        if not log:
            continue
        w = log[-500:] if len(log) >= 500 else log
        raw = sum(e["cd_total"] for e in w) / len(w)
        cp = sum(e["cd_pressure"] for e in w) / len(w)
        cf = sum(e["cd_friction"] for e in w) / len(w)
        print(
            f"[MON] step={engine.step_count:6d} nlog={len(log):5d} "
            f"win{len(w)}_raw_cd_tot={raw:.6f} cd_p={cp:.6f} cd_f={cf:.6f}",
            flush=True,
        )


def run_with_monitor(self, steps=None):
    self._mon_stop = False
    t = threading.Thread(target=_monitor, args=(self,), daemon=True)
    t.start()
    try:
        return _orig_run(self, steps)
    finally:
        self._mon_stop = True


GeneralSimEngine.run = run_with_monitor

if __name__ == "__main__":
    runmod.main()