#!/usr/bin/env python3
"""End-to-end suboff_re1000 run with live progress logging.

Runs benchmarks/pending/suboff_re1000/run.py's own main() unchanged (same
engine, same common modules, same result.json / final_field.pt artifacts)
but wraps GeneralSimEngine._sample_forces to print a running window-mean
of the engine-sampled Cd every ``PROG_EVERY`` force samples so long runs
can be monitored.  All command-line args after the script name are passed
straight through to run.py.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO = Path("/root/TensorLBM_feat2")
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.general_sim import GeneralSimEngine  # noqa: E402

PROG_EVERY = int(os.environ.get("PROG_EVERY", "100"))  # force samples (10 steps each)
_ORIG = GeneralSimEngine._sample_forces
_t0 = time.time()
_last = {"n": 0, "t": _t0}


def _patched(self, dpS, nu_lb):
    _ORIG(self, dpS, nu_lb)
    n = len(self.forces_log)
    if n > 0 and n % PROG_EVERY == 0 and n != _last["n"]:
        _last["n"] = n
        now = time.time()
        win = self.forces_log[-PROG_EVERY:]
        cp = sum(e["cd_pressure"] for e in win) / len(win)
        cf = sum(e["cd_friction"] for e in win) / len(win)
        dt = now - _last["t"]
        ms = dt / (PROG_EVERY * 10) * 1000
        _last["t"] = now
        print(
            f"[prog] step~{n * 10} cd_p={cp:.5f} cd_f={cf:.5f} "
            f"cd_tot={cp + cf:.5f} (frontal-norm) | {ms:.0f} ms/step "
            f"| elapsed {now - _t0:.0f}s",
            flush=True,
        )


GeneralSimEngine._sample_forces = _patched

# hand the remaining argv to run.py
sys.argv = [str(REPO / "benchmarks/pending/suboff_re1000/run.py")] + sys.argv[1:]
sys.path.insert(0, str(REPO))
import runpy  # noqa: E402

runpy.run_path(sys.argv[0], run_name="__main__")