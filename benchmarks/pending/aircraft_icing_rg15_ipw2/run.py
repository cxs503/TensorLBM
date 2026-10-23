"""RG-15 IPW2 aircraft-icing benchmark -- run dispatcher (archive edition).

Usage (one GPU via CUDA_VISIBLE_DEVICES):
    python run.py b3     # 3 official cases, glaze multishot production
    python run.py b4     # case 3.3 + official IPW 7-bin DSD
    python run.py b5     # shot/accel invariance grid (8 cells)
    python run.py b6     # beta resolution ladder (nx 320/480/640)
    python run.py b7     # final-geometry flow-field render (CPU)
    python run.py all    # b3 -> b4 -> b5 -> b6 -> b7 in order

Outputs land in ./output/{b3,b4,b5,b6,b7}/; the RG-15 geometry and MCCS
reference cuts are read from ./reference/. The b4 arm requires the IC-E-D1
warmup-bins fix (branch exp/icing-warmup-bins); mono-disperse arms are
bitwise-identical on pristine vs fixed trees (A/B 40/40 arrays).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARMS = {
    "b3": ["_b3_b4_glaze.py"],
    "b4": ["_b3_b4_glaze.py", "--b4"],
    "b5": ["_b5_invariance.py"],
    "b6": ["_b6_beta_res.py"],
    "b7": ["_b7_render.py"],
}


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    order = list(ARMS) if which == "all" else [which]
    assert all(w in ARMS for w in order), f"unknown arm: {order}"
    for w in order:
        cmd = [sys.executable, str(HERE / ARMS[w][0]), *ARMS[w][1:]]
        print(f"[run.py] {w}: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True, cwd=str(HERE))


if __name__ == "__main__":
    main()
