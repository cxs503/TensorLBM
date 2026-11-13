#!/usr/bin/env python3
"""Record the machine-verifiable part of the delivery (code pointers + pytest).

Writes ``artifacts/verification.json``:
  * source line of each boundary-condition function under test,
  * sha256 of the two source files touched,
  * the pytest summary of ``tests/test_zou_he_channel_reconstruction.py``.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ART = HERE / "artifacts"
ART.mkdir(exist_ok=True)

TARGETS = {
    "line_inlet_velocity": "zou_he_inlet_velocity",
    "line_inlet_pressure": "zou_he_inlet_pressure",
    "line_outlet_pressure": "zou_he_outlet_pressure",
    "line_apply_pressure": "apply_zou_he_pressure_channel_boundaries",
}


def line_of(path: Path, name: str) -> int:
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if re.match(rf"^def {name}\(", line):
            return i
    raise SystemExit(f"{name} not found in {path}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main() -> int:
    bnd = REPO / "src" / "tensorlbm" / "boundary" / "boundaries.py"
    init = REPO / "src" / "tensorlbm" / "__init__.py"
    tests = REPO / "tests" / "test_zou_he_channel_reconstruction.py"

    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(tests), "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    tail = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()][-1:]
    out = {
        **{k: line_of(bnd, v) for k, v in TARGETS.items()},
        "sha256_boundaries": sha256(bnd),
        "sha256_init": sha256(init),
        "pytest": {
            "cmd": "python3 -m pytest tests/test_zou_he_channel_reconstruction.py -q",
            "returncode": proc.returncode,
            "summary": tail[0] if tail else "",
        },
    }
    (ART / "verification.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0 if proc.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())