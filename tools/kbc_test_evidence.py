#!/usr/bin/env python3
"""Run the campaign test suite and record the machine-readable evidence.

Writes ``artifacts/kbc/p495w4-n03/tests.json`` with the pass/fail counts, the
per-module coverage of the new code and the pytest invocation, so the report can
quote them without hand-copying numbers.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    cmd = [sys.executable, "-m", "pytest", "tests/test_kbc_channel.py", "-q",
           "--cov=tensorlbm.kbc_collision", "--cov=tensorlbm.kbc_channel",
           "--cov=tensorlbm.kbc_channel_native", "--cov-report=term-missing"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    passed = re.search(r"(\d+) passed", out)
    failed = re.search(r"(\d+) failed", out)
    cover = {}
    for line in out.splitlines():
        m = re.match(r"src/tensorlbm/(kbc_\w+)\.py\s+(\d+)\s+(\d+)\s+(\d+)%", line.strip())
        if m:
            cover[m.group(1)] = {"stmts": int(m.group(2)), "miss": int(m.group(3)),
                                 "cover_pct": int(m.group(4))}
    total = re.search(r"^TOTAL\s+(\d+)\s+(\d+)\s+(\d+)%", out, re.M)
    payload = {
        "command": " ".join(cmd[1:]),
        "passed": int(passed.group(1)) if passed else 0,
        "failed": int(failed.group(1)) if failed else 0,
        "returncode": proc.returncode,
        "coverage": cover,
        "total_cover_pct": int(total.group(3)) if total else None,
        "green": proc.returncode == 0,
    }
    out_dir = ROOT / "artifacts/kbc/p495w4-n03"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tests.json").write_text(json.dumps(payload, indent=1))
    print(json.dumps(payload, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
