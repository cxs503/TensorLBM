#!/usr/bin/env python3
"""Turn archived campaign logs into machine-readable failure records.

A diverged case is a *result* (honest FAIL), so it must land in the same place
as a completed one: ``case_data/<case>.failed.json``.  This script only parses
the archived logs under ``artifacts/p223w3_n02_channel_dns/logs/`` - no numbers
are typed by hand.

Usage::

    python benchmarks/p223w3_n02_channel_dns/archive_failures.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_HERE))

import campaign  # noqa: E402  (case matrix: re_tau / steps / cs live there)

LOGS = _REPO / "artifacts" / "p223w3_n02_channel_dns" / "logs"
OUT = _HERE / "case_data"
MESSAGE = re.compile(
    r"diverged: (?P<bad>\d+) of (?P<total>\d+) populations are non-finite "
    r"\(grid (?P<nx>\d+)x(?P<ny>\d+)x(?P<nz>\d+) tau0=(?P<tau0>[\d.]+) "
    r"regularized=(?P<reg>\d+)\)")



def main() -> int:
    OUT.mkdir(exist_ok=True)
    written = []
    for log in sorted(LOGS.glob("case_*.log")):
        name = log.stem.replace("case_", "")
        text = log.read_text(errors="replace")
        if "RuntimeError" not in text or "diverged" not in text:
            continue
        msg = MESSAGE.search(text)
        case = campaign.case_matrix().get(name) or campaign.RETIRED.get(name)
        if not (msg and case):
            print(f"skip {log.name}: cannot parse ({'no kernel message' if not msg else 'not in the campaign matrix'})")
            continue
        ny = int(msg.group("ny"))
        nz, nx = int(msg.group("nz")), int(msg.group("nx"))
        record = {
            "case": name,
            "source_log": str(log.relative_to(_REPO)),
            "geometry": {
                "nx": nx, "ny": ny, "nz": nz, "cells": nx * ny * nz,
                "re_tau": case.re_tau,
                "u_tau": case.u_tau,
                "half_height": 0.5 * (ny - 2),
                "dy_plus": case.re_tau / (0.5 * (ny - 2)),
                "steps": case.steps,
                "warmup": case.warmup,
                "smagorinsky_cs": case.cs,
                "regularized": int(msg.group("reg")),
                "tau0": float(msg.group("tau0")),
            },
            "error": f"les_chan diverged: {msg.group('bad')} of {msg.group('total')} "
                     f"populations non-finite (grid {nx}x{ny}x{nz} "
                     f"tau0={msg.group('tau0')} regularized={msg.group('reg')})",
            "non_finite_fraction": int(msg.group("bad")) / int(msg.group("total")),
        }
        (OUT / f"{name}.failed.json").write_text(json.dumps(record, indent=1))
        written.append(name)
    print("wrote failed records for:", ", ".join(written) if written else "(none)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())