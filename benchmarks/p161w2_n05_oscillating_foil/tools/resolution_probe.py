#!/usr/bin/env python
"""Diagnostic: does the measured thrust increment converge with resolution / kernel support?

Runs one fixed physical case (pure heave, ``h0/c = 0.15``, ``k = 2``, St = 0.19)
plus its locked-body reference on several grids and with two delta kernels.  The
thrust increment ``dC_T = C_T + C_D0`` is the quantity the verification face
needs; the Garrick closed form at these parameters is ``C_T = 0.30``.

Writes ``artifacts/ext_<label>.json`` payloads (same schema as run_sweep).
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
sys.path.insert(0, BENCH)
sys.path.insert(0, os.path.join(BENCH, "..", "..", "src"))

import run_sweep as S  # noqa: E402


SPECS = [
    dict(label="ext_lock_c32_hat", mode="locked", k=2.0, chord=32, cycles=6, kernel="hat", passes=2),
    dict(label="ext_heave_c32_hat", mode="heave", k=2.0, h0c=0.15, chord=32, cycles=8,
         kernel="hat", passes=2),
    dict(label="ext_lock_c32_nosolid", mode="locked", k=2.0, chord=32, cycles=6, solid=False),
    dict(label="ext_heave_c32_nosolid", mode="heave", k=2.0, h0c=0.15, chord=32, cycles=8,
         solid=False),
    dict(label="ext_lock_c64", mode="locked", k=2.0, chord=64, cycles=6),
    dict(label="ext_heave_c64", mode="heave", k=2.0, h0c=0.15, chord=64, cycles=8),
    dict(label="ext_lock_c96", mode="locked", k=2.0, chord=96, cycles=6),
    dict(label="ext_heave_c96", mode="heave", k=2.0, h0c=0.15, chord=96, cycles=8),
]


def _run(spec):
    return S.run_case(spec, 32)


def main() -> int:
    outdir = os.path.join(BENCH, "artifacts")
    with ProcessPoolExecutor(max_workers=4) as ex:
        for payload in ex.map(_run, SPECS):
            with open(os.path.join(outdir, f"{payload['label']}.json"), "w") as fh:
                json.dump(payload, fh)
            c = payload["coeffs"]
            print(
                f"{payload['label']:18s} grid={payload['grid']} C_T={c['C_T']:+.4f} "
                f"th={payload['theory']['C_T']:+.4f} cv={payload['extra'].get('C_T_control_volume'):+.4f}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
