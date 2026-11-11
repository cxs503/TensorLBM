#!/usr/bin/env python
"""Control-volume cross-check with the downstream plane outside the outlet sponge.

The campaign payloads integrated the momentum flux at ``x = nx - 3`` which sits
*inside* the 6 % outlet sponge, biasing the control-volume force estimate by
~10 %.  The solver now exposes ``cv_planes`` (``x2`` outside the sponge); this
script re-runs a few representative cases so the corrected agreement can be
reported.  Outputs ``artifacts/ext2_<label>.json``.
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
    dict(label="cv_lock_ref", mode="locked", k=1.0, cycles=6, lock_target="lock_ref"),
    dict(label="cv_heave_h015_k1", mode="heave", k=1.0, h0c=0.15, cycles=8,
         lock_target="heave_h015_k1"),
    dict(label="cv_heave_h015_k2", mode="heave", k=2.0, h0c=0.15, cycles=8,
         lock_target="heave_h015_k2"),
    dict(label="cv_gci_c42", mode="heave", k=1.0, h0c=0.15, chord=42, cycles=8,
         lock_target="gci_c42"),
]


def _run(spec):
    return S.run_case(spec, 32)


def main() -> int:
    outdir = os.path.join(BENCH, "artifacts")
    with ProcessPoolExecutor(max_workers=4) as ex:
        for p in ex.map(_run, SPECS):
            with open(os.path.join(outdir, f"ext2_{p['label']}.json"), "w") as fh:
                json.dump(p, fh)
            print(
                f"{p['label']:18s} C_T={p['coeffs']['C_T']:+.4f} "
                f"cv_C_T={-p['extra']['C_D_control_volume']:+.4f} "
                f"rel={abs(p['coeffs']['C_T'] + p['extra']['C_D_control_volume'])/max(abs(p['coeffs']['C_T']),1e-9):.3f}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
