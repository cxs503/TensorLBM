"""Shard-able driver for the pre-declared MCMP stratified-Poiseuille matrix.

The matrix and every acceptance threshold are declared in
``docs/MCMP_STRATIFIED_POISEUILLE_SPEC.md`` *before* results are inspected; this
driver only enumerates that matrix.  Each case writes one JSON record under
``artifacts/mcmp_stratified_poiseuille/records/`` plus a ``SUMMARY`` line on
stdout (one-line, greppable from cluster logs).

Usage::

    PYTHONPATH=src:. python cases/mcmp_stratified_poiseuille/run_matrix.py 1 4
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cases.mcmp_stratified_poiseuille.run_sweep import run_case  # noqa: E402

OUT_DIR = os.path.join("artifacts", "mcmp_stratified_poiseuille", "records")

STEPS = {25: 20000, 33: 30000, 49: 45000}

#: (kind, ny, G_12, force_scheme, tau_lower, tau_upper)
MATRIX = [
    # --- primary: stratified (segregating) coupling, two force schemes ---
    ("two_phase", 25, -2.5, "shift", 1.0, 0.75),
    ("two_phase", 25, -2.5, "guo", 1.0, 0.75),
    ("two_phase", 33, -2.5, "shift", 1.0, 0.75),
    ("two_phase", 33, -2.5, "guo", 1.0, 0.75),
    ("two_phase", 49, -2.5, "shift", 1.0, 0.75),
    ("two_phase", 49, -2.5, "guo", 1.0, 0.75),
    # --- falsification arm: the parameter set of the QR audit (G12 = -1) ---
    ("two_phase", 25, -1.0, "shift", 1.0, 0.75),
    ("two_phase", 25, -1.0, "guo", 1.0, 0.75),
    ("two_phase", 33, -1.0, "shift", 1.0, 0.75),
    ("two_phase", 33, -1.0, "guo", 1.0, 0.75),
    # --- wall-convention calibration through the same code path ---
    ("walls", 25, 0.0, "shift", 1.0, 1.0),
    ("walls", 33, 0.0, "shift", 1.0, 1.0),
    ("walls", 49, 0.0, "shift", 1.0, 1.0),
    ("walls", 25, 0.0, "shift", 0.75, 0.75),
    ("walls", 33, 0.0, "shift", 0.75, 0.75),
    ("walls", 49, 0.0, "shift", 0.75, 0.75),
]


def tag(case) -> str:
    kind, ny, g12, force, tl, tu = case
    if kind == "walls":
        return f"walls_ny{ny}_tau{tl:g}"
    return f"tp_ny{ny}_g{abs(g12):g}_{force}"


def main(argv) -> int:
    shard = int(argv[1]) if len(argv) > 1 else 1
    nshards = int(argv[2]) if len(argv) > 2 else 1
    os.makedirs(OUT_DIR, exist_ok=True)
    todo = [c for i, c in enumerate(MATRIX) if i % nshards == shard - 1]
    print(f"shard {shard}/{nshards}: {len(todo)} of {len(MATRIX)} cases", flush=True)
    for case in todo:
        kind, ny, g12, force, tl, tu = case
        t0 = time.time()
        rec = run_case(
            ny=ny, G_12=g12, force_scheme=force, tau_lower=tl, tau_upper=tu,
            steps=STEPS[ny], mode=kind, u_eq="self",
        )
        rec["tag"] = f"{tag(case)}_shard{shard}"
        with open(os.path.join(OUT_DIR, f"{tag(case)}.json"), "w") as fh:
            json.dump(rec, fh, indent=1)
        line = dict(tag=tag(case), kind=kind, ny=ny, g12=g12, force=force,
                    finite=rec["finite"], steady=rec["steady"],
                    seconds=round(time.time() - t0, 1))
        if kind == "two_phase":
            line.update(dict(
                Q_mix=round(rec["Q_mix"], 5), phi_contrast=round(rec["phi_contrast"], 3),
                iface=round(rec["interface_crossing"], 3),
                u_max_mix=round(rec["u_max_mix"], 5),
                spurious_mix=round(rec["spurious_mix"], 5),
                trace_vel=round(rec["trace_velocity_max"], 4),
                errs={rule: {k: round(v, 4) for k, v in ref.items() if k.startswith("err_Q")}
                      for rule, ref in rec["analytical"].items()},
            ))
        else:
            line["wall"] = {k: {kk: round(vv, 5) for kk, vv in v.items() if kk != "H"}
                            for k, v in rec["wall_calibration"].items()}
        print("SUMMARY " + json.dumps(line), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))