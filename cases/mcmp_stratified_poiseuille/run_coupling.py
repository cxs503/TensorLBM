"""Secondary matrix: velocity-coupling x force-scheme at the segregating coupling.

Motivation (recorded *after* the primary matrix, disclosed in the report): with
the legacy velocity-shift forcing the ``mixture`` and ``viscous`` velocity
couplings diverge at ``|G_12| >= 1.5`` (prior claim tkc:tensorlbm:mcmp-viscous-
velocity-coupling).  This script re-tests that claim with the second-order Guo
forcing, which changes the momentum bookkeeping at the interface, and measures
the mixture flow rate of every stable combination.

Usage::

    PYTHONPATH=src:. python cases/mcmp_stratified_poiseuille/run_coupling.py <shard> <nshards>
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cases.mcmp_stratified_poiseuille.run_sweep import run_case  # noqa: E402

OUT_DIR = os.path.join("artifacts", "mcmp_stratified_poiseuille", "records")

STEPS = {25: 20000, 33: 30000}

#: (ny, u_eq, force_scheme)
MATRIX = [
    (25, "self", "shift"), (25, "self", "guo"),
    (25, "mixture", "shift"), (25, "mixture", "guo"),
    (25, "viscous", "shift"), (25, "viscous", "guo"),
    (33, "self", "shift"), (33, "self", "guo"),
    (33, "mixture", "shift"), (33, "mixture", "guo"),
    (33, "viscous", "shift"), (33, "viscous", "guo"),
]


def main(argv) -> int:
    shard = int(argv[1]) if len(argv) > 1 else 1
    nshards = int(argv[2]) if len(argv) > 2 else 1
    os.makedirs(OUT_DIR, exist_ok=True)
    todo = [c for i, c in enumerate(MATRIX) if i % nshards == shard - 1]
    print(f"coupling shard {shard}/{nshards}: {len(todo)} of {len(MATRIX)}", flush=True)
    for ny, u_eq, force in todo:
        tag = f"cp_ny{ny}_{u_eq}_{force}"
        t0 = time.time()
        rec = run_case(ny=ny, steps=STEPS[ny], G_12=-2.5, u_eq=u_eq,
                       force_scheme=force, mode="two_phase")
        rec["tag"] = tag
        with open(os.path.join(OUT_DIR, f"{tag}.json"), "w") as fh:
            json.dump(rec, fh, indent=1)
        line = dict(tag=tag, ny=ny, u_eq=u_eq, force=force, finite=rec["finite"],
                    steady=rec["steady"], seconds=round(time.time() - t0, 1))
        if rec["finite"]:
            line.update(dict(
                Q_mix=round(rec["Q_mix"], 5), phi=round(rec["phi_contrast"], 3),
                iface=round(rec["interface_crossing"], 3),
                u_max=round(rec["u_max_mix"], 5),
                s_over_u=round(rec["spurious_mix"] / max(rec["u_max_mix"], 1e-30), 3),
                errs={rule: round(ref["err_Q_mix"], 4)
                      for rule, ref in rec["analytical"].items()},
            ))
        print("SUMMARY " + json.dumps(line), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))