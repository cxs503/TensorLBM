"""A1 helper: compute Stokes-damping rows in small chunks and merge them.

The full A1 table (6 configurations x 2000 nonlinear box steps) does not fit in a
single interactive shell slot on a loaded node, so the campaign can build the
same JSON record incrementally:

    python scripts/p269w2_n04_a1_chunk.py --out artifacts/p269w2_n04_local 32:0.6,32:0.8
    python scripts/p269w2_n04_a1_chunk.py --out artifacts/p269w2_n04_local 64:0.6,64:0.8

Rows are merged (de-duplicated by ``(n, tau)``) into
``<out>/points/001_a1_stokes.json`` in exactly the schema the campaign's
``aggregate()`` expects.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.stability.channel_shear import stokes_damping_benchmark  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", help="comma separated n:tau pairs, e.g. 32:0.6,64:1.2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-steps", type=int, default=2000)
    args = ap.parse_args(argv)

    points = Path(args.out) / "points"
    points.mkdir(parents=True, exist_ok=True)
    dest = points / "001_a1_stokes.json"
    record = json.loads(dest.read_text()) if dest.exists() else {
        "kind_task": "a1_stokes", "rows": []
    }
    have = {(int(r["n"]), float(r["tau"])) for r in record["rows"]}
    for item in args.spec.split(","):
        n_s, tau_s = item.split(":")
        n, tau = int(n_s), float(tau_s)
        if (n, tau) in have:
            print(f"[skip] n={n} tau={tau}")
            continue
        t0 = time.time()
        row = stokes_damping_benchmark(n=n, mode=1, tau=tau, n_steps=args.n_steps)
        record["rows"].append(row)
        have.add((n, tau))
        dest.write_text(json.dumps(record, indent=1))
        print(f"[done] n={n} tau={tau} {time.time()-t0:.1f}s "
              f"rel_meas={row['rel_error_measured']:.2e} rel_vn={row['rel_error_von_neumann']:.2e}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())