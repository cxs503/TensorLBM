"""Snapshot run: VOF interface + pressure field for the report figures.

Saves ``F`` and ``p`` at a handful of experimental times so the flow-field
figure is generated from a real run (no synthetic fields).

Usage
-----
    python benchmarks/free_surface_dambreak_wigley/run_snapshots.py \
        --cells 16 --out runs/p351w1_n07/snapshots.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.free_surface import DamBreakMoyce  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, default=16)
    ap.add_argument("--wall", default="no-slip")
    ap.add_argument("--out", default="runs/p351w1_n07/snapshots.npz")
    args = ap.parse_args(argv)

    db = DamBreakMoyce(cells_per_a=args.cells, wall=args.wall)
    db.cfg.poisson_tol = 1.0e-5
    targets = [0.0, 0.41, 1.19, 1.97, 2.64, 2.96]      # experimental axis T
    fields = {"F": [], "p": [], "T": targets, "dx": db.cfg.dx, "dy": db.cfg.dy,
              "lx": db.cfg.lx, "ly": db.cfg.ly, "cells_per_a": args.cells,
              "wall": args.wall}
    for T in targets:
        if T > 0.0:
            db.sim.run(T / np.sqrt(2.0))
        fields["F"].append(db.sim.F.copy())
        fields["p"].append(db.sim.p.copy())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **{k: np.array(v) for k, v in fields.items()})
    meta = {"cells_per_a": args.cells, "wall": args.wall, "targets": targets,
            "n_steps": db.sim.n_steps, "mass": db.sim.ledger.as_dict(),
            "front": [float(np.max(np.flatnonzero(np.any(f[:2, :] > 0.5, axis=0))) + 1) * db.cfg.dx
                      for f in fields["F"]]}
    Path(out).with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta["front"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())