"""Dam-break verification driver (Martin & Moyce 1952) for the VOF solver.

Runs one case (grid resolution x wall condition x viscosity) and writes a JSON
artifact with the front history, the RMSE against the 14 digitised experimental
points and the mass ledger.  Nothing is hard-coded: every number in the output
comes from :mod:`tensorlbm.free_surface`.

Usage
-----
    python benchmarks/free_surface_dambreak_wigley/run_dambreak.py \
        --cells 32 --wall no-slip --nu 1e-3 --out runs/p351w1_n07/dambreak
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.free_surface import DamBreakMoyce  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, default=32, help="cells across the dam width a")
    ap.add_argument("--wall", default="no-slip", choices=["free-slip", "no-slip"])
    ap.add_argument("--nu", type=float, default=1.0e-3)
    ap.add_argument("--cfl", type=float, default=0.25)
    ap.add_argument("--tol", type=float, default=1.0e-4, help="Poisson relative tolerance")
    ap.add_argument("--aspect", type=float, default=2.0)
    ap.add_argument("--t-end-mm", type=float, default=2.96)
    ap.add_argument("--out", default="runs/p351w1_n07/dambreak")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"cells{args.cells}_{args.wall}_nu{args.nu:g}"
    t0 = time.time()
    db = DamBreakMoyce(cells_per_a=args.cells, wall=args.wall, nu_l=args.nu,
                       cfl=args.cfl, aspect=args.aspect)
    db.cfg.poisson_tol = args.tol
    result = db.run(t_end_mm=args.t_end_mm)
    result["wall_seconds"] = time.time() - t0
    result["tag"] = tag
    result["grid"] = {"nx": db.cfg.nx, "ny": db.cfg.ny,
                      "lx": db.cfg.lx, "ly": db.cfg.ly, "dx": db.cfg.dx}
    (out / f"{tag}.json").write_text(json.dumps(result, indent=1))
    print(f"[{tag}] steps={result['n_steps']} rmse={result['rmse']:.4f} "
          f"rel={result['rmse_relative_percent']:.2f}% "
          f"mass={result['mass']['max_relative_drift']:.2e} "
          f"wall={result['wall_seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())