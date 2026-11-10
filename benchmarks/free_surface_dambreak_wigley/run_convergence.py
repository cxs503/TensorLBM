"""Order-of-accuracy artifact for the VOF interface transport.

Criterion (d) of the verification face asks for the L2 convergence order on
three grids with R^2 >= 0.95 of the log-log fit.  The transport is exercised on
the analytic single-vortex field of :func:`tensorlbm.free_surface.advect_interface_convergence`,
where the reference field is produced by an RK4 flow-map integration that is
completely independent of the advection scheme under test.

Usage
-----
    python benchmarks/free_surface_dambreak_wigley/run_convergence.py \
        --out runs/p351w1_n07/convergence.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.free_surface import advect_interface_convergence  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--t-end", type=float, default=1.0)
    ap.add_argument("--out", default="runs/p351w1_n07/convergence.json")
    args = ap.parse_args(argv)

    smooth = advect_interface_convergence(sizes=tuple(args.sizes), t_end=args.t_end,
                                          width=0.02, cfl=0.3)
    sharp = advect_interface_convergence(sizes=tuple(args.sizes), t_end=args.t_end,
                                         width=0.004, cfl=0.3)
    payload = {
        "method": "single-vortex interface transport, MUSCL/superbee flux form, "
                  "reference = RK4 backward flow map (n_rk=400)",
        "t_end": args.t_end,
        "resolved_interface": smooth,
        "sharp_interface": sharp,
        "criteria": {
            "target_order": 1.5,
            "target_r2": 0.95,
            "resolved_order": smooth["order"],
            "resolved_r2": smooth["r2"],
            "pass": bool(smooth["order"] >= 1.5 and smooth["r2"] >= 0.95),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    print(json.dumps(payload["criteria"], indent=1))
    for name in ("resolved_interface", "sharp_interface"):
        d = payload[name]
        print(name, "h=", [round(h, 5) for h in d["h"]],
              "l2=", [f"{e:.3e}" for e in d["l2"]],
              f"order={d['order']:.2f} r2={d['r2']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())