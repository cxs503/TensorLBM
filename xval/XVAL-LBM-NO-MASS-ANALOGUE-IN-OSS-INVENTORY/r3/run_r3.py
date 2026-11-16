"""Round-3 arms for XVAL-LBM-NO-MASS-ANALOGUE-IN-OSS-INVENTORY.

Round 2 (``r2/``) closed the anchor with the claim "C_I within 6.0-14.6 %" using the
exact unsteady-Stokes reference of ``tensorlbm.oscillatory_cylinder_inertia``.  Round 3
found that reference is generated from a Stokes-layer thickness that is ``sqrt(2)`` too
thick, so the round-2 reference overstates ``Re Gamma`` by 5-19 %.  This driver re-uses
the round-2 machinery verbatim (same protocol gate, same ramp, same three estimators,
same reference construction) and adds the *resolution* rungs that the corrected
similarity numbers demand: at fixed physics (beta = 50, KC, Ma, Re, box/D) the Stokes
layer in cells is ``delta = D/sqrt(pi beta)``, so D = 16 -> 32 -> 64 is
delta = 1.28 -> 2.55 -> 5.11 cells.

Usage
-----
    python xval/XVAL-LBM-NO-MASS-ANALOGUE-IN-OSS-INVENTORY/r3/run_r3.py --list
    python .../run_r3.py --arms r3_b50_d32,r3_b50_d64 --threads 4 --out <dir>
    python .../run_r3.py --arms r3_b50_d64 --ref-only      # potential reference only
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
R2 = HERE.parent / "r2"


def _load_r2():
    spec = importlib.util.spec_from_file_location("oc_r2", R2 / "oc_added_mass_r2.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["oc_r2"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


r2 = _load_r2()
R2Case = r2.R2Case

# omega = 2 pi/(200 D/16): keeps omega*D = 0.50 and the drive 102 acoustic linewidths
# away from the first box eigenfrequency at every D (the round-2 detuning rule).
ARMS = {
    "r3_b50_d16": R2Case("r3_b50_d16", D=16, beta=50.0, omega=2.0 * math.pi / 200.0,
                         max_spin_periods=120.0, gate_window=3.0, gate_tol=0.005),
    "r3_b50_d32": R2Case("r3_b50_d32", D=32, beta=50.0, omega=2.0 * math.pi / 400.0,
                         max_spin_periods=80.0, gate_window=3.0, gate_tol=0.005),
    "r3_b25_d32": R2Case("r3_b25_d32", D=32, beta=25.0, omega=2.0 * math.pi / 400.0,
                         max_spin_periods=80.0, gate_window=3.0, gate_tol=0.005),
    "r3_b100_d32": R2Case("r3_b100_d32", D=32, beta=100.0, omega=2.0 * math.pi / 400.0,
                          max_spin_periods=80.0, gate_window=3.0, gate_tol=0.005),
    "r3_b50_d64": R2Case("r3_b50_d64", D=64, beta=50.0, omega=2.0 * math.pi / 800.0,
                         max_spin_periods=40.0, gate_window=3.0, gate_tol=0.005),
}
ARM_ORDER = ["r3_b50_d32", "r3_b50_d64", "r3_b25_d32", "r3_b100_d32", "r3_b50_d16"]


def run_one(name: str, threads: int, out_root: Path, ref_only: bool,
            ref_levels: tuple[int, ...]) -> dict:
    case = ARMS[name]
    t0 = time.time()
    result: dict
    if ref_only:
        nx = int(round(case.box * case.D))
        ref = r2.reference_same_mask(nx, case.D, levels=ref_levels)
        result = {
            "case": asdict(case),
            "status": "ok",
            "mode": "reference_only",
            "reference": {"same_mask": ref},
            "delta_true_cells": case.D / math.sqrt(math.pi * case.beta),
            "wall_s": round(time.time() - t0, 2),
        }
    else:
        try:
            result = r2.run_arm(case, threads=threads)
        except Exception as exc:                       # honest failure, recorded
            result = {"case": asdict(case), "status": "failed",
                      "failure_reason": repr(exc)}
        result["delta_true_cells"] = case.D / math.sqrt(math.pi * case.beta)
        result["wall_s"] = round(time.time() - t0, 2)
    d = out_root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(f"[done] {name} status={result.get('status')} wall={result['wall_s']:.1f}s "
          f"delta_true={result['delta_true_cells']:.3f} cells", flush=True)
    if result.get("status") == "ok" and not ref_only:
        print(json.dumps({k: result.get(k) for k in ("stokes_layer_cells", "kpi",
                                                     "err_inphase_pct")},
                         indent=2, default=str)[:1500], flush=True)
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arms", default="")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="")
    ap.add_argument("--ref-only", action="store_true")
    ap.add_argument("--ref-levels", default="1,2",
                    help="lattice refinement levels of the potential reference solve")
    args = ap.parse_args(argv)
    if args.list:
        for n in ARM_ORDER:
            c = asdict(ARMS[n])
            c["delta_true_cells"] = ARMS[n].D / math.sqrt(math.pi * ARMS[n].beta)
            print(n, c)
        return 0
    names = [n for n in (args.arms.split(",") if args.arms else ARM_ORDER) if n]
    out_root = Path(args.out) if args.out else HERE / "runs"
    levels = tuple(int(x) for x in args.ref_levels.split(",") if x)
    for name in names:
        run_one(name, args.threads, out_root, args.ref_only, levels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())