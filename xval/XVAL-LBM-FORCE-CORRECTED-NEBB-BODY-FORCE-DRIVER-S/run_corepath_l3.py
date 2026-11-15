#!/usr/bin/env python3
"""L3 artifact: the force-corrected NEBB wall exercised through the *main* solver.

Runs ``GPUCorePathSolver`` (D3Q19 core path, ``wall_bc="nebb"``) and its
half-way bounce-back control to steady state and compares both against their
own analytic reference (``channel_analytic_velocity_nebb`` for the node-wall
closure, ``channel_analytic_velocity`` for the half-cell closure).  Writes
``artifacts/corepath/result.json`` + ``nebb_bodyforce_xval.png``.

    python xval/XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S/run_corepath_l3.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO / "src"))

from tensorlbm.verification.nebb_bodyforce.run import (  # noqa: E402
    OWNER_THRESHOLD,
    STRETCH,
    run_core_path_arm,
)


def figure(result: dict, path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - optional dependency
        return False
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for key, arm in result["arms"].items():
        prof = arm["profile_ux_fluid"]
        ref = arm["analytic_ux_fluid"]
        y = [i + 1 for i in range(len(prof))]  # fluid rows 1..ny-2
        style = {"nebb": ("o-", "NEBB (wall_bc='nebb')"), "halfway": ("s--", "half-way BB")}
        mk, label = style.get(arm["wall_bc"], ("-", key))
        ax.plot(prof, y, mk, ms=3, label=f"{label}  err={arm['err_global_vs_analytic']:.2e}")
        ax.plot(ref, y, "k:", lw=1)
    ax.plot([], [], "k:", label="analytic parabola")
    ax.set_xlabel("$u_x$ (lattice)")
    ax.set_ylabel("y (lattice rows)")
    ax.set_title("main core-path solver, ny=18 body-force channel")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(_HERE / "artifacts" / "corepath"))
    ap.add_argument("--ny", type=int, nargs="+", default=[18])
    ap.add_argument("--max-steps", type=int, default=40_000)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = build_result(args.ny, args.max_steps)
    res["figure"] = "nebb_bodyforce_xval.png" if figure(res, out / "nebb_bodyforce_xval.png") else None
    (out / "result.json").write_text(json.dumps(res, indent=2, default=float))
    print("wrote", out / "result.json")
    return 0


def build_result(ny_list, max_steps: int = 40_000) -> dict:
    """Run both wall closures on the core path and return the artifact payload."""
    res = {
        "anchor": "XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S",
        "entry": "python xval/XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S/"
                 "run_corepath_l3.py",
        "solver": "tensorlbm.gpu_core_path.GPUCorePathSolver (D3Q19, main core path)",
        "reference": "channel_analytic_velocity_nebb: ux(y)=a/(2 nu) y (H-y), H=ny-1 for "
                     "wall_bc='nebb'; channel_analytic_velocity (H=ny-2) for 'halfway'",
        "thresholds": {"owner_15pct": OWNER_THRESHOLD, "ecosystem_3pct": STRETCH},
        "arms": {},
    }
    for ny in ny_list:
        for bc in ("halfway", "nebb"):
            t0 = time.perf_counter()
            arm = run_core_path_arm(bc, ny=ny, max_steps=max_steps)
            print(f"{bc} ny={ny} err={arm['err_global_vs_analytic']:.4e} "
                  f"steps={arm['steps_to_steady']} ({time.perf_counter() - t0:.1f}s)",
                  flush=True)
            res["arms"][f"corepath_{bc}_ny{ny}"] = arm
    return res


if __name__ == "__main__":
    raise SystemExit(main())
