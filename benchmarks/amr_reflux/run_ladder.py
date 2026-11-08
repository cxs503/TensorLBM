#!/usr/bin/env python3
"""1-D advection AMR reflux ladder: run the rungs, dump JSON + figures.

    PYTHONPATH=src python benchmarks/amr_reflux/run_ladder.py \
        --out artifacts/amr_reflux_ladder
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tensorlbm.amr_reflux import (
    AdvectionAMR1D,
    AdvectionLadderConfig,
    run_convection_ladder,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=str, default="artifacts/amr_reflux_ladder")
    p.add_argument("--n-coarse", type=int, default=96)
    p.add_argument("--steps", type=int, default=120)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result = run_convection_ladder(steps=args.steps, n_coarse=args.n_coarse)
    payload = result.to_dict()
    (out / "ladder.json").write_text(json.dumps(payload, indent=1))

    # ---- time series for the figure ------------------------------------
    cfg = AdvectionLadderConfig(n_coarse=args.n_coarse)
    series = {}
    for label, reflux in (("reflux_on", True), ("reflux_off", False)):
        solver = AdvectionAMR1D(cfg)
        inv0 = solver.inventory()["total"]
        drift, mismatch = [], []
        for _ in range(args.steps):
            diag = solver.step(reflux=reflux)
            inv = solver.inventory()["total"]
            drift.append(abs(inv - inv0) / inv0)
            mismatch.append(diag["interface_mismatch"])
        series[label] = {
            "drift_rel": drift,
            "mismatch": mismatch,
            "final_profile": solver.composite_solution().tolist(),
        }
    exact = AdvectionAMR1D(cfg).exact_solution(0.45, 0.06, args.steps * cfg.dt).tolist()
    coarse_only = AdvectionAMR1D(cfg, patches=())
    coarse_only.run(args.steps, reflux=True)
    payload["series"] = series
    payload["exact"] = exact
    payload["coarse_only"] = coarse_only.phi.tolist()
    (out / "ladder.json").write_text(json.dumps(payload, indent=1))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    x = [(i + 0.5) / args.n_coarse for i in range(args.n_coarse)]
    axes[0, 0].plot(x, exact, "k-", lw=1.2, label="exact")
    axes[0, 0].plot(x, series["reflux_on"]["final_profile"], "o-", ms=2.5, label="AMR + reflux")
    axes[0, 0].plot(x, series["reflux_off"]["final_profile"], "--", label="AMR, no reflux")
    axes[0, 0].plot(x, payload["coarse_only"], ":", label="coarse only")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("phi")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_title("1-D advection, 2:1 patch after 120 coarse steps")

    axes[0, 1].semilogy(series["reflux_on"]["drift_rel"], label="reflux on")
    axes[0, 1].semilogy(series["reflux_off"]["drift_rel"], label="reflux off")
    axes[0, 1].axhline(1e-10, color="k", ls="--", lw=0.8, label="1e-10 criterion")
    axes[0, 1].set_xlabel("coarse step")
    axes[0, 1].set_ylabel("relative mass drift")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].set_title("composite mass conservation")

    axes[1, 0].semilogy([abs(v) for v in series["reflux_on"]["mismatch"]], label="interface mismatch")
    axes[1, 0].axhline(1e-10, color="k", ls="--", lw=0.8)
    axes[1, 0].set_xlabel("coarse step")
    axes[1, 0].set_ylabel("|delta mass| at interface")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].set_title("coarse/fine interface flux imbalance")

    rung_text = "\n".join(
        f"{r['rung']}: {'PASS' if r['pass'] else 'FAIL'}" for r in payload["rungs"]
    )
    axes[1, 1].axis("off")
    axes[1, 1].text(0.02, 0.95, rung_text, va="top", family="monospace", fontsize=9)
    axes[1, 1].set_title("ladder rungs")
    fig.tight_layout()
    fig.savefig(out / "ladder.png", dpi=140)
    print(json.dumps({"pass": payload["pass"], "rungs": [
        {"rung": r["rung"], "pass": r["pass"]} for r in payload["rungs"]
    ]}, indent=1))
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())