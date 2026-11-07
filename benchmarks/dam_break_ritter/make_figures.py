"""Render the Ritter dam-break benchmark figures from the run artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parents[1]
SRC_ROOT = REPO_ROOT / "src"
for _p in (str(SRC_ROOT), str(REPO_ROOT)):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from tensorlbm.shallow_water import (  # noqa: E402
    ShallowWaterConfig,
    ritter_depth,
    run_ritter_dam_break,
)

FIG_DIR = BENCH_DIR / "figures"


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    report = json.loads((BENCH_DIR / "result.json").read_text(encoding="utf-8"))
    cases = report["cases"]
    ref = max(cases, key=lambda c: c["nx"])

    # ---- figure 1: front law
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.9))
    t = [f["t"] for f in ref["front_series"]]
    ax1.plot(t, [f["front_analytic"] for f in ref["front_series"]], "k-", lw=1.6,
             label=r"Ritter $x_f=2\sqrt{gh_0}\,t$")
    ax1.plot(t, [f["front_mass"] for f in ref["front_series"]], "o", ms=3.5,
             color="tab:blue", label="simulation (integral / mass front)")
    ax1.plot(t, [f["front_geometric"] for f in ref["front_series"]], "s", ms=3.0,
             color="tab:orange", label="simulation (geometric tip tracker)")
    ax1.set_xlabel(r"$t\sqrt{g/h_0}$")
    ax1.set_ylabel(r"$x_f/h_0$")
    ax1.set_title(f"front position, nx={ref['nx']}")
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3)

    for case in cases:
        tt = [f["t"] for f in case["front_series"]]
        ax2.plot(tt, [f["err_mass_pct"] for f in case["front_series"]], "o-", ms=3,
                 lw=1.0, label=f"nx={case['nx']} mass front")
    ax2.axhspan(-5, 5, color="tab:green", alpha=0.15, label="±5 % gate")
    ax2.axhline(0.0, color="k", lw=0.8)
    ax2.set_xlabel(r"$t\sqrt{g/h_0}$")
    ax2.set_ylabel("front error [%]")
    ax2.set_title("front-law error vs time")
    ax2.legend(fontsize=6.5)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ritter_front_law.png", dpi=130)
    plt.close(fig)

    # ---- figure 2: self-similar depth profiles vs the analytic collapse
    cfg = ShallowWaterConfig(nx=ref["nx"])
    run = run_ritter_dam_break(cfg)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.9))
    import torch

    x = torch.linspace(cfg.x_min, cfg.x_max, cfg.nx, dtype=torch.float64)
    plot_times = (2.0, 3.5, 5.0)
    for t_target in plot_times:
        idx = min(range(len(run.time)), key=lambda k: abs(run.time[k] - t_target))
        tt = run.time[idx]
        xs = [float(v) for v in x]
        ax1.plot(xs, run.profiles[idx][1], "-", lw=1.3, label=rf"sim $t^*$={tt:.2f}")
        ax1.plot(xs, [ritter_depth(v, tt, cfg.h0, cfg.g) for v in xs], "k--", lw=0.9)
        ax2.plot(
            [v / tt for v in xs],
            [h * tt * tt for h in run.profiles[idx][1]],
            "-", lw=1.3, label=rf"sim $t^*$={tt:.2f}",
        )
    xs = [float(v) for v in x]
    ax2.plot([v / 2.0 for v in xs], [ritter_depth(v, 2.0, cfg.h0, cfg.g) * 4.0 for v in xs],
             "k--", lw=1.0, label="Ritter similarity form")
    ax1.set_xlabel(r"$x/h_0$")
    ax1.set_ylabel(r"$h/h_0$")
    ax1.set_title("depth profiles (dashed = analytic)")
    ax1.set_xlim(-2.5, 10.5)
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3)
    ax2.set_xlabel(r"$\xi/t = x/(t\sqrt{gh_0})$")
    ax2.set_ylabel(r"$h\,t^2/h_0$")
    ax2.set_title("self-similar collapse")
    ax2.set_xlim(-1.5, 2.2)
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ritter_profiles.png", dpi=130)
    plt.close(fig)
    print(f"wrote {FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())