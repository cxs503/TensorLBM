"""Render the Rayleigh--Taylor benchmark figures from the run artifacts."""

from __future__ import annotations

import csv
import json
import math
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

FIG_DIR = BENCH_DIR / "figures"


def _history(path: Path) -> tuple[list[float], list[float], list[float]]:
    ts, az, ac = [], [], []
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            ts.append(float(row["step"]))
            ac.append(float(row["a_column_moment"]))
            az.append(float(row["a_interface_zc"]))
    return ts, az, ac


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    report = json.loads((BENCH_DIR / "result.json").read_text(encoding="utf-8"))
    cases = report["cases"]
    lam = [c["lam"] for c in cases]

    # ---- figure 1: amplitude growth in the linear regime
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    colours = plt.cm.viridis([0.15, 0.5, 0.85])
    for idx, c in enumerate(cases):
        hist = BENCH_DIR / "runs" / f"lam{c['lam']}" / "growth_history.csv"
        if not hist.exists():
            continue
        ts, az, _ = _history(hist)
        a_hi = c["fit_a_over_lam"][1] * c["lam"]
        sel = [(t, a) for t, a in zip(ts, az) if 0 < a <= a_hi]
        ax.semilogy(ts, az, ".", ms=2.5, color=colours[idx], alpha=0.55)
        if sel:
            t0, a0 = sel[0]
            tt = [s[0] for s in sel]
            ax.semilogy(
                tt,
                [a0 * math.exp(c["sigma_sim"] * (t - t0)) for t in tt],
                "-",
                lw=1.2,
                color=colours[idx],
                label=(
                    rf"$\lambda$={c['lam']}: $\sigma_{{sim}}$={c['sigma_sim']:.5f} "
                    rf"($\sigma_{{th}}$={c['sigma_analytic']:.5f})"
                ),
            )
    ax.set_xlabel("lattice steps $t$")
    ax.set_ylabel(r"interface mode amplitude $a(t)$")
    ax.set_title("Rayleigh–Taylor linear growth (Boussinesq D2Q9, $A=1/3$, $\\eta_0=\\lambda/100$)")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rt_amplitude_growth.png", dpi=130)
    plt.close(fig)

    # ---- figure 2: normalised growth rate + convergence
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.9))
    ratios = [c["ratio_sim_over_analytic"] for c in cases]
    ax1.axhspan(0.97, 1.03, color="tab:green", alpha=0.15, label="±3 % gate")
    ax1.plot(lam, ratios, "o-", color="tab:blue", label=r"$\sigma_{sim}/\sqrt{Agk}$")
    visc = [c["ratio_sim_over_viscous"] for c in cases]
    ax1.plot(lam, visc, "s--", color="tab:orange", label=r"$\sigma_{sim}/\sigma_{visc}$")
    ax1.axhline(1.0, color="k", lw=0.8)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(lam)
    ax1.set_xticklabels([str(v) for v in lam])
    ax1.set_xlabel(r"perturbation wavelength $\lambda$ [cells]")
    ax1.set_ylabel(r"normalised growth rate")
    ax1.set_title("growth-rate ratio vs analytic")
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3)

    errs = [abs(r - 1.0) for r in ratios]
    ax2.loglog(lam, errs, "o-", color="tab:red", label="|σ_sim/σ_th − 1|")
    gci = report["grid_convergence"]
    for key, val in gci.get("observed_orders", {}).items():
        ax2.annotate(
            f"p({key})={val:.2f}",
            xy=(lam[-1], errs[-1]),
            xytext=(-10, 18),
            textcoords="offset points",
            fontsize=7,
        )
    ax2.set_xticks(lam)
    ax2.set_xticklabels([str(v) for v in lam])
    ax2.set_xlabel(r"$\lambda$ [cells]")
    ax2.set_ylabel("relative error")
    ax2.set_title("grid refinement of the normalised growth rate")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rt_convergence.png", dpi=130)
    plt.close(fig)
    print(f"wrote {FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())