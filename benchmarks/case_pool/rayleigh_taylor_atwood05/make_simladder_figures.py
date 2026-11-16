"""Figures of the similarity ladder of ``CASE-RT-INSTABILITY-ATWOOD``.

Three panels, all built from ``artifacts/simladder/*/case.json`` (+ the per-step
``history.csv``); no number is typed in by hand.

fig1  Fr vs the diffuse-interface thickness ``k xi`` -- the resolution question:
      if the terminal Froude number is a property of the problem and not of the
      grid, the ladder must be flat; the Layzer-Goncharov limit ``1/sqrt(pi C)``
      and the +-15 % case gate are drawn as references, and the linear
      extrapolation to ``k xi -> 0`` is shown.
fig2  Fr vs the lattice Mach number -- the compressibility question, including
      the Mach-contaminated ``lam = 96`` arm of the *older* suite
      (``artifacts/late``, ``g`` raised with ``lam``) as the counter-example.
fig3  front trajectories ``h_bubble/sigma``, ``h_spike/sigma`` vs ``tau`` for the
      ladder arms with the fitted plateau over ``tau in [4, 6]`` -- the raw
      evidence behind ``Fr``.

Usage
-----
    python3 make_simladder_figures.py            # writes figures/simladder_*.png
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BENCH = Path(__file__).resolve().parent
SIM = BENCH / "artifacts" / "simladder"
LATE = BENCH / "artifacts" / "late"
FIG = BENCH / "figures"
C_2D = 1.0
GATE_PCT = 15.0
TAU_LO, TAU_HI = 4.0, 6.0


def _cases(root: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(Path(root).glob("*/case.json"))]


def _hist(case_dir: Path) -> list[dict]:
    with open(case_dir / "history.csv") as fh:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(fh)]


def fr_lz() -> float:
    return 1.0 / math.sqrt(math.pi * C_2D)


def fig_fr_vs_kxi(summary: dict) -> Path:
    rows = {r["name"]: r for r in summary["rows"]}
    lad = [rows[n] for n in summary["criterion"]["ladder_arms"] if n in rows]
    w4 = rows.get("S_w4_l064_g2e-4")
    fz = fr_lz()

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.axhspan(fz * (1 - GATE_PCT / 100), fz * (1 + GATE_PCT / 100),
               color="0.85", zorder=0, label=f"+-{GATE_PCT:g} % case gate")
    ax.axhline(fz, color="k", lw=1.2, ls="--",
               label=r"Layzer-Goncharov $1/\sqrt{\pi C}$, $C=1$")
    if len(lad) >= 2:
        xs = [r["k_xi"] for r in lad]
        ys = [r["fr"] for r in lad]
        ax.plot(xs, ys, "o-", color="C0", ms=8, lw=1.4,
                label=r"similarity ladder ($Ma$, $\zeta$, $k a_0$ fixed)")
        for r in lad:
            ax.annotate(rf"$\lambda$={r['lam']:.0f}", (r["k_xi"], r["fr"]),
                        textcoords="offset points", xytext=(6, 6), fontsize=9)
        ex = summary["ladder"]["extrapolation_kxi0"]
        if math.isfinite(ex.get("intercept", float("nan"))):
            ax.plot([0.0, max(xs)], [ex["intercept"], ex["intercept"] + ex["slope"] * 0.0],
                    ":", color="C0", lw=1.0)
            ax.plot([0.0], [ex["intercept"]], "s", mfc="none", mec="C0", ms=9,
                    label=rf"$k\xi\to0$ extrapolation = {ex['intercept']:.3f} "
                          rf"($R^2$={ex['r2']:.3f})")
    if w4:
        ax.plot([w4["k_xi"]], [w4["fr"]], "D", mfc="none", mec="C1", ms=9,
                label=rf"$k\xi$ control (ac_width 4 cells, $\lambda$={w4['lam']:.0f})")
    ax.set_xlabel(r"interface thickness $k\xi = 2\pi\,\xi_{cells}/\lambda$")
    ax.set_ylabel(r"terminal Froude number $Fr = u/\sqrt{A g \lambda}$")
    ax.set_title("Terminal front velocity vs interface thickness\n"
                 r"CASE-RT-INSTABILITY-ATWOOD, $A=1/2$, single mode")
    ax.set_xlim(0.0, max([r["k_xi"] for r in summary["rows"]]) * 1.12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out = FIG / "simladder_fr_vs_kxi.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_fr_vs_ma(summary: dict) -> Path:
    rows = {r["name"]: r for r in summary["rows"]}
    fz = fr_lz()
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.axhspan(fz * (1 - GATE_PCT / 100), fz * (1 + GATE_PCT / 100),
               color="0.85", zorder=0, label=f"+-{GATE_PCT:g} % case gate")
    ax.axhline(fz, color="k", lw=1.2, ls="--", label=r"Layzer-Goncharov $C=1$")
    for name, r in rows.items():
        marker = "o" if name.startswith("S_lad") or name == "S_ref_l064_g2e-4" else "s"
        ax.plot([r["ma_peak"]], [r["fr"]], marker, ms=8,
                color="C0" if marker == "o" else "C3",
                label=name if marker == "s" else None)
        ax.annotate(f"{r['lam']:.0f}", (r["ma_peak"], r["fr"]),
                    textcoords="offset points", xytext=(6, -10), fontsize=8)
    # the Mach-contaminated arm of the older suite (g raised together with lam)
    for c in _cases(LATE):
        if abs(c["lam"] - 96) < 1e-9:
            ax.plot([c["ma_peak"]], [c["fr_meas"]], "x", color="C4", ms=10,
                    label=r"older suite $\lambda$=96 ($g\propto\lambda$, $Ma$ not fixed)")
            ax.annotate(f"Ma={c['ma_peak']:.2f}\nFr={c['fr_meas']:.3f}",
                        (c["ma_peak"], c["fr_meas"]), textcoords="offset points",
                        xytext=(-70, 6), fontsize=8, color="C4")
    ax.set_xlabel(r"lattice Mach number $Ma = u_{max}/c_s$")
    ax.set_ylabel(r"terminal Froude number $Fr$")
    ax.set_title("Terminal front velocity vs lattice Mach number\n"
                 "similarity ladder vs the $g\\propto\\lambda$ arm of the older suite")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    out = FIG / "simladder_fr_vs_ma.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_fronts(summary: dict) -> Path:
    names = summary["criterion"]["ladder_arms"]
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    colors = ["C0", "C2", "C3", "C1"]
    for i, name in enumerate(names):
        d = SIM / name
        if not (d / "history.csv").exists():
            continue
        hist = _hist(d)
        tau = [r["tau"] for r in hist]
        hb = [r["h_bubble_over_lam"] for r in hist]
        hs = [r["h_spike_over_lam"] for r in hist]
        c = colors[i % len(colors)]
        ax.plot(tau, hb, "-", color=c, lw=1.2, label=f"{name} bubble")
        ax.plot(tau, hs, "--", color=c, lw=1.2, label=f"{name} spike")
        win = [(t, b, s) for t, b, s in zip(tau, hb, hs) if TAU_LO <= t <= TAU_HI]
        if len(win) >= 3:
            for k, col in ((1, c), (2, c)):
                xs = [w[0] for w in win]
                ys = [w[k] for w in win]
                n = len(xs)
                mx, my = sum(xs) / n, sum(ys) / n
                b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / \
                    sum((x - mx) ** 2 for x in xs)
                ax.plot([TAU_LO, TAU_HI], [my - b * (mx - TAU_LO), my + b * (TAU_HI - mx)],
                        "-", color="k", lw=2.2, alpha=0.5 if k == 1 else 0.5)
    ax.axvspan(TAU_LO, TAU_HI, color="0.9", zorder=0, label=r"$Fr$ fit window")
    ax.set_xlabel(r"$\tau = t\sqrt{A g/\lambda}$")
    ax.set_ylabel(r"front position $h/\lambda$")
    ax.set_title("Bubble / spike trajectories and the constant-velocity window\n"
                 "black segments: least-squares plateaus used for $Fr$")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, ncol=2, loc="upper left")
    fig.tight_layout()
    out = FIG / "simladder_fronts.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main() -> int:
    summary = json.loads((SIM / "summary.json").read_text())
    outs = [fig_fr_vs_kxi(summary), fig_fr_vs_ma(summary), fig_fronts(summary)]
    for o in outs:
        print(f"[figures] {o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())