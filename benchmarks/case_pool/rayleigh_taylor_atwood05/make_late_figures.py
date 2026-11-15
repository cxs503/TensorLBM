"""Figures for the late-time front-velocity study of CASE-RT-INSTABILITY-ATWOOD.

Reads only the artifacts written by ``run_late.py`` (``artifacts/late/<case>/
history.csv`` + ``case.json``) -- no run, no hard-coded numbers.

    fig_late1_fronts.png    front trajectories h/lam vs tau: the two tips, the
                            linear normal-mode growth, the Layzer asymptote, and
                            (dashed) the legacy ``max(zc)-mean(zc)`` estimator
                            whose drift is the artefact this study removes.
    fig_late2_velocity.png  instantaneous front velocity in units of the Layzer
                            velocity, the +-15 % gate band, the pre-declared
                            window, and the multi-valued-column count (interface
                            topology diagnostic) on a twin axis.

Usage: ``python3 make_late_figures.py``
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parents[2]
_extra = [str(REPO_ROOT / "src"), str(REPO_ROOT), str(BENCH_DIR), str(BENCH_DIR / "ref")]
sys.path[:] = [q for q in sys.path if q not in _extra]
sys.path[:0] = _extra
import late_analysis as la  # noqa: E402

LATE = BENCH_DIR / "artifacts" / "late"
FIG = BENCH_DIR / "figures"


def load(case_dir: Path) -> tuple[dict, list[dict]]:
    meta = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    with (case_dir / "history.csv").open(encoding="utf-8") as fh:
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(fh)]
    return meta, rows


def fig_fronts(cases: list[tuple[dict, list[dict]]]) -> Path:
    fig, axes = plt.subplots(1, len(cases), figsize=(5.2 * len(cases), 4.4), squeeze=False)
    for ax, (meta, rows) in zip(axes[0], cases):
        tau = [r["tau"] for r in rows]
        lam, g, a = meta["lam"], meta["g"], meta["atwood"]
        h0 = rows[0]["h_bubble_over_lam"]
        ax.plot(tau, [r["h_bubble_over_lam"] for r in rows], "C0-", lw=1.8, label="bubble tip (sim)")
        ax.plot(tau, [r["h_spike_over_lam"] for r in rows], "C1--", lw=1.8, label="spike tip (sim)")
        ax.plot(tau, [r["h_up_zc_over_lam"] for r in rows], "C3:", lw=1.2,
                label=r"legacy $\max(z_c)-\overline{z_c}$")
        ax.plot(tau, [r["h_dn_zc_over_lam"] for r in rows], "C4:", lw=1.2,
                label=r"legacy $\overline{z_c}-\min(z_c)$")
        # linear normal-mode growth: a/lam = h0 exp(sigma t), sigma t = tau sqrt(2 pi)
        lin = [h0 * math.exp(t * math.sqrt(2.0 * math.pi)) for t in tau]
        ax.plot(tau, lin, "k-.", lw=1.0, label=r"linear $a_0e^{\sigma t}$")
        # Layzer asymptote started at the end of the linear phase (a/lam = 0.1)
        fr = la.layzer_froude(a, g, lam, la.C_2D)
        t0 = math.log(0.1 / h0) / math.sqrt(2.0 * math.pi)
        ax.plot(tau, [0.1 + fr * max(0.0, t - t0) for t in tau], "k--", lw=1.0,
                label=rf"Layzer $Fr={fr:.3f}$")
        ax.axvspan(la.TAU_LO, la.TAU_HI, color="0.85", zorder=0, label="gate window")
        ax.set_ylim(0, max(2.6, min(4.0, max(r["h_bubble_over_lam"] for r in rows) * 1.1)))
        ax.set_xlabel(r"$\tau = t\sqrt{Ag/\lambda}$")
        ax.set_ylabel(r"front displacement $h/\lambda$")
        ax.set_title(f"{meta['name']}\n" + r"$A=1/2$, $\lambda$="
                     f"{lam}, g={g:.1e}, {meta['layers']:.0f}$\\lambda$ box", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIG / "fig_late1_fronts.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_velocity(cases: list[tuple[dict, list[dict]]]) -> Path:
    fig, axes = plt.subplots(1, len(cases), figsize=(5.2 * len(cases), 4.4), squeeze=False)
    for ax, (meta, rows) in zip(axes[0], cases):
        tau = [r["tau"] for r in rows]
        lam, g, a = meta["lam"], meta["g"], meta["atwood"]
        sq = math.sqrt(a * g * lam)
        # centred finite difference of the tip trajectory, in units of the Layzer velocity
        fr_lz = la.layzer_froude(a, g, lam, la.C_2D)
        u = [float("nan")]
        for i in range(1, len(rows) - 1):
            dt = tau[i + 1] - tau[i - 1]
            dh = rows[i + 1]["h_bubble_over_lam"] - rows[i - 1]["h_bubble_over_lam"]
            u.append(dh / dt * lam * math.sqrt(a * g / lam) / sq / fr_lz)
        u.append(float("nan"))
        ax.plot(tau, u, "C0-", lw=1.5, label="bubble tip, inst. $u/Fr_{LZ}$")
        ax.axhline(1.0, color="k", lw=1.0, label="Layzer C=1")
        ax.axhspan(0.85, 1.15, color="C2", alpha=0.15, label="gate $\\pm15\\%$")
        ax.axvspan(la.TAU_LO, la.TAU_HI, color="0.85", zorder=0)
        ax.set_ylim(0.0, 1.6)
        ax.set_xlabel(r"$\tau$")
        ax.set_ylabel(r"$u\,/\,(\sqrt{Ag\lambda}\,Fr_{LZ})$")
        ax.set_title(f"{meta['name']}: measured $Fr={meta['fr_meas']:.3f}$, "
                     f"$C_{{eff}}={meta['c_eff']:.2f}$",
                     fontsize=9)
        ax2 = ax.twinx()
        ax2.plot(tau, [r["multi_columns"] for r in rows], "C5-", lw=0.9, alpha=0.7)
        ax2.set_ylabel("columns with >1 crossing", color="C5", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="C5", labelsize=7)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIG / "fig_late2_velocity.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    cases = []
    for d in sorted(LATE.glob("*/case.json")):
        cases.append(load(d.parent))
    if not cases:
        print("no artifacts: run run_late.py first", file=sys.stderr)
        return 1
    for p in (fig_fronts(cases), fig_velocity(cases)):
        print(f"wrote {p.relative_to(REPO_ROOT)}")
    if os.environ.get("RT_SHOW"):
        print(json.dumps([c[0]["name"] for c in cases]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
