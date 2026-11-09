#!/usr/bin/env python3
"""gci_v4_figs — figures for the gci_v4 artifact (numbers read from the artifact/raw files).

Outputs PNGs next to the artifact: ``figures_gci_v4/fig_{1..4}_*.png``.
Every plotted number comes from ``gci_v4.json`` or the raw probe/archive files, so the
figures cannot drift away from the judged record.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_HERE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIGDIR = _HERE / "figures_gci_v4"


def _find(name: str) -> Path:
    for cand in (_HERE / "artifacts" / name, _REPO_ROOT / "artifacts" / name):
        if cand.exists():
            return cand
    return _HERE / "artifacts" / name


def main() -> None:
    FIGDIR.mkdir(exist_ok=True)
    art = json.loads((_HERE / "gci_v4.json").read_text())
    anchor_ct = art["anchor"]["CT"]
    anchor_re = art["anchor"]["Re"]

    # ---------------- figure 1: Re sweep at L=48 (both closures) ----------------
    probe = json.loads(_find("gci_v4_probe.json").read_text())
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for closure, marker in (("mrt", "o"), ("smagorinsky", "s")):
        rows = [r for r in probe["rows"] if r["closure"] == closure]
        re = [r["Re_target"] for r in rows]
        ct = [r["ct_last"] for r in rows]
        ax.plot(re, ct, marker=marker, label=f"LBM L=48, {closure}", lw=1.4)
    ax.axhline(anchor_ct, color="k", ls="--", lw=1.2,
               label=f"DTRC anchor C_T = {anchor_ct:.4e}")
    cf = [0.075 / (math.log10(r) - 2.0) ** 2 for r in (1e5, 1e6, 1e7)]
    ax.plot([1e5, 1e6, 1e7], cf, color="0.55", ls=":", lw=1.2, label="ITTC-1957 C_F (friction only)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Re (hull length)")
    ax.set_ylabel("C_T (bare-hull wetted area)")
    ax.set_title("LBM drag collapses as Re grows at fixed resolution\n(L=48 probe, 20-step transient read)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_1_re_sweep_L48.png", dpi=150)
    plt.close(fig)

    # ---------------- figure 2: archived resolved-Re ladders ----------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ls = [96, 112, 128]
    arms = art["arm_archived_resolved_Re_ladders"]
    styles = {
        "AFF3_sail_Re1000_standard": ("C0", "o", "AFF-3 with sail, Re=1000, standard"),
        "AFF1_bare_Re1000_mix50": ("C1", "s", "AFF-1 bare hull, Re=1000, mix50"),
    }
    cf_ref = 1.328 / math.sqrt(1000.0)
    for name, arm in arms.items():
        if "ct" not in arm:
            continue
        col, mk, lab = styles.get(name, ("C2", "^", name))
        ax1.plot(ls, [arm["ct"][str(L)] for L in ls], marker=mk, color=col, label=lab)
        ax2.plot(ls, [arm["err_pct"][str(L)] for L in ls], marker=mk, color=col, label=lab)
    ax1.axhline(cf_ref, color="k", ls="--", lw=1.2, label=f"Blasius Cf(1000) = {cf_ref:.4e}")
    ax2.axhspan(-3, 3, color="g", alpha=0.12, label="+/-3% band")
    ax2.axhline(0.0, color="k", lw=0.8)
    ax1.set_xlabel("L (cells per hull length)")
    ax1.set_ylabel("C_T")
    ax1.set_title("Archived three-tier ladders (real runs, Re=1000)")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=7)
    ax2.set_xlabel("L (cells per hull length)")
    ax2.set_ylabel("error vs Blasius [%]")
    ax2.set_title("Same ladders: pre-asymptotic, no positive observed order")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_2_archived_ladders.png", dpi=150)
    plt.close(fig)

    # ---------------- figure 3: why the anchor Re is unreachable ----------------
    req = art["resolution_requirement"]
    tiers = req["tiers"]
    Ls = [t["L"] for t in tiers]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax1.plot(Ls, [t["y_plus_first_cell"] for t in tiers], "o-", color="C3")
    for y, lab in ((1.0, "y+=1 (wall-resolved)"), (30.0, "y+=30"), (500.0, "y+=500 (wall-model ceiling)")):
        ax1.axhline(y, color="0.4", ls="--", lw=0.9)
        ax1.text(97, y * 1.25, lab, fontsize=7, color="0.3")
    ax1.set_yscale("log")
    ax1.set_xlabel("L (cells per hull length)")
    ax1.set_ylabel("y+ of the first cell centre")
    ax1.set_title("First-cell wall distance at the DTRC anchor Re\n(ITTC-1957 u_tau)")
    ax1.grid(alpha=0.3, which="both")
    Lgrid = [10 ** (2 + 0.01 * i) for i in range(0, 401)]
    ax2.plot(Lgrid, [26.66 * L**3 for L in Lgrid], color="C0", label="cells = 26.66 L^3")
    ax2.plot(Ls, [26.66 * L**3 for L in Ls], "ko", ms=4, label="frozen ladder tiers")
    for key, lab in (("wall_model_upper_yplus_500", "wall-model L~450"), ("wall_resolved_yplus_1", "wall-resolved L~2e5")):
        Ln = req["requirements"][key]["L_cells_needed"]
        ax2.axvline(Ln, color="0.4", ls="--", lw=0.9)
        ax2.text(Ln * 0.55, 1e6, lab, rotation=90, fontsize=7, color="0.3")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("L (cells per hull length)")
    ax2.set_ylabel("cells in the frozen domain formula")
    ax2.set_title("Resolution the anchor Re actually demands")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_3_resolution_requirement.png", dpi=150)
    plt.close(fig)

    # ---------------- figure 4: anchor-Re time history (L=48) ----------------
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    plotted = False
    for name, col in (("gci_v4_probe_anchor_mrt400.json", "C0"),
                      ("gci_v4_probe_anchor_smag400.json", "C3"),
                      ("gci_v4_probe_anchor_mrt300.json", "C0"),
                      ("gci_v4_probe_anchor_smag300.json", "C3")):
        p = _find(name)
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        r = d["rows"][0]
        series = r.get("ct_series") or []
        if not series:
            continue
        steps = [5 * (i + 1) for i in range(len(series))]
        ax.plot(steps, series, color=col, lw=1.3,
                label=f"L=48 {r['closure']} ({len(series)} samples)")
        plotted = True
    if plotted:
        ax.axhline(anchor_ct, color="k", ls="--", lw=1.2, label=f"DTRC anchor {anchor_ct:.4e}")
        ax.set_xlabel("step")
        ax.set_ylabel("C_T (bare-hull wetted area)")
        ax.set_title("Anchor-Re attempted run at L=48: no settled window\n(never reaches the anchor plateau)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGDIR / "fig_4_anchor_re_history.png", dpi=150)
    plt.close(fig)
    print(f"figures written to {FIGDIR}")


if __name__ == "__main__":
    main()