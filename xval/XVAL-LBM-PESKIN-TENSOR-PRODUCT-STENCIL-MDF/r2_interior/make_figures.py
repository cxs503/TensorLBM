"""Figures for the round-2 XVAL: solid-interior treatment in the L4 cylinder run.

Reads ``artifacts/screen.json`` and ``artifacts/ladder.json`` (written by
``run_l4_interior.py``); nothing is hand-typed.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
FIG = HERE / "figures"


def load(name: str):
    p = ART / name
    return json.loads(p.read_text()) if p.exists() else None


def fig_history(screen: dict) -> Path:
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
    colours = {"penalty": "tab:red", "none": "tab:orange", "hard": "tab:blue", "hard_rho1": "tab:green"}
    for c in screen["cases"]:
        lab = c["case"]["interior"]
        h = c["hist"]
        ax[0].plot(h["t"], h["cd"], color=colours.get(lab, "k"), lw=1.4, label=f"interior={lab}")
        ax[1].plot(h["t"], h["umax_in_solid_hold"] if "umax_in_solid_hold" in h else h["umax_solid"],
                   color=colours.get(lab, "k"), lw=1.2, label=f"interior={lab}")
    ref = screen["cases"][0]["cd_reference"]
    for lvl, ls in ((0.15, "--"), (0.0, "-")):
        ax[0].axhspan(ref * (1 - lvl), ref * (1 + lvl), color="grey", alpha=0.18 if lvl else 0,
                      label=f"$\\pm${int(lvl*100)}% of $C_d$={ref}" if lvl else None)
    ax[0].axhline(ref, color="k", lw=1)
    ax[0].set_xlabel("$t\\,U/D$")
    ax[0].set_ylabel("$C_d$")
    ax[0].set_yscale("log")
    ax[0].set_title(f"cylinder drag, Re={screen['cases'][0]['case']['re']:.0f}, "
                    f"D={screen['cases'][0]['case']['D']} cells")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3, which="both")
    ax[1].set_xlabel("$t\\,U/D$")
    ax[1].set_ylabel(r"$\max|\mathbf{u}|$ inside the body  [$U$]")
    ax[1].set_title("solid permeability (interior treatment)")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    p = FIG / "r2_f1_interior_histories.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def fig_bars(screen: dict, ladder: dict | None) -> Path:
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
    labels = [c["case"]["interior"] for c in screen["cases"]]
    cds = [c["cd_mean_hold"] for c in screen["cases"]]
    errs = [abs(c["cd_rel_error"]) * 100 for c in screen["cases"]]
    x = range(len(labels))
    ax[0].bar(x, cds, color=["tab:red", "tab:orange", "tab:blue", "tab:green"])
    ref = screen["cases"][0]["cd_reference"]
    ax[0].axhline(ref, color="k", ls="-", lw=1.2, label=f"literature $C_d$={ref} (Dennis & Chang)")
    ax[0].axhline(ref * 1.15, color="k", ls="--", lw=1, label="+15% owner bar")
    ax[0].set_xticks(list(x))
    ax[0].set_xticklabels(labels, rotation=15)
    ax[0].set_ylabel("$C_d$ (hold-out mean)")
    ax[0].set_title("interior treatment -> drag")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3, axis="y")
    ax[1].bar(x, errs, color=["tab:red", "tab:orange", "tab:blue", "tab:green"])
    ax[1].axhline(15.0, color="k", ls="--", lw=1, label="15% owner bar")
    ax[1].set_yscale("log")
    ax[1].set_xticks(list(x))
    ax[1].set_xticklabels(labels, rotation=15)
    ax[1].set_ylabel("$|C_d - C_{d,ref}|/C_{d,ref}$  [%]")
    ax[1].set_title("absolute error vs literature")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3, axis="y", which="both")
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    p = FIG / "r2_f2_variant_bars.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def fig_ladder(ladder: dict) -> Path:
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    for c in ladder["cases"]:
        cc = c["case"]
        ax.plot(cc["D"], c["cd_mean_hold"], "o", ms=8,
                label=f"Re={cc['re']:.0f}, h={cc['h_marker']}, D={cc['D']}")
        ax.errorbar(cc["D"], c["cd_mean_hold"], yerr=c["cd_std_hold"], color="grey", lw=0.8, capsize=3)
    refs = {}
    for c in ladder["cases"]:
        refs[c["case"]["re"]] = c["cd_reference"]
    for i, (re, ref) in enumerate(sorted(refs.items())):
        ax.axhline(ref, ls="--", lw=1.0, color=["k", "tab:purple"][i % 2],
                   label=f"literature Re={re:.0f}: {ref}")
        ax.axhspan(ref * 0.85, ref * 1.15, color="grey", alpha=0.12)
    ax.set_xlabel("body resolution D [cells/diameter]")
    ax.set_ylabel("$C_d$")
    ax.set_title("grid / marker-resolution ladder (hard interior)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    p = FIG / "r2_f3_ladder.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def main() -> int:
    screen = load("screen.json")
    ladder = load("ladder.json")
    made = []
    if screen:
        made.append(fig_history(screen))
        made.append(fig_bars(screen, ladder))
    if ladder:
        made.append(fig_ladder(ladder))
    for p in made:
        print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())