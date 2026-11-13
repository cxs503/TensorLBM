#!/usr/bin/env python3
"""Figures for the graph-unroll BPTT LES cross-validation.

Reads ``artifacts/xval_graph_unroll_bptt_les/results.json`` (written by
``scripts/xval_graph_unroll_bptt_les.py``) and writes four PNGs next to it:

1. ``fig1_twin_loss.png``   — twin loss curve, BPTT iterates vs golden/sweep.
2. ``fig2_ke_dns.png``      — kinetic-energy decay, reference vs arms.
3. ``fig3_identifiability.png`` — physical (DNS) loss curve flatness.
4. ``fig4_gradient.png``    — BPTT gradient vs central FD (log scale).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "xval_graph_unroll_bptt_les"


def main() -> int:
    r = json.loads((OUT / "results.json").read_text())

    # ---- fig1: twin loss curves -------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, kind in zip(axes, ("tgv", "fourier")):
        for key, entry in r["stage1_twin"][kind].items():
            truth = float(key.split("_")[-1])
            grid = [c for c, _ in entry["sweep"]]
            vals = [v for _, v in entry["sweep"]]
            ax.semilogy(grid, vals, "-", label=f"loss sweep (truth {truth:g})")
            ax.axvline(truth, color="k", ls=":", lw=1)
            hist = entry["bptt_history"]
            ax.semilogy([c for c, _ in hist], [max(v, 1e-14) for _, v in hist], "o-",
                        ms=3, label=f"BPTT iterates (final {entry['bptt']['cs']:.3f})")
            ax.plot([entry["golden"]["cs"]], [entry["golden"]["loss"]], "r*", ms=12,
                    label=f"golden ({entry['golden']['cs']:.3f})")
        ax.set_xlabel("$C_s$ [-]")
        ax.set_ylabel("relative-$L_2$ field loss $J$ [-]")
        ax.set_title(f"identical twin — {kind}")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle("Graph-unroll BPTT calibrates the LES constant on a planted truth", y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_twin_loss.png", dpi=140)
    plt.close(fig)

    # ---- fig2: KE decay vs DNS proxy -------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, kind in zip(axes, ("tgv", "fourier")):
        d = r["stage2_physical"][kind]
        steps = sorted(int(k) for k in d["ke_ref"])
        ax.plot(steps, [d["ke_ref"][str(s)] for s in steps], "k-", lw=2.2,
                label=f"reference (nx={d['reference']['nx']}, filtered)")
        for name, style in (("no_model", "s--"), ("upstream_0.17", "^--"),
                            ("bptt", "o-"), ("golden", "x:")):
            if name not in d["ke_arms"]:
                continue
            ax.plot(steps, [d["ke_arms"][name][str(s)] for s in steps], style, ms=4,
                    label=f"{name} (Cs={dict(no_model=0.0, upstream_0_17=0.17).get(name, d.get(name, {}).get('cs', 0)) or 0:.3f})"
                    if name in ("no_model", "upstream_0.17") else
                    f"{name} (Cs={_cs_of(d, name):.3f})")
        ax.set_xlabel("time step $t$ [-]")
        ax.set_ylabel(r"kinetic energy $E$ $= 0.5\,\langle u^2\rangle$ [-]")
        ax.set_title(f"physical case — {kind}")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle("Energy decay: coarse LES arms vs the fine-grid reference", y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_ke_dns.png", dpi=140)
    plt.close(fig)

    # ---- fig3: identifiability of the physical loss -----------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, kind in zip(axes, ("tgv", "fourier")):
        d = r["stage2_physical"][kind]
        grid = [c for c, _ in d["sweep"]]
        vals = [v for _, v in d["sweep"]]
        ax.plot(grid, vals, "o-", ms=3, label="loss $J(C_s)$ (DNS target)")
        ax.axvline(d["bptt"]["cs"], color="C3", ls="--",
                   label=f"BPTT $C_s^*$={d['bptt']['cs']:.3f}")
        ax.axvline(d["golden"]["cs"], color="C2", ls=":",
                   label=f"golden $C_s^*$={d['golden']['cs']:.3f}")
        flat = (max(vals) - min(vals)) / d["sweep_range"]["loss_at_cs0"] * 100.0
        ax.set_title(f"{kind}: sweep range {flat:.1f}% of $J(0)$")
        ax.set_xlabel("$C_s$ [-]")
        ax.set_ylabel("relative-$L_2$ field loss $J$ [-]")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle("Identifiability of a single LES constant from field matching", y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_identifiability.png", dpi=140)
    plt.close(fig)

    # ---- fig4: gradient agreement ----------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for kind, marker in (("tgv", "o"), ("fourier", "s")):
        for key, entry in r["stage1_twin"][kind].items():
            cs = [g["cs"] for g in entry["x1_gradient"]]
            ax.loglog(cs, [abs(g["finite_difference"]) for g in entry["x1_gradient"]],
                      marker, color="C0", ms=6, alpha=0.8,
                      label="central FD" if key.endswith("0.17") and kind == "tgv" else None)
            ax.loglog(cs, [abs(g["autograd"]) for g in entry["x1_gradient"]],
                      "x", color="C3", ms=8, alpha=0.8,
                      label="autograd (BPTT)" if key.endswith("0.17") and kind == "tgv" else None)
    ax.set_xlabel("$C_s$ [-]")
    ax.set_ylabel(r"$|dJ/dC_s|$ [-]")
    ax.set_title("BPTT gradient vs central finite difference (twin problems)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_gradient.png", dpi=140)
    plt.close(fig)

    print("wrote figures to", OUT)
    return 0


def _cs_of(d: dict, name: str) -> float:
    if name == "bptt":
        return d["bptt"]["cs"]
    if name == "golden":
        return d["golden"]["cs"]
    return 0.0


if __name__ == "__main__":
    raise SystemExit(main())