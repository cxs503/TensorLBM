#!/usr/bin/env python3
"""Figures + programmatic number dump for XVAL-LBM-ORACLE-MINIMAL-D2Q9-FS.

Reads ``artifacts/xval_results.json`` (written by ``run_xval.py``) and emits
PNG figures plus ``artifacts/numbers.md``.  Every number that appears in the
report is taken from here (no hand-copied values).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
REPO = HERE.parent.parent

MM = json.loads((REPO / "src" / "artifacts" / "martin_moyce_1952_dam_break.json").read_text())
MM_T = np.array([d["t_star"] for d in MM["data"]])
MM_Z = np.array([d["X_star"] for d in MM["data"]])
RIT = json.loads((REPO / "src" / "artifacts" / "ritter_1892_dam_break_analytical.json").read_text())
RIT_T = np.array([d["t_star"] for d in RIT["data"]])
RIT_Z = np.array([d["X_star"] for d in RIT["data"]])

LABEL = {"tensorlbm_koerner": "TensorLBM Körner (D2Q9)", "oss_port": "OSS oracle port (Maarten-vd-Sande/lbm)"}
COLOR = {"tensorlbm_koerner": "C0", "oss_port": "C3"}
STYLE = {"tensorlbm_koerner": "-", "oss_port": "--"}


def load(tag="xval_results.json"):
    return json.loads((ART / tag).read_text())


def runs_ok(payload):
    return [r for r in payload["runs"] if r.get("status") == "ok"]


def pick(payload, arm, a, tau=1.0, g=None):
    for r in payload["runs"]:
        if r["arm"] != arm or r.get("a") != a or abs(r.get("tau", -1) - tau) > 1e-12:
            continue
        if g is not None and abs(r.get("g", -1) - g) > 1e-15:
            continue
        if r.get("rho_prev_alias", True) is False or r.get("interface_rho", "oss") != "oss":
            continue
        return r
    return None


def fig_front(payload, out):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax = axes[0]
    ax.plot(MM_T, MM_Z, "ko-", ms=4, lw=1.2, label="Martin & Moyce 1952 (digitised)")
    ax.plot(RIT_T, RIT_Z, "k:", lw=1.0, label="Ritter 1892 (inviscid)")
    for arm in ("tensorlbm_koerner", "oss_port"):
        r = pick(payload, arm, 32)
        if not r:
            continue
        T = [s["T"] for s in r["series"]]
        Z = [s["Z"] for s in r["series"]]
        ax.plot(T, Z, STYLE[arm], color=COLOR[arm], lw=1.8,
                label=f"{LABEL[arm]} (a=32, τ=1)")
    ax.set_xlim(0, 4.2)
    ax.set_ylim(0.8, 4.5)
    ax.set_xlabel(r"$T = t\sqrt{g/h_0}$")
    ax.set_ylabel(r"$Z = x_{front}/a$")
    ax.set_title("2D dam-break front position (a = 32, τ = 1)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper left")
    ax = axes[1]
    for arm in ("tensorlbm_koerner", "oss_port"):
        r = pick(payload, arm, 32)
        if not r or "vs_mm" not in r:
            continue
        pts = r["vs_mm"]["points"]
        ax.plot([p["T"] for p in pts], [p["rel_err_pct"] for p in pts],
                STYLE[arm], color=COLOR[arm], marker="s", ms=3,
                label=f"{LABEL[arm]}: max {r['vs_mm']['max_abs_rel_err_pct']:.1f}%")
    ax.axhline(15, color="r", ls=":", lw=1.2, label="pre-declared 15 % gate")
    ax.axhline(-15, color="r", ls=":", lw=1.2)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel(r"$T = t\sqrt{g/h_0}$")
    ax.set_ylabel("relative front-position error [%]")
    ax.set_title("Error vs Martin & Moyce (a = 32, τ = 1)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_fields(payload, out):
    arms = [a for a in ("tensorlbm_koerner", "oss_port") if pick(payload, a, 32)]
    if not arms:
        return None
    fig, axes = plt.subplots(len(arms), 2, figsize=(9.5, 3.1 * len(arms)), squeeze=False)
    for i, arm in enumerate(arms):
        r = pick(payload, arm, 32)
        ff = r.get("final_fields")
        if not ff:
            continue
        umag = np.array(ff["umag"], dtype=float)
        fill = np.array(ff["fill"], dtype=float)
        im = axes[i][0].imshow(umag.T, origin="lower", cmap="inferno", aspect="auto")
        axes[i][0].set_title(f"{LABEL[arm]}\n|u| at T={r['series'][-1]['T']:.2f}", fontsize=8)
        plt.colorbar(im, ax=axes[i][0], fraction=0.03)
        im = axes[i][1].imshow(fill.T, origin="lower", cmap="Blues", aspect="auto", vmin=0, vmax=1)
        axes[i][1].set_title("fill level ε", fontsize=8)
        plt.colorbar(im, ax=axes[i][1], fraction=0.03)
        for j in range(2):
            axes[i][j].set_xlabel("x [lattice]")
            axes[i][j].set_ylabel("y [lattice]")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_convergence(payload, out):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax = axes[0]
    for arm in ("tensorlbm_koerner", "oss_port"):
        xs, ys = [], []
        for a in sorted({r["a"] for r in runs_ok(payload) if r["arm"] == arm}):
            r = pick(payload, arm, a)
            if r and "vs_mm" in r:
                xs.append(1.0 / a)
                ys.append(r["vs_mm"]["max_abs_rel_err_pct"])
        if xs:
            ax.plot(xs, ys, marker="o", ls=STYLE[arm], color=COLOR[arm], label=LABEL[arm])
            for x, y in zip(xs, ys):
                ax.annotate(f"a={int(round(1/x))}", (x, y), fontsize=7,
                            textcoords="offset points", xytext=(3, 4))
    ax.axhline(15, color="r", ls=":", lw=1.2, label="15 % gate")
    ax.set_xlabel("1/a  (lattice cells per column width)")
    ax.set_ylabel("max |rel err| vs M&M [%]")
    ax.set_title("Grid convergence of the front-position error")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5)
    ax = axes[1]
    if payload.get("arm_vs_arm"):
        gaps = payload["arm_vs_arm"]
        aa = [g["a"] for g in gaps]
        mx = [g["max_abs_rel_diff_pct"] for g in gaps]
        mn = [g["mean_abs_rel_diff_pct"] for g in gaps]
        ax.plot(aa, mx, "ks-", label="max")
        ax.plot(aa, mn, "k^--", label="mean")
        ax.axhline(15, color="r", ls=":", lw=1.2, label="15 % gate")
    ax.set_xlabel("a (resolution)")
    ax.set_ylabel("arm-to-arm |ΔZ|/Z [%]")
    ax.set_title("TensorLBM vs OSS oracle on the common T grid")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_tau(payload, out):
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(MM_T, MM_Z, "ko-", ms=3.5, lw=1.0, label="Martin & Moyce 1952")
    for arm in ("tensorlbm_koerner", "oss_port"):
        for tau, alpha in ((0.6, 0.55), (1.0, 1.0), (1.4, 0.55)):
            r = pick(payload, arm, 32, tau=tau)
            if not r:
                continue
            ax.plot([s["T"] for s in r["series"]], [s["Z"] for s in r["series"]],
                    STYLE[arm], color=COLOR[arm], alpha=alpha, lw=1.4,
                    label=f"{arm} τ={tau:g}")
    ax.set_xlabel(r"$T = t\sqrt{g/h_0}$")
    ax.set_ylabel(r"$Z = x_{front}/a$")
    ax.set_title("Relaxation-time sensitivity (a = 32)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6.5, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_inventory(payload, out):
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for arm in ("tensorlbm_koerner", "oss_port"):
        r = pick(payload, arm, 32)
        if not r or not r.get("inventory"):
            continue
        dt = np.sqrt(r["g"] / r["h0"])
        ax.semilogy([p["step"] * dt for p in r["inventory"]],
                    [max(abs(p["drift_pct"]), 1e-9) for p in r["inventory"]],
                    STYLE[arm], color=COLOR[arm],
                    label=f"{LABEL[arm]} (max {r.get('inventory_drift_max_pct', float('nan')):.2f} %)")
    ax.set_xlabel(r"$T = t\sqrt{g/h_0}$")
    ax.set_ylabel("|liquid-inventory drift| [%]")
    ax.set_title("Mass ledger drift (a = 32)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_mach(payload, out):
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.plot(MM_T, MM_Z, "ko-", ms=3.5, lw=1.0, label="Martin & Moyce 1952")
    for arm in ("tensorlbm_koerner", "oss_port"):
        r = pick(payload, arm, 16, g=0.1)
        if not r:
            continue
        ax.plot([s["T"] for s in r["series"]], [s["Z"] for s in r["series"]],
                STYLE[arm], color=COLOR[arm], lw=1.6,
                label=f"{LABEL[arm]} g=0.1 (OSS shipped)")
    ax.set_xlabel(r"$T = t\sqrt{g/h_0}$")
    ax.set_ylabel(r"$Z = x_{front}/a$")
    ax.set_title("OSS shipped gravity (g=0.1, Ma≈6, clamp active)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def numbers(payload) -> str:
    lines = ["# XVAL numbers (programmatic dump — do not hand-edit)\n"]
    lines.append("## Run table\n")
    lines.append("| label | arm | a | τ | g | ν | steps | u_max | Z max err % | mean err % | "
                 "pass 15% | inventory drift % | wall s |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in payload["runs"]:
        if r.get("status") != "ok":
            lines.append(f"| {r.get('label', r['arm'])} | {r['arm']} | - | - | - | - | - | - | - | - | "
                         f"{r.get('status')} {r.get('error', '')} | - | - |")
            continue
        m = r.get("vs_mm", {})
        lines.append(
            f"| {r['label']} | {r['arm']} | {r['a']} | {r['tau']:g} | {r['g']:g} | "
            f"{r.get('nu', float('nan')):.4f} | {r.get('steps')} | {r.get('u_max', 0):.4f} | "
            f"{m.get('max_abs_rel_err_pct', float('nan'))} | {m.get('mean_abs_rel_err_pct', float('nan'))} | "
            f"{m.get('pass_15pct')} | {r.get('inventory_drift_max_pct', float('nan'))} | "
            f"{r.get('wall_seconds')} |")
    lines.append("\n## Arm-to-arm (TensorLBM vs OSS oracle)\n")
    for g in payload.get("arm_vs_arm", []):
        lines.append(f"* a={g['a']} τ={g['tau']:g}: max {g['max_abs_rel_diff_pct']} %, "
                     f"mean {g['mean_abs_rel_diff_pct']} %, pass15={g['pass_15pct']}")
    lines.append("\n## Controls\n")
    for c in payload.get("controls", []):
        lines.append(f"* {c['arm']}: u_max = {c['u_max_lattice']:.4e} ({c['steps']} steps)")
    return "\n".join(lines) + "\n"


def main() -> int:
    payload = load()
    outs = []
    outs.append(fig_front(payload, ART / "fig_front_vs_mm.png"))
    outs.append(fig_fields(payload, ART / "fig_field_maps.png"))
    outs.append(fig_convergence(payload, ART / "fig_convergence.png"))
    outs.append(fig_tau(payload, ART / "fig_tau_sensitivity.png"))
    outs.append(fig_inventory(payload, ART / "fig_inventory_drift.png"))
    outs.append(fig_mach(payload, ART / "fig_mach_probe.png"))
    txt = numbers(payload)
    (ART / "numbers.md").write_text(txt)
    print("\n".join(str(o) for o in outs if o))
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())