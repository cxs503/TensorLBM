#!/usr/bin/env python3
"""m18218 audit figures — built ONLY from the author's shipped artifact
(xval_results.json, snapshot sha256 in evidence_snapshot/SHA256SUMS.txt) plus
the auditor's independent re-run numbers in repro_results.json.

No hand-typed physics numbers: every curve is read from a JSON file on disk.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
SNAP = HERE / "evidence_snapshot"
ROOT = HERE.parents[1]


def load():
    art = json.loads((SNAP / "artifacts" / "xval_results.json").read_text())
    mm = json.loads((ROOT / "src" / "artifacts" /
                     "martin_moyce_1952_dam_break.json").read_text())
    repro = json.loads((HERE / "repro_results.json").read_text())
    return art, mm, repro


def pick(runs, arm, a, rho_fixed=False):
    for r in runs:
        if r["arm"] != arm or r.get("a") != a or r.get("status") != "ok":
            continue
        if rho_fixed and r.get("rho_prev_alias", True) is not False:
            continue
        if not rho_fixed and r.get("rho_prev_alias", True) is False:
            continue
        return r
    return None


def fig_main(art, mm, repro, out):
    mm_t = np.array([d["t_star"] for d in mm["data"]])
    mm_z = np.array([d["X_star"] for d in mm["data"]])
    arms = [
        ("tensorlbm_koerner", 32, False, "C0", "-",
         "TensorLBM Körner (a=32)"),
        ("oss_port", 32, True, "C2", "--",
         "OSS port, rho_prev alias fixed (a=32)"),
        ("oss_port", 32, False, "C3", "-.",
         "OSS port as shipped (a=32)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.3))
    ax = axes[0]
    ax.plot(mm_t, mm_z, "ko-", ms=4, lw=1.2, label="Martin & Moyce 1952 (digitised)")
    for arm, a, fixed, c, ls, lab in arms:
        r = pick(art["runs"], arm, a, fixed)
        if not r:
            continue
        T = [s["T"] for s in r["series"]]
        Z = [s["Z"] for s in r["series"]]
        ax.plot(T, Z, ls, color=c, lw=1.7,
                label=f"{lab}: max {r['vs_mm']['max_abs_rel_err_pct']:.1f} %")
    ax.axvspan(1.5, 4.0, color="0.85", alpha=0.6, lw=0)
    ax.text(2.4, 1.35, "pre-declared window\nexecuted only to T=1.5\n(audit: T=3.75, same max err)",
            fontsize=7, ha="center", color="0.25")
    ax.set_xlabel(r"$T=t\sqrt{g/h_0}$")
    ax.set_ylabel(r"$Z=x_{front}/a$")
    ax.set_title("Dam-break front position, author artifact")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper left")

    ax = axes[1]
    for arm, a, fixed, c, ls, lab in arms:
        r = pick(art["runs"], arm, a, fixed)
        if not r or "vs_mm" not in r:
            continue
        pts = r["vs_mm"]["points"]
        ax.plot([p["T"] for p in pts], [p["rel_err_pct"] for p in pts],
                ls, color=c, marker="o", ms=3, lw=1.4,
                label=f"{lab.split(' (')[0]}: max {r['vs_mm']['max_abs_rel_err_pct']:.1f} %")
    ax.axhspan(-15, 15, color="tab:green", alpha=0.10, lw=0)
    ax.axhline(15, color="r", ls=":", lw=1.2, label="15 % gate")
    ax.axhline(-15, color="r", ls=":", lw=1.2)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel(r"$T$")
    ax.set_ylabel("relative error vs M&M [%]")
    ax.set_title("Error inside the gate band (audit re-ran each point)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_diag(art, repro, out):
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.1))
    ax = axes[0]
    ltb = [r for r in art["runs"] if r["arm"] == "tensorlbm_koerner"]
    lo = [r for r in art["runs"] if r["arm"] == "oss_port"
          and r.get("rho_prev_alias", True) is not False
          and r.get("interface_rho", "oss") == "oss"]
    a16 = [pick(art["runs"], "tensorlbm_koerner", 16)]
    for grp, lab, c in ((ltb, "TensorLBM arm", "C0"), (lo, "OSS port as shipped", "C3")):
        if not grp:
            continue
        ax.plot([r["a"] for r in grp],
                [abs(r.get("inventory_drift_max_pct", np.nan)) for r in grp],
                "o-", color=c, label=f"{lab} (inventory drift)")
    ax.set_yscale("log")
    ax.set_xlabel("a (cells per column width)")
    ax.set_ylabel("max |liquid-inventory drift| [%]")
    ax.axhline(1.0, color="r", ls=":", lw=1.2, label="pre-declared 1 % ledger gate")
    ax.set_title("Mass ledger: TensorLBM keeps it, OSS port does not")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    _ = a16

    ax = axes[1]
    labels, vals, cols = [], [], []
    for r in repro:
        if r.get("status") != "ok":
            continue
        lab = r.get("label", "")
        if r.get("a") != 32:
            continue
        if r.get("arm") == "oss_port" and r.get("rho_prev_alias", True) is not False:
            continue
        if r.get("arm") == "tensorlbm_koerner":
            key = "TensorLBM tmax=" + ("1.5" if r.get("steps") == 1698 else "4.0")
            cols.append("C0")
        else:
            key = "OSS port rho_prev fixed"
            cols.append("C2")
        labels.append(key)
        vals.append(r["vs_mm"]["max_abs_rel_err_pct"])
    idx = np.arange(len(labels))
    ax.bar(idx, [float(v) for v in vals], color=cols, width=0.55)
    for i, v in zip(idx, vals):
        ax.text(i, float(v) + 0.4, f"{float(v):.2f}%", ha="center", fontsize=8)
    ax.axhline(15, color="r", ls=":", lw=1.2, label="15 % gate")
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("max |rel err| vs M&M [%]")
    ax.set_title("Auditor independent re-run (this mission)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


if __name__ == "__main__":
    art, mm, repro = load()
    print(fig_main(art, mm, repro, HERE / "fig_m18218_front_vs_mm.png"))
    print(fig_diag(art, repro, HERE / "fig_m18218_diagnostics.png"))