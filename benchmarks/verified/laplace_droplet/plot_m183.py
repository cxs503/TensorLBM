#!/usr/bin/env python
"""Figures for the m183 laplace_droplet sweep report (all data read from JSON)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ART = REPO / "artifacts" / "m183_laplace_droplet"
RUNS = REPO / "runs" / "m183_laplace"
RES = json.loads((ART / "sweep_result.json").read_text(encoding="utf-8"))

plt.rcParams.update({"figure.dpi": 140, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "savefig.bbox": "tight"})


def fig_fields():
    fig = plt.figure(figsize=(11, 3.6))
    # (a) 2D field, largest droplet
    ax = fig.add_subplot(1, 3, 1)
    d2 = RUNS / "2d_R40"
    rho2 = np.load(d2 / "snapshot.npy")
    im = ax.imshow(rho2, origin="lower", cmap="viridis")
    ax.contour(rho2, levels=[0.5 * (rho2.max() + rho2.min())], colors="w", linewidths=0.7)
    ax.set_title("(a) 2D D2Q9  $\\rho$  (R=40, L=160)")
    ax.set_xlabel("x [lu]"); ax.set_ylabel("y [lu]"); ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046)
    # (b) 3D centre slice -- largest 3D run that has a snapshot
    ax = fig.add_subplot(1, 3, 2)
    cands = sorted([d for d in RUNS.glob("3d_R*") if (d / "snapshot.npy").exists()],
                   key=lambda d: float(d.name.split("_R")[1]))
    d3 = cands[-1] if cands else None
    if d3 is None:
        ax.set_visible(False)
        ax = fig.add_subplot(1, 3, 2)
        rho3 = np.zeros((2, 2))
    else:
        rho3 = np.load(d3 / "snapshot.npy")
    im = ax.imshow(rho3, origin="lower", cmap="viridis")
    ax.contour(rho3, levels=[0.5 * (rho3.max() + rho3.min())], colors="w", linewidths=0.7)
    ax.set_title(f"(b) 3D D3Q19  $\\rho$  centre slice ({d3.name if d3 else 'n/a'})")
    ax.set_xlabel("x [lu]"); ax.set_ylabel("y [lu]"); ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046)
    # (c) radial profiles (all runs)
    ax = fig.add_subplot(1, 3, 3)
    for d in sorted(RUNS.glob("*_R*")):
        f = d / "profile.json"
        if not f.exists():
            continue
        tag = "2D" if d.name.startswith("2d") else "3D"
        p = json.loads(f.read_text())
        ax.plot(p["r"], p["rho"], "C0" if tag == "2D" else "C1", lw=1.0,
                label=f"{tag} R={d.name.split('_R')[1]}")
    ax.set_xlim(0, 60); ax.set_xlabel("r [lu]"); ax.set_ylabel(r"$\rho(r)$")
    ax.set_title("(c) shell-averaged profiles")
    ax.legend(fontsize=6, loc="upper right")
    ax.axhline(1.957, ls=":", c="k", lw=0.7)
    ax.axhline(0.1596, ls=":", c="k", lw=0.7)
    fig.savefig(ART / "fig1_fields_profiles.png")
    plt.close(fig)


def fig_laplace():
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, tag, lbl in ((axs[0], "2d", "2D D2Q9  $\\Delta p=\\sigma/R$"),
                         (axs[1], "3d", "3D D3Q19  $\\Delta p=2\\sigma/R$")):
        d = RES["dims"].get(tag, {})
        if "per_radius" not in d:
            ax.set_visible(False); continue
        R = np.array([p["R_eq"] for p in d["per_radius"]])
        dp = np.array([p["dp"] for p in d["per_radius"]])
        y = dp if tag == "2d" else dp / 2.0
        x = 1.0 / R
        xx = np.linspace(0, x.max() * 1.08, 50)
        ax.plot(x, y, "o", ms=6, label="measured (steady state)")
        ax.plot(xx, d["sigma_eff_fit"] * xx, "-", lw=1.2,
                label=f"origin fit  $\\sigma$={d['sigma_eff_fit']:.5f}")
        ax.plot(xx, d["slope_free"] * xx + d["intercept_free"], "--", lw=1.0,
                label=(f"free fit  a={d['slope_free']:.5f}, b={d['intercept_free']:.2e}"))
        for p in d["per_radius"]:
            ax.annotate(f"R={p['R_init']:.0f}", (1 / p["R_eq"], p["dp"] / (2 if tag == "3d" else 1)),
                        textcoords="offset points", xytext=(6, -10), fontsize=7)
        try:  # overlay the stored verified baseline (result.json) for trend comparison
            stored = json.loads((HERE / "result.json").read_text())
            bp = stored["dims"][tag]["per_radius"]
            ax.plot([1 / q["R_eq"] for q in bp],
                    [q["dp"] / (2 if tag == "3d" else 1) for q in bp],
                    "s", mfc="none", ms=7, label="stored verified baseline")
            ax.legend(fontsize=7, loc="upper left")
        except Exception:
            pass
        ax.set_xlabel("$1/R_{eq}$  [lu$^{-1}$]")
        ax.set_ylabel("$\\Delta p$" if tag == "2d" else "$\\Delta p/2$")
        ax.set_title(f"{lbl}\n$R^2$={d['R2']:.6f}, max$|\\sigma_i-\\sigma_{{fit}}|$="
                     f"{d['max_dev_sigma_pct']:.2f}%", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
        ax.set_xlim(0, x.max() * 1.1); ax.set_ylim(0, y.max() * 1.1)
        # residual panel text
        res_txt = ", ".join(f"{p['dev_sigma_pct']:+.2f}%" for p in d["per_radius"])
        ax.text(0.98, 0.03, f"per-radius dev: {res_txt}", transform=ax.transAxes,
                ha="right", fontsize=6.5)
    fig.savefig(ART / "fig2_laplace_fit.png")
    plt.close(fig)


def fig_repro():
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axs[0]
    width = 0.35
    labels, cur, sto = [], [], []
    for tag in ("2d", "3d"):
        d = RES["dims"].get(tag, {})
        for row in d.get("vs_stored_result_json", {}).get("matched_radii", []):
            labels.append(f"{tag.upper()} R={row['R_init']:.0f}")
            cur.append(row["sigma_i_this_run"]); sto.append(row["sigma_i_stored"])
    idx = np.arange(len(labels))
    ax.bar(idx - width / 2, sto, width, label="stored result.json")
    ax.bar(idx + width / 2, cur, width, label="this independent run")
    ax.set_xticks(idx); ax.set_xticklabels(labels, rotation=30, fontsize=7)
    ax.set_ylabel("$\\sigma_i$  [lu]"); ax.set_ylim(0.054, 0.058)
    ax.set_title("(a) per-radius $\\sigma_i$: rerun vs stored baseline")
    ax.legend(fontsize=7)
    ax = axs[1]
    devs = [abs(c - s) / s * 100 for c, s in zip(cur, sto)]
    ax.bar(idx, devs, 0.5, color="C2")
    ax.axhline(3.0, ls="--", c="r", lw=1.0, label="3% gate")
    for i, v in enumerate(devs):
        ax.annotate(f"{v:.2f}%", (i, v), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=7)
    ax.set_xticks(idx); ax.set_xticklabels(labels, rotation=30, fontsize=7)
    ax.set_ylabel("$|\\sigma_i^{rerun}-\\sigma_i^{stored}|/\\sigma_i^{stored}$  [%]")
    ax.set_ylim(0, max(3.5, max(devs) * 1.4))
    ax.set_title("(b) reproduction error vs stored baseline")
    ax.legend(fontsize=7)
    fig.savefig(ART / "fig3_reproducibility.png")
    plt.close(fig)


def fig_mech():
    d = RES["dims"]["2d"].get("mechanical_route_2d")
    if not d or not d.get("per_radius"):
        return
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axs[0]
    for m in d["per_radius"]:
        ax.plot(m["_profile"]["r"], m["_profile"]["dP"],
                label=f"R={m['R_init']:.0f} ($\\sigma_{{mech}}$={m['sigma_mech']:.5f})")
    ax.set_xlim(0, 50); ax.set_xlabel("r [lu]")
    ax.set_ylabel("$P_{rr}-P_{tt}$  [lu]")
    ax.set_title("(a) mechanical-route integrand $\\int(P_{rr}-P_{tt})\\,dr$")
    ax.legend(fontsize=7)
    ax = axs[1]
    R = [m["R_init"] for m in d["per_radius"]]
    sm = [m["sigma_mech"] for m in d["per_radius"]]
    sl = [m["sigma_laplace_i"] for m in d["per_radius"]]
    x = np.arange(len(R)); w = 0.35
    ax.bar(x - w / 2, sl, w, label="Laplace route $\\Delta p\\,R_{eq}$")
    ax.bar(x + w / 2, sm, w, label="mechanical route $\\int(P_{rr}-P_{tt})dr$")
    for i, (a, b) in enumerate(zip(sl, sm)):
        ax.annotate(f"{abs(b-a)/a*100:.2f}%", (i, max(a, b)),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([f"R={r:.0f}" for r in R])
    ax.set_ylabel("$\\sigma$  [lu]"); ax.set_ylim(0.050, 0.060)
    ax.set_title("(b) two independent $\\sigma$ routes")
    ax.legend(fontsize=7)
    fig.savefig(ART / "fig4_mechanical_route.png")
    plt.close(fig)


def fig_conv():
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, tag in ((axs[0], "2d"), (axs[1], "3d")):
        hist = RES["dims"].get(tag, {}).get("history", {})
        for R, h in hist.items():
            st = [p["step"] for p in h]; dp = [p["dp"] for p in h]
            ax.plot(st, dp, "o-", ms=3, lw=1.0, label=f"R={R}")
        ax.set_xlabel("step"); ax.set_ylabel("$\\Delta p(t)$  [lu]")
        ax.set_title(f"{tag.upper()} pressure-jump convergence")
        ax.legend(fontsize=7)
    fig.savefig(ART / "fig5_convergence.png")
    plt.close(fig)


def fig_domain_and_cost():
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))
    ax = axs[0]
    dom = RES.get("domain_independence", {})
    if dom.get("runs"):
        import collections
        by_R = collections.defaultdict(list)
        for r in dom["runs"]:
            by_R[(r["dim"], r["R_init"])].append((r["domain_factor"], r["sigma_i"]))
        # baseline df=4 from the main sweep
        for tag, dim in (("2d", 2), ("3d", 3)):
            for p_ in RES["dims"].get(tag, {}).get("per_radius", []):
                by_R[(dim, float(p_["R_init"]))].append((4.0, p_["sigma_i"]))
        for (dim, R), pts in sorted(by_R.items()):
            pts.sort()
            xs = [q[0] for q in pts]; ys = [q[1] for q in pts]
            ax.plot(xs, ys, "o-", label=f"{dim}D R={R:.0f}")
        ax.set_xlabel("domain factor  $L/R$"); ax.set_ylabel("$\\sigma_i=\\Delta p\\,R_{eq}$  [lu]")
        ax.set_title("(a) periodic-box (domain) sensitivity of $\\sigma_i$")
        ax.legend(fontsize=7)
        mn = min(q[1] for pts in by_R.values() for q in pts)
        mx = max(q[1] for pts in by_R.values() for q in pts)
        ax.set_ylim(mn * 0.995, mx * 1.005)
    else:
        ax.set_visible(False)
    ax = axs[1]
    t = ART / "timing_probe.json"
    if t.exists():
        d = json.loads(t.read_text())
        labels = [r["label"] for r in d["rows"]]
        vals = [r["ms_per_step"] for r in d["rows"]]
        cols = ["C0" if r["dim"] == 2 else "C1" for r in d["rows"]]
        ax.barh(range(len(vals)), vals, color=cols)
        ax.set_yticks(range(len(vals))); ax.set_yticklabels(labels, fontsize=7)
        ax.set_xscale("log"); ax.set_xlabel("ms / step  (log, CPU eager)")
        ax.set_title(f"(b) cost per step; canonical serial full sweep $\\approx$"
                     f" {d['serial_total_s_at_6000_steps']/60:.0f} min")
        for i, v in enumerate(vals):
            ax.annotate(f"{v:.1f}", (v, i), textcoords="offset points", xytext=(3, -2), fontsize=6)
    ax.invert_yaxis()
    fig.savefig(ART / "fig6_domain_and_cost.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_fields(); fig_laplace(); fig_repro(); fig_mech(); fig_conv(); fig_domain_and_cost()
    for f in sorted(ART.glob("fig*.png")):
        print(f, f.stat().st_size)