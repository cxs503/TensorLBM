#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DE-X-CMB5-HYDRORR-STAB figures: depth curves, death-drift, dose ladder."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

M27044 = "/root/tp-agent/work/m27044/runs/cmb3"
MINE = "/root/tp-agent/work/m27100/runs"
OUT = "/root/tp-agent/work/m27100/development/audit"
EXP_DEPTH, EXP_MS_PC = 1.857434, 10.39446
DT = 1.2160714285714285e-2


def load(p):
    with open(p) as f:
        return json.load(f)


def curves(d):
    h = d["history"]
    cs = d["contact_step"]
    t = [r["time_ms"] - (cs or 0) * DT for r in h]
    dep = [r.get("depth_over_D", 0.0) for r in h]
    return t, dep


def main():
    refs = {n: load(f"{M27044}/{n}/result.json") for n in ("ctrl", "combo_hydro", "hydro_rr", "hydro_only")}
    mine = {}
    for n in ("cmb5_d075", "cmb5_d050", "cmb5_ul13"):
        p = f"{MINE}/{n}/result.json"
        if os.path.exists(p):
            mine[n] = load(p)

    # ---- fig 1: depth curves ----
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=150)
    styles = {"ctrl": ("k", "--", 1.2), "hydro_only": ("0.5", ":", 1.2),
              "hydro_rr": ("r", "-", 1.8), "combo_hydro": ("b", "-", 1.8),
              "cmb5_d075": ("g", "-", 2.0), "cmb5_d050": ("m", "-", 2.0),
              "cmb5_ul13": ("c", "-", 2.0)}
    labels = {"ctrl": "ctrl (BGK, no hydro)", "hydro_only": "hydro_only (died 22.4 ms pc)",
              "hydro_rr": "hydro_rr 1x dose (died 15.3 ms pc)",
              "combo_hydro": "combo_hydro champion 1.750541D (full window)",
              "cmb5_d075": "CMB5 0.75x dose", "cmb5_d050": "CMB5 0.50x dose",
              "cmb5_ul13": "CMB5 contingency u_limit=0.13"}
    for name, d in list(refs.items()) + list(mine.items()):
        t, dep = curves(d)
        c, ls, lw = styles[name]
        ax.plot(t, dep, color=c, ls=ls, lw=lw, label=labels[name])
    ax.plot([EXP_MS_PC], [EXP_DEPTH], "k*", ms=14, zorder=5)
    ax.annotate("experiment 1.857434D @ 10.394 ms pc", (EXP_MS_PC, EXP_DEPTH),
                xytext=(EXP_MS_PC + 1.5, EXP_DEPTH + 0.06), fontsize=8)
    ax.axhline(1.750541, color="b", lw=0.6, alpha=0.5)
    ax.set_xlabel("post-contact time (ms)")
    ax.set_ylabel("cavity depth (D)")
    ax.set_title("CMB5-HYDRORR-STAB: RR dose ladder on HYDRO base (cells=8, full window)")
    ax.legend(fontsize=7, loc="center right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_depth.png")
    plt.close(fig)

    # ---- fig 2: death-drift of hydro_rr ----
    d = refs["hydro_rr"]
    h = d["history"]
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
    ax.plot([r["step"] for r in h], [r["volume_drift"] for r in h], "r-", lw=1.2,
             label="volume_drift (hydro_rr 1x)")
    ax.axhline(1e-4, color="k", ls="--", lw=1.0, label="health gate 1e-4")
    ax.axvline(1305, color="r", ls=":", lw=1.0)
    ax.annotate("death: step 1305, drift 1.124e-4\nmach 0.106 (gate 0.3 not touched)",
                (1305, 1.124e-4), xytext=(900, 8e-5), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xlabel("step")
    ax.set_ylabel("volume_drift")
    ax.set_title("hydro_rr death cause: volume_drift gate (not mach)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_drift.png")
    plt.close(fig)

    # ---- fig 3: dose ladder ----
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=150)
    pts = []
    if "cmb5_d050" in mine:
        t, dep = curves(mine["cmb5_d050"])
        pts.append((0.50, max(dep), mine["cmb5_d050"]["stop_reason"]))
    if "cmb5_d075" in mine:
        t, dep = curves(mine["cmb5_d075"])
        pts.append((0.75, max(dep), mine["cmb5_d075"]["stop_reason"]))
    if "cmb5_ul13" in mine:
        t, dep = curves(mine["cmb5_ul13"])
        pts.append((0.75, max(dep), mine["cmb5_ul13"]["stop_reason"] + " (ul13)"))
    pts.append((1.00, 1.777132, "health_limit (died 15.3 ms pc)"))
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, "o-", color="g", lw=1.5, label="CMB5 dose ladder (HYDRO base)")
    for x, y, st in pts:
        ax.annotate(st, (x, y), xytext=(6, 4), textcoords="offset points", fontsize=7)
    ax.axhline(1.750541, color="b", ls="--", lw=1.0)
    ax.annotate("champion combo_hydro 1.750541D", (0.51, 1.756), fontsize=8, color="b")
    ax.axhline(EXP_DEPTH, color="k", ls=":", lw=1.0)
    ax.annotate("experiment 1.857434D", (0.51, 1.862), fontsize=8)
    ax.set_xlabel("RR dose (x)")
    ax.set_ylabel("peak cavity depth (D)")
    ax.set_title("RR dose response on HYDRO base (z_ref 24.5)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_dose.png")
    plt.close(fig)
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
