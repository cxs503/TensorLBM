#!/usr/bin/env python3
"""P187-W1-N08 出图：Nu(t) 历程 / 堵塞律 / 窗长律 / 可分离性 / 场量云图。

所有数据取自 ``runs/`` 与 ``artifacts/p187_w1_n08/*.json``（零手抄）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.confined_cylinder_law import MISSION_WINDOWS, STATIONARY_WINDOWS  # noqa: E402

RUNS = ROOT / "runs" / "p187_w1_n08"
ART = ROOT / "artifacts" / "p187_w1_n08"
FIG = ART / "figures"


def _load(runs_dir: Path, status_ok=("done",)) -> dict:
    out = {}
    for p in sorted(runs_dir.glob("hd*_D*.json")):
        r = json.loads(p.read_text())
        if r.get("status") not in status_ok:
            continue
        out[(round(float(r["geometry"]["h_over_d"]), 4), int(round(r["geometry"]["D_eff"])))] = r
    return out


def fig_history(runs: dict, metric: str, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), sharey=False)
    key = "nu_global" if metric == "balance" else "nu_local"
    label = "Nu from inlet-outlet enthalpy balance" if metric == "balance" else "Nu from wall heat-flux integral"
    hds = sorted({k[0] for k in runs if k[1] == max(kk[1] for kk in runs)})
    fine_d = max(k[1] for k in runs)
    cmap = plt.get_cmap("viridis")
    for i, hd in enumerate(hds):
        r = runs[(hd, fine_d)]
        t = np.asarray(r["times"], float)
        s = np.asarray(r[key], float)
        axes[0].plot(t, s, lw=0.9, color=cmap(i / max(len(hds) - 1, 1)), label=f"H/D={hd:g}")
    for w in MISSION_WINDOWS:
        axes[0].axvspan(w[0], w[1], color="grey", alpha=0.06)
    axes[0].set_xlabel("t  [D/U]")
    axes[0].set_ylabel(label)
    axes[0].set_title(f"(a) Nu(t) time history, grid D_eff={fine_d}")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(alpha=0.3)

    # (b) 起转瞬变衰减：Nu(t)-Nu(window tail) vs t
    for i, hd in enumerate(hds):
        r = runs[(hd, fine_d)]
        t = np.asarray(r["times"], float)
        s = np.asarray(r[key], float)
        m = (t >= 60) & (t < 105)
        ref = s[m].mean()
        axes[1].semilogx(t[1:], (s[1:] - ref) / ref * 100.0, lw=0.9,
                         color=cmap(i / max(len(hds) - 1, 1)), label=f"H/D={hd:g}")
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set_xlabel("t  [D/U]  (log)")
    axes[1].set_ylabel("(Nu - Nu_ref)/Nu_ref  [%]")
    axes[1].set_title("(b) burn-in transient vs late-window reference")
    axes[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_blockage(summary: dict, wt: dict, path: Path):
    crit = summary["criteria"]
    fine_d = summary["grid_levels"]
    rows = wt["primary"]
    hds = sorted({r["h_over_d"] for r in rows})
    # 固定窗长=最长平稳窗的均值随 H/D
    taus = sorted({r["tau"] for r in rows})
    tau_sel = max(taus)
    ys = [np.mean([r["nu"] for r in rows if r["h_over_d"] == hd and abs(r["tau"] - tau_sel) < 1e-9]) for hd in hds]
    p = crit["C1"]["multiplicative"]["params"]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    x = np.array([1.0 / hd for hd in hds])
    ax.plot(x, ys, "o", ms=6, label=f"LBM (tau={tau_sel:g} D/U)")
    xx = np.linspace(0, x.max() * 1.05, 200)
    tt = 1.0 + p["b"] * tau_sel ** (-p["beta"])
    ax.plot(xx, p["nu_inf"] * (1.0 + p["a"] * xx ** p["m"]) * tt, "-",
            label=f"fit: Nu_inf={p['nu_inf']:.3f}, a={p['a']:.3f}, m={p['m']:.3f}")
    cb = summary["nu_churchill_bernstein"]
    ax.axhline(cb, color="crimson", ls="--", lw=1.2, label=f"Churchill-Bernstein {cb:.3f}")
    base = crit["C2"]["baseline_widest_domain"]
    ax.plot([1 / base["h_over_d"]], [base["nu"]], "s", color="darkorange", ms=7,
            label=f"naive baseline (widest domain), dev {base['deviation_pct']:.1f}%")
    ax.plot([0], [crit["C2"]["nu_inf_extrapolated"]], "*", color="green", ms=13,
            label=f"extrapolated Nu_inf {crit['C2']['nu_inf_extrapolated']:.3f} "
                  f"(dev {crit['C2']['deviation_pct']:.1f}%)")
    ax.set_xlabel("D/H  (0 = unconfined)")
    ax.set_ylabel("Nu")
    ax.set_title("Blockage law f(H/D)=1+a(D/H)^m and zero-blockage extrapolation")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_window(summary: dict, wt: dict, path: Path):
    crit = summary["criteria"]
    rows = wt["primary"] + wt["mission"]
    hds = sorted({r["h_over_d"] for r in rows})
    taus = sorted({r["tau"] for r in rows})
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    cmap = plt.get_cmap("viridis")
    for i, hd in enumerate(hds):
        rr = [r for r in rows if r["h_over_d"] == hd]
        by_tau = {}
        for r in rr:
            by_tau.setdefault(r["tau"], []).append(r["nu"])
        taus_i = sorted(by_tau)
        ax.plot(taus_i, [np.mean(by_tau[t]) for t in taus_i], "o-", ms=4, lw=1,
                color=cmap(i / max(len(hds) - 1, 1)), label=f"H/D={hd:g}")
    p = crit["C1"]["multiplicative"]["params"]
    tt = np.linspace(min(taus), max(taus), 200)
    for i, hd in enumerate(hds):
        ax.plot(tt, p["nu_inf"] * (1 + p["a"] * hd ** (-p["m"])) * (1 + p["b"] * tt ** (-p["beta"])),
                ":", lw=1, color=cmap(i / max(len(hds) - 1, 1)))
    ax.set_xlabel("statistical window length tau  [D/U]")
    ax.set_ylabel("window-averaged Nu")
    ax.set_title("Window-truncation law t(tau)=1+b tau^-beta (dots: per-H/D means, dotted: fit)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_separability(summary: dict, wt: dict, path: Path):
    crit = summary["criteria"]
    p = crit["C1"]["multiplicative"]["params"]
    rows = wt["primary"] + wt["mission"]
    obsv = np.array([r["nu"] for r in rows])
    pred = np.array([
        p["nu_inf"] * (1 + p["a"] * r["h_over_d"] ** (-p["m"])) * (1 + p["b"] * r["tau"] ** (-p["beta"]))
        for r in rows
    ])
    rel = (pred / obsv - 1.0) * 100.0
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    axes[0].plot(obsv, pred, "o", ms=4)
    lo, hi = min(obsv.min(), pred.min()), max(obsv.max(), pred.max())
    axes[0].plot([lo, hi], [lo, hi], "k--", lw=1)
    axes[0].set_xlabel("measured Nu")
    axes[0].set_ylabel("Nu = Nu_inf f(H/D) t(tau)")
    axes[0].set_title(f"(a) separability: max |resid|={crit['C1']['law_max_abs_residual_pct']:.2f}%")
    axes[0].grid(alpha=0.3)
    hds = sorted({r["h_over_d"] for r in rows})
    taus = sorted({r["tau"] for r in rows})
    grid = np.full((len(hds), len(taus)), np.nan)
    for r, e in zip(rows, rel):
        i = hds.index(r["h_over_d"])
        j = taus.index(r["tau"])
        grid[i, j] = e
    im = axes[1].imshow(grid, aspect="auto", cmap="coolwarm",
                        vmin=-2, vmax=2, origin="lower")
    axes[1].set_xticks(range(len(taus)), [f"{t:g}" for t in taus])
    axes[1].set_yticks(range(len(hds)), [f"{h:g}" for h in hds])
    axes[1].set_xlabel("tau [D/U]")
    axes[1].set_ylabel("H/D")
    axes[1].set_title("(b) relative residual [%] (blank = window not measured)")
    fig.colorbar(im, ax=axes[1], label="%")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_field(path_npz: Path, out: Path, nu_balance: float, nu_cb: float):
    d = np.load(path_npz)
    T, ux, uy, solid = d["T"], d["ux"], d["uy"], d["solid"]
    ny, nx = T.shape
    speed = np.sqrt(ux ** 2 + uy ** 2)
    Tm = np.where(solid, np.nan, T)
    Sm = np.where(solid, np.nan, speed)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True)
    im0 = axes[0].imshow(Tm, origin="lower", cmap="inferno", aspect="equal")
    axes[0].set_title(f"(a) temperature field  (Nu_balance={nu_balance:.3f}, C-B={nu_cb:.3f})")
    fig.colorbar(im0, ax=axes[0], label="T*")
    im1 = axes[1].imshow(Sm, origin="lower", cmap="viridis", aspect="equal")
    axes[1].set_title("(b) velocity magnitude |u|")
    fig.colorbar(im1, ax=axes[1], label="|u|")
    axes[1].set_xlabel("x [lattice]")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metric", default="balance")
    ap.add_argument("--showcase", default=str(RUNS / "fig_hd08_D11_field.npz"))
    args = ap.parse_args(argv)
    FIG.mkdir(parents=True, exist_ok=True)
    summary = json.loads((ART / "criteria.json").read_text())
    wt = json.loads((ART / "window_table.json").read_text())
    runs = _load(RUNS)

    fig_history(runs, args.metric, FIG / "fig1_nu_history.png")
    fig_blockage(summary, wt, FIG / "fig2_blockage_law.png")
    fig_window(summary, wt, FIG / "fig3_window_law.png")
    fig_separability(summary, wt, FIG / "fig4_separability.png")
    sc = Path(args.showcase)
    if sc.exists():
        j = sc.with_name(sc.name.replace("_field.npz", ".json"))
        nb = nu_cb = float("nan")
        if j.exists():
            rr = json.loads(j.read_text())
            nb = rr["final"]["nu_balance_mean_tail"]
            nu_cb = rr["nu_churchill_bernstein"]
        fig_field(sc, FIG / "fig5_field.png", nb, nu_cb)
    print("figures ->", sorted(p.name for p in FIG.glob("*.png")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())