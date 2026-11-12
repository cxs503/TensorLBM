"""Figures for CASE-RAYLEIGH-TAYLOR.

Reads only artifacts written by ``run_case.py`` (``artifacts/runs/*/history.csv``,
``artifacts/diagnostics/tip_velocity.csv``); nothing is retyped.  Panel (d) is a
short field snapshot produced here (documented as a 2.6-tau restart, so its time
stamp is not the end of the gated run).
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parents[2]
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_case import (  # noqa: E402
    CaseSpec,
    front_metrics,
    goncharov_velocity,
    init_rt_state,
    rt_step,
)
from tensorlbm.d2q9 import macroscopic  # noqa: E402

OUT = BENCH_DIR / "artifacts" / "figures"
RUNS = BENCH_DIR / "artifacts" / "runs"
DIAG = BENCH_DIR / "artifacts" / "diagnostics" / "tip_velocity.csv"
REFCASE = "A033_l064_g2e-4_tau0515"


def read_history(name: str) -> dict[str, list[float]]:
    rows = list(csv.DictReader((RUNS / name / "history.csv").open()))
    cols: dict[str, list[float]] = {k: [] for k in rows[0]}
    for r in rows:
        for k, v in r.items():
            cols[k].append(float(v))
    return cols


def fig_linear() -> Path:
    h = read_history(REFCASE)
    import json
    case = json.loads((RUNS / REFCASE / "case.json").read_text())
    lam = case["spec"]["lam"]
    a = [x * lam for x in h["a_mode_over_lam"]]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    lo, hi = case["sigma_fit_a_over_lam"][0] * lam, case["sigma_fit_a_over_lam"][1] * lam
    ax[0].semilogy(h["step"], a, ".", ms=2, color="#1f77b4", label=r"$a(t)$ (column moment)")
    sel = [(t, y) for t, y in zip(h["step"], a) if lo <= y <= hi]
    if sel:
        ts = [s[0] for s in sel]
        ax[0].semilogy(ts, [s[1] for s in sel], color="#d62728", lw=1.2, label="fit window")
    t0, a0 = h["step"][0], a[0]
    ts = [t0, h["step"][-1]]
    ax[0].semilogy(ts, [a0 * math.exp(case["sigma_sim"] * (t - t0)) for t in ts], "k--", lw=1,
                   label=rf"$\sigma_{{sim}}$={case['sigma_sim']:.4f}")
    ax[0].semilogy(ts, [a0 * math.exp(case["sigma_viscous"] * (t - t0)) for t in ts], ":", color="green",
                   label=rf"$\sigma_{{visc}}$={case['sigma_viscous']:.4f}")
    ax[0].set_xlabel("t [lattice steps]")
    ax[0].set_ylabel(r"$a/\lambda$  (x $\lambda$)")
    ax[0].set_title(f"linear regime, $A$={case['spec']['atwood']:.3f}, $\\lambda$={lam}")
    ax[0].legend(fontsize=7, loc="lower right")
    ax[0].grid(alpha=0.3)

    names, errs = [], []
    for d in sorted(RUNS.iterdir()):
        f = d / "case.json"
        if not f.exists():
            continue
        c = json.loads(f.read_text())
        names.append(c["name"].replace("A033_", "A1/3 ").replace("A050_", "A1/2 ")
                     .replace("_g2e-4", " g2e-4").replace("_g1e-4", " g1e-4"))
        errs.append(c["sigma_rel_error_vs_viscous_pct"])
    order = sorted(range(len(errs)), key=lambda i: errs[i])
    ax[1].barh([names[i] for i in order], [errs[i] for i in order], color="#2ca02c")
    ax[1].axvline(0, color="k", lw=0.8)
    for s in (-10, 10):
        ax[1].axvline(s, color="r", ls="--", lw=1)
    ax[1].set_xlabel("sigma_sim/sigma_visc - 1  [%]   (gate: dashed red, +/-10 %)")
    ax[1].set_title("linear growth rate, all suite cases")
    ax[1].tick_params(labelsize=7)
    ax[1].grid(alpha=0.3, axis="x")
    fig.tight_layout()
    p = OUT / "fig1_linear_growth.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_fronts() -> Path:
    h = read_history(REFCASE)
    d = list(csv.DictReader(DIAG.open()))
    tau = [float(r["tau"]) for r in d]
    v_int = [float(r["v_interface"]) for r in d]
    uyc = [float(r["uy_crest"]) for r in d]
    phi_hi = [float(r["phi_max"]) for r in d]
    case_spec = CaseSpec("ref", atwood=1 / 3, lam=64, g=2e-4, tau=0.515, tau_phi=0.501)
    ub = goncharov_velocity(1 / 3, 2e-4, 64, "b")
    us = goncharov_velocity(1 / 3, 2e-4, 64, "s")
    unit_t = math.sqrt(1 / 3 * 2e-4 / 64)     # tau per step
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    tt = [t * unit_t for t in h["step"]]
    ax[0].plot(tt, h["h_bubble"], color="#1f77b4", label=r"$h_b$ (bubble front)")
    ax[0].plot(tt, h["h_spike"], color="#d62728", lw=1, label=r"$h_s$ (spike front)")
    for ref, style, lab in ((ub, "--", r"$u_b^{Goncharov}$"), (us, ":", r"$u_s^{Goncharov}$")):
        ax[0].plot(tt, [ref * t for t in tt], style, color="k" if "b" in lab else "gray", lw=1,
                   label=lab + f" = {ref:.4f}")
    ax[0].axvspan(4, 6, color="orange", alpha=0.12, label="pre-declared window")
    ax[0].set_xlabel(r"$\tau = t\sqrt{Ag/\lambda}$")
    ax[0].set_ylabel("front distance from mean level [cells]")
    ax[0].set_ylim(0, 60)
    ax[0].set_title("fronts race ahead of the potential-flow plateau")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)

    ax2 = ax[1]
    ax2.plot(tau, [v * unit_t for v in v_int], color="#ff7f0e", label=r"$dh_b/dt$ (interface sweep)")
    ax2.plot(tau, uyc, color="#1f77b4", label=r"$u_y$ at bubble tip (fluid)")
    ax2.plot(tau, [-float(r["uy_trough"]) for r in d], color="#2ca02c", lw=1,
             label=r"$|u_y|$ at spike tip (fluid)")
    ax2.axhline(ub, color="k", ls="--", lw=1, label=f"Goncharov $u_b$ = {ub:.4f}")
    ax2.axhline(us, color="gray", ls=":", lw=1, label=f"Goncharov $u_s$ = {us:.4f}")
    ax2.axvspan(4, 6, color="orange", alpha=0.12)
    ax2b = ax2.twinx()
    ax2b.plot(tau, phi_hi, color="#9467bd", lw=1, alpha=0.7, label=r"$\max|\phi|$ (right)")
    ax2b.set_ylabel(r"$\max|\phi|$", color="#9467bd")
    ax2b.tick_params(axis="y", colors="#9467bd")
    ax2b.axhline(1.0, color="#9467bd", lw=0.6, ls="-.")
    ax2.set_xlabel(r"$\tau$")
    ax2.set_ylabel("velocity [lattice units]")
    ax2.set_title("root cause: interface sweeps $\\sim$7$\\times$ faster than the fluid")
    ax2.legend(fontsize=6.5, loc="upper left")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    p = OUT / "fig2_fronts_and_tip_velocity.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_field(tau_end: float = 2.6) -> Path:
    spec = CaseSpec("snap", atwood=1 / 3, lam=64, g=2e-4, tau=0.515, tau_phi=0.501, tau_end=tau_end)
    n = int(round(tau_end * math.sqrt(64 / (1 / 3 * 2e-4))))
    cfg = spec.cfg(n)
    torch.manual_seed(0)
    f, g, phi = init_rt_state(cfg)
    for _ in range(n):
        f, g = rt_step(f, g, cfg, phi=phi)
        phi = g.sum(0)
    _rho, ux, uy = macroscopic(f)
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 5.2))
    im0 = ax[0].imshow(phi.numpy(), origin="lower", cmap="RdBu_r", vmin=-1.6, vmax=1.6, aspect="auto")
    ax[0].set_title(rf"$\phi$ at $\tau$={tau_end} ($\max|\phi|$={float(phi.abs().max()):.2f})")
    fig.colorbar(im0, ax=ax[0], fraction=0.046)
    im1 = ax[1].imshow(uy.numpy(), origin="lower", cmap="viridis", aspect="auto")
    ax[1].set_title(r"$u_y$ [lattice]")
    fig.colorbar(im1, ax=ax[1], fraction=0.046)
    for a in ax:
        a.set_xlabel("x [cells]")
        a.set_ylabel("y [cells]")
    fig.tight_layout()
    p = OUT / "fig3_field_snapshot.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in (fig_linear, fig_fronts, fig_field):
        print("[fig]", fn())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())