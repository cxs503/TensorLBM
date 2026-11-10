#!/usr/bin/env python
"""P96W3-N04 图件生成：从 artifacts/p96w3_n04/*.json 直接出图（零手抄）。

产出 figures/p96w3_n04/*.png：
  fig1_sigma_ra.png     σ*(Ra) 扫描 + 零交点（Ra_c 测量）
  fig2_sigma_k.png      σ*(k) 抛物拟合（k_c 测量）
  fig3_nu_convergence.png  各分辨率 Nu(t) 收敛历史
  fig4_nu_ra.png        Nu–Ra：LBM vs de Vahl Davis / Clever–Busse 文献带
  fig5_gci.png          Nu vs 1/H 与 Richardson 外推（GCI）
  fig6_fields.png       RB 定常辊与 DHC 的 T 场 + 流线
  fig7_accuracy.png     C6 正弦波衰减的 α 相对误差 vs 网格（log-log）
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "p96w3_n04"
FIG = ROOT / "figures" / "p96w3_n04"
FIG.mkdir(parents=True, exist_ok=True)


def load(name: str) -> dict | None:
    p = ART / f"{name}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _runs(prefix: str) -> dict[str, dict]:
    out = {}
    for p in sorted(ART.glob(f"{prefix}*.json")):
        out[p.stem] = json.loads(p.read_text())
    return out


def fig_sigma_ra() -> str | None:
    d = load("c1_rac_ny48")
    if not d:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for key, style, label in (("coarse_step_0.05", "o-", "ΔRa = 0.05 Ra_c"),
                              ("fine_step_0.025", "s--", "ΔRa = 0.025 Ra_c")):
        blk = d[key]
        ra = np.array([a["ra"] for a in blk["arms"]], dtype=float)
        sg = np.array([a["sigma_star"] for a in blk["arms"]], dtype=float)
        ax.plot(ra, sg, style, label=label, ms=5)
        ax.axvline(blk["ra_c"], color="0.75", lw=0.8)
        ax.annotate(f"Ra_c={blk['ra_c']:.1f}", (blk["ra_c"], sg.min()),
                    rotation=90, fontsize=8, va="bottom", ha="right")
    ax.axvline(1707.762, color="crimson", ls=":", label="Chandrasekhar 1707.76")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xlabel("Ra")
    ax.set_ylabel(r"$\sigma^* = \sigma H^2/\alpha$")
    ax.set_title("C1 linear stability: measured growth rate vs Ra (ny=48, Γ=2)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    p = FIG / "fig1_sigma_ra.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return str(p)


def fig_sigma_k() -> str | None:
    d = load("c1_kc_ny48")
    if not d:
        return None
    k = np.array([a["kH"] for a in d["arms"]], dtype=float)
    sg = np.array([a["sigma_star"] for a in d["arms"]], dtype=float)
    a, b, c = d["parabola"]["a"], d["parabola"]["b"], d["parabola"]["c"]
    kk = np.linspace(k.min() * 0.95, k.max() * 1.05, 200)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(k, sg, "o", label="measured σ*(k)")
    ax.plot(kk, a * kk**2 + b * kk + c, "-", label="parabola fit")
    ax.axvline(d["k_c_measured"], color="C1", ls="--",
               label=f"k_c measured = {d['k_c_measured']:.4f}")
    ax.axvline(3.11632, color="crimson", ls=":", label="a_c = 3.11632")
    ax.set_xlabel("a = kH")
    ax.set_ylabel(r"$\sigma^*$")
    ax.set_title(f"C1 most-unstable wavenumber (Ra=1.2 Ra_c, ny=48)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = FIG / "fig2_sigma_k.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return str(p)


def fig_nu_convergence() -> str | None:
    runs = _runs("c3_") | _runs("c2_")
    if not runs:
        return None
    keys = [k for k in runs if runs[k].get("nu_history")]
    keys = keys[:8]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for k in keys:
        d = runs[k]
        h = d["nu_history"]
        n = len(h)
        steps = np.linspace(d["steps"] / n, d["steps"], n)
        label = f"{d['case']} Ra={d['ra']:g} ny={d['ny']}"
        ax.plot(steps / max(d["t_turn"], 1e-9), h, lw=1.2, label=label)
    ax.set_xlabel("t / t_turnover")
    ax.set_ylabel("Nu(t)")
    ax.set_title("Nu convergence histories (wall-flux estimator)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    p = FIG / "fig3_nu_convergence.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return str(p)


DE_VAHL = {1e3: 1.118, 1e4: 2.243, 1e5: 4.519, 1e6: 8.800}
BAND = {1e4: (2.10, 2.30), 1e5: (4.20, 4.80), 1e6: (7.60, 8.80)}


def fig_nu_ra() -> str | None:
    dhc = _runs("c3_")
    rolls = _runs("c2_")
    if not dhc and not rolls:
        return None
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for ra, nu in sorted(DE_VAHL.items()):
        ax.plot(ra, nu, "k*", ms=11, label="de Vahl Davis (1983)" if ra == 1e3 else None)
    for ra, (lo, hi) in sorted(BAND.items()):
        ax.fill_between([ra * 0.94, ra * 1.06], lo, hi, color="C2", alpha=0.18)
    for tag, rs in (("DHC", dhc), ("RB rolls", rolls)):
        if not rs:
            continue
        ra = np.array([rs[k]["ra"] for k in rs], dtype=float)
        nu = np.array([rs[k]["nu"] for k in rs], dtype=float)
        ny = np.array([rs[k]["ny"] for k in rs], dtype=float)
        order = np.lexsort((ny, ra))
        ax.plot(ra[order], nu[order], "o", ls="none", alpha=0.8,
                label=f"{tag} (this work)")
        for x, y, n in zip(ra, nu, ny):
            ax.annotate(f"{int(n)}", (x, y), fontsize=6, xytext=(3, 3),
                        textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Ra")
    ax.set_ylabel("Nu")
    ax.set_title("C2/C3 Nu–Ra vs literature (labels = ny)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = FIG / "fig4_nu_ra.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return str(p)


def fig_gci() -> str | None:
    dhc = _runs("c3_")
    rolls = _runs("c2_")
    if not dhc and not rolls:
        return None
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    series = []
    for tag, rs in (("DHC", dhc), ("RB rolls", rolls)):
        by_ra: dict[float, list[tuple[int, float]]] = {}
        for k, d in rs.items():
            by_ra.setdefault(float(d["ra"]), []).append((int(d["ny"]), float(d["nu"])))
        for ra, pts in sorted(by_ra.items()):
            pts.sort()
            if len(pts) < 2:
                continue
            series.append((tag, ra, pts))
    for tag, ra, pts in series:
        ny = np.array([p[0] for p in pts], dtype=float)
        nu = np.array([p[1] for p in pts], dtype=float)
        ax.plot(1.0 / ny, nu, "o-", label=f"{tag} Ra={ra:g}", ms=4)
    ax.set_xlabel("1/ny")
    ax.set_ylabel("Nu")
    ax.set_title("Grid refinement / Richardson extrapolation")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6)
    fig.tight_layout()
    p = FIG / "fig5_gci.png"
    fig.savefig(fig and p, dpi=140)
    plt.close(fig)
    return str(p)


def fig_accuracy() -> str | None:
    d = load("c6")
    if not d:
        return None
    rows = d["rows"]
    n = np.array([r["n"] for r in rows], dtype=float)
    err = np.array([abs(r["alpha_rel_error"]) for r in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.loglog(n, err, "o-", label="measured |Δα|/α")
    ax.loglog(n, err[0] * (n[0] / n) ** 2, "k--", label=r"$\propto N^{-2}$ (2nd order)")
    ax.loglog(n, err[0] * (n[0] / n) ** 4, "r:", label=r"$\propto N^{-4}$ (4th order)")
    ax.set_xlabel("N (sine-wave grid)")
    ax.set_ylabel(r"$|\alpha_{\rm eff}-\alpha|/\alpha$")
    ax.set_title("C6 thermal diffusivity inversion (C5/C6)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = FIG / "fig7_accuracy.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return str(p)


def main() -> int:
    made = {}
    for name, fn in (("fig1", fig_sigma_ra), ("fig2", fig_sigma_k),
                     ("fig3", fig_nu_convergence), ("fig4", fig_nu_ra),
                     ("fig5", fig_gci), ("fig7", fig_accuracy)):
        try:
            out = fn()
        except Exception as exc:  # pragma: no cover - 出图不应阻塞交付
            print(f"[fig {name}] failed: {exc}")
            out = None
        if out:
            made[name] = out
            print(f"[fig {name}] {out}")
    (ART / "figures.json").write_text(json.dumps(made, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())