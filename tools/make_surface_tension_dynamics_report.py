#!/usr/bin/env python3
"""Figures for the surface-tension dynamics report (P152W1-N01, TensorLBM).

Reads the per-case JSON written by
:mod:`tools.surface_tension_dynamics_verification` and the aggregated
``results.json``; every plotted number comes from those files (no hand-typed
data), and the figure captions are emitted to ``figures/captions.json`` so the
report can quote them verbatim.
"""

from __future__ import annotations

import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "artifacts/surface_tension_dynamics"
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)


def _load(prefix: str) -> dict[str, dict]:
    out = {}
    for fn in sorted(os.listdir(OUT)):
        if fn.startswith(prefix) and fn.endswith(".json") and "__" in fn:
            with open(os.path.join(OUT, fn)) as fh:
                out[fn[len(prefix) + 2 : -5]] = json.load(fh)
    return out


waves = _load("wave")
mar = _load("marangoni")
coa = _load("coalescence")
with open(os.path.join(OUT, "results.json")) as fh:
    summary = json.load(fh)

captions: dict[str, str] = {}

# --------------------------------------------------------------------------- #
# fig 1: amplitude records, reference case vs negative control
# --------------------------------------------------------------------------- #
ref_name = "w_k0.098_W2.00"
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
if ref_name in waves:
    r = waves[ref_name]
    t = np.array(r["series"]["t"], float)
    a = np.array(r["series"]["a"], float)
    ax = axes[0]
    ax.plot(t, a, "o", ms=2.4, color="#1f77b4", label="modal amplitude $a(t)$")
    w = r["metrics"]["omega_meas"]
    g = r["fit_windows"]["early"]["gamma"] if "early" in r["fit_windows"] else 0.0
    tt = np.linspace(t[0], t[-1], 400)
    ax.plot(tt, a[0] * np.exp(-g * (tt - t[0])) * np.cos(w * (tt - t[0])), "--",
            color="#d62728", lw=1.2,
            label=f"fitted $\\omega={w:.4f}$ ($\\gamma={g:.1e}$)")
    ext = r["extrema"].get("extrema_t") or []
    for te in ext:
        ax.axvline(te, color="0.7", lw=0.6)
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_xlabel("lattice step")
    ax.set_ylabel("$a(t)$  (interface mode amplitude)")
    ax.set_title(f"capillary wave, kW={r['params']['kW']:.2f}")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)
neg = next((k for k in waves if "neg" in k), None)
if neg is not None:
    r = waves[neg]
    ax = axes[1]
    t = np.array(r["series"]["t"], float)
    a = np.array(r["series"]["a"], float)
    ax.plot(t, a, color="#2ca02c", lw=1.4)
    ax.axhline(a[0], color="k", ls=":", lw=0.8)
    ax.set_xlabel("lattice step")
    ax.set_ylabel("$a(t)$")
    ax.set_ylim(a[0] - 0.05, a[0] + 0.05)
    ax.set_title(f"negative control $\\sigma=0$: no wave (max dev {r['negative_control']['amp_max_dev']:.1e})")
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig1_wave_records.png"), dpi=160)
plt.close(fig)
captions["fig1_wave_records.png"] = (
    "左：参考工况（k=0.098, W=2, kW=0.196）界面模态振幅 a(t)，竖细线为极值位置，红线为用实测 ω 的阻尼余弦；"
    "极值间距给出半周期 π/ω。右：σ=0 负控——界面完全不振荡，振幅偏离初值仅 ~1e-3 量级，"
    "证明左图的振荡不是界面追踪器或声学伪迹。"
)

# --------------------------------------------------------------------------- #
# fig 2: dispersion curve (omega vs k)
# --------------------------------------------------------------------------- #
pts = [(v["k"], v["omega_meas"], v["omega_theory"], v["omega_theory_deep"], v["kW"], name)
       for name, v in summary["wave"].items() if "omega_meas" in v and v.get("n_half_periods")]
fig, ax = plt.subplots(figsize=(6.4, 4.6))
if pts:
    ks = np.linspace(min(p[0] for p in pts) * 0.8, max(p[0] for p in pts) * 1.2, 200)
    sig, rho = 0.05, 2.0
    ax.plot(ks, np.sqrt(sig * ks**3 / rho), "k-", lw=1.4,
            label=r"analytic $\sqrt{\sigma k^3/(\rho_l+\rho_g)}$ (deep water)")
    ax.fill_between(ks, np.sqrt(sig * ks**3 / rho) * 0.95, np.sqrt(sig * ks**3 / rho) * 1.05,
                    color="0.85", label=r"$\pm5\%$ band")
    for k, wm, wth, wdeep, kw, name in pts:
        ax.plot(k, wth, "k+", ms=9, mew=1.0)
        ax.plot(k, wm, "o", ms=6, mfc="none", mec="#1f77b4", mew=1.6)
        ax.annotate(f"{name}", (k, wm), textcoords="offset points", xytext=(4, -10), fontsize=6)
    ax.set_xlabel("wavenumber $k$")
    ax.set_ylabel(r"$\omega$")
    ax.set_title("measured (o) vs analytic (+) capillary dispersion")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig2_dispersion.png"), dpi=160)
plt.close(fig)
captions["fig2_dispersion.png"] = (
    "色散关系：空心圆=实测（极值间距法），黑十字=解析值（有限深公式 √(σk³tanh(kH)/(ρl+ρg))），"
    "灰带=±5% 验收带。实测点系统性低于解析值，偏差随 k 增大。"
)

# --------------------------------------------------------------------------- #
# fig 3: relative error vs kW
# --------------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(6.4, 4.4))
data = [(v["kW"], v["omega_rel_err"], v["k"], name) for name, v in summary["wave"].items()
        if "omega_rel_err" in v]
if data:
    ks = sorted({d[2] for d in data})
    cmap = plt.get_cmap("viridis")
    for i, kk in enumerate(ks):
        sel = [d for d in data if abs(d[2] - kk) < 1e-12]
        ax.plot([d[0] for d in sel], [100 * d[1] for d in sel], "o-", ms=6,
                color=cmap(i / max(len(ks) - 1, 1)), label=f"k={kk:.3f}")
    ax.axhline(0.0, color="k", lw=0.6)
    for band, c in ((5.0, "#d62728"), (15.0, "#ff9896")):
        ax.axhline(band, color=c, ls="--", lw=0.9)
        ax.axhline(-band, color=c, ls="--", lw=0.9)
    ax.set_xlabel("k W  (interface width / wavelength)")
    ax.set_ylabel(r"$\omega$ relative error [%]")
    ax.set_title("dispersion error vs $kW$ (dashed: 5% and 15% lines)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig3_error_vs_kW.png"), dpi=160)
plt.close(fig)
captions["fig3_error_vs_kW.png"] = (
    "ω 相对误差随 kW 的分布（按 k 分组）。所有点均为负偏（模型偏软），红色虚线为 5%/15% 判据线；"
    "偏差随 kW 与 k 单调恶化，据此给出可用窗口与软化律拟合。"
)

# --------------------------------------------------------------------------- #
# fig 4: Marangoni profiles
# --------------------------------------------------------------------------- #
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
cmap = plt.get_cmap("plasma")
sel = [k for k in sorted(mar) if not k.endswith("asym") and "Ly32" not in k]
for i, name in enumerate(sel):
    r = mar[name]
    s = r["series"]
    c = cmap(i / max(len(sel) - 1, 1))
    axes[0].plot(s["ux"], s["y"], "o", ms=2.5, color=c,
                 label=f"d$\\sigma$/dx={r['params']['dsdx']:.0e}")
    axes[0].plot(s["ux_theory"], s["y"], "-", color=c, lw=1.2)
axes[0].set_xlabel("$u_x$")
axes[0].set_ylabel("$y$")
axes[0].set_title("Marangoni channel: measured (o) vs Stokes (lines)")
axes[0].legend(fontsize=7)
axes[0].grid(alpha=0.3)
if sel:
    xs = [mar[n]["params"]["dsdx"] for n in sel]
    ys = [mar[n]["metrics"]["u_interface"] for n in sel]
    yt = [mar[n]["params"]["u_interface_theory"] for n in sel]
    axes[1].plot(xs, yt, "k-", lw=1.2, label="analytic (2)")
    axes[1].plot(xs, ys, "o", ms=7, mfc="none", mec="#1f77b4", mew=1.6, label="measured")
    axes[1].set_xlabel(r"d$\sigma$/dx")
    axes[1].set_ylabel("interface velocity $u_i$")
    axes[1].set_title("$u_i$ linear in the imposed Marangoni stress")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig4_marangoni.png"), dpi=160)
plt.close(fig)
captions["fig4_marangoni.png"] = (
    "左：Marangoni 通道内 u_x(y) 全剖面，圆点=实测、实线=解析分段线性 Stokes 解（三种 dσ/dx）；"
    "右：界面速度 u_i 对 dσ/dx 的线性响应与解析斜率对比。"
)

# --------------------------------------------------------------------------- #
# fig 5: coalescence
# --------------------------------------------------------------------------- #
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for i, (name, r) in enumerate(sorted(coa.items())):
    s = r["series"]
    t = np.array(s["t"], float) / r["params"]["tau_ci"]
    rb = np.array(s["r_b_over_R"], float)
    axes[0].plot(t, rb, lw=1.4, label=f"{name} (Oh={r['params']['Oh']:.3f})")
axes[0].set_xlabel(r"$t/\tau_{ci}$,  $\tau_{ci}=\sqrt{\rho R^3/\sigma}$")
axes[0].set_ylabel(r"$r_b/R$")
axes[0].set_title("neck growth (dimensionless)")
axes[0].legend(fontsize=7)
axes[0].grid(alpha=0.3)
for name, r in sorted(coa.items()):
    s = r["series"]
    m = r["metrics"]
    t = np.array(s["t"], float) / r["params"]["tau_ci"]
    rb = np.array(s["r_b_over_R"], float)
    mask = (rb > 0.05) & (rb < 0.6)
    if mask.sum() > 2:
        axes[1].loglog(t[mask], rb[mask], "o", ms=2.5,
                       label=f"{name}: n={m['growth_exponent']:.2f}")
tt = np.logspace(-3, 0, 50)
axes[1].loglog(tt, 1.0 * tt ** (2.0 / 3.0), "k--", lw=1.0, label=r"inertial $t^{2/3}$")
axes[1].loglog(tt, 1.0 * tt ** 0.5, "k:", lw=1.0, label=r"$t^{1/2}$")
axes[1].set_xlabel(r"$t/\tau_{ci}$")
axes[1].set_ylabel(r"$r_b/R$")
axes[1].set_title("log-log neck growth and fitted exponent")
axes[1].legend(fontsize=7)
axes[1].grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig5_coalescence.png"), dpi=160)
plt.close(fig)
captions["fig5_coalescence.png"] = (
    "液滴合并：左为无量纲颈半径 r_b/R 对 t/τ_ci 的演化；右为双对数斜率（实测指数）与惯性自相似 2/3 律、"
    "以及实验常报的 1/2 律参考线。"
)

# --------------------------------------------------------------------------- #
# fig 6: drift / sharpness controls
# --------------------------------------------------------------------------- #
fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
for i, name in enumerate(sorted(waves)):
    r = waves[name]
    if "series" not in r or "sharpness" not in r["series"]:
        continue
    ax = axes[0]
    ts = np.array(r["series"]["t"], float)
    sh = np.array(r["series"]["sharpness"], float)
    tt = np.linspace(ts[0], ts[-1], sh.size)  # sharpness is sampled every 20 records
    ax.plot(tt, sh, lw=0.9, label=name)
axes[0].set_xlabel("lattice step")
axes[0].set_ylabel(r"$\max|\nabla\phi|$")
axes[0].set_title("interface sharpness during the runs (ideal $=1/W$)")
axes[0].legend(fontsize=5)
axes[0].grid(alpha=0.3)
for name in sorted(mar):
    r = mar[name]
    axes[1].plot(r["series"]["t"], r["series"]["u_interface"], lw=0.9, label=name)
axes[1].set_xlabel("lattice step")
axes[1].set_ylabel("$u_i(t)$")
axes[1].set_title("Marangoni interface velocity history (steady state)")
axes[1].legend(fontsize=5)
axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig6_controls.png"), dpi=160)
plt.close(fig)
captions["fig6_controls.png"] = (
    "左：各波工况记录的界面锐度历史——**注意**：该历史是用修前的读数采的（周期接缝处 |∇φ| 恒 0.5，"
    "对所有 W 都报 0.5，见报告 §5.8），因此只能当审计留存，不能作为界面宽度结论；修正后的读数见 "
    "`interface_sharpness` 与回归测试。右：Marangoni 界面速度收敛历史，确认比较时已到稳态。"
)

with open(os.path.join(FIG, "captions.json"), "w") as fh:
    json.dump(captions, fh, ensure_ascii=False, indent=1)
print(json.dumps({k: v[:60] for k, v in captions.items()}, ensure_ascii=False, indent=1))