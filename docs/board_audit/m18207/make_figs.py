#!/usr/bin/env python3
"""m18207 审计图：①σ 误差 vs 界面分辨率（含 15% owner 带 + O(h^2) 参考）；②P_nn(x) 均匀性剖面。

全部数字来自交付模块实算（/root/tp-agent/work/m18114/src），零手抄。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
W = Path("/root/tp-agent/work/m18114")
sys.path.insert(0, str(W / "src"))
from tensorlbm.free_energy_stress import pressure_tensor  # noqa: E402
from tensorlbm.phasefield.binary_equilibrium_pressure import (  # noqa: E402
    EquilibriumPressureConfig,
    periodic_planar_profile,
    planar_pressure_diagnostics,
)

torch.set_num_threads(1)
FIGS = HERE / "figs"
FIGS.mkdir(parents=True, exist_ok=True)
res = (1.5, 2.0, 3.0)
data = {}
for xi in res:
    cfg = EquilibriumPressureConfig(interface_cells=xi, nx=128, ny=16, steps=2000)
    p = cfg.params
    f = periodic_planar_profile(16, 128, p)
    d = planar_pressure_diagnostics(f, p)
    pxx, pyy, _ = pressure_tensor(f.phi, p)
    data[xi] = {"sigma_err": d.sigma_relative_error, "p_coex": f.p_coex_closed_form,
                "pnn": (pxx[0] / f.p_coex_closed_form).tolist(),
                "ptt": (pyy[0] / f.p_coex_closed_form).tolist(),
                "unif": d.p_nn_max_deviation_relative}
(FIGS / "fig_data.json").write_text(json.dumps({str(k): v for k, v in data.items()}))

# fig1 — convergence with owner band
fig, ax = plt.subplots(figsize=(7.2, 4.6))
xi = list(data)
err = [abs(data[k]["sigma_err"]) * 100 for k in xi]
ax.axhspan(0, 15, color="#cfe8cf", alpha=0.7, label="owner accuracy band (<15%)")
ax.semilogy(xi, err, "o-", color="#1f4e79", label="measured |Δσ|/σ (delivered module)")
ref = [err[0] * (xi[0] / k) ** 2 for k in xi]
ax.semilogy(xi, ref, "k--", lw=1, label="O(h²) reference through ξ/h=1.5")
for k, e in zip(xi, err):
    ax.annotate(f"{e:.2f}%", (k, e), textcoords="offset points", xytext=(4, 6), fontsize=9)
ax.set_xlabel("interface resolution ξ/h (lattice cells)")
ax.set_ylabel("surface-tension relative error [%]")
ax.set_title("Free-energy binary LBM: σ from discrete pressure tensor vs analytic (2√2/3)√(κA³)/B")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGS / "fig1_sigma_convergence.png", dpi=140)
plt.close(fig)

# fig2 — P_nn / P_tt profiles
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
for axx, k in zip(axes, xi):
    axx.plot(range(128), data[k]["pnn"], label=r"$P_{nn}/p_{coex}$", color="#c0392b", lw=1.4)
    axx.plot(range(128), data[k]["ptt"], label=r"$P_{tt}/p_{coex}$", color="#2e86c1", lw=1.4)
    axx.axhline(1.0, color="k", ls=":", lw=0.8)
    axx.set_title(f"ξ/h={k}: max|P_nn-mean|/p_coex={data[k]['unif']*100:.1f}%", fontsize=9)
    axx.set_xlabel("x [lattice]")
    axx.grid(alpha=0.3)
axes[0].set_ylabel("pressure / p_coex")
axes[0].legend(fontsize=8)
fig.suptitle("Free-energy pressure tensor across two periodic flat interfaces (analytic equilibrium field)", fontsize=10)
fig.tight_layout()
fig.savefig(FIGS / "fig2_pnn_profiles.png", dpi=140)
plt.close(fig)
print("figs:", sorted(p.name for p in FIGS.iterdir()))
print("sigma err %:", [round(abs(data[k]["sigma_err"]) * 100, 3) for k in xi])
print("uniformity %:", [round(data[k]["unif"] * 100, 2) for k in xi])