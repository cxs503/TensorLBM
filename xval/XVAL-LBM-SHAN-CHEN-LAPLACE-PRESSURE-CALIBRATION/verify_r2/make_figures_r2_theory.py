#!/usr/bin/env python3
"""R2 理论腿出图（无需求解器）：SC EOS/Maxwell 构造 + Korteweg 锚的闭式校验。"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from run_r2 import G_EFF, kappa3, maxwell_coexistence, p_eos, sigma_korteweg_1d  # noqa: E402

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"


def main() -> None:
    d = json.loads((ART / "r2_summary.json").read_text())
    mw = d["maxwell"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    rho = torch.linspace(1e-3, 3.2, 2000, dtype=torch.float64)
    p = p_eos(rho)
    axes[0].plot(rho, p, color="C0", lw=1.4, label="p(ρ)=ρc_s²+G_eff c_s²ψ²/2")
    axes[0].axhline(mw["p_c"], color="k", ls=":", lw=1, label=f"p_c(Maxwell)={mw['p_c']:.5f}")
    axes[0].axvline(mw["rho_v"], color="C3", ls="--", lw=1)
    axes[0].axvline(mw["rho_l"], color="C3", ls="--", lw=1)
    axes[0].plot([mw["rho_v"], mw["rho_l"]], [mw["p_c"], mw["p_c"]], "o", color="C3",
                 label=f"共存 ρ_v={mw['rho_v']:.4f}, ρ_l={mw['rho_l']:.4f}")
    for ps in mw["p_spinodal"]:
        axes[0].axhline(ps, color="0.6", lw=0.7)
    axes[0].set_xlabel("ρ"); axes[0].set_ylabel("p")
    axes[0].set_title("SC EOS + Maxwell 等面积构造（解析基准）")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    n, half = 192, 48.0
    x = torch.arange(n, dtype=torch.float64)
    k3 = kappa3(2)["kappa3"]
    ws, num, closed = [], [], []
    for w in (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0):
        shape = 0.5 * (torch.tanh((x - (n / 2 - half)) / w) - torch.tanh((x - (n / 2 + half)) / w))
        prof = (0.02 + 0.02 * shape).float()  # Δρ→0 ⇒ ψ_ρ≈1，闭式成立
        out = sigma_korteweg_1d(prof, k3, half)
        dpsi = float(1.0 - math.exp(-0.04) - (1.0 - math.exp(-0.02)))
        ws.append(w); num.append(out["sigma_K"]); closed.append(2.0 * abs(G_EFF * k3) * dpsi**2 / (3.0 * w))
    axes[1].plot(ws, num, "o-", color="C0", label="数值 ∫(∂_nψ)²dn 后的 σ_K")
    axes[1].plot(ws, closed, "s--", color="C3", label="闭式 2|G_eff κ3|Δψ²/(3w)")
    rel = max(abs(a - b) / b for a, b in zip(num, closed))
    axes[1].set_xlabel("tanh 界面宽 w"); axes[1].set_ylabel("σ_K")
    axes[1].set_title(f"Korteweg 锚自检：数值 vs 闭式（最大偏差 {rel*100:.2f}%）")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    (ART / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(ART / "figures" / "fig_r2_theory.png", dpi=130)
    print("saved", ART / "figures" / "fig_r2_theory.png", "rel_err", rel)


if __name__ == "__main__":
    main()
