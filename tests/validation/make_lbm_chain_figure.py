#!/usr/bin/env python3
"""从 docs/lbm_verification_registry.json 生成复核图（不需要重算）。

左：Step B Couette 数值剖面 vs 解析剖面（数据取自注册表里记录的实际数组，
    不是手抄）；右：Step A–D 相对误差 + τ 扰动负控，对照 1% 判据线。

用法：python tests/validation/make_lbm_chain_figure.py [--out docs/assets/xxx.png]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REG = ROOT / "docs" / "lbm_verification_registry.json"

LABELS = {
    "shear_wave": "A shear wave\n(nu)",
    "couette": "B Couette\n(profile)",
    "poiseuille": "C Poiseuille\n(profile)",
    "wall_friction": "D wall friction\n(Cf)",
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "assets" / "lbm_verification_chain.png"))
    args = ap.parse_args(argv)

    reg = json.loads(REG.read_text())
    meas = {m["key"]: m for m in reg["measurements"]}
    thr = reg["threshold_pct"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.2))

    # --- 左：Couette 剖面 ---
    st = meas["couette"]["settings"]
    u = st["u_numeric"]
    ue = st["u_exact"]
    y = [i + 1 for i in range(len(u))]
    ax1.plot(ue, y, "-", lw=2.0, color="#1f77b4", label="analytic  $U(y-0.5)/(n_y-1.5)$")
    ax1.plot(u, y, "o", ms=5, mfc="none", color="#d62728", label="LBM (half-way BB)")
    ax1.set_xlabel("$u_x / U$")
    ax1.set_ylabel("y [lattice]")
    ax1.set_title(f"Step B Couette: max err = {meas['couette']['rel_err_pct']:.3f}%")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(alpha=0.3)

    # --- 右：误差 vs 1% 判据 ---
    keys = ["shear_wave", "couette", "poiseuille", "wall_friction"]
    errs = [max(meas[k]["rel_err_pct"], 1e-4) for k in keys]
    nc = reg["negative_controls"]["tau_perturbation"]
    xs = range(len(keys))
    ax2.bar(list(xs), errs, color="#2ca02c", width=0.6, label="Step A-D (measured)")
    ax2.bar([len(keys)], [nc["deviation_pct"]], color="#d62728", width=0.6,
            label="negative control (tau +0.02)")
    ax2.axhline(thr, color="k", ls="--", lw=1.2, label=f"pass threshold {thr:g}%")
    ax2.set_yscale("log")
    ax2.set_xticks(list(xs) + [len(keys)])
    ax2.set_xticklabels([LABELS[k] for k in keys] + ["negative\ncontrol"], fontsize=8)
    ax2.set_ylabel("relative error [%] (log)")
    ax2.set_title("analytic-chain errors vs 1% gate (gate has teeth)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, axis="y")

    for i, k in enumerate(keys):
        ax2.annotate(f"{meas[k]['rel_err_pct']:.3f}%", (i, errs[i]), ha="center",
                     va="bottom", fontsize=7)
    ax2.annotate(f"{nc['deviation_pct']:.2f}%", (len(keys), nc["deviation_pct"]),
                 ha="center", va="bottom", fontsize=7)

    fig.suptitle("TensorLBM m3579 — binding analytic verification chain (Step A-D)",
                 fontsize=10)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"wrote {out} ({out.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())