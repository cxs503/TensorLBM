#!/usr/bin/env python
"""CASE-RAYLEIGH-BENARD-RA1E4 图件生成（matplotlib，纯读取 runs/*.json）。

产出：
  figures/nu_vs_grid.png   —— Nu vs 网格 + de Vahl Davis 基准线 + ±5% 容差带
  figures/nu_history.png   —— Nu(t) 收敛过程（各档）
  figures/error_table.png  —— Nu/u_max/v_max 相对误差条形图
用法：python benchmarks/case_rayleigh_benard_ra1e4/make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "runs" / "case_rayleigh_benard_ra1e4"
FIG = REPO / "figures" / "case_rayleigh_benard_ra1e4"
NU_REF, U_REF, V_REF = 2.243, 16.178, 19.617


def load() -> list[dict]:
    keep = []
    for p in sorted(RUNS.glob("*.json")):
        d = json.loads(p.read_text())
        d["_file"] = p.name
        keep.append(d)
    return keep


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    data = load()
    if not data:
        raise SystemExit("no runs/*.json")
    # 主序列：tau 相同、nx 递增（网格收敛）
    by_tau: dict[float, list[dict]] = {}
    for d in data:
        by_tau.setdefault(round(float(d["config"]["tau"]), 3), []).append(d)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for tau, rows in sorted(by_tau.items()):
        rows = sorted(rows, key=lambda r: r["config"]["nx"])
        nxs = [r["config"]["nx"] for r in rows]
        nus = [r["result"]["nu_grad2"] for r in rows]
        ax.plot(nxs, nus, "o-", label=f"LBM tau={tau}")
    ax.axhline(NU_REF, color="k", ls="--", label=f"de Vahl Davis 1983 Nu={NU_REF}")
    ax.fill_between([min(r["config"]["nx"] for r in data),
                     max(r["config"]["nx"] for r in data)],
                    NU_REF * 0.95, NU_REF * 1.05, color="k", alpha=0.08,
                    label="±5% tolerance")
    ax.set_xlabel("grid nx (=ny), lattice units")
    ax.set_ylabel("Nu (hot-wall average, grad2)")
    ax.set_title("Natural convection square cavity Ra=1e4, Pr=0.71")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "nu_vs_grid.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for d in sorted(data, key=lambda r: r["config"]["nx"]):
        h = d.get("nu_history") or []
        if not h:
            continue
        ax.plot([x["step"] for x in h], [x["nu"] for x in h],
                label=f"nx={d['config']['nx']} tau={d['config']['tau']}")
    ax.axhline(NU_REF, color="k", ls="--", lw=1, label="ref 2.243")
    ax.set_xlabel("iteration")
    ax.set_ylabel("Nu")
    ax.set_title("Nu convergence history")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "nu_history.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    labels, nu_e, u_e, v_e = [], [], [], []
    for d in sorted(data, key=lambda r: (r["config"]["tau"], r["config"]["nx"])):
        r = d["result"]
        labels.append(f"nx{d['config']['nx']}/tau{d['config']['tau']}")
        nu_e.append(100.0 * (r["nu_grad2"] / NU_REF - 1.0))
        u_e.append(100.0 * (r["u_max_nd"] / U_REF - 1.0))
        v_e.append(100.0 * (r["v_max_nd"] / V_REF - 1.0))
    n = len(labels)
    idx = range(n)
    w = 0.26
    ax.bar([i - w for i in idx], nu_e, w, label="Nu err %")
    ax.bar(list(idx), u_e, w, label="u_max err %")
    ax.bar([i + w for i in idx], v_e, w, label="v_max err %")
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(5, color="r", ls=":", lw=1)
    ax.axhline(-5, color="r", ls=":", lw=1, label="±5% gate")
    ax.set_xticks(list(idx))
    ax.set_xticklabels(labels, rotation=20, fontsize=8)
    ax.set_ylabel("relative error vs de Vahl Davis [%]")
    ax.set_title("Case CASE-RAYLEIGH-BENARD-RA1E4 error per quantity")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "error_table.png", dpi=140)
    plt.close(fig)
    print("figures ->", FIG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())