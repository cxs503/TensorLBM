#!/usr/bin/env python
"""Inject the LBM verification table into the study document.

The document ``docs/SELFPROPULSION_KCS_2026-11-07.md`` carries the placeholder
``RESULT_TABLE_TUNNEL``; this script renders the table from
``artifacts/selfpropulsion_kcs/tunnel.json`` so that no number is copied by
hand (single source of truth = the LBM artefact).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def render(tunnel: dict) -> str:
    cfg, me, ch = tunnel["config"], tunnel["measured"], tunnel["checks"]
    u_in = cfg["u_in"]
    lines = [
        f"算例：`nz×ny×nx = {cfg['nz']}×{cfg['ny']}×{cfg['nx']}`，盘半径 R = "
        f"{cfg['radius_cells']:.0f} 格，U = {u_in}，τ = {cfg['tau']}，"
        f"C_s = {cfg['smagorinsky_cs']}，{cfg['n_steps']} 步（后 {cfg['sample_steps']} 步时间平均），"
        f"墙钟 {tunnel.get('wall_clock_s', float('nan')):.0f} s。",
        "",
        "| 核验项 | 实测 / 偏差 | 判读 |",
        "|---|---|---|",
        f"| 推力记账 Σf_x vs T* | {ch['thrust_accounting_rel_error']:.2e} | 注入动量逐格精确 |",
        f"| 盘面诱导速度 u_a/U | {me['u_induced'] / u_in:.4f} | 体积力产生射流加速 |",
        f"| Δp 实测 | {me['delta_p']:.3e} | 压力跳变为动量源的结果（未施加） |",
        f"| Δp vs u_a(U+u_a) | {100 * ch.get('delta_p_rel_error_vs_momentum_theory', float('nan')):.1f}% | "
        f"局部动量理论核验（有效） |",
        f"| Δp vs T/A_geo | {100 * ch['delta_p_rel_error_vs_T_over_A']:.1f}% | "
        f"受通道阻塞：A_eff/A_geo ≈ {me['delta_p_from_thrust'] / abs(me['delta_p']):.2f}，只作量级判读 |",
        f"| 远场尾流比 (u_w−U)/(u_d−U) | {me['far_wake_ratio']:.3f}（理论 2） | "
        f"定速直通道质量守恒使其必然 <2，仅趋势判读 |",
        f"| 角动量通量 vs 力矩（全断面） | {100 * ch['swirl_rel_error_vs_torque']:.1f}% | "
        f"周期侧边界下径向通量成对抵消，可作守恒核验 |",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tunnel", default="artifacts/selfpropulsion_kcs/tunnel.json")
    ap.add_argument("--doc", default="docs/SELFPROPULSION_KCS_2026-11-07.md")
    args = ap.parse_args()
    tunnel = json.loads((REPO / args.tunnel).read_text())
    doc = REPO / args.doc
    text = doc.read_text()
    if "RESULT_TABLE_TUNNEL" not in text:
        print("placeholder already filled; nothing to do")
        return
    doc.write_text(text.replace("RESULT_TABLE_TUNNEL", render(tunnel)))
    print("document updated")


if __name__ == "__main__":
    main()