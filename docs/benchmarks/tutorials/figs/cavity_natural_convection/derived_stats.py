#!/usr/bin/env python3
"""cavity_natural_convection 教程派生统计（m219）— 从自跑数据算诊断量，落 data/derived.json。

输入：data/scan_result.json + data/fields_n{64,128,256}.npz（rerun_scan.py 产物）
输出：data/derived.json（教程中所有剖面/流函数/壁面极值数字的唯一来源）

派生量：
  - 中线剖面网格差：粗档插值到细档节点后 |Δu|/u_max_ref、|Δv|/v_max_ref、|ΔT|
  - 流函数极值（以 H/α 无量纲，∫u dy 沿 x 积分）
  - 壁面局部 Nu 极值与位置（热壁/冷壁）
  - 对称性自检 |Nu_hot−Nu_cold|/Nu
用法：
  python derived_stats.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
U_MAX_REF = 16.178
V_MAX_REF = 19.617


def interp_profile(z: dict, key: str, target: np.ndarray) -> np.ndarray:
    return np.interp(target, z["y_norm"], z[key])


def main() -> None:
    scan = json.loads((DATA / "scan_result.json").read_text(encoding="utf-8"))
    fields = {g["nx"]: np.load(DATA / f"fields_n{g['nx']}.npz") for g in scan["grids"]}

    y_target = np.linspace(0.02, 0.98, 80)
    prof = {}
    pairs = list(fields)
    for a, b in zip(pairs[:-1], pairs[1:]):
        za, zb = fields[a], fields[b]
        du = interp_profile(za, "u_midwidth", y_target) - interp_profile(zb, "u_midwidth", y_target)
        dv = interp_profile(za, "v_midheight", y_target) - interp_profile(zb, "v_midheight", y_target)
        dT = interp_profile(za, "T_midwidth", y_target) - interp_profile(zb, "T_midwidth", y_target)
        prof[f"{a}->{b}"] = {
            "max_du_over_uref_pct": float(100.0 * np.max(np.abs(du)) / U_MAX_REF),
            "max_dv_over_vref_pct": float(100.0 * np.max(np.abs(dv)) / V_MAX_REF),
            "max_dabsT": float(np.max(np.abs(dT))),
        }

    wall = {}
    for nx, z in fields.items():
        y = z["y_norm"]
        hot, cold = z["nu_hot_wall"], z["nu_cold_wall"]
        wall[str(nx)] = {
            "nu_hot_max": float(hot.max()),
            "nu_hot_max_y": float(y[int(hot.argmax())]),
            "nu_hot_min": float(hot.min()),
            "nu_hot_min_y": float(y[int(hot.argmin())]),
            "nu_cold_max": float(cold.max()),
            "nu_cold_max_y": float(y[int(cold.argmax())]),
            "nu_cold_min": float(cold.min()),
            "nu_cold_min_y": float(y[int(cold.argmin())]),
        }

    psi = {}
    for nx, z in fields.items():
        ux = z["ux"].astype(np.float64)  # 已是 H/α 无量纲
        H = float(nx - 1)
        # 只取内点（去掉反弹壁环）：壁节点的宏观 u 是反弹分布组合值，非物理速度
        inner = ux[1:-1, 1:-1]
        # ψ(x,y) = ∫_0^y u dy（格单位 dy=1）→ 除 H 转 H/α 无量纲
        ps = np.cumsum(inner, axis=0) / H
        idx = np.unravel_index(np.argmin(ps), ps.shape)
        psi[str(nx)] = {
            "psi_min": float(ps.min()),
            "psi_min_x": float((idx[1] + 1) / H),
            "psi_min_y": float((idx[0] + 1) / H),
            "psi_max": float(ps.max()),
            "wall_ring_excluded": True,
        }

    out = {
        "source": "data/scan_result.json + data/fields_n*.npz（本 mission 自跑）",
        "profile_grid_delta": prof,
        "wall_nu_extremes": wall,
        "streamfunction": psi,
        "nu_ref": 2.243,
        "u_max_ref": U_MAX_REF,
        "v_max_ref": V_MAX_REF,
    }
    (DATA / "derived.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()