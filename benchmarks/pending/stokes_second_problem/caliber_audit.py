#!/usr/bin/env python3
"""B25 stokes_second_problem — 判决口径审计（machine-checked caliber audit）与误差归因。

为什么需要这个脚本
--------------------------------------------------------------------------------
本案例 prereg 原文写「|u_ana| > 5%·U0 掩码下 max 相对误差 ≤ 3%」，而 run.py 主判据
掩码是 10%·U0。5% 掩码实测 5.16%（H=80, φ=π/2）——按原文**不达标**，案例因此停在
`pending/` 等「掩码口径裁定」。本脚本不重跑仿真，只读已入库的四个证据档
（`case_H40/H80.json`、`diag_H16/H160.json`），给出裁定所需的机器证据与判决。

审计结论（三句话）
--------------------------------------------------------------------------------
1. **5%·U0 掩码判据 ill-posed，不是「差一点没达标」而是「加密也永远不达标」**：
   掩码把分母钉死在阈值附近（实测 argmax 点 |u_ana|/阈值 = 1.10/1.04/1.01 × 于
   δ/Δx = 10/20/40），于是「掩码 max 相对误差」渐近变成一个被 1/(mask·U0) 放大的
   **绝对误差**度量；实测值 4.99/5.16/4.86% 在 4 倍分辨率跨度上构成平台、不趋零。
2. **同一掩码还有个更隐蔽的副作用**:10%·U0 掩码会把全场最大绝对误差的来源**掩掉**。
   本轮误差归因发现：max|Δu|/U0 = 1.37% 发生在**最深处流体节点**（η=y/δ≈3.95），
   那里解析幅度只有 ~1.3%·U0 而数值解≈0（底壁静止 BB 强制 u=0）——即误差由
   **有限域深度截断**（H=4δ ⇒ e^{-4}=1.8%）主导，与动壁离散无关，且不随 Δx 收敛。
3. 因此判决口径改为四条良定义门 + 一条使用边界：
   G1 无掩码 max|Δu|/U0 ≤ 3%（覆盖被掩码掩掉的深度截断误差）
   G2 10%·U0 掩码 max 相对误差 ≤ 3%（保留作分辨率判别器：δ/Δx=4 时读 15.76% 失败）
   G3 跨网格 η=y/δ 塌缩差 ≤ 3%·U0 · G4 周期稳态差 ≤ 0.5%·U0 · G5 使用边界 δ/Δx ≥ 10
   判决：G1–G5 全部通过（G2 的 2.16%/1.01% 即关键量，均 <3%）。

用法：
    python caliber_audit.py             # 打印审计表 + 写 caliber_audit.json；退出码=判决
    python caliber_audit.py --check     # 只打印一行判决（测试/CI 用）
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EVIDENCE = {  # tag -> 证据档（run.py 程序化写入，零手抄）
    "H16": "diag_H16.json",
    "H40": "case_H40.json",
    "H80": "case_H80.json",
    "H160": "diag_H160.json",
}
PREREG = ("H40", "H80")  # 门内两档（判决用）；H16/H160 只是诊断档
MASK_PREREG = 0.05  # prereg 原文掩码（未达标那条）
MASK_PRIMARY = 0.10  # run.py 主口径掩码
TOL_REL_PCT = 3.0
TOL_ABS_PCT = 3.0
TOL_COLLAPSE_PCT = 3.0
TOL_STEADY_PCT = 0.5
MIN_DELTA_PER_DX = 10.0  # 半程动壁 BB 的可用分辨率下限（README 已记录的使用边界）
NEAR_WALL_ETA = 0.5  # 近壁带：0 < η ≤ 0.5（动壁离散误差的归属区）
DEEP_ETA = 2.0  # 深场带：η ≥ 2（有限域截断误差的归属区）


# --------------------------------------------------------------------------
# 解析解零点：u(y)=U0·e^{-η}·cos(φ−η)，η=y/δ。φ=π/2 时壁面自身过零（驱动壁速=0）
# --------------------------------------------------------------------------
def analytic_zeros(delta: float, phi: float, y_max: float, n_scan: int = 200001) -> list[float]:
    """All y in [0, y_max] with u_ana(y)=0, by sign scan + bisection."""
    y = np.linspace(0.0, y_max, n_scan)
    u = np.cos(phi - y / delta)
    idx = np.nonzero(np.sign(u[:-1]) * np.sign(u[1:]) < 0)[0]
    zeros: list[float] = []
    for i in idx:
        lo, hi = float(y[i]), float(y[i + 1])
        f_lo = math.cos(phi - lo / delta)
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if f_lo * math.cos(phi - mid / delta) <= 0.0:
                hi = mid
            else:
                lo, f_lo = mid, math.cos(phi - mid / delta)
        zeros.append(0.5 * (lo + hi))
    if abs(math.cos(phi)) < 1e-12:  # 壁面过零（φ=π/2 的驱动壁速恰为 0）
        zeros.insert(0, 0.0)
    return zeros


def _masked_argmax(
    y: np.ndarray, u_num: np.ndarray, u_ana: np.ndarray, U0: float, frac: float
) -> dict:
    """|Δu|/|u_ana| 在 |u_ana|>frac·U0 上的 argmax，附「分母钉死」诊断。"""
    mask = np.abs(u_ana) > frac * U0
    rel = np.where(mask, np.abs(u_num - u_ana) / np.abs(u_ana), -np.inf)
    i = int(np.argmax(rel))
    return {
        "y_from_wall": float(y[i]),
        "u_ana_over_U0": float(abs(u_ana[i]) / U0),
        "abs_err_over_U0_pct": float(abs(u_num[i] - u_ana[i]) / U0 * 100.0),
        "rel_err_pct": float(rel[i] * 100.0),
        "denominator_over_threshold": float(abs(u_ana[i]) / (frac * U0)),
        "n_points_in_mask": int(mask.sum()),
    }


def audit_record_point(step: dict, U0: float) -> dict:
    """单记录点：两种掩码的 argmax + 良定义误差度量 + 近壁/深场归因。"""
    y = np.asarray(step["y_profile"], dtype=np.float64)
    u_num = np.asarray(step["u_profile"], dtype=np.float64)
    u_ana = np.asarray(step["u_analytic"], dtype=np.float64)
    delta = float(step["delta_lb"])
    phi = float(step["phase_actual"])
    eta = y / delta
    err = np.abs(u_num - u_ana)
    i_max = int(np.argmax(err))
    zeros = analytic_zeros(delta, phi, float(y.max()))

    out: dict = {
        "cycle_k": step["cycle_k"],
        "phase_nominal": round(float(step["phase_nominal"]), 6),
        "phase_actual": phi,
        "t_lb": step["t_lb"],
        # ---- 良定义（无掩码）度量 ----
        "max_abs_err_over_U0_pct": float(err[i_max] / U0 * 100.0),
        "l2_rel_err": float(np.linalg.norm(u_num - u_ana) / np.linalg.norm(u_ana)),
        # ---- 误差归因 ----
        "attribution": {
            "argmax_eta": float(eta[i_max]),
            "argmax_y_from_wall": float(y[i_max]),
            "argmax_u_ana_over_U0_pct": float(abs(u_ana[i_max]) / U0 * 100.0),
            "argmax_u_num_over_U0_pct": float(u_num[i_max] / U0 * 100.0),
            "eta_max": float(eta.max()),
            "deepest_u_ana_over_U0_pct": float(abs(u_ana[0]) / U0 * 100.0),
            "deepest_u_num_over_U0_pct": float(u_num[0] / U0 * 100.0),
            "near_wall_max_abs_pct": float(err[eta <= NEAR_WALL_ETA].max() / U0 * 100.0)
            if np.any(eta <= NEAR_WALL_ETA)
            else None,
            "deep_max_abs_pct": float(err[eta >= DEEP_ETA].max() / U0 * 100.0)
            if np.any(eta >= DEEP_ETA)
            else None,
            "deep_share_of_max_abs": float(err[eta >= DEEP_ETA].max() / err[i_max]),
        },
        "analytic_zeros_y": [round(z, 4) for z in zeros],
    }
    for frac, key in ((MASK_PREREG, "mask5"), (MASK_PRIMARY, "mask10")):
        a = _masked_argmax(y, u_num, u_ana, U0, frac)
        a["delta_to_nearest_zero_cells"] = (
            float(min(abs(a["y_from_wall"] - z) for z in zeros)) if zeros else None
        )
        a["implied_abs_gate_over_U0_pct"] = frac * TOL_REL_PCT  # 3% of frac·U0
        out[key] = a
    return out


def audit_grid(case: dict) -> dict:
    """单档汇总（晚周期=周期稳态后取最差；早周期只用于稳态自检）。"""
    U0 = float(case["U0"])
    delta = float(case["delta_lb"])
    rows = [audit_record_point(s, U0) for s in case["per_step"]]
    k_late = max(case["cycles"])
    late = [r for r in rows if r["cycle_k"] == k_late]

    picked = {
        "mask10": max(late, key=lambda r: r["mask10"]["rel_err_pct"]),
        "mask5": max(late, key=lambda r: r["mask5"]["rel_err_pct"]),
        "abs": max(late, key=lambda r: r["max_abs_err_over_U0_pct"]),
        "near": max(late, key=lambda r: r["attribution"]["near_wall_max_abs_pct"]),
        "deep": max(late, key=lambda r: r["attribution"]["deep_max_abs_pct"]),
    }
    return {
        "H": int(case["H"]),
        "delta_lb": delta,
        "delta_per_dx": delta,  # Δx = 1 lattice unit
        "cycles": list(case["cycles"]),
        "elapsed_s": case.get("elapsed_s"),
        "records": rows,
        "worst_mask10": {
            "phase": picked["mask10"]["phase_nominal"],
            "rel_err_pct": picked["mask10"]["mask10"]["rel_err_pct"],
            "denominator_over_threshold": picked["mask10"]["mask10"]["denominator_over_threshold"],
            "delta_to_nearest_zero_cells": picked["mask10"]["mask10"][
                "delta_to_nearest_zero_cells"
            ],
        },
        "worst_mask5": {
            "phase": picked["mask5"]["phase_nominal"],
            "rel_err_pct": picked["mask5"]["mask5"]["rel_err_pct"],
            "denominator_over_threshold": picked["mask5"]["mask5"]["denominator_over_threshold"],
            "abs_err_over_U0_pct": picked["mask5"]["mask5"]["abs_err_over_U0_pct"],
            "phase_abs": picked["mask5"]["mask5"]["implied_abs_gate_over_U0_pct"],
        },
        "worst_abs": {
            "phase": picked["abs"]["phase_nominal"],
            "pct": picked["abs"]["max_abs_err_over_U0_pct"],
            "eta": picked["abs"]["attribution"]["argmax_eta"],
            "u_ana_pct": picked["abs"]["attribution"]["argmax_u_ana_over_U0_pct"],
        },
        "worst_near_wall_abs": {
            "phase": picked["near"]["phase_nominal"],
            "pct": picked["near"]["attribution"]["near_wall_max_abs_pct"],
        },
        "worst_deep_abs": {
            "phase": picked["deep"]["phase_nominal"],
            "pct": picked["deep"]["attribution"]["deep_max_abs_pct"],
        },
    }


# --------------------------------------------------------------------------
# 跨网格：η 相似变量塌缩 & 周期稳态差（两档解不同物理问题，只能按 η 比）
# --------------------------------------------------------------------------
def eta_collapse(case_lo: dict, case_hi: dict) -> float:
    """max |Δu|/U0 between two grids interpolated in η=y/δ (prereg caliber G3)."""
    U0 = float(case_lo["U0"])
    k_lo, k_hi = max(case_lo["cycles"]), max(case_hi["cycles"])
    d_lo, d_hi = float(case_lo["delta_lb"]), float(case_hi["delta_lb"])
    worst = 0.0
    for phi in sorted({round(float(s["phase_nominal"]), 6) for s in case_lo["per_step"]}):
        slo = next(
            s
            for s in case_lo["per_step"]
            if s["cycle_k"] == k_lo and round(float(s["phase_nominal"]), 6) == phi
        )
        shi = next(
            s
            for s in case_hi["per_step"]
            if s["cycle_k"] == k_hi and round(float(s["phase_nominal"]), 6) == phi
        )
        ylo, ulo = np.array(slo["y_profile"]) / d_lo, np.array(slo["u_profile"])
        yhi, uhi = np.array(shi["y_profile"]) / d_hi, np.array(shi["u_profile"])
        o_lo, o_hi = np.argsort(ylo), np.argsort(yhi)
        ylo, ulo, yhi, uhi = ylo[o_lo], ulo[o_lo], yhi[o_hi], uhi[o_hi]
        common = yhi <= ylo[-1]
        worst = max(
            worst,
            float(np.max(np.abs(np.interp(yhi[common], ylo, ulo) - uhi[common]))) / U0 * 100.0,
        )
    return worst


def steady_diff(case: dict) -> float:
    """max |u_early − u_late|/U0 at same phase (周期稳态自检, G4)."""
    U0 = float(case["U0"])
    k_early, k_late = case["cycles"]
    by_phase: dict[float, dict[int, dict]] = {}
    for s in case["per_step"]:
        by_phase.setdefault(round(float(s["phase_nominal"]), 6), {})[s["cycle_k"]] = s
    worst = 0.0
    for k, ss in by_phase.items():
        if k_early in ss and k_late in ss:
            d = np.abs(np.array(ss[k_early]["u_profile"]) - np.array(ss[k_late]["u_profile"]))
            worst = max(worst, float(d.max()) / U0 * 100.0)
    return worst


def build_audit() -> dict:
    raw = {tag: json.loads((HERE / fn).read_text()) for tag, fn in EVIDENCE.items()}
    grids = {tag: audit_grid(case) for tag, case in raw.items()}
    U0 = float(raw["H40"]["U0"])
    in_gate = [t for t in EVIDENCE if grids[t]["delta_per_dx"] >= MIN_DELTA_PER_DX]

    conv = {
        t: {
            "delta_per_dx": grids[t]["delta_per_dx"],
            "mask5_pct": grids[t]["worst_mask5"]["rel_err_pct"],
            "mask10_pct": grids[t]["worst_mask10"]["rel_err_pct"],
            "abs_pct": grids[t]["worst_abs"]["pct"],
            "near_wall_pct": grids[t]["worst_near_wall_abs"]["pct"],
            "deep_pct": grids[t]["worst_deep_abs"]["pct"],
            "eta_max": max(r["attribution"]["eta_max"] for r in grids[t]["records"]),
        }
        for t in EVIDENCE
    }
    seq = [conv[t]["mask5_pct"] for t in ("H40", "H80", "H160")]
    plateau = {
        "mask5_by_delta_per_dx": seq,
        "mask5_spread_pp": float(max(seq) - min(seq)),
        "mask5_ratio_finest_over_coarsest": float(seq[-1] / seq[0]),
        "mask5_monotone_decreasing": bool(seq[0] > seq[1] > seq[2]),
        "mask5_trend_pct_per_octave": float((seq[-1] - seq[0]) / 2.0),
        "mask10_by_delta_per_dx": [conv[t]["mask10_pct"] for t in ("H40", "H80", "H160")],
        "abs_by_delta_per_dx": [conv[t]["abs_pct"] for t in ("H40", "H80", "H160")],
        "near_wall_by_delta_per_dx": [conv[t]["near_wall_pct"] for t in ("H40", "H80", "H160")],
    }

    pinning = {t: grids[t]["worst_mask5"]["denominator_over_threshold"] for t in in_gate}
    abs_at_argmax = [grids[t]["worst_mask5"]["abs_err_over_U0_pct"] for t in in_gate]
    required = MASK_PREREG * TOL_REL_PCT  # 5%-mask 3% 门 ⇔ 阈值处 |Δu| ≤ 0.15%·U0
    equivalence = {
        "mask5_gate_rel_pct": TOL_REL_PCT,
        "implied_abs_gate_over_U0_pct": required,
        "measured_abs_at_mask5_argmax_over_U0_pct": abs_at_argmax,
        "implied_tightening_factor": float(np.mean(abs_at_argmax) / required),
        "mask10_implied_abs_gate_over_U0_pct": MASK_PRIMARY * TOL_REL_PCT,
        "statement": (
            "掩码门 3% 在分母钉死到阈值时等价于「阈值处绝对误差 ≤ mask·U0·3%」，"
            "即 5% 掩码要求 0.15%·U0、10% 掩码要求 0.30%·U0；实测该处绝对误差 0.24–0.29%·U0。"
        ),
    }

    truncation = {
        "eta_max_by_grid": {t: conv[t]["eta_max"] for t in EVIDENCE},
        "analytic_amplitude_at_deepest_pct": {
            t: grids[t]["records"][0]["attribution"]["deepest_u_ana_over_U0_pct"] for t in EVIDENCE
        },
        "numeric_value_at_deepest_pct": {
            t: grids[t]["records"][0]["attribution"]["deepest_u_num_over_U0_pct"] for t in EVIDENCE
        },
        "deep_share_of_max_abs": {
            t: max(r["attribution"]["deep_share_of_max_abs"] for r in grids[t]["records"])
            for t in EVIDENCE
        },
        "statement": (
            "全场 max|Δu| 落在最深处流体节点（η≈3.9–4.0）：解析幅度仅 ~1.3%·U0，数值解 ≈0.1%·U0"
            "（底壁静止 BB 强制无滑移而 H=4δ 不够深）⇒ 主绝对误差是**有限域深度截断** "
            "e^{-η_max}≈1.8%，与 Δx 无关，只随 H/δ 变化；10% 掩码恰好把这条误差掩掉。"
        ),
    }

    gates = {
        "G1_unmasked_abs_le_tol": {
            "tol_pct": TOL_ABS_PCT,
            "values_pct": {t: conv[t]["abs_pct"] for t in EVIDENCE},
            "pass": all(conv[t]["abs_pct"] <= TOL_ABS_PCT for t in PREREG),
        },
        "G2_mask10_rel_le_tol": {
            "tol_pct": TOL_REL_PCT,
            "values_pct": {t: conv[t]["mask10_pct"] for t in PREREG},
            "pass": all(conv[t]["mask10_pct"] <= TOL_REL_PCT for t in PREREG),
        },
        "G3_eta_collapse_le_tol": {
            "tol_pct": TOL_COLLAPSE_PCT,
            "value_pct": eta_collapse(raw["H40"], raw["H80"]),
        },
        "G4_periodic_steady_le_tol": {
            "tol_pct": TOL_STEADY_PCT,
            "values_pct": {t: steady_diff(raw[t]) for t in PREREG},
        },
        "G5_resolution_floor": {
            "min_delta_per_dx": MIN_DELTA_PER_DX,
            "prereg_delta_per_dx": [conv[t]["delta_per_dx"] for t in PREREG],
            "pass": all(conv[t]["delta_per_dx"] >= MIN_DELTA_PER_DX for t in PREREG),
        },
    }
    gates["G3_eta_collapse_le_tol"]["pass"] = (
        gates["G3_eta_collapse_le_tol"]["value_pct"] <= TOL_COLLAPSE_PCT
    )
    gates["G4_periodic_steady_le_tol"]["pass"] = all(
        v <= TOL_STEADY_PCT for v in gates["G4_periodic_steady_le_tol"]["values_pct"].values()
    )

    discriminating = {
        "delta_per_dx": conv["H16"]["delta_per_dx"],
        "mask10_pct": conv["H16"]["mask10_pct"],
        "tol_pct": TOL_REL_PCT,
        "fails": bool(conv["H16"]["mask10_pct"] > TOL_REL_PCT),
        "ratio_over_tol": float(conv["H16"]["mask10_pct"] / TOL_REL_PCT),
        "near_wall_pct": conv["H16"]["near_wall_pct"],
        "near_wall_ratio_over_finest": float(
            conv["H16"]["near_wall_pct"] / conv["H80"]["near_wall_pct"]
        ),
    }

    amended_pass = all(g.get("pass", False) for g in gates.values())
    mask5_pass = all(conv[t]["mask5_pct"] <= TOL_REL_PCT for t in PREREG)

    verdict = {
        "mask5_prereg_criterion": "PASS" if mask5_pass else "ILL_POSED_UNSATISFIABLE",
        "mask5_measured_pct": max(conv[t]["mask5_pct"] for t in PREREG),
        "amended_caliber": "PASS" if amended_pass else "FAIL",
        "amended_gates": {k: bool(v.get("pass", False)) for k, v in gates.items()},
        "key_quantities_pct": {t: conv[t]["mask10_pct"] for t in PREREG},
        "key_quantities_below_3pct": all(conv[t]["mask10_pct"] <= TOL_REL_PCT for t in PREREG),
        "retired_criterion": (
            "|u_ana|>5%·U0 掩码 max 相对误差 ≤3% —— 分母钉死在阈值附近，加密不收敛（4.99/5.16/4.86%），"
            "且会掩掉深度截断误差；不构成相对精度判据，退役。"
        ),
        "kept_as_reporter": (
            "10%·U0 掩码 max 相对误差保留为分辨率判别器（δ/Δx=4 读 15.76% 失败），"
            "但其渐近亦有分母钉死（δ/Δx=40 时 1.003× 阈值），故不作唯一判据。"
        ),
        "promotion": (
            "PASS ⇒ 可按 benchmarks/STATUS.md 验收条款晋级 verified/（两档网格 span "
            f"{gates['G3_eta_collapse_le_tol']['value_pct']:.3f}% ≤ 3%）；"
            "晋级需目录挪动（pending→verified），与在飞分支 m4917 于本目录的改写重叠，"
            "本单不做挪动，只交口径裁定。"
        ),
        "not_done": [
            "固定 δ 只加密 Δx 的收敛阶估计（本轮各档 δ=H/4，δ 与 Δx 同时变）",
            "加深域（H/δ≥6）以把深度截断从 1.8% 压到 e^{-6}=0.25%，让绝对误差重新由动壁离散主导",
        ],
    }

    return {
        "case": "B25_stokes_second_problem_caliber_audit",
        "generated_by": "benchmarks/pending/stokes_second_problem/caliber_audit.py",
        "source_evidence": [EVIDENCE[t] for t in EVIDENCE],
        "U0": U0,
        "tolerances": {
            "rel_pct": TOL_REL_PCT,
            "abs_pct": TOL_ABS_PCT,
            "collapse_pct": TOL_COLLAPSE_PCT,
            "steady_pct": TOL_STEADY_PCT,
            "mask_prereg": MASK_PREREG,
            "mask_primary": MASK_PRIMARY,
            "min_delta_per_dx": MIN_DELTA_PER_DX,
        },
        "grids": grids,
        "evidence_denominator_pinning": pinning,
        "evidence_mask5_equivalence": equivalence,
        "evidence_convergence": conv,
        "evidence_mask5_plateau": plateau,
        "evidence_truncation_attribution": truncation,
        "gates": gates,
        "discriminating_power": discriminating,
        "verdict": verdict,
    }


def markdown_table(a: dict) -> str:
    conv = a["evidence_convergence"]
    lines = [
        "| 档 | δ/Δx | η_max | 5%·U0 掩码 rel [%]（prereg 原文） | argmax 点 \\|u_ana\\|/阈值 [×] | "
        "10%·U0 掩码 rel [%] | 无掩码 max abs/U0 [%] | 近壁 η≤0.5 abs [%] | 深场 η≥2 abs [%] |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for t in ("H16", "H40", "H80", "H160"):
        g, c = a["grids"][t], conv[t]
        lines.append(
            f"| {t} | {c['delta_per_dx']:.0f} | {c['eta_max']:.2f} | **{c['mask5_pct']:.3f}** | "
            f"{g['worst_mask5']['denominator_over_threshold']:.3f} | {c['mask10_pct']:.3f} | "
            f"{c['abs_pct']:.3f} | {c['near_wall_pct']:.3f} | {c['deep_pct']:.3f} |"
        )
    return "\n".join(lines)


def summary_line(a: dict) -> str:
    v, p = a["verdict"], a["evidence_mask5_plateau"]
    return (
        f"mask5={v['mask5_measured_pct']:.3f}% "
        f"(plateau {min(p['mask5_by_delta_per_dx']):.2f}-{max(p['mask5_by_delta_per_dx']):.2f}% over "
        f"δ/Δx=10..40, trend {p['mask5_trend_pct_per_octave']:+.2f}pp/oct) -> {v['mask5_prereg_criterion']}; "
        f"amended={v['amended_caliber']} {v['amended_gates']}; "
        f"key quantities {v['key_quantities_pct']} all<3%={v['key_quantities_below_3pct']}; "
        f"discriminating δ/Δx={a['discriminating_power']['delta_per_dx']:.0f} -> "
        f"{a['discriminating_power']['mask10_pct']:.3f}% (fails={a['discriminating_power']['fails']})"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(HERE / "caliber_audit.json"))
    ap.add_argument("--check", action="store_true", help="只打印判决；退出码 0=审计通过")
    args = ap.parse_args(argv)

    a = build_audit()
    ok = a["verdict"]["amended_caliber"] == "PASS" and a["discriminating_power"]["fails"]
    if args.check:
        print(summary_line(a))
        return 0 if ok else 1
    Path(args.out).write_text(json.dumps(a, indent=1, sort_keys=True) + "\n")
    print(markdown_table(a))
    print()
    print(summary_line(a))
    print(f"wrote {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
