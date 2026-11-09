"""Build ``benchmarks/verified/suboff_anchor/anchor_points.json``.

工程 ①：DARPA SUBOFF 裸艇（AFF-1）C_T(U) 锚点表逐字落盘（≥5 个 (Re, C_T) 对，
每点带引句 + 表号/页码或仓内文件路径+行号，机器校验命中率 100%）。
工程 ②：ITTC-78 C_F 程序化复算 + (1+k)=C_T/C_F 锚内一致性检查。
工程 ③：锚 Re 阶梯的可执行设计（最细可执行网格 h 与算力；相似性换算误差预算公式+数值）。

零手抄纪律（三条，可执行）：
1. 每个数字都从 ``sources/anchor_points_transcriptions.json`` 里 quote 的正则解析
   （quote 必须等于对应 file 第 line 行原文，失配即构建失败）。
2. 公式系数（0.075/2、0.455/2.58、0.37/-0.2、26.66、150、u_lb、m）同样解析，不写字面量。
3. AST 自检：脚本内不得出现等于任一派生量（C_T/C_F/Re/(1+k)/h）的浮点字面量。

Regenerate: ``python3 benchmarks/verified/suboff_anchor/build_anchor_points.py``
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from decimal import Decimal, getcontext
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SRC_TRANS = HERE / "sources" / "anchor_points_transcriptions.json"
SRC_MLUPS = HERE / "sources" / "cluster_mlups_measurement.json"
GCI_JSON = REPO / "benchmarks/verified/suboff_sail_resistance/gci_v4.json"
OUT_JSON = HERE / "anchor_points.json"
OUT_FIG = HERE / "figures" / "anchor_points_overview.png"

MISSION_ID = 8530
TASK_KEY = "paper-exp-lbm-suboff-tau-A"

# ---- predeclared thresholds (判据先于动手；这些是门槛，不是结果) ------------
CRIT = {
    "c1_min_points": 5,
    "c1_min_quote_hit_rate_pct": 100.0,
    "c2_re_rel_dev_max_pct": 0.1,
    "c3_cf_residual_max_abs": 1e-12,
    "c3_one_plus_k_consistency_max_pct": 1.0,
    "c4_evidence_budget_hours": 72.0,
    "band_pct": 3.0,
}

REQUIREMENTS = {  # 判据原文（任务书）
    "C1": "锚点值 ≥5 个 (Re, C_T)，逐点附引句+表号/页码或仓内数据文件路径+行号，引句机检命中率 100%",
    "C2": "Re = U·L/ν 与所抄 Re 相对偏差 ≤0.1%",
    "C3": "ITTC-78 C_F 复算与解析式残差 <1e-12；(1+k) 两锚互算一致性 ≤1%",
    "C4": "给出「锚 Re 可达性」二值结论 + 若不可达的相似性换算误差预算（公式+数值）",
    "C5": "零手抄结构自检（AST：脚本内无等于派生量的浮点字面量）",
}


# --------------------------------------------------------------------------- #
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rnd(x: float, n: int = 10) -> float:
    return float(f"{x:.{n}g}")


def rel_pct(value: float, ref: float) -> float:
    return (value - ref) / ref * 100.0


def num(pattern: str, text: str, group: int = 1) -> float:
    m = re.search(pattern, text)
    if not m:
        raise SystemExit(f"[parse] pattern {pattern!r} not found in {text!r}")
    return float(m.group(group).replace("−", "-"))


def verify_quotes(trans: dict) -> tuple[dict[str, dict], dict]:
    """逐字校验：每条 quote 必须等于 file 第 line 行，且能命中其 regex。"""
    verified: dict[str, dict] = {}
    misses = []
    for q in trans["quotes"]:
        path = REPO / q["file"]
        blob = path.read_text(encoding="utf-8")
        lines = blob.split("\n")
        line_ok = 0 <= q["line"] - 1 < len(lines) and lines[q["line"] - 1] == q["text"]
        embedded = q["text"] in blob
        rx_ok = bool(re.search(q["regex"], q["text"]))
        if not (line_ok and embedded and rx_ok):
            misses.append({"id": q["id"], "line_exact": line_ok,
                           "substring_in_file": embedded, "regex_hit": rx_ok})
        verified[q["id"]] = {
            "id": q["id"],
            "file": q["file"], "line": q["line"], "text": q["text"],
            "grade": q["grade"], "locator": q["locator"],
            "source_file_sha256": sha256(path),
            "machine_check": {"line_exact": line_ok, "substring_in_file": embedded,
                              "regex_hit": rx_ok},
        }
    n = len(trans["quotes"])
    report = {
        "n_quotes": n,
        "n_hits": n - len(misses),
        "hit_rate_pct": (n - len(misses)) / n * 100.0,
        "misses": misses,
        "rule": "quote == 该 file 第 line 行原文（逐字相等）+ 正则可解析",
    }
    if misses:
        raise SystemExit(f"[verbatim] quote verification failed: {misses}")
    return verified, report


def cf_routes(re_: float, coef: float, shift: float, exponent: float) -> dict[str, float]:
    """C_F = coef / (log10 Re - shift)**exponent，三条独立求值路线。"""
    direct = coef / (math.log10(re_) - shift) ** exponent
    ln10 = math.log(10.0)
    via_ln = coef * ln10**exponent / (math.log(re_) - shift * ln10) ** exponent
    getcontext().prec = 60
    dec = Decimal(coef) / ((Decimal(re_).ln() / Decimal(10).ln() - Decimal(shift))
                           ** Decimal(exponent))
    return {"direct_log10": direct, "via_natural_log": via_ln,
            "decimal_60digit": float(dec)}


def cf_analytic(re_: float, coef: float, shift: float, exponent: float) -> float:
    return coef / (math.log10(re_) - shift) ** exponent


# --------------------------------------------------------------------------- #
def build() -> dict:
    trans = json.loads(SRC_TRANS.read_text(encoding="utf-8"))
    q, quote_report = verify_quotes(trans)

    # ---- geometry / fluid conventions (parsed from quotes) ----------------
    L = num(r"\| ([\d.]+) \|", q["GEO_L"]["text"])
    S = num(r"\| ([\d.]+) \|", q["GEO_S"]["text"])
    rho = num(r"ρ=([\d.]+) kg/m³", q["CTDEF"]["text"])
    nu = num(r"ν=([\d.e+-]+) m²/s", q["REDEF"]["text"])
    nu_15 = num(r"nu_15, nu_20 = ([\d.e+-]+),", q["NU_T_ANCHORS"]["text"])
    nu_20 = num(r"nu_15, nu_20 = [\d.e+-]+, ([\d.e+-]+)", q["NU_T_ANCHORS"]["text"])
    band = re.search(r"密度带 (\d+)–(\d+)", q["CTDEF"]["text"])
    rho_lo, rho_hi = float(band.group(1)), float(band.group(2))
    analytic_s_coef = num(r"([\d.]+)·L²", q["SCAD"]["text"])

    KN = 1852.0 / 3600.0  # exact by definition (1 nmi = 1852 m)

    cf_it = re.search(r"([\d.]+)/\(log10 Re - ([\d.]+)\)\^([\d.]+)", q["CF_ITTC78"]["text"])
    cf_coef, cf_shift, cf_exp = (float(cf_it.group(i)) for i in (1, 2, 3))
    cf_sc = re.search(r"([\d.]+)/\(log10 Re\)\^([\d.]+)", q["CF_SCHOENHERR"]["text"])
    sch_coef, sch_exp = float(cf_sc.group(1)), float(cf_sc.group(2))

    def cf78(re_: float) -> float:
        return cf_analytic(re_, cf_coef, cf_shift, cf_exp)

    def cf_schoenherr(re_: float) -> float:
        return sch_coef / math.log10(re_) ** sch_exp

    # ---- anchor points: DTRC Table 14 p.23 bare hull + 4-digit Re ladder --
    t14 = [q[f"T14_B{i}"] for i in range(1, 7)]
    t14_full = [q[f"T14_FULL_R{i}"] for i in range(1, 7)]
    t3 = [q[f"T3_F{i}"] for i in range(1, 6)]
    in25 = [q[f"IN25_R{i}"] for i in range(1, 7)]

    points, cf_residuals = [], []
    for row, ladder in zip(t14, in25):
        kn = num(r'bare_hull", 1, ([\d.]+),', row["text"])
        rt = num(r'bare_hull", 1, [\d.]+, ([\d.]+)\)', row["text"])
        cells = [c.strip() for c in ladder["text"].strip("|").split("|")]
        kn_l, re_tr = float(cells[0]), float(cells[2])
        ct_l = float(cells[4])
        if abs(kn_l - kn) > 0:
            raise SystemExit(f"[pairing] speed mismatch: {kn_l} vs {kn}")
        u = kn * KN
        re_calc = u * L / nu
        ct = rt / (0.5 * rho * u**2 * S)
        routes = cf_routes(re_calc, cf_coef, cf_shift, cf_exp)
        cf_val = routes["direct_log10"]
        resid = {k: abs(v - cf_val) for k, v in routes.items()}
        cf_residuals.append({"point_id": f"AFF1_{kn:g}kn", "Re": rnd(re_calc),
                             "abs_residual": {k: f"{v:.2e}" for k, v in resid.items()}})
        points.append({
            "point_id": f"AFF1_{kn:g}kn",
            "U_kn": kn,
            "U_mps": rnd(u),
            "R_T_N": rt,
            "Re_transcribed": re_tr,
            "Re_transcribed_grade": ladder["grade"],
            "Re_transcribed_source": {"file": ladder["file"], "line": ladder["line"],
                                      "text": ladder["text"]},
            "Re_computed": rnd(re_calc),
            "re_rel_dev_pct": rnd(rel_pct(re_calc, re_tr), 6),
            "CT": rnd(ct),
            "CT_in_re_ladder": ct_l,
            "CT_vs_re_ladder_pct": rnd(rel_pct(ct, ct_l), 4),
            "CF_ittc78": rnd(cf_val),
            "one_plus_k": rnd(ct / cf_val),
            "CF_recompute_residual_abs": {k: f"{v:.2e}" for k, v in resid.items()},
            "quotes": [
                {"id": row["id"], "file": row["file"], "line": row["line"],
                 "text": row["text"], "locator": row["locator"], "grade": row["grade"]},
                {"id": ladder["id"], "file": ladder["file"], "line": ladder["line"],
                 "text": ladder["text"], "locator": ladder["locator"],
                 "grade": ladder["grade"]},
            ],
        })

    # ---- C1 ---------------------------------------------------------------
    c1_pass = (len(points) >= CRIT["c1_min_points"]
               and quote_report["hit_rate_pct"] >= CRIT["c1_min_quote_hit_rate_pct"]
               and all(p["quotes"][0]["locator"] and p["quotes"][1]["file"] for p in points))
    c1 = {
        "n_points": len(points),
        "min_required": CRIT["c1_min_points"],
        "n_points_with_locator_and_line": sum(
            1 for p in points if all(x["locator"] and x["line"] for x in p["quotes"])),
        "quote_hit_rate_pct": quote_report["hit_rate_pct"],
        "quote_rule": quote_report["rule"],
    }

    # ---- C2: Re consistency ----------------------------------------------
    head = points[0]
    re_devs = [abs(p["re_rel_dev_pct"]) for p in points]
    retired_re = num(r"\| ([\d.]+)×10⁷ \|", q["RETIRED_ANCHOR"]["text"]) * 1e7
    retired_dev = rel_pct(head["Re_computed"], retired_re)
    nu_implied = head["U_mps"] * L / retired_re
    t_implied = 15.0 + 5.0 * (nu_15 - nu_implied) / (nu_15 - nu_20)
    t_declared = 15.0 + 5.0 * (nu_15 - nu) / (nu_15 - nu_20)
    c2 = {
        "max_abs_re_rel_dev_pct": rnd(max(re_devs), 6),
        "threshold_pct": CRIT["c2_re_rel_dev_max_pct"],
        "per_point_pct": {p["point_id"]: p["re_rel_dev_pct"] for p in points},
        "what_it_validates": (
            "对 4 位有效数字的仓内 Re 阶梯（IN25-T6 表 2，与锚同 L/ν 口径）逐点复算："
            "验证单位换算 / 特征长度 / 转录无漂移。它不验证物理 ν —— ν 假设的影响经 C_F 斜率"
            "另计入 conventions_uncertainty；且该阶梯本身的 Re 是仓内派生量（grade=DERIVED_IN_REPO），"
            "故本判据是『一致 / 无漂移』检查，不是独立物理验证。"),
        "retired_two_sig_fig_anchor": {
            "Re_transcribed": retired_re,
            "source": {"file": q["RETIRED_ANCHOR"]["file"], "line": q["RETIRED_ANCHOR"]["line"],
                       "text": q["RETIRED_ANCHOR"]["text"],
                       "grade": q["RETIRED_ANCHOR"]["grade"],
                       "locator": q["RETIRED_ANCHOR"]["locator"]},
            "re_computed": head["Re_computed"],
            "rel_dev_pct": rnd(retired_dev, 4),
            "verdict": ("RETIRED —— 两位有效数字与 4 位阶梯相差 %.2f%%，不能满足 0.1%% 级判据；"
                        "这正是任务书所述 #8800 C2 PARTIAL_FAIL 的来源。" % abs(retired_dev)),
            "reconciliation": {
                "nu_implied_m2_s": rnd(nu_implied),
                "T_implied_C": rnd(t_implied, 4),
                "T_of_declared_nu_C": rnd(t_declared, 4),
                "note": ("用仓内两个 ν 锚点 ν(15 °C)=%.3e / ν(20 °C)=%.3e 线性内插反解水温："
                         "所抄 1.2e7 对应约 %.1f °C、声明的 ν 对应约 %.1f °C，两者仅差约 %.1f °C。"
                         "即 %.2f%% 的差异完全可由拖曳水池水温解释 ⇒ 『两位有效数字锚』无法区分"
                         "『水温不同』与『转录错』，故弃用它、改用 4 位有效数字阶梯，而不是放宽判据。"
                         % (nu_15, nu_20, t_implied, t_declared,
                            abs(t_implied - t_declared), abs(retired_dev))),
            },
        },
    }
    c2_pass = max(re_devs) <= CRIT["c2_re_rel_dev_max_pct"]

    # ---- C3a: C_F recompute residual -------------------------------------
    max_cf_resid = max(max(float(v) for v in r["abs_residual"].values())
                       for r in cf_residuals)
    c3a = {
        "formula": f"C_F = {cf_coef} / (log10(Re) - {cf_shift})^{cf_exp}",
        "formula_source": q["CF_ITTC78"]["locator"],
        "routes": ["direct_log10", "via_natural_log", "decimal_60digit"],
        "per_point": cf_residuals,
        "max_abs_residual": f"{max_cf_resid:.3e}",
        "threshold_abs_residual": CRIT["c3_cf_residual_max_abs"],
        "note": ("ITTC-78 性能预报法所用的摩擦线即上式（模型-实船相关线 ITTC-1957）；"
                 "模型尺度光滑艇体不加 ΔC_F 粗糙度补偿，也不加实船相关津贴 C_A。"),
    }
    c3a_pass = max_cf_resid < CRIT["c3_cf_residual_max_abs"]

    # ---- C3b: (1+k) two-anchor cross calculation -------------------------
    pair_tol_pct = 0.5   # predeclared: two transcriptions pair only if |dU|/U <= 0.5%
    pairs, unpaired = [], []
    for row in t14_full:
        kn = num(r'full", 8, ([\d.]+),', row["text"])
        rt = num(r'full", 8, [\d.]+, ([\d.]+)\)', row["text"])
        u_a = kn * KN
        best = None
        for row_b in t3:
            u_b = num(r"\| ([\d.]+) \|", row_b["text"])
            rt_b = num(r"\| [\d.]+ \| ([\d.]+) \|", row_b["text"])
            d = abs(u_b - u_a) / u_a
            if best is None or d < best[0]:
                best = (d, u_b, rt_b, row_b)
        d, u_b, rt_b, row_b = best
        if d * 100.0 > pair_tol_pct:
            unpaired.append({"U_kn": kn, "U_a_mps": rnd(u_a), "nearest_U_b_mps": u_b,
                             "nearest_speed_match_rel_pct": rnd(d * 100, 4)})
            continue
        k_a = (rt / (0.5 * rho * u_a**2 * S)) / cf78(u_a)
        k_b = (rt_b / (0.5 * rho * u_b**2 * S)) / cf78(u_b)
        pairs.append({
            "U_kn": kn, "U_a_mps": rnd(u_a), "U_b_mps": u_b,
            "speed_match_rel_pct": rnd(d * 100, 4),
            "RT_a_N": rt, "RT_b_N": rt_b,
            "one_plus_k_A_T14": rnd(k_a),
            "one_plus_k_B_CFDletters": rnd(k_b),
            "rel_diff_pct": rnd(rel_pct(k_b, k_a), 4),
            "quotes": [{"id": row["id"], "file": row["file"], "line": row["line"]},
                       {"id": row_b["id"], "file": row_b["file"], "line": row_b["line"]}],
        })
    max_pair_diff = max(abs(p["rel_diff_pct"]) for p in pairs)

    dup_speeds = [float(x) for x in
                  re.findall(r"\d+\.\d+", q["T14_DUP_AFF1"]["text"].split("|")[2])]
    dup_forces = [float(x) for x in
                  re.findall(r"\d+\.\d+", q["T14_DUP_AFF1"]["text"].split("|")[3])]
    double_entry = {
        "speeds_match": [p["U_kn"] for p in points] == dup_speeds,
        "forces_match": [p["R_T_N"] for p in points] == dup_forces,
        "n_compared": min(len(dup_forces), len(points)),
        "max_rel_diff_pct": rnd(max(abs(rel_pct(a, b)) for a, b in
                                    zip([p["R_T_N"] for p in points], dup_forces)), 6),
        "sources": [q["T14_DUP_AFF1"]["locator"],
                    "src/tensorlbm/suboff_reference_data.py::SUBOFF_TOW_TANK_RESISTANCE_TABLE14"],
        "note": "同一 Table 14 的仓内第二处转录 vs 冻结常量表：逐点逐字相等 ⇒ 无转录漂移。",
    }
    c3b = {
        "cross_anchor": {
            "definition": ("(1+k)_A 由 DTRC Table 14 p.23 config 8 量纲力计算；"
                           "(1+k)_B 由 CFD Letters 2024 Table 3 EFD 列计算（同一水池试验的两次独立转录）；"
                           "同一 ν、同一湿面积口径 S=%g m²。" % S),
            "pairing_tolerance_pct": pair_tol_pct,
            "pairs": pairs,
            "unpaired": unpaired,
            "max_abs_rel_diff_pct": rnd(max_pair_diff, 4),
            "note_paired": ("CFD Letters Table 3 EFD 止于 8.230 m/s（≈16 kn），故 Table 14 的 17.79 kn 行无对偶转录，"
                            "按预声明容差 %.1f%% 判为 unpaired（不计入一致性）。" % pair_tol_pct),
        },
        "bare_hull_double_entry": double_entry,
        "threshold_pct": CRIT["c3_one_plus_k_consistency_max_pct"],
        "note": ("互算一致性度量『两次转录是否同源 + 口径是否一致』；它并不证明形式因子沿 Re 为常数 —— "
                 "后者由 C4 的相似性误差预算单独量化（ladder_one_plus_k_spread / E_k）。"),
    }
    c3b_pass = max_pair_diff <= CRIT["c3_one_plus_k_consistency_max_pct"]

    opk = [p["one_plus_k"] for p in points]
    ladder_spread = {
        "one_plus_k_by_point": {p["point_id"]: p["one_plus_k"] for p in points},
        "min": rnd(min(opk)), "max": rnd(max(opk)),
        "spread_pct_of_mean": rnd((max(opk) - min(opk)) / (sum(opk) / len(opk)) * 100, 4),
        "note": "逐点 (1+k)=C_T/C_F 沿 Re 并非常数；这是相似性换算误差预算 E_k 的经验来源。",
    }

    # ---- C4: reachability + compute --------------------------------------
    meas = json.loads(SRC_MLUPS.read_text(encoding="utf-8"))
    gci = json.loads(GCI_JSON.read_text(encoding="utf-8"))
    raw_mlups = meas["headline_d3q19_mrt_64cube_mlups"]
    prod = gci["cost_model"]["measured_ms_per_step_L96_full_domain"]
    prod_cells = float(re.search(r"\(([\d,]+) cells\)", prod["source"]).group(1).replace(",", ""))
    prod_mlups = prod_cells / (prod["ms_per_step"] / 1000.0) / 1e6
    u_lb = num(r"u_lb = ([\d.]+)", gci["ladder_protocol"]["Re_matching"])
    delta_coef = num(r"([\d.]+) \* Re \*\* (-?[\d.]+)", q["BLLAW_DELTA"]["text"])
    delta_exp = num(r"[\d.]+ \* Re \*\* (-?[\d.]+)", q["BLLAW_DELTA"]["text"])
    cells_coef = num(r'"cells": ([\d.]+) \* L\*\*3', q["CELLSLAW"]["text"])
    steps_coef = num(r'"steps_frozen_protocol": ([\d.]+) \* L', q["STEPSLAW"]["text"])
    yp_decl = re.search(r"\[(\d+),(\d+)\]", q["YPLUS_BAND"]["text"])
    yplus_decl_lo, yplus_decl_hi = float(yp_decl.group(1)), float(yp_decl.group(2))
    yp_cal = re.search(r"\[(\d+),(\d+)\]", q["YPLUS_CALIB"]["text"])
    yplus_cal_lo, yplus_cal_hi = float(yp_cal.group(1)), float(yp_cal.group(2))
    delta_cells_min = 20.0   # predeclared outer-layer resolution requirement delta/h >= 20

    rates = {"measured_raw_kernel_2node": 2 * raw_mlups * 1e6,
             "measured_production_2node": 2 * prod_mlups * 1e6}
    budget = CRIT["c4_evidence_budget_hours"]

    def requirement_N(re_: float, n: float) -> dict:
        cf = cf78(re_)
        utu = math.sqrt(cf / 2.0)
        dol = delta_coef * re_**delta_exp
        cells = cells_coef * n**3
        steps = steps_coef * n
        yplus = 0.5 * utu * re_ / n
        return {
            "N_cells_per_hull_length": rnd(n, 6),
            "h_m": rnd(L / n, 6),
            "h_over_L_denominator": rnd(n, 3),
            "cells": rnd(cells, 6),
            "steps": rnd(steps, 6),
            "lattice_updates": rnd(cells * steps, 8),
            "wall_hours": {k: rnd(cells * steps / v / 3600.0, 4) for k, v in rates.items()},
            "yplus_first_cell": rnd(yplus, 6),
            "cells_across_delta": rnd(dol * n, 4),
            "tau_minus_half": rnd(3.0 * u_lb * n / re_, 3),
            "inside_calibrated_yplus_band": bool(yplus_cal_lo <= yplus <= yplus_cal_hi),
            "inside_declared_operating_band": bool(yplus_decl_lo <= yplus <= yplus_decl_hi),
            "resolves_boundary_layer_20_cells": bool(dol * n >= delta_cells_min),
        }

    per_anchor = {}
    for p in points:
        re_ = p["Re_computed"]
        cf = cf78(re_)
        utu = math.sqrt(cf / 2.0)
        dol = delta_coef * re_**delta_exp
        n_cal = 0.5 * utu * re_ / yplus_cal_hi
        n_decl = 0.5 * utu * re_ / yplus_decl_hi
        n_wr = 0.5 * utu * re_
        n_d20 = delta_cells_min / dol
        n_cred = max(n_cal, n_d20)
        per_anchor[p["point_id"]] = {
            "Re": re_,
            "CF_ittc78": rnd(cf),
            "u_tau_over_U": rnd(utu),
            "delta_over_L": rnd(dol, 6),
            "delta_m": rnd(dol * L, 6),
            "N_for_calibrated_yplus_upper": rnd(n_cal, 6),
            "N_for_declared_operating_yplus_upper": rnd(n_decl, 6),
            "N_for_wall_resolved_yplus1": rnd(n_wr, 6),
            "N_for_20_cells_across_delta": rnd(n_d20, 6),
            "controlling_constraint": "delta/h>=20" if n_d20 > n_cal else "calibrated y+ band",
            "requirement_credible_wmles": requirement_N(re_, n_cred),
            "requirement_declared_operating_band_only": requirement_N(re_, n_decl),
            "requirement_wall_resolved": requirement_N(re_, n_wr),
        }

    head_re = points[0]["Re_computed"]
    cred = per_anchor[points[0]["point_id"]]["requirement_credible_wmles"]
    cred_h_raw = cred["wall_hours"]["measured_raw_kernel_2node"]
    cred_h_prod = cred["wall_hours"]["measured_production_2node"]
    reachable = bool(cred_h_raw <= budget)
    last = per_anchor[points[-1]["point_id"]]

    def finest_within_budget(rate: float) -> dict:
        n_exec = (budget * 3600.0 * rate / (cells_coef * steps_coef)) ** 0.25
        d = requirement_N(head_re, n_exec)
        d["wall_hours"] = rnd(d["cells"] * d["steps"] / rate / 3600.0, 4)
        d["verdict"] = ("executable in budget but NOT credible "
                        "(outside calibrated y+ band and delta/h<20)"
                        if not (d["inside_calibrated_yplus_band"]
                                and d["resolves_boundary_layer_20_cells"])
                        else "executable and credible")
        return d

    exec_design = {k: finest_within_budget(v) for k, v in rates.items()}

    c4 = {
        "predeclared_decision_rule": (
            "『可达』= 分辨率同时满足：(i) 首层格心 y+∈[%g,%g]（壁面律标定带，引句 YPLUS_CALIB）"
            " 且 (ii) δ/h≥%g（边界层≥20 格）：在此『可信分辨率』下，单锚在 %.0f h（两级 128 核集群预算）"
            "内跑完。只算『算得动』不算可达。"
            % (yplus_cal_lo, yplus_cal_hi, delta_cells_min, budget)),
        "two_bands_disclosed": {
            "calibrated_band_used_as_gate": [yplus_cal_lo, yplus_cal_hi],
            "calibrated_band_source": q["YPLUS_CALIB"]["locator"],
            "declared_operating_band_NOT_used_as_gate": [yplus_decl_lo, yplus_decl_hi],
            "declared_operating_band_source": q["YPLUS_BAND"]["locator"],
            "why": ("预注册的操作带 [%g,%g] 只是『壁模型仍能闭合』的外沿，不是精度带；"
                    "在锚 Re 下取 y+≤%g 只需 N≈%.0f 格/艇长，此时 δ/h≈%.2f 格（连一层边界层都装不下）"
                    "—— 用它当可信判据会把『算得动』误当『可信』。"
                    % (yplus_decl_lo, yplus_decl_hi, yplus_decl_hi,
                       per_anchor[points[0]["point_id"]]["N_for_declared_operating_yplus_upper"],
                       per_anchor[points[0]["point_id"]]["requirement_declared_operating_band_only"]
                       ["cells_across_delta"])),
        },
        "cost_model": {
            "cells": f"{cells_coef} * N^3   （N = 每艇长格数；引句 CELLSLAW，冻结域 padding 1/4/1/1/1/1）",
            "steps": f"{steps_coef} * N      （引句 STEPSLAW，冻结统计窗协议）",
            "yplus_first_cell": "0.5 * u_tau_lb / nu_lb = 0.5 * (u_tau/U) * Re / N",
            "delta_over_L": f"{delta_coef} * Re^({delta_exp})   （引句 BLLAW_DELTA）",
            "u_tau_over_U": "sqrt(C_F/2)，C_F 取 ITTC-78 复算值",
            "u_lb": f"{u_lb}（gci_v4.json::ladder_protocol.Re_matching，晶格来流速度）",
            "throughput": {
                "measured_raw_kernel_2node_MLUPS": rnd(2 * raw_mlups, 4),
                "measured_production_2node_MLUPS": rnd(2 * prod_mlups, 4),
                "raw_kernel": {
                    "file": "benchmarks/verified/suboff_anchor/sources/cluster_mlups_measurement.json",
                    "job_id": meas["cluster_job_id"], "node": meas["node"],
                    "command": meas["command"], "selection": meas["selection_rule"],
                    "two_node_scaling": "x2 for n315 + n330 (128 cores each)",
                },
                "production": {
                    "file": "benchmarks/verified/suboff_sail_resistance/gci_v4.json",
                    "key": "cost_model.measured_ms_per_step_L96_full_domain",
                    "cells": prod_cells, "ms_per_step": prod["ms_per_step"],
                    "derived_MLUPS_per_node": rnd(prod_mlups, 4),
                    "note": "整解器路径（含壁模型/观测器/sponge）在 99 控制面单节点实测",
                },
            },
        },
        "per_anchor": per_anchor,
        "finest_executable_grid_within_budget": exec_design,
        "binary_conclusion": {
            "anchor_Re_lowest": head_re,
            "credible_wmles_N": cred["N_cells_per_hull_length"],
            "credible_wmles_cells": cred["cells"],
            "credible_wmles_hours_optimistic_rate": cred_h_raw,
            "credible_wmles_hours_production_rate": cred_h_prod,
            "budget_hours": budget,
            "REACHABLE_AT_CREDIBLE_RESOLUTION": reachable,
            "verdict": "NOT REACHABLE at credible resolution" if not reachable
                       else "REACHABLE within budget",
            "statement": (
                "锚 Re=%.4e 的可信分辨率（y+≤%g 且 δ/h≥%g）需 N≈%.0f 格/艇长（h≈%.2f mm）、%.3e 胞："
                "实测原始核 %.2f MLUPS/节点（2 节点 %.1f MLUPS）需 %.3e h；生产求解器实测 %.3f MLUPS/节点"
                "（2 节点 %.2f MLUPS）需 %.3e h —— 分别是 %.0f h 预算的 %.0f× 与 %.0f×。⇒ 锚 Re 在可信分辨率下不可达。"
                "（对照：若只按预注册操作带 y+≤%g 取网格，N≈%.0f、δ/h≈%.2f 格，能算但毫无分辨率意义——见 two_bands_disclosed。）"
                % (head_re, yplus_cal_hi, delta_cells_min, cred["N_cells_per_hull_length"],
                   cred["h_m"] * 1000.0, cred["cells"], raw_mlups, 2 * raw_mlups, cred_h_raw,
                   prod_mlups, 2 * prod_mlups, cred_h_prod, budget,
                   cred_h_raw / budget, cred_h_prod / budget,
                   yplus_decl_hi,
                   per_anchor[points[0]["point_id"]]["N_for_declared_operating_yplus_upper"],
                   per_anchor[points[0]["point_id"]]["requirement_declared_operating_band_only"]
                   ["cells_across_delta"])),
            "measured_evidence_that_coarse_anchor_Re_runs_are_meaningless": {
                "file": q["YPLUS_STARVED"]["file"], "line": q["YPLUS_STARVED"]["line"],
                "text": q["YPLUS_STARVED"]["text"],
                "reading": "仓内实测：锚 Re 档在 y+ 均值 1878–4660（远超标定带）给出最差 C_T 误差 2962.7% —— 即『算得动』的欠解析档不能用于任何精度声明。",
            },
        },
        "path_b_external_wmles_practice": None,   # filled below
        "redesigned_ladder": {
            "as_issued": "6 个 Re 锚（%.3e … %.3e）× 3 档网格（N=96/112/128）—— #8854 实测 61–144 h/档"
                         % (head_re, last["Re"]),
            "finest_executable_design": (
                "预算内最细档（实测原始核率，N≈%.0f，h≈%.1f mm，L/h≈%.0f，y+≈%.0f，δ/h≈%.1f，τ−0.5≈%.1e）：%.0f h。"
                "该档在预算内但不可信（y+ 超出标定带、δ/h<20）—— 只能用于『库路径连通性/稳定性』验证，不能进精度判定。"
                % (exec_design["measured_raw_kernel_2node"]["N_cells_per_hull_length"],
                   exec_design["measured_raw_kernel_2node"]["h_m"] * 1000.0,
                   exec_design["measured_raw_kernel_2node"]["h_over_L_denominator"],
                   exec_design["measured_raw_kernel_2node"]["yplus_first_cell"],
                   exec_design["measured_raw_kernel_2node"]["cells_across_delta"],
                   exec_design["measured_raw_kernel_2node"]["tau_minus_half"],
                   exec_design["measured_raw_kernel_2node"]["wall_hours"])),
            "executable_plan_next": [
                "①把『6 锚 × 3 档』砍成『1 锚（Re=%.4e）+ 1 档』；按仓内标定带立档需 N≈%.0f，"
                "在当前实测率下不可达（需 %.0f× 提速），故立档路径只能是 Path B（外部实践 1e8 胞级 WMLES）"
                "＋ 先补齐高 y+ 带的壁模型验证；" %
                (head_re, cred["N_cells_per_hull_length"], cred_h_raw / budget),
                "②短期只用『锚 Re 附近 ≤2× 的 Re 敏感性』（同网格抬/降 Re）报告 Re 斜率，不声称绝对阻力精度；",
                "③任何跨 Re 的换算必须附 similarity_error_budget（E 已 >3%，见下节），不得裸对比；",
                "④高 Re 端（%.3e）在同网格下 δ/h=%.1f 格，物理上不可解析，必须从阶梯中移除；" %
                (last["Re"], last["delta_over_L"] * 128),
                "⑤航速阶梯本身可保留为『判据锚』（本文件 anchor_points，C1–C3 已验证），"
                "被砍掉的是『逐锚可信模拟』—— 锚可用不等于算得动，这一点必须在报告里分开写。",
            ],
            "throughput_gap_x": rnd(cred_h_prod / budget, 4),
        },
    }

    # ---- C4b: similarity (same Fr / different Re) error budget -----------
    def similarity_budget(re_s: float) -> dict:
        re_a = head_re
        k = head["one_plus_k"]
        cf_a, cf_s = cf78(re_a), cf78(re_s)
        ct_a = k * cf_a
        correction_pct = (ct_a - k * cf_s) / ct_a * 100.0
        e_line = abs(k * ((cf_schoenherr(re_a) - cf_schoenherr(re_s)) - (cf_a - cf_s))) / ct_a * 100.0
        return {
            "Re_sim": rnd(re_s),
            "Re_anchor": re_a,
            "CF_ittc78_sim": rnd(cf_s),
            "CF_ittc78_anchor": rnd(cf_a),
            "similarity_correction_pct_of_CT": rnd(correction_pct, 4),
            "E_friction_line_pct": rnd(e_line, 4),
        }

    loo = [abs(rel_pct(pi["one_plus_k"], pj["one_plus_k"]))
           for i, pi in enumerate(points) for j, pj in enumerate(points) if i != j]
    e_k = max(loo)
    sims = [similarity_budget(re_s) for re_s in (1e5, 1e6)]
    for s in sims:
        s["E_form_factor_single_constant_pct"] = rnd(e_k, 4)
        s["E_total_quantified_pct"] = rnd(
            math.sqrt(s["E_friction_line_pct"] ** 2 + e_k**2), 4)
        s["supports_band_pct"] = bool(s["E_total_quantified_pct"] <= CRIT["band_pct"])
    g = 9.80665
    fr_head = head["U_mps"] / math.sqrt(g * L)
    c4b = {
        "formula": (
            "单一形状因子 + 摩擦线外推：C_T(Re) = (1+k)·C_F(Re) ⇒ 由模拟点 Re_s 换算到锚 Re_a 的修正量 "
            "ΔC_T = (1+k)·[C_F(Re_a) − C_F(Re_s)]；相对误差预算 E = sqrt(E_line² + E_k²)（仅可量化项）。"
            "E_line = |(1+k)[(C_F^Sch(Re_a) − C_F^Sch(Re_s)) − (C_F^ITTC(Re_a) − C_F^ITTC(Re_s))]| / C_T(Re_a)；"
            "E_k = max_{i≠j} |(1+k)_i/(1+k)_j − 1|（锚阶梯上的单常数形式因子预测残差，经验上界）。"),
        "froude_note": ("SUBOFF 为深潜体（引句 SUBMERGED：拖曳水池深潜阻力，无自由面兴波）⇒ Froude 相似性不是约束；"
                        "最低锚 Fr=U/sqrt(gL)=%.3f 仅作记录，Re 是唯一相似参数。" % fr_head),
        "term_E_line_source": (f"两条独立摩擦线：ITTC-78 `{cf_coef}/(log10Re−{cf_shift})^{cf_exp}`"
                               f"（引句 CF_ITTC78）与 Schoenherr `{sch_coef}/(log10Re)^{sch_exp}`"
                               "（引句 CF_SCHOENHERR）"),
        "term_E_k_source": "锚阶梯 6 点逐点 (1+k)（ladder_one_plus_k_spread）",
        "term_unquantified": [
            "试验重复性（拖曳水池典型 ±2–5%）：原报告不确定度声明离线不可达 ⇒ WITHHELD，未计入",
            "低 Re 模拟侧的转捩/分离形态与锚 Re 不同（形状阻力机理差）：无可引用量化 ⇒ WITHHELD",
        ],
        "evaluations": sims,
        "conclusion": (
            "最有利的模拟点 Re_s=%.0e 仍有 %.2f%% 的量化误差（= %.2f× 目标带），Re_s=%.0e 达 %.2f%%（= %.2f×）"
            " ⇒ 相似性换算不能支撑「<%g%%」判定；<%g%% 的对比必须在锚 Re 的 ≤2× 邻域直接模拟。"
            % (sims[1]["Re_sim"], sims[1]["E_total_quantified_pct"],
               sims[1]["E_total_quantified_pct"] / CRIT["band_pct"],
               sims[0]["Re_sim"], sims[0]["E_total_quantified_pct"],
               sims[0]["E_total_quantified_pct"] / CRIT["band_pct"],
               CRIT["band_pct"], CRIT["band_pct"])),
    }
    # Path B: apply the externally practised WMLES cell budget (in-repo quote) instead of the
    # repo's calibrated y+ band, to separate "compute blocker" from "wall-model validation blocker".
    cells_ext = num(r"([\d.]+)×10⁸ 胞", q["N_CELLS_EXTERNAL"]["text"]) * 1e8
    n_ext = (cells_ext / cells_coef) ** (1.0 / 3.0)
    path_b = requirement_N(head_re, n_ext)
    path_b.update({
        "source": {"file": q["N_CELLS_EXTERNAL"]["file"], "line": q["N_CELLS_EXTERNAL"]["line"],
                   "text": q["N_CELLS_EXTERNAL"]["text"], "grade": q["N_CELLS_EXTERNAL"]["grade"]},
        "cells_external_practice": cells_ext,
        "yplus_band_vs_calibrated": ("outside" if not path_b["inside_calibrated_yplus_band"]
                                     else "inside"),
        "executable_at_raw_kernel_rate": bool(
            path_b["wall_hours"]["measured_raw_kernel_2node"] <= budget),
        "verdict": ("可达（仅算力层面）：原始核率下 %.0f h ≤ %g h；但 y+≈%.0f 在仓内标定带 [%g,%g] 之外，"
                    "需一个在 y+ 30–500 已被独立验证的壁模型才能支撑 3%% 判定 —— 这是『壁模型验证』缺口，"
                    "不是纯算力缺口。生产求解器率下 %.0f h。"
                    % (path_b["wall_hours"]["measured_raw_kernel_2node"], budget,
                       path_b["yplus_first_cell"], yplus_cal_lo, yplus_cal_hi,
                       path_b["wall_hours"]["measured_production_2node"])),
        "wall_model_gap": ("仓内唯一实测：y+ 均值 1878–4660 的锚 Re 档最差 C_T 误差 2962.7%"
                           "（引句 YPLUS_STARVED）⇒ 现有壁模型未在高 y+ 带验证，Path B 暂不可用。"),
    })
    c4["path_b_external_wmles_practice"] = path_b

    c4_pass = bool(c4["binary_conclusion"]["verdict"]) and bool(c4b["evaluations"])

    # ---- convention-chain uncertainty (anchor-side resolvability) --------
    d_rho = (rho_hi - rho_lo) / 2.0 / rho * 100.0
    d_speed = (0.005 / points[0]["U_kn"]) * 2 * 100.0
    d_force = (0.005 / points[0]["R_T_N"]) * 100.0
    d_s = abs(rel_pct(analytic_s_coef * L**2, S))
    m_nu = num(r"m=(−?[\d.]+)", q["NU_SENS"]["text"])
    d_nu = (nu_20 - nu_15) / nu * 100.0
    d_nu_ct = abs(m_nu) * d_nu
    rss = math.sqrt(d_rho**2 + d_speed**2 + d_force**2 + d_s**2 + d_nu_ct**2)
    convention = {
        "terms_pct": {
            "rho_band_997_1000": rnd(d_rho, 4),
            "speed_last_digit_rounding": rnd(d_speed, 4),
            "force_last_digit_rounding": rnd(d_force, 4),
            "wetted_surface_convention": rnd(d_s, 4),
            "nu_assumption_15_20C_via_m": rnd(abs(d_nu_ct), 4),
        },
        "rss_total_pct": rnd(rss, 4),
        "band_pct": CRIT["band_pct"],
        "conclusion": ("锚侧口径链 RSS=%.3f%% ≪ %g%% 目标带 ⇒ 3%% 带在锚侧可分辨；"
                       "试验重复性未量化（WITHHELD），任何 <3%% 声明必须写成『在声明口径下』。"
                       % (rss, CRIT["band_pct"])),
        "nu_sensitivity_quote": {"file": q["NU_SENS"]["file"], "line": q["NU_SENS"]["line"],
                                 "text": q["NU_SENS"]["text"]},
    }

    # ---- C5: AST no-hand-copy self-check ---------------------------------
    literals = [n.value for n in ast.walk(ast.parse(Path(__file__).read_text(encoding="utf-8")))
                if isinstance(n, ast.Constant) and isinstance(n.value, float)]
    derived = ([p["CT"] for p in points] + [p["CF_ittc78"] for p in points]
               + [p["Re_computed"] for p in points] + [p["one_plus_k"] for p in points]
               + [v["requirement_credible_wmles"]["h_m"] for v in per_anchor.values()])
    violations = [{"derived": rnd(v, 8), "literal": lit}
                  for v in derived for lit in literals
                  if lit != 0.0 and abs(lit - v) <= 1e-3 * abs(v)]
    if violations:
        raise SystemExit(f"[no-hand-copy] derived values typed as literals: {violations}")
    c5 = {"n_float_literals": len(literals), "n_derived_checked": len(derived),
          "violations": violations, "pass": True,
          "rule": "脚本内无浮点字面量落在任一派生量 ±0.1% 相对带内"}

    criteria = [
        {"id": "C1", "requirement": REQUIREMENTS["C1"], "measured": c1,
         "verdict": "PASS" if c1_pass else "FAIL"},
        {"id": "C2", "requirement": REQUIREMENTS["C2"],
         "measured": {"max_abs_re_rel_dev_pct": c2["max_abs_re_rel_dev_pct"],
                      "threshold_pct": c2["threshold_pct"], "n_points": len(points)},
         "verdict": "PASS" if c2_pass else "FAIL"},
        {"id": "C3a", "requirement": REQUIREMENTS["C3"].split("；")[0],
         "measured": {"max_abs_residual": c3a["max_abs_residual"],
                      "threshold": CRIT["c3_cf_residual_max_abs"]},
         "verdict": "PASS" if c3a_pass else "FAIL"},
        {"id": "C3b", "requirement": REQUIREMENTS["C3"].split("；")[1],
         "measured": {"max_abs_rel_diff_pct": c3b["cross_anchor"]["max_abs_rel_diff_pct"],
                      "threshold_pct": c3b["threshold_pct"]},
         "verdict": "PASS" if c3b_pass else "FAIL"},
        {"id": "C4", "requirement": REQUIREMENTS["C4"],
         "measured": {"binary_verdict": c4["binary_conclusion"]["verdict"],
                      "budget_hours": c4["binary_conclusion"]["budget_hours"],
                      "similarity_E_total_pct": [s["E_total_quantified_pct"] for s in sims]},
         "verdict": "PASS" if c4_pass else "FAIL"},
        {"id": "C5", "requirement": REQUIREMENTS["C5"], "measured": c5,
         "verdict": "PASS" if c5["pass"] else "FAIL"},
    ]
    overall = "PASS" if all(c["verdict"] == "PASS" for c in criteria) else "FAIL"

    return {
        "schema": "suboff-anchor-points/v1",
        "case": "DARPA SUBOFF Model 4740 — AFF-1 裸艇 C_T(U) 锚点表（逐字落盘 + ITTC-78 复算 + 可执行设计）",
        "task_key": TASK_KEY,
        "mission_id": MISSION_ID,
        "generated_by": "benchmarks/verified/suboff_anchor/build_anchor_points.py",
        "generated_for": "research_plan 第 A 项（锚逐字核销补全）—— 承 #8800 C1 PARTIAL_FAIL / #8854 锚 Re 阶梯不可执行",
        "overall_verdict": overall,
        "criteria_summary": {c["id"]: c["verdict"] for c in criteria},
        "criteria_predeclared": {
            "thresholds": CRIT,
            "yplus_band_declared_operating": [yplus_decl_lo, yplus_decl_hi],
            "yplus_band_calibrated": [yplus_cal_lo, yplus_cal_hi],
            "delta_cells_min": delta_cells_min,
            "note": "判据与门槛在动手前固定于脚本常量；本文件所有 measured 值由脚本注入，禁改判据。",
        },
        "criteria": criteria,
        "conventions": {
            "CT_definition": "C_T = R_T / (0.5 · rho · U^2 · S)（引句 CTDEF）",
            "reynolds_definition": "Re = U · L / nu，L = L_max（引句 REDEF）",
            "rho_kg_m3": rho,
            "rho_band_kg_m3": [rho_lo, rho_hi],
            "nu_m2_s": nu,
            "nu_basis": "淡水（拖曳水池），仓内冻结参考锁的 15 °C 声明假设；ν(15 °C)/ν(20 °C) 两点见引句 NU_T_ANCHORS",
            "L_m": L,
            "S_wet_m2": S,
            "S_wet_basis": ("实验口径湿表面积（CFD Letters 2024 Table 1 WSA，引句 GEO_S）；"
                            "与解析回转面 %.5f·L²=%.4f m² 差 %.2f%%"
                            % (analytic_s_coef, analytic_s_coef * L**2, d_s)),
            "knot_to_mps": KN,
            "knot_to_mps_note": "按定义 1852/3600 m/s（精确），与仓内 0.514444 差 ~1e-6 相对，不影响判据",
            "test_conditions": [
                "模型尺度（L_max=4.356 m）拖曳水池总阻力，深潜（无自由面兴波，引句 SUBMERGED）",
                "水温/水质：原报告记录不在离线可达范围 ⇒ ν 为声明假设（15 °C，影响经 C_F 斜率传递）",
                "试验不确定度声明：离线不可达 ⇒ WITHHELD，未计入任何误差预算",
            ],
            "ittc78_cf": {"coefficient": cf_coef, "log10_shift": cf_shift, "exponent": cf_exp,
                          "source": q["CF_ITTC78"]["locator"]},
            "schoenherr_cf": {"coefficient": sch_coef, "exponent": sch_exp,
                              "source": q["CF_SCHOENHERR"]["locator"]},
        },
        "anchor_points": points,
        "anchor_points_completeness": c1,
        "reynolds_consistency": c2,
        "ladder_one_plus_k_spread": ladder_spread,
        "ittc78_cf_recompute": c3a,
        "one_plus_k_consistency": c3b,
        "reachability": c4,
        "similarity_error_budget": c4b,
        "conventions_uncertainty": convention,
        "no_hand_copy": c5,
        "external_quotes": {
            "n_quotes": quote_report["n_quotes"],
            "hit_rate_pct": quote_report["hit_rate_pct"],
            "rule": quote_report["rule"],
            "offline_disclosure": trans.get("offline_disclosure"),
            "quotes": q,
        },
        "provenance": {
            "repo": "TensorLBM",
            "script": "benchmarks/verified/suboff_anchor/build_anchor_points.py",
            "transcriptions": "benchmarks/verified/suboff_anchor/sources/anchor_points_transcriptions.json",
            "transcriptions_sha256": sha256(SRC_TRANS),
            "cluster_measurement": "benchmarks/verified/suboff_anchor/sources/cluster_mlups_measurement.json",
            "cluster_measurement_sha256": sha256(SRC_MLUPS),
            "figure": "benchmarks/verified/suboff_anchor/figures/anchor_points_overview.png",
            "test": "tests/test_suboff_anchor_points.py",
            "regenerate": "python3 benchmarks/verified/suboff_anchor/build_anchor_points.py",
            "environment": {"python": __import__("sys").version.split()[0], "numpy_free": True},
            "upstream_artifacts_reused": [
                "src/tensorlbm/suboff_reference_data.py::SUBOFF_TOW_TANK_RESISTANCE_TABLE14（Table 14 常量）",
                "docs/survey/IN25_T6_external_mature_tech_survey.md 表 2（4 位有效数字 Re 阶梯）",
                "benchmarks/pending/suboff_highre_resistance/refs_lock.md（几何/口径/CFD Letters Table 3）",
                "benchmarks/verified/suboff_sail_resistance/gci_v4.json（成本模型、冻结协议、整解器吞吐）",
                "sources/cluster_mlups_measurement.json（本单新测：loong64 集群原始核吞吐）",
            ],
        },
        "downstream_contract": {
            "use_this_as": "LBM 阻力对比的唯一外部锚：anchor_points[].CT / Re / CF_ittc78 / one_plus_k（6 点 AFF-1 阶梯）",
            "error_definition": "err% = (C_T,sim − C_T,anchor)/C_T,anchor × 100，参考面积 S=5.998 m² 与 ν=1.139e-6 m²/s 必须与 conventions 一致",
            "do_not_use": [
                "Re=1.2e7 两位有效数字锚（见 c2.retired_two_sig_fig_anchor，已 RETIRED）",
                "任何未经 external_quotes 逐字转录的 C_T / Re",
                "跨湿面积口径混用（AFF-8 用 6.35 m² 时须换算）",
            ],
            "claim_guard": (
                "在锚 Re 直接模拟前，任何『<3%%』声明要么标『欠解析网格（y+≈%g）』，要么必须附 similarity_error_budget；"
                "相似性换算路径的量化误差已 > 3%%（见 similarity_error_budget.conclusion）。"
                % exec_design["measured_raw_kernel_2node"]["yplus_first_cell"]),
        },
    }


# --------------------------------------------------------------------------- #
def plot(art: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    conv = art["conventions"]
    pts = art["anchor_points"]
    re_list = [p["Re_computed"] for p in pts]
    ct_list = [p["CT"] for p in pts]
    k_list = [p["one_plus_k"] for p in pts]
    k_head = ct_list[0] / pts[0]["CF_ittc78"]
    coef = conv["ittc78_cf"]["coefficient"]
    shift = conv["ittc78_cf"]["log10_shift"]
    expo = conv["ittc78_cf"]["exponent"]

    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.0))

    # (1) C_T(Re) ladder vs ITTC-78 friction line
    ax = axes[0]
    re_grid = [10 ** (6.8 + 0.02 * i) for i in range(90)]
    cf_grid = [coef / (math.log10(r) - shift) ** expo for r in re_grid]
    ax.loglog(re_grid, cf_grid, "k--", lw=1.3, label=f"ITTC-78 $C_F$")
    ax.loglog(re_grid, [c * k_head for c in cf_grid], "--", color="#1f77b4", lw=1.1,
              label=f"$(1+k)C_F$, $1+k$={k_head:.4f} (lowest anchor)")
    ax.loglog(re_list, ct_list, "o-", color="#d62728", ms=7, lw=1.4,
              label="AFF-1 measured $C_T$ (DTRC Table 14 p.23)")
    for p in pts:
        ax.annotate(f"{p['U_kn']:g} kn", (p["Re_computed"], p["CT"]),
                    textcoords="offset points", xytext=(6, -10), fontsize=7.5)
    ret = art["c2"]["retired_two_sig_fig_anchor"] if "c2" in art else None
    ax.set_xlabel("Re = U·L/ν")
    ax.set_ylabel("$C_T$, $C_F$")
    ax.set_title("① 6 AFF-1 anchor points vs ITTC-78 friction line")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")

    # (2) (1+k) = C_T / C_F along the ladder
    ax2 = axes[1]
    ax2.semilogx(re_list, k_list, "s-", color="#2ca02c", ms=7, label="pointwise $(1+k)=C_T/C_F$")
    mean_k = sum(k_list) / len(k_list)
    ax2.axhline(mean_k, color="#7f7f7f", ls=":", lw=1.2,
                label=f"ladder mean {mean_k:.4f}")
    spread = art["ladder_one_plus_k_spread"]["spread_pct_of_mean"]
    ax2.axhspan(min(k_list), max(k_list), color="#2ca02c", alpha=0.08,
                label=f"ladder spread {spread:.2f}% of mean")
    for p, k in zip(pts, k_list):
        ax2.annotate(f"{k:.4f}", (p["Re_computed"], k), textcoords="offset points",
                     xytext=(6, 6), fontsize=7.5)
    ax2.set_xlabel("Re = U·L/ν")
    ax2.set_ylabel("$(1+k)$")
    ax2.set_title("② form factor is NOT constant along Re (E$_k$ source)")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=8, loc="lower right")

    # (3) compute cost vs resolution
    ax3 = axes[2]
    reach = art["reachability"]
    cells_coef = float(reach["cost_model"]["cells"].split("*")[0].strip())
    steps_coef = float(reach["cost_model"]["steps"].split("*")[0].strip())
    tp = reach["cost_model"]["throughput"]
    rates = {"raw kernel (measured, 2 nodes)": tp["measured_raw_kernel_2node_MLUPS"] * 1e6,
             "production solver (measured, 2 nodes)": tp["measured_production_2node_MLUPS"] * 1e6}
    n_grid = [40 * (1.12 ** i) for i in range(28)]
    for (label, rate), color in zip(rates.items(), ("#1f77b4", "#d62728")):
        hours = [cells_coef * n**3 * steps_coef * n / rate / 3600.0 for n in n_grid]
        ax3.loglog(n_grid, hours, "-", color=color, lw=1.6, label=label)
    ax3.axhline(72.0, color="#2ca02c", ls="--", lw=1.3, label="budget 72 h / anchor")
    for key, color in (("measured_raw_kernel_2node", "#1f77b4"),
                       ("measured_production_2node", "#d62728")):
        d = reach["finest_executable_grid_within_budget"][key]
        ax3.plot([d["N_cells_per_hull_length"]], [d["wall_hours"]], "o", color=color, ms=8)
        ax3.annotate(f"N≈{d['N_cells_per_hull_length']:.0f}\nh={d['h_m']*1000:.1f} mm\n"
                     f"y$^+$≈{d['yplus_first_cell']:.0f}",
                     (d["N_cells_per_hull_length"], d["wall_hours"]),
                     textcoords="offset points", xytext=(-10, 12), fontsize=7.5, color=color)
    n_cred = reach["per_anchor"][pts[0]["point_id"]]["requirement_credible_wmles"]
    ax3.axvline(n_cred["N_cells_per_hull_length"], color="#9467bd", ls=":", lw=1.4,
                label=f"credible WMLES $y^+\\leq${art['criteria_predeclared']['yplus_band_calibrated'][1]:g}"
                      f" (N≈{n_cred['N_cells_per_hull_length']:.0f})")
    ax3.set_xlabel("N = cells per hull length  (h = L/N)")
    ax3.set_ylabel("wall hours per anchor")
    ax3.set_title("③ anchor-Re cost: credible resolution is out of budget")
    ax3.grid(True, which="both", alpha=0.3)
    ax3.legend(fontsize=7.5, loc="upper left")

    fig.suptitle(f"paper-exp lbm-suboff-tau A — SUBOFF anchor points ({art['overall_verdict']})",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT_FIG, dpi=140)
    plt.close(fig)


def main() -> int:
    art = build()
    blob = json.dumps(art, indent=2, ensure_ascii=False) + "\n"
    OUT_JSON.write_text(blob, encoding="utf-8")
    plot(art)
    print(json.dumps(art["criteria_summary"], indent=1, ensure_ascii=False))
    print("overall:", art["overall_verdict"])
    for p in art["anchor_points"]:
        print(f"  {p['point_id']:<14} Re={p['Re_computed']:.6e}  CT={p['CT']:.6e}  "
              f"CF={p['CF_ittc78']:.6e} (1+k)={p['one_plus_k']:.5f}  dRe={p['re_rel_dev_pct']:+.4f}%")
    print("reachability:", art["reachability"]["binary_conclusion"]["verdict"])
    print("similarity:", [s["E_total_quantified_pct"] for s in
                          art["similarity_error_budget"]["evaluations"]], "%")
    print("wrote", OUT_JSON.relative_to(REPO), "and", OUT_FIG.relative_to(REPO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
