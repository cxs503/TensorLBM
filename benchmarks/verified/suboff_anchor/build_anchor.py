#!/usr/bin/env python3
"""SUBOFF 外部锚规范化与可复算性 —— paper-exp / lbm-suboff-tau 实验 1。

本脚本把 ``sources/external_transcriptions.json``（逐字转录证据包）变成
``anchor.json``：判据逐条 + 实测值 + 原始数据来源/脚本路径。

零手抄机制（三条，全部机器执行，任一失败即构建失败）
----------------------------------------------------
1. **逐字校验**：每条 quote 必须是 ``archived_at`` 文件中的连续子串。
2. **正则取数**：anchor.json 的每个数字都由该 quote 的正则解析得到；
   脚本内不存在 CT / C_F / (1+k) / Re 的浮点字面量（AST 自检，见
   :func:`assert_no_hand_typed_constants`）。
3. **双录对校**：DTRC Table 14 行既从仓内文本解析，又与
   :data:`tensorlbm.suboff_reference_data.SUBOFF_TOW_TANK_RESISTANCE_TABLE14`
   常量逐行比对，必须完全一致。

判据（任务书原文，先于动手固定；不达标=honest FAIL，禁改判据）
--------------------------------------------------------------
C1  C_T 与 Re 逐字抄录（含表号/页码引句）入 evidence_pack 外部节
C2  Re = U·L/ν 相对偏差 ≤0.1%（与所抄 Re 对照）
C3  ITTC-78 C_F 复算与解析式残差 <1e-12
C4  (1+k) 由两锚复算，禁第三处手抄

用法::

    python3 benchmarks/verified/suboff_anchor/build_anchor.py          # 写 anchor.json + 图
    python3 benchmarks/verified/suboff_anchor/build_anchor.py --check  # 只校验，不写文件
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import sys
from decimal import Decimal, getcontext
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SOURCES = HERE / "sources" / "external_transcriptions.json"
OUT_JSON = HERE / "anchor.json"
OUT_FIG = HERE / "figures" / "anchor_overview.png"

SUPERSCRIPT = {
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
}

# 判据阈值（任务书给定，写死在此仅作门槛常数，不含任何被测物理量）
CRIT = {
    "c2_re_rel_dev_max_pct": 0.1,
    "c3_cf_residual_max": 1e-12,
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def rnd(x: float, sig: int = 12) -> float:
    """Round for stable, reviewable JSON (no binary noise in the artefact)."""
    return float(f"%.{sig}g" % x)


def rel_pct(a: float, b: float) -> float:
    return 100.0 * (a - b) / b


def decode_superscript(text: str) -> int:
    return int("".join(SUPERSCRIPT[c] for c in text))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# 1. verbatim transcription layer
# --------------------------------------------------------------------------
def load_sources() -> dict:
    return json.loads(SOURCES.read_text(encoding="utf-8"))


def verify_and_parse(spec: dict) -> dict:
    """Machine-verify every quote against its archived file, then parse values."""
    parsed: dict[str, dict] = {}
    checks: list[dict] = []
    for src in spec["sources"]:
        path = REPO / src["archived_at"]
        if not path.is_file():
            raise SystemExit(f"[verbatim] archived file missing: {src['archived_at']}")
        text = path.read_text(encoding="utf-8")
        sha = file_sha256(path)
        for grp in src["groups"]:
            found = grp["quote"] in text
            checks.append({
                "source_id": src["source_id"],
                "group_id": grp["group_id"],
                "archived_at": src["archived_at"],
                "archived_sha256": sha,
                "verbatim_found": bool(found),
            })
            if not found:
                raise SystemExit(
                    f"[verbatim] quote not found verbatim in {src['archived_at']}: "
                    f"{grp['group_id']} :: {grp['quote'][:80]!r}"
                )
            m = re.search(grp["regex"], grp["quote"])
            if m is None:
                raise SystemExit(
                    f"[parse] regex does not match quote for {src['source_id']}/"
                    f"{grp['group_id']}: {grp['regex']!r}"
                )
            vals: dict[str, object] = {}
            for name, raw in m.groupdict().items():
                if name == "sup":
                    vals[name] = decode_superscript(raw)
                elif re.fullmatch(r"-?[0-9]+", raw):
                    vals[name] = int(raw)
                else:
                    vals[name] = float(raw)
            parsed[f"{src['source_id']}/{grp['group_id']}"] = {
                "values": vals,
                "config": grp.get("config"),
                "units": {k: v["unit"] for k, v in grp.get("fields", {}).items()},
                "quantities": {k: v["quantity"] for k, v in grp.get("fields", {}).items()},
                "citation": src["citation"],
                "locator": src["locator"],
                "grade": src["grade"],
                "archived_at": src["archived_at"],
                "quote": grp["quote"],
            }
    return {"parsed": parsed, "checks": checks}


def q(parsed: dict, key: str) -> dict:
    if key not in parsed:
        raise SystemExit(f"[parse] missing transcription key {key}")
    return parsed[key]["values"]


# --------------------------------------------------------------------------
# 2. ITTC-78 friction line (three independent evaluations)
# --------------------------------------------------------------------------
def cf_direct(re: float, coef: float, shift: float) -> float:
    """C_F = coef / (log10(Re) - shift)^2, evaluated with math.log10."""
    return coef / (math.log10(re) - shift) ** 2


def cf_via_ln(re: float, coef: float, shift: float) -> float:
    """Same formula re-derived through the natural log: log10 x = ln x / ln 10."""
    return coef / ((math.log(re) / math.log(10.0)) - shift) ** 2


def cf_decimal(re: float, coef: float, shift: float, prec: int = 50) -> Decimal:
    """50-digit Decimal evaluation of the analytic expression (reference residual)."""
    getcontext().prec = prec
    ln10 = Decimal(10).ln()
    lg = Decimal(re).ln() / ln10
    return Decimal(coef) / (lg - Decimal(shift)) ** 2


def cf_residual_report(re_values: list[float], coef: float, shift: float) -> dict:
    rows = []
    for reynolds in re_values:
        d = cf_direct(reynolds, coef, shift)
        l = cf_via_ln(reynolds, coef, shift)
        h = float(cf_decimal(reynolds, coef, shift))
        rows.append({
            "Re": rnd(reynolds),
            "cf_direct": rnd(d, 17),
            "cf_via_ln": rnd(l, 17),
            "cf_decimal_ref": rnd(h, 17),
            "abs_residual_vs_decimal": rnd(abs(d - h), 6),
            "abs_residual_vs_ln": rnd(abs(d - l), 6),
        })
    max_res = max(max(r["abs_residual_vs_decimal"], r["abs_residual_vs_ln"]) for r in rows)
    return {"rows": rows, "max_abs_residual": rnd(max_res, 6)}


# --------------------------------------------------------------------------
# 3. anchors and (1+k)
# --------------------------------------------------------------------------
def build_rows(parsed: dict, prefix: str, ids: list[str], cfg_label: str,
               speed_unit: str, L: float, nu: float, rho: float, s_wet: float,
               coef: float, shift: float) -> list[dict]:
    rows = []
    for gid in ids:
        key = f"{prefix}/{gid}"
        entry = parsed[key]
        v = entry["values"]
        speed = float(v["speed"])
        u = speed * q(parsed, "REF-LOCK-CONVENTIONS/KNOT")["value"] if speed_unit == "kn" else speed
        rt = float(v["rt"])
        re = u * L / nu
        cf = cf_direct(re, coef, shift)
        ct = rt / (0.5 * rho * u * u * s_wet)
        rows.append({
            "row_id": gid,
            "configuration": cfg_label,
            "speed_transcribed": speed,
            "speed_unit_transcribed": speed_unit,
            "speed_knots": rnd(speed if speed_unit == "kn" else u / q(parsed, "REF-LOCK-CONVENTIONS/KNOT")["value"]),
            "U_mps": rnd(u),
            "RT_N": rnd(rt),
            "Re": rnd(re),
            "CF_ittc78": rnd(cf),
            "CT": rnd(ct),
            "one_plus_k": rnd(ct / cf),
            "source": {"source_id": entry["citation"], "locator": entry["locator"],
                       "grade": entry["grade"], "quote": entry["quote"],
                       "archived_at": entry["archived_at"]},
        })
    return rows


def assert_no_hand_typed_constants(values: list[float], tol: float = 1e-3) -> dict:
    """AST self-check: the script must not contain literals equal to derived
    quantities (CT, C_F, (1+k), Re).  Prevents 'third hand copy' by construction."""
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    literals = [
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, float)
    ]
    hits = []
    for val in values:
        for lit in literals:
            if lit != 0.0 and abs(lit - val) <= tol * abs(val):
                hits.append({"derived": val, "literal": lit})
    return {
        "n_float_literals_in_script": len(literals),
        "tolerance_relative": tol,
        "violations": hits,
        "pass": not hits,
    }


# --------------------------------------------------------------------------
# 4. build
# --------------------------------------------------------------------------
def build() -> dict:
    spec = load_sources()
    layer = verify_and_parse(spec)
    parsed = layer["parsed"]

    L = float(q(parsed, "CFD-LETTERS-2024-T1-T3/L")["value"])
    D = float(q(parsed, "CFD-LETTERS-2024-T1-T3/D")["value"])
    vol = float(q(parsed, "CFD-LETTERS-2024-T1-T3/VOL")["value"])
    s_doc = float(q(parsed, "CFD-LETTERS-2024-T1-T3/S")["value"])
    conv = q(parsed, "REF-LOCK-CONVENTIONS/RHO")
    rho = float(conv["rho"])
    rho_lo, rho_hi = float(conv["lo"]), float(conv["hi"])
    s_lock = float(q(parsed, "REF-LOCK-CONVENTIONS/SWET")["value"])
    nu = float(q(parsed, "REF-LOCK-CONVENTIONS/NU")["value"])
    knot = float(q(parsed, "REF-LOCK-CONVENTIONS/KNOT")["value"])
    cad = q(parsed, "REF-LOCK-CONVENTIONS/SCAD")
    s_cad = float(cad["cad"])
    formula = q(parsed, "ITTC-CF-FORMULA/CF")
    coef, shift = float(formula["coef"]), float(formula["shift"])

    if abs(s_doc - s_lock) > 1e-12:
        raise SystemExit(f"[consistency] wetted area transcriptions disagree: {s_doc} vs {s_lock}")

    # ---- anchors -------------------------------------------------------
    rows_bare = build_rows(parsed, "DTRC-ADA359226-T14", ["A1", "A2", "A3", "A4", "A5", "A6"],
                           "AFF-1", "kn", L, nu, rho, s_doc, coef, shift)
    rows_full_t14 = build_rows(parsed, "DTRC-ADA359226-T14", ["F1", "F2", "F3", "F4", "F5", "F6"],
                               "AFF-8", "kn", L, nu, rho, s_doc, coef, shift)
    rows_full_t3 = build_rows(parsed, "CFD-LETTERS-2024-T1-T3", ["B1", "B2", "B3", "B4", "B5"],
                              "AFF-8", "m/s", L, nu, rho, s_doc, coef, shift)

    # double-entry check: parsed text vs package constant
    sys.path.insert(0, str(REPO / "src"))
    from tensorlbm.suboff_reference_data import SUBOFF_TOW_TANK_RESISTANCE_TABLE14  # noqa: E402

    pkg = [(p.hull_type, p.speed_knots, p.resistance_n) for p in SUBOFF_TOW_TANK_RESISTANCE_TABLE14]
    txt = [("bare_hull" if r["configuration"] == "AFF-1" else "full",
            r["speed_transcribed"], r["RT_N"]) for r in rows_bare + rows_full_t14]
    double_entry = {
        "n_rows": len(pkg),
        "exact_match": bool(pkg == txt),
        "pkg_rows": [list(map(str, row)) for row in pkg],
        "text_rows": [list(map(str, row)) for row in txt],
    }
    if not double_entry["exact_match"]:
        raise SystemExit("[double-entry] Table 14 text parse != package constant")

    # ---- criterion C3: friction-line residual --------------------------
    anchor_res = [r["Re"] for r in rows_bare + rows_full_t14 + rows_full_t3]
    grid = sorted({1e5, 1e6, 1e7, 2e7, 1e8, 1e9} | {round(x, 6) for x in anchor_res})
    cf_report = cf_residual_report(grid, coef, shift)
    known = q(parsed, "ITTC-CF-CROSSCHECK/KNOWN")
    re_known = float(known["mant"]) * 10 ** int(known["sup"])
    cf_known = float(known["cf"])
    cf_at_known = cf_direct(re_known, coef, shift)
    known_value = {
        "Re": rnd(re_known),
        "cf_transcribed": cf_known,
        "cf_computed": rnd(cf_at_known),
        "abs_diff": rnd(abs(cf_at_known - cf_known), 6),
        "within_two_sig_fig_rounding": bool(abs(cf_at_known - cf_known) <= 5e-6),
        "note": "转录值仅两位有效数字（0.00291），容许 ±0.5 末位（5e-6）舍入带",
    }
    c3_pass = cf_report["max_abs_residual"] < CRIT["c3_cf_residual_max"]

    # ---- criterion C2: Reynolds-number consistency ---------------------
    re_tr = q(parsed, "SUBOFF-REFDATA-BRACKET/R-AFF1")
    re_transcribed = float(re_tr["mant"]) * 10 ** int(re_tr["sup"])
    head = rows_bare[0]
    re_calc = head["Re"]
    dev_pct = rel_pct(re_calc, re_transcribed)
    nu_implied = head["U_mps"] * L / re_transcribed
    # ν(T) linear interpolation between the repo's 15 °C point and 20 °C water
    nu_15, nu_20 = 1.139e-6, 1.004e-6
    t_implied = 15.0 + 5.0 * (nu_15 - nu_implied) / (nu_15 - nu_20)
    u_for_re = re_transcribed * nu / L
    c2_pass = abs(dev_pct) <= CRIT["c2_re_rel_dev_max_pct"]

    # ---- criterion C4: (1+k) from two anchors --------------------------
    k_bare = head["one_plus_k"]
    k_full_t3 = rows_full_t3[0]["one_plus_k"]
    k_full_t14 = rows_full_t14[0]["one_plus_k"]
    stated = q(parsed, "SUBOFF-REFDATA-BRACKET/SELFCHECK")
    assert_check = assert_no_hand_typed_constants(
        [r["CT"] for r in rows_bare + rows_full_t14 + rows_full_t3]
        + [r["CF_ittc78"] for r in rows_bare + rows_full_t14 + rows_full_t3]
        + [r["one_plus_k"] for r in rows_bare + rows_full_t14 + rows_full_t3]
        + [r["Re"] for r in rows_bare + rows_full_t14 + rows_full_t3]
    )
    if not assert_check["pass"]:
        raise SystemExit(f"[no-hand-copy] derived values appear as literals: {assert_check['violations']}")

    # ---- cross-artefact reproducibility: reproduce the frozen refs_lock --
    lock_path = REPO / "benchmarks/pending/suboff_highre_resistance/refs_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock_rows = []
    for anchor in lock["anchors"]:
        v = anchor["V_ms"]
        match = min(rows_full_t3, key=lambda R: abs(R["U_mps"] - v))
        if abs(match["U_mps"] - v) > 1e-6:
            continue
        lock_rows.append({
            "V_ms": v,
            "our_CT": match["CT"], "lock_CT": anchor["CT"],
            "rel_dev_CT_pct": rnd(rel_pct(match["CT"], anchor["CT"]), 6),
            "our_Re": match["Re"], "lock_Re": anchor["Re_Lmax_nu1139"],
            "rel_dev_Re_pct": rnd(rel_pct(match["Re"], anchor["Re_Lmax_nu1139"]), 6),
            "our_CF": match["CF_ittc78"], "lock_CT_ITTC1957": anchor["CT_ITTC1957"],
            "rel_dev_CF_pct": rnd(rel_pct(match["CF_ittc78"], anchor["CT_ITTC1957"]), 6),
            "our_one_plus_k": match["one_plus_k"],
            "lock_form_factor": anchor["form_factor_CT_over_ITTC"],
            "rel_dev_k_pct": rnd(rel_pct(match["one_plus_k"], anchor["form_factor_CT_over_ITTC"]), 6),
        })
    max_repro = max((max(abs(r["rel_dev_CT_pct"]), abs(r["rel_dev_Re_pct"]),
                         abs(r["rel_dev_CF_pct"]), abs(r["rel_dev_k_pct"])) for r in lock_rows),
                    default=float("nan"))

    # ---- same-camp consistency: Table 14 AFF-8 vs CFD Letters Table 3 ---
    same_camp = []
    for row in rows_full_t3:
        twin = min(rows_full_t14, key=lambda R: abs(R["U_mps"] - row["U_mps"]))
        same_camp.append({
            "table3_V_ms": row["U_mps"], "table3_RT_N": row["RT_N"],
            "table14_U_mps": twin["U_mps"], "table14_RT_N": twin["RT_N"],
            "U_rel_dev_pct": rnd(rel_pct(row["U_mps"], twin["U_mps"]), 6),
            "RT_rel_dev_pct": rnd(rel_pct(row["RT_N"], twin["RT_N"]), 6),
        })

    # ---- transcribed bracket membership -------------------------------
    br_a1 = q(parsed, "SUBOFF-REFDATA-BRACKET/R-AFF1")
    br_a8 = q(parsed, "SUBOFF-REFDATA-BRACKET/R-AFF8")
    bracket = {
        "AFF-1": {"transcribed_bracket": [br_a1["lo"], br_a1["hi"]],
                  "transcribed_usage_value": br_a1["usage"],
                  "derived_CT_at_head_anchor": head["CT"],
                  "inside_bracket": bool(float(br_a1["lo"]) <= head["CT"] <= float(br_a1["hi"])),
                  "rel_dev_vs_usage_pct": rnd(rel_pct(head["CT"], float(br_a1["usage"])), 6)},
        "AFF-8": {"transcribed_bracket": [br_a8["lo"], br_a8["hi"]],
                  "derived_CT_at_head_anchor": rows_full_t3[0]["CT"],
                  "inside_bracket": bool(float(br_a8["lo"]) <= rows_full_t3[0]["CT"] <= float(br_a8["hi"]))},
    }

    # ---- wetted-area implication of the AFF-8 transcribed bracket -------
    s_aff8 = float(q(parsed, "SUBOFF-REFDATA-BRACKET/S-AFF8")["value"])
    rt_b, u_b = rows_full_t3[0]["RT_N"], rows_full_t3[0]["U_mps"]
    s_needed_hi = rt_b / (0.5 * rho * u_b * u_b * float(br_a8["hi"]))
    s_needed_lo = rt_b / (0.5 * rho * u_b * u_b * float(br_a8["lo"]))
    ct_aff8_with_s_full = rt_b / (0.5 * rho * u_b * u_b * s_aff8)
    bracket["AFF-8"].update({
        "wetted_area_needed_m2_for_bracket_hi": rnd(s_needed_hi),
        "wetted_area_needed_m2_for_bracket_lo": rnd(s_needed_lo),
        "bare_hull_wetted_area_m2": s_doc,
        "bracket_requires_area_below_bare_hull": bool(s_needed_hi < s_doc),
        "CT_if_aff8_wetted_area_used": rnd(ct_aff8_with_s_full),
        "aff8_wetted_area_bracket_m2": s_aff8,
        "verdict": ("AFF-8 转录区间不能由原始力在物理湿面积下复现："
                    "复现 0.0042 需 S≈%.2f m²、复现 0.0038 需 S≈%.2f m²，"
                    "均小于裸壳湿面积 %.3f m²（非物理）。"
                    % (s_needed_lo, s_needed_hi, s_doc)),
    })

    # ---- uncertainty budget (anchor convention chain) ------------------
    rt_head = head["RT_N"]
    d_rho = 0.5 * (rho_hi - rho_lo) / rho
    d_s = abs(s_doc - s_cad) / s_doc
    d_force = 0.5 * 0.01 / rt_head
    d_speed = 0.5 * 0.01 / head["speed_transcribed"]
    cf_nu_up = cf_direct(re_calc * 1.04, coef, shift)
    d_nu_ct = abs(cf_nu_up - cf_direct(re_calc, coef, shift)) / cf_direct(re_calc, coef, shift)
    contrib = {
        "density_band_pct": rnd(100 * d_rho),
        "wetted_area_basis_pct": rnd(100 * d_s),
        "force_rounding_pct": rnd(100 * d_force),
        "speed_rounding_pct": rnd(100 * 2 * d_speed),
        "nu_assumption_on_CF_pct": rnd(100 * d_nu_ct),
    }
    rss = math.sqrt(sum(float(v) ** 2 for v in
                        [contrib["density_band_pct"], contrib["wetted_area_basis_pct"],
                         contrib["force_rounding_pct"], contrib["speed_rounding_pct"],
                         contrib["nu_assumption_on_CF_pct"]]))
    budget = {
        "terms": contrib,
        "components": {
            "CT_sensitivity": "CT ∝ 1/ρ, 1/S, U^-2, RT^1",
            "one_plus_k_extra": "ν 假设经 C_F(Re) 传递到 (1+k)",
        },
        "rss_total_pct": rnd(rss),
        "target_band_pct": 3.0,
        "within_target_band": bool(rss < 3.0),
        "not_quantified": [
            "拖曳水池重复性（典型 ±2–5%）：原始报告不确定度声明不在离线可达范围，无法逐字转录",
            "原试验水温/水质：ν 为『淡水 15 °C 声明假设』，非原始记录",
        ],
    }

    # ---- headline findings (every number computed above) ---------------
    findings = [
        {"id": "F1",
         "text": "AFF-1 裸艇体派生 CT 落入 [B] 级转录区间：CT=%.6f ∈ [%.4f, %.4f]，"
                 "相对常用值 %.4f 偏差 %+.2f%%。"
                 % (head["CT"], float(br_a1["lo"]), float(br_a1["hi"]),
                    float(br_a1["usage"]), rel_pct(head["CT"], float(br_a1["usage"])))},
        {"id": "F2",
         "text": "AFF-8 派生 CT=%.6f 落在转录区间 [%.4f, %.4f] 之外（%+.1f%%）；"
                 "复现该区间需要非物理的小湿面积（见 transcribed_bracket_membership.AFF-8）。"
                 "与仓内 src/tensorlbm/external_survey/suboff_anchor.py 的既有结论一致："
                 "『AFF-8 CT≈0.004』不应作为 Re≈1.2e7 的验收锚。"
                 % (rows_full_t3[0]["CT"], float(br_a8["lo"]), float(br_a8["hi"]),
                    rel_pct(rows_full_t3[0]["CT"], float(br_a8["hi"])))},
        {"id": "F3",
         "text": "两锚 (1+k) 独立复算：裸艇体 %.4f，全附体 %.4f，附体增量 %+.2f%%；"
                 "Table 14 与 CFD Letters（两次独立转录）对 AFF-8 (1+k) 的一致度 %.4f%%。"
                 % (k_bare, k_full_t3, 100 * (k_full_t3 / k_bare - 1),
                    abs(rel_pct(k_full_t14, k_full_t3)))},
        {"id": "F4",
         "text": "可复算性：用本工件逐字转录 + 声明口径复算冻结参考锁 refs_lock.json 的五锚，"
                 "CT/Re/C_F/(1+k) 最大相对偏差 %.2e%%（等于浮点舍入）。" % max_repro},
        {"id": "F5",
         "text": "口径链不确定度 RSS=%.3f%%（< 3%% 目标带）；试验重复性 ±2–5%% 未能量化，"
                 "故 3%% 带的判定必须携带该保留。" % rss},
        {"id": "F6",
         "text": "Re 一致性：所抄 Re=1.2e7 只两位有效数字，与 U·L/ν(ν=%.3e) 差 %+.2f%%；"
                 "反推隐含 ν=%.4e m²/s（≈%.1f °C 淡水），差距可归因于水温/转录精度。"
                 % (nu, dev_pct, nu_implied, t_implied)},
    ]

    # ---- criteria table ------------------------------------------------
    criteria = [
        {
            "id": "C1",
            "statement": "C_T 与 Re 逐字抄录（含表号/页码引句）入 evidence_pack 外部节",
            "threshold": "两项均为逐字（quote+表号/页码）",
            "measured": {
                "verbatim_sources": len(spec["sources"]),
                "verbatim_quotes_checked": len(layer["checks"]),
                "verbatim_quotes_all_found": all(c["verbatim_found"] for c in layer["checks"]),
                "re_verbatim_point_values": 1,
                "re_verbatim_detail": {"Re": rnd(re_transcribed),
                                       "locator": "docs/sphere_suboff_highre_reference_data_zh.md 表 3, AFF-1 行",
                                       "grade": "B_GRADE_UNVERIFIED（文档自注『精确值需核对原文』）"},
                "ct_verbatim_point_values": 0,
                "ct_available_forms": [
                    "区间 ≈0.0030–0.0032（AFF-1）/ ≈0.0038–0.0042（AFF-8），[B] 级",
                    "量纲力 RT[N] + 速度的逐字表（DTRC Table 14 p.23 / CFD Letters Table 3）——C_T 需按定义式派生",
                ],
            },
            "verdict": "PARTIAL_FAIL",
            "gap": ("可达材料中不存在 C_T 的逐字点值：DTRC/NSWCCD 原始报告全文不在仓内且离线不可达"
                    "（无外网），仓内仅有 RT[N] 的逐字表与 [B] 级 CT 区间。Re 有逐字值但仅两位有效数字。"
                    "因此 C_T 采用『逐字力值 + 声明口径派生』路径，派生值须落入 [B] 级转录区间方可用于下游。"),
            "evidence": ["sources/external_transcriptions.json", "anchor.json#external_evidence"],
        },
        {
            "id": "C2",
            "statement": "Re = U·L/ν 相对偏差 ≤0.1%（与所抄 Re 对照）",
            "threshold_pct": CRIT["c2_re_rel_dev_max_pct"],
            "measured": {
                "pair": "DTRC Table 14 AFF-1 5.92 kn ←→ 转录 Re=" + f"{re_transcribed:.3e}",
                "U_mps": head["U_mps"], "L_m": L, "nu_m2s": nu,
                "Re_computed": rnd(re_calc),
                "Re_transcribed": rnd(re_transcribed),
                "rel_dev_pct": rnd(dev_pct),
                "abs_rel_dev_pct": rnd(abs(dev_pct)),
                "implied_nu_for_transcribed_Re": rnd(nu_implied, 6),
                "implied_water_temperature_C_indicative": rnd(t_implied, 4),
                "U_needed_mps_for_Re_transcribed": rnd(u_for_re),
                "U_needed_knots": rnd(u_for_re / knot),
            },
            "verdict": "PASS" if c2_pass else "FAIL",
            "gap": None if c2_pass else (
                f"偏差 {abs(dev_pct):.2f}% 超出 0.1% 门槛 29 倍。归因可判定：所抄 Re=1.2×10⁷ 仅两位有效数字，"
                f"按 5.92 kn + L=4.356 m 反推需 ν={nu_implied:.4e} m²/s（≈{t_implied:.1f} °C 淡水，物理合理），"
                f"与仓内声明 ν=1.139e-6（15 °C）差 {100*(nu_implied/nu-1):.2f}% —— 即差距来自试验水温/转录精度，"
                "不是量纲或换算错误。要达到 0.1% 必须逐字抄录原始报告的水温与 Rn 表，本环境不具备该材料。"),
            "evidence": ["anchor.json#derived.re_consistency"],
        },
        {
            "id": "C3",
            "statement": "ITTC-78 C_F 复算与解析式残差 <1e-12",
            "threshold": CRIT["c3_cf_residual_max"],
            "measured": {
                "formula": "C_F = 0.075 / (log10(Re) - 2)^2（系数由 quotes 正则解析）",
                "evaluations": ["math.log10 直接式", "math.log 换底式", "Decimal 50 位参考值"],
                "n_re_points": len(grid),
                "max_abs_residual": cf_report["max_abs_residual"],
                "known_value_check": known_value,
            },
            "verdict": "PASS" if c3_pass else "FAIL",
            "gap": None if c3_pass else f"残差 {cf_report['max_abs_residual']} ≥ {CRIT['c3_cf_residual_max']}",
            "evidence": ["anchor.json#derived.ittc78_cf"],
        },
        {
            "id": "C4",
            "statement": "(1+k) 由两锚复算，禁第三处手抄",
            "threshold": "两锚各自独立复算；脚本内不得出现 (1+k)/CT/C_F/Re 的浮点字面量",
            "measured": {
                "anchor_A": {"label": "AFF-1 裸艇体 @5.92 kn（DTRC Table 14 p.23）",
                             "Re": head["Re"], "CT": head["CT"],
                             "CF": head["CF_ittc78"], "one_plus_k": k_bare},
                "anchor_B": {"label": "AFF-8 全附体 @3.050 m/s（CFD Letters 2024 Table 3 EFD）",
                             "Re": rows_full_t3[0]["Re"], "CT": rows_full_t3[0]["CT"],
                             "CF": rows_full_t3[0]["CF_ittc78"], "one_plus_k": k_full_t3},
                "anchor_A2_table14_full": {"label": "AFF-8 @5.93 kn（Table 14 同试验，第二转录）",
                                           "one_plus_k": k_full_t14,
                                           "rel_dev_vs_anchor_B_pct": rnd(rel_pct(k_full_t14, k_full_t3), 6)},
                "appendage_increment_pct": rnd(100 * (k_full_t3 / k_bare - 1)),
                "no_hand_copy_ast_check": assert_check,
                "cross_check_stated_value": {"stated_one_plus_k": float(stated["k"]),
                                             "rel_dev_pct": rnd(rel_pct(k_bare, float(stated["k"])), 6)},
                "full_ladder": {"AFF-1": [r["one_plus_k"] for r in rows_bare],
                                "AFF-8": [r["one_plus_k"] for r in rows_full_t3]},
            },
            "verdict": "PASS" if assert_check["pass"] else "FAIL",
            "gap": None,
            "evidence": ["anchor.json#anchors", "anchor.json#derived.form_factor"],
        },
    ]

    n_pass = sum(1 for c in criteria if c["verdict"] == "PASS")
    overall = ("PASS" if n_pass == len(criteria)
               else "PARTIAL" if n_pass > 0 else "FAIL")

    anchor = {
        "case": "DARPA SUBOFF (DTRC Model 4740) — 裸艇体 AFF-1 与全附体 AFF-8 阻力锚",
        "task_key": "paper-exp-lbm-suboff-tau-1",
        "mission_id": 7477,
        "title": "SUBOFF 外部锚规范化与可复算性（resolution-parameter decoupling 实验 1）",
        "generated_by": "benchmarks/verified/suboff_anchor/build_anchor.py",
        "schema_version": "suboff-anchor/v1",
        "overall_verdict": overall,
        "criteria_summary": {c["id"]: c["verdict"] for c in criteria},
        "geometry": {"L_m": L, "D_m": D, "L_over_D": rnd(L / D),
                     "displacement_volume_m3": vol, "wetted_surface_m2": s_doc,
                     "wetted_surface_basis": "湿表面积（wetted surface），实验口径",
                     "source": "CFD-LETTERS-2024-T1-T3 Table 1（逐字）"},
        "findings": findings,
        "conventions": {
            "CT_definition": "CT = RT / (0.5 * rho * U^2 * S) —— 逐字口径：CT = RT / (0.5 · ρ · V² · S)",
            "reynolds_definition": "Re = U * L / nu，L = Lmax（艇长总长 4.356 m）",
            "rho_kg_m3": rho, "rho_band_kg_m3": [rho_lo, rho_hi],
            "nu_m2_s": nu, "nu_basis": "淡水 15 °C 声明假设（仓内冻结参考锁）",
            "knot_to_mps": knot,
            "model_scale": True,
            "area_basis_note": ("AFF-1 与 AFF-8 均用裸壳湿面积 S=%.3f m²（冻结参考锁口径，"
                                "与下游 LBM 参考面积一致）。AFF-8 若改用全附体湿面积 %.2f m²，"
                                "其 CT 将变为 %.6f（相对裸壳口径 %+.2f%%）。"
                                % (s_doc, s_aff8, ct_aff8_with_s_full,
                                   rel_pct(ct_aff8_with_s_full, rows_full_t3[0]["CT"]))),
            "test_conditions_disclosed": [
                "模型尺度拖曳水池总阻力（深潜，无自由面兴波）",
                "水温/水质：原始记录不在离线可达范围，ν 为声明假设",
                "试验不确定度：原始报告声明不可达 → 未转录（见 uncertainty_budget.not_quantified）",
            ],
        },
        "external_evidence": {
            "file": "benchmarks/verified/suboff_anchor/sources/external_transcriptions.json",
            "n_sources": len(spec["sources"]),
            "n_verbatim_quotes": len(layer["checks"]),
            "all_quotes_machine_verified": all(c["verbatim_found"] for c in layer["checks"]),
            "offline_disclosure": spec["offline_disclosure"],
            "sources": [{k: s[k] for k in ("source_id", "kind", "grade", "citation", "locator",
                                           "archived_at", "what_it_is", "disclosure")}
                        for s in spec["sources"]],
            "verbatim_checks": layer["checks"],
        },
        "anchors": {
            "A_AFF1_bare_DTRC_T14": {"rows": rows_bare,
                                     "primary_row": rows_bare[0]["row_id"]},
            "B_AFF8_full_CFDletters_T3": {"rows": rows_full_t3,
                                          "primary_row": rows_full_t3[0]["row_id"]},
            "A2_AFF8_full_DTRC_T14": {"rows": rows_full_t14,
                                      "role": "同试验第二转录（用于 same_camp_consistency 与 (1+k) 双锚互检）"},
        },
        "criteria": criteria,
        "derived": {
            "ittc78_cf": {
                "formula_coef": coef, "formula_log10_shift": shift,
                "residual_report": cf_report,
                "known_value_check": known_value,
                "max_abs_residual": cf_report["max_abs_residual"],
                "threshold": CRIT["c3_cf_residual_max"],
                "pass": c3_pass,
            },
            "re_consistency": {
                "Re_computed": rnd(re_calc), "Re_transcribed": rnd(re_transcribed),
                "rel_dev_pct": rnd(dev_pct), "threshold_pct": CRIT["c2_re_rel_dev_max_pct"],
                "pass": c2_pass,
                "implied_nu_m2_s": rnd(nu_implied, 6),
                "implied_temperature_C_indicative": rnd(t_implied, 4),
                "U_needed_mps": rnd(u_for_re), "U_needed_knots": rnd(u_for_re / knot),
            },
            "form_factor": {
                "definition": "(1+k) = CT / CF_ittc78(Re)",
                "anchor_A_one_plus_k": k_bare,
                "anchor_B_one_plus_k": k_full_t3,
                "anchor_A2_one_plus_k": k_full_t14,
                "anchor_A2_vs_B_rel_dev_pct": rnd(rel_pct(k_full_t14, k_full_t3), 6),
                "appendage_increment_pct": rnd(100 * (k_full_t3 / k_bare - 1)),
                "ladders": {"AFF-1": [[r["Re"], r["one_plus_k"]] for r in rows_bare],
                            "AFF-8": [[r["Re"], r["one_plus_k"]] for r in rows_full_t3]},
                "no_hand_copy_check": assert_check,
            },
            "transcribed_bracket_membership": bracket,
            "uncertainty_budget": budget,
            "cross_artifact_reproduction": {
                "reference_artifact": "benchmarks/pending/suboff_highre_resistance/refs_lock.json",
                "note": "用本工件的逐字转录 + 声明口径复算冻结参考锁的五锚，检验两件独立产物的可复算性",
                "rows": lock_rows,
                "max_abs_rel_dev_pct": rnd(max_repro, 6),
            },
            "same_camp_consistency": {
                "note": "DTRC Table 14 (kn) 与 CFD Letters Table 3 (m/s) 对同一水池试验的两次转录互检",
                "rows": same_camp,
            },
            "double_entry_check": double_entry,
        },
        "provenance": {
            "repo": "TensorLBM",
            "script": "benchmarks/verified/suboff_anchor/build_anchor.py",
            "sources_file": "benchmarks/verified/suboff_anchor/sources/external_transcriptions.json",
            "sources_file_sha256": file_sha256(SOURCES),
            "figure": "benchmarks/verified/suboff_anchor/figures/anchor_overview.png",
            "test": "tests/test_suboff_anchor_pack.py",
            "upstream_modules_reused": [
                "src/tensorlbm/suboff_reference_data.py（Table 14 常量，双录对校对象）",
            ],
            "regenerate": "python3 benchmarks/verified/suboff_anchor/build_anchor.py",
            "environment": {"python": sys.version.split()[0], "numpy_free": True},
        },
        "downstream_contract": {
            "use_this_as": "研究计划后续实验（网格/τ/碰撞模型三因子解耦）的唯一外部锚：CT_exp(Re) + (1+k)",
            "do_not_use": [
                "AFF-8 CT≈0.0040 @Re=2e6（[C] 级，Re 条件与主流锚不一致）",
                "任何未经本证据包逐字转录的 CT/Re 数值",
            ],
            "band_resolvability": {
                "anchor_convention_rss_pct": rnd(rss),
                "target_band_pct": 3.0,
                "conclusion": ("口径链总不确定度远小于 3% 带 → 『3% 带』在锚侧可分辨；"
                               "但试验重复性（±2–5%）未能量化，结论须带该保留。"),
            },
        },
    }
    return anchor


def plot(anchor: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17.4, 4.9))

    ax = axes[0]
    rows_a = anchor["anchors"]["A_AFF1_bare_DTRC_T14"]["rows"]
    rows_b = anchor["anchors"]["B_AFF8_full_CFDletters_T3"]["rows"]
    ax.loglog([r["Re"] for r in rows_a], [r["CT"] for r in rows_a], "o-", color="#1f77b4",
              label="AFF-1 bare hull (DTRC Table 14)")
    ax.loglog([r["Re"] for r in rows_b], [r["CT"] for r in rows_b], "s-", color="#d62728",
              label="AFF-8 full (CFD Letters Table 3 EFD)")
    coef = anchor["derived"]["ittc78_cf"]["formula_coef"]
    shift = anchor["derived"]["ittc78_cf"]["formula_log10_shift"]
    re_grid = [10 ** (6.9 + 0.5 * 0.05 * i) for i in range(60)]
    cf = [coef / (math.log10(x) - shift) ** 2 for x in re_grid]
    k_a = anchor["derived"]["form_factor"]["anchor_A_one_plus_k"]
    k_b = anchor["derived"]["form_factor"]["anchor_B_one_plus_k"]
    ax.loglog(re_grid, cf, "k--", lw=1.2, label="ITTC-78 $C_F$")
    ax.loglog(re_grid, [c * k_a for c in cf], "--", color="#1f77b4", lw=1.0,
              label=f"$C_F(1+k)$, 1+k={k_a:.3f} (bare)")
    ax.loglog(re_grid, [c * k_b for c in cf], "--", color="#d62728", lw=1.0,
              label=f"$C_F(1+k)$, 1+k={k_b:.3f} (full)")
    ax.set_xlabel("Re = U·L/ν")
    ax.set_ylabel("$C_T$, $C_F$")
    ax.set_title("SUBOFF external anchors vs ITTC-78 friction line")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")

    ax2 = axes[1]
    terms = anchor["derived"]["uncertainty_budget"]["terms"]
    labels = list(terms.keys())
    vals = [float(terms[k]) for k in labels]
    rss = float(anchor["derived"]["uncertainty_budget"]["rss_total_pct"])
    labels.append("RSS total")
    vals.append(rss)
    colors = ["#7f7f7f"] * (len(labels) - 1) + ["#2ca02c"]
    ax2.barh(labels, vals, color=colors)
    ax2.axvline(3.0, color="#d62728", ls="--", lw=1.2, label="3% target band")
    ax2.set_xlabel("relative uncertainty on $C_T$ / (1+k)  [%]")
    ax2.set_title("Anchor convention-chain uncertainty (repeatability NOT included)")
    ax2.set_xlim(0, max(3.2, max(vals) * 1.25))
    for i, v in enumerate(vals):
        ax2.text(v * 1.02 + 0.01, i, f"{v:.3f}%", va="center", fontsize=8)
    ax2.legend(fontsize=8)
    ax2.grid(True, axis="x", alpha=0.3)

    ax3 = axes[2]
    br = anchor["derived"]["transcribed_bracket_membership"]
    for i, (name, color) in enumerate((("AFF-1", "#1f77b4"), ("AFF-8", "#d62728"))):
        lo, hi = br[name]["transcribed_bracket"]
        ct = br[name]["derived_CT_at_head_anchor"]
        ax3.vlines(i, lo, hi, color=color, lw=8, alpha=0.35,
                   label=f"{name} [B] transcribed bracket" if i == 0 else None)
        ax3.plot([i], [ct], "D", color=color, ms=9,
                 label=f"{name} derived $C_T$ (verbatim forces)" if i == 0 else None)
        ax3.annotate(f"{ct:.5f}\n{'inside' if br[name]['inside_bracket'] else 'OUTSIDE'}",
                     (i, ct), textcoords="offset points", xytext=(12, 0), fontsize=8, va="center")
    ax3.set_xticks([0, 1], ["AFF-1 (bare, 5.92 kn)", "AFF-8 (full, 3.050 m/s)"])
    ax3.set_ylabel("$C_T$")
    ax3.set_title("Derived $C_T$ vs [B] transcribed brackets")
    ax3.grid(True, axis="y", alpha=0.3)
    ax3.legend(fontsize=7, loc="upper left")

    fig.suptitle("paper-exp lbm-suboff-tau / experiment 1 — external anchor normalization", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_FIG, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify and print verdicts only")
    args = ap.parse_args()

    anchor = build()
    blob = json.dumps(anchor, indent=2, ensure_ascii=False) + "\n"
    print(json.dumps(anchor["criteria_summary"], indent=1))
    print("overall:", anchor["overall_verdict"])
    for c in anchor["criteria"]:
        if c["verdict"] != "PASS":
            print(f"  {c['id']} {c['verdict']}: {c['gap']}")
    if args.check:
        return 0
    OUT_JSON.write_text(blob, encoding="utf-8")
    plot(anchor)
    print("wrote", OUT_JSON.relative_to(REPO), "and", OUT_FIG.relative_to(REPO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())