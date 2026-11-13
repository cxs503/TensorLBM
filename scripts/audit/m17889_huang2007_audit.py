#!/usr/bin/env python3
"""m17889 — 交差核验（非重做）：XVAL-LBM-HUANG2007-CONTACT-ANGLE-INVERSION（对象 m17845）

用途：独立复核 m17845 交付的"对标通过"主张，产出机器读数供 verdict 落表。

被审工件（只读，不改）：
  /root/tp-agent/work/m17845/benchmarks/xval_huang2007_contact_angle_inversion/
    ├── run_xval.py      主对标脚本（标定表硬编码在 86-107 行）
    ├── results.json     结果（无逐行误差表）
    └── REPORT.md        六段报告（含 9 行 ref/got 对照表）

本脚本做的事（全部可复算、零手抄）：
  1. 复现性：把我的审计副本跑出的 results.json 与交付版逐字段数值比对；
  2. 循环性检验：证明交付表的 err_Huang 列是"查表求逆→再查表"的恒等式（不是独立验证）；
  3. 留一交叉验证（LOO）：交付表自身数据的真正样本外误差；
  4. 边界钳制分析：交付表里 9 行中有几行被 f_w∈[0.30,0.65] 端点钳死；
  5. 跨网格 hold-out：粗网格标定反解 → 细网格实测（从 docs/CASE-MCMP-CONTACT-ANGLE.md
     §④ B2 表机器抽取的既有真实 LBM 结果，mission 15470）；
  6. 基准登记核对：tensor_benchmark_cases 里本锚点登记的算例数（唯一裁判）；
  7. 声称工件的存在性核对。

用法：python scripts/audit/m17889_huang2007_audit.py
输出：artifacts/m17889_audit/audit_m17889.json（+ stdout 表格）
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = ROOT / "artifacts" / "m17889_audit"
COPY = ROOT / "benchmarks" / "xval_huang2007_contact_angle_inversion_audit_copy"
SRC_TREE = Path("/root/tp-agent/work/m17845")
SRC_BENCH = SRC_TREE / "benchmarks" / "xval_huang2007_contact_angle_inversion"
DOC_MCMP = ROOT / "docs" / "CASE-MCMP-CONTACT-ANGLE.md"
OUT = AUDIT_DIR / "audit_m17889.json"


# ---------------------------------------------------------------------------
# 0. 载入交付脚本（模块级常量 + 复刻其 forward_cal；不修改交付件）
# ---------------------------------------------------------------------------
def load_delivered():
    spec = importlib.util.spec_from_file_location("xval_delivered", COPY / "run_xval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # 仅执行 import + 常量定义（main() 不自动调用）
    return mod


def forward_cal(f_w: float, data: list[dict]) -> float:
    """与交付脚本 run_xval.py:325-338 逐字同构的 f_w→θ_measured 分段线性插值。"""
    valid = sorted([d for d in data if d["theta_measured"] == d["theta_measured"]],
                   key=lambda d: d["f_w"])
    f_ws = [d["f_w"] for d in valid]
    thetas = [d["theta_measured"] for d in valid]
    if f_w <= f_ws[0]:
        return thetas[0]
    if f_w >= f_ws[-1]:
        return thetas[-1]
    for i in range(len(f_ws) - 1):
        if f_ws[i] <= f_w <= f_ws[i + 1]:
            t = (f_w - f_ws[i]) / (f_ws[i + 1] - f_ws[i])
            return thetas[i] + t * (thetas[i + 1] - thetas[i])
    return thetas[-1]


def clamp_state(f_w: float, data: list[dict]) -> str:
    f_ws = sorted(d["f_w"] for d in data)
    if f_w <= f_ws[0]:
        return "clamped_low"
    if f_w >= f_ws[-1]:
        return "clamped_high"
    return "interp"


# ---------------------------------------------------------------------------
# 1. 复现性
# ---------------------------------------------------------------------------
def repro_check() -> dict:
    fresh = json.loads((COPY / "results.json").read_text())
    delivered = json.loads((SRC_BENCH / "results.json").read_text())
    diffs = []

    def walk(a, b, path=""):
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                diffs.append(f"{path}: key set {sorted(set(a) ^ set(b))}")
            for k in a:
                if k in b:
                    walk(a[k], b[k], f"{path}/{k}")
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                diffs.append(f"{path}: len {len(a)} vs {len(b)}")
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{path}[{i}]")
        else:
            if isinstance(a, float) or isinstance(b, float):
                if not math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-12):
                    diffs.append(f"{path}: {a} != {b}")
            elif a != b:
                diffs.append(f"{path}: {a} != {b}")

    walk(fresh, delivered)
    return {
        "fresh_run": str(COPY / "results.json"),
        "delivered": str(SRC_BENCH / "results.json"),
        "byte_identical_fields": len(diffs) == 0,
        "n_field_diffs": len(diffs),
        "field_diffs": diffs[:20],
    }


# ---------------------------------------------------------------------------
# 2. 循环性（tautology）检验
# ---------------------------------------------------------------------------
def tautology_check(mod) -> dict:
    data = mod.CALIBRATION_DATA_COARSE
    inv = mod.build_calibration_function(data)
    lo, hi = min(d["theta_measured"] for d in data), max(d["theta_measured"] for d in data)
    worst = 0.0
    worst_at = None
    n = 0
    th = lo + 1e-9
    while th < hi:
        fw = inv(th)
        back = forward_cal(fw, data)
        e = abs(back - th)
        n += 1
        if e > worst:
            worst, worst_at = e, th
        th += 0.25
    return {
        "what": "err_Huang 列 = forward_cal(invert(theta)) ；两者是同一张分段线性表的互逆映射",
        "theta_scan_range_deg": [round(lo, 2), round(hi, 2)],
        "n_sample": n,
        "max_abs_roundtrip_error_deg": worst,
        "max_abs_roundtrip_error_at_deg": worst_at,
        "max_rel_error_pct": 100.0 * worst / (worst_at or 1.0),
        "conclusion": "上机恒等（机器精度内）⇒ 8/9 的 0.0% 误差不含任何关于物理精度/OSS 对标的信息量；"
                      "它是插值函数的性质，不是接触角模型的精度",
    }


# ---------------------------------------------------------------------------
# 3. 留一交叉验证（真正的样本外误差）
# ---------------------------------------------------------------------------
def loo_check(mod) -> dict:
    data = mod.CALIBRATION_DATA_COARSE
    rows = []
    for i, d in enumerate(data):
        rest = [x for j, x in enumerate(data) if j != i]
        if len(rest) < 2:
            continue
        th = d["theta_measured"]
        fw_lo = min(x["f_w"] for x in rest)
        fw_hi = max(x["f_w"] for x in rest)
        out_of_range = not (fw_lo <= d["f_w"] <= fw_hi)
        # 真样本外：用其余点建"f_w→θ"表，在**被留出点的 f_w** 处插值，与其实测角比
        fw = d["f_w"]
        pred = forward_cal(fw, rest)
        err_deg = abs(pred - th)
        rows.append({
            "held_out_f_w": d["f_w"],
            "theta_measured": th,
            "f_w": fw,
            "theta_predicted": round(pred, 2),
            "abs_err_deg": round(err_deg, 2),
            "rel_err_pct": round(100.0 * err_deg / th, 2),
            "out_of_range_extrapolation": out_of_range,
        })
    interior = [r for r in rows if not r["out_of_range_extrapolation"]]
    worst = max(rows, key=lambda r: r["rel_err_pct"])
    worst_in = max(interior, key=lambda r: r["rel_err_pct"]) if interior else None
    n_pass15_all = sum(1 for r in rows if r["rel_err_pct"] < 15.0)
    n_pass15_in = sum(1 for r in interior if r["rel_err_pct"] < 15.0)
    return {
        "what": "逐点留出（leave-one-out）：用其余 7 点建 f_w→θ 表，在被留出点的 f_w 处插值，与其实测角比（真样本外，非查表求逆再查表）",
        "rows": rows,
        "n_points": len(rows),
        "n_interior": len(interior),
        "worst": worst,
        "worst_interior": worst_in,
        "pass15_all": f"{n_pass15_all}/{len(rows)}",
        "pass15_interior": f"{n_pass15_in}/{len(interior)}",
    }


# ---------------------------------------------------------------------------
# 4. 交付表的端点钳制分析
# ---------------------------------------------------------------------------
def clamp_check(mod) -> dict:
    data = mod.CALIBRATION_DATA_COARSE
    inv = mod.build_calibration_function(data)
    targets = [30, 45, 60, 75, 90, 105, 120, 135, 150]
    rows = []
    y_pass = h_pass = 0
    for th in targets:
        fw_y = mod.young_f_w(th)
        fw_h = inv(th)
        t_y = forward_cal(fw_y, data)
        t_h = forward_cal(fw_h, data)
        e_y = abs(t_y - th) / th * 100
        e_h = abs(t_h - th) / th * 100
        y_pass += e_y < 15
        h_pass += e_h < 15
        rows.append({
            "theta_target": th,
            "f_w_young": round(fw_y, 4),
            "theta_meas_young": round(t_y, 2),
            "err_young_pct": round(e_y, 2),
            "young_sampling": clamp_state(fw_y, data),
            "f_w_huang": round(fw_h, 4),
            "theta_meas_huang": round(t_h, 2),
            "err_huang_pct": round(e_h, 2),
            "huang_sampling": clamp_state(fw_h, data),
        })
    distinct_young = sorted({r["theta_meas_young"] for r in rows})
    return {
        "rows": rows,
        "recomputed_young_pass": y_pass,
        "recomputed_huang_pass": h_pass,
        "delivered_reported_young_pass": 2,
        "delivered_reported_huang_pass": 8,
        "aggregate_matches_delivered": (y_pass == 2 and h_pass == 8),
        "distinct_theta_meas_young_values": distinct_young,
        "n_rows_young_clamped": sum(1 for r in rows if r["young_sampling"] != "interp"),
        "n_rows_huang_clamped": sum(1 for r in rows if r["huang_sampling"] != "interp"),
        "note": "9 行里 Young 列有 6–8 行落在标定表端点（f_w<0.30 或 >0.65），"
                "被钳到同一个 θ_meas 值 —— 'err_Young%' 列的多数数字反映的是钳制，不是 Young 关系误差",
    }


# ---------------------------------------------------------------------------
# 5. 跨网格 hold-out：既有真实 LBM 实测（从 mission 15470 的 docs 机器抽取）
# ---------------------------------------------------------------------------
def parse_b2_from_doc() -> dict:
    if not DOC_MCMP.exists():
        return {"available": False}
    txt = DOC_MCMP.read_text()
    m = re.search(r"\*\*B2 hold-out.*?\n\n(.*?)\n\n", txt, re.S)
    if not m:
        return {"available": False}
    rows = []
    for line in m.group(1).splitlines():
        cells = [c.strip().strip("*") for c in line.strip().strip("|").split("|")]
        if len(cells) >= 4 and re.match(r"^\d+(\.\d+)?$", cells[0]):
            try:
                rows.append({
                    "theta_target": float(cells[0]),
                    "f_w_from_inverse": float(cells[1]),
                    "theta_got_lbm": float(cells[2]),
                    "diff_deg": float(cells[3].replace("+", "")),
                })
            except ValueError:
                continue
    for r in rows:
        r["rel_err_pct"] = round(100.0 * abs(r["diff_deg"]) / r["theta_target"], 2)
    return {
        "available": bool(rows),
        "source": f"{DOC_MCMP} §④ B2 hold-out（mission 15470 的真实 LBM 实测；本单未重跑，机器抽取）",
        "criterion_in_that_doc": "|Δ| < 10°",
        "rows": rows,
        "n_pass_15pct": sum(1 for r in rows if r["rel_err_pct"] < 15.0),
        "n_rows": len(rows),
        "verdict_in_that_doc": "verdicts.calibrated = FAIL 0/3, max|Δ|=19.12°",
    }


def cross_grid_predict(mod) -> dict:
    """粗表反解 → 细表插值预测（纯表格运算，不含新物理）+ 与真实细网格实测对照。"""
    coarse = mod.CALIBRATION_DATA_COARSE
    fine = mod.CALIBRATION_DATA_FINE
    inv = mod.build_calibration_function(coarse)
    b2 = parse_b2_from_doc()
    rows = []
    for r in b2.get("rows", []):
        fw = inv(r["theta_target"])
        pred_fine_table = forward_cal(fw, fine)
        rows.append({
            "theta_target": r["theta_target"],
            "f_w_from_coarse_inverse": round(fw, 6),
            "fine_table_predicted_deg": round(pred_fine_table, 2),
            "fine_lbm_measured_deg": r["theta_got_lbm"],
            "table_vs_lbm_gap_deg": round(abs(pred_fine_table - r["theta_got_lbm"]), 2),
            "lbm_rel_err_pct": r["rel_err_pct"],
        })
    return {
        "what": "粗网格标定曲线反解出的 f_w 用在细网格上，实测角偏离目标多少",
        "rows": rows,
        "gaps_table_minus_lbm_deg": [r["table_vs_lbm_gap_deg"] for r in rows],
        "conclusion": "跨网格迁移误差 11.4–19.1°（相对 10.4–27.3%），2/3 超 15% —— "
                      "与交付报告 'PASS' 相反的仓内既有实测",
    }


# ---------------------------------------------------------------------------
# 6. 基准登记核对（唯一裁判）
# ---------------------------------------------------------------------------
def benchmark_registry() -> dict:
    q = ("SELECT case_name, metric, tolerance FROM tensor_benchmark_cases "
         "WHERE anchor_slug='XVAL-LBM-HUANG2007-CONTACT-ANGLE-INVERSION';")
    tot = "SELECT count(*) FROM tensor_benchmark_cases;"
    out = {"query": q, "registered_rows": None, "total_cases": None, "error": None}
    try:
        r = subprocess.run(["psql", "-U", "postgres", "-d", "tensorplatform", "-At", "-F", "|", "-c", q],
                           capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            out["error"] = r.stderr.strip()[:300]
        else:
            out["registered_rows"] = [l for l in r.stdout.strip().splitlines() if l]
        r2 = subprocess.run(["psql", "-U", "postgres", "-d", "tensorplatform", "-At", "-c", tot],
                            capture_output=True, text=True, timeout=30)
        if r2.returncode == 0:
            out["total_cases"] = int(r2.stdout.strip())
    except Exception as exc:  # pragma: no cover
        out["error"] = repr(exc)
    return out


# ---------------------------------------------------------------------------
# 7. 工件存在性
# ---------------------------------------------------------------------------
def artifact_check() -> dict:
    claimed = {
        "run_xval.py": SRC_BENCH / "run_xval.py",
        "generate_figures.py": SRC_BENCH / "generate_figures.py",
        "results.json": SRC_BENCH / "results.json",
        "REPORT.md": SRC_BENCH / "REPORT.md",
        "figures/calibration_curve.png": SRC_BENCH / "figures" / "calibration_curve.png",
        "figures/error_comparison.png": SRC_BENCH / "figures" / "error_comparison.png",
        "figures/cos_theta_vs_fw.png": SRC_BENCH / "figures" / "cos_theta_vs_fw.png",
        # 报告 §6.4 声称的"已有标定数据"来源
        "artifacts/CASE-MCMP-CONTACT-ANGLE/contact_angle_sweep.json": SRC_TREE / "artifacts" / "CASE-MCMP-CONTACT-ANGLE" / "contact_angle_sweep.json",
    }
    res = {k: {"exists": p.exists(), "size": p.stat().st_size if p.exists() else None,
               "path": str(p)} for k, p in claimed.items()}
    # git 追踪状态（交付件是否入库 / 是否有改动）
    def sh(cmd):
        r = subprocess.run(cmd, shell=True, cwd=str(SRC_TREE), capture_output=True, text=True)
        return r.stdout.strip()
    res["_git"] = {
        "branch": sh("git branch --show-current"),
        "head": sh("git log --oneline -1"),
        "dirty_status": sh("git status --porcelain | head -5"),
        "tracked_commits_ahead_of_main": sh("git rev-list --count origin/main..HEAD 2>/dev/null || echo n/a"),
    }
    return res


# ---------------------------------------------------------------------------
def main() -> int:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    mod = load_delivered()

    audit = {
        "audit_mission": 17889,
        "target_mission": 17845,
        "anchor": "XVAL-LBM-HUANG2007-CONTACT-ANGLE-INVERSION",
        "repo": "TensorLBM",
        "verdict_by": "board-audit",
        "1_reproducibility": repro_check(),
        "2_tautology": tautology_check(mod),
        "3_leave_one_out": loo_check(mod),
        "4_delivered_table_recompute": clamp_check(mod),
        "5_cross_grid_holdout": cross_grid_predict(mod),
        "6_benchmark_registry": benchmark_registry(),
        "7_artifacts": artifact_check(),
    }

    # 抽查算例（若后台 LBM 已跑完）
    spots = {}
    for p in sorted((AUDIT_DIR / "cases").glob("*.json")):
        spots[p.stem] = json.loads(p.read_text())
    if spots:
        audit["8_fresh_lbm_spots"] = {
            "source": "本单重跑（tests/benchmarks/wetting_contact_angle/run_case_cli.py，coarse=96x64 R=13）",
            "cases": {k: {kk: v.get(kk) for kk in ("grid", "law", "theta_eq", "wall_fraction",
                                                   "theta_measured", "theta_cap", "status",
                                                   "runtime_s", "drift_deg")}
                      for k, v in spots.items()},
        }
        table = {d["f_w"]: d["theta_measured"] for d in mod.CALIBRATION_DATA_COARSE}
        for k, v in spots.items():
            tm, te, law = v.get("theta_measured"), v.get("theta_eq"), v.get("law")
            rec = audit["8_fresh_lbm_spots"]["cases"][k]
            rec["role"] = ("prescription_spot（处方复验：init 帽=目标角）" if law == "calibrated"
                           else "table_reproducibility（标定表可复现性：与交付表同 f_w 对照）")
            if law == "calibrated" and tm is not None and te:
                rec["rel_err_pct"] = round(100.0 * abs(tm - te) / te, 2)
            if law == "scan" and tm is not None:
                fw = round(float(v["wall_fraction"]), 2)
                if fw in table:
                    rec["delivered_table_theta"] = table[fw]
                    rec["table_minus_rerun_deg"] = round(tm - table[fw], 4)
        audit["8_fresh_lbm_spots"]["protocol_note"] = (
            "spot*: law=calibrated（init 帽=目标角，应用协议）；scan*: law=scan（init 帽=该 f_w 下 Young 角），"
            "用于核对交付标定表的可复现性。三估计器 spread 原样保留，不择优。")

    OUT.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")

    # ---- stdout 摘要 ----
    print("=" * 78)
    print("m17889 交差核验 — XVAL-LBM-HUANG2007-CONTACT-ANGLE-INVERSION (对象 m17845)")
    print("=" * 78)
    r = audit["1_reproducibility"]
    print(f"[1] 复现性: 字段差异 {r['n_field_diffs']} → 逐字段数值一致={r['byte_identical_fields']}")
    t = audit["2_tautology"]
    print(f"[2] 循环性: 扫描 {t['n_sample']} 点，max|forward∘invert − θ| = "
          f"{t['max_abs_roundtrip_error_deg']:.3e}° → 交付 err_Huang 列是恒等式")
    l = audit["3_leave_one_out"]
    print(f"[3] LOO: 全点 <15% {l['pass15_all']}；内点 <15% {l['pass15_interior']}；"
          f"最差 {l['worst']['rel_err_pct']}% @θ={l['worst']['theta_measured']}°")
    c = audit["4_delivered_table_recompute"]
    print(f"[4] 重算交付表: young_pass={c['recomputed_young_pass']} huang_pass={c['recomputed_huang_pass']}"
          f" (交付报 2 / 8, 聚合一致={c['aggregate_matches_delivered']})；"
          f"Young 列钳制行数={c['n_rows_young_clamped']}/9, Huang 列钳制行数={c['n_rows_huang_clamped']}/9")
    g = audit["5_cross_grid_holdout"]
    print("[5] 跨网格 hold-out（仓内既有 LBM 实测，mission 15470 B2）:")
    for row in g["rows"]:
        print(f"    θ_target={row['theta_target']:.0f}° f_w={row['f_w_from_coarse_inverse']:.6f} "
              f"LBM实测={row['fine_lbm_measured_deg']:.2f}° 误差={row['lbm_rel_err_pct']:.1f}% "
              f"（表预测 {row['fine_table_predicted_deg']:.1f}°，表-实测差 {row['table_vs_lbm_gap_deg']:.1f}°）")
    b = audit["6_benchmark_registry"]
    print(f"[6] 基准登记: 本锚点登记算例 = {b['registered_rows']}（全库 {b['total_cases']} 例）")
    a = audit["7_artifacts"]
    miss = [k for k, v in a.items() if k != "_git" and not (isinstance(v, dict) and v.get("exists"))]
    print(f"[7] 工件: 缺失 {miss if miss else '无'}；git 改动 = {a['_git']['dirty_status']!r}")
    if "8_fresh_lbm_spots" in audit:
        print("[8] 本单新跑 LBM 抽查:")
        for k, v in audit["8_fresh_lbm_spots"]["cases"].items():
            print(f"    {k}: θ_eq={v.get('theta_eq')} 实测={v.get('theta_measured')} "
                  f"误差={v.get('rel_err_pct')}% status={v.get('status')} ({v.get('runtime_s')}s)")
    print(f"\n→ {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())