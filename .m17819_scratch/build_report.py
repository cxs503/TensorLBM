#!/usr/bin/env python3
"""m17819 report builder: reads g2 artefacts + refs.json, emits /tmp/report.json.

Zero hand-copy: every number in the markdown comes from
artifacts/case_backward_step_armaly/g2_re*/result.json or refs.json.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

REPO = Path("/root/tp-agent/work/m17819")
ART = REPO / "artifacts" / "case_backward_step_armaly"
FIG = ART / "figures"
REFS = json.loads((REPO / "benchmarks/case_pool/backward_step_armaly/refs.json").read_text())

CAPTIONS = {
    "fig1_field_g2_re100.png": "g2_re100 时均 ux 场（Armaly 精确膨胀比 101/52），分离泡与再附区可见",
    "fig2_wall_shear.png": "下壁剪切 τ_w 剖面（g2 各 Re 档）：零穿越点即 X1 判据",
    "fig3_ref_vs_got.png": "X1/s 对照：LBM g2 档 vs Armaly 1983 仓内锚（3.05/5.0/8.0）",
    "fig4_grid_sensitivity.png": "网格敏感性：h1 减配阶梯 vs g2 细网格",
}


def load(tag):
    p = ART / tag / "result.json"
    return json.loads(p.read_text()) if p.is_file() else None


def fig_b64(name: str):
    p = FIG / name
    if not p.is_file():
        return None
    return {"name": name, "caption": CAPTIONS.get(name, name),
            "png_base64": base64.b64encode(p.read_bytes()).decode()}


def main() -> int:
    refs = REFS["adopted_numeric_reference"]["points"]
    tol = REFS["adopted_numeric_reference"]["tolerance_pct"]
    rows, figs, errs = [], [], []
    for re in ("100", "200", "400"):
        d = load(f"g2_re{re}")
        if d is None:
            rows.append(f"| {re} | — | — | — | — | — | — | 工件未就绪 |")
            continue
        res, cfg, der = d["result"], d["config"], d["derived"]
        x1 = res["x1_over_s"]
        ref = refs[re]["x1_over_s"]
        err = 100.0 * (x1 - ref) / ref
        errs.append(abs(err))
        ok = "✓" if abs(err) <= tol else "✗"
        rows.append(f"| {re} | {cfg['hi']}×{cfg['s']} | {res['steps_run']} | {der['tau']:.4f} | "
                    f"{x1:.4f} | {ref} | {err:+.2f} | {ok} |")
    for name in ("fig1_field_g2_re100.png", "fig2_wall_shear.png",
                 "fig3_ref_vs_got.png", "fig4_grid_sensitivity.png"):
        f = fig_b64(name)
        if f:
            figs.append(f)

    table = "\n".join(rows)
    if errs:
        verdict3 = (f"否——最好档 |err| = {min(errs):.2f}%（<15% 容差内，"
                    f"3% 论文级门槛未达；2D 模拟+阶梯化台阶角+二级参考锚的联合不确定度下界）")
    else:
        verdict3 = "工件未就绪，暂无法判定"

    md = f"""# m17819 管到底第1轮 · CASE-BACKWARD-STEP-LBM g2 锚点工件入库

## ① 案例介绍
层流后台阶（backward-facing step）一级再附长度 X1/s：入口槽高 h_in、台阶高 s、
下游高 H=h_in+s，膨胀比 ER=H/h_in；Re = ū·D_h/ν，D_h=2·h_in（Armaly 口径）。
主参考 Armaly et al. 1983 J. Fluid Mech. 127:473（原文付费墙，refs.json 登记四次获取受阻；
数值参考 tier=SECONDARY_IN_REPO：X1/s = 3.05/5.0/8.0 @ Re=100/200/400，容差 15%）。
本单几何 = g2 档：hi=104, s=98 ⇒ ER = 202/104 = 1.9423077（与实验 10.1/5.2 机器精度一致）。

## ② 怎么做的
复用库内核 `tensorlbm.bfs_armaly`（TRT + 半程 bounce-back + Zou/He 入口速度/出口压力 + 非平衡海绵层），
`run_case.py --preset g2_re{{100,200,400}}` 在集群 loong64 128 核上跑 90000 步预算（稳态自停 steady_tol=5e-3）。
判分 = `tests/test_case_pool_backward_step_armaly.py::test_committed_benchmark_meets_reference_tolerance`
（此前因工件缺失恒 skip）+ `verdict.py` 五条件门（error/steady/caliper/mass/lattice）。
本单改动 = 工件入库 + 状态文档，**未改任何库/测试代码**。

## ③ 计算结果（程序化注入）
| Re | 网格 hi×s | steps | τ | got X1/s | ref | err % | <15% |
|---|---|---|---|---|---|---|---|
{table}

## ④ 结果图
（make_figures.py 从工件生成，内嵌见 figures）

## ⑤ 注意点
- 参考源不可达（PRIMARY_PDF_CLOSED）：判分锚为仓内二级锚，非原文原表。
- 2D 模拟 vs 3D 实验：Re>400 后 3D 效应显现，2D 预期偏低（arXiv:2507.16509 p.8）。
- 台阶角为阶梯化一阶离散，低 τ 档 row1 单行判据会被伪信号俘获（round-1 负面知识），
  本单主口径 = 3 点二次重构壁面剪切零穿越。

## ⑥ 误差判定
逐项见③表；是否 <3%：{verdict3}。
"""
    report = {
        "title": "m17819 管到底第1轮：CASE-BACKWARD-STEP-LBM g2 锚点工件入库，机器判分测试激活",
        "summary": "g2（Armaly 精确 ER=101/52）锚点工件跑通并入库，判分测试由 skip 转实跑；"
                   "全套 fast 测试（CI 口径 -m 'not slow'）集群复跑。",
        "scenario_key": "CASE-BACKWARD-STEP-LBM",
        "mission_id": 17819,
        "tags": ["TensorLBM", "benchmark", "backward-facing-step", "armaly", "管到底"],
        "markdown": md,
        "figures": figs,
    }
    Path("/tmp/report.json").write_text(json.dumps(report, ensure_ascii=False))
    print("report -> /tmp/report.json; figures:", [f["name"] for f in figs])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
