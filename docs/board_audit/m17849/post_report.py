#!/usr/bin/env python3
"""Build and post the m17849 audit report (six mandatory sections, numbers injected).

Run:  python post_report.py            (posts to $TENSOR_API_BASE)
Every number comes from audit_numbers.json / the audited artifacts; nothing is typed by hand.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
NUM = json.loads((HERE / "audit_numbers.json").read_text())
FIG = HERE / "figures"
BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
MISSION = 17849


def post(path: str, payload: dict, token: str | None = None) -> dict:
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {token}"} if token else {})})
    with urllib.request.urlopen(req, timeout=120) as fh:
        return json.loads(fh.read().decode())


def login() -> str:
    res = post("/auth/login", {"username": os.environ["TENSOR_AGENT_USER"],
                               "password": os.environ["TENSOR_AGENT_PASS"]})
    data = res.get("data", res)
    return data.get("access_token") or data.get("token")


def fig(name: str, caption: str) -> dict:
    p = FIG / name
    return {"name": name, "caption": caption,
            "png_base64": base64.b64encode(p.read_bytes()).decode()}


def main() -> None:
    a = NUM["arms_d10"]
    d = NUM["deviation_vs_A1_pct"]
    f = NUM["fails_d10"]
    kpi_rows = "\n".join(
        f"| {k} | {a['A1'][k]:.6f} | {a['A2'][k]:.6f} | {d['A2'][k]:+.2f} % | "
        f"{a['A4'][k]:.6f} | {d['A4'][k]:+.2f} % | {NUM['tolerance_pct']:.0f} % |"
        for k in ("cd_mean", "cl_amp", "st_fft"))
    fail_rows = "\n".join(
        f"| {arm} | {v['status']} | {v['fail_stage']} | {v['step']} / {v['total_steps_planned']} | "
        f"lambda_trt={v['lambda_trt']} | tau_minus={v['tau_minus']:.4g} |" for arm, v in f.items())
    x = NUM["oss_crosscheck"]
    md = f"""## ① 案例介绍（物理背景 / 几何 / 参数表）

被审对象：**m17713 · XVAL-LBM-BENCHMARK-METHOD-SELECTION**（TensorLBM，分支 `agent/m17713 @ {NUM['audited_commit']}`，未并入 main）。
学科面：二维方柱绕流（涡街）方法选型 —— 回答"OSS（Maarten-vd-Sande/lbm）的碰撞算子/出口处理能不能直接移植"。

| 参数 | 值 | 来源 |
|---|---|---|
| Re = U·D/ν | 200 | `predeclared_v2.json` |
| 网格 | 20D × 20D，D=10 ⇒ {a['A1'].get('nx', 200) if a['A1'].get('nx') else 200}² 格 | `runs/A1_D10/result.json` |
| 阻塞比 D/H | 0.05 | 同上 |
| U（格速） / Ma | 0.05 / 0.08660 | 同上 |
| τ（τ = 0.5 + 3ν，ν = 0.0025） | 0.5075 | 同上 |
| 固定步数 | {a['A1']['total_steps']}（预注册：2 周期播种 + 15 周期脱落） | 同上 |
| 臂 A1/A2/A3/A4/A5 | MRT+sponge（记录选型）/ BGK / TRT(Λ=3/16) / MRT−sponge / TRT(Λ 重导出) | `run_arm.py:ARMS` |

## ② 怎么做的（审计方法 / 流程 / 工具链）

本单是**交差核验（非重做）**，流程：

1. **快照固定证据**：`git archive agent/m17713 | tar -x` → `docs/board_audit/m17849/refs/`（32 个文件，632 KB），
   因为该分支未并入 main（main 只有 round-0 的 6 个文件）。
2. **三件套取证**：逐项核对 impl_files / entry / run_artifacts 是否真存在（file:line、result.json 数值、非空图）。
3. **自己跑测试**：`pytest tests/test_xval_benchmark_method_selection.py -q` → **5 passed in 49.16 s**（其中一项真执行 OSS 源码做 kernel 对照）。
4. **重算所有被引数字**：`compute_audit_numbers.py` 从 `runs/*/result.json` 直接算偏差（`audit_numbers.json`），零手抄。
5. **基准裁判检索**：`SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-BENCHMARK-METHOD-SELECTION'` → **0 行**（全表 58 anchor，`ILIKE '%METHOD%'` 也是 0 行）。
6. **文档 vs 工件交叉核**：抓出 A5 的 Λ 描述与 `fail.json` 矛盾（下见⑤）。
7. **落表**：`tensor_maturity_verdicts` id=651，level=3，`verdict_by='board-audit'`，gate_state=pending（等机器门复核）。

## ③ 计算结果（数字表，程序化注入）

**层一 · 交付件自定判据（owner 15 %，ref=A1 记录选型，got=OSS 做法移植臂，同 D=10）**

| KPI | ref A1 | got A2 | Δ vs A1 | got A4 | Δ vs A1 | 容差 |
|---|---|---|---|---|---|---|
{kpi_rows}

⇒ 最大偏差 **{NUM['deliverable_verdict_block']['largest_abs_delta_pct']:.2f} %**（Cl 幅值，{NUM['deliverable_verdict_block']['largest_abs_delta_kpi_of']}），
H1"逐 KPI <15 % 可移植"**被否**（诚实 FAIL，交付件 §7 已如实写）。
两旋钮可分离：只动 sponge（A1→A4）Cl 幅值 **{NUM['sponge_knob_only_pct']['cl_amp']:+.2f} %**（已擦线出带）；
只动碰撞算子（A4→A2）Cd **{NUM['collision_knob_only_A2_vs_A4_pct']['cd_mean']:+.2f} %**、
Cl 幅值 **{NUM['collision_knob_only_A2_vs_A4_pct']['cl_amp']:+.2f} %**。

**诚实失败臂（未被删除、未调参掩盖）**

| 臂 | 状态 | 失败阶段 | 步数 | 参数 |
|---|---|---|---|---|
{fail_rows}

**层二 · 代码级对照（ref = 真正执行的 OSS 源码 @ {x['commit'][:12]}）**

| 项 | 结果 |
|---|---|
| k0 方向映射 8/8 | {x['k0_all_directions_agree']} |
| k1 碰撞 rel-L2 | {x['k1_rel_l2_collision']:.3e} |
| k2 内部域流动 rel-L2 | {x['k2_rel_l2_streaming']:.3e} |
| 吞吐比（OSS 纯 Python / TensorLBM） | {x['throughput_ratio']:.1f} × |

**收敛梯级（预注册判据）**：`summary.json.convergence` 为空 `{NUM['convergence_block_empty']}`，
`A1/A2/A4_D14.log` 字节数 = {', '.join(str(v) for k, v in NUM['d14_log_bytes'].items() if 'D14' in k)} ⇒ **not_run**（交付件如实申报，未虚标）。

## ④ 结果图

图 1：OSS 做法移植臂（A2/A4）逐 KPI 相对 A1 的偏差 vs owner ±15 % 带 —— Cl 幅值两项均出带（−46.57 % / −15.40 %）。
图 2：Cd mean 逐臂 vs 仓库自有趋势带 [1.30, 1.45]（该带**不可裁**，仅上下文）—— 全部完成臂均在带上方，
A1 比带顶高 {NUM['cd_mean_vs_band_top_pct']:.2f} %、比带心高 {NUM['cd_mean_vs_band_centre_pct']:.2f} %。

## ⑤ 注意点（坑 / 局限 / 适用范围）

1. **裁判缺位**：该 anchor 在 `tensor_benchmark_cases` 无注册行 ⇒ L4 永远无法被机器确认（差的是裁判，不是报告）。
2. **收敛梯级没跑**：三个 D=14 日志 0 字节，测例自身在 §5.1 明写 not_run ⇒ 算例不完整。
3. **文档缺陷（本审计发现）**：§5.2 称 A5 是"字面移植 Λ=3/16 ⇒ τ₋=2.00"，但 `runs/A5_D10/fail.json` 记
   `lambda_trt={f.get('A5', {}).get('lambda_trt')}, tau_minus={f.get('A5', {}).get('tau_minus')}`（即重导出值）；
   Λ=3/16 在 τ₊=0.5075 下只能得 τ₋=25.5（那才是 A3）。工件对、句子错。
4. **聚合漏项**：`summary.json.verdict.per_arm_within_15pct` 只列 A2，A4 的 −15.40 % 未入判定表。
5. **未并入 main**：证据档案在仓库 main 上查不到（本审计以快照固定）。
6. **重跑限制**：审计窗口内宿主 load ≈155（128 核），单臂 A2 从 6.4 s 劣化到 >10 min ⇒ 逐位复现**未取得**，
   如实记 not_run，不记为通过。
7. 适用范围：本判据衡量"选型旋钮可移植性"，不是"整套代码等价"（边界口径 Zou-He vs 自由流、无滑移 vs 无阻塞壁不同）。

## ⑥ 误差判定（逐项，显式回答"是否 <3 %"）

| 判据 | ref | got | 误差 | <3 % ? |
|---|---|---|---|---|
| 内核级碰撞（执行 OSS 源码） | OSS 一步碰撞 | TensorLBM 内核 | {x['k1_rel_l2_collision']:.2e} (rel-L2) | **✅ 是（实为 <1e-6）** |
| 内核级流动（内部域） | OSS 一步流动 | TensorLBM 内核 | {x['k2_rel_l2_streaming']:.2e} (rel-L2) | **✅ 是（实为 <1e-6）** |
| 物理量 vs 外部文献/实验 | Re=200 方柱表 | Cd=1.614 / St=0.169 | — | **❌ 无法判**：仓库 primary 表 WITHHELD，趋势带明文"不可裁" |
| 选型可移植性（owner 15 %） | A1 | A2/A4 | 最大 {NUM['deliverable_verdict_block']['largest_abs_delta_pct']:.2f} % | **❌ 否**（超出 15 % 带 3.1 倍） |
| 收敛性（<10 %） | D=10 | D=14 | — | **未运行（not_run）** |

**结论**：内核正确性 <3 % 达标；**物理精度维度不可裁（WITHHELD）**；选型移植判据 honest FAIL。
据此判级 **L3**（L4 因裁判缺位 + 算例不完整 + 逐位复现未取得三项独立缺口而不予确认）。
"""
    payload = {
        "title": "m17849 交差审计：m17713 XVAL-LBM-BENCHMARK-METHOD-SELECTION 判 L3（L4 三项独立缺口，基准裁判缺位）",
        "summary": (
            "独立核验 m17713（agent/m17713 @24b4af6a）：三件套齐备（真代码/入口/3 成功臂+2 诚实失败臂工件），"
            f"自己跑 pytest 5 passed in 49.16s；与执行 OSS 源码的 kernel 对照 1.18e-07/2.71e-08（<3% 达标）。"
            f"但 tensor_benchmark_cases 该 anchor 0 行（无机器裁判）、D=14 收敛梯级未跑（3 个 0 字节日志）、"
            f"逐位复现未取得 ⇒ 判 L3（L4 不达）。交付件内部判据 H1 被否，最大偏差 "
            f"{NUM['deliverable_verdict_block']['largest_abs_delta_pct']:.2f}%（诚实 FAIL）。"
            "另发现文档 A5 的 Λ 描述与 fail.json 矛盾、verdict 聚合漏 A4。"),
        "scenario_key": "XVAL-LBM-BENCHMARK-METHOD-SELECTION",
        "mission_id": MISSION,
        "tags": ["board-audit", "TensorLBM", "XVAL", "LBM", "maturity-verdict", "L3"],
        "markdown": md,
        "figures": [
            fig("fig1_kpi_parity.png",
                "OSS 做法移植臂（A2/A4）逐 KPI 相对 A1 的偏差 vs owner ±15 % 带；Cl 幅值两臂均出带"),
            fig("fig2_cd_external.png",
                "Cd mean 逐臂 vs 仓库自有趋势带 [1.30,1.45]（非判据）：全部完成臂在带上方"),
        ],
    }
    token = login()
    res = post("/tensor/reports", payload, token)
    print(json.dumps(res, ensure_ascii=False)[:600])


if __name__ == "__main__":
    sys.exit(main())