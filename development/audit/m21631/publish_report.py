#!/usr/bin/env python3
"""Publish the m21631 board-audit report to the platform (/tensor/reports).

Figures are inlined as base64 from audit/m21631/figs/ so the report is self-contained.
Run:  python3 audit/m21631/publish_report.py
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"
API = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")

MARKDOWN = """\
# 交差核验 m21548 · CASE-BACKSTEP-FVM-RE100（TensorLBM）→ 功能实质 L1，裁定返修

## ① 案例
后台阶层流分离—再附着，观测量为主再附着长度 `X1/s`（台阶高归一）。锚点 `CASE-BACKSTEP-FVM-RE100`
（feature_ledger：domain=TensorLBM，status=**open**）。几何/参数：生产档 `h_in=52, s=49`
⇒ ER=101/52=1.9423077（精确等于 Armaly 10.1/5.2），`Re=ū·D_h/ν=100`，`D_h=2h_in`，入口充分发展抛物线 `ū=0.1`，
half-way bounce-back + TRT(Λ=3/16)，出口 Zou/He + 海绵层。参考：Armaly 1983 实验 **3.05**、Erturk 2008 二维 **2.878**，
两门独立各 10 %。被核验单 m21548（succeeded，`fail_stage=budget_exhausted+forced_finish`）。

## ② 怎么做的
交差核验、非重做。三路取证：(a) 磁盘 worktree `/srv/tensor/agent/work/m21548` 的 `artifacts/`；(b) DB 四表
（`tensor_benchmark_cases` / `tensor_dev_missions` / `feature_ledger` / `tensor_reports`）；(c) git 分支归属与 trunk 差异。
基准裁判取唯一口径：`SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='CASE-BACKSTEP-FVM-RE100'`。
另在审计机跑两条**短链**（h1_re100 6000 步、g1_re100 smoke 3000 步）验证代码链能否 输入→计算→输出，
并用 `/srv/tensor/agent/maturity_gate.py::judge()` 预览机器裁定。工具链：psql + git + pytest + matplotlib。

## ③ 计算结果（程序化注入）
| 检查 | 结果 |
|---|---|
| `tensor_benchmark_cases` 该 anchor | **0 行**（全表 139 行）⇒ 无容差裁判、无 got |
| `artifacts/case_backward_step_armaly/` | **空目录（0 项）** ⇒ run_artifacts 缺 |
| 交付测试复跑 | **1 failed, 10 passed, 1 skipped in 0.31 s** |
| 失败断言 | `test_each_condition_fails_independently:162 assert 'caliper' in []` |
| git `--contains 8c1a35b3e` | 仅 `agent/m21548`（未并 main） |
| trunk `verdict.py` | `:46 TOLERANCE_PCT = 15.0`（本单声称修掉的缺陷仍在主线） |
| trunk `ladder.py` / 本单测试 | **不存在** |
| `tensor_reports mission_id=21548` | 0 行（报告从未发布） |
| 审计复现 h1_re100 6000 步 | `X1/s=2.9054`，`steady_at_step=null`，`max_speed=10.586`，质量通量误差 12.2 % |
| 审计复现 g1_re100 smoke 3000 步 | `X1/s=2.6499`，`steady_at_step=null`，`max_speed` 峰值 3.678 |
| gate 预览 | `declared 1 → gate_level 1`；`declared 2 → gate_level 1（缺 entry/run_artifacts）`；`link_original → 21548` |

## ④ 图
图 1 = 审计复现的平均流场 `u_x` 云图（h1_re100，6000 步），台阶线与再附线标注；标题显式写"未收敛、非本单工件"。
图 2 = `X1/s` 随步数轨迹 vs 双参考 ±10 % 带，显示两条复现都未稳态（6 k 步不能当验收点）。
图 3 = evidence 键 × L1–L5 判据矩阵，红格（run_artifacts / integration_callers / benchmark）就是机器降级的依据。
三图均由 `audit/m21631/make_figs.py` 从 `audit/m21631/audit_repro/` 的工件生成，零手抄。

## ⑤ 注意点
* "作业 succeeded" ≠ "有工件"：cj3511 rc=0，但 fetch 只回传递交快照（206 MB 源码 tar），run 产物未落盘；取证必须以磁盘
  上的 `result.json/field.npz/wall_shear.csv` 为准。
* "分支有增量" ≠ "主线有增量"：本单 7 个改动文件全在未合并分支上，trunk 一份没有。
* 自声明门 ≠ 基准裁判：`refs.json` 的 3.05/2.878 + 10 % 是仓内自设，不能替代注册表登记行。
* 历史数值不可继承：m15519 的 `re100_m16.json`（2.693，ER=2.0，12000 步）几何/预算都与生产档不等价。
* 观察未定论（F6）：6000 步复现出现 `max_speed=10.59`（`ū=0.1`）与 12 % 质量通量缺口，提示生产点需加通量/速度监控。
* slug 含 FVM，本单只覆盖 LBM 臂；TensorFVM 臂未覆盖。

## ⑥ 误差判定
| 量 | ref | got | 误差 | <10 % | <3 % |
|---|---|---|---|---|---|
| `X1/s`@Re=100 vs Armaly 1983 | 3.05 | **无** | 不可计算 | **未判定** | **未判定** |
| `X1/s`@Re=100 vs Erturk 2008 | 2.878 | **无** | 不可计算 | **未判定** | **未判定** |
| 审计复现（非验收） | 3.05 / 2.878 | 2.9054 | −4.7 % / +0.95 % | 名义在带内但未稳态，**不判通过** | 不判 |

**是否 <3 %：无法回答**（本算例没有任何收敛点数值）。honest FAIL 如实登记，不做粉饰。

## 判级与裁定
L1（数学/单元验证，含 1 个失败用例）达；L2 因 `run_artifacts` 真空不达；L3 未并 main、无调用方；
L4 无注册基准+无 got（WITHHELD）；L5 外部锚仅二手 digitisation。verdict 已落表
`tensor_maturity_verdicts#887`（level=1，board-audit），周期 gate 将写 `reopened` 并为 m21548 开返修轮。
返修最小闭环：生产点 170k 步跑完并取回工件 → 双门判定 → 修 caliper 用例 → ladder/refs/测试入 main + 注册基准行。
"""


def main() -> None:
    login = subprocess.run(
        ["curl", "-s", "-X", "POST", f"{API}/auth/login", "-H", "Content-Type: application/json",
         "-d", json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                           "password": os.environ["TENSOR_AGENT_PASS"]})],
        capture_output=True, text=True, check=True)
    tok = json.loads(login.stdout)["access_token"]

    figs = []
    for name, cap in [
        ("fig1_ux_field_audit_repro.png",
         "审计复现的平均流场 u_x（h1_re100，ER=51/26，6000 步，未收敛）：台阶线与再附线 X1/s=2.905；非本单交付工件"),
        ("fig2_x1s_vs_steps.png",
         "X1/s 随步数轨迹 vs Armaly 3.05 / Erturk 2.878 的 ±10 % 带：两条审计复现均未达稳态，6k 步不能作为验收点"),
        ("fig3_evidence_matrix.png",
         "evidence 键 × maturity_gate L1–L5 判据矩阵：run_artifacts/integration_callers/benchmark 为红格，即机器降级到 L1 的依据"),
    ]:
        figs.append({"name": name, "caption": cap,
                     "png_base64": base64.b64encode((FIGS / name).read_bytes()).decode()})

    payload = {
        "title": "交差核验 m21548 · CASE-BACKSTEP-FVM-RE100（后台阶 Re=100）→ 功能实质 L1，裁定返修",
        "summary": "m21548 的算例无任何运行工件、基准注册表 0 行、交付测试 1 failed、分支未并入 main；"
                   "判定 L1（数学/单元验证），L2/L4 不可达，verdict#887 落表，gate 将判 reopened。",
        "scenario_key": "CASE-BACKSTEP-FVM-RE100",
        "mission_id": 21631,
        "tags": ["audit", "board-audit", "TensorLBM", "CASE-BACKSTEP-FVM-RE100", "m21548",
                 "maturity-L1", "reopened"],
        "markdown": MARKDOWN,
        "figures": figs,
    }
    Path("/tmp/report.json").write_text(json.dumps(payload))
    r = subprocess.run(
        ["curl", "-s", "-X", "POST", f"{API}/tensor/reports",
         "-H", f"Authorization: Bearer {tok}", "-H", "Content-Type: application/json",
         "-d", "@/tmp/report.json"], capture_output=True, text=True, check=True)
    print(r.stdout[:600])


if __name__ == "__main__":
    main()