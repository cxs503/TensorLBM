"""Publish the m17855 board-audit report to the platform (six-element paper-grade report)."""
from __future__ import annotations

import base64
import json
import os
import urllib.request

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
FIG = "/root/tp-agent/work/m17855/artifacts/board_audit_m17855/fig_defaults_divergence.png"
PROBE = json.load(open("/root/tp-agent/work/m17855/artifacts/board_audit_m17855/probe.json"))


def post(path, payload, token=None):
    req = urllib.request.Request(BASE + path, method="POST",
                                 data=json.dumps(payload).encode())
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.status, json.loads(r.read() or b"{}")


st, auth = post("/auth/login", {"username": os.environ["TENSOR_AGENT_USER"],
                                "password": os.environ["TENSOR_AGENT_PASS"]})
tok = auth["access_token"]

tr = PROBE["trace_defaults"]
dflt = PROBE["defaults_registry_run"]
smoke = PROBE["ci_smoke_run"]
cli = PROBE["documented_cli"]

md = f"""## ① 案例介绍

被审对象=m17790（锚点 `XVAL-LBM-EXAMPLE-PER-MAIN-ANTIPATTERN`，仓 TensorLBM，status=succeeded），
本次为**交差核验单 m17855（board-audit，非重做）**。物理对象：2D 方形顶盖驱动方腔，D2Q9 BGK/RLBM，
交付自带默认参数 `n=65, Re=100, u_lid=0.1 → τ=0.692`，对标 Ghia et al. (1982, JCP 48:387) 的中心线 u/v 表。

锚点验收条件（`feature_ledger` 程序化读取）：`<15%（owner 精度标准：对标误差 >15% 不可接受）`，status=`open`。
`tensor_benchmark_cases` 中该锚点**没有 case/metric/tolerance 行**（查询返回 0 行），故数值裁判取锚点级 <15%
＋交付自述的外部基准 Ghia 1982。

## ② 怎么做的

1. 读 mission 行（id=17790）：`head_commit=null`、`checks={{}}`、`branch=null`、`report_owed=true`、`tensor_reports` 0 行；
2. 读被审工作树 `/root/tp-agent/work/m17790`（commit `ee1f41f1`，`git show --stat`：7 文件 +738 行，无任何 run 工件）；
3. 独立复跑（不采信自述）：12 个交付单测、仓内正统 `tests/test_lid_driven_cavity.py`、交付默认配置逐 step 追踪、
   registry 端到端、文档声明 CLI；
4. 机器口径比对：`pyproject.toml` 的 `package-dir=src / pythonpath=['src'] / testpaths=['tests']` 决定"什么才算被打包/被 CI 跑"。

工具链：pytest 8.x + torch 2.7.1(CPU, float64)、matplotlib 3.10.8；复跑脚本 `scripts/audit_m17855_probe.py`
（stage=trace/guard/smoke/cli/figure），工件 `artifacts/board_audit_m17855/`。

## ③ 计算结果（数字表，全部来自 probe.json 程序化注入）

| 核验项 | 期望 | 实测 | 判定 |
|---|---|---|---|
| 交付单测 | 12 pass（自述） | 12 passed / 6.45 s | 与自述一致 |
| 单测是否验精度 | 应有 Ghia 容差断言 | 0 处；smoke 用 n=17,Re=10 → `ghia_available={str(smoke['metrics'].get('ghia_available')).lower()}` | 未触及对标分支 |
| 默认配置稳定性 | τ=0.692 稳定 | `max|u_x|` step10={tr['trace'][1]['max_abs_ux']:.4f}（>u_lid=0.1）、step35=6.917e+107、**NaN onset=step {tr['nan_onset_step']}** | 发散 |
| Ghia 指标（默认配置） | l2_u,l2_v < 0.05 | l2_u={dflt['metrics']['l2_u']}, l2_v={dflt['metrics']['l2_v']}, max_u={dflt['metrics']['max_u']} | 无数值 |
| 失败是否被吞 | success=False | success={str(dflt['success']).lower()}, error={dflt['error']}, elapsed={dflt['elapsed_s']} s | 静默假成功 |
| NaN 守卫 | 拦截发散 | 条件 `step%500==0 and step>0` → `max_steps≤500` 永不触发 | 死守卫 |
| 文档 CLI | 4 条命令可用 | `python3 -m tensorlbm.examples.runner --list` → rc={cli['returncode']}, `ModuleNotFoundError: No module named 'tensorlbm.examples'` | 死路径 |
| 交付 run 工件 | 有 | 0（commit 仅代码/文档/测试；`head_commit=null`） | 无 |
| 平台报告 | 六要素 | `tensor_reports` mission 17790 = 0 行 | 未发 |
| 基准登记 | case/metric/tolerance | `tensor_benchmark_cases` 0 行 | 无 |
| 迁移覆盖 | 声明"100+ scripts" | 实际 1 个 example | 声明>实际 |
| 正统树现有能力 | — | `tests/test_lid_driven_cavity.py` 20 passed / 14.15 s | 能跑的实现没被包 |

## ④ 结果图

图 1 = 默认配置 `max|u_x|` 随 step 演化（对数纵轴）：step 10 即越过顶盖速度 0.1，step 35 达 6.9e107，step 40 起全 NaN
（红点线=NaN onset，绿虚线=u_lid）。
**没有场量云图的原因（如实声明）**：被审交付在自带默认配置下 step 40 即发散，速度场全为 NaN，
无法生成有物理意义的云图；也没有 `ref vs got` 比较曲线，因为 got 是 NaN（这正是判据失败的直接体现）。
可复现的替代图即本图——它直接给出"发散发生时刻"这一缺陷证据。

## ⑤ 注意点（坑与局限）

- **双源码树陷阱**：仓内 `src/tensorlbm/`（1256 个 .py，被打包/被 CI 跑）与 `TensorLBM/src/tensorlbm/`（76 个 .py 旧快照）
  并存；m17790 把新模块放进后者 → 不进 wheel、不进 CI、文档 CLI 必然 `ModuleNotFoundError`。核验任何"新模块"先问"在不在 `src/`"。
- **测试绿≠验证做过了**：12 个绿测试走的全是 `Re=10`（Ghia 表无此行 → `except KeyError` 静默降级），
  `ghia_tolerance=0.05` 一次都没被触发；这正是"测试只执行代码、不断言不变量"的教科书案例。
- **守卫 off-by-one**：`step%500==0 and step>0` 使 `max_steps=500` 时守卫永不执行，NaN 场以 `success=True` 落盘。
- 局限：未跑完整 `max_steps=5000` 默认算例（约 20 min，0.24 s/step）；因 step 40 已 NaN，
  5000 步时守卫会在 step 500 抛 `RuntimeError → success=False`，两种 max_steps 下结论同为"默认配置不可用"，不影响判级。
- 本审计只读被审 worktree，未修改被审代码；所有数字均在 m17855 工作树内以 `scripts/audit_m17855_probe.py` 重跑可复现。

## ⑥ 误差判定（vs 基准，逐项回答是否 <3%）

| 关键量 | 基准（ref） | 实测（got） | 相对误差 | 是否 <3% |
|---|---|---|---|---|
| 中心线 u(y) 剖面 L2 | Ghia 1982, Re=100 | NaN | 不可计算（发散） | **否**（无法达成） |
| 中心线 v(x) 剖面 L2 | Ghia 1982, Re=100 | NaN | 不可计算 | **否** |
| max\\|u−u_ref\\|（u 向） | Ghia 1982 | NaN | 不可计算 | **否** |
| 稳定性判据 | τ>0.5 且场有界 | τ=0.692 但 step 40 NaN | — | **否** |

锚点级验收 `<15%`：**got=NaN，误差不可度量 ⇒ 判 FAIL（honest FAIL，不粉饰）**。

## 判级与落表

`level=1`（L1 单元/冒烟验证成立；L2 因**零 run 工件 + 文档入口不可用**不成立；L3/L4 更不成立），
`verdict=reopened`：`tensor_maturity_verdicts` id=647（repo=TensorLBM, scope=`m17790 XVAL-LBM-EXAMPLE-PER-MAIN-ANTIPATTERN`,
mission_id=17855, verdict_by='board-audit'），evidence 八键含真实文件:行号/工件路径/测试名/复现命令。

**返修清单**：①模块迁到 `src/tensorlbm/examples/` 并改包正统 `lid_driven_cavity` 解算器；②`n≥129,Re=100` 跑出
`l2_u/l2_v<0.05` 并写成 benchmark 测试；③修 NaN 守卫使 NaN→`success=False`；④登记 `tensor_benchmark_cases`
（metric=l2 vs Ghia, tolerance=<0.15）并提交 run 工件；⑤迁移 ≥10 个 example 后再声称消除 antipattern；⑥补发六要素报告。

审计报告全文：`docs/board_audit_m17855.md`；复现：`python3 scripts/audit_m17855_probe.py {{trace|guard|smoke|cli|figure}}`。
"""

png = base64.b64encode(open(FIG, "rb").read()).decode()
payload = {
    "title": "交差核验 m17790 · XVAL-LBM-EXAMPLE-PER-MAIN-ANTIPATTERN：examples 框架真代码但默认算例 NaN、零 run 工件 → L1/reopened",
    "summary": ("对 m17790（TensorLBM examples framework，commit ee1f41f1）做交差核验：12 单测独立复跑通过但只验管道；"
                "自带默认配置 n=65/Re=100/τ=0.692 在 step 40 发散为 NaN 且返回 success=True；文档 CLI 报 ModuleNotFoundError"
                "（模块落在非打包树 TensorLBM/src）；tensor_benchmark_cases 该锚点 0 行，got=NaN 使 <15% 验收 FAIL。"
                "判级 L1，verdict=reopened，verdict 落表 id=647。"),
    "scenario_key": "board_audit_xval_lbm_example_per_main_m17790",
    "mission_id": 17855,
    "tags": ["board-audit", "TensorLBM", "XVAL", "m17790", "maturity-gate", "honest-FAIL"],
    "markdown": md,
    "figures": [{
        "name": "fig_defaults_divergence.png",
        "caption": ("m17790 example 在自带默认配置（n=65, Re=100, τ=0.692）下的 max|u_x| 演化：step 10 越过 u_lid=0.1，"
                    "step 35 达 6.9e107，step 40 起 NaN（红点线）；交付仍返回 success=True。"),
        "png_base64": png,
    }],
}
st, res = post("/tensor/reports", payload, tok)
print("report status", st, "id", res.get("id"))
print(json.dumps({k: v for k, v in res.items() if k != "markdown"}, ensure_ascii=False)[:400])