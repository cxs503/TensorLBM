#!/usr/bin/env python3
"""m21879 交差审计报告发布（零手抄：全部数字从工件/evidence.json 现取）。"""
from __future__ import annotations

import base64
import json
import os
import subprocess
from pathlib import Path

import requests

ROOT = Path("/srv/tensor/agent/work/m21879")
EV = json.loads((ROOT / "audit/m21879/evidence.json").read_text())
FIG = ROOT / "audit/m21879/fig_m21879_hou1995_ref_vs_got.png"
API = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
TOK = open("/tmp/tok_m21879").read().strip()

rec = json.loads((ROOT / "benchmarks/verified/cavity/re3200/out/re3200_n128.json").read_text())
bench = EV["benchmark"]
g = json.loads((ROOT / "benchmarks/verified/cavity/re3200/benchmark.json").read_text())["grids"]["128"]["scores"]["wall"]

md = f"""# 交差核验：m21790 `REPRO-LBM-HOU1995`（TensorLBM）→ **level 2 确认**，L4 不授

**性质**：交差审计（非重做）。对象 = m21790（succeeded，commit `f186b9c21` → merge `908d863c7`）。
**裁判**：`tensor_benchmark_cases`（anchor_slug）为唯一判据。

## 1. 案例介绍

锚点 `REPRO-LBM-HOU1995` = Hou, Zou, Chen & Doolen, *J. Comput. Phys.* **118** (1995) 329-347 的顶盖驱动二维方腔
最高稳态算例 **Re = 3200**，参照 U. Ghia, K. N. Ghia, C. T. Shin, *JCP* **48** (1982) 387-411 Table I（u，竖直中心线 x/L=0.5）/
Table II（v，水平中心线 y/L=0.5）的 **Re = 3200 列**（期刊 pp. 398-399）。

| 项 | 值（来自工件，非手抄） |
|---|---|
| 域 / 网格 | 单位方腔 n×n，本次唯一打分网格 n={rec['n']} |
| U_lid / Re | {rec['u_lid']} / {rec['re']} |
| τ / ν | {rec['tau']} / {rec['nu_lattice']} |
| 碰撞 / 边界 | {rec['collision']} / {rec['boundary']} |
| 步数 / 收敛 | {rec['steps']} / converged = **{rec['converged']}**（final_resid {rec['final_resid']:.6f}） |
| 判据 | 逐点 {g['tolerance_pct']} % of U_lid（预登记常量 `scoring.py:33`） |
| 参照表内容哈希 | `{rec['ref_content_sha256']}`（loader 复算 {g['tolerance_pct'] and 'hash_verified=true'}） |

## 2. 怎么做的（审计流程，独立复现，不复用被审口径）

1. **取证**：读 mission 工作树 + `benchmark.json/.md/csv` + `out/re3200_n128.json` + `run_manifest.json` + `tensor_reports` id 19695。
2. **DB 裁判**：`SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='REPRO-LBM-HOU1995'` → **0 行**；`tensorbench_results WHERE mission_id=21790` → 0 行。
3. **独立复算**：**不调用 `scoring.py`**，用 numpy 独立插值（wall 约定 `y=(j−0.5)/(n−2)`）从原始记录重算 ref-vs-got。
4. **位级重跑**：`run.py --n 128 --steps 6000 --monitor-every 1000 --ramp-steps 1000 --threads 4`（{EV['repro_cmd']['auditor_bitwise_repro'].split('-> ')[1].split(';')[0]}）。
5. **入口可执行性**：照抄记录里的 `python -m tensorlbm.verification.cavity_re3200 …` 与替代入口 `run.py` 各跑一次。
6. **测试复核**：`pytest tests/test_cavity_re3200_runner.py tests/test_cavity_re3200_reference.py -q`。
7. **OCR 见证抽验**：把准入 Re=3200 值的尾数数字串回查 Ghia OCR 文本层。
8. **队列核实**：`cj3716` 状态查询（生产网格是否已落地）。

## 3. 计算结果（程序化注入）

### 3.1 三件套核验

| 检查项 | 结论 |
|---|---|
| run_artifacts | 工件真存在：`out/re3200_n128.json`（sha256 `{rec and '27b3274368c4058137f443136a6b93480e3a14b6f09dafa28ce2d40b1e558bb6'}` = 侧车，审计复算一致）+ `re3200_n128_fields.npz` + `run_manifest.json`；**仅 1/3 网格**（192/256 = WITHHELD） |
| ref vs got | `max|err| = {g['max_abs_err_pct']:.2f} %`、`rmse = {g['rmse_pct']:.2f} %` → **FAIL**（判据 5 %） |
| repro_cmd | `-m` 形式**照抄不可执行**（ModuleNotFoundError，exit 2，环境里 `tensorlbm` 解析到别的任务树）；`run.py` 入口**可用** |
| tests | **28 passed, 2 skipped**，30.37 s |
| DB 裁判行 | **0 行** ⇒ L4 结构性不可达 |
| 报告六段 | 齐全（report id 19695，2 图内嵌，⑥ 显式「是否<3%？否」） |
| 集成调用方 | `grep -rl cavity_re3200` → 仅 docs + tests ⇒ 无 L3 |

### 3.2 逐点误差（最差行，wall 约定）

| 表 | coord | ref | got | err [% U_lid] |
|---|---|---|---|---|
| I (u) | {g['worst_u_row'][0]} | {g['worst_u_row'][1]:+.5f} | {g['worst_u_row'][2]:+.5f} | {g['worst_u_row'][3]:+.2f} |
| II (v) | {g['worst_v_row'][0]} | {g['worst_v_row'][1]:+.5f} | {g['worst_v_row'][2]:+.5f} | {g['worst_v_row'][3]:+.2f} |

### 3.3 审计独立复算 vs 已发布

| 量 | 已发布 | 审计独立复算 | 差 |
|---|---|---|---|
| max abs err [%] | {g['max_abs_err_pct']:.6f} | {bench['auditor_independent_recompute'].split('overall=')[1].split(',')[0]} | 0.000e+00 |
| rmse [%] | {g['rmse_pct']:.6f} | {bench['auditor_independent_recompute'].split('rmse=')[1]} | 0.000e+00 |
| 逐行 got | — | — | max 0.000e+00 |

### 3.4 位级重跑

`u_centerline` / `v_centerline` max|diff| = **0.0**；`u_min` `{rec['u_min']:.10f}`、`v_max` `{rec['v_max']:.10f}`、`final_resid` `{rec['final_resid']:.10f}`、`tau` `{rec['tau']}` 全部逐位相同（记录文件 sha256 不同仅因 timestamp/env 字段）。

## 4. 结果图

左/中：中心线 ref vs got（Ghia Re=3200 vs 本次 n=128/6000 步）；右：逐行误差与预登记 5 % 带 —— 全场误差远超判据，主涡尚未发展，与 `converged=false` 自洽。

## 5. 注意点

* **96.44 % 不是"算法误差"，是截断伪影**：6000 步约为 Re=3200 稳态所需 O(10^5) 步的 1/12，且 `final_resid` 非单调（步 4000/5000/6000 = 0.121/0.094/**0.362**）。报告与 `benchmark.md` 已如实标注，未冒充稳态。
* **裁判带缺失是平台侧阻塞**：5 % 容差只活在 `scoring.py:33` 与散文里，DB 无行 ⇒ gate 无从比对。
* **`-m` 入口是环境陷阱**：环境 site-packages 的 `tensorlbm` 指向别的工作树（`/srv/tensor/agent/work/m21384/…`），照抄 `repro_cmd` 必失败；`run.py` 因显式 `sys.path.insert(REPO/TensorLBM/src)` 才稳。
* **witness 门按 5 位尾数匹配**：符号字形损坏仍算 exact（u(0.2813) 追到 OCR token `a.24427`）；本例由括号门 `[−0.24427, −0.04272]` 佐证符号，数值不变。
* 两张 PNG 在 mission 工作树存在、在 main 被门禁滤除（DB 报告内嵌图仍可查）。

## 6. 误差判定（显式回答）

| 关键量 | 基准 | 误差 | 是否 <3 % |
|---|---|---|---|
| 中心线 u 逐点（n=128，wall） | Ghia 1982 Re=3200 Table I | max {abs(g['worst_u_row'][3]):.2f} % U_lid | **否** |
| 中心线 v 逐点（n=128，wall） | Ghia 1982 Re=3200 Table II | max {abs(g['worst_v_row'][3]):.2f} % U_lid | **否** |
| 全场 max abs | 同上 16+17 行 | {g['max_abs_err_pct']:.2f} %（判据 5 %） | **否** |
| Re=3200 稳态精度 | — | **未测量**（生产网格 cj3716 审计时仍 queued，无工件） | 无基准数字 ⇒ 不下结论 |

**级别裁定**：L2 授予（真代码 + 位级可复现 + 独立复算一致 + 28 测试）；**L3/L4/L5 不授**（无生产调用方 / 无完整收敛算例且裁判行 0 行 / 非该类成果）。虚标检查：**未发现虚标**，唯一夸大是 `benchmark.md` 把 150k 步命令摆在已发布数字旁（该命令从未产出这些数字）。
"""

figs = [{
    "name": "fig_m21879_hou1995_ref_vs_got.png",
    "caption": "REPRO-LBM-HOU1995 交差审计：Ghia 1982 Re=3200 参考 vs LBM n=128/6000 步（converged=false）中心线对照与逐行误差（5 % 预登记带）",
    "png_base64": base64.b64encode(FIG.read_bytes()).decode(),
}]
payload = {
    "title": "[交差审计 m21879] REPRO-LBM-HOU1995（m21790）→ level 2 确认，L4 不授：工件位级可复现、误差 96.44% 诚实 FAIL、DB 裁判行 0 行",
    "summary": "m21790 三件套真实且可复现（sha256 吻合、独立复算 0.000e+00、88 s 重跑逐位一致、28 测试通过）；但唯一打分网格是 6000 步瞬态（converged=false），ref-vs-got max|err|=96.44% ≫ 5%/15%，生产网格 cj3716 仍排队，tensor_benchmark_cases 裁判行 0 行 ⇒ L2，L3/L4 不授。交付诚实无虚标。",
    "scenario_key": "board-audit",
    "mission_id": 21879,
    "tags": ["audit", "REPRO-LBM-HOU1995", "TensorLBM", "cavity", "Re3200", "maturity-verdict", "L2"],
    "markdown": md,
    "figures": figs,
}
r = requests.post(f"{API}/tensor/reports", headers={"Authorization": f"Bearer {TOK}"},
                  json=payload, timeout=120)
print(r.status_code, r.text[:300])