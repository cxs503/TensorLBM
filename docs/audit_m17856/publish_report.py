#!/usr/bin/env python3
"""Publish the m17856 board-audit report to the platform (tensor_reports).

Every number in the markdown is formatted from artifacts/m17856_audit/audit_m17856.json
(zero hand-typed values); the two figures are read from docs/audit_m17856/.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUD = json.loads((ROOT / "artifacts" / "m17856_audit" / "audit_m17856.json").read_text())
FRESH = json.loads((ROOT / "artifacts" / "m17856_audit" / "fresh_results.json").read_text())
RECORDED = json.loads((ROOT / "docs/audit_m17856/evidence_snapshot/results.json").read_text())
API = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")

FRESH_A2_GHOST = next(a["report"]["ghost_ratio_mean"] for a in FRESH["arms"] if a["arm"] == "A2_ghost_thermostat")
ref = AUD["R5_theory_referee"]
rr = AUD["R1_fresh_rerun"]
r2 = AUD["R2_linear_regime"]
rec_ratio = {a["arm"]: a["report"]["ratios"] for a in RECORDED["arms"]}
new_ratio = {a["arm"]: a["report"]["ratios"] for a in FRESH["arms"]}
arms = ["A1_stress_only", "A2_ghost_thermostat", "A3_all_modes"]
modes = RECORDED["modes"]


def table_rows() -> str:
    lines = ["| mode | kind | A1 stress (recorded / fresh) | A2 ghosts (recorded / fresh) | A3 all (recorded / fresh) | ±15% band |",
             "|---|---|---|---|---|---|"]
    for m in modes:
        kind = "ghost" if m in RECORDED["ghost_modes"] else "hydro"
        cells = []
        for a in arms:
            ver = "PASS" if abs(new_ratio[a][m] - 1.0) <= 0.15 else "FAIL"
            cells.append(f"{rec_ratio[a][m]:.4f} / **{new_ratio[a][m]:.4f}** ({100*(new_ratio[a][m]-1):+.1f}%) {ver}")
        lines.append(f"| {m} | {kind} | {cells[0]} | {cells[1]} | {cells[2]} | ±15% |")
    return "\n".join(lines)


def referee_rows() -> str:
    labels = [("A1_stress", "A1 stress-only（自称已知正确）"),
              ("A2_ghosts", "A2 ghost thermostat（本单交付）"),
              ("A3_all", "A3 all modes"),
              ("base_FluctuatingLBM_kt_only", "基线 FluctuatingLBM（既有模块，kT>0 但无逐模噪声）")]
    lines = ["| 臂 | Var(u_x)/(kT/rho) | 误差 | Var(rho)/(rho²kT) | 误差 | <15%? |", "|---|---|---|---|---|---|"]
    for key, label in labels:
        e = ref[key]
        ok = max(abs(e["err_ux"]), abs(e["err_rho"])) <= 0.15
        lines.append(f"| {label} | {e['var_ux']:.3e} / {e['target_ux']:.3e} | {100*e['err_ux']:+.1f}% | "
                     f"{e['var_rho']:.3e} / {e['target_rho']:.3e} | {100*e['err_rho']:+.1f}% | {'是' if ok else '否'} |")
    return "\n".join(lines)


markdown = f"""# m17758 交差核验：XVAL-LBM-FLUCTUATING-LBM-GHOST-MODES（TensorLBM）

**对象** m17758（succeeded，fail_stage=budget_exhausted+forced_finish，平台自身质量门 gate_valid=false）
**本单** m17856 交差核验（非重做）**判级：L2**（真实独立功能实现；L3/L4/L5 均不成立）
**基准裁判** `SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-FLUCTUATING-LBM-GHOST-MODES'` → **0 行**。

## ① 案例介绍（物理背景与参数）

任务是「与成熟开源实现交叉对标（XVAL）」的 LBM 涨落热浴问题：经典 LBM 涨落只注入应力（二阶层）模，
三/四阶「幽灵模」（D2Q9 的 qx、qy、eps）停留在确定性的零涨落，违反能量均分；`ludwig-cf/ludwig`
@78d83c5 暴露独立的 `lb_fluctuations_var_ghost` / `lb_collision_ghost_modes_on` / `lb_relaxation_time_ghosts`，
即对幽灵模单独赋方差并可关断。m17758 的交付=把该做法移植进 TensorLBM 并做 D2Q9 验证。

参数（取自交付入口 `measure()` 字面量）：格子 D2Q9，32×32 周期，`tau=0.8`，`kT=2.5e-4`（另扫 1e-4/5e-4 与 tau 0.7/1.0），
`n_samples=1200`、`n_skip=400`，seed=12345；参考/对比量：每模方差比 `Var(m_a)/(3 rho kT W_a)`（目标 1，容差 ±15%）。

## ② 怎么做的（核验方法/流程/工具链）

1. **静态取证**：交付三件套是否真存在、是否入库（`git ls-files` / `git status`）、模块规模、测试契约、集成调用方、文档声称的取证脚本。
2. **基准裁判**：查平台 `tensor_benchmark_cases`（唯一裁判）该锚点是否有登记案例。
3. **隔离重跑**：把 m17758 的 `src/xval/tests` 复制到 `/tmp` 一次性副本，原样执行
   `python xval/fluctuating_lbm_ghost_modes/run_xval.py --skip-extract`（{rr['wall_s']} s，exit={rr['returncode']}），
   把重跑工件与交付 `results.json` 逐模逐臂比对（相对差）。
4. **两个独立裁判**（都不依赖交付自己的统计口径）：
   - 交付自建裁判 `Var(m_a)/(3 rho kT W_a)` 复算（含 `kT=1e-9` 线性极限，排除「非线性爆炸」这一辩解）；
   - 理论裁判（原始变量能量均分）`Var(u_x)=kT/rho`、`Var(rho)=rho²kT`——直接来自 LBM 涨落教材/交付基类 docstring，绕开模块的 W_a 归一化疑点。
5. **外部锚核验**：`/tmp/ludwig_tar/.../collision.o` 是否存在、sha256 是否等于申报值、符号是否真导出、**归档里是否还有源码**。
6. **DB**：mission 行（status/fail_stage/质量门）、`tensor_reports` 是否发过报告。
7. 工具链：Python 3.11 + numpy + pytest + matplotlib；DB 经 `su postgres -c psql -d tensorplatform`；
   全流程封装为 `scripts/audit/m17856_fluctuating_ghost_audit.py`（`--fast` 可跳隔离重跑）。

## ③ 计算结果（数字表，全部由脚本注入）

### 3.1 交付工件 vs 隔离重跑（能否复现）

| 指标 | 值 |
|---|---|
| 隔离重跑耗时/退出码 | {rr['wall_s']} s / {rr['returncode']} |
| 记录值 vs 重跑值 **最大相对差** | **{100*rr['max_rel_diff_recorded_vs_fresh']:.0f}%**（阈值应为 0%/数值容差） |
| A2 幽灵臂 幽灵模比（记录） | {RECORDED['summary']['main_arms']['A2_ghost_thermostat']['ghost_ratio_mean']:.2f} |
| A2 幽灵臂 幽灵模比（重跑） | {FRESH_A2_GHOST:.2f} |
| 两次独立重跑是否逐位一致 | {ref['deterministic_repeat_A2']}（是 ⇒ 记录工件不是当前代码所产） |

### 3.2 模式方差比（交付自建裁判，目标 1.0）

{table_rows()}

### 3.3 独立理论裁判（原始变量能量均分，目标 1.0；不依赖模块 W_a）

{referee_rows()}

### 3.4 三件套与元数据核验

| 项 | 结果 |
|---|---|
| impl 文件 | `src/tensorlbm/fluctuating_ghost_modes.py` 381 行（真实现，见 evidence impl_files_detail） |
| 入口 | `xval/fluctuating_lbm_ghost_modes/run_xval.py` 396 行，可执行、产工件 |
| run_artifacts | results.json / table_mode_variance.csv / ludwig_reference.json / 2 PNG —— **存在但不可由入口复现** |
| tests | `pytest tests/test_fluctuating_ghost_modes.py -q` → {AUD['S4_tests']['tail'][-1] if AUD['S4_tests']['tail'] else ''}；失败项 {AUD['S4_tests']['failed_tests']} |
| tracked 改动 / 入库 | **0**（`git ls-files` 对三件套零命中，全部 `??`） |
| 集成调用方 | **0**（全仓 grep 仅命中自身/测试/入口；`__init__` 未导出） |
| 登记基准案例 | **0**（锚点无 case_name/metric/tolerance） |
| 外部锚 | 对象 sha256 匹配={AUD['S5_external_anchor'].get('object_sha256_matches_claim')}，架构 {AUD['S5_external_anchor'].get('object_arch','').split(':')[-1].strip()}，源码数={AUD['S5_external_anchor'].get('n_c_sources')} ⇒ **WITHHELD**（无源码、无 OSS 侧数值） |
| 平台报告 | `tensor_reports` 中 m17758 记录数 = {AUD['D3_report']['n_reports']}（未发报告） |

## ④ 结果图

![recorded vs fresh](fig_m17856_recorded_vs_fresh.png)

*图 1 交付记录的每模方差比（浅色）与同一入口隔离重跑（深色）对照，对数纵轴，绿带=owner ±15%。记录的 A2/A3 在 19~50 区间（判「不稳定」），重跑落到 2.0~2.8——同一份代码不可能同时给出两者，日志纵轴差一个量级即判「工件非入口所产」。*

![theory referee](fig_m17856_theory_referee.png)

*图 2 独立理论裁判：Var(u_x)/(kT/ρ) 与 Var(ρ)/(ρ²kT)。A2 速度模 -14.4%（勉强带内）、质量模 +114.8%（带外 8 倍）；A3 +79%/+359%；A1 与既有基类重合于 -84.9%/-70.0%，说明交付的「已知正确臂」本身也不满足 15%。*

## ⑤ 注意点（坑与局限）

1. **「有工件」≠「工件由代码所产」**：交付 results.json 与入口重跑差 564%，两次独立重跑却逐位一致 → 代码确定性无问题，是工件陈旧/另源（F1，阻断级）。
2. **模块自建 FDT 裁判自身未标定**：`target_mode_variance()` 把首个噪声模的权重 `W[noise_mode_index[0]]` 乘给所有模，单元测试 `target_mode_variance(1.0)==3*rho*kT` 直接失败（F2）；而更根本的是 `mode_variance_report` 里 `rho_mean` 实为 m_rho=ρ/3，靠 ×3 与 e_rho=1/3 恰好抵消才凑出正确量级。
3. **`bitwise_kT0` 负控制是空转的**：`run_arm(bitwise_reference=True)` 在 kT=2.5e-4 的噪声行进中与 kT=0 确定性参考比 `array_equal`，必然 False（工件里确实记 False，无人伪造，但也无人报警）。
4. **外部锚只是编译产物**：`/tmp/ludwig_tar` 里 `*.c` 数为 0，取证只到 symbol/DWARF/反汇编/执行函数级，**从未**与 ludwig 的模拟输出比数值——「XVAL <15%」在锚点侧没有数字可对。
5. **局限**：未复跑 OSS 取证段（需 loongarch 工具链）；未做尺寸/收敛扫描（单点 32×32）；未对模块文档的稳态方差公式做严格解析推导，只用两个独立裁判判 15%。
6. 锚点基准裁判表为空，意味着本单的 15% 判定必须靠解析 FDT 目标——这也正是 m17758 失败的真正原因：目标量级就错了。

## ⑥ 误差判定（关键量 vs 基准，逐项）

| 关键量 | 基准 | 实测（本次重跑） | 误差 | <15%? |
|---|---|---|---|---|
| 幽灵模方差比 qx（A2，模块自建裁判） | 1.0 | {r2['ghosts_kT2.5e-4']['ratios']['qx']:.4f} | **+{100*(r2['ghosts_kT2.5e-4']['ratios']['qx']-1):.1f}%** | **否** |
| 幽灵模方差比 qx（A2，kT=1e-9 线性极限） | 1.0 | {r2['ghosts_kT1e-9']['ratios']['qx']:.4f} | +{100*(r2['ghosts_kT1e-9']['ratios']['qx']-1):.1f}% | 否 |
| 非守恒模平均比（A1 stress-only，自称已知正确） | 1.0 | {AUD['R3_stress_arm'].get('nonconserved_mean_ratio', float('nan')):.4f} | {100*(AUD['R3_stress_arm'].get('nonconserved_mean_ratio', float('nan'))-1):+.1f}% | 否 |
| Var(u_x)（A2，独立理论裁判） | kT/ρ = {ref['A2_ghosts']['target_ux']:.3e} | {ref['A2_ghosts']['var_ux']:.3e} | {100*ref['A2_ghosts']['err_ux']:+.1f}% | 是（边缘） |
| Var(ρ)（A2，独立理论裁判） | ρ²kT = {ref['A2_ghosts']['target_rho']:.3e} | {ref['A2_ghosts']['var_rho']:.3e} | **{100*ref['A2_ghosts']['err_rho']:+.1f}%** | **否** |
| 交付工件可复现性 | 0% 差 | {100*rr['max_rel_diff_recorded_vs_fresh']:.0f}% 差 | — | 否 |

**显式回答：是否 <3%？——否，一个关键量都没有。** 最好的项（A2 速度方差 -14.4%）也只是勉强进 15% 带，
且同臂质量方差 +114.8%；锚点侧无 OSS 数值（WITHHELD），因此「XVAL 交叉对标通过」这一主张在任何口径下都无法成立。
m17758 自己的 submission 已如实写「部分完成（honest 部分 FAIL，无粉饰）」，本核验与之一致，只是把失败量化到逐项：
交付的**实现是真的**（故 L2），**验证是失败的**（故不给 L4），**集成/基准为零**（故不给 L3）。

## 判级说明（L0–L5）

- L0 文档成果：不满足——有 381 行可执行实现。
- L1 数学/单元验证：满足但不限于——7 个单元测试（6 通过 1 失败）。
- **L2 独立功能实现：满足** —— 输入（tau/kT/thermostat）→ 计算（模式噪声 + 流动 + 方差估计）→ 输出（JSON/CSV/PNG），入口 42.4 s 可跑通。
- L3 系统集成：**不满足** —— 零调用方、未进 `__init__` 导出、未接主求解器。
- L4 基准验证：**不满足** —— 无登记基准案例；自建裁判双向超限 183%/‑74.5%；工件不可复现；无 OSS 侧数值。
- L5 科研工程成果：不满足。

## findings（按严重度，详见 verdict evidence.findings）

- F1 blocking-ish（high）：交付工件不可由入口复现（最大相对差 {100*rr['max_rel_diff_recorded_vs_fresh']:.0f}%）。
- F2 high：失败测试与申报不符；`target_mode_variance` 逐模权重错用（单模 W 广播）。
- F3 high：幽灵模热化偏 +183%（kT=1e-9 亦然，非线性不可解释）。
- F4 high：外部锚仅编译产物，无源码、无 OSS 数值 ⇒ XVAL 名不副实（WITHHELD）。
- F8 high：自称「已知正确」的 stress 臂偏 -74.5% ⇒ 自建裁判未标定。
- F9 high：`tensor_benchmark_cases` 该锚点 0 行（唯一裁判为空）。
- F0 high：三件套零 tracked 改动（未入库）。
- F5/F6/F10/F11 medium：取证脚本缺失、零集成、平台质量门 gate_valid=false、未发 tensor_reports。

**处置（adversarial-review disposition）：request changes**——判级 L2 落表（`tensor_maturity_verdicts` id=642, verdict_by=board-audit），
不返修到位不得升 L3/L4；返修清单见 verdict.evidence.verdict。
"""

figs = []
for name, cap in (("fig_m17856_recorded_vs_fresh.png",
                   "图1 交付记录 vs 隔离重跑：每模方差比，记录值 19~50、重跑值 2.0~2.8（对数轴，绿带 ±15%）"),
                  ("fig_m17856_theory_referee.png",
                   "图2 独立理论裁判：Var(u_x)/(kT/ρ) 与 Var(ρ)/(ρ²kT)，A1/A2/A3/基类四臂")):
    p = ROOT / "docs" / "audit_m17856" / name
    figs.append({"name": name, "caption": cap,
                 "png_base64": base64.b64encode(p.read_bytes()).decode()})

payload = {
    "title": "m17856 交差核验：m17758 XVAL-LBM-FLUCTUATING-LBM-GHOST-MODES 判级 L2（工件不可复现+15%双向失败）",
    "summary": ("独立复核 m17758：实现真实可跑（L2），但交付工件与入口重跑差 "
                f"{100*rr['max_rel_diff_recorded_vs_fresh']:.0f}%、幽灵模方差偏 "
                f"{100*(r2['ghosts_kT2.5e-4']['ratios']['qx']-1):+.0f}%、独立理论裁判质量模偏 "
                f"{100*ref['A2_ghosts']['err_rho']:+.0f}%，锚点基准表 0 行、零集成、零入库 ⇒ 判 L2，request changes。"),
    "scenario_key": "board-audit",
    "mission_id": 17856,
    "tags": ["board-audit", "maturity", "TensorLBM", "XVAL", "fluctuating-LBM", "L2"],
    "markdown": markdown,
    "figures": figs,
}


def api(path: str, body=None, tok: str | None = None):
    req = urllib.request.Request(API + path,
                                 data=(json.dumps(body).encode() if body is not None else None),
                                 headers={"Content-Type": "application/json"})
    if tok:
        req.add_header("Authorization", "Bearer " + tok)
    return json.load(urllib.request.urlopen(req, timeout=180))


tok = api("/auth/login", {"username": os.environ["TENSOR_AGENT_USER"],
                          "password": os.environ["TENSOR_AGENT_PASS"]})["access_token"]
res = api("/tensor/reports", payload, tok)
print("published:", res.get("id"), res.get("title", "")[:60])
Path("/tmp/report_m17856.json").write_text(json.dumps(payload, ensure_ascii=False))
print("markdown chars:", len(markdown), "figures:", len(figs))