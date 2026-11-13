#!/usr/bin/env python3
"""Publish the m17884 board-audit report (numbers injected from audit_numbers.json)."""
from __future__ import annotations

import base64
import json
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
NUM = json.loads((HERE / "audit_numbers.json").read_text())
G = NUM["grade"]
C = NUM["controls"]
LAD = NUM["resolution_ladder"]
DIFF = NUM["repro_numeric_diffs"]
B = NUM["benchmark_judge"]


def f(x, n=6):
    return f"{x:.{n}g}"


rows_grade = "\n".join([
    f"| `u_x` rel.L2 @T=800 | 解析 TG | {f(G['ux_rel_l2_oss'])} | {f(G['ux_rel_l2_oss'])} | ✓ | ✓ |",
    f"| `u_y` rel.L2 @T=800 | 解析 TG | {f(G['uy_rel_l2_oss'])} | {f(G['uy_rel_l2_oss'])} | ✓ | ✓ |",
    f"| `rho` rel.L2 @T=800 (O(u²) 涨落归一) | 解析 TG | {f(G['rho_rel_l2_oss'])} | {f(G['rho_rel_l2_oss'])} | ✓ | ✓ |",
    f"| `nu` 由 E(t) 反演 | {f(G['nu_ref'],8)} | {f(G['nu_fit_oss'],8)} | {f(G['nu_rel_err'])} | ✓ | ✓ |",
    f"| 质量 ρ₀·NX·NY | {G['mass_expected']:.0f} | {G['mass_final_oss']:.9f} | {f(G['mass_rel_err_oss'],3)} | ✓ | ✓ |",
    f"| 仓内 AB vs fused `u_x` | — | {f(G['ab_vs_fused_ux_rel_l2'],3)} | — | ✓ | ✓ |",
])

rows_ctrl = "\n".join(
    f"| `{k}` | {f(v['analytic_ux_or_uy_err'])} | {v['gate']} |" for k, v in C.items())

rows_res = "\n".join(
    f"| {p['nx']} | {p['steps']} | {f(p['ux_rel_l2'])} | {f(p['rho_rel_l2'])} | {f(p['nu_rel_err'])} |"
    for p in LAD)

diff_lines = "\n".join(
    f"| `{k}` | {v['committed']:.6g} | {v['fresh']:.6g} | 计时字段 |" for k, v in DIFF.items())

MAG = G["rho_magnitude_check"]
orders = G["resolution_order_rederived"]

md = f"""# 交差审计 m17884：m17839 `XVAL-LBM-FUSED-PUSH-PULL-AA-PATTERN` 判级 L2

**对象**：TensorLBM mission 17839，commit `eee0988a`（`XVAL: cross-validate fused push-pull (upstream OSS ch.13) against OSS port and AB reference`），平台报告 16786。
**审计**：mission 17884，`verdict_by='board-audit'`，判决行已落 `tensor_maturity_verdicts`。
**一句话结论**：交付的**每一个数字都真**（重跑位级复现、12 测试全绿、负控制确实能打假），但**级别=L2**——基准裁判行缺位（L4 不可确认）、新模块零调用方未接入主求解器（L3 不成立）、上游 CUDA 从未执行。

## ① 案例介绍（被审对象是什么）

上游对标源：`lbm-principles-practice/code` chapter13/gpu 的 fused push-pull（AA pattern）——把 streaming 与 collision 融进一个 kernel：pull 八个平移到邻居的碰撞后分布量 → 由这九个数算 ρ/u → BGK 弛豫 → 直接写进**另一组**八方向缓冲（rest 方向留在自己的数组），全程不存在中间 streamed 场。上游存 17 doubles/node。

物理算例：二维 Taylor–Green 涡衰减，D2Q9 BGK，周期边界，格点单位。

| 参数 | 值 |
|---|---|
| NX×NY | 64×64（分辨率梯 32/64/128） |
| ν, τ | {f(G['nu_ref'],8)}, 1.0 |
| u_max | 0.02（Ma = 0.0346） |
| 步数 | 800（T/τ_decay ≈ 2.57） |
| 初值 | 上游 `taylor_green_eval(t=0)` 平衡态，含 O(u²) 压强项 |
| 参考解 | 闭式解析 TG（制造解验证） |
| 门限 | 15%（owner 精度标准）；更严目标 3% |

## ② 怎么做的（审计方法）

1. 取证：读 `result.json`（2532 个数值叶）、`docs/XVAL_m17839_FUSED_PUSH_PULL_AA_PATTERN.md`、`run.py`、`oss_port.py`、`src/tensorlbm/fused_push_pull.py`，并把 `oss_port.py` 逐行对照上游 `ref/LBM.cu:181-252`（pull 索引 / 矩 / `tauinv=2/(6ν+1)` 折叠 / 写另一缓冲）。
2. **独立重跑**：`python xval/fused_push_pull_aa_pattern/run.py --out /tmp/…`（<2 min），与交付工件做结构比对。
3. **机判基准查表**：`SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-FUSED-PUSH-PULL-AA-PATTERN'` → **0 行**（全平台 {B['total_registered_cases']} 行）。
4. 测试实跑：`pytest tests/test_fused_push_pull_xval.py -q` → **12 passed in 6.21s**。
5. 上游快照完整性：复算 `ref/{{LBM.cu,LBM.h,main.cu}}` sha256，与 `ref/PROVENANCE.json` 三行全部一致。
6. 文档零手抄验证：重跑 `make_report.py` → 交付文档 **字节完全相同**（`git status` 仍干净）。
7. 量级复核：独立跑 OSS 转写 801 步，量 ρ/u 的**绝对**误差（不止相对范数）。

## ③ 计算结果（审计员重算，非引用）

### 3.1 解析门限（交付内部判据 + owner 15%/3%）

| 量 | 参考 | got | 误差 | <15% | <3% |
|---|---|---|---|---|---|
{rows_grade}

### 3.2 复现性：重跑 vs 交付工件

结构比对 2532 个叶：**{NUM['repro_diff_count']} 处差异，全部是 wall_s 计时字段；非计时差异 {NUM['repro_non_timing_diff_count']} 处**。

| 路径 | 交付值 | 重跑值 | 性质 |
|---|---|---|---|
{diff_lines}

（附带记录：交付 `torch_ab.wall_s`=117.1 s vs 重跑 1.43 s（81×），原跑在重负载主机上；交付未做任何 Mlups/带宽主张，`storage` 明确写"structural, not a bandwidth measurement"，故不构成缺陷。）

### 3.3 负控制（门限能否咬人）

| 控制臂 | 解析误差 | 判定 |
|---|---|---|
{rows_ctrl}

`flip_all`（全局速度取负）PASS 是**已自曝的盲区**：TG 状态对全局符号翻转对称。

### 3.4 分辨率梯与收敛阶

| NX | steps | `u_x` rel.L2 | `rho` rel.L2 | ν 反演误差 |
|---|---|---|---|---|
{rows_res}

审计员重算阶：**p₁ = {orders[0]:.10f}，p₂ = {orders[1]:.10f}**，与交付记录完全一致。ρ 残差随 h 几乎不动（0.1126 → 0.0289 → 0.0207），支持"不随加密消失"的说法。

### 3.5 ρ 诊断的量级复核（审计补充）

交付的 2.89% 是**按 O(u²) 涨落归一**的相对范数（沿用上游 `LBM.cu:231-262` 约定）。审计员独立测得绝对量级：`max|Δρ| = {MAG['abs_rho_err_max']:.3g}`，`rms|Δρ| = {MAG['abs_rho_err_rms']:.3g}`，而解析 ρ 涨落本身只有 `rms = {MAG['analytic_rho_fluct_rms']:.3g}`；t=801 处相对值 {MAG['rel_l2_rho_t801']:.4f}（相邻步在 0.023–0.029 间摆动，与站立声学模一致）。声学包络周期预测 `π/(c_s·2k) = {G['rho_envelope_period_predicted']:.4f}` vs 实测 {G['rho_envelope_period_measured']:.4f}（差 {100*G['rho_envelope_period_rel_err']:.2f}%）。**结论：ρ 的 2.89% 不是 3% 物理误差，而是 1e-7 量级的诊断放大。**

## ④ 结果图

fig1：四个受判量与 15%/3% 带（对数轴）。fig2：分辨率梯 + 重算斜率。fig3：负控制 + 复现差异范围（4 个计时项、0 个数值项）。

## ⑤ 注意点（审计发现，按严重度）

* **A1（阻断 L4）**：`tensor_benchmark_cases` 对该 anchor **0 行**（平台共 {B['total_registered_cases']} 行），按任务榜铁律"基准判据是唯一裁判"，无论交付内部数字多漂亮都无法授予 L4。
* **A2（阻断 L3）**：`src/tensorlbm/fused_push_pull.py:122` 的 `FusedPushPullStepper` 在 `src/`、`benchmarks/`、`examples/` 内**零引用**，`src/tensorlbm/__init__.py` 导出命中 **0**；模块是 periodic-only，墙体仍留 `solver`。跨臂 AB 参考调用仓内 `solver.stream`/`collide_bgk` 只证明库内数值一致（8.35e-14），不是接入。
* **A3（表述）**："OSS arm" 是同作者写的 NumPy 转写，上游 CUDA **从未编译执行**（无 GPU/nvcc）。"arm-to-arm bit-identical" 只证明转写可复现，不证明与上游二进制一致；文档 §4.2/§6.1 标签正确，标题易被过度解读。
* **A4（量级）**：见 3.5，ρ 的 2.89% 需给绝对量级才不会误导。
* **A5（控制面覆盖）**：负控制注入在 `OSSFusedStepper` 变体（`run.py:176 _variant_step`），未对交付类注入故障。
* **A6**：`flip_all` 盲区经审计确认真实存在（已在 §6.2 自曝，诚实加分，round 2 应加非对称算例）。
* 适用边界：本结果只在周期边界、单相、无外力、Ma≈0.035 的 TG 上成立；不能外推为"求解器整体精度"。

## ⑥ 误差判定：是否 <3%？

| 关键量 | vs 基准 | 误差 | <3%？ |
|---|---|---|---|
| `u_x` rel.L2 | 解析 TG | {f(G['ux_rel_l2_oss'])} | **是** |
| `u_y` rel.L2 | 解析 TG | {f(G['uy_rel_l2_oss'])} | **是** |
| `rho` rel.L2 | 解析 TG（O(u²) 涨落归一） | {f(G['rho_rel_l2_oss'])} | **是（数值上），但基准是 1e-6 量级场，绝对误差 {MAG['abs_rho_err_max']:.3g}，不宜读作 3% 物理偏差** |
| `nu` 反演 | {f(G['nu_ref'],8)} | {f(G['nu_rel_err'])} | **是** |
| 跨臂 `u_x` | OSS 转写 | {f(G['cross_arm_ux_rel_l2_oss_vs_torch_fused'],3)} | **是（且总体量位级相同）** |
| 外部可裁基准（实验/文献/注册行） | — | — | **无**：`tensor_benchmark_cases` 0 行，锚为制造解 ⇒ L4 不可确认（honest 缺口，不粉饰） |

**判级：L2**（L1/L2 通过；L3 缺集成，L4 缺裁判行与上游执行，L5 继承 L4）。返修清单：①为 anchor 注册 `tensor_benchmark_cases` 行并把 `result.json` 登记为不可变工件；②把 fused 步进接入 `solver`/边界/API 并加调用方；③补一次上游参考实现的可执行对照（容器化 nvcc 或等价的可执行参考）或明确把锚降格为"制造解验证"；④报告标题改成"对 NumPy 转写的位级复现"，并给出 ρ 绝对量级。
"""

figs = []
for p, cap in [
    (HERE / "figures" / "fig1_grades_vs_analytic.png", "四个受判量 vs 解析 Taylor-Green，15% 门限与 3% 目标带（对数轴）"),
    (HERE / "figures" / "fig2_resolution_ladder.png", f"分辨率梯 NX=32/64/128，审计员重算阶 {orders[0]:.2f}/{orders[1]:.2f}"),
    (HERE / "figures" / "fig3_controls_and_repro.png", f"负控制可失败性 + 复现差异范围（{NUM['repro_diff_count']} 项计时 / {NUM['repro_non_timing_diff_count']} 项数值）"),
]:
    figs.append({"name": p.stem, "caption": cap,
                 "png_base64": base64.b64encode(p.read_bytes()).decode()})

payload = {
    "title": "交差审计：m17839 XVAL fused push-pull 数字全真但判级 L2（无基准裁判行 / 未接入求解器）",
    "summary": (
        f"对 m17839（commit eee0988a，TensorLBM）做交差核验：重跑 run.py 与交付 result.json 逐叶比对，"
        f"{NUM['repro_diff_count']} 处差异全为 wall_s 计时、非计时差异 {NUM['repro_non_timing_diff_count']} 处；"
        f"12/12 测试通过；解析门限 ux/uy rel.L2={G['ux_rel_l2_oss']:.6f}（<3%）、nu 反演 {G['nu_rel_err']:.3g}、"
        f"质量误差 {G['mass_rel_err_oss']:.2g}；负控制 nu×1.25→{C['nu_bias_25pct']['analytic_ux_or_uy_err']:.3f}、"
        f"W/E 互换→{C['mirror_x']['analytic_ux_or_uy_err']:.3f} 均 FAIL；收敛阶重算 {orders[0]:.4f}/{orders[1]:.4f} 与记录一致；"
        f"上游源码 sha256 三件全部复算一致、文档重渲染字节相同。"
        f"判级 L2：tensor_benchmark_cases 对该 anchor 0 行（唯一裁判缺位→L4 不可确认）、新模块零调用方零导出（L3 不成立）、"
        f"上游 CUDA 从未执行（跨臂位级相同只证明 NumPy 转写可复现）。verdict 已落表 mission_id=17884, level=2, board-audit。"
    ),
    "scenario_key": "xval_lbm_fused_push_pull_aa_pattern_audit",
    "mission_id": 17884,
    "tags": ["TensorLBM", "XVAL", "board-audit", "m17839", "fused-push-pull", "Taylor-Green", "maturity-L2"],
    "markdown": md,
    "figures": figs,
}
Path("/tmp/report_m17884.json").write_text(json.dumps(payload))
print("payload written", len(md), "chars,", len(figs), "figures")

# publish
token = subprocess.run(
    ["curl", "-s", "-X", "POST", os.environ["TENSOR_API_BASE"] + "/auth/login",
     "-H", "Content-Type: application/json",
     "-d", json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                       "password": os.environ["TENSOR_AGENT_PASS"]})],
    capture_output=True, text=True).stdout
tok = json.loads(token).get("access_token") or json.loads(token).get("token")
resp = subprocess.run(
    ["curl", "-s", "-X", "POST", os.environ["TENSOR_API_BASE"] + "/tensor/reports",
     "-H", f"Authorization: Bearer {tok}", "-H", "Content-Type: application/json",
     "-d", "@/tmp/report_m17884.json"], capture_output=True, text=True).stdout
print(resp[:600])