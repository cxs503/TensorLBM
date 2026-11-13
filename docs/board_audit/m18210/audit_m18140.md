# 交差审计 m18140 — XVAL-LBM-INSTRUCTION-DSL-TESTS-SH（TensorLBM）

**审计单**: m18210 (board-audit) · **被审单**: m18140 (succeeded, 返修轮1 for m17968) · **日期**: 2026-11-13
**verdict**: tensor_maturity_verdicts id=723 · level=**L2** · disposition=**reopened（返修）**

## ① 案例介绍（物理背景/几何/参数）

m18140 的 mission goal 是「返修轮1」：m17968（instruction DSL，本锚点）经 maturity_gate 复核仅 L1，要求补齐 L2/L3/L4 evidence 并真正做完原目标（<15% 判据）。

被审交付物实际内容（工作树 /root/tp-agent/work/m18140）：
- `src/tensorlbm/shallow_water_lbm.py`（233 行）：1-D 浅水方程 D1Q2 BGK LBM 求解器（f→h, g→hu 双分布函数，±λ 速度，s=1/(0.5+σ)）
- `tests/test_shallow_water_lbm.py`（107 行，7 测试）
- `xval/XVAL-LBM-INSTRUCTION-DSL-TESTS-SH/`：run_xval.py 驱动（287 行）+ REPORT.md + predeclared.json + artifacts（result.json + 2 PNG）

算例：① 小振幅高斯波（h0=1, g=1, t=1, 振幅 0.01）；② Ritter 溃坝（h0=1, g=1, t=3, 干床）。分辨率 200/400/800，CFL=0.4，λ=4，σ_h=1e-3，σ_q=0.1。

## ② 怎么做的（审计方法/流程）

1. **取证**：ls/stat 工件；`git ls-files`/`git status --porcelain` 核 commit 状态；SQL 核 tensor_benchmark_cases(anchor) / tensor_maturity_verdicts(17968,18092) / tensor_dev_missions(18140) / tensor_reports(18140)。
2. **测试复跑**：`pytest tests/test_shallow_water_lbm.py -q` → 7 passed in 5.07 s。
3. **repro_cmd 实跑**：`python xval/.../run_xval.py` → rc=0，~30 s，result.json 重新生成，数值与交付版逐位一致（零手抄成立）。
4. **基准裁判**：`SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-INSTRUCTION-DSL-TESTS-SH'` → 3 行（CH-POISEUILLE-ANALYTIC-DSL / TAYLOR-GREEN-DECAY-DSL / PYLBM-SCHEME-OPERATOR-PARITY），均 status=passed——但经核这 3 行是 m18092 审计针对 **m17968 的 instruction DSL 交付物**补注册的，与 m18140 的浅水交付物**主题不符**，不能裁判本单。
5. **集成面 grep**：`shallow_water_lbm|run_swe_lbm|SWE_LBM_Config` 仅命中自身驱动/测试；`__init__.py` 未导出。

## ③ 计算结果（程序化注入，零手抄）

| 算例 | 分辨率 | FV L1 | FV L2 | LBM L1 | LBM L2 | LBM 稳定 |
|---|---|---|---|---|---|---|
| smooth_wave | 200 | 0.02% | 0.03% | **77.44%** | **83.23%** | ✅ |
| smooth_wave | 400 | 0.01% | 0.02% | NaN | NaN | ❌ |
| smooth_wave | 800 | 0.00% | 0.01% | NaN | NaN | ❌ |
| dam_break | 200 | 1.66% | 2.50% | NaN | NaN | ❌ |
| dam_break | 400 | 1.08% | 1.68% | NaN | NaN | ❌ |
| dam_break | 800 | 0.66% | 1.08% | NaN | NaN | ❌ |

（FV=本仓既有 `shallow_water.py` Rusanov，非 m18140 新增；LBM=m18140 新增。审计重跑与交付 result.json 逐位一致。）

## ④ 结果图

交付物自带对照图（FV vs LBM vs 解析解深度剖面），审计直接引用：
- smooth_wave_comparison.png（1268×586）：LBM 在 nx200 已严重偏离解析解
- dam_break_comparison.png（1268×586）：LBM 全分辨率发散，FV 贴合 Ritter 自相似解

## ⑤ 注意点（坑/局限/适用范围）

- **F1 范围偏离（阻塞）**：goal 是返修 m17968（instruction DSL），实际交付是全新浅水 LBM 求解器；锚点基准 3 行（instruction DSL）未被报告触及。
- **F2 LBM 未达 15%（阻塞）**：smooth_wave/nx200 L1=77.44%>15%；其余 5 组 LBM 全部不稳定（NaN）。mission 核心目标（验证 LBM<15%）未达成。
- **F3 交付物未入 commit（阻塞）**：git status 3 行 `??`；mission result head_commit=null/branch=null → 平台自动推送无从推起（与 m17968 同病）。
- **F4 无平台报告**：tensor_reports 无 mission_id=18140 行（仅工作树 REPORT.md）。
- **F5 报告自判 L4 过强**：REPORT.md 成熟度表称「L4 ✅ 有对照表和复现命令」——对照表+复现命令≠基准验证通过；LBM 臂未过精度判据，L4 不成立。
- 诚实面：报告如实标注 LBM 失败模式（干床奇异性/激波捕捉不足/边界反射），honest-FAIL 未粉饰，值得肯定。

## ⑥ 误差判定（逐项，显式回答 <3%？）

- FV 臂（既有模块）：worst 2.50%（dam_break/nx200 L2）→ **<3%：是**（且 <15%：是）。
- LBM 臂（m18140 新增）：smooth_wave/nx200 L1=77.44%、L2=83.23% → **<3%：否；<15%：否**；其余 5 组不稳定无有效数值。
- 锚点基准 3 行（instruction DSL）：0.686%/0.0809%/1.11e-16 均在容差内 → 通过，但**裁判对象是 m17968 交付物，非本单**。
- 结论：本单新增能力（LBM 浅水求解器）**未通过** owner 精度标准；判 **L2**（真代码 input→compute→output，7 测试绿，零 MOCK；L3 无调用方、L4 精度未过）。

## 判级与处置

- L1 单元验证=过；L2 独立功能实现=过；L3 系统集成=未达；L4 基准验证=未达（精度未过）；L5 不宣称。
- **disposition=reopened**：核心目标未达成 + 交付物未 commit + 范围偏离。返修方向：① 明确本单与 instruction DSL 锚点的关系（改锚点或改交付）；② LBM 方案升级（多速度/熵稳定碰撞）或改判教学级定位并降判据；③ 交付物 commit 入 main。
