# phasefield_water_entry（pending）

**状态：⏸️ PENDING / BLOCKED（B6 已关；S0–S1 探针已实测；S2 一臂已跑通但未进闸门）**

> 2026-10-31 进展：`wagner_theory` 楔形锚已按 π²/4 定版并加了回归测试（**B6 关闭**）；
> `cac_moving_body.py` 提供运动固体 + Ladd MEM，静态 Archimedes 闸门通过
> （两相 1.0251 / 均相 1.0027）；S2 ζ=8° 一臂跑满 1200 步（有限、φ≤1.014），
> 但斜率 +273.7 %、R² 0.613、max\|u\|/V 9.78，**三项门全不过**。
> 根因已定位到 u ≠ 0 时的力读数约定（见 `NOTES.md` 的 S2 节与下一步闸门）。
> 详细机器数字与四个已修 bug：`NOTES.md`「第二轮探针」。

| 阻塞项 | 一句话 | 关闭判据 |
|---|---|---|
| **B1** 无运动固体 + MEM 力 | ✅ 部分关闭：运动固体 + MEM 已实现，F(t) 已能输出；静态 Archimedes 闸门 1.0251。⛔ u ≠ 0 的力读数仍无独立闸门 | ① F(t) 输出 ✅；② 动量守恒闸门（`F_body` vs `dP/dt − F_g − F_walls`）≤ 1 % ⛔ |
| **B2** 墙—相场质量源 | 已归因（p1 十臂）：重力不是驱动源，侧壁主导界面爬升，φ 增长是饱和型（水箱加倍 → 小 856×），Neumann ghost 更差 13× | S0 门通过（φ 漂移 ≤ 1e-4/1e3 步）——目前只有加倍水箱臂接近 |
| **B6** Wagner 楔形锚不一致 | ✅ **已关闭**：三套约定显式化并锚定 π²/4，`dF/dt/(ρV³) = π³/(8tan²ζ)`，压力积分恒等式 rel 2e-6，`tests/test_water_entry_slamming.py` 14 passed | 已满足 |
| B4 | CAC 为 BGK 家族，文献显示 BGK 在入水 t≈0.2 s 崩 | S2 ζ=2° 跑满测窗且 max\|u\|/V ≤ 3 |
| B5 | 匹配 We 需 σ ≈ 1e-5，超出 verified 的 σ 线性窗口 | 补 σ=2e-5/1e-5 的 Laplace 臂 ≤ 1.5% |
| B7 | 无接触角/湿润边界 | S2–S4 不阻塞（默认中性）；S5 前置 |

## 定位

入水（water entry）是相场两相 LBM 的**标准应用**而**非标准基准**：GitHub 上没有任何公开的"相场 LBM 入水"算例（见调研报告），因此本目录的作用是**自建算例定义 + 用解析/实验数字定锚**，并严格区分"能跑通"与"物理正确"。

- 调研与对照：[`docs/benchmarks/phasefield_water_entry_survey_20261030.md`](../../../docs/benchmarks/phasefield_water_entry_survey_20261030.md)
- 锁定规格（算例、无量纲数、门限、预算）：[`prereg.md`](prereg.md)
- 探针机器数字：[`NOTES.md`](NOTES.md)
- 参考锚（机器可读）：[`refs/anchors.json`](refs/anchors.json)
- 探针脚本：[`probe/p0_hydrostatic.py`](probe/p0_hydrostatic.py)、[`probe/p0b_static_body.py`](probe/p0b_static_body.py)

## 为什么必须是 CAC，而不是仓库里其它的"相场"

| 候选 | 结论 | 证据 |
|---|---|---|
| `cac_lbm`（conservative Allen–Cahn，D3Q19 双分布，ρ(φ) 显式） | ✅ 唯一可用内核 | `verified/laplace_cac_phasefield`（σ 标定 R/密度比迁移 ≤0.445%） |
| `multiphase.free_energy_step`（Swift FE，2D） | ❌ 禁用 | ρ ≡ 1、仅 Boussinesq 浮力 → 无密度比；`pending/hadamard_rybczynski_droplet/prereg.md` 模块清单 |
| `allen_cahn_lbm`（3D AC 早期模块） | ❌ 禁用 | 少数相液滴 25→1000 步完全溶解（同案 E4） |
| 相场之外：CG 双 τ / SC-MCMP | 仅作对照 | `verified/poiseuille_two_phase_cg`；`pending/two_phase_poiseuille` FAIL 26.8%/33.2% |

## 立即可以做的三件事（不依赖 B1）

1. **S0 复跑 + 归因**：把 P0 的 φ 漂移按"壁面通量 / 界面厚度 / 时间步"三个开关拆开（`axes="pwp"` + 无重力对照），定位源项 → 关 B2。
2. **锚定版**：修 `wagner_theory` 楔形总力与 C_s，加 `tests/test_water_entry_slamming.py` 的锚回归（Eq.(45) 的 `dF/dt/(ρV³) = π²/(8tan²ζ)`）→ 关 B6。
3. **σ 低幅臂**：在 σ = 2e-5 / 1e-5 重复 Laplace 标定（W = 4），确认 Route-B 常数仍成立 → 关 B5。

## 治理

验收口径沿用 [`benchmarks/STATUS.md`](../../STATUS.md)：每个检查点达标 **且** 两条网格跨度达标；本 benchmark 因 Wagner 边缘奇点把"网格收敛"放宽为"斜率跨度 ≤20%"（prereg §3），该放宽必须在 REPORT 里显式披露。任何"看起来像入水"的图片都不构成通过证据。
