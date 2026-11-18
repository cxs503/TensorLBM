# phasefield_water_entry（pending）

**状态：待验证。4° 楔体在第 141 步首次产生非有限载荷，未达到预注册测窗。**

## 4° 恒速楔体案例说明

对称刚性楔体向静止液池匀速下行，D3Q19 保守 Allen–Cahn 双分布负责流体与相场，移动反弹边界和动量交换法输出整楔竖向力。展向 4 层周期，网格 z×y×x = 4×62×230，半长 100 格，速度 0.01 格/步，初始气隙 1 格。先静置 400 步，再申请推进 1,200 步。水和气的惯性密度比为 100。液滴入池采用另一套二维求解器，见[液滴说明书](../../../docs/benchmarks/droplet_impact.md)。

```bash
PYTHONPATH=src python benchmarks/pending/phasefield_water_entry/run_s2_wedge.py \
  --arm zeta4_dw100 --threads 1 --out-dir outputs/s2
```

| 观测 | 本次实算 | 判定 |
|---|---|---|
| 有限性 | 名义接触在第 100 步；第 140 步竖向力约 1.60e35 格子单位，第 141 步非有限 | 失败，程序写出 JSON 后返回 1 |
| 速度 | 已获得的有限采样中，最大速度至少为入水速度的 17.16 倍 | 超过上限 3 |
| 斜率、R² | 预注册 h/Dw ∈ [0.02, 0.10] 对应第 300–1,100 步，本次无有效拟合 | 两项均为 `null` |
| 跨网格与后续物体 | 本次只复算 4°、Dw=100；S3 圆柱和 S4 球体的正式晋级依赖 S2 | 待验证 |

[逐步力记录与参数](evidence/s2_zeta4_dw100.json) · [机器判定](result.json)

### 如何与 Wagner 比对

De Rosis 与 Tafuni 式 (45) 对**半楔面压力合力**给出每单位展长的无量纲斜率 π³/(8 tan²ζ)。4° 时为 792.6330；本算例 ρ=1、V=0.01，格子单位斜率为 0.0007926330。论文式 (46)–(47) 以 0.002 s 和 0.004 s 两点求割线；在论文 Dw=1 m、V=1 m/s 下对应 h/Dw=0.002 和 0.004。仓库的整楔 MEM 合力、局部拟合窗和载荷分量与原文存在差异。正式对标须直接测量同一半楔面压力合力、统一展长与方向并在相同无量纲时刻取样；单臂完成内部检查时标记为 `DIAGNOSTIC_PASS`，仍须另行完成上述文献对标。本次没有可信的载荷误差百分比。[原文 §3.4 与图 11](https://doi.org/10.1111/mice.12651)

早期 8° 探针跑满 1,200 步，旧拟合口径报告斜率偏差 +273.7%、R²=0.613、最大速度比 9.78。8° 没有预注册的斜率阈值。该记录与本次 4° 结果使用不同角度和测窗。[历史记录](NOTES.md)

| 阻塞项 | 一句话 | 关闭判据 |
|---|---|---|
| **B1** 无运动固体 + MEM 力 | ✅ 部分关闭：运动固体 + MEM 已实现，F(t) 已能输出；静态 Archimedes 闸门 1.0251。⛔ u ≠ 0 的力读数仍无独立闸门 | ① F(t) 输出 ✅；② 动量守恒闸门（`F_body` vs `dP/dt − F_g − F_walls`）≤ 1 % ⛔ |
| **B2** 墙—相场质量源 | 已归因（p1 十臂）：重力不是驱动源，侧壁主导界面爬升，φ 增长是饱和型（水箱加倍 → 小 856×），Neumann ghost 更差 13× | S0 门通过（φ 漂移 ≤ 1e-4/1e3 步）——目前只有加倍水箱臂接近 |
| **B6** Wagner 楔形锚不一致 | ✅ **已关闭**：三套约定显式化并锚定 π²/4，`dF/dt/(ρV³) = π³/(8tan²ζ)`，压力积分恒等式 rel 2e-6，`tests/test_water_entry_slamming.py` 14 passed | 已满足 |
| B4 | CAC 为 BGK 家族，文献显示 BGK 在入水 t≈0.2 s 崩 | S2 ζ=2° 跑满测窗且 max\|u\|/V ≤ 3 |
| B5 | 匹配 We 需 σ ≈ 1e-5，超出 verified 的 σ 线性窗口 | 补 σ=2e-5/1e-5 的 Laplace 臂 ≤ 1.5% |
| B7 | 无接触角/湿润边界 | S2–S4 不阻塞（默认中性）；S5 前置 |

## 定位

本目录管理移动固体穿越气液自由面的验收。各阶段的参考、适用范围和状态分别登记。

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

## 晋级前缺少的证据

1. 复核 4° 工况第 100–141 步的动量平衡、压力和界面，定位不稳定来源。
2. 同时输出稳定的整楔 MEM 力与半楔压力积分，验证单位、符号、展长归一化及文献测窗。
3. 完成预注册角度与两档网格，记录相场有界性、力斜率、线性度及对独立参考的误差。

## 治理

验收口径沿用 [`benchmarks/STATUS.md`](../../STATUS.md)：每个检查点达标 **且** 两条网格跨度达标；本 benchmark 因 Wagner 边缘奇点把"网格收敛"放宽为"斜率跨度 ≤20%"（prereg §3），该放宽必须在 REPORT 里显式披露。任何"看起来像入水"的图片都不构成通过证据。
