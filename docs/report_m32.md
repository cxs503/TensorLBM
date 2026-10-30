# SDAA 验证扫荡补报：poiseuille_3d_annulus (Mission 32)

> **Mission 32 补报单** | **honest-FAIL**（direct_observable: 3.24% > 3%）

## 1. 案例介绍

3D 环隙 Poiseuille 流——LBM 经典验证基准，同心环隙不可压层流，有精确解析解。

| 参数 | 值 |
|---|---|
| R_o / a | 20 / 10 lu (a/R_o=0.5) |
| 域 L/R_o | 6, 网格 43x43x120 |
| D3Q19 BGK | tau=0.8, nu=0.1 |
| u_in / rho_out | 0.02 / 1.0 (Zou-He) |
| Re / Ma | 4.0 / 0.0522 |
| 壁面 | bounce_back_cells_3d |

解析解：$$u(r) = \frac{G}{4\nu}\left[R_o^2 - r^2 + \frac{R_o^2 - a^2}{\ln(R_o/a)}\ln\frac{r}{R_o}\right]$$

评估通道：CH1=有效几何帧(LSQ反演 delta_i,delta_o,G_fit)；CH2=名义几何帧。

## 2. 方法

- 独立脚本 `scripts/sdaa_sweep_poiseuille_3d_annulus.py`（不改业务代码）
- 设备：SDAA `/dev/tcaicard11`，eager 模式（teco_inductor 后端不完整）
- 无 triton 依赖；compile_route 自动回退
- 步数 3000（墙钟 1200s 预算），归档 CUDA 49400 步
- 6 门判据：library_entry / direct_observable / monotone_convergence / reference_locked_first / case_readme / figures

## 3. 计算结果

| 指标 | SDAA (3000步) | CUDA (49400步) |
|---|---|---|
| eff_max_bin (CH1) | **3.24%** | 1.16% |
| eff_l2_bin | 1.51% | 0.62% |
| delta_i - a | -0.1088 | -0.3807 |
| delta_o - R_o | 0.2085 | 0.2105 |
| G_fit/G_Q | 0.9936 | 0.9960 |
| 质量漂移 | 2.90% | 3.21% |
| 稳态 | False | True |

逐 bin 采样（r, u_SDAA, u_ref_CH1, err%）：

| Bin | r | u_SDAA | u_ref | err% |
|---|---|---|---|---|
| 0 | 10.46 | 0.00795 | 0.00697 | 14.03 |
| 2 | 12.34 | 0.02312 | 0.02327 | 0.64 |
| 4 | 14.42 | 0.02992 | 0.02999 | 0.26 |
| 6 | 16.45 | 0.02688 | 0.02697 | 0.33 |
| 9 | 19.47 | 0.00758 | 0.00722 | 4.85 |

判据门：4 PASS / 1 FAIL (direct_observable) / 1 N/A (monotone_convergence 单档)

## 4. 结果图

图1=径向速度剖面+逐bin误差；图2=判据门通过情况。（见 figures 字段内嵌 png）

## 5. 注意点

- SDAA teco_inductor 后端 EXPAND fallback 索引越界 -> 必须 eager
- 步数不足(3000 vs 49400) -> steady=False -> direct_observable FAIL
- delta_i 差异 0.27 格源于未稳态，非求解器缺陷
- 单档运行无法评估单调收敛
- G_fit/G_Q=0.994 质量守恒良好；L2=1.51%<3% 整体吻合

## 6. 误差判定

| 关键量 | SDAA | 基准 | 误差 | <3%? |
|---|---|---|---|---|
| eff_max_bin | 3.24% | 1.16% | +2.07% | **NO** |
| eff_l2_bin | 1.51% | 0.62% | +0.89% | YES |
| G_fit/G_Q | 0.9936 | 0.9960 | D=0.0024 | YES |

**eff_max_bin_central_pct 是否<3%？ NO** (3.24% > 3%)。根因：步数不足，非求解器缺陷。

**honest-FAIL**：如实报告，不粉饰。

---
*Mission 32 补报 | 2026-10-29*
