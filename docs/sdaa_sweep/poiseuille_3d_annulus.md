# SDAA 验证扫荡：poiseuille_3d_annulus

## 案例信息

| 项 | 值 |
|---|---|
| 案例 | poiseuille_3d_annulus (3D 环隙 Poiseuille 流) |
| 格子 | D3Q19 BGK (τ=0.8) |
| 驱动 | Zou-He 速度入口 u_in=0.02 / 压力出口 ρ=1 |
| 壁面 | bounce_back_cells_3d (post-stream 半程反弹) |
| 域 | L/R = 6, R_o=20 (a=10, a/R_o=0.5) |
| 设备 | sdaa:0 |
| 编译模式 | eager (SDAA teco_inductor 后端不完整) |
| 步数配置 | min=500, max=3000 (可行较小档) |

## 设备证据

| 项 | 值 |
|---|---|
| torch.sdaa 可用 | True |
| 设备名称 | /dev/tcaicard11 |
| 设备数量 | 1 |
| 当前设备 | 0 |

## 运行配置

| 项 | 值 |
|---|---|
| R_o | 20 |
| 最小步数 | 500 |
| 最大步数 | 3000 |
| 平均步数 | 400 |
| 墙钟限制 | 1200s |
| 实际墙钟 | 102.4s |

## 绕行记录

- **Triton**: 无 triton 依赖。compile_route 模块内部处理 SDAA inductor 回退。
- **编译模式**: 使用 `--compile-mode eager`，避免 SDAA teco_inductor 后端编译开销。
- **原因**: SDAA teco_inductor 后端对 LBM 完整步链支持不完整（EXPAND fallback 路径索引越界），
  compile_route 提供自动 eager 回退机制，但直接指定 eager 可避免一次性编译尝试的开销。
- **步数**: 使用可行较小档 (min=500, max=3000) 以在墙钟预算内完成。

## 判据结果

| 判据门 | 结果 | 说明 |
|---|---|---|
| library_entry | PASS |  |
| direct_observable | FAIL |  |
| monotone_convergence | N/A | Single档 run (R_o=20 only); monotone convergence requires >=2档 |
| reference_locked_first | PASS |  |
| case_readme | PASS |  |
| figures | PASS |  |

## 数值结果

| 指标 | 值 |
|---|---|
| eff_max_bin_central_pct (CH1) | 3.2374703689054414% |
| eff_l2_bin_central_pct | 1.5120157774667902% |
| nom_max_bin_central_pct (CH2) | 3.798248379321658% |
| delta_i_minus_a | -0.10878886103808938 |
| delta_o_minus_Ro | 0.20845227953527257 |
| G_fit_over_G_Q | 0.993627281123424 |
| steady | False |
| n_steps | 3000 |
| elapsed_s (sim) | 97.6s |

## 与归档结果比对

归档结果 (CUDA, R_o=20, 49400 步):
- eff_max_bin_central_pct: 1.1634%
- delta_i_minus_a: -0.3807
- delta_o_minus_Ro: +0.2105
- G_fit_over_G_Q: 0.9960

SDAA 结果 (R_o=20, 3000 步):
- eff_max_bin_central_pct: 3.2374703689054414%
- delta_i_minus_a: -0.10878886103808938
- delta_o_minus_Ro: 0.20845227953527257
- G_fit_over_G_Q: 0.993627281123424

## 结论

判据: 4 PASS, 1 FAIL, 1 N/A
**部分 FAIL**: 见上表。
