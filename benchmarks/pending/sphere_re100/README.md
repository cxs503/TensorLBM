# 球 Re=100 阻力系数（sphere_re100）（pending）

**状态：❌ 未达标(根因类)——全部测力路径（压力+摩擦积分 / MEM 三变体 / surface-only Ladd MEM）在阶梯球面上 >3%；根因已定源为阶梯几何偏差（双曲率界面动量通量一致性偏高 ~+9%），需光滑边界表示（BFL）——该路线已在 `sphere_re100_mem/w8a/` 达标待晋级**

## 物理问题

均匀来流绕球（Re=U·D/ν=100），观测量 = 阻力系数 Cd。参考（`result.json`）：Schiller-Naumann Cd_ref = 1.091731（`result_mem_surface.json` reference 节另载 Clift-Grace-Weber 交叉式）。

三代测力路径（全部真实模拟、extrap='none'、无修正项）：

1. **GeneralSimEngine PARAMETRIC_SPHERE**（D3Q19、MRT auto、ForceMethod.BOTH）：D=40 域 140×120×120，16000 步收敛；
2. **压力积分参数扫描**（p0_method × friction_formula 4×4，`scan_p0_friction_summary.json`）；
3. **surface-only Ladd MEM**（cylinder_3d 采纳口径移植，BGK+冻结 solid+half-way BB+far_field_bc_3d 链，`result_mem_surface.json`）。

## 计算结果现状

**路径 1（D=40，16000 步，末 100 样本时间平均）**：

| 力法 | Cd | 误差 vs SN 1.091731 |
|------|----|--------------------|
| 压力+摩擦积分 cd_total（extrap=none） | 0.9109 | −16.56%（cd_p 0.5566/−49.01%，cd_f 0.3543/−67.55%） |
| MEM standard（Ladd 和形式） | 3.9804 | +264.6% |
| MEM galilean | 7.5344 | +590.13% |
| MEM bg_sub（G15 修复） | 3.9804 | +264.6%（与 standard 位相同——修复无效） |

**路径 2（p0×friction 扫描，时间平均）**：p0 维度完全退化（闭合面 Σn̂dA=0，四种 p0_method 逐位相同）；friction 维度：

| friction | D40 cd_total | D40 err | D60 cd_total（central） | D60 err |
|----------|-------------|---------|------------------------|---------|
| standard | 0.9112 | −16.53% | — | — |
| central | 1.1142 | +2.05% | 1.1283 | **+3.35%（加密即翻车）** |
| faces（edd3e80 后） | 0.97 | −11.15% | 0.9894 | −9.37% |

central 的 D40"达标"是巧合抵消（central 采样离壁第二格速度且无距离归一、系统性高估壁剪，对冲压力低估，加密后失衡反超 3%）；faces 方向正确且收敛（−11.15%→−9.37%）但两档均 >3%。

**路径 3（surface-only Ladd MEM，lateral 16；D8–D14 行为短程均值、D12/D18 为判定对）**：

| D | surface 格数 | Cd_mem_surface | err |
|---|-------------|---------------|-----|
| 8 | 134 | 1.2501 | +14.5% |
| 10 | 222 | 1.2518 | +14.66% |
| 12 | 354 | **1.1916** | **+9.15%** |
| 14 | 446 | 1.217 | +11.47% |
| 18 | 794 | **1.1891** | **+8.92%** |

同场三口径（D=12，域 204×192×192、7000 步）：cd_mem_surface 1.1916（+9.15%）/ cd_mem_all 1.2301（+12.68%）/ interior 伪项 0.0385。D12 与 D18 跨度 0.23%（网格收敛成立）但都收敛到 ~+9%——**收敛到了错误的值**。

## 不达标清单

（战役严格标准：共性模块入口 + 直接观测量对参考 ≤3% + ≥2 档网格单调收敛；修正/还原变换不算直接模拟）

1. **路径 1 ✗**：cd_total −16.56%、MEM 三变体 +264.6~+590.13%，D=40 收敛后仍远超 3%。
2. **路径 2 ✗**：central D60 +3.35% 超线且误差随加密增大（违反单调收敛条款的语义）；faces 两档 −11.15%/−9.37% 均 >3%。
3. **路径 3 ✗**：两档（D12/D18）span 0.23% ≤3% 但 both_within_3pct=false（+9.15%/+8.92% >3%）——网格收敛到错误值。
4. 域/阻塞杠杆已否证：D10 lateral 16（堵塞 6.25%）vs lateral 32（3.125%）step1000 Cd 1.3177 vs 1.3178（四位小数逐位相同）。

## 根因/诊断

已定源（`result_mem_surface.json` why_it_fails + 交叉仪器）：

- **阶梯球几何偏差**：体积掩膜阶梯球高估界面动量通量；surface-only MEM 与独立控制体（CV）动量平衡仪表同读同量级正偏（两独立仪器一致，`result_mem_surface.json` cross_check 节）→ 非 MEM 口径 bug，是几何/分辨率极限。
- **MEM 历史爆炸的机制**（路径 1）：Ladd 和式在近壁链子集混合不同时间层，termA 携带不抵消的自由流背景——该失效模式已由库 PR #309（wet-node 估计器 `momentum_exchange_wet_node`）修复，但修复后的直接力路线判档记录在 `sphere_re100_mem/`（本目录路径 1/2 为修复前历史记录）。
- **压力积分 G12**：extrap='none' 取近壁格心压力漏驻点压力升（cd_p −49.01%），p0 无杠杆（数学 no-op）。
- 历史旁证：光滑边界（octree）历史单档能进参考带但速度不可行（`result_mem_surface.json` 字符串记载）。

## 晋级路径（转 verified 的条件）

1. **换光滑边界表示（BFL 插值边界）+ 大域**：即 `sphere_re100_mem/w8a/` 已执行的路线——T1 D40 +2.771%、T2 D60 +2.540% 双过 3% 门且单调（判决 PASS，晋级候选待 owner）。本案例与该案例同问题（球 Re=100 Cd），达标后可一并处理。
2. 预算参考：w8a T2（D60 大域 420×420×480、24000 步）机器记录显存峰值 26.672595977783203 GiB（5090 32G 可跑），600 步机器探针 wall_s=192.86548471450806 s → 全档约 4–5 GPU·h（估算，由每步成本外推）。
3. 达标后由归档 PR 将 `pending/sphere_re100`（连同 `sphere_re100_mem`）→ `verified/`，历史力法扫描以披露条款随迁。

<!-- PROVENANCE
[{"v": 1.091731, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100/result.json", "k": "Cd_ref_SN"},
 {"v": 0.9109, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100/result.json", "k": "time_avg_last100.cd_total"},
 {"v": -16.56, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100/result.json", "k": "err_pct_vs_SN.cd_total"},
 {"v": 0.5566, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100/result.json", "k": "time_avg_last100.cd_pressure"},
 {"v": -49.01, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100/result.json", "k": "err_pct_vs_SN.cd_pressure"},
 {"v": 0.3543, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100/result.json", "k": "time_avg_last100.cd_friction"},
 {"v": -67.55, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100/result.json", "k": "err_pct_vs_SN.cd_friction"},
 {"v": 3.9804, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100/result.json", "k": "time_avg_last100.cd_mem_standard"},
 {"v": 264.6, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100/result.json", "k": "err_pct_vs_SN.cd_mem_standard"},
 {"v": 7.5344, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100/result.json", "k": "time_avg_last100.cd_mem_galilean"},
 {"v": 590.13, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100/result.json", "k": "err_pct_vs_SN.cd_mem_galilean"},
 {"v": 3.9804, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100/result.json", "k": "time_avg_last100.cd_mem_bgsub"},
 {"v": 0.9112, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100/scan_p0_friction_summary.json", "k": "D40_time_avg.standard.cd_total"},
 {"v": -16.53, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100/scan_p0_friction_summary.json", "k": "D40_time_avg.standard.err_pct"},
 {"v": 1.1142, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100/scan_p0_friction_summary.json", "k": "D40_time_avg.central.cd_total"},
 {"v": 2.05, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100/scan_p0_friction_summary.json", "k": "D40_time_avg.central.err_pct"},
 {"v": 1.1283, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100/scan_p0_friction_summary.json", "k": "D60_central.cd_total"},
 {"v": 3.35, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100/scan_p0_friction_summary.json", "k": "D60_central.err_pct"},
 {"v": 0.97, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100/scan_p0_friction_summary.json", "k": "faces_retest_2026-08-20.D40_faces.cd_total"},
 {"v": -11.15, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100/scan_p0_friction_summary.json", "k": "faces_retest_2026-08-20.D40_faces.err_pct"},
 {"v": 0.9894, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100/scan_p0_friction_summary.json", "k": "faces_retest_2026-08-20.D60_faces.cd_total"},
 {"v": -9.37, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100/scan_p0_friction_summary.json", "k": "faces_retest_2026-08-20.D60_faces.err_pct"},
 {"v": 1.2501, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100/result_mem_surface.json", "k": "mesh_trend_lateral16[0].cd_mem_surface"},
 {"v": 14.5, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100/result_mem_surface.json", "k": "mesh_trend_lateral16[0].err_pct"},
 {"v": 1.2518, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100/result_mem_surface.json", "k": "mesh_trend_lateral16[1].cd_mem_surface"},
 {"v": 14.66, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100/result_mem_surface.json", "k": "mesh_trend_lateral16[1].err_pct"},
 {"v": 1.1916, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100/result_mem_surface.json", "k": "mesh_trend_lateral16[2].cd_mem_surface"},
 {"v": 9.15, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100/result_mem_surface.json", "k": "mesh_trend_lateral16[2].err_pct"},
 {"v": 1.217, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100/result_mem_surface.json", "k": "mesh_trend_lateral16[3].cd_mem_surface_step3000"},
 {"v": 11.47, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100/result_mem_surface.json", "k": "mesh_trend_lateral16[3].err_pct_step3000"},
 {"v": 1.1891, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100/result_mem_surface.json", "k": "mesh_trend_lateral16[5].cd_mem_surface_step4000"},
 {"v": 8.92, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100/result_mem_surface.json", "k": "mesh_trend_lateral16[5].err_pct_step4000"},
 {"v": 1.2301, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100/result_mem_surface.json", "k": "converged_grid.cd_mem_all"},
 {"v": 12.68, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100/result_mem_surface.json", "k": "converged_grid.err_mem_all_pct"},
 {"v": 0.0385, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100/result_mem_surface.json", "k": "converged_grid.cd_mem_interior"},
 {"v": 0.23, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100/result_mem_surface.json", "k": "two_grids.grid_span_pct"},
 {"v": 1.3177, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100/result_mem_surface.json", "k": "domain_blockage_probe.D10_lateral16_step1000_cd_mem_surface"},
 {"v": 1.3178, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100/result_mem_surface.json", "k": "domain_blockage_probe.D10_lateral32_step1000_cd_mem_surface"},
 {"v": 26.672595977783203, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100_mem/w8a/mem_sparse_D60big.json", "k": "meta.mem_peak_loop_gib"},
 {"v": 134, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100/result_mem_surface.json", "k": "mesh_trend_lateral16[0].n_surface"},
 {"v": 222, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100/result_mem_surface.json", "k": "mesh_trend_lateral16[1].n_surface"},
 {"v": 354, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100/result_mem_surface.json", "k": "mesh_trend_lateral16[2].n_surface"},
 {"v": 446, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100/result_mem_surface.json", "k": "mesh_trend_lateral16[3].n_surface"},
 {"v": 794, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100/result_mem_surface.json", "k": "mesh_trend_lateral16[5].n_surface"},
 {"v": 204, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100/result_mem_surface.json", "k": "converged_grid.nx"},
 {"v": 192, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100/result_mem_surface.json", "k": "converged_grid.ny"},
 {"v": 7000, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100/result_mem_surface.json", "k": "converged_grid.n_steps"},
 {"v": 2.7707656824703957, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100_mem/w8a/result.json", "k": "runs[0].err_pct"},
 {"v": 2.5400861325477218, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100_mem/w8a/result.json", "k": "runs[1].err_pct"},
 {"v": 192.86548471450806, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100_mem/w8a/mem_sparse_D60big.json", "k": "meta.wall_s"},
 {"v": 420, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100_mem/w8a/result.json", "k": "runs[1].domain_lu[0]"},
 {"v": 480, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100_mem/w8a/result.json", "k": "runs[1].domain_lu[2]"}]
-->
