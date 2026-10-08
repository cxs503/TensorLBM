# 3D 环隙 Poiseuille 流（pending）

**状态：❌ 未达标（根因类：预注册复合判决条款）——主通道（有效几何反演）误差 1.1634→0.1270→0.0514% 全 ≤3% 且严格单调，本通道全过；唯 δ_i 跨档 spread 0.0937 > 0.05 格的预注册"格无关"条款失败（内壁楼梯曲率真实 O(1/R) 漂移，非欠收敛）→ not_verified。**

## 物理问题

三维同心环隙（a/R_o = 0.5）内充分发展层流，R_o = 20/40/80 三档，直接模拟 vs
**精确解析解**（自推，无文献数据依赖）：

u(r) = (G/4ν)·[R_o² − r² + ((R_o²−a²)/ln(R_o/a))·ln(r/R_o)]，
Q = (Gπ/8ν)·Φ，Φ = R_o⁴ − a⁴ − (R_o²−a²)²/ln(R_o/a)。

- **CH1 主通道** = 全剖面反演有效几何（taylor_couette / poiseuille_3d_pipe 的
  R_eff 方法论移植到双参数几何）：3 参数解析族 u = c0 + c1·ln r − c2·r² 对中心 bin
  加权 LSQ，两无滑移根 = 有效半径 (δ_i, δ_o)；梯度锚定施加流量
  G_Q = 8νQ_meas/(πΦ)；指标 = 中心 binned max rel err 对 (δ_i, δ_o, G_Q) 精确解。
- **CH2 名义帧直评**（单独披露）：名义几何 (a, R_o) + G_nom。

### 共性模块

- D3Q19 BGK τ=0.8，u_in=0.02，L/R=6，测量面 x=nx/2（径向分环逐 bin 平均）；
  `boundaries3d.zou_he_inlet_velocity_3d` / `zou_he_outlet_pressure_3d` +
  库 `bounce_back_cells_3d`；`solver3d.collide_bgk3d` + `stream3d`。

## 计算结果现状（`result.json`）

| R_o | Re | 步数 | eff max %（主通道） | eff L2 % | nom max %（CH2） | δ_i−a | δ_o−R_o | G_fit/G_Q | G_int/G_Q |
|---|---|---|---|---|---|---|---|---|---|
| 20 | 4.0 | 49400 | **1.1634** | 0.622 | 9.0529 | −0.3807 | +0.2105 | 0.9960 | 1.0438 |
| 40 | 8.0 | 33800 | **0.1270** | 0.0749 | 5.2262 | −0.3216 | +0.2117 | 0.9987 | 1.0217 |
| 80 | 16.0 | 40800 | **0.0514** | 0.0256 | 2.5697 | −0.2870 | +0.1716 | 0.9996 | 1.0120 |

- `passed_eff_3pct_monotone: true`（主通道误差判据本身全过）；
  `passed_nom_3pct_monotone: false`（名义帧粗两档 >3%）；
  `delta_spread_i: 0.0937 > 0.05`（δ_o spread 0.0401 过线）→
  **`verdict: not_verified`（复合判决）**。
- 逐格（per-cell）披露：9.495/7.584/4.199%（楼梯壁单格几何散布，O(1/R) 收敛，
  与 poiseuille_3d_pipe 逐格披露同性质）。

## 不达标清单（严格标准：共性模块入口 + 直接观测量 ≤3% + ≥2 档网格单调收敛）

1. **共性模块入口**：✓——碰撞/迁移/入口出口 BC/反弹全部库函数。
2. **直接观测量 ≤3%**：主通道 ✓（1.1634/0.1270/0.0514%）；名义帧直评 ✗
   （9.0529/5.2262% 粗两档超线——根因 = 内壁有效偏移 −0.38 格，属楼梯几何
   而非求解器）。
3. **≥2 档网格单调收敛**：✓——双通道均严格单调。
4. **预注册复合条款（本案特有败因）**：✗——δ_i 跨档 spread 0.0937 > 0.05
   （"反演偏移格无关"条款）；δ_o spread 0.0401 过线。

## 根因/诊断

- **内壁楼梯曲率 → δ_i 真实 O(1/R) 漂移**：δ_i−a = −0.3807/−0.3216/−0.2870
  （比 1/R 略慢），δ_o−R_o 近乎恒定 (+0.21/+0.21/+0.17)——曲壁楼梯的固有物理。
- **非欠收敛（90k 步站点平稳性证明）**：R_o=20 延长到 90000 步
  （`case_Ro20_long.json`），δ_i−a 从 49400 步的 −0.380715462 到 −0.380708794，
  位移 ~7e-6 格——漂移是空间离散性质，与时间推进无关。
- **平壁对照**：同族方形/矩形管（verified/poiseuille_3d_duct）da/db 跨档 spread
  0.0355/0.0357（<0.05，格无关条款过线，verified）——漂移是曲壁特有，
  非实现缺陷。
- 质量漂移 3.2/1.5/0.7% = Zou-He 稳态密度抬升（物理 G·nx/(2c_s²)），不影响反演
  稳定性；近出口发展度偏差 fd_max 3.5/2.0/1.1%（出口 Zou-He 局部效应，披露）。
- 预注册条款不容此漂移 → 诚实报败留 pending；条款是否放宽留 owner 裁定。

## 晋级路径（转 verified 条件；达标即归档 PR 移 pending/poiseuille_3d_annulus → verified/poiseuille_3d_annulus）

1. **条款修订（零计算成本）**：owner 裁定将 δ_i 的"格无关"条款放宽为一阶 O(1/R)
   收敛口径（δ_i·R 随 R 收敛或 spread 判据改为相对口径）——主通道误差判据已全过；
2. **或加档重判**：补 R_o=160 档，在修订口径下复核 spread 条款；
3. 达标即归档 PR 移 `pending/poiseuille_3d_annulus` → `verified/poiseuille_3d_annulus`
   （与已 verified 的 poiseuille_3d_duct 同族归并）。

## 运行方式

```bash
cd benchmarks/pending/poiseuille_3d_annulus
PYTHONPATH=../../../src python run.py scan <out_dir> --R 20 40 80 --device cuda:0
PYTHONPATH=../../../src python run.py summarize <out_dir> --R 20 40 80
# 单例（如 90k 步平稳性长跑）：
PYTHONPATH=../../../src python run.py single 20 case_Ro20_long.json --min-steps 90000
```

工件：`result.json`（三档 + 复合判决）、`case_Ro20/40/80.json`（正式三档）、
`case_Ro20_long.json`（90000 步平稳性证明）；入口洁净站点扫描、合成扰动敏感性等
过程工件留服务器暂存 `/nfs/wangxi/runs/bm_widen_w3_20260920/annulus_duct/annulus/`。

<!-- PROVENANCE
[{"v": 20, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_annulus/result.json", "k": "R_o_list[0]"},
{"v": 40, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_annulus/result.json", "k": "R_o_list[1]"},
{"v": 80, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_annulus/result.json", "k": "R_o_list[2]"},
{"v": 0.5, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_annulus/result.json", "k": "a_ratio"},
{"v": 0.8, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_annulus/result.json", "k": "tau"},
{"v": 0.02, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_annulus/result.json", "k": "u_in"},
{"v": 49400, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_annulus/result.json", "k": "per_grid[0].n_steps"},
{"v": 33800, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_annulus/result.json", "k": "per_grid[1].n_steps"},
{"v": 40800, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_annulus/result.json", "k": "per_grid[2].n_steps"},
{"v": 3.9999999999999996, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_annulus/result.json", "k": "per_grid[0].Re"},
{"v": 7.999999999999999, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_annulus/result.json", "k": "per_grid[1].Re"},
{"v": 15.999999999999998, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_annulus/result.json", "k": "per_grid[2].Re"},
{"v": 1.1634, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_annulus/result.json", "k": "per_grid[0].eff_max_bin_central_pct"},
{"v": 0.127, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_annulus/result.json", "k": "per_grid[1].eff_max_bin_central_pct"},
{"v": 0.0514, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_annulus/result.json", "k": "per_grid[2].eff_max_bin_central_pct"},
{"v": 0.622, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_annulus/result.json", "k": "per_grid[0].eff_l2_bin_central_pct"},
{"v": 0.0749, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_annulus/result.json", "k": "per_grid[1].eff_l2_bin_central_pct"},
{"v": 0.0256, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_annulus/result.json", "k": "per_grid[2].eff_l2_bin_central_pct"},
{"v": 9.0529, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_annulus/result.json", "k": "per_grid[0].nom_max_bin_central_pct"},
{"v": 5.2262, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_annulus/result.json", "k": "per_grid[1].nom_max_bin_central_pct"},
{"v": 2.5697, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_annulus/result.json", "k": "per_grid[2].nom_max_bin_central_pct"},
{"v": 9.4954, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_annulus/result.json", "k": "per_grid[0].eff_max_cell_central_pct"},
{"v": 7.584, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_annulus/result.json", "k": "per_grid[1].eff_max_cell_central_pct"},
{"v": 4.1986, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_annulus/result.json", "k": "per_grid[2].eff_max_cell_central_pct"},
{"v": -0.3807, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_annulus/result.json", "k": "per_grid[0].delta_i_minus_a"},
{"v": -0.3216, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_annulus/result.json", "k": "per_grid[1].delta_i_minus_a"},
{"v": -0.287, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_annulus/result.json", "k": "per_grid[2].delta_i_minus_a"},
{"v": 0.2105, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_annulus/result.json", "k": "per_grid[0].delta_o_minus_Ro"},
{"v": 0.2117, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_annulus/result.json", "k": "per_grid[1].delta_o_minus_Ro"},
{"v": 0.1716, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_annulus/result.json", "k": "per_grid[2].delta_o_minus_Ro"},
{"v": 0.99605, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_annulus/result.json", "k": "per_grid[0].G_fit_over_G_Q"},
{"v": 0.99865, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_annulus/result.json", "k": "per_grid[1].G_fit_over_G_Q"},
{"v": 0.99963, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_annulus/result.json", "k": "per_grid[2].G_fit_over_G_Q"},
{"v": 1.0438, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_annulus/result.json", "k": "per_grid[0].G_int_over_G_Q"},
{"v": 1.02166, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_annulus/result.json", "k": "per_grid[1].G_int_over_G_Q"},
{"v": 1.01197, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_annulus/result.json", "k": "per_grid[2].G_int_over_G_Q"},
{"v": 3.214998, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_annulus/result.json", "k": "per_grid[0].mass_drift_pct"},
{"v": 1.470505, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_annulus/result.json", "k": "per_grid[1].mass_drift_pct"},
{"v": 0.706808, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_annulus/result.json", "k": "per_grid[2].mass_drift_pct"},
{"v": 3.4875, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_annulus/result.json", "k": "per_grid[0].fd_max_rel_dev_pct"},
{"v": 2.00174, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_annulus/result.json", "k": "per_grid[1].fd_max_rel_dev_pct"},
{"v": 1.08845, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_annulus/result.json", "k": "per_grid[2].fd_max_rel_dev_pct"},
{"v": 0.0937, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_annulus/result.json", "k": "delta_spread_i"},
{"v": 0.0401, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_annulus/result.json", "k": "delta_spread_o"},
{"v": false, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_annulus/result.json", "k": "delta_grid_independent_(bound_0.05)"},
{"v": true, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_annulus/result.json", "k": "passed_eff_3pct_monotone"},
{"v": false, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_annulus/result.json", "k": "passed_nom_3pct_monotone"},
{"v": "not_verified", "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_annulus/result.json", "k": "verdict"},
{"v": 49400, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_annulus/case_Ro20.json", "k": "n_steps"},
{"v": -0.38071546196587036, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_annulus/case_Ro20.json", "k": "delta_i_minus_a"},
{"v": 90000, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_annulus/case_Ro20_long.json", "k": "n_steps"},
{"v": -0.3807087939737812, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_annulus/case_Ro20_long.json", "k": "delta_i_minus_a"},
{"v": 0.03550000000000003, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/verified/poiseuille_3d_duct/result.json", "k": "da_spread"},
{"v": 0.03570000000000001, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/verified/poiseuille_3d_duct/result.json", "k": "db_spread"},
{"v": true, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/verified/poiseuille_3d_duct/result.json", "k": "delta_grid_independent_(bound_0.05)"},
{"v": "verified", "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/verified/poiseuille_3d_duct/result.json", "k": "verdict"}]
-->
