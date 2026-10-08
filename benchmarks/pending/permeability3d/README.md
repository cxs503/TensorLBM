# 3D 周期球阵列 Stokes 渗透率（pending）

**状态：🟡 达标待晋级（φ=0.343 档）——总判 FAIL 由 φ=0.125/0.216 两个稀释档拖累（fp32 注入地板 + 负 O(Re) 惯性修正，均非 solver 共性路径缺陷）；φ=0.343 在冻结三档规则与修正四档规则下双 PASS（严格单调 +4.63→+3.10→+2.05%，四档再 +1.45%，最细 ≤3%），verified 候选待 owner 晋级。**

## 物理问题

简单立方周期球阵列的 Stokes 渗透率——verified/permeability（2D 周期圆柱方阵）的三维
对应。参考 = **Zick & Homsy 1982**（JFM 115:13–26）无量纲曳力表
K(φ) = F/(6πμUa)，U = 表观（Darcy）速度，φ = 真实固相分数：

- 双源交叉锁定：Basilisk `src/test/spheres.c` `zick` 表逐字 + Holmes et al. 2011
  （DOI 10.1002/nag.898）文献列，协议点互差 ≤0.015%（0.5236 发散点不取）；
  独立旁证 Hasimoto 1959 稀疏展开在稀释端吻合（NOTES 1.6）。
- φ 语义 = 真实固相分数；φ 档取表值格点 {0.125, 0.216, 0.343}（参考不插值；
  名义指令集 {0.10, 0.20, 0.30} 偏离已披露）。

### 共性模块

- D3Q19 `collide_bgk3d` → `stream3d` →（流体掩码）体力 → `bounce_back_cells_3d`，
  全周期单胞；τ=1.0（ν=1/6 精确）；run.py 零手写核。
- 3D 单相体力库内无助手：driver 层对 `_apply_body_force_2d` 的 D3Q19 逐字转写
  （一阶 Guo/Luo 形，每步恰注 ρ·a，与 2D 先例同款披露）。
- 测量映射：直接观测量 = 间隙平均速度 ⟨u_x⟩_fluid，表观速度 U=(1−φ)⟨u_x⟩；
  k_sim = ν(1−φ)⟨u_x⟩/a_body，k_ref = N³/(6π·a_nom·K_table)；
  **err = |k_ref/k_sim − 1|**（JSON 存无符号幅值；下表符号按 K_sim 与 K_ref 的
  大小关系恢复——正 = K_sim 偏低 = 渗透率偏低 = 曳力偏高；判据即该幅值 ≤3%）。

## 计算结果现状（`result.json`，fs=10 正式通道，N=64/96/128/160）

| φ | N | K_sim | K_ref 表 | err（符号恢复） |
|---|---|---|---|---|
| 0.125 | 64 | 4.164878 | 4.292 | +3.05% |
| 0.125 | 96 | 4.276434 | 4.292 | +0.36% |
| 0.125 | 128 | 4.230033 | 4.292 | +1.47% |
| 0.125 | 160 | 4.344759 | 4.292 | **−1.21%** |
| 0.216 | 64 | 7.194948 | 7.4423 | +3.44% |
| 0.216 | 96 | 7.311932 | 7.4423 | +1.78% |
| 0.216 | 128 | 7.246953 | 7.4423 | +2.70% |
| 0.216 | 160 | 7.114119 | 7.4423 | **+4.61%** |
| 0.343 | 64 | 14.721106 | 15.402 | +4.63% |
| 0.343 | 96 | 14.939179 | 15.402 | +3.10% |
| 0.343 | 128 | 15.092724 | 15.402 | +2.05% |
| 0.343 | 160 | 15.182334 | 15.402 | **+1.45%** |

- 判定规则：冻结三档 = N=64/96/128 严格单调 + err(128)≤3%；修正四档（Amendment 1）=
  最细档 N=160 ≤3% + 包络收敛（对楼梯 O(1/N) 振荡免疫）。
- 逐 φ 判决：0.125 冻结 FAIL / 修正 PASS；0.216 双 FAIL（err(160)=4.61%>3%）；
  **0.343 双 PASS**；总判（两规则同判）FAIL。
- 全案 steady_detected=true、Ma_max ≤0.0136、质量漂移 ≤6.1e-4 相对。

## 不达标清单（严格标准：共性模块入口 + 直接观测量 ≤3% + ≥2 档网格单调收敛）

1. **共性模块入口**：✓——物理核全走库共性路径（D3Q19 collide/stream/反弹）；
   唯一 driver 层转写为 3D 体力（库无 3D 单相助手，披露在案）。
2. **直接观测量 ≤3%**：部分 ✗——φ=0.343 四档全过（最细 1.45%）；φ=0.125 N64/128
   超线（3.05%/1.47%→修正档最细 1.21% 过）；φ=0.216 N64/128/160 超线
   （3.44%/2.70%/4.61%）。
3. **≥2 档网格单调收敛**：φ=0.343 严格单调 ✓；φ=0.125（0.36→1.47 反向）、
   φ=0.216（1.78→2.70→4.61 反向）✗。

## 根因/诊断（稀释档失败的定量解剖，均非 solver 共性路径缺陷）

1. **fp32 注入地板**：正式跑前预检（φ=0.343, N=128）K(fs=1/10/100) =
   124.366 / 132.622 / 129.311，散布 6.64% > 0.5% 阈 → 地板激活，正式协议取 fs=10
   （fs=100 地板净但 Re_p≈5 带惯性 +2.6% 不可取；fs=10 折中 Re_p≈0.5）。
   细 N 稀释档 a_body ∝ N⁻³ 缩进注入量化域：fs=1 诊断链 φ=0.125 N64/96 err
   +7.40%/+7.56%（地板扰乱域内）。
2. **负 O(Re) 惯性修正偏稀释侧**：fs=10 下 Re_p≈0.5（协议目标 0.05）；τ 探针
   （φ=0.216/N96 固定 Re_p）K(τ=0.75/1.25) = 7.526 / 7.217，对 τ=1.0 正式同档
   N96（K=7.312）分别为 +2.93%/-1.30% → 全程反弹 BB τ 耦合存在、τ=1 处二阶小。
3. φ=0.343 高固相档 a_body 大、离注入地板远，故双规则全过——严格标准下
   **verified 候选**（四档单调、最细 1.45%）。

## 晋级路径（转 verified 条件；达标即归档 PR 移 pending/permeability3d → verified/permeability3d）

1. **φ=0.343（现成候选）**：owner 裁定晋升——冻结三档（2.05% 最细）与修正四档
   （1.45% 最细）双过，可整档拆出或整目录晋级；
2. **φ=0.125/0.216 补跑**：fp64 或 fs≥100 + 稳定域内 Re_p→0.05 消注入地板与惯性
   偏移后重跑四档阶梯，单调 + 最细 ≤3% 即并入；
3. 达标即归档 PR 移 `pending/permeability3d` → `verified/permeability3d`。

## 运行方式

```bash
cd benchmarks/pending/permeability3d
PYTHONPATH=../../../src python run.py case --phi 0.343 --n 160 --force-scale 10   # 单案
PYTHONPATH=../../../src python run.py report                                       # 聚合 result.json
PYTHONPATH=../../../src python verify.py                                          # 独立复算（170 项检查）
```

工件：`result.json`（env/协议/全案/双规则判定）、`case_phi*.json`（正式 + fs1/fs3 诊断 +
τ 探针 + N32 冒烟）、`NOTES.md`（预注册快照 + 修订链）。完整过程工件留服务器暂存
`/nfs/wangxi/runs/bm_widen_w6_20260921/permeability3d/`。

<!-- PROVENANCE
[{"v": 4.292, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "reference.locked_table_protocol_points.0.125"},
{"v": 7.4423, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "reference.locked_table_protocol_points.0.216"},
{"v": 15.402, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "reference.locked_table_protocol_points.0.343"},
{"v": 4.1648782047793045, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "amendment1_ladders.phi0.125.tiers[0].K_sim"},
{"v": 4.276434118190498, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "amendment1_ladders.phi0.125.tiers[1].K_sim"},
{"v": 4.230032657495238, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "amendment1_ladders.phi0.125.tiers[2].K_sim"},
{"v": 4.344759279501507, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "amendment1_ladders.phi0.125.tiers[3].K_sim"},
{"v": 7.19494790462511, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "amendment1_ladders.phi0.216.tiers[0].K_sim"},
{"v": 7.311932220783275, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "amendment1_ladders.phi0.216.tiers[1].K_sim"},
{"v": 7.246952522833029, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "amendment1_ladders.phi0.216.tiers[2].K_sim"},
{"v": 7.114119104502873, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "amendment1_ladders.phi0.216.tiers[3].K_sim"},
{"v": 14.72110608832727, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "amendment1_ladders.phi0.343.tiers[0].K_sim"},
{"v": 14.939179311648926, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "amendment1_ladders.phi0.343.tiers[1].K_sim"},
{"v": 15.092724138038292, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "amendment1_ladders.phi0.343.tiers[2].K_sim"},
{"v": 15.182333569832728, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "amendment1_ladders.phi0.343.tiers[3].K_sim"},
{"v": 0.030522331979557116, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "amendment1_ladders.phi0.125.tiers[0].err"},
{"v": 0.003639920873161673, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "amendment1_ladders.phi0.125.tiers[1].err"},
{"v": 0.0146493768540914, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "amendment1_ladders.phi0.125.tiers[2].err"},
{"v": 0.012143199682068562, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "amendment1_ladders.phi0.125.tiers[3].err"},
{"v": 0.03437858045030273, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "amendment1_ladders.phi0.216.tiers[0].err"},
{"v": 0.017829456740062533, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "amendment1_ladders.phi0.216.tiers[1].err"},
{"v": 0.026955810259758906, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "amendment1_ladders.phi0.216.tiers[2].err"},
{"v": 0.04613092509083905, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "amendment1_ladders.phi0.216.tiers[3].err"},
{"v": 0.04625290433934337, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "amendment1_ladders.phi0.343.tiers[0].err"},
{"v": 0.030980328885280795, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "amendment1_ladders.phi0.343.tiers[1].err"},
{"v": 0.020491718998708608, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "amendment1_ladders.phi0.343.tiers[2].err"},
{"v": 0.014468555124078408, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "amendment1_ladders.phi0.343.tiers[3].err"},
{"v": 64, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "amendment1_ladders.phi0.125.tiers[0].N"},
{"v": 96, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "amendment1_ladders.phi0.125.tiers[1].N"},
{"v": 128, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "amendment1_ladders.phi0.125.tiers[2].N"},
{"v": 160, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "amendment1_ladders.phi0.125.tiers[3].N"},
{"v": "FAIL", "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "ladders.phi0.125.verdict"},
{"v": false, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "ladders.phi0.125.strictly_monotonic"},
{"v": "FAIL", "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "ladders.phi0.216.verdict"},
{"v": false, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "ladders.phi0.216.strictly_monotonic"},
{"v": "PASS", "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "ladders.phi0.343.verdict"},
{"v": true, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "ladders.phi0.343.strictly_monotonic"},
{"v": "PASS", "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "amendment1_ladders.phi0.125.verdict"},
{"v": "FAIL", "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "amendment1_ladders.phi0.216.verdict"},
{"v": "PASS", "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "amendment1_ladders.phi0.343.verdict"},
{"v": "FAIL", "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "per_phi_verdict.phi0.125"},
{"v": "FAIL", "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "per_phi_verdict.phi0.216"},
{"v": "PASS", "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "per_phi_verdict.phi0.343"},
{"v": "PASS", "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "per_phi_verdict_amended.phi0.125"},
{"v": "FAIL", "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "per_phi_verdict_amended.phi0.216"},
{"v": "PASS", "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "per_phi_verdict_amended.phi0.343"},
{"v": 0.00015, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "reference.two_source_cross_max_rel_diff_protocol_points"},
{"v": "FAIL", "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "overall_verdict"},
{"v": "FAIL", "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "overall_verdict_amended"},
{"v": 10.0, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "formal_force_scale"},
{"v": 0.03, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "tolerance_finest"},
{"v": 1.0, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "protocol.tau"},
{"v": 0.05, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "protocol.Re_target"},
{"v": 0.06638616302733391, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "fs_precheck.max_rel_spread"},
{"v": 124.36595541361314, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "fs_precheck.k_sim_by_fs.1"},
{"v": 132.6221340047514, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "fs_precheck.k_sim_by_fs.10"},
{"v": 129.31110489689812, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "fs_precheck.k_sim_by_fs.100"},
{"v": 0.07398997357410897, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "fs1_double_report.phi0.125_N64.err_primary"},
{"v": 0.07558890086392234, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "fs1_double_report.phi0.125_N96.err_primary"},
{"v": 0.011506408772164134, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "fs1_double_report.phi0.343_N96.err_primary"},
{"v": 7.5263399306658885, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "tau_probe.tau0.75.K_sim"},
{"v": 7.217058832744019, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "tau_probe.tau1.25.K_sim"},
{"v": 0.013560916920278446, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "amendment1_ladders.phi0.343.tiers[0].Ma_max"},
{"v": 0.0006093756694800501, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "amendment1_ladders.phi0.125.tiers[3].mass_drift_max_rel"},
{"v": 0.3430018424987793, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "amendment1_ladders.phi0.343.tiers[2].phi_actual"},
{"v": 3.0522331979557116, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "amendment1_ladders.phi0.125.tiers[0].err", "k2": "amendment1_ladders.phi0.125.tiers[0].err", "op": "scale100"},
{"v": 0.3639920873161673, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "amendment1_ladders.phi0.125.tiers[1].err", "k2": "amendment1_ladders.phi0.125.tiers[1].err", "op": "scale100"},
{"v": 1.46493768540914, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "amendment1_ladders.phi0.125.tiers[2].err", "k2": "amendment1_ladders.phi0.125.tiers[2].err", "op": "scale100"},
{"v": 1.2143199682068562, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "amendment1_ladders.phi0.125.tiers[3].err", "k2": "amendment1_ladders.phi0.125.tiers[3].err", "op": "scale100"},
{"v": 3.437858045030273, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "amendment1_ladders.phi0.216.tiers[0].err", "k2": "amendment1_ladders.phi0.216.tiers[0].err", "op": "scale100"},
{"v": 1.7829456740062533, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "amendment1_ladders.phi0.216.tiers[1].err", "k2": "amendment1_ladders.phi0.216.tiers[1].err", "op": "scale100"},
{"v": 2.6955810259758906, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "amendment1_ladders.phi0.216.tiers[2].err", "k2": "amendment1_ladders.phi0.216.tiers[2].err", "op": "scale100"},
{"v": 4.613092509083905, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "amendment1_ladders.phi0.216.tiers[3].err", "k2": "amendment1_ladders.phi0.216.tiers[3].err", "op": "scale100"},
{"v": 4.625290433934337, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "amendment1_ladders.phi0.343.tiers[0].err", "k2": "amendment1_ladders.phi0.343.tiers[0].err", "op": "scale100"},
{"v": 3.0980328885280795, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "amendment1_ladders.phi0.343.tiers[1].err", "k2": "amendment1_ladders.phi0.343.tiers[1].err", "op": "scale100"},
{"v": 2.04917189987086, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "amendment1_ladders.phi0.343.tiers[2].err", "k2": "amendment1_ladders.phi0.343.tiers[2].err", "op": "scale100"},
{"v": 1.4468555124078408, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "amendment1_ladders.phi0.343.tiers[3].err", "k2": "amendment1_ladders.phi0.343.tiers[3].err", "op": "scale100"},
{"v": 7.398997357410897, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "fs1_double_report.phi0.125_N64.err_primary", "k2": "fs1_double_report.phi0.125_N64.err_primary", "op": "scale100"},
{"v": 7.558890086392234, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "fs1_double_report.phi0.125_N96.err_primary", "k2": "fs1_double_report.phi0.125_N96.err_primary", "op": "scale100"},
{"v": 6.638616302733391, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "fs_precheck.max_rel_spread", "k2": "fs_precheck.max_rel_spread", "op": "scale100"},
{"v": 2.9322989246698183, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "tau_probe.tau0.75.K_sim", "k2": "amendment1_ladders.phi0.216.tiers[1].K_sim", "op": "pct_vs_ref"},
{"v": -1.2975145990766879, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "tau_probe.tau1.25.K_sim", "k2": "amendment1_ladders.phi0.216.tiers[1].K_sim", "op": "pct_vs_ref"},
{"v": 2.5605141263723707, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json", "k": "fs_precheck.k_sim_by_fs.10", "k2": "fs_precheck.k_sim_by_fs.100", "op": "pct_vs_ref"}]
-->
