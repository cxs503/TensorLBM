# Gebart 纤维床横向渗透率：周期方阵/六角圆柱阵列（pending）

**状态：❌ 未达标（根因类，三失效模式全部定源）——冻结门 FAIL（sq/hex 双 FAIL）：hex 近堆积极限档 = 参考侧（Gebart 渐近式对精确解偏高 +2.4%/+3.9%，已由方阵精确表双通道锁定）；sq 0.40 = 几何量化地板振荡（gap-96 扩展案深入门内 −0.65%）；S1 = 真实 O(Re) 惯性。方阵 4 主档中 3 档（0.30/0.50/0.60）逐档全过门且严格单调 = verified 候选（owner 决策）。**

## 物理问题

纤维床（周期圆柱阵列）横向渗透率 K⊥——两类阵列：

- **方阵（sq）**：W×W 单圆柱居中周期胞（W4-A 渗透率案例继承），主参考 =
  **Sangani & Acrivos 1982 精确 f 表**（W4-A 双转录逐位一致；k_ref = π/(vf·f)·R²）；
- **六角（hex）**：W×H 超胞（H=round(√3·W)），主参考（近堆积 0.70/0.75 主档）=
  **Gebart 1992 近堆积闭式** K/R² = C1·(√(vf_max/vf)−1)^2.5（sq C1=16/(9π√2)、
  hex 16/(9π√6)）；**hex 精确表不可得**（Koch & Ladd 1997 / Wang & Sangani 1997
  闭源，8 通道检索见 prereg §3），Gebart 闭式对每档作一致性通道。
- 口径：Darcy/表观（K = μ·U_sup/G），K_sim = (1−vf_actual)·ν·⟨u_x⟩_fluid/a_body，
  名义几何（无有效半径再标定）；无外推。

### 共性模块

- D2Q9 BGK τ=1（ν=1/6 精确），全周期单元胞；步序 collide → stream →（流体掩码）
  体力 → bounce-back；物理核全部走 tensorlbm 库入口（铁律 grep 零命中，
  `compile_route` 适配）。

## 计算结果现状（`result.json` 冻结 report，err 对档主参考）

| 档 | 阶梯 err (%) | 单调 | finest (%) | 过门 | 档类 |
|----|--------------|------|-----------|------|------|
| sq_vf0.3 | 15.23, 4.78, 2.18, 0.22 | True | 0.22 | True | main |
| sq_vf0.4 | 17.25, 6.92, 2.42, 2.63 | False | 2.63 | False | main |
| sq_vf0.5 | 16.03, 5.08, 2.96, 0.24 | True | 0.24 | True | main |
| sq_vf0.6 | 17.19, 5.99, 1.64, 0.88 | True | 0.88 | True | main |
| sq_vf0.7 | 17.12, 6.47, 3.85 | True | 3.85 | False | disclosure |
| hex_vf0.3 | 5.32, 2.74, 1.22 | True | 1.22 | True | disclosure |
| hex_vf0.45 | 5.55, 2.58, 2.16 | True | 2.16 | True | disclosure |
| hex_vf0.6 | 3.89, 2.09, 0.62 | True | 0.62 | True | disclosure |
| hex_vf0.7 | 1.66, 0.47, 0.13, 2.10 | False | 2.10 | False | main |
| hex_vf0.75 | 2.38, 0.34, 0.85, 2.03 | False | 2.03 | False | main |

- 判据（冻结）：err（对档主参考）随加密严格单调下降 AND finest ≤3% AND 全档稳态；
  主档门（sq 0.30/0.40/0.50/0.60 对 S&A 精确表；hex 0.70/0.75 对 Gebart）定
  per-array 判决——**sq: FAIL（0.40 一档破）、hex: FAIL（两主档全破）、总判 FAIL**。
- R3 参考自产门（`hex_ref_gate/`，Schur-complement Stokes 解算器 + 51 case 阶梯池，
  0.5% 三估计器门）：vf0.3/0.4/0.5 PASS，vf0.6 差 0.052pp（fixed-α1.5 −0.552%）、
  vf0.7 FAIL（Aitken₃ −0.797%）、vf0.75 FAIL——按 prereg 11.5 停止，未产 hex 值。

| vf | ref_f | fixed-α1.0 (%) | fixed-α1.5 (%) | Aitken₃ (%) | 过门 |
|----|-------|----------------|----------------|-------------|------|
| 0.3 | 102.9 | -0.169 | -0.228 | -0.073 | True |
| 0.4 | 217.89 | -0.085 | -0.250 | -0.339 | True |
| 0.5 | 532.55 | +0.199 | -0.245 | -0.390 | True |
| 0.6 | 1763 | -0.188 | -0.552 | -0.283 | False |
| 0.7 | 13520 | +0.156 | -0.432 | -0.797 | False |
| 0.75 | 126300 | +1.762 | -0.942 | -7.671 | False |

## 不达标清单（严格标准：共性模块入口 + 直接观测量 ≤3% + ≥2 档网格单调收敛）

1. **共性模块入口**：✓——物理核全部库入口（铁律），几何/测量为 driver 层。
2. **直接观测量 ≤3%**：主档全部数值过线（sq 最细 0.22/2.63/0.24/0.88%、
   hex 0.70/0.75 最细 2.10/2.03%）；唯披露档 sq_vf0.7 最细 3.85% 超线（非门）。
3. **≥2 档网格单调收敛**：✗——三个档单调破缺：sq_vf0.4（2.42→2.63，+0.21pp）、
   hex_vf0.7（0.13→2.10）、hex_vf0.75（0.34→0.85→2.03）——判据的单调条款是
   本案例唯一实质败因。

## 根因/诊断（三种失效模式，全部机读定源）

1. **hex 近堆积 = 参考侧**：解析 LBM 的 K 在 hex 0.70/0.75 真实低于 Gebart——
   误差随 gap 增大持续走负（楼梯伪影应衰减归零，参考偏差则随楼梯份额缩小显形）。
   阶梯分解（1+signed = (1+staircase)·(1+gebart_bias)，DIAGNOSTICS.md D1，
   独立复核）：hex 0.75 阶梯项 [4.9, 2.9, 1.6, 0.5]%（gap 12/24/48/72）、
   Gebart 偏差 +2.4%；hex 0.70 阶梯项 [6.3, 4.4, 4.0, 1.8]%、偏差 +3.9%。
   Gebart 对精确表的锁定偏差（方阵阶梯原位复现）：vf0.70 +2.978%、vf0.75 +0.268%。
   排除项：过早稳态（S2 全过）、惯性（S1 hex0.75 对 0.073%）。
   附：这是 hex 0.70/0.75 横向渗透率的首批解析 LBM 测量（求解器经四独立通道验证）。
2. **sq 0.40 = 几何量化地板振荡**：正式阶梯 +17.25/+6.92/+2.42/+2.63% 在第 3→4 档
   回升 ~0.2pp；gap-96 扩展案返回 −0.6475%（门内深处）→ 回升是分辨率地板处的
   半径量化振荡（dR/R 离散被渗透率对半径的强敏感放大），非发散趋势。
3. **S1 = 真实 O(Re) 惯性**：fs 10→20 力线性在 sq0.3 W64 破容差（0.256% > 0.1%），
   继承力标 Re_cell=0.5 下的真实惯性；hex0.75 对过（0.073%）。物理量、测得正确、
   按冻结协议如实记 FAIL（不重跑）；fs=20 诊断跑因 W4-A fp32 裕度守卫在 gap 96
   拒绝 fs=10 而运行在 Re_cell≈1.0。
   sq0.70 披露档补全：3 档正式阶梯 17.12/6.47/3.85%（冻结非门），gap-72 扩展
   −1.4686% 已入门内，按冻结 prereg 维持非门（档冻结 3 档）。

## 晋级路径（转 verified 条件；达标即归档 PR 移 pending/permeability_fibrous → verified/permeability_fibrous）

1. **方阵 verified 候选（现成）**：sq 0.30/0.50/0.60 三主档全过（最细
   0.22/0.24/0.88%，严格单调）——owner 晋级决策（整档拆出或与 0.40 合并处置）；
2. **hex 精确参考取得（owner 菜单）**：(a) M1 cut-cell 重跑方阵全阵自产精确表
   （prereg §3 注册变体，轨道推荐）；(b) 门放宽到 1%（vf0.6/0.7 即过；hex 误差带
   等比放宽）；(c) 购 Koch-Ladd 1997 / Wang-Sangani 1997 表；(d) 维持 Gebart 门
   接受 FAIL（零成本；参考侧偏差 +2.4%/+3.9% 已定量定源）；
3. **sq 0.40**：加 gap≥96 rung 或改包络收敛判据（同 permeability3d 修正规则）重判。
4. 达标即归档 PR 移 `pending/permeability_fibrous` → `verified/permeability_fibrous`。

## 运行方式

```bash
cd benchmarks/pending/permeability_fibrous
PYTHONPATH=../../../src python run.py case --array sq --vf 0.5 --w 238 --device cuda:0  # 单案
PYTHONPATH=../../../src python run.py report --out-dir out                              # 聚合判决
```

R3 自产门（`hex_ref_gate/src/`，queue_sq/queue_hex/extrapolate 等）与扩展定源案
（`out_diag/`）独立可跑。工件：`result.json`（冻结 report 原文 + wave9_record 附录）、
`out/`（40 份 case 机器档案）、`out_diag/`（4 案）、`prereg.md`、`DIAGNOSTICS.md`、
`hex_ref_gate/`（NOTES + prereg + src + out：51 case + 双门 JSON + owner report）。
完整过程工件留服务器暂存 `/nfs/wangxi/runs/bm_widen_w9_20260929/fibrous_perm/`。

<!-- PROVENANCE
[{"v": 1.0, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "settings.tau"},
{"v": 0.05, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "settings.Re_target"},
{"v": 10.0, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "settings.force_scale_default"},
{"v": "FAIL", "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "verdict.benchmark_overall"},
{"v": "FAIL", "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "verdict.overall.sq"},
{"v": "FAIL", "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "verdict.overall.hex"},
{"v": true, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "verdict.per_array.sq.main_tiers.sq_vf0.3"},
{"v": false, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "verdict.per_array.sq.main_tiers.sq_vf0.4"},
{"v": true, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "verdict.per_array.sq.main_tiers.sq_vf0.5"},
{"v": true, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "verdict.per_array.sq.main_tiers.sq_vf0.6"},
{"v": false, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "verdict.per_array.hex.main_tiers.hex_vf0.7"},
{"v": false, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "verdict.per_array.hex.main_tiers.hex_vf0.75"},
{"v": 15.231435079977329, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "verdict.per_tier.sq_vf0.3.errors_pct[0]"},
{"v": 4.782804330150681, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "verdict.per_tier.sq_vf0.3.errors_pct[1]"},
{"v": 2.18439458411559, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "verdict.per_tier.sq_vf0.3.errors_pct[2]"},
{"v": 0.2160566801704511, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "verdict.per_tier.sq_vf0.3.errors_pct[3]"},
{"v": 17.25228839329256, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "verdict.per_tier.sq_vf0.4.errors_pct[0]"},
{"v": 6.920144431623343, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "verdict.per_tier.sq_vf0.4.errors_pct[1]"},
{"v": 2.418694925947107, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "verdict.per_tier.sq_vf0.4.errors_pct[2]"},
{"v": 2.6283936232617267, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "verdict.per_tier.sq_vf0.4.errors_pct[3]"},
{"v": 16.031529384966483, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "verdict.per_tier.sq_vf0.5.errors_pct[0]"},
{"v": 5.083774163499544, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "verdict.per_tier.sq_vf0.5.errors_pct[1]"},
{"v": 2.957754152099601, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "verdict.per_tier.sq_vf0.5.errors_pct[2]"},
{"v": 0.2424357355092388, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "verdict.per_tier.sq_vf0.5.errors_pct[3]"},
{"v": 17.19004279588514, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "verdict.per_tier.sq_vf0.6.errors_pct[0]"},
{"v": 5.993759157473222, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "verdict.per_tier.sq_vf0.6.errors_pct[1]"},
{"v": 1.6381862785754153, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "verdict.per_tier.sq_vf0.6.errors_pct[2]"},
{"v": 0.8756241746041393, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "verdict.per_tier.sq_vf0.6.errors_pct[3]"},
{"v": 17.120656150017165, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "verdict.per_tier.sq_vf0.7.errors_pct[0]"},
{"v": 6.467105687375541, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "verdict.per_tier.sq_vf0.7.errors_pct[1]"},
{"v": 3.851477863002062, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "verdict.per_tier.sq_vf0.7.errors_pct[2]"},
{"v": 5.3237831758605525, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "verdict.per_tier.hex_vf0.3.errors_pct[0]"},
{"v": 2.73925185489865, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "verdict.per_tier.hex_vf0.3.errors_pct[1]"},
{"v": 1.221918649335052, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "verdict.per_tier.hex_vf0.3.errors_pct[2]"},
{"v": 5.548163690917041, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "verdict.per_tier.hex_vf0.45.errors_pct[0]"},
{"v": 2.581773666622955, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "verdict.per_tier.hex_vf0.45.errors_pct[1]"},
{"v": 2.159710824148986, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "verdict.per_tier.hex_vf0.45.errors_pct[2]"},
{"v": 3.886490541268972, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "verdict.per_tier.hex_vf0.6.errors_pct[0]"},
{"v": 2.0908822545763206, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "verdict.per_tier.hex_vf0.6.errors_pct[1]"},
{"v": 0.6218225965950896, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "verdict.per_tier.hex_vf0.6.errors_pct[2]"},
{"v": 1.6613065077422107, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "verdict.per_tier.hex_vf0.7.errors_pct[0]"},
{"v": 0.4678446386337187, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "verdict.per_tier.hex_vf0.7.errors_pct[1]"},
{"v": 0.13106477515814063, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "verdict.per_tier.hex_vf0.7.errors_pct[2]"},
{"v": 2.0988291146574634, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "verdict.per_tier.hex_vf0.7.errors_pct[3]"},
{"v": 2.3771076248163325, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "verdict.per_tier.hex_vf0.75.errors_pct[0]"},
{"v": 0.3383014680396368, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "verdict.per_tier.hex_vf0.75.errors_pct[1]"},
{"v": 0.8473364265903727, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "verdict.per_tier.hex_vf0.75.errors_pct[2]"},
{"v": 2.0250093124438706, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "verdict.per_tier.hex_vf0.75.errors_pct[3]"},
{"v": false, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "verdict.per_tier.sq_vf0.4.monotone_decreasing"},
{"v": false, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "verdict.per_tier.hex_vf0.7.monotone_decreasing"},
{"v": false, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "verdict.per_tier.hex_vf0.75.monotone_decreasing"},
{"v": true, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "verdict.per_tier.sq_vf0.3.monotone_decreasing"},
{"v": true, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "verdict.per_tier.sq_vf0.5.monotone_decreasing"},
{"v": true, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "verdict.per_tier.sq_vf0.6.monotone_decreasing"},
{"v": 102.9, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "reference.sangani_f_table_sq.0.3"},
{"v": 217.89, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "reference.sangani_f_table_sq.0.4"},
{"v": 532.55, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "reference.sangani_f_table_sq.0.5"},
{"v": 1763.0, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "reference.sangani_f_table_sq.0.6"},
{"v": 13520.0, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "reference.sangani_f_table_sq.0.7"},
{"v": 126300.0, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "reference.sangani_f_table_sq.0.75"},
{"v": 2.978, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "reference_deviation_disclosure.sq_gebart_vs_sangani_exact_pct.0.7"},
{"v": 0.268, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "reference_deviation_disclosure.sq_gebart_vs_sangani_exact_pct.0.75"},
{"v": 2.4, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "wave9_record.failure_attribution.hex_reference_side.staircase_decomposition.hex_0.75_gebart_bias_pct"},
{"v": 3.9, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "wave9_record.failure_attribution.hex_reference_side.staircase_decomposition.hex_0.70_gebart_bias_pct"},
{"v": 4.9, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "wave9_record.failure_attribution.hex_reference_side.staircase_decomposition.hex_0.75_staircase_pct_by_gap_12_24_48_72[0]"},
{"v": 2.9, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "wave9_record.failure_attribution.hex_reference_side.staircase_decomposition.hex_0.75_staircase_pct_by_gap_12_24_48_72[1]"},
{"v": 1.6, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "wave9_record.failure_attribution.hex_reference_side.staircase_decomposition.hex_0.75_staircase_pct_by_gap_12_24_48_72[2]"},
{"v": 0.5, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "wave9_record.failure_attribution.hex_reference_side.staircase_decomposition.hex_0.75_staircase_pct_by_gap_12_24_48_72[3]"},
{"v": 6.3, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "wave9_record.failure_attribution.hex_reference_side.staircase_decomposition.hex_0.70_staircase_pct_by_gap_12_24_48_72[0]"},
{"v": 4.4, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "wave9_record.failure_attribution.hex_reference_side.staircase_decomposition.hex_0.70_staircase_pct_by_gap_12_24_48_72[1]"},
{"v": 4.0, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "wave9_record.failure_attribution.hex_reference_side.staircase_decomposition.hex_0.70_staircase_pct_by_gap_12_24_48_72[2]"},
{"v": 1.8, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "wave9_record.failure_attribution.hex_reference_side.staircase_decomposition.hex_0.70_staircase_pct_by_gap_12_24_48_72[3]"},
{"v": -0.6474881319222692, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "wave9_record.failure_attribution.sq_0.40_quantization_floor.extension_signed_err_pct"},
{"v": -1.4686040664330768, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "wave9_record.failure_attribution.sq_0.70_disclosure_completion.extension_signed_err_pct"},
{"v": 0.07324999527964504, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "sensitivity.S1_force_linearity.case_hex_vf0.75_w265_fs20.json.delta_pct"},
{"v": 0.1, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "sensitivity.S1_force_linearity.case_hex_vf0.75_w265_fs20.json.tol_pct"},
{"v": 0.2557244059488428, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "sensitivity.S1_force_linearity.case_sq_vf0.3_w64_fs20.json.delta_pct"},
{"v": 0.1, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "sensitivity.S1_force_linearity.case_sq_vf0.3_w64_fs20.json.tol_pct"},
{"v": 161.38611649329235, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "wave9_record.extension_runs.cases[0].K_sim_darcy"},
{"v": -3.6963492824352295, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "wave9_record.extension_runs.cases[0].signed_err_pct"},
{"v": 194.3886374677627, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "wave9_record.extension_runs.cases[1].K_sim_darcy"},
{"v": -1.6283636878920427, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "wave9_record.extension_runs.cases[1].signed_err_pct"},
{"v": 511.7185572717894, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "wave9_record.extension_runs.cases[2].K_sim_darcy"},
{"v": -0.6474881319222692, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "wave9_record.extension_runs.cases[2].signed_err_pct"},
{"v": 121.84143177524994, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "wave9_record.extension_runs.cases[3].K_sim_darcy"},
{"v": -1.4686040664330768, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "wave9_record.extension_runs.cases[3].signed_err_pct"},
{"v": 0.5761571753998865, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json", "k": "ladders.sq_vf0.3[0].Re_cell_achieved"},
{"v": -0.16910853386309377, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/hex_ref_gate/out/sq_gate2.json", "k": "vf0.3.est_errs.fixed10.err_pct"},
{"v": -0.2280976641777288, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/hex_ref_gate/out/sq_gate2.json", "k": "vf0.3.est_errs.fixed15.err_pct"},
{"v": -0.07277542127638315, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/hex_ref_gate/out/sq_gate2.json", "k": "vf0.3.est_errs.aitken3.err_pct"},
{"v": true, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/hex_ref_gate/out/sq_gate2.json", "k": "vf0.3.gate_pass_3est"},
{"v": -0.08519124687147439, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/hex_ref_gate/out/sq_gate2.json", "k": "vf0.4.est_errs.fixed10.err_pct"},
{"v": -0.25041764443365366, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/hex_ref_gate/out/sq_gate2.json", "k": "vf0.4.est_errs.fixed15.err_pct"},
{"v": -0.33927639488661177, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/hex_ref_gate/out/sq_gate2.json", "k": "vf0.4.est_errs.aitken3.err_pct"},
{"v": true, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/hex_ref_gate/out/sq_gate2.json", "k": "vf0.4.gate_pass_3est"},
{"v": 0.19902603481196657, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/hex_ref_gate/out/sq_gate2.json", "k": "vf0.5.est_errs.fixed10.err_pct"},
{"v": -0.24483811335509476, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/hex_ref_gate/out/sq_gate2.json", "k": "vf0.5.est_errs.fixed15.err_pct"},
{"v": -0.390079617530259, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/hex_ref_gate/out/sq_gate2.json", "k": "vf0.5.est_errs.aitken3.err_pct"},
{"v": true, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/hex_ref_gate/out/sq_gate2.json", "k": "vf0.5.gate_pass_3est"},
{"v": -0.18796620821104337, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/hex_ref_gate/out/sq_gate2.json", "k": "vf0.6.est_errs.fixed10.err_pct"},
{"v": -0.5520652028774164, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/hex_ref_gate/out/sq_gate2.json", "k": "vf0.6.est_errs.fixed15.err_pct"},
{"v": -0.28285839899955256, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/hex_ref_gate/out/sq_gate2.json", "k": "vf0.6.est_errs.aitken3.err_pct"},
{"v": false, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/hex_ref_gate/out/sq_gate2.json", "k": "vf0.6.gate_pass_3est"},
{"v": 0.15586169892045643, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/hex_ref_gate/out/sq_gate2.json", "k": "vf0.7.est_errs.fixed10.err_pct"},
{"v": -0.4319506770494086, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/hex_ref_gate/out/sq_gate2.json", "k": "vf0.7.est_errs.fixed15.err_pct"},
{"v": -0.7967327932867474, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/hex_ref_gate/out/sq_gate2.json", "k": "vf0.7.est_errs.aitken3.err_pct"},
{"v": false, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/hex_ref_gate/out/sq_gate2.json", "k": "vf0.7.gate_pass_3est"},
{"v": 1.7615533703343456, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/hex_ref_gate/out/sq_gate2.json", "k": "vf0.75.est_errs.fixed10.err_pct"},
{"v": -0.9415624682871049, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/hex_ref_gate/out/sq_gate2.json", "k": "vf0.75.est_errs.fixed15.err_pct"},
{"v": -7.670830474559532, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/hex_ref_gate/out/sq_gate2.json", "k": "vf0.75.est_errs.aitken3.err_pct"},
{"v": false, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/hex_ref_gate/out/sq_gate2.json", "k": "vf0.75.gate_pass_3est"}]
-->
