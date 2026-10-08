# 背向台阶 Re=100（backward-facing step，Armaly/Erturk 口径）（pending）

**状态：❌ 未达标(根因类)——τ-matched 收敛梯子最好档 -4.33% > 3% 门，残余偏差已完整定源为与 Re 无关的格点伪差（bounce-back 滑移 + 台阶角分辨率），且参考本身受限（实验 3.05 vs 2D 数值 2.878 散布 ~6%）**

（Wave-2 W2-C 定源记录，PR #301 入库；本 README 为案例导引，完整判决见 `result.json` 与暂存 `NOTES.md` 预注册。）

## 物理问题

Armaly 1983（JFM 127:473）背向台阶装置：入口段高 h_i=0.52 cm、台阶高 s=0.49 cm、下游高 H=1.01 cm，ER=1.9423；Re = Ū·2h_i/ν = 100（Armaly 原文口径，ν 按离散入口 profile 平均定标到 Re=100.00 精确）。观测量为底壁再附着长度 X_r（主判据 = 流函数 ψ 基再附着，等价 Erturk X1 定义）。

参考（`result.json` reference 节 + `REFERENCE_AUDIT.md`）：

- 主参考 Erturk 2008 Table 1（ER=1.942）：X_r = 2.878（验收带 [2.792, 2.965]）；
- 实验锚 Armaly 1983（NAFEMS 数字化）：3.05；
- 交叉：Denham 实验 vs Erturk 在 Re_h 174–425 一致 ≤1.2%，Armaly 实验 vs Erturk 在目标点差 +6.0%；
- `REFERENCE_AUDIT.md` 另录 Gartling 1990 Re=800/ER=2 口径（X1/H 数字化簇，具体数字见该文件）为备选参考审计，未用于本判决。

## 计算结果现状

判定梯子 = τ-matched（τ=0.602 三档固定、U_max ∝ 1/h_i，(τ-1/2)·Δx 逐档收缩——bounce-back 滑移真正收敛的序列）：

| 档 | h_i（入口半高格数） | U_max | X_r(ψ) | 误差 vs Erturk 2.878 | 误差 vs Armaly 3.05 |
|----|----|----|----|----|----|
| L1 | 51 | 0.05 | 2.6800 | -6.88% | -12.13% |
| L2p | 82 | 0.0310976 | 2.7177 | -5.57% | -10.89% |
| L3p | 123 | 0.0207317 | 2.7534 | -4.33% | -9.72% |

- 单调收敛成立（X_r 逐档升、误差逐档降），最好档 -4.33% 未进 3% 门。
- 同网格 Re 扫描（L3 网格、τ=0.602）：Re=150 X_r=3.7854（对 Erturk 内插 3.889 为 -2.66%）、Re=200 X_r=4.8047（对表值 4.900 为 -1.94%）——绝对偏差近似常数（-0.125/-0.104/-0.095），伪差与 Re 无关。
- 预注册固定 U_max 梯子（τ 漂移 0.602/0.664/0.746）L3 档 X_r(ψ)=2.8053、-2.53%，含 τ 漂移混杂不可字面判读（NOTES §11 修订 2）。

## 不达标清单

（战役严格标准：库共性模块入口 + 直接观测量对参考 ≤3% + ≥2 档网格单调收敛；修正/还原变换不算直接模拟）

1. **精度门 ✗**：τ-matched 梯子三档 -6.88/-5.57/-4.33%，最好档 -4.33% > 3%（门 = `result.json` 判定口径）。
2. **单调收敛条款 ✓**：误差逐档严格下降（见上表）。
3. **参考受限**：目标点两锚 2.878 vs 3.05 互斥（散布 ~6% > 3%），任何模拟不可能同时满足两锚——判决取决于主参考选择，控制方裁决。
4. 口径披露：预注册 row-1 测量被证明偏低（修订 1）；连续极限外推 2.845±0.04（对 Erturk -1.1%）为模型推断，预注册规定不作达标依据。

## 根因/诊断

已定源（`result.json` artifact_decomposition 节）：**与 Re 无关的格点伪差**，分解模型 X_r(网格,τ) = X∞ + A·Δx^p + B·(τ-1/2)·Δx：

- B 三次独立测定：41.8（L1 vs S3b）/ 35.6（L2 vs L2p）/ 48.9（L3p vs S5b）——Δx 标度成立；
- 滑移贡献：L1 +0.087 / L2p +0.052 / L3p +0.036；
- 去滑移后分辨率序列 2.593 → 2.666 → 2.718，外推 X∞ = 2.845（不确定度 0.04，对 Erturk -1.1%、对 Armaly -6.7%）；
- L3 网格 τ 全谱 2.7534（τ=0.602）/ 2.8053（0.746）/ 2.8650（0.869）平滑单调，伪差被完全包络；
- 旧 pending 基线大幅正偏差的根因是 Re 口径错位 + 上游段过短（NOTES §8 预注册分析，S1/S2 敏感性实验佐证：域长/入口剖面变化影响 ≤0.0028）。

## 晋级路径（转 verified 的条件）

1. **控制方参考裁决**（零计算成本）：Erturk（更严）或 Armaly 实验锚二选一；若采实验锚，现有 L3p 档 -9.72% 仍远超 3%，无捷径。
2. 若维持 Erturk：**追加 τ-matched 细档 L4**（延续 U_max ∝ 1/h_i、τ=0.602 的序列再细一档），使最好档直接进 3%（外推表明 X∞ 在 -1.1% 处，物理上可达）；计算量为 L3p（最细档 h_i=123）的约 2–3×（估算）。
3. 或改判 Re=150/200 工况点（同网格/τ 已达 -2.66%/-1.94%），按战役标准补第二档网格形成阶梯。
4. 达标后由归档 PR 将本目录 `pending/backward_step` → `verified/backward_step`（参考受限条款须一并写入）。

<!-- PROVENANCE
[{"v": 2.878, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json", "k": "reference.primary.value"},
 {"v": 2.792, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json", "k": "reference.primary.acceptance_band[0]"},
 {"v": 2.965, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json", "k": "reference.primary.acceptance_band[1]"},
 {"v": 3.05, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json", "k": "reference.experimental_anchor.value"},
 {"v": 51, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json", "k": "ladder_tau_matched.rows[0].h_i"},
 {"v": 0.05, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json", "k": "ladder_tau_matched.rows[0].u_max"},
 {"v": 2.68, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json", "k": "ladder_tau_matched.rows[0].xr_psi"},
 {"v": -6.88, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json", "k": "ladder_tau_matched.rows[0].err_psi_vs_erturk_pct"},
 {"v": -12.13, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json", "k": "ladder_tau_matched.rows[0].err_psi_vs_armaly_pct"},
 {"v": 82, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json", "k": "ladder_tau_matched.rows[1].h_i"},
 {"v": 0.0310976, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json", "k": "ladder_tau_matched.rows[1].u_max"},
 {"v": 2.7177, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json", "k": "ladder_tau_matched.rows[1].xr_psi"},
 {"v": -5.57, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json", "k": "ladder_tau_matched.rows[1].err_psi_vs_erturk_pct"},
 {"v": -10.89, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json", "k": "ladder_tau_matched.rows[1].err_psi_vs_armaly_pct"},
 {"v": 123, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json", "k": "ladder_tau_matched.rows[2].h_i"},
 {"v": 0.0207317, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json", "k": "ladder_tau_matched.rows[2].u_max"},
 {"v": 2.7534, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json", "k": "ladder_tau_matched.rows[2].xr_psi"},
 {"v": -4.33, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json", "k": "ladder_tau_matched.rows[2].err_psi_vs_erturk_pct"},
 {"v": -9.72, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json", "k": "ladder_tau_matched.rows[2].err_psi_vs_armaly_pct"},
 {"v": 3.7854, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json", "k": "re_sweep_diagnostic.rows[1].xr_psi"},
 {"v": 3.889, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json", "k": "re_sweep_diagnostic.rows[1].erturk"},
 {"v": -2.66, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json", "k": "re_sweep_diagnostic.rows[1].err_pct"},
 {"v": 4.8047, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json", "k": "re_sweep_diagnostic.rows[2].xr_psi"},
 {"v": 4.9, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json", "k": "re_sweep_diagnostic.rows[2].erturk"},
 {"v": -1.94, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json", "k": "re_sweep_diagnostic.rows[2].err_pct"},
 {"v": -0.125, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json", "k": "re_sweep_diagnostic.rows[0].abs_dev"},
 {"v": -0.104, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json", "k": "re_sweep_diagnostic.rows[1].abs_dev"},
 {"v": -0.095, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json", "k": "re_sweep_diagnostic.rows[2].abs_dev"},
 {"v": 2.8053, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json", "k": "ladder_pregregistered_fixed_umax.rows[2].xr_psi"},
 {"v": -2.53, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json", "k": "ladder_pregregistered_fixed_umax.rows[2].err_psi_vs_erturk_pct"},
 {"v": 41.8, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json", "k": "artifact_decomposition.B_determinations[0].B"},
 {"v": 35.6, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json", "k": "artifact_decomposition.B_determinations[1].B"},
 {"v": 48.9, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json", "k": "artifact_decomposition.B_determinations[2].B"},
 {"v": 0.087, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json", "k": "artifact_decomposition.slip_contribution_at_tau_matched_levels.L1"},
 {"v": 0.052, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json", "k": "artifact_decomposition.slip_contribution_at_tau_matched_levels.L2p"},
 {"v": 0.036, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json", "k": "artifact_decomposition.slip_contribution_at_tau_matched_levels.L3p"},
 {"v": 2.593, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json", "k": "artifact_decomposition.resolution_sequence_after_slip_removal[0]"},
 {"v": 2.666, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json", "k": "artifact_decomposition.resolution_sequence_after_slip_removal[1]"},
 {"v": 2.718, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json", "k": "artifact_decomposition.resolution_sequence_after_slip_removal[2]"},
 {"v": 2.845, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json", "k": "artifact_decomposition.extrapolated_continuum_limit.value"},
 {"v": 0.04, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json", "k": "artifact_decomposition.extrapolated_continuum_limit.uncertainty"},
 {"v": -1.1, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json", "k": "artifact_decomposition.extrapolated_continuum_limit.err_vs_erturk_pct"},
 {"v": -6.7, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json", "k": "artifact_decomposition.extrapolated_continuum_limit.err_vs_armaly_pct"},
 {"v": 2.865, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json", "k": "artifact_decomposition.tau_bracket_at_L3_grid.tau=0.869"},
 {"v": 0.0028, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json", "k": "sensitivity.S2_uniform_inlet.delta_vs_L1"},
 {"v": 0.602, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json", "k": "ladder_pregregistered_fixed_umax.rows[0].tau"},
 {"v": 0.746, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json", "k": "ladder_pregregistered_fixed_umax.rows[2].tau"},
 {"v": 0.664, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json", "k": "ladder_pregregistered_fixed_umax.rows[1].tau"},
 {"v": 0.869, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json", "k": "artifact_decomposition.tau_bracket_at_L3_grid", "in_str": true},
 {"v": 0.52, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json", "k": "convention.geometry", "in_str": true},
 {"v": 0.49, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json", "k": "convention.geometry", "in_str": true},
 {"v": 1.01, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json", "k": "convention.geometry", "in_str": true},
 {"v": 1.9423, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json", "k": "convention.geometry", "in_str": true},
 {"v": 100.0, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json", "k": "convention.re_definition", "in_str": true},
 {"v": 1.942, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json", "k": "reference.primary.source", "in_str": true},
 {"v": 1.2, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json", "k": "reference.cross_checks", "in_str": true},
 {"v": 6.0, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json", "k": "reference.cross_checks", "in_str": true}]
-->
