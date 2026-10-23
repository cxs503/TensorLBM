# NACA0012 翼型 α=5° Re=1000 平均升力（naca_0012）（pending）

**状态：❌ 未达标(根因类，参考受限)——4 档 Cl 对锁定参考 0.132 误差 +85~99% ≫ 3% 且不单调；参考侧图文自不一致（图 0.132 vs 文字锚 0.26，差 ~2 倍）已定源，模拟侧 α=5° 周期振荡的物理来源未定源**

（Wave-3 W3-D 判决记录；本 README 为案例导引，完整判决见 `result.json`、参考提取全记录 `reference_extract.md`。）

## 物理问题

NACA0012 二维翼型定常外流，α=5°、Re=1000（弦长 Re），观测量 = 平均升力系数 Cl（附带 Cd）。域 14c×8c 通道型（上下无滑移壁，低堵塞），u_in=0.06（Ma=0.10392304845413262），MRT（τ=3ν+0.5），Ladd 动量交换测力（post-stream pre-bounce-back，wet-node），阶梯边界无曲面处理。

参考（`reference_extract.md` 全记录 + `result.json` reference 节）：

- **锁定（判据用）Cl_ref = 0.132**：主源 Kurtulus 2015（DOI 10.1260/1756-8293.7.3.301）论文无数值表，唯一数据 = Figure 4a 数字化（轴梯机器验证、双 panel 互证，提取细节见 `reference_extract.md`）；
- **披露列 Cl = 0.26**：Kurtulus 自身文字锚（升力线斜率锚线性外推）与 Di Iilio 2020 自家 HLBM 所隐含的常规约定值——图文自不一致，Figure 4 的 Cl 轴疑似少标 2×（0.26/0.132）；
- 两源差 ~85% >> 3% → 预注册已判**参考受限**，判据列用 0.132、披露列用 0.26，无参考选购。

## 计算结果现状

| 档 | 网格 | Cl（均值） | err @0.132 | err @0.26 | Cd | Cl 振荡峰峰（末 1/4 窗） |
|----|------|-----------|-----------|-----------|-----|------------------------|
| c60 | 840×480 | 0.24459388852119446 | +85.29840039484428% | +5.93% | 0.13865303993225098 | 0.23172518610954285 |
| c90 | 1260×720 | 0.2628062069416046 | +99.09561131939742% | +1.08% | 0.13555090129375458 | 0.24649539589881897 |
| c120 | 1680×960 | 0.2587105631828308 | +95.99285089608394% | +0.50% | 0.13474631309509277 | 0.24360638856887817 |
| c150 | 2100×1200 | 0.26283320784568787 | +99.11606654976353% | +1.09% | 0.13398689031600952 | 0.2527572512626648 |

- 两档边缘：c60→c90 +7.44594173244522%、c90→c120 −1.5584273318490736%、c120→c150 +1.5935354985654688% → **Cl 不单调**（顶三档在 ±1.6% 带内振荡收敛）。
- **Cd 严格单调收敛**：0.13865303993225098 → 0.13398689031600952（收敛值与 Kurtulus 文字 Cd(5°)≈0.13 量级一致）。
- 敏感性/探针：c90 域高 8c→12c → Cl=0.2571578621864319；旧库默认域（3.33c×1.33c 高堵塞）探针 Cl=0.4215288758277893（旧档配置不可考的佐证）。

## 不达标清单

（战役严格标准：共性模块入口 + 直接观测量对参考 ≤3% + ≥2 档网格单调收敛；修正/还原变换不算直接模拟）

1. **精度门 ✗（判据列）**：对锁定参考 0.132 全档 +85~99%（all_pass_3pct=false）。
2. **单调条款 ✗**：Cl 两档边缘符号交替（two_level_edge_pct = +7.45/−1.56/+1.59）。
3. **参考受限 ✗**：cl_ref 0.132 vs 披露列 0.26 散布 ~85%（reference_limited=true）——若改按 0.26，顶三档 0.50–1.09% 但 c60 +5.93% 且仍不单调；判决为 FAIL（verdict=FAIL）。
4. **稳态性异常**：Kurtulus 称 α<8° 振幅为零，实测全档存在周期性 Cl 振荡（峰峰 ~0.23-0.25），steady=false。

## 根因/诊断

- **参考侧（已定源）**：主源图文自不一致——Figure 4a 数字化值 vs 论文自身文字锚隐含值约 2 倍差（0.26/0.132）；Di Iilio 2020 转绘的 Kurtulus 方点与本次数字化逐点一致（证实提取正确），其自家 HLBM 站文字锚侧 → Figure 4 的 Cl 轴疑似少标 2×。
- **模拟侧（部分定源）**：rule-5 诊断（logs/diag_correct_mass.json，c60 双跑）排除质量校正致振——去掉 correct_mass 后振荡 std 0.060805750716073345 vs 有校正 0.06075789569213366（几乎相同），Cl 均值 0.24486931937712209 vs 0.2445938962477225（差 ~0.1%）；重标定因子偏差在 1e-7 量级。振荡为该配置（通道域+阶梯边界+MRT 低 τ）的真实流动尺度现象，**物理来源未定源**（候选：域声学模态/阶梯边界激发的分离泡不稳定性；参考 Di Iilio 曲面边界 HLBM 在更低攻角报告稳态）。
- Ma≈0.104，可压缩误差 O(Ma²) 量级（披露项）。

## 晋级路径（转 verified 的条件）

1. **参考重锁（控制方决策，零计算成本）**：0.132（图）vs 0.26（文字锚+独立源）二选一或另寻带数值表的 OA 一手源；当前两锚互斥，任何模拟不可能同时满足。
2. 若采 0.26 族：需修复单调与振荡问题——**换曲面边界表示**（BFL/插值边界，参考球系 w8a 的 BFL 路线）+ 域声学诊断（加大域/吸波层），目标 c90 以上档单调进 3%。
3. 预算参考：现有 4 档机器记录 wall_s 74.5–138.4 s/档（`result.json` levels.*.wall_s，GPU 5090）——同规模重跑成本低，主要不确定性在边界表示改造的工程量。
4. 达标后由归档 PR 将本目录 `pending/naca_0012` → `verified/naca_0012`（参考受限/图文自不一致披露随迁）。

<!-- PROVENANCE
[{"v": 0.132, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/naca_0012/result.json", "k": "reference.cl_ref_locked"},
 {"v": 0.26, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/naca_0012/result.json", "k": "reference.cl_ref_disclosure_text_implied"},
 {"v": 0.24459388852119446, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/naca_0012/result.json", "k": "levels.c60.cl_mean"},
 {"v": 85.29840039484428, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/naca_0012/result.json", "k": "levels.c60.cl_err_vs_locked_pct"},
 {"v": 5.925427491848288, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/naca_0012/result.json", "k": "levels.c60.cl_err_vs_text_pct"},
 {"v": 0.13865303993225098, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/naca_0012/result.json", "k": "levels.c60.cd_mean"},
 {"v": 0.23172518610954285, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/naca_0012/result.json", "k": "levels.c60.cl_amp_pp_lastquarter"},
 {"v": 74.50010251998901, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/naca_0012/result.json", "k": "levels.c60.wall_s"},
 {"v": 0.2628062069416046, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/naca_0012/result.json", "k": "levels.c90.cl_mean"},
 {"v": 99.09561131939742, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/naca_0012/result.json", "k": "levels.c90.cl_err_vs_locked_pct"},
 {"v": 1.0793103621556175, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/naca_0012/result.json", "k": "levels.c90.cl_err_vs_text_pct"},
 {"v": 0.13555090129375458, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/naca_0012/result.json", "k": "levels.c90.cd_mean"},
 {"v": 0.24649539589881897, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/naca_0012/result.json", "k": "levels.c90.cl_amp_pp_lastquarter"},
 {"v": 0.2587105631828308, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/naca_0012/result.json", "k": "levels.c120.cl_mean"},
 {"v": 95.99285089608394, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/naca_0012/result.json", "k": "levels.c120.cl_err_vs_locked_pct"},
 {"v": 0.4959372373727686, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/naca_0012/result.json", "k": "levels.c120.cl_err_vs_text_pct"},
 {"v": 0.13474631309509277, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/naca_0012/result.json", "k": "levels.c120.cd_mean"},
 {"v": 0.24360638856887817, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/naca_0012/result.json", "k": "levels.c120.cl_amp_pp_lastquarter"},
 {"v": 0.26283320784568787, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/naca_0012/result.json", "k": "levels.c150.cl_mean"},
 {"v": 99.11606654976353, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/naca_0012/result.json", "k": "levels.c150.cl_err_vs_locked_pct"},
 {"v": 1.0896953252645605, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/naca_0012/result.json", "k": "levels.c150.cl_err_vs_text_pct"},
 {"v": 0.13398689031600952, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/naca_0012/result.json", "k": "levels.c150.cd_mean"},
 {"v": 0.2527572512626648, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/naca_0012/result.json", "k": "levels.c150.cl_amp_pp_lastquarter"},
 {"v": 7.44594173244522, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/naca_0012/result.json", "k": "reference.two_level_edge_pct[0]"},
 {"v": -1.5584273318490736, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/naca_0012/result.json", "k": "reference.two_level_edge_pct[1]"},
 {"v": 1.5935354985654688, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/naca_0012/result.json", "k": "reference.two_level_edge_pct[2]"},
 {"v": 0.2571578621864319, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/naca_0012/result.json", "k": "levels.c90_ny12c.cl_mean"},
 {"v": 0.4215288758277893, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/naca_0012/result.json", "k": "levels.c60_legacy_ny1p33c.cl_mean"},
 {"v": 0.10392304845413262, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/naca_0012/result.json", "k": "levels.c60.mach"},
 {"v": 1000.0, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/naca_0012/result.json", "k": "levels.c60.re_chord"},
 {"v": 5.0, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/naca_0012/result.json", "k": "levels.c60.alpha_deg"},
 {"v": 138.43121695518494, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/naca_0012/result.json", "k": "levels.c90_ny12c.wall_s"},
 {"v": 0.06075789569213366, "f": "/nfs/wangxi/runs/bm_widen_w3_20260920/naca_cl/logs/diag_correct_mass.json", "k": "with.cl_std"},
 {"v": 0.060805750716073345, "f": "/nfs/wangxi/runs/bm_widen_w3_20260920/naca_cl/logs/diag_correct_mass.json", "k": "without.cl_std"},
 {"v": 0.2445938962477225, "f": "/nfs/wangxi/runs/bm_widen_w3_20260920/naca_cl/logs/diag_correct_mass.json", "k": "with.cl_mean"},
 {"v": 0.24486931937712209, "f": "/nfs/wangxi/runs/bm_widen_w3_20260920/naca_cl/logs/diag_correct_mass.json", "k": "without.cl_mean"}]
-->
