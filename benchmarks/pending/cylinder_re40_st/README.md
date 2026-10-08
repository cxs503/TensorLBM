# 圆柱定常绕流 Re=20/40（cylinder_re40_st）（pending）

**状态：❌ 未达标(根因类)——W11-A 自由流复活档 formal 阶梯 honest FAIL（六档 err 5.43–6.06% 全部 >3%），根因已定源为上游域截断（up=10D 入口 Dirichlet 自由流钳制）；单变量阶梯证明 up 扩到 40D 即入门（D40 实测 +2.236%）**

目录内含两代档案：2026-08-19 旧档（通道域、无效参考）与 2026-10-07/08 W11-A 复活档（预注册 + formal 阶梯 + 地板诊断，暂存 `/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/`）。

## 物理问题

2D 圆柱定常绕流（亚脱落区，Re < Re_c≈47），名义 Re 阶梯 {20, 40}（Re=30 因无 OA 一手参考弃档）。观测量 = 定常阻力系数 Cd = F_x/(0.5·ρ·U²·D_eff)。求解器复用 verified/cylinder 家族：D2Q9 + collide_mrt + far_field_bc_2d（入口/两侧自由流 Dirichlet、出口零梯度、障碍 bounce-back）+ Ladd 动量交换（post-stream pre-bounce-back）+ 下游 sponge。

参考双锁（`ref/ref_lock.json`，Fornberg 1980 JFM 98:819 主锁 + Gautier 2013 谱方法交叉）：

- Re=40：**1.498**（±0.0005）；交叉 Gautier 1.49 / Dennis-in-Fornberg 1.494，主锁 vs 交叉差 ≤0.54%；17 行文献簇散布 9.15% → 单一主锁条款；
- Re=20：**2.0001**（±0.0002）；交叉 Dennis-in-Fornberg 1.998，差 0.11% → 双锁成立；
- 第二腿：自研极坐标 FD 求解器独立验证（diag/fd4_*.json，Aitken 外推极限与主锁一致、直跑档亦达有效数字位，见 ref_lock own_solver 节）。

R_eff 口径：area-equivalent（R_area=√(N_solid/π)），逐档重锁 ν_lb=u_lb·D_eff/Re（G3 机器链核验 tau 相对偏差全档 0.0）。

## 计算结果现状

**旧档（2026-08-19，已判无效）**：FeatFlow/Schäfer-Turek 型 CHANNEL 域，Re=40，参考为双点对数插值、无文献背书、不确定度 ±5%（result.json ref_investigation 字段，字符串记载）——实测 Cd 4.0374（D40 τ065）/3.9435（D80 τ065）/4.1872（D40 τ08）/3.9525（D80 τ08），verified=false，判定"不可靠需独立 NS 参考或用户定夺"。该代档只作历史记录。

**W11-A formal 阶梯（2026-10-07/08，自由流域 40D×40D，up=10D，u_lb=0.05，Ma=0.0866）**：

| Re | 档 | τ | Cd | Cd_ref | err | 步数 |
|----|----|----|----|--------|-----|------|
| 20 | D40 | 0.8000433194722565 | 2.1212576639754612 | 2.0001 | +6.057580319757072% | 48000 |
| 20 | D80 | 1.0999075887722611 | 2.1172753059375715 | 2.0001 | +5.858472373259893% | 96000 |
| 20 | D160 | 1.6999643990086333 | 2.1176905168536173 | 2.0001 | +5.8792318810868105% | 192000 |
| 40 | D40 | 0.6500216597361282 | 1.5852098888020207 | 1.498 | +5.821754926703648% | 48000 |
| 40 | D80 | 0.7999537943861306 | 1.5801115167974609 | 1.498 | +5.481409666052128% | 96000 |
| 40 | D160 | 1.0999821995043166 | 1.5793775112093078 | 1.498 | +5.432410628124695% | 192000 |

全部六档 stop_reason=plateau（60 t_c 下限、末 20 t_c 均值估计器）。

## 不达标清单

（战役严格标准：共性模块入口 + 直接观测量对参考 ≤3% + ≥2 档网格单调收敛；修正/还原变换不算直接模拟）

1. **G1 精度门 ✗**：re20 三档 g1_all_within_3pct=false、re40 三档同（最好档 +5.432410628124695% > 3%）。
2. **G2 单调收敛**：re40 ✓（5.82→5.48→5.43 严格降，g2_strict_monotone=true）；re20 ✗（6.06→5.86→5.88 回升，g2_nonincreasing=false，D80→D160 边沿 +0.021pp）。
3. **G3 重锁链 ✓**：两 Re 档 g3_relock_chain_ok=true。
4. **判决**：re20/re40 verdict 均 **FAIL**（honest FAIL，不改门不调参）；旧档（channel/插值参考）不具证据资格。

## 根因/诊断

已定源（`diag/floor_20261008/`，2026-10-08，单变量阶梯、固定 D40/Re40/τ、掩膜逐位相同 n_solid=1257）：

| 探针 | up | LX | H | Cd | err |
|------|----|----|---|----|-----|
| 基线（=formal D40） | 10D | 40D | 20D | 1.5852098888020207 | +5.821754926703648% |
| E-B | 20D | 50D | 20D | 1.5427210274817453 | +2.9853823419055736% |
| E-A | 20D | 60D | 20D | 1.5435162658547352 | +3.038469015669909% |
| E-C | 10D | 40D | 40D | 1.5839477439189122 | +5.737499594052897% |
| E-D | 40D | 70D | 20D | 1.5314928033831259 | **+2.2358346717707533%** |

- 分解：上游 10D→20D 贡献 −2.8364pp（主因）；侧距 20D→40D 仅 −0.0843pp、下游 30D→40D 影响同样可忽略（两档 err 仅在第 3 位小数分开）——**流向效应几乎全部来自上游入口平面的 Dirichlet 自由流钳制**（压制圆柱前势流减速）。
- 与球系 W7-B 的入口钳制定源（err≈E0+C·(a/x)³）同族机制。
- 诊断档明确 diagnostic_only（不进判定门），formal/ 不动、判决维持。

## 晋级路径（转 verified 的条件）

1. **W11-A2 扩上游重跑 formal 阶梯**：域形 up≥40D / LX=70D（E-D 已证 D40 实测 +2.2358346717707533% 入 3% 门），D∈{40,80,160} × Re∈{20,40} 全六档重跑 + G1/G2/G3 门重判。
2. 预算（由机器档外推，估算）：E-D（D40，nx=2800×ny=1600，48000 步）wall_s=608.6558876037598；D160@LX70 域胞数为 64×（11200×6400×192000）→ 单档名义 ≈11 GPU·h，六档全阶梯 + 重跑余量 ≈25–35 GPU·h（红线 48 内）。
3. 注意 re20 G2 边沿（D80→D160 +0.021pp）在扩域后需复查单调条款。
4. 达标后由归档 PR 将本目录 `pending/cylinder_re40_st` → `verified/cylinder_re40_st`（旧 channel 档以披露条款随迁或废弃）。

<!-- PROVENANCE
[{"v": 4.0374, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/cylinder_re40_st/result.json", "k": "measured.D40_tau065.cd"},
 {"v": 3.9435, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/cylinder_re40_st/result.json", "k": "measured.D80_tau065.cd"},
 {"v": 4.1872, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/cylinder_re40_st/result.json", "k": "measured.D40_tau08.cd"},
 {"v": 3.9525, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/cylinder_re40_st/result.json", "k": "measured.D80_tau08.cd"},
 {"v": 2.0001, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/ref/ref_lock.json", "k": "locked.re20.primary.value"},
 {"v": 0.0002, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/ref/ref_lock.json", "k": "locked.re20.primary.unc"},
 {"v": 1.998, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/ref/ref_lock.json", "k": "locked.re20.cross[0].value"},
 {"v": 0.10510510510511575, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/ref/ref_lock.json", "k": "locked.re20.agreement_pct.fornberg_vs_dennis"},
 {"v": 1.498, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/ref/ref_lock.json", "k": "locked.re40.primary.value"},
 {"v": 0.0005, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/ref/ref_lock.json", "k": "locked.re40.primary.unc"},
 {"v": 1.49, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/ref/ref_lock.json", "k": "locked.re40.cross[0].value"},
 {"v": 1.494, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/ref/ref_lock.json", "k": "locked.re40.cross[1].value"},
 {"v": 0.5369127516778528, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/ref/ref_lock.json", "k": "locked.re40.agreement_pct.fornberg_vs_gautier"},
 {"v": 9.15032679738563, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/ref/ref_lock.json", "k": "locked.re40.agreement_pct.cluster17_spread_pct"},
 {"v": 2.1212576639754612, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/formal/ladder_summary.json", "k": "re20.rows[0].cd"},
 {"v": 6.057580319757072, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/formal/ladder_summary.json", "k": "re20.rows[0].err_pct"},
 {"v": 0.8000433194722565, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/formal/ladder_summary.json", "k": "re20.rows[0].tau"},
 {"v": 48000, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/formal/ladder_summary.json", "k": "re20.rows[0].steps"},
 {"v": 2.1172753059375715, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/formal/ladder_summary.json", "k": "re20.rows[1].cd"},
 {"v": 5.858472373259893, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/formal/ladder_summary.json", "k": "re20.rows[1].err_pct"},
 {"v": 1.0999075887722611, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/formal/ladder_summary.json", "k": "re20.rows[1].tau"},
 {"v": 96000, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/formal/ladder_summary.json", "k": "re20.rows[1].steps"},
 {"v": 2.1176905168536173, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/formal/ladder_summary.json", "k": "re20.rows[2].cd"},
 {"v": 5.8792318810868105, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/formal/ladder_summary.json", "k": "re20.rows[2].err_pct"},
 {"v": 1.6999643990086333, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/formal/ladder_summary.json", "k": "re20.rows[2].tau"},
 {"v": 192000, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/formal/ladder_summary.json", "k": "re20.rows[2].steps"},
 {"v": 0.02075950782691738, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/formal/ladder_summary.json", "k": "re20.err_diffs_pct[1]"},
 {"v": 2.0001, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/formal/ladder_summary.json", "k": "re20.cd_ref_primary"},
 {"v": 1.5852098888020207, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/formal/ladder_summary.json", "k": "re40.rows[0].cd"},
 {"v": 5.821754926703648, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/formal/ladder_summary.json", "k": "re40.rows[0].err_pct"},
 {"v": 0.6500216597361282, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/formal/ladder_summary.json", "k": "re40.rows[0].tau"},
 {"v": 48000, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/formal/ladder_summary.json", "k": "re40.rows[0].steps"},
 {"v": 1.5801115167974609, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/formal/ladder_summary.json", "k": "re40.rows[1].cd"},
 {"v": 5.481409666052128, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/formal/ladder_summary.json", "k": "re40.rows[1].err_pct"},
 {"v": 0.7999537943861306, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/formal/ladder_summary.json", "k": "re40.rows[1].tau"},
 {"v": 96000, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/formal/ladder_summary.json", "k": "re40.rows[1].steps"},
 {"v": 1.5793775112093078, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/formal/ladder_summary.json", "k": "re40.rows[2].cd"},
 {"v": 5.432410628124695, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/formal/ladder_summary.json", "k": "re40.rows[2].err_pct"},
 {"v": 1.0999821995043166, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/formal/ladder_summary.json", "k": "re40.rows[2].tau"},
 {"v": 192000, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/formal/ladder_summary.json", "k": "re40.rows[2].steps"},
 {"v": 1.498, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/formal/ladder_summary.json", "k": "re40.cd_ref_primary"},
 {"v": 1.5427210274817453, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/diag/floor_20261008/summary.json", "k": "single_variable_ladder[1].cd"},
 {"v": 2.9853823419055736, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/diag/floor_20261008/summary.json", "k": "single_variable_ladder[1].err_pct"},
 {"v": 20.0, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/diag/floor_20261008/summary.json", "k": "single_variable_ladder[1].up_d"},
 {"v": 1.5435162658547352, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/diag/floor_20261008/summary.json", "k": "single_variable_ladder[2].cd"},
 {"v": 3.038469015669909, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/diag/floor_20261008/summary.json", "k": "single_variable_ladder[2].err_pct"},
 {"v": 1.5839477439189122, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/diag/floor_20261008/summary.json", "k": "single_variable_ladder[3].cd"},
 {"v": 5.737499594052897, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/diag/floor_20261008/summary.json", "k": "single_variable_ladder[3].err_pct"},
 {"v": 1.5314928033831259, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/diag/floor_20261008/summary.json", "k": "single_variable_ladder[4].cd"},
 {"v": 2.2358346717707533, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/diag/floor_20261008/summary.json", "k": "single_variable_ladder[4].err_pct"},
 {"v": 40.0, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/diag/floor_20261008/summary.json", "k": "single_variable_ladder[4].up_d"},
 {"v": 70.0, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/diag/floor_20261008/summary.json", "k": "single_variable_ladder[4].lx_d"},
 {"v": -2.8363725847980747, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/diag/floor_20261008/summary.json", "k": "single_variable_ladder[1].delta_vs_baseline_pp"},
 {"v": -0.08425533265075114, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/diag/floor_20261008/summary.json", "k": "single_variable_ladder[3].delta_vs_baseline_pp"},
 {"v": 608.6558876037598, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/diag/floor_20261008/E-D_up40_lx70_h20.json", "k": "wall_s"},
 {"v": 2800, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/diag/floor_20261008/E-D_up40_lx70_h20.json", "k": "nx"},
 {"v": 1600, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/diag/floor_20261008/E-D_up40_lx70_h20.json", "k": "ny"},
 {"v": 48000, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/diag/floor_20261008/E-D_up40_lx70_h20.json", "k": "steps"},
 {"v": 1257, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/formal/re40_D40.json", "k": "n_solid"},
 {"v": 0.08660254037844387, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/formal/re40_D40.json", "k": "ma"},
 {"v": 0.05, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/formal/re40_D40.json", "k": "u_lb"},
 {"v": 40.0, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/formal/re40_D40.json", "k": "lx_d"},
 {"v": 10.0, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/formal/re40_D40.json", "k": "up_d"},
 {"v": 20.0, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/formal/re40_D40.json", "k": "h_d"}]
-->
