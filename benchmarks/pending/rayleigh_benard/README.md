# Rayleigh-Bénard 对流 rayleigh_benard（线性起稳增长率 γ(Ra)）（pending）

**状态：❌ 未达标（W11-C 正式判决 honest FAIL，2026-10-08）——G1 相对门/G3 符号门过（64/128 两级全臂），G2 单调门败（E=1.224→1.006→3.101% 非单调，延长后 256 档 4.632%）+ G4 健康门败（ny256 两臂窗散布 2.09/2.55% 超限，N×1.5 延长后 3.10/3.98% 仍超=拟合质量条款 FAIL）；根因=近临界区格子算子有效 Ra_c 比连续谱上移，放大成 γ 系统性偏移。**

> 本目录有两层记录：**旧档**（2026-08-19 非正式，README 自相矛盾、result.json
> 无判决字段，已被 W11-C 预注册判为不可继承并封存）与 **W11-C 正式判决**
> （2026-10-08 闭环，staging 机器文件，见下）。

## 物理问题

水平板间温差驱动的自然对流线性起稳（Rayleigh-Bénard onset）：2D 双分布 BGK
（D2Q9 水力 τ=0.855 + D2Q5 温度 τ_T=1.0 → Pr=0.71 精确），Boussinesq 浮力
（Guo 力格式，对导热基态零力）；x 周期、上下无滑移等温（half 构造）。
观测量 = 温度扰动本征模符号投影幅 A_T(t) 的指数增长率 γ_sim，与冻结参考
γ_ref（自研 Chebyshev 谱特征值解算器，文献锚 Ra_c=1707.762 / a_c=3.1163，
对文献 7 位吻合）对比。Ra 臂 = {0.85, 0.95, 1.05, 1.15, 1.3}×Ra_c(a)
（臂 1.0 仅报告）；格点阶梯 ny ∈ {64, 128, 256}（fp64 + correct_mass 每步）。

## 计算结果现状（W11-C 正式档，2026-10-08）

**门禁结果**（verdict.json / verdict_ext.json）：

| 门 | 内容 | 结果 |
|----|------|------|
| G1 | 逐臂 \|γ_sim/γ_ref-1\|≤3%，≥2 个格点水平全臂过 | ✓（ny64、ny128 全臂；ny256 两臂败） |
| G2 | E(ny)=sqrt(mean rel²) 严格单调降 64→128→256 | ✗（0.012238→0.010061→0.031010；延长后 256 档 0.046322） |
| G3 | 符号门（亚临界 γ<0、超临界 γ>0，\|γ\|≥2×窗散布） | ✓（15/15 门臂全过；臂 1.0 三级 γ_sim 均为负，报告项） |
| G4 | 健康（质量/u_max/θmax/窗散布≤2%） | ✗（ny256 臂 0.95/1.05 窗散布超限） |

**逐臂 rel_pct（γ_sim vs γ_ref，%）**：

| ny \ frac | 0.85 | 0.95 | 1.05 | 1.15 | 1.3 |
|-----------|------|------|------|------|-----|
| 64  | +0.53 | +1.85 | -1.83 | -0.57 | -0.34 |
| 128 | +0.24 | +1.50 | -1.63 | -0.28 | -0.16 |
| 256 | +0.48 | **+4.62** | **-5.10** | -0.64 | -0.24 |
| 256（N×1.5 延长） | — | **+6.92** | **-7.66** | — | — |

ny256 失败臂明细：臂 0.95 γ_sim=-0.6048104900825649 vs γ_ref=-0.5781058046012082
（rel +4.619342215354205%，三窗散布 0.020941616526539135）；臂 1.05
γ_sim=0.5355700761516998 vs γ_ref=0.5643721122314042（rel -5.103376913119851%，
散布 0.025539138977476783）。按预注册 L61-63 条款各做 N×1.5 单次延长
（3013173 / 3086496 步）后散布反而 0.030955184045900284 / 0.0397985223664529、
rel 恶化到 +6.923402318641836 / -7.658546700837876 → 该两臂诚实 FAIL
（拟合质量条款，判据值不动）。**VERDICT: FAIL。**

**旧档（本目录 result.json，2026-08-19，非正式）**：仅记录临界行为定性捕捉
（Ra=1300 静止 / Ra=1e4、1e5 对流的 u_max 量级，字符串字段），无 Nu/γ 判据
数字；W11-C 预注册已裁定其不可继承。

## 不达标清单（严格标准：共性模块入口 + 直接观测量 ≤3% + ≥2 档网格单调收敛）

- 共性模块入口：✓（collide_bgk_force + pre_streaming_bounce_back + stream +
  correct_mass，库原语；温度壁 BC half 构造为 run 层编排，披露于 prereg）。
- 直接观测量 ≤3%：部分 ✗（15 门臂中 13 臂 ≤1.85%，但 ny256 臂 0.95/1.05
  为 +4.62/-5.10%，延长后 +6.92/-7.66%）。
- ≥2 档网格单调收敛（本案 G2 条款）：✗（E 非单调，256 档反而最大）。
- 附带：G4 拟合质量条款 ✗（两臂窗散布 >2%，延长不救）。

## 根因/诊断

- **近临界区格子算子 Ra_c 上移**：冻结的精确格子线性算子（staging
  `ref/lattice_predictions.json`）给出 Ra_c 格点漂移 ny64/128/256 =
  +0.145949715952165% / +0.08666696637815718% / +0.07291570026752847%。
  臂 Ra 按 own-solver 连续谱冻结，而 γ 在近临界区对 (Ra-Ra_c) 高度敏感
  （臂 0.95/1.05 距临界仅 ±5%），算子的 Ra_c 偏移被放大成 γ 的系统性偏差
  ——量级与 ny256 臂观测偏差吻合（算子预测臂 1.3 的 γ_pred=3.2093525920826314
  vs γ_ref=3.2099517995174547，与实测 γ_sim=3.2022410596403126 同向）。
- ny256 拟合散布超限与该系统偏移同源（信号本身偏离指数律的纯净形态）。
- 运行健康面良好：mass_drift≈0（correct_mass 后逐位 0 或 -2.2e-16）、
  符号门 15/15、三级 13/15 臂在 1% 附近——失败是**参考口径问题不是步进器 bug**
  （fp64 步进器≡算子闭合 0.07% 已在 Phase-0 验证）。

## 晋级路径

- **γ_ref 改用格子算子自有特征值重锁**：`lattice_predictions.json` 已含各级
  逐臂 γ_pred 机器值；以算子谱替代连续谱做参考（或等效地对臂 Ra 做算子级
  Ra_c 修正）后，写新 prereg 重跑三级正式档（owner 决策）。
- 预算参考：prereg §7 估正式档 ~8 GPU·h（fp64，含复跑裕量 ×2）；延长条款
  已消耗的 5M 步级档成本应计入复评。
- 达标即归档 PR 移 `pending/rayleigh_benard` → `verified/`（连同 staging 的
  18+2 份 f_*/ext_f_* 结果 JSON 与 prereg 收据）。

## 复现（W11-C 正式档，staging）

```
staging: /nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/
  formal/formal_run.py        # 三级×六臂批式正式运行器（fp64）
  formal/verify.py            # 判据管线（只读 JSON 与 ref_table.json）
  formal/verdict.json         # 基础判决（FAIL）
  formal/verdict_ext.json     # N×1.5 延长判决（FAIL）
  formal/results/f_*.json 18 件 + ext_f_*.json 2 件
  prereg/prereg.md            # 预注册（冻结门禁与窗协议）
```

本目录 run.py = 旧档脚本（Nu 口径 RB 变体），历史参考用。

<!-- PROVENANCE
[{"v": true, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/formal/verdict.json", "k": "gates.G1_rel3_at_least_2_levels"},
 {"v": false, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/formal/verdict.json", "k": "gates.G2_monotone_E"},
 {"v": true, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/formal/verdict.json", "k": "gates.G3_sign"},
 {"v": false, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/formal/verdict.json", "k": "gates.G4_health"},
 {"v": 0.012238098621827004, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/formal/verdict.json", "k": "gates.E_ny.64"},
 {"v": 0.010061048317178623, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/formal/verdict.json", "k": "gates.E_ny.128"},
 {"v": 0.031010378209105776, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/formal/verdict.json", "k": "gates.E_ny.256"},
 {"v": 0.04632196814652002, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/formal/verdict_ext.json", "k": "gates.E_ny.256"},
 {"v": 0.5308405973175168, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/formal/verdict.json", "k": "arm_rows[0].rel_pct"},
 {"v": 1.8483776932231688, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/formal/verdict.json", "k": "arm_rows[1].rel_pct"},
 {"v": -1.8302457725138277, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/formal/verdict.json", "k": "arm_rows[2].rel_pct"},
 {"v": -0.5683381267489573, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/formal/verdict.json", "k": "arm_rows[3].rel_pct"},
 {"v": -0.3427145214191074, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/formal/verdict.json", "k": "arm_rows[4].rel_pct"},
 {"v": 0.2367133179655756, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/formal/verdict.json", "k": "arm_rows[5].rel_pct"},
 {"v": 1.5006041269005177, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/formal/verdict.json", "k": "arm_rows[6].rel_pct"},
 {"v": -1.6267393683763953, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/formal/verdict.json", "k": "arm_rows[7].rel_pct"},
 {"v": -0.28316636894163416, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/formal/verdict.json", "k": "arm_rows[8].rel_pct"},
 {"v": -0.16408703889028153, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/formal/verdict.json", "k": "arm_rows[9].rel_pct"},
 {"v": 0.48411940142605303, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/formal/verdict.json", "k": "arm_rows[10].rel_pct"},
 {"v": 4.619342215354205, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/formal/verdict.json", "k": "arm_rows[11].rel_pct"},
 {"v": -5.103376913119851, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/formal/verdict.json", "k": "arm_rows[12].rel_pct"},
 {"v": -0.6382203840838652, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/formal/verdict.json", "k": "arm_rows[13].rel_pct"},
 {"v": -0.24021357200133542, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/formal/verdict.json", "k": "arm_rows[14].rel_pct"},
 {"v": -0.6048104900825649, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/formal/verdict.json", "k": "arm_rows[11].gamma_sim"},
 {"v": -0.5781058046012082, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/formal/verdict.json", "k": "arm_rows[11].gamma_ref"},
 {"v": 0.020941616526539135, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/formal/verdict.json", "k": "arm_rows[11].slope_spread_rel"},
 {"v": 0.5355700761516998, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/formal/verdict.json", "k": "arm_rows[12].gamma_sim"},
 {"v": 0.5643721122314042, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/formal/verdict.json", "k": "arm_rows[12].gamma_ref"},
 {"v": 0.025539138977476783, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/formal/verdict.json", "k": "arm_rows[12].slope_spread_rel"},
 {"v": 6.923402318641836, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/formal/verdict_ext.json", "k": "arm_rows[11].rel_pct"},
 {"v": 0.030955184045900284, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/formal/verdict_ext.json", "k": "arm_rows[11].slope_spread_rel"},
 {"v": 3013173, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/formal/verdict_ext.json", "k": "arm_rows[11].steps"},
 {"v": -7.658546700837876, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/formal/verdict_ext.json", "k": "arm_rows[12].rel_pct"},
 {"v": 0.0397985223664529, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/formal/verdict_ext.json", "k": "arm_rows[12].slope_spread_rel"},
 {"v": 3086496, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/formal/verdict_ext.json", "k": "arm_rows[12].steps"},
 {"v": 3.2022410596403126, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/formal/verdict.json", "k": "arm_rows[14].gamma_sim"},
 {"v": 0.00145949715952165, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/ref/lattice_predictions.json", "k": "levels[0].Ra_c_lattice_drift"},
 {"v": 0.0008666696637815718, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/ref/lattice_predictions.json", "k": "levels[1].Ra_c_lattice_drift"},
 {"v": 0.0007291570026752847, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/ref/lattice_predictions.json", "k": "levels[2].Ra_c_lattice_drift"},
 {"v": 3.2093525920826314, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/ref/lattice_predictions.json", "k": "levels[2].arms[5].gamma_pred"},
 {"v": 3.2099517995174547, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/ref/lattice_predictions.json", "k": "levels[2].arms[5].gamma_ref"},
 {"v": false, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/rayleigh_benard/result.json", "k": "verified"},
 {"v": 3.1163, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/ref/anchor_and_convergence.json", "k": "anchor.literature.a_c"},
 {"v": 1707.762, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/ref/anchor_and_convergence.json", "k": "anchor.literature.Ra_c"},
 {"v": 2.0941616526539135, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/formal/verdict.json", "k": "arm_rows[11].slope_spread_rel", "k2": "arm_rows[11].slope_spread_rel", "op": "scale100"},
 {"v": 2.5539138977476783, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/formal/verdict.json", "k": "arm_rows[12].slope_spread_rel", "k2": "arm_rows[12].slope_spread_rel", "op": "scale100"},
 {"v": 3.0955184045900284, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/formal/verdict_ext.json", "k": "arm_rows[11].slope_spread_rel", "k2": "arm_rows[11].slope_spread_rel", "op": "scale100"},
 {"v": 3.97985223664529, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/formal/verdict_ext.json", "k": "arm_rows[12].slope_spread_rel", "k2": "arm_rows[12].slope_spread_rel", "op": "scale100"}]
-->
