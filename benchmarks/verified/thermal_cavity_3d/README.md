# W9-A: 3D 差热立方腔自然对流（Pr=0.71）— thermal_cavity_3d

## 概述

- 问题：立方腔差热自然对流——左壁等温 T_hot=1、右壁等温 T_cold=0、
  四壁 no-slip、其余面绝热；D3Q19 BGK（速度）+ D3Q7（温度）双分布，
- 判决：**VERIFIED CANDIDATE (all gates pass; promotion owner decision)**——三条 Ra 阶梯全过门（finest |err| = 0.55% / 1.47% / 2.37% ≤ 3%，|err| 随加密严格单调下降）。晋级 verified/ 为 owner 决策。
- 入库名自评：**thermal_cavity_3d**。门判内容是 Pr=0.71 的 3D 立方腔（对 3D 文献簇中位），高 Pr（100/1000）只是能力披露节、不进判据，故名不含 hp；与既有 thermal_cavity（2D DVD）、cavity_3d_full（等温）区分。

## 配方（判据档冻结）

- 全部物理来自 tensorlbm 库入口，本条目 run.py 零本地物理实现（铁律 grep `def (collide|stream|equilibrium|bounce|zou_he|far_field)` 零命中）：
  - 速度壁：全反弹 `boundaries3d.bounce_back_cells_3d`，streaming 之后施加（bb=post，壁在节点上）——与 #303 修复的节点等温温度壁同位配对；
  - 温度壁：`thermal3d.apply_temperature_boundaries_3d`（#300/#303 修复版温度链）；温度碰撞前壁节点宏观 u 置零（#300 2D 同款）；
  - 浮力：`thermal_common.apply_buoyancy_3d` 碰撞后原始增量（无 Guo 1−1/(2τ) 因子），**只加流体节点**（force_mask，见披露 2）；
  - 碰撞/迁移：`solver3d.collide_bgk3d` / `stream3d`；初场：线性导热 profile T(x)=1−x/L、ρ=1、u=0。
- 参数标定：τ_f=0.6、ν=(τ_f−1/2)/3、D3Q7 电导率自标定 α=(τ_T−1/2)/4（W=[1/4, 1/8×6]）、τ_T=4α+1/2、 gβ=Ra·ν·α/L³、L=nx−1（等温面在节点上）、ΔT=1。
- Nu 口径：整壁 grad1（壁节点−首内邻单向一阶差分，含角点）面积平均，Nu=(Nu_hot+Nu_cold)/2；判据窗 = 记录历史末 20% 均值（采样 500 步）。直接观测量，无外推。

## 参考值（预注册锁定）

3D 立方腔 Nu 无单一权威解；主参考 = 六源文献簇中位（Jiménez-Islas et al. 2014 Rev. Mex. Ing. Quim. 13(1):259-277 Table 3 转录，控制器独立核对）：

| Ra | 簇中位 Nu | Tric | Lo | Ravnik | Brahim | W&S | Bessonov |
|----|-----------|------|----|--------|--------|-----|----------|
| ra1e3 | 1.07065 | 1.07 | 1.07 | 1.0713 | 1.0712 | — | — |
| ra1e4 | 2.05552 | 2.0542 | 2.054 | 2.0591 | 2.056 | 2.0624 | 2.055 |
| ra1e5 | 4.34110 | 4.337 | 4.335 | 4.357 | 4.3432 | 4.3665 | 4.339 |

## 结果

判据：per Ra ladder: finest-grid |Nu err| vs cluster median <= 3% AND |err| monotonically decreasing with refinement (>= 3 grids per Ra); direct observable, no extrapolation (Richardson vetoed by controller ruling)

| Ra | N | steps | Nu_win | err vs 簇中位 | 窗漂移 | Ma |
|----|---|-------|--------|-------------|--------|----|
| ra1e3 | 32 | 40000 | 1.05762 | -1.22% | +0.0001% | 0.009 |
| ra1e3 | 48 | 90000 | 1.06243 | -0.77% | -0.0001% | 0.006 |
| ra1e3 | 64 | 160000 | 1.06474 | -0.55% | +0.0002% | 0.005 |
| ra1e4 | 32 | 60000 | 1.91430 | -6.87% | -0.0000% | 0.050 |
| ra1e4 | 48 | 120000 | 1.96733 | -4.29% | +0.0000% | 0.033 |
| ra1e4 | 64 | 200000 | 1.99172 | -3.10% | +0.0001% | 0.024 |
| ra1e4 | 96 | 360000 | 2.01456 | -1.99% | +0.0001% | 0.016 |
| ra1e4 | 128 | 640000 | 2.02534 | -1.47% | +0.0002% | 0.012 |
| ra1e5 | 64 | 200000 | 4.10924 | -5.34% | +0.0000% | 0.092 |
| ra1e5 | 96 | 360000 | 4.19841 | -3.29% | -0.0001% | 0.061 |
| ra1e5 | 128 | 800000 | 4.23821 | -2.37% | -0.0001% | 0.046 |

- ra1e3：err -1.22 → -0.77 → -0.55%，|err| 严格单调下降 = True，finest -0.55%（N=64）。
- ra1e4：err -6.87 → -4.29 → -3.10 → -1.99 → -1.47%，|err| 严格单调下降 = True，finest -1.47%（N=128）。
- ra1e5：err -5.34 → -3.29 → -2.37%，|err| 严格单调下降 = True，finest -2.37%（N=128）。
- 稳态/自检（判据 11 案最大值，逐案值在 result.json）：|窗漂移| 0.0002%、|sumT 漂移| 3.0e-04、|热/冷壁 Nu 失衡| 0.0065%、Ma 0.092、NaN 案数 0。
- Nu 可独立重算：out/ 每案存 `wall_fields_end` 四整壁原始温度场，按 `nu_convention.recompute_hint` 重算须复现 history 末行（对阶梯案重合差 ≤1e-3%，result.json 内逐案记录）。

### τ 不变性（浮力原始增量无 Guo 因子的证据）

| 档 | τ=0.6 Nu | τ=0.9 Nu | diff |
|----|----------|----------|------|
| t09fm_ra1e4_n32 | 1.91430 | 1.90864 | -0.296% |
| t09fm_ra1e4_n64 | 1.99172 | 1.99044 | -0.064% |

τ 0.6→0.9 Nu 变化 ≤0.30%：浮力以原始增量注入不引入 τ 阶误差（若带 Guo 1−1/(2τ) 因子而碰撞不匹配，会出现 ~O(1/τ) 系统漂移）。

### fp32 vs fp64（纯导热对，库温度壁臂）

- nu_hot(fp32) = 1.8134567737579346，nu_hot(fp64) = 1.813457338051353，相对差 3.112e-07（**7 位有效数字一致，非逐位同**，见披露 1）。

### 能力档：高 Pr（稳定性披露，非判据）

3D 高 Pr 腔无公开解表可锁（Turan et al. 只有 correlation），以下为长跑稳定性披露：

| 档 | Pr | dtype | steps | Nu_win | 末行 nu_hot/nu_cold | u_max | 窗漂移 | NaN |
|----|----|-------|-------|--------|--------------------|-------|--------|-----|
| cap_pr100 | 100 | fp32 | 5600000 | 1.93903 | 1.93936 / 1.93851 | 2.10e-04 | -0.0180% | None |
| cap_pr1000_fp64 | 1000 | fp64 | 56000000 | 1.92998 | 1.93008 / 1.92988 | 2.08e-05 | +0.0000% | None |

- 两档全程无 NaN；窗漂移 0.0180% 以内。温度扩散时间尺度随 Pr 增长（τ_T=4ν/Pr+0.5→0.5），故步数阶梯 5.6M→56M。

## 强制披露

1. fp32 vs fp64: agreement is 7 significant digits (rel dev 3.112e-07 on the conduction-pair nu_hot), not bitwise identity
2. force_mask is a driver-layer flag (library untouched, src_patched/ empty): the buoyancy increment is multiplied by the fluid mask; without it the bb=post stack develops a slow linear instability (NaN at 17k/55k/131k steps for N=32/48/64) from forcing bounced solid nodes
3. bb=post (full-way bounce-back, wall on the node) is a BC recipe choice pairing with the #303 node-located isothermal temperature walls; the half-way pre-bounce alternative leaves a wall-location mismatch and only first-order convergence (archived in staging)
4. tau=0.6 for all gated ladders (tau=0.9 reaches Ma=0.22 at Ra=1e5, compressibility ~-1.7%); tau invariance on the fixed stack documented in tau_invariance
5. capability tiers have no cluster reference (no published high-Pr 3D cavity solution table exists; Turan et al. provide correlations only) and are stability disclosures, not gates

## 运行

```bash
python run.py --ra 1e4 --n 64 --steps 200000            # 单案（判据档配方）
python run.py --ra 1e3 --n 32 --steps 40000 --device cpu  # CPU 冒烟
```

输出 `out/probe_<tag>.json`（tag 默认 fm_ra{RA}_n{N}）：history + `wall_fields_end` 原语 + config。判据档完整步数见结果表。

## 交付物（本目录）

- `run.py` — 基准入口（固定判据配方，--ra/--n/--steps 选案）
- `result.json` — 机器切片汇总（判据数含来源档案路径；对控制器锚点表核对 PASS：PASS vs ctrl/w9a_verify_out_20260930.txt (13 ladder Nu/err + 3 monotone + 2 tau pairs + 2 capability tiers + fp32/fp64 pair + config echo)）
- `out/` — 17 份冻结机器档案：判据阶梯 11（probe_fm_*）+ τ=0.9 对照 2（probe_t09fm_*）+ 高 Pr 能力档 2（probe_cap_*）+ 导热 fp32/fp64 对 2（probe_conv*）

## 参考

- Jiménez-Islas, H., et al. (2014). Rev. Mex. Ing. Quim. 13(1), 259-277, Table 3（六源簇：Tric/Lo/Ravnik/Brahim/Wakashima-Saitoh/Bessonov）。

## 判定

- 判据门：PASS all three Ra ladders（finest err -0.55% / -1.47% / -2.37%，全部 |err|≤3% 且单调）。
- 真实模拟（无外推/无修正）：是（BC 与浮力方案选择全部披露）。
- `result.json` verdict = VERIFIED CANDIDATE (all gates pass; promotion owner decision)。
