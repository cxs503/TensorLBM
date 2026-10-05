# W9-B: Hadamard–Rybczynski 液滴终端速度（蠕变流）— FAIL（模块能力）

## 一句话结论

**FAIL·模块能力**：库内任何 3D 多相配置都无法表示带黏度比 λ≠1 的浮力黏性液滴——每个可用模块的 λ 都锁死等于密度比（单 τ），预注册的 λ 阶梯退化（λ=1 ⇔ 零密度对比 ⇔ 零浮力 ⇔ U_t=0 不可测）。形式化 Phase-2 阶梯从未启动；结局由后续相场开发弧补齐（Wave-10 M1，PR #323 已合，verified/laplace_cac_phasefield）。

## 基准与参考（预注册锁定）

- 终端速度：U_t = (2/3) a^2 g (rho_o - rho_i) (1+lambda) / (mu_o (2+3lambda)), lambda = mu_i/mu_o。
- 格上 λ-ρ 锁形式：U_t = (2/3) a^2 g (1 - r^2) / (nu (2 + 3r)), r = rho_i/rho_o。
- 来源：drag form Pigeonneau 1998 (thesis, literature aggregate) + terminal-velocity form Clift, Grace & Weber 2005 (via Wikipedia Hadamard-Rybczynski equation); force balance closes the two; limit checks lambda->inf = Stokes (2/9), lambda->0 = clean bubble (1/3, 1.5x rigid)。
- 手算例（r=0.5, a=12, ν=1/6, g=8.1e-06）机器复算：U_t 相对偏差 2.22e-16（门 1e-12）；prereg printed Re=0.14396685714285715, exact 144*U_t=0.14396708571428576 (rel dev 1.59e-06); U_t check unaffected。

## 模块排除（E1–E6 动态证据 + 15 案包络）

| 证据 | 模块 | 关键机器数字 |
|------|------|--------------|
| E1 | Color-Gradient 3D | 总密度比 1.029→1.0024（100→2000 步，声学坍缩；初始 0.5 对比案） |
| E2 | SC-MCMP 3D D3Q19, G=+2.5 | NaN@50 步 |
| E3 | SC-MCMP 3D D3Q19, G=−2.5 | 压碎反转 r1_ctr 1.221→0.596（50→250 步，核内组分 2 占 0.332）|
| E4 | Allen-Cahn 3D | 少数相液滴完全溶解（体素 2492→0，trace 25→1000 步）|
| E5 | MCMP D3Q27, G=+2.5 | NaN@50 步 |
| E6 | MCMP D3Q27, G=−2.5 | 同压碎反转 r1_ctr 1.221→0.587（50→250 步）|

15 案对比×G 包络（probe/p0b_envelope.json (contrasts 1.05-2.1 x G in [-2.5, +1.5])）：**1 NaN + 14 mixed/反转，持续保持对比度的液滴案 0**。逐案表在 result.json。

静态排除分支（依据既有归档，无需运行）：

- SCMP 3D: lambda fixed ~12.3 by coexistence; spurious max|u| 0.137-0.140 >> U_t ~ 1e-3 (verified/laplace_droplet archive)
- SC-MCMP unequal tau: W5-A structural bias 26.8/33.2% anti-convergent + NaN (pending/two_phase_poiseuille)
- phasefield free-energy package: lambda == 1 (Boussinesq), no buoyancy contrast possible
- free-surface family: one-fluid + flag, no viscous droplet interior; multiphase.py: 2D only, same locks

## 设计约束（等效原理）

- equivalence principle：periodic domain + per-mass gravity on all components = free fall, no relative motion; walls (momentum sink) are mandatory for buoyancy benchmarks — the W4-A-validated tank composition (channel|tank union + full-way bounce-back) was probed innocent (probe/p0b_diag.json: G=0 + walls stable, spurious decays 1.9e-2 -> 5e-4)

## 库缺陷清单（G1–G5）

| id | 缺陷 | 现状 |
|----|------|------|
| G1 | multiphase3d G_12 sign bug: docstring + StaticDroplet3DConfig validation demand G_12 > 0 for phase separation, implementation makes +G attractive (mixing; +2.5 NaNs) — segregation needs G < 0; validated default G=0.9 yields a MIXED state so the library's own sigma_eff measurement is vacuous; same wording bug in multiphase3d_d3q27 | fixed by Wave-10 track M2 (MCMP3D G1), archive pending |
| G2 | no module separates viscosity ratio from density ratio with acceptable accuracy (headline) | open (Wave-10 M1 phasefield rebuild addresses the long-term path) |
| G3 | no 3D two-phase module sustains a total-density contrast droplet (kills every buoyant-drop/bubble benchmark class in 3D): CG collapses acoustically (EOS), SC-MCMP inverts by pressure crush at every probed G/contrast | open |
| G4 | allen_cahn_lbm is an unvalidated orphan (zero tests/benchmarks): curvature-less sigma force, no solid-skip in collision, non-conserving phase equation, hardcoded mobility/tau_phi; fix path = Fakhari/Geier full scheme | open |
| G5 | bounce_back_cells_3d hardcodes the 19-direction OPPOSITE table (fails on 27-q fields; boundaries_d3q27.bounce_back_cells_27 exists — composition footgun) | open |

## 结局

- 当时结论：HR needs a from-scratch phase-field development。
- 落地：delivered by the revival arc: phasefield development R2 -> Wave-10 track M1, merged as PR #323 (laplace_cac_phasefield in verified/)。

## 交付物（本目录）

- 规范证据：out/evidence.json (generator out/evidence.py)。
- 探针：
  - probe/p0_static.json (CG collapse + first MCMP NaN scan)
  - probe/p0b_scan.json (G>=1 asym NaN matrix)
  - probe/p0b_diag.json (walls innocent)
  - probe/p0b_gsweep.json (G window)
  - probe/p0b_envelope.json (15-case contrast x G envelope)
  - probe/p_final_diligence.json (AC W/sigma sensitivity + D27 inversion)
- 完整性注记：probe/p1_ac.{log,json} were corrupted by an accidental concurrent double launch; the AC conclusions are independently re-established in out/evidence.json E4 and p_final_diligence.json (clean single-process runs)
- `result.json`（机器切片，全部数字来自 out/evidence.json 与 probe/*.json，手算例复算门 1e-12 PASS）；`prereg.md`/`NOTES.md`（冻结预注册与笔记）。

## 判定

- benchmark_overall = **FAIL（模块能力）**：基准不可运行，非数值未达标；E1–E6 + 包络 + 静态分支为判决证据，G1–G5 为库侧行动清单。
- 该 FAIL 的修复路径已走通（Wave-10 M1 相场重建，#323）。
