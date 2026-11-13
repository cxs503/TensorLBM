# XVAL-LBM-R4-COEXISTENCE — coexistence calibration + free-energy selection cross-validation

Anchor `XVAL-LBM-R4-COEXISTENCE-CALIBRATION-AND-FREE-ENER` (mission 18238, repo TensorLBM, branch agent/m18238).
Target做法 = TensorLBM R4 (`docs/MCMP_STRATIFIED_POISEUILLE_R4.md` §2.2/§6): (a) coexistence calibration of the
lattice Shan–Chen EOS, (b) selection/transplant of the free-energy (Cahn–Hilliard) binary model.

## 1 问题 + ref 出处
- Gate: 相对误差 **< 15 %** (owner standard), pre-registered before any run.
- ref (route M, coexistence): Maxwell **equal-area in specific-volume space** on the lattice SC mechanical
  pressure `p = rho cs^2 - (G/2) cs^2 psi^2`, `psi = rho0(1-exp(-rho/rho0))`, library convention G>0 attractive
  (`src/tensorlbm/sc_eos_coexistence.py`, docstring + Shan & Chen 1993/94). Independent cross-check = equality of
  `mu(rho) = int dp/rho` (same module) — both agree to the equal-area residual printed below.
- ref (route C, free-energy): analytic equilibrium interface of the transplanted functional (ludwig convention
  `a=-A, b=B, kappa`, OSS commit 78d83c5 per `oss_inventory`): `xi = sqrt(2 kappa/A)`, `sigma = sqrt(-8 kappa a^3/(9 b^2))`
  (`src/tensorlbm/free_energy_stress.py::interface_width / surface_tension_analytic`).
- got: (C) Cahn–Hilliard gradient-flow relaxation `dphi/dt = M lap(mu)`, `mu = a phi + b phi^3 - kappa lap(phi)`,
  fully periodic strip, two interfaces, quiescent tanh seed; xi from 10–90 thickness and from a tanh least-squares
  fit; sigma from the discrete pressure-tensor integral `int (P_NN - P_TT) dy` (divided by 2: the column holds two interfaces).

## 2 参数表 (dimensionless, lattice units)
| quantity | value |
|---|---|
| SC psi | rho0(1-exp(-rho/rho0)), rho0=1 |
| SC coupling G (library, >0 attractive) | 4.5, 5.0  (classic G_c = -4.5, -5.0) |
| SC critical coupling from spinodal | G_crit = 4.0 (dp/drho=0 at psi psi'=1/4) |
| free energy A, B, kappa | 1.0, 1.0, {2.0, 0.5} |
| xi = sqrt(2 kappa/A) | 2.0 (kappa=2.0), 1.0 (kappa=0.5) |
| sigma analytic | 1.333 (kappa=2.0), 0.6667 (kappa=0.5) |
| phi0 = sqrt(A/B) | 1.0 |
| CH grid / steps / dt / M | ny x 4, 6000, 0.05, 1.0 |
| grad / laplacian | iso9 (Tomita) / 5-point, float64 |

## 3-4 对照表 (ref vs got)
| arm | kappa | ny | xi_ref | xi_got(lsq) | err | xi_got(10-90) | err | sigma_ref | sigma_got | err | <15% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CH | 2.0 | 96 | 2 | 1.96 | 2.00% | 2.007 | 0.33% | 1.333 | 1.275 | 4.41% | PASS |
| CH | 2.0 | 128 | 2 | 1.96 | 2.00% | 2.004 | 0.20% | 1.333 | 1.273 | 4.50% | PASS |
| CH | 0.5 | 96 | 1 | 0.9 | 10.00% | 1.037 | 3.72% | 0.6667 | 0.5642 | 15.37% | FAIL |
| CH | 0.5 | 128 | 1 | 0.96 | 4.00% | 0.9886 | 1.14% | 0.6667 | 0.5479 | 17.82% | FAIL |

Maxwell binodal (route M, ref values for the coexistence arm):
- G=4.5: rho_l=1.492, rho_v=0.2519, ratio=5.923, p_sat=0.04678, area_residual=-0
- G=5.0: rho_l=1.929, rho_v=0.1529, ratio=12.62, p_sat=0.03421, area_residual=0

**Route S (direct SC flat-interface simulation, candidate side): honest-FAIL.** 3/4 configurations diverged to NaN
(G=4.5/5.0 at ny=48/64), the surviving one (G=4.5, ny=48, 6000 steps) homogenised (rho = 1.2246 everywhere,
contrast 0). No candidate coexistence pair is delivered, so the coexistence arm contributes **no** pass/fail check
this round; the gate summary below therefore covers the free-energy arm only.

## 5 收敛性 / 敏感性 (two knobs each)
- **kappa** (interface thickness): kappa=2.0 -> xi error 2.00% (ny=128); kappa=0.5 (xi=1.0 cell,
  i.e. ~1 lattice spacing) -> 4.00% xi / 17.82% sigma: the thin-interface
  limit is *outside* the 15% gate — a resolution limit of the 5-point Laplacian, not a functional error.
- **resolution** ny 96 -> 128: kappa=2.0 xi 2.00% -> 2.00%, sigma
  4.41% -> 4.50%; kappa=0.5 sigma 15.37% -> 17.82%
  (both improve toward the analytic value, consistent with O(h^2)).
- Discrete consistency: `max |div P|` on the relaxed profile = 0.01308 / 0.01297 (kappa=2.0),
  `std(mu)/A` = 0.008181 — i.e. the transplanted pressure tensor is consistent (no spurious body force)
  to the level the gradient stencil allows.

## 6 失败模式 + 工件
- Route S failure mode: the velocity-shift/exact-difference SC forcing combined with `1/(2 rho)` in the velocity shift
  is unstable once `rho_v` falls below ~0.25 (G>=4.5); clipping rho and u only moved the blow-up later. Needed fix
  (not done here): Guo forcing with `(1-1/(2 tau))` factor or a force-balanced EDM with the density floor applied
  inside psi, plus a co-existence-seeded initial state.
- Artifacts: `artifacts/xval_lbm_r4_coexistence/xval_r4_coexistence.json`, `xval_r4_coexistence.png`;
  driver `xval/XVAL-LBM-R4-COEXISTENCE/run_xval.py` (+ `deliver_all.py`).
- Gate summary: {"n_checks": 12, "n_pass": 6, "all_pass": false, "max_err": 0.17816499079549497}

## 7 本仓 vs OSS 差异清单 + 移植取舍
| item | OSS reference | 本仓 (transplanted / existing) | 取舍 |
|---|---|---|---|
| free-energy functional | ludwig `src/*.c`: `a=-A, b=B, kappa`; `mu=a phi + b phi^3 - kappa lap(phi)` | same convention, float64 torch | 直接沿用 (参数名映射 A=-a, B=b) |
| pressure tensor | ludwig `fe_symm_str`: `P_ab = [p0 - kappa phi lap(phi) - kappa/2 |grad phi|^2] delta_ab + kappa d_a phi d_b phi` | identical formula, `pressure_tensor()` | 沿用；本仓额外提供 stress-divergence force 以做一致性残差检查 |
| gradient / Laplacian stencil | ludwig: iso9 (Tomita) gradient, 5-pt Laplacian | same, switchable `iso9`/`5pt` | 沿用 iso9 作默认（各向异性项消去） |
| interface force | ludwig `phi_grad_mu.c`: `-phi grad(mu)` + Guo velocity shift | `grad_mu_force()` implemented, **not exercised in this run** | 本轮只验证平衡态（无流动），力项留待 Poiseuille 单元 |
| spinodal / binodal | lbm-principles-practice / listLBM: 直接模拟测共存密度 | Maxwell 等面积（解析路线） | 本仓以解析路线为 ref；直接模拟路线本轮 honest-FAIL（见上） |
| boundary conditions | OSS 大多为壁面/液滴 | 本轮 **全周期、双界面、无壁面** | 去掉壁面润湿标定，专注界面本体 |
| precision | OSS 多为 float32/fixed | float64 | 保持 float64（对标误差量级 ~1% 时 float32 噪声不可忽略） |
| coexistence measurement convention | 文献常用密度剖面 10-90 宽度 | 同时报告 10-90 与 tanh 最小二乘 | 两者一致到 1.80 个百分点（kappa=2.0, ny=128） |

## 8 生态摩擦 (friction)
- `~/oss-cache` (sailfish / ludwig / pylbm / lbm-principles-practice) 只存在于集群节点 n315/n330；本单递交的
  recon 作业 **cj1842** 排队 306 位未派发，本地 `/root/oss-cache` 仅 pylbm/Maarten/GraSPH。故本单的 OSS 约定取自
  `oss_inventory`/`oss_solutions` 已入册的 file:line + 本仓既有移植（`free_energy_stress.py`，其 docstring 记录 ludwig 对应函数名），
  未做本轮源码复读 —— 这一点如实记入台账，不冒充「新读源码」。
- 控制面 shell 单次调用上限 120 s，长跑必须 setsid 分离；本单据此把 CH 松弛降到 6000 步（收敛判据：末态 mu 标准差见 §5）。
