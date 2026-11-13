# XVAL: 自由能压强张量体力（ludwig `FE_FORCE_METHOD_STRESS_DIVERGENCE` vs TensorLBM `-φ∇μ`）

锚点：`XVAL-LBM-FREE-ENERGY-PRESSURE-TENSOR-BODY-FORCE`（m17763，TensorLBM）
判据：相对误差 < 15%（owner 精度标准）

## ① 问题 + ref 出处

二元流体自由能 LBM 的毛细体力有两种等价写法：化学势梯度式 `F = -φ∇μ`（TensorLBM
`multiphase.free_energy_step` 现行实现）与压强张量散度式 `F = -∇·P`（ludwig 的
`FE_FORCE_METHOD_STRESS_DIVERGENCE`）。二者在连续极限恒等，离散实现不同。本单把
OSS 做法按其实现取证后移植进 TensorLBM，与现行做法在同一离散场、同一网格上对照，
并以解析表面张力/Laplace 曲率平衡为独立基准。

**ref 出处（零手抄，取证链）**：生态内 OSS 快照 `/tmp/ludwig_tar/ludwig-master`
（ludwig-cf/ludwig，loongarch64 构建产物，含 `.o` DWARF 与可执行文件；源码文本不在快照内）。
因此做法**从编译产物反解**，逐条有函数名/行号/常量证据：

| 事实 | 证据 |
|---|---|
| 体力方案枚举 `FE_FORCE_METHOD_{NO_FORCE, PHI_GRADMU, PHI_GRADMU_CORRECTION, STRESS_DIVERGENCE, RELAXATION_SYMM, RELAXATION_ANTI}` | `fe_force_method.o` 符号 + 字符串 |
| 压强张量实现 `fe_symm_str` / `fe_symm_str_v`，局部变量 `a,b,kappa,phi,delsq,grad,p0,d` | `symmetric.o` DWARF（decl_line 334/372）+ 反汇编 |
| `mu = a φ + b φ³ − κ ∇²φ` | `fe_symm_mu` 反汇编（三处 fmul + fmadd） |
| `p0 = 0.5 a φ² + 0.75 b φ⁴` | `fe_symm_str_v` @line 404，常量 `.LC3=0.5, .LC4=0.75` |
| `d = p0 − κ φ ∇²φ − 0.5 κ |∇φ|²`；`P_ab = d δ_ab + κ ∂_aφ ∂_bφ` | 同上 @line 405/410 |
| `σ = sqrt(−8 κ a³ / (9 b²))` | `fe_symm_interfacial_tension` + 常量 `.LC15=−8, .LC16=9` |
| 散度式：先按站点填 `P_ab`（`pth_stress_set`），再差分取散度（`pth_kernel*`） | `phi_force_stress.o`：`pth_kernel/pth_stress_set/pth_stress` |

## ② 方法与实现

新增 `src/tensorlbm/free_energy_stress.py`（2D 周期，torch，无新求解器）：
`chemical_potential`、`bulk_pressure`、`pressure_tensor`、`stress_divergence_force`
（`F=-∇·P`，中心差分）、`grad_mu_force`（`F=-φ∇μ`）、`surface_tension_analytic`、
`planar_profile`、`droplet_profile`；梯度可选 `5pt` 与 `iso9`（Tomita 各向同性 9 点，
ludwig 流体梯度默认类）。实验：`experiments/xval_ludwig_fe_stress/run_xval.py`
（结果 `results.json`），图 `make_figures.py`。

**差异清单（本仓 vs OSS）与移植取舍**

| 维度 | ludwig | 本仓（移植） | 取舍理由 |
|---|---|---|---|
| 体力形式 | `STRESS_DIVERGENCE`：站点张量后差分散度 | 同（新增）；同时保留 `-φ∇μ` | 逐项对齐便于对照 |
| 张量公式 | `p0 − κφ∇²φ − ½κ|∇φ|²` + `κ∂φ∂φ` | 完全一致 | 恒等式已反解验证 |
| 梯度/拉普拉斯 | 可选 5pt / Tomita 9pt / 3D 27pt，可配 | 5pt 与 iso9；拉普拉斯固定 5pt | FreeEnergy 求解器为 2D |
| 体力入碰撞 | 力先存 `fe->force`，由碰撞/forcing 方案消费（Guo 类，另有 RELAXATION_SYMM/ANTI 变体） | `free_energy_step` 用平衡态速度移位 | 未改求解器，避免改动既有验证基线；列入后续 |
| 边界 | 固体/润湿/外场化学势插座齐全 | 周期 | 本单只做体积分对照 |
| 维度 | 2D/3D | 2D | 3D 仅换模板 |

## ③ 参数表（无量纲，lattice 单位）

| 参数 | 值 | 说明 |
|---|---|---|
| A, B | 1, 1 | `a=-A`，φ₀=√(A/B)=1 |
| κ | 2, 8, 32 | ξ=√(2κ/A)=2, 4, 8 |
| ρ₀ / τ / ν | 不参与 | 静态场对照，不积分时间 |
| 网格 | 256×256（平面）/ 128×128、256×256（液滴） | h=1 |
| 判据窗口 | ±8ξ（平面）；正半轴 r∈(0,R+6ξ) | 排除周期边界伪界面 |

## ④ 对照表（ref vs got，误差打勾叉；<15% 为 ✓）

参考值：解析 σ=(2√2/3)√(κA³)/B、二维曲率平衡 ΔP=σ/R。
"got" = 由 ludwig 压强张量离散求和得到的 σ / 曲率平衡；`-φ∇μ` 为 TensorLBM 现行做法。

| 量 | 解析 ref | ludwig 张量 got | 误差 | 判 |
|---|---|---|---|---|
| σ, ξ=2 (κ=2) | 1.3333 | 1.2521 | 6.09% | ✓ |
| σ, ξ=4 (κ=8) | 2.6667 | 2.6233 | 1.63% | ✓ |
| σ, ξ=8 (κ=32) | 5.3333 | 5.3112 | 0.41% | ✓ |
| σ 曲面 (R=32, ξ=4) | 2.6667 | 2.6228 | 1.64% | ✓ |
| σ 曲面 (R=64, ξ=8) | 5.3333 | 5.3110 | 0.42% | ✓ |
| 曲率平衡 ∫(P_NN−P_TT)/r dr vs σ_flat/R (R=32) | 0.0820 | 0.0824 | 0.53% | ✓ |
| 曲率平衡 (R=64) | 0.0830 | 0.0834 | 0.52% | ✓ |
| 平衡态体力残差 max|F|/(σ/ξ²)，ξ=2 | 0 | 5.30e-2 | 5.3% | ✓ |
| 同上，ξ=4 | 0 | 2.09e-2 | 2.1% | ✓ |
| 同上，ξ=8 | 0 | 5.79e-3 | 0.58% | ✓ |
| 两种写法的力场差 max|F_s−F_g|/max|F_g|（ξ=4） | 0（连续恒等） | 1.04 | 104% | ✗ |

最后一行是**诚实 FAIL 项**：在平衡剖面上两式都只剩 O(h²) 截断残差（≪ σ/ξ²），
两者的**相对差**因此被放大到 ~100%（互为同量级残差，非物理差异）。结论必须按绝对尺度读：
两式都满足"平衡态体力→0"到 <6%（ξ≥2）/<1%（ξ≥8）。

## ⑤ 收敛性 / 参数敏感性（≥2 点）

- 界面分辨收敛（κ=2→8→32，即 ξ=2→4→8）：σ 相对误差 6.09% → 1.63% → 0.41%，
  观测阶 ≈2（2.4 与 2.0），符合二阶中心差分的 O(h²/ξ²)。
- 体力残差：5.30e-2 → 2.09e-2 → 5.79e-3，同样二阶收敛；`-φ∇μ` 臂 5.02e-2 → 1.59e-2 → 4.73e-3，
  两臂量级相同（张量臂略大 ~10–30%）。
- 梯度模板 5pt vs iso9：平面场（一维剖面）下两模板**代数恒等**（4/12+2/12=1/2），
  结果逐位相同 → 该对照对一维剖面不构成独立变量；曲面/多维才可分辨（本单未展开，列为局限）。
- 曲率敏感性（R/ξ=8 保持）：R=32 与 R=64 的 σ_curved 误差 1.64%/0.42%，随 ξ 减小而减小，
  与平界面同阶。

## ⑥ 失败模式与工件 / 局限

- 失败模式1（已修）：周期盒内一维 tanh 剖面在周期边界处产生**未分辨伪界面**（单个格点跳变），
  会污染 σ 与力场对照 → 用 ±8ξ 窗口排除；液滴对照只用正半轴（一个界面）。
- 失败模式2（诚实记录）：对称自由能两相体相压力相同（p0(±φ₀)=A²/4B），
  `P_NN` 越界面为常数，**经典 ΔP=σ/R 不能从该张量直接读出**（实测 Pnn_centre−Pnn_far≈7e-6）；
  需密度耦合的总压力（LBM 的 cs²ρ 项）才能观测 → 本单改以曲率平衡恒等式作替代判据。
- 失败模式3：ludwig 侧无可用算例输入/运行入口（快照仅含 `target/a.out` 的 kerneltest 与
  不相关 `.o`），故 ref 采用"实现取证 + 解析基准"，而非跑 OSS 算例跑分；属证据等级限制，如实声明。
- 工件：`src/tensorlbm/free_energy_stress.py`、`experiments/xval_ludwig_fe_stress/{run_xval.py,
  make_figures.py,results.json,figures/xval_fe_stress.png}`、`tests/test_free_energy_stress.py`。

## 判据结论

σ（平面与曲面）与曲率平衡全部 <15%（最差 6.09%，均在 ξ≥4 时 <2%）→ **XVAL 通过**。
唯一未达标项为"两式力场相对差"，已证明其为 O(h²) 截断残差层面的人工指标（绝对量 <6% σ/ξ²），
不作为物理精度判据；非粉饰：若要求力场逐点 <15%，本单判 honest-FAIL 并在 ⑤ 给出原因与收敛补救路径。
