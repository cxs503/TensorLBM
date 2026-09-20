# TensorLBM 共性模块功能矩阵 Benchmark

> 思路：不逐个试错，而是按**共性模块功能矩阵（碰撞 × 湍流 × 边界 × 物理案例）**系统性组合排查。
> 每个组合跑一个物理案例，误差 ≤3%（真实模拟、无外推）才入库 verified/。

## 功能矩阵盘点（2026-08-18）

### 碰撞模型（Collision）
| 编号 | 模型 | 入口 | 格 |
|------|------|------|-----|
| C1 | BGK | solver.collide_bgk / collide_bgk3d | D2Q9/D3Q19 |
| C2 | MRT | collide_mrt / general_sim AUTO | D2Q9/D3Q19 |
| C3 | cumulant | collide_cumulant_d2q9/d3q19/d3q27 | 全 |
| C4 | cascaded | collide_cascaded_d3q19/d3q27 | D3Q19/D3Q27 |
| C5 | KBC/entropic | collide_kbc_d3q19 / entropic_kbc | D3Q19/D3Q27 |
| C6 | D3Q27 MRT | d3q27_collide | D3Q27 |

### 湍流模型（Turbulence/LES）
| 编号 | 模型 | 入口 |
|------|------|------|
| T0 | 无（DNS/层流） | — |
| T1 | Smagorinsky | collide_smagorinsky_bgk/mrt (2D/3D) |
| T2 | WALE | collide_wale_bgk/mrt (2D/3D) |
| T3 | Vreman | collide_vreman_bgk/mrt (2D/3D) |
| T4 | 动态 Smagorinsky | collide_dynamic_smagorinsky_* |
| T5 | DES/DDES | ddes.py / des_turbulence.py |
| T6 | RANS k-ε | rans_ke.py |

### 边界条件（Boundary）
| 编号 | 类型 | 入口 |
|------|------|------|
| B1 | Zou-He 速度入口 | zou_he_inlet_velocity(_3d) |
| B2 | Zou-He 压力出口 | zou_he_outlet_pressure(_3d) |
| B3 | far-field | far_field_bc_2d/3d |
| B4 | bounce-back（无滑移） | bounce_back_cells(_3d) |
| B5 | free-slip | free_slip_cells_3d / *_walls_3d |
| B6 | wall-function | wall_function_3d / wall_function_d3q27 |
| B7 | periodic | stream 模运算 / engine 自动 |

### 几何/物理案例（Problem）
| 编号 | 案例 | 参考基准 | 判别力 |
|------|------|---------|--------|
| P1 | Taylor-Green 2D 涡衰减 | 解析 γ=2νk² | **碰撞/耗散** |
| P2 | Poiseuille 2D 管流 | 解析抛物线 | **边界/粘性** |
| P3 | 方腔流 Re=100 | Ghia 1982 | **对流/角点** |
| P4 | 球 Re=100 阻力 | Cd=1.087 | **阻力/壁面** |
| P5 | 圆柱 Re=100 | Cd≈1.35 | **分离流/St** |

## 组合策略（同一问题只保留最优配置，精力放在不同问题上）

**原则**：每个物理问题只做 1 个 benchmark（用该问题下最优的碰撞/湍流/边界组合），
不扫描组合矩阵。矩阵的价值是**盘点能力**（知道有什么），不是穷举测试。

| 案例 | 采用配置 | 参考基准 | 状态 |
|------|---------|---------|------|
| P1 Taylor-Green | C1 BGK（已验证） | 解析 γ=2νk² | ✅ 已入库 -0.035% |
| P2 Poiseuille | C1 BGK + Zou-He | 解析抛物线 | ✅ 已入库 0.18% |
| P3 方腔流 Re=100 | C2 MRT + BB 壁面 | Ghia 1982 | 子 agent 进行中（GPU1） |
| P4 球 Re=100 | GeneralSimEngine（MRT+BB） | Cd=1.087 | 子 agent 进行中 |
| P5 圆柱 Re=100 | MRT+far-field | Cd≈1.35 | 子 agent 进行中 |
| P6 SUBOFF Re=1000 | GeneralSimEngine | Ct≈0.004 | 子 agent 完成，审阅中 |
| P7 NACA 翼型 | airfoil_benchmark | Cl 文献 | 子 agent 中断，待重派 |
| P8 Blasius 平板 | turbulent_channel | f'(η) 解析 | 子 agent 中断，待重派 |
| P9 空化气泡 | cavitation（需先修 EOS 缺口） | RP 理论 | 缺口分析完成 |
| P10 后向台阶 | backward_facing_step（库入口+τ-matched 梯子） | Erturk 2008 2.878（Armaly 1983 几何 ER=1.9423） | ❌ −4.33%（最细档）未达标：参考簇散布 ~6%>3% 互斥 + BB 滑移伪差 B=42±7（2026-09-20 定源记录） |
| P11 环形 Taylor-Couette | C1 BGK + 旋转壁 Ladd BB（rotating_cylinder） | 解析 u=Ar+B/r + 力矩 M=4πνB（R_eff 剖面反演） | ✅ 已入库 2026-09-19，0.154%（最细档） |
| P12 Stokes 第二问题 | C1 BGK + Zou-He 振荡盖 + specular 远场 | 解析 U·e^{−ky}·cos(ωt−ky) | ✅ 已入库 2026-09-19，0.008%（最细档） |
| P13 Womersley 振荡管流 | C1 BGK + Zou-He 驱动 + pre-stream 半程 BB | cosh 复数解析 | ✅ 已入库 2026-09-19，0.117%（最细档） |
| P14 start-up Poiseuille | C1 BGK + Zou-He + pre-stream 半程 BB | 奇 n 级数 | ✅ 已入库 2026-09-19，0.165%（最细档） |
| P15 热自然对流方腔 | D2Q9 流场 + thermal D2Q5（壁 BC 修复 PR #300） | de Vahl Davis 1983（Nu/u_max/v_max） | ✅ 已入库 2026-09-20，Nu −0.83%/−1.41%、u −0.91%/−0.73%、v −1.12%/−0.89%（128/256 档；N=64 粗档如实判败披露） |
| P16 液滴振荡（严格标准复活） | SCMP SC94 collide + stream，R=128/160/224 | Rayleigh ω²=6σ/((ρl+ρv)R³)，σ_i 逐档自测 | ✅ 重新入库 2026-09-20，原始 ω_d −2.005%（最细档，零修正直接比） |
| P17 线性声学平面波 | D2Q9 BGK 线性化（声学扰动叠加平衡态） | 连续 c_s 与 νk² 衰减（自推离散理论三链闭环） | ✅ 已入库 2026-09-20，波速 +0.013% / 衰减 +0.080%（最细档） |

**新问题方向（未覆盖）**：多相/自由表面（Laplace/溃坝）、声学、RANS 通道、
D3Q27 高精度、AMR 网格收敛、壁面函数高 Re。

**Wave-1 问题类拓宽（2026-09-19）**：旋转流（P11）与受迫非定常解析族（P12-P14）
已入库；浮力驱动（thermal 腔）被库 thermal 壁 BC 缺陷阻塞、界面失稳（RT）在
SCMP 参数面内不可行（详见 TODO.md B38/B39）。Wave-2 候选：声学平面波/偶极子、
Taylor-Aris 分散、矩形管道 Shah-London 级数 + 环隙 Poiseuille、NACA Cl 重启、
后向台阶细化。

## 判定标准（不变）

- 真实模拟（无外推），误差 ≤3% 才入库 verified/
- 共性模块路径：GeneralSimEngine / 库 solver / 物理 run_* 入口（禁手写 collide-stream）
- 每个组合一个 result.json + README.md

### 编译路由标准（2026-08-19 增补）

verified/ 案例（本标准设立时点的全部 15 个）的整步步进链（collide → BC → stream → BC）统一经
`benchmarks/compile_route.py` 路由——它是 `tensorlbm.compile_utils`
（`validate_compile_mode` + `compile_step`）在 benchmarks 侧的唯一适配器：

- 统一 CLI 开关 `--compile-mode {eager,default,max-autotune-no-cudagraphs}`，
  默认 `default`（编译）；`eager` 供 A/B 对拍。
- 步序号与每 N 步监测（残差、稳态判据、NaN 守卫）留在编译域外的 eager 驱动环。
- cudagraph 类模式由共享白名单拒绝（与 LBM 反馈环结构性冲突）。
- 三模式实测矩阵（15 案例误差 / eager 加速比 / 冷编译开销，RTX 5090）见
  `/nfs/wangxi/triton_bench_20260819/bench_compile_route/FINAL_MATRIX.md`；
  入库标准不变：误差 ≤3%。
- 注：main 后续新增的 `cylinder/re200` 与 `sod_shock_tube` 尚未按本标准接入，
  待后续 PR 以同一矩阵流程补测后接入。
