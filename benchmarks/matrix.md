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
| P18 矩形管 Poiseuille（Shah-London） | D3Q19 BGK + Zou-He 入口/出口 + bounce_back_cells_3d 半程（compile_route） | 自推双奇正弦级数；Po 交叉 Shah-London 方 56.9083 / 2:1 62.1922 | ✅ 已入库 2026-09-20，eff 0.0336%（方 W128）/ 0.0231%（2:1 W128） |
| P19 Taylor-Aris 剪切分散 | D2Q9 流（pre-stream 半程 BB + 周期体力）+ D2Q5 被动标量（内联解缠零通量壁，披露） | Deff/D = 1+Pe²/210（Taylor 1953 / Aris 1956 缝隙） | ❌ 容差 6/6 过（0.05-0.26%）但单调 0/3——负 O(1/H²)+正地板变号结构，定源记录留 pending |
| P20 NACA0012 Cl（α=5° Re=1000） | D2Q9 MRT + apply_simple_channel_boundaries + BB + MEM 力（airfoil_benchmark mask） | Kurtulus 2015 Fig4a 数字化 0.132±0.015（锁定）；文字锚 0.26 披露 | ❌ FAIL+参考受限：+85~99% 不单调（对 0.26 亦 0.5-5.9% 不单调）；参考图文自不一致 2.07×/1.3×；Cd 严格单调 0.1387→0.1340 |
| P21 环隙 Poiseuille（a/R=0.5） | D3Q19 BGK + Zou-He + bounce_back_cells_3d（楼梯曲壁） | 自推精确解 u(r)（3 参数 c0+c1lnr−c2r² 族 + Φ 流量式） | ❌ not_verified 复合：eff 1.16→0.13→0.05% 全过单调，但 δi spread 0.0937>0.05 违格无关条款（内壁曲率真实 O(1/R) 漂移；平壁 duct 对照格无关） |
| P22 两相 Poiseuille（SC-MCMP） | porous_media 入口（collide_sc_two_component + stream + BB + apply_wall_wettability_sc，w4b_lib 编排） | 库自备 _two_phase_poiseuille_analytical（名义黏度比 M=2） | ❌ FAIL 反收敛：26.82→33.17%（阈值 9-11 倍）；分量自速度平衡结构性偏差（PR #305 xfail 锁证） |
| P23 毛管侵入 Washburn（SC-MCMP） | 同上（固 seam+气储层+水汇，K=1 定尺寸） | Washburn 两段 ODE（σ 不可测→仅黏性项）；β=0.5 | ❌ FAIL：β 0.5954/0.5159、绝对 +689%/+949%、W128@164 NaN（稳定包络 W≤64）；2β 估计器缺陷已修 PR #305 |
| P24 渗透率（周期圆柱方阵 Stokes） | D2Q9 BGK τ=1.0 + 周期胞 + 掩码体力 _apply_body_force_2d + BB（compile_route） | Sangani–Acrivos 1982 f 表（Basilisk 逐字交叉），k_ref=d²/((1−φ)f) | ✅ 已入库 2026-09-21：fs=10 协议 φ=0.3 最细 1.38%、φ=0.5 五档 73.87→…→0.0484%；fs=1 链并列披露（fp32 注入舍入伪差注记） |
| P25 3D 差热立方腔自然对流（Pr=0.71，Ra=1e3/1e4/1e5） | D3Q19 BGK + D3Q7 温度（τ_T=4α+1/2 自标定）+ 全反弹 BB（bb=post）+ apply_buoyancy_3d 原始增量（force_mask 流体掩码） | 六源 3D 簇中位（Jiménez-Islas 2014 Table 3：1.07065/2.05552/4.34110） | ✅ verified 候选 verified/thermal_cavity_3d/：finest −0.55/−1.47/−2.37% 全 ≤3% 且 \|err\| 严格单调；Pr=100/1000 能力档无 NaN（披露非门）；fp32≡fp64 7 位有效（非逐位）+ force_mask 驱动层披露 |
| P26 H–R 液滴终端速度（λ 阶梯） | 库 3D 多相全族 probe（CG3D/SC-MCMP19/AC3D/MCMP27，96³） | Hadamard–Rybczynski U_t（Pigeonneau 1998 + Clift–Grace–Weber 2005，力平衡闭环） | ❌ FAIL·模块能力 pending/hadamard_rybczynski_droplet/：λ 锁死=ρ 比（单 τ），λ=1 ⇔ U_t=0；E1–E6 + 15 案包络 0 sustained；G1–G5 库缺陷清单；结局 PR #323（相场重建已入库） |
| P27 纤维横向渗透率（方阵/六角圆柱阵列 Stokes） | D2Q9 BGK τ=1 + 周期胞 + 掩码体力 _apply_body_force_2d + BB（compile_route；sq 单圆柱 / hex W×√3W 双位） | sq=Sangani–Acrivos 1982 精确 f 表；hex=Gebart 1992 闭式（精确 hex 表付费墙，8 通道搜寻） | ❌ FAIL（诚实）pending/permeability_fibrous/：hex=参考侧偏差 +3.9%/+2.4%（阶梯分解 D1）；sq0.40 量化振荡破单调（gap-96 扩展 −0.6475%）；方阵 0.30/0.50/0.60 三档 verified 候选（最细 0.22/0.24/0.88%）；R3 参考自产门 FAIL（vf0.6 差 0.052pp） |
| P28 两层 Poiseuille（颜色梯度 per-color τ） | D2Q9 双色 + src/tensorlbm/color_gradient2d.py（per-color τ + recoloration β=0.9 + 一阶色梯度力）+ solver.stream + bounce_back_cells | 两层 Stokes 闭式解（名义黏度比 1/10；界面=上色层首行节点半格位） | ✅ verified/poiseuille_two_phase_cg/：relL2 0.211/0.155%（等黏度）、1.092/0.560%（μ 比 10）双族随加密严格单调、全 ≤3%；双色质量漂移 ≤5.3e-14/步/色 |


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

#### 编译不可用设备的自动 eager 回退（2026-09-29 增补）

`mode='default'` 在 **CUDA** 上是真编译（生产路径，不回退）。在 **Hygon/SDAA**
上 `torch.compile` 走 `teco_inductor`（Torch-SDAA **3.1.1a3**），其后端对整步
LBM 链仍不完整：`compile_route._enable_sdaa_inductor()` 已修 L1–L4 四处后端
缺陷（device codegen 注册 / `Reduction` 陈旧绑定 / `TecoScheduling._sizes` /
`load_count` 契约），但上游仍有 L5（teco `EXPAND` 回退路径 `reduction_shape`
越界）——属已安装包内部实现，不再继续 hack。

为此 `route_step`（编译模式）不再直接抛错中断 benchmark，而是返回
`_CompileWithEagerFallback`：**首次**调用试编译版，捕获
`InductorError/AssertionError/IndexError/RuntimeError` 后打印醒目 banner
（含异常与回退原因），**其后每步走 eager**（与 `--compile-mode eager` 逐位一致，
保证 suboff_re1000 / square_cylinder 等入库用例在 SDAA 上可复现）。判定结果经
`compile_status_of(step)` 读出并写入 `result.json` 的 `compile_mode_effective`
（`compiled` / `eager_fallback` / `eager`）——静默回退的运行与真编译运行由此可区分。
CUDA 上该封装是 no-op（首次即成功并保持编译）。
