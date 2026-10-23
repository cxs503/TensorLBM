# X1 跨模型交叉验证电池 prereg（冻结版 v1）

轨道: X 突破计划 X1（跨模型交叉验证电池，≤10 GPU·h 上限）。
上游档案全部只读: bm_mp_20261001/{m1_cac_fix, m2_mcmp, m4_pseudopotential}、
bm_mp_20261003/{t2_taylor, t3_coalescence}、bm_mp_20261004/c1_color_gradient、
worktrees mp_w10 @ 1eae1c3b / mp_w11 @ a84745e56f。

## 0. 电池地位（预注册）

本电池是**归因/披露仪器，不是判决仪器**: 无 PASS/FAIL 总门，输出 =
每物理类 × 每模型族的闭合质量表 + 一页归因表，供控制器/owner 做
"哪类基准交给哪个模型族"的分配决策。跨家族 **σ 数值不可比**（格子物理、
密度比、σ 来源不同）——明令禁止跨模型比较 σ 绝对值；可比量 =
无量纲闭合质量（σ(R) 平坦度、Laplace 线性残差、µ 比闭合 relL2、
伪电流 u_max、质量/动量健康）。

## 1. 参考锁（先锁后跑; 判据脚本从机器源重提并断言一致后才跑）

| 家族 | 引擎树 | 运行时断言 |
|---|---|---|
| CAC 相场 | mp_w11 cac_lbm | md5 ccc671ee4738d9266967e74f1fd8924b |
| CAC-B2（µ 律修复版） | m1_cac_fix src_patched（mp_w10 基底） | m1 sha256 制式（同 b2_poiseuille.py 头部） |
| CG 颜色梯度 | c1 src_patched color_gradient2d | md5 81203d6d6f00104f4853a6db81815fe0 |
| SC 伪势 | m4 src_patched（mp_w10 基底） | m4 code_sha256 制式 |
| MCMP | m2 src_patched（mp_w10 基底, formal 同款 import 树） | m2 assert tensorlbm_file 前缀 |

冻结机器源（judgment 脚本逐字段重提+断言，本文引用值仅作核对锚）:

| 量 | 机器源 |
|---|---|
| MCMP σ_ref=σ_cyl(192)=0.11169168382612461; σ_sph 9 臂; err_by_N; umax_mean | m2 formal/alternative_all18.json + formal/sphere_N*.json tail_stats |
| MCMP µ 比Poiseuille H64/H128（名义 M=2, 实测 µ 比 2.0215418925016726; 锁定参考系数 ref_a_locked_pervolume/ref_b_kernel_permalss; u_profile_sim） | m2 formal/poiseuille_confirm3d.json |
| SC Laplace 主档 σ(L)=0.32022612/0.32649180/0.32942224（PR@Tr0.55）; 残差门; sim ratio 400–557; determinism replicates | m4 phase0/out/b_laplace.json（+ (c) 伪电流档如需 u_max） |
| CAC-B2 case A（µ_l 0.1/µ_g 0.05, harmonic, NY100, 50k, W=5, σ=0.001, mob 0.1, Uc=1e-4, H_eff=(NY−1)/2） | m1 out/b2_a_ny100.json + tests/b2_poiseuille.py |
| T2 C16b（R16, λ=1, Ca≈0.10, box6, µ0.85）D_I/ratio_sh/Ca_ach | t2 out/verdict.json arms.C16b |
| CG 解析求值器 | c1 scripts/cg_reference.py md5 e78c6d1eb22d628182636c0df74fb26a |

## 2. Case (a) 静滴 Laplace σ 自测（4 族）

每族按**自家冻结协议**跑/消费，几何匹配原则 = 各自 R 阶梯 + 固定主分辨率点。
**消费优先（防重复烧卡）**: 若 XA-3（out/xa3_selftest_stp_R*_P.json）或
XA-5（out/scan_*_beta0.7.json）已执行，其输出直接消费（判据脚本断言其
prereg md5 与本电池登记一致），本电池不重跑。

| 臂组 | 来源 | 协议（逐字冻结） | 新烧卡 |
|---|---|---|---|
| A-CAC | 新 3 run（或消费 XA-3 变体 P） | basis-C 周期（axes ppp）, R∈{16,24,32} box=6R, µ0.85, σ_in=0.007, W=4, mob 0.05, p0 0.01, fp64, 12000 步采样 500, 窗=末 4 采样, σ_eff=dp·r_eq/2（bands φ>0.9/φ<0.1） | ~0.9 GPU·h |
| A-CG | 新 4 run（或消费 XA-5 β0.7 主网格） | run_calib.py 逐字, A=0.04, β=0.7, τ=1, ρ 0.65/0.05, fp64, 30000 步, σ_eff=dp·R 尾 4; 点 = (128,25)+(200,{20,30,40}) | ~0.1 GPU·h |
| A-MCMP | **纯消费冻结档** | sphere 9 臂 tail_stats（σ 拟合在 alternative_all18）; u_max=umax_mean | 0 |
| A-SC | **纯消费冻结档** | b_laplace.json 主判读（R/L 保持读法, verdict 字段） | 0 |

无量纲比较子（每族）:
- **F_σ** = 自家 R 阶梯上 σ 的 (max−min)/mean（CAC: 3 点 dp·r_eq/2;
  CG: 3 点 (200,{20,30,40}) dp·R; MCMP: |σ_sph(N)/σ_ref−1| 三 N 点的
  极差; SC: σ(L) 三 L 点极差/均值——其阶梯是分辨率维, 如实标注）。
- **Laplace 线性残差**: SC=RMS 残差/max(ΔP)（档内自带）; MCMP=过原点
  LSQ dp vs 2/r_eq 拟合质量（档内 r2/series）; CAC/CG=dp·R 逐点 vs
  尾均值离散（披露列）。
- **u_max 尾值**（伪电流）: 各家尾窗均值; SC 若 (b) 档无则取 (c) 能力档
  并标注, 再无则 N/A 如实。
- **密度比披露列**: CAC 1 / CG 13 / MCMP 6.03 / SC 400–557（不进门）。
- **CAC 特有列**: σ_eff/σ_in 映射（唯一有 σ 输入的族; T3 先例 −7.6% 族）。

健康门（新跑臂全过才计入）: 无 NaN; 窗内 u_max 平台（末样-首样)/均值
≤0.5%; φ/质量漂移 ≤1e-6（CAC）、逐色 ≤1e-10/步/色（CG）。

## 3. Case (b) µ 比 2 两层 Poiseuille（3 族; 等密度、半程流体、解析逐臂）

| 臂组 | 来源 | 配置 | 新烧卡 |
|---|---|---|---|
| B-CG | **新 2 run** | c1 G2 协议逐字（nx=8, β=0.9, A=0.04, fp64, 壁面平面约定 cg_reference + 实测 j_first_blue）; **新 τ 对 (1.5,1.0) → ν 比恰 2**; ny∈{64,128}; gx 由解析逆定（cg_reference 前向计算 unit-gx 解析 u_max → gx=A/that, 跑前机器断言解析 u_max=0.04±1e-6）; 步数 60000/200000（eq 档表）; 若尾 20% u_max 平台 |Δ|/均值 >0.5% → 预授权**恰一次**延至 g10 表 80000/250000 并披露 | ~0.2 GPU·h |
| B-CAC | **新 2 run** | m1 B2 case A 逐字（harmonic µ 律, NY∈{100,200}, 50k 步, W=5, σ=0.001, mob 0.1, Uc=1e-4, gravity_mode uniform, axes pwp, fp32）; u_ana + fd_residual ≤1e-10 机器复跑作回执; **NY100 臂兼确定性回执**: 尾段 u_max_sim/剖面与 b2_a_ny100.json 相对差 ≤1e-9（不过 → 停报环境漂移） | ~0.3 GPU·h |
| B-MCMP | **纯消费 + 后处理** | poiseuille_confirm3d.json H64/H128: relL2(u_profile_sim, 解析(实测 µ_w/µ_g, 锁定 ref 系数)) 后处理重算; 解析求值器须过独立 FD 算子残差 ≤1e-10（M1 A3 制式）; 若 m2 锁定参考隐含特定壁约定, 两种读法并列披露 | 0 |

比较子: 逐臂 relL2 + 两档单调 + 实测 µ 比 vs 名义 2 + 接口位置自测 +
u_max 收敛度。CG 族门沿用 c1 G2 冻结门（relL2 ≤3%, 逐色质量 ≤1e-10/步/色）
仅作家族内参照, 不构成跨族判决。

## 4. Case (c) Taylor 剪切 D(Ca), λ=1, Ca~0.1

| 臂组 | 来源 | 说明 |
|---|---|---|
| C-CAC | **纯消费冻结档** | t2 verdict.json arms.C16b（R16, λ=1, Ca_ach≈0.10）: D_I, D_SH(Ca_ach) 因子（ref/taylor_ref.py 冻结）, ratio_sh, 在/出带 [0.96,1.04] 状态披露 |
| C-CG | **条件臂** | P0 CPU 探针（零 GPU）: 冻结 c1 内核是否支持剪切驱动（运动壁/等效）。不支持 → 如实记 "CG 剪切不支持（冻结内核）", case c 收缩为 CAC 行。支持 → **先锁参考**: 2D Stokes 小 Ca 变形律 D=k(λ)·Ca 的 k 用独立 2D Stokes 解算器（边界积分或 FD, 圆夹杂物 λ=1）机器锁定（常数 µ 极限自检 ≤1e-8 相对）, 锁定档案先落 out/x1_c_ref_2d.json 才准跑; 然后 1 GPU run（R=20, box 6R, 周期剪切, Ca_ach 目标 0.1, Ca 用 case-a CG 自测 σ@(200,20) 消费值）; 输出 D_sim/(k·Ca_ach)−1 披露列（无门） |

## 5. 预注册判读（见数字后禁改）

- 电池无总门; 每新跑臂须过 §2/§3 健康门才计入表。
- 归因表列 = {F_σ, 线性残差, u_max, 密度比, µ 比闭合 relL2, D(Ca) 披露};
  行 = {CAC, CG, SC, MCMP} × {Laplace, Poiseuille-µ2, Taylor}。
- 预注册禁令: 跨族比较 σ 绝对值; 用电池结果直接给任何旧轨翻案
  （升级/翻案权在控制器）; 在电池内调参重跑（无闭环重驱）。
- 确定性回执（B-CAC NY100）不过 → 全电池停报, 无任何表。

## 6. 预算与止损

- 新烧卡合计 ≈ 1.5–2.0 GPU·h（A-CAC 0.9 + A-CG 0.1 + B-CG 0.2 +
  B-CAC 0.3 + C-CG ≤0.1）; **止损 4 GPU·h**; 任务绝对上限 10 GPU·h。
- GPU6 分时: 每批前 nvidia-smi 快照入本目录 ctrl/, 窗口记 NOTES.md;
  GPU5/T1 与 GPU0-4/7 禁碰; OMP_NUM_THREADS ≤16。
- 止损: 任一新跑臂失稳/NaN → 单臂记 N/A 继续; ≥3 臂 N/A 或预算触顶 →
  停批回报。消费档字段缺失 → 该列 N/A 如实披露, 不补跑。

## 7. 输出与纪律

- out/x1_crossval.json（全臂原始 + 健康门 + 消费档切片与 md5 回执 +
  参考锁断言记录; 判据数字全部机器提取）; out/x1_attribution_table.md
  （一页归因表, 由 JSON 渲染, 零手抄）。
- prereg/NOTES append-only; 作废档 superseded_runs/; 不改任何上游档案
  与 worktree; 不 push/PR; FluidX3D 禁看。
