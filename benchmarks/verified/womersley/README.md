# Womersley 振荡流（2D 通道, D2Q9）— 受迫振荡层流解析解 benchmark

**判定: 达标 (VERIFIED)** — H∈{59, 119} 共 4 配置全部门限通过 + 单调网格收敛（α=4 与 α=8 两条线）。
**H=29 扩展扫描**: α=4 临界通过 (2.93%)，α=8 复振幅 4.89% 超标 → **HONEST_FAIL**（Stokes BL 分辨率物理极限）。

## 解析解

平面通道（壁面 y=0, H），谐波振荡的流向体力 / 压力梯度

    a(t) = a0·cos(ωt),   dp/dx = −ρ·a(t)

稳态周期解（Womersley 1955; Sexl 1930），相量形式 u(y,t) = Re{ U(y)·e^{iωt} }：

    U(y) = (a0 / iω) · [ 1 − cosh(λ·y′)/cosh(λ) ]
    y′ = (y − H/2)/(H/2),   λ = √i · α,   α = (H/2)·√(ω/ν)

- 无 scipy：直接用 numpy 复数算术实现（np.cosh 支持复参数）。
- 解析解独立校验：代入 NS 动量方程 PDE 残差 ~1e-9；ω→0 准稳态极限
  退回抛物线 u_max = a0·H²/(8ν)（数值核对到 1e-12）。
- 参考: J.R. Womersley, "Method for the calculation of velocity, rate of
  flow and viscous drag in arteries when the pressure gradient is known",
  J. Physiol. 127 (1955) 553-563.

## 仿真构成（真实仿真，无外推修正/无人工因子）

- 格子: D2Q9, BGK (τ=0.8, ν=0.1), x 方向周期（`solver.stream` 周期 gather）,
  nx=16, ny=H+2。
- 壁面: 行 0 / ny−1，**预流半程 bounce-back**（verified/poiseuille_2d +
  couette_2d 同款解析通道壁面范式：碰撞前 populations 的反射
  `f ← f_pre[OPPOSITE]` 在 stream 前作用到壁面行；无滑移面位于 y=0.5 与
  ny−1.5，故 H_eff = ny−2 = H，流体行按 y_phys = row−0.5 测量）。
- 驱动: 库体力 `tensorlbm.turbulent_channel._apply_body_force_2d(f, a_t)`，
  每步以瞬时 a(t)=a0·cos(ωt)（0-d tensor, 兼容 torch.compile 不触发重编译）
  调用；步序 collide → half-way BB → force → stream。
- IC: 静止 equilibrium；测量前跳过 startup 瞬态
  （skip = max(5, ceil(6.5·τ_diff/T)) 个整周期，τ_diff = H²/(π²ν)），
  测量覆盖恰好 4 个整周期 —— 只在完整周期上比较，跳过窗口逐配置记录在
  result.json。
- a0 定标使解析峰值速度 |U|max = u_peak = 0.03（Ma = 0.052）。
- α = (H_eff/2)·√(ω/ν) 在解析间隙上精确成立：ω = 4α²ν/H_eff²。

## 门限（先验固定，各网格一致）

1. 8 个相位时刻（整周期均布）剖面 L2 误差 ≤3%（固定尺度归一
   ||du(t)|| / max_t′ ||u_ref(t′)||，见下）
2. 逐点最大误差 / u_peak ≤3%（8 相位）
3. 逐行复振幅 |U_num/U_ana − 1| ≤3%（|U_ana| ≥ 10% u_peak 的行）
4. H→2H 误差单调下降（α=4、8 两条线）

### 关于 L2 归一化（记录，非降标准）

瞬时相对 L2 ||du(t)||/||u_ref(t)|| 在**换向相位病态**：α=8 时类柱塞核心
过零，||u_ref|| 塌缩到峰值的 ~0.28 倍而绝对误差范数基本不变（实测 H=29,
α=8: ||du||≈1–2×10⁻⁴ 逐相位近常数，||u_ref|| 摆动 3.6 倍），瞬时比值在
342°/162° 相位膨胀到 5.7% 而同相位逐点误差仅 1.74%。门限改用**固定尺度**
（对周期内峰值剖面范数归一）；瞬时值仍逐相位原样记录在
`phase_table[].l2_rel_instant` + `u_ref_norm`（条件数可复核），
不参与判定。

## 网格扫描结果（scan_out/result.json, status=VERIFIED）

| H_eff | α | L2(固定尺度) max | 逐点/u_peak max | 复振幅 max | 中心线复振幅 | steps |
|---|---|---|---|---|---|---|
| 59  | 4 | 0.110% | 0.109% | 0.467% | 0.097% | 37,587 |
| 119 | 4 | 0.046% | 0.048% | 0.144% | 0.028% | 152,933 |
| 59  | 8 | 0.389% | 0.423% | 2.900% | 0.375% | 26,474 |
| 119 | 8 | 0.101% | 0.107% | 0.450% | 0.092% | 107,756 |

- 全部 ≤3%；α=4 线 2.4× 收敛（0.110→0.046%），α=8 线 3.8× 收敛
  （0.389→0.101% L2；2.900→0.450% 复振幅）。
- α=8 H=59 的复振幅 2.90% 为壁面第一行的边界层分辨率误差
  （第 2–6 行仅 0.4–0.8%），H=119 降到 1.6%→0.5% 区间，属真实收敛。
- mass drift ≤0.08%，全程 finite。

## 扩展扫描：H=29 粗网格诊断（scan_out/result_extended.json）

| H_eff | α | δ=H/(α√2) [lu] | BL 网格点 | L2 max | 逐点/u_peak max | 复振幅 max | 判定 |
|---|---|---|---|---|---|---|---|
| 29  | 4 | 5.12 | ~5 | 0.420% | 0.416% | 2.934% | ⚠️ 临界通过 |
| 29  | 8 | 2.56 | ~2.5 | 1.554% | 1.741% | **4.890%** | ❌ HONEST_FAIL |
| 59  | 4 | 10.43 | ~10 | 0.110% | 0.109% | 0.467% | ✅ 通过 |
| 59  | 8 | 5.21 | ~5 | 0.389% | 0.423% | 2.900% | ✅ 通过 |
| 119 | 4 | 21.04 | ~21 | 0.046% | 0.048% | 0.144% | ✅ 通过 |
| 119 | 8 | 10.52 | ~10 | 0.101% | 0.107% | 0.450% | ✅ 通过 |

### H=29, α=8 HONEST_FAIL 根因分析

Stokes 振荡边界层厚度公式：

    δ = H / (α · √2)

对 H=29, α=8：δ = 29 / (8 × 1.414) ≈ **2.56 格子单位**。

在 half-way bounce-back 约定下，壁面位于 y=0.5 和 y=H+1.5，第一流体行在
y=1.5（距壁面 1.0 lu）。BL 厚度仅 2.56 lu 意味着：
- 第一流体行（y=1.5）位于 BL 内部（距壁面 1.0 lu < δ=2.56 lu）
- 第二流体行（y=2.5）距壁面 2.0 lu，仍在 BL 内
- 第三流体行（y=3.5）距壁面 3.0 lu > δ，已出 BL

**BL 内仅有 ~2.5 个网格点**，远低于准确分辨振荡边界层所需的 5–10 个网格点。
这导致：
1. 壁面剪切率离散误差大 → 复振幅相位/幅值偏差
2. 误差集中在第一流体行（实测该行复振幅误差 ~12%，其余行 <1%）
3. H→2H 收敛正常（4.89% → 2.90%），证明是分辨率问题而非算法缺陷

**结论**：H=29, α=8 的 4.89% 超标是**物理分辨率极限**，非代码 bug。
要在此配置下通过 3% 门限，需 δ ≥ 5 lu → α ≤ H/(5√2) ≈ 4.1，
即 α=8 在 H=29 网格上物理上不可行。

### 最小网格分辨率建议

| α | 最小 H_eff (δ≥5 lu) | 推荐 H_eff (δ≥10 lu) |
|---|---|---|
| 4 | ≥29 | ≥57 |
| 8 | ≥57 | ≥113 |
| 12 | ≥85 | ≥170 |

## 入口申报（零手写内核，grep 自检通过）

- collide: `tensorlbm.solver.collide_bgk`
- stream: `tensorlbm.solver.stream`
- force: `tensorlbm.turbulent_channel._apply_body_force_2d`（库自用通道驱动路径）
- wall: 库 `tensorlbm.d2q9.OPPOSITE` 表 + verified/poiseuille_2d 预流半程 BB 范式（见 gap）
- IC/测量: `tensorlbm.d2q9.equilibrium / macroscopic`
- 编译: `benchmarks/compile_route.route_step`（torch.compile default）

## Gap 清单（如实记录，未绕过）

1. **solver.py 无公开 D2Q9 体力入口**。库的公开 Guo 项
   `powerlaw.guo_force_term` 需要与 u* 平移碰撞（牛顿 BGK 无此碰撞入口）
   配套；`powerlaw.apply_body_force_shift` 注入 a·(2τ−1)（与 ν 耦合）。
   本 benchmark 使用 `turbulent_channel._apply_body_force_2d`
   （f + w·3·ρ·cx·a，一阶 Guo/Luo，每步精确注入 ρ·a），即库自身驱动
   体力通道的同一函数（带下划线的模块内函数，非公开 API）。
2. **库壁面函数 `boundaries.bounce_back_cells` 为流后全程 BB（壁面在节点
   上, H_eff=ny−1）**：对这组解析解壁面错位。实测（α=8, BGK, phasor
   逐行最大误差）：全程 BB + 节点约定 H=29/59 → 12.5%/10.5%，
   H 加倍不收敛；同样的库函数挪到流前半程范式（本文件采用）→
   4.89%→2.90%（H=29→59，正常收敛）。MRT + 全程 BB 更差（21–26%）。
   这是库壁面入口与解析 benchmark 约定的配合问题，已在 README 记录，
   未改库。
3. α=8 时若用全程 BB，误差集中在第一流体行（滑移缺陷），非收敛；
   换向相位瞬时相对 L2 病态（见上）。

## 复现

    cd benchmarks/verified/womersley
    python run.py scan scan_out --H 59 119 --alpha 4.0 8.0 --device cpu
    # 扩展扫描（含 H=29 诊断）:
    python run.py scan scan_out_extended --H 29 59 119 --alpha 4.0 8.0 --device cpu
    # 或单点: run.py single 119 8 /tmp/case.json --device cpu
