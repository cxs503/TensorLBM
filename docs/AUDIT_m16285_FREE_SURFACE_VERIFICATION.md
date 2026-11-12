# `free_surface_verification` 审计：两处自证式"验证"（MOCK）+ Wagner 闭式因子错（缺陷修复）

锚点 `NEW:自由表面验证模块的两处"验证"实为自证式仿真，且 Wagner 闭式与其自带压力积分互不一致`
· mission m16285 · branch `agent/m16285` · repo TensorLBM · 受影响文件
`src/tensorlbm/free_surface_verification.py`

---

## 1. 案例介绍（对象 / 背景 / 参数）

`src/tensorlbm/free_surface_verification.py` 是自由表面族的"验证"入口，四个 case：
`DamBreakVerification`（Ritter 解析 vs 彩色梯度 LBM）、`WigleyWaveVerification`（Michell 薄船）、
`WaterEntrySlamming`（Wagner 1932 楔形体入水）、`NegativeControls`（单相极限 / 零表面张力）。
`run_all_verifications()` 汇总成 `all_pass`。

本轮审计的问题在**证据链**，不在算例物理：模块对外报 PASS 的三项中，两项从未接触任何求解器。

| 参数（本次复算） | 值 |
|---|---|
| 楔形斜升角 β | 4° |
| 入水速度 V | 0.01（格子单位） |
| 密度 ρ | 1.0 |
| 取样时刻 t | 500 |
| Wagner 湿半宽 c(t)=πVt/(2tanβ) | 56.16 |
| 网格（Wigley 几何） | 200×25 / 400×50 / 800×100（xz 中点求积） |

## 2. 怎么做的（方法 / 流程 / 工具链）

1. 先按"声明 → 代码路径 → 是否触碰求解器"三问逐 case 复算执行路径（不改代码，纯读 + 跑）。
2. 拿仓内**独立实现** `tensorlbm.lng_membrane_sloshing.impact`（eqs.(1)–(4)，自带复势
   `w = -iV(z-√(z²-c²))` 的边界条件测试）作为外部口径对照，逐项比 c(t)、p(x,t)、F(t)。
3. 再拿"模块自带压力剖面积分"与"模块自带力斜率"互比（**内部闭合**，不依赖任何外部结论）。
4. 用 `p = -ρ∂φ/∂t`（平板势 `φ=-V√(c²-x²)` 的中心差分）作第三条独立路径。
5. 修复后写成可失败的不变量测试 `tests/test_free_surface_verification_audit_m16285.py`，
   并给出 fail-before / pass-after 残差梯图（程序化注入，零手抄）。
6. 工具链：numpy/pytest/matplotlib；证据 `artifacts/free_surface_verify_m16285/evidence.json`
   （由 `make_evidence.py` 生成）。

## 3. 计算结果（数字表，`evidence.json` 注入）

### 3.1 `WaterEntrySlamming`：自证式测量 + 双闭式互斥

| 量 | 审计前（shipped） | 基准 / 修复后 | 相对误差 |
|---|---|---|---|
| 力斜率 dF/dt | `ρV³π³/(8tan²β)` = 7.9263e-4 | `ρV³π³/(4tan²β)` = 1.58527e-3 | 审计前 **+50.00 %**（恰为 2 倍）；修复后 1.4e-16 |
| 压力 p(ξ=0) | `ρV²` = 1.000e-4 | `ρV²π/(2tanβ)` = 2.2463e-3 | 审计前 −95.5 %（缺 ċ/V = π/(2tanβ) 因子） |
| 自身压力积分 vs 自身力（t=500） | ∫p dx /(dF/dt·t) = 0.088848 | 1.0 | 审计前 **−91.11 %**，与解析比 (4/π)tanβ = 0.089034 一致 |
| "measured_force_slope" vs theory | 1.4e-16（构造性） | 由 p(x,t) 求积独立测得 | 审计前 **不可失败**（r²=1.0, verdict 恒 PASS） |

"恒 PASS"的机理（代码级）：旧 `run()` 里 `F_wagner = dF_dt_theory * t`，随后对这条**由理论
公式自身生成**的直线做最小二乘 → 斜率必等于理论值到机器精度，`relative_error=1.37e-16 < 0.15`
永远成立。`nx/ny/n_steps/device` 五个字段全部未使用（无求解器），却被当作仿真配置写进结果字典。

### 3.2 `WigleyWaveVerification`：`dispersion_check` 是代数恒等式

`U = Fr√L` → `λ = 2πU²/g`（g=1）→ `k = 1/U²` → `ω = √(gk) = 1/U` → `c = ω/k = U`，
故 `speed_ratio ≡ 1`（实测残差 **0.0**，Fr=0.2/0.45 同样成立）。旧代码把 `|ratio−1|<0.1`
写成 `verdict="PASS"`，即对任意 Fr、L 恒 PASS —— 非验证。

修复后该 case 如实标 `verdict="WITHHELD"`（`simulation_verified=False`），并把**真正有代码的地方**
（Wigley 型线 `y=(B/2)(1-(2x/L)²)(1-(z/T)²)`）换成可失败检查：离散几何积分出的方形系数应收敛到解析
`Cb = V/(LBT) = 4/9`。

| 网格 (nx×nz) | Cb（离散） | 相对 4/9 误差 |
|---|---|---|
| 200×25 | 0.44453889 | 2.125e-4 |
| 400×50 | 0.44446806 | 5.313e-5 |
| 800×100 | 0.44445035 | 1.328e-5 |

观测收敛阶 **2.000**（中点求积对被积函数为逐向二次 → 二阶），`geometry_verdict=PASS`。

### 3.3 `run_all_verifications()` 汇总

旧实现里 `negative_controls` 恒写 `"PASS"`（不读子项 verdict），且 `all_pass` 把两个自证 case
一起算进 PASS。修复后拆成 `solver_cases`（`dam_break` + 真实 negative controls）、
`analytic_only`、`withheld` 三栏，`WITHHELD` 绝不并入 `all_pass`。

## 4. 图

- `docs/figures/free_surface_verification_audit_m16285.png`
  (a) 压力剖面 p/ρV²(ξ)：旧（水平 1.0）/ 修正（∝1/√(1−ξ²)）/ `impact.py` 点重合；
  (b) F(t)：旧斜率只有基准一半，修正闭式与 p(x,t) 求积、`impact.py` 三条曲线重合。
- `docs/figures/free_surface_verification_audit_m16285_residuals.png`
  残差梯：旧力 5.0e-1、旧内部闭合 9.1e-1 → 新力 1.4e-16、新闭合 3.85e-4、新势函数差分 1.12e-7。

## 5. 注意点（坑与局限）

- **"测试绿"不代表验证**：原 `tests/test_free_surface_verification.py` 只断言键存在
  （如 `assert "measured_force_slope" in results`），执行成功即通过 —— 这正是两处自证式
  验证能长期潜伏的原因。新测试改为断言不变量（含"篡改理论式而测量不得跟随"的可失败性测试）。
- **本模块仍无入水 LBM 仿真**：修复的是"闭式算对 + 证据等级如实"，不是"LBM 对上 Wagner"。
  后者继续 `WITHHELD`，需另建彩色梯度入水算例（见 `benchmarks/pending/phasefield_water_entry`）。
- 求积残差 3.85e-4 不是物理误差：喷溅根 `1/√(1−ξ²)` 可积奇点使中点求积只有 `O(√h)` 收敛，
  故默认 `QUADRATURE_N = 1_000_001`；门限设 2e-3 并显式记录采样数。
- 旧 `tests/test_free_surface_verification.py` 全量运行 >120 s（含 dam break 实算），本轮只跑了
  `Wigley/WaterEntry` 两个类（8 passed）；dam break / negative controls 未重跑，**不作通过声明**。

## 6. 误差判定

| 关键量 | 基准来源 | 修复前误差 | 修复后误差 | <3 % ? |
|---|---|---|---|---|
| dF/dt | `impact.py` 独立实现（+ 复势 BVP） | +50.00 % | 1.4e-16 | 修复后 是 |
| p(x,t) 逐点 | `impact.py` | 最大 −95.5 %（ξ=0） | 0.0（逐位一致） | 修复后 是 |
| ∫p dx / F（内部闭合） | 模块自身闭式 | −91.11 % | 3.85e-4 | 修复后 是 |
| p vs −ρ∂φ/∂t | 平板势中心差分 | —（旧压力无常数因子，必错） | 1.12e-7 | 是 |
| Cb（Wigley 几何） | 解析 4/9 | 未测（旧无此检查） | 1.33e-5（阶 2.000） | 是 |
| `dispersion_check` | —— 代数恒等式，无基准 | 恒 PASS（伪证） | 已改判 `WITHHELD` | 不适用（无基准，声明原因：无波形仿真） |