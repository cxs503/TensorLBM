# 验证欠账清偿：turbulent_channel3d 壁面动量交换（Ladd）应力记账缺陷修复（2026-11-08, m6442）

## 0. 结论（一句话）

`docs/turbulent_channel3d_verification_r1.md` §5-1 留下的「δ 未修」欠账已定位并修复：
`_wall_flux_accum` 在**平均窗口开始前从不归零**，导致起飞段（spin-up）的动量交换通量被
**除以更短的窗口长度**——孪生里恰好 `nsteps//2` 步通量 ÷ `nsteps//2` 步窗口 = **τ_w 高估 2×**
（`u_tau` ×√2 = **+42 %**，与旧文档实测 42 % 完全一致），生产路径 `run()` 里则是
`(warmup+window)/window` 倍（campaign 默认 warmup=5000/interval=100 ⇒ **51×**）。
修复后孪生 `u_tau(wall_exchange)/u_tau(body_force) = 1.0041`（0.41 %），生产路径
稳态窗口 `τ_w` 比 = **0.9990**（0.10 %）。新判据已成测试（红→绿闭环：修复前 1.4218 / 1.5053）。

## 1. 案例介绍（物理背景与参数）

平面槽道 D3Q19 TRT，两层壁（y=0、y=H），体加速度 `a` 驱动；层流 Poiseuille 稳态的壁面
剪应力有**闭式值** τ_w = ρ a h（h=半高），因此壁面应力可被逐点裁决，无需 DNS 对照。
Ladd 动量交换给出独立的第二应力通道：每步在壁面格点累加 Σ c_x (f_post + f⁺)，
窗口平均后 τ_w^ex = −flux/(2·A·n_steps)（两面壁）。

| 参数 | 值 |
|---|---|
| 网格 nx×ny×nz | 8×18×8（Ny=18, H=16, h=8） |
| 壁面格式 | halfway（q=0.5），负控 q=0.625 interpolated |
| 孪生 nu=0.05, u_max=0.05 | Re_τ_twin = √(2·u_max·h/nu) ≈ 4.0, u_τ,twin = 0.025 |
| 生产路径 | a=1e-4, τ_w 精确 = 1e-4, 5000 步, sample_interval=10, warmup∈{0,500,1000,2500} |
| 设备 | CPU float32 单线程（`torch.set_num_threads(1)`） |

## 2. 怎么做的（方法/流程）

1. **复现**：探针脚本对 `run_laminar_poiseuille` 与 `run()` 同时输出
   `u_tau_body_force`（=√(ah)）与 `u_tau_wall_exchange`，测到 1.4218（孪生）与 warmup 相关的额外放大（生产）。
2. **根因重构（独立执行路径推演）**：`_bounce_back` 每步累加 `_wall_flux_accum`，
   但只在采样步 flush+清零（`step > warmup and (step−warmup)%interval==0`；
   孪生里 `step > nsteps//2 and step%25==0`）。窗口前的步数被计入分子，分母却只数窗口步数。
   单步预算探针（旧文档）给出 `tau_w_code/tau_w_exact = 1.0007`，说明**每步公式是对的**，
   缺陷在聚合口径——与旧文档判断一致。
3. **修复（最小改动，`src/` 三处）**：
   * `run()`：`step <= warmup_steps` 时清零累加器（起飞通量不进平均）；
   * `run_laminar_poiseuille()`：`step == nsteps//2` 时清零；
   * 分母由「假定 `count*25`」改为**实测窗口步数** `flux_steps`（最后一块不满 25 步时
     分子分母口径一致，如 nsteps=510：实测 245 步 vs 旧假定 250 步 = 2 % 偏差）。
4. **判据冻结**（`tests/test_turbulent_channel3d.py`，3 例新增）：
   * `test_laminar_twin_wall_exchange_stress_matches_body_force[500/510]`：比值 vs 1 误差 <2 %；
   * `test_production_run_does_not_book_spinup_wall_flux`：把通道初值装在**闭式抛物稳态**上
     （每步壁面通量为常数），warmup=100/interval=10/steps=300，比值 vs 1 误差 <3 %；
     修复前该项为 1.5053（= 300/200 记账放大），修复后通过。
   * 两例都用闭式 τ_w=a h 作基准，不依赖求解器互比。

## 3. 计算结果（`artifacts/wall_exchange_bookkeeping_20261108/results.json`）

### 3.1 孪生：壁面应力 vs 闭式 a·h（步数扫描）

| steps | u_τ^we/u_τ^bf | τ_w 比 | 相对误差 | <3 % ? |
|---|---|---|---|---|
| 500 | 1.004081 | 1.008179 | 0.82 % | 是 |
| 1000 | 1.006672 | 1.013388 | 1.34 % | 是 |
| 2000 | 1.009510 | 1.019110 | 1.91 % | 是 |
| 4000 | 1.012852 | 1.025869 | 2.59 % | 是 |
| 修复前 500 | 1.421826 | 2.021658 | **+102 %** | **否** |

### 3.2 生产 `run()`：稳态窗口（τ_w 精确 = a h = 1e-4）

| warmup | τ_w^we/τ_w^bf | 说明 |
|---|---|---|
| 0 | 0.9148 | 窗口含起飞段，通量尚未达稳态（物理，非记账） |
| 500 | 0.9646 | 窗口前移，趋向 1 |
| 1000 | 0.9850 | — |
| 2500 | **0.9990** | 稳态窗口，0.10 % |

单调收敛到 1 证明记账口径已正确；修复前的同一窗口会叠加
`(warmup+window)/window` 结构性放大（warmup=100 时 1.5×，见 §3.3 测试）。

### 3.3 负控（修复必须不动的东西）

* `q=0.625` interpolated 壁面剖面对闭式抛物线 L2 误差 **3.36 %**——旧文档 §5-2 记录的
  插值不收敛问题**未受影响**（该欠账仍在，未在本单关闭）。

## 4. 结果图

`fig1_wall_exchange_bookkeeping.png`：左=孪生步数扫描（修复后 ⊗ 精确值，红虚线=修复前 √2），
右=生产路径 warmup 扫描收敛到 1。

## 5. 注意点 / 局限（不粉饰）

1. 孪生的比值随步数**缓慢上漂**（0.82 %→2.59 %），仍 <3 %。原因：初值装在连续抛物线，
   离散稳态与之差 ~1 %，通量在窗口内对离散稳态仍有微弱重整；这是**离散稳态 vs 连续**的
   残差，不是记账缺陷（记账缺陷是步数无关的常数 2× 偏置，已在负控中区分）。
2. 本单只裁决**层流锚**（闭式 τ_w=a h）。τ_w 与壁面函数/插值反弹的组合、
   `stress_fit` 路线、Re_τ=180/395 湍流仍不在本阶梯内（旧文档 §5-3/§5-4 欠账未动）。
3. 每步公式的浮点口径：累加器仍是 Python float（CPU）标量；GPU 上 `float()` 同步每步一次，
   本次未改（性能议题，超出最小改动）。
4. 全部测量 CPU float32 单线程；load15 >100 的共享机，本单为轻量算例（~30 k 步·单元）。

## 6. 误差判定（逐项 vs 基准，是否 <3 %）

| 关键量 | 基准 | 修复后 | 是否 <3 % |
|---|---|---|---|
| 孪生 u_τ(wall_exchange) | u_τ=√(a h)=0.025 | +0.41 %…+1.29 % | **是** |
| 孪生 τ_w(wall_exchange) | a h | +0.82 %…+2.59 % | **是** |
| 生产 τ_w(wall_exchange)，稳态窗口 | a h=1e-4 | −0.10 % | **是** |
| 生产 τ_w（warmup=0，含起飞段） | a h（稳态值） | −8.5 % | **否**（窗口含瞬态，物理原因，非缺陷；已声明） |
| 修复前孪生 τ_w | a h | +102 % | **否**（缺陷本体的证据） |

## 7. 证据指针

* 代码：`src/tensorlbm/turbulence/turbulent_channel3d.py`（`run()` 清零 + 孪生零窗口 + 实测分母）
* 测试：`tests/test_turbulent_channel3d.py`（新增 3 例；红→绿：修复前 1.4218/1.4218/1.5053）
* 工件：`artifacts/wall_exchange_bookkeeping_20261108/{run_experiment.py, results.json, run.log, fig1_wall_exchange_bookkeeping.png}`
* 上游欠账：`docs/turbulent_channel3d_verification_r1.md` §5-1（本单关闭）；
  §5-2（插值反弹不收敛）与 §5-3（Re_τ 湍流阶梯）**仍开放**。