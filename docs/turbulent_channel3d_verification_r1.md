# D3Q19 TRT 湍流槽道求解器：解析验证阶梯与力项缺陷清偿（R1）

> 对象：`src/tensorlbm/turbulence/turbulent_channel3d.py`（随 merge #891 合入 main 的 832 行模块，
> **合入时无任何 pytest 用例、无验证文档**，benchmark 仍停留在 `benchmarks/pending/channel_turb_3d/`）。
> 本单=验证欠账清偿：给已合并模块补解析锚 + 定位并修复一个**使整个战役驱动力失效**的缺陷。
> 所有数字由 `artifacts/turbulent_channel3d_verification/run_probe.py` 程序化生成
> （`results.json` + 两张 png），无手抄。

## 1. 案例介绍

**物理背景.** 压力驱动的三维平面槽道，半高 `h`、高 `H = ny-2`，流向/展向周期、两壁无滑移。
驱动力用常量体力 `a`（Guo 1993）表示，因此定常流的流向动量平衡是**精确**的：

```
tau_w = rho * a * h,   u_tau = sqrt(a h),   a = 2 u_tau^2 / H,   nu = u_tau h / Re_tau
```

模块自称内部有三条互相独立的摩擦速度路线（`body_force` / `wall_exchange`(Ladd 动量交换) /
`stress_fit`），"三者的相互吻合是战役的自洽门"。本次验证只用**闭式解**做锚，不做求解器互比。

**验证几何/参数（全部格点单位，CPU float32 单线程）.**

| 支路 | 网格 `(nx,ny,nz)` | 参数 | 解析参照 |
|---|---|---|---|
| A 力冲量 | (8,18,8) | `a=1e-4`，`nu ∈ {0.002,0.02,0.04,0.12,0.2}` | `dU/dt = a`（精确） |
| A′ 战役粘性 | (8,18,8) | `nu = u_tau·h/180 = 2.22e-3`，`tau+ = 0.5067` 量级 | 同上 |
| B 剪切模态 | (8,18,8) | `nu=0.04`，`u_x ∝ sin(pi y/H)`，2000 步 | `nu = (tau+ - 1/2)/3` |
| C 层流孪生 | (8,18,8) | 几何取战役 `Re_tau=180`，孪生重标定到 `u_max=0.05` | `u(y) = a y (H-y)/(2 nu)` |
| 负控 | 同上 | `u_max_target=None`（沿用战役 `Re_tau`） | —— |

`u_max = u_tau·Re_tau/2` 是这组参数化下的恒等式：`Re_tau=180, u_tau=0.01` 给出 `u_max=0.896`，
即 **Ma = 1.55 的高超声速层流孪生**（见 §3 表 3）。这条耦合是缺陷②的来源。

## 2. 怎么做的

1. 先读模块，确认无任何测试引用（`grep -rn turbulent_channel3d tests/` 只命中无关文件）。
2. 用最小探针切分物理：**均匀流 + 纯体力**隔离力项 → 发现 `dU/dt ≠ a`；
   **无外力剪切模态**隔离黏性 → 确认体相碰撞/流播无问题。
3. 对不同 `tau+` 扫描两种 `force_scheme`，把偏差拟合成 `s+` 的函数，定位到 TRT 力项权重。
4. 修默认值 + 加 `validate()` 封闭；再给层流孪生加低马赫数重标定（几何不变）。
5. 用闭式解锚 + 负控写成 18 条 pytest（`tests/test_turbulent_channel3d.py`，33 s）。

## 3. 计算结果

**表 1 — 力项动量源（锚 A，`artifacts/.../fig1_force_source.png`）**

| `nu` | `tau+` | `s+` | `trt_split`: `(ΔU/Δt)/a` | `guo_sp`: `(ΔU/Δt)/a` |
|---|---|---|---|---|
| 0.0020 | 0.5060 | 1.9763 | **1.0001** | 0.0277 |
| 0.0200 | 0.5600 | 1.7857 | **1.0000** | 0.2451 |
| 0.0400 | 0.6200 | 1.6129 | **1.0000** | 0.4359 |
| 0.1200 | 0.8600 | 1.1628 | **0.9998** | 0.9083 |
| 0.2000 | 1.1000 | 0.9091 | **0.9994** | 1.1601 |
| 2.22e-3（战役 Re_tau=180 粘性） | 0.5067 | 1.9947 | **1.0000** | **0.0063** |

`guo_sp` 的注入率近似 `2.11 - 1.045·s+`：`s+ → 2`（即 `tau+ → 1/2`，战役的真实工作点）
时**注入量趋于 0**。在战役粘性下它只给出 **0.63 %** 的驱动力。

**表 2 — 剪切模态黏性与层流孪生（锚 B / C，`fig2_laminar_twin.png`）**

| 量 | 数值 | 解析/基准 | 相对误差 | <3 % |
|---|---|---|---|---|
| `nu_eff`（剪切模态 500→2000 步） | 0.039939 | 0.040000 | **0.15 %** | 是 |
| 层流孪生 L2（halfway） | 6.76e-3 | — | 0.68 % | 是 |
| 层流孪生最大逐点误差（halfway） | — | `u_max` 归一 | **0.94 %** | 是 |
| 层流孪生峰值比 `u_max^sim/u_max^exact` | 0.99065 | 1 | 0.94 % | 是 |
| 实测壁面原点（halfway，`q=0.5`） | 0.4725 | 0.5 | 5.5 %（绝对偏差 −0.0275 格） | 是（几何量） |
| 层流孪生 L2（interpolated `q=0.625`） | 2.46e-2 | — | **2.46 %** | 是（但见 §5 局限） |
| 层流孪生最大逐点（interpolated） | — | `u_max` 归一 | **5.62 %** | 否 |
| 壁面原点指针（interpolated，几何应为 0.375） | 0.495 | 0.375 | **32 %** | 否 |

**表 3 — 负控（锚必须能失败）**

| 负控 | 观测 | 结论 |
|---|---|---|
| `guo_sp` 在战役 `tau+` 下的注入率 | 0.0063 a | 锚 A 对力项缺陷有鉴别力（>150× 差距） |
| 沿用战役参数（`u_max_target=None`）的层流孪生 | `u_max = 0.896`，`Ma = 1.55`，`L2 = nan` | 旧标定路径在战役 `Re_tau` 下**必然发散** |
| 重标定后的孪生（`u_max_target=0.05`） | `tau+ = 0.65`，`u_max = 0.0498`，L2 = 6.8e-3 | 修复后同一战役几何可标定 |

## 4. 修了什么（`src/`，最小改动）

| # | 改动 | 依据 |
|---|---|---|
| 1 | `force_scheme` 默认 `"guo_sp"` → **`"trt_split"`**，并把注释里被写反的结论改正 | 表 1：`trt_split` 在全部 `tau+` 上 `dU/dt = a`（≤0.06 %）；`guo_sp` 是受 τ 调制的错解 |
| 2 | `validate()` 新增 `force_scheme` 白名单 | 原实现里拼错的值会**静默落到 else 分支**（等价于 `trt_split`），不可审计 |
| 3 | `run_laminar_poiseuille` 新增 `u_max_target=0.05` / `nu_target=0.05` 重标定：`Re_tau_twin = sqrt(2 u_max h/nu_target)`、`u_tau_twin = 2u_max/Re_tau_twin`，`ny/wall_origin/q/wall_scheme` 逐位不变 | 表 3：孪生的 `u_max = u_tau Re_tau/2` 与战役 `Re_tau` 强耦合，`Re_tau=180` 时 Ma=1.55 必发 NaN；壁面原点是**几何量**，不需要在战役 `Re_tau` 下测 |

`u_max_target=None` 保留旧行为（供负控与历史复现）。

## 5. 注意点 / 局限（含未完成项，不粉饰）

1. **`wall_exchange`（Ladd 动量交换）路线在孪生里数值不可用**：孪生报告
   `u_tau_wall_exchange = 0.03556`，而 `body_force` 路线 = 0.025（孪生自身 `u_tau`），
   即 τ_w 高估 2×（>40 % in u_tau）。**但**同一累加器在定常态单步预算探针里给出
   `tau_w_code/tau_w_exact = 1.0007`。说明缺陷在孪生的**聚合口径**（窗口采样/复位）
   而非每步公式。本单**未修**（超出最小改动范围），已固化为待办：
   复核 `run_laminar_poiseuille` 中 `flux += _wall_flux_accum` 的窗口边界
   （`step > nsteps//2 and step % 25 == 0` 与 `n_flux_steps = count*25` 的一致性）。
2. **插值反弹（`q≠0.5`）不收敛到解析抛物线**：`q=0.625` 下 L2 2.46 %→3.11 %（800→2000 步
   不降），最大逐点 5.6 %，且壁面原点指针恒返回 ≈0.495 而非几何值 0.375。
   因此**`wall_origin ≠ 0.5` 目前不得用于 `y+` 标定**；战役跑的是 halfway。
   该观测已写成测试 `test_wall_origin_semantics_is_a_documented_limitation` 冻结，
   将来修好会红，强制重审。
3. 本单只验证**层流锚**。Re_tau=180/395 的壁面解析湍流（MKM DNS、Reichardt 复合律、
   `u+ = log(y+)/κ + B`）需要 `benchmarks/pending/channel_turb_3d/` 的完整战役，
   在 load15 > 100 的共享 CPU 上不可行，**未做**（本单只保证战役的前置门：
   驱动力、黏性、壁面原点三项在闭式解上成立）。
4. `stress_fit` 路线、Smagorinsky 开关、`turbulence_energy()` 统计未进入本阶梯。
5. 全部测量 CPU float32 单线程；数值与线程数无关。

## 6. 误差判定（逐项 vs 基准）

| 关键量 | 基准 | 相对误差 | 是否 <3 % |
|---|---|---|---|
| 力项动量源 `dU/dt/a`（`trt_split`，6 个 τ） | 1（精确） | ≤0.06 % | **是** |
| 体相黏性 `nu_eff` | `(tau+-1/2)/3` | 0.15 % | **是** |
| 层流剖面（halfway）L2 / 最大逐点 | 闭式抛物线 | 0.68 % / 0.94 % | **是** |
| 层流峰值速度（halfway） | 闭式峰值 | 0.94 % | **是** |
| 壁面原点（halfway） | 0.5 | 绝对 −0.0275 格（5.5 %） | **是** |
| 层流剖面（interpolated `q=0.625`）最大逐点 | 闭式抛物线 | 5.62 % | **否**（见 §5-2） |
| 壁面原点指针（interpolated） | 0.375 | 32 % | **否**（见 §5-2，未修） |
| `u_tau^wall_exchange`（孪生聚合） | 孪生 `u_tau=0.025` | 42 % | **否**（见 §5-1，未修） |

三条 `否` 都对应明确的未清偿项，不是测量噪声：两条已写成会红的冻结测试，一条写进待办。

## 7. 证据指针

* 代码：`src/tensorlbm/turbulence/turbulent_channel3d.py`（默认值 + `validate` + 孪生重标定）
* 测试：`tests/test_turbulent_channel3d.py`（18 条，33 s，`PYTHONPATH=src OMP_NUM_THREADS=1 pytest -q tests/test_turbulent_channel3d.py`）
* 工件：`artifacts/turbulent_channel3d_verification/{run_probe.py,results.json,fig1_force_source.png,fig2_laminar_twin.png}`
* 复核：`PYTHONPATH=src OMP_NUM_THREADS=1 python3 artifacts/turbulent_channel3d_verification/run_probe.py`