# D3Q19 BGK 解析解验证 — R1（validation-debt burn-down）

**范围.** `tests/test_lbm_verification.py`（4 个解析解案例 + 1 个负控）、
`scripts/verify_lbm_analytic_convergence.py`（provenance 脚本）、
`tests/ci_quarantine.txt`（4 个节点 ID 退役）。
**`src/` 未改动** —— 本变更只做验证，所以下面所有数字同时就是对
`solver3d.collide_bgk3d` / `stream3d`、`boundaries3d.bounce_back_cells_3d`、
`d3q19.equilibrium3d` / `macroscopic3d` **当前合并态**的陈述。

**状态.** 测试改写与 quarantine 退役已随 `5cd12e87` 落在 `main`
（`tests/test_lbm_verification.py` 由「sdaa 钉扎 + `return bool`」改写为
「设备无关 + assert」）。本次变更补齐被提交门禁滤除的两件产物：**本文档**与
**provenance 脚本**，并给案例函数加了只影响默认值的网格/步数关键字参数
（默认参数即断言算例），使脚本走的是**断言同一段代码**，公开表格不会与 CI 漂移。

| 部件 | 落点 | 本变更是否改动 |
|---|---|---|
| `tests/test_lbm_verification.py` 由脚本改写为 assert 用例 | `5cd12e87`（已在 main） | 仅加 `nx`/`ny`/`steps`/`tau`/`bounce_back` 关键字参数，默认值不变 |
| `tests/ci_quarantine.txt` 4 个节点 ID 退役 | `5cd12e87`（已在 main） | 不改（已确认无残留） |
| `scripts/verify_lbm_analytic_convergence.py` + `artifacts/lbm_analytic_verification/*` | 本变更 | 新增 |
| `docs/lbm_analytic_verification_r1.md`（本文） | 本变更 | 新增 |

## 1. 原先的问题（记录原因，已在 main 修复）

改写前（`5cd12e87^`）的模块自称「100% 解析解验证，误差<1%」，实际**零验证信号**：

| 缺陷（改写前） | 后果 | 证据 |
|---|---|---|
| `def test_step_a_shear_wave(device="sdaa:0")` —— 默认参数被 pytest 当作 **fixture 请求** | 四个用例在收集期即 error（`fixture 'device' not found`），在任何 runner 上都跑不起来 | `git show 5cd12e87^:tests/test_lbm_verification.py`（4 处同样签名） |
| 每个用例 `print("PASS")` 后 `return passed`（bool） | pytest 丢弃返回值 → 即使设备存在，模块也**不可能失败** | 同上（`return passed` 4 处） |
| 4 个 node ID 躺在 `tests/ci_quarantine.txt` | `tests/conftest.py:93` 把命中项挂成 **non-strict xfail**，收集错误被吞掉，CI 绿灯对此无任何信息 | `git show 5cd12e87 -- tests/ci_quarantine.txt`（4 行删除） |

改写后：设备默认 CPU（`TENSORLBM_TEST_DEVICE` 可覆盖）、判据写成 `assert`、
逐分量 Python 循环改成整张量布尔掩码（数学逐点等价）。

## 2. 方法

四个案例都有**格点单位下的闭式解**，残差因此是离散误差而不是模型对模型的比较。

| 案例 | 解析基线 | 比较量 |
|---|---|---|
| A 剪切波衰减 | `u_y = A₀ cos(kx) e^{−νk²t}`，`ν = (τ−½)/3` | 从 `ln A(t)` 反解的 ν |
| B Couette | `u_x = U (y−y_w1)/(y_w2−y_w1)`，`y_w1=0.5`、`y_w2=n_y−1` | 剖面逐点误差（÷U） |
| C Poiseuille | `u_x = G(y−y_w1)(y_w2−y)/2ν`，`y_w1=0.5`、`y_w2=n_y−1.5` | 峰值速度（`G` 由目标 `u_max=0.05` 反设） |
| D 壁面摩擦 | `τ_w = ν ∂u/∂y‖_wall` ⇒ `C_f = 2ν/(L U)` | 由壁面首节点单侧梯度求 `C_f` |

**每步算子顺序（`_wall_phase` 语义，四案例一致）**：collide → NoDynamics
（固体格点回填**碰撞前**快照）→ 半路反弹 → 体力源项（C）→ stream →
（B/D）顶盖平衡态覆盖 → x 向周期闭合。反弹在 stream 之前，因此壁面位于首层流体
节点外**半格**：`y=0.5`（底）/`y=n_y−1.5`（Poiseuille 顶壁）。

数值设置：`τ = 1.0`（ν = 1/6）、全部算例 `float32`（脚本的剪切波阶数研究另用
`float64` 以避开舍入底）、`u ≤ 0.05`（`Ma = u√3 ≈ 0.09`）、CPU 单线程。
`nx=80, n_z=4`，`n_y=12`（断言算例）；A 为 `n_x=64, n_y=n_z=4, 100` 步；
B/C/D 为 1000 步（收敛性见 §6.2）。

**能量/动量口径提醒**：C 的体力以常量加速度源项 `f_q += w_q·3·c_qx·G` 注入
（Guo 源项的主项，未含 `u·F` 的 `O(Ma²)` 修正），B/D 的顶盖是最上层流体面写
`u_top` 的**平衡态覆盖**，不是 Zou/He 动量壁；两者都是本模块自带的边界实现，
本节只测量它们对解析解的偏差，不主张库内 `guo_body_force_d3q19` 或任何墙函数被覆盖。

## 3. 结果

下表由 `scripts/verify_lbm_analytic_convergence.py` 直接调用测试模块函数生成
（`artifacts/lbm_analytic_verification/tables.md`，非手工抄录）：

Device `cpu`, asserted dtype `float32`, machine `loongarch64`, test tolerance 1 %.

| case | analytic quantity | numerical | analytic | rel. error |
|---|---|---|---|---|
| A `nx=64, 100 steps` | $\nu=(\tau-\frac12)/3$ | 0.166662166 | 0.166666667 | 2.700e-03 % |
| C `n_y=12, 1000 steps` | $u_{peak}$ | 0.01241680 | 0.01237500 | 0.3378 % |
| D `n_y=12, 1000 steps` | $C_f=2\nu/(L U)$ | 0.63489495 | 0.63492063 | 4.045e-03 % |
| B `n_y=12, 1000 steps` | $u_x(y)$ profile | — | — | 1.337e-03 % |

| case A: nx | $k\Delta x$ | ν measured (float64) | ν exact | rel. error |
|---|---|---|---|---|
| 16 | 0.3927 | 0.166659224902 | 0.166666667 | 4.465e-03 % |
| 32 | 0.1963 | 0.166666206336 | 0.166666667 | 2.762e-04 % |
| 64 | 0.0982 | 0.166666637970 | 0.166666667 | 1.722e-05 % |

| case C: $n_y$ | $H$ | $u_{peak}$ numerical | $u_{peak}$ analytic | rel. error |
|---|---|---|---|---|
| 8 | 6 | 0.01226864 | 0.01215278 | 0.9534 % |
| 12 | 10 | 0.01241680 | 0.01237500 | 0.3378 % |
| 16 | 14 | 0.01245551 | 0.01243622 | 0.1691 % |

| case D: $n_y$ | $u$ first node | $C_f$ numerical | $C_f$ analytic | rel. error |
|---|---|---|---|---|
| 8 | 0.00384609 | 1.02562507 | 1.02564103 | 1.556e-03 % |
| 12 | 0.00238085 | 0.63489446 | 0.63492063 | 4.123e-03 % |
| 16 | 0.00172402 | 0.45973891 | 0.45977011 | 6.786e-03 % |

| case B: $n_y$ | profile rel. error |
|---|---|
| 8 | 5.024e-04 % |
| 12 | 1.326e-03 % |
| 16 | 2.234e-03 % |

| steady budget (n_y=12) | 1000 steps | 2000 steps |
|---|---|---|
| B_couette | 1.337e-03 % | 1.326e-03 % |
| D_friction | 4.045e-03 % | 4.123e-03 % |

| negative control | measured signal | test tolerance |
|---|---|---|
| A run at $\tau=1.05$ vs $\nu(\tau=1)$ | 9.877 % | 1 % → detected |
| B Couette with bounce-back disabled | 95.238 % | 1 % → detected |

Runtime 176.3 s on one CPU thread, observed order: A ['4.01', '4.00'], C ['2.03', '2.06'].

![解析解验证收敛性](../artifacts/lbm_analytic_verification/convergence.png)

*图：左 = 剪切波粘度的 `kΔx` 收敛（float64，实测斜率 4.0）；中 = 三个槽道案例的
高度细化（浮点断言 dtype）；右 = `n_y=12` 的 Poiseuille 剖面与解析抛物线叠加。*

读法：

* **A（剪切波）**：float64 下 `4.465e-3 → 2.762e-4 → 1.722e-5 %`，`kΔx` 减半时
  误差降 16 倍 ⇒ 实测阶数 **4.01 / 4.00**，即该模式的色散误差是 `O(k⁴)`，
  比 `O(k²)` 更强。float32 断言路径把误差钉在 `~3e-3 %` 的舍入底上，
  因此 1% 判据对该案例有 3 个数量级以上的裕度。
* **C（Poiseuille）**：`0.9534 → 0.3378 → 0.1691 %`，随 `H = n_y−2`
  以 **2.0 阶**下降（实测 2.03 / 2.06），符合半路反弹 + 一阶源项下
  剖面量的预期二阶精度。
* **B/D（Couette 剖面与壁面摩擦）**：线性剖面在格点上**逐点精确**
  （离散 Laplace 作用于线性场为零），实测残差是 `float32` 舍入与
  未完全衰减瞬态的混合（`5e-4…2.2e-3 %`），因此它们是关于**无滑移/动壁闭合**的
  锐利判据，而不是离散阶数研究。

## 4. 负控（判据有鉴别力吗）

两条负控作为脚本的 gate 与 §3 同表输出：

| 负控 | 扰动 | 实测信号 | 判据 1% |
|---|---|---|---|
| A | `τ = 1.05` 跑剪切波，与 `ν(τ=1.00)` 比 | **9.877 %** 偏差 | 检出（≈10× 裕度） |
| B | 保留 NoDynamics 但**关掉壁面反弹** | Couette 剖面误差 **95.238 %** | 检出（≈95× 裕度） |

另有仓库内的负控用例
`test_no_dynamics_restore_must_use_pre_collision_snapshot`：把固体格点回填改成
**碰撞后**快照，200 步后场量差必须 > 1e-3（现测通过）。三条负控合起来说明
1% 的阈值不是装饰——写错反弹/回填顺序会被抓到。

## 5. 阈值与裕度

| 用例 | 判据 | 断言算例实测 | 裕度 |
|---|---|---|---|
| A 剪切波 ν | 1 % | 2.700e-03 % | ≈370× |
| B Couette 剖面 | 1 % | 1.337e-03 % | ≈750× |
| C Poiseuille 峰值 | 1 % | 0.3378 % | **≈3×** |
| D 壁面 `C_f` | 1 % | 4.045e-03 % | ≈247× |

C 的裕度刻意只有 ~3×：它是唯一由**离散误差主导**的案例，随 `H` 细化会自然收紧
（`n_y=16` 时 0.169 %，裕度 ≈6×）；把阈值放宽到"好看"会掩盖回归。

## 6. 注意点 / 局限

1. **只有 `τ = 1`。** 所有断言算例都在 ν = 1/6 上。BGK 的 τ 依赖性（半路反弹的
   τ 相关滑移）**未覆盖**；τ 扫描是下一增量（`shear_wave_case(..., tau=...)` 已留口，
   负控就用它跑 `τ=1.05`）。
2. **低 Mach。** `u ≤ 0.05`（`Ma ≈ 0.09`），`O(Ma²) ≈ 8e-3`（0.8%）与 C 的
   0.34% 同量级，因此**不能**仅凭单点误差声称"压缩性可忽略"。可分离的依据是：
   `u_max` 在细化中固定，而误差随 `H` 以 2 阶下降——`Ma²` 偏置不随网格变，
   既然误差随网格变小，主导项就是离散误差。
3. **B/D 不是阶数研究。** 线性剖面的精确性使这两个案例的残差停在舍入/瞬态量级；
   它们检验的是闭合（无滑移、壁面梯度提取），不要把它们当作二阶精度的证据。
4. **`artifact` 里的 `steady_budget` 是必要的诚实项。** `n_y=12` 上 1000 与 2000 步
   的 B/D 误差比值为 0.99 / 1.02，说明断言预算已到稳态。但 `n_y=16` 的 Couette
   瞬态在 1000 步**未完全衰减**（同函数实测：`C_f` 误差 8.05e-2 % @1000 步 vs
   6.79e-3 % @2000 步，两者都在 1% 内）；因此 §3 的高度细化统一用 2000 步
   （`CHANNEL_STEPS`）。该探针是一次性调用，不在 `convergence.json` 内，
   可用 `friction_case(cpu, ny=16, steps=1000)` 复现。
5. **float32 是断言路径。** 断言在 CPU/float32 上裁决；float64 只用于 §3 的
   剪切波阶数研究（否则 `1e-2 %` 量级的舍入底会把阶数淹没）。若在 GPU/SDAA 上
   跑 `TENSORLBM_TEST_DEVICE=cuda:0`，dtype 仍是 float32，判据不变。
6. **x 向周期是"复制倒数第二列"的闭合**（`f[..., 0] = f[..., -2]`），不是通用周期
   BC；对这四个 x 均匀案例无害（被比较量取 `x = nx//2` 切片，且沿 x 无梯度），
   但不要把它当作可复用的出入口边界条件。
7. **壁面剪应力来自速度梯度**（壁面 + 首节点的单侧差分），不是 Ladd 动量交换。
   本变更**未**做动量交换交叉核对；若要做，必须先对齐"反弹在 stream 前/后"的
   约定，否则两者差 30% 量级纯属约定错配。
8. **不是求解器改动。** `src/` 与 main 字节一致；若未来重构破坏这些案例，
   测试会红——这正是目的。

## 7. 复现

```bash
# 断言层（5 个用例：A/B/C/D + 回填顺序负控）
python -m pytest -q tests/test_lbm_verification.py        # 5 passed

# provenance 层：重算本文所有数字 + 图 + 表，并自检（canonical < 1%、细化收敛、负控被检出）
python scripts/verify_lbm_analytic_convergence.py         # 退出码 0 = 全部通过
```

产物（与本文件同一提交、由上面第二条命令重新生成，非手改）：

| 路径 | 内容 |
|---|---|
| `artifacts/lbm_analytic_verification/convergence.json` | §3/§4/§5 的每一个数字（含 canonical、细化、`steady_budget`、负控、实测阶数） |
| `artifacts/lbm_analytic_verification/convergence.png` | 上图（3 联图） |
| `artifacts/lbm_analytic_verification/tables.md` | §3 的 markdown 表（本文直接内联该文件） |

脚本在本机（loongarch64、CPU 单线程）整跑 **176 s**。