# 二维 Couette–Poiseuille 混合驱动通道流 — 解析叠加剖面验证（couette_poiseuille）

> 2D Mixed Couette–Poiseuille Channel Flow: Sliding Plate + Body Force, Verified Against the Analytic Superposition

**摘要** — TensorLBM D2Q9 BGK 求解器在「上壁以 `u_wall` 滑动 + 恒定体积力（等效恒定压力梯度）驱动」的平行平板通道上的解析验证。存档档 `H=32`、12000 步：壁法向剖面 L2 相对误差 **7.40e-04**、剖面最大偏差为峰值的 0.067%、速度极值位置的离散峰与解析极值点相差 1 个格子（**1.56%**）、动壁节点速度相对误差 **0.05%**、截面平均速度相对误差 **0.06%**、流体域质量漂移 **6.5e-05**（float32 舍入量级）。真实模拟、无外推（extrap: none）、无任何修正因子。

本教程与前作的分工：`couette_2d` 只验证线性剪切，`poiseuille_2d` 只验证（压力差驱动的）中心对称抛物线；本案例把两者**叠加**，速度极值**不在通道中心**（自相似驱动下 `y_v/H = 0.75`），因此同时检验「动壁动量交换」与「体积力强迫」两条实现路径。只实现其中一条时剖面形状仍近似正确，但平均速度会系统性偏低 `u_wall/2`，极值位置会退回 `y_v/H = 0.5` 或 `1.0`——这正是本教程断言能抓住的失败模式。

## 1. 物理与数学背景

二维不可压通道流，`y ∈ [0, H]`：下壁静止（无滑移 `u(0) = 0`），上壁以 `U_w` 沿 x 方向滑动（`u(H) = U_w`），流体受单位质量体积力 `g`（等效压力梯度 `dp/dx = -ρg`）。稳态解为 Couette 线性解与 Poiseuille 抛物线的叠加：

```
nu * u''(y) + g = 0,      u(0) = 0,      u(H) = U_w
u(y)      = g/(2*nu) * y * (H - y) + U_w * y / H
y_v       = H/2 + nu*U_w/(g*H)              （速度极值位置，偏离中心）
u_max     = u(y_v)
ubar      = g*H^2/(12*nu) + U_w/2           （截面平均速度）
Re        = ubar*H/nu,     Ma = u_max/c_s,  c_s = 1/sqrt(3)
```

本案例用**自相似驱动** `g = 4*ν*U_w/H²`，于是 `y_v/H = 3/4` 与网格无关，网格加密研究才是在细化同一个问题。两个退化分支都被断言覆盖：`g = 0`（`--accel 0`）→ 纯 Couette 线性剖面；`U_w → 0` → 纯 Poiseuille 中心对称抛物线。

**数值方法**（只用库原语，无修正因子）

- D2Q9 格子、BGK 单弛豫碰撞；`tensorlbm.d2q9.equilibrium` / `macroscopic`、`tensorlbm.solver.stream`。
- 体积力用二阶 Guo (2002) 强迫：平衡态速度半步移位 `u* = u + F/(2ρ)`，外加离散源项
  `S_i = (1 - 1/(2τ)) w_i [3(c_ix - u_x) + 9(c_i·u)c_ix] F_x`。
- 壁面：**前流迁移（pre-streaming）半程反弹**，通过两行 ghost row 实现，使无滑移面精确位于 `y = 0.5` 与 `y = ny - 1.5`，有效缝高 `H = ny - 2`；上壁 ghost row 额外加入 Ladd 动量交换项 `-2 w_i ρ (c_i·U_w)/c_s²`。
- 流向周期（无进出口闭合误差），因此比对是纯粹的壁法向一维验证。

**参考解** — 上述解析叠加解（Couette–Poiseuille 混合流，工科流体力学教科书精确解）。解析值在**与半程反弹一致的壁面位置**（`y* = j - 0.5`，`H = ny - 2`）上取值，不做任何重新标定。

**参考文献**

- Guo, Z., Zheng, C. & Shi, B. (2002). Discrete lattice effects on the forcing term in the lattice Boltzmann method. *Phys. Rev. E* **65**, 046308.
- Ladd, A. J. C. (1994). Numerical simulations of particulate suspensions via a discretized Boltzmann equation. *J. Fluid Mech.* **271**, 285–309.（运动壁面动量交换）
- Succi, S. (2001). *The Lattice Boltzmann Equation for Fluid Dynamics and Beyond*. Oxford University Press.

## 2. 计算条件设置

存档档参数取自 `benchmarks/verified/couette_poiseuille/result.json`（程序化注入，下同）。

| 参数 | 取值 | 说明 |
|---|---|---|
| 格子 / 碰撞 | D2Q9 / BGK | `d2q9.equilibrium`、`d2q9.macroscopic`、`solver.stream` |
| 强迫 | Guo (2002) 体积力 | `F = ρg`，`S_i` 见上 |
| 下壁 / 上壁 | 无滑移半程反弹 / Ladd 动壁 | ghost row `0` 与 ghost row `ny-1` |
| 网格 | `ny = 34`、`nx = 8` | 有效缝高 `H = ny - 2 = 32`，流向周期 |
| `τ` / `ν` | 0.8 / 0.1 | `ν = (τ - 0.5)/3` |
| 动壁速度 `U_w` | 0.02 | 格子单位 |
| 体积力 `g` | 7.8125e-06 | 自相似驱动 `4νU_w/H²`，令 `y_v/H = 0.75` |
| 步数（存档档） | 12000 | 每 1000 步记录剖面变化，判稳门 1e-3 |
| Re / Ma | 5.3 / 0.039 | `Re = ubar H/ν`，`Ma = u_max/c_s` |

## 3. 软件使用步骤

**环境** — Python ≥ 3.11；依赖 torch / numpy / matplotlib；仓库 `src` 可导入（`pip install -e .` 或 `PYTHONPATH=src`）。

**步骤 1**：获取仓库并进入根目录

```bash
git clone https://github.com/cxs503/TensorLBM.git && cd TensorLBM
```

**步骤 2**：安装/指向 tensorlbm 包

```bash
pip install -e .   # 或 export PYTHONPATH=$PWD/src
```

**步骤 3**：快档复现（`ny = 18`、3000 步，CPU 约 5 s）

```bash
python benchmarks/verified/couette_poiseuille/run.py single /tmp/cp_fast.json --ny 18 --steps 3000
```

**步骤 4**：入库存档档复现（`ny = 34`、12000 步，CPU 约 25 s）

```bash
python benchmarks/verified/couette_poiseuille/run.py single benchmarks/verified/couette_poiseuille/result.json
```

脚本把全部指标与门限写进 JSON，任一 `checks.*.ok == false` 时**退出码非零**——这是本教程的机器可裁部分。

**步骤 5**：网格加密 + 纯 Couette 对照扫描

```bash
python benchmarks/verified/couette_poiseuille/run.py scan /tmp/cp_scan
```

**步骤 6**：出图（`u_x` 云图 + 剖面比对 + 偏差曲线）

```bash
python docs/benchmarks/demos/couette_poiseuille_demo.py \
    --out docs/benchmarks/tutorials/figs/couette_poiseuille/fields.png
```

### 教程可执行片段（含断言）

下面这段是教程的「一页版」：跑快档、把全部断言跑一遍，任一条不满足即 `AssertionError`（门限与 `run.py` 的 `TOL_*` 常量一致）。

```python
import importlib.util

spec = importlib.util.spec_from_file_location(
    "cp", "benchmarks/verified/couette_poiseuille/run.py"
)
cp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cp)

res = cp.run_case(ny=18, steps=3000)          # CPU 约 5 s，accel 默认自相似
met, par = res["metrics"], res["params"]

assert res["all_ok"], res["checks"]                       # 文档化的 7 道门（见 §4 表 1）
assert met["profile_l2"] < 0.02                           # profile_l2
assert met["peak_pos_rel_err"] < 0.05                     # peak_position
assert met["moving_wall_uc_rel_err"] < 0.03               # moving_wall_uc
assert met["ubar_rel_err"] < 0.02                         # mean_velocity
assert met["mass_drift"] < 1e-3                           # mass_drift
assert 0.5 < met["peak_y_star"] / par["H"] <= 1.0         # 极值偏离中心

couette = cp.run_case(ny=18, steps=3000, accel=0.0)       # g = 0 对照
assert couette["all_ok"] and couette["metrics"]["profile_l2"] < 0.02
print(f"L2={met['profile_l2']:.3e}  peak_y*/H={met['peak_y_star'] / par['H']:.3f}")
```

### 教学内核：时间推进的 6 行

`run.py` 的主循环只做这几件事，全部是可读的库原语：

```python
rho, ux, uy = macroscopic(f)                       # 矩量
ux_star = ux + 0.5 * accel / rho                   # Guo 半步力移位
feq = equilibrium(rho, ux_star, uy)                # 平衡态
f = f + (feq - f) / tau + guo_source(ux_star, accel, tau, dev)   # 碰撞 + 强迫源
apply_bounce_back(f, rho, u_wall, dev)             # 两壁：下无滑移 / 上动壁
f = stream(f)                                      # 迁移（流向周期）
```

## 4. 计算结果

**表 1 入库存档结果（`ny=34`、12000 步，CPU）与 7 道门**

| 门（`checks` 键） | 量 | 数值 | 门限 | 通过？ |
|---|---|---|---|---|
| `profile_l2` | 壁法向剖面 L2 相对误差（32 点） | 7.40e-04 | 2% | ✅ |
| `peak_position` | 极值位置 `y*`（数值 / 解析） | 24.5 / 24.0（差 1 格）= 1.56% | 5% | ✅ |
| `moving_wall_uc` | 动壁节点速度（数值 / 解析） | 0.0203 / 0.020303 = 0.05% | 3% | ✅ |
| `mean_velocity` | 截面平均速度（数值 / 解析） | 0.016657 / 0.016667 = 0.06% | 2% | ✅ |
| `mass_drift` | 流体域质量漂移 | 6.5e-05 | 1e-3 | ✅ |
| `steady_state` | 末 1000 步剖面变化 / 峰值 | 6.63e-07 | 1e-3 | ✅ |
| `finite` | 全部分布函数有限 | True | True | ✅ |

补充量：剖面最大偏差 1.497e-05（峰值的 0.067%）；固定壁邻节点速度 0.000918 vs 解析 0.000928；`Re = 5.3`、`Ma = 0.039`；墙钟耗时 24.2 s（单核 CPU，12000 步）。

*来源：`benchmarks/verified/couette_poiseuille/result.json`（含 32 点 `u_num` / `u_exact` / `y_star` 数组）。*

![图 1 存档档（34×8 × 12000 步）三连图：左=`u_x` 云图（上壁滑动拖出剪切、体积力把剖面鼓成偏离中心的形状；壁面位于 y=0.5 与 y=ny−1.5）；中=壁法向剖面，LBM 标记与解析叠加解（实线）重合，灰虚线为纯 Couette 对照，红点线为解析极值点 y_v；右=LBM 与解析的逐点偏差，量级 1e-5（剖面 L2 相对误差 7.40e-04）。](figs/couette_poiseuille/fields.png)

*图 1 存档档（34×8 × 12000 步）三连图：左=`u_x` 云图；中=壁法向剖面（LBM vs 解析叠加解，灰虚线为纯 Couette 对照，红点线为解析极值点 y_v）；右=逐点偏差。*

## 5. 与解析解的比较

**表 2 关键量 vs 解析解（逐项相对误差与是否 <3%）**

| 量 | 数值解 | 解析解 | 相对误差 | <3%？ |
|---|---|---|---|---|
| 剖面 L2 范数误差 | — | — | 7.40e-04 | ✅ 是 |
| 剖面最大偏差（/峰值） | — | — | 0.067% | ✅ 是 |
| 极值位置 `y*` | 24.5 | 24.0 | 1.56% | ✅ 是（=1 个格子，离散分辨率极限） |
| 动壁节点速度 | 0.020293 | 0.020303 | 0.05% | ✅ 是 |
| 固定壁邻节点速度 | 0.000918 | 0.000928 | 0.04%（以峰值为基准） | ✅ 是 |
| 截面平均速度 | 0.016657 | 0.016667 | 0.06% | ✅ 是 |

**网格趋势**（`run.py scan`，每档按 `H²/ν` 比例给步数，驱动取自相似使 `y_v/H` 恒定）：`ny=18`/3000 步 L2 = 2.04e-03，`ny=34`/12000 步 L2 = 7.40e-04（加密 2 倍、误差降 2.8 倍，接近二阶）。误差在 `10⁻³` 量级**饱和**，主导项不是网格分辨率，而是（a）离散强迫项的二阶截断与（b）`Ma ≈ 0.04` 的可压缩误差。归档只保留上面两个网格点；`scan` 子命令另有 `ny=66`/48000 步一档（单核约 5–8 min），未纳入本表的判定。

**误差判定（显式回答）**：表 2 每一项关键量相对误差均 **< 3%**（最大项 1.56%，且该项受 1 个格子离散分辨率支配）；剖面 L2 误差 0.074% 远低于 2% 门限。结论：本案例在存档档下通过解析验证。

## 6. 注意点 / 局限 / 适用范围

1. **离散峰位置受格距限制**：解析极值 `y_v/H = 0.75` 即 `y* = 24.0`，而 `H = 32` 时离散节点只能取 `y* = j - 0.5`（最近 23.5 / 24.5），故 `peak_y_star = 24.5`、位置误差恒为 0.5 个格子 = 1.56%。这是分辨率极限而非实现错误；若要更细的峰位置判定，需插值或加密网格，**本教程没有对位置量做插值**。
2. **质量漂移是 float32 舍入**：闭合通道内流体域质量严格守恒，实测漂移 ≈ 5e-9/步累积到 6.5e-05，门限因此设为 1e-3；不要把它解读为物理泄漏。改用 float64 或减少步数可线性降低该量。
3. **单相、低 Re、低 Ma 范围**：`Re = 5.3`、`Ma = 0.039`，验证的是求解器与强迫/动壁实现，不代表湍流或可压缩能力。`Ma` 更高时 BGK 的可压缩误差会进入百分级，需要 MRT/TRT 或格子声速修正。
4. **`H = ny - 2` 是必须核对的口径**：前流迁移半程反弹把壁面放在 `y = 0.5` 与 `y = ny - 1.5`；若误用 `H = ny - 1` 或 `ny`，解析斜率与 `ubar` 会偏移 3%–6%，剖面误差从 1e-3 跳到百分级。任何改动 ghost row 处理的 PR 必须重跑本教程。
5. **极值不在中心是本案例的判别力来源**：若漏掉动壁动量项，L2 误差升到 1e-1 量级、`ubar` 偏低约 `U_w/2 = 0.01`（相对 60%）、`peak_y*/H` 退化到 0.5。`tests/test_couette_poiseuille_tutorial.py` 的 `test_mean_velocity_superposition` 与 `test_peak_is_off_centre` 就是为此设置的负例哨兵。
6. **未包含 docx/pdf 导出**：本教程只提供 Markdown。仓库其他教程附带的 docx/pdf 由外部 pandoc 流水线生成，本工作树无 pandoc，因此未伪称已生成。
7. **适用面**：判据只覆盖「单相 + 平行平板 + 单向流」的壁面/强迫实现正确性；曲面动壁（Ladd 曲面、IBM、BFL、倾覆动边界）不在此验证范围内。

## 7. 复现与验收

```bash
# 1) 快档（约 5 s）
PYTHONPATH=src python benchmarks/verified/couette_poiseuille/run.py single /tmp/cp_fast.json --ny 18 --steps 3000

# 2) 机器可裁的教程断言（含纯 Couette 对照），约 15 s
PYTHONPATH=src TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest tests/test_couette_poiseuille_tutorial.py -q
```

`tests/test_couette_poiseuille_tutorial.py` 覆盖：7 道文档化门限、纯 Couette 对照、质量/判稳、入库存档与教程表格数字一致性（防止手抄漂移）、README 与教程索引的链接存在性。