# D3Q27 MRT 物理验证：把「内部一致」升级为「对解析解有量化的正确」

> 仓库 TensorLBM · 模块 `src/tensorlbm/d3q27_mrt_physical_validation.py` ·
> 测试 `tests/test_d3q27_mrt_physical_validation.py`（20 项）·
> 审计工具 `tools/d3q27_mrt_physical_validation_audit.py` ·
> 工件 `artifacts/d3q27_mrt_physical_validation/`（audit.json + 3 图）
> **证据等级**：REAL（对闭式解析解的定量对标），不是 MOCK/STUB。
> **总体判定**：全部 22 项检查 PASS，物理锚点最大相对误差见下方生成表
> `max_physics_rel_error`（< 3 % 的可接受线）。

## 1. 案例介绍（物理背景 / 几何 / 参数表）

### 1.1 为什么要做这件事

`docs/d3q27_composition_evidence_r1.md`（组件级）与
`docs/d3q27_full_composition_test.md` / `src/tensorlbm/d3q27_full_composition_test.py`
（组合级）已经把 D3Q27 MRT 的**可执行性与内部一致性**钉住了：平衡态是不动点、
质量/动量守恒、bounce-back 对合性、力-动量平衡、确定性、源码哈希。这些证据证明了
"算子跑得通且自洽"，但**没有一条把它和真实流体力学方程对上**——一个把
`ν = (τ−1/2)/3` 写成 `ν = 2(τ−1/2)/3` 的 bug 会完整通过上述所有检查。

本工作补的正是这条验证欠账：用**闭式解析解**（精确的 Navier–Stokes 解，不是文献
拟合曲线）裁决 D3Q27 MRT 的 collide+stream 组合、输出组合（`macroscopic27`）和
bounce-back 壁面组合，并给出逐项误差。

### 1.2 三个解析锚点 + 一个守恒律

| # | 锚点 | 物理设置 | 闭式解 | 观测量 |
|---|---|---|---|---|
| A1 | 平面剪切波 | 周期立方盒，`u_x = A sin(k y)`，`k = 2π/n` | 精确非定常 NSE 解（对流项恒为 0）：`E(t) = E(0) e^{-2νk²t}`，`ν = (τ−1/2)/3` | 有效运动粘度 `ν_eff` |
| A2 | 一维线性声波 | 周期盒，`ρ' = A sin(k x)`，`u_x = c_s A sin(k x)`（一阶声学关系） | `ω = c_s k`，`c_s = 1/√3`（格子单位） | 声速 `c_s` 与衰减率 |
| A3 | 双壁无滑移本征模 | `y=0`、`y=ny−1` 两排静止固体（`bounce_back_cells_27`），x/z 周期 | `u_x = A sin(π(y−y_w)/H) e^{-ν(π/H)²t}`，`H` 为两壁面间距 | 衰减率 → 壁面有效位置 `y_w` |
| A4 | 周期守恒律 | 任意正分布，周期 collide+stream，200 步，float64 | 质量、动量是精确不变量 | 相对漂移 |

参数表（默认审计配置）：

| 参数 | 符号 | 取值 |
|---|---|---|
| 格子 | — | D3Q27（`C`，27 方向全权重） |
| 碰撞 | — | MRT（`collide_mrt27`），`s_e=1.19`、`s_eps=1.4`、`s_q=1.2`、`s_pi=s_e` |
| 松弛时间 | τ | 0.65（对照 0.68 / 0.75） |
| 解析运动粘度 | ν | `(τ−1/2)/3 = 0.05`（格子单位） |
| 剪切波网格 | n | 16 / 24 / 32（另 12 用于长期稳定性） |
| 声波网格 | n | 16 / 32 |
| 壁面算例 | ny | 16 / 24 / 32（图 3 另含 40），nx=nz=4 |
| 初始扰动幅值 | A | 2D 剪切 0.01；声波 1e-3（线性区） |
| 精度 | — | 全程 `float64`、CPU |
| 采样 | — | 剪切/壁面每 20 步，声波每 5 步 |

## 2. 怎么做的（方法 / 流程 / 关键设置 / 工具链）

1. **不新写求解器**：探针直接调用库内 `collide_mrt27` / `stream27_roll` /
   `bounce_back_cells_27` / `macroscopic27` / `equilibrium27`，避免"验证代码自己
   实现一遍物理"的假证据。测试 `test_streaming_is_used_by_the_anchors` /
   `test_bounce_back_is_used_by_the_wall_anchor` 用 `inspect.getsource` 锁住这一点。
2. **从时间序列反演物性，而不是从公式假设物性**：剪切波用 `log E(t)` 线性拟合
   斜率 `−2νk²`；声波用 `k`-模复系数的相位斜坡反演 `ω`（相位逐步解缠，跨度
   < π），因此声速的测量不预设 `c_s`。
3. **独立交叉方法（k 空间谱）**：用 autograd 取 `J = ∂collide_mrt27/∂f`
   在静止平衡态处的 Jacobian（27×27），与流迁相位因子
   `diag(e^{-i k c_x})` 组成一步线性算子 `Λ(k)`，本征值给出
   `ν = −ln|λ_shear|/k²`、`c_s = arg(λ_sound)/k`、衰减 `−ln|λ_sound|`。
   这条路径**不迭代时间步**，与 (1)(2) 的时域测量互为独立验证。
4. **壁面位置由数据反演**：`H_eff = π/√(rate/ν)`，再由 `H_eff` 反算
   `y_w`。这使"是不是把壁面放错了半个格子"变成可测量的数字，而不是靠
   "剖面看起来像抛物线"。
5. **控制项防空心通过**：BGK 极限（全部非守恒率 `=1/τ` 时 MRT 必须
   bit-for-bit 复现 `collide_bgk27`）、体粘率独立性（`s_e` 从 1.19 变 1.60，
   剪切支不变）、τ 灵敏度（τ→0.68 时测得 ν 必须整体平移约 6 %，证明测量
   不是在返回常数）。
6. **工具链**：

```bash
# 全量审计（本文件表格 + 3 图 + audit.json，实测 ~2 min @8 线程 CPU）
python tools/d3q27_mrt_physical_validation_audit.py

# CI 快速冒烟（单网格，无图）
python tools/d3q27_mrt_physical_validation_audit.py --quick

# 判据测试
pytest -q tests/test_d3q27_mrt_physical_validation.py
```

文中所有数字由审计工具程序化注入（下表即 `audit.json` 的渲染），零手抄。

## 3. 计算结果

<!-- BEGIN GENERATED BLOCK (tools/d3q27_mrt_physical_validation_audit.py) -->
| check | aspect | measured | reference | rel. error | tolerance | status |
|---|---|---|---|---|---|---|
| `shear_viscosity_grid16` | streaming_collision_coupling/viscosity | 0.0504457 | 0.05 | 0.8915 % | 3.0000 % | **PASS** |
| `shear_viscosity_grid24` | streaming_collision_coupling/viscosity | 0.0501975 | 0.05 | 0.3950 % | 3.0000 % | **PASS** |
| `shear_viscosity_grid32` | streaming_collision_coupling/viscosity | 0.050111 | 0.05 | 0.2220 % | 3.0000 % | **PASS** |
| `sound_speed_grid16` | output/acoustics | 0.575763 | 0.57735 | 0.2748 % | 3.0000 % | **PASS** |
| `sound_speed_grid32` | output/acoustics | 0.576962 | 0.57735 | 0.0673 % | 3.0000 % | **PASS** |
| `spectral_vs_timedomain_viscosity_grid16` | cross_method/viscosity | 0.0504457 | 0.0504457 | 0.0000 % | 1.0000 % | **PASS** |
| `spectral_vs_timedomain_sound_speed_grid16` | cross_method/acoustics | 0.575763 | 0.575763 | 0.0000 % | 1.0000 % | **PASS** |
| `spectral_vs_timedomain_attenuation_grid16` | cross_method/attenuation | 0.010987 | 0.0109897 | 0.0247 % | 5.0000 % | **PASS** |
| `spectral_vs_timedomain_viscosity_grid32` | cross_method/viscosity | 0.050111 | 0.050111 | 0.0000 % | 1.0000 % | **PASS** |
| `spectral_vs_timedomain_sound_speed_grid32` | cross_method/acoustics | 0.576959 | 0.576962 | 0.0005 % | 1.0000 % | **PASS** |
| `spectral_vs_timedomain_attenuation_grid32` | cross_method/attenuation | 0.00274397 | 0.00275056 | 0.2396 % | 5.0000 % | **PASS** |
| `wall_mode_decay_ny16` | wall_treatment/no_slip | 0.0025252 | 0.00251776 | 0.2955 % | 3.0000 % | **PASS** |
| `wall_position_ny16` | wall_treatment/effective_wall_offset | 0.510319 | 0.5 | 2.0639 % | 5.0000 % | **PASS** |
| `wall_mode_decay_ny24` | wall_treatment/no_slip | 0.0010208 | 0.00101959 | 0.1187 % | 3.0000 % | **PASS** |
| `wall_position_ny24` | wall_treatment/effective_wall_offset | 0.506524 | 0.5 | 1.3049 % | 5.0000 % | **PASS** |
| `wall_mode_decay_ny32` | wall_treatment/no_slip | 0.00054866 | 0.000548311 | 0.0636 % | 3.0000 % | **PASS** |
| `wall_position_ny32` | wall_treatment/effective_wall_offset | 0.50477 | 0.5 | 0.9540 % | 5.0000 % | **PASS** |
| `periodic_mass_conservation` | conservation/mass | 1.64284e-14 | 0 | 0.0000 % | 0.0000 % | **PASS** |
| `periodic_momentum_conservation` | conservation/momentum | 5.23207e-13 | 0 | 0.0000 % | 0.0000 % | **PASS** |
| `bgk_limit_equivalence` | control/bgk_limit | 5.77316e-15 | 0 | 0.0000 % | 0.0000 % | **PASS** |
| `bulk_rate_independence` | control/bulk_rate | 3.27828e-09 | 0 | 0.0000 % | 0.1000 % | **PASS** |
| `tau_sensitivity_control` | control/tau_sensitivity | 0.060504 | 0.06 | 0.8400 % | 1.0000 % | **PASS** |
<!-- END GENERATED BLOCK -->

要点（数值见上表对应 id）：

- **剪切支**：`shear_viscosity_grid16/24/32` 的误差随网格单调下降，且
  `ν_eff/ν − 1` 与 `k²` 成正比（`test_shear_viscosity_trend_matches_k_squared`），
  即误差是**空间截断**，不是模型系数错。拟合出的截断系数
  ≈ `0.058 k²`（τ=0.65，本格子/D3Q27），是离散阻尼的固有性质。
- **声学支**：`sound_speed_grid16/32` 误差都远小于 3 %，且同样随 `k²` 收敛；
  声衰减与 k 空间本征值一致（`spectral_vs_timedomain_attenuation_*`）。
- **双壁无滑移**：`wall_mode_decay_ny16/24/32` 的衰减率与 `H = ny − 2` 的解析
  本征值符合；`wall_position_ny*` 反演出 `y_w` 收敛到 **0.5 格**——这是 full-way
  bounce-back 的正确壁面约定（壁面在固体格心向流体侧半个格距处），不是"调参数
  调出来的"。
- **守恒律**：`float64` 下 200 步质量/动量相对漂移都在舍入量级。
- **控制项**：BGK 极限差 1e-15 量级；`s_e` 改变对剪切支无影响（1e-9）。

## 4. 结果图

| 图 | 内容 | 读法 |
|---|---|---|
| `artifacts/d3q27_mrt_physical_validation/figures/shear_wave_decay.png` | 左：三档网格的 `E(t)/E(0)`（标记=LBM，实线=解析 `e^{-2νk²t}`）；右：与解析解的绝对偏差（对数轴） | 曲线与解析线重合，偏差随时间缓慢上升即截断误差累积，不是稳定性问题 |
| `artifacts/d3q27_mrt_physical_validation/figures/eigenvalue_spectrum.png` | 左：`Λ(k)` 的 27 个本征值复平面分布；右：ν、`c_s`、衰减随网格的变化 | 守恒模在 λ=1，剪切模在实轴（纯衰减），声模共轭对（`±c_s k`） |
| `artifacts/d3q27_mrt_physical_validation/figures/wall_offset_inference.png` | 左：壁面本征模能量衰减 vs 解析；右：反演壁面偏移随 `ny` 的收敛 | 右图收敛到 0.5 即验证了壁面约定；若实现把壁面放在格心，点会落在 0 附近 |

## 5. 注意点（踩过的坑 / 局限 / 适用范围）

1. **固体格内必须 NoDynamics**：组合步按 `step27` 的约定写
   （`torch.where(mask, f, collide(f))` → stream → bounce-back）。若对固体格也做
   MRT 碰撞，壁面本征模衰减率会被抬高（本工作实测：ny=16 时误差由 +0.3 %
   恶化到 +15 %）。这是壁面组合的**正确用法约束**，已在模块 docstring 写明。
2. **壁面宽度 `H = ny − 2` 而非 `ny − 1`**：`bounce_back_cells_27` 是 full-way
   bounce-back（在固体格内反射），无滑移平面在固体格心向流体侧 0.5 格。用
   `H = ny − 1` 做参考会把 15 % 的"约定错"误判成"物理错"——本工作第一版就踩了
   这个坑，靠反演 `y_w` 才发现。
3. **声波初始化必须单向**：`ρ' = A sin(kx)` 单独给会形成驻波，k-模相位不呈
   线性斜坡（`unwrap` 结果无意义）；必须同时给 `u_x = c_s A sin(kx)` 才是
   右行波。驻波版本的错误实现在本工作第一次尝试中把 `c_s` 测成 0.297（−150 %）。
4. **动量的相对漂移要有物理量纲**：初值若取"静止+噪声"，总动量为噪声量级，
   用 1.0 归一化会把舍入误差放大成"看起来不守恒"。故守恒检查用带均匀漂移
   `u = 0.05` 的状态，动量尺度是物理量。
5. **适用范围**：单相、静止几何、均匀周期网格、CPU `float64`、`Re = O(1)`
   格子单位。**不覆盖**壁函数/湍流模型、多相与自由面、运动边界与 AMR、
   加速器后端、高雷诺数长期稳定性——这些在 artifact 的 `withheld` 字段显式列出。
6. **不改变能力矩阵**：本证据不把 `general_capability_matrix` 的 D3Q27
   `WITHHELD_D3Q27_COMPOSITION` 翻成 SUPPORTED。那条判据要的是**平台级完整
   组合路径**的证据，物理锚点只解除"没有物理论证"这一条质疑，须由平台组合
   证据持有者合并判定。

## 6. 误差判定（关键量 vs 基准，逐项）

| 关键量 | 基准 | 误差（见生成表） | 是否 < 3 % |
|---|---|---|---|
| 有效运动粘度 `ν_eff`（τ=0.65） | `(τ−1/2)/3 = 0.05` 解析 | `shear_viscosity_grid16/24/32` | **是**（最粗网格亦 < 1 %） |
| 声速 `c_s` | `1/√3 = 0.57735` 解析 | `sound_speed_grid16/32` | **是**（≤0.28 %） |
| 壁面本征模衰减率 | `ν(π/H)²`，`H = ny−2` | `wall_mode_decay_ny16/24/32` | **是**（≤0.30 %） |
| 壁面有效位置 `y_w` | 0.5 格（full-way BB 约定） | `wall_position_ny16/24/32` | **是**（相对偏差 ≤2.1 %，收敛到 0.5） |
| k 空间本征值 vs 时域测量 | 互为独立方法 | `spectral_vs_timedomain_*` | **是**（ν/c_s ≤0.001 %，衰减 ≤0.24 %） |
| 质量守恒（float64, 200 步） | 精确不变量 | `periodic_mass_conservation` | **是**（1e-14 量级） |
| 动量守恒（float64, 200 步） | 精确不变量 | `periodic_momentum_conservation` | **是**（1e-12 量级） |
| τ 灵敏度控制（τ=0.68） | `(0.68−1/2)/3 = 0.06` | `tau_sensitivity_control` | **是**（< 1 %，且相对基线平移 > 4 %） |

结论：D3Q27 MRT 的 collide+stream 组合、输出组合与静态 bounce-back 壁面组合
在单相层流解析基准下**全部关键量 < 3 %**，误差来源可解释为空间截断并随网格
收敛；无任何锚点需要"放宽判据才通过"。