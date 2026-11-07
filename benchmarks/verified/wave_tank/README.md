# 二维波浪水池基准：Stokes 二阶造波 / 消波 + 两点法反射系数 + 质量守恒

**判定: HONEST FAIL（部分判据通过，详见 §5 判据结果表）**
——水池、解析波、探针与两点法分析链均为真实实现并通过单元测试；但造波机工况下
的色散/谐波/反射精度未达阈值，根因已在 §7 定位（含两条已修复的建模与实现缺陷）。

本 benchmark 在 TensorLBM 中建立一个真实求解的二维数值波浪水池：自由面使用仓内既有的
**相场（conservative Allen–Cahn / CAC）两相模型** `tensorlbm.cac_lbm.CACSim`（不新写界面格式），
左端用松弛区（relaxation zone）按 Stokes 二阶解析解造波，右端用同法消波，用探针阵列按
**Goda & Suzuki 两点法**分离入射/反射波，逐项验证色散关系、波形谐波、消波效率与质量守恒。

## 1. 物理背景与几何（格子单位 dx = dt = 1, cs² = 1/3）

- 准二维板 `nz = 3`（z 方向周期），使 D3Q19 在 x–y 平面内退化为各向同性二维格式；
  底/顶/左/右均为壁面（full-way bounce-back），上方留 12 行气相，重力 `g = 6.5e-3` 沿 −y。
- 静水深 `h = kh·λ/2π` 按 kh 选取（表 1），λ ≈ 84–108 格；一阶波幅 a = ε/k ≈ 0.27–1.37 格。
- 入口松弛区 0.7λ + 传播段 0.6λ + 出口松弛区 0.7λ（生产扫描）；
  探针在入口区出口外 0.20λ 与 0.37λ（Δx = 0.17λ，避开 kΔx = nπ/2 退化点）。

| kh | h [格] | λ [格] | c 理论 | T [步] | 域 nx×ny×nz | 网格单元 |
|---|---|---|---|---|---|---|
| 0.7 | 12 | 107.7 | 0.2462 | 437.5 | 217×26×3 | 16 926 |
| 1.5 | 24 | 100.5 | 0.2911 | 345.4 | 202×38×3 | 23 028 |
| 3.0 | 40 | 83.8 | 0.2786 | 300.7 | 170×54×3 | 27 540 |

相场参数：ρ_l/ρ_g = 10，ν = μ_l/ρ_l = 0.008（τ_g = 0.524），界面宽 W = 2 格，σ = 1e-4，
mobility = 0.05，p0 = 0.01。扫描 `kh ∈ {0.7,1.5,3.0} × ε = ka ∈ {0.02,0.08}`，
外加负控（关掉消波区）与网格加密（Δx 减半：h、ν 加倍、g 减半，保持 kh、ε、c 与波雷诺数）。

## 2. 方法

### 2.1 解析波（`src/tensorlbm/wave_tank/stokes2.py`）

    η(x,t) = a cosθ + (k a²/4)·(3−σ²)/σ³·cos2θ,   θ = kx − ωt,  σ = tanh kh

色散 `ω² = g_eff k tanh(kh)`（Newton 反解），速度场 = 一阶 + 与 η 相容的 2ω 势：

    u = aω cosh k(y+h)/sinh kh · cosθ + 2kC cosh 2k(y+h) · cos2θ
    w = aω sinh k(y+h)/sinh kh · sinθ + 2kC sinh 2k(y+h) · sin2θ
    2kC sinh 2kh = 2ω η₂ − a²ωk coth(kh)

C 由**二阶运动学自由面条件** `w = η_t + u η_x`（在 y = η 上展开到 O(a²)）唯一确定，
深水 C → 0（经典深水结果）。该条件由单测 `test_kinematic_free_surface_condition`
以 O((ka)²/σ³) 的机器可算残差校验（6 组 kh/ε 全通过）。

### 2.2 有效重力（本次实测发现的建模要点）

CAC 模型对**两相都施加 ρg**，自由面位移把液体压向气相、气相浮力抵消一部分，
故界面恢复力对应 **g_eff = g(1 − ρ_g/ρ_l) = 0.9 g**。首轮直接用 g 代入解析式时，
实测波长 89 格 vs 理论 100.5（−11%，恰为 k ∝ 1/g 的效应），相位失配导致表观反射系数虚高。
修正后 `wave_of_config` 一律用 g_eff。

### 2.3 造波 / 消波 = 松弛区（`tank.WaveTank`）

    f ← (1−w) f + w f_eq(φ_t, u_t),   g ← (1−w) g + w g_eq(p_t, ρ_t, u_t)
    w(x) = cos²(π ξ / 2L_zone)  （壁面 1，出口 0）

入口目标 = §2.1 的 Stokes 二阶解（幅度按 1.5 周期 cosine ramp 升起），
目标压力取静水压 `p_t = p0 + ρ_t g max(η_s − y, 0)`；出口目标 = 静止平水面。
初始场用静水压平衡的 g 初始化（否则水体自由落体数百步）。

### 2.4 探针与两点法（`gauges.py`）

- 水面高度 = **液柱积分** η(x) = Σ_y φ − η_ref（对对称 tanh 界面严格等于界面中心，
  亚格精度、与界面宽度无关；优于 φ=0.5 等值线）。
- 整数周期稳态窗最小二乘拟合 `mean + a1cos(ωt−φ1) + a2cos(2ωt−φ2)`。
- 两点法解 `c(x) = A e^{−ikx} + B e^{+ikx}`（A 入射 +x，B 反射 −x）得 KR = |B|/|A|；
  2ω 分量用 k₂ = k(2ω) 单独分离。**时间约定必须与拟合一致**（见 §7 陷阱 2）。
- 相速度由两点复合幅相位差求 k（对理论 k 做分支展开），c = ω/k。

## 3. 判据（机器可算）

| 判据 | 内容 | 阈值 |
|---|---|---|
| C1 | 实测相速度 vs 闭式 `c² = (g/k)tanh(kh)`，≥3 个 kh | < 1 % |
| C2a | 一阶谐波入射幅值误差 | < 3 % |
| C2b | 二阶谐波入射幅值误差 | < 10 % |
| C3 | 消波反射系数（松弛区 0.7λ） | < 5 %；负控 > 30 % |
| C4 | 质量相对漂移 | < 0.5 %/周期（整数周期窗） |
| C5 | Δx 减半后波形变化（相位对齐后 RMS/波幅） | < 3 % |
| C6 | 纳入 benchmarks/verified 索引 + pytest 全绿 | — |
| C7 | 单算例墙钟 | < 20 min |

## 4. 结果

| case | kh | eps | c theory | c measured | err % | a1 incident | err % | a2 incident | err % | KR % | mass drift %/T | steps | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kh0.7_eps0.02 | 0.7 | 0.02 | 0.2462 | 0.2670 | 8.45 | 0.3416 | 0.36 | 0.0266 | 30.24 | 10.92 | -0.073 | 2188 | 409 |
| kh0.7_eps0.08 | 0.7 | 0.08 | 0.2462 | 0.2635 | 7.02 | 1.2774 | 6.86 | 0.5276 | 61.15 | 5.17 | -0.372 | 2188 | 406 |
| kh1.5_eps0.02 | 1.5 | 0.02 | 0.2911 | 0.3302 | 13.45 | 0.3253 | 1.65 | 0.0043 | 9.03 | 11.65 | -0.030 | 1727 | 408 |
| kh1.5_eps0.08 | 1.5 | 0.08 | 0.2911 | 0.2356 | 19.04 | 1.3137 | 2.64 | 0.0271 | 64.03 | 13.87 | -0.028 | 2764 | 1139 |
| kh1.5_eps0.08_dx2 | 1.5 | 0.08 | 0.2911 | 0.3365 | 15.61 | 2.5698 | 0.38 | 0.0176 | 88.28 | 13.24 | -0.116 | 3454 | 1478 |
| kh1.5_eps0.08_noreflect | 1.5 | 0.08 | 0.2911 | 0.2679 | 7.94 | 1.3251 | 3.52 | 0.0440 | 41.49 | 15.87 | -0.147 | 1727 | 398 |
| kh3.0_eps0.02 | 3.0 | 0.02 | 0.2786 | 0.3258 | 16.95 | 0.3137 | 17.63 | 0.0005 | 82.95 | 13.92 | -0.002 | 1503 | 381 |
| kh3.0_eps0.08 | 3.0 | 0.08 | 0.2786 | 0.3276 | 17.59 | 1.3011 | 21.98 | 0.0280 | 35.57 | 14.93 | -0.045 | 1503 | 345 |

| criterion | pass | detail |
|---|---|---|
| C1_dispersion | FAIL | tol=0.01 |
| C1b_wavemaker_phase_speed | FAIL | two-point phase speed 0.2-0.4 lambda from the wavemaker; start-up transient / reflected wave bias this estimate (see README section 7), reported for transparency tol=0.01 |
| C2_first_harmonic | FAIL | tol=0.03 |
| C2_second_harmonic | FAIL | eps=0.02 cases have sub-cell second-harmonic amplitude; see report tol=0.1 |
| C3_reflection | FAIL | tol=0.05 neg_ctrl_KR=0.159 |
| C4_mass | PASS | tol=0.005 |
| C5_grid_convergence | FAIL | tol=0.03 |
| C7_cost | FAIL |  |
| ALL | FAIL | |

图（`artifacts/wave_tank/figures/`）：
`fig_field_kh1.5_eps0.08.png`、`fig_field_kh3.0_eps0.08.png`（相场云图 + 液柱积分水面线）、
`fig_probes.png`（探针时序 vs 解析）、`fig_dispersion.png`（C1）、
`fig_reflection.png`（C3）、`fig_harmonics.png`（C2）、`fig_mass.png`（C4）。

## 5. 判据结论（如实）

- **C4 质量守恒：PASS** —— 全部算例 |漂移| ≤ 0.37 %/周期（阈值 0.5），CAC 相场的守恒性成立。
- **C2a 一阶谐波：部分通过** —— kh0.7_eps0.02 (0.36 %)、kh1.5_eps0.02 (1.7 %)、
  kh1.5_eps0.08 (2.6 %)、kh0.7_eps0.08 (6.9 %)；kh3.0 两例 17.6 %/22.0 % 超标 → **整体 FAIL**。
- **C1 色散：FAIL** —— 两点法相速度误差 7.0 %/19.0 %/17.6 %（kh=0.7/1.5/3.0）；
  辅助的驻波自由衰减实测（无造波机）给出 6.0 %/8.0 %/22.6 %，见 §7 的协议诊断。
- **C2b 二阶谐波：FAIL** —— 9 %–83 %（ε=0.02 的案例二阶幅值本就亚格量级，信噪比不足）。
- **C3 反射：FAIL** —— 生产算例 5.2 %–14.9 %；负控仅 15.9 %（应 > 30 %，说明两点法在
  强反射场里本身就失准，见 §7）。
- **C5 网格收敛：FAIL** —— 相位对齐后 RMS/波幅 = 41 %（阈值 3 %），两套网格的水面形状差异显著。
- **C7 成本：生产算例 345–1139 s（< 1200 s）通过；加密算例 1478 s 超标。**
- **C6：** 已纳入 `benchmarks/verified/wave_tank/`（README + run.py + free_decay.py）并写
  `artifacts/sdaa_wave_tank.json`；新增 `tests/test_wave_tank.py` 31 项全绿。
  仓内既有 `tests/test_public_api_lazy_exports.py::test_lazy_table_matches_type_checking_block`
  在**本分支本就是红的**（`RotatingFrameLBM` 等 12 项 lazy 表漂移，与本次改动无关，
  未触碰 `src/tensorlbm/__init__.py`）。

## 6. 复现

```bash
PYTHONPATH=src python benchmarks/verified/wave_tank/run.py --list
PYTHONPATH=src python benchmarks/verified/wave_tank/run.py --run kh1.5_eps0.08
PYTHONPATH=src python benchmarks/verified/wave_tank/run.py --run-all      # 8 个算例
PYTHONPATH=src python benchmarks/verified/wave_tank/run.py --analyze      # -> results.json
PYTHONPATH=src python benchmarks/verified/wave_tank/run.py --figures
PYTHONPATH=src python benchmarks/verified/wave_tank/run.py --sdaa         # -> sdaa_wave_tank.json
PYTHONPATH=src python benchmarks/verified/wave_tank/run.py --markdown     # 本 README 的表
PYTHONPATH=src python benchmarks/verified/wave_tank/free_decay.py --kh 1.5 # 驻波衰减诊断
PYTHONPATH=src python -m pytest tests/test_wave_tank.py -q
```

## 7. 根因诊断与踩坑（本节点的核心工程产出）

**(1) 有效重力**（已修复）：见 §2.2。忽略 ρ_g/ρ_l 会把波数偏 11–12 %。

**(2) 复振幅时间约定**（已修复，代价一整轮扫描）：`fit_harmonics` 拟合的是
`Re[c e^{+iωt}]`，则 +x 行波的空间依赖是 `e^{−ikx}`。两点法若按 `e^{+ikx}` 写，
入射/反射互换、相速度符号翻转 —— 首轮结果出现 c = −0.24、KR = 6000 %。
现已在 `two_probe_split` / `phase_speed_from_phase` 的 docstring 里写死约定，
并由 `test_two_probe_split_recovers_known_reflection`、
`test_phase_speed_from_phase_and_unwrapping` 锁定。

**(3) 数值稳定性**：ν = 0.004（τ_g = 0.512）约 230 步后发散（u_max 0.01 → 0.33 → NaN）；
ν = 0.008（0.524）稳定，ν = 0.012 轨迹几乎相同 ⇒ 粘性不主导动力学。
σ = 1e-4 与 1e-3 结果无可分辨差异 ⇒ 表面张力不是误差源。

**(4) 测量协议是当前精度的瓶颈（未解决，下一步）**：
- 探针离造波区只有 0.2–0.4λ、Δx = 0.17λ。两点法在 Δx 很小时，相位差对残余驻波分量
  （KR ≈ 10 %）极敏感：`d(phase)/dx` 的偏差不随 Δx→0 消失，而 `|A|,|B|` 的分离仍可用
  （这解释了为什么 a1 误差只有 0.4–3 %，而 c 误差高达 7–19 %）。
  同一算例改用**沿一个波长的空间拟合**（`e^{±ikx}` 最小二乘 + k 扫描）得到 λ 偏差 −3.8 %。
- 驻波自由衰减诊断同样不稳：FFT 窗太短 + 波形在封闭域内不是整数个半波长，
  测得 ω 偏差 6 %–23 %（g_eff/g 从 1.01 到 1.35），只有一个早前的短诊断给出 1.3 %。
- 下一次迭代的确定方案：(a) 传播段加长到 ≥2λ、探针间距 ≥0.5λ 且离壁 ≥1.5λ；
  (b) 三点及以上最小二乘 + 对 k 的联合拟合（Goda 多探针法），用 |B| 最小化定 k；
  (c) 暖机 ≥6 个周期以排除启动瞬态（当前 2–4 个周期）；
  (d) 二阶谐波用更小的界面宽 W（1.5 格）与更大的 ε（0.12）提高信噪比。
- 成本约束：上述改动使单算例格数 ×2、步数 ×2（≈4×），需在 loong64 集群上跑。

**(5) 共享 CPU**：所有算例在 99（128 核，负载 280–330）上完成，墙钟含抢占，
ms/step 在 140–430 之间波动（同尺寸算例差 2 倍以上）。