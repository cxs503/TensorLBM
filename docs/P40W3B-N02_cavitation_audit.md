# P40W3B-N02 空化动力学验证面：翼型片状/云状空化 —— 数值链路审计与诚实判定

> 任务 P40W3B-N02（TensorLBM，分支 agent/m7590）· 全部数字由
> `scripts/p40w3b_n02_cavitation_diagnostics.py` 程序化写入
> `artifacts/p40w3b_n02/*.json`，本文零手抄（引用的每个数字都标注来源文件）。

## 结论先行（honest verdict）

| 目标 | 判据 | 本节点结果 | 判定 |
|---|---|---|---|
| (a) 片状空化 `L_cav/c`，σ=1.0/0.8/0.6 vs Brennen | 误差 ≤15% | 0.090/0.126/0.204 vs 0.25/0.40/0.65 → **误差 64.0% / 68.5% / 68.6%** | **FAIL**（不可接受） |
| (a) `δ_cav/c` vs Brennen | 误差 ≤15% | 0.020/0.040/0.080 vs 0.025/0.038/0.055 → −20.0% / +5.3% / +45.5% | **FAIL**（1/3 项过线） |
| (a) 网格收敛 dx=c/50/100/200 | 三档收敛 | 50 档 step 569 发散、**100 档 step 425 就发散**（`sheet_cpc100.json`）→ 加密反而更早失稳，不存在收敛解；200 档未跑 | **WITHHELD（网格不收敛）** |
| (b) 云状空化 `f_cav`/`St_cav` σ=0.5/0.4 vs Franc & Michel 1985 | 误差 ≤15% | 无法给出：稳定窗口仅 t≈2.3 c/U，不足以包含一个脱落周期 | **NOT_SUPPORTED** |
| 相变源项闭式（Schnerr–Sauer）动态校验 | 与 Rayleigh–Plesset 惯性解一致 | 闭式松弛时间 = 0.707×RP 时间；积分后气泡半径 = 1.023×RP 渐近解 | **REAL（已验证，2.3%）** |
| 面元法流动引擎（非升力分支） | 对解析圆柱解 | 表面速度最大相对误差 **1.4e-10**，`Cp_min=−2.998`（解析 −3） | **REAL（已验证）** |

**先说清楚**：本节点**没有**把 15% 判据做出来，也没有伪造它。上一轮 P41-N02
（`docs/P41-N02_cavitation_airfoil_validation.md`，m6692）已给出 honest FAIL
（49%~78% 误差）。本节点的增量不是"再报一次 FAIL"，而是**把那条数值链路打开、
量出失败的确切原因、修掉其中可修的部分、并给出一个已验证的替代引擎（面元法）
及其尚未验证的边界**。所有诊断量都是可复现的闭式或一次短跑，脚本可重跑。

## ① 案例介绍

**物理背景**：片状空化（sheet cavity）与云状空化（cloud cavitation）是水翼/螺旋桨
推力崩溃、空蚀剥蚀的直接来源。教科书验证面：

* (a) NACA0015，c=10 mm，α=5°，U∞=10 m/s，Re=1e5，σ=1.0/0.8/0.6；
  基准 Brennen (1994/1995) 水洞实验 `L_cav/c`、`δ_cav/c`，与 Merkle–Feng–Buelow
  (1998) 数值解；
* (b) Clark-Y，c=10 mm，α=6°，U∞=10 m/s，Re=1e5，σ=0.5/0.4；
  基准 Franc & Michel (1985, JFM 154) 的脱落频率与 Strouhal 数。

**几何（运行实测，非抄录）**：NACA0015 最大厚度 0.1500c @ x/c=0.300、围面积
0.1021c²；Clark-Y 解析拟合 t_max=0.117 @ 0.40（`airfoil_surface()`，与
Abbott & von Doenhoff 1959 锚值一致，见 `tests/test_cavitation_airfoil.py::
test_naca0015_thickness_and_area`）。

**参数表（本次实际运行）**

| 量 | 片状空化 | 云状空化 |
|---|---|---|
| 翼型 | NACA0015 | Clark-Y（解析拟合） |
| α | 5° | 6° |
| σ | 1.0 / 0.8 / 0.6 | 0.5 / 0.4 |
| Re | 1.0e5 | 1.0e5 |
| 域 | 6c × 3c，LE 在 (1.5c, 1.5c) | 同 |
| 网格 | 50 / 100 cells/chord（200 未跑） | 50 / 100 / 200 |
| 运行步数 | 500（50 档 step 569 / 100 档 step 425 发散，见 §⑤） | — |

## ② 怎么做的

### 2.1 三套引擎（路线声明，行使方法自主权）

任务书建议"伪势 + Lee/Schnerr-Sauer 源项"。P41-N02 已实测该路线在 1e3 密度比、
50~200 cells/chord 下不可用（界面无法解析），改用**匀质混合物** NS
（`cavitation_airfoil.MixtureCavitationSolver`）。本节点在此基础上**不开新模型**，
而是先把这条链路的每个环节用闭式关系量清楚（新模块
`src/tensorlbm/cavitation_numerics.py`），再补一套**经典自由流线（等压空泡）
面元法引擎**（`src/tensorlbm/cavitation_free_streamline.py`）。

**偏差与理由**：判据要求的是"与 Brennen/Franc 实验对比"的物理能力，不是指定算法。
当混合物引擎被证明在此分辨率下数值不稳定时，诚实的增量是把不稳定量化 + 提供
可验证的替代路线，而不是把一个发散的场当作"结果"上报。

### 2.2 本节点的四个诊断关系（全部闭式/一维积分，`cavitation_numerics`）

1. **压力基准**：分步投影解的是纯 Neumann 压力泊松，`p` 只定到常数，而
   Schnerr–Sauer 源项比较的是**绝对**压力与 `p_v`。物理常数由入口 `p_inf=0` 定：
   `pressure_anchor()`。P41-N02 的判据用未锚定的 `p` → 蒸发开关由任意常数决定。
2. **Brinkman 罚函数滑移**：每步 `u*=(u+dtR)/(1+dt/eta)`，稳态体内滑移
   `1/(1+dt/eta)`。P41-N02 的 `eta=1e-3`、`dt=cfl·h=0.004` → 解析滑移 **20%**，
   实测体内最大速度 0.130（`stability.json` 同款跑法）。
3. **相变时间尺度**：闭式 `t_SS = R_b/√(6Δp)` vs Rayleigh–Plesset
   `t_RP = R√(ρ/3Δp)`，比值恒为 **0.707**（`closure.json::rows`）。
4. **闭式积分校验**：以 `schnerr_sauer_rate` 本身积分 ODE，气泡半径与 RP 惯性
   渐近解 `R=t√(2Δp/3ρ)` 相差 **2.3%**（`closure.json::integrated`）。

### 2.3 工具链

`scripts/p40w3b_n02_cavitation_diagnostics.py`（跑例 + JSON 落盘）→
`scripts/p40w3b_n02_cavitation_diagnostics.py --wave closure/panel`（闭式量）→
`scripts/p40w3b_n02_figures.py`（图，全部从 JSON 读数）→ 本文。
`pytest tests/test_cavitation_numerics.py`（13 例，含 1 个 strict-xfail 记录已知缺口）。

## ③ 计算结果（数字表）

### (a) 片状空化：`L_cav/c`、`δ_cav/c`（`artifacts/p40w3b_n02/sheet.json`，50 cells/chord，t=2.0 c/U 瞬态）

| σ | L/c 本工作（锚定判据） | L/c 未锚定（P41-N02 判据） | 输运 α 判据 | Brennen 实验 | 误差（锚定） | Merkle 数值 | 误差（锚定） |
|---|---|---|---|---|---|---|---|
| 1.0 | 0.090 | 0.006 | 0.000 | 0.250 | 64.0% | 0.230 | 60.9% |
| 0.8 | 0.126 | 0.108 | 0.000 | 0.400 | 68.5% | 0.370 | 65.9% |
| 0.6 | 0.204 | 0.150 | 0.000 | 0.650 | 68.6% | 0.600 | 66.0% |

| σ | δ/c 本工作 | Brennen | 误差 | 压力基准偏移 c₀ | α_max（输运场） | C_L | C_D |
|---|---|---|---|---|---|---|---|
| 1.0 | 0.020 | 0.025 | −20.0% | −0.0600 | 1.5e-7 | 0.653 | 1.036 |
| 0.8 | 0.040 | 0.038 | +5.3% | −0.0600 | 3.5e-3 | 0.653 | 1.036 |
| 0.6 | 0.080 | 0.055 | +45.5% | −0.0600 | 3.2e-2 | 0.653 | 1.037 |

要点（全部可在 JSON 里复算）：

* **压力基准决定判据**：同一张压力场，锚定/未锚定给出 0.090 vs 0.006（σ=1.0），
  差 15 倍——这就是 P41-N02 报出 0.054 的直接原因（偏移虽只有 −0.060，但等压
  判据的过渡宽度是 0.05|p_v|≈0.02，相当于 3 个过渡宽度）。
* **输运汽相场 2.0 c/U 内几乎不生成**（α_max ≤ 3.2e-2），所以 `alpha`-判据恒为 0：
  P41-N02 改用等压替代不是选择，而是被源项/限幅器逼出来的。
* **升力合理、阻力荒谬**：C_L=0.653（NACA0015 α=5° 势流约 0.62~0.65，合理），
  但 **C_D=1.036**，而 Re=1e5 的实验值约 0.02（50 倍）——Brinkman 反力不是水动力
  （体内滑移 13~20% 的必然结果）。
* **趋势仍然对**：L/c 随 σ 单调下降（1.0→0.6 增 2.3 倍），但整体比实验短 3 倍，
  与"单相等压判据结构性低估 ~2 倍"的经典认识一致——即便锚定正确，判据本身也
  不可能命中 15%。

### (b) 云状空化：`f_cav` / `St_cav`

**未给出（NOT_SUPPORTED）**。原因不是算不出来，而是**没有可用的时间窗**：
引擎在 step 569（t=2.28 c/U，`artifacts/p40w3b_n02/stability.json`）发散，
而云状脱落周期对应 St≈0.2 → 一个周期 5 c/U，至少要 20~40 c/U 才能做 FFT。
此外闭式等压/势流模型本身不含回射流，无法产生云状脱落。两个原因都写在这里，
不用任何"MOCK 频率"充数。

### (c) 已验证的能力（本节点真正拿到的 REAL 结果）

| 项 | 基准 | 本工作 | 误差 | 证据 |
|---|---|---|---|---|
| 圆柱势流表面速度 | `q_θ=−2U sinθ` | 面元法 160 面板 | **1.4e-10** 相对 | `panel.json::cylinder` |
| 圆柱 `Cp_min` | −3 | −2.998 | 0.05% | 同上 |
| 闭式相变积分气泡半径 | `R=t√(2Δp/3ρ)` | 由 α 反解 | **2.3%** | `closure.json::integrated` |
| 闭式松弛时间 | `R_b/√(6Δp)` | 解析实现 | 闭式一致 | `tests/test_cavitation_numerics.py` |

### 图（`docs/figures/p40w3b_n02_*.png`，均由 `p40w3b_n02_figures.py` 生成）

1. `p40w3b_n02_lcav_vs_sigma.png` —— L/c vs σ：本工作（锚定/未锚定）vs Brennen vs Merkle；
2. `p40w3b_n02_closure_timescales.png` —— Schnerr–Sauer 松弛时间 vs RP 时间；闭式 ODE 积分 vs `α∝t³`；
3. `p40w3b_n02_divergence.png` —— `max|u|` 随步数指数增长与 step 569 发散点；
4. `p40w3b_n02_field_map.png` —— σ=0.8 压力场云图 + `p=p_v` 等压线（即空泡轮廓）+ C_L(t)；
5. `p40w3b_n02_panel_verification.png` —— 面元法 vs 解析圆柱表面速度。

## ④ 误差判定（逐项，显式回答"是否<3%"）

| 量 | 基准 | 误差 | <3%？ |
|---|---|---|---|
| `L_cav/c` (σ=1.0/0.8/0.6) | Brennen 实验 | 64.0% / 68.5% / 68.6% | **否** |
| `L_cav/c` | Merkle 数值 | 60.9% / 65.9% / 66.0% | **否** |
| `δ_cav/c` (σ=0.8) | Brennen | 5.3% | **否**（>3%，但进入了 15% 判据线） |
| `δ_cav/c` (σ=1.0/0.6) | Brennen | 20.0% / 45.5% | **否** |
| `St_cav` | Franc & Michel 1985 | 无值 | 不适用（NOT_SUPPORTED） |
| 面元法圆柱表面速度 | 解析解 | 1.4e-10 | **是** |
| 闭式相变积分半径 | RP 渐近解 | 2.3% | **是** |

## ⑤ 注意点、失败原因与适用范围

**发散的确切证据**（`stability.json`，可复现）：单相（c_dest=0）跑 NACA0015
σ=0.8、50 cells/chord、cfl=0.2、Smagorinsky 0.1，`max|u|` 在 step 250→550 由
1.33 指数增长到 3.14，位置在尾缘下游的尾迹（i≈172→279），step 569 出现 NaN。
显式格式预算（`closure.json::budget`）全部安全：CFL 0.20、粘性数 1.1e-3、SGS 数
2.0e-3；加出口海绵层（sponge）与更强 Smagorinsky（0.25/0.40）、更小 CFL（0.10）
均未阻止发散。→ 结论：**不是 CFL 违约，是尾迹/出口处的边界-对流不稳定**
（纯 Neumann 压力、无一阶出口压力条件、尾缘阶梯面）。这是引擎级缺陷，不是本节点
一个补丁能修的；已如实记为未完成项。

**为什么 15% 判据在此分辨率下结构性不可达**：Re=1e5 时 NACA0015 的湍流边界层
在 x=c 处 δ/c≈0.037，200 cells/chord 只有 7 个格子覆盖整层（粘性底层 y⁺≈1 →
<0.1 格），且引擎只有 Smagorinsky（无壁面模型）；片状空泡长度由吸力面压力恢复
位置决定，恰恰是壁面模型的敏感量。文献里 200 cells/chord 能到 10% 的结果都带
k-ω SST + 壁面函数。→ 判据本身与所用的无壁面模型 LES 分辨率不相容。

**已修/已提供**：
* 压力基准（`pressure_anchor` / `CavitationFlowConfig.pressure_anchor=True`，
  默认关以保持 P41-N02 结果逐位可复现）；
* Brinkman `eta` 反解（`brinkman_eta_for_leakage(dt, 0.01)` → 4.0e-5，
  即 1% 滑移；历史值 1e-3 给 20%）；
* 相变闭式时间尺度与积分校验（此前只有静态前因子校验）；
* 面元法流动引擎（非升力分支已验证）+ 空泡自由边界迭代骨架（升力分支**未**验证）。

**已知缺口（明确未做）**：
* 面元法 Kutta 闭合给出薄翼理论 0.527 倍（`panel.json::lifting_branch`，strict
  xfail 锁定）→ 等压空泡自由边界结果 **WITHHELD**，不写进任何表；
* 引擎的出口压力条件/尾迹稳定性重写（需要混合 BC 压力泊松，工作量大）；
* 100/200 cells/chord 的完整网格收敛：100 档实测 3 个 σ 全部在 step 425 发散
  （`sheet_cpc100.json`），比 50 档的 569 更早 → 该引擎在当前闭环下**没有网格
  收敛解**，这本身就是否定性结论（不是分辨率不足，是格式不稳定）。

## ⑥ 复现

```bash
python3 scripts/p40w3b_n02_cavitation_diagnostics.py --wave closure   # 闭式量 + 面元校验
python3 scripts/p40w3b_n02_cavitation_diagnostics.py --wave stability --steps 900
python3 scripts/p40w3b_n02_cavitation_diagnostics.py --wave sheet --steps 500 --cpc 50
python3 scripts/p40w3b_n02_figures.py
python3 -m pytest tests/test_cavitation_numerics.py -q   # 13 passed, 1 xfailed
```

## 参考文献

* Brennen, C. E. (1994) *Hydrodynamics of Pumps*; (1995) *Cavitation and Bubble Dynamics*, ch. 7.
* Merkle, C. L., Feng, J. & Buelow, P. E. O. (1998) 3rd Int. Symp. Cavitation, Grenoble.
* Schnerr, G. H. & Sauer, J. (2001) 4th Int. Conf. Multiphase Flow, New Orleans.
* Franc, J.-P. & Michel, J.-M. (1985) *J. Fluid Mech.* 154, 63-90; Franc (2001) CAV2001.
* Katz, J. & Plotkin, A. (2001) *Low-Speed Aerodynamics*, 2nd ed. (Hess-Smith 面元法).