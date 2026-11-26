# XVAL-LBM-REGULARISED-BC-NONEQUILIBRIUM-STRESS-REC — 第 2 轮审计与推进（调研结合）

> Repository migration note: current campaign files are at [development/verification/regularised_bc_2d_reference](../../development/verification/regularised_bc_2d_reference/), [development/verification/regularised_bc_noneq_stress](../../development/verification/regularised_bc_noneq_stress/). Original mission paths in the narrative refer to the frozen historical revision. Historical verdicts are not current acceptance.

任务号 m21104｜分支 agent/m21104｜基准出处：joconnor22/LIFE `src/Grid.cpp::regularisedBC`（L386-470，Latt & Chopard 2006 / Latt et al. 2008 正则化边界）、
D2Q9/D3Q19 解析解（平面 Couette、体积力 Poiseuille、Stokes 第二问题有限深 Stokes 层）。
owner 精度判据：对标误差 <15% 不可接受线、<3% 目标线。

---

## 0 已有实现清单（远端对账，本仓 main `f5478957d`；禁重复实现）

| 文件 | 来历 | 能力 | 本轮动作 |
|---|---|---|---|
| `src/tensorlbm/regularised_bc.py`（841 行，NumPy，D2Q9/D3Q19） | m18423 / PR #2303 | `regularised_face` / `complete_unknowns` / `project_second_order` / `apply_regularised_wall`；D3Q19 槽道求解器 `run_channel(wall_bc="regularised")` 真实调用 | 未改，重跑复算（数字与前轮一致） |
| `src/tensorlbm/regularised_bc_2d.py`（248 行，Torch，D2Q9） | m18240 / PR #2213 | `apply_regularised_bc_2d` 向量化端点/壁面/角点 | 未改（审计未发现缺陷） |
| `xval/XVAL-LBM-REGULARISED-BC-NONEQUILIBRIUM-STRESS-REC/`（run_xval + artifacts + REPORT） | m18423 | identity / steady / oscillating 三臂 | 重跑复现；补 `make_round2.py`（前轮 REPORT 引用的 `make_report.py` 实际缺失） |
| `xval/XVAL-LBM-REG-BC-NEQ-STRESS/{reference.py,run_xval.py}` | m18240 | LIFE 逐行标量转写 + 2D 端到端驱动 | **修 4 处缺陷**（下述） |
| `tests/test_regularised_bc.py` / `tests/test_regularised_bc_2d.py` | m18423 / m18240 | 15 个用例 | 2D 用例全红 → 修复后 15/15 绿 |

前轮 m21063 的事实基线（诚实记录）：其 `fail_stage=budget_exhausted`，`failure_reason="模型未调用 finish（终局: budget_exhausted）"`，
`evidence_ignored=["tests/test_regularised_bc_2d.py"]`，唯一改动文件是同名测试。**失败不是方法失败，而是过程失败 + 恰好撞上本单要修的缺陷面。**

---

## 1 问题设定与基准出处

正则化边界（non-equilibrium stress reconstruction）把边界节点的非平衡应力

    Pi_neq^{ab} = Σ_q c_a c_b (f_q − f_q^eq)

在「已重构的种群集合」上累加，然后用二阶 Hermite 投影重写全部种群

    f_q = f_q^eq + (w_q / 2 c_s^4) Q_qab Pi_neq^{ab},  Q_qab = c_a c_b − c_s² δ_ab

未知入射种群的闭合沿用 Liang/Zou-He 型：速度类 `rho = (2 f⁺ + f⁰)/(1 − n·u)`，压力类 `u_n = n(1 − (2f⁺+f⁰)/rho)`，
非平壁角点只做密度一阶外推。

基准 (external_anchor)：
1. **LIFE 逐行转写**（上游 `Grid.cpp` md5 已固定，`xval/XVAL-LBM-REG-BC-NEQ-STRESS/oss_src/Grid.cpp`）；
2. **解析解**：平面 Couette `u_x = U y/H`；体积力 Poiseuille `u_x = F/(2ν) y(H−y)`；
   Stokes 第二问题有限深 Stokes 层 `u(y,t) = Re{ U sinh[k(H−y)]/sinh(kH) e^{iωt} }, k=(1+i)/δ, δ=√(2ν/ω)`。

## 2 方法与实现（本轮）

* **独立复算**：重跑 `run_xval.py --arm all`，逐位复现前轮工件（identity 5.873e-16、steady/oscillating 同前），
  说明前轮数字不是手抄。
* **对抗审计**（adversarial-review）：复核「参考臂是否可信」。发现 `reference.py` 的 `extrapolate` / `zero_gradient`
  沿法向走步时把物理法向 `(n_x, n_y)` 直接当数组步长用——字段是 `[y, x]` 索引，行必须按 `n_y`、列按 `n_x` 走。
  旧实现行按 `n_x` 走，等于把模板旋转 90°：在沿另一轴均匀的纯平壁上无感，但**压力出口切向零梯度与角点密度外推全错**。
* **修复**（最小改动）：`reference.py` 引入 `_step(normal) → (n_y, n_x)` 单一映射点，两处调用共用；
  `tests/test_regularised_bc_2d.py` 中 `|momentum| < 1e-12` 的错误不变量改为真正的守恒式
  `Σc f = rho·u`（速度入口的动量本来就不是 0，旧断言把正确实现判成错的），并补「非平衡部分质量/动量零」检查。
* **2D 端到端驱动修复**（`xval/XVAL-LBM-REG-BC-NEQ-STRESS/run_xval.py`）：
  (a) `np.linalg.solve(A(n,4,4), b(n,4))` 批量形状错 → `b[..., None]`；
  (b) 入口 y 动量方程 f6/f8 符号反 → 破坏对角对称；
  (c) 入口把解出的 `rho` 写进 `f0`（rest 种群）→ 每步注质量；
  (d) `K2_momentum_after` 用 `|Σc f|` 当不变量 → 改为 `|Σc f − rho u|`。
* **新增**：`make_round2.py`（工件 + 三张图，全部从机器产出 JSON 注入）；2D 驱动加 `--arms/--res/--steps` 与 profile 导出。

## 3 参数表（本轮全部无量纲）

| 量 | 1D D3Q19（args 复算） | 2D D2Q9（本轮新增，torch） |
|---|---|---|
| ν | 0.1（τ=0.8） | H=16: 8.0e-3（τ=0.524）；H=32: 1.6e-2（τ=0.548） |
| Re = u_max·H/ν | — | 100 |
| Ma = u_max/c_s | 0.02/0.5774 = 3.46e-2 | 0.05/0.5774 = 8.66e-2 |
| 槽高 | ny=16 / 32（H=15/31），nx=2 周期 | H=16、32（ny=H+1，nx=4H+1） |
| 壁面闭包 | regularised / 半程 bounce-back | regularised（对照臂见 §6） |
| 振荡壁 | U0=0.02，T=200/100 步，6 周期，末周期 DFT | — |
| 收敛判据 | 每 200 步 max\|Δu\|/max\|u\| < 1e-10 | 每 500 步 max\|Δu\| < 1e-13（本轮未达，见 §5） |
| 精度 | float64 CPU | float64 CPU |

## 4 对照表（逐项 ref vs got）

### 4.1 参考臂修复前后（节点级，48 个边界节点，D2Q9）

| 指标 | 修复前 | 修复后 | 判据 |
|---|---|---|---|
| 与 LIFE 转写最大节点偏差（非角点） | **21.1997 %** ✗ | 3.335e-14 % ✓ | <1e-8 % |
| 角点偏差 | 22.2–36.2 % ✗ | 0.0 % ✓ | <15 % |
| 压力出口切向零梯度朝向 | 沿错轴 | 沿法向 | — |
| `pytest tests/test_regularised_bc_2d.py` | 2 failed / 2 passed | 4 passed | 全绿 |
| 全仓本锚点相关用例 | 13 passed / 2 failed | **15 passed** | 全绿 |

### 4.2 闭包恒等性（1D，合成边界态 × 4 法向，重跑）

| 指标 | 参考 | got | 判据 |
|---|---|---|---|
| 非平衡应力保真 | 精确 S | 5.873e-16 | <1e-13 ✓ |
| 三阶矩 ghost | 0 | 1.735e-17 | <1e-13 ✓ |
| 与 OSS 标量转写一致 | LIFE Grid.cpp 逐行 | 1.388e-17 | <1e-12 ✓ |

### 4.3 稳态（解析解参考，1D D3Q19）

| 算例 | 闭包 | ny | L2 相对误差 | <15% | <3% |
|---|---|---|---|---|---|
| Couette | regularised | 16 | 4.007e-9 % | ✓ | ✓ |
| Couette | regularised | 32 | 5.121e-8 % | ✓ | ✓ |
| Couette | bounce_back | 16 | 6.319e-9 % | ✓ | ✓ |
| Couette | bounce_back | 32 | 4.425e-8 % | ✓ | ✓ |
| Poiseuille | regularised | 16 | 1.129 % | ✓ | ✓ |
| Poiseuille | regularised | 32 | 0.2691 % | ✓ | ✓ |
| Poiseuille | bounce_back | 16 | 0.6427 % | ✓ | ✓ |
| Poiseuille | bounce_back | 32 | 0.1400 % | ✓ | ✓ |

### 4.4 振荡壁 Stokes 层（1D，末周期 DFT）

| 闭包 | T(步) | δ | H/δ | 复振幅 L2 | <15% | <3% | 幅值误差 | 相位误差(掩膜) | 逐周期漂移 |
|---|---|---|---|---|---|---|---|---|---|
| regularised | 200 | 2.523 | 18.6 | 2.874 % | ✓ | ✓ | 0.441 % | 3.02e-2 rad | 4.0e-5 |
| regularised | 100 | 1.784 | 26.3 | 5.624 % | ✓ | ✗ | 1.031 % | 5.77e-2 rad | 5.3e-5 |
| bounce_back | 200 | 2.523 | 18.2 | 4.653 % | ✓ | ✗ | 0.479 % | 4.83e-2 rad | 3.3e-5 |
| bounce_back | 100 | 1.784 | 25.8 | 9.181 % | ✓ | ✗ | 1.177 % | 9.36e-2 rad | 4.0e-5 |

### 4.5 2D D2Q9 torch 端口端到端（本轮新增，Re=100，入口给定抛物线）

| H | 步数 | K1 节点级 vs LIFE | K2 Π_neq 保真 | K2 动量不变量 | K3 剖面最大误差 | K4 场 L2 | K5 质量漂移 | K6 出口 u_y |
|---|---|---|---|---|---|---|---|---|
| 16 | 1500 | 3.335e-14 % ✓ | 1.39e-17 ✓ | 1.39e-17 ✓ | 0.8155 % ✓ | 1.1629 % ✓ | 1.4226 % | 1.0711 % |
| 32 | 1500 | 6.668e-14 % ✓ | 1.73e-17 ✓ | 2.08e-17 ✓ | 1.0452 % ✓ | 1.3316 % ✓ | 1.6273 % | 1.2654 % |

壁面滑移 K7 = 0.0 %（壁面节点速度严格为零，正则化闭包本身即壁面，无半格偏移）。

### 4.6 判据打勾汇总

* 节点级与 OSS/参考一致：✓（3.3e-14 %，判据 1e-8 %）
* 稳态 Couette/Poiseuille <3 %：✓（8/8 项）
* 振荡壁 <15 %：✓（4/4 项）；其中 2 项进 <3 %
* 2D 端到端 <3 %：✓（K3 0.82 %/1.05 %，K4 1.16 %/1.33 %）
* 动量/应力不变量：✓（1e-17）
* **质量守恒：✗**（速度入口 Zou-He 型闭合每步不守恒，K5≈1.4–1.6 %；属闭合固有性质，见 §6）

## 5 收敛性 / 参数敏感性

* 1D Poiseuille 网格：ny=16 → 32，regularised 误差 1.129 % → 0.2691 %（≈4.2×，接近二阶），bounce_back 0.6427 % → 0.1400 %（≈4.6×）。
* 振荡壁 δ 分辨：T=200 → 100（H/δ 18.6 → 26.3，δ 2.52 → 1.78 格），regularised 复振幅误差 2.874 % → 5.624 %；薄 Stokes 层是主要误差来源。
* 2D 端口时间收敛：H=32 在 1500 步时 K3=1.0452 %，3000 步时 K3=0.7167 %（同代码、同初值），说明 1500 步尚未完全定常；
  表 §4.5 保留 1500 步数字，另记 3000 步值，未做挑选。
* 2D 分辨率：H=16 → 32 时 K3 0.8155 % → 1.0452 %（未见二阶收敛）。诚实说明：本例误差由入口/出口闭合与 K6（出口 u_y≈1.07–1.27 %）主导，
  不是壁面闭包误差，故不能用来判阶；判阶请用 §5 第一条的 1D 体积力算例。

## 6 失败模式与工件

**失败模式（实测，全部登记）**
1. `reference.py` 模板朝向错（旋转 90°）→ 与生产端口假性 21 % 不符，且把正确实现判成错。已修，`_step` 单点映射。
2. 测试不变量写错（`|Σc f| < 1e-12`）：速度入口动量本就非零，正确实现被误判。已改为 `Σc f = rho u`。
3. 2D 驱动手写 Zou-He 基线三处缺陷：`solve` 批量形状、入口 y 动量 f6/f8 符号、入口 `rho` 写入 `f0`。
   修完仍发散（H=16、Re=100、约 1e3 步内 |u|→∞）。**结论：不用这个手写基线当对照臂**；正确做法是复用
   `tensorlbm.boundary.boundaries.zou_he_inlet_velocity/zou_he_outlet_pressure`（该函数只接受标量入口速度，抛物线入口需另建接口），
   列为后续项，不在本轮"造一个能跑的对照"来自欺。
4. 正则化速度入口不守恒质量（K5≈1.4–1.6 %）：Zou/He 型 `rho` 闭合的固有性质，不是端口 bug；闭腔/长时算例要配质量修正。
5. 前轮 m21063：预算耗尽未 finish（过程失败），与本锚点技术结论无关。

**工件**
* `xval/XVAL-LBM-REGULARISED-BC-NONEQUILIBRIUM-STRESS-REC/artifacts/round2_summary.json`（本轮机器产出汇总）
* `xval/XVAL-LBM-REG-BC-NEQ-STRESS/out/res_h{16,32}.json`（2D 端到端原始记录，含 K1–K7 与剖面）
* `xval/XVAL-LBM-REGULARISED-BC-NONEQUILIBRIUM-STRESS-REC/figures/fig{1,2,3}_*.png`
* 1D 三臂原始工件 `artifacts/{identity,steady,oscillating,summary}.json`（重跑复现，未改动）

**repro_cmd**

```bash
# 1) 全仓本锚点用例（15 个）
PYTHONPATH=src python -m pytest tests/test_regularised_bc.py tests/test_regularised_bc_2d.py -q
# 2) 1D 三臂（约 100 s）
PYTHONPATH=src python development/verification/regularised_bc_noneq_stress/run_xval.py --arm all
# 3) 2D 端到端分辨率阶梯（单分辨率约 10–60 s；长跑请递交 compute gate）
PYTHONPATH=src python development/verification/regularised_bc_2d_reference/run_xval.py --arms reg --res 16,32 --steps 1500
# 4) 工件 + 图
PYTHONPATH=src python development/verification/regularised_bc_noneq_stress/make_round2.py
```

## 7 误差判定（逐项显式回答）

| 项 | 误差 | <15 % | <3 % |
|---|---|---|---|
| 节点级 vs LIFE 转写（非角点） | 3.335e-14 % | ✓ | ✓ |
| 角点 vs LIFE 转写 | 0.0 % | ✓ | ✓ |
| 1D Couette regularised ny=16/32 | 4.0e-9 / 5.1e-8 % | ✓ | ✓ |
| 1D Poiseuille regularised ny=16/32 | 1.129 / 0.2691 % | ✓ | ✓ |
| 1D Poiseuille bounce_back ny=16/32 | 0.6427 / 0.1400 % | ✓ | ✓ |
| 振荡壁 regularised T=200 | 2.874 % | ✓ | ✓ |
| 振荡壁 regularised T=100 | 5.624 % | ✓ | ✗（诚实记未达 3 %） |
| 振荡壁 bounce_back T=200 / T=100 | 4.653 / 9.181 % | ✓ | ✗ |
| 2D D2Q9 K3（H=16 / 32） | 0.8155 / 1.0452 % | ✓ | ✓ |
| 2D D2Q9 K4（H=16 / 32） | 1.1629 / 1.3316 % | ✓ | ✓ |
| 质量守恒 K5 | 1.4–1.6 % | —（非精度类判据） | ✗（闭合固有） |

**结论**：本锚点总账判据 <15 % 全部满足，主判据（节点级对标 + 2D 端到端剖面）全部进 <3 %；
两处诚实未达标：振荡壁 T=100（5.62 %）与质量守恒（1.4–1.6 %，非精度指标）。第 1 轮遗留的两条红测试本轮修复为 15/15 绿，
根因在参考臂与测试不变量，**生产实现本身未发现缺陷**（这一点是负面结论：前轮"端口复制错"的假设被证伪）。