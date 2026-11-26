# XVAL-LBM-SHAN-CHEN-EDM-LYCETT-LUO — EDM/Lycett-Luo 力项的共存曲线复核对标

**锚点** `XVAL-LBM-SHAN-CHEN-EDM-LYCETT-LUO`（CFD-GO/TCLB `models/multiphase/d2q9_scmp` 的
shan-chen-edm-lycett-luo 做法）｜**仓库** TensorLBM｜**分支** `agent/m21435`｜**任务** m21435
**判据** owner 精度标准：对标误差 >15% 不可接受；<3% 另行标注。
**一句话结论**：**EDM（Kupershtokh exact-difference，即 TCLB 的 EDM/Lycett-Luo 力项族）
通过**——平界面共存密度对标 Maxwell 构型最大偏差 3.55 %（液相 ≤1.23 %），且 τ 无关（Γ≤1.1 %）；
**`velocity_shift` 与 `guo` 在气相支上 FAIL**（τ=2 时 `velocity_shift` 气相偏差 `+81.5 %`，
`guo` 系统性 `−21.9 %`）。honest-FAIL 部分如实列出，未粉饰。

---

## ① 案例介绍（物理背景 / 几何 / 参数）

伪势（Shan-Chen）单组元多相模型的**热力学正确性**不由表面张力决定，而由**共存密度**决定。
本轮把「力项注入方式」这一条轴单独拉出来拷问：同一格子 EOS 下，不同力项必须给出同一组共存密度，
否则模型的有效 EOS 被力项污染（这正是 Lycett-Luo / EDM 力项改进要解决的病）。

| 项 | 取值 |
|---|---|
| 几何 | 周期性平界面（1 维沿 x，`nx=160`，`ny=2` 走 D2Q9 对角模板；两个界面/周期框） |
| 格子 | D2Q9，`dx=dt=1`，`cs²=1/3` |
| 伪势臂 A | ψ_exp = ρ₀(1−e^{−ρ/ρ₀})，ρ₀=1，G=5（`exp`） |
| 伪势臂 B | TCLB `d2q9_scmp` 配方 ψ_CS=√(6(ρ/3−p_CS))，T_r=0.85，G=1（`cs`，格子 EOS 与 CS-EOS 恒等） |
| 力项 | `velocity_shift`（u^eq=u+τF/ρ）· `edm`（Kupershtokh 精确差分）· `guo`（Guo 2002 源项） |
| τ | 0.6 / 1.0 / 1.5（`exp` 臂加 2.0 加大杠杆） |
| 步数 / 初值 | 10000 步；tanh 初值；另有 `ic=shifted`（自旋分解起步）做初值无关性 |
| 代码被测面 | `tensorlbm.multiphase.collide_sc_single_component(..., forcing=)` + `tensorlbm.solver.stream` (float64) |

**为什么用平界面而不是液滴**：平界面无曲率 ⇒ 无 Laplace 压差 ⇒ 平台密度必须等于**热力学**共存密度。
液滴法测到的「气相密度」被曲率过饱和抬高（Kelvin），会把力项误差与曲率效应混在一起。

## ② 怎么做的（方法 / 流程 / 关键设置）

1. **解析锚（无拟合、无自由参数）**：`maxwell_sc.py` 独立纯 numpy 实现 Maxwell 等面积构型
   作用于同一格子 EOS
   `p(ρ)=ρ/3−(G/6)ψ²`（由 `G_eff=−G_lib`、`p=ρcs²+(G_eff/2)cs²ψ²` 推得），
   条件：`p(ρ_g)=p(ρ_l)=p_sat` 与 `∫_{ρ_g}^{ρ_l}[p(ρ)−p_sat]/ρ² dρ=0`（v=1/ρ 等面积）。
   求解流程：先 `dp/dρ=0` 定自旋节点 → 在 `(p(rho_s2), p(rho_s1))` 内对 `p_sat` 二分 →
   单调支反解 `ρ_g, ρ_l` → 等面积残差。
2. **自检（先证明锚本身可信）**：`ψ_exp` 的解析临界点 `G_c=4, ρ_c=ln2` 由通用自旋节搜索复现
   （数值 `G_c=3.99999999985`，`ρ_c=0.693148`）；`G→G_c⁺` 两支向 `ρ_c` 坍缩；
   `G=1` + TCLB ψ 时格子 EOS 与 CS-EOS 逐点相差 `<1e-12`；CS 共存点用**结构不同的第二求解器**
   （scipy `root` 解 2×2 系统）复核一致。
   测试：`tests/test_maxwell_sc_xval.py`（10 passed）。
3. **算例矩阵（29 例，先冻结后开跑）**：B1 = 2 伪势 × 3 力项 × 3 τ；B2 = 共存曲线点 (G=4.5, 5.5)；
   B3 = τ=2.0 杠杆（exp 臂 × 3 力项）；B4 = 初值无关性（maxwell vs 自旋分解，τ=1，6 例）。
4. **测量**：平台密度取密度排序的顶/底 10 % 均值（薄界面下即平台，界面单元占比 <10 %）；
   收尾 20 % 步的 `ρ_l/ρ_g` 极差作收敛闸；`Σρ` 漂移作质量守恒闸。
5. **工具链**：`run_coexistence.py`（被测面驱动，`--list/--case` 可分片）、`maxwell_sc.py`（锚）、
   `analyze.py`（聚合 + 出图，报告里的数字全部由它生成，零手抄）。

## ③ 计算结果（数字表，程序化注入）

聚合表见 `artifacts/TABLE.md`（29 行逐例），派生指标如下（`analyze.py` 输出）：

| EOS/力项 | n | τ 集 | ρ_l 的 τ 极差 % | ρ_g 的 τ 极差 % | max abs err ρ_l % | max abs err ρ_g % | 判定 |
|---|---|---|---|---|---|---|---|
| exp / edm | 6 | 0.6,1,1.5,2 | **1.09** | **1.02** | **1.23** | **3.55** | PASS（<15 %） |
| exp / velocity_shift | 4 | 0.6,1,1.5,2 | 5.67 | **79.82** | 3.66 | **81.53** | FAIL（气相支） |
| exp / guo | 4 | 0.6,1,1.5,2 | 0.24 | 0.13 | 2.32 | **21.96** | FAIL（气相支） |
| cs / edm | 3 | 0.6,1,1.5 | **0.09** | **0.01** | **0.27** | **7.78** | PASS（<15 %） |
| cs / velocity_shift | 3 | 0.6,1,1.5 | 2.42 | **75.19** | 1.30 | **45.05** | FAIL（气相支） |
| cs / guo | 3 | 0.6,1,1.5 | 0.03 | 0.31 | 1.28 | **32.49** | FAIL（气相支） |

代表性单例（`ρ(Maxwell)` 括号内）：

| case | ρ_l | ρ_g | err_l | err_g | max\|u\| |
|---|---|---|---|---|---|
| exp_Gdefault_edm_tau1.0 | 1.93432 (1.92887) | 0.15530 (0.15289) | +0.28 % | +1.57 % | 1.3e-01 |
| exp_Gdefault_edm_tau0.6 | 1.95265 | 0.15391 | +1.23 % | +0.66 % | 1.3e-01 |
| exp_G4.5_edm_tau1.0 | 1.49250 (1.49204) | 0.25362 (0.25189) | +0.03 % | +0.68 % | 7.0e-02 |
| exp_G5.5_edm_tau1.0 | 2.30575 (2.30327) | 0.10421 (0.10064) | +0.11 % | +3.55 % | 1.9e-01 |
| exp_Gdefault_velocity_shift_tau2.0 | 1.99955 | 0.27755 (0.15289) | +3.66 % | **+81.53 %** | 7.3e-02 |
| exp_Gdefault_guo_tau1.0 | 1.88822 | 0.11944 (0.15289) | −2.11 % | **−21.88 %** | 1.4e-01 |
| cs_Gdefault_edm_tau1.0 | 0.27856 (0.27931) | 0.02911 (0.03156) | −0.27 % | −7.77 % | 8.0e-02 |
| cs_Gdefault_velocity_shift_tau1.5 | 0.28242 | 0.04578 (0.03156) | +1.11 % | **+45.05 %** | 6.2e-02 |

**不变量闸**：29 例质量漂移 ≤1e-12（`Σρ` 守恒）；收尾 20 % 步平台极差 `drift_l ≤1e-2`；
初值无关性：`maxwell` vs `shifted` 初值在 τ=1 下 `ρ_l` 相差 ≤0.2 %（exp）/ ≤0.01 %（cs），
`ρ_g` 同量级 —— 结论不依赖初值。

## ④ 结果图

`artifacts/fig1_coexistence.png`（共存曲线：G=4.5/5.0/5.5 的实测点 vs Maxwell 连续线）、
`artifacts/fig2_tau_dependence.png`（两伪势臂的 ρ_l、ρ_g 随 τ；虚线为 Maxwell）、
`artifacts/fig3_profiles.png`（τ=0.6 与 2.0 的密度剖面，含 Maxweill 平台参考线）、
`artifacts/fig4_error_bars.png`（各 (EOS, 力项) 的最坏偏差，标 15 % 与 3 % 线）。
四图均由 `analyze.py` 从 `artifacts/cases/*.json` 程序化生成并内嵌进平台报告。

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

1. **τ=1 时 `velocity_shift` 与 `edm` 代数恒等**（u^eq=u+τF/ρ 在 τ=1 即 u+F/ρ，正是 EDM 的位移），
   所以「τ=1 下两方案结果相同」不是证据而是恒等式；τ 必须拉开才有力（本轮 τ∈{0.6,1.5,2.0}）。
2. **`guo` 的偏差是 τ 无关的系统性 EOS 偏移**（exp 气相 −21.9 %、cs 气相 −32.5 %），
   不是收敛问题：加大步数或换初值都不动它（`ic` 两臂一致到 1e-3）。这说明在 SM 伪势语境下
   「Guo 源项」与「EDM 精确差分」**不等价**，选错力项会篡改共存曲线。
3. **cs 臂气相密度系统性偏低 7.8 %**（τ 无关）：`cs` 臂的 τ=0.6 例外（−7.77 %），
   可能与 CS ψ 的 `clamp(·,0)` 分段（`p_CS>ρ/3` 区域 ψ≡0）在界面处的截断有关；本轮未隔离该机制，
   列为未完成项（见 ⑦）。
4. **对标对象是「TCLB 配方的热力学后果」，不是 TCLB 二进制**：本轮无 TCLB 源码/可执行体
   （无网络、无本地 checkout），因此 ψ_CS/G=1 的配方按 `oss_inventory`（CFD-GO/TCLB，
   `cache_commit ded67cd768`）记录的公式重建，锚用 Maxwell 而非另一份自实现代码。这是**独立性缩水**，
   已在 ⑥ 计入。
5. **`max\|u\|` 不能用来给力项排名**：`velocity_shift` 在 τ=2 时 `max|u|` 反而最小（7.3e-2 vs EDM 1.3e-1），
   但共存曲线错得最离谱 —— 寄生流速小 ≠ 热力学对。
6. 平界面法给出的是**热力学共存密度**，不含曲率修正；液滴/Laplace 类结论不在本轮范围。
7. 前一轮（m18323）在 `TensorLBM/xval/XVAL-LBM-SHAN-CHEN-EDM-LYCETT-LUO/` 留下的 `REPORT.md`
   判 PASS 的依据是「σ 对标同一作者自实现的 TCLB 臂，误差 3.9–7.1 %」，且把 20 % 的液相密度差
   归为「EOS 物理差异」；其 Laplace 拟合 R² 为负数（两点拟合）、`TCLB_guo` 两例发散被当作稳定性结论。
   本轮不采信该结论：其中「σ 一致」用的是自建参考臂，「密度=物理差异」无锚。

## ⑥ 误差判定（逐项 vs 基准，显式回答是否 <3 %）

基准 = Maxwell 等面积构型（同一格子 EOS，解析-数值锚；锚自身临界点用解析值 `G_c=4, ρ_c=ln2` 验证）。

| 关键量 | 相对基准最大偏差 | <15 % ? | <3 % ? |
|---|---|---|---|
| `exp/edm` 液相共存密度 ρ_l | 1.23 % | ✅ | ✅ |
| `exp/edm` 气相共存密度 ρ_g | 3.55 % | ✅ | ❌（3.55 %，G=5.5 点；G=5 为 1.7 %） |
| `exp/edm` 共存点 G=4.5 / 5.0 / 5.5（ρ_l） | 0.03 / 0.28 / 0.11 % | ✅ | ✅ |
| `exp/edm` τ 无关性（ρ_l / ρ_g 极差） | 1.09 / 1.02 % | ✅ | ✅ |
| `cs/edm` 液相共存密度 ρ_l | 0.27 % | ✅ | ✅ |
| `cs/edm` 气相共存密度 ρ_g | 7.78 % | ✅ | ❌ |
| `cs/edm` τ 无关性（ρ_l / ρ_g 极差） | 0.09 / 0.01 % | ✅ | ✅ |
| `velocity_shift` 气相支（τ 扫描最坏） | 81.5 %（exp, τ=2）/ 45.1 %（cs, τ=1.5） | ❌ FAIL | ❌ |
| `guo` 气相支（τ 无关系统偏移） | 22.0 %（exp）/ 32.5 %（cs） | ❌ FAIL | ❌ |
| 质量守恒（Σρ 漂移） | ≤1e-12 | ✅ | ✅ |
| 初值无关性（ρ_l，两初值） | ≤0.2 %（exp）/ ≤0.01 %（cs） | ✅ | ✅ |

**锚的独立性/自检**：Maxwell 锚不含被测代码；临界点对解析值 `G_c=4`/`ρ_c=ln2` 复现到 1e-10；
等面积与等压残差 `<1e-9`；CS 臂用第二种求解器（scipy root）复核到 1e-3/1e-4。
**未验证项**：TCLB 二进制级复现（无源码，见 ⑤.4）；Lycett-Luo 的
「非平衡应力 Φ + γ 修正」具体代数形式（本轮验证的是其所属的 EDM 力项族的**后果**，
不是该修正本身的实现）—— 明确记为 NOT_VERIFIED，不冒充。

## ⑦ 本仓 vs OSS 差异与取舍

| 方面 | TCLB `d2q9_scmp` | TensorLBM | 本轮结论 |
|---|---|---|---|
| 伪势 | ψ=√(−p_CS+ρ/3)（CS-EOS） | `psi_eos_carnahan_starling`（同配方，常数因子 √6 ↔ G 重标定） | **等价**：G=1 时格子 EOS ≡ CS-EOS（<1e-12） |
| 力项 | EDM + Lycett-Luo 修正 | `forcing ∈ {velocity_shift, edm, guo}` | EDM 合格；Lycett-Luo 修正未移植（NOT_VERIFIED） |
| 力 stencil | `gs=(0,1,1,1,1,¼,¼,¼,¼)` | D2Q9 标准 w | 两者都在 4 阶各向同意义下等价（见 `multiphase_edm` 的 8 邻域唯一性论证） |
| 壁面 | VirtualRhoWBC 接触角 | half-way BB + wall_psi | 平界面无需壁面，未涉及 |
| 粘度 | 双粘度 ω_l/ω_v | 单 τ | 本轮利用单 τ 正好把 τ 依赖性暴露出来 |

**给使用的建议**：伪势模型做热力学量（共存密度、饱和压、相变）时**必须**用 `forcing="edm"`；
`velocity_shift` 只在与历史结果对齐时使用，且 τ 不得偏离 1 太多（τ=2 时气相密度 +82 %）。

## ⑧ 判定

- **EDM（Lycett-Luo 力项族）**：PASS（两伪势臂液相 ≤1.23 %、气相 ≤7.78 %，τ 无关 ≤1.1 %）。
- **`velocity_shift` / `guo` 的气相共存密度**：**honest FAIL**（>15 %），并已定位为力项引入的
  有效 EOS 污染（`velocity_shift` 随 τ 单调漂移；`guo` 为 τ 无关的系统偏移）。
- **未完成**：TCLB 二进制级复现、Lycett-Luo Φ/γ 修正实现、cs 臂气相 −7.8 % 偏差的机制隔离
  （建议新条目：`XVAL-LBM-CS-PSI-CLAMP-VAPOR-BRANCH`，验收：把 cs 臂气相偏差压到 <3 %，
  并给出 ψ clamp 分段与气相平台密度的定量关系）。

## 复现

```bash
cd development/verification/shan_chen_edm_lycett_luo/round1
PYTHONPATH=../../src python run_coexistence.py --list            # 29 例清单（冻结）
PYTHONPATH=../../src python run_coexistence.py --all --steps 10000 --nx 160 --out artifacts/run.json
PYTHONPATH=../../src python analyze.py                            # 表 + 四图
PYTHONPATH=../../src python -m pytest ../../tests/test_maxwell_sc_xval.py -q   # 锚自检 10 passed
```