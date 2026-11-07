# 分区流固耦合附加质量不稳定性定标：1D 解析收敛因子与 IQN-ILS/Aitken 稳定化

**节点** IN33-N03 / P33-N3 ｜ **仓** TensorLBM ｜ **模块** `src/tensorlbm/fsi_partitioned.py`
**产物** `artifacts/fsi_partitioned/`（`oned_scaling.json`、`criteria.json`、`figures/`、`viv_validation_status.json`）
**测试** `tests/test_fsi_partitioned.py`（19 条，全绿）

---

## 1. 案例介绍（物理背景与参数）

分区（partitioned / staggered）流固耦合把流体与结构作为两个独立求解器交替求解。当结构"轻"、流体"重"时，
流体对界面加速度的惯性反作用（附加质量 $m_a$）远大于结构质量 $m_s$，交替迭代会放大界面误差并**发散**——
即经典的 *added-mass instability*（Causin, Gerbeau & Nobile 2005；Förster, Wall & Ramm 2007）。
船舶总体性能里对应的正是螺旋桨-轴系、舵-鳍、轻质附体在重流体中的强耦合问题，以及 VIV 类弹性响应。

### 1.1 1D 参考模型

刚性自由度（质量 $m_s$、刚度 $k$、阻尼 $c$）被不可压流体包裹，流体对界面加速度 $a=\ddot d$ 的反作用力

$$G(a) = F_{ext} - m_a a \qquad(\text{附加质量的定义})$$

$m_a$ 由 1D 不可压通道精确给出：长 $L$、截面 $A$、密度 $\rho_f$ 的通道被活塞封闭时
$F=-\rho_f A L\,a$，故

$$m_a=\rho_f A L \quad (\texttt{added\_mass\_from\_channel})$$

精确耦合方程（一致性问题，注意是 $m_s+m_a$）：

$$(m_s+m_a)\,a + c\,v + k\,d = F_{ext}$$

### 1.2 参数表（无量纲）

| 量 | 记法 | 取值 |
|---|---|---|
| 附加质量比 | $\beta=m_a/m_s$ | 0.25 / 1 / 2.5 / 10（C1、C3）；0.25…2.0（C4 扫描） |
| 时间步 | $\Delta t$ | 0.01（Newmark $\gamma=1/2,\ \beta_N=1/4$） |
| 界面松弛 | $\omega$ | 1（不松弛）/ 常数 / Aitken 动态 |
| 收敛容差 | tol | $10^{-6}$（C3）、$10^{-10}$（C1 测量） |

---

## 2. 怎么做（方法）

### 2.1 分区迭代的统一收敛因子

分区迭代固定点：$a^{k+1}=S(G(a^k))$，其中结构算子用**迭代质量** $M_{eff}$ 求解 Newmark 方程

$$S(F)=\frac{F+\text{known}}{M_{eff}},\qquad M_{eff}=M_{iter}+c\gamma\Delta t+k\beta_N\Delta t^2$$

两者线性，界面误差以**精确**因子收缩：

$$\boxed{\ \rho(\omega)=\left|1-\omega\left(1+\beta_{eff}\right)\right|,\qquad \beta_{eff}=\frac{m_a}{M_{eff}}\ }$$

由此得到两种工程配置（本模块 `variant`）：

| 配置 | $M_{iter}$ | $\rho$（$\omega=1$） | 临界 $\beta$ |
|---|---|---|---|
| `explicit`（朴素交错） | $m_s$ | $\rho=\beta$ | $\beta_{crit}=1$ → $\beta>1$ 发散 |
| `wet_mass`（结构侧已含附加质量） | $m_s+m_a$ | $\rho=\dfrac{\beta}{1+\beta}<1\ \forall\beta$ | 无有限临界值，但 $\rho\to1$（$\beta\to\infty$） |

`wet_mass` 配置即任务书给的定标式 $\rho=\beta/(1+\beta)$，物理含义清晰：**流体质团占系统总惯性
（$m_s+m_a$）的比例**就是每次子迭代残留的误差放大率；$\beta=10$ 时 $\rho=0.909$，要收敛到 $10^{-6}$
需要约 145 次子迭代——所以尽管不发散，也必须上加速器。

> 诚实说明：`wet_mass` 配置把附加质量同时放进结构质量算子与流体反力，收敛到的固定点是
> $(m_s+2m_a)a=F_{ext}$，比精确解偏小（double counting）。因此工程上必须用下面的
> θ-split 一致化写法（或 Robin 界面条件）。本节点用 `wet_mass` 只做**迭代率**定标，
> 用 `explicit` 做**物理一致性**校验（见 §3.4）。

### 2.2 一致性保持的 θ-split 族

把附加质量拆成隐式 $\theta m_a$ + 显式 $(1-\theta)m_a$：

$$\rho(\theta)=\frac{(1-\theta)\beta}{1+\theta\beta}$$

固定点始终是精确解 $(m_s+m_a)a=F_{ext}$（对任意 $\theta$）。$\theta=0$ 退化为朴素交错，$\theta=1$ 为
全隐式 Robin 耦合（$\rho=0$，一步收敛）。这条曲线（图 5）给出了"用多少隐式附加质量换多少收敛率"的定标。

### 2.3 Aitken 动态松弛（Küttler–Wall / Irwin–Kesavan 向量形式）

$$\omega_k=-\omega_{k-1}\frac{r_{k-1}\cdot\left(r_k-r_{k-1}\right)}{\lVert r_k-r_{k-1}\rVert^2},
\qquad x^{k+1}=x^k+\omega_k r^k$$

对线性界面算子，**第一次更新就给出最优松弛** $\omega^\ast=1/(1+\beta_{eff})$，使 $\rho=0$，两步收敛。
模块同时保留任务书里给出的"比值形式" $\omega_k=\omega_{k-1}(1-\Delta r_k/\Delta r_{k-1})$
（`AitkenRatio`）用于对照——它不是标准动态松弛，实测会把 $\omega$ 压到下限、收敛反而变差（§3.5）。

### 2.4 IQN-ILS（Degroote 2009）

界面残差 $r(x)=H(x)-x$，用历史的 $(x_j,r_j)$ 构造多割线：解最小二乘

$$\min_c\left\lVert r_k-\sum_j c_j\,\Delta r_j\right\rVert_2,\qquad
x_{k+1}=x_k-\sum_j c_j\,\Delta x_j$$

实现含列范数过滤（`drop_tol`）、旧样本回收（`max_pairs`）、步长限幅（`step_limit`），
并把隐含松弛 $\omega=|\Delta x\cdot r|/|r|^2$ 作为可观测量输出。对线性界面算子单割线即精确 → 3 次迭代到位。

### 2.5 工具链

`numpy` 实现加速器；`pytest` 做判据；`matplotlib` 出图；`scipy.integrate.solve_ivp` 做精确解基准；
重计算（VIV 探针）走 外协方集群（`run_cluster`，loong64 双节点）。

---

## 3. 计算结果

### 3.1 C1 — 1D 解析收敛因子 vs 数值实测

`measure_contraction_1d` 在固定时间步的子迭代问题里测残差几何平均比
$\rho_{num}=\exp(\frac1W\sum\ln\lVert r_{i+1}\rVert/\lVert r_i\rVert)$（跳过 3 步暂态，窗口 10 步）。

| $\beta$ | $\rho$ 实测（explicit） | 解析 $\beta$ | $\rho$ 实测（wet mass） | 解析 $\beta/(1+\beta)$ | 相对误差 |
|---|---|---|---|---|---|
| 0.25 | 0.2500000 | 0.25 | 0.2000000 | 0.2 | $<10^{-15}$ |
| 1.0  | 1.0000000 | 1.0  | 0.5000000 | 0.5 | $<10^{-15}$ |
| 2.5  | 2.5000000 | 2.5  | 0.7142857 | 0.714286 | $<10^{-15}$ |
| 10.0 | 10.0000000 | 10.0 | 0.9090909 | 0.909091 | $<10^{-15}$ |

**判据 C1：最大相对误差 7.1e-16 < 5% → PASS**（4 个 $\beta$ 值，远超"≥3 个"）。

### 3.2 C1b — 松弛迭代的完整公式校验

$\beta=1$ 时扫描 $\omega\in\{0.05,0.2,0.4,0.6,0.8,1.0\}$，实测 $\rho_{num}$ 与 $|1-2\omega|$
逐点吻合（相对误差 $<10^{-8}$，图 2）。这条公式是所有加速器设计的出发点。

### 3.3 C2 — 临界附加质量比

解析：$\rho=\beta\Rightarrow\beta_{crit}=1$。
数值：以"实测 $\rho\ge1$ 即发散"为判据，对 $\beta\in[0.1,8]$ 做 30 次二分：

* 实测 $\beta_{crit}=1.0000003$（括号 $[0.9999998,\ 1.0000008]$）
* 相对误差 $2.8\times10^{-7}$

**判据 C2：误差 $<10^{-6}$ < 10% → PASS**。扫描点（$0.9$ 收敛 / $1.1$ 发散）与边界一致。

### 3.4 物理一致性校验（C1 的支撑证据）

用 `explicit`（一致）配置跑 500 步 Newmark 时域解（$m_s=1,m_a=0.5,k=4,c=0.2$，$F_{ext}=0.7$），
与 `solve_ivp`（rtol 1e-12）积分同一 ODE 的位移比较：**相对误差 $<5\times10^{-3}$**（测试
`test_partitioned_solution_matches_exact_ode`）。这说明分区机器本身（Newmark + 子迭代 + 界面残差）
不是"好看的数字"，而是真在解耦合方程。

### 3.5 C3 — $\beta=10$ 的稳定化

| 配置 | 加速器 | 子迭代数 | 最终残差 | 收敛 |
|---|---|---|---|---|
| wet_mass（$\rho=0.909$） | 无 | 97 | $9.7\times10^{-7}$ | 是（慢） |
| wet_mass | Aitken | 3 | $\sim10^{-19}$ | 是 |
| wet_mass | IQN-ILS | 3 | 0（机器零） | 是 |
| explicit（$\rho=10$，发散） | 无 | 15（截断） | 残差涨到 $10^{13}$ | **否** |
| explicit | Aitken | 3 | $<10^{-8}$ | 是 |
| explicit | IQN-ILS | 3 | $<10^{-8}$ | 是 |

**判据 C3：IQN-ILS 在 $\beta=10$ 下残差 $<10^{-6}$（实际为 0，3 次迭代）→ PASS**；
同时给出"不加加速器时朴素交错直接发散（$10^{40}$）"的负控证据（图 3）。

### 3.6 C4 — Aitken 松弛因子自适应区间

$\beta\in[0.25,2]$ 扫描 8 点，Aitken 第一次更新即给出 $\omega=1/(1+\beta)$：

* 实测 $\omega\in[0.3333,\ 0.8000]$ → **落在 $[0.3,0.8]$**
* 与解析最优值的最大相对误差 $<10^{-9}$
* 松弛后 $\rho\to0$（动态松弛精确湮灭线性误差）

**判据 C4：PASS**（图 4）。注意 $\beta=10$ 时 $\omega^\ast=1/11=0.091$，天然在 $[0.3,0.8]$ 之外——
验收区间对应 $\beta\in[0.25,2.33]$，这一点在报告里明确标注，不做模糊处理。

对照：任务书"比值形式" $\omega_k=\omega_{k-1}(1-\Delta r_k/\Delta r_{k-1})$ 实测把 $\omega$ 压到下限
（0.05），$\beta=0.25$ 时 $\rho$ 从 0 恶化到 0.92。**结论：实际应采用标准动态松弛形式**，
比值形式只作为文档化对照保留（`AitkenRatio`，测试
`test_aitken_ratio_form_is_documented_as_non_optimal`）。

### 3.7 C5 — VIV 锁定区验证：**FAIL（如实报告）**

计划：D2Q9 + 弹性支撑圆柱，$Re=100$、$m^\*=10$、$\zeta=0$，扫 $U^\*=3\ldots7.5$ 与文献
（$A/D\approx0.5$、锁定区 $U^\*\approx4$–$7$）比较（容差 10%）。实际未达成：

| 探针 | 固定圆柱 $C_d$（$Re=100$） | $St$ | 参考值 | 状态 |
|---|---|---|---|---|
| 直接力 IBM（`fsi_viv_ibm.py`） | 32.27（$D=16$） | 0.213 | $C_d=1.34$, $St=0.164$ | 载荷放大约 24 倍；$D=12$ 时 2000 步后 NaN |
| 半步反弹 + 动量交换（`fsi_viv_bb.py`） | 0.687（$D=12$） | 1.92（伪） | 同上 | 动量交换载荷出现周期 2 棋盘振荡 |

根因分析（已写入报告 JSON 与模块 docstring）：
1. 直接力 IBM 在密集标记下每个标记都成为独立动量源，重叠标记把界面总力放大；且 $Re=100$ 要求
   $\tau=0.53$，BGK + 密集内部填充在该松弛时间下不稳定。
2. 半步反弹法的流体侧稳定，但阶梯边界上动量交换式使用了固体节点反弹后的分布，
   与流体侧入射分布配对不当 → 载荷逐步反号（棋盘混叠）。修正方向已知（改为纯流体侧
   对称配对并核对 Ladd 修正符号），但需要重新标定与更长的算例预算。
3. 集群 loong64 节点上 torch 在 `OMP_NUM_THREADS>1` 时出现非确定性 `free(): corrupted unsorted chunks`
   崩溃，进一步压缩了可用的调试迭代次数。

**结论：C5 判 FAIL，未产出任何 VIV 振幅数字，不做与文献的对比，也不声称通过。** 两个探针模块
`VALIDATION_STATUS = "WITHHELD"`，由测试 `test_viv_probes_declare_withheld_validation_status`
机器把关，避免下游误用。证据：`artifacts/fsi_partitioned/viv_validation_status.json`（含集群 job id）。

### 3.8 C6 — pytest

`python -m pytest tests/test_fsi_partitioned.py -q` → **19 passed**（判据要求 ≥10）。

---

## 4. 图

| 文件 | 内容 |
|---|---|
| `figures/fig1_contraction_vs_beta.png` | $\rho$ vs $\beta$：解析两条曲线 + 实测点 + $\beta_{crit}=1$ |
| `figures/fig2_relaxation_formula.png` | $\rho(\omega)$ 解析 vs 实测（$\beta=1$） |
| `figures/fig3_stabilisation_beta10.png` | $\beta=10$ 残差史：无/Aitken/IQN-ILS，两种配置 |
| `figures/fig4_aitken_omega.png` | $\omega$ vs $\beta$：Aitken、比值形式、解析最优、验收带 |
| `figures/fig5_theta_split.png` | $\rho(\theta)$：显式→隐式连续过渡 |
| `figures/fig6_viv_validation_status.png` | C5 失败证据：$C_d$、$St$ 与参考值对比 |

---

## 5. 注意点（踩坑与适用性）

1. **"$\rho=\beta/(1+\beta)$" 的适用面**：它是 `wet_mass`（结构侧已含附加质量）配置的迭代率。
   工程实现若真把附加质量同时放进结构算子与流体反力，会 double counting，固定点偏小；
   要保一致性请用 θ-split / Robin 界面条件（§2.2，图 5）。
2. **朴素交错的 $\beta_{crit}=1$ 会被刚度抬高**：真实判据是 $\beta_{eff}=m_a/M_{eff}<1$，
   $M_{eff}=m_s+c\gamma\Delta t+k\beta_N\Delta t^2$——弹簧-阻尼项增大有效动态质量，即小时间步/高刚度
   会改善稳定性（测试 `test_beta_critical_scales_with_dt_through_newmark_mass`）。
3. **Aitken 用残差本身，不应用残差差分**：写成 $\Delta r_{k-1}\cdot(\Delta r_k-\Delta r_{k-1})$ 会把
   $\omega$ 单调压向 0；正确形式见 §2.3。
4. **IQN-ILS 的 3 次迭代收敛是线性界面算子的性质**，不能外推到非线性（真实流动）情形；
   非线性下需要 `max_pairs`/`drop_tol`/步长限幅，且收敛率取决于界面算子的谱。
5. **单自由度、线性附加质量模型**是定标工具，不是通用 FSI 求解器；多自由度时 $\beta$ 变成矩阵，
   收敛性由附加质量矩阵与结构质量矩阵的广义特征值决定。
6. **C5 未完成**：流场侧载荷标定是本次唯一硬缺口，任何"VIV 振幅"结论都无据。

---

## 6. 误差判定（逐项，显式回答）

| 关键量 | 基准 | 误差 | 是否 <3% |
|---|---|---|---|
| $\rho$（4 个 $\beta$，wet mass） | $\beta/(1+\beta)$ | $<10^{-15}$ | **是** |
| $\rho$（4 个 $\beta$，explicit） | $\beta$ | $<10^{-15}$ | **是** |
| $\rho(\omega)$（6 个 $\omega$） | $\lvert1-\omega(1+\beta)\rvert$ | $<10^{-8}$ | **是** |
| $\beta_{crit}$ | 1 | $2.8\times10^{-7}$ | **是** |
| $\omega_{Aitken}$ | $1/(1+\beta)$ | $<10^{-9}$ | **是** |
| 时域位移（vs `solve_ivp` 精确解） | 数值 ODE | $<5\times10^{-3}$ | **是** |
| 固定圆柱 $C_d$（$Re=100$） | 1.34 | $+2300\%$ | **否（C5 FAIL）** |
| 固定圆柱 $St$（$Re=100$） | 0.164 | $+30\%$（IBM，未验证） | 否 |

---

## 7. 复现

```bash
PYTHONPATH=src python experiments/fsi_partitioned_scaling.py oned     # 1D 判据 (秒级)
PYTHONPATH=src python experiments/fsi_partitioned_figures.py          # 图 + criteria.json
PYTHONPATH=src python -m pytest tests/test_fsi_partitioned.py -q      # 19 条
# VIV 探针（重计算，走集群；当前为 WITHHELD）
python experiments/fsi_partitioned_scaling.py fixed
```

## 8. 参考文献

* Causin, Gerbeau & Nobile (2005) *CMAME* 194, 4506–4527 — added-mass effect 与分区算法设计。
* Förster, Wall & Ramm (2007) *CMAME* 196, 1278–1293 — 顺序交错耦合的人工附加质量不稳定性。
* Küttler & Wall (2008) *Comput. Mech.* 43, 61–72 — 动态松弛 FSI 不动点求解器。
* Degroote, Bathe & Vierendeels (2009) *Comput. Fluids* 38, 1202–1213 — IQN-ILS 界面拟牛顿。
* Irwin & Kesavan (2005) *Comput. Struct.* 83, 1177–1193 — Aitken 松弛的分区仿真框架。