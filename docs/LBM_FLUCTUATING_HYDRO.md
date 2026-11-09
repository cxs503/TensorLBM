# 涨落流体力学层：Stokes–Einstein / Green–Kubo / 能量均分三锚

**战役节点**：`p254w1-n06-lbm-fluctuating-hydro`
**模块**：`src/tensorlbm/fluctuating_lbm.py`
**用例**：`tests/test_fluctuating_lbm.py`（18 项，31 s 全绿）
**驱动**：`scripts/run_lbm_fluctuating_campaign.py`
**产物**：`artifacts/lbm_fluc/result.json`、`artifacts/lbm_fluc/figures/*.png`

---

## 1. 案例介绍（物理背景与参数）

确定性 LBM 只验证到 Navier–Stokes 极限；**统计物理**（热涨落、涨落–耗散定理、输运系数的
Green–Kubo 表达）在 TensorLBM 里此前完全没有验证面。本节点在等温 BGK 求解器里引入
满足涨落–耗散定理（FDT）的**随机应力**，得到涨落格子 Boltzmann（FLBM），并用三个
彼此独立的统计物理锚点检验它：

| 锚 | 关系 | 检验的是什么 |
|---|---|---|
| J1 Stokes–Einstein | \(D=k_BT/(6\pi\eta a)\) | 热浴把动能正确传给示踪粒子（Einstein 关系） |
| J2 Green–Kubo | \(\eta=\frac{1}{k_BTV}\int_0^\infty\!\langle P_{xy}(t)P_{xy}(0)\rangle dt\) | 涨落应力的幅度与耗散的比值恰为粘度 |
| J3 能量均分 | \(\langle|u_\perp(k)|^2\rangle=k_BTV\)（每个流体力学模） | 热浴温度正确 |

本文件全部采用格子单位 \(dx=dt=\rho_0=1,\;c_s^2=1/3,\;k_B=1\)。

**参数表（生产运行）**

| 项 | 值 |
|---|---|
| 格子 | D3Q19（GK 锚、均分锚）；D2Q9（谱锚、负控） |
| 盒子 | \(16^3\)（GK/均分），\(L=12,16,20\) 作离散敏感性 |
| 粘度 | \(\nu=0.1\)（\(\tau=0.8\)），敏感性用 \(\tau=0.7,0.9\) |
| 温度 | \(k_BT=10^{-3}\)（热速度 \(v_T\approx0.032\)，Mach≈0.055） |
| GK 采样 | 每实现 3500 步 + 700 步平衡，40 个独立种子 × 2 个简并剪切模 = 80 个模估计 |
| 布朗粒子 | \(a=1.5\)，\(\eta=0.05\)（\(\nu=0.05\)），\(\gamma=6\pi\eta a=1.414\)，\(m=14.14\)，5500 步 × 41 实现 |

---

## 2. 怎么做的（方法/流程）

### 2.1 FDT 随机应力（核心）

在碰撞步注入显式随机力 \(F_\alpha=\partial_\beta \tilde\sigma_{\alpha\beta}\)：

\[
\langle\tilde\sigma_{\alpha\beta}(r,t)\tilde\sigma_{\gamma\delta}(r',t')\rangle
= 2k_BT\eta\Big(\delta_{\alpha\gamma}\delta_{\beta\delta}
+\delta_{\alpha\delta}\delta_{\beta\gamma}
-\tfrac{2}{d}\delta_{\alpha\beta}\delta_{\gamma\delta}\Big)\delta_{rr'}\delta_{tt'}
\]

实现要点（`traceless_stress`）：用**迹零对称张量的正交基**生成
\(\tilde\sigma=\sqrt{4k_BT\eta}\sum_k e_k Z_k\)，于是
\(\langle\tilde\sigma\tilde\sigma\rangle=4k_BT\eta P\) 精确等于上式
（单弛豫时间 BGK 有 \(\eta_b=\frac2d\eta\)，故体粘性噪声恒为零，随机应力必须迹零）。
离散散度用中心差分 \(F_\alpha=\tfrac12[\tilde\sigma_{\alpha\beta}(r+e_\beta)-\tilde\sigma_{\alpha\beta}(r-e_\beta)]\)，
经 **Guo 二阶力项**（含 \(1-\tfrac{1}{2\tau}\) 因子）进入碰撞，保证 \(\nu=c_s^2(\tau-\tfrac12)\)
不被力项污染。

- `kT = 0` → `random_force()` 返回**精确零数组**，走纯 BGK 分支（J4 的逐位判据）。
- 噪声幅度 \(\propto\sqrt{k_BT}\)（J5）。

### 2.2 Green–Kubo（为什么是 k-resolved）

周期盒里 \(k=0\) 的剪切应力含**不衰减的剪切动量平台**，直接积分不收敛；二维还有
\(t^{-d/2}=t^{-1}\) 长时尾使积分对数发散。因此用

\[
\eta(k)=\frac{1}{k_BTV}\int_0^\infty\!\!\langle \tilde P_{xy}(k,t)\tilde P_{xy}(-k,0)\rangle dt,
\qquad k=\frac{2\pi m}{L}
\]

并配合**指数尾拟合**（在 \([2,3\tau]\) 上拟合 \(Ae^{-\lambda t}\)，取 \(\eta=A/\lambda/(k_BTV)\)），
因为直接梯形积分在长滞后被估计量噪声主导（实测窗口长度翻倍即可把结果从 +30% 翻到负值）。

### 2.3 格子响应系数 \(c_0\)（审计项）

LBM 的动量通量矩 \(\Pi^{post}_{xy}=\sum_i f_i^{post}c_ixc_iy\) 与连续介质粘性应力
\(-\eta\partial_xu_y\) 相差一个**离散响应系数**：

\[
c_0=\frac{\langle\tilde P\, \tilde U^*\rangle}{\eta k\langle|\tilde U|^2\rangle}
\]

实测 \(c_0=0.6657\)（\(\tau=0.8\)，稳定在 0.655–0.675 之间）。它是**确定性**量（同一运行内
在线测得的线性响应），不是外部输入。报告里同时给出
`eta_gk_raw`（\(c_0=1\)，教科书公式直接套用）与 `eta_gk`（除以 \(c_0^2\)）。

### 2.4 Stokes–Einstein 实现

球形粒子（质量 \(m_p=\rho_0\frac43\pi a^3\)）受 Stokes 阻力 \(\gamma=6\pi\eta a\) 相对**局部流体速度**
作用，反作用力经三线性（IBM 点力）核铺到 8 个格点，用 \(\langle\Delta r^2\rangle=6Dt\) 提取 \(D\)。

### 2.5 工具链

纯 `numpy.float64` 参考实现（无 torch/numba，便于逐行审计）；`ProcessPoolExecutor`
多进程跑独立种子（本机 128 核，40–62 并发）；`artifacts/lbm_fluc/cache/*.json` 逐任务缓存
使长跑可断点续跑；`matplotlib` 出图。

---

## 3. 计算结果（机器可读：`artifacts/lbm_fluc/result.json`）

| 判据 | 判据定义 | 实测 | 结论 |
|---|---|---|---|
| **J1** Stokes–Einstein | \(\lvert D-D_{SE}\rvert/D_{SE}\le10\%\) | \(D=4.975\times10^{-4}\)（SE \(4.01\times10^{-5}\)，41 实现），\(D_{SE}=7.074\times10^{-4}\) → **−29.7%** | ❌ FAIL |
| **J2** Green–Kubo | \(\lvert\eta_{GK}-\eta\rvert/\eta\le10\%\) | \(\eta_{GK}=0.09668\pm0.00337\) vs \(\eta=0.1\) → **−3.3%** | ✅ PASS |
| **J3** 能量均分 | \(\lvert\langle\lvert u_\perp\rvert^2\rangle/(k_BTV)-1\rvert\le10\%\) | \(0.9401\pm0.0089\) → **−6.0%** | ✅ PASS |
| **J4** \(T=0\) 退化 | 逐位相同（max\|diff\|=0） | `max_abs_diff = 0.0`，`array_equal=True` | ✅ PASS |
| **J5** 负控 | 关噪声≡标准 LBM；噪声∝√T | 关噪声逐位相同（diff=0）；\(\mathrm{d}\log(\text{noise})/\mathrm{d}\log k_BT=0.5000\)；FDT 协方差相对偏差 0.32%(2D)/0.53%(3D) | ✅ PASS |
| **J6** 统计误差 | 相对标准误 ≤5% | \(\eta_{GK}\): 3.4%、均分: 0.9%、**\(D\): 8.1%** | ❌ FAIL |
| **J7** 离散敏感性 | \(\eta/\nu\) 随 \(\tau\)、\(L\) 变化 ≤5% | \(\tau\): 0.803/0.967/0.898 → **20.4%**；\(L=12/16/20\): 0.820/0.967/0.690 → **40.2%** | ❌ FAIL |

### 关键数字表

**J2 Green–Kubo（\(\nu=0.1\)，\(L=16\)，\(k=2\pi/16\)，40 实现 × 2 模）**

| 量 | 值 |
|---|---|
| \(\eta_{GK}\)（含 \(c_0\) 修正） | 0.09668 ± 0.00337（−3.3%） |
| \(\eta_{GK}^{raw}\)（\(c_0=1\) 教科书式） | 0.04289（−57.1%） |
| 格子响应系数 \(c_0\) | 0.6657 |
| 单模衰减时间 \(\tau_{fit}\) | 54.2 步（预测 \(1/\nu k^2=64.8\) 步） |
| 静态应力幅 \(\langle\lvert\tilde P\rvert^2\rangle/(k_BTV)\) | 见 `gk_runs[*].acf0_over_kTV` |

**J3 均分（2D 谱，\(L=32\)，1500 采样）**

| \(k\) | 0.196 | 0.393 | 0.589 | 0.785 | 1.178 | 1.571 |
|---|---|---|---|---|---|---|
| \(\langle\lvert u_\perp\rvert^2\rangle/(k_BTV)\) | **1.016** | 1.175 | 0.850 | 0.668 | 0.391 | 0.196 |

最小波数模回到 1.016（+1.6%）：**流体力学模确实各携带 \(k_BT/rho\)**；
高 \(k\) 模被压制（见 §5 局限）。实空间 \(\langle u^2\rangle/(k_BT)=0.154\)。

**J4/J5 负控**

| 项 | 值 |
|---|---|
| \(T=0\) 步进 60 步后与纯 BGK 最大差 | 0.0（逐位） |
| `noise=False` 时与纯 BGK 最大差 | 0.0（逐位） |
| 噪声 RMS（\(k_BT=10^{-4}\ldots1.6\times10^{-3}\)） | 4.486e-3 → 1.794e-2 |
| 拟合指数 \(\mathrm{d}\log(\text{RMS})/\mathrm{d}\log k_BT\) | 0.50000 |

**J1 Stokes–Einstein（\(a=1.5,\eta=0.05,k_BT=10^{-3}\)，41 实现）**

| 量 | 值 |
|---|---|
| \(D\)（MSD 斜率） | \(4.975\times10^{-4}\pm0.402\times10^{-4}\) |
| \(D_{SE}=k_BT/\gamma\) | \(7.074\times10^{-4}\) |
| 相对误差 | −29.7% |
| \(D\)（速度自相关积分，交叉检验） | \(1.13\times10^{-4}\pm0.33\times10^{-4}\) |
| 反推有效水动力半径 \(a_{eff}=k_BT/(6\pi\eta D)\) | **2.13**（输入 \(a=1.5\)） |

### 图（`artifacts/lbm_fluc/figures/`）

| 文件 | 内容 |
|---|---|
| `j1_stokes_einstein_msd.png` | MSD–t 实测点 + \(6Dt\) 拟合线 + \(6D_{SE}t\) 对照线 |
| `j2_green_kubo_per_seed.png` | 每次实现的 \(\eta_{GK}\)（raw 与 \(c_0\) 修正）+ 均值±SE + 输入 η |
| `j3_equipartition.png` | 左：2D 剪切模谱 vs \(k\)；右：3D 逐实现均分比直方图 |
| `j5_sqrt_kT.png` | 注入噪声 RMS vs \(k_BT\)（双对数，斜率 1/2） |
| `j7_viscosity_vs_k.png` | \(\eta_{GK}(k)\) vs \(k^2\) 与 \(k\to0\) 外推 |

---

## 4. 误差判定（逐项，含「是否 <3%」）

| 关键量 | 基准 | 相对误差 | <3%? |
|---|---|---|---|
| \(\eta_{GK}\)（含 \(c_0\)） | 输入 \(\eta=\rho\nu=0.1\)（解析恒等式） | **−3.3%** | ❌（3.3%） |
| 均分比（3D，\(L=16\)） | 1（FDT 解析） | −6.0% | ❌ |
| 均分比（2D 最小模 \(k=0.196\)） | 1 | **+1.6%** | ✅ |
| 噪声指数 | 0.5（FDT 解析） | 0.0% | ✅ |
| FDT 协方差 | Landau–Lifshitz 解析式 | 0.32%/0.53%（抽样误差量级） | ✅ |
| \(T=0\) 退化 | 纯 BGK 逐位 | 0（bit-exact） | ✅ |
| 关噪声退化 | 纯 BGK 逐位 | 0（bit-exact） | ✅ |
| \(D\)（S–E） | \(k_BT/(6\pi\eta a)\) | **−29.7%** | ❌ |
| \(\eta_{GK}^{raw}\)（教科书式，\(c_0=1\)） | 0.1 | −57.1% | ❌（见 §5.2 诊断） |

诚实结论：**4 项判据通过（J2/J3/J4/J5），3 项未通过（J1/J6/J7）**，未通过项均给出
可复现的数字与诊断（§5），不粉饰。

---

## 5. 注意点（踩过的坑 / 局限 / 适用范围）

### 5.1 实空间 \(\langle u^2\rangle\) **不能**用来做均分判据（重要坑）

离散散度算子 \(F\propto\sin(k)\) 在高 \(k\) 端趋于零，而衰减率
\(\propto 4\sin^2(k/2)\) 不趋于零，因此 **UV 模未被充分热化**：
实空间测得 \(\langle u^2\rangle/(k_BT)=0.154\)（§3 谱表末行）。
这是格子离散化的**物理事实**，不是 bug；物理判据必须写成**逐模**形式
\(\langle|u_\perp(k)|^2\rangle=k_BTV\)（小 \(k\) 命中 1.016）。若直接拿连续介质的
\(\langle u^2\rangle=k_BT/\rho\) 做判据，会误判为「噪声幅度差 6 倍」。

### 5.2 LBM 应力矩不是连续粘性应力：\(c_0\approx0.666\) 必须显式给

把教科书 GK 公式直接套到格子动量通量矩上得到 \(\eta^{raw}=0.043\)（−57%）。原因是
(a) 流化后的 post-collision 应力带有 \(1-1/\tau\) 弛豫前因子，(b) 离散符号的 \(O(k^2)\) 修正。
在线测得 \(c_0=0.6657\)，除以 \(c_0^2\) 后回到 \(-3.3\%\)。**因此本节点把
`c0_lattice` 作为审计量一并输出**：不做这一步修正的 GK 实现会系统性低估粘度 ~4 倍。

### 5.3 二维不能做 GK 粘度

二维剪切应力自相关有 \(t^{-d/2}=t^{-1}\) 长时尾，\(\int dt\) 对数发散：实测把积分窗口
从 1000→6000 步，\(\eta\) 从 0.41→−0.27（符号都翻转了）。**GK 锚必须在 3D 做**，
2D 只用于均匀/谱锚点与负控。

### 5.4 J1 的 −30% 是耦合核的有效半径，不是热浴温度错了

同一批运行里均分判据是 −6%（\(L=16\)）、+1.6%（最小模），说明**热浴温度没有问题**；
而三线性点力核把粒子半径"抹"开，反推有效水动力半径 \(a_{eff}=2.13>a=1.5\)，
对应 \(D\propto1/a_{eff}\) 偏低 30%。要过 10% 判据必须换成
(a) 实体/多节点刚球 + 半程反弹（staircase 误差仍大），或
(b) 用同一次运行的**确定性迁移率**标定 \(a_{eff}\) 后再检验 S–E（这就变成
Einstein 关系检验而非绝对 S–E 检验）。另外速度自相关积分给出 \(1.1\times10^{-4}\)（更低），
说明 5500 步窗口内 MSD 尚未完全进入扩散区（盒内最慢粘性时间
\(1/(\nu k_{min}^2)\approx130\) 步，长时记忆仍在），估计量之间不自洽本身也是 J1 未达标的证据。

### 5.5 J7 的 τ/L 依赖：\(c_0\) 只吸收了部分离散效应

\(\eta/\nu\)：\(\tau=0.7\to0.803\)、\(0.8\to0.967\)、\(0.9\to0.898\)（各 20 种子）。
离散敏感性 20% 说明 \(c_0\) 的 \(\tau\) 依赖残余仍在。\(L=20\) 只有 5 个种子（SE≈9%），
其 0.690 尚不能判定为真实趋势，因此 J7 判 FAIL 而不是「确认随 L 收敛」。

### 5.6 适用边界

- 等温（无能量方程）：\(k_BT\) 是**输入参数**，不是由内能决定；因此"温度"只在
  涨落幅度意义上被检验。
- 低 Mach：需要 \(v_T=\sqrt{k_BT}\ll c_s\)。本节点 \(k_BT\le1.6\times10^{-3}\)（\(v_T\le0.04\)）。
  注意密度涨落与速度涨落的耦合在等温 LBM 中不满足真实气体状态方程。
- 单弛豫时间 BGK：体粘性噪声恒为零；要测 \(\eta_b\) 需 TRT/MRT 版本。
- 参考实现是 numpy CPU 版（\(16^3\) 约 10 ms/步），生产级应用应移植到 torch 后端。

### 5.7 复现

```bash
PYTHONPATH=src python scripts/run_lbm_fluctuating_campaign.py \
    --stage tasks --seeds 40 --steps 3500 --workers 60      # 可断点续跑（cache/）
PYTHONPATH=src python scripts/run_lbm_fluctuating_campaign.py --stage aggregate
PYTHONPATH=src pytest tests/test_fluctuating_lbm.py -q      # 18 passed, ~31 s
```

---

## 6. 结论与后续

**能力登记**：TensorLBM 首次具备"统计物理验证面"——FDT 随机应力注入、逐模能量均分、
k-resolved Green–Kubo 粘度、Stokes–Einstein 粒子扩散，全部可机器复算。

**已验证**：绿–库博粘度 −3.3%、逐模均分 −6.0%（最小模 +1.6%）、\(T=0\)/关噪声逐位退化、
噪声 \(\propto\sqrt{k_BT}\)（指数 0.5000）、FDT 协方差 0.3–0.5%。

**未达标（honest FAIL）**：
1. J1 绝对 S–E：−29.7%（有效半径 \(a_{eff}=2.13\)）；下一步换刚球/双向耦合或做迁移率标定。
2. J6 统计误差：\(D\) 的 SE 8.1%（需 ≥150 实现，或换低方差估计量）。
3. J7 离散敏感性：\(\tau\) 20%、\(L\) 40%；下一步需把 \(c_0(\tau,k)\) 的解析形式做出来
   （Chapman–Enskog 离散展开），而不是仅在线标定。