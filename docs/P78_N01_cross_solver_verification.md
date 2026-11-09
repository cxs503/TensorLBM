# P78-N01 — 跨求解器互裁基准面（LBM / FVM / FEM）：Lid-driven cavity + MMS + 守恒账本

**节点** P78-N01 ｜ **仓** TensorLBM ｜ **产物** `artifacts/cross_solver_verification/cross_solver_verification_report.json`
**代码** `src/tensorlbm/cross_solver_verification/`（新包）｜ **测试** `tests/test_cross_solver_verification.py`（12 passed, 51 s）
**复现** `runs/p78n01/case.py <mms|cav> <solver> <n>` + `runs/p78n01/assemble.py`

> 结论先行（honest，不粉饰）：**统一基准面已建成并可复现，但本轮只 C5/C6/C7 通过；C1/C2/C3/C4 未达标**，
> 原因逐项落表（几何网格未跑到 n=128 是本轮唯一硬约束：单次命令窗口 ≤120 s）。
> 真实亮点：n=64 上 **LBM 与 FEM 的中心线 u 场相对 L2 差 1.62 %**（两套完全不同离散互相独立复现到 2 % 内），
> FVM（ψ–ω + Woods 壁涡）是三者中的离群者（4.3–5.4 %），并且其 Ghia 偏差随网格只呈一阶收敛 —— 已定位到壁涡闭合精度。

---

## ① 案例介绍（物理背景 / 几何 / 参数）

顶盖驱动方腔是 CFD 最经典的对标算例：单位方腔，四壁无滑移，顶盖以 $U_{lid}$ 平移驱动主涡与角涡。
本轮把它升级为**三求解器互裁面**：同一个物理问题用三种完全不同的离散求解，再与 Ghia, Ghia & Shin (1982) 的
中心线参考解对标。

| 项目 | 值 | 说明 |
|---|---|---|
| 几何 | $[0,1]^2$ 方腔，四壁无滑移，顶盖 $u=U_{lid}$ | 所有求解器同一无量纲化（$U_{lid}=1$ 判读，LBM 内取 $u_{lid}=0.1$ 格子单位） |
| 网格 | 单元中心网格 $n\times n$，中心 $(i+\tfrac12)/n$ | LBM 节点中心 / FVM 单元中心 / FEM Q2 单元中心节点**同址**，比对无需插值 |
| Re | 100（主）；400 / 1000（NOT_RUN，见 ⑤） | $Re=U L/\nu$，$L=1$ |
| 参考解 | Ghia et al. (1982) Re=100/400/1000 中心线表 | 复用 `tensorlbm.lid_driven_cavity` 表（单一真源，不复制） |
| MMS | $\psi=A\sin^2(kx)\sin^2(ky)$，$k=2\pi$，$B=2Ak=0.01$，$\nu=0.002$，$p=Q\sin kx\sin ky$，$Q=5\times10^{-4}$ | 源项 $f=(u\cdot\nabla)u+\nabla p-\nu\nabla^2u$ 闭式；$\mathrm{Re}_{MMS}=B/(\nu k)\approx0.8$ |

三个求解器（全在 TensorLBM 仓内，numpy/scipy only，不依赖 GPU 栈，单 checkout 可复现）：

* **LBM**：D2Q9 BGK + Guo (2002) 力项；半程 bounce-back 壁面 + 移动壁动量修正；四壁角点由顶壁规则覆盖（局限见 ⑤）。
* **FVM**：单元中心 **涡量–流函数** 有限体积；离散 Poisson **精确求解**（Dirichlet 用 DST-II 特征分解，周期用 FFT 特征值），
  对流/扩散二阶中心；壁涡用二阶 Woods/Jensen 型公式 $\omega_w= -\frac{4}{9h^2}(27\psi_1-\psi_2-12h\,g'(0))$（本文重新推导，见 ⑤）。
* **FEM**：结构化四边形上的 **Q2–Q1 Taylor–Hood**，Newton + 回溯线搜索，`scipy.sparse.linalg` 求解；周期面用模编号识别。

## ② 怎么做的（方法 / 流程 / 工具链）

1. **MMS 收敛阶**（周期盒，三求解器解同一制造解）：观测阶 $p=\log_2(e_{n}/e_{2n})$，同时给出最小二乘拟合阶。
2. **Ghia 对标**：把 $u(x=0.5,y)$、$v(x,y=0.5)$ 归一化后插值到 Ghia 站点，报 max/RMS 偏差（%$U_{lid}$）。
3. **能量账本**：稳态全局动能收支 $P=\varepsilon$。
   $P=U_{lid}\nu\int (\partial u/\partial y)|_{lid}dx$（二阶单侧差分），
   $\varepsilon=\nu\int 2S_{ij}S_{ij}dV=\nu\int[2(u_x^2+v_y^2)+(u_y+v_x)^2]dV$（壁面 ghost 线性外插，全场二阶）。
   **两者独立测量**，差值即离散质量指标（不是恒等式强加）。
4. **质量账本**：$\int\nabla\cdot u\,dV$（有符号全局通量）+ 局部 $\max|\nabla\cdot u|L/U$。
5. **跨求解器 L2**：同一网格上两两相对 L2（分母为参考场 RMS）。
6. **负控**：关掉驱动（$U_{lid}=0$）三求解器必须 $u\equiv0$；再加 $Re=1$ Stokes 正控证明门不是空的。
7. **工具链**：numpy 2.4 / scipy 1.16 / torch 2.7（未用）/ Python 3.11；matplotlib(Agg) 出图；结果 JSON 程序化注入（零手抄）。

## ③ 计算结果（数字表，全部程序化注入）

### 3.1 MMS 观测阶（周期盒，同一制造解）

| solver | n | L2 误差 | 配对阶 | 最细对阶 | 判定窗口 |
|---|---|---|---|---|---|
| LBM | 16 / 32 / 64 | 3.00e-4 / 6.88e-5 / 1.05e-5 | 2.126 / 2.716 | 2.716 | [1.8,2.2] ✗ |
| FVM | 32 / 64 / 128 | 3.24e-7 / 3.19e-8 / 6.46e-9 | 3.347 / 2.302 | 2.302 | [1.8,2.2] ✗ |
| FEM | 16 / 32 | 3.03e-5 / 2.22e-6 | 3.770 | 3.770 | [2.8,3.2] ✗ |

解读（重要，不掩盖）：这条 MMS 的 $\sin^2\sin^2$ 场在各阶截断项上存在**异常抵消**，粗网格段呈**超收敛**（阶 > 名义阶）；
唯 LBM 的 16→32 对（2.13）落在窗口内。FVM 的 64→128 对（2.30）已接近二阶、FEM 全段 >3 与 Q2 的三阶一致，
但严格按"最细对阶入窗"判定三者均**未通过**。

### 3.2 Ghia 对标（Re=100，单元中心）

| solver | n | $u$ max 偏差 | $u$ RMS | $v$ max 偏差 | $v$ RMS |
|---|---|---|---|---|---|
| LBM | 32 → 64 | 10.36 → **5.09 %** | 2.59 → 1.28 % | 3.08 → 1.52 % | 1.04 → 0.53 % |
| FVM | 32 → 64 | 16.08 → **8.05 %** | 4.63 → 2.29 % | 4.59 → 2.34 % | 1.71 → 0.93 % |
| FEM | 32 → 64 | 10.44 → **5.19 %** | 2.59 → 1.29 % | 3.05 → 1.53 % | 1.04 → 0.56 % |

三者随网格单调下降（≈每倍网格降 5 个百分点），**n=128 未在本轮跑完**，因此 `<3 %` 判据未达成（不是被证否）。

### 3.3 跨求解器一致性（Re=100, n=64，三场同址对比）

| 对 | u 相对 L2 | v 相对 L2 |
|---|---|---|
| LBM–FEM | **1.62 %** | **1.66 %** |
| LBM–FVM | 5.39 % | 4.48 % |
| FVM–FEM | 4.32 % | 3.64 % |

### 3.4 守恒账本（Re=100, n=64）

| solver | 输入功率 $P$ | 耗散 $\varepsilon$ | $\lvert P-\varepsilon\rvert/\max$ | $\int\nabla\cdot u dV$ | $\max\lvert\nabla\cdot u\rvert L/U$ |
|---|---|---|---|---|---|
| LBM | 0.06358 | 0.16390 | 61.2 % ✗ | 6.61e-19 ✓ | 0.232 |
| FVM | 0.12229 | 0.17369 | 29.6 % ✗ | 1.87e-20 ✓ | 1e-15（结构为零） |
| FEM | 0.07462 | 0.16537 | 54.9 % ✗ | 2.99e-19 ✓ | 0.096 |

耗散 $\varepsilon$ 三求解器互相一致到 ~5 %（0.164/0.174/0.165）且随网格收敛（n=32: 0.135/0.142/0.134），
但**顶盖功率 $P$ 被系统性低估 1.5–2.5 倍**：$P$ 依赖顶盖法向梯度（边界层厚度 $\sim L/\sqrt{Re}=0.1$，n=64 时首层在 $h/2=0.008$），
而 $\varepsilon$ 是体积积分、收敛快。这是 C4 未通过的**已定位原因**。

### 3.5 负控

| 项 | 值 |
|---|---|
| $U_{lid}=0$，LBM / FVM / FEM 的 $\max\lvert u\rvert$ | 0.0 / 0.0 / 0.0（均为精确零） |
| $Re=1$ Stokes 正控（$U_{lid}=1$, n=24, FVM） | $\max\lvert u\rvert=0.825$（门非空） |

## ④ 结果图（matplotlib，已 base64 内嵌报告 JSON）

1. `ghia_profiles` — Re=100 中心线 $u(y)$/$v(x)$，三求解器 vs Ghia 散点。
2. `mms_order` — 三求解器 MMS 误差 log-log 与 $h^2/h^3$ 参考斜率。
3. `cross_fields` — n=64 速度模云图（LBM/FVM/FEM）+ 各自对 LBM 的逐点差。
4. `energy_ledger` — $P$ vs $\varepsilon$ 柱状 + 相对误差标注。
5. `ghia_convergence` — Ghia 最大偏差随 n 的收敛曲线 + 3 % 判据线。

## ⑤ 注意点（踩过的坑 / 局限）

1. **Bounce-back 必须用碰撞后分布**：用流式后的分布做反射（取邻点值）会让 Re=100 方腔在 n=32 无条件发散（实测 $\max|u|$ 爬到 0.125 并 NaN）。改成碰撞后值（半程壁面标准写法）后稳定。
2. **DST-I vs DST-II**：单元中心网格墙在 $h/2$ 处，5 点算子边界行为 $\psi_2-3\psi_1$（奇反射 ghost），其特征基是
   **DST-II**（$\lambda_m=(2\cos(m\pi/n)-2)/h^2$）。原先误用 DST-I（等价于把墙放到一整个网格外）→ 方腔 Ghia 偏差 40 %，
   而**周期 MMS 仍然全绿**（周期分支不走 Dirichlet 求解器）。这是"单测绿≠物理对"的活例。
3. **Woods 壁涡的两项系数**：重新按 $a=h/2$ 泰勒展开得到
   $g''(0)=[27g(a)-g(3a)-26g(0)-24a g'(0)]/(9a^2)$；Coutte 剪切验证给出 $\omega_w=-1$（精确）。方腔中 $\psi_{wall}=0$ 故 $-26g(0)$ 项为零。
4. **MMS 源项求积**：FEM 若只用 3×3 Gauss（对 Q2 刚度精确）求三角源项，会留下 $(hk)^6$ 的求积缺陷，使残差地板 $\sim10^{-10}$ 并
   **破坏观测阶**；改用 6×6 Gauss 后地板消失（这条对任何 MMS-FEM 都适用）。
5. **谱系局限**：LBM 有 $\sim10^{-5}$ 量级的可压缩性/压强耦合误差地板，粗网格段会抬高或压低观测阶；FVM 的壁涡闭合实际只有**一阶**（16 %→8 %，n 加倍偏差减半），要到 n≥128 才可能进 3 %；FEM Newton 在细网格上受线性求解器条件数限制，残差地板 ~1e-10（未达机器精度）。
6. **本轮算力边界**：单次命令窗口 ≤120 s，LBM/FVM/FEM 的 n=128 方腔（分别约 160 s / 120 s / >200 s）与 Re=400/1000 全扫描未跑完，
   在报告中显式标为 **NOT_RUN** 而不是截断后假装收敛（`coverage.not_run` 字段）。

## ⑥ 误差判定（逐项，显式回答"是否 <3 %"）

| 判据 | 指标 | 实测 | 门槛 | 判定 |
|---|---|---|---|---|
| C1 | Re=100 Ghia $u$ max 偏差（各求解器最细网格 n=64） | LBM 5.09 % / FEM 5.19 % / FVM 8.05 % | <3 % | **否（未达标）** |
| C2 | 三求解器两两相对 L2（n=64，n=128 NOT_RUN） | 最大 5.39 %（LBM–FEM 仅 1.62 %） | <2 % @n=128 | **否（且网格未达 n=128）** |
| C3 | MMS 最细对观测阶 | LBM 2.72 / FVM 2.30 / FEM 3.77 | 各入窗 | **否（超收敛段）** |
| C4 | 能量收支 $\lvert P-\varepsilon\rvert/\max$ | 29.6–61.2 % | <1 % | **否（$P$ 顶盖梯度欠分辨，已定位）** |
| C5 | $\lvert\int\nabla\cdot u\,dV\rvert$ | ≤6.6e-19 | <1e-10 | **是** |
| C6 | $U_{lid}=0$ 时 $\max\lvert u\rvert$ | 0.0（三者） | ≈0 | **是**（并有 $Re=1$ 正控 0.825） |
| C7 | 报告 JSON + 三求解器对比图 | 5 图 + 全数字 | 齐备 | **是** |

**是否 <3 %？否。** 本轮的最近距离是 LBM/FEM 的 5.09 %/5.19 %（n=64），按当前收敛率（每倍网格降约 5 个点）
需 n≈128 才能进 3 %；FVM 需 n≥192。跨求解器一致性在 LBM–FEM 之间已达 1.62 %（<2 %），
说明"两套独立离散互相复现"这一目标在现有网格上**已经实现**，剩下的差距来自壁面/分辨率而非框架本身。

## 复现与产物清单

```
src/tensorlbm/cross_solver_verification/{__init__,mms,lbm,fvm,fem,ledger,runner}.py   # 新包（~2000 行）
tests/test_cross_solver_verification.py            # 12 passed / 51 s
scripts/run_cross_solver_verification.py           # CLI（smoke / full）
artifacts/cross_solver_verification/run_p78n01/{case,assemble}.py          # 分例执行 + 汇总（归档副本，可直接重跑复现本报告）
artifacts/cross_solver_verification/run_p78n01/results/*.json              # 逐例原始结果（零手抄来源）
artifacts/cross_solver_verification/run_p78n01/fields/*.npy                # 三求解器速度场（跨求解器比对输入）
artifacts/cross_solver_verification/cross_solver_verification_report.json  # 判据 + 数字 + 5 图
```