# XVAL-LBM-FILL-GRADIENT-NORMAL — TensorLBM vs Maarten-vd-Sande/lbm fill-gradient interface normal

Mission 17757 · repo TensorLBM · branch agent/m17757 · anchor XVAL-LBM-FILL-GRADIENT-NORMAL

## ① 问题与 ref 出处

**问题**：TensorLBM 的自由面界面法向（fill-gradient-normal）是否与成熟开源实现 Maarten-vd-Sande/lbm 的同名做法一致（owner 判据：对标误差 <15%）？

**ref 出处（零手抄，全部来自平台知识库）**：
- `oss_solutions` id=85：`Maarten-vd-Sande/lbm | NEW:free-surface-surface-normal-from-fill | fill-gradient-normal`
  - method 字段原文：「界面法向由填充率 ε 的相邻差分 n=0.5*(ε_{i-1}-ε_{i+1}, ε_{j-1}-ε_{j+1}) 给出，再用 n·e_opp>0 判据把『指向液体外』的方向用 feq+feq_opp-f_opp 重构覆盖」
  - files 字段：`lbm/streamcollide.py:181-187, lbm/streamcollide.py:96-100`
  - conventions 字段原文：「法向未归一化（只用符号）；ε 需 GAS=0、FLUID=1 的正确取值——本仓因 streamcollide.py:166 优先级 bug 使 GAS 返回 0.5，法向被低估 2×，不可照抄」
- 本仓对标对象：`src/tensorlbm/free_surface_lbm.py:642 _compute_interface_normal`（fill=mass/rho，中心差分，归一化，符号 liquid→gas）
- 开工三查：`negative_knowledge_ledger`（keywords=lbm/free-surface/normal）无同域否定条目；`knowledge_base` 无 fill-gradient-normal 条目；`oss_solutions` 命中 id=85（本单对标源）。

## ② 方法与实现

四臂对照，**所有臂吃同一 fill 场**，只变法向公式，隔离每个旋钮：

| 臂 | 公式 | 归一化 | ε 取值 |
|---|---|---|---|
| A1 TensorLBM as-is | n = -∇fill/\|∇fill\|，roll 中心差分（free_surface_lbm.py:642 逐行移植，与真实函数 max diff = 0.0 验证） | 是 | fill=mass/rho |
| A2 OSS clean | n = 0.5*(ε_{i-1}-ε_{i+1}, ε_{j-1}-ε_{j+1})（oss id=85 method 字段逐字） | 否 | GAS=0, FLUID=1 |
| A3 OSS w/ bug | 同 A2 | 否 | GAS=0.5（streamcollide.py:166 优先级 bug） |
| A4 analytic | 构造界面的精确法向（圆/球=径向，平面=常向量） | — | — |

几何：2D 圆（N=64/128/256，R=N/4）、2D 斜平面（θ=30°, N=256）、3D 球（N=32/64/128，R=N/4）、2D 圆抹平场（w=2/4/8, N=256）。周期边界（与两实现一致）。CPU torch float64。

## ③ 参数表（无量纲数）

本单为纯几何/模板 XVAL（无流动），无量纲参数=分辨率比：

| 参数 | 取值 | 说明 |
|---|---|---|
| R/dx | 16 / 32 / 64 | 圆/球半径（格数），收敛梯 |
| w/dx | 2 / 4 / 8 | 抹平界面宽度（格数），敏感性 |
| θ | 30° | 平面倾角 |
| Re, Ma, Fr | 不适用 | 无流动，纯法向模板测试 |

## ④ 对照表（ref vs got + 误差打勾叉）

**主判据（对标=交叉实现 A1 vs A2，方向误差 1-mean|cos|）**：

| 案例 | 臂 | dir_err | 角度 | <15% |
|---|---|---|---|---|
| circle-sharp N=64 | A1 vs A2 | 0.00% | 0.0 deg | ✅ |
| circle-sharp N=128 | A1 vs A2 | 0.00% | 0.0 deg | ✅ |
| circle-sharp N=256 | A1 vs A2 | 0.00% | 0.0 deg | ✅ |
| plane-sharp theta=30.0 N=256 | A1 vs A2 | 0.00% | 0.0 deg | ✅ |
| sphere-sharp N=32 | A1 vs A2 | 0.00% | 0.0 deg | ✅ |
| sphere-sharp N=64 | A1 vs A2 | 0.00% | 0.0 deg | ✅ |
| sphere-sharp N=128 | A1 vs A2 | 0.00% | 0.0 deg | ✅ |

**主判据结论：A1 vs A2 方向误差 = 0.0（全部 7 案例）→ ✅ 通过（<15%）。TensorLBM 的 fill-gradient 法向与 OSS 做法方向完全一致。**

**方法精度诊断（A1/A2 vs 解析解，属共享模板的固有属性，非实现差异）**：

| 案例 | 臂 | dir_err | 角度 | <15% |
|---|---|---|---|---|
| circle-sharp N=64 | A1 vs A4 | 4.48% | 14.3 deg | ✅ |
| circle-sharp N=128 | A1 vs A4 | 5.34% | 15.5 deg | ✅ |
| circle-sharp N=256 | A1 vs A4 | 5.10% | 15.2 deg | ✅ |
| plane-sharp theta=30.0 N=256 | A1 vs A4 | 7.83% | 21.5 deg | ✅ |
| sphere-sharp N=32 | A1 vs A4 | 10.24% | 23.8 deg | ✅ |
| sphere-sharp N=64 | A1 vs A4 | 8.55% | 21.7 deg | ✅ |
| sphere-sharp N=128 | A1 vs A4 | 8.40% | 21.4 deg | ✅ |
| circle-smeared w=2.0 N=256 | A1 vs A4 | 1.75% | 6.8 deg | ✅ |
| circle-smeared w=4.0 N=256 | A1 vs A4 | 1.18% | 4.5 deg | ✅ |
| circle-smeared w=8.0 N=256 | A1 vs A4 | 0.71% | 2.8 deg | ✅ |

幅值关系：A2（OSS 未归一化）平均幅值 = 0.586（A1 恒为 1）；A3/A2 幅值比 = 0.500（=0.5，复现知识库记载的 2× 低估）。

## ⑤ 收敛性 / 参数敏感性（≥2 点）

1. **锐界面分辨率梯（圆，sharp）**：dir_err = 4.48% → 5.34% → 5.10%
   —— 不收敛到 0，存在 ~5% 轴偏置底（staircase 效应：中心差分在 0/1 场上多数界面格只跨一个轴）。
2. **抹平宽度梯（圆，smeared）**：dir_err = 1.75% → 1.18% → 0.71%
   —— 界面被分辨（w≥2 格）后误差 <2%，w=8 时 0.71%（2.8°）。**方法在分辨界面下是准的。**
3. **3D 球分辨率梯**：dir_err = 10.24% → 8.55% → 8.40%
   —— 3D 底 ~8.4%（21.4°），比 2D 圆高（三维 staircase 更粗），同样不随 N 收敛到 0。
4. **GAS ε 取值敏感性**：ε_gas 0→0.5 只把幅值减半（0.586→0.293），方向不变（A1 vs A2 仍 0.0）——符号判据对幅值不敏感，但幅值依赖该约定。

## ⑥ 失败模式与工件

**失败模式（如实）**：
- 锐 0/1 fill 场上，fill-gradient 法向有固有轴偏置（圆 ~5%、平面 ~8%、球 ~8-10% dir_err）——两实现共有，非 TensorLBM 缺陷；生产 Körner 模型的 fill=mass/rho 是连续量，实际误差介于锐/抹平两端之间。
- OSS 的 GAS=0.5 优先级 bug 使幅值低估 2×（本实验复现：mag_ratio=0.500）——**不可照抄**，与知识库记载一致。
- 本实验脚本自身踩坑（已修）：numpy 2.4.6 的 `np.mgrid` 不可调用（MGridClass not callable）→ 改 meshgrid；`meshgrid(indexing='xy')` 返回 (X,Y) 顺序写反导致解析法向转置（一度报 51° 假误差，经单格核对发现）。

**本仓 vs OSS 差异清单（做法/参数/边界条件）与移植取舍**：

| # | 维度 | TensorLBM | OSS (Maarten-vd-Sande/lbm) | 移植取舍 |
|---|---|---|---|---|
| 1 | 归一化 | 归一化（单位法向） | 不归一化（只用符号） | 保留 TensorLBM 归一化：质量再分配等幅值相关用途需要；OSS 的符号用途归一化后同样满足 |
| 2 | fill 定义 | fill=mass/rho（独立质量账本，连续） | ε 离散 {0, 0.5, 1} | 保留 TensorLBM 连续 fill：梯度更平滑；OSS 离散 ε 的 staircase 偏置更大 |
| 3 | 维度 | 3D（D3Q19） | 2D（D2Q9） | 模板同构（中心差分），3D 为 2D 直接加 z 轴，无额外近似 |
| 4 | 符号约定 | liquid→gas（-∇fill） | liquid→gas（0.5*(ε_{i-1}-ε_{i+1})） | 一致，无需转换（已验证 +x 界面 n=(+1,0)） |
| 5 | 边界 | 周期 roll（生产中对 near 界面格掩码） | 周期 roll（全格） | 一致；TensorLBM 掩码只影响作用域不影响模板 |
| 6 | 下游用途 | 质量再分配（Körner） | n·e_opp>0 判据 + feq+feq_opp-f_opp 重构 | 若移植 OSS 重构式，须保持 liquid→gas 符号；TensorLBM 现有 average-surround 重构（oss id=84）已单独 XVAL，不混用 |
| 7 | 已知 bug | 无 | streamcollide.py:166 GAS→0.5 | 移植时显式取 GAS=0（A2 臂），A3 臂仅作 bug 复现 |

**工件**：`xval/fill_gradient_normal/`：`predeclared.json`（跑前预声明）、`run_xval.py`、`make_figures.py`、`results.json`、`figures/fig1_circle_normals.png`、`figures/fig2_convergence.png`、`figures/fig3_sphere_normals.png`。

**生态摩擦（如实记）**：
- `knowledge_base`/`oss_solutions`/`negative_knowledge_ledger` 在 `/api/v1` 下无端点（404），开工三查走 Postgres 直查（`runuser -u postgres psql -d tensorplatform`）——与 m17112/mcmp_laplace 同摩擦，建议给智能体文档化只读接口。
- 无网络 checkout Maarten-vd-Sande/lbm 源码 → OSS 臂为 oss_solutions id=85 公式移植（零手抄：公式逐字取自 method 字段），非独立代码运行。

## 误差判定（逐项）

| 关键量 | 基准 | got | 误差 | <15%? |
|---|---|---|---|---|
| circle-sharp N=256 A1 vs A2 方向 | OSS 法向 | 0.0 | 0.00% | ✅ |
| plane-sharp theta=30.0 N=256 A1 vs A2 方向 | OSS 法向 | 0.0 | 0.00% | ✅ |
| sphere-sharp N=128 A1 vs A2 方向 | OSS 法向 | 0.0 | 0.00% | ✅ |
| circle-sharp N=256 A1 vs 解析 | 径向解析法向 | 5.10% | 5.10% | ✅ |
| plane-sharp A1 vs 解析 | 常向量解析法向 | 7.83% | 7.83% | ❌（锐界面轴偏置，方法固有） |
| sphere-sharp N=128 A1 vs 解析 | 径向解析法向 | 8.40% | 8.40% | ❌（锐界面轴偏置，方法固有） |
| circle-smeared w=8 A1 vs 解析 | 径向解析法向 | 0.71% | 0.71% | ✅ |

**显式回答**：对标判据（TensorLBM vs OSS，owner 15% 线）= **0.00% < 15% ✅ 通过**。vs 解析解的锐界面误差 5-10% 为方法固有轴偏置（两实现共有），抹平界面下 <2% ✅。
