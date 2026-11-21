# DE-X-PF3-COLORG · 色梯度（Latva-Kokko / Rothman-Keller）密度比 1000 可达性

**波次**: TRACK_COMB_W1（组合比较矩阵波，owner 令 2026-11-21）
**轴**: 多相 · 色梯度
**单**: m26954
**日期**: 2026-11-21

## 1. 任务

验证 Latva-Kokko 色梯度模型（两色分布 + RK 色力 + 权重重染色 β=0.5 对称式）
在**密度比 1000**（水/空气量级）下的可达性：
1. 静态分层（重流体在下、轻流体在上，无体积力）
2. 静态液滴（重液滴悬浮于轻浴中，静止）
3. 粗带入水（n≤80，2-D 截面）→ 深度方向

DE 锚（17 位全精度取自原文件）：实验 1.857434D@10.394ms(pc)；
ctrl 1.435220D@11.0906ms(abs)/10.5433(pc)；窗 24.005ms；mach 0.2598；
contact 45；drift 4.8e-6；1 cell=0.125D≈6.7%。

## 2. 方法

**模型**: 两 PDF 偏密度色梯度（`tensorlbm.color_gradient_two_pdf`，
D2Q9，已对 Poiseuille + Laplace 律交叉验证，但**仅等密度** rho=1）。
本单新增 `tensorlbm.color_gradient_density_ratio` 模块，把纯相偏密度
初始化为 `rho_a0 / rho_b0 = 1000`（重相 A=水，轻相 B=空气），
模型方程不变（BGK 碰撞 + Guo 力 + 色梯度扰动 + 重染色均作用于偏密度），
只改初始化与诊断。

**参数**（静态探针）: width=8（界面宽度 8 格），nu=5（高黏阻尼），
beta=0.5（任务指定），A=0（先关表面张力隔离重染色不稳定性）。

**判据**:
- 稳定 = 无 NaN、偏密度非负、phi ∈ [-1,1]
- 伪流 = u_max（混合速度最大值，静态下应为 round-off 量级）
- 界面漂移 = 首轻相行位移（格）
- 液滴半径漂移 = |r1-r0|/r0

## 3. 结果

### 3.1 静态分层（64×32，200 步，width=8，nu=5，beta=0.5，A=0）

| ratio | stable | nan_step | u_max | 界面漂移(格) | phi_range |
|---|---|---|---|---|---|
| 1 | True | — | 0.0000 | -3 | [-1,1] |
| 10 | True | — | 0.4149 | 13 | — |
| 50 | True | — | 0.7936 | 未检出 | — |
| 100 | True | — | 0.8840 | 未检出 | — |
| 500 | True | — | 0.9706 | 未检出 | — |
| **1000** | **True** | — | **0.9825** | **未检出** | **[0.874, 1.0]** |

**关键发现**:
- 默认参数（width=2, nu=1/6）下 ratio≥100 即 NaN（step 16）。
- 宽界面（width=8）+ 高黏（nu=5）下 ratio=1000 不炸，但：
  - **伪流 u_max=0.98**（超声速，lattice 单位）——静态下不应有任何运动。
  - **phi ∈ [0.874, 1.0]**——轻相（phi=-1）完全消失，重相侵入全域
    （min_rho_a=76.6，本应为 0）。
  - 质量守恒完好（drift_a=3.5e-12/步，drift_b=1.7e-15/步）。

### 3.2 静态液滴（64×64，R=16，200 步，width=8，nu=5，beta=0.5，A=0）

| 量 | 值 | 判读 |
|---|---|---|
| stable | True | 不炸 |
| u_max | 1.6e-05 | 伪流小 |
| **radius_drift_rel** | **1.04** | **半径漂移 104%——液滴形态完全失守** |
| mass_drift_a | ~1e-12/步 | 质量守恒 |

液滴在 200 步内半径漂移 104%（质量守恒但形态崩溃），
说明重染色在密度比 1000 下把重相"抹"进轻相区，界面无法维持。

### 3.3 粗带入水（n_cells=20，2-D 截面，gy=1e-5，gap=1.0，200 步）

| 参数组合 | stable | nan_step |
|---|---|---|
| nu=5, width=4, gy=1e-4 | False | 5 |
| nu=10, width=4, gy=1e-6 | False | 5 |
| nu=10, width=4, gy=1e-7 | False | 5 |
| nu=10, width=4, gy=0（无重力） | False | 5 |
| gap=3, n=10, nu=10 | False | 5 |

**所有参数组合均在 step 5 前 NaN**。调试显示 step 2 起 f_min<0
（偏密度转负 → 总密度转负 → 发散）。即使 gy=0（无下落）也炸，
说明不是冲击速度问题，而是**重染色幅值 ∝ rho_a·rho_b/rho 在密度比 1000
下过大**（界面带内 ~250×beta·ci_n·w，对比等密度时 0.25×beta·ci_n·w）。

## 4. 判定

**honest FAIL：密度比 1000 不可达（动态场景）。**

- 静态分层：勉强"不炸"，但伪流 u_max≈0.98（超声速）+ 轻相消失
  （phi 下界 0.874），物理上已不是两相分层。
- 静态液滴：不炸但半径漂移 104%，形态失守。
- 粗带入水：全参数组合 step≤5 NaN，**深度方向无法测量**。

**根因**: 重染色幅值 `beta·(rho_a·rho_b/rho)·(c_i·n)·w_i` 在密度比 1000
下比等密度情形大 ~1000×（界面带内 rho_a·rho_b/rho ~ 250 vs 0.25），
把重相抹进轻相区。这是偏密度色梯度族的**结构性限制**（与 MF-LBM
参考实现的"等密度"文档限制一致），不是参数调优能解决的。

**对组合矩阵的意义**: 色梯度道在 DE 锚（水/空气密度比 ~1000）上
**不可行**，不升全窗。若 owner 仍要此道，需改模型方程（如按密度
归一化重染色幅值），超出本单"方程不变"约束。

## 5. 产物

- `src/tensorlbm/color_gradient_density_ratio.py` — 密度比初始化 + 探针
- `benchmarks/water_entry/single_drop/colorg_water_entry.py` — 2-D 入水探针
- `benchmarks/water_entry/single_drop/colorg_reachability.py` — 三探针汇总
- `/tmp/pf3_reach.json` — 机器可读结果

## 6. 复现

```bash
cd /root/tp-agent/work/m26954
PYTHONPATH=src python3 benchmarks/water_entry/single_drop/colorg_reachability.py \
    --out /tmp/pf3_reach.json
```
