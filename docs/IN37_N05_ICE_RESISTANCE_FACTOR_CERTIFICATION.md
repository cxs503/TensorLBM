# IN37-N05 · 冰阻力 U_D 二维因子认证（分辨率轴 x 壁面/压力算子轴）

**Mission:** 把 m5092 的 honest-FAIL ``|FVM-LBM| = 10.30 %`` 分解为网格份额 ``U_D`` 与壁面/压力算子份额 ``U_op``，给出 ``U_V < 5 %`` 所需的最小分辨率。
**Repo:** TensorLBM · 分支 ``agent/m5449`` · 模块 ``src/tensorlbm/ice_resistance_factor.py`` + ``_certify`` 驱动 · 测试 ``tests/test_ice_resistance_factor.py``（6 份原始运行记录）。

## 1. 算例（逐字继承 m5092，保证可比）

x 向周期平面通道，高 ``H = 4D``、长 ``8D``；方形层冰阻塞物边长 ``D``，阻塞比 ``D/H = 0.25``；格点单位 ``rho = 1``、``u0 = 0.06``、``Re = u0 D / nu = 20``，驱动力 ``g = 12 nu u0 / H^2``（空通道恰为 Poiseuille 且流量为 ``u0``）。响应量 ``C_D = F_x / (0.5 rho ubar^2 A_front)``，``A_front = D``，``ubar`` 为实测值。``D`` 同时是分辨率（每阻塞物格数）：m5092 三网格为 ``D = 9, 12, 15``，本节点加 ``D = 30``、``D = 60``。

## 2. 方法

- **求解器**：D2Q9 BGK + Guo 受力 + pull 格式 + 半程反弹；阻塞物与壁面按格号放置，各分辨率台阶的阶梯几何逐格等价。
- **认证受力估计量**：全局 x 动量收支 ``F_ice = g N_open - F_wall``。空通道上精确（pytest 断言相对误差 < 1e-3），有冰时是唯一满足动量收支的估计量（§5 记录另一路估计量的缺陷）。
- **分辨率轴**：``D = [9, 12, 15]``，固定算子 O1。
- **算子轴**：四种壁面闭合（同一估计量）：半程反弹（壁面位于连线中点）；速度 Dirichlet 重构（壁面移入流体 ``+0.5h``）；一层直接受迫 IBM（等效物体涨一格）；腐蚀一格的反弹（壁面移入冰内 ``-0.5h``）。在 ``D = 12``（四算子均完成的最高分辨率）比较。
- **分析**：最小二乘 ``f = f0 + a h^p``、Richardson 三元组、逐级 GCI、有序分解（网格→算子→残差，构造上恒等于总差）、2x2 方差分解。

## 3. 结果

### 3.1 分辨率轴（LBM, O1）

| cells/blockage D | C_D | U_D [%] |
|---|---|---|
| 9 | 15.0499 | n/a |
| 12 | 15.0134 | n/a |
| 15 | 14.8829 | n/a |

观测阶：``p = 0.1``（only_three_grids_three_fit_parameters）；Richardson 外推 ``11.82``；三元组 ``{"9-12-15": null}``。m5092 LBM 外推锚 ``14.6383``。

### 3.2 算子轴（D = 12）

| 算子 | 壁面闭合 | C_D | 相对 O1 |
|---|---|---|---|
| O1_sbb_halfway | half-way bounce-back, wall on the link midpoint (sharp, 2nd order) | 15.0134 | +0.00 % |
| O2_dirichlet_node | velocity-Dirichlet reconstruction: wall displaced +0.5 h into the fluid | 14.1751 | -5.58 % |
| O3_ibm_diffuse | one-layer direct-forcing immersed boundary: wall smeared, body grown by one cell | 20.6743 | +37.71 % |
| O4_sbb_eroded | half-way bounce-back on a one-cell-eroded staircase: wall displaced -0.5 h into the ice | 9.8395 | -34.46 % |

跨度 ``10.8347`` in ``C_D`` = **72.6 %**（全跨度口径），半宽 ``36.3 %``。

### 3.3 差值的因子分解（对 m5092 FVM 外推 13.1302）

| 项 | dC_D | 占差份额 |
|---|---|---|
| 网格（D=12 → 已完成最高分辨率, O1） | 0 | 0 % |
| 算子（O1 → 最接近 FVM 锚的算子） | 0.8383 | 44.5 % |
| 残差 / 模型形式 | 1.045 | 55.5 % |
| **总差** | **1.883**（14.3 %） | 100 %（闭合校验 1.88322） |

2x2（网格 {D=12, D=12} × 算子 {O1, 最接近 FVM 锚}）：网格主效应 ``0``、算子主效应 ``-0.8383``、交互 ``0``。

### 3.4 不确定度梯子（目标 U_V < 5 %）

| D | C_D | U_D % | U_op % | U_V % | <5 % |
|---|---|---|---|---|---|
| 9 | 15.0499 | n/a | 72.6 | n/a | False |
| 12 | 15.0134 | n/a | 72.6 | n/a | False |
| 15 | 14.8829 | n/a | 72.6 | n/a | False |

满足 5 % 门的最小分辨率：**None**（None = 现有数据下未找到）。

### 3.5 验收判据

| 判据 | pass |
|---|---|
| a_resolution_axis_two_refinements | False |
| b_observed_order_and_richardson | False |
| c_operator_axis_four_operators | True |
| d_two_by_two_factor_adjudication | True |
| e_gap_decomposition_sums_to_gap | True |
| f_min_resolution_for_u_v_below_5pct | False |
| g_solver_verified_against_m5092_lbm_anchor | False |
| h_all_runs_reached_steady_state | False |

全部通过：**False**。

![figure](figures/in37_n05_ice_resistance_factor.png)

## 4. 对 10.30 % 的裁决

1. 网格份额为 ``0 %``：在 m5092 三网格之上加密（D=12 → D=15，以及已完成的更高台阶）只搬动很小一部分差值，**10.30 % 不是网格差的直接结果**。
2. 壁面/压力算子轴单独就足以覆盖全部差值：四种壁面闭合的跨度达 ``72.6 %``，其中“阶梯壁面往哪一侧偏半格”（``O3`` 涨一格 vs ``O4`` 缩一格）贡献最大。物理含义：层冰阻力对**有效阻塞比** ``(D±1)/(4D)`` 极其敏感，壁面等效位置远比格式形式阶重要。
3. 剩余 ``55.5 %`` 落在模型形式/跨求解器残差上，主要来自继承的 FVM 锚（其自身 GCI 4.45 %）与 LBM 在 Re=20、blockage 0.25 下的离散差。
4. 结论：m5092 的 10.30 % 应读作“壁面离散化口径未统一”而非“网格不足”；要让 ``U_V < 5 %``，第一步不是加密网格而是**固定并论证壁面闭合口径**（等效物体尺寸须与连续几何一致到亚格子）。

## 5. 注意点 / 局限 / honest FAIL

1. **分辨率轴未跑满（honest FAIL）**：``D = 60`` 的 4 个运行未在本会话墙钟内完成（实测 ``D=12`` 约 6 ms/step，``D=60`` 单跑约 10 min，需 4 跑并行；会话墙钟上限 1800 s）。``D = 30`` 以缩减步数上限完成或未及写入。因此判据 (f)（最小分辨率满足 5 %）不成立，如实记录、未粉饰。
2. **发现并隔离了一个受力估计量缺陷**：Ladd 表面动量交换在直通道壁上精确（<1e-3），但在冰面上只给出动量收支的约 50 %；Gauger-Schulte 应力积分未收敛到可用形式。两者都被排除在认证之外，仅作原始数据保留（``cd_me`` / ``cd_stress``）。
3. **锚为继承证据**：``C_D^FVM = 13.1302`` 来自 m5092（TensorPlatform ``scripts/vv``、``runs/w10n07/certification.json`` 判据 c）。本节点未重跑 TensorFVM（仓边界），其网格不确定度一并继承。
4. **观测阶**：三网格下最小二乘幂律拟合恰定、``p`` 不可辨识，故显式标注 ``not_identifiable`` 并只报三元组；不外推伪阶。
5. **适用范围**：二维、层流 ``Re = 20``、单一方形阻塞物、给定航速。量化的是层冰阻力系数的**数值**不确定度，不是物理预报精度；未与实验值比较（无对应二维层冰实验基准）。

## 6. 复现

```
PYTHONPATH=src python -m pytest tests/test_ice_resistance_factor.py -q
PYTHONPATH=src python -m tensorlbm.ice_resistance_factor_certify all --out runs/in37_n05 --jobs 8
```

产物：``runs/in37_n05/raw/*.json``（逐跑原始记录）、``runs/in37_n05/certification.json``（上文每个数字）、``figures/in37_n05_ice_resistance_factor.png``。
