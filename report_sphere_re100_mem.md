# sphere_re100_mem 平台报告：球 Re=100 绕流阻力 BFL 大域基准

## 1. 物理问题表述

均匀来流绕过直径为 D 的球体，雷诺数 Re = U·D/ν = 100。该工况低于涡脱落的下临界
Re，尾流为定常轴对称结构。核心观测量为阻力系数 Cd = Fx / (½ρU²·πR²)。

**参考标准**：Schiller–Naumann 经验公式 Cd = 24/Re·(1 + 0.15·Re^0.687)，在 Re=100
时给出 Cd_ref = 1.0917310910948732。交叉验证：S-N(0.681) = 1.06852，
Clift–Gauvin = 1.10923。

## 2. 数值方法

- **求解器**：D3Q19 格子 Boltzmann，MRT 碰撞算子（`collide_mrt3d_low_memory`）
- **边界表示**：Bouzidi–Firdaouss–Lallemand（BFL）插值 bounce-back，亚格距 q 由
  `compute_q_sphere` 给出，壁面为光滑插值而非阶梯体素化
- **力法口径**：BFL 逐链动量账本（`bfl_force_ledger_sparse`，实验室系），即
  F = Σ_links Δ(动量)，仅对壁面相邻链求和
- **远场条件**：hard far field，每 200 步质量校正
- **库侧解锁**：PR #316 稀疏边界链核 + `general_sim` 低显存初始化，使大域 D60
  在 32G GPU 上可跑（峰值 26.67 GiB；同域密集核第 1 步即 OOM）

## 3. 判定结果

### 3.1 四门标准

| 门 | 要求 | 实际 | 结果 |
|---|---|---|---|
| 共性模块入口 | src/tensorlbm 库路径 | D3Q19 + collide_mrt3d_low_memory + BFL | ✅ |
| 直接观测量 ≤3% | |err| ≤ 3% | T1 +2.7708%, T2 +2.5401% | ✅ |
| ≥2 档单调收敛 | |err| 严格单调降 | 2.7708% → 2.5401% | ✅ |
| 参考先锁定后跑 | 预注册冻结 | result.json 先锁，W8-A 后跑 | ✅ |

### 3.2 两档数值结果

| 档 | D | 域 (nz×ny×nx) | BFL 链数 | 步数 | Cd（末 20% 窗） | err | 稳态漂移 |
|---|---|---|---|---|---|---|---|
| T1 | 40 | 280×280×320 | 28 722 | 12 000 | 1.121980 | +2.7708% | −0.145% |
| T2 | 60 | 420×420×480 | 64 794 | 24 000 | 1.119462 | +2.5401% | −0.042% |

误差跨度 |2.7708 − 2.5401| = 0.2307% ≤ 3%，严格单调收敛。

### 3.3 位一致链与独立 CV 对账

- **稀疏核 ≡ 密集核**：T1 稀疏核对 W7 密集核 240 采样 × (cd/cl/cs/mass) 逐位相同
- **驱动链位等**：W7 dense 驱动 == W8-A dense 驱动（D20，6 步 × 4 量逐位相同）
- **CV 动量平衡**：壁外 6 格盒独立 CV 仪表与逐链账本窗均一致至 0.008–0.015%

## 4. 域形收敛签名（入口距离主导）

W7 域形诊断矩阵显示误差由入口平面 Dirichlet 钳制主导：

| up/D | Cd | err |
|---|---|---|
| 1.25 | 1.2149 | +11.28% |
| 2.00 | 1.1553 | +5.83% |
| 2.75 | 1.1380 | +4.23% |
| 3.00 | 1.1347 | +3.94% |
| 4.00 | 1.1279 | +3.31% |
| 大域 lat3/up3/down4 | 1.1220 | +2.77% |

五点最小二乘拟合 err ≈ E₀ + C·(a/x)³（R² = 0.9998），验证入口 Dirichlet 钳制
是主因。大域（lat 2→3, up 1.25→3, down 2.25→4）联合将误差压进 3% 门。

## 5. 审计溯源

所有判决数字从以下工件直读：
- `verified/sphere_re100/w8a/result.json` — 完整结果 + judgment (pass=true)
- `verified/sphere_re100/w8a/t1_d40big_12k.json` — T1 原始数据
- `verified/sphere_re100/w8a/t2_d60big_24k.json` — T2 原始数据
- `verified/sphere_re100/w8a/bit_check.json` — 位一致链
- `verified/sphere_re100/w8a/mem_sparse_D60big.json` — 稀疏核显存证据
- `verified/sphere_re100/w8a/verify.py` — 独立验证脚本

## 6. 局限与缓解

- **仅两档网格**：受 GPU 32G 显存约束（D60 已用 26.7 GiB）；D80 需 64G 卡或多卡分域
- **参考簇散布**：S-N 主锁位于交叉族低端（T1 对 Clift-Gauvin 差约 +1.1pp），
  锁定参考不重开
- **阶梯球负结果**：surface-only Ladd MEM 在阶梯球面上收敛到 +9%（非口径 bug，
  是双曲率阶梯几何偏差），已在 verified/sphere_re100/ 中披露

## 7. 归档决策

本案与 `sphere_re100`（pending）为同一物理问题的不同力法路径。`sphere_re100`
的全部测力路径因阶梯几何偏差 >3%，而本案的 BFL 插值边界 + 大域 + 稀疏核路线
已满足四门标准。`sphere_re100` 同步标记为 superseded，指向 `verified/sphere_re100_mem/`。

<!-- PROVENANCE
[{"v": 1.0917310910948732, "f": "verified/sphere_re100/w8a/result.json", "k": "reference.cd_ref"},
 {"v": 2.7707656824703957, "f": "verified/sphere_re100/w8a/result.json", "k": "runs[0].err_pct"},
 {"v": 2.5400861325477218, "f": "verified/sphere_re100/w8a/result.json", "k": "runs[1].err_pct"},
 {"v": true, "f": "verified/sphere_re100/w8a/result.json", "k": "judgment.pass"}
]
-->
