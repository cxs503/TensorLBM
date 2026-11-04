# D2Q9 2D 解析解验证（m3621）

## 背景：验证欠账

3D 求解器（d3q19）已有完整 4 步解析解验证体系（`tests/test_lbm_verification.py`，
剪切波/ Couette / Poiseuille / 摩擦阻力，误差 <1%）。但 **2D 求解器（d2q9）**
此前只有 `tests/test_d2q9.py` 的 equilibrium 往返测试，缺少对
「碰撞 + 迁移 + 反弹边界 + 体力」整条物理链的解析基准复核。

本单补齐该欠账：新增 `tests/test_d2q9_analytic_verification.py`，
为 2D D2Q9 求解器提供与 3D 同构的 4 个解析解验证案例。

## 验证案例（每个都有精确解析解）

| Step | 物理 | 验证对象 | 解析解 | 阈值 |
|---|---|---|---|---|
| A | 剪切波衰减 | 粘度 ν | `u_y = A·cos(kx)·exp(-νk²t)` | <1% |
| B | Couette 流 | 反弹 + 无滑移 | `u(y)=U·(y-0.5)/(ny-1-0.5)` | <1% |
| C | Poiseuille 流 | 体力 + 速度剖面 | `u(y)=G/(2ν)·(y-0.5)·(H+0.5-y)` | <1% |
| D | 摩擦阻力 | 壁面剪切应力 | `Cf = 2ν/((H-0.5)·U)` | <1% |

## 结果（CPU，小网格，轻量）

| Step | 数值 | 解析 | 相对误差 | 判定 |
|---|---|---|---|---|
| A | ν=0.166680 | ν=0.166667 | 0.01% | PASS |
| B | max_err | — | 0.00% | PASS |
| C | u_max=0.012416 | u_max=0.012375 | 0.34% | PASS |
| D | Cf=0.6349 | Cf=0.6349 | 0.00% | PASS |

**4/4 全部通过，误差均 <1%** —— 2D D2Q9 求解器基础物理链（碰撞+迁移+反弹+体力）
与解析解一致。

## 运行方式

```bash
PYTHONPATH=src python tests/test_d2q9_analytic_verification.py   # 独立脚本
python -m pytest tests/test_d2q9_analytic_verification.py -q     # pytest 收集
```

## 注意点 / 局限

- 案例 C（Poiseuille）误差 0.34% 略高于其余，源于 Guo 体力在有限网格上的
  离散近似（与 3D 套件同量级），仍远低于 1% 阈值。
- 本验证覆盖 BGK 碰撞 + 周期迁移 + 半路反弹 + Guo 体力；MRT 碰撞、
  Zou/He 进出口等更高阶边界未在此复核（可作后续增量）。
- 网格均为小规模（ny≤12, nx≤80），适合高负载环境下的轻量回归。
