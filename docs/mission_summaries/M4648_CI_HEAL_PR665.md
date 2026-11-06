# m4648 / PR #671 复开：3-D WSS 回归测试的**解析量级**校验（round 1，m4830）

## 1 背景与红因定位

PR #671（分支 `agent/m4648`）曾修 PR #665 的 CI 复红，核心是两处**真测试缺陷**
（`pytest.raises(ImportError)` 要求历史 bug 常驻；`pytest.approx(0-d tensor)` 掩盖真数字）。
owner 2026-11-05 裁定关闭该 PR，本轮从**当前 main** 重建、单一功能点重提。

重建前先对表，当前 main（`22da640f`，Merge #709）与 #671 的差距已不是原来那份：

| #671 内容 | 当前 main 状态 | 本轮动作 |
|---|---|---|
| `tests/test_control_volume_force.py` `pytest.approx(float(bfl_force[0]))` | **已在 main**（m3379 heal 顺带落地） | 无改动，回归验证 |
| `tests/test_regress_wall_multi_gpu.py` 两例去 `pytest.raises(ImportError)`，改正向回归 | **已在 main**，但只断言「可调用 / 有限 / 全零 / 非零」 | **补解析量级断言**（本轮唯一代码改动） |
| `docs/mission_summaries/M4648_CI_HEAL_PR665.md` | 缺失 | 新增本文 |

即：main 上的两例只能证明"函数不再崩、f_neq 路径没死"，**不能证明算得对**——权重、
预因子、应力分量选错都照样通过。本轮把「对不对」钉成解析等式。

## 2 方法

被测实现 `src/tensorlbm/boundary/wall_shear.py::wss_from_fneq_3d`（D3Q19）：

```
prefactor = -(1 - 1/(2 tau))
sigma_ij  = prefactor * sum_q f_neq_q c_i c_j
wss       = sqrt(sigma_xy^2 + sigma_xz^2 + sigma_yz^2) * (1 - mask)
```

构造**矩精确已知**的状态：`f = f_eq(rho=1, u=(u0,0,0)) + a*c_x*c_y`。
均匀流下平衡态无 `c_x c_y` 矩（对 `c_y` 奇函数成对抵消），故 `f_neq = a*c_x*c_y` 是唯一
非平衡成分，于是

```
sigma_xy = -(1 - 1/(2 tau)) * a * sum_q (c_x c_y)^2 ,  sigma_xz = sigma_yz = 0
wss      = |sigma_xy|   （均匀，逐格相同）
```

`sum_q (c_x c_y)^2` 由 `d3q19.C` **程序化求和**（实测 = 4），无手抄常数。
两例分别用 `a=1e-4, tau=0.8`（带固体层 mask 的用例）与 `a=1e-3, tau=0.8`（无 mask 用例）。

## 3 计算结果（程序化，无手抄）

| 用例 | a | tau | 解析 —(1−1/2τ)·a·Σ(c_x c_y)² | 实测 wss | 相对误差 |
|---|---|---|---|---|---|
| mask 用例（整套流体格） | 1e-4 | 0.8 | 1.5000000000e-04 | 1.5000114217e-04 | 7.61e-06 |
| 无 mask 用例 | 1e-3 | 0.8 | 1.5000000000e-03 | 1.4999988489e-03 | 7.67e-07 |

计算精度 float32（张量默认 dtype），误差即 float32 舍入；容差取 `rel=1e-4`（mask 用例）
与 `rel=1e-5`（无 mask 用例），均为舍入预算的 10 倍以上余量。

**负控**（证明断言有鉴别力，非恒真）：

| 负控 | 期望表现 | 实测 |
|---|---|---|
| 预因子漏掉 `1/(2τ)`（1.0 代替 0.375） | 断言必须失败 | 偏 **62.5%**，远超容差 |
| `f_neq` 全零（只喂平衡态） | wss 恒 0 | `count_nonzero == 0`，与 `sigma=0` 一致 |

## 4 注意点

- 不要把 `macroscopic3d(f)` 反算出的 `rho/ux` 喂回解析用例：float32 下矩反算的 1e-7 级
  误差经 `f_eq` 放大到 `f_neq` 上，会把 a=1e-4 的用例打到 ~0.1% 误差，`rel=1e-5` 必红。
  解析块因此显式给定期望矩（`rho=1, u=(0.1,0,0)`）。
- 被测函数返回**幅值**，`sigma_xy` 的符号信息已丢；`f_neq = a·c_x·c_z` 这类换分量的扰动
  在幅值上不可区分（负控实测同值），故不能拿"换分量"当负控，须用预因子/权重级负控。
- 容差不是越紧越好：过紧会在 loong64/不同 BLAS 上假红，过松则退化成原来的"只查有限"。

## 5 误差判定

| 关键量 | 基准 | 误差 | 是否 <3% |
|---|---|---|---|
| wss（a=1e-4，mask 用例） | 解析 `|σ_xy|` | 7.61e-06 | **是** |
| wss（a=1e-3，无 mask 用例） | 解析 `|σ_xy|` | 7.67e-07 | **是** |
| `tests/test_control_volume_force.py` 嵌套 CV 冲量不变性 | tight vs wide CV | ≤3.6e-14（历史记录） | **是** |

无残留 red；本文件不修改任何求解器算法，只补测试与文档。