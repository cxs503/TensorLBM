# XVAL-LBM-BITFLAG-NEIGHBOUR-MASK — 与 Maarten-vd-Sande/lbm 的 bitflag 做法交叉对标（第 1 轮）

仓库：TensorLBM｜分支：agent/m17502｜日期：2026-11-13

## ① 问题 + ref 出处

对标对象：`Maarten-vd-Sande/lbm`（Python/NumPy LBM，含自由表面），只读缓存于
`/root/oss-cache/Maarten-vd-Sande_lbm`，commit `573a01c`。**关键做法逐条从源码查得（零手抄）**：

| 做法 | OSS 出处（file:line） |
|---|---|
| 单元分类用位标志 | `lbm/util.py:31-48` — `FLUID=1, INTERFACE=2, GAS=4, OBSTACLE=8, INLET=16, OUTLET=32`；聚合邻域位 `NO_FLUID_NEIGH=64, NO_EMPTY_NEIGH=128, NO_IFACE_NEIGH=256` |
| 邻域位每步重建、下步先清 | `lbm/update_types.py:118-132`（8 方向扫描）/ `lbm/update_types.py:29-30`（清零） |
| streaming 为 pull 扫描：每开放格 ×8 方向查**邻居** `cell_type` 是否属 `FLUID|INTERFACE|INLET|OUTLET` | `lbm/streamcollide.py:56`、`:63` |
| 邻居关闭 → 半程反弹（half-way BB）`fin[n]=fout[v_inv[n]]` | `lbm/streamcollide.py:88-90` |
| 入口 Zou-He 型密度闭合（速度给定） | `lbm/streamcollide.py:218-233` |
| 出口零梯度（只拷入流分量） | `lbm/streamcollide.py:210-217` |
| 步序 collide → stream → update_types | `lbm/time_evolve.py:11-27` |
| 速度钳位 `maxvel=sqrt(2/3)` | `lbm/streamcollide.py:131-133` |

因此 OSS 的「bitflag + 邻域」是**按相位聚合**的邻域信息，且**只服务自由表面质量记账**；
streaming 循环里仍是**每开放格 8 次邻居类型判定**。本单要回答：把邻域信息提升到**逐方向**位掩码
（D2Q9 8 bit/节点，`uint8`）后，(a) 与参考实现语义是否等价？(b) 判定次数能否下降？(c) 物理结果是否不变？

判据（先于计算声明）：掩码核与 OSS 式扫描核**逐位一致**（max |Δ| = 0）；物理锚点误差 <15%（owner 标准，
本次预声明更严：解析剖面 <3%）；判定次数比 >2（几何相关，见下）。

## ② 方法与实现

新增：
- `src/tensorlbm/bitflag_neighbour_mask.py` — 位标志单元分类（与 OSS 同位值）、
  `neighbour_mask()`（逐方向关闭链路掩码：bit `i-1` 置位 ⇔ `x - c_i` 关闭或出域，即 OSS 走反弹的那一支）、
  4 个 streaming 核：`stream_scan_{scalar,vector}`（OSS 忠实转写）与 `stream_mask_{scalar,vector}`
  （掩码驱动，内部节点 `mask == 0` 走无分支直线路径）、`StreamStats` 工作量记账、
  `oss_border_guard_wraps()`（复刻 OSS 边界守卫缺陷）。
- `src/tensorlbm/bitflag_d2q9.py` — 轻量 NumPy D2Q9 求解器（(9,nx,ny)、OSS 速度序），
  二阶平衡态 + BGK + Guo 二阶力项、半程反弹（含运动壁动量项）、入口 Zou–He、出口可选闭合、
  动量交换法测力，边界核可切换 `mode="mask"|"scan"`。
- `tests/test_bitflag_neighbour_mask.py`、`benchmarks/xval_bitflag_neighbour_mask/run_xval.py`。

关键推导（自出，不抄）：入口速度闭合
`rho = (Σ_{0..5} f + f3+f4+f5)/(1-u_x)`，`f6 = f3 + (2/3)ρu_x`、`f7 = f5 + (1/6)ρu_x`、`f8 = f4 + (1/6)ρu_x`；
出口压力闭合 `ρ u_x = f0+f1+f2 + 2(f6+f7+f8) − ρ`、`u_y = 3(f2−f1)/(2ρ)`，
`f3 = f6 − (2/3)ρu_x`、`f4 = f8 − (1/6)ρ(u_x+u_y)`、`f5 = f7 − (1/6)ρ(u_x−u_y)`。
（第 2 版：首版把 x 动量式用于压力出口，得到恒等式 → 发散，改用密度式后稳定。）

## ③ 参数表（无量纲）

| 用例 | 网格 | ν_lb | τ | U_max/U_mean | Re | 步数 |
|---|---|---|---|---|---|---|
| 算子等价（随机流场 + 圆柱通道） | 40×24 | — | — | — | — | 1 扫描 |
| 平面 Poiseuille（入口驱动） | 80×22 | 0.1 | 0.8 | U_max=0.02，Re_H=u H/ν=1.68 | 1.68 | 30000（已跑） |
| Couette（运动壁 + 线性入口） | 40×22 | 0.1 | 0.8 | U_w=0.02 | U_w H/ν=4.2 | 12000（已跑） |
| Schäfer–Turek 2D-1 圆柱 | D=12/16/20 | u_mean D/20 | 0.5+3ν | u_mean=0.05（U_max=0.075） | 20 | 已递交集群 cj1569/cj1570（排队 404，未回） |

## ④ 对照表（ref vs got，打勾/叉）

算子层（`runs/xval_bitflag_neighbour_mask/unit.jsonl`，程序化产出）：

| 量 | 参考 | 结果 | 判据 | 判定 |
|---|---|---|---|---|
| 掩码向量核 vs OSS 式扫描向量核（随机流场，40×24 圆柱通道） | 逐位一致 | max\|Δ\|=0（开放格） | =0 | ✅ |
| 掩码标量核 vs 扫描标量核 | 逐位一致 | 一致 | =0 | ✅ |
| 向量核 vs 标量核（掩码） | 逐位一致 | 一致 | =0 | ✅ |
| 运动壁（`6 w_i ρ c_i·u_w` 项）掩码 vs 扫描 | 逐位一致 | 一致 | =0 | ✅ |
| 掩码语义 vs OSS 反弹分支条件（逐格 8 方向） | 相等 | 相等 | =0 | ✅ |
| Poiseuille 解析剖面（mask/scan） | 抛物线 `u=g(y-½)(H-(y-½))/2ν` 形状 | 剖面最大相对误差 17.4%（u_max 实测 0.01895 vs 0.02），出口通量漂移 −17.6%，ρ 升至 2.74 | <3% | ❌ honest-FAIL |
| Couette 解析直线 | `u=U_w(y-½)/H` | 最大相对误差 93%（u_max 0.01206 vs 0.01857） | <3% | ❌ honest-FAIL |
| 圆柱 Cd vs Schäfer–Turek 2D-1（REFERENCE['cd']=5.57953523384，in-repo 锚 `benchmarks/verified/cylinder/re20_st/result.json`） | 5.5795 | 未得到（集群排队未回；本地 D=12 未在时限内收敛） | <15% | ⏳ 未完成 |

工作量（同几何 40×24 圆柱通道，单次扫描）：

| 指标 | scan（OSS 式） | mask | 比 |
|---|---|---|---|
| 邻居类型判定 | 6624（=8×828 开放格） | 0 | — |
| 掩码字判定 / 位判定 | 0 / 0 | 828 / 1248 | — |
| 总判定数 | 6624 | 2076 | **3.19×**（随内点数增加而升高；D=16 圆柱几何按解析计数为 179k vs 30k ≈ 5.9×） |
| 反弹/流入链路数 | 444 / 6180 | 444 / 6180 | 完全一致（等价性证据） |
| 标量扫描墙钟（Python，5 次均值） | 0.0102 s | 0.0086 s | 1.19×（解释器主导，仅作参考） |
| 向量扫描墙钟 | 0.00027 s | 0.00034 s | 0.79×（numpy 下掩码多一次 uint8 读，**无加速**） |

## ⑤ 收敛性/敏感性（至少两点）

1. **步数敏感性（Poiseuille）**：4000 步时 u_max/u_ref=0.92，20000 步 0.95，30000 步 0.947 且出口通量仍在
   增长（−17.6%）。通量沿 x 单调增大 ⇒ 稳态未达，且出口闭合本身引入质量不守恒（见 ⑥）。
2. **出口闭合敏感性**：`zero_gradient_all`（整列镜像）ρ 涨到 2.74（质量泵入）；`zero_gradient`（仅入流分量拷贝）
   与 `zou_he`（压力闭合）通量漂移降到 ±1.2%~−17.6% 区间但都不满足 <0.5% 预声明门；`outflow`（丢弃出域分量）
   直接发散（保留在代码中并注明不可用）。
3. **掩码规模敏感性**（几何解析计数）：关闭链路占比 D=12:5.4%、D=16:4.3%、D=20:3.6%，内点占比 90%→94%，
   故掩码判定比随网格升高（3.2×→5.9×）。

## ⑥ 失败模式与工件 + 生态摩擦

失败模式：
- **入口/出口闭合未做对**是物理用例失败的唯一根因，与掩码无关：掩码核与扫描核在整场仿真中逐位一致
  （`mask_vs_scan_bit_identical=true`），因此物理偏差完全继承自闭合选择。
- 出口 `zero_gradient_all` 造成质量泵入（ρ→2.74）；`outflow`（丢弃出域分量）发散；`zou_he` 收敛但通量漂移 1.2%~17.6%。
- 环境坑：本机 `tensorlbm` 是 editable 安装且指向**另一个 worktree**（`/root/tp-agent/work/m17514/src`），
  pytest 里即使 `sys.path.insert` 前置仍被 conftest 先导入的旧模块劫持（新函数签名冲突），
  故 `tests/test_bitflag_neighbour_mask.py::test_mask_kernels_match_scan_kernels` 在本环境 FAIL
  （`TypeError: stream_mask_scalar() missing 1 required positional argument: 'mask'`，来自 m17514 的同名不同签名函数）；
  等价性本身已由 `run_xval.py unit`（显式插 `src` 前置）证明为 true。

与 OSS 的差异清单 + 移植取舍：

| 维度 | OSS | 本仓（本单） | 取舍理由 |
|---|---|---|---|
| 邻域信息 | 聚合 3 位（按相位：NO_FLUID/NO_EMPTY/NO_IFACE），每步重建 | 逐方向 8 位 `uint8`，仅在几何变化时重建 | 逐方向可让 streaming 无需邻居类型查询；静态几何省掉每步重建 |
| 消费位置 | 只用于自由表面质量记账（`interface_mass_exchange`） | 直接驱动 streaming / 反弹分支选择 | 把该位图用在真正的热点上 |
| streaming 判定 | 8 次邻居 `cell_type` 位测试/开放格 | 1 次掩码字测试 + 关闭链路位测试 | 内点（>90%）零方向判定 |
| 边界守卫 | `0 < ix_next >= nx`（链式比较，负索引不拦，回绕到对侧边界） | 显式 padding，出域链路一律记为关闭 | 参考实现的实测缺陷，未移植（`oss_border_guard_wraps` 复刻并测试） |
| 自由表面 | INTERFACE/GAS + 法向修正 + 质量重分布 | **未移植** | 本单目标是边界链路选择，自由表面属独立能力面 |
| 入口闭合 | 速度给定 + `+x` 分量置平衡态（`streamcollide.py:31-33`） | Zou-He 非平衡反弹闭合（自推导并验证） | 二阶精度更可控 |
| 出口闭合 | 零梯度（仅入流分量） | Zou-He 压力 / 零梯度 / 全列镜像 三档可选 | 便于做闭合敏感性 |
| 力项 | 把重力直接减到 `fout`（一阶） | Guo 二阶力项 + 半力速度修正 | 解析剖面一致性 |
| 速度钳位 | `|u| <= sqrt(2/3)` | 无 | 钳位会污染误差统计 |
| 数据类型 | float32 | float64 | 逐位一致性判定需要双精度 |

工件（worktree 内）：
- `src/tensorlbm/bitflag_neighbour_mask.py`、`src/tensorlbm/bitflag_d2q9.py`
- `benchmarks/xval_bitflag_neighbour_mask/run_xval.py`（子命令 unit/poiseuille/couette/sweep/cylinder/aggregate）
- `runs/xval_bitflag_neighbour_mask/unit.jsonl`、`poiseuille_*.jsonl`、`couette.jsonl`、`logs/`
- `tests/test_bitflag_neighbour_mask.py`

结论：**(a) 等价性成立且逐位可证**（掩码 = OSS 式扫描，含运动壁）；**(b) 判定次数 3.2×（解析计数最高 5.9×）**；
**(c) 物理精度本次未达标**（Poiseuille 17.4%、Couette 93%，根因是出口闭合而非掩码），
圆柱 Cd 因集群排队未回而未得，按 honest-FAIL 记账，两轮法接棒应优先修出口闭合（建议：压力出口 +
通量修正，或改用周期性驱动 + 体力，回避入/出流）。
