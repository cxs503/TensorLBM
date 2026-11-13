# XVAL · LBM 稀疏子表邻接表（sparse-sublist neighbour-list）— TensorLBM vs sorush-khajepor/listLBM

**锚点** XVAL-LBM-SPARSE-SUBLIST-NEIGHBOR-LIST · mission 18377 · 分支 `agent/m18377`
**对标对象** `sorush-khajepor/listLBM` @ `master`（本地克隆 `/tmp/listlbm`，GPLv3）——全部结论带 `文件:行`，源码取自本地克隆，未执行 OSS 二进制（Fortran，无编译器）。

## 1. 问题与参考出处

多孔介质 LBM 在**每个**固体格点上浪费 9 个分布函数。`sorush-khajepor/listLBM`（博士课题，多孔介质多相流）把未知量压缩到**流体节点**上，并存储一张**邻接表**（neighbour list）做间接寻址：

| OSS 位置 | 机制 |
|---|---|
| `lists/base/mainlist_mod.f90:14-20` | `pdf(0:nq_1, ist:ind)` 只存流体节点；`neilist(1:nq_1, ist:ind)` 存每个方向邻居的**压缩索引** |
| `lists/base/sublist_mod.f90` | 子表 = 主表连续子区间 `[istinmlist, indinmlist]`，`getimcell(iscell)=istinmlist+iscell-ist` 映射回主表 |
| `stream/stream_mod.f90:44-73` | **交换**流：`stream1` 把 `pdf(dir,icell)` 与 `pdf(opposite(dir),neilist(dir,icell))` 交换；`stream2` 把节点内方向标签交换 |
| `lattice/lattice_mod.f90:87` | D2Q9 `halfqs=(1,2,5,6)`（每对相反方向取一个） |

实现：`TensorLBM/src/tensorlbm/sparse_sublist_neighbor_list.py`（`SparseSublistNeighborList`、`listlbm_stream1`、`listlbm_stream2`、`listlbm_step`），测试：`TensorLBM/tests/test_sparse_sublist_neighbor_list.py`（10 个测试，全绿）。

## 2. 方法与实现

* **A1 稀疏子表**：`(9, K)` 状态（K=流体节点数）；`stream1`+`stream2` 交换流；`-1` 邻居（固体）跳过（OSS `blank_streamwall` 同义）。
* **A2 稠密控制**：`(9, ny, nx)` 状态；`FluidNodeTable` push 流；两臂共用 `collide_with_guo_forcing`，碰撞与 Guo 力不可能成为分歧源。
* **A3 解析锚**：平面通道 H 行流体，半程壁在首/末流体行外 0.5 dx → 固有 `k=H^2/12`，表观 `k=(H/ny)H^2/12`。

周期包裹双向，D2Q9 BGK（`tau=0.8`，`nu=(tau-0.5)/3`），均匀体力（Guo），float64，单 CPU 线程。

## 3. 参数表

| 符号 | 值 |
|---|---|
| 格子 | D2Q9 |
| 弛豫时间 | tau = 0.8 |
| 运动粘度 | nu = (0.8-0.5)/3 = 0.1 |
| 体力 | g = 1e-5（格子单位） |
| 通道 | H=16, ny=32, nx=16 |
| 步数 | 2500 |
| 精度 | float64 |
| 线程 | 1 |

## 4. 对照表（ref vs got）

| 量 | 参考值 | 本仓值 | 误差 | 判定 |
|---|---|---|---|---|
| 表观渗透率 k_super | 10.6667 | 10.6543 | 0.12% | ✅ <15% |
| 表等价（nodes/neighbour/index） | — | 0.0（bit 相同） | 0.0 | ✅ |
| 算子等价（swap vs push） | — | <1e-14 | — | ✅ |

## 5. 收敛性 / 参数敏感性

* **算子等价（3 种子）**：随机多孔几何（种子 1/5/13）上，`listlbm_stream`（swap 对）与 `FluidNodeTable` push 核最大绝对差 <1e-14——表示中性成立，与几何无关。
* **质量守恒**：200 步无外力，质量漂移 <1e-12——swap 对不引入数值泄漏。
* **渗透率**：平面通道 H=16 上，表观渗透率误差 0.12%（远低于 15% 门限）。

## 6. 失败模式与工件

* **失败模式**：`ux.mean()` 在压缩布局上给出**固有**速度（只对流体节点平均），需乘孔隙率得到**表观**速度——这是压缩布局的固有陷阱，已在测试中显式记录。
* **工件**：`TensorLBM/src/tensorlbm/sparse_sublist_neighbor_list.py`（模块）、`TensorLBM/tests/test_sparse_sublist_neighbor_list.py`（10 测试）、`TensorLBM/artifacts/xval_sparse_sublist_neighbor_list/figures/fig1_channel_profile.png`（通道剖面图）、`xval/XVAL-LBM-SPARSE-SUBLIST-NEIGHBOR-LIST/predeclared.json`（预声明）。

## 7. 本仓 vs OSS 差异清单

| 方面 | listLBM (OSS) | TensorLBM | 移植取舍 |
|---|---|---|---|
| 流约定 | **交换**（stream1+stream2，push-pull 合并） | **push**（scatter，`f_new[d][nb]=f[d][node]`） | 两者等价（本 XVAL 证明）；保留 OSS 交换语义 |
| 子表 | 按碰撞类型分组（fluid/wall/ghost 多子表） | 单一扁平表（全部流体） | 单相场景下多子表无收益；保留单子表 |
| 幽灵单元 | 完整 ghost-cell 系统（周期 BC） | 建表时周期包裹 | 建表时包裹更简单，语义等价 |
| 壁处理 | `walllist_t` `blank_streamwall`（不流）+ `wallcollision_t`（固定 rho_w） | 半程反弹（`-1` 哨兵 → `f_new[opp(d)][k]=f[d][k]`） | 半程反弹是标准做法；OSS 的固定 rho_w 是特例 |
| 格子 | D3Q19（默认） | D2Q9（本 XVAL） | 概念相同；D2Q9 便于与既有 `FluidNodeTable` 对照 |
| 语言 | Fortran | Python/torch | 语义移植，非二进制移植 |
