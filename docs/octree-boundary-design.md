# 矩形块主体 + 边界八叉树局部加密：混合架构技术设计

- 状态：设计（design-control），未实现任何源码改动
- 关联仓库：`/data/TensorLBM`（PyTorch LBM，Python 3.11，`PYTHONPATH=src`）
- 参考项目：`~/refs/3D-LBM-AMR`（Haider-BA，C++ 八叉树 LBM）、`~/refs/Octree-LBM-solver`（bagalrohit-stack，C++23 八叉树 AMR LBM）
- 背景：`NestedStaticBlockAMR3D`（`static_block_amr.py:672`）矩形块嵌套 AMR 成熟（球体 5.16% 精度），但矩形盒包裹球面存在边角料浪费。目标：主体保持矩形块，物面边界格点用八叉树局部加密（真贴体），消除边角料。

---

## 1. 两个参考项目八叉树实现对比

### 1.1 3D-LBM-AMR（Haider-BA，C++，D3Q19，MRT/BGK + LES）

| 要素 | 实现 | 对本设计的价值 |
|---|---|---|
| 节点结构 | `Node{child[8], Parent*, nb[19], level, image, edge, f/feq/ftemp/delta}`；`child` 指针数组，满八分（细化必生 8 子） | `edge` 标志（刚细化/界面节点需插值）与 `image`（边界类型编码：fluid=0, BB=1–4, CB1=5, CB2=6）是核心概念 |
| 细化 | `Refine()`：offset=`0.25*2^-level`，8 子坐标=父中心±offset，逐子调用 `Geometry()` 判 `image`；CB 节点分配 `delta[19]` | 物面几何在**每个新叶子生成时即时重判**，天然贴体 |
| 邻接 | `ChildNeighbor()`：从父 `nb` 推导 8 子的 19 链接；父邻居未细化（`nb[k]->child[0]==NULL`）→ 子对应链接置 NULL 且 `edge=1` | 邻接在细化/粗化后**重建**而非增量维护；`edge` 驱动插值 |
| 粗细界面 | `Remap()` 加密时强制 m 层逐级降级缓冲带（2:1 平衡），细化区不贴域边界 | 2:1 平衡由**判据层**保证，不在邻居表层做 |
| 时间步进 | `LRLBM()` 递归按层推进：每层 Target 走 2 步（`dt_l=dt_0/2^l`）。序列：SpatialInterp1（粗→细 f 插值）→ March1（Stream1 半个方向集 `{1,2,5,6,9,10,11,12,13}`）→ March2（Stream2+边界+宏量+碰撞）→ March3（CurvedB）→ SpatialInterp2（插到 ftemp）→ TemporalInterp（`ftemp=0.5*(f+ftemp)`）→ 递归细层 → 返回后第二组 March → TemporalInterp2（ftemp→f）→ SpatialAverage（细→粗限制） | 细层每粗步 2 子步、粗层两个半时间点分别作细层两步的边界条件；**时间插值**（TemporalInterp）补齐细层中间时刻 |
| 粗细空间插值 | `SpatialInterp1`：`f_coarse = 0.015625*(27f + 9Σ面邻 + 3Σ边邻 + 1角邻)`（3D 二阶），对 f 与 feq 分别插值；`f_child = feq_c*(1-SF) + SF*f_c`，`SF = ω_l(1-ω_{l+1}) / ((1-ω_l)ω_{l+1}·2)`（Filippova-Hänel 型非平衡重缩放）；`SpatialAverage` 为逆过程（8 子均值 + 逆 SF） | 与 TensorLBM `rescale_nonequilibrium` 同一物理（粘性一致重缩放），公式可对照移植 |
| 曲壁边界 | `CurvedB()`：对 CB 节点每方向 k（`nb[k]` 为流体）：δ<0.5 时 `f[k]=nb[k]->f[m]·(1-2δ)+temp[m]·2δ`；δ≥0.5 时 `f[k]=nb[k]->f[m]·(2δ-1)/(2δ)+temp[m]/(2δ)`；δ 由 `ComputeDelta()` 按解析几何**逐方向**计算 | 即 Bouzidi-Firdaouss-Lallemand（BFL）插值反弹；TensorLBM 已有向量化等价物（`bfl_common`） |
| 力 | `BBForce()`：`F += v[opp]·(f[i]+nb[i]->f[opp])` 动量交换；按层缓冲 `FXbuff` 累加 | MEM 力在**最细层**累加、按层归一，与 TensorLBM BFL `return_force` 一致 |
| 加密判据 | `Remap()`：`Smag + image + Σnb_image > Crit`，级联缓冲带，远离域边界 | 动态判据（涡量/Smag）可选；物面判据（image）始终参与 |

### 1.2 Octree-LBM-solver（bagalrohit-stack，C++23，D3Q19 BGK，教学型 task1–7 里程碑）

| 要素 | 实现 | 对本设计的价值 |
|---|---|---|
| 树结构 | SoA：`nodes_`（`std::vector<Node>`，`Node{refined_, phantom_, childrenIdx_}`）+ `levelStart_/levelCount_` 按层记账；子节点连续存放（`childrenStartIndex()+0..7`）；描述符字符串（`.`叶子 / `R`细化 / `P`幽灵 / `X`细化+幽灵）构造任意树 | 树结构、几何、数据**三层分离**；内部节点（phantom）不出现在 cell 枚举中——直接对应"壳层叶子 = 只有叶子有 population"的设计 |
| 空间索引 | `MortonIndex`（uint64，每层 3 bit，最深 21 层）：`fromPath/getPath/gridCoordinates/fromGridCoordinates/parent/child` | Z 序编码叶子 → 紧凑枚举 + 拓扑哈希（检查点/一致性校验用） |
| 遍历 | `OctreeCursor`（路径栈）+ 策略迭代器：`PreOrderDFSPolicy`（全树）、`HorizontalPolicy(level)`（只遍历指定层，Z 序） | 按层 Z 序枚举 = 每层一个连续索引区间，与 TensorLBM 张量布局天然契合 |
| 邻接 | `CellGridBuilder`：`levels()` 选层 + `neighborhood()` 给 D3Q19 偏移集 → `adjacencyLists_[offset]` 存每叶子邻居枚举索引 | **关键教训**：`build()` 用 `MortonIndex::fromGridCoordinates(level, …)` 只做**同层**查找，跨层邻居一律 `NO_NEIGHBOR`——该仓库未实现 1:2 粗细界面数据交换（LBM 测试全用均匀网格）。混合架构必须**显式注册跨层界面链接**，不能依赖同层查找 |
| LBM 核 | `InitializePdfs / ComputeMacroscopicQuantities / Collide(BGK) / Stream(pull + try/catch 周期性)` | 简单 BGK、无 MRT/BFL；数据布局 `GridVector<T,Q>`（SoA，(Q, n_cell)）与 TensorLBM `(Q, nz, ny, nx)` 同构 |
| 几何 | `OctreeGeometry{origin, sidelength}`，`dx(level)=side/2^l`，`cellCenter/cellBoundingBox` 由 Morton 推 | 叶子的世界坐标全部可由 Morton 解析，不存每叶子坐标 |

### 1.3 对比结论（直接指导本设计）

1. **邻接重建 vs 增量维护**：两个项目都在拓扑变化后重建邻接表；本设计采用"拓扑一次性预计算为张量索引表"（Python 下避免逐节点遍历）。
2. **跨层界面**：3D-LBM-AMR 靠 `edge` 标志 + 缓冲带（隐式），Octree-LBM-solver 完全没有。本设计采用**显式 interface-link 注册表**（见 §3.2），与 TensorLBM `kinetic_flux_register` 的链接注册思想一致。
3. **时间步进**：3D-LBM-AMR 的 LRLBM 递归 + TemporalInterp 是细层 2 子步的标准做法，与 `NestedStaticBlockAMR3D._advance_interface` 递归等价；TensorLBM 用时间 lerp 的 ghost 填充替代拆分流（更向量化），本设计保留 ghost 填充路线。
4. **BFL 与力**：两参考项目与 TensorLBM 的 BFL/MEM 公式完全同源，直接复用 TensorLBM 向量化实现，仅需把"roll 流"换成"叶子 donor 表 gather"。

### 1.4 术语与概念对照（后续章节沿用）

| 3D-LBM-AMR | Octree-LBM-solver | TensorLBM（既有） | 本设计 |
|---|---|---|---|
| `Node.child[8]` 指针树 | `nodes_` 流 + `childrenStartIndex` | 矩形块张量切片 | Morton 编码 + 层区间枚举 |
| `nb[19]` 邻居指针 | `adjacencyLists_[offset]`（同层） | `neighbor_table` 张量 | `neighbor_table` + 跨层注册表 |
| `edge=1`（需插值） | phantom / `NO_NEIGHBOR` | ghost 层 | 壳层外表面 ghost 叶子 |
| `SpatialInterp1/2` + `SF` | （未实现粗细插值） | `rescale_nonequilibrium` + 三线性 | donor 表 + lerp + 重缩放（同构） |
| `TemporalInterp`（ftemp=0.5(f+ftemp)） | （未实现） | `_fill_ghost` 时间 lerp | 父级两时刻 lerp（同构） |
| `SpatialAverage`（逆 SF） | （未实现） | `restrict_populations_2to1` | 叶子体积平均 + 重缩放 |
| `CurvedB` + `ComputeDelta` | （无 BFL） | `bfl_bounce_back_common` | gather 版 BFL（公式复用） |
| `BBForce` + `FXbuff` | （无力） | BFL `return_force` | MEM 力 + 子步权重归一 |
| `Remap` 判据 + 缓冲带 | （无判据） | `boundary_layer_indicator_3d` | 几何判据 + 过渡带 |

---

## 2. TensorLBM 可复用接口盘点

| 模块/接口 | 位置 | 复用方式 |
|---|---|---|
| `NestedStaticBlockAMR3D` / `StaticBlockAMR3D` | `static_block_amr.py:672/255` | 主体矩形块运行时**原样保留**；`Advance3D` 回调签名 `(f, tau, level_idx, substep) → AMRAdvanceResult(populations, post_collision)` 作为壳层 advance 的统一接口；`convective_refined_tau` 粘性链直接复用 |
| `StaticBlockAMR3D._fill_ghost` / `_build_ghost_sampling_plan` | `static_block_amr.py:392/446` | 时间插值（父级两时刻 lerp）+ 三线性空间插值 + `rescale_nonequilibrium` 的 ghost 填充范式；壳层 ghost 填充推广为"父块单元 donor 表"（§3.2） |
| `_restrict_physical` / `_filter_fine_interface` | `static_block_amr.py:506/534` | 细→粗限制（2×2×2 体积平均 + 重缩放）与界面滤波可直接套用于壳层→块限制 |
| `kinetic_flux_register`（`KineticInterfaceTransfer`、`build_kinetic_interface_links`、`observe_kinetic_interface_transfer`、`apply_face_local_reflux`） | `static_block_amr.py` 导入 | 守恒账本核心；八叉树阶梯面 reflux 按**链接局部**（link-local）扩展（阶梯面链接计数与平面不同，需逐链接注册，沿用其"每条链接计一次、含边角"原则） |
| `fixed_nested_interface.py` | `fixed_nested_interface.py:57/87` | 平面界面 D3Q27 粗细重建原语（`repeat_interleave` 注入 / 2×2 平均限制）作为"面局部"重建的参照与单测对照 |
| `bfl_common.bfl_bounce_back_common` | `bfl_common.py:107` | 全方向向量化 BFL（q<0.5 线性 / q≥0.5 二次，`fluid_boundary_mask`+`q_field`）；壳层版只需把 `torch.roll` 上游查询换成 donor 表 gather |
| `compute_q_sphere_common` / `compute_q_generic_common` / `compute_q_stl_common` | `bfl_common.py:380/671/522`；`triangle_link_intersection.py` | 每叶子每方向的 δ/q 场生成（解析球 + STL 三角链接求交），贴体加密的几何输入 |
| `sphere_amr_common.FineSphere` / `build_fine_sphere` / `bfl_sphere_advance` / `fine_sphere_advance` | `sphere_amr_common.py:42/141/574/603` | 细块几何重采样（partial-shell 鲁棒）与 BFL 驱动的 advance 范式；`bfl_mask/bfl_q` 字段布局直接沿用 |
| `boundary_layer_indicator_3d` | `adaptive_refinement.py:287` | 3D 卷积扩张近壁指示器（掩膜 → 距壁 δ 内流体=1）→ 八叉树加密的**种子判据** |
| `amr_checkpoint`（`save/resume_amr_checkpoint`、`build_checkpoint_signature`） | `amr_checkpoint.py` | 检查点/恢复 + 配置签名 fail-closed；扩展：加入八叉树拓扑哈希（Morton 集 hash） |
| `amr_capability_contract` / `local_refinement_amr_capability_contract_r1.md` | `amr_capability_contract.py` / `docs/` | 新路径默认 `WITHHELD`，验收通过前不得宣称可用（平台 fail-closed 哲学） |
| `suboff_static_amr.py`（`plan_suboff_static_amr`、`build_fine_suboff_mask`） | `suboff_static_amr.py:368/426` | 主体矩形块规划与"细块几何从 CAD 重新光栅化"模式（绝不重复粗体素） |
| 文档基线 | `docs/static-block-amr.md`、`docs/amr-interface-literature-audit-2026-08-01.md` | 验收门（uniform-fine 脉冲、力不变性、界面误差目标）直接继承 |

---

## 3. 混合架构设计

### 3.0 总体结构（三层）

```
L0 粗全域矩形网格（NestedStaticBlockAMR3D 主体，含 1–2 层嵌套块）
        │ 2:1 平面界面（既有机制，不变）
L1 最细矩形块（矩形盒，包裹物面 + 尾流）
        │ ★ 壳层界面（八叉树外表面阶梯面，本设计新增）
L2 八叉树边界壳层（物面周围 δ 层内叶子，BFL 贴体；壳层内最多 2 级深度）
```

- 壳层完全内嵌于 L1 块内部（挖空盒），**不触碰** L1 与外界的平面界面，既有的平面界面机制零改动。
- L2 叶子深度 `d ∈ {1, 2}`（`dx_f = dx_c/2`、`dx_c/4`），起始实现固定深度 1，P3 后开放深度 2。
- 物面落在壳层叶子内部（BFL 贴体，非阶梯逼近）；壳层外边界为阶梯面，其与 L1 的交换走 §3.2 界面。

### 3.1 八叉树边界层数据结构（`octree_boundary/geometry.py`）

1. **壳层域**：`boundary_layer_indicator_3d(mask, re, bl_thickness_cells=δ)` 的 near-wall 流体集合 ∪ 物面 q 场非零集合，再加 1 层过渡带（保证 2:1 平衡、无悬空节点——对应 3D-LBM-AMR `Remap` 缓冲带）。
2. **编码与枚举**：壳层叶子用 Morton 编码（Octree-LBM-solver 方案，uint64，3 bit/层）。`level_start/level_count` 按层记账；叶子按层 Z 序枚举为连续索引 `i ∈ [0, n_leaf)`。拓扑一次性构建：
   - `leaf_morton: (n_leaf,) int64`；`leaf_level: (n_leaf,) int8`
   - `leaf_center / leaf_box: (n_leaf, 3) / (n_leaf, 2, 3)`（由 Morton 解析，不存坐标也行，存了便于调试）
   - `neighbor_table: (Q, n_leaf) int64`，值 ∈ {叶子枚举 | `-1`=壳层外（界面）| `-2`=物面内（固体）| `-3`=周期性/域边界}。**跨层（1:2）界面**：细叶子朝粗方向的邻居记为其"父块单元"（见 §3.2 donor 表），粗叶子朝细方向的邻居记为其 4/8 个细子叶子的集合（`interface_fanout`），由注册表显式给出。
3. **物面 q 场**（`octree_boundary/qfield.py`）：对每个叶子、每个方向 d，用 `compute_q_generic_common`（STL）或 `compute_q_sphere_common`（解析球）求 `q[d,i] ∈ (0,1]`；`fluid_boundary_mask[d,i] = (q>0)`。q 场在**叶子坐标**上直接求值（壳层叶子世界坐标已知），不经过块坐标映射——真贴体。
4. **拓扑校验**：邻接对称性（`neighbor_table[d,i]=j ⇒ neighbor_table[opp[d],j]=i`，跨层链接按注册表核对）、2:1 平衡（相邻叶子深度差 ≤1）、壳层无悬空节点、与 L1 块的包含关系完备。

张量布局约定（全部为紧凑连续张量，热路径零 Python 循环）：

```python
# 每层叶子数 n_l = level_count[l]；总叶子 n_leaf = Σ n_l
leaf_morton   : (n_leaf,)  int64   # Morton 编码（Z 序）
leaf_level    : (n_leaf,)  int8    # 深度
leaf_center   : (n_leaf, 3) float32
leaf_box      : (n_leaf, 2, 3) float32   # [min, max] 角点
level_start   : (L+1,) int64       # 每层枚举区间 [start_l, start_{l+1})
neighbor_table: (Q, n_leaf) int64  # 叶子枚举 | -1 壳层外(界面) | -2 物面内 | -3 域边界
q_field       : (Q, n_leaf) float32
bfl_mask      : (Q, n_leaf) bool
f_leaf        : (Q, n_leaf) float32       # 壳层 population（SoA，同 GridVector 风格）
ghost_donor   : (n_ghost, 3) int64        # ghost 叶子 → L1 块宿主单元 (z,y,x)
interface_links: (n_link, 2) int64        # (叶子 i, 方向 d) 穿出壳层的链接注册
upstream_donor: (Q, n_leaf) int64         # BFL 上游点 x-c_d 的叶子枚举（-2 固体）
```

块侧视角：L1 块新增一张 `shell_covered: (nz, ny, nx) bool`（壳层覆盖区 = 块内"固体"），块 collision/stream 对覆盖区冻结；壳层对其负责。块侧 `coarse_interface_links` 与壳层 `interface_links` 在 reflux 时配对（与 `build_kinetic_interface_links` 相同的链接语义）。

### 3.2 与矩形块的界面（壳层 ↔ L1 块）

1. **包含关系（donor 表）**：每个壳层叶子中心投影到 L1 块 with-ghost 坐标，得宿主单元 `(z,y,x)`；壳层 ghost（外表面 1 层虚拟叶子，不存 population）的 donor 是宿主单元 + 时间插值 + 三线性 + `rescale_nonequilibrium`——与 `_fill_ghost` 同构，仅 donor 索引从规则网格变为叶子→块单元映射表 `(n_ghost, 3)`。
2. **界面链接注册（interface-link 注册表）**：遍历每个叶子每个方向 d：若 `neighbor_table[d,i]==-1`（穿出壳层），注册链接 `(i, d)`；反方向 `(宿主单元, opp[d])` 在 L1 侧。此注册表即 `build_kinetic_interface_links` 的八叉树推广（阶梯面：每个方向、每条链接各计一次，含边/角）。
3. **通量与 reflux**：L1 每粗步内壳层走 `2^d_max` 子步；每子步用 `observe_kinetic_interface_transfer` 在注册表上观察壳层侧出/入通量（按细体积缩放），粗步末合并，与 L1 侧通量做 `apply_face_local_reflux` 的 link-local 修正（阶梯面不做面平均近似）。守恒账本 `PopulationRefluxLedger` 全链路保留。
4. **限制（壳层 → 块）**：块内被壳层覆盖的单元，其 population 由壳层叶子 2×2×2（或按深度加权）体积平均 + `rescale_nonequilibrium`（τ 链用 `convective_refined_tau`）回填；块内未被壳层覆盖的单元照常由块自己推进。挖空盒内的块单元在壳层激活期间**冻结**（由块侧 solid 掩膜扩展实现，即把壳层覆盖区当"块内固体"处理，壳层负责其物理）。

### 3.3 时间步进（`octree_boundary/stepping.py`）

- 根步循环：L0/L1 由 `NestedStaticBlockAMR3D.step` 推进（不变）；壳层在 L1 每步内做 `2^d_max` 子步（`advance` 回调签名与 `Advance3D` 一致）。
- 每壳层子步序列（参照 LRLBM 但保持向量化）：
  1. ghost 填充：父级（L1）两时刻 `lerp` 时间插值 + 三线性空间插值 + 非平衡重缩放（等价 3D-LBM-AMR 的 TemporalInterp + SpatialInterp 合体）；
  2. collide（`advance` 回调内完成，返回 `AMRAdvanceResult`）；
  3. stream：push 语义经 `neighbor_table` 的 gather/scatter（**不用 roll**，叶子不连续）；
  4. BFL 贴体重构（§3.4）；
  5. 界面通量观察、力累加。
- 壳层内若存在深度差（d=2 区嵌套在 d=1 区内）：d=1 叶子在 d=2 区边界用时间插值补中间时刻（TemporalInterp 语义：`f_tmp = 0.5*(f+f_next)`），d=2 区每 d=1 步走 2 子步——递归调度与 `_advance_interface` 同构，P3 实现。
- 子步权重：力与通量按 `1/2^(level-d)` 归一（参照 `FXbuff` 分层累加 + `subcycled_force.py`）。

根步伪代码（`HybridOctreeBoundaryAMR3D.step`，d_max=1 时）：

```python
def step(self, advance, *, tau_by_level):
    # 1) 块层推进（既有 NestedStaticBlockAMR3D.step，壳层覆盖区冻结）
    coarse_old = self.block.coarse_f.clone()
    block_ledgers = self.block.step(advance, tau_by_level=tau_by_level)
    # 2) 壳层在 L1 步内推进 2^d_max 个子步
    for s in range(2 ** self.shell.d_max):
        alpha = s / (2 ** self.shell.d_max)
        parent_t = torch.lerp(coarse_old, self.block.finest_f, alpha)
        self.shell.fill_ghost(parent_t, tau_source=tau_c, tau_target=tau_f)   # donor 表 + lerp + 重缩放
        out, post = advance(self.shell.f_leaf, tau_f, self.shell_level, s)     # Advance3D 回调
        out = self.shell.stream_gather(out)                                    # neighbor_table gather/scatter
        out = self.shell.bfl_apply(out, post)                                  # gather 版 BFL + MEM 力
        self.shell.f_leaf = out
        self.shell.observe_interface_transfer(post)                            # 界面通量（细体积加权）
    # 3) 壳层 → 块限制 + reflux
    restricted = self.shell.restrict_to_block()                                # 叶子体积平均 + 重缩放
    self.block.paste_shell_restriction(restricted)
    self.block.apply_shell_reflux(self.shell.transfer_ledger)                  # link-local reflux
    return block_ledgers + (self.shell.last_reflux,)
```

该序列与 `_advance_interface` 的"粗步 → 细子步（ghost 时间插值）→ 限制 + reflux"完全同构，仅 ghost 填充与 stream 的索引机制不同；d=2 时壳层内再套一层同构递归。

### 3.4 物面 Bouzidi BFL（`octree_boundary/bfl.py`）

- 直接复用 `bfl_common.bfl_bounce_back_common` 的公式与分支（q<0.5 线性：`2q·f_d + (1-2q)·f_d(x-c_d)`；q≥0.5 二次：`f_d/(2q) + (2q-1)/(2q)·f_opp`）。
- 唯一改动：上游点 `x-c_d` 的查询由 `torch.roll` 改为**按方向 donor 表 gather**（`upstream_donor[d]` = 叶子 i 沿 -d 的邻居枚举；物面内记 -2，该方向由 BFL mask 排除）。逐方向向量化循环（19/27 次 gather），与 `_bouzidi_bounce_back_d3q27` 的循环结构一致。
- 动壁修正（`bfl_moving_wall_correction` / ramp 启动）原样保留；`fluid_boundary_mask`、`q_field` 均为壳层紧凑索引张量 `(Q, n_leaf)`。

### 3.5 力闭合（`octree_boundary/force.py`）

1. **动量交换（MEM）**：BFL `return_force` 逐壳层子步累加（细叶子上求值），按子步权重归一到根步——对应 3D-LBM-AMR `BBForce` + 分层缓冲。单链接公式（与 `bfl_d3q19`/`_bouzidi_bounce_back_d3q27` 一致）：

   ```
   F_root += Σ_substeps w_sub · Σ_links c_d · (f_d + f_bc),   w_sub = 2^(-(d_max - d_leaf))
   ```

   （`f_bc` 为 BFL 重构的未知 population，`c_d` 为链接方向；物面在细叶子上求值，天然贴体。）
2. **控制体积（CV）交叉验证**：复用 `sphere_bfl_control_volume` / `control_volume_force`；**CV 采样面必须避开壳层外界面与界面滤波壳**（文献审计 gate 5：独立力不得采样数值界面处理），几何清除门 fail-closed。
3. 输出：MEM 力、CV 力、二者偏差、每子步力历史（检查点持久化）。

### 3.6 加密判据（P4 开放）

- **P1–P3 固定壳层**：几何判据唯一——`boundary_layer_indicator_3d` 近壁 δ + 物面局部曲率半径 `r_curv < k·dx` 处加一级（球体/椭球解析可算，STL 用相邻三角法向角估计）。
- **P4 动态**：可选 Smag/涡量判据（3D-LBM-AMR `Remap` 思路）驱动壳层外扩/内缩；任何重网格必须保持 2:1 平衡 + 过渡带，重网格事件间隔 ≥ N 步（避免抖动），且拓扑重建后全量重建 donor/邻居/链接注册表（快照式重建，不做增量维护）。

---

## 4. 分阶段开发计划

| 阶段 | 内容 | 关键交付 | 验收要点 |
|---|---|---|---|
| P1 几何 | 壳层掩膜/Morton/邻居表/跨层注册/donor 表/q 场 | `geometry.py`、`topology.py`、`qfield.py` | 邻接对称、2:1 平衡、壳层体积 <0.5% 误差、donor 覆盖完备 |
| P2 步进 | 壳层 advance/ghost/限制/reflux/子步调度（d=1） | `stepping.py`、`hierarchy.py` 骨架 | 质量漂移 <1e-6、reflux 残差 <1e-10、平面特例与 `StaticBlockAMR3D` 一致 |
| P3 物理 | gather BFL/MEM 力/CV 力/动壁 ramp/壳层内 d=2 | `bfl.py`、`force.py` | 低 Re 球 Cd 偏差 ≤2%、MEM/CV 偏差 ≤5%、界面位置无关 |
| P4 验证 | 三路线对照/网格收敛/动态加密试点/合同升级 | 验证脚本与报告 | cell saving 提升 ≥10pt、Cd 与基线一致、5 条 promotion gates 全过 |

### P1 几何（`octree_boundary/geometry.py`、`qfield.py`、`topology.py`）

- 壳层掩膜（指示器 + 过渡带）、Morton 编码与叶子枚举、`neighbor_table`（含跨层注册）、叶子↔块 donor 表、interface-link 注册表、q 场（解析球先行）、拓扑校验与 VTK 导出。
- **验收标准**：① 邻接对称性与 2:1 平衡单测全绿；② 跨层链接注册完备（细→粗 fanout 与粗→细 donor 一一对应）；③ 球体壳层叶子体积 vs 解析球壳体积误差 < 0.5%；④ donor 表覆盖每个 ghost 叶子且落在 L1 物理域内；⑤ VTK 可视化人工确认贴体（无锯齿物面、无悬空节点）。

### P2 步进（`octree_boundary/stepping.py`、`hierarchy.py` 骨架）

- 壳层 advance（donor 表 gather 流 + collide 回调 + ghost 填充 + 限制 + reflux ledger）、子步调度（d_max=1 固定）、`HybridOctreeBoundaryAMR3D` 主运行时组合块层+壳层。
- **验收标准**：① 均匀自由流跨壳层界面：相对质量漂移 < 1e-6，reflux 残差 < 1e-10，无 population 为负/NaN；② 平滑密度脉冲穿越壳层：与 uniform-fine 参考（限制回块控制体积）对比，壳层区密度 RMS 误差 ≤ 均匀粗网格对应值；③ 平面界面特例（壳层退化为矩形盒）与 `StaticBlockAMR3D` 逐链接通量一致（回归对照）；④ 力接口占位返回 0 时全程有限。

### P3 物理（`octree_boundary/bfl.py`、`force.py`）

- gather 版 BFL、MEM 力、CV 力交叉验证、子步权重归一、动壁 ramp；开放壳层内 d=2 深度。
- **验收标准**：① 静止球（解析 q 场）：BFL 重构后球面法向速度穿透 < 1e-3 量级；② 低 Re 球（Re=10/100）：Cd 与 Schiller-Naumann 相关式偏差 ≤ 2%（统计收敛后）；③ MEM 力与 CV 力偏差 ≤ 5%，且二者均不随壳层界面位置平移而漂移（界面敏感性测试）；④ d=2 壳层：球面附近速度剖面与 d=1 收敛一致（无界面反射可见）。

### P4 验证（对照 `examples/amr_sphere_shell_l3_validate.py`）

- 同一球体算例三路线对照：矩形块路线（基线）、混合路线、uniform-fine 参考；网格收敛（3 套）；动态加密可选试点。
- **验收标准**：① 混合路线 `cell_saving_fraction` 高于矩形块路线（消除边角料，目标相对提升 ≥ 10 个百分点）；② Cd/尾流速度与矩形块路线偏差在统计噪声内（且与 uniform-fine 一致）；③ 通过 `amr-interface-literature-audit` 全部 5 条 promotion gates（质量守恒、L90 延续、CV 清除、几何相似、力不变性）；④ 经 `amr_capability_contract` 评审升级为 `AVAILABLE`（在此之前一律 `WITHHELD`）。

---

## 5. 风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| 跨层 1:2 界面邻居缺失（Octree-LBM-solver 未实现跨层邻接） | 细叶子流不到粗数据，界面通量错 | 显式 interface-link 注册表 + donor/fanout 双向表（§3.2），拓扑校验强制对称 |
| 叶子紧凑索引下 `torch.roll` 失效 | stream/BFL 上游查询错位 | 全部改为按方向 donor 表 gather/scatter；单测验证 gather 结果与规则网格 roll 一致（平面特例） |
| 阶梯面 reflux 链接计数与平面不一致 | 守恒账本残差 | link-local 逐链接注册（含边/角），不做面平均近似；平面特例回归对照 `StaticBlockAMR3D` |
| 壳层深度 > 2 子步数爆炸（4×/8×） | 性能与调度复杂度 | 起始固定 d=1；d=2 仅物面局部；动态判据 P4 才开放 |
| 力采样污染（界面数值处理进力） | Cd 假信号 | CV 几何清除门 fail-closed + MEM/CV 双路交叉验证 + 界面平移敏感性测试 |
| 八叉树拓扑漂移（检查点/恢复） | 断点续算错位 | `amr_checkpoint` 签名扩展：Morton 集 hash 进配置签名，不匹配即拒绝恢复 |
| Python 层逐节点遍历慢 | 每步开销大 | 拓扑一次性预计算为张量索引表；热路径全向量化；重网格快照式重建（低频） |
| 与既有平面界面机制冲突 | 回归风险 | 壳层严格内嵌于最细块内部（挖空盒），不触碰 L1 外平面界面；既有单测全保留 |
| 过度宣称能力（平台 fail-closed 哲学） | 合同违约 | 新路径默认 `WITHHELD`，逐阶段验收后才升级 |

---

## 6. 代码组织文件清单

```
docs/octree-boundary-design.md                    # 本文档
src/tensorlbm/octree_boundary/
  __init__.py                                     # 导出 HybridOctreeBoundaryAMR3D 等
  geometry.py        # 壳层掩膜、Morton 编码、叶子枚举、neighbor_table、2:1 平衡、拓扑校验
  topology.py        # interface-link 注册表、跨层 donor/fanout 表、reflux 链接查询
  qfield.py          # 每叶子每方向 q 场（解析球 / STL 泛型，委托 bfl_common）
  stepping.py        # 壳层 advance、ghost 填充、限制、子步调度（d_max=1/2）
  bfl.py             # gather 版 Bouzidi BFL + 动壁修正（委托 bfl_common 公式）
  force.py           # MEM 力（子步归一）、CV 力交叉验证、界面清除门
  hierarchy.py       # HybridOctreeBoundaryAMR3D 主运行时（组合块层 + 壳层 + 账本）
  checkpoint.py      # 壳层检查点（Morton 拓扑 hash + populations + 力历史）
tests/
  test_octree_boundary_geometry.py    # P1 验收
  test_octree_boundary_stepping.py    # P2 验收（含平面特例回归对照）
  test_octree_boundary_bfl.py         # P3 验收（q 场恢复、力一致性）
  test_hybrid_amr_sphere.py           # P4 验收（三路线对照）
examples/
  hybrid_octree_sphere_validate.py    # 混合路线球体验证入口（对齐 amr_sphere_shell_l3_validate.py）
```

不修改任何既有源码；既有 `static_block_amr.py`、`sphere_amr_common.py`、`bfl_common.py` 等保持只读，壳层通过导入与组合接入。
