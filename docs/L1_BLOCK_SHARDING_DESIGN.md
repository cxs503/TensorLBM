# L1 块分片化设计方案（Phase 2：x 域分解 + halo 交换）

> 目标：把 L1 中间块从"每 rank 全量复制"改为"按 x 域分解，每 rank 只算自己的
> slab + halo 交换"，消除 Phase 1 的冗余计算与 window 全量 all_gather，
> 恢复 4 卡以上的强扩展。
> 本文件为静态设计与实现计划，不涉及代码落地。
> 前置文档：`docs/L1_MIDDLE_BLOCK_INTEGRATION_DESIGN.md`（§3a Phase 2 / §6.2.1）。

## 0. 背景：Phase 1 的成本结构（为什么要分片）

当前 L1 块实现（`src/tensorlbm/octree_boundary/l1_block.py`，`L1BlockDistributed`）：

- **数据**：每 rank 冗余持有全块 `(27, nz_l1+2, ny_l1+2, nx_l1+2)` 带 ghost 张量；
  实测案例 L1 块 60×60×120（432k cell ≈ 46.7MB/rank，默认 R6 球约 90×44×44 ≈ 18.8MB）。
- **计算**：每 root step 全块 2 子步 cumulant collide + `stream27_roll` + freeze，
  每个 rank 都算**整个块**（W 卡 = W 倍冗余 FLOPs）。
- **通信**：每 root step 3 次 window chunked all_gather（old / new / post，
  `(27, ~40k) ≈ 4.3MB` 每次，分块 <3MB/msg）+ 壳层 stage 自身的全帧 broadcast
  （壳层 restrict/reflux 补丁，`(27, nz_l1, ny_l1, nx_l1) ≈ 18.8MB`，分块）。

**观测到的瓶颈**（父任务上下文）：4 卡后每步时间 ≈ 固定成本 ~4.4s 不再下降。
组成分解：

| 成本项 | 随 W 的缩放 | 说明 |
|---|---|---|
| L1 复制计算（2 子步 × 全块） | O(W) 冗余，**不降** | 每 rank 都算整个 60×60×120 块 |
| 3 × window all_gather | O(W·4.3MB) 总消息量，串行 | 每 rank 收 3×4.3MB |
| 壳层 all_gather（每子步 (27, n_leaf)） | O(W·n_leaf) 总消息量 | 既有架构，非本次范围 |
| coarse 演化（slab 本地） | O(1/W) | 已扩展良好 |

分片化目标：**消除前两项**（L1 复制计算 + window 全量 all_gather），
壳层 all_gather 成为新的扩展地板（见 §7 后续路线）。

## 1. 设计总览

### 1a. 核心思路：与 coarse 同构的 x-slab 分解

完全复用 coarse 域的成熟模式（`octree_integrated_validate.py` L352-364 的
`halo_exchange` + `stream27_roll`）：

- L1 块按 **coarse slab 对齐**分解为 x-slab：rank r 拥有 L1 x 区间
  `[lo_l1, hi_l1)`，其中

  ```
  lo_l1 = 2·(max(lo, box.x0) − box.x0)      # lo/hi = 该 rank 的 coarse 列区间
  hi_l1 = 2·(min(hi, box.x1) − box.x0)
  nx_l1_local = hi_l1 − lo_l1               # 可为 0（slab 与 box 不相交）
  ```

- 每 rank 张量：`(27, nz_l1+2, ny_l1+2, nx_l1_local+2)` —— 一个带 ghost 的 slab，
  z/y 全幅（不分解），x 方向 1 列 halo（左列 local 0、右列 local nx_l1_local+1），
  与 coarse slab `(27, nz, ny, nx_local+2)` 完全同构。
- L1 物理列 g 位于 local 索引 `g − lo_l1 + 1`（与 coarse 的 `g − lo + 1` 同约定）。

**对齐分解的三个关键性质**（贯穿全文，全部由 `box.x0/x1` 与 coarse slab 边界
`lo/hi` 的对齐保证）：

1. **ghost fill 零新增通信**：rank r 的 L1 ghost 单元所需 coarse 父单元
   （2:1 injection 采样，含 box±1 环）全部落在 rank r 自己的 coarse slab + 其
   1 列 coarse halo 内（§3a 证明）→ 不再需要 window all_gather。
2. **restriction 零新增通信**：rank r 拥有的每个 coarse box 单元的 8 个 L1 子单元
   （2×2×2 块）全部在 rank r 自己的 slab 内 → L1→coarse 2:1 restrict 全本地。
3. **reflux 掩码静态可切片**：box 界面 links / 壳层观测 links 全是静态几何掩码，
   在全局帧构建一次后按 x 切片到各 rank，逐格观测部分和 + (Q,) all_reduce 组装
   （§3d）。

### 1b. 与 Phase 1 的差异对照

| 维度 | Phase 1（复制，现状） | Phase 2（分片，本设计） |
|---|---|---|
| L1 张量 | `(27, nz_l1+2, ny_l1+2, nx_l1+2)` 全块/rank | `(27, nz_l1+2, ny_l1+2, nx_l1_local+2)` slab/rank |
| L1 计算 | 全块 × W 冗余 | 每 rank 只算自己 slab（O(1/W)） |
| L1 ghost fill | 从 gathered window lerp 采样 | 从**本 rank coarse slab** lerp 采样（零通信） |
| L1 子步间通信 | 无 | 1 次 halo 交换（2 个 all_gather，~0.5MB/侧） |
| window all_gather ×3 | 有（~13MB/步） | **删除** |
| coarse↔L1 耦合 | 全帧窗口 restrict/reflux | 本地 restrict + 逐格观测 all_reduce + 本地 reflux 应用 |
| 壳层接口（l1_old/l1_f/l1_post） | 全帧复制张量直接传 | 1 次全帧 gather 组装 + 既有 broadcast 复用（§4） |
| 确定性 | 位一致 | 演化路径位一致；reflux 观测求和顺序变化 → roundoff 级一致（§6） |

### 1c. 每 root step 时序（Phase 2）

```
# ---- 阶段 A：coarse 演化（完全不变）----
coarse_old = coarse_f.clone()              # 本地 slab 克隆（lerp 锚点）
post = collide_coarse(coarse_f); halo_exchange(post); stream; BB; far_field

# ---- 阶段 B：L1 块 stage（slab 本地 + halo 交换）----
for s in {0, 1}:
    fill_ghost_slab(lerp(coarse_old, coarse_f, s/2))     # 本 rank coarse slab 采样，零通信
    post = collide_l1(l1_slab, tau_l1)
    post_frozen = where(solid, before, post)              # freeze 先于交换（见 §3b）
    halo_exchange_l1(post_frozen)                          # 仅 x-cut 列，2 个 all_gather ~0.5MB
    streamed = stream27_roll(post_frozen)                 # wrap 伪影落在 halo 列（§3b）
    l1_slab = where(solid, before, streamed)
    捕获 posts_phys_slab / posts_ghost_slab（per-rank）
    fill_ghost_slab(lerp(coarse_old, coarse_f, (s+1)/2))

# ---- 阶段 C：壳层 stage（接口不变）----
l1_f_phys = gather_full_l1_field(slab 物理切片)           # 1 次 chunked all_gather（§4）
step_octree_shell_distributed(octree, advance_shell,
    l1_full_pre, l1_f_phys, tau_coarse=tau_l1,
    l1_post=posts_phys_full, reflux=True, ...)            # 内部 rank0 修改 + 全帧 broadcast（既有）
l1_full_pre = 广播后的全帧场                           # 即下一步的 pre 场，零额外通信
block.set_physical(extract_slab(l1_full_pre))             # 取回本 rank slab

# ---- 阶段 D：L1→coarse 耦合（本地 + 小 all_reduce）----
restricted = restrict_populations_2to1(本 rank 拥有的 box 区)   # 全本地
coarse_transfer, fine_transfer = 逐格观测部分和 → all_reduce((2Q,))   # 两个 (Q,) 合并
requested = project_onto_conserved_moments(全局 mismatch)   # 各 rank 相同
apply 到本 rank 的 box 环 exterior cells（预切片掩码，逐格元素操作）
写回 coarse_f slab（restrict + reflux 补丁）
力累积 + 报告（不变）
```

## 2. 数据布局

### 2a. slab 张量（持久状态）

```
l1_slab: (27, nz_l1+2, ny_l1+2, nx_l1_local+2)   # dtype=float32，device=dev
```

- `nx_l1_local = 2·(min(hi, box.x1) − max(lo, box.x0))`，偶数（边 rank 部分 slab
  也是偶数：2×整数）。
- 物理内部 `[:, 1:-1, 1:-1, 1:-1]`；z/y ghost 层（0 / nz_l1+1 / 0 / ny_l1+1）
  每子步由 coarse slab lerp 采样填充（同 Phase 1 语义）；
  x ghost 列分两类：
  - **box 全局面列**（`lo_l1 == 0` 的左列 / `hi_l1 == nx_l1` 的右列）：由 coarse
    lerp 采样填充（AMR 界面，同 Phase 1）；
  - **x-cut 列**（slab 边界，邻居 rank 非空）：由 L1 halo 交换填充（真实演化的
    邻居值，**不是** coarse 采样值），并**从 coarse-fill ghost 掩码中排除**。
- 与 coarse slab 的对应关系（建立一次，静态）：

```
xc_map: (nz_l1+2, ny_l1+2, nx_l1_local+2)  → coarse 列全局坐标
coarse slab 本地索引 = xc_map − lo + 1     # 含 halo 列，值域 [0, nx_local+1]
zc_map / yc_map: 全幅（z/y 不分解）
```

### 2b. 静态几何掩码（构建一次，按 x 切片）

| 掩码/links | 全局帧构建 | per-rank 切片 |
|---|---|---|
| `box_owned`（coarse box 内部） | 窗口帧 `(nz_w, ny_w, nx_w)` | x ∈ [lo_l1/2 对应 coarse 列] 切片 |
| `box_links`（coarse↔L1 界面） | `build_kinetic_interface_links(box_owned)` | 切片后仅用于观测部分和 |
| `l1_fine_links`（L1 侧界面） | L1 with-ghost 帧 | x ∈ [lo_l1, hi_l1) 切片 |
| `exterior_cells` / `receiving`（reflux 修正掩码） | 窗口帧一次 roll 计算 | 切片到各 rank（含 coarse halo 列） |
| `ghost_mask_local`（coarse-fill 掩码） | L1 with-ghost 帧 | 切片 + 排除 x-cut 列 |
| `l1_solid_q`（freeze 掩码） | `octree._solid` 映射到 L1 with-ghost | 切片（halo 列置 False，见 §3b） |

> 注：`build_kinetic_interface_links` 要求 owned volume "严格内嵌"
> （`inside[:, :, -1].any()` 等检查，`kinetic_flux_register.py` L117-122）。
> per-rank 切片掩码**在全局帧构建后再切片**，规避该检查（box 面恰在 slab 边缘列时
> 局部帧会误触发）。所有掩码静态 → 切片开销一次性。

### 2c. 负载均衡与空 slab

- 对齐分解下：box 内部 rank 得 `2·nx_local` 列（如 W=16、nx_local=6 → 12 列），
  两个边缘 rank 得部分 slab，box 外侧 rank 得空 slab（跳过 L1 计算，仅参与
  集体通信，贡献零值）。
- 目标规模（W ≥ 8）下 box 宽度（≈50-90 coarse 列）≫ slab 宽度，边缘/空 rank
  占比小，均衡可接受。
- **小 W 备选**（W ≤ 4 时若失衡显著）：改为均衡 base+extra 分解
  （复用 `split_leaf_bounds` 语义），代价是 ghost fill 的 coarse 父单元可能落在
  邻居 coarse slab → 需为 ghost 采样引入一个小型 donor gather（§5 风险 4 的缓解）。
  默认采用对齐分解。

## 3. 算子改造细节

### 3a. ghost fill：从"窗口采样"到"本 rank coarse slab 采样"

**对齐性质证明**（`lo_l1 = 2·(max(lo,box.x0) − box.x0)` 等）：

- L1 ghost 单元（local x = 0 或 nx_l1_local+1，含 z/y 方向任意位置）的 coarse 父
  单元 x 坐标（`(xoff − 1)//2` 语义，clamp 到窗口 `[box.x0−1, box.x1+1]`）：
  - 左边界（`lo_l1 > 0` 时）：父 x = `max(lo,box.x0) − 1 ≥ lo − 1`，且 `< hi`
    → 落在本 rank coarse slab 或左 halo 列（`g − lo + 1 = 0`，coarse halo 交换
    已刷新）。
  - 右边界（`hi_l1 < nx_l1` 时）：父 x = `min(hi,box.x1) ≤ hi` → 本 rank slab
    或右 halo 列。
  - box 全局面（`lo_l1 == 0` / `hi_l1 == nx_l1`）：父 x = `box.x0 − 1`（左，
    ≥ lo−1 ✓）/ `box.x1`（右，≤ hi ✓）。
  - z/y 父坐标全幅（z/y 不分解），任意 L1 ghost 的 z/y 父都在窗口 z/y 范围内。
- 因此 `_fill_ghost` 改写为：`lerp(coarse_old, coarse_f, alpha)`（本 rank slab）
  → 按 `xc_map − lo + 1` gather → `rescale_nonequilibrium(tau_c → tau_l1, ratio=2)`
  → `torch.where(ghost_mask_local, sampled, l1_slab)`。**零通信**。
- 初始化（step 1）：`initialize_from_slab(coarse_f)` 同上（等价 Phase 1 的
  `initialize_from_window`）；`initialize_uniform` 不变（逐格）。

### 3b. 子步演化：collide → freeze → halo 交换 → stream

子步 s 的固定顺序（与 Phase 1 位一致的**必要**条件）：

```
1. fill_ghost_slab(lerp(coarse_old, coarse_f, s/2))   # z/y 面 + box 全局面（本地）
2. post = collide_fn(l1_slab, tau_l1)                 # 逐格
3. post_frozen = where(l1_solid_q, before, post)      # freeze 必须先于交换：
   # 越过 x-cut 的流值必须是 post_frozen（固体单元流的是 frozen before，
   # 与 Phase 1 全张量语义一致）；若先交换再 freeze，邻居拉到固体 post，
   # 位不一致。
4. halo_exchange_l1(post_frozen)                       # 见下
5. streamed = stream27_roll(post_frozen)               # 逐格 roll
6. l1_slab = where(l1_solid_q, before, streamed)
7. 捕获 posts_phys_slab（物理切片 post_frozen）/ posts_ghost_slab（with-ghost）
8. fill_ghost_slab(lerp(coarse_old, coarse_f, (s+1)/2))  # 同 Phase 1 双填充
```

**`stream27_roll` 在 slab 上的正确性（关键）**：torch.roll 是周期性的，直接对
`(…, nx_l1_local+2)` 张量 roll 会在 slab 边界产生 wrap 伪影。但：

- x 方向位移只有 ±1（D3Q27 相邻格），halo 宽恰好 1；
- pull-stream `out[g] = f_post[g − sx]`：`sx=+1` 时 local 1 读 local 0（左 halo，
  交换已刷新 ✓）；`sx=−1` 时 local nx_l1_local 读 local nx_l1_local+1（右 halo ✓）；
- wrap 伪影只落在 local 0 与 local nx_l1_local+1 两个 **halo 列**上——它们在下一次
  交换前不会被任何真实单元读取（真实单元只读本列/邻居的 halo 列，而 halo 列在每次
  stream 前都被交换刷新）。
- 这正是 coarse 域已用的同款技巧（`halo_exchange(post)` 先于 `stream27_roll(post)`），
  无需新算子。

**`halo_exchange_l1`**（复用 coarse 的 all_gather 模式）：

```
# 仅当 hi_l1 < nx_l1（有右邻居）：右真实列 (local nx_l1_local) 全列
#   (27, nz_l1+2, ny_l1+2, 1) → all_gather → 取 gathered[(rank+1)%W] → 写 local 0
# 仅当 lo_l1 > 0（有左邻居）：左真实列 (local 1) 同理 → 写 local nx_l1_local+1
# 空 slab rank：参与 all_gather（发零），不读写
# 消息 ~457KB/侧（27×92×46×4B），<3MB 无需分块；每子步 2 个 all_gather，每 root 步 4 个
```

交换**整列**（含 z/y ghost 层）是必须的：Phase 1 中流经 cut 位置单元的 z/y ghost
值是 coarse 采样值，邻居的整列（含其 z/y ghost）跨 cut 传递，值语义一致。

### 3c. 壳层 stage 接口：全帧组装 + 既有 broadcast 复用

`step_octree_shell_distributed`（`distributed_stepping.py` L380-826）需要**全帧**
L1 场：`l1_old`/`l1_f`（ghost plan 按全局坐标采样 + restrict 目标）与
`l1_post`（壳层 reflux 的 coarse 侧观测，全帧 links）。Phase 2 的组装：

| 接口 | Phase 2 来源 | 通信 |
|---|---|---|
| `l1_old`（pre 场） | **复用上一步壳层 stage 结束时的全帧 broadcast 结果**（`l1_full_pre`） | 0（免费） |
| `l1_f`（post-L1-stage 场） | `gather_full_l1_field_chunked(物理 slab)`：chunked all_gather + 求和组装（同 `gather_window_chunked` 技巧，每列恰有一个 owner 贡献非零 → 位精确） | 1 次 ~18.8MB（分块 <3MB） |
| `l1_post`（post-collision ×2） | **方案 A（零 stepper 改动）**：对每子步 post_frozen 物理切片再各做一次全帧 gather | 2 次 ~18.8MB |
| | **方案 B（推荐，小改 stepper）**：调用方逐格观测本 rank slab 的壳层 links 部分和 → all_reduce → 以 `KineticInterfaceTransfer` 传入（stepper 加可选参数 `l1_post_transfer`，~10 行） | 2 个 (Q,) all_reduce |

- 壳层 stage 结束时 rank 0 的全帧 chunked broadcast（既有，L759-772）保持不变，
  它同时充当**下一步的 `l1_full_pre`**（广播后各 rank 位一致；Phase 1 中
  `set_physical(l1_f_phys)` 写入的正是该场——语义完全对齐）。
- 随后 `block.set_physical(extract_slab(l1_full_pre))`：各 rank 从全帧场切回自己的
  slab 列。
- 壳层内部（`restrict_shell_to_block`、壳层 reflux、`full_pc` all_gather、
  `interleaved_leaf_indices`、BFL）**全部不变**。

### 3d. L1→coarse 耦合：本地 restrict + 观测 all_reduce + 本地 reflux

1. **restrict（全本地）**：对本 rank 拥有的 box 单元（coarse x ∈
   `[max(lo,box.x0), min(hi,box.x1))`，即 L1 x ∈ `[lo_l1, hi_l1)` 对应偶数块）：
   `restrict_populations_2to1(物理切片子区)`（2×2×2 均值，块边界与 slab 对齐，
   偶数跨度）→ `rescale_nonequilibrium(tau_l1 → tau_coarse, spatial_ratio=1/2)`
   → 直接写回本 rank coarse slab 的对应列。**零通信**。
2. **观测部分和（两个 (Q,) 合并为一次 all_reduce）**：
   - coarse 侧：`observe_kinetic_interface_transfer(本 rank coarse post slab, 切片 box_links)`
   - fine 侧：对 `posts_ghost_slab` 逐子步 `observe(…, 切片 l1_fine_links, cell_volume=1/8)` 累加
   - `dist.all_reduce(torch.cat([coarse_transfer, fine_transfer]))`（(2Q,) ≈ 216×4B，单消息）
3. **reflux 应用（逐格元素操作，各 rank 相同修正向量）**：
   `raw_mismatch = fine − coarse`（全局）→ `project_onto_conserved_moments`
   → 各 rank 得到**相同的** `requested` (Q,) → 用预切片静态掩码
   （exterior cells 含 coarse halo 列）对本 rank 的 box 环单元做
   `_apply_population_total` 级修正（含 `maximum_correction_fraction=0.2` 限幅，
   逐格基于本单元库存，无跨 rank 读取）。
   - 注意 `apply_face_local_reflux` 内部对 `outgoing_origins` 的 roll（L319-322）
     是几何掩码运算 → **在全局帧构建时一次算好 receiving/exterior 掩码再切片**，
     每步不再 roll。
4. **回写**：restrict + reflux 补丁直接写入 `coarse_f` slab（含 halo 列由下次
   coarse `halo_exchange` 刷新，同现状）。`write_window_back` 删除。
5. **ledger**：各 rank 报告部分量（corrected_links 计数、residual 和）→ all_reduce
   组装全局 ledger（验收门禁用）。

## 4. 通信清单（每 root step，Phase 2 vs Phase 1）

| 通信 | Phase 1 | Phase 2 | 变化 |
|---|---|---|---|
| coarse halo 交换 | 2 × all_gather（~3MB） | 同 | 不变 |
| window all_gather ×3 | 3 × 4.3MB | **0** | **删除** |
| L1 halo 交换 | 0 | 4 × all_gather（2 子步 × 2 侧，~0.5MB/侧） | 新增（微小） |
| coarse↔L1 transfer | 0（全帧本地） | 1 × all_reduce（2Q≈216 值） | 新增（微小） |
| L1 全帧 gather（壳层接口） | 0（复制） | 1 × ~18.8MB（方案 A 另 +2 × 18.8MB；方案 B 不 +） | 新增 |
| 壳层全帧 broadcast | 1 × ~18.8MB（既有） | 同（兼作下步 pre 场） | 不变 |
| 壳层 f_leaf all_gather + 力 all_reduce | 既有 | 同 | 不变 |

净效果：**L1 复制计算 O(W) 冗余 → O(1/W)**；window 13MB/步 → 删除；
新增 ~0.5-1MB（halo+reduce）+ 1 次全帧 gather（18.8MB，方案 A 为 3 次）。
壳层 all_gather（每子步 (27, n_leaf)，n_leaf≈10⁵ → ~10MB/子步 × 4 子步）成为
主导项（§7）。

## 5. 代码点清单

### 修改：`src/tensorlbm/octree_boundary/l1_block.py`（核心，~1.5-2 人日）

| 符号 | 改动 |
|---|---|
| `L1BlockDistributed.__init__` | 新增 `lo`/`hi`（coarse slab 边界）参数；计算 `lo_l1/hi_l1/nx_l1_local`；slab 张量替代全块；`xc_map` 改为 coarse slab 本地索引（`−lo+1`）；`ghost_mask_local` 排除 x-cut 列；`solid_q` 切片（halo 列 False）；box_links/l1_fine_links/观测与修正掩码全局构建后按 x 切片；`has_left/has_right` 标志 |
| `_sample_window` → `_sample_slab(coarse_slab)` | 采样源从 gathered window 改为本 rank coarse slab（含 halo 列） |
| `_fill_ghost` | 签名改为 `(coarse_old_slab, coarse_new_slab, alpha)`，本地 lerp + 采样 |
| `_advance` | 不变（collide/stream/freeze 逐格，slab 上直接可用） |
| **新增** `halo_exchange_l1(post_frozen)` | 2 个 all_gather 交换 x-cut 整列（含 z/y ghost），空 slab rank 发零 |
| `step()` | 新子步顺序（§3b）：fill → collide → freeze → exchange → stream → freeze；捕获 per-rank posts |
| `restrict_and_reflux` | 本地 restrict + 观测部分和 + (2Q,) all_reduce + 本地 reflux 应用 + 写 coarse slab；ledger 部分量 all_reduce |
| **新增** `gather_full_l1_field_chunked(...)` | 物理 slab → 全帧场（chunked all_gather 求和技巧，<3MB/msg） |
| **新增** `extract_slab_from_full(...)` | 全帧场 → 本 rank slab 列 |
| `initialize_from_window` → `initialize_from_slab` | 初始化采样源改本 rank coarse slab |
| `gather_window_chunked` / `write_window_back` | 删除或保留为兼容回退（Phase 1 参考实现） |

### 修改：`examples/octree_integrated_validate.py`（~0.5-1 人日）

- L1 块构造传入 `lo/hi`；主循环按 §1c 时序改写（阶段 B/C/D）；删除 window gather；
  维护 `l1_full_pre`（跨步）；新增 `--l1-shard aligned` 开关与 per-rank slab 统计
  打印（`nx_l1_local`、空 slab 数、各阶段计时）。
- 方案 B 时：调用方计算壳层观测部分和并传 `l1_post_transfer`。

### 修改：`src/tensorlbm/octree_boundary/distributed_stepping.py`（方案 A 零改动）

- 方案 A：**零改动**（全帧接口保持不变）。
- 方案 B（推荐，~0.5 人日）：`step_octree_shell_distributed` 增加可选参数
  `l1_post_transfer: KineticInterfaceTransfer | None`——若给定则跳过内部
  `observe_kinetic_interface_transfer(l1_post, observation_links)`（L730-745），
  直接用传入的全局 transfer。
- 可选 2b-2（~1 人日）：donor-request ghost fill（§8 优化路线），仿
  `sharding.py` 的 `remote_buf`/request 机制，把全帧 gather 也消除。

### 不动（验证过的复用面）

`geometry.py`、`stepping.py`（`build_ghost_plan`/`_fill_ghost_impl`/
`restrict_shell_to_block` 等全帧接口）、`sharding.py`、`static_block_amr.py`、
`amr_common.py`、`kinetic_flux_register.py`、壳层 stepper 主体。

### 新增：验证 harness（~1 人日）

- **lockstep 线程模拟**（扩展 `l1_d2_lockstep_sim.py` 的 fake-dist 模式）：
  16 线程 barrier-lockstep，验证 L1 halo 交换的集体对称性（rank 分歧即死锁，
  等价 TCCL）。
- **位一致性对拍**：1-4 rank 上 Phase 1（复制）vs Phase 2（分片）同初始条件跑
  N 步：L1 演化路径（ghost fill / collide / stream / freeze / halo）**位精确**一致；
  reflux 观测求和顺序变化 → `atol=1e-6`（float32 roundoff）级一致；Cd/mass drift
  对齐。

## 6. 正确性与确定性契约

- **位精确路径**（与 Phase 1 逐位一致）：L1 ghost fill（本 slab 采样值 = 窗口
  采样值，窗口 gather 是精确拷贝）、collide/stream/freeze（逐格确定性算子）、
  L1 halo 交换（精确拷贝）、全帧 gather 组装（每列单 owner + 零加法，IEEE
  `v+0=v` 位精确）、restrict（每 coarse 单元 8 子单元同 rank 同顺序）。
- **roundoff 级路径**（有先例：`sharding.py` docstring 明确"roundoff-level
  agreement, not bitwise"）：coarse↔L1 reflux 的观测求和顺序（per-rank 部分和 →
  all_reduce vs 全帧一次求和）、壳层 reflux 的 transfer 组装。验收用
  `atol/rtol ~ 1e-6`，不承诺位一致。
- **验收标准**：R6 sphere Re=100：Cd_mem ≈ 单卡参考 1.1093（blocked 1.654）；
  每级 reflux 残差 <1e-8；joint mass drift 对齐单卡；4 rank 对拍 Phase 1/2 差异
  在 roundoff 内；16 rank 强扩展曲线恢复下降。

## 7. 预期扩展

- **计算**：L1 每 rank 计算量 ∝ `nx_l1/W · ny_l1 · nz_l1 · 2 子步`；
  W=16 时约为 Phase 1 的 1/16（60×60×120 → 60×60×7.5/rank）。内存同样 1/W。
- **通信**：window 13MB/步删除；新增 ~0.5-1MB（halo + reduce）+ 1 次全帧 gather
  （18.8MB，方案 A 3 次）。与壳层每步 40-65MB 的 all_gather 相比占比小。
- **强扩展预期**：固定成本 ~4.4s 中 L1 复制计算（~2-3s）→ ~0.2-0.3s（W=16），
  window gather（~0.5-1s）→ ~0.2-0.4s；地板降至壳层主导的 ~1.5-2s，
  扩展延续到 W=8-16。
- **后续瓶颈（非本设计范围）**：壳层 `f_leaf` 全量 all_gather
  （O(W·n_leaf) 消息量）成为新地板 → 走 `sharding.py::shard_octree_shell`
  （fine_devices Morton 分片 + 跨片 request 计划）路线，把壳层也变为近邻通信。

## 8. 风险

| 风险 | 等级 | 说明与缓解 |
|---|---|---|
| slab stream wrap 伪影 | 高（须先验证） | roll 伪影恰落在 halo 列才正确；**顺序硬约束**：exchange 必须先于 stream，freeze 必须先于 exchange。缓解：位对拍单测（slab vs 全张量 stream 输出逐位一致）+ lockstep 模拟 |
| coarse halo 宽度（1）是否够 ghost fill | 中 | 依赖 box 严格内嵌（≥1 coarse margin）：`build_window_indices` 已校验；`plan_body_shell_box` pad≥1 保证。构建期断言 + 文档化 |
| reflux 求和顺序变化 | 低 | 从位一致降为 roundoff 级一致（有 sharding.py 先例）；验收 atol=1e-6；观测掩码静态切片避免每步 roll |
| 对齐分解负载失衡（小 W） | 中 | box 宽度 ≫ slab 宽度时仅边缘 2 rank 部分 slab + 外侧空 slab；W≥8 可接受；小 W 备选均衡 base+extra 分解（代价：ghost 采样 donor gather） |
| 全帧 gather 消息量（方案 A 3×18.8MB） | 中 | <3MB 分块沿用；若实测影响扩展 → 方案 B（观测 all_reduce）或 2b-2（donor-request，全帧 gather 也消除）；pre 场复用 broadcast 零成本 |
| 空 slab rank 的集体对称性 | 低 | 空 rank 参与全部 all_gather/all_reduce（发零）；lockstep 模拟覆盖 rank 分歧死锁场景 |
| TCCL 消息上限 | 低 | 全部新消息 <3MB（halo ~0.5MB；gather 分块；transfer all_reduce 单消息 ~1KB） |
| 确定性回归 | 低 | 演化路径位一致、reflux roundoff 级；对拍 harness 纳入 CI 式回归 |
| `l1_full_pre` 复用正确性 | 低 | 广播场即 Phase 1 `set_physical` 写入场，语义等价；单测校验跨步 pre 场一致性 |

## 9. 工作量

| 项 | 人日 |
|---|---|
| `l1_block.py` slab 重构（对齐分解、halo 交换、本地 restrict/reflux、gather/extract 助手） | 1.5-2 |
| 示例主循环集成 + 开关/计时/统计 | 0.5-1 |
| lockstep + 位对拍验证 harness | 1 |
| 扩展性基准与调优（分块、overlap：pre 场 gather 可与 coarse/L1 阶段异步重叠） | 0.5 |
| **核心合计** | **~3.5-5** |
| 方案 B（稀疏 l1_post 观测，stepper ~10 行） | +0.5 |
| 可选 2b-2（donor-request ghost fill，消除全帧 gather） | +1 |
| **含全部优化** | **~5-6.5** |

## 10. 实现步骤（建议顺序）

1. **锁步模拟先行**：在 `l1_d2_lockstep_sim.py` 模式上实现 slab + halo 交换的
   集体模式，验证 rank 对称性与顺序约束（无 GPU）。
2. `l1_block.py` 重构：slab 张量 + 掩码切片 + `_sample_slab`/`_fill_ghost` +
   `halo_exchange_l1` + `step()` 新顺序。
3. 位对拍：1-4 rank，Phase 1（复制）vs Phase 2（分片），断言演化路径位一致、
   reflux roundoff 级一致。
4. `restrict_and_reflux` 本地化 + transfer all_reduce + 本地应用；ledger 组装。
5. 主循环集成（方案 A 起步 → 测量 → 按需切方案 B）。
6. 扩展性基准：W ∈ {1,2,4,8,16}，输出 per-step 分解（coarse/L1 子步/壳层/通信），
   确认 4 卡以上恢复扩展。
7. （可选）donor-request ghost fill，把全帧 gather 也消除，对齐 `sharding.py` 路线。

## 11. 复用清单

- coarse slab 模式：`octree_integrated_validate.py` L352-364（`halo_exchange`）、
  L233-249（slab 布局与 `g − lo + 1` 索引约定）。
- `l1_block.py` 现有算子：`_sample_window`（改采样源）、`_advance`、
  `restrict_and_reflux`（改本地化）、`gather_window_chunked` 的求和组装技巧
  （改造成 `gather_full_l1_field_chunked`）。
- `kinetic_flux_register.py::build_kinetic_interface_links /
  observe_kinetic_interface_transfer / apply_face_local_reflux`（全局帧构建 + 切片；
  `apply_face_local_reflux` 的 roll 仅在构建期执行一次）。
- `distributed_stepping.py::step_octree_shell_distributed`（全帧接口零/微改动；
  既有 chunked broadcast 兼作下步 pre 场）。
- `sharding.py` 的 request/remote_buf 模式（可选 2b-2 的 donor-request 先例）。
- `l1_d2_lockstep_sim.py` 的 fake-dist barrier-lockstep（验证 harness 模板）。
