# L1 中间块集成设计方案（coarse → L1 → 壳层叶）

> 目标：给两级集成框架（coarse 域分解 → 壳层叶 2x）加入 L1 中间细化层，
> 对齐单卡三级 AMR（coarse → L1 2x → 壳层叶 4x，Cd=1.1093），
> 修复两级框架系统性低估 34%（Cd≈0.82）的问题。
> 本文件为静态分析与设计，不涉及 GPU 运行。

## 0. 现状对照（关键差异点）

| 维度 | 集成两级（现状） | 单卡三级（参考，正确） |
|---|---|---|
| coarse 布局 | x-slab 域分解 (27,nz,ny,nx_local+2) + halo all_gather | 全场 (27,nz,ny,nx) |
| 中间层 | **无**（壳层直接嵌 coarse） | L1 块（2x，壳层+尾迹 bbox，带 ghost） |
| 壳层叶子 | d_max=1 → 2x coarse，2 子步/root | d_max=2 → 4x coarse（L1 的 2x），4 子步/root |
| 壳层 ghost 源 | `coarse_sparse`：壳区真实 + 壳外均匀来流（P2 缺陷） | 真实演化的 L1 全场 |
| 时间插值 | `l1_old = l1_f`（P3 缺陷） | `l1_old_phys ≠ l1_f_phys`（真实 lerp） |
| 壳层父级 tau | `tau_coarse` | `config1.tau_fine`（= L1 tau） |
| reflux | 已有（分布式 stepper 已补 fine_transfer 观测，脚本已传 l1_post） | L1↔壳 reflux + coarse↔L1 reflux（两级） |
| 壁面 | coarse halfway BB + 壳 BFL（双壁面） | L0/L1 freeze + 壳 BFL（单壁面） |

结论：P0-P3 项修复（reflux、facade、sparse、时间插值）在当前代码中已基本落地，
残余 25-34% 偏差主要来自**缺少 L1 中间细化层**——壳层父场是均匀来流污染的
稀疏 coarse，而不是真实演化的 2x 细化场。

## 1. 单卡 AMR 层级结构要点（static_block_amr.py）

- `StaticBlockAMRConfig`：`box`(coarse 坐标 BoxRegion)、`tau_coarse`、`ratio=2`、`ghost=1`、
  `reflux=True`、`ghost_interpolation`（injection/trilinear）。
- `StaticBlockAMR3D`：
  - `fine_f` = `_sample_parent_with_ghost(coarse_f, config)`：2x 采样 + `rescale_nonequilibrium`
    （tau_c→tau_f，spatial_ratio=2）。
  - `step()`：coarse 推进 1 次（substep=-1）→ 每个 fine 子步用
    `parent_t = lerp(coarse_old, coarse_new, s/2)` 做 **时间插值 ghost fill** → fine collide+stream
    → 2 子步后 `_restrict_physical()`（`restrict_populations_2to1` + neq rescale）→
    `apply_face_local_reflux`（coarse_interface_links 上 coarse_post 观测 + fine_post 子步累积）。
- `NestedStaticBlockAMR3D._advance_interface`：递归推进；`child.coarse_f = parent.fine_f`；
  tau 链 `convective_refined_tau`：tau_f = 0.5 + 2(tau_c − 0.5)；每接口 restrict+reflux。
- 壳层侧（stepping.py）：`build_ghost_plan(octree, l1_shape)` 三线性采样 L1；
  `_fill_ghost_impl` 带 `scale = tau_f/(2^lev·taus[0])·(1−1/tau_f)` 的 neq 松弛；
  `restrict_shell_to_block` 体积加权 restrict；`build_shell_coarse_links` 阶梯界面 + 固体剔除。

## 2. 集成脚本主循环现状（octree_integrated_validate.py L340-460）

每个 root step：
1. coarse 域分解演化：`post=collide_coarse` → `halo_exchange(post)` → `stream27_roll` →
   halfway BB → `far_field`（rank0 入口 / 末 rank 出口外推 / y·z 面重置 + sponge）。
2. 稀疏场构建：`shell_cells` = 壳 mask 膨胀 GHOST_PAD=6 的 coarse 全局单元集；
   每 rank 从自己 slab 取 `sc_local` → 分块 all_gather → `full_sc`（同法 `full_sc_post`）；
   `l1_post` = 均匀来流 eq + post 值；`coarse_sparse` = 均匀来流 eq + 壳区真实值；
   **`l1_old = l1_f = coarse_sparse`**（无时间插值）。
3. `step_octree_shell_distributed(octree, advance_shell, l1_old, l1_f, tau_coarse=tau_coarse,
   l1_post=l1_post, reflux=True, ...)` → `(ledger, local_mem, restricted, cells)`。
4. 回写：`coarse_f[:, sc_z, sc_y, sc_xx] = l1_f[:, ...]`（restriction + reflux patch 进 per-rank slab）。
5. 力：BFL MEM 每子步累积 → all_reduce。

## 3. L1 中间块设计

### 3a. L1 网格与域分解

**L1 只覆盖壳层 + 尾迹区域（body-fitted box），不是全场。**

- box 计算：复用单卡 `plan_body_shell_box(solid_mask, shell_margin, wake_cells, pad=wall_margin)`
  在 coarse 全局坐标算 `BoxRegion`；或直接取 `octree._shell_mask`（L1 坐标）的 bbox 映射回 coarse。
  默认建议对齐单卡：shell_margin=6、wake_cells=32、wall_margin=8（R6 球）。
- L1 形状：`(nz_l1, ny_l1, nx_l1) = ((z1−z0)·2, (y1−y0)·2, (x1−x0)·2)`；
  L1 张量带 1-cell ghost：`(27, nz_l1+2, ny_l1+2, nx_l1+2)`（与 StaticBlockAMR3D.fine_f 同布局）。
- **域分解：Phase 1 全场复制（每 rank 冗余演化，零新增通信）**。
  理由：① 现状已经在每 rank 复制全场 `l1_old/l1_f/l1_post/coarse_sparse`（(27,nz,ny,nx)），
  复制是既定模式；② 避免给 L1 2 子步再引入一套 x-slab halo 交换；③ 所有操作确定性
  （cumulant collide + torch.roll stream + freeze where）→ 各 rank 位一致，无需同步；
  ④ L1 box 很小（R6 默认约 90×44×44 ≈ 17 万 cell ≈ 18MB/rank），冗余计算量≈一次 coarse step。
  Phase 2（可选优化）：L1 x-slab 域分解，与 coarse 同构，每 L1 子步加 halo 交换。
- coarse 侧只需一个"窗口"：box + 1-cell 环（coarse 坐标），供 ghost 采样 / restrict 目标 /
  reflux 外环修正使用。窗口单元集 ⊂ 现有 `shell_cells` 膨胀区（要求 GHOST_PAD ≥ wall_margin+2，
  把 GHOST_PAD 从 6 提到 max(6, pad+2)）。

### 3b. L1 演化（collide + stream，host 从 coarse 插值，边界处理）

- L1 是**跨 root step 持久**的层级（不是每步从 coarse 重建）；每 root step 推进 2 子步
  （ratio=2），与单卡 `amr.step()` 的 L1 块阶段完全同构。
- 初始化（仅 step 1）：`_sample_parent_with_ghost` 语义——从 coarse 窗口 2x 采样 +
  `rescale_nonequilibrium(tau_c→tau_l1, spatial_ratio=2)`。
- 每子步 s ∈ {0,1}：
  1. ghost fill：`parent_t = lerp(coarse_window_old, coarse_window_new, s/2)`（**修复 P3 时间插值**），
     trilinear/injection（默认 injection，对齐单卡）+ neq rescale，写入 L1 ghost 层
     （直接复用 `_GhostSamplingPlan`/`_fill_ghost` 逻辑，坐标从 coarse box 换算）。
  2. 演化：`l1_post_s = advance_l1(l1_f, tau_l1)` —— cumulant D3Q27 collide +
     `stream27_roll`（带 ghost 全张量流，ghost 已被 coarse 填充，语义与 StaticBlockAMR3D 相同）+
     **freeze 固体掩码**（`torch.where(l1_solid_q, before, collided)`，掩码来自
     `octree._solid` 映射到 L1 with-ghost，同单卡 l1_solid_q 构造）。
  3. 捕获 `l1_post_s`（物理切片）供壳层 reflux 观测（列表，同单卡 `l1_posts`）。
- 边界处理：L1 box 六面均为 AMR 界面——ghost fill 从 coarse 窗口时间插值；
  **不施加 far-field / sponge / BB**（这些属于外域 coarse；box 严格内嵌，满足
  `StaticBlockAMRConfig` 的 "strictly interior with coarse-cell margin" 约束）。
- 壁面：coarse 的 halfway BB 区域（物体附近）每步被 L1 restrict 覆盖，实际近乎惰性；
  L1 freeze 阻止 L1 穿过固体；壁面力仍由壳层 BFL 独享（对齐单卡 freeze+BFL 语义）。
  可选建议：启用 L1 后加 `--no-coarse-bb` 完全对齐单卡（消除三重壁面处理的不确定性）。

### 3c. 壳层叶子 host 从 coarse 改到 L1

- `build_octree_shell(L1_shape, center_l1, radius_l1, bl_thickness_cells, d_max=2, ...)`：
  - `center_l1 = (cx·2 − box.x0·2, cy·2 − box.y0·2, cz·2 − box.z0·2)`（L1 物理坐标）；
  - `radius_l1 = radius_coarse · 2`；
  - `d_max=2` → 叶子 4x coarse，`n_substeps=4`/root；`leaf_host_cell` 自动落在 L1 坐标。
- `octree.f_leaf` 初始化：`l1_fine[:, host+GHOST]` 采样（同单卡 L254-261）。
- `ghost_plan = build_ghost_plan(octree, L1_shape)`——三线性采样**真实 L1 场**
  （修复 P2：不再有均匀来流污染；solid_fallback 保留）。
- `step_octree_shell_distributed(octree, advance_shell, l1_old_phys, l1_f_phys,
  tau_coarse=tau_l1, l1_post=l1_posts, reflux=True, ...)`：
  - `tau_coarse=tau_l1 = taus[1]`，`tau_shell = taus[2]`（d_max=2 链）；
  - `l1_old/l1_f` = L1 块推进前后物理切片（真实两时刻，lerp 正确）；
  - `l1_post` = 2 个 L1 post 状态列表（分布式 stepper L611-626 已支持列表）；
  - `restrict_shell_to_block` 的 `taus=[tau_l1, tau_shell]` 自动正确（父级 tau 变 L1）。
  - 该函数**无需改动**（octree.meta["shape"] = L1 shape → ghost plan / restrict / reflux 全自动适配）。

### 3d. restrict/reflux 从 L1 回 coarse

新增 `step_l1_block_distributed`（或 `L1Block` 类），完成块阶段 + L1→coarse 耦合：

1. **L1→coarse restrict**：`restricted = restrict_populations_2to1(l1_f_phys)`（2×2×2 均值）
   + `rescale_nonequilibrium(tau_l1→tau_coarse, spatial_ratio=1/2)`；
   `coarse_window_new[box] = restricted`。
2. **coarse 侧 reflux（新接口）**：
   - coarse transfer：`observe_kinetic_interface_transfer(coarse_window_post, box_links)`，
     `box_links = build_kinetic_interface_links(box_mask, q=27)`（box 边界 crossing links）；
   - fine transfer：从 L1 侧观测——L1 的 post 状态在 box 边界链接上的通量
     （可复用 `observe_kinetic_interface_transfer` 对 `l1_post_s` + L1 侧 links 观测，
     或按 `_advance_interface` 中 fine_interface_links 语义实现）；
   - `apply_face_local_reflux(coarse_window_new, box_links, coarse_transfer, fine_transfer,
     maximum_correction_fraction=0.2, stencil="exterior_cells")`——修正 box 外 1-cell 环。
3. **回写**：`coarse_window_new`（box+环，约 40k cell）按 x 归属写回各 rank 自己的
   `coarse_f` slab（`global_x − lo + 1` 偏移，同现有 sc_in 回写模式）；
   环外列由下一次 `halo_exchange` 自动刷新。

### 3e. 与现有 coarse 域分解的耦合

- coarse 流水线（collide / halo_exchange / stream / BB / far_field / sponge）**完全不变**。
- 新增 3 个耦合点：
  1. `coarse_old = coarse_f.clone()`（root step 前，每 rank slab 克隆，~3MB/rank）；
  2. 窗口采集：把 `shell_cells` 稀疏采集替换/扩展为 box 窗口单元集，分块 all_gather
     三个时间点 `coarse_window_old / _new / _post`（可合并为一次调用减少消息数）；
  3. 窗口 patch 回写（3d 第 3 步）替代现有 sc_in 回写。
- 力链不变：`local_mem` 累积 + all_reduce；`dynamic_area` 改用新 `radius_leaf`
  （见风险 3）。

### 时序（每 root step）

```
coarse_old = clone(coarse_f)
coarse 演化（现有 1 步）
窗口采集 old/new/post（chunked all_gather ×3）
L1 块阶段：2 子步（ghost←lerp(coarse_old,coarse_new) 窗口；collide+stream+freeze）→ l1_posts
壳层阶段：step_octree_shell_distributed(L1 场, tau_l1, d_max=2, 4 子步, reflux) → l1_f 含 restrict+reflux
L1→coarse：restrict box + 界面 reflux → coarse_window_new
窗口写回 per-rank coarse_f
力累积 + 报告（不变）
```

## 4. 工程量评估

### 新增函数

| 函数 | 位置 | 规模 | 复用 |
|---|---|---|---|
| `step_l1_block_distributed(...)` 或 `L1Block` 类 | 新 `src/tensorlbm/octree_boundary/l1_block.py`（或并入 distributed_stepping.py） | ~150-250 行 | `restrict_populations_2to1`、`rescale_nonequilibrium`、`build_kinetic_interface_links`、`observe_kinetic_interface_transfer`、`apply_face_local_reflux`（全部公开）；ghost 采样逻辑从 `_GhostSamplingPlan`/`_fill_ghost` 复制适配（~60 行）或提为公共助手 |
| 窗口采集助手（old/new/post 三时间点 chunked all_gather） | 同上 | ~40 行 | 复用现有 sc chunk 模式 |
| L1 box 规划助手（box → 窗口单元集 → GHOST_PAD 校验） | 同上 | ~30 行 | `plan_body_shell_box` |

### 修改函数

| 函数 | 改动 |
|---|---|
| `octree_integrated_validate.py` | ① 几何：octree 改建于 L1 shape/center/radius（d_max=2）；② tau 链 `taus[1]=tau_l1`、`taus[2]=tau_shell`；③ 主循环插入 L1 块阶段 + 双回写；④ `dynamic_area` 的 radius_leaf 公式；⑤ `--d-max 2` 默认/说明 |
| `distributed_stepping.py` | 基本零改动（`step_octree_shell_distributed` 已参数化）；可选把 ghost-fill 采样提为公共助手供 L1 复用 |
| `static_block_amr.py` | 零改动（或仅将 `_sample_parent_with_ghost`/`_GhostSamplingPlan` 提为公共 API） |
| `geometry.py` / `stepping.py` | 零改动 |

### 工作量

- 阶段 1（最小可行）：~0.5-1 人日新代码 + 0.5 人日脚本改造 + 1-2 人日验证。
- 阶段 2（L1 域分解 / 全递归合并）：各 ~1-2 人日。

## 5. 风险评估

| 风险 | 等级 | 说明与缓解 |
|---|---|---|
| TCCL 消息大小 | 低-中 | n_leaf 随 d_max=2 增大（壁面邻近产生 depth-2 叶，n_leaf 约 ×2-4）；每子步 all_gather (27,n_leaf) 已分块 <3MB；窗口采集 (27×~40k×4B≈4.3MB) 需沿用分块；restricted broadcast 同理。**所有新增通信必须沿用 3MB/msg 分块模式，禁止整张 all_gather/broadcast** |
| n_leaf 变化 / 负载均衡 | 中 | depth-2 叶集中在壁面 → 连续 Morton 分片负载不均加剧；`--interleave` 轮转已缓解（保持开启）；BFL 链接数随叶数增长，力观测开销上升；`n_leaf_l1/n_leaf_l2` 需打印监控 |
| 力面积公式 | 中 | `radius_leaf` 从 `R/2^-1` 变为 `R·2/2^-2 = 8R`（16x 面积）；`dt_leaf=dx_leaf`；MEM 力与 dynamic_area 同尺度缩放，Cd 比值不变——但**必须**同步更新，否则 Cd 差 16x |
| 双 reflux 界面 | 中 | coarse↔L1 与 L1↔壳 两级 reflux 叠加；验收标准：每级残差 <1e-8、joint mass drift 与单卡同级（~1e-9/step）；L1 环外 coarse 值由 halo_exchange 刷新，回写窗口必须含 1-cell 环避免 staleness |
| L1 冗余计算 | 低（Phase 1） | 每 rank 复制 L1（~18MB/rank，计算≈一次 coarse step）；16 rank 时总 FLOPs 浪费 ≤2x；Phase 2 改 x-slab 分解消除 |
| 确定性 | 低 | 全部算子确定性（cumulant、roll、where、分块 gather 求和顺序固定）→ 各 rank 位一致，无需同步；验证时对拍 rank 间 L1 张量 |
| 三重壁面处理 | 低 | coarse BB 区域被 L1 restrict 每步覆盖（近乎惰性）；建议 `--no-coarse-bb` 对齐单卡 freeze+BFL 语义，消除歧义 |
| 壳层 ghost 三线性越界 | 低 | 壳严格内嵌于 L1 box（shell_margin>0）；`build_ghost_plan` clamp 兜底；solid_fallback 保留 |
| 时间插值窗口一致性 | 低 | old/new/post 三个窗口必须用**同一单元集**采集（同一索引数组），避免采样错位 |

## 6. 最小可行实现步骤（分阶段）

### 阶段 1：L1 覆盖壳层区（核心修复，预期恢复 25-34% 偏差）

1. **几何**：`plan_body_shell_box(solid, shell_margin=6, wake_cells=32, pad=wall_margin=8)` →
   `box`；`build_octree_shell(L1_shape, center_l1, radius_l1, bl, d_max=2)`；
   GHOST_PAD 提到 `max(6, pad+2)`；窗口单元集 = box+1 环。
2. **窗口采集**：`coarse_window_old/new/post` 三时间点分块 all_gather
   （复用 sc chunk 模式；`coarse_old = coarse_f.clone()` 于 coarse 演化前）。
3. **L1 块阶段**：新增 `step_l1_block_distributed`——L1 持久张量 + 2 子步
   （ghost←lerp 窗口 + collide/stream/freeze）+ `l1_posts` 捕获。
4. **壳层接入**：`step_octree_shell_distributed(octree, advance_shell, l1_old_phys, l1_f_phys,
   tau_coarse=tau_l1, l1_post=l1_posts, reflux=True, interleave=True)`。
5. **L1→coarse 回写**：restrict box + box 界面 reflux + 窗口 patch 写回 per-rank slab。
6. **力与输出**：`radius_leaf` 公式更新（`R·2·2^d_max`）；`n_leaf_l1/l2` 监控；
   `--no-coarse-bb` 建议。
7. **验收**：R6 sphere Re=100，Cd_mem vs 单卡 1.1093（blocked-ref 1.654）；每级 reflux
   残差 <1e-8；mass drift 对齐单卡；rank 间 L1 张量位一致对拍。

### 阶段 2：全场 / 性能优化（可选）

- 2.1 L1 x-slab 域分解：与 coarse 同构，每 L1 子步 halo 交换，消除冗余计算；
   ghost fill 改为从 coarse 的 rank-local 值 + 邻居 halo 采样（不再需要窗口 gather）。
- 2.2 全递归合并：把壳层作为 L1 的 level-2 内嵌（`NestedStaticBlockAMR3D` 式），
   L1 与壳层 reflux 合并为逐接口递归，消除两级间时序近似（对齐 `_advance_interface`）。
- 2.3 若 Phase 1 验证中 coarse 端 BB 与 L1 freeze 交互异常，再评估 coarse 壁面统一为 freeze。

## 7. 复用清单（避免重复造轮子）

- `src/tensorlbm/fixed_nested_transfer.py::restrict_populations_2to1`
- `src/tensorlbm/amr_population_transfer.py::rescale_nonequilibrium`
- `src/tensorlbm/kinetic_flux_register.py::build_kinetic_interface_links / observe_kinetic_interface_transfer / apply_face_local_reflux`
- `src/tensorlbm/static_block_amr.py::_sample_parent_with_ghost / _GhostSamplingPlan / _fill_ghost`（建议提为公共 API）
- `src/tensorlbm/octree_boundary/stepping.py::build_ghost_plan / _fill_ghost_impl / restrict_shell_to_block / build_shell_coarse_links / _tau_chain`
- `src/tensorlbm/octree_boundary/distributed_stepping.py::step_octree_shell_distributed / split_leaf_bounds / interleaved_leaf_indices`
- `src/tensorlbm/amr_shell_planning.py::plan_body_shell_box`
