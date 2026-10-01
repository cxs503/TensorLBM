# 集成框架系统性低估 ~25% 根因分析

## 结论摘要

集成框架 Cd=0.80-0.83 vs 单卡 AMR Cd=1.1093 vs 参考 1.092，低估 ~25%。
经静态代码分析，根因按优先级排序如下：

| 优先级 | 根因 | 预估贡献 | 修复难度 |
|--------|------|----------|----------|
| **P0** | reflux=False + 分布式 stepper 缺失 fine_transfer 观测 | ~15-20% | 中 |
| **P1** | _LocalShellFacade 跨 rank 上游 donor 丢弃 (fp_up=0) | ~3-5% | 中 |
| **P2** | coarse_sparse ghost 采样（壳外均匀来流） | ~2-3% | 低 |
| **P3** | l1_old=l1_f 无时间插值 | ~1-2% | 低 |

合计预期改善：Cd 从 ~0.82 → ~1.09-1.13，恢复全部 25% 偏差。

---

## 1. 力计算链差异对比

### 1a. 粗场演化

| | 集成框架 | 单卡 AMR |
|---|---|---|
| **粗场布局** | 域分解 x-slab `(Q,nz,ny,nx_local+2)` + halo_exchange | 全场 L1 block `(Q,nz,ny,nx)` |
| **演化** | collide + halo + stream27_roll + halfway BB + far_field | amr.step(advance): collide + stream + sponge + freeze solid |
| **sponge** | y/z 面 sponge (width=16, strength=0.2) | 完整 sponge 层 |
| **solid 处理** | coarse halfway BB + shell BFL | freeze solid (L1 固体冻结) + shell BFL |

**代码位置**: `examples/octree_integrated_validate.py` L226-306 (coarse operators)
**单卡**: `examples/octree_sphere_validate.py` L418-421 (amr.step)

### 1b. Shell ghost 采样

| | 集成框架 | 单卡 AMR |
|---|---|---|
| **ghost 源场** | `coarse_sparse`：全场均匀来流 + 壳区真实值（GHOST_PAD=6 膨胀） | 实际演化 L1 场（全场真实尾迹/缺陷） |
| **时间插值** | `l1_old = l1_f = coarse_sparse` → lerp=恒定 | `l1_old_phys ≠ l1_f_phys` → lerp=时间演化 |

**代码位置**: `examples/octree_integrated_validate.py` L397-405
```python
coarse_sparse = eq27b(torch.ones(...), torch.full(..., u_in), ...)  # 均匀来流
coarse_sparse[:, shell_cells[:, 0], shell_cells[:, 1], shell_cells[:, 2]] = full_sc  # 仅壳区
l1_old = coarse_sparse  # ← 与 l1_f 相同！
l1_f = coarse_sparse    # ← 无时间插值
```

**单卡**: `examples/octree_sphere_validate.py` L419-423
```python
l1_old_phys = l1_fine[..., GHOST:-GHOST].clone()  # advance 前
l1_f_phys = l1_fine[..., GHOST:-GHOST]             # advance 后（不同！）
```

### 1c. reflux

| | 集成框架 | 单卡 AMR |
|---|---|---|
| **reflux** | **False** (显式禁用) | **True** (默认) |
| **l1_post** | **None** (未提供) | L1 post-collision 序列 |
| **fine_transfer** | **未观测** (分布式 stepper 缺失) | 每子步观测 + 累积 |
| **coarse_transfer** | N/A (reflux=False) | observe_kinetic_interface_transfer |
| **reflux 修正** | 无 | apply_face_local_reflux (守恒修正) |

**代码位置**: `examples/octree_integrated_validate.py` L406-411
```python
_ledger, local_mem, restricted, cells = step_octree_shell_distributed(
    octree, advance_shell, l1_old, l1_f,
    tau_coarse=tau_coarse, l1_post=None,          # ← 无 post-collision
    ghost_plan=None, bfl_fn=bfl_fn, rank=rank, world_size=world_size,
    reflux=False, interleave=args.interleave,     # ← 显式禁用！
)
```

**单卡**: `examples/octree_sphere_validate.py` L432-438
```python
shell_ledger = step_octree_shell(
    octree, shell_advance, l1_old_phys, l1_f_phys,
    tau_coarse=config1.tau_fine,
    l1_post=l1_posts if config1.reflux else None,  # ← reflux=True 时提供
    shell_level=1, ghost_plan=ghost_plan,
    bfl_fn=bfl_callback, force_ledger=ledger,
)
```

### 1d. BFL 力公式

| | 集成框架 | 单卡 AMR |
|---|---|---|
| **octree 对象** | `_LocalShellFacade` (局部切片) | 完整 `OctreeGrid` |
| **neighbor_table** | 跨 rank 源 → **-1 (丢弃)** | 完整全局表 |
| **remote_values** | **不暴露** | N/A (单卡无需) |
| **wall_velocity** | None (无移动壁 ramp) | bfl_ramp_wall_velocity (启动 ramp) |
| **force 累积** | 每 rank 局部 force → all_reduce | 全局 link_sink 组装 |

**代码位置**: `src/tensorlbm/octree_boundary/distributed_stepping.py` L166-206 (_LocalShellFacade)
```python
# 跨 rank 源映射为 -1：
nt = octree.neighbor_table[:, idx].cpu()  # (Q, n_local) 全局 enum
pos = torch.full((octree.n_leaf,), -1, dtype=torch.int64)
pos[idx] = torch.arange(self.n_leaf, dtype=torch.int64)
remapped = pos[nt.clamp(min=0)]  # 非本 rank → -1
sentinel = nt < 0
remapped[sentinel] = nt[sentinel]  # 保留 SHELL_OUTSIDE/SOLID/FANOUT
self.neighbor_table = remapped.to(device)
# 注意：不暴露 remote_values, remote_pos, fan_off, fan_len
```

**BFL 上游 donor 处理** (`src/tensorlbm/octree_boundary/bfl.py` L224-229):
```python
up = nt[od, idx]           # facade: 跨 rank → -1
fp_up = torch.zeros_like(fp_d)
valid = up >= 0             # -1 不 valid → fp_up 保持 0！
if bool(valid.any()):
    fp_up[valid] = f_prev[d, up[valid]].to(torch.float64)
# remote 分支 (L230-239) 永不触发：facade 不暴露 remote_values
```

---

## 2. 重点分析：reflux=False 是否导致质量/动量泄漏

### 2.1 reflux 机制原理

reflux（kinetic flux register）确保壳层-粗场界面的质量/动量守恒：

1. **fine_transfer 观测**（每子步）：`observe_shell_interface_transfer` 记录壳侧界面通量
   - outgoing: `sum(post_collision[d, i] * vol_i)` — 壳→粗的出射
   - incoming: `sum(ghost[d, i] * vol_i)` — 粗→壳的入射

2. **coarse_transfer 观测**（子步后）：`observe_kinetic_interface_transfer` 记录粗侧界面通量

3. **mismatch 投影**：`raw_mismatch = fine_transfer.net_outgoing - coarse_transfer.net_outgoing`
   - 投影到守恒矩（质量 + 3 动量分量）
   - `project_onto_conserved_moments` 丢弃非守恒动能模式

4. **修正应用**：`apply_face_local_reflux` 修正粗场外单元，使界面通量匹配

### 2.2 reflux=False 的后果

**无 reflux 时**：
- 壳层 restriction 覆盖粗场界面单元（正确）
- 但粗场**外部单元不修正** → 界面通量不匹配 → **动量泄漏**
- 粗场未"感受"壳层吸收的阻力 → 粗场动量偏高
- ghost fill 从偏高的粗场采样 → 壳层边界条件偏高
- 壳层速度偏高 → 壁面剪切/压力偏低 → **Cd 低估**

### 2.3 关键发现：分布式 stepper reflux 机制不完整

**即使设置 reflux=True 也会崩溃**，因为分布式 stepper 缺失 fine_transfer 观测：

`src/tensorlbm/octree_boundary/distributed_stepping.py`:
```python
# L358: fine_transfer 初始化为 None
fine_transfer = None

# L361-512: 子步循环中——没有 observe_shell_interface_transfer 调用！
for s in range(n_substeps):
    ...
    # 无 fine_transfer = observe_shell_interface_transfer(...)
    ...

# L555-569: reflux=True 路径
if reflux:
    coarse_transfer = observe_kinetic_interface_transfer(l1_post, coarse_links)
    l1_f, report = apply_face_local_reflux(
        l1_f, coarse_links, coarse_transfer, fine_transfer,  # ← None！
    )
    # → kinetic_flux_register.py L314:
    #   raw_mismatch = fine_transfer.net_outgoing - ...
    #   → AttributeError: 'NoneType' has no attribute 'net_outgoing'
```

**对比**：进程内分片 stepper (`step_octree_shell_sharded`) 正确实现了 fine_transfer 观测：
```python
# stepping.py L1355-1362:
if reflux:
    observed = _assemble_shell_transfer(shards, ...)  # ← 有！
    fine_transfer = observed if fine_transfer is None else fine_transfer + observed
```

**根因**：分布式 stepper 从 `step_octree_shell_sharded` 移植时，遗漏了 fine_transfer 观测逻辑。
`observe_shell_interface_transfer` 使用全局 `interface_links` + 局部 `post_collision`，索引不兼容
（全局 enum 索引局部张量会越界），需要用 all-gathered `full_pc` 适配。

### 2.4 reflux=True 的守恒效果

单卡 AMR 的 reflux 残差 ~1e-9（质量守恒），但**动量修正是非平凡的**：
- `apply_face_local_reflux` 将 fine-coarse 动量 mismatch 投影到 4 个守恒矩
- 修正粗场外部单元的动量，使界面动量交换闭合
- 这正是壳层阻力正确传递到远场的关键机制

---

## 3. BFL 力公式差异分析

### 3.1 集成框架的 bfl_apply_gather + facade

`_LocalShellFacade` 的 `neighbor_table` 重映射：
- 本 rank 源 → 局部列号 (≥0) ✓
- 跨 rank 源 → **-1** ✗（`bfl_apply_gather` 中 `valid = up >= 0` 为 False → `fp_up=0`）
- SHELL_OUTSIDE/SOLID/FANOUT → 保留 ✓

**Bouzidi 重建影响**：
- 线性分支 (q<0.5): `f_bc = 2q·fp_d + (1-2q)·fp_up` → `fp_up=0` 时丢失 `(1-2q)·fp_up` 项
- 二次分支 (q≥0.5): `f_bc = fp_d/(2q) + (2q-1)/(2q)·fp_opp` → **不使用 fp_up，不受影响**

**力影响**：`force = sum c_d·(fp_d + f_bc)`，线性分支的 corrupted f_bc 导致力偏差。
方向：丢失的 `(1-2q)·fp_up` 项为正（population 值），`c_d` 指向体内，
上游侧（c_d 指向 -x）的误差更大（更高压力/密度），净效果为**阻力低估**。

### 3.2 单卡完整 octree

单卡使用完整 `OctreeGrid`，`neighbor_table` 为全局表，所有上游 donor 可达。
`bfl_apply_gather` 的 `valid` 分支正确采集所有 donor。

### 3.3 进程内分片 stepper 的正确做法

`step_octree_shell_sharded` 使用 `shard.remote_values = shard.remote_buf`（跨分片值缓冲），
`bfl_apply_gather` 的 REMOTE 分支（L230-239）正确解析跨分片 donor。
`_LocalShellFacade` **不暴露** `remote_values`/`remote_pos`/`fan_off`/`fan_len`，
导致 REMOTE 分支永不触发。

### 3.4 FANOUT 问题

facade 的 `interface_fanout = octree.interface_fanout`（全局键），
但 `bfl_apply_gather` 中 `leaf_i = int(idx[pos])` 是**局部位置**（非全局 enum），
`octree.interface_fanout.get((leaf_i, od), [])` 查找失败 → 回退 `fp_up = fp_d`（错误）。

---

## 4. 最可疑根因 + 修复方案

### P0: 开启 reflux + 实现分布式 fine_transfer 观测

**最可疑根因**。reflux=False 直接导致壳层-粗场动量交换不守恒，
是 ~15-20% 低估的主因。且分布式 stepper 的 reflux 路径不完整（缺失 fine_transfer）。

**修复步骤**：

1. **在分布式 stepper 子步循环中添加 fine_transfer 观测**：
   ```python
   # distributed_stepping.py 子步循环中添加：
   if reflux:
       # 用 full_pc (all-gathered) 观测全局 fine-side transfer
       observed = observe_shell_interface_transfer(
           octree, gplan_fill, full_pc, ghost_vals,
       )
       fine_transfer = (
           observed if fine_transfer is None else fine_transfer + observed
       )
   ```
   注意：`observe_shell_interface_transfer` 使用 `octree.interface_links`（全局 enum）
   和 `full_pc`（全局 (Q,n_leaf)），索引兼容。但 `ghost_vals`/`plan` 是局部的，
   `incoming` 部分需要 all-reduce。

2. **集成脚本设置 reflux=True + 提供 l1_post**：
   ```python
   # octree_integrated_validate.py:
   # 粗场 post-collision 需 all-gather 为全场
   post_full = all_gather_coarse_post(post)  # 需实现
   _ledger, local_mem, restricted, cells = step_octree_shell_distributed(
       ..., l1_post=post_full, reflux=True, ...
   )
   ```

3. **适配 observe_shell_interface_transfer 的 incoming 部分**：
   outgoing 用 `full_pc`（每 rank 算全场，冗余但正确）；
   incoming 用局部 ghost_vals，需 all-reduce 累加。

**预期改善**: Cd ~0.82 → ~1.00-1.05

### P1: 修复 _LocalShellFacade 跨 rank 上游

**次要根因**。跨 rank BFL 边界链接的 fp_up=0，corrupts 线性分支。

**修复方案**（二选一）：

**方案 A（推荐）**：facade 暴露 remote_values
```python
class _LocalShellFacade:
    def __init__(self, octree, local_indices, device, full_pc=None):
        ...
        # 跨 rank 源映射为 REMOTE 而非 -1
        remote_mask = (nt >= 0) & (pos[nt.clamp(min=0)] < 0)
        remapped[remote_mask] = REMOTE
        # 暴露 remote_values = full_pc 的跨 rank 部分
        self.remote_values = full_pc  # (Q, n_leaf) 全局
        self.remote_pos = ...  # 需构建 REMOTE→全局列的映射
```

**方案 B（简单）**：BFL 用 full_pc + 完整 octree
```python
# distributed_stepping.py:
if bfl_fn is not None:
    # 用完整 octree + full_pc，每 rank 算全场力
    result = bfl_fn(octree, out_g, full_pc, gplan_fill, ghost_vals, ...)
    # force 需除以 world_size（每 rank 算了全场）
```

**预期改善**: Cd ~1.00-1.05 → ~1.05-1.10

### P2: 用全场粗场替代 coarse_sparse

**中等根因**。ghost fill 从 sparse 场采样，壳外为均匀来流。

**修复**：
```python
# all-gather 粗场为全场 (Q, nz, ny, nx)
coarse_full = all_gather_coarse(coarse_f)  # 需实现
l1_old = coarse_full  # 全场真实值
l1_f = coarse_full
```

**预期改善**: Cd ~1.05-1.10 → ~1.08-1.12

### P3: 添加时间插值

**次要根因**。l1_old=l1_f 无时间演化。

**修复**：
```python
coarse_old = coarse_f.clone()  # advance 前
# ... coarse evolve ...
coarse_new = coarse_f          # advance 后
l1_old = coarse_old_full       # 全场 pre-advance
l1_f = coarse_new_full         # 全场 post-advance
```

**预期改善**: Cd ~1.08-1.12 → ~1.09-1.13

---

## 5. 修复优先级总结

```
当前:  Cd ≈ 0.82  (低估 ~25%)
  │
  ├─ P0: 开启 reflux + 实现 fine_transfer 观测
  │     → Cd ≈ 1.00-1.05  (恢复 ~15-20%)
  │
  ├─ P1: 修复 facade 跨 rank 上游 (remote_values)
  │     → Cd ≈ 1.05-1.10  (恢复 ~3-5%)
  │
  ├─ P2: 全场粗场替代 coarse_sparse
  │     → Cd ≈ 1.08-1.12  (恢复 ~2-3%)
  │
  └─ P3: 添加时间插值 (l1_old ≠ l1_f)
        → Cd ≈ 1.09-1.13  (恢复 ~1-2%)

目标:  Cd ≈ 1.09-1.13  (匹配单卡 1.1093)
```

**最关键的一步是 P0**：reflux 是壳层-粗场动量守恒的核心机制，
单卡 AMR 的正确性（1.1093）正是依赖 reflux=True。
分布式 stepper 遗漏了 fine_transfer 观测，导致即使开启 reflux 也会崩溃，
这是需要首先修复的基础设施缺陷。
