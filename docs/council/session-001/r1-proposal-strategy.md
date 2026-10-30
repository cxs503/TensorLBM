author-model: qwen3.7-plus

# 议事提案 r1 · 架构视角：跨库耦合阶梯、物理战役路线、长期技术债

> session-001 · 提案人槽位 · 视角：架构
> 纪律：只写本槽位文件；数字引用 00_context.md 已核实记录；不确定处标注 [待验]。

---

## 提案总览

| # | 方向 | 一句话目标 | 峰值卡位 | 预算量级 | 阶梯范式 |
|---|------|-----------|---------|---------|---------|
| A | 船舶阻力物理战役楔形 | 从已验证 2D Couette 沿 S0→S4 推进到 3D 船型阻力，端到端证明 SDAA 栈可用于阻力预报 | 12 | ~10 人日 | wedge S0→S4 |
| B | 跨库耦合阶梯 LBM↔FVM↔BEM | 按 S0→S3 建立 LBM-FVM-BEM 耦合框架，为多物理场打基础 | 8 | ~8 人日 | SI S0→S3 |
| C | 技术债清偿：Triton 消除 + 跨卡框架固化 | 将 5 个 triton 依赖模块改写纯 torch，固化跨卡域分解 | 4 | ~5 人日 | 内部重构 |

**32 卡分配总表（按阶段）**

| 阶段 | 时间窗 | A 阻力战役 | B 耦合阶梯 | C 技术债 | 扫荡线 | 值班 | 合计 |
|------|--------|-----------|-----------|---------|--------|------|------|
| P1 | day 1-2 | 8 (S0-S1) | 4 (S0) | 4 (M1-M2) | 10 | 2 | 28 |
| P2 | day 3-5 | 12 (S2-S3) | 4 (S1) | 4 (M3-M4) | 8 | 2 | 30 |
| P3 | day 6-10 | 12 (S4) | 8 (S2-S3) | 0 | 8 | 2 | 30 |

> 说明：三方向按阶段错峰，同一阶段内并行；扫荡线随 in-flight 消化逐步释放卡位回补。峰值不超过 30 卡，留 2 卡弹性。

---

## 方向 A：船舶阻力物理战役楔形路线（S0→S4）

### 目标

从已验证的 2D Couette（L2=1.80e-4，跨卡逐位复现）出发，沿 wedge S0→S4 阶梯推进到 3D 船型阻力，端到端证明 SDAA 纯 torch 栈可用于船舶阻力预报，每阶段配论文级进展报告。

### Mission 粒度拆解

#### S0 — 2D 基准扩展（锚点冻结 + 验证）

| 项 | 内容 |
|----|------|
| repo | TensorLBM (`src/tensorlbm/`, `cases/`, `benchmarks/`) |
| 任务 | ① 冻结 2D 圆柱绕流文献锚点：Re=40 Cd≈1.50 (Tanaka)、Re=100 Cd≈1.33 (Schlichting)、Re=1000 Cd≈0.47 (Achenbach)，写 anchors.md（零手抄，每个值标出处坐标）；② SDAA 上卡跑 2D 圆柱 D2Q9，3 个 Re 各 1 卡；③ 跨卡验证（5→0 逐位复现，复用 m10 范式） |
| 机器判据 | Cd 相对误差 <5%；L2 <1e-3；跨卡逐位复现 PASS |
| 预算 | 2 impl + 1 ver，4 卡，1 天 |
| 前置 | 无（S0 是起点，Couette 已通） |

#### S1 — 3D 球体绕流（单物理 3D 跃迁）

| 项 | 内容 |
|----|------|
| repo | TensorLBM (`solver3d.py`, `obstacles.py`, `boundaries3d.py`) |
| 任务 | ① D3Q19 球体绕流 Re=10/100/1000，单卡起步→4 卡域分解；② Cd 对标 Schlichting 表；③ 体素化球体用解析公式（非 STL），排除几何管线干扰 |
| 机器判据 | Cd <8% vs Schlichting；60k 步无 NaN；4 卡 vs 单卡 L2 <1e-6 |
| 预算 | 2 impl + 2 ver，8 卡（含参数扫描），2 天 |
| 前置 | S0 PASS |

#### S2 — 3D Wigley 船型阻力（首次船型）

| 项 | 内容 |
|----|------|
| repo | TensorLBM (`suboff_cmk_kbc_runner.py` 或等价 3D runner, `geometry_voxel.py`) |
| 任务 | ① Wigley 标准船型解析几何体素化（无需 STL）；② D3Q19 阻力计算 Fr=0.16/0.30；③ Cf 对标 ITTC 1957 平板线，Cp 对标实验；④ 压力分布可视化 |
| 机器判据 | RT 总阻力 <10% vs 实验；Cf <5% vs ITTC 1957；波形定性吻合 |
| 预算 | 3 impl + 2 ver + 1 doc，12 卡，3 天 |
| 前置 | S1 PASS；解析几何体素化路径在 SDAA 验证 |

#### S3 — SUBOFF AFF-8 阻力（已有参考数据）

| 项 | 内容 |
|----|------|
| repo | TensorLBM (`cases/suboff_hybrid/`, `docs/suboff_*`) |
| 任务 | ① SUBOFF 全附体 AFF-8 几何体素化（解析+STL 混合）；② D3Q19 阻力 Re≈1.2e7（缩比）；③ Ct 对标 ITTC / docs/suboff_reference_data_r1.md 已有参考值；④ 8 卡域分解生产跑 |
| 机器判据 | Ct <10% vs ITTC；60k 步无 NaN；GLUPS 报告 |
| 预算 | 2 impl + 2 ver + 1 mod + 1 doc，8 卡，3 天 |
| 前置 | S2 PASS；STL 体素化路径在 SDAA 验证 [待验：当前 STL ingest 仅 CPU] |

#### S4 — 自由表面船型阻力（终极楔形）

| 项 | 内容 |
|----|------|
| repo | TensorLBM (`free_surface/`, `multiphase/`) |
| 任务 | ① 自由表面 LBM（Shan-Chen 或 He-Luo）+ 船型绕流耦合；② KVLCC2 或 Series 60 阻力+波形；③ 对标 Tokyo 2015 CFD Workshop 实验数据 |
| 机器判据 | RT <15% vs 实验；波形纵切面定性吻合；无质量泄漏 |
| 预算 | 3 impl + 2 ver + 1 mod + 1 doc，12 卡，5 天 |
| 前置 | S3 PASS；自由表面模块在 SDAA 验证 [待验：docs/free_surface_* 显示模块存在但未在 SDAA 跑过] |

### 卡位分配

- P1（day 1-2）：S0+S1 共 8 卡（4 卡 S0 圆柱 + 4 卡 S1 球体起步）
- P2（day 3-5）：S2+S3 共 12 卡（8 卡 Wigley + 4 卡 SUBOFF 预热）
- P3（day 6-10）：S4 共 12 卡（自由表面生产跑）

### 风险与依赖

1. **STL 体素化路径**：docs 显示 `stl_ingest` 存在但标注 CPU-only；S3+ 需要 GPU 体素化或 CPU 预处理+GPU 求解分离。缓解：S2 用解析几何绕过，S3 前先做 1 个 mission 验证 STL→体素 SDAA 路径。
2. **自由表面模块**：docs/free_surface_* 系列文档丰富但均为设计/审计文档，无 SDAA 上卡记录。缓解：S4 前先做 smoke test（1 卡，2D 溃坝 vs M&M 1952 锚点，已有 docs/MP_DAMBREAK_S0_ANCHORS_MIRROR.md）。
3. **D3Q19 跨卡域分解**：m10 仅验证 2D 跨卡；3D 跨卡未验证。缓解：S1 球体即做 4 卡域分解验证。

### 放弃条件

- **S1 球体 Cd 误差 >15%** 且 3 天内无法收敛 → 暂停 S2+，回到 S0 扩展更多 2D 案例（圆柱不同 Re、后台阶流），积累调试经验。
- **S3 STL 体素化在 SDAA 上无法工作** 且 CPU 预处理耗时 >1h → 降级为解析近似船型（Wigley 延伸），S4 推迟。
- **S4 自由表面 2D smoke test 失败**（质量泄漏 >1%/1000 步）→ S4 推迟，先做单相 S3 深化（多 Fr、网格收敛）。

---

## 方向 B：跨库耦合阶梯 LBM↔FVM↔BEM（SI S0→S3）

### 目标

按 Staged Integration S0→S3 建立 LBM-FVM-BEM 三求解器耦合框架，复用已有 coupling contract（docs/tensordem_coupling_contract.md, docs/ecosystem-integration.md），为多物理场（阻力+声学+结构）打基础。

### Mission 粒度拆解

#### S0 — FVM/BEM standalone SDAA smoke test + 兼容性普查

| 项 | 内容 |
|----|------|
| repo | TensorFVM (`/root/tensor-ecosystem/TensorFVM/src/tensorfvm/`), TensorBEM (`/root/tensor-ecosystem/TensorBEM/src/tensorbem/`) |
| 任务 | ① 复用 m11 范式：普查 TensorFVM 和 TensorBEM 的模块，分类纯 torch(A) / triton 依赖(B) / 其他；② 各跑通 1 个 benchmark（FVM: lid-driven cavity Re=1000；BEM: 球体声散射解析解）；③ SDAA 上卡验证 |
| 机器判据 | 普查报告完成（A/B 分类）；各 benchmark L2 <1e-2；SDAA 上卡 PASS |
| 预算 | 2 impl + 1 ver + 1 arch，4 卡，1.5 天 |
| 前置 | 无 |

#### S1 — LBM→FVM 单向耦合（远场→近壁）

| 项 | 内容 |
|----|------|
| repo | TensorPhysics (coupling driver [待建]), TensorLBM, TensorFVM |
| 任务 | ① 在 TensorPhysics 中实现 coupling driver 原型（参考 tensordem_coupling_contract 的 exchange packet 格式）；② LBM 跑远场 → 导出边界场 → FVM 接收为入口边界 → FVM 跑近壁；③ 验证界面通量守恒 |
| 机器判据 | 界面通量守恒 <1%；耦合场 vs standalone L2 <5%；时间戳同步 PASS |
| 预算 | 2 impl + 1 ver + 1 arch，4 卡，2 天 |
| 前置 | S0 PASS（FVM 在 SDAA 可跑） |

#### S2 — LBM↔FVM 双向迭代耦合（Schwarz 交替）

| 项 | 内容 |
|----|------|
| repo | TensorPhysics, TensorLBM, TensorFVM |
| 任务 | ① 实现 Schwarz 交替迭代（LBM↔FVM 界面交换，每步迭代至收敛）；② 验证迭代收敛性；③ 对比单向耦合，量化双向修正量 |
| 机器判据 | 迭代收敛 <10 步；界面残差 <1e-4；双向 vs 单向差异 <3% |
| 预算 | 2 impl + 1 ver + 1 arch，6 卡，3 天 |
| 前置 | S1 PASS |

#### S3 — LBM+FVM+BEM 三求解器耦合（阻力+声学）

| 项 | 内容 |
|----|------|
| repo | TensorPhysics, TensorLBM, TensorFVM, TensorBEM |
| 任务 | ① LBM 远场流 → FVM 近壁压力 → BEM 声学辐射（ hull 表面压力 → 远场声压）；② 三场时间同步；③ 对标球体声散射解析解 |
| 机器判据 | 声压级 vs 解析解 <10%；三场时间同步 PASS；耦合框架可复现 |
| 预算 | 3 impl + 2 ver + 1 arch + 1 doc，8 卡，4 天 |
| 前置 | S2 PASS；BEM S0 PASS |

### 卡位分配

- P1（day 1-2）：S0 共 4 卡（FVM 2 卡 + BEM 2 卡）
- P2（day 3-5）：S1 共 4 卡
- P3（day 6-10）：S2+S3 共 8 卡（S2 用 4 卡，S3 用 4 卡，错峰）

### 风险与依赖

1. **TensorFVM/BEM triton 依赖未知**：m11 仅普查了 TensorLBM。如果 FVM/BEM 有大量 triton 依赖，S0 普查会发现，需先走方向 C 的消除路径。缓解：S0 普查是第一步，结果决定后续。
2. **TensorPhysics coupling driver 不存在**：docs/ecosystem-integration.md 定义了接口契约但无实现。缓解：S1 第一个 mission 就是建 driver 原型，预算已含。
3. **跨库张量传递**：不同库可能用不同 tensor layout。缓解：exchange packet 用显式序列化（JSON schema 已有），不依赖内存共享。

### 放弃条件

- **S0 普查发现 TensorFVM 或 TensorBEM 有 >5 个 triton 硬依赖且无法用纯 torch 绕过** → 降级为 LBM+DEM 耦合（已有 tensordem_coupling_contract.md，DEM 已有 2D 耦合代码 `ice_coupling_2d.py`），跳过 FVM/BEM。
- **S1 coupling driver 原型 2 天内无法实现基本交换** → 暂停，先做 LBM 内部多域分解（方向 C 的 M4），积累跨域经验后再回来。
- **S2 Schwarz 迭代不收敛**（>50 步未收敛）→ 降级为松耦合（每 N 步交换一次，不迭代），标注精度损失。

---

## 方向 C：技术债清偿 — Triton 消除 + 跨卡框架固化

### 目标

将 TensorLBM 的 5 个 triton 依赖模块改写为纯 torch 等价实现，消除 SDAA 上 triton 解释模式瓶颈；固化跨卡域分解框架，使 32 卡可线性扩展。

### Mission 粒度拆解

#### M1 — Triton 模块功能边界审计

| 项 | 内容 |
|----|------|
| repo | TensorLBM (`src/tensorlbm/backends/triton_backend.py`, `triton_fused_obstacle` 等) |
| 任务 | ① 列出 5 个 triton 依赖模块的完整清单（模块路径、功能、输入/输出 tensor 形状）；② 对每个模块评估纯 torch 替代可行性（是否依赖 tl.arange power-of-2 等约束）；③ 输出替代方案设计文档 |
| 机器判据 | 5/5 模块审计完成；每个模块有明确的替代方案（直接替代/需重构/不可替代）；文档提交 |
| 预算 | 1 arch + 1 impl，0 卡（纯分析），0.5 天 |
| 前置 | 无 |

#### M2 — triton_fused_obstacle 纯 torch 等价（最大模块）

| 项 | 内容 |
|----|------|
| repo | TensorLBM (`src/tensorlbm/backends/torch_5op_step.py` 已有 5-op 纯 torch 路径，需扩展覆盖 obstacle 融合) |
| 任务 | ① 实现 `torch_fused_obstacle` 纯 torch 等价；② 与 triton interpret 模式逐值对比；③ 性能 benchmark（SDAA vs interpret） |
| 机器判据 | L2 vs triton interpret <1e-6；SDAA 性能 ≥ interpret 模式 10×；回归测试 PASS |
| 预算 | 2 impl + 1 ver，2 卡，2 天 |
| 前置 | M1 完成 |

#### M3 — 其余 4 个 triton 模块纯 torch 等价

| 项 | 内容 |
|----|------|
| repo | TensorLBM |
| 任务 | ① 逐个实现纯 torch 替代；② 全量回归测试（15 verified 基准全 <3%）；③ 更新 `backends/__init__.py` 路由，默认走纯 torch |
| 机器判据 | 5/5 模块纯 torch 路径通过；15 verified 基准回归 PASS；无 triton import 依赖 |
| 预算 | 2 impl + 1 ver，2 卡，2 天 |
| 前置 | M2 完成 |

#### M4 — 跨卡域分解框架固化

| 项 | 内容 |
|----|------|
| repo | TensorLBM (`src/tensorlbm/hardware.py`, `backends/`, `triton_distributed_notes.md` 已有 halo exchange 设计) |
| 任务 | ① 纯 torch halo exchange 实现（参考 triton_distributed_notes.md 的 Q=19 面传输设计，但用 torch.roll/tensor copy）；② 多卡域分解 API（auto-shard by Nx×Ny×Nz）；③ 4 卡线性加速比验证 |
| 机器判据 | 4 卡域分解 L2 vs 单卡 <1e-6；线性加速比 >0.7；API 文档+示例 |
| 预算 | 1 arch + 2 impl + 1 ver + 1 infra，4 卡，2 天 |
| 前置 | M3 完成（确保无 triton 依赖干扰） |

### 卡位分配

- P1（day 1-2）：M1+M2 共 4 卡（M1 用 0 卡，M2 用 2 卡 + 2 卡验证）
- P2（day 3-5）：M3+M4 共 4 卡（M3 用 2 卡，M4 用 4 卡，错峰）
- P3：无（技术债清偿应在 P2 结束）

### 风险与依赖

1. **纯 torch 性能**：triton 融合内核在 NVIDIA 上有优势，但 SDAA 上 triton 只有 interpret 模式（慢），纯 torch 一定更快。无风险。
2. **tl.arange power-of-2 约束**：triton 内核用 Q_PAD=32（D3Q19 的 next power-of-2）。纯 torch 无此约束，直接用 Q=19，反而更高效。无风险。
3. **跨卡 halo exchange 正确性**：docs/triton_distributed_notes.md 已有详细设计（ghost plane 位置、wrap 对齐），纯 torch 实现需验证边界正确性。缓解：M4 含 4 卡 vs 单卡逐位验证。

### 放弃条件

- **M1 审计发现某模块功能无法用纯 torch 表达**（如依赖 triton 特有的 atomic 操作）→ 标记为永久 triton 依赖，在文档中声明限制，该模块走 CPU fallback。
- **M4 跨卡 halo exchange 4 卡 vs 单卡 L2 >1e-6** 且 2 天内无法修复 → 降级为 2 卡验证（已由 m10 证明可行），多卡扩展推迟，不影响其他方向。

---

## 三方向协同关系

```
方向 C (技术债) ──解除 triton 瓶颈──→ 方向 A (阻力战役) 可全速跑
方向 C (跨卡框架) ──提供域分解 API──→ 方向 A S1+ 多卡扩展
方向 B S0 (FVM/BEM 普查) ──发现 triton 依赖──→ 反馈给方向 C 补充清单
方向 A S3 (SUBOFF) ──提供阻力场──→ 方向 B S3 (声学耦合) 的输入
```

**推荐执行顺序**：C 的 M1 立即启动（0 卡，不占资源）；A 的 S0 与 B 的 S0 并行启动；C 的 M2-M4 在 P1-P2 完成，为 A 的 S2+ 和 B 的 S1+ 解锁全速路径。

---

## 不确定项诚实标注

1. **[待验] TensorFVM/BEM 的 SDAA 兼容性**：未做普查，方向 B S0 是第一步。
2. **[待验] STL 体素化 SDAA 路径**：docs 显示 CPU-only，方向 A S3 前需验证。
3. **[待验] 自由表面模块 SDAA 可跑性**：docs 丰富但无上卡记录，方向 A S4 前需 smoke test。
4. **[待验] TensorPhysics coupling driver 是否已有骨架**：docs 有契约无实现证据，方向 B S1 需从零建。
5. **预算量级为人日估算**：基于 m10（2D Couette 1 天完成）和 m11（普查 1 天完成）的经验外推，3D 和耦合复杂度更高，实际可能超期 30-50%。

---

*本提案仅代表架构视角个人意见，不构成派发指令。终审归 owner。*
