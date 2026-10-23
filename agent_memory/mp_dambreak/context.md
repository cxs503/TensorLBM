# mp_dambreak 线交接笔记（常设线首班 → 下一个自己）

> 线手册：`/nfs/wangxi/tensor_eco/agent_work/mp_dambreak/README.md`
> memory_key: mp_dambreak
> 当前班次：m206 S0（审计+锚点冻结+路线裁决）
> 上一班次：无（首班）

## 当前状态

**S0 已完成**，三件交付物已落本分支 commit `98cf1a4c`：
1. `docs/MP_DAMBREAK_S0_AUDIT.md` — 现状审计（逐条 file:line）
2. `docs/MP_DAMBREAK_S0_ANCHORS_MIRROR.md` — 文献锚点冻结镜像（原件在 worktree 外 `prereg/anchors.md`，已 chmod a-w）
3. `docs/MP_DAMBREAK_S0_VERDICT.md` — 路线裁决书

## 在飞事项

无在飞事项。S0 为审计/冻结/裁决任务，不涉及求解器实现。等 owner 验收后进入 S1。

## 关键结论（写给下一个自己）

### 1. 三个历史终裁案例的状态

| 案例 | 终裁 commit | 状态 | 核心障碍 |
|------|------------|------|---------|
| dam_break_sc (SCMP 2D) | d6588f1e (m163) | TERMINAL | 密度比 12.3 vs ~800；反向 G 依赖（T=1/T=2.96 对 G 方向相反） |
| dam_break_3d_mm (自由面 3D) | 2d00e392 (m91) | TERMINAL | 12/12 反收敛；前锋推进与账本闭合是同一开关（RC#5） |
| Wave-10 M3 (自由面 2D/3D) | root_cause_chain.json | FAIL | 同 m91，缺陷随分辨率放大 |
| Wave-10 M4 (高密度比伪势) | CLOSURE.md | 部分通过 | Laplace PASS；Maxwell/振荡 FAIL；溃坝零实测 |

### 2. 推荐路线

- **S1 首选 = 路线 A（2D 自由面 FS2D）**：距门限最近（T=1 超 0.34 点），waLBerla 论文原码也不过（+5.9~+13.1 点），我们比论文近 3-8 点。
- **回退 = 路线 B（高密度比伪势）**：M4 暂存修改已就绪（修正符号 ψ_EOS + opt-in forcing），但溃坝零实测 + 反向 G 依赖未解。
- **最终回退 = 路线 C（MCMP）**：零实测，从零开始。

### 3. 文献锚点已冻结

- **M&M 1952**：14 点表（Lethe 官方脚本 = M&M 物理实验数据，§11.2 钉死）
  - 检查点：T=1→1.326, T=2→2.360, T=2.96→3.670
  - 时间轴：T = t·√(2g/a)，X = x_front/a
  - 旧表（T=1→1.5, T=2→2.7）已 RETRACTED
- **Lobovský 2014**：ACCESS BLOCKED（6 次网络尝试全失败），不阻塞 S1
- 锚点原件：`/nfs/wangxi/tensor_eco/agent_work/mp_dambreak/prereg/anchors.md`（chmod a-w）

### 4. 关键路径与坑

- **自由面 Route D/A/B 全死**：`docs/free_surface_receiver_gate_redesign.md`（mp_w10 worktree），唯一活线是 §6.2 closure redesign
- **SCMP 反向 G 依赖**：这是 SCMP 路线的结构性障碍——降 G 修晚期但炸早期。高密度比是否解之未验证。
- **waLBerla 对照数据**：`/nfs/wangxi/scratch/dbs_eos/walberla/runs/` 下 7 组 `breaking-dam.txt`（数值模拟输出，非实验锚点）
- **mp_w10 worktree**：`/nfs/wangxi/worktrees/mp_w10 @ 1eae1c3b`（只读，不要修改），存放 Wave-10 残值
- **计算产物**：`/nfs/wangxi/runs/bm_mp_20261001/` 下 m3_freesurface/ 和 m4_pseudopotential/

### 5. S1 最小风险执行路径

1. 修 T=1 早朝过冲（+6.8~+6.9%，静水初始化？）
2. 打开 τ 窗口（T=1 需 τ∈[0.70,0.75]，T=2.96 需 τ∈[0.55,0.70]，交集空）
3. 两档网格单调收敛
4. 若 1-3 不够：receiver gate §6.2 closure redesign

### 6. 测试

- `pytest tests/` 有 pre-existing 失败（gallium_pf 缺模块、suboff/wall_model GPU 相关）
- 本任务未修改 src/ 或 tests/，4376 passed / 160 failed（全 pre-existing）

## 坑与关键路径

- **不要碰 mp_w10 worktree**：它是只读的 Wave-10 残值库，任何修改会破坏审计证据链。
- **anchors.md 是 append-only + chmod a-w**：修正只追加不覆盖。
- **不要外推 M&M 表**：T_max=2.96，T=3.0 需外推被禁止。
- **旧参考表（T=1→1.5, T=2→2.7）已 RETRACTED**：不要再用。
- **密度比 12.3 是 SCMP ψ_exp(G=5) 的硬限**：要 ~800 需 PR@T_r≈0.4-0.5 或 MCMP。
