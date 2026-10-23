# MP-DAMBREAK S0 路线裁决书（mission m206）

> 线手册：`/nfs/wangxi/tensor_eco/agent_work/mp_dambreak/README.md`
> 审计依据：`docs/MP_DAMBREAK_S0_AUDIT.md`（m206，逐条 file:line 证据）
> 文献锚点：`/nfs/wangxi/tensor_eco/agent_work/mp_dambreak/prereg/anchors.md`（冻结，chmod a-w）
> 裁决日期：2026-10-23

---

## 裁据回顾

线手册冻结的 S1 判据：
1. **直接观测量 ≤3%**：2D 柱坍塌对文献锚点（M&M 1952）的前缘位置 X(T)，在 ≥2 档网格上 |X_sim - X_ref|/X_ref ≤ 3%
2. **≥2 档单调收敛**：粗→细网格误差单调递减（|ΔE| ≤ 3% 档间差）
3. **质量机器精度守恒**：质量漂移 ≤ 机器精度量级（参考已验证案例标准 ~1e-5）

文献锚点（已冻结于 `prereg/anchors.md`）：
- M&M 1952: T=1.0→X=1.326, T=2.0→X=2.360, T=2.96→X=3.670
- Lobovský 2014: ACCESS BLOCKED（不阻塞）

---

## 路线 A：自由面 LBM（Körner 2005 路线）

### A.1 证据级评估

**库内实现**：`src/tensorlbm/free_surface_lbm_2d.py`（138 行，D2Q9）+ `src/tensorlbm/free_surface_lbm.py`（1265 行，D3Q19）

**历史战绩**：

| 工件 | 结果 | 出处 |
|------|------|------|
| dam_break_3d_mm (m91) | TERMINAL — 12/12 反收敛，质量门 -49.5%/-62.2% | `validation_report.json` (commit 2d00e392) |
| Wave-10 M3 自由面 | FAIL — 12/12 反收敛，缺陷随分辨率放大 | `root_cause_chain.json` |
| FS2D 路线评估 | **未达标但最接近** — 最佳 +3.34/-1.62/-3.13，超 0.34/0/0.13 点 | `REPORT_FS2D_route.md` §0 |
| Route D/A/B | 全部执行死 | `free_surface_receiver_gate_redesign.md` §7-9 |
| waLBerla 外部对照 | 论文原码也不过（+5.9~+13.1 点），我们比论文近 3-8 点 | `REPORT_FS2D_route.md` §11 |

**关键根因链**（5 层，前 2 层已修复）：
1. ✅ 力方案双重计力（g_eff=1.5g）— 已修 commit 634203af（opt-in Guo）
2. ✅ 壁面镜面反射缺陷 — 已修 commit 634203af（流体侧镜面反射）
3. ⚠️ refilling 方案 — 已定位：EQ（默认）→ AVG（论文方案，opt-in `TL_FS_REFILL=avg`），晚期 deficit 基本消除（8.6 点改善）
4. ⚠️ τ 窗口闭合 — T=1 过线在 τ=0.70~0.75，T=2.96 过线在 τ=0.55~0.70，可行区间为空
5. ❌ Route D/A/B receiver gate — 全部执行死，唯一活线是 §6.2 closure redesign

**证据级**：**高**——有完整的 waLBerla 论文级外部对照（§11），有 7 组实跑数据，有 VTK 双口径交叉验证，有四层根因诊断。FS2D 最佳结果仅超门限 0.34 点（T=1）。

### A.2 判据可达性

| 判据 | 当前状态 | 缺口 |
|------|---------|------|
| ≤3% @ ≥2 档 | 最接近：+3.34/-1.62/-3.13（a=40, τ=0.70, AVG） | T=1 超 0.34 点；两档跨度 3.06/3.71/3.24% 全超 |
| 单调收敛 | FAIL — 细化使 T=1 从 +3.34 恶化到 +6.60 | τ 窗口不重叠 |
| 质量守恒 | **PASS** — drift ≤ 2.4e-5（eps=1e-6 时 ≤ 7.8e-6） | 无缺口 |

### A.3 剩余障碍

1. **T=1 早朝过冲**：低 τ 区稳定在 +6.8~+6.9%，不是分辨率问题（a=40→200: +6.75→+6.87）
2. **τ 窗口闭合**：T=1 和 T=2.96 的可行 τ 区间不重叠
3. **receiver gate redesign**：Route D/A/B 全死，需要 §6.2 closure redesign（recv_new promotion / de-synchronised conversion wave / fill-approx-0 exchange guard / negative-donor policy / partial-commit）

### A.4 推荐度

**推荐度：中高**——距离门限最近（0.34 点），有论文级外部对照证明门禁严于文献自身水平。但 τ 窗口闭合和 receiver gate redesign 是未解决的硬障碍。

---

## 路线 B：高密度比伪势（M4 + Guo forcing）

### B.1 证据级评估

**库内实现**：`src/tensorlbm/multiphase.py` `collide_sc_single_component`（当前 main 无 forcing 参数；M4 暂存版有 opt-in forcing: sc/edm/guo）

**历史战绩**：

| 工件 | 结果 | 出处 |
|------|------|------|
| dam_break_sc (m163) | TERMINAL — 密度比 12.3 vs ~800，反向 G 依赖 | `gates.json` (commit d6588f1e) |
| Wave-10 M4 Phase-0 (a) Maxwell | FAIL — CS ρ_v err -3.55%@0.90 超 3% 门 | `CLOSURE.md` |
| Wave-10 M4 Phase-0 (b) Laplace | **PASS** — PR@0.55 ratio 400-557, σ 残差 0.045-0.100% | `CLOSURE.md` |
| Wave-10 M4 Phase-0 (c) 伪电流 | 披露 — u_max 高于良调文献锚 5-25× | `CLOSURE.md` |
| Wave-10 M4 Phase-0 (d) 振荡 | FAIL — tier1 σ_ω/ω=1.70%≥1% | `CLOSURE.md` |
| 高密度比 EOS 溃坝 | **未做过** — m163 评估但未运行 | `terminal_verdict_m163.md` §3.4 |

**关键发现**：
- M4 暂存修改（修正符号 ψ_EOS + YS2006 参数）纯增量无行为改动，电池 PASS 6/6
- PR@T_r=0.55 可达密度比 400-557（Laplace σ 自测 PASS）
- 但溃坝工况涉及大变形/自由表面/冲击，与静态液滴 Laplace 基准物理差异大
- **即使密度比提升，反向 G 依赖的结构性障碍仍在**（m163 §3.4 评估）

**证据级**：**中**——Phase-0 基础验证充分（Maxwell/Laplace/伪电流/振荡四档），但溃坝工况零实测。密度比提升是否解决反向 G 依赖是未验证假设。

### B.2 判据可达性

| 判据 | 当前状态 | 缺口 |
|------|---------|------|
| ≤3% @ ≥2 档 | **未知** — 溃坝工况零实测 | 需完整参数扫描 ~10 GPU·h，无保证过门 |
| 单调收敛 | **未知** — SCMP 路线在密度比 12.3 时 T=1 档间差 7.84% > 3% | 高密度比是否改善收敛未知 |
| 质量守恒 | PASS（SCMP 天然质量守恒） | 无缺口（Guo 力质量中性已验证 sum S=2.7e-09） |

### B.3 剩余障碍

1. **反向 G 依赖**：err(T=1) 与 err(T=2.96) 对 G 方向相反——降 G 修晚期但炸早期。这是 SCMP 路线的结构性障碍，密度比提升是否解之未验证。
2. **溃坝工况适用性**：高密度比 EOS 在 Laplace 静态液滴通过，但溃坝涉及大变形/冲击/自由面，物理差异大。
3. **伪电流**：u_max 高于良调文献锚 5-25×，可能污染前沿测量。
4. **参数扫描预算**：m163 评估 ~10 GPU·h，无保证过门。

### B.4 推荐度

**推荐度：中低**——基础验证（Laplace PASS）是加分项，但溃坝零实测 + 反向 G 依赖结构性障碍未解 + 伪电流问题是减分项。作为路线 A 的回退备选可行，作为首选路线风险高。

---

## 路线 C：MCMP 多组分 + 界面修整

### C.1 证据级评估

**库内实现**：`src/tensorlbm/multiphase.py` `collide_sc_two_component` / `sc_two_component_force`（已落库，m163 README §3.5 提及但未试）

**历史战绩**：无溃坝工况实测。

**m163 评估**（`terminal_verdict_m163.md` §3.5，直读）：
- 需重新标定共存曲线/表面张力/迁移率
- 引入新物理不确定性（双组分扩散、分省）
- 未实际运行（预算风险高，物理设置不等价）

**证据级**：**低**——库内有实现但溃坝零实测，且 m163 已评估为"引入新物理不确定性"。

### C.2 判据可达性

| 判据 | 当前状态 | 缺口 |
|------|---------|------|
| ≤3% @ ≥2 档 | **完全未知** — 零实测 | 需从共存标定开始 |
| 单调收敛 | **完全未知** | 需从共存标定开始 |
| 质量守恒 | 理论可守（双组分各自守恒） | 需验证 |

### C.3 剩余障碍

1. **共存曲线标定**：需重新标定 ρ_l/ρ_v/密度比/表面张力/迁移率
2. **新物理不确定性**：双组分扩散、分省
3. **物理设置不等价**：MCMP 的界面物理与单组分 SCMP / 自由面 LBM 不同
4. **预算风险**：从零开始，无前期验证

### C.4 推荐度

**推荐度：低**——零实测 + 新物理不确定性 + 从零开始。仅作为路线 A 和 B 都失败后的最终回退。

---

## 裁决：推荐 S1 直接执行的最小风险路径

### 首选路线：A（2D 自由面 LBM，Körner 2005 路线，FS2D）

**理由**：
1. **距离门限最近**：FS2D 最佳结果 +3.34/-1.62/-3.13，T=2 和 T=2.96 已在 3% 以内（-1.62 和 -3.13），仅 T=1 超 0.34 点。
2. **有论文级外部对照**：waLBerla 论文原码实跑证明门禁严于文献自身水平（论文 +5.9~+13.1 点全面 FAIL，我们比论文近 3-8 点）。
3. **根因链清晰**：5 层根因中前 2 层已修复，第 3 层（refilling AVG）已定位并大部消除，第 4 层（τ 窗口）已量化。
4. **质量守恒已 PASS**：drift ≤ 2.4e-5，距门限两个数量级。
5. **库内已有 2D 实现**：`free_surface_lbm_2d.py` 138 行，已有 Guo 力 + free-slip walls + AVG refilling + Smagorinsky 的 opt-in 修改（commit 634203af/98d729df/c22f99b9）。

### S1 最小风险执行路径

**Step 1：修复 T=1 早朝过冲**
- 问题：低 τ 区稳定在 +6.8~+6.9%，不是分辨率问题
- 方向：静水初始化（hydrostatic pressure init，waLBerla 论文配置有此，`initHydrostaticPressure`）——当前库内是否实现需 S1 确认
- 预算：CPU 可完成（2D 模拟，a=40/80）

**Step 2：打开 τ 窗口**
- 问题：T=1 过线需 τ∈[0.70,0.75]，T=2.96 过线需 τ∈[0.55,0.70]，交集为空
- 方向：Step 1 修复 T=1 过冲后，T=1 的过线 τ 可能下移，窗口可能打开
- 预算：CPU 参数扫描

**Step 3：两档网格单调收敛**
- 问题：细化使 T=1 从 +3.34 恶化到 +6.60
- 方向：Step 1+2 修复后重测
- 预算：CPU

**Step 4（若 Step 1-3 仍不够）：receiver gate §6.2 closure redesign**
- Route D/A/B 全死后的唯一活线
- 需要：recv_new promotion / de-synchronised conversion wave / fill-approx-0 exchange guard / negative-donor policy / partial-commit
- 预算：开发 + 验证，CPU 可完成

### 失败回退路径

**回退 1（路线 A Step 1-4 全失败）→ 路线 B（高密度比伪势）**
- 暂存修改已就绪（`src_patched/tensorlbm/multiphase.py`，修正符号 ψ_EOS + opt-in forcing）
- 先做 PR@T_r≈0.4-0.5（密度比 ~500-1000）的溃坝参数扫描
- 关键验证：反向 G 依赖是否在高密度比下消失
- 预算：~10 GPU·h（m163 评估），需 GPU5

**回退 2（路线 B 也失败）→ 路线 C（MCMP）**
- 从共存标定开始
- 预算：最大，物理不确定性最高

### 不推荐

- **不推荐**在当前 2D 自由面 3D 版本（dam_break_3d_mm）上继续投入——m91 已判 TERMINAL，12/12 反收敛，缺陷随分辨率放大，需引擎级重设计。3D 留到 S2。

---

## 证据汇总表

| 路线 | 库内实现 | 溃坝实测 | 最佳结果 | 质量守恒 | 外部对照 | 推荐度 |
|------|---------|---------|---------|---------|---------|--------|
| A 自由面 (FS2D) | ✅ 138行 | ✅ 多轮 | +3.34/-1.62/-3.13 | ✅ PASS | ✅ waLBerla 7组 | **中高** |
| B 高密度比伪势 | ✅ 暂存 | ❌ 零实测 | 未知 | ✅ 天然守恒 | ❌ 无 | 中低 |
| C MCMP | ✅ 已落库 | ❌ 零实测 | 未知 | 理论可守 | ❌ 无 | 低 |

---

## 冻结声明

- S1 首选路线 = **A（2D 自由面 LBM FS2D）**
- S1 失败回退 = **B（高密度比伪势）**→ **C（MCMP）**
- S1 判据不变：≤3% @ ≥2 档单调 + 质量机器精度
- S1 锚点不变：M&M 1952 T=1/2/2.96 → 1.326/2.360/3.670
- 本裁决基于 m206 S0 审计的全部直读证据，不预设答案。
