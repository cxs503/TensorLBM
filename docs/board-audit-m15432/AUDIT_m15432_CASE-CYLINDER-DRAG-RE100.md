# 交差审计：m15432 CASE-CYLINDER-DRAG-RE100（TensorLBM）

**审计任务**: m15712 · **被审任务**: m15432（succeeded，auto-approve，report `tensor_reports.id=14637`）
**仓库**: TensorLBM · **被审 commit**: `68813473 "m15432: 圆柱阻力 Re=100 无约束 2-D 算例（Schäfer & Turek 1996 锚点）"` · **PR**: #1829（merged）
**审计裁定**: **L3 —— 独立功能实现 + 主求解器内核接入；L4 基准验证不成立**（主判据 Cd 超差 +25.41% ≫ 5%，无网格收敛链）
**maturity gate 建议**: `reopened`（返修：Cd 项返修 + 基准裁判登记 + 收敛链补齐）

---

## ① 案例与基准（ref 出处与交付声明）

圆柱绕流 Re=100 是钝体阻力的标准验证题。被审单交付的是**无约束 2-D 圆柱**的时均 Cd、时均 Cl 与 St。

| 项 | 被审单声明 | 审计核验 |
|---|---|---|
| 基准文献 | Schäfer, Rannacher & Turek (1996), *Flow control by means of fictitious bodies*, Springer | **⚠ 未闭环**：工作树内无 ref PDF/表格摘录/页码（无 `ref_provenance.json`、无 sha256），与任务书「ref 原文/手册查得、引用原表原页（零手抄）」不符 → **F4** |
| ref 数值 | Cd=1.3796, Cl=0.0005, St=0.164 | 与「无界 2-D 圆柱 Re=100」通用区间（Cd≈1.33–1.38, St≈0.164–0.166）量级自洽；但同族经典 DFG 2D-2 算例是 **24.4% 阻塞比通道**（Cd≈3.22, St≈0.30），两者不是同一构型，引用需原表才能定案（见 F4） |
| 容差 | <5%（`tensor_case_pool` id=11 `CYLINDER-DRAG-RE100` tol_policy=`<5%`，status=dispatched） | ✔ 与任务书一致 |
| 机器裁判 | 任务书要求对照 `tensor_benchmark_cases` 的 `anchor_slug='CASE-CYLINDER-DRAG-RE100'` 逐项核 | **✗ 该行不存在**（`SELECT count(*) ... = 0`，全表 21 行）→ **F7**：无机器裁判，判据回退到 case_pool 声明 + task goal 文本 |

交付的算例配置（`artifacts/cylinder_re100_D10/cylinder_re100_unconfined.json`）：

| 量 | 值 |
|---|---|
| 网格 | 250 × 100（上游 5D + 下游 20D，横向 10D），D=10 格点 |
| 阻塞比 B=D/H | **10.0%**（声明为 "unconfined"） |
| Re / Ma / τ | 100 / 0.173 / 0.53（D2Q9 BGK，float32） |
| 边界 | `far_field_bc_2d`：入口+两侧面=自由流平衡，出口零梯度；圆柱半程反弹 |
| 力 | Ladd 动量交换（后迁移、反弹前） |
| 统计 | 预热 8000 步 + 采样 18000 步，双半窗漂移 <1% 停 |

---

## ② 审计方法

1. **取证**：只读被审工作树 `/root/tp-agent/work/m15432`，逐项核三件套（run_artifacts / ref-vs-got / repro_cmd）。
2. **独立复算**（`audit_m15432_recompute.py` → `audit_recompute.json`）：**不读报告数字**，从交付工件的 18000 点逐点序列重算 cd_mean / cl_mean / cl_amp / period / St / 双半窗漂移，与自报值逐项比对（一致 → 非伪造）。
3. **复现**：以被审分支自身 `src` 为路径（`PYTHONPATH=src`）跑 `--pilot` 冒烟，确认入口可执行、无 NaN、结论方向一致（22 s）。
4. **物理自洽**：从交付场重绘涡量、查入口/出口速度、尾迹不对称性、再循环长度。
5. **对抗核查**：读实现（318 行）、力核 `compute_obstacle_forces`（Ladd 2Σc_i f_i）、`far_field_bc_2d`；查 stationarity 判据的统计口径；查 ref 出处与机器裁判；查 repro 的 import 解析路径。
6. **判级**：按 L0–L5 的功能实质定级，报告体量不改变级别。

---

## ③ 计算结果（审计独立复算，零手抄）

### 3.1 工件内部一致性（reported vs 审计重算）

| 量 | 审计重算 | 报告/工件自报 | 偏差 |
|---|---|---|---|
| Cd 均值 | 1.730097 | 1.730097 | 0.00e+00 |
| Cl 均值 | −8.748e−05 | −8.748e−05 | 0.00e+00 |
| Cl 幅值 | 0.476036 | 0.476036 | 0.00e+00 |
| 涡脱周期 | 584.131 步（30 次过零） | 584.131 | 0.00e+00 |
| St = D/(T·U) | 0.171195 | 0.171195 | 0.00e+00 |
| 双半窗 Cd 漂移 | 0.028396% | 0.028396% | 0.00e+00 |

→ 交付工件是**真算出来的**（18000 点全精度序列、场数据、图均可重算复现，无拼接/伪造迹象）。

### 3.2 基准对照（唯一裁判 = case_pool 声明 <5%）

| 关键量 | 实测（D=10, B=10%） | 基准 | 相对误差 | <5%? |
|---|---|---|---|---|
| **Cd（主判据）** | **1.7301** | 1.3796 | **+25.41%** | **✗** |
| Cl 均值（对称性） | −8.75e−05 | 0.0005 | \|Δ\|=5.9e−04 ≪ Cl 幅值 0.476 | ✓（对称性） |
| St | 0.1712 | 0.164 | +4.39% | ✓ |
| Cl 幅值 | 0.4760 | 报告未给 ref 幅值 | **未判** | —（报告写"量级正确 ✓"无对照支撑，F5） |

**显式回答「是否 <5%」：Cd 否（+25.41%），St 是（+4.39%）。主判据不达标 → 基准验证不成立。**

### 3.3 物理自洽性（交付场数据）

| 检查 | 结果 | 判读 |
|---|---|---|
| 入口平均 ux / 出口平均 ux | 0.10000 / 0.09985（设 u_in=0.10） | ✔ 边界条件按设计生效 |
| 场形状 / 圆柱位置 | 100×250 / (cx,cy)=(50,50), D=10 | ✔ 与配置一致 |
| 涡量幅值（p99.5） | 0.0265（max 0.0916） | ✔ 尾迹量级正常 |
| 尾迹中心线 ux<0 区 | x=53…71 → Lr≈2.2D（瞬时快照，最小 ux=−0.0168） | ⚠ 反流极弱、仅 ~9 格，零交叉提取受分辨率噪声支配；**不作为判据**（文献 Re=100 时均 Lr≈0.9–1.0D，本快照无法定论） |

### 3.4 审计独立复跑（冒烟，非判据配置）

`PYTHONPATH=src OMP_NUM_THREADS=1 python3 -m tests.validation.cylinder_re100_unconfined --pilot` → 22 s 完成、无 NaN：
110×60（B=16.7%）、4000 步 → **Cd=1.8134（+31.4%）**、St=0.1655（+0.9%）。
方向与主算例一致（Cd 高估、St 稳健），排除"工件来自另一套代码"的可能。

---

## ④ 结果图

`docs/board-audit-m15432/fig_audit_m15432.png`（四联图，全部由交付原始数据重绘）：
A 基准 ref-vs-got 柱状（Cd ✗ / St ✓，±5% 带）；B 采样窗 9 个 2000 步窗 Cd 均值（平坦但水平 1.73，不在 ±5% 带内）；
C 同仓三次运行 Cd vs 阻塞比（D=10 pilot B=16.7% → 1.813；D=10 B=10% → 1.730；m15407 D=16 B=7.14% → 1.5445；文献带 1.33–1.38）；
D 由 `samples.wz` 重绘的涡量场（反造假核对）。

---

## ⑤ 审计发现（按严重度）

| # | 严重度 | 发现 | 证据 |
|---|---|---|---|
| **F1** | **阻断（判级）** | 主判据 Cd 超差 **+25.41%** ≫ 5%，基准验证不成立；报告已如实标 honest-FAIL，未粉饰 | `artifacts/cylinder_re100_D10/cylinder_re100_unconfined.json` comparisons[0]；`docs/cylinder_re100_m15432.md` §④ |
| **F2** | **高（设计缺陷）** | 交付构型与基准构型不匹配：横向仅 **10D（B=10%）** 且两侧面**强加自由流 Dirichlet**，却按"无约束"基准比较；无阻塞修正讨论。B 与分辨率同向变化，现有三点（1.813/1.730/1.545）**无法分离**二者贡献 | json `derived.blockage=0.1`；`src/tensorlbm/boundary/boundaries.py:201-222`；图 C |
| **F3** | **高（误差分析缺失）** | 无网格收敛链：D=20 收敛点从未跑（`cj1159` 审计时仍 `queued` 55.6 min，任务已 succeeded 关闭）；报告 §⑤"D=10 已收敛"混淆**时间定常**与**网格收敛**，为无支撑推断 | `docs/cylinder_re100_m15432.md` §⑤ 第 2 条；`run_cluster ps cj1159` → `queued` |
| **F4** | **中（出处闭环）** | ref 无页码/原表摘录、无 PDF 与 sha256 归档，"Schäfer/Rannacher/Turek 1996"归属不可独立核验（同族 DFG 2D-2 为 24.4% 阻塞通道、Cd≈3.22，数值与本单不是同一构型） | 工作树 `find` 无 ref 工件；报告 §① |
| **F5** | **中（未判量当已判）** | Cl 幅值 0.476 交付但**无 ref 对照**，报告写"✓（量级）"属未经验证的加勾；Re=100 二维典型幅值 ≈0.32–0.33，即潜在 +45% 量级偏差未被计入 | json `derived.cl_amp=0.476036`；报告 §④ 第 2 行 |
| **F6** | **中（可复现性卫生）** | repro_cmd 无 `PYTHONPATH=src`：`import tensorlbm` 解析到**另一工作树** `/root/tp-agent/work/m13963/src`（editable 安装）。两树 `boundaries.py`/`solver.py` 有 diff（m15432 仅多 `input_contract` 入参校验，τ=0.53 数值等价）→ 结论不受影响，但记录的复现命令没有钉住被审代码 | `python3 -c "import tensorlbm"` → m13963；`diff` 结果 |
| **F7** | **中（系统性）** | `tensor_benchmark_cases` **无 `CASE-CYLINDER-DRAG-RE100` 行**（0/21），任务书要求的"基准裁判"不存在；上一单 CASE-KARMAN-RE100 审计同样 0/19 → 算例池与裁判表脱节，属生态摩擦 | `SELECT count(*) FROM tensor_benchmark_cases WHERE anchor_slug='CASE-CYLINDER-DRAG-RE100'` = 0 |
| **F8** | 低（口径） | stationarity 判据用 9000-vs-9000 双半窗均值，会把慢漂移部分相消：首/末 1000 点均值 1.7258→1.7327（+0.40%），报告只写"drift 0.028%"；对 25% 结论无实质影响 | json `cd_tail` 重算；`tests/validation/cylinder_re100_unconfined.py:179` |

**无缺陷项（已核）**：力核 `compute_obstacle_forces` 实现即 Ladd 标准式 `F=2Σc_i f_i`（`:164`），调用时序（迁移后、反弹前）与文档一致（`:166-167`）；三图非空白（std 31–57，ink 2.8–29%）；字段/图/JSON 互洽。

---

## ⑥ 判级（L0–L5）

| 级 | 要求 | 本单 | 判定 |
|---|---|---|---|
| L1 | 数学/单元验证 | 有（力核/BGK 由仓库其它测试覆盖） | ✔ |
| L2 | 独立功能实现（真代码 输入→计算→输出） | `tests/validation/cylinder_re100_unconfined.py` 318 行真跑，出 18000 点序列+场+图 | ✔ |
| L3 | 系统集成（接入主求解器/平台） | 只用库内核：`collide_bgk_matmul`(solver.py:221) / `stream`(:423) / `bounce_back_cells`(boundaries.py:40) / `compute_obstacle_forces`(:164) / `far_field_bc_2d`(:201) / `cylinder_mask`(:9)，零手写碰撞-迁移；case_pool id=11 登记；报告 id=14637 | ✔ |
| L4 | 基准验证（完整算例+误差分析+可复现） | 主判据 **Cd +25.41% ≫ 5%**；无网格收敛链；ref 出处未闭环 | **✗** |
| L5 | 科研工程成果 | — | ✗ |

**裁定：L3。** 报告 6 段齐全、honest-FAIL 如实标注（应肯定），但报告体量与诚实不改变功能实质；L4 的核心门槛"主判据过容差 + 误差分析"未达。

---

## ⑦ 复现（审计侧）

```bash
# 1) 工件独立复算（读被审工件 + 审计冒烟跑）
python3 docs/board-audit-m15432/audit_m15432_recompute.py     # -> audit_recompute.json
# 2) 图（四联图，含由 samples.wz 重绘的涡量）
python3 docs/board-audit-m15432/make_audit_figures.py         # -> fig_audit_m15432.png
# 3) 审计冒烟复跑（22 s，非判据配置）
cd /root/tp-agent/work/m15432 && PYTHONPATH=src OMP_NUM_THREADS=1 \
  python3 -m tests.validation.cylinder_re100_unconfined --pilot --emit-dir /tmp/m15712_pilot
```

## ⑧ 未运行 / 未核（如实声明）

- **未重跑主算例全量**（342 s）以逐点复算 Cd：交付序列自洽 + 冒烟跑方向一致已足够支撑结论；全量重跑对 +25% 的判定无改变。
- **未跑 D=20 / 更大域**（审计已递交 `cj1175`：D=10, height=20D 诊断点，审计时排 361 位，未纳入判据）。
- **未独立取得 ref 原文表格**（沙箱无外网）：F4 标记为"未闭环/不可核验"，不据此判"造假"。
- **未核**：TensorLBM 是否另有 MRT 版本库内核能改善 Cd（超出核验范围，转交返修单）。

## ⑨ Disposition

**request changes / reopened**。返修最小集：
1. 出一次**真无约束**配置（B ≤ 2%，`height_d ≥ 50`；或加阻塞修正并在报告显式声明）并给 D=16/20 至少两点收敛链，目标 Cd 进入 1.38 ±5%；
2. 把 `CASE-CYLINDER-DRAG-RE100` 登记进 `tensor_benchmark_cases`（case/ref/tolerance），让 gate 有机器裁判；
3. ref 出处附页码/原表摘录 + sha256（零手抄纪律）；
4. 判定量清单里**显式包含 Cl 幅值**（或写明"不判"），不得对无 ref 的量打勾；
5. repro_cmd 加 `PYTHONPATH=src`（或在仓内 CI 固定 editable 安装指向本仓）。