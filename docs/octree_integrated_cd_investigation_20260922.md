# 集成 octree 球形绕流 Cd 偏高 2.5x 排查存档（2026-09-22）

> 工作区 `/root/TensorLBM_feat2` · 分支 `feat2-octree-integrated` · 归档 commit 起点 `18047d2`
> 目的：固化本轮（2026-09-22）"集成 R6 d_max=2 阻力 Cd 2.72 vs 参考 1.0917（≈2.5x）"
> 的系统性排查成果——21 项假设的判定、关键数据矩阵、已确认 bug 与修复、未解之谜与剩余方向。
>
> 关联文档：
> - `docs/shell_bfl_force_analytic_validation_20260922.md`（壳层 BFL 力链解析基准验证）
> - `docs/shell_bfl_moving_wall_coefficient_audit_20260922.md`（运动壁 −6/−3/q 系数核查）
> - `docs/L1_MIDDLE_BLOCK_INTEGRATION_DESIGN.md`（L1 中间块集成设计，§8 为本轮发现回填）

---

## a) 问题陈述

**现象**：集成框架（coarse 域分解 → L1 中间块 → octree 壳层叶，`d_max=2`）在 R6 球
Re=100 上收敛 `Cd_mem = 2.7255`（2000 步）；而参考/单卡口径为：

| 口径 | Cd | 说明 |
|---|---|---|
| Schiller–Naumann 经验式（无限域） | **1.0917** | 主参考值 |
| 单卡均匀网格外推（R9/R12/R15） | 1.1263 | `docs/evidence/amr-sphere-drag-validation-r1.json` |
| 单卡三级 AMR 收敛（R6） | 1.2432 | `static_block_amr` 单卡路径 |
| 单卡三级 AMR 收敛（R10） | 1.1093 | 单卡三级，设计文档验收锚 |
| blocked-ref（有限域经验修正） | 1.654 | `--blockage-correction simple` |

**偏差**：`2.7255 / 1.0917 = 2.497 ≈ 2.5x`（对单卡 R6 收敛 1.2432 则是 2.19x）。
集成侧**系统性偏高**，方向为"力超出"，而非单卡早期的"力不足（0.55，未收敛瞬态）"。

**任务**：定位这 2.5x 的机理。本轮系统性排查了 21 项假设（§b），确认了 3 类
真实 bug（§d），并把剩余未知收敛到下一节（§e）的方向。

**关键判据/工具**：把壳层 BFL 力链从耦合动力学中**解耦**——用解析写死的冻结流场
（均匀来流 + 静止壁的 plain-BB 恒零；刚性旋转 + 共转壁的解析零力矩）检验力链的
**细化不变性**（d_max=1↔2）与**约定一致性**。这一探针直接命中了混合层级力折算 bug。

---

## b) 21 项假设排查表

> 判定图例：~~排除~~ = 已排除为非根因；**确认** = 确认为真实 bug（见 §d）。
> "方法"多为 A/B 对拍（同一配置改单一开关）或 CPU 解析场冻结探针。

| # | 假设（曾怀疑的根因） | 方法 | 结果 | 结论 |
|---|---|---|---|---|
| 1 | **BFL 公式/插值结构**错 | Bouzidi 文献核对（arXiv:1906.05445）＋变体扫描 V1–V6＋共转滑移解析零检验 | 系数对 `(-6, -(3/q))` **正确**；真正的错在插值源 `f_opp_post`（应为流体侧后碰撞 `fp_d`） | **确认**（结构 bug，见 §d/§b-12） |
| 2 | **几何**（bl 厚度 / 壳层形状） | bl 3→6 A/B；单卡 vs 集成 n_leaf / 链接数对拍 | bl 3→6 BFL 链接数**不变**（266312 = 266312）；单卡/集成 R6 d1 几何完全相同（n_leaf 199552, links 66416） | ~~排除~~ |
| 3 | **面积公式**错 | 单卡 vs 集成 wall-link 叶 dx / `radius_leaf` / `dynamic_area` 三路对拍 | 集成 wall 叶 44552 **全在 level-2**；`radius_leaf=48`、`area=13.0288`，与单卡**一致** | ~~排除~~（历史 count-weighted 面积 bug 见 §d-3） |
| 4 | **Re 定义**口径错（R vs 2R） | `tau_from_re` 的 `L_ref` 口径审计 | 历史 bug（曾用 R → 实际跑在 Re=200），已修为 `L_ref=2R`（`81cc844`）；现集成/单卡同口径 | ~~排除~~（历史 bug 已修，见 §d-2） |
| 5 | **尾流盒**（wake box / `wake_cells`） | bbox / 缩放审计；R10 d1 异常归因 | R10 d1 异常源于 `wake_cells` 为**绝对值未随 R 缩放**（1.95D vs R6 3.25D，−40%）；已对齐单卡 | ~~排除~~ |
| 6 | **ghost 深度**不足 | `ghost=1` vs `ghost=3` A/B（`69d027c`） | g1 = g3，差异全在 coarse 层物理 | ~~排除~~ |
| 7 | **L1 初始化**（uniform eq vs coarse window 采样） | L1 从 coarse window + neq 注入（`6c71a62`） vs 均匀 eq；Cd 对拍 | 改动落地（对齐 `StaticBlockAMR3D`），但未消除 2.5x 差距 | ~~排除~~（非主要根因） |
| 8 | **粗层固体**处理（BB / freeze / none） | `--coarse-bb` / `--coarse-freeze` / `--no-coarse-bb` A/B | `coarse-freeze` 无效果（2.83 vs 2.93 BB）；粗层被 L1 restrict 覆盖，近乎惰性 | ~~排除~~ |
| 9 | **far-field yz**（yz 面硬 eq reset vs 零梯度外推） | `--far-field-yz-extrapolate` A/B（`9e3184f`） | 无显著效果 | ~~排除~~ |
| 10 | **ghost neq** 松弛缩放错 | ghost 采样 neq 松弛量审计（scale = `tau_f/(2^lev·taus[0])·(1−1/tau_f)`） | 已修正但不解释 2.5x | ~~排除~~ |
| 11 | **混合层级力折算**缺 dx² 空间因子 | CPU 冻结解析场 d1→d2 **细化不变性**（plain-BB 恒零证明记账干净） | d1→d2 链接 ×4.01 但力 ×3.71（每链几乎不变）；`leaf_force_spatial_weights=2^-2(level-ref)` 修复后比值 **3.71→0.93** | **确认**（见 §d-1） |
| 12 | **`f_opp_post`** 插值源错（Bouzidi） | 变体扫描 V1–V6＋共转旋转壁解析零力矩（应 ≡0） | 现状 `Tz=−390.9 ∝ Ω`（破坏解析零检验）；V4（标准 `fp_d`）→ `Tz≈0`，但 case A Fx 70→233 | **确认**（旋转壁路径；待真实阻力基准复核，见 §d-4） |
| 13 | **d1 路径**更接近参考 | 集成 d1 vs d2 收敛对拍 | 集成 d1 `Cd=4.1490`（**更差**）；单卡 R6 d1 3.74 是低分辨率物理值 | ~~排除~~（d1 并不接近参考） |
| 14 | **启动 ramp**（`bfl_ramp_wall_velocity`）污染收敛 Cd | ramp 审计；`ramp_steps` 后壁速归零 | `_ramp_activation` 在 ramp 结束后壁速精确为 0，不影响收敛 Cd | ~~排除~~ |
| 15 | **堵塞 β**（域太小） | 扩域 `ny=96` A/B；`--blockage-correction simple/glauert/off` | 扩域 **无效**（Cd 不变）；堵塞仅解释 ~2–15%（blocked-ref 1.654） | ~~排除~~ |
| 16 | **ghost donor 源**（L1 场 vs coarse 稀疏场） | `--ghost-from-l1` A/B | 仅 **−5~−11%**（3.05 → 2.89），仍 5.2x vs 单卡 0.56 | ~~排除~~（非根因） |
| 17 | **D3Q27 方向覆盖**不足（壳层 BFL 漏方向） | `bfl_mask` 方向完备性审计 | 无缺失方向的证据 | ~~排除~~ |
| 18 | **L1 界面滤波**缺失（界面 neq 失配） | 实现 `l1_block interface_filter` + `--l1-interface-filter` CLI；CPU 对拍 | 关闭时位一致；开启时守恒 4.4e-16；但集成 Cd **无变化**（壳层 ghost 走 coarse 直采，绕过滤波） | ~~排除~~ |
| 19 | **阻塞修正**未启用 | `--blockage-correction` 变体 | 无效果 | ~~排除~~ |
| 20 | **空间因子**（dx² 折算）缺失（与 #11 同源验证） | CPU 冻结场含/不含 `include_spatial` A/B（step 25/50/75） | 29.68/7.42 = **4.0x**；14.54/3.63 ≈ 4.0x；12.98/3.24 ≈ 4.0x | **确认**（与 #11 同一 bug，见 §d-1） |
| 21 | **lattice**（D3Q19 vs D3Q27）效应 | 单卡同配置 D3Q19 vs D3Q27 收敛对拍 | D3Q19 step200 `Cd=1.5072` vs D3Q27 step150 `0.56`（≈3x）；单卡 D3Q27 被 **SDAA 大索引死锁**阻断，无法收敛到 1.09 | **未解之谜**（见 §e） |

**小结**：21 项中 **18 项排除**、**3 项确认为真实 bug**（#11/#20 混合层级力折算、
#12 `f_opp_post`）、**1 项（#21 lattice）列为未解**。

---

## c) 关键数据矩阵

### c.1 收敛值（Cd_mem，R6 球 Re=100，除非注明）

| 配置 | Cd | 步数 | 备注 |
|---|---|---|---|
| 集成 L1-block `d_max=2` | **2.7255** | 2000 | 主嫌疑值（4 子步/root） |
| 集成 legacy 两级 `d_max=1` | **4.1490** | 收敛 | 更差，非"接近参考" |
| 集成 L1-block `d_max=2`（修复前基线） | 2.92 | — | 混合层级折算修复前后对比起点 |
| 集成 `--ghost-from-l1` | 2.89 | — | vs 3.05（coarse 直采），仅 −5~−11% |
| 单卡三级 AMR（R6） | 1.2432 | 收敛 | `static_block_amr` |
| 单卡三级 AMR（R10） | 1.1093 | 收敛 | 设计文档验收锚 |
| 单卡均匀网格外推 | 1.1263 | — | R9/R12/R15 |
| **Schiller–Naumann（参考）** | **1.0917** | — | 无限域经验式 |
| blocked-ref | 1.654 | — | 有限域经验修正 |

**比值**：集成 d2 / 参考 = `2.7255 / 1.0917 = 2.497 ≈ 2.5x`；集成 d2 / 单卡 R6 = `2.19x`。

### c.2 瞬态（未收敛时不可判负）

| 配置 | Cd | 步 | 说明 |
|---|---|---|---|
| 单卡 D3Q19 | **1.5072** | 200 | 仍在下降，向 1.24（R6）收敛 |
| 单卡 D3Q27 | **0.56** | 150 | 仍在上升/低位；单卡 0.54–0.55 是**未收敛瞬态**（不可作为"集成偏高"的对照） |

> ⚠️ 教训：早期"单卡 0.54 vs 集成 2.9"的对照是**瞬态误差**。收敛后单卡 R6 = 1.2432，
> 真实差距是 **集成 2.92 vs 单卡 1.24（2.35x）**，不是 5.4x。（`2a6489f`）

### c.3 A/B 对比（同配置改单一开关）

| A/B 轴 | A | B | 比值/差 | 判定 |
|---|---|---|---|---|
| 空间因子（含/不含 `include_spatial`）step 25 | 29.68 | 7.42 | **4.0x** | 确认 dx² 空间因子 |
| ↑ step 50 | 14.54 | 3.63 | 4.01x | ↑ |
| ↑ step 75 | 12.98 | 3.24 | 4.01x | ↑ |
| ghost donor 源（coarse 直采 / L1-field donor） | 3.05 | 2.89 | −5~−11% | 排除 |
| 面积（集成 / 单卡） | `radius_leaf=48, area=13.0288` | 同左 | 1.00x | 排除 |
| bl 厚度（3 / 6）链接数 | 266312 | 266312 | 1.00x | 排除几何 |
| ghost 深度（1 / 3） | g1 | g3 | 相等 | 排除 |
| lattice（单卡 D3Q19@200 / D3Q27@150） | 1.5072 | 0.56 | ≈2.7x | **未解** |

### c.4 CPU 冻结解析场力链（`shell_bfl_force_analytic_report.json` / 既有文档 §3）

| 用例 | 解析答案 | d1（现状） | d2（现状） | d2（修复权重） |
|---|---|---|---|---|
| A 均匀来流 + 静止壁 plain-BB | 0 | `+0.0`（精确） | `+0.0` | — |
| A 壳层 BFL Fx（D3Q27） | — | `+70.37` | `+264.26`（raw 时间权重） | `+65.33`（折入 dx²） |
| A 壳层 BFL Fx（D3Q19 报告 JSON） | — | `+70.79` | `+264.26` | `+66.07` |
| A ideal-reflection 交叉参照 | — | `+276.64` | — | — |
| B 刚性旋转 + 共转壁 Tz（解析 0，D3Q27） | **0** | `−390.9`（Ω=.002） | `−402.3` | — |
| B ↑（Ω=.004） | 0 | `−781.8`（严格 ∝ Ω） | `−804.6` | — |
| B plain-BB Tz | 0 | `≈−1.9e−14`（精确） | 同 | — |

**关键读法**：
- plain-BB 在所有用例精确为 0 ⇒ **链接集/权重记账/分片求和干净**（`_bfl_force_chain_audit.py`：
  integrated/single Fx ratio = 1.0000）⇒ 2.5x 不是分片/邻居表问题。
- d1→d2 力 ×3.71、链接 ×4.01、**每链几乎不变**（1.06e−3 → 9.81e−4）⇒ 混合层级**未带空间因子**。
- 共转壁 `Tz` 严格 ∝ Ω（非 O(u³) 压缩伪影），且随细化增长 ⇒ 运动壁插值**结构**错。

---

## d) 已确认的 bug 与修复

### d.1 混合层级力折算缺失 dx² 空间因子  ✅ 已修（`18eedca`）

**问题**：`ShellForceLedger` / `leaf_force_weights` 只带**时间**权重（子步数 `2^-(d_max−d_leaf)`、
除以 `2^d_max`），**没有携带空间因子** `(dx_leaf/dx_ref)^2 = 2^-2(level−ref)`。当壳层同时含
depth-1（dx=0.5）与 depth-2（dx=0.25）叶时，冲量被按同一权重累加（`force.convert_leaf_force_to_l1()`
假定**单一** `dx_leaf`，混合层级下不成立）。深度 2 叶各量 ×4，权重膨胀 **1.758x**。

**修复**：新增 `bfl.leaf_force_spatial_weights(octree, reference_level=1)` = `2^-2(level−ref)`；
`force.substep_force_weights(..., include_spatial=True)` 成为**默认**；`convert_leaf_force_to_l1`
在混合层级上 **fail-closed**（拒绝以单值 dx 折算）。

**验证**：CPU 冻结场 d1→d2 力比 **3.71 → 0.93**（目标 ~1）；A/B 空间因子精确 **4.0x**（§c.3）。

**溯源**：depth-2 叶由 `f97a0f2` 引入（有意的，`d_max` 恒为 2），但激活了此归一化 bug。

### d.2 Re 定义用错特征长度（R vs 2R）  ✅ 已修（`81cc844`）

**问题**：集成脚本曾用**半径** R 作 `L_ref`，实际跑在 **Re=200** 而非 Re=100（2x 误差）。
所有早期集成 Cd（0.82/0.94/3.74/2.75/4.57/1.10）均为 Re=200 值，与单卡 Re=100 不可比。

**修复**：统一到 `tensorlbm.lbm_re_tau.tau_from_re(u, L_ref=2R, Re)`（D 口径，`Re = u·2R/ν`）。
本轮已将集成脚本迁移到共享模块（`lbm_re_tau`），行为等价。

### d.3 面积公式（`dynamic_area` 解析分辨率）  ✅ 已修（`dcf899e` / `01d0752` / `6702666` / `18eedca`）

**问题**：曾用**全壳层计数加权均值 dx**（`dx_leaf_coarse`）当作 wall-link 分辨率。
但壁面 BFL 链接**只落在最细的壁面相邻叶**（d_max=2 → level-2），而 level-1 外带（R10 d2 占 58% 叶）
**不carry 任何 wall link**，导致面积被 `(dx_wall/dx_count)^2` 缩小、Cd 同倍放大
（R10 d2：36.19/14.54 = 2.49x，Cd 9.33 → 3.75）。

**修复**：迁移到共享 `tensorlbm.drag_normalize`：
- `compute_wall_link_dx(octree, l1_block=...)` 直接取壁面叶的层级 dx（无 wall link 时回退最细层）；
- `leaf_radius_from_dx(radius_coarse, dx_leaf)` = `radius_coarse / dx_leaf`；
- `dynamic_area(u, radius_leaf)` = `0.5·u²·π·radius_leaf²`。

与 d.1 配套：`leaf_weights` 折入 `(dx/dx_ref)²`（reference = level-1）后，面积必须用 **level-1**
叶 dx 归一（力与面积各带一个 dx²，Cd 不变）——对单层级壁面壳是**精确 no-op**。

**本轮证据**：集成 wall 叶 44552 **全在 level-2**；集成/单卡 `radius_leaf=48`、`area=13.0288` **完全一致**
⇒ 面积不再是本轮 2.5x 的根因（但对 R10 d2 等混合层级仍是必需的既有修复）。

### d.4 P0 ghost level 失配  ✅ 已修（`0a2583f`）

**问题**：`build_ghost_plan_coarse_parent` 在 `_fill_ghost_impl` 里用了 **L1 层级**（偏一层），
导致 `tau_f` 错、`2^lev` 错 ⇒ neq 注入 **2.18x 过强**（解释 SUBOFF P0 的 1.10→1.28 偏差）。

**修复**：给 plan 增加 `lev`（coarse-frame 层级），贯穿 shard slice/merge 传播。
CPU 验证：coarse-parent 模式缩放 **2.183 → 0.988**（真实叶上精确 1.0）。

### d.5 `f_opp_post` 插值源错（Bouzidi，运动壁路径）  ⚠️ 已定位，待复核后采纳

**问题**：`octree_boundary/bfl.py` 两分支都以 `f_opp_post = f[opp]`（**从固体侧流入**的后流值）
作插值第一项；Bouzidi 标准（`docs/octree-boundary-design.md` §3.4、`bfl_common`）用**流体侧后碰撞量**
`fp_d = f_prev[d](x_f)`、`fp_up = f_prev[d](x_f−c_d)`、`fp_opp = f_prev[opp](x_f)`。
系数对 `(-6, -(3/q))` **本身正确**（arXiv:1906.05445 式 (11a)(11b)），错的是**结构**。
后果：共转旋转壁解析零力矩被破坏，`Tz = −390.9 ∝ Ω`。

**变体扫描**（`docs/shell_bfl_moving_wall_coefficient_audit_20260922.md` §5）：

| 变体 | 结构 / 系数 | d1 Tz | case A Fx |
|---|---|---|---|
| V1 现状 | `f_opp_post` / `(-6,−3/q)` | −390.9 | 70.37 |
| **V4 推荐** | **标准 `fp_d` / `(-6,−3/q)`** | **−0.155** | 233.54 |
| V6 备选 | `f_opp_post` / `(−6(1−2q), 0)` | −0.008 | 70.37 |

**当前状态**：**未采纳**。V4 把旋转壁残差压低 3–4 量级，但会把 case A 的 Fx 从 70→233（×3.3）。
**静止壁路径保留 `f_opp_post`**（与 `main` 的 Cd 3.0→1.09 修复一致），故不能简单全局改。
**采纳前必须**在真实球体阻力基准（`sphere_bfl_control_volume` / 单卡收敛 Cd）或 Couette/TC 上复核 Cd。

**附带独立发现**：`bfl_common`/`bfl_d3q19` 的二次支运动壁系数为 `-6`（无 q），与 Bouzidi 式 (11b)
的 `1/(2q)` 不符——共转滑移判据在 q>0.5 给出 O(Ma) 伪力（现有测试只在 q=0.5 覆盖）。

### d.6 L1 块二次 freeze（refreeze）偏离设计文档  ✅ 已定位并实现开关（`af94bc3`，默认待切）

**文档要求**（`docs/L1_MIDDLE_BLOCK_INTEGRATION_DESIGN.md` §3b）与单卡 `StaticBlockAMR3D`
都只做**一次** freeze（`where(l1_solid_q, before, collided)`）：固体内部跳过碰撞，但**参与
streaming**——固体因此像一个"输送机"，把流进来的布居重新喷向下游。

**实现**（`l1_block.py:563-567`）多了一次：

```python
before = self.l1_f
post = self.collide_fn(before, self.tau_l1)
post_frozen = torch.where(self.l1_solid_q, before, post)
streamed = self.stream_fn(post_frozen)
frozen = torch.where(self.l1_solid_q, before, streamed)   # ← 第二次 freeze（偏离）
```

**后果**：固体内部**完全不参与对流** → 近尾迹被掏空 → 壳层 BFL 感受更强不平衡 → 力偏高。

**证据**：
- CPU 步进对拍探针 `_audit_l1_vs_ref_cpu.py`：`no_refreeze=True` 时固体内部 diff
  **恰为 0.000e+00**（与单卡**位级等价**）；默认配置固体内部差 `-1.06e-2`、
  流体侧 `max|d|=2.4e-4`、coarse 层 `1.5e-3`。单步相对漂移 `4.2e-6`，
  在 2000 步尺度足以解释壳体力约 34% 的偏高。
- 纯 L1 探针（`/tmp/l1_freeze_probe.py`）：固体内部 ux `0.06000`（refreeze）
  vs `0.06062`（no-refreeze）；流体区 `max|Δux|=6.3e-3`。
- 60 步双路产物：单卡下游中心列 ux ≈ `0.00003`（强尾迹亏损）vs 集成 `0.06000`
  （=来流值，**完全透明**）；`band_speed` `0.0457` vs `0.0579`（+27%）。
- GPU A/B（同配置 150 步）：`--l1-no-refreeze` `step150=2.0015` vs 现状 `3.0465`
  （**-34%**，且仍在陡降）。

**开关**：`L1BlockDistributed(no_refreeze=True)` / 集成 CLI `--l1-no-refreeze`
（默认 `False` = 历史行为不变；建议在收敛 A/B 确认后把默认切到 `True`）。

**验收 gate 建议**：断言"有固体 + 默认配置 与单卡 `max|d| < 1e-12`"（当前默认不满足）。

---

## e) 未解之谜与剩余方向

### e.1 未解之谜

1. **lattice 敏感（#21）——已量化，D3Q27 侧仍异常**：
   - 单卡 D3Q19（P3 8000 步收敛）`1.2432` ≈ 参考 `1.0917` ✅（此路径正确）
   - 单卡 D3Q27（1200 步，SDAA 死锁修复后可跑）step400 `Cd_mem=0.559343 / Cd_cv=0.560533`
     （mem/cv 自洽）→ **偏低 49%**
   - 集成 D3Q27 `2.7255`（refreeze 路径收敛）/ `2.0015`（no-refreeze step150，陡降中）
   - ⇒ **D3Q19 路径正确，D3Q27 路径单卡偏低、集成偏高——D3Q27 自身有系统性问题**
     （BFL 力链已排除：CV 算子/权重/方向/力学全对）
   - 待解：D3Q27 cumulant 碰撞 vs D3Q19（LES 对照跑中）或 D3Q27 流场演化
2. **单卡 wall 叶层级——已测**：单卡 wall 叶 `{2: 44552}` 全在 level-2，与集成**完全一致**
   （links 266312 亦同）⇒ 力链几何口径差异**排除**。
3. **`f_opp_post`（运动壁）**：CPU 解析共转壁 Tz 残差仍在（d_max=2, Ω=0.002 → `Tz=-77.5`，
   解析应为 0）；V4（标准 `fp_d`）给出 `Tz≈0`。**仅影响运动壁**，静止壁（球体）主路径不受影响。
4. **集成侧 refreeze bug 的收敛影响**：no-refreeze step150 已降 34% 且陡降，
   收敛值待 1200 步跑完确认（若收敛 ≈1.0–1.2 ⇒ refreeze 是主因）。

### e.2 剩余方向（按优先级）

1. **在真实球体阻力基准上跑 V4**：确认运动壁结构修复（`fp_d`）在真实收敛 Cd 上的净效果，
   再决定是否采纳；若采纳需按静止壁/运动壁分支分别处理以免回归 `main` 的 3.0→1.09 修复。
2. **打通单卡 D3Q27 收敛**（解 SDAA 大索引死锁）→ 消除 #21 lattice 未解项，给出同 lattice 对照。
3. **确认单卡 wall 叶层级**，把"面积=力所在 lattice"的口径在单卡/集成两侧钉死。
4. **把 CPU 冻结解析场探针上升为常设 gate**：
   - case A：均匀来流 + 静止壁，plain-BB 恒 0；**d_max 细化不变性**（d1/d2 ratio ≈1，现 0.93）；
   - case B：刚性旋转 + 共转壁，`Tz ≈ 0`（现状 −390.9，明显不合格）；
   - Ω 扫描（分离线性链残差 vs O(u³) 压缩残差）× d_max 扫描。
5. **时变 BFL 力验证**（womersley 锁相法）：给壁速 `u_w(t)=u_w0·cos(ωt)`，取复力幅值与解析对比，
   覆盖 `_ramp_activation` 时变段。

---

## f) 工具与脚本清单

### 本轮新增 / 关键脚本

| 文件 | 说明 |
|---|---|
| `scripts/validate_shell_bfl_force_analytic_cpu.py` | **壳层 BFL 力链解析场 CPU 验证**（case A 均匀场 plain-BB 零基准 + d_max 细化不变性；case B 刚性旋转共转壁解析零力矩；case C 约定间距；含每层力分解与 `2^-2(L-1)` 重标定）。`--dmax 1 2 --omega 0.002 0.004 [--lattice D3Q19/D3Q27]`，CPU ~60s |
| `scripts/validate_shell_bfl_force_analytic_cpu_lattice.py` | 同上 driver 的 **lattice 参数化版**（`LATTICE` 全局 + `--lattice`），用于 D3Q19/D3Q27 对拍 |
| `shell_bfl_force_analytic_report.json` | 上述探针输出工件（逐用例 F/T、plain-BB 参照、ideal-reflection 参照、每层 n_links/F、`Fx_shell_over_ideal_reflection`） |
| `_bfl_force_chain_audit.py` | **分片对拍复核**：均匀解析场上 single-card vs integrated vs 16-rank 分片，Fx ratio 全 **1.0000**（证明记账干净） |
| `examples/octree_integrated_validate.py` | 集成验证主脚本（本轮的 A/B 试验台）。关键开关：`--l1-block/--no-l1-block`、`--d-max`、`--ghost-from-l1`、`--l1-interface-filter[(-width/-strength)]`、`--coarse-freeze`、`--no-coarse-bb`、`--far-field-yz-extrapolate`、`--blockage-correction {simple,glauert,off}`、`--wake-cells`、`--wall-margin`、`--shell-margin`、`--warmup-steps`、`--report-interval`、`--ny/--nx/--nz`（域扩展）。本轮新增 `[area] wall-link leaf levels` 诊断打印（确认集成 wall 叶 44552 全在 level-2） |

### 相关源码（共享模块，单点真相）

| 文件 | 相关 API |
|---|---|
| `src/tensorlbm/octree_boundary/bfl.py` | `bfl_apply_gather`、`leaf_force_weights`、`leaf_force_spatial_weights`、`bfl_ramp_wall_velocity`、`_ramp_activation`（运动壁 `f_opp_post` 结构待修，§d.5） |
| `src/tensorlbm/octree_boundary/force.py` | `substep_force_weights(include_spatial=True)`、`convert_leaf_force_to_l1`（混合层级 fail-closed）、CV clearance gate |
| `src/tensorlbm/octree_boundary/l1_block.py` | `L1BlockDistributed`、`interface_filter=(width, strength)`、`build_window_indices`、`gather_window_chunked`、`write_window_back`、`initialize_from_window` |
| `src/tensorlbm/octree_boundary/stepping.py` | `build_ghost_plan`、`_fill_ghost_impl`、`restrict_shell_to_block`、`build_shell_coarse_links`、`_tau_chain` |
| `src/tensorlbm/octree_boundary/distributed_stepping.py` | `step_octree_shell_distributed`、`split_leaf_bounds`、`interleaved_leaf_indices` |
| `src/tensorlbm/drag_normalize.py` | `compute_wall_link_dx`、`leaf_radius_from_dx`、`dynamic_area`、`compute_blockage_factor`（面积/堵塞**单点真相**） |
| `src/tensorlbm/lbm_re_tau.py` | `tau_from_re(u, L_ref=2R, Re)`（D 口径，Re 单点真相） |
| `src/tensorlbm/amr_interface_filter.py` | `damp_interface_nonequilibrium`（L1 界面滤波，§b-18） |
| `src/tensorlbm/static_block_amr.py` | `StaticBlockAMR3D` 单卡三级参考路径 |

### 诊断脚本族（`scripts/diag_*.py`，本轮使用/关联）

`diag_bfl_tau_chain.py`、`diag_momentum_audit.py`、`diag_octree_freeze.py`、`diag_octree_step.py`、
`diag_shell_rings.py`、`diag_rn_d2_area.py`（面积）、`diag_r6_r8_geometry*.py`（几何）、
`diag_mem_links.py`（力链）、`diag_ghost_*.py`（ghost 供给）、`diag_reflux_probe.py`（reflux）等。

### 复现

```bash
cd /root/TensorLBM_feat2

# 1) 壳层 BFL 力链解析场探针（d_max 细化不变性 + 共转壁解析零力矩）
python scripts/validate_shell_bfl_force_analytic_cpu.py --dmax 1 2 --omega 0.002 0.004   # CPU ~60s
python scripts/validate_shell_bfl_force_analytic_cpu_lattice.py --lattice D3Q19
python scripts/validate_shell_bfl_force_analytic_cpu_lattice.py --lattice D3Q27

# 2) 分片对拍（Fx ratio 应全 1.0000）
python _bfl_force_chain_audit.py                                                          # CPU ~1min

# 3) 集成收敛 + A/B（示例；需 GPU/分布式环境）
python examples/octree_integrated_validate.py --geo sphere --radius 6 --reynolds 100 \
    --l1-block --d-max 2 --steps 2000 --warmup-steps 50 --report-interval 50 \
    [--ghost-from-l1] [--l1-interface-filter] [--coarse-freeze] [--far-field-yz-extrapolate]
```

---

## 附：本轮 commit 时间线（`feat2-octree-integrated`）

| commit | 内容 |
|---|---|
| `18047d2` | A/B：空间因子精确 4x；`--ghost-from-l1` −5~−11%；L1 界面滤波无效果；单卡/集成面积一致；单卡 D3Q27 被 SDAA 死锁阻断（**本轮 HEAD**） |
| `5224aa2` | V4 A/B：ghost donor 源非根因（3.05→2.89）；V3 L1 界面滤波实现（CPU 位一致/守恒 4.4e-16） |
| `3e97374` | BFL 运动壁系数核查：`-6/-(3/q)` 正确；结构 `f_opp_post` vs `fp_d` 是错；变体扫描 V1–V6 |
| `18eedca` | **FIX 混合层级力折算** `2^-2(level-ref)`（d1/d2 比 3.71→0.93）；新增解析场 CPU 探针 |
| `2a6489f` | coarse-freeze 无效果；迁移 `lbm_re_tau`/`drag_normalize`；单卡 0.54 是未收敛瞬态（收敛 R6=1.2432, R10=1.1093） |
| `0a2583f` | **FIX P0 ghost lev 失配**（neq 注入 2.183→0.988） |
| `81cc844` | **FIX Re 口径**（R→2R，曾跑在 Re=200） |
| `dcf899e` | **FIX 面积**（wall-link 叶 dx，非计数加权均值；R10 d2 Cd 9.33→3.75） |
| `69d027c` | ghost 深度无关（g1=g3）；SUBOFF L1 P0；堵塞升级；分片复核 |
| `6c71a62` | L1 init 从 coarse window 采样（neq 注入） |
| `9e3184f` | 单卡 R6 d2 基线揭示双向偏差；`--far-field-yz-extrapolate` |
| `97385a6` | 共享模块 `lbm_re_tau`/`drag_normalize`/`octree_shell_runner`；bl 3→6 不改链接数（几何排除） |