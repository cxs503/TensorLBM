# 交差审计 · m17863 — XVAL-LBM-INDIRECT-ADDRESSING-WALL-TABLES（TensorLBM）

**审计单** m17967（verdict_by=`board-audit`） · 被审单 m17863（`succeeded`） · 审计日 2026-11-13
**被审交付树** `/root/tp-agent/work/m17863`（`git status`：5 组交付物**全部未跟踪**，`head_commit=None`）
**审计者独立工件** `artifacts/xval_repro/`、`artifacts/probe_wall_force_component.{py,json}`、
`artifacts/probe_link_sets.{py,json}`、`artifacts/m17967_numbers.{json,md}`、`artifacts/figures/fig1..3*.png`

## 0. 裁定（先说结论）

| 项 | 结论 |
|---|---|
| **级别** | **L2**（独立功能实现）。L3 系统集成缺：`wall_tables.py` 无任何主求解器/稠密 BFL 调用方；文件甚至未进仓（未提交）。 |
| **基准判据** | `tensor_benchmark_cases` 里该 anchor **原本 0 行**（交付时未登记，见 §4-F5）→ 本审计登记 2 行（§5）并用它们逐项打分。 |
| **物理 KPI** | u_max/k_super/τ_w(剖面) 全落在 <15% 带内（最大 9.50%@H8），但 **3/8 项 >3% 生态线**；预注册的 H=32 一级未跑（如实披露）。 |
| **发现数** | 2 项 major（1 项报告过度声明、1 项能力被错误 WITHHELD 且未修）+ 4 项 minor。**非造假**：无伪造数字，所有数字可复算，报告 §⑥ 已如实列出两项 FAIL。 |
| **处置** | `approve with notes`（L2 确认）+ 请求修改 2 处（F1 一行索引、F2 文档叙述），返修动作由 maturity_gate 自动裁定。 |

## 1. 独立性声明与审计范围

* 我未参与 m17863 实现；本次只读其工作树 + `tensor_benchmark_cases` + OSS 源码 `/tmp/mflbm_build/src`（codeload tarball，未被修改）。
* 限制：① OSS 二进制未执行（CUDA/MPI/OpenACC，无 GPU）——因此"vs OSS 数值"这一层**不存在**，只有"vs 解析解"与"vs OSS 源码语义"两层；② 我复跑在同一台 99 机上，**不能**复核 m17863 记录的 120 s 单命令预算声明本身；③ 未审 m17863 的 agent 会话日志，只审工件。
* 未运行的检查（记 `not_run`，不记 passed）：H=32 收敛级；OSS 二进制执行；`wall_tables.py` 与 TensorLBM 主求解器联合仿真（不存在调用方）；GPU 路径。

## 2. 执行路径重建（代码读法）

`build_wall_tables(walls, periodic_axes)` → 分类（1 固体/0 流体/2 固体边界/-1 流体边界）→ `solid_boundary.neighbor_list + la_weight`（CSR）→ `fluid_boundary.normal`（ISO8，默认投影式）→ 同一次遍历产出 `links(fluid, solid, dir, q)` → `la_weight_extrapolate` 消费节点表；`run_arm.py` 用 `links` 驱动半步反弹（半程 BB），用 `Σ2 c_i f_i` 做动量交换壁面力，并用"稠密掩码扫描"做 A2 对照。

## 3. 取证与复现（我跑了什么）

| 检查 | 命令 | 结果 |
|---|---|---|
| 单测 | `python3 -m pytest tests/test_wall_tables.py -q`（在 m17863 树） | **13 passed / 58.85 s**，38 条 assert（OSS 分类规则、la_weight=Σw_equ、链接几何一致、稠密暴力交叉、线性场重心 1e-12、平面/球面法向、OSSP 逐字式各向异性、占用上界、非法输入、退化几何） |
| 复跑全臂 | `python3 xval/.../run_arm.py --out ../m17967/artifacts/xval_repro` | **54.6 s 完成**（<120 s 预算，`repro_cmd` 可执行 ✅）；结果与交付 `results.json` 逐字段相同，仅 3 个计时字段不同（表遍历 0.142→0.156 ms、稠密 2.506→3.125 ms）→ **可复现 ✅** |
| 基准打分 | 见 §5 两张表（本审计登记的 `tensor_benchmark_cases` 行） | 物理臂 8/8 在 15% 内，3 项 >3% |
| 外部锚点抽检 | `sed -n 89,104p Module.F90`；`Geometry_preprocessing.F90:126-141`；`Phase_gradient.F90:14-28` | **三条 file:line 引用逐字成立** ✅（类型字段 `ix,iy,iz,i_fluid_num,neighbor_list(1:18),la_weight`、分类标 2/-1、`Σ w_equ(ie)·φ/la_weight` 全部对上） |
| 交付物存在性 | 见 §5 evidence | impl/entry/run_artifacts 全部在盘 ✅；**但均未 git 提交** ⚠ |

## 4. 分级发现（按严重度）

### F1 — major（报告过度声明）：A1/A2「逐位相同 ✅（预注册 1e-12）」与自家工件矛盾
* **声明**：`docs/XVAL_m17863_*.md` §④ 表「表示等价…判定（预注册 1e-12）✅ 逐位相同」、§⑤「表/稠密两路逐位相同」、误差判定表「表/稠密表示等价 0.0e+00 ✅」。
* **反证（交付自己的工件）**：`artifacts/xval_indirect_addressing_wall_tables/results.json` → `identity_A1_vs_A2.H8.within_predeclared_1e_12 = false`、`bitwise_identical = false`、`wall_force_rel_diff = 9.115e+286`（H16 为 3.065e+287）；复跑同一 harness 时终端就打印 `[identity] {"H8": false, "H16": false}`。
* **根因（我的定位）**：A2 稠密臂的壁面力累加取 `run_arm.py:183-184` 的 `jj == 0` / `jj == shape[1]-1`，而这两个 j 是**固体行**，掩码在那里恒为空（`artifacts/probe_link_sets.json`：`dense_force_sites.selected_any = false`，链接只出现在 j=1..H）→ `force_dense ≡ [0,0]`，再用它做分母得 `9.1e+286`。即：**流场差 0.0 是真的（链接集也真的相同，40 vs 40，集合相等），但"预注册 1e-12 判决"在壁面力这一项上是 FAIL，报告只引用了通过的那一半。**
* **修复**：`jj == 1` / `jj == shape[1]-2`（或直接用 `links.solid_flat` 的壁面归属），并把 `bitwise_identical/within_predeclared_1e_12` 原样写进报告；`wall_force_rel_diff` 在分母为 0 时应输出 `None` 而非 1e286（该字段本身是"以零为分母"的病态指标）。
* **性质**：选择性引用（不是伪造数字）。若被下游当作"两条独立实现互相验证"，会高估等价性强度——链接集相同已由我独立核对为真，故 A1≡A2 的结论方向仍成立，只是强度是"同一集合的两种遍历"，不是独立实现互证。

### F2 — major（能力被错误 WITHHELD + 未修）：动量交换壁面剪力其实成立到 1e-6，报告归因错误
* **声明**：`docs` §⑥ 诚实 FAIL 第 1 条：`tau_w`（动量交换路线）**WITHHELD**，"链接集只含指向壁面的方向时不能直接当剪切用"，两壁相消到 7.7e-14。
* **反证（我的探针 `artifacts/probe_wall_force_component.py`，同一 `links`、同一 `f_post`）**：

| 臂 | 解析 τ_w=gH/2 | 交付 KPI（投影 c_y=壁法向） | 交付相对误差 | 审计改投 c_x（流向） | 相对误差 |
|---|---|---|---|---|---|
| H8 τ=0.8 | 4.000000e-05 | 2.2742e-14 | 100.0000% | 3.999999e-05 | **1.438e-07** |
| H16 τ=0.8 | 8.000000e-05 | 7.6565e-14 | 100.0000% | 7.999987e-05 | **1.602e-06** |

* **根因**：`run_arm.py:168` `per_link = 2.0*(vals - W64[link_dir]) * C64[link_dir, 1]` 取的是 **y（壁法向）** 分量；直通道里法向合力按对称性本就为 0。剪力是 x 分量（`C64[link_dir, 0]`）。dense 臂同错（`:182`、`:210`）。
* **后果**：交付把"本单最有价值的输出"（链接表直接给出壁面剪力/受力，且精度到舍入级）记成"不可用"，改用剖面差分（4.75%@H16、9.50%@H8）——**能力被系统性低估，且未修**。这是本次审计最实质的发现：一行索引之差（`1`→`0`），换来 4.75%/9.50% → 1e-7/1e-6。
* **性质**：诚实 FAIL 的方向没错（不该把 100% 误差记成通过），但**归因错误**（不是链接集的极限，是投影轴写错），且修复成本极低而未修。

### F3 — major（L3 缺口）：无集成调用方，且交付未入仓
* `grep -rn wall_tables --include=*.py` 在 TensorLBM 树上只命中 `tests/test_wall_tables.py` 与 `xval/.../run_arm.py`；主求解器（`bfl_common.py` 稠密 BFL 路径、`d3q19` 步进器）**无调用**。TensorLBM 当前 main（`m17967` 检出 = `222585d6` 线）里**根本不存在** `src/tensorlbm/wall_tables.py`。
* m17863 的 `result.git_status` 与本次现场 `git status` 一致：5 组交付物全是 `??`，`head_commit=None` → 交付只活在它自己的工作树里，未提交、未入仓、未推送。因此"平台侧可见能力"为 0，L3 无从谈起。

### F4 — minor（KPI 覆盖）：3/8 项越过 3% 生态线；H=32 未跑
* `k_super(H8)=8.05%`、`u_max(H8)=2.38%`、`τ_w(剖面,H8)=9.50%`、`τ_w(剖面,H16,τ0.6)=6.21%`、`k_super(H16,τ1.0)=3.27%` → 全部 <15%（owner 带内），但生态 <3% 线只被 H16/τ0.8 的 u_max 0.59%、k 1.65%、profile 0.20% 满足。报告误差判定表已如实标 ❌，不构成虚标。
* 预注册 H1/H3 含 H=32；未跑（报告 §⑥.2 如实披露，`predeclared.json` 存 amendment）。**收敛阶只有两点**，"一阶收敛"是两点插值而非拟合（不得当作已验证）。

### F5 — minor（基准登记缺失）：`tensor_benchmark_cases` 交付时 0 行
* `SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-INDIRECT-ADDRESSING-WALL-TABLES'` → 0 rows（锚点共 66 个 slug、75 行）。基准判据无从裁判，故本审计按 m17888 先例补登记 2 行（§5），并在行内标注 `registered by audit m17967`。
* 同时：`docs/XVAL_m17863_*.md` 引用的 `docs/oss-inventory/oss-inv-lanl-MF-LBM.md` **在交付树中不存在**（`ls docs/ | grep oss` 无命中），OSS 侧唯一证据链是 `/tmp/mflbm_build/src`（仍在盘，我抽检 3 条 file:line 全部成立 ✅）。OSS 锚点成立，但引用的盘点文档缺件。

### F6 — minor（数值/语义）：τ=0.6 臂未达稳态；连续化节点值 192%
* `channel_A1_H16_tau0.6`：`steady_at = null`（5000 步用尽，`STEADY_TOL=1e-6` 未达），其 KPI（u_max 0.92%）来自**未判稳**状态 → 该臂应标"未收敛"，不能与另两臂并列为收敛结果（报告未区分）。
* `continuation.output_vs_exact_at_node_rel = 1.92`（192%）：`results.json` 自带 note 说明算子只承诺壁面重心值，文档也标"语义不同，非数值误差"。结论：**不记为误差，但该字段名（rel error）易误读**，应改名 `node_value_semantic_mismatch`。

## 5. 基准打分（本审计登记的 `tensor_benchmark_cases` 两行）

### 5.1 CH-POISEUILLE-ANALYTIC-A1（解析平面 Poiseuille）— 3 项态：**passed**

| 臂 | u_max got / ref | 误差 | k_super got / ref | 误差 | τ_w(剖面) got / ref | 误差 | steps / steady |
|---|---|---|---|---|---|---|---|
| H8 τ0.8 | 7.809999e-04 / 8.000000e-04 | 2.3750% | 4.6100 / 4.2667 | 8.0469% | 3.6200e-05 / 4.0000e-05 | 9.5000% | 1000 / 1000 |
| H16 τ0.8 | 3.180994e-03 / 3.200000e-03 | 0.5940% | 19.2770 / 18.9630 | 1.6541% | 7.6200e-05 / 8.0000e-05 | 4.7502% | 3400 / 3400 |
| H16 τ0.6 | 9.511405e-03 / 9.600000e-03 | 0.9229% | 18.9940 / 18.9630 | 0.1616% | 7.5031e-05 / 8.0000e-05 | 6.2113% | 5000 / **None** |
| H16 τ1.0 | 1.914999e-03 / 1.920000e-03 | 0.2605% | 19.5830 / 18.9630 | 3.2714% | 7.8333e-05 / 8.0000e-05 | 2.0834% | 2200 / 2200 |

### 5.2 WALL-SHEAR-MOMENTUM-EXCHANGE（链接表直接给壁面剪力）— 3 项态：**pending**（交付路径 FAIL，改轴后 PASS）

| 臂 | 解析 τ_w | 交付 KPI(c_y) | 误差 | 审计 c_x | 误差 |
|---|---|---|---|---|---|
| H8 | 4.000000e-05 | 2.2742e-14 | 100.0000% | 3.999999e-05 | 1.438e-07 |
| H16 | 8.000000e-05 | 7.6565e-14 | 100.0000% | 7.999987e-05 | 1.602e-06 |

## 6. 级别判定（L0–L5，机器判据见 `/root/tp-agent/maturity_gate.py` SCHEME）

| 级 | 判据 | 本单是否满足 | 证据 |
|---|---|---|---|
| L0 文档 | — | 是（但不止文档） | `docs/XVAL_m17863_*.md`（六段齐全） |
| L1 数学/单元 | tests 非空 | ✅ | 13 passed / 38 assert / 58.85 s |
| **L2 独立功能** | impl 非占位 + entry + run_artifacts | ✅ | `src/tensorlbm/wall_tables.py`（27 134 B，`build_wall_tables:394`、`iso8_wall_normal:325`、`la_weight_extrapolate:553`、`dense_la_weight_extrapolate:606`）+ `run_arm.py` + `results.json/table.csv/fig1-5.png` |
| L3 系统集成 | + integration_callers | ❌ | 主求解器/BFL 无调用方；文件未提交入仓 |
| L4 基准验证 | + benchmark + repro_cmd | 物理臂形似 L4（解析 ref + 误差分析 + 54.6 s 复现），但门禁要求先过 L3 → **不予主张** | §5.1 |
| L5 科研工程 | + external_anchor | 外部锚点（OSS file:line）成立，但 L4 未达 → 不予主张 | `/tmp/mflbm_build/src/{Module,Geometry_preprocessing,Phase_gradient}.F90` |

**裁定：L2**（"独立功能实现"实质：真代码 输入→计算→输出，且自证到解析解；未进主求解器）。

## 7. 需要返修的动作（建议，供 maturity_gate 返修单引用）

1. `run_arm.py:168,182,210`：动量交换力投影轴 `C64[·,1] → C64[·,0]`；同步修 dense 臂 `jj==0/-1 → jj==1/-2`；把 τ_w(动量交换) 升为第一类 KPI（预期 1e-6 级）。
2. 报告 §④/§⑤：把 `within_predeclared_1e_12=false` 原样写出，删掉"✅ 逐位相同"的预注册判决口径（改述为"流场 0.0 相同；壁面力项因 dense 臂累加部位错误未计分，audit m17967 已定位"）。
3. 把 `wall_tables` 接进主求解器（`bfl_common` 的稠密 BFL 近壁算子走表遍历），才有 L3；并 `git add` 交付物（当前全 `??`）。
4. 补 H=32 一级（≥3 点收敛阶）与 τ=0.6 臂的稳态判据；`steady_at=null` 的臂不得计入收敛表。
5. `docs/oss-inventory/oss-inv-lanl-MF-LBM.md` 补进仓（当前引用了不存在的路径）。