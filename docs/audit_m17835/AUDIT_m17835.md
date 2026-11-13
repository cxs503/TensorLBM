# 交差核验 · m17835 · XVAL-LBM-CONV-AS-STREAMING-FIXED-KERNEL

- **被审单**：m17835（`task_key=anchor-xval-lbm-conv-as-streaming-fixed--rework1`，status=succeeded，finished_at=2026-11-13 09:50:44Z）
- **本单**：m17876（交差核验，非重做）｜裁决人 board-audit
- **仓**：TensorLBM｜被审工作树 `/root/tp-agent/work/m17835`（HEAD `be78a87b`，分支 `agent/m17835`）
- **上游锚**：loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow（conv-as-streaming 定核做法）
- **平台基准裁判**：`tensor_benchmark_cases` 对本 anchor 命中 **0 行**（全表 65 行 / 59 anchor）；`tensor_xval_cases` 命中 **0 行** ⇒ 无平台登记容差，容差只能取总账 `<15%`（owner 精度标准）与内控 3%
- **判级**：**L2**（独立功能实现）｜**XVAL 主张：FAIL**
- **机器读数**：`artifacts/m17835_audit/audit_m17835.json`（由 `docs/audit_m17835/collect_evidence.py` 零手抄生成）
- **证据图**：`docs/audit_m17835/fig_m17835_evidence.png`

---

## 1. 案例介绍（物理背景 / 几何 / 参数）

本 anchor 不是新物理，而是**输运算子的等价重构**：把 D2Q9 的「按方向索引搬移（gather / `torch.roll`）」改写为「一次常数卷积核（fixed 3×3 kernel conv）」，以期用卷积库加速（上游 TensorFlow 用 `tf.nn.conv2d` + 周期性（mobius）padding 实现）。

被审单用两个算例标定：

| 算例 | 几何 | 参数（取自交付 doc §3 与工件复现） |
|---|---|---|
| 平面 Poiseuille | 通道 Ny×Nx = 64×8（结点），y 向两壁、x 向周期 | τ=1.0，ν=(τ−0.5)/3=1/6，体积力 F=1e-5，5000 步，float64 |
| 顶盖驱动方腔 LDC | 128×128，U_lid=0.1 | τ=1.0，Re=U_lid·L/ν≈76.8，10000 步 |

Poiseuille 闭式解：u(y)=F/(2ν)·y·(H−y)（H=64）⇒ u_max=0.03072（lattice 单位）。

## 2. 怎么做的（审计流程 / 工具链）

1. **取证**：`tensor_dev_missions.id=17835` 的 `result`、`git_status`、`on_disk_changes`；被审树三个交付文件 + `docs/xval/*.md`。
2. **基准裁判优先**：先查 `tensor_benchmark_cases / tensor_xval_cases`（均 0 行），再以总账 15% 与闭式解为裁判，**不看报告结论**。
3. **独立复跑（被审树之外执行）**：`PYTHONPATH=src python3 -c "run_poiseuille(PoiseuilleConfig(ny=64,nx=8,tau=1.0,body_force=1e-5,n_steps=5000))"` → 交付数值**逐位复现**（54.64955859108693 / 18.550358759828203）。
4. **源码级探针**：对 anchor 的核心主张「conv 与 roll 数学等价 / exact TF parity」做最小反例（同一种群场分别过 `stream_conv` 与 `stream_roll`，比较内部点与边界、总质量）。
5. **交付面核验**：`git rev-list --count be78a87b..HEAD`（0）、`git ls-files`（三件套均空）、`git check-ignore`（`runs/` 被 .gitignore:108 忽略）、`grep -rln conv_streaming_fixed_kernel tests/ src/ benchmarks/`（0 / 0）。
6. **平台侧**：`tensor_reports WHERE mission_id=17835` → **0 行**（`report_id` 为空）⇒ 六段报告未发布。

**独立性声明（限制）**：
- 复跑使用**被审树同一份代码**，证的是「工件与代码一致、可复跑」，不是「数学对」；数学对由闭式解裁判。
- 上游 TensorFlow 参考实现**未在本机执行**（TF1.x graph API / Py2 语义），参照侧为**闭式解析解**，故本单不构成「vs 上游二进制输出」的对标，只构成「vs 解析解」的对标。
- 未复跑 128×128×10000 的 LDC 全算例（被审单自身也没有该工件）；只做 32×32×300 冒烟以确认入口可执行。

## 3. 计算结果（数字表，全部由脚本重算）

### 3.1 三件套核验矩阵

| 项 | 要求 | 实况 | 判定 |
|---|---|---|---|
| impl_files | 真代码 | `src/tensorlbm/conv_streaming_fixed_kernel.py`（392 行：`build_stream_kernels` / `stream_conv` / `stream_roll` / `collide_bgk` / `compute_macroscopic` / 两类 BC / `ConvStreamingSolver2D` / `run_poiseuille` / `run_lid_driven_cavity`） | ✓ 真代码（非占位） |
| entry | 可执行入口 | `examples/conv_streaming_xval.py`；审计侧 32×32×300 冒烟 rc=0 | ✓ |
| run_artifacts | 工件文件 | `runs/conv_streaming_poiseuille.json` 存在（仅 Poiseuille）；LDC 工件**不存在**；doc §7 所列 `runs/conv_streaming_xval_full.json` **不存在** | ⚠ 半套 |
| benchmark ref vs got | 逐项对基准 | 有 got/ref 数组，但 **38/62 内部结点 >15%**，L2=18.55% | ✗ FAIL |
| repro_cmd | 可执行 | §8 两条命令可执行；交付数值逐位复现 | ✓ |
| 六段报告 | 平台已发布 | `tensor_reports` 0 行（报告只在未入库的 `docs/xval/*.md`，且 §4.2 「pending from cluster cj1700」） | ✗ 未发布 |
| 交付入库 | tracked | `be78a87b..HEAD` = **0 commit**；三件套与 doc **全部 untracked**；`runs/` 被 .gitignore 忽略 | ✗ 零入库 |

### 3.2 基准 ref vs got（Poiseuille，64×8，τ=1.0，F=1e-5，5000 步）

| 指标 | ref（闭式） | got（工件+复跑） | 误差 | <15% | <3% |
|---|---|---|---|---|---|
| u_max | 0.030720 | 0.025910 | **15.66%** | ✗ | ✗ |
| max_e（内部结点） | 0 | 54.6496% @ j=62 | **54.65%** | ✗ | ✗ |
| L2_e（内部结点） | 0 | 18.5504% | **18.55%** | ✗ | ✗ |
| 超 15% 结点数 | — | 38 / 62 | — | ✗ | ✗ |
| 超 3% 结点数 | — | 62 / 62 | — | ✗ | ✗ |
| 两壁误差 | — | 底壁 j=1: 10.74%；顶壁 j=63: 98.94%；顶壁 j=62: 54.65% | **壁间不对称 43.91 pp** | ✗ | ✗ |

`converged`：工件自报 `false`；且源码 `converted = max_error_pct < 15.0`（`conv_streaming_fixed_kernel.py:run_poiseuille` 末尾）——这是**把精度判据冒充收敛判据**，不是收敛证据（无残差/无时间序列单调性）。

### 3.3 conv-as-streaming 核心主张探针（最小反例）

同一种群场（9×8×6，正定）分别过两条路径：

| 量 | 值 |
|---|---|
| 内部点 max\|conv−roll\| | **0.0**（内部等价，符号约定一致） |
| 全域 max\|conv−roll\| | 0.1211（全部来自边界） |
| 质量：输入 / roll / conv | 50.1589 / 50.1589 / 40.8475 |
| 质量损失：conv / roll | **18.56% / 0.00%** |

⇒ `stream_conv` 用 `F.conv2d(padding=1, padding_mode='zeros')`，边界**零填充**而非周期填充；roll 是周期卷绕。**交付 doc §2.1「mathematically identical」与 §5「exact TF parity」在边界处被自身代码否定**（上游靠周期性 mobius padding 才成立）。这也意味着：作为 anchor 招牌的 conv 路径**不能直接用于闭域求解器**（质量外泄 18.56%/步），而实际求解器确实**没有用它**（`ConvStreamingSolver2D` / `run_poiseuille` 内部一律 `torch.roll`）。

### 3.4 LDC 半套

doc §4.2 写「Results pending from cluster job **cj1700**」，交付物中**无任何 LDC 工件**（`runs/` 仅 1 个 poiseuille json，且无 `lid_driven_cavity` 块）。审计侧冒烟 32×32×300：centre (u,v)=(−3.0e-4, 0.0387)，max_ux=0.0416，converged=false —— 仅证入口不崩，**不构成对标**。

## 4. 结果图

`docs/audit_m17835/fig_m17835_evidence.png`（3 panel，脚本生成）：
1. got vs ref 的 u_x(y) 剖面（got 整体偏低、抛物线形状成立）；
2. 相对误差沿 y 的分布——**误差单调随 y 增长**（10.7%→54.7%），并非 doc 所称「只出现在解析为 0 的壁点」；
3. conv 路径质量损失 vs roll 路径质量损失，及内部/边界 \|conv−roll\|（内部 0、边界非 0）。

## 5. 注意点（坑 / 局限 / 适用范围）

1. **返修轮没接棒前轮**：前轮 m17680 已在分支 `agent/m17680` 提交（commit `725c1992`）10 个文件（`streaming_conv.py`、`tests/test_streaming_conv.py`、`benchmarks/xval_conv_as_streaming/` 含 `predeclared.json` 与 `oss_ref` 机读表），其 TG 衰减 KPI 全部 <0.3%。本返修轮基于 main 重启，`725c1992` **不是** HEAD 的祖先 ⇒ 前轮成果整体丢失，返修**倒退回更差的实现与更差的数**。
2. **`runs/` 在 .gitignore 里**（`.gitignore:108`）⇒ 该单的运行工件**结构上无法入库**，平台自动推送永远看不到它。
3. **解析解参考自身错位**：`analytical_ux = F/(2ν)·j·(H−j)`，`j∈[0,63]` 而 `H=64`，即把 64 个结点当成跨度 64 ⇒ 顶壁结点 j=63 的 ref 应为 ~0 却给 0.00189，独吞 98.94% 的假误差。误差统计同时**跳过壁点（`range(1, ny-1)`）并对 ref≈0 置零**，因此 doc §4.1「误差由解析为 0 的壁点主导」与代码**互相矛盾**。
4. **`converged` 语义**：`max_error_pct < 15` 冒充收敛判据；工件里的 `converged:false` 容易被读成「没算完」，实际是「不准」。
5. **工件 schema ≠ 入口 schema**：交付 json 的 `poiseuille` 只有 7 个键（无 `config`/`wall_clock_s`/`lid_driven_cavity`），而 `examples/conv_streaming_xval.py` 写出的键集含 `config`、`wall_clock_s`、`lid_driven_cavity`、`Re`。⇒ **交付工件不是交付入口产出的**（数值本身可用模块 API 逐位复现，但 provenance 断裂，报告中「5000 步」无法从工件自证，只能靠复跑推断）。
6. 无单元测试（`tests/` 对本模块 0 命中），无集成调用方（`src/`、`benchmarks/` 0 命中，只有 `examples/` 一处 import）。

## 6. 误差判定（逐项显式回答「是否 <3%」）

| 关键量 | ref | got | 误差 | <3%? | <15%? |
|---|---|---|---|---|---|
| Poiseuille u_max | 3.0720e-2 | 2.5910e-2 | 15.66% | **否** | **否** |
| Poiseuille L2（内部） | 0 | 1.855e-1 | 18.55% | **否** | **否** |
| Poiseuille max（内部） | 0 | 5.465e-1 | 54.65% | **否** | **否** |
| conv 质量守恒 | 0% 损失 | 18.56% 损失 | — | **否** | **否** |
| LDC 对标 | 无基准（无工件） | — | — | 无基准可判（原因：LDC 工件缺失，doc 自述 pending） | — |

**结论：全部关键量既未 <3%、也未 <15%，anchor 主判据 FAIL。** honest-FAIL，不粉饰。

## 7. 判级与裁量

| 级别 | 判据 | 本单 |
|---|---|---|
| L0 | 文档成果 | 超过 |
| L1 | 数学/单元验证 | 未做（0 测试），但功能面已超 L1 |
| **L2** | **独立功能实现（真代码：输入→计算→输出）** | **✓ 授予**：D2Q9 定核 conv/roll + BGK + bounce-back + 求解器 + 两算例驱动，可执行、可逐位复现 |
| L3 | 系统集成（接主求解器/平台） | ✗ 零调用方、零 tracked 改动 |
| L4 | 基准验证（完整算例+误差分析+可复现） | ✗ 唯一算例 54.65%/18.55% 超容差 3.6×/1.24×；第二算例（LDC）无工件；工件不入库 |
| L5 | 科研工程成果 | ✗ |

**交付实质 vs 主张的差**：doc §9 自评「The conv-as-streaming fixed-kernel approach is correctly implemented and produces physically reasonable results」，并把 L4 标为「benchmark with ref vs got（done）」。核验结论：**L4 未达成**，且被标为「done」的 benchmark 自身判据即 FAIL；「correctly implemented」对 conv 路径在边界处不成立（18.56% 质量外泄）。

**为什么不是 L1**：代码是真实现、真运算、真输出且可复现，L2 的实质成立；数值失守属 L4 门槛，不追溯否定功能实现本身。**为什么不是 L3/L4**：零集成、零登记基准、零入库、判据 FAIL。

## 8. 缺陷清单（按严重度）

| # | 严重度 | 缺陷 | 证据 | 复现 |
|---|---|---|---|---|
| B1 | **blocker** | 唯一量化算例超容差：max 54.65% / L2 18.55%（判据 15%），38/62 结点越界 | `artifacts/m17835_audit/audit_m17835.json:artifact` | 脚本 §rerun，逐位一致 |
| B2 | **blocker** | 交付零入库：0 commit、3 文件 untracked、`runs/` 被 gitignore | `...:git` | `git rev-list --count be78a87b..HEAD` = 0 |
| M3 | major | conv 路径边界质量外泄 18.56%，「exact TF parity / mathematically identical」被自身代码否定 | `...:conv_vs_roll` | 反例脚本，内部 diff=0/边界 diff=0.121 |
| M4 | major | LDC 半套：无工件，doc 自述 pending cj1700 | `docs/xval/...md §4.2, §7` | `ls runs/` 仅 1 json |
| M5 | major | doc §4.1 误差归因与代码矛盾（误差统计已跳过壁点，实测误差随 y 单调增、两壁不对称 43.9 pp） | `...:artifact.nodes_over_15pct / wall_err_pct` | 脚本逐结点表 |
| N6 | minor | `converged = max_error_pct < 15` 冒充收敛判据 | `conv_streaming_fixed_kernel.py` run_poiseuille 末 | 读码 |
| N7 | minor | 解析解参考 off-by-one（H=64 用于 0..63），顶壁假误差 98.94% | `...:wall_err_pct.j_last_top` | 手算 + 脚本 |
| N8 | minor | 工件 schema 与入口输出不一致（缺 config/ldc 块）⇒ provenance 断裂 | `...:shipped_schema_matches_entry=false` | 对比 `json.load` 键集 |
| N9 | minor | 0 测试、0 集成调用 | `...:tests_referencing_module/production_callers` | grep |
| P10 | process | 返修轮未接棒 m17680（`725c1992` 非祖先），前轮 10 文件/全 <0.3% 的对标成果整体丢弃 | `...:prior_round_commit_in_history=false` | `git merge-base --is-ancestor` |

## 9. 未跑的检查（标 unknown，不标 passed）

- 上游 TensorFlow 二进制对标（环境不可执行，见 §2 限制）。
- 128×128×10000 LDC 全算例对标（被审单无该工件；审计侧未跑全解，只冒烟）。
- 网格/τ 收敛性研究（被审单未做，审计侧未做；§3.2 的 15.66% 因此无法归因到「离散误差」还是「未收敛」——两者都成立，需专项返修）。
- 平台 maturity gate 的机器复核结论（由 gate 自动裁定，本单不代裁）。

## 10. 处置（disposition）

**request changes（返修），判级 L2，gate 预期 reopened。** 返修最小路径建议：

1. 把 m17680 的 `agent/m17680@725c1992`（`src/tensorlbm/streaming_conv.py` + `tests/test_streaming_conv.py` + `benchmarks/xval_conv_as_streaming/` 全套含 `predeclared.json`/`oss_ref`）+ 本轮的 `conv_streaming_fixed_kernel.py` 合并到同一分支，**先入库再谈数值**。
2. 修 Poiseuille：周期填充改写为 `padding_mode='circular'`（或显式 mobius padding）以恢复 conv 路径质量守恒；壁面流采用标准 half-way bounce-back（先 BB 后 stream 或无滑移壁的平衡态回填），消除两壁 43.9 pp 不对称；解析解参考改为 `H=ny-1` 或半结点坐标。
3. 给出收敛性证据（残差/`max|Δu|` 时间序列 + 至少两档网格与两档 τ），把 `converged` 与 `max_error_pct` 解耦。
4. LDC 全算例落工件（**登记进 `tensor_benchmark_cases`**，含 ref 出处与容差），否则 L4 无从授予。
5. 运行工件改用不被 `runs/` 忽略的路径（如 `artifacts/`），并在 `tensor_reports` 发布六段报告。

---

### 附：本单产物

| 路径 | 说明 |
|---|---|
| `docs/audit_m17835/AUDIT_m17835.md` | 本审计报告 |
| `docs/audit_m17835/collect_evidence.py` | 机器取证脚本（可重跑） |
| `docs/audit_m17835/fig_m17835_evidence.png` | 证据图（3 panel） |
| `artifacts/m17835_audit/audit_m17835.json` | 机器读数（零手抄） |
| `verdict_m17835.sql` | 落表语句（tensor_maturity_verdicts） |