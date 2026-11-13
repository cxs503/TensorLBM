# AUDIT m17677 — XVAL-LBM-COMPRESSED-FLUID-NODE-NEIGHBOUR-TABLE（交差核验）

- 被审 mission：`17677`（succeeded，`budget_exhausted+forced_finish`），worktree `/root/tp-agent/work/m17677`
- 审计 mission：17825（交差核验单，非重做）；`verdict_by='board-audit'`
- 被审树 HEAD：`9bcc6db6 Merge #2023 (m17533) via push-merge`；交付全部为未跟踪文件（6 条 `??`）

## ① 案例介绍

对象是 openLBMPM（PorousMediaSimulation）的 *compressed fluid-node neighbour table* 语义：
多孔介质里只有流体节点带未知量，把 9 个分布函数压实到 `K` 个流体节点，并保留一张
`(K,8)` 的邻居表把「流体节点 + 方向 → 邻居的压实索引（或 -1=固体）」映射好，
流式改为 push/scatter + 半程反弹。XVAL 的设计是双锚：(a) 表示等价性——压缩表 vs 仓内稠密全格在
同一时间步下浮点逐位一致；(b) 物理锚点——平面槽道渗透率 vs 解析 `k_super=(H/ny)H²/12`。

| 参数 | 值 | 来源 |
|---|---|---|
| 碰撞/forcing | D2Q9 BGK + Guo 体力，tau=0.8，nu=0.1000 | `results.json` |
| 精度 | float64，单线程 torch 2.7.1，numpy 2.4.6 | `results.json` |
| 步数预算 | 1200（`steady_tol=1e-09`） | `results.json` |
| 算例 | C1 槽道 H=16/ny=32；C2 方形颗粒 p=0.90/0.75/0.60 N=64；C5 g×2；C6 N=96 | `results.json` |
| 预声明（PLAN） | 槽道 H=32/ny=64、N=128/256、12000 步 | `xval/.../predeclared.json` |

## ② 怎么做的

1. **取三件套原件**：`results.json`、`run_xval.py` / `run_xval_fast.py`、`tests/`、日志与 `predeclared.json`。
2. **基准裁判先问库**：`tensor_benchmark_cases` 按 `anchor_slug` 查登记案例（唯一裁判）。
3. **数值重算**：从记录的 `<u>_super` 反算 `k_super=nu<u>/g`，与工件记录比对（0.0e+00 相对差）；误差对解析锚；另用独立 numpy 代码重写压实映射与邻居表，与被审实现逐项比对。
4. **收敛探针**：交付工件停在 1200 步且 `converged=false`，故在被审代码上把 C1 跑到 6000 步，看 k 是收敛还是巧合。
5. **逐位复现**：把交付 runner 的步数常量**程序化**钉到工件 `steps_cap`，运行并与工件逐算例比对。
6. **落表**：`tensor_maturity_verdicts`（repo=TensorLBM, scope=m17677 …）,由 `maturity_gate` 机器复核。

## ③ 计算结果

### 3.1 基准裁判读数

- SQL：`SELECT count(*) FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-COMPRESSED-FLUID-NODE-NEIGHBOUR-TABLE'` → **0 行**
- 结论：该锚点在 `tensor_benchmark_cases` 里**没有任何登记案例/容差**，因此「基准判据」这一维度对该锚点不存在——只能用交付自带解析锚做人工核对（下表）。

### 3.2 交付工件逐算例（记录值，k 由 <u> 反算复核）

| case | steps | converged | k_super 记录 | k_super 反算 | 相对差 | k vs 解析 [%] | A1 vs A2 ux_maxabs_rel | 足迹比 dense/compressed |
|---|---|---|---|---|---|---|---|---|
| C1_channel_H16 | 1200 | False | 10.553348 | 10.553348 | 0.0e+00 | 1.062 | 0.0e+00 | 1.014 |
| C2_porous_p90_N64 | 1200 | False | 34.500280 | 34.500280 | 0.0e+00 | — | 0.0e+00 | 0.562 |
| C2_porous_p75_N64 | 1200 | False | 13.392723 | 13.392723 | 0.0e+00 | — | 0.0e+00 | 0.676 |
| C2_porous_p60_N64 | 1200 | False | 5.406307 | 5.406307 | 0.0e+00 | — | 0.0e+00 | 0.832 |
| C5_porous_p75_N64_g2e5 | 1200 | False | 13.391759 | 13.391759 | 0.0e+00 | — | 0.0e+00 | 0.676 |
| C6_porous_p75_N96 | 1200 | False | 25.488418 | 25.488418 | 0.0e+00 | — | 0.0e+00 | 0.676 |

C1 槽道解析锚 `k_ref = (H/ny)H²/12 = 10.666667`；交付 k_super=10.553348 ⇒ 误差 1.062%。其余 5 个算例 `k_reference=null`（无解析锚），只有 A1/A2 表示一致性可核。

### 3.3 收敛探针（审计者独立运行，被审代码原样）

| steps | converged | k_super | k_intrinsic | vs 解析 [%] | rho_drift | Ma |
|---|---|---|---|---|---|---|
| 1200 | False | 10.553348 | 21.106697 | -1.062 | 7.8e-14 | 0.0055 |
| 3000 | False | 10.654903 | 21.309806 | -0.110 | 2.0e-13 | 0.0055 |
| 6000 | True | 10.655000 | 21.310000 | -0.109 | 3.7e-13 | 0.0055 |

→ 交付工件的 1.062%（1200 步）是**瞬态值**；收敛值在 6000 步为 10.655000（-0.109%）。两者都远优于 15% 验收带，但「shot 精度」被瞬态偶然放大，报告写 1.06% 属保守而非夸大。

### 3.4 邻居表独立重算（numpy 直接写，不复用被审代码）

- 5×6 掩膜（19 个流体节点）：`nodes/compaction` 一致=True，`index` 映射一致=True，`(K,8)` 邻居表一致=True（不一致条目 0）
- 重算流体节点数 19 vs 实现 19

### 3.5 工件逐位复现

钉 `STEPS=1200`（= 工件 `steps_cap`）后重跑交付 runner：最大 k 相对差 **0.0e+00**，全算例逐位相同=True。

| case | k 工件 | k 复现 | 相对差 | k 误差工件 [%] | k 误差复现 [%] |
|---|---|---|---|---|---|
| C1_channel_H16 | 10.553348 | 10.553348 | 0.0e+00 | 1.062 | 1.062 |
| C2_porous_p90_N64 | 34.500280 | 34.500280 | 0.0e+00 | — | — |
| C2_porous_p75_N64 | 13.392723 | 13.392723 | 0.0e+00 | — | — |
| C2_porous_p60_N64 | 5.406307 | 5.406307 | 0.0e+00 | — | — |
| C5_porous_p75_N64_g2e5 | 13.391759 | 13.391759 | 0.0e+00 | — | — |
| C6_porous_p75_N96 | 25.488418 | 25.488418 | 0.0e+00 | — | — |

### 3.6 结构面读数（三件套是否真存在）

| 项 | 读数 |
|---|---|
| 文件存在：impl_fluid_node_table | True |
| 文件存在：impl_init | True |
| 文件存在：tests | True |
| 文件存在：entry_predeclared | True |
| 文件存在：runner_fast | True |
| 文件存在：artifact_results | True |
| 文件存在：predeclared_spec | True |
| 声明入口算例 | ['C1_channel_H32', '[f-string] C2_porous_p{int(p*100)}_N128', 'C5_porous_p75_N128_g2e5', 'C6_porous_p75_N256'] |
| 工件算例 | ['C1_channel_H16', 'C2_porous_p90_N64', 'C2_porous_p75_N64', 'C2_porous_p60_N64', 'C5_porous_p75_N64_g2e5', 'C6_porous_p75_N96'] |
| 声明入口步数 `STEPS_CAP` | 12000 |
| 工件 `steps_cap` | 1200 |
| 当前提交 runner 步数 | 2500 |
| 声明入口能产出该工件 | False |
| 当前提交 runner 能产出该工件 | False |
| 模块所在 canonical 路径存在 | False（`/root/tp-agent/work/m17677/src/tensorlbm/porous`） |
| 从仓根 `import tensorlbm.porous` | False（ModuleNotFoundError: No module named 'tensorlbm.porous'） |
| 模块外部调用方（主求解器/平台） | 0 |
| 实现占位符计数 | 0 |
| 模块自引文档 `docs/xval_compressed_fluid_node_neighbour_table.md` 存在 | False |

![evidence](../artifacts/m17677_audit/figures/fig_m17677_evidence.png)

## ④ 结果图

左：C1 槽道渗透率相对解析锚的误差随步数（灰带=owner ±15% 验收带，方框=交付工件 1200 步瞬态点，
圆点=本次收敛探针），可见交付点仍在瞬态、收敛后被审实现误差反而降到 0.1% 量级。
右：交付工件的**状态足迹比** `dense/compressed`（int64 邻居表）；p=0.90 时 <1，
即高孔隙率下压缩布局比稠密布局**更占内存**——这是工件里真实存在的反向结果，报告未粉饰。
图：`artifacts/m17677_audit/figures/fig_m17677_evidence.png`（matplotlib 由 `audit_m17677.json` 数值程序化生成）。

## ⑤ 注意点（陷阱/局限/适用范围）

1. **交付落在影子树里**：模块与测试写在 `TensorLBM/src|tests|...`（仓内一份 174 文件的旧嵌套副本），而不是仓的 canonical `src/tensorlbm/`；从仓根 `import tensorlbm.porous` 失败（`ModuleNotFoundError: No module named 'tensorlbm.porous'`）。能力对本仓用户**不可用**，是本次最大结构性缺陷。
2. **入口与工件不一致**：`run_xval.py`（预声明全梯）跑的是 H=32/ny=64、N=128/256、12000 步，与工件 6 算例（H=16/ny=32、N=64/96、1200 步）不同；当前 `run_xval_fast.py` 已把步数改成 2500，即**用提交原样的脚本产不出该工件**，必须把常量钉回工件值才能逐位复现（本审计已程序化做到）。
3. **预声明梯未完成**：`predeclared.json` 声明 N=128/256 与 12000 步；`run.log` 显示 H=32 跑完 12000 步（k_err≈0.028%）后在 C2 N=128 处被截断，成果没进工件；`run2.log` 是 argparse 报错残留，而 `fast.log` 才是工件来源。
4. **全算例 `converged=false`**：1200 步内达不到 1e-9；`ladder_note` 声称「KPI 也在 HALF 步采样」，但代码里 `HALF` 只出现在说明字符串里，从未用于采样 ⇒ 声称的步数自收敛检查**没有实现**。
5. **外部锚不可执行**：被对标 OSS（openLBMPM）是 CUDA 代码，无 GPU 无法运行；交付做的是语义移植，`predeclared.json` 已如实声明为比较局限。provenance 的 `file:line` 无法在本环境核对（磁盘上无 OSS 源码）⇒ 外部锚 WITHHELD。
6. **A1/A2 对比只在流式层**：两臂共享碰撞/forcing 实现，故 0.0 只证明「压实+push 流式 ≡ 稠密 pull 流式」，这正是声明的主张；不得外推为「物理正确」。
7. **内嵌文档悬空**：`porous/__init__.py` 引用 `docs/xval_compressed_fluid_node_neighbour_table.md`，该文件从未生成（`make_report.py` 没跑过），交付里没有任何六段报告，`tensor_reports` 无 mission 17677 行。
8. 无测试/无登记的基准案例、无 `repro_cmd` 记录、capability_delta 为 `no_substantive_delta`（finish 阶段未声明增量）。

## ⑥ 误差判定（逐项回答是否 <3%）

| 关键量 | 基准 | 交付 got | 误差 | <3%? |
|---|---|---|---|---|
| C1 k_super（1200 步，工件） | 解析 `(H/ny)H²/12`=10.666667 | 10.553348 | 1.062% | 是 |
| C1 k_super（6000 步，审计收敛探针） | 同上 | 10.655000 | -0.109% | 是 |
| C1 u_max（工件 profile） | 解析 0.0031875 | 0.0031493 | 1.199% | 是 |
| A1 vs A2 表示等价（6 算例） | 预声明 1e-8 | ux_maxabs_rel=0.0（逐位） | 0 | 是 |
| C2/C5/C6 渗透率 | **无基准**（`k_reference=null`，`tensor_benchmark_cases` 0 行） | — | 无法判定 | 否（无基准，声明原因） |

其余算例没有任何解析/文献/实验基准，也没有登记进 `tensor_benchmark_cases`，故不做误差声明——这直接封住了 L4「基准验证」的资格。

## 判级

**L2（独立功能实现）**。理由：真代码（表构建/压缩 push 流式/半程反弹/Guo forcing 完整可跑）、真测试（12 passed 本次复跑）、真工件（6 算例数值逐位可复现，无造假），并有解析锚误差 -0.11%（收敛）与 1.06%（工件）。
不授 L3：模块落在影子树、canonical 包不可导入、外部调用方 0；
不授 L4：`tensor_benchmark_cases` 该锚点 0 行登记、`repro_cmd` 未随交付记录、预声明梯未完成且 `converged=false`；不授 L5：外部 OSS 锚未执行（无 GPU，已如实声明为语义移植）。

## 结论

- **工件真实**：k 可反算（相对差 ≤0.0e+00），逐位复现（最大相对差 0.0e+00），邻居表独立重算一致，无 m17636 式的 10 倍压缩/清零造假。
- **但级别只能到 L2**：见上「不授 L3/L4」。返修方向（供 reopened 参考）：把模块搬进 canonical `src/tensorlbm/porous/` 并接一个主求解器/平台调用方；跑完预声明梯（或明确改预声明并把收敛判据写进代码）；在 `tensor_benchmark_cases` 登记 C1 解析锚与容差；补 `repro_cmd` 与六段报告。
