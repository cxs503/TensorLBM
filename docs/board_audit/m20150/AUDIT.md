# m20150 交差审计 — m20069 `GAP-DIFFLBM-ADJOINT-VERDICT`（TensorLBM）

* 审计单：**m20150**（board-audit，交差核验，非重做）｜被审单：**m20069**（succeeded）｜锚点：`GAP-DIFFLBM-ADJOINT-VERDICT`
* 被审标识：commit `3dabe126c526f8d69a1d252d7a4a76be6185a610`（分支 `agent/m20069`，**未合入 main**：`git ls-tree main | grep shape_gradient` → 0 条）
* 审计方式：读被审工作树 `/root/tp-agent/work/m20069`（只读）+ 独立重跑测试 + 独立查 `tensor_benchmark_cases` + 取回集群作业 cj2485 节点日志
* 全部数字由 `docs/board_audit/m20150/verify_m20069.py` 程序化重算 → `artifacts/m20150_audit/verify_output.json`（零手抄）
* **裁定：L2（独立功能实现）**；L3/L4/L5 明确不达。门控请求返修（reopen）。
* **独立重跑（集群 cj2519，默认设置）**：shipped JSON **不可复现**——同一条 entry 不带 `--max-steps` 时跑 6000 步、FD 参考值变成 −2.99e−3（shipped −0.2883，差 96.5×），`cos(ship,AD)` 变成 **−1.000/−0.759**（shipped +1.000/+0.996）。判决方向（禁用冻结场代理）在默认跑下**更强**，但报告里发布的证据表本身不成立。

---

## ① 案例介绍（物理背景 / 对象 / 参数）

被审对象是 TensorLBM 的 `src/tensorlbm/adjoint.py`：它把 LBM 流场当作**冻结场**，只对几何/目标求形状导数
（`adjoint.py:240-330` 独立复核确认：`J.backward()` 只得到 `dJ/drho, dJ/dux, dJ/duy`，随后按边界节点折叠
`dJ/dy_i·∂field/∂y_i`，**没有任何 `∂f/∂y_c` 项**、没有穿越求解器的反传）。锚点要求给出「冻结场代理不可用于
形状优化」的论文级判决，并补齐含流场响应的梯度入口。

物理问题（被审单自设）：二维平面槽道（x 周期、y 双固壁）、体积力驱动的单圆盘绕流；目标函数 `J = C_L`
（Ladd 动量交换测力）；设计参数 `y_c` = 盘心横向位置。

| 配置 | nx×ny | D | y_c/H | τ | ν | u_bulk | Re | Ma | D/H | 角色 |
|---|---|---|---|---|---|---|---|---|---|---|
| Q_yc050 | 48×32 | 6 | 0.50 | 0.56 | 0.0200 | 0.05 | 15.0 | 0.087 | 0.231 | shipped `quick` 冒烟 |
| Q_yc055 | 48×32 | 6 | 0.55 | 0.56 | 0.0200 | 0.05 | 15.0 | 0.087 | 0.231 | shipped `quick` 冒烟 |
| R1_yc050 | 96×… | 16 | 0.50 | — | — | — | — | — | — | 论文分辨率（**只有基态流场快照，无梯度结果**） |

## ② 怎么做的（审计方法 / 工具链）

1. **三件套存在性 + 指纹**：12 个声明工件（impl/artifact/figures/docs）逐一 `sha256`（C1，全部存在，无一虚标）。
2. **基准裁判**：直连 `tensorplatform` 库 `SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='GAP-DIFFLBM-ADJOINT-VERDICT'`（C2）。
3. **ref vs got**：从 shipped JSON 重算 `|g_ad-g_fd|/|g_fd|`、`cos`、`g_frozen/g_ad`，并与报告 4.1 表逐格比对（C3）。
4. **报告生成器体检**：扫 `benchmarks/adjoint_verdict_report.py` 中打勾/打叉是否为计算值（C4）。
5. **集成面**：`grep -rn "shape_gradient|optimization_path_eligible" src/`（C5）。
6. **论文级 campaign 取证**：`/root/tp-agent/cluster/jobs/cj2485.log`、`queue.log`、取回 198 MB 工件树（C6）。
7. **生产代码里的"实测值"**：比对 `adjoint.py:81-93` 字面量与本次 artifact（C7）。
8. **独立重跑**：`PYTHONPATH=src python3 -m pytest tests/test_shape_gradient.py -q`（C8）。

## ③ 计算结果（程序化，逐项）

### 3.1 三件套核验

| 项 | 声明 | 实测 | 判定 |
|---|---|---|---|
| impl 文件 | `shape_gradient.py`(612 行)/`adjoint.py`/bench(543 行)/report(418 行) | 全部存在，sha256 已录 | 实 ✓ |
| run_artifacts | `artifacts/adjoint_verdict_diff_lbm.json` + 2×npz + 3×png | 全部存在；JSON `mode=quick`、`generated_at=2026-11-14T08:55:36Z`、`wall_seconds=57.2` | 实 ✓ |
| benchmark ref vs got | 报告 4.1 表 | 重算与报告逐格一致（`-0.28826/0.15470`、`-0.14156/0.14966`） | 实 ✓（数值取自本单自己的冒烟跑） |
| tests | 13 passed | **13 passed in 6.46 s**（审计侧重跑，`tests/test_shape_gradient.py`） | 实 ✓ |
| `tensor_benchmark_cases` | 「总账预置判据」 | **该 anchor 0 行**（全表 118 行，TensorLBM 55 行；`%ADJOINT%/%DIFFLBM%` 仅命中 `GAP-DIFFLBM-3D-COST`） | **无机器裁判 ✗** |
| repro_cmd | `--mode paper --max-steps 20000 …` | **该命令即失败路径**：cj2485 节点 rc=255，R1 基态 `C_D=-43.99`，未写 paper JSON | ✗ |

### 3.2 基准判据逐项（以 shipped artifact 为 got）

| 判据（被审单自设预置） | 阈值 | got | 判定 |
|---|---|---|---|
| AD vs FD 相对误差 Q_yc050 | ≤5% | **153.67%** | ✗（超 30.7×） |
| AD vs FD 相对误差 Q_yc055 | ≤5% | **205.72%** | ✗（超 41.1×） |
| cos(ship, AD) Q_yc050 / Q_yc055 | ≥0.99 | **1.000 / 0.996** | ✓（见 ⑤-F2：报告把这行硬写成 ✗/NO） |
| cos(frozen, AD) Q_yc050 / Q_yc055 | ≥0.99 | **1.000 / 0.980** | ✓ / ✗ |
| 一步优化方向 ΔC_L>0（AD/ship/frozen） | — | 三者均 **−0.3693 / −0.2237** | ✗ |
| 前向守恒（质量漂移） | 机器精度 | −8.41e−11 / −8.46e−11 | ✓ |
| 证据等级自评 | — | **WITHHELD (reduced-scale smoke run only)** | 诚实自评，与审计一致 |

### 3.3 独立重跑（默认设置，集群 cj2519 / n330 / 2 threads / 293.4 s）

| 量 | shipped artifact（700 步，带隐藏 override） | 独立重跑（默认，6000 步） | 判定 |
|---|---|---|---|
| 收敛步数 | 700（`residual=1.00`，**未收敛**） | 6000 | 不同 |
| Q_yc050 FD 参考 `dC_L/dy_c` | −0.2882637515 | **−0.002986389**（差 96.5×） | **不可复现** |
| Q_yc055 FD 参考 | −0.1415564623 | **−0.003196747**（差 44.3×） | **不可复现** |
| cos(ship, AD) Q_yc050 / Q_yc055 | +1.000 / +0.996 | **−1.000 / −0.759** | **符号翻转** |
| AD vs FD 相对误差（n=100） | 153.7% / 205.7% | 129.2% / 131.0% | 两者都 ≫5% |
| AD 对 x_c 的伪敏感度 | 3.5e−15 / −1.1e−2 | `rel err vs FD = 2.09e8`（AD 3.47e−3 vs FD 1.66e−11） | AD 在该参数上给出伪梯度 |

复现：`run_compute submit --repo TensorLBM --stage_dir /root/tp-agent/work/m20069 --cmd "PYTHONPATH=src OMP_NUM_THREADS=2 python3 benchmarks/adjoint_verdict_diff_lbm.py --mode quick --out /tmp/x.json --threads 2"` → cj2519；日志存证 `artifacts/m20150_audit/cj2519_repro.log`；解析见 `verify_output.json:C9`。

## ④ 结果图

![梯度源对比与 AD-vs-FD 误差](figures/m20150_audit/audit_gradients_vs_fd.png)

*图 1*：同一目标、同一参数下四条梯度源（FD 仲裁为负，AD/frozen/adjoint.py 全为正，**符号相反**）与 AD-vs-FD 相对误差对 5% 阈值的 30–41 倍超出。

![展开长度与后传窗口](figures/m20150_audit/audit_unroll_convergence.png)

*图 2*：报告 5.1 的「展开长度收敛」——n=25/50/100 得 0.522/0.530/0.155，均未趋向 FD 的 −0.288；且 artifact 记录 n=100 的 `window=50`（截断后传），非单调本身被截断混淆。

![论文路径基态阻力](figures/m20150_audit/audit_paper_path_cd.png)

*图 3*：quick 尺度 `C_D=+3.53` 合理，论文分辨率 R1 基态 `C_D=−43.99`（非物理），cj2485 节点 rc=255。

![可复现性缺口](figures/m20150_audit/audit_repro_gap.png)

*图 4*：已发布 FD 参考值 vs 默认设置独立重跑（差 44–96×）；默认设置下 `cos(ship,AD)=−1.000/−0.759`——禁用结论在默认跑下更强，而报告发布的 +1.000/+0.996 不可复现。

## ⑤ 注意点（严重度排序，附证据与复现）

| # | 严重度 | 发现 | 证据 / 复现 |
|---|---|---|---|
| **F0** | **blocking（可复现性）** | shipped artifact **只在 `--max-steps 700`（未收敛瞬态，残差 1.00）下产生**；同一条 entry 用默认设置跑 6000 步后 FD 参考值差 44–96×、`cos(ship,AD)` 由 +1.000 翻为 −1.000。报告 4.3 表把 700 步/残差 1.00 写成「收敛步数/收敛解」，属**误标**。 | `verify_output.json:C9` + `artifacts/m20150_audit/cj2519_repro.log` + `configs/*/base/steps=700,residual=1.00` |
| **F0b** | high（方法正确性） | 重跑显示 AD 对第二参数 `x_c` 给出 3.47e−3，而 FD 为 1.66e−11（`rel err = 2.09e8`）——周期方向本应零敏感，AD 该项是伪梯度；shipped artifact 的 `g_ad_xc=3.5e−15` 掩盖了该缺陷。 | `verify_output.json:C9`（cj2519 输出行 `AD vector (y_c,x_c) … rel err=[1.31, 2.09e8]`） |
| **F1** | **blocking（L4 不可达）** | 论文级 campaign 从未产出；报告称「cj2485 仍在排队未取回」，实际 08:59:28 派发 n315、**09:05:28 终态**，节点 `rc=255`，唯一产物是 `fields_R1_yc050.npz`，R1 基态 **C_D=−43.99**（非物理），无 paper JSON。失败未被上报。 | `verify_output.json:C6`；`/root/tp-agent/cluster/jobs/cj2485.log`；`queue.log`；取回树 `artifacts/adjoint_verdict_diff_lbm/` 仅 3 npz |
| **F2** | **high（证据完整性）** | 报告"判决"段的头条判据是**字面量**：`adjoint_verdict_report.py:198` 硬写 `NO —— 冻结场代理解不被允许…`，`:294-296` 把 cos 三行硬写 `✗`。实测 cos(ship,AD)=1.000/0.996、cos(frozen,AD)=1.000/0.980，按报告自己写的阈值 ≥0.99 应打 ✓。禁用结论本身由**另一条**判据（AD-vs-FD 154–206% ≫ 5%）与代码结构支撑，但报告的头条不成立。 | `verify_output.json:C4`（命中行号 198/294/295/296） |
| **F3** | **high（溯源）** | 生产代码 `adjoint.py:81-93` 的 `FROZEN_FIELD_VERDICT` 硬编码"实测值" `cos_vs_unrolled_ad=-1.000`、`amplitude_ratio_vs_true=2.289`，与本单 artifact（cos 1.000/0.980；|g_frozen/g_ad| = 5.17/5.37）**互相矛盾且无 run 指针**。 | `verify_output.json:C7` |
| **F4** | medium（实验设计/标注） | `n=100` 的"展开式 AD"实为 **window=50 截断后传**（artifact `configs/*/ad/n=100/window=50`），因此 0.522→0.530→0.155 的非单调不能当作"收敛到有限时窗形状导数"；报告把 FD/AD 全差归因于"瞬态主导"（单因归因），无窗口扫描拆分（瞬态 vs 截断）的对照。 | `verify_output.json:C3` + artifact 原始字段 |
| **F5** | medium（表述越界） | commit message「matches central finite differences (rel < 2e-4, tests/test_shape_gradient.py, 13 passed)」只在 **n=30、40×32、ε=0.6** 的短窗滚动（`tests/test_shape_gradient.py:111-137`）成立；工程尺度是 154–206%。原句未限定适用范围。 | `verify_output.json:C8` + 测试源码 |
| **F6** | medium（回归锁定） | 13 项测试只断言**代码契约**，没有一项把 `artifacts/adjoint_verdict_diff_lbm.json` 与 `FROZEN_FIELD_VERDICT` 绑定 → F3 这类漂移可长期不被发现。 | `tests/test_shape_gradient.py` 全量扫描（无 json 读取） |
| **F7** | minor（交付面） | `3dabe126c` 仅在 `agent/m20069`，main 无 `shape_gradient.py`；下游（优化环/平台）无法 import。 | `git ls-tree -r --name-only main | grep shape_gradient` → 0 |
| **F8** | minor（平台侧，非本仓） | 调度器把 cj2485 记为 `state=succeeded, rc=0`，与节点日志 `rc=255` 冲突（记账缺陷，建议 Platform 立单）。 | `run_cluster ps cj2485` vs `jobs/cj2485.log` |

**未跑 / 未知（unknown，不记作 pass）**：paper 模式全量复现（R2 ny=144 需 >7200 s）；R1 `C_D=−43.99` 的**根因**未定位（只证明现象与作业失败，未做二分）；文献级外部锚点（Pironneau 1973 / Ataei & Salehipour 2024）只用于概念引用，**未复现任何外部数值**；审计侧对 quick 冒烟的逐位复核作业 cj2519 递交后仍在运行，故「逐位可复现」本轮记为 **not_run**（不记 pass）。

## ⑥ 误差判定（关键量 vs 基准，逐项，显式回答是否 <3%）

| 关键量 | 基准 | got | 相对误差 | <3%？ |
|---|---|---|---|---|
| Poiseuille `u_max` | 解析 `gH²/(8ν)` | pytest 断言通过（`1e-12` 量级） | ~1e−12 | **是** |
| 质量守恒（streaming） | 解析（逐位置换） | 漂移 −8.4e−11 | ~1e−10 | **是** |
| AD vs FD（n=30 短窗） | 中心差分 | pytest `<2e-4` | 2e−4 | **是**（但适用范围=短窗，见 F5） |
| AD vs FD（工程尺度 Q_yc050/055，shipped 700 步） | 中心差分（未收敛解） | 153.67% / 205.72% | **153.67% / 205.72%** | **否（>15%，不可接受）** |
| AD vs FD（默认 6000 步，独立重跑） | 中心差分（收敛态） | 129.22% / 131.02% | **129.22% / 131.02%** | **否（>15%）** |
| FD 参考值可复现性 | shipped vs 重跑 | −0.2883 / −0.1416 vs −0.002986 / −0.003197 | **96.5× / 44.3×** | **否** |
| 冻结场代理 vs 真梯度 Q_yc050/055 | 展开式 AD/FD | 377.3% / 667.9% | — | **否** |
| 论文分辨率 R1 基态 `C_D` | 应为正（Ladd 动量交换） | **−43.99** | 符号错误 | **否（非物理）** |
| 外部文献 `C_L` | 无（仓库内无该配置基准） | — | — | **无基准 → 无法判 <3%（声明原因：单盘槽道 Re=15 无既有参考值，本单以自设 FD 仲裁替代）** |

**结论**：前向求解器的解析校验在 3% 以内（1e−12/1e−10），但**判决所依赖的梯度量在工程尺度超差 129–206%**，论文分辨率的基态力测量非物理（C_D=−43.99），且已发布的 FD 参考值**不可复现**（差 44–96×，cos 符号翻转）——m20069 的 WITHHELD 自评是诚实的，本审计**不得**把它升级为 L4；同时 F0 说明报告的**数值表本身**需要重出。

## ⑦ 判级（L0–L5，级别=功能实质）

| 级 | 判据 | 结论 | 依据 |
|---|---|---|---|
| L0 | 文档成果 | 达到 | `docs/GAP_DIFFLBM_ADJOINT_VERDICT.md`（6 段齐全，163 行）+ 3 图 |
| L1 | 数学/单元验证 | **达到** | 13 passed（6.46 s）：Poiseuille 解析、质量守恒 1e−12、AD-vs-FD 短窗 2e−4、门控契约 |
| L2 | 独立功能实现（真代码 输入→计算→输出） | **达到** | `shape_gradient.py:1-612` 真前向 D2Q9+Guo+软反弹+Ladd 探针，三路梯度真实产出；`adjoint.py:76/95-109` 门控真逻辑（默认拒绝） |
| L3 | 系统集成（接入主求解器/平台） | **不达** | `src/` 内 0 个 caller（C5）；`optimization_path_eligible` 仅被 bench/test 调用；文档结论#3 自认"第 2 轮再接线" |
| L4 | 基准验证（完整算例+误差分析+可复现） | **不达** | anchor 在 `tensor_benchmark_cases` **0 行**；paper campaign 失败（F1）；AD-vs-FD 153–206%；头条判据硬编码（F2） |
| L5 | 科研工程成果 | 不达 | 无外部数值锚点、无集成、无非定常扩展 |

**裁定：level = 2。** 相对 m20069 自评（隐含"判决+新入口已交付"）**下调 2 级**：判决的机制（门控 + shape_derivative_only 声明）与新的含流场响应入口是真实代码（故不低于 L2），但「论文级判决」这一交付承诺在基准与论文两端都缺证据（故不高于 L2）。

差异写明（虚标核验）：①「总账预置判据」——总账里没有该 anchor 的任何 benchmark 行，属**自设判据**，不构成机器裁判；②「论文分辨率 campaign 已递交、排队中」——已终态、已失败、未上报；③`FROZEN_FIELD_VERDICT` 的 cos=−1.000/ratio=2.289 与本单 artifact 矛盾，属**无出处的硬编码实测值**。

## ⑧ 处置（disposition）与返修要求

**disposition：request changes（确认 L2，L4 重开）**。返修项（按优先级）：

1. 修 paper 路径（R1 `C_D=−43.99` 根因 + cj2485 rc=255 的崩溃点），跑完 R1/R2 × 两点 × 双步长 FD + Richardson，产出 `mode=paper` JSON；`evidence_grade` 才有资格从 WITHHELD 升。
2. 在 `tensor_benchmark_cases` 注册该 anchor 的行（case_name/metric/tolerance/repo=TensorLBM），让 gate 有机器裁判。
3. 去掉 `adjoint_verdict_report.py:198/294-296` 的字面量，全部由 JSON 计算；报告头条改为计算值。
4. `adjoint.py:81-93` 的 `FROZEN_FIELD_VERDICT` 要么删除字面量改为从 artifact 读取，要么补 run 指针；并加一条读 artifact 的回归测试（F6）。
5. 补后传窗口扫描（window=25/50/100/全窗）以拆分「瞬态 vs 截断」，重写 5.1 的归因。
6. 把 `shape_gradient_through_flow` 接入形状优化环（L3），并让 `optimization_path_eligible` 在环内被强制调用。
7. **可复现性（F0）**：把步数/收敛判据写进报告的 predeclared 块并在正文引用，废除 700 步这种未收敛 override 作为发布数值；用默认设置重出 JSON（或加 `--max-steps` 显示记录），并把 4.3 的「收敛解」改成实际收敛判定；顺带修 AD 对 `x_c` 的伪梯度（F0b）。

---

*审计产物*：`docs/board_audit/m20150/verify_m20069.py`（8 项检查，重跑 ~10 s）、`artifacts/m20150_audit/verify_output.json`、`docs/board_audit/m20150/make_figures.py` + 3 图、`verdict_m20150.sql`（已落表）。