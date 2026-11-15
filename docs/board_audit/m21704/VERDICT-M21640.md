# 交差审计 m21640 — XVAL-LBM-TWO-COMPONENT-SC-LAYERED-POISEUILLE-EDM

- **审计单**: m21704（board-audit，任务榜 2026-11-11「结榜要回来交差」）
- **被审对象**: mission m21640（TensorLBM, status=succeeded, `fail_stage=budget_exhausted+forced_finish`），提交 `8a8d86d60`，已并入 `main`（`5cf348523 Merge #2485 (m21640)`）
- **审计工作树**: `/srv/tensor/agent/work/m21704`（`agent/m21704`，HEAD `15ef2a4c0`），被审工作树只读引用 `/srv/tensor/agent/work/m21640`
- **基准裁判**: `tensor_benchmark_cases` id=149，`anchor_slug='XVAL-LBM-TWO-COMPONENT-SC-LAYERED-POISEUILLE-EDM'`，`status='passed'`，`tolerance='<15% per KPI (owner standard; self target 3%) … got: …'`
- **结论**: 功能实质 **L4（基准验证）**，处置 **approve with notes**（3 项整改，见 §7）。基准主判据（4 项 KPI ≤ 8.84%）成立且**逐位可复现**；但入库台账把 `status` 记成 passed 时**漏记了本单自己预注册的 `interface_position<=5%H` 子门 FAIL**，且「静态界面伪流」这条 KPI 在本配置下**结构性恒为 0、不可失败**。

---

## 1. 取证清单（三件套是否真存在）

| 项 | 指针 | 判定 |
|---|---|---|
| 实现 | `TensorLBM/src/tensorlbm/verification/layered_poiseuille/two_component_sc.py:1-401`（MCMP 求解器） | REAL |
| 库内核（集成） | `TensorLBM/src/tensorlbm/multiphase/__init__.py:492 sc_two_component_force`、`:558 collide_edm_common`、`:470 _wall_extend`；由 `two_component_sc.py:57-58` 导入、`:231`、`:273` 调用 | REAL |
| 解析参考 | `.../analytic.py:120+ solve_layered`（2N×2N LU，与求解器无关） | REAL，见 §3 独立复核 |
| 入口 | `.../run_two_component_sc_validation.py:1-409`（`python -m …`，`--matrix/--case/--figure-only`） | REAL |
| run_artifacts | `artifacts/xval_two_component_sc_layered_poiseuille_edm/results.json`（28 303 B，sha256 `c902d46f6f51c5e82eb36539e30273e1e116b4ba7312ae1f5c64961e1f41ce8f`，与 `sha256.txt` 一致）、`two_component_sc_layered.png`（155 418 B） | 存在且哈希自洽 |
| tests | `TensorLBM/tests/test_two_component_sc_layered_poiseuille.py`（10 个 test） | 重跑 10 passed in 58.23 s（`15ef2a4c0`） |
| benchmark ref vs got | DB id=149 `metric`/`tolerance` 文本 vs `results.json` | 数值对得上（唯一例外见 F3：control 0.25 % vs 0.3675 %） |
| repro_cmd | `docs/XVAL-LBM-TWO-COMPONENT-SC-LAYERED-POISEUILLE-EDM-STATUS.md:90-104` | 可执行；**注意** `--matrix core` 在 `all_gates_pass=false` 时按设计 `return 1`（不是崩溃） |

## 2. 复现（逐位）

在**合并后**的 main（`15ef2a4c0`）上重跑，与 m21640 工件逐字段比对：

| case | 指标 | 工件 m21640 | 本单重跑 | 完全相同 |
|---|---|---|---|---|
| `control_single` | `eps_inf` | 0.003675132925035144 | 0.003675132925035144 | ✅ |
| `control_mixture` | `eps_inf` | 0.0036728811223318983 | 0.0036728811223318983 | ✅ |
| `layered_r167` | `u_peak_err` / `eps_inf` / `eps_l2_rel` / `Q_err` | 0.07437226515465295 / 0.08837783187947695 / 0.08234053870518139 / 0.08159435994696285 | 逐位相同 | ✅ |
| `layered_r167` | `y_iface` | 19.207644419279987 | 19.207644419279987 | ✅ |
| `layered_r167` | 全场剖面 `ux_total` | — | `ra['profiles']==rb['profiles']` | ✅ |

工件里的 `environment.git_commit` 记的是 `1e5e0891b`（提交前的父提交，运行时代码未提交），而工件本身随 `8a8d86d60` 入库 —— 见 F4，属**归属注记缺陷**，不影响可复现性（上表已证）。

## 3. 基准裁判复核（唯一裁判）

**参考解独立复核**（不碰求解器，只用 `solve_layered` 的正则性）：`u(0)=0`、`u(H)=3.5e-18`；逐层 `u''=-g/ν`（ν=0.1 层 −4.987e-4 vs −5.000e-4，ν=0.1667 层 −2.993e-4 vs −3.000e-4）；界面处 `u` 连续（0.04800 两侧）、`νu'` 连续（跨界面差分给出 u′ 比 −0.6 = ν_lo/ν_hi）；`u_peak=0.0490` 且**低于**同 g、ν=0.1 单流体峰值 0.064（分层变黏使峰值下降，方向正确）。参考解成立。

**KPI 判定**（owner 标准 15 % 不可接受；自定目标 3 %）：

| case | u_peak_err | eps_inf | eps_l2_rel | Q_err | y_iface / 误差 | 15 % 门 | 5 %H 子门 |
|---|---|---|---|---|---|---|---|
| `control_single` | 0.367 % | **0.3675 %** | 0.370 % | 0.377 % | — | PASS | — |
| `control_mixture` | 0.367 % | **0.3673 %** | 0.370 % | 0.377 % | 16.000 | PASS | — |
| `layered_r167`（注册 KPI 例） | **7.44 %** | **8.84 %** | **8.23 %** | **8.16 %** | 19.21 vs 16.0 → **10.02 %H** | PASS(4/4) | **FAIL** |
| `layered_r233`（原本 not_run，本单补跑） | 10.15 % | **13.80 %** | 11.25 % | 10.11 % | 18.63 → 8.22 %H | PASS(4/4) | FAIL |
| `layered_r167_h48`（原本 not_run，本单补跑） | **0.07 %** | 4.87 % | 3.82 % | 3.36 % | 24.25 vs 24.0 → 0.52 %H | PASS(4/4) | **PASS** |
| `layered_static` | — | 0.0 | `nan` | 0.0 | 18.09 | 「伪流 0.0 %」PASS | — |

补跑工件：`TensorLBM/artifacts/audit_m21704/results_rerun_all6.json`（6/6 core+full 例），脚本 `artifacts/audit_m21704/make_figures.py`，图 `fig1_layered_profile_vs_exact.png`、`fig2_kpi_errors.png`。

**关键新证据**：把 m21640 如实声明「未跑」的两例补上后，`layered_r167_h48`（H=48）**全部预注册门通过**（含界面 0.52 %H），且 `eps_l2` 3.82 % 逼近 3 % 自定目标 —— 说明 H=32 的 8.8 %/10 %H 主要是**界面仅 1 格宽的扩散过渡在粗网格上无法定位**导致的离散化效应，不是模型错。`layered_r233`（黏度比 2.33）`eps_inf=13.8 %`，距 15 % 门只剩 1.2 pp —— 高黏度比下精度余量很薄，这一点在 m21640 的「7.4/8.8/8.2 %」口径里看不到。

## 4. 发现（按严重度）

**F1 — MAJOR / 台账与自身预注册判据矛盾（可交付性阻断）**
`results.json` 自身的 `gates.rule` 写了 `layered … interface position<=5%H`，`records[layered_r167].gates['interface_position<=5%H']=false`，顶层 `summary.all_gates_pass=false`，入口按设计 `return 1`；而 `tensor_benchmark_cases` id=149 被写成 `status='passed'`，`metric`/`tolerance` 文本只列 4 项 15 % KPI，**未记录该子门 FAIL**。
证据：`artifacts/.../results.json`（`summary`、`records[2].gates`）；DB `id=149`。
复现：`python -m …run_two_component_sc_validation --case layered_r167`（文本末行 `all_gates_pass: false`，退出码 1）。
影响：单看台账会以为整套预注册判据全过。**诚实度不由台账体现，而由 STATUS §3/§4 与 mission result 体现——那两处如实写了 FAIL**（这一点对 m21640 有利，必须记入）。

**F2 — MAJOR / 「静态界面伪流」KPI 结构性恒 0，不可失败（证据空转）**
`layered_static` 配置 `nx=1` 且初值在 x 上完全均匀。x 均匀时 SC 力的 x 分量为 `-G ψ_a Σ_i w_i dx_i ψ_b(x+c_i)`，±x 邻居成对抵消（`w_i dx_i = -w_j dx_j`，`ψ_b` 同值），故 `F_x ≡ 0`，`u_x` 恒为 0 —— 该门**永远不会失败**，不能证明伪流受控。
本单实测（`artifacts/audit_m21704/static_kpi_vacuity.py|json`，ny=18/1500 步，同物理）：
`nx=1 无噪声 max|u_x| = 0.000e+00`；`nx=8 均匀 = 0.000e+00`（逐位 0）；`nx=8 加 1e-3 密度扰动 = 9.08e-14`；`噪声 0.05 → 6.2e-11`；`0.2 → 5.6e-10`。
即：报告的 0.0 % 是**对称性产物**，不是伪流测量。注册 `metric` 文本把「spurious current of the static interface」列为判据，属**证据空转**。真正的伪流检验需要斜/曲界面或持续扰动（3D 或 voxel 化界面）。
复现：`PYTHONPATH=src python artifacts/audit_m21704/static_kpi_vacuity.py`。

**F3 — MEDIUM / 台账与文档里的 control 数字不可复现（0.25 % vs 0.3675 %）**
commit `8a8d86d60` 标题、`STATUS.md §3`、DB `tolerance` 文本均写 `control 0.25 %`；而工件 `results.json` 与本次逐位重跑均为 `control_single eps_inf = 0.3675 %`（`control_mixture 0.3673 %`）。0.25 % 在仓库任何工件里都找不到来源（旧 `quick_validation_results.json` 里也没有；那是上轮 177.78 % 的另一配置）。两值都在 3 % 内，**不改变判定**，但违反「got 值必须从工件程序化转写」的报告纪律。

**F4 — MEDIUM / 工件归属注记缺陷**
`results.json.environment.git_commit = 1e5e0891b`（父提交，运行时工作树是脏的），而工件随 `8a8d86d60` 入库；`environment.argv` 只记了最后一次 `--case layered_r167`，但 `records` 是 4 次独立调用的合并产物（入口 `_load()` 会累积 `results.json` 已有记录）。后果：单看工件无法定位「这组数字对应哪份代码」。本次已用**逐位复现**补上这一环（§2）。

**F5 — MEDIUM / 工件不是合法 JSON**
`results.json` 含 3 个裸 `NaN`（`control_single.y_iface`、`layered_static.eps_l2_rel` 等）。`json.loads(..., parse_constant=…)` 严格解析报错，PostgreSQL `jsonb` 直接拒收 —— 对「工件自动入库/机器复核」链路是实质障碍。修法：`None`（null）或 `"nan"` 字符串。

**F6 — MINOR / 测试弱化预注册判据**
`tests/test_two_component_sc_layered_poiseuille.py:144 test_layered_poiseuille_within_15pct` 取 `min(nominal, interface_matched)`（best-of-two），即允许求解器在两条参考里挑对自己有利的一条；预注册 KPI 是**对 nominal H/2 参考**。本例 nominal 更优（7.44 % < 9.74 %），故结论不变，但断言方向被放宽了。
`…:121 test_static_interface_has_no_spurious_flow`（nx=2、初值 x 均匀）与 F2 同因，为空转断言。

**F7 — MINOR / 文档数字手抄残留**
`STATUS.md §3` 表中 `layered_r167` 的 `u_peak_err` 绝对值列写成 `6.30e-2 / 6.79e-2`，工件为 `4.530e-2 / 4.894e-2`（误差百分比 7.4 % 正确，两侧绝对值各差约 39 %）；同文称「11 physical assertions」，文件实为 10 个 test。

**F8 — MINOR / 交付时矩阵不完整（已如实声明）**
交付时 `layered_r233`、`layered_r167_h48` 未跑（`summary.not_run`、`STATUS §4` 均写明，且不主张收敛阶）。本单补跑后见 §3：h48 全过、r233 余量 1.2 pp。

## 5. 未做的检查（不得读作已通过）

- 无**外部锚**：参考解是解析解，没有第二套独立 LBM/CFD 代码或实验对照 → 不构成本锚「XVAL（交叉验证）」意义上的外部交叉验证，故不给 L5。
- 无正式网格收敛研究：只有 H=32/H=48 **两点**，未估收敛阶、未做三套网格（`cases.py` 里有 S32/S64/S96、L3_32/64/96 例，但那是单组分 solver 的另一套矩阵）。
- 伪流未真正测到（F2）；3D、曲界面、大密度比均未覆盖。
- 只验了 CPU/float64/torch 路径；GPU 与并行一致性未验。
- `STATUS §2` 声称 `wall_mode="dry"` 使 `eps_inf` 从 6.8 % 升到 14.4 %（"measured"），仓库内**无对应工件**，本单未复核。
- 未审计 `tensorlbm.multiphase.step_sc` 单组分生产路径本身（只审计了本单新增的 MCSP 两个内核被正确调用）。

## 6. 判定

- **L4（基准验证：完整算例 + 误差分析 + 可复现）**。理由：真代码（输入→计算→输出）且驱动库内核（L2/L3），一个完整算例对**独立解析参考**给出 4 项 KPI 误差 7.4–8.8 %（owner 15 % 门内），含层内黏度拟合、界面匹配诊断、控制例 0.37 % 的缺陷探测，工件带 sha256 且**逐位可复现**（本单在合并后 main 上验证）。
- **不给 L5**：无外部锚（独立代码/实验）、无科研新结论（MCMP+EDM 为既有方法），符合「级别=功能实质」。
- **不因 F1/F2 降级**：F1 是**台账记录**与自定子门冲突（且作者在 STATUS/mission result 里如实披露了 FAIL），F2 是某一条 KPI 的证据空转，两者都不使「基准验证能力」不存在；但两者都必须在台账里改正，否则下一轮审计仍会以台账为准误判。

## 7. 整改项（非阻断，建议下一轮做）

- **R1**：更新 `tensor_benchmark_cases` id=149 —— `status` 改为 `passed_with_caveats`（或保留 passed 但在 `metric` 里显式写 `interface_position 10.02%H > 5%H subgate FAIL at H=32`），并登记 H=48 结果（全门通过）。
- **R2**：把 `layered_static` 换成**能失败**的伪流算例（斜界面/持续扰动/3D 液滴），或从注册 `metric` 里删掉该 KPI。
- **R3**：0.25 % → 0.3675 %（control_single）；`git_commit` 改记 `HEAD + 工作树 diff 哈希`；`NaN` → `null`；`test_layered_poiseuille_within_15pct` 去掉 `min()`，对 nominal 参考断言。

---

### 附：本单产物

| 文件 | 内容 |
|---|---|
| `TensorLBM/artifacts/audit_m21704/m21640_results.json` | 被审工件的只读副本（sha256 已核） |
| `TensorLBM/artifacts/audit_m21704/results_rerun_all6.json` | 合并后 main 上 6/6 core+full 例逐位重跑 |
| `TensorLBM/artifacts/audit_m21704/static_kpi_vacuity.py` / `.json` | F2 的反证探针与结果 |
| `TensorLBM/artifacts/audit_m21704/make_figures.py` | 程序化出图（零手抄） |
| `TensorLBM/artifacts/audit_m21704/fig1_layered_profile_vs_exact.png` | 速度剖面 vs 精确解（nominal / 界面匹配）+ φ_A 剖面 |
| `TensorLBM/artifacts/audit_m21704/fig2_kpi_errors.png` | 各例 KPI vs 15 % 门 / 3 % 目标 |
| `docs/board_audit/m21704/verdict_m21704.sql` | verdict 落表语句（evidence 为真实指针） |

**独立性限制**：审计与被审同属 TensorLBM 生态、共用同一台机构建；本审计复用了被审代码的解析参考模块（但独立复核了它的正则性，见 §3），并**未能**取得任何外部/实验数据做第三方对照。以上判定不覆盖被审单未声明、且本单未抽查到的代码路径。