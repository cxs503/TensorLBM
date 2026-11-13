# 交差审计 AUDIT_m17896 — XVAL-LBM-DIFFUSE-INTERFACE-WETTING-VIA-WALL-ORDER

- 审计单：m18085（owner 2026-11-11 任务榜令·交差核验单，非重做）
- 被审单：m17896（`TensorLBM`，status=succeeded，branch `agent/m17896`，交付 commit `656d09ed`）
- 锚点：`XVAL-LBM-DIFFUSE-INTERFACE-WETTING-VIA-WALL-ORDER`（对标 OSS：`ebinan92/Fingering_dynamics`，`lattice_boltzmann/validation.py` 的 `psi_wall_list` 幽灵层润湿）
- 审计性质：独立复核（agent 自报数字 ↔ 工件 ↔ 机器裁判三方对表）；**未重做实验**，仅做一处独立复算（两点的 bit 级复现）与一处集成路径探针
- 裁定：**L3**（不是 L4；理由见 §5）

---

## 1. 取证清单（三件套逐项核验）

| 项 | m17896 交了什么 | 核验方式 | 结论 |
|---|---|---|---|
| `impl_files` | `src/tensorlbm/wetting/wall_order_parameter_2d.py`（693 行；`young_cos_theta` L122、`analytic_contact_angle` L143、`grad_laplacian` L169、`WallOrderWettingConfig` L196、`initialise_drop_on_wall` L246、`step` L357、`relax_to_equilibrium` L397、`contact_angle_from_field` L538、`run_to_equilibrium` L649） | 读源码 + 逐函数核对物理（D2Q9、自由能双井、各向同性 9 点算子、Guo 力、半程反弹、壁面幽灵行） | **真实现**（非占位、非随机、非硬编码） |
| `entry` | `xval/LBM_WALL_ORDER_PARAMETER_WETTING/run_xval.py`（234 行；`arm_port` L72、`arm_cac` L111、`sweep` L145、`summarize` L161、`main` L208）；集成入口 `src/tensorlbm/wetting/ladder.py:184 contact_wall_order` | 读源码 + 审计期亲手执行 | **存在且可执行** |
| `run_artifacts` | `xval/.../runs/port_psi±*/{result.json,psi.npz}`（10 套）+ `summary.json` + `figures/fig1,fig2` | 打开 npz 核字段（`(56,80) float64`，`min=-0.9926 / max=1.0092`，`mass_error=1.6e-16`） | **真实场量**；但 `runs/*` **未入 git**（只有 `summary.json` + 2 张图被跟踪），见 F6 |
| `tests` | `tests/test_xval_wall_order_wetting.py`（8 个 test 函数 / 参数化后 16 例） | 审计期实跑：`pytest tests/test_xval_wall_order_wetting.py -q` → **16 passed, 59.87s** | **通过（复核）** |
| `benchmark` | `summary.json` 的 ref vs got 表 | 对照 `tensor_benchmark_cases` 行 + 自己重新解析 `runs/*/result.json` | **交付了，但判据未过**：见 §3 |
| `repro_cmd` | `python3 xval/LBM_WALL_ORDER_PARAMETER_WETTING/run_xval.py sweep && python3 ... summarize`（`predeclared.json.environment.repro`） | 审计期用等价调用（同参数、同 30000 迭代上限）独立复算 2 点 | **可执行、逐位一致，0.0 deg 偏差**（§4） |

## 2. 基准裁判（唯一裁判 = `tensor_benchmark_cases`）

```
SELECT case_name, metric, tolerance FROM tensor_benchmark_cases
WHERE anchor_slug='XVAL-LBM-DIFFUSE-INTERFACE-WETTING-VIA-WALL-ORDER';
```

裁判行（844 字符，原样落盘于 `artifacts/board_audit/m18085/audit_m17896.json:benchmark_case_row`）关键内容：

- 度量：`static contact angle theta [deg]` per point；6 点 ψ_w 扫描 @ξ=3 + ξ=2/3/4 分辨率扫描（56×80，x 周期）
- 容差：`<15% per point`（owner 标准；`predeclared.json` 的 `accept_pct=15.0`）
- **裁判行自己已经写明结论**：「Auditor m17853 re-derived … median |err| 1.60%, 9/10 inside gate, psi_w=+0.70 → 29.52% FAIL, 10/10 rows hit the 30000-iteration cap (converged=false) **=> gate NOT passed**; external OSS anchor **WITHHELD** (NaN; cj1656 queued)」

机器侧第二把锁：`GET /api/v1/tensor/bench/l4-verdict/17896` →
`{"status":"CLAIMED_NOT_PROVEN","l4_benchmark_pass":false,"rows":0,"failed_checks":["execution_success","source_cluster_job","cluster_job_bound","ref_got_error_present","within_tolerance"],"reason":"no tensorbench_results row for this mission"}`。

→ **L4 判据（完整算例 + 误差分析 + 可复现 + 平台复算行）不成立**：平台侧 `tensorbench_results` 一行都没有。

## 3. 逐点误差表（审计期从 `runs/*/result.json` 重新解析，零手抄）

| tag | ψ_w | ξ | θ_got [deg] | θ_ref [deg] | err | <15% | converged | iters | drift | layer_frac | cap_measurable |
|---|---|---|---|---|---|---|---|---|---|---|---|
| port_psi-0.50_xi3.0 | −0.50 | 3 | 133.18 | 133.43 | −0.19% | ✅ | ✗ | 30000 | — | 0.120 | True |
| port_psi-0.35_xi3.0 | −0.35 | 3 | 118.60 | 120.24 | −1.36% | ✅ | ✗ | 30000 | — | 0.128 | True |
| port_psi+0.00_xi2.0 | 0.00 | 2 | 89.04 | 90.00 | −1.07% | ✅ | ✗ | 30000 | — | 0.089 | True |
| port_psi+0.00_xi3.0 | 0.00 | 3 | 88.11 | 90.00 | −2.10% | ✅ | ✗ | 30000 | — | 0.148 | True |
| port_psi+0.00_xi4.0 | 0.00 | 4 | 86.70 | 90.00 | −3.67% | ✅ | ✗ | 30000 | — | 0.210 | True |
| port_psi+0.35_xi2.0 | 0.35 | 2 | 60.87 | 59.76 | +1.85% | ✅ | ✗ | 30000 | — | 0.109 | True |
| port_psi+0.35_xi3.0 | 0.35 | 3 | 59.86 | 59.76 | +0.15% | ✅ | ✗ | 30000 | 2.07e-05 | 0.190 | True |
| port_psi+0.35_xi4.0 | 0.35 | 4 | 58.55 | 59.76 | −2.04% | ✅ | ✗ | 30000 | — | 0.279 | True |
| port_psi+0.50_xi3.0 | 0.50 | 3 | 47.17 | 46.57 | +1.29% | ✅ | ✗ | 30000 | — | 0.273 | True |
| **port_psi+0.70_xi3.0** | **0.70** | 3 | 20.11 | 28.54 | **−29.52%** | **❌** | ✗ | 30000 | 4.0e-04 | **0.998** | **False** |

门线汇总（`summary.json.gates`，审计期重算一致）：窗内 `n_in_window=9`，median |err| **1.3566%**，max **3.6713%**，全部 <15%；窗外 1 点 **29.52% FAIL**；`all_converged=false`（0/10）；`n_cac=0`。

## 4. 独立复核（审计自己跑出来的数字）

1. **复现（防 MOCK）**：审计在**被审 commit `656d09ed` 的源码树**上重跑 2 点（ψ_w=+0.35, +0.70，ξ=3，30000 迭代上限，8 线程，float64），与 m17896 交付值 **逐位一致**（`abs_diff_deg = 0.0`）：59.85617493741043 / 20.114821952526363；`mass_error=1.6e-16`（质量守恒到机器精度）、`drift=2.07e-5 / 4.0e-4`。
   → 数字**不是**手抄、不是随机、不是硬编码；停在上限是对的（没到 tol），`converged=false` 是**诚实**上报。
2. **解析锚（ref 的独立性审查）**：用 4×10⁵ 点求积独立重推壁面过剩能差
   `E_sw = ∫_{ψ_bulk→ψ_w} sqrt(2κ(f−f(ψ_b)))dψ`，`σ = ∫√(2κf)dψ = 2aξ/3`，`cosθ=(E_gas−E_liq)/σ`
   → 与闭式 `(3ψ_w−ψ_w³)/2` 的最大偏差 **1.8e-12**（6 点，`artifacts/board_audit/m18085/young_law_energy_check.json`）。
   **结论两面都记**：公式**对**；但 ref 与模型**共用同一自由能**，所以它是**自洽性检查**，不是外部实验锚。
3. **集成路径探针**：`ladder.contact_wall_order(0.35, steps=60, shape=(48,48), r=12)` → `CAC2D(wall_phi_policy='wall_order')` 端到端跑通，3.3 s 出测量值 57.01° vs young 59.76°（−4.61%，60 步未平衡，**不作物理结论**）。
   → 集成通路**可执行**；但被审单自己交付的 CAC 臂 **n_cac=0**（`predeclared.json` 里它被指定为「independent arm」）。

## 5. 判级 L0–L5

| 级 | 判据 | 本单 |
|---|---|---|
| L0 文档 | 只有文档 | 否 |
| L1 数学/单元 | 公式+单测 | 已超越（不止于此） |
| **L2 独立功能实现** | 真代码：输入→计算→输出 | **✅**：693 行自由能 LBM 模块 + 入口 + 10 套真实运行工件；审计**逐位复现** |
| **L3 系统集成** | 接入主求解器/平台 | **✅**：`CAC2D(wall_phi_policy='wall_order')`（`src/tensorlbm/cac_lbm2d.py:311-356`）由 `ladder.contact_wall_order`（`wetting/ladder.py:184-247`）调用；`test_cac_wall_order_policy_derives_the_young_angle` / `test_cac_wall_order_ghost_matches_prescribed_angle_policy` 实跑通过（16 passed）；审计探针端到端跑通 |
| L4 基准验证 | 完整算例 + 误差分析 + 可复现 **且判据过** | **❌**：①ψ_w=+0.70 一点 29.52% > 15%；②0/10 收敛（30000 迭代上限）；③外部 OSS 锚 **WITHHELD**（参考实现 NaN，且 doc 引用的 `runs/oss_psi+0.00/result.json` **在交付树里不存在**）；④平台 `tensorbench_results` 0 行 → `l4-verdict=CLAIMED_NOT_PROVEN` |
| L5 科研工程成果 | 新物理/工程结论并复现 | 否 |

**裁定：level = 3。** 差异（虚标/超额声明）逐条写进 §6；L4 只差「判据真正过 + 外部锚拿到数字 + 平台复算行」这三件，**功能实质是 L3**。

## 6. 发现（按严重度）

| # | 严重度 | 发现 | 证据 |
|---|---|---|---|
| F1 | **BLOCKING（对 L4）** | 判据未过：ψ_w=+0.70 一点 −29.52% > 15%，10 行全部 `converged=false`（迭代上限 30000）。机制已诚实给出（壁面吸附层吞液滴，`layer_fraction=0.998`、`cap_measurable=false`），且预注册了「不可测即标记剔除、不得调参」，但**判据就是没通过** | `summary.json.gates`；本审计 §3 逐点表；`result.json.measurement` |
| F2 | **BLOCKING（对 L4）** | 外部 OSS 锚 WITHHELD，且**引用的工件不存在**：doc §一 F4 写「200 步 psi 发 NaN（`runs/oss_psi+0.00/result.json`）」，但该路径在交付树中**不存在**（`find` 无 `oss_*` 文件；`finish_runs.sh` 调用的 `oss_runner.py` 也不存在） | `find /root/tp-agent/work/m17896 -name 'oss_*'` → 空；`ls xval/.../runs` → 仅 port_* |
| F3 | MAJOR | 预注册的「独立臂」（CAC2D）**零行交付**：`n_cac=0`，`gates.cac_all_within_accept=false`。于是「两个支撑」实际只剩一个自洽支撑 | `summary.json` (`n_cac:0`, `cac_rows: []`) |
| F4 | MAJOR | 平台复算行缺失：`/tensor/bench/l4-verdict/17896` = `CLAIMED_NOT_PROVEN`，`rows=0` | API 实查 |
| F5 | MINOR | 报告与工件数字不一致：doc 头写「窗内 8/10 点 <3%（中位 1.29%）」，工件是 **9** 点窗内、中位 **1.3566%**、max **3.67%**（>3%）。doc 是上一轮数字未同步 | `docs/XVAL-...-PARAMETER.md:8` vs `summary.json.gates` |
| F6 | MINOR | 运行工件未入库：`runs/*/result.json`、`psi.npz` **未 git-tracked**，只存在于工作树 `/root/tp-agent/work/m17896`（工作树销毁即丢；`repro_cmd` 可重建，但「工件指针」名不副实） | `git ls-files xval/LBM_WALL_ORDER_PARAMETER_WETTING/` 仅 7 项（无 runs/*） |
| F7 | MINOR | 交付 commit `656d09ed` 只存在于 `agent/m17896` 分支，**未进入主线**（`git branch --contains 656d09ed` → 仅 `agent/m17896`；m18085 主线树上 `wall_order_parameter_2d.py` 仍是 m17726 旧版，无 `young_cos_theta`） | `git log`/`diff` 实查 |
| F8 | NOTE | `finish_runs.sh` 用 `--tag` 参数，而 `run_xval.py:main()` 不解析 `--tag` → 该脚本的 ξ∈{2,4} 循环按现状会 argparse 报错（现存的 ξ2/ξ4 工件另有来源）。复现脚本与入口轻微不同步 | `run_xval.py:208-233` vs `finish_runs.sh` |

**未做/未验证（NOT RUN，不许算过）**：CAC 臂的平衡物理解（审计只做了 60 步连通性探针，未做 2500 步物理解）；OSS 参考实现的真实数字（仍无）；集群侧 `cj1656` 的实际执行结果（未取回）。

## 7. 处置

- **confirmed / reopened**：由 maturity_gate 机器裁定（本单给出 level=3 与上表差异，若 gate 认为 L3 的 `integration_callers` 证据不足则可回退 L2）。
- **建议返修项（不改本轮结论，只列清单）**：(a) 补 CAC 臂到平衡并给出 err；(b) 拿到外部 OSS 锚数字或正式降级为「自洽性验证」并删除 doc 中不存在的工件引用；(c) 把 `runs/*` 纳入 git 或改 `repro_cmd` 为唯一工件来源并声明；(d) 把 +0.70 点与「不可测窗」写进裁判行之外的 `validated_window` 引用处，避免下游误读 9/9 为全通过；(e) 同步 doc 头数字；(f) 合并 `656d09ed` 进主线。

---

### 本审计产物
- `artifacts/board_audit/m18085/audit_m17896.json`（证据包：逐点表/裁判行/复现/门线/缺口）
- `artifacts/board_audit/m18085/fig_audit_m17896.png`（图 a：θ(ψ_w) 测得 vs 解析；图 b：逐点 |err| 对数轴 + 15% 门线）
- `artifacts/board_audit/m18085/repro_m17896.py` + `repro_m17896.json`（审计期独立复算，逐位一致）
- `artifacts/board_audit/m18085/young_law_energy_check.json`（解析锚的独立求积复核，1.8e-12）
- `verdict_m17896.sql`（落表 `tensor_maturity_verdicts`，level=3，verdict_by='board-audit'）