# m17877 交差核验（board-audit）：m17724 `XVAL-LBM-CUPY-DROPIN-NO-KERNEL-FUSION` → 判级 L2

- 受审对象：m17724（TensorLBM，status=succeeded，fail_stage=`budget_exhausted+forced_finish`）
- 受审工作树：`/root/tp-agent/work/m17724`，branch `agent/m17724`，HEAD `9bcc6db6`（**HEAD 内不含本单任何改动**）
- 对标对象（参考实现）：`ebinan92/Fingering_dynamics` @ `150a573`（实机克隆在 `/tmp/Fingering_dynamics`，与报告引用 commit `150a5739e` 一致）
- 审计工作树：`/root/tp-agent/work/m17877`（本报告 + 重跑脚本 + 证据快照）
- 判级：**L2（独立功能实现）**；L3/L4/L5 均不达标，理由见 §4/§5。
- `tensor_benchmark_cases` 对该锚点 **0 行**（全表 65 行 / 59 个 anchor），故"基准裁判表"对本单**无登记判据**，裁判只能由「本单自declared 判据 σ/R<15% + 审计独立重算」承担（§3）。

---

## 1. 取证：三件套是否真存在

| 三件套 | 结论 | 证据 |
|---|---|---|
| run_artifacts 文件 | **存在但缺件**：`xval/artifacts/{primary_results.json, fig1..fig3}.png` 共 4 个；`xval_results.json`（入口真正会写的文件名）**不存在**，`fig4_sensitivity.png` **不存在**；收敛/敏感性数值无任何机器工件 | `evidence_snapshot/primary_results.json`（sha256 `5f6db1d2…`）、`SHA256SUMS.txt` |
| benchmark ref vs got 数值 | **标杆侧无 ref**：`tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-CUPY-DROPIN-NO-KERNEL-FUSION'` → 0 行；交付自用 ref = 解析解 σ/R（σ 由 `κ=0.75σξ`、`a=−2κ/ξ²` 反推，**模型自校准**，见 F8） | `runuser -u postgres psql -d tensorplatform -c "SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='…'"` → 0 行 |
| repro_cmd 可执行 | **可执行但名不符**：`python3 xval/xval_fingering_dynamics.py` 跑通（审计在只读导入下重跑主算例 18.6 s），但它写的是 `xval_results.json`（`xval_fingering_dynamics.py:577`），交付的 `primary_results.json` **全仓无任何脚本产生**（`grep -rn primary_results --include=*.py` → 0 命中）；入口还会连带跑收敛+敏感性研究，交付工件里没有它们的数 | F4 |

### 独立重跑（本单核心证据）

| 项 | 交付值 | 审计重跑值 | 一致性 |
|---|---|---|---|
| Δp (R=15, T=2000) | 6.671389172852038e-4 | 6.671389172852038e-4 | **逐位相同** |
| 解析 σ/R | 6.666666666666667e-4 | 6.666666666666667e-4 | — |
| 误差 | 0.0708% (<15% ✓) | 0.0708% | ✓ |
| ψ 界 | [−1.0015, 0.9996] | [−1.001473, 0.999598] | ✓ |
| u_max | 9.6e-6 | 9.640969e-6 | ✓ |
| 文档 "R=10, 800 步, 1.0064e-3, 0.64%" | — | 1.0064342678267058e-3 / 0.6434% | ✓（工件中无此条，但数值真实） |

生数据：`docs/audit_m17877/audit_rerun.json`（脚本 `audit_run.py`，只读导入 m17724 模块，未改动受审树）。
图：`docs/audit_m17877/fig_m17877_evidence.png`。

---

## 2. 六段齐全性

`docs/XVAL_LBM_CUPY_DROPIN_NO_KERNEL_FUSION.md`（139 行）六段齐：①Problem+Reference(5-24) ②Method(25-44) ③Parameter Table(45-61) ④Comparison Table(62-77) ⑤Convergence & Sensitivity(78-91) ⑥Failure Modes & Artifacts(92-113)，另附差异清单(114-127) 与 friction(128-139)。**形式齐 ≠ 内容对**：④→ 见 F1；⑤ → 见 F2。

---

## 3. 误差判定（逐项，显式回答「是否<3%」）

| 关键量 | ref | got | 误差 | <3%？ |
|---|---|---|---|---|
| Δp, R=15, T=2000（交付定义 p=ρ/3+ψμ/3） | σ/R = 6.6667e-4 | 6.6714e-4 | **0.0708%** | **是**（单点） |
| 同上，但取 1200–1999 步的**时间带**（每 25 步采样 32 点） | σ/R | — | 0.235%–6.835%（均值 2.667%） | **否**（带内部分相位 >3%；"0.07%" 是单相位采样，非收敛值） |
| Δp, R=15, 改用 **OSS 原生压力定义** p=ρ/3+ψμ（fingering.py:130，同时进入 feq） | σ/R | 6.7274e-4 | **0.911%** | **是**（且 u_max 6.4e-7，比交付定义小 15 倍） |
| Δp, R=10, T=800 | σ/R = 1e-3 | 1.00643e-3 | **0.6434%** | **是** |
| σ 的独立路径：σ = ∮κ|∇ψ|²dA/(2πR)（自由能泛函，非压力） | 0.01 | 0.00971 | **2.90%** | **是（贴线）** |
| σ 由 Δp·R 反推 | 0.01 | 0.010007 | 0.071% | 是 |

交叉路线（压力 σ vs 毛细能量 σ）互洽 2.9% → 说明 `κ=0.75σξ` 的自校准不是空转，Laplace 检验有实质内容（F8）。

---

## 4. 判级依据（功能实质，非文档篇幅）

- **L1（数学/单元验证）已达**：真实积分器 + 解析判据 + 可复现。
- **L2（独立功能实现）已达**：`SOYFreeEnergyLBM`（`xval_fingering_dynamics.py:67-207`）是真代码，输入（R,σ,ξ,τ）→ 2000 步 f/g 双分布 D2Q9 演化（碰撞+Guo 力+周期 stream）→ 输出（Δp, u_max, ψ 界），审计逐位复现。与 OSS 源码逐项核对：μ(149)、κ/a(41-42)、feq/geq 的 A0/A1_8/B0/B1_8(177-198)、Guo 源(201-206)、∇ψ/∇²ψ 各向同性模板(233-296) **均逐字对应**（∇ψ 权重×12 = [4,±1]；∇²ψ 权重 = [2/3 正交, 1/6 对角, −20/6 中心]，与移植版等价，已手算核对）。
- **L3 未达**：零集成。`git status` 全 `??`（3 项未跟踪）、本单 **0 个 commit**、`src/` 无引用、无 caller、无测试（`grep -rn "xval_fingering\|SOYFreeEnergy" tests/ src/ benchmarks/` → 0 命中）。唯一库耦合是 `from tensorlbm.cac_lbm2d import CAC2D`（只读导入，可导）。
- **L4 未达**：裁判表 0 行；且 XVAL 的"OS"侧**从未执行**——交付里没有任何 OSS 输出数值，也没有同算例对照；TensorLBM 侧 `free_energy_step` / `CAC2D` 的对标结论**无工件**（见 F3）。收敛证书缺失（F2）。入口产不出交付工件（F4）。
- **L5 未达**：无科研增量（无指进物理、无新现象、无新方法）。

---

## 5. 严重度排序的发现

**F1（high，主张造假/不实｜可复现）** 报告②表把压力公式标为 "identical"，与 OSS 源码不符。
- OSS：`lattice_boltzmann/fingering.py:130` `return 1/3*self.rho + self.psi*self.mu`
- 移植：`xval_fingering_dynamics.py:136-137` `return rho/3.0 + psi*mu/3.0`
- 影响：审计用 OSS 定义重跑 → 0.911%（仍 <15%，但 u_max 由 9.64e-6 降到 6.4e-7）。即"0.0708%"是**本移植私有压力定义**下的数，不是 OSS 定义下的数。复现：`python3 docs/audit_m17877/audit_probe2.py`（P1）。

**F2（high，结论过度｜可复现）** ⑤段"error decreases monotonically … confirming the droplet approaches true equilibrium"不成立。
- 交付只采样 4 个点：98.7% → 19.0% → 3.98% → 3.50% → 0.07%，本身已非单调收敛序列。
- 审计 1200–1999 步每 25 步重采样：误差 **0.235%–6.835%**（均值 2.667%），u_max **1.09e-6–3.76e-5** 有回升；第 1975 步误差 2.296%，第 1999 步 0.07%。"0.07%"是**单相位采样**，不是收敛值。诚实写法应为"时间带 2.7%（max 6.8%）< 15%"。

**F3（high，证据缺失）** TensorLBM 侧对标结论无任何工件。
- "built-in `free_energy_step` 相场从 [−1,1] 漂到 [0.46,0.71]" —— 工件中无该运行的任何输出；
- "CAC2D ran successfully" —— `run_cac2d_droplet` 的诊断（`xval_fingering_dynamics.py:325-336`）会被写进 `results["tensorlbm_cac2d"]`，但交付工件 `primary_results.json` 里没有该键（审计侧 `key_present=false`）；
- 结论：XVAL 实为**单向**——OSS 算法被读、被重写、被解析解检验，但两端**从未在同一算例上比过数**。adjacent 事实：`tensor_reports` 对该 mission **无记录**（`SELECT … WHERE mission_id=17724` → 0 行）。

**F4（medium，可复现性）** 入口 ≠ 工件。`__main__`(536) 走 run_cross_validation→convergence→sensitivity→plot→写 `xval_results.json`(577)；交付目录只有 `primary_results.json`，无脚本生成它 ⇒ 按文档 repro 路径重跑得到的是另一套（更全的）文件，交付物本身无法由入口复现。另：`fig3_laplace_vs_radius.png` 存在但收敛数值无工件，`fig4_sensitivity.png` 缺。

**F5（medium，不可交付风险）** 零 tracked 改动：`git -C /root/tp-agent/work/m17724 status --short` → 3 行 `??`；HEAD 无本单 commit；`evaluate result.head_commit=null, branch=null` ⇒ 平台 push 无从发生，工作树可回收 ⇒ 交付会蒸发。审计已把 4 个文件逐字节冻结进 `docs/audit_m17877/evidence_snapshot/`（sha256 见 `SHA256SUMS.txt`），评审可对快照复核。

**F6（low，已澄清为正）** 文档 ⑤ 的 "R=10, 800 步, 0.64%" 无工件，但审计重跑得 0.6434%（1.0064343e-3），**数值真实**，属"缺工件不缺事实"。

**F7（low）** 文档 ④ 表末行写 "Step 2000"，工件 `dp_history` 末点实际是 **step 1999**（`primary_results.json`）。

**F8（info，方法论）** 判据自校准：`κ=0.75σξ` ⇒ 连续极限 σ_model=(4/3)κ/ξ=σ_input（与 OSS 41-42 同式），故 "analytical σ/R" 不是外部锚，而是同一套参数的连续性断言。它仍有检验力（压力/力/模板错一处即偏），但不能算 L4 的"外部标杆"。审计补了两条独立路径（毛细能量 σ、OSS 压力定义重算）来解耦这一点。另注：OSS 仓 **无 LICENSE**（`/dev/shm/base/docs/oss_inventory/OSS-INV-Fingering_dynamics.md` 记 reference-only），本单仅移植算法（可），但报告未记许可提示。

### 未跑的检查（mark unknown, 不记 pass）
- `CAC2D` 静滴对标：未跑（交付自承"not directly comparable"，无判据）；`free_energy_step` 漂移主张：未复跑（无工件、无重跑脚本）。
- 真指进算例（含壁面/障碍/Zou-He）与 OSS 侧同算例对照：未跑（OSS 原码 H=W=380、MAX_T=1000 的算例不在本单提交范围，且无 ref 值）。
- GPU/CuPy 路径：未测（报告自承无 GPU）。
- 收敛阶（网格/时间步加密研究）：未测。

---

## 6. 处置（disposition）

**approve with notes → 判 L2**（消账不返修：核心 <15% 判据在两条独立路线下均成立且逐位可复现；缺的是集成/工件纪律，不是物理）。

返修建议（若后续战役重开该锚点）：
1. 把压力定义与 OSS 对齐或**显式登记差异**（F1），并对两条定义各报一列。
2. 报"时间带误差"而非单相位采样（F2），补稳态判据（如 dp 的滑动窗口幅度 < 判据）。
3. 补 OSS 侧同算例数值（哪怕小网格）与 `free_energy_step`/`CAC2D` 的对照工件（F3）。
4. 交付 commit 入库 + 入口/工件名对齐 + pytest 守护（F4/F5）。
5. 该锚点在 `tensor_benchmark_cases` 登记一行（metric: Laplace Δp vs σ/R；tol: 15%），否则以后每单都要人工兜裁判。

---

## 7. 工件清单（本审计工作树内，全部可复核）

- `docs/audit_m17877/audit_run.py` / `audit_rerun.json` —— 主算例 + R=10 独立重跑
- `docs/audit_m17877/audit_probe2.py` / `audit_probe2.json` —— OSS 压力定义变体 + 收敛时间带
- `docs/audit_m17877/gen_fig_summary.py` / `audit_summary.json` / `fig_m17877_evidence.png` —— 图与机读摘要（数字零手抄）
- `docs/audit_m17877/evidence_snapshot/{xval_fingering_dynamics.py, primary_results.json, XVAL_LBM_CUPY_DROPIN_NO_KERNEL_FUSION.md, fig3_laplace_vs_radius.png, SHA256SUMS.txt}` —— 受审交付逐字节冻结
- `verdict_m17877.sql` —— 落表语句（gate 可机器复核）