# 交差审计 m17582 — XVAL-LBM-ALLEN-CAHN-CSF-FORCE-CHAIN（受审 m17515）

* 受审工作树：`/root/tp-agent/work/m17515`
* 受审提交：`48ae879cc0714aa76d38ade3b1de8e39100b6a35` — 48ae879c XVAL-LBM-ALLEN-CAHN-CSF-FORCE-CHAIN round-2: register entry, wire operator into production Allen-Cahn kernel, machine-readable ref-vs-got benchmark
* 审计任务：m17582（board-audit）｜锚点：`XVAL-LBM-ALLEN-CAHN-CSF-FORCE-CHAIN`
* 审计脚本：`docs/board_audit/m17582/audit.py`（本文件由 `render.py` 从 `audit.json` 渲染）

## 0. 判定摘要

| 项 | 值 |
|---|---|
| 判定级别 | **L3**（L2 独立功能 + L3 系统集成成立；L4 基准验证不成立） |
| 机器裁判行 | `tensor_benchmark_cases` 该锚点 **0 行**（全表 55 行）→ 容差为作者自述 |
| 交付变体最差力场误差 | isotropic_d3q19 = 1.338e-14 |
| 生产默认变体最差力场误差 | central7 = 20.583%（> 15% 判据） |
| 越线算例 | C3-droplet-W3 |
| σ 反演最差误差 | -11.752%（<15% 通过；3% 目标 未达） |
| 可复现 | 审计员独立重跑：decision 全等=True，CSV 字节全等=True |
| 判据分辨力 | 5% 算子误差 → 8.337e-02/2.887e-02，仍 <15% |
| 处分 | request_changes（见 §7） |

## 1. 交付物三件套核验（存在性优先于数值）

| 项目 | 证据 | 结论 |
|---|---|---|
| 入口可执行 | `benchmarks/xval_allen_cahn_csf_force_chain/run.py`（29 行，转调 `tensorlbm.xval.allen_cahn_csf.main`） | 真入口 |
| 运行工件 | `artifacts/xval_allen_cahn_csf_force_chain/`：fig_force_chain_droplet.png, fig_force_chain_refinement.png, fig_force_error_slice.png + JSON/CSV/manifest | 存在 |
| 基准表 | `benchmarks/xval_allen_cahn_csf_force_chain/result.json`（由 `make_result.py` 从工件再生成） | 存在 |
| 复现命令 | `python benchmarks/xval_allen_cahn_csf_force_chain/run.py --out artifacts/xval_allen_cahn_csf_force_chain` | 审计员实跑通过（`repro/rerun.log`） |
| 部件指纹 | shipped JSON `0cd904073e5c320cbd30b0ea72eb8f29355e10030a73f64c83c069bb76b5f498` / 重跑 JSON `4bb1eb2ea58ce1aeab4d5954abebfc3253f10f041a40a3f6b753ea2bd9ef62c3` | 见 §3 |

## 2. ref-vs-got 逐项核（唯一裁判 = 15% 判据）

| 算例 | 变体 | rel L2 力 | 通过 15% | σ 反演误差 | σ 通过 |
|---|---|---|---|---|---|
| C1-planar | central7 | 5.020e-16 | ✔ | n/a | ✔ |
| C1-planar | isotropic_d3q19 | 2.918e-16 | ✔ | n/a | ✔ |
| C2-droplet-R16 | central7 | 1.121e-01 | ✔ | -4.002% | ✔ |
| C2-droplet-R16 | isotropic_d3q19 | 1.170e-14 | ✔ | -6.470% | ✔ |
| C3-droplet-W3 | central7 | 2.058e-01 | ✘ | -7.582% | ✔ |
| C3-droplet-W3 | isotropic_d3q19 | 6.372e-15 | ✔ | -11.752% | ✔ |
| C3-droplet-W4 | central7 | 1.121e-01 | ✔ | -4.002% | ✔ |
| C3-droplet-W4 | isotropic_d3q19 | 1.170e-14 | ✔ | -6.470% | ✔ |
| C3-droplet-W6 | central7 | 3.990e-02 | ✔ | -0.862% | ✔ |
| C3-droplet-W6 | isotropic_d3q19 | 1.338e-14 | ✔ | -1.976% | ✔ |
| C4-droplet-R8 | central7 | 6.881e-02 | ✔ | -2.248% | ✔ |
| C4-droplet-R8 | isotropic_d3q19 | 5.488e-15 | ✔ | -4.441% | ✔ |
| C4-droplet-R12 | central7 | 8.977e-02 | ✔ | -2.894% | ✔ |
| C4-droplet-R12 | isotropic_d3q19 | 7.912e-15 | ✔ | -5.123% | ✔ |
| C4-droplet-R16 | central7 | 1.121e-01 | ✔ | -4.002% | ✔ |
| C4-droplet-R16 | isotropic_d3q19 | 1.170e-14 | ✔ | -6.470% | ✔ |
| C5-droplet-sigma2e-02 | central7 | 1.121e-01 | ✔ | -4.002% | ✔ |
| C5-droplet-sigma2e-02 | isotropic_d3q19 | 1.170e-14 | ✔ | -6.470% | ✔ |
| C5-droplet-sigma5e-03 | central7 | 1.121e-01 | ✔ | -4.002% | ✔ |
| C5-droplet-sigma5e-03 | isotropic_d3q19 | 1.170e-14 | ✔ | -6.470% | ✔ |

* 表内 20 行（10 算例 × 2 变体），但**互异度量组仅 6 组**：['C2-droplet-R16', 'C3-droplet-W4', 'C4-droplet-R16', 'C5-droplet-sigma2e-02', 'C5-droplet-sigma5e-03'] 的 rel-L2 完全相同（σ 只做线性缩放，相对度量对其不变）→ 算例数被高报约 3.3×。
* `decision` 旗标 {'isotropic_within_15pct': True, 'central7_within_15pct': True}；与算例表矛盾的旗标：['central7_within_15pct']。decision[central7_within_15pct] is computed from C1-planar + C2 only (allen_cahn_csf.py:580-583) and reads True while C3-droplet-W3 central7 = 20.6% > 15%; the result.json headline does list the failing case
* 报告 headline 如实列出越线算例：`cases_central7_failing_15pct = ['C3-droplet-W3']`

## 3. 可复现性（审计员独立重跑）

* 重跑工件：`/root/tp-agent/work/m17582/docs/board_audit/m17582/repro/xval_allen_cahn_force_chain.json`（本次审计亲自执行）
* `decision` 块逐键全等 = **True**，最大绝对偏差 = 0.000e+00
* `kernel_integration` 全等 = **True**；CSV 字节全等 = **True**
* 数值块（cases/kernel_integration/decision/operator_weights/thresholds）全等 = **True**；整份 JSON 的 sha256 差异仅由 ['artifacts', 'repro_cmd']（输出目录路径/工件清单）造成。
* 结论：复现性为「实跑可复现」，不是文档承诺。

## 4. 物理锚（唯一非代码参照）

* σ 反演 vs 给定 σ：随 R 增大 **8:-4.44%, 12:-5.12%, 16:-6.47%**（W=4 固定）→ 误差随细化变大，未向 0 收敛。
* 随界面厚度：**W=3:-11.75%, W=4:-6.47%, W=6:-1.98%**（R=16 固定）→ 薄界面是弱项，最差 -11.75% 已逼近 15% 判据线。
* σ 误差在 lbmpy 侧自值完全相同（差异 ~1e-14）→ 该偏差是 CSF 力积分在该 W 下的离散固有偏差，双方共有。
* 仓库内已有 `src/tensorlbm/gas_layer/laplace.py, src/tensorlbm/phasefield/laplace_face.py`（Young–Laplace Δp 实测面），本次 XVAL **未接入**：即「与另一代码一致」有，而「与解析/实验一致」缺。

## 5. 集成活性（L3 的真问题）

* 生产调用点 8 处，其中**显式选择 operator 的仅 1 处**：
  - `src/tensorlbm/allen_cahn_lbm.py:385` force_mu_grad_phi( … operator= 是 )
  - `src/tensorlbm/allen_cahn_lbm.py:454` step_allen_cahn_field( … operator= **否** )
  - `src/tensorlbm/multiphase3d.py:852` force_minus_phi_grad_mu( … operator= **否** )
  - `src/tensorlbm/multiphase3d_d3q27.py:756` force_minus_phi_grad_mu( … operator= **否** )
  - `src/tensorlbm/phasefield/ch_validation.py:98` force_minus_phi_grad_mu( … operator= **否** )
  - `src/tensorlbm/phasefield/laplace_face.py:293` force_minus_phi_grad_mu( … operator= **否** )
  - `src/tensorlbm/phasefield/laplace_face.py:294` force_mu_grad_phi( … operator= **否** )
  - `src/tensorlbm/phasefield/static_droplet.py:178` force_minus_phi_grad_mu( … operator= **否** )
* 默认算子：central7 (allen_cahn_lbm.py:227/319, free_energy.py:99/113)。
* 耦合求解器：allen_cahn_lbm.py:360 raises unless allow_unverified_hydro=True (documented UNVERIFIED coupling); its AC call at :454 does not thread operator=
* 判读：算子**确实**被贯通进生产 Allen-Cahn 核（§6 的变异实验证明贯通是「活的」），但**被验证的那个变体在 8 个生产调用点里只有 1 个选中**；两个多相求解器仍走 central7，而 central7 在 C3-droplet-W3 的力场误差 20.6% 已越 15% 判据 → 改进「已交付、未部署」。

## 6. 判据分辨力（变异实验）

* 基线 rel L2 = 4.563e-15
* 拉普拉斯 ×1.05 → rel L2 = 8.337e-02
* 梯度 x 分量 ×1.05 → rel L2 = 2.887e-02
* 能看见 5% 算子误差 = **True** ⇒ 度量不是装饰；但 a 5% operator error surfaces as 2.9e-2/8.3e-2 -> would still pass the 15% gate; i.e. the gate tolerates ~9% operator error。

## 7. 分级判定与整改

**L2 ✔**：`phasefield/isotropic_fd.py`（真算子，含边界策略复用）+ `free_energy.py` 算子分派 + `allen_cahn_lbm.py` 贯通 + `xval/allen_cahn_csf.py`（lbmpy 符号表达式→numpy 求值 + 度量）——输入→计算→输出完整。

**L3 ✔**：生产核 `step_allen_cahn_field` 内 `force_mu_grad_phi(..., operator=operator)`（allen_cahn_lbm.py:385）被变异实验证明是活路径；13 项测试全绿（审计员实跑）。

**L4 ✘**（四条硬理由）：
1. 机器裁判缺行：`SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-ALLEN-CAHN-CSF-FORCE-CHAIN'` → **0 行**，15% 容差为作者自述，无机器可裁的参考值与容差登记。
2. 交付变体的一致性是「结构性的」：`operator_weights` 记录 isotropic 移植与 lbmpy 权重 max_abs_diff = 0，故 1.338e-14 只证明移植保真，不证明物理正确；真正独立的比较（central7 vs lbmpy）最差 20.583% 越线。
3. 物理锚只有 σ 反演，最差 -11.752%，且随 R 单调变差、不收敛；解析 Laplace 定律未纳入。
4. 无时间推进的耦合算例：`allen_cahn_step` 为 UNVERIFIED 硬门且不贯通 operator，因此「力链在求解器里被消费」这一点未被验证。
另附两条登记缺陷：算例数高报（20 行/6 互异）；`decision.central7_within_15pct=True` 与 C3-droplet-W3 的 20.583% 自相矛盾。

**升 L4 的最小整改清单**：
1. 在 `tensor_benchmark_cases` 登记该锚点一行（metric=rel_l2_force, tolerance=0.15, ref_source=lbmpy 2.0 sha256）。
2. 把 `isotropic_d3q19` 接进 `multiphase3d.py:852` / `multiphase3d_d3q27.py:756`（或给出保持 central7 的量化理由），并补一个走求解器的时间推进静滴算例（spurious current vs W）。
3. 用仓库既有 `phasefield/laplace_face.py` 做 Δp = 2σ/R 解析锚，误差逐项列出并回答是否 <3%。
4. 修 `decision` 旗标（用全表而非 C1/C2 两例计算）并去重算例表（σ 扫描标注为「相对度量不变」）。
