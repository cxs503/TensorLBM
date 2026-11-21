# 交差审计：m17366 XVAL-LBM-AKAI2018-GEOMETRICAL-WETTING（board-audit，非重做）

## ① 案例介绍
对象=lanl/MF-LBM 几何润湿边界条件（Akai et al. 2018, PRE 98 043302；`multiphase_3D/0.src/Phase_gradient.F90:209-265 @11be324`，BSD-3）移植进 TensorLBM 的 D2Q9 color-gradient 求解器。物理=n分量色梯度模型下壁面规定接触角 θ 的局部 ghost 条件：φ_s = φ_f + cotθ·|∂φ/∂t|（wall-normal 单侧差分）。算例族=96×64 网格、R=14 lattice、6000 步、θ_target ∈ {30,60,90} + 对照臂（wetting=False, θ=90）。

## ② 怎么做的（审计流程，非重做）
1. 定位被审对象：`work/m17366` @ `a2dcddc8`（分支工作树，非 main；main 无 `benchmarks/xval_akai2018_wetting/`）；
2. 逐项核三件套：impl / run_artifacts / repro_cmd；
3. 基准裁判：psql 直查 `tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-...-WETTING'` → **0 行**（全表 53 行，无 AKAI/WETTING 任何一提）；
4. 平台机器门：`GET /api/v1/tensor/bench/l4-verdict/17366` → `CLAIMED_NOT_PROVEN`, rows=0, failed_checks=[execution_success, source_cluster_job, cluster_job_bound, ref_got_error_present, within_tolerance]；`/tensor/repro/verdicts?mission_id=17366` → 0 条；
5. 复跑 repro：审计方在工作树内重跑解析脚本与单测（见 ⑤）。

## ③ 计算结果（全部从 out_*/results.json 程序化读取，零手抄）
| 臂 | θ_target | θ_local（壁面 BC 反读） | θ_circle（液滴圆拟合） | err% | |u|max |
|---|---|---|---|---|---|
| ctl60 | 90.0 | 90.00 | 142.42 | +58.25% | 3.47e-03 |
| gk30 | 30.0 | 30.00 | nan | +nan% | 1.04e-02 |
| gk60 | 60.0 | 60.00 | 124.60 | +107.66% | 4.97e-03 |
| gk60_R10 | 60.0 | 60.00 | 134.09 | +123.48% | 4.98e-03 |
| gk60_b090 | 60.0 | 60.00 | 101.49 | +69.15% | 5.36e-03 |
| gk90 | 90.0 | 90.00 | 52.54 | -41.62% | 1.83e-02 |

解析验证：`analytic_verification.json` n_cases=22，n_pass_15pct=22，max_rel_err=1.579e-14%，median_abs_err_deg=8.88e-16°。

## ④ 结果图
图1（左）求解器级 got vs target：壁面 BC 反读 θ_local 与规定角逐臂精确重合，但液滴外形圆拟合 θ_circle 完全跑偏（gk60 124.60°、gk90 52.54°、ctl60 142.42°、gk30 未定义）。图2（右）局部 BC 恒等式检查：22 例误差 ~1e-14（代数恒等，非独立验证）。
原因分析：|u|max 5e-3…1.8e-2 的伪流持续把液帽推离圆帽，cap 估计器全程 nan。

## ⑤ 注意点（审计意见）
- **analytic_verification 是循环自证**：`verify_bc_analytic.py` 注入解析 tanh 界面 → `geometrical_wetting_ghost` → `wall_local_contact_angle` 用同一构造的逆算子反读，故 0.00%/1e-14 是代数恒等式，**不能当作独立验证或基准**。它能证明"BC 代码按定义执行"，不能证明"润湿物理正确"。
- 该 anchor 在 `tensor_benchmark_cases` **无登记行**，即没有可作唯一裁判的预注册基准；按平台 L4 机器门此单为 CLAIMED_NOT_PROVEN（未在集群重算 ref/got/error）。
- 求解器级全部臂 FAIL（含对照臂），gk30 的 θ_circle 为 nan（圆拟合退化）—— **报告与日志如实写 honest-FAIL，无粉饰**，这一点审计确认与工件一致。
- 外部锚 `Phase_gradient.F90:209-265` 仅见代码 docstring 引用（commit 11be324），本地无 MF-LBM 源码副本，**未能独立复核** → 记 cited-unverified。
- 边界：本单不含 TensorLBM main 分支合入验证；main @ c05d7af3 无 `cg_wetting2d.py`，成果仅存在于 m17366 分支。

## ⑥ 误差判定
| 关键量 | 基准 | 实测 | 误差 | <3%？ | <15%？ |
|---|---|---|---|---|---|
| 局部壁面 BC 反读角 θ_local | 规定 θ（同构造逆算子） | 与规定值逐位相符 | ~1e-14% | 是（**但为代数恒等，无验证力**） | 是 |
| 端到端液滴接触角 θ_circle | 规定 θ=60 | 124.60° | **+107.66%** | 否 | **否** |
| 端到端 θ_circle | 规定 θ=90 | 52.54° | **−41.62%** | 否 | **否** |
| 端到端 θ_circle | 对照臂 θ=90 | 142.42° | **+58.25%** | 否 | **否** |
| 端到端 θ_circle | 规定 θ=30 | nan（圆拟合退化） | 不可定义 | 否 | 否 |

**显式回答：端到端关键量不满足 <3%，亦不满足 <15%（owner 精度标准），本单按判据为 honest-FAIL。**

## 判级（L0–L5）：**L3**
- L1 满足（数学/单元验证：局部 BC 恒等式 22/22、单测 12 passed）；
- L2 满足（真代码：`cg_wetting2d.py` 253 行，输入 φ/θ → 输出 ghost φ_s，非硬编码/随机）；
- L3 满足（接入主求解器：`run.py:175,186` → `color_gradient2d.py:127` `phi_solid` 可选参数，缺省 None 逐位兼容，6000 步真跑）；
- **L4 不满足**：无可裁判基准（`tensor_benchmark_cases` 0 行）+ 平台 L4 门 CLAIMED_NOT_PROVEN（rows=0）+ 端到端误差 41.6%…107.7% 远超 15%；
- L5 不满足。

处置：**reopened 建议**——L4 缺件清单=①在 `tensor_benchmark_cases` 登记该 anchor（case_name/metric/tolerance/ref_source）；②把端到端算例纳入集群重算产 ref/got/error 行；③修伪流/圆拟合（换 cap 或 trim 估计器、降伪流：提高 τ 或加界面厚度/改用 Guo 强制平衡）后重跑，误差达标方可报 L4。BC 层（L1-L3）保留，不返工。

审计独立性限制：审计方未参与 m17366 实施；对 m17366 分支只读复核（重跑脚本用 PYTHONPATH 指向该分支 src，产物写入本单 `audits/m17444_akai2018_repro/`）；无 MF-LBM 上游源码，外部锚未独立复核。
