# m18157 交差核验：m17996（XVAL-LBM-NEGATIVE-NO-OSS-REFERENCE）到底交付了什么

**结论一句话**：交付是**真的**——真 lbmpy 2.0 双臂对照、8 个算例格、产物 90 个文件已入 git、独立复跑 K(t) **逐点 bitwise 一致**、16 个 claim 全部 <15%（worst 0.1560%）；但级别只能是 **L2**：没有主求解器/平台调用方（L3 不着）、平台基准裁判 `tensor_benchmark_cases` 该 anchor **0 行**（L4 无机器裁判行），且分支未并入生态 main。

## ① 案例介绍（物理背景/几何/参数表）

被审对象是 2D Taylor–Green 涡衰减：周期方盒 `[0,L)²`、D2Q9 单松弛 BGK/SRT、ρ0=1、dx=dt=1、格心 `x_i=i+0.5`、float64、`f=f_eq` 精确初始化。闭式精确解 `u=-U0 cos(kx)sin(ky)e^{-2νk²t}`、`v=+U0 sin(kx)cos(ky)e^{-2νk²t}`、`K=K0 e^{-4νk²t}`（不可压 NS 精确解，对流项为纯梯度被压力吸收）。

| 格 | n | U0 | τ | ν | Re | Ma | k | 步 |
|---|---|---|---|---|---|---|---|---|
| main | 64 | 0.050 | 0.80 | 0.10000 | 32.0 | 0.0866 | 0.09817 | 400 |
| conv_n32 | 32 | 0.050 | 0.80 | 0.10000 | 16.0 | 0.0866 | 0.19635 | 400 |
| conv_n128 | 128 | 0.050 | 0.80 | 0.10000 | 64.0 | 0.0866 | 0.04909 | 400 |
| sens_u0_0.020 | 64 | 0.020 | 0.80 | 0.10000 | 12.8 | 0.0346 | 0.09817 | 400 |
| sens_u0_0.100 | 64 | 0.100 | 0.80 | 0.10000 | 64.0 | 0.1732 | 0.09817 | 400 |
| sens_tau_0.70 | 64 | 0.050 | 0.70 | 0.06667 | 48.0 | 0.0866 | 0.09817 | 400 |
| sens_tau_1.00 | 64 | 0.050 | 1.00 | 0.16667 | 19.2 | 0.0866 | 0.09817 | 400 |
| matched_n128 | 128 | 0.050 | 0.80 | 0.10000 | 64.0 | 0.0866 | 0.04909 | 1600 |

参考臂 = `lbmpy 2.0`（FAU/pystencils，Apache-2.0；sha256 `080a6d37…`，路径 `/usr/lib/python3.11/site-packages/lbmpy/__init__.py`）。annotated anchor 语义：**参考不可解析就必须 WITHHELD + 负知识登记，不许凭"看着差不多"放行**。

## ② 怎么做的（核验方法，独立复算优先）

1. **三件套取证**：被审工作树 `/root/tp-agent/work/m17996`（commit `af7b184c`）逐项核 impl/工件/测试/命令；`git ls-files | grep -c '^artifacts/xval_negative_no_oss_reference/'` = **90**（产物真进了 commit，不只是磁盘残留）。
2. **不信报告，自己重算**：本单脚本 `docs/board_audit/m18157/verify_m17996.py` 自带 γ_K 拟合（ln K 对 t，窗口 K/K0∈[0.2,0.9]）、自带场 L2 与朝向自检、自带 CSV 解析，不调用被审脚本的任何误差函数。
3. **独立复跑**：把 `scripts/` + `src/` + `tests/` + `xval/<anchor>/` 拷到 `/tmp/rep17996` 只读复跑 → 与原件 K(t) 逐点比。
4. **对抗性负向控制**：直接 import 被审 `XvalReferenceGate`，造探针"参考不可用 + 误差 0.00%"，看它敢不敢 PASS；再造伪造 analytic id、空 reference_id、15.1%/14.9% 边界。
5. **平台裁判对表**：SQL 查 `tensor_benchmark_cases`（anchor 行数）、`negative_knowledge_ledger`（注册是否真发生）、`tensor_reports`（六段与图数）。
6. 级别按【功能实质】给，并按 gate 方案（L2=impl+entry+artifacts；L3 加 integration_callers；L4 加 benchmark+repro）自查，不凑级。

## ③ 计算结果（程序化注入，本单重算 vs 报告）

| 格 | n | γ_theory=4νk² | γ in-house(重算) | γ lbmpy(重算) | 报告 in-house vs 理论 % | 重算 in-house vs 理论 % | γ 双臂相对差 % | u_max 相对差 % | 场 L2 % (报告) | 场 L2 % (重算) |
|---|---|---|---|---|---|---|---|---|---|---|
| main | 64 | 3.855314e-03 | 3.853514e-03 | 3.853502e-03 | +0.0467 | -0.0467 | 0.000308 | 0.000444 | 0.0217 | 0.0217 |
| conv_n32 | 32 | 1.542126e-02 | 1.542653e-02 | 1.542648e-02 | +0.0342 | +0.0342 | 0.000296 | 0.000061 | 0.0010 | 0.0010 |
| conv_n128 | 128 | 9.638286e-04 | 9.650667e-04 | 9.651108e-04 | +0.1285 | +0.1285 | 0.004571 | 0.010438 | 0.1560 | 0.1560 |
| sens_u0_0.020 | 64 | 3.855314e-03 | 3.855959e-03 | 3.855959e-03 | +0.0167 | +0.0167 | 0.000009 | 0.000006 | 0.0035 | 0.0035 |
| sens_u0_0.100 | 64 | 3.855314e-03 | 3.845072e-03 | 3.844862e-03 | +0.2657 | -0.2657 | 0.005473 | 0.009453 | 0.0889 | 0.0889 |
| sens_tau_0.70 | 64 | 2.570209e-03 | 2.568775e-03 | 2.568766e-03 | +0.0558 | -0.0558 | 0.000340 | 0.002322 | 0.0351 | 0.0351 |
| sens_tau_1.00 | 64 | 6.425524e-03 | 6.425241e-03 | 6.425208e-03 | +0.0044 | -0.0044 | 0.000505 | 0.000004 | 0.0084 | 0.0084 |
| matched_n128 | 128 | 9.638286e-04 | 9.632965e-04 | 9.632958e-04 | +0.0552 | -0.0552 | 0.000071 | 0.001314 | 0.0598 | 0.0598 |

- **报告 vs 独立重算最大偏差**：γ vs 理论 `5.31e-01` 百分点；γ 双臂相对差 `9.21e-13` 百分点；场 L2 `2.78e-16` 百分点 → 报告数字**不是手写**。（in-house 对理论 5 格符号相反、|值|相同，属符号约定差异，不构成虚标。）
- **朝向自检复现**：main 初场 direct `7.126e-14%` vs transposed `141.421%` → 转置约定判定成立，不是"对称所以看着像"。
- **产物自洽**：manifest 3/3 文件 sha256 相符；probe 记录 `record_sha256=813fc5ce…`。
- **负知识台账**：`negative_knowledge_ledger` 实注册 **6** 条（id 516–521，registered_by=`tensorlbm.xval.oss_reference`），对应 6 个无 OSS 参考能力（shan_chen_pseudopotential_multiphase, free_surface_volume_tracking, immersed_boundary_ibm, adaptive_mesh_refinement, thermal_energy_transport_lbm, acoustic_pulse_absorbing_boundary）。

## ④ 图

1. **fig1**（左）K(t) 半对数：in-house / lbmpy / 解析衰减率三线重合到看不出差别；（右）各格场 L2 相对差与 15% 容差线，worst 出现在 n=128（0.1560%），说明两臂离散细节差异随分辨率略升但仍差两个量级。
2. **fig2**（左）报告值 vs 本单独立重算散点（全落在 y=x）；（右）对抗性负向控制 5 项的裁定结果。

## ⑤ 注意点（坑与局限）

- **γ_K 的 3e-6 一致性近乎同源**：两臂都是同 τ 的 BGK/SRT，且该算例衰减率解析精确，所以真正的判别量是场 L2（worst 0.156%），不是 γ。报告没有夸大这点，核验时也不应把它当"独立实现互证强"的证据。
- **没有主求解器调用方**：`run_lbmpy` 定义在 `src/tensorlbm/xval/runners.py:547`，但 `src/tensorlbm/xval/campaign.py:31` 只 import `run_tensorlbm/run_palabos/run_openfoam`，cli 也没导出 `oss_reference` → 能力停留在实验脚本层，故 L3 不达。
- **平台基准裁判空表**：`tensor_benchmark_cases` 中该 anchor **0 行**（全表 NEGATIVE 类亦 0 行）。数值达标是"报告自证 + 本单重算复核"，不是平台机器裁判行 → L4 不能成立。建议注册 `main`/`conv_n128` 两格（metric=`field_l2_rel_pct`, tolerance=15%）。
- **未并入 main**：`git -C /root/tensor-ecosystem/TensorLBM merge-base --is-ancestor af7b184c HEAD` → **NOT_MERGED**；交付只在 `agent/m17996`。
- **摘要约定小瑕**：`manifest.probe_sha256` 用的是 probe 内部 `record_sha256`，不是文件 sha256（`0287bd08…`），协议没写 canonicalization，第三方无法独立复算该摘要。
- **run_info 不可 bitwise 复现**：`run_info_lbmpy.json` 的 `method` 字段是 object repr（含 `0x…` 内存地址）；KPI 与场 CSV 仍逐点一致。

## ⑥ 误差判定（是否 <3%？逐项回答）

| 关键量 | 基准 | 偏差 | <3%? |
|---|---|---|---|
| γ_K（8 格，in-house vs 闭式精确解） | 4νk² | 最大 0.2657% | ✅ 是 |
| γ_K（8 格，lbmpy vs 闭式精确解） | 4νk² | 最大 0.2711% | ✅ 是 |
| 场 L2（in-house vs lbmpy，8 格） | lbmpy 2.0 | 最大 0.1560%（n=128） | ✅ 是 |
| 初场朝向自检（direct vs transposed） | 同一公式 | 7.13e-14% vs 141.4% | ✅ 是（差 15 个量级） |
| 复跑可复现性（K(t) 逐点） | 原件 | 16/16 臂 bitwise 相同 | ✅ 是 |

**误差判定结论**：所有关键量对基准误差均 **<3%**（最大 0.1560%），远低于 owner 15% 带。**但**：这是数值证据，不是平台基准裁判行；报告未夸大，级别仍按功能实质给 L2。

---

审计证据：`docs/board_audit/m18157/verify_output.json`、`insert_verdict.sql`（verdict 行 id=695）、`evidence/`（summary/probe/coverage/payloads/原始 CSV 归档副本）、本报告图 2 张。被审单测试实跑：`18 passed, 1 warning in 15.43s`。
