# 交差审计 m18158 · 对象 m18106（锚点 XVAL-LBM-NO-BEM-CP-REPO-IN-OSS-INVENTORY）

**裁定：L2（confirmed at L2）· disposition = approve with notes + request changes**
审计者 board-audit / 证据包 `artifacts/m18106_audit/`（recompute.json + table.csv + 3 图 + sha256 provenance）

## ① 案例与对象
- 物理：二维差分加热方腔（DHC）自然对流，左壁 T=1 / 右壁 T=0 / 上下绝热，Pr=0.71，Ra=1e5；库入口
  `tensorlbm.thermal.simulate_natural_convection`（D2Q9 速度 + D2Q5 温度双分布 BGK、Guo 浮力、ABB 等温壁）。
- 外部锚：de Vahl Davis G. (1983) Int. J. Numer. Methods Fluids 3(3):249-264, Table 1 (Ra=1e5, Pr=0.71)——文献谱系锚（本地 `~/oss-cache` 无热双分布自然对流实现，与 m17996 否定性知识一致）。
- 被审交付：m18106（第 2 轮），交付树 `xval/XVAL-LBM-NO-BEM-CP-REPO-IN-OSS-INVENTORY/`，
  2 个 run 工件 + 预注册协议 + 未执行的 analyze/finalize/make_doc 管线；平台报告 tensor_reports id=16990。

## ② 怎么做的（审计方法）
1. **取证**：逐文件 sha256 入 `recompute.json.provenance`（12 个交付文件；含 2 个 0 字节日志）。
2. **独立复跑**：`audit_repro_run.py` 按交付配置逐字重跑（N=64、τ=0.6、250k 步上限、stop_tol 1e-3、patience 5），
   并额外读取交付驱动从未读的库字段 `u_max_x / u_max_x_neg`（速度分量），另加 N=32 一档。
3. **双口径比价**：同一原始数据分别按交付映射（max|u| 矢量模）与参考定义（max|u_x| 分量）对锚。
4. **基准裁判**：`SELECT ... FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-NO-BEM-CP-REPO-IN-OSS-INVENTORY'`
   → 交付时 **0 行**（裁判缺失，本审计补注册 3 行：1 passed / 2 failed）。
5. **判级**：按 maturity_gate JUDGE_SCHEME（L2=impl_files+entry+run_artifacts；L3 另需 integration_callers；L4 另需 benchmark+repro_cmd）。

## ③ 计算结果（程序化注入，零手抄）
| run | metric | ref | got | err | band(15%/3%) |
|---|---|---|---|---|---|
| n64_tau0p6 | nu_grad2 | 4.5190 | 4.0342 | -10.728% | PASS_15PCT |
| n64_tau0p6 | u_max_nd__delivered_mapping | 34.7200 | 66.5765 | +91.752% | FAIL_15PCT |
| n64_tau0p6 | v_max_nd | 68.6000 | 65.1876 | -4.974% | PASS_15PCT |
| n64_tau0p8 | nu_grad2 | 4.5190 | 3.6774 | -18.623% | FAIL_15PCT |
| n64_tau0p8 | u_max_nd__delivered_mapping | 34.7200 | 77.3989 | +122.923% | FAIL_15PCT |
| n64_tau0p8 | v_max_nd | 68.6000 | 72.1406 | +5.161% | PASS_15PCT |
| repro_n32_tau0p6 | nu_grad2 | 4.5190 | 3.4442 | -23.785% | FAIL_15PCT |
| repro_n32_tau0p6 | u_max_nd__delivered_mapping | 34.7200 | 70.0266 | +101.689% | FAIL_15PCT |
| repro_n32_tau0p6 | v_max_nd | 68.6000 | 64.0557 | -6.624% | PASS_15PCT |
| repro_n32_tau0p6 | u_max_nd__component_max_abs_ux | 34.7200 | 50.4737 | +45.374% | FAIL_15PCT |
| repro_n64_tau0p6 | nu_grad2 | 4.5190 | 4.0342 | -10.728% | PASS_15PCT |
| repro_n64_tau0p6 | u_max_nd__delivered_mapping | 34.7200 | 66.5765 | +91.752% | FAIL_15PCT |
| repro_n64_tau0p6 | v_max_nd | 68.6000 | 65.1876 | -4.974% | PASS_15PCT |
| repro_n64_tau0p6 | u_max_nd__component_max_abs_ux | 34.7200 | 43.7451 | +25.994% | FAIL_15PCT |

- **复现性**：审计复跑 Nu = 4.03422，交付 Nu = 4.03422，
  偏差 +0.0000%（steps_done 同为 24000，211.7 s）→ **逐位复现**。
- **收敛趋势**（τ=0.6，Nu 误差）：N=32 -23.785% → N=64 -10.728%（单调改善）；
  定义对齐的 u_max 误差：N=32 +45.374% → N=64 +25.994%。
- **u_max 定义错配**：交付 `run_xval.py` 用库字段 `u_max`（= max√(ux²+uy²)，矢量模）对参考 34.72（= max|u_x|，分量）；
  模/分量比 = {'N32_tau0.6': 1.3874, 'N64_tau0.6': 1.5219}（按 max|u_x| 基准 1.491/1.539；max|u| ≥ max|u_x| 恒成立 ⇒ 系统性高估，属定义性而非收敛性）；且定义对齐误差随加密单调下降（+45.374% → +25.994%）。
- **τ 敏感性**：Nu(τ=0.6) vs Nu(τ=0.8) 跨度 8.844%，且 τ=0.8 臂 -18.623% 超出 owner 15% 带（该臂 ma=0.295、T_max=1.0343 过冲）。
- **协议完成度**：预注册 3 档网格 [64, 96, 128]，实产 JSON = ['n64_ra1e5.json', 'n64_ra1e5_tau0p8.json']；
  0 字节日志 = ['runs/n96_ra1e5.log', 'runs/n128_ra1e5.log', 'runs/finalize.log']；`summary.json`=False，
  `oss_survey.json`=False，图 = []。
- **持久性**：m18106 工作树 HEAD=fc86f5080e86bf1f164e6a9b355eca2fe5f1cc6d、branch=agent/m18106，
  本锚点 xval 已跟踪文件 = 0 个，未跟踪源码 = 5 个，且 `runs/` 目录被 `.gitignore:108` 忽略 ⇒ **交付未入库，run 工件即使提交也推不上**。
- **旁证（真伪交叉核）**：仓库既有 `benchmarks/verified/thermal_cavity/result.json`（m15868 入库，独立驱动）给 Nu=4.0736，
  与 m18106 的 4.0342 相差 1.0% ⇒ 交付数字为真实计算产物；同时 `tests/test_thermal_cavity_ra1e5.py` 对同一锚声明 8% 容差却 **2 failed / 10 passed / 1 skipped**（仓库侧既存缺陷，非本单引入）。

## ④ 图
- `fig1_nu_history.png`：两条交付 run + 审计复跑的 Nu(t) 与 4.519 参考线、±15%/±3% 带。
- `fig2_ref_vs_got.png`：N=64 τ=0.6 三个预注册指标的 ref vs got，并排展示「交付映射（max|u|）」与「定义对齐（max|u_x|）」两读。
- `fig3_convergence_and_missing_grids.png`：误差随网格单调收敛，N=96/128 预注册格点标红「NOT RUN」。
- 无场量云图：库入口 `simulate_natural_convection` **只返回标量统计、不返回场数组**（thermal.py:331–471），
  出云图需重写主循环 = 破坏审计独立性，故不做（已记 checks_not_run）。

## ⑤ 注意点 / 局限
- 速度参考值的转写：无离线 PDF，参考 34.72/68.60 依赖仓内两张独立表 + 交付 predeclared.json 转写（交付方自行标 WITHHELD，处理正确）。
- 审计只复跑 τ=0.6；τ=0.8 臂仅按交付原始 JSON 复算（未逐位复核）。
- 未重做 N=96/128（属交付方范围，属返修项，不属审计范围）。
- 既有仓库测试 `test_thermal_cavity_ra1e5.py` 的 8% 容差与实测 ~10% 冲突，是仓侧独立待修项（F6，非本单引入）。

## ⑥ 误差判定（逐项，显式回答是否 <3%）
| 量 | 误差 | <3%? | <15%? |
|---|---|---|---|
| Nu（N=64, τ=0.6，主指标） | -10.728% | **否** | 是 |
| Nu（N=64, τ=0.8） | -18.623% | 否 | **否** |
| Nu（N=32, τ=0.6，审计） | -23.785% | 否 | 否 |
| u_max（交付映射 max|u|） | +91.752% | 否 | 否（且定义错配，数值无效） |
| u_max（定义对齐 max|u_x|） | +25.994% | 否 | 否 |
| v_max（max u_y） | -4.974% | 否 | 是 |

**结论（honest）**：无一指标进入 3% 口径；主指标 Nu 只在 N=64/τ=0.6 过 15% 线；u_max 真误差 +26%（非交付声称的 +92%）。
交付代码真实、可逐位复现（L2 成立），但基准验证未完成（L4 不成立：3 档网格只出 1 档、分析层未跑、1/3 指标定义错配、τ=0.8 越带、交付未入库）。

## 判级与落库
- `tensor_maturity_verdicts`：repo=TensorLBM, scope=`m18106 XVAL-LBM-NO-BEM-CP-REPO-IN-OSS-INVENTORY`, mission_id=18158、level=2、verdict_by=board-audit
  （id=707；用 maturity_gate.judge(2, evidence, 'TensorLBM', 18158) 本地预演 → (2, []) 无缺项）。
- `tensor_benchmark_cases`：补注册 3 行（Nu@τ=0.6 passed；Nu@τ=0.8 failed；u_max 定义对齐 failed）——交付时该锚点裁判为空。
- 返修建议：F1 改指标为 max|u_x|（库已暴露 u_max_x/u_max_x_neg）；F2 补跑 N=96/128 + analyze 管线；F3 提交入库；F4 解释或删除 τ=0.8 臂。
