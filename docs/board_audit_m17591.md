# 交差审计 m17591 · XVAL-LBM-BITWISE-REFDATA-REGRESSION-PLUS-BENCHMAR（TensorLBM）

审计单: m17630（board-audit）· 被审: m17591（status=succeeded, auto-approve dev_acceptance）
审计人: board-audit · 日期: 2026-11-13 · 判级: **L2**（verdict 已落 tensor_maturity_verdicts）

## ① 案例介绍（物理背景/几何/参数）

m17591 交付一个统一交叉验证框架 `tensorlbm.verification.xval`（bitwise 复现 + 参考数据回归 + 基准计时），
含 3 个算例：
- Poiseuille 2D（H=16, τ=0.8, g=1e-5, ν=0.1, Re=0.128, 5000 步, Guo 强迫, no-slip）——解析解 u(y)=g/(2ν)y(H-y)；
- Lid-driven cavity Re=100（n=33 测试 / n=65 注册, u_lid=0.1, Ma≈0.173）——Ghia et al. (1982) JCP 48:387 参考表；
- Womersley 脉动 α=5（H=59, u_peak=0.03）——注册但**未测**（报告自认）。

## ② 怎么做的（审计方法/流程/工具链）

1. 取证：读 m17591 工作树（/root/tp-agent/work/m17591）、mission result（API /tensor/missions/17591）、
   报告 docs/XVAL_LBM_REPORT.md、5 个新模块 + 测试 + 2 脚本 + refdata。
2. 基准裁判：`SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-BITWISE-REFDATA-REGRESSION-PLUS-BENCHMAR'`
   → **0 行**（表内 55 行，无此锚点；LBM 相关锚点仅 GAP-LBM-MCMP-POISEUILLE / GAP-LBM-SPHERE / CASE-LBM-SQUARE-CYL-NORBERG）。
3. 独立复跑：在 m17591 工作树跑 `pytest tests/test_xval_bitwise_regression.py -v`（90.40s）；
   另写独立脚本 /tmp/audit_xval_numbers.py 重算全部关键量（Poiseuille×2 次、cavity n=33×3000 步、
   cavity 500 步×2 次哈希、H=8/16/32 收敛），零手抄，数字见 runs/m17630_audit/repro_evidence.json。
4. 集成面核查：grep 全仓 xval 调用方；集群作业日志全量 grep（/root/tp-agent/cluster/jobs/）。
5. 判级对照：同族已裁 verdict（m17520 L3 confirmed / m17605 L3 pending / m17520 L4 WITHHELD 理由）。

## ③ 计算结果（审计复跑数字，程序化注入）

| 量 | 报告值 | 本单复跑 | 一致 |
|---|---|---|---|
| Poiseuille ε∞ | 0.003608 (0.361%) | 0.003607843111901292 | ✓ |
| Poiseuille ε₂ | 0.004921 (0.492%) | 0.004920911776990591 | ✓ |
| mass drift | 3.08e-13 | 3.0808688933348094e-13 | ✓ |
| Cavity L₂(u) n=33 | 0.0880 (8.80%) | 0.08801679484372127 | ✓ |
| Cavity L₂(v) n=33 | 0.0658 (6.58%) | 0.06580940145056142 | ✓ |
| Cavity max(u) | 0.1183 (11.83%) | 0.11829614445307896 | ✓ |
| Cavity max(v) | 0.0974 (9.74%) | 0.09737130321716775 | ✓ |
| Bitwise Poiseuille | c8acf2cc…1188d | c8acf2cc9ab91154…f7338651188d（前后缀全对） | ✓ |
| Bitwise Cavity | 59f81e58…cd7e | 59f81e58d6a767de…c4b0cd7e（前后缀全对） | ✓ |
| 收敛 e8/e16/e32 | 1.460/0.361/0.090% | 1.4603/0.3608/0.0899% | ✓ |
| 收敛比 8→16 / 16→32 | 4.05 / 4.01 | 4.0476 / 4.0118 | ✓ |
| pytest | 29/29 (91s) | 29 passed (90.40s) | ✓ |

工件核验：artifacts/xval/ 4 张 PNG 均有效（file 校验）；refdata/poiseuille_2d_H16_v1.npz + manifest.json
（sha256/scalars 与复跑一致）；Ghia 参考数据=既有模块 lid_driven_cavity/reference_data.py 硬编码 Ghia 1982 表
（u(y=0.5)=-0.15612 与发表值一致，非本单新造）。

## ④ 结果图（被审工件图，非本单新生成）

- xval_poiseuille_profile.png：Poiseuille 速度剖面 vs 解析解（误差 0.36%）
- xval_cavity_profile.png：cavity 中线速度 vs Ghia 1982
- xval_convergence.png：H=8/16/32 二阶收敛
- xval_bitwise_repro.png：双跑哈希一致
（本单为审计单，图=被审工件原图，随报告内嵌。）

## ⑤ 注意点（发现清单，按严重度）

- **F1（L4 阻断）**：tensor_benchmark_cases 对该锚点 0 行——基准裁判无正式参考行。
  对照已确认先例 m17520（L3 confirmed）："L4 WITHHELD: referee table empty (0 registered benchmark cases)"。
- **F2（L3 阻断）**：xval 为独立验证包，未接入主求解器/CI/平台；全仓调用方仅 2 个脚本 + 测试文件
  （grep 证据：scripts/run_xval_cavity_benchmark.py:23,26 与 scripts/generate_xval_figures.py:213）。
  对照 m17520（L3 confirmed）要求 "genuine runtime integration into the production … path"——本单不满足。
- **F3（流程缺陷，重大）**：分支 agent/m17591 **零 commit**（git log main..HEAD 为空），10 个交付文件全部
  untracked（git status 全 `??`）。"succeeded" 但无 git 历史，平台自动推送无物可推；本单证据指针指向
  m17591 工作树内未跟踪文件。
- **F4（声明无据）**：报告称 n=65 "submitted to cluster"——集群作业日志全量 grep 无此作业，无 n=65 工件；
  n=65 case 注册但从未运行（测试套件只跑 n=33）。
- **F5（轻微）**：报告首行标题截断（"ISE-REFDATA-REGRESSION-PLUS-BENCHMAR"）。
- **F6（轻微）**：cavity 对照表 "Reference (Ghia)=0.0" 为设计使然（误差量=偏差本身，参考标量置 0），
  真实 Ghia 对比经 compute_errors 完成，数字真实；但表格呈现易误读。
- **F7（轻微）**：Womersley 算例注册未测（报告自认）；benchmark 计时值（1.291s/0.020s/0.084 MLUPS）
  机器相关，本单未独立复测（计时非验收量，基础设施 3 项测试绿）。

## ⑥ 误差判定（关键量 vs 基准，逐项）

| 判据 | 容差 | got | 判定 |
|---|---|---|---|
| Poiseuille ε∞ | <15%（锚点总账）/ <2%（case 内紧门） | 0.361% | ✓ <3% |
| Poiseuille ε₂ | <15% / <2% | 0.492% | ✓ <3% |
| Cavity L₂(u) | <15% | 8.80% | ✗ 不 <3%（<15% 达标） |
| Cavity L₂(v) | <15% | 6.58% | ✗ 不 <3%（<15% 达标） |
| Bitwise 复现 | 哈希全等 | 全等 | ✓ |
| 收敛阶 | ~4（二阶） | 4.05/4.01 | ✓ |

**显式回答**：Poiseuille 两量 <3% 成立；Cavity 两量 8.80%/6.58% **不 <3%**（n=33 粗网格，报告未声明 3% 线，
锚点总账线 15% 内达标）。基准裁判（tensor_benchmark_cases）对该锚点无行，正式裁判缺位——
got 数值真实可复现，但"基准判据唯一裁判"无法逐行打勾，按先例 L4 不成立。

## 判级结论

**L2**（独立功能实现：真代码 输入→计算→输出，29 测试绿，全部数字独立复现）。
- L3 不成立：无主求解器/平台集成（F2）。
- L4 不成立：基准裁判表 0 行（F1）+ n=65 腿缺失（F4）+ Womersley 腿未测（F7）。
- 附 4 图 + 147 行报告不改变级别（级别=功能实质）。
- 消账条件建议：① 补 commit 入分支（F3）；② 注册 benchmark case（F1）；③ 跑 n=65 腿并留工件（F4）。
