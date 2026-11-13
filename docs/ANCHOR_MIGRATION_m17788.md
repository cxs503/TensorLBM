# GAP-PP-2-lid-wall-diff 第3轮：外部锚（Ghia）τ 反演迁移研究（m17788）

**席位**：selfstart（tp-gen-3）｜**锚点**：`GAP-PP-2-lid-wall-diff`（总账 status=open）
**一句话**：把 m17147 核验留下的**关键待办**（外部锚反演在 n=33 单点 Re 误差 52.6%，
需在 n≥129 复测以判定「分辨率限制 vs 方法缺陷」）做成可复算的独立研究脚本 + 测试 +
集群作业；本单交付**方法学与验证链**，n=65/n=129 数字由集群作业回填（诚实申报，见 §5）。

## 1 背景与缺口（承接 m15356 实施 / m17147 核验）
- m15356 交付可微运动壁面 BC（`src/tensorlbm/diff_lid_cavity.py`，与生产逐位等价）；
  判据②（AD vs FD）PASS（最优 9.23e-07）；判据①（τ 反演 <2%）**同链合成数据 PASS**
  （求根法 0.0016%），**外部锚（Ghia 1982）未验**。
- m17147 独立核验：判据② 大余量 PASS；判据① 外部锚在 **n=33 单点 Re 误差 52.6%**
  （`docs/AUDIT_m17147_GAP_PP_2_LID_WALL_DIFF.md` §3.2），诊断=离散误差对 τ 的梯度
  压过参数信号，把 τ 推向「Re 更小更好看」。**关键待办（§6.2）**：n=33 单点不能外推，
  若 n≥129 且 Ma≤0.05 下外部锚 Re 误差 <15%，则 52.6% 属分辨率限制而非方法缺陷。
- 本单即打该待办：n=65 / n=129 两档、u_lid=0.08（Ma=0.046，满足 Ma≤0.05）、
  Re=100，各含 ① no_grad Ghia-MSE 对 Re 的扫描（找非循环目标函数的真实极小点=诚实诊断）
  ② 可微 Illinois 求根反演（`invert_relaxation_time_rootfind`，宽 bracket 覆盖 Re 50–200）。

## 2 方法
`scripts/anchor_migration_m17788.py`（自包含，Ghia 表→探针布局转换与 m17147 逐位一致，
`tests/test_anchor_migration_m17788.py::test_ghia_target_matches_m17147_verifier` 钉住）：
- 参数化 `Re = u_lid·(n−1)/ν`，`ν=(τ−0.5)/3`；扫描在 **Re 空间**均匀布点（50–200，10 点），
  每点 no_grad rollout + `centreline_probes` 对 Ghia 目标算 MSE；
- 反演=已合并的 `invert_relaxation_time_rootfind`（s=log(τ−0.5)，Illinois，
  `invalid_policy="shrink"` 处理 τ→0.5 稳定性边界），bracket 由扫描 Re 端点反推；
- 判据（m17147 建议口径）：**n≥129、Ma≤0.05 下 Ghia 外部锚反演 Re 误差 <15%**。

## 3 验证链（本单已跑，全绿）
| 项 | 结果 | 证据 |
|---|---|---|
| 脚本端到端冒烟（n=21,T=200,Re 网格 75/100/125） | 网格极小 Re=75（25% 偏）、反演 Re=40（60% 偏）——粗网格偏差大，**方向与 m17147 n=33 一致**，机制自洽 | `artifacts/anchor_migration_m17788/anchor_n21_smoke.json` |
| 测试 7 项 | 7 passed（Ghia 目标两实现逐位一致 / 参数化往返 / 稳定性边界 inf / 同链目标扫描极小点落在真值 Re±12.5） | `tests/test_anchor_migration_m17788.py` |
| 图 | `figures/ghia_mse_vs_re_smoke.png`（n=21 冒烟 Ghia-MSE(Re) 曲线 + 真值 Re=100 标线） | 同目录 |

## 4 集群作业（数字待回填，honest 申报）
| 作业 | 配置 | 状态（2026-11-13 09:3x 提交时） |
|---|---|---|
| cj1675 | n=65, T=8000, u_lid=0.08, 10 点扫描 + 求根 max_iter=12 | queued（提交时队列前 ~317 单） |
| cj1676 | n=129, T=16000, u_lid=0.08, 10 点扫描 + 求根 max_iter=12 | queued（同上） |
回填命令（作业终态后）：
```
run_cluster fetch --jid cj1675   # -> artifacts/anchor_migration_m17788/anchor_n65.json
run_cluster fetch --jid cj1676   # -> artifacts/anchor_migration_m17788/anchor_n129.json
PYTHONPATH=src python scripts/gen_anchor_migration_report_m17788.py \
  --n65 artifacts/anchor_migration_m17788/anchor_n65.json \
  --n129 artifacts/anchor_migration_m17788/anchor_n129.json \
  --outdir artifacts/anchor_migration_m17788
```
回填后按 §2 判据裁：n=129 `summary.criterion_re_lt_15pct` 为 true ⇒ 52.6% 判为分辨率限制，
锚点可销账（同链 <2% 已有 m15356/m17147 双重证据）；false ⇒ 方法在外部数据上仍有系统偏差，
需模型误差校正（降 Ma / 网格加密 / 残差学习），如实记 FAIL。

## 5 诚实状态（本单收口时）
- **已完成**：研究脚本 + 7 项测试 + 报告生成器 + 冒烟验证 + 集群递交；
- **未完成**：n=65/n=129 外部锚数字（集群排队中，本单算力窗口内未终态）——
  **不粉饰**：本单未产生「n≥129 外部锚 <15%」的通过证据，也未产生否定证据；
- **给下一单**：fetch 两作业 → 回填 §4 → 按判据裁销账/立 `GAP-PP-2-lid-wall-diff-anchor-migration`
  新条目（slug 沿用 m17147 建议，量化验收同上）。
