# 交差核验 m17147 — GAP-PP-2-lid-wall-diff（可微运动壁面 BC / 方腔 τ 反演）

**席位**：verifier（tp-ver-4）｜**对象**：`src/tensorlbm/diff_lid_cavity.py`（已合并）
+ `docs/GAP_PP_2_LID_WALL_DIFF.md`（第 1/2 轮实施报告）
**结论一句话**：判据②（梯度 AD vs FD）**独立复现 PASS 且余量极大**（Richardson 外推 1.7e-11）；
判据①（τ 反演 <2%）**只在「同链合成数据」这一自洽解读下成立**——用本仓唯一的**外部锚**
（Ghia 1982 Re=100 表）做反演，Re 恢复误差 **52.6%**（n=33），属**声明范围过宽**，
不是代码缺陷（disposition：approve with notes）。

## 1 独立性声明与范围
* 本次核验**不使用**实施方驱动脚本 `scripts/run_gap_pp2_lid_wall_diff.py`，另写
  `scripts/verify_lid_wall_diff_m17147.py`（同一模块，独立功能量/独立算例参数）。
* 工件目录 `artifacts/gap_pp2_lid_wall_diff/`（实施方声称的 steady/sweep/fdcheck/invert 全部
  JSON/pt）**在本分支不存在**（`ls` = No such file），即原判据证据链**不可复算**，
  本轮全部数字由本单脚本重新生成并落 `artifacts/verify_m17147/`。
* 复算分辨率/步数**有意不同**于实施方（n=32/33 vs 32/65/48），以检验机制而非复述数字。
* 未做：n=65/129 稳态 Ghia RMSE 全分辨率阶梯（超单干算力与时窗，见 §6）。

## 2 判据② 梯度 AD vs FD — 独立复现 **PASS（大余量）**
功能量**不是**实施的「中线探针 MSE」，而是内部区失配
`J = mean[(ux-ux_ref)²+(uy-uy_ref)²]`（参考场取 τ_true+0.02、u=1.2u_lid，故评估点非极小点，
导数不退化）；中心差分 h 阶梯 + Richardson 外推（`fd_n32_t600.json`，n=32,T=600,float64）。

| 参数 | AD | FD h=1e-4 rel err | FD h=1e-5 rel err | Richardson rel err |
|---|---|---|---|---|
| dJ/dτ | -7.9909e-04 | 1.11e-05 | 1.12e-07 | **8.79e-12** |
| dJ/du_lid | -3.0215e-03 | 5.45e-08 | 5.62e-10 | **1.70e-11** |

* 误差随 h 二阶下降（h 减 10 倍误差减 ~100 倍），外推与 AD 一致到 1e-11 →
  **梯度实现正确的强证据**（单一估计器不可能自证此台阶）。
* **注意**：判据「<1e-4」只在 **h ≤ 1e-4** 成立；本算例 h=1e-3 时 dτ 的 FD 相对误差
  1.11e-03、h=1e-2 时 **1.15e-01**——这是中心差分的截断误差，不是 AD 误差。
  引用该判据时必须同时写 h。
* 报告 JSON 里的 `worst_fd_rel_err_h_ge_1e-4`=0.115 是本脚本聚合式的**口径残留**
  （把大 h 也纳入了“h≥1e-4”），读取时以上表为准（诚实标注，不改数）。
* 二阶引擎交叉校验：`torch.func.jvp`（前向模式）在本环境报
  `AttributeError: 'function' object has no attribute 'graph'`；复步（complex step，无截断）
  不可用——`solver.collide_bgk` 的 `torch.clamp(rho,min=1e-12)` 无复数核。二者**记为未跑**，
  不当作通过（`second_engine_checks` 字段）。Taylor/割线一致性作为替代：Δ=1e-4 时
  线性预测残差 1.8e-12（相对 J 2.6e-05），residual/Δ² 在 Δ=1e-2→1e-3 稳定 ⇒ 一阶项正确。

## 3 判据① τ 反演 — **两种解读，结论相反**
### 3.1 同链（truth=同一可微链生成）— **PASS，机制可独立复现**
`repro_n33_t1500.json`：n=33（H=32），T=1500，τ_true=0.596（Re=100），τ_init=0.60，
Illinois 求根 dL/ds=0，11 次求值，247 s。

| 量 | 值 |
|---|---|
| τ_rec | 0.59599785 |
| τ 相对误差 | **3.60e-06**（<2% ✓） |
| Re_rec | 100.0022（真值 100） |
| 残差 RMS | 1.63e-06 |

⇒ 实施方「求根法收口」的机制在本席**独立脚本、不同分辨率**上复现；其 n=65 报的 0.0016%
量级可信。

### 3.2 外部锚（truth=Ghia 1982 Re=100 表，模型没生成过的数据）— **FAIL，误差 52.6%**
`anchor_n33_t1500.json`（同 n=33/T=1500，目标=GHIA_RE100 在 15 个 Ghia 站位插值后的
u/u_lid,v/u_lid 向量；插值在表内结点逐位精确，`tests/test_verify_lid_wall_diff_m17147.py` 已钉）：

| 量 | 值 |
|---|---|
| τ_true / Re_true | 0.596 / 100 |
| τ_rec | **0.70251** |
| Re_rec | **47.4** |
| Re 相对误差 | **52.6%**（>>2%，>>15%） |
| Ghia-MSE 在目标上的最小值位置 | τ 越大（Re 越小）越低，**本网格无 Re≈100 内点极小** |

Ghia-MSE 曲线（`ghia_mse_curve`，同 T）：
Re=126.3→3.06e-03，100→1.62e-03，82.8→1.17e-03，70.6→1.01e-03，61.5→9.58e-04（仍在下降；
Re≥171 的点因 n=33 欠分辨发散为 NaN，已如实记录）。
**诊断**：这是 solver-in-the-loop 的**模型误差 ≠ 参数误差**：n=33 的 LBM 在 Re=100 相对 Ghia
本身就有 ~4.5%（n=65 亦 4.53%，见实施方 §4.1）的离散误差，该偏差对 τ 的梯度**压过**了参数信号，
故外部数据把 τ 推向「Re 更小更好看」的方向。实施方自己的 §5 敏感性表（n=65：RMSE u 在
τ+0.05/Re=84.7 处最小）**已预示同一偏差**，只是未被写成反演结论。

## 4 逐项判定
| 判据（原文） | 判定 |
|---|---|
| ② 梯度 FD 交叉 <1e-4 | **PASS**（h=1e-4：1.1e-05 / 5.5e-08；Richardson 1.7e-11） |
| ① Re=100 τ 反演恢复误差 <2% | **PASS（仅同链合成数据，3.6e-06）；FAIL（外部锚，Re 误差 52.6%）** |
| 与生产链逐位等价（实施方 §4.3 声明） | **PASS**：n=21/T=50、n=33/T=40 上 `max|Δf|=0.0`（独立实现 `_prod_rollout`） |

## 5 分级发现
* **F1（中/范围类，必修文档）**「τ 反演 <2%」未标注适用范围：证据全部为**同链自洽**（真值由
  被测链生成），对外部数据的可迁移性未验且实测失败（§3.2）。建议验收文本改为
  「同链自洽反演 <2%；外部锚（Ghia）迁移误差已实测 = 52.6%，属离散偏差支配，需降 Ma /
  加密网格 / 模型误差校正后重测」。
* **F2（低）** 判据证据工件不入仓（`artifacts/gap_pp2_lid_wall_diff/` 不存在）⇒ 外部无法复算，
  违反「零手抄、可溯源」的基线要求；本轮已把可复算 JSON 落在 `artifacts/verify_m17147/`。
* **F3（低/工具）** 无复数步与前向模式两个独立梯度引擎（环境限制，原因已定位：`torch.clamp`
  与 `torch.func.jvp` 兼容性），本轮以 FD h 阶梯 + Richardson + Taylor 残差替代，可信度足够
  但仍是残余风险。
* **F4（信息）** 判据② 跨度的正确读法：h≤1e-4 才算；h=1e-2 的 11.5% 是差分截断，不是实现误差。

## 6 未跑 / 未覆盖（明示，不算通过）
1. 实施方 n=65/n=129 稳态 Ghia RMSE 阶梯（4.53%/3.31%）与时序工件：未复算（工件缺失 + 算力窗口）；
2. 外部锚反演的分辨率/马赫数收敛性（n=33 单点结论不能外推到 n=65/129）——**这是该发现的
   关键待办**：若 n=129 上外部锚误差降到 <15%，则 F1 仅为分辨率限制而非方法缺陷；
3. 噪声鲁棒性（本单 shard3 已递交集群 cj1439，队列未到，无结果 → 记未跑）；
4. 3D（`lid_driven_cavity_3d`）与 AMR 方腔的可微壁面：模块只覆盖 D2Q9。

## 7 复现
```
PYTHONPATH=src python scripts/verify_lid_wall_diff_m17147.py fd \
  --n 32 --steps 600 --threads 8 --out artifacts/verify_m17147/fd_n32_t600.json
PYTHONPATH=src python scripts/verify_lid_wall_diff_m17147.py repro \
  --n 33 --steps 1500 --tau-init 0.60 --max-iter 8 --out artifacts/verify_m17147/repro_n33_t1500.json
PYTHONPATH=src python scripts/verify_lid_wall_diff_m17147.py anchor \
  --n 33 --steps 1500 --tau-init 0.60 --max-iter 8 --out artifacts/verify_m17147/anchor_n33_t1500.json
python -m pytest tests/test_verify_lid_wall_diff_m17147.py -q     # 6 passed
```
工件：`artifacts/verify_m17147/{fd_n32_t600,repro_n33_t1500,anchor_n33_t1500}.json`、
`figures/verify_summary.png`（三联图：FD 阶梯 / τ 反演两种解读 / Ghia 目标 vs Re）。

## 8 判定（disposition）
**approve with notes**：代码侧无缺陷（BC 闭式解与生产逐位等价、梯度可证、求根反演机制可复现）；
**要求**按 F1 修订验收文本的适用范围，并把「外部锚迁移误差」列为后续节点
（建议 slug `GAP-PP-2-lid-wall-diff-anchor-migration`，验收：n≥129、Ma≤0.05 下 Ghia 外部锚
反演 Re 误差 <15%；量化口径同上脚本 `anchor` 子命令）。
