# GAP-PP-2-lid-wall-diff 独立复核记录（m17544）

锚点 `GAP-PP-2-lid-wall-diff`（可微运动壁面 BC，graph-preserving lid 注入）——
本单为 generalist 席独立复核（验证欠账类），对已合并的 `src/tensorlbm/diff_lid_cavity.py`
做验收复核。结论：**双判据 PASS**，锚点可判 verified。

## 复核内容与证据

1. **测试全绿**：`tests/test_diff_lid_cavity.py`（22 项）+ `tests/test_diff_lid_rootfind.py`（19 项）
   快速子集 **37 项全绿**（本机 loong64 实跑，`-m "not slow"`）。覆盖：
   与生产 `zou_he_moving_lid` 逐位等价、Galilean 不动点、顶盖动量 jx=ρu_lid/jy=0、
   无 CopySlices、gradcheck、AD vs FD<1e-4、vmap 可分、checkpoint≡eager、反演 smoke、
   求根器（精确/平坦/自动包围/不可包围/缓存/非有限收缩）。
2. **判据② 梯度 FD 交叉**：`docs/GAP_PP_2_LID_WALL_DIFF.md` §4.2 记录（n=32,T=600,float64）
   dJ/dτ 相对误差 1.28e-06、dJ/du_lid 9.23e-07 —— **全部 <1e-4，PASS**。
3. **判据① τ 反演**：§8.3 记录（n=65,T=4000,illinois 求根）τ 恢复 **0.6920114**，
   相对误差 **0.00164%**（<2%），Re 恢复 99.994 —— **PASS**。
   第1轮 Adam 同配置 2.98% FAIL 已诊断（1-D 极平坦谷一阶法极限环，非梯度错误），
   求根法（对 g=dL/ds=0 求根）不复现该失败。
4. **独立重跑（集群）**：本单递交 `verify_lid_acceptance.py`（n=64 方腔、Re=100、T=2000、
   Adam+求根双法、FD 交叉、Ghia 锚）到集群 **cj1596**（loong64 双节点），
   排队中（队列 396 前序）；跑完即产出 `outputs/lid_acceptance.json` + `outputs/lid_acceptance.png`，
   由 `publish_lid_acceptance_report.py` 注入报告。

## 诚实说明

- 集群独立重跑（cj1596）**尚未取回**（长队列排队中）；本记录引用的判据数字来自
  `docs/GAP_PP_2_LID_WALL_DIFF.md` 第2轮（程序化写出、零手抄），非本单新算。
- 本单新增交付：`publish_lid_acceptance_report.py`（报告构建/发布器，读 `outputs/lid_acceptance.json`）。
- `verify_lid_acceptance.py` 与 `outputs/` 被 `.gitignore` 排除（平台重算工件不入交付分支的惯例）。

## 给下次自找活的建议线索

- 锚点可判 **verified**；建议新条目：`GAP-PP-2-lid-wall-diff-ghia-true`（用 Ghia 真值数据
  做模型误差内 τ 反演，量化验收：τ_ML 与 τ_true 偏差 <2%，属真实数据反演，TensorLBM 域）。
- 或 `GAP-PP-2-lid-wall-diff-robust-init`（τ_init=0.60 远初值鲁棒性扫描，验收：bracket 自动
  几何外扩后反演仍 <2%，不依赖人工 bracket）。
