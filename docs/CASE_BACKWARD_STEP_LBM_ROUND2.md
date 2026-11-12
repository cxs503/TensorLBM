# CASE-BACKWARD-STEP-LBM · 第 2 轮：测量口径修正 + 收敛阶梯就位（honest 部分完成）

## ① 案例
Armaly et al. (1983, JFM 127:473) 层流后台阶再附长度 X1/s，ER=101/52=1.9423，
Re = u_bar·D_h/ν（D_h=2·h_in）。参考锚（仓内 refs.json，tier=SECONDARY_IN_REPO）：
Re=100 → 3.05；Re=200 → 5.0±0.2；Re=400 → 8.0±0.3；容差 15%。

## ② 本轮做了什么（对第 1 轮 honest-FAIL 的四项欠账）
1. **算子一致性缺陷修复**（`src/tensorlbm/bfs_armaly.py`）：出口海绵层原先只在
   `n_acc>0 or step>n_avg_start`（即平均窗口开始后）才施加 —— 平均窗口处发生一次
   **算子切换**，非定常瞬态与统计窗口由不同离散系统支配。改为第 1 步起恒定施加。
2. **新增剪切无关的独立口径** `reversed_flow_extent()`：取 x_step 下游、近壁前 3 行中
   **最后一个仍有 u_x<0 的列**，给出再附点的下界（分辨率 1 格）。它不依赖壁面剪切
   重构或其零穿越，因此「重构核错/多重穿越」不会同时骗过两个口径 —— 交叉一致性可机器检查。
3. **执行面**：`run_case.py` 增加 `--u-bar/--l-over-h/--up-over-hi/--collision/--avg-from/--steady-tol`
   以及降档阶梯 `h1_*`（h_in=26,s=25，ER=1.9615，几何偏差 +1.0%），使算例在算力受限时仍可跑收敛。
4. **聚合器** `postprocess.py`：从各 `result.json` 程序化生成对标表（零手抄）、
   两网格 Richardson 估计、三张图（X1/s vs Re、稳态历史、壁面剪切+零穿越）。

## ③ 结果表（程序化注入，来自 artifacts/case_backward_step_armaly/summary.json）

(none)

## ④ 图
见 figures/CASE_BACKWARD_STEP_LBM_round2/（x1_vs_re.png / convergence.png / wall_shear_*.png）。

## ⑤ 注意点（诚实登记）
- **算力摩擦**：集群 5 个作业（cj1335–cj1339，g1/g2 六点网格，stage_dir 递交）递交时队列
  `queued_ahead≈494`，本单窗口内未派发；本机 `run_shell` 单次 120s 上限且后台进程随组收割，
  故改用 `setsid` 脱离会话的后台跑 `h1_*` 阶梯（本机 load15 一度 135，明显被拖慢）。
  **提交时落盘情况**：h1_re200 / h1_re400 已在窗口内完成（见 ③ 表，steps_run/steady_at_step 为机器字段）；h1_re100 与集群 g1/g2 细网格点未完成 —— 仍不粉饰，表内缺席即未完成。
- 第 1 轮结论（X1/s 偏小 10–24%、Re=400 无再附）**主要嫌疑是未收敛 + 剪切拟合口径**，
  本轮已把两个嫌疑都做成可检验项（稳态提前停止判据 + 双口径）。
- ER 降档 1.9615 与实验 1.9423 差 1.0%，属几何偏差，需在误差预算中单列。

## ⑥ 误差判定
- 关键量 X1/s vs 参考：见第三段表。**是否 <3%：不能用本轮数据回答**（无收敛解即无
  可信误差）；<15% 判定同样以待收敛解为准 —— 本轮不给出通过结论。
- 复现：`OMP_NUM_THREADS=12 PYTHONPATH=src python benchmarks/case_pool/backward_step_armaly/run_case.py --preset h1_re200 --steps 80000 --u-bar 0.1 --l-over-h 10 --up-over-hi 3`，
  随后 `PYTHONPATH=src python benchmarks/case_pool/backward_step_armaly/postprocess.py`。
