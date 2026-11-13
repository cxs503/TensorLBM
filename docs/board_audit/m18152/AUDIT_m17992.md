# 交差审计 m17992 — XVAL-LBM-MODEL-MATRIX-CI-PLUS-COVERAGE（TensorLBM）

**审计单** m18152（board-audit）｜**被审** m17992（succeeded，auto-approve）｜**日期** 2026-11-13
**基准裁判** `tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-MODEL-MATRIX-CI-PLUS-COVERAGE'` → **0 行**
**裁定** **L2**（verdict id=704，gate_state=pending 待 maturity_gate 机器复核）

---

## ① 审计对象与问题

m17992 对标 CFD-GO/TCLB 的 model-matrix-ci-plus-coverage 做法：为 TensorLBM 建模型矩阵
（lattice × collision × physics × boundary）+ CI 风格测试套件 + 覆盖度追踪 + 与 TCLB 对照。
判据 <15%（owner 精度标准）。本单=交差核验，非重做。

## ② 审计方法（独立复现路径）

1. 取证：读 m17992 工作树交付物（`xval/model_matrix_ci_plus_coverage/` + `docs/XVAL-LBM-MODEL-MATRIX-CI-PLUS-COVERAGE.md`）；
2. 基准裁判：查 `tensor_benchmark_cases`（该 anchor 0 行）；
3. 复跑 repro_cmd：`python xval/model_matrix_ci_plus_coverage/run_matrix.py --quick`（28.8s，rc=0），
   6 项数值与 result.json 逐项相等；
4. 代码走读：run_matrix.py 693 行，确认真调 tensorlbm 求解器（非桩）；
5. git 交付态核查：branch/commit/远端分支。

## ③ 验收三件套核验

| 项 | 结果 | 证据 |
|---|---|---|
| run_artifacts 存在 | ✅ | result.json（6 results, pass_rate=0.1667, all_pass=false）；fig1 81922B/fig2 198684B/fig3 52233B 真 PNG（file 校验 2250x750/1200x1200/1500x900） |
| benchmark ref vs got | ⚠️ 1/6 过 | 见 ④；mission 自判据 <15% |
| repro_cmd 可执行 | ✅ | 本审计复跑 28.8s，73.616/57.505/100/100/False/0.6435 逐位复现 |

## ④ ref vs got 逐项（判据 <15%）

| 用例 | ref | got | 误差 | <15% |
|---|---|---|---|---|
| TC01 Poiseuille 2D | u_max=0.05 | u_max_sim=**0.0** | 73.62% | ❌ |
| TC02 Couette 2D | u_wall=0.05 | u_wall_sim=**0.0** | 57.51% | ❌ |
| TC03 Lid cavity (Ghia1982) | u_min=-0.2058, v_max=0.1715 | u_min_sim=**0.0**, v_max_sim=**0.0** | 100%/100% | ❌ |
| TC04 Poiseuille 3D | u_center=0.05 | u_center_sim=**-1.73e-08** | 100.0% | ❌ |
| TC06 Laplace law (SC) | 存在压力跳变 | delta_p=-0.0062, has_jump=**false** | — | ❌ |
| TC09 Acoustic pulse | 能量比 0.1–1.0 | energy_ratio=0.6435, c=0.5774 | — | ✅ |

**是否 <3%：否**（5/6 FAIL，57–100%；唯一过的 TC09 是能量守恒量级检查，非精度基准）。
报告六段（①–⑥）齐全；honest FAIL 未粉饰。

## ⑤ 发现（按严重度）

- **F1 (blocking-for-durability)**：交付物全部 untracked（`?? docs/...md`、`?? xval/model_matrix_ci_plus_coverage/`），
  branch agent/m17992 无 commit、无远端分支；mission 系 TP_AUTO_APPROVE_DEV_ACCEPTANCE
  （server_quality_audit.gate_valid=false）。worktree 回收即丢失。
- **F2 (基准裁判缺位)**：tensor_benchmark_cases 该 anchor 0 行 → L4 唯一裁判缺位。
- **F3 (harness 缺陷，报告归因误导)**：TC01–TC04 sim 值恰为 0.0。run_matrix.py:116-118 零速度初始化
  且无压力梯度驱动 → 流永不发展；零流加多少步都为零，报告 ⑤.1「步数不足」归因不成立
  （TC03「边界条件实现问题」部分诚实）。失败源于测试 harness 缺陷，非求解器精度问题。
- **F4 (报告 ④ 表)**：got 列填误差百分比而非实测值（实测值在 result.json，可溯源）。
- **F5 (正面)**：honest FAIL 未粉饰；repro_cmd 可复现；3 张真图非占位；覆盖度计算逻辑真实。

## ⑥ 判级与裁定

- **L2 成立**：真代码 input→compute→output（run_matrix.py 真跑 tensorlbm 求解器、覆盖度计算真实、可复现）。
- **L3 不予**：独立 xval 脚本，无集成点、无消费者（仅消费 tensorlbm 模块）。
- **L4 不予**：benchmark 0 行（唯一裁判缺位）+ 5/6 物理 FAIL 57–100% + 失败源于 harness 缺陷。
- **L5 不予**。
- **裁定 L2**，已落表 `tensor_maturity_verdicts` id=704（verdict_by=board-audit，gate_state=pending）。
- **返修建议**（若 reopened）：① 交付物 commit 上分支；② 注册 benchmark case 行；
  ③ 修 harness（TC01 加压力梯度/正确初始化，TC03 边界实现）后重跑全步数；④ 报告 ④ 表 got 列填实测值。
