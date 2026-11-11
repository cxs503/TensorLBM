# GAP-PP-4-opt-loop · τ/Re 恢复 demo（第 2 轮·调研结合）

论文牵引：JOUR-DIFFLBM 第 4 节反演实验。工件：`artifacts/opt_loop_tau_recovery.json`（schema `tensorlbm/opt-loop-tau-recovery/v1`，18 次独立反演）、`figures/opt_loop_tau_recovery.png`、模块 `src/tensorlbm/opt_loop_tau_recovery.py`、测试 `tests/test_opt_loop_tau_recovery.py`（9 passed）。

## ① 问题设定 + 基准出处

Taylor–Green 衰减（周期方盒，一个波长横跨 n 个格距，k=2π/n）：

    u_x = A u0 sin(kx)cos(ky) exp(−2νk²t),   u_y = −A u0 cos(kx)sin(ky) exp(−2νk²t)

* **基准出处（external anchor）**：上式是不可压 Navier–Stokes 的精确解（Taylor & Green 1937, Proc. R. Soc. A 158:499；
  衰减率 2νk² 的标准推导见 Pope, *Turbulent Flows* §9）。同一式也是 BGK-LBM 的衰减模态，ν=(τ−1/2)/3（格距单位），
  故衰减率可反演 τ，得 Re=u0·L/ν（L=n 格距）。
* **已知/未知**：几何 (n,k)、u0、观测时刻已知；未知 τ（→ν、Re）与初始幅值 A（nuisance）。
* **验收判据（判据先于动手，未中途改）**：(a) demo 端到端跑通 + 机读收敛 JSON；(b) TG 衰减 Re 恢复 <3%；(c) 3 种子可复现。

## ② 方法与实现（三条选型，各有对照证据）

1. **梯度＝全程可微 LBM**：`tensorlbm.autograd_path.rollout`（D3Q19 BGK＋周期 gather streaming），τ 以图连通 0 维张量进入碰撞步，
   梯度穿过全部时间步；本 demo 进一步让初始幅值 A 也在图中（`TaylorGreenCase.initial_field(amplitude)`→`equilibrium3d`）。
2. **选型 A（必需）：把 A 作为 nuisance 剖掉**。晶格采样使离散峰值仅连续峰值的 0.962（n=16），叠加平衡态启动的压缩性亏欠，
   仿真场 ≈ 解析场 × 近常数 0.9755；A 钉死时拟合只能靠放慢衰减吸收，产生系统偏差（控制臂见 ④：−3.42%，超门）；A 自由时降到 +0.18%。
3. **选型 B：优化变量取 log ν**（保证 τ>1/2，步后 box-clamp 到 [0.5005, 1.9]），Adam(lr=0.3，2 步热身) + **L-BFGS(strong Wolfe, 10 步) 收尾**。
   纯 Adam 十步仍 ~1% 偏差，L-BFGS 把 twin 臂压到机器精度（对照固化在 `test_lbfgs_polish_is_what_converges_the_loop`）。
4. **损失**：各观测时刻相对场残差等权 L=(1/N_t)Σ_t‖u_sim−u_obs‖²/‖u_obs‖²；观测窗跳过前 10%（声学启动瞬态）。
5. **梯度正确性自检**：`gradient_check()` 中心差分校验 dL/dlogν，解析观测臂相对差 2.06e-06（n=16）/ 8.86e-06（n=32）。

代码指针：`TaylorGreenCase.{analytic,initial_field,times}`、`simulate`、`loss_and_norms`、`recover`（含 L-BFGS 抛光与逐次线搜索记录）、
`run_job`/`assemble`（分片与合并）、`main --shard i n`。

## ③ 参数表（格距/无量纲单位）

| 量 | 值 | 说明 |
|---|---|---|
| u0 | 0.05 | 初始峰值速度（Ma≈0.087，压缩性误差 O(Ma²)≈0.8%） |
| τ_true | 0.55 | 真值，各臂共用 |
| ν_true | 0.0166666667 | (τ−0.5)/3 |
| n | 16 / 32 | 每波长格数（箱长 L=n 格距） |
| Re_true | 48.0000 (n=16) / 96.0000 (n=32) | u0·L/ν |
| 观测时刻 | [21, 78, 137, 195] (n=16) / [79, 311, 545, 778] (n=32) | 跨 1 个 e-folding，跳前 10% |
| 观测噪声 | 0 / 0.2%·u0 / 1.0%·u0 | 高斯，按种子可复现 |

## ④ 对照表（机器从 JSON 生成，零手抄）

| track | 数据源 | A | n | seeds | τ_hat（逐种子） | Re_hat（逐种子） | 相对误差 [%] | max\|err\| | 判定 |
|---|---|---|---|---|---|---|---|---|---|
| analytic16 | analytic | free | 16 | 0,1,2 | 0.550090/0.550090/0.550090 | 47.914/47.914/47.914 | +0.1796/+0.1796/+0.1796 | 0.1796% | PASS |
| analytic32 | analytic | free | 32 | 0,1,2 | 0.549950/0.549950/0.549950 | 96.096/96.096/96.096 | -0.0999/-0.0999/-0.0999 | 0.0999% | PASS |
| twin16 | twin | free | 16 | 0,1,2 | 0.550000/0.550000/0.550000 | 48.000/48.000/48.000 | -0.0000/+0.0000/+0.0000 | 0.0000% | PASS |
| analytic16_fixed | analytic | pinned | 16 | 0,1,2 | 0.548292/0.548292/0.548292 | 49.698/49.698/49.698 | -3.4163/-3.4163/-3.4163 | 3.4163% | FAIL（对照臂，故意） |
| twin16_noise0.2 | twin | free | 16 | 0,1,2 | 0.549950/0.550017/0.550005 | 48.048/47.984/47.995 | -0.1001/+0.0341/+0.0099 | 0.1001% | PASS |
| twin16_noise1.0 | twin | free | 16 | 0,1,2 | 0.549751/0.550085/0.550025 | 48.241/47.918/47.976 | -0.4988/+0.1705/+0.0500 | 0.4988% | PASS |

判据逐条：

* (a) 端到端 + 机读收敛 JSON：**PASS**——schema `tensorlbm/opt-loop-tau-recovery/v1`，每次反演逐迭代记录 loss / τ / ν / Re / 相对误差，
  含 L-BFGS 线搜索的每一次函数评估（phase=`adam`/`lbfgs_eval`/`lbfgs`）。
* (b) Re 恢复 <3%（受门 5 臂）：**PASS**，最差 0.4988%；解析基准臂 n=16 +0.1796%、n=32 +0.0999%、twin 臂 0.0000%。
* (c) 3 种子可复现：**PASS（附诚实说明）**——种子控制初值抖动与噪声抽样；无噪臂 L-BFGS 收敛到同一唯一最优故数值相同（spread=0），
  有噪臂给出真实离散：0.2% 噪声 spread=0.1342%，1.0% 噪声 spread=0.6692%。
* 控制臂（A 钉死）3.4163%：**超门且与种子无关**——这是选型 A 的反证，不是回归。
* 梯度校验：n=16 2.06e-06、n=32 8.86e-06（相对差）。
  twin 臂报告值 1.0 是因为该点恰为精确最优（FD≈0，比值无意义），如实标注、不作为证据。

## ⑤ 收敛性 / 参数敏感性

* **两分辨率（网格收敛）**：解析基准臂偏差 n=16 → n=32：+0.1796% → +0.0999%（约 1.8× 减小，方向符合二阶截断预期）；对应最优幅值 A_hat 1.0277 → 1.0057，趋近 1+晶格采样亏欠。
* **噪声敏感性（两档）**：0.2%·u0 → max 0.1001%；1.0%·u0 → max 0.4988%（近似线性放大，仍在门内）。
* **优化器对照**：纯 Adam 无 L-BFGS 在 twin 臂十步仍偏 ~1%；加 10 步 L-BFGS 收到机器精度（测试固化）。
* 收敛曲线：`figures/opt_loop_tau_recovery.png` 面板 B（18 次反演逐迭代误差，log 纵轴 + 3% 门线）；
  面板 A 为衰减曲线 vs 精确解（n=16，实测衰减率与解析 2νk² 一致）；面板 C 为各臂最差误差柱状图。

## ⑥ 失败模式 / 工件 / 未完成项

* 前轮 m15358 honest-FAIL（fail_stage/原因未留）教训：本轮先做参数侦察（τ 可识别性、幅值 nuisance 必要性、优化器对照）再定判据与轨道。
* 已知失败模式（诚实清单）：
  1. 幅值钉死 → 系统偏差 −3.42%（控制臂故意保留）；
  2. 观测窗含启动瞬态 → 偏差与分辨率同阶，故统一跳过前 10%；
  3. 无噪数据下 3 种子 spread=0 是真实性质（同一唯一最优），不是“只跑了一次”；
  4. **未完成**：知识库 DB 侧留痕。`oss_inventory` / `oss_solutions` / `negative_knowledge_ledger` 对当前 DB 账号      **permission denied**（`psql` 实测），方法选型因此落盘为仓内工件（本文件 + JSON `selection` 字段）而非 `oss_solutions` 行。
     因此任务书 ①（读集群 ~/oss-cache 源码横向）与 ③（oss_solutions 写入）**未完成**，仅完成 ②（方法谱系）+ ④（结合实现）。
* repro_cmd：
  * 整跑：`PYTHONPATH=$PWD/src python -m tensorlbm.opt_loop_tau_recovery --out artifacts/opt_loop_tau_recovery.json`（≈20 min/核）
  * 分片（HPC 用）：`--shard i 6` 六片并行后 `--merge artifacts/opt_loop_shard_*.json`
  * 图：`python scripts/make_opt_loop_figures.py`；测试：`python -m pytest tests/test_opt_loop_tau_recovery.py -q`
* 工件：`artifacts/opt_loop_tau_recovery.json`、`artifacts/opt_loop_shard_{0..5}.json`、`artifacts/opt_loop_fixed_track.json`、
  `artifacts/logs/shard_*.log`、`figures/opt_loop_tau_recovery.png`。

## ⑥′ 误差判定（显式回答是否 <3%）

**是。** 受门各臂最大相对误差 0.4988%（解析基准 n=32 臂 +0.0999%、twin 臂 0.0000%），距 3% 门有 6× 余量。唯一超门的是**故意保留的控制臂** +3.4163%（幅值钉死），作为方法选型的反证。
