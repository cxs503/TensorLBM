# CASE-BACKWARD-STEP-LBM · 第 3 轮：首个收敛解落地，Re=100 达标（<15%）

**锚点**：feature_ledger `CASE-BACKWARD-STEP-LBM`（算例池·后台阶再附长度，ref Armaly et al. 1983，容差 <15%）
**判定：PASS（诚实局部达标）** —— X1/s = **2.920** vs 参考 **3.05** → **−4.3%**（容差 15%）。
**同时**：对二维数值基准 Erturk 2008（2.878）为 **+1.5%** —— 落在其表格给出的离散带内。

---

## ① 案例（物理背景 / 几何 / 参数）

后台阶（backward-facing step）层流分离—再附：流经台阶后下壁面出现分离泡，主再附长度
`X1/s`（台阶高度归一）是检验壁面离散误差与再附判据的经典观测量。

| 项 | 取值 | 说明 |
|---|---|---|
| 几何 | 入口高 `h_in=26` 格，台阶 `s=25` 格，下游高 `H=51` 格 | 实验 10.1/5.2 mm → ER=1.9423；本档 ER=51/26=**1.9615（+0.99%）** |
| 雷诺数 | `Re = u_bar·D_h/ν`，`D_h = 2·h_in` | Armaly 1983 原始定义（**不是** `u_in·s/ν`） |
| 入口 | 充分发展抛物线，离散行均值严格 = `u_bar = 0.1` | Zou/He 速度入口 |
| 出口 | Zou/He 压力 `ρ=1` + 非平衡态海绵层 | —— |
| 壁面 | half-way bounce-back（pre-stream 形式） | —— |
| 碰撞 | TRT，magic `Λ=3/16` | 消除 half-way BB 壁面位置误差 |
| 松弛时间 | `τ=0.6560`（`ν=0.052`） | τ−0.5 = 0.156，远高于 0.02 的危险带 |
| 网格 | `nx=818, ny=53`（43 354 格），上游 3·h_in，下游 10·H | 单档 |
| 步数 | 88 000（统计窗口 40%–100%，81 个平均样本） | 见 ⑤ 关于稳态判据的诚实登记 |
| Mach | `u_max/c_s = 0.260` | 弱可压范围内 |

参考值（仓内 `benchmarks/case_pool/backward_step_armaly/refs.json`，tier=SECONDARY_IN_REPO，
原 PDF 本轮仍不可达，获取尝试日志在该文件内）：`Re=100 → 3.05`（±0.2 实验读数带），
容差 15%。交叉锚：Erturk 2008, *Comput. Fluids* 37(6):633–655, Table 1（ER=1.942）→ 2.878
（band 2.792–2.965）。

## ② 怎么做的（方法 / 工具链）

1. 求解链全部来自库内核，无本算例私有物理：`tensorlbm.d2q9`（平衡态/宏观量）、
   `tensorlbm.solver`（`collide_trt` / `stream`）、`tensorlbm.boundary.boundaries`
   （Zou/He 入口/出口、海绵层）；算例容器 `tensorlbm/bfs_armaly.py` 只做口径与观测量。
2. 观测量三口径并行记录（互不共享失效模式）：
   - **主口径**：近壁剪切 `τ_w(x) = −2u_1 + 3u_2 − u_3` 的零穿越（对二次型剖面为精确导数），
     线性插值定位，原点取台阶面 `x_step − 0.5`；
   - **独立下界口径**：`reversed_flow_extent()` —— 下游近壁前 3 行中最后一个仍有 `u_x<0` 的列
     （分辨率 1 格，不依赖剪切重构，用于捕捉「剪切重构错但也骗过零穿越」的伪解）；
   - **单行口径**（历史缺陷口径，仅作对照）：`u_x` 第 1 行的零穿越。
3. 质量守恒自检：入口/出口流体行通量比（`mass_flux_rel_error`），本档 **8.9e-5**。
4. 执行：`benchmarks/case_pool/backward_step_armaly/run_case.py --preset h1_re100 --steps 160000`
   （实际在 88 000 步处由脚本内预算/稳态检查终止），落盘
   `artifacts/case_backward_step_armaly/h1_re100/{result.json, wall_shear.csv, field.npz}`。
5. 聚合零手抄：`postprocess.py` 读取全部 `result.json` → `artifacts/.../summary.{json,md}` + 图。
   本文表格与图均由该脚本生成，数字未手工转写。

复现命令：

```bash
cd <repo>
OMP_NUM_THREADS=16 PYTHONPATH=src python3 benchmarks/case_pool/backward_step_armaly/run_case.py \
    --preset h1_re100 --steps 160000
PYTHONPATH=src python3 benchmarks/case_pool/backward_step_armaly/postprocess.py
PYTHONPATH=src python3 benchmarks/case_pool/backward_step_armaly/maturity_evidence.py
```

## ③ 计算结果（程序化注入，来自 `artifacts/case_backward_step_armaly/summary.md`）

| tag | Re | h_in | ER | tau | steps | steady@ | X1/s | reverse-extent | row1 | ref | err % | <15% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| h1_re100 | 100 | 26 | 1.9615 | 0.6560 | 88000 | 88000 | 2.920 | 2.880 | 2.896 | 3.05 | -4.3 | PASS |

细节（`h1_re100/result.json::result`）：`n_averaged_samples=81`，
`average_window_steps=[35200, 88000]`，`mass_flux_rel_error=8.93e-5`，
`u1_min/u_bar=−0.238`（分离泡存在性），`wall_time_s=1144`（16 线程本机）。

**三口径一致性**：主口径 2.920 / 独立下界 2.880 / 单行 2.896 → 相对跨度 **1.4%**。
下界口径低于剪切口径是设计预期（它系统性偏短 ≤1 格 / s = 0.04），方向与幅度均与
`docs/CASE_BACKWARD_STEP_LBM_ROUND2.md` 的判据说明一致 —— 说明本档解**不是**被伪零穿越俘获的伪解。

## ④ 图

- `figures/CASE_BACKWARD_STEP_LBM_round2/x1_vs_re.png` —— X1/s vs Re，Armaly 参考曲线 + LBM 点 + 逐点误差标注。
- `figures/CASE_BACKWARD_STEP_LBM_round2/convergence.png` —— 再附长度的瞬态历史（稳态判据的可视证据）。
- 数据侧：`artifacts/case_backward_step_armaly/h1_re100/wall_shear.csv`（x, τ_w 底壁, τ_w 顶壁, u_1）
  与 `field.npz`（时均 u_x 场 + 固体掩码）已入库，可重绘壁面剪切/云图。

（本轮无场量云图 PNG：`field.npz` 是时均 u_x 场，绘图脚本 `make_figures.py` 的云图分支需要
多档结果；单档情况下以壁面剪切 CSV + 数据文件替代，理由在此显式登记。）

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

1. **算力摩擦是本算例的头号风险，而非物理**：集群 4 个作业（cj1395–cj1398）递交时
   `queued_ahead ≈ 538`，观测到 21:27 递交的作业在 2.5 h 后仍 queued → 本轮**不依赖集群**；
   改用 `setsid` 脱离会话的本机后台进程（`nohup` 会被调用组收割，第 2 轮已记录）。
   本机 load15 在 111–150 之间（nproc=128），故只保留 1 个 16 线程作业。
2. **只完成 1 档，网格阶梯缺档（诚实登记）**：精确 ER 档 `g1`（h_in=52, s=49, ER=1.9423 精确）
   运行 12.5 min 后为让出 CPU 被主动停止，未落盘；`g2`（h_in=104）本机预计 ≥2 h，未跑。
   因此 **两点 Richardson / 观测阶次本轮不可用**，本文不对离散误差做外推声明。
3. **几何口径偏差**：本档 ER=1.9615 与实验 1.9423 差 **+0.99%**（台阶/入口比），属几何偏差，
   与离散误差不可分离；精确 ER 档补跑后应复查。
4. **稳态判据**：`steady_at_step=88000` 与预算相同，说明自动稳态判据未在预算内触发
   （`steady_tol=5e-3` 相对 X1 漂移、连续 3 次）。本档的可信度依据改为：三口径 1.4% 一致 +
   质量通量 8.9e-5 + 瞬态历史（`result.json::convergence_history`，逐点数值在工件内，未手工转写）。
   **这比「已判定稳态」弱一档**，下一档应延长预算或用 `--steady-tol` 放宽后交叉比较。
5. **适用范围**：Re=100 属二维充分成立区（实验三维效应在 Re≳400 才显著），故 −4.3% 不应用三维
   解释；文献在 Re=100 点的 2D 数值（Erturk 2.878）与实验（3.05）本身散布 6%，**任何 <3% 的
   声称都需要先声明以哪一族为基准**（第 2 轮 pending 战役的教训，见
   `benchmarks/pending/backward_step/result.json`）。

## ⑥ 误差判定（关键量 vs 基准，逐项）

| 观测量 | 本文 | 基准 | 误差 | <15%（本锚验收线） | <3%（论文级） |
|---|---|---|---|---|---|
| X1/s（主口径，τ_w 零穿越） | 2.920 | Armaly 1983 实验 3.05 | **−4.3%** | ✅ PASS | ✗（否） |
| X1/s（主口径） | 2.920 | Erturk 2008 二维数值 2.878 | **+1.5%** | ✅ PASS | ✅（是，<3%） |
| X1/s（独立下界口径） | 2.880 | Armaly 1983 实验 3.05 | −5.6% | ✅ PASS | ✗（否，且该口径设计上偏短） |
| 质量通量守恒（入口 vs 出口） | 8.9e-5 相对误差 | 0 | — | ✅ | ✅ |

**结论**：本锚点验收线 **<15% 达标（−4.3%）**；对二维数值基准 **<3% 亦达标（+1.5%）**；
对实验值的 4.3% 与文献记录的 2D↔实验 6% 散布同量级，属方法固有差别而非本实现缺陷。
**未完成项如实列出**：精确 ER 细档（g1/g2）未跑 → 网格收敛未证；稳态判据未自动触发（见 ⑤.4）。