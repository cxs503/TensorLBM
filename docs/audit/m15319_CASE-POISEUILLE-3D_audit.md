# 交差审计 · m15263 CASE-POISEUILLE-3D（3D 方管泊肃叶 · TensorLBM）

审计单：m15319（board-audit，非重做）· 被审单：m15263（status=succeeded）
交付 commit：`c0c682e7` "m15263: matured mission deliverables"（4 files, +613）
审计日期：2026-11-11 · 独立复跑：本工作树 `artifacts/m15319/repro_run/`

---

## 0. 结论（disposition: **request changes** · 级别 **L2**）

| 项 | 判定 |
|---|---|
| 三件套 | run_artifacts ✅存在（但未入 commit）/ benchmark ref-vs-got ✅数值可核 / repro_cmd ✅**实跑逐位复现** |
| 基准裁判 `tensor_benchmark_cases` | **ABSENT**：`anchor_slug='CASE-POISEUILLE-3D'` 命中 **0 行** → 无机器裁判（全表 16 行，无该 anchor） |
| 自报误差 | f·Re 30.5043 vs 56.908 = **46.40 %**，脚本自带门 `<5 %` → **self-FAIL（报告⑥如实写「未通过」，无粉饰）** |
| 审计重算真误差 | 按仓内既有口径 f·Re = 2gD_h²/(ν·u_mean) → **122.017 vs 56.908 = +114.41 %** |
| 级别 | **L2**（真代码、输入→计算→输出、确定性可复现）；**不达 L3/L4** |

一句话：**交付的 46.4 % 是「量纲定义错 4 倍」与「求解器少算一半」两个反向错误相抵后的假象；真实偏差 114 %。该算例不构成 benchmark 验证（L4），只构成一个真实可运行的独立实现（L2）。**

---

## 1. 取证清单（三件套逐项）

| 三件套 | 证据 | 判定 |
|---|---|---|
| run_artifacts | `/root/tp-agent/work/m15263/runs/square_duct_poiseuille/{results.json(709B),u_field.npy(4.7KB),uy_profile.npy,uz_profile.npy,velocity_profiles.png(133KB),repro_cmd.txt}` | 存在非空；但 `git ls-files runs/square_duct_poiseuille` = **0**（未入库，commit 只含 4 个源文件）→ 工件不可随分支追溯 |
| benchmark ref vs got | results.json：ref f·Re=56.908（Shah & London 1978, Table 49, p.103）/ got f·Re=30.5043，error 46.397 %，passed=false | 数值自洽可核；**口径与引用文献不一致**（见 §3 B1） |
| repro_cmd | `python bench_square_duct_poiseuille.py --resolution 24 --tau 1.0 --steps 10000 --force 1e-05` | **PASS**：审计实跑（75.3 s）得到 u_mean=4.759978e-04、f·Re=30.5043，与交付 results.json **逐位一致** → 确定性可复现 |
| 报告六段 | `tensor_reports id=14382`：①案例介绍 ②方法 ③计算结果 ④结果图 ⑤注意点 ⑥误差判定 全部存在，figures=1 | **PASS**（六段齐；⑥明写 46.4 %>5 % 未通过，诚实） |
| 基准裁判 | `SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='CASE-POISEUILLE-3D'` → **0 rows** | **ABSENT** → 无机器裁判，L4 机器证书不可能取得 |

## 2. 审计复跑与独立参考量（数字全部程序化生成）

* `artifacts/m15319/repro_run/`：审计侧独立复跑（exit=0），与交付逐位一致。
* `artifacts/m15319/poisson_independent.py` / `fd_convergence.py`：独立有限差分解 Poisson ∇²u=−g/μ（不用级数、不用 LBM），R=gH²/(νu_mean) 随网格 24.15→26.00→27.14→27.77（n=10/20/40/80），向 28.455 收敛但受网格限制（仅作量级佐证，**不作为裁判**）。
* 权威参考 = Shah & London 56.908，且仓内已存在同口径常量：`src/tensorlbm/duct_poiseuille_validation.py:78-79 LITERATURE_POISEUILLE_NUMBER=56.91 / LITERATURE_RESISTANCE=28.455`（R=gL²/(ν·u_mean)，f·Re=2R）。

## 3. 缺陷清单（按严重度）

### B1【阻塞·高】metric 定义比引用文献口径小 4 倍 → 该 benchmark **天然不可能通过**
`solver.py:236` `compute_fRe = force*D_h**2/(2.0*nu*u_mean)`，而 Shah & London 的 f=(−dp/dx)D_h/(½ρu²) 给出 f·Re = **2gD_h²/(νu)** = 2R。
证据（程序化）：把**精确解析** u_mean=1.0206e-3 喂进 m15263 的 metric → **14.2275 = 56.908/4**（误差 −75.00 %，`artifacts/m15319/audit_metrics.json:metric_on_exact_u`）。即求解器再完美也过不了门。仓内既有 `LITERATURE_RESISTANCE=28.455` 已是同口径反证。
（对照：圆管 Darcy f·Re=64，方管 56.91 —— 与「2R」口径一致；m15263 的写法对应 R/2。）

### B2【阻塞·高】`analytical.py` Boussinesq 级数前因子 16 应为 4 → 解析剖面快 4 倍，误差被 u_max 归一化掩盖
`analytical.py:104` `prefactor = 16.0*a**2/(mu*pi**3)*(-dpdx)`（y∈[−a/2,a/2] 半宽 a/2 的约定下正确系数为 **4**）。
证据：把该级数代入它声称求解的方程，∇²u/(−g/μ) = **4.0000 ± 0.016**（`series_laplacian_check.json`）→ 残差恰好 4 倍；除以 4 后 u_mean=1.0103e-3，与精确 1.0206e-3 相差 **−1.00 %**（`audit_metrics.json:series4_minus_exact_pct`）。
后果：报告④唯一的「LBM vs 解析」曲线按 u_max 归一化后形状一致，**幅值 4 倍错误不可见**；`square_duct_mean_velocity/max_velocity` 全部偏大 4 倍。

### B3【阻塞·高】求解器少注入一半动量（缺 Guo 半力速度修正）→ 真实 f·Re 误差 **+114 %**
`_collide_and_force` 用力项 `(1−ω/2)*3*w*F*(c_ix−u_x)`，但宏观速度 `u=Σcf/ρ` 未加 `F/(2ρ)` 修正；τ=1 → ω=1 → 有效驱动仅 **(1−ω/2)F = 0.5F**。
证据（两 τ 双探针，`tau_scaling_probe.py`）：u_mean/u_exact = **0.4596**（τ=1.0，H=16）、**0.1473**（τ=0.6，ω=1.667，预测 1−ω/2=0.1667）；跨 τ 比值 **0.624**，与「力项被 (1−ω/2) 缩放」预测 **0.60** 相差 4.0 %，与「力项正确、只有 ν 影响」预测 **0.20** 相差 212 % → 力项缩放假设成立，8–12 % 残差归网格离散。
基准算例（H=22, τ=1）：u_mean = 4.7600e-4 vs 精确 1.0206e-3 = **0.4664**；u_max = 1.0631e-3 vs 精确 2.1394e-3 = **0.5020**。
换成正确口径：f·Re = 2gD²/(νu) = **122.017** → 误差 **+114.41 %**（自报 −46.40 %）。报告③⑤把 46 % 归因于「3D 角区 bounce-back 离散效应／分辨率不足」——**归因错误**：把 H 从 16 加到 22（及我的 FD 收敛研究）显示幅值缺额与网格无关，而与 (1−ω/2) 无关地恒为 ~0.47。

### B4【中】无测试、无集成调用方
`grep -rln 'square_duct' tests/` = 0；包外调用方仅 `bench_square_duct_poiseuille.py` 自身（`integration_callers: none`）。`__init__.py` 未进入 `tensorlbm/__init__.py` 或主求解器路径 → 不达 L3。

### B5【中】交付工件未入库 + 与既有能力重复
commit `c0c682e7` 仅 4 个源文件，`runs/` 未跟踪 → 复现须重跑（本轮已重跑，逐位一致）。
仓内**已有**方管锚：`src/tensorlbm/duct_poiseuille_validation.py`（m14877，含级数自检「wall=0/四重对称/∇²u=−g/ν」与文献 56.91 对表）＋既有工件 `artifacts/sdaa_poiseuille_3d_duct.json`（Po_exact=56.908，Po_sim=51.465，−9.56 %）。m15263 在既有能力之外新增的是一个**口径错误**的复本 → 净能力增量≈0，且有反向误导风险。

### B6【中】无机器裁判
`tensor_benchmark_cases` 无 `CASE-POISEUILLE-3D` 行 → 该算例池条目无判据可裁（本审计按 owner 交差令以 Shah & London 56.908 + 仓内 28.455 口径替代裁判，并如实标注）。

## 4. 未做 / 未跑（诚实边界）
* **未修缺陷**：本单为「交差核验，非重做」，B1–B3 均未改码。修复配方见 §5。
* 模块无既有测试可跑（`tests/` 无该包用例）；未新增 pytest（属返修轮职责）。
* 独立 FD 佐证受网格限制（R 27.77 @ n=80，一阶收敛迹象），**不作裁判**，仅作量级证据。
* 未验证 GPU/SDAA 后端（交付为 numpy CPU 路径）。

## 5. 升 L4 路径（可执行修复配方）
1. `solver.py:236` → `return 2.0*force*D_h**2/(nu*u_mean)`（与仓内 `LITERATURE_RESISTANCE=28.455` 对齐），并在报告写明 f·Re=2R 定义。
2. `analytical.py:104` → `prefactor = 4.0*a**2/(mu*np.pi**3)*(-dpdx)`；把 `series_laplacian_check`（∇²u=−g/μ）与 wall=0/对称性做成 pytest 前置断言（照抄 m14877 自检思路）。
3. `_collide_and_force` 补 Guo 半力：`u* = Σcf + F/(2ρ)`（equilibrium 与力项都用 u*），力项改为 `3w[(c−u*)·F] + 9w(c·u*)(c·F) − 3w(u*·F)`；改后 u_mean/u_exact 应从 0.47 回到 1±0.03。
4. 入口补 `tests/test_square_duct_poiseuille.py`：级数自检 + f·Re∈56.908±5 % + u_max/u_mean≈2.096±5 %（同时给出 B3 修正前后对照 = S2 级证据）。
5. 为 anchor 登记 `tensor_benchmark_cases`（metric 写明 `fRe=2gD_h²/(νu_mean), D_h=ny-2, 限幅 <5 %`）+ 产出 ≥1 行 tensorbench_results，方可机器 L4。

## 6. 审计侧工件
* `artifacts/m15319/repro_run/`（独立复跑，逐位一致）
* `artifacts/m15319/tau_scaling_probe.py|json`（B3 双 τ 探针）
* `artifacts/m15319/poisson_independent.py|json`、`fd_convergence.py|json`（独立 FD 佐证）
* `artifacts/m15319/series_laplacian_check.py|json`（B2 泊松残差 = 4.0000）
* `artifacts/m15319/analytic_check.py|json`、`make_audit_figure.py`、`audit_metrics.json`、`audit_figure.png`
