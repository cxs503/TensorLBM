# AUDIT-RUNTEST-6 · TensorLBM 抽样运行测试审计（真跑，不读 PR 描述）

锚点 `AUDIT-RUNTEST-6`（owner 2026-11-11 第三项令）· 仓库 `cxs503/TensorLBM` · 任务 m14496 · 本文件由 `scripts/audit_runtest_r6/build_report.py` 从原始工件程序化生成（零手抄：表中每个数字都来自 `runs/audit_runtest_r6/` 下的日志/JSON）。

## 0. 执行环境与勘误

- Python 3.11.6 · torch 2.7.1 · numpy 2.4.6 · Arch loongarch64 · pytest 9.2.0
- 工作树：本次审计的工作树 `work/m14496`（本地）；集群 外协方双节点 loong64 已递交 4 个作业（`cj848/cj849/cj851/cj852`），队列 250+ 深，本单结束时仍未派发 → 全量重算**在本机执行**，单步均以 detached 方式跑并落日志（本机 shell 单次调用 120 s 上限，长算例用 `setsid` 脱离）。
- 诚实声明：集群侧 16 分片全量测试与 sphere D60 复算**未在预算内完成**；
  sphere Re=100 的 D40 复算在本机递交（见 §2），若日志未落定则按未完成记录。

## 1. ① 核心测试套件全量

命令（原始日志 `runs/audit_runtest_r6/pytest_full_local.log`，JUnit `runs/audit_runtest_r6/junit_local.xml`）：

```
PYTHONPATH=src python -m pytest tests -q -p no:cacheprovider --timeout=1800 -m 'not gpu' -n 28 --durations=30 --junitxml=runs/audit_runtest_r6/junit_local.xml
```

**部分完成（honest）**：JUnit 未落盘，套件在本单预算内未跑到 sessionfinish。

| 项 | 值 |
|---|---|
| 日志 | `artifacts/audit_runtest_6/pytest_full_local.log` |
| 最后进度行 | ...........................................F..FF........................ [ 43%] |
| -q 进度字符计数 | {"F": 284, "E": 19, "s": 92, "x": 26, ".": 5593} |
| 说明 | JUnit absent: the full sweep did not reach sessionfinish inside the budget (host contention, load average ~500-700 on 128 cores). Counts are the raw -q progress characters, i.e. a LOWER BOUND on executed tests, and F marks are not yet resolvable to test ids. |

因此 §1 不给通过率：**只承认「已执行到日志所示百分比、已出现若干 F 标记」**，不给未经验证的通过率数字。

### 1c. 抽样运行（确定性系统抽样，唯一在本机跑完的一次完整运行）

抽样规则（先声明后执行）：`ls tests/test_*.py | sort | awk 'NR%20==1'`（42 个文件，覆盖 tests/ 全目录的系统抽样）+ 本次新增的 `tests/test_poiseuille_wallmodel_arms.py`；文件清单 `artifacts/audit_runtest_6/sample_files.txt`。

```
PYTHONPATH=src python -m pytest <42 sampled files> tests/test_poiseuille_wallmodel_arms.py -q --timeout=600 -n 8 --tb=line --junitxml=.../junit_sample.xml
```

| 项 | 数值 |
|---|---|
| 用例数 | 834 |
| 通过 | 822 |
| 失败 | 10 |
| 异常 | 0 |
| 跳过 | 2 |
| 通过率（不含 skip） | 98.80% |
| 测试体累计耗时 (s) | 901.4 |

失败清单（原样，全部）：

```
test_lubrication_conductance::test_closure_is_exact_no_op_above_cutoff  [failure] assert (0.060481228216868584 == 0.16666666666666666)
test_lubrication_conductance::test_closure_cubic_law_and_monotonicity  [failure] assert 1.0 == 0.7333333333333334 ± 7.3e-07
test_thermal_verification_p96w2n04::test_rb_nu_at_ra1e4  [failure] AssertionError: Ra=1e4: Nu=0.954 vs ref 2.700, err=0.6468
test_thermal_verification_p96w2n04::test_rb_energy_balance  [failure] AssertionError: Energy imbalance 0.0824 > 5%
test_thermal_verification_p96w2n04::test_dhc_nu_at_ra1e4  [failure] AssertionError: Ra=1e4: Nu=1.584 vs ref 2.243, err=0.2937
test_thermal_verification_p96w2n04::test_rb_dimensional_consistency  [failure] AssertionError: Nu spread 1.7847 > 2% (base=0.9519, dt2=2.6508)
test_wetting_n01::test_sweep_returns_25_cases  [failure] ValueError: no fit window produced a valid contact angle
test_thermal_verification_p96w2n04::test_dhc_dimensional_consistency  [failure] AssertionError: Nu spread 0.1951 > 2% (base=2.0026, dt2=2.3933)
test_gas_layer::test_design_point_closure_extrapolation_bounds  [failure] assert 0.0 > 0.0
test_wetting_n01::test_sweep_metadata  [failure] ValueError: no fit window produced a valid contact angle
```

## 2. ② 代表性 benchmark / 科学计算算例（真跑到数值结果）

| 算例 | 调用 | 关键量 vs 解析/参考 | 守恒性 | 耗时 |
|---|---|---|---|---|
| B13 Poiseuille 2D (H=60, Zou-He 压差驱动) | `benchmarks/verified/poiseuille_2d/run.py single 60` | u_max err 0.179% / L2 0.002 | mass drift -8.751e-06% | 925.2 s |
| Taylor-Green 2D (Re=100, 128²) | `benchmarks/verified/taylor_green_2d/run.py` | energy decay rate err -0.0348% (analytic exp(-2*nu*k^2*t)) | mass drift 5.221e-05 (rel) | 165 s |
| shear_wave_decay H=64 | `benchmarks/verified/shear_wave_decay/run.py` | γ_vel err 0.0511% (解析 νk²) | mass drift 2.009e-05 | 23 s |
| shear_wave_decay H=128 | `benchmarks/verified/shear_wave_decay/run.py` | γ_vel err 0.0112% (解析 νk²) | mass drift 9.215e-05 | 191 s |
| MMS 2D（仓内自述 HONEST FAIL：不稳定） | `benchmarks/verified/mms_convergence_2d/run.py --grids 16 32 48` | 3/3 网格 finite=False (全 NaN 爆掉) | 与仓内 HONEST FAIL 自述一致（见 §6 D8） | — |

原始日志：`runs/audit_runtest_r6/bench_taylor_green_2d.log`、`bench_shear_wave_decay.log`、`bench_poiseuille_3d_pipe.log`、`bench_mms_convergence_2d.log`、`poiseuille_H60.log`、`sphere_D40_local.log`。

## 3. ③ 负向测试（错误输入必须报错，而非静默返回占位值）

脚本 `scripts/audit_runtest_r6/negative_probe.py`（13 个探针 + 1 个正向对照 + 1 个枚举对照），原始结果 `runs/audit_runtest_r6/negative_probe.json`。判据先于执行：REJECTED 才算 PASS。

| 统计 | 数值 |
|---|---|
| 探针数 | 13 |
| 被拒（域内 ValueError/TypeError） | 5 |
| 静默返回有限值 | 4 |
| 静默返回 NaN/Inf | 2 |
| 低层异常（未做域校验） | 2 |
| 对照通过（证明探针能看见拒绝） | True |
| PASS / FAIL | 5 / 8 |

| id | 探针 | 期望 | 实测 | 判定 |
|---|---|---|---|---|
| N00 | POSITIVE CONTROL: tau<=0.5 through a validating module | REJECT | REJECTED: ValueError: tau must exceed 0.5 for stability, got 0.4 | PASS |
| N01 | core BGK collision, tau=0.4 (negative lattice viscosity) | REJECT | SILENT: shape=(9, 8, 12) finite=True max|f|=4.444e-01 | FAIL |
| N02 | core BGK collision, tau=0.0 (division by zero) | REJECT | SILENT_NAN: shape=(9, 8, 12) finite=False max|f|=inf | FAIL |
| N03 | equilibrium at negative density | REJECT | SILENT: finite=True sum=-96.0000 | FAIL |
| N04 | equilibrium with wrong (broadcastable) velocity shapes | REJECT | REJECTED: ValueError: rho, ux, and uy shapes must match: rho=(8, 12), ux=(8,), uy=(12,) | PASS |
| N05 | Zou-He outlet with zero prescribed density | REJECT | SILENT_NAN: finite=False max|f|=nan | FAIL |
| N06 | engine config with viscosity=0 (Re=inf) | REJECT | LOW_LEVEL: ZeroDivisionError: float division by zero | FAIL |
| N07 | engine config with resolution=0 | REJECT | SILENT: engine built with resolution=0 | FAIL |
| N08 | engine config with max_steps=-5 | REJECT | SILENT: engine built with max_steps=-5 | FAIL |
| N09 | unknown collision model name (control: enum must reject) | REJECT | REJECTED: ValueError: 'not_a_collision_model' is not a valid CollisionModel | PASS |
| N10 | negative Smagorinsky constant | REJECT | REJECTED: ValueError: C_s must be non-negative | PASS |
| N11 | benchmark CLI: Poiseuille channel height H=0 | REJECT | LOW_LEVEL: rc=1 wrote_result=False last_line='ZeroDivisionError: float division by zero' | FAIL |
| N12 | benchmark CLI: sphere diameter D=0 | REJECT | REJECTED: rc=1 wrote_result=False last_line='ValueError: nx must be positive, got 0' | PASS |

要点（审计发现，按严重度）：

1. **核心路径不校验物理非法输入**：`solver.collide_bgk(tau=0.4)`（负格子黏度）静默返回有限值，`tau=0.0` 静默返回 Inf；同一仓内 `boundary/partial_slip_wall.py:86`、`cumulant.py:1524`、`acoustics_porous_absorber.py:489` 等十余处对同一判据是抛错的 → **库边界不一致**（不是「能不能」，是「哪条路径」）。
2. **N05 Zou-He 出流给定 ρ=0 静默产出 NaN**（`boundaries.zou_he_outlet_pressure(f, 0.0)`），无任何诊断；调用方拿到 NaN 场后要自己一路追。
3. **`GeneralSimEngine` 配置零校验**（`general_sim.py` 全文 0 个 `raise ValueError`）：`viscosity=0` 到 `reynolds_number` 才 `ZeroDivisionError`；`resolution=0`、`max_steps=-5` 静默建引擎。
4. **算例驱动 CLI 亦不校验**：`poiseuille_2d/run.py single 0` → `ZeroDivisionError`（低层）；`sphere_re100/run.py --D 0` 由体素化层抛出 `ValueError: nx must be positive`（可接受但非驱动层给出）。

## 4. ④ 收敛性 / 守恒性检验

### 4.1 空间收敛阶（Poiseuille 2D，网格倍增 r=2）

| 代码版本 | 臂(wall model) | q/Δ | 解析参考 | L2 相对误差(粗→细) | 观测阶 p |
|---|---|---|---|---|---|
| 改动前（HEAD） | halfway | n/a | n/a | 0.005 / 0.001 | 1.917 |
| 改动前（HEAD） | effective | n/a | n/a | 0.005 / 0.001 | 1.917 |
| 改动前（HEAD） | bouzidi | n/a | n/a | 0.005 / 0.001 | 1.917 |
| 改动后 H=10/20 | halfway | 0.5 | true | 0.005 / 0.001 | 1.917 |
| 改动后 H=10/20 | subcell_halfway | 0.25 | true | 0.102 / 0.050 | 0.981 |
| 改动后 H=10/20 | effective | 0.25 | effective | 0.006 / 0.001 | 1.922 |
| 改动后 H=10/20 | bouzidi | 0.25 | true | 0.009 / 0.002 | 2.057 |
| 改动后 H=20/40 @40000 步 | halfway | 0.5 | true | 0.001 / 5.343e-04 | 1.370 |
| 改动后 H=20/40 @40000 步 | subcell_halfway | 0.25 | true | 0.050 / 0.025 | 1.003 |
| 改动后 H=20/40 @40000 步 | effective | 0.25 | effective | 0.001 / 6.119e-04 | 1.227 |
| 改动后 H=20/40 @40000 步 | bouzidi | 0.25 | true | 0.002 / 3.308e-04 | 2.612 |
| 改动后 H=20/40 @8000 步（未达稳态对照） | halfway | 0.5 | true | 0.001 / 0.006 | -2.018 |
| 改动后 H=20/40 @8000 步（未达稳态对照） | subcell_halfway | 0.25 | true | 0.050 / 0.020 | 1.281 |
| 改动后 H=20/40 @8000 步（未达稳态对照） | effective | 0.25 | effective | 0.001 / 0.006 | -1.929 |
| 改动后 H=20/40 @8000 步（未达稳态对照） | bouzidi | 0.25 | true | 0.002 / 0.004 | -0.965 |
| 改动后 H=10/20 @20000 步 | halfway | 0.5 | true | 0.005 / 0.001 | 1.917 |
| 改动后 H=10/20 @20000 步 | subcell_halfway | 0.25 | true | 0.102 / 0.050 | 0.981 |
| 改动后 H=10/20 @20000 步 | effective | 0.25 | effective | 0.006 / 0.001 | 1.922 |
| 改动后 H=10/20 @20000 步 | bouzidi | 0.25 | true | 0.009 / 0.002 | 2.057 |

结论：

- **改动前三个臂逐位相同**（三种「方法」其实是同一段代码跑三遍）→ 见 §5 缺陷 D1。
- **台阶与网格对有关（新发现 D11）**：H=10/20 上 halfway p=1.917、bouzidi p=2.057；但 H=20/40（40 000 步，稳态标志全 True）上 halfway 落到 p=1.370、bouzidi p=2.612 —— 即该臂在测试段内**没有干净的渐近阶**，原「二阶」结论只对 (10,20) 这一对成立。
- 改动后物理自洽：`halfway`(q=0.5δ) p≈2；`subcell_halfway`（真实壁在 0.25δ 但边界仍按 0.5δ）p≈1（壁位误差一阶）；`effective`（同一流，改按边界实际执行的等效壁打分）p≈2；`bouzidi`（0.25δ 插值反弹）p≈2 且误差比 naive 臂小一个量级。

### 4.2 守恒性（质量）

| 算例 | 质量漂移 | 对照 |
|---|---|---|
| Taylor-Green 2D (4900 步, 周期域) | 5.221e-05 (rel) | 解析期望 0 |
| shear_wave_decay H=64 | 2.009e-05 (rel) | 解析期望 0 |
| shear_wave_decay H=128 | 9.215e-05 (rel) | 解析期望 0 |
| Poiseuille 2D H=60 (Zou-He 压差边界) | -8.750559270119352e-06 % | ρ 斜率 vs 名义 0.99999 |

### 4.3 图（matplotlib 生成，`scripts/audit_runtest_r6/make_figures.py`）

- `figures/audit_runtest_r6/conv_ladder.png` — 四臂 L2 相对误差 vs H（双对数），叠加 p=1/p=2 参考斜率和 HEAD（三臂相同）的星标点。
- `figures/audit_runtest_r6/bench_gate.png` — 所有实测算例的 |误差|% 与 3% 验收门对照。

## 5. 「文档声称 vs 实测」逐条对照表（差异 = 审计发现）

| # | 对象 | 文档/工件声称 | 实测 | 判定 | 说明 |
|---|---|---|---|---|---|
| 1 | taylor_green_2d | -0.03472820018259745 | -0.03477907489089382 | REPRODUCED | energy-decay rate error vs analytic -0.0348% (claim -0.0347%) |
| 2 | taylor_green_2d (mass conservation) | 5.2332871874228494e-05 | 5.221366259888856e-05 | REPRODUCED | relative mass drift 5.221e-05 over 4900 steps |
| 3 | shear_wave_decay H=64 | 0.05108901588489956 | 0.05108901588488831 | REPRODUCED | viscous decay rate vs analytic nu*k^2, error 0.0511% (claim 0.0511%) |
| 4 | shear_wave_decay H=128 | 0.011249094731377432 | 0.011249094731444925 | REPRODUCED | viscous decay rate vs analytic nu*k^2, error 0.0112% (claim 0.0112%) |
| 5 | poiseuille_convergence_order (arm distinctness) | 3 declared arms -> 4 distinct error curves | 3 declared arms -> 1 distinct error curves | DEFECT | identical l2_rel to all digits: the 'effective' arm falls through to apply_halfway_bb and the 'bouzidi' arm at q=0.5 is algebraically half-way BB, so the 3-arm  |
| 6 | poiseuille_convergence_order (own acceptance criterion (a)) | 1.0 | 1.9172719496686719 | CLAIM CONTRADICTED BY ARTIFACT | artifact and this audit's rerun both give p=1.917; the in-repo README says p∈[1.9,2.1] PASS -- the docstring criterion and the README disagree with each other,  |
| 7 | poiseuille_convergence_order (Bouzidi arm) | not tested | arm present with numeric results | ARTIFACT MISLABELLED | result.json carries a 'bouzidi' arm whose numbers are bit-identical to the halfway arm; README admits the interpolation was never exercised |
| 8 | mms_convergence_2d | documented unstable | {"case": "mms_convergence_2d", "lattice": "D2Q9", "collision": "bgk",  | REPRODUCED (negative claim holds) | rerun with --grids 16 32 48 reproduces the documented behaviour |
| 9 | test suite | full suite runnable | see log / summary below | SEE MEASURED SUMMARY | executed here; counts are in pytest_summary.json |

## 6. 审计发现清单（含定位）

| id | 严重度 | 位置 | 发现 | 证据 |
|---|---|---|---|---|
| D1 | 高（虚假验证） | `benchmarks/verified/poiseuille_convergence_order/run.py:168-181` | 三个壁面臂 `halfway`/`effective`/`bouzidi` 全部 `q_wall=0.5`，`effective` 落到同一个 `apply_halfway_bb`，`bouzidi` 在 q=0.5 与半程反弹代数等价 → 三臂结果逐位相同 | 本次重跑 `conv_repro.json`（未改动代码，H=10/20 三臂 l2_rel 完全相同）；改动后 `conv_fix_H10_20.json` 四臂互异 |
| D2 | 高（文档自相矛盾） | 同上 `run.py` 模块 docstring vs `README.md` | docstring 预登记判据 (a)「halfway ≈ 一阶」与 README「p∈[1.9,2.1] PASS」互斥，实测 p=1.917 | `result.json` + 本次重跑 |
| D3 | 中（工件误导） | 同上 `result.json` 的 `bouzidi` 臂 | README 自述「Bouzidi 未测」，result.json 却带数值 bouzidi 臂（= halfway 的副本） | `README.md` Limitations + `conv_repro.json` |
| D4 | 中（宿主依赖） | `benchmarks/verified/poiseuille_3d_pipe/run.py:627,640` | `--device` 默认 `cuda:2`，CPU-only 宿主直接 AssertionError；STATUS.md 却记为 verified「analytic」 | `runs/audit_runtest_r6/bench_poiseuille_3d_pipe.log`（首跑 cuda 失败）+ CPU 复跑 |
| D5 | 中（缺校验） | `src/tensorlbm/general_sim.py`（全文 0 个 `raise ValueError`） | 引擎配置 viscosity=0/resolution=0/max_steps<0 全部静默接受 | `negative_probe.json` N06/N07/N08 |
| D6 | 中（静默 NaN） | `src/tensorlbm/boundaries.py: zou_he_outlet_pressure` | ρ_out=0 → NaN 场，无诊断 | `negative_probe.json` N05 |
| D7 | 中（库边界不一致） | `src/tensorlbm/solver.py:67 collide_bgk` | 核心碰撞不校验 τ>0.5，同仓十余个模块校验 → 同一非法输入「看路径决定是否报错」 | `negative_probe.json` N00/N01/N02 |
| D8 | 低 | `benchmarks/verified/mms_convergence_2d/` | 仓内自述 HONEST FAIL（不稳定），本次复跑一致；无 result.json，仅 STATUS.md | `runs/audit_runtest_r6/mms_conv.json` |
| D10 | 中 | `benchmarks/verified/poiseuille_convergence_order/run.py`（改动前无稳态判据） | 步数固定而瞬态时间 ~H²/ν：H=20/40 在 8000 步下观测阶变负（-2.02/-1.93/-0.97），即默认 `--steps 20000` 的 3 档梯级本身会给出垃圾阶 | `conv_fix_H20_40_steps8000_underconverged.json` vs `conv_fix_H20_40.json`(40000 步) |
| D11 | 中（结论适用范围） | 同上 | 观测阶随网格对变化：halfway 在 (10,20) 上 p=1.917、在 (20,40) 上 p=1.370；bouzidi 2.057→2.612 ⇒ 「二阶」只在 (10,20) 这一对成立 | §4.1 表 |
| D12 | 中（真实测试失败） | 抽样套件 10 条失败 | 其中 5 条是物理判据失败：Rayleigh-Bénard Nu 对参考偏差 64.7%/29.4%、能量平衡 8.24%>5%、Nu 维度一致性 spread 178%/19.5%；另有 wetting N01 接触角拟合窗口无效 2 条、gas-layer 闭合 2 条、设计点外推 1 条 | `artifacts/audit_runtest_6/pytest_sample.log` |
| D9 | 低 | 同名脚本 `benchmarks/verified/poiseuille_2d/run.py` | CLI 对 H=0 抛 `ZeroDivisionError`（低层异常，非域校验） | `negative_probe.json` N11 |

## 7. 已执行的修复（D1/D2/D3）

- `benchmarks/verified/poiseuille_convergence_order/run.py`：新增 `ARMS` 表（`q`/`kernel`/`reference` 三元组），几何按 `H_true = ny - 3 + 2q` 反解 `ny`，保证四臂在同一真实高度上比较；每个 case 同时记录 `l2_rel_true_wall` 与 `l2_rel_effective_wall`，使「半程反弹对等效壁二阶、对真实壁一阶」这一教科书结论可在同一次运行里审计；收敛比改用真实高度。默认臂表扩为 `halfway / subcell_halfway / effective / bouzidi`。
- 模块 docstring 的预登记判据按物理更正（q=0.5 → p≈2；亚格子壁被忽略 → p≈1；等效壁参考 → p≈2；Bouzidi 插值 → p≥1.8），并显式记录本次审计发现的旧行为。
- 回归测试 `tests/test_poiseuille_wallmodel_arms.py`（5 条断言：四臂互异 + 各自物理阶）。**改动前 5 errors / 改动后 5 passed**（`runs/audit_runtest_r6/regression_baseline_HEAD.log` vs `regression_fixed.log`）。

## 8. 证据分级与未完成项（honest）

- 本次审计的执行面证据：测试套件（全量两次未跑完 + 确定性抽样 834 例跑完，见 §1/§1c）、4 个算例数值结果、13 个负向探针、4 臂×3 组网格的收敛梯级 + 质量守恒。所有原始日志/JSON 已归档在 `artifacts/audit_runtest_6/`（运行期写在 `runs/audit_runtest_r6/`，`runs/` 被 gitignore，故同时归档），本报告数字全部由脚本从这些工件注入。
- **未完成**：①集群 16 分片全量重算 `cj848` 与本机并行跑了同一套件，集群侧在预算内未派发（队列 250+）；②sphere Re=100 D40/D60 未复算（D40 递交后被让出 CPU，日志 `sphere_D40_local.log`）、`poiseuille_3d_pipe` R=20 的 CPU 复跑（`--device cpu --compile-mode eager`）在争抢下 80 min 未收敛到稳态，只留下 D4 的 CUDA 缺省缺陷证据；③`benchmarks/STATUS.md` 的 38 个 verified 算例未逐一复跑（本单按令「抽样」）。
- 本报告不复述 PR 描述，只报实测；跑不通的按原样记录（见 §1 失败清单、§3 FAIL 探针）。

