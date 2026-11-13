# 交差审计 m17895 — XVAL-LBM-EXAMPLE-PER-MAIN-ANTIPATTERN（TensorLBM）

**裁定 = L2**（L4 机器裁定 `CLAIMED_NOT_PROVEN` rows=0；L3 无生态内消费者）｜被审 commit `222585d6`（分支 agent/m17895）｜审计单 m18086

## ① 案例介绍

被审对象是 TensorLBM 的 example-per-main 反模式治理：上游 sorush-khajepor/listLBM 每个例子自带 `main`/格子/时间推进/后处理；TensorLBM `examples/` 放大到 161 个文件。交付物是
(1) 声明式算例运行时 `src/tensorlbm/example_harness.py`（CaseSpec→run_case→JSON run record），(2) AST 反模式审计器 `src/tensorlbm/audit/example_antipattern.py`，(3) 一条 thin example，(4) 8 臂 xval 与机器裁判。

物理算例（D2Q9/BGK/全 Guo(2002)/半程反弹壁/float64 CPU）：压力驱动平面槽道 ny=16/32/48, nx=8, τ=0.8(ν=0.1), g=1e-5；Taylor–Green 32×32 周期, U0=0.05, 800 步。闭式参照 `u_max=gH²/(8ν), H=ny−2` 与 TG 能量 `exp(−4νk²t)`。

## ② 怎么做的（审计方法）

1. 取证：`git archive 222585d6` 到隔离目录，逐项核 impl_files/entry/run_artifacts/tests/benchmark/repro_cmd。
2. 独立复跑：pytest 16 项 + xval 分支臂（a1_32,a2 = 63.2s；a3,a4,a5,a6 = 23.5s），并与交付工件逐值比对；channel ny=32 以 `keep_history=False` 重跑，`record_digest`/`field_digest` 与工件逐比特一致。
3. 基准裁判：查 `tensor_benchmark_cases`（本 anchor 0 行）与 `tensor_case_pool`（0 行），并调平台机器裁定 `GET /api/v1/tensor/bench/l4-verdict/17895`。
4. 集成面：查 `origin/main`（fc86f508）是否含交付文件、是否有主求解器/平台侧调用方。
5. 文档一致性：报告数字 vs run_artifacts 程序化导出逐行对表。

## ③ 计算结果（ref vs got，全部读自 run_artifacts）

槽道 u_max vs 闭式 `gH²/(8ν)`：

| ny | u_max(got) | u_max(ref) | rel err | profile L2 | mass drift | steps | field_digest[:16] |
|---|---|---|---|---|---|---|---|
| 16 | 0.00243100 | 0.00245000 | 0.7755% | 0.3633% | -2.86e-13 | 4400 | b3454dae855f83af |
| 32 | 0.01123098 | 0.01125000 | 0.1691% | 0.0793% | -7.51e-13 | 12000 | 8bebedb948b54d40 |
| 48 | 0.02640609 | 0.02645000 | 0.1660% | 0.1224% | -9.55e-13 | 15000 | 601ff4b4d7f2a897 |

Taylor–Green 衰减率：got `1.54386517e-02` vs ref `1.54212569e-02` = **0.1128%**。
独立 example-main 对照（A2, torch.roll 推式）：u_max 相对差 `6.124291821339015e-13`, 剖面 L2 差 `nan`。
反模式量值：独立 main 占比 `0.9752`，重复行占比 `0.3138`，重复驱动块 `232` 个；检测器判别 gap `0.6212`。

交付机器裁判 11 项（summary.json `all_pass=True`，本次审计复跑关键项逐值一致）：

| check | value | threshold | pass |
|---|---|---|---|
| C10_antipattern_present_in_repo | None | 0.2 | True |
| C11_stability_rule | None | None | True |
| C1_channel_umax_within_owner_band | 0.007755102269816609 | 0.15 | True |
| C2_channel_umax_within_3pct_target | 0.007755102269816609 | 0.03 | True |
| C3_profile_l2_target_at_finest | 0.0012242133131070254 | 0.03 | True |
| C4_resolution_monotone_umax_err | None | None | True |
| C5_mass_drift | 9.550138457825597e-13 | 1e-10 | True |
| C6_tg_decay_target | 0.001127977207684141 | 0.03 | True |
| C7_harness_matches_independent_control | 6.124291821339015e-13 | 1e-08 | True |
| C8_determinism | None | None | True |
| C9_detector_discriminates | 0.6212 | 0.2 | True |

## ④ 结果图

本单为交付物审计（文档/工件核验+复跑），不产生新的物理场；交差判据全部是数值表/指纹/机器裁定，无可归因于本审计的新场量，故不生成云图。被审交付的两张图（`docs/xval_example_per_main/figs/fig1_channel_profile_convergence.png`、`fig2_tg_decay_and_antipattern.png`）由被审方 `make_figs.py` 从 arm JSON 生成，本审计核对了其数据源（arm_a1_*/arm_a3/arm_a6）。

## ⑤ 注意点（发现，按严重度）

- **F1 阻塞(L4)**：`tensor_benchmark_cases` 与 `tensor_case_pool` 对该 anchor 均 0 行；平台 `l4-verdict/17895 = CLAIMED_NOT_PROVEN, rows=0`，failed_checks 含 `ref_got_error_present/within_tolerance`。交付自设 3%/15% 阈值（predeclared.json）未登记为平台裁判 → L4 不予。
- **F2 阻塞(L3)**：`origin/main` fc86f508 不含任何交付文件（分支未合并），生态内零消费者；分支内仅 `api.py` 再导出 + thin example 消费，无主求解器/平台调用方（依赖方向是 harness→solver），L3 不达。
- **F3 低(文档)**：`example_harness.py:31` docstring 写 `H = ny-1`，而 `wall_height` 实现、测试与报告均为 `ny-2`——易误导后续维护者。
- **F4 低(报告)**：REPORT §③ C11 写 `ρ∈[0.99942,1.00000]`，工件 `summary.json` 为 `[0.9999999999990441, 0.9999999999997147]`。该行不是程序化注入（与「零手抄」表述冲突），但 PASS 结论不变（质量漂移 9.55e-13 与 ρ≈1−1e-13 自洽）。
- **F5 提示(适用范围)**：C9 判别力 0.6212 vs 0.20 仅基于植入夹具；repo 级重复度指标（31.4% 行 / 97.5% 独立 main）没有外部参照，指标定义本身即交付物，属自证而非外部基准。

## ⑥ 误差判定（关键量 vs 基准，逐项）

| 关键量 | got | 基准 | 相对误差 | <3%？ |
|---|---|---|---|---|
| 槽道 u_max ny=16 | 0.00243100 | 0.00245000 | 0.7755% | 是 |
| 槽道 u_max ny=32 | 0.01123098 | 0.01125000 | 0.1691% | 是 |
| 槽道 u_max ny=48 | 0.02640609 | 0.02645000 | 0.1660% | 是 |
| 剖面 L2 ny=48 | — | 闭式抛物线 | 0.1224% | 是 |
| TG 能量衰减率 | 1.54386517e-02 | 1.54212569e-02 | 0.1128% | 是 |
| 与独立 example-main | 0.0112309779 | 0.0112500000（对照臂同值） | 6.124e-13 | 是（远严于 1e-8） |
| 质量守恒 | — | 0 | 9.55e-13 | 是 |

物理关键量全部 <3%（最差 0.7755% 在 ny=16，随分辨率下降）。**但 L4 仍不成立**：唯一合法裁判（tensor_benchmark_cases/平台 l4-verdict）为 0 行，误差表只能算交付自证；L3 亦因无生态调用方不成立。功能实质（真实现+入口+可复现工件+不变量测试）达 **L2**。

## 复现

```bash
cd <repo>; export PYTHONPATH=src
python xval/XVAL-LBM-EXAMPLE-PER-MAIN-ANTIPATTERN/run_xval.py --repo-root .     # 8 臂约 2.5 min
python -m pytest tests/test_example_harness.py tests/test_example_antipattern_audit.py -q
```
本审计复跑记录：`{"pytest": "PYTHONPATH=src pytest tests/test_example_harness.py tests/test_example_antipattern_audit.py -q -> 16 passed in 22.58s", "xval_a1_32_a2": "run_xval.py --arms a1_32,a2 -> 63.2s, C7 reproduced 6.124291821339015e-13", "xval_a3_a4_a5_a6": "run_xval.py --arms a3,a4,a5,a6 -> 23.5s, C6=0.001127977207684141 C9=0.6212 C10 pass", "digest": "channel ny=32 rerun keep_history=False -> record_digest 288a7c6c... == shipped; field_digest 8bebedb9... == shipped"}`
