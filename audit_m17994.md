# 交差核验 m17994 —— XVAL-LBM-MPI-1D-SLAB-UNPACKED-HALO-TESTALL-OVERLA（TensorLBM）

**裁定：L2（独立功能实现）· confirmed 路径 · verdict id 705（board-audit）· 详见 `docs/audit_m17994/AUDIT_REPORT.md`**

| 核验项 | 结果 | 指针 |
|---|---|---|
| 真代码（impl） | ✔ 617 行 D2Q9 y-slab，零拷贝 unpacked halo + Testall 轮询 + 4 传输替身 | `src/tensorlbm/parallel/slab_unpacked.py:386,449,553,69`（m17994 @ `9ae15ea2`，证据快照 `docs/audit_m17994/evidence/SNAPSHOT_slab_unpacked_m17994.py`） |
| 测试 | ✔ **10 passed / 8.65 s**，断言=不变量（逐位相等 `assert_array_equal`、闭式 8/6 字节比、剪切波 <1%） | `tests/test_parallel_slab_unpacked.py` |
| 工件 + 复现 | ✔ 审计独立复跑 9 臂，**6 个确定性 KPI 相对差 = 0**；全量 `~65 s` 落 `COMPLETE` | `docs/audit_m17994/evidence/audit_rerun_results.json`、`kpi_audit_rerun.csv`、figA |
| 精度 vs 基准 | ✔ 探针/场 L2 **0.013780%**（解析剪切波）≪ 15%，亦 ≪3%；N=64×75 步 0.055118%；t=600 步 0.014709% | `evidence/audit_step_amendment_check.json` |
| 分解中性 | ✔ NP=2/4 × 3 种 halo 约定 vs NP=1：`identity_maxrel = 0.0`（逐位，非 1e-12 以内） | `evidence/kpi.csv` |
| 通信记账 | ✔ wire 32768 / 24576 B/步 = **8/6 闭式**；staging 0 vs 12288 B/步 | fig2 |
| 平台基准判据行 | ✘ **无**：`tensor_benchmark_cases`/`tensor_xval_cases`/`repro_pool` 对该 anchor_slug 均 0 行 | psql 查询见报告 §1 |
| 系统集成 | ✘ 零调用方（`grep slab_unpacked\|tensorlbm.parallel` 仅命中 `parallel/__init__.py` 自身）→ L3 不成立 | 报告 §1 |
| OSS 对照侧 | ✘ 未执行（镜像无 mpi4py / mpirun）；ref=台账做法+自写替身，属语义移植 | 报告 F2 |

## 三条须记的发现

1. **F1（中）步数口径**：`predeclared.json` 写 600 步、`results.json` 实际 300 步、mission `checks` 反向写成 600 步。审计补跑 600 步误差 0.014709%（300 步 0.013780%）→ **物理结论不变**，属预声明纪律缺陷，须补 amendment。
2. **F3（中）墙钟结论不可复现**：同机复跑 MLUPS 最大差 **2.763×**；overlap speedup 符号稳定（<1）但量级 0.475→0.205；轮询扫描「64 行比 8 行慢」在复跑中**翻转**（17.53/13.45/16.25 s → 11.37/6.11/6.32 s）。轮询**次数**（29.8/3.4/1.0 次/步）可复现。→ 性能表须带噪声带/n>1。
3. **F4（低-中）API 隐患**：`global_scalars()` 第 7 项 `max|u|` 走 `allreduce_sum` → NP=4 报 0.258 而真值 0.064（报告⑥已披露，代码未防）。

**判级理由**：功能真实、数值可复现且精度远超 owner 标准（故不是 MOCK/STUB，也非造假）；但它是**未接入任何求解器的独立验证模块**，对标另一侧未执行、平台无判据行 —— 按「级别=功能实质」只能是 **L2**，与隐含的 L4 声称差 2 级。补级路径：把 `parallel` 包接进 `blockforest`/主求解器（F5），并为该锚登记判据行。

**maturity_gate 预演**：`judge(2, evidence)` → `(2, [])`（机器同意）；`judge(4, …)` → `(2, ['L3 缺: integration_callers(grep 无引用)', …])`。gate 在审计单终态后的周期自动裁定。