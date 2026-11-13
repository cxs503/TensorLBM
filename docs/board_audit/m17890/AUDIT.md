# 交差核验单 m17890 — m17846 / XVAL-LBM-IB3D-DIFFUSE-INTERFACE-MDF

仓 TensorLBM · 被核对象 mission 17846（status=succeeded，deepseek-v4-flash）· 核验人任务 17890（board-audit）
判级结论：**L2（独立功能实现）** · 基准判定：**FAIL**（Cd 误差 +29.3% / −17.0% vs 预声明 <15% 门）· 交付入库：**未 commit**

## ① 取证方法论（三件套 + 基准裁判）

| 核验项 | 做法 | 结果 |
|---|---|---|
| run_artifacts 真存在 | 直接读 `/root/tp-agent/work/m17846/benchmarks/xval_ib3d_diffuse_interface_mdf/artifacts/*`（5 个 physics/operator JSON + 5 个 step 日志 + 4 张图） | ✅ 存在且内容自洽 |
| benchmark ref vs got 数值 | 程序化从 `xval_summary.json` 读数（零手抄）+ 独立复算 Schiller–Naumann | ✅ 数值与报告一致 |
| repro_cmd 可执行 | 在隔离副本 `/tmp/m17890rt`（`src/` 软链交付源码）实跑 pytest / operator / physics | ✅ 逐位一致（见③） |
| 基准裁判（唯一裁判） | `SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-IB3D-DIFFUSE-INTERFACE-MDF'` | ⚠️ **0 行**：交付单未登记机器可核基准 → 本单按 mission goal 预声明判据（<15%）+ Schiller–Naumann 参考值复算，并补登记 cases 行（status=failed） |
| 报告六段 | `tensor_reports id=16794`（published, 4 figures）+ 仓内 `docs/ibm3d_mdf/REPORT.md` | ✅ ①问题+ref出处 ②方法 ③参数表 ④对照表 ⑤收敛/敏感性 ⑥失败模式+工件 齐全，FAIL 如实写 |

## ② 独立复跑（我自己跑出来的数，不是引用交付）

| 复跑项 | 命令 | 结果 vs 交付 |
|---|---|---|
| 单测 | `python3 -m pytest tests/test_ibm3d_mdf.py -q` | **10 passed**（5.74 s），与交付声明一致 |
| 算子层 | `python3 run_xval.py --stage operator`（float64, 642 标记, 24×28×32） | 模板权重相对差 0 / 0 / 1.9e-16、索引命中 1.000、网格力 0 / 0 / 5.1e-16、Σds 3.06e-16 —— **与交付逐项相同** |
| 物理层 | `python3 run_xval.py --stage physics --tag auditor_n5 --impl mdf --n-iter 5 --kernel peskin4pt --max-steps 3000 --threads 8` | Cd_drive=**1.4098021406601322**（交付 1.4098021406601322）、Cd_marker=**0.90516944448701**（交付 0.90516944448701）、ref=1.0905399、mass_drift 7.58e-05 → **逐位一致**，err_drive +29.28% / err_marker −17.00% |

结论：交付数值**可复现**（float32 CPU 8 线程确定性），报告未夸大数值。

## ③ 基准判据逐项（ref vs got，容差 15%）

| 算例 | 口径 | got | ref | 误差 | <15%？ |
|---|---|---|---|---|---|
| MDF iter1 peskin4pt | Cd_drive / Cd_marker | 1.4246 / 0.9140 | 1.0906 | **+30.63% / −16.19%** | ❌ / ❌ |
| MDF iter5 peskin4pt | 同上 | 1.4098 / 0.9052 | 1.0905 | **+29.28% / −17.00%** | ❌ / ❌ |
| MDF iter8 peskin4pt | 同上 | 1.4092 / 0.9048 | 1.0905 | **+29.22% / −17.03%** | ❌ / ❌ |
| MDF iter5 cosine4pt | 同上 | 1.4266 / 0.9123 | 1.0906 | **+30.81% / −16.35%** | ❌ / ❌ |
| legacy 单次直接力（本仓基线） | — | 未完成（报告时在跑） | 1.0906 | — | ⚠️ 无对照 |

参考值复算：`Cd=24/Re·(1+0.15Re^0.687)`，Re=100 → **1.09174**；交付用实测 Re≈100.23 → 1.09054（差 0.11%，不影响判别）。
结论：**物理层判据不通过**（4 例 × 2 口径全越界），且两个独立口径互差 ~55%、把参考值夹在中间 → 未时间收敛。

## ④ 分级裁定 L2（附差距）

- **L1 数学/单元验证**：`tests/test_ibm3d_mdf.py` 10 项（parity、单位分解、Σds 面积、伴随性/动量守恒、n_iter=1 与显式闭式一致、滑移残差单调、零滑移零力、Cd 往返、越界不缠绕）→ 实跑 10 passed ✅
- **L2 独立功能实现**：`src/tensorlbm/ibm3d_mdf.py`（20 726 B，真实现：`build_stencil/interpolate/spread/surface_weights/multi_direct_forcing/drag_coefficient`，非 placeholder）+ 入口 `benchmarks/.../run_xval.py` + 运行工件 ✅ → **达 L2**
- **L3 系统集成**：`grep -rn "ibm3d_mdf\|multi_direct_forcing" src/ app/ TensorLBM/` 仅命中自身与 benchmarks；`run_xval.py:stage_physics` 属 benchmark 侧驱动脚本，且调用私有 `tensorlbm.ibm_sphere_surface_force._lbm_step`（只用 `solver3d` 的 equilibrium/stream/macroscopic 组件）→ **主求解器零调用方，不授 L3**
- **L4 基准验证**：算例完整、误差分析完整、可复现 ✅，但基准**未过 15% 门** → 不授 L4
- **L5**：不授

## ⑤ 反夸大的实证发现（adversarial 视角）

1. **交付未入库**：m17846 分支 `agent/m17846` 相对 base `be78a87b` **0 commit**，`git status` 全为 `??`（4 项 untracked）。平台推送的是分支 commit → 现状下 GitHub 上无此交付物。
2. **基准未登记**：`tensor_benchmark_cases` 该锚点 0 行（对照：同期 m17853 已登记 `XVAL-LBM-DIFFUSE-INTERFACE-WETTING-VIA-WALL-ORDER`）→ 交差时机器裁判无从复核。本单已补登记（status=failed）。
3. **Cd_marker 逐报告步在 0.0–3.4 间振荡**（step 日志：0.0089 … 3.4316），Re=100 球应为定常轴对称 → 均值 0.905 虽统计上可用，但瞬时量非物理，佐证"未收敛"而非"已收敛但口径不同"。
4. **死代码**：`run_xval.py` `drive_force = g * float(rho.sum() * 0.0) + g * float((rho * fluid).sum())` —— 首项恒 0（残留式），易被误读为"总驱动力"。
5. **未执行 OSS 原版 JAX**：以 `jaxshim/` NumPy 替身执行 vendored 源码（jaxlib 无 loongarch64 轮子，已如实记 friction）；sha256 逐文件复核 **9/9 一致**（我复算 `PROVENANCE.json`）→ 源码保真成立，但"跑 JAX 原版"仍是缺口。
6. 报告把 legacy 基线列为"未完成"，未用未跑完的数充数 → 诚实，符合证据纪律。

## ⑥ 结论与建议

- 判级 **L2**：真代码 + 单测 + 可复现工件（算子层与 OSS 逐位 parity 1e-16 级），但零集成、物理基准 FAIL、交付未 commit。
- 建议返修方向（写进 evidence.recommended_action）：① 先 commit 并登记 benchmark case；② 驱动掩膜改 `r > D/2 + 2` 或改进出口 BC 消除驱动-IBM 壳层重复注力；③ 加 IBM 力时域平滑并跑 ≥10⁴ 步判平台；④ 接主求解器公开 API 才可申 L3。

工件：本目录 `audit_numbers.json`（程序化读数）、`fig_m17890_cd_gate.png`、`auditor_operator_rerun.json`、`auditor_physics_rerun_n5.json`、`auditor_physics_rerun_n5.log`。