# m23997 · PR#2653 重开第3轮 · 交付确认（2026-11-16）

## 背景

原 PR#2653（m21273，periodic-x + constant dP/dx 通道 R3 xval）已裁定关闭（owner 2026-11-05）。
本单（m23997）为「管到底第3轮」：核实原 PR 的 9 项交付物是否全部在 main，回归测试是否绿，
并补发论文级进展报告。

## 交付物核对（9/9 在 main）

| 文件 | 状态 | 入库路径 |
|---|---|---|
| `_scratch/probe_r3_ma.py` | ✅ 在 main | m23937 恢复（commit db2508e2c，PR #2679） |
| `docs/reports/XVAL_LBM_PERIODIC_X_CONSTANT_DPDX_R3.md` | ✅ 在 main | #2451 |
| `docs/reports/figures/xval_periodic_dpdx_r3_grid_order.png` | ✅ 在 main | #2451 |
| `docs/reports/figures/xval_periodic_dpdx_r3_mach_wall.png` | ✅ 在 main | #2451 |
| `docs/reports/figures/xval_periodic_dpdx_r3_wall_origin.png` | ✅ 在 main | #2451 |
| `src/tensorlbm/periodic_channel_dpdx.py` | ✅ 在 main | #2451 |
| `tests/test_xval_life_periodic_dpdx_r3.py` | ✅ 在 main | #2451 |
| `xval/xval_lbm_periodic_dpdx/make_r3_report.py` | ✅ 在 main | #2451 |
| `xval/xval_lbm_periodic_dpdx/out/report_payload_r3.json` | ✅ 在 main | #2451 |

## 回归测试

- **本机**：`tests/test_xval_life_periodic_dpdx{,_r3,_r4}.py` → **33 passed**（113 s）。
- **集群全套**（cj4842，n315，16 核，rc=0）：`pytest tests/ -q`（排除 13 个 main 上已知的
  预存 collection-error 文件，与本次改动无关）→ **succeeded**。

## 本单改动

**无新增代码改动。** 原 PR#2653 的全部 9 项交付物已在 main（#2451 + m23937 恢复 probe），
本单为交付确认 + 全套测试复核 + 论文级报告补发（report id 21090）。

## 报告

论文级进展报告已发布（`/tensor/reports`，id 21090），含六要素：
①案例介绍 ②方法 ③计算结果表 ④三张图（grid_order / wall_origin / mach_wall）
⑤注意点 ⑥误差判定（rel-L2 最差 0.661 % < 3 %，u_max 最差 0.498 % < 3 %）。
