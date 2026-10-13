# w11/ — W11-A 判决机器档案（cylinder_re40_st，verdict FAIL）

本目录是 2026-10 Wave-11 判决工件的入库副本；判决叙述、门禁口径与全部
判据数的 PROVENANCE 溯源见上级 [README.md](../README.md)。本 README 的
全部表格由机器脚本生成（`w11_archive_gen/gen_archive.py`，判决数字零手抄；
核验明细 `w11_archive_gen/verify_manifest.json`，均在战役暂存内）。

## 判决文件（md5）

| 文件 | md5 |
|---|---|
| `formal/verdict.json` | `d4326c7de93a32f97444c5df3ddfb5a8` |

## 判决读数（机器读自判决 JSON）

| 项 | 值 | 备注 |
|---|---|---|
| re20 D40 | Cd 2.1212576639754612 | err +6.0576% |
| re20 D80 | Cd 2.1172753059375715 | err +5.8585% |
| re20 D160 | Cd 2.1176905168536173 | err +5.8792% |
| re20 gates | G1 False / G2strict False / G3 True | tier FAIL |
| re40 D40 | Cd 1.5852098888020207 | err +5.8218% |
| re40 D80 | Cd 1.5801115167974609 | err +5.4814% |
| re40 D160 | Cd 1.5793775112093078 | err +5.4324% |
| re40 gates | G1 False / G2strict True / G3 True | tier FAIL |
| overall | FAIL | Honest FAIL both Re tiers. G1 fails all six rungs: errors plateau at +5.43..+6.0… |

## 归档清单（55 文件，156419 B）

| 路径 | B | md5 |
|---|---|---|
| `diag/fd2_smoke.json` | 1028 | `e7ddc82c47b8d6d40161cb98f88ca1db` |
| `diag/fd3_smoke.json` | 687 | `1ea83e0960c13e543e0b3273fca24c9c` |
| `diag/fd4_analysis.json` | 5668 | `ce660ea1149467e79ac103037d85a3ea` |
| `diag/fd4_base0.json` | 1102 | `b671a6d6e6dc2aff7f09517215adfb03` |
| `diag/fd4_base1.json` | 1104 | `ac48d214ba2f95871dc14a2928b23aff` |
| `diag/fd4_base2.json` | 1104 | `e4384c57795f57281759983415ce8ee7` |
| `diag/fd4_base3.json` | 1103 | `81407f9336291efdc707f74884dcb487` |
| `diag/fd4_refine1.json` | 1100 | `10beb61fb475a7a25890021284586680` |
| `diag/fd4_refine2.json` | 1107 | `fef4c6b4e3b8761eec0c63089892cb0b` |
| `diag/fd4_refine3.json` | 1102 | `b93894c69ae08273c89ef657fa1ff38e` |
| `diag/fd4_rmax0.json` | 1101 | `978483e0b6e0fdcd52be1355483b3820` |
| `diag/fd4_rmax1.json` | 1103 | `af66b559ef2b701b3c041cf8bfca3c69` |
| `diag/fd4_smoke.json` | 1099 | `0ef3ccd0bd6191730078cadbe4d4d02b` |
| `diag/fd4_th0.json` | 1103 | `de6515132175cc8c2173169d70ec4d03` |
| `diag/fd4_th1.json` | 1106 | `9dc79c355207460f6f014addd1adaf03` |
| `diag/fd_smoke.json` | 889 | `c884ab2de711bd92300d203c2d1a012d` |
| `diag/floor_20261008/E-A.log` | 1352 | `fede5d3a0570d2d7d676e5a7eb3e869c` |
| `diag/floor_20261008/E-A_up20_lx60_h20.json` | 1013 | `830ccc0b5b00fc9c45055eb422a12cec` |
| `diag/floor_20261008/E-B.log` | 1355 | `5e6d67a0460f7b5172851ffce3cd3286` |
| `diag/floor_20261008/E-B_up20_lx50_h20.json` | 1016 | `8a765247070052aec57276c69d7abeba` |
| `diag/floor_20261008/E-C.log` | 1354 | `65681fd83f82071899d616f893254e32` |
| `diag/floor_20261008/E-C_up10_lx40_h40.json` | 1014 | `9a4f9209f0fbfc08ec996695d0e93015` |
| `diag/floor_20261008/E-D.log` | 1355 | `3477854797911c4cd54ca6327fe825a8` |
| `diag/floor_20261008/E-D_up40_lx70_h20.json` | 1015 | `ea07a1286f1789558ad06d0a3539bfb5` |
| `diag/floor_20261008/analysis.json` | 9582 | `39e9799e69d113ccab166fb98bba0ea3` |
| `diag/floor_20261008/diag_floor.py` | 6478 | `19eac10fba9f057eebc199b192c65313` |
| `diag/floor_20261008/floor_analysis.py` | 5421 | `e5fc0c631cbe9c370c71ee77e6a7db35` |
| `diag/floor_20261008/floor_summary.py` | 2006 | `aaf785d87e897b7bba76487dff461324` |
| `diag/floor_20261008/summary.json` | 2545 | `f3ee0f1623f1c127a33096c78e5163fa` |
| `diag/geom.json` | 5166 | `c2e460d9a90c3f4b49405630adcdee38` |
| `formal/ladder_summary.json` | 2737 | `79d26936dd24d1d72104548c2fe08ddc` |
| `formal/re20_D160.json` | 1012 | `c983ecba292df07e5a6d802b0662a3f6` |
| `formal/re20_D40.json` | 1002 | `f547c0af742b5f7f4d3daf716ba2f737` |
| `formal/re20_D80.json` | 1006 | `0d1d1182f2c356b239b4f1f4712dd65f` |
| `formal/re40_D160.json` | 1013 | `6d0c67f28e8a2e30ce88f1de1117940c` |
| `formal/re40_D40.json` | 1004 | `d3dcd63d0089a736a38b3558c0146af8` |
| `formal/re40_D80.json` | 1004 | `9ff2984317ff563cab79c3dbd6de628c` |
| `formal/verdict.json` | 3176 | `d4326c7de93a32f97444c5df3ddfb5a8` |
| `prereg/prereg.md` | 11931 | `90860b9f7dffa777d66851d38883302f` |
| `ref/extracted_primary.json` | 3701 | `36199512e11495b026962ad57f718a0c` |
| `ref/ref_lock.json` | 15044 | `4de75169bb8671ada586b5ca5232aabb` |
| `src/analyze_fd4.py` | 2503 | `7c6b489cbf35aa814cbe4b90f9ec97d0` |
| `src/build_ladder.py` | 2119 | `8bb1f11bb40b7fd29c2b183f585e45ee` |
| `src/build_reflock.py` | 8902 | `34736bb202999cc86b82a439ba0b7cd6` |
| `src/diag_domain.py` | 1137 | `3caaa3074a4c13d7a83492256ef6f6b0` |
| `src/diag_geom.py` | 4361 | `5b9b95c9c4d53b3a6c35fe1d7e9fe3dc` |
| `src/diag_geom_report.py` | 1127 | `c62f8831dd629d5e8455dec90db14bfe` |
| `src/extract_refs.py` | 4901 | `c906ecf841db34877f7d79d815fb297f` |
| `src/formal_driver.sh` | 810 | `a7f30635f5b5a3a3fcff04dd40d57a43` |
| `src/fornberg_dump.py` | 623 | `b0d2c65179602d4eb86e55c995969803` |
| `src/indep_solver.py` | 13274 | `0267806f18639eb4b92c531deaf1429e` |
| `src/pdf_probe.py` | 560 | `d8fb363dc7259a58e56fda7643b5c886` |
| `src/ref_locate.py` | 1371 | `50114f45218c0b7aaa3e303c0dec9ba2` |
| `src/run_formal.py` | 7085 | `6cdb9d47dc37f0421bed3470a610b1c7` |
| `src/test_solver_units.py` | 2838 | `90a981a6968d9a960e2740f7dce9eec8` |

## 复现（已在临时副本逐位复现，暂存未动）

- `src/build_ladder.py`（BASE 重定向至副本）重生成 `formal/ladder_summary.json`：逐位一致（`79d26936dd24d1d72104548c2fe08ddc`），且与 `formal/verdict.json` 内 pinned md5 相同。

## 暂存留档（不入库）

- formal/trace_re*.npz、diag/fd*_field.npz、diag/floor_20261008/trace_*.npz（追迹/场二进制）
- pdf/（Fornberg 1980、Rajani 2009、refsol1310.6641 原文与源码包）、ref/fornberg1980_fulltext.txt、ref/refsol_src/
- pilot/（Phase-0 试点 JSON+npz）与 src/pilot_lb_cpu.py、src/probe_growth.py
- logs/（除 floor E-*.log 已入库外的运行日志）

staging 根：`/nfs/wangxi/runs/bm_widen_w11_20261006/`

## 披露

- 本目录 13 个 .py 经 ruff format/check --fix 规范化（import 重排/拆分、空白、未用 import 删除：见 verify_manifest.json:lint_normalization 的逐文件staging/repo md5 对照）；AST 等价已验证，判决复现不受影响（上文逐位一致）。
- run/verify 脚本内 `BASE` 仍指向战役暂存绝对路径（冻结记录原样入库，未改写）。
- README PROVENANCE 复核：64/64 条数字对归档 JSON 逐键机器复核一致（64/64 all_match=True）。
