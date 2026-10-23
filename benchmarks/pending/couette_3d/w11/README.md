# w11/ — W11-D 判决机器档案（couette_3d，verdict FAIL）

本目录是 2026-10 Wave-11 判决工件的入库副本；判决叙述、门禁口径与全部
判据数的 PROVENANCE 溯源见上级 [README.md](../README.md)。本 README 的
全部表格由机器脚本生成（`w11_archive_gen/gen_archive.py`，判决数字零手抄；
核验明细 `w11_archive_gen/verify_manifest.json`，均在战役暂存内）。

## 判决文件（md5）

| 文件 | md5 |
|---|---|
| `gate/verdict_v2.json` | `e14f11ddd58e715e4c8d647f04250cf3` |

## 判决读数（机器读自判决 JSON）

| 项 | 值 | 备注 |
|---|---|---|
| G1 accuracy | True |  |
| G2a fp32 floor | True |  |
| G2b fp64 exactness | True |  |
| G3 health | False | H128 fp32 steady=false (24000 cap) |
| verdict | FAIL |  |

## 归档清单（28 文件，115344 B）

| 路径 | B | md5 |
|---|---|---|
| `diag/couette_diag.py` | 5641 | `a31b91a72c43c5deeacd83a6f3f8e762` |
| `diag/couette_diag2.py` | 4693 | `67894d7afff9f8e36c5c5196ee371948` |
| `diag/diag_arms.json` | 18378 | `0f6ac244fe5c9e98fd6d6b62ff5e142c` |
| `diag/diag_arms.stdout.log` | 3730 | `3684234ce8e0b20eb977f5e95c5b16cf` |
| `diag/diag_dtype.json` | 10252 | `a0be5508820b2c8658534ee10ff62843` |
| `diag/diag_dtype.stdout.log` | 2984 | `0d0dd76ec33ad7ee1bae41c5d2f3897b` |
| `diag/pilot/case_H16.json` | 1661 | `87506d5812209aee575dc1067c9dcd15` |
| `diag/pilot/case_H32.json` | 2195 | `d7cc435f042cc3edb4698dff3c868c8f` |
| `diag/pilot/case_H64.json` | 3214 | `db154bcd2e22ef481e38e15109ffe9da` |
| `diag/pilot/pilot.log` | 7328 | `04ecb108b06650ca7ded842a40c2e875` |
| `diag/pilot/provenance.txt` | 120 | `6c8fbe7574b4b693da3c302d9f8b87df` |
| `diag/pilot/run_pilot.sh` | 663 | `0524918751110783653df97548d7d797` |
| `diag/v2_smoke/case_H16_fp32.json` | 1934 | `df8ddc31b2058b119e3cadbc1ddfa83b` |
| `gate/fp32/case_H128_fp32.json` | 5523 | `406e4adc9c07263b784259e702397c8b` |
| `gate/fp32/case_H32_fp32.json` | 2461 | `591034251b441b1871c1003f42492cba` |
| `gate/fp32/case_H64_fp32.json` | 3493 | `35b75bb9494b9fd1ebdfded8ba375a6a` |
| `gate/fp32.run.log` | 431 | `c93c6f517abaa746a375b28380231b65` |
| `gate/fp64/case_H128_fp64.json` | 5551 | `60a9c72ffb48e2ed88f9bb54cf25e8a2` |
| `gate/fp64/case_H16_fp64.json` | 1929 | `028d0d3030e73f507f076d08380a7e9b` |
| `gate/fp64/case_H32_fp64.json` | 2481 | `0b0bfaa86169651f32fbeae2194129ae` |
| `gate/fp64/case_H64_fp64.json` | 3506 | `6a11e2ac8f6df06027359acfae2330af` |
| `gate/fp64.run.log` | 578 | `33d431fd95deeb5803134fec5622f68e` |
| `gate/verdict_v2.json` | 2710 | `e14f11ddd58e715e4c8d647f04250cf3` |
| `prereg.md` | 10560 | `c6ad6e2ecb9e85a1e51adfa76472ad6a` |
| `prereg.md.md5` | 292 | `6435d5dd50fc17f1e99fa20ddbbde25f` |
| `reflock.json` | 1007 | `f5667400278fcc210227fdd9d787dd20` |
| `run_v2.py` | 8519 | `6244d1e4af57a61b755304e773c93ffc` |
| `verify_v2.py` | 3510 | `b399d6eef57574dccb37a049ac8c2100` |

## 复现（已在临时副本逐位复现，暂存未动）

- `verify_v2.py`（本目录内相对路径即可运行）重生成 `gate/verdict_v2.json`：逐位一致（`e14f11ddd58e715e4c8d647f04250cf3`，exit 1=FAIL 预期）。

## 暂存留档（不入库）

- （无——本案例判决链工件全量入库；staging 留 d_siblings/ 共享层 gate_run.sh 编排器属 sphere27 姊妹档）

staging 根：`/nfs/wangxi/runs/bm_widen_w11_20261006/`

## 披露

- 本目录 4 个 .py 经 ruff format/check --fix 规范化（import 重排/拆分、空白、未用 import 删除：见 verify_manifest.json:lint_normalization 的逐文件staging/repo md5 对照）；AST 等价已验证，判决复现不受影响（上文逐位一致）。
- run/verify 脚本内 `BASE` 仍指向战役暂存绝对路径（冻结记录原样入库，未改写）。
- README PROVENANCE 复核：38/38 条数字对归档 JSON 逐键机器复核一致（38/38 all_match=True）。
