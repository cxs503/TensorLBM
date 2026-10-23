# MANIFEST — benchmarks/archived/x_campaign_20261004/xa_rehab

Append-only archive assembled 2026-10-06 from staging `/nfs/wangxi/runs/x_campaign_20261004/xa_rehab`
(byte-identical copy; sha256 proof in MANIFEST_evidence.csv).

## Excluded Python files (lint gate)

Gate = `ruff check` AND `ruff format --check` (venv ruff 0.16.3, repo
config of worktree /nfs/wangxi/worktrees/x_arch, which is exactly the
configuration the final gate `ruff check benchmarks/archived` uses).
Files failing either command were dropped from the archive and are
preserved in staging only:

| path | sha256 | md5 | ruff check | ruff format | note |
|---|---|---|---|---|---|
| `_packaging/pack_result.py` | `c2a4fed73f085b6ebdd5f17c42e891e92b7c8fb2a213fb453211af3193ee25ff` | `a1856a1d43b5bd594ede290c8eb6f093` | rc=0, 0 error(s) | rc=1 (would reformat) | preserved in staging `/nfs/wangxi/runs/x_campaign_20261004/xa_rehab/_packaging/pack_result.py` only |
| `_packaging/smoke_driver.py` | `eaf6e7a7b7cbe9f5fb4d9f5cd51184e3a0642a56017b24114af467a91289b57e` | `eaf84d65cc1ea46766992e7b60f6135f` | rc=0, 0 error(s) | rc=1 (would reformat) | preserved in staging `/nfs/wangxi/runs/x_campaign_20261004/xa_rehab/_packaging/smoke_driver.py` only |
| `archive_c1g2/color_gradient2d.py` | `2d13e8df8a8ebf75b5d3ef8705ef88faa74814268bda9f814a2977344c588e18` | `81203d6d6f00104f4853a6db81815fe0` | rc=0, 0 error(s) | rc=1 (would reformat) | preserved in staging `/nfs/wangxi/runs/x_campaign_20261004/xa_rehab/archive_c1g2/color_gradient2d.py` only |
| `archive_c1g2/run.py` | `9d5936ec834c590dc27595884c2870e00b1a925551e7761b883383428d40c67a` | `3576b36137232699b090ff4d343781a1` | rc=1, 1 error(s) | rc=0 (clean) | preserved in staging `/nfs/wangxi/runs/x_campaign_20261004/xa_rehab/archive_c1g2/run.py` only |
| `ctrl/ctrl_x1.py` | `d787971aa95d24574e527787940678eab7ab47fbfbbf89eb65a2113910ad589c` | `16ad500e1f9a44eae812d9b9b5f8619d` | rc=0, 0 error(s) | rc=1 (would reformat) | preserved in staging `/nfs/wangxi/runs/x_campaign_20261004/xa_rehab/ctrl/ctrl_x1.py` only |
| `ctrl/ctrl_xa3.py` | `be95996202e4bb6aadebbcf844453132a7c65c450c1152e4a7f630e88b02d321` | `e8b68b9e7fee1bb14d80aae9e53d09cc` | rc=0, 0 error(s) | rc=1 (would reformat) | preserved in staging `/nfs/wangxi/runs/x_campaign_20261004/xa_rehab/ctrl/ctrl_xa3.py` only |
| `ctrl/ctrl_xa5.py` | `9cfb27bc2ef05dc147bde7f2e1f795b0affc2cc6dced9f1266887d4494b67d4d` | `6c21c217a12fce5ec5a5dd68cd644667` | rc=1, 2 error(s) | rc=1 (would reformat) | preserved in staging `/nfs/wangxi/runs/x_campaign_20261004/xa_rehab/ctrl/ctrl_xa5.py` only |
| `x1_crossval/ctrl/x1_cg_health_recheck.py` | `6bd7b717f8be48da6cb3c6ed62ba59a27749657a18bf2cdf897cbac66089a3c8` | `d0bd0ba1af4298b0eee65ebd6bf6f915` | rc=1, 1 error(s) | rc=1 (would reformat) | preserved in staging `/nfs/wangxi/runs/x_campaign_20261004/xa_rehab/x1_crossval/ctrl/x1_cg_health_recheck.py` only |
| `x1_crossval/scripts/b2_poiseuille_x1.py` | `6c00e97123d81e77c79768861e8af06f6aa31a842a2b8767d09bb2ee031fa733` | `cbc5c1fe9bfe01d10460eaf3af552db8` | rc=1, 3 error(s) | rc=1 (would reformat) | preserved in staging `/nfs/wangxi/runs/x_campaign_20261004/xa_rehab/x1_crossval/scripts/b2_poiseuille_x1.py` only |
| `x1_crossval/x1_judge.py` | `2fb6afb9dc687c8611e2cf12084800b6cbcdaec812692e7323cc5fa81299ba62` | `8afc67a20643efbf1e10b77d976945c0` | rc=0, 0 error(s) | rc=1 (would reformat) | preserved in staging `/nfs/wangxi/runs/x_campaign_20261004/xa_rehab/x1_crossval/x1_judge.py` only |
| `x1_crossval/x1_run.py` | `4f3edf3c17720df34f3c0857a3d0091ef578919647aa073af9ce7c909d0bcc13` | `d842fa960900f32c17cc53aa6a8ea08a` | rc=1, 1 error(s) | rc=1 (would reformat) | preserved in staging `/nfs/wangxi/runs/x_campaign_20261004/xa_rehab/x1_crossval/x1_run.py` only |
| `xa2_m2g2_noise/ctrl/ctrl_xa2.py` | `e8a5fb46fec9f6e71327da64276abd5861aed6e97a63c7ac73b7237aea19a707` | `03146432173f9546c49836382bb2c8c8` | rc=0, 0 error(s) | rc=1 (would reformat) | preserved in staging `/nfs/wangxi/runs/x_campaign_20261004/xa_rehab/xa2_m2g2_noise/ctrl/ctrl_xa2.py` only |
| `xa2_m2g2_noise/xa2_judge.py` | `e1085ed62a060fe78592630050ac38c8b90442777b987e4825bfbad5a54934cb` | `8b828d77e87ead4d73d85a2620a43c40` | rc=0, 0 error(s) | rc=1 (would reformat) | preserved in staging `/nfs/wangxi/runs/x_campaign_20261004/xa_rehab/xa2_m2g2_noise/xa2_judge.py` only |
| `xa2_m2g2_noise/xa2_run.py` | `36584250de86eaf4a279859710a28f8753253011f75348d1269c772077ebe438` | `26e0df2f3c06cc712e123f8b280b38be` | rc=0, 0 error(s) | rc=1 (would reformat) | preserved in staging `/nfs/wangxi/runs/x_campaign_20261004/xa_rehab/xa2_m2g2_noise/xa2_run.py` only |
| `xa3_t2_sigma/xa3_judge.py` | `9957b2a1402c72c06b74698cf7b1ca10680e52a43b76fe4eb59238c666658ee4` | `9c40f1404eea4617b61ae071eda2914b` | rc=1, 1 error(s) | rc=1 (would reformat) | preserved in staging `/nfs/wangxi/runs/x_campaign_20261004/xa_rehab/xa3_t2_sigma/xa3_judge.py` only |
| `xa3_t2_sigma/xa3_run.py` | `6071a4e9a7e733666150ffdd8c1320075e75f811c486f8954b65e2a82bd8901a` | `008c0c82362bea9f2e6c5c75eefc1f05` | rc=0, 0 error(s) | rc=1 (would reformat) | preserved in staging `/nfs/wangxi/runs/x_campaign_20261004/xa_rehab/xa3_t2_sigma/xa3_run.py` only |
| `xa4_m7_servo/xa4_judge.py` | `ce48c2a5b0dd63acd0e4817f912e33117dcff556b992ea305989f7cdee197190` | `5c2cee84bb0d8218315d28fba63d9214` | rc=0, 0 error(s) | rc=1 (would reformat) | preserved in staging `/nfs/wangxi/runs/x_campaign_20261004/xa_rehab/xa4_m7_servo/xa4_judge.py` only |
| `xa4_m7_servo/xa4_run.py` | `d16c7b7f30f09e08bf86d9c7bdda96857e1ad1a96a45185f529ccf2e9cfe9fab` | `66ae199e70baac984b6c4726d9a7cc6c` | rc=1, 1 error(s) | rc=1 (would reformat) | preserved in staging `/nfs/wangxi/runs/x_campaign_20261004/xa_rehab/xa4_m7_servo/xa4_run.py` only |
| `xa5_cg_beta/xa5_judge.py` | `efdb6187a7051ac85e2aba60268050641c4faaba297dc597c58da93496b21bc6` | `06454068485290688edf09bc15d06673` | rc=0, 0 error(s) | rc=1 (would reformat) | preserved in staging `/nfs/wangxi/runs/x_campaign_20261004/xa_rehab/xa5_cg_beta/xa5_judge.py` only |
| `xa5_cg_beta/xa5_run.py` | `f267d5dcd186055f17d20861be76a7dbf72ac672b56b95ef8a3378139b4e8266` | `e9120190221c6ac9702c08ac6bbb35e7` | rc=0, 0 error(s) | rc=1 (would reformat) | preserved in staging `/nfs/wangxi/runs/x_campaign_20261004/xa_rehab/xa5_cg_beta/xa5_run.py` only |

## Other exclusions / notes

- `__pycache__/`, `*.pyc`, `.ruff_cache/` directories: tool caches, not evidence.
