# MANIFEST — benchmarks/archived/x_campaign_20261004/xb_engine

Append-only archive assembled 2026-10-06 from staging `/nfs/wangxi/runs/x_campaign_20261004/xb_engine`
(byte-identical copy; sha256 proof in MANIFEST_evidence.csv).

## Excluded Python files (lint gate)

Gate = `ruff check` AND `ruff format --check` (venv ruff 0.16.3, repo
config of worktree /nfs/wangxi/worktrees/x_arch, which is exactly the
configuration the final gate `ruff check benchmarks/archived` uses).
Files failing either command were dropped from the archive and are
preserved in staging only:

| path | sha256 | md5 | ruff check | ruff format | note |
|---|---|---|---|---|---|
| `ctrl/ctrl_xb2.py` | `4f34180983fe0e10c5a45b3ae6d37291fa78b83f946f1d5a47c7589844bbcd31` | `015e00f987a51cbf9ab34fc13e5b015c` | rc=0, 0 error(s) | rc=1 (would reformat) | preserved in staging `/nfs/wangxi/runs/x_campaign_20261004/xb_engine/ctrl/ctrl_xb2.py` only |
| `refs/b5_hr_droplet.py` | `48dd349929e0bc13638d895d5158f730a4547c690cfec810ab133c370b689e6a` | `f21a9a5f7326581790258c396730b474` | rc=1, 2 error(s) | rc=1 (would reformat) | preserved in staging `/nfs/wangxi/runs/x_campaign_20261004/xb_engine/refs/b5_hr_droplet.py` only |
| `refs/calib_sigma.py` | `dae819f6e6a1a316dfcf6fce65e75f1369d9bb3051810ecc01efacf7258adeea` | `41d430fa9014e1048fe557cda0cb8bdb` | rc=1, 3 error(s) | rc=1 (would reformat) | preserved in staging `/nfs/wangxi/runs/x_campaign_20261004/xb_engine/refs/calib_sigma.py` only |
| `refs/run_formal_c.py` | `a43109de2a94f9f0e32d012b0859c8fd7d53026ab1ed0155585ba6adef1fef62` | `622c41a000ff781f3c219b6b3f201e98` | rc=1, 1 error(s) | rc=1 (would reformat) | preserved in staging `/nfs/wangxi/runs/x_campaign_20261004/xb_engine/refs/run_formal_c.py` only |
| `refs/scmp_lib.py` | `35f81f0754dfaad4a255ab9e5977d4f61df0abe8805f80ecda640545f72d0b04` | `4cdecef5ef26245a36ac586e2cfb60bb` | rc=1, 4 error(s) | rc=1 (would reformat) | preserved in staging `/nfs/wangxi/runs/x_campaign_20261004/xb_engine/refs/scmp_lib.py` only |
| `scmp_lib_xb.py` | `e6a9aa4146703e627a9f0b8f0d11f5f478bce6832912e296a3ab502cbebffeff` | `97d2259bbc7373d2dd769943cf0743c5` | rc=1, 3 error(s) | rc=1 (would reformat) | preserved in staging `/nfs/wangxi/runs/x_campaign_20261004/xb_engine/scmp_lib_xb.py` only |
| `src_patched/tensorlbm/cac_lbm.py` | `6447a37a0857be7d00e89852b5a5cfccb818572fd1ba2232a5dbc201678bca13` | `edfa261570f867208d9763292f814b7e` | rc=0, 0 error(s) | rc=1 (would reformat) | preserved in staging `/nfs/wangxi/runs/x_campaign_20261004/xb_engine/src_patched/tensorlbm/cac_lbm.py` only |
| `tests/test_iso4.py` | `afd99bba0e39cbb696b957bbfbc2104080e01de0aa8e409c212e10ef4377583b` | `986e88548ed42956a2f6d14bae2e1e9d` | rc=0, 0 error(s) | rc=1 (would reformat) | preserved in staging `/nfs/wangxi/runs/x_campaign_20261004/xb_engine/tests/test_iso4.py` only |
| `tests/test_multirange.py` | `ead4a165450776b251d944e03fd487930a124b0e5e73bdce935c1de5d662e53a` | `56c1f7065c8eecb4794601a773b23e81` | rc=1, 1 error(s) | rc=1 (would reformat) | preserved in staging `/nfs/wangxi/runs/x_campaign_20261004/xb_engine/tests/test_multirange.py` only |
| `xb1_formal.py` | `20f9ed2e1610e39414b1b9c71d0c439ee688741b5a9c9ee7c93e0f7bb51cd568` | `3e885d7c244a53083af10409a7adeb1d` | rc=1, 1 error(s) | rc=1 (would reformat) | preserved in staging `/nfs/wangxi/runs/x_campaign_20261004/xb_engine/xb1_formal.py` only |
| `xb2_formal.py` | `2c7ed5657a31aed578537f667a8e2321107b5536b70ba6bc49be25dd33884a2e` | `ca25c95af0ce253617a5e631a313bc11` | rc=1, 1 error(s) | rc=1 (would reformat) | preserved in staging `/nfs/wangxi/runs/x_campaign_20261004/xb_engine/xb2_formal.py` only |

## Other exclusions / notes

- `__pycache__/`, `*.pyc`, `.ruff_cache/` directories: tool caches, not evidence.
- `out/xb2_v3_done.marker`: 5-byte completion flag written by relay waiter xb2_v3_waiter2.sh after the xb2 v3 formal run finished (crash-timeline evidence; kept).
- `out/xb2_partial.json / out/xb2_v3_partial.json`: partial-output JSONs of the crashed v1 run and the v3 run; preserved as crash-timeline evidence per task policy.
