# MANIFEST — benchmarks/archived/xh_hybrid_20261006

Append-only archive assembled 2026-10-06 from staging `/nfs/wangxi/runs/xh_hybrid_20261006`
(byte-identical copy; sha256 proof in MANIFEST_evidence.csv).

## Excluded Python files (lint gate)

Gate = `ruff check` AND `ruff format --check` (venv ruff 0.16.3, repo
config of worktree /nfs/wangxi/worktrees/x_arch, which is exactly the
configuration the final gate `ruff check benchmarks/archived` uses).
Files failing either command were dropped from the archive and are
preserved in staging only:

| path | sha256 | md5 | ruff check | ruff format | note |
|---|---|---|---|---|---|
| `phase1/scripts/e2diag.py` | `a5c7578739167645e9872f3647767d035aaccd441067924c605f80b249977be4` | `9bb152f654442732b6c6aee317664cd7` | rc=1, 1 error(s) | rc=1 (would reformat) | preserved in staging `/nfs/wangxi/runs/xh_hybrid_20261006/phase1/scripts/e2diag.py` only |
| `phase1/scripts/e3diag.py` | `f768ecdd4cce4e06aa1181e155706424f8f834974ba500dc1b665c323e91b8bc` | `529e1d26d19c27a12c9b6c41d52f0008` | rc=1, 1 error(s) | rc=1 (would reformat) | preserved in staging `/nfs/wangxi/runs/xh_hybrid_20261006/phase1/scripts/e3diag.py` only |
| `phase1/scripts/r5_pressure_probe.py` | `e3ed1c8ebd96a2dceb3ff09879e7be1b83d20e43c8f4660c5b12883f28ef3aa4` | `4314ce0d71914989116fb24537906331` | rc=1, 1 error(s) | rc=1 (would reformat) | preserved in staging `/nfs/wangxi/runs/xh_hybrid_20261006/phase1/scripts/r5_pressure_probe.py` only |
| `phase1/scripts/r5diag_2d.py` | `7f122c147a722d08b8131329a9aef86f9cef41a1c88351e104b740ab851d51af` | `dde77ec6a9888665a8906a0e6809cd15` | rc=1, 1 error(s) | rc=1 (would reformat) | preserved in staging `/nfs/wangxi/runs/xh_hybrid_20261006/phase1/scripts/r5diag_2d.py` only |
| `phase1/scripts/t0_fixture.py` | `775e44d6a8ec36781046960ebc5e30bf9a9306ee52b4f1591a52f39cef101e42` | `a2647c793094ccb021c6ad07285aa1b1` | rc=1, 6 error(s) | rc=1 (would reformat) | preserved in staging `/nfs/wangxi/runs/xh_hybrid_20261006/phase1/scripts/t0_fixture.py` only |
| `phase1/scripts/t1_laplace.py` | `bd29c7d6a88384ac95654e53706ab57f89a0abe6f97365e22a4fc913646638e1` | `03c2b299e820dd5a5ef769cc6dda425e` | rc=1, 1 error(s) | rc=1 (would reformat) | preserved in staging `/nfs/wangxi/runs/xh_hybrid_20261006/phase1/scripts/t1_laplace.py` only |
| `phase1/scripts/t2_poiseuille.py` | `ff6fcf172cf586c5819b44e8a9b69ee798e7183c817b22a2c6b1fa1029b3b390` | `7c9242f26f7e4c41aa9e600f1aee76a6` | rc=1, 3 error(s) | rc=1 (would reformat) | preserved in staging `/nfs/wangxi/runs/xh_hybrid_20261006/phase1/scripts/t2_poiseuille.py` only |
| `phase1/scripts/t3_hadroplet.py` | `7866ea8e2520a5a20036f7e701e82b3081278fe2056b2594421951a7a07fc78c` | `3343afb4c32397713e1b642349f45e3e` | rc=1, 1 error(s) | rc=1 (would reformat) | preserved in staging `/nfs/wangxi/runs/xh_hybrid_20261006/phase1/scripts/t3_hadroplet.py` only |
| `phase1/scripts/t3_hadroplet_v1_neverrun.py` | `efcba5b095ec2f51c4db3ed636b6f40949443e7de3107f2f7cc41f7ccf69ba0e` | `fed94aea88ab8b5fdf0009d9fc9865e9` | rc=1, 1 error(s) | rc=1 (would reformat) | preserved in staging `/nfs/wangxi/runs/xh_hybrid_20261006/phase1/scripts/t3_hadroplet_v1_neverrun.py` only |
| `phase1/scripts/t6_density10.py` | `c0e13e7d835725bf7e875c3e02e60f1a345bf93f7b724abff4047aae9acb378f` | `2a55b7f0ce39649d0749cb307dfece22` | rc=1, 1 error(s) | rc=1 (would reformat) | preserved in staging `/nfs/wangxi/runs/xh_hybrid_20261006/phase1/scripts/t6_density10.py` only |
| `phase1/scripts/xh1_env.py` | `24d0c841e50fa591d83598b785a4c9505a59e97dd11dd0459a617ea5573a8024` | `2571a3a9d509853c232c2c532877697b` | rc=0, 0 error(s) | rc=1 (would reformat) | preserved in staging `/nfs/wangxi/runs/xh_hybrid_20261006/phase1/scripts/xh1_env.py` only |
| `probe/p1_cg_force_compat.py` | `41b5978fc1ee7780f01e0aa263e3473a2e2c01425b8af1c3e92f4e38875fc0a6` | `4e176f96c1585518ffe15ef5ef22c039` | rc=1, 1 error(s) | rc=1 (would reformat) | preserved in staging `/nfs/wangxi/runs/xh_hybrid_20261006/probe/p1_cg_force_compat.py` only |
| `probe/p2_force_portability.py` | `f06e61bafdc44164156ffad5d8cc7620d1e07fdd606b6b0b4bd7a1e2107bc83d` | `636ea0b9888e8535f9df42f54d3f25d7` | rc=1, 1 error(s) | rc=1 (would reformat) | preserved in staging `/nfs/wangxi/runs/xh_hybrid_20261006/probe/p2_force_portability.py` only |
| `probe/p3_density_contrast.py` | `b079b983565a8c8bb5b2f912bc7fac05bdf0e5a94c3b256db5b8370bc320eaa0` | `952e0b6b2563d9bc83dbcb8f72a99e01` | rc=1, 1 error(s) | rc=1 (would reformat) | preserved in staging `/nfs/wangxi/runs/xh_hybrid_20261006/probe/p3_density_contrast.py` only |
| `probe/p4_cac_poiseuille_rootcause.py` | `319bcdcae6d48a6942b9ee5f255c45fdbec7dc799e6b2062816800fde33da43c` | `8aaa3c0f3e437eede2949f7af79ed844` | rc=1, 1 error(s) | rc=1 (would reformat) | preserved in staging `/nfs/wangxi/runs/xh_hybrid_20261006/probe/p4_cac_poiseuille_rootcause.py` only |
| `probe/xh_common.py` | `0a14579c863adee630969f599534bd47e618ab54276de755260c7be29d6c74ff` | `5045f0dc55c06760f3f6cd599767636b` | rc=0, 0 error(s) | rc=1 (would reformat) | preserved in staging `/nfs/wangxi/runs/xh_hybrid_20261006/probe/xh_common.py` only |

## Other exclusions / notes

- `__pycache__/`, `*.pyc`, `.ruff_cache/` directories: tool caches, not evidence.
- `engine/hybrid_cg_cac.py`: md5-frozen evaluated artifact (md5
  02dbe4b1ec5e337f94e91f132c16d5e3), included byte-identical and
  therefore EXEMPT from the lint gate; its single preserved ruff
  finding is disclosed verbatim in this track README.md.
