# MANIFEST — benchmarks/archived/x_campaign_20261004/xc_freesurface

Append-only archive assembled 2026-10-06 from staging `/nfs/wangxi/runs/x_campaign_20261004/xc_freesurface`
(byte-identical copy; sha256 proof in MANIFEST_evidence.csv).

## Excluded Python files (lint gate)

Gate = `ruff check` AND `ruff format --check` (venv ruff 0.16.3, repo
config of worktree /nfs/wangxi/worktrees/x_arch, which is exactly the
configuration the final gate `ruff check benchmarks/archived` uses).
Files failing either command were dropped from the archive and are
preserved in staging only:

| path | sha256 | md5 | ruff check | ruff format | note |
|---|---|---|---|---|---|
| `scripts/judge_w1_battery.py` | `98be2d0723ec493ec9d33754ec3ecbed1f87dcfd36e7284268c960581dea03f7` | `2559ee07e9465ef729a1ed62fb55bb17` | rc=1, 1 error(s) | rc=1 (would reformat) | preserved in staging `/nfs/wangxi/runs/x_campaign_20261004/xc_freesurface/scripts/judge_w1_battery.py` only |
| `scripts/probe_front_attribution.py` | `38cd7cbed8146c0695d83e5615ba517bbf811b3ecbe3987ff2cc7eb799294fed` | `4a04c24855b29ada0b196383c3747d71` | rc=1, 2 error(s) | rc=1 (would reformat) | preserved in staging `/nfs/wangxi/runs/x_campaign_20261004/xc_freesurface/scripts/probe_front_attribution.py` only |
| `scripts/probe_w1_channels.py` | `a33e4395ad0ce840ed47273d5b354324972653458c7b00105073074e479267d4` | `e76200795716d22efb08bcd086d375d7` | rc=1, 4 error(s) | rc=1 (would reformat) | preserved in staging `/nfs/wangxi/runs/x_campaign_20261004/xc_freesurface/scripts/probe_w1_channels.py` only |
| `scripts/run_m3_dambreak_w1.py` | `0e9d557493f519ea1bfe0d58950064c139f11852b4b905273e94da50a6f49fa1` | `867a95a7b78701ca1909adc896afc178` | rc=1, 1 error(s) | rc=1 (would reformat) | preserved in staging `/nfs/wangxi/runs/x_campaign_20261004/xc_freesurface/scripts/run_m3_dambreak_w1.py` only |
| `scripts/run_w1_battery.py` | `834078ff5b3204657bade63a61ece92113680d49172401b659a4b009fac91dc3` | `07398a053ecd7b5f363c70b4ac1c1a4c` | rc=1, 1 error(s) | rc=1 (would reformat) | preserved in staging `/nfs/wangxi/runs/x_campaign_20261004/xc_freesurface/scripts/run_w1_battery.py` only |
| `scripts/smoke_w1_ab.py` | `79d87d22df9c90475badcba794b84291dbe2a79a67c08f64b3743fc8ee3f6aed` | `413c29dc3a234c84d496795c4592fc69` | rc=1, 1 error(s) | rc=1 (would reformat) | preserved in staging `/nfs/wangxi/runs/x_campaign_20261004/xc_freesurface/scripts/smoke_w1_ab.py` only |
| `src_patched/free_surface_engine_2d_w1.py` | `cd09c987ce6fcbb5f934b5aa8fd7a267cdd7bc450eaf495a2dcc03dfccb129f4` | `8782d48dc7967caee07c8521f5aa22d8` | rc=1, 1 error(s) | rc=1 (would reformat) | preserved in staging `/nfs/wangxi/runs/x_campaign_20261004/xc_freesurface/src_patched/free_surface_engine_2d_w1.py` only |

## Other exclusions / notes

- `__pycache__/`, `*.pyc`, `.ruff_cache/` directories: tool caches, not evidence.
