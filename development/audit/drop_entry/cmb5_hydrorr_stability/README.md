# DE-X-CMB5-HYDRORR-STAB

Mission m27100 explored RR dose interpolation for the Santini case 3 drop entry.
The archive contains [the preregistration](preregistration.md),
[the original verdict](cmb5_verdict.json) and four raw histories.
The maintained benchmark is [single drop into a deep pool](../../../../benchmarks/water_entry/single_drop/README.md).

| Arm | Raw history | Historical termination |
| --- | --- | --- |
| RR dose 0.50 | [cmb5_d050](arms/cmb5_d050/result.json.gz) | `health_limit` on step 4 |
| RR dose 0.75 | [cmb5_d075](arms/cmb5_d075/result.json.gz) | `health_limit` on step 4 |
| RR dose 0.75, velocity limit 0.13 | [cmb5_ul13](arms/cmb5_ul13/result.json.gz) | `health_limit` on step 4 |
| RR full-dose comparison | [cmb5_parity](arms/cmb5_parity/result.json.gz) | 200-step comparison |

The first three runs stopped before contact. The parity comparison was a short
historical check. Repeating the original cross-run comparison or publishing its
figures requires the CMB3 raw histories from the m27044 workspace. The
[original tools](original_tools/) retain their workspace-specific inputs.
The [migration ledger](../../../../docs/architecture/migrations/cmb5-hydrorr-stability-2026-11-23.json)
records all prior Git paths and the exact duplicate verdict consolidated here.
