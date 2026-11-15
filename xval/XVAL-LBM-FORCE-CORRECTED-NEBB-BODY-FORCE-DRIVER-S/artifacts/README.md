# Committed run artifacts — XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S

`runs/` is gitignored (`.gitignore:108`), which is why Rounds 1–4 produced numbers that never
entered the tree and the maturity gate could only see prose. Everything the gate machine-checks
lives **here**, committed:

| file | producer | role |
|---|---|---|
| `results.json` | `benchmarks/xval/xval_vivsim_nebb_bodyforce.py` | arm A (vivsim transcription) vs arm B (`tensorlbm.nebb_bodyforce`), wall/scheme/grid sensitivity, cross-arm agreement |
| `run.log` | same | full stdout of that run |
| `cli_guo_ny10.json` | `python -m tensorlbm.nebb_bodyforce --ny 10 --scheme guo` | package CLI artifact |
| `fig_profile.png`, `fig_cross_arm.png`, `fig_convergence.png`, `fig_field.png` | harness | profile vs analytic, arm agreement, grid convergence, field |
| `corepath/result.json` | `run_corepath_l3.py` | **L3**: the same wall through `GPUCorePathSolver` (`wall_bc='nebb'`) vs the half-way control, each against its own analytic reference |
| `corepath/nebb_bodyforce_xval.png` | same | profile comparison figure |
| `evidence.json` | `benchmarks/xval/nebb_bodyforce_evidence.py --strict` | machine-checkable bundle (impl/entry/artifacts/callers/benchmark/repro_cmd); `ok=true` only if every check passes |

Reproduce: see `repro_cmd` in `evidence.json` (also rendered in `../REPORT_R5.md` §5).
