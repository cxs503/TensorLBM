# SDAA dtype strategy probe (m306)

This developer comparison used Couette flow and Taylor–Green decay to compare
fp32, tf32, and bf16 on the recorded SDAA device. The original selection gate
required at most 0.1% error and at least 1.3× speedup against fp32. See the
[original JSON](dtype_probe.json) for the device, recorded metrics, and verdict.

- [Error comparison](dtype_error_comparison.png)
- [Speed comparison](dtype_speedup_comparison.png)
- [Couette profiles](couette_profile_comparison.png)
- [Taylor–Green energy decay](tg_energy_decay.png)

The existing checkout command is `python scripts/dtype_probe.py --device sdaa:0`.
New output defaults to `outputs/device-dtype/`; `--out` and `--plot-dir` still
take explicit paths. The [migration record](../../../../docs/architecture/migrations/device-dtype-m306-2026-11-22.json)
pins the original Git files and hashes. These are historical device comparison
results for the stated configurations.
