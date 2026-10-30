# Free-Surface Independent Reference Evaluation

## Purpose

Establish **independent** error metrics for TensorLBM free-surface / multiphase
cases against literature references that are **not** sourced from this
repository's own code.

This addresses QR-P0: "TensorLBM 多相/自由面算例需要独立误差指标，自洽测试不算独立验证。"

## Independent References (台账)

| # | Reference | Observable | Source code |
|---|-----------|-----------|-------------|
| 1 | Martin & Moyce (1952) Phil Trans R Soc Lond A 244:312-324 | Dam-break surge front Z(T), α=2 | Fig.5 |
| 2 | Koshizuka & Oka (1996) Nucl Sci Eng 123:421-434 | Dam-break front X(T) | Fig.4 |
| 3 | Clanet et al. (2004) J Fluid Mech 517:199-208 | Max spreading β_max=0.5·We^(1/4) | Fig.8 |
| 4 | Inertial spreading law (dimensional analysis) | β(t)~We^(1/4)·(t/t_c)^(1/2) | — |

Reference data files live in `reference_data/` as CSV — they are **not** imported
from `src/tensorlbm/dam_break_3d.py` (which contains `_MARTIN_MOYCE_FRONT` and
`_KOSHIZUKA_FRONT`).

## Structure

```
verified/free_surface_independent_eval/
├── reference_data/
│   ├── martin_moyce_1952_front.csv      # Ref 1
│   ├── koshizuka_oka_1996_front.csv     # Ref 2
│   └── droplet_spreading_references.csv # Ref 3
├── eval_dam_break.py        # Run dam-break at 2 mesh levels, compare to Refs 1&2
├── eval_droplet_spreading.py # Run droplet spreading at 2 mesh levels, compare to Refs 3&4
├── gate.py                  # Gate script (exit 0=pass, 1=fail)
├── results_dam_break/       # Output: error_table.json + figures/
├── results_droplet/         # Output: error_table.json + figures/
└── gate_result.json         # Gate verdict
```

## Usage

```bash
cd /root/tp-agent/work/m506
PYTHONPATH=src python verified/free_surface_independent_eval/eval_dam_break.py
PYTHONPATH=src python verified/free_surface_independent_eval/eval_droplet_spreading.py
python verified/free_surface_independent_eval/gate.py
echo $?  # 0=pass, 1=fail
```

## Gate thresholds (FIXED, never adjusted)

- **Hard threshold**: 15% — gate fails if any error exceeds this
- **Target**: 3% — aspirational, reported but not gated

## Honest-FAIL discipline

The gate script will exit 1 if errors exceed 15%. This is by design.
Thresholds are not adjusted to make the gate pass.
