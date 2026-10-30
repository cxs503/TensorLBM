# SDAA Sweep: droplet_oscillation Benchmark

## Overview

This document describes the SDAA validation sweep for the `droplet_oscillation` benchmark from TensorLBM's verified benchmark suite.

## Benchmark Description

The droplet_oscillation benchmark measures the capillary oscillation frequency of a 2D droplet using the Shan-Chen single-component pseudopotential LBM model. The benchmark validates:

- **Physical model**: SC94 single-component pseudopotential LBM (D2Q9)
- **Observable**: Raw damped frequency ω_d of the m=2 capillary mode
- **Theory**: Rayleigh prediction ω_th = √(6σ/((ρ_l+ρ_v)R³))
- **Validation**: Compare observed ω_d against theory at multiple radii

## Validation Criteria (Pre-registered)

The benchmark has strict pre-registered criteria:

1. **|err| ≤ 3% at every level** (basis C primary)
   - err = (ω_d - ω_th) / ω_th × 100%
   - Basis C uses R-matched static surface tension and densities

2. **|err| strictly decreasing with R** (monotone convergence)
   - Demonstrates grid convergence

3. **Each level simulated ≥ 4 theory periods**
   - Ensures sufficient sampling for frequency extraction

4. **Fit sigma_omega/omega_d < 1%**
   - Statistical uncertainty of the fit

5. **Mass drift < 2e-3**
   - Conservation check

## Configuration

### Original Benchmark (NVIDIA RTX 5090)
- **Radii**: [128, 160, 224]
- **Domain**: L = 4R (periodic)
- **Runtime**: ~1 hour
- **Device**: CUDA (RTX 5090)

### SDAA Sweep Configuration
- **Radii**: [64, 80, 96] (reduced for SDAA sweep budget)
- **Domain**: L = 4R (periodic)
- **Runtime**: ~10-20 minutes (estimated)
- **Device**: SDAA (/dev/tcaicard1)

**Note**: Smaller radii are used to fit within the 20-minute SDAA sweep budget. The physics remains the same; only the grid resolution is reduced.

## Model Details

### Lattice Boltzmann Model
- **Collision**: SC94 single-component pseudopotential BGK
- **Stream**: Periodic pull (tensorlbm.solver.stream)
- **Equation of state**: ψ = 1 - exp(-ρ)
- **Interaction strength**: G_eff = -5.0 (physical), G_lib = +5.0 (library sign convention)
- **Relaxation time**: τ = 1.0
- **Precision**: float32

### Initial Conditions
- **Droplet shape**: Elliptic tanh profile
- **Radius**: R(θ) = R₀(1 + 0.05·cos(2θ))
- **Interface width**: W = 4
- **Perturbation**: ε = 0.05 (m=2 mode)
- **Initial densities**: ρ_l = 1.957, ρ_v = 0.1596

### Measurement Protocol
- **Primary observable**: RxRy(t) = R_x - R_y (interface extent difference)
- **Window**: Skip 500 steps + 2.0 theory periods
- **Sampling**: Every 50 steps
- **Fit**: Damped sine LSQ (nested grid zoom + profile likelihood)

### Theory Basis C (Primary)
- **Surface tension**: σ_i(R) = dp × R_eq_static (from static Laplace run)
- **Densities**: ρ_in(R), ρ_out(R) from band means in static run
- **Frequency**: ω_th = √(6σ_i / ((ρ_in + ρ_out) × R_eq³))

## Device Evidence

```
Device type: sdaa
Device count: 1
Device name: /dev/tcaicard1
```

## Results

### Status: IN PROGRESS

The SDAA sweep is currently running. Full results will be updated upon completion.

### Expected Results Structure

The results JSON (`artifacts/sdaa_droplet_oscillation.json`) contains:

```json
{
  "benchmark": "droplet_oscillation",
  "device_evidence": {
    "device_type": "sdaa",
    "device_count": 1,
    "device_name": "/dev/tcaicard1"
  },
  "device_used": "sdaa:0",
  "radii": [64, 80, 96],
  "total_wall_time_s": <float>,
  "results": {
    "R64": {
      "R_init": 64,
      "L": 256,
      "wall_time_s": <float>,
      "static": {
        "sigma_i": <float>,
        "rho_in_static": <float>,
        "rho_out_static": <float>,
        "converged": <bool>,
        ...
      },
      "oscillation": {
        "omega_d": <float>,
        "omega_theory_C": <float>,
        "err_pct_primary": <float>,
        "sigma_omega_rel_pct": <float>,
        "mass_drift": <float>,
        "T_theory_C": <float>,
        "max_steps": <int>,
        ...
      }
    },
    ...
  },
  "criteria": {
    "err_le_3pct_all_levels": <bool>,
    "err_monotone_decreasing": <bool>,
    "simulated_ge_4_periods_all_levels": <bool>,
    "sigma_omega_rel_lt_1pct_all_levels": <bool>,
    "mass_drift_lt_2e3_all_levels": <bool>
  },
  "all_criteria_pass": <bool>
}
```

## Scripts

### Main Script
- **Path**: `scripts/sdaa_sweep_droplet_oscillation.py`
- **Purpose**: Full SDAA sweep with radii [64, 80, 96]
- **Usage**: `python3 scripts/sdaa_sweep_droplet_oscillation.py`

### Minimal Proof-of-Concept
- **Path**: `scripts/sdaa_sweep_droplet_oscillation_minimal.py`
- **Purpose**: Quick verification with single radius R=64
- **Usage**: `python3 scripts/sdaa_sweep_droplet_oscillation_minimal.py`

## Notes

1. **No Triton dependency**: The benchmark uses pure PyTorch operations, no TRITON_INTERPRET workaround needed.

2. **Small radii limitation**: Radii < 64 (e.g., R=32) cause NaN issues due to droplet size being too small for the model.

3. **Runtime scaling**: Runtime scales as O(R² × steps) ≈ O(R³) due to grid size and number of steps both scaling with R.

4. **Criteria validation**: All criteria are evaluated programmatically from the simulation output, no manual inspection.

5. **Honest reporting**: If any criterion fails, it will be reported as FAIL with the specific reason. No criteria relaxation or CPU result substitution.

## Comparison with Original Benchmark

| Aspect | Original (CUDA) | SDAA Sweep |
|--------|----------------|------------|
| Radii | [128, 160, 224] | [64, 80, 96] |
| Runtime | ~1 hour | ~10-20 min |
| Device | RTX 5090 | /dev/tcaicard1 |
| Criteria | Same | Same |
| Model | Same | Same |

The reduced radii maintain the same physics but at lower resolution. The criteria thresholds remain unchanged.

## Next Steps

1. Complete the SDAA sweep run
2. Validate all criteria
3. Compare results with original CUDA benchmark
4. Document any discrepancies or device-specific observations

## References

- Original benchmark: `benchmarks/verified/droplet_oscillation/`
- README: `benchmarks/verified/droplet_oscillation/README.md`
- Results: `benchmarks/verified/droplet_oscillation/result.json`
- Notes: `benchmarks/verified/droplet_oscillation/NOTES.md`
