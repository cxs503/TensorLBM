# Cross-Validation Method Selection: XVAL-LBM-BITWISE-REFDATA-REGRESSION-PLUS-BENCHMAR

## Feature Slug
`XVAL-LBM-BITWISE-REFDATA-REGRESSION-PLUS-BENCHMAR`

## Decision
Implement a unified cross-validation framework for TensorLBM that provides:
1. **Bitwise reproducibility** verification via SHA-256 state hashing
2. **Reference-data regression** testing via golden .npz snapshots
3. **Benchmarking** infrastructure with statistical timing

## Alternatives Considered

| Alternative | Pros | Cons | Verdict |
|---|---|---|---|
| **A: SHA-256 state hashing** | Collision-resistant; deterministic; no external deps | Sensitive to dtype/device changes (by design) | **Selected** |
| B: CRC32 checksum | Faster | Collision-prone for large arrays | Rejected |
| C: Element-wise comparison with tolerance | More flexible | Not bitwise; tolerance choice is arbitrary | Rejected (use for error metrics, not repro) |
| **A: .npz compressed storage** | Portable; numpy-native; version-controllable | Binary format | **Selected** |
| B: HDF5 storage | Supports chunking/parallel I/O | Heavy dependency (h5py) | Rejected (overkill) |
| C: JSON serialization | Human-readable | Inefficient for arrays; precision loss | Rejected |
| **A: pytest-based test runner** | Standard; CI-friendly; parametrize | Limited to pass/fail | **Selected** |
| B: Custom HTML report generator | Rich output | Extra maintenance; not CI-native | Rejected (future extension) |

## Method Selection Rationale

### Bitwise Reproducibility
- **Reference**: Krüger et al. (2017) "The Lattice Boltzmann Method", §4.4 — deterministic collision/streaming operators produce identical results given identical initial conditions.
- **Implementation**: SHA-256 hash of the raw byte representation of the distribution function array `f`. Two runs with identical seeds and configuration must produce identical hashes.
- **Key insight**: LBM solvers are inherently deterministic (no stochastic terms in BGK collision). Non-reproducibility indicates a bug (e.g., uninitialized RNG, race condition in parallel code).

### Reference-Data Regression
- **Reference**: Standard practice in production CFD codes (OpenFOAM `postProcess -func residuals`, SU2 regression tests, Nektar++ verification).
- **Implementation**: Store golden reference data as compressed `.npz` files alongside source code. Each case has a manifest entry with SHA-256, scalar metrics, and acceptance thresholds.
- **Key insight**: Reference data must be version-controlled so that regressions are detected by CI, not by manual inspection.

### Benchmarking
- **Reference**: ASME V&V 20-2009 §5.3 (iterative convergence); OpenFOAM benchmark methodology (warm-up + N repetitions, report mean/std/min/max).
- **Implementation**: `BenchmarkConfig` with configurable warm-up and repetition count; `BenchmarkResult` with MLUPS calculation.
- **Key insight**: Warm-up eliminates JIT compilation and cache effects; multiple repetitions give statistical confidence.

### Verification Cases
1. **2-D Poiseuille flow** (analytical): u(y) = g/(2ν) y(H-y). Tests Guo forcing + half-way bounce-back. Acceptance: ε∞ < 2%.
2. **Lid-driven cavity Re=100** (experimental): Ghia et al. (1982). Tests Zou-He BC + vortex structure. Acceptance: L₂ < 15%.
3. **Womersley pulsatile α=5** (analytical): Womersley (1955). Tests time-dependent forcing + phase averaging. Acceptance: amplitude/phase < 15%.

## External Anchors
- Guo, Z. et al. (2002). "Discrete lattice effects on the forcing term in the lattice Boltzmann method." Phys. Rev. E 65, 046308.
- Ghia, U., Ghia, K.N., & Shin, C.T. (1982). "High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method." J. Comp. Phys. 48(3), 387-411.
- Womersley, J.R. (1955). "Method for the calculation of velocity, rate of flow and viscous drag in arteries when the pressure gradient is known." J. Physiol. 127:553.
- Krüger, T. et al. (2017). *The Lattice Boltzmann Method*. Springer.
- ASME V&V 20-2009. "Standard for Verification and Validation in Computational Fluid Dynamics and Heat Transfer."

## Lessons from Previous Attempt (m17505)
- Previous attempt honest-FAIL'd with no recorded fail_stage or failure_reason.
- Root cause: likely attempted too much without incremental verification.
- Mitigation: this round uses incremental development with pytest verification at each step.

## Migration Path
The framework is implemented as `tensorlbm.verification.xval` with no changes to existing solver code. Cases are registered via `XValRegistry` and can be extended without modifying the framework.
