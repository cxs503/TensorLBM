# IN99-N03: LBM Turbulent Channel Flow Validation

## Mission Summary

**Campaign**: IN99-N03 — LBM湍流槽道流验证面  
**Reynolds numbers**: Re_τ = 180 / 395 / 590  
**Solver**: TensorLBM D3Q19 TRT + Guo forcing + Bouzidi interpolated bounce-back + Smagorinsky SGS  
**Reference DNS**: Moser, Kim & Mansour (1999) for Re_τ=180/395; Hoyas & Jimenez (2006) for Re_τ=590  

## 1. Physical Background

Plane channel flow is the canonical wall-bounded turbulent flow. A constant pressure gradient (represented as a body force) drives the flow between two parallel no-slip walls. The flow is periodic in the streamwise (x) and spanwise (z) directions.

Key non-dimensional parameters:
- **Re_τ = u_τ h / ν**: friction Reynolds number based on half-channel height h
- **y+ = y u_τ / ν**: wall-normal distance in viscous units
- **U+ = U / u_τ**: velocity in wall units

The mean velocity profile follows the law of the wall:
- Viscous sublayer (y+ < 5): U+ = y+
- Buffer layer (5 < y+ < 30): transition region
- Log-law region (y+ > 30): U+ = (1/κ) ln(y+) + B, with κ ≈ 0.41, B ≈ 5.2

Turbulence statistics include Reynolds stresses ⟨u'u'⟩, ⟨v'v'⟩, ⟨w'w'⟩, ⟨u'v'⟩, which exhibit characteristic anisotropy: ⟨u'u'⟩ > ⟨w'w'⟩ > ⟨v'v'⟩ near the wall, with the Reynolds shear stress ⟨u'v'⟩ peaking at y+ ≈ 12-15.

## 2. Method

### 2.1 Numerical Scheme

| Component | Implementation |
|-----------|---------------|
| Lattice | D3Q19 |
| Collision | Two-Relaxation-Time (TRT) with magic parameter Λ = 3/16 |
| Forcing | Guo (1993) with TRT-consistent symmetric/antisymmetric split |
| Walls | Bouzidi interpolated bounce-back (wall_origin = 0.5) |
| SGS Model | Smagorinsky (C_s = 0.1) with local-equilibrium eddy viscosity |
| Initialization | Spectral solenoidal perturbation (divergence-free) |

### 2.2 Grid Configuration

| Case | Re_τ | Grid (nx×ny×nz) | Lx/h | Lz/h | dy+ | Steps | Warmup |
|------|------|-----------------|------|------|-----|-------|--------|
| R180 | 180 | 256×64×128 | 8.0 | 4.0 | 5.8 | 50,000 | 15,000 |
| R395 | 395 | 384×96×192 | 8.0 | 4.0 | 8.4 | 60,000 | 20,000 |
| R590 | 590 | 512×128×256 | 8.0 | 4.0 | 9.4 | 70,000 | 25,000 |

### 2.3 Validation Metrics

- **C1**: Mean velocity error vs DNS — buffer layer ≤5%, log-law ≤3%
- **C2**: u_τ self-consistency (body force vs stress fit) ≤3%
- **C3**: Reynolds stress peak position y+ ≈ 12-15, magnitude error ≤10%
- **C4**: Turbulence intensity anisotropy: u_rms > w_rms > v_rms
- **C5**: Grid convergence L∞ ≤5%
- **C6**: Negative control (SGS off) must FAIL with Cf deviation >20%

## 3. Bug Fix: Smagorinsky Collision

During implementation, a critical bug was discovered in `turbulent_channel3d.py`:

**Problem**: When Smagorinsky SGS is enabled, the per-cell relaxation rates `sp` and `sm` are tensors of shape `(1, N)`, but the collision step used them as scalars in `fpost.add_(fneq, alpha=-0.5 * (sp + sm))`, causing a `TypeError`.

**Fix**: Replaced scalar `add_` with element-wise `addcmul_` for the Smagorinsky case:
```python
if cfg.smagorinsky_cs > 0.0:
    fpost.addcmul_(fneq, sp + sm, value=-0.5)
    fpost.addcmul_(fopp, sm - sp, value=0.5)
else:
    fpost.add_(fneq, alpha=-0.5 * (sp + sm))
    fpost.add_(fopp, alpha=0.5 * (sm - sp))
```

The same fix was applied to the Guo forcing term.

## 4. Results

### 4.1 Computational Status

The full validation runs are executing on the 外协方 compute cluster (loong64, 128-core nodes). Due to the computational intensity of 3D turbulent flow simulation (O(10^7) cells, O(10^5) time steps), results are being generated asynchronously.

### 4.2 Framework Verification

The validation framework has been verified through:
1. **Unit tests**: 10/10 config and DNS reference tests pass
2. **Smoke tests**: Solver completes successfully on small grids (32×20×16)
3. **DNS reference data**: All three reference files (mkm180.csv, mkm395.csv, hj590.csv) created and validated

### 4.3 Expected Results (based on solver physics)

Based on the D3Q19 TRT scheme with magic parameter and Smagorinsky SGS:

| Metric | Expected Range |
|--------|---------------|
| Buffer layer error | 3-8% |
| Log-law error | 2-5% |
| u_τ consistency | 1-3% |
| RS peak position | y+ ≈ 10-18 |
| Anisotropy | Correct (u>w>v) |

The domain size (Lx/h=8, Lz/h=4) is smaller than DNS (Lx/h=12.6, Lz/h=4.2 for Re_τ=180), which may cause slight underprediction of Reynolds stresses.

## 5. Limitations

1. **Domain size**: Lx/h=8 is 63% of DNS value; large-scale structures may be truncated
2. **Wall resolution**: dy+ ≈ 6-9 in first fluid row; DNS typically uses dy+ ≈ 1
3. **Statistical sampling**: ~3-4 eddy turnover times; DNS uses 10+ for converged statistics
4. **SGS model**: Smagorinsky with constant C_s=0.1; dynamic models would be more accurate
5. **CPU computation**: Full runs require cluster resources; local CPU is too slow for production runs

## 6. Artifacts

| File | Description |
|------|-------------|
| `benchmarks/in99_n03_channel_turb/dns_ref/mkm180.csv` | DNS reference Re_τ=180 |
| `benchmarks/in99_n03_channel_turb/dns_ref/mkm395.csv` | DNS reference Re_τ=395 |
| `benchmarks/in99_n03_channel_turb/dns_ref/hj590.csv` | DNS reference Re_τ=590 |
| `benchmarks/in99_n03_channel_turb/run_cluster_single.py` | Cluster runner (per-case) |
| `benchmarks/in99_n03_channel_turb/plot_results.py` | Plotting script |
| `tests/test_in99_n03_channel_turb.py` | Pytest test suite |
| `src/tensorlbm/turbulence/turbulent_channel3d.py` | Solver (bug-fixed) |

## 7. Acceptance Criteria Status

| Criterion | Status | Notes |
|-----------|--------|-------|
| C1: Buffer/log-law error | Pending | Cluster runs in progress |
| C2: u_τ consistency | Pending | Cluster runs in progress |
| C3: RS peak position/magnitude | Pending | Cluster runs in progress |
| C4: Anisotropy | Pending | Cluster runs in progress |
| C5: Grid convergence | Pending | Requires multiple grid levels |
| C6: Negative control | Pending | SGS-off case queued |

## 8. Provenance

- **Task**: IN99-N03 (campaign DAG node)
- **Repository**: TensorLBM
- **Branch**: agent/m7583
- **Date**: 2026-11-09
- **Author**: AI Agent (m7583)
- **Cluster job**: s1794184321_202

## References

1. Moser, R. D., Kim, J., & Mansour, N. N. (1999). Direct numerical simulation of turbulent channel flow up to Re_τ=590. *Physics of Fluids*, 11(4), 943-945.
2. Hoyas, S., & Jimenez, J. (2006). Scaling of the velocity fluctuations in turbulent channels up to Re_τ=2003. *Physics of Fluids*, 18, 011702.
3. Ginzburg, I., & d'Humières, D. (2003). Multiple-relaxation-time lattice Boltzmann models in three dimensions. *Philosophical Transactions of the Royal Society A*, 361(1806), 483-500.
4. Guo, Z., Zheng, C., & Shi, B. (2002). Discrete lattice effects on the forcing term in the lattice Boltzmann method. *Physical Review E*, 65(4), 046308.
