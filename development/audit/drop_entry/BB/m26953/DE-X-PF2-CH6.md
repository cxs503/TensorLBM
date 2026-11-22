# DE-X-PF2-CH6: Lee-Lin Consistent CH6 Phase Field

## Summary
Implemented Lee-Lin consistent Cahn-Hilliard (CH6) phase field model using D3Q19-isotropic operators for both chemical potential and Korteweg force. The CH6 model uses isotropic Laplacian and gradient operators from the D3Q19 stencil (6 axis neighbors at weight 1/3, 12 face-diagonal neighbors at weight 1/6) to ensure pressure and flux are discretized at the same order.

## Key Changes
1. **ch6.py**: New module with `CH6FreeEnergy` class using isotropic operators, `ch6_equilibrium()` function, and `ch6_step_3d()` collision operator
2. **ch6_validation.py**: Validation tests for planar interface, static droplet, and linear CH mode
3. **isotropic_fd.py**: Fixed sign bugs in `isotropic_gradient_3d()` and `_diag_neighbor()` functions
4. **multiphase3d.py**: Added `geq_convention='ch6'` support to `free_energy_step_3d()`
5. **phasefield/__init__.py**: Exported CH6 symbols

## Validation Results
- **Planar interface**: ✓ PASS (2.84% σ error, comparable to CH4's 2.42%)
- **Static droplet**: ✗ FAIL (high spurious currents ~0.5, expected < 1e-3)
- **Linear CH mode**: ✗ FAIL (6.72% rate error, expected < 5%)

## Known Issues
1. Static droplet shows significantly higher spurious currents than CH4
2. Linear CH mode rate error exceeds 5% tolerance
3. Only planar interface validation passes

## Files
- `src/tensorlbm/phasefield/ch6.py`
- `src/tensorlbm/phasefield/ch6_validation.py`
- `src/tensorlbm/phasefield/isotropic_fd.py` (bug fixes)
- `src/tensorlbm/multiphase3d.py` (ch6 support)
- `docs/DE-X-PF2-CH6.md`
