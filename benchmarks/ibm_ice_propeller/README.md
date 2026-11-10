# IBM ice–propeller coupling benchmark (`P233W2-N02`)

Driver + artifacts for the immersed-boundary validation of the ice/propeller
node. The solver lives in `src/tensorlbm/ibm_moving_body.py`; the physics
discussion, pitfalls and the criteria table are in
`docs/P233W2_N02_IBM_ICE_PROPELLER.md`.

## What is validated

| criterion | what it measures | artifact |
|---|---|---|
| AC1 | 17 pytest cases: kernel algebra, rigid kinematics, static/moving, plane/curved, single/multiple bodies, momentum audits | `tests/test_ibm_moving_body.py` |
| AC2 | fixed & rotating cylinder loads at `Re = 100` against literature (`<= 10 %`) | `cylinder_*.json` |
| AC3 | ice-floe / rotating-blade load time history + hydrodynamic baseline | `ice_blade*.json` |
| AC4 | momentum exchange `<= 1e-6` relative (prescribed + free body) | `audit.json` |
| AC5 | three-grid ladder, error decreasing monotonically | `cylinder_*.json` |
| AC6 | figures + `verdicts.json` (velocity/vorticity, convergence, `Cd(Re)`, load series) | `figures/ibm_ice_propeller/` |

## Commands

```bash
# unit tests
PYTHONPATH=src python -m pytest tests/test_ibm_moving_body.py -q

# momentum audits
PYTHONPATH=src python benchmarks/ibm_ice_propeller/run_validation.py \
    audit --d-cells 10 --n-steps 400 --out artifacts/ibm_ice_propeller/audit.json

# one cylinder case (grid ladder: repeat with --d-cells 16/24/32)
PYTHONPATH=src python benchmarks/ibm_ice_propeller/run_validation.py \
    cylinder --d-cells 24 --n-steps 30000 --warmup 12000 --u-in 0.1 \
    --label fixed_d24 --out artifacts/ibm_ice_propeller/cylinder_fixed_d24.json

# rotating cylinder (spin ratio alpha = omega R / U)
PYTHONPATH=src python benchmarks/ibm_ice_propeller/run_validation.py \
    cylinder --d-cells 32 --alpha 0.5 --n-steps 40000 --warmup 15000 --u-in 0.1 \
    --label rotating_d32 --out artifacts/ibm_ice_propeller/cylinder_rotating_d32.json

# ice floe vs rotating blade
PYTHONPATH=src python benchmarks/ibm_ice_propeller/run_validation.py \
    ice --n-steps 9000 --out artifacts/ibm_ice_propeller/ice_blade.json

# criteria table + figures + report body
PYTHONPATH=src python benchmarks/ibm_ice_propeller/run_validation.py \
    aggregate --indir artifacts/ibm_ice_propeller --out artifacts/ibm_ice_propeller/verdicts.json
PYTHONPATH=src python benchmarks/ibm_ice_propeller/run_validation.py \
    figures --indir artifacts/ibm_ice_propeller --outdir figures/ibm_ice_propeller
python benchmarks/ibm_ice_propeller/make_report.py \
    --artifacts artifacts/ibm_ice_propeller --figures figures/ibm_ice_propeller
```

All quantities are lattice units (`dx = dt = rho = 1`).

## Notes

* `--u-in 0.1` is used for the cylinder runs: at `Re = 100` it puts the
  relaxation time at `tau = 0.55 … 0.60`, where the TRT collision is stable
  (`tau = 0.53` with `u_in = 0.05` diverges with BGK, and is marginal even with
  TRT).
* The `ice` task uses the semi-implicit added-mass coupling because an ice floe
  (`rho_ice/rho_water = 0.895`) is lighter than its 2-D added mass. Use
  `--explicit-coupling --floe-density 3.0` to reproduce the exact momentum audit
  variant.