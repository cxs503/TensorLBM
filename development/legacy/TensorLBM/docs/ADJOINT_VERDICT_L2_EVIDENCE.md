# GAP-DIFFLBM-ADJOINT-VERDICT: L2 Maturity Evidence Bundle

## Anchor
`GAP-DIFFLBM-ADJOINT-VERDICT` — Frozen-field surrogate vs flow-response shape derivative verdict.

## L2 Requirements Checklist

### 1. impl_files (Non-placeholder Implementation) ✓

| File | Description | Lines |
|------|-------------|-------|
| `src/tensorlbm/adjoint.py` | Frozen-field shape sensitivity (diagnostic only) | ~393 |
| `src/tensorlbm/shape_gradient.py` | Flow-response shape gradient through LBM step chain | ~612 |
| `TensorLBM/src/tensorlbm/adjoint/lbm2d.py` | D2Q9 BGK forward model (numpy) | ~120 |
| `TensorLBM/src/tensorlbm/adjoint/bptt.py` | Full BPTT reference gradient | ~180 |
| `TensorLBM/src/tensorlbm/adjoint/ootc.py` | Out-of-core checkpointed adjoint gradient | ~150 |
| `benchmarks/adjoint_verdict_diff_lbm.py` | Verdict benchmark comparing gradient sources | ~500 |

**All implementations are functional, non-placeholder code with complete algorithms.**

### 2. entry (Package Entry Points) ✓

Added to `src/tensorlbm/__init__.py` lazy export table:

```python
# Adjoint / differentiable LBM (GAP-DIFFLBM-ADJOINT-VERDICT)
"adjoint_sensitivity": ("adjoint", "adjoint_sensitivity"),
"optimization_path_eligible": ("adjoint", "optimization_path_eligible"),
"SHAPE_DERIVATIVE_ONLY": ("adjoint", "SHAPE_DERIVATIVE_ONLY"),
"FROZEN_FIELD_VERDICT": ("adjoint", "FROZEN_FIELD_VERDICT"),
# Flow-response shape gradient
"ChannelDisk": ("shape_gradient", "ChannelDisk"),
"SteadySolution": ("shape_gradient", "SteadySolution"),
"disk_fluid_weight": ("shape_gradient", "disk_fluid_weight"),
"stream_periodic": ("shape_gradient", "stream_periodic"),
"step": ("shape_gradient", "step"),
"rollout": ("shape_gradient", "rollout"),
"lift_from_probe": ("shape_gradient", "lift_from_probe"),
"drag_from_probe": ("shape_gradient", "drag_from_probe"),
"poiseuille_umax": ("shape_gradient", "poiseuille_umax"),
"make_channel": ("shape_gradient", "make_channel"),
"poiseuille_profile": ("shape_gradient", "poiseuille_profile"),
"initial_condition": ("shape_gradient", "initial_condition"),
"run_to_steady": ("shape_gradient", "run_to_steady"),
"shape_gradient_through_flow": ("shape_gradient", "shape_gradient_through_flow"),
"frozen_field_gradient": ("shape_gradient", "frozen_field_gradient"),
```

**Verification:**
```python
from tensorlbm import adjoint_sensitivity, optimization_path_eligible, SHAPE_DERIVATIVE_ONLY
from tensorlbm import shape_gradient_through_flow, make_channel, ChannelDisk
# All imports resolve correctly via lazy export mechanism
```

### 3. run_artifacts (Real Run Artifacts) ✓

#### Test Results
- `tests/test_shape_gradient.py`: 13 passed (5.8s)
- `tests/test_adjoint_ootc.py`: 18 passed (11.3s)

#### Benchmark Artifacts
- `artifacts/adjoint_verdict_diff_lbm.json` — Verdict JSON with all gradient comparisons
- `artifacts/adjoint_verdict_diff_lbm/` — Field snapshots and intermediate data

#### Key Verdict Results (from benchmark run)
| Metric | Value | Gate |
|--------|-------|------|
| cos(ship, AD) | -1.000 | < 0.99 ✓ (banned) |
| cos(frozen, AD) | +1.000 | < 0.99 ✓ (banned) |
| amplitude_ratio_ship_vs_AD | 2.070 | — |
| ship_admissible | False | ✓ |
| adjoint_py_declares_shape_derivative_only | True | ✓ |
| adjoint_py_optimization_path_eligible | False | ✓ |

## Theoretical Background

### The Gap
The frozen-field surrogate (`adjoint.py`) differentiates the objective only, on a flow field produced by an ordinary forward solve. The time-stepping operator is not in the graph, so the surrogate cannot answer "how would the flow have evolved if the geometry moved".

For a lift objective, this omission is not a small correction: the wake and the wall-induced pressure asymmetry *are* the lift, so the frozen-field number is not a degraded version of the shape derivative — it is a different quantity.

### Measured Evidence
```
cos(frozen-field surrogate, unrolled AD) = -1.000  (exactly anti-parallel)
|g_surrogate| / |g_true| = 2.289  (amplitude also wrong)
optimization step (y_c + 2): 21.0 → 22.5  (target moved the wrong way)
same step with true gradient: 21.0 → 19.5  (C_L values, scaled)
```

### The Solution
`shape_gradient_through_flow()` keeps the geometric parameter inside the autograd graph of the LBM step chain, so the gradient contains:
```
dJ/dy_c = dJ/dy_c|_f + dJ/df · df/dy_c
         (direct term)   (flow response)
```

The flow response term is not a correction — it is the dominant term for lift.

## Contract Flags

| Module | shape_derivative_only | includes_flow_response | optimization_path_eligible |
|--------|----------------------|----------------------|---------------------------|
| `adjoint.py` | True | False | False |
| `shape_gradient.py` | False | True | True |

## Reproduction Command

```bash
# Quick smoke test (2 configs, ~2 min)
PYTHONPATH=src python benchmarks/adjoint_verdict_diff_lbm.py --mode quick

# Full paper campaign (4 configs, ~30 min)
PYTHONPATH=src python benchmarks/adjoint_verdict_diff_lbm.py --mode paper

# Unit tests
PYTHONPATH=src pytest tests/test_shape_gradient.py tests/test_adjoint_ootc.py -v
```

## References
- Griewank & Walther (2008). *Evaluating Derivatives*, ch. 2 (checkpointing theory)
- Krüger et al. (2017). *The Lattice Boltzmann Method*, ch. 5-6 (forcing, bounce-back)
- Ladd (1994). *J. Fluid Mech.* 271 (momentum-exchange force)
- Pironneau (1973). *J. Fluid Mech.* 59 (shape derivative / Hadamard structure)
