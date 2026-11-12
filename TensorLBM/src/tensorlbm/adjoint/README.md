# Adjoint / Differentiable LBM Module

## Overview

The `tensorlbm.adjoint` module provides gradient computation for the D2Q9 BGK
Lattice Boltzmann Method via two complementary approaches:

| Method | Memory | Speed | Use Case |
|--------|--------|-------|----------|
| **BPTT** (full storage) | O(T·N·Q) | Fast | Reference / small problems |
| **OOTC** (checkpointed) | O(k·N·Q) | Slower (recompute) | Large problems, memory-constrained |

Where T = time steps, N = grid points, Q = 9 (D2Q9 velocities), k = checkpoint interval.

## Theory

### Forward Model

The D2Q9 BGK LBM evolves distribution functions f_i(x,t) via:

```
f_i^{n+1}(x) = f_i^n(x - e_i) - (1/τ)[f_i^n(x - e_i) - f_i^{eq}(x - e_i)]
```

with periodic boundaries. The equilibrium distribution is the standard
second-order expansion:

```
f_i^{eq} = w_i ρ [1 + 3(e_i·u) + 9/2(e_i·u)² - 3/2|u|²]
```

### Objective Function

```
J(τ) = ½ ||u_T(τ) - u_ref||²
```

### BPTT (Backpropagation Through Time)

The adjoint (costate) λ propagates backward:

```
λ^n_j(y) = μ_j(y) - (1/τ)[μ_j(y) - (Bμ)_j(y)]
```

where μ is the anti-streamed costate and B = df^eq/df is the collision Jacobian.

The parameter gradient is:

```
dJ/dτ = Σ_n Σ_y (1/τ²)(f_i^n(y) - f_i^{eq,n}(y)) μ_i(y)
```

### OOTC (Out-of-Core / Checkpointed)

Instead of storing all T forward states, OOTC stores checkpoints every k steps
on disk. During the backward pass, each segment is recomputed from its checkpoint.
Because the LBM is deterministic, recomputation is bit-identical to the original
forward run, so the OOTC gradient equals the BPTT reference up to floating-point
summation order.

**Memory savings**: For T=10,000, N=128², k=100:
- BPTT: ~55 GiB (impractical)
- OOTC: ~0.25 GiB (feasible)

## API

```python
from tensorlbm.adjoint import (
    initial_pulse,    # Create initial condition
    step,             # One LBM step
    macro,            # Extract macroscopic fields
    bptt_gradient,    # Full-storage reference gradient
    ootc_gradient,    # Checkpointed gradient
)

# Create problem
f0 = initial_pulse(ny=128, nx=128, amp=0.05, sigma=16.0)
tau = 1.0
T = 10000

# Get reference terminal velocity
f = f0.copy()
for _ in range(T):
    f = step(f, tau)
_, ux_t, uy_t = macro(f)
ux_ref = ux_t * 0.9  # perturbed reference
uy_ref = uy_t * 0.9

# BPTT reference (stores all states)
dJ_bptt, J, (ux_T, uy_T), states = bptt_gradient(tau, T, f0, ux_ref, uy_ref)

# OOTC (checkpointed, k=100)
dJ_ootc, J, (ux_T, uy_T), n_ckpt, ckpt_path = ootc_gradient(
    tau, T, f0, ux_ref, uy_ref, k=100
)
```

## Validation

The OOTC gradient matches the BPTT reference with relative error < 1e-15
(machine precision) for all tested checkpoint intervals k ∈ [1, T].

See `tests/test_adjoint_ootc.py` for the full test suite and
`benchmarks/adjoint_ootc_bench.py` for performance benchmarks.

## References

- Griewank, A. & Walther, A. (2008). *Evaluating Derivatives: Principles and
  Techniques of Algorithmic Differentiation*, 2nd ed. Springer. Chapter 2.
- Chen, L., Beckman, R. & Patterson, H. (1996). Optimizing checkpoint placement
  for reversible computation.
- Krüger, T. et al. (2017). *The Lattice Boltzmann Method*. Springer. Chapter 2.
