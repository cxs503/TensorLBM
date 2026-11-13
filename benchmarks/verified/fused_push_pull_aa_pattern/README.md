# Verified: fused push-pull AA pattern (anchor `XVAL-LBM-FUSED-PUSH-PULL-AA-PATTERN`)

Reference practice: ``lbm-principles-practice/code`` chapter 13 GPU kernel
``gpu_stream_collide_save`` (MIT).  It fuses streaming and collision into one
pass: pull the eight shifted neighbours of the *post-collision* field, rebuild
the moments from those nine numbers, relax, write the next buffer.  No array
ever holds "streamed but not yet collided" populations.

TensorLBM side: :func:`tensorlbm.solver.step_stream_collide` with
``kernel="fused_push_pull"`` (step-level kernel in
:mod:`tensorlbm.fused_push_pull`), against the textbook AB reference
``kernel="ab"`` = ``stream(collide_bgk(...))``.

## Files

| file | role |
|---|---|
| ``run.py`` | runs both kernels on the Taylor-Green ladder + controls, writes ``result.json`` and ``figures/`` |
| ``verify.py`` | machine referee over ``result.json``; exit 0 = pass; locked constants in ``LOCKED`` |
| ``register_case.sql`` | idempotent referee row for ``tensor_benchmark_cases`` |
| ``make_doc.py`` | renders ``docs/XVAL_m17902_FUSED_PUSH_PULL_AA_PATTERN_INTEGRATION.md`` from ``result.json`` (zero hand-copied numbers) |

## Run

```bash
python benchmarks/verified/fused_push_pull_aa_pattern/run.py           # ~3 min, 3 grids, 2 kernels, 2 controls
python benchmarks/verified/fused_push_pull_aa_pattern/run.py --quick    # smoke, ~3 s
python benchmarks/verified/fused_push_pull_aa_pattern/verify.py         # exit 0 = pass
```

## What the referee pins

1. **Physics**: both kernels reproduce the closed-form Taylor-Green decay, with
   ``u_x``/``u_y`` relative L2 far inside the 3% target (owner gate 15%).
2. **Scheme equivalence**: the fused kernel collides *after* streaming, the AB
   reference *before*, so the states are compared at the same level —
   ``f_max|f_fused − collide_bgk(f_ab)|`` is round-off, and the velocity fields
   agree directly (no label offset).  Reported separately is the raw one-collision
   state offset ``max|f_fused − f_ab|``, which is **not** an error.
3. **Conservation**: relative mass drift over every run ≪ 1e-12.
4. **Convergence**: error falls with grid at the expected second order.
5. **Controls**: two deliberately wrong arms (viscosity +25%, mirrored initial
   field) exceed the 15% gate, so a green referee means something.
