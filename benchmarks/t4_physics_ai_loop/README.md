# T4 physics-AI flow-map benchmark

The drivers produce LBM field sequences, train a Fourier neural operator
flow map on the training Reynolds numbers, and compare predictions and
feedback on held-out Reynolds numbers. They load
`tensorlbm.ai.flow_field_loop` from `src/` in this checkout.

From the repository root:

```bash
python benchmarks/t4_physics_ai_loop/bench_t4_medium.py --threads 16
python benchmarks/t4_physics_ai_loop/bench_t4_physics_ai_loop.py --threads 16
```

Both accept `--out` and `--stage`. Use a completed production dataset for the
training-only stage. The [historical campaign report](../../docs/T4_PHYSICS_AI_LOOP.md)
and [residual flow-map follow-up](../../docs/T4_FLOW_MAP_RESIDUAL_m18630.md)
record the criteria, known failures, and the limits of the prior results.
