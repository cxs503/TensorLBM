# Agent instructions

Before making changes, read the [Tensor ecosystem vision](https://github.com/cxs503/TensorPlatform/blob/main/docs/architecture/TENSOR_ECOSYSTEM_AI_ERA_VISION.md) and this repository's [project charter](docs/PROJECT_CHARTER.md). These documents define ownership boundaries; the source code and tests define what is actually implemented.

## Required development behavior

- State the goal, why it belongs in this repository, and which public or cross-repository contracts it affects.
- Preserve the repository boundary: TensorLBM owns lattice-Boltzmann numerical methods and solver-local verification. Read docs/PROJECT_CHARTER.md and the ecosystem vision. Keep global research orchestration in TensorPhysics and generic agent infrastructure in TensorPlatform. Provide reproducible cases, quantitative diagnostics, supported-use limits, and reference/grid/time-step evidence; do not infer correctness from visual plausibility.
- Inspect relevant code, tests, recent changes, and existing benchmark records before editing.
- Prefer stable, versioned public interfaces; avoid private cross-repository imports and duplicated sources of truth.
- Include runnable tests/examples and quantitative evidence appropriate to the change. Report exact commands, results, failures, untested paths, assumptions, and limitations.
- Distinguish implementation, unit tests, integration tests, benchmarks, domain validation, and production deployment evidence. Never claim more than the evidence establishes.
- Keep changes scoped, reviewable, and reversible. Do not optimize for PR count, code volume, or workflow completion alone.
