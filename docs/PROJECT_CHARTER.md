# TensorLBM — lattice Boltzmann solver charter

**Status:** repository responsibility and agent-development contract  
**Audience:** maintainers, contributors, autonomous coding agents  
**Canonical ecosystem vision:** [Tensor Ecosystem: Vision for the AI Era](https://github.com/cxs503/TensorPlatform/blob/main/docs/architecture/TENSOR_ECOSYSTEM_AI_ERA_VISION.md)

## Mission

Provide independently usable, testable lattice-Boltzmann numerical methods and fluid/multiphysics solver capabilities for the Tensor ecosystem.

## This repository owns

- LBM lattice, collision/streaming, boundary conditions, turbulence and supported multiphysics kernels
- Solver-local APIs, diagnostics, numerical stability limits, and method-specific verification
- Reproducible solver examples and quantitative benchmark evidence
- Solver outputs and metadata needed for downstream orchestration

## Explicit non-goals

- Global agent planning or cross-repository research orchestration
- A competing general-purpose agent platform or domain-neutral infrastructure layer
- Claiming engineering validity from visually plausible contours or software tests alone

## Integration contract

Remain independently installable and usable. Expose documented, parameterized solver entry points and structured diagnostics/artifacts for TensorPhysics workflows; use TensorPlatform only through shared platform contracts. Do not move specialist solver logic into TensorPlatform.

## Quality and evidence gates

For each advertised capability, provide a runnable case, numerical invariants where applicable, grid/time-step sensitivity, comparison to analytical/reference/experimental data when available, and explicit limits. Label unvalidated prototypes honestly.

## Agent instructions

Before implementing a substantial change, an agent must:

1. Read this charter, the README, the relevant code/tests, and recent PRs.
2. Explain why the change belongs in this repository and identify affected cross-repository contracts.
3. Prefer a stable, versioned public interface over private imports or copied logic.
4. Include executable tests and a reproducible example or benchmark appropriate to the change.
5. Report exactly what was run, what passed/failed, and what was not tested.
6. Distinguish implementation, unit-test, integration, benchmark, validation, and production evidence.
7. Never claim a result is scientifically or engineering-validated unless the relevant domain evidence supports that claim.
8. Avoid adding general platform features here when they belong in TensorPlatform; avoid adding specialist domain rules to TensorPlatform.
9. Keep changes scoped, reviewable, and reversible. Do not equate PR count or workflow completion with quality.

## Current maturity

The charter defines intended boundaries, not a claim that every model or benchmark is validated.

This charter describes responsibility and acceptance expectations; it does not override the source code or claim that planned functionality is implemented. Update the README and this document when ownership changes, and resolve conflicting documentation explicitly.
