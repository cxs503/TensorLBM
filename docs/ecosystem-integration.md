# Tensor ecosystem integration

TensorLBM remains an independently usable Lattice Boltzmann solver and fluid-dynamics research toolkit. Its numerical library and solver-specific validation remain authoritative for LBM behavior.

## Responsibilities

- LBM lattices, collision/streaming operators, boundary conditions and fluid models.
- LBM-specific CPU/GPU execution, field diagnostics and fluid benchmark cases.
- Solver-native APIs and reproducible examples for fluid-only applications.
- Export of documented fields, loads and artifacts for downstream coupling.

## Boundaries with sibling projects

- **TensorPhysics** coordinates runs, adapters, time alignment, multiphysics exchange and run provenance; it should call stable TensorLBM interfaces rather than duplicate LBM kernels.
- **TensorPlatform** provides shared dataset/model catalog, lineage, permissions and AI lifecycle services. TensorLBM may retain solver-specific demos and pipelines while common platform functions are integrated incrementally.
- **TensorFVM** is an alternative finite-volume fluid solver; do not imply numerical equivalence or interchangeable validation without matched benchmarks.
- **TensorFEM** and **TensorDEM** consume or exchange documented loads/fields through explicit coupling adapters, not private module imports.

## Coupling contract

Every exported field/load should identify units, coordinate frame, mesh/lattice location, shape, physical time, sign convention and interpolation/integration method. For fluid-to-particle or fluid-to-structure transfer, verify total force and moment (and mass/energy where relevant); state clearly when a demonstration is not yet validated FSI.

## Recommended integration path

1. Keep fluid-only benchmark runs executable without TensorPlatform or TensorPhysics.
2. Expose a small versioned adapter for solver configuration, stepping, diagnostics and artifact export.
3. Register run manifests and output artifacts with TensorPhysics; store large arrays in artifact storage.
4. Register validated datasets and derived AI experiments with TensorPlatform.
5. Add coupled engineering cases only after standalone and interface-level tests pass.

## Repository map

- [TensorPlatform](https://github.com/cxs503/TensorPlatform) — shared data and AI services.
- [TensorPhysics](https://github.com/cxs503/TensorPhysics) — scientific workflow orchestration and multiphysics coordination.
- [TensorLBM](https://github.com/cxs503/TensorLBM) — lattice Boltzmann solver.
- [TensorFVM](https://github.com/cxs503/TensorFVM) — finite-volume solver.
- [TensorFEM](https://github.com/cxs503/TensorFEM) — finite-element solver.
- [TensorDEM](https://github.com/cxs503/TensorDEM) — discrete-element solver.
