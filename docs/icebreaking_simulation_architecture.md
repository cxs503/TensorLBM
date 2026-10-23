# Icebreaker–Sea-Ice Simulation: Initial Engineering Architecture

> Status: architecture baseline; not a claim that a validated icebreaking solver already exists.
>
> Scope: continuous icebreaker advance through level ice, broken-ice interaction, and time-resolved resistance. The first executable milestone is 2-D, with a deliberate path to 3-D.

## 1. Modeling decision

Use a partitioned multiphysics architecture rather than placing ice mechanics directly inside the LBM collision/streaming kernel.

- **TensorLBM** owns water/air flow, free-surface or multiphase treatment, hull-fluid interaction, hydrodynamic force integration, and time-series output.
- **TensorDEM** owns sea-ice discrete elements, intact-ice bonds, bond failure, floe fragmentation, ice–ice and ice–hull contact, and ice-body kinematics.
- **TensorFEM** is an optional later subsystem for local elastic hull/ice structural response. It is not required for the first rigid-hull icebreaking baseline.
- A small coupling layer owns synchronization, unit conversion, exchange fields, conservation checks, and run metadata. It must not silently duplicate physics owned by a solver.

Do not start by attempting a fully coupled 3-D simulation. First prove the force and fracture contracts in a small 2-D case.

## 2. Physics and numerical methods

### Fluid: TensorLBM

Reuse the existing TensorLBM multiphase and moving-boundary/FSI capabilities where their contracts and validation are appropriate. Do not assume that an existing water-entry example already solves icebreaking.

Initial fluid model:
- 2-D multiphase LBM for water and air; select a model only after checking interface stability and density/viscosity ratio requirements.
- Moving hull boundary with consistent momentum-exchange or immersed-boundary force accounting.
- Output pressure/hydrodynamic force separately from DEM contact force.
- Keep lattice units inside the solver and convert to SI only at defined interfaces.

### Ice: TensorDEM

Represent intact level ice as a connected assembly of particles (or polygonal elements in 2-D) with bonds. The initial engineering model should include:
- translational and rotational rigid-body state per element;
- normal and tangential bond stiffness, bond strength, and a documented failure criterion;
- post-failure contact with friction and damping;
- ice–hull contact with independently configurable friction;
- broken-piece tracking and optional removal only under an explicit, logged boundary policy.

A bond breaking is an irreversible topology/state change and must be recorded. Avoid calibrating bond parameters solely to match one resistance curve.

### Hull and ship motion

Start with a prescribed forward speed and rigid hull geometry. This isolates icebreaking resistance from propulsion/control and 6-DOF stability. Add surge/heave/pitch or full 6-DOF motion only after the prescribed-speed baseline passes force-balance and timestep checks.

## 3. Coupling contracts

All exchange data must state coordinate convention, SI/lattice units, time level, and sign convention.

| Interface | Producer → consumer | Minimum exchanged data |
|---|---|---|
| Hull kinematics | ship/hull driver → TensorLBM + TensorDEM | pose, linear/angular velocity, hull surface/segments, physical time |
| Fluid load | TensorLBM → force ledger / ship driver | pressure and viscous force/moment on hull, SI units after conversion |
| Ice–hull contact | TensorDEM → force ledger / ship driver | contact force/moment on hull, contact IDs, SI units |
| Ice geometry | TensorDEM → TensorLBM | current ice masks/immersed surfaces and motion at the coupling time |
| Fluid traction on ice | TensorLBM → TensorDEM | per-body or per-surface hydrodynamic force, with an explicit interpolation/integration method |
| Bond/fracture events | TensorDEM → output/diagnostics | bond IDs, event time, failure mode, released energy if available |
| Resistance history | force ledger → outputs | total resistance plus fluid, ice-contact, and other contributions |

The force ledger must prevent double counting. In particular, fluid force on the hull and DEM ice-contact force on the hull are distinct contributions. Action/reaction pairs must be reported consistently.

## 4. Coupling loop and timestep policy

At each physical coupling time:

1. Advance or predict hull pose.
2. Update hull boundary and current ice geometry for the fluid solver.
3. Advance the fluid solver for the agreed substeps.
4. Integrate fluid traction on the hull and ice surfaces.
5. Transfer ice-surface fluid loads to TensorDEM.
6. Advance DEM contact dynamics and bond failure for its substeps.
7. Accumulate DEM ice–hull contact force/moment.
8. Update hull state if dynamic motion is enabled; otherwise enforce prescribed motion.
9. Write diagnostics, fracture events, and restart state.

The initial implementation may use explicit partitioned coupling, but it must expose the fluid timestep, DEM timestep, subcycle ratio, and force-transfer timing in run metadata. Choose the DEM substep from contact/bond stiffness stability requirements; do not hard-code a universal ratio. Add coupling sub-iterations only if a measured stability or energy-balance problem requires them.

## 5. Resistance definition

For a prescribed-speed baseline, report the signed longitudinal resistance as separate terms:

- hydrodynamic hull resistance;
- ice–hull contact/fracture resistance from DEM;
- any explicitly modeled appendage or auxiliary contribution;
- total resistance, computed from the declared sign convention.

Also report instantaneous force, moving average, peak force, mean over a declared steady interval, and the interval used. Do not compare means taken over different startup/transient windows.

## 6. Verification ladder

### M0 — interface and bookkeeping tests
- unit conversion round trips;
- force sign and action/reaction tests;
- deterministic one-body and one-contact tests;
- total-force decomposition sums to total within tolerance;
- checkpoint/restart preserves body and bond states.

### M1 — DEM mechanics without fluid
- two bonded particles under tension/shear;
- bond failure threshold and irreversibility;
- two-body collision and friction;
- a bonded 2-D ice strip pushed by a rigid indenter;
- timestep/subcycling sensitivity.

### M2 — fluid and moving hull without ice
- static hull force baseline;
- prescribed-speed hull in still water;
- grid and timestep refinement of longitudinal force;
- force integration and boundary-motion sensitivity.

### M3 — 2-D coupled icebreaking
- prescribed-speed rigid hull through bonded level ice;
- fracture pattern, piece-size distribution, contact-force history, and total resistance;
- sensitivity to particle spacing, bond strength, friction, and speed;
- compare against an explicitly documented experimental/literature reference when one is selected.

### M4 — 3-D transition
- extrude/replace 2-D elements with 3-D DEM particles and a 3-D hull;
- validate buoyancy/hydrodynamic loading and contact independently;
- establish mesh/particle resolution and runtime scaling before long runs.

### M5 — engineering extensions
- dynamic surge and 6-DOF ship response;
- realistic hull geometry and operational speed sweeps;
- optional TensorFEM structural response;
- calibration, uncertainty, and reproducibility reports.

## 7. First implementation deliverables

The first code PR should be narrow and testable:

1. Add typed configuration and exchange-data contracts for the icebreaking case.
2. Add a force ledger that stores fluid-hull and DEM-hull contributions separately and computes total resistance using one sign convention.
3. Add unit tests for force decomposition, invalid units/time levels, and action/reaction.
4. Add a small synthetic coupling test with mocked fluid and DEM providers; do not present it as a physical icebreaking result.
5. Add a case schema and metadata fields for physical units, grid/particle resolution, material/contact parameters, timestep/subcycling, and resistance averaging window.

The actual DEM fracture solver should live in TensorDEM unless there is an explicit decision to prototype a minimal reference implementation inside TensorLBM. Avoid copying a second, divergent DEM implementation into TensorLBM.

## 8. Acceptance criteria

A milestone is not complete merely because the script runs. Each milestone must include:
- reproducible configuration and software/version metadata;
- tests for its stated numerical contracts;
- force histories and diagnostic artifacts;
- documented limitations and a comparison target or reason no reference is yet available;
- no unsupported claim of quantitative validation.

## 9. Open engineering decisions

Resolve these with explicit evidence before committing to a production model:
- whether the initial fluid interface uses the existing CG or Shan–Chen path, based on stability and density-ratio tests;
- the DEM element shape and contact law supported by TensorDEM;
- the selected bond constitutive law and failure criterion for the target ice regime;
- the hull–ice friction calibration source;
- experimental/literature dataset and resistance averaging window;
- the coupling timestep and acceptable energy/force imbalance.


## 10. Implemented traction-to-wrench utility

The module `tensorlbm.icebreaking.integrate_surface_traction` now provides a small integration seam for already-reconstructed hull traction samples:

- Inputs: global sample positions (m), traction vectors acting on the hull (Pa), surface quadrature weights (m²), and a moment reference origin (m).
- Outputs: `WrenchSI` containing the integrated force (N) and moment (N·m).
- Validation: matching sample counts, three-component vectors, finite values, and non-negative area weights.
- Tests: force integration, moment-arm/origin behavior, and invalid-input rejection.

This utility deliberately does not reconstruct wall stress, derive area weights from an LBM boundary, or establish the action/reaction convention of any particular force method. The caller must document those choices and verify the integrated result against an independent momentum-exchange or control-volume calculation before using it in a resistance result. It is an interface building block, not yet a TensorLBM–TensorDEM coupled solver.
