# SUBOFF OpenLB source audit — second pass (2026-11-20)

**Scope:** static review of actual TensorLBM default-branch files and OpenLB 1.9r0 published Bouzidi source. No numerical run, no external binary execution, no claim of corrected physics.

## New confirmed findings

### F6 — duplicated halfway-link code in AMR common wall chain (P0)
`src/tensorlbm/boundary/wall_chain.py::bfl_halfway_links_3d` separately creates `q_field = torch.full(masks.shape, 0.5)` and uses `torch.roll` to identify neighbors. The uniform high-Re runner `src/tensorlbm/suboff_wallmodel_les_runner.py::_halfway_links` contains the same pattern. A single-runner patch will **not** fix AMR. Replace both with a shared cut-link builder, where q comes from physical geometry intersection; retain halfway as explicitly labeled approximation/fallback. Tests must cover uniform and AMR grid coordinates, six nonperiodic faces, diagonals, and level-scaled geometry.

### F7 — D3Q19 'cumulant' is a reduced stress/Hermite reconstruction (P1, characterization)
`src/tensorlbm/cumulant.py::collide_cumulant_d3q19` describes relaxation of 2nd-order stress and projection into a 2nd-order Hermite subspace; this is not equivalent to the full nonlinear high-order D3Q27 cumulant transform. Do not treat naming as validation. Compare against D3Q19 MRT/TRT and D3Q27 cumulant at matched Re, Ma, boundaries and grid. Measure transport coefficients, Galilean invariance, ghost-mode decay, kinetic period-two oscillations, drag convergence. This is an algorithmic characterization, not a proven defect.

### F8 — control-volume observer is more complete than a simple force sum (positive finding, conditional)
`src/tensorlbm/control_volume_force.py::observe_control_volume_force` calculates net momentum import minus fluid momentum change, with validation of strictly interior nonperiodic control volumes. The SUBOFF high-Re runner invokes this with `periodic_axes=()`. Retain this observer and test whether wall forcing, sponge, far-field replacement and collision terms cross the control-volume budget. A mismatch with BFL wall force is not automatically a bug in the CV formula; it may indicate omitted sources, inconsistent sampling or wall-force double counting.

### F9 — BFL + Guo wall-stress composition (P0 validation need)
`src/tensorlbm/boundary/wall_model.py::bfl_wall_function_3d` combines BFL, wall stress and optional Guo forcing. Test separately (i) BFL only, (ii) wall stress only, (iii) both, under exact impulse/momentum accounting. Record the applied population-space forcing and the reported force; assert they agree in sign, units and time level. Avoid assuming the wall-model Guo body force is equivalent to OpenLB's moving-wall Bouzidi velocity term.

## External reference details
OpenLB 1.9r0 `src/boundary/setBouzidiBoundary.h`, `BouzidiPostProcessor::apply`, lines 130–166 in rendered source: for each valid link, `q<=0.5` uses the boundary and solid-side populations; `q>0.5` also uses the opposite fluid-side neighbor. The geometry setup populates link distances from boundary intersections. This establishes the reference *mechanism*; verify population orientation when translating to TensorLBM's pull streaming.

https://openlb.gitlab.io/release/api/d3/d53/setBouzidiBoundary_8h_source.html

## Required immediate work
1. Add independent nonwrapping D3Q19 link-builder tests (including boundary-touching solids).
2. Implement shared geometry-aware link fractions for bare hull; use explicit fallback ledger for appendages until geometric intersection exists.
3. Assert uniform and AMR wall-chain consistency at identical physical resolution.
4. Write impulse-budget and force-observer closure tests with and without Guo forcing.
5. Characterize D3Q19 cumulant against other collisions, without premature replacement.
6. Only then resume high-Re SUBOFF campaign.

## Validation status
All findings above are **static**. No new CFD errors, numerical orders, test passes or physical validation claims are asserted.
