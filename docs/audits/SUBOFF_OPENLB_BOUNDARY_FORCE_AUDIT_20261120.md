# SUBOFF / OpenLB boundary and force source audit (2026-11-20)

Status: **static source audit; numerical reproduction NOT RUN; no physical validation claim**.

## Source lock
TensorLBM default branch inspected through GitHub connector on 2026-11-20. Relevant paths:
- `src/tensorlbm/suboff_wallmodel_les_runner.py`
- `src/tensorlbm/obstacles.py`
- `src/tensorlbm/boundary/boundaries3d.py`
- `src/tensorlbm/suboff_amr_wallmodel_runner.py`
- `benchmarks/pending/suboff_highre_resistance/README.md`

External reference: OpenLB 1.9r0 `src/boundary/setBouzidiBoundary.h`, https://openlb.gitlab.io/release/api/d3/d53/setBouzidiBoundary_8h_source.html (BouzidiPostProcessor and setBouzidiBoundary). This is a conceptual and source-mechanism comparison, not a claim of bit-for-bit equivalence.

## Findings

### F1 (confirmed implementation limitation, P0): halfway distance substituted for curved geometry
`suboff_wallmodel_les_runner._halfway_links(solid)` sets `q_field = torch.full(masks.shape, 0.5)` for all directions. OpenLB's setter computes `q=distance/(dx*|c_i|)` from the analytic boundary indicator for each fluid/solid link, using 0.5 only as a fallback. Its postprocessor has distinct `q<=0.5` and `q>0.5` branches. TensorLBM's high-Re runner therefore cannot represent curved-link location through this helper. **Do not assume other TensorLBM BFL paths share this limitation.**

Required correction: generate real cut-link fractions from the SUBOFF radius profile and appendage geometry, validate direction conventions and nearest fluid neighbor, retain explicitly counted fallback links. Compare with an analytic plane/sphere and a reference implementation before integrating with wall stress.

### F2 (confirmed code risk, P0): periodic wrap in nonperiodic link detection
`_halfway_links` uses `torch.roll(solid, shifts=(-cz,-cy,-cx), dims=(0,1,2))` and then `~solid & neighbour` without clipping wrapped faces. A solid touching a face can create false links at the opposite face. Fix by non-wrapping shifted views and explicitly exclude off-domain links. Test solids on each of six faces, diagonal D3Q19 directions, and translation invariance.

### F3 (interface contract risk, P0): force observer aggregates all solid-cell populations
`obstacles.compute_obstacle_forces_3d` computes `2*sum(c_i*f_i*solid_mask)` after streaming, before bounce-back, with no explicit per-direction fluid-solid link mask. For certain streaming/solid-population conventions this may agree with momentum exchange, but it is not a general cut-link force observer and may count interior populations or conflict with interpolated BFL. **Not proven to be the force estimator used by the high-Re SUBOFF runner**, which primarily uses a control-volume observer. Test exact one-link exchange, interior population invariance, sign, and compare boundary and control-volume forces.

### F4 (API semantics discrepancy, P1): f_pre accepted but ignored
`boundary/boundaries3d.bounce_back_cells_3d` explicitly deletes `f_pre`, documenting that callers must restore NoDynamics state beforehand. `teaching/06_bb_bug_fix.py` describes a corrected call with `f_pre`. This is a documentation/contract inconsistency, **not yet proof of a runtime bug**. Trace `lbm_step_correct` and all SUBOFF call sites, then write a pre/post-collision population-state test.

### F5 (physics-validation blocker, P0): high-Re wall model not independently closed
`SuboffWallModelLesConfig` distinguishes physical `reynolds` from `resolved_reynolds`; `wall_nu = U*L/Re_physical`, `collision_nu=U*L/Re_resolved`. This is a modeling choice, not intrinsically wrong, but its near-wall traction, bulk dissipation and SGS terms require independent flat-plate/channel-flow validation. Existing formal high-Re five-anchor run fails with >1000% resistance error, high y+ and major force-observer disagreement (see pending benchmark README). Do not assert that simply increasing grid count resolves the discrepancy.

## Required test ladder (fail-closed)
1. D3Q19 opposite-direction and coordinate-order invariants.
2. Plane cut at q=0.25, 0.5, 0.75; independently computed BFL populations; missing-neighbor fallback and conservation.
3. Six-domain-face non-wrapping link masks, diagonal links, solid-interior population perturbation.
4. Static uniform equilibrium zero net force; single-link impulse sign; translation/rotation symmetry; force-moment consistency.
5. Couette/Poiseuille wall location and convergence order; sphere/cylinder force with mesh refinement.
6. SUBOFF AFF-1 grid/time/domain convergence and three-force-observer closure, then AFF-8 experiment at matching Re and normalization.

## Exit gates
- All code fixes with independent unit tests and regression tests; no source copying from GPL-licensed OpenLB into Apache-licensed TensorLBM.
- Force-observer mean discrepancy <=2% (target), stationarity block drift <=1% (target), explicit uncertainty.
- AFF-1 and AFF-8 absolute resistance error <=5% and appendage resistance ratio error <=3% at matching Re before setting physical_validation=true.
- Archive solver commit SHA, geometry hashes, reference-data provenance, config, raw time series and reproducible commands.
- No benchmark promotion based solely on Richardson self-convergence, surrogate interpolation or synthetic populations.

## References
- OpenLB: https://openlb.gitlab.io/release/api/d3/d53/setBouzidiBoundary_8h_source.html
- waLBerla: https://github.com/lssfau/walberla (architecture context only, not line-by-line audited)
- TensorLBM: https://github.com/cxs503/TensorLBM

This document records findings and a repair acceptance contract only. It does **not** assert that any fixes were implemented or simulations run.
