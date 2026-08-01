# SUBOFF direct-CFD development log — 2026-08-01

This log prevents failed trials from being silently discarded or presented as
validation.  Primary acceptance remains the Liu & Huang AFF-1/AFF-8 tow data.

## Confirmed findings

1. The former BFL startup blended reflected populations with populations
   streamed from solid nodes.  Full BFL with a relative-velocity ramp removes
   that non-physical blend, but does not by itself settle the force.
2. A new internal kinetic control volume agrees with BFL link force to 0.002%
   on the Re=100 sphere benchmark (R=8).  SUBOFF force excursions therefore
   cannot be dismissed as a post-processing sign/factor error.
3. The former Guo wall source used `tau_w/y1` in a unit-volume boundary cell.
   At the common `y1=0.5`, applied streamwise momentum was twice the reported
   wall traction.  Commit `13a1ce2` changes the source to `tau_w*A/V` and adds
   an exact momentum-vs-reported-traction regression.
4. MRT+Smagorinsky AMR became non-finite near step 900.  D3Q19 cumulant+
   Smagorinsky extended this to step 1660.  A moment-preserving positivity
   limiter prevented NaNs but the force still grew monotonically after about
   step 1400.  Enlarging the domain and refinement patch did not cure it.  All
   shell-reflux AMR candidates were stopped; the next AMR implementation must
   use face-local flux registers and conservative coarse/fine reconstruction.
5. Direct uniform L120 with the corrected wall traction remained finite
   through at least step 1800.  At that point it reported approximately
   103 N pressure and 82 N wall shear.  One convective time is 2000 steps at
   `L=120, U_lu=0.06`, so this is not a settled resistance result.
6. R=8 sphere, Re=100, D=16 cells: `Cd_CV=1.38065`,
   `Cd_BFL=1.38067`, observer difference 0.0017%, correlation error 26.5%.
   Increasing only the resolution to D=24 gave `Cd=1.36671` (25.2% error).
   Keeping D=16 but increasing transverse width from 4D to 8D gave
   `Cd=1.19869` (9.8% error), proving domain blockage was dominant.
7. An incoming-only non-equilibrium extrapolation boundary is implemented
   separately from the legacy hard-equilibrium boundary.  On the R=8, 8D-wide
   sphere it changed mean Cd by less than 0.0001%; it therefore preserves the
   baseline but has not yet demonstrated lower long-period reflection.
8. The periodic-span cylinder at Re=100, D=24 and 8.3D transverse width gave
   `Cd_CV=1.47698` and `Cd_BFL=1.47699` (observer difference 0.00078%).  The
   value is 11.1% above the nominal unconfined reference 1.33.  Increasing the
   width to 16.7D gave `Cd_CV=1.26448`, `Cd_BFL=1.26449` (difference 0.00053%),
   4.93% below the reference.  Domain blockage is therefore confirmed, but the
   5000-step sampling window spans only about two expected shedding periods.
   Strouhal/lift-spectrum diagnostics and a longer window are required before
   interpreting the remaining error.
9. The D3Q27 wall-stress source contained the same obsolete `tau_w/y1`
   scaling already removed from D3Q19.  Both lattices now share the
   `tau_w*A/V` traction contract and exact population-momentum regression.
10. Force admission no longer depends on an instantaneous reference crossing
    or only three report points.  Equal-duration block means now check range,
    first-half/second-half drift and linear trend, and fail closed with fewer
    than four complete blocks.
11. The failed shell-reflux implementation has been replaced in the runtime
    by a link-local kinetic register.  It observes post-collision/pre-stream
    populations, integrates two fine substeps in physical cell-volume units,
    preserves uniform flow, and corrects only exterior interface links.  It
    remains an implementation candidate pending long uniform-grid comparison.
12. The standard L120 domain is only 2.5 hull lengths long: 0.375L upstream
    of the bow and 1.125L downstream of the stern.  This is inadequate for a
    validation-grade external-flow wake and is consistent with the recurring
    low-frequency pressure-force excursions.  Hull placement is now explicit,
    and the default equilibrium-difference sponge excludes the prescribed
    inlet; the next direct comparison uses an elongated downstream domain.
13. The completed L120 hard-equilibrium run averaged 122.29 N over its final
    2000 steps (39.92% high) with 38.18% block drift.  Its wall friction was
    67.10 N, corresponding to `Cf≈0.00245`, about 14% below ITTC-1957 at the
    physical Reynolds number.  A finite two-sided flat-plate external-flow
    benchmark now isolates this wall-stress bias before further SUBOFF tuning.
14. The paired L120 incoming-only non-equilibrium run also failed: its final
    2000-step mean was 139.34 N (59.43% high) with 4.33% block drift.  Both
    short-domain boundary variants show the same order of force-observer gap
    and low-frequency pressure excursions.  Non-equilibrium reconstruction is
    therefore retained as a cleaner open boundary, not claimed as a cure for
    an under-sized wake domain.
15. Uniform L160 in the same short-domain proportions also failed, averaging
    159.22 N (82.18% high) with 11.70% block drift.  Its 75.81 N wall friction
    is only about 3.3% below the ITTC context value (~78.4 N), versus the L120
    run's ~14% deficit.  The dominant error is pressure resistance (83.42 N),
    not a need to tune friction upward.  The CV versus BFL-total gap also fell
    from about 43 N at L120 to 12.9 N at L160, indicating spatial convergence
    of the curved moving-slip boundary diagnostic even though the short wake
    domain remains physically unacceptable.
16. The first finite flat-plate production run (`Re=1e6`, plate L=256,
    collision `Re=1e5`) gave `Cf=0.004493`, only 4.15% below ITTC-1957, and
    its control-volume versus BFL-plus-wall-stress total differed by 0.029%.
    It is not admitted: eight block means span 8.44%, linear trend is 6.25%,
    and the positivity limiter peaked at 0.238% of cells.  A lower collision
    Reynolds sensitivity is running with the same physical wall viscosity.
17. A 30,000-step Re=100 cylinder run in the 16.7D-wide domain gave
    `Cd=1.26146` with 0.00064% CV/BFL disagreement, but failed the physical
    gate: Cd was 5.15% low, `St=0.1368` was 16.6% low, and eight Cd blocks
    spanned 6.58%.  The benchmark now writes resumable population/force-history
    checkpoints; a 60,000-step inlet-sponge-free run is in progress rather
    than restarting every extension from an empty history.
18. Lowering only the flat-plate collision Reynolds number from `1e5` to
    `2e4` removed all positivity limiting and produced excellent stationarity
    (0.054% block range, 0.040% trend) with 0.029% force-observer disagreement.
    However, `Cf=0.005212` is 11.19% above ITTC.  Collision-Re sensitivity is
    therefore material; a declared 20k/50k/100k sequence is used instead of
    selecting whichever numerical viscosity happens to match the correlation.
19. The elongated L120 domain (1L upstream, 3L downstream; 72-cell outlet
    sponge, no inlet sponge) still failed: 155.24 N mean, 77.62% error and
    19.52% block range.  More importantly, the three force paths disagree:
    CV total 155.24 N, BFL-link plus wall stress 200.56 N, and sampled surface
    pressure plus wall stress 60.11 N (surface pressure −7.00 N).  No one of
    these is promoted by proximity to experiment.  A curved moving-slip
    boundary force ledger and nested-CV invariance test now precede further
    SUBOFF accuracy claims.
20. The healthy-GPU R12 sphere in an 8D transverse domain completed with
    `Cd=1.16053`, 6.30% above Schiller–Naumann, and 0.00248% BFL/CV
    disagreement.  This improves monotonically from the geometrically similar
    R8 result `Cd=1.19869` (9.80% high), but still fails the 5% drag and 1%
    stationarity gates (1.67% block range, 1.61% trend).  R14 is running with
    checkpoints; extrapolated R16 peak memory (~22.7 GiB) is deferred until
    R14 establishes the remaining resolution trend.
21. Extending the 16.7D-wide cylinder to 60,000 steps changed the diagnosis:
    the full post-10k mean `Cd=1.35881` is only 2.17% high and `St=0.17364`
    is 5.88% high, with 0.00055% force-observer disagreement over 21.7 cycles.
    It is still rejected because early blocks rise from ~1.225 while the final
    four settle near 1.45 (20.6% trend over the admitted window).  A 25D-wide
    rerun uses a 30k warmup to separate long startup from residual blockage.
22. The flat-plate L256/L512 comparison exposed a grid contract error in the
    original wall-stress path: both grids evaluated the wall law at the first
    cell centre (`y=0.5`), so refinement changed the physical exchange height.
    A reusable sparse trilinear exchange sampler now evaluates velocity and
    wall distance at a declared wall-normal location while retaining BFL slip
    impermeability and conservative Guo traction.  Population assimilation is
    deliberately excluded from this path.  Manufactured linear-field tests,
    positive-distance validation, and exact source-momentum/traction tests
    cover the new module.  Flat-plate and direct-SUBOFF checkpoints record the
    exchange distance, preventing physically different runs from being
    resumed into one history.
23. Wall stress now emits an applicability ledger: requested/active exchange
    nodes, rejected sample fraction, exchange distance, min/mean/max `y+`,
    mean friction velocity, and the full shear-force vector.  Samples outside
    the domain or with a solid-contaminated trilinear stencil are rejected
    rather than clipped.  Flat-plate and direct-SUBOFF outputs persist the
    ledger; more than 1% rejected exchange samples fails admission.  This
    prevents an integrated-force match from hiding an invalid wall-law region
    or geometry-intersection sampling error.
24. The L512 first-cell flat-plate run at collision `Re=2e4` completed with
    `Cf=0.00606261`, 29.34% above ITTC-1957.  It was exceptionally stationary
    (0.00277% eight-block range), required no positivity limiting, and its CV
    versus BFL-plus-stress totals differed by only 0.0599%.  This is strong
    negative evidence rather than numerical noise: paired with L256
    `Cf=0.0052118`, refinement while holding `y=0.5` lattice cells changes the
    physical sampling height and increases the wall stress.  The first-cell
    formulation is rejected as grid dependent; the replacement campaign fixes
    `y/L` with L256/exchange=3 and L512/exchange=6.
25. Nested control volumes are now an admission criterion rather than a
    passive table.  A reusable assessment compares one primary force balance
    with independently enclosing volumes, rejects non-finite values, and
    requires at least two auxiliary volumes.  Direct SUBOFF admission requires
    every auxiliary mean to agree with the primary within 1%; a close tow-tank
    match from one arbitrarily placed control surface cannot pass this gate.
26. Wall applicability reductions are cadence-controlled (50 steps by
    default) so production GPU runs do not synchronize min/mean/max reductions
    every time step.  The cadence is part of checkpoint identity, and a BFL
    run with no collected wall-applicability sample fails closed.
27. The elongated L120 collision-`Re=1e5` run finished at 156.37 N (78.92%
    high) with 12.66% drift and is rejected.  Its first nested-CV report
    incorrectly compared a 50-step-cadence auxiliary mean (~−7983 N) against
    the primary every-step mean.  Direct checkpoint audit at the identical 40
    timestamps shows margin 4 and 12 differ from the margin-8 primary by only
    0.00113% and 0.00168%.  This is sampling-phase aliasing, not spatial CV
    failure.  Checkpoint schema v3 now stores the primary CV at every auxiliary
    timestamp and gates only paired means; every-step primary history remains
    the resistance/time-stationarity observer.  Sparse and dense temporal
    averages are never mixed again.
28. The R14 sphere completed at `Cd=1.155887`, 5.88% above the
    Schiller–Naumann reference, with only 0.00310% CV/BFL disagreement.  It
    continues the monotone R8→R12→R14 spatial trend, but fails both the 5%
    drag gate and 1% stationarity gate (1.45% block range, 1.39% trend).  The
    final block is lower at 1.14977, so the saved state is extended from 5000
    to 8000 steps before risking the estimated ~22.7 GiB R16 allocation.
29. Static AMR now has a uniform-fine reference benchmark independent of
    SUBOFF.  A 24-step moving smooth perturbation used 72.27% fewer cells,
    held relative mass drift to `1.02e-7`, reflux residual to `1.46e-11`, kept
    all populations positive, and improved density RMS versus uniform coarse
    both inside the refined block (`4.00e-6`→`3.40e-6`) and on the interface
    shell (`3.63e-6`→`2.69e-6`).  This admits the short smooth-interface test,
    not a body-force or long SUBOFF AMR claim.
30. Extending that interface benchmark to 100 coarse steps retained
    `3.05e-7` mass drift, `1.46e-11` maximum reflux residual, positivity and no
    limited directions.  Density RMS stayed 4.67% better than uniform coarse,
    but streamwise-velocity RMS was 1.99% worse.  The admission contract now
    exposes both ratios, requires density improvement, and permits at most 5%
    velocity regression; the candidate passes while the residual reflection
    remains explicit evidence for later higher-order prolongation work.
31. A reusable spatial-convergence assessor now requires at least three
    strictly increasing resolutions, checks monotonicity, fits the observed
    order in `phi(N)=phi_inf+a*N^-p`, and reports the extrapolated limit,
    finest-grid discretisation error and fit RMS.  A manufactured second-order
    sequence recovers `p=2` and its exact limit.  The pending fixed-physical-
    height flat-plate campaign therefore adds L384/exchange=4.5 between
    L256/exchange=3 and L512/exchange=6; two-grid agreement is insufficient.
32. The complete L256/exchange=3 Musker flat-plate run passed every single-grid
    gate: `Cf=0.00459729` (1.924% below ITTC), 0.283% eight-block range,
    0.028% linear trend, 0.0286% CV/BFL-total disagreement, no positivity
    limiting and no rejected exchange sample.  Exchange `y+` ranged 533–808
    with mean 563.  This replaces the grid-dependent first-cell candidate but
    remains a single-grid result until L384/L512 establish spatial convergence
    and the streamwise stress distribution is audited.
33. Extending R14 sphere from 5000 to 8000 steps gives a full post-2500 mean
    `Cd=1.15145`, still rejected at 5.47% reference error and 1.25% trend.
    A separately audited final-3000 window is stationary (0.169% block range,
    0.161% trend) with 0.00581% CV/BFL disagreement, but `Cd=1.14776` remains
    5.132% above Schiller–Naumann.  The late window is not substituted for the
    declared full window; it identifies a small residual spatial/model bias
    for the running R15 point to resolve.
34. Flat-plate checkpoint/result schema v3 expands restart identity to include
    domain/plate proportions, ramp, sponge, control-volume margin, LES
    constant, positivity policy and wall diagnostic cadence.  The new
    multi-grid assessor accepts at least three individually admitted v3 runs
    only after exact configuration identity, invariant domain ratios and
    invariant exchange-height/plate-length ratio.  Existing v2 runs are
    retained as development evidence but fail formal provenance rather than
    being retroactively relabelled.
35. The 25D-wide D24 cylinder completed 60,000 steps.  Its full post-30k mean
    `Cd=1.38966` is 4.49% high, but is rejected because eight blocks rise from
    1.242 to 1.444 (13.55% trend); `St=0.17300` is 5.49% high.  A separately
    audited final-15k window is stationary (0.911% range, 0.746% trend) but
    settles at `Cd=1.44097` (8.34% high) and `St=0.17521` (6.83% high, only
    6.57 cycles).  Thus startup-biased full-window averaging cannot be used to
    claim drag accuracy.  CV/BFL disagreement is only 0.00053% and density is
    stable; the queued D32 run tests spatial resolution at the same 25D width.
36. The L512/exchange=6 Musker flat plate also passed every v2 single-grid
    gate: `Cf=0.00466897` (0.395% below ITTC), 0.00147% block range, 0.00091%
    trend, 0.0821% force-observer difference, no limiter and no rejected
    samples.  Mean exchange `y+` is 566 (range 552–588), close to L256's 563
    because `exchange_distance/L` is fixed; the two `Cf` values differ by
    about 1.56%.  This is strong grid-consistency evidence, but v2 provenance
    remains below the formal v3 multi-grid gate, whose rerun chain is active.
37. The static-AMR SUBOFF runner is upgraded from an execution demo to a
    resumable single-grid evidence producer.  Atomic checkpoints now preserve
    coarse/fine populations, force histories, wall `y+`/rejection histories,
    limiter maximum, per-population reflux residual and reporting history under
    a complete configuration identity.  The fine hull uses analytical SUBOFF
    normals, wetted-area wall traction and optional exchange-location stress.
    Admission now gates experimental error, force stationarity, CV/BFL
    agreement, positivity, reflux and wall sampling while keeping physical
    validation false.  A 4→6-step CPU save/resume composition test passes with
    `5.46e-12` reflux residual and correctly rejects its unphysical tiny-grid
    resistance.
38. Wall traction now uses a reusable orientation-aware BFL surface measure,
    `N_axial/(|nx|+|ny|+|nz|)`, instead of uniformly spreading analytical area
    over nodes.  Axis-aligned and 45-degree manufactured patches pass exactly;
    calibration preserves a declared total without flattening local weights.
    AFF-1 uses analytical normals/area.  AFF-8 calibrates a gradient-based bare
    proxy and applies the same factor to hull+sail+stern planes, adding their
    area explicitly.  A coarse L24 composition probe gives 179.79 bare and
    195.41 full square lattice units with zero unweighted active nodes.  Direct
    checkpoint/result schema v4 prevents old uniform-area histories from
    resuming into the new force sequence.  The common stationarity assessor
    also now fails closed, rather than dividing by zero, when only one complete
    time block exists.
39. The completed v2 fixed-height flat-plate sequence is monotonic:
    L256/L384/L512 `Cf` values are 0.004597289/0.004657755/0.004668967.  Its
    mathematical fit gives observed order 3.739, extrapolated
    `Cf=0.004674770`, and 0.124% finest-grid distance; the limit is 0.272%
    below ITTC.  This is stored with `physical_validation=false` because v2
    lacks the expanded configuration identity.  The generic CLI now names its
    gate `mathematical_fit_admitted`, never bare `admitted`; only the active v3
    case-specific aggregator may issue a formal convergence admission.
40. V3 L256 completed and reproduces the v2 result exactly at reported
    precision (`Cf=0.004597289308156069`) while persisting the expanded
    provenance.  The flat-plate aggregator now also checks
    `sponge_width/L` and `cv_margin/L`; the active sequence uses 24/36/48 and
    6/9/12 at L256/L384/L512.  This prevents different relative sponge or
    force-observer placement from masquerading as spatial convergence.
41. The safe-memory R15 sphere point completed with 18.72 GiB peak allocation.
    Its post-2500 mean `Cd=1.15650` is 5.93% above Schiller–Naumann, with
    1.68% block range and 1.62% trend, so it is rejected; CV/BFL disagreement
    remains only 0.00327%.  This 5000-step state is earlier in its transient
    than the R14 8000-step late window and is not mixed into an observed-order
    fit.  It is retained for a later equal-time resume rather than using an
    unsafe ~22.7 GiB R16 run on a 23.69 GiB device.
42. Production static-AMR evidence is promoted to schema v3 and now persists
    the complete physical, boundary, collision, wall, refinement and scaled
    domain identity needed for a legitimate grid sequence.  The reusable
    `suboff_amr_convergence` assessor rejects mixed AFF variants, speeds,
    physics, wall laws, domain ratios, refinement margins, exchange heights or
    non-dimensional run times before fitting
    `R(N)=R_inf+a N^-p`.  Final admission requires all three single-grid gates,
    monotonic spatial convergence, acceptable fit/discretisation error and an
    extrapolated resistance within the experimental tolerance.  Six focused
    assessor tests and the full static-AMR/interface group pass; the active
    exploratory AFF-1 v2 run remains diagnostic and cannot enter this v3 gate.
43. SUBOFF BFL preprocessing now accepts the solver's existing CAD solid mask
    instead of rebuilding the full geometry twice, and performs the ten-step
    link bisection in FP32.  The requested q resolution is only about 1/1024
    lattice units, so FP64 coordinates added cost without usable wall-location
    information on the single-precision solver state.  Rebuilt and reused
    paths produce bit-identical masks/q fields in the regression test; the
    BFL plus static-AMR production group passes 17 tests.  This optimization
    is common to AFF-1/AFF-8 and future analytical SUBOFF campaigns.
44. The first L120/fine-L240 AFF-1 AMR trajectory was stopped at step 1000 as
    negative evidence, not averaged into a resistance claim.  Although its
    state remained finite (`rho=0.912..1.059`, maximum speed 0.225), the
    hidden historical population-reflux residual reached 25.79, the positivity
    limiter reached 0.295%, and instantaneous control-volume resistance ranged
    from about -1592 to 1854 N versus the 87.4 N experiment.  The audit also
    exposed a one-sided reflux guard that could amplify such an instability:
    removal was capped but positive injection was unbounded.  Reflux
    corrections are now symmetrically limited to 20% of
    local directional inventory; unapplied corrections remain explicit
    residuals and therefore cannot pass conservation.  V3 checkpoints/results
    additionally retain requested/applied peaks and limited-direction counts.
    The strengthened AMR groups pass 25 tests.  Its safety rerun exposed
    `raw request=159.93`, four limited directions and residual 25.99 by the
    step-750 checkpoint, so it too was stopped and retained as negative
    evidence rather than used for resistance averaging.
45. Static AMR reflux now projects the raw Q-population flux mismatch onto the
    four collision invariants (mass and three momentum components) before a
    bounded face-local correction.  It no longer injects non-conserved stress
    and higher-order kinetic modes merely because coarse and fine collision
    states represent them differently.  FP32 lattice weights receive an
    algebraic moment-closure correction, and randomized D3Q19 tests recover all
    four requested moments at state precision.  The updated 100-step uniform-
    fine comparison uses 72.27% fewer cells, has `4.07e-7` relative mass drift,
    `1.46e-11` maximum conserved-moment reflux residual, no limited directions,
    6.30% lower refined-region density error and only 3.55% velocity-error
    regression versus uniform coarse; it passes the existing 5% interface
    gate.  The raw kinetic mismatch and projected correction are both retained
    as diagnostics.  The exact AFF-1 conserved-moment rerun then completed
    1000 steps with 7.90 GiB peak allocation, zero positivity limiting, zero
    limited reflux directions and `1.86e-9` residual.  It stays finite with
    maximum coarse speed 0.0681, whereas the old kinetic-mode reflux failed by
    this time.  Its 500-sample startup mean is correctly rejected
    (`R=235.9 N`, 170% error, 15.2% observer difference, 5.9% trend); a clean
    8000-step/4000-warmup production candidate is active on the freed GPU.
46. V3 L384 flat-plate validation completed and exactly reproduces the v2
    fixed-height result: `Cf=0.0046577554833`, 0.635% from ITTC-1957, with
    0.00078% stationary-block range, 0.00068% trend, 0.0582% force-observer
    difference, zero positivity limiting and zero rejected exchange samples.
    Its mean sampled `y+=565.44`; the run passes every single-grid gate.  L512
    remains active, so the formal three-grid v3 fit is not issued early.
47. V3 L512 then completed and the provenance-gated L256/L384/L512 sequence is
    formally admitted.  `Cf` increases monotonically from 0.0045972893 through
    0.0046577555 to 0.0046689666; the fit gives observed order 3.739,
    `Cf_inf=0.0046747705`, 0.124% finest-grid discretisation error and zero
    reported fit RMS for the three-point model.  The extrapolated value is only
    0.272% below ITTC-1957.  Every source run passes its own force, stationarity,
    positivity and exchange gates, while all physical identities and scaled
    domain/sponge/CV/exchange lengths match.  The committed convergence artifact
    records SHA-256 hashes of the three full remote source JSON files.
48. The AFF-1 conserved-moment production grid sequence is launched at coarse
    L90/L120/L150 (effective fine L180/L240/L300).  Their coarse domains are
    exactly 5L by 1L by 1L; wall margins are L/15, wake refinement is 5L/6,
    sponge width is L/10, fine CV margin is L/30 and exchange height is
    3L_f/256.  Steps/warmup/statistical window/ramp also scale with fine
    resolution as 33.333/16.667/4.167/4.167 per fine-length cell.  The largest
    allocation is estimated below 18 GiB, and each grid has an independent
    checkpoint.  The convergence CLI now records source SHA-256 hashes just as
    the admitted flat-plate chain does.
49. Sphere checkpoint/result provenance is upgraded to v2 after auditing the
    old R15 state.  V1 persisted only shape, radius, Reynolds number and lattice
    speed (plus a later inlet-sponge default), so it could not prove that CV
    margin, center, ramp, sponge width/strength or outer-boundary mode remained
    unchanged on resume.  V2 stores and compares all of those fields plus the
    collision identity and warmup; a save/resume mismatch regression test
    passes.  The old 5000-step state remains negative evidence and is not
    resumed.  A clean R15 8000-step run uses explicit scaled parameters on the
    otherwise idle GPU 0 instead.
50. The clean R15 launch revealed that physical GPU 0 was not actually idle:
    two unrelated processes occupied about 11 GiB, and the new run exited on
    its own allocation failure without touching them.  A reusable CUDA memory
    budget gate now compares an empirical peak with live free memory plus a
    configurable reserve before allocating solver populations.  Sphere uses
    1000 bytes/cell (slightly above the measured R15 coefficient), while static
    AMR uses its existing 943 bytes/allocated-cell estimate; both reserve an
    additional 1 GiB and persist the preflight numbers in results.  Ten focused
    memory/sphere/static-AMR tests pass.  R15 waits for a genuinely free safe
    device instead of competing with another workload.
51. Cylinder checkpoint/result provenance is likewise promoted to v2 with the
    complete center, collision, warmup/ramp, sponge, CV, open-boundary and
    periodic-axis identity plus the live-memory preflight.  Save/resume and
    changed-sponge rejection tests pass.  The already-running D32/80k process
    remains isolated on its loaded v1 code and is treated as diagnostic; future
    formal cylinder sequences use v2 rather than retroactively relabelling it.
52. The L120 AFF-1 run exposed a backend edge case in the positivity limiter
    after step 1750: an empty selected-cell tensor reached `alpha.min()` and
    raised even though the flow had not produced NaN.  Empty selections are
    now an explicit identity operation (`limited_cells=0`, `alpha=1`), and
    non-finite floors are rejected before comparison.  The limiter plus static-
    AMR regression group passes six tests.  The production run resumes from its
    atomic step-1500 checkpoint with unchanged physical identity; the lost 250
    transient steps are recomputed rather than reconstructed.
53. The resumed v3 AFF-1 sequence then exposed a separate physical-contract
    error before any result was admitted.  At L90/step4500 the mean wall shear
    alone was about 229 N (latest 750-step mean 238 N) versus 87.4 N total tow
    resistance, with mean `y+≈75`.  The runner had passed the artificial
    `Re_collision=1e5` viscosity to the wall law.  The validated flat-plate
    path correctly separates collision viscosity from
    `nu_wall=U_lattice*L_fine/Re_physical`; applying the former to physical
    traction overpredicts skin friction.  All three v3 trajectories were
    stopped and retained as negative evidence.  A common
    `physical_wall_lattice_viscosity` contract now serves flat plate and
    SUBOFF; static-AMR result/checkpoint schema v4 persists the physical wall
    Reynolds, fine wall viscosity and viscosity basis.  Twenty focused wall,
    flat-plate, AMR and convergence tests pass, and v3 checkpoints are
    intentionally non-resumable under the corrected physics.
54. The local D32, 25D-wide cylinder reached 80000 steps and is rejected after
    window audit.  The full post-40000 mean `Cd=1.3650` is only 2.63% high and
    has 0.00067% CV/BFL disagreement, but its eight blocks rise from 1.211 to
    1.438 (17.0% trend).  The final 15000 samples are stationary (0.61% trend)
    but `Cd=1.4352` is 7.91% high and only 4.95 shedding cycles are observed;
    the final 10000 samples give `St=0.16850` (2.75% error) but only 3.16
    cycles.  No window simultaneously meets drag, stationarity, Strouhal and
    cycle-count gates.  A committed sidecar records full launch provenance,
    four late-window assessments and SHA-256 hashes while explicitly retaining
    `physical_validation=false` for the legacy-v1 source.
55. The v4 L120 1500-step wall-physics diagnostic completed with physical
    `Re=13.21M`, `nu_wall_fine=1.09e-6`, zero positivity limiting, no limited
    reflux directions and `3.73e-9` maximum residual.  Wall shear over the
    post-750 samples is 80.5 N instead of the v3 collision-viscosity value near
    190--240 N.  CV resistance still falls across all eight blocks
    (`166.6`→`136.9 N`, 20.2% trend) and the two observers differ by 27.3%, so
    the short run is correctly rejected as transient.  The clean v4
    L90/L120/L150 production sequence is active.

## Rejected candidates

- Empirical tow-table interpolation: useful only as an engineering baseline,
  never direct CFD validation.
- First-cell Spalding exchange replacement without a developed boundary
  layer: produced large transient/negative pressure and failed stability.
- Slower wall activation and half lattice Mach number: delayed but did not
  remove pressure excursions.
- Population-uniform shell reflux: can deplete small diagonal populations;
  replaced by proportional, positivity-limited reflux.

## Active campaign on the Wuxi GPU host

Results and logs are under:

`/home/wxsc/TensorLBM-cfd-20260801/results/amr_campaign_20260801`

Active direct-grid comparisons cover:

- legacy hard-equilibrium versus incoming-only non-equilibrium far field at
  uniform L120;
- uniform L120 versus uniform L160 spatial convergence;
- a long L120 time history beyond five nominal convective times;
- R=12 sphere in an approximately 8D transverse domain.

The former static-block shell-reflux AMR is explicitly excluded from physical
claims.  Its failed logs are retained as negative evidence.  The replacement
face-local kinetic coupling is implemented but is not promoted until its
long-run interface and force behaviour agree with a uniform fine grid.

No result is admitted before at least five convective times, three settled
windows, two independent force observers, and three effective resolutions.
