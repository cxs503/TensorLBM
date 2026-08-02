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
56. Independent force closure is verified below the production level before
    attributing the v4 observer gap to a formula.  On a transient curved sphere
    step, BFL slip pressure plus Guo wall shear agrees with the enclosing
    control-volume momentum balance to `3.6e-8%`.  A six-step SUBOFF static-AMR
    composition gives 0.30% mean disagreement while its force falls by an order
    of magnitude, with `3.64e-12` reflux residual.  The focused BFL/CV suite
    passes 24 tests.  Thus the 1500-step production gap is not excused, but the
    primitive force formulas close; long-run AMR sampling, CV placement and
    pressure-transient decay remain the hypotheses tested by the active runs.
57. A checkpoint pressure audit at v4 L90/step3500 gives 80--86 N from
    analytical-normal surface integration and about 90 N from BFL link
    momentum exchange; these pressure observers agree in sign and scale.  The
    simultaneous CV-derived pressure is much lower, so matching the experiment
    with CV alone is not accepted.  The hypothesis that separate float32 total-
    momentum sums caused the gap was tested and rejected: both paths already
    accumulate in float64 and agree on a large manufactured CV.  The common CV
    observer nevertheless now subtracts populations locally before reduction,
    which is better conditioned, but no accuracy claim is attached to that
    mechanical improvement.  AMR substep/CV placement remains under audit.
58. The first v4 L90 production segment completed at 6000 steps and is
    rejected despite several windows crossing the tow value.  Its post-3000
    mean is 102.45 N (17.2% high), but block means span 2.86--153.63 N with
    46.2% trend; CV/BFL disagreement is 39.3% and maximum speed reaches 0.231.
    Wall shear remains near 80 N, so pressure/acoustic oscillation dominates.
    The unfinished L120/L150 copies with identical damping were stopped rather
    than extrapolated.  Three L90 boundary sensitivities now isolate (a) wider
    0.2L, strength-0.3 sponge without inlet damping, (b) the same sponge on all
    faces, and (c) all-face damping at lattice speed 0.04 with time/ramp/window
    increased by 1.5 to preserve non-dimensional duration.  Only a stationary,
    force-consistent candidate will seed the next three-grid sequence.
59. The production runner now records three independent force-observation
    families instead of relying on one disputed total.  Schema v5 samples the
    primary CV and at least two geometrically nested CVs on the identical fine
    substep, integrates pressure independently on the analytical SUBOFF
    surface using the calibrated BFL area weights, and retains BFL link
    pressure plus coupled wall shear.  Checkpoints persist every paired sample
    and its complete configuration.  Admission fails closed unless two
    auxiliary CVs agree with the primary CV, analytical-surface total agrees
    with the BFL total, and the existing BFL/CV, stationarity, positivity,
    reflux, wall-sampling and experiment gates all pass.  The three-grid
    convergence assessor accepts only v5 records and additionally enforces
    scaled auxiliary-CV and sampling-interval equivalence.  Seven focused
    production/resume/convergence tests pass.  This instrumentation will
    distinguish AMR inventory-placement error from a real pressure/acoustic
    transient without selecting whichever observer happens to match the tow
    value.
60. The paired v5 ledger also separates non-physical momentum introduced by
    the collision and positivity operators from wall momentum exchange.  For
    each sampled fine substep it records the raw CV force, collision-plus-
    limiter momentum source, and the source-corrected CV closure diagnostic.
    The corrected value is never substituted for physical resistance: a
    numerical source above 1% rejects the run, while corrected-CV/BFL closure
    is a second algebraic gate.  This identifies whether a long-run observer
    split is caused by a non-conservative operator, without hiding that defect
    behind a post-processing correction.
61. Production admission now enforces physical duration directly rather than
    relying on campaign naming.  The initial default required at least five
    body-length convective times in total and two after warmup.  The measured
    L90 force spectrum later exposed an approximately 1.5-time dominant mode,
    so schema v6 raises this to eight times total and five after warmup.  The
    active four-time L90 boundary runs remain sensitivity evidence only even
    if a late force window passes; no three-grid sequence is assessed from a
    history containing only one or two cycles of its slowest resolved mode.
62. Both U=0.06 wider-sponge L90 sensitivities completed and are rejected.
    Without inlet damping the post-warmup CV mean is 96.25 N but its eight
    blocks span 0.96--202.35 N, CV/BFL differs by 54.5%, and maximum speed is
    0.205.  With inlet damping the mean is 106.98 N, blocks span
    -24.38--178.52 N, observer difference is 50.0%, and maximum speed falls to
    0.146.  Inlet damping reduces the velocity excursion but neither option is
    stationary; matching the tow value during an oscillation is again not
    accepted.  GPU capacity released by these runs now executes a v5 L90
    candidate with a two-convective-time body ramp, three-time warmup and six
    total convective times so the numerical-source and three-observer ledgers
    can distinguish startup forcing from force-accounting error.
63. The matched-duration U=0.04 all-face-damped sensitivity is also rejected.
    Lower Mach reduces maximum speed to 0.076 and eliminates positivity
    limiting, but the post-warmup mean remains 112.37 N, its eight blocks span
    16.07--193.77 N, and CV/BFL differs by 69.3%.  Its late 1125-step window
    crosses 87.4 N and then falls below it before returning, so lower Mach has
    changed the oscillation amplitude/phase rather than produced a stationary
    solution.  This also rules out positivity limiting as the sole cause of
    the observer split.
64. A deterministic acoustic-return regression now exercises the exact
    production chain (D3Q19 cumulant-Smagorinsky at `tau=0.500324`, incoming-
    only NEE, equilibrium-difference sponge, and `U=0.06`).  A 30-cell,
    strength-0.3 outlet layer reduces maximum returned density-perturbation
    energy from `8.86e-9` to `2.94e-12` (>3000x) on the test domain.  Thus the
    sponge/NEE implementation absorbs a planar low-amplitude pulse; the SUBOFF
    oscillation cannot be attributed to a completely ineffective outlet
    operator.  Three-dimensional domain extent, body-startup forcing and
    numerical momentum sources remain separately instrumented.
65. The v5 long-ramp checkpoint at step 6750 resolves the force-accounting
    ambiguity.  In its latest 1125-step paired window the primary and two
    auxiliary CVs agree within 0.1%; collision-plus-limiter momentum source is
    about 0.09% of force; and analytical-surface total differs from the primary
    CV by about 3.1%.  The wall-frame BFL link total is nevertheless about 23%
    higher.  Over all post-warmup samples the physical totals are 116.87 N
    (CV) and about 113 N (surface), versus 148.69 N from wall-frame link
    exchange.  Schema v6 therefore admits only against independent nested-CV
    and analytical-surface agreement; it retains BFL/CV and source-corrected-
    CV/BFL differences as explicit diagnostics but no longer treats a
    fictitious slip-wall reference-frame force as a conservative total-force
    identity.  Experiment error, stationarity, numerical-source, duration and
    three-grid gates are unchanged, so this does not promote the still-
    oscillatory L90 run.
66. Enlarging the L90 outer domain from `450x90x90` to `540x135x135`
    increases measured peak allocation from about 3.33 to 8.68 GiB but does
    not remove the low-frequency force mode.  Its post-warmup block means span
    48.99--220.95 N, stationarity range is 144%, CV mean is 119.31 N and the
    run is rejected.  Nested CVs still agree within 0.031% and numerical
    momentum source is only 0.073%, so neither AMR conservation nor simple
    transverse blockage explains the oscillation.  The normal-domain slow-
    ramp history is extended in v5 to eight convective times for diagnosis,
    while a clean v6 12000-step run now repeats the identical evolution under
    the conservative CV/surface admission contract.
67. The extended v5 slow-ramp run reaches eight convective times but remains
    rejected: mean CV resistance is 117.91 N, eight block means span
    103.60--129.75 N (22.2% range), and the finest-window CV is 117.37 N.
    Nested CV spread is 0.073% and numerical momentum source is 0.068%, so the
    remaining 35% tow error is physical/discretisation rather than a hidden
    conservation correction.  Production schema v6 also separates
    `numerical_quality_admitted` from `single_grid_admitted`: coarse and medium
    grids may enter Richardson convergence when they pass stability,
    conservation, observer and duration gates even if each is not yet within
    5% of experiment; only the extrapolated grid sequence may make the final
    experimental-accuracy claim.
68. Time convergence now distinguishes unsteady load amplitude from precision
    of the mean.  Equal-duration batch range remains reported, while admission
    uses early/late drift, linear trend and a two-sided 95% Student-t half-width
    computed from independent batch means.  This avoids the incorrect rule
    that a physically periodic force must have less than 1% instantaneous
    block range, while still rejecting a short reference crossing or a mean
    whose confidence interval is wider than the requested tolerance.  Six
    focused stationarity tests and 15 sphere/cylinder/flat/SUBOFF integration
    tests pass under the revised common contract.
69. The clean v6 three-grid campaign uses an exactly scalable integer design,
    not rounded near-equivalents.  For coarse hull lengths L90/L120/L150:
    auxiliary CV margins are `3,9`/`4,12`/`5,15`, surface-force intervals are
    30/40/50, wall-diagnostic intervals are 60/80/100, primary CV margins are
    6/8/10, and sponge widths are 18/24/30.  Domain, wake, ramp, warmup,
    averaging and total steps scale 3:4:5 as well.  The convergence assessor
    now explicitly checks report, wall-diagnostic and surface sampling
    intervals per fine-hull resolution, preventing nominally similar but
    temporally inequivalent records from entering one Richardson fit.
70. Canonical sphere validation is upgraded to schema v3.  It retains the
    complete force history but reports an explicit tail window whose duration
    is measured in sphere-diameter convective times (minimum five by default),
    so startup samples are never silently mixed into a settled mean.  A v2
    checkpoint may be continued only with an explicit migration flag, exact
    shared-physics identity, and a persisted SHA-256 of the source checkpoint;
    the migration policy and source step remain in every subsequent v3
    checkpoint/result.  Numerical-quality admission is separated from the 5%
    drag-reference gate, matching the SUBOFF grid-convergence contract.
71. Exact L120 and L150 production configurations complete real CUDA
    allocation and two-step composition smokes at 7.89 and 15.34 GiB peak,
    respectively.  Both fit the live-memory guard on the available 24 GiB
    devices; volumetric scaling places L180 beyond the safe budget.  The clean
    L90 (`12000` steps) and L120 (`16000` steps) v6 records now run concurrently
    with every spatial and temporal parameter in the 3:4 ratio.  L150
    (`20000` steps) is queued for the GPU released by the sphere benchmark.
72. Sphere validation now has its own fail-closed three-grid assessor and CLI
    with SHA-256 source records.  It requires v3 source schemas, numerical-
    quality admission, invariant Re/Mach/collision/boundary physics, and exact
    scaling of domain, CV margin, sponge, ramp, warmup, report and tail-window
    durations by radius.  Only the Richardson-extrapolated Cd may pass the
    Schiller-Naumann 5% reference gate.  The intended fresh sequence is
    R9/R12/R15 with exactly proportional integer configurations; the active
    migrated R15 run remains a single-resolution diagnostic because its old
    CV margin is not part of that new sequence.
73. Cylinder validation is upgraded to schema v3 with an explicit tail window,
    Student-t drag-mean convergence, configurable minimum shedding cycles, and
    separate numerical-quality versus Cd/Strouhal reference gates.  A new
    hashed-source three-grid assessor requires exact scaling of domain, CV,
    sponge and all time windows, invariant periodic span, and simultaneous
    Richardson convergence of both mean Cd and shedding Strouhal number.  The
    earlier D32 result remains rejected evidence; it is not relabelled under
    the new schema.
74. AFF-8 geometry admission now measures the rasterized configuration rather
    than trusting nominal CAD dimensions.  The common static-AMR resolution
    assessor records bare-hull, sail-only and fin-only cell counts, actual sail
    thickness, both cruciform-fin thicknesses and the number of halfway
    appendage links.  A convergence member requires at least 16 cells across
    the hull diameter and three across every appendage thickness; an absolute
    experimental-reference claim requires 24 and four, respectively.  At the
    exact fine-hull resolutions L180/L240/L300, measured appendage thicknesses
    are 3/4/5 cells, so L180 may be the coarse Richardson member but cannot
    make a standalone AFF-8 accuracy claim.  Output also distinguishes the
    legacy bare-hull analytical `wetted_area_lu2` from the calibrated full-
    configuration force-integration area; an AFF-8 run is rejected if its
    calibrated area does not exceed the bare-hull area.  Forty-three focused
    CAD/static-AMR production and resolution tests pass.
75. The exact L90/L120/L150 AFF-1 production sequence is encoded in
    the versioned production launcher instead of relying on copied
    terminal commands.  The launcher fixes every 3:4:5 spatial and temporal
    parameter, resumes only the matching level checkpoint, binds one physical
    GPU explicitly and can wait for one exact predecessor PID before replacing
    it.  This makes the queued L150 takeover reproducible without polling or
    terminating unrelated GPU work.
76. The fresh sphere R9/R12/R15 validation sequence is likewise encoded in
    `scripts/run_sphere_v3_equivalent_level.sh`.  Domain dimensions, radius,
    CV margin, sponge, ramp, warmup, reporting, checkpoint and tail-statistics
    windows are exact multiples of 3:4:5.  Each level may wait on the exact PID
    of its predecessor and then `exec` in place, permitting a deterministic
    one-GPU queue while preserving distinct checkpoints and source records.
77. The SUBOFF three-grid assessor now consumes the geometry-resolution gate.
    Every AFF-8 member must carry measured v6 component evidence and pass the
    convergence-member threshold, while the finest member must also pass the
    absolute-reference threshold.  Missing AFF-8 geometry evidence therefore
    fails closed even if an older result labels itself numerically converged.
    AFF-1 records remain backward-compatible because diameter is reconstructed
    unambiguously from the recorded fine hull length; this preserves the
    already-running exact AFF-1 campaign without weakening future AFF-8 claims.
78. The exact SUBOFF launcher also supports the paired AFF-8 sequence through
    `SUBOFF_HULL_TYPE=full`.  It retains the identical L90/L120/L150 mesh and
    duration ratios, writes variant-isolated `suboff-v8-aff8-*` artifacts and
    activates the measured appendage-resolution/area gates.  The default stays
    AFF-1, so the already queued L150 command and its checkpoint names are
    unchanged.  AFF-8 execution remains deliberately behind the AFF-1 result
    review rather than competing for the three occupied production GPUs.
79. The bare-hull analytical wet area is corrected from a circumference-only
    approximation to the full surface-of-revolution metric
    `2*pi*R*L integral rho*sqrt(1+(R/L drho/dxi)^2) dxi`.  Independent
    300-by-120 triangle integration agrees within 0.2%; the former expression
    was 1.16% low because it omitted bow/stern meridional slope.  Since this
    area scales the applied wall traction, the production result/checkpoint
    schema advances to v7 and the convergence assessor admits only v7 source
    records.  The currently running internally consistent v6 sequence remains
    diagnostic grid/time evidence and will not be relabelled as final physical
    validation.  Its launcher is superseded by
    the current `scripts/run_suboff_v8_equivalent_level.sh`; the later v8
    revision also corrects force-frame provenance.  Fifty-eight focused CAD,
    area, AMR and convergence tests pass.
80. The first fresh exact sphere member, R9 on 216x144x144, completes 7200
    steps with an eight-convective-time tail.  Tail mean Cd is 1.165145 by the
    control volume and 1.165198 by BFL (0.00449% observer difference); the 95%
    confidence half-width is 0.00721%, early/late drift is 0.0125%, and trend
    is 0.0227%.  It therefore passes every numerical-quality gate but, as
    expected for the coarse member, is 6.72% above Schiller-Naumann and fails
    the standalone 5% reference gate.  It is retained for Richardson fitting,
    not discarded or tuned.  The exact R12 successor started automatically on
    the same GPU, with R15 waiting on its exact PID.
81. Production scheduling is advanced without terminating any CFD solve.  The
    not-yet-started v6 L150 waiter was removed after the wet-area defect was
    proven; the running v6 L90/L120 diagnostics continue to natural completion.
    Three detached v7 launchers now wait separately on those exact two PIDs and
    the migrated sphere R15 PID, then replace them in place on physical GPUs
    1/2/3 with fresh v7 L90/L120/L150 runs.  This starts the corrected three-
    grid campaign at the earliest safe time and avoids spending GPU3 on a v6
    level that the new fail-closed assessor cannot admit.
82. The migrated sphere R15 diagnostic completes its explicit 4000-step,
    eight-convective-time tail with Cd=1.145560 by control volume and 1.145629
    by BFL.  Observer spread is 0.00603%, the 95% confidence half-width is
    0.00611%, drift/trend are 0.0104%/0.0192%, and reference error is 4.93%, so
    every single-grid v3 gate passes.  Its v2-checkpoint SHA-256 and migration
    policy remain recorded.  It is not inserted into the fresh R9/R12/R15 fit
    because CV margin, ramp, warmup and report intervals are not the exact
    3:4:5 configuration of that sequence.
83. The first detached v7 L150 handoff exposed a launcher-only import failure:
    the remote Python environment did not have the repository `src` directory
    on `PYTHONPATH`.  It failed before geometry/grid allocation and produced no
    CFD record.  Both common launchers now export the current checkout's `src`
    path and provide a no-allocation `TENSORLBM_PREFLIGHT_ONLY=1` import probe;
    two subprocess regression tests execute that probe.  The failed log is
    retained, fresh waiters replace the affected exact PIDs, and corrected v7
    L150 is running as PID 1442221 on physical GPU3.
84. A developed-flow force-closure reproduction identifies the remaining
    observer discrepancy.  BFL returned Galilean-invariant momentum exchange
    in the frame of the *numerical tangential slip velocity*; that velocity is
    a wall-model closure, not physical body motion.  Once populations become
    non-equilibrium, this wall-frame diagnostic no longer equals the discrete
    laboratory-frame population impulse required by a fixed control volume.
    The common BFL API now makes the force frame explicit and the stationary-
    body wall model uses laboratory-frame exchange after wall activation;
    wall-frame force is retained for genuinely moving-wall diagnostics and
    smooth startup.  A 100-step AMR reproduction reduces source-corrected
    BFL/CV mismatch from 8.27% to 0.000496%, while a curved non-equilibrium unit
    test proves laboratory-frame closure to 1e-11.  SUBOFF advances to schema
    v8 and flat-plate output to v4 with `link_force_frame` in provenance.  The
    just-started v7 L90/L150 jobs were stopped before their first report and
    are superseded; the original v6 L120 continues naturally.  Fifty-one focused
    BFL, wall, flat-plate, AMR, convergence and launcher tests pass.
85. Remote v8 preflight imports the intended checkout explicitly, then the
    corrected campaign starts cleanly: L90 is PID 1446335 on physical GPU1,
    L150 is PID 1446431 on physical GPU3, and the L120 launcher waits on the
    still-running v6 L120 PID 1428395 before taking physical GPU2.  v7 startup
    logs are retained with `superseded-force-frame` names; no v7 result is
    eligible for the v8 convergence assessor.
86. L90 force-history spectral evidence shows a dominant period of 2500 steps
    (1.67 hull convective times), an integrated autocorrelation time of 201
    steps, only 37.4 effective samples in the former tail, and a 3.23% standard
    error estimate.  SUBOFF v8 therefore gains an explicit final statistics
    window, matching the sphere/cylinder evidence contract.  Full histories
    remain checkpointed, while mean force, Student-t stationarity and every
    paired observer use only the declared tail; analysis-window choice is not
    part of the physics checkpoint signature, so a longer run may be resumed
    and assessed with a longer exact tail without restarting.  The three-grid
    assessor requires the resolved tail to scale with hull resolution, and the
    launcher fixes 7500/10000/12500-step tails for L90/L120/L150.  Jobs that
    had not reached their first report were superseded once more rather than
    allowing an ambiguous startup-inclusive mean into production evidence.
87. The common force-stationarity report now also quantifies temporal
    correlation instead of treating every time step as independent.  An FFT
    autocovariance gives the first zero crossing, integrated autocorrelation
    time, effective sample count and autocorrelation-adjusted standard error;
    the resolved spectral peak reports dominant period and its power fraction.
    These are diagnostic additions—the conservative Student-t batch, drift and
    trend gates are not relaxed.  A repeated-sample regression proves that the
    effective count decreases appropriately, and 25 focused force/sphere/
    cylinder/flat/SUBOFF tests pass with the expanded report.
88. A remote launcher regression catches one stale deployed file: the SUBOFF
    v8 launcher had the import-only preflight, while the remote sphere launcher
    was still its preceding revision.  The attempted test process timed out
    before producing a checkpoint/result and was terminated by the test
    harness; an exact process audit found no residual sphere PID.  The sphere
    launcher is now synchronized, direct remote preflight resolves the intended
    `src/tensorlbm/__init__.py`, and both launcher subprocess tests pass on the
    Wuxi host.  Existing CFD PIDs and services were untouched.
89. Production-scale v8 confirms the force-frame correction before the
    statistics window opens.  At L90 step 3000 (full wall activation), the
    instantaneous nested-CV resistance is 121.456 N and laboratory-frame BFL
    plus wall stress is 121.613 N: raw difference 0.156 N or 0.129%.  The
    corresponding phase of v6 using artificial-slip wall-frame exchange was
    about 41.7 N apart.  Numerical-source pairing begins only after the step
    4500 warmup, so this checkpoint is conservation evidence, not an admitted
    drag mean; the run continues unchanged into its declared tail.
90. Closure remains valid through the large startup oscillation rather than
    only near one favourable crossing.  At steps 3375 and 3750 the CV forces
    are -176.448 N and 211.348 N, while BFL plus wall stress gives -176.384 N
    and 211.464 N: absolute residuals 0.064 N and 0.116 N.  This sign-changing
    test rules out accidental relative agreement at step 3000 and isolates the
    remaining oscillation as flow/boundary physics to be time-averaged, not a
    broken force observer.
91. Cylinder validation is prepared as a fresh v4 R9/R12/R15 sequence rather
    than reusing the rejected D32 legacy record.  Checkpoint identity now
    excludes the analysis-only tail choice, output records the requested and
    resolved tails, and force-frame provenance is explicit.  Absolute Re=100
    admission requires at least 5D upstream, 10D downstream and 10D lateral
    centre distance; the exact launcher uses 40R x 40R domains, giving
    6D/14D/10D.  For radii 9/12/15 it scales steps 54000/72000/90000, warmup
    31500/42000/52500 and tails 22500/30000/37500 exactly 3:4:5.  Each tail
    spans about 12.3 expected shedding cycles, exceeding the eight-cycle gate.
    Thirteen cylinder/convergence/launcher tests pass.  The sequence is queued
    behind the exact sphere chain and therefore does not contend for its GPU.
92. The local one-GPU queue is concrete and PID-linked: cylinder R9 waiter
    4004293 follows the exact sphere R15 process/waiter 3855756; cylinder R12
    waiter 4004961 follows R9; cylinder R15 waiter 4005447 follows R12.  Each
    launcher uses `exec`, so its PID remains stable when computation begins and
    downstream levels cannot start early.
93. Flat-plate validation is also rebuilt as a strict v4 space-time sequence.
    The earlier L256/L384/L512 evidence used 30000 steps and 15000-step warmup
    on every grid, corresponding to unequal convective durations, so its good
    spatial fit is retained as historical evidence but not treated as the new
    production proof.  v4 records an explicit analysis tail and requires total,
    warmup, ramp, tail, report and wall-diagnostic intervals divided by plate
    length to be invariant.  The exact launcher uses total steps
    32000/48000/64000, warmup and tails 16000/24000/32000, and proportionally
    scaled exchange height, sponge, CV and diagnostics.  Twelve focused
    flat-plate/convergence/launcher tests pass; this sequence will follow the
    cylinder chain on the same local GPU.
94. The flat-plate queue is now PID-linked as well: L256 waiter 4015500 follows
    cylinder R15 waiter/process 4005447, L384 waiter 4016131 follows L256, and
    L512 waiter 4016750 follows L384.  Thus sphere, cylinder and flat-plate
    validation form one deterministic GPU queue with no polling race or
    resource contention.
95. The active v8 L90 checkpoints provide the first long-tail production audit
    of the laboratory-frame correction.  At step 7500, across all 3000
    post-warmup steps, primary CV and BFL-plus-wall-stress means are 110.754 N
    and 110.859 N, a 0.0943% difference; the stepwise residual has 0.107 N RMS
    and 0.180 N maximum magnitude.  At the 100 explicitly paired coarse steps,
    adding the measured numerical momentum source reduces the mean CV/BFL
    difference from 0.0929% to 0.00134%.  Auxiliary CV margins 3/6/9 agree to
    within 0.057%, while the independent surface-pressure integration remains
    9.99% above the CV and is therefore diagnostic only.  The same
    checkpoint's incomplete-tail resistance is 110.754 N, 26.72% above the
    87.4 N experiment: conservation is now demonstrated, but physical and
    statistical convergence are not.  A reusable checkpoint auditor groups
    fine substeps by coarse step and reports history, raw/source-corrected,
    nested-CV and surface-observer closure without promoting an in-progress
    checkpoint to validated CFD evidence.
96. Force decomposition narrows the remaining physical error.  In the L90 v8
    step-7500 checkpoint, mean modeled wall shear is 76.30 N and the residual
    CV contribution after subtracting it is 34.45 N.  In the independently
    running L120 v6 diagnostic at step 12000, those values are 70.93 N and
    25.08 N.  For context only, ITTC-1957 at the 13.21-million physical Reynolds
    number and the 5.9227 m2 analytical wetted area gives 78.41 N friction,
    leaving 8.99 N between that correlation and the 87.4 N total experiment.
    Thus neither grid supports increasing wall shear to match total drag; the
    dominant excess decreases with refinement in the CV remainder associated
    mainly with pressure/form drag.  The v6 link-pressure observer is excluded
    because it predates the laboratory-frame correction.  The checkpoint
    auditor now persists force decomposition, wall-model y+ applicability and
    AMR/positivity quality alongside observer closure so this diagnosis is
    repeated identically at every resolution.
97. The wall exchange-height contract is now resolution-planned rather than
    inherited blindly from the Re=1e6 flat plate.  ITTC-1957 plus lattice
    similarity predicts y+=5855 for the current fixed `y/L=0.01171875` at the
    SUBOFF physical Re=13.21 million; the L90 runtime mean y+=5753 independently
    confirms the estimate.  Because L90/L120/L150 scale exchange distance with
    hull cells, their physical sampling location and expected y+ do not
    converge even though geometry does.  A common estimator now consumes the
    actual finest-level body resolution and exchange distance, while a planner
    reports the minimum resolution and number of additional 2:1 wall-normal
    levels needed for a declared y+ target.  With the exchange sampler's
    approximately one-finest-cell minimum, L150 requires one additional local
    2:1 surface level: effective L=600 predicts y+=833, inside the existing
    100--1000 engineering target.  This is a quantitative design requirement
    for the post-campaign surface-shell refinement, not a reason to alter the
    active three-grid trajectories.
98. The conservative static-block runtime now supports an arbitrary strictly
    nested sequence of 2:1 blocks instead of only one parent/child interface.
    A three-level step advances L0/L1/L2 exactly 1/2/4 times, time-interpolates
    every child ghost layer, rescales non-equilibrium stress at each level,
    restricts finest-to-coarsest, and applies an independent face-local kinetic
    reflux ledger at both interfaces.  Mixed replacement-only/conservative
    hierarchies and broken tau chains fail closed.  A 50-root-step D3Q19 MRT
    non-equilibrium test remains finite and positive with 6.00e-8 relative
    root mass drift and 1.82e-12 maximum reflux residual; exact scheduling,
    uniform moving equilibrium, cell savings and checkpoint-level state
    relinking have dedicated regressions.  This establishes the common
    multi-level mechanism required by the y+ plan, but does not yet admit a
    curved-wall SUBOFF surface shell; geometry ownership and force closure on
    the added interface remain mandatory integration evidence.
99. SUBOFF now has a matching second-level geometry planner.  Its inner box is
    expressed in the allocated outer-level coordinates (including the ghost
    layer), fails if the requested wall/wake margin is clipped, and regenerates
    analytical CAD at 4x coarse resolution rather than repeating level-1
    voxels.  An exact L150 AFF-1 planning probe with four parent-cell wall
    margin and eight-cell wake allocates an 86x86x634 L2 physical block,
    increases the complete hierarchy to 25,231,128 cells, predicts 22.16 GiB
    peak by the measured 943 B/cell model, and resolves hull length/diameter as
    600/70.01 cells while retaining 97.66% savings versus a uniform 4x domain.
    The analytical L2 body contains 1,830,129 cells.  This fits the intended
    high-memory GPU class on paper, but an actual CUDA preflight remains
    mandatory because the empirical estimate is not an allocation guarantee.
100. A three-level curved-wall integration test now assigns geometry and force
     exclusively to L2 while L0/L1 remain parent transport levels.  Over three
     root steps (12 finest substeps), a BFL sphere with Musker/Guo wall stress
     is evaluated only on L2; at every substep the independent finest-level
     control volume agrees with laboratory-frame BFL plus applied wall stress
     within 0.002%.  Both kinetic interfaces simultaneously retain maximum
     reflux residual below 2e-10.  This closes the common scheduling,
     ownership and instantaneous-force ledger for a curved body.  SUBOFF still
     requires its own short three-level reproduction because its analytical
     hull, area weights, exchange sampler and elongated wake are more demanding
     than the manufactured sphere.
101. The exact v8 L90 trajectory completes all 12000 steps and is formally
     rejected rather than tuned: its declared 7500-step tail gives 115.121 N,
     31.72% above the 87.4 N experiment.  Eight batch means span
     91.43--134.52 N, with 6.57% trend and 10.00% 95% confidence half-width,
     so stationarity fails independently of reference error.  Conservation
     evidence remains strong: raw CV/BFL means differ by 0.0912%, numerical
     source is 0.0912%, source-corrected paired observers differ by 0.000355%,
     and nested CV spread is 0.0541%; positivity and reflux gates pass.  The
     separate surface-pressure path is 11.37% high and fails its 5% diagnostic
     gate.  Mean runtime y+=5750 agrees with the new 5855 prior.  Post-hoc
     analysis of the immutable raw tail finds a 3750-step (2.5-convective-time)
     dominant period, 295-step integrated autocorrelation time, only 25.4
     effective samples and 4.82% autocorrelation-adjusted standard error; the
     declared tail spans only two dominant cycles.  The immutable result and
     separate audit are preserved locally and remotely.  The checkpoint auditor
     performs this expanded analysis because the process loaded the preceding
     stationarity module before those fields were added.
102. The independent three-level SUBOFF smoke runner passes its first real CUDA
     L120 allocation on the freed physical GPU1.  L0/L1/L2 contain 13,570,112
     allocated cells versus 552,960,000 for a uniform 4x domain (97.55%
     saving); measured peak is 7.808 GiB versus the deliberately conservative
     11.918 GiB estimate.  Exact L2 CAD resolves length/diameter as 480/56.01
     cells and owns all geometry and force.  At the first fully activated root
     step, source-corrected CV and BFL-plus-wall-stress differ by 0.000286%,
     the two reflux residuals are 7.28e-12 and 5.46e-12, no reflux direction or
     population is limited, no wall sample is rejected, and all states remain
     finite.  Mean exchange y+=1303, substantially below the one-level ~5750
     but still above the 1000 planning target.  The startup resistance
     magnitude is explicitly not assessed: this artifact admits allocation and
     integration only, with accuracy/time/grid validation all false.
103. The same integration advances an exact L150 three-level root step after
     calibrating the memory model explicitly from L120 and adding 20% margin
     (742 B/cell).  The original 943 B/cell gate first rejected the case after
     mask allocation, so no unsafe allocation occurred.  The calibrated gate
     reserves 1 GiB and predicts 17.436 GiB; measured peak is 15.010 GiB on the
     23.69 GiB card.  The hierarchy has 25,231,128 cells, 97.66% savings versus
     uniform 4x, exact L2 length/diameter 600/70.01 cells and 1,830,129 body
     cells.  Source-corrected force closure is 0.000147%; interface residuals
     are 3.64e-12/1.64e-11, with no limiting, rejection or non-finite state.
     Mean y+=1088, close to but still 8.8% above the 1000 planning target; the
     difference from the 833 ideal prior is explained by the curved-link
     sampler's actual `max(exchange, y1+0.5)` distance.  As with L120, this is
     allocation/integration evidence only and makes no drag-accuracy claim.
104. The fresh exact sphere R12 sequence completes 9600 steps with a declared
     3200-step (8-convective-time) tail.  Cd_CV=1.14960034 and
     Cd_BFL=1.14966316 differ by 0.00546%; eight block means span only 0.0221%,
     trend is 0.0205% and the 95% half-width is 0.00652%.  It is numerically
     admitted but correctly fails physical admission: error against
     Schiller--Naumann is 5.30069%, just above the 5% gate.  Re-analysis of the
     immutable tail gives 944-step integrated autocorrelation time, 3.39
     effective samples and 0.00403% autocorrelation-adjusted standard error;
     the residual 0.3007-point miss is therefore spatial/model bias, not mean
     uncertainty.  Exact R15 has automatically started from a fresh state and
     the cylinder sequence remains PID-gated behind it.
105. Extending the L150 three-level CUDA smoke from one to 20 fully activated
     root steps remains finite and positive at the same 15.010 GiB peak.  The
     maximum source-corrected force-observer difference is 0.00206%, maximum
     outer/inner reflux residuals are 1.86e-9/7.45e-9, and no reflux direction,
     population or wall sample is limited/rejected.  Mean y+ stays near 1100.
     Startup force decays from 13.27 kN to 3.00 kN, demonstrating why this
     deliberately impulse-started integration test cannot be interpreted as
     resistance; its sole admission remains multi-level stability,
     conservation, wall ownership and allocation.  A production nested run
     must add a long smooth activation, checkpoint/restart and the same
     convective-time statistics used by v8 before any accuracy comparison.
106. The three-level runner now closes the first of those production gaps with
     atomic multi-level checkpoints.  The checkpoint stores L0/L1/L2
     populations, stepwise paired-force evidence and accumulated limiter,
     reflux and wall-sampling maxima; restore validates a physics/grid
     signature and relinks shared parent/child tensors before advancing.
     Requested final steps remain outside the signature so a trajectory can be
     extended without relabelling its physics.  A CPU 1-step checkpoint resumed
     to step 2 reproduces both records and passes the same integration gates.
     Smooth activation is supported, and force-closure admission now considers
     only fully activated steps, preventing the intentional moving-wall-frame
     startup diagnostic from contaminating laboratory-frame closure evidence.
107. The independent v6 L120 diagnostic finishes and is rejected despite an
     apparently favourable reference error.  Its full 10000-step post-warmup
     CV mean is 91.207 N (4.36% high), but eight blocks span 77.80--104.83 N,
     half-mean drift is 10.73%, trend 14.32% and 95% half-width 9.78%.  The
     old 1500-step sparse paired window averages only 85.34 N, demonstrating
     that tail choice and sampling phase can reverse the apparent error while
     the low-frequency mode remains unresolved.  Its pre-v8 wall-frame BFL and
     surface paths fail observer gates and are not repaired post hoc.  The
     immutable negative artifact is retained; exact v8 L120 has automatically
     started on the same physical GPU with a uniform 10000-step explicit tail
     and laboratory-frame force ledger.
108. CUDA checkpoint/restart is now exercised rather than inferred from the
     CPU regression.  A three-level L90 run writes a 428 MiB atomic checkpoint
     after step 1, a fresh process restores all three tensors
     (`450x90x90`, `358x48x48`, `396x60x60`, each D3Q19) and advances to step
     2.  The resumed artifact retains records `[1,2]`, declares
     `resumed_from_step=1`, closes source-corrected force within 0.000126%, and
     keeps both reflux residuals below 1.82e-12.  Measured peak is 3.366 GiB.
     This admits restart mechanics and state relinking; it does not turn the
     two-step startup into a resistance result.
109. The nested runner now carries the same anti-cherry-picking analysis
     contract needed before a long trajectory: declared warmup and final
     statistics windows, root-step convective durations, eight-block
     stationarity, FFT autocorrelation/effective sample count, reference error,
     and a fail-closed single-grid-candidate decision.  Reporting and wall y+
     reductions have independent cadences, while force and both reflux ledgers
     remain every-step; the outer non-equilibrium boundary is reimposed after
     sponge damping.  Short smoke tests continue to report integration-only,
     because they fail duration/stationarity even when force closure passes.
     Final step count and analysis-window choice remain outside checkpoint
     physics identity, allowing longer unbiased resumes from the same state.
110. The long-run evidence contract now adds two auxiliary finest-level
     control volumes and independent surface-pressure integration at a declared
     cadence.  Auxiliary means are paired with the primary CV at identical
     timestamps and must agree within 1%; surface pressure plus simultaneous
     wall shear must agree with the paired CV within 5%.  These sparse
     observers are checkpointed with each step record but are not recomputed
     every finest substep, avoiding unnecessary GPU synchronisation.  Both
     gates are mandatory for a nested single-grid candidate, closing the same
     sampling-phase and surface-observer loopholes found in the one-level
     campaign.
111. AFF-8 appendage-link ownership is moved out of the one-level example into
     the common SUBOFF geometry module.  The axisymmetric body retains
     analytical BFL intersections, while sail/fin links outside that body
     receive an audited q=0.5 fallback and an exact count; malformed D3Q19
     link fields fail closed.  The one-level v8 path is refactored to call the
     common function without changing its physics.
112. The three-level runner now accepts both AFF-1 and AFF-8.  At L2 it
     regenerates bare body, sail and cruciform fins from analytical CAD,
     measures actual component cell counts/thicknesses and appendage halfway
     links, uses gradient normals for the full geometry, and transfers the
     bare-body analytical area calibration to the complete wet surface.  A CPU
     AFF-8 curved-wall smoke exercises the full collision/BFL/wall/CV/surface
     chain and records nonzero sail, fin and appendage-link evidence while
     keeping physical validation false.  No AFF-8 GPU campaign is launched
     before AFF-1 long-run behaviour is understood.
113. A deterministic three-level campaign launcher fixes L90/L120/L150 at the
     same 3:4:5 space-time ratios as v8: 12000/16000/20000 total steps,
     4500/6000/7500 warmup, 7500/10000/12500 final tails, proportionally scaled
     activation/report/surface/y+ cadence and checkpoints, constant one-finest-
     cell exchange distance, and identical inner margin/CV ownership.  It
     supports AFF-1/AFF-8, resume, exact physical-GPU selection, PID chaining
     and import-only preflight.  Launcher subprocess coverage passes alongside
     the existing sphere/cylinder/flat/v8 launchers.
114. The three-level queue is now concrete on physical GPU1.  Active L150 PID
     1471212 runs first; exact L90 waiter PID 1473702 is gated on it, and exact
     L120 waiter PID 1473870 is gated on L90.  Both waiters use the tested
     launcher and stable `exec` handoff, so no second calculation can enter the
     card early.  Physical GPU2 and GPU3 continue one-level v8 L120/L150,
     while the local GPU continues sphere R15; no service or port is touched.
115. Nested results now have a separate fail-closed three-grid assessor rather
     than being coerced into the one-level v8 schema.  It requires source
     integration/duration/stationarity/nested-CV/surface gates, exact identity
     of physical and finest-cell wall/CV contracts, invariant domain/outer
     mesh/time ratios, all measured geometry members, a finest absolute
     geometry, monotone fitted spatial order/error/RMS and an extrapolated
     experiment error within 5%.  A manufactured second-order sequence is
     admitted; changing only exchange distance or failing one surface observer
     blocks physical validation.  A CLI writes the immutable convergence
     artifact when all three queued records exist.
116. Future nested records now preserve measured exchange distance and
     min/mean/max y+ at the cadence level, plus final-tail reductions.  This
     distinguishes the requested one-cell exchange from the curved-link
     sampler's actual `max(exchange,y1+0.5)` location and prevents interpreting
     an ideal ITTC prior as runtime applicability evidence.  The active L150
     process predates this output-only addition and remains unchanged; queued
     L90/L120 will load it when their processes start.
117. Adding AFF-8 changes the nested output/checkpoint semantic version to v3
     before queued L90/L120 start.  The already-running L150 remains immutable
     v2, whose runner hard-coded AFF-1 but omitted `hull_type` from configuration.
     The convergence assessor accepts that legacy source only when its measured
     geometry explicitly says `bare_hull`, normalises all source identities and
     requires the three hull types to agree.  It never substitutes a default or
     accepts an unknown/mixed geometry.
118. v3 restart has a narrow audited bridge for the active L150 v2 checkpoint:
     only `bare_hull` may migrate, and only when the stored signature equals
     the current signature after removing exactly the newly explicit hull type
     and changing version 3 to 2.  Every other physics/grid/cadence field must
     match byte-for-byte; AFF-8 and partial matches fail.  The resumed output
     records `resumed_legacy_v2_checkpoint=true`, and its next atomic save is
     native v3.  A regression rewrites only those two legacy fields and proves
     1-to-2-step recovery.
119. The first production-length nested L150 attempt is rejected at root step
     625 because all force observers are non-finite.  Its exact PID alone is
     terminated and the log is retained; no resistance value is extracted.
     The originally queued L90/L120 jobs also exposed an environment defect:
     the launcher assumed a repository `.venv` that does not exist on the
     Wuxi node.  Interpreter selection now honours `TENSORLBM_PYTHON`, then a
     local venv, then `python3`, and rejects a non-executable path explicitly.
120. A reusable low-frequency population-health observer now reports, per
     hierarchy level, population extrema, density extrema and peak lattice
     speed as host scalars.  The nested runner also fails immediately when the
     already-paid positivity reduction reports a non-finite collision or wall
     state, naming root step, level and stage instead of continuing to a later
     `NaN` force report.  Health records survive checkpoints, while a zero
     cadence keeps the production hot path unchanged.
121. A matched L150 diagnostic localises the instability before `NaN`: through
     step 175 all levels remain positive, but the finest level reaches peak
     speed 0.109.  At step 200 its positivity limiter is active, density spans
     0.866--1.366 and peak speed reaches 0.455.  The unbounded fine-to-coarse
     reconstruction then creates negative populations on L1 by step 225 and
     on L0 by step 250; at step 350 the finest density spans 0.036--64.0 and
     peak speed is 1.254.  This trajectory is physically invalid and is
     stopped there.  The causal order identifies AMR transfer amplification,
     rather than the still-stable one-level wall/open-boundary path, as the
     first repair target.
122. Three explicit common transfer controls are implemented for the matched
     A/B run.  Fine-to-coarse restriction can project non-equilibrium content
     onto the six second-order Hermite stress moments before convective
     rescaling, filtering higher kinetic modes without empirical force tuning.
     The one-cell ghost shell can use cell-centred trilinear interpolation;
     its donor plan is cached and samples only the shell instead of allocating
     a full fine-block temporary every substep.  Finally, restriction can apply
     the moment-preserving positivity limiter before the parent consumes the
     state, with per-interface limited fraction and minimum alpha persisted as
     admission evidence.  All three remain explicit physics/numerics options,
     old unfiltered checkpoints have a narrow baseline-only migration path,
     and 36 focused transfer/hierarchy/runner tests pass.
123. The matched transfer-stabilised L150 A/B confirms that exchange repair is
     necessary but not sufficient.  Trilinear shell interpolation, second-
     order restriction filtering and pre-replacement positivity keep L0/L1
     populations non-negative, yet L2 still reaches peak speed 0.439 at step
     200 and 0.895 at step 250.  L120 behaves similarly.  Both diagnostics are
     stopped and retained; their force histories are not used.  Per-interface
     limiter fraction/minimum alpha and reflux residual are now emitted at the
     same cadence as per-level population health.
124. Doubling the inner transverse buffer from 8 to 16 finest cells does not
     change the pre-instability trajectory: L150 reaches peak speed 0.438 at
     step 200 and 0.893 at step 250.  An L90 run with the same expanded buffer
     finishes 700 steps only because positivity repeatedly clips the state;
     it first exceeds the 0.3 weakly-compressible speed gate at step 200 and
     reaches transfer-limiter fractions 0.39%/0.76%, so it is rejected.  The
     planner now persists exact wall/downstream buffer thickness in parent and
     finest cells rather than leaving “enough margin” implicit.
125. Peak-speed localisation shows the causal path.  Through step 150 the L2
     maximum lies on a near-wall fluid node at the bow.  At step 200 the peak
     has propagated into the near-tail/wake region, 33--34 cells from the
     allocated boundary, where it reaches 0.796.  Disabling wall-stress Guo
     forcing changes that value only from 0.79636 to 0.79671; therefore the
     instability is driven by BFL impermeability/startup plus insufficient
     resolved collision dissipation, not the wall-law shear source or a
     coarse/fine boundary placed too close to the body.
126. A fail-closed speed gate is added to the health cadence and future
     production launchers enable it.  The limit is explicit (default 0.3,
     versus inlet 0.06), and a violation names the exact root step after
     persisting the level/interface health line.  A read-only nested checkpoint
     auditor independently loads all hierarchy tensors on CPU; the active
     unfiltered L90 checkpoint at step 1500 is finite but has already limited
     0.6001% of cells, with peak speeds 0.176/0.150/0.167 and therefore remains
     ineligible despite later force recovery.
127. Collision isolation rejects several tempting labels.  Raising Smagorinsky
     `Cs` from 0.05 to 0.15 barely changes the step-200 peak (0.796 to 0.786).
     Entropic KBC with eight gamma iterations is worse and crosses the gate at
     step 75 (0.561).  Reducing the cumulant bulk relaxation rate to 0.5 crosses
     at step 175 (0.396).  Lower resolved Reynolds supplies the missing shear
     diffusion monotonically: Re=20000 still peaks at 0.615 by step 200,
     Re=10000 recovers to 0.122 at step 200 but jumps to 0.609 at step 225,
     while Re=5000 remains clean through step 225 with peak 0.0783, density
     0.925--1.041 and zero collision/transfer limiting.  This is stability
     evidence only; permanent Re=5000 is not accepted as the target physics.
128. Static and arbitrary-depth AMR now accept an optional instantaneous tau
     pair/chain.  Every collision call, coarse-to-fine ghost reconstruction
     and fine-to-coarse non-equilibrium rescaling uses the same validated
     convective chain; inconsistent dynamic tau values fail before advancing.
     A common continuation schedule ramps inverse Reynolds number with a
     raised-cosine profile and zero endpoint derivatives.  This enables the
     production strategy “safe Re=5000 startup, then gradual return to
     Re=100000” without changing wall-law physical Reynolds or silently
     mismatching interface physics.  Forty-two focused continuation, AMR and
     runner tests pass.
129. A matched L90 continuation pilot validates the startup path through the
     target collision Reynolds number.  It runs at Re=5000 through root step
     300, follows the raised-cosine inverse-Re schedule to Re=100000 at step
     600, and remains finite through step 700.  Across 28 health snapshots the
     maximum speed is 0.089989, density remains in 0.914403--1.041628, both
     transfer interfaces have zero limited cells, the collision positivity
     limiter remains inactive, and maximum reflux residual is
     `5.820766e-11`.  This is startup-stability evidence, not a drag result:
     the 3000-step wall ramp is intentionally incomplete and the last force
     closure error is 49.96%.  Immutable log/JSON SHA-256 values are
     `a78e4683bf81157a2ad6c0e20aa1229b1558b30b4f977b16b41d2e7a1ed25e76`
     and `220a77d7c3433395a4ac1867b742c5b802d86f6805ad979f88a71b86775660e7`.
130. Force admission is now continuation-aware.  Wall-activated samples whose
     instantaneous collision Reynolds number differs from the requested
     target cannot enter force closure, stationarity, experimental-error or
     duration statistics.  Output separately records wall-activated and
     target-Re sample counts plus an explicit target-Re-reached flag.  Health
     records persist instantaneous Reynolds, the full level tau chain and the
     target-Re flag, making a resumed ramp independently auditable.
131. A 12000-step L90 production continuation is active on the released Wuxi
     GPU.  It preserves the verified Re=5000-to-100000 startup, fully activates
     the Musker wall source by step 3000, warms through step 4500, and reserves
     7500 target-Re steps (five convective times) for statistics.  Its first
     health snapshot at step 60 is finite with peak speed 0.060803 and no
     collision, restriction or reflux limiting.  In parallel, the L150 fixed
     Re=2000 diagnostic remains clean through step 250 with peak speed 0.07044
     and zero transfer limiting; this run only determines a safe high-
     resolution startup state and is not eligible for resistance validation.
132. The continuation path is now a reproducible common campaign rather than
     a hand-written command.  The legacy v3 launcher retains its cold-start
     defaults, while the v4 wrapper selects resolution-specific safe startup
     Reynolds numbers, a smooth continuation interval, larger finest-level
     wall/wake buffers, trilinear ghost interpolation, second-order
     restriction regularisation and transfer positivity.  It writes distinct
     v4 checkpoints, remains fully environment-overridable, and passes launcher
     argument-capture tests at L150.
133. A geometry-independent, fail-closed nested-startup assessor now separates
     numerical startup evidence from force validation.  It audits population
     finiteness and positivity, density, the weakly-compressible speed gate,
     collision/transfer limiter fractions, reflux conservation and actual
     arrival at the requested collision Reynolds number.  Applied to the L90
     pilot it reports `startup_stability_pass`, 101 target-Re steps, peak speed
     0.089989, minimum population 0.0160873 and zero limiter use.  The persisted
     assessment has SHA-256
     `369f65b998332b9a73bcd407dbfd9acd92a1a6d7866b9f776e3949691768961d`;
     by contract this status makes no resistance-accuracy claim.
134. The L150 fixed-Re=2000 startup diagnostic completes 700 steps and passes
     the same strict startup assessment.  Across 28 snapshots its peak speed
     is 0.0726924, density stays in 0.936876--1.049194, minimum population is
     0.0167503, both AMR interfaces remain un-limited and maximum health-cadence
     reflux residual is `5.820766e-11`.  Log, result and assessment SHA-256
     values are respectively
     `cc4960c9ee16c1e88ccdf10b2148194cc75dd7111b717356fcc73e76c6b4de54`,
     `0681d2c6a0c45417a62ceec69deef9ba639b5c00265085358762a0e8f84c9faa`
     and `904cb845c1a1ee0f4b1617a6b8b5fea8dc68ed2446a3aa8f8771972a472c5a11`.
     The queued L150 v4 production run then starts on that released GPU with
     Re=2000 through step 500, continuation to Re=100000 at step 1000, a
     5000-step wall ramp and health checks every 100 steps.  Low-Re force from
     the diagnostic remains explicitly excluded from physical validation.
135. Continuation validation now reports and gates three distinct durations:
     total trajectory time, time actually spent at the target collision
     Reynolds number, and time with both target Reynolds number and a fully
     activated wall.  The selected force window remains a fourth, stricter
     duration.  Production admission requires at least 7.5 target-Re
     convective times in addition to eight total and five sampled convective
     times, so a long low-Re startup cannot satisfy the physical-duration gate.
136. The unfiltered L90 continuation reveals a later coarse/fine-interface
     mode that the 700-step pilot could not expose.  It stays mild through
     approximately step 1380, then the finest-level peak repeatedly approaches
     the allocated block boundary rather than the hull.  At step 2250 the
     hierarchy is still finite, but peak speed has oscillated up to 0.191,
     cumulative collision positivity limiting is `3.09161e-5`, maximum reflux
     residual is `1.78814e-7`, and instantaneous force observers do not close.
     The trajectory is stopped at step 2280 and retained as an unfiltered
     baseline, not force evidence.  Its immutable rejected log and step-2250
     checkpoint SHA-256 values are
     `f9364a57d01e00cd30e6a62d6276ce5ec26e4c161b25f3b571f0e83b1c7c696a`
     and `78dbd82d2ff256105fcb7a7d75e80f73036bc7e421665661b7f78b2aaa9272a1`.
137. A common AMR transition-shell filter is implemented for D3Q19 and D3Q27.
     A raised-cosine shell acts in physical fine cells adjacent to a resolution
     transition.  The first prototype damped the complete non-equilibrium
     population; a low-amplitude hydrodynamic pulse showed a 4.6% increase in
     interface density error, so that design was rejected at SUBOFF step 720
     and its log retained with SHA-256
     `28df39102ac46cea3e2c696cd91a2070c0b10ce87db02dc57742c79788da2029`.
138. The replacement filter projects the non-equilibrium population onto its
     complete symmetric second-order stress and damps only the residual
     higher-order kinetic modes.  Cell density, all three momentum components
     and all six viscous-stress moments are invariant.  On the same 24-step
     hydrodynamic pulse, `width=4`, `strength=0.2` changes interface density RMS
     from `2.660644e-6` to `2.662313e-6` (0.063%) while slightly reducing the
     refined-region density and velocity errors; mass drift remains
     `3.051718e-7` and the benchmark stays admitted.  Filtering is forbidden
     if its shell intersects a solid or adjacent wall-fluid cell.  Width and
     strength remain explicit checkpointed parameters and zero by default.  A
     fresh matched L90 v6 kinetic-filter A/B is active against the immutable
     unfiltered trajectory before production promotion.
139. The one-level equivalent-L120 campaign completes 16000 root steps.  Its
     final 10000-step mean resistance is 86.67285 N, 0.83198% below the 87.4 N
     experiment.  Independent BFL-plus-wall-stress and primary control-volume
     means differ by 0.16528%, the two auxiliary control volumes differ by at
     most 0.09457%, numerical momentum source is 0.16526%, maximum positivity
     limiting is `4.7338e-5`, and maximum reflux residual is `2.3842e-7`.
     Nevertheless the run is correctly rejected: the block-mean range is
     36.67%, autocorrelation-adjusted 95% confidence half-width is 9.90%, and
     the separate normal-pressure surface integral differs from the paired
     control volume by 45.25%.  The result is strong force-scale evidence but
     not a converged validation point.  Result/log SHA-256 values are
     `9fbcede94bebe8ffa9632c0fdca704278b1aa03fcce4dc384a2e3b69d977b950`
     and `89ac1b2f12fbe66d35489c50ffb4aa2ba8e8e0ab587ff3ba666688b5b6c3c010`.
     Its released GPU automatically starts the L120 nested continuation
     baseline with 80-step health cadence.
140. The failed third observer is isolated without changing the flow solution.
     Re-evaluating the final L120 checkpoint with the original first-node
     surface pressure gives 75.15 N pressure drag, versus a 15.04 N BFL
     pressure mean.  Exact quadratic wall-normal extrapolation reduces the
     final snapshot to 18.87 N with the near-wall reference and 17.19 N with
     the inlet free-stream reference.  A manufactured quadratic pressure field
     confirms the three-point extrapolation recovers the analytical wall value
     exactly.  The v9 campaign therefore changes only this independent
     observer to inlet-reference quadratic extrapolation; collision, boundary,
     wall stress, BFL and control-volume forces remain bitwise on the v8 path.
     The v9 L120 time-history A/B is queued behind the active one-level L150
     run and uses distinct checkpoints.
141. AMR interface filtering is now geometrically isolated from every
     independent control-volume force observer.  A common half-open-box
     clearance assessment accounts for the physical ghost boundary, filter
     width and the radius-one D3Q19/D3Q27 streaming source stencil; the nested
     SUBOFF runner fails closed during preflight if any primary or auxiliary
     flux surface can sample a filtered cell.  Exact L90 planning finds only
     three cells between the largest auxiliary CV and the physical fine-block
     interface, so its largest force-admissible filter is width two.  The
     active width-four L90 experiment remains a stability-only A/B and cannot
     be promoted as force evidence.  The guard, planning provenance and public
     API are covered by 69 focused AMR/SUBOFF regression tests.
142. The width-four, strength-0.2 kinetic-only L90 A/B is rejected at step
     1560.  Against 26 exactly aligned health records, peak speed is only 3.64%
     below the unfiltered baseline (`0.18367` versus `0.19061`), while the
     candidate already reaches a negative minimum population
     (`-4.4671e-5`) and triggers collision positivity limiting.  The exact-PID
     job was stopped and its immutable log SHA-256 is
     `73d862e4b6b925f555931def2bcf6f6c91106f3b743e85c667f6eb58f1541d2e`.
     A force-admissible width-two, strength-1.0 candidate is now running on the
     released GPU after passing the uniform-fine pulse benchmark; it preserves
     mass and changes interface density RMS by 0.24%, so it is a materially
     stronger but still hydrodynamically audited test.
143. Coarse-to-fine prolongation now has an independent second-order Hermite
     regularization option.  It removes non-hydrodynamic ghost modes before
     every fine substep ghost fill (and during initial child allocation), while
     preserving density, momentum and the complete viscous-stress tensor.
     The option is default-off, explicit in launchers, checkpoints and
     three-grid identity checks, and is separate from fine-to-coarse
     restriction regularization.  In the 24-step uniform-fine pulse benchmark,
     prolongation regularization alone changes interface density RMS by 0.30%
     with identical mass drift; combined with the force-safe width-two,
     strength-1.0 kinetic filter it remains admitted, slightly improves
     refined-region velocity error, and changes refined density RMS by 0.23%.
144. The legacy uncontinued nested L90 v3 trajectory finishes all 12000 steps
     and is retained only as negative evidence.  Its 7500-step mean is
     96.7900 N (10.74% high), but block means span 93.73%, the adjusted 95%
     confidence half-width is 25.60%, the surface observer differs by 25.84%,
     and positivity limiting reaches 0.6001% without cadence-resolved health
     provenance.  It therefore fails independently of reference proximity.
     Result/log SHA-256 values are
     `73660fae62d4930db6d65e99eeb30f558d1ac21c7263f4df7ab4653a51e667ad`
     and `ceb8745b7b48e921fa9e9627773357535ec177081074146a24b9e1ff44cd0ddf`.
145. Nested three-grid provenance now distinguishes dimensionless solver
     identity from physical mesh placement.  Inner refinement margins, wake
     extent, primary/auxiliary control-volume locations and wall exchange
     distance must scale with coarse hull length; leaving any one fixed in
     lattice cells fails the convergence gate.  The new v10 launcher uses an
     exact integer 3:4:5 L90/L120/L150 design: inner margins 9/12/15, inner
     wakes 12/16/20, primary CV margins 6/8/10, auxiliary margins
     3,9 / 4,12 / 5,15, and wall sampling distances
     2.109375/2.8125/3.515625.  Its largest preflight is 22.51 GiB at L150,
     below a 24-GiB RTX 3090 budget but intentionally close enough that the
     runtime memory guard remains mandatory.  Earlier constant-inner-margin
     runs are stability evidence only, never formal grid convergence members.
146. The unfiltered continued L120 baseline reproduces the finest-interface
     failure at step 1840: peak lattice speed reaches `0.15867`, the minimum
     population reaches the `1.8626e-9` positivity floor, collision limiting
     activates, and the peak lies four cells from the allocated boundary.
     Its exact-PID process was stopped; log/checkpoint SHA-256 values are
     `79e475b8322f55f7ad80795b083cf92cef24b6480fb25740236556cad934c9f5`
     and `d3cebcedd2cbacdfcb1528b9d2529aa984b541973c859f0d464e895ddb314f0d`.
     The released GPU now runs the full v10 scaled L90 configuration with
     regularized restriction and prolongation plus the force-safe width-two,
     strength-1.0 kinetic filter.  It may continue to 12000 steps only if its
     60-step health cadence survives the known 1400--2400 instability window.
147. The AMR design is cross-checked against three primary grid-refinement
     studies in `docs/amr-interface-literature-audit-2026-08-01.md`.  The audit
     confirms the need for non-equilibrium rescaling, fine-to-coarse filtering,
     space-time reconstruction and population-wise transfer accounting, while
     explicitly recording two open gaps: a manufactured local conservation
     test for cell-centred coarse-to-fine interpolation, and an independent
     viscosity audit before the experimental KBC collision can be considered a
     production stabilizer.  No proprietary PowerFLOW/XFlow behavior is
     inferred without public technical evidence.
148. The force-admissible width-two, strength-1.0 kinetic filter is also
     rejected.  It crosses speed 0.1 at step 1500, speed 0.15 and the
     positivity floor at step 1560—exactly the same onset as the unfiltered
     baseline—and reaches speed 0.18870 with limiter activation by step 1680.
     Log/checkpoint SHA-256 values are
     `e971f7994549dc4104761a3aa935695a723a2243c9fff735b359301c0d452e88`
     and `e26a98e6f456febdfd82ad08d65bd633ef4700b258995a1a25d8151770ea6328`.
     This rules out unresolved higher-order kinetic content as the dominant
     mode; GPU1 now isolates interface placement by expanding the transverse
     finest-level wall buffer from 16 to 24 cells without any interface filter
     or prolongation change.
149. Coarse-to-fine Hermite regularization alone is likewise rejected.  Its
     aligned threshold steps are unchanged (speed 0.1 at 1500, speed 0.15 and
     positivity floor at 1560); limiter use reaches `4.6168e-5` and peak speed
     0.18062 by step 1620.  Log/checkpoint SHA-256 values are
     `ef064dec7675654f811c00604a73a5a01d4fe0765dbef920c09b31e1a4e11f06`
     and `6dc9abf8fdbc3de3de2ff86e3b0060d1a65c4c96879e53a821fdf57caaa2bfb1`.
     GPU2 now supplies a second placement response point with a 32-cell
     finest transverse buffer; satisfying that geometry also requires moving
     the outer interface from its six-cell to eight-cell coarse margin, which
     passed a 6.96-GiB preflight.
150. Interface health now separates correction quality from the defect being
     corrected.  A common population-inventory moment reduction reports raw
     mass mismatch and the norm of raw three-component momentum mismatch before
     reflux, while the existing ledger continues to report bounded correction,
     limiter use and post-correction residual.  Per-interface maxima persist in
     checkpoints and final results.  This prevents a small corrected residual
     from hiding a large recurring coarse/fine transfer imbalance; 53 focused
     flux-register, AMR runtime and SUBOFF tests pass.
151. The L150 continuation baseline reaches the same failure at step 2400,
     when wall activation is approximately 47%: peak speed is 0.21156, the
     minimum population reaches the positivity floor, and the peak is three
     cells from the finest allocated boundary.  Log/checkpoint SHA-256 values
     are `697d3fd2308c73b03665f84ebfc846841be5d3bdcaa41553887b8d03888fbaec`
     and `204227e2a7ac7a04a440f9287af3dfd11a301f88e488a748030f0c2ec4661716`.
     The released GPU now runs a matched L90 diagnostic with wall stress
     disabled.  This trajectory is deliberately nonphysical and can establish
     causality only; it is excluded from every resistance acceptance path.
152. D3Q19 BFL wall startup now supports independent no-penetration and
     wall-shear activation while preserving the legacy single-ramp behavior by
     default.  The nested runner exposes separate ramp durations, records both
     activation fractions at health cadence, checkpoints their resolved values,
     requires both to finish before force sampling is eligible, and treats both
     as scaled time parameters in three-grid convergence.  Unit tests verify
     that full normal impermeability can coexist with zero applied shear and
     that invalid activation fractions fail closed.  This is a diagnostic and
     startup-control capability, not a change to already running trajectories.
153. Population health is now a runtime gate rather than an end-of-run note.
     Production defaults require every cadence sample to remain above
     `1e-8` and cumulative positivity limiting to remain at or below `1e-6`;
     both thresholds are explicit configuration, checkpoint and convergence
     identity fields.  Crossing either threshold writes the health record and
     then fails closed immediately.  Runs without health cadence remain usable
     only as integration smoke tests, while every convergence source already
     requires `population_health_target_met=true`.  This excludes v10-style
     trajectories that touch the limiter floor and later appear to recover.
154. The combined v10 scaled L90 candidate is rejected and stopped at step
     2040.  It does not delay either speed threshold, crosses the population
     floor and limiter gate one cadence earlier than the baseline (step 1500),
     reaches a negative population of `-1.5701e-5`, and attains peak speed
     0.20036.  Its later partial speed recovery is immaterial under the new
     history-aware health gate.  Log/checkpoint SHA-256 values are
     `fb658fbb0623831b16ffd5ee13d0891c9c7fe3bd8ed615bc6fb25fed6061f1b3`
     and `81183c00ca937b8ec2b7d96a8122eaaaa3a34788ff89999cca2352db1c8b1c41`.
     GPU3 now tests a split startup: BFL normal impermeability completes during
     the initial low-Re 300 steps, while physical wall shear retains the
     3000-step ramp.
155. The exact equivalent Re=100 sphere sequence is complete at R9/R12/R15.
     Control-volume drag coefficients are `1.1651455`, `1.1496003` and
     `1.1455558`; all three records pass duration, stationarity, independent
     BFL-link observer and numerical-quality gates.  The sequence is monotonic
     with observed order `4.1844`, Richardson-extrapolated Cd `1.1429363` and
     a `0.2292%` estimated finest-grid discretisation error.  The extrapolated
     value is `4.6903%` above Schiller--Naumann, within the declared 5% physical
     validation target.  The canonical convergence record includes SHA-256
     hashes of all three immutable source JSON files.
156. Enlarging the finest transverse wall clearance from 16 to 24 fine cells
     is rejected.  It advances speed thresholds from steps 1500/1560 to
     1380/1440, reaches the population floor at 1440 rather than 1560 and uses
     the collision positivity limiter on `1.5636e-4` of cells.  Log/checkpoint
     SHA-256 values are `b75292f246205c9a2ea1a6d615218f1a7152573d132e1b6a54c59650f016fedc`
     and `8e7d3d7221fe07f9aa3f9248c0e5c136111715c7fd5469e4710c63186c14777d`.
     Thus simply moving the transverse fine/coarse interface outward is not a
     stability remedy; the 32-cell placement response remains active.
157. The no-wall-shear causal diagnostic is rejected at step 1560.  Disabling
     physical wall stress changes neither the 0.1/0.15 speed threshold steps
     nor the first population-floor step relative to the unfiltered baseline;
     limiter use reaches `1.2366e-5`.  Meanwhile the newly exposed inner raw
     interface defects grow to mass `3.2432` and momentum norm `0.78243` before
     reflux.  Log/checkpoint SHA-256 values are
     `97d1465e34fa773a817cd5b452cc7553fb8a5ba235ed7af45de75dd10e79c76a`
     and `7cec790945927dea66606b7a071580c223108959458d7436d5c18843cd440ea0`.
     Wall-law shear is therefore not the sole trigger; the next common-module
     work targets conservative space-time interface transfer and the normal
     geometry/interface coupling rather than retuning the wall law.
158. Reflux now has an opt-in `crossing_links` stencil that closes mass and
     three-component momentum using a weighted minimum-norm projection over
     only directions that physically cross the refinement boundary.  Unlike
     the legacy exterior-cell stencil, it cannot modify the rest population
     or tangential/non-crossing directions.  Forty-four focused flux-register,
     single/nested AMR and pulse-validation tests pass.  In the matched 24-step
     pulse A/B it preserves the `1.0172e-7` mass drift, lowers the maximum
     conserved correction from `2.5527e-4` to `7.9711e-5`, lowers reflux
     residual from `2.1828e-11` to `7.2760e-12`, and improves global/refined
     density plus refined velocity RMS.  Interface density RMS rises by 0.40%,
     so the method is pulse-admitted for a strict SUBOFF A/B but remains
     non-default until that high-Re test passes.
159. The 20,000-step one-level L150 run completes but is rejected.  Mean
     control-volume resistance is `101.4449 N`, `16.07%` above the `87.4 N`
     experiment, and the sampling window remains strongly nonstationary
     (`39.53%` block range, `19.97%` half-mean drift, `33.41%` linear trend).
     The primary/auxiliary control volumes agree within `0.10%` and the BFL
     link observer differs by only `0.166%`, but surface-pressure-plus-wall
     stress differs from the control volume by `72.78%`.  The complete result
     is frozen with canonical SHA-256
     `a60af351102aea2f835e3a3b83e2e23f03c034ddf899e7df3dbdab700351576c`;
     the queued v9 L120 run now tests quadratic surface-pressure extrapolation.
160. Moving both hierarchy interfaces to provide 32 finest transverse cells
     is also rejected.  Its speed and population-floor onset is step 1440,
     earlier than the unfiltered baseline's step 1560 floor, and limiter use
     reaches `8.3268e-5`.  Log/checkpoint SHA-256 values are
     `4920cd64fdd6ee1ababb372f1122b21fc48ca5ec9464486b09ff028fb25e36e4`
     and `1a3ee1bf3c732b1d01c9942774e3d4b5b3de16f881863fc1345f1fbcf896778d`.
     Together with the 16- and 24-cell cases, this rules out transverse
     interface displacement alone as a production remedy.
161. Transfer positivity is now genuinely bidirectional.  The existing
     `enforce_transfer_positivity` option limits both fine-to-coarse
     restriction and coarse-to-fine initial/ghost prolongation while retaining
     the conserved density and momentum of each transferred state.  Every
     reflux ledger and health record reports the prolongation-limited fraction
     and minimum non-equilibrium scale separately; the aggregate transfer gate
     uses the worst value from either direction.  Fifty-seven focused
     single/nested AMR, SUBOFF smoke and startup-health tests pass.  This closes
     a semantic and safety gap but does not retroactively admit any trajectory.
162. Split wall activation is rejected by the strict runtime at step 1560.
     Normal impermeability was already fully active at step 300 while wall
     shear was only `53.14%` active at failure; nevertheless the population
     floor and `1.9374e-5` collision-limiter fraction appear at the baseline
     onset.  Log/checkpoint SHA-256 values are
     `b1fc4b14df1727943f144c11a79e86e8f2f468c1c66d059c804cc4b1915719e0`
     and `669fca75f0bf60a09db28e4886b5d9b8f21304c5ba62de2644de08d9720d7069`.
     The wall startup order is therefore ruled out as the primary instability
     mechanism.  Its released GPU now isolates bidirectional transfer
     positivity on the otherwise matched baseline interface.
163. Collision admission now includes a reusable periodic shear-wave decay
     audit that measures recovered viscosity rather than trusting a collision
     label.  At `tau=0.8` the target kinematic viscosity is `0.1`: BGK recovers
     `0.1002065` (0.207% error) and cumulant `0.1000517` (0.052% error), while
     the current entropic KBC kernel recovers `0.1710753` (71.08% error).
     Sixty KBC/contract/audit tests pass with KBC deliberately rejected by the
     new 2% physical-property gate.  KBC remains withheld from SUBOFF until its
     collision formulation both recovers target viscosity and passes entropy,
     positivity and interface audits; apparent stability cannot substitute for
     the correct Reynolds number.
164. Nested SUBOFF results now expose `collision_viscosity_target_met`, and
     the fail-closed three-grid assessor requires it for every source record.
     The cumulant production path passes; `entropic_kbc` remains callable only
     for diagnostics and can no longer become a physical single-grid candidate
     or enter a convergence claim while its shear-wave audit is rejected.
165. First-order coarse-to-fine ghost injection is rejected at step 1560, the
     same population-floor onset as the trilinear baseline; collision limiter
     use reaches `1.1542e-5`.  Log/checkpoint SHA-256 values are
     `d900466b3c668874b4415a54dbd5a7edf69ebd5b81578333828442049c201e57`
     and `ffa6eff23b14db9537144eaa28b026b0de63d879d3274f8951e43d06a3f9543b`.
     Spatial interpolation order alone is therefore ruled out.  The released
     GPU runs the factorial combination of stream-link-local reflux and the
     newly bidirectional transfer-positivity guard.
166. Nested reflux diagnostics now accumulate every child-interface invocation
     over a complete root step.  Previously the inner interface, which advances
     twice per root step in a three-level 2:1 hierarchy, overwrote its first
     ledger with the second.  Requested/applied corrections, residual and raw
     mismatch now sum in consistent root-step inventory units, while transfer
     limiter fractions and alphas retain the worst invocation.  Fifty-one
     focused single/nested runtime and SUBOFF tests pass.  The solver had always
     applied both corrections; this change makes health evidence match that
     actual subcycling rather than silently reporting only its final half.
167. Reflux health now records the maximum fraction by which any corrected
     direction's actual interface inventory is changed.  The matched pulse
     baselines require only `1.1562e-6` (legacy exterior-cell stencil) and
     `4.8261e-6` (crossing-link stencil), whereas a small final residual can
     otherwise conceal a much larger applied correction.  Production nested
     runs default to a fail-closed `1e-3` ceiling, persist per-interface maxima
     in checkpoints/results and include the threshold in convergence identity.
     Eighty-six focused interface, runtime, launcher and convergence tests pass.
168. The pulse-admitted crossing-link reflux candidate is rejected by the L90
     high-Re test at step 1440, 120 steps earlier than the baseline floor.
     Peak speed reaches `0.23746`, collision limiting `1.1760e-4`, inner raw
     mass/momentum mismatch `255.82`/`16.14`, and the inner reflux residual
     `2.0401` with one limited direction.  Log/checkpoint SHA-256 values are
     `874b26a24ac420294b20dd593dd3951c9dfc09e8ee92604af86646bd063d60f8`
     and `cee170dc42fa64cba37f717dc7e301be290ba4e3f7e24dbbbf3fbe789c00de0c`.
     This demonstrates why small-pulse admission is necessary but insufficient.
     The option remains experimental and is withheld from production defaults.
     GPU2 now repeats the slow-Re candidate with bidirectional transfer health,
     root-step ledger accumulation and the new applied-correction gate enabled.
169. The exact cylinder Re=100 R9 member completes 54,000 steps with a
     22,500-step statistics window spanning `13.01` shedding cycles.  It is a
     numerically admitted coarse-grid convergence source: drag block range is
     `0.665%` and independent BFL/control-volume drag differs by only
     `0.00042%`.  Its `Cd=1.44725` and `St=0.173455` miss the declared
     single-grid references by `8.82%` and `5.77%`, respectively, so it is not
     individually admitted.  R12 started automatically; R12/R15 and the
     extrapolated sequence, not a changed reference, determine final physical
     validation.  Canonical R9 SHA-256 is
     `313ee8efaa57fc09803cc8f0ce639f2de3df17573bf066de5ae940cd0723e4cd`.
170. Extending the resolved-Re continuation from step 600 to step 1800 delays
     but does not remove the nested instability.  The candidate fails the
     population gate at step 2520, only 720 steps after reaching Re=100,000,
     with peak speed `0.17086`, limiter fraction `1.3603e-5` and inner raw
     mass/momentum mismatch `3.0617`/`0.34347`.  Log/checkpoint SHA-256 values
     are `20c7b745e2c3451fb92af53a20a33bd137b92390883e598700aa46136ea5dd4f`
     and `23ef5303daf9453745c392dec467cfea1468d7c62988a519a5f146ad5a5270dd`.
     Slow startup is retained as a control but rejected as a stability fix.
     GPU6 now isolates `C_s=0.10` SGS dissipation under the standard ramp and
     all new health gates; any stable result still requires model sensitivity.
171. Bidirectional transfer positivity is physically inactive in the matched
     L90 trajectory and is rejected as a stability remedy at step 1560.  Both
     prolongation and restriction report zero limited fraction through failure,
     while the population floor, peak speed `0.19061` and collision-limiter
     onset reproduce the baseline.  Log/checkpoint SHA-256 values are
     `62a41933ec8bee8a3156f9c5cad4e592dc1f8e17b5d4e33e5f4305219d0631dd`
     and `6bd506193372e2da75b536c57e7de49686710dddef967e25fbafe043ab543ebb`.
     The safety feature remains enabled, but cannot be credited with stability.
     GPU3 now supplies the `C_s=0.15` counterpart to the `C_s=0.10` SGS test.
172. The D3Q19 cumulant module now exposes the exact local Smagorinsky
     `tau_eff` field used by collision.  This permits checkpoint-level audits
     of added subgrid viscosity and effective Reynolds-number distribution for
     the `C_s=0.10/0.15` candidates, instead of describing them only by an input
     coefficient.  Unit tests verify equilibrium returns the molecular tau,
     non-equilibrium stress can only increase tau, and invalid states fail
     closed.  Stability alone will not promote a coefficient without this
     dissipation audit and subsequent force/grid sensitivity.
173. The crossing-link reflux failure reproduces exactly in v18 with
     bidirectional transfer positivity: the level health, raw mismatch,
     step-1440 population floor and peak speed are identical to v16, while both
     transfer limiters remain inactive until failure.  Reproduction
     log/checkpoint SHA-256 values are
     `d55de5ab01c53811e2bf9f9155ad7cfc15d316205668f293691c18a21b894eff`
     and `400542944be97738978ec4451cbc0addf783ebc441df8d897562e358f23be45f`.
     The negative result is therefore deterministic rather than a GPU anomaly.
     GPU1 now supplies the `C_s=0.20` upper sensitivity point.
174. SGS dissipation can now be audited directly from any nested checkpoint.
     `scripts/inspect_nested_smagorinsky_checkpoint.py` reads the collision tau
     chain recorded at the latest health sample and evaluates each hierarchy
     level in bounded-memory chunks.  It reports min/mean/max `tau_eff`, mean
     and maximum eddy viscosity, and their ratios to molecular viscosity.
     Chunk-invariance and input-contract tests pass.  This artifact is required
     before a stable `C_s` candidate can advance to resistance convergence.
175. The first checkpoint SGS audit covers v19 at step 900 with `C_s=0.05`.
     Mean eddy/molecular viscosity ratios on levels 0/1/2 are only
     `0.000295`, `0.002277` and `0.001621`; local maxima are `0.8077`,
     `0.2110` and `0.0642`.  Thus the baseline coefficient adds negligible
     domain-mean viscosity while acting locally in the strongest coarse-level
     gradients.  The immutable audit SHA-256 is
     `eae9ea0f37f0426601de449d14aaf9ab9bc2ff99796414b138f6557612452f62`.
     Higher-`C_s` candidates will be compared at matched checkpoint phases.
176. Matched target-Re step-600 audits quantify the `C_s=0.10/0.15`
     dissipation.  For 0.10, mean eddy/molecular viscosity ratios on levels
     0/1/2 are `0.0166/0.1285/0.0922` with local maxima
     `42.13/11.41/3.41`; for 0.15 the means rise to
     `0.0371/0.2882/0.2068` and maxima to `93.18/25.42/7.63`.
     Thus 0.15 materially lowers the effective Re across the refined levels;
     stability must be weighed against this model change, with 0.10 preferred
     if it clears the same health window.  Audit SHA-256 values are
     `56baf724194e3bd480e4c67da45ff10ef63675b018df4906ff929c73b2b4bc00`
     and `3b277e7f636bcf5247ab5664f506bf0541b74adf456c8851c4e3c2b83f0646cd`.
177. A primary-source KBC candidate is implemented without replacing the
     rejected legacy API.  It uses `beta=1/(2*tau)`, relaxes the deviatoric
     shear deviation by `2*beta`, applies the analytical entropic scalar-product
     gamma only to bulk/higher-order deviations, and clamps gamma to the
     population positivity interval.  The periodic shear wave recovers
     viscosity `0.1000517` for target `0.1` (0.052% error), versus 71.08% for
     the legacy kernel; mass/momentum, positivity and no-material-entropy-rise
     tests pass (50 focused KBC/audit tests total).  Result SHA-256 is
     `2b559bd7e7ba7e437f178659212f094ddec0ecc1df5cc81e7fb750ac6381a450`.
     The new kernel remains experimental until stronger entropy, canonical-flow
     and AMR tests pass; it is not yet enabled in SUBOFF.
178. The nested SUBOFF runner can now execute `natural_kbc` explicitly as a
     diagnostic collision path.  It remains excluded from
     `collision_viscosity_target_met` and therefore cannot become a physical
     candidate or convergence source.  Seventy-two focused nested-runner,
     KBC, viscosity and entropy tests pass.  This enables a fail-closed L90
     interface experiment once a GPU is released without weakening admission.
179. The `C_s=0.10` candidate passes the historical step-1560 failure point
     with positive populations and zero limiter use, but its step-1500 SGS
     audit quantifies substantial modeled dissipation: mean eddy/molecular
     viscosity ratios are `0.0433/0.5510/0.7828` on levels 0/1/2, with local
     maxima `46.88/12.48/4.57`.  Thus it is a potentially stable LES candidate,
     not a DNS-equivalent Re=100,000 solution.  Audit SHA-256 is
     `55568700cd2f2e691d932dbe31009bbf4ff98a0f74e3bbb816a9b7fb6462beed`.
     It must still finish the health window and pass coefficient/grid/force
     sensitivity before any production promotion.
180. The collision viscosity audit is repeated near the SUBOFF relaxation
     limit at `tau=0.5002` with a 64-cell shear wavelength and 800 steps.
     Cumulant recovers `6.78077e-5` (1.712% error) and experimental natural KBC
     `6.78469e-5` (1.770% error) for target `6.66667e-5`; both meet the 2%
     gate, while BGK is at 2.964%.  This rules out the legacy KBC-style
     viscosity drift in the new formulation at high Re, but does not replace
     nonlinear/AMR validation.  Source hashes are recorded in
     `docs/evidence/collision-viscosity-high-re-limit-r1.json`.
181. Production nested launchers now pass `C_s` explicitly from
     `TENSORLBM_CS_SMAG` (default `0.05`) rather than relying on a hidden parser
     default, and the runner rejects coefficients outside `[0,0.3]`.  This
     makes any resumed long SGS trajectory reproducible in its command line,
     checkpoint identity and convergence configuration.
182. The audited slow-Re repeat v19 reproduces the v15 failure exactly at step
     2520.  Its inner applied reflux correction fraction is `1.3011e-4`, below
     the new `1e-3` gate, so population health is independently the first
     failing contract.  Reproduction log/checkpoint SHA-256 values are
     `476a819d898590c21d048ce1543618962fae7900697d6e132d8bbbc1dc32e62d`
     and `4966bf7cb81819a4058595c6bdd024a87b807f3b9f69aec2ae843964fc820d43`.
     GPU2 now runs the viscosity-correct natural-KBC L90 diagnostic, still
     excluded from physical acceptance pending nonlinear evidence.
183. A read-only nested-checkpoint wall auditor now reconstructs the exact
     SUBOFF CAD, BFL links, analytical normals and calibrated surface-area
     weights, then evaluates the frozen finest-level population field without
     BFL application, forcing or time advance.  This permits wall-model
     applicability audits across checkpoint generations without mixing old and
     new wall-source dynamics in a resumed trajectory.
184. The tangential pressure-gradient diagnostic no longer uses Cartesian
     central differences that can sample arbitrary solid-node populations.
     A common 26-neighbour, inverse-distance-weighted least-squares module uses
     fluid neighbours only, projects the fitted gradient onto the wall tangent
     plane and fails closed on rank-deficient stencils.  Linear-field recovery,
     solid-value isolation, rank-deficiency and exact count aggregation tests
     pass together with the wall-force and nested-runner regressions.
185. The corrected-source v23 step-3000 checkpoint has 91.82% of exchange-node
     samples in the declared `30 <= y+ <= 1000` range, versus 5.20% for the
     pre-source-correction v20 step-12000 state.  With the solid-aware gradient,
     v23's pressure-gradient parameter has median `0.9701`; 51.04% of nodes are
     at or below one and 29.84% exceed ten.  This is diagnostic-only evidence
     that a spatially selective non-equilibrium wall treatment is needed; it is
     not a force correction or physical validation.  Versioned audit SHA-256
     values are `d4a66eabd569976d3a4d17f8c31c7609a4703b523a77facf1240f3b96898902d`
     for v23 and `c217b4ff8572bf8e8a218cc4e3aa90fe51623957e460dabe8e84907f100a4b99`
     for v20.
186. The fluid-only gradient sampler now retains the tangential gradient
     vector and projects it onto the local exchange-velocity direction.  The
     signed v23 audit is nearly balanced at the median (`-0.00625`) rather than
     uniformly adverse: 21.16% of samples have a signed parameter above one,
     while 15.62% are below minus one.  Any non-equilibrium closure must
     therefore preserve sign and be activated locally; a global shear multiplier
     is rejected.  The CPU-reproducible signed audit SHA-256 is
     `b484a9e7218711bba51755823de9a365210e408d01415d50d690fef5d1b31e19`.
187. A geometry-agnostic 20-bin axial exposure profile localizes the signed
     wall diagnostic without changing populations.  For v23, the largest
     strong-adverse fractions occur over normalized hull-length intervals
     `0.05-0.20` and `0.80-0.95`; `0.75-0.80` is predominantly favourable.
     The first and last 5% contain large fractions of both signs, so a future
     non-equilibrium selector must also use local flow direction, a low-shear
     guard and temporal hysteresis rather than hard-coded axial zones.  Audit
     SHA-256 is
     `39931c82e1d3c0825746611c31cd3f982b9fdd559af47266785e978c002fdcdd`.
188. A common non-equilibrium wall selector is implemented as a force-agnostic
     state machine.  It requires persistent signed adverse-gradient evidence,
     uses separate entry/exit thresholds, and immediately clears nodes with
     invalid evidence, low friction velocity or out-of-policy `y+`.  The
     selector deliberately has no wall-stress correction law and is not wired
     into a production runner; it must first pass zero-gradient flat-plate and
     adverse-gradient canonical validation.  Ten focused selector, gradient
     and checkpoint-diagnostic tests pass.
189. A read-only finite-flat-plate checkpoint auditor now reconstructs the
     halfway BFL links, two-sided plate normals and exact wall viscosity, then
     reuses the same frozen-state wall diagnostic as SUBOFF.  Its manufactured
     uniform-flow test reports zero tangential pressure gradient and verifies
     byte-equivalent saved populations after inspection.  This allows the
     active long flat-plate trajectory to test selector false positives without
     restarting it under newly loaded code.
190. The first active flat-plate audit exposed a diagnostic boundary-condition
     defect: only the middle of three periodic spanwise planes had a full-rank
     pressure-gradient stencil.  The common fluid-only least-squares sampler
     now accepts explicit periodic spatial axes and wraps neighbour indices;
     flat-plate runtime/audit paths declare periodic z while SUBOFF keeps the
     non-periodic default.  Thirty-five focused wall, BFL, flat-plate and
     checkpoint tests pass, including full coverage on a three-plane periodic
     manufactured field.

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
