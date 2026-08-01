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
