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
   limiter is now under long-run test; triggering fraction is recorded.
5. Direct uniform L120 with the corrected wall traction remained finite
   through at least step 1800.  At that point it reported approximately
   103 N pressure and 82 N wall shear.  One convective time is 2000 steps at
   `L=120, U_lu=0.06`, so this is not a settled resistance result.
6. R=8 sphere, Re=100, D=16 cells: `Cd_CV=1.38065`,
   `Cd_BFL=1.38067`, observer difference 0.0017%, correlation error 26.5%.
   R=12 and higher resolutions are required before curved-wall admission.

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

Active comparisons cover:

- uniform L120 versus AMR effective L240;
- AMR effective L360 (diameter about 42 cells);
- standard versus enlarged external domain;
- standard versus enlarged refinement patch;
- coupled versus diagnostic uncoupled wall traction;
- Cumulant, WALE and positivity-limited collision paths.

No result is admitted before at least five convective times, three settled
windows, two independent force observers, and three effective resolutions.
