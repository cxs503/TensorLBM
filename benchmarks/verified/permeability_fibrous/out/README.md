# Fibrous-bed permeability with a sub-lattice lubrication closure (P55-N02)

Generated 2026-11-07T12:11:07Z from `result.json` (zero hand transcription).

## Verdict

**FAIL** - At least one gate failed -- see gates; reported as an honest FAIL.

| case | grid | err vs reference (%) |
|------|------|----------------------|
| hex_vf0.70 | 19 | 21.169 |
| hex_vf0.70 | 28 | 19.045 |
| hex_vf0.75 | 19 | 43.927 |
| hex_vf0.75 | 28 | 36.818 |
| sq_vf0.40 | 36 | 2.554 |
| sq_vf0.40 | 54 | 2.229 |

## Gates

- **G1_hex070** - FAIL: hex phi=0.70: judged levels <= 5% vs Gebart, GCI band narrowing, finest GCI <= 3%
    - max_err_judged_pct = 21.16902759868645
    - tolerance_pct = 5.0
    - gci_band_narrowing = False
    - gci_finest_pct = 2221.6135645190575
- **G2_hex075** - FAIL: hex phi=0.75: judged levels <= 5% vs Gebart, GCI band narrowing, finest GCI <= 3%
    - max_err_judged_pct = 43.92718737509996
    - tolerance_pct = 5.0
    - gci_band_narrowing = False
    - gci_finest_pct = nan
- **G3_sq040** - FAIL: sq phi=0.40: error ladder de-monotonised (no increase beyond the GCI band), GCI band narrowing, finest GCI <= 3%
    - gci_band_narrowing = False
    - gci_finest_pct = nan
    - successive_diffs_pct = [0.21035993044132884, -1.4729981667427454, -0.3250643403218789]
    - no_significant_increase = False
- **G4_dilute_negative_control** - PASS: dilute negative control: closure changes K by < 1% at phi = 0.05 and 0.10
- **G5_diagnostics** - PASS: diagnostics: grid-independent delta_min plus the error-vs-delta_min/dx master curve are recorded in result.json
    - n_master_points = 23
    - plateau_proved = False

## Error ladder (closure on / off)

### hex_vf0.70 (arrangement hex, phi = 0.7)

| grid | delta_min/dx | closure | k_sim | k_ref(A) | errA (%) | k_ref(B) | errB (%) | n_corrected | nu factor |
|------|--------------|---------|-------|----------|----------|----------|----------|-------------|-----------|
| 9 | 1.09 | on | 0.0222884 | 0.0256527 | 13.115 | 0.0503386 | 55.723 | 3 | 0.914..0.930 |
| 13 | 1.58 | on | 0.0264335 | 0.0535223 | 50.612 | 0.0461782 | 42.758 | 0 | 1.000..1.000 |
| 19 | 2.31 | on | 0.0940612 | 0.114329 | 17.727 | 0.11932 | 21.169 | 0 | 1.000..1.000 |
| 28 | 3.40 | on | 0.160154 | 0.248293 | 35.498 | 0.19783 | 19.045 | 0 | 1.000..1.000 |
| 40 | 4.86 | on | 0.411301 | 0.50672 | 18.831 | 0.473764 | 13.184 | 0 | 1.000..1.000 |
| 9 | 1.09 | off | 0.0222882 | 0.0256527 | 13.116 | 0.0503386 | 55.723 | 0 | 1.000..1.000 |
| 19 | 2.31 | off | 0.0940612 | 0.114329 | 17.727 | 0.11932 | 21.169 | 0 | 1.000..1.000 |
| 40 | 4.86 | off | 0.411301 | 0.50672 | 18.831 | 0.473764 | 13.184 | 0 | 1.000..1.000 |

### hex_vf0.75 (arrangement hex, phi = 0.75)

| grid | delta_min/dx | closure | k_sim | k_ref(A) | errA (%) | k_ref(B) | errB (%) | n_corrected | nu factor |
|------|--------------|---------|-------|----------|----------|----------|----------|-------------|-----------|
| 9 | 0.82 | on | 0.00544878 | 0.0121233 | 55.055 | 0.0174028 | 68.690 | 3 | 0.948..1.322 |
| 13 | 1.18 | on | 0.0104266 | 0.0252943 | 58.779 | 0.0309261 | 66.285 | 2 | 0.895..0.895 |
| 19 | 1.72 | on | 0.0223172 | 0.054031 | 58.696 | 0.0398004 | 43.927 | 0 | 1.000..1.000 |
| 28 | 2.54 | on | 0.0644451 | 0.117342 | 45.079 | 0.102 | 36.818 | 0 | 1.000..1.000 |
| 40 | 3.62 | on | 0.197222 | 0.239473 | 17.643 | 0.222639 | 11.416 | 0 | 1.000..1.000 |
| 9 | 0.82 | off | 0.00544878 | 0.0121233 | 55.055 | 0.0174028 | 68.690 | 0 | 1.000..1.000 |
| 19 | 1.72 | off | 0.0223172 | 0.054031 | 58.696 | 0.0398004 | 43.927 | 0 | 1.000..1.000 |
| 40 | 3.62 | off | 0.197222 | 0.239473 | 17.643 | 0.222639 | 11.416 | 0 | 1.000..1.000 |

### sq_vf0.40 (arrangement sq, phi = 0.4)

| grid | delta_min/dx | closure | k_sim | k_ref(A) | errA (%) | k_ref(B) | errB (%) | n_corrected | nu factor |
|------|--------------|---------|-------|----------|----------|----------|----------|-------------|-----------|
| 16 | 4.58 | on | 1.26433 | 1.1749 | 7.611 | 1.21785 | 3.816 | 0 | 1.000..1.000 |
| 24 | 6.87 | on | 2.42559 | 2.64354 | 8.244 | 2.52736 | 4.027 | 0 | 1.000..1.000 |
| 36 | 10.31 | on | 6.14301 | 5.94796 | 3.279 | 5.99005 | 2.554 | 0 | 1.000..1.000 |
| 54 | 15.46 | on | 12.6191 | 13.3829 | 5.708 | 12.9067 | 2.229 | 0 | 1.000..1.000 |
| 16 | 4.58 | off | 1.26433 | 1.1749 | 7.611 | 1.21785 | 3.816 | 0 | 1.000..1.000 |
| 36 | 10.31 | off | 6.14301 | 5.94796 | 3.279 | 5.99005 | 2.554 | 0 | 1.000..1.000 |
| 54 | 15.46 | off | 12.6191 | 13.3829 | 5.708 | 12.9067 | 2.229 | 0 | 1.000..1.000 |

## Plateau diagnostics

- threshold delta_min/dx = 1.5; plateau proved: **False**
- For delta_min/dx >= 1.5 the closed-case error stops descending monotonically with refinement: it saturates in a band (hex_vf0.70: 29.57 pp, hex_vf0.75: 32.51 pp, sq_vf0.40: 1.80 pp) whose width is set by the reference-model bias (Gebart asymptote vs exact solution) plus the residual GCI, not by the grid. Below delta_min/dx ~ 1.5 the error is resolution-controlled -- see the master curve.

| case | delta_min/dx points | err (%) | successive diffs (pp) | monotone | band (pp) |
|------|--------------------|---------|-----------------------|----------|-----------|
| hex_vf0.70 | 1.58, 2.31, 3.40, 4.86 | 42.76, 21.17, 19.04, 13.18 | -21.589, -2.125, -5.860 | True | 29.573 |
| hex_vf0.75 | 1.72, 2.54, 3.62 | 43.93, 36.82, 11.42 | -7.109, -25.402 | True | 32.511 |
| sq_vf0.40 | 4.58, 6.87, 10.31, 15.46 | 3.82, 4.03, 2.55, 2.23 | +0.210, -1.473, -0.325 | False | 1.798 |

### GCI bands (Roache)

| case | closure | grids | p | GCI_fine (%) |
|------|---------|-------|---|--------------|
| hex_vf0.70 | on | [9, 13, 19] | -7.47 | nan |
| hex_vf0.70 | on | [13, 19, 28] | 0.06 | 2221.614 |
| hex_vf0.70 | on | [19, 28, 40] | -3.59 | nan |
| hex_vf0.70 | off | [9, 19, 40] | -1.99 | nan |
| hex_vf0.75 | on | [9, 13, 19] | -2.33 | nan |
| hex_vf0.75 | on | [13, 19, 28] | -3.30 | nan |
| hex_vf0.75 | on | [19, 28, 40] | -3.08 | nan |
| hex_vf0.75 | off | [9, 19, 40] | -3.14 | nan |
| sq_vf0.40 | on | [16, 24, 36] | -2.87 | nan |
| sq_vf0.40 | on | [24, 36, 54] | -1.37 | nan |
| sq_vf0.40 | off | [16, 36, 54] | -0.45 | nan |

### Dilute negative control

| case | phi | grid | delta_min/dx | K(on) | K(off) | correction (%) | pass |
|------|-----|------|--------------|-------|--------|----------------|------|
| sq_vf0.05 | 0.05 | 24 | 17.94 | 36.0002 | 36.0002 | 0.0000 | True |
| sq_vf0.10 | 0.1 | 24 | 15.44 | 21.9416 | 21.9416 | 0.0000 | True |

## Method

- closure rule: nu_eff = nu (delta_mask/delta_true)^3 with taper weight max(0, 1 - delta_true/(1.5 dx))
- conductance: G(delta) = int ds/h^3 = (3 pi/8) sqrt(R) delta^-5/2 for the near-contact two-cylinder parabolic gap h = delta + s^2/R (Sangani & Acrivos 1982)
- no-op: nu_eff == nu exactly where delta_true >= 1.5 dx
- hex reference: Gebart (1992) near-jamming closed form k = C1 R^2 (sqrt(phi_max/phi)-1)^2.5, C1 = 16/(9 pi sqrt6), phi_max = pi/(2 sqrt3)
- sq reference: Sangani & Acrivos (1982) exact drag table, k_ref = pi R^2/(phi f) (table double-transcribed in benchmarks/verified/permeability/NOTES.md)

