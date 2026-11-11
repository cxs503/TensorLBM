# Oscillating-foil propulsion verification face (node `p161w2-arb-n05`)

Mature, machine-checkable verification surface for **flapping / oscillating-foil
propulsion** in TensorLBM: Theodorsen–Garrick closed-form anchor, an immersed
boundary D2Q9 solver for a NACA 0012 section in pitch–plunge motion, and the
acceptance criteria C1–C9 evaluated from stored JSON payloads only (no hand
copied numbers).

```
src/tensorlbm/flapping_propulsion.py     analytic anchor (Theodorsen C(k), Garrick C_T/C_P/eta,
                                         thrust reversal line, GCI, energy ledger)
src/tensorlbm/oscillating_foil_lbm.py    LBM solver: D2Q9 BGK + Guo forcing + multi-direct
                                         forcing IBM, NACA 0012 with solid interior,
                                         force/power, CV flux, energy ledger, cycle stats
src/tensorlbm/oscillating_foil_criteria.py  criteria C1..C9 (pure functions, PASS/FAIL/WITHHELD)
src/tensorlbm/moving_boundary.py         run_oscillating_foil_verification() entry point
tests/test_oscillating_foil_analytic.py  analytic anchors + criteria (PASS *and* FAIL controls)
tests/test_oscillating_foil_lbm.py       Taylor–Green dissipation/ledger validation + smoke tests
```

## Quick start

```bash
# 1. analytic anchor tables (fast, pure Python)
python tools/analytic_anchor.py --out artifacts/analytic.json

# 2. one LBM case (writes artifacts/<label>.json with C_T, C_P, eta, ledger, SHA-256)
python run_case.py --label heave_h015_k1 --mode heave --k 1.0 --h0c 0.15 --cycles 8

# 3. the whole matrix + criteria (one process per case)
python run_sweep.py --workers 12

# 4. re-evaluate criteria from stored payloads (no re-run)
python tools/evaluate_only.py

# 5. figures + report
python make_figures.py
python make_fields.py --label fields_heave_k2 --k 2.0 --h0c 0.15   # vorticity snapshots
python make_report.py
```

## What the criteria mean

| id | question | verdict rule |
|---|---|---|
| C1 | does the measured thrust increment `dC_T = C_T + C_D0` match Garrick in the small-amplitude linear region? | rel. err <= 15 %, else `WITHHELD` when the analytic thrust is below the measured force noise floor (3 sigma of the per-cycle mean) |
| C1b | is `dC_T ~ (h0/c)^2` in the linear region? | observed amplitude ratio within 15 % of 4 |
| C2 | does `eta(k)` peak where Garrick peaks? | peak location within one grid step and peak value within 15 % |
| C3 | degenerate limits of the anchor (`k->0` quasi-steady, `C(0)=1`, `C(inf)=1/2`, pure-heave closed form) | all within 2 % / 1e-6 / 1e-3 / 1e-10 |
| C4 | closed-box energy ledger `W = dKE + Phi` | residual <= 5 % at the fine resolution **and** decreasing with resolution |
| C5 | is the mean thrust monotone in `St`? | no drop larger than the combined 3 sigma noise, Spearman >= 0.9 |
| C5b | does a purely pitching section reverse thrust between `k=0.5` and `k=1`? | sign pattern matches theory, sign change present, monotone in `k` |
| C6 | grid convergence of `dC_T` on three chords | GCI <= 10 % and observed order 0.5 <= p <= 3 |
| C7 | negative-control trio: locked body (must be drag), IBM off (must be zero force), heave phase flip (mean thrust invariant) | all three behave as prescribed |
| C8 | bit-level reproducibility | identical SHA-256 of the physics payload (timing fields excluded) |
| C9 | IBM surface force vs an independent control-volume momentum balance | worst relative difference <= 15 % |

## Honest status of the current campaign

`artifacts/results.json` (29 cases, chord 32, Re 500, U 0.15, 8 cycles after a
2-cycle warm-up):

* **PASS** C3, C4, C5, C7, C8, C9
* **WITHHELD** C1, C1b — the analytic thrust in the linear region
  (`C_T ~ 1e-3`) is below the demonstrated force-measurement floor
  (`3 sigma ~ 1e-2`) of the immersed-boundary force at chord 32
* **FAIL** C2, C5b, C6 — the measured thrust increment is ~10x below Garrick at
  the amplitudes where the linear theory is already stretched, the pitching
  section does not reverse, and `dC_T` is not yet in the asymptotic grid range

Diagnosis (see `report.md` §3.4 and §5): the 4-point Peskin kernel smears the
3.8-cell-thick section, so the *effective* body is blunter than NACA 0012 —
`C_D0 = 0.25` versus 0.02–0.05 from laminar literature. Reducing the kernel
support to 2 points doubles `dC_T` (0.019 -> 0.043) and increases the input
power 8x, and `dC_T` grows monotonically with chord (c = 24/32/42:
-0.010 / 0.006 / 0.021). Next step: chord >= 96 with a 2-point kernel (or local
refinement at the leading edge) before re-running C1/C2.

## Notes / traps

* A **thin ellipse** (`t/c = 0.04`) is thinner than the kernel support at
  chord <= 32: the effective body becomes blunt, the drag is inflated and the
  motion hardly changes the mean force. Use a real NACA 0012 contour with a
  solid interior.
* The heave velocity amplitude is `h_dot_max = 2 k U (h0/c)`; weak
  compressibility requires `h_dot_max <~ 0.15`. `k=3` with `h0/c=0.25`
  (`h_dot_max = 0.225`) diverges — that NaN is kept in the payloads on purpose.
* The downstream control-volume plane must sit **outside** the outlet sponge,
  otherwise the flux estimate is biased by ~10 % (`tools/cvfix_probe.py`).
* Force histories of shedding cases have per-cycle standard deviations ~0.1:
  average at least 8 cycles, and compare against the noise floor, not against
  zero.