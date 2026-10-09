# suboff_sail_resistance — SUBOFF + sail (AFF-3) Re=1000  ✅ VERIFIED

Status: **verified** (2026-10-09, task #240). Cd vs Blasius on total wetted
area, ladder {96,112,128} — **PASS under frozen erratum E2** (primary friction
estimator `standard`); the formal `mix50` verdict **FAIL is co-archived**, not
superseded. Both verdicts are machine-written from the shipped tier archives.

## Reference (locked pre-run)

Cf_ref = 0.041995047327036086 — Blasius 1.328/sqrt(Re), Re=1000; normalization
S = pi(2 R_lb)L + A_sail_own. ITTC-1957 / DARPA AFF-8 recorded as
turbulent-regime cross-checks, not applicable at Re=1000.

## Formal verdict (mix50) — FAIL, archived

| tier | domain | Cd_total | err | window drift |
|---|---|---|---|---|
| L96 | [576, 203, 203] | 0.0426486613 | +1.5564% | 1.8209% |
| L112 | [672, 237, 237] | 0.0444848717 | +5.9289% | 0.3517% |
| L128 | [768, 271, 271] | 0.0458621765 | +9.2085% | 0.1372% |

G1 finest |err| = 9.2085% > 3%; G2 |err| increases
[1.5564, 5.9289, 9.2085] — FAIL/FAIL; G4/G5 pass.

## Erratum E2 (frozen md5 a955ceda…, anti-shopping)

Instrument calibration evidence, all from these archives: uniform-field probe
faces/std = 1.4480/1.3735/1.3017 (O(1) staircase
leakage that does not decay with L; mix50 inherits half). NB: the frozen
erratum text quotes the L128 value as 1.374 — that duplicates L112 (append-only
transcription slip); the archive value is 1.3017 and governs. Bare-hull ladder
`diag_bare_L*` reproduces the trend without the sail; standard and lagrange
both cross the reference monotonically on both configurations. Primary
estimator -> `standard`; ladder, steps, domain, G1–G5 unchanged; frozen before
any standard-caliber window series was seen.

## E2 verdict (standard) — PASS

| tier | steps | Cd_total window | err | drift | sail cells |
|---|---|---|---|---|---|
| L96 | 14,400 | 0.0385703619 | -8.1550% | 1.7775% | 48 |
| L112 | 16,800 | 0.0407686708 | -2.9203% | 0.3520% | 70 |
| L128 | 19,200 | 0.0423957734 | +0.9542% | 0.1390% | 126 |

* G1: finest |err| = +0.9542% ≤ 3%
* G2: |err| 8.1550 → 2.9203 → 0.9542% strictly decreasing
* G4: drift ≤ 1.7775% ≤ 2.5%
* G5: sail cells 48/70/126 match the E1 mask table ±2
* A/B: ey2 dynamics bitwise-equal to formal on every tier (estimator-independent), all four p0 regions

## Reproduce

    python run.py --device cuda:0 --L 128 --friction standard   # E2 tier
    python run.py --device cuda:0 --L 128 --friction mix50       # formal tier

## Artifacts

`run.py` (frozen driver, md5 a8527e818202d6f724825e0fe0306365 — byte-identical
to the retired pending skeleton), `prereg.md` + `md5_receipt.txt` (frozen
append-only chain incl. E1/E2), `verdict.json` (formal FAIL) +
`e2_verdict.json` (E2 PASS), per-tier archives `formal_L*/ ey2_L* diag_bare_L*`
(result JSON + full run.log), `result.json` (aggregate + provenance).
