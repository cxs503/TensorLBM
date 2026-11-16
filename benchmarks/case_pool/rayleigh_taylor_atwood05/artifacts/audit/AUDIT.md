# Independent audit of the CASE-RT-INSTABILITY-ATWOOD evidence bundle

| id | check | value | status |
|---|---|---|---|
| A0 | audit closed forms vs ref/ref_lock.py | 0 | OK |
| A1 | linear sigma re-derived from history.csv (primary window) | 0.00277247 | OK |
| A2 | terminal Fr re-derived from front histories (tau in [4,6]) | 0.595584 | OK |
| A3 | terminal gate vs Layzer-Goncharov (C=1) | 5.5645 | PASS |
| A4 | C_eff = 1/(pi Fr^2) self-consistency | 0.897355 | OK |
| A5 | lattice Mach number from max_speed | 0.548258 | OK |
| A6 | Boussinesq front symmetry |h_bub - h_spk|/max | 0.000392355 | OK |
| A7 | velocity_in_window unit consistency (tau-slope vs lattice) | 1 | OK |
| A8 | F4 'nonlinear velocity unmeasurable' claim vs later artefacts | 1 | STALE |
| A9 | k xi -> 0 extrapolation of the sharp-interface residual (W, P) | 0 | OK |
| A10 | REPORT.md percentage numbers traced to artefacts | 8 | UNTRACED |

OK/PASS: 9  flagged: 2

## A0 — audit closed forms vs ref/ref_lock.py  [OK]
```json
{
  "sigma_inviscid": {
    "audit": 0.0031332853432887507,
    "ref_lock": 0.0031332853432887507,
    "rel_dev": 0.0
  },
  "sigma_viscous": {
    "audit": 0.003085464497242449,
    "ref_lock": 0.003085464497242449,
    "rel_dev": 0.0
  },
  "goncharov_spike": {
    "audit": 0.06383076486422923,
    "ref_lock": 0.06383076486422923,
    "rel_dev": 0.0
  },
  "goncharov_bubble": {
    "audit": 0.03685270927694245,
    "ref_lock": 0.03685270927694245,
    "rel_dev": 0.0
  }
}
```
## A1 — linear sigma re-derived from history.csv (primary window)  [OK]
```json
{
  "sigma_audit": 0.00277247256322754,
  "sigma_case_json": 0.00277247256322754,
  "rel_dev_pct": 0.0,
  "r2": 0.9996760230873188,
  "n_points": 85,
  "sigma_viscous_closed_form": 0.003085464497242449,
  "err_vs_viscous_pct": -10.144078283663195,
  "err_vs_viscous_case_json_pct": -10.144078283663195
}
```
## A2 — terminal Fr re-derived from front histories (tau in [4,6])  [OK]
```json
{
  "fr_audit": 0.5955839123635018,
  "fr_case_json": 0.5955839210752473,
  "rel_dev_pct": -1.4627234090625052e-06,
  "slope_bubble": 0.5957152906583081,
  "slope_spike": 0.5954525340686955,
  "r2_bubble": 0.9932706154414649,
  "r2_spike": 0.993291207261482,
  "n_points": 267,
  "tau_window": [
    4.0,
    6.0
  ]
}
```
## A3 — terminal gate vs Layzer-Goncharov (C=1)  [PASS]
```json
{
  "fr_audit": 0.5955839123635018,
  "fr_lz": 0.5641895835477563,
  "err_pct": 5.564499900606212,
  "gate_pct": 15.0,
  "case_json_gate": true
}
```
## A4 — C_eff = 1/(pi Fr^2) self-consistency  [OK]
```json
{
  "c_eff_audit": 0.8973548424233477,
  "c_eff_case_json": 0.8973548161717092,
  "note": "C_eff is an exact restatement of Fr, not an independent measurement"
}
```
## A5 — lattice Mach number from max_speed  [OK]
```json
{
  "ma_audit": 0.5482581664754297,
  "ma_case_json": 0.5482582056035612,
  "compressibility_error_scale_ma2_pct": 30.0587017107
}
```
## A6 — Boussinesq front symmetry |h_bub - h_spk|/max  [OK]
```json
{
  "max_rel_dev": 0.0003923550179661215,
  "case_json": 0.023471601433572988,
  "note": "the solved equations are invariant under (phi, y) -> (-phi, -y); a large asymmetry would be a measurement fault, not physics"
}
```
## A7 — velocity_in_window unit consistency (tau-slope vs lattice)  [OK]
```json
{
  "u_true": 0.0123,
  "u_with_tau_unit": 0.0123,
  "u_without_tau_unit": 9.84,
  "factor": 0.00125,
  "note": "the pre-fix signature (no tau_unit) is kept callable so the old numbers stay reproducible; callers must pass tau_unit"
}
```
## A8 — F4 'nonlinear velocity unmeasurable' claim vs later artefacts  [STALE]
```json
{
  "report_f4_row": "| F4 | \u975e\u7ebf\u6027\uff08\u52bf\u6d41\u7ec8\u901f\uff09\u652f\u81c2\u5728\u672c\u914d\u7f6e\u4e0b**\u6d4b\u4e0d\u5230** | A=1/2\u3001a\u2080=0.01\u03bb\u30018\u03bb \u76d2\u5b50 | \u754c\u9762\u5728 \u03c4\u22481.9 \u5373\u8fbe a/\u03bb=0.6 \u62a4\u680f\uff0c\u800c\u52bf\u6d41\u7ec8\u901f\u7a97 \u03c4\u2208[4,6] \u9700\u8981 a/\u03bb\u2248e^{9.9}\u00b70.01\u03bb \u226b \u76d2\u9ad8 | \u5982\u5b9e\u5224\u201c\u4e0d\u53ef\u6d4b\u201d\uff1b\u82e5\u8981\u590d\u73b0\u7ec8\u901f\uff0c\u9700 a\u2080/\u03bb \u964d\u5230 1e-5 \u91cf\u7ea7\u6216\u6539\u7528\u53ef\u538b\u7f29/\u591a\u76f8\u4e24\u6d41\u4f53\u6a21\u578b |",
  "late_artefact_exists": true,
  "late_tau_reached": 6.4575000000000005,
  "note": "REPORT.md was written before the late-time suite; the row is kept for provenance but is superseded by artifacts/late and by the similarity ladder (artifacts/simladder)"
}
```
## A9 — k xi -> 0 extrapolation of the sharp-interface residual (W, P)  [OK]
```json
{
  "W": {
    "intercept_pct": -5.53134300376742,
    "slope_pct_per_kxi": -30.374001149462096,
    "r2": 0.9987609490146199,
    "recorded_e0_pct": -5.53134300376742,
    "abs_dev_pp": 0.0,
    "n_points": 3
  },
  "P": {
    "intercept_pct": -3.2770334114125563,
    "slope_pct_per_kxi": -22.775682415115217,
    "r2": 0.9997796258056312,
    "recorded_e0_pct": -3.2770334114125563,
    "abs_dev_pp": 0.0,
    "n_points": 3
  },
  "note": "the W estimator extrapolates to -5.53 % and the P (peak instantaneous rate) estimator to -3.28 %; neither reaches the 3 % platform standard, both stay inside the +-15 % case gate"
}
```
## A10 — REPORT.md percentage numbers traced to artefacts  [UNTRACED]
```json
{
  "n_quoted": 42,
  "untraced": [
    "-13.56%",
    "-13.57%",
    "-13.61%",
    "-13.71%",
    "-14.53%",
    "-15.06%",
    "-15.29%",
    "-15.32%"
  ],
  "note": "markdown-only quantities (hand-written discussion numbers) are reported here instead of being assumed programmatic"
}
```
