# XVAL-LBM-HUANG2007-CONTACT-ANGLE-INVERSION — hold-out + transferability round (m22445)

Anchor: `XVAL-LBM-HUANG2007-CONTACT-ANGLE-INVERSION` (je-santos/MPLBM-UT
`huang2007-contact-angle-inversion`: calibrate θ(f_w), then invert).

Deliverable directory:
[`benchmarks/xval_huang2007_contact_angle_inversion/holdout/`](../benchmarks/xval_huang2007_contact_angle_inversion/holdout/REPORT.md)
(`predeclared.json` gates → `run_xval.py` → `evidence/*.json` → `summary.json` → `REPORT.md`).

## What this round checked (verification debt of the merged m18090 deliverable)

| # | question | answer (all programmatic) |
|---|---|---|
| 1 | Is the coarse calibration curve monotone in cos θ? | **No.** d(cos) = −0.0122 on f_w∈[0.60, 0.65]; the target window θ∈[95.16°, 95.86°] has **3 consistency branches** |
| 2 | Does the non-monotonicity mean the inversion returns a wrong wall fraction? | **No for the angle** — both extreme branches reproduce 95.5° to 0.36 % / 0.38° spread 0.70°. It is estimator noise (~0.7°), but it does make `invert()` multi-valued, so it is now exposed instead of silent |
| 3 | Does the calibration transfer to another viscosity? | **Yes, exactly.** τ=1.2 reproduces the τ=1.0 angles (83.06° / 105.22°, identical) |
| 4 | Does it transfer to the production grid? | **Inside the gate but degraded.** 90°: 7.71 % → 11.03 %; 120°: 2.57 % → 7.97 % (1.2–4.3× worse, 90° near the gate) |

11/11 measured hold-out rows are inside the owner 15 % gate (max 11.03 %); only 4/11 are
inside a strict 3 % band, and no resolution-transfer row is — stated, not glossed.

## Library change (minimal, additive)

`src/tensorlbm/contact_angle_inversion/`:

* `inversion_branches(cal, theta)` — every wall fraction whose calibration curve crosses the
  target cos θ (the ambiguity made explicit).
* `is_monotone(cal)` — whether the sweep is single-valued.
* `invert(..., on_ambiguous="first"|"last"|"mean"|"nan")` — `"first"` keeps the historical
  behaviour, `"nan"` refuses to pick one of several equally consistent branches.

Tests: `tests/test_contact_angle_inversion_ambiguity.py` (6 tests, includes an assertion that
every committed hold-out row is inside the gate, so the evidence cannot silently drift).

## Reproduce

```bash
cd benchmarks/xval_huang2007_contact_angle_inversion/holdout
python run_xval.py calibrate --threads 4   # 106.7 s, 7 points
python run_xval.py analyse
python run_xval.py matrix --workers 4      # 9 rows, ~2 min
python run_xval.py report
cd ../../..
pytest tests/test_contact_angle_inversion_ambiguity.py -q
```
