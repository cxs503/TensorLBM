# P162W1-N07 — oscillating-wing (flapping-foil) propulsion verification face

Verification face for flapping-foil propulsion in TensorLBM: closed-form
Theodorsen/Garrick analytic anchors → LBM thrust / input-power / efficiency →
energy budget → three-grid GCI.

## Layout

| file | what |
|---|---|
| `../../src/tensorlbm/flapping_propulsion.py` | analytic core: `C(k)=F+iG` (Hankel + Bessel routes), Theodorsen lift/moment, Garrick thrust/power/efficiency, thrust-reversal line, Roache GCI, energy ledger |
| `../../src/tensorlbm/flapping_lbm_2d.py` | D2Q9 BGK + Guo forcing + direct-forcing IBM around a thin prescribed-motion foil; momentum-exact force/power accounting; deterministic (bit-reproducible) |
| `run_case.py` | one case → JSON + CSV (measured coefficients, theory, energy budget, field snapshots) |
| `run_sweep.py` | all cases in parallel (20 procs) → `artifacts_p162n07/results.json` + criteria C1..C8 |
| `make_figures.py` | 5 report figures from `results/` |
| `make_report.py` | report markdown with programmatically injected numbers |
| `../../tests/test_flapping_propulsion.py` | 23 unit/smoke tests (analytic anchors, GCI, IBM sign conventions, reproducibility) |

## Conventions

* `h` positive **downward**, `L` positive **up**, `alpha` positive nose-up,
  pitch axis `a*b` from mid-chord (`a=-1` leading edge, `0` mid-chord).
* `k = ωc/(2U)`, `St = f·A/U` with `A = 2h0`.
* Thrust (measured) `C_T = <F_fluid,x>/(½ρU²c)`, positive = propulsive.
* Input power `C_P = <Σ F_marker·v_body>/(½ρU³c)`, positive = foil feeds the fluid.

## Run

```bash
python run_sweep.py --outdir artifacts_p162n07 --workers 20        # ~25 min on 128 cores
python make_figures.py --outdir artifacts_p162n07 --figdir figures
python make_report.py  --outdir artifacts_p162n07
python -m pytest ../../tests/test_flapping_propulsion.py -q
```

## Headline outcome

The analytic anchor face (Theodorsen/Garrick/efficiency/limits/reversal/GCI) is
verified to machine precision. The LBM-side quantitative criteria are **honest
FAIL** at the specified small amplitude: at `h0/c=0.02`, `alpha0=2°` the inviscid
Garrick thrust is `O(1e-3)` while the viscous drag of the same body is
`O(0.2)`, so a thrust/drag ratio of ~1% cannot be resolved to 5% by any viscous
solver. See `docs/P162W1_N07_FLAPPING_WING.md` for the full error table.