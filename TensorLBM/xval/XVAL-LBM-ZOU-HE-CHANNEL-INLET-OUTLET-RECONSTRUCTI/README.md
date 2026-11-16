# XVAL-LBM-ZOU-HE-CHANNEL-INLET-OUTLET-RECONSTRUCTI

Cross-validation of `loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow`'s
**zou-he-channel-inlet-outlet-reconstruction** against the exact plane-Poiseuille
solution of a pressure-driven channel.

* `src/tensorlbm/boundary/zou_he_channel.py` — Zou-He (1997) reconstruction for
  all four walls, two modes (`velocity`: prescribe `u`, reconstruct `rho`;
  `pressure`: prescribe `rho`, reconstruct the normal velocity), vectorised over
  the boundary layer so a parabolic inlet profile can be imposed directly, plus
  `zou_he_channel()` wiring the west/east pair.
* `src/tensorlbm/verification/zou_he_channel/solver.py` — D2Q9/BGK channel with
  half-way bounce-back walls, no body force, no periodic wrap; both ends closed
  by the reconstruction.  Block time averaging is used because soft pressure
  ends do not damp acoustic waves.
* `tests/test_zou_he_channel.py` — moment-exactness tests per wall and mode
  (they assert the *moments* the closure claims to impose, which is what caught
  the diagonal-pairing defect), plus channel-vs-Poiseuille acceptance tests.
* `case.py` / `campaign.py` — run one arm / a shard of the campaign; records land
  in `artifacts/xval_zouhe/`.
* `make_report.py` — aggregates the JSON records into `REPORT.md` and
  `figures/*.png`; every number in the report is read back from the records.

* `refinement.py` / `run_ladder.sh` / `make_refinement_report.py` — the
  grid-refinement (h-) study added after the campaign: fixed `L/H = 8`, longer
  averaging, residual telemetry, observed order of accuracy.  Output:
  `REFINEMENT.md` + `figures/fig4..fig6`.

Reproduce (light, ~30 s per case):

```bash
cd TensorLBM
PYTHONPATH=src python xval/XVAL-LBM-ZOU-HE-CHANNEL-INLET-OUTLET-RECONSTRUCTI/case.py \
    --arm pressure-pressure --height 16 --length 128 --steps 8000 \
    --burn-in 2000 --avg-window 500 --ramp 0 \
    --out artifacts/xval_zouhe/case_pp_H16.json \
    --npz artifacts/xval_zouhe/field_pp_H16.npz
PYTHONPATH=src python xval/XVAL-LBM-ZOU-HE-CHANNEL-INLET-OUTLET-RECONSTRUCTI/make_report.py
```

Headline (pressure-driven arm, `drho = 4e-3`, `tau = 0.65`): `u_x,max` within
**0.35 %** (H = 16) and **0.23 %** (H = 24) of the exact parabola, interior mass
flux constant to `4e-7` relative, fitted pressure gradient within `2e-5` of the
nominal `drho/(nx-1)`.  See `REPORT.md` for the full table, the figures and the
negative results (inlet closure is *not* the error-dominant term).
## Grid-refinement follow-up

`REFINEMENT.md` reports the h-study (H = 16, 24, 32 at `L/H = 8`): the mean-field
metrics are insensitive to the averaging schedule (0.02 % under a doubled
window, 0.07 % against the independent campaign record), the shape-only profile
error converges at **p = 2.05** and the raw max-error at **p = 1.53** -- i.e. the
Zou-He channel closure is a consistent second-order-except-amplitude discretisation,
and the 0.35 % campaign number is discretisation error, not averaging noise.

```bash
bash xval/XVAL-LBM-ZOU-HE-CHANNEL-INLET-OUTLET-RECONSTRUCTI/run_ladder.sh
PYTHONPATH=src python xval/XVAL-LBM-ZOU-HE-CHANNEL-INLET-OUTLET-RECONSTRUCTI/make_refinement_report.py
PYTHONPATH=src python -m pytest tests/test_zou_he_channel_refinement.py -q
```

## Validity envelope + independent reproduction

`REPORT.md` validates one point of parameter space (`drho = 4e-3`, `tau = 0.65`)
and `REFINEMENT.md` refines `H` at that point.  `ENVELOPE.md` adds the two axes
the closure actually feels and a second code path:

* `independent.py` — the same case re-solved in pure NumPy with the textbook
  D2Q9 table, the Zou-He algebra re-derived from mass + x-momentum + y-momentum
  and loop-based streaming; imports nothing from `tensorlbm`.  It reproduces the
  library record to round-off (profile deviation `2e-15`), so the headline
  number is a property of the method, not of one implementation.
* `run_envelope.sh` — 11 library runs: `drho` in `[4e-4, 1.6e-2]` (Ma 0.0012 to
  0.047) at `tau = 0.65`, and `tau` in `[0.55, 1.50]` at `drho = 4e-3`.
* `make_envelope_report.py` — aggregates them into `ENVELOPE.md` +
  `figures/fig7..fig9` (every number read back from the JSON records).

Headline of the envelope: all 11 cells are inside the 3 % target, but `tau` is
the real axis — `1.31e-3` at `tau = 1.0` up to `1.70e-2` at `tau = 1.5`, i.e.
13x the floor, while a 40x change of `drho` costs only 1.44x.  The error is
explained by one measured wall-position parameter,

    delta(tau) = 1/64 - (tau - 1/2)^2/12      (max residual 1.1e-4, tau in [0.55, 1.5])
    rel_umax  ~= 4 |delta| / H                (matches all six tau cells to <=17 %)

so at these resolutions the error budget is *wall position*, and the Zou-He
closure adds nothing measurable on top of it — the quantitative form of the
negative result in `REPORT.md` section 5.

```bash
bash xval/XVAL-LBM-ZOU-HE-CHANNEL-INLET-OUTLET-RECONSTRUCTI/run_envelope.sh
python3 xval/XVAL-LBM-ZOU-HE-CHANNEL-INLET-OUTLET-RECONSTRUCTI/make_envelope_report.py
PYTHONPATH=src python3 xval/XVAL-LBM-ZOU-HE-CHANNEL-INLET-OUTLET-RECONSTRUCTI/independent.py \
    --height 16 --length 128 --steps 8000 --burn-in 2000 --avg-window 500 \
    --out artifacts/xval_zouhe/indep_H16.json
PYTHONPATH=src python3 -m pytest tests/test_zou_he_channel_envelope.py -q
```
