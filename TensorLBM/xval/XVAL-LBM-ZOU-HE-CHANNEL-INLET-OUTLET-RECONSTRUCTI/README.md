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