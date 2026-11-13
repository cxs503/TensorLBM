# XVAL-LBM-DAM-BREAK-SLOT-MUSCL2-LOCAL-HLL-SSPRK2

Cross-validation of the in-house second-order shallow-water dam-break solver
(`src/tensorlbm/shallow_water.py`, `scheme="muscl2"`: minmod-MUSCL + local-speed
HLL + SSP-RK2) against the mature open-source suite **Clawpack 5.14**
(`shallow_hlle_1D` / `shallow_roe_with_efix_1D` + `ClawSolver1D` MUSCL-Hancock,
`SharpClawSolver1D` WENO5+SSP104).

* `predeclared.json` — decision / hypothesis / metrics / thresholds, written
  before the OSS arm was run.
* `run_arms.py` — six arms × resolution ladder × CFL sweep (~30 s CPU).
* `analyze.py` — metrics, criteria, `artifacts/result.json`, `artifacts/table.csv`,
  two figures under `figures/`.
* `make_doc.py` — report `docs/XVAL-LBM-DAM-BREAK-SLOT-MUSCL2-LOCAL-HLL-SSPRK2.md`
  (all numbers injected from `result.json`).
* `artifacts/` — arm profiles + aggregated result (versioned, `runs/` is gitignored).

```bash
pip install clawpack                       # 5.14.0; source build on loongarch64
python xval/dam_break_slot_muscl2_local_hll_ssprk2/run_arms.py
python xval/dam_break_slot_muscl2_local_hll_ssprk2/analyze.py
python xval/dam_break_slot_muscl2_local_hll_ssprk2/make_doc.py
pytest tests/test_xval_dam_break_slot.py -q
```

Verdict **PASS**: scheme-level agreement with the independent OSS code is
0.29–1.86 % (< 3 %), the raw 7.2–9.3 % gap is fully attributed to the positive
dry floor the OSS Riemann solvers require (NaN at eps ≤ 1e-3).
