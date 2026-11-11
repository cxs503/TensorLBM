# B14 — `poiseuille_2d_bodyforce`

2-D plane Poiseuille flow (D2Q9 BGK, no-slip half-way bounce-back, constant body
force) validated against the exact incompressible parabola
`u(y) = g/(2 nu) y (H-y)`; includes the fix of a mass-sink defect in the Guo
forcing source term.

| | |
|---|---|
| solver | `tensorlbm.ice_hydro.lbm_channel.D2Q9Channel` (library code, reused as-is) |
| reference | analytic steady solution, re-checked inside this directory (`verify_reference`) |
| acceptance | pre-declared **2 %** on `eps_inf`, `eps_2`, `amp_err`, `curv_res`; `mass_drift <= 1e-6` |
| reference case | `H=32`, `tau=0.8`, `g=1e-5`, `Re=2.7`, `Ma=0.022`, 11 000 steps (adaptive steady gate) |
| result | `eps_inf = 8.99e-04`, `eps_2 = 1.23e-03`, `mass_drift = -6.9e-13`, fitted order 2.00 |
| tutorial | `docs/benchmarks/tutorials/poiseuille_2d_bodyforce.md` |

Reproduce (CPU, ~100 s, deterministic):

```bash
export PYTHONPATH=TensorLBM/src
python benchmarks/verified/poiseuille_2d_bodyforce/run.py res
python benchmarks/verified/poiseuille_2d_bodyforce/run.py sens --only tau,forcing
python benchmarks/verified/poiseuille_2d_bodyforce/run.py sens --only steps
python benchmarks/verified/poiseuille_2d_bodyforce/run.py sens --only legacy
python benchmarks/verified/poiseuille_2d_bodyforce/run.py merge
python benchmarks/verified/poiseuille_2d_bodyforce/make_figures.py
python benchmarks/verified/poiseuille_2d_bodyforce/make_doc.py
pytest TensorLBM/tests/test_poiseuille_2d_bodyforce.py -q
```

Artefacts: `result.json` (all metrics + profiles), `res_raw.json`,
`sens_raw_*.json`, `legacy_h64.json`, `figs/*.png`.

Defect note: before the fix the Guo source term was truncated to
`3 (c_ix - u_x)`, so `sum_i S_i = -3 u_x g != 0` — a per-step mass sink
(`mass_drift = -1.14e-02` after 30 k steps at `H=64`) that made the amplitude
drift with the step count, i.e. no steady state. The fix adds the zero-sum
partner `9 (c_i.u) c_ix`; `LegacyGuoSinkChannel` in `run.py` keeps the old
kernel as the negative control and `test_truncated_guo_source_is_detected`
guards against a regression.