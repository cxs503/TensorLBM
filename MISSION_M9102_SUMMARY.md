# Mission m9102 — ELBM (entropic lattice Boltzmann) in TensorLBM

**Node**: `p485w1-n04-lbm-elbm-turbulence` · branch `agent/m9102` · model D3Q27

## What was built

| file | content |
|---|---|
| `src/tensorlbm/elbm.py` | entropic equilibria (closed-form D3Q27 Gauss, Lagrange-dual Newton for any lattice / Gauss+Bose-Einstein), entropy functionals `H_G`/`H_BE`/KL, per-cell entropy-condition solve, ELBM collision D3Q27 + D3Q19, effective-viscosity map |
| `src/tensorlbm/elbm_channel.py` | torch reference channel driver: Guo forcing with half-force shift, half-way bounce-back, wall-unit statistics, DNS gate function (criteria a–d) |
| `src/tensorlbm/elbm_channel_kernel.c` | fused OpenMP port of the same algorithm (O(10) Mcell/s vs ~0.1 Mcell/s for the torch path) |
| `src/tensorlbm/elbm_channel_native.py` | hash-cached build + ctypes bindings |
| `tests/test_elbm.py` | 20 acceptance tests (H-theorem, positivity, viscosity, laminar Poiseuille, kernel↔reference, DNS-gate self-test) |
| `tools/elbm_channel_report.py` | run → wall-unit profiles, criteria gates, CSV + figures |
| `tools/elbm_validation_report.py` | H-theorem / equilibrium / laminar figures + JSON |
| `tools/elbm_build_report.py` | assembles and POSTs the mission report |
| `docs/elbm_entropic_lbm.md` | method, conventions, validation chain, limits |

## Physics results

**Verified (REAL):**

* closed-form D3Q27 entropic equilibrium equals the independent Lagrange-dual
  minimiser to **1.7e-16**; moments reproduced exactly; strictly positive;
* H-theorem: `H(f*) ≤ H(f)` **per cell** for τ ∈ {0.5002 … 2.0} over 480
  distorted cells per case — `max ΔH = 0` (the solve lands exactly on the
  entropy root where the limiter is active);
* positivity: `min f > 0` after every collision in every test;
* `α ∈ (0, 1]`, `α = 1` ⇔ BGK; laminar Poiseuille recovered to **0.27 %**;
* shear-wave decay rate = `ν k²` within 5 %;
* OpenMP kernel = torch reference to **1.6e-5** relative per population;
* 20/20 pytest green.

**Turbulent channel vs MKM-1999 DNS — honest FAIL on the accuracy criteria.**
Three kernel runs (all with per-step entropy drop <= 1.1e-6, positivity, no
divergence):

| case | Re_tau | Delta+ | <alpha> | log slope (err) | y+(u_rms peak) | C_f err | max TKE/u_tau^2 | (d) stability |
|---|---|---|---|---|---|---|---|---|
| elbm_re180 | 180 | 5.6 | 0.266 | 0.984 (60%) | 19.7 | 62.6% | 0.079 | PASS |
| elbm_re180_small | 180 | 11.2 | 0.307 | 5.635 (131%) | 354.4 | 47.5% | 0.055 | PASS |
| elbm_re395 | 395 | 12.3 | 0.194 | 1.921 (21%) | 746.8 | 52.6% | 0.212 | PASS |

DNS references (same gate code): MKM 180 gives slope 1/kappa = 2.439 exactly by
construction of the fit on its own profile, u_rms peak at y+ = 15.28, C_f =
0.008136, max TKE/u_tau^2 = 4.10.

At every affordable resolution (Delta+ = 5.6 … 12.3) the entropy limiter
engages hard wherever the shear is unresolved (field-mean `alpha` = 0.19–0.31),
the extra dissipation thickens the wall layer and the flow **relaminarises**:
the mean profile becomes a plug (`U+ ~ 12.5` flat), the turbulent kinetic energy
is two orders of magnitude below DNS, the fitted log-law slope is meaningless
and `C_f` is 47–63 % off. No error criterion reaches the 3 % bar.

Diagnosis (the reusable knowledge): ELBM's H-theorem limiter is *not* a
sub-grid model — it removes the non-equilibrium content it cannot resolve, so
it needs Δ+ ≲ 2 (where `α → 1` and the BGK viscosity is recovered exactly, as
the laminar and shear-wave tests show). For coarse grids the correct
combination is the KBC decomposition already in
`tensorlbm.entropic_kbc` (shear modes relaxed at the exact BGK rate, only the
ghost modes entropically), which keeps production without the bulk dissipation.

## Artefacts

```
artifacts/elbm/validation/{validation.json,elbm_h_theorem.png,elbm_equilibrium.png,elbm_laminar.png}
artifacts/elbm/<case>/{gates.json,profile.csv,history.csv,summary.json,*_uplus.png,*_stresses.png,*_stability.png}
artifacts/elbm/raw/<case>.npz            # raw kernel output (all numbers traceable)
artifacts/elbm/logs/                     # run + pytest logs
```

## Reproduce

```bash
PYTHONPATH=src pytest tests/test_elbm.py -q
PYTHONPATH=src python tools/elbm_validation_report.py --out artifacts/elbm/validation
PYTHONPATH=src OMP_NUM_THREADS=64 python -m tensorlbm.elbm_channel --re-tau 180 --steps 50000  # torch path
# or the kernel:
LABEL=case RETAU=180 NX=64 NY=66 NZ=32 STEPS=26000 WARMUP=8000 python .m9102/run_case.py
PYTHONPATH=src python tools/elbm_channel_report.py artifacts/elbm/raw/case.npz --re-tau 180 --u-tau 0.012
```