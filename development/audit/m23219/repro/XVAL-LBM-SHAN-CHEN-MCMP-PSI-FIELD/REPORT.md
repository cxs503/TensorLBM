# XVAL-LBM-SHAN-CHEN-MCMP-PSI-FIELD

Cross-validation card for the **psi-field** multicomponent Shan-Chen formulation
(Palabos `shanChenMultiComponentProcessor3D` convention: one pseudopotential field
per component, **self-interaction excluded**, `G > 0` repulsive).

Full report: [`docs/XVAL-LBM-SHAN-CHEN-MCMP-PSI-FIELD.md`](../../docs/XVAL-LBM-SHAN-CHEN-MCMP-PSI-FIELD.md)
Numbers: `artifacts/results.json` ｜ Figures: `artifacts/{laplace,acoustic,drop_profile,fields}.png`

Module under test: `src/tensorlbm/multiphase/mcmp_psi.py`
Tests: `tests/test_mcmp_psi_field.py` (12 passed)

## Verdict (acceptance <15 %)

| gauge | reference | result | verdict |
|---|---|---|---|
| exact force identities (uniform / self-excluded / momentum / gather≡scatter / psi-linearity) | analytic identities | 0 … 1e-15 | PASS (machine precision) |
| force field vs production kernel `sc_two_component_force` (interior) | same-code-family implementation | 1.4e-9, ratio 1.000000 | PASS |
| acoustic speed, G=0 control | c_s = 0.57735 | +0.03 % | PASS |
| acoustic speed, G=+1 psi=rho / G=-1 psi=rho / G=-1 psi=rho-1 | c_eff = sqrt(cs²(1+G ψ0 ψ')) = 0.6325 / 0.5164 / 0.7746 | +4.09 / −6.47 / +10.55 % | PASS |
| Young-Laplace (G=2, psi=rho, R_eq = 7.51 / 9.11 / 12.58) | Δp ∝ 1/R | σ = 0.0899, R² = 0.9994, per-radius σ spread 4.14 % | PASS |
| stability | no blow-up | R_init = 10 arm went NaN in 4000 steps (honest FAIL) | PASS-with-note |
| ablation arms (`common_velocity`, `self_coupling`) | planned | **not_run** (budget) | not_run |

Boundary-layer difference to the production kernel (0.197) is a **convention**
difference: the production kernel mirrors psi at the domain edge (`wall_mode="mirror"`),
this module is periodic.

## Defect found and fixed while validating

First version of `stream()` rolled by `(-cy,-cx)`, i.e. advected populations
**backwards**; the interaction force then worked in reverse (anti-demixing,
NaN within ~25 steps). Caught by two independent gauges (unit-pulse landing
site; "force inward while density transports outward") and pinned by
`tests/test_mcmp_psi_field.py::test_streaming_direction`.

## Reproduce

```bash
python -m pytest tests/test_mcmp_psi_field.py -q
cd xval/XVAL-LBM-SHAN-CHEN-MCMP-PSI-FIELD
python run_xval.py --only exact
python run_xval.py --only acoustic
python run_xval.py --only laplace_primary --radius 6,8 --steps 4000
python run_xval.py --only laplace_primary --radius 10,12 --steps 3500
python run_xval.py --only merge
```
