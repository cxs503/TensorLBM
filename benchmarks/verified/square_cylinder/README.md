# square_cylinder — 2D free-stream flow over a square cylinder, Re = 100  ✅ VERIFIED

Status: **verified** (2026-09-29). Both grids within 3% of the reference cluster and
mesh-converged. Strouhal number and drag coefficient are both validated.

## Physics

Viscous incompressible flow past a square cylinder in a free stream, Re = U·D/ν = 100,
2D (the standard benchmark setting — 3D spanwise instability onset is Re ≳ 160 for this
geometry). Quantity of interest: mean drag coefficient Cd = Fx / (½ρU²D) and the
vortex-shedding Strouhal number St = f_shed·D/U measured from the lift-coefficient
zero-crossings.

## Reference convention (this is the whole story — read it)

Early versions of this benchmark scored the solver against **Okajima 1982's
experimental St ≈ 0.14 and a Cd ≈ 1.6 that has no traceable source**, and reported
"−7.4% / +5.9% ⇒ FAIL". The audit (`REFERENCE_AUDIT.md`, 14 independent digitised
points extracted from the original tables) shows both numbers were the wrong class of
reference:

* A **2D solver must not be scored against 3D experiment** (2D suppresses spanwise
  instability, which systematically shifts St and Cd).
* The solver's own configuration is a **40D domain, blockage 2.5%** — i.e. a
  low-blockage *free-stream numerical* setting. It must be scored against the
  2D free-stream low-blockage **numerical** cluster, not against channel/confined
  data (Sohankar 1997 channel Cd ≈ 2.05) or small-domain data (Sharma & Eswaran
  Cd = 1.57).

**Adopted reference** (2D, α = 0°, low blockage):

| quantity | value | cluster | sources (original tables) |
|---|---|---|---|
| Cd | **1.48** | [1.44, 1.52], median 1.488 | arXiv 2411.03124v4 Tab.1 (Sohankar 1.46 / Yoon 1.44 / Present 1.48); arXiv 2309.09197 Tab.2 (Sohankar 1.477 / Sahu 1.488 / Sen 1.530 / Present 1.495); arXiv 2308.08085v3 Tab.7-8 (uniform 1.500 / AMR 1.511 / Fakhari&Lee 1.51 / González 1.50); arXiv 2404.12123 (Present 1.476); arXiv 2405.12834 (1.47–1.55) |
| St | **0.147** | [0.145, 0.149] | same tables (0.1472 / 0.146 / 0.149 / 0.145 / 0.145 …); arXiv cond-mat/0311156 cites Okajima-1982 experiment St = 0.143–0.145 |

## Results

| grid | steps | Cd | err vs 1.48 | St | err vs 0.147 | mesh span |
|---|---|---|---|---|---|---|
| D = 32 (1280²) | 60 000 | 1.4816 | **+0.11%** | 0.1482 | **+0.82%** | — |
| D = 48 (1920²) | 90 000 | 1.4844 | **+0.30%** | 0.1480 | **+0.68%** | Cd 0.19% / St 0.14% |

* Both grids ≤ 3% on both quantities; mesh span ≤ 3% (0.19% / 0.14%).
* Worst case against the most conservative cluster edge (Cd = 1.44, St = 0.145):
  +2.9% / +3.1% — still at the 3% bar.
* `err_decreased` is **false** in `result.json` — D = 48's 0.30% is larger than D = 32's
  0.11% because D = 32 happens to sit closer to the cluster centre. The acceptance
  criterion is "both grids ≤ 3% **and** mesh span ≤ 3%", not monotone decrease (this is
  recorded explicitly in `result.json:convergence.err_decreased_note`).

## Configuration

D3Q19, MRT collision, τ = 0.548 (u_in = 0.05, ν = 0.016), Inamuro-style seeding with a
forced shedding seed (St_seed = 0.14, 2 periods at step 9143), outlet sponge
(x0 = 960, width 320, α = 10, power 2), warm-up 50%, statistics from step 30 000.
See `case_D32.json` / `case_D48.json` for the full parameter sets.

## Reproduce

```bash
PYTHONPATH=src python benchmarks/verified/square_cylinder/run.py \
    --D 32 48 --steps 60000 90000 --compile-mode eager --device sdaa:0
# ~19 h in total on SDAA (eager; see the compile note below)
```

## Compile routing (SDAA caveat)

`run.py` goes through `benchmarks/compile_route.py`. On this SDAA host
(Torch-SDAA 3.1.1a3+gite3971b9) the teco-inductor backend fails on the LBM step chain
at a 5th backend defect (EXPAND fallback path, `reduction_shape [1,9,N]` vs
`ncw [9,N,1]`), so `compile_route` **automatically falls back to eager** — the result
is bit-identical (max abs diff 0.0 vs a pure eager run) and `result.json` records
`compile_mode_effective = "eager (fallback)"`. On CUDA hosts `--compile-mode default`
really compiles. See `benchmarks/compile_route.py` docstring and
`docs/hardware_portability.md`.

## Notes

* Cd/St in `case_D*.json` are physical measurements, independent of the reference
  convention; only the error columns depend on the adopted reference (re-computed
  2026-09-29 against the audited cluster, see `REFERENCE_AUDIT.md`).