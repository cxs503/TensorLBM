# CASE-TAYLOR-GREEN-3D — Taylor-Green 3D decay on D3Q27 (case-pool entry)

Mission 15374 (case pool round 1). One-command reproduction:

```bash
python benchmarks/case_taylor_green_3d/run_case.py --all      # full sweep -> runs/case_taylor_green_3d/*.json
python benchmarks/case_taylor_green_3d/make_report.py         # JSON -> tables + figures + summary.md
```

The exactness proof is a separate, independent run:

```bash
python benchmarks/case_taylor_green_3d/run_case.py --mode verify   # spectral NS-residual test
```

## The case

Periodic box, lattice units `dx = dt = 1`, box `N³`, D3Q27 BGK, `float64`,
`ν = (τ − 1/2)/3`. Initial field: the z-invariant Taylor-Green mode

```
u(x,y,0) = A ( sin(kx) cos(ky), −cos(kx) sin(ky), 0 ),   k = 2π/N
```

## The reference (and why it is a *reference*)

For this field the 3D convective term is **curl-free** (a pure gradient), so it is
absorbed by `p = −(A²/4) e^{−4νk²t} (cos 2kx + cos 2ky)` and the NS equations reduce to
`∂u/∂t = ν ∇²u` with `∇²u = −|k|²u`. Hence, exactly,

```
u(t) = u(0) e^{−ν|k|²t},   |k|² = 2k²,   E(t) = E(0) e^{−2ν|k|²t}
```

`run_case.py --mode verify` **machine-checks** this instead of asserting it (spectral
derivatives at N=32, `float64`):

| case | curl(C)/‖C‖ | NS momentum residual | verdict |
|---|---|---|---|
| z-invariant TG (`tg`) | 2.7e−15 | 1.6e−14 | exact solution → usable as reference |
| classic z-dependent TG (`tg3d`) | 2.3e−01 | — | **no closed form** — cannot be a reference |
| Beltrami/ABC (`abc`) | 6.5e−15 | 2.3e−14 | exact, z-dependent, unsteady → companion case |

Because the z-dependent vortex has no closed form, the z-dependent 3D branch is
exercised by the **Beltrami (ABC) flow**
`u₀ = A(sin kz + cos ky, sin kx + cos kz, sin ky + cos kx)` — genuinely 3D, z-dependent,
and exact: `curl(u₀) = k u₀` (residual 2.1e−15) ⇒ `u(t) = u₀ e^{−νk²t}`.

**Zero hand-copying.** Every reference number in the tables is computed from the closed
form inside the run (`exact_fields`), not typed from a paper. The formula's provenance is
the classical Taylor–Green initial field (Taylor & Green 1937, *Proc. R. Soc. A*
158(895):499–512); the decaying closed form is the standard exact 2-D Navier–Stokes
solution, and here its exactness is re-established numerically by the residual test above
instead of being taken on trust.

## Metrics and tolerances (predeclared in `prereg.json`)

| id | metric | tolerance |
|---|---|---|
| M1 | relative error of `E(T)/E(0)` vs exact | < 8% |
| M2 | relative error of the fitted energy decay exponent vs `2ν|k|²` | < 8% |
| M3 | relative L2 velocity error at T | < 8% |
| M4 | observed convergence order (24→48) | ≥ 1.8 |
| M5 | mass drift `|M(T)−M(0)|/M(0)` | < 1e-12 |
| S1 | decay exponent correct for τ ∈ {0.55, 0.6, 0.7, 0.8} | < 8% each |
| S2 | L2 error growth with amplitude (Ma = A√3 up to 0.035) | reported table |

## Artefacts

* `prereg.json` — decision, alternatives, predeclared thresholds, deviations.
* `run_case.py` — the only executable behind every number.
* `make_report.py` — JSON → markdown tables + PNG figures (`runs/case_taylor_green_3d/figures`).
* `runs/case_taylor_green_3d/{exactness_verification,convergence,primary_decay,tau_sensitivity,amp_sensitivity,abc_zdependent}.json`
* `docs/CASE_TAYLOR_GREEN_3D.md` — the paper-style report (problem, method, parameter
  table, comparison table, sensitivity, failure modes, ecosystem frictions).
* `tests/test_case_taylor_green_3d.py` — CI guard: exactness flags + 200-step decay run
  inside tolerance on N ∈ {16, 24} (≈16 s).

## Scope note

The case validates the **D3Q27 bulk/periodic decay path** (equilibrium, BGK collision,
streaming, macroscopic recovery) against an exact solution, i.e. the numerical
viscosity/compressibility behaviour of the solver. It is not a wall-boundary or
turbulence-resolution test.