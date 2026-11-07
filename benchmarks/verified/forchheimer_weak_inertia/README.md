# forchheimer_weak_inertia — self-produced weak-inertia drag reference

**Status: verified campaign (DAG node IN44-N02).**

The inertial (Forchheimer) correction of a periodic cylinder array is measured
with the library D2Q9/TRT solver and judged against a reference that is
**generated in-house**, not digitised from a published figure: an analytic
asymptotic structure (weak-inertia theorem, dilute-lattice logarithm) plus
self-consistent extrapolations (time tail, Mach number, viscosity).

## Question

For a periodic array driven by a body force `a`,

```
R(Re) = (a/U_s)|_Re / (a/U_s)|_Stokes = c0 + c1 Re + c2 Re^2 + O(Re^3)
Re    = U_s D / nu ,  D = 2R nominal cylinder diameter
```

* is `c1` zero, as the weak-inertia theorem demands for a centro-symmetric
  microstructure (Mei & Auriault 1991; Wodie & Levy 1991; Koch & Ladd 1997)?
* does the self-produced quadratic reference predict held-out Reynolds numbers?
* how large is the compressibility (`Ma -> 0`) correction at fixed `Re`?
* what does `R(Re, phi)` look like for square `phi = 0.40` and hex `phi = 0.55/0.60`?
* does the drag return to the Stokes value as `Re -> 0`?

## Method (three ingredients that make a 1e-4 signal measurable)

1. **Common protocol.** Every case of a group runs the *same* number of steps
   from rest with the *same* viscosity; only the body force changes. A Stokes
   response to a step force is linear, `u(t) = a G(t)`, so the relative transient
   is identical across the group and cancels exactly in the resistance ratio.
2. **Exact float64 weights.** The library weight table is stored as `float32`;
   converting it to float64 keeps a `1 + 7.45e-9` bias, which pumps mass into the
   `float64` equilibrium and shows up as a linear density ramp
   (`+7.45e-9/tau` per step — 5e-5 over 1e5 steps). Fixed in
   `tensorlbm.d2q9` (dtype-aware exact table) after measuring the ramp.
3. **Viscosity ladder.** At fixed `Re` the drag *must not* depend on `nu`. A
   dedicated ladder (`nu = 0.2 … 4`, same `Re`, same geometry) exposes a purely
   numerical `ln nu` systematic of the staircase wall position under the split
   forcing (`1/f = A + B ln nu`, ~0.2 % fit residual; a constant or linear-in-`nu`
   model is off by 2.9 % / 5.1 %). It cancels inside a fixed-`nu` group and is
   de-trended, not fitted away, for the Mach sweep — the only group that must
   vary `nu`.

## Running

```
# one case (parallelisable; each writes case_<id>.json)
python3 benchmarks/verified/forchheimer_weak_inertia/run.py plan
python3 benchmarks/verified/forchheimer_weak_inertia/run.py case --spec '<json>'
# aggregate: result.json, gates.json, figures/
python3 benchmarks/verified/forchheimer_weak_inertia/run.py report
```

Physics kernels are the library's (`collide_trt` / `stream` /
`bounce_back_cells` / `equilibrium`); the module
`tensorlbm.porous_weak_inertia` adds geometry, the force scheme, the protocol
and the estimators.

## Artefacts

| file | content |
|---|---|
| `result.json` | every case, every gate, all checks, provenance |
| `gates.json` | gate verdicts with the numbers behind them |
| `figures/` | drag-excess law, Mach extrapolation, `R(Re, phi)`, dilute log law, velocity field |
| `cases_manifest.json` | case inventory |

Narrative: `docs/verification/forchheimer_weak_inertia.md`.