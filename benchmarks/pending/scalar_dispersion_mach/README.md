# scalar_dispersion_mach — Pe=40 Taylor–Aris slit dispersion: the lattice-Mach law and the real cause of "+8.685 %"

**Node** `p405w1-n06-lbm-scalar-dispersion` · **module** `tensorlbm.scalar_dispersion_mach` ·
**doc** `docs/scalar_dispersion_mach_20261111.md` · **artifacts** `artifacts/scalar_dispersion_mach/`

## What was wrong

`docs/passive_scalar_verification_surface.md` (m8228) reported the slit-channel Taylor–Aris
dispersion at `Pe = 40` as **+8.685 %** and attributed it to the `O(Ma²)` lattice error of the
D2Q5 passive-scalar kernel. That attribution is wrong: the reading came from a **single-domain
modal meter with `kH = 2π·H/nx = 0.83`**, and the modal decay rate is only `k²·D_eff(0)` in the
long-wave limit. Measured here at fixed `Pe`, `H`, `D`:

| `L/H` | 7.5 | 15 | 30 | 60 |
|---|---|---|---|---|
| `kH` | 0.84 | 0.42 | 0.21 | 0.105 |
| error `[%]` | **+7.0** | −1.3 | −3.1 | −3.5 |

so the `+8.7 %` is a finite-`kH` artefact of the *meter*, and the two orders of magnitude of its
`kH` dependence dominate every Mach effect in the problem.

## What is measured instead

* `D_eff` from the modal meter, **extrapolated to `kH → 0`** (quadratic in `(kH)²` over the three
  smallest `kH`; observed refinement order 2.1).
* the **pure lattice-Mach law** on a plug-flow control (uniform advection, no shear, so
  `D_eff = D` exactly): `err = -3⟨u²⟩`, i.e. linear in `Ma²` (m6540's D2Q5 law, re-derived here
  from scratch on the same meter).
* the residual of the shear case is **Mach-independent** — a transverse-discretisation bias that
  shrinks with `H` (−3.5 % at `H=17`, −2.0 % at `H=20`, same `Pe=40`). This falsifies the seeded
  "error ~ Ma²" attribution for the `Pe=40` case; see `results.json.ablations.shear_case_mach_fit`.

## Gates

| gate | criterion |
|---|---|
| G1 | `Pe=40`, shipped kernel, `u ≤ 0.05`, `kH→0`: \|err\| ≤ 3 % |
| G1b | `Pe=40`, `u = 0.094` with the `O(u²)` Hermite equilibrium: \|err\| ≤ 3 % |
| G2 | plug-flow control: `err(Ma²)` linear, `R² ≥ 0.99` |
| G2b | corrected kernel removes the residual Mach law |
| G3 / G3b | `Pe ∈ [10,100]`: max \|err\| ≤ 3 % (shipped / corrected) |
| G4 / G4b | `Sc ∈ [0.1,10]` at `Pe=10`: max \|err\| ≤ 3 % (shipped / corrected) |
| G5 | refinement order in `(kH)²` ≥ 2, transverse discrete-Aris order ≥ 2 |
| G6 | `pytest tests/test_scalar_dispersion_mach.py` — 16 tests |

## Run

```bash
python verify_scalar_dispersion_fix.py --list
python verify_scalar_dispersion_fix.py --parallel 12 --out-dir artifacts/scalar_dispersion_mach
python verify_scalar_dispersion_fix.py --assemble  --out-dir artifacts/scalar_dispersion_mach
python verify_scalar_dispersion_fix.py --figs      --out-dir artifacts/scalar_dispersion_mach
python make_doc.py                                  # docs/scalar_dispersion_mach_20261111.md
pytest -q tests/test_scalar_dispersion_mach.py
```

`--include-optional` adds the expensive refinement points (`kH = 0.105` at high `Pe`, the
`L/H = 60` designed case). Cases are cached individually in `<out-dir>/case_NNN.json`, so an
interrupted scan resumes where it stopped.

## Iron rule

Every physics kernel comes from the library (`tensorlbm.thermal.temperature_*`,
`tensorlbm.taylor_aris.aris_dispersion_coefficient`); the only self-written numerical piece is a
LAPACK-free Thomas solve for the discrete Aris problem (cross-checked against the library solver,
because m8228 measured garbage solves from the cluster loong64 LAPACK for pinned tridiagonal
systems).