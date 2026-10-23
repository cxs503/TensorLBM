# backward_step — reference audit (Gartling 1990, Re=800, expansion ratio 2:1)

**Adopted reference (2-D, incompressible, laminar, ER = 2):**

| quantity | value | sources |
|---|---|---|
| X1 (primary lower-wall reattachment, /H) | **6.10** | Gartling 1990 (as tabulated by ECN/TNO E–11–042 §4), Gresho et al. 1993 |
| X1 (same quantity, /h_step) | 12.20 | Gartling 1990, Gresho et al., Keskar & Lyn 1999 (12.19), Grigoriev & Dargush 1999 (12.18) |
| X2 (upper-wall separation, /H) | 4.85 | Gartling 1990 (ECN/TNO E–11–042 §4) |
| X3 (upper-wall reattachment, /H) | 10.48 | Gartling 1990 (ECN/TNO E–11–042 §4) |

## Caliber (this is the whole story)

Gartling's nondimensionalisation uses the **downstream channel height `H`** and the
**mean inlet velocity `ū`**:

```
geometry : channel height H, inlet = upper half, step height h = H/2  (ER = 2)
inlet    : fully developed parabolic, u = 24 y (H/2 − y) → ū = 1, u_max = 1.5
Reynolds : Re = ū H / ν = 800        (based on H, NOT on the step height)
```

* `X1 = 6.10` is expressed in units of `H`. In units of the **step height** `h = H/2`
  the identical number is **`X1/h = 12.20`** (both figures appear in the literature —
  do not mix them; a solver that reproduces 12.2 h necessarily reproduces 6.1 H).
* The task brief writes "X_r/h ≈ 6.1". The only self-consistent reading is
  **X_r / H_channel = 6.10** (h = channel/inlet height here). Normalising the *step*
  height against 6.1 would be a factor-2 caliber error — the same class of mistake
  that produced the false failures catalogued in `benchmarks/STATUS.md`.

## Sources (original tables, extracted verbatim where possible)

1. **ECN/TNO report ECN-E–11-042 (2011), §4 "Flow over a backward-facing step"**
   (open-access PDF `publications.tno.nl/…/e11042.pdf`): reproduces Gartling's
   benchmark and states verbatim
   *"the values provided by Gartling [9]: X1 = 6.10, X2 = 4.85, X3 = 10.48"*,
   with `Re = ūH/ν = 800`, `u_max = 1.5`, `ū = 1`, ER = 2, L/H = 7.5/15/30
   (solution independent of L once L/H > 7).
2. **arXiv:2507.16509 (2025), "A Finite Volume … Benchmarking MHD Flows over
   Backward-Facing Steps"**, comparison table (quantity `(x_rs − x_s)/h`, i.e.
   step-height units): Gartling **12.20**, Gresho et al. **12.20**,
   Keskar & Lyn **12.19**, Grigoriev & Dargush **12.18**, Erturk 11.834,
   Kim & Moin 11.90. Geometry stated: L = 15, H = 1, **h = 0.5 → ER = 2**.
   12.20 × (h/H = 0.5) = **6.10 H** — agrees with source 1 exactly.
3. Gartling, D. K. (1990), *A test problem for outflow boundary conditions — flow
   over a backward-facing step*, Int. J. Numer. Meth. Fluids **11**(7):953–967
   (the primary source, cited by both of the above).

## Why the step-height cluster is tight (12.18–12.20)

Codes that use Gartling's exact setup (inlet profile imposed at the step plane, no
upstream channel) agree to two decimals. Erturk (2008) and Kim & Moin (1985) use a
different inlet treatment (an added upstream channel / different profile), which shifts
the reattachment ~2–3 % low — **excluded from the accepted cluster**, documented here so
the ~2–3 % spread is not mistaken for solver error.

## Acceptance

Observable = **X1 / H_channel** vs reference **6.10** (equivalent to X1/h_step vs 12.20).
Two grids, each within 3 %, and the two-grid span within 3 %.