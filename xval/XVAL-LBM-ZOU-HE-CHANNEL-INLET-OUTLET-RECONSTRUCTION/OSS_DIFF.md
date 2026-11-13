# OSS / literature survey — Zou–He inlet/outlet **reconstruction**

Round 2 (m18793) of `XVAL-LBM-ZOU-HE-CHANNEL-INLET-OUTLET-RECONSTRUCTION`.
Round 1 (`m18664`) ended in an honest-FAIL with an empty `fail_stage`; the
research step below was therefore run **before** any implementation work, as
the anchor's round-2 protocol demands.

## ① Horizontal survey of the OSS cache (node-local `~/oss-cache`)

| repo | revision available | boundary-condition inventory (source) | Zou–He present? |
|---|---|---|---|
| `pylbm` | working tree + `.git` in `~/oss-cache/pylbm` | `pylbm/boundary.py`: `BounceBack`, `BFLBounceBack`, `LinearBounceBack`, `Neumann` (+ the `Equilibrium`-type on-node initialisation path). All are **node-based** closures — the unknown populations are never reconstructed from a linear system of mass/momentum constraints. | **No** |
| `Maarten-vd-Sande/lbm` | `~/oss-cache/Maarten-vd-Sande_lbm` | `lbm/update_types.py`: `update_types` / `average_surround` only; inlets/outlets are imposed by overwriting with the equilibrium of the prescribed state. | **No** |
| `hschwane/GraSPH` | `~/oss-cache/hschwane/GraSPH` | SPH, not a lattice-Boltzmann code — out of scope (recorded for completeness). | **n/a** |

Evidence: `grep -ril "zou.he\|zou_he\|zouHe" ~/oss-cache` returns **no source
hit** (only binary pack/notebook matches inside `.git` objects and rendered
notebooks).  The inventory query named in the task
(`SELECT * FROM oss_inventory WHERE feature_slug ILIKE '%RECONSTRUCTI%'`) could
**not** be executed: the `oss_inventory` / `oss_solutions` /
`negative_knowledge_ledger` relations and the `/oss_solutions/search`,
`/knowledge_base/search` API routes do **not** exist in this environment
(`psql`: no such table; API: HTTP 404).  This is recorded as a *not-run*, not
as a passed check.

**Consequence for this anchor.** There is no OSS implementation of the Zou–He
reconstruction to copy the *structure* from, and therefore no OSS numeric
reference to diff against. The method spectrum we can legitimately transfer
from the OSS cache is the **alternative** family:
`pylbm`'s on-node `BounceBack` / `LinearBounceBack` (interpolated bounce-back,
Bouzidi-family) and `Neumann` closures, i.e. the "no reconstruction" branch.
That is exactly the branch the repo already offers as
`apply_simple_channel_boundaries` (equilibrium inlet + zero-gradient outlet),
which is kept in the repo as the comparison arm.

## ② Method genealogy (literature / textbook standard)

| element | canonical source | form used here |
|---|---|---|
| velocity inlet (west), unknown `f1, f5, f8` | Zou, Q. & He, X. (1997), *On the velocity and pressure boundary conditions of the lattice Boltzmann method*, Phys. Fluids **9**(6), 1591–1603, §II. | `rho = (f0+f2+f4+2(f3+f6+f7))/(1-ux)`, `f1 = f3 + (2/3) rho ux`, `f5 = f7 - (f2-f4)/2 + rho ux/6 + rho uy/2`, `f8 = f6 + (f2-f4)/2 + rho ux/6 - rho uy/2` |
| pressure/density outlet (east), unknown `f3, f6, f7` | Zou & He (1997) §III; Krüger, Kusumaatmaja, Kuzmin, Shardt, Silva & Viggen (2017), *The Lattice Boltzmann Method: Principles and Practice*, Springer, §8.2.2 eq. (8.44)–(8.46). | `ux = -1 + (f0+f2+f4+2(f1+f5+f8))/rho_out`, `f3 = f1 - (2/3) rho_out ux`, `f7 = f5 + (f2-f4)/2 - rho_out ux/6`, `f6 = f8 - (f2-f4)/2 - rho_out ux/6` |
| **pressure inlet (west)** — the missing half of the pair | Zou & He (1997) §IV (channel driven by a pressure difference); Krüger et al. (2017) §8.2.4 | `ux = 1 - (f0+f2+f4+2(f3+f6+f7))/rho_in`, then the same `f1, f5, f8` as the velocity inlet |
| tangential-velocity closure of the pressure BC | Krüger et al. (2017) §8.2.2 (the `±rho uy/2` term); Zou & He (1997) treat the `uy = 0` special case | `±rho uy/2` added to `f5/f8` (inlet) and `f7/f6` (outlet) |
| full-way bounce-back wall placement | Ladd (1994) J. Fluid Mech. **271**, 285; Ziegler (1993) J. Stat. Phys. **71**, 1171 | no-slip plane at the **wall-node centre** ⇒ channel width `H = ny - 1` |
| reference solution | Schlichting & Gersten, *Boundary-Layer Theory*, 8th ed., §5 (plane Poiseuille) | `u(y) = G y (H-y) / (2 nu)`, `G = dp/dx = (rho_in - rho_out)/3 / L`, `u_max = G H²/(8 nu)` |
| entrance-length check (why the measurement column is legitimate) | Schlichting §5: `L_e/H ~ 0.05 Re` | `Re = u_max H/nu <= 10` ⇒ `L_e/H <= 0.5`; measured at `L/H >= 6` |

The reconstruction is *not* an approximation with a tunable: it is the unique
solution of the three linear constraints (`sum f`, `sum f cx`, `sum f cy`) with
the non-equilibrium bounce-back closure `f_i^neq = f_opp(i)^neq` on one pair.
Hence the closure assertion in the test suite uses a round-off threshold
(`1e-12`), not a tolerance.

## ③ Method selection and transfer plan (what was actually adopted)

| option | verdict | reason |
|---|---|---|
| **(a)** OSS-style on-node equilibrium / interpolated bounce-back (pylbm `LinearBounceBack`) | rejected as the primary arm | mitigates the wall offset but does not solve the open-boundary problem at all: an inlet still needs an imposed state, and a zero-gradient outlet is reflective (mass drift) |
| **(b)** Zou–He reconstruction — already in the repo, extended here | **adopted** | exact mass/momentum closure at the boundary column, non-reflective by construction, and it is what `apply_zou_he_channel_boundaries` already exposes |
| **(c)** Extrapolation / NSCBC / convective outlet (`nscbc_outlet_2d`, `nonequilibrium_extrapolation_outlet_2d`, `zero_gradient_outlet_2d`, all already in the repo) | kept as documented alternatives | better for acoustic/compressible outflow; not needed for the laminar channel benchmark, which is the anchor's acceptance case |
| **(d)** Vendor an OSS implementation | not possible | no OSS Zou–He found (§①) |

Transfer plan executed in this round:

1. the pre-existing velocity-inlet/pressure-outlet pair stayed **bit-compatible**
   (`uy_out = 0` default reproduces the historical formula, asserted in
   `tests/test_zou_he_channel_reconstruction.py::test_outlet_pressure_default_matches_historical_formula`);
2. added the missing **pressure inlet** (`zou_he_inlet_pressure`) so that the
   textbook pressure-driven channel — the case the anchor is named after —
   can be run with Zou/He reconstruction at **both** ends;
3. added the tangential-velocity closure (`uy_out`) to the outlet so the
   outlet column no longer silently forces `uy = 0`;
4. added `apply_zou_he_pressure_channel_boundaries` as the drop-in companion of
   `apply_zou_he_channel_boundaries`;
5. validated all of it against the analytic Poiseuille solution, not against
   another code — `REPORT.md` §③–§⑤.