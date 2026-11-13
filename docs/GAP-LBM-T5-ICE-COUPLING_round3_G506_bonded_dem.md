# GAP-LBM-T5-ICE-COUPLING · round 3 (G5-06): 3-D breakable-ice (cohesive bond) DEM leg

*Mission m17705 · repo TensorLBM · branch `agent/m17705` · round 3 of the T5 flagship node.
Round 1 (m15659) built the 3-D load-exchange interface + invariant gates; round 2 (m17029,
G5-07) drove it with a real D3Q19 field and left **G5-06** open: "3-D bonded DEM (breakable
ice) absent — only 2-D bonded lattice". This round closes G5-06 for the DEM leg: the
`LinearSpringDashpot` contact law is normal-only (`f_n = max(0, ...)`, no tension), so a
bonded ice cover can never fracture; `BondedDEM` adds a tension-capable cohesive bond that
ruptures at a strain/force threshold and books the released fracture energy.*

**Status: G5-06 closed (B1–B5 gates PASS).** Every number below is injected programmatically
from `tests/test_dem_cohesive_bond.py` (9 passed).

## 1. Case / what was missing

The T5 acceptance decomposes into R1 (LBM water), R2 (FEM structure), R3 (**DEM ice that
fails/breaks**). The DEM leg (`src/tensorlbm/dem/contact.py::LinearSpringDashpot`) is
normal-only — `f_n = max(0, k δ - η v_n)` — so it can *repel* but never *fracture* a bonded
ice cover. The only breakable-ice code in the tree is 2-D (`ice_coupling_2d.py` via the
external `tensordem` package, and the legacy `ice_hydro/bonded_cover.py` on a duplicate
package root that is not importable from `src/tensorlbm`). G5-06 = the 3-D DEM leg has no
cohesive/breakable bond.

## 2. Method (what is new)

`src/tensorlbm/dem/contact.py::BondedDEM` — a breakable-ice DEM that operates on particle
*arrays* by index (the same layout `IceDEM3D` uses, so it drops straight into the
`three_way` DEM leg):

1. **Bond formation** — every pair of particles in contact at initialisation
   (`d ≤ r_i + r_j`) gets a `Bond` with `rest_length = r_i + r_j`.
2. **Tension-capable force** — an intact bond applies `F = k_b (d − rest) n̂` (tension if
   `d > rest`, compression if `d < rest`) plus a normal dashpot. This is the *only* place
   in the 3-D DEM that can pull particles apart.
3. **Rupture** — a bond breaks (irreversibly) when `|d−rest|/rest > breaking_strain`, or
   `|v_t|/rest > shear_breaking_strain`, or `|F| > bond_strength`. On rupture the pair
   reverts to the no-tension `LinearSpringDashpot` and the released elastic energy
   `0.5 k_b (d−rest)²` is booked into `fracture_energy_released` for the energy ledger.
4. **Fragment pattern** — a 3-particle chain under a tip pull breaks the *far* bond first
   (weakest link), leaving a 2-particle fragment + 1 free particle.

## 3. Gates (B1–B5) and results

| gate | assertion | result |
|---|---|---|
| B1 | contacting pairs form a bond; separated pairs do not | PASS |
| B2 | a bonded pair pulled apart carries a *tensile* force (no-tension law returns 0) | PASS |
| B3 | exceeding `breaking_strain` ruptures the bond irreversibly | PASS |
| B4 | a ruptured bond releases `0.5 k (d−rest)²`, summed to <1e-12 | PASS |
| B5 | a 3-particle chain under a tip pull breaks the far bond first | PASS |

## 4. Error / consistency

* Fracture-energy ledger: `0.5 k_b (d−rest)²` matches the analytic value to <1e-12
  (B4) — the energy bookkeeping is exact, not an approximation.
* No-tension fallback after rupture returns ~0 on further separation (B2/B3) — the
  post-fracture pair is physically a *free* fragment, not a re-bonded one.
* Existing three_way / LBM-DEM-FSI suites still pass (43 tests) — the addition is
  additive, no regression.

## 5. Not done / limits (honest)

* **Not yet wired into a full coupled run**: `BondedDEM` is a standalone, gated module;
  the `IceDEM3D` leg of `coupled3d.py` still uses the normal-only law. Wiring the bonded
  law into the end-to-end 3-D chain (so a real LBM field drives a *fracturing* ice cover)
  is the next round's step — it needs the D=20 G5 field number (cj1400, still queued) to
  be accuracy-adequate first.
* **No 3-D shell/solid FEM** (G5-05): the structural side is still a 1-D/modal beam; the
  interface is backend-agnostic so a shell backend can be added without changing the
  transfer algebra.
* **T1 multiphase** (G5-04) remains the blocking R4 prerequisite — unchanged.

## 6. Reproduction

```bash
PYTHONPATH=src python -m pytest tests/test_dem_cohesive_bond.py -q   # 9 passed
PYTHONPATH=src python -m pytest tests/test_three_way_coupled3d.py \
    tests/test_three_way_interface3d.py tests/test_lbm_dem_fsi_coupling.py -q  # 43 passed
```
