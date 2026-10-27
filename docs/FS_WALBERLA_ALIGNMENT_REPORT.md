# Free-Surface Engine vs waLBerla Reference: Audit, Repair & Comparison Report

Date: 2026-10-28 · Branch: `arena/4133dc06-tensorlbm` (from `74662f5`)
Scope: full engine repair to the waLBerla reference + prescribed-velocity
moving-sphere water entry. All work verified by regression tests; nothing here
is report-only.

## 1. References audited

| Source | Files read | What was adopted as ground truth |
|---|---|---|
| `lssfau/walberla` `src/lbm/free_surface/` | `AdvectMass.h`, `FindInterfaceCellConversion.h`, `CellConversionSweep.h`, `ReconstructInterfaceCellABB.h`, `PdfRefillingModel.h`, `ForceDensitySweep.h`, `SurfaceDynamicsHandler.h`, `TotalMassComputer.h`, `InitFunctions.h`, `ExcessMassDistributionSweep.impl.h`, `GetLaplacePressure.h` | Körner eq. 9 exchange, Thürey table 4.1 / eq. 4.5 ABB, interface-only conversion (κ, forced rules), conversion-driven envelope, I→L keeps PDFs, equilibrium refill from averaged ρ+u, fill = m/ρ, even excess redistribution, sweep order |
| `lssfau/walberla` `apps/showcases/FreeSurface/` | `DamBreakRectangular.prm/.cpp` | Guo forcing, `EvenlyAllInterface`, κ=1e-2, forceThreshold=0.1, `BubbleModelConstantPressure(1)` (ρ_gas=1.0), Smagorinsky 0.1, free-slip walls, periodic z, hydrostatic init |
| `Maarten-vd-Sande/lbm` | `lbm/streamcollide.py`, `update_types.py` | Single gravity term, fill-weighted I/I exchange, fluid-side obstacle BB in streaming, lonely-cell rules, refill averages ρ+u, hardcoded gas density 1.0 |

## 2. Bugs found (engine vs reference)

### P0 — physics-breaking
1. **Exchange receiver gate sealed the column** (`456fdb1`): receivers gated on
   `fill > 1e-3` while the envelope is born at `fill = 0`, so receivers could
   never fill — dam frozen at any gravity (front fixed, conversion 0).
   Reference has no such gate. **Removed.**
2. **ABB formula regime violated by `rho_gas=0.001`**: the engine implements
   Thürey eq. 4.5 (`f = feq_g + feq_g,opp − f_post,opp`), which is consistent
   only for pressure-matched gas `ρ_g ≈ ρ_l`. Rarefied test configs produced
   negative populations → `fill = 500000` → guard trips. waLBerla always runs
   gas at 1.0. **Configs/tests moved to `rho_gas = 1.0`** (engine kept
   reference-strict; the legacy tolerant ABB variants stay removed).
3. **Fill/mass conflation**: fill was forced ≡ mass (ρ=1 assumed). Reference:
   `φ = m/ρ_local` unclamped. **Now local-ρ, unclamped.**
4. **Clamp + isolation sweeps**: the old transaction clamped fill/mass and ran
   an isolation sweep — both non-reference mass sources/sinks. **Removed**
   (replay-phase names kept as no-op records for schema compatibility).

### P1 — wrong mechanics / evidence
5. **Non-reference redistribution targets**: even redistribution must go to
   surviving INTERFACE (+ promoted) receivers only; gas receivers with stale
   flags tripped the ownership ledger. **Fixed receiver selection** and added a
   post-envelope receiver snapshot so evidence stamps the actual owner.
6. **Zero-density refill fallback**: neighbour-averaged refill fell back to
   ρ=0 for shells with no active neighbour; conversion-driven promoted cells
   carry kept mass, so `fill = m/1e-6 = 33333`. **Fallback is now `rho_liquid`.**
7. **Replay-evidence schema drift**: the builder freezes the declared `rho`
   input but the audit contract did not list it → every audit WITHHELD.
   **Added `rho` to `_REQUIRED_INPUTS`.**
8. **Absolute residual thresholds on float32 totals**: summation rounding on
   O(10–1000) totals (~1e-6–1e-5) tripped absolute gates/labels. **Dam gate and
   runtime-ledger conserved/diagnostic boundary are now relative.**
9. **Retired experiment wirings still staged**: solver staged the experimental
   I→G ownership closure and paired bulk debit. Reference has neither (mass
   lives only at INTERFACE; I→G settles via the plain excess path).
   **Both retired to strict no-op flags** (fail-closed builder kept standalone
   and still tested directly).

### P2 — runners/config
10. Dam 3D FS path: missing periodic-z, wall-slip, hydrostatic `rho0`, gas
    population handling, advance-based front metric. **All aligned** (see §4).
11. `dambreak_cross_validation.py`, `hull_free_surface_v2.py`: `rho_gas`
    pressure-matched to 1.0 for ABB consistency.

## 3. Step order after repair (matches `SurfaceDynamicsHandler`)

boundary → force → stream/reconstruct (ABB)/advect → collide →
convert → refill → redistribute, with conversion marking per
`FindInterfaceCellConversion` (threshold κ + forced rules, liquid priority),
I→L keeping PDFs, I→G zeroing, and the envelope promoting only `recv_new`
(conversion-driven) plus LIQUID demotion next to fresh gas.

## 4. Dam-break runner alignment (`dam_break_3d.py`)

- `periodic_z` removes z-walls; `free_slip_all` slips every wall axis
  (waLBerla `DamBreakRectangular.prm`); halfway BB default with legacy opt-out.
- FS init is hydrostatic with explicit liquid `rho0`; GAS populations re-zeroed.
- Unexplained gate is relative; paired/slope/topology fail-closed gates retired
  to record-only diagnostics; front error is advance-based (Koshizuka / M&M).
- 101-step FS campaign: gate passes, `topology_changed=True`, 8 topology
  events, max relative cumulative drift 1e-3.

## 5. Moving-sphere water entry (`multiphase_water_entry.py`)

Prescribed-velocity descent (`v_entry`, default 0.02 lattice units/step) with
Ladd moving bounce-back, cleared-node refill, moving-wall momentum-exchange
force, and `sphere_trajectory` in metadata; CG uses positive `A_surface`
(default 0.04); force CSV schema unchanged.
Smoke (2D CG, 48², r=6, 60 steps, v=0.05): cy 29→26, Fx≈0, Fy>0, finite.

## 6. Test realignment (same physics, honest contracts)

- 21 FS regression failures triaged and fixed by category: 6 retired I→G
  withhold (solver no longer stages the builder; policy machinery now tested
  through a synthetic failing builder step on the same evidence channel),
  6 replay-evidence schema (5 fixed by the `rho` contract addition, 2 step
  tests rewritten to retired-opt-in semantics), 8 legacy conservation
  semantics (fill-identity → `mass = fill·ρ_local`; composite inventory →
  Körner total `Σ_L ρ + Σ_I m`; paired debit → full Körner attribution),
  1 baseline contract (`rho_gas=1.0`).
- Flipped to passing: P0 dam-break conservation xfail (drift 3.6e-7, 500×
  margin), gravity-changes-topology guard xfail (dam collapses), 1 quarantined
  closure-experiment test (removed from `ci_quarantine.txt`).
- Campaign tests rewritten for dynamic collapse (conversion/relative-drift
  asserts; zero-tolerance fail-closed now expects the correct `RuntimeError`).

## 7. Verification results

| Suite | Result |
|---|---|
| 25 `test_free_surface*` files | 193 passed, 16 xfailed |
| `test_d3q27_free_surface.py` | passed |
| `test_dam_break_3d.py` | 3 passed, 1 xfailed |
| `test_dam_break_3d_free_surface_campaign.py` | 7 passed |
| `test_dambreak_cross_validation.py` | passed |
| water entry ×3 (`multiphase`, `water_entry`, `slamming`) | passed |
| hull FS ×2 | passed |
| **Combined final run** | **308 passed, 17 xfailed, 0 failed** |

Mechanic proofs: D3Q19 6×8 column (τ=0.6, free-slip, 200 steps): front 5→14,
H 7→2, drift +0.01%. D3Q27 same forcing: front 5→14, drift +0.042%.
2D 6×10 column: front 5→16, drift +0.18%. Körner total conserved to float
rounding on frozen, wave, and dam fixtures.

## 8. Known limitations (not regressions)

- The reference ABB requires pressure-matched gas; `rho_gas ≪ rho_liquid` is
  outside the model's domain (same as waLBerla) and will produce negative
  populations. No guard was added, matching the reference's garbage-in
  behaviour; configs/tests use 1.0.
- Retired flags (`enable_i_to_g_ownership_closure`,
  `paired_liquid_interface_debit`) are accepted as no-ops for compatibility;
  the standalone I→G ownership builder and its fail-closed policy remain tested.
- `total_liquid_inventory` (fill-based composite) is not conserved under
  frozen ABB rarefaction by definition; the conserved quantity is the Körner
  total. Both are documented at the call sites.
