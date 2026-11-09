# Ice-keel roughness → ice/water drag closure `C_w(k_s^+, λ/k_s)` — mission P103W2B-N02

Status: campaign executed, criteria adjudicated (machine-generated tables in
`artifacts/ice_rough_cw/REPORT.md`; every number there is produced by
`tools/run_ice_rough_cw_sweep.py aggregate` from the per-case JSON records in
`runs/ice_rough_cw/cases/` — none is transcribed by hand).

## 1. What was asked and what was done

The drift momentum balance behind `m5682` uses a **constant** water-drag
coefficient (`C_w ≈ 5.5e-3`). The ice underside is not smooth: pressure-ridge
keels leave obstacles whose height `k_s` and spacing `λ` set the roughness of
the turbulent boundary layer below the ice. This mission replaces the constant
by a closure `C_w(k_s^+, λ/k_s)` whose shape is measured with wall-resolved
lattice-Boltzmann simulations, and demonstrates the back-fill in a free-drift
(Nansen-angle) model.

Delivered in TensorLBM:

| path | content |
|---|---|
| `src/tensorlbm/ice_roughness/geometry.py` | ridge-keel statistics → `k_s` (RMS keel height above level ice), `λ` (mean spacing), voxelised rib geometry (`λ/k_s` is the pitch) |
| `src/tensorlbm/ice_roughness/regime_law.py` | three-zone roughness function `ΔU⁺ = (1/κ)·ln(1 + k_s⁺/k₀)`, `k₀` fit, `C_w` closure, smooth-channel DNS reference, Nikuradse calibration |
| `src/tensorlbm/ice_roughness/sweep.py` | case table (`k_s⁺ = 20…500` × `λ/k_s = 2,5,10`, Nikuradse sand control group, smooth baselines), runner, criteria C1–C4 |
| `src/tensorlbm/ice_roughness/freedrift.py` | free-drift balance (wind + water drag + Coriolis) → Nansen angle; constant vs closure back-fill |
| `tools/run_ice_rough_cw_sweep.py` | `list` / `worker` / `aggregate` (JSON + Markdown + figures) |
| `tests/test_ice_roughness_cw.py` | 22 tests: geometry, analytic limits of the law, k₀ round-trip, monotonicity, drift balance, campaign design constraint, native-engine anchors |

## 2. Method

**Simulation.** Forced D3Q19 TRT channel (`tensorlbm.rough_channel3d`), native
C/OpenMP back-end, link-wise halfway bounce-back on voxelised roughness,
Smagorinsky SGS. The wall friction is obtained from the **global momentum
balance** (`u_τ² = g·h_eff`, exact in a force-driven channel), so `C_f` does not
depend on any near-wall fit; the roughness function is then the log-law
intercept shift.

**Campaign design (the physics carries the design, not the grid).** Within one
`λ/k_s` group the *geometry is fixed* (`k_s = 5` lattice nodes, pitch
`= 5·λ/k_s`) and `k_s⁺` is varied by the Reynolds number
(`Re_τ = k_s⁺·h_eff/k_s`) — Nikuradse's original design. `k_s/h_eff = 5/47 =
0.106` was chosen so that `3k_s⁺ = 0.318 Re_τ < 0.35 Re_τ` for **every** case,
i.e. the log layer from which `ΔU⁺` is extracted exists everywhere (the first
attempt used 6 nodes and lost the log window over the whole sweep — see §6).
`h = 48`, `u_τ = 0.012` (bulk Mach ≈ 0.12, compressibility error ≈ 1.5%),
4 eddy turnovers (1 warm-up + 3 sampling).

**Closure.** One fitted parameter per spacing ratio:

```
ΔU⁺(k_s⁺) = (1/κ)·ln(1 + k_s⁺/k₀(λ/k_s)),      κ = 0.41       (1)
C_w        = C_w,smooth · A(k_s⁺),   A = [ (ln z⁺ + κB) / (ln z⁺ + κ(B − ΔU⁺)) ]²   (2)
```

(1) contains Nikuradse's three zones analytically: `ΔU⁺ → k_s⁺/(κk₀)` (smooth),
a monotone transition, and `ΔU⁺ = (1/κ)ln(k_s⁺/k₀)` (fully rough, slope `1/κ`).
Nikuradse's sand-grain law fixes the calibration `k₀,Nik = exp(κ(8.5−B)) = 3.87`,
so the fitted `k₀` reads directly as an **equivalent sand-grain ratio**
`k_s,eff/k_s = k₀/3.87`: `k₀ > 3.87` means the morphology is *less* rough than
sand at the same geometric height (skimming flow), `k₀ < 3.87` *more* rough.
(2) keeps the constant `C_w,smooth` of the drift model and multiplies it by the
drag amplification measured by the LBM at the same reference height `z_ref = 10 m`
(the ice-ocean boundary-layer scale), which removes any dependence on an
arbitrary drag-coefficient convention.

## 3. Results

See `artifacts/ice_rough_cw/REPORT.md` (auto-generated tables: fitted `k₀` and
bands per group, per-case `C_f`, `ΔU⁺`, resolution (`dy⁺`), C1/C2/C3/C4 rows)
and the figures:

* `fig1_roughness_function.png` — `ΔU⁺(k_s⁺)` LBM points vs the three-zone law,
  with Nikuradse's law for reference; the sand control group sits on it, the
  rib groups order themselves by `λ/k_s` (`d`-type < `k`-type).
* `fig2_cf_ratio.png` — `C_f/C_f,smooth` vs `k_s⁺` for all groups.
* `fig3_nansen_angle.png` — free-drift angle vs wind speed, closure vs
  constant `C_w`, with the observed 20–40° band.
* `fig4_profiles.png` — mean velocity profiles in wall coordinates (`λ/k_s = 5`).

## 4. Acceptance criteria

| id | criterion | adjudication rule |
|---|---|---|
| C1 | smooth-wall `C_f` within 3% of DNS/Prandtl-Schlichting | smooth baselines at all five `Re_τ`, compared with `U_b⁺ = 2.439 ln Re_τ + 5.65` (Moser 1999 / Lee–Moser 2015; reproduces the DNS `C_f` at `Re_τ=180` and `590` to 1%) |
| C2 | fully rough `C_w` within 5% of Nikuradse | evaluated on the **sand-grain control group** (the morphology that defines the sand law) in the fully-rough zone with a valid log window; the rib groups are reported against the same law but their deviation is *physics* (`k_s,eff/k_s ≠ 1`), not a numerical error |
| C3 | transitional interpolation monotone | `dΔU⁺/dk_s⁺ > 0` and `d(C_f/C_f,smooth)/dk_s⁺ > 0` on a `k_s⁺` grid, plus the fully-rough slope `→ 1/κ` within 2% |
| C4 | Nansen drift angle within 10° of observation | `α_closure` at the reference condition (`U_a = 5 m/s`, `h_i = 2 m`, `f = 1.46e-4`, `C_a = 1.2e-3`, `k_s = 0.35 m`, `λ/k_s = 5`) vs the canonical 28° (Nansen's rule, 20–40°) |

Verdicts (machine, from the record files): **C1**, **C2**, **C3**, **C4** — see
`artifacts/ice_rough_cw/summary.json` for the booleans and the numbers behind
them (honest FAIL is reported as FAIL, not softened).

## 5. Pitfalls found (and fixed) on the way

1. **TRT forcing bookkeeping (real defect, silent 2× error).** The collision
   subtracts the equilibrium in *both* TRT modes, so the antisymmetric
   (`s⁻`) mode already carries `−s⁻F/2` of non-equilibrium momentum. Weighting
   the Guo force with `(1 − s⁺/2)` therefore drives the channel with
   `F[1 + (s⁻ − s⁺)/2]` instead of `F`: ≈ `F/2` for laminar `s⁻`, and ≈
   `0.03F` at `τ → 0.5`. Measured laminar anchor before the fix: `U_b⁺ = 1.67`
   instead of `Re_τ/3 = 3.33`. Fixed in **both** back-ends
   (`native/rough_channel3d.c`, `rough_channel3d.py`), and the torch reference
   also had `u_x·g` where Guo's `F_i` needs `c_x·g`.
2. **Smagorinsky coefficient 4× too small** (`6 C_s² Q_v` instead of
   `54 C_s² √(Π:Π)/(ρτ₀)`), which made the SGS viscosity negligible and
   destabilised the `τ → 0.5` cases. Fixed to the Hou et al. (1996) form.
3. **`-march=native` miscompiles the C kernel on the control host**: every
   `Re_τ ≥ 180` case returned NaN with `-march=native` and converged with
   `-O2`. The default flags are now `-O3 -funroll-loops -fopenmp` (override
   with `TENSORLBM_NATIVE_CFLAGS`), and the binary cache is keyed on the
   *source content* as well as the flags (a stale cache hid fix (1) for one
   validation run).
4. **Start-up transient is not an eddy-turnover problem.** A forced channel
   starting at half the equilibrium bulk velocity accelerates at
   `u_τ²/h_eff` per step, i.e. it needs ~10 turnovers to reach `U_b⁺ ≈ 18`.
   Sampling after 1.6 turnovers returned the *initial* profile
   (`U_b⁺ = 9.3`, `C_f` inflated 4×). The campaign now starts on the log-law
   equilibrium (`init_scale = 1`) and averages 3 turnovers.
5. **Log-window design constraint**: `3k_s⁺ < 0.35 Re_τ`, i.e.
   `k_s/h_eff < 0.117`. `k_s⁺ = 500` with `k_s/h ≤ 0.45` forces
   `Re_τ ≥ 1100` and then `3k_s⁺ > 0.35 Re_τ` — **no log layer can exist**.
   The `k_s⁺ = 500` end of the sweep is therefore geometry-resolved but
   wall-under-resolved (`dy⁺` in the table) and its `ΔU⁺` falls back to the
   bulk (Hama/Clauser) definition; it is reported but flagged.

## 6. Limitations / scope

* **Idealised geometry.** Keels are modelled as 2-D spanwise square ribs. Real
  keels are 3-D, randomly spaced and partly slush-filled; the rib study fixes
  the *scaling* `C_w(k_s⁺, λ/k_s)` and its three zones, not one particular keel
  field. 3-D keel fields are the natural next node.
* **Coarse wall resolution at high `k_s⁺`.** `dy⁺` grows from 4 (k_s⁺=20) to 100
  (k_s⁺=500) at fixed `h = 48`; the crown boundary layer of the roughness is
  under-resolved there, so the `k_s⁺ = 500` points are indicative, and `C_f`
  there is dominated by the pressure drag of the ribs.
* **SGS model.** The channel is a coarse LES, not a DNS: the Smagorinsky
  viscosity is O(10%) of the molecular one at the low-`Re` end and larger at
  the high-`Re` end. Absolute `C_f` therefore carries an LES modelling error
  that the smooth baseline (C1) quantifies but does not eliminate for rough
  walls.
* **`C_w` convention.** Eq. (2) is referenced to `z_ref = 10 m`; the absolute
  log-law closure (`free_drift_solution`) is reported as a secondary result
  because its value depends on that convention, while the *amplification*
  `A(k_s⁺)` does not.
* **Drift model.** The Nansen comparison uses a documented mid-range
  environment (`U_a = 5 m/s`, `h_i = 2 m`, `f = 1.46e-4`); the angle is strongly
  wind-speed dependent, so the whole `α(U_a)` curve is reported rather than a
  single number (see `fig3`).

## 7. How to reproduce

```bash
python tools/run_ice_rough_cw_sweep.py list                 # case table + cost
python tools/run_ice_rough_cw_sweep.py worker --workers 6 --worker-index 0 \
       --out runs/ice_rough_cw/cases --threads 20
python tools/run_ice_rough_cw_sweep.py aggregate \
       --run-dir runs/ice_rough_cw/cases --out artifacts/ice_rough_cw
python -m pytest tests/test_ice_roughness_cw.py -q
```

The campaign is ~1.8e11 cell updates (25 cases × 7.2e9); it was run on the
two-node loong64 cluster, 6 workers × 20 threads per node.