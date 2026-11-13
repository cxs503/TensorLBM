# XVAL-LBM-QUASI-STATIC-PC-LADDER-WITH-CHECKPOINT (m18236)

**Anchor**: XVAL-LBM-QUASI-STATIC-PC-LADDER-WITH-CHECKPOINT · **Repo**: TensorLBM
**OSS reference**: `je-santos/MPLBM-UT`, tag `quasi-static-pc-ladder-with-checkpoint`
(`src/2-phase_LBM/ShanChen.cpp:526-537,570-583,820-849,875-903`; OSS inventory
`docs/oss-inventory/MPLBM-UT.md` §OSS-SOL)
**Verdict: honest FAIL on the numerical acceptance gate** (protocol移植 verified, physical
anchor not reached). Details and hand-off below.

## 1. 问题 + ref 出处

The OSS practice is a *protocol*, not a kernel: ramp the driving capillary pressure in
`num_pc_steps` equal rungs, settle each rung until the **relative change of the mean
velocity** falls below `convergence` (`%/iteration`, with `it_conv` sampling and `it_max`
cap), and write run number + full lattice state between rungs so a long drainage ladder can
be interrupted and resumed. The step size is tied to the smallest throat,
`dRho = 6 sigma cos(theta)/r_min` with `p = rho/3`, i.e. one rung = the Young-Laplace entry
pressure of that throat `dP = 2 sigma cos(theta)/r_min` (3D).

This repository had **no** ladder/checkpoint driver for a capillary-pressure ramp: the CAC2D
solver can be advanced, and `checkpoint.py` can save a *single* tensor, but nothing
implemented the rung schedule, the settle criterion, a complete-state checkpoint, or
resume-across-rungs. That gap is what this mission fills (`src/tensorlbm/pc_ladder.py`).

## 2. 方法与实现

* `pc_stair_schedule()` — rung pressures; `dim=3, span="oss"` reproduces the upstream step
  exactly (`2 sigma cos(theta)/r_min`), `dim=2` halves it for D2Q9 (one principal curvature),
  `span="full"` subdivides the same total range (needed for a *single* throat).
* `tapered_channel()` — venturi-like channel whose half-width `R(x)` is a cosine bump
  quantised to whole rows (`R_wide=20`, `r_min=8`, throat at x=78, half-length 16); walls are
  horizontal faces only apart from unit staircase steps, because the contact-angle wall model
  of `cac_lbm2d` is derived for a `y`-normal plate.
* `settle()` — the upstream criterion re-expressed as a **rate**
  `rel_rate = |u1-u0| / (max(u0, u_floor) * it_conv) < conv`, so it no longer depends on the
  sampling interval; `it_max`, `it_conv`, `min_iters` mirror the upstream XML fields.
* `save_ladder_state()` / `load_ladder_state()` — **complete** state `(f, g, phi*u_prev)`
  concatenated, written through `tensorlbm.checkpoint` (tensor + JSON) with a **SHA-256** of
  the payload in the metadata; resume verifies the digest and refuses a tampered file.
* Measurement chain: `curvature_field()` computes `kappa = -div(grad phi/|grad phi|)`
  (positive where the liquid side is concave, so `p_liquid - p_gas = sigma kappa`);
  `meniscus_state()` returns centre-row `x_lead/x_trail`, plug length and band-averaged
  `kappa_lead/kappa_trail`; `young_laplace_residual()` compares
  `sigma(kappa_lead - kappa_trail)` with the imposed `drho * g * L` (force balance of the plug,
  exact and independent of any capillary assumption).

Geometry/parameters: 44x128 lattice, `rho_l=1, rho_g=0.1, mu_l=0.2, mu_g=0.02, sigma=0.05`,
interface width `W=3`, `theta=60 deg` (contact wall policy), `drho = 0.9`, plug
`x in [26,56]` (`L = 30`), body force `g = dP/(drho*L)` with `gravity_mode="rho"`.

## 3. 参数表（无量纲）

| quantity | value | note |
|---|---|---|
| grid | 44x128 | D2Q9 CAC + HCH |
| `Re` (creep scale) | < 1 | quasi-static, `u_max ~ 1e-2` |
| `Ca = mu u/sigma` | ~ 4e-2 | from `u ~ 1e-2` |
| density ratio | 10 | `rho_l/rho_g` |
| `W/r_min` | 0.375 | diffuse interface vs throat |
| `r_min/R_wide` | 0.40 | taper strength |
| ladder range | `dP in [5.2e-4, 3.1e-3]` | 6 rungs, `span="full"` |
| OSS step (`dim=2`) | `sigma cos(theta)/r_min = 3.1e-3` | one rung = full entry pressure |
| `convergence` | `1e-4` /iter | OSS default |
| `it_conv`, `it_max` | 50, 200 | reduced for wall-clock |

## 4. 对照表（ref vs got）

### E0 — measurement chain (static droplet, R=20, 64x64)

| quantity | ref | got | error | gate |
|---|---|---|---|---|
| curvature `kappa` | `1/R_eq = 0.049854` | `0.049880` | **0.05 %** ✅ | < 15 % |
| Laplace pressure `dp = sigma kappa` | `2.494e-3` | `1.537e-3` (field `p`) | **38.4 %** ❌ | < 15 % |

The *geometric* curvature estimator is excellent (0.05 %), but the **hydrodynamic pressure
field of the HCH scheme is not a usable capillary-pressure probe** at this resolution
(`sigma_eff = dp/kappa = 0.0308` vs input `0.05`, i.e. 38 % low). Consequence for the ladder:
the acceptance number must come from the curvature route only.

### E1 — quasi-static ladder (6 rungs)

| rung | dP | g | steps | converged | x_lead | kappa_lead | kappa_trail | YL rel.err |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 200 | False | 54.0 | 0.0145 | 0.016 | 7.32e+25 |
| 1 | 0.000521 | 1.93e-05 | 200 | False | nan | nan | nan | nan |
| 2 | 0.00104 | 3.86e-05 | 200 | False | nan | nan | nan | nan |
| 3 | 0.00156 | 5.79e-05 | 200 | False | nan | nan | nan | nan |
| 4 | 0.00208 | 7.72e-05 | 200 | False | nan | nan | nan | nan |
| 5 | 0.0026 | 9.65e-05 | 200 | False | nan | nan | nan | nan |
| 6 | 0.00313 | 0.000116 | 200 | False | nan | nan | nan | nan |

Rungs 1-6 return `x_lead = nan`: the plug is no longer detected on the centre row after the
first forced rung, so the Young-Laplace residual is undefined for every rung of interest
(the rung-0 entry is also degenerate because `g = 0` divides by zero). **Gate not met.**

### E4 — sensitivities (rung count / convergence tolerance)

| variant | rungs | YL rel.err per rung |
|---|---|---|
| rungs4 | 5 | 7.32e+25, nan, nan, nan, nan |
| conv1e-2 | 7 | 2.58e+25, 0.479, 0.512, nan, nan, nan, nan |

With `conv = 1e-2` the ladder does settle (rungs 1-2) and the residual is **48 % / 51 %**,
i.e. 3x above the 15 % gate; afterwards the same plug-loss failure appears. Loosening the
convergence tolerance does not repair the measurement -> the failure is geometric/model, not
protocol.

### E2 — checkpoint/resume

The experiment script's resume comparison aborted with a **script bug**
(`'RungRecord' object is not subscriptable`), so no end-to-end resume number is reported
here. The protocol itself is covered by `tests/test_pc_ladder.py`
(`test_ladder_checkpoint_roundtrip_and_resume`): 3-rung ladder, SHA-256 present, resume
metadata correct, tampered payload rejected. **4/4 tests pass.**

## 5. 收敛性 / 参数敏感性

* rung count 4 vs 6: same failure point (plug lost after the first forced rung) -> rung
  count is not the cause;
* convergence tolerance `1e-4` vs `1e-2`: with the loose tolerance two rungs do settle, and
  both are 48-51 % off -> settling quality is not the dominant error either;
* spurious-current level is the discriminating quantity: 1.3e-3 (neutral walls) vs 1.3e-2
  (contact walls) — one decade above, which is where the plug integrity is lost.

## 6. 失败模式与工件

**Failure mode** (`E1-physics`): with `wall_phi_policy="contact"` on the quantised taper the
spurious currents rise to `u_max ~ 1.3e-2` (neutral walls: `1.3e-3`), the liquid plug stops
being a single centre-row interval, and `meniscus_state()` returns `nan`. The 1-cell vertical
staircase faces of the quantised taper are the prime suspect: `cac_lbm2d.refresh_solid_ghosts`
derives the wall flux for a `y`-normal plate, and an earlier case in this repository already
documented that applying that derivation to walls of another orientation "leaked into the
streamwise direction" (`cac_lbm2d.py` docstring). A second, independent limitation is that the
HCH pressure field is 38 % off the Young-Laplace value for a static droplet here, so any
pressure-based acceptance number is unusable.

**Artifacts** (all under `TensorLBM/xval/XVAL-LBM-QUASI-STATIC-PC-LADDER-WITH-CHECKPOINT/`):
`run_xval.py`, `artifacts/results.json`, `artifacts/fig_ladder.png`, `artifacts/fig_e0.png`,
`artifacts/ckpt/rung_XXX/` (complete-state checkpoints with SHA-256).
Module: `src/tensorlbm/pc_ladder.py`; tests: `tests/test_pc_ladder.py` (4 passed).

**Hand-off (round 2)**
1. make the wetting wall model orientation-aware (or build the taper from `wall_order`
   ghost values smoothed over the staircase), then re-run E1;
2. replace the implied `L` by the *measured* plug length `x_lead - x_trail` per rung (already
   returned by `meniscus_state`) to remove the constant-length assumption;
3. use the curvature route as the single acceptance number (pressure route retired: 38 %);
4. add the `RungRecord.as_dict()` conversion in `e2_resume` (known script bug) and report the
   end-to-end resume deviation.

**Diff list: this repo vs OSS (做法/参数/边界条件 + 移植取舍)**

| aspect | MPLBM-UT (OSS) | this repo | choice |
|---|---|---|---|
| dimension / lattice | 3D D3Q19, Palabos 2-component Shan-Chen | 2D D2Q9 CAC phase field | re-derived (no kernel reuse) |
| driving | outlet density `rho = 1 - dP/3` (pressure BC) | body force `rho g` on a periodic loop, `g = dP/(drho L)` | momentum-equivalent, stable BW |
| rung step | `dRho = 6 sigma cos(theta)/r_min` = one entry pressure | same total range, `span="full"` subdivides it | single throat needs sub-step resolution (D3) |
| convergence | mean-velocity change `< conv` `%/iter`, `it_conv=1000`, `it_max=5e5` | same, re-expressed as a rate per iteration | `it_conv/it_max` reduced for wall clock |
| checkpoint | `run_num.dat` + `saveBinaryBlock` per lattice | one concatenated `(f,g,phi*u_prev)` tensor + JSON + SHA-256 | adds integrity + tamper detection |
| wetting | `BounceBack(G_ads)` (Huang 2007) | Young wall ghost (`contact_angle_deg`) | **the transplanted model is the weak link here** |
| interface | sharp-ish Shan-Chen, `W` implicit | diffuse `W=3`, `W/r_min=0.375` | curvature estimator validated to 0.05 % |
| acceptance | plotting only (no asserts) | `young_laplace_residual()` + `tests/test_pc_ladder.py` | turned into a gate |
