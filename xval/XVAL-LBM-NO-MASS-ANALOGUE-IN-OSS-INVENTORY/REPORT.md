# XVAL-LBM-NO-MASS-ANALOGUE-IN-OSS-INVENTORY — verification-debt closure (re-run, protocol audit, verdict)

*Repo: TensorLBM · anchor: `XVAL-LBM-NO-MASS-ANALOGUE-IN-OSS-INVENTORY` · prior closure: mission m21062 (merge #2429) · this pass: re-execution + adversarial review of that closure*

## 0. Why this document exists

Mission m21062 closed this anchor with three committed JSON files
(`oss_survey.json`, `predeclared.json`, `report_payload.json`) and a report claiming:

> D = 8 / 12 / 16 → `C_I` = 2.1588 / 2.1002 / 1.9897, residual 0.0185 / 0.0042 / 0.0043,
> "+75…99 % above the potential-flow anchor … the bias is an **estimator/geometry**
> problem, not a resolution artefact" → recorded as honest FAIL.

It committed **no run script, no raw run JSON**, and its `repro_cmd` pointed at
`run_xval.py`, a file that does not exist in the repository. The claim was therefore
not re-executable. This pass re-executes it from the library code
(`src/tensorlbm/oscillatory_cylinder_inertia.py`, **unmodified**) and audits the three
things the first pass could not show: reproducibility, protocol convergence, and
reference quality.

Reproduction artefacts (all in this directory):

| file | content |
|---|---|
| `run_xval.py` | full arm generator (rep / protocol / beta / kc / box / e3 groups, raw JSON per arm) |
| `local_evidence.py` | the light sub-set re-run locally (cluster queue > 50 at the time of writing) |
| `analyze.py` | recomputes every number below from `runs/*/result.json` (nothing typed by hand) |
| `make_figures.py` | figures 1–4 |
| `runs/*/result.json` | raw per-arm output used by `analyze.py` |
| `summary.json`, `summary.md` | machine-generated tables |
| `reference/potential_grid.json` | grid convergence of the same-mask potential reference |

## 1. OSS inventory claim — re-verified, and it holds (with a coverage caveat)

Independent re-scan of the inventory named by m21062 (`/root/oss-cache`, the only
checkouts present):

| checkout | .py files | `added_mass` / `virtual_mass` / `mass_analogue` / `inertia_coefficient` hits | free-surface mass exchange |
|---|---|---|---|
| `Maarten-vd-Sande_lbm` | 7 | 0 | `lbm/streamcollide.py::interface_mass_exchange` (Thürey/Körner VOF) |
| `pylbm` | 194 | 0 | 0 |
| `hschwane/GraSPH` | 0 (C++ SPH code, not an LBM code) | — | — |

So: the inventory supplies a free-surface **mass-exchange** operator and **no** FSI
added-mass ("mass analogue") reference — the anchor's premise is confirmed. Caveat,
because the conclusion is weaker than it reads: the inventory contains exactly **one**
LBM implementation, so "no analogue in the OSS inventory" says nothing about
OpenLB / Palabos / waLBerla / lettuce, none of which are in the inventory. The
correct statement is *"the current inventory cannot anchor an FSI added-mass
validation; the anchor must be analytic (potential flow)"*, which is what the library
module already does.

## 2. What was measured and how

Library module (not modified): doubly periodic D2Q9 box, fixed staircase cylinder of
reference diameter `D`, uniform harmonic volume force `a(t) = a0 cos(ωt)` on fluid nodes
(Guo source), two force estimators E1 (Ladd momentum exchange) and E2 (pressure +
viscous traction), first-harmonic decomposition over an integer number of periods.
Parameters of the m21062 ladder: `box = 4D`, `β = D²/(νT) = 50`, `KC = 2πU₀/(ωD) = 0.2`,
`u₀ = 0.04`, `transient = 1.5` periods, `measure = 3` periods (library defaults
`CASES["base"]`-family).

Added in this pass:

* **E3** — a third, independent estimator from the *global* fluid momentum balance
  `F_body = a(t)N_fluid − dP/dt` (with the Guo half-force correction
  `P_stored = P + a N/2`), which uses neither the Ladd exchange nor the surface traction.
* **Protocol ladder** — spin-up 1.5 → 3 → 8 periods at fixed physics.
* **Reference grid study** — `potential_added_mass` (same mask, same periodic domain)
  re-run at `nx = 64 / 128 / 256` for the same physical problem (`box = 4D`).
* **Compressibility diagnostics** — near-wall `max|ρ−1|`, fluid mass budget, box-mean
  pressure oscillation vs surface pressure dipole.

## 3. Results

### 3.1 The m21062 ladder does not reproduce (factor 8–11)

| D | nx | τ | `C_I` E1 (re-run) | `C_I` E1 (m21062) | ratio | `C_Q` E1 (re-run) | same-mask potential `c_a` | err vs anchor (re-run) | residual |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 32 | 0.596 | **0.2649** | 2.1588 | 0.123 | 1.4280 | 1.0214 | −74.1 % | 0.348 |
| 12 | 48 | 0.644 | **0.2222** | 2.1002 | 0.106 | 1.4426 | 1.0057 | −77.9 % | 0.338 |
| 16 | 64 | 0.692 | **0.1755** | 1.9897 | 0.088 | 1.4116 | 0.9681 | −81.9 % | 0.337 |

Not only the value but the *sign of the conclusion* changes: the reported claim was
`+75…99 % above` the anchor, the executable code gives `−74…−82 % below`. Figure 2.

### 3.2 The protocol is not converged; the fit residual never meets a 5 % gate

| transient [periods] | measure [periods] | `C_I` E1 | `C_I` E2 | `C_Q` E1 | residual E1 | residual E2 |
|---|---|---|---|---|---|---|
| 1.5 | 3 | +0.1755 | +0.0510 | 1.412 | 0.337 | 0.344 |
| 3 | 3 | −0.0662 | −0.2154 | 1.877 | 0.124 | 0.125 |
| 8 | 5 | −0.9268 | −1.1832 | 2.365 | 0.081 | 0.080 |

`C_I` crosses zero and keeps moving; the residual stays 8–34 % (m21062 reported
0.004–0.019, i.e. 20–80× better than the code can produce in this configuration).
The physical reason is visible in the bulk velocity: `U_amp` is still growing at
13 periods (0.0410 → 0.0476 from 1.5 → 8 periods of spin-up) while the viscous
box spin-up time is `L²/(π²ν) ≈ 6500` steps ≈ 81 periods at β = 50. **Any
single-window decomposition at the m21062 protocol is meaningless.** Figure 3.

### 3.3 Three independent estimators agree with each other and disagree with theory

`β = 50, D = 16, box = 4D`, transient 3 / measure 3 (same run):

| estimator | `C_I` | `C_Q` | residual |
|---|---|---|---|
| E1 Ladd momentum exchange | −0.066 | 1.877 | 0.124 |
| E2 surface traction | −0.215 | 2.138 | 0.125 |
| E3 global momentum balance | +0.010 | 3.279 | 0.017 (momentum fit) |
| potential flow (same mask) | **+0.968** | 0 | — |

The three estimators bracket each other in the in-phase channel, so this is not a
single-estimator defect: the *flow solution itself* carries a near-zero in-phase and a
huge quadrature force on the body. Two facts explain why that is not the potential-flow
added mass:

1. **Surface pressure dipole is 1.77× the potential value** (0.0889 vs 2ρa₀R = 0.0503),
   i.e. the body force channel is larger than potential flow allows.
2. **The box is not solving the incompressible problem**: near the staircase wall the
   density excursion reaches `max|ρ−1| = 0.209` (21 % — two orders above `Ma²/2 ≈ 2·10⁻³`
   at `Ma = 0.069`) and the fluid mass budget drifts `−6.46 %` over the measurement
   window. The box-mean pressure oscillation is only 1.3 % of the dipole, so this is
   **not** a plane acoustic mode (the acoustic eigenfrequencies of the 64² box,
   0.0566 and 0.113, are near the drive ω = 0.0785 but do not carry the signal): it is a
   near-wall compressibility / mass-budget artefact of this configuration.
   Figure 1 shows the field; the vorticity layer is physical, the density field is not.

### 3.4 The potential reference itself is not a 3 % reference at these resolutions

| D (box = 4D) | nx | `c_a` raw (÷ρπR²) | `c_a` area-normalised | mask area / πR² |
|---|---|---|---|---|
| 16 | 64 | 1.0015 | 0.9681 | 1.0345 |
| 32 | 128 | 0.9501 | 0.9410 | 1.0096 |
| 64 | 256 | 0.9286 | 0.9255 | 1.0034 |

The same-mask FV potential solve still drifts 4.4 % between nx = 64 and nx = 256, so at
the resolution m21062 used (nx = 64) the reference carries an `O(h)` error of the same
order as the owner's 15 % budget/10× larger than the platform's 3 % caliber. In
addition, m21062's "R1 = 1.000 (unbounded analytic)" is **not the right anchor** for a
`box = 4D` periodic cell (the same-domain value is 0.93–0.97); using it introduces a
spurious +3…+7 % before any physics is discussed. Figure 4.

## 4. Verdict

* The anchor's **inventory** claim (no FSI added-mass analogue in the OSS inventory)
  **stands** — re-verified independently, with a coverage caveat (one LBM code only).
* m21062's **quantitative claim is not reproducible and its attribution is unsupported**:
  the numbers cannot be produced by the committed code at the stated setting, the
  protocol residual is 34 % (not 0.4 %), `C_I` is protocol-dependent and crosses zero,
  and the reference it compares against is mis-specified.
* Therefore the honest verdict for the FSI added-mass capability is **WITHHELD**, not
  FAIL: at `β = 50, KC = 0.2, box = 4D, τ ≈ 0.69` with volume forcing, the configuration
  is dominated by near-wall compressibility (21 % density excursion) and an unconverged
  viscous spin-up, so no estimator measures a potential-flow added mass — neither a
  passing one nor a failing one.
* Nothing in `src/` was changed: the defects found are **case-design / protocol /
  evidence-binding** defects, not solver defects.

**Remediation carried forward** (proposed ledger entry, see §6):

1. spin-up ≥ 3 box viscous times (≈ 80 periods at β = 50) or a periodic-state
   criterion on the bulk velocity envelope, with a hard gate `resid ≤ 0.05` on the
   first-harmonic fit;
2. drive frequencies kept away from `2π n c_s/L`, and a mass-budget gate
   `|Δmass|/mass ≤ 10⁻⁶` per window;
3. use the same-mask potential reference at `nx ≥ 8D` (or Richardson-extrapolated),
   never the unbounded `C_a = 1`, and quote its own grid error;
4. force estimator E1 to include the Guo half-force correction at boundary nodes.

## 5. Figures

| figure | content |
|---|---|
| `figures/fig1_field_snapshot.png` | velocity + vorticity field at peak drive (D=16, β=50, box=4D) |
| `figures/fig2_reproduction_gap.png` | m21062 claimed `C_I` vs re-run vs same-mask potential reference |
| `figures/fig3_protocol_convergence.png` | `C_I` (changes sign) and fit residual vs spin-up |
| `figures/fig4_potential_ref_grid.png` | grid convergence of the potential reference + staircase-area bias |

## 6. Suggested next ledger entry

`XVAL-LBM-OSCILLATORY-CYLINDER-ADDED-MASS-PROTOCOL` (domain: TensorLBM verification)

> Acceptance: at `D = 16…32`, `box = 4D…6D`, `KC ≤ 0.05`, `β → ∞` (τ ≥ 0.51), with
> spin-up ≥ 3 box viscous times and first-harmonic residual ≤ 0.05 and
> `|Δmass|/mass ≤ 1e-6`: `C_I` from E1/E2/E3 must agree within 5 % **and** lie within
> 15 % of the same-mask potential reference evaluated at `nx ≥ 8D` (extrapolated);
> `C_Q ≤ 0.3`. Owner 15 % / platform 3 % caliber. Honest FAIL allowed and expected if
> the compressibility channel cannot be removed at these Stokes numbers.

## 7. What was not done (honest limits)

* The full ladder (`beta`, `kc`, `box`, `e3` at 40-period spin-up) was submitted to the
  cluster as `cj4228`; the queue depth (>50) exceeded this slot's budget, so the tables
  above use the light local sub-set (`local_evidence.py`, each arm < 20 s on 4 threads).
  The generator for the full ladder is committed and runnable unchanged.
* E3 suffers catastrophic cancellation at these parameters (`a₀N_fluid ≈ 12.2` vs force
  ≈ 0.6): it is quoted as a consistency check only, not as a precision estimator.
* No independent (non-`potential_added_mass`) potential-flow solver was built; the
  reference audit in §3.4 is a self-consistency/grid study, which is exactly what shows
  it cannot support a 3 % claim.