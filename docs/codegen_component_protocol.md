# Component protocol + template codegen for LBM kernels (XVAL anchor)

Anchor: `XVAL-LBM-COMPONENT-PROTOCOL-TEMPLATE-CODEGEN` (mission 17744, repo TensorLBM).
Status: **anchor criterion met** (`PASS`); one self-imposed side criterion kept as a
recorded FAIL (`T4b_div_magnitude`) and diagnosed below -- never retuned after the run.

Package: `src/tensorlbm/codegen/` (`protocol.py`, `components.py`, `templates.py`, `generate.py`,
`reference.py`). Harness: `xval/component_protocol_codegen/`. Tests:
`tests/test_component_protocol_codegen.py`.

---

## 1. Problem setting and benchmark provenance

**Problem.** A code generator is only trustworthy if the generated kernel can be *cross-validated*
against something independent. The anchor asks for three things at once:

1. a **component protocol** -- a machine-checkable declaration of what an LBM component is and what
   it claims physically (lattice, relaxation rates, transport coefficients);
2. a **template code generator** -- components emit statements, the generator bakes constants and
   unrolls the stencil into a specialised kernel;
3. an **XVAL anchor** -- the generated kernel must reproduce an analytic solution to within the owner
   accuracy criterion (< 15 %).

**Reference solution (external anchor).** 2D Taylor-Green vortex on a periodic domain, single
wavenumber, for which the incompressible Navier-Stokes solution is closed form:

* u = u0 cos(k x) sin(k y) e^(-2 nu k^2 t), v = -u0 sin(k x) cos(k y) e^(-2 nu k^2 t);
* kinetic energy per unit mass E(t) = u0^2/4 * e^(-4 nu k^2 t), so a fit of ln E vs t returns nu.

Sources: G. I. Taylor, *On the decay of vortices in a viscous fluid*, Phil. Mag. 46 (1923) 671-674
(vortex decay e^(-2 nu k^2 t)); S. Succi, *The Lattice Boltzmann Equation* (2001) ch. 5 and
T. Krueger et al., *The Lattice Boltzmann Method: Principles and Practice* (2017) sec. 5.2/9.1
(nu = (tau - 1/2) cs^2, D2Q9 equilibrium); I. Ginzburg, F. Verhaeghe, D. d'Humieres,
*Two-relaxation-time lattice Boltzmann scheme*, Commun. Comput. Phys. 3 (2008) 427-478 (TRT split and
the magic parameter Lambda = 3/16). Units: lattice units throughout, `dx = dt = 1`, so the lattice
wavenumber is `k_lat = 2 pi k / N` -- using the physical `k` in the lattice fit understates the
decay by `(2 pi/N)^2`, a unit trap this work hit and fixed (see section 6).

## 2. Method and implementation

**Protocol (`src/tensorlbm/codegen/protocol.py`).**
`StencilSpec` holds `ex/ey/w/opposite` and validates the D2Q9 moments a second-order equilibrium
relies on (`sum w = 1`, zero first moment, `sum w e_a e_b = cs2 delta_ab`, `sum w e_x^4 = 3 cs2^2`,
`sum w e_x^2 e_y^2 = cs2^2`). `ComponentSpec` declares `(name, family, params)` and is what a
component *claims*; `CollisionComponent` is a `runtime_checkable` Protocol over
`s_plus/s_minus/transport_coefficients/emit_collision`. `validate_component` enforces the
**self-consistency invariant** that makes the cross-validation meaningful:

* `nu_declared == (1/s_plus - 1/2)/3` (the claim must follow from the emitted relaxation);
* TRT: `magic == (1/s_minus - 1/2)(1/s_plus - 1/2)`;
* both rates inside the stable interval (0, 2); emitted statements non-empty, placeholder-free and
  defining every post-collision level `p<i>`.

Lineage: pylbm `pylbm/stencil.py` + `pylbm/generator/generator.py::Generator.add_routine`
(local copy `~/oss-cache/pylbm`, MIT) -- stencil object + routine registry driving code emission;
lettuce `cuda_native/_default_code_gen.py` via `oss_solutions` id=119 -- per-component code
templates instantiated for a stencil. Nothing is copied: the emitted language here is Python/NumPy.

**Code generation (`templates.py`, `generate.py`).** One module template
(`module.d2q9.v1`) with strict rendering: missing field -> `KeyError`, unused substitution ->
`ValueError`, residual `{field}` -> `ValueError`. `generate.build_source` emits

* moments unrolled over the 9 directions with a hand-written sign layout;
* the discrete equilibrium split into its symmetric / antisymmetric parts
  `feq_plus_i = w_i rho (1 + 4.5 (e_i.u)^2 - 1.5 u^2)`, `feq_minus_i = 3 w_i rho (e_i.u)`;
* the collision statements of the component (BGK: `p_i = f_i - s(f_i - feq_i)`; TRT: relax
  `fp_i`/`fm_i` with `s_plus`/`s_minus`, and `fm_j = -fm_i` for an opposite pair);
* pull streaming `f_i(x) <- f_i^post(x - e_i)` via explicit `np.roll` shifts per direction.

All stencil constants, weights and relaxation rates appear as **literals** in the emitted text; the
sha256 of the source is the provenance handle (`kernel.sha256`, recorded per arm).

**Independent reference (`reference.py`).** Broadcasting moments (`tensordot`), equilibrium obtained
by averaging against the opposite direction (`feq_plus = (feq + feq_opp)/2`) instead of the analytic
even/odd forms, per-direction `np.roll` streaming. Different code path = real cross-check.

**Case driver (`xval/component_protocol_codegen/run_arm.py`).** One arm = (family, N, u0), 120 steps,
samples every 2 steps, viscosity from a least-squares fit of `ln E` over `t in [20, 120]`,
`nu_meas = -slope/(4 k_lat^2)`. Thresholds are in `predeclared.json`, written before the first run.

## 3. Parameter table

All arms share tau = 0.56 (nu = 0.02 lattice), rho0 = 1, D2Q9, periodic boundaries, k = 1 mode,
120 steps, fit window t in [20, 120], TRT magic Lambda = 3/16.

| quantity | value | dimensionless group |
|---|---|---|
| lattice sound speed | cs^2 = 1/3 | -- |
| relaxation (BGK) | s = 1/tau = 1.7857 | -- |
| relaxation (TRT) | s_plus = 1.7857, s_minus = 0.2759 | Lambda = 3/16 |
| viscosity | nu = (tau-1/2)cs^2 = 0.02 | Re = u0 L/nu = 6.28 (u0=0.02), 12.57 (u0=0.04) |
| Mach number | -- | Ma = sqrt(3) u0 = 0.0346 / 0.0693 |
| resolution | N = 32, 64 | k_lat = 0.19635 / 0.09817 |
| time step | dt = 1 | sampling interval 2, fit samples 51 |

| arm | family | N | k_lat | u0 | Ma | tau | nu declared | nu measured | err % | field L2 rel | max abs(gen-ref) | kernel sha256 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | BGK | 32 | 0.196350 | 0.02 | 0.0346 | 0.56 | 0.020000 | 0.019997 | 0.014 | 1.108e-02 | 8.3e-17 | `3e08ef91ca1a1c80` |
| A2 | BGK | 64 | 0.098175 | 0.02 | 0.0346 | 0.56 | 0.020000 | 0.019924 | 0.381 | 1.002e-02 | 8.3e-17 | `3e08ef91ca1a1c80` |
| A3 | TRT | 32 | 0.196350 | 0.02 | 0.0346 | 0.56 | 0.020000 | 0.019955 | 0.224 | 1.083e-02 | 1.1e-16 | `f2c405262451813b` |
| A4 | TRT | 64 | 0.098175 | 0.02 | 0.0346 | 0.56 | 0.020000 | 0.019914 | 0.431 | 1.001e-02 | 1.1e-16 | `f2c405262451813b` |
| A5 | BGK | 64 | 0.098175 | 0.04 | 0.0693 | 0.56 | 0.020000 | 0.019694 | 1.530 | 1.984e-02 | 8.3e-17 | `3e08ef91ca1a1c80` |
| A6 | TRT | 64 | 0.098175 | 0.04 | 0.0693 | 0.56 | 0.020000 | 0.019686 | 1.571 | 1.984e-02 | 1.1e-16 | `f2c405262451813b` |

## 4. Comparison table (reference vs generated) and criterion checklist

Reference = analytic Taylor-Green decay (`nu_declared` is the closed-form value the ComponentSpec
declares; `nu_measured` is recovered numerically). `max abs(gen-ref)` is the cross-implementation
control against `reference.py` on a fixed seeded population field.

| criterion | declared threshold | measured | verdict |
|---|---|---|---|
| T1_nu_vs_declared | 15.0 | 1.5709135933320524 | PASS |
| T2_codegen_vs_reference | 1e-12 | 1.1102230246251565e-16 | PASS |
| T3_field_l2_vs_analytic | 0.15 | 0.019839495230593803 | PASS |
| T4a_mass_conservation | 1e-12 | 1.2212453270876722e-14 | PASS |
| T4b_div_magnitude | 1e-10 | 0.00020912570894086446 | **FAIL** |
| T4c_div_scaled_diagnostic | None | None | INFO (post hoc) |
| T5_grid_convergence | 10.0 | None | PASS |
| T6_family_agreement | 15.0 | None | PASS |

Aggregate: max viscosity error **1.571 %** (criterion 15 %), max analytic field error
**1.984%**, max generated-vs-reference deviation **1.1e-16** (machine precision).

**Is the anchor criterion (< 15 %) met? YES** -- worst arm 1.571 %, i.e. also inside the
tighter 3 % band. The recorded FAIL is `T4b_div_magnitude`: the predeclared 1e-10 bound on
`max|div u|` presumed a machine-zero divergence. LBM is weakly compressible, so `div u = O(Ma^2)`
by construction (Krueger et al. 2017 sec. 9.1). Evidence that this is the compressibility term and
not a defect -- `div_max` scales exactly like `Ma^2` when u0 is doubled:

| family | grid | div at u0 | div at 2 u0 | ratio | O(Ma^2) prediction |
|---|---|---|---|---|---|
| BGK | N=64 | 5.187e-05 | 2.087e-04 | 4.023 | 4.0 |
| TRT | N=64 | 5.201e-05 | 2.091e-04 | 4.021 | 4.0 |

and the same central-difference operator applied to the *analytic* field returns ~1e-18, so the
metric is not measuring its own truncation error:

| arm | max abs(div u) | div/(u0 k_lat) | same operator on analytic field |
|---|---|---|---|
| A1 | 7.943e-05 | 2.0228% | 7.8e-18 |
| A2 | 5.187e-05 | 2.6418% | 8.5e-18 |
| A3 | 8.011e-05 | 2.0400% | 7.8e-18 |
| A4 | 5.201e-05 | 2.6487% | 8.5e-18 |
| A5 | 2.087e-04 | 5.3142% | 1.7e-17 |
| A6 | 2.091e-04 | 5.3253% | 1.7e-17 |

Mass conservation, which is the invariant that would actually break if collide+stream were
inconsistent, holds at 1e-14 (criterion `T4a_mass_conservation`, PASS).

## 5. Convergence and parameter sensitivity

**Grid convergence** (same tau, same u0 = 0.02; criterion < 10 %):

| family | case | nu(N=32) | nu(N=64) | delta |
|---|---|---|---|---|
| BGK | u0=0.02 | 0.019997 | 0.019924 | 0.369 % |
| TRT | u0=0.02 | 0.019955 | 0.019914 | 0.208 % |

**Collision-family agreement** at identical (N, u0) -- the protocol-generality check:

| grid | case | nu(BGK) | nu(TRT) | delta |
|---|---|---|---|---|
| N=32 | u0=0.02 | 0.019997 | 0.019955 | 0.211 % |
| N=64 | u0=0.02 | 0.019924 | 0.019914 | 0.050 % |
| N=64 | u0=0.04 | 0.019694 | 0.019686 | 0.042 % |

**Mach-number sensitivity**: doubling u0 from 0.02 to 0.04 moves the recovered viscosity from
0.019924 to 0.019694 (BGK, N=64), i.e. +1.149 pp of relative error, and the analytic field error
from 1.002e-02 to 1.984e-02: the O(Ma^2) compressibility error grows as expected and stays well inside
the 15 % criterion. Two grids x two Mach numbers x two collision families = the sensitivity spread
required by the report contract; the fit residual stays at O(1e-4) in ln E for every arm
(`fit.fit_rms` in the arm JSON).

Figures (generated by `analyze.py`, not hand-drawn):

* `artifacts/xval_component_protocol_codegen/fig_energy_decay.png` -- semilog kinetic-energy decay,
  numerical vs analytic, one panel per collision family;
* `artifacts/xval_component_protocol_codegen/fig_nu_vs_declared.png` -- recovered viscosity per arm
  against the declared value with the 15 % criterion.

## 6. Failure modes, corrections and artifacts

Three defects were caught by the cross-validation itself, and are recorded because they are the
rationale for the checks that now exist:

1. **Equilibrium split coefficient.** The first emitter wrote `1.5*(e.u)^2` in `feq_plus` instead of
   `4.5*(e.u)^2`; the generated-vs-reference difference was 2.7e-3 on an O(1e-2) field and the
   equivalence test failed immediately. Fix: `4.5*`, verified down to 5.6e-17.
2. **TRT antisymmetric sign.** `fm_j` was emitted with the sign of `fm_i` for opposite pairs, so the
   antisymmetric part did not flip; mass drifted (1.0016 vs 0.9993) and TRT diverged from the
   reference by 2.9e-2. Fix: `fm_j = -fm_i`; deviation now 2.2e-16 and mass is conserved to 2.2e-16.
3. **Unit trap in the viscosity estimator.** The first harness used the physical wavenumber k = 1 in
   the lattice decay fit, understating nu by (2 pi/N)^2 and reporting a 96 % "error" while the
   simulation was in fact exact to 0.01 %. Fix: `k_lat = 2 pi k/N`; the reported errors dropped to
   O(0.1 %) and the measured decay slope matched `4 nu k_lat^2` to 4 digits.

Remaining limitation (honest): `T4b_div_magnitude` is recorded as FAILED with its mis-specification
diagnosed rather than retuned; the wall-clock figures are single-node NumPy (~0.1-0.2 s per arm), so
no cluster job was used; only D2Q9 + BGK/TRT are covered -- the protocol admits more families, the
equilibrium split assumes the standard second-order D2Q9 form.

Reproduce (all commands from the repository root, total < 10 s):

```bash
for a in A1 A2 A3 A4 A5 A6; do python xval/component_protocol_codegen/run_arm.py --arm $a; done
python xval/component_protocol_codegen/analyze.py
python xval/component_protocol_codegen/make_doc.py
python -m pytest tests/test_component_protocol_codegen.py -q
```

Artifacts: `artifacts/xval_component_protocol_codegen/` -- `arm_A{1..6}.json` (per-arm metrics,
samples, kernel sha256, equivalence check), `table.csv`, `summary.json` (criteria + verdict),
`fig_energy_decay.png`, `fig_nu_vs_declared.png`.

## 7. Open-source survey and method transfer (round-2 requirement)

| source | what was read | what transferred |
|---|---|---|
| `~/oss-cache/pylbm` (MIT) | `pylbm/stencil.py`, `pylbm/generator/generator.py`, `pylbm/generator/{codegen,ast}.py` | stencil object + routine registry + "emit statements, not parameters" split; `StencilSpec`/`ComponentSpec`/`register_component` mirror it |
| `oss_solutions` id=119 (lettuce, MIT) | `lettuce/base.py`, `cuda_native/_{generator,template,_default_code_gen}.py` pointers recorded there | one module template, per-component code blocks, constants baked per stencil; realised here at NumPy level |
| lbmpy / walberla (survey in the same row) | method only | generative sweep design (generated kernels per stencil) -- no code copied |

Selection recorded in `oss_solutions` with `feature_slug = 'XVAL-LBM-COMPONENT-PROTOCOL-TEMPLATE-CODEGEN'`:
pure-Python/NumPy target (no JIT dependency, runs on the CI x86 and the loong64 cluster alike),
strict-template + protocol-invariant design (cheap to verify), independent reference kernel as the
regression surface. Migration path for a future GPU/JIT backend: keep `CollisionComponent.emit_*`
as the extension point and swap the module template -- the protocol checks and the XVAL harness stay
unchanged, which is the reusability that makes this a knowledge asset rather than a one-off script.
