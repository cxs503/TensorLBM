# 3D MCMP (D3Q19 Shan-Chen) — stability findings, round 1 (GAP-PP-1-3DMCMP-STABILITY)

All numbers below are measured by the scripts in this repository on this worktree;
the commands are listed next to them. Nothing is hand-typed into a result file —
`cases/mcmp_3d_stability/make_report.py` builds the benchmark table and the
figures directly from `artifacts/mcmp_3d_stability/**`.

## 1. Defect found: the D3Q19 "Guo" forcing branch over-injected the force by (2 − 1/(2τ))

`src/tensorlbm/multiphase3d.py::_bgk_collision_guo_3d` built its equilibrium on the
*velocity-shift* velocity `u + τF/ρ` **and** added the explicit Guo (2002) source term
`(1 − 1/(2τ))·w·[…]·F`. The two are alternative discrete approximations of the same
forcing (the 2D path `multiphase.guo_force_term` says so explicitly: "not with the
velocity-shift form `u + τ F/ρ`"), so the 3D branch injected
`τ F/ρ · (1/τ) + (1 − 1/(2τ))F = (2 − 1/(2τ))·F` per step — 1.5× the applied force at τ=1.

Measured (uniform ρ=1 box, constant body force g=1e-4, 200 steps, u should equal g·t):

| path | u/(g t) before fix | u/(g t) after fix |
|---|---|---|
| velocity-shift (`use_guo=False`) | 0.99999 | 0.99999 |
| Guo (`use_guo=True`) | **1.50002** | **1.00001** |

Repro: `python -c` snippet in `tests/test_mcmp3d_stability.py::test_guo_forcing_injects_exactly_the_applied_force`
(`PYTHONPATH=src pytest tests/test_mcmp3d_stability.py`).

Consequence for stability: an effective force 1.5× too large meant the 3D two-component
window closed at a much weaker coupling than the 2D one. With the fix the 3D window moves
towards the 2D-calibrated range (see `benchmarks/verified/mcmp_3d_stability/REPORT.md`).

## 2. Additions (all backward compatible, `G_self=0` / default ψ is bit-identical)

* `sc_two_component_force_3d` / `collide_sc_two_component_3d` accept
  `G_self` (scalar or `(G_11, G_22)`) — the SC self-interaction term
  `-G_σσ ψ_σ Σ_i w_i c_i ψ_σ(x−c_i)`, sign convention identical to the library's
  single-component kernel (**G > 0 = attractive**). Verified in
  `tests/test_mcmp3d_stability.py::test_self_coupling_sign_convention`.
* both kernels accept `psi1_fn` / `psi2_fn` (default `psi_linear`), so EOS-derived
  pseudopotentials (`psi_exp`, `make_psi_carnahan_starling`) can be used in the
  two-component 3D model. With ψ=ρ the force algebra is unchanged (bit-identical,
  asserted in the same test file).

## 3. What is NOT achieved in round 1: Δρ ≥ 100 (honest FAIL)

The linear pseudopotential ψ=ρ has the mechanical EOS `p = cs²ρ + (G/2)ρ²`; the Maxwell
construction for the coexistence ratio gives only ρ_l/ρ_g ≈ 2.6 at |G|=3 (and the
measured 3D numbers are of the same order, see the scan tables). Therefore **Δρ ≥ 100 is
structurally unreachable with the linear-ψ MCMP kernel**, regardless of the coupling —
the genome of the model, not a tuning problem.

Non-linear ψ attempts (measured on 48×12×12, `use_guo=True`, two-component, self-coupling
ramped in over 2500 steps):

| attempt | parameters | outcome |
|---|---|---|
| ψ_exp component 1, G_11 = 4.5, G_12 = −1.0 | init ρ1 1.2/0.05, iw=3 | survives 1200 steps, Δρ_component1 = 7.6 (u_max 0.37 startup); ratio far below 100 |
| ψ_exp, G_11 = 5.5 / 6.5 / 7.5 (no ramp) | init ρ1 1.2/0.05 | diverges at step 50 |
| ψ_exp, G_11 ramped 0.2→5.0 over 2500 steps | init ρ1 1.0/0.10, iw=6 | diverges at step 2500 (i.e. at the target coupling) |
| Carnahan–Starling ψ (a=5, a=8), G_11 = 1.0 | init ρ1 0.36/0.03, iw=4 | diverges at step 50 |

**fail_stage**: *parameter-regime / constitutive-model search* — the coupling grid and the
scanner are in place, but the high-density-ratio constitutive route is not stabilised.

**What is missing (input for round 2)**:
1. a density-ratio-calibrated pseudopotential for the *two-component* SC (the CS factory
   in this repo was calibrated for the single-component kernel and yields ψ values that
   are too stiff at the interface when combined with a cross-coupling term);
2. a gradual coupling ramp whose rate is tied to the interface relaxation time (a fixed
   linear ramp over 2500 steps is not enough: the moisture wave catches up with G);
3. possibly a pressure-evolution / well-balanced (EDM) cross-force split in 3D — the 2D
   kernel has `sc_form="edm"` (Kupershtokh 2009) but the 3D force has no equivalent yet.

## 4. Trap list (for TRAPS)

* A non-periodic tanh slab in a periodic box creates a `ρ` jump across the seam; that
  alone produces `u_spurious = cs²|∇ρ|/ρ` and was the first bogus "instability" observed
  here. Always build the profile as two smooth interfaces (`slab_profile`).
* `u_max` of a *single* component is not a stability diagnostic at high density contrast
  (it contains `F/ρ` of the dilute component). Use the mixture velocity.
* Per-component and mixture density ratios differ by orders of magnitude in MCMP; a
  record that only stores "density ratio" is ambiguous. Both are stored in the records.