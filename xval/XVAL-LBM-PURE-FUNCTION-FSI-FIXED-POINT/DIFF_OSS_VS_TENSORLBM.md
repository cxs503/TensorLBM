# Diff list: TensorLBM vs haimingz/vivsim — pure-function fixed-point FSI coupling

Anchor `XVAL-LBM-PURE-FUNCTION-FSI-FIXED-POINT`, round 1 XVAL.
Numbers: `benchmarks/xval_vivsim_fsi_fixed_point/artifacts/xval_fsi_fixed_point.json`
(narrative: `REPORT.md` in this directory).

## 1. Approach differences (做法)

| aspect | vivsim (OSS) | TensorLBM | measured consequence |
|---|---|---|---|
| coupling style | fixed-point sub-iteration, **fluid field frozen** inside the time step (`ib_u` sliced before collision, `:120-121`) | interface fixed point on the acceleration with the same frozen linear operator (`fsi_partitioned.solve_partitioned_1d`) | identical when configured identically: `explicit`/no-accelerator/1 iteration == `FSI_ITER=1`, max relative diff 4e-15 (Q1) |
| added-mass treatment | force resultant **plus** `a * CYL_AREA` (`:136`), i.e. the wet mass re-enters the RHS | implicit on the solid side: `variant="theta_split", theta=1` (new this round) | :136 cancels the interface added mass → dry dynamics, identity 1e-15 (Q3); θ=1 == converged strong coupling, 5e-16 |
| sub-iteration budget | fixed count `FSI_ITER` (1 in the VIV example), no relaxation, no convergence test | adaptive: Aitken / IQN-ILS with residual tolerance | identical answer at m\*≥2 (1e-13); at β≥1 the OSS loop **diverges** while Aitken needs 3 iterations (Q2 / §5.2) |
| structure integrator | `dyn.newmark_2dof`, γ=1/2, β=1/4, always re-evaluated from the *step-start* state | Newmark γ=1/2, β=1/4, same algebra | Newton step differs by 0.0 (exact) |
| function purity | pure functions + `scan` carry, structurally JIT/grad-transparent (no `jax.grad` use case in repo) | stateful Python objects, not traceable | out of scope numerically; blocks differentiable FSI |
| collision / forcing / walls (VIV example) | regularized collision, EDM forcing, force-corrected NEBB inlet, equilibrium outlet | not exercised here (linear interface operator) | see `xval/DIFF_OSS_VS_TENSORLBM.md` for the NEBB/body-force comparison |

## 2. Parameter / calibration differences (参数)

* **Frequency calibration.** OSS `K = (2π FN)² M (1 + 1/MR)` (`:58`) makes `FN` the **wet**
  natural frequency — so the physical target is the `(m_s+m_a)` oscillator. Both arms here
  use the same `K`; the reference is the closed-form wet response.
* **Damping.** OSS VIV runs `DR = 0` (undamped); this benchmark uses `ζ = 0.02` total so that
  frequency and amplitude errors are separable. Sensitivity to the drive frequency
  (`w_f/w_n = 0.8 / 1.2`) is in the report §5.3.
* **Time step.** OSS native `dt = 1` in lattice units (`dt/T_n = 1/6400`); the sweep adds
  `dt = 16` and `dt = 64` (`T_n/400`, `T_n/100`). The TL strong coupling stays inside 15 %
  at all three; the measured convergence order is 2.00 (Newmark).
* **Added-mass coefficient.** `C_M = 1` (potential flow, circular section). OSS's `:136`
  uses exactly the displaced mass `CYL_AREA` — the same `C_M = 1`.

## 3. Boundary-condition differences (边界条件) — why they do not enter here

The OSS VIV coupling additionally relies on (i) a local IBM region
(`dynamic_slice`/`dynamic_update_slice`), (ii) MDF with `IB_ITER` inner iterations on the
marker velocity, and (iii) force-corrected NEBB at the inlet. All three act on the *fluid*
block, which this XVAL replaces by one documented linear interface operator; a porting of the
coupling scheme must therefore be re-checked against them (see §4, item 3).

## 4. Porting tradeoffs (移植取舍)

1. **Adopt the loop, not the compensation line.** The fixed-point skeleton (recompute the
   structural state from the step-start state, pure operators) is already TensorLBM's
   `solve_partitioned_1d` and is reproduced to machine precision (Q1). The verbatim
   `h += a * CYL_AREA` line **must not** be copied into TensorLBM: under the added-mass
   interface operator it turns the scheme into the dry oscillator (Q3), which costs
   4.9 %–73 % in wet natural frequency and 91 %–100 % in the resonant response.
2. **Keep the accelerator.** At `β = m_a/m_s ≥ 1` the unrelaxed fixed point does not converge
   however many iterations are spent (measured BLOWUP at `FSI_ITER = 10/40`, m\* ≤ 1), while
   Aitken/IQN-ILS converge in 3 iterations at every mass ratio.
3. **Use θ-split for cost.** `variant="theta_split", theta=1` is exact after **one**
   sub-iteration (5e-16 against the converged solution) — the same one-iteration cost the OSS
   gets from `FSI_ITER=1`, but with the correct fixed point. Recommended default for the
   VIV/added-mass regime; `theta=0` reproduces the old explicit scheme bit-for-bit.
4. **Do not touch `variant="wet_mass"`.** Its double counting is documented
   (`docs/fsi_partitioned_added_mass.md:65-67`) and is now independently measured
   (>91 % response error); θ-split supersedes it for physics-consistent runs.
5. **Not yet portable:** the differentiable-FSI claim. The OSS is structurally `grad`-ready
   but ships no differentiation use case; TensorLBM has no autodiff path through the time
   loop. Out of scope for this round.