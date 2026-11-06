# D3Q19 BGK analytic verification — R1 (validation-debt burn-down)

**Scope.** `tests/test_lbm_verification.py` + `scripts/verify_lbm_analytic_convergence.py`.
**Status.** Four analytic cases now run in CI and assert. All four removed from
`tests/ci_quarantine.txt`. **No solver source was modified** — this is a
verification-only change; the numbers below are therefore also a statement about
`solver3d`, `boundary/boundaries3d`, `boundary/wall_model` **as merged today**.

## 1. What was wrong before (and why it mattered)

The previous revision of `tests/test_lbm_verification.py` claimed
"100% 解析解验证, 误差<1%" for four cases. It delivered **zero** validation signal:

| Defect | Consequence |
|---|---|
| All four cases defaulted to `device="sdaa:0"` | `torch.device("sdaa:0")` raises on every non-SDAA runner → the tests errored everywhere they were actually run. |
| No `assert` anywhere; each test `print`ed `PASS/FAIL` and `return`ed a bool | pytest discards return values, so even on a working device the module **could not fail**. |
| `f[:, :, :, 0] = f[:, :, :, -2]` and `f[:, :, :, -1] = f[:, :, :, -2]` after streaming | Overwrote two x-planes with a third, destroying x-periodicity instead of relying on it. |
| Body force added as `w_q 3 c_xq G` (simple forcing), Yet called "Guo" | Never exercised the library's actual forcing kernel. |
| Step B drove the top wall by overwriting the top *fluid* plane with an equilibrium | Not a wall BC: mass/momentum were injected without a no-slip closure. |

Because the four node IDs sat in `tests/ci_quarantine.txt`, conftest turned
their errors into **non-strict xfails** — the green CI badge reported nothing
about them. That is the validation-credit gap this change closes.

## 2. Method

Each case has a **closed-form solution in lattice units**, so the residual is a
discretisation error, not a model-to-model comparison. Reference implementation:
`float64`, D3Q19 BGK, `tau = 1.0` (ν = 1/6), Mach ≈ 0.035–0.06.

Common step order (documented in `_wall_phase`): collide → NoDynamics restore on
solid cells → half-way bounce-back → stream → (Couette only) Zou/He moving lid.
Bounce-back precedes streaming, so the wall lies **half a cell** outside the
first fluid node: `y_w1 = 0.5`, `y_w2 = n_y − 1.5`, driven lid plane `y = n_y − 1`.

| Case | Analytic baseline | Quantity compared |
|---|---|---|
| A | `u_y = A₀ cos(kx) e^{−νk²t}` | ν from a least-squares fit of `ln A(t)`; `ν = (τ−½)/3` |
| B | `u_x = U(y−y_w1)/(y_w2−y_w1)` | profile error (÷U) and `C_f = 2ν(du/dy)/U²` |
| C | `u_x = G(y−y_w1)(y_w2−y)`/2ν | peak velocity, profile shape, both vs analytic |
| D | `τ_w = ν ∂u/∂y|_wall = G·H/2` | `τ_w` from a 4-point LSQ parabola through the wall + first 3 nodes |

Channel cases run `3H²/ν + 500` steps, so the measured residual is the
**steady-state** discretisation error rather than an unfinished transient.

Device resolution: `TL_TEST_DEVICE` env override → CUDA if available → CPU.
Caches: `poiseuille_profile` is memoised on its parameters so cases C and D
share one 11 s run. Total module runtime on 1 CPU thread: **31.7 s**.

## 3. Results (programmatically produced, `docs/assets/lbm_analytic_convergence.json`)

### 3.1 Shear-wave viscosity (case A), `τ = 1`, ν_exact = 0.166666667

| nx | kΔx | ν measured | rel. error |
|---|---|---|---|
| 16 | 0.3927 | 0.166659225 | 4.465e-03 % |
| 32 | 0.1963 | 0.166666206 | 2.762e-04 % |
| 64 | 0.0982 | 0.166666638 | 1.722e-05 % |

Error falls as `k²`-or-faster over two grid halvings (4.5e-3 → 1.7e-5 %, ×260
for ×4 in nx): the shear mode is reproduced essentially exactly, i.e. the BGK
relaxation rate and the streaming stencil agree with the continuum dispersion
relation to the leading truncation order.

### 3.2 Poiseuille flow (cases C and D), library `guo_body_force_d3q19`, `τ = 1`

| n_y | H | peak-velocity err | profile-shape err | τ_w err (analytic G·H/2) |
|---|---|---|---|---|
| 8 | 6 | 0.9524 % | 0.6590 % | 1.7588 % |
| 12 | 10 | 0.3367 % | 0.2730 % | 1.0553 % |
| 16 | 14 | 0.1709 % | 0.1480 % | 0.7538 % |

Peak error halves per refinement step in H (0.95 → 0.34 → 0.17 %), i.e.
**second order**, which is what half-way bounce-back + Guo forcing should give.
At `n_y = 12` the numerical profile is `u_peak = 0.019733` vs analytic
`0.019800` — a uniform 0.34 % amplitude deficit, with the *shape* matching to
0.27 %, so the discrepancy is a small effective-viscosity offset, not a wrong
profile. The wall-shear estimate converges more slowly (0.75–1.76 %), because
differentiating the discrete solution at the wall amplifies the near-wall error;
its tolerance is therefore set at 2 % rather than 1 %.

### 3.3 Couette flow (case B), static wall + Zou/He driven lid

| n_y | profile err (÷U) | du/dy numerical | du/dy analytic | C_f err |
|---|---|---|---|---|
| 8 | 7.1e-13 % | 0.020000 | 0.020000 | < 1e-6 % |
| 12 | 2.4e-12 % | 0.00190476 | 0.00190476 | < 1e-6 % |

The steady profile is the analytic straight line **to round-off**. This is the
expected result for a linear exact solution (the discrete Laplacian of a linear
field vanishes), so case B is a sharp test of the no-slip/moving-wall closure
rather than of the discretisation order — and it is exactly the quantity that
the old equilibrium-overwrite "lid" could not test.

## 4. Negative controls (can the thresholds fail?)

Added as tests so the claim is machine-checked, not asserted in prose:

| Control | Perturbation | Measured signal | Test tolerance |
|---|---|---|---|
| A | run at `τ = 1.05`, compare to `ν(τ=1.00)` | **9.93 %** deviation | 0.5 % → detected |
| B | channel with bounce-back disabled | profile err **88.9 %** | 1.0 % → detected |

So a stale analytic baseline or a missing no-slip wall is caught with ≥10×
margin; the tolerances are discriminating, not decorative.

## 5. Thresholds and their provenance

| Test | Tolerance | Observed | Safety factor |
|---|---|---|---|
| A shear-wave ν | 0.5 % | 2.8e-04 % (nx=32) | ~1800× |
| B Couette profile | 1.0 % | 2.4e-12 % (n_y=12) | round-off |
| B Couette C_f | 1.0 % | <1e-6 % | — |
| C Poiseuille peak/shape | 1.0 % | 0.34 % / 0.27 % | ~3× |
| D wall shear τ_w | 2.0 % | 1.06 % | ~2× |

Case C/D tolerances are deliberately only ~2–3× the observed error: they should
tighten as the grid refines, not hide a regression.

## 6. Points to note / limitations

1. **`τ = 1` only.** All numbers are at `τ = 1.0` (ν = 1/6). BGK's
   viscosity-dependence (`τ`-dependent bounce-back slip) is *not* covered here;
   a `τ` sweep is the obvious next increment.
2. **Low Mach.** `u ≤ 0.02` (Ma ≈ 0.06). Compressibility error `O(Ma²)` is below
   the discretisation error at these resolutions, so the cases cannot separate
   the two. Above Ma ≈ 0.2 the analytic baselines would need the
   compressibility correction.
3. **Wall shear from a velocity fit, not momentum exchange.** The LSQ-parabola
   route is a velocity-gradient measurement. An independent Ladd
   momentum-exchange cross-check was *attempted* and did **not** close: in this
   module's BC phase order, `tensorlbm.momentum_exchange.momentum_exchange_standard`
   returned total force `6.93e-3` against the force-balance value `G·N_fluid =
   5.33e-3` (+30 %) at `n_y = 12`. This is **not** claimed as a library defect —
   the Ladd sum form assumes a post-stream half-way bounce-back convention with
   the solid populations in a specific state, whereas this module reflects
   *before* streaming. It is recorded here as an open item that needs its own
   convention-matched study; it is deliberately excluded from the assertions.
4. **Single precision untested.** The reference path is float64 to isolate
   discretisation from round-off. float32 CI would need a per-case tolerance
   (~1e-3 relative) that this study does not provide.
5. **Not a solver change.** Nothing in `src/` was touched; if a future refactor
   breaks these cases, the tests fail — that is the whole point.

## 7. Reproduction

```bash
OMP_NUM_THREADS=2 python -m pytest -q tests/test_lbm_verification.py   # 7 passed, 31.7 s
OMP_NUM_THREADS=2 python scripts/verify_lbm_analytic_convergence.py    # docs/assets JSON + PNG
```

Provenance committed alongside this document (written by the command above,
never hand-edited):

| Path | Content |
|---|---|
| `docs/assets/lbm_analytic_convergence.json` | every number in §3, written by the script |
| `docs/assets/fig_lbm_analytic_convergence.png` | figure: shear-wave $k^2$ convergence, Poiseuille refinement, profile overlay |

The script also mirrors both files plus a stdout log into `artifacts/m3634/` as a
local run record. Those copies are deliberately **not** committed: `artifacts/`
is the repository's run-record directory, so the committed provenance has to
live under `docs/` for the numbers to stay verifiable from a clean checkout.

The `pytest_*.log` scratch files of the original working session are not part of
the change set; the test result is reproducible with the `pytest` line above
(`7 passed`, ≈32 s on one CPU thread).