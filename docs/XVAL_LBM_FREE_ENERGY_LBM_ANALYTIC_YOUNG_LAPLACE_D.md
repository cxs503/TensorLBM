# XVAL-LBM-FREE-ENERGY-LBM-ANALYTIC-YOUNG-LAPLACE-D — analytic Young-Laplace calibration of a free-energy-LBM droplet

**Anchor:** `XVAL-LBM-FREE-ENERGY-LBM-ANALYTIC-YOUNG-LAPLACE-D` ｜ **Repo:** TensorLBM ｜
**Owner criterion:** benchmark error < 15 % (stretch 3 %)

**Verdict: `WITHHELD` (honest, nothing is overridden).** The *protocol* and the *measurement
chain* are delivered and verified against the analytic free energy to < 3 % on the chain
itself; the *graded LBM campaign* did not complete inside this ticket (the compute gate job
`cj2932` stayed queued). No number below stands in for the missing campaign, and no
acceptance band was widened.

---

## ① Problem setting + reference provenance

The companion anchor `XVAL-LBM-FREE-ENERGY-BINARY-EQUILIBRIUM-PRESSURE` grades a **flat**
interface (`P_nn = p_coex`). The Laplace law needs a **curved** interface, and for a 2-D
droplet the coefficient is **one** curvature:

```
delta p = sigma (1/R1 + 1/R2) = sigma / R            (cylinder, NOT 2 sigma / R)
```

with `f(phi) = A(-phi^2/2 + phi^4/4) + (kappa/2)|grad phi|^2`:

| quantity | closed form | verified how (this delivery) |
|---|---|---|
| interface width | `xi = sqrt(2 kappa/A)` | tanh fit of the measured profile, C4 |
| surface tension | `sigma = (2/3) sqrt(2 kappa A) = (4/3) kappa/xi` | **numerical** integral of `kappa phi'^2` over the analytic profile, C1 (agreement 1.8e-13) |
| coexistence pressure | `p_coex = A/4` (equal wells) | bulk means `p_in`, `p_out` |
| moment balance | `dP_rr/dr + (P_rr-P_tt)/r = 0` | `dp = ∫(P_rr-P_tt)/r dr`, `sigma = ∫(P_rr-P_tt) dr` |
| surface of tension | `R_sigma = ∫ r(P_rr-P_tt)dr / ∫(P_rr-P_tt)dr` | first moment of the measured kernel |

Sources: Swift, Orlandini, Osborn & Yeomans, PRL **78**, 19 (1996); Briant & Yeomans, PRE
**69**, 031602 (2004); Rowlinson & Widom, JCP **76**, 1878 (1982) for the pressure-tensor form
and the surface-of-tension radius.

**The calibration subtlety this anchor is about.** A diffuse interface has no unique radius,
so `delta p * R` can be made to agree or disagree with `sigma` by choosing `R`. Three
definitions are therefore produced and reported side by side: `R_contour` (phi = 0 level
set), `R_fit` (`atanh` tanh fit — exact for the model profile) and `R_sigma` (surface of
tension). The headline KPI uses `R_fit`; the spread between the definitions is itself graded
(C5).

## ② Method & implementation

| component | file | role |
|---|---|---|
| new capability | `src/tensorlbm/phasefield/laplace_droplet_calibration.py` | droplet field, radial reduction, three radius definitions, polar pressure projection, exact moment-balance integrals, bulk-band extraction, case runner, grading |
| transport kernel (reused, unchanged) | `src/tensorlbm/phasefield/free_energy_pressure.py` | `collide_and_stream`, `macroscopic`, `pressure_tensor_measured` — the runner does not re-implement physics |
| criteria, declared before runs | `xval/XVAL-LBM-FREE-ENERGY-LBM-ANALYTIC-YOUNG-LAPLACE-D/predeclared.json` | C1–C7 bands, parameter window, limitations |
| campaign entry | `xval/XVAL-LBM-FREE-ENERGY-LBM-ANALYTIC-YOUNG-LAPLACE-D/run_xval.py` | runs the case set, writes `summary.json` + 3 figures, exit 0/1 |
| tests | `tests/test_laplace_droplet_calibration.py` | 22 numeric gates (`pytest -q`: **22 passed**) |

Protocol: periodic box, static droplet `phi = -tanh((r-R)/xi)`, populations started at local
equilibrium and advanced by the production kernel; `p_in`/`p_out` from bulk bands at
`|r-R| >= 3 xi`; `P_rr - P_tt` azimuthally averaged on 0.25-lu radial bins and integrated
across **one** interface. A guard rejects integration windows with more than one kernel lobe,
because a window spanning two interfaces returns `2 sigma` — a defect that shipped once in
the SOY xval and is now a regression guard.

## ③ Results — what was actually measured (programmatic, nothing hand-copied)

**C1 analytic anchor** — σ from the closed form vs σ from the numerical gradient-energy
integral `kappa ∫ phi'^2 dx`: ratio `1.000000000000` for (A,κ) = (0.1,0.2), (0.05,0.1),
(0.2,0.45), (1,1); worst relative deviation **1.8e-13**.

**Analysis chain on the *analytic* pressure tensor** (no LBM: isolates post-processing from
simulation; `_scratch/analytic_chain.json`, figure in the platform report):

| R [lu] | ξ [lu] | ξ/h | ∫(P_rr−P_tt)dr / σ | R_σ − R [lu] | dp_tensor / (σ/R) |
|---|---|---|---|---|---|
| 20 | 2.0 | 4.0 | 0.9353 (−6.5 %) | +0.003 | 0.9388 |
| 26 | 3.0 | 6.0 | 0.9732 (−2.7 %) | +0.006 | 0.9775 |
| 36 | 3.0 | 6.0 | 0.9710 (−2.9 %) | +0.003 | 0.9733 |

Reading: the **radius machinery is exact** (`R_σ` to 0.006 lu) and the residual bias is a pure
**stencil-resolution** effect of the nine-point isotropic operators — 6.5 % at 4 cells/ξ,
2.7 % at 6 cells/ξ, flat in R. Any LBM error beyond ~3 % at ξ = 3 is therefore attributable to
the simulation, not to the protocol.

**LBM side (exploratory, NOT graded — campaign `cj2932` never ran).** Probe at N = 64, R = 10,
ξ = 2 (A = 0.1, κ = 0.2, Γ = 0.05, τ_φ = 1): `dp·R/σ` drifted from **+10.5 % at 400 steps to
−7.6 % at 800 steps** — the state was still relaxing (acoustic transient of the
matched-density initial field); the spurious current decayed 3.7e-3 → 2.9e-3. A non-converged
state carries no verdict, which is exactly why the campaign needs ≥ 800 steps per case and was
submitted as a job.

**Stability window (measured — a real limitation of this kernel).** The explicit
Cahn-Hilliard branch diverges at `dt = 1` for `κ ≳ 0.29` (ξ = 3 at A = 0.1): `|phi|` grew
1.0 → 791 within 40 steps while `|u|` grew 8.8e-3 → 12. Lowering the mobility (Γ 0.05 → 0.01,
τ_φ = 1) does not move the threshold, so it is not a CH CFL alone. The graded window is
therefore `κ ≤ 0.2` (ξ = 2 at A = 0.1); widening it is a separate defect fix, not a criterion
change.

## ④ Figures

* Platform report figure: azimuthally averaged profile `φ̄(r)` of the synthetic droplet and the
  analytic anisotropy kernel whose radial integral equals σ (single interface).
* `xval/.../artifacts/fig_laplace_convergence.png`, `fig_pressure_jump_history.png`,
  `fig_profile_and_pressure_anisotropy.png` — produced by `run_xval.py`, i.e. only once the
  campaign actually runs; absent here because the job is still queued.

## ⑤ Note / limitations (declared up front in `predeclared.json`)

1. 2-D cylinder only; the 3-D `2 sigma/R` coefficient is **not** tested.
2. The finite `xi/R` correction `O((xi/R)^2)` is expected and reported as an `R/xi` trend,
   never subtracted.
3. Graded window `κ ≤ 0.2` (see ③); outside it the kernel is unstable at `dt = 1`.
4. The droplet cannot dissolve (`φ` is conserved by construction); the measured drift is
   graded (C7) rather than assumed.
5. `R_σ` uses the non-negative part of the measured kernel: the continuum kernel
   `κ|∇φ|² ≥ 0`, and letting round-off-negative lobes into the first moment pushed `R_σ`
   outside the integration window (found while testing, fixed).

## ⑥ Error judgement (explicit)

| quantity | vs reference | < 3 % ? |
|---|---|---|
| σ closed form `(2/3)√(2κA)` vs `(4/3)κ/ξ` | algebraic identity | **yes** (1e-13) |
| σ closed form vs numerical `κ∫φ'²dx` | same functional, numeric | **yes** (1.8e-13) |
| `∫(P_rr−P_tt)dr`, ξ = 3 (analytic field) | analytic σ | yes, −2.7 % / −2.9 % |
| `∫(P_rr−P_tt)dr`, ξ = 2 (analytic field) | analytic σ | no, −6.5 % (stencil resolution) |
| `R_σ` recovery | imposed R | **yes** (≤ 0.006 lu) |
| `dp·R_fit` vs analytic σ **on a simulated droplet** | analytic σ | **NOT MEASURED** — `cj2932` queued, no result; the exploratory state was non-converged (+10.5 % → −7.6 %) |

Honest answer to "is the key quantity < 3 %?": **unknown for the LBM droplet.** The chain is
verified to < 3 %; the graded simulation did not run. Nothing here should be read as a PASS on
the Laplace law itself.

## ⑦ Next step (for the follow-up ticket)

Re-submit `xval/XVAL-LBM-FREE-ENERGY-LBM-ANALYTIC-YOUNG-LAPLACE-D/run_xval.py` when the queue
drains (it takes ~4 min at N = 64/80), then either (a) the PASS lands with `summary.json` and
the three figures, or (b) the measured error isolates the kernel defect — the natural
follow-up being the `κ ≥ 0.29` instability fix, which would also permit `ξ = 3` (2.7 % chain
bias instead of 6.5 %).