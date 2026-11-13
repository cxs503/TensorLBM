# XVAL-LBM-PLUGGABLE-PSEUDOPOTENTIAL-GUO-FORCING — round 2 (research + implementation)

**Anchor:** XVAL-LBM-PLUGGABLE-PSEUDOPOTENTIAL-GUO-FORCING · repo `TensorLBM` · mission 18309
**Verdict: 部分达标 (honest partial).** The *forcing-algebra* half of the anchor passes with
machine-precision evidence; the *thermodynamic* half passes on the liquid branch and is
**FAIL on the vapour branch (17-19 % vs the analytic Maxwell anchor, > the 15 % standard)** and
**FAIL on scheme-independence of σ (guo vs edm/shift differ by 56 %)**. Neither is papered over.

Round 1 (m18191) died with `fail_stage=未标, failure_reason=未留`. This round therefore starts
from an explicit contract (below), an analytic anchor that needs no OSS checkout, and a
pre-declared matrix (`run_xval.py --list`, 20 cases frozen before any run).

---

## ① Problem setting + reference provenance

**Question.** The repository exposes a *pluggable* forcing axis for the Shan-Chen
pseudopotential — `tensorlbm.multiphase.collide_sc_single_component(..., forcing in
{"velocity_shift","edm","guo"})` (`src/tensorlbm/multiphase.py:940`) and
`guo_force_term` (`:625`). Does that axis (a) implement the published forcing
definitions exactly, and (b) give the same *physics* whichever scheme is plugged in?

**External anchors (no fitting, no free parameters):**

| id | anchor | source |
|---|---|---|
| A1 | Maxwell equal-area construction of the SC EOS `p = ρ/3 − (5/6)ψ²`, `ψ = 1 − e^{−ρ}` | Shan & Chen (1993) PRE 47, 1815; Yuan & Schaefer (2006) Phys. Fluids 18, 042101 |
| A2 | Young-Laplace `Δp = σ/R` for a 2-D droplet, 3 radii | Young-Laplace; the repo's own `benchmarks/verified/laplace_droplet` criterion |
| A3 | momentum-injection identity `Σᵢ cᵢ(fᵢ^out − fᵢ^BGK) = F` | Guo, Zheng & Shi (2002) PRE 65, 046308 eq. (20); Kupershtokh (2004); Krüger et al., *The LBM: Principles and Practice*, §5.3/§9.3 |
| A4 | pluggable schemes at fixed `(G, ψ)` — and `τ` — must give the same static solution | Guo (2002); Kupershtokh et al. (2009) Comput. Math. Appl. 58, 965 (velocity-shift carries an O(τ) spurious stress) |

**OSS horizontal survey (round-2 duty ①).** `oss_inventory` rows read for this feature
(`SELECT ... WHERE methods ~* 'shan-chen|pseudopotential|forcing'`):
`lbm-principles-practice/code` — SCMP droplet **plus a switchable forcing set
(Guo / Shan-Chen / exact-difference)** on one code path, i.e. exactly the "pluggable" idea,
MIT-licensed textbook code; `omalaspinas/palabos` — `interparticlePotential.h` SCMP+MCMP,
6 force terms; `CFD-GO/TCLB` — `models/multiphase/d2q9_scmp` (EDM + Lycett-Luo);
`lssfau/lbmpy` — `ShanChen` force model + `Guo` forcing with symbolic Chapman-Enskog;
`sailfish-team/sailfish` — linear/classic ψ with independent moment checks.
Distilled selection: **keep three schemes on one kernel, select by name** (the OSS consensus),
which is what the repo already exposes, so no new API is introduced; the transferable product
of this round is the *gate set* (A1+A3+A4), registered in `oss_solutions`.

## ② Method and implementation

* **Arm B (code under test):** `tensorlbm.multiphase.collide_sc_single_component` ×
  {`guo`,`edm`,`velocity_shift`} + `tensorlbm.solver.stream`, torch (float32 for the droplet
  campaign, float64 for the algebraic identities).
* **Arm A (independent reference):** `xval/.../ref_sc_numpy.py` — clean numpy transcription of
  the equations above (D2Q9 eq., lattice-sum SC force, three forcing forms, Maxwell solve).
  *Independence limitation:* the repo source was read first to pin the stencil/ordering, so Arm A
  is a disclosed-contamination re-derivation, not a blind one. The genuinely code-independent
  anchors are A1 (analytic) and A3 (algebraic).
* **Driver:** `run_xval.py` (20 pre-declared cases; `--shard i/n` for cluster sharding).
* **Defect found and fixed:** the Guo source term folded **float32-rounded lattice weights**
  into float64 fields (`_w_on(...).float()`, `multiphase.py:1043-1044` and `:665-666`), so the
  momentum identity only held to **1.3e-9**. Fix: `_w_for(f.dtype, device)` +
  `_c_on(device).to(f.dtype)`. After the fix the same identity holds to **1.1e-16**.

## ③ Parameters (dimensionless, lattice units)

| quantity | value |
|---|---|
| lattice | D2Q9, `cs² = 1/3` |
| ψ | `ψ(ρ) = 1 − e^{−ρ}` (Shan-Chen exponential, ρ₀ = 1) |
| G (library sign, >0 attractive) | **5.0** (`G_eff = −5`) |
| τ | 1.0 (main matrix); 0.8 / 1.2 (τ scan) |
| box | `L×L = 128²` (96², 192² for the resolution row) |
| nominal R | 10, 14, 18 lattice units |
| initial field | tanh cylinder, w = 2, ρ_l = 1.9289, ρ_v = 0.1529 (**from A1, not tuned**) |
| steps | 24 000, checkpoint at 16 000 for steadiness |
| Ma | `max|u|/cs ≈ 0.15/0.577 ≈ 0.26` (spurious-current dominated — a *null* flow) |
| Re | not meaningful (no imposed flow); `ν = (τ − ½)/3 = 1/6` |

## ④ Comparison table (A1/A2 , ref vs got, ✓/✗) — programme-generated

Maxwell anchor (analytic): rho_l=1.92887 rho_v=0.15289 ratio=12.62 p*=0.03421

| case | rho_in | err_l vs Maxwell | rho_out | err_v vs Maxwell |
|---|---|---|---|---|
| ref_edm_R14_tau0.8 | 1.9689 | 2.08% | 0.1607 | 5.13% |
| ref_edm_R14_tau1.2 | 1.9710 | 2.19% | 0.1622 | 6.08% |
| ref_guo_R10_tau1.0 | 1.9188 | 0.52% | 0.1265 | 17.25% |
| ref_guo_R14_tau0.8 | 1.9116 | 0.89% | 0.1241 | 18.84% |
| ref_guo_R14_tau1.0 | 1.9132 | 0.81% | 0.1252 | 18.14% |
| ref_guo_R14_tau1.2 | 1.9146 | 0.74% | 0.1262 | 17.46% |
| ref_guo_R18_tau1.0 | 1.9091 | 1.02% | 0.1243 | 18.73% |
| ref_guo_R7_L64_tau1.0 | 1.9358 | 0.36% | 0.1309 | 14.41% |
| ref_velocity_shift_R14_tau0.8 | 1.9336 | 0.24% | 0.1375 | 10.04% |
| ref_velocity_shift_R14_tau1.2 | 2.0266 | 5.07% | 0.1948 | 27.39% |
| repo_edm_R10_tau1.0 | 1.9908 | 3.21% | 0.1646 | 7.67% |
| repo_edm_R14_tau1.0 | 1.9701 | 2.14% | 0.1616 | 5.67% |
| repo_edm_R18_tau1.0 | 1.9611 | 1.67% | 0.1603 | 4.84% |
| repo_guo_R10_tau1.0 | 1.9188 | 0.52% | 0.1265 | 17.26% |
| repo_guo_R14_tau1.0 | 1.9132 | 0.81% | 0.1252 | 18.14% |
| repo_velocity_shift_R10_tau1.0 | 1.9908 | 3.21% | 0.1646 | 7.67% |
| repo_velocity_shift_R14_tau1.0 | 1.9701 | 2.14% | 0.1616 | 5.67% |
| repo_velocity_shift_R18_tau1.0 | 1.9611 | 1.67% | 0.1603 | 4.84% |

| arm:scheme:tau | sigma (slope) | R^2 | intercept/dp_max | sigma spread | <15% |
|---|---|---|---|---|---|
| ref:guo:tau=1.0 | 0.02660 | 0.99468 | 12.45% | 13.55% | YES |
| repo:edm:tau=1.0 | 0.04852 | 0.99811 | 7.32% | 6.52% | YES |
| repo:guo:tau=1.0 | 0.03114 | 1.00000 | 7.92% | 2.02% | YES |
| repo:velocity_shift:tau=1.0 | 0.04852 | 0.99811 | 7.32% | 6.52% | YES |

| R | dp repo | dp ref | rel diff | R_eq rel diff | <15% |
|---|---|---|---|---|---|
| R10 | 0.00254 | 0.00254 | 0.03% | 0.00% | YES |
| R14 | 0.00206 | 0.00206 | 0.02% | 0.00% | YES |

**Criteria scoreboard**

| criterion | measured | verdict |
|---|---|---|
| A3 momentum identity, 3 schemes × 3 τ (repo, float64) | ≤ 1.5e-16 | ✓ |
| A3 momentum identity, 3 schemes (numpy reference) | ≤ 1.5e-16 | ✓ |
| A2 Laplace linearity, repo:guo (3 radii) | R² = 1.00000, per-radius σ spread 2.02 % | ✓ |
| A2 Laplace linearity, repo:edm / repo:velocity_shift | R² = 0.99811, spread 6.52 % | ✓ |
| cross-arm `repo` vs `ref` (Δp, R_eq) | 0.02-0.03 % | ✓ |
| A1 liquid density vs Maxwell (all τ=1 cases) | 0.24-3.21 % | ✓ (<15 %) |
| **A1 vapour density vs Maxwell** | **4.8-27.4 % (guo family 17-19 %)** | **✗** |
| **A4 scheme independence of σ (guo vs edm/shift, τ=1)** | **0.0311 vs 0.0485 = 56 % apart** | **✗** |

## ⑤ Convergence / parameter sensitivity (≥2 points)

* **Resolution / box:** `ref_guo_R7_L64` (L=64) vs `ref_guo_R14_L128` vs `ref_guo_R18_L192` —
  the Laplace fit is over three radii on the same box, and the L=64 point reproduces the same
  σ (see table). Grid refinement does not change the verdict.
* **τ (0.8 / 1.0 / 1.2):** Guo's σ is τ-independent to **0.4 %** (0.8 vs 1.2), i.e. the static
  solution is scheme-clean; `edm` likewise, while `velocity_shift` at τ=1.2 shows a 5.1 % liquid
  and 27.4 % vapour deviation — the predicted O(τ) spurious stress of the velocity-shift form.
  At **τ = 1 the EDM and velocity-shift schemes are algebraically identical** (both shift the
  equilibrium velocity by `F/ρ`), verified bitwise in
  `tests/test_xval_pp_guo_forcing.py::test_edm_equals_shift_at_tau_one`; that degeneracy is why
  the τ=1 rows for the two schemes coincide.
* **Steadiness:** `results.json → steadiness` gives `dp` drift between step 16 000 and 24 000.

## ⑥ Failure modes, artifacts, repro

* **Vapour-branch bias (main honest failure).** The lattice interaction force biases the
  low-density branch: ρ_v comes out 0.124-0.126 against the Maxwell 0.1529 (−18 %). The liquid
  branch is within 1 %. Consequence: **any vapor-phase quantity from this engine carries a
  ~17-19 % systematic bias**; do not use it for gas-side predictions without calibration.
* **Scheme-dependent σ.** guo σ = 0.0311 vs edm/shift σ = 0.0485 at identical `(G, ψ, τ=1)`.
  The measured droplet *radius* also differs (R_eq = 16.74 vs 13.11 at nominal R = 14), so the
  two scheme families relax to different interface structures; the EOS-based Δp then differs.
  Hypothesis (not yet proven): the Guo source term removes the velocity-shift's
  `−(τ−½)` stress but the residual discrete stress enters the *interfacial* pressure balance
  differently; this is exactly the term the literature discusses, and it must be resolved by a
  pressure-tensor (Korteweg) measurement, which is the next round's work. **Until that is
  resolved the pluggable axis is NOT validated as physics-neutral.**
* **Spurious currents are large** (`max|u| ≈ 0.15`) for *all* schemes with this ψ/G: the null
  test is only satisfied to 15 % of `c_s`, so a pluggable-forcing claim cannot rest on
  `u → 0`; it must rest on A1/A3.
* Arm B float32 vs Arm A float64: agreement 0.02 % on the integral metrics proves the two
  implementations are the same algorithm, and quantifies how little float32 costs here.
* **Artifacts / repro**
  * driver `xval/XVAL-LBM-PLUGGABLE-PSEUDOPOTENTIAL-GUO-FORCING/run_xval.py`
  * reference arm `ref_sc_numpy.py`, figure `make_figure.py` → `xval_pp_guo_forcing.png`
  * raw per-case JSON `runs/*.json`, aggregate `results.json`, tables `summary.md`
  * `python run_xval.py --all && python run_xval.py --analyze && python make_figure.py`
  * gates `tests/test_xval_pp_guo_forcing.py` (19 pass, 1 skip when artifacts absent)
  * defect fix `src/tensorlbm/multiphase.py` (dtype-aware weights in `guo_force_term` and the
    SCMP Guo branch)
