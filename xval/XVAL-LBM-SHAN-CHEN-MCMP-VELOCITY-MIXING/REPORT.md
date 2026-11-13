# XVAL-LBM-SHAN-CHEN-MCMP-VELOCITY-MIXING — cross-validation against `sorush-khajepor/listLBM`

Round 1 · open-source cross-validation · anchor `XVAL-LBM-SHAN-CHEN-MCMP-VELOCITY-MIXING`
Repository TensorLBM · branch `agent/m18328` · design frozen in `xval/XVAL-LBM-SHAN-CHEN-MCMP-VELOCITY-MIXING/predeclared.json`

**Decision (predeclared, answered below): do not port the OSS mixing rule — neither of its two
readings survives the comparison (one is 136 % off the analytic two-layer Poiseuille solution, the
other diverges). Keep `u_eq="mixture"` (barycentric, τ-shift) as the documented MCMP default for
driven flows, and adopt exactly one piece of the OSS path: its depleted-cell guard.**

---

## 1. Problem, and where the reference comes from

Shan–Chen multicomponent (MCMP) LBM has one free choice that every implementation makes
differently: **which velocity goes into each component's equilibrium**. Each component carries its
own momentum `j_σ = ρ_σ u_σ` and its own interaction force `F_σ`; the equilibrium therefore needs a
"mixed" velocity, and there is no unique prescription for it. TensorLBM ships three conventions
(`u_eq = "self" | "mixture" | "viscous"`, `tensorlbm/multiphase.py:collide_sc_two_component`);
`sorush-khajepor/listLBM` ships a fourth, inventoried as `shan-chen-mcmp-velocity-mixing`
(oss_solutions id **44**, in-repo dump `xval/horizontal_survey_empty/data/oss_solutions_dump.csv`
line 40):

> 多组分 SC：初始化时预计算 othercomp/othercomppotid 索引表，每步对 ncomps-1 个其它组分求相互作用力并叠加体积力
> acc*rho；碰撞用「速度混合、无源项」方案 u=(sum_i sum_q c f + F)/(rho_1+rho_2)+F/rho_i 后各组分用
> f_eq(rho_i,u)，rho 和过小则置 u=0。
> — `collision/pseudopotential/mcmp_mod.f90:114-155, 161-180, 182-236, 239-246, 248-278`,
> `examples/2d/drop_rise_mcmp_8boxes/main.f90:96-125`; per-pair G table `mcmp_mod.f90:263`;
> `extforce` is an **acceleration** (multiplied by ρ); the entry itself flags that the code
> normalises the force term by the total density but the force by the component density
> (`mcmp_mod.f90:142`) and that the canonical Guo path (`:161-180`) is *not* bound.

Zero-hand-copy discipline: every number below is rendered from `artifacts/result.json` (written by
`run_xval.py`) through `analyze.py`; the OSS algebra is re-derived from the inventoried text and its
`file:line` citations, not copied. **Access limitation (ecosystem friction):** this container has no
outbound network (`git clone https://github.com/sorush-khajepor/listLBM` produced no working tree) and
`psql` on the platform DB returns *permission denied* for `oss_solutions`, so the upstream Fortran
could not be read line-by-line — the comparison is against the inventoried description plus its
citations, and the ambiguity of that description is treated as a *result*, not resolved silently.

### The two readings of the OSS formula

With `j_s = ρ_s u_s`, `ρ_tot = ρ_1+ρ_2`:

| reading | equilibrium velocity | note |
|---|---|---|
| **LBM4** ("the accumulated force is the *total* force") | `u_σ = [Σ_s j_s + (F_1+F_2)]/ρ_tot + F_σ/ρ_σ` | the force is inside the mixture momentum |
| **LBM3** ("the force is the *colliding component's* force", the literal reading) | `u_σ = [Σ_s j_s + F_σ]/ρ_tot + F_σ/ρ_σ` | different mixture velocity per component |

Both are implemented as separate arms; both are also *unstable-looking* on inspection, which is
exactly what the runs then measured.

## 2. Method, implementation, tool chain

Two factors are crossed, because the OSS description differs from ours in **two** independent
respects and conflating them would make the verdict unattributable:

* the **mixing rule** (who averages whose momentum, and what it is normalised by);
* the **shift coefficient** (ours `τ F/ρ`, the OSS "no source term" `1·F/ρ`; for the velocity-shift
  family the momentum delivered per step is `s·F/τ`, so `s=1` coincides with ours only at `τ=1`).

Seven arms (`xval/.../velocity_mixing_arms.py`), each a pure function of the populations:

| arm | mixing | shift | guard |
|---|---|---|---|
| `GOT-MIX` | barycentric `Σj/ρ_tot` (shipped `u_eq="mixture"`) | τ | – |
| `GOT-SELF` | per-component (shipped `u_eq="self"`) | τ | – |
| `LBM4` | OSS reading A | 1 | OSS |
| `LBM3` | OSS reading B | 1 | OSS |
| `LBM4-TAU` | OSS reading A | τ | OSS |
| `CTRL-NORM` | barycentric but force shift normalised by ρ_tot (deliberate defect) | τ | – |
| `LIB-ENTRY` | the shipped `cases.two_layer_poiseuille.run_two_layer_poiseuille` | – | – |

**Controlled variable:** every arm calls the *same* `tensorlbm.multiphase.sc_two_component_force`
with the same arguments, and the same init/stream/step loop — only the mixing step differs. That is
what makes the arm differences attributable to the anchor's subject.

Workloads:
* **W1 static droplet** (n=64, tanh interface w=2, τ=1, 2500-step equal-time window) — parasitic
  current, coexistence densities, Young–Laplace `σ = Δp·R` with `p = cs²ρ_tot + G_12 cs² ρ_1ρ_2`.
* **W2 stratified two-layer Poiseuille** (ny=25/41, X-periodic, no-slip rows 0 and ny−1, mirror wall,
  `G_x = 5e-5`, `G_12 = −2.5/-2.0`, 12 000 steps) against the closed-form two-layer solution of
  `cases/mcmp_stratified_poiseuille/analytic.py` with `H = ny−2` and mixture-rule viscosities.
* **W3 momentum ledger** (n=48, periodic, no body force, 400 steps) — the discriminator of H2.
* **W4 depleted-cell guard** — one collision on a field with a 2×2 cell at `ρ_1=1e-8`.

Tool chain: pure torch 2.7.1 CPU (loong64), numpy, matplotlib; `run_xval.py --only W1|W2|W3` + `--merge`
then `analyze.py`. Raw data `artifacts/result.json`, tables `artifacts/tables.md` / `table.csv`,
verdicts `artifacts/analysis.json`.

### Deviations from the frozen design (no threshold moved)

1. W1 stopping rule became a **fixed 2500-step equal-time window** instead of "relax until
   |u|max plateaus", because the mixture/OSS arms never plateau (see §5). Every run still reports its
   own `converged` flag.
2. The `LIB-ENTRY` harness control was listed in `predeclared.json → arms` but not in the primary
   block; it was run at the primary W2 configuration.

## 3. Results (all numbers from `artifacts/tables.md` / `analysis.json`)

### 3.1 W2 primary: ny=25, τ=(1,1), G_12=−2.5, 12 000 steps (analytic `Q_band = 0.3036`, `u_max = 0.0198`)

| arm | Q (band) | err Q vs analytic | u_max | err u_max | err profile L2 | steady |
|---|---|---|---|---|---|---|
| GOT-MIX | 0.3093 | **1.86 %** | 0.02713 | 37.01 % | 18.26 % | yes |
| GOT-SELF | 0.2896 | 4.61 % | 0.01913 | 3.40 % | 8.30 % | yes |
| LBM4 | 0.7296 | **140.31 %** | 0.06420 | 224.23 % | 150.03 % | no |
| LBM3 | NaN | diverged | – | – | – | no |
| CTRL-NORM | 0.1870 | 38.41 % | 0.01204 | 39.17 % | 39.16 % | yes |
| LIB-ENTRY | 0.3093 | 1.86 % | 0.02713 | 37.01 % | 18.26 % | yes |

**Harness control:** `LIB-ENTRY` vs `GOT-MIX` differ by `0.00e+00` on Q, u_max and every profile
entry — this harness really executes the shipped solver, it is not a re-typed substitute.

**Anchor comparison (GOT-MIX vs OSS, predeclared 15 % criterion):**

| arm | dev Q | dev u_max | verdict |
|---|---|---|---|
| LBM4 | **135.9 %** | 136.6 % | FAIL (>15 %) |
| LBM3 | diverged (NaN by 12 000 steps; 4.3 s vs 32 s wall clock) | – | FAIL (no result) |
| GOT-SELF (contrast) | 6.4 % | 29.5 % | PASS |
| CTRL-NORM (defect) | 39.5 % | 55.6 % | FAIL — the metric has teeth |

### 3.2 W1 static droplet (2500 steps, equal-time)

| G_12 | arm | max\|u_mix\| | converged | ρ1 centre | ρ1 corner | σ = Δp·R |
|---|---|---|---|---|---|---|
| −2.5 | GOT-SELF | 9.71e−03 | **yes** | 0.915 | 0.141 | 0.1735 |
| −2.5 | GOT-MIX | 8.56e−02 | no | 1.073 | 0.018 | 0.3424 |
| −2.5 | LBM4 | 1.82e−01 | no | 0.915 | 0.015 | −0.072 |
| −2.5 | LBM3 | NaN (diverged) | no | – | – | – |
| −1.5 | GOT-SELF | 3.25e−04 | no | 0.228 | 0.228 | – (no separation) |

Parasitic-current ordering at G_12=−2.5: **LBM4 : GOT-MIX : GOT-SELF = 18.7 : 8.8 : 1**. The OSS
mixing is the *worst* of the three by this metric, and the per-component reading does not even stay
finite. Young–Laplace consistency over R ∈ {8,12,16} at G_12=−2.0: radius residual **GOT-MIX 24.1 %**,
**LBM4 82.4 %**.

### 3.3 W3 momentum ledger (400 steps, closed periodic box, no body force)

| arm | ΔP_y | ΔP_y/step | finite |
|---|---|---|---|
| GOT-MIX | 5.34e−05 | 1.34e−07 | yes |
| GOT-SELF | −4.96e−05 | −1.24e−07 | yes |
| LBM4 | −1.45e−04 | −3.62e−07 | yes |
| LBM4-TAU | −1.45e−04 | −3.62e−07 | yes |
| CTRL-NORM | **−1.311** | −3.28e−03 | yes |
| LBM3 | NaN | – | **no** |

The box sum of `F_1+F_2` is exactly `0` (it is a discrete gradient on a periodic box: 0.000e+00), so
the *global* momentum of a gradient force is conserved by construction; what the mixing rule can leak
is the redistribution. Measured ordering: OSS reading A leaks 2.7× the barycentric form, the
mis-normalised control leaks 2.4e4×, and reading B does not survive at all.

### 3.4 W4 depleted-cell guard (one 2×2 cell at ρ_1=1e-8)

| arm | finite | max\|u_mix\| in patch | max\|u_mix\| global |
|---|---|---|---|
| GOT-MIX | yes | **3.25e−02** | 6.21e−02 |
| LBM4 (guard) | yes | **0** | 0.124 |
| LBM3 (guard) | yes | **0** | 0.110 |

Our `clamp(ρ_tot, 1e-12)` lets a depleted cell carry a 0.032 velocity from a 1e-3 momentum remnant;
the OSS guard zeroes it. This is the one OSS behaviour that is strictly better and costs nothing.

## 4. Figures (`figures/`, generated by `analyze.py` from `result.json`)

| file | content |
|---|---|
| `fig1_deviation_bars.png` | deviation of every arm from GOT-MIX on Q, with the 15 % owner line and the 3 % in-house line |
| `fig2_droplet_spurious.png` | W1 parasitic-current level per mixing convention at G_12 ∈ {−1.5, −2.5} |
| `fig3_momentum_ledger.png` | W3 global-momentum drift versus step, all arms |
| `fig4_profile_primary.png` | W2 τ=(1,1): u_mix profiles of all arms against the analytic parabola |
| `fig5_profile_shift_factor.png` | W2 τ_upper=0.75 block (all arms non-finite — the figure documents the divergence) |

## 5. Convergence and parameter sensitivity

1. **Grid resolution (ny = 25 → 41, G_12=−2.5):** GOT-MIX error on Q goes 1.86 % → 10.68 %
   (u_max 37 % → 46 %, profile L2 18.3 % → 18.1 %); LBM4 stays at 133 % and is non-steady at both
   resolutions. The integral flux is therefore resolution-sensitive at the ~10 % level on this
   coarse two-layer configuration (interface width 3 lattice units over 23 rows), which is the same
   order as the owner 15 % band — so the *anchor comparison* (136 %) is safe, while the *absolute*
   accuracy claim for GOT-MIX must be quoted as ≈2 % at ny=25 and ≈11 % at ny=41, not as a single
   number.
2. **Coupling strength (G_12 = −2.5 → −2.0, ny=25):** GOT-MIX err Q 1.86 % → 0.00 %, LBM4
   140 % → 114 %. The OSS defect is not a strong-coupling artefact; it is present at both couplings.
3. **τ asymmetry (τ_upper = 0.75):** every arm went non-finite (GOT-MIX included). This is the
   already-documented library caveat (mixture form diverges for |G_12| ≥ 1.5 with unequal τ), so the
   shift-coefficient factor **could not be isolated experimentally** — an honest gap, not a pass. It
   is instead settled analytically: for the velocity-shift family the delivered momentum is `s·F/τ`,
   so the OSS coefficient 1 delivers `F/τ` (33 % excess at τ=0.75) while our τ-shift delivers exactly
   `F`.
4. **Relaxation window (W1):** the `self` arm converges (|u|max plateau at ≈2250 steps), the mixture
   and OSS arms do not converge at all within 8000 steps (|u|max 0.05–0.08, liquid density
   over-compressed to ρ1≈1.07 against the coexistence value 0.88). The W1 comparison is therefore an
   equal-time comparison; its numbers are differences between *transient* states, and are reported as
   such.

### Where the design was mis-chosen (honest)

* W1 at G_12=−1.5 and the whole R-sweep at G_12=−2.0 fall **below the separation threshold of the
  linear-ψ SC model** on this lattice: the `self` arm stays homogeneous (ρ1 = 0.228 = 0.302 everywhere)
  instead of forming a droplet. Those rows carry no Young–Laplace information for the `self` arm; the
  mixture/OSS arms do separate there, which is a *dynamic* pile-up driven by their own parasitic flow,
  not a coexistence measurement. A correct R-sweep must be run at G_12 ≤ −2.5; that is the next round's
  first item.
* Consequently the "σ across R" residual (GOT-MIX 24.1 %, LBM4 82.4 %) is contaminated by the same
  issue and is reported as an indicator of *how much the OSS mixing distorts the density field*, not
  as a calibrated surface tension.

## 6. Failure modes, and what would falsify this report

* The OSS literal reading (LBM3) **diverges** in every workload (droplet, channel, ledger) — the
  documented "force term normalised by total density while the force is normalised by component
  density" is not a cosmetic wart, it is fatal once the interaction force is not negligible.
* The OSS reading A (LBM4) is finite but systematically over-advects: it adds the whole inter-component
  force to the mixture momentum, so at |G_12| = 2.5 the advecting velocity carries an O(0.04) interface
  term — the same order as the entire driven flow (u_max = 0.02). Effect: +140 % flux in the channel,
  8.7–18.7× the parasitic current in a droplet, 2.7× the momentum redistribution leak.
* **Falsifiers, checked:** `CTRL-NORM` (a one-line mis-normalisation) exceeds 15 % on every W1/W2
  comparison (`control_teeth = True`), and `LIB-ENTRY` reproduces `GOT-MIX` bit-for-bit
  (`HARNESS-OK`, rel diff `0.00e+00`). If either had failed, the whole comparison would be void.
* The measurement is *relative to this library's* force discretisation and wall model; the upstream
  weights are 3× the lattice weights (their G must be rescaled by 1/3) and their walls are fixed-ρ wet
  nodes rather than our mirrored solid mask — neither changes the mixing algebra, but an absolute
  quantitative mapping to their `drop_rise_mcmp_8boxes` case is not claimed.

## 7. Difference list: TensorLBM vs `listLBM` shan-chen-mcmp, and the porting trade-offs

| aspect | `listLBM` (inventory id 44) | TensorLBM | porting decision |
|---|---|---|---|
| mixture velocity | force inside the mixture momentum: `(Σj+F)/ρ_tot + F_σ/ρ_σ` | `Σj/ρ_tot + τF_σ/ρ_σ` (`u_eq="mixture"`) | **do not port**: 136 % off the analytic channel solution |
| force normalisation | total density in the force term, component density in the force (flagged inconsistent in the inventory) | component density throughout | **do not port**: divergent in all three workloads |
| shift coefficient | 1 ("no source term") ⇒ delivers `F/τ` | `τ` ⇒ delivers exactly `F` | **do not port**: only coincident at τ=1 |
| depleted-cell handling | `u = 0` if `ρ_tot` below a floor | `clamp(ρ_tot, 1e-12)` | **port**: measured 3.25e−02 → 0 in a depleted cell, zero cost |
| pseudopotential | pluggable strategy (CS / VW / SC / ψ=ρ), interaction weights 3× lattice weights | linear `ψ=ρ` on this MCMP path, `G_12` absorbs the weight factor | no change (equivalent up to G rescaling) |
| force discretisation | backward-gather pair force, single stencil | same "standard" form **plus** exact-difference (EDM, Kupershtokh) option | keep ours (EDM not exercised here) |
| walls | fixed-`ρ_wall` "wet nodes" + Zou–He in/outflow, blank collision | solid mask + post-streaming bounce-back, `dry`/`mirror` fill | no change (wall model held fixed across arms) |
| component bookkeeping | precomputed `othercomp`/`othercomppotid` index tables over `ncomps` | dense per-component tensors | no change (sparsity is an orthogonal, already-inventoried feature) |
| streaming | in-place swap streaming over `halfqs` | batched pull-stream (`torch`) | no change |
| guardrail on dipoles | – | `mcmp_verification` guardrails (`centre_not_liquid`, `non_finite_field`) | keep |

**Net porting outcome:** one guard, no algorithm. The OSS file remains useful as a *warning*: its own
inventory row already names the inconsistency that the runs here quantify (140 % / divergence), which
is why "read the OSS method, then measure it" beats "adopt the OSS method".

## 8. Friction log (ecosystem)

* **No outbound network in the work container**: `git clone` of the upstream repo yields no tree, so
  the Fortran sources cited by the inventory cannot be read directly. Consequence: the algebra was
  re-derived from the inventoried description; the two readings are kept as separate arms instead of
  one being assumed. A vendored snapshot of the OSS repos used by the inventory would remove this
  whole class of ambiguity.
* **`oss_solutions` table not readable** by the platform DB role (`permission denied`); the in-repo CSV
  dump was used instead. The dump and the table can drift apart silently.
* **`torch.set_num_threads` default 128 on this 128-core loong64 box makes these small-grid kernels
  18× slower** (600 droplet steps: 58.3 s at 128 threads, 3.27 s at 1 thread). Any LBM micro-benchmark
  on the control plane must pin threads; recorded here because it silently wastes hours of wall clock.
* **120 s wall limit per shell command** on the control plane, so the matrix is run as detached
  `setsid` processes (`run_xval.py --only W1|W2|W3` + `--merge`); the cluster submission
  (`run_cluster`, cj1894) sat 299 jobs deep in the queue, so a ~10 min single-core matrix is faster
  done locally than queued.