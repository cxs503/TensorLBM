# XVAL: AA-pattern fused push–pull stream–collide (verified)

**Anchor**: `XVAL-LBM-FUSED-PUSH-PULL-AA-PATTERN`
**Reference practice**: `lbm-principles-practice/code` — fused push–pull AA pattern
(Krüger *et al.*, *The Lattice Boltzmann Method: Principles and Practice*, ch. 5;
halfway bounce-back rule Eq. 5.58)
**Repo**: TensorLBM · **Code**: `src/tensorlbm/aa_fused_push_pull.py`
**Tests**: `tests/test_aa_fused_push_pull.py` (10 passed) ·
**Evidence**: `artifacts/xval_fused_push_pull_aa/`

---

## 1. What the pattern is, and what "fused" actually buys

The AA ("Abstract Array") pattern fuses collision *and* streaming into one pass
by exploiting the one property of the collision operator that matters here:

> `rho(x,t)` and `u(x,t)` are collision invariants — identical before and after
> the collision step.

Therefore the population that *leaves* a source node `x − c_i` after collision
can be reconstructed while pulling, without ever materialising a post-collision
array:

```
f_i(x, t+1) = f_i(x − c_i, t) + ω·[ f_i^eq(x − c_i, t) − f_i(x − c_i, t) ]
              └──── pull pre-collision ────┘   └── collision increment at SOURCE ──┘
```

Both terms are read at the **source** node. This is what makes the fused pull
*scheme*-equivalent (not merely steady-state-equivalent) to the canonical
"collide everywhere, then stream" push scheme.

The paper-adjacent trap: a pull loop that streams **first** and then collides
**at the destination** is *not* this pattern. Streaming and collision do not
commute; the destination-colliding variant simulates a different (Klemm/LGA-
ordered) scheme. It coincides with the push scheme in a fully periodic domain
only because there the two operations are permutations of the same set of local
updates — and it diverges as soon as a boundary changes the local update set.

### 1.1 Bounce-back written inside the pull loop

Halfway bounce-back in pull form states: if the upstream node `x − c_i` of
direction `i` is solid, the incoming population is the reflection of the
population leaving `x` towards the wall,

```
f_i(x, t+1) = f_{opp(i)}^post(x, t) + 6·w_i·ρ(x)·(c_i · u_wall) ,   c_{opp(i)} = −c_i
```

`f^post` is again available as `f + increment` **at the same node**, so no extra
storage is needed. `u_wall` is sampled from the adjacent solid node, and the
sign of the wall term is fixed by the requirement that a wall moving in `+x`
boosts the population travelling in `+x`.

Because the reflection is written at the *fluid* node, mass is exchanged
symmetrically within the cell and the total fluid mass is conserved to
round-off. A bounce-back written on *solid* nodes (`f[i, solid] = f[opp, solid]`)
does not have this property.

### 1.2 Body force

Guo's second-order forcing (Guo, Zheng & Shi, *JCP* 165 (2000) 288):

```
S_i = w_i [ 3(c_i − u)·F + 9(c_i·u)(c_i·F) ]
f_i^post = f_i + ω(f_i^eq − f_i) + (1 − ω/2)·S_i ,      ρu = Σ c_i f_i + F/2
```

so steady plane Poiseuille flow has `u_max = gH²/(8ν)`, `ν = (τ − ½)/3`.

---

## 2. Independent review of the prior artifact (adversarial pass)

`docs/xval_aa_pattern_fused_stream_collide.md` (mission m17976) claimed the
anchor complete with "L2 Complete". Re-running its kernel shows the claim is
overstated; the following are blocking:

| # | Severity | Finding | Evidence |
|---|----------|---------|----------|
| F1 | **blocking** | `aa_fused_stream_collide.aa_fused_stream_collide()` streams first and collides at the **destination** — that is not the AA fused push–pull pattern; the "collision invariance" fusion is absent. | code read; §1 above |
| F2 | **blocking** | Bounce-back is applied **on solid nodes** with a sequential in-place loop over `i = 1..8` (`f_out[i, solid] = f_out[opp, solid]`), so pairs are overwritten in loop order and the rule is neither halfway bounce-back nor mass conserving. | closed walled box, 500 steps, perturbed rest state: relative mass drift **2.4 × 10⁻⁵** (legacy) vs **3.4 × 10⁻¹⁴** (this work) |
| F3 | **blocking** | The wall-bounded comparison against Ghia (n = 33, `L2_u = 0.247`, mass 3.94 vs 0.24) is presented as evidence of "physical validity"; with F2 the wall-bounded runs are not a valid LBM solution, so the comparison cannot support any accuracy claim. | same doc §4.2 |
| F4 | misleading | "Different intermediate states with solid boundaries … expected and documented" is used to explain away an O(1) mass discrepancy. With a correct halfway bounce-back the fused and push schemes agree to round-off *including* walls, so the discrepancy was a defect, not an intrinsic ordering property. | §3.2 below (`max |Δf| = 0.0`) |
| F5 | non-blocking | Correctly verified: periodic equivalence (max diff 5.55 × 10⁻¹⁷) and stability; those results are reproduced here (bit-identical, `0.0`). | §3.1 |

Disposition of the prior artifact: **request changes → superseded** by this
document and by `src/tensorlbm/aa_fused_push_pull.py`. The legacy module is left
untouched (other missions import it) and its defect is pinned by
`test_legacy_kernel_leaks_mass_with_walls`, which fails loudly if the old kernel
is ever changed without re-validation.

---

## 3. Verification results

All runs: D2Q9, **float64**, single core, BGK. `f` shape `(9, ny, nx)`.

### 3.1 Fused vs push, fully periodic (200 steps, perturbed rest state)

| Metric | Value | Gate | Verdict |
|---|---|---|---|
| `max |f_fused − f_push|` | **0.0** | < 1e-14 | PASS |

### 3.2 Fused vs push, walled channel + halfway bounce-back (300 steps)

| Metric | Value | Gate | Verdict |
|---|---|---|---|
| `max |f_fused − f_push|` | **0.0** | < 1e-12 | PASS |

The fused kernel and the independent push oracle are **bit-identical**, walls and
moving walls included. (The oracle is an independently written
collide → stream → bounce-back pass; the fused pass reads the collision
increment at the source node instead of materialising `f^post`.)

### 3.3 Mass conservation, closed walled box (16×16, 500 steps)

| Kernel | relative mass drift |
|---|---|
| `fused_push_pull_step` (this work) | **3.41 × 10⁻¹⁴** |
| `collide_stream_step` (oracle) | 3.41 × 10⁻¹⁴ |
| legacy `aa_fused_step` | 2.43 × 10⁻⁵ ❌ |

### 3.4 Plane Couette (moving wall via bounce-back wall term)

`n_fluid = 8`, `u_wall = 0.02`, `ν = 0.05`, 4000 steps, analytic `u(y) = U·y/H`:

| Metric | Value | Gate | Verdict |
|---|---|---|---|
| `max|u − u_analytic| / U` | **8.12 × 10⁻¹⁴** | < 3 % | PASS |
| relative L2 | 9.88 × 10⁻¹⁴ | < 3 % | PASS |

This is the test that fixes the *sign* of the moving-wall term: a wrong sign
produces a mirrored profile (≈ 87 % error), as was observed once in the harness
before the moving wall was placed on the correct side.

### 3.5 Plane Poiseuille, grid study (Guo body force)

`g = 1e-6`, `ν = 0.05`, `τ = 0.65`, `nx = 4` periodic in x, `H = n_fluid`.

| `n_fluid` | steps | `u_max` numeric | `u_max` analytic | `u_max` rel. err | profile L2 | steady residual |
|---|---|---|---|---|---|---|
| 16 | 4 000 | 6.35014e-4 | 6.4e-4 | **0.779 %** | 0.510 % | 0.264 % |
| 32 | 15 000 | 2.55340e-3 | 2.56e-3 | **0.258 %** | 0.185 % | 0.046 % |
| 48 | 33 000 | 5.75025e-3 | 5.76e-3 | **0.169 %** | 0.134 % | 0.021 % |

Transverse velocity noise `max|u_y|/u_max ≈ 3 × 10⁻¹²` (fully developed,
incompressible). The residual error is the discrete-equilibrium/steady-state
truncation of BGK at fixed `g/ν`; it decreases monotonically with resolution and
is not a bounce-back error (the Couette test, which is exactly representable,
closes to machine precision).

### 3.6 Error verdict vs the owner precision standard

| Quantity | Reference | Error | < 3 %? |
|---|---|---|---|
| Couette profile | analytic linear | 8.1e-14 | **yes** |
| Poiseuille `u_max` (n=16/32/48) | `gH²/8ν` | 0.78 / 0.26 / 0.17 % | **yes** |
| Poiseuille profile L2 (n=16/32/48) | analytic parabola | 0.51 / 0.19 / 0.13 % | **yes** |
| Fused vs push oracle | independent code path | 0.0 (bit-identical) | **yes** |
| Mass drift, closed box | exact conservation | 3.4e-14 | **yes** |

---

## 4. Cost

~2.6 ms/step on a single CPU core for a 4×18 lattice (torch `roll` per
direction, op-launch bound, 0.026–0.076 MLUPS). No performance claim is made:
the fusion here is an *algorithmic/verification* statement. The paper-level
benefit of the AA pattern (no intermediate `f^post` array) only materialises in
a compiled kernel; the Triton path is tracked separately.

## 5. Limitations / applicability

1. **D2Q9, BGK, float64, CPU.** No MRT/regularised collision, no 3D.
2. **Halfway bounce-back only** (flat or staircased walls); no interpolation-based
   curved-wall bounce-back, no Zou–He velocity inlet.
3. **Guo forcing only** at constant body force; no pressure-gradient inlet/outlet.
4. Lagged vs fused comparison is done in double precision; in float32 the
   bit-identity claim should not be made.
5. The verification domain is plane-channel flow (very well conditioned). No
   statement is made about 3D turbulent wall-bounded flow, where halfway
   bounce-back has a grid-dependent wall location error.

## 6. Reproduction

```bash
cd TensorLBM
PYTHONPATH=src pytest tests/test_aa_fused_push_pull.py -q      # 10 tests, ~50 s
python3 xval/fused_push_pull_aa/run_xval.py --quick            # smoke, ~40 s
python3 xval/fused_push_pull_aa/run_grid_point.py --n-fluid 48 --steps 33000  # one grid point
```

Artifacts: `artifacts/xval_fused_push_pull_aa/xval_fused_push_pull_aa.json`,
`pt16.json`, `pt32.json`, `pt48.json`, `fig1_poiseuille_profiles.png`,
`fig2_couette.png`, `fig3_equivalence_mass.png`.

Cluster jobs submitted for the full study (`cj2541`, `cj2548`) were killed
unconsumed: the first staged a pre-fix harness, the second was superseded once
the same grid points had been completed locally as three short single-core runs.
No result in this document depends on a cluster run.