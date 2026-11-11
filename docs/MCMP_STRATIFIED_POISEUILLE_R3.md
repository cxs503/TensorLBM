# Stratified two-layer Poiseuille (MCMP) — round 3: force-discretisation study + L4 evidence bundle

**Anchor** `GAP-LBM-MCMP-POISEUILLE` · mission m15063 (stands on m14431 base + m14689 round-2
repair) · model: Shan–Chen two-component D2Q9 + BGK, `u_eq="viscous"`, `force_scheme="guo"`,
neutral-wetting (`mirror`) SC wall.

All numbers below are **generated** by `cases/mcmp_stratified_poiseuille/r3_benchmark.py` from the
raw records in `artifacts/mcmp_stratified_poiseuille/final/` into
`benchmarks/verified/mcmp_stratified_poiseuille/result_r3.json` (source of truth; per-record
sha256 in `provenance`). Nothing is hand-typed.

## 1. Case, geometry, parameters

Plane channel, `nx = 4` (x-periodic, uniform in x), height `ny ∈ {25, 33, 49}` rows, driven by a
uniform body force `G_x = 5e-5` per unit **mixture** volume. Component 1 (`rho_heavy = 0.7`)
fills the lower half, component 2 (`rho_light = 0.3`) the upper half; the SC repulsion
(`G_12 = -2.5`) holds a diffuse interface of ≈3 cells. Both layers co-move, so the reference is
the classical two-layer Poiseuille solution

```
u'' = -G/nu_layer,  u(0)=0, u(H)=0,  u_l(h)=u_u(h),  nu_l u_l'(h) = nu_u u_u'(h)
```

closed form in `cases/mcmp_stratified_poiseuille/analytic.py` (`layered_poiseuille`, validated
against the single-layer parabolic limit). `nu_sigma = cs²(tau_sigma - 1/2)`; two viscosity rules
are reported because the "true" layer viscosity of a diffuse two-component interface is
model-dependent: **mixture** `nu_layer = Σ rho_sigma nu_sigma` (continuum limit of the co-moving
BGK model, primary) and **component** `nu_layer = nu of the component occupying the layer`
(the QR-audit convention).

| parameter | value |
|---|---|
| `ny` / `nx` | 25, 33 (acceptance) / 49 (hold-out) / 4 |
| `G_x` | 5e-5 |
| `G_12` | −2.5 (unchanged from rounds 1–2: **no parameter shopping**) |
| `rho_heavy`, `rho_light` | 0.7, 0.3 |
| `tau_lower`, `tau_upper` | 1.0, 0.75 (nu = 1/6, 1/12) |
| walls | post-streaming bounce-back, rows 0 and ny−1 → halfway planes at 0.5 and ny−1.5, `H = ny−2` |
| SC wall | `mirror` (neutral wetting) |
| steps | 20000 / 30000 / 45000 |

## 2. What this round adds

1. **Exact-difference (well-balanced) interaction force, `sc_form="edm"`** — new option in
   `tensorlbm.multiphase.sc_two_component_force` / `collide_sc_two_component`. The pair force is
   split as
   `-G12 rho1∇rho2 = -G12[½∇(rho1 rho2) + ½(rho1∇rho2 − rho2∇rho1)]`: a pure discrete gradient
   (balanceable by the pressure of the same operator, cannot generate vorticity) plus a
   momentum-conserving exchange term (Kupershtokh et al. 2009). Same continuum limit as the
   historical backward-gather form, which stays the default and is bit-identical (asserted).
2. **Round-3 matrix driver** `run_final_matrix.py` — 12 pre-declared cells
   `{standard, edm} × {driven, static} × ny ∈ {25,33,49}`, shardable (`--shard i/n`).
3. **Library entry point** `tensorlbm.cases.two_layer_poiseuille.run_two_layer_poiseuille`
   (+ `scripts/run_two_layer_poiseuille_check.py`), i.e. the case is reachable from *library* code,
   not only from the benchmark directory; `tests/test_two_layer_poiseuille_entry.py` exercises it.
4. **L4 evidence bundle**: `result_r3.json` (ref vs got + errors + verdicts + provenance +
   `repro_cmd` + `external_anchor`), figures `docs/figures/r3_*.png`, this report.

## 3. Results

### 3.1 Criterion ① interface position < 5 % — **PASS**

| ny | sc_form | `y_int` | analytic `y = 0.5+(ny−2)/2` | err |
|---|---|---|---|---|
| 25 | standard | 11.54 | 12.00 | −3.84 % |
| 25 | edm | 11.57 | 12.00 | −3.57 % |
| 33 | standard | 15.53 | 16.00 | −2.95 % |
| 33 | edm | 15.53 | 16.00 | −2.85 % |
| 49 (hold-out) | standard | 24.33 | 24.00 | +1.38 % |

### 3.2 Criterion ② two-phase volume flow rate < 10 % — **PASS**

`Q_mix = Σ_y u_x,mix`, `u_mix = (rho1 u1 + rho2 u2)/(rho1+rho2)`: the trace component's
`1/rho_sigma`-amplified velocity is not a continuum velocity and is not used (round-1 finding).

| ny | sc_form | `Q_mix` | analytic (mixture rule) | err mixture | err component |
|---|---|---|---|---|---|
| 25 | standard | 0.3912 | 0.4074 | −3.97 % | −6.46 % |
| 25 | edm | 0.3924 | 0.4074 | −3.68 % | −6.17 % |
| 33 | standard | 0.9705 | 0.9975 | −2.71 % | −5.24 % |
| 33 | edm | 0.9725 | 0.9975 | −2.51 % | −5.03 % |
| 49 (hold-out) | standard | 3.4867 | 3.4764 | +0.29 % | −2.30 % |

Note on convention: under the *rejected* cell-centre convention (`H = ny−1`) the same records read
−15 % / −12 % / −6 % (mixture rule). The wall convention was calibrated once in round 1
(`--mode walls`: `H = ny−2` reproduces `u_max`/`Q` to 0.06–1.8 %, `H = ny−1` misses by 8 %/12 % at
ny=25) and is applied unchanged here. This is the single largest systematic in the comparison and it
is resolved by calibration, not by fitting the flow-rate threshold.

### 3.3 Criterion ③ interface spurious current < 0.005·U_max — **FAIL (honest)**

Static control (`G_x = 0`, same stratification and walls), `S_ifc = max|u_y,mix|` inside the
interface band `|alpha−0.5| < 0.25`, normalised by the analytic `u_max` of the driven case at the
same resolution:

| ny | sc_form | `S_ifc` | `S_all` | `U_max` (analytic) | `S_ifc/U_max` |
|---|---|---|---|---|---|
| 25 | standard | 9.30e−3 | 1.02e−2 | 2.66e−2 | 0.350 |
| 25 | edm | 9.14e−3 | 1.03e−2 | 2.66e−2 | 0.344 |
| 33 | standard | 8.66e−3 | 9.67e−3 | 4.83e−2 | 0.180 |
| 33 | edm | 8.46e−3 | 1.03e−2 | 4.83e−2 | 0.175 |
| 49 | standard | 7.35e−3 | 9.56e−3 | 1.11e−1 | 0.066 |

- The gate needs `S_ifc < 1.3e−4 … 5.6e−4`; the model delivers `≈7e−3`, i.e. 13×–50× too much.
- **EDM gains the transient only.** At 3000 steps the edm arm suppresses the interface-band
  current 15× (7.6e−4 vs 1.1e−2, ny=25, `artifacts/…` probe log in the mission record), but by
  20 000–45 000 steps the two arms coincide (0.34 vs 0.35 at ny=25). The long-time state is
  therefore *not* the initial force imbalance: the two components keep segregating (heavy fraction
  overshoots to `phi_contrast ≈ 2.0`, i.e. rho1 → >0.7 near the lower wall), so the late-time
  interface is thinner than the 3-cell tanh and the steady residual is dominated by the unresolved
  sharp-gradient layer, not by the discrete curl of the pair force.
- `fail_stage`: **model/state identification** — the criterion is defined at the *steady* state of a
  segregation-driven interface that this two-component SC model keeps thinning (no surface-tension
  equilibrium is imposed: total density is uniform, so there is no Laplace pressure to hold the
  interface). Consequence: `S_ifc` plateaus at `O(1e-2)` for every tested force discretisation.
- Tried and ruled out (rounds 2–3): dry→mirror wall (8.7× gain, done in R2); resolution 25→49
  (5× in the ratio, not enough); `|G12|` scan (R2: ratio 0.0019 at `|G12|=2.0`, ny=49 — but that
  regime has weaker stratification, `phi = 0.67`, and the driven acceptance cell was never
  validated there); standard vs edm force (transient only).
- Round-3 inputs for the method upgrade: (i) impose a surface-tension-bearing free-energy /
  Cahn–Hilliard model so the interface has an equilibrium width and a Laplace pressure; (ii) or
  define ③ on a *static droplet / flat interface at coexistence* (where `S_ifc` is the classical
  parasitic current of the force scheme and EDM's 15× transient gain is the steady gain too);
  (iii) or normalise by the capillary velocity `sigma/mu` instead of the body-force `U_max`.

### 3.4 Criterion ④ resolution trend — **PASS**

`|err(y_int)|`: 3.84 % → 2.95 % → 1.38 %; `|err(Q_mix)|`: 3.97 % → 2.71 % → 0.29 % (standard arm);
edm arm 3.57 % → 2.85 % for the interface and 3.68 % → 2.51 % for the flow rate. Both criteria
converge monotonically towards zero with the sign of the interface error changing at ny=49, i.e.
the residual is the first-order half-cell wall calibration plus the diffuse-interface thickness.

### 3.5 Criterion ⑤ report — **this document** (six elements; figures below)

## 4. Figures

| figure | content |
|---|---|
| `docs/figures/r3_interface_and_profile.png` | heavy-fraction profile (ny=49) and `u_x,mix(y)` for ny=25/33/49 vs the two-layer analytic (dotted) |
| `docs/figures/r3_spurious_current.png` | static-control `S_ifc(t)` decay (semi-log) and the criterion-③ ratio per cell, gate line at 0.005 |
| `docs/figures/r3_error_trend.png` | criteria ①/② error vs `ny` with the ±5 % / ±10 % acceptance bands |

## 5. Caveats

- float32, CPU, deterministic; differences below ~1e-6 are not physical.
- ny=49 driven record is `steady=False` at 45 000 steps (its `Q` error, +0.29 %, is inside the gate
  but it is labelled transient and does not enter the acceptance flag).
- Interface ≈3–4 cells: at ny=25 that is ~15 % of the channel, so the analytic `h` is a
  volume-averaged position; no extrapolation to density ratio >10, `|G_12| > 3`, or 3D.
- Criterion ③ is reported as FAIL; no threshold in `spec.json` was changed after the runs.

## 6. Reproduction

```bash
PYTHONPATH=src python cases/mcmp_stratified_poiseuille/run_final_matrix.py       # 12 cells (~35 min CPU)
PYTHONPATH=src:. python cases/mcmp_stratified_poiseuille/r3_benchmark.py         # result_r3.json + figures
PYTHONPATH=src:. python -m pytest tests/test_two_layer_poiseuille_entry.py \
    tests/test_mcmp_wall_neutral.py tests/test_mcmp_stratified_poiseuille.py -q
PYTHONPATH=src:. python scripts/run_two_layer_poiseuille_check.py --ny 25 --steps 20000
```

## 7. Artifact map

| path | content |
|---|---|
| `artifacts/mcmp_stratified_poiseuille/final/*.json` | 12 round-3 records (driven: profiles + flux; static: full `S(t)` series) |
| `benchmarks/verified/mcmp_stratified_poiseuille/result_r3.json` | five criteria, ref-vs-got per cell, provenance (sha256), `repro_cmd`, `external_anchor` |
| `src/tensorlbm/multiphase.py` | `wall_mode={"dry","mirror"}` (R2) + `sc_form={"standard","edm"}` (R3) |
| `src/tensorlbm/cases/two_layer_poiseuille.py` | library entry point (integration caller) |
| `cases/mcmp_stratified_poiseuille/run_final_matrix.py` | pre-declared round-3 matrix driver |
| `cases/mcmp_stratified_poiseuille/r3_benchmark.py` | benchmark assembler + figures |

## 8. External anchors

- Shan & Doolen (1995) *J. Stat. Phys.* **81**:379 — multicomponent Shan–Chen LBM.
- Martys & Chen (1996) *J. Comput. Phys.* **126**:276 — viscous (viscosity-weighted) velocity coupling.
- Guo, Zheng & Shi (2002) *Phys. Rev. E* **65**:046308 — second-order forcing.
- Kupershtokh, Medvedev & Karpov (2009) *Comput. Math. Appl.* **58**:965 — exact difference method / well-balanced SC forcing.
- Krüger et al. (2017) *The Lattice Boltzmann Method*, Springer, ch. 9 — spurious currents at diffuse interfaces, and the two-layer Poiseuille analytic reference (ch. 4).
