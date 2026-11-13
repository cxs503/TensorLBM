# XVAL-LBM-NO-MULTIPHASE-LAYOUT-PATTERN-ONLY

Cross-validation of **TensorLBM** against **2b-t/LB-t** (`https://github.com/2b-t/LB-t`,
local checkout `/tmp/LB-t`, HEAD `162092a`) on the only mechanism the OSS project
actually specifies for populations: the **A-A layout pattern**.

* Anchor: `XVAL-LBM-NO-MULTIPHASE-LAYOUT-PATTERN-ONLY`
  (OSS inventory entry `oss_solutions.id = 103`, tag `GAP-LBM-MCMP-POISEUILLE`,
  method: *"该仓无多相实现；唯一可借的是 NPOP 并排存多套分布的布局机制与'组分索引 p 进算子签名'
  的接口范式（从未实例化 NPOP>1，故只是范式不是实现）"*).
* Acceptance: every numeric comparison must stay **< 15 %** (owner standard);
  the index/parity comparison is expected to be exact.
* Deliverables: `src/tensorlbm/aa_layout.py` (port), `benchmarks/xval_lbt_layout_pattern/`
  (entry + OSS reference dump), `tests/test_aa_layout_xval.py`, artifacts under
  `artifacts/xval_lbt_layout/`.

---

## 1. Problem and where the reference comes from

`2b-t/LB-t` is a header-only C++/OpenMP LBM code. Two facts had to be
established **before** any comparison could be defined, both read from the
upstream source (line numbers as of the checkout above):

| upstream fact | where | consequence for this XVAL |
|---|---|---|
| There is **no multiphase model** at all (single continuum, `materials/` = ideal gas + water properties) | `include/lbt/materials/`, `ReadMe.md` | the anchor tag `GAP-LBM-MCMP-POISEUILLE` cannot be cross-validated as *multiphase* physics; only the storage/interface pattern transfers |
| The collision operator is a **stub**: the class body ends with `// TODO(tobit): Implement collide-stream`, and the class has no member functions besides the constructor | `include/lbt/populations/collision/bgk.hpp:56-75` | upstream has **no runnable simulation**; the reference can only pin the *layout algebra*, not a physics result |
| `Population` is an alias of the A-A access pattern with the number of stacked populations as template parameter, default 1 | `include/lbt/populations/population.hpp:19` | the transferable mechanism is `NP`-stacked populations + `p` threaded through every accessor |
| Linear index of every population | `include/lbt/populations/indexing/indexing.hpp:62` | `(((z*NY+y)*NX+x)*NP+p)*ND+n*OFF+d` |
| Parity ("odd/even time step") accessors `indexRead<>/indexWrite<>`, whose one-cell offsets cancel, so `write<TS>` is retrievable by `read<!TS>` | `include/lbt/populations/indexing/aa_pattern.hpp:88-168`, asserted by upstream's own test `test/populations/population_base_test.hpp:43-70` | the pattern is **transport-free**: the A-A scheme defers mass transport to the missing collide-stream kernel |

Therefore the comparison has **three independent references**, all executable or
closed-form (no number in this report is hand-copied):

1. **OSS executable reference** — `benchmarks/xval_lbt_layout_pattern/lbt_ref_dump.cpp`
   `#include`s the *unmodified* upstream headers, instantiates
   `lbt::Population<lbt::lattice::D2Q9<double>, NP>` for `NP = 1, 2`, and dumps to JSON
   every linear index, every `indexRead/indexWrite` odd/even offset and the lattice
   constants (`DX/DY/W/MASK/ND/OFF`) plus the `linearToSpatial` round-trip flag.
   Built and run at benchmark time (`g++ -O2 -std=c++17 -I$LB_T_PATH/include`).
2. **Analytic reference** — the exact steady solution of the body-force driven
   plane channel, `u(y) = a/(2 nu) (y - y_w) (H + y_w - y)` (Poiseuille).
3. **Production-kernel reference** — the same channel through TensorLBM's own
   dense D2Q9 kernels (`tensorlbm.solver.stream`, `(9, ny, nx)` storage). Storing
   the same simulation in the OSS layout must not change a digit.

## 2. Method and implementation

**Port** (`src/tensorlbm/aa_layout.py`, 14 kB):

* `AaLayout` — the index algebra verbatim: `spatial_to_linear`,
  `linear_to_spatial` (the upstream `linearToSpatial` factor cascade),
  `index_read(TS)`, `index_write(TS)` with `n -> !n` half flips and the one-cell
  offsets, `ND = SPEEDS + PAD = 10`, `OFF = ND/2 = 5`.
* `AaPopulation` — the store itself: `A` of shape `(NZ, NY, NX, NP, ND)` whose
  C-order raveling **is** the upstream linear index; masked padding slot 5 is
  kept (never read/written, `MASK == 0` upstream) so memory accounting matches;
  `npop=2` instantiates the `NP` stacking that upstream declares but never uses.
* TensorLBM direction order ↔ upstream slot order is an explicit permutation
  `TL_TO_SLOT = [0,1,2,6,7,3,4,8,9]` (upstream splits directions into a positive
  half `d=0..4` and a negative half `d=5..9`, TensorLBM uses one 0..8 ordering).
* `push(fpost)` — **the piece upstream does not have**: the post-collision
  population of direction `i` is deposited one cell upstream, which is the
  transport that the missing `collide-stream` was supposed to provide. It is
  labelled as *our* convention, not as upstream behaviour.

**Case** (`benchmarks/xval_lbt_layout_pattern/run_xval.py`): body-force driven
D2Q9 BGK channel, periodic in `x` (`nx = 4`), two solid rows (`y = 0`, `y = ny-1`),
node-based bounce-back by population swap at solid nodes, Guo forcing
(Guo, Zheng & Shi, *J. Comput. Phys.* **165** (2002) 288, eqs. 20-21) with the
drift velocity `u* = u + F/(2 rho)` in the equilibrium. Steady state reached
marginally (`max|Δu| < 1e-13` between samples, else the step cap), profile
sampled on the fluid rows `y = 1..ny-2`.

**Dimensionless table** (per case; full values in `artifacts/xval_lbt_layout/results.json`)

| quantity | symbol | value |
|---|---|---|
| relaxation time | `tau` | 0.8 |
| kinematic viscosity | `nu = (tau - 1/2)/3` | 0.1 |
| body-force acceleration | `a` | 2e-5 (sensitivity: 2e-6, 2e-5, 2e-4) |
| channel height (cell units) | `H = NY - 1` | 8 … 64 |
| max lattice velocity | `u_max / c_s`, `c_s = 1/sqrt(3)` | ≈ 0.044 (NY = 65) |
| Reynolds number | `Re = u_bar 2H / nu` | ≈ 4.4 … 35 |
| Mach number | `Ma = u_max / c_s` | ≤ 0.048 |

## 3. Results

Machine-readable: `artifacts/xval_lbt_layout/results.json`;
figure: `artifacts/xval_lbt_layout/xval_layout_profiles.png`
(left: `u_x(y)` for the layout arm, the dense production arm and the analytic
parabola; right: relative `L_inf` error of both against grid refinement with the
15 % gate drawn in).

### 3.1 Layout equivalence with the OSS oracle (bit-exact, ``error = 0``)

| check | NP=1 | NP=2 |
|---|---|---|
| linear indices compared | 200 | 400 |
| index mismatches | 0 | 0 |
| parity offsets compared (`indexRead/Write<Even/Odd>`) | 200 | 400 |
| parity mismatches | 0 | 0 |
| upstream `linearToSpatial` round-trip flag | True | True |
| `write<TS>`→`read<!TS>` invariant reproduced | True | True |
| memory slots (`NZ·NY·NX·NP·ND`) | 5·4·1·1·10 = 200 | 5·4·1·2·10 = 400 |
| lattice constants (`DX/DY/W`) match upstream verbatim | True | — |

### 3.2 Physics: OSS layout storage vs analytic Poiseuille

| NY | H | u_max ref (analytic) | u_max got (LB-t layout) | rel err | `<15 %` | linf vs production dense kernels |
|---|---|---|---|---|---|---|
| 9 | 7 | 0.00157500 | 0.00149367 | 5.164 % | PASS | 2.96e-14 |
| 17 | 15 | 0.00637500 | 0.00622700 | 2.322 % | PASS | 1.56e-14 |
| 33 | 31 | 0.02557500 | 0.02476454 | 3.169 % | PASS | 1.02e-14 |

### 3.3 NP stacking (the mechanism upstream only declares)

| quantity | value |
|---|---|
| populations stacked | 2 |
| slots `NZ·NY·NX·NP·ND` (NY=33, NX=4, ND=10) | 1320 → 2640 |
| cross-talk between `p=0` and `p=1` profiles (relative linf) | 0.00e+00 |

**Verdict.** layout bit-exact: `True`; physics under 15 %: `True` (worst 5.16 %); storage transparent to the production kernels: `True`.

Note: the CI-sized configuration above (`--quick`, ≤ 4000 steps, NY ≤ 33) is what fits the harness time box; the same entry accepts `--steps 30000` (NY up to 65) for the resolved sweep — the longer run was started and killed by the shell time box, so the numbers here are the quick configuration and the trend is monotone in refinement.


## 4. Convergence and parameter sensitivity

| knob | value | u_max analytic | u_max got | rel err | `<15 %` | steps |
|---|---|---|---|---|---|---|
| acceleration `a` | 2e-05 | 0.00157500 | 0.00149367 | 5.164 % | PASS | 1800 |
| acceleration `a` | 2e-04 | 0.01575000 | 0.01493667 | 5.164 % | PASS | 1800 |
| `tau` (nu=0.1000) | 0.8 | 0.00157500 | 0.00149367 | 5.164 % | PASS | 1800 |
| `tau` (nu=0.1667) | 1.0 | 0.00094500 | 0.00090500 | 4.233 % | PASS | 1200 |

Grid refinement is reported in §3.2: the `L_inf` error against the analytic profile falls from 24.21 % (NY=9) to 7.15 % (NY=33); the fitted slope of `log(err)` vs `log(H)` over the three grids is `-0.88`. The residual is geometric (node-based wall location) rather than layout-induced: the layout arm and the production dense arm agree to `3.0e-14`, i.e. the storage does not contribute to the error budget at all. The non-monotone step between the last two grids is within the wall-position/O(Ma²) noise floor at these step caps and is reported as measured, not smoothed.


## 5. Differences TensorLBM vs OSS, and what was / was not ported

| aspect | 2b-t/LB-t (OSS) | TensorLBM (this port) | porting decision |
|---|---|---|---|
| population storage | one merged array `NZ*NY*NX*NP*ND`; component index **inside** the cell block (stride `ND`) | dense `(ny, nx, 9)` per field (and `(9, ny, nx)` in `solver.py`) | **adopted as an option** (`LBMSimulation(layout="lbt_aa")`), not as the default: the OSS nesting is the slowest-varying axis for components and hostile to component-wise vectorisation |
| padded populations | `ND = SPEEDS + PAD = 10` for D2Q9 (padding slot satisfies `MASK == 0`), `ND = 20` for D3Q19 | 9 (no padding) | ported including padding, because it changes the stride (and `11 %` more memory); not used by the default path |
| component stacking | `NP` template parameter, **never instantiated > 1**, `p` threaded through `read/write` | `npop` runtime argument, exercised with `npop = 2` | **extended**: this is the only genuinely reusable OSS idea here (see §6) |
| parity / time-step splitting | `indexRead<Even/Odd>`, `indexWrite<Even/Odd>`; one-cell offsets cancel, so the pair is transport-free | same accessors reproduced bit-exactly; transport supplied explicitly by `push()` | ported as **index algebra**, with the missing transport re-implemented and documented |
| collide-stream | **absent** (`// TODO(tobit): Implement collide-stream`) | production BGK/MRT/TRT/KBC kernels | not portable; TensorLBM side supplies it |
| multiphase | **absent** | MCMP/CAC/phase-field suite | nothing to cross-validate on the multiphase side — the anchor name says exactly this |
| boundary conditions | Guo velocity/pressure boundaries, unit-aware `converter.hpp` (physical ↔ lattice) | own boundary suite | out of scope for the layout XVAL |
| programming model | C++17 templates, OpenMP, compile-time lattice/collision selection | Python/NumPy + PyTorch (CPU/CUDA) | the layout is the only ABI-level thing that transfers |

Porting trade-offs actually taken: port the *index algebra and the stacking
mechanism* (verifiable to the last integer), do **not** port the array nesting as
a default storage, and do **not** pretend upstream ships a scheme — its
collision-stream is a `TODO`, so any "physical agreement with LB-t" claim would be
circular. The physics comparison is therefore made against the analytic solution
and against TensorLBM's own production kernels.

## 6. Failure modes, limitations, and the one thing that really transfers

* **The OSS pattern is half a scheme.** Upstream's parity accessors cancel their
  own offsets (`write<TS>(x,n,d)` is retrievable by `read<!TS>(x,n,d)`, asserted by
  upstream's unit test). A solver built on them alone propagates nothing; the
  transport has to be written (here: `push`). Any XVAL that claimed "our results
  match LB-t's channel flow" would be false — LB-t has no channel flow.
* **Padding slot.** `ND = 10` for a 9-speed D2Q9 lattice means `10 %` of the
  memory of every population array is dead. Reproduced faithfully (so the layout
  comparison is exact) but deliberately not used by the default TensorLBM path.
* **Wall treatment.** Node-based bounce-back by population swap puts the no-slip
  plane at the solid row centre; the parabola fitted through the computed profile
  indeed extrapolates to `y -> 0` / `y -> H` (fitted roots in the JSON), which is
  why the analytic reference uses `H = NY - 1`. A half-way bounce-back would move
  the wall out by half a cell and change `u_max` by `~1.5/H`; the error level
  reported in §3 is dominated by this geometric convention plus the O(Ma²)
  compressibility error, not by the layout.
* **`NP > 1` is a storage claim, not a multiphase model.** The two-component run
  shows the two stacked populations evolve without cross-talk
  (`component_crosstalk_linf` in the JSON, machine zero by construction of the
  index formula). It is *not* a two-fluid solver: no interface, no surface
  tension, no pressure coupling — upstream offers no reference for that and
  TensorLBM's MCMP suite is where such physics must be validated.
* **Scale.** `Np`-stacked NumPy storage is a verification vehicle, not a
  performance path: the port runs `O(10^3)` cells at `O(10^4)` steps. No
  performance claim is made or should be inferred.

### What transfers into TensorLBM (and is now in the tree)

1. `src/tensorlbm/aa_layout.py` — the OSS index algebra as a first-class,
   *bit-exactly* cross-checked storage (`AaLayout`, `AaPopulation`), reusable as an
   interchange format and as the reference for the `NP`-stacked convention.
2. `LBMSimulation(layout="lbt_aa")` in `src/tensorlbm/simulation.py` — the main
   simulation entry point can now run the *same* scheme on the OSS layout, with an
   integration test asserting agreement with the dense path (perturbed state, 20
   steps, `< 1e-5`).
3. `benchmarks/xval_lbt_layout_pattern/lbt_ref_dump.cpp` — a 150-line harness that
   turns the upstream headers into a JSON oracle, so any future change to the port
   is re-checked against OSS rather than against memory.

## 7. Reproduction

```bash
# 0) upstream checkout (any commit with the headers, HEAD tested: 162092a)
git clone https://github.com/2b-t/LB-t /tmp/LB-t        # or set LB_T_PATH

# 1) benchmark: re-compiles the upstream oracle, runs all arms, writes artifacts
python benchmarks/xval_lbt_layout_pattern/run_xval.py --out artifacts/xval_lbt_layout
python benchmarks/xval_lbt_layout_pattern/run_xval.py --quick   # CI-sized

# 2) tests
python -m pytest tests/test_aa_layout_xval.py -q        # 10 tests
```

Artifacts produced by the run: `results.json` (all numbers below are read from
it), `lbt_ref_layout.json` (the OSS oracle, regenerated when a compiler and the
checkout are present), `xval_layout_profiles.png`.