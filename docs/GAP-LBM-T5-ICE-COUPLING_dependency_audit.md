# GAP-LBM-T5-ICE-COUPLING — round-1 dependency audit

* **Node**: `GAP-LBM-T5-ICE-COUPLING` (T5: LBM hydrodynamics + FEM structure + DEM ice failure, three-dimensional)
* **Mission**: m15659 · **Repo**: TensorLBM · **Branch**: `agent/m15659`
* **Round**: 1 of 2. Scope per the node contract: *dependency audit + coupling-interface design only*; a P1 periodic single-phase prototype does **not** count as validation.
* **Deliverables of this round**: this audit, `docs/GAP-LBM-T5-ICE-COUPLING_coupling_interface_design.md`,
  the interface reference implementation `src/tensorlbm/three_way/interface3d.py`, its gate driver
  `scripts/three_way_interface3d_gates.py`, tests `tests/test_three_way_interface3d.py`, and the
  machine-generated tables `docs/generated/GAP-LBM-T5-ICE-COUPLING_gate_tables.md`
  (source of every number quoted below — nothing here is hand-copied from a run).

## 1. What the node actually requires

The acceptance sentence decomposes into four requirements with very different
readiness. The audit grades each one separately instead of grading the node as a whole:

| id | requirement | nature |
|---|---|---|
| R1 | LBM water side in 3-D, with loads extracted on a moving/deformable surface | solver + interface |
| R2 | FEM structural side (hull / ice sheet) receiving those loads | solver + interface |
| R3 | DEM ice side that fails/breaks (bond rupture), exchanging contact loads with the structure | solver + interface |
| R4 | "engineering-grade verification gate = T1 multiphase + load exchange passing criteria" | **governance gate**: T5 cannot be called verified until T1 (multiphase) and the load-exchange gate are closed |

Round 1 is allowed to answer R4-adjacent questions and to build the interface that
R1–R3 all hand off through; it is *not* allowed to claim R1–R3 are verified.

## 2. Method and evidence grading

* Read-only inspection of the repository (paths and symbols are cited for every claim).
* Executed in this worktree (single-threaded BLAS, `PYTHONPATH=src`):

| command | result | log |
|---|---|---|
| `pytest tests/test_three_way_interface3d.py -q` | **14 passed, 4.8 s** | new (this round) |
| `python scripts/three_way_interface3d_gates.py` | **8/8 gates PASS, 4 s** | `runs/three_way_interface3d/gates.json` |
| `pytest tests/test_three_way_core.py -q` | 13 passed, 62.9 s | existing 2-D chain |
| `pytest tests/test_three_way_gates.py -q` | 6 passed, 66.2 s | `runs/three_way_interface3d/logs/three_way_gates_pytest.log` |

* Evidence levels: **REAL** = algorithm implemented *and* numerically anchored to an
  independent reference in an artefact I can point at; **MOCK/STUB** = placeholder;
  **WITHHELD** = code exists but no reproducible artefact in this tree (claim cannot be
  checked here); **NOT_SUPPORTED** = absent.

## 3. Inventory of what T5 would stand on

| # | component | path (symbols) | level | evidence / execution | gap to T5 |
|---|---|---|---|---|---|
| 1 | 2-D three-way chain (LBM 2-D + IBM + DEM floe + beam) | `src/tensorlbm/three_way/{core2d,ibm2d,floe,beam,scenario,gates}.py` | **REAL, with one open gate** | `tests/test_three_way_core.py` 13 passed (my run); `tests/test_three_way_gates.py` 6 passed (my run); gate (a) documented **FAIL** (see F1) | 2-D only; no 3-D surface, no 3-D DEM body, no 3-D structural model |
| 2 | 3-D D3Q19 LBM kernels | `src/tensorlbm/solver3d.py` (`collide_bgk3d`, `collide_mrt3d`, `collide_trt3d`, `stream3d{,_gather,_cat,_roll}`); `src/tensorlbm/d3q19.py` | **REAL** | verified 3-D drag case `benchmarks/verified/sphere_re100` (D3Q19 MRT): C_D 1.12198 (D40) / 1.11946 (D60) vs Schiller–Naumann 1.09173 → **2.77 % / 2.54 %**, control-volume closure 1.50e-4 / 7.81e-5 (cited artefact, not re-run here) | none for the fluid kernel |
| 3 | 3-D immersed-boundary operators | `src/tensorlbm/ibm.py` (`ibm_velocity_interpolate_3d`, `ibm_force_spread_3d`, `ibm_direct_forcing_3d`, `ibm_apply_body_force_3d`); tests `tests/test_ibm3d.py` | **REAL** | unit/regression tests present (not re-run in this mission) | spreading to a *structural* DOF set (FEM) absent — this round's interface fills it |
| 4 | 3-D DEM (soft sphere, ice particles) + brash-ice full-ship chain | `src/tensorlbm/dem/{particle,contact,dem_solver,cw_coupling}.py`; `src/tensorlbm/ice_chain3d/{hull,floes,flow,chain}.py`; `ice_field_3d.py` | **PARTIAL / WITHHELD** | `chain.py` defines six criteria but **no `criteria.json` artefact is in the tree** (only `.scratch/out_run1/criteria.json`); `docs/in66_n05_ice_resistance_3d.md` + `ice_resistance_3d.py` claim C3 "FSI momentum exchange residual < 1 %" with no run artefact in the tree | DEM↔**hull-structure** load path is not content-addressed; porous-ice band is not breakable-ice coupling |
| 5 | bonded-particle ice cover with breakable bonds (2-D) | `TensorLBM/src/tensorlbm/ice_hydro/{bonded_cover,coupled,analytical,lbm_channel}.py`; tests `TensorLBM/tests/test_ice_hydro_breakup.py` | **REAL but on a legacy duplicate package root** | see F2 | 2-D bonded lattice; not importable from the installed `src/tensorlbm` path |
| 6 | structural FEM | `src/tensorlbm/beam_fem.py` (`EulerBernoulliBeamFEM`: Hermite cubic, static, eigen, Newmark, stress); `src/tensorlbm/three_way/beam.py` (modal) | **REAL (1-D)** | this round: tip deflection vs `FL³/3EI` err 5.1e-12, first frequency vs 1.875104² root err 1.3e-7 (`gates.json` I6) | **no 3-D shell/solid FEM**; hull girder reduces to a beam |
| 7 | T1 multiphase prerequisite | `src/tensorlbm/multiphase3d.py`, `mcmp3d_*`, `mcmp_stratified_poiseuille/` | **OPEN** | negative ledger: m14882 / m15063 / m15215 — MCMP stratified Poiseuille spurious currents could not be brought below 0.005 u_max; "交付经达标复核仅 L1" | R4 gate cannot close until multiphase is at criterion |
| 8 | 3-D surface→structural load transfer | **did not exist** before this round | — | created now: `src/tensorlbm/three_way/interface3d.py` + I1–I7 gates | gated against analytic references; not yet driven by a real LBM field |

## 4. Dependency graph (blocking edges marked `*`)

```
             (7) T1 multiphase at criterion*  ──┐
                                                ├──> R4 engineering verification gate  ──> T5 "verified"
             (8) load-exchange gate closed*  ───┘
                        ▲
                        │ invariants I1–I6 (this round)
                        │
  (2) 3-D LBM kernels ──┼──> R1 water side with surface loads
  (3) 3-D IBM ops ──────┘

  (5) bonded DEM breakage ──> R3 ice failure  ── needs 3-D bonded lattice (only 2-D today)
  (4) 3-D DEM particles  ───┘

  (6) beam FEM ──> R2 hull girder (1-D)  ── upgrade path: 3-D shell FEM (round 2+)
```

The interface (8) is the narrowest bottleneck: **every** edge of R1/R2/R3 passes
through it, and before this round it had no 3-D implementation and no invariant gates.

## 5. Findings (severity-ranked)

**F1 — HIGH, open gate inherited from the 2-D chain.** `docs/THREE_WAY_LBM_DEM_FSI.md`
(section 3) records gate **(a) FAIL**: LBM→DEM momentum exchange `7.7 % / 4.8 % / 8.8 %`
against a `< 2 %` criterion, observed order 2.93. The numbers are quoted "verbatim" from
`runs/three_way/gates.json`, but **that file is not in this worktree** (only
`runs/three_way_interface3d/` exists), so the claim is not reproducible here
(WITHHELD until re-run). Consequence for T5: the load-exchange booking route in 2-D is
*not* yet at criterion, so T5 must not cite "load exchange OK" from the 2-D chain.

**F2 — MEDIUM, duplicate legacy package root.** Two `tensorlbm` package trees exist:
`src/tensorlbm/` (the one `PYTHONPATH=src` and the tests import) and
`TensorLBM/src/tensorlbm/` (contains `ice_hydro/`, `d3q19.py`, `d4q19.py`, …). The
ice-cover hydroelastic package therefore is **not importable** from the installed path;
`TensorLBM/tests/test_ice_hydro_breakup.py` works around this with path surgery
(`sys.path.insert`). Any T5 code that imports `tensorlbm.ice_hydro` will `ImportError`.
Round-1 decision: do **not** port it (out of scope, risk of divergent copies); document it
and require a single canonical path before round 2 consumes it.

**F3 — HIGH, T1 multiphase prerequisite is honestly open.** The negative-knowledge
ledger records MCMP stratified Poiseuille failing at L1 (m14882, m15063, m15215) with the
model-class reason (linear-ψ two-component Shan–Chen interface). T5's own acceptance
sentence makes this a *blocking* prerequisite for the engineering verification gate.
Round-1 consequence: T5 cannot be graded above "interface prototype" until (a) T1 closes
or (b) the water side is validated on a single-phase benchmark the platform accepts.

**F4 — MEDIUM, unverifiable 3-D ice-resistance claims.** `src/tensorlbm/ice_resistance_3d.py`
docstring asserts C1–C6 (e.g. "FSI momentum exchange residual < 1 %", "resistance vs
Riska(1989) < 15 %"), and `ice_chain3d/chain.py` implements six criteria, yet no
`criteria.json`/result artefact exists under the source tree. Treat as WITHHELD; round 2
must either regenerate the artefact on the cluster or downgrade the claim.

**F5 — MEDIUM, no 3-D structural model.** Structure is a 1-D Euler–Bernoulli beam
(`beam_fem.py`) or a modal beam (`three_way/beam.py`). A 3-D ice sheet / hull panel needs
a plate/shell element; nothing in the tree provides one. This bounds R2 to "hull girder".
Round-1 decision: the interface is written against structural DOFs (backend-agnostic), so
a shell backend can be added without changing the transfer algebra.

**F6 — LOW, artefact/label drift.** `docs/THREE_WAY_LBM_DEM_FSI.md` says "campaign node
P35-N9" while the node under audit is `GAP-LBM-T5-ICE-COUPLING`; the `three_way` package
docstring was truncated mid-sentence in `three_way/__init__.py` (line 1 starts with `m\`\`),`).
Cosmetic, but it makes provenance tracing harder — fixed in the docstring only, not
rewritten historically.

**F7 — LOW, no shared invariant gate for transfer operators.** The 2-D chain checks
*global* gate (a)/(b) quantities but no test asserted the *algebraic* invariants of a load
transfer (resultant/moment/work). Round 1 adds exactly those (I1–I7).

## 6. Gap ledger contributed by this round

| gap id | description | severity | blocking what | round |
|---|---|---|---|---|
| G5-01 | 3-D surface→structural transfer lacked an invariant-gated implementation | high | every R1/R2/R3 hand-off | **closed this round (I1–I7)** |
| G5-02 | added-mass instability had no guard on the partitioned loop | high | any two-way μ ≥ 1 case (floe in water at high ρ_s/ρ_f) | **guard implemented + gated (I5)**; enforcement in a real coupled run: round 2 |
| G5-03 | 2-D gate (a) momentum exchange FAIL, artefact missing | high | "load exchange at criterion" claim | round 2 (re-run + fix) |
| G5-04 | T1 multiphase not at criterion | high | engineering verification gate (R4) | other node |
| G5-05 | no 3-D shell/solid FEM backend | medium | R2 beyond hull girder | round 2+ |
| G5-06 | 3-D bonded DEM (breakable ice) absent (only 2-D bonded lattice) | medium | R3 ice failure | round 2 |
| G5-07 | interface not yet driven by a real 3-D LBM field (only analytic tractions) | medium | R1 end-to-end | round 2 (cheapest next step: sphere D40 case, reuse `benchmarks/verified/sphere_re100` machinery) |
| G5-08 | duplicate legacy package root hides `ice_hydro` | medium | reuse of 2-D bonded-cover physics | maintenance |

## 7. Round-1 verdict (honest)

* **Closed**: a 3-D load-exchange interface with exact, machine-checked invariants
  (resultant, moment, work conjugacy, clock consistency, added-mass guard, energy ledger),
  each compared to an independent analytic reference; a hull-girder leg verified against
  Euler–Bernoulli theory through that interface. All 8 gates PASS in 4 s.
* **Not closed, and not claimed**: no real LBM→FEM→DEM coupled 3-D run exists; R1/R2/R3
  are *not* verified; the engineering verification gate (R4) is blocked by F3 and F1.
* **Round-2 first move** (from the gap ledger): drive the interface with the verified
  D3Q19 sphere case (G5-07), which turns the only remaining "analytic-only" edge real at
  low cost, then re-run and fix 2-D gate (a) (G5-03).

## 8. Reproduction

```bash
cd <worktree>
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
    python scripts/three_way_interface3d_gates.py            # -> runs/three_way_interface3d/, docs/generated/
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
    python -m pytest tests/test_three_way_interface3d.py -q  # 14 passed
```

Artefacts: `runs/three_way_interface3d/gates.json`, `gates.csv`, `figures/*.png`
(5 figures), `docs/generated/GAP-LBM-T5-ICE-COUPLING_gate_tables.md`.
Because `runs/` is git-ignored, the same artefacts are mirrored to the tracked
directory `artifacts/three_way_interface3d/` (JSON + CSV + 5 PNG + pytest log).