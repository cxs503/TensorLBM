# Generic-run API — one endpoint through the common modules

Status: implemented (2026-08-20). This page documents the generic-run
fusion step of `PLATFORM_ANALYSIS.md` §4.2, built on the benchmark
compile-route standard (PR #180): a single simulation endpoint whose
execution path runs **exclusively** through the package's common modules
and routes its whole-step chain exactly like the verified benchmark
suite.

* Endpoint family: `POST /api/sim/generic` (+ `GET …/cases`,
  `GET …/{job_id}/status`, `GET …/{job_id}/result`)
* Router: `app/backend/routers/generic_sim.py`
* Engine (case registry + driver): `app/backend/services/generic_run.py`
* Tests: `app/tests/test_generic_sim_api.py`
* The ~20 case-specific endpoints in `app/backend/routers/solver.py`
  and the 3-D STL/parametric `POST /api/simulations/generic-run` are
  **unchanged**; this API is the additive target they converge onto.

## Why

The platform self-diagnosis (`PLATFORM_ANALYSIS.md`) found two gaps:

1. the solver router had grown ~20 case-specific endpoints, each with
   its own parameters and call code, none using the common interface;
2. the "generic-run fusion" was written down but never landed.

The generic-run API closes both: one endpoint, every case, and the case
is *data* — a registry entry composing common-module primitives — not a
branch in platform code.

## Request contract

```json
POST /api/sim/generic
{
  "case": "cavity",                    // registry key (see GET /api/sim/generic/cases)
  "grid": {"nx": 64},                  // case-specific grid params (omitted = default)
  "physics": {"Re": 100.0, "u_lid": 0.06},
  "steps": 0,                          // 0 = case default
  "collision": "auto",                 // auto | bgk | mrt
  "compile_mode": "default",           // eager | default | max-autotune-no-cudagraphs
  "device": "cpu",                     // any torch device string
  "seed": 0,
  "monitor_interval": 0                // 0 = auto
}
```

Response: `{job_id, case, steps, collision, compile_mode, status_url,
result_url}` — the job runs asynchronously through the shared job
manager (same pattern as `/api/benchmarks/*`), with per-step diagnostics
broadcast on the global `/ws` WebSocket. Invalid input is 422 with the
registry reason; cudagraph-class compile modes are rejected with the
shared `compile_utils` structural reason.

Result payload (from `GET /api/sim/generic/{job_id}/result`):
`{case, family, lattice, grid, physics, collision, compile{requested_mode,
canonical_mode, routed, route}, steps, metrics{…per case…}, finite,
elapsed_s, modules_used[…], …}`.

## Case registry → common modules map

| case | BC chain (all common modules) | metrics | common-module entry points |
|------|-------------------------------|---------|---------------------------|
| `cavity` | `collide_mrt/bgk` → pre-half-way-BB (3 walls) → `stream` → `zou_he_moving_lid` | centreline `u_mid/u_bot/v_mid` + `compare_ghia` | `solver.collide_mrt`, `solver.stream`, `d2q9.equilibrium/macroscopic`, `lid_driven_cavity.make_cavity_wall_mask / zou_he_moving_lid / compare_ghia / GHIA_*` |
| `poiseuille` | collide → pre-half-way-BB (walls) → `stream` → Zou-He pressure inlet + `zou_he_outlet_pressure` | profile L2 / u_max error vs analytic parabola | `solver.collide_bgk/mrt`, `solver.stream`, `d2q9.equilibrium/macroscopic`, `boundaries.make_channel_wall_mask / zou_he_outlet_pressure` |
| `couette` | collide → pre-moving-wall-BB → `stream` (periodic x) | profile L2 / top-velocity error vs linear ramp | `solver.collide_bgk/mrt`, `solver.stream`, `d2q9.equilibrium/macroscopic` |
| `shear_wave` | `stream` → `collide_bgk` (fully periodic) | energy-decay error vs exp(−2νk²t) | `solver.collide_bgk`, `solver.stream`, `d2q9.equilibrium/macroscopic` |
| `cylinder` | `collide_mrt(tau_field=sponge)` → `stream` → Ladd forces → `far_field_bc_2d` | Cd mean, Cl amplitude, `detect_strouhal` | `solver.collide_mrt(tau_field=…)`, `solver.stream`, `boundaries.cylinder_mask / far_field_bc_2d / make_sponge_strength / compute_obstacle_forces`, `postprocess.detect_strouhal` |

Compile routing (every case, no exceptions):

```
app/backend/services/generic_run.py
  └─ benchmarks/compile_route.py            (loaded by path — the very
       │                                     adapter every verified
       │                                     benchmark uses, PR #180)
       └─ tensorlbm.compile_utils
            ├─ validate_compile_mode        (cudagraph-class → ValueError)
            └─ compile_step                 (None = byte-identical eager)
```

The driver keeps the step index, monitoring `.item()` syncs and
cancellation checks **outside** the compiled domain (compile_utils
lessons 2/4); only the pure whole-step chain `f -> f'` (plus per-step
force tensors for `cylinder`) is routed.

Three generic BC primitives that the benchmark scripts define inline —
pre-streaming half-way bounce-back, its moving-wall extension, and the
Zou-He pressure inlet (mirror of the common
`zou_he_outlet_pressure`) — live once in `services/generic_run.py` so
platform and benchmarks share semantics; they are parametrised helpers,
not case branches.

## Parity guarantee

`app/tests/test_generic_sim_api.py::test_parity_with_benchmark_cavity*`
runs `benchmarks/verified/cavity/re100/run.py::run_case` and the
endpoint on identical grid/steps/mode and asserts the centreline metrics
agree to machine precision: the generic path composes the *same*
common-module chain in the same order, so the trajectories are
bit-identical on eager and agree to float rounding on compiled routing.
This is the regression gate that keeps the endpoint and the verified
benchmarks from drifting apart.

## Migration order for the case-specific endpoints

`solver.py` endpoints stay untouched until their generic twin exists;
then migrate front-end callers, then deprecate. Suggested order
(cheapest validation first, each step gated by a parity test like the
cavity one):

1. **2-D internal/periodic families** — `lid-driven-cavity`,
   `backward-facing-step`, `turbulent-channel`, `pipeline-flow` →
   registry entries reusing `boundaries.apply_zou_he_channel_boundaries`
   / `make_channel_wall_mask` (near-mechanical: they already call
   `run_lid_driven_cavity` etc., which share these primitives).
2. **2-D external flow** — `cylinder-flow`, `rotating-cylinder`,
   `cumulant-cylinder-flow` → extend the `cylinder` entry with a
   `rotation` physics param (moving-wall injection already exists as a
   generic primitive).
3. **Body-force / scalar families** — `propeller-open-water`,
   `actuator-disk`, `passive-scalar-transport` → registry entries adding
   the common force/scalar kernels as pre-stream steps.
4. **3-D STL/parametric external flow** — fold
   `POST /api/simulations/generic-run` (STL + sphere/cylinder/ellipsoid/
   suboff through the 9 verified 3-D common modules) in as the 3-D
   branch of this same contract: `lattice: d3q19/d3q27`, geometry
   `source: stl | parametric`, force path
   `drag_pressure + drag_friction`, compile routing already shared.
5. **Retire** the migrated endpoints (return a deprecation banner for
   one release, then remove).

Rule for reviewers during migration: a new case must land as a registry
entry that composes existing common-module primitives; if it needs a new
kernel, the kernel belongs in `src/tensorlbm/`, not in the platform
layer.
