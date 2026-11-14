# GAP-DIFFLBM-SOLID-LIMIT — volumetric-solid limit convergence and hard-wall drag consistency

**Status: PARTIAL / honest-pending.** The closure work, the unit-level invariants and the
campaign harness are delivered and green; the 14-shard acceptance campaign is **submitted and
running** (`cj2488`, scan over `arm x resolution x mask scale`). Only **1 of 14** shards had
finished when this report was written, so the two ledger criteria (**D1** mask-scale invariance
`<= 5 %`, **D2** hard-wall deviation `<= 10 %`) are **NOT MEASURED** yet — they are not claimed
as passed. `fail_stage = cluster_queue_waiting` (see §6).

---

## ① Problem setting and benchmark provenance

A soft/volume representation of a solid body (the only differentiable option for a DiffLBM
pipeline: a binary mask has zero gradient everywhere) carries the body as a solid **volume
fraction** `m(x) in [0,1]`. The closure that turns `m` into a wall rule decides whether the model
has a *solid limit*:

| symbol | meaning |
|---|---|
| `s in (0,1]` | **mask scale** — how sharp the volume representation is; `s = 1` is the solid limit |
| `sigma = (1-s) * sigma_max` | volume-preserving Gaussian blur of the fraction field, `sigma_max = 1` cell |
| `phi = (m - 1/2)/|grad m|` | reconstructed interface distance (positive inside the solid) |
| `q in (0,1]` | sub-grid wall distance along a lattice link (Bouzidi's parameter) |

The mask scale is defined so that **the geometry does not change** with `s`: the blur is volume
preserving, so the solid volume and (to `O(sigma^2/R)`) the interface position are identical at
every scale. Drag that moves with `s` is therefore a *numerical* artefact of the volume
representation, not physics — which is exactly what the ledger criterion D1 measures.

**Case**: Schäfer & Turek (1996) benchmark **2D-1**, an in-channel cylinder, `Re = u_mean D/nu = 20`,
steady, blockage `D/H = 1/4.1`. Reference drag `Cd = 5.57953523384` — provenance chain is the
official spectral reference dataset recorded (sha-pinned) in-repo at
`benchmarks/verified/cylinder/re20_st/result.json`, consumed through
`tensorlbm.cylinder_st2d1.REFERENCE`. The solver chain (Zou/He parabolic inlet, Zou/He pressure
outlet, MRT collision, Ladd momentum exchange) is reused verbatim from
`tensorlbm.cylinder_st2d1`; **only the cylinder wall rule differs between arms**, so the
comparison is controlled.

Literature anchors for the closures: Walsh, Burwinkle & Saar (2009) partial bounce-back for
partially filled cells; Bouzidi, Firdaouss & Lallemand (2001, *Phys. Fluids* 13:3452) interpolated
bounce-back; Parker & Youngs (1992) / Scardovelli & Zaleski (2000, *Annu. Rev. Fluid Mech.*
31:567) for the `(m-1/2)/|grad m|` interface reconstruction; Ladd (1994) for momentum exchange.

## ② Method and implementation

Three arms, identical in everything except the cylinder closure (`src/tensorlbm/volume_solid.py`):

* **`hardwall`** — repository half-way bounce-back on the binary mask `m >= 0.5`
  (`tensorlbm.boundaries.bounce_back_cells`) + Ladd momentum exchange
  (`compute_obstacle_forces`). The hard-wall reference of criterion D2.
* **`amplitude`** — `partial_bounce_back`: per node reflection coefficient
  `theta = clip(2m, 0, 1)`, opposite population pairs mixed with weight `theta` (Walsh et al.
  2009). This is the "penalty proportional to solid fraction" closure a soft-mask solver gives
  for free; its reflectivity follows `m`, so **smoothing the mask lowers the reflection in the
  interface band**.
* **`interface`** — `interface_bounce_back`: the *same* field read as **geometry**. The interface is
  reconstructed (`signed_distance_from_fraction`) and no-slip is imposed on it with Bouzidi
  interpolated bounce-back written on the fluid side (post-stream array, `f_post` from the same
  step): `q >= 1/2 : f_j = (1/2q) f_i^post + (1-1/2q) f_j^post`; `q < 1/2 : f_j = 2q f_i^post + (1-2q) f_i^post(x-c_i)`.

Drag for **all** arms is measured by one estimator so the arms are comparable: the discrete
control-volume momentum budget `force = net momentum imported across the box faces - change of
fluid momentum inside` (`run_case.momentum_balance_force_2d`, the 2D counterpart of
`tensorlbm.control_volume_force`, whose box helper needs `nz >= 3`). The pair-form momentum
exchange (`partial_bounce_back_force`) is recorded beside it as an independent cross-check.

Key code pointers: `src/tensorlbm/volume_solid.py` (`vof_circle`, `blur_mask`,
`mask_scale_to_sigma`, `signed_distance_from_fraction`, `partial_bounce_back`,
`partial_bounce_back_force`, `interface_bounce_back`), `cases/GAP-DIFFLBM-SOLID-LIMIT/run_case.py`
(arms, control volume, `evaluate`), `campaign.py` (sharded matrix), `summarize.py` (verdict +
figures), `tests/test_volume_solid.py`.

**Unit-level evidence (all green, `pytest tests/test_volume_solid.py` → 9 passed)**

| invariant | result |
|---|---|
| `amplitude` with a binary mask == `bounce_back_cells` | bit-identical (`torch.equal`), also after blur+threshold |
| `interface` on a **plane** wall (all `q = 1/2`) == hand-written fluid-side half-way rule | bit-identical |
| `partial_bounce_back_force` == `compute_obstacle_forces` on the solid ring | `< 1e-12` absolute |
| mass conservation of the amplitude closure; momentum lost == force given to the solid | `< 1e-10` |
| VOF solid volume vs `pi R^2` | 314.27 vs 314.16 (0.03 %) |
| blur conserves solid volume | rel. `< 1e-6` |
| `phi` error vs the analytic disc distance (band `|d| < 1`) | median **0.038** cell, max 0.60 cell |
| `phi` error at mask scale 1.0 / 0.75 / 0.5 | median **0.038 / 0.038 / 0.033** cell (blur-invariant) |

## ③ Parameter table (all non-dimensional numbers)

| parameter | value | note |
|---|---|---|
| Re | 20 | `u_mean D / nu` |
| u_mean (lattice) | 0.1 | fixed across resolutions (`--u-lb 0.1`), U_max = 1.5 u_mean |
| nu (lattice) | 0.1 (D=20) / 0.16 (D=32) | `nu = u_mean D / Re` |
| tau (MRT) | 0.8 (D=20) / 0.98 (D=32) | `nu = (tau-1/2)/3` |
| Ma | 0.173 | `u_mean / c_s`, `c_s = 1/sqrt(3)` |
| blockage D/H | 1/4.1 | Schäfer–Turek 2D-1 |
| resolutions D | 20, 32 cells | 2-point resolution ladder |
| mask scales s | 1.0, 0.75, 0.5 | `sigma = 0, 0.25, 0.5` cell |
| VOF supersampling | 16 x 16 per cell | volume error 0.03 % |
| control volume | `cx +- 12`, `cy +- 12` cells | strictly interior, encloses the cylinder |
| steps | up to 50 000, avg window 2 000 | plateau tolerance 1e-3 |

## ④ Comparison table (ref vs got, zero hand-copy — generated from the run JSONs)

Measured so far (1 of 14 shards, `artifacts/GAP-DIFFLBM-SOLID-LIMIT/runs/amplitude_D20_s1.json`):

| quantity | ref | got | error | verdict |
|---|---|---|---|---|
| Cd, `amplitude`, D=20, s=1.0 (control volume) | 5.579535 | **6.33239** | **+13.49 %** | ✗ (>15 % not reached but >5 %) |
| Cd, same run, pair-form momentum exchange | – | 6.33240 | estimator agreement **4e-8 rel.** | ✓ estimator validated |
| run steadiness (first vs second half of the window) | – | 6.33234 vs 6.33245 | 1.6e-3 % | ✓ steady |
| **D1** `|Cd(s=0.5) − Cd(s=1.0)|/Cd <= 5 %` | – | – | – | **NOT MEASURED** (shards pending) |
| **D2** `|Cd(arm) − Cd(hardwall)|/Cd <= 10 %` | – | – | – | **NOT MEASURED** (shards pending) |

The +13.5 % of the `amplitude` arm at `s = 1` is **not attributed** in this report: the hard-wall
arm at the same resolution is still running, so we cannot yet separate (a) partial reflection on
fluid-side cells with `0 < m < 0.5` (`theta = 2m > 0`) — i.e. the body is effectively thickened by
the closure — from (b) the D=20 resolution itself. The candidate (a) is the mechanism the
`interface` arm is designed to remove and is the first thing the finished campaign will test.

## ⑤ Convergence / sensitivity evidence

Available now (unit level, no solver): interface reconstruction error is **flat across mask scale**
(median 0.038 / 0.038 / 0.033 cell for `s = 1.0 / 0.75 / 0.5`) while the amplitude closure's
reflectivity in the interface band changes by a factor up to 2 at the same scales — the mechanism
the fix targets. Solver-level sensitivity (2 resolutions x 3 mask scales x 3 arms) is exactly what
`campaign.py` enumerates; the figures `figures/cd_vs_mask_scale.png` and `figures/criteria.png` are
produced by `summarize.py` **as soon as the shards land** (they are not fabricated here).

## ⑥ Failure modes, artefacts, reproduction

* **Where this round stopped**: computation, not method. `cj2488` (14 shards, `stage_dir=` this
  worktree) was queued 27th behind 43 in-flight jobs; at 43 min it had completed 1 shard
  (871 s/run at D=20 under a 2-node queue). `FAIL_STAGE = cluster_queue_waiting`.
* **Bug found and fixed during the round** (affects only wall time, not validity):
  `_block_drift` sliced the last block as `series[-size:0]` (empty), so the plateau detector never
  fired and every run went to `max_steps = 50 000` — 3-4x the necessary wall time. Fixed to an
  absolute-index slice (`tests`-level check: flat series → drift 0.0, ramp → 0.43). The delivered
  record was produced with 50 000 steps and is steady to 1.6e-3 %, so the number stands.
* **Known limitations**: (i) a one-step lag between the fluid-side interpolated bounce-back and the
  repository's solid-node bounce-back (standard; irrelevant at steady state); (ii) the `interface`
  arm resets solid interior cells to rest equilibrium each step, so it must not be used for
  unsteady pressure histories *inside* the body (it is a no-slip closure, not a solid-state model);
  (iii) D2 uses the `hardwall` binary-mask arm as denominator: it is a *staircase* body, so part of
  `D2` is the staircase error of the reference arm, not an error of the volumetric closure.
* **Artefacts**: `artifacts/GAP-DIFFLBM-SOLID-LIMIT/runs/*.json` (one per combo),
  `results.json` + `table.csv` (from `summarize.py`), `figures/solid_representation.png`,
  `figures/cd_vs_mask_scale.png`, `figures/criteria.png`.
* **repro_cmd**: `bash cases/GAP-DIFFLBM-SOLID-LIMIT/repro.sh --full`
  (sharded form: `PYTHONPATH=src python cases/GAP-DIFFLBM-SOLID-LIMIT/campaign.py --index {i}
  --out artifacts/GAP-DIFFLBM-SOLID-LIMIT/runs --resolutions 20 32 --scales 1.0 0.75 0.5
  --max-steps 50000 --min-steps 8000 --avg-window 2000 --plateau-tol 1e-3 --u-lb 0.1`, `{i}` in 0..13).
* **Round-2 input**: finish the 13 remaining shards (same command), then `summarize.py` prints the
  D1/D2 table; if `amplitude` fails D1 and `interface` passes both, the ledger defect is closed. If
  `interface` also fails D2, the next lever is D2's denominator (staircase) — compare against a
  same-geometry `interface`-with-binary mask instead of the staircase arm.
