# XVAL-LBM-SINGLE-NODE-MULTI-GPU-HOST-STAGED-HALO

Task anchor `XVAL-LBM-SINGLE-NODE-MULTI-GPU-HOST-STAGED-HALO` (mission m18373, repo `TensorLBM`,
branch `agent/m18373`).  Criterion: cross-validation error < 15 % (owner accuracy standard).

* **Subject (本仓)**: `src/tensorlbm/host_staged_halo.py` (new port of the reference scheme),
  plus the two transports already in the repo — `domain_decomposition.HaloExchange`
  (z-slab, 5 crossing populations) and `multi_gpu.D3Q19GlooTransport` (full 19-plane p2p ring).
* **Reference (OSS)**: ProjectPhysX/FluidX3D, `single-node-multi-GPU` host-staged halo.
  Every reference fact below is quoted with `file:line` from a checkout of the upstream tree
  (`/tmp/fx3d`, files `lbm.cpp`, `kernel.cpp`, `defines.hpp`); nothing is transcribed from prose.
* **Artifacts**: `tools/xval_host_staged_halo/run_xval_host_staged_halo.py` (driver),
  `artifacts/xval_host_staged_halo/{summary.json, table_*.csv, figures/*.png}`,
  tests `tests/test_host_staged_halo.py` (24 tests, all passing).

---

## ① Problem and reference source of record

A single node with several GPUs cannot share device memory portably, so a multi-GPU LBM must
stage the halo through host memory.  FluidX3D's scheme (the reference) is:

| # | Reference fact | Where |
|---|---|---|
| R1 | global grid trimmed to a multiple of the device count, with a warning | `lbm.cpp:722-724` |
| R2 | local extent `N/D + 2*(D>1)` per decomposed axis (one ghost layer each side); device `d = x + (y + z*Dy)*Dx`; domain offset `x*N/D - H` | `lbm.cpp:727-733` |
| R3 | `get_area(direction)` = `{Ny*Nz, Nz*Nx, Nx*Ny}` of the **local** array (halo rows included) | `lbm.cpp:1283-1287` |
| R4 | two host staging buffers `p`/`m` per device, each `Amax x max(transfers*sizeof(fpxx), 17)` bytes, `Amax` = largest face area over decomposed directions | `lbm.cpp:1305-1316` |
| R5 | one selective in-device copy + one device→host copy per side on extract; host→device + selective in-device copy on insert | `lbm.cpp:1343-1353` |
| R6 | per direction: extract **all** devices → `finish_queue` barrier → **host pointer swaps** → insert **all** devices; directions serialized x → y → z | `lbm.cpp:1355-1382` |
| R7 | exactly **5 crossing distributions per face and side** for D3Q19 (`xp=(1,7,13,9,15)`, `xm=(2,8,14,10,16)`, `yp=(3,7,14,11,17)`, `ym=(4,8,13,12,18)`, `zp=(5,9,16,11,18)`, `zm=(6,10,15,12,17)`) | `kernel.cpp:904-931` |
| R8 | extract from layers `1` / `N-2`, insert into layers `0` / `N-1` | `kernel.cpp:2091-2137` |
| R9 | neighbour is the modulo-wrapped device index (periodic exchange layer) | `lbm.cpp:1360/1369/1378` |

The repo had no host-staged transport at all: `HaloExchange` keeps its staging buffers on the
compute device and exchanges via `dist.batch_isend_irecv`/`copy_`, and `D3Q19GlooTransport` ships
all 19 populations of the boundary plane.  So the reference scheme was **ported** (`host_staged_halo.py`)
before it could be measured.  R7 is verified programmatically against the reference's *own*
velocity set (`kernel.cpp:871-877`): the reference index table maps exactly onto the geometric set
`{q : c_axis = ±1}` (test `test_reference_transfer_table_is_the_crossing_set`).

## ② Method and implementation

Ported semantics (all of R1–R9 preserved):

* `HostStagedLayout` — trimming (R1), local/owned extents and ghost offsets (R2), face areas of the
  local array (R3), periodic device neighbours (R9).
* `HostStagedHalo` — one staging buffer pair per device sized for `Amax` (R4); extract/insert on the
  layers of R8 with the 5-population sets of R7; per-direction `extract → barrier → swap → insert`
  schedule in x,y,z order (R6); two staging modes: `swap` (reference, zero host copy) and `copy`
  (naive host memcpy, the controlled comparison arm).  Copy volume is **measured** at the copy sites
  (bytes counted from the tensors actually touched), not estimated.
* `HostStagedD3Q19` — periodic D3Q19 BGK solver over the device grid (collide owned cells → halo
  exchange of post-collision populations → pull streaming into owned cells), used to test
  decomposition invariance and physics.

Three evidence arms, thresholds declared before the runs (in the driver docstring):

* **D1 invariance** — `Dx*Dy*Dz` field vs single-device field: rel-L2 < 1e-10 (float64), < 1e-4 (float32).
* **D2 accounting** — measured staging bytes vs the reference formula
  `devices x 2 sides x (read+write) x area x 5 x 4 B` per decomposed direction: error < 15 %.
* **D3 physics** — Taylor–Green decay rate vs analytic `4*nu*k^2`: error < 15 %.
* **D4 staging mode** — swap must move 0 B host-to-host and must not change the solution.

Deduplication (开工三查): `knowledge_base`/`negative_knowledge_ledger` hold the sibling anchors
`XVAL-LBM-CARTESIAN-MPI` (pylbm MPI topology) and `XVAL-LBM-ASYNC-COPY-WAITANY-OVERLAP` (TCLB
async copy / `MPI_Waitany`); neither covers host-staged pointer-swap staging on a single node, and
`oss_solutions` has no FluidX3D row for this feature — so this anchor is not a re-run of an existing
one.  `docs/ai4s-roadmap.md` already lists “selective halo (FluidX3D × XLB)” as a wanted capability.

## ③ Parameter table (lattice units, dimensionless)

| Quantity | Value |
|---|---|
| lattice | D3Q19, BGK, standard store, FP32 for transport / FP64 for physics arms |
| grids | 24³, 32³, 40³ (physics, invariance), 32³–128×128×64 (transport accounting) |
| device grids | 1×1×1, 2×1×1, 1×1×2, 1×2×1, 2×2×2, 1×1×4, 4×2×1, 1×2×4, 2×2×4 |
| initial fields | ABC: `u = u0(sin y cos z, sin z cos x, sin x cos y)`; TG (2-D embedded in 3-D) |
| `u0` / `Ma = u0*sqrt(3)` | 0.01/0.0173, 0.05/0.0866, 0.10/0.1732 |
| `nu` / `tau = 3nu+0.5` | 0.01/0.53, 0.02/0.56, 0.05/0.65, 0.10/0.80 |
| `k` | `2*pi/n`, n = 24/32/40 → 0.2618/0.1963/0.1571 |
| steps | 120 (invariance), 160 (sensitivity), 200 (TG), 200 (transport: 1 exchange) |
| staging | host tensors, `pin_memory` when a CUDA device exists |
| per-face payload | 5 distributions = 20 B (FP32) |

## ④ Comparison table (ref vs got)

**D2 — staging volume per exchange (measured vs reference formula), `table_transport.csv`**

| grid | devices | ref bytes | ours bytes | rel err | verdict | copy-arm host bytes | repo `HaloExchange` | repo `GlooTransport` |
|---|---|---|---|---|---|---|---|---|
| 32³ | 1 (1×1×1) | 0 | 0 | 0 | ✔ | 0 | 40 960 | 155 648 |
| 32³ | 2 (2×1×1) | 163 840 | 163 840 | 0.000 | ✔ | 81 920 | 81 920 | 155 648 |
| 32³ | 2 (1×1×2) | 163 840 | 163 840 | 0.000 | ✔ | 81 920 | 81 920 | 155 648 |
| 32³ | 8 (2×2×2) | 983 040 | 983 040 | 0.000 | ✔ | 491 520 | 81 920 | 155 648 |
| 64³ | 4 (4×1×1) | 2 621 440 | 2 621 440 | 0.000 | ✔ | 1 310 720 | 1 310 720 | 1 572 864 |
| 64³ | 8 (2×2×2) | 7 864 320 | 7 864 320 | 0.000 | ✔ | 3 932 160 | 1 310 720 | 1 572 864 |
| 128×64×64 | 4 (1×1×4) | 2 621 440 | 2 621 440 | 0.000 | ✔ | 1 310 720 | 1 310 720 | 1 572 864 |

Staging allocation also matches R4 exactly (`table_layout.csv`: `ref_buffer_bytes` =
`2*devices*Amax*max(5*4,17)`), e.g. 32³ 2×1×1 → 81 920 B for both sides of both devices.

**D1 — decomposition invariance (ABC flow), `table_invariance.csv`**

| dtype | D | rel L2 vs single device | K rel diff | verdict |
|---|---|---|---|---|
| float64 | 1×1×2, 1×2×1, 2×1×1, 1×1×4, 1×2×4, 4×2×1 | 0 … 4.3e-15 | < 1e-15 | ✔ (< 1e-10) |
| float64 | 2×2×2 | 4.3e-15 | < 1e-15 | ✔ |
| float32 | same 7 grids | ≤ 3.6e-06 | ≤ 1e-6 | ✔ (< 1e-4) |

The invariance is exact for **every** grid shape (corners included): the sequential x→y→z passes
fill the *edge/corner* ghost columns with the neighbour's already-updated ghost data, which is why a
triple-diagonal population is correct even though D3Q19 has no `(1,1,1)` velocity.

**D3 — Taylor–Green decay vs analytic `4*nu*k^2`, `table_tg_physics.csv` (HONEST FAIL)**

| grid | nu | gamma LBM | gamma analytic | rel err | verdict |
|---|---|---|---|---|---|
| 24³ | 0.02 | 0.001724 | 0.005483 | 68.6 % | ✘ |
| 32³ | 0.02 | 0.001425 | 0.003084 | 53.8 % | ✘ |
| 40³ | 0.02 | 0.002280 | 0.001974 | 15.5 % | ✘ |
| 24³ | 0.05 | 0.002058 | 0.013708 | 85.0 % | ✘ |
| 32³ | 0.05 | 0.001653 | 0.007711 | 78.6 % | ✘ |
| 40³ | 0.05 | 0.002417 | 0.004935 | 51.0 % | ✘ |

Every decomposition row is bit-identical to its single-device row, so the failure is **not** in the
transport: it belongs to the minimal solver's absolute dissipation (see ⑤/⑥).

**D4 — staging mode**, `table_timing.csv`:

| grid / D | swap host copy | copy-arm host copy | swap median time | copy median time |
|---|---|---|---|---|
| 64³ / 2×2×2 | 0 B | 1.11 MB | 12.3 ms | 14.7 ms |
| 64³ / 1×1×4 | 0 B | 655 kB | 2.25 ms | 2.83 ms |
| 128×64×64 / 2×1×1 | 0 B | 655 kB | 2.13 ms | 2.39 ms |

The pointer swap removes 100 % of the host-to-host copy volume and 1.1–1.20x of the exchange wall
time on this (CPU-only) host — the reference's motivation, reproduced.  The swap and copy arms give
bit-identical solutions (rel L2 0 … 4.3e-15).

## ⑤ Convergence / parameter sensitivity

* **Decomposition count and shape** (`table_invariance.csv`): 1 → 8 devices, slab (1-D), 2-D and 3-D
  device grids, `1×1×4` and `4×2×1` anisotropic shapes: invariance error stays at the round-off
  floor (float64 4.3e-15, float32 3.6e-6).  No degradation with more faces to exchange, i.e. the
  corner/edge ordering claim is supported.
* **Lattice Mach number and collision rate** (`table_sensitivity.csv`): `u0 ∈ {0.01,0.05,0.10}`
  (`Ma` 0.017–0.173) × `nu ∈ {0.01,0.02,0.05,0.10}` (`tau` 0.53–0.80) on the fixed 2×1×2 grid vs
  single device: worst-case invariance error 7.1e-14, i.e. the scheme is insensitive to both the
  compressibility level and the collision rate; density stays within the compressibility level
  (max `|rho-1|` in the table).
* **Grid** (D3 table): as expected, the analytic-vs-LBM gap does **not** converge with grid; it is
  dominated by a grid-independent term (see ⑥) — a further indication that the missing physics is in
  the solver operator, not in the transport.

## ⑥ Failure modes, limitations, artifacts

**Honest FAIL (D3).** The absolute TG decay of `HostStagedD3Q19` misses the analytic rate by
15–85 % and, unlike a viscous decay, does not scale with `nu` (measured `gamma` stays at
0.0017–0.0024 per step for `nu` = 0.01…0.10 and for all grids).  Recorded in
`artifacts/xval_host_staged_halo/summary.json → honest_fail`, with the follow-up path: diff
collide+stream of the port against the repo's collision kernels on an identical state, and check the
one-mode shear eigenvalue `1 - omega*(1-cos k)` against the measured decay.  Consequence for use:
the transport (`HostStagedHalo`) is validated and reusable; the container solver
(`HostStagedD3Q19`) must **not** be used as a physics solver until that audit is done.

**Environment limitation.** No CUDA device on this node (`torch.cuda.is_available() == False`), so
pinned host memory and PCIe copies are not exercised; the D2 comparison is a byte-accounting
comparison (measured tensor volumes) and the D4 timing is host-CPU memcpy, not PCIe.  On a
multi-GPU host the same driver runs the same numbers with `pin_memory=True` and real
`cudaMemcpyAsync`; only the timing column would change.

**Port scope (things deliberately not reproduced).** A-A pair storage and the esoteric-pull
`t`-parity split streaming (`kernel.cpp:2108-2123`) are storage optimisations; the staged payload is
5 slots per face cell either way, so the volume cross-validation is unaffected.  The 17-byte
`rho_u_flags` packet (R4's floor) is only needed with free surface / force field extensions and is
not carried by this port.  This reference version's `communicate_field` has no chunked transfer, so
no chunking was ported.

**Diff list: TensorLBM vs FluidX3D (做法 / 参数 / 边界条件) and port trade-offs**

| dimension | FluidX3D (ref) | this port | repo `HaloExchange` | repo `D3Q19GlooTransport` |
|---|---|---|---|---|
| decomposition | 3-D `Dx*Dy*Dz`, trim to multiples (R1,R2) | same | z-slab only, requires exact divisibility, raises `ValueError` | z-slab, ring of 2 ranks |
| ghost layers | 2 per decomposed axis, layers 0 / N-1 (R2,R8) | same | `(Q, nz+2, ny, nx)` → same geometry on z | pad by concat: 1 ghost plane each side |
| payload per face | 5 crossing distributions (R7) | same (20 B FP32) | 5 on z only | **19** full plane (3.8x more bytes) |
| staging memory | pinned host, `Amax x max(5*4,17)` per buffer, 2/device (R4) | host (pinned on CUDA), `5 x Amax` per buffer, 2/device | device tensors, `5 x ny x nx` | device tensors, `19 x ny x nx` |
| host-to-host copy | **0** (pointer swap, R6) | 0 (measured), object swap | 1 recv copy per side | n/a (direct p2p) |
| schedule | per direction extract→barrier→swap→insert, x→y→z (R6) | same | one z exchange, `batch_isend_irecv` + `waitall` | `isend/irecv` + `waitall`, no overlap |
| periodic wrap | modulo device index (R9) | same | modulo | modulo |
| precision | FP32 / FP16 with A-A pairs | FP32 / FP64 standard store | FP32 (configurable dtype) | FP32 |
| in-domain BCs | solids, free surface, forces, temperature | none (periodic only) | none | none |

Trade-offs taken: the reference's *schedule and staging semantics* are ported verbatim because they
are what decides the communicated volume, the barrier count and the exactness on domain edges; the
device-side kernels become tensor views (portable, CPU-validatable, and still one device-side
selective copy + one staging copy per side, which is what the accounting counts); the A-A storage
trick and the 17-B auxiliary packet are dropped because they change neither the volume model nor the
correctness claim.  The port is strictly additive — `HaloExchange`/`D3Q19GlooTransport` are
untouched, so no existing path changes behaviour.

**Artifacts**

```
artifacts/xval_host_staged_halo/summary.json        # checks, worst cases, honest_fail, environment
artifacts/xval_host_staged_halo/table_layout.csv    # trimming / extents / Amax / buffer bytes vs R1-R4
artifacts/xval_host_staged_halo/table_transport.csv # D2 accounting + repo transports + copy-arm volume
artifacts/xval_host_staged_halo/table_invariance.csv# D1, 7 decompositions x float32/float64 x 2 staging modes
artifacts/xval_host_staged_halo/table_tg_physics.csv# D3 (honest FAIL)
artifacts/xval_host_staged_halo/table_sensitivity.csv# Ma x nu sweep on a 2x1x2 grid
artifacts/xval_host_staged_halo/table_timing.csv    # D4 swap vs memcpy exchange time
artifacts/xval_host_staged_halo/figures/fig1_staging_volume.png
artifacts/xval_host_staged_halo/figures/fig2_invariance.png
artifacts/xval_host_staged_halo/figures/fig3_tg_error.png
artifacts/xval_host_staged_halo/figures/fig4_timing.png
```

Reproduce: `PYTHONPATH=src python tools/xval_host_staged_halo/run_xval_host_staged_halo.py`
(~7 min, 8 threads, CPU only); tests: `python -m pytest tests/test_host_staged_halo.py -q` (24 pass).

**生态摩擦 (friction log)**: (1) the OSS inventory (`oss_solutions`) has no FluidX3D row for this
feature, so the reference had to be re-read from source; a `high-performance-gpu-lbm` family row for
FluidX3D multi-GPU transport would save the next agent the checkout.  (2) `import tensorlbm` in this
container resolves to another worktree (`/root/tp-agent/work/m17918/...`), so every run needs an
explicit `PYTHONPATH=src`; a repo-local `.pth`/editable install per worktree would remove a recurring
foot-gun.  (3) The shell tool kills the whole process group at its timeout, so long runs must be
started with `setsid nohup … &`.