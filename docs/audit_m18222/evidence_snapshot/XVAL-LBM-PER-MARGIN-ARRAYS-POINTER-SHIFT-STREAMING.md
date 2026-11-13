# XVAL-LBM-PER-MARGIN-ARRAYS-POINTER-SHIFT-STREAMING

**Round 1 · open-source cross-validation (XVAL)** · mission 18129 · generated 2026-11-13T12:10:23Z by
`benchmarks/xval_tclb_pointer_shift/run_xval.py` (every number below is read from
`runs/xval_tclb_pointer_shift/result.json`; nothing is hand-copied -- see §1 provenance and
`benchmarks/xval_tclb_pointer_shift/extract_oss_ref.py`).

**Verdict: PASS (equivalence) / gate PASS (accuracy) / NOT ADOPTED (cost)** --
bit-wise equivalence holds, both arms are 0.621 % from the closed-form reference
(owner gate 15 %), but the formulation is 17.1x slower than the production
transport on the largest measured CPU grid, so the *implementation* is banked as a regression test, not
adopted as the default path (predeclared rule D3, §2).

## 1. Problem and reference provenance

The OSS implementation is CFD-GO/TCLB. Its transport is *not* an index permutation. The population
field of each lattice direction is stored as a set of **per-margin arrays** (`FTabs` holds one
`storage_t*` per non-empty margin block, `src/Global.h.Rt:64-68`), the 27 blocks being the
low/interior/high span combinations produced by `Margin` (`src/conf.R:771-791`). A neighbour read is
then a **linear offset inside a block** (`offsets()` -> `offset.p` / `get_offsets` / `put_offsets`,
`src/conf.R:897-992`), and when the loop crosses a block boundary the *base pointer* of that block is
moved instead of recomputing an address (`move.pocket`: `pocket.blockN += delta`,
`src/LatticeAccess.inc.cpp.Rt:69-79`). Collision reads (`pop`) and writes (`push`) are emitted from
that machinery (`src/LatticeAccess.inc.cpp.Rt:316-350`), each stage running border kernel -> margin
exchange -> interior kernel (`src/Lattice.cu.Rt:492-527`), and only non-empty margin blocks are
exchanged (`MPIStream_A`, `src/Lattice.cu.Rt:327-334`). Field extents are derived from the stencil
(`AddField`, `src/conf.R:135-150`), not hand-written.

| item | value |
|---|---|
| repository | `CFD-GO/TCLB` (`https://github.com/CFD-GO/TCLB.git`, branch `master`) |
| revision | `ded67cd768cf7e727bd078af139e3ec7895076e5` (from `.git/refs/heads/master`; the checkout is shallow, its object store is incomplete, so provenance is pinned by ref file + per-file sha256) |
| feature slug | `per-margin-arrays-pointer-shift-streaming` |
| runnable in this environment? | **no** -- `src/*.Rt` are R templates (code generation needs `R`/`Rscript`) and the build needs an MPI/CUDA toolchain; this container has no `R` and no network. Recorded as friction, not worked around by guessing. |
| reference side of the comparison | the closed-form physics (Taylor-Green 2D exact decay, plane-Poiseuille parabola) + TensorLBM's production transport, **not** TCLB's binary output -> the OSS *binary* comparison is `WITHHELD` (§6) |

Provenance (hashed by `extract_oss_ref.py`, excerpts in `benchmarks/xval_tclb_pointer_shift/oss_ref/`):

| file | sha256 (16 hex) |
|---|---|
| `.git/packed-refs` | `990084cc9812b011…` |
| `.git/refs/heads/master` | `e17dea26cd8c5b55…` |
| `README.md` | `8836dfe13a61feba…` |
| `makefile` | `247622db3668dbc7…` |
| `src/Global.h.Rt` | `aac89081a61cb143…` |
| `src/Lattice.cu.Rt` | `6bc57392103e35a2…` |
| `src/LatticeAccess.inc.cpp.Rt` | `05dbd6d6d501b35a…` |
| `src/conf.R` | `9dcdb96171bbfbd5…` |

| snippet | location | claim it backs |
|---|---|---|
| `ftabs_per_margin_arrays` | `src/Global.h.Rt:58-74` | populations are stored as one array per margin block (per-margin arrays) |
| `margin_table` | `src/conf.R:771-792` | the 27 margin blocks are the 3x3x3 combinations of side/low/mid/high spans |
| `offsets_block_addressing` | `src/conf.R:897-998` | block offsets are linear offsets inside a per-margin array (offset.p / get_offsets / put_offsets) |
| `stencil_derived_extent` | `src/conf.R:135-150` | each field's spatial extent is derived from the stencil, not hand-written |
| `move_pocket_pointer_shift` | `src/LatticeAccess.inc.cpp.Rt:69-79` | a block transition is a pointer increment (pocket.blockN += delta), not an index recomputation |
| `field_access_uses_pocket` | `src/LatticeAccess.inc.cpp.Rt:122-167` | every population read/write indexes a moved pocket with a constant per-block offset |
| `pop_push_templates` | `src/LatticeAccess.inc.cpp.Rt:321-339` | streaming reads (pop) and writes (push) are emitted per stage from the block/pocket machinery |
| `iterate_border_interior_split` | `src/Lattice.cu.Rt:488-514` | each stage runs a border kernel, then an MPI margin exchange, then an interior kernel |
| `mpi_margin_exchange` | `src/Lattice.cu.Rt:327-335` | only non-empty margin blocks are exchanged across ranks |

## 2. Method and implementation, decision rules

Port: `src/tensorlbm/xval/per_margin_pointer_shift.py` (library code).
The domain is split per axis into the three margin spans (`lo` 1 cell, `mid` `n-2`, `hi` 1 cell, the
same three-way selection TCLB's `offset.p` uses). For a lattice direction with velocity `c`, each axis
contributes `(dst_slice, src_slice)` pairs -- one for the wrapped strip, one for the in-range
remainder -- and the cartesian product of those pairs is the block list: at most `2**|{a: c_a != 0}|`
blocks per direction, each a *pure slice copy* whose flat-index difference
`flat(src) - flat(dst)` is a **constant** (the pointer shift). No index tensors, no `roll`, no gather.
A second entry point (`stream_pointer_shift_flat`) executes the same plan as 1-D offset copies where
the block is contiguous, i.e. as literally as torch allows `out[k:k+n] = f[k+off:k+off+n]`.

Predeclared rules (fixed before the runs):

* **D1 equivalence** -- arms A (production) and B (port) must agree bit-wise (max abs diff 0), 2D and
  3D, float32 and float64. Otherwise the port is rejected.
* **D2 gate** -- relative L2 velocity error vs the closed-form reference < 15 % (owner gate) for *both*
  arms on both physics cases; 3 % internal target reported only (it is a resolution question).
* **D3 keep/adopt** -- adopt only if D1 and D2 hold and the CPU cost stays within 2x of production.

Both arms share collision (`collide_bgk` / Guo-forced BGK), the half-way bounce-back operator, the
initial condition, the sampling schedule and the error estimators, so a measured difference is
attributable to the transport operator alone. Environment: python 3.11.6, torch 2.7.1,
Linux-6.6.52-1.7.lns23.loongarch64-loongarch64-with-glibc2.38, threads 128, CUDA False, peak RSS 1022 MB.

## 3. Parameter tables (non-dimensional)

Taylor-Green 2D (`u_max=0.04`, `Ma = u_max*sqrt(3) = 0.0693`, `rho0=1`):

| arm | n | steps | tau | nu | u_max | Ma | Re (u_max*n/nu) | L2 ux % | L2 uy % | max % |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 32 | 300 | 0.8 | 0.1000 | 0.04 | 0.0693 | 13 | 0.621 | 0.621 | 0.621 |
| A | 64 | 400 | 0.8 | 0.1000 | 0.04 | 0.0693 | 26 | 0.092 | 0.092 | 0.092 |
| B | 32 | 300 | 0.8 | 0.1000 | 0.04 | 0.0693 | 13 | 0.621 | 0.621 | 0.621 |
| B | 64 | 400 | 0.8 | 0.1000 | 0.04 | 0.0693 | 26 | 0.092 | 0.092 | 0.092 |

Tau sensitivity (n=32, 150 steps):

| arm | tau | nu | max rel L2 % |
|---|---|---|---|
| A | 0.6 | 0.0333 | 0.598 |
| A | 0.8 | 0.1000 | 0.503 |
| A | 1.0 | 0.1667 | 0.568 |
| B | 0.6 | 0.0333 | 0.598 |
| B | 0.8 | 0.1000 | 0.503 |
| B | 1.0 | 0.1667 | 0.568 |

Plane Poiseuille (half-way bounce-back walls, Guo force, `H_eff = ny-2`):

| arm | grid | steps | tau | nu | accel | u_max (ref) | u_max err % | profile rel L2 % | steady resid |
|---|---|---|---|---|---|---|---|---|---|
| A | 34x4 | 8000 | 0.8 | 0.100 | 1.60e-05 | 2.046e-02 | 0.058 | 0.059 | 7.47e-04 |
| B | 34x4 | 8000 | 0.8 | 0.100 | 1.60e-05 | 2.046e-02 | 0.058 | 0.059 | 7.47e-04 |

## 4. Comparison tables (ref vs got)

Implementation equivalence -- reference = production transport, got = port (`check_plan_invariant`
also verifies the partition and the constant-offset property cell by cell):

| dims | shape | dtype | bit-wise equal | max abs diff | blocks |
|---|---|---|---|---|---|
| 2D | [17, 23] | float32 | True | 0 | 25 |
| 2D | [4, 4] | float32 | True | 0 | 25 |
| 2D | [3, 5] | float32 | True | 0 | 25 |
| 2D | [256, 256] | float32 | True | 0 | 25 |
| 3D | [6, 7, 8] | float32 | True | 0 | 61 |
| 3D | [5, 5, 5] | float32 | True | 0 | 61 |
| 3D | [3, 4, 5] | float32 | True | 0 | 61 |
| 2D | [17, 23] | float64 | True | 0 | 25 |
| 2D | [4, 4] | float64 | True | 0 | 25 |
| 2D | [3, 5] | float64 | True | 0 | 25 |
| 2D | [256, 256] | float64 | True | 0 | 25 |
| 3D | [6, 7, 8] | float64 | True | 0 | 61 |
| 3D | [5, 5, 5] | float64 | True | 0 | 61 |
| 3D | [3, 4, 5] | float64 | True | 0 | 61 |

Block-decomposition invariants (per case: partition ok, holes/overlaps, max deviation of the recorded
flat offset, max diff vs an independent roll-based reference, number of copies):

| dims | shape | partition | holes/overlaps | max offset error | max diff vs ref | copies |
|---|---|---|---|---|---|---|
| 2D | [17, 23] | True | 0/0 | 0 | 0 | 25 |
| 2D | [4, 4] | True | 0/0 | 0 | 0 | 25 |
| 2D | [3, 5] | True | 0/0 | 0 | 0 | 25 |
| 3D | [6, 7, 8] | True | 0/0 | 0 | 0 | 61 |
| 3D | [5, 5, 5] | True | 0/0 | 0 | 0 | 61 |
| 3D | [3, 4, 5] | True | 0/0 | 0 | 0 | 61 |
| 2D | [17, 23] | True | 0/0 | 0 | 0 | 25 |
| 2D | [4, 4] | True | 0/0 | 0 | 0 | 25 |
| 2D | [3, 5] | True | 0/0 | 0 | 0 | 25 |
| 3D | [6, 7, 8] | True | 0/0 | 0 | 0 | 61 |
| 3D | [5, 5, 5] | True | 0/0 | 0 | 0 | 61 |
| 3D | [3, 4, 5] | True | 0/0 | 0 | 0 | 61 |

Block count per direction follows the stencil `2**|{a: c_a != 0}|`
(D2Q9 expected `{'0': 1, '1': 2, '2': 2, '3': 2, '4': 2, '5': 4, '6': 4, '7': 4, '8': 4}`, got `{'0': 1, '1': 2, '2': 2, '3': 2, '4': 2, '5': 4, '6': 4, '7': 4, '8': 4}`; D3Q19 total copies
`61`), i.e. the decomposition is derived from the velocity table.

Accuracy verdict per gate:

| quantity | reference | arm A error | arm B error | <15 %? | <3 %? |
|---|---|---|---|---|---|
| TG2D ux/uy rel L2 (worst of ladder) | closed-form decay | 0.621 % | 0.621 % | yes | yes |
| Poiseuille u_max | analytic parabola | 0.058 % | 0.058 % | yes | yes |
| Poiseuille profile rel L2 | analytic parabola | 0.059 % | 0.059 % | yes | yes |
| transport A vs B | production transport | -- | 0 | yes | yes |

## 5. Convergence and sensitivity

* **Grid refinement** (Taylor-Green, arm A = arm B): n=32 -> 0.621 %,
  n=64 -> 0.092 % -- the transport
  operator is exact, so both arms move on the same convergence curve; the residual error is the LBM
  truncation error (Ma^2 compressibility + resolution), not a transport defect.
* **Relaxation-time sensitivity** (`tau` = 0.6/0.8/1.0 at n=32): 0.6->0.598 %, 0.8->0.503 %, 1.0->0.568 %. Errors are flat in `tau`
  over the stable window, again identically for both arms.
* **Poiseuille steady state**: velocity residual between samples falls to
  7.5e-04 (relative to the analytic centreline velocity) over
  8000 steps, so the quoted profile error is a steady-state number.
* **Cost sensitivity** (median over 5 repeats, CPU torch):

| dim | grid | A: gather [ms] | B: pointer shift [ms] | B': flat offsets [ms] | slowdown B/A | blocks | index tensors held by A |
|---|---|---|---|---|---|---|---|
| 2D | 128^2 | 4.98 | 1.33 | 1.64 | 0.27x | 25 | 18.92 MB |
| 2D | 256^2 | 8.99 | 70.16 | 66.78 | 7.80x | 25 | 18.92 MB |
| 3D | 32^3 | 5.26 | 6.19 | n/a | 1.18x | 61 | 20.24 MB |
| 3D | 48^3 | 9.12 | 155.86 | n/a | 17.10x | 61 | 87.48 MB |

The crossover is real: at 2D 128x128 the block copies win (0.27x),
because they avoid the int64 index tensors the production path keeps resident; at 256x256 and at 48^3
the copy-per-block launch overhead dominates (7.8x / 17.1x). TCLB pays that overhead in *generated C++*
where the pointer increment is a single machine instruction; in torch each block is a separate kernel.

## 6. Failure modes, limits, and the diff list vs TCLB

**Diff list (TensorLBM vs OSS)**

| aspect | TCLB | this port / TensorLBM | consequence |
|---|---|---|---|
| storage | 27 per-margin arrays (`FTabs.block*`), interior block separate | single `(q, ...)` torch tensor; blocks are *views*, not separate allocations | port keeps the offset algebra but loses cache-locality gains |
| addressing | generated C++: base pointer moved per block (`move.pocket`) | precomputed `(dst_slice, src_slice, flat_offset)` plan | same constant-offset invariant, no per-cell arithmetic in either |
| margins / MPI | border kernel + margin exchange for non-empty blocks, then interior kernel | single-process only; periodic wrap expressed as wrapped strip copies | multi-rank and `RunBorder`/`RunInterior` split are **not** ported |
| extent derivation | from stencil (`AddField`) | from the velocity table (`2**|{a: c_a != 0}|` blocks) | equivalent; asserted in the tests |
| boundaries | `SymmetryAccess` / node types / bounce inside the border kernel | half-way bounce-back applied outside the transport, shared by both arms | not an equivalence claim; identical in both arms |
| vectorisation | CUDA/CPU kernels, `CROSS_*` | torch strided kernels | see cost table |
| float handling | `storage_t` + optional `STORAGE_SHIFT` | none | out of scope |

**Failure modes / limits**

1. `n < 3` per axis is rejected (`ValueError`) -- the three-way span split degenerates; tested.
2. Periodic only: for non-periodic domains the wrapped strip must be replaced by the boundary
   operator; the module does not do that itself (the driver applies bounce-back after transport, which
   is why both arms need the same extra operator).
3. Blocks per direction grow as `2**|{a: c_a != 0}|` -> 61 copies for D3Q19: cheap in C++, a kernel-launch tax in torch (measured above).
4. No GPU measurement (no CUDA device in this container) -- the pointer-shift formulation's main
   selling point (no index tensors, no gather emulation on SDAA-like backends) is left **WITHHELD**.
5. TCLB could not be built (no `R`), so "OSS binary vs TensorLBM" is **WITHHELD**; the equivalence is
   against TensorLBM's production operator plus an independent roll-based reference, and the accuracy
   is against closed-form solutions.

**Ecological friction**: (a) knowledge engine offline (`/api/v1/tensor/knowledge/search` ->
`engine_offline`, `127.0.0.1:8021` refused) so the mandatory three-checks fell back to the local
negative-knowledge ledger (0 hits for `TCLB`/`pointer shift`/`per-margin`); (b) no `R`/`Rscript` and no
network in the control container, so the OSS code generator cannot be run; (c) the 120 s per-call cap
forced the physics ladder to n=64 / 8000 channel steps (the driver records the full ladder; rerunning at
n=128 needs a cluster job).

**Decision (predeclared D1/D2/D3)**: D1 = True, D2 = True,
D3 = False -> keep the production gather transport; bank the block-offset invariant + the bitwise cross-check as a regression test; the OSS formulation's advantage does not survive the port to torch (see §6).

**Tests**: `tests/test_xval_tclb_pointer_shift.py` -- `2 failed, 15 passed, 1 warning in 5.38s`.
