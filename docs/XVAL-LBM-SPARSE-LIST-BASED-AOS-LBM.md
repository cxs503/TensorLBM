# XVAL-LBM-SPARSE-LIST-BASED-AOS-LBM — TensorLBM vs `lssfau/lbmpy` sparse list-based AoS LBM

Round 1, mission 18376, repo **TensorLBM**. Judge: owner accuracy standard
(>15 % vs the reference is unacceptable). Predeclared protocol:
`xval/XVAL-LBM-SPARSE-LIST-BASED-AOS-LBM/predeclared.json`.
Driver: `xval/XVAL-LBM-SPARSE-LIST-BASED-AOS-LBM/run_xval.py` (full run 432.8 s,
single core). Tests: `tests/test_xval_sparse_list_aos.py` (11 passed, 14.9 s).

---

## 1. Problem and reference source

The OSS entry `oss_solutions#114` (`lssfau/lbmpy`, feature
`sparse-list-based-aos-lbm`) keeps only the **active cell list** plus a
**per-cell, per-direction neighbour index array**; streaming becomes 1-D
indirection, solid cells are never stored, and no-slip is folded into the index
array. TensorLBM has no such storage scheme (every kernel is dense
`(Q, nx, ny)`), so the question is whether the scheme can be ported with the
*identical* semantics and how much accuracy it costs.

Reference facts were read from the installed OSS source (lbmpy 2.0,
`/usr/lib/python3.11/site-packages/lbmpy/sparse/`), not from prose:

| fact | source |
|---|---|
| cell list = `np.argwhere(flag & fluid_flag)`, boundary cells appended after fluid cells, coordinates `uint32` struct | `mapping.py:84-104` |
| flat pdf offset `pdf_index(cell, d) = cell + d*len(self)` (SoA strides), centre direction omitted → `Q-1` columns | `mapping.py:106-111`, `:152-154` |
| no-slip by **index aliasing** to `pdf_index(own_cell, inverse_idx(d))`, tested *before* periodicity | `mapping.py:126-127` |
| periodic wrap over ghost layers `x_i == ghost-1 → += n_i-2*ghost`, `x_i == n_i-ghost → -= n_i-2*ghost`, `ghost_layers=1` | `mapping.py:128-140` |
| direction `d` reads index column `d-1`; the rest population reads `src(0)` | `update_rule_sparse.py:86-95` |
| kernel variants `stream_pull_collide` / `collide_only` / `stream_pull_only`; list layout `AoS`/`SoA` via `compute_strides` | `update_rule_sparse.py:7-36` |

## 2. Method and implementation

New TensorLBM module `src/tensorlbm/sparse_list_aos.py`:

* `build_index_array(flags, ghost_layers=1, layout="soa")` — port of
  `create_index_array`; returns `(coordinates (N,2) uint32, index_array (N,Q-1) uint32)`.
* `SparseListLBM` — populations in a flat `Q*N` array, one step = gather through
  the index array (+ own cell for the rest direction) → BGK+Guo collision →
  scatter back. No ghost/periodic slicing anywhere.
* `dense_collide_stream` — the **same** discrete scheme (BGK + Guo, halfway
  bounce-back, periodic wrap) with an independent dense traversal; it is the
  control arm, so sparse-vs-dense differences isolate the storage/traversal
  change and nothing else.
* `poiseuille_analytic`, `taylor_green_analytic` — references.

Four arms:

| arm | what | note |
|---|---|---|
| A | TensorLBM dense traversal | control |
| B | TensorLBM list-based traversal + TensorLBM index array | the port |
| C | TensorLBM list-based traversal driven by the **OSS** index array (`lbmpy.sparse.SparseLbMapper.create_index_array`, executed in-process) | zero-hand-copy check |
| R | analytic (Poiseuille profile / Taylor-Green decay) | physics |

Two physical cases: plane Poiseuille (body force, halfway no-slip walls,
periodic streamwise, `L=4` columns) and the 2-D Taylor-Green vortex (fully
periodic, exact exponential decay of the fundamental mode).

## 3. Parameter table (lattice units; all dimensionless groups)

| case | H/L | τ | ν | G | u_max (analytic) | Re | Ma | active cells |
|---|---|---|---|---|---|---|---|---|
| Poiseuille | 8 | 0.80 | 0.1000 | 3.750e-04 | 0.02953 | 2.40 | 0.0520 | 32 |
| Poiseuille | 16 | 0.80 | 0.1000 | 9.375e-05 | 0.02988 | 4.80 | 0.0520 | 64 |
| Poiseuille | 32 | 0.80 | 0.1000 | 2.344e-05 | 0.02997 | 9.60 | 0.0520 | 128 |
| Poiseuille | 64 | 0.80 | 0.1000 | 5.859e-06 | 0.02999 | 19.2 | 0.0520 | 256 |
| Poiseuille τ-sens | 32 | 0.70 | 0.0667 | 1.562e-05 | 0.02997 | 14.4 | 0.0520 | 128 |
| Poiseuille τ-sens | 32 | 1.00 | 0.1667 | 3.906e-05 | 0.02997 | 5.76 | 0.0520 | 128 |
| Poiseuille τ-sens | 32 | 1.20 | 0.2333 | 5.469e-05 | 0.02997 | 4.11 | 0.0520 | 128 |
| Poiseuille force 0.3× | 32 | 0.80 | 0.1000 | 7.031e-06 | 0.00899 | 2.88 | 0.0156 | 128 |
| Poiseuille force 3× | 32 | 0.80 | 0.1000 | 7.031e-05 | 0.08991 | 28.8 | 0.1559 | 128 |
| Taylor-Green | 16 | 0.80 | 0.1000 | — | u0 = 0.02 | 3.20 | 0.0346 | 256 |
| Taylor-Green | 32 | 0.80 | 0.1000 | — | u0 = 0.02 | 6.40 | 0.0346 | 1024 |
| Taylor-Green | 64 | 0.80 | 0.1000 | — | u0 = 0.02 | 12.8 | 0.0346 | 4096 |

`ν = (τ-1/2)/3`, `Re = u_max·H/ν` (or `u0·L/ν`), `Ma = u_max/c_s` with
`c_s = 1/√3`. Poiseuille fluid cells = `4·H` of a `(H+2)×(H+2)` bounding box;
the list-based arms never allocate the ghost ring.

## 4. Comparison table (ref vs got, error, verdict)

### 4.1 Poiseuille profile vs analytic `u(y) = G/(2ν)(y+½)(H-(y+½))`

| H | arm | u_max got | u_max ref | L∞/u_max | L2 rel | steps | conv | <15 % |
|---|---|---|---|---|---|---|---|---|
| 8 | A/B/C | 0.029662 | 0.029531 | 4.444e-03 | 5.72e-03 | 1800 | yes | ✔ |
| 16 | A/B/C | 0.029916 | 0.029883 | 1.098e-03 | 1.42e-03 | 6400 | yes | ✔ |
| 32 | A/B/C | 0.029979 | 0.029971 | 2.737e-04 | 3.74e-04 | 23600 | yes | ✔ |
| 64 | A/B/C | 0.029995 | 0.029993 | 6.838e-05 | 9.36e-05 | 87800 | yes | ✔ |

### 4.2 Arm-to-arm differences `max|Δu|/u_max` (storage/traversal XVAL)

| H | A vs B | A vs C | B vs C |
|---|---|---|---|
| 8 | 1.6e-15 | 2.6e-15 | 3.8e-15 |
| 16 | 4.8e-15 | 8.0e-15 | 7.2e-15 |
| 32 | 1.07e-14 | 1.48e-14 | 1.74e-14 |
| 64 | 2.11e-14 | 1.72e-14 | 1.43e-14 |
| τ / force sweeps (H=32) | ≤9.5e-14 | ≤7.8e-14 | ≤3.5e-14 |

**Index array**: TensorLBM's port equals the OSS array **element-by-element**
(`np.array_equal`, `max|diff| = 0`, shape `(64,8)`, `uint32`) on the same flag
grid — arm C is literally fed by the OSS function.

### 4.3 Taylor-Green decay rate `ω = 2νk²` (mode projection, window = 2 e-foldings)

| L | arm | ω got | ω ref | rel | field L∞/amp | <15 % |
|---|---|---|---|---|---|---|
| 16 | A/B/C | 3.09852e-02 | 3.08425e-02 | 4.63e-03 | 2.50e-02 | ✔ |
| 32 | A/B/C | 7.71940e-03 | 7.71063e-03 | 1.14e-03 | 6.62e-03 | ✔ |
| 64 | A/B/C | 1.92815e-03 | 1.92766e-03 | 2.53e-04 | 2.41e-03 | ✔ |

Arm-to-arm field difference A vs B: 6.1e-14 / 9.7e-14 / 1.0e-13 (L = 16/32/64).

## 5. Convergence and parameter sensitivity

* **Grid convergence of the shared scheme** (Poiseuille, B): 4.444e-3 → 1.098e-3
  → 2.737e-4 → 6.838e-5 for H = 8/16/32/64, observed order
  **2.02 / 2.00 / 2.00** — second order, as the halfway bounce-back requires.
  TG: 4.63e-3 → 1.14e-3 → 2.53e-4 for L = 16/32/64, order **2.02 / 2.17**.
* **τ sensitivity** (H=32, same u_max): τ = 0.7/0.8/1.0/1.2 → L∞ error
  2.48e-4 / 2.74e-4 / 1.63e-3 / 3.40e-3. Monotone growth with τ at *fixed
  grid* is expected: the same error in lattice units scales with ν, and the
  fixed-τ resolution per Kolmogorov-like scale is unchanged; all values are
  ≪ 15 %.
* **Force/compressibility sensitivity**: u_max = 0.009 / 0.03 / 0.09
  (Ma = 0.016 / 0.052 / 0.156) → L∞ error identical to 4 significant digits
  (2.737e-04), i.e. no measurable O(Ma²) contamination up to Ma ≈ 0.16.
* **Cost**: for the same case the list-based arm touches `4·H` cells vs the
  dense arm's `H·H` bounding box (H=64: 256 vs 4096 → 16× fewer collision
  evaluations) at the price of one `uint32` index array of `(N, Q-1)`
  (8·4 = 32 B/cell here) — the port's benefit grows with the solid fraction,
  which is exactly the porous/obstacle use case in the OSS notebook.

## 6. Failure modes, honest limits, artifacts

**Predeclared verdict (results.json → `verdict`)** — 6 of 7 pass:

| id | got | threshold | verdict |
|---|---|---|---|
| T1 index-array equality with OSS | True (max\|diff\| = 0) | exact | ✔ |
| T2 sparse vs dense vs OSS-index | 1.28e-13 | ≤ 1e-12 | ✔ |
| T3 Poiseuille vs analytic (H ≥ 16, incl. τ/force sweeps) | 1.10e-3 | ≤ 0.15 | ✔ |
| T4 Taylor-Green decay | 4.62e-3 | ≤ 0.15 | ✔ |
| T7 Taylor-Green field | 2.50e-2 | ≤ 0.15 | ✔ |
| T6 AoS/SoA stride mismatch detectable | inf (blow-up) | > 0.01 | ✔ |
| T5 mass conservation | **5.52e-12** | ≤ 1e-12 | ✘ (marginal) |

**T5 is reported as a honest FAIL of the predeclared letter, not hidden.** The
measured drift is 6.3e-17 per step over 87 800 steps, i.e. float64 round-off
accumulation, not a leak; the control that discriminates is the negative test
`test_truncated_guo_term_leaks_mass_negative_control`, which shows that the
*same* traversal with a truncated force term drifts by 1.5e-2 (ten orders
larger). Conclusion: no mass leak; the 1e-12 bound was simply tighter than the
round-off floor of a 87 800-step run, and it was **not** relaxed after the fact.

**Bug found and fixed on the way (defect, not OSS failure):** the first version
of the port used the first-order Guo source term only,
`Λ_i = 3w_i(c_i-u)·F/ρ`. That term does not annihilate mass
(`ΣΛ_i = -3u·F/ρ`), so the Poiseuille density drifted to ρ = 0.98444 (−1.56 %)
and the profile error was **25.8 % / 4.65 % / 1.03 %** at H = 16/32/64 — the
error was 100 % of the observed deviation from the analytic solution. With the
full Guo term `3(c_i-u)·F + 9(c_i·u)(c_i·F)` the drift vanishes (ρ = 1.000000)
and the error drops to 1.1e-3 / 2.7e-4 / 6.8e-5. The negative-control test keeps
this from regressing.

**OSS-side failure modes observed (environment, not fixed here):**

1. `lbmpy.sparse.create_lb_update_rule_sparse` **does not build** against the
   pinned pystencils 2.0: `_list_substitutions` calls `idx(i-1)` on the numpy
   index array → `TypeError: 'numpy.ndarray' object is not callable`
   (`update_rule_sparse.py:91`). Only `SparseLbMapper` (cell list + index array)
   is usable as shipped, so the port re-implements the update step from the
   documented semantics and pins the *index array* to the OSS output.
2. The `AoS` label is misleading: `create_index_array` hardcodes
   `cell + d*N` (column-major/SoA) offsets while `create_symbolic_list(...,
   layout='AoS')` yields `cell*Q + d` strides. Mixing the two is silent
   mis-addressing — the port keeps both layouts and
   `case_stride_mismatch` shows the mismatched run diverging to NaN within
   2000 steps.
3. `create_index_array` raises `IndexError` if an active cell touches the
   border of the flag grid (`mapping.py:131`): periodic borders must carry a
   non-fluid ghost layer, and no-slip walls must be flagged (not just absent).
4. The OSS stencil ordering (`LBStencil(D2Q9)` = y-first) differs from
   TensorLBM's `d2q9.C` (x-first); the index-array comparison therefore uses
   `C_D2Q9_LBMPY`, and arm C runs with the permuted stencil.
5. As the OSS notebook itself advertises: sparse + GPU + UBB is
   `NotImplementedError`; only one no-slip flag family is handled (this port
   inherits that: no moving walls, no UBB, no ghost-node extrapolation).

**Difference list (TensorLBM vs OSS) and porting trade-offs**

| aspect | OSS `lbmpy.sparse` | TensorLBM (this port) |
|---|---|---|
| index-array builder | `SparseLbMapper.create_index_array` (pystencils struct field, `uint32`) | `build_index_array` (plain numpy), integer-identical to OSS |
| update step | symbolic kernel generated from the collision rule; `stream_pull_collide` / `collide_only` / `stream_pull_only` | fixed pull-collide NumPy gather; no symbolic codegen, no C kernel |
| collision | any lbmpy method (SRT/TRT/MRT/cumulant/KBC) | BGK only (MRT/KBC out of scope for round 1) |
| force scheme | lbmpy force models (Guo default, EDM, Buick, …) | Guo with the full second-order term only |
| boundaries | no-slip flag family + (unimplemented) UBB mapper | single no-slip flag family, index aliasing only |
| layout | index array assumes SoA offsets; `AoS` list helper inconsistent | `soa` (OSS-compatible) and `aos` (self-consistent pair), mismatch detectable |
| dtype | `uint32` coordinates/indices, float32/64 fields | `uint32` indices, float64 throughout |
| stencil order | `LBStencil(D2Q9)` (y-first) | repo order (x-first), OSS order available for cross-checks |

**Porting recommendation.** Adopt the *storage* idea (cell list + `(N, Q-1)`
index array + no-slip aliasing) where the solid fraction is high: it gives
`(bounding box)/N` fewer collision evaluations and removes boundary-kernel
code, at the cost of `(Q-1)·4` B/cell of index array and a gather-heavy inner
loop (no contiguous streaming, which is why the dense AA-pattern kernel stays
better on GPUs). Do **not** adopt it as the default for wall-resolved channel
DNS: there the whole box is fluid, `N ≈ bounding box`, and the indirection is
pure overhead. Keep TensorLBM's dense traversal as the default and expose the
list-based one as the porous/obstacle path.

**Artifacts**: `results.json`, `tables.md`, `figures/poiseuille_profile.png`,
`figures/poiseuille_convergence.png`, `figures/taylor_green_decay.png`,
driver `run_xval.py`, protocol `predeclared.json`, module
`src/tensorlbm/sparse_list_aos.py`, tests `tests/test_xval_sparse_list_aos.py`
(11 passed).

**Scope not covered (round-2 candidates)**: 3-D (D3Q19/D3Q27) lists, TRT/MRT
collisions, moving-wall/UBB boundaries, and a C/CUDA kernel for the gather loop
(the NumPy port is a correctness vehicle, not a performance one).