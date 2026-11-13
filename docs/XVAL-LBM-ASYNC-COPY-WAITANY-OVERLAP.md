# XVAL-LBM-ASYNC-COPY-WAITANY-OVERLAP — TensorLBM vs CFD-GO/TCLB async-copy / MPI_Waitany overlap

Mission: m17454 · repo `TensorLBM` · branch `agent/m17454`

## ① Problem and reference source of record

`TensorLBM` has no communication/compute overlap: `D3Q19GlooTransport.step`
(`src/tensorlbm/multi_gpu.py`) collides the whole slab, packs, posts both
`isend`/`irecv`, `wait()`s every request, and only then streams. The whole
exchange is exposed between collision and streaming.

Reference implementation (`CFD-GO/TCLB`, inventoried in `oss_solutions` id 58,
`async-copy-waitany-overlap`). Facts read from the source, not transcribed
prose (checkout at `/tmp/TCLB`):

| Reference fact | Where |
|---|---|
| `MPIStream_A()` issues the async device→host copies of the outgoing halo on `outStream` | `src/Lattice.cu.Rt:327-331` |
| `MPIStream_B()` syncs `outStream`, then `MPI_Irecv` **all** buffers, then `MPI_Isend` all buffers | `src/Lattice.cu.Rt:334-374` |
| `CROSS_MPI_WAITANY` switches between `MPI_Waitany` + copy-one-at-a-time and `MPI_Waitall` + copy-all | `src/Lattice.cu.Rt:365-380` |
| `MPI_Waitall` is always applied to `sendreq` only | `src/Lattice.cu.Rt:381` |
| Our scheduled order: `RunBorder` → `MPIStream_A` → `RunInterior` → `MPIStream_B` | `src/Lattice.cu.Rt:496-530` |
| `bufsize[i]`/`bufnumber` are allocated per **actual** neighbour only | `src/Lattice.h.Rt:70-72`, `src/Lattice.cu.Rt:334-340` |

## ② Method and implementation

New package `src/tensorlbm/comm_overlap/`:

* `backend.py` — `TorchDistBackend` (real `torch.distributed` gloo p2p) and
  `VirtualClockBackend` (deterministic latency / bandwidth / pack / compute cost
  model). Both expose the same primitives: `post_recv`, `post_send`, `waitany`,
  `waitall`, `deliver`, `run_compute(kind, units, fn)`, and record a cost
  split (`stall_us`, `compute_us` per kind, ordered event log).
* `waitany_pipeline.py` — `D3Q19OverlapTransport` with three arms:
  * `blocking` = shipped TensorLBM order (control);
  * `waitall` = TCLB with `CROSS_MPI_WAITANY` **off**;
  * `waitany` = TCLB with `CROSS_MPI_WAITANY` **on**.
  Both pipelined arms run exactly the TCLB order: collide the two MPI-facing
  columns (`RunBorder`), pack them, post **all** requests, run the interior
  collide+stream under the exchange (`RunInterior`), then consume one arrival per
  `Waitany` return; `waitall` is applied to the send side only.
* `virtual_case.py` — single-process simulated decomposition (non-uniform slabs),
  ghost payloads supplied from an independently computed ground-truth
  post-collision field, so an assembly error cannot hide inside the timing model.

`scripts/xval_waitany_overlap_m17454.py` reproduces every table and figure below.

## ③ Parameter table

| Quantity | Value |
|---|---|
| Lattice / populations | D3Q19, 19 populations, float64 |
| Global grid (sweeps) | `nx` = 12…64, `nz`×`ny` = 2×3 per rank |
| Decomposition | periodic x-ring, non-uniform slabs (`cuts_for`), 2…4 ranks |
| Reference speedup arms | `blocking` / `waitall` / `waitany`, identical math |
| Virtual clock model | `latency` 1…100 µs, bandwidth 32 B/µs, collide 2 µs/col, stream 2 µs/col, pack 4 µs/plane |
| Halo payload | 19·2·3·8 = 912 B per direction per step |
| Steps per measurement | 3 (last step reported), 4 for correctness |
| Real-comm arm | `torchrun --nproc_per_node=3`, gloo, slabs 3/3/4 of nx=10 |

No dimensionless groups apply (the study is a schedule study; the flow physics is
deliberately a pointwise fixture — see §⑤).

## ④ Reference vs measured

Correctness first: **all arms are bit-identical** to the monolithic serial step and,
on the real 3-rank gloo group, to the shipped `D3Q19GlooTransport`
(`mismatch = 0`, `max_abs = 0.0`, world sizes 2/3/4, non-uniform cuts).

Latency sweep (3 ranks, cuts 6/5/5, per-rank mean):

| latency (µs) | blocking stall (µs) | waitany stall (µs) | analytic bound max(0,T_comm−T_int) | error | step speedup |
|---|---|---|---|---|---|
| 1 | 29.5 | 22.8 | 22.8 | 0.00 % ✅ | 1.293 |
| 2 | 30.5 | 23.8 | 23.8 | 0.00 % ✅ | 1.287 |
| 5 | 33.5 | 26.8 | 26.8 | 0.00 % ✅ | 1.269 |
| 10 | 38.5 | 31.8 | 31.8 | 0.00 % ✅ | 1.245 |
| 20 | 48.5 | 41.8 | 41.8 | 0.00 % ✅ | 1.207 |
| 50 | 78.5 | 71.8 | 71.8 | 0.00 % ✅ | 1.141 |
| 100 | 128.5 | 121.8 | 121.8 | 0.00 % ✅ | 1.092 |

`blocking` stall equals the full exposed transfer `T_comm` in every row (error
vs its own bound 0.00 %) — i.e. the shipped path hides nothing.

`MPI_Waitany` gain over `MPI_Waitall` (arrival jitter between the two directions,
uniform 4/4/4 slabs, per-buffer consume 2 µs):

| arrival spread (µs) | 0 | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|---|
| measured gain (µs/step) | 0.00 | 1.00 | 2.00 | 2.00 | 2.00 | 2.00 | 2.00 |
| `min(spread, consume)` | 0.00 | 1.00 | 2.00 | 2.00 | 2.00 | 2.00 | 2.00 |

Error 0.00 % at every point: the switch buys **min(spread, per-buffer consume)**
and nothing more, and saturates as soon as the spread exceeds the per-buffer
consumption cost. Gate `|error| < 15 %` → **PASS** (measured 0.00 %).

Grid / comm-to-compute ratio sweep (latency 5 µs):

| nx_local | T_comm/T_int | blocking stall % | waitany stall % | error vs bound |
|---|---|---|---|---|
| 4.0 | 8.38 | 58.3 | 59.6 | 0.00 % ✅ |
| 5.3 | 5.02 | 53.4 | 54.2 | 0.00 % ✅ |
| 8.0 | 2.79 | 45.6 | 43.4 | 0.00 % ✅ |
| 10.7 | 1.93 | 39.8 | 32.7 | 0.00 % ✅ |
| 16.0 | 1.20 | 31.8 | 11.1 | 0.00 % ✅ |
| 21.3 | 0.87 | 26.4 | 0.0 | 0.00 % ✅ |

Note the two leftmost rows: when the exchange is longer than the interior window,
the pipelined arm is **slightly worse** than blocking (59.6 % vs 58.3 %), because
it pays two small border kernels that blocking fuses — an honest crossover, not a
win. The pipelined schedule only pays off once `T_comm/T_int < ~3`.

Figures: `figures/xval_waitany_fig1_stall_vs_latency.png`,
`figures/xval_waitany_fig2_hidden_window.png`,
`figures/xval_waitany_fig3_waitany_law.png`.

## ⑤ Convergence / sensitivity (≥2 parameters)

1. **Latency α (1→100 µs)**: exposed stall of both arms grows linearly with α;
   the pipelined arm always sits `min(T_comm, T_int)` below blocking, and the
   measured value tracks the analytic bound to 0.00 % over the whole range.
2. **Comm/compute ratio (0.87→8.38)**: overlap fraction goes 100 % → 0 %; the
   schedule reaches zero exposed stall only below ratio ≈ 1.
3. **Arrival spread (0→32 µs)**: the `Waitany` switch follows `min(spread, consume)`
   exactly and saturates at 2 µs.

Decomposition invariance (the "grid convergence" analogue for a comm schedule):
error is 0.0 for world sizes 2/3/4 and for non-uniform cuts 6/5/5 — the result
does not depend on the decomposition, so the sweep is not hiding a
decomposition-specific artefact.

## ⑥ Failure modes, differences vs OSS, and what was not proven

Difference list (TensorLBM vs TCLB) and the transfer decision:

| Aspect | TCLB | This port | Transfer decision |
|---|---|---|---|
| Buffers | one buffer per **actual** neighbour (`bufsize[i]`, `bufnumber`) | 2 x-ring neighbours, one 19-population column each | adopt per-neighbour buffers; no y/z halos exist here so `bufnumber`=2 |
| Async copy | separate `inStream`/`outStream` CUDA streams (`outStream.run(0)`) | CPU `contiguous()` pack; no device copy | not transferable (no CUDA leg in this transport); pack charged in the model |
| Kernel split | `RunBorder` → `MPIStream_A` → `RunInterior` → `MPIStream_B` | same order, and interior streaming (not only the copy) runs under the exchange | adopt (widened window: CPU has no async copy to wait for) |
| Switch | compile-time `CROSS_MPI_WAITANY` / `CROSS_MPI_OLD` | runtime `mode` = `waitany`/`waitall`/`blocking` | adopt; `blocking` arm doubles as the A/B control |
| Send side | `MPI_Waitall(sendreq)` always | identical (`waitall_calls == 1` in the `waitany` arm) | adopt |
| Communicator | `MPMD.local` zone/cross-zone topology | single periodic x-ring | out of scope |

Failure modes found:

* **gloo has no reachable progress engine.** `Work.is_completed()` on a gloo
  `irecv` never reports completion by polling (verified: 10 s poll timeout on all
  three ranks), so a literal `MPI_Waitany` cannot be reproduced on this backend.
  A thread-per-receive progress shim worked for a single pair but **deadlocked**
  with two concurrent receives per rank; it was removed. Consequence: the
  real-3-rank arm verifies correctness and wiring (`posts/waitany/waitall` =
  `2/2/1` vs `2/0/2`), the completion-order *advantage* is quantified only on the
  deterministic backend. This is stated rather than glossed over.
* **Physical fields are a fixture, not physics.** The collision operator is a
  pointwise affine map, chosen because the border/interior split is only legal for
  cell-local operators; no drag/lift or dimensionless physical quantity is claimed
  here, and no field contour is shown (the schedule does not change the fields —
  they are bit-identical, which is itself the evidence).
* Narrow slabs (<3 columns) cannot be split into border/interior and are rejected
  with a message pointing at the blocking arm.

Artifacts: `artifacts/xval_waitany_overlap_m17454.json` (all tables, raw),
`src/tensorlbm/comm_overlap/`, `tests/test_comm_overlap_waitany.py` (19 cases),
`tests/test_comm_overlap_gloo_torchrun.py` (3-rank real gloo A/B).