# GAP-PP-1-triton-bwd — Differentiable fused Triton LBM step (forward + adjoint kernel pair)

**Anchor**: `GAP-PP-1-triton-bwd` (ledger: *100-step Triton step `∂loss/∂τ` vs eager autograd
relative difference < 1e-5; FD cross-check < 1e-4; throughput ≥ 2× eager*), paper pull:
JOUR-DIFFLBM §2/§4. Mission 15355, branch `agent/m15355`, repo TensorLBM.

**Verdict (honest, per criterion)**

| # | Criterion | Status | Numbers |
|---|---|---|---|
| 1 | 100-step `∂loss/∂τ`, fused-Triton vs eager autograd, rel < 1e-5 | **PASS** | 1.95e-6 (τ=0.80), 1.68e-6 (τ=0.60), 7.9e-8 (n=8³, 20 steps) |
| 2 | FD cross-check rel < 1e-4 | **PASS ≤50 steps, NOT MET at 100 steps** | 4.4e-7…1.07e-5 at 2–50 steps; **1.27e-4** at 100 steps = fp32 roundoff floor (proved by FD `h`-convergence, §5) |
| 3 | Throughput ≥ 2× eager | **NOT MEASURABLE / WITHHELD** | no CUDA device on any host reachable from this worktree (`/dev/nvidia*` absent, `torch.cuda.is_available()==False`); `TRITON_INTERPRET=1` is a NumPy interpreter, not a throughput claim |

Criterion 3 is **device-gated, not a code defect**: `benchmarks/triton_fused_bwd_validation.py
--device cuda` measures it unchanged on any GPU host (it times the same calls and records
`throughput.status = measured`).

---

## 1. Problem setting and benchmark provenance

*Object.* TensorLBM's fast periodic D3Q19 step `tensorlbm.triton_fused.triton_fused` (stream +
BGK collision fused into one Triton kernel; module docstring reports 8.6 GLUPS at n=256 on an
RTX 5090, 77 % of achievable bandwidth) is **detached**: nothing flows back from a loss to the
relaxation time `τ` or to the initial distribution. The eager graph
(`tensorlbm.autograd_path.differentiable_step`, built on `solver3d.stream3d` /
`solver3d.collide_bgk3d`) does keep the operators in the autograd graph, at the cost of
materialising every intermediate in DRAM (~19 gathers + ~15 elementwise kernels per step).

*Benchmark provenance (external anchors).*
- Discrete adjoint/inverse-design of LBM time stepping: Um, Holl, Thuerey et al., *Solver-in-the-Loop*,
  NeurIPS 2020 (arXiv:2007.00084); Ataei & Salehipour, *XLB: distributed multi-GPU LBM*, CPC 300:109187 (2024)
  — both make the *time-stepping operator itself* part of the backward graph (the paradigm this module implements).
- BGK collision and D3Q19 equilibrium as implemented here: `tensorlbm.d3q19` (canonical `C`/`W` tables), the
  operator identity `f_post = f - (f - f_eq)/τ`, `f_eq = w_q ρ[1 + 3(c_q·u) + 4.5(c_q·u)² - 1.5u²]`
  (Krüger et al., *The Lattice Boltzmann Method*, Springer 2017, §3.2/§5.2).
- Operator order: the fused kernel is **stream-then-collide** (pull form, `f_str_q(i)=f_q(i-c_q)`), which is
  exactly `collide_bgk3d(stream3d(f), τ)` — verified to 2.5e-7 in fp32 (§3). `autograd_path.differentiable_step`
  is the **collide-then-stream** ordering `S∘C`; the two are different maps, so the gradient comparison must be
  made against the same map. This module states which one it implements (documented in the module docstring).

## 2. Method

### 2.1 Forward (existing, reused — single source of truth)

`triton_fused_step(f, τ)` wraps the audited kernel `triton_fused(f, τ, block_x=block_y=block)` in a
`torch.autograd.Function`. One kernel launch per step; no intermediate DRAM buffers.

### 2.2 Backward: hand-derived discrete adjoint, one kernel (`_step_adjoint_kernel`)

With `k = 1/τ`, `ḡ = ∂L/∂f_out` (adjoint of the post-step distribution) and `f_str` the post-stream
state of the *same* cell block:

1. **Collision adjoint.** `∂f_eq,q/∂f_str,p = w_q A_q + w_q (3+9c_q·u)(c_q·c_p − c_q·u) − 3w_q(cu_p − u²)`,
   `A_q = 1 + 3cu_q + 4.5cu_q² − 1.5u²`. The `c_q·c_p` factor is the **lattice Gram matrix**
   (`G[p,q] = c_p·c_q ∈ {0,1,2}`), contracted per cell block with `tl.dot(G, h)`,
   `h_q = ḡ_q w_q (3+9cu_q)`, in `input_precision="ieee"` (tf32 would cost ~1e-3 and break the 1e-5 gate).
   Result:
   `f̄_str,p = (1−k)ḡ_p + k[ S₀ − S₂ + (G h)_p − 3 cu_p S_t + 3 u² S_t ]`,
   `S₀ = Σ_q ḡ_q w_q A_q`, `S₂ = Σ_q h_q cu_q`, `S_t = Σ_q ḡ_q w_q`.
   **Implementation note (real defect found by the gradient check, fixed):** an earlier version replaced
   `c_q·c_p` by `cu_p = c_p·u`. That substitution is wrong — it makes the leading `3cu_p` cancel and
   produced `∂f_eq/∂f` errors of 4 %–120 %, invisible in the forward pass. The pure-torch VJP check
   (`docs` §5 and `tests/test_triton_fused_diff.py`) is what caught it; the Gram form agrees to 1.2e-8.
2. **Streaming adjoint.** Streaming is a permutation, so its adjoint is the scatter
   `f̄_q(i − c_q) += f̄_str,q(i)` — the *same* wrapped offsets as the forward pull, hence **no atomics**
   (for fixed `q`, `i ↦ i−c_q` is a bijection). Verified exactly (0.0 relative) against autograd in isolation.
3. **τ adjoint.** `∂f_out/∂k = −(f_str − f_eq)`, `dk/dτ = −1/τ²` ⇒
   `∂L/∂τ = τ⁻² Σ_{cells} Σ_q ḡ_q (f_str,q − f_eq,q)`. Per-program partial sums go to a scratch buffer
   (one slot per program, `torch.sum` reduction) — no float atomics, so `∂L/∂τ` is **bitwise reproducible**
   for a fixed grid/block shape.

### 2.3 Objective (why the loss is trajectory-accumulated)

`L = (1/N) Σ_t mean(u_x²)(t)`. A *terminal*-only loss is unusable at 100 steps: unforced periodic BGK
decays to the uniform state, `∂L/∂τ` falls to ~1e-8 by step 50 — below the fp32 roundoff of the loss
(|L| ~ 1e-7) — so a relative gate there measures noise, not the adjoint (measured: 5.2e-3 at τ=0.8 and
93× at τ=0.6 with the terminal loss, before this change). Accumulating over the trajectory keeps the
early-time sensitivity in the gradient, which is also the objective used by the solver-in-the-loop papers.

## 3. Parameters (dimensionless)

| Quantity | Value(s) |
|---|---|
| Lattice | D3Q19, `c_s² = 1/3`, `w_0 = 1/3`, `w_{1..6} = 1/18`, `w_{7..18} = 1/36` |
| Box | `n³`, n = 4 (headline, 64 cells), n = 8 (512 cells) — periodic in x, y, z |
| Relaxation time τ | 0.80, 0.60 (`ω = 1/τ` = 1.25, 1.667) |
| Kinematic viscosity (lattice units) | `ν = (τ − 1/2)/3` = 0.100, 0.0333 |
| Initial condition | equilibrium at `ρ = 1`, `u = 0.05·randn(3,n³)`, `u_max ≈ 0.18` ⇒ `Ma = u/c_s ≈ 0.31` |
| Rollout length N | 2, 5, 10, 20, 50, 100 steps |
| Time step | `Δt = 1`, `Δx = 1` (lattice units) |
| dtype | fp32 (kernels), fp64 (eager reference for FD) |
| Tile / programs | `BLOCK_X = BLOCK_Y = n`, one program per z-slice; `tl.dot` needs `BLOCK_X·BLOCK_Y ≥ 16` |

## 4. Results — reference vs obtained

Measured artifact: `runs/m15355/gap_pp1_triton_bwd/verify.json` (+ `fd_study.json`, figure PNG),
produced by `benchmarks/triton_fused_bwd_validation.py`. Execution mode for these numbers:
`TRITON_INTERPRET=1` NumPy interpreter, CPU (the *same* kernel source; not compiled GPU code).

Forward parity (kernel vs `collide_bgk3d(stream3d(f))`, 1 step, fp32): 2.46e-7 (n=4), 3.18e-7 (n=4, τ=0.6),
2.46e-7 (n=8).

| n | τ | steps | `∂L/∂τ` Triton | `∂L/∂τ` eager autograd | rel (gate 1e-5) | FD fp64 (h=1e-4) | rel (gate 1e-4) |
|---|---|---|---|---|---|---|---|
| 4 | 0.80 | 100 | −1.31468e-03 | −1.31468e-03 | **1.95e-6 ✓** | −1.31457e-03 | 1.27e-4 ✗ (fp32 floor) |
| 4 | 0.60 | 100 | −2.30383e-02 | −2.30383e-02 | **1.68e-6 ✓** | −2.30389e-02 | **2.46e-6 ✓** |
| 8 | 0.80 | 20 | −6.6030e-05 | −6.6030e-05 | **7.86e-8 ✓** | −6.6033e-05 | **3.93e-7 ✓** |

Step-count sweep (n=4, τ=0.80) — the gate is met at every horizon, the FD floor appears only at long N:

| steps | `∂L/∂τ` Triton | `∂L/∂τ` eager | rel vs eager | rel vs FD |
|---|---|---|---|---|
| 2 | −2.5277e-02 | −2.5277e-02 | 2.21e-7 | 4.37e-7 |
| 5 | −2.2706e-02 | −2.2706e-02 | 8.20e-8 | 5.89e-7 |
| 10 | −1.2982e-02 | −1.2982e-02 | 2.87e-7 | 1.36e-6 |
| 20 | −6.5723e-03 | −6.5723e-03 | 7.09e-7 | 1.83e-6 |
| 50 | −2.6291e-03 | −2.6291e-03 | 3.99e-6 | 1.07e-5 |
| 100 | −1.3147e-03 | −1.3147e-03 | 1.95e-6 | 1.27e-4 |

## 5. FD step-size study (is 1.27e-4 an FD artifact or a gradient error?)

fp64 central differences of the trajectory loss at N=100, τ=0.80 (`fd_study.json`):

| h | FD | rel vs Triton gradient |
|---|---|---|
| 3e-3 | −1.3148248e-03 | 3.23e-4 |
| 1e-3 | −1.3145955e-03 | 1.49e-4 |
| 3e-4 | −1.3145695e-03 | 1.29e-4 |
| 1e-4 | −1.3145672e-03 | 1.27e-4 |
| 3e-5 | −1.3145669e-03 | 1.27e-4 |

The FD value converges and **plateaus** at 1.27e-4 → the residual is *not* FD truncation; it is the
fp32 forward trajectory's own accumulated rounding (≈1e-7 relative per step over 100 steps, amplified
~1000× in the gradient), shared by the fp32 eager reference (which agrees with Triton to 1.95e-6).
Consequently: gate 2 is met at every horizon ≤50 steps and at τ=0.60/100 steps; at τ=0.80/100 steps the
fp32 floor is 1.27× the gate. Remedies, in preference order: (i) fp64 variant of the fused step
(no fp64 `tl` arithmetic path yet — `_FusedStep` rejects non-fp32 and points at the eager reference),
(ii) keep the 1e-4 gate on ≤50-step horizons, (iii) mixed precision: fp64 forward state, fp32 kernel
compute. **This is declared as a partial miss, not glossed over.**

## 6. Failure modes, scope limits, and artifacts

* **Scope**: D3Q19, fp32, periodic box, BGK only. Walls / inlet / outlet / IBM / forcing (Guo) are **not**
  in the fused kernel — for those use `autograd_path.differentiable_step` (eager). A forcing term is also
  what would make a *terminal* 100-step objective τ-sensitive; listed as the next extension.
* `tl.dot` requires `BLOCK_X·BLOCK_Y ≥ 16` (tile ≥ 4×4); smaller tiles raise in Triton, not silently wrong.
* `∂L/∂τ` of a **1-step** rollout is degenerate (`∂u/∂τ = 0` because BGK conserves mass/momentum): the
  relative difference there is roundoff/roundoff (1.3 in the sweep's N=1 check). Gradients are meaningful
  from 2 steps on; the harness therefore evaluates from N ≥ 2.
* `ρ_safe` clamp (`ρ > 1e-12`) in the kernel is non-differentiable at the clamp point; unreachable for
  physical fields, documented in the kernel.
* Throughput criterion remains **unmeasured** — no CUDA device reachable (`/dev/nvidia*` absent,
  `torch.cuda.device_count() == 0`, cluster nodes are CPU-only loong64), so no speed-up is claimed.
  Interpreter timings (40 s vs 1.5 s for 100 steps) are recorded in `verify.json` only as
  interpreter overhead, explicitly **not** as a performance statement.
* Bit-exactness: `∂L/∂τ` reduction is deterministic (per-program partials, no float atomics).

**Artifacts**

| Path | What |
|---|---|
| `src/tensorlbm/triton_fused_diff.py` | forward+adjoint kernel pair, autograd Function, rollout, eager reference |
| `tests/test_triton_fused_diff.py` | 8 tests (forward parity, conservation, `dL/df`, `dL/dτ` vs autograd + FD, fp64 rejection, env flag) — 8 passed |
| `benchmarks/triton_fused_bwd_validation.py` | acceptance harness (writes JSON + figure) |
| `runs/m15355/gap_pp1_triton_bwd/verify.json` | measured numbers, criteria block, environment block |
| `runs/m15355/gap_pp1_triton_bwd/fd_study.json` | FD `h`-convergence table |
| `runs/m15355/gap_pp1_triton_bwd/gap_pp1_triton_bwd.png` | accuracy-vs-steps + wall-time figure |

**Reproduce (CPU emulation of the kernels, ~2.5 min)**

```bash
cd <repo> && TRITON_INTERPRET=1 PYTHONPATH=src \
  python benchmarks/triton_fused_bwd_validation.py
TRITON_INTERPRET=1 PYTHONPATH=src python -m pytest tests/test_triton_fused_diff.py -q
```

**Reproduce on a GPU host (adds criterion 3)**

```bash
PYTHONPATH=src python benchmarks/triton_fused_bwd_validation.py --device cuda --steps 100 --n 64
```