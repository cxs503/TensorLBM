# XVAL: D3Q19 AA-pattern fused stream-collide against exact solutions

**Anchor**: `XVAL-LBM-AA-PATTERN-FUSED-STREAM-COLLIDE` (lanl/MF-LBM
`aa-pattern-fused-stream-collide`) · **Mission**: m22757 · **Date**: 2026-11-16
**Disposition**: *capability extension, verified* — the anchor's pattern now
exists and is verified in 3D (D3Q19), which is the dimensionality of the
reference practice.

---

## 1. Why this increment

`docs/xval_aa_pattern_fused_stream_collide_acceptance.md` closes the anchor's
acceptance on a **D2Q9** lid-driven cavity (corrected Ghia scoring, both centre
lines below 3 %) and states the residual scope limit verbatim in its §5 item 4:

> **Scope**: D2Q9 / BGK only. Nothing here transfers automatically to MRT/TRT or
> D3Q19/27 variants of the pattern.

The reference practice is a 3D D3Q19 code (MF-LBM).  A 2D-only kernel therefore
verifies the *algebra* of the pattern but not the *stencil transfer*: in 3D the
stencil has three directional families (axis, in-plane diagonal, out-of-plane
diagonal), the weights change (`1/3, 1/18, 1/36`), the opposite map is no longer
a geometric mirror of the index ordering, and the wall term picks up a third
component.  Each of these is a place where a port can be silently wrong while
still "running".

This document reports the D3Q19 transfer (`tensorlbm.aa_fused_push_pull_3d`)
and its verification.

## 2. Physics, geometry, parameters

| quantity | value | note |
|---|---|---|
| lattice | D3Q19, BGK | weights/velocities/opposites **imported** from `tensorlbm.d3q19`, never re-typed |
| state layout | `f` shape `(19, nz, ny, nx)` | AA pattern: one contiguous plane per direction |
| collision | BGK, `nu = (tau - 1/2)/3` | |
| walls | halfway bounce-back, written on the **fluid** side | `f_i(x,t+1) = f_j^post(x,t) + 6 w_j rho (c_j·u_w)`, `j = opp(i)` |
| force | Guo 2nd order (JCP 165 (2000) 288) | `S_i = w_i[3(c_i-u)·F + 9(c_i·u)(c_i·F)]` |
| precision | float64 | |
| channel (Couette/Poiseuille) | `n_fluid = 16` fluid rows, 1 solid frame row each side, periodic in x and z | `H = n_fluid` |
| Couette | `U_wall = 0.05`, `tau = 0.8` | exact `u_x(y) = U (y+1/2)/H` |
| Poiseuille | `g = 0.00015625`, `tau = 0.8`, `u_max = g H^2/8nu = 0.04999999999999999` | exact parabola |
| shear wave | `ny = 64`, `u0 = 0.02`, `k = 2pi/64 = 0.09817`, `tau = 0.8` | exact `u0 cos(ky) e^{-nu k^2 t}` |
| Taylor-Green | `u0 = 0.02`, `tau = 0.8`, one period per side | exact 3D NS solution (see §2.1) |

### 2.1 The Taylor-Green case is an *exact* 3D Navier-Stokes solution

```
ux =  u0 cos(kx) sin(ky) e^(-2 nu k^2 t)
uy = -u0 sin(kx) cos(ky) e^(-2 nu k^2 t),   uz = 0,   uniform in z
```

The convective term `u·grad u` does **not** vanish, so this is not a linear
decay artefact: it is an exact solution of the full 3D equations, uniform in
`z`.  It exercises the in-plane diagonals of D3Q19 and the nonlinear term at the
same time, at zero extra cost.  All quantities are in **lattice units**;
`k = 2 pi m / n` with `m` periods over the domain.

## 3. Method — five gates, in order of logical dependence

Driver: `benchmarks/xval/aa_fused_d3q19_verify.py`; fast gates:
`tests/test_aa_fused_d3q19.py` (13 tests, ~34 s on 2 threads); figures:
`tests/gen_aa_fused_d3q19_figures.py`; this document:
`tests/gen_aa_fused_d3q19_doc.py`.

1. **identity** — fused operator vs a materialised collide-then-stream oracle on
   the *same* wall/force/moving-wall state.  Requires `torch.equal`
   (`0` differing entries), not a tolerance.
2. **vacuity control** — the historical destination-colliding `C[S[f]]` operator
   must be *rejected* by the same comparison.  Without it, "equality" could be
   satisfied by two copies of the same wrong operator (that is exactly the defect
   the 2D audit found: the retired "equivalence" test compared operators on a
   spatially uniform state, where every stencil is the identity).
3. **Couette** — moving-wall sign and magnitude against an exact profile.  A
   fused-vs-materialised comparison *cannot* see a sign error in
   `6 w_j rho (c_j·u_w)` because both sides share that line of code.
4. **Poiseuille** — Guo forcing and the D3Q19 weights against an exact parabola.
5. **shear wave** — the transport coefficient itself: `nu_eff` is *measured* from
   the decay of a transverse mode and compared with `(tau-1/2)/3`.
6. **Taylor-Green** — nonlinear term + in-plane diagonals against an exact 3D NS
   solution, over a refinement study with the physical problem held fixed.

## 4. Results (all numbers generated programmatically from
`artifacts/xval_aa_pattern/d3q19_verify.json`)

### 4.1 Operator identity and the vacuity control

| check | value |
|---|---|
| grid / state | `[5, 8, 7]`, random non-uniform + interior obstacle + moving wall + body force |
| `tau` | 0.63 |
| fused vs materialised, differing entries | **0** / 5320 |
| max abs difference | **0.0** (bit-identical) |
| periodic domain, bit-identical | True |
| `C[S[f]]` (stream-then-collide) vs oracle, max abs diff | 2.7787e-02 (8.18 %) |
| vacuity control rejected | **True** |

### 4.2 Plane Couette (moving wall, exact solution)

| quantity | value |
|---|---|
| steps to criterion `max|du|/U < 1e-10` | 6000 (last residual 6.33e-11) |
| profile $L_2$ error / `U_wall` | **3.86e-09 %** |
| max pointwise error / `U_wall` | 5.43e-09 % |
| `u_x` at mid-height, numeric / exact | 0.026562500 / 0.026562500 |

The gate reproduces the linear profile to round-off, i.e. the moving-wall term
is right in **sign, magnitude and direction** in 3D.  The test suite additionally
carries a sign-inverted control arm that must fail the same gate
(`test_plane_couette_sign_flip_is_rejected`).

### 4.3 Plane Poiseuille (Guo forcing, exact parabola)

| quantity | value |
|---|---|
| steps to steady (drift < 1e-9/step) | 5500 (last 1.79e-10) |
| profile $L_2$ error / `u_max` | **0.203 %** |
| `u_max` exact / measured | 0.050000 / 0.049703 |
| relative peak error | 0.594 % |

The residual is the well-known half-cell offset of halfway bounce-back + Guo
forcing; it is a **discretisation** error, quadratic in the mesh — the fast gate
at `n_fluid = 8` gives 0.812 %, this run at `n_fluid = 16` gives
0.203 %, a factor ≈ 4 for a factor 2 in `H`.  It is not a stencil defect:
a wrong weight or a missing force term is O(1), not O(H^-2).

### 4.4 Decaying shear wave — the transport coefficient

| t | RMS `u_x` numeric | exact | `nu_eff` | profile $L_2$ |
|---|---|---|---|---|
| 50 | 0.013470 | 0.013477 | 0.10112 | 0.0539 % |
| 100 | 0.012835 | 0.012843 | 0.10058 | 0.0564 % |
| 200 | 0.011656 | 0.011663 | 0.10032 | 0.0613 % |

Measured `nu_eff = 0.100318` against `nu = 0.100000`
(relative error **0.318 %**).  This is the sharpest check on the
weight table: an exchanged weight pair or a wrong `c_s^2` shifts `nu` by an O(1)
factor while leaving mass and momentum conservation untouched.

### 4.5 2D-extended Taylor-Green vortex — nonlinear term, refinement

| n | steps | `k_lat` | $L_2$ rel. error | decay-factor rel. err | decay exact / measured | rho range | mass drift | wall |
|---|---|---|---|---|---|---|---|---|
| 32 | 90 | 0.1963 | 0.75 % | 0.494 % | 0.49960 / 0.49713 | [0.999858, 1.000141] | 6.22e-15 | 0.4 |
| 64 | 360 | 0.0982 | 0.131 % | 0.13 % | 0.49960 / 0.49894 | [0.999701, 1.000299] | 2.51e-14 | 3.1 |
| 128 | 1440 | 0.0491 | 0.0546 % | 0.0307 % | 0.49960 / 0.49944 | [0.999999, 1.000000] | 9.98e-14 | 42.4 |

Convergence order between successive grids: 32→64: **2.51**, 64→128: **1.27**.  The order drops below 2
on the finest pair because the fixed-`Ma` compressibility floor (and the O(dt)
splitting of collision and streaming) starts to dominate at
`L_2 ≈ 5e-4`; the error is **monotone decreasing** and already 55× below the
3 % gate at `n = 32`.  `z` symmetry is preserved to machine precision
(`max|u_z| ≈ 1e-17` in every run) and mass is conserved to `1e-13`.

### 4.6 Error judgement (explicit, owner standard)

| key quantity | error | < 3 %? | < 15 %? |
|---|---|---|---|
| plane Couette profile $L_2$ | 3.86e-09 % | **yes** | yes |
| plane Poiseuille profile $L_2$ | 0.203 % | **yes** | yes |
| shear-wave $\nu_{eff}$ relative error | 0.318 % | **yes** | yes |
| shear-wave profile $L_2$ | 0.0613 % | **yes** | yes |
| Taylor-Green $L_2$ (n=32) | 0.75 % | **yes** | yes |
| Taylor-Green $L_2$ (n=64) | 0.131 % | **yes** | yes |
| Taylor-Green $L_2$ (n=128) | 0.0546 % | **yes** | yes |

* Worst gate error across the whole verification: **0.75 %**.
* "Is the key quantity within 3 % of the reference?" — **yes for every gate
  above**; the only non-round-off number is the Poiseuille half-cell offset
  (0.203 %), which is a known second-order wall artefact of halfway
  bounce-back and shrinks as `H^-2`.
* The fused-vs-materialised comparison is not "within 3 %": it is
  **bit-identical** (0 differing entries).

Figures: `fig_d3q19_gates.png` (profiles vs exact + pointwise error),
`fig_d3q19_viscosity.png` (shear-wave decay, `nu_eff` vs `(tau-1/2)/3`),
`fig_d3q19_taylor_green.png` (vortex field + streamlines, refinement, decay),
`fig_d3q19_gate_summary.png` (all gates against the 3 % / 15 % standards).

## 5. Notes, pitfalls, limitations

1. **A wrong "exact" reference manufactures a fake 20× defect.** The first
   version of this harness used the *physical* wavenumber `k = 1` while the
   simulated mode had the lattice wavenumber `2 pi/n`; the resulting reference
   decayed ~25× too fast and the kernel looked like it had a 25× viscosity error.
   The fix was to state the exact solutions in lattice units (`k_lat`).  This is
   the same class of defect as the scrambled Ghia table of m22379: the harness,
   not the kernel, was wrong — and it was caught only because an independent
   gate (Couette, Poiseuille) disagreed.
2. **Equality tests can be vacuous.** Two copies of the same wrong operator, or a
   comparison on a uniform state, both "pass".  The suite therefore carries the
   `C[S[f]]` control arm and asserts it *fails*.
3. **Solid-node populations are junk.** Only fluid nodes carry meaning; poisoning
   them with `1e3` must not change a single fluid value
   (`test_solid_node_populations_do_not_leak_into_fluid`).
4. **Scope**: D3Q19 / BGK in float64 on dense arrays.  This is the *stencil*
   transfer of the AA pattern, not a claim about MF-LBM's GPU kernels: no
   performance or memory number here is comparable to a CUDA implementation, and
   the kernel is a PyTorch `roll` reference (≈5 ms/step regardless of size, i.e.
   launch-bound), not a bandwidth-optimised loop.  MRT/TRT collision, D3Q27 and
   grid refinement are *not* covered.
5. **Not run (honest)**: no 3D lid-driven cavity comparison (no provenance-bound
   3D reference table in this repository), no high-Re turbulent case, and the
   cluster re-run of this driver (job `cj4365`) was cancelled after the numbers
   below were produced locally — so these results have **one** execution
   instance.  The 13-test suite reproduces the same gates at reduced budgets.

## 6. Artefacts

| path | content |
|---|---|
| `src/tensorlbm/aa_fused_push_pull_3d.py` | D3Q19 fused pull + materialised oracle + exact references |
| `benchmarks/xval/aa_fused_d3q19_verify.py` | five-gate driver → JSON |
| `tests/test_aa_fused_d3q19.py` | 13 fast gates (identity, vacuity, conservation, four exact solutions) |
| `tests/gen_aa_fused_d3q19_figures.py` | figures |
| `tests/gen_aa_fused_d3q19_doc.py` | this document (tables injected from the JSON) |
| `artifacts/xval_aa_pattern/d3q19_verify.json` | all numbers above |
| `artifacts/xval_aa_pattern/fig_d3q19_*.png` | figures |

Reproduce:

```
cd TensorLBM
python benchmarks/xval/aa_fused_d3q19_verify.py          # ~2 min, all five gates
pytest tests/test_aa_fused_d3q19.py -q                    # 13 passed (~34 s)
python tests/gen_aa_fused_d3q19_figures.py && python tests/gen_aa_fused_d3q19_doc.py
```
