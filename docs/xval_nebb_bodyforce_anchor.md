# External anchor — XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S

This file is the *external* anchor for the anchor node
`XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S` (TensorLBM): it records what
the delivered numbers are judged against, independently of our own code, and
what the comparison does **not** prove.

## 1. Analytic reference (primary referee, closed form)

Steady plane-Poiseuille flow in a channel of height `H` driven by a constant
body force `g_x` (equivalently a constant pressure gradient `g_x = dp/dx`):

```
u_x(y) = g_x / (2 nu) * y * (H - y),      u_max = g_x H^2 / (8 nu)
```

* `y` measured from the **wall node plane** for the node-wall closure
  (`H = ny - 1`, walls sit *on* rows `0` and `ny-1`), and from the
  **half-cell plane** for half-way bounce-back (`H = ny - 2`).
  Conflating the two conventions injects a ~9% error at `ny = 10` and is the
  single largest trap in this benchmark (see `REPORT_R4.md` §3.2).
* Reference: G. K. Batchelor, *An Introduction to Fluid Dynamics*, CUP 1967,
  §4.2 (plane Poiseuille, exact solution of the Navier–Stokes equations).

## 2. Body-force scheme (the thing under test)

* X. He, S. Chen, G. D. Doolen, "A novel thermal model for the lattice
  Boltzmann method in incompressible limit", *J. Comput. Phys.* **146** (1998)
  282–300 — the `u* = u + F/(2 rho)` half-force step used by the wall-velocity
  correction.
* Z. Guo, C. Zheng, B. Shi, "Discrete lattice effects on the forcing term in
  the lattice Boltzmann method", *Phys. Rev. E* **65** (2002) 046308 — the
  discrete forcing term `F_i = (1 - w_i/(2 c_s^2)) ...` and its `u + F/(2 rho)`
  velocity shift; this is the `guo` arm.
* A. Kupershtokh, D. Medvedev, D. Karpov, "On equations of state in a lattice
  Boltzmann method", *Comput. Math. Appl.* **58** (2009) 965–974 — the
  exact-difference / EDM forcing variant; this is the `edm` arm.

## 3. Wall closure under test

* Force-corrected non-equilibrium bounce-back (NEBB, a.k.a. Zou–He style
  pressure/velocity closure) reconstructed from the off-equilibrium stress,
  with the wall velocity replaced by the half-force-step value and the unknown
  wall density **recovered from the interior along the wall normal**:
  * Q. Zou, X. He, "On the pressure and velocity boundary conditions for the
    lattice Boltzmann BGK model", *Phys. Fluids* **9** (1997) 1591–1598;
  * M. Hecht, J. Harting, "Implementation of on-site velocity boundary
    conditions for D3Q19 lattice Boltzmann", *J. Stat. Mech.* (2010) P01018 —
    the recovered-density / closure-consistency argument used at the corner.

## 4. Independent implementation cross-check (OSS arm)

* `haimingz/vivsim` (JAX, Apache-2.0): `vivsim/lbm/basic.py`,
  `vivsim/lbm/forcing/{edm,guo}.py`, `vivsim/lbm/boundary/nebb.py`,
  `examples/2d/poiseuille_channel.py`.  Transcribed line-by-line into NumPy
  (`OssPort` in `benchmarks/xval/xval_vivsim_nebb_bodyforce.py`) because JAX is
  not installable in this environment — `oss_binary_executed: false`, sha256 of
  every transcribed file recorded in `results.json:oss_provenance`.

## 5. What is anchored, what is not

| quantity | anchor | status |
|---|---|---|
| steady `u_x(y)` profile | closed form above | anchored, rel-err ≤ 3e-14 |
| arm-vs-arm difference | two independent transcriptions | ≤ 6e-14 -- but **not independent evidence**: the `edm`/`guo`/`recovered` arms are one velocity trajectory to 6.5e-14 (they differ only in a momentum-neutral population mode at O(g²)), see `docs/xval_nebb_bodyforce_transient.md` §3.3 |
| wall position (node vs half-cell) | `H = ny-1` vs `H = ny-2` | anchored, negative control separates them (0.25 vs 3e-14) |
| grid-refinement order, D3Q19 core path | `O(h^2)` expectation | **not** confirmed (measured 1.44) — honest open item, `artifacts/r4_invariants.json` |
| transient / unsteady channel start-up | impulsively started Poiseuille series (Batchelor 1967 §4.2), checked against an independent implicit-FD solution | **anchored** -- `docs/xval_nebb_bodyforce_transient.md`, err ≤ 1.2e-3 of `u_max`, decay rate ≡ `pi^2 nu/H^2` to 0.13 % |
| 3-D duct, curved walls, turbulence | — | not supported by this driver |
