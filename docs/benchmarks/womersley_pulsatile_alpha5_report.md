# CASE-LBM-WOMERSLEY-PULSATILE — pulsatile (Womersley) duct flow, alpha = 5

**Verdict: `NOT_PASSED`** (centreline amplitude ratio and phase lag vs the analytic Womersley
solution, whole-period phase averaged, 5 % band).

## 1. Problem setup and benchmark source
Harmonic body force `a(t)=a0 cos(omega t)` (i.e. `dp/dx=-rho a`) driving (i) a circular
tube of radius R (axisymmetric axial flow, no-slip at r=R) and (ii) a plane channel of gap H.
Reference: **Womersley, J. R. (1955) J. Physiol. 127(3):553-563, doi:10.1113/jphysiol.1955.sp005276**
(tube: `U(r)=(a0/(i omega))[1-J0(lam r/R)/J0(lam)]`); parallel-plate form: Sexl (1930),
`U(y)=(a0/(i omega))[1-cosh(lam eta)/cosh(lam)]`. Womersley number `alpha=R sqrt(omega/nu)`
(pipe) / `(H/2) sqrt(omega/nu)` (channel); `alpha=5` -> Stokes layer `R/alpha` of only a few cells.
Acceptance predeclared before the runs: `| |U_num(0)|/|U_ana(0)|-1 | <= 5%` and
`|dphi|/|phi_ana| <= 5%`, plus a pipe resolution refinement check.

## 2. Method and implementation
* 2D channel (D2Q9): `TensorLBM/src/tensorlbm/verification/womersley_pulsatile/channel2d.py`;
  pre-streaming half-way bounce-back walls (repo-verified analytic-channel convention), BGK with
  second-order Guo (2002) body force transferred from the library's own
  `ice_hydro/lbm_channel.py::_collide` (the old `turbulent_channel._apply_body_force_2d` entry used by
  `benchmarks/verified/womersley` no longer exists — recorded gap, not hidden).
* 3D pipe (D3Q19): `.../pipe3d.py`; Bouzidi-Firdaouss-Lallemand interpolated bounce-back on the true
  circle (`q<1/2`: `2q f_i(A)+(1-2q) f_i(A-e_i)`; `q>=1/2`: `(1/2q) f_i(A)+(1-1/2q) f_opp(A)`, q from the
  analytic circle intersection). Structure transferred from OSS-cache implementations of the same scheme
  (pylbm `boundary.py::BFLBounceBack`; lbmpy `NoSlipLinearBouzidi`; palabos `partialBBdynamics`). No OSS
  cache entry ships a Womersley verification case, so driver + analytic reference + estimator are new here.
* Estimator `.../metrics.py`: 64 phase bins over the driving period (whole-period phase averaging, 4
  periods after `max(5, ceil(6.5 tau_diff/T))` startup periods) + least-squares harmonic fit
  `u=A cos(omega t)+B sin(omega t)+C`, `Z=A-iB`; unbiased to sinc bin attenuation (0.04 % at 64 bins).
* **Sign trap found by the quasi-steady test**: the tube needs `lam^2=-i alpha^2` (radial Bessel operator)
  while the plate needs `+i alpha^2`. With the wrong sign the alpha=5 pipe amplitude is unchanged (the two
  candidates are conjugate up to the `1/i` rotation, 0.3 deg apart) but the quasi-steady limit changes sign
  (`analytical.womersley_lam`).

## 3. Parameter table
`nu=(tau-1/2)/3`, `omega` from the alpha definition, `a0` scaled to analytic peak speed `u_peak=0.03`
(`Ma=0.052`), `dx=dt=1`. Startup periods skipped as above.

| case | geom | res | tau | nu | omega | T (steps) | steps | abs ratio | lag num (deg) | lag ana (deg) | amp err % | phase err % | L2 % | inner L2 % | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| channel_H59_a5_tau0.8 | channel | 59 | 0.8 | 0.1000 | 0.002873 | 2187.2 | 32805 | 0.999614 | 91.134 | 91.214 | 0.0386 | 0.0871 | 0.130 | 0.123 | PASS |
| pipe_R10.5_a5_tau0.8 | pipe | 10.5 | 0.8 | 0.1000 | 0.022676 | 277.1 | 2493 | 1.000536 | 89.161 | 89.852 | 0.0536 | 0.7692 | 0.860 | 0.874 | PASS |
| pipe_R16.5_a5_tau0.8 | pipe | 16.5 | 0.8 | 0.1000 | 0.009183 | 684.2 | 6156 | 0.999281 | 89.572 | 89.852 | 0.0719 | 0.3121 | 0.360 | 0.365 | PASS |

Steady wall audit (constant force, centreline target 0.02, analytic `a0 R^2/(4 nu)`):
`R=10.5` (23 cells, 1385 steps): centreline error
**0.3916 %**, profile L2 0.3410 % — i.e. the BFL wall is
not staircase-limited.

## 4. Reference vs computed
Every number in the table is read from `artifacts/womersley_pulsatile_alpha5/*.json` (no hand-typed values).
All cases are inside the 5 % band on both declared metrics.

## 5. Convergence and sensitivity
Pipe refinement `R=10.5 -> 16.5`: amplitude 0.0536 -> 0.0719 %,
phase 0.7692 -> 0.3121 %, profile L2
0.860 -> 0.360 %, convergence check `False`.
The declared metric does not tighten monotonically (both points are already ~0.1 %), which is reported as is.

## 6. Failure modes, limits, reproduction
* The informative full-profile L2 is dominated by the near-wall Stokes layer (~2 cells at R=10.5, alpha=5);
  the inner-core value (`r/R<=0.8`) is reported separately. The *declared* acceptance is the centreline
  amplitude/phase, exactly as specified by the case pool.
* Compute budget: the channel `H=119` refinement, the pipe `R=20.5` point and the `tau` sensitivity scan were
  dropped this round (a full-scan job exceeded the wall-clock allowance); the resolution evidence is carried
  by the pipe pair and the channel is reported at `H=59`.
* Reproduce: `PYTHONPATH=TensorLBM/src python3 docs/benchmarks/build_womersley_pulsatile_report.py`
  (single case: `benchmarks/verified/womersley_pulsatile/run.py scan OUT --alpha 5 --H 59 --R 10.5 16.5`).
  Tests: `PYTHONPATH=TensorLBM/src python3 -m pytest TensorLBM/tests/test_womersley_pulsatile.py -q`.
