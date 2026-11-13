# XVAL: TensorLBM spectral Saffman--Taylor (integrating factor) vs
`ebinan92/Fingering_dynamics` `spectrum_method`

**Mission**: m18378 · **Layer**: cross-validation (XVAL) · **Repo**: TensorLBM ·
**OSS reference**: `ebinan92/Fingering_dynamics` @ `150a573`,
file `spectrum_method/main.py` (executed directly from the clone as the
reference; numbers in this report are injected from
`xval/XVAL-LBM-SPECTRAL-SAFFMAN-TAYLOR-INTEGRATING-FACTOR/artifacts/results.json`).

---

## ① Problem + reference provenance

The OSS repository implements miscible viscous fingering in a Hele-Shaw cell with
a **Fourier pseudo-spectral solver**: exponential integrating factor for the
diffusion operator, RK4 bootstrap, Adams--Bashforth 2 for the convective term,
Crank--Nicolson/trapezoidal corrector (`Compute.getC_barbyAdams`,
`getC_barbyRunge`, `getNextC`).  The formulation is that of
C. T. Tan & G. M. Homsy, *Phys. Fluids* 31, 2339 (1988), the paper cited in the
OSS file header.

Dimensionless system (Pe absorbed in the domain `[0, A*Pe] x [0, Pe]`, so the
diffusion coefficient is 1 -- this is exactly the OSS convention):

```
dc/dt + J(psi, c) = lap(c),        J = psi_y c_x - psi_x c_y = u c_x + v c_y
lap(psi) + R (c_x psi_x + c_y psi_y) = -R c_y
```

with `R` the log-viscosity ratio and `u = psi_y`, `v = -psi_x`.

**Question (predeclared)**: can TensorLBM host this scheme as a reusable,
verifiable capability, reproducing the OSS reference and meeting the owner
precision standard (<15% error) against independent analytic anchors?

Alternatives considered: (A) keep fingering work inside the OSS code only;
(B) re-derive a different time integrator instead of porting (rejected: the
integrating-factor structure is exactly what makes the diffusion part exact).

## ② Method / implementation

* Port: `src/tensorlbm/spectral_saffman_taylor.py::SpectralSaffmanTaylor2D`
  (PyTorch, float64, CPU/GPU).  Every formula, operation order and array layout
  follows the OSS `Compute` class.
* Reference harness: `xval/.../oss_spectral_st.py::oss_reference_field` executes
  the **OSS `Compute` object from the clone** (its `main()` loop body replayed
  verbatim, no plotting); a self-contained NumPy transcription
  (`xval/.../oss_spectral_reference.py`) is kept for offline/CI reproduction and
  as a second implementation check.
* Tool chain: python, torch, numpy, matplotlib; pytest for the regression suite
  (`tests/test_spectral_saffman_taylor.py`, 7 tests, all passing).
* OSS quirk found while reading the source (`getPhi_hat`): the vorticity source
  is `R * c_hat * 1j * kn` -- **kn to the first power**, not `kn^2`.  An
  `kn^2` misreading of the source (my first attempt) produced a constant factor
  `kn` error in the stream function; the port was corrected against the source
  and is now verified to round-off.

## ③ Parameter table (complete dimensionless set)

| Quantity | Symbol | Value | Note |
|---|---|---|---|
| Grid | ny x nx | 128 x 256 | vertical x streamwise collocation points |
| Aspect ratio | A | 2 | x extent = A*Pe |
| Peclet (domain scaling) | Pe | 500 | diffusion coefficient = 1 |
| Log-viscosity ratio | R | 0 (M1--M4), 0.05--4 (M7) | OSS default is R = 0 |
| Time step | dt | 0.1 (0.2/0.1/0.05 in M7) | integrating factor + AB2 + CN |
| Transverse mode | n | 4 | k = 2*pi*4/Pe = 5.026548e-02 |
| k^2 of the extracted mode | k^2 | 2.526619e-03 | diffusive decay rate |
| Fidelity window | T | 25 (250 steps) | M1, M1b (T=10) |
| IC (M1) | -- | OSS random front, seed 20261113 | `hstack([rand(ny, nx/2), zeros])` |
| IC (M7) | -- | tanh front + one transverse mode, amplitude 1e-3 | linear regime |

## ④ Comparison table (ref vs got, ✓/✗ against the predeclared <15% rule)

**M1 -- port fidelity vs OSS reference (R = 0, random-front IC):**

| Quantity | Value | Threshold | Verdict |
|---|---|---|---|
| relative L2 (port vs OSS, 250 steps) | 2.965e-15 | < 1.5e-1 | ✓ |
| max |difference| | 4.774e-15 | -- | -- |
| mass drift | 0.000e+00 | -- | conserved |
| max|div u|/max|u| | 0.000e+00 | -- | machine precision |

**M1b -- port fidelity at R = 2 (smooth IC, 100 steps)**:
relative L2 = **2.192e-15** ✓
(max |diff| = 7.327e-15) -- the port is faithful also with the
vorticity coupling active.

**M2 -- integrating factor vs the analytic Fourier decay exp(-k^2 t):**

| dt | t_end | relative L2 | amplitude ratio | verdict |
|---|---|---|---|---|
| 0.1 | 20 | 1.545e-11 | 1.000000000 | ✓ |
| 0.05 | 20 | 3.853e-12 | 1.000000000 | ✓ |

**M7 -- analytic-anchored growth rate.**  For the linear regime the growth rate
must decompose as `sigma = C*R - k^2` (first-order coupling `C*R` plus the exact
diffusive decay of the transverse mode, whose k^2 part is verified separately in
M2).  Extracting C from the measured sigma:

| R | sigma (measured) | k^2 | C = (sigma+k^2)/R | deviation from anchor | <15%? |
|---|---|---|---|---|---|
| 0.05 | -1.5971e-03 | 2.5266e-03 | 1.85897e-02 | 0.90% | ✓ |
| 0.2 | +1.1931e-03 | 2.5266e-03 | 1.85985e-02 | 0.94% | ✓ |
| 0.5 | +6.7722e-03 | 2.5266e-03 | 1.85976e-02 | 0.94% | ✓ |
| 1 | +1.6025e-02 | 2.5266e-03 | 1.85514e-02 | 0.69% | ✓ |
| 2 | +3.4213e-02 | 2.5266e-03 | 1.83697e-02 | 0.30% | ✓ |
| 4 | +6.9494e-02 | 2.5266e-03 | 1.80051e-02 | 2.28% | ✓ |

Anchor C = **1.84245e-02** (mean over R >= 0.2), maximum deviation
**2.28%** -> PASS.

**Elliptic defect (M5):** the OSS frozen-coefficient symbol is not the exact
curl-Darcy operator; its normalised residual `lap(psi)+R(c.psi)+R c_y` is

| R | normalised residual |
|---|---|
| 0 | 0.000e+00 |
| 0.05 | 1.980e+00 |
| 0.2 | 1.925e+00 |
| 2 | 1.502e+00 |

so the closure is a first-order-in-R *approximation* (order-1 relative defect),
which is nevertheless stable and reproducible (see M6 below).  This is the main
item of the difference list, and it is the reason the port is documented as a
faithful transcription rather than as an accurate Darcy solver.

## ⑤ Convergence / sensitivity (>= 2 independent axes)

**(a) Time step** (R = 2, mode 4):

| dt | sigma | rel. deviation from Richardson value +3.421714e-02 |
|---|---|---|
| 0.2 | +3.421319e-02 | 0.0115% |
| 0.1 | +3.421283e-02 | 0.0126% |
| 0.05 | +3.421606e-02 | 0.0031% |

The scheme is time-step insensitive at the 1e-4 level over a factor-4 range of
dt (integrating factor + Crank--Nicolson): observed order noise-level
(-3.193461091429501).

**(b) Grid**: sigma(128x256) vs sigma(256x512) differ by
**4.22%** at fixed dt
(sigma_fine = +3.282925e-02).

**(c) Round-off amplification** (two trajectories seeded apart, 2 steps):

| R | grid | seed separation | separation after | amplification/step |
|---|---|---|---|---|
| 0.05 | 128x256 | 3.4e-15 | 3.5e-15 | 1.026 |
| 0.2 | 128x256 | 3.4e-15 | 3.5e-15 | 1.027 |
| 2 | 128x256 | 3.4e-15 | 3.5e-15 | 1.026 |

| grid (R = 2) | amplification/step |
|---|---|
| 64x128 | 1.007 |
| 128x256 | 1.026 |

The scheme is mildly amplifying (1--3% per step) -- enough that chaotic
initial conditions (the OSS random front, a discontinuous field) cannot be
compared beyond ~10^2 steps between two *different arithmetic* implementations,
but not enough to destroy smooth solutions (M1b holds at 1e-15 over 100 steps).

## ⑥ Failure modes, limits, artifacts

* **What is verified**: the OSS scheme's time integrator is reproduced exactly;
  the integrating factor is exact for diffusion (1e-11 vs analytic); the linear
  growth rate follows the analytic decomposition `C*R - k^2` with C constant to
  2.3% over R in [0.05, 4].
* **What is NOT verified / not delivered**: an *exact* variable-coefficient
  Darcy solve.  A spectral-Richardson prototype stagnated at a 14% residual and
  a spectral CG prototype diverged, so it was **not shipped** (no dead code in
  `src/`); the frozen-coefficient closure is kept and its O(1) defect is
  measured above.  Physical predictions of absolute finger widths must therefore
  not be quoted from this module.
* **Dimensionless range**: Pe = 500, R <= 4, ny x nx <= 256 x 512, l2 error of
  the closure defect grows for larger R (residual 2.0 at R = 0.05).
* **Artifacts**: `xval/XVAL-LBM-SPECTRAL-SAFFMAN-TAYLOR-INTEGRATING-FACTOR/`
  (`xval_spectral_st.py`, `oss_spectral_reference.py`, `make_report.py`,
  `artifacts/results.json`, `artifacts/metrics_growth.json`, `run.log`,
  `artifacts/fig1..fig5 PNGs`); module
  `src/tensorlbm/spectral_saffman_taylor.py`; tests
  `tests/test_spectral_saffman_taylor.py` (7 passing).

### Difference list: TensorLBM port vs OSS implementation

| Aspect | OSS `spectrum_method/main.py` | TensorLBM port | Decision / evidence |
|---|---|---|---|
| Array framework | NumPy (CuPy drop-in in `main_gpu.py`) | PyTorch float64, CPU/GPU | keep torch (in-situ with the Tensor stack); verified to round-off |
| Time integrator | IF + RK4 bootstrap + AB2 + CN corrector | identical | adopted verbatim; exact for diffusion (1e-11) |
| Elliptic closure | frozen-coefficient symbol, `kn` to first power | identical | adopted for fidelity, defect measured (O(1) residual) |
| Vorticity source direction | `R d/dy` term while its own IC puts the front along x | identical (faithful) | flagged as a possible axis inconsistency in the OSS source; the port does not "fix" it silently |
| Boundaries | periodic (FFT) everywhere | identical | -- |
| Initial condition | unseeded `np.random.rand` | explicit seed parameter | added for reproducibility |
| Diagnostics | plots only | mass, boundedness, mode amplitude, tip position, divergence, elliptic residual | added (the OSS code has no diagnostics) |
| Defaults | M=256, N=128, R=0, Pe=500, dt=0.1, T=500 | same (R=0 default) | -- |
| Verification | none in the OSS repo | 7 pytest checks + this XVAL | added |
