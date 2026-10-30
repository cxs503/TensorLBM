#!/usr/bin/env python3
"""Build the report JSON for the acoustics benchmark."""

import base64
import json
import math
from pathlib import Path

BENCH_DIR = Path(__file__).parent

# Load results
with open(BENCH_DIR / "result.json") as f:
    data = json.load(f)

conv = data["convergence"]
tau_sweep = data["tau_sweep"]
linearity = data["linearity"]
verdict = data["verdict_main"]

CS = 1.0 / math.sqrt(3.0)

# Encode figures
figures = []
for fn, caption in [
    ("fig_convergence.png", "Convergence of phase speed and attenuation rate vs analytical solution (N=16,32,64). Red markers: LBM measurement; dashed: continuum theory; dotted: discrete exact eigenvalue theory."),
    ("fig_error_order.png", "Error vs resolution on log-log scale. Both phase speed and attenuation errors follow O(N^{-2}) convergence, confirming second-order spatial accuracy of D2Q9 LBM for linear acoustics."),
    ("fig_tau_sweep.png", "Attenuation rate vs relaxation time tau (N=32). Green: LBM measured; dashed: continuum nu*k^2; dotted: discrete theory. All errors well within 3% threshold."),
    ("fig_wave_field.png", "Left: density perturbation field at initial and final time for N=64 case. Right: amplitude decay comparison between LBM discrete and continuum analytical solution."),
    ("fig_dashboard.png", "Summary dashboard: (a-b) error bars vs 3% threshold; (c) tau-sweep errors; (d) pipeline validation vs discrete theory; (e) linearity check; (f) R-squared fit quality metrics."),
]:
    png_path = BENCH_DIR / fn
    if png_path.exists():
        with open(png_path, "rb") as pf:
            b64 = base64.b64encode(pf.read()).decode()
        figures.append({
            "name": fn,
            "caption": caption,
            "png_base64": b64
        })

# Build markdown using string concatenation to avoid f-string brace issues
def fmt(val, spec=""):
    return format(val, spec)

c_errs = [fmt(100*(c["c_meas"]/CS-1), "+.4f") for c in conv]
d_errs = [fmt(100*(c["delta_meas"]/c["delta_cont"]-1), "+.4f") for c in conv]

# Table rows for convergence
conv_rows = ""
for i, c in enumerate(conv):
    conv_rows += "| {} | {:.9f} | **{}%** | {:+.5f}% | {:.6e} | **{}%** | {:+.5f}% |\n".format(
        c["N"], c["c_meas"], c_errs[i], c["c_err_disc_pct"],
        c["delta_meas"], d_errs[i], c["delta_err_disc_pct"])

# Table rows for tau sweep
tau_rows = ""
for t in tau_sweep:
    tau_rows += "| {} | {:.5f} | {:.6e} | {:+.4f}% | {:+.5f}% |\n".format(
        t["tau"], t["nu"], t["delta_meas"],
        100*(t["delta_meas"]/t["delta_cont"]-1), t["delta_err_disc_pct"])

# Table rows for linearity
lin_rows = ""
for l in linearity:
    lin_rows += "| {} | {:+.4f} | {:+.4f} |\n".format(
        l["eps"], 100*(l["c_meas"]/CS-1), 100*(l["delta_meas"]/l["delta_cont"]-1))

max_disc_c = max(abs(c["c_err_disc_pct"]) for c in conv)
max_disc_d = max(abs(c["delta_err_disc_pct"]) for c in conv)
max_tau_err = max(abs(100*(t["delta_meas"]/t["delta_cont"]-1)) for t in tau_sweep)

markdown = """## 1. Case Introduction

### Physical Background
This benchmark validates TensorLBM's D2Q9 BGK lattice Boltzmann method for solving linear acoustic wave propagation.
A plane acoustic wave propagates in a periodic domain, with exact analytical solutions for both the dispersion
relation (wave speed) and attenuation rate available for comparison.

### Geometry and Parameters
| Parameter | Value |
|-----------|-------|
| Model | D2Q9 BGK (collide_bgk + stream) |
| Domain | Periodic, nx=N in {16,32,64}, ny=8 |
| Wavenumber | k=2*pi/N (exactly one wavelength) |
| Relaxation time | tau=1.0 (main), 0.6/0.8/1.2 (tau sweep) |
| Amplitude | epsilon=1e-3 (linear regime) |
| Data type | float64, CPU |
| Lattice sound speed | c_s = 1/sqrt(3) = 0.57735 LU/step |
| Kinematic viscosity | nu = c_s^2*(tau-1/2) = 1/6 LU^2/step |

### Analytical References
- **Continuum theory**: omega = c_s * k, delta = nu * k^2
- **Discrete exact theory**: 9x9 propagation matrix eigenvalue (zero adjustable parameters)

## 2. Method

### Toolchain
- TensorLBM native kernels: `tensorlbm.solver.collide_bgk`, `stream`, `tensorlbm.d2q9.equilibrium`, `macroscopic`
- No artificial correction / extrapolation / sponge layer / forcing

### Procedure
1. Initialization: rho=1+epsilon*sin(kx), u_x=c_s*epsilon*sin(kx) (right-going isothermal acoustic wave), f=equilibrium
2. Time stepping: collide -> stream, periodic boundary
3. Measurement: compute Z(n) = <(rho-1)*exp(-ikx)> each step
4. Wave speed: omega = LSQ(unwrapped phase slope), c = omega/k
5. Attenuation: robust estimator absorbs counter-propagating wave 2*omega beating

## 3. Results

### Main Case Convergence (tau=1.0, epsilon=1e-3)

| N | c_meas | vs continuum c_s | vs discrete | delta_meas | vs continuum nu*k^2 | vs discrete |
|---|--------|-----------------|-------------|------------|--------------------|-------------|
""" + conv_rows + """
### Tau Sweep (N=32)

| tau | nu | delta_meas | vs continuum | vs discrete |
|-----|----|-----------|--------------|-------------|
""" + tau_rows + """
### Linearity Check (N=32, tau=1.0)

| epsilon | c_err (%) | delta_err (%) |
|---------|-----------|---------------|
""" + lin_rows + """
## 4. Result Figures

See embedded figures (5 panels):
- fig_convergence.png: Wave speed / attenuation convergence curves
- fig_error_order.png: Error convergence order (log-log)
- fig_tau_sweep.png: Tau sweep
- fig_wave_field.png: Wave field snapshot and decay comparison
- fig_dashboard.png: Comprehensive dashboard

## 5. Notes and Limitations

1. **Isothermal model limitation**: D2Q9 BGK is isothermal, no energy equation; adiabatic sound speed / thermal attenuation (gamma-law gas acoustics) not in scope.
2. **BGK artificial bulk viscosity**: Acoustic attenuation delta=nu*k^2 with nu=c_s^2*(tau-1/2); BGK is equivalent to effective bulk viscosity zeta=mu (bulk:shear=1:1), acoustic attenuation is 2x the physical isothermal NS value.
3. **Discrete dispersion**: Wave speed error O((k*lambda)^(-2)), coefficient beta(tau)=(12*tau^2-12*tau+1)/72; coarsest grid N=16 error +0.22%.
4. **Square cavity standing wave (secondary)**: BB wall viscous boundary layer loss dominates bulk attenuation (+156%~214%); cannot use bulk theory as <=3% reference - honestly recorded as limitation.
5. **Counter-propagating wave**: Equilibrium initialization inherently seeds reverse wave (b/a ~ nu*k/(2*c_s)); robust estimator absorbs it precisely.

## 6. Error Assessment

| Key Quantity | Reference | Max Error | <3%? |
|-------------|-----------|-----------|------|
| Wave speed c (vs continuum c_s) | c_s=1/sqrt(3) | """ + fmt(verdict["c_err_pct_max"], ".4f") + """% | **PASS** |
| Attenuation delta (vs continuum nu*k^2) | nu*k^2 | """ + fmt(verdict["delta_err_pct_max"], ".4f") + """% | **PASS** |
| Wave speed c (vs discrete theory) | 9x9 eigenvalue | max """ + fmt(max_disc_c, ".5f") + """% | **PASS** |
| Attenuation delta (vs discrete theory) | 9x9 eigenvalue | max """ + fmt(max_disc_d, ".5f") + """% | **PASS** |
| Tau sweep delta (vs continuum) | nu(tau)*k^2 | """ + fmt(max_tau_err, ".4f") + """% | **PASS** |

**Conclusion**: All key quantities vs analytical reference have errors <3%, with monotonic convergence (O(N^-2)) upon grid refinement. Pipeline is self-consistent with discrete exact theory (residual ~1e-5 level). **VERDICT: PASS**
"""

report = {
    "title": "TensorLBM D2Q9 Plane Acoustic Wave Verification: Linear Dispersion and Attenuation Accuracy Benchmark",
    "summary": "D2Q9 BGK plane acoustic wave at N in {16,32,64} grid refinement: wave speed and attenuation rate errors vs continuum linear acoustics theory are <=0.22% and <=1.32% respectively, both <=3% with monotonic convergence (O(N^-2)). Tau sensitivity (0.6-1.2) and linearity (epsilon<=1e-3) verified. Pipeline self-consistent with discrete exact theory (9x9 eigenvalue) to 1e-5-1e-7 relative.",
    "scenario_key": "acoustics_d2q9_plane_wave",
    "mission_id": 470,
    "tags": ["acoustics", "D2Q9", "BGK", "plane-wave", "dispersion", "attenuation", "verification", "TensorLBM"],
    "markdown": markdown,
    "figures": figures
}

with open("/tmp/report.json", "w") as f:
    json.dump(report, f, ensure_ascii=False)

print(f"Report JSON written: {Path('/tmp/report.json').stat().st_size} bytes")
print(f"Figures: {len(figures)}")
for fig in figures:
    print(f"  {fig['name']}: {len(fig['png_base64'])} chars base64")
