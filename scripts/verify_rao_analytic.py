"""Analytic verification harness for ``tensorlbm.rao_analysis`` (m3688).

Produces, programmatically (no hand-copied numbers):

  artifacts/rao_analytic_verification/verification.json   -- error table
  artifacts/rao_analytic_verification/rao_vs_analytic.png -- 3-panel figure

References
----------
* |H(ω)| = |F| / √[(C − (M+A)ω²)² + (Bω)²]   (Faltinsen 1990, ch. 3)
* m_n = (A/4)·B^{(n−4)/4}·Γ(1−n/4)            (Bretschneider/ISSC spectrum)
* E[max]/2√m0 = √(2 ln N)/2                   (Longuet-Higgins 1952)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from tensorlbm.rao_analysis import (
    compute_natural_frequencies,
    compute_rao_from_hydrodynamics,
    compute_rao_from_timeseries,
    spectral_response_analysis,
)

OUT = Path("artifacts/rao_analytic_verification")
OUT.mkdir(parents=True, exist_ok=True)

HEAVE = 2
M_BODY, A_ADDED, B_DAMP, C_STIFF, F_UNIT = 1000.0, 500.0, 2000.0, 1.0e5, 1.0
OMEGA_N = math.sqrt(C_STIFF / (M_BODY + A_ADDED))

report: dict = {
    "reference": {
        "closed_form": "F/sqrt((C-(M+A)w^2)^2+(B w)^2)",
        "dof": "heave",
        "M": M_BODY,
        "A": A_ADDED,
        "B": B_DAMP,
        "C": C_STIFF,
        "omega_n": OMEGA_N,
    }
}


def heave_inputs(omega: torch.Tensor):
    n = omega.numel()
    mass = torch.zeros(6, 6, dtype=torch.float64)
    mass[HEAVE, HEAVE] = M_BODY
    added = torch.zeros(n, 6, 6, dtype=torch.float64)
    added[:, HEAVE, HEAVE] = A_ADDED
    damping = torch.zeros(n, 6, 6, dtype=torch.float64)
    damping[:, HEAVE, HEAVE] = B_DAMP
    stiffness = torch.zeros(6, 6, dtype=torch.float64)
    stiffness[HEAVE, HEAVE] = C_STIFF
    excitation = torch.zeros(n, 6, dtype=torch.float64)
    excitation[:, HEAVE] = F_UNIT
    return added, damping, stiffness, mass, excitation


def closed_form(omega: torch.Tensor) -> torch.Tensor:
    return F_UNIT / torch.sqrt(
        (C_STIFF - (M_BODY + A_ADDED) * omega**2) ** 2 + (B_DAMP * omega) ** 2
    )


# ---------------------------------------------------------------- 1. RAO sweep
omega = torch.linspace(0.2 * OMEGA_N, 5.0 * OMEGA_N, 601, dtype=torch.float64)
added, damping, stiffness, mass, excitation = heave_inputs(omega)
rao = compute_rao_from_hydrodynamics(omega, added, damping, stiffness, mass, excitation)
reference = closed_form(omega)
rel = ((rao.rao_amplitude[:, HEAVE] - reference).abs() / reference).numpy()

# Reproduce the previous float32 inverse-of-complex64 path for the same grid.
old_rel = []
for i in range(omega.numel()):
    w = float(omega[i])
    # The previous implementation could only run with every DOF populated, so
    # the comparison uses 1 kg / 1e-3 N/m dummy entries on the other five DOFs
    # (Z stays invertible and heave stays decoupled over this sweep).
    z = torch.zeros(6, 6, dtype=torch.complex128)
    z[HEAVE, HEAVE] = -(M_BODY + A_ADDED) * w**2 + 1j * w * B_DAMP + C_STIFF
    for dof in range(6):
        if dof != HEAVE:
            z[dof, dof] = -1.0 * w**2 + 1e-3
    z32 = z.to(torch.complex64)
    value = torch.abs(torch.linalg.inv(z32)[HEAVE, HEAVE].to(torch.complex128)).item() * F_UNIT
    old_rel.append(abs(value - float(reference[i])))  # absolute; relative below
old_rel = torch.tensor(old_rel)
old_rel_err = old_rel / reference

report["rao_sweep"] = {
    "n_frequency": int(omega.numel()),
    "omega_over_omega_n": [0.2, 5.0],
    "max_rel_error_float64_path": float(rel.max()),
    "mean_rel_error_float64_path": float(rel.mean()),
    "max_rel_error_previous_float32_path": float(old_rel_err.max()),
    "mean_rel_error_previous_float32_path": float(old_rel_err.mean()),
    "inactive_dofs": list(rao.inactive_dofs),
    "rao_dtype": str(rao.rao_complex.dtype),
}

# ------------------------------------------------------- 2. resonance/limits
omega_res = torch.tensor([OMEGA_N], dtype=torch.float64)
add_r, damp_r, stiff_r, mass_r, exc_r = heave_inputs(omega_res)
res = compute_rao_from_hydrodynamics(omega_res, add_r, damp_r, stiff_r, mass_r, exc_r)
peak_ref = F_UNIT / (B_DAMP * OMEGA_N)
report["resonance"] = {
    "omega_n": OMEGA_N,
    "amplitude_numeric": float(res.rao_amplitude[0, HEAVE]),
    "amplitude_closed_form": peak_ref,
    "rel_error": abs(float(res.rao_amplitude[0, HEAVE]) - peak_ref) / peak_ref,
    "phase_rad": float(res.rao_phase[0, HEAVE]),
    "phase_error_rad": abs(float(res.rao_phase[0, HEAVE]) + math.pi / 2),
}

omega_lim = torch.tensor([1e-4, 5.0e3], dtype=torch.float64)
add_l, damp_l, stiff_l, mass_l, exc_l = heave_inputs(omega_lim)
lim = compute_rao_from_hydrodynamics(omega_lim, add_l, damp_l, stiff_l, mass_l, exc_l)
report["limits"] = {
    "static_rel_error": abs(float(lim.rao_amplitude[0, HEAVE]) - F_UNIT / C_STIFF)
    / (F_UNIT / C_STIFF),
    "high_freq_rel_error": abs(
        float(lim.rao_amplitude[1, HEAVE]) - F_UNIT / ((M_BODY + A_ADDED) * 5000.0**2)
    )
    / (F_UNIT / ((M_BODY + A_ADDED) * 5000.0**2)),
}

# ---------------------------------------------------- 3. natural frequencies
mass_d = torch.diag(
    torch.tensor([2e3, 2e3, M_BODY, 4e3, 5e3, 6e3], dtype=torch.float64)
)
added_d = torch.diag(torch.tensor([1e2, 1e2, A_ADDED, 1e3, 1e3, 1e3], dtype=torch.float64))
stiff_d = torch.diag(torch.tensor([1e3, 1e3, C_STIFF, 5e5, 5e5, 5e5], dtype=torch.float64))
wn = compute_natural_frequencies(mass_d, added_d, stiff_d)
wn_ref = torch.sqrt(torch.diagonal(stiff_d) / torch.diagonal(mass_d + added_d))
report["natural_frequencies_decoupled"] = {
    "numeric": [float(v) for v in wn],
    "closed_form": [float(v) for v in wn_ref],
    "max_rel_error": float(((wn - wn_ref).abs() / wn_ref).max()),
    "heave_omega_n": OMEGA_N,
}

mass_c = torch.zeros(6, 6, dtype=torch.float64)
mass_c[HEAVE, HEAVE], mass_c[HEAVE, 4], mass_c[4, HEAVE], mass_c[4, 4] = 1000.0, 1.2e4, 1.2e4, 8.0e5
added_c = torch.zeros(6, 6, dtype=torch.float64)
added_c[HEAVE, HEAVE], added_c[4, 4] = 500.0, 2.0e5
stiff_c = torch.zeros(6, 6, dtype=torch.float64)
stiff_c[HEAVE, HEAVE], stiff_c[4, 4] = 1.0e5, 6.0e7
stiff_c[HEAVE, 4] = stiff_c[4, HEAVE] = 1.5e6
wn_c = compute_natural_frequencies(mass_c, added_c, stiff_c)
pairs = torch.tensor([HEAVE, 4])
total_c = mass_c + added_c
det_res = [
    abs(
        float(
            torch.linalg.det(
                stiff_c[pairs][:, pairs] - float(v) ** 2 * total_c[pairs][:, pairs]
            )
        )
    )
    for v in wn_c[pairs]
]
report["natural_frequencies_coupled"] = {
    "numeric": [float(v) for v in wn_c[pairs]],
    "det_C_minus_w2M": det_res,
    "det_scale": 1.0e5 * 6.0e7,
}

# ------------------------------------------- 4. Bretschneider / SRA moments
HS, T1 = 6.0, 10.0
A_COEF = 173.0 * HS**2 / T1**4
B_COEF = 691.0 / T1**4
omega_spec = torch.linspace(0.02, 40.0, 60001, dtype=torch.float64)
spectrum = A_COEF / omega_spec**5 * torch.exp(-B_COEF / omega_spec**4)
sra = spectral_response_analysis(omega_spec, spectrum, torch.ones_like(omega_spec))
m0_ref = A_COEF / 4.0 * B_COEF**-1.0 * math.gamma(1.0)
m2_ref = A_COEF / 4.0 * B_COEF**-0.5 * math.gamma(0.5)
report["spectrum_sra"] = {
    "Hs": HS,
    "T1": T1,
    "A": A_COEF,
    "B": B_COEF,
    "m0_numeric": float(sra["m0"]),
    "m0_closed_form": m0_ref,
    "m0_rel_error": abs(float(sra["m0"]) - m0_ref) / m0_ref,
    "m2_numeric": float(sra["m2"]),
    "m2_closed_form": m2_ref,
    "m2_rel_error": abs(float(sra["m2"]) - m2_ref) / m2_ref,
    "m0_over_Hs2_16": m0_ref / (HS**2 / 16.0),
    "significant_amplitude": float(sra["significant_amplitude"]),
    "significant_amplitude_closed_form": 2.0 * math.sqrt(m0_ref),
    "significant_amplitude_rel_error": abs(
        float(sra["significant_amplitude"]) - 2.0 * math.sqrt(m0_ref)
    )
    / (2.0 * math.sqrt(m0_ref)),
    "mean_period": float(sra["mean_period"]),
    "mean_period_closed_form": 2.0 * math.pi * math.sqrt(m0_ref / m2_ref),
    "rayleigh_max_ratio": float(sra["max_response_estimate"])
    / float(sra["significant_amplitude"]),
    "rayleigh_closed_form": math.sqrt(2.0 * math.log(1000)) / 2.0,
}

# ------------------------------------------------------ 5. cross-spectrum RAO
N_FFT, DT = 4096, 0.125
time = torch.arange(N_FFT, dtype=torch.float64) * DT
omega_true = 2.0 * math.pi * 128 / (N_FFT * DT)
elevation = torch.cos(omega_true * time)
response_ts = 0.5 * torch.cos(omega_true * time - 0.3)
omega_ts, amp_ts, phase_ts = compute_rao_from_timeseries(time, elevation, response_ts)
idx = int(torch.argmin(torch.abs(omega_ts - omega_true)))
# Previous behaviour: unclamped noise bins (S_eta_eta clamped at 1e-30 only).
eta_win = elevation * torch.hann_window(N_FFT, periodic=False, dtype=torch.float64)
res_win = response_ts * torch.hann_window(N_FFT, periodic=False, dtype=torch.float64)
ETA = torch.fft.rfft(eta_win)
RESP = torch.fft.rfft(res_win)
old_amp = torch.abs(ETA.conj() * RESP / torch.clamp((ETA.conj() * ETA).real, min=1e-30))
report["cross_spectrum"] = {
    "omega_true": omega_true,
    "amplitude_numeric": float(amp_ts[idx]),
    "amplitude_closed_form": 0.5,
    "amplitude_rel_error": abs(float(amp_ts[idx]) - 0.5) / 0.5,
    "phase_numeric": float(phase_ts[idx]),
    "phase_error_rad": abs(float(phase_ts[idx]) + 0.3),
    "argmax_omega_numeric": float(omega_ts[int(torch.argmax(amp_ts))]),
    "max_amplitude_numeric": float(amp_ts.max()),
    "max_amplitude_previous_path": float(old_amp.max()),
    "previous_path_bin": float(omega_ts[int(torch.argmax(old_amp))]),
}

# ------------------------------------------------------------- regression set
mass_p = torch.zeros(6, 6, dtype=torch.float64)
mass_p[HEAVE, HEAVE] = M_BODY
report["regressions"] = {
    "partial_dof_rao_raises": False,
    "partial_dof_inactive": list(rao.inactive_dofs),
    "partial_dof_natural_freq_nan_count": int(
        torch.isnan(
            compute_natural_frequencies(
                mass_p, torch.zeros(6, 6, dtype=torch.float64), stiff_d
            )
        ).sum()
    ),
    "unsorted_omega_m0": float(
        spectral_response_analysis(
            torch.linspace(3.0, 0.05, 200, dtype=torch.float64),
            A_COEF / torch.linspace(3.0, 0.05, 200, dtype=torch.float64) ** 5
            * torch.exp(-B_COEF / torch.linspace(3.0, 0.05, 200, dtype=torch.float64) ** 4),
            torch.ones(200, dtype=torch.float64),
        )["m0"]
    ),
}

# ------------------------------------------------------------------ figure
fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.6), constrained_layout=True)

ax = axes[0]
ax.plot(omega.numpy() / OMEGA_N, reference.numpy(), "-", lw=1.6, label="closed form")
ax.plot(
    omega[::20].numpy() / OMEGA_N,
    rao.rao_amplitude[::20, HEAVE].numpy(),
    "o",
    ms=3.2,
    mfc="none",
    label="Z(ω) solve (float64)",
)
ax.set_yscale("log")
ax.set_xlabel(r"$\omega/\omega_n$")
ax.set_ylabel(r"$|H|$  [m/N]")
ax.set_title("heave RAO vs analytic oscillator")
ax.axvline(1.0, color="0.6", ls=":", lw=0.9)
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

ax = axes[1]
ax.semilogy(omega.numpy() / OMEGA_N, rel, "-", lw=1.4, label="float64 path (current)")
ax.semilogy(
    omega.numpy() / OMEGA_N,
    old_rel_err.numpy(),
    "-",
    lw=1.0,
    color="tab:red",
    label="complex64 inverse (previous)",
)
ax.set_xlabel(r"$\omega/\omega_n$")
ax.set_ylabel("relative error vs closed form")
ax.set_title("accuracy of the 6x6 solve")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

ax = axes[2]
ax.plot(omega_spec.numpy(), spectrum.numpy(), lw=1.3, label=r"$S_\eta(\omega)$")
ax.plot(
    omega_spec.numpy(),
    sra["response_spectrum"].numpy(),
    lw=1.0,
    ls="--",
    label=r"$S_r$ (|RAO|=1)",
)
ax.axvline(OMEGA_N, color="0.6", ls=":", lw=0.9)
ax.set_xlim(0.0, 3.0)
ax.set_xlabel(r"$\omega$ [rad/s]")
ax.set_ylabel(r"$S$ [m$^2\cdot$s]")
ax.set_title(
    f"Bretschneider Hs={HS:g} m, T1={T1:g} s\n"
    f"m0 err {report['spectrum_sra']['m0_rel_error']:.1e}, "
    f"2√m0={report['spectrum_sra']['significant_amplitude']:.3f} m"
)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

fig.savefig(OUT / "rao_vs_analytic.png", dpi=110)
fig.savefig(OUT / "rao_vs_analytic_small.png", dpi=70)

(OUT / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
print("figure bytes:", (OUT / "rao_vs_analytic_small.png").stat().st_size)