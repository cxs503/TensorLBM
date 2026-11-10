"""Generate figures for von Neumann stability analysis report."""

import numpy as np
import matplotlib.pyplot as plt
from tensorlbm.von_neumann import (
    BGKCollision,
    D2Q9Lattice,
    TRTCollision,
    MRTCollision,
    compute_dispersion_curve,
    von_neumann_eigenvalues,
)
from tensorlbm.von_neumann.analysis import eigenvalues_to_omega, identify_physical_modes

# Set up plotting
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 10

lat = D2Q9Lattice()
tau = 0.8

# Create collision operators
bgk = BGKCollision(lat, tau)
trt_magic = TRTCollision(lat, tau, Lambda=3.0/16.0)
mrt = MRTCollision(lat, tau_pxx=tau, tau_pxy=tau, tau_e=tau, tau_eps=tau, tau_qj=tau)

# Figure 1: Dispersion curves for BGK
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

k_dir = np.array([1.0, 0.0])
result_bgk = compute_dispersion_curve(lat, bgk, k_dir, k_max=np.pi, n_points=100)

# Acoustic mode dispersion
ax = axes[0, 0]
valid = ~np.isnan(result_bgk.omega_re_acoustic)
ax.plot(result_bgk.k_magnitudes[valid], result_bgk.omega_re_acoustic[valid], 'b-', linewidth=2, label='BGK')
ax.plot(result_bgk.k_magnitudes[valid], np.sqrt(lat.cs2) * result_bgk.k_magnitudes[valid], 'k--', label='Theory: $c_s k$')
ax.set_xlabel('|k|')
ax.set_ylabel('Re(ω)')
ax.set_title('Acoustic Mode Dispersion (BGK)')
ax.legend()
ax.grid(True, alpha=0.3)

# Shear mode dissipation
ax = axes[0, 1]
valid = ~np.isnan(result_bgk.omega_im_shear)
ax.plot(result_bgk.k_magnitudes[valid]**2, -result_bgk.omega_im_shear[valid], 'b-', linewidth=2, label='BGK')
ax.plot(result_bgk.k_magnitudes[valid]**2, bgk.nu * result_bgk.k_magnitudes[valid]**2, 'k--', label=f'Theory: νk² (ν={bgk.nu:.3f})')
ax.set_xlabel('k²')
ax.set_ylabel('-Im(ω)')
ax.set_title('Shear Mode Dissipation (BGK)')
ax.legend()
ax.grid(True, alpha=0.3)

# TRT comparison
result_trt = compute_dispersion_curve(lat, trt_magic, k_dir, k_max=np.pi, n_points=100)

ax = axes[1, 0]
valid_bgk = ~np.isnan(result_bgk.omega_re_acoustic)
valid_trt = ~np.isnan(result_trt.omega_re_acoustic)
ax.plot(result_bgk.k_magnitudes[valid_bgk], result_bgk.omega_re_acoustic[valid_bgk], 'b-', linewidth=2, label='BGK')
ax.plot(result_trt.k_magnitudes[valid_trt], result_trt.omega_re_acoustic[valid_trt], 'r-', linewidth=2, label='TRT (Λ=3/16)')
ax.plot(result_bgk.k_magnitudes[valid_bgk], np.sqrt(lat.cs2) * result_bgk.k_magnitudes[valid_bgk], 'k--', label='Theory')
ax.set_xlabel('|k|')
ax.set_ylabel('Re(ω)')
ax.set_title('Acoustic Mode: BGK vs TRT')
ax.legend()
ax.grid(True, alpha=0.3)

# Viscosity comparison
ax = axes[1, 1]
valid_bgk = ~np.isnan(result_bgk.omega_im_shear)
valid_trt = ~np.isnan(result_trt.omega_im_shear)
ax.plot(result_bgk.k_magnitudes[valid_bgk]**2, -result_bgk.omega_im_shear[valid_bgk], 'b-', linewidth=2, label=f'BGK (ν={bgk.nu:.3f})')
ax.plot(result_trt.k_magnitudes[valid_trt]**2, -result_trt.omega_im_shear[valid_trt], 'r-', linewidth=2, label=f'TRT (ν={trt_magic.nu:.3f})')
ax.plot(result_bgk.k_magnitudes[valid_bgk]**2, bgk.nu * result_bgk.k_magnitudes[valid_bgk]**2, 'b--', alpha=0.5)
ax.plot(result_trt.k_magnitudes[valid_trt]**2, trt_magic.nu * result_trt.k_magnitudes[valid_trt]**2, 'r--', alpha=0.5)
ax.set_xlabel('k²')
ax.set_ylabel('-Im(ω)')
ax.set_title('Shear Mode Dissipation: BGK vs TRT')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('artifacts/von_neumann_dispersion.png', dpi=150, bbox_inches='tight')
print("Saved: artifacts/von_neumann_dispersion.png")

# Figure 2: Stability envelope
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Max eigenvalue vs tau
tau_values = np.linspace(0.51, 2.0, 50)
k_mags = np.linspace(0.1, np.pi, 50)
angles = np.linspace(0, np.pi/2, 8)

max_lam_bgk = []
max_lam_trt = []

for tau_val in tau_values:
    bgk_temp = BGKCollision(lat, tau_val)
    trt_temp = TRTCollision(lat, tau_val, Lambda=3.0/16.0)
    
    max_bgk = 0.0
    max_trt = 0.0
    
    for angle in angles:
        k_dir_temp = np.array([np.cos(angle), np.sin(angle)])
        for km in k_mags:
            k = km * k_dir_temp
            
            eigs_bgk = von_neumann_eigenvalues(lat, bgk_temp, k)
            eigs_trt = von_neumann_eigenvalues(lat, trt_temp, k)
            
            # Non-conserved modes
            nc_bgk = np.abs(np.abs(eigs_bgk) - 1.0) > 0.01
            nc_trt = np.abs(np.abs(eigs_trt) - 1.0) > 0.01
            
            if np.any(nc_bgk):
                max_bgk = max(max_bgk, np.max(np.abs(eigs_bgk[nc_bgk])))
            if np.any(nc_trt):
                max_trt = max(max_trt, np.max(np.abs(eigs_trt[nc_trt])))
    
    max_lam_bgk.append(max_bgk)
    max_lam_trt.append(max_trt)

ax = axes[0]
ax.plot(tau_values, max_lam_bgk, 'b-', linewidth=2, label='BGK')
ax.plot(tau_values, max_lam_trt, 'r-', linewidth=2, label='TRT (Λ=3/16)')
ax.axhline(y=1.0, color='k', linestyle='--', label='Stability limit')
ax.set_xlabel('τ')
ax.set_ylabel('Max |λ| (non-conserved)')
ax.set_title('Stability Envelope')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.1])

# Eigenvalue spectrum at specific k
ax = axes[1]
k = np.array([1.0, 0.0])
eigs_bgk = von_neumann_eigenvalues(lat, BGKCollision(lat, 0.8), k)
eigs_trt = von_neumann_eigenvalues(lat, TRTCollision(lat, 0.8, Lambda=3.0/16.0), k)

ax.scatter(eigs_bgk.real, eigs_bgk.imag, c='blue', s=50, alpha=0.6, label='BGK')
ax.scatter(eigs_trt.real, eigs_trt.imag, c='red', s=50, alpha=0.6, label='TRT')

# Unit circle
theta = np.linspace(0, 2*np.pi, 100)
ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, label='|λ|=1')

ax.set_xlabel('Re(λ)')
ax.set_ylabel('Im(λ)')
ax.set_title('Eigenvalue Spectrum at k=(1,0)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('artifacts/von_neumann_stability.png', dpi=150, bbox_inches='tight')
print("Saved: artifacts/von_neumann_stability.png")

# Figure 3: Isotropy test
fig, ax = plt.subplots(figsize=(8, 6))

angles_test = np.linspace(0, 2*np.pi, 36)
k_mag = 0.01
cs_values = []

for angle in angles_test:
    k_dir_temp = np.array([np.cos(angle), np.sin(angle)])
    k = k_mag * k_dir_temp
    
    eigs = von_neumann_eigenvalues(lat, BGKCollision(lat, 1.0), k)
    modes = identify_physical_modes(eigs, lat, k)
    omega = modes["omega"]
    
    if len(modes["acoustic"]) > 0:
        acoustic_idx = modes["acoustic"][np.argmax(np.abs(omega[modes["acoustic"]].real))]
        cs_num = np.abs(omega[acoustic_idx].real) / k_mag
        cs_values.append(cs_num)
    else:
        cs_values.append(np.nan)

cs_values = np.array(cs_values)
cs_mean = np.nanmean(cs_values)
cs_theory = np.sqrt(lat.cs2)

ax.plot(np.degrees(angles_test), cs_values, 'b-', linewidth=2, label='Numerical')
ax.axhline(y=cs_theory, color='k', linestyle='--', label=f'Theory: {cs_theory:.6f}')
ax.axhline(y=cs_mean, color='r', linestyle=':', alpha=0.5, label=f'Mean: {cs_mean:.6f}')

ax.set_xlabel('Angle (degrees)')
ax.set_ylabel('Numerical sound speed')
ax.set_title('D2Q9 Isotropy Test (k=0.01)')
ax.legend()
ax.grid(True, alpha=0.3)

variation = np.nanmax(np.abs(cs_values - cs_mean)) / cs_mean
ax.text(0.05, 0.95, f'Max variation: {variation:.2e}', transform=ax.transAxes, 
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('artifacts/von_neumann_isotropy.png', dpi=150, bbox_inches='tight')
print("Saved: artifacts/von_neumann_isotropy.png")

print("\nAll figures generated successfully!")
