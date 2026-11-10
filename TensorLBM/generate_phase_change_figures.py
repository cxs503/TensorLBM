"""Generate 4 verification figures for multiphase phase-change (p250-n05).

Figure 1: P-rho isotherms (EOS verification)
Figure 2: Phase diagram (coexistence curve)
Figure 3: Bubble growth curves (Mikic-Rohsenow)
Figure 4: Condensation film thickness (Nusselt)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorlbm.multiphase_phase_change_verification import (
    VDWEOS, maxwell_construction, mikic_rohsenow_growth,
    nusselt_film_thickness, verify_bubble_growth, verify_condensation_film,
)

OUT = Path(__file__).resolve().parents[1] / "TensorLBM" / "artifacts" / "m9226"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})

# ============================================================================
# Figure 1: P-rho isotherms
# ============================================================================
def fig1_prho_isotherms():
    eos = VDWEOS(a=1.0, b=1.0/3.0, R=1.0)
    T_c = eos.T_c
    fig, ax = plt.subplots(figsize=(8, 5))

    rho = np.linspace(0.01, 2.9, 500)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    T_r_vals = [0.5, 0.6, 0.7, 0.8, 0.9]

    for i, T_r in enumerate(T_r_vals):
        T = T_r * T_c
        p = eos.pressure(rho, T)
        ax.plot(rho, p, color=colors[i], label=f'$T_r$ = {T_r}', linewidth=1.5)

        # Mark coexistence points
        try:
            coex = maxwell_construction(eos, T)
            ax.plot([coex.rho_v, coex.rho_l], [coex.p_sat, coex.p_sat],
                    'ko', markersize=6, zorder=5)
            ax.annotate(f'$P_{{sat}}$={coex.p_sat:.4f}',
                       xy=(coex.rho_l, coex.p_sat),
                       xytext=(coex.rho_l+0.1, coex.p_sat+0.02),
                       fontsize=8, color=colors[i])
        except:
            pass

    ax.set_xlabel('Density $\\rho$')
    ax.set_ylabel('Pressure $P$')
    ax.set_title('VDW EOS: P-$\\rho$ Isotherms with Maxwell Coexistence')
    ax.legend(loc='upper left')
    ax.set_xlim(0, 3)
    ax.set_ylim(-1.5, 2)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / 'fig1_prho_isotherms.png', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {OUT / 'fig1_prho_isotherms.png'}")


# ============================================================================
# Figure 2: Phase diagram
# ============================================================================
def fig2_phase_diagram():
    eos = VDWEOS(a=1.0, b=1.0/3.0, R=1.0)
    T_c = eos.T_c
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    T_r_vals = np.linspace(0.35, 0.95, 30)
    rho_v, rho_l, p_sat = [], [], []

    for T_r in T_r_vals:
        try:
            coex = maxwell_construction(eos, T_r * T_c)
            rho_v.append(coex.rho_v)
            rho_l.append(coex.rho_l)
            p_sat.append(coex.p_sat)
        except:
            pass

    T_arr = np.array(T_r_vals[:len(rho_v)]) * T_c

    # Left: T-rho diagram
    ax1.plot(rho_v, T_arr, 'b-o', markersize=3, label='Vapor (saturated)')
    ax1.plot(rho_l, T_arr, 'r-s', markersize=3, label='Liquid (saturated)')
    ax1.fill_betweenx(T_arr, rho_v, rho_l, alpha=0.1, color='purple', label='Two-phase region')
    ax1.set_xlabel('Density $\\rho$')
    ax1.set_ylabel('Temperature $T$')
    ax1.set_title('Phase Diagram: T-$\\rho$')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: P_sat vs T
    ax2.plot(T_arr, p_sat, 'g-o', markersize=4, linewidth=2)
    ax2.set_xlabel('Temperature $T$')
    ax2.set_ylabel('Saturation Pressure $P_{sat}$')
    ax2.set_title('Clausius-Clapeyron: $P_{sat}(T)$')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / 'fig2_phase_diagram.png', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {OUT / 'fig2_phase_diagram.png'}")


# ============================================================================
# Figure 3: Bubble growth
# ============================================================================
def fig3_bubble_growth():
    result = verify_bubble_growth([0.01, 0.05, 0.1])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    t = np.array(result.time_values)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i, (dT, Ja) in enumerate(zip(result.superheat_values, result.ja_values)):
        R_anal = np.array(result.R_growth_analytical[i])
        R_sim = np.array(result.R_growth_sim[i])
        err = result.max_relative_error[i]

        ax1.plot(t, R_anal, '--', color=colors[i], linewidth=2,
                label=f'$\\Delta T$={dT} (Mikic-Rohsenow)')
        ax1.plot(t, R_sim, '-', color=colors[i], linewidth=2,
                label=f'$\\Delta T$={dT} (Stefan)')

        ax2.plot(t, np.abs(R_anal - R_sim) / np.maximum(np.abs(R_anal), 1e-15),
                '-', color=colors[i], linewidth=2,
                label=f'$\\Delta T$={dT}, Ja={Ja:.3f}, err={err:.1%}')

    ax1.set_xlabel('Time $t$')
    ax1.set_ylabel('Bubble Radius $R(t)$')
    ax1.set_title('Bubble Growth: Mikic-Rohsenow vs Stefan')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Time $t$')
    ax2.set_ylabel('Relative Error')
    ax2.set_title('Growth Rate Comparison')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    fig.tight_layout()
    fig.savefig(OUT / 'fig3_bubble_growth.png', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {OUT / 'fig3_bubble_growth.png'}")


# ============================================================================
# Figure 4: Condensation film
# ============================================================================
def fig4_condensation_film():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Film thickness vs deltaT
    deltaT = np.linspace(0.01, 2.0, 100)
    d_nu = [nusselt_film_thickness(dt, 1.0) for dt in deltaT]

    ax1.plot(deltaT, d_nu, 'b-', linewidth=2, label='Nusselt $\\delta(x=1)$')
    ax1.set_xlabel('$\\Delta T = T_{sat} - T_{wall}$')
    ax1.set_ylabel('Film Thickness $\\delta$')
    ax1.set_title('Nusselt Film Condensation: $\\delta \\sim \\Delta T^{1/4}$')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Film profile at different heights
    x_vals = np.linspace(0.01, 5.0, 100)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    dT_vals = [0.1, 0.5, 1.0, 2.0]

    for i, dT in enumerate(dT_vals):
        d_film = [nusselt_film_thickness(dT, x) for x in x_vals]
        ax2.plot(x_vals, d_film, '-', color=colors[i], linewidth=2,
                label=f'$\\Delta T$ = {dT}')

    ax2.set_xlabel('Height $x$')
    ax2.set_ylabel('Film Thickness $\\delta(x)$')
    ax2.set_title('Film Thickness Profile: $\\delta \\sim x^{1/4}$')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / 'fig4_condensation_film.png', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {OUT / 'fig4_condensation_film.png'}")


if __name__ == '__main__':
    fig1_prho_isotherms()
    fig2_phase_diagram()
    fig3_bubble_growth()
    fig4_condensation_film()
    print("\nAll 4 figures generated successfully!")
