"""Figures for the P46W3-N04 MIZ coupling benchmark.

    python benchmarks/miz_coupling/make_figures.py

Writes five PNGs into ``benchmarks/miz_coupling/figures/``.  Every figure is
generated from the same module functions that the acceptance benchmark uses, so
the curves cannot drift from the numbers in ``results/acceptance.json``.
"""

from __future__ import annotations

import json
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from src.tensorlbm.miz_resistance import (  # noqa: E402
    MIZShip, SQUIRE_WINDOW, attenuation_correlation, characteristic_floe_size,
    energy_ledger, flexural_wavelength, flexural_wavenumber,
    floe_size_distribution, ice_resistance, open_water_resistance,
    resistance_coupled, resistance_superposed, wave_added_resistance,
    wave_attenuation_coefficient, wave_energy_profile)
from src.tensorlbm.miz_seakeeping import rao_curves  # noqa: E402

FIGDIR = os.path.join(_HERE, 'figures')
os.makedirs(FIGDIR, exist_ok=True)
RESDIR = os.path.join(_HERE, 'results')


def fig_attenuation() -> str:
    """(1) Wave energy decay inside the MIZ and alpha vs concentration."""
    Hs, Tp, h = 1.5, 8.0, 0.5
    k = flexural_wavenumber(2 * math.pi / Tp, h)
    a = characteristic_floe_size(Hs, Tp, h)
    E0 = 1025.0 * 9.81 * Hs ** 2 / 8.0
    x = np.linspace(0.0, 1500.0, 400)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for C in (0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
        al = wave_attenuation_coefficient(C, a, h, k)
        ax[0].plot(x, wave_energy_profile(E0, al, x) / E0,
                   label=f'C={C:.1f}, alpha={al*1e3:.2f}e-3 m$^{{-1}}$')
    ax[0].set_xlabel('distance into the ice field x [m]')
    ax[0].set_ylabel('E(x)/E$_0$  (E = $\\rho g H_s^2/8$)')
    ax[0].set_title('(a) wave energy decay  E = E$_0$ exp(-2$\\alpha$x)')
    ax[0].set_yscale('log')
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)

    Cs = np.linspace(0.02, 0.9, 40)
    al = np.array([wave_attenuation_coefficient(c, a, h, k) for c in Cs])
    corr = attenuation_correlation(list(Cs), a, h, k)
    ax[1].plot(Cs, al * 1e3, 'o-', ms=3, label='model')
    ax[1].plot(Cs, (corr['slope'] * Cs + corr['intercept']) * 1e3, '--',
               label=f"linear fit, r = {corr['r']:.4f}")
    ax[1].set_xlabel('floe concentration C$_{floe}$ [-]')
    ax[1].set_ylabel('$\\alpha$ [10$^{-3}$ m$^{-1}$]')
    ax[1].set_title('(a) attenuation coefficient vs concentration')
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIGDIR, 'fig_a_attenuation.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_floe_size() -> str:
    """(2) a_floe/lambda_peak against the Squire window, and the FSD."""
    lo, hi = SQUIRE_WINDOW
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    Tps = np.linspace(5.0, 13.0, 40)
    for h, col in zip((0.3, 0.5, 1.0, 1.5), ('C0', 'C1', 'C2', 'C3')):
        ratios = []
        for Tp in Tps:
            a = characteristic_floe_size(1.5, Tp, h)
            ratios.append(a / flexural_wavelength(2 * math.pi / Tp, h))
        ax[0].plot(Tps, ratios, color=col, label=f'h$_{{ice}}$ = {h} m')
    ax[0].axhspan(lo, hi, color='green', alpha=0.12,
                  label=f'Squire window [{lo}, {hi}]')
    ax[0].set_xlabel('peak period T$_p$ [s]  (H$_s$ = 1.5 m)')
    ax[0].set_ylabel('a$_{floe}$ / $\\lambda_{peak}$ [-]')
    ax[0].set_title('(b) characteristic floe size vs wavelength')
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    for Tp in (6.0, 8.0, 10.0):
        a_arr, p_arr, stats = floe_size_distribution(1.5, Tp, 0.5)
        if len(a_arr):
            ax[1].step(a_arr, p_arr, where='mid',
                       label=f'T$_p$ = {Tp:.0f} s, mean = '
                             f'{stats["a_mean"]:.1f} m')
    ax[1].set_xlabel('floe size a [m]')
    ax[1].set_ylabel('P(a) [1/m] (normalised)')
    ax[1].set_title('(b) spectrum-derived floe size distribution P(a)')
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIGDIR, 'fig_b_floe_size.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_resistance() -> str:
    """(3) resistance components and coupled vs superposed comparison."""
    ship = MIZShip()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    Cs = np.linspace(0.05, 0.95, 20)
    R_ow = open_water_resistance(ship) / 1e3
    R_wave = wave_added_resistance(ship, 1.5, 8.0) / 1e3
    R_ice, R_brk, R_sub = [], [], []
    a = characteristic_floe_size(1.5, 8.0, 0.5)
    for C in Cs:
        d = ice_resistance(ship, C, a, 0.5)
        R_ice.append(d['R_ice'] / 1e3)
        R_brk.append(d['R_break'] / 1e3)
        R_sub.append(d['R_subm'] / 1e3)
    ax[0].plot(Cs, R_ice, 'k-', label='R$_{ice}$ total')
    ax[0].plot(Cs, R_brk, '--', label='R$_{break}$ (fracture)')
    ax[0].plot(Cs, R_sub, ':', label='R$_{subm}$ (submergence)')
    ax[0].axhline(R_ow, color='C0', label=f'R$_{{ow}}$ = {R_ow:.0f} kN')
    ax[0].axhline(R_wave, color='C1', label=f'R$_{{wave}}$ = {R_wave:.0f} kN')
    ax[0].set_xlabel('C$_{floe}$ [-]')
    ax[0].set_ylabel('resistance [kN]')
    ax[0].set_title('(c) resistance components, H$_s$=1.5 m, T$_p$=8 s, h=0.5 m')
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)

    errs, naive, Cref = [], [], []
    for C in (0.2, 0.3, 0.5, 0.7, 0.9):
        for Hs in (1.0, 1.5, 2.5):
            sup = resistance_superposed(ship, Hs, 8.0, C, 0.5, x_ship=250.0)
            cpl = resistance_coupled(ship, Hs, 8.0, C, 0.5, 250.0)
            sup_n = resistance_superposed(ship, Hs, 8.0, C, 0.5, x_ship=250.0,
                                          use_local_wave=False)
            errs.append(abs(cpl['R_total'] - sup['R_total']) / sup['R_total'])
            naive.append(abs(cpl['R_total'] - sup_n['R_total']) / sup_n['R_total'])
            Cref.append(C)
    idx = np.arange(len(errs))
    ax[1].bar(idx - 0.2, np.array(errs) * 100, 0.4,
              label='consistent superposition (H$_s$ at ship)')
    ax[1].bar(idx + 0.2, np.array(naive) * 100, 0.4,
              label='naive superposition (edge H$_s$)')
    ax[1].axhline(10.0, color='r', ls='--', label='10 % acceptance line')
    ax[1].set_xlabel('condition index (C, H$_s$ sweep; x$_{ship}$ = 250 m)')
    ax[1].set_ylabel('|R$_{coupled}$ - R$_{superposed}$| / R [%]')
    ax[1].set_title('(c) superposition error vs coupled model')
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3, axis='y')
    fig.tight_layout()
    path = os.path.join(FIGDIR, 'fig_c_resistance.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_rao() -> str:
    """(4) roll / heave RAO in open water and in the MIZ."""
    ship = MIZShip()
    a = characteristic_floe_size(1.5, 8.0, 0.5)
    r = rao_curves(ship, 0.5, a, 0.5)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(r['freq'], r['roll_ow'], label='open water')
    ax[0].plot(r['freq'], r['roll_miz'], label='MIZ, C=0.5, h=0.5 m')
    peak_ow = r['roll_ow'].max()
    peak_miz = r['roll_miz'].max()
    ax[0].axhline(peak_ow, color='C0', ls=':', lw=1)
    ax[0].axhline(peak_miz, color='C1', ls=':', lw=1)
    ax[0].set_xlabel('frequency [Hz]')
    ax[0].set_ylabel('roll RAO [rad per m of wave amplitude]')
    ax[0].set_title(f'(d) roll RAO: peak {peak_ow:.3f} -> {peak_miz:.3f} '
                    f'({100*(1-peak_miz/peak_ow):.1f} % reduction)')
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    ax[1].plot(r['freq'], r['heave_ow'], label='heave open water')
    ax[1].plot(r['freq'], r['heave_miz'], label='heave MIZ')
    ax[1].plot(r['freq'], r['pitch_ow'], label='pitch open water')
    ax[1].plot(r['freq'], r['pitch_miz'], label='pitch MIZ')
    ax[1].set_xlabel('frequency [Hz]')
    ax[1].set_ylabel('heave [m/m] , pitch [rad/m]')
    ax[1].set_title('(d) heave and pitch RAO, open water vs MIZ')
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIGDIR, 'fig_d_roll_rao.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_energy() -> str:
    """(5) energy ledger terms and closure residual."""
    ship = MIZShip()
    Cs = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    led = [energy_ledger(C, characteristic_floe_size(1.5, 8.0, 0.5), 0.5,
                         1.5, 8.0, 300.0, ship=ship) for C in Cs]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    idx = np.arange(len(Cs))
    trans = np.array([x['P_trans'] for x in led]) / 1e3
    scat = np.array([x['P_scat'] for x in led]) / 1e3
    frict = np.array([x['P_frict'] for x in led]) / 1e3
    fract = np.array([x['P_fract'] for x in led]) / 1e3
    ship_p = np.array([x['P_ship'] for x in led])
    ax[0].bar(idx, trans, label='transmitted')
    ax[0].bar(idx, scat, bottom=trans, label='scattered by floes')
    ax[0].bar(idx, frict + fract, bottom=trans + scat,
              label='floe drag + fracture')
    ax[0].bar(idx, ship_p / 1e3, bottom=trans + scat + frict + fract,
              label='ship motion damping')
    ax[0].plot(idx, [x['P_in'] / 1e3 for x in led], 'k_', ms=18,
               label='incident P$_{in}$')
    ax[0].set_xticks(idx)
    ax[0].set_xticklabels([f'{c:.1f}' for c in Cs])
    ax[0].set_xlabel('C$_{floe}$ [-]')
    ax[0].set_ylabel('power [kW per m of wave crest]')
    ax[0].set_title('(e) energy ledger, H$_s$=1.5 m, T$_p$=8 s, L=300 m')
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3, axis='y')

    ax[1].plot(idx, [x['residual'] * 100 for x in led], 'o-',
               label='ledger residual')
    ax[1].axhline(5.0, color='r', ls='--', label='5 % acceptance line')
    ax2 = ax[1].twinx()
    ax2.plot(idx, [x['mech_over_diss'] for x in led], 's--', color='gray',
             label='mechanical / dissipative share')
    ax2.set_ylabel('(floe drag + fracture) / dissipative share [-]')
    ax[1].set_xticks(idx)
    ax[1].set_xticklabels([f'{c:.1f}' for c in Cs])
    ax[1].set_xlabel('C$_{floe}$ [-]')
    ax[1].set_ylabel('residual [%]')
    ax[1].set_title('(e) closure residual and dissipation split')
    h1, l1 = ax[1].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax[1].legend(h1 + h2, l1 + l2, fontsize=7)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIGDIR, 'fig_e_energy_ledger.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    paths = [fig_attenuation(), fig_floe_size(), fig_resistance(), fig_rao(),
             fig_energy()]
    meta = {}
    for p in paths:
        meta[os.path.basename(p)] = os.path.getsize(p)
    os.makedirs(RESDIR, exist_ok=True)
    with open(os.path.join(RESDIR, 'figures.json'), 'w') as fh:
        json.dump(meta, fh, indent=2)
    print('\n'.join(f'{p}  ({meta[os.path.basename(p)]} bytes)' for p in paths))


if __name__ == '__main__':
    main()
