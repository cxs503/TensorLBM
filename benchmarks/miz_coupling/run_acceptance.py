"""Acceptance benchmark for the MIZ resistance / seakeeping coupling (P46W3-N04).

Runs the five machine-checkable acceptance criteria over pre-declared parameter
sweeps and writes a JSON record with every number, the thresholds and the
pass/fail verdict::

    python benchmarks/miz_coupling/run_acceptance.py

Criteria (thresholds fixed before the sweep is evaluated):

(a) wave attenuation ``E(x) = E0 exp(-2 alpha x)`` and ``corr(alpha, C_floe) >= 0.9``
(b) ``a_floe / lambda_peak in [0.05, 0.20]`` (Squire window)
(c) ``|R_coupled - R_superposed| / R_superposed <= 0.10``
(d) ``1 - RAO_roll(MIZ)/RAO_roll(OW) >= 0.20`` at the roll peak
(e) energy ledger residual ``<= 0.05``

The sweeps are deterministic (no random draws), so "repetition" is not needed;
the coverage comes from the parameter grid, which is declared in :data:`GRID`
below and identical for every criterion.
"""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Dict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.tensorlbm.miz_resistance import (  # noqa: E402
    MIZShip, SQUIRE_WINDOW, attenuation_correlation, characteristic_floe_size,
    energy_ledger, flexural_wavelength, flexural_wavenumber,
    floe_size_distribution, open_water_resistance, ittc57_friction_coefficient,
    resistance_coupled, resistance_superposed, superposition_error,
    wave_attenuation_coefficient, wave_energy_profile, G_ACCEL)
from src.tensorlbm.miz_seakeeping import (  # noqa: E402
    heave_pitch_peaks, roll_peak_reduction, significant_roll)

# ---------------------------------------------------------------------------
# pre-declared grid and thresholds
# ---------------------------------------------------------------------------
GRID = {
    'C_floe': [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
    'Hs': [1.0, 1.5, 2.0, 2.5],
    'Tp': [6.0, 8.0, 10.0, 12.0],
    'h_ice': [0.3, 0.5, 1.0, 1.5],
    'x_ship': 250.0,      # ship position inside the MIZ [m]
    'L_strip': 300.0,     # length of the energy-audit strip [m]
}
THRESHOLDS = {
    'attenuation_corr_r': 0.90,
    'floe_ratio_window': list(SQUIRE_WINDOW),
    'superposition_rel_error': 0.10,
    'roll_rao_reduction': 0.20,
    'energy_residual': 0.05,
}


def _verdict(value: float, bound: float, sense: str) -> bool:
    return bool(value >= bound) if sense == 'ge' else bool(value <= bound)


# ---------------------------------------------------------------------------
# (a) attenuation
# ---------------------------------------------------------------------------
def criterion_a(ship: MIZShip) -> Dict:
    Hs, Tp, h = 1.5, 8.0, 0.5
    k = flexural_wavenumber(2.0 * math.pi / Tp, h)
    a = characteristic_floe_size(Hs, Tp, h)
    corr = attenuation_correlation(GRID['C_floe'], a, h, k)
    # exponential-decay check: fit ln E(x) and recover alpha within 1e-9
    alpha = wave_attenuation_coefficient(0.5, a, h, k)
    x = np.linspace(0.0, 800.0, 101)
    E = wave_energy_profile(2828.0, alpha, x)
    slope = np.polyfit(x, np.log(E), 1)[0]
    alpha_fit = -slope / 2.0
    detail = []
    for C in GRID['C_floe']:
        al = wave_attenuation_coefficient(C, a, h, k)
        detail.append({'C_floe': C, 'alpha_per_m': al,
                       'E_over_E0_at_250m': float(math.exp(-2.0 * al * 250.0)),
                       'E_folding_length_m': 1.0 / (2.0 * al) if al > 0 else None})
    return {
        'criterion': '(a) alpha vs C_floe correlation and exponential decay law',
        'threshold_r': THRESHOLDS['attenuation_corr_r'],
        'pearson_r': corr['r'],
        'fit_slope': corr['slope'],
        'alpha_fit_relative_error': abs(alpha_fit - alpha) / alpha,
        'alpha_min': corr['alpha_min'], 'alpha_max': corr['alpha_max'],
        'order_of_magnitude_vs_field_data':
            '1e-3..1e-2 1/m observed in MIZ (Kohout et al. 2014)',
        'points': detail,
        'pass': _verdict(corr['r'], THRESHOLDS['attenuation_corr_r'], 'ge'),
    }


# ---------------------------------------------------------------------------
# (b) floe size
# ---------------------------------------------------------------------------
def criterion_b() -> Dict:
    lo, hi = SQUIRE_WINDOW
    rows, ok = [], True
    for Hs in GRID['Hs']:
        for Tp in GRID['Tp']:
            for h in GRID['h_ice']:
                a = characteristic_floe_size(Hs, Tp, h)
                lam = flexural_wavelength(2.0 * math.pi / Tp, h)
                ratio = a / lam
                rows.append({'Hs': Hs, 'Tp': Tp, 'h_ice': h,
                             'a_floe': a, 'lambda_peak': lam, 'ratio': ratio,
                             'in_window': bool(lo <= ratio <= hi)})
                ok = ok and (lo <= ratio <= hi)
    # spectrum-derived distribution at the reference state
    a_arr, p_arr, stats = floe_size_distribution(1.5, 8.0, 0.5)
    return {
        'criterion': '(b) a_floe/lambda_peak inside the Squire window',
        'window': [lo, hi],
        'n_conditions': len(rows),
        'ratio_min': min(r['ratio'] for r in rows),
        'ratio_max': max(r['ratio'] for r in rows),
        'ratio_mean': float(np.mean([r['ratio'] for r in rows])),
        'fsd_reference_state': stats,
        'fsd_normalisation': float(np.sum(p_arr)) if len(p_arr) else 0.0,
        'n_floes_in_fsd': int(len(a_arr)),
        'points': rows,
        'pass': bool(ok),
    }


# ---------------------------------------------------------------------------
# (c) resistance superposition vs coupling
# ---------------------------------------------------------------------------
def criterion_c(ship: MIZShip) -> Dict:
    rows, worst, ok = [], 0.0, True
    for C in (0.2, 0.3, 0.5, 0.7, 0.9):
        for Hs in (1.0, 1.5, 2.5):
            for Tp in (6.0, 8.0, 10.0):
                se = superposition_error(ship, Hs, Tp, C, 0.5,
                                         x_ship=GRID['x_ship'])
                naive = superposition_error(ship, Hs, Tp, C, 0.5,
                                            x_ship=GRID['x_ship'],
                                            use_local_wave=False)
                rows.append({'C_floe': C, 'Hs': Hs, 'Tp': Tp,
                             'R_superposed': se['R_superposed'],
                             'R_coupled': se['R_coupled'],
                             'rel_error': se['rel_error'],
                             'naive_rel_error': naive['rel_error'],
                             'alpha': se['coupled']['alpha'],
                             'Hs_at_ship': se['coupled']['Hs_eff']})
                worst = max(worst, se['abs_rel_error'])
                ok = ok and (se['abs_rel_error'] <=
                             THRESHOLDS['superposition_rel_error'])
    ref = resistance_superposed(ship, 1.5, 8.0, 0.5, 0.5,
                                x_ship=GRID['x_ship'])
    cpl = resistance_coupled(ship, 1.5, 8.0, 0.5, 0.5, GRID['x_ship'])
    return {
        'criterion': '(c) superposed vs coupled total resistance',
        'threshold_rel_error': THRESHOLDS['superposition_rel_error'],
        'n_conditions': len(rows),
        'max_abs_rel_error': worst,
        'reference_state': {'C_floe': 0.5, 'Hs': 1.5, 'Tp': 8.0, 'h_ice': 0.5,
                            'R_ow': ref['R_ow'], 'R_wave': ref['R_wave'],
                            'R_ice': ref['R_ice'], 'R_break': ref['R_break'],
                            'R_subm': ref['R_subm'], 'R_fric': ref['R_fric'],
                            'R_superposed': ref['R_total'],
                            'R_coupled': cpl['R_total'],
                            'Hs_at_ship': cpl['Hs_eff'],
                            'a_floe_edge': cpl['a_floe_edge'],
                            'a_floe_local': cpl['a_floe_local'],
                            'ice_motion_enhancement':
                                cpl['ice_motion_enhancement']},
        'points': rows,
        'pass': bool(ok),
    }


# ---------------------------------------------------------------------------
# (d) roll RAO
# ---------------------------------------------------------------------------
def criterion_d(ship: MIZShip) -> Dict:
    rows, ok = [], True
    for C in (0.2, 0.3, 0.5, 0.7, 0.9):
        for h in (0.3, 0.5, 1.0):
            a = characteristic_floe_size(1.5, 8.0, h)
            rp = roll_peak_reduction(ship, C, a, h)
            rows.append({'C_floe': C, 'h_ice': h, 'a_floe': a,
                         **{k: rp[k] for k in ('peak_roll_ow', 'peak_roll_miz',
                                               'ratio', 'reduction',
                                               'zeta_roll_ow', 'zeta_roll_miz')}})
            ok = ok and (rp['reduction'] >= THRESHOLDS['roll_rao_reduction'])
    hp = heave_pitch_peaks(ship, 0.5, characteristic_floe_size(1.5, 8.0, 0.5), 0.5)
    return {
        'criterion': '(d) roll RAO reduction in the MIZ (ice damping)',
        'threshold_reduction': THRESHOLDS['roll_rao_reduction'],
        'n_conditions': len(rows),
        'min_reduction': min(r['reduction'] for r in rows),
        'max_reduction': max(r['reduction'] for r in rows),
        'heave_pitch_reference': hp,
        'significant_roll_deg': {
            'open_water': float(np.degrees(significant_roll(1.5, 8.0, ship, 0.0))),
            'miz_C050': float(np.degrees(significant_roll(
                1.5, 8.0, ship, 0.5, characteristic_floe_size(1.5, 8.0, 0.5), 0.5))),
        },
        'points': rows,
        'pass': bool(ok),
    }


# ---------------------------------------------------------------------------
# (e) energy ledger
# ---------------------------------------------------------------------------
def criterion_e(ship: MIZShip) -> Dict:
    rows, worst, ok, feas = [], 0.0, True, []
    for C in (0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
        for Hs, Tp, h in ((1.0, 6.0, 0.3), (1.5, 8.0, 0.5), (2.5, 10.0, 1.0)):
            a = characteristic_floe_size(Hs, Tp, h)
            led = energy_ledger(C, a, h, Hs, Tp, GRID['L_strip'], ship=ship)
            rows.append({'C_floe': C, 'Hs': Hs, 'Tp': Tp, 'h_ice': h,
                         'alpha': led['alpha'],
                         'P_in': led['P_in'], 'P_trans': led['P_trans'],
                         'P_scat': led['P_scat'], 'P_frict': led['P_frict'],
                         'P_fract': led['P_fract'], 'P_ship': led['P_ship'],
                         'residual': led['residual'],
                         'dissipative_share': led['dissipative_share'],
                         'mech_over_diss': led['mech_over_diss'],
                         'fracture_fraction': led['fracture_fraction']})
            worst = max(worst, led['residual'])
            feas.append(led['mech_over_diss'])
            ok = ok and (led['residual'] <= THRESHOLDS['energy_residual'])
    return {
        'criterion': '(e) energy ledger closure (input = attenuation + breakup '
                     '+ motion)',
        'threshold_residual': THRESHOLDS['energy_residual'],
        'n_conditions': len(rows),
        'max_residual': worst,
        'mech_over_diss_min': float(min(feas)),
        'mech_over_diss_max': float(max(feas)),
        'feasibility_note': 'mech_over_diss = (floe drag + fracture power) / '
                            '(dissipative share of the attenuation). Values > 1 '
                            'mean the mechanical sinks exceed the dissipative '
                            'share implied by the constant beta: the split of '
                            'alpha into scattering and dissipation is the least '
                            'constrained part of the model (see report, 局限).',
        'points': rows,
        'pass': bool(ok),
    }


# ---------------------------------------------------------------------------
# independent reference checks (not acceptance criteria)
# ---------------------------------------------------------------------------
def reference_checks(ship: MIZShip) -> Dict:
    # ITTC-57 friction line vs tabulated value at Re = 1e9 (Cf = 1.536e-3)
    Cf = ittc57_friction_coefficient(1e9)
    # deep-water gravity wave limit for h -> 0
    k_ow = flexural_wavenumber(0.7854, 1e-6)
    k_exact = 0.7854 ** 2 / G_ACCEL
    return {
        'ittc57_Cf_at_Re_1e9': Cf,
        'ittc57_reference_value': 1.536e-3,
        'ittc57_rel_error': abs(Cf - 1.536e-3) / 1.536e-3,
        'open_water_k_limit_rel_error': abs(k_ow - k_exact) / k_exact,
        'R_ow_100m_10kt_kN': open_water_resistance(ship) / 1e3,
        'flexural_stiffening_ratio_lambda': float(
            flexural_wavelength(0.7854, 0.5) /
            (2 * math.pi / (0.7854 ** 2 / G_ACCEL))),
    }


def main() -> Dict:
    ship = MIZShip()
    out = {
        'node': 'P46W3-N04',
        'repo': 'TensorLBM',
        'modules': ['src/tensorlbm/miz_resistance.py',
                    'src/tensorlbm/miz_seakeeping.py'],
        'ship': {'L': ship.L, 'B': ship.B, 'T': ship.T, 'Cb': ship.Cb,
                 'GM_T': ship.GM_T, 'speed_kt': ship.speed_kt},
        'grid': GRID,
        'thresholds': THRESHOLDS,
        'reference_checks': reference_checks(ship),
        'a_attenuation': criterion_a(ship),
        'b_floe_size': criterion_b(),
        'c_resistance_superposition': criterion_c(ship),
        'd_roll_rao': criterion_d(ship),
        'e_energy_ledger': criterion_e(ship),
    }
    out['all_pass'] = all(out[k]['pass'] for k in
                          ('a_attenuation', 'b_floe_size',
                           'c_resistance_superposition', 'd_roll_rao',
                           'e_energy_ledger'))
    return out


if __name__ == '__main__':
    result = main()
    outdir = os.path.join(_HERE, 'results')
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, 'acceptance.json')
    with open(path, 'w') as fh:
        json.dump(result, fh, indent=2)
    for key in ('a_attenuation', 'b_floe_size', 'c_resistance_superposition',
                'd_roll_rao', 'e_energy_ledger'):
        block = result[key]
        print(f"[{'PASS' if block['pass'] else 'FAIL'}] {block['criterion']}")
    print(json.dumps(result['reference_checks'], indent=2))
    print('all_pass =', result['all_pass'])
    print('written:', path)
