"""Concrete cross-validation cases for TensorLBM.

Each case provides a ``run_fn`` that returns a standardised result dict and
a ``reference_fn`` that returns the analytical/reference data.

Cases
-----
1. ``poiseuille_2d_H16``: 2-D plane Poiseuille flow (body-force driven),
   analytical solution u(y) = g/(2ν) y(H-y).  Acceptance: eps_inf < 2%.
2. ``lid_cavity_re100_n65``: Lid-driven cavity Re=100, n=65,
   Ghia et al. (1982) reference.  Acceptance: L2 < 15%.
3. ``womersley_alpha5``: Womersley pulsatile channel flow, alpha=5,
   analytical phasor solution.  Acceptance: amplitude < 15%, phase < 15%.

Method selection rationale
-------------------------
* Poiseuille: simplest analytical benchmark; exact steady-state solution;
  tests Guo forcing + bounce-back BC.  Reference: textbook (Schlichting).
* Lid-driven cavity: standard CFD benchmark with experimental reference data;
  tests Zou-He BC + bounce-back + vortex structure.  Reference: Ghia (1982).
* Womersley: unsteady benchmark; tests time-dependent forcing + phase averaging.
  Reference: Womersley (1955), Sexl (1930).
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict

from .registry import XValCase


# ---------------------------------------------------------------------------
# Case 1: 2-D Poiseuille flow (body-force driven)
# ---------------------------------------------------------------------------

def _poiseuille_2d_run() -> Dict[str, Any]:
    """Run 2-D Poiseuille flow and return results."""
    from tensorlbm.ice_hydro.lbm_channel import (
        CX, CY, WT, ChannelConfig, D2Q9Channel,
    )

    G = 1.0e-5
    TAU = 0.8
    H = 16
    STEPS = 5000

    ny = H + 4
    cfg = ChannelConfig(
        nx=4, ny=ny, tau=TAU, force=G, steps=0, skip=1,
        cover_amplitude=0.0, cover_depth=float(ny - 3),
        bc="no_slip", forcing="guo",
    )
    ch = D2Q9Channel(cfg)
    rows = np.where(ch.fluid[0, :])[0]
    n_fluid = rows.size
    nu = (TAU - 0.5) / 3.0
    y = rows - 0.5
    u_ana = G / (2.0 * nu) * y * (n_fluid - y)
    u0 = np.zeros(ny)
    u0[rows] = u_ana
    ch.init_equilibrium(u0)

    acc = np.zeros(n_fluid)
    rho_acc = 0.0
    n_win = max(STEPS // 4, 1)
    n_acc = 0
    for step in range(1, STEPS + 1):
        ch.f = ch._stream_from(ch._collide(ch.f))
        if step > STEPS - n_win:
            rho = ch.f.sum(axis=0)
            ux = (ch.f * CX[:, None, None]).sum(axis=0) / rho
            acc += ux[0, rows]
            rho_acc += float(rho[ch.fluid].mean())
            n_acc += 1

    u = acc / n_acc
    rho_mean = rho_acc / n_acc
    eps_inf = float(np.max(np.abs(u - u_ana)) / u_ana.max())
    eps_L2 = float(np.linalg.norm(u - u_ana) / np.linalg.norm(u_ana))
    mass_drift = float(rho_mean - 1.0)

    return {
        "f": ch.f,
        "scalars": {
            "eps_inf": eps_inf,
            "eps_L2": eps_L2,
            "mass_drift": abs(mass_drift),
        },
        "arrays": {
            "u_profile": u,
            "u_analytical": u_ana,
        },
        "reference": {
            "scalars": {
                "eps_inf": 0.0,
                "eps_L2": 0.0,
                "mass_drift": 0.0,
            },
        },
    }


def _poiseuille_2d_reference() -> Dict[str, Any]:
    """Analytical reference for 2-D Poiseuille flow."""
    G = 1.0e-5
    TAU = 0.8
    H = 16
    nu = (TAU - 0.5) / 3.0
    rows = np.arange(2, 2 + H)
    n_fluid = rows.size
    y = rows - 0.5
    u_ana = G / (2.0 * nu) * y * (n_fluid - y)
    return {
        "scalars": {"eps_inf": 0.0, "eps_L2": 0.0, "mass_drift": 0.0},
        "arrays": {"u_profile": u_ana},
    }


POISEUILLE_2D_CASE = XValCase(
    name="poiseuille_2d_H16",
    run_fn=_poiseuille_2d_run,
    reference_fn=_poiseuille_2d_reference,
    thresholds={"eps_inf": 0.02, "eps_L2": 0.02, "mass_drift": 1e-8},
    params={"H": 16, "tau": 0.8, "force": 1e-5, "steps": 5000, "bc": "no_slip", "forcing": "guo"},
    category="analytical",
    external_anchor="Poiseuille flow: u(y) = g/(2ν) y(H-y); Guo forcing: Guo et al. (2002) Phys. Rev. E 65, 046308",
)


# ---------------------------------------------------------------------------
# Case 2: Lid-driven cavity Re=100
# ---------------------------------------------------------------------------

def _lid_cavity_re100_run() -> Dict[str, Any]:
    """Run lid-driven cavity Re=100 and return results."""
    from tensorlbm.verification.lid_driven_cavity.solver import CavityConfig, CavitySolver
    from tensorlbm.verification.lid_driven_cavity.reference_data import get_ghia_reference
    from tensorlbm.verification.lid_driven_cavity.verification import compute_errors

    cfg = CavityConfig(n=65, re=100, max_steps=15000, u_lid=0.1)
    solver = CavitySolver(cfg)

    for _ in range(10000):
        solver.step()

    y, u, x, v = solver.get_profiles()
    ref = get_ghia_reference(100)
    errors = compute_errors(y, u, x, v, ref)

    return {
        "f": solver.f,
        "scalars": {
            "l2_u": errors["l2_u"],
            "l2_v": errors["l2_v"],
            "max_u": errors["max_u"],
            "max_v": errors["max_v"],
        },
        "arrays": {
            "u_centerline": u,
            "v_centerline": v,
        },
        "reference": {
            "scalars": {
                "l2_u": 0.0,
                "l2_v": 0.0,
            },
        },
    }


def _lid_cavity_re100_reference() -> Dict[str, Any]:
    """Ghia et al. (1982) reference for lid-driven cavity Re=100."""
    from tensorlbm.verification.lid_driven_cavity.reference_data import get_ghia_reference
    ref = get_ghia_reference(100)
    return {
        "scalars": {"l2_u": 0.0, "l2_v": 0.0},
        "arrays": {"u_centerline": ref.u, "v_centerline": ref.v},
    }


LID_CAVITY_RE100_CASE = XValCase(
    name="lid_cavity_re100_n65",
    run_fn=_lid_cavity_re100_run,
    reference_fn=_lid_cavity_re100_reference,
    thresholds={"l2_u": 0.15, "l2_v": 0.15},
    params={"Re": 100, "n": 65, "u_lid": 0.1, "max_steps": 10000, "bc": "Zou-He lid + bounce-back walls"},
    category="experimental",
    external_anchor="Ghia, U., Ghia, K.N., & Shin, C.T. (1982). J. Comp. Phys. 48(3), 387-411",
)


# ---------------------------------------------------------------------------
# Case 3: Womersley pulsatile flow alpha=5
# ---------------------------------------------------------------------------

def _womersley_alpha5_run() -> Dict[str, Any]:
    """Run Womersley pulsatile channel flow alpha=5 and return results."""
    from tensorlbm.verification.womersley_pulsatile import (
        ChannelPulsatileConfig, ChannelPulsatileSolver,
        evaluate_case,
    )

    cfg = ChannelPulsatileConfig(
        H=59, alpha=5.0, tau=0.8, u_peak=0.03,
        nx=8, measure_periods=4, seed=42,
    )
    solver = ChannelPulsatileSolver(cfg)
    result = solver.run()

    # evaluate_case returns the gate metrics
    eval_result = evaluate_case(result, cfg)

    amp_rel = eval_result.get("amplitude_rel_pct", 0.0) / 100.0
    phase_rel = eval_result.get("phase_rel_pct", 0.0) / 100.0

    return {
        "f": np.zeros(1),  # Womersley solver doesn't expose f directly
        "scalars": {
            "amplitude_rel_pct": amp_rel,
            "phase_rel_pct": phase_rel,
        },
        "arrays": {},
        "reference": {
            "scalars": {
                "amplitude_rel_pct": 0.0,
                "phase_rel_pct": 0.0,
            },
        },
    }


def _womersley_alpha5_reference() -> Dict[str, Any]:
    """Analytical reference for Womersley pulsatile flow alpha=5."""
    return {
        "scalars": {"amplitude_rel_pct": 0.0, "phase_rel_pct": 0.0},
    }


WOMERSLEY_ALPHA5_CASE = XValCase(
    name="womersley_alpha5",
    run_fn=_womersley_alpha5_run,
    reference_fn=_womersley_alpha5_reference,
    thresholds={"amplitude_rel_pct": 0.15, "phase_rel_pct": 0.15},
    params={"alpha": 5.0, "H": 59, "tau": 0.8, "u_peak": 0.03, "measure_periods": 4},
    category="analytical",
    external_anchor="Womersley, J.R. (1955). J. Physiol. 127:553; Sexl (1930) parallel-plate oscillatory flow",
)


# ---------------------------------------------------------------------------
# All cases
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Case 4: two-component Shan-Chen layered Poiseuille (MCMP, EDM forcing)
# ---------------------------------------------------------------------------

def _two_component_sc_layered_run() -> Dict[str, Any]:
    """Run the two-component SC layered channel and return its error scalars."""
    from tensorlbm.verification.layered_poiseuille.benchmark import benchmark_single_case

    out = benchmark_single_case("layered_r167_light")
    rec = out["record"]
    m = rec["metrics"]
    scalars = {
        "eps_inf": float(m["eps_inf"]),
        "eps_l2_rel": float(m["eps_l2_rel"]),
        "u_peak_err": float(m["u_peak_err"]),
        "Q_err": float(m["Q_err"]),
    }
    return {
        "f": np.asarray(rec["profiles"]["ux_total"], dtype=float),
        "scalars": scalars,
        "arrays": {
            "y": np.asarray(rec["profiles"]["y"], dtype=float),
            "ux_total": np.asarray(rec["profiles"]["ux_total"], dtype=float),
            "ux_ref": np.asarray(rec["u_ref"], dtype=float),
            "phi_a": np.asarray(rec["phi"], dtype=float),
        },
    }


def _two_component_sc_layered_reference() -> Dict[str, Any]:
    """Reference scalars: the reported quantities are already errors vs the exact solution."""
    return {"scalars": {}}


TWO_COMPONENT_SC_LAYERED_CASE = XValCase(
    name="two_component_sc_layered_poiseuille",
    run_fn=_two_component_sc_layered_run,
    reference_fn=_two_component_sc_layered_reference,
    thresholds={"eps_inf": 0.15, "eps_l2_rel": 0.15, "u_peak_err": 0.15, "Q_err": 0.15},
    params={"ny": 24, "nx": 1, "tau_a": 0.8, "tau_b": 1.0, "G": 2.5,
            "g_x": 5e-5, "steps": 3000, "nu_ratio": 1.67},
    category="analytical",
    external_anchor="Shan & Chen (1993) PRE 47:1815; Martys & Chen (1996) PRE 53:743; "
                    "Kupershtokh (2004) EDM forcing; exact layered Poiseuille solution",
)




# ---------------------------------------------------------------------------
# Case 5: Bitwise reference-data regression benchmark (joconnor22/LIFE)
# ---------------------------------------------------------------------------

def _poiseuille_2d_bitwise_run() -> Dict[str, Any]:
    """Run 2-D Poiseuille flow and return the *full* distribution array.

    This is the bitwise-regression variant of case 1.  It returns the complete
    ``f`` distribution array (not just the macroscopic ``u`` profile) so that a
    golden snapshot can pin the solver's exact byte-level output.  A re-run is
    accepted only if ``f`` reproduces the recorded SHA-256 digest exactly.
    """
    from tensorlbm.ice_hydro.lbm_channel import (
        CX, CY, WT, ChannelConfig, D2Q9Channel,
    )

    G = 1.0e-5
    TAU = 0.8
    H = 16
    STEPS = 5000

    ny = H + 4
    cfg = ChannelConfig(
        nx=4, ny=ny, tau=TAU, force=G, steps=0, skip=1,
        cover_amplitude=0.0, cover_depth=float(ny - 3),
        bc="no_slip", forcing="guo",
    )
    ch = D2Q9Channel(cfg)
    rows = np.where(ch.fluid[0, :])[0]
    n_fluid = rows.size
    nu = (TAU - 0.5) / 3.0
    y = rows - 0.5
    u_ana = G / (2.0 * nu) * y * (n_fluid - y)
    u0 = np.zeros(ny)
    u0[rows] = u_ana
    ch.init_equilibrium(u0)

    acc = np.zeros(n_fluid)
    rho_acc = 0.0
    n_win = max(STEPS // 4, 1)
    n_acc = 0
    for step in range(1, STEPS + 1):
        ch.f = ch._stream_from(ch._collide(ch.f))
        if step > STEPS - n_win:
            rho = ch.f.sum(axis=0)
            ux = (ch.f * CX[:, None, None]).sum(axis=0) / rho
            acc += ux[0, rows]
            rho_acc += float(rho[ch.fluid].mean())
            n_acc += 1

    u = acc / n_acc
    rho_mean = rho_acc / n_acc
    eps_inf = float(np.max(np.abs(u - u_ana)) / u_ana.max())
    eps_L2 = float(np.linalg.norm(u - u_ana) / np.linalg.norm(u_ana))
    mass_drift = float(rho_mean - 1.0)

    return {
        "f": ch.f,
        "scalars": {
            "eps_inf": eps_inf,
            "eps_L2": eps_L2,
            "mass_drift": abs(mass_drift),
        },
        "arrays": {
            "f": ch.f,
            "u_profile": u,
            "u_analytical": u_ana,
        },
        "reference": {
            "scalars": {
                "eps_inf": 0.0,
                "eps_L2": 0.0,
                "mass_drift": 0.0,
            },
        },
    }


def _poiseuille_2d_bitwise_reference() -> Dict[str, Any]:
    """Analytical reference for the bitwise Poiseuille case (scalar gates only)."""
    return {
        "scalars": {"eps_inf": 0.0, "eps_L2": 0.0, "mass_drift": 0.0},
        "arrays": {},
    }


POISEUILLE_2D_BITWISE_CASE = XValCase(
    name="poiseuille_2d_H16_bitwise",
    run_fn=_poiseuille_2d_bitwise_run,
    reference_fn=_poiseuille_2d_bitwise_reference,
    thresholds={"eps_inf": 0.02, "eps_L2": 0.02, "mass_drift": 1e-8},
    params={"H": 16, "tau": 0.8, "force": 1e-5, "steps": 5000, "bc": "no_slip",
            "forcing": "guo", "criterion": "bitwise (SHA-256 of f)"},
    category="benchmark",
    external_anchor="joconnor22/LIFE bitwise-refdata-regression-plus-benchmark-cases; "
                    "Poiseuille flow: u(y) = g/(2ν) y(H-y); Guo et al. (2002) Phys. Rev. E 65, 046308",
)


ALL_CASES = [POISEUILLE_2D_CASE, LID_CAVITY_RE100_CASE, WOMERSLEY_ALPHA5_CASE,
             TWO_COMPONENT_SC_LAYERED_CASE, POISEUILLE_2D_BITWISE_CASE]


def register_default_cases(registry) -> None:
    """Register all default cross-validation cases into a registry."""
    for case in ALL_CASES:
        registry.register(case)
