#!/usr/bin/env python3
"""Evidence runner for the grey-diffuse radiosity verification round (R1).

Produces, in ``outputs/thermal_radiation_r1/``:

* ``evidence.json`` — every number quoted in the report, computed here (nothing
  hand-transcribed);
* ``fig1_low_emissivity_error.png`` — relative error vs the closed-form
  two-plate flux for the legacy fixed-point solve and for the exact solve;
* ``fig2_energy_imbalance.png`` — closed-enclosure radiative energy imbalance
  Σ q_i A_i produced by the legacy equal-view-factor default as a function of
  the surface-area spread (the new code rejects these inputs).

The legacy code is not re-implemented from memory: it is loaded verbatim from
the git object ``HEAD:src/tensorlbm/thermal_radiation.py``.

Usage
-----
    PYTHONPATH=src python3 scripts/thermal_radiation_radiosity_benchmark.py
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import tensorlbm.thermal_radiation as new_mod  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "thermal_radiation_r1"
OUT.mkdir(parents=True, exist_ok=True)

SIGMA = new_mod.STEFAN_BOLTZMANN


def load_legacy_module() -> types.ModuleType:
    src = subprocess.run(
        ["git", "show", "HEAD:src/tensorlbm/thermal_radiation.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    legacy = types.ModuleType("legacy_thermal_radiation")
    legacy.__dict__["__file__"] = "<git HEAD>"
    sys.modules["legacy_thermal_radiation"] = legacy  # dataclasses need the module registered
    exec(compile(src, "<git HEAD:thermal_radiation.py>", "exec"), legacy.__dict__)
    return legacy


def q_two_plates(T1, T2, e1, e2):
    """Incropera Eq. 13.23."""
    return SIGMA * (T1**4 - T2**4) / (1.0 / e1 + 1.0 / e2 - 1.0)


def q_spheres(T1, T2, e1, e2, A1, A2):
    """Incropera Eq. 13.24."""
    return SIGMA * (T1**4 - T2**4) / (1.0 / e1 + (A1 / A2) * (1.0 / e2 - 1.0))


def surface(mod, T, e, A=1.0, alpha=0.5):
    return mod.SurfaceRadiationProps(
        temperature=T, emissivity=e, area=A, solar_absorptance=alpha
    )


def cfg(mod, surfaces, view_factors=None, max_iter=None):
    kw = {}
    if view_factors is not None:
        kw["view_factors"] = view_factors
    if max_iter is not None:
        kw["max_iter"] = max_iter
    return mod.RadiationEnclosureConfig(surfaces=surfaces, **kw)


def main() -> None:
    legacy = load_legacy_module()
    evidence = {"sigma": SIGMA, "gate_rel": 0.03, "cases": [], "figures": []}

    # --- Case table: analytic vs legacy vs current -------------------------
    plate_cases = [
        (400.0, 300.0, 0.90, 0.85),
        (800.0, 300.0, 0.90, 0.90),
        (500.0, 450.0, 0.30, 0.70),
        (400.0, 300.0, 0.20, 0.20),
        (400.0, 300.0, 0.10, 0.10),
        (400.0, 300.0, 0.05, 0.05),
        (400.0, 300.0, 0.02, 0.10),
    ]
    for T1, T2, e1, e2 in plate_cases:
        ref = q_two_plates(T1, T2, e1, e2)
        leg = legacy.compute_net_radiation_flux(
            cfg(legacy, [surface(legacy, T1, e1), surface(legacy, T2, e2)])
        ).net_flux[0]
        cur = new_mod.compute_net_radiation_flux(
            cfg(new_mod, [surface(new_mod, T1, e1), surface(new_mod, T2, e2)])
        ).net_flux[0]
        evidence["cases"].append(
            {
                "case": "two_parallel_plates",
                "T1_K": T1,
                "T2_K": T2,
                "eps1": e1,
                "eps2": e2,
                "reference_W_m2": ref,
                "legacy_W_m2": leg,
                "current_W_m2": cur,
                "legacy_rel_err": (leg - ref) / ref,
                "current_rel_err": (cur - ref) / ref,
            }
        )

    sphere_cases = [
        (500.0, 300.0, 0.90, 0.50, 1.0, 4.0),
        (500.0, 300.0, 0.05, 0.90, 1.0, 4.0),
        (400.0, 350.0, 0.60, 0.20, 0.25, 9.0),
        (500.0, 300.0, 1.00, 1.00, 1.0, 2.5),
    ]
    for T1, T2, e1, e2, A1, A2 in sphere_cases:
        vf = [[0.0, 1.0], [A1 / A2, 1.0 - A1 / A2]]
        ref = q_spheres(T1, T2, e1, e2, A1, A2)
        leg = legacy.compute_net_radiation_flux(
            cfg(legacy, [surface(legacy, T1, e1, A1), surface(legacy, T2, e2, A2)], vf)
        ).net_flux[0]
        cur = new_mod.compute_net_radiation_flux(
            cfg(new_mod, [surface(new_mod, T1, e1, A1), surface(new_mod, T2, e2, A2)], vf)
        ).net_flux[0]
        evidence["cases"].append(
            {
                "case": "concentric_spheres",
                "T1_K": T1,
                "T2_K": T2,
                "eps1": e1,
                "eps2": e2,
                "A1": A1,
                "A2": A2,
                "reference_W_m2": ref,
                "legacy_W_m2": leg,
                "current_W_m2": cur,
                "legacy_rel_err": (leg - ref) / ref,
                "current_rel_err": (cur - ref) / ref,
            }
        )

    # --- Closed-enclosure energy balance, equal-area three-surface ---------
    temps_eps = [(500.0, 0.8), (300.0, 0.6), (350.0, 0.95)]
    leg_res = legacy.compute_net_radiation_flux(
        cfg(legacy, [surface(legacy, T, e) for T, e in temps_eps])
    )
    cur_res = new_mod.compute_net_radiation_flux(
        cfg(new_mod, [surface(new_mod, T, e) for T, e in temps_eps])
    )
    scale = sum(abs(q) for q in cur_res.net_flux)
    evidence["energy_balance_equal_area"] = {
        "flux_legacy": leg_res.net_flux,
        "flux_current": cur_res.net_flux,
        "residual_legacy_W": sum(leg_res.net_flux),
        "residual_current_W": new_mod.radiative_energy_residual(cur_res.net_flux, [1.0] * 3),
        "scale_W": scale,
    }

    # --- Energy imbalance of the legacy default vs area spread -------------
    spread = []
    for A2 in [1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]:
        areas = [1.0, A2, 0.5]
        surf_leg = [surface(legacy, T, e, a) for (T, e), a in zip(temps_eps, areas)]
        r_leg = legacy.compute_net_radiation_flux(cfg(legacy, surf_leg))
        imbalance = sum(q * a for q, a in zip(r_leg.net_flux, areas))
        scale_i = sum(abs(q) * a for q, a in zip(r_leg.net_flux, areas)) or 1.0
        try:
            new_mod.compute_net_radiation_flux(
                cfg(new_mod, [surface(new_mod, T, e, a) for (T, e), a in zip(temps_eps, areas)])
            )
            verdict = "accepted"
        except ValueError:
            verdict = "ValueError (fail-closed)"
        spread.append(
            {
                "areas": areas,
                "area_ratio": max(areas) / min(areas),
                "legacy_imbalance_W": imbalance,
                "legacy_rel_imbalance": abs(imbalance) / scale_i,
                "current_verdict": verdict,
            }
        )
    evidence["area_spread"] = spread

    # --- Solar / single-surface spot checks -------------------------------
    solar = new_mod.SolarSettings(enabled=True, irradiance=1000.0, direction=(0.0, -1.0, 0.0))
    import torch

    normals = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
    evidence["solar"] = {
        "alpha_I_cos0_W_m2": new_mod.solar_flux_on_surface(normals, solar, 0.8)[0].item(),
        "expected_W_m2": 0.8 * 1000.0,
        "alpha_I_cos60_W_m2": new_mod.solar_flux_on_surface(
            torch.tensor([[math.sin(math.radians(60)), math.cos(math.radians(60)), 0.0]],
                         dtype=torch.float64),
            solar,
            0.8,
        )[0].item(),
        "expected_cos60_W_m2": 0.8 * 1000.0 * 0.5,
    }
    single = new_mod.compute_net_radiation_flux(
        cfg(new_mod, [surface(new_mod, 400.0, 0.3, 2.0)])
    )
    evidence["single_surface"] = {
        "flux_W_m2": single.net_flux[0],
        "eps_sigma_T4": 0.3 * SIGMA * 400.0**4,
    }
    evidence["emission_label"] = {
        "total_emitted_power_W": new_mod.compute_net_radiation_flux(
            cfg(new_mod, [surface(new_mod, 400.0, 0.05), surface(new_mod, 300.0, 0.05)])
        ).total_emitted_power,
        "blackbody_sigma_T4_sum_W": SIGMA * (400.0**4 + 300.0**4),
        "grey_eps_sigma_T4_sum_W": 0.05 * SIGMA * (400.0**4 + 300.0**4),
    }

    # --- Figure 1: error vs emissivity ------------------------------------
    eps_list = [0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 0.9]
    leg_err, cur_err = [], []
    for e in eps_list:
        ref = q_two_plates(400.0, 300.0, e, e)
        leg = legacy.compute_net_radiation_flux(
            cfg(legacy, [surface(legacy, 400.0, e), surface(legacy, 300.0, e)])
        ).net_flux[0]
        cur = new_mod.compute_net_radiation_flux(
            cfg(new_mod, [surface(new_mod, 400.0, e), surface(new_mod, 300.0, e)])
        ).net_flux[0]
        leg_err.append(abs((leg - ref) / ref) * 100.0)
        cur_err.append(abs((cur - ref) / ref) * 100.0)
    evidence["fig1"] = {"eps": eps_list, "legacy_rel_err_pct": leg_err,
                        "current_rel_err_pct": cur_err}

    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=110)
    ax.semilogy(eps_list, [max(v, 1e-15) for v in leg_err], "o-", label="legacy fixed point (max_iter=50)")
    ax.semilogy(eps_list, [max(v, 1e-15) for v in cur_err], "s-", label="exact linear solve (current)")
    ax.axhline(3.0, color="crimson", ls="--", lw=1.2, label="3 % acceptance gate")
    ax.axhline(15.0, color="darkred", ls=":", lw=1.0, label="15 % unacceptable")
    ax.set_xlabel("surface emissivity ε (both plates)")
    ax.set_ylabel("|q − q_analytic| / q_analytic  [%]")
    ax.set_title("Two infinite parallel plates (Incropera 13.23), T = 400/300 K")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p1 = OUT / "fig1_low_emissivity_error.png"
    fig.savefig(p1)
    plt.close(fig)
    evidence["figures"].append(str(p1.relative_to(ROOT)))

    # --- Figure 2: energy imbalance vs area ratio -------------------------
    ratios = [c["area_ratio"] for c in spread]
    imb = [abs(c["legacy_imbalance_W"]) for c in spread]
    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=110)
    ax.plot(ratios, imb, "o-", color="#c1440e")
    ax.set_xlabel("area spread max(A)/min(A)  (surfaces 1.0 : A2 : 0.5 m²)")
    ax.set_ylabel("|Σ qᵢAᵢ| of legacy default  [W]")
    ax.set_title("Equal-view-factor default: energy imbalance grows with area spread\n"
                 "(current code raises ValueError for all of these)", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p2 = OUT / "fig2_energy_imbalance.png"
    fig.savefig(p2)
    plt.close(fig)
    evidence["figures"].append(str(p2.relative_to(ROOT)))

    (OUT / "evidence.json").write_text(json.dumps(evidence, indent=2))
    print(json.dumps(
        {
            "n_cases": len(evidence["cases"]),
            "max_legacy_rel_err": max(abs(c["legacy_rel_err"]) for c in evidence["cases"]),
            "max_current_rel_err": max(abs(c["current_rel_err"]) for c in evidence["cases"]),
            "energy_residual_legacy_W": evidence["energy_balance_equal_area"]["residual_legacy_W"],
            "max_area_imbalance_W": max(imb),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()