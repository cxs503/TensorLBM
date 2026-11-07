"""Campaign driver: gates -> DR-theta-Q_g closure -> net-power domain -> artifacts.

The driver runs, in this order

1. **Single-phase gate** (``Re_tau = 180`` and a second lower ``Re_tau``): the
   wall-shear estimator and the profile against the exact laminar channel.
2. **Interface gate**: static-bubble Young-Laplace law at a density ratio well
   above 100, with the interface width and the spurious-current level.
3. **Mass gates**: (a) closed periodic two-phase channel with no gas source --
   total mass must not drift; (b) a gas-supplied channel -- the injected mass
   ledger must close against the change of inventory.
4. **Drag-reduction closure**: ``DR(theta)`` from a wall-wettability sweep (the
   plate's gas-retention capacity sets the coverage) and ``DR(C_q)`` from a
   mass-conserving gas-supply sweep, both at matched throughput.
5. **Independent references**: exact stratified-sandwich theory evaluated on the
   *measured* void-fraction profile (at the lattice viscosity ratio) and the
   published drag-reduction envelope (sha256-anchored).
6. **Net-power ledger**: positive-net-benefit domain over ``(U, C_q)`` with the
   compressor-efficiency sensitivity (0.5 / 0.6 / 0.7) and its monotonicity
   assertion.

Every number that ends up in the JSON report is produced by the solvers; the
markdown summary and the figures are generated from the same records.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..air_lubrication import stratified_drag_reduction_from_alpha
from . import economics as econ
from . import literature
from .economics import (
    EconomicAssumptions,
    compressor_specific_work,
    efficiency_sensitivity,
    positive_net_benefit_boundary,
)
from .laplace import LaplaceConfig, run_laplace_gate
from .single_phase import run_single_phase_gate
from .two_phase import TwoPhaseConfig, run_case

#: Campaign constants (lattice units unless stated).
EOS_TEMPERATURE = 0.55 * 0.07291903717301021  # PR, T_r = 0.55
RHO_LIQUID = 7.997835999814006
RHO_GAS = 0.0195
TAU = 1.0
NY = 40
NX = 64
U_TARGET = 0.06
N_STEPS = 2500
SAMPLE_EVERY = 100


def _psi_liquid() -> float:
    """Pseudopotential of the coexisting liquid (the wetting-wall scale)."""
    import torch

    from ..multiphase import make_psi_eos

    return float(
        make_psi_eos("pr", EOS_TEMPERATURE)(torch.tensor([RHO_LIQUID], dtype=torch.float64))
    )


def _channel_cfg(**overrides: Any) -> TwoPhaseConfig:
    base = dict(
        ny=NY,
        nx=NX,
        tau=TAU,
        eos_temperature=EOS_TEMPERATURE,
        rho_liquid=RHO_LIQUID,
        rho_gas=RHO_GAS,
        U_target=U_TARGET,
        n_steps=N_STEPS,
        sample_every=SAMPLE_EVERY,
        x_boundary="periodic",
    )
    base.update(overrides)
    return TwoPhaseConfig(**base)


def _drag_reduction(tau_w: float, tau_w_1ph: float) -> float:
    return 1.0 - tau_w / tau_w_1ph


def _monotone_increasing(xs: Iterable[float], ys: Iterable[float], tol: float = 1e-6) -> bool:
    pairs = sorted(zip(xs, ys, strict=True))
    return all(b[1] >= a[1] - tol for a, b in zip(pairs, pairs[1:]))


def _saturation_stats(C_q: list[float], DR: list[float]) -> dict[str, Any]:
    """Quantify the "rise then saturate" shape of ``DR(C_q)``."""
    pairs = sorted(zip(C_q, DR, strict=True))
    dr = np.array([p[1] for p in pairs], dtype=float)
    cq = np.array([p[0] for p in pairs], dtype=float)
    gains = np.diff(dr)
    first = float(gains[0]) if gains.size else 0.0
    last = float(gains[-1]) if gains.size else 0.0
    spread = float(dr.max() - dr.min())
    plateau = spread <= 0.05
    decelerating = (first > 1e-4 and last < 0.5 * first) or plateau
    return {
        "C_q": cq.tolist(),
        "DR": dr.tolist(),
        "gains": gains.tolist(),
        "first_gain": first,
        "last_gain": last,
        "DR_spread": spread,
        "plateau": bool(plateau),
        "no_runaway_rise": bool(decelerating),
        "monotone": bool(_monotone_increasing(cq.tolist(), dr.tolist())),
    }


def _stratified_reference(record: dict[str, Any]) -> dict[str, Any]:
    """Exact stratified-sandwich DR evaluated on the measured void profile.

    Independent of the lattice: the measured ``alpha(y)`` across the half channel
    is fed to the exact solution of ``d/dy(mu du/dy) = -G`` at the *lattice*
    viscosity ratio ``mu_gas/mu_liquid = rho_gas/rho_liquid`` (equal relaxation
    times), and the resulting drag reduction is compared with the measured one.
    """
    cfg = record["config"]
    ny = cfg["ny"]
    half = (ny - 1) // 2
    alpha = np.array(record["alpha_profile"][1 : half + 1], dtype=float)
    lam = cfg.get("rho_gas", RHO_GAS) / cfg.get("rho_liquid", RHO_LIQUID)
    out = stratified_drag_reduction_from_alpha(alpha, H=float(half + 0.5), lam=lam)
    return {
        "DR_theory": float(out["dr"]),
        "lam": float(lam),
        "alpha_wall": float(alpha[0]),
        "H_half": float(half + 0.5),
    }


def _interp_dr_model(
    points_Cq: list[float],
    points_DR: list[float],
    plateau_below: bool = False,
):
    """Monotone piecewise-linear ``DR(C_q)`` closure through the measured points.

    ``plateau_below=True`` holds the lowest measured DR constant below the
    smallest measured ``C_q`` (retention-controlled layer); the default falls
    linearly to the no-gas limit ``DR(0) = 0``.
    """
    pairs = sorted((float(c), float(d)) for c, d in zip(points_Cq, points_DR, strict=True))
    if plateau_below:
        pairs = [p for p in pairs if p[0] > 0.0]
    else:
        pairs = [(0.0, 0.0)] + [p for p in pairs if p[0] > 0.0]
    cq = np.array([p[0] for p in pairs])
    dr = np.maximum.accumulate(np.array([p[1] for p in pairs]))
    dr = np.clip(dr, 0.0, 1.0)

    def model(c: float) -> float:
        return float(np.interp(float(c), cq, dr))

    model.points = (cq.tolist(), dr.tolist())  # type: ignore[attr-defined]
    return model


def _figures(
    out_dir: Path,
    records: dict[str, Any],
    dr_theta: dict[str, Any],
    dr_cq: dict[str, Any],
    net: dict[str, Any],
    sens: dict[str, Any],
) -> list[dict[str, str]]:
    """Render the campaign figures; returns name/caption records."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs: list[dict[str, str]] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1 -- single-phase gate
    sp = records["single_phase_re180"]
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    yp = np.array(sp["y_plus"])
    ax.semilogx(yp, sp["u_plus"], "o", ms=4, label="LBM (Re_tau=180)")
    order = np.argsort(yp)
    ax.semilogx(yp[order], np.array(sp["u_plus_exact"])[order], "-", label="exact laminar")
    mask = yp > 30
    if mask.any():
        ax.semilogx(yp[mask], np.array(sp["log_law_context_only"])[mask], "--", lw=1,
                    label="log law (context only)")
    ax.set_xlabel("y+")
    ax.set_ylabel("u+")
    ax.set_title(f"Single-phase gate: Cf err {sp['cf_rel_error']*100:.2f}%")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out_dir / "fig1_single_phase_profile.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    figs.append({"name": p.name, "caption": "Single-phase gate: u+ vs y+ at Re_tau=180, "
                 "exact laminar law and log law (context).", "path": str(p)})

    # 2 -- Laplace gate
    lap = records["laplace"]
    invr = [1.0 / r["radius"] for r in lap["per_radius"]]
    dp = [r["delta_p"] for r in lap["per_radius"]]
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.plot(invr, dp, "o", ms=6, label="measured")
    xx = np.linspace(0.0, max(invr) * 1.1, 10)
    ax.plot(xx, lap["sigma_eff"] * xx, "-", label=f"Laplace fit sigma={lap['sigma_eff']:.3g}")
    ax.set_xlabel("1/R")
    ax.set_ylabel("Delta p")
    ax.set_title(f"Laplace gate: ratio {lap['min_density_ratio']:.0f}, err {lap['max_rel_error']*100:.1f}%")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out_dir / "fig2_laplace.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    figs.append({"name": p.name, "caption": "Static-bubble Laplace law at density ratio > 100.",
                 "path": str(p)})

    # 3 -- DR vs theta with stratified theory
    th = [c["theta"] for c in dr_theta["cases"]]
    drv = [c["drag_reduction"] for c in dr_theta["cases"]]
    theory = [c["theory"]["DR_theory"] for c in dr_theta["cases"]]
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    order = np.argsort(th)
    ax.plot(np.array(th)[order], np.array(drv)[order], "o-", label="LBM (SC two-phase)")
    ax.plot(np.array(th)[order], np.array(theory)[order], "s--", label="stratified exact (measured alpha)")
    ax.set_xlabel("coverage theta")
    ax.set_ylabel("DR = 1 - tau_w,2ph/tau_w,1ph")
    ax.set_title("Drag reduction vs gas coverage")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out_dir / "fig3_dr_vs_theta.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    figs.append({"name": p.name, "caption": "DR vs coverage theta: lattice vs exact stratified theory.",
                 "path": str(p)})

    # 4 -- DR vs C_q with literature envelope
    cq = [c["C_q"] for c in dr_cq["cases"]]
    drc = [c["drag_reduction"] for c in dr_cq["cases"]]
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.axhspan(0.10, 0.40, color="0.85", label="published band 10-40 %")
    ax.semilogx(cq, drc, "o-", label="LBM")
    ax.set_xlabel("C_q = Q_g/(U L)")
    ax.set_ylabel("DR")
    ax.set_title("DR vs dimensionless gas flow")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = out_dir / "fig4_dr_vs_cq.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    figs.append({"name": p.name, "caption": "DR vs C_q against the published 10-40 % band.",
                 "path": str(p)})

    # 5 -- net power domain + eta_c sensitivity
    U = np.array(net["U_values"])
    Cq = np.array(net["C_q_values"])
    netgrid = np.array(net["net_power_density"])
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    cs = ax.contourf(U, Cq, netgrid.T, levels=14, cmap="RdBu_r")
    ax.contour(U, Cq, netgrid.T, levels=[0.0], colors="k", linewidths=1.5)
    fig.colorbar(cs, ax=ax, label="net power density [W/m^2]")
    ax.set_xlabel("ship speed U [m/s]")
    ax.set_ylabel("C_q")
    ax.set_yscale("log")
    ax.set_title(f"Net power domain (eta_c={net['eta_c']})")
    fig.tight_layout()
    p = out_dir / "fig5_net_power_domain.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    figs.append({"name": p.name, "caption": "Net power density over (U, C_q); black line = break-even.",
                 "path": str(p)})

    # 6 -- field snapshot (alpha and streamwise velocity)
    last = dr_theta["cases"][-1] if dr_theta["cases"] else {}
    snap = (last.get("record") or {}).get("snapshot")
    if snap:
        alpha = np.array(snap["alpha"])
        ux = np.array(snap["ux"])
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
        im0 = axes[0].imshow(alpha, origin="lower", cmap="viridis", aspect="auto")
        axes[0].set_title("gas volume fraction alpha")
        fig.colorbar(im0, ax=axes[0])
        im1 = axes[1].imshow(ux, origin="lower", cmap="coolwarm", aspect="auto")
        axes[1].set_title("streamwise velocity u_x")
        fig.colorbar(im1, ax=axes[1])
        for a in axes:
            a.set_xlabel("x")
            a.set_ylabel("y (plate at y=0)")
        fig.tight_layout()
        p = out_dir / "fig6_field_snapshot.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        figs.append({"name": p.name, "caption": "Gas-layer channel field: alpha and u_x.",
                     "path": str(p)})
    return figs


def run_campaign(
    out_dir: str | Path = "artifacts/gas_layer_in39_n06",
    device: str = "cpu",
    n_steps: int = N_STEPS,
    resume: bool = True,
) -> dict[str, Any]:
    """Run the whole campaign and write JSON + figures into ``out_dir``.

    ``resume=True`` reuses ``records_partial.json`` (written right after the
    simulations) so that the report/economics stage can be re-run without
    repeating the lattice runs.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records: dict[str, Any] = {}
    partial_path = out / "records_partial.json"
    if resume and partial_path.exists():
        cached = json.loads(partial_path.read_text())
        records["single_phase_re180"] = cached["single_phase"]
        records["single_phase_re50"] = cached.get("single_phase_re50", cached["single_phase"])
        records["laplace"] = cached["laplace"]
        records["mass_gate"] = cached["mass_gate"]
        tau_1ph = float(cached["coverage_sweep"]["tau_w_1ph"])
        theta_cases = cached["coverage_sweep"]["cases"]
        cq_cases = cached["supply_sweep"]["cases"]
        ref = {"U_measured": float(cached["coverage_sweep"].get("U_single_phase", U_TARGET))}
        records["coverage_sweep"] = {"tau_w_1ph": tau_1ph, "cases": theta_cases}
        records["supply_sweep"] = {"cases": cq_cases}
        records["reference_single_phase"] = ref
        return _assemble_report(
            out=out, records=records, tau_1ph=tau_1ph, theta_cases=theta_cases,
            cq_cases=cq_cases, ref=ref,
        )

    # --- 1. single-phase gate ------------------------------------------------
    records["single_phase_re180"] = run_single_phase_gate(
        re_tau_target=180.0, ny=130, nx=4, n_steps=2500, n_warmup=500, device=device
    )
    records["single_phase_re50"] = run_single_phase_gate(
        re_tau_target=50.0, ny=130, nx=4, n_steps=2000, n_warmup=400, device=device
    )

    # --- 2. interface gate ---------------------------------------------------
    records["laplace"] = run_laplace_gate(
        LaplaceConfig(
            nx=96, ny=96, radii=(12.0, 16.0, 20.0), n_steps=3000, device=device
        )
    )

    # --- 3. mass gates -------------------------------------------------------
    mass_closed = run_case(
        _channel_cfg(
            enable_sc=True,
            wall_psi=0.3 * _psi_liquid(),
            n_steps=max(600, n_steps // 4),
            sample_every=100,
        )
    )
    mass_source = run_case(
        _channel_cfg(
            enable_sc=True,
            source=True,
            source_rate=0.3,
            source_x_fraction=(0.1, 0.2),
            wall_psi=0.3 * _psi_liquid(),
            n_steps=max(600, n_steps // 4),
        )
    )
    records["mass_gate"] = {
        "closed_no_source": {
            "mass_relative_drift": mass_closed["mass_relative_drift"],
            "ledger_relative_error": mass_closed["ledger_relative_error"],
            "steps": mass_closed["steps"],
        },
        "supplied_gas_ledger": {
            "mass_relative_drift": mass_source["mass_relative_drift"],
            "mass_injected": mass_source["mass_injected"],
            "ledger_relative_error": mass_source["ledger_relative_error"],
            "steps": mass_source["steps"],
        },
        "gate": {
            "closed_drift_le_1e_6": bool(abs(mass_closed["mass_relative_drift"]) <= 1e-6),
            "ledger_le_1e_6": bool(abs(mass_source["ledger_relative_error"]) <= 1e-6),
        },
    }

    # --- 4a. throughput reference (single-phase, same geometry) -------------
    ref = run_case(_channel_cfg(enable_sc=False, n_steps=n_steps))
    tau_1ph = ref["wall_shear"]
    records["reference_single_phase"] = ref

    # --- 4b. coverage sweep through the wall gas-retention (wettability) ----
    psi_l = _psi_liquid()
    theta_cases = []
    for frac in (1.0, 0.7, 0.45, 0.25, 0.1):
        rec = run_case(_channel_cfg(wall_psi=frac * psi_l, n_steps=n_steps))
        alpha_profile = [
            float(v)
            for v in np.array(rec["snapshot"]["alpha"]).mean(axis=1)
        ] if rec.get("snapshot") else []
        rec["alpha_profile"] = alpha_profile
        dr = _drag_reduction(rec["wall_shear"], tau_1ph)
        theory = _stratified_reference(rec) if alpha_profile else {"DR_theory": float("nan")}
        theta_cases.append(
            {
                "wall_psi_fraction": frac,
                "wall_psi": frac * psi_l,
                "theta": rec["theta_measured"],
                "layer_thickness": rec["layer_thickness"],
                "wall_shear": rec["wall_shear"],
                "drag_reduction": dr,
                "bubbles": rec["bubbles"],
                "theory": theory,
                "record": rec,
            }
        )
    records["coverage_sweep"] = {"tau_w_1ph": tau_1ph, "cases": theta_cases}

    # --- 4c. gas-supply sweep ------------------------------------------------
    cq_cases = []
    for rate in (0.05, 0.125, 0.3, 0.75, 2.0):
        rec = run_case(
            _channel_cfg(
                wall_psi=0.3 * psi_l,
                source=True,
                source_rate=rate,
                source_x_fraction=(0.1, 0.2),
                n_steps=n_steps,
            )
        )
        cq_cases.append(
            {
                "source_rate": rate,
                "C_q": rec["C_q"],
                "theta": rec["theta_measured"],
                "layer_thickness": rec["layer_thickness"],
                "wall_shear": rec["wall_shear"],
                "drag_reduction": _drag_reduction(rec["wall_shear"], tau_1ph),
                "mass_injected": rec["mass_injected"],
                "ledger_relative_error": rec["ledger_relative_error"],
                "bubbles": rec["bubbles"],
                "record": rec,
            }
        )
    records["supply_sweep"] = {"tau_w_1ph": tau_1ph, "cases": cq_cases}

    (partial_path).write_text(
        json.dumps(
            {
                "single_phase": records["single_phase_re180"],
                "single_phase_re50": records["single_phase_re50"],
                "laplace": records["laplace"],
                "mass_gate": records["mass_gate"],
                "coverage_sweep": {
                    "tau_w_1ph": tau_1ph,
                    "U_single_phase": ref["U_measured"],
                    "cases": [
                        {k: v for k, v in c.items() if k not in ("record", "alpha_profile")}
                        for c in theta_cases
                    ],
                },
                "supply_sweep": {
                    "cases": [{k: v for k, v in c.items() if k != "record"} for c in cq_cases]
                },
            },
            indent=2,
        )
    )

def _assemble_report(
    out: Path,
    records: dict[str, Any],
    tau_1ph: float,
    theta_cases: list[dict[str, Any]],
    cq_cases: list[dict[str, Any]],
    ref: dict[str, Any],
) -> dict[str, Any]:
    """Gates, economics, figures and JSON/markdown artifacts from the records."""
    # --- 5. gates on the closure shape --------------------------------------
    thetas = [c["theta"] for c in theta_cases]
    drs = [c["drag_reduction"] for c in theta_cases]
    theta_monotone = _monotone_increasing(thetas, drs)
    sat = _saturation_stats([c["C_q"] for c in cq_cases], [c["drag_reduction"] for c in cq_cases])
    theory_err = [
        abs(c["drag_reduction"] - c["theory"]["DR_theory"]) / max(abs(c["theory"]["DR_theory"]), 1e-9)
        for c in theta_cases
        if math.isfinite(c["theory"].get("DR_theory", float("nan")))
        and abs(c["theory"]["DR_theory"]) > 1e-6
    ]
    # literature envelope at the measured C_q of the supply sweep
    lit = literature.envelope_check_sweep(
        [c["C_q"] for c in cq_cases], [c["drag_reduction"] for c in cq_cases]
    )

    # --- 6. economics --------------------------------------------------------
    dr_model = _interp_dr_model([c["C_q"] for c in cq_cases], [c["drag_reduction"] for c in cq_cases])
    # Two honest extrapolations below the smallest simulated C_q, because the
    # closure there is *not* measured:
    #   "plateau"  -- the wall retains its layer, so DR stays at the lowest
    #                 measured value (optimistic),
    #   "linear"   -- DR falls linearly to the no-gas limit DR(0) = 0
    #                 (pessimistic; the interpolant already starts at (0, 0)).
    dr_model_plateau = _interp_dr_model(
        [c["C_q"] for c in cq_cases],
        [c["drag_reduction"] for c in cq_cases],
        plateau_below=True,
    )
    measured_cq = sorted(c["C_q"] for c in cq_cases)
    # The profitable domain sits well below the experimental C_q band (see the
    # report), so the grid must reach down to 1e-5; points below the smallest
    # simulated C_q are flagged as extrapolated in the artifact.
    cq_grid = sorted(
        {1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 0.01} | set(measured_cq) | {0.2, 0.5, 1.0}
    )
    U_grid = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0]
    assumptions = EconomicAssumptions()
    net = positive_net_benefit_boundary(dr_model, cq_grid, U_grid, eta_c=0.6, assumptions=assumptions)
    sens = efficiency_sensitivity(dr_model, cq_grid, U_grid, (0.5, 0.6, 0.7), assumptions)
    net_plateau = positive_net_benefit_boundary(
        dr_model_plateau, cq_grid, U_grid, eta_c=0.6, assumptions=assumptions
    )
    sens_plateau = efficiency_sensitivity(
        dr_model_plateau, cq_grid, U_grid, (0.5, 0.6, 0.7), assumptions
    )
    records["economics"] = {
        "eta_c_sensitivity": {
            k: v for k, v in sens.items() if k != "boundaries"
        },
        "net_domain_eta_0.6": net,
        "dr_closure": {
            "C_q": dr_model.points[0],  # type: ignore[attr-defined]
            "DR": dr_model.points[1],  # type: ignore[attr-defined]
        },
        "net_domain_plateau_extrapolation": {
            "positive_area_fraction": net_plateau["positive_area_fraction"],
            "min_profitable_speed": net_plateau["min_profitable_speed"],
            "eta_c_sensitivity_positive_area": sens_plateau["positive_area_fraction"],
            "eta_c_monotone": sens_plateau["monotone_in_eta_c"],
        },
        "closure_extrapolation_note": (
            "Below the smallest simulated C_q the closure is not measured. Two "
            "bounds are reported: 'net_domain_eta_0.6' falls linearly to DR(0)=0 "
            "(pessimistic), 'net_domain_plateau_extrapolation' holds the lowest "
            "measured DR (optimistic, retention-controlled layer)."
        ),
        "measured_C_q_range": [min(measured_cq), max(measured_cq)],
        "C_q_extrapolated_below_measured": [
            c for c in cq_grid if c < min(measured_cq)
        ],
        "break_even_C_q_eta_0.6": (
            float(dr_model(max(measured_cq)) * econ.wall_shear_from_friction(8.0, 100.0)
                  * 0.6 / compressor_specific_work())
        ),
        "compressor_specific_work": compressor_specific_work(),
        "assumptions": {
            "plate_length": assumptions.plate_length,
            "draft": assumptions.draft,
            "eta_c": assumptions.eta_c,
        },
    }

    # --- 7. gate verdicts ----------------------------------------------------
    gates = {
        "single_phase_cf_le_3pct": bool(
            records["single_phase_re180"]["cf_rel_error"] <= 0.03
            and records["single_phase_re50"]["cf_rel_error"] <= 0.03
        ),
        "single_phase_profile_le_3pct": bool(
            records["single_phase_re180"]["profile_rel_error"] <= 0.03
        ),
        "laplace_le_5pct": bool(records["laplace"]["gate"]["laplace_rel_error_le_5pct"]),
        "density_ratio_ge_100": bool(records["laplace"]["gate"]["density_ratio_ge_100"]),
        "interface_width_ge_3": bool(records["laplace"]["gate"]["interface_width_ge_3"]),
        "mass_drift_le_1e_6": bool(records["mass_gate"]["gate"]["closed_drift_le_1e_6"]),
        "gas_ledger_le_1e_6": bool(records["mass_gate"]["gate"]["ledger_le_1e_6"]),
        "dr_monotone_in_theta": bool(theta_monotone),
        "dr_saturates_in_cq": bool(sat["no_runaway_rise"]),
        "literature_band": bool(
            lit["n_inside_band"] > 0
            and all(
                min(m["rel_error_to_center"] for m in r["matches"]) <= 0.15
                for r in lit["records"]
                if r["matches"]
            )
        ),
        "eta_c_monotone": bool(sens["monotone_in_eta_c"]),
    }

    # --- 8. figures and artifacts ------------------------------------------
    if not ((theta_cases[-1].get("record") or {}).get("snapshot") if theta_cases else None):
        # resumed runs carry no field snapshot; capture one cheaply so the field
        # figure is always reproducible from the artifact
        snap_rec = run_case(
            _channel_cfg(wall_psi=0.1 * _psi_liquid(), n_steps=1100, sample_every=100)
        )
        if theta_cases:
            theta_cases[-1]["record"] = {"snapshot": snap_rec.get("snapshot")}
    figures = _figures(out, records, records["coverage_sweep"], records["supply_sweep"], net, sens)

    report: dict[str, Any] = {
        "mission": "IN39-N06 gas-layer / microbubble drag reduction closure",
        "repository": "TensorLBM",
        "module": "tensorlbm.gas_layer",
        "reference_table_sha256": literature.REFERENCE_TABLE_SHA256,
        "lattice_parameters": {
            "eos": "Peng-Robinson (Yuan & Schaefer 2006 parameters)",
            "T_r": 0.55,
            "eos_temperature": EOS_TEMPERATURE,
            "G": 1.0,
            "tau": TAU,
            "U_target": U_TARGET,
            "ny": NY,
            "nx": NX,
            "rho_liquid": RHO_LIQUID,
            "rho_gas": RHO_GAS,
        },
        "gates": gates,
        "gates_passed": int(sum(bool(v) for v in gates.values())),
        "gates_total": len(gates),
        "single_phase": {
            "re180": {k: records["single_phase_re180"][k] for k in
                      ("cf_measured", "cf_exact", "cf_rel_error", "tau_w_measured",
                       "tau_w_exact", "tau_w_rel_error", "profile_rel_error",
                       "re_tau_measured", "u_bulk_measured", "u_bulk_exact")},
            "re50": {k: records["single_phase_re50"][k] for k in
                     ("cf_measured", "cf_exact", "cf_rel_error", "profile_rel_error",
                      "re_tau_measured")},
        },
        "laplace": {
            "sigma_eff": records["laplace"]["sigma_eff"],
            "max_rel_error": records["laplace"]["max_rel_error"],
            "min_density_ratio": records["laplace"]["min_density_ratio"],
            "min_interface_width": records["laplace"]["min_interface_width"],
            "per_radius": records["laplace"]["per_radius"],
        },
        "mass_gate": records["mass_gate"],
        "coverage_sweep": {
            "tau_w_1ph": tau_1ph,
            "U_single_phase": ref["U_measured"],
            "cases": [
                {k: v for k, v in c.items() if k not in ("record", "alpha_profile")}
                for c in theta_cases
            ],
        },
        "supply_sweep": {
            "cases": [{k: v for k, v in c.items() if k != "record"} for c in cq_cases],
            "saturation": sat,
        },
        "theory_comparison": {
            "note": (
                "exact stratified-sandwich theory on the measured alpha(y) at the "
                "lattice viscosity ratio; it bounds the lubrication limit. The "
                "lattice layer is not a passive stratified lubricant (Shan-Chen "
                "vapour carries spurious currents and a diffuse interface), so a "
                "disagreement here is a model limit, reported, not tuned away."
            ),
            "max_rel_error": max(theory_err) if theory_err else None,
            "mean_rel_error": float(np.mean(theory_err)) if theory_err else None,
            "n_points": len(theory_err),
            "within_15pct": bool(all(e <= 0.15 for e in theory_err)) if theory_err else False,
        },
        "literature": lit,
        "economics": records["economics"],
        "figures": [{k: f[k] for k in ("name", "caption")} for f in figures],
    }
    (out / "campaign.json").write_text(json.dumps(report, indent=2))
    (out / "SUMMARY.md").write_text(_summary_markdown(report))
    report["artifacts"] = {
        "campaign_json": str(out / "campaign.json"),
        "summary_md": str(out / "SUMMARY.md"),
        "figures": [str(out / f["name"]) for f in figures],
    }
    return report


def _summary_markdown(report: dict[str, Any]) -> str:
    g = report["gates"]
    lines = [
        "# IN39-N06 gas-layer drag-reduction closure (TensorLBM)",
        "",
        "## Gates",
        "",
        "| gate | passed |",
        "|---|---|",
    ]
    for k, v in g.items():
        lines.append(f"| {k} | {'PASS' if v else 'FAIL'} |")
    lines += [
        "",
        f"Passed {report['gates_passed']}/{report['gates_total']}.",
        "",
        "## Single-phase calibration",
        "",
        f"- Re_tau=180: Cf measured {report['single_phase']['re180']['cf_measured']:.5f} "
        f"vs exact {report['single_phase']['re180']['cf_exact']:.5f} "
        f"(error {report['single_phase']['re180']['cf_rel_error']*100:.2f} %)",
        f"- Re_tau=180 profile error {report['single_phase']['re180']['profile_rel_error']*100:.2f} %",
        "",
        "## Interface (Laplace)",
        "",
        f"- sigma = {report['laplace']['sigma_eff']:.4g}, max Laplace error "
        f"{report['laplace']['max_rel_error']*100:.2f} %, density ratio "
        f"{report['laplace']['min_density_ratio']:.0f}, interface width "
        f"{report['laplace']['min_interface_width']:.2f} cells",
        "",
        "## Drag reduction",
        "",
        "| theta | DR | theory | wall_psi fraction |",
        "|---|---|---|---|",
    ]
    for c in report["coverage_sweep"]["cases"]:
        lines.append(
            f"| {c['theta']:.3f} | {c['drag_reduction']:.3f} | "
            f"{c['theory'].get('DR_theory', float('nan')):.3f} | {c['wall_psi_fraction']:.2f} |"
        )
    lines += ["", "| C_q | theta | DR |", "|---|---|---|"]
    for c in report["supply_sweep"]["cases"]:
        lines.append(f"| {c['C_q']:.4f} | {c['theta']:.3f} | {c['drag_reduction']:.3f} |")
    lines += [
        "",
        "## Economics",
        "",
        f"- compressor specific work {report['economics']['compressor_specific_work']:.1f} J/m^3 "
        f"(eta_c = {report['economics']['assumptions']['eta_c']})",
        f"- positive-area fraction at eta_c=0.6: "
        f"{report['economics']['net_domain_eta_0.6']['positive_area_fraction']:.3f}",
        f"- eta_c sensitivity monotone: {g['eta_c_monotone']}",
    ]
    return "\n".join(lines) + "\n"