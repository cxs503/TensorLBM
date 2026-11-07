#!/usr/bin/env python3
"""Yield-stress (Herschel-Bulkley) rheology verification: plug core, yield
surface, pressure-drop/flow-rate curve, and negative controls.

Physics of the case
-------------------
A Herschel-Bulkley material ``tau = tau_y + K*gamma_dot**n`` does not flow while
the local shear stress is below ``tau_y``.  Three analytic anchors are exercised
here, each with an independently computable exact solution:

1. **Pipe Poiseuille plug core (3-D D3Q19).**  For a body-force driven pipe the
   momentum balance fixes the stress field, ``sigma(r) = a r/2``, so the plug
   radius is ``r_c = R tau_y/tau_w`` and the flow rate follows the closed-form
   H-B law (:func:`tensorlbm.herschel_bulkley.hb_pipe_flow_rate`).
2. **Plate Couette yield surface (2-D D2Q9).**  In a pure Couette gap the stress
   is uniform, so yield is global and there is *no* interior yield surface (a
   documented degeneracy, used as a negative control).  With a moving upper
   plate plus an adverse gradient tuned so that the stress vanishes at the
   resting lower plate, the stress is ``sigma(y) = beta y``, the plug is the
   static layer ``0 <= y <= y_yield`` with ``y_yield = h tau_y/tau_w`` and
   ``tau_w = beta h``.
3. **Slush pipeline pressure drop vs flow rate.**  Ice-slurry (slush)
   parameters from :func:`tensorlbm.herschel_bulkley.slush_rheology` feed the
   same pipe solver; the resulting ``DeltaP-Q`` curve is compared point by point
   with the analytic H-B Poiseuille law and with the Newtonian curve.

Numerical method
----------------
* ``tensorlbm.herschel_bulkley_lbm``: variable-viscosity BGK with the
  Papanastasiou-regularised H-B viscosity, Guo body force with per-cell
  relaxation time, half-way bounce-back walls (stationary and moving), and
  **wall-aware** velocity gradients (solid neighbours contribute at the half-way
  distance, otherwise ``nu ~ tau_y/gamma_dot`` near a wall is biased by up to a
  factor of two -- this was the dominant error before the fix).
* The plug is *not* rigid in a regularised model (it creeps at
  ``sigma/nu_plug``), so the plug radius is (a) probed with a stated
  shear-rate threshold applied identically to the numerical and analytic
  profiles and (b) cross-checked against the exact 1-D solution of the
  *regularised* model (:func:`tensorlbm.herschel_bulkley.
  hb_regularized_reference_profile`), which isolates discretisation error from
  regularisation bias.

Run::

    python benchmarks/verified/rheology_yield/run.py [--fast]

Outputs ``result.json``, ``gates.json`` and ``figs/*.png`` next to this script.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

torch.set_num_threads(1)  # tiny grids: multithreading is pure overhead

from tensorlbm import herschel_bulkley as hb  # noqa: E402
from tensorlbm import herschel_bulkley_lbm as hl  # noqa: E402

FIGDIR = HERE / "figs"

# ---------------------------------------------------------------------------
# Case parameters (all in lattice units, rho = 1)
# ---------------------------------------------------------------------------

R_PIPE = 20.0           # nominal pipe radius
A_NEWTON = 4.5e-5       # body force for the Newtonian geometry calibration
PHI = 1.0 / 3.0         # plug fraction tau_y/tau_w
U_TARGET = 0.03         # target centreline velocity
NU_REF = {"bingham": 0.1, "hb": 0.05}
M_REG = 1.0e6           # Papanastasiou rate (plug viscosity = tau_y*m, clamped)
NU_MAX = 10.0           # plug viscosity clamp
EPS_PROBE = 0.05        # shear-rate threshold for the plug probe


def _pipe_cases(fast: bool) -> dict[str, dict]:
    """Return the rheology/body-force combinations for the pipe study."""
    bingham = hb.hb_pipe_design_case(1.0, PHI, U_TARGET, NU_REF["bingham"], R_PIPE)
    hb06 = hb.hb_pipe_design_case(0.6, PHI, U_TARGET, NU_REF["hb"], R_PIPE)
    a_b = bingham["a"]
    a_h = hb06["a"]
    cases = {
        "bingham_a1.00": {"rheo": bingham["rheo"], "a": a_b, "tag": "bingham"},
        "bingham_a0.70": {"rheo": bingham["rheo"], "a": 0.70 * a_b, "tag": "bingham"},
        "bingham_a1.45": {"rheo": bingham["rheo"], "a": 1.45 * a_b, "tag": "bingham"},
        "hb06_a1.00": {"rheo": hb06["rheo"], "a": a_h, "tag": "hb"},
        "hb06_a0.75": {"rheo": hb06["rheo"], "a": 0.75 * a_h, "tag": "hb"},
        "hb06_a1.35": {"rheo": hb06["rheo"], "a": 1.35 * a_h, "tag": "hb"},
    }
    if fast:
        cases = {k: cases[k] for k in ("bingham_a1.00", "hb06_a1.00")}
    return cases


def _warm_start(rr: torch.Tensor, a: float, rheo, m: float, nu_max: float) -> torch.Tensor:
    """1-D regularised reference velocity mapped onto the cross-section."""
    r = rr.flatten().to(torch.float64)
    order = torch.argsort(r)
    coord = r[order]
    ref = hb.hb_regularized_reference_profile(coord, a * coord / 2.0, rheo, m=m, nu_max=nu_max)
    u = torch.empty_like(coord)
    u[order] = ref["u"]
    return u.reshape(rr.shape).to(torch.float32)


def run_pipe(R: float, a: float, rheo, m: float, nu_max: float, n_steps: int,
             warm: bool, tol: float = 1.0e-10) -> dict:
    """Run one pipe case and return the measured diagnostics."""
    n = 2 * int(math.ceil(R)) + 4
    _, rr = hl._pipe_solid_mask(n, R)
    u_init = _warm_start(rr, a, rheo, m, nu_max) if warm else None
    t0 = time.time()
    res = hl.run_hb_pipe(R, rheo, a, n_steps=n_steps, n_axial=2, m=m, nu_max=nu_max,
                         check_every=500, tol=tol, u_init=u_init)
    wall = time.time() - t0
    q_num = res["flow_rate"] / 2.0          # n_axial = 2 identical planes
    meas = hl.measure_pipe_plug_radius(res["ux"], res["rr"], res["solid"], rheo,
                                       m=m, nu_max=nu_max, eps=EPS_PROBE)
    coord = torch.tensor(meas["profile_r"], dtype=torch.float64)
    u_num = torch.tensor(meas["profile_u"], dtype=torch.float64)
    return {"res": res, "meas": meas, "coord": coord, "u_num": u_num,
            "q_num": q_num, "wall_time_s": wall}


def analyse_pipe(name: str, R: float, a: float, rheo, m: float, nu_max: float,
                 n_steps: int, warm: bool, R_eff: float) -> dict:
    """Full analytic comparison for one pipe case."""
    out = run_pipe(R, a, rheo, m, nu_max, n_steps, warm)
    tau_w = a * R_eff / 2.0
    q_an = hb.hb_pipe_flow_rate(tau_w, R_eff, rheo)
    yc = hb.hb_pipe_yield_radius(R_eff, tau_w, rheo.tau_y)
    coord, u_num = out["coord"], out["u_num"]
    u_an = hb.hb_pipe_velocity_profile(coord, R_eff, tau_w, rheo)
    reg = hb.hb_regularized_reference_profile(coord, a * coord / 2.0, rheo, m=m, nu_max=nu_max)
    r_probe_num = hl.probe_plug_radius(coord, u_num, eps=EPS_PROBE)
    r_probe_an = hl.probe_plug_radius(coord, u_an, eps=EPS_PROBE)
    r_probe_reg = hl.probe_plug_radius(coord, reg["u"], eps=EPS_PROBE)
    # profile error over the yielded annulus (r >= r_c), where the H-B law holds
    yielded = coord >= yc["r_c"]
    prof_err = (u_num[yielded] - u_an[yielded]) / torch.clamp(u_an[yielded], min=1e-30)
    payload = {
        "case": name,
        "R_nominal": R,
        "R_eff_used": R_eff,
        "body_force_a": a,
        "rheology": rheo.as_dict(),
        "regularisation": {"m": m, "nu_max": nu_max},
        "steps": out["res"]["steps"],
        "converged": bool(out["res"]["converged"]),
        "wall_time_s": round(out["wall_time_s"], 1),
        "flow_rate_lbm": out["q_num"],
        "flow_rate_analytic": q_an,
        "flow_rate_rel_err": out["q_num"] / q_an - 1.0 if q_an else None,
        "tau_w_momentum_balance": tau_w,
        "plug_radius_momentum_balance": yc["r_c"],
        "plug_fraction_momentum_balance": yc["plug_fraction"],
        "yielded_annulus_thickness": yc["yielded_annulus_thickness"],
        "plug_radius_probe_lbm": r_probe_num,
        "plug_radius_probe_analytic": r_probe_an,
        "plug_radius_probe_regularised_reference": r_probe_reg,
        "plug_radius_rel_err_vs_analytic_probe": (
            None if (r_probe_num is None or r_probe_an is None) else r_probe_num / r_probe_an - 1.0
        ),
        "plug_radius_rel_err_vs_regularised_reference": (
            None if (r_probe_num is None or r_probe_reg is None) else r_probe_num / r_probe_reg - 1.0
        ),
        "centreline_velocity_lbm": float(u_num[0]),
        "centreline_velocity_analytic": float(u_an[0]),
        "centreline_rel_err": float(u_num[0] / u_an[0] - 1.0),
        "profile_max_rel_err_yielded": float(prof_err.abs().max()) if len(prof_err) else None,
        "profile_rms_rel_err_yielded": (
            float(torch.sqrt((prof_err**2).mean())) if len(prof_err) else None
        ),
        "profile_r": coord.tolist(),
        "profile_u_lbm": u_num.tolist(),
        "profile_u_analytic": u_an.tolist(),
        "profile_u_regularised": reg["u"].tolist(),
    }
    return payload


# ---------------------------------------------------------------------------
# Plane-channel (plate) cases
# ---------------------------------------------------------------------------


def analyse_couette(name: str, h: int, rheo, beta: float, m: float, nu_max: float,
                    n_steps: int) -> dict:
    """Adverse-gradient Couette-Poiseuille: yield surface at ``h*tau_y/tau_w``."""
    res = hl.run_hb_couette(h, rheo, beta, n_steps=n_steps, m=m, nu_max=nu_max,
                            check_every=500, tol=1.0e-11)
    u_prof = res["u_profile"]
    y = res["y"]
    meas = hl.measure_yield_surface(u_prof, y, rheo, m=m, nu_max=nu_max, eps=EPS_PROBE)
    yy = torch.tensor(meas["profile_y"], dtype=torch.float64)
    u_an = hb.hb_couette_adverse_profile(yy, beta, float(h), rheo)
    y_probe_num = hl.probe_plug_radius(yy, torch.tensor(meas["profile_u"], dtype=torch.float64),
                                       eps=EPS_PROBE)
    y_probe_an = hl.probe_plug_radius(yy, u_an, eps=EPS_PROBE)
    meta = res["analytic_meta"]
    reg = hb.hb_regularized_reference_profile(yy, beta * yy, rheo, m=m, nu_max=nu_max)
    y_probe_reg = hl.probe_plug_radius(yy, reg["u"], eps=EPS_PROBE)
    return {
        "case": name,
        "gap_h": h,
        "beta_adverse_gradient": beta,
        "rheology": rheo.as_dict(),
        "regularisation": {"m": m, "nu_max": nu_max},
        "wall_speed_U": meta["U"],
        "tau_w_analytic": meta["tau_w"],
        "y_yield_analytic": meta["y_yield"],
        "y_yield_analytic_fraction": meta["plug_fraction"],
        "y_yield_probe_lbm": y_probe_num,
        "y_yield_probe_analytic": y_probe_an,
        "y_yield_probe_regularised_reference": y_probe_reg,
        "y_yield_rel_err_vs_analytic_probe": (
            None if (y_probe_num is None or y_probe_an is None) else y_probe_num / y_probe_an - 1.0
        ),
        "y_yield_rel_err_vs_regularised_reference": (
            None if (y_probe_num is None or y_probe_reg is None) else y_probe_num / y_probe_reg - 1.0
        ),
        "gamma_dot_wall_analytic": meta["gamma_dot_wall"],
        "u_max_lbm": meas["u_max"],
        "u_max_analytic": float(u_an.max()),
        "u_max_rel_err": meas["u_max"] / float(u_an.max()) - 1.0,
        "converged": bool(res["converged"]),
        "steps": res["steps"],
        "profile_y": meas["profile_y"],
        "profile_u_lbm": meas["profile_u"],
        "profile_u_analytic": u_an.tolist(),
    }


def run_negative_controls(fast: bool) -> dict:
    """Three negative controls: no yield stress -> no plug/no yield surface."""
    out: dict[str, dict] = {}

    # (1) Newtonian pipe: tau_y = 0 -> the plug must vanish
    rheo_n = hb.validate_rheology(0.0, NU_REF["bingham"], 1.0)
    n = 2 * int(math.ceil(R_PIPE)) + 4
    _, rr = hl._pipe_solid_mask(n, R_PIPE)
    res = hl.run_hb_pipe(R_PIPE, rheo_n, A_NEWTON, n_steps=3000 if fast else 8000,
                         n_axial=2, m=1.0, nu_max=None, check_every=500, tol=1.0e-10)
    meas = hl.measure_pipe_plug_radius(res["ux"], res["rr"], res["solid"], rheo_n,
                                       m=1.0, nu_max=None, eps=EPS_PROBE)
    q = res["flow_rate"] / 2.0
    # hydraulic radius of the digital staircase pipe, from the flow rate only
    r_eff = (8.0 * NU_REF["bingham"] * q / (math.pi * A_NEWTON)) ** 0.25
    coord = torch.tensor(meas["profile_r"], dtype=torch.float64)
    u_an = hb.hb_pipe_velocity_profile(coord, r_eff, A_NEWTON * r_eff / 2.0, rheo_n)
    prof_err = (torch.tensor(meas["profile_u"], dtype=torch.float64) - u_an)
    prof_err = (prof_err / torch.clamp(u_an, min=1e-30)).abs()
    out["newtonian_pipe"] = {
        "case": "negative_control_newtonian_pipe",
        "rheology": rheo_n.as_dict(),
        "R_nominal": R_PIPE,
        "R_eff_from_flow_rate": r_eff,
        "flow_rate_lbm": q,
        "flow_rate_analytic": hb.hb_pipe_flow_rate(A_NEWTON * r_eff / 2.0, r_eff, rheo_n),
        "plug_radius_probe": meas["r_c"],
        "plug_radius_over_R": None if meas["r_c"] is None else meas["r_c"] / r_eff,
        "profile_max_rel_err": float(prof_err[: int(len(prof_err) * 0.95)].max()),
        "converged": bool(res["converged"]),
        "steps": res["steps"],
    }

    # (2) pure plane Couette (uniform stress): no interior yield surface
    h = 40
    for tag, tau_y in (("zero_yield", 0.0), ("yield_bingham", 2.0e-4)):
        rheo = hb.validate_rheology(tau_y, 0.1, 1.0)
        U = 0.02
        res = hl.run_hb_channel(h, rheo, a=0.0, u_wall=U, n_steps=4000 if fast else 12000,
                                m=M_REG, nu_max=NU_MAX, check_every=500, tol=1.0e-12)
        meas = hl.measure_yield_surface(res["u_profile"], res["y"], rheo, m=M_REG,
                                        nu_max=NU_MAX, eps=EPS_PROBE)
        u = res["u_profile"][1:-1].to(torch.float64)
        yy = res["y"][1:-1].to(torch.float64)
        g = torch.abs(hl.wall_aware_gradient(u.unsqueeze(1), torch.ones_like(u).unsqueeze(1).bool(), 0).squeeze(1))
        tau_w = tau_y + 0.1 * (U / h)  # from the imposed plate speed
        out[f"pure_couette_{tag}"] = {
            "case": f"negative_control_pure_couette_{tag}",
            "rheology": rheo.as_dict(),
            "gap_h": h,
            "wall_speed": U,
            "tau_w_imposed": tau_w,
            "yielded_everywhere": tau_w > tau_y,
            "yield_surface_probe": meas["y_yield"],
            "shear_rate_uniformity_max_min_ratio": float(g.max() / torch.clamp(g.min(), min=1e-30)),
            "gamma_dot_mean_lbm": float(g.mean()),
            "gamma_dot_analytic": (tau_w - tau_y) / rheo.K,
            "profile_linearity_max_rel_err": float(
                (u - U * yy / h).abs().max() / (U * 1.0)
            ),
            "converged": bool(res["converged"]),
            "steps": res["steps"],
        }

    # (3) sub-yield plane channel: stress below tau_y everywhere -> no motion
    h3 = 40
    rheo = hb.validate_rheology(2.0e-3, 0.1, 1.0)     # tau_y far above tau_w
    a_sub = 1.0e-5                                    # tau_w = a*h/2 = 2e-4 << 2e-3
    res = hl.run_hb_channel(h3, rheo, a=a_sub, u_wall=0.0, n_steps=3000 if fast else 10000,
                            m=M_REG, nu_max=NU_MAX, check_every=500, tol=1.0e-13)
    u = res["u_profile"][1:-1].to(torch.float64)
    # reference velocity scale of the same channel if the fluid yielded
    yielded = hb.validate_rheology(0.0, 0.1, 1.0)
    u_ref = float(hb.hb_plane_velocity_profile(torch.tensor(0.0), float(h3),
                                               a_sub * h3 / 2.0, yielded))
    out["sub_yield_channel"] = {
        "case": "negative_control_sub_yield_channel",
        "rheology": rheo.as_dict(),
        "gap_h": h3,
        "body_force_a": a_sub,
        "tau_w": a_sub * h3 / 2.0,
        "u_max_lbm": float(u.abs().max()),
        "u_max_over_unyielded_reference": float(u.abs().max()) / u_ref if u_ref else None,
        "converged": bool(res["converged"]),
        "steps": res["steps"],
    }
    return out


# ---------------------------------------------------------------------------
# Slush pipeline curve
# ---------------------------------------------------------------------------


def slush_curve(R_eff: float, fast: bool) -> dict:
    """Slush (ice-slurry) rheology -> DeltaP-Q curve, LBM vs analytic H-B."""
    pts = []
    phi_ice = 0.25 if fast else 0.25
    rheo = hb.slush_rheology(phi_ice)
    n = 2 * int(math.ceil(R_PIPE)) + 4
    _, rr = hl._pipe_solid_mask(n, R_PIPE)
    # body forces chosen so that tau_w brackets the slush yield stress
    a_scale = 6.0 * rheo.tau_y / R_eff
    for k, frac in enumerate((0.55, 1.0, 1.6) if not fast else (1.0,)):
        a = frac * a_scale
        u_init = _warm_start(rr, a, rheo, M_REG, NU_MAX)
        res = hl.run_hb_pipe(R_PIPE, rheo, a, n_steps=2500, n_axial=2, m=M_REG,
                             nu_max=NU_MAX, check_every=500, tol=1.0e-10, u_init=u_init)
        q = res["flow_rate"] / 2.0
        tau_w = a * R_eff / 2.0
        q_an = hb.hb_pipe_flow_rate(tau_w, R_eff, rheo)
        dpdl = 2.0 * tau_w / R_eff
        pts.append({
            "index": k,
            "body_force_a": a,
            "dP_dL": dpdl,
            "tau_w": tau_w,
            "plug_fraction": min(1.0, rheo.tau_y / tau_w),
            "Q_lbm": q,
            "Q_analytic": q_an,
            "Q_rel_err": q / q_an - 1.0 if q_an else None,
            "converged": bool(res["converged"]),
            "steps": res["steps"],
        })
    # Newtonian reference curve at the same carrier viscosity and gradients
    newton = hb.validate_rheology(0.0, 1.0, 1.0)
    newton_pts = []
    for p in pts:
        a = p["body_force_a"]
        newton_pts.append({
            "body_force_a": a,
            "Q_analytic_newtonian": hb.hb_pipe_flow_rate(a * R_eff / 2.0, R_eff, newton),
        })
    return {
        "case": "slush_pipeline_curve",
        "ice_fraction": phi_ice,
        "rheology": rheo.as_dict(),
        "R_eff_used": R_eff,
        "points": pts,
        "newtonian_reference": newton_pts,
        "max_abs_Q_rel_err": max(abs(p["Q_rel_err"]) for p in pts if p["Q_rel_err"] is not None),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def make_figures(result: dict) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGDIR.mkdir(parents=True, exist_ok=True)
    made = []

    # Figure 1: pipe velocity profiles (plug core) for both rheologies
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, key, title in (
        (axes[0], "bingham_a1.00", f"Bingham (n=1), phi=tau_y/tau_w={PHI:.2f}"),
        (axes[1], "hb06_a1.00", "Herschel-Bulkley n=0.6"),
    ):
        d = result["pipe_cases"].get(key)
        if d is None:
            continue
        r = d["profile_r"]
        ax.plot(r, d["profile_u_lbm"], "o", ms=4, label="LBM (D3Q19)")
        ax.plot(r, d["profile_u_analytic"], "-", lw=1.8, label="analytic H-B")
        ax.plot(r, d["profile_u_regularised"], "--", lw=1.2, label="1-D regularised ref.")
        ax.axvline(d["plug_radius_momentum_balance"], color="0.4", ls=":", lw=1.2,
                   label=r"analytic $r_c=R\tau_y/\tau_w$")
        ax.set_xlabel("r / lattice units")
        ax.set_ylabel(r"$u_x$ / lattice units")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Pipe Poiseuille plug core: LBM vs analytic Herschel-Bulkley", fontsize=11)
    fig.tight_layout()
    p1 = FIGDIR / "pipe_plug_core.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    made.append(p1.name)

    # Figure 2: pressure drop vs flow rate (slush curve + H-B branches)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for tag, marker, color in (("bingham", "o", "tab:blue"), ("hb", "s", "tab:red")):
        pts = [(d["flow_rate_analytic"], d["body_force_a"])
               for k, d in result["pipe_cases"].items() if k.startswith(tag)]
        pts.sort()
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", color=color, lw=1.2,
                    label=f"H-B analytic ({tag})")
        pts_num = sorted((d["flow_rate_lbm"], d["body_force_a"])
                         for k, d in result["pipe_cases"].items() if k.startswith(tag))
        if pts_num:
            ax.plot([p[0] for p in pts_num], [p[1] for p in pts_num], marker, color=color,
                    ms=6, mfc="none", label=f"LBM ({tag})")
    sc = result.get("slush_curve")
    if sc and sc["points"]:
        ax.plot([p["Q_lbm"] for p in sc["points"]], [p["dP_dL"] for p in sc["points"]],
                "d", color="tab:green", ms=8, label="LBM slush (phi=%.2f)" % sc["ice_fraction"])
        ax.plot([p["Q_analytic"] for p in sc["points"]],
                [p["dP_dL"] for p in sc["points"]], "x", color="tab:green", ms=9,
                label="analytic H-B (slush)")
    ax.set_xlabel("flow rate Q")
    ax.set_ylabel(r"pressure gradient $\Delta P/L$ (= body force $a$, rho=1)")
    ax.set_title("Pressure drop vs flow rate: yield-stress vs Newtonian branch", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p2 = FIGDIR / "pressure_drop_flow_rate.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    made.append(p2.name)

    # Figure 3: plate Couette yield surface
    cc = result["couette_cases"]
    if cc:
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        for key, style in zip(sorted(cc), ("-", "--", ":")):
            d = cc[key]
            ax.plot(d["profile_u_lbm"], d["profile_y"], style, lw=1.6,
                    label=f"{d['case']}")
            if d["y_yield_analytic"]:
                ax.axhline(d["y_yield_analytic"], color="0.6", ls=":", lw=1.0)
        ax.set_xlabel(r"$u_x$ / lattice units")
        ax.set_ylabel("y above resting plate")
        ax.set_title("Couette-Poiseuille (adverse gradient): static plug and yield surface",
                     fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        p3 = FIGDIR / "couette_yield_surface.png"
        fig.savefig(p3, dpi=150)
        plt.close(fig)
        made.append(p3.name)
    return made


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true",
                    help="reduced case set / shorter runs (R=14 pipe, fewer body forces)")
    args = ap.parse_args(argv)
    global R_PIPE
    if args.fast:
        R_PIPE = 14.0
    t_start = time.time()

    # --- geometry calibration + negative control 1 ------------------------
    neg = run_negative_controls(args.fast)
    R_eff = neg["newtonian_pipe"]["R_eff_from_flow_rate"]
    print(f"[geom] R_eff = {R_eff:.4f} (nominal {R_PIPE})")

    # --- pipe cases -------------------------------------------------------
    pipe = {}
    n_steps = 2500 if args.fast else 6000
    for name, spec in _pipe_cases(args.fast).items():
        d = analyse_pipe(name, R_PIPE, spec["a"], spec["rheo"], M_REG, NU_MAX,
                         n_steps=n_steps, warm=True, R_eff=R_eff)
        pipe[name] = d
        print(f"[pipe] {name}: Q err {d['flow_rate_rel_err']*100:+.2f}%  "
              f"r_c {d['plug_radius_probe_lbm']} vs {d['plug_radius_probe_analytic']} "
              f"(err {d['plug_radius_rel_err_vs_analytic_probe']}+)")

    # --- grid refinement (primary Bingham case at R = 28) -----------------
    grid = {}
    if not args.fast:
        base = _pipe_cases(False)["bingham_a1.00"]
        # calibrate the hydraulic radius of the R=28 staircase pipe the same way
        rheo_n = hb.validate_rheology(0.0, NU_REF["bingham"], 1.0)
        res28 = hl.run_hb_pipe(28.0, rheo_n, A_NEWTON, n_steps=8000, n_axial=2, m=1.0,
                               nu_max=None, check_every=500, tol=1.0e-10)
        q28 = res28["flow_rate"] / 2.0
        r_eff28 = (8.0 * NU_REF["bingham"] * q28 / (math.pi * A_NEWTON)) ** 0.25
        d28 = analyse_pipe("bingham_R28", 28.0, base["a"], base["rheo"], M_REG, NU_MAX,
                           n_steps=8000, warm=True, R_eff=r_eff28)
        grid["R28"] = d28
        grid["R28"]["R_eff_from_flow_rate"] = r_eff28

    # --- plate Couette cases ---------------------------------------------
    couette = {}
    bingham = hb.hb_pipe_design_case(1.0, PHI, U_TARGET, NU_REF["bingham"], R_PIPE)["rheo"]
    hb06 = hb.hb_pipe_design_case(0.6, PHI, U_TARGET, NU_REF["hb"], R_PIPE)["rheo"]
    beta = 3.0e-4 / 40.0     # tau_w = beta*h = 3e-4 at h=40 -> plug fraction 0.5
    for tag, rheo, h in (("bingham_h40", bingham, 40), ("hb06_h40", hb06, 40),
                         ("bingham_h60", bingham, 60)):
        if args.fast and tag != "bingham_h40":
            continue
        d = analyse_couette(tag, h, rheo, beta, M_REG, NU_MAX,
                            n_steps=6000 if args.fast else 20000)
        couette[tag] = d
        print(f"[couette] {tag}: y_yield {d['y_yield_probe_lbm']} vs analytic "
              f"{d['y_yield_analytic']} (probe-an {d['y_yield_probe_analytic']})")

    # --- slush curve ------------------------------------------------------
    slush = slush_curve(R_eff, args.fast)
    print(f"[slush] max |Q rel err| = {slush['max_abs_Q_rel_err']*100:.2f}%")

    # --- gates ------------------------------------------------------------
    def rel(x):
        return None if x is None else float(x)

    pipe_q_errs = [abs(d["flow_rate_rel_err"]) for d in pipe.values()]
    slush_q_errs = [abs(p["Q_rel_err"]) for p in slush["points"] if p["Q_rel_err"] is not None]
    plug_errs = [abs(d["plug_radius_rel_err_vs_analytic_probe"]) for d in pipe.values()
                 if d["plug_radius_rel_err_vs_analytic_probe"] is not None]
    plugreg_errs = [abs(d["plug_radius_rel_err_vs_regularised_reference"]) for d in pipe.values()
                    if d["plug_radius_rel_err_vs_regularised_reference"] is not None]
    cou_errs = [abs(d["y_yield_rel_err_vs_analytic_probe"]) for d in couette.values()
                if d["y_yield_rel_err_vs_analytic_probe"] is not None]
    coureg_errs = [abs(d["y_yield_rel_err_vs_regularised_reference"]) for d in couette.values()
                   if d["y_yield_rel_err_vs_regularised_reference"] is not None]
    prof_errs = [d["profile_max_rel_err_yielded"] for d in pipe.values()
                 if d["profile_max_rel_err_yielded"] is not None]

    gates = [
        {
            "id": "G1_pipe_plug_radius",
            "description": "pipe plug radius: threshold probe on the LBM profile vs the "
                           "same probe on the analytic H-B profile",
            "values": [rel(e) for e in plug_errs],
            "worst": max(plug_errs) if plug_errs else None,
            "threshold": 0.03,
            "passed": bool(plug_errs) and max(plug_errs) <= 0.03,
        },
        {
            "id": "G1b_pipe_plug_radius_vs_regularised_reference",
            "description": "plug radius vs the exact 1-D solution of the regularised model "
                           "(isolates discretisation from regularisation bias)",
            "values": [rel(e) for e in plugreg_errs],
            "worst": max(plugreg_errs) if plugreg_errs else None,
            "threshold": 0.03,
            "passed": bool(plugreg_errs) and max(plugreg_errs) <= 0.03,
        },
        {
            "id": "G2_couette_yield_surface",
            "description": "plate Couette yield surface y_yield/h*tau_w/tau_y: probe on the "
                           "LBM profile vs the same probe on the analytic profile",
            "values": [rel(e) for e in cou_errs],
            "worst": max(cou_errs) if cou_errs else None,
            "threshold": 0.03,
            "passed": bool(cou_errs) and max(cou_errs) <= 0.03,
        },
        {
            "id": "G2b_couette_yield_surface_vs_regularised_reference",
            "description": "plate yield surface vs the 1-D regularised reference solution",
            "values": [rel(e) for e in coureg_errs],
            "worst": max(coureg_errs) if coureg_errs else None,
            "threshold": 0.03,
            "passed": bool(coureg_errs) and max(coureg_errs) <= 0.03,
        },
        {
            "id": "G3_pressure_drop_flow_rate",
            "description": "DeltaP-Q互证: LBM flow rate vs analytic H-B Poiseuille law "
                           "(all body forces, both rheologies)",
            "values": [rel(e) for e in pipe_q_errs],
            "worst": max(pipe_q_errs) if pipe_q_errs else None,
            "threshold": 0.03,
            "passed": bool(pipe_q_errs) and max(pipe_q_errs) <= 0.03,
        },
        {
            "id": "G3b_slush_pipeline_curve",
            "description": "slush (phi=0.25) pipeline DeltaP-Q points vs analytic H-B law",
            "values": [rel(e) for e in slush_q_errs],
            "worst": max(slush_q_errs) if slush_q_errs else None,
            "threshold": 0.03,
            "passed": bool(slush_q_errs) and max(slush_q_errs) <= 0.03,
        },
        {
            "id": "G4_profile_shape",
            "description": "max |u_lbm-u_analytic|/u_analytic over the yielded annulus (pipe)",
            "values": [rel(e) for e in prof_errs],
            "worst": max(prof_errs) if prof_errs else None,
            "threshold": 0.03,
            "passed": bool(prof_errs) and max(prof_errs) <= 0.03,
        },
        {
            "id": "N1_no_yield_no_plug",
            "description": "negative control: tau_y=0 pipe has no plug core "
                           "(plug probe returns None or <= 0.1R) and matches the "
                           "Newtonian Hagen-Poiseuille profile",
            "values": [neg["newtonian_pipe"]["plug_radius_over_R"]],
            "worst": neg["newtonian_pipe"]["plug_radius_over_R"],
            "threshold": 0.10,
            "passed": (neg["newtonian_pipe"]["plug_radius_over_R"] is None
                       or neg["newtonian_pipe"]["plug_radius_over_R"] <= 0.10),
        },
        {
            "id": "N2_pure_shear_no_interior_yield_surface",
            "description": "negative control: pure plane Couette (uniform stress) has no "
                           "interior yield surface and an (almost) linear profile",
            "values": [neg["pure_couette_zero_yield"]["yield_surface_probe"],
                       neg["pure_couette_yield_bingham"]["yield_surface_probe"]],
            "worst": max(
                neg["pure_couette_zero_yield"]["profile_linearity_max_rel_err"],
                neg["pure_couette_yield_bingham"]["profile_linearity_max_rel_err"],
            ),
            "threshold": 0.03,
            "passed": (
                neg["pure_couette_zero_yield"]["yield_surface_probe"] is not None
                and neg["pure_couette_zero_yield"]["yield_surface_probe"] < 0.25 * 40
                and neg["pure_couette_yield_bingham"]["yield_surface_probe"] is not None
                and neg["pure_couette_yield_bingham"]["yield_surface_probe"] < 0.25 * 40
                and neg["pure_couette_zero_yield"]["profile_linearity_max_rel_err"] <= 0.03
                and neg["pure_couette_yield_bingham"]["profile_linearity_max_rel_err"] <= 0.03
            ),
        },
        {
            "id": "N3_below_yield_no_flow",
            "description": "negative control: body force whose wall stress stays below tau_y "
                           "produces (essentially) no flow",
            "values": [neg["sub_yield_channel"]["u_max_over_unyielded_reference"]],
            "worst": neg["sub_yield_channel"]["u_max_over_unyielded_reference"],
            "threshold": 0.01,
            "passed": neg["sub_yield_channel"]["u_max_over_unyielded_reference"] is not None
            and neg["sub_yield_channel"]["u_max_over_unyielded_reference"] <= 0.01,
        },
    ]

    all_passed = all(g["passed"] for g in gates)
    figures = make_figures({"pipe_cases": pipe, "couette_cases": couette,
                            "slush_curve": slush})

    result = {
        "case": "rheology_yield",
        "title": "Herschel-Bulkley yield-stress rheology: pipe plug core, plate yield "
                 "surface, slush DeltaP-Q curve",
        "module": ["tensorlbm.herschel_bulkley", "tensorlbm.herschel_bulkley_lbm"],
        "lattice": {"pipe": "D3Q19", "plate": "D2Q9", "collision": "variable-viscosity BGK "
                                                             "(Papanastasiou-regularised H-B)",
                    "forcing": "Guo (2002) velocity-shift, per-cell relaxation time",
                    "wall": "half-way bounce-back (stationary and moving)"},
        "geometry_calibration": {
            "R_nominal": R_PIPE,
            "R_eff": R_eff,
            "method": "hydraulic radius from the Newtonian flow rate, "
                      "R_eff = (8 nu Q/(pi a))^(1/4)",
        },
        "parameters": {
            "plug_fraction_target": PHI,
            "u_target": U_TARGET,
            "nu_ref": NU_REF,
            "m_regularisation": M_REG,
            "nu_max_clamp": NU_MAX,
            "probe_eps": EPS_PROBE,
        },
        "pipe_cases": pipe,
        "grid_refinement": grid,
        "couette_cases": couette,
        "slush_curve": slush,
        "negative_controls": neg,
        "comparison_method": "exact analytic H-B solutions (closed-form pipe/slit flow rate, "
                             "pipe/slit velocity profile, Couette-Poiseuille yield surface) "
                             "plus the exact 1-D solution of the regularised model; identical "
                             "threshold probes applied to numerics and analytics",
        "primary_metric": "plug radius (pipe), yield-surface position (plate), flow rate "
                          "(DeltaP-Q), all relative errors against the analytic H-B solution",
        "gates": {g["id"]: g["passed"] for g in gates},
        "gates_worst": {g["id"]: g["worst"] for g in gates},
        "all_gates_passed": all_passed,
        "figures": figures,
        "platform": {"python": platform.python_version(), "torch": torch.__version__,
                     "machine": platform.machine()},
        "wall_time_s": round(time.time() - t_start, 1),
        "notes": [
            "The campaign brief quotes r_c = R(1-tau_y/tau_w); the plug relation is "
            "r_c = R*tau_y/tau_w and R(1-tau_y/tau_w) is the yielded-annulus thickness. "
            "Both are reported.",
            "A regularised yield-stress model has no perfectly rigid plug: the core creeps "
            "at sigma/nu_plug, which is why the plug probe is compared both to the ideal "
            "H-B profile and to the exact 1-D solution of the regularised model.",
            "Pure plane Couette has a uniform stress, hence no interior yield surface; that "
            "degeneracy is the N2 negative control rather than a missing capability.",
        ],
        "verified": bool(all_passed),
        "saved_to": str(HERE),
    }

    (HERE / "result.json").write_text(json.dumps(result, indent=2, default=str))
    (HERE / "gates.json").write_text(json.dumps(
        {"case": "rheology_yield", "all_passed": all_passed, "gates": gates,
         "thresholds": "relative errors, <= 3% unless stated otherwise"}, indent=2))
    print("\n=== GATES ===")
    for g in gates:
        print(f"  {g['id']:48s} worst={g['worst']}  passed={g['passed']}")
    print(f"  ALL PASSED: {all_passed}")
    print(f"[done] {time.time() - t_start:.0f}s -> {HERE}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())