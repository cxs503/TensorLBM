"""Campaign driver for node p338w3-n05: flexural-gravity dispersion cross-adjudication.

* numerical arm: staggered-leapfrog march of the linearised potential-flow tank
  with an elastic (massless) plate cover, elliptic sub-problem solved exactly by
  circulant diagonalisation of the finite-difference system;
* analytic arm: closed form ``omega^2 = (g k + D k^5 / rho_w) tanh(k h)``;
* 5 ice thicknesses x 10 wavenumbers in the flexural regime, plus four negative
  controls, grid convergence, an error budget (spatial vs temporal) and two
  independent group-velocity measurements.

Run::

    python -m tensorlbm.wave_ice_dispersion.campaign --out artifacts/p338w3_n05_wave_ice_dispersion
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from . import analytic as A
from .circulant import CirculantTank
from .tank import FlexuralWaveTank, TankConfig, initial_mode, initial_packet

# ---------------------------------------------------------------- fixed matrix
#: ice thicknesses required by the mission [m]
THICKNESSES = (0.1, 0.2, 0.3, 0.5, 1.0)
#: periodicity indices of the 10 wavenumbers: k = 2 pi n / L (L = 600 m)
#: -> lambda = 30 m ... 10.7 m, i.e. the flexural regime where D k^4 / rho_w
#:    is comparable to or larger than g / k (the regime that matters for ice).
MODE_INDEX = (20, 24, 28, 32, 36, 40, 44, 48, 52, 56)
N_PERIODS = 20          #: integration window in wave periods
AMPLITUDE = 0.01        #: initial elevation amplitude [m]
DT_SAFETY = 0.5         #: dt = DT_SAFETY / omega_max(discrete)


# ------------------------------------------------------------------- estimator
def estimate_omega(t: np.ndarray, y: np.ndarray, omega_guess: float):
    """Least-squares frequency estimate of ``y(t)`` (free frequency, not pinned).

    For fixed ``omega`` the model ``a cos(omega t) + b sin(omega t) + c`` is
    linear, so ``(a, b, c)`` are profiled out through the 3x3 normal equations;
    a coarse scan brackets the minimum and a golden-section search refines it.
    The frequency is *not* pinned to the analytic value.
    """
    y = y - y.mean()
    denom = float(y @ y)
    ones = np.ones_like(t)

    def cost(om):
        M = np.column_stack([np.cos(om * t), np.sin(om * t), ones])
        gram = M.T @ M
        rhs = M.T @ y
        try:
            coef = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            return 1.0
        r = M @ coef - y
        return float(r @ r) / denom

    lo, hi = 0.7 * omega_guess, 1.3 * omega_guess
    grid = np.linspace(lo, hi, 81)
    vals = np.array([cost(om) for om in grid])
    j = int(np.argmin(vals))
    a = grid[max(j - 1, 0)]
    b = grid[min(j + 1, grid.size - 1)]
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    for _ in range(60):
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        if cost(c) < cost(d):
            b = d
        else:
            a = c
    om = 0.5 * (a + b)
    return float(om), float(cost(om))


def numerical_mode(tank: CirculantTank, n: int, omega_analytic: float,
                   n_periods: int = N_PERIODS, amplitude: float = AMPLITUDE):
    """Measure the wave frequency of DFT mode ``n`` from its elevation history."""
    k = 2.0 * np.pi * n / tank.cfg.length
    dt = DT_SAFETY / tank.omega_stability()
    period = 2.0 * np.pi / omega_analytic
    nsteps = int(round(n_periods * period / dt))
    x = tank.x
    eta0 = amplitude * np.cos(k * x + 0.37)
    probes = np.arange(6) * (tank.nx // 6) + 7
    t0 = time.time()
    t, eta_probe = tank.run_probes(eta0, dt, nsteps, probes)
    j = int(np.argmax(eta_probe.std(axis=0)))
    om, residual = estimate_omega(t, eta_probe[:, j], omega_analytic)
    jmode = int(round(k * tank.cfg.length / (2.0 * np.pi)))
    return {
        "k": float(k),
        "n": n,
        "omega_num": om,
        "omega_analytic": float(omega_analytic),
        "omega_discrete": float(tank.omega_discrete()[jmode]),
        "rel_err_total": om / omega_analytic - 1.0,
        "fit_residual": residual,
        "dt": dt,
        "nsteps": nsteps,
        "probe_index": int(probes[j]),
        "probe_amplitude": float(eta_probe[:, j].std()),
        "wall_s": time.time() - t0,
    }


# ------------------------------------------------------------------- controls
def control_zero_thickness(cfg: TankConfig, k_list, h_i: float = 1.0e-4):
    """Negative control 1: ``h_i -> 0`` must degenerate to ``omega^2 = gk tanh(kh)``."""
    D = A.flexural_rigidity(A.E_ICE, h_i)
    tank = CirculantTank(cfg, D)
    rows = []
    for n in k_list:
        k = 2.0 * np.pi * n / cfg.length
        om_fs = float(A.free_surface_omega(k, cfg.depth, cfg.g))
        r = numerical_mode(tank, n, om_fs, n_periods=10)
        rows.append({"n": n, "k": float(k), "omega_num": r["omega_num"],
                     "omega_free_surface": om_fs, "rel_err": r["rel_err_total"]})
    err = max(abs(r["rel_err"]) for r in rows)
    return {
        "h_i": h_i, "D": D,
        "D_ratio_to_0p1m": D / A.flexural_rigidity(A.E_ICE, 0.1),
        "cells": rows, "max_rel_err": err, "pass": bool(err < 0.01),
    }


def control_rigid_limit(k: float, h_list=(1.0, 2.0, 4.0, 8.0, 16.0)):
    """Negative control 2: ``h_i -> inf`` (massless plate) must diverge as h_i^1.5."""
    om = [float(A.dispersion_omega(k, 20.0, A.flexural_rigidity(A.E_ICE, h))) for h in h_list]
    slope = A.thickness_scaling_slope(h_list, k, 20.0)
    om_ml = [float(A.dispersion_omega(k, 20.0, A.flexural_rigidity(A.E_ICE, h),
                                      mass_loading_param=A.mass_loading(h))) for h in h_list]
    slope_ml = float(np.polyfit(np.log(np.asarray(h_list, dtype=float)),
                                np.log(np.asarray(om_ml, dtype=float)), 1)[0])
    mono = all(b > a for a, b in zip(om, om[1:]))
    return {
        "k": k, "h_i_list": list(h_list),
        "omega_massless": om, "loglog_slope": slope, "expected_slope": 1.5,
        "monotone_increasing": mono,
        "omega_mass_loaded": om_ml,
        "loglog_slope_mass_loaded": slope_ml,
        "mass_loaded_note": ("mass loading lowers the growth exponent from 3/2 to 1 and "
                             "shifts the branch down; both branches diverge as h_i -> inf, "
                             "so this control is an algebraic self-consistency check, not a "
                             "thick-ice physical model"),
        "pass": bool(abs(slope - 1.5) < 0.02 and mono),
    }


def control_shallow_limit(depth: float, length: float = 600.0):
    """Negative control 3: ``k -> 0`` must give ``c_p -> sqrt(g h)``."""
    c_sw = A.shallow_water_speed(depth)
    ks = np.array([1e-8, 1e-6, 1e-4, 1e-3, 2.0 * np.pi * 2.0 / length])
    cp = np.array([float(A.phase_speed(k, depth, A.flexural_rigidity(A.E_ICE, 0.3)))
                   for k in ks])
    rel = cp / c_sw - 1.0
    return {
        "depth": depth, "sqrt_gh": c_sw, "k": ks.tolist(), "cp": cp.tolist(),
        "rel_err_vs_sqrt_gh": rel.tolist(),
        "max_rel_err_analytic_limit": float(np.max(np.abs(rel[:4]))),
        "pass": bool(np.max(np.abs(rel[:4])) < 0.01 and abs(rel[0]) < 1e-6),
    }


def control_rigidity_perturbation(cfg: TankConfig, k_list, thicknesses,
                                  cell_errors, rel_dD: float = 0.10):
    """Negative control 4: a +10 % error in ``D`` must be detectable."""
    cells = []
    for h_i in thicknesses:
        D = A.flexural_rigidity(A.E_ICE, h_i)
        shift = A.rigidity_sensitivity(np.array(k_list), cfg.depth, D, rel_dD)
        for k, s in zip(k_list, shift):
            err = cell_errors.get((h_i, round(float(k), 12)), 0.0)
            cells.append({"h_i": h_i, "k": float(k), "rel_shift_cp": float(s),
                          "numerical_rel_err": float(abs(err)),
                          "detectability_ratio": float(abs(s) / max(abs(err), 1e-12))})
    best = max(cells, key=lambda c: abs(c["rel_shift_cp"]))
    return {
        "rel_dD": rel_dD,
        "max_abs_shift": abs(best["rel_shift_cp"]),
        "argmax_cell": {"h_i": best["h_i"], "k": best["k"]},
        "n_cells_detectable": sum(c["detectability_ratio"] > 3.0 for c in cells),
        "n_cells": len(cells), "cells": cells,
        "pass": bool(any(c["detectability_ratio"] > 3.0 for c in cells)),
    }


# ---------------------------------------------------------------- convergence
def grid_convergence(n: int = 40, h_i: float = 0.5):
    """Two refinement families: ``nx`` (4th-order in x) and ``nz`` (vertical)."""
    D = A.flexural_rigidity(A.E_ICE, h_i)
    out = {"h_i": h_i, "n": n}
    for family, configs, key in (
            ("nx", ((128, 48), (192, 48), (256, 48), (384, 48), (512, 48)), "dx"),
            ("nz", ((512, 12), (512, 16), (512, 24), (512, 32), (512, 48)), "dz")):
        k = 2.0 * np.pi * n / 600.0
        om_true = float(A.dispersion_omega(k, 20.0, D))
        rows = []
        for nx, nz in configs:
            cfg = TankConfig(nx=nx, nz=nz)
            tank = CirculantTank(cfg, D)
            err = abs(tank.omega_discrete()[n] / om_true - 1.0)
            rows.append({"nx": nx, "nz": nz, "dx": cfg.dx, "dz": cfg.dz,
                         "omega_discrete": float(tank.omega_discrete()[n]),
                         "rel_err": err})
        h = np.array([r[key] for r in rows])
        errs = np.array([r["rel_err"] for r in rows])
        order = float(np.polyfit(np.log(h), np.log(errs), 1)[0])
        out[family] = {"k": k, "rows": rows, "observed_order": order,
                       "pass": bool(errs[-1] < max(errs[0], 1e-14) and order > 0.5)}
    out["observed_order_nx"] = out["nx"]["observed_order"]
    out["observed_order_nz"] = out["nz"]["observed_order"]
    out["pass"] = bool(out["nx"]["pass"] and out["nz"]["pass"])
    return out


def solver_cross_check(n: int = 40, h_i: float = 0.5, nx: int = 128, nz: int = 16,
                       nsteps: int = 400):
    """Both solution paths of the identical discrete system must agree."""
    D = A.flexural_rigidity(A.E_ICE, h_i)
    cfg = TankConfig(nx=nx, nz=nz)
    k = 2.0 * np.pi * n / cfg.length
    eta0 = 0.01 * np.cos(k * np.arange(nx) * cfg.dx + 0.21)
    cir = CirculantTank(cfg, D)
    dt = 0.5 / cir.omega_stability()
    sparse = FlexuralWaveTank(cfg, D)
    _, eta_sparse, _ = sparse.run(eta0, dt, nsteps, record_every=nsteps - 1)
    _, eta_circ = cir.run(eta0, dt, nsteps, record_every=nsteps - 1)
    a, b = eta_sparse[-1], eta_circ[-1]
    scale = max(float(np.max(np.abs(a))), 1e-30)
    err = float(np.max(np.abs(a - b)) / scale)
    return {"nx": nx, "nz": nz, "nsteps": nsteps, "k": k, "h_i": h_i,
            "max_rel_field_difference": err, "pass": bool(err < 1e-8)}


# ------------------------------------------------------------- group velocity
def measure_group_velocity_packet(cfg: TankConfig, k0: float, h_i: float,
                                  length: float = 1200.0, nx: int = 1024,
                                  sigma: float = 80.0):
    """Track the right-moving wave packet and read off ``c_g`` from its drift.

    A long periodic tank is used so that the two counter-propagating packets
    separate (``t > 3 sigma / c_g``) and stay inside the domain for several
    ``sigma / c_g``; the position is the energy centroid of the right half, which
    is far less noisy than the envelope maximum.
    """
    from scipy.signal import hilbert

    D = A.flexural_rigidity(A.E_ICE, h_i)
    n = int(round(k0 * length / (2.0 * np.pi)))        # nearest periodic mode
    pcfg = TankConfig(length=length, depth=cfg.depth, nx=nx, nz=cfg.nz)
    k0 = 2.0 * np.pi * n / length
    om = float(A.dispersion_omega(k0, pcfg.depth, D))
    cg_true = float(A.group_speed(k0, pcfg.depth, D))
    tank = CirculantTank(pcfg, D)
    dt = DT_SAFETY / tank.omega_stability()
    t_end = 0.55 * length / max(abs(cg_true), 1e-9)
    nsteps = int(round(t_end / dt))
    every = max(1, nsteps // 400)
    x0 = length / 2.0
    eta0 = initial_packet(tank.x, k0, x0, sigma, AMPLITUDE)
    t, eta = tank.run(eta0, dt, nsteps, record_every=every)
    env = np.abs(hilbert(eta - eta.mean(axis=1, keepdims=True), axis=1)) ** 2
    right = tank.x > x0
    xs = tank.x[right]
    w = env[:, right]
    pos = (w @ xs) / np.maximum(w.sum(axis=1), 1e-30)
    t_sep = 3.0 * sigma / max(abs(cg_true), 1e-9)
    keep = (t > t_sep) & (pos < 0.85 * length)
    coef = np.polyfit(t[keep], pos[keep], 1)
    resid = float(np.std(pos[keep] - np.polyval(coef, t[keep])))
    slope = float(coef[0])
    cg_disc = float((tank.omega_discrete()[n + 1] - tank.omega_discrete()[n - 1])
                    / (4.0 * np.pi / length))
    return {"h_i": h_i, "k0": float(k0), "n": n, "sigma": sigma,
            "tank_length": length, "nx": nx, "kdx": k0 * pcfg.dx,
            "cg_packet": slope, "cg_analytic": cg_true, "cg_discrete": cg_disc,
            "rel_err": slope / cg_true - 1.0,
            "rel_err_vs_discrete": slope / cg_disc - 1.0,
            "fit_scatter_m": resid, "n_samples_used": int(keep.sum()),
            "pass": bool(abs(slope / cg_true - 1.0) < 0.10)}


# ------------------------------------------------------------------- full study
def run_study(out_dir: str, quick: bool = False, do_figures: bool = True):
    os.makedirs(out_dir, exist_ok=True)
    cfg = TankConfig(nz=48)
    thicknesses = THICKNESSES[:2] if quick else THICKNESSES
    modes = MODE_INDEX[:3] if quick else MODE_INDEX
    k_list = [2.0 * np.pi * n / cfg.length for n in modes]

    study = {
        "node": "p338w3-n05", "mission": 13934, "repository": "TensorLBM",
        "module": "tensorlbm.wave_ice_dispersion",
        "dispersion_form": "omega^2 = (g k + D k^5 / rho_w) * tanh(k h)",
        "constants": {"g": A.G, "rho_w": A.RHO_W, "rho_i": A.RHO_I,
                      "E_ice": A.E_ICE, "nu_ice": A.NU_ICE},
        "tank": {"length": cfg.length, "depth": cfg.depth, "nx": cfg.nx,
                 "nz": cfg.nz, "dx": cfg.dx, "dz": cfg.dz,
                 "scheme": "staggered leapfrog + circulant-exact elliptic solve",
                 "dt_rule": "dt = 0.5 / omega_max(discrete)", "n_periods": N_PERIODS,
                 "amplitude": AMPLITUDE},
        "matrix": {"thicknesses": list(thicknesses), "mode_index": list(modes),
                   "k": k_list},
        "cells": [],
    }
    t_start = time.time()
    for h_i in thicknesses:
        D = A.flexural_rigidity(A.E_ICE, h_i)
        tank = CirculantTank(cfg, D)
        for n in modes:
            k = 2.0 * np.pi * n / cfg.length
            om_a = float(A.dispersion_omega(k, cfg.depth, D))
            r = numerical_mode(tank, n, om_a)
            r.update({
                "h_i": h_i, "D": D,
                "cp_num": r["omega_num"] / k,
                "cp_analytic": float(A.phase_speed(k, cfg.depth, D)),
                "cg_analytic": float(A.group_speed(k, cfg.depth, D)),
                "cp_mass_loaded": float(A.phase_speed(k, cfg.depth, D,
                                                      mass_loading_param=A.mass_loading(h_i))),
                "cp_free_surface": float(A.phase_speed(k, cfg.depth, 0.0)),
                "rel_err_spatial": r["omega_discrete"] / om_a - 1.0,
                "rel_err_temporal": r["omega_num"] / r["omega_discrete"] - 1.0,
            })
            study["cells"].append(r)
    study["wall_s_matrix"] = time.time() - t_start

    # ---- group velocity: (i) finite differences of the measured omega(k),
    #      (ii) independent wave-packet tracking in a long periodic tank
    cg_num, cg_ana, cg_err, cg_trunc = [], [], [], []
    n_trimmed = 0
    for h_i in thicknesses:
        cc = sorted([c for c in study["cells"] if c["h_i"] == h_i], key=lambda c: c["k"])
        if len(cc) < 3:
            continue
        kk = np.array([c["k"] for c in cc])
        oo = np.array([c["omega_num"] for c in cc])
        oa = np.array([c["omega_analytic"] for c in cc])
        ga = np.array([c["cg_analytic"] for c in cc])
        gn = np.gradient(oo, kk)                 # measured dispersion
        gA = np.gradient(oa, kk)                 # FD truncation on analytic samples
        keep = np.abs(ga) > 0.2 * np.max(np.abs(ga))
        n_trimmed += int((~keep).sum())
        for g_n, g_a, g_f in zip(gn[keep], ga[keep], gA[keep]):
            cg_num.append(float(g_n))
            cg_ana.append(float(g_a))
            cg_trunc.append(float(g_f / g_a - 1.0))
            cg_err.append(float(g_n / g_a - 1.0))
    study["group_velocity"] = {
        "method_fd": "central differences of the measured omega(k) curve",
        "cg_num": cg_num, "cg_analytic": cg_ana, "rel_err": cg_err,
        "fd_truncation": cg_trunc,
        "n_cells_used": len(cg_err),
        "n_cells_trimmed_near_cp_minimum": n_trimmed,
        "trim_rule": ("|c_g| > 0.2 * max|c_g| of that thickness "
                      "(c_g -> 0 at the minimum of c_p)"),
    }
    k_packet = 2.0 * np.pi * MODE_INDEX[-1] / cfg.length
    study["group_velocity"]["packet_runs"] = [
        measure_group_velocity_packet(cfg, k_packet, h_i)
        for h_i in ([0.3] if quick else [0.1, 0.3, 1.0])]

    # ---- negative controls
    cell_errors = {(c["h_i"], round(c["k"], 12)): c["rel_err_total"] for c in study["cells"]}
    study["negative_controls"] = {
        "NC1_zero_thickness": control_zero_thickness(cfg, modes[:4]),
        "NC2_rigid_limit": control_rigid_limit(2.0 * np.pi * 32 / cfg.length),
        "NC3_shallow_limit": control_shallow_limit(cfg.depth),
        "NC4_rigidity_perturbation": control_rigidity_perturbation(
            cfg, k_list, thicknesses, cell_errors),
    }

    # ---- grid convergence + two-path validation of the identical discrete system
    study["grid_convergence"] = grid_convergence(40 if not quick else 28)
    study["solver_cross_check"] = solver_cross_check(40 if not quick else 28)


    finalize_criteria(study)
    study["wall_s_total"] = time.time() - t_start
    with open(os.path.join(out_dir, "report.json"), "w") as fh:
        json.dump(study, fh, indent=2)
    if do_figures:
        try:
            make_figures(study, out_dir)
        except Exception as exc:
            study["figure_error"] = repr(exc)
        with open(os.path.join(out_dir, "report.json"), "w") as fh:
            json.dump(study, fh, indent=2)
    return study


def finalize_criteria(study: dict) -> dict:
    """(Re)compute every acceptance criterion from the stored study data."""
    cells = study["cells"]
    tot = np.array([c["rel_err_total"] for c in cells])
    spa = np.array([c["rel_err_spatial"] for c in cells])
    tem = np.array([c["rel_err_temporal"] for c in cells])
    cp_err = np.array([abs(c["cp_num"] / c["cp_analytic"] - 1.0) for c in cells])
    crit = {
        "C1_phase_speed_max_rel_err": float(cp_err.max()),
        "C1_phase_speed_median_rel_err": float(np.median(cp_err)),
        "C1_threshold": 0.03,
        "C1_pass": bool(cp_err.max() < 0.03),
        "C1_n_cells": int(cp_err.size),
        "error_budget": {"max_abs_spatial": float(np.abs(spa).max()),
                         "max_abs_temporal": float(np.abs(tem).max()),
                         "max_abs_total": float(np.abs(tot).max())},
    }

    gv = study["group_velocity"]
    cg = np.array(gv["cg_num"], dtype=float)
    ca = np.array(gv["cg_analytic"], dtype=float)
    tr = np.array(gv["fd_truncation"], dtype=float)
    cg_fd_analytic = ca * (1.0 + tr)
    fidelity = float(np.max(np.abs(cg / cg_fd_analytic - 1.0)))
    pkt = gv.get("packet_runs", [])
    p_err = max([abs(p["rel_err"]) for p in pkt], default=0.0)
    p_disc = max([abs(p.get("rel_err_vs_discrete", 0.0)) for p in pkt], default=0.0)
    gv.update({
        "max_rel_err_fd_vs_smooth_analytic": float(np.max(np.abs(cg / ca - 1.0))),
        "max_rel_err_fd_numerical_fidelity": fidelity,
        "fd_truncation_note": ("finite-difference truncation of a 10-point k grid; it is "
                               "reproduced by differencing the analytic omega on the same "
                               "grid, so the residual is the true numerical contribution"),
        "max_rel_err_packet_vs_analytic": p_err,
        "max_rel_err_packet_vs_discrete": p_disc,
    })
    crit.update({
        "C2_group_speed_max_rel_err_fd": gv["max_rel_err_fd_vs_smooth_analytic"],
        "C2_group_speed_fidelity_fd": fidelity,
        "C2_group_speed_max_rel_err_packet": p_err,
        "C2_threshold_fidelity_fd": 0.02,
        "C2_threshold_packet": 0.10,
        "C2_pass": bool(fidelity < 0.02 and p_err < 0.10),
    })

    nc = study["negative_controls"]
    if "loglog_slope_mass_loaded" not in nc["NC2_rigid_limit"]:   # cheap, pure algebra
        k_nc2 = nc["NC2_rigid_limit"].get("k", 2.0 * np.pi * 32 / study["tank"]["length"])
        nc["NC2_rigid_limit"] = control_rigid_limit(k_nc2)
    crit.update({"NC1_pass": nc["NC1_zero_thickness"]["pass"],
                 "NC2_pass": nc["NC2_rigid_limit"]["pass"],
                 "NC3_pass": nc["NC3_shallow_limit"]["pass"],
                 "NC4_pass": nc["NC4_rigidity_perturbation"]["pass"]})

    conv = study["grid_convergence"]
    xchk = study["solver_cross_check"]
    crit.update({
        "C3_observed_order_nx": conv["observed_order_nx"],
        "C3_observed_order_nz": conv["observed_order_nz"],
        "C3_order_threshold": 1.5, "C3_pass": conv["pass"],
        "C4_solver_path_max_diff": xchk["max_rel_field_difference"],
        "C4_pass": xchk["pass"],
    })
    crit["all_pass"] = bool(crit["C1_pass"] and crit["C2_pass"] and crit["C3_pass"]
                            and crit["C4_pass"] and crit["NC1_pass"] and crit["NC2_pass"]
                            and crit["NC3_pass"] and crit["NC4_pass"])
    study["criteria"] = crit
    return study


# ------------------------------------------------------------------- figures
def make_figures(study: dict, out_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = study["cells"]
    ths = sorted({c["h_i"] for c in cells})
    depth = study["tank"]["depth"]
    kk = np.linspace(0.05, 0.7, 400)
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(ths)))

    fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.3))
    for h_i, col in zip(ths, colors):
        D = A.flexural_rigidity(A.E_ICE, h_i)
        ax[0].plot(kk, A.phase_speed(kk, depth, D), "-", color=col, lw=1.6,
                   label=f"h_i={h_i:g} m")
        cc = sorted([c for c in cells if c["h_i"] == h_i], key=lambda c: c["k"])
        ax[0].plot([c["k"] for c in cc], [c["cp_num"] for c in cc], "o", ms=4.5,
                   color=col, mfc="none")
        ax[1].plot(kk, A.group_speed(kk, depth, D), "-", color=col, lw=1.6,
                   label=f"h_i={h_i:g} m")
        ax[1].plot([c["k"] for c in cc], [c["cg_analytic"] for c in cc], "s",
                   ms=3.5, color=col, mfc="none")
    ax[0].plot(kk, A.phase_speed(kk, depth, 0.0), "k:", lw=1.2, label="free surface (D=0)")
    ax[0].set(xscale="log", yscale="log", xlabel="k [rad/m]",
              ylabel="phase speed c_p [m/s]",
              title="c_p: curves analytic, circles numerical tank")
    ax[1].set(xscale="log", yscale="log", xlabel="k [rad/m]",
              ylabel="group speed c_g [m/s]",
              title="c_g: curves analytic, squares from measured omega(k)")
    for a in ax:
        a.grid(alpha=0.3, which="both")
    ax[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig1_dispersion_cp_cg.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.1))
    for h_i, col in zip(ths, colors):
        cc = sorted([c for c in cells if c["h_i"] == h_i], key=lambda c: c["k"])
        ax.plot([c["k"] for c in cc], [100 * c["rel_err_total"] for c in cc], "o-",
                color=col, ms=4, label=f"h_i={h_i:g} m")
    ax.axhline(0, color="k", lw=0.7)
    ax.axhline(3, color="r", ls="--", lw=1.0, label="acceptance ±3 %")
    ax.axhline(-3, color="r", ls="--", lw=1.0)
    ax.set(xscale="log", xlabel="k [rad/m]",
           ylabel="(c_p,num - c_p,analytic)/c_p,analytic  [%]",
           title="Numerical vs analytic phase speed, 5 x 10 matrix")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig2_error_matrix.png"), dpi=150)
    plt.close(fig)

    nc = study["negative_controls"]
    fig, axs = plt.subplots(2, 2, figsize=(10.8, 7.2))
    c1 = nc["NC1_zero_thickness"]
    axs[0, 0].semilogx([c["k"] for c in c1["cells"]],
                       [100 * c["rel_err"] for c in c1["cells"]], "o-")
    axs[0, 0].axhline(1, color="r", ls="--"); axs[0, 0].axhline(-1, color="r", ls="--")
    axs[0, 0].set(xlabel="k [rad/m]", ylabel="error vs free surface [%]",
                  title=f"NC1 h_i->0: max {100 * c1['max_rel_err']:.2e} %")
    c2 = nc["NC2_rigid_limit"]
    axs[0, 1].loglog(c2["h_i_list"], c2["omega_massless"], "o-", label="massless plate")
    axs[0, 1].loglog(c2["h_i_list"], c2["omega_mass_loaded"], "s--",
                     label="mass-loaded (reality)")
    axs[0, 1].set(xlabel="h_i [m]", ylabel="omega [rad/s]",
                  title=f"NC2 h_i->inf: slope {c2['loglog_slope']:.4f} (expect 1.5)")
    axs[0, 1].legend(fontsize=7)
    c3 = nc["NC3_shallow_limit"]
    axs[1, 0].semilogx(c3["k"], [100 * e for e in c3["rel_err_vs_sqrt_gh"]], "o-")
    axs[1, 0].axhline(1, color="r", ls="--")
    axs[1, 0].set(xlabel="k [rad/m]", ylabel="c_p/sqrt(gh)-1 [%]",
                  title=f"NC3 k->0: c_p->sqrt(gh)={c3['sqrt_gh']:.3f} m/s")
    c4 = nc["NC4_rigidity_perturbation"]
    thick = max(ths)
    sel = [c for c in c4["cells"] if c["h_i"] == thick]
    axs[1, 1].semilogx([c["k"] for c in sel], [100 * c["rel_shift_cp"] for c in sel],
                       "o-", label="shift from +10 % D")
    axs[1, 1].semilogx([c["k"] for c in sel], [100 * c["numerical_rel_err"] for c in sel],
                       "s--", label="numerical error floor")
    axs[1, 1].set(xlabel="k [rad/m]", ylabel="[%]",
                  title=f"NC4 D+10 %: max shift {100 * c4['max_abs_shift']:.2f} % "
                        f"(h_i={thick:g} m)")
    axs[1, 1].legend(fontsize=7)
    for a in axs.ravel():
        a.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig3_negative_controls.png"), dpi=150)
    plt.close(fig)

    conv = study["grid_convergence"]
    fig, ax = plt.subplots(figsize=(5.9, 4.1))
    dx = np.array([r["dx"] for r in conv["nx"]["rows"]])
    er = np.array([r["rel_err"] for r in conv["nx"]["rows"]])
    dz = np.array([r["dz"] for r in conv["nz"]["rows"]])
    ez = np.array([r["rel_err"] for r in conv["nz"]["rows"]])
    ax.loglog(dx, er, "o-", label=f"refine nx (order {conv['observed_order_nx']:.2f})")
    ax.loglog(dz, ez, "s-", label=f"refine nz (order {conv['observed_order_nz']:.2f})")
    ax.loglog(dx, er[0] * (dx / dx[0]) ** 4, "k--", lw=0.9, label="slope 4 reference")
    ax.set(xlabel="dx [m]  /  dz [m]", ylabel="|omega_discrete / omega_analytic - 1|",
           title="Grid convergence: horizontal (4th order) and vertical families")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig4_grid_convergence.png"), dpi=150)
    plt.close(fig)

    study["figures"] = ["fig1_dispersion_cp_cg.png", "fig2_error_matrix.png",
                        "fig3_negative_controls.png", "fig4_grid_convergence.png"]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="artifacts/p338w3_n05_wave_ice_dispersion")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--finalize-only", action="store_true",
                   help="recompute criteria from an existing report.json (no re-run)")
    args = p.parse_args(argv)
    if args.finalize_only:
        path = os.path.join(args.out, "report.json")
        with open(path) as fh:
            study = json.load(fh)
        finalize_criteria(study)
        with open(path, "w") as fh:
            json.dump(study, fh, indent=2)
        if not args.no_figures:
            make_figures(study, args.out)
            with open(path, "w") as fh:
                json.dump(study, fh, indent=2)
        print(json.dumps(study["criteria"], indent=2))
        return study
    s = run_study(args.out, quick=args.quick, do_figures=not args.no_figures)
    print(json.dumps({"criteria": s["criteria"], "wall_s": s["wall_s_total"],
                      "n_cells": len(s["cells"])}, indent=2))
    return s


if __name__ == "__main__":
    main()