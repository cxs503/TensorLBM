"""XVAL-LBM-NO-MASS-ANALOGUE-IN-OSS-INVENTORY / round 2 -- gated re-measurement.

Why round 2
-----------
Round 1 (mission m21062 + re-run in ``../REPORT.md``) established that the library
module ``tensorlbm.oscillatory_cylinder_inertia`` never measures a potential-flow
added mass in its own ``base``-family protocol, and that the failure is a
case-design defect (unconverged spin-up, cold-start acoustic ringing, reference
mis-specified).  Its remediation list (§4 of ``../REPORT.md``) is implemented here:

  R1  spin-up is not a fixed number of periods but a *gate*: the trailing-window
      first harmonic of both the in-phase force and the box-mean velocity must stop
      moving (sampled every period) before the measurement window opens;
  R2  the drive is ramped on over ``ramp_periods`` (a cold start with a step in the
      body force excites the box acoustic modes, whose damping time is long);
  R3  the drive frequency is checked against the acoustic eigenfrequencies of the
      periodic box and against their linewidth ``Gamma = nu k^2`` -- detuning is
      quoted in linewidths, not in percent;
  R4  the reference is the same-mask potential solve evaluated on the same domain
      *and* on a 2x refined grid, Richardson-extrapolated, with its own error bar;
  R5  three force estimators (E1 Ladd exchange, E2 surface traction, E3 exact global
      momentum balance with the Guo half-force correction) must agree.

Nothing in ``src/`` is changed.  Every number in the R2 report is recomputed from
the raw ``runs/*/result.json`` written here.

Usage
-----
    python xval/XVAL-LBM-NO-MASS-ANALOGUE-IN-OSS-INVENTORY/r2/oc_added_mass_r2.py --list
    python .../oc_added_mass_r2.py --arm gate_b50_d32 --out <dir>
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from tensorlbm.boundary.boundaries import bounce_back_cells, compute_obstacle_forces
from tensorlbm.boussinesq import apply_guo_forcing
from tensorlbm.d2q9 import C, equilibrium, macroscopic
from tensorlbm.fsi_added_mass import potential_added_mass
from tensorlbm.oscillatory_cylinder_inertia import (
    _surface_normals,
    _traction_force,
    circle_mask,
    geometry_readout,
    harmonic_decompose,
    inertia_coefficient,
    stokes_cylinder_coefficients,
)
from tensorlbm.solver import stream

CS = 1.0 / math.sqrt(3.0)


# --------------------------------------------------------------------------- #
# case definition
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class R2Case:
    """Gated added-mass arm (all lengths in lattice units)."""

    name: str
    D: int = 32                  # cylinder reference diameter [cells]
    box: float = 4.0             # periodic box side / D
    beta: float = 50.0           # Stokes number D^2/(nu T)
    omega: float = 0.0153        # drive angular frequency [1/steps]
    u0: float = 0.01             # nominal far-field velocity amplitude (= a0/omega)
    ramp_periods: float = 3.0    # cosine ramp-on of the body force
    max_spin_periods: float = 200.0  # hard cap; the gate normally closes much earlier
    gate_window: float = 2.0     # trailing window used by the convergence gate
    gate_tol: float = 0.01       # relative change of amplitude per sampling period
    measure_periods: int = 3     # must be an integer number of periods
    forced_spin_periods: float = 0.0  # >0: skip the gate, spin exactly this many periods
    interior: str = "8"          # population zeroing of the solid core: "4" | "8" | "none"


# All arms keep the acoustic wavelength at least 7 diameters long (omega*D <= 0.5),
# the drive >= 50 acoustic linewidths away from the box eigenfrequencies, and the
# Stokes layer inside the box gap.
CASES: dict[str, R2Case] = {
    # beta ladder at D = 32 (delta >= 2.5 lattice cells for beta <= 100)
    "r2_b400_d32": R2Case("r2_b400_d32", D=32, beta=400.0,
                          omega=2.0 * math.pi / 400.0, max_spin_periods=120.0,
                          gate_window=3.0, gate_tol=0.005),
    "r2_b100_d32": R2Case("r2_b100_d32", D=32, beta=100.0,
                          omega=2.0 * math.pi / 400.0, max_spin_periods=120.0,
                          gate_window=3.0, gate_tol=0.005),
    "r2_b50_d32": R2Case("r2_b50_d32", D=32, beta=50.0,
                         omega=2.0 * math.pi / 400.0, max_spin_periods=120.0,
                         gate_window=3.0, gate_tol=0.005),
    "r2_b25_d32": R2Case("r2_b25_d32", D=32, beta=25.0,
                         omega=2.0 * math.pi / 400.0, max_spin_periods=120.0,
                         gate_window=3.0, gate_tol=0.005),
    # window-independence control: same physics, fixed 40-period spin-up
    "r2_b50_d32_spin40": R2Case("r2_b50_d32_spin40", D=32, beta=50.0,
                                omega=2.0 * math.pi / 400.0,
                                forced_spin_periods=40.0),
    # resolution ladder at beta = 50 (same physical problem, two diameters)
    "r2_b50_d16": R2Case("r2_b50_d16", D=16, beta=50.0,
                         omega=2.0 * math.pi / 200.0, max_spin_periods=120.0,
                         gate_window=3.0, gate_tol=0.005),
    # domain ladder at D = 32, beta = 50
    "r2_b50_box3_d32": R2Case("r2_b50_box3_d32", D=32, box=3.0, beta=50.0,
                              omega=2.0 * math.pi / 400.0, max_spin_periods=120.0,
                              gate_window=3.0, gate_tol=0.005),
    # linearity check: half the drive amplitude
    "r2_b50_d32_kc_half": R2Case("r2_b50_d32_kc_half", D=32, beta=50.0, u0=0.005,
                                 omega=2.0 * math.pi / 400.0, max_spin_periods=120.0,
                                 gate_window=3.0, gate_tol=0.005),
    # defect A/B: the library's original 4-neighbour interior set
    "r2_b50_d32_int4": R2Case("r2_b50_d32_int4", D=32, beta=50.0,
                              omega=2.0 * math.pi / 400.0, max_spin_periods=120.0,
                              gate_window=3.0, gate_tol=0.005, interior="4"),
    # local (nx = 64) beta ladder: same physics, cheap enough for the 8-core/10-min
    # local envelope; the nx = 128 ladder is submitted separately (cluster).
    "r2l_b25_d16": R2Case("r2l_b25_d16", D=16, beta=25.0, omega=2.0 * math.pi / 200.0,
                          max_spin_periods=60.0, gate_window=3.0, gate_tol=0.005),
    "r2l_b100_d16": R2Case("r2l_b100_d16", D=16, beta=100.0, omega=2.0 * math.pi / 200.0,
                           max_spin_periods=60.0, gate_window=3.0, gate_tol=0.005),
    "r2l_b50_d16_int4": R2Case("r2l_b50_d16_int4", D=16, beta=50.0,
                               omega=2.0 * math.pi / 200.0, max_spin_periods=60.0,
                               gate_window=3.0, gate_tol=0.005, interior="4"),
    "r2l_b50_d16_kc_half": R2Case("r2l_b50_d16_kc_half", D=16, beta=50.0, u0=0.005,
                                  omega=2.0 * math.pi / 200.0, max_spin_periods=60.0,
                                  gate_window=3.0, gate_tol=0.005),
    # short smoke arm
    "smoke": R2Case("smoke", D=16, box=4.0, beta=50.0, omega=2.0 * math.pi / 200.0,
                    u0=0.01, ramp_periods=2.0, max_spin_periods=12.0,
                    measure_periods=2),
}
ARM_ORDER = [k for k in CASES if k != "smoke"] + ["smoke"]


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def interior_nodes(mask: torch.Tensor, mode: str = "8") -> torch.Tensor:
    """Solid cells whose populations may be zeroed without touching fluid.

    ``mode="4"`` is the library's 4-neighbour definition; it leaves solid cells
    that are *diagonally* adjacent to fluid in the interior set, so zeroing them
    injects a zero population into a fluid node every step.  ``mode="8"`` uses the
    full 8-neighbourhood (the population really cannot reach a fluid node).
    """
    fluid = ~mask
    if mode == "none":
        return torch.zeros_like(mask)
    dil = torch.zeros_like(mask)
    offs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if mode == "8":
        offs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for dj, di in offs:
        dil |= torch.roll(torch.roll(fluid, int(dj), 0), int(di), 1)
    return mask & ~dil


def reference_same_mask(nx: int, D: float, levels: tuple[int, ...] = (1, 2)) -> dict:
    """Same-mask potential added mass on the arm's domain + a refined-grid check.

    ``levels`` multiply the lattice resolution: level 2 is the same physical
    problem on ``2*nx`` cells with radius ``2*R``.
    """
    out = {"levels": [], "domain_nx": int(nx), "D": float(D)}
    rows = []
    for lev in levels:
        ref = potential_added_mass(int(nx * lev), int(nx * lev), radius=D / 2.0 * lev)
        area_in = math.pi * (D / 2.0 * lev) ** 2   # same physical circle, lev x finer grid
        mask = circle_mask(int(nx * lev), int(nx * lev), D * lev)
        geo = geometry_readout(mask)
        rows.append({
            "level": lev,
            "nx": int(nx * lev),
            "c_a_raw": float(ref["c_a"]),
            "c_a_area": float(ref["c_a"]) * area_in / geo["n_solid"],
            "mask_area_over_circle": geo["n_solid"] / area_in,
        })
    out["levels"] = rows
    # grid error estimate from the two finest levels (O(h) staircase bias)
    if len(rows) >= 2:
        c1, c2 = rows[-2]["c_a_area"], rows[-1]["c_a_area"]
        out["c_a_best"] = c2
        out["c_a_extrapolated"] = 2.0 * c2 - c1   # Richardson, first order
        out["reference_grid_error_pct"] = 100.0 * abs(c2 - c1) / abs(c2)
    else:
        out["c_a_best"] = rows[-1]["c_a_area"]
        out["c_a_extrapolated"] = float("nan")
        out["reference_grid_error_pct"] = float("nan")
    return out


# --------------------------------------------------------------------------- #
# the arm
# --------------------------------------------------------------------------- #
def run_arm(case: R2Case, threads: int = 4, dtype: torch.dtype = torch.float64) -> dict:
    torch.set_num_threads(int(threads))
    D = case.D
    nx = ny = int(round(case.box * D))
    omega = case.omega
    T = 2.0 * math.pi / omega
    a0 = omega * case.u0
    nu = D * D / (case.beta * T)
    tau = 3.0 * nu + 0.5
    if tau <= 0.52:
        raise ValueError(f"tau={tau:.4f} too close to 0.5 (beta={case.beta}, D={D}, T={T:.0f})")

    mask = circle_mask(nx, ny, D)
    fluid = ~mask
    geo = geometry_readout(mask)
    core = interior_nodes(mask, case.interior)
    normals = _surface_normals(mask)

    rho0 = torch.ones((ny, nx), dtype=dtype)
    zero = torch.zeros_like(rho0)
    f = equilibrium(rho0, zero, zero)
    f = bounce_back_cells(f, mask)
    Fx = torch.zeros((ny, nx), dtype=dtype)
    Fy = torch.zeros_like(Fx)
    n_fluid = int(fluid.sum())

    ramp_steps = max(1, int(round(case.ramp_periods * T)))
    max_spin_steps = int(round(case.max_spin_periods * T))
    measure_steps = int(round(case.measure_periods * T))
    gate_steps = int(round(case.gate_window * T))

    ts: list[float] = []
    f_mem: list[float] = []
    f_trac: list[float] = []
    u_mean: list[float] = []
    p_phys: list[float] = []
    mass: list[float] = []
    drho: list[float] = []
    gate: list[dict] = []

    def harmonic(y: np.ndarray, t0: float, t1: float) -> dict:
        return harmonic_decompose(ts, y, omega, t0=t0, t1=t1, nh=2)

    t_start = time.time()
    step = 0
    converged = False
    measure_start = 0

    # ---- spin-up with gate -------------------------------------------------- #
    amp_prev = None
    if case.forced_spin_periods > 0.0:
        # protocol control: discard a fixed, predeclared number of periods instead of
        # trusting the trailing-window gate (used to prove window independence).
        max_spin_steps = int(round(case.forced_spin_periods * T))
        converged = True
    while step < max_spin_steps:
        step += 1
        t = float(step)
        ramp = 1.0
        if step <= ramp_steps:
            ramp = 0.5 * (1.0 - math.cos(math.pi * step / ramp_steps))
        a_t = a0 * ramp * math.cos(omega * t)
        Fx[fluid] = a_t
        old_solid = f[:, mask].clone()
        f = apply_guo_forcing(f, torch.stack([Fx, Fy]), tau)
        f[:, mask] = old_solid
        f = stream(f)
        fx, _fy = compute_obstacle_forces(f, mask)
        rho, ux, uy = macroscopic(f)
        ftx, _fty = _traction_force(f, mask, tau, normals, rho, ux, uy)
        f = bounce_back_cells(f, mask)
        f[:, core] = 0.0
        ts.append(t)
        f_mem.append(float(fx))
        f_trac.append(ftx)
        u_mean.append(float((rho * ux)[fluid].sum()) / float(rho[fluid].sum()))
        p_phys.append(float((rho * ux)[fluid].sum() + 0.5 * a_t * n_fluid))
        mass.append(float(rho[fluid].sum()))
        drho.append(float((rho[fluid] - 1.0).abs().max()))
        # gate sampling once per period, after the ramp
        if case.forced_spin_periods <= 0.0 and step > ramp_steps and step % int(round(T)) == 0 and step >= gate_steps + int(round(T)):
            t1 = t + 1.0
            w_mem = harmonic(np.asarray(f_mem), t1 - gate_steps, t1)
            w_u = harmonic(np.asarray(u_mean), t1 - gate_steps, t1)
            amp = w_mem["amp"]
            gate.append({
                "step": step,
                "periods": step / T,
                "F_amp": amp,
                "U_amp": w_u["amp"],
                "U_phase": w_u["phase_rad"],
                "drho_max": drho[-1],
                "rel_change": (None if amp_prev is None or amp_prev == 0.0
                               else abs(amp - amp_prev) / abs(amp_prev)),
            })
            if amp_prev is not None and amp_prev > 0.0:
                if abs(amp - amp_prev) / abs(amp_prev) < case.gate_tol:
                    converged = True
                    measure_start = step + 1
                    break
            amp_prev = amp

    if not converged:
        measure_start = step + 1  # honest: report the gate as not closed
    if case.forced_spin_periods > 0.0:
        measure_start = step + 1

    # ---- measurement window ------------------------------------------------- #
    n_total = measure_start + measure_steps - 1
    while step < n_total:
        step += 1
        t = float(step)
        a_t = a0 * math.cos(omega * t)
        Fx[fluid] = a_t
        old_solid = f[:, mask].clone()
        f = apply_guo_forcing(f, torch.stack([Fx, Fy]), tau)
        f[:, mask] = old_solid
        f = stream(f)
        fx, _fy = compute_obstacle_forces(f, mask)
        rho, ux, uy = macroscopic(f)
        ftx, _fty = _traction_force(f, mask, tau, normals, rho, ux, uy)
        f = bounce_back_cells(f, mask)
        f[:, core] = 0.0
        ts.append(t)
        f_mem.append(float(fx))
        f_trac.append(ftx)
        u_mean.append(float((rho * ux)[fluid].sum()) / float(rho[fluid].sum()))
        p_phys.append(float((rho * ux)[fluid].sum() + 0.5 * a_t * n_fluid))
        mass.append(float(rho[fluid].sum()))
        drho.append(float((rho[fluid] - 1.0).abs().max()))

    wall = time.time() - t_start

    # ---- analysis ----------------------------------------------------------- #
    t0 = float(measure_start)
    t1 = t0 + float(measure_steps)
    dec_mem = harmonic(np.asarray(f_mem), t0, t1)
    dec_trac = harmonic(np.asarray(f_trac), t0, t1)
    dec_u = harmonic(np.asarray(u_mean), t0, t1)
    dec_p = harmonic(np.asarray(p_phys), t0, t1)

    # E3: F_on_body = a0 N_fluid cos(wt) - dP/dt,  P = sum_fluid(rho u) + a0 N_f/2
    f_e3_cos = a0 * n_fluid - omega * dec_p["sin"]
    f_e3_sin = 0.0 + omega * dec_p["cos"]
    kpi = {}
    for tag, dec, dd in (
        ("E1_mem_area", dec_mem, geo["D_area"]),
        ("E1_mem_halfway", dec_mem, geo["D_halfway"]),
        ("E2_trac_area", dec_trac, geo["D_area"]),
        ("E2_trac_halfway", dec_trac, geo["D_halfway"]),
    ):
        kpi[tag] = {"C_I": inertia_coefficient(dec["cos"], a0, dd),
                    "C_Q": inertia_coefficient(dec["sin"], a0, dd), "D": dd}
    kpi["E3_global_area"] = {"C_I": inertia_coefficient(f_e3_cos, a0, geo["D_area"]),
                             "C_Q": inertia_coefficient(f_e3_sin, a0, geo["D_area"]),
                             "D": geo["D_area"]}

    ref = reference_same_mask(nx, D, levels=(1, 2))
    c_ref = ref["c_a_best"]                       # potential, same mask, 2x refined grid
    c_rich = ref["c_a_extrapolated"]
    c_level1 = ref["levels"][0]["c_a_area"]       # potential, same mask, arm resolution
    stokes = stokes_cylinder_coefficients(case.beta)
    # product reference: outer potential flow sees the confined staircase mask, the
    # inner Stokes layer (delta < 1/10 of the gap to the wall) sees a flat wall, so the
    # two corrections factorise to O(delta/gap).
    c_ref_viscous = c_level1 * stokes["Gamma_re"]
    area_ratio = geo["n_solid"] / (math.pi * (D / 2.0) ** 2)

    win = np.arange(len(ts)) >= int(t0)
    mass_w = np.asarray(mass)[win]
    drho_w = np.asarray(drho)[win]
    m_drift = float((mass_w[-1] - mass_w[0]) / mass_w[0])

    # acoustic context
    k_ac = 2.0 * math.pi / nx
    gamma_ac = nu * k_ac ** 2
    ac_freqs = [2.0 * math.pi * CS * n / nx for n in (1, 2, 3)]
    detune = [abs(omega - w) / max(gamma_ac, 1e-30) for w in ac_freqs[:2]]

    result = {
        "case": asdict(case),
        "status": "ok",
        "geometry": geo,
        "numerics": {
            "nx": nx, "D": D, "n_fluid": n_fluid, "n_core_zeroed": int(core.sum()),
            "nu": nu, "tau": tau, "T_steps": T, "omega": omega, "a0": a0,
            "Ma": case.u0 * math.sqrt(3.0), "delta_over_R": 2.0 * math.sqrt(2.0 / (case.beta * math.pi)),
            "Re": case.u0 * D / nu, "KC": 2.0 * math.pi * case.u0 / (omega * D),
            "acoustic_omega": ac_freqs, "acoustic_linewidth_gamma": gamma_ac,
            "detune_in_linewidths": detune,
            "wall_s": round(wall, 2), "n_steps_run": step, "dtype": str(dtype).split(".")[-1],
        },
        "protocol": {
            "ramp_periods": case.ramp_periods,
            "gate_closed": bool(converged),
            "gate_periods_samples": len(gate),
            "gate_last": gate[-1] if gate else None,
            "spin_up_periods": (measure_start - 1) / T,
            "measure_start_step": measure_start,
            "measure_periods": case.measure_periods,
            "E1_resid_rel": dec_mem["resid_rel"],
            "E2_resid_rel": dec_trac["resid_rel"],
            "E3_resid_rel": dec_p["resid_rel"],
            "U_resid_rel": dec_u["resid_rel"],
        },
        "gate_trace": gate,
        "achieved": {
            "U_amp_box_mean": dec_u["amp"],
            "U_over_u0": dec_u["amp"] / case.u0,
            "U_phase": dec_u["phase_rad"],
            "KC_measured": 2.0 * math.pi * dec_u["amp"] / (omega * geo["D_halfway"]),
            "drho_max_window": float(drho_w.max()),
            "mass_rel_drift_window": m_drift,
        },
        "harmonic": {"E1_mem": dec_mem, "E2_trac": dec_trac, "E3_P": dec_p, "U": dec_u},
        "kpi": kpi,
        "reference": {
            "same_mask": ref,
            "c_a_area": c_ref,
            "c_a_level1": c_level1,
            "c_a_extrapolated": c_rich,
            "analytic_unbounded": 1.0,
            "stokes": stokes,
            "c_ref_viscous": c_ref_viscous,
            "mask_area_ratio": area_ratio,
        },
        "err_inphase_pct": {
            # inviscid anchors (round-1 references; wrong reference at finite beta)
            "E1_vs_potential": 100.0 * (kpi["E1_mem_area"]["C_I"] - c_ref) / c_ref,
            "E3_vs_potential": 100.0 * (kpi["E3_global_area"]["C_I"] - c_ref) / c_ref,
            # viscous anchor: same-mask potential x exact unsteady-Stokes Gamma(beta)
            "E1_vs_stokes": 100.0 * (kpi["E1_mem_area"]["C_I"] - c_ref_viscous) / c_ref_viscous,
            "E2_vs_stokes": 100.0 * (kpi["E2_trac_area"]["C_I"] - c_ref_viscous) / c_ref_viscous,
            "E3_vs_stokes": 100.0 * (kpi["E3_global_area"]["C_I"] - c_ref_viscous) / c_ref_viscous,
            # per-pi R^2 normalisation of the same comparison
            "E1_per_piR2_vs_Gamma_re": 100.0 * (kpi["E1_mem_area"]["C_I"] * area_ratio
                                               - stokes["Gamma_re"]) / stokes["Gamma_re"],
            "C_Q_vs_Gamma_im": 100.0 * (kpi["E1_mem_area"]["C_Q"] - stokes["C_Q_viscous"])
                             / stokes["C_Q_viscous"],
        },
        "stokes_layer_cells": stokes["delta_over_R"] * D / 2.0,
        "estimator_spread_pct": abs(kpi["E1_mem_area"]["C_I"] - kpi["E2_trac_area"]["C_I"]),
        "stability_ok": bool(np.all(np.isfinite(f_mem)) and np.all(np.isfinite(drho))),
    }
    return result


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", default="")
    ap.add_argument("--arms", default="", help="comma separated")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    if args.list:
        for name in ARM_ORDER:
            print(name, asdict(CASES[name]))
        return 0
    names = [n for n in (args.arms.split(",") if args.arms else
                         ([args.arm] if args.arm else ARM_ORDER)) if n]
    out_root = Path(args.out) if args.out else Path(__file__).resolve().parent / "runs"
    for name in names:
        case = CASES[name]
        print(f"[run] {name}: D={case.D} box={case.box}D beta={case.beta} "
              f"omega={case.omega:.5f} u0={case.u0} interior={case.interior}", flush=True)
        try:
            result = run_arm(case, threads=args.threads)
        except Exception as exc:  # honest failure is recorded, not hidden
            result = {"case": asdict(case), "status": "failed", "failure_reason": repr(exc)}
        d = out_root / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps({k: result.get(k) for k in
                          ("status", "protocol", "kpi", "err_inphase_pct",
                           "achieved", "estimator_spread_pct")},
                         indent=2, default=str)[:2000], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())