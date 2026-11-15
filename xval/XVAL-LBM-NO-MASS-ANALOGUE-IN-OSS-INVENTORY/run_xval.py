"""XVAL-LBM-NO-MASS-ANALOGUE-IN-OSS-INVENTORY -- verification-debt closure.

Purpose
-------
The anchor was closed by mission m21062 with a report claiming that the library
module ``tensorlbm.oscillatory_cylinder_inertia`` measures an added-mass
coefficient ``C_I`` of 2.16 / 2.10 / 1.99 at D = 8/12/16 (box = 4D, beta = 50,
KC = 0.2) and that this is an "honest FAIL, +75..99 % above the potential-flow
anchor, not a resolution artefact".  That commit left no run script, no raw run
JSONs, and its ``repro_cmd`` pointed at a file that does not exist, so the claim
could not be re-executed.

This script re-executes the measurement from the library code (not modified) and
adds the controls the first pass was missing:

  rep       A. reproduce the m21062 ladder (D = 8/12/16, beta = 50).
  protocol  B. spin-up / window convergence at fixed physics (D = 16).
  beta      C. Stokes ladder at a converged spin-up: does C_I approach the
               same-mask potential value as beta -> inf?
  kc        D. Keulegan-Carpenter ladder: linear-response check (KC -> 0).
  box       E. domain ladder: the drive frequency is compared with the acoustic
               eigenfrequencies of the periodic box (0.0785 vs 0.0566 / 0.113 at
               nx = 64) -- a resonant compressible channel would show up as a
               box-mean pressure oscillation that must vanish for an
               incompressible added-mass measurement.
  e3        F. a third force estimator from the global fluid momentum balance
               ``F = a(t) N_fluid - dP/dt`` (independent of E1/E2), including the
               Guo half-force correction of the stored momentum.
  ref       G. grid convergence of the same-mask potential reference itself.

Every number in the report is recomputed by ``analyze.py`` from the raw JSONs
written here; nothing is typed by hand.

Usage
-----
    python xval/XVAL-LBM-NO-MASS-ANALOGUE-IN-OSS-INVENTORY/run_xval.py --group all
    python xval/XVAL-LBM-NO-MASS-ANALOGUE-IN-OSS-INVENTORY/run_xval.py --group protocol
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from tensorlbm.boundary.boundaries import bounce_back_cells, compute_obstacle_forces
from tensorlbm.boussinesq import apply_guo_forcing
from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.oscillatory_cylinder_inertia import (
    ArmCase,
    _interior_nodes,
    _surface_normals,
    _traction_force,
    circle_mask,
    geometry_readout,
    harmonic_decompose,
    inertia_coefficient,
    potential_reference,
    run_arm,
)
from tensorlbm.solver import stream

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------- arm tables
BASE = dict(D=16, box=4.0, kc=0.2, beta=50.0, u0=0.04)


def _arm(name: str, **kw) -> dict:
    cfg = dict(BASE)
    cfg.update(kw)
    cfg["name"] = name
    return {"name": name, "cfg": cfg}


REP_ARMS = [_arm(f"rep_d{D}", D=D, transient=1.5, measure=3.0) for D in (8, 12, 16)]

PROTOCOL_ARMS = [
    _arm("proto_t1p5_m3", transient=1.5, measure=3.0),
    _arm("proto_t3_m3", transient=3.0, measure=3.0),
    _arm("proto_t5_m5", transient=5.0, measure=5.0),
    _arm("proto_t8_m5", transient=8.0, measure=5.0),
    _arm("proto_t20_m5", transient=20.0, measure=5.0),
    _arm("proto_t40_m5", transient=40.0, measure=5.0),
]

BETA_ARMS = [
    _arm(f"beta_{int(b)}", beta=float(b), transient=40.0, measure=5.0)
    for b in (50, 200, 800)
]

KC_ARMS = [
    _arm(f"kc_{str(k).replace('.', 'p')}", kc=float(k), transient=40.0, measure=5.0)
    for k in (0.05, 0.1, 0.2)
]

BOX_ARMS = [
    _arm(f"box_{int(b)}", box=float(b), transient=40.0, measure=5.0) for b in (4, 6)
]

E3_ARMS = [
    _arm("e3_beta_50", beta=50.0, transient=40.0, measure=5.0),
    _arm("e3_beta_800", beta=800.0, transient=40.0, measure=5.0),
]

GROUPS = {
    "rep": REP_ARMS,
    "protocol": PROTOCOL_ARMS,
    "beta": BETA_ARMS,
    "kc": KC_ARMS,
    "box": BOX_ARMS,
    "e3": E3_ARMS,
}


# ------------------------------------------- extended arm (diagnostics + E3)
def diagnostic_arm(case: ArmCase, threads: int = 4, dtype: torch.dtype = torch.float64):
    """``run_arm`` integration plus global diagnostics.

    Records, on every step:
      * ``F_mem`` / ``F_trac``  -- the library's two force estimators,
      * ``P``  = sum_fluid rho u  -- fluid momentum (E3 estimator input),
      * ``pbar`` = box-mean pressure, ``drho_max`` = max |rho - 1|,
      * ``mass`` = sum_fluid rho (Guo forcing must not create mass),
      * ``p_front`` / ``p_back`` -- surface-adjacent pressure dipole.

    E3 (global momentum balance of the fluid subdomain, Guo forcing with density
    ``rho`` and force density ``a`` on fluid nodes only):

        d/dt sum rho u = a N_fluid - F_body + (half-force correction)

    ``macroscopic()`` returns ``sum_c c f`` which for Guo forcing is
    ``rho u - F/2``, so the stored momentum is ``P + a(t) N_fluid / 2`` and

        F_body^cos = a0 N_fluid - omega P_s,
        F_body^sin = omega P_c + omega a0 N_fluid / 2 .
    """
    torch.set_num_threads(int(threads))
    D = case.D
    nx = ny = int(round(case.box * D))
    T = case.kc * D / case.u0
    omega = 2.0 * math.pi / T
    a0 = omega * case.u0
    nu = D * D / (case.beta * T)
    tau = 3.0 * nu + 0.5
    if tau <= 0.5:
        raise ValueError(f"tau={tau:.5f} <= 0.5: beta={case.beta} not reachable at D={D}")
    mask = circle_mask(nx, ny, D)
    fluid = ~mask
    geo = geometry_readout(mask)
    interior = _interior_nodes(mask)
    normals = _surface_normals(mask)
    nfluid = int(fluid.sum())
    cx = (nx - 1) / 2.0
    jx, ix, iy = int(cx + D / 2) + 1, int(cx - D / 2) - 1, int(cx)

    rho0 = torch.ones((ny, nx), dtype=dtype)
    zero = torch.zeros_like(rho0)
    f = equilibrium(rho0, zero, zero)
    f = bounce_back_cells(f, mask)
    Fx = torch.zeros((ny, nx), dtype=dtype)
    Fy = torch.zeros_like(Fx)

    start = int(round(case.transient * T))
    n_steps = start + int(round(case.measure * T))
    ts, f_mem, f_trac, P, pbar, drho, mass, pf, pb = [], [], [], [], [], [], [], [], []
    t_wall0 = time.time()
    for step in range(1, n_steps + 1):
        t = float(step)
        a_t = a0 * math.cos(omega * t)
        Fx[fluid] = a_t
        old = f[:, mask].clone()
        f = apply_guo_forcing(f, torch.stack([Fx, Fy]), tau)
        f[:, mask] = old
        f = stream(f)
        fx, _ = compute_obstacle_forces(f, mask)
        rho, ux, uy = macroscopic(f)
        ftx, _ = _traction_force(f, mask, tau, normals, rho, ux, uy)
        P.append(float((rho * ux)[fluid].sum()))
        pbar.append(float(rho[fluid].mean()) / 3.0)
        drho.append(float((rho[fluid] - 1.0).abs().max()))
        mass.append(float(rho[fluid].sum()))
        pf.append(float(rho[iy, jx]) / 3.0)
        pb.append(float(rho[iy, ix]) / 3.0)
        f = bounce_back_cells(f, mask)
        f[:, interior] = 0.0
        ts.append(t)
        f_mem.append(float(fx))
        f_trac.append(float(ftx))
    wall = time.time() - t_wall0

    t0 = float(start)
    t1 = t0 + case.measure * T
    dec = {tag: harmonic_decompose(ts, y, omega, t0=t0, t1=t1, nh=2)
           for tag, y in (("E1", f_mem), ("E2", f_trac), ("P", P),
                          ("pbar", pbar), ("pf", pf), ("pb", pb))}
    vol = math.pi * geo["D_area"] ** 2 / 4.0
    f_cos3 = a0 * nfluid - omega * dec["P"]["sin"]
    f_sin3 = omega * dec["P"]["cos"] + omega * a0 * nfluid / 2.0
    kpi = {}
    for tag, d in (("E1_mem_area", dec["E1"]), ("E2_trac_area", dec["E2"])):
        kpi[tag] = {"C_I": inertia_coefficient(d["cos"], a0, geo["D_area"]),
                    "C_Q": inertia_coefficient(d["sin"], a0, geo["D_area"]),
                    "resid_rel": d["resid_rel"]}
    kpi["E3_momentum_area"] = {"C_I": f_cos3 / (vol * a0), "C_Q": f_sin3 / (vol * a0),
                               "resid_rel": dec["P"]["resid_rel"],
                               "F_cos": f_cos3, "F_sin": f_sin3}
    pot = potential_reference(nx, ny, D)
    # acoustic / compressible channel: the box-mean pressure of a purely
    # incompressible added-mass response is zero, so |pbar|/|p_dipole| is a
    # direct measure of how much of the force channel is compressible.
    p_dipole = math.hypot(dec["pf"]["cos"] - dec["pb"]["cos"],
                          dec["pf"]["sin"] - dec["pb"]["sin"])
    return {
        "case": asdict(case),
        "status": "ok",
        "geometry": geo,
        "numerics": {"nx": nx, "ny": ny, "n_fluid": nfluid, "nu": nu, "tau": tau,
                     "T_steps": T, "omega": omega, "a0": a0, "n_steps": n_steps,
                     "measure_start_step": start, "wall_s": round(wall, 2),
                     "acoustic_omega_k1": 2 * math.pi * (1 / math.sqrt(3)) / nx,
                     "acoustic_omega_k2": 4 * math.pi * (1 / math.sqrt(3)) / nx},
        "kpi": kpi,
        "diagnostics": {
            "pbar_amp": dec["pbar"]["amp"], "pbar_resid": dec["pbar"]["resid_rel"],
            "drho_max_last_period": max(drho[-int(round(T)):]),
            "mass_rel_drift": (mass[-1] - mass[0]) / mass[0],
            "p_dipole_amp": p_dipole,
            "p_dipole_over_potential": p_dipole / (2 * a0 * D / 2),
            "pbar_over_dipole": dec["pbar"]["amp"] / max(p_dipole, 1e-30),
            "U_amp_box_mean": dec["P"]["amp"] / max(float(fluid.sum()), 1.0),
        },
        "harmonic": {k: {"cos": v["cos"], "sin": v["sin"], "amp": v["amp"],
                         "resid_rel": v["resid_rel"]} for k, v in dec.items()},
        "reference": {k: v for k, v in pot.items() if not isinstance(v, bool)},
    }


# ----------------------------------------------------------------------- driver
def run_group(name: str, out_root: Path, threads: int) -> list[dict]:
    rows = []
    for arm in GROUPS[name]:
        case = ArmCase(**arm["cfg"])
        print(f"[{name}] {arm['name']}: D={case.D} beta={case.beta} KC={case.kc} "
              f"box={case.box}D transient={case.transient} measure={case.measure}", flush=True)
        t0 = time.time()
        try:
            if name == "e3":
                res = diagnostic_arm(case, threads=threads)
            else:
                res, _trace = run_arm(case, threads=threads)
                res["status"] = "ok"
        except Exception as exc:  # honest failure: recorded, never hidden
            res = {"case": asdict(case), "status": "failed", "failure_reason": repr(exc)}
        res["arm"] = arm["name"]
        res["group"] = name
        res["wall_s_total"] = round(time.time() - t0, 2)
        d = out_root / arm["name"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "result.json").write_text(json.dumps(res, indent=2, sort_keys=True, default=str) + "\n")
        e1 = res.get("kpi", {}).get("E1_mem_area", {})
        print(f"    status={res['status']} E1 C_I={e1.get('C_I')} C_Q={e1.get('C_Q')} "
              f"resid={e1.get('resid_rel')} wall={res['wall_s_total']}s", flush=True)
        rows.append({"arm": arm["name"], "group": name, "status": res["status"],
                     "cfg": arm["cfg"]})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--group", default="all", choices=[*GROUPS, "all"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=str(HERE / "runs"))
    args = ap.parse_args(argv)
    out_root = Path(args.out)
    names = list(GROUPS) if args.group == "all" else [args.group]
    manifest = []
    for name in names:
        manifest += run_group(name, out_root, args.threads)
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[done] {len(manifest)} arms -> {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())