#!/usr/bin/env python3
"""XVAL-LBM-NO-DEM-REPO-IN-OSS-INVENTORY — round 2 driver.

The OSS cache (``/root/oss-cache``) holds three LBM/SPH repositories and **no**
DEM / granular-contact repository (see ``survey_oss.md``).  A DEM leg therefore
cannot be cross-validated against an OSS twin; this driver validates it against
closed-form and literature references instead (``lineage.md``), and reports the
state of the fluid-side (LBM) leg that the coupling consumes.

Arms (all predeclared in ``predeclared.json`` before the first run):

A1/A2  particle-wall restitution           vs e = exp(-pi zeta/sqrt(1-zeta^2))
B1/B2  contact duration / peak overlap     vs t_c = pi/(omega_n sqrt(1-zeta^2))
C1/C2  Stokes-closure settling             vs u_t = |drho| g D^2/(18 mu)   [H3]
D1/D2  Schiller-Naumann settling           vs Clift-Gauvin terminal velocity
E      confined-box LBM Cd(Re)             vs Clift-Gauvin (fluid-side leg)
F      momentum-exchange ledger            vs exact closure (< 1e-10)

Everything runs through the *real* modules of this repository
(``tensorlbm.dem.DEMSolver``, ``tensorlbm.dem.contact.LinearSpringDashpot``,
``tensorlbm.dem.cw_coupling.MomentumExchange``); nothing is re-implemented here
except the closed-form references, which are written out in this file so that a
reader can check them line by line.

Usage
-----
    python run_xval.py               # full run, writes results.json
    python run_xval.py --quick       # coarse time steps only
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from tensorlbm.dem import DEMConfig, DEMSolver, IceParticle, LinearSpringDashpot  # noqa: E402
from tensorlbm.dem.contact import detect_particle_wall_contact  # noqa: E402
from tensorlbm.dem.cw_coupling import ConstantCwCallback, MomentumExchange  # noqa: E402
from tensorlbm.drag_closure import cd_clift_gauvin, terminal_velocity  # noqa: E402

torch.set_num_threads(1)

RHO_P = 917.0      # ice
RHO_F = 1000.0     # water
MU = 1.0e-3        # Pa.s
G = 9.81           # m/s^2

F64 = torch.float64


def rel_err(sim: float, ref: float) -> float:
    """Signed relative error (sim - ref)/ref."""
    if ref == 0.0:
        return float("nan")
    return (sim - ref) / ref


# ---------------------------------------------------------------------------
# Closed-form references (written out, not called from the code under test)
# ---------------------------------------------------------------------------


def analytic_restitution(zeta: float) -> float:
    """Kelvin-Voigt restitution: e = exp(-pi zeta / sqrt(1 - zeta^2))."""
    return math.exp(-math.pi * zeta / math.sqrt(1.0 - zeta * zeta))


def analytic_contact(zeta: float, omega_n: float, v0: float) -> dict:
    """Contact duration and peak overlap of the under-damped linear oscillator.

    x(t) = (v0/omega_d) exp(-zeta omega_n t) sin(omega_d t), x(0)=0, x'(0)=v0.
    """
    omega_d = omega_n * math.sqrt(1.0 - zeta * zeta)
    t_c = math.pi / omega_d
    t_star = math.atan(omega_d / (zeta * omega_n)) / omega_d
    x_max = (v0 / omega_d) * math.exp(-zeta * omega_n * t_star) * math.sin(omega_d * t_star)
    return {"t_c": t_c, "delta_max": x_max, "omega_n": omega_n, "omega_d": omega_d}


# ---------------------------------------------------------------------------
# Arms A/B — particle-wall contact
# ---------------------------------------------------------------------------


def contact_arm(steps_per_contact: int, zeta: float = 0.2,
                radius: float = 1.0e-3, k: float = 1.0e5,
                v0: float = 0.05, clip: bool = True) -> dict:
    """Drop a sphere onto a wall and measure e, t_c and delta_max."""
    mass = RHO_P * (4.0 / 3.0) * math.pi * radius ** 3
    omega_n = math.sqrt(k / mass)
    eta = 2.0 * zeta * math.sqrt(k * mass)     # zeta = eta / (2 sqrt(k m))
    ref = analytic_contact(zeta, omega_n, v0)
    dt = ref["t_c"] / steps_per_contact

    model_cls = LinearSpringDashpot if clip else _NoClipKelvinVoigt
    model = model_cls(
        normal_stiffness=k,
        tangential_stiffness=0.0,
        normal_damping=eta,
        tangential_damping=0.0,
        friction_coefficient=0.0,
    )
    cfg = DEMConfig(
        time_step=dt,
        gravity=torch.zeros(3, dtype=F64),          # isolate contact from gravity
        contact_model=model,
        domain_bounds=(torch.tensor([-1.0, -1.0, 0.0], dtype=F64),
                       torch.tensor([1.0, 1.0, 1.0], dtype=F64)),
        enable_wall_contacts=True,
        drag_model="none",
    )
    solver = DEMSolver(cfg)
    particle = IceParticle(
        position=torch.tensor([0.0, 0.0, radius], dtype=F64),
        velocity=torch.tensor([0.0, 0.0, -v0], dtype=F64),
        angular_velocity=torch.zeros(3, dtype=F64),
        radius=radius,
        density=RHO_P,
    )
    solver.initialize([particle])

    wall_pos = torch.tensor([0.0, 0.0, 0.0], dtype=F64)
    wall_normal = torch.tensor([0.0, 0.0, 1.0], dtype=F64)

    n_steps = 6 * steps_per_contact
    overlap_max = 0.0
    n_contact = 0
    tensile_clipped = 0
    vz_end_of_contact = 0.0
    contact_started = False
    for _ in range(n_steps):
        in_contact, overlap, _ = detect_particle_wall_contact(particle, wall_pos, wall_normal)
        if in_contact:
            contact_started = True
            n_contact += 1
            overlap_max = max(overlap_max, overlap)
            v_n = float(particle.velocity[2])
            if k * overlap - eta * v_n < 0.0:
                tensile_clipped += 1          # the no-tension clip is active
            vz_end_of_contact = float(particle.velocity[2])
        elif contact_started and n_contact > 0 and abs(float(particle.velocity[2])) > 0.0:
            break                              # left contact: free flight from here
        solver.step(None)

    e_sim = float(particle.velocity[2]) / v0   # vz > 0 after the bounce
    return {
        "steps_per_contact": steps_per_contact,
        "no_tension_clip": clip,
        "dt_s": dt,
        "zeta": zeta,
        "eta_Ns_per_m": eta,
        "mass_kg": mass,
        "omega_n_rad_s": omega_n,
        "e_sim": e_sim,
        "e_analytic": analytic_restitution(zeta),
        "e_rel_err": rel_err(e_sim, analytic_restitution(zeta)),
        "t_c_sim_s": n_contact * dt,
        "t_c_analytic_s": ref["t_c"],
        "t_c_rel_err": rel_err(n_contact * dt, ref["t_c"]),
        "delta_max_sim_m": overlap_max,
        "delta_max_analytic_m": ref["delta_max"],
        "delta_max_rel_err": rel_err(overlap_max, ref["delta_max"]),
        "delta_max_over_radius": overlap_max / radius,
        "n_contact_steps": n_contact,
        "tensile_clipped_steps": tensile_clipped,
        "vz_at_separation_m_s": vz_end_of_contact,
        "e_ratio_sim_over_analytic": e_sim / analytic_restitution(zeta),
    }


class _NoClipKelvinVoigt(LinearSpringDashpot):
    """Diagnostic control: the same Kelvin-Voigt law **without** the no-tension clip.

    ``LinearSpringDashpot`` clamps ``f_n = max(0, k delta - eta v_n)``; during the
    restitution phase the dashpot term makes the force negative, so the clamp
    removes part of the damping.  This control keeps the tension, which isolates
    the clamp's contribution from the integrator's.
    """

    def compute_contact_force(self, p1, p2, normal, overlap, relative_velocity):
        info = super().compute_contact_force(p1, p2, normal, overlap, relative_velocity)
        v_n = float(torch.dot(relative_velocity, normal))
        f_unclipped = self.normal_stiffness * overlap - self.normal_damping * v_n
        delta = f_unclipped - float(torch.dot(info.normal_force, normal))
        info.normal_force = info.normal_force + delta * normal
        info.total_force = info.total_force + delta * normal
        return info


# ---------------------------------------------------------------------------
# Arms C/D — settling under a drag closure
# ---------------------------------------------------------------------------


def settling_arm(diameter: float, drag_model: str, dt_div: int,
                 reference: str = "clift_gauvin") -> dict:
    """Settle one ice sphere in quiescent water and measure the terminal speed.

    Buoyancy is folded into ``g_eff = g (1 - rho_f/rho_p)`` (standard DEM
    reduction: the solver applies the full particle weight, so the fluid
    pressure force is carried by the effective gravity).  For ice in water the
    net weight is negative and the sphere rises.
    """
    g_eff = G * (1.0 - RHO_F / RHO_P)
    radius = 0.5 * diameter
    tau_p = 2.0 * RHO_P * radius ** 2 / (9.0 * MU)     # Stokes relaxation time
    dt = tau_p / dt_div

    cfg = DEMConfig(
        time_step=dt,
        gravity=torch.tensor([0.0, 0.0, -g_eff], dtype=F64),
        contact_model=LinearSpringDashpot(),
        domain_bounds=None,
        enable_wall_contacts=False,
        drag_model=drag_model,
        fluid_density=RHO_F,
        fluid_viscosity=MU,
    )
    solver = DEMSolver(cfg)
    particle = IceParticle(
        position=torch.zeros(3, dtype=F64),
        velocity=torch.zeros(3, dtype=F64),
        angular_velocity=torch.zeros(3, dtype=F64),
        radius=radius,
        density=RHO_P,
    )
    solver.initialize([particle])

    zeros = torch.zeros(3, dtype=F64)
    n_steps = 30 * dt_div
    samples = []
    for i in range(n_steps):
        solver.step(zeros)
        if i >= n_steps // 2:
            samples.append(float(particle.velocity[2]))
    u_sim = sum(samples) / len(samples)

    u_stokes = abs(RHO_P - RHO_F) * G * diameter ** 2 / (18.0 * MU)
    u_ref = abs(terminal_velocity(diameter, RHO_P, RHO_F, MU, gravity=G, model=reference))
    re_sim = RHO_F * abs(u_sim) * diameter / MU
    return {
        "diameter_m": diameter,
        "drag_model": drag_model,
        "dt_div": dt_div,
        "dt_s": dt,
        "tau_p_s": tau_p,
        "u_sim_m_s": u_sim,
        "u_reference_m_s": u_ref,
        "reference": reference,
        "u_stokes_m_s": u_stokes,
        "rel_err_vs_reference": rel_err(abs(u_sim), u_ref),
        "rel_err_vs_stokes": rel_err(abs(u_sim), u_stokes),
        "Re_sim": re_sim,
        "converged_tail_drift_rel": (
            abs(samples[-1] - samples[0]) / max(abs(samples[-1]), 1e-30)
        ),
    }


# ---------------------------------------------------------------------------
# Arm E — fluid-side (LBM) leg, read from the repo's own confined-box run
# ---------------------------------------------------------------------------


def lbm_leg() -> dict:
    """Compare the repo's LBM Cd(Re) (confined periodic box) with Clift-Gauvin."""
    path = _REPO / "benchmarks" / "pending" / "sphere_drag_closure" / "results.json"
    rows = []
    if not path.exists():
        return {"available": False, "path": str(path), "rows": rows}
    data = json.loads(path.read_text())
    for rec in data.get("cd_table", []):
        re = float(rec["Re"])
        cd_ref = float(rec["cd_clift_gauvin"])
        best = (rec.get("cd_lbm_lambda_inf") or rec.get("cd_lbm_D_inf")
                or rec.get("cd_lbm_lam_1_8"))
        best_label = ("lambda_extrapolated" if rec.get("cd_lbm_lambda_inf")
                      else "resolution_extrapolated" if rec.get("cd_lbm_D_inf")
                      else "raw_run_lambda")
        for key, label in (("cd_lbm_lam_1_8", "raw_run_lambda"),
                           ("cd_lbm_D_inf", "resolution_extrapolated"),
                           ("cd_lbm_hasimoto_corrected", "hasimoto_corrected"),
                           ("__best__", best_label)):
            val = best if key == "__best__" else rec.get(key)
            if val is None:
                continue
            rows.append({
                "Re": re,
                "variant": label,
                "cd_lbm": float(val),
                "cd_clift_gauvin": cd_ref,
                "rel_err": rel_err(float(val), cd_ref),
                "tag": rec.get("tag"),
                "cd_method": rec.get("cd_method"),
            })
    worst = max(rows, key=lambda r: abs(r["rel_err"])) if rows else None
    return {
        "available": True,
        "path": str(path.relative_to(_REPO)),
        "source_node": data.get("node"),
        "source_mission": data.get("mission"),
        "rows": rows,
        "worst_row": worst,
        "n_within_15pct": sum(1 for r in rows if abs(r["rel_err"]) < 0.15),
        "n_rows": len(rows),
        "n_within_15pct_best": sum(1 for r in rows
                                   if r["variant"] != "hasimoto_corrected" and abs(r["rel_err"]) < 0.15),
    }


# ---------------------------------------------------------------------------
# Arm F — momentum-exchange ledger (exact invariant of the coupling interface)
# ---------------------------------------------------------------------------


def ledger_arm() -> dict:
    """Run the callback-driven drag path and close the impulse ledger."""
    radius = 1.0e-3
    dt = 1.0e-5
    cfg = DEMConfig(
        time_step=dt,
        gravity=torch.zeros(3, dtype=F64),
        contact_model=LinearSpringDashpot(),
        domain_bounds=None,
        enable_wall_contacts=False,
        drag_model="cw_callback",
        cw_callback=ConstantCwCallback(5.5e-3),
        fluid_density=RHO_F,
        fluid_viscosity=MU,
    )
    solver = DEMSolver(cfg)
    solver.exchange = MomentumExchange(rho_water=RHO_F)
    particle = IceParticle(
        position=torch.zeros(3, dtype=F64),
        velocity=torch.tensor([0.0, 0.0, 0.01], dtype=F64),
        angular_velocity=torch.zeros(3, dtype=F64),
        radius=radius,
        density=RHO_P,
    )
    solver.initialize([particle])
    fluid = torch.tensor([0.0, 0.0, 0.002], dtype=F64)   # counter-flowing water
    for _ in range(50):
        solver.step(fluid)
    res = solver.exchange.residual()
    return {
        "relative_residual": res["relative_residual"],
        "n_events": res["n_events"],
        "particle_impulse": res["particle_impulse"],
        "water_reaction": res["water_reaction"],
        "net_impulse": res["net_impulse"],
        "c_w": 5.5e-3,
        "dt_s": dt,
        "n_steps": 50,
    }


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="coarse time steps only")
    ap.add_argument("--out", default=str(_HERE.parent / "results.json"))
    args = ap.parse_args()

    steps = (40, 80, 160) if not args.quick else (20, 40)
    tdivs = (25, 100) if not args.quick else (10, 25)
    diameters = (1.0e-4, 2.0e-4, 5.0e-4)

    out = {
        "anchor": "XVAL-LBM-NO-DEM-REPO-IN-OSS-INVENTORY",
        "mission_id": 18215,
        "repo": "TensorLBM",
        "round": 2,
        "constants": {"rho_p": RHO_P, "rho_f": RHO_F, "mu": MU, "g": G},
        "arms": {},
    }

    out["arms"]["contact"] = [contact_arm(n) for n in steps]
    out["arms"]["contact_no_clip_control"] = [
        contact_arm(n, clip=False) for n in steps
    ]
    out["arms"]["stokes_settling"] = [
        settling_arm(d, "stokes", n, reference="stokes") for d in diameters for n in tdivs
    ]
    out["arms"]["schiller_naumann_settling"] = [
        settling_arm(d, "schiller_naumann", n) for d in diameters for n in tdivs
    ]
    out["arms"]["lbm_leg"] = lbm_leg()
    out["arms"]["ledger"] = ledger_arm()

    # ---- verdicts -------------------------------------------------------
    def within(rows, key, tol=0.15):
        return all(abs(r[key]) < tol for r in rows)

    contact = out["arms"]["contact"]
    sn = out["arms"]["schiller_naumann_settling"]
    st = out["arms"]["stokes_settling"]
    lbm = out["arms"]["lbm_leg"]
    ledger = out["arms"]["ledger"]

    out["verdict"] = {
        "A_contact_restitution_within_15pct": within(contact, "e_rel_err"),
        "A_contact_restitution_within_2pct": within(contact, "e_rel_err", 0.02),
        "B_contact_duration_within_15pct": within(contact, "t_c_rel_err"),
        "B_peak_overlap_within_15pct": within(contact, "delta_max_rel_err"),
        "C_stokes_tautology_within_15pct": within(st, "rel_err_vs_reference"),
        "D_schiller_naumann_vs_literature_within_15pct": within(sn, "rel_err_vs_reference"),
        "D_worst_rel_err": max(abs(r["rel_err_vs_reference"]) for r in sn),
        "F_ledger_residual_below_1e-10": ledger["relative_residual"] < 1.0e-10,
        "E_lbm_fluid_leg_within_15pct_best": lbm.get("n_within_15pct_best"),
        "E_lbm_worst_rel_err": lbm["worst_row"]["rel_err"] if lbm["available"] and lbm["rows"] else None,
        "lbm_leg_gating": False,
        "note": (
            "Arm C is a tautology check (H3): the solver and the reference share the Stokes "
            "closure, so it proves the force->acceleration->velocity plumbing only.  The "
            "acceptance verdict is carried by arms A/B (closed-form contact) and D "
            "(literature drag law).  Arm E is reported, not gating."
        ),
    }

    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["verdict"], indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())