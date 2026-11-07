#!/usr/bin/env python
"""KCS self-propulsion campaign driver (body-force propeller + hull wake).

Subcommands
-----------
``tunnel``   Actuator-disk momentum-theory verification in an LBM duct.
``hull``     KCS double-body LBM run: propeller-plane wake field, hull pressure
             drag (propeller off) and, optionally, the actuator-disk-induced
             resistance increase (propeller on).
``analyze``  Assemble the self-propulsion point, the efficiency chain, the
             figures and the machine-readable evidence file from the LBM
             artefacts.

All numbers written by ``analyze`` are computed programmatically (no hand
copied values); the reference values (SIMMAN 2010 targets) live in
:class:`tensorlbm.selfpropulsion.KCSModelCase`.

Usage
-----
    python scripts/selfpropulsion_kcs.py tunnel  --out artifacts/selfpropulsion_kcs/tunnel.json
    python scripts/selfpropulsion_kcs.py hull    --out artifacts/selfpropulsion_kcs/hull.json
    python scripts/selfpropulsion_kcs.py analyze --out artifacts/selfpropulsion_kcs
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from tensorlbm.d3q19 import OPPOSITE, C, equilibrium3d, macroscopic3d
from tensorlbm.selfpropulsion import (
    KCS_MODEL,
    KP505Geometry,
    SectionPolar,
    blade_element_momentum,
    build_actuator_disk_field,
    empirical_thrust_deduction,
    empirical_wake_fraction,
    holtrop_relative_rotative_efficiency,
    hull_pressure_resistance,
    ittc78_resistance,
    kp505_reference_table,
    open_water_series,
    plane_wake_stats,
    power_chain,
    relative_rotative_efficiency,
    run_actuator_disk_momentum_case,
    solve_self_propulsion,
    solve_self_propulsion_curve,
)
from tensorlbm.ship_cad import hull_block_coefficient, kcs_hull_mask
from tensorlbm.solver3d import collide_bgk3d, stream3d

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _mirror_index_map() -> dict[int, int]:
    """Map every D3Q19 direction to its z-mirrored partner (slip wall)."""
    lut = {}
    c = np.asarray(C.cpu().numpy() if hasattr(C, "cpu") else C, dtype=int)
    for i in range(c.shape[0]):
        target = (c[i, 0], c[i, 1], -c[i, 2])
        for j in range(c.shape[0]):
            if (c[j, 0], c[j, 1], c[j, 2]) == target:
                lut[i] = j
                break
    return lut


def static_bounce_back(f: torch.Tensor, solid: torch.Tensor) -> torch.Tensor:
    """Full-way bounce-back on solid cells (static walls, no-slip)."""
    out = f.clone()
    opp = OPPOSITE.to(dtype=torch.long, device=f.device)
    for i in range(int(f.shape[0])):
        out[i][solid] = f[int(opp[i].item())][solid]
    return out


def apply_slip_top(f: torch.Tensor, lut: dict[int, int]) -> torch.Tensor:
    """Specular (free-slip) reflection on the last z plane (waterline)."""
    f_new = f.clone()
    for i, j in lut.items():
        if int(C[i, 2]) < 0:                      # entering from above
            f_new[i, -1, :, :] = f[j, -1, :, :]
    return f_new


def run_hull_case(
    *,
    length_cells: int = 132,
    nx: int = 200,
    ny: int = 56,
    nz: int = 20,
    re_length: float = 4000.0,
    u_in: float = 0.05,
    n_steps: int = 6000,
    warmup_steps: int = 4000,
    sample_steps: int = 1000,
    smagorinsky_cs: float = 0.0,
    propeller_on: bool = False,
    propeller_thrust_lattice: float = 0.0,
    propeller_torque_lattice: float = 0.0,
    propeller_x_over_l: float = 0.9825,
    device: str = "cpu",
) -> dict:
    """KCS double-body LBM run (waterline = slip symmetry plane, no free surface).

    Returns the hull-pressure drag, the propeller-plane wake statistics and the
    axial-velocity slice used for the wake figure.
    """
    dev = torch.device(device)
    case = KCS_MODEL
    beam_over_l = case.beam / case.length
    draft_over_l = case.draft / case.length
    beam_cells = beam_over_l * length_cells
    draft_cells = max(3, int(round(draft_over_l * length_cells)))
    cz_keel = nz - 1 - draft_cells          # waterline = top plane
    cx = 0.5 * nx - 0.45 * length_cells     # upstream clearance ~0.2 L
    cy = 0.5 * (ny - 1)

    mask = kcs_hull_mask(nx, ny, nz, cx, cy, cz_keel, length_cells, beam_cells,
                         draft_cells, dev)
    cb_mask = hull_block_coefficient(mask, beam_cells, draft_cells, length_cells)

    walls = torch.zeros_like(mask)
    walls[:, 0, :] = True          # port/starboard tank walls (no-slip)
    walls[:, -1, :] = True
    walls[0, :, :] = True          # tank bottom (no-slip far below the keel)
    solid = mask | walls

    nu_lat = u_in * length_cells / re_length
    tau = 3.0 * nu_lat + 0.5
    if tau <= 0.5:
        raise ValueError(f"tau must exceed 0.5 (got {tau})")

    rho0 = torch.ones((nz, ny, nx), dtype=torch.float32, device=dev)
    ux0 = torch.full_like(rho0, u_in)
    zeros = torch.zeros_like(rho0)
    f = equilibrium3d(rho0, ux0, zeros, zeros, device=dev)
    f_inlet = equilibrium3d(torch.ones((nz, ny, 1), dtype=torch.float32, device=dev),
                            torch.full((nz, ny, 1), u_in, dtype=torch.float32, device=dev),
                            torch.zeros((nz, ny, 1), dtype=torch.float32, device=dev),
                            torch.zeros((nz, ny, 1), dtype=torch.float32, device=dev),
                            device=dev)

    disk = None
    if propeller_on:
        r_disk = 0.5 * case.propeller.diameter / case.length * length_cells
        hub_r = r_disk * case.propeller.hub_ratio
        x_disk = cx + (case.propeller_x_over_l - 0.5) * length_cells
        z_disk = cz_keel + (case.draft + case.propeller_shaft_z) / case.length * length_cells
        rr = np.linspace(case.propeller.hub_ratio, 1.0, 40)
        load = np.clip(1.0 - (rr - 0.75) ** 2 / 0.75**2, 0.05, None)
        disk = build_actuator_disk_field(
            nz, ny, nx, center=(x_disk, cy, z_disk), radius_cells=r_disk,
            hub_radius_cells=hub_r, r_over_R=rr, dT_dr=load, dQ_dr=load,
            thrust_lattice=propeller_thrust_lattice,
            torque_lattice=propeller_torque_lattice, thickness_cells=2, device=dev)
        disk["x_plane"] = int(round(x_disk))
        disk["z_plane"] = float(z_disk)
        disk["r_cells"] = float(r_disk)
        dy = torch.arange(ny, device=dev, dtype=torch.float32)[None, :, None] - cy
        dz = torch.arange(nz, device=dev, dtype=torch.float32)[:, None, None] - z_disk
        r_safe = disk["radius"].clamp(min=1e-6)
        disk["fy"] = torch.where(disk["mask"], -disk["ftheta"] * dz / r_safe,
                                 torch.zeros_like(disk["fx"]))
        disk["fz"] = torch.where(disk["mask"], disk["ftheta"] * dy / r_safe,
                                 torch.zeros_like(disk["fx"]))

    lut = _mirror_index_map()
    ux_acc = torch.zeros((nz, ny, nx), dtype=torch.float64, device=dev)
    rho_acc = torch.zeros((nz, ny, nx), dtype=torch.float64, device=dev)
    n_acc = 0
    fx_hist: list[float] = []

    for step in range(1, n_steps + 1):
        f = collide_bgk3d(f, tau=tau)
        if disk is not None:
            from tensorlbm.selfpropulsion import apply_body_force_guo
            f = apply_body_force_guo(f, disk["fx"], disk["fy"], disk["fz"], inplace=True)
        f = stream3d(f)
        f = static_bounce_back(f, solid)
        f = apply_slip_top(f, lut)
        f[:, :, :, 0] = f_inlet[:, :, :, 0]
        f[:, :, :, -1] = f[:, :, :, -2]
        if step > warmup_steps and step <= warmup_steps + sample_steps:
            rho_t, ux_t, _, _ = macroscopic3d(f)
            ux_acc += ux_t.to(torch.float64)
            rho_acc += rho_t.to(torch.float64)
            n_acc += 1
        if step % 200 == 0 or step == n_steps:
            rho_h, ux_h, uy_h, uz_h = macroscopic3d(f)
            fx_hist.append({"step": step, "max_abs_u": float(ux_h.abs().max().item()),
                            "mean_rho": float(rho_h.mean().item())})

    ux_mean = (ux_acc / max(n_acc, 1)).to(torch.float32)
    rho_mean = (rho_acc / max(n_acc, 1)).to(torch.float32)
    rho_f, ux_f, uy_f, uz_f = macroscopic3d(f)

    drag = hull_pressure_resistance(mask, rho_mean)
    stern_lo = int(cx + 0.35 * length_cells)
    stern = hull_pressure_resistance(mask, rho_mean, x_min=stern_lo)

    x_plane = disk["x_plane"] if disk is not None else int(cx + 0.5 * length_cells) + 2
    z_plane = disk["z_plane"] if disk is not None else float(cz_keel + 0.45 * draft_cells)
    r_plane = disk["r_cells"] if disk is not None else 0.5 * case.propeller.diameter / \
        case.length * length_cells
    # the probe plane must intersect water: walk aft if the annulus is solid
    zc = torch.arange(nz, device=dev, dtype=torch.float32)[:, None]
    yc = torch.arange(ny, device=dev, dtype=torch.float32)[None, :]
    ann_mask = ((yc - cy) ** 2 + (zc - z_plane) ** 2 <= r_plane**2)
    best = (-1, x_plane)
    for x_try in range(x_plane, min(nx - 2, x_plane + 12)):
        cover = int((ann_mask & (~mask[:, :, x_try])).sum().item())
        if cover > best[0]:
            best = (cover, x_try)
        if cover > 0:
            break
    x_plane = best[1]
    fluid_plane = ~mask[:, :, x_plane]
    wake = plane_wake_stats(
        ux_mean[:, :, x_plane], center=(x_plane, cy, z_plane),
        radius_cells=r_plane, hub_radius_cells=0.18 * r_plane,
        freestream=u_in, fluid=fluid_plane, n_radial=8, n_sectors=12)

    mid_z = int(round(cz_keel + 0.5 * draft_cells))
    slice_ux = ux_mean[mid_z, :, :].cpu().numpy().tolist()
    slice_solid = mask[mid_z, :, :].cpu().numpy().astype(int).tolist()

    return {
        "config": {
            "length_cells": length_cells, "nx": nx, "ny": ny, "nz": nz,
            "re_length": re_length, "u_in": u_in, "tau": tau,
            "nu_lattice": nu_lat, "smagorinsky_cs": smagorinsky_cs,
            "n_steps": n_steps, "warmup_steps": warmup_steps,
            "sample_steps": sample_steps, "device": str(device),
            "propeller_on": propeller_on,
            "propeller_thrust_lattice": propeller_thrust_lattice,
            "draft_cells": draft_cells, "beam_cells": beam_cells,
            "cz_keel": cz_keel, "cx": cx, "cy": cy,
        },
        "geometry_validation": {
            "cb_mask": cb_mask, "cb_published": case.cb,
            "cb_rel_error": abs(cb_mask - case.cb) / case.cb,
            "l_over_b_mask": length_cells / beam_cells,
            "l_over_b_published": case.length / case.beam,
            "draft_cells": draft_cells, "beam_cells": beam_cells,
        },
        "drag": {"total_pressure_drag_lattice": drag["pressure_resistance"],
                 "stern_pressure_drag_lattice": stern["pressure_resistance"],
                 "n_faces_plus": drag["n_faces_plus"],
                 "n_faces_minus": drag["n_faces_minus"],
                 "profile": drag["profile"],
                 "obstacle_force_x_history": fx_hist},
        "wake": {"x_plane": x_plane, "z_plane": z_plane, "r_cells": r_plane,
                 "w": wake["w"], "u_mean_over_U": wake["u_mean_over_U"],
                 "n_cells": wake["n_cells"],
                 "radial_profile": wake["radial_profile"],
                 "sector_profile": wake["sector_profile"]},
        "slice": {"z_index": mid_z, "ux": slice_ux, "solid": slice_solid},
        "wall_clock_s": None,
    }


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


def analyze(
    *,
    tunnel_path: Path,
    hull_path: Path | None,
    outdir: Path,
) -> dict:
    """Build the self-propulsion point, the efficiency chain and the evidence file."""
    case = KCS_MODEL
    geom: KP505Geometry = case.propeller
    polar = SectionPolar()
    tunnel = (json.loads(Path(tunnel_path).read_text())
              if tunnel_path is not None and Path(tunnel_path).exists() else {})
    hull = json.loads(Path(hull_path).read_text()) if hull_path and Path(hull_path).exists() else None

    # --- open water: BEM vs measured KP505 ---------------------------------
    ref = kp505_reference_table()
    bem = open_water_series(geom, polar, ref["J"])
    ow_rows = []
    for j, kt_ref, kq_ref, b in zip(ref["J"], ref["KT"], ref["KQ"], bem):
        ow_rows.append({
            "J": j,
            "KT_ref": kt_ref, "KT_bem": b["kt"],
            "KT_rel_error": (b["kt"] - kt_ref) / kt_ref,
            "KQ_ref": kq_ref, "KQ_bem": b["kq"],
            "KQ_rel_error": (b["kq"] - kq_ref) / kq_ref,
            "eta0_ref": j * kt_ref / (2 * math.pi * kq_ref),
            "eta0_bem": b["eta0"],
        })
    design = [row for row in ow_rows if abs(row["J"] - 0.7) < 1e-6][0]
    kt_max_err = float(np.max(np.abs([r["KT_rel_error"] for r in ow_rows])))
    kq_max_err = float(np.max(np.abs([r["KQ_rel_error"] for r in ow_rows])))

    # section-polar uncertainty band on the open-water solution
    band = []
    for cl_alpha in (5.0, 5.7, 6.2):
        for alpha0 in (-4.5, -3.5, -2.5):
            for cd0 in (0.0060, 0.0085, 0.0120):
                pol = SectionPolar(cl_alpha=cl_alpha, alpha_0_deg=alpha0, cd0=cd0)
                sol = blade_element_momentum(geom, pol, 0.7)
                band.append({"cl_alpha": cl_alpha, "alpha_0_deg": alpha0, "cd0": cd0,
                             "KT": sol["kt"], "10KQ": 10 * sol["kq"], "eta0": sol["eta0"]})
    kt_band = (float(min(b["KT"] for b in band)), float(max(b["KT"] for b in band)))

    # --- resistance (model scale, ITTC-78) ---------------------------------
    res = ittc78_resistance(
        length=case.length, beam=case.beam, draft=case.draft, cb=case.cb,
        speed=case.speed, rho=case.rho, nu=case.nu, wetted_area=case.wetted_area,
        form_factor=1.069, cm=case.cm, cwp=case.cwp, residual_coefficient=0.58e-3)
    rt = res["R_T"]

    # --- wake fraction and thrust deduction --------------------------------
    w_design = empirical_wake_fraction(case.cb)
    t_design = empirical_thrust_deduction(w_design)
    cp = case.cb / case.cm
    eta_r_design = holtrop_relative_rotative_efficiency(
        prismatic_coefficient=cp, lcb_percent_l=-1.5,
        pitch_ratio=geom.design_pitch_ratio)
    sp = solve_self_propulsion(
        resistance=rt, speed=case.speed, geometry=geom, polar=polar,
        wake_fraction=w_design, thrust_deduction=t_design, rho=case.rho)
    chain = power_chain(
        resistance=rt, speed=case.speed, thrust=sp["thrust"], torque=sp["torque"],
        n_rps=sp["n_rps"], wake_fraction=w_design, thrust_deduction=t_design,
        eta_R=eta_r_design, eta0=sp["eta0"])

    # --- self-propulsion point from the measured open-water curve ----------
    sp_curve = solve_self_propulsion_curve(
        resistance=rt, speed=case.speed, diameter=geom.diameter, rho=case.rho,
        wake_fraction=w_design, thrust_deduction=t_design,
        j_table=ref["J"], kt_table=ref["KT"], kq_table=ref["KQ"])
    chain_curve = power_chain(
        resistance=rt, speed=case.speed, thrust=sp_curve["thrust"],
        torque=sp_curve["torque"], n_rps=sp_curve["n_rps"],
        wake_fraction=w_design, thrust_deduction=t_design,
        eta_R=eta_r_design, eta0=sp_curve["eta0"])

    # --- eta_R from the measured (coarse-grid) wake field -------------------
    eta_r_block = None
    if hull is not None:
        wake_field = {"w": hull["wake"]["w"],
                      "radial_profile": hull["wake"]["radial_profile"],
                      "sector_profile": hull["wake"]["sector_profile"]}
        eta_r_block = relative_rotative_efficiency(
            geom, polar, wake_field=wake_field, n_rps=sp["n_rps"],
            speed=case.speed, rho=case.rho, n_sections=30)

    # --- note on the LBM tunnel case ---------------------------------------
    # The tunnel run is a *verification* case: its disk loading is chosen to make
    # the induced velocity measurable and is therefore not the KCS operating
    # point.  No induced-velocity correction is transferred to the KCS wake
    # fraction (that would require a model-scale hull-wake simulation, see the
    # documented limitation).
    w_induction_shift = None

    criteria = {
        "w_vs_simman": {
            "computed": w_design,
            "reference": case.reference_wake_fraction,
            "rel_error": abs(w_design - case.reference_wake_fraction) / case.reference_wake_fraction,
            "threshold": 0.10,
            "pass": abs(w_design - case.reference_wake_fraction) / case.reference_wake_fraction < 0.10,
        },
        "t_vs_simman": {
            "computed": t_design,
            "reference": case.reference_thrust_deduction,
            "rel_error": abs(t_design - case.reference_thrust_deduction) / case.reference_thrust_deduction,
            "threshold": 0.15,
            "pass": abs(t_design - case.reference_thrust_deduction) / case.reference_thrust_deduction < 0.15,
        },
        "eta_H_consistency": {
            "from_w_t": chain["eta_H"],
            "from_power_chain": chain["eta_H_check"],
            "rel_error": abs(chain["eta_H"] - chain["eta_H_check"]) / chain["eta_H"],
            "simman_implied": (1.0 - case.reference_thrust_deduction) /
                              (1.0 - case.reference_wake_fraction),
            "rel_error_vs_simman_implied": abs(
                chain["eta_H"] - (1.0 - case.reference_thrust_deduction) /
                (1.0 - case.reference_wake_fraction)) /
                ((1.0 - case.reference_thrust_deduction) / (1.0 - case.reference_wake_fraction)),
            "threshold": 0.05,
            "pass": abs(chain["eta_H"] - chain["eta_H_check"]) / chain["eta_H"] < 0.05,
        },
    }

    result = {
        "case": {"length": case.length, "beam": case.beam, "draft": case.draft,
                 "cb": case.cb, "speed": case.speed, "fr": case.fr,
                 "rho": case.rho, "nu": case.nu, "wetted_area": case.wetted_area,
                 "propeller": {"D": geom.diameter, "Z": geom.n_blades,
                               "EAR": geom.area_ratio(), "EAR_raw": geom.raw_area_ratio(),
                               "hub_ratio": geom.hub_ratio}},
        "open_water": {"rows": ow_rows, "design_row": design,
                       "max_kt_rel_error": kt_max_err, "max_kq_rel_error": kq_max_err,
                       "uncertainty_band": band, "kt_band_J07": kt_band},
        "resistance": res,
        "self_propulsion": sp,
        "power_chain": chain,
        "self_propulsion_curve": sp_curve,
        "power_chain_curve": chain_curve,
        "wake": {"w_design": w_design, "t_design": t_design,
                 "wake_induction_shift": w_induction_shift,
                 "eta_R_holtrop": eta_r_design, "cp": cp},
        "eta_R": eta_r_block,
        "criteria": criteria,
        "tunnel_verification": tunnel.get("checks", {}),
        "tunnel_measured": tunnel.get("measured", {}),
        "hull_run": None if hull is None else {
            "config": hull["config"],
            "geometry_validation": hull["geometry_validation"],
            "wake": {k: v for k, v in hull["wake"].items() if k != "radial_profile"},
            "radial_profile": hull["wake"]["radial_profile"],
            "sector_profile": hull["wake"]["sector_profile"],
            "drag": {k: v for k, v in hull["drag"].items() if k != "profile"},
        },
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "selfpropulsion_kcs.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("tunnel")
    t.add_argument("--out", required=True)
    t.add_argument("--nz", type=int, default=32)
    t.add_argument("--ny", type=int, default=40)
    t.add_argument("--nx", type=int, default=104)
    t.add_argument("--radius", type=float, default=8.0)
    t.add_argument("--thrust", type=float, default=0.15)
    t.add_argument("--torque", type=float, default=0.02)
    t.add_argument("--steps", type=int, default=5000)
    t.add_argument("--device", default="cpu")

    h = sub.add_parser("hull")
    h.add_argument("--out", required=True)
    h.add_argument("--length-cells", type=int, default=132)
    h.add_argument("--nx", type=int, default=200)
    h.add_argument("--ny", type=int, default=56)
    h.add_argument("--nz", type=int, default=20)
    h.add_argument("--re", type=float, default=4000.0)
    h.add_argument("--steps", type=int, default=6000)
    h.add_argument("--warmup", type=int, default=4000)
    h.add_argument("--sample", type=int, default=1000)
    h.add_argument("--propeller-on", action="store_true")
    h.add_argument("--thrust-lattice", type=float, default=0.0)
    h.add_argument("--torque-lattice", type=float, default=0.0)
    h.add_argument("--device", default="cpu")

    a = sub.add_parser("analyze")
    a.add_argument("--out", required=True)
    a.add_argument("--tunnel", default="artifacts/selfpropulsion_kcs/tunnel.json")
    a.add_argument("--hull", default="artifacts/selfpropulsion_kcs/hull_potential.json")

    args = ap.parse_args()
    torch.set_num_threads(int(torch.get_num_threads()))
    if args.cmd == "tunnel":
        t0 = time.time()
        out = run_actuator_disk_momentum_case(
            nz=args.nz, ny=args.ny, nx=args.nx, radius_cells=args.radius,
            thrust_lattice=args.thrust, torque_lattice=args.torque,
            n_steps=args.steps, warmup_steps=int(args.steps * 0.6),
            sample_steps=int(args.steps * 0.2), device=args.device)
        out["wall_clock_s"] = time.time() - t0
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(json.dumps(out["checks"], indent=2))
    elif args.cmd == "hull":
        t0 = time.time()
        out = run_hull_case(
            length_cells=args.length_cells, nx=args.nx, ny=args.ny, nz=args.nz,
            re_length=args.re, n_steps=args.steps, warmup_steps=args.warmup,
            sample_steps=args.sample, propeller_on=args.propeller_on,
            propeller_thrust_lattice=args.thrust_lattice,
            propeller_torque_lattice=args.torque_lattice, device=args.device)
        out["wall_clock_s"] = time.time() - t0
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out))
        print(json.dumps({"geometry": out["geometry_validation"],
                          "wake": {"w": out["wake"]["w"]},
                          "drag": out["drag"]["total_pressure_drag_lattice"],
                          "wall_clock_s": out["wall_clock_s"]}, indent=2))
    else:
        result = analyze(tunnel_path=Path(args.tunnel),
                         hull_path=Path(args.hull), outdir=Path(args.out))
        print(json.dumps(result["criteria"], indent=2))


if __name__ == "__main__":
    main()