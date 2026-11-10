"""P262W1-N02 — three-phase (ice / water / air) contact-line campaign.

What this campaign measures (nothing is typed in by hand: every number below is
produced by a lattice-Boltzmann run inside ``main`` and written to
``results.json`` / ``criteria.json``):

A. ``gamma_lg`` — water/air tension from the **Laplace law** of a free droplet in
   a periodic box.  The pressure difference uses the Shan-Chen equation of state
   validated in this node against coexistence (0.3 %), so no constant is copied
   from a table.
B. ``gamma_sg - gamma_sl`` — wetting tension by **thermodynamic integration** in
   the wall coupling ``psi_w`` (:func:`tensorlbm.contact_line.three_phase.wetting_tension_ti`),
   the only route that survives the rigid-wall check (see the falsified
   pressure-tensor hypothesis in the node report).
C. ``theta_meas`` — equilibrium contact angle of a sessile droplet (circle fit of
   the level set, ``theta_fit.angle_estimate``), recorded as a relaxation
   trajectory so that equilibration is visible, not assumed.
D. Cox-Voinov law — the same droplet driven by a body force; the steady slide
   speed gives ``Ca = mu U_CL / gamma_lg`` and the advancing angle gives
   ``theta_d``; the fit reuses :func:`tensorlbm.contact_line.scaling.cox_voinov_fit`.
E. Grid convergence of ``U_CL`` on three grids (order from a Richardson estimate).
F. Mass conservation of every production run.

Machine-checkable criteria (F1-F6 of the node):
F1  ``|theta_meas - theta_Young| <= 3 deg`` per wall fraction (theta_Young from B).
F2  Cox-Voinov fit ``R^2 >= 0.95``.
F3  contact-line velocity convergence order ``>= 0.5`` (three grids).
F4  mass drift ``<= 0.1 %`` for every run.
F5  ``pytest tests/test_contact_line_three_phase.py`` (run separately; the count is
    reported in the node report).
F6  artifacts ``contact_angle_relaxation.png``, ``cox_voinov_fit.png``,
    ``mass_conservation.json`` present in the output directory.

Usage
-----
    python benchmarks/three_phase_contact_line/run.py --out runs/p262w1_n02
    python benchmarks/three_phase_contact_line/run.py --out ... --jobs 8
    python benchmarks/three_phase_contact_line/run.py --out ... --quick   # smoke only
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

from tensorlbm.contact_line import three_phase as tp  # noqa: E402
from tensorlbm.contact_line.engine import (  # noqa: E402
    Channel,
    WallStepper,
    default_coexistence,
    sessile_drop_state,
)
from tensorlbm.contact_line.scaling import ScalingPoint, cox_voinov_fit  # noqa: E402
from tensorlbm.d2q9 import macroscopic  # noqa: E402

RHO_L, RHO_V, SIGMA_DOC = default_coexistence()
LEVEL = 0.5 * (RHO_L + RHO_V)
TAU = 1.0
G_SC = 5.0
#: Dynamic viscosity of the lattice liquid, ``mu = rho (tau - 1/2) c_s^2``.
MU = RHO_L * (TAU - 0.5) * (1.0 / 3.0)

#: Wall wettabilities of the static ladder ``psi_w / psi(rho_l)``.
STATIC_FRACTIONS = (0.50, 0.60, 0.70, 0.80)
#: Wall used for the driven (Cox-Voinov) and convergence sweeps.
DRIVE_FRACTION = 0.60
#: Streamwise body-force accelerations of the Cox-Voinov sweep.
DRIVE_GX = (4.0e-5, 8.0e-5, 1.6e-4, 3.0e-4)
#: Grids of the convergence study (ny, nx).
CONVERGENCE_GRIDS = ((48, 96), (64, 128), (80, 160))


# ---------------------------------------------------------------------------
# workers (fork-ed; each must set its own thread count, see IN100-N02)
# ---------------------------------------------------------------------------


def _init_threads(n: int = 1) -> None:
    torch.set_num_threads(max(1, int(n)))


def _centroid_x(rho: torch.Tensor, solid: torch.Tensor) -> float:
    """Streamwise centre of mass of the fluid (used to split advancing/receding)."""
    w = torch.where(solid, torch.zeros_like(rho), rho)
    xs = torch.arange(rho.shape[1], dtype=rho.dtype).view(1, -1)
    return float((w * xs).sum() / w.sum())


def _relax_sessile(
    fraction: float,
    *,
    ny: int,
    nx: int,
    steps: int,
    gx: float = 0.0,
    drive: str = "uniform",
    track_from: int | None = None,
    keep_field: bool = False,
    init_deg: float = 110.0,
) -> dict:
    """Relax (and optionally drive) a sessile droplet; return angles + diagnostics."""
    _init_threads(1)
    channel = Channel(nx=nx, ny=ny, wall_rows=2)
    solid = channel.solid_mask()
    radius = 0.32 * ny
    # a circular cap of angle ``init_deg`` starts closer to equilibrium than the
    # tangential circle (180 deg), which matters because spreading is slow
    height = -radius * math.cos(math.radians(init_deg))
    f, _ = sessile_drop_state(channel, radius=radius, height=height)
    stepper = WallStepper(
        channel,
        G=G_SC,
        tau=TAU,
        rho_l=RHO_L,
        rho_v=RHO_V,
        wall_fraction=fraction,
        forcing="edm",
        drive=drive,
    )
    rho_init = macroscopic(f)[0]
    m0 = float(rho_init[~solid].sum())
    m_tot0 = float(rho_init.sum())
    track_from = int(0.4 * steps) if track_from is None else int(track_from)
    traj: list[list[float]] = []
    front: list[list[float]] = []
    rear: list[list[float]] = []
    x_prev: dict[str, float] = {}
    centroid_track: list[list[float]] = []
    cx_unwrapped = _centroid_x(rho_init, solid)
    cx_prev = cx_unwrapped
    unwrapped: dict[str, float] = {"front": 0.0, "rear": 0.0}
    rho = macroscopic(f)[0]
    drift_max = 0.0
    for n in range(1, steps + 1):
        f = stepper.step(f, gx)
        rho = macroscopic(f)[0]
        m = float(rho.sum())
        drift_max = max(drift_max, abs(m - m_tot0) / m_tot0)
        if n % 200 == 0 and (n >= track_from or n % 2000 == 0):
            try:
                est = tp.sessile_contact_angle(
                    rho, y_wall=channel.y_wall(), solid=solid, level=LEVEL, side="cap"
                )
                traj.append([n, float(est["theta_deg"]), float(est["unc_deg"])])
            except ValueError:
                pass
        if n % 100 == 0:
            cx = _centroid_x(rho, solid)
            cx_unwrapped += (cx - cx_prev + 0.5 * nx) % nx - 0.5 * nx
            cx_prev = cx
            if n >= track_from:
                centroid_track.append([float(n), float(cx_unwrapped)])
        if gx != 0.0 and n >= track_from and n % 100 == 0:
            centre = _centroid_x(rho, solid)
            for side in ("front", "rear"):
                x = tp.contact_line_position(
                    rho, level=LEVEL, y_wall=channel.y_wall(), side=side, centre_x=centre
                )
                if not math.isfinite(x):
                    continue
                if side in x_prev:
                    d = (x - x_prev[side] + 0.5 * nx) % nx - 0.5 * nx
                    unwrapped[side] += d
                x_prev[side] = x
                (front if side == "front" else rear).append(
                    [float(n), float(unwrapped[side] + x)]
                )
    rho = macroscopic(f)[0]
    m1 = float(rho.sum())
    m_fluid1 = float(rho[~solid].sum())
    ux, uy = stepper.velocity(f)
    speed2d = torch.sqrt(ux**2 + uy**2)
    centre = _centroid_x(rho, solid)
    liq_mask = (rho > 1.5) & (~solid)
    out_sp = {
        "spurious_max_liquid": float(speed2d[liq_mask].max()) if bool(liq_mask.any()) else float("nan"),
        "spurious_mean_liquid": float(speed2d[liq_mask].mean()) if bool(liq_mask.any()) else float("nan"),
    }
    finite = bool(torch.isfinite(rho).all())
    # the wall rows carry the initial liquid density and must not count as liquid
    liquid = (rho > 1.5) & (~solid)
    footprint = 0
    height = 0
    if bool(liquid.any()):
        cols = torch.nonzero(liquid.any(dim=0), as_tuple=False).flatten()
        rows = torch.nonzero(liquid.any(dim=1), as_tuple=False).flatten()
        footprint = int(cols.max() - cols.min() + 1)
        height = int(rows.max() - rows.min() + 1)
    spans = bool(footprint >= 0.85 * nx)
    out: dict = {
        "finite": finite,
        "footprint": footprint,
        "height": height,
        "spans_domain": spans,
        "wall_fraction": float(fraction),
        "ny": int(ny),
        "nx": int(nx),
        "steps": int(steps),
        "gx": float(gx),
        "drive": drive,
        "mass_drift": abs(m1 - m_tot0) / m_tot0,
        "mass_drift_max": drift_max,
        "mass_drift_fluid_only": abs(m_fluid1 - m0) / m0,
        "theta_traj": traj,
        "umax": float(speed2d[~solid].max()),
        "umean": float(speed2d[~solid].mean()),
        **out_sp,
        "centre_x": centre,
        "front_track": front,
        "rear_track": rear,
        "theta_final": traj[-1][1] if traj else float("nan"),
        "theta_final_unc": traj[-1][2] if traj else float("nan"),
    }
    cv = tp.contact_line_velocity([(t, x) for t, x in centroid_track])
    out["velocity_centroid"] = cv["velocity"]
    out["velocity_centroid_stderr"] = cv["stderr"]
    out["velocity_centroid_r2"] = cv["r2"]
    for side, track in (("front", front), ("rear", rear)):
        if len(track) >= 5:
            v = tp.contact_line_velocity([(s, x) for s, x in track])
            out[f"velocity_{side}"] = v["velocity"]
            out[f"velocity_{side}_stderr"] = v["stderr"]
            out[f"velocity_{side}_r2"] = v["r2"]
        else:
            out[f"velocity_{side}"] = float("nan")
            out[f"velocity_{side}_stderr"] = float("nan")
            out[f"velocity_{side}_r2"] = float("nan")
    if gx != 0.0 and len(front) >= 5:
        try:
            est = tp.sessile_contact_angle(
                rho, y_wall=channel.y_wall(), solid=solid, level=LEVEL, side="front",
                centre_x=centre,
            )
            out["theta_adv"] = float(est["theta_deg"])
            out["theta_adv_unc"] = float(est["unc_deg"])
        except ValueError:
            out["theta_adv"] = float("nan")
            out["theta_adv_unc"] = float("nan")
        try:
            est = tp.sessile_contact_angle(
                rho, y_wall=channel.y_wall(), solid=solid, level=LEVEL, side="rear",
                centre_x=centre,
            )
            out["theta_rec"] = float(est["theta_deg"])
            out["theta_rec_unc"] = float(est["unc_deg"])
        except ValueError:
            out["theta_rec"] = float("nan")
            out["theta_rec_unc"] = float("nan")
    if keep_field:
        out["rho_field"] = rho.to(torch.float32).numpy().tolist()
    return out


def _wetting(fraction: float, ny: int, ladder: int, steps: int) -> dict:
    """Wetting tension by thermodynamic integration + the falsified tensor route.

    Both routes are reported: the pressure-tensor integral is kept as the *negative*
    evidence (it measures wall pressure, not wall tension) so the report can cite it.
    """
    _init_threads(1)
    out = tp.wetting_tension_ti(
        fraction, ny=ny, n_ladder=ladder, n_steps=steps, G=G_SC, tau=TAU
    ).to_dict()
    gamma_sl_pt, gamma_sg_pt = tp.wall_tension_pair(
        fraction, ny=ny, n_steps=max(600, steps // 2), n_resolutions=1
    )
    out["gamma_sl_pt"] = gamma_sl_pt.gamma
    out["gamma_sg_pt"] = gamma_sg_pt.gamma
    out["delta_gamma_pressure_tensor"] = gamma_sg_pt.gamma - gamma_sl_pt.gamma
    return out


def _droplet(n: int, radius: float, steps: int) -> dict:
    _init_threads(1)
    out = tp.free_droplet(n=n, radius=radius, n_steps=steps, G=G_SC, tau=TAU)
    out.pop("rho_field", None)
    return out


# ---------------------------------------------------------------------------
# campaign driver
# ---------------------------------------------------------------------------


def _pool(jobs: int):
    import multiprocessing as mp

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    return mp.get_context("fork").Pool(max(1, int(jobs)))


def run_campaign(out: Path, *, jobs: int = 8, quick: bool = False) -> dict:
    t_start = time.time()
    out.mkdir(parents=True, exist_ok=True)
    steps_static = 3000 if quick else 16000
    steps_drive = 4000 if quick else 30000
    ladder = 5 if quick else 9
    ti_steps = 800 if quick else 3500
    fractions = STATIC_FRACTIONS if not quick else (0.55, 0.85)
    gx_list = DRIVE_GX if not quick else (5.0e-5, 2.0e-4)

    results: dict = {"parameters": {
        "rho_l": RHO_L, "rho_v": RHO_V, "G": G_SC, "tau": TAU, "mu": MU,
        "level": LEVEL, "quick": quick, "steps_static": steps_static,
        "steps_drive": steps_drive, "ladder": ladder, "ti_steps": ti_steps,
        "static_fractions": list(fractions), "drive_gx": list(gx_list),
        "drive_fraction": DRIVE_FRACTION, "convergence_grids": list(CONVERGENCE_GRIDS),
    }}

    with _pool(jobs) as pool:
        # ---- A: water/air tension (Laplace) -----------------------------------
        n_drop, r_drop, st_drop = (80, 15.0, 3000) if quick else (128, 24.0, 7000)
        drop = pool.apply_async(_droplet, (n_drop, r_drop, st_drop))
        # ---- B: wetting tension ladder (TI) -----------------------------------
        jobs_ti = [
            pool.apply_async(_wetting, (f, 48, ladder, ti_steps)) for f in fractions
        ]
        # ---- C: static contact angles ----------------------------------------
        jobs_static = [
            pool.apply_async(
                _relax_sessile,
                (f,),
                {"ny": 64, "nx": 128, "steps": steps_static, "keep_field": (i == 0)},
            )
            for i, f in enumerate(fractions)
        ]
        static = [j.get() for j in jobs_static]
        wet = [j.get() for j in jobs_ti]
        drop_res = drop.get()
        results["gamma_lg"] = {k: v for k, v in drop_res.items()}
        results["wetting"] = wet
        results["static"] = static

        gamma_lg = float(drop_res["gamma"])
        results["gamma_lg"]["documented_laplace"] = SIGMA_DOC
        results["gamma_lg"]["relative_difference_vs_documented"] = gamma_lg / SIGMA_DOC - 1.0

        # Young angle per wall fraction, and the F1 error against the measured angle
        f1_rows = []
        for frac, w, s in zip(fractions, wet, static):
            theta_y = tp.young_angle_deg(w["gamma_sg"], w["gamma_sl"], gamma_lg)
            theta_m = float(s["theta_final"])
            f1_rows.append({
                "wall_fraction": frac,
                "gamma_sl": w["gamma_sl"],
                "gamma_sg": w["gamma_sg"],
                "delta_gamma": w["delta_gamma"],
                "delta_gamma_pressure_tensor": w.get("delta_gamma_pressure_tensor", float("nan")),
                "cos_young": w["delta_gamma"] / gamma_lg,
                "theta_young_deg": theta_y,
                "theta_measured_deg": theta_m,
                "theta_measured_unc_deg": float(s["theta_final_unc"]),
                "error_deg": abs(theta_m - theta_y) if math.isfinite(theta_y) else float("nan"),
                "mass_drift": float(s["mass_drift"]),
                "footprint": int(s.get("footprint", 0)),
                "height": int(s.get("height", 0)),
                "spans_domain": bool(s.get("spans_domain", False)),
                "angle_valid": bool(math.isfinite(theta_m) and not s.get("spans_domain", False)),
            })
        results["young"] = f1_rows

        # ---- D: Cox-Voinov sweep ---------------------------------------------
        jobs_drive = [
            pool.apply_async(
                _relax_sessile,
                (DRIVE_FRACTION,),
                {"ny": 64, "nx": 128, "steps": steps_drive, "gx": gx, "drive": "liquid",
                 "keep_field": (i == 0)},
            )
            for i, gx in enumerate(gx_list)
        ]
        # ---- E: grid convergence at the middle drive -------------------------
        gx_conv = gx_list[len(gx_list) // 2]
        jobs_conv = [
            pool.apply_async(
                _relax_sessile,
                (DRIVE_FRACTION,),
                {"ny": ny, "nx": nx, "steps": steps_drive, "gx": gx_conv, "drive": "liquid"},
            )
            for (ny, nx) in CONVERGENCE_GRIDS
            if (ny, nx) != (64, 128)
        ]
        drive = [j.get() for j in jobs_drive]
        conv_extra = [j.get() for j in jobs_conv]
    results["drive"] = drive
    results["convergence_extra"] = conv_extra

    # Cox-Voinov fit
    # theta_e of the Cox-Voinov fit: the *measured* equilibrium angle of the
    # driven wall (direct geometry), with the Young prediction as fallback.
    theta_e = next(
        (r["theta_measured_deg"] for r in f1_rows
         if abs(r["wall_fraction"] - DRIVE_FRACTION) < 1e-9 and r.get("angle_valid", True)),
        float("nan"),
    )
    if not math.isfinite(theta_e):
        theta_e = next(
            (r["theta_young_deg"] for r in f1_rows
             if abs(r["wall_fraction"] - DRIVE_FRACTION) < 1e-9),
            float("nan"),
        )
    points: list[ScalingPoint] = []
    for d in drive:
        if not d.get("finite", True):
            continue
        u = float(d.get("velocity_centroid", float("nan")))
        if not math.isfinite(u) or u <= 0.0:
            u = 0.5 * (abs(d.get("velocity_front", float("nan"))) + abs(d.get("velocity_rear", float("nan"))))
        if not math.isfinite(u) or u <= 0.0:
            continue
        ca = MU * u / gamma_lg
        radius = 0.32 * d["ny"]
        points.append(
            ScalingPoint(
                gx=float(d["gx"]),
                capillary=float(ca),
                velocity=float(u),
                velocity_stderr=float(
                    0.5 * (d.get("velocity_front_stderr", 0.0) + d.get("velocity_rear_stderr", 0.0))
                ),
                theta_a_deg=float(d.get("theta_adv", float("nan"))),
                theta_a_unc_deg=float(d.get("theta_adv_unc", float("nan"))),
                theta_r_deg=float(d.get("theta_rec", float("nan"))),
                theta_r_unc_deg=float(d.get("theta_rec_unc", float("nan"))),
                theta_e_deg=float(theta_e),
                theta_e_unc_deg=float("nan"),
                wall_fraction=DRIVE_FRACTION,
                extra={"radius": radius, "centre_x": d.get("centre_x")},
            )
        )
    fit = cox_voinov_fit(points, macro_length=0.32 * 64)
    results["cox_voinov"] = {
        "theta_e_deg": theta_e,
        "macro_length": 0.32 * 64,
        "fit": fit.to_dict(),
        "points": [p.to_dict() for p in points],
        "resolvability_note": (
            "all points with a measurable slide speed are admitted; the "
            "resolvability gate of scaling.py is applied by the report writer"
        ),
    }

    # Convergence order from the three grids (Richardson on U_CL)
    conv_runs = {}
    for d in drive:
        if abs(d["gx"] - gx_conv) < 1e-30 and d["ny"] == 64:
            conv_runs[(64, 128)] = d
    for d in conv_extra:
        conv_runs[(d["ny"], d["nx"])] = d
    grids = sorted(conv_runs)
    order = float("nan")
    velocities = {}
    for g in grids:
        d = conv_runs[g]
        u = float(d.get("velocity_centroid", float("nan")))
        if not math.isfinite(u):
            u = 0.5 * (abs(d.get("velocity_front", float("nan"))) + abs(d.get("velocity_rear", float("nan"))))
        velocities[f"{g[0]}x{g[1]}"] = u
    if len(grids) == 3:
        hs = [1.0 / g[0] for g in grids]
        us = [velocities[f"{g[0]}x{g[1]}"] for g in grids]
        if all(math.isfinite(u) for u in us):
            d12, d23 = abs(us[0] - us[1]), abs(us[1] - us[2])
            if d23 > 0.0 and d12 > 0.0:
                order = math.log(d12 / d23) / math.log(hs[0] / hs[1])
    results["convergence"] = {
        "gx": gx_conv,
        "wall_fraction": DRIVE_FRACTION,
        "grids": [f"{g[0]}x{g[1]}" for g in grids],
        "velocities": velocities,
        "richardson_order": order,
        "mass_drift": {f"{g[0]}x{g[1]}": conv_runs[g]["mass_drift"] for g in grids},
    }

    # ---- F: mass conservation table ------------------------------------------
    mass_rows = [
        {"run": f"static_f{f:.2f}", "mass_drift": float(s["mass_drift"]),
         "mass_drift_max": float(s["mass_drift_max"])}
        for f, s in zip(fractions, static)
    ] + [
        {"run": f"drive_gx{d['gx']:.1e}", "mass_drift": float(d["mass_drift"]),
         "mass_drift_max": float(d["mass_drift_max"])}
        for d in drive
    ] + [
        {"run": f"conv_{d['ny']}x{d['nx']}", "mass_drift": float(d["mass_drift"]),
         "mass_drift_max": float(d["mass_drift_max"])}
        for d in conv_extra
    ]
    mass_summary = {
        "criterion": "total mass variation <= 0.1 %",
        "runs": mass_rows,
        "max_drift": max(r["mass_drift"] for r in mass_rows) if mass_rows else float("nan"),
        "max_drift_running": max(r["mass_drift_max"] for r in mass_rows) if mass_rows else float("nan"),
    }
    results["mass_conservation"] = mass_summary
    (out / "mass_conservation.json").write_text(json.dumps(mass_summary, indent=2))

    # ---- criteria ------------------------------------------------------------
    errs = [r["error_deg"] for r in f1_rows
            if math.isfinite(r["error_deg"]) and r.get("angle_valid", True)]
    f1_ok = bool(errs) and max(errs) <= 3.0
    f2_ok = bool(fit.r2 >= 0.95)
    f3_ok = bool(math.isfinite(order) and order >= 0.5)
    f4_ok = bool(mass_summary["max_drift"] <= 1e-3)
    criteria = {
        "F1_young_angle_within_3deg": {
            "pass": f1_ok,
            "max_error_deg": max(errs) if errs else float("nan"),
            "per_wall": f1_rows,
        },
        "F2_cox_voinov_r2_ge_0.95": {
            "pass": f2_ok, "r2": fit.r2, "slope": fit.slope,
            "intercept": fit.intercept, "n_points": fit.n_points,
            "decades": fit.decades,
            "micro_length_lattice": fit.micro_length,
        },
        "F3_contact_line_velocity_order_ge_0.5": {
            "pass": f3_ok, "order": order,
            "velocities": velocities,
        },
        "F4_mass_conservation_le_0.1pct": {
            "pass": f4_ok, "max_drift": mass_summary["max_drift"],
        },
        "F6_artifacts": {
            "pass": None,
            "files": ["contact_angle_relaxation.png", "cox_voinov_fit.png",
                      "mass_conservation.json"],
        },
    }
    results["criteria"] = criteria
    results["runtime_s"] = time.time() - t_start
    (out / "results.json").write_text(json.dumps(results, indent=2))
    _figures(out, results)
    criteria["F6_artifacts"]["pass"] = all((out / f).exists() for f in
                                          criteria["F6_artifacts"]["files"])
    results["criteria"] = criteria
    (out / "criteria.json").write_text(json.dumps(criteria, indent=2))
    (out / "results.json").write_text(json.dumps(results, indent=2))
    return results


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def _figures(out: Path, results: dict) -> None:
    import numpy as np

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1) contact angle relaxation
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for s in results["static"]:
        traj = np.asarray(s["theta_traj"], dtype=float)
        if traj.size == 0:
            continue
        ax.errorbar(traj[:, 0], traj[:, 1], yerr=traj[:, 2], lw=1.2, ms=2.5,
                    label=rf"$\psi_w/\psi(\rho_l)={s['wall_fraction']:.2f}$")
    for row in results["young"]:
        if math.isfinite(row["theta_young_deg"]):
            ax.axhline(row["theta_young_deg"], ls=":", lw=0.9,
                       color=ax.lines[-1].get_color() if ax.lines else "k")
    ax.set_xlabel("lattice step")
    ax.set_ylabel(r"contact angle $\theta$ [deg]")
    ax.set_title("Sessile water drop on ice wall: angle relaxation\n"
                 "(dotted: Young prediction from measured interfacial tensions)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "contact_angle_relaxation.png", dpi=160)
    plt.close(fig)

    # 2) Cox-Voinov fit
    pts = results["cox_voinov"]["points"]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    if pts:
        ca = np.array([p["capillary"] for p in pts])
        th = np.array([math.radians(p["theta_a_deg"]) for p in pts])
        the = math.radians(results["cox_voinov"]["theta_e_deg"])
        cube = th**3 - the**3
        ax.plot(9.0 * ca, cube, "o", ms=5, label="LBM (advancing)")
        slope = results["cox_voinov"]["fit"]["slope"]
        icpt = results["cox_voinov"]["fit"]["intercept"]
        if math.isfinite(slope):
            xs = np.linspace(0.0, float(np.max(9.0 * ca)) * 1.05, 10)
            ax.plot(xs, slope * xs / 9.0 + icpt, "-",
                    label=rf"fit: $K=9\ln(L/l_m)={slope:.2f}$, $R^2={results['cox_voinov']['fit']['r2']:.3f}$")
        ax.set_xlabel(r"$9\,Ca$ (Cox-Voinov abscissa, $Ca=\mu U_{CL}/\gamma$)")
        ax.set_ylabel(r"$\theta_d^3-\theta_Y^3$ [rad$^3$]")
        ax.set_title("Cox-Voinov law on the water/ice contact line")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "cox_voinov_fit.png", dpi=160)
    plt.close(fig)

    # 3) overview: density field of the reference static drop + theta map
    static0 = next((s for s in results["static"] if "rho_field" in s), None)
    drive0 = next((d for d in results["drive"] if "rho_field" in d), None)
    if static0 or drive0:
        fig, axes = plt.subplots(1, 2 if (static0 and drive0) else 1,
                                 figsize=(11 if (static0 and drive0) else 5.5, 3.6),
                                 squeeze=False)
        for ax, data, title in (
            (axes[0, 0], static0, "static"),
            (axes[0, -1], drive0, "driven"),
        ):
            if data is None:
                continue
            field = np.asarray(data["rho_field"], dtype=float)
            im = ax.imshow(field, origin="lower", cmap="viridis")
            ax.set_title(f"{title}: $\\psi_w/\\psi(\\rho_l)={data['wall_fraction']:.2f}$, "
                         f"$g_x={data['gx']:.1e}$")
            ax.set_xlabel("x [lattice]")
            ax.set_ylabel("y [lattice]")
            fig.colorbar(im, ax=ax, shrink=0.85, label=r"$\rho$")
        fig.tight_layout()
        fig.savefig(out / "density_fields.png", dpi=160)
        plt.close(fig)

    # 4) wetting tension and Young/metry comparison
    young = results["young"]
    if young:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        fr = [r["wall_fraction"] for r in young]
        ax.plot(fr, [r["theta_young_deg"] for r in young], "o-", label=r"Young: $\arccos(\Delta\gamma/\gamma_{lg})$")
        ax.errorbar(fr, [r["theta_measured_deg"] for r in young],
                    yerr=[r["theta_measured_unc_deg"] for r in young], fmt="s--",
                    label="LBM sessile drop (circle fit)")
        ax.set_xlabel(r"wall coupling $\psi_w/\psi(\rho_l)$")
        ax.set_ylabel(r"contact angle $\theta$ [deg]")
        ax.set_title("Young's equation as a prediction")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "young_equation.png", dpi=160)
        plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--quick", action="store_true", help="smoke run (numbers not for acceptance)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    res = run_campaign(out, jobs=args.jobs, quick=args.quick)
    print(json.dumps(res["criteria"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())