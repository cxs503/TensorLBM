"""S2 — wedge constant-velocity water entry vs Wagner (phase-field CAC + moving body).

Case definition (locked in ``prereg.md`` §2.1, with the resolution amendment
recorded in ``NOTES.md``): a symmetric 2-D wedge (extruded over 4 periodic z
layers) enters a quiescent water tank at constant vertical velocity.  The
force on the body is measured with Ladd's momentum-exchange method and
compared with Wagner (1932) in the pi^2/4 convention,

    F(t) = rho_l V^3 pi^3 t / (8 tan^2 zeta)      [per unit span]
    dF/dt / (rho_l V^3) = pi^3 / (8 tan^2 zeta)

Because Wagner's force is exactly linear in t, the compared quantity is the
**slope** dF/dt (a point-wise force would be dominated by the definition of
t = 0 and by the diffuse interface).

Gate bands are deadrise-dependent (prereg §3): 2 deg <= 10 %, 3 deg <= 35 %,
4 deg <= 50 %, mirroring the published reference implementation's own errors
(6.3 / 29.5 / 44.2 %).

Run
---
    PYTHONPATH=src python benchmarks/pending/phasefield_water_entry/run_s2_wedge.py \
        --out-dir /tmp/s2 --arm zeta4_dw200
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from tensorlbm import cac_lbm as m
from tensorlbm.cac_moving_body import MovingBodyCAC, wall_box, wedge_mask
from tensorlbm.wagner_theory import WedgeEntryParams, wagner_wedge_force_slope

GATES = {2.0: 0.10, 3.0: 0.35, 4.0: 0.50}


@dataclass(frozen=True)
class WedgeRun:
    zeta_deg: float = 4.0
    half_length: float = 100.0  # D_w in lattice cells
    v_lat: float = 0.01  # entry velocity (Ma = 0.017); 0.05 blows the CAC up
    nz: int = 4
    depth_water: int = 48  # water column below the initial surface (12 interface widths)
    air_height: int = 14  # room for the run-up sheet, not a full free surface
    gap: float = 1.0  # apex stand-off above the interface at t = 0
    n_pre: int = 400  # hydrostatic pre-relaxation with the body held back
    n_measure: int = 1200  # measurement window (steps) -> 12 cells of penetration
    side_margin: float = 1.15  # domain half-width in units of D_w
    rho_g: float = 0.01
    nu: float = 0.1
    sigma: float = 0.01
    width: float = 4.0
    mobility: float = 0.05
    g: float = 1e-4
    thickness: float = 6.0

    @property
    def shape(self) -> tuple[int, int, int]:
        nx = int(round(2.0 * self.side_margin * self.half_length))
        ny = self.depth_water + self.air_height
        return (self.nz, ny, nx)

    @property
    def y_interface(self) -> float:
        return float(self.depth_water)

    @property
    def x_centre(self) -> float:
        return self.shape[2] / 2.0


def run(config: WedgeRun, out_dir: Path) -> dict:
    shape = config.shape
    nz, ny, nx = shape
    torch.manual_seed(0)
    wall = wall_box(shape, z_periodic=True)
    sim = m.CACSim(
        shape,
        rho_l=1.0,
        rho_g=config.rho_g,
        mu_l=config.nu,
        mu_g=config.nu * config.rho_g,
        sigma=config.sigma,
        width=config.width,
        mobility=config.mobility,
        p0=0.01,
        gravity=(0.0, -config.g, 0.0),
        gravity_mode="rho",
        axes="pwp",
        wall_mask=wall,
        wall_phi_policy="neumann",
    )
    sim.initialize(m.init_phi_layers(shape, width=config.width, y0=config.y_interface))
    driver = MovingBodyCAC(sim, static_wall=wall)

    zeta = math.radians(config.zeta_deg)
    y_apex0 = config.y_interface + config.gap

    def body_at(y_apex: float) -> torch.Tensor:
        return (
            wedge_mask(
                shape,
                apex_xy=(config.x_centre, y_apex),
                deadrise_rad=zeta,
                half_length=config.half_length,
                thickness=config.thickness,
            )
            & ~wall
        )

    body = body_at(y_apex0)
    driver.set_body(body)
    t0 = time.time()

    # --- pre-relaxation: body held back, hydrostatics settle -------------
    for _ in range(config.n_pre):
        driver.step(body, (0.0, 0.0, 0.0))
    d = m.diagnostics(sim)
    pre = {
        "step": 0,
        "u_max": d["u_max_fluid"],
        "phi_min": d["phi_min_fluid"],
        "phi_max": d["phi_max_fluid"],
    }
    print(
        f"[s2] pre-relax done in {time.time() - t0:.0f}s  max|u| {d['u_max_fluid']:.2e}  "
        f"phi [{d['phi_min_fluid']:.3f}, {d['phi_max_fluid']:.3f}]",
        flush=True,
    )

    # --- measurement: constant-velocity entry ---------------------------
    series = []
    t_start = time.time()
    for step in range(1, config.n_measure + 1):
        y_apex = y_apex0 - config.v_lat * step
        driver.step(body_at(y_apex), (0.0, -config.v_lat, 0.0))
        series.append(
            {
                "step": step,
                "t": float(step),
                "penetration": config.v_lat * step - config.gap,
                "fy": driver.force[1],
                "fy_phase": driver.force_phase[1],
                "fz": driver.force[2],
                "y_apex": y_apex,
            }
        )
        if step % 50 == 0:
            dd = m.diagnostics(sim)
            series[-1]["u_max"] = dd["u_max_fluid"]
            series[-1]["phi_min"] = dd["phi_min_fluid"]
            series[-1]["phi_max"] = dd["phi_max_fluid"]
            print(
                f"[s2]   step {step:4d}  Fy {driver.force[1]:+.4e}  "
                f"pen {series[-1]['penetration']:.1f}  max|u| {dd['u_max_fluid']:.2e}",
                flush=True,
            )

    # --- compare slope ---------------------------------------------------
    # analytic slope with rho_l = 1, v = v_lat, in lattice units
    params = WedgeEntryParams(beta=zeta, v_entry=config.v_lat, rho=1.0)
    analytic_slope = wagner_wedge_force_slope(params)
    # fit over the linear window: penetration in [0.02, 0.10] * D_w is not
    # reachable at these angles, so fit over the whole measured window and
    # report both the fit and its R^2.
    arm = f"zeta{config.zeta_deg:g}_dw{config.half_length:g}"
    contact_step = int(math.ceil(config.gap / config.v_lat))
    used = [s for s in series if s["step"] > contact_step]
    if len(used) < 3:
        raise SystemExit(
            f"[s2] {arm}: only {len(used)} post-contact samples "
            f"(contact at step {contact_step}); raise --n-measure"
        )
    xs = [s["t"] for s in used]
    ys = [s["fy"] for s in used]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    slope = sxy / sxx
    ss_res = sum((y - (my + slope * (x - mx))) ** 2 for x, y in zip(xs, ys, strict=True))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # the Ladd sum runs over every periodic z layer -> per-unit-span force
    slope_span = slope / config.nz
    err = (slope_span - analytic_slope) / analytic_slope
    gate = GATES.get(config.zeta_deg)
    u_over_v = max((s.get("u_max", 0.0) for s in series), default=float("nan")) / config.v_lat
    gate_pass = None if gate is None else bool(abs(err) <= gate)
    result = {
        "arm": arm,
        "case": "S2_wedge_constant_velocity",
        "config": asdict(config),
        "shape": list(shape),
        "cells": nz * ny * nx,
        "deadrise_deg": config.zeta_deg,
        "analytic_slope_dimensionless": analytic_slope,
        "measured_slope_total_nz": slope,
        "measured_slope": slope_span,
        "measured_slope_dimensionless": slope_span / (1.0 * config.v_lat**3),
        "slope_rel_error": err,
        "fit_r2": r2,
        "gate": gate,
        "gate_pass": gate_pass,
        "gate_r2_pass": bool(r2 >= 0.98) if r2 == r2 else False,
        "gate_u_max_pass": bool(u_over_v <= 3.0) if u_over_v == u_over_v else False,
        "max_u_over_v": u_over_v,
        "verdict": "PASS"
        if (gate_pass and r2 >= 0.98 and u_over_v <= 3.0)
        else ("FAIL" if gate_pass is not None else "NO_GATE"),
        "pre_relax": pre,
        "contact_step": contact_step,
        "fit_window_steps": [int(xs[0]), int(xs[-1])] if xs else None,
        "fit_window_penetration_cells": [
            config.v_lat * xs[0] - config.gap,
            config.v_lat * xs[-1] - config.gap,
        ]
        if xs
        else None,
        "phi_range": [
            min((s.get("phi_min", 0.0) for s in series), default=float("nan")),
            max((s.get("phi_max", 1.0) for s in series), default=float("nan")),
        ],
        "wall_time_s": time.time() - t_start,
        "series": series,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"s2_{arm}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"[s2] zeta={config.zeta_deg:g} D_w={config.half_length:g}: slope {slope:+.4e} vs analytic "
        f"{analytic_slope:+.4e} -> err {err:+.1%} (gate {gate if gate is None else f'{gate:.0%}'}) "
        f"R2 {r2:.4f}",
        flush=True,
    )
    return result


ARMS = {
    # workhorse resolution: D_w = 100 cells, penetration window 0..12 cells
    "zeta8_dw100": dict(zeta_deg=8.0, half_length=100.0),
    "zeta4_dw100": dict(zeta_deg=4.0, half_length=100.0),
    "zeta3_dw100": dict(zeta_deg=3.0, half_length=100.0),
    "zeta2_dw100": dict(zeta_deg=2.0, half_length=100.0),
    # resolution check (two grids, same physics)
    "zeta4_dw200": dict(zeta_deg=4.0, half_length=200.0, depth_water=96, air_height=28),
    "zeta2_dw200": dict(
        zeta_deg=2.0, half_length=200.0, depth_water=96, air_height=28, n_measure=1800
    ),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--arm",
        nargs="+",
        default=["zeta4_dw100"],
        choices=sorted(ARMS),
        help="one or more arms to run in this process (sequentially)",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("/tmp/s2"))
    ap.add_argument("--n-measure", type=int, default=None)
    ap.add_argument("--n-pre", type=int, default=None)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    results = []
    for arm in args.arm:
        kw = dict(ARMS[arm])
        if args.n_measure is not None:
            kw["n_measure"] = args.n_measure
        if args.n_pre is not None:
            kw["n_pre"] = args.n_pre
        results.append(run(WedgeRun(**kw), args.out_dir))
    summary = [
        {
            "arm": r["arm"],
            "zeta_deg": r["config"]["zeta_deg"],
            "dw": r["config"]["half_length"],
            "slope": r["measured_slope"],
            "slope_analytic": r["analytic_slope_dimensionless"],
            "slope_error": r["slope_rel_error"],
            "r2": r["fit_r2"],
            "gate": r["gate"],
            "verdict": r["verdict"],
        }
        for r in results
    ]
    (args.out_dir / "s2_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
