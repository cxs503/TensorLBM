"""S2 — wedge constant-velocity water entry vs Wagner (phase-field CAC + moving body).

Case definition (locked in ``prereg.md`` §2.1, with the resolution amendment
recorded in ``NOTES.md``): a symmetric 2-D wedge (extruded over 4 periodic z
layers) enters a quiescent water tank at constant vertical velocity.  The
force on the body is measured with Ladd's momentum-exchange method and
compared diagnostically with the semi-wedge anchor in De Rosis & Tafuni Eq. 45,

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


def arm_metrics(config: WedgeRun, pre: dict, series: list[dict]) -> dict:
    """Score the registered penetration window. Missing observations cannot pass."""
    first = math.ceil((config.gap + 0.02 * config.half_length) / config.v_lat)
    last = math.floor((config.gap + 0.10 * config.half_length) / config.v_lat)
    samples = [r for r in series if first <= r["step"] <= last]
    complete = len(samples) >= 3 and [r["step"] for r in samples] == list(range(first, last + 1))
    invalid = next(
        (r for r in [pre, *series] if any(not math.isfinite(v) for v in r.values())), None
    )
    params = WedgeEntryParams(beta=math.radians(config.zeta_deg), v_entry=config.v_lat, rho=1.0)
    reference = wagner_wedge_force_slope(params)
    scale = params.rho * config.v_lat**3
    slope = r2 = err = None
    if complete and invalid is None:
        xs = [r["t"] for r in samples]
        ys = [r["fy"] for r in samples]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sum(
            (x - mx) ** 2 for x in xs
        )
        sst = sum((y - my) ** 2 for y in ys)
        if sst > 0:
            r2 = (
                1 - sum((y - my - slope * (x - mx)) ** 2 for x, y in zip(xs, ys, strict=True)) / sst
            )
        err = slope / config.nz / reference - 1
    speeds = [r["u_max"] for r in series if "u_max" in r and math.isfinite(r["u_max"])]
    u_over_v = max(speeds) / config.v_lat if speeds else None
    phase = [
        r
        for r in series
        if "phi_min" in r and math.isfinite(r["phi_min"]) and math.isfinite(r["phi_max"])
    ]
    gate = GATES.get(config.zeta_deg)
    checks = {
        "gate_pass": None if gate is None else err is not None and abs(err) <= gate,
        "gate_r2_pass": r2 is not None and r2 >= 0.98,
        "gate_u_max_pass": u_over_v is not None and u_over_v <= 3.0,
    }
    full_run = bool(series) and series[-1]["step"] == config.n_measure
    verdict = "FAIL"
    if invalid is None and full_run and complete:
        verdict = (
            "NO_GATE" if gate is None else ("DIAGNOSTIC_PASS" if all(checks.values()) else "FAIL")
        )
    return {
        "verdict": verdict,
        "scope": "Single-arm diagnostic: full-body vertical MEM force per span versus Eq. 45 semi-wedge pressure. A pass does not qualify the entry case.",
        "first_nonfinite_step": invalid["step"] if invalid is not None else None,
        "measurement_completed": full_run,
        "requested_fit_window_steps": [first, last],
        "fit_window_steps": [first, last] if slope is not None else None,
        "fit_window_penetration_cells": [r["penetration"] for r in (samples[0], samples[-1])]
        if slope is not None
        else None,
        "contact_step": math.ceil(config.gap / config.v_lat),
        "analytic_slope": reference,
        "analytic_slope_dimensionless": reference / scale,
        "measured_slope_total_nz": slope,
        "measured_slope": slope / config.nz if slope is not None else None,
        "measured_slope_dimensionless": slope / (config.nz * scale) if slope is not None else None,
        "slope_rel_error": err,
        "fit_r2": r2,
        "gate": gate,
        **checks,
        "max_u_over_v": u_over_v,
        "phi_range": [min(r["phi_min"] for r in phase), max(r["phi_max"] for r in phase)]
        if phase
        else None,
    }


def json_observation(row):
    return {k: v if math.isfinite(v) else None for k, v in row.items()}


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
    steps = config.n_measure if all(math.isfinite(v) for v in pre.values()) else 0
    for step in range(1, steps + 1):
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
        if step % 50 == 0 or step == config.n_measure or not math.isfinite(driver.force[1]):
            dd = m.diagnostics(sim)
            series[-1]["u_max"] = dd["u_max_fluid"]
            series[-1]["phi_min"] = dd["phi_min_fluid"]
            series[-1]["phi_max"] = dd["phi_max_fluid"]
            print(
                f"[s2]   step {step:4d}  Fy {driver.force[1]:+.4e}  "
                f"pen {series[-1]['penetration']:.1f}  max|u| {dd['u_max_fluid']:.2e}",
                flush=True,
            )

        if not all(math.isfinite(v) for v in series[-1].values()):
            break

    arm = f"zeta{config.zeta_deg:g}_dw{config.half_length:g}"
    result = {
        "arm": arm,
        "case": "S2_wedge_constant_velocity",
        "config": asdict(config),
        "shape": list(shape),
        "cells": nz * ny * nx,
        "deadrise_deg": config.zeta_deg,
        **arm_metrics(config, pre, series),
        "pre_relax": json_observation(pre),
        "wall_time_s": time.time() - t_start,
        "series": [json_observation(row) for row in series],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"s2_{arm}.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        f"[s2] {arm}: {result['verdict']}, first nonfinite step={result['first_nonfinite_step']}",
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
            "slope_analytic": r["analytic_slope"],
            "first_nonfinite_step": r["first_nonfinite_step"],
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
    return 0 if all(r["verdict"] == "DIAGNOSTIC_PASS" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
