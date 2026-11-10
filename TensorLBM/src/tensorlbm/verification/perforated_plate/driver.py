"""Verification driver: KC / beta / GCI matrices + negative controls.

The driver is splittable so the matrix can be distributed over the cluster:
``python -m tensorlbm.verification.perforated_plate.driver --shard k/n``
runs cases ``k, k+n, k+2n, ...`` and writes one JSON per case.

Usage
-----
    python -m ...driver --list
    python -m ...driver --out runs/pp --shard 0/6
    python -m ...driver --out runs/pp --assemble
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

from .solver import PerforatedPlateConfig, PerforatedPlateSolver
from .analysis import (
    analyze_simulation,
    first_harmonic,
    phase_average,
)
from .reference import (
    sharp_edge_cd,
    womersley_channel_u,
    womersley_channel_amplitude,
    womersley_wall_shear_amplitude,
    stokes_layer_thickness,
)

# ---------------------------------------------------------------------------
# Case matrix
# ---------------------------------------------------------------------------

#: log-spaced body-force ladder for the KC sweep (realised KC is reported per
#: case; the ladder spans roughly KC = 0.5 ... 20 for the reference geometry)
#: body-force ladder of the KC sweep.  The ladder is chosen so that the loop
#: inertia (proportional to F0) and the plate drag both stay in the measurable
#: range; realised KC is reported per case (approx. 0.5 ... 27).
KC_FORCE_LADDER = [2.4e-6, 3.9e-6, 6.4e-6, 1.04e-5, 1.7e-5, 2.77e-5,
                   4.5e-5, 7.35e-5, 1.2e-4, 1.9e-4]

#: reference geometry of the KC sweep: beta = 3*8/96 = 0.25
BASE = dict(nx=256, ny=98, n_holes=3, hole_size=8, plate_thickness=2,
            tau=0.5075, omega=2.0 * math.pi / 1600.0, warmup_cycles=3,
            n_cycles_measure=2, samples_per_cycle=32)

#: beta sweep: d = 6 fixed, ny = 98 -> beta = n_holes * 6 / 96
BETA_CASES = [(1, 0.0625), (2, 0.125), (3, 0.1875), (5, 0.3125), (6, 0.375)]
BETA_BASE = dict(nx=192, ny=98, plate_thickness=2, hole_size=6, tau=0.51,
                 omega=2.0 * math.pi / 1200.0, warmup_cycles=3,
                 n_cycles_measure=2, samples_per_cycle=32)
BETA_KC_TARGET = 12.0

GCI_CASES = [  # self-similar geometry, ratio r = 2 between levels
    dict(nx=102, ny=34, hole_size=4, plate_thickness=1, tau=0.515,
         omega=2.0 * math.pi / 800.0),
    dict(nx=204, ny=66, hole_size=8, plate_thickness=2, tau=0.53,
         omega=2.0 * math.pi / 1600.0),
    dict(nx=390, ny=130, hole_size=16, plate_thickness=4, tau=0.56,
         omega=2.0 * math.pi / 3200.0),
]


def _cd_guess(beta: float) -> float:
    return sharp_edge_cd(beta)


def _force_for_target_u_h(beta: float, u_h: float, nx: int, omega: float) -> float:
    """Body force amplitude that realises ``u_h`` in the harmonic balance

    ``F0 * Lx = sqrt((omega*beta*u_h)^2 + (0.5*K_q*u_h^2/Lx)^2)``

    which combines the inertia of the (periodic) fluid loop with the
    quasi-steady plate loss ``K_q = 1/C_d^2``.
    """
    k_q = 1.0 / _cd_guess(beta) ** 2
    inertia = omega * beta * u_h
    loss = 0.5 * k_q * u_h * u_h / float(nx)
    return math.hypot(inertia, loss)


def build_cases() -> list[dict]:
    """Return the full verification matrix as a list of case dictionaries."""
    cases: list[dict] = []

    for i, f0 in enumerate(KC_FORCE_LADDER):
        cases.append(dict(
            id=f"kcA{i:02d}", kind="kc", force_amplitude=f0,
            **{**BASE, "label": f"kc_{f0:.1e}"},
        ))

    for (n_holes, beta_t) in BETA_CASES:
        cfg = dict(BETA_BASE)
        cfg.update(n_holes=n_holes)
        u_h = BETA_KC_TARGET * cfg["hole_size"] * cfg["omega"] / (2.0 * math.pi)
        f0 = _force_for_target_u_h(beta_t, u_h, cfg["nx"], cfg["omega"])
        cases.append(dict(
            id=f"bA_{int(round(beta_t * 1000)):03d}", kind="beta",
            force_amplitude=f0, beta_target=beta_t,
            **{**cfg, "label": f"beta_{beta_t:.4f}"},
        ))

    for i, lev in enumerate(GCI_CASES):
        cfg = dict(BASE)
        cfg.update(lev)
        cfg.update(n_holes=2, warmup_cycles=2, n_cycles_measure=2)
        beta_t = 2 * lev["hole_size"] / (lev["ny"] - 2)
        u_h = 0.04
        f0 = _force_for_target_u_h(beta_t, u_h, cfg["nx"], cfg["omega"])
        cases.append(dict(
            id=f"gci{i}", kind="gci", force_amplitude=f0,
            beta_target=beta_t, level=i,
            **{**cfg, "label": f"gci_L{i}"},
        ))

    # negative control: unobstructed oscillating channel (no plate at all)
    cases.append(dict(
        id="nc_channel", kind="nc", force_amplitude=5.0e-5, beta=1.0,
        nx=192, ny=66, n_holes=1, hole_size=6, plate_thickness=2,
        tau=0.65, omega=2.0 * math.pi / 3000.0, warmup_cycles=3,
        n_cycles_measure=2, samples_per_cycle=24, probe_wall_shear=True,
        label="nc_channel",
    ))
    return cases


# ---------------------------------------------------------------------------
# Single case
# ---------------------------------------------------------------------------

def _config_from_case(case: dict) -> PerforatedPlateConfig:
    keys = ("nx", "ny", "n_holes", "hole_size", "plate_thickness", "tau",
            "force_amplitude", "omega", "warmup_cycles", "n_cycles_measure",
            "samples_per_cycle", "label")
    kwargs = {k: case[k] for k in keys if k in case}
    kwargs["beta"] = case.get("beta", -1.0)
    kwargs["probe_wall_shear"] = bool(case.get("probe_wall_shear", False))
    return PerforatedPlateConfig(**kwargs)


def run_case(case: dict, outdir: str | None = None) -> dict:
    """Run and analyse a single case; returns a JSON-serialisable dict."""
    cfg = _config_from_case(case)
    t0 = time.time()
    solver = PerforatedPlateSolver(cfg)
    result = solver.run()
    wall = time.time() - t0

    beta = float(result["beta"])
    an = analyze_simulation(result, cfg.omega, beta, cfg.ny,
                            n_phase_bins=cfg.samples_per_cycle,
                            plate_thickness=cfg.plate_thickness)
    T = 2.0 * math.pi / cfg.omega
    d_h = float(cfg.hole_size)
    kc = an["u_h_amp"] * T / d_h
    re_h = an["u_h_amp"] * d_h / cfg.nu

    out = {
        "id": case["id"],
        "kind": case["kind"],
        "label": case.get("label", case["id"]),
        "beta": beta,
        "beta_target": case.get("beta_target", beta),
        "nx": cfg.nx, "ny": cfg.ny, "hole_size": cfg.hole_size,
        "n_holes": cfg.n_holes, "plate_thickness": cfg.plate_thickness,
        "tau": cfg.tau, "nu": cfg.nu,
        "period": T, "force_amplitude": cfg.force_amplitude,
        "steps": int(result["time_series"]["time"].size),
        "wall_seconds": wall,
        "KC": kc, "Re_h": re_h,
        "u_h_amp": an["u_h_amp"], "q_amp": an["q_amp"],
        "C_d": an["C_d"], "C_m": an["C_m"], "K_m": an["K_m"], "K_q": an["K_q"],
        "r_squared": an["r_squared"],
        "C_d_stations": an["C_d_stations"], "C_m_stations": an["C_m_stations"],
        "phase_lag_deg": an["phase_lag_deg"],
        "phase_lag_deg_stations": an["phase_lag_deg_stations"],
        "phi": an["phi"], "phi_stations": an["phi_stations"],
        "phi_model": an["phi_model"],
        "e_kin": an["e_kin"],
        "zeta_eff": an["zeta_eff"], "zeta_eff_stations": an["zeta_eff_stations"],
        "zeta_model": an["zeta_model"],
        "stationarity": an["stationarity"],
        "C_d_reference": sharp_edge_cd(beta),
        "dp_inertial_amp": an["dp_inertial_amp"],
        "dp_loss_amp": an["dp_loss_amp"],
        "phase": {
            "theta": np.asarray(an["theta"]).round(6).tolist(),
            "q": np.asarray(an["phase_q"]).round(9).tolist(),
            "dp": np.asarray(an["phase_dp"]).round(9).tolist(),
            "dp_stations": np.asarray(an["phase_dp_stations"]).round(9).tolist(),
        },
    }

    # ---- field snapshot near the peak flow (for the report figure) --------
    snaps = result.get("snapshots", [])
    if snaps and outdir:
        qvals = [abs(float(np.sum(s["ux"][solver.i_q, 1:-1]))) for s in snaps]
        kk = int(np.argmax(qvals))
        snap = snaps[kk]
        vort = (np.roll(snap["uy"], -1, axis=0) - np.roll(snap["uy"], 1, axis=0)
                - (np.roll(snap["ux"], -1, axis=1) - np.roll(snap["ux"], 1, axis=1)))
        np.savez_compressed(
            os.path.join(outdir, f"field_{case['id']}.npz"),
            ux=snap["ux"].astype(np.float32),
            uy=snap["uy"].astype(np.float32),
            vorticity=vort.astype(np.float32),
            solid=solver.solid.astype(np.int8),
            t=float(snap["t"]), kc=float(kc),
        )

    # ---- negative-control post-processing --------------------------------
    if case["kind"] == "nc":
        out.update(_negative_control_metrics(cfg, result, solver))
    return out


def _negative_control_metrics(cfg: PerforatedPlateConfig, result: dict,
                              solver: PerforatedPlateSolver) -> dict:
    """NC1 (unobstructed channel pressure budget) and NC2 (wall shear)."""
    ts = result["time_series"]
    time = ts["time"]
    omega = cfg.omega
    h = float(cfg.ny - 2)
    nu = cfg.nu
    f0 = float(cfg.force_amplitude)
    width = float(solver.window_width)

    theta, dp_meas = phase_average(time, ts["dp_stations"], omega, 48)
    _, q_meas = phase_average(time, ts["flow_rate"], omega, 48)
    _, tau_meas = phase_average(time, ts["force_lower_wall_x"], omega, 48)
    tau_meas = tau_meas / float(cfg.nx)          # force -> wall shear stress

    a_dp = first_harmonic(theta, dp_meas)
    a_q = first_harmonic(theta, q_meas)
    a_tau = first_harmonic(theta, tau_meas)

    # analytic Womersley references sampled on the same phase grid
    t_ref = theta / omega
    y = np.arange(cfg.ny, dtype=np.float64) - 0.5 * (cfg.ny - 1)
    u_prof = np.array([womersley_channel_u(y, tt, h, nu, f0, omega)
                       for tt in t_ref])
    q_ref = u_prof[:, 1:-1].sum(axis=1)
    from .reference import womersley_wall_shear
    tau_ref = np.array([womersley_wall_shear(tt, h, nu, f0, omega)
                        for tt in t_ref])

    a_q_ref = first_harmonic(theta, q_ref)
    a_tau_ref = first_harmonic(theta, tau_ref)

    # NC1: without a plate the streamwise pressure field is x-periodic and
    # x-uniform (the uniform body force needs no streamwise pressure gradient),
    # so the station-derived plate pressure drop must vanish.  The natural
    # yardstick is the driving pressure scale rho*F0*W over the probe window.
    dp_drive_scale = abs(f0) * float(solver.window_width)

    def _rel(a, b):
        if abs(b) == 0:
            return float("nan")
        return float(abs(a - b) / abs(b))

    # velocity-profile comparison: harmonic amplitude of u(y,t)
    snaps = result.get("snapshots", [])
    prof_err_l2 = float("nan")
    prof = None
    prof_ref = None
    if len(snaps) > 4:
        amps = np.array([first_harmonic(np.array([omega * s["t"] for s in snaps]),
                                        np.array([s["ux"][solver.i_q, j] for s in snaps]))
                         for j in range(cfg.ny)])
        prof_ref = womersley_channel_amplitude(y, h, nu, f0, omega)
        prof = np.abs(amps)
        inner = slice(1, cfg.ny - 1)
        num = np.sum((prof[inner] - prof_ref[inner]) ** 2)
        den = np.sum(prof_ref[inner] ** 2)
        prof_err_l2 = float(math.sqrt(num / max(den, 1e-30)))

    stokes_delta = stokes_layer_thickness(omega, nu)
    return {
        "nc": {
            "stokes_delta": stokes_delta,
            "stokes_delta_over_h": stokes_delta / h,
            "dp_amp_measured": float(abs(a_dp)),
            "dp_amp_analytic": 0.0,
            "dp_drive_scale": dp_drive_scale,
            "dp_rel_error": float(abs(a_dp) / max(dp_drive_scale, 1e-30)),
            "dp_phase_measured_deg": math.degrees(np.angle(a_dp)),
            "dp_phase_analytic_deg": 0.0,   # x-uniform field: no streamwise dp
            "q_amp_measured": float(abs(a_q)),
            "q_amp_analytic": float(abs(a_q_ref)),
            "q_rel_error": _rel(abs(a_q), abs(a_q_ref)),
            "tau_amp_measured": float(abs(a_tau)),
            "tau_amp_analytic": float(abs(a_tau_ref)),
            "tau_rel_error": _rel(abs(a_tau), abs(a_tau_ref)),
            "tau_phase_measured_deg": math.degrees(np.angle(a_tau)),
            "tau_phase_analytic_deg": math.degrees(np.angle(a_tau_ref)),
            "tau_phase_error_deg": float(abs(math.degrees(np.angle(a_tau / a_tau_ref)))),
            "profile_rel_error_l2": prof_err_l2,
            "y": y.tolist(),
            "profile_amp": None if prof is None else prof.round(8).tolist(),
            "profile_amp_ref": None if prof_ref is None else np.asarray(prof_ref).round(8).tolist(),
        }
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Perforated-plate verification driver")
    ap.add_argument("--out", default="runs/perforated_plate_n6")
    ap.add_argument("--shard", default=None, help="i/n -> run cases i, i+n, ...")
    ap.add_argument("--only", default=None, help="comma separated case ids")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--assemble", action="store_true")
    args = ap.parse_args(argv)

    cases = build_cases()
    if args.list:
        for c in cases:
            print(f"{c['id']:12s} kind={c['kind']:5s} F0={c['force_amplitude']:.2e} "
                  f"nx={c['nx']} ny={c['ny']} d={c['hole_size']} "
                  f"t={c['plate_thickness']} tau={c['tau']} T={2*math.pi/c['omega']:.0f}")
        return 0

    os.makedirs(args.out, exist_ok=True)

    if args.assemble:
        from .assemble import assemble
        path = assemble(args.out)
        print(f"assembled -> {path}")
        return 0

    if args.only:
        wanted = set(x.strip() for x in args.only.split(","))
        cases = [c for c in cases if c["id"] in wanted]
    elif args.shard:
        k, n = (int(v) for v in args.shard.split("/"))
        cases = [c for i, c in enumerate(cases) if i % n == k]

    for c in cases:
        path = os.path.join(args.out, f"case_{c['id']}.json")
        if os.path.exists(path):
            print(f"[skip] {c['id']} (exists)", flush=True)
            continue
        print(f"[run ] {c['id']} kind={c['kind']} F0={c['force_amplitude']:.2e}",
              flush=True)
        t0 = time.time()
        res = run_case(c, outdir=args.out)
        with open(path, "w") as fh:
            json.dump(res, fh, indent=1)
        print(f"[done] {c['id']} KC={res['KC']:.2f} C_d={res['C_d']:.4f} "
              f"lag={res['phase_lag_deg']:.1f} T={time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())