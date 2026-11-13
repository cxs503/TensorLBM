"""Execute one arm of ``XVAL-LBM-LAYERED-POISEUILLE-ANALYTIC-INLET``.

Usage
-----
    python -m tensorlbm.verification.layered_poiseuille.run_case --arm S32 \
        --out artifacts/xval_layered_poiseuille_analytic_inlet

Writes one JSON per arm with the analytic reference, the numerical KPIs, the
TCLB-port integral globals and the run bookkeeping.  Nothing is post-processed
here: the analysis script reads these JSONs only.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from typing import Dict, List

import numpy as np

from . import analytic as A
from . import cases
from .metrics import (layer_fluxes, peak_metrics, profile_errors, rel_err,
                      track_interface_y, update_my_globals)
from .solver import ChannelConfig, LayeredChannel


def _reference(spec: Dict) -> A.LayeredSolution:
    height = float(spec["height"])
    interfaces = [f * height for f in spec["interface_fracs"]]
    return A.solve_layered(height=height, interfaces=interfaces, nus=spec["nus"], g=1.0)


def run_arm(spec: Dict, outdir: str) -> Dict:
    t0 = time.time()
    height = int(spec["height"])
    interfaces = [f * height for f in spec["interface_fracs"]]
    ref_unit = A.solve_layered(height=float(height), interfaces=interfaces,
                               nus=spec["nus"], g=1.0)

    cfg = ChannelConfig(
        height=height, nx=int(spec["nx"]), nus=tuple(spec["nus"]),
        interface_fracs=tuple(spec["interface_fracs"]), u_peak=float(spec["u_peak"]),
        x_boundary=spec["x_boundary"], init_mode=spec["init_mode"],
        inlet_from_analytic=spec.get("inlet_from_analytic", True),
        steps=int(spec["steps"]), sample_every=int(spec["sample_every"]),
        tol_steady=float(spec["tol_steady"]), label=spec["label"],
    )
    ch = LayeredChannel(cfg, unit_profile=lambda y: ref_unit.u(y))
    g = ch.g
    ref = A.solve_layered(height=float(height), interfaces=interfaces,
                          nus=spec["nus"], g=g)          # exact solution at the arm force

    y_wall = ch.y_wall
    u_exact = ref.u(y_wall)

    # analytic inlet: profile at the inlet plane is the exact developed profile
    ch.set_inlet_profile(u_exact)

    # ---- transient bookkeeping: steps needed to reach eps_inf < 1e-3 ------- #
    trace: List[Dict[str, float]] = []
    x_dev = None if cfg.x_boundary == "periodic" else int(round(2.0 * height))

    def monitor(step: int, solver: LayeredChannel) -> None:
        prof = solver.profile(x_dev)
        e = profile_errors(prof, u_exact)
        trace.append({"step": step, **e})
        if step % max(cfg.sample_every * 10, 1) == 0:
            print(f"  {spec['label']} step={step} eps_inf={e['eps_inf']:.3e}", flush=True)

    info = ch.run(monitor=monitor)
    wall = time.time() - t0

    # ---- final snapshot ---------------------------------------------------- #
    rho, ux, uy = ch.velocity()
    prof = ch.profile(x_dev)
    errs = profile_errors(prof, u_exact)
    q_num = layer_fluxes(prof, y_wall, interfaces)          # discrete functional, numerical
    q_ana = layer_fluxes(u_exact, y_wall, interfaces)        # same functional, exact samples
    q_ana_cont = [ref.flow_rate_layer(j) for j in range(len(ref.nus))]   # continuum integral
    pk = peak_metrics(prof, y_wall)

    # static phase marker: pf = 0 in the bottom layer, 1 above the first interface
    # (TCLB convention: pf >= 0.5 is the "liquid" set).  The interface is prescribed,
    # so this marker doubles as the input of the TCLB interface-tracking port.
    layer_idx = np.searchsorted(np.array(interfaces, dtype=float), y_wall, side="right")
    pf = np.zeros((ch.ny, ch.nx))
    pf[1:-1, :] = (layer_idx >= 1).astype(float)[:, None]
    flux_mask = np.zeros_like(pf, dtype=bool)
    flux_mask[1:-1, :] = True
    glob = update_my_globals(ux, uy, pf, flux_mask=flux_mask)

    iface_tracked = track_interface_y(pf[1:-1, 0], y_wall) if len(interfaces) == 1 else None

    # ---- geometry check (single phase only) -------------------------------- #
    geom = None
    if len(spec["nus"]) == 1:
        h_eff = A.fit_effective_height(y_wall, prof, g, float(spec["nus"][0]))
        geom = {"h_eff_fit": h_eff, "h_target": ch.height,
                "dev": h_eff - ch.height,
                "rel_dev": abs(h_eff - ch.height) / ch.height}

    # ---- dimensionless numbers -------------------------------------------- #
    ma = float(spec["u_peak"]) * np.sqrt(3.0)
    re_layers = [float(spec["u_peak"]) * ch.height / nu for nu in spec["nus"]]

    t_conv = None
    for row in trace:
        if row["eps_inf"] < 1e-3:
            t_conv = row["step"]
            break

    record = {
        "arm": spec["label"],
        "family": spec["family"],
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in spec.items()},
        "geometry": {"height": ch.height, "ny": ch.ny, "nx": ch.nx,
                     "interfaces": interfaces, "wall_at": [0.5, ch.ny - 1.5],
                     "check": geom},
        "dimensionless": {
            "u_peak": float(spec["u_peak"]), "g": g,
            "ma": ma, "ma_limit": 0.1, "cs": float(np.sqrt(1.0 / 3.0)),
            "re_height_per_layer": re_layers,
            "viscosity_ratio": (max(spec["nus"]) / min(spec["nus"])),
            "tau_per_layer": [0.5 + 3.0 * nu for nu in spec["nus"]],
            "interface_frac": list(spec["interface_fracs"]),
        },
        "analytic": {
            "u_peak": ref.u_peak(), "y_peak": ref.y_peak(),
            "q_layers": q_ana_cont, "q_total": ref.flow_rate(),
            "q_layers_sameFunctional": q_ana,
            "q_reduction_bias": [rel_err(a, b) for a, b in zip(q_ana, q_ana_cont)],
            "u_mean": ref.mean_velocity(),
            "stress_continuity_residual": max(
                [abs(float(ref.shear_stress(np.array([i - 1e-9]))[0])
                     - float(ref.shear_stress(np.array([i + 1e-9]))[0])) for i in interfaces]
                or [0.0]),
        },
        "numerics": {
            **errs,
            "u_peak_num": pk["u_peak_num"], "y_peak_num": pk["y_peak_num"],
            "u_peak_err": rel_err(pk["u_peak_num"], ref.u_peak()),
            "y_peak_err_abs": abs(pk["y_peak_num"] - ref.y_peak()),
            "q_layers_num": q_num,
            "q_layers_err": [rel_err(a, b) for a, b in zip(q_num, q_ana)],
            "q_total_num": float(sum(q_num)),
            "q_total_err": rel_err(sum(q_num), sum(q_ana)),
            "q_total_err_continuum": rel_err(sum(q_num), ref.flow_rate()),
            "mass_total": float(rho[1:-1].sum()),
        },
        "globals_tclb_port": glob.as_dict(),
        "interface_tracking": {"tracked_y": iface_tracked,
                               "imposed_y": interfaces[0] if len(interfaces) == 1 else None},
        "run": {**info, "wall_seconds": wall, "t_conv_1e3": t_conv,
                "trace_every": cfg.sample_every, "domain_x": cfg.x_boundary},
        "profile": {"y_wall": y_wall.tolist(), "u_num": prof.tolist(), "u_exact": u_exact.tolist()},
        "trace": trace,
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "platform": platform.platform(), "cpu_count": os.cpu_count()},
    }
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"arm_{spec['label']}.json")
    with open(path, "w") as fh:
        json.dump(record, fh, indent=1)
    record["_path"] = path
    print(f"[{spec['label']}] eps_inf={errs['eps_inf']:.3e} eps_l2={errs['eps_l2_rel']:.3e} "
          f"q_err={record['numerics']['q_total_err']:.3e} steps={info['steps']:.0f} "
          f"wall={wall:.1f}s -> {path}", flush=True)
    return record


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", default="artifacts/xval_layered_poiseuille_analytic_inlet")
    args = ap.parse_args(argv)
    spec = cases.arm_by_label(args.arm)
    cap = int(os.environ.get("LP_STEP_CAP", "0") or 0)     # optional smoke-run cap
    if cap:
        spec = {**spec, "steps": min(int(spec["steps"]), cap)}
    run_arm(spec, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())