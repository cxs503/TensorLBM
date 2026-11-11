"""Time-resolved spurious-current diagnostic for the stratified MCMP channel.

Why a separate driver
---------------------
The acceptance metric of the GAP-LBM-MCMP-POISEUILLE ledger is the *interface
spurious current* normalised by the driven-flow velocity scale,

    ratio(t) = S(t) / u_max,mix(t),      S(t) = max |u_y,mix| over fluid rows,

with the gate ``ratio < 0.005``.  Two effects enter ``S`` and they must not be
conflated:

1. **transient drift** — the diffuse interface is still relaxing towards its
   equilibrium position (wall wetting, initial tanh width, acoustic settling).
   This produces a *bulk* ``u_y`` that decays in time.
2. **parasitic current** — the steady residual of the Shan-Chen force
   discretisation at the interface.  It does not decay and is (for a flat
   interface) confined to the interface band.

The diagnostic therefore records three quantities at every sample:

* ``S_all``  — max |u_y,mix| over all fluid rows (what the runner reports);
* ``S_ifc``  — max |u_y,mix| inside the interface band ``|alpha-0.5| < 0.25``;
* ``S_bulk`` — max |u_y,mix| outside that band (pure transient drift signal).

It also runs the **static control** (``--gx 0``): the same stratification, same
wall treatment, no body force.  Its late-time plateau is the parasitic-current
floor of the model at that resolution — the quantity that criterion 3 actually
asks about — while its decay curve gives the transient time scale.

Writes one JSON per run with the full time series (records are large; see
``docs/MCMP_SPURIOUS_CURRENT_R2.md`` for the analysis and the verdict).

Usage::

    PYTHONPATH=src python cases/mcmp_stratified_poiseuille/spurious_diagnostic.py \
        --ny 25 --steps 40000 --gx 0 --wall-mode mirror --out artifacts/.../sp_gx0_ny25.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402

from tensorlbm.boundary.boundaries import bounce_back_cells  # noqa: E402
from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.multiphase import collide_sc_two_component  # noqa: E402
from tensorlbm.solver import stream  # noqa: E402

IFC_BAND = 0.25


def _init(ny, nx, half, rho_heavy, rho_light, width):
    ys = torch.arange(ny, dtype=torch.float32).view(ny, 1).expand(ny, nx)
    alpha = 0.5 * (1.0 - torch.tanh((ys - float(half)) / width))
    rho1 = rho_heavy * alpha + rho_light * (1.0 - alpha)
    rho2 = rho_light * alpha + rho_heavy * (1.0 - alpha)
    z = torch.zeros((ny, nx))
    return equilibrium(rho1, z, z), equilibrium(rho2, z, z)


def _crossing(alpha_profile) -> float:
    a = alpha_profile.tolist()
    for j in range(len(a) - 1):
        if (a[j] - 0.5) * (a[j + 1] - 0.5) <= 0 and a[j] != a[j + 1]:
            return j + (0.5 - a[j]) / (a[j + 1] - a[j])
    return float("nan")


def run(
    ny: int = 25,
    nx: int = 4,
    half: int | None = None,
    tau_lower: float = 1.0,
    tau_upper: float = 0.75,
    gx: float = 5e-5,
    g12: float = -2.5,
    steps: int = 20000,
    u_eq: str = "viscous",
    force_scheme: str = "guo",
    wall_mode: str = "mirror",
    sc_form: str = "standard",
    rho_heavy: float = 0.7,
    rho_light: float = 0.3,
    width: float = 3.0,
    sample_every: int = 200,
) -> dict:
    half = ny // 2 if half is None else half
    wall = torch.zeros((ny, nx), dtype=torch.bool)
    wall[0, :] = True
    wall[-1, :] = True
    f1, f2 = _init(ny, nx, half, rho_heavy, rho_light, width)
    r1_0, _, _ = macroscopic(f1)
    r2_0, _, _ = macroscopic(f2)
    mass0 = (float(r1_0.sum()), float(r2_0.sum()))

    series = []
    t0 = time.time()
    for step in range(1, steps + 1):
        f1, f2 = collide_sc_two_component(
            f1, f2, G_12=g12, tau1=tau_lower, tau2=tau_upper, gx=gx,
            solid_mask=wall, u_eq=u_eq, force_scheme=force_scheme,
            wall_mode=wall_mode, sc_form=sc_form,
        )
        f1, f2 = stream(f1), stream(f2)
        f1, f2 = bounce_back_cells(f1, wall), bounce_back_cells(f2, wall)
        if step % sample_every and step != steps:
            continue
        r1, ux1, uy1 = macroscopic(f1)
        r2, ux2, uy2 = macroscopic(f2)
        rho_tot = torch.clamp(r1 + r2, min=1e-12)
        ux = (r1 * ux1 + r2 * ux2) / rho_tot
        uy = (r1 * uy1 + r2 * uy2) / rho_tot
        alpha = ((r1 - rho_light) / (rho_heavy - rho_light)).mean(dim=1)
        uy_f = uy[~wall]
        ux_f = ux[~wall]
        a_f = alpha[1:-1]
        band = (a_f - 0.5).abs() < IFC_BAND
        uy_band = uy[1:-1][band]
        uy_bulk = uy[1:-1][~band]
        series.append(dict(
            step=step,
            S_all=float(uy_f.abs().max()) if uy_f.numel() else 0.0,
            S_ifc=float(uy_band.abs().max()) if uy_band.numel() else 0.0,
            S_bulk=float(uy_bulk.abs().max()) if uy_bulk.numel() else 0.0,
            u_max_mix=float(ux_f.abs().max()) if ux_f.numel() else 0.0,
            y_int=_crossing(alpha.clamp(0.0, 1.0)),
            phi_contrast=float(a_f.max() - a_f.min()),
            mass_drift_1=float(r1.sum()) / mass0[0] - 1.0,
            mass_drift_2=float(r2.sum()) / mass0[1] - 1.0,
            finite=bool(torch.isfinite(f1).all() and torch.isfinite(f2).all()),
        ))
        if not series[-1]["finite"]:
            break

    rec = dict(
        config=dict(ny=ny, nx=nx, half=half, tau_lower=tau_lower, tau_upper=tau_upper,
                    gx=gx, g12=g12, steps=steps, u_eq=u_eq, force_scheme=force_scheme,
                    wall_mode=wall_mode, sc_form=sc_form,
                    rho_heavy=rho_heavy, rho_light=rho_light,
                    interface_width=width, sample_every=sample_every),
        wall_time_s=time.time() - t0,
        runtime=dict(python=sys.version.split()[0], torch=torch.__version__),
        series=series,
    )
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ny", type=int, default=25)
    ap.add_argument("--nx", type=int, default=4)
    ap.add_argument("--half", type=int, default=None)
    ap.add_argument("--tau-lower", type=float, default=1.0)
    ap.add_argument("--tau-upper", type=float, default=0.75)
    ap.add_argument("--gx", type=float, default=5e-5)
    ap.add_argument("--g12", type=float, default=-2.5)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--u-eq", default="viscous")
    ap.add_argument("--force-scheme", default="guo")
    ap.add_argument("--wall-mode", default="mirror", choices=["dry", "mirror"])
    ap.add_argument("--sc-form", default="standard", choices=["standard", "edm"])
    ap.add_argument("--sample-every", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    rec = run(ny=args.ny, nx=args.nx, half=args.half, tau_lower=args.tau_lower,
              tau_upper=args.tau_upper, gx=args.gx, g12=args.g12, steps=args.steps,
              u_eq=args.u_eq, force_scheme=args.force_scheme,
              wall_mode=args.wall_mode, sc_form=args.sc_form,
              sample_every=args.sample_every)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(rec, fh, indent=1)
    s = rec["series"]
    tail = s[-min(5, len(s)):]
    print("SUMMARY " + json.dumps(dict(
        ny=rec["config"]["ny"], gx=rec["config"]["gx"], wall_mode=rec["config"]["wall_mode"],
        sc_form=rec["config"]["sc_form"],
        steps=s[-1]["step"] if s else 0, wall_time_s=round(rec["wall_time_s"], 1),
        S_all=tail[-1]["S_all"], S_ifc=tail[-1]["S_ifc"], S_bulk=tail[-1]["S_bulk"],
        u_max_mix=tail[-1]["u_max_mix"],
        ratio_all=tail[-1]["S_all"] / tail[-1]["u_max_mix"] if tail[-1]["u_max_mix"] else float("inf"),
        y_int=tail[-1]["y_int"], phi_contrast=tail[-1]["phi_contrast"],
        steady=bool(len(tail) >= 3 and max(abs(t["S_all"] - tail[0]["S_all"]) for t in tail)
                    <= 1e-3 * max(abs(tail[0]["S_all"]), 1e-30)),
    )))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())