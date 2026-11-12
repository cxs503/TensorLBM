#!/usr/bin/env python3
"""CASE-MCMP-CAPILLARY-RISE — Jurin capillary rise on the CAC phase-field solver.

Why this driver exists
----------------------
Route A/B of this case pool (``run_case.py`` / ``sc_closed_box.py``) are
Shan-Chen pseudopotential runs: their wall wetting is a *pseudo-density*
``psi_w`` whose effective angle is not under control (measured 88 deg for a
wall declared at 35 deg), so the capillary drive they build cannot be compared
with Jurin's law at all.  This driver uses the library's Young wetting boundary
condition instead -- the conservative Allen-Cahn (CAC) solver
``tensorlbm.cac_lbm2d`` with a **prescribed contact angle** -- and reuses the
measurement operators of the wetting ladder ``tensorlbm.wetting``.

Reference (evaluated from the simulated inputs, never transcribed)::

    h = 2 sigma cos(theta) / (drho g w)          Jurin (1718) Phil. Trans. 30, 739
                                                 de Gennes, Brochard-Wyart & Quere,
                                                 Capillarity and Wetting Phenomena, ch. 1-2
                                                 (plate channel: r -> w, mean curvature
                                                  2 cos(theta)/w)

``sigma`` is *measured* in the same run (flat-interface integral, case 1 of the
wetting ladder), ``theta`` is the value prescribed to the wall boundary
condition, ``drho = rho_l - rho_g``, ``g`` the body force and ``w`` the plate
separation -- every number in the ref comes from the artifact.

Calibration leg (independent of the rise)
-----------------------------------------
1. ``sigma``: plane liquid slab, ``sigma = 0.5 * int (f + k/2 |grad phi|^2) dy``
   over the two interfaces (``ladder.sigma_flat``);
2. ``theta``: sessile drop on a horizontal plate with the same wetting wall --
   the *realised* angle is recovered by circle fit and by spherical-cap
   inversion of the diffuse profile (``ladder.contact``).  The rise run is blind
   with respect to this leg: it does not feed back into the wall condition.

Usage
-----
    PYTHONPATH=src python3 benchmarks/case_pool/capillary_rise/jurin_cac.py \
        --theta 60 --gap 20 --gravity 1e-4 --steps 20000 --out artifacts/jurin_cac_theta60.json
    PYTHONPATH=src python3 benchmarks/case_pool/capillary_rise/jurin_cac.py --quick

Artifacts land next to ``--out``; every number of the report is read back from
that JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]

GATE_PCT = 10.0          # case-pool acceptance (<10 %)
STOP_SPREAD_PCT = 2.0    # stopping rule on the last window of h


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (torch.Tensor,)):
        return _jsonable(obj.tolist())
    if isinstance(obj, bool) or obj is None or isinstance(obj, (int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    try:
        return float(obj)
    except Exception:  # pragma: no cover
        return str(obj)


def jurin_ref(sigma: float, theta_deg: float, drho: float, gravity: float, gap: float) -> float:
    """Jurin (1718) rise height for a 2-D plate channel of width ``gap``."""
    return 2.0 * sigma * math.cos(math.radians(theta_deg)) / (drho * gravity * gap)


def window_stat(traj: list[dict], key: str = "h", frac: float = 0.2) -> dict:
    """Mean / spread of the last ``frac`` of the trace (the predeclared stopping metric)."""
    n = max(2, int(len(traj) * frac))
    tail = traj[-n:]
    vals = [float(t[key]) for t in tail]
    mean = sum(vals) / len(vals)
    spread = 100.0 * (max(vals) - min(vals)) / abs(mean) if mean else float("inf")
    u_tail = [float(t["u_max"]) for t in tail]
    return {
        "mean": mean,
        "n": n,
        "min": min(vals),
        "max": max(vals),
        "spread_pct": spread,
        "u_max_tail": sum(u_tail) / len(u_tail),
        "steps": [int(t["step"]) for t in tail],
    }


def run_case(
    theta_deg: float = 60.0,
    gap: int = 20,
    gravity: float = 1.0e-4,
    sigma: float = 0.03,
    shape: tuple[int, int] = (160, 88),
    steps: int = 20000,
    sample_every: int = 200,
    mu_l: float = 0.05,
    mu_g: float = 0.005,
    calibrate: bool = True,
    contact_calib: bool = False,
    quick: bool = False,
) -> dict:
    """One capillary-rise run plus its independent calibration leg."""
    import numpy as np

    from tensorlbm import cac_lbm2d as cac
    from tensorlbm.wetting import ladder as L
    from tensorlbm.wetting import metrics as M

    if quick:
        shape, steps, sample_every = (96, 64), 2500, 250
        calibrate = calibrate and True

    t0 = time.time()
    mu_eff = 0.5 * (mu_l + mu_g)  # only reported (Ca is not a gate here)

    # ---- calibration leg -------------------------------------------------
    cal = {}
    if calibrate:
        sc = L.sigma_flat(sigma=sigma, steps=4000 if not quick else 1500)
        cal["sigma_flat"] = {k: sc[k] for k in ("sigma_meas", "ratio_to_input", "u_max", "wall_time_s")}
    if calibrate and contact_calib:
        ct = L.contact(theta_deg=theta_deg, sigma=sigma, steps=3500 if not quick else 1500,
                       shape=(112, 112) if not quick else (72, 72))
        cal["contact"] = {
            "theta_input_deg": ct["theta_input_deg"],
            "theta_circle_deg": ct["meas"]["circle_fit"].get("theta_deg"),
            "theta_cap_deg": ct["meas"]["cap_inversion"].get("theta_deg"),
            "u_max": ct["u_max"],
            "wall_time_s": ct["wall_time_s"],
        }
    sigma_meas = (cal.get("sigma_flat") or {}).get("sigma_meas") or sigma

    # ---- rise leg --------------------------------------------------------
    rr = L.jurin(theta_deg=theta_deg, sigma=sigma, shape=shape, gap=gap, pool=20,
                 gravity=gravity, steps=steps, sample_every=sample_every,
                 mu_l=mu_l, mu_g=mu_g)
    traj = rr["traj"]
    eq = window_stat(traj)
    drho = 1.0 - 0.1  # BASE of the wetting ladder: rho_l=1.0, rho_g=0.1

    h_ref = jurin_ref(sigma_meas, theta_deg, drho, gravity, gap)
    h_got = eq["mean"]
    err_pct = 100.0 * (h_got - h_ref) / h_ref if h_ref else float("inf")
    theta_circle = (cal.get("contact") or {}).get("theta_circle_deg")
    h_ref_arc = jurin_ref(sigma_meas, theta_circle, drho, gravity, gap) if theta_circle else None

    cap = gravity * drho * gap  # Bond number: drho g w^2 / sigma
    bond = drho * gravity * gap * gap / sigma_meas

    res = {
        "case": "CASE-MCMP-CAPILLARY-RISE",
        "solver": "tensorlbm.cac_lbm2d (conservative Allen-Cahn + Young wetting BC)",
        "reference": {
            "law": "h = 2 sigma cos(theta) / (drho g w)",
            "citation": "Jurin (1718) Phil. Trans. R. Soc. 30, 739-747; "
                        "de Gennes, Brochard-Wyart & Quere, Capillarity and Wetting Phenomena, ch. 2 "
                        "(plate channel: r -> w)",
            "sigma_source": "measured in-run (flat-interface integral)",
            "theta_source": "prescribed to the wall boundary condition",
        },
        "inputs": {
            "theta_deg": float(theta_deg),
            "gap_w": int(gap),
            "gravity": float(gravity),
            "sigma_input": float(sigma),
            "rho_l": 1.0,
            "rho_g": 0.1,
            "drho": drho,
            "mu_l": float(mu_l),
            "mu_g": float(mu_g),
            "interface_width": 4.0,
            "mobility": 0.05,
            "shape": list(shape),
            "steps": int(steps),
            "sample_every": int(sample_every),
            "bond_number": bond,
            "capillary_pressure": 2.0 * sigma_meas * math.cos(math.radians(theta_deg)) / gap,
            "mu_eff": mu_eff,
        },
        "calibration": cal,
        "measured": {
            "h": h_got,
            "h_window": eq,
            "u_max_allnodes": rr["u_max"],
            "sigma_meas": sigma_meas,
        },
        "ref": h_ref,
        "got": h_got,
        "error_pct": err_pct,
        "abs_error_pct": abs(err_pct),
        "gate_pct": GATE_PCT,
        "pass": bool(abs(err_pct) < GATE_PCT),
        "converged": bool(eq["spread_pct"] < STOP_SPREAD_PCT),
        "ref_secondary_measured_angle": h_ref_arc,
        "traj": traj,
        "traj_json": "traj" in rr,
        "wall_time_s": time.time() - t0,
        "env": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "threads": torch.get_num_threads(),
        },
    }
    # drop the (large) trace from the primary record but keep a decimated copy
    stride = max(1, len(traj) // 40)
    res["traj_decimated"] = traj[::stride]
    res.pop("traj", None)
    return {k: _jsonable(v) for k, v in res.items()}


def figure(res: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    tr = res["traj_decimated"]
    st = [t["step"] for t in tr]
    ax[0].plot(st, [t["y_channel"] for t in tr], label="meniscus y (channel)")
    ax[0].plot(st, [t["y_reservoir"] for t in tr], label="pool level (reservoir)")
    ax[0].set_xlabel("step")
    ax[0].set_ylabel("y [lattice]")
    ax[0].legend(fontsize=7)
    ax[0].set_title(f"levels, theta={res['inputs']['theta_deg']:.0f} deg")

    ax[1].plot(st, [t["h"] for t in tr], "C2", label="h(t) = meniscus - pool")
    ax[1].axhline(res["ref"], color="k", ls=":", label=f"Jurin ref = {res['ref']:.3f}")
    ax[1].axhline(res["got"], color="C3", ls="--", label=f"measured = {res['got']:.3f}")
    ax[1].set_xlabel("step")
    ax[1].set_ylabel("h [lattice]")
    ax[1].set_title(f"capillary rise, err = {res['error_pct']:+.1f} %")
    ax[1].legend(fontsize=7)

    ax[2].plot(st, [t["u_max"] for t in tr], "C4")
    ax[2].set_xlabel("step")
    ax[2].set_ylabel("max |u| (all fluid)")
    ax[2].set_title("spurious-current floor")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--theta", type=float, default=60.0, help="prescribed Young contact angle [deg]")
    ap.add_argument("--gap", type=int, default=20, help="plate separation w [lattice]")
    ap.add_argument("--gravity", type=float, default=1.0e-4, help="body force g [lattice]")
    ap.add_argument("--sigma", type=float, default=0.03, help="free-energy surface tension parameter")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--sample-every", type=int, default=200)
    ap.add_argument("--ny", type=int, default=160)
    ap.add_argument("--nx", type=int, default=88)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--with-contact", action="store_true",
                    help="also run the 112x112 sessile-drop angle calibration (~2 min)")
    ap.add_argument("--no-calibration", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=Path, default=REPO / "benchmarks/case_pool/capillary_rise/artifacts/jurin_cac.json")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    res = run_case(theta_deg=args.theta, gap=args.gap, gravity=args.gravity, sigma=args.sigma,
                   shape=(args.ny, args.nx), steps=args.steps, sample_every=args.sample_every,
                   calibrate=not args.no_calibration, contact_calib=args.with_contact,
                   quick=args.quick)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    png = args.out.with_suffix(".png")
    if not args.no_figure:
        figure(res, png)
    print(f"[jurin] theta={res['inputs']['theta_deg']:.0f} deg gap={res['inputs']['gap_w']} "
          f"sigma_meas={res['measured']['sigma_meas']:.5f} "
          f"h_ref={res['ref']:.3f} h_got={res['got']:.3f} err={res['error_pct']:+.2f}% "
          f"spread={res['measured']['h_window']['spread_pct']:.2f}% "
          f"pass={res['pass']} -> {args.out}")
    return 0 if res["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())