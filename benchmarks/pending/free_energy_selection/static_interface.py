"""Round-5 (R4 follow-up): free-energy binary model, static flat-interface control.

Anchor: ``XVAL-LBM-R4-COEXISTENCE-CALIBRATION-AND-FREE-ENER``.

Round 4 (``docs/MCMP_STRATIFIED_POISEUILLE_R4.md``) measured the *pseudopotential*
(Shan-Chen, two component) static control at the model's own coexistence state and
refuted the hypothesis that criterion ③ (``S_ifc < 0.005 u_max``) is reachable inside
that model class: the plateau ``S_ifc = 8.6e-3`` (ratio 0.0767 vs gate 0.005) is not
interface-localised at all -- ``S_bulk = S_all`` in both pair-force arms.  The round
therefore *selected* the free-energy binary model (Swift/Orlandini/Osborn/Yeomans
PRL 78, 19 (1996); sailfish semantics in
``src/tensorlbm/phasefield/free_energy_pressure.py``) as the migration target, but it
never measured the free-energy arm under the same metric.

This module supplies that missing measurement, and nothing else:

* a **quiescent flat interface** in a fully periodic strip (no walls, no body force),
  initialised with the analytic profile ``phi = tanh(s/xi)`` where ``s`` is the signed
  distance to the band edge.  The strip carries a ``band``-cell layer of ``phi = +1``
  in the middle, so both interfaces are smooth and the periodicity of the box is
  respected (a single monotone ``tanh`` across a periodic box would impose an
  infinitely sharp seam at the wrap-around -- that artefact alone diverges the run);
* the **same** interface-band definition as R4 (``|a - 1/2| < 1/4`` with the normalised
  order parameter ``a = (phi + 1)/2``, the free-energy analogue of R4's
  ``alpha = (rho1 - rho2)/(rho1 + rho2)``), reporting ``S_ifc`` (band max ``|u_y|``),
  ``S_all`` (global max ``|u|``) and ``S_bulk`` (max ``|u|`` outside the band);
* the **same** interface-width reduction as R4, ``w_10_90`` from the normalised profile
  and ``w_tanh = w_10_90 / (2 atanh 0.8)``, measured on a monotone window around one
  interface, so the free-energy arm can run at the *matched* width of the
  coexistence-calibrated Shan-Chen interface (``w_10_90 = 4.645`` cells);
* the flat-interface surface tension ``sigma = integral (P_NN - P_TT)`` per interface
  from the measured pressure tensor, as the analytic-tension sanity check.

The analytic anchors ``xi = sqrt(2 kappa / A)``, ``w_10_90 = 2 atanh(0.8) xi`` and
``sigma = (2/3) sqrt(2 kappa A)`` are never used as results: they are compared against
the measured fields (prerequisites (a)/(b) of the round-4 migration plan).

Usage::

    PYTHONPATH=src:. python benchmarks/pending/free_energy_selection/static_interface.py \
        --arm matched --ny 64 --steps 8000 \
        --out artifacts/free_energy_selection/fe_static_matched.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

import torch  # noqa: E402

from tensorlbm.phasefield.free_energy_pressure import (  # noqa: E402
    FreeEnergyBinaryParams,
    collide_and_stream,
    initial_condition,
    macroscopic,
    mean_pressure,
    pressure_tensor_measured,
    tanh_profile,
)

#: 10-90 thickness -> tanh width (identical convention to the Shan-Chen arm).
TANH_10_90 = 2.0 * math.atanh(0.8)

#: Interface band of criterion ③ in the normalised order parameter ``a in [0, 1]``.
BAND = 0.25

#: Reference operating point of the two-layer Poiseuille anchor (rounds 1-4).
G_X_REF = 5e-5
H_REF = 47.0          # wall spacing of the ny = 49 probe
TAU_F_REF = 1.0       # -> nu = cs2 (tau_f - 1/2) = 1/6, close to the heavy Shan-Chen layer

#: 10-90 width of the coexistence-calibrated Shan-Chen interface (round 4).
W_SC_COEXISTENCE = 4.645


def _params(arm: str, tau_phi: float, gamma: float) -> FreeEnergyBinaryParams:
    """Free-energy parameters per arm (``A = 0.1`` well depth everywhere)."""
    if arm == "standard":
        # production default of the pressure-tensor kernel: xi = 2, sigma = 2/15
        kappa = 0.2
    elif arm == "matched":
        # xi chosen so that w_10_90 equals the Shan-Chen coexistence width (4.645 cells)
        xi_target = W_SC_COEXISTENCE / TANH_10_90
        kappa = 0.5 * 0.1 * xi_target ** 2
    else:  # pragma: no cover - argparse choices guard this
        raise ValueError(f"unknown arm {arm!r}")
    return FreeEnergyBinaryParams(A=0.1, kappa=kappa, Gamma=gamma,
                                  tau_f=TAU_F_REF, tau_phi=tau_phi)


def _crossing(profile, level: float) -> float:
    """Interpolated index where a monotone profile crosses ``level`` (NaN if none)."""
    for j in range(len(profile) - 1):
        lo, hi = float(profile[j]), float(profile[j + 1])
        if (lo - level) * (hi - level) <= 0.0 and lo != hi:
            return j + (lo - level) / (lo - hi)
    return float("nan")


def _width_10_90(a_row: torch.Tensor, edge: float, span: int = 10) -> tuple[float, float]:
    """10-90 width and interpolated ``a = 1/2`` position on a window around ``edge``.

    ``a_row`` is the row-averaged normalised order parameter; ``edge`` is the analytic
    position of one band edge, so the window ``[edge - span, edge + span]`` is monotone
    (decreasing from inside the band to outside it).
    """
    ny = a_row.shape[0]
    lo = max(0, int(math.floor(edge - span)))
    hi = min(ny - 1, int(math.ceil(edge + span)))
    vals = a_row[lo:hi + 1].tolist()
    if len(vals) < 3:
        return float("nan"), float("nan")
    def cross(level):
        return _crossing(vals, level)
    y90, y10, y50 = cross(0.9), cross(0.1), cross(0.5)
    w = abs(y10 - y90) if (y90 == y90 and y10 == y10) else float("nan")
    return w, (lo + y50 if y50 == y50 else float("nan"))


def run_static_interface(
    arm: str = "matched",
    ny: int = 64,
    nx: int = 4,
    band: int = 32,
    steps: int = 8000,
    tau_phi: float = 2.0,
    gamma: float = 0.05,
    rho0: float = 1.0,
    sample_every: int = 100,
    dtype: torch.dtype = torch.float64,
) -> dict:
    """Run the quiescent flat interface and return the measured record.

    Periodicity is used in *both* directions.  The layer of ``phi = +1`` of width
    ``band`` sits in the middle of the ``ny`` rows, so the two interfaces are at
    ``ny/2 -/+ band/2`` and are ``band`` cells apart (>> the interface thickness).
    No walls and no forcing: any residual velocity is spurious.
    """
    p = _params(arm, tau_phi, gamma)
    xi = p.xi
    ys = torch.arange(ny, dtype=dtype).view(ny, 1).expand(ny, nx)
    signed = (band / 2.0) - (ys - ny / 2.0).abs()
    phi0 = tanh_profile(signed, 0.0, xi).contiguous()
    f, g = initial_condition(phi0, p, rho0=rho0)
    phi_mass0 = float(phi0.sum())
    rho_mass0 = float(f.sum())
    edge_hi = ny / 2.0 + band / 2.0

    series = []
    t0 = time.time()
    for step in range(1, steps + 1):
        f, g = collide_and_stream(f, g, p)
        if step % sample_every and step != steps:
            continue
        rho, ux, uy, phi = macroscopic(f, g)
        finite = bool(torch.isfinite(f).all() and torch.isfinite(g).all())
        speed = torch.sqrt(ux * ux + uy * uy)
        a = 0.5 * (phi + 1.0)
        in_band = (a - 0.5).abs() < BAND
        a_row = a.mean(dim=1)
        w90, y50 = _width_10_90(a_row, edge_hi)
        Pxx, Pyy, _ = pressure_tensor_measured(f)
        # normal direction = y; two interfaces -> half the integral
        sigma_band = 0.5 * float((Pyy - Pxx).mean(dim=1).sum())
        series.append(dict(
            step=step,
            S_ifc=float(uy[in_band].abs().max()) if bool(in_band.any()) else float("nan"),
            S_all=float(speed.max()),
            S_bulk=float(speed[~in_band].max()) if bool((~in_band).any()) else 0.0,
            y_int=y50,
            w_10_90=w90,
            sigma_band=sigma_band,
            phi_max=float(phi.max()),
            phi_min=float(phi.min()),
            rho_max=float(rho.max()),
            rho_min=float(rho.min()),
            p_mean=float(mean_pressure(f).mean()),
            dphi=float((phi.sum() - phi_mass0) / (ny * nx)),
            drho=float((f.sum() - rho_mass0) / rho_mass0),
            finite=finite,
        ))
        if not finite:
            break

    last = series[-1]
    tail = [h for h in series if h["step"] >= 0.75 * last["step"]]

    def _mean(key):
        vals = [h[key] for h in tail if h[key] == h[key]]
        return sum(vals) / len(vals) if vals else float("nan")

    w_meas = _mean("w_10_90")
    xi_meas = w_meas / TANH_10_90
    sifc = [h["S_ifc"] for h in tail if h["S_ifc"] == h["S_ifc"]]
    # log-decade rate of the residual over the whole run (diagnostic: transient or plateau)
    sf = [h for h in series if h["S_ifc"] == h["S_ifc"] and h["S_ifc"] > 0]
    decay = float("nan")
    if len(sf) >= 2:
        s0, s1 = sf[0], sf[-1]
        decay = -math.log10(s1["S_ifc"] / s0["S_ifc"]) / (s1["step"] - s0["step"]) * 1000.0
    summary = dict(
        arm=arm,
        A=p.A, kappa=p.kappa, Gamma=p.Gamma, tau_f=p.tau_f, tau_phi=p.tau_phi,
        nu=p.viscosity, mobility=p.mobility,
        xi_analytic=xi, xi_measured=xi_meas,
        xi_rel_err=(xi_meas - xi) / xi if xi_meas == xi_meas else float("nan"),
        w_10_90_analytic=TANH_10_90 * xi, w_10_90_measured=w_meas,
        w_rel_err=(w_meas - TANH_10_90 * xi) / (TANH_10_90 * xi) if w_meas == w_meas else float("nan"),
        sigma_analytic=p.sigma, sigma_band=_mean("sigma_band"),
        sigma_rel_err=((_mean("sigma_band") - p.sigma) / p.sigma
                       if _mean("sigma_band") == _mean("sigma_band") else float("nan")),
        S_ifc=last["S_ifc"],
        S_ifc_mean=sum(sifc) / len(sifc) if sifc else float("nan"),
        S_ifc_tail_max=max(sifc) if sifc else float("nan"),
        S_ifc_tail_min=min(sifc) if sifc else float("nan"),
        S_all=last["S_all"], S_bulk=last["S_bulk"],
        S_ifc_decades_per_1k=decay,
        y_int=last["y_int"],
        y_drift=(max(h["y_int"] for h in tail if h["y_int"] == h["y_int"])
                 - min(h["y_int"] for h in tail if h["y_int"] == h["y_int"]))
        if any(h["y_int"] == h["y_int"] for h in tail) else float("nan"),
        phi_max=last["phi_max"], rho_min=last["rho_min"], rho_max=last["rho_max"],
        dphi_per_cell=last["dphi"], drho=last["drho"],
        ny=ny, nx=nx, band=band, steps=steps, sample_every=sample_every,
        dtype=str(dtype).replace("torch.", ""),
        wall_time_s=time.time() - t0,
        finite=last["finite"],
    )
    return dict(params=summary, series=series)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="matched", choices=["standard", "matched"])
    ap.add_argument("--ny", type=int, default=64)
    ap.add_argument("--nx", type=int, default=4)
    ap.add_argument("--band", type=int, default=32)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--tau-phi", type=float, default=2.0)
    ap.add_argument("--gamma", type=float, default=0.05)
    ap.add_argument("--sample-every", type=int, default=100)
    ap.add_argument("--float32", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    rec = run_static_interface(
        arm=args.arm, ny=args.ny, nx=args.nx, band=args.band, steps=args.steps,
        tau_phi=args.tau_phi, gamma=args.gamma, sample_every=args.sample_every,
        dtype=torch.float32 if args.float32 else torch.float64,
    )
    s = rec["params"]
    print("SUMMARY " + json.dumps(s))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(rec, open(args.out, "w"), indent=1)
        print("wrote " + args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())