#!/usr/bin/env python3
"""XVAL-LBM-HORIZONTAL-SURVEY-EMPTY — inertia coefficient (added mass) of a
circular cylinder in oscillatory flow, measured in TensorLBM.

Why this case: the horizontal OSS survey (`survey.py`, `survey_summary.json`)
returns *zero* census rows, *zero* ledger rows and *zero* source hits for the
oscillatory-flow inertia-coefficient / added-mass / PMM family, so there is no
third-party implementation to transplant code or parameter conventions from.
The supply therefore has to come from the literature anchor instead:

    Morison et al. (1950); Sarpkaya (1986) -- for a circular cylinder in an
    unbounded ideal fluid the inertia coefficient is exactly C_M = 1 + C_a = 2
    (added-mass coefficient C_a = 1, i.e. m_a = rho * pi * D^2 / 4 per unit
    depth).  KC = U_max * T / D and beta = D^2 / (nu * T) are the two
    similarity numbers; the ideal-fluid limit is beta -> infinity, KC -> 0.

Numerics (all predeclared in ``predeclared.json``):

  * D2Q9, doubly periodic box (no wall blockage), uniform harmonic body force
    ``a(t) = a0 cos(omega t)`` applied to fluid nodes only, via the library's
    second-order Guo source ``tensorlbm.boussinesq.apply_guo_forcing``.
  * Fixed circular obstacle, halfway bounce-back (``boundary.bounce_back_cells``)
    applied after streaming and after the force sample, so the Ladd
    momentum-exchange sample ``boundary.compute_obstacle_forces`` is taken
    post-stream / pre-bounce-back (the repo-recorded correct ordering).
  * Two *independent* force estimators:
      E1 Ladd momentum exchange on the solid nodes;
      E2 fluid momentum budget  F_cyl = sum_fluid a(t) - dP/dt.
  * Morison decomposition F = C_M rho V dU/dt + 0.5 C_D rho D U|U| by linear
    least squares over the measurement window, with U(t) taken from the
    *measured* box-mean velocity (2-harmonic fit) so no far-field velocity is
    assumed.

Usage:
  python xval/horizontal_survey_empty/run_arm.py --case base
  python xval/horizontal_survey_empty/run_arm.py --case base --shard 0 --shards 4
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tensorlbm.boundary.boundaries import bounce_back_cells, compute_obstacle_forces  # noqa: E402
from tensorlbm.boussinesq import apply_guo_forcing  # noqa: E402
from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.solver import stream  # noqa: E402


# --------------------------------------------------------------------------- #
# predeclared case matrix
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Case:
    name: str
    D: int = 24          # obstacle reference diameter, lattice cells
    kc: float = 1.0      # Keulegan-Carpenter number (nominal, from a0/omega)
    beta: float = 100.0  # Stokes number D^2/(nu*T)
    u0: float = 0.04     # far-field velocity amplitude (nominal)
    box: float = 12.0    # box side in D (doubly periodic, so this is a period)
    transient: float = 1.5
    measure: float = 3.0


CASES: dict[str, Case] = {
    # resolution ladder (convergence): D = 16 / 24 / 40 at KC=1, beta=100
    "res16": Case("res16", D=16),
    "base": Case("base", D=24),
    "res32": Case("res32", D=32),
    # Keulegan-Carpenter ladder at D=24, beta=100
    "kc05": Case("kc05", D=24, kc=0.5),
    "kc20": Case("kc20", D=24, kc=2.0),
    # Stokes-number ladder at D=24, KC=1
    "beta50": Case("beta50", D=24, beta=50.0),
    "beta200": Case("beta200", D=24, beta=200.0),
    # box-period ladder (periodic-image sensitivity) at D=24, KC=1, beta=100
    "box16": Case("box16", D=24, box=16.0),
}
CASE_ORDER = ["base", "res16", "res32", "kc05", "kc20", "beta50", "beta200", "box16"]


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def circle_mask(nx: int, ny: int, D: int, device) -> tuple[torch.Tensor, dict]:
    """Solid mask for a circle of *reference* diameter D centred on the box.

    Returns the mask plus the three geometric readings used in the report:
    the input diameter, the area-equivalent diameter and the halfway-bounce-back
    wetted diameter (solid cells inflated by half a cell, i.e. area A + P/2).
    """
    y, x = torch.meshgrid(
        torch.arange(ny, device=device), torch.arange(nx, device=device), indexing="ij"
    )
    dx = x - (nx - 1) / 2.0
    dy = y - (ny - 1) / 2.0
    mask = dx.square() + dy.square() <= (D / 2.0) ** 2
    area = int(mask.sum())
    # perimeter: solid cells with at least one 4-neighbour fluid cell
    fl = torch.zeros_like(mask)
    fl[1:, :] |= mask[:-1, :]
    fl[:-1, :] |= mask[1:, :]
    fl[:, 1:] |= mask[:, :-1]
    fl[:, :-1] |= mask[:, 1:]
    fl &= ~mask
    perim = int(fl.sum())
    geo = {
        "D_input": float(D),
        "n_solid": area,
        "D_area": 2.0 * math.sqrt(area / math.pi),
        "perimeter_cells": perim,
        "D_halfway": 2.0 * math.sqrt((area + perim / 2.0) / math.pi),
    }
    return mask, geo


# --------------------------------------------------------------------------- #
# harmonic analysis
# --------------------------------------------------------------------------- #
def harmonic_fit(y: np.ndarray, omega: float, t: np.ndarray, nh: int = 2) -> dict:
    """Least-squares fit y ~ a0 + sum_k [a_k cos(k w t) + b_k sin(k w t)]."""
    cols = [np.ones_like(t)]
    for k in range(1, nh + 1):
        cols += [np.cos(k * omega * t), np.sin(k * omega * t)]
    A = np.stack(cols, axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    fit = A @ coef
    return {
        "coef": coef,
        "fit": fit,
        "u_cos": float(coef[1]),
        "u_sin": float(coef[2]),
        "amp": float(math.hypot(coef[1], coef[2])),
        "phase_rad": float(math.atan2(-coef[2], coef[1])),
        "resid_rms": float(np.sqrt(np.mean((y - fit) ** 2))),
        "resid_rel": float(np.sqrt(np.mean((y - fit) ** 2)) / (np.std(y) + 1e-30)),
    }


def morison_ls(F: np.ndarray, U: np.ndarray, dU: np.ndarray, D_h: float, rho: float = 1.0) -> dict:
    """Least-squares Morison decomposition -> (C_M, C_D) with 2x2 normal equations."""
    V = math.pi * D_h ** 2 / 4.0
    b1 = rho * V * dU
    b2 = 0.5 * rho * D_h * U * np.abs(U)
    B = np.stack([b1, b2], axis=1)
    G = B.T @ B
    rhs = B.T @ F
    try:
        cm, cd = np.linalg.solve(G, rhs)
    except np.linalg.LinAlgError:  # pragma: no cover
        cm, cd = float("nan"), float("nan")
    pred = B @ np.asarray([cm, cd])
    return {
        "C_M": float(cm),
        "C_D": float(cd),
        "fit_r2": float(1.0 - np.sum((F - pred) ** 2) / (np.sum((F - F.mean()) ** 2) + 1e-30)),
        "cond": float(np.linalg.cond(G)),
        "inertia_share": float(np.sum(b1 * b1) / (np.sum(B * B) + 1e-30)),
    }


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run(case: Case, threads: int = 8) -> dict:
    torch.set_num_threads(max(1, threads))
    torch.set_default_dtype(torch.float64)
    device = torch.device("cpu")
    D = case.D
    nx = ny = int(round(case.box * D))
    T = case.kc * D / case.u0           # period (steps), KC = U_max*T/D
    omega = 2.0 * math.pi / T
    a0 = omega * case.u0                # body-force amplitude for U_max = u0
    nu = D ** 2 / (case.beta * T)
    tau = 0.5 + 3.0 * nu
    if tau <= 0.505:
        raise ValueError(f"tau={tau:.4f} too close to 0.5 for a stable BGK run")

    mask, geo = circle_mask(nx, ny, D, device)
    fluid = ~mask
    n_fluid = int(fluid.sum())
    n_steps = int(round((case.transient + case.measure) * T))

    # interior solid nodes: solid cells with no 4-neighbour fluid cell.  They
    # must be held at zero populations, otherwise they accumulate junk that
    # contaminates the bounce-back flux (see docs section 5, measured).
    dil = torch.zeros_like(mask)
    dil[1:, :] |= fluid[:-1, :]
    dil[:-1, :] |= fluid[1:, :]
    dil[:, 1:] |= fluid[:, :-1]
    dil[:, :-1] |= fluid[:, 1:]
    interior = mask & ~dil

    rho0 = torch.ones((ny, nx), device=device)
    f = equilibrium(rho0, torch.zeros_like(rho0), torch.zeros_like(rho0))
    f = bounce_back_cells(f, mask)

    Fx_field = torch.zeros((ny, nx), dtype=torch.float64, device=device)
    Fy_field = torch.zeros_like(Fx_field)

    ts, us, fs_ladd, p_mom, a_ext = [], [], [], [], []
    t0 = time.time()
    for step in range(1, n_steps + 1):
        t = float(step)
        a_t = a0 * math.cos(omega * t)
        Fx_field[fluid] = a_t
        old_solid = f[:, mask].clone()   # keep the scheduled wall populations through collision
        f = apply_guo_forcing(f, torch.stack([Fx_field, Fy_field]), tau)
        f[:, mask] = old_solid
        f = stream(f)
        fx_ladd, _fy = compute_obstacle_forces(f, mask)
        f = bounce_back_cells(f, mask)
        f[:, interior] = 0.0

        rho, ux, uy = macroscopic(f)
        m = rho * fluid
        us.append(float((m * ux).sum() / (m.sum() + 1e-30)))
        p_mom.append(float((m * ux).sum()))
        fs_ladd.append(float(fx_ladd))
        a_ext.append(a_t * float(m.sum()))
        ts.append(t)
        if step % 500 == 0 and not bool(torch.isfinite(f).all()):
            raise RuntimeError(f"blow-up at step {step}")
    wall = time.time() - t0

    t = np.asarray(ts)
    U = np.asarray(us)
    F1 = np.asarray(fs_ladd)
    P = np.asarray(p_mom)
    Ae = np.asarray(a_ext)
    # momentum-budget force: F_cyl = F_ext - dP/dt  (central difference, NaN-padded)
    F2 = np.full_like(P, np.nan)
    F2[1:-1] = Ae[1:-1] - (P[2:] - P[:-2]) / (t[2:] - t[:-2])

    i0 = int(round(case.transient * T))
    sl = slice(i0, None)
    sl_m = slice(i0, len(t) - 1)  # drop last sample: central difference is NaN there
    u_fit = harmonic_fit(U[sl], omega, t[sl], nh=2)
    # d/dt[a_c cos(wt) + b_s sin(wt)] = -w a_c sin + w b_s cos
    dU = -omega * u_fit["u_cos"] * np.sin(omega * t[sl_m]) + omega * u_fit["u_sin"] * np.cos(omega * t[sl_m])
    U_meas = U[sl_m]
    F1_meas = F1[sl_m]
    F2_meas = F2[sl_m]

    kpis = {}
    for tag, F, Dh in (
        ("E1", F1_meas, geo["D_halfway"]),
        ("E2", F2_meas, geo["D_halfway"]),
        ("E1_area", F1_meas, geo["D_area"]),
        ("E2_area", F2_meas, geo["D_area"]),
    ):
        kpis[tag] = morison_ls(F, U_meas, dU, Dh)

    ref = 2.0
    out = {
        "case": asdict(case),
        "numerics": {
            "nx": nx, "ny": ny, "n_fluid": n_fluid, "D": D, "nu": nu, "tau": tau,
            "T_steps": T, "omega": omega, "a0": a0, "nu": nu,
            "n_steps": n_steps, "transient_steps": i0, "measure_steps": int(n_steps - i0),
            "Ma": case.u0 * math.sqrt(3.0), "wall_s": round(wall, 1),
            "ms_per_step": round(1000.0 * wall / n_steps, 3),
        },
        "geometry": geo,
        "n_interior_zeroed": int(interior.sum()),
        "achieved": {
            "U_amp": u_fit["amp"],
            "KC": math.pi * 2.0 * u_fit["amp"] / (omega * geo["D_halfway"]),
            "beta": geo["D_halfway"] ** 2 / (nu * T),
            "U_fit_resid_rel": u_fit["resid_rel"],
            "U_harmonic2_rel": float(math.hypot(u_fit["coef"][3], u_fit["coef"][4])
                                     / (u_fit["amp"] + 1e-30)),
        },
        "kpi": kpis,
        "reference": {"C_M": ref, "source": "Morison 1950 / Sarpkaya 1986 ideal-fluid cylinder"},
        "err_E1_halfway_pct": 100.0 * (kpis["E1"]["C_M"] - ref) / ref,
        "err_E1_area_pct": 100.0 * (kpis["E1_area"]["C_M"] - ref) / ref,
        "estimator_spread_pct": 100.0 * abs(kpis["E1"]["C_M"] - kpis["E2"]["C_M"]) / ref,
        "stability_ok": bool(np.isfinite(F1).all() and np.isfinite(F2).all()),
    }
    return out, {"t": t, "U": U, "F1": F1, "F2": F2, "dU": dU, "a_t": Ae}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="", choices=["", *CASE_ORDER])
    ap.add_argument("--index", type=int, default=-1, help="shard picker: run CASE_ORDER[index::shards]")
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.case:
        names = [args.case]
    elif args.index >= 0:
        names = CASE_ORDER[args.index :: args.shards]
    else:
        names = CASE_ORDER

    out_root = Path(args.out) if args.out else (Path(__file__).resolve().parent / "runs")
    for name in names:
        case = CASES[name]
        print(f"[run] {name}: D={case.D} KC={case.kc} beta={case.beta} box={case.box}D", flush=True)
        try:
            res, trace = run(case, threads=args.threads)
        except Exception as exc:  # noqa: BLE001 - honest failure, recorded
            res = {"case": asdict(case), "status": "failed", "failure_reason": repr(exc)}
            print(json.dumps(res), flush=True)
            (out_root / name).mkdir(parents=True, exist_ok=True)
            (out_root / name / "result.json").write_text(json.dumps(res, indent=2) + "\n")
            continue
        res["status"] = "ok"
        d = out_root / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "result.json").write_text(json.dumps(res, indent=2) + "\n")
        np.savez_compressed(d / "trace.npz", **trace)
        print(json.dumps({k: res[k] for k in
                          ("case", "achieved", "kpi", "err_E1_halfway_pct",
                           "estimator_spread_pct", "numerics")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())