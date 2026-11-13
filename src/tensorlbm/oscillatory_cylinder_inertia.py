"""Inertia (added-mass) coefficient of a circular cylinder in oscillatory flow.

Capability
----------
Measures the first-harmonic *in-phase* (added-mass) and *quadrature* (damping)
force coefficients of a fixed circular obstacle in a doubly periodic D2Q9 box
driven by a uniform harmonic body force

    a(t) = a0 cos(omega t),   applied to fluid nodes only (second-order Guo source),

with two *independent* force estimators on the same run:

  E1  Ladd momentum exchange on the solid nodes (``boundary.compute_obstacle_forces``),
      sampled post-stream / pre-bounce-back;
  E2  pressure + viscous traction integrated over the staircase surface
      (``p = rho/3``, ``tau_visc = -(1 - 1/(2 tau)) Pi^neq``).

Reference
---------
The conventional invariant of this configuration is the *added-mass* coefficient
``C_a = m_a/(rho V)``, not the Morison ``C_M = 1 + C_a``: a body held fixed in a
uniformly body-forced periodic fluid sees no ambient pressure gradient, so the
Froude-Krylov term is absent.  In the body force momentum balance

    rho a0 N_fluid = dP/dt + F_body ,   P = int_fluid rho u ,

the disturbance potential ``phi = a0 t psi`` obeys ``lap psi = 0``,
``dpsi/dn = -n_x`` and unsteady Bernoulli gives ``p/rho = -a0 psi`` -- the
``a0 x`` terms of ``dphi/dt`` and of the body-force potential cancel exactly.
Hence ``F = rho a0 int |grad psi|^2 = C_a rho V a0`` with ``C_a -> 1`` as the
box grows (``fsi_added_mass.potential_added_mass`` supplies the same-mask,
same-domain value; it is the reference used by this module).

KC = 2 pi U0 / (omega D) and beta = D^2/(nu T) are the similarity numbers; a
harmonic decomposition of the force is only a linear-response measurement while
KC << 1 (at KC = 1, beta = 100 the fit residual is O(0.5) of the signal, i.e. the
nonlinear / history part is no longer a small correction).

Usage
-----
    python -m tensorlbm.oscillatory_cylinder_inertia --list
    python -m tensorlbm.oscillatory_cylinder_inertia --case base --out runs/x
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .boundary.boundaries import bounce_back_cells, compute_obstacle_forces
from .boussinesq import apply_guo_forcing
from .d2q9 import C, equilibrium, macroscopic
from .solver import stream

__all__ = [
    "ArmCase",
    "CASES",
    "circle_mask",
    "geometry_readout",
    "harmonic_decompose",
    "inertia_coefficient",
    "potential_reference",
    "run_arm",
    "main",
]


@dataclass(frozen=True)
class ArmCase:
    """Predeclared arm configuration (lattice units)."""

    name: str
    D: int = 24            # obstacle reference diameter [cells]
    kc: float = 1.0        # Keulegan-Carpenter number (nominal, from a0/omega)
    beta: float = 100.0    # Stokes number D^2/(nu T)
    u0: float = 0.04       # nominal far-field velocity amplitude
    box: float = 12.0      # box side / D (doubly periodic)
    transient: float = 1.5  # periods discarded
    measure: float = 3.0    # periods kept (integer -> orthogonal harmonics)


CASES: dict[str, ArmCase] = {
    "base": ArmCase("base"),
    "res16": ArmCase("res16", D=16),
    "res32": ArmCase("res32", D=32),
    "kc05": ArmCase("kc05", kc=0.5),
    "kc20": ArmCase("kc20", kc=2.0),
    "beta50": ArmCase("beta50", beta=50.0),
    "beta200": ArmCase("beta200", beta=200.0),
    "box16": ArmCase("box16", box=16.0),
    "quick": ArmCase("quick", D=12, box=6.0, transient=1.0, measure=2.0),
}
CASE_ORDER = ["base", "res16", "res32", "kc05", "kc20", "beta50", "beta200", "box16"]


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def circle_mask(nx: int, ny: int, D: float, device: Any = "cpu") -> torch.Tensor:
    """Solid mask of a circle of reference diameter ``D`` centred on the box."""
    y, x = torch.meshgrid(
        torch.arange(ny, device=device), torch.arange(nx, device=device), indexing="ij"
    )
    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
    return (x - cx).square() + (y - cy).square() <= (D / 2.0) ** 2


def geometry_readout(mask: torch.Tensor) -> dict[str, float]:
    """Area / perimeter / three diameter readings of a solid mask.

    ``D_area``  = area-equivalent diameter of the cell mask;
    ``D_halfway`` = wetted diameter when the no-slip surface sits half a cell
    outside the last solid cell (area A + P/2), i.e. the halfway bounce-back
    convention.  Both are reported because a cylinder's added mass scales like
    D^2, so a 5% diameter ambiguity is a 10% coefficient ambiguity.
    """
    nx = int(mask.shape[1])
    fluid = ~mask
    shift = torch.zeros_like(mask)
    shift[1:, :] |= mask[:-1, :]
    shift[:-1, :] |= mask[1:, :]
    shift[:, 1:] |= mask[:, :-1]
    shift[:, :-1] |= mask[:, 1:]
    perim = int((shift & fluid).sum())
    area = int(mask.sum())
    return {
        "nx": float(nx),
        "n_solid": float(area),
        "perimeter_cells": float(perim),
        "D_area": 2.0 * math.sqrt(area / math.pi),
        "D_halfway": 2.0 * math.sqrt((area + perim / 2.0) / math.pi),
        "solid_fraction": float(area) / float(mask.numel()),
    }


def _interior_nodes(mask: torch.Tensor) -> torch.Tensor:
    """Solid cells with no 4-neighbour fluid cell (their populations must be zeroed)."""
    fluid = ~mask
    dil = torch.zeros_like(mask)
    dil[1:, :] |= fluid[:-1, :]
    dil[:-1, :] |= fluid[1:, :]
    dil[:, 1:] |= fluid[:, :-1]
    dil[:, :-1] |= fluid[:, 1:]
    return mask & ~dil


def _surface_normals(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Outward normals (fluid -> solid) of the staircase surface and the face count."""
    nx = int(mask.shape[1])
    fluid = ~mask
    jj, ii = torch.meshgrid(torch.arange(nx), torch.arange(nx), indexing="ij")
    nx_s = torch.zeros((nx, nx), dtype=torch.float64)
    ny_s = torch.zeros((nx, nx), dtype=torch.float64)
    for i in range(1, 9):
        jn = (jj + int(C[i, 1])) % nx
        inn = (ii + int(C[i, 0])) % nx
        hit = fluid & mask[jn, inn]
        nx_s[hit] += float(C[i, 0])
        ny_s[hit] += float(C[i, 1])
    norm = torch.sqrt(nx_s**2 + ny_s**2)
    face = (norm > 0).to(torch.float64)
    return nx_s / norm.clamp(min=1e-30), ny_s / norm.clamp(min=1e-30), face


def _traction_force(f: torch.Tensor, mask: torch.Tensor, tau: float,
                    normals: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                    rho: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor) -> tuple[float, float]:
    """Force on the solid from the pressure and viscous traction at the wall."""
    nxu, nyu, face = normals
    feq = equilibrium(rho, ux, uy)
    fn = f - feq
    ccx = C[:, 0].view(9, 1, 1).to(f.dtype)
    ccy = C[:, 1].view(9, 1, 1).to(f.dtype)
    pxx = (ccx * ccx * fn).sum(0)
    pxy = (ccx * ccy * fn).sum(0)
    pyy = (ccy * ccy * fn).sum(0)
    press = rho / 3.0
    k1 = 1.0 - 0.5 / tau
    # force ON the solid: +p n (pressure pushes into the body) - tau.n,
    # with n the outward normal of the fluid (fluid -> solid).
    tx = press * nxu + k1 * (pxx * nxu + pxy * nyu)
    ty = press * nyu + k1 * (pxy * nxu + pyy * nyu)
    return float((tx * face).sum()), float((ty * face).sum())


# --------------------------------------------------------------------------- #
# harmonic analysis
# --------------------------------------------------------------------------- #
def harmonic_decompose(t: Any, y: Any, omega: float, t0: float | None = None,
                       nh: int = 1, t1: float | None = None) -> dict[str, Any]:
    """Least-squares harmonic decomposition ``y = dc + sum_k [c_k cos + s_k sin]``.

    ``t0`` selects the start of the measurement window (default: window start at
    ``t[0]``).  The window should hold an integer number of periods so that the
    bases are orthogonal and the DC term cannot leak into the first harmonic.
    """
    import numpy as np

    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.ones_like(t, dtype=bool)
    if t0 is not None:
        m &= t >= t0
    if t1 is not None:
        m &= t < t1
    tt, yy = t[m], y[m]
    cols = [np.ones_like(tt)]
    for k in range(1, nh + 1):
        cols += [np.cos(k * omega * tt), np.sin(k * omega * tt)]
    A = np.stack(cols, axis=1)
    coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
    fit = A @ coef
    resid = float(np.sqrt(np.mean((yy - fit) ** 2)))
    amp = float(math.hypot(coef[1], coef[2]))
    return {
        "dc": float(coef[0]),
        "cos": float(coef[1]),
        "sin": float(coef[2]),
        "amp": amp,
        "phase_rad": float(math.atan2(coef[1], coef[2])),
        "resid_rms": resid,
        "resid_rel": float(resid / (np.std(yy) + 1e-300)),
        "n_samples": int(tt.size),
        "harmonics": [{"k": k, "cos": float(coef[2 * k - 1]), "sin": float(coef[2 * k])}
                      for k in range(1, nh + 1)],
    }


def inertia_coefficient(F_cos: float, a0: float, D: float, rho: float = 1.0) -> float:
    """Added-mass coefficient from the first harmonic in phase with the drive ``a(t)``.

    ``F(t) = rho V C_I a0 cos(omega t) + (damping) sin(omega t)`` with
    ``V = pi D^2 / 4`` per unit depth.
    """
    vol = math.pi * D * D / 4.0
    return float(F_cos / (rho * vol * a0))


def potential_reference(nx: int, ny: int, D: float, rho: float = 1.0) -> dict[str, float]:
    """Same-mask, same-domain potential-flow added mass (reuses ``fsi_added_mass``)."""
    from .fsi_added_mass import potential_added_mass

    ref = potential_added_mass(nx, ny, radius=D / 2.0, rho=rho)
    area_in = math.pi * (D / 2.0) ** 2
    mask = circle_mask(nx, ny, D)
    geo = geometry_readout(mask)
    a_solid = geo["n_solid"]
    a_half = geo["n_solid"] + geo["perimeter_cells"] / 2.0
    out = dict(ref)
    out["c_a_area"] = ref["c_a"] * area_in / a_solid
    out["c_a_halfway"] = ref["c_a"] * area_in / a_half
    out["area_solid"] = a_solid
    out["area_halfway"] = a_half
    return out


# --------------------------------------------------------------------------- #
# arm
# --------------------------------------------------------------------------- #
def run_arm(case: ArmCase, threads: int = 4, dtype: torch.dtype = torch.float64) -> tuple[dict, dict]:
    """Run one arm case; returns (result dict, trace dict)."""
    torch.set_num_threads(int(threads))
    D = case.D
    nx = ny = int(round(case.box * D))
    T = case.kc * D / case.u0
    omega = 2.0 * math.pi / T
    a0 = omega * case.u0
    nu = D * D / (case.beta * T)
    tau = 3.0 * nu + 0.5
    if tau <= 0.5:
        raise ValueError(f"tau={tau:.5f} <= 0.5: beta={case.beta} is not reachable at this D")
    mask = circle_mask(nx, ny, D)
    fluid = ~mask
    geo = geometry_readout(mask)
    interior = _interior_nodes(mask)
    normals = _surface_normals(mask)

    rho0 = torch.ones((ny, nx), dtype=dtype)
    zero = torch.zeros_like(rho0)
    f = equilibrium(rho0, zero, zero)
    f = bounce_back_cells(f, mask)
    Fx = torch.zeros((ny, nx), dtype=dtype)
    Fy = torch.zeros_like(Fx)

    measure_start = int(round(case.transient * T))
    n_steps = measure_start + int(round(case.measure * T))
    ts: list[float] = []
    f_mem: list[float] = []
    f_trac: list[float] = []
    u_mean: list[float] = []
    t0 = __import__("time").time()
    for step in range(1, n_steps + 1):
        t = float(step)
        a_t = a0 * math.cos(omega * t)
        Fx[fluid] = a_t
        old_solid = f[:, mask].clone()
        f = apply_guo_forcing(f, torch.stack([Fx, Fy]), tau)
        f[:, mask] = old_solid
        f = stream(f)
        fx, _fy = compute_obstacle_forces(f, mask)
        rho, ux, uy = macroscopic(f)
        ftx, _fty = _traction_force(f, mask, tau, normals, rho, ux, uy)
        f = bounce_back_cells(f, mask)
        f[:, interior] = 0.0
        ts.append(t)
        f_mem.append(float(fx))
        f_trac.append(ftx)
        u_mean.append(float((rho * ux)[fluid].sum()) / float(rho[fluid].sum()))
    wall = __import__("time").time() - t0

    t_meas = float(measure_start)
    t_end = t_meas + case.measure * T
    dec_mem = harmonic_decompose(ts, f_mem, omega, t0=t_meas, t1=t_end, nh=2)
    dec_trac = harmonic_decompose(ts, f_trac, omega, t0=t_meas, t1=t_end, nh=2)
    dec_u = harmonic_decompose(ts, u_mean, omega, t0=t_meas, t1=t_end, nh=2)
    a0_meas = dec_u["amp"] * omega
    kpi = {}
    for tag, dec, dd in (
        ("E1_mem_halfway", dec_mem, geo["D_halfway"]),
        ("E1_mem_area", dec_mem, geo["D_area"]),
        ("E2_trac_halfway", dec_trac, geo["D_halfway"]),
        ("E2_trac_area", dec_trac, geo["D_area"]),
    ):
        kpi[tag] = {
            "C_I": inertia_coefficient(dec["cos"], a0, dd),
            "C_Q": inertia_coefficient(dec["sin"], a0, dd),
            "D": dd,
        }
    pot = potential_reference(nx, ny, D)
    c_ref_a = pot["c_a_area"]
    err = 100.0 * (kpi["E1_mem_area"]["C_I"] - c_ref_a) / c_ref_a
    result = {
        "case": asdict(case),
        "geometry": geo,
        "numerics": {
            "nx": nx, "ny": ny, "n_fluid": int(fluid.sum()), "nu": nu, "tau": tau,
            "T_steps": T, "omega": omega, "a0": a0, "n_steps": n_steps,
            "measure_start_step": measure_start, "Ma": case.u0 * math.sqrt(3.0),
            "wall_s": round(wall, 2), "dtype": str(dtype).split(".")[-1],
        },
        "achieved": {
            "U_amp_box_mean": dec_u["amp"],
            "a0_measured": a0_meas,
            "a0_nominal": a0,
            "a0_over_nominal": a0_meas / a0,
            "KC_halfway": 2.0 * math.pi * dec_u["amp"] / (omega * geo["D_halfway"]),
            "beta_halfway": geo["D_halfway"] ** 2 / (nu * T),
            "Re_halfway": 2.0 * math.pi * dec_u["amp"] * geo["D_halfway"] / (omega * nu * T),
            "U_resid_rel": dec_u["resid_rel"],
        },
        "harmonic": {"E1_mem": dec_mem, "E2_trac": dec_trac, "U": dec_u},
        "kpi": kpi,
        "reference": {
            "potential_same_mask_area": pot["c_a_area"],
            "potential_same_mask_halfway": pot["c_a_halfway"],
            "analytic_unbounded": 1.0,
            "source": "fsi_added_mass.potential_added_mass (same mask/domain) + analytic C_a=1",
        },
        "err_inphase_pct_area": err,
        "estimator_spread_pct": abs(kpi["E1_mem_area"]["C_I"] - kpi["E2_trac_area"]["C_I"]),
        "stability_ok": bool(all(math.isfinite(v) for v in f_mem + f_trac)),
    }
    trace = {"t": ts, "F_mem": f_mem, "F_trac": f_trac, "U": u_mean, "omega": omega,
             "a0": a0, "A_solid": geo["n_solid"]}
    return result, trace


# --------------------------------------------------------------------------- #
# CLI entry
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    import numpy as np

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--case", default="", choices=["", *CASE_ORDER, "quick"])
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    if args.list:
        for name in [*CASE_ORDER, "quick"]:
            print(name, asdict(CASES[name]))
        return 0
    names = [args.case] if args.case else CASE_ORDER
    out_root = Path(args.out) if args.out else Path.cwd() / "runs" / "oscillatory_cylinder_inertia"
    for name in names:
        case = CASES[name]
        print(f"[run] {name}: D={case.D} KC={case.kc} beta={case.beta} box={case.box}D", flush=True)
        try:
            result, trace = run_arm(case, threads=args.threads)
            result["status"] = "ok"
        except Exception as exc:  # honest failure, recorded
            result = {"case": asdict(case), "status": "failed", "failure_reason": repr(exc)}
            trace = {}
        d = out_root / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        if trace:
            np.savez_compressed(d / "trace.npz", **trace)
        print(json.dumps({k: result.get(k) for k in
                          ("status", "achieved", "err_inphase_pct_area",
                           "estimator_spread_pct")}, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())