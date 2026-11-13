"""XVAL-LBM-R4-COEXISTENCE — cross-validation of the R4 coexistence-calibration and
free-energy-selection route (TensorLBM, R4 = docs/MCMP_STRATIFIED_POISEUILLE_R4.md §2.2/§6).

Two independent routes are compared, both pre-registered before any run:

  ROUTE M (reference / "ref")   Maxwell equal-area construction on the lattice Shan-Chen
      mechanical pressure p(rho) = rho cs^2 - (G/2) cs^2 psi^2, in SPECIFIC-VOLUME space,
      with the chemical-potential equality mu(rho_v)=mu(rho_l) as an independent correctness
      check.  Implementation: `tensorlbm.sc_eos_coexistence.maxwell_coexistence`
      (conventions from the OSS catalogue: ludwig-cf/ludwig commit 78d83c5 pressure/force
      convention + sailfish symbolic pressure tensor, see §"diff list").

  ROUTE S (candidate / "got")   Direct measurement of the coexisting densities from a
      flat-interface lattice-Boltzmann run (D2Q9, exact-difference/velocity-shift forcing,
      fully periodic strip: two interfaces, no walls, no body force), i.e. the classic
      coexistence calibration of Shan & Chen (1994) that R4 adopted in place of re-using a
      nominal density pair.

  ROUTE C (free-energy selection, R4 §6)  Cahn-Hilliard gradient-flow relaxation of the
      transplanted free-energy functional (ludwig convention a<0, b>0, kappa):
          mu = a*phi + b*phi^3 - kappa*lap(phi),   dphi/dt = M lap(mu)
      vs the analytic equilibrium interface of the same functional:
          phi = phi0 tanh((y-y0)/xi),  xi = sqrt(2 kappa/A),  phi0 = sqrt(A/B)
          sigma = (2 sqrt2 / 3) sqrt(kappa A^3) / b
      Measured -> xi from the 10-90 thickness and from a tanh least-squares fit; sigma from
      the discrete pressure-tensor integral sigma = int (P_NN - P_TT) dy.

Gate (pre-registered, owner standard): |error| < 15 % on every compared quantity.
Sensitivity (>=2 knobs): lattice resolution and kappa for ROUTE C; resolution for ROUTE S.

Outputs: artifacts/xval_lbm_r4_coexistence/xval_r4_coexistence.json  (+ png figure path).
No hand-typed numbers: every number in the report is read from that JSON.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
torch.set_default_dtype(torch.float64)
torch.set_num_threads(4)

from tensorlbm import free_energy_stress as fes                      # noqa: E402
from tensorlbm.sc_eos_coexistence import maxwell_coexistence        # noqa: E402

OUT = os.path.join(ROOT, "artifacts", "xval_lbm_r4_coexistence")
os.makedirs(OUT, exist_ok=True)

GATE = 0.15
# ---------------------------------------------------------------- D2Q9 machinery
W = torch.tensor([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)
CX = torch.tensor([0, 1, 0, -1, 0, 1, -1, -1, 1])
CY = torch.tensor([0, 0, 1, 0, -1, 1, 1, -1, -1])


def _rollc(f, i):
    return torch.roll(torch.roll(f, -int(CY[i]), dims=0), -int(CX[i]), dims=1)


def sc_psi(rho, rho0=1.0):
    """Pseudopotential used by the library SC route (psi = rho0 (1 - exp(-rho/rho0)))."""
    return rho0 * (1.0 - torch.exp(-rho / rho0))


def sc_flat_interface(G, ny=96, nx=4, steps=30000, tau=1.0, seed_w=4.0,
                      rho_l0=2.2, rho_v0=0.25, rho0=1.0, avg_from=2000):
    """Fully periodic strip with TWO planar interfaces; no walls, no body force.

    Library convention: G > 0 attractive (sc_eos_coexistence docstring).
    Force = exact-difference (Kupershtokh): F = -G psi(x) sum_i w_i c_i psi(x-c_i),
    applied through the velocity shift u* = u + F/(2 rho).
    """
    y = torch.arange(ny).view(-1, 1).double()
    seed = rho_v0 + (rho_l0 - rho_v0) * 0.5 * (
        torch.tanh((y - ny / 4) / seed_w) - torch.tanh((y - 3 * ny / 4) / seed_w)
    )
    rho = seed.expand(ny, nx).contiguous()
    f = (W.view(9, 1, 1) * rho.unsqueeze(0)).contiguous()
    acc = torch.zeros(ny, nx)
    nacc = 0
    for step in range(steps):
        rho = f.sum(0)
        ux = (f * CX.view(9, 1, 1)).sum(0) / rho
        uy = (f * CY.view(9, 1, 1)).sum(0) / rho
        ps = sc_psi(rho, rho0)
        sx = torch.zeros_like(rho)
        sy = torch.zeros_like(rho)
        for i in range(1, 9):
            pnb = torch.roll(torch.roll(ps, int(CY[i]), dims=0), int(CX[i]), dims=1)
            sx = sx + W[i] * CX[i] * pnb
            sy = sy + W[i] * CY[i] * pnb
        Fx = -G * ps * sx
        Fy = -G * ps * sy
        usx = ux + Fx / (2.0 * rho)
        usy = uy + Fy / (2.0 * rho)
        cu = 3.0 * (CX.view(9, 1, 1) * usx + CY.view(9, 1, 1) * usy)
        usq = 1.5 * (usx * usx + usy * usy)
        feq = W.view(9, 1, 1) * rho.unsqueeze(0) * (1.0 + cu + 0.5 * cu * cu - usq)
        f = (1.0 - 1.0 / tau) * f + feq / tau
        f = torch.stack([_rollc(f[i], i) for i in range(9)])
        if step >= steps - avg_from:
            acc = acc + rho
            nacc += 1
    rho_avg = acc / max(nacc, 1)
    mid = rho_avg[ny // 2, :]
    vap = rho_avg[0, :]
    # +-20 % around each interface at ny/4 and 3ny/4 excluded as interface band
    band = slice(ny // 4 - 6, ny // 4 + 6)
    band2 = slice(3 * ny // 4 - 6, 3 * ny // 4 + 6)
    liq_band = slice(ny // 2 - ny // 8, ny // 2 + ny // 8)
    vap_band = slice(0, ny // 8)
    del band, band2
    rho_l = float(rho_avg[liq_band, :].mean())
    rho_v = float(rho_avg[vap_band, :].mean())
    umax = float(torch.sqrt(ux ** 2 + uy ** 2).max())
    return {"G_library": G, "ny": ny, "steps": steps, "tau": tau,
            "rho_l": rho_l, "rho_v": rho_v, "spurious_u_max": umax,
            "profile_mid_column": [float(v) for v in rho_avg[:, 0]]}


def ch_relax(A=1.0, B=1.0, kappa=2.0, ny=128, nx=4, steps=40000, dt=0.05, M=1.0, seed_w=6.0):
    """Cahn-Hilliard gradient flow of the transplanted free-energy functional (periodic)."""
    p = fes.FreeEnergyParams(A=A, B=B, kappa=kappa)
    phi0 = math.sqrt(p.A / p.B)
    y = torch.arange(ny).view(-1, 1).double()
    phi = phi0 * torch.tanh((y - ny / 4) / seed_w) * torch.tanh((3 * ny / 4 - y) / seed_w)
    # -> -phi0 in the middle band, +phi0 in the two outer bands (two interfaces)
    phi = phi.expand(ny, nx).contiguous().clone()
    for _ in range(steps):
        mu = fes.chemical_potential(phi, p)
        phi = phi + dt * M * fes.laplacian(mu)
        phi.clamp_(-1.3 * phi0, 1.3 * phi0)
    return p, phi


def measure_interface(phi, p, ny):
    """xi (10-90 half-thickness / 2*atanh0.8) and tanh least-squares fit; sigma from tensor."""
    col = phi[:, 0]
    # interface located between y=ny/4 and y=3ny/4 : take the monotone branch around ny/4
    lo, hi = int(0.1 * ny), int(0.4 * ny)
    seg = col[lo:hi]
    vmax = float(seg.max())
    vmin = float(seg.min())
    # phi goes from +phi0 (top of segment) to -phi0
    thr90 = vmin + 0.9 * (vmax - vmin)
    thr10 = vmin + 0.1 * (vmax - vmin)
    def first_cross(thr):
        for i in range(len(seg) - 1):
            if (seg[i] - thr) * (seg[i + 1] - thr) <= 0:
                w = (seg[i] - thr) / (seg[i] - seg[i + 1] + 1e-30)
                return lo + i + w
        return float("nan")
    y90 = first_cross(thr90)   # closest to +phi0 side
    y10 = first_cross(thr10)
    w1090 = abs(y10 - y90)
    xi_1090 = w1090 / (2.0 * math.atanh(0.8))
    # tanh least-squares fit on the full column (y0 from the mid crossing, xi on a grid)
    yy = torch.arange(ny).double()
    phi0 = math.sqrt(p.A / p.B)
    y0 = first_cross(vmin + 0.5 * (vmax - vmin))
    best = (float("inf"), float("nan"))
    for k in range(300):
        xi = 0.30 + 0.01 * k
        model = (phi0 * torch.tanh((yy - y0) / xi)
                 - phi0 * torch.tanh((yy - (ny - y0)) / xi) - phi0)
        res_i = float(((model - col) ** 2).sum())
        if res_i < best[0]:
            best = (res_i, xi)
    xi_lsq = best[1]
    # sigma from discrete pressure tensor on the measured profile
    Pxx, Pyy, _ = fes.pressure_tensor(phi, p, scheme="iso9")
    sigma_int = float((Pyy[:, 0] - Pxx[:, 0]).sum())
    # discrete consistency: the stress-divergence force on the equilibrium profile
    Fx, Fy = fes.stress_divergence_force(phi, p, scheme="iso9")
    mu = fes.chemical_potential(phi, p)
    return {
        "xi_10_90": float(xi_1090),
        "xi_lsq": float(xi_lsq),
        "sigma_tensor": sigma_int,
        "sigma_analytic": float(fes.surface_tension_analytic(p)),
        "xi_analytic": float(fes.interface_width(p)),
        "max_abs_divP": float(torch.sqrt(Fx ** 2 + Fy ** 2).max()),
        "mu_std_over_A": float(mu.std() / p.A),
        "phi0_measured_liquid": float(col[int(0.45 * ny):int(0.55 * ny)].mean()),
    }


def relerr(got, ref):
    return abs(got - ref) / abs(ref)


def main():
    t0 = time.time()
    res = {"gate": GATE, "torch": torch.__version__,
           "conventions": {
               "SC library convention": "G>0 attractive, p = rho*cs^2 - (G/2)*cs^2*psi^2",
               "psi": "rho0*(1-exp(-rho/rho0)), rho0=1",
               "free energy": "ludwig a=-A, b=B, kappa; mu = a phi + b phi^3 - kappa lap(phi)",
           },
           "route_M": [], "route_S": [], "route_C": []}

    # ---- Route M: Maxwell equal-area (ref)
    psi_fn = lambda r: sc_psi(r, 1.0)          # noqa: E731
    for G in (4.0, 4.5, 5.0, 5.5):
        m = maxwell_coexistence(G=G, psi_fn=psi_fn, rho_lo=1e-5, rho_hi=50.0, n=20000)
        d = m.as_dict() if hasattr(m, "as_dict") else dict(m)
        res["route_M"].append({"G": G, **{k: (float(v) if isinstance(v, (int, float)) else str(v))
                                          for k, v in d.items()}})
    print("route M done", time.time() - t0)

    # ---- Route S: direct flat-interface simulation (got)
    for G in (4.0, 4.5, 5.0, 5.5):
        for ny, steps in ((96, 25000), (128, 25000)):
            r = sc_flat_interface(G=G, ny=ny, nx=4, steps=steps)
            res["route_S"].append({k: v for k, v in r.items() if k != "profile_mid_column"})
            if ny == 96:
                res.setdefault("profiles_SC", {})[str(G)] = r["profile_mid_column"]
            print("route S", G, ny, r["rho_l"], r["rho_v"], f"{time.time()-t0:.1f}s")

    # ---- Route C: free-energy selection (got) vs analytic (ref)
    for kappa in (2.0, 0.5):
        for ny in (96, 128):
            p, phi = ch_relax(A=1.0, B=1.0, kappa=kappa, ny=ny, nx=4, steps=20000)
            m = measure_interface(phi, p, ny)
            res["route_C"].append({"kappa": kappa, "A": 1.0, "B": 1.0, "ny": ny, **m})
            if ny == 128:
                res.setdefault("profiles_CH", {})[str(kappa)] = [float(v) for v in phi[:, 0]]
            print("route C", kappa, ny, m["xi_lsq"], m["sigma_tensor"], f"{time.time()-t0:.1f}s")

    # ---- compare tables + gates
    cmp_rows = []
    m_by_G = {round(r["G"], 3): r for r in res["route_M"]}
    for s in res["route_S"]:
        mm = m_by_G.get(round(s["G"], 3))
        if not mm:
            continue
        key_l = next((k for k in mm if "rho_l" in k or "liquid" in k), None)
        key_v = next((k for k in mm if "rho_v" in k or "vapour" in k or "vapor" in k), None)
        if not (key_l and key_v):
            continue
        e_l = relerr(s["rho_l"], float(mm[key_l]))
        e_v = relerr(s["rho_v"], float(mm[key_v]))
        cmp_rows.append({"G": s["G"], "ny": s["ny"], "kind": "coexistence",
                         "ref_rho_l": float(mm[key_l]), "got_rho_l": s["rho_l"],
                         "err_rho_l": e_l, "pass_rho_l": e_l < GATE,
                         "ref_rho_v": float(mm[key_v]), "got_rho_v": s["rho_v"],
                         "err_rho_v": e_v, "pass_rho_v": e_v < GATE,
                         "spurious_u_max": s["spurious_u_max"]})
    for c in res["route_C"]:
        e_xi = relerr(c["xi_lsq"], c["xi_analytic"])
        e_xi2 = relerr(c["xi_10_90"], c["xi_analytic"])
        e_s = relerr(c["sigma_tensor"], c["sigma_analytic"])
        cmp_rows.append({"kind": "free_energy", "kappa": c["kappa"], "ny": c["ny"],
                         "ref_xi": c["xi_analytic"], "got_xi": c["xi_lsq"], "err_xi": e_xi,
                         "got_xi_1090": c["xi_10_90"], "err_xi_1090": e_xi2,
                         "ref_sigma": c["sigma_analytic"], "got_sigma": c["sigma_tensor"],
                         "err_sigma": e_s,
                         "pass_xi": max(e_xi, e_xi2) < GATE, "pass_sigma": e_s < GATE})
    res["comparison"] = cmp_rows
    allp = [v for r in cmp_rows for k, v in r.items() if k.startswith("pass_")]
    res["gate_summary"] = {"n_checks": len(allp), "n_pass": int(sum(bool(v) for v in allp)),
                           "all_pass": bool(all(allp))}
    res["wall_s"] = time.time() - t0

    with open(os.path.join(OUT, "xval_r4_coexistence.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res["gate_summary"]))
    for r in cmp_rows:
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}))
    print("wall", res["wall_s"])


if __name__ == "__main__":
    main()