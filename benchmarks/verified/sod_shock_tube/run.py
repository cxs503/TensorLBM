#!/home/wxsc/anaconda3/envs/ftw-env/bin/python
"""B31: Isothermal Sod shock tube with the existing D2Q9 lattice BGK (verified).

Physics: the D2Q9 equilibrium enforces the ISOTHERMAL EOS p = rho/3 (cs^2=1/3,
no energy equation).  The correct analytical reference is therefore the
isothermal Riemann problem solution (NOT the gamma=1.4 Sod solution).

Isothermal Riemann solution, left (rhoL,0) / right (rhoR,0):
  shock density ratio r solves  ln(rhoL/(r*rhoR)) = (r-1)/sqrt(r)
  middle state: rho_star = r*rhoR, u_star = cs*(r-1)/sqrt(r)
  shock speed:  W = cs*sqrt(r)
  rarefaction:  head at s=-cs, tail at s=u_star-cs, inside rho = rhoL*exp(-(s+cs)/cs)

Setup: 1D tube (ny=4), periodic in both x,y.  The periodic x-boundary creates a
mirror Riemann problem at x=0 with reversed states; its waves stay out of the
comparison region as long as t < min(xd/cs, (nx-xd)/W).  All diagnostics use
the clean window [cs*t, nx - W*t] and a shock-tracking window
[xd+0.5*W*t, xd+1.5*W*t].

Resolutions: nx=2000 (t<=400), nx=4000 (t<=800), tau=0.8 (nu=0.1).
IC: rho_L=1.0, rho_R=0.25 (density ratio 4:1), u=0, f=feq (exact cell jump).

Verdict: both resolutions stable, L2_rho<=3% (normalized by jump), |W_rel|<=3%,
|rho_mid_rel|<=3%, |u_mid_rel|<=3%, error not increasing with resolution.
Supplementary: acoustic shock tube eps=0.01 (linear acoustics, Ma~0.01) and
eps=0.25 (Ma_mid~0.26) — wave speed vs cs.
"""

import json
import math
import os
import sys

sys.path.insert(0, "/home/wxsc/cxs/TensorLBM/src")
import numpy as np
import torch

torch.set_num_threads(32)
torch.manual_seed(42)

from tensorlbm.d2q9 import equilibrium, macroscopic  # common-module path
from tensorlbm.solver import collide_bgk, stream  # common-module path

CS = 1.0 / math.sqrt(3.0)
HERE = os.path.dirname(os.path.abspath(__file__))
RESULT = {
    "case": "sod_shock_tube_isothermal_d2q9",
    "verified": False,
    "reference": "isothermal Riemann solution (self-computed, p=rho/3)",
}


def iso_riemann(rhoL, rhoR, cs=CS):
    a = rhoL / rhoR
    f = lambda r: math.log(a / r) - (r - 1.0) / math.sqrt(r)
    lo, hi = 1.0 + 1e-12, a
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    r = 0.5 * (lo + hi)
    return r * rhoR, cs * (r - 1.0) / math.sqrt(r), cs * math.sqrt(r)


def ana_profile(x, t, rhoL, rhoR, rs, us, W, xd, cs=CS):
    s = (x - xd) / max(t, 1e-12)
    rho = np.empty_like(x)
    u = np.empty_like(x)
    m1 = s < -cs
    m2 = (s >= -cs) & (s < us - cs)
    m3 = (s >= us - cs) & (s < W)
    m4 = s >= W
    rho[m1] = rhoL
    u[m1] = 0.0
    rho[m2] = rhoL * np.exp(-(s[m2] + cs) / cs)
    u[m2] = s[m2] + cs
    rho[m3] = rs
    u[m3] = us
    rho[m4] = rhoR
    u[m4] = 0.0
    return rho, u


def run(nx, ny, rhoL, rhoR, tau, steps, snapshots):
    xd = nx // 2
    x = torch.arange(nx, dtype=torch.float32)
    rho = torch.where(x < xd, torch.full_like(x, rhoL), torch.full_like(x, rhoR))
    rho = rho.unsqueeze(0).expand(ny, nx)
    ux = torch.zeros(ny, nx, dtype=torch.float32)
    uy = torch.zeros(ny, nx, dtype=torch.float32)
    f = equilibrium(rho, ux, uy)
    profs = {}
    bad = None
    for t in range(steps + 1):
        if t in snapshots:
            rr, uu, _ = macroscopic(f)
            profs[t] = (rr[0].numpy().copy(), uu[0].numpy().copy())
        f = stream(f)
        f = collide_bgk(f, tau)
        if t % 50 == 0 and bad is None:
            rr, _, _ = macroscopic(f)
            if torch.isnan(rr).any() or (rr < 0).any():
                bad = t
    rr, uu, _ = macroscopic(f)
    if bad is None and (torch.isnan(rr).any() or (rr < 0).any()):
        bad = steps
    return profs, bad


def metrics(profs, rhoL, rhoR, rs, us, W, xd, nx, tlist):
    x = np.arange(nx, dtype=np.float64)
    out = {}
    for t in tlist:
        rho, u = profs[t]
        rhoA, uA = ana_profile(x, t, rhoL, rhoR, rs, us, W, xd)
        lo, hi = int(math.ceil(CS * t)) + 1, int(nx - W * t) - 1
        mask = np.zeros(nx, bool)
        mask[max(lo, 0) : min(hi, nx)] = True
        jump = rhoL - rhoR
        L2r = float(np.sqrt(np.mean((rho[mask] - rhoA[mask]) ** 2)) / jump)
        L2u = float(np.sqrt(np.mean((u[mask] - uA[mask]) ** 2)) / max(us, 1e-12))
        # smooth-region L2 (exclude shock transition +/-10 cells)
        wlo, whi = int(xd + 0.5 * W * t), int(xd + 1.5 * W * t)
        g = np.abs(np.gradient(rho))
        g[:wlo] = 0.0
        g[whi:] = 0.0
        xsh = int(np.argmax(g))
        mask2 = mask.copy()
        mask2[max(0, xsh - 10) : min(nx, xsh + 10)] = False
        L2r_smooth = float(np.sqrt(np.mean((rho[mask2] - rhoA[mask2]) ** 2)) / jump)
        # local shock oscillation (window +-25)
        win = rho[max(0, xsh - 25) : min(nx, xsh + 25)]
        osc = float(np.max(win) - rs) / jump  # overshoot above middle state
        # middle-state check
        tail = xd + (us - CS) * t
        mid = (x > tail + 10) & (x < xsh - 10)
        u_mid = float(u[mid].mean()) if mid.any() else float("nan")
        rho_mid = float(rho[mid].mean()) if mid.any() else float("nan")
        out[t] = dict(
            L2_rho=L2r,
            L2_u=L2u,
            L2_rho_smooth=L2r_smooth,
            shock_x=xsh,
            osc_frac=osc,
            rho_mid=rho_mid,
            u_mid=u_mid,
        )
    ts = sorted(out.keys())
    # linear fit of shock position vs t (all snapshots)
    A = np.vstack([np.array(ts, float), np.ones(len(ts))]).T
    coeff, *_ = np.linalg.lstsq(A, [out[t]["shock_x"] for t in ts], rcond=None)
    W_fit = float(coeff[0])
    out["W_fit"] = W_fit
    out["W_ana"] = W
    out["W_rel"] = (W_fit - W) / W
    out["rho_mid_rel"] = (out[ts[-1]]["rho_mid"] - rs) / rs
    out["u_mid_rel"] = (out[ts[-1]]["u_mid"] - us) / us
    out["L2_rho_final"] = out[ts[-1]]["L2_rho"]
    out["L2_u_final"] = out[ts[-1]]["L2_u"]
    out["osc_frac_final"] = out[ts[-1]]["osc_frac"]
    return out


def main():
    sod_cases = []
    for nx, steps, snaps in [(2000, 400, [100, 200, 300, 400]), (4000, 800, [200, 400, 600, 800])]:
        rhoL, rhoR = 1.0, 0.25
        profs, bad = run(nx, 4, rhoL, rhoR, 0.8, steps, snaps)
        rs, us, W = iso_riemann(rhoL, rhoR)
        m = metrics(profs, rhoL, rhoR, rs, us, W, nx // 2, nx, snaps)
        per_t = {str(t): {k2: v2 for k2, v2 in m[t].items()} for t in snaps}
        sod_cases.append(
            dict(
                nx=nx,
                steps=steps,
                first_bad_step=bad,
                rho_star=rs,
                u_star=us,
                Ma_mid=us / CS,
                per_t=per_t,
                **{k: v for k, v in m.items() if not isinstance(k, int)},
            )
        )
        print(
            f"nx={nx}: bad={bad} L2r={m['L2_rho_final']:.5f} "
            f"L2r_smooth={m['L2_rho_smooth'] if 'L2_rho_smooth' in m else m[snaps[-1]]['L2_rho_smooth']:.5f} "
            f"W_fit={m['W_fit']:.4f} W_rel={m['W_rel'] * 100:.2f}% "
            f"rho_mid_rel={m['rho_mid_rel'] * 100:.3f}% u_mid_rel={m['u_mid_rel'] * 100:.3f}% "
            f"osc_frac={m['osc_frac_final'] * 100:.2f}%"
        )
        np.savez_compressed(
            os.path.join(HERE, f"profiles_nx{nx}.npz"),
            **{f"t{t}": profs[t][0] for t in snaps},
            **{f"u{t}": profs[t][1] for t in snaps},
        )

    # ---- acoustic supplementary (weak-compressible validation) ----
    acou = []
    for eps, steps, snaps in [(0.01, 400, [100, 200, 300, 400]), (0.25, 800, [200, 400, 600, 800])]:
        nx = 4000
        rhoL, rhoR = 1.0 + eps, 1.0 - eps
        profs, bad = run(nx, 4, rhoL, rhoR, 0.8, steps, snaps)
        rs, us, W = iso_riemann(rhoL, rhoR)
        m = metrics(profs, rhoL, rhoR, rs, us, W, nx // 2, nx, snaps)
        acou.append(
            dict(
                eps=eps,
                first_bad_step=bad,
                Ma_mid=us / CS,
                W_fit=m["W_fit"],
                W_ana=W,
                W_rel=m["W_rel"],
                u_mid_rel=m["u_mid_rel"],
                L2_rho_final=m["L2_rho_final"],
            )
        )
        print(
            f"acoustic eps={eps}: bad={bad} W_rel={m['W_rel'] * 100:.3f}% "
            f"u_mid_rel={m['u_mid_rel'] * 100:.3f}% L2r={m['L2_rho_final']:.5f}"
        )

    # ---- verdict ----
    ok = all(c["first_bad_step"] is None for c in sod_cases)
    for c in sod_cases:
        ok &= (
            c["L2_rho_final"] <= 0.03
            and abs(c["W_rel"]) <= 0.03
            and abs(c["rho_mid_rel"]) <= 0.03
            and abs(c["u_mid_rel"]) <= 0.03
        )
    # monotone convergence: L2 not increasing with resolution
    ok &= sod_cases[1]["L2_rho_final"] <= sod_cases[0]["L2_rho_final"] * 1.02
    RESULT.update(
        dict(
            verified=bool(ok),
            tau=0.8,
            nu=0.1,
            ny=4,
            sod_cases=sod_cases,
            acoustic_supplementary=acou,
            note="Reference is the ISOTHERMAL Riemann solution (p=rho/3). "
            "gamma=1.4 Sod solution is NOT applicable to D2Q9.",
        )
    )
    with open(os.path.join(HERE, "result.json"), "w") as fh:
        json.dump(RESULT, fh, indent=1)
    print("VERDICT:", "VERIFIED" if ok else "NOT VERIFIED")
    print("DONE")


if __name__ == "__main__":
    main()
