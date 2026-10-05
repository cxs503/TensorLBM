#!/usr/bin/env python3
"""Mechanism diagnostic for the Blasius flat-plate benchmark.

Isolates the domain-scale acceleration / effective-scale excess.  Options to
toggle: plate on/off, wall=bottom/mid/none, outlet zouhe/convective/nscbc,
top specular/neumann/freestream, domain height, inlet-distance.

Reports per probe: profile u/U, Cf (correct 3-pt weights (-2,3,-1) on
s=0.5,1.5,2.5), inner edge velocity, far-field edge velocity, overshoot,
delta*/theta/H using BOTH the external U and the local edge velocity,
effective compression K = Cf_ref/Cf_sim, and Blasius L2.
"""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import numpy as np
import torch

from tensorlbm.boundaries import zou_he_outlet_pressure, nscbc_outlet_2d
from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.solver import collide_bgk, collide_trt, stream

SPEC = torch.tensor([0, 1, 4, 3, 2, 8, 7, 6, 5], dtype=torch.int64)


def blasius_tab(eta_max=20.0, h=0.002):
    n = int(round(eta_max / h)); etas = np.arange(n + 1) * h
    def rhs(s): return np.array([s[1], s[2], -0.5 * s[0] * s[2]])
    def shoot(a):
        s = np.array([0.0, 0.0, a])
        for _ in range(n):
            k1 = rhs(s); k2 = rhs(s + .5*h*k1); k3 = rhs(s + .5*h*k2); k4 = rhs(s + h*k3)
            s = s + h/6*(k1+2*k2+2*k3+k4)
        return s[1] - 1.0
    lo, hi = 0.30, 0.37; flo = shoot(lo)
    for _ in range(60):
        mid = .5*(lo+hi); fm = shoot(mid)
        if flo*fm <= 0: hi = mid
        else: lo, flo = mid, fm
    a0 = .5*(lo+hi); fp = np.zeros(n+1); s = np.array([0.,0.,a0])
    for i in range(1, n+1):
        k1 = rhs(s); k2 = rhs(s + .5*h*k1); k3 = rhs(s + .5*h*k2); k4 = rhs(s + h*k3)
        s = s + h/6*(k1+2*k2+2*k3+k4); fp[i] = s[1]
    return etas, fp, a0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=440)
    ap.add_argument("--ny", type=int, default=1400)
    ap.add_argument("--le", type=int, default=20)
    ap.add_argument("--plate_len", type=int, default=400)
    ap.add_argument("--probes", type=str, default="80,200")
    ap.add_argument("--U", type=float, default=0.05)
    ap.add_argument("--nu", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--avg", type=int, default=400)
    ap.add_argument("--wall", default="bottom", choices=["bottom", "mid", "none"])
    ap.add_argument("--x0", type=float, default=0.0,
                    help="virtual leading-edge distance upstream of the inlet; "
                         "if >0 the inlet imposes the Blasius profile for x_eff=x0")
    ap.add_argument("--outlet", default="zouhe", choices=["zouhe", "convective", "nscbc"])
    ap.add_argument("--top", default="specular", choices=["specular", "neumann", "freestream"])
    ap.add_argument("--collision", default="bgk", choices=["bgk", "trt"])
    ap.add_argument("--device", default="sdaa:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    nx, ny, le, pl = args.nx, args.ny, args.le, args.plate_len
    probes = [int(v) for v in args.probes.split(",")]
    U, nu = args.U, args.nu
    tau = 3*nu + 0.5
    dev = torch.device(args.device)
    te = le + pl

    solid = torch.zeros((ny, nx), dtype=torch.bool, device=dev)
    if args.wall == "bottom":
        ys = 0; solid[0, le:te] = True
    elif args.wall == "mid":
        ys = ny // 2; solid[ys, le:te] = True
    else:
        ys = None
    mask = solid[ys] if ys is not None else None

    f = equilibrium(torch.ones((ny, nx), device=dev),
                    torch.full((ny, nx), U, device=dev),
                    torch.zeros((ny, nx), device=dev), device=dev)
    m0 = float(f.sum().item())
    spec = SPEC.to(dev)
    if args.x0 > 0.0:
        # synthetic inflow: Blasius profile at the inlet for x_eff=x0
        etas0, fp0, _ = blasius_tab()
        sc0 = math.sqrt(nu * args.x0 / U)
        yy = np.arange(ny, dtype=np.float64)  # distance above wall (wall at 0.5)
        prof0 = U * np.interp(yy / sc0, etas0, fp0)
        prof0[0] = 0.0
        prof0[prof0 > U] = U
        f_in_t = torch.tensor(prof0, device=dev, dtype=torch.float32).view(ny, 1)
        feq_in = equilibrium(torch.ones((ny, 1), device=dev), f_in_t,
                             torch.zeros((ny, 1), device=dev))[:, :, 0].contiguous()
    else:
        feq_in = equilibrium(torch.ones((ny, 1), device=dev),
                             torch.full((ny, 1), U, device=dev),
                             torch.zeros((ny, 1), device=dev))[:, :, 0].contiguous()

    def step(f_):
        f_ = collide_trt(f_, tau, 3/16.) if args.collision == "trt" else collide_bgk(f_, tau)
        f_ = f_.clone()
        if ys is not None:
            if args.wall == "bottom":
                f_[2, 0, :] = torch.where(mask, f_[4, 1, :], f_[2, 0, :])
                f_[5, 0, :-1] = torch.where(mask[:-1], f_[7, 1, 1:], f_[5, 0, :-1])
                f_[6, 0, 1:] = torch.where(mask[1:], f_[8, 1, :-1], f_[6, 0, 1:])
            else:
                f_[2, ys, :] = torch.where(mask, f_[4, ys+1, :], f_[2, ys, :])
                f_[5, ys, :-1] = torch.where(mask[:-1], f_[7, ys+1, 1:], f_[5, ys, :-1])
                f_[6, ys, 1:] = torch.where(mask[1:], f_[8, ys+1, :-1], f_[6, ys, 1:])
                f_[4, ys, :] = torch.where(mask, f_[2, ys-1, :], f_[4, ys, :])
                f_[7, ys, 1:] = torch.where(mask[1:], f_[5, ys-1, :-1], f_[7, ys, 1:])
                f_[8, ys, :-1] = torch.where(mask[:-1], f_[6, ys-1, 1:], f_[8, ys, :-1])
        if args.top == "specular":
            f_[:, -1, :] = f_[:, -2, :][spec]
        elif args.top == "neumann":
            f_[:, -1, :] = f_[:, -2, :]
        f_ = stream(f_)
        f_ = f_.clone()
        f_[:, :, 0] = feq_in
        if args.outlet == "zouhe":
            f_ = zou_he_outlet_pressure(f_, 1.0)
        elif args.outlet == "convective":
            f_[:, :, -1] = f_[:, :, -2]
        else:
            f_ = nscbc_outlet_2d(f_, rho_target=1.0)
        if args.top == "freestream":
            f_[:, -1, :] = feq_in
        return f_

    t0 = time.time()
    for i in range(1, args.steps + 1):
        f = step(f)
        if i % 5000 == 0:
            _, ux, _ = macroscopic(f)
            print(f"  step {i}: umax={float(ux.max()):.5f} finite={bool(torch.isfinite(f).all())}", flush=True)
    elapsed = time.time() - t0

    _, uf, _ = macroscopic(f)
    umax = float(uf.max())
    prof_acc = torch.zeros((len(probes), ny), device=dev)
    for _ in range(args.avg):
        f = step(f)
        _, ux, _ = macroscopic(f)
        for j, p in enumerate(probes):
            prof_acc[j] += ux[:, p]
    prof_acc /= args.avg
    prof_np = prof_acc.cpu().numpy()
    _, ux_f, _ = macroscopic(f)
    mass_drift = (float(f.sum().item()) - m0) / m0 * 100.0

    etas, fp, a0 = blasius_tab()
    # correct 3-pt weights
    s3 = np.array([0.5, 1.5, 2.5])
    A = np.vstack([np.ones(3), s3, s3**2]).T
    w = np.linalg.solve(A.T, np.array([0.0, 1.0, 0.0]))

    out = dict(config=vars(args), umax=umax, mass_drift_pct=mass_drift,
               finite=bool(torch.isfinite(f).all().item()), elapsed_s=round(elapsed, 1),
               ny=ny, nx=nx, probes={})
    for j, p in enumerate(probes):
        x_eff = p - le + args.x0
        if x_eff <= 0:
            continue
        Rex = U * x_eff / nu
        scale = math.sqrt(nu * x_eff / U)
        if ys is None:
            rows = np.arange(0, ny); yw = 0.0
        elif args.wall == "bottom":
            rows = np.arange(1, ny); yw = 0.5
        else:
            rows = np.arange(ys+1, ny); yw = ys + 0.5
        y = rows - yw
        u = prof_np[j][rows]
        up = prof_np[j]
        u1, u2, u3 = up[rows[0]], up[rows[1]], up[rows[2]]
        G = float(w @ np.array([u1, u2, u3]))
        Cf = 2*nu*G/U**2
        Cf_ref = 0.6641146724303926/math.sqrt(Rex)
        eta = y/scale
        # local edge velocity: max u in the inner region (first 6*delta99 ~ 180)
        inner = u[: int(min(len(u), 6*4.91*scale))]
        u_e_in = float(inner.max())
        u_e_far = float(np.mean(u[-30:]))
        # overshoot location
        idx = int(np.argmax(u_e_in if False else u))
        prof = list(zip(eta.tolist(), (u/U).tolist()))
        # integrals with external U and local edge
        dstar_u = float(np.trapezoid(1 - u/U, y))
        theta_u = float(np.trapezoid(u/U*(1-u/U), y))
        dstar_e = float(np.trapezoid(1 - u/u_e_in, y)) if u_e_in > 0 else float('nan')
        theta_e = float(np.trapezoid(u/u_e_in*(1-u/u_e_in), y)) if u_e_in > 0 else float('nan')
        H_u = dstar_u/theta_u if theta_u else float('nan')
        H_e = dstar_e/theta_e if theta_e else float('nan')
        fp_ref = np.interp(eta, etas, fp)
        # L2 in inner region
        m = (eta > 0.05) & (eta < 5.0)
        l2 = float(np.linalg.norm((u/U - fp_ref)[m])/np.sqrt(m.sum()))
        out["probes"][str(p)] = dict(
            x_eff=x_eff, Rex=Rex, scale=scale,
            Cf=Cf, Cf_ref=Cf_ref, Cf_err_pct=(Cf-Cf_ref)/Cf_ref*100,
            K_compress=Cf_ref/Cf,
            G=G, u1_over_U=float(u1/U),
            u_e_inner_over_U=round(u_e_in/U, 5),
            u_e_far_over_U=round(u_e_far/U, 5),
            overshoot_max_over_U=round(float(u.max()/U), 5),
            overshoot_eta=round(float(eta[idx]), 3),
            dstar_U=dstar_u, dstar_ue=dstar_e,
            theta_U=theta_u, theta_ue=theta_e,
            H_U=H_u, H_ue=H_e,
            dstar_ref=1.7208*x_eff/math.sqrt(Rex),
            theta_ref=0.664*x_eff/math.sqrt(Rex),
            l2=l2,
            prof=[(round(e, 4), round(v, 6)) for e, v in prof if e <= 40],
        )
    # global u(x) along several rows (fractions of height, and near-plate rows)
    _, uxf, _ = macroscopic(f)
    uxf = uxf.cpu().numpy()
    if ys is None:
        ys_eff = ny//2
    else:
        ys_eff = ys
    rows_report = {"h10": max(ys_eff-10, 0), "h20": max(ys_eff+20, 0) if args.wall != "mid" else ys_eff+20,
                   "far": ny-3}
    out["u_rows"] = {}
    for k, r in rows_report.items():
        r = int(min(max(r, 0), ny-1))
        out["u_rows"][k] = dict(row=r, u_over_U=[round(float(v)/U, 5) for v in uxf[r][::max(1, nx//40)]])
    print(json.dumps({k: v for k, v in out.items() if k not in ("probes","u_rows")}, indent=2))
    for p, d in out["probes"].items():
        print(f"probe {p}: Cf_err={d['Cf_err_pct']:+7.2f}% K={d['K_compress']:.3f} "
              f"ue_in={d['u_e_inner_over_U']:.4f} ue_far={d['u_e_far_over_U']:.4f} "
              f"os={d['overshoot_max_over_U']:.4f}@{d['overshoot_eta']} "
              f"H_U={d['H_U']:.3f} H_ue={d['H_ue']:.3f} l2={d['l2']:.3f}")
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()