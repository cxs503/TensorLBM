"""Faithful torch port of lbm-principles-practice chapter9/shanchen.cpp (nfluids=1)."""
import sys, math, time; sys.path.insert(0, "src")
import torch; torch.set_default_dtype(torch.float64)
from tensorlbm.d2q9 import C, W, equilibrium

CX = [int(c[0]) for c in C]; CY = [int(c[1]) for c in C]


def psi(rho, rho0=1.0):
    return rho0 * (1.0 - torch.exp(-rho / rho0))


def init_droplet(n=64, R=15.0, rhol=2.1, rhog=0.15):
    yy, xx = torch.meshgrid(torch.arange(n, dtype=torch.float64),
                            torch.arange(n, dtype=torch.float64), indexing="ij")
    inside = ((yy - n / 2.) ** 2 + (xx - n / 2.) ** 2) <= R * R
    rho = torch.where(inside, torch.tensor(rhol, dtype=torch.float64),
                      torch.tensor(rhog, dtype=torch.float64))
    return rho, torch.zeros_like(rho), torch.zeros_like(rho)


def sc_forces(rho, g, rho0=1.0):
    ph = psi(rho, rho0)
    fx = torch.zeros_like(rho); fy = torch.zeros_like(rho)
    for i in range(1, 9):
        sh = torch.roll(ph, shifts=(-CY[i], -CX[i]), dims=(0, 1))
        fx = fx + W[i] * CX[i] * sh
        fy = fy + W[i] * CY[i] * sh
    return -g * ph * fx, -g * ph * fy


def velocity(f, Fx, Fy):
    rho = f.sum(0)
    ux = (f * C[:, 0].view(9, 1, 1)).sum(0) + 0.5 * Fx
    uy = (f * C[:, 1].view(9, 1, 1)).sum(0) + 0.5 * Fy
    return rho, ux / rho, uy / rho


def push(f, fx, fy, tau):
    rho, ux, uy = velocity(f, fx, fy)
    omega = 1.0 / tau
    feq = equilibrium(rho, ux, uy)
    cx = C[:, 0].view(9, 1, 1); cy = C[:, 1].view(9, 1, 1)
    w = W.view(9, 1, 1)
    cu = cx * ux.unsqueeze(0) + cy * uy.unsqueeze(0)
    Fx = fx.unsqueeze(0); Fy = fy.unsqueeze(0)
    Ux = ux.unsqueeze(0); Uy = uy.unsqueeze(0)
    # Guo source, written exactly as in the reference (3*(cx-ux)+9*cx*cu)*Fx + (...)*Fy
    forcing = w * (1.0 - 0.5 * omega) * (
        (3.0 * (cx - Ux) + 9.0 * cx * cu) * Fx + (3.0 * (cy - Uy) + 9.0 * cy * cu) * Fy)
    f2 = f * (1.0 - omega) + feq * omega + forcing
    out = torch.empty_like(f2)
    for i in range(9):
        out[i] = torch.roll(f2[i], shifts=(CY[i], CX[i]), dims=(0, 1))  # push towards x+c
    return out


def pressure(rho, g, rho0=1.0):
    return rho / 3.0 + g * psi(rho, rho0) ** 2 / 6.0


def laplace(rho, g):
    p = pressure(rho, g); n = rho.shape[0]
    rho_g = float(rho[:4, :4].mean()); rho_l = float(rho[n // 2 - 2:n // 2 + 2, n // 2 - 2:n // 2 + 2].mean())
    dp = float(p[n // 2 - 2:n // 2 + 2, n // 2 - 2:n // 2 + 2].mean()) - float(p[:4, :4].mean())
    rho_av = 0.5 * (rho_g + rho_l); prof = rho[n // 2]; rad = float("nan")
    for x in range(0, n // 2):
        if float(prof[x]) > rho_av:
            drho = float(prof[x] - prof[x - 1]); dx = (rho_av - float(prof[x - 1])) / drho
            rad = (n / 2. - x) + (1. - dx); break
    return dict(rho_g=rho_g, rho_l=rho_l, dp=dp, r=rad, gamma=dp * rad)


def run(n=64, R=15.0, rhol=2.1, rhog=0.15, g=-4.7, tau=1.0, steps=20000, rep=1000):
    rho, ux, uy = init_droplet(n, R, rhol, rhog)
    f = equilibrium(rho, ux, uy)
    t0 = time.time()
    for s in range(1, steps + 1):
        rr = f.sum(0); fx, fy = sc_forces(rr, g)
        f = push(f, fx, fy, tau)
        if not torch.isfinite(f).all():
            print(f"refport DIV at {s}"); return None
        if s % rep == 0:
            r = f.sum(0); L = laplace(r, g)
            _, vx, vy = velocity(f, *sc_forces(r, g))
            print(f"  s={s:6d} rg={L['rho_g']:.4f} rl={L['rho_l']:.4f} dp={L['dp']:+.5e} "
                  f"r={L['r']:.3f} gamma={L['gamma']:+.5e} "
                  f"umax={float(torch.sqrt(vx**2+vy**2).max()):.4f} ({time.time()-t0:.0f}s)", flush=True)
    return f


if __name__ == "__main__":
    import os
    run(steps=int(os.environ.get("STEPS", "300")), rep=int(os.environ.get("REP", "100")),
        g=float(os.environ.get("G", "-4.7")))