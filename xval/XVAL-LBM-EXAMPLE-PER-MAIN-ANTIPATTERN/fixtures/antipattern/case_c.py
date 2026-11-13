"""Control fixture: the anti-pattern -- a stand-alone main per case."""
import argparse
import json

import torch

from tensorlbm.d2q9 import C, OPPOSITE, W_EXACT


def build(ny, nx, tau, g):
    rho = torch.ones((ny, nx), dtype=torch.float64)
    ux = torch.zeros((ny, nx), dtype=torch.float64)
    uy = torch.zeros((ny, nx), dtype=torch.float64)
    c = C.to(torch.float64)
    cx = c[:, 0].view(9, 1, 1)
    cy = c[:, 1].view(9, 1, 1)
    w = torch.tensor(W_EXACT, dtype=torch.float64).view(9, 1, 1)
    cu = cx * ux + cy * uy
    f = w * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * (ux * ux + uy * uy))
    closed = torch.zeros((9, ny, nx), dtype=torch.bool)
    for i in range(9):
        if int(C[i, 1]) == 0:
            continue
        src = torch.arange(ny).view(ny, 1).expand(ny, nx) - int(C[i, 1])
        closed[i] = ((src == 0) | (src == ny - 1)) & (src >= 0) & (src <= ny - 1)
    return f, closed, w, cx, cy


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ny", type=int, default=32)
    ap.add_argument("--nx", type=int, default=8)
    ap.add_argument("--tau", type=float, default=0.9)
    ap.add_argument("--g", type=float, default=1e-5)
    ap.add_argument("--steps", type=int, default=12000)
    return ap.parse_args()


def main():
    ns = parse()
    f, closed, w, cx, cy = build(ns.ny, ns.nx, ns.tau, ns.g)
    for _ in range(ns.steps):
        rho = f.sum(0)
        ux = (f * cx).sum(0) / rho
        uy = (f * cy).sum(0) / rho
        ux_eq = ux + 0.5 * ns.g / rho
        cu = cx * ux_eq + cy * uy
        feq = w * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * (ux_eq * ux_eq + uy * uy))
        src = (1.0 - 0.5 / ns.tau) * (
            3.0 * w * ((cx - ux_eq) * ns.g) + 9.0 * w * (cu * (cx * ns.g))
        )
        fp = f - (f - feq) / ns.tau + src
        new = torch.empty_like(fp)
        for i in range(9):
            new[i] = torch.roll(fp[i], shifts=(int(C[i, 1]), int(C[i, 0])), dims=(0, 1))
        for i in range(1, 9):
            if closed[i].any():
                new[i][closed[i]] = fp[OPPOSITE[i]][closed[i]]
        f = new
    rho = f.sum(0)
    ux = (f * cx).sum(0) / rho + 0.5 * ns.g / rho
    print(json.dumps({"u_max": float(ux[ns.ny // 2, 0].abs())}))


if __name__ == "__main__":
    main()
