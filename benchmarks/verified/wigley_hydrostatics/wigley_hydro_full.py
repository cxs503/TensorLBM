"""Corrected full hydrostatic convergence of the library Wigley voxel mask."""
from __future__ import annotations
import math
import torch

A = {"C_B": 4 / 9, "C_wp": 2 / 3, "C_M": 2 / 3, "C_P": 2 / 3,
     "KB_over_T": 5 / 8, "IT_over_B3L": 4 / 105, "IL_over_L3B": 1 / 30}


def full_mask(nx, ny, nz, cx, cy, cz_keel, L, B, T, chunk=32, device="cpu"):
    zz = torch.arange(nz, device=device, dtype=torch.float32)
    yy = torch.arange(ny, device=device, dtype=torch.float32)
    xx = torch.arange(nx, device=device, dtype=torch.float32)
    x_norm = (xx - cx) / (L / 2.0)
    z_norm = (zz - (cz_keel + T)) / T
    in_length = x_norm.abs() <= 1.0
    in_draft = (z_norm >= -1.0) & (z_norm <= 0.0)
    rows = []
    for k0 in range(0, nz, chunk):
        k1 = min(k0 + chunk, nz)
        zc = z_norm[k0:k1].view(-1, 1, 1)
        hb = (B / 2.0) * (1.0 - x_norm.view(1, 1, nx) ** 2) * (1.0 - zc ** 2)
        valid = in_length.view(1, 1, nx) & in_draft[k0:k1].view(-1, 1, 1)
        hb = torch.where(valid, hb, torch.full_like(hb, -1.0))
        rows.append((yy.view(1, ny, 1) - cy).abs() <= hb)
    return torch.cat(rows, dim=0)


def measure(L, B, T, device="cpu"):
    margin = 1.0
    nx = int(math.ceil(L)) + 2 * int(math.ceil(margin * L)) + 2
    ny = int(math.ceil(B)) + 2 * int(math.ceil(margin * B)) + 3
    nz = int(math.ceil(T)) + 2 * int(math.ceil(margin * T)) + 2
    cx = nx / 2.0
    cy = float(int(round(ny / 2.0)))
    cz_keel = float(int(margin * T))
    m = full_mask(nx, ny, nz, cx, cy, cz_keel, L, B, T, device=device)
    mf = m.to(torch.float64)
    V = float(mf.sum())
    zz = torch.arange(nz, device=device, dtype=torch.float64).view(-1, 1, 1)
    yy = torch.arange(ny, device=device, dtype=torch.float64).view(1, -1, 1)
    xx = torch.arange(nx, device=device, dtype=torch.float64).view(1, 1, -1)
    sz = float((mf * zz).sum())
    sx = float((mf * xx).sum())
    # exposed top face
    exp = m.clone()
    exp[:-1] = m[:-1] & ~m[1:]
    exp[-1] = False
    ef = exp.to(torch.float64)
    A_wp = float(ef.sum())
    IT = float((ef * (yy - cy) ** 2).sum())
    IL = float((ef * (xx - cx) ** 2).sum())
    # midship section
    ix = int(round(cx))
    x_norm_i = (ix - cx) / (L / 2.0)
    z_norm = (torch.arange(nz, device=device, dtype=torch.float64) - (cz_keel + T)) / T
    hb_i = (B / 2.0) * (1.0 - x_norm_i ** 2) * (1.0 - z_norm ** 2)
    valid_i = (abs(x_norm_i) <= 1.0) & (z_norm >= -1.0) & (z_norm <= 0.0)
    hb_i = torch.where(valid_i, hb_i, torch.full_like(hb_i, -1.0))
    A_m = float(((yy.view(1, -1, 1) - cy).abs() <= hb_i.view(-1, 1, 1)).sum())
    r = dict(L=L, n=(nx, ny, nz), V=V, A_wp=A_wp, A_m=A_m,
             C_B=V / (L * B * T), C_wp=A_wp / (L * B), C_M=A_m / (B * T),
             C_P=V / (A_m * L), KB_over_T=(sz / V - cz_keel) / T,
             IT_over_B3L=IT / (B ** 3 * L), IL_over_L3B=IL / (L ** 3 * B),
             LCB_over_L=(sx / V - cx) / L)
    r["err_pct"] = {k: (r[k] - A[k]) / A[k] * 100 for k in A}
    return r


if __name__ == "__main__":
    for L in (80, 160, 320, 640, 960):
        r = measure(float(L), L / 10.0, L / 16.0)
        print(f"L={L:4d} {str(r['n']):>18s} " + " ".join(
            f"{k}={r[k]:.5g}({r['err_pct'][k]:+.2f}%)" for k in A), flush=True)