#!/usr/bin/env python3
"""Probe Series60 geometry: Cb, wetted areas (analytic + face-count), dims."""
import sys, math
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tensorlbm.ship_cad import (
    series60_hull_mask, hull_block_coefficient, hull_statistics,
    theoretical_block_coefficient, _series60_half_beam,
)

LX, LB, BT = 64, 6.5, 2.74
L = float(LX); B = L / LB; T = B / BT
print(f"L={L} B={B:.4f} T={T:.4f} L/B={LB} B/T={BT} L/T={L/T:.2f}")

# analytic Cb
hb = lambda x, z: _series60_half_beam(np.asarray(x, float), np.asarray(z, float))
xs = np.linspace(-1, 1, 4001); zs = np.linspace(0, 1, 2001)
X, Z = np.meshgrid(xs, zs, indexing="ij")
vol = np.trapezoid(np.trapezoid(hb(X, Z), zs, axis=1), xs) / (2.0)
print(f"analytic Cb (dbl int /2) = {vol:.5f}")

# analytic wetted areas, scaled by B/2 (=1 normalised halfbeam -> B/2) and L/2, T
# lateral surface (both sides): 2 * int sqrt(1 + dY/dx^2 + dY/dz^2) dx dz
x = np.linspace(-1, 1, 2001); z = np.linspace(0, 1, 1001)
Xg, Zg = np.meshgrid(x, z, indexing="ij")
Yg = hb(Xg, Zg)
dYdx = np.gradient(Yg, x, axis=0)
dYdz = np.gradient(Yg, z, axis=1)
# physical: Yphys = (B/2)*Yg ; xphys=(L/2)*xi ; zphys=T*zeta
dYdx_phys = (B / 2) * dYdx / (L / 2)
dYdz_phys = (B / 2) * dYdz / T
g = np.sqrt(1 + dYdx_phys**2 + dYdz_phys**2)
S_lat = 2 * np.trapezoid(np.trapezoid(g, z, axis=1) * T, x) * (L / 2)
S_deck = 2 * np.trapezoid(Yg[-1, :] if False else hb(x, np.ones_like(x)) * (B / 2), x) * (L / 2)
print(f"analytic S_lat(both sides)={S_lat:.5f}  S_deck={S_deck:.5f}  S_tot={S_lat+S_deck:.5f}")
print(f"  pi*D*L cylinder proxy (D=B): {math.pi*B*L:.5f}")
print(f"  ship formula L(2T+B)sqrt(Cb): {L*(2*T+B)*math.sqrt(0.6):.5f}")
print(f"  S_tot/L^2={ (S_lat+S_deck)/L**2:.5f}  S_lat/L^2={S_lat/L**2:.5f}")

# voxel face count area at a few resolutions
def face_area(LX, LB, BT):
    L = float(LX); B = L / LB; T = B / BT
    ny = int(round(L / LB * 3)); nz = int(round(L / BT / LB * 2.5))
    ny = max(ny, 24); nz = max(nz, 16)
    nx = LX + 40
    cx = nx * 0.25; cy = ny * 0.5; czk = nz * 0.5 - T / 2
    m = series60_hull_mask(nx, ny, nz, cx, cy, czk, L, B, T, torch.device("cpu"))
    s = m
    total = 0
    for dim in (0, 1, 2):
        a = torch.roll(s, 1, dim); b = torch.roll(s, -1, dim)
        total += int((s & ~a).sum()) + int((s & ~b).sum())
    # internal faces double count; only count where neighbor is fluid
    fn = 0
    for dim in (0, 1, 2):
        fn += int((s & ~torch.roll(s, 1, dim)).sum())
        fn += int((s & ~torch.roll(s, -1, dim)).sum())
    return fn, int(s.sum()), nx, ny, nz

for LX in (48, 64, 96):
    fn, ns, nx, ny, nz = face_area(LX, LB, BT)
    m = series60_hull_mask(nx, ny, nz, nx*0.25, ny*0.5, nz*0.5 - (float(LX)/LB/BT)/2,
                           float(LX), float(LX)/LB, float(LX)/LB/BT, torch.device("cpu"))
    cb = hull_block_coefficient(m, beam=float(LX)/LB, draft=float(LX)/LB/BT, length=float(LX))
    print(f"LX={LX}: grid {nx}x{ny}x{nz} solid={ns} face_count={fn} (S/ L^2={fn/float(LX)**2:.5f}) Cb_num={cb:.4f}")

print("stats:", hull_statistics("series60", L, B, T))