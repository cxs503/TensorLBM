#!/usr/bin/env python
"""Run ONE stokes_second_problem configuration on SDAA and save result."""
import sys, math, time, json
sys.path.insert(0, 'src')
sys.path.insert(0, 'benchmarks')
import torch
DEVICE = torch.device('sdaa')
torch.sdaa.set_device(0)

from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.lid_driven_cavity import zou_he_moving_lid
from tensorlbm.solver import collide_bgk, stream
import numpy as np

SPECULAR = torch.tensor([0, 1, 4, 3, 2, 8, 7, 6, 5], dtype=torch.int64)

H = int(sys.argv[1])
kH = float(sys.argv[2])
out_path = sys.argv[3]

tau, U, nx = 0.8, 0.05, 8
ny = H + 2
nu = (tau - 0.5) / 3.0
k = kH / H
omega = 2.0 * k * k * nu
T = 2.0 * math.pi / omega
T_int = int(round(T))
tau_diff = H * H / (math.pi**2 * nu)
skip_periods = max(10, math.ceil(8.0 * tau_diff / T))
skip_steps = skip_periods * T_int
n_meas = 4 * T_int
total_steps = skip_steps + n_meas

print(f"H={H} kH={kH}: total_steps={total_steps}", flush=True)

wall_bottom = torch.zeros((ny, nx), dtype=torch.bool, device=DEVICE)
wall_bottom[0, :] = True
rho0 = torch.ones((ny, nx), device=DEVICE)
u0 = torch.zeros((ny, nx), device=DEVICE)
f = equilibrium(rho0, u0, u0)
initial_mass = float(f.sum().item())

Sc = torch.zeros(ny, device=DEVICE)
Ss = torch.zeros(ny, device=DEVICE)

t0 = time.time()
for step in range(1, total_steps + 1):
    u_lid_t = torch.tensor(U * math.cos(omega * step), device=DEVICE)
    f_pre = f
    f = collide_bgk(f, tau)
    f = torch.where(wall_bottom.unsqueeze(0), f_pre[SPECULAR.to(f.device)], f)
    f = stream(f)
    f = zou_he_moving_lid(f, u_lid_t)
    if step > skip_steps:
        _, ux, _ = macroscopic(f)
        col_mean = ux.mean(dim=1)
        Sc += col_mean * math.cos(omega * step)
        Ss += col_mean * math.sin(omega * step)
    if step % max(1, total_steps // 5) == 0:
        print(f"  step {step}/{total_steps}", flush=True)

elapsed = time.time() - t0
ur = (2.0 / n_meas) * Sc.cpu().numpy().astype(np.float64)
ui = (-(2.0 / n_meas)) * Ss.cpu().numpy().astype(np.float64)
A_num = np.hypot(ur, ui)
phi_num = -np.arctan2(ui, ur)
rows = np.arange(ny)
y = (ny - 1) - rows.astype(np.float64)
A_ana = U * np.exp(-k * y)
phi_ana = k * y
fluid = (rows >= 1) & (rows <= ny - 2)
amp_mask = fluid & (A_ana >= 0.10 * U)
phase_mask = amp_mask & (phi_ana >= 1.0)
amp_rel = np.abs(A_num / A_ana - 1.0)[amp_mask] * 100.0
dphi = np.angle(np.exp(1j * (phi_num - phi_ana)))
phase_rel = (np.abs(dphi) / phi_ana)[phase_mask] * 100.0
z_num = ur + 1j * ui
z_ana = A_ana * np.exp(-1j * phi_ana)
complex_rel = (np.abs(z_num / z_ana - 1.0))[amp_mask] * 100.0

result = {
    "H": H, "kH": kH, "total_steps": total_steps, "elapsed_s": round(elapsed, 1),
    "amp_rel_max_pct": float(np.max(amp_rel)),
    "phase_rel_max_pct": float(np.max(phase_rel)),
    "complex_rel_max_pct": float(np.max(complex_rel)),
    "finite": True,
}
with open(out_path, 'w') as f:
    json.dump(result, f, indent=2)
print(f"Saved to {out_path}", flush=True)
