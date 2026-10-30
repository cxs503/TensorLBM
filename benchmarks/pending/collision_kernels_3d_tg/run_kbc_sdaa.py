#!/usr/bin/env python
"""Run KBC D3Q19 benchmark on SDAA to verify PR #312 fix."""
import sys, os, json, time, math
sys.path.insert(0, '/root/tp-agent/work/m443/src')
import torch
import torch_sdaa
import numpy as np

from tensorlbm.entropic_kbc import collide_kbc_d3q19
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import stream3d

HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE = 'sdaa:0'
DTYPE = torch.float32
RE = 24.0
U0 = 0.05
LADDER = (64, 96, 128)

def k_of(n): return 2.0 * math.pi / float(n)

def auto_steps(n, re, u0, record_every, target_efolds=10.0, cap=10000, min_steps=2000):
    nu = u0 * n / re
    k = 2.0 * math.pi / n
    gamma_e = 6.0 * nu * k * k
    steps = int(math.ceil(target_efolds / gamma_e / record_every) * record_every)
    return min(max(steps, min_steps), cap)

def run_case(n, steps, record_every=50):
    device = DEVICE
    dt = DTYPE
    k = k_of(n)
    nu = U0 * n / RE
    tau = 0.5 + 3.0 * nu
    gamma_e_theory = 6.0 * nu * k * k
    
    z, y, x = torch.meshgrid(
        torch.arange(n, device=device, dtype=dt),
        torch.arange(n, device=device, dtype=dt),
        torch.arange(n, device=device, dtype=dt),
        indexing='ij',
    )
    ux0 = U0 * torch.sin(k * x) * torch.cos(k * y) * torch.cos(k * z)
    uy0 = -U0 * torch.cos(k * x) * torch.sin(k * y) * torch.cos(k * z)
    uz0 = torch.zeros_like(ux0)
    rho = torch.ones_like(ux0)
    f = equilibrium3d(rho, ux0, uy0, uz0)
    
    e0_theory = U0 * U0 / 8.0
    ex0 = float((0.5 * (ux0 * ux0)).mean().item())
    ey0 = float((0.5 * (uy0 * uy0)).mean().item())
    
    times = [0]
    energies = [ex0 + ey0]
    wall0 = time.time()
    
    for step in range(1, steps + 1):
        f = collide_kbc_d3q19(stream3d(f), tau)
        if step % record_every == 0:
            _, uxm, uym, uzm = macroscopic3d(f)
            ex = float((0.5 * (uxm * uxm)).mean().item())
            ey = float((0.5 * (uym * uym)).mean().item())
            ez = float((0.5 * (uzm * uzm)).mean().item())
            times.append(step)
            energies.append(ex + ey + ez)
    
    wall = time.time() - wall0
    
    t = np.asarray(times[1:], dtype=np.float64)
    lnE = np.log(np.asarray(energies[1:], dtype=np.float64))
    a_e, b_e = np.polyfit(t, lnE, 1)
    gamma_e_sim = -a_e
    resid = lnE - (b_e + a_e * t)
    r2 = 1.0 - float(np.sum(resid**2) / np.sum((lnE - lnE.mean()) ** 2))
    
    err_e_pct = (gamma_e_sim - gamma_e_theory) / gamma_e_theory * 100.0
    
    return {
        'n': n, 'tau': tau, 'steps': steps, 'wall_sec': wall,
        'gamma_e_theory': gamma_e_theory, 'gamma_e_sim': gamma_e_sim,
        'err_e_pct': err_e_pct, 'r2': r2,
        'times': times, 'energies': energies,
    }

results = []
for n in LADDER:
    steps = auto_steps(n, RE, U0, 50)
    print(f"Running KBC D3Q19 N={n}, steps={steps}...")
    res = run_case(n, steps)
    print(f"  tau={res['tau']:.4f}, gamma_sim={res['gamma_e_sim']:.6e}, "
          f"gamma_theory={res['gamma_e_theory']:.6e}, err={res['err_e_pct']:+.4f}%, R2={res['r2']:.6f}")
    results.append(res)

# Save results
output = {
    'kernel': 'kbc_d3q19_fixed',
    'lattice': 'd3q19',
    'device': DEVICE,
    'dtype': 'float32',
    're': RE, 'u0': U0,
    'cases': results,
    'monotone': all(abs(results[i+1]['err_e_pct']) < abs(results[i]['err_e_pct']) for i in range(len(results)-1)),
}
with open(os.path.join(HERE, 'kbc_fixed_sdaa_results.json'), 'w') as f:
    json.dump(output, f, indent=2)
print("\nResults saved to kbc_fixed_sdaa_results.json")
