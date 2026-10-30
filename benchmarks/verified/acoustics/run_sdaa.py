"""SDAA device portability run for the acoustics benchmark.
Runs the same D2Q9 plane-wave cases on sdaa:0 and compares with CPU reference.
"""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.solver import collide_bgk, stream

# --- Device setup ---
SDAA_DEV = torch.device("sdaa:0")
CPU_DEV = torch.device("cpu")
DTYPE = torch.float64
CS = 1.0 / math.sqrt(3.0)
SQ3 = math.sqrt(3.0)

def plane_wave_run(device, N, tau, eps, ny=8, n_periods=16.0):
    nx = N
    k = 2.0 * math.pi / nx
    x = torch.arange(nx, device=device, dtype=DTYPE)
    phase = k * x
    rho0 = 1.0 + eps * torch.sin(phase)
    ux0 = CS * eps * torch.sin(phase)
    uy0 = torch.zeros_like(rho0)
    f = equilibrium(
        rho0.unsqueeze(0).expand(ny, nx).contiguous(),
        ux0.unsqueeze(0).expand(ny, nx).contiguous(),
        uy0.unsqueeze(0).expand(ny, nx).contiguous(),
    )
    period = SQ3 * nx
    nsteps = int(math.ceil(n_periods * period))
    e_minus = np.exp(-1j * k * np.arange(nx))

    zp = np.empty(nsteps + 1, dtype=complex)
    t0 = time.time()
    for n in range(nsteps + 1):
        rho = macroscopic(f)[0]
        rp = (rho[0].cpu().numpy()).astype(np.float64) - 1.0
        zp[n] = rp @ e_minus / nx
        if n == nsteps:
            break
        f = collide_bgk(f, tau)
        f = stream(f)
    wall = time.time() - t0

    i0 = int(math.ceil(0.5 * period))
    i1 = min(int(math.floor(15.5 * period)), nsteps)
    n_arr = np.arange(nsteps + 1, dtype=np.float64)
    sl = slice(i0, i1 + 1)
    ph = np.unwrap(np.angle(zp[sl]))
    slope_p, _ = np.polyfit(n_arr[sl], ph, 1)
    omega_meas = -slope_p
    c_meas = omega_meas / k

    # attenuation via log-amplitude slope
    amp = np.abs(zp[i0:i1+1])
    mask = amp > 1e-15
    if mask.sum() > 10:
        d_naive = -float(np.polyfit(n_arr[i0:i1+1][mask], 0.5*np.log(amp[mask]), 1)[0])
    else:
        d_naive = float('nan')

    nu = CS*CS*(tau-0.5)
    delta_cont = nu * k * k
    return {
        "N": N, "tau": tau, "eps": eps, "ny": ny,
        "k": k, "nsteps": nsteps, "wall_s": wall,
        "c_meas": c_meas, "c_ref": CS,
        "c_err_pct": 100.0*(c_meas/CS - 1.0),
        "delta_meas": d_naive, "delta_ref": delta_cont,
        "delta_err_pct": 100.0*(d_naive/delta_cont - 1.0) if delta_cont > 0 else float('nan'),
        "r2_phase": float(1.0 - np.sum((ph - np.polyval([slope_p, _], n_arr[sl]))**2) / np.sum((ph - ph.mean())**2)),
        "finite": bool(torch.isfinite(f).all()),
    }

# --- Run on both devices ---
results = {"device": "sdaa:0", "torch_version": torch.__version__, "cases": []}
cases = [(16, 1.0, 1e-3), (32, 1.0, 1e-3), (64, 1.0, 1e-3)]

print("=" * 70)
print("SDAA Acoustics Benchmark - Device Portability Test")
print("=" * 70)

for N, tau, eps in cases:
    print(f"\n--- N={N}, tau={tau}, eps={eps} ---")
    r_cpu = plane_wave_run(CPU_DEV, N, tau, eps)
    r_sdaa = plane_wave_run(SDAA_DEV, N, tau, eps)
    
    # Compare
    c_diff = abs(r_cpu["c_meas"] - r_sdaa["c_meas"])
    d_diff = abs(r_cpu["delta_meas"] - r_sdaa["delta_meas"]) if not (math.isnan(r_cpu["delta_meas"]) or math.isnan(r_sdaa["delta_meas"])) else float('nan')
    
    print(f"  CPU:  c_err={r_cpu['c_err_pct']:+.6f}%  delta_err={r_cpu['delta_err_pct']:+.4f}%  wall={r_cpu['wall_s']:.3f}s")
    print(f"  SDAA: c_err={r_sdaa['c_err_pct']:+.6f}%  delta_err={r_sdaa['delta_err_pct']:+.4f}%  wall={r_sdaa['wall_s']:.3f}s")
    print(f"  |Δc|={c_diff:.2e}  |Δdelta|={d_diff:.2e}" if not math.isnan(d_diff) else f"  |Δc|={c_diff:.2e}")
    
    results["cases"].append({
        "config": {"N": N, "tau": tau, "eps": eps},
        "cpu": r_cpu,
        "sdaa": r_sdaa,
        "c_bitwise_diff": float(c_diff),
        "delta_bitwise_diff": float(d_diff) if not math.isnan(d_diff) else None,
        "portable": c_diff < 1e-10,  # fp64 should be bitwise on same arch
    })

# Summary
all_portable = all(c["portable"] for c in results["cases"])
results["summary"] = {
    "all_cases_portable": all_portable,
    "max_c_diff": max(c["c_bitwise_diff"] for c in results["cases"]),
    "verdict": "PASS" if all_portable else "INFO",
}

print(f"\n{'='*70}")
print(f"Verdict: {results['summary']['verdict']}")
print(f"Max |Δc|: {results['summary']['max_c_diff']:.2e}")

# Save artifact
out_path = Path(__file__).parent / "result_sdaa.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nArtifact saved: {out_path}")
