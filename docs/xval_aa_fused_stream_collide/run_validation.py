#!/usr/bin/env python3
"""XVAL-LBM-AA-PATTERN-FUSED-STREAM-COLLIDE — comprehensive cross-validation."""
import sys, os, json, time, math
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'src'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'TensorLBM', 'src'))

import torch
torch.set_default_dtype(torch.float64)
import numpy as np

from tensorlbm import (
    aa_fused_stream_collide, separate_stream_collide, aa_fused_step, separate_step,
)
from tensorlbm import (
    fused_push_pull_step, analytic_poiseuille, analytic_couette, channel_masks, macroscopic,
)
from tensorlbm import (
    taylor_green_fields, equilibrium_upstream, tau_from_nu, tg_l2_errors,
)

C9 = torch.tensor([(0,0),(1,0),(0,1),(-1,0),(0,-1),(1,1),(-1,1),(-1,-1),(1,-1)], dtype=torch.float64)
W9 = [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]
GATE = 0.15

GHIA_U = np.array([[0,.0],[.0547,-.03717],[.0625,-.04192],[.0938,-.06376],[.1563,-.10618],[.2292,-.14950],[.3125,-.18919],[.3542,-.20649],[.5,-.23626],[.6146,-.20186],[.6875,-.15743],[.75,-.10372],[.8021,-.05653],[.8542,-.01653],[.9063,.02135],[.9375,.05185],[.9531,.07411],[.9688,.10417],[.9766,.12380],[1,.16449]])
GHIA_V = np.array([[0,.0],[.0625,.09233],[.0938,.10091],[.1094,.10272],[.1563,.09445],[.2344,.06406],[.3281,.02792],[.3906,.00523],[.4531,-.01587],[.5,-.02587],[.5469,-.03171],[.6094,-.02927],[.6797,-.01859],[.7578,-.00369],[.8125,.00960],[.8594,.02526],[.8906,.03933],[.9297,.06134],[.9609,.08486],[1,.17907]])

def profile_error(numeric_lattice, y_star, ref, U, interior=False):
    num = np.asarray(numeric_lattice, dtype=float) / U
    ys = np.asarray(y_star, dtype=float)
    vals = np.interp(ref[:, 0], ys, num)
    vals = np.where(ref[:, 0] >= ys[-1], num[-1], vals)
    sel = (ref[:, 0] > 0.0) & (ref[:, 0] < 1.0) if interior else np.ones(len(ref), bool)
    return float(np.sqrt(np.mean((vals[sel] - ref[sel, 1]) ** 2)))

results = {"anchor": "XVAL-LBM-AA-PATTERN-FUSED-STREAM-COLLIDE", "gate": GATE}

# 1. Bit-identity
print("=== 1. Bit-identity ===")
ny, nx = 32, 32; f = torch.randn(9, ny, nx) * 0.01; f[0] = 1.0 + f[0]; tau = 0.8
f1, f2 = f.clone(), f.clone(); diffs = []
for _ in range(200):
    f1 = aa_fused_stream_collide(f1, tau); f2 = separate_stream_collide(f2, tau)
    diffs.append((f1 - f2).abs().max().item())
results["bit_identity"] = {"max_abs_diff_200steps": max(diffs), "bit_identical": max(diffs) == 0.0}
print(f"  max|diff|={max(diffs):.2e}, bit-identical={max(diffs)==0.0}")

# 2. Taylor-Green
print("\n=== 2. Taylor-Green ===")
nu = 1./6.; tau = tau_from_nu(nu); u_max = 0.02; rho0 = 1.0; tg = []
for scale in [2, 4]:
    nx = 32*scale; ny = nx; nsteps = 200*scale*scale; nsave = 50*scale*scale
    rho, ux, uy = taylor_green_fields(ny, nx, u_max=u_max, nu=nu, rho0=rho0, t=0)
    f = equilibrium_upstream(rho, ux, uy); t0 = time.time(); samples = []
    for n in range(nsteps):
        f = aa_fused_stream_collide(f, tau)
        if (n+1) % nsave == 0:
            rho_f = f.sum(0); ux_f = (f*C9[:,0].view(-1,1,1)).sum(0)/rho_f; uy_f = (f*C9[:,1].view(-1,1,1)).sum(0)/rho_f
            err = tg_l2_errors(rho_f, ux_f, uy_f, nx=nx, ny=ny, u_max=u_max, nu=nu, t=n+1, rho0=rho0)
            samples.append({'t': n+1, **{k: float(v) for k, v in err.items()}})
    elapsed = time.time() - t0; final = samples[-1]
    tg.append({"scale": scale, "NX": nx, "nsteps": nsteps, "elapsed_s": elapsed, "final_errors": final})
    print(f"  scale={scale} NX={nx}: energy={final['energy']:.4e}, ux={final['ux']:.4e}, uy={final['uy']:.4e}")
results["taylor_green"] = tg

# 3. Couette
print("\n=== 3. Couette ===")
couette = []
for L in [16, 32, 64]:
    U = 0.05; tau = 0.8; nu = (tau-0.5)/3.; n = L+2
    solid = torch.zeros((n,1), dtype=torch.bool); solid[0,0]=True; solid[-1,0]=True
    f = torch.zeros((9,n,1)); f[0]=4./9.
    for i in range(1,9): f[i]=W9[i]
    steps = max(4000, L*200); t0 = time.time()
    for _ in range(steps): f = aa_fused_step(f, tau, solid, u_lid=U, lid_row=-1)
    elapsed = time.time()-t0
    rho_f = f.sum(0); ux_f = (f*C9[:,0].view(-1,1,1)).sum(0)/rho_f
    u_num = ux_f[1:L+1,0].numpy(); u_ana = U*(np.arange(1,L+1)-0.5)/L
    rel_err_max = float(np.abs(u_num-u_ana).max()/U)
    rel_err_l2 = float(np.sqrt(np.mean((u_num-u_ana)**2))/U)
    couette.append({"L":L,"U":U,"tau":tau,"nu":nu,"steps":steps,"rel_err_max":rel_err_max,"rel_err_l2":rel_err_l2,"elapsed_s":elapsed,"gate_1pct":rel_err_max<0.01})
    print(f"  L={L}: rel_err_max={rel_err_max:.2e}, gate_1%={'PASS' if rel_err_max<0.01 else 'FAIL'}")
results["couette"] = couette

# 4. Poiseuille
print("\n=== 4. Poiseuille ===")
NU_P=0.05; G_P=1e-6; TAU_P=3.*NU_P+0.5; pois = []
for n_fluid in [8, 16, 24, 48]:
    nx=4; ny=n_fluid+2; solid=channel_masks(nx,ny,1); fluid=~solid
    force=torch.zeros((2,ny,nx),dtype=torch.float64); force[0]=G_P
    f=torch.zeros((9,ny,nx),dtype=torch.float64); f[0]=4./9.
    for i in range(1,9): f[i]=W9[i]
    steps=max(2500,n_fluid*200); t0=time.time()
    for _ in range(steps): f=fused_push_pull_step(f,TAU_P,solid=solid,force=force)
    elapsed=time.time()-t0
    _,ux,uy=macroscopic(f,force); prof=ux[fluid].reshape(n_fluid,nx).mean(1)
    analytic,u_max=analytic_poiseuille(n_fluid,G_P,NU_P)
    u_max_rel_err=float(abs(prof.max()-u_max)/u_max)
    profile_rel_l2=float((prof-analytic).norm()/analytic.norm())
    pois.append({"n_fluid":n_fluid,"steps":steps,"u_max_rel_err":u_max_rel_err,"profile_rel_l2":profile_rel_l2,"elapsed_s":elapsed,"gate_15pct":u_max_rel_err<GATE})
    print(f"  n={n_fluid}: u_max_err={100*u_max_rel_err:.2f}%, L2={100*profile_rel_l2:.2f}%, gate_15%={'PASS' if u_max_rel_err<GATE else 'FAIL'}")
results["poiseuille"] = pois

# 5. Cavity
print("\n=== 5. Cavity vs Ghia Re=100 ===")
cavity = []
for L in [32, 64]:
    U=0.1; Re=100.; tau=0.5+3.*(U*L/Re); nu=(tau-0.5)/3.; n=L+2
    solid=torch.zeros((n,n),dtype=torch.bool); solid[0,:]=True; solid[-1,:]=True; solid[:,0]=True; solid[:,-1]=True
    fluid=~solid; f=torch.zeros((9,n,n)); f[0]=4./9.
    for i in range(1,9): f[i]=W9[i]
    max_steps=max(20000,L*500); monitor=2000; ramp=2000; min_steps=20000
    t0=time.time(); converged_at=None; prev_u=None; stable_hits=0
    for s in range(1,max_steps+1):
        u_lid=U*min(1.,s/ramp) if ramp else U
        f=aa_fused_step(f,tau,solid,u_lid=u_lid,lid_row=-1)
        if s%monitor==0:
            rho_f=f.sum(0); ux_f=(f*C9[:,0].view(-1,1,1)).sum(0)/rho_f; uy_f=(f*C9[:,1].view(-1,1,1)).sum(0)/rho_f
            u=np.stack([ux_f.numpy(),uy_f.numpy()])
            res=float("nan") if prev_u is None else float(np.abs(u-prev_u).max()/U)
            wt=1; ys=(np.arange(L)+0.5)/L
            uxc=ux_f[wt:wt+L,n//2].numpy(); uyc=uy_f[n//2,wt:wt+L].numpy()
            L2_u=profile_error(uxc,ys,GHIA_U,U); L2_v=profile_error(uyc,ys,GHIA_V,U)
            L2_u_int=profile_error(uxc,ys,GHIA_U,U,interior=True); L2_v_int=profile_error(uyc,ys,GHIA_V,U,interior=True)
            prev_u=u
            if s>=min_steps and np.isfinite(res) and res<1e-8:
                stable_hits+=1
                if stable_hits>=2: converged_at=s; break
            else: stable_hits=0
    elapsed=time.time()-t0
    rho_f=f.sum(0); ux_f=(f*C9[:,0].view(-1,1,1)).sum(0)/rho_f; uy_f=(f*C9[:,1].view(-1,1,1)).sum(0)/rho_f
    wt=1; ys=(np.arange(L)+0.5)/L
    uxc=ux_f[wt:wt+L,n//2].numpy(); uyc=uy_f[n//2,wt:wt+L].numpy()
    L2_u=profile_error(uxc,ys,GHIA_U,U); L2_v=profile_error(uyc,ys,GHIA_V,U)
    L2_u_int=profile_error(uxc,ys,GHIA_U,U,interior=True); L2_v_int=profile_error(uyc,ys,GHIA_V,U,interior=True)
    mass_drift=float(abs(f[:,fluid].sum()-(4./9.+8./9.)*fluid.sum())/((4./9.+8./9.)*fluid.sum()))
    cavity.append({"L":L,"U":U,"Re":Re,"tau":tau,"nu":nu,"steps_run":s,"converged_at":converged_at,"elapsed_s":elapsed,
        "L2_u_full":L2_u,"L2_v_full":L2_v,"L2_u_interior":L2_u_int,"L2_v_interior":L2_v_int,
        "fluid_mass_drift_rel":mass_drift,"gate_15pct_u":L2_u<GATE,"gate_15pct_v":L2_v<GATE,
        "gate_3pct_u_int":L2_u_int<0.03,"gate_3pct_v_int":L2_v_int<0.03})
    print(f"  L={L}: L2_u={L2_u:.4f}, L2_v={L2_v:.4f}, L2_u_int={L2_u_int:.4f}, L2_v_int={L2_v_int:.4f}, gate_15%={'PASS' if L2_u<GATE and L2_v<GATE else 'FAIL'}")
results["cavity"] = cavity

# Summary
print("\n=== SUMMARY ===")
all_pass = True
for name, rl in [("couette",couette),("poiseuille",pois),("cavity",cavity)]:
    for r in rl:
        if name=="couette": ok=r["gate_1pct"]; print(f"  {name} L={r['L']}: {'PASS' if ok else 'FAIL'} (err={r['rel_err_max']:.2e})")
        elif name=="poiseuille": ok=r["gate_15pct"]; print(f"  {name} n={r['n_fluid']}: {'PASS' if ok else 'FAIL'} (err={100*r['u_max_rel_err']:.2f}%)")
        elif name=="cavity": ok=r["gate_15pct_u"] and r["gate_15pct_v"]; print(f"  {name} L={r['L']}: {'PASS' if ok else 'FAIL'} (L2_u={r['L2_u_full']:.4f}, L2_v={r['L2_v_full']:.4f})")
        if not ok: all_pass = False
results["overall_gate_15pct"] = all_pass
print(f"\nOverall <15% gate: {'PASS' if all_pass else 'FAIL'}")

out_path = os.path.join(HERE, 'result.json')
with open(out_path, 'w') as fp: json.dump(results, fp, indent=2, default=str)
print(f"Saved to {out_path}")
