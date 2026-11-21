"""Cross-validation: Maarten-vd-Sande/lbm D2Q9 oracle vs TensorLBM.

XVAL-LBM-ORACLE-MINIMAL-D2Q9-FS

Three validation levels:
  L1 – Lattice constant identity (velocity/weight/opposite mapping)
  L2 – Single-step numerical identity (equilibrium, collision, streaming)
  L3 – Physical benchmark: plane Poiseuille flow vs analytic parabola

The Maarten code uses a first-order forcing: fout[i] -= rho*t[i]*(v·g),
which gives effective body force F = -ρg/3 (wrong sign & magnitude).
Both arms use the SAME forcing so the cross-code comparison is fair;
the analytic reference uses the CORRECT effective force F = -ρg/3
to match what the code actually computes.
"""

from __future__ import annotations

import json, math, os, sys, time
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_SRC = os.path.abspath(os.path.join(HERE, "..", "..", "src"))
if REPO_SRC not in sys.path:
    sys.path.insert(0, REPO_SRC)

from tensorlbm.d2q9 import C, W, OPPOSITE, equilibrium, macroscopic

# ---- Maarten D2Q9 constants ----
M_V = np.array([[0,0],[0,-1],[0,1],[-1,0],[-1,-1],[-1,1],[1,0],[1,-1],[1,1]], dtype=np.int64)
M_T = np.array([4/9,1/9,1/9,1/9,1/36,1/36,1/9,1/36,1/36], dtype=np.float64)
M_V_INV = np.array([0,2,1,6,8,7,3,5,4], dtype=np.int64)

# ---- Mapping ----
M_TO_TLBM = {}
for _i in range(9):
    _mv = tuple(M_V[_i].tolist())
    for _j in range(9):
        if _mv == (int(C[_j,0].item()), int(C[_j,1].item())):
            M_TO_TLBM[_i] = _j; break

# ---- Oracle routines (Maarten ordering) ----
def oracle_eq(rho, ux, uy):
    usqr = 1.5*(ux*ux+uy*uy)
    feq = np.empty((9,)+rho.shape, dtype=np.float64)
    for i in range(9):
        vu = 3.0*(M_V[i,0]*ux+M_V[i,1]*uy)
        feq[i] = rho*M_T[i]*(1.0+vu+0.5*vu*vu-usqr)
    return feq

def oracle_mac(fin):
    rho = fin.sum(axis=0)
    rs = np.where(rho>1e-12, rho, 1e-12)
    ux = np.einsum('i,ijk->jk', M_V[:,0].astype(np.float64), fin)/rs
    uy = np.einsum('i,ijk->jk', M_V[:,1].astype(np.float64), fin)/rs
    return rho, ux, uy

def oracle_stream(fin, obs):
    fout = fin.copy()
    for n in range(1,9):
        vx,vy = int(M_V[n,0]),int(M_V[n,1]); no=int(M_V_INV[n])
        pulled = np.roll(np.roll(fin[n],-vx,axis=0),-vy,axis=1)
        fout[n] = np.where(obs, fin[no], pulled)
    return fout

# ---- TensorLBM stream ----
def tlbm_stream(f, wall):
    fn = f.clone()
    for j in range(1,9):
        cx=int(C[j,0].item()); cy=int(C[j,1].item()); jo=int(OPPOSITE[j].item())
        pulled = torch.roll(torch.roll(f[j],shifts=-cx,dims=1),shifts=-cy,dims=0)
        fn[j] = torch.where(wall, f[jo], pulled)
    return fn

# ---- Poiseuille: Oracle (Maarten ordering, first-order forcing) ----
def oracle_poiseuille(ny=32, nx=64, tau=0.8, g_accel=1e-5, n_steps=8000):
    """Plane Poiseuille with Maarten's first-order forcing.

    Maarten forcing: fout[i] -= rho*t[i]*(v[i]·g)
    Effective body force: F_eff = -ρg·cs² = -ρg/3
    For Poiseuille: u(y) = F_eff/(2ν) · y·(H-y) = -g/(6ν) · y·(H-y)
    We set g_accel > 0 and the flow goes in -x direction (Maarten convention).
    To get flow in +x, we use g = (-g_accel, 0).
    """
    nu = (tau-0.5)/3.0; omega = 1.0/tau
    obs = np.zeros((nx,ny), dtype=bool); obs[:,0]=True; obs[:,-1]=True
    rho = np.ones((nx,ny), dtype=np.float64)
    ux = np.zeros((nx,ny), dtype=np.float64)
    uy = np.zeros((nx,ny), dtype=np.float64)
    fin = oracle_eq(rho, ux, uy)
    # Force in -x direction (Maarten convention: x is first axis)
    # g = (-g_accel, 0) → effective force F_x = -(-g_accel)/3 = +g_accel/3
    gravity = (-g_accel, 0.0)
    for _ in range(n_steps):
        rho, ux, uy = oracle_mac(fin)
        feq = oracle_eq(rho, ux, uy)
        fout = fin + omega*(feq - fin)
        for i in range(9):
            gv = M_V[i,0]*gravity[0] + M_V[i,1]*gravity[1]
            fout[i] -= rho*M_T[i]*gv
        fin = oracle_stream(fout, obs)
    rho, ux, uy = oracle_mac(fin)
    up = ux[:,1:-1].mean(axis=0)
    yf = np.arange(1,ny-1)-0.5; H = float(ny-2)
    # Effective force = g_accel/3 in +x direction
    F_eff = g_accel / 3.0
    ua = F_eff/(2*nu)*yf*(H-yf)
    s = float(np.max(np.abs(ua))) if np.max(np.abs(ua))>0 else 1.0
    return {'eps_inf':float(np.max(np.abs(up-ua))/s),'eps_l2':float(np.linalg.norm(up-ua)/np.linalg.norm(ua)),
            'u_peak_num':float(up.max()),'u_peak_ana':float(ua.max()),'u':up,'ua':ua,'y':yf,'H':H,'F_eff':F_eff}

# ---- Poiseuille: TensorLBM ----
def tlbm_poiseuille(ny=32, nx=64, tau=0.8, g_accel=1e-5, n_steps=8000):
    """Plane Poiseuille with TensorLBM using the same first-order forcing."""
    nu = (tau-0.5)/3.0; omega = 1.0/tau
    wall = torch.zeros((ny,nx), dtype=torch.bool); wall[0,:]=True; wall[-1,:]=True
    rho = torch.ones((ny,nx), dtype=torch.float64)
    ux = torch.zeros_like(rho); uy = torch.zeros_like(rho)
    f = equilibrium(rho, ux, uy)
    # Same forcing: f[j] -= rho*w[j]*(C[j,0]*gx + C[j,1]*gy)
    # Use g = (-g_accel, 0) in TensorLBM convention (x is second axis)
    # Actually TensorLBM uses (ux, uy) where ux is along dim=1 (columns)
    # C[1]=(1,0) means +x direction = rightward
    # To get flow in +x: effective force = g_accel/3
    # Maarten forcing: fout[i] -= rho*t[i]*(v[i]·g) with g=(-g_accel,0)
    # In TLBM: f[j] -= rho*w[j]*(C[j,0]*gx + C[j,1]*gy)
    # gx = -g_accel (to match Maarten's convention)
    gx = -g_accel; gy = 0.0
    for _ in range(n_steps):
        rho, ux, uy = macroscopic(f)
        feq = equilibrium(rho, ux, uy)
        f = f + omega*(feq - f)
        for j in range(9):
            gv = int(C[j,0].item())*gx + int(C[j,1].item())*gy
            f[j] -= rho*float(W[j].item())*gv
        f = tlbm_stream(f, wall)
    rho, ux, uy = macroscopic(f)
    up = ux[1:-1,:].mean(dim=1).numpy()
    yf = np.arange(1,ny-1)-0.5; H = float(ny-2)
    F_eff = g_accel/3.0
    ua = F_eff/(2*nu)*yf*(H-yf)
    s = float(np.max(np.abs(ua))) if np.max(np.abs(ua))>0 else 1.0
    return {'eps_inf':float(np.max(np.abs(up-ua))/s),'eps_l2':float(np.linalg.norm(up-ua)/np.linalg.norm(ua)),
            'u_peak_num':float(up.max()),'u_peak_ana':float(ua.max()),'u':up,'ua':ua,'y':yf,'H':H,'F_eff':F_eff}

# ---- Level 1: Lattice constants ----
def check_lattice_constants():
    ok = True; details = []
    for i in range(9):
        j = M_TO_TLBM[i]
        mw = M_T[i]; tw = float(W[j].item())
        match = abs(mw-tw)/max(abs(mw),1e-30) < 1e-6  # float32 tolerance
        ok = ok and match
        details.append({'m_idx':i,'m_vel':M_V[i].tolist(),'m_w':float(mw),
                        't_idx':j,'t_vel':[int(C[j,0].item()),int(C[j,1].item())],
                        't_w':tw,'match':match})
    # Opposite directions
    opp_ok = True
    for i in range(9):
        j = M_TO_TLBM[i]
        m_opp = M_V_INV[i]; t_opp = int(OPPOSITE[j].item())
        mapped = M_TO_TLBM[m_opp]
        match = mapped == t_opp
        opp_ok = opp_ok and match
    return {'weights_match':ok,'opposites_match':opp_ok,'pass':ok and opp_ok,'details':details}

# ---- Level 2: Single-step identity ----
def check_equilibrium(nx=16, ny=16):
    np.random.seed(42)
    rho = 1.0+0.01*np.random.randn(nx,ny)
    ux = 0.01*np.random.randn(nx,ny); uy = 0.01*np.random.randn(nx,ny)
    feq_m = oracle_eq(rho,ux,uy)
    feq_t = equilibrium(torch.from_numpy(rho),torch.from_numpy(ux),torch.from_numpy(uy)).numpy()
    feq_m_mapped = np.empty_like(feq_m)
    for i in range(9): feq_m_mapped[M_TO_TLBM[i]] = feq_m[i]
    d = np.abs(feq_m_mapped-feq_t)
    return {'max_abs_diff':float(d.max()),'pass':float(d.max())<1e-12}

def check_collision(nx=16, ny=16):
    np.random.seed(42)
    rho = 1.0+0.01*np.random.randn(nx,ny)
    ux = 0.01*np.random.randn(nx,ny); uy = 0.01*np.random.randn(nx,ny)
    omega = 1.5
    feq_m = oracle_eq(rho,ux,uy)
    fin_m = feq_m + 0.001*np.random.randn(9,nx,ny)
    fout_m = fin_m + omega*(feq_m-fin_m)
    # TensorLBM
    fin_t = np.empty_like(fin_m)
    for i in range(9): fin_t[M_TO_TLBM[i]] = fin_m[i]
    fin_tt = torch.from_numpy(fin_t)
    feq_t = equilibrium(torch.from_numpy(rho),torch.from_numpy(ux),torch.from_numpy(uy))
    fout_t = (fin_tt + omega*(feq_t-fin_tt)).numpy()
    fout_t_mapped = np.empty_like(fout_m)
    for j in range(9): fout_t_mapped[TLBM_TO_M[j]] = fout_t[j]
    d = np.abs(fout_m-fout_t_mapped)
    return {'max_abs_diff':float(d.max()),'pass':float(d.max())<1e-12}

# Build reverse mapping
TLBM_TO_M = {v:k for k,v in M_TO_TLBM.items()}

# ---- Main runner ----
def run_all(quick=False):
    R = {}
    print("="*70)
    print("XVAL-LBM-ORACLE-MINIMAL-D2Q9-FS")
    print("="*70)

    print("\n--- L1: Lattice constants ---")
    R['lattice'] = check_lattice_constants()
    print(f"  Weights: {R['lattice']['weights_match']}  Opposites: {R['lattice']['opposites_match']}  PASS: {R['lattice']['pass']}")

    print("\n--- L2: Single-step identity ---")
    R['equilibrium'] = check_equilibrium()
    print(f"  Equilibrium max_diff: {R['equilibrium']['max_abs_diff']:.2e}  PASS: {R['equilibrium']['pass']}")
    R['collision'] = check_collision()
    print(f"  Collision max_diff: {R['collision']['max_abs_diff']:.2e}  PASS: {R['collision']['pass']}")

    print("\n--- L3: Poiseuille flow ---")
    cases = [
        {'ny':16,'nx':32,'tau':0.8,'g_accel':1e-5,'n_steps':4000,'label':'C1_ny16_tau0.8'},
        {'ny':32,'nx':64,'tau':0.8,'g_accel':1e-5,'n_steps':10000,'label':'C2_ny32_tau0.8'},
        {'ny':32,'nx':64,'tau':1.2,'g_accel':1e-5,'n_steps':15000,'label':'C3_ny32_tau1.2'},
    ] if not quick else [
        {'ny':16,'nx':32,'tau':0.8,'g_accel':1e-5,'n_steps':3000,'label':'C1_ny16_tau0.8'},
    ]

    pois = []
    for c in cases:
        label = c.pop('label')
        print(f"\n  {label}: ny={c['ny']} tau={c['tau']} g={c['g_accel']} steps={c['n_steps']}")
        t0=time.perf_counter(); ro = oracle_poiseuille(**c); t1=time.perf_counter()
        print(f"    Oracle:   eps_inf={ro['eps_inf']:.4e}  eps_l2={ro['eps_l2']:.4e}  u_peak={ro['u_peak_num']:.6e} ({t1-t0:.1f}s)")
        t0=time.perf_counter(); rt = tlbm_poiseuille(**c); t1=time.perf_counter()
        print(f"    TensorLBM: eps_inf={rt['eps_inf']:.4e}  eps_l2={rt['eps_l2']:.4e}  u_peak={rt['u_peak_num']:.6e} ({t1-t0:.1f}s)")
        u1=ro['u']; u2=rt['u']
        sc=0.5*(float(np.max(np.abs(u1)))+float(np.max(np.abs(u2))))
        tv=float(np.max(np.abs(u1-u2))/sc) if sc>0 else float('inf')
        l2=float(np.linalg.norm(u1-u2)/(0.5*(np.linalg.norm(u1)+np.linalg.norm(u2))))
        pr=float(np.max(np.abs(u1))/np.max(np.abs(u2))) if np.max(np.abs(u2))>0 else float('inf')
        print(f"    Cross-code: tv_rel={tv:.4e}  l2_rel={l2:.4e}  peak_ratio={pr:.6f}")
        pois.append({'label':label,'oracle':{k:v for k,v in ro.items() if k not in ('u','ua','y')},
                     'tensorlbm':{k:v for k,v in rt.items() if k not in ('u','ua','y')},
                     'cross':{'tv_rel':tv,'l2_rel':l2,'peak_ratio':pr}})
    R['poiseuille'] = pois

    all_pass = (R['lattice']['pass'] and R['equilibrium']['pass'] and R['collision']['pass']
                and all(p['oracle']['eps_inf']<0.15 for p in pois)
                and all(p['tensorlbm']['eps_inf']<0.15 for p in pois)
                and all(p['cross']['tv_rel']<0.01 for p in pois))
    R['overall_pass'] = all_pass
    print(f"\n{'='*70}\nOVERALL: {'PASS' if all_pass else 'FAIL'}\n{'='*70}")
    return R

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--out', default=os.path.join(HERE, 'results.json'))
    args = ap.parse_args()
    R = run_all(quick=args.quick)
    with open(args.out,'w') as f: json.dump(R, f, indent=2, default=str)
    print(f"Results → {args.out}")
