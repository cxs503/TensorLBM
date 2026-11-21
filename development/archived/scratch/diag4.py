import torch, sys, time
sys.path.insert(0, "xval/XVAL-LBM-LINEAR-INTERPOLATED-BB-ALPHA-SCHEME")
torch.set_num_threads(6)
import verify_v3_moving_wall as v
from tensorlbm.boundary.interpolated_bounce_back import alpha_bfl

def stream_mode(f, mode):
    out = torch.empty_like(f)
    for i in range(v.Q):
        sy, sx = int(v.CI[i,1]), int(v.CI[i,0])
        src = torch.roll(f[i], shifts=(sy, sx), dims=(0,1))
        if mode == "keep":
            if sy > 0: src[:sy,:] = f[i][:sy,:]
            elif sy < 0: src[sy:,:] = f[i][sy:,:]
        out[i] = src
    if mode == "wrap_guard":
        out[:,0,:] = v.WI.view(v.Q,1); out[:,-1,:] = v.WI.view(v.Q,1)
    return out

def run(mode, steps=3000, U=0.02, db=0.25, dt_=0.25, n_fluid=8, wv_on=True):
    ch = v.Channel(n_fluid, db, dt_)
    wv = ch.wall_velocity(U, 0.0) if wv_on else None
    yj = ch.y_fluid; H = ch.height; u_ref = U*(1-yj/H)
    u0 = torch.zeros((ch.ny, ch.nx), dtype=v.DT); u0[ch.rows[0]:ch.rows[1],:] = u_ref.unsqueeze(1)
    f = v.equilibrium(torch.ones((ch.ny,ch.nx),dtype=v.DT), u0, torch.zeros_like(u0))
    f = torch.where(ch.fluid.unsqueeze(0), f, v.WI)
    rec=[]; m0=float(f.sum())
    for n in range(1, steps+1):
        rho, mx, my = v.moments(f)
        fp = v.collide(f, rho, mx/rho, my/rho, "bgk", 0.1, None)
        fp = torch.where(ch.fluid.unsqueeze(0), fp, f)
        fs = stream_mode(fp, mode)
        f = v.interpolated_bounce_back_2d(fs, fp, ch.sdf, ch.solid, alpha=alpha_bfl, wall_velocity=wv)
        if n % 1000 == 0:
            p = v.profile(f, 0.0, ch, 0.0)
            rec.append((n, float(torch.linalg.norm(p-u_ref)/torch.linalg.norm(u_ref)),
                        (float(f.sum())-m0)/m0, float((v.CX*f).sum()),
                        float(f[ch.rows[0],:,0].sum())))
    return rec

for mode in ("plain","keep","wrap_guard"):
    t0=time.time(); rec = run(mode)
    print(mode, "t=%.0fs"%(time.time()-t0))
    for n,l2,dM,mom,fb in rec: print("  n=%5d l2=%.3e  dM=%+.3e  Px=%.8e"%(n,l2,dM,mom))
