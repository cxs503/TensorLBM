import torch, sys, time
sys.path.insert(0, "xval/XVAL-LBM-LINEAR-INTERPOLATED-BB-ALPHA-SCHEME")
torch.set_num_threads(6)
import verify_v3_moving_wall as v
from tensorlbm.boundary.interpolated_bounce_back import alpha_bfl

def stream_pad(f):
    """streaming with zero padding in y (no wrap), periodic in x."""
    out = torch.zeros_like(f)
    ny, nx = f.shape[1], f.shape[2]
    for i in range(v.Q):
        sy, sx = int(v.CI[i,1]), int(v.CI[i,0])
        src = torch.roll(f[i], shifts=(-sy, -sx), dims=(0,1))
        if sy > 0:
            src[:sy, :] = 0.0
        elif sy < 0:
            src[sy:, :] = 0.0
        out[i] = src
    return out

def run(mode, steps=7500, U=0.02, db=0.25, dt_=0.25, n_fluid=8):
    ch = v.Channel(n_fluid, db, dt_)
    wv = ch.wall_velocity(U, 0.0)
    yj = ch.y_fluid; H = ch.height; u_ref = U*(1-yj/H)
    u0 = torch.zeros((ch.ny, ch.nx), dtype=v.DT); u0[ch.rows[0]:ch.rows[1],:] = u_ref.unsqueeze(1)
    f = v.equilibrium(torch.ones((ch.ny,ch.nx),dtype=v.DT), u0, torch.zeros_like(u0))
    f = torch.where(ch.fluid.unsqueeze(0), f, v.WI)
    rec=[]
    for n in range(1, steps+1):
        rho, mx, my = v.moments(f)
        fp = v.collide(f, rho, mx/rho, my/rho, "bgk", 0.1, None)
        fp = torch.where(ch.fluid.unsqueeze(0), fp, f)
        if mode=="wrap_guard":
            fs = torch.empty_like(fp)
            for i in range(v.Q):
                fs[i] = torch.roll(fp[i], shifts=(int(v.CI[i,1]), int(v.CI[i,0])), dims=(0,1))
            fs[:,0,:]=v.WI.view(v.Q,1); fs[:,-1,:]=v.WI.view(v.Q,1)
        else:
            fs = stream_pad(fp)
        f = v.interpolated_bounce_back_2d(fs, fp, ch.sdf, ch.solid, alpha=alpha_bfl, wall_velocity=wv)
        if n % 2500 == 0:
            p = v.profile(f, 0.0, ch, 0.0)
            l2 = float(torch.linalg.norm(p-u_ref)/torch.linalg.norm(u_ref))
            mass = float(f.sum())
            rec.append((n, l2, mass, float(f[0].sum())))
    return rec

for mode in ("pad",):
    t0=time.time(); rec = run(mode)
    print(mode, "t=%.0fs"%(time.time()-t0))
    base = rec[0][2]
    for n,l2,mass,sh in rec: print("  n=%5d l2=%.3e  dM=%+.3e"%(n,l2,(mass-base)/base))
