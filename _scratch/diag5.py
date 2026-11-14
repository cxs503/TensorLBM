import torch, sys
sys.path.insert(0, "xval/XVAL-LBM-LINEAR-INTERPOLATED-BB-ALPHA-SCHEME")
torch.set_num_threads(6)
import verify_v3_moving_wall as v
from tensorlbm.boundary.interpolated_bounce_back import alpha_bfl, alpha_cli

def one_step_dM(alpha, wv_on, U=0.02, db=0.25, dt_=0.25, n_fluid=8):
    ch = v.Channel(n_fluid, db, dt_)
    wv = ch.wall_velocity(U, 0.0) if wv_on else None
    yj = ch.y_fluid; H = ch.height; u_ref = U*(1-yj/H)
    u0 = torch.zeros((ch.ny, ch.nx), dtype=v.DT); u0[ch.rows[0]:ch.rows[1],:] = u_ref.unsqueeze(1)
    f = v.equilibrium(torch.ones((ch.ny,ch.nx),dtype=v.DT), u0, torch.zeros_like(u0))
    f = torch.where(ch.fluid.unsqueeze(0), f, v.WI)
    res=[]
    for n in range(20):
        m0 = float(f.sum())
        rho, mx, my = v.moments(f)
        fp = v.collide(f, rho, mx/rho, my/rho, "bgk", 0.1, None)
        fp = torch.where(ch.fluid.unsqueeze(0), fp, f)
        fs = torch.empty_like(fp)
        for i in range(v.Q):
            fs[i] = torch.roll(fp[i], shifts=(int(v.CI[i,1]), int(v.CI[i,0])), dims=(0,1))
        m_post = float(fs.sum())
        f = v.interpolated_bounce_back_2d(fs, fp, ch.sdf, ch.solid, alpha=alpha, wall_velocity=wv)
        res.append((m0, m_post, float(f.sum())))
    return res[-1]

for name, alpha, wv in (("bfl+term", alpha_bfl, True), ("bfl_zeros", alpha_bfl, "zero"),
                        ("bfl+none", alpha_bfl, False), ("halfway", alpha_cli, False)):
    w = None if wv is False else wv
    m0, mp, m1 = one_step_dM(alpha, w)
    print(f"{name:>10}: m_before={m0:.12f} after_stream={mp:.12f} after_bc={m1:.12f}  dM_bc={m1-mp:+.6e}  dM_step={m1-m0:+.6e}")
