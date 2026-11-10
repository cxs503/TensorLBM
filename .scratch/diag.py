import sys, time, math
sys.path.insert(0,'/root/tp-agent/work/m9874/src')
import torch; torch.set_num_threads(2)
from tensorlbm.contact_line.engine import Channel, WallStepper, sessile_drop_state
from tensorlbm.contact_line import three_phase as tp
from tensorlbm.d2q9 import macroscopic
import multiprocessing as mp

def one(arg):
    f_w, gx, forcing, drive, steps = arg
    torch.set_num_threads(1)
    ch=Channel(nx=128,ny=64,wall_rows=2); solid=ch.solid_mask()
    r=0.32*64; h=-r*math.cos(math.radians(110.0))
    f,_=sessile_drop_state(ch, radius=r, height=h)
    st=WallStepper(ch,G=5.0,tau=1.0,wall_fraction=f_w,forcing=forcing,drive=drive)
    rho=macroscopic(f)[0]; m0=float(rho.sum())
    cx0=float((torch.where(solid,torch.zeros_like(rho),rho)*torch.arange(128.)).sum()/torch.where(solid,torch.zeros_like(rho),rho).sum())
    track=[]; cxs=cx0
    for n in range(1,steps+1):
        f=st.step(f,gx)
        if n%200==0:
            rho=macroscopic(f)[0]
            w=torch.where(solid,torch.zeros_like(rho),rho)
            cx=float((w*torch.arange(128.)).sum()/w.sum())
            d=(cx-cxs+64)%128-64; cxs=cx
            track.append((n,cxs))
    rho=macroscopic(f)[0]
    v=tp.contact_line_velocity([(float(a),float(b)) for a,b in track[::5]])
    ux,uy=st.velocity(f); sp=torch.sqrt(ux**2+uy**2)
    liq=(rho>1.5)&(~solid)
    return dict(f=f_w,gx=gx,forcing=forcing,drive=drive,centroid_v=v['velocity'],r2=v['r2'],
                sp_max=float(sp[liq].max()), sp_mean=float(sp[liq].mean()),
                drift=abs(float(rho.sum())-m0)/m0, umax_all=float(sp[~solid].max()))

if __name__=="__main__":
    cases=[(0.6,0.0,'edm','uniform',14000),(0.6,1e-4,'edm','liquid',14000),(0.6,3e-4,'guo','uniform',14000),(0.6,1e-3,'guo','uniform',14000)]
    t0=time.time()
    with mp.get_context("fork").Pool(4) as p:
        for r in p.map(one, cases):
            print("f=%.2f gx=%.1e %s/%s v=%.5f r2=%.2f sp_max=%.4f sp_mean=%.5f drift=%.1e umax_all=%.3f" % (
              r['f'],r['gx'],r['forcing'],r['drive'],r['centroid_v'],r['r2'],r['sp_max'],r['sp_mean'],r['drift'],r['umax_all']), flush=True)
    print('t', time.time()-t0)
