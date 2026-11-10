import sys, time, json
sys.path.insert(0,'/root/tp-agent/work/m9874/src'); sys.path.insert(0,'/root/tp-agent/work/m9874/benchmarks')
import torch; torch.set_num_threads(4)
sys.path.insert(0,'/root/tp-agent/work/m9874/benchmarks/three_phase_contact_line')
import run as mod
import multiprocessing as mp
if __name__ == "__main__":
    jobs = [(0.6, 1e-3), (0.6, 3e-3), (0.6, 1e-2)]
    t0=time.time()
    with mp.get_context("fork").Pool(3) as p:
        res = [p.apply_async(mod._relax_sessile, (f,), {"ny":64,"nx":128,"steps":14000,"gx":gx,"drive":"liquid"}) for f,gx in jobs]
        for (f,gx),r in zip(jobs, res):
            d = r.get()
            print("f=%.2f gx=%.1e vf=%.4f vr=%.4f thA=%.1f thR=%.1f th_cap=%.1f umax=%.3f drift=%.2e" % (
                f, gx, d['velocity_front'], d['velocity_rear'], d.get('theta_adv',float('nan')), d.get('theta_rec',float('nan')), d['theta_final'], d['umax'], d['mass_drift']), flush=True)
    print("t", time.time()-t0)
