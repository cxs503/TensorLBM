import sys, time, math
sys.path.insert(0, 'src'); sys.path.insert(0, 'benchmarks/xval_job_checkpoint')
import numpy as np
from run_xval import taylor_green, kinetic_energy
from tensorlbm import LBMSimulation

TAU = 0.55
NU = (TAU - 0.5) / 3.0

def ke_trace(n, steps, sample):
    sim = taylor_green(n, n)
    ke0 = kinetic_energy(sim)
    out = [(0, ke0)]
    for s in range(1, steps + 1):
        sim.step()
        if s % sample == 0:
            out.append((s, kinetic_energy(sim)))
    return out

for n, steps, sample in ((64, 400, 10), (128, 400, 10)):
    t0 = time.perf_counter()
    tr = ke_trace(n, steps, sample)
    dt = time.perf_counter() - t0
    t = np.array([a for a, _ in tr], float); ke = np.array([b for _, b in tr], float)
    # late-window fit
    for lo in (0, 100, 200):
        m = t >= lo
        lam = -np.polyfit(t[m], np.log(ke[m]), 1)[0]
        ana = 4 * NU * (2 * math.pi / n) ** 2
        print(f"n={n} window>={lo}: lam={lam:.6e} analytic={ana:.6e} err={100*(lam-ana)/ana:+.2f}%")
    print(f"  wall {dt:.1f}s")