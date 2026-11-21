"""Order ladder of the time-Taylor series field at the case's primary point.

Reports, for n = 32, T = 1 eddy time (t = 102 lattice), the L2 difference between
successive truncation orders, to test whether the series *field* (as opposed to
its energy/enstrophy) is converged at accessible orders.
"""
import json, sys, time
import numpy as np

sys.path.insert(0, "src")
from tensorlbm.tg3d_lbm_check import tg3d_parameters
from tensorlbm.tg3d_series import TG3DSeries, taylor_green_classic

A, RE, N = 0.05, 160.0, 32
p = tg3d_parameters(N, amplitude=A, reynolds=RE)
steps = max(2, int(round(1.0 / p["eddy_time_per_step"])))
t = float(steps)
rel = lambda a, b: 100.0 * float(np.sqrt(np.sum((a - b) ** 2) / np.sum(b ** 2)))

fields, rows = {}, []
for order in (12, 16, 20, 24, 28, 32):
    t0 = time.time()
    ser = TG3DSeries(taylor_green_classic(A), box=float(N), nu=p["nu"], order=order)
    u, v, w = ser.velocity_field(t, n=N)
    fields[order] = np.stack([u, v, w])
    tail = ser.relative_tail(t)
    rows.append(dict(order=order, wall_s=round(time.time() - t0, 1),
                     energy_ratio=float(ser.energy(t) / ser.energy(0.0)),
                     enstrophy_ratio=float(ser.enstrophy(t) / ser.enstrophy(0.0)),
                     tail_energy=float(tail["energy_last_order_rel"]),
                     tail_enstrophy=float(tail["enstrophy_last_order_rel"])))
    print(json.dumps(rows[-1]), flush=True)

orders = sorted(fields)
pairs = [dict(a=a, b=b, field_rel_pct=rel(fields[a], fields[b])) for a, b in zip(orders, orders[1:])]
ref = orders[-1]
vs_top = [dict(order=o, field_rel_pct_vs_top=rel(fields[o], fields[ref])) for o in orders[:-1]]
out = dict(n=N, eddy_times=1.0, steps=steps, t_lattice=t, nu=p["nu"],
           ladder=rows, successive=pairs, vs_top_order=vs_top, top_order=ref)
print(json.dumps(out, indent=1))
open("_scratch/m22117_order_ladder.json", "w").write(json.dumps(out, indent=2) + "\n")
