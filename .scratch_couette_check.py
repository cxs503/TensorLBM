import json, time
from tensorlbm.spherical_force_method import run_couette
out=[]
for off in (0.0, 0.3, -0.35, 0.75):
    t=time.time(); r=run_couette(offset=off, steps=9000); r["wall_s"]=round(time.time()-t,1)
    out.append(r)
    print(f"off={off:+.2f} q={r['q_mean']:.3f} Fbot={r['shear_bottom_stress']:.6e} Ftop={r['shear_top_stress']:.6e} an={r['shear_analytic']:.6e} err_bot={r['rel_error_bottom']*100:.2f}% err_top={r['rel_error_top']*100:.2f}% L2={r['profile_l2_error']:.2e} ({r['wall_s']}s)", flush=True)
json.dump(out, open("/tmp/couette_check.json","w"), default=str)
