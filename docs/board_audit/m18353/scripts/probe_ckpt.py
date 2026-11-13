"""Audit probe: checkpoint integrity + delivered-vs-rerun determinism."""
import json, sys, hashlib
sys.path.insert(0, '/root/tp-agent/work/m18236/src')
import torch
from tensorlbm import cac_lbm2d as cac, pc_ladder as P
torch.set_num_threads(1)
D = "/root/tp-agent/work/m18236/TensorLBM/xval/XVAL-LBM-QUASI-STATIC-PC-LADDER-WITH-CHECKPOINT/artifacts"
CFG = dict(rho_l=1.0, rho_g=0.1, mu_l=0.2, mu_g=0.02, sigma=0.05, width=3.0,
           mobility=0.05, mu_law="harmonic", p0=0.01, gravity=(0.0, 0.0),
           gravity_mode="rho", axes="pp")
out = {}
sim = cac.CAC2D((44,128), **CFG)
meta = P.load_ladder_state(D + "/ckpt/main/rung_000", sim)   # verify_digest default True
out["main_rung0_restore_ok"] = True
out["main_rung0_meta"] = {k: meta[k] for k in ("rung","step","digest")}
out["restored_finite"] = bool(torch.isfinite(sim.f).all().item())
# same for the last rung, which the delivery marks NaN
sim2 = cac.CAC2D((44,128), **CFG)
m5 = P.load_ladder_state(D + "/ckpt/main/rung_005", sim2)
out["main_rung5_restore_ok"] = True
out["main_rung5_finite_f"] = bool(torch.isfinite(sim2.f).all().item())
out["main_rung5_step"] = m5["step"]
# determinism: delivered vs audit rerun (numeric equality of the E0/E1 logs)
a = json.load(open(D + "/results.json"))
b = json.load(open("/root/tp-agent/work/m18353/.audit_tmp/xvalrun/artifacts/results.json"))
def flat(d, pre=""):
    r = {}
    if isinstance(d, dict):
        for k, v in d.items(): r.update(flat(v, f"{pre}.{k}" if pre else k))
    elif isinstance(d, list):
        for i, v in enumerate(d): r.update(flat(v, f"{pre}[{i}]"))
    else: r[pre] = d
    return r
fa, fb = flat(a["cases"]), flat(b["cases"])
keys = [k for k in fa if k in fb]
same = [k for k in keys if (fa[k] == fb[k]) or (fa[k] != fa[k] and fb[k] != fb[k])]
diff = {k: (fa[k], fb.get(k)) for k in keys if k not in same}
out["determinism"] = {"compared": len(keys), "identical_incl_nan": len(same), "n_diff": len(diff),
                      "diff_examples": dict(list(diff.items())[:6])}
print(json.dumps(out, indent=1, default=str))
