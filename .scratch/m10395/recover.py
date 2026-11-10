import sys, json
sys.path.insert(0, "src")
from pathlib import Path
from tensorlbm.sloshing_forced.campaign import case_table, recover_case
out = Path("artifacts/p88w1")
zeta = json.load(open(out / "decay_m1.json"))["zeta"]
cases = {c["id"]: c for c in case_table()}
print("zeta", zeta)
for cid in ("sweep_r1.20",):
    s = recover_case(out, cases[cid], zeta)
    print("recovered", cid, "a1 %.5f p_amp %.1f massdrift %.4f" % (s["a1"]["amplitude"], s["pressure"]["p_amp_Pa"], s["max_mass_drift"]))
