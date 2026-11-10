import sys, json, glob, os
sys.path.insert(0, "src")
from pathlib import Path
from tensorlbm.sloshing_forced.campaign import reanalyse_forced

out = Path("artifacts/p88w1")
zeta = json.load(open(out / "decay_m1.json"))["zeta"]
done = []
for p in sorted(out.glob("*.json")):
    cid = p.stem
    if cid == "campaign":
        continue
    d = json.load(open(p))
    if d.get("kind") != "forced":
        continue
    s = reanalyse_forced(out, cid, zeta)
    done.append(cid)
    print(cid, "a1 %.5f" % s["a1"]["amplitude"], "p_amp %.1f" % s["pressure"]["p_amp_Pa"],
          "snr", [round(g["snr"], 1) for g in s["pressure"]["raw_gauges"]])
print("reanalysed", len(done))