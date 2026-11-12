"""Audit probe (m15892): re-run the m15468 3D static-droplet point and expose the
fields the delivered artifact dropped (history length, rho_min/rho_max)."""
import json, sys
sys.path.insert(0, "/root/tp-agent/work/m15468/src")
import torch  # noqa
from tensorlbm import pseudopotential_edm3d as edm  # noqa

run = edm.static_droplet(R=8.0, n=32, G_eff=-5.0, tau=1.0, stencil="d3q19",
                         form="difference", forcing="kupershtokh",
                         steps=400, check_every=100)
h = run.pop("history")
run["history_len"] = len(h)
run["history"] = h
with open("/root/tp-agent/work/m15892/audit/probe_3d.json", "w") as f:
    json.dump(run, f, indent=2, default=float)
print(json.dumps({k: v for k, v in run.items() if k != "history"}, indent=2, default=float))
