"""Attribute the slow liquid-inventory drift to a solver operator (solo T5)."""
import sys
from collections import defaultdict

sys.path.insert(0, "src")
sys.path.insert(0, ".")
import torch

torch.set_num_threads(1)
import benchmarks.water_entry.single_drop.free_surface_probe as fsp
import tensorlbm.free_surface_lbm as fs

N_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 150
SIMPLE = "--simple" in sys.argv

orig = fs.free_surface_step
ledger = {"steps": []}


def wrapped(*a, **k):
    k["runtime_ledger"] = ledger
    return orig(*a, **k)


fsp.free_surface_step = wrapped
result = fsp.probe(8, 0.06, N_STEPS, 0.18, simple=SIMPLE, release=1.53)

agg = defaultdict(float)
gross = defaultdict(float)
dominant = defaultdict(int)
for rec in ledger["steps"]:
    for ev in rec["operator_attribution"]["events"]:
        if ev["tracked_mass"]:
            agg[ev["operator"]] += ev["net_delta"]
            gross[ev["operator"]] += ev["gross_magnitude"]
    dominant[rec["operator_attribution"]["dominant_operator"]] += 1

print("stop_reason:", result["stop_reason"])
print(
    "last row:",
    {k: result["history"][-1][k] for k in ("step", "time_ms", "volume_drift")},
)
print("operator net deltas (tracked mass):")
for op, v in sorted(agg.items(), key=lambda kv: -abs(kv[1])):
    print(f"  {op:16s} net={v:+.6e}  gross={gross[op]:.6e}")
print("dominant-operator counts:", dict(dominant))
rows = ledger["steps"]
print("first/last mass_drift:", rows[0]["mass_drift"], rows[-1]["mass_drift"])
print("unexplained sum:", sum(r["unexplained"] for r in rows))
