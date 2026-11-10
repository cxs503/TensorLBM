"""Ladder diagnostic: is the Nyquist-aware Smagorinsky closure stable and turbulent?"""
import argparse, json, math, sys, time, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from tensorlbm.turbulence.les_channel_verification import run_case

ap = argparse.ArgumentParser()
ap.add_argument("--case", action="append", default=None,
                help="h:steps:warmup:strain:cs")
ap.add_argument("--u-tau", type=float, default=0.01)
ap.add_argument("--threads", type=int, default=128)
args = ap.parse_args()
torch.set_num_threads(args.threads)

cases = args.case or ["8:8000:2000:forward_gradient:0.12", "12:10000:3000:forward_gradient:0.12"]
out = {}
for spec in cases:
    h, steps, warmup, strain, cs = spec.split(":")
    label = f"h{h}_{strain}_cs{cs}"
    t0 = time.time()
    rec = {"spec": spec}
    try:
        art = run_case(half_height=int(h), steps=int(steps), warmup_steps=int(warmup),
                       u_tau=args.u_tau, cs=float(cs), strain_source=strain,
                       sample_interval=20, report_interval=1000, label=label)
        hist = art["result"]["history"]
        tkes = [hh["tke"] for hh in hist]
        rec.update({
            "ok": True,
            "summary": art["summary"],
            "criteria": {k: {"status": v["status"], "metric": v.get("metric")} for k, v in art["criteria"]["checks"].items()},
            "tke_first": tkes[0], "tke_last": tkes[-1],
            "tke_min": min(tkes), "tke_max": max(tkes),
            "u_bulk_hist": [hh["u_bulk"] for hh in hist],
            "ma_max": max(hh.get("max_mach", 0.0) for hh in hist),
            "seconds": time.time() - t0,
        })
        Path(f".scratch/diag_{label}.json").write_text(json.dumps(art["summary"]))
    except Exception as exc:  # honest FAIL, keep going
        rec.update({"ok": False, "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-800:], "seconds": time.time() - t0})
    out[spec] = rec
    print(json.dumps(rec, indent=1, default=str)[:2500], flush=True)

Path(".scratch/ladder_result.json").write_text(json.dumps(out, indent=1, default=str))
print("LADDER DONE")
