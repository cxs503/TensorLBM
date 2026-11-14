"""Run every case in cases.json sequentially (cluster entry point)."""
import json, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
cases = json.loads((HERE / "cases.json").read_text())
fails = []
for i, c in enumerate(cases):
    t0 = time.time()
    print(f"=== case {i} {c['tag']} ===", flush=True)
    r = subprocess.run([sys.executable, "run_shard.py", "--index", str(i)], cwd=HERE)
    print(f"=== case {i} {c['tag']} rc={r.returncode} wall={time.time()-t0:.0f}s ===", flush=True)
    if r.returncode != 0:
        fails.append(c["tag"])
print("FAILED:", fails, flush=True)
