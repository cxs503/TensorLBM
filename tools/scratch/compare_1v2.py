"""决定性对比: 1 卡 vs 2 卡 force_proxy (全叶子 |f| 之和应一致)。"""
import sys, json, subprocess

def run(nproc, steps=2):
    cmd = ["torchrun", f"--nproc_per_node={nproc}",
           "examples/octree_distributed_validate.py",
           "--geo", "sphere", "--nx", "64", "--ny", "48", "--nz", "48",
           "--radius", "6", "--steps", str(steps), "--report-interval", "2",
           "--output", f"/tmp/dump_{nproc}card.json"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd='/root/TensorLBM_feat2')
    try:
        d = json.load(open(f"/tmp/dump_{nproc}card.json"))
        return r.returncode, d
    except Exception:
        return r.returncode, None

for n in [1, 2]:
    rc, d = run(n)
    if d:
        print(f"{n}card: exit={rc} force_proxy={d['force_proxy']:.4f} per_step={d['per_step_s']:.3f}")
    else:
        print(f"{n}card: exit={rc} (no json)")
