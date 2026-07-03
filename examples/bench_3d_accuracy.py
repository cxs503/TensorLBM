"""
TensorLBM 高精度基准测试 — GPU加密网格 对标文献值
"""
import json, math, time
from pathlib import Path

from tensorlbm import CylinderFlowConfig, run_cylinder_flow
from tensorlbm.pipeline_flow import PipelineFlowConfig, run_pipeline_flow
from tensorlbm.turbulent_channel import TurbulentChannelConfig, run_turbulent_channel
from tensorlbm.sphere_flow import SphereFlowConfig, run_sphere_flow
from tensorlbm.backward_facing_step import BackwardFacingStepConfig, run_backward_facing_step

OUTPUT = Path("outputs/benchmarks_fine")

REF = {
    "cylinder_st": 0.166, "cylinder_cd": 1.35,
    "pipeline_st": 0.183,
    "turbulent_loglaw": 3.0,
    "sphere_cd_re50": 1.57, "sphere_cd_re100": 1.09, "sphere_cd_re200": 0.77,
    "bfs_reattach": 7.0,  # Re=100, reattachment length / step height
}

def header(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")

def result(label, sim, ref, unit=""):
    err = abs(sim - ref) / ref * 100
    status = "✓" if err < 15 else "✗"
    print(f"  {label:<28} sim={sim:9.4f}{unit}  ref={ref:9.4f}{unit}  err={err:5.1f}%  {status}")
    return err < 15

# ============================================================
# 1. 圆柱绕流 Re=100 — 800×200, 40K步, GPU
# ============================================================
header("1. Cylinder Re=100 (800×200, 40K steps, GPU)")
cfg = CylinderFlowConfig(nx=800, ny=200, radius=20.0, u_in=0.08, re=100.0,
    n_steps=40000, output_interval=2000,
    output_root=OUTPUT, run_name="cylinder_fine", overwrite=True, device="sdaa")
t0 = time.perf_counter()
run_dir = run_cylinder_flow(cfg)
t1 = time.perf_counter() - t0
meta = json.loads((run_dir / "run_metadata.json").read_text())
st = meta.get("strouhal", 0)
diag = meta["diagnostics"]
cd_vals = [d["cd"] for d in diag[-10:] if isinstance(d.get("cd"), float)]
cd = sum(cd_vals)/len(cd_vals) if cd_vals else 0
print(f"  Time: {t1:.0f}s")
r1a = result("Strouhal", st, REF["cylinder_st"])
r1b = result("Cd mean", cd, REF["cylinder_cd"])

# ============================================================
# 2. 后向台阶 Re=100 — 400×80, 30K步, GPU  
# ============================================================
header("2. BFS Re=100 (400×80, 30K steps, GPU)")
cfg = BackwardFacingStepConfig(nx=400, ny=80, step_h=40, re=100.0, u_in=0.05,
    n_steps=30000, output_interval=2000,
    output_root=OUTPUT, run_name="bfs_fine", overwrite=True, device="sdaa")
t0 = time.perf_counter()
run_dir = run_backward_facing_step(cfg)
t1 = time.perf_counter() - t0
meta = json.loads((run_dir / "run_metadata.json").read_text())
reattach = meta.get("reattachment_length_ratio", 0)
print(f"  Time: {t1:.0f}s")
r2 = result("Reattach L/H", reattach, REF["bfs_reattach"])

# ============================================================
# 3. 管道绕流 Re=200 e/D=0.5 — 400×160, 30K步, GPU
# ============================================================
header("3. Pipeline e/D=0.5 Re=200 (400×160, 30K steps, GPU)")
cfg = PipelineFlowConfig(nx=400, ny=160, diameter=20.0, gap_ratio=0.5,
    u_in=0.05, re=200.0, n_steps=30000, output_interval=2000,
    output_root=OUTPUT, run_name="pipeline_fine", overwrite=True, device="sdaa")
t0 = time.perf_counter()
run_dir = run_pipeline_flow(cfg)
t1 = time.perf_counter() - t0
meta = json.loads((run_dir / "run_metadata.json").read_text())
st_pipe = meta.get("strouhal", 0)
print(f"  Time: {t1:.0f}s")
r3 = result("Strouhal", st_pipe, REF["pipeline_st"])

# ============================================================
# 4. 湍流通道 Re_τ=100 — 256×128, 80K步, GPU
# ============================================================
header("4. Turbulent Channel Re_τ=100 (256×128, 80K steps, GPU)")
cfg = TurbulentChannelConfig(nx=256, ny=128, re_tau=100.0,
    n_steps=80000, output_interval=5000,
    output_root=OUTPUT, run_name="channel_fine", overwrite=True, device="sdaa")
t0 = time.perf_counter()
run_dir = run_turbulent_channel(cfg)
t1 = time.perf_counter() - t0
meta = json.loads((run_dir / "run_metadata.json").read_text())
rms = meta.get("log_law_rms_error", 999)
print(f"  Time: {t1:.0f}s")
r4 = result("RMS log-law err", rms, REF["turbulent_loglaw"], "%")

# ============================================================
# 5. 3D 球体 Re=50/100/200 Cd 基准
# ============================================================
sphere_ok = 0
for re_val, cd_ref, label in [(50,1.57,"Re=50"),(100,1.09,"Re=100"),(200,0.77,"Re=200")]:
    header(f"5. 3D Sphere {label} (120×60×60, 1000 steps, GPU)")
    cfg = SphereFlowConfig(nx=120, ny=60, nz=60, radius=8.0, u_in=0.06, re=re_val,
        n_steps=1000, output_interval=200,
        output_root=OUTPUT, run_name=f"sphere_re{re_val}", overwrite=True, device="sdaa")
    t0 = time.perf_counter()
    run_dir = run_sphere_flow(cfg)
    t1 = time.perf_counter() - t0
    meta = json.loads((run_dir / "run_metadata.json").read_text())
    diag = meta["diagnostics"]
    cd_vals = [d.get("cd",0) for d in diag[-5:] if "cd" in d]
    cd_mean = sum(cd_vals)/len(cd_vals) if cd_vals else 0
    print(f"  Time: {t1:.0f}s")
    if result(f"Cd {label}", cd_mean, cd_ref): sphere_ok += 1

# ============================================================
# Summary
# ============================================================
header("BENCHMARK SUMMARY")
total = 8
passed = sum([r1a, r1b, r2, r3, r4]) + (sphere_ok >= 2)
print(f"\n  Passed: {passed}/{total} (threshold: <15% error)")
print(f"  GPU: NVIDIA RTX 3090 24GB")
