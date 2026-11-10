"""Verbose short diagnostic: watch the divergence (or not) of the LES channel."""
import argparse, json, math, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/"src"))
import torch
from tensorlbm.turbulence.les_channel3d import LESChannel, LESChannelConfig
ap=argparse.ArgumentParser()
ap.add_argument("--h", type=int, default=8)
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--warmup", type=int, default=500)
ap.add_argument("--u-tau", type=float, default=0.01)
ap.add_argument("--cs", type=float, default=0.12)
ap.add_argument("--strain", default="forward_gradient")
ap.add_argument("--wall-model", default="none")
ap.add_argument("--report", type=int, default=100)
ap.add_argument("--threads", type=int, default=24)
ap.add_argument("--label", default="diag")
ap.add_argument("--dtype", default="float32")
ap.add_argument("--pert", type=float, default=0.35)
ap.add_argument("--no-van-driest", action="store_true")
ap.add_argument("--tau-minus-max", type=float, default=None)
a=ap.parse_args()
torch.set_num_threads(a.threads)
h=a.h
cfg=LESChannelConfig(nx=int(round(2*math.pi*h)), nz=int(round(math.pi*h)), ny=2*h+2,
  re_tau=180., u_tau=a.u_tau, steps=a.steps, warmup_steps=a.warmup, sample_interval=10,
  report_interval=a.report, smagorinsky_cs=a.cs, strain_source=a.strain,
  wall_model=a.wall_model, wall_model_interval=5, label=a.label,
  dtype=a.dtype, perturbation_amplitude=a.pert, van_driest=not a.no_van_driest, tau_minus_max=a.tau_minus_max)
les=LESChannel(cfg)
print("cells",les.N,"nu",les.nu,"tau+",sess_tau if False else cfg.tau_plus,"dy+",cfg.dy_plus, flush=True)
try:
    res=les.run_les(verbose=True)
    Path(f".scratch/trace_{a.label}.json").write_text(json.dumps({"history":res["history"],"summary":{
        "u_bulk":res["mean_velocity_bulk"],"re_bulk":res["reynolds_number_bulk"],
        "uv_peak":res["reynolds_stress"]["uv_minus_peak_plus"],
        "uv_peak_y":res["reynolds_stress"]["uv_minus_peak_y_plus"],
        "u_tau_exchange":res["friction_velocity"]["wall_exchange"],
        "u_tau_fit":res["friction_velocity"]["stress_fit"],
        "u_plus":res["wall_units"]["u_plus"][:12],
        "y_plus":res["wall_units"]["y_plus"][:12],
    }}))
    print("OK", flush=True)
except Exception as e:
    print("FAIL", type(e).__name__, e, flush=True)
