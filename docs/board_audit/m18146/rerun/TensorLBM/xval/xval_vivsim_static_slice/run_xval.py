"""XVAL run: vivsim jax-pure-function-aa-pattern-static-slice  ->  TensorLBM.

Generates:
  artifacts/xval_vivsim_static_slice.json   (all numbers, zero hand-copy)
  figures/xval_vivsim_profile.png          (analytical vs static-slice vs roll)
  figures/xval_vivsim_timing.png           (per-operator + full-step timing)
  artifacts/convergence.json               (error vs timesteps, >=2 points)

Run:  python TensorLBM/xval/xval_vivsim_static_slice/run_xval.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from tensorlbm import xval_vivsim_static_slice as X  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
ART = os.path.join(HERE, "artifacts")
FIG = os.path.join(HERE, "figures")
os.makedirs(ART, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

# ----------------------------- parameters (from vivsim example) ------------- #
NY, NX = 10, 10          # y (walls) x x (periodic)  -- vivsim: NX=10, NY=10
NU = 0.2                 # kinematic viscosity (lattice units)
GX = 0.001               # body force in +x
OMEGA = X.get_omega(NU)  # 1/(3 nu + 0.5)
TM = 100000              # vivsim total timesteps
DT = torch.float64


def profile_err(ux_prof: torch.Tensor, ana: torch.Tensor) -> dict:
    rel = (ux_prof - ana).abs() / ana.max()
    return {
        "max_rel_err": float(rel.max().item()),
        "mean_rel_err": float(rel.mean().item()),
        "max_abs_err": float((ux_prof - ana).abs().max().item()),
    }


def run_case(stream_fn, n_steps, tag):
    f, rho, ux, uy, _ = X.run_poiseuille(
        stream_fn, NY, NX, NU, GX, n_steps, dtype=DT
    )
    ana = X.analytical_ux(NY, GX, NU)
    ux_prof = ux[:, NX // 2]
    return {
        "tag": tag,
        "n_steps": n_steps,
        "profile": [float(v) for v in ux_prof.tolist()],
        "analytical": [float(v) for v in ana.tolist()],
        "err": profile_err(ux_prof, ana),
        "mass": float(rho.sum().item()),
        "max_abs_f": float(f.abs().max().item()),
        "finite": bool(torch.isfinite(f).all().item()),
    }


def main():
    out = {
        "anchor": "XVAL-LBM-JAX-PURE-FUNCTION-AA-PATTERN-STATIC-SLIC",
        "oss_ref": "haimingz/vivsim (slug NEW:LBM-GPU-JAX-FUSION)",
        "params": {
            "ny": NY, "nx": NX, "nu": NU, "gx": GX, "omega": OMEGA,
            "tm": TM, "dtype": "float64", "device": "cpu",
            "collision": "BGK", "forcing": "EDM(=Guo term)",
            "boundary": "force-corrected NEBB (top,bottom)",
            "convention": "full distribution rho=sum(f) [TensorLBM]",
        },
    }

    # ---- 1) correctness: reference pattern vs analytical -------------------- #
    out["correctness"] = {}
    out["correctness"]["static_slice"] = run_case(X.stream_static_slice, TM, "static_slice")
    out["correctness"]["roll_std"] = run_case(X.stream_roll_std, TM, "roll_std")
    out["correctness"]["roll_tensorlbm"] = run_case(X.stream_roll, TM, "roll_tensorlbm")

    # ---- 2) equivalence: static-slice == roll_std (same math) --------------- #
    f0 = torch.randn(9, NY, NX, dtype=DT)
    d = (X.stream_static_slice(f0) - X.stream_roll_std(f0)).abs().max().item()
    out["equivalence_static_slice_vs_roll_std_maxdiff"] = d

    # ---- 3) convergence: error vs timesteps (>=2 points) -------------------- #
    conv = []
    for ns in [10000, 25000, 50000, 100000]:
        r = run_case(X.stream_static_slice, ns, f"conv_{ns}")
        conv.append({"n_steps": ns, **r["err"]})
    out["convergence"] = conv

    # ---- 4) parameter sensitivity: grid size -------------------------------- #
    sens = []
    for ny in [10, 20, 40]:
        f, rho, ux, uy, _ = X.run_poiseuille(X.stream_static_slice, ny, NX, NU, GX, 40000, dtype=DT)
        ana = X.analytical_ux(ny, GX, NU)
        sens.append({"ny": ny, **profile_err(ux[:, NX // 2], ana)})
    out["sensitivity_grid"] = sens

    # ---- 5) performance: per-operator + full step --------------------------- #
    out["performance"] = benchmark()

    # ---- write artifacts ----------------------------------------------------- #
    with open(os.path.join(ART, "xval_vivsim_static_slice.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    make_figures(out)
    print(json.dumps({k: out[k] for k in
                      ["equivalence_static_slice_vs_roll_std_maxdiff",
                       "convergence", "sensitivity_grid"]}, indent=2))
    print("correctness:", {k: v["err"]["max_rel_err"] for k, v in out["correctness"].items()})
    print("perf:", json.dumps(out["performance"], indent=2))


def benchmark():
    """Per-operator + full-step timing, vivsim method (warmup + repeat)."""
    N_WARMUP, N_REPEAT = 10, 200
    ny, nx = 128, 128
    f = X.init_equilibrium(ny, nx, GX, DT)
    omega = OMEGA
    g = torch.zeros(2, ny, nx, dtype=DT); g[0] = GX

    def timeit(fn, *args):
        for _ in range(N_WARMUP):
            fn(*args)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        for _ in range(N_REPEAT):
            fn(*args)
        return (time.perf_counter() - t0) / N_REPEAT * 1e6  # us/call

    rho, ux, uy = X.macroscopic(f)
    u = torch.stack([ux, uy])
    feq = X.equilibrium(rho, ux, uy)

    perf = {"grid": f"{ny}x{nx}", "n_warmup": N_WARMUP, "n_repeat": N_REPEAT, "unit": "us/call"}
    perf["stream_static_slice"] = timeit(X.stream_static_slice, f)
    perf["stream_roll_std"] = timeit(X.stream_roll_std, f)
    perf["stream_roll_tensorlbm"] = timeit(X.stream_roll, f)
    perf["collide_bgk"] = timeit(X.collide_bgk, f, feq, omega)
    perf["forcing_edm"] = timeit(X.forcing_edm, f, g, u)
    perf["macroscopic"] = timeit(X.macroscopic, f)
    perf["equilibrium"] = timeit(X.equilibrium, rho, ux, uy)

    # full step (each streaming variant)
    for tag, sfn in [("step_static_slice", X.stream_static_slice),
                     ("step_roll_std", X.stream_roll_std),
                     ("step_roll_tensorlbm", X.stream_roll)]:
        step = X.make_poiseuille_step(sfn, omega, GX, ny, nx, DT)
        perf[tag] = timeit(step, f)

    ss, rs = perf["stream_static_slice"], perf["stream_roll_std"]
    perf["speedup_static_slice_vs_roll_std"] = rs / ss if ss > 0 else None
    return perf


def make_figures(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # profile comparison
    fig, ax = plt.subplots(figsize=(6, 4))
    y = list(range(NY))
    ana = out["correctness"]["static_slice"]["analytical"]
    ax.plot(y, ana, "-", lw=2, label="Analytical  u=gx/(2nu)·y(H-y)")
    ax.plot(y, out["correctness"]["static_slice"]["profile"], "o",
            label=f"static-slice (vivsim pattern)  err={out['correctness']['static_slice']['err']['max_rel_err']*100:.2f}%")
    ax.plot(y, out["correctness"]["roll_tensorlbm"]["profile"], "s",
            label=f"TensorLBM roll (shifts=-c)  err={out['correctness']['roll_tensorlbm']['err']['max_rel_err']*100:.2f}%")
    ax.set_xlabel("y (wall at y=0, y=NY-1)")
    ax.set_ylabel("u_x (lattice units)")
    ax.set_title("Poiseuille channel  D2Q9 BGK+EDM  nu=0.2 gx=0.001")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3, linestyle=":")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "xval_vivsim_profile.png"), dpi=130)
    plt.close(fig)

    # timing comparison
    perf = out["performance"]
    keys = ["stream_static_slice", "stream_roll_std", "stream_roll_tensorlbm",
             "collide_bgk", "forcing_edm", "macroscopic", "equilibrium"]
    fig, ax = plt.subplots(figsize=(7, 4))
    vals = [perf[k] for k in keys]
    ax.bar(range(len(keys)), vals, color=["#2a7de1", "#e1642a", "#999999", "#777", "#777", "#777", "#777"])
    ax.set_yscale("log")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([k.replace("stream_", "").replace("_tensorlbm", " (TL)") for k in keys], rotation=20, fontsize=8)
    ax.set_ylabel("time (us/call, log)")
    ax.set_title(f"Per-operator timing  {perf['grid']}  (warmup={perf['n_warmup']}, repeat={perf['n_repeat']})")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "xval_vivsim_timing.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
