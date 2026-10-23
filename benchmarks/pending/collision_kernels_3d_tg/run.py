#!/usr/bin/env python
"""W5-C: collision-kernel coverage benchmark — 3D Taylor-Green vortex decay.

Covers the previously unbenchmarked collision kernels of tensorlbm with the
same protocol as benchmarks/verified/taylor_green_3d (D3Q19 BGK precedent):

    kernels : collide_cumulant_d3q27, collide_cascaded_d3q27, collide_mrt27,
              collide_kbc_d3q19, collide_trt27, collide_rlbm27
    physics : periodic N^3, u = (U0 sin(kx)cos(ky)cos(kz),
              -U0 cos(kx)sin(ky)cos(kz), 0), k = 2*pi/N,
              nu = U0*N/Re, tau = 0.5 + 3*nu, Re = 24, U0 = 0.05
    metric  : kinetic-energy decay rate gamma_E_sim = -d(ln E)/dt (LSQ over
              recorded samples, t >= record_every) vs gamma_E = 6*nu*k^2
              (modes (+-k,+-k,+-k) => |kappa|^2 = 3k^2; identical for D3Q19
              and D3Q27 since both have cs^2 = 1/3 and nu = (tau-1/2)/3)
    verdict : |err_E| <= 3% at every tier, |err_E| strictly decreasing across
              the ladder, R^2 >= 0.999, >= 2 tiers (precedent judge() semantics,
              including the err_vel <= 3% secondary gate)

All collide/stream/equilibrium/macroscopic operations are library calls from
the pinned read-only worktree; this script only orchestrates initialization
(analytic TG field -> library equilibrium) and observation.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch

WORKTREE = "/nfs/wangxi/worktrees/bm_w5"
WORKTREE_SRC = WORKTREE + "/src"
sys.path.insert(0, WORKTREE_SRC)

import tensorlbm  # noqa: E402

assert tensorlbm.__file__.startswith(WORKTREE_SRC), f"wrong tensorlbm copy: {tensorlbm.__file__}"

from tensorlbm.cascaded_collision import collide_cascaded_d3q27  # noqa: E402
from tensorlbm.cumulant import collide_cumulant_d3q27  # noqa: E402
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d  # noqa: E402
from tensorlbm.d3q27 import (  # noqa: E402
    collide_bgk27,
    collide_mrt27,
    collide_rlbm27,
    collide_trt27,
    equilibrium27,
    macroscopic27,
    stream27_roll,
)
from tensorlbm.entropic_kbc import collide_kbc_d3q19  # noqa: E402
from tensorlbm.solver3d import stream3d  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CANONICAL = dict(re=24.0, u0=0.05, dtype="float32")
LADDER = (64, 96, 128)

# Kernel registry: every entry uses the library collision operator with its
# documented default relaxation parameters (pre-registered in NOTES.md).
KERNELS = {
    "cumulant_d3q27": {
        "lattice": "d3q27",
        "op": lambda f, tau: collide_cumulant_d3q27(f, tau),
        "params": "tau; omega_b=1.0, omega_odd=1.0, omega_even=1.0 (defaults)",
        "module": "tensorlbm.cumulant.collide_cumulant_d3q27",
    },
    "cascaded_d3q27": {
        "lattice": "d3q27",
        "op": lambda f, tau: collide_cascaded_d3q27(f, tau),
        "params": "tau; s_bulk=1.0, s_3=1.0, s_4=1.0 (defaults)",
        "module": "tensorlbm.cascaded_collision.collide_cascaded_d3q27",
    },
    "mrt27": {
        "lattice": "d3q27",
        "op": lambda f, tau: collide_mrt27(f, tau),
        "params": "tau; s_e=1.19, s_eps=1.4, s_q=1.2 (defaults)",
        "module": "tensorlbm.d3q27.collide_mrt27",
    },
    "kbc_d3q19": {
        "lattice": "d3q19",
        "op": lambda f, tau: collide_kbc_d3q19(f, tau),
        "params": "tau; max_iter=28, tol=1e-8 (defaults)",
        "module": "tensorlbm.entropic_kbc.collide_kbc_d3q19",
    },
    "trt27": {
        "lattice": "d3q27",
        "op": lambda f, tau: collide_trt27(f, tau),
        "params": "tau_plus=tau; lambda_trt=3/16 (default, Ginzburg 2008)",
        "module": "tensorlbm.d3q27.collide_trt27",
    },
    "rlbm27": {
        "lattice": "d3q27",
        "op": lambda f, tau: collide_rlbm27(f, tau),
        "params": "tau (no extra parameters)",
        "module": "tensorlbm.d3q27.collide_rlbm27",
    },
    # Diagnostic reference control only (chain entry 2026-09-21T07:0xZ):
    # same lattice as the D3Q27 kernels, same collision as the verified
    # D3Q19 precedent -- attributes the ladder err trend to lattice+tau
    # coupling vs kernel. Not one of the six pre-registered kernels.
    "bgk27": {
        "lattice": "d3q27",
        "op": lambda f, tau: collide_bgk27(f, tau),
        "params": "tau (diagnostic control)",
        "module": "tensorlbm.d3q27.collide_bgk27",
    },
}

LATTICE_FUNCS = {
    "d3q19": dict(equilibrium=equilibrium3d, macroscopic=macroscopic3d, streaming=stream3d),
    "d3q27": dict(equilibrium=equilibrium27, macroscopic=macroscopic27, streaming=stream27_roll),
}


def worktree_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", WORKTREE, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def auto_steps(
    n: int,
    re: float,
    u0: float,
    record_every: int,
    target_efolds: float = 10.0,
    cap: int = 10000,
    min_steps: int = 2000,
) -> int:
    """Steps so that E/E0 decays to e^-target_efolds (precedent rule)."""
    nu = u0 * n / re
    k = 2.0 * math.pi / n
    gamma_e = 6.0 * nu * k * k
    steps = int(math.ceil(target_efolds / gamma_e / record_every) * record_every)
    return min(max(steps, min_steps), cap)


def build_initial_field(n: int, u0: float, dt, device, lattice: str):
    """Analytic 3D Taylor-Green initial field -> library equilibrium."""
    z, y, x = torch.meshgrid(
        torch.arange(n, device=device, dtype=dt),
        torch.arange(n, device=device, dtype=dt),
        torch.arange(n, device=device, dtype=dt),
        indexing="ij",
    )
    ux0 = u0 * torch.sin(k_of(n) * x) * torch.cos(k_of(n) * y) * torch.cos(k_of(n) * z)
    uy0 = -u0 * torch.cos(k_of(n) * x) * torch.sin(k_of(n) * y) * torch.cos(k_of(n) * z)
    uz0 = torch.zeros_like(ux0)
    rho = torch.ones_like(ux0)
    f = LATTICE_FUNCS[lattice]["equilibrium"](rho, ux0, uy0, uz0)
    return f, ux0, uy0


def k_of(n: int) -> float:
    return 2.0 * math.pi / float(n)


def run_case(
    kernel: str,
    n: int,
    re: float,
    u0: float,
    steps: int,
    record_every: int = 50,
    device: str = "cuda:5",
    dtype: str = "float32",
) -> dict:
    info = KERNELS[kernel]
    lattice = info["lattice"]
    funcs = LATTICE_FUNCS[lattice]
    dt = torch.float64 if dtype == "float64" else torch.float32

    k = k_of(n)
    nu = u0 * n / re
    tau = 0.5 + 3.0 * nu
    gamma_e_theory = 6.0 * nu * k * k
    gamma_vel_theory = 3.0 * nu * k * k
    gamma_task_formula = 2.0 * nu * k * k  # 2D-TG velocity rate, documented only
    re_eff = u0 / (nu * k)

    f, ux0, uy0 = build_initial_field(n, u0, dt, device, lattice)
    mass0 = float(f.sum().item())
    e0_theory = u0 * u0 / 8.0

    op = info["op"]

    times: list[int] = [0]
    energies: list[float] = [0.0]
    exs: list[float] = [0.0]
    eys: list[float] = [0.0]
    ezs: list[float] = [0.0]
    umaxs: list[float] = [0.0]
    wmaxs: list[float] = [0.0]
    wall0 = time.time()
    for step in range(1, steps + 1):
        f = op(funcs["streaming"](f), tau)
        if step % record_every == 0:
            _, uxm, uym, uzm = funcs["macroscopic"](f)
            ex = float((0.5 * (uxm * uxm)).mean().item())
            ey = float((0.5 * (uym * uym)).mean().item())
            ez = float((0.5 * (uzm * uzm)).mean().item())
            um = float((uxm * uxm + uym * uym + uzm * uzm).sqrt().max().item())
            wm = float(uzm.abs().max().item())
            times.append(step)
            energies.append(ex + ey + ez)
            exs.append(ex)
            eys.append(ey)
            ezs.append(ez)
            umaxs.append(um)
            wmaxs.append(wm)
    wall = time.time() - wall0

    ex0 = float((0.5 * (ux0 * ux0)).mean().item())
    ey0 = float((0.5 * (uy0 * uy0)).mean().item())
    energies[0] = ex0 + ey0
    exs[0] = ex0
    eys[0] = ey0
    ezs[0] = 0.0
    umaxs[0] = float((ux0 * ux0 + uy0 * uy0).sqrt().max().item())
    wmaxs[0] = 0.0

    t = np.asarray(times[1:], dtype=np.float64)
    lnE = np.log(np.asarray(energies[1:], dtype=np.float64))
    a_e, b_e = np.polyfit(t, lnE, 1)
    gamma_e_sim = -a_e
    resid = lnE - (b_e + a_e * t)
    r2 = 1.0 - float(np.sum(resid**2) / np.sum((lnE - lnE.mean()) ** 2))

    urms = np.sqrt(2.0 * np.asarray(energies, dtype=np.float64))
    lnU = np.log(urms[1:])
    a_u, _ = np.polyfit(t, lnU, 1)
    gamma_vel_sim = -a_u

    half = len(t) // 2
    g_h1 = -float(np.polyfit(t[:half], lnE[:half], 1)[0])
    g_h2 = -float(np.polyfit(t[half:], lnE[half:], 1)[0])

    finite = bool(np.isfinite(energies).all())
    mass_end = float(f.sum().item())

    return {
        "kernel": kernel,
        "lattice": lattice,
        "n": n,
        "re": re,
        "u0": u0,
        "nu": nu,
        "tau": tau,
        "k": k,
        "re_eff": re_eff,
        "steps": steps,
        "record_every": record_every,
        "dtype": dtype,
        "device": device,
        "finite": finite,
        "gamma_e_theory": gamma_e_theory,
        "gamma_e_sim": gamma_e_sim,
        "err_e_pct": (gamma_e_sim - gamma_e_theory) / gamma_e_theory * 100.0,
        "gamma_vel_theory": gamma_vel_theory,
        "gamma_vel_sim": gamma_vel_sim,
        "err_vel_pct": (gamma_vel_sim - gamma_vel_theory) / gamma_vel_theory * 100.0,
        "gamma_task_formula_2d": gamma_task_formula,
        "err_vs_task_formula_pct": (gamma_e_sim - gamma_task_formula) / gamma_task_formula * 100.0,
        "r2": r2,
        "gamma_e_half1": g_h1,
        "gamma_e_half2": g_h2,
        "e0_theory": e0_theory,
        "e0_meas": energies[0],
        "ez0_meas": ezs[0],
        "ez_max": max(ezs),
        "w_max": max(wmaxs),
        "mass_drift_rel": (mass_end - mass0) / mass0,
        "n_samples": len(times) - 1,
        "wall_sec": wall,
        "times": times,
        "energies": energies,
        "exs": exs,
        "eys": eys,
        "ezs": ezs,
        "umaxs": umaxs,
        "wmaxs": wmaxs,
    }


def save_case(res: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    keep = {
        k: v
        for k, v in res.items()
        if k not in ("times", "energies", "exs", "eys", "ezs", "umaxs", "wmaxs")
    }
    name = f"case_{res['kernel']}_N{res['n']}.json"
    with open(os.path.join(out_dir, name), "w") as fh:
        json.dump(keep, fh, indent=2, ensure_ascii=False)
    hist = np.column_stack(
        [
            res["times"],
            res["energies"],
            res["exs"],
            res["eys"],
            res["ezs"],
            res["umaxs"],
            res["wmaxs"],
        ]
    )
    np.savetxt(
        os.path.join(out_dir, f"energy_history_{res['kernel']}_N{res['n']}.csv"),
        hist,
        header="step,energy,ex,ey,ez,umax,wmax",
        delimiter=",",
        comments="",
    )
    return name


def judge(kernel: str, cases: list[dict], tol_pct: float = 3.0) -> dict:
    """Grid-convergence judgment, precedent semantics (finite tiers only)."""
    usable = [c for c in cases if c.get("finite", True)]
    dropped = [c["n"] for c in cases if not c.get("finite", True)]
    by_n = {c["n"]: c for c in usable}
    ns = sorted(by_n)
    if not ns:
        return {"kernel": kernel, "verified": False, "judgment": "FAIL: no finite tier"}
    errs = [by_n[n]["err_e_pct"] for n in ns]
    errs_v = [by_n[n]["err_vel_pct"] for n in ns]
    r2s = [by_n[n]["r2"] for n in ns]
    converged = all(abs(errs[i + 1]) < abs(errs[i]) for i in range(len(errs) - 1))
    within = all(abs(e) <= tol_pct for e in errs)
    within_v = all(abs(e) <= tol_pct for e in errs_v)
    expo = all(r2 >= 0.999 for r2 in r2s)
    verified = bool(within and within_v and converged and expo and len(ns) >= 2)
    return {
        "kernel": kernel,
        "grids": ns,
        "dropped_nonfinite": dropped,
        "err_e_pct_per_grid": dict(zip(ns, errs)),
        "err_vel_pct_per_grid": dict(zip(ns, errs_v)),
        "r2_per_grid": dict(zip(ns, r2s)),
        "converged_monotone": converged,
        "within_tol": within,
        "within_tol_vel": within_v,
        "exponential_r2_ok": expo,
        "tol_pct": tol_pct,
        "verified": verified,
        "judgment": (
            "PASS: |err_E| <= 3% every tier, strictly decreasing with N, R^2 >= 0.999"
            if verified
            else "FAIL: see per-tier err_e_pct / monotonicity / R^2"
        ),
    }


def env_block(device: str) -> dict:
    return {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "tensorlbm_path": tensorlbm.__file__,
        "worktree_commit": worktree_commit(),
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(torch.device(device))
        if torch.device(device).type == "cuda"
        else "cpu",
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="W5-C collision-kernel TG3D benchmark")
    ap.add_argument("--kernel", default="all", choices=["all", *KERNELS])
    ap.add_argument("--n", type=int, default=0, help="0 = full ladder 64/96/128")
    ap.add_argument("--re", type=float, default=CANONICAL["re"])
    ap.add_argument("--u0", type=float, default=CANONICAL["u0"])
    ap.add_argument("--steps", type=int, default=0, help="0 = auto")
    ap.add_argument("--record-every", type=int, default=50)
    ap.add_argument("--dtype", choices=["float32", "float64"], default=CANONICAL["dtype"])
    ap.add_argument("--device", default="cuda:5")
    ap.add_argument("--out", default=HERE)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    names = list(KERNELS) if args.kernel == "all" else [args.kernel]
    result_path = os.path.join(args.out, "result.json")
    result = {"benchmark": "kernel_coverage_taylor_green_3d", "kernels": {}}
    if os.path.exists(result_path):
        with open(result_path) as fh:
            result = json.load(fh)
    result.setdefault("env", env_block(args.device))
    result["env"] = env_block(args.device)
    result["protocol"] = (
        "3D Taylor-Green decay, periodic N^3, u0=0.05, Re=24 (tau=0.5+3*nu per tier), "
        "gamma_E_theory = 6*nu*k^2 with nu=u0*N/Re, k=2*pi/N; LSQ fit of ln E over "
        "recorded samples; verdict: |err_E|<=3% all tiers AND strictly decreasing "
        "AND R^2>=0.999 AND err_vel<=3% (precedent taylor_green_3d semantics)"
    )

    for name in names:
        grids = LADDER if args.n == 0 else [args.n]
        cases, case_files = [], []
        for n in grids:
            steps = args.steps or auto_steps(n, args.re, args.u0, args.record_every)
            print(
                f"=== {name} N={n}^3 Re={args.re} U0={args.u0} steps={steps} "
                f"dtype={args.dtype} device={args.device} ===",
                flush=True,
            )
            res = run_case(
                name,
                n,
                args.re,
                args.u0,
                steps,
                record_every=args.record_every,
                device=args.device,
                dtype=args.dtype,
            )
            case_files.append(save_case(res, args.out))
            cases.append(res)
            print(
                f"  gamma_E_sim={res['gamma_e_sim']:.6e} theory={res['gamma_e_theory']:.6e} "
                f"err_E={res['err_e_pct']:+.4f}% err_vel={res['err_vel_pct']:+.4f}% "
                f"R2={res['r2']:.6f} wall={res['wall_sec']:.1f}s",
                flush=True,
            )
        verdict = judge(name, cases)
        result["kernels"][name] = {
            "lattice": KERNELS[name]["lattice"],
            "module": KERNELS[name]["module"],
            "params": KERNELS[name]["params"],
            "cases": [
                {
                    k: v
                    for k, v in r.items()
                    if k not in ("times", "energies", "exs", "eys", "ezs", "umaxs", "wmaxs")
                }
                for r in cases
            ],
            "case_files": case_files,
            "convergence": verdict,
        }
        print(f"  verdict: {verdict['judgment']}", flush=True)
        with open(result_path, "w") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)

    print(f"-> {result_path}")


if __name__ == "__main__":
    main()
