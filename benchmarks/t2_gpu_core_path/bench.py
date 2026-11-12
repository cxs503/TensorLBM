#!/usr/bin/env python3
"""T2 core-path benchmark — accuracy, backend parity/speed-up, precision tiers.

Runs the unified core path (``src/tensorlbm/gpu_core_path.py``) on two
T3-verified cases with closed-form references and writes machine-readable
artifacts plus figures:

* **channel**  body-force Poiseuille channel vs the analytic parabola
  (two relaxation times × two resolutions);
* **Taylor–Green** vortex decay vs the exact solution (two resolutions × two
  viscosities);
* **backend**  eager (reference) vs ``torch.compile`` (fused kernel): field
  parity and measured MLUPS;
* **precision** FP64FP64 / FP32FP32 / FP32FP16 on the same case.

Everything is written under ``artifacts/t2_gpu_core_path/``; nothing here
reports a speed-up without the accuracy number taken on the same run.

Repro::

    TL_THREADS=1 python benchmarks/t2_gpu_core_path/bench.py            # full
    TL_THREADS=1 python benchmarks/t2_gpu_core_path/bench.py --quick     # CI scale
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from tensorlbm.gpu_core_path import (  # noqa: E402
    GPUCorePathConfig,
    GPUCorePathSolver,
    channel_analytic_velocity,
    run_channel_case,
    run_taylor_green_case,
)

OUT = REPO_ROOT / "artifacts" / "t2_gpu_core_path"


def _figures(payload: dict) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - matplotlib missing
        print(f"figures skipped: {exc}")
        return []
    made = []

    # 1) channel profile: got vs analytic, two resolutions
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for case in payload["channel_profile"]:
        ax.plot(case["u_ref"], case["y"], "k--", lw=1.0, label="analytic (ny=%d)" % case["ny"])
        ax.plot(case["u_got"], case["y"], "o", ms=3, label="core path (ny=%d)" % case["ny"])
    ax.set_xlabel("u_x (lattice)")
    ax.set_ylabel("y (lattice)")
    ax.set_title("Poiseuille channel: core path vs analytic")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    p = OUT / "channel_profile.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    made.append(str(p.relative_to(REPO_ROOT)))

    # 2) Taylor–Green energy decay vs exact
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for case in payload["taylor_green"]:
        steps = [r["step"] for r in case["samples"]]
        ax.semilogy(steps, [r["ke_ref"] for r in case["samples"]], "k--", lw=1.0,
                    label=f"exact (nx={case['cfg']['nx']})")
        ax.semilogy(steps, [r["ke_got"] for r in case["samples"]], "o-", ms=3,
                    label=f"core path (nx={case['cfg']['nx']})")
    ax.set_xlabel("step")
    ax.set_ylabel("mean kinetic energy")
    ax.set_title("Taylor-Green decay vs exact solution")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    p = OUT / "taylor_green_decay.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    made.append(str(p.relative_to(REPO_ROOT)))

    # 3) channel spin-up convergence
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for case in payload["channel_spinup"]:
        steps = [h["step"] for h in case["u_max_history"]]
        ax.semilogx(steps, [100.0 * h["err_pct"] for h in case["u_max_history"]], "-",
                    label=f"ny={case['ny']} (tau={case['tau']})")
    ax.axhline(3.0, color="r", ls=":", lw=1.0, label="3% acceptance")
    ax.set_xlabel("step")
    ax.set_ylabel("|u_max error| (%)")
    ax.set_title("Channel spin-up from rest: error vs step")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    p = OUT / "channel_convergence.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    made.append(str(p.relative_to(REPO_ROOT)))
    return made


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="CI scale (shorter runs)")
    ap.add_argument("--backend-check", action="store_true", default=True)
    args = ap.parse_args(argv)
    torch.set_num_threads(int(os.environ.get("TL_THREADS", "1")))
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    payload: dict = {"generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    # ---- 1. channel: two resolutions x two relaxation times (accuracy) ----
    resolutions = (18, 34) if not args.quick else (18,)
    taus = (0.8, 1.2)
    channel_rows = []
    for ny in resolutions:
        for tau in taus:
            cfg = GPUCorePathConfig(ny=ny, tau=tau, steps=6000 if not args.quick else 3000,
                                    label=f"channel_ny{ny}_tau{tau}")
            row = run_channel_case(cfg, check_every=200, steady_tol=2e-5)
            channel_rows.append(
                {
                    "case": "channel",
                    "ny": ny,
                    "tau": tau,
                    "nu": cfg.nu,
                    "body_force": cfg.body_force,
                    "u_tau": cfg.u_tau,
                    "re_tau": cfg.re_tau,
                    "ma": cfg.ma,
                    "steps": row["steps"],
                    "steady_at_step": row["steady_at_step"],
                    "u_max_ref": row["u_max_ref"],
                    "u_max_got": row["u_max_got"],
                    "u_max_err_pct": 100.0 * row["u_max_rel_err"],
                    "profile_l2_pct": 100.0 * row["profile_rel_l2"],
                    "mass_drift": row["mass_drift"],
                    "mlups": row["mlups"],
                    "device": row["device"],
                    "backend": row["backend"],
                    "u_max_history": [
                        {
                            "step": h["step"],
                            "u_max": h["u_max"],
                            "err_pct": 100.0 * abs(h["u_max"] - row["u_max_ref"]) / row["u_max_ref"],
                        }
                        for h in row["u_max_history"]
                    ],
                }
            )
            print(
                f"channel ny={ny} tau={tau}: u_max err {channel_rows[-1]['u_max_err_pct']:.3f}% "
                f"L2 {channel_rows[-1]['profile_l2_pct']:.3f}% steps={row['steps']}",
                flush=True,
            )
    payload["channel"] = channel_rows
    payload["channel_spinup"] = [
        {
            "ny": r["ny"],
            "tau": r["tau"],
            "u_max_history": r["u_max_history"],
        }
        for r in channel_rows
    ]
    payload["channel_profile"] = []
    for ny in resolutions:
        cfg = GPUCorePathConfig(ny=ny, tau=0.8, steps=6000 if not args.quick else 3000,
                                label=f"profile_ny{ny}")
        row = run_channel_case(cfg, check_every=200, steady_tol=2e-5)
        payload["channel_profile"].append(
            {"ny": ny, "y": row["y"], "u_got": row["u_got"], "u_ref": row["u_ref"]}
        )

    # ---- 2. Taylor-Green: two resolutions x two viscosities ----
    tg_rows = []
    for nx in ((32, 64) if not args.quick else (32,)):
        for tau in (0.6, 0.9):
            cfg = GPUCorePathConfig(
                nx=nx, ny=nx, nz=8, tau=tau, init="taylor_green", tg_u0=0.02,
                steps=(600 if not args.quick else 300), label=f"tg_nx{nx}_tau{tau}",
            )
            row = run_taylor_green_case(cfg, samples=4)
            tg_rows.append(
                {
                    "case": "taylor_green",
                    "cfg": {"nx": nx, "tau": tau, "nu": cfg.nu, "tg_u0": cfg.tg_u0},
                    "steps": row["steps"],
                    "ke0_ref": row["ke0_ref"],
                    "ke0_got": row["ke0_got"],
                    "ke0_err_pct": 100.0 * row["ke0_rel_err"],
                    "max_rel_err_pct": 100.0 * row["max_rel_err"],
                    "mlups": row["mlups"],
                    "samples": row["samples"],
                }
            )
            print(
                f"taylor-green nx={nx} tau={tau}: max KE err "
                f"{tg_rows[-1]['max_rel_err_pct']:.3f}%",
                flush=True,
            )
    payload["taylor_green"] = tg_rows

    # ---- 3. backend parity + throughput (eager vs torch.compile) ----
    parity = []
    for ny in resolutions:
        base = dict(ny=ny, tau=0.8, steps=100, init="parabolic", precision="FP64FP64")
        a = GPUCorePathSolver(GPUCorePathConfig(backend="eager", label="parity_eager", **base))
        a.step(50)
        t0 = time.perf_counter()
        a.step(100)
        t_eager = (time.perf_counter() - t0) / 100.0
        b = GPUCorePathSolver(GPUCorePathConfig(backend="torch_compile", label="parity_compiled", **base))
        t0 = time.perf_counter()
        b.step(1)  # compilation happens here
        compile_s = time.perf_counter() - t0
        b.step(49)
        t0 = time.perf_counter()
        b.step(100)
        t_compiled = (time.perf_counter() - t0) / 100.0
        max_diff = float((a._fa - b._fa).abs().max())
        scale = float(a._fa.abs().max())
        cells = a.cfg.nx * a.cfg.ny * a.cfg.nz
        parity.append(
            {
                "ny": ny,
                "cells": cells,
                "device": str(a.device),
                "eager_ms_per_step": 1e3 * t_eager,
                "compiled_ms_per_step": 1e3 * t_compiled,
                "speedup": t_eager / t_compiled if t_compiled else None,
                "eager_mlups": cells / t_eager / 1e6,
                "compiled_mlups": cells / t_compiled / 1e6,
                "compile_s": compile_s,
                "max_abs_field_diff": max_diff,
                "field_scale": scale,
                "rel_field_diff": max_diff / scale if scale else None,
                "backend_note": b.backend_note,
            }
        )
        print(
            f"backend ny={ny}: eager {1e3 * t_eager:.2f} ms  compiled {1e3 * t_compiled:.2f} ms "
            f"speedup {t_eager / t_compiled:.2f}x  rel diff {max_diff / scale:.2e}",
            flush=True,
        )
    payload["backend_parity"] = parity

    # ---- 4. precision tiers ----
    tiers = []
    for tier in ("FP64FP64", "FP32FP32", "FP32FP16"):
        try:
            cfg = GPUCorePathConfig(
                ny=18, tau=0.8, steps=2000, init="parabolic", precision=tier,
                backend="eager", label=f"precision_{tier}",
            )
            row = run_channel_case(cfg, check_every=500, steady_tol=1e-9)
            tiers.append(
                {
                    "precision": tier,
                    "compute_dtype": row["compute_dtype"],
                    "u_max_err_pct": 100.0 * row["u_max_rel_err"],
                    "profile_l2_pct": 100.0 * row["profile_rel_l2"],
                    "mass_drift": row["mass_drift"],
                    "mlups": row["mlups"],
                }
            )
        except Exception as exc:  # pragma: no cover - e.g. fp16 unsupported
            tiers.append({"precision": tier, "error": f"{type(exc).__name__}: {exc}"})
        print(f"precision {tier}: {tiers[-1]}", flush=True)
    payload["precision"] = tiers

    # ---- 5. multi-GPU path availability (honest not_run on CPU host) ----
    payload["multi_gpu"] = {
        "cuda_available": bool(torch.cuda.is_available()),
        "triton_available": False,
        "status": "not_run",
        "reason": "no CUDA device on this host (torch.cuda.is_available() == False): "
                  "DistributedGPUCorePathSolver and the Triton kernels require CUDA",
    }
    payload["environment"] = {
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "platform": sys.platform,
        "wall_clock_s": time.perf_counter() - t_start,
    }
    payload["figures"] = _figures(payload)

    out = OUT / "bench.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out.relative_to(REPO_ROOT)}  ({payload['environment']['wall_clock_s']:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())