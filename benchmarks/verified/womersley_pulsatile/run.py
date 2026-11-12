#!/usr/bin/env python
"""Womersley pulsatile-flow benchmark driver (case CASE-LBM-WOMERSLEY-PULSATILE).

Reference
---------
Womersley, J. R. (1955). Method for the calculation of velocity, rate of flow
and viscous drag in arteries when the pressure gradient is known.
J. Physiol. 127(3), 553-563.  doi:10.1113/jphysiol.1955.sp005276

Acceptance (predeclared, case pool)
-----------------------------------
At the Womersley number ``alpha = 5`` (whole-period phase averaged):
  * centreline velocity amplitude ratio vs the analytic solution
    ``| |U_num(0)|/|U_ana(0)| - 1 | <= 5 %``
  * centreline phase lag vs analytic, relative
    ``|dphi| / |phi_ana| <= 5 %``
plus a resolution-refinement check (error must not grow from the coarse to the
fine grid).  Two geometries are exercised with the *same* analytic machinery:

  channel : 2D D2Q9, parallel plates (Sexl 1930 form of the same solution)
  pipe    : 3D D3Q19, circular tube (the geometry of Womersley 1955 itself),
            curved wall resolved with BFL interpolated bounce-back

Usage
-----
    python benchmarks/verified/womersley_pulsatile/run.py scan \
        artifacts/womersley_pulsatile_alpha5 --alpha 5 \
        --H 59 119 --R 10.5 20.5 --tau 0.8 \
        --tau-scan 0.7 0.8 0.9 --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "TensorLBM" / "src"))

from tensorlbm.verification.womersley_pulsatile import (  # noqa: E402
    ChannelPulsatileConfig,
    PipePulsatileConfig,
    convergence_verdict,
    run_channel_case,
    run_pipe_case,
)


def _slim(result: dict) -> dict:
    """JSON-friendly copy without the bulky phase-averaged arrays."""
    out = dict(result)
    prof = out.get("profile")
    if prof:
        keep = {
            k: v
            for k, v in prof.items()
            if k
            in (
                "y",
                "r",
                "centerline_phase_averaged",
                "centerline_ana_waveform",
                "phase_centers",
                "profile_ana_phasor_re",
                "profile_ana_phasor_im",
                "profile_num_phasor_re",
                "profile_num_phasor_im",
                "amp_ratio",
                "amp_ratio_masked",
                "complex_rel_pct",
                "complex_rel_pct_masked",
                "phase_lag_num_deg",
                "phase_lag_ana_deg",
                "phase_centers",
            )
        }
        out["profile"] = keep
    return out


def run_channel(H: int, alpha: float, tau: float, device: str, out_dir: Path) -> dict:
    t0 = time.time()
    cfg = ChannelPulsatileConfig(H=H, alpha=alpha, tau=tau, device=device)
    r = run_channel_case(cfg)
    r["case_label"] = f"channel_H{H}_a{alpha:g}_tau{tau:g}"
    r["elapsed_s"] = round(time.time() - t0, 1)
    (out_dir / f"{r['case_label']}.json").write_text(json.dumps(r, indent=1))
    print(
        f"[channel H={H} alpha={alpha:g} tau={tau:g}] "
        f"amp={r['metrics']['amplitude_rel_pct']:.3f}% "
        f"phase={r['metrics']['phase_rel_pct']:.3f}% "
        f"profile_l2={r['metrics']['profile_l2_pct']:.3f}% "
        f"gate={'PASS' if r['gate']['passed'] else 'FAIL'} "
        f"steps={r['total_steps']} elapsed={r['elapsed_s']}s",
        flush=True,
    )
    return r


def run_pipe(R: float, alpha: float, tau: float, out_dir: Path, nx: int = 2) -> dict:
    t0 = time.time()
    cfg = PipePulsatileConfig(R=R, alpha=alpha, tau=tau, nx=nx)
    r = run_pipe_case(cfg)
    r["case_label"] = f"pipe_R{R:g}_a{alpha:g}_tau{tau:g}"
    r["elapsed_s"] = round(time.time() - t0, 1)
    (out_dir / f"{r['case_label']}.json").write_text(json.dumps(r, indent=1))
    print(
        f"[pipe R={R} alpha={alpha:g} tau={tau:g}] "
        f"amp={r['metrics']['amplitude_rel_pct']:.3f}% "
        f"phase={r['metrics']['phase_rel_pct']:.3f}% "
        f"profile_l2={r['metrics']['profile_l2_pct']:.3f}% "
        f"gate={'PASS' if r['gate']['passed'] else 'FAIL'} "
        f"steps={r['total_steps']} elapsed={r['elapsed_s']}s",
        flush=True,
    )
    return r


def run_pipe_steady(R: float, tau: float, out_dir: Path) -> dict:
    cfg = PipePulsatileConfig(R=R, alpha=5.0, tau=tau, steady=True)
    r = run_pipe_case(cfg)
    r["case_label"] = f"pipe_steady_R{R:g}_tau{tau:g}"
    (out_dir / f"{r['case_label']}.json").write_text(json.dumps(r, indent=1))
    print(
        f"[pipe steady R={R}] centreline rel={r['u_center_rel_pct']:.3f}% "
        f"profile_l2={r['profile_l2_pct']:.3f}% steps={r['steps']}",
        flush=True,
    )
    return r


def scan(
    out_dir: Path,
    alpha: float = 5.0,
    H_list=(59, 119),
    R_list=(10.5, 20.5),
    tau: float = 0.8,
    tau_scan=(),
    device: str = "cpu",
    geometry: str = "both",
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "case": "CASE-LBM-WOMERSLEY-PULSATILE",
        "reference": {
            "citation": "Womersley 1955, J. Physiol. 127(3):553-563",
            "doi": "10.1113/jphysiol.1955.sp005276",
            "also": "Sexl 1930 (parallel-plate form of the same solution)",
        },
        "acceptance": {
            "alpha": alpha,
            "amplitude_tolerance_pct": 5.0,
            "phase_tolerance_pct": 5.0,
            "metric": "whole-period phase-averaged centreline amplitude ratio and "
            "phase lag vs analytic",
        },
        "alpha": alpha,
        "tau": tau,
        "channel_H": list(H_list),
        "pipe_R": list(R_list),
        "tau_scan": list(tau_scan),
        "geometry": geometry,
    }

    channel_cases: list[dict] = []
    pipe_cases: list[dict] = []
    if geometry in ("both", "channel"):
        for H in H_list:
            channel_cases.append(run_channel(H, alpha, tau, device, out_dir))
        summary["channel_resolution_check"] = convergence_verdict(channel_cases)
    if geometry in ("both", "pipe"):
        for R in R_list:
            pipe_cases.append(run_pipe(R, alpha, tau, out_dir))
        summary["pipe_resolution_check"] = convergence_verdict(pipe_cases)
        summary["pipe_steady_wall_audit"] = [
            run_pipe_steady(R, tau, out_dir) for R in R_list
        ]

    tau_cases: list[dict] = []
    if tau_scan and geometry in ("both", "channel"):
        for t in tau_scan:
            tau_cases.append(run_channel(max(H_list), alpha, t, device, out_dir))
        summary["tau_sensitivity"] = [
            {
                "tau": c["tau"],
                "nu": c["nu"],
                "omega": c["omega"],
                "period_steps": c["period_steps"],
                "amplitude_rel_pct": c["metrics"]["amplitude_rel_pct"],
                "phase_rel_pct": c["metrics"]["phase_rel_pct"],
                "passed": c["gate"]["passed"],
            }
            for c in tau_cases
        ]

    per_case = [
        {
            k: c[k]
            for k in (
                "case_label",
                "geometry",
                "resolution",
                "alpha",
                "tau",
                "nu",
                "omega",
                "period_steps",
                "a0",
                "u_peak_ana",
                "Ma",
                "total_steps",
                "elapsed_s",
            )
        }
        | {"metrics": c["metrics"], "passed": c["gate"]["passed"]}
        for c in channel_cases + pipe_cases + tau_cases
    ]
    summary["cases"] = per_case
    summary["all_cases_passed"] = all(
        c["gate"]["passed"] for c in channel_cases + pipe_cases
    )
    conv_ok = all(
        summary.get(k, {}).get("passed", True)
        for k in ("channel_resolution_check", "pipe_resolution_check")
    )
    summary["convergence_passed"] = bool(conv_ok)
    summary["status"] = (
        "VERIFIED" if summary["all_cases_passed"] and conv_ok else "NOT_PASSED"
    )
    (out_dir / "result.json").write_text(json.dumps(_slim(summary), indent=1))
    print(f"\nstatus={summary['status']} convergence={summary['convergence_passed']}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)

    p = sub.add_parser("scan")
    p.add_argument("out_dir")
    p.add_argument("--alpha", type=float, default=5.0)
    p.add_argument("--H", type=int, nargs="+", default=[59, 119])
    p.add_argument("--R", type=float, nargs="+", default=[10.5, 20.5])
    p.add_argument("--tau", type=float, default=0.8)
    p.add_argument("--tau-scan", type=float, nargs="*", default=[])
    p.add_argument("--geometry", choices=["both", "channel", "pipe"], default="both")
    p.add_argument("--device", default="cpu")

    args = ap.parse_args()
    scan(
        Path(args.out_dir),
        alpha=args.alpha,
        H_list=tuple(args.H),
        R_list=tuple(args.R),
        tau=args.tau,
        tau_scan=tuple(args.tau_scan),
        device=args.device,
        geometry=args.geometry,
    )


if __name__ == "__main__":
    main()