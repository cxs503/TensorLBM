#!/usr/bin/env python
"""Collect device-level PASS/FAIL evidence for the startup_poiseuille benchmark.

Why this script exists
----------------------
`run.py` writes one JSON per grid.  A verification *claim* needs more than
that: it must record (a) which machine/accelerator actually ran the steps,
(b) whether `torch.compile` really compiled or silently fell back to eager
(`compile_route` does exactly that on SDAA), (c) the a-priori thresholds
re-checked *from the artifacts* rather than trusted from the run log, and
(d) a cross-device reproducibility check (SDAA vs CPU trajectory bit/level
agreement).  This script is read-only over the run artifacts: it never
re-simulates, so the numbers it reports are exactly the ones in the JSONs.

Outputs
-------
* ``evidence.json`` — machine-readable artifact (device evidence + per-grid
  metric vs threshold + verdict + cross-device delta).
* three PNGs (``fig_profiles`` / ``fig_centerline`` / ``fig_convergence``)
  for the report figures.

Usage
-----
    collect_evidence.py sdaa_out --baseline /tmp/sp_H29_cpu_eager.json \\
        --out-json sdaa_out/evidence.json --fig-dir sdaa_out/figures
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import sys
from pathlib import Path

import numpy as np

TOL_PCT = 3.0  # a-priori acceptance threshold (identical on every grid)
N_ODD = np.arange(1, 1000, 2)


def startup_profile(y_phys: np.ndarray, H: float, nu: float, a: float, t: float) -> np.ndarray:
    """Analytic suddenly-started channel-flow series (float64), same as run.py."""
    u_max = a * H * H / (8.0 * nu)
    yh = np.asarray(y_phys, dtype=np.float64) / H
    n = N_ODD.astype(np.float64)
    lam = np.minimum((n * np.pi / H) ** 2 * nu * t, 700.0)
    fac = (1.0 / n**3)[:, None] * np.sin(np.outer(n, yh) * np.pi) * np.exp(-lam)[:, None]
    return u_max * (4.0 * yh * (1.0 - yh) - (32.0 / np.pi**3) * fac.sum(axis=0))


def analytic_centerline(case: dict, tstar: float) -> float:
    t = tstar * case["H_eff"] ** 2 / case["nu_lb"]
    return float(startup_profile(np.array([case["H_eff"] / 2.0]), case["H_eff"], case["nu_lb"], case["a"], t)[0])


def load_cases(scan_dir: Path) -> tuple[list[dict], dict]:
    cases = []
    for p in sorted(scan_dir.glob("case_H*.json")):
        cases.append(json.loads(p.read_text()))
    cases.sort(key=lambda r: r["H"])
    summary = {}
    rp = scan_dir / "result.json"
    if rp.exists():
        summary = json.loads(rp.read_text())
    if not cases:
        raise SystemExit(f"no case_H*.json under {scan_dir}")
    return cases, summary


def check_case(c: dict) -> dict:
    m = {
        "centerline_err_max_over_umax_pct": c["centerline_err_max_over_umax_pct"],
        "profile_l2_max_pct": c["profile_l2_max_pct"],
        "profile_pointwise_max_over_umax_pct": c["profile_pointwise_max_over_umax_pct"],
    }
    return {
        "H": c["H"],
        "H_eff": c["H_eff"],
        "tau": c["tau"],
        "nu_lb": c["nu_lb"],
        "a": c["a"],
        "u_max_final": c["u_max_final"],
        "Ma": c["Ma"],
        "t_end_steps": c["t_end"],
        "metrics_pct": m,
        "thresholds_pct": {
            "centerline": TOL_PCT,
            "profile_l2": TOL_PCT,
            "profile_pointwise": TOL_PCT,
        },
        "pass": bool(all(v <= TOL_PCT for v in m.values())),
        "finite": bool(c["finite"]),
        "mass_drift_pct": c["mass_drift_pct"],
        "compile_mode": c.get("compile_mode"),
        "compile_status": c.get("compile_status"),
        "compile_mode_effective": c.get("compile_mode_effective"),
        "compile_status_reason": (c.get("compile_status_reason") or "")[:400] or None,
        "device_info": c.get("device_info"),
        "elapsed_s": c.get("elapsed_s"),
    }


def cross_device(case_sdaa: dict | None, baseline_path: Path | None) -> dict | None:
    """Level agreement of the final-state profile vs a reference run (other device).

    *baseline_path* may be a single case JSON or a scan directory, in which
    case the case file with the same H is picked (``case_H{H}.json``).
    """
    if case_sdaa is None or baseline_path is None or not Path(baseline_path).exists():
        return None
    baseline_path = Path(baseline_path)
    if baseline_path.is_dir():
        baseline_path = baseline_path / f"case_H{case_sdaa['H']}.json"
        if not baseline_path.exists():
            return None
    b = json.loads(baseline_path.read_text())
    if b["H"] != case_sdaa["H"]:
        return {
            "baseline": str(baseline_path),
            "comparable": False,
            "reason": f"H mismatch: baseline H={b['H']} vs sdaa H={case_sdaa['H']}",
        }
    u_s = np.asarray(case_sdaa["u_final_num"], dtype=np.float64)
    u_b = np.asarray(b["u_final_num"], dtype=np.float64)
    if u_s.shape != u_b.shape:
        return {"baseline": str(baseline_path), "comparable": False, "reason": "shape mismatch"}
    d = np.abs(u_s - u_b)
    return {
        "baseline": str(baseline_path),
        "baseline_device": b.get("device_info", {}).get("device"),
        "baseline_compile_status": b.get("compile_status"),
        "comparable": True,
        "H": b["H"],
        "max_abs_delta_over_umax_pct": float(d.max() / case_sdaa["u_max_final"] * 100.0),
        "rms_delta_over_umax_pct": float(np.sqrt((d**2).mean()) / case_sdaa["u_max_final"] * 100.0),
        "identical_bits": bool(np.array_equal(u_s, u_b)),
        "note": "final state at t* = last mark; both scans use the same a-priori thresholds",
    }


def make_figures(cases: list[dict], fig_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    out = []

    # ---- fig 1: transient velocity profiles, numeric vs analytic (largest H)
    c = cases[-1]
    y = np.asarray(c["y_profile"], dtype=np.float64)
    t_last = max(c["tstar_marks"])
    profs = c.get("profiles") or {}
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    if profs:
        marks = sorted(float(k) for k in profs)
        cmap = plt.get_cmap("viridis")
        for i, ts in enumerate(marks):
            u = np.asarray(profs[str(ts)] if str(ts) in profs else profs[f"{ts:g}"], dtype=np.float64)
            col = cmap(i / max(len(marks) - 1, 1))
            t = ts * c["H_eff"] ** 2 / c["nu_lb"]
            ua = startup_profile(y, c["H_eff"], c["nu_lb"], c["a"], t)
            ax.plot(u, y, "o", ms=3.0, color=col, label=f"LBM t*={ts:g}")
            ax.plot(ua, y, "-", lw=1.0, color=col, alpha=0.85)
        ax.plot([], [], "o", color="0.35", label="LBM (markers)")
        ax.plot([], [], "-", color="0.35", label="Fourier series")
    else:  # fallback: only the final (last-mark) profile is persisted
        u = np.asarray(c["u_final_num"], dtype=np.float64)
        t = t_last * c["H_eff"] ** 2 / c["nu_lb"]
        ua = startup_profile(y, c["H_eff"], c["nu_lb"], c["a"], t)
        ax.plot(u, y, "o", ms=4, label=f"LBM  t*={t_last:g}")
        ax.plot(ua, y, "-", lw=1.4, label="Fourier series")
    ax.set_xlabel("$u_x$  [lattice units]")
    ax.set_ylabel("$y$  [lattice units, half-way walls at 0.5 / $n_y$-1.5]")
    ax.set_title(f"startup Poiseuille, H={c['H']} ($\\tau$={c['tau']}, Ma={c['Ma']:.3f})\n"
                 "transient profiles: LBM vs analytic")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    p = fig_dir / "fig_profiles.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out.append(str(p))

    # ---- fig 2: centreline history + error, numeric vs analytic
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.2))
    for c_i in cases:
        rows = c_i["centerline_table_every10"]
        if not rows:
            continue
        t = np.array([r["tstar"] for r in rows])
        un = np.array([r["u_c_num"] for r in rows])
        ua = np.array([r["u_c_ana"] for r in rows])
        lbl = f"H={c_i['H']}"
        a1.plot(t, un, "o", ms=3, label=f"{lbl} LBM")
        a1.plot(t, ua, "-", lw=1.0, alpha=0.8, label=f"{lbl} series")
        a2.semilogy(t, np.abs(un - ua) / c_i["u_max_final"] * 100.0, "-", lw=1.2,
                    label=f"{lbl} |err|/umax")
    a2.axhline(TOL_PCT, color="r", ls="--", lw=1.0, label=f"threshold {TOL_PCT:g}%")
    a1.set_xlabel("$t^* = \\nu t / H^2$")
    a1.set_ylabel("$u_c$  [lattice units]")
    a1.set_title("centreline response to a suddenly applied force")
    a1.grid(alpha=0.3)
    a1.legend(fontsize=7)
    a2.set_xlabel("$t^* = \\nu t / H^2$")
    a2.set_ylabel("$|u_c^{num}-u_c^{ana}|/u_{max}$  [%]")
    a2.set_title("centreline error history")
    a2.grid(alpha=0.3, which="both")
    a2.legend(fontsize=7)
    fig.tight_layout()
    p = fig_dir / "fig_centerline.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out.append(str(p))

    # ---- fig 3: per-mark profile error + grid convergence vs 3% budget
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.2))
    for c_i in cases:
        pt = c_i["profile_table"]
        ts = [r["tstar"] for r in pt]
        a1.plot(ts, [r["l2_rel"] * 100.0 for r in pt], "o-", ms=4, label=f"H={c_i['H']} L2")
        a1.plot(ts, [r["max_abs_over_umax"] * 100.0 for r in pt], "s--", ms=3.5, alpha=0.8,
                label=f"H={c_i['H']} max pt")
    a1.axhline(TOL_PCT, color="r", ls="--", lw=1.0, label=f"budget {TOL_PCT:g}%")
    a1.set_yscale("log")
    a1.set_xlabel("$t^* = \\nu t / H^2$")
    a1.set_ylabel("profile error  [%]")
    a1.set_title("profile error at the a-priori marks")
    a1.grid(alpha=0.3, which="both")
    a1.legend(fontsize=7)
    Hs = [c_i["H"] for c in cases]
    a2.plot(Hs, [c_i["centerline_err_max_over_umax_pct"] for c_i in cases], "o-",
            label="centreline max")
    a2.plot(Hs, [c_i["profile_l2_max_pct"] for c_i in cases], "s--", label="profile L2 max")
    a2.axhline(TOL_PCT, color="r", ls="--", lw=1.0, label=f"budget {TOL_PCT:g}%")
    a2.set_xlabel("H (channel height, lattice nodes)")
    a2.set_ylabel("max error  [%]")
    a2.set_title("grid convergence H -> 2H")
    a2.grid(alpha=0.3)
    a2.legend(fontsize=7)
    fig.tight_layout()
    p = fig_dir / "fig_convergence.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out.append(str(p))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scan_dir", type=str)
    ap.add_argument("--baseline", type=str, default=None, help="CPU single-case json (H must match)")
    ap.add_argument("--out-json", type=str, default=None)
    ap.add_argument("--fig-dir", type=str, default=None)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    scan_dir = Path(args.scan_dir)
    cases, summary = load_cases(scan_dir)
    checks = [check_case(c) for c in cases]

    conv = {
        "metric": "centerline_err_max_over_umax_pct",
        "values": [c["centerline_err_max_over_umax_pct"] for c in cases],
        "err_decreased_H_to_2H": bool(
            len(cases) < 2 or cases[-1]["centerline_err_max_over_umax_pct"]
            < cases[0]["centerline_err_max_over_umax_pct"]
        ),
    }
    all_pass = all(c["pass"] for c in checks) and conv["err_decreased_H_to_2H"]

    dev = cases[-1].get("device_info") or {}
    evidence = {
        "case": "startup_poiseuille",
        "claim": "transient (start-up) plane-channel flow reaches the analytical Fourier-series "
                 "solution within the a-priori 3% budget on the SDAA device",
        "verdict": "PASS" if all_pass else "FAIL",
        "thresholds_pct": {"centerline": TOL_PCT, "profile_l2": TOL_PCT,
                           "profile_pointwise": TOL_PCT, "grid_convergence": "strictly decreasing"},
        "device_evidence": {
            "requested_device": dev.get("device"),
            "device_name": dev.get("device_name"),
            "torch_version": dev.get("torch_version"),
            "accelerator_backend_version": dev.get("backend_version"),
            "compile_status": cases[-1].get("compile_status"),
            "compile_mode_effective": cases[-1].get("compile_mode_effective"),
            "compile_status_reason": (cases[-1].get("compile_status_reason") or "")[:600] or None,
            "visible_devices_env": __import__("os").environ.get("SDAA_VISIBLE_DEVICES")
            or __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
        },
        "grids": checks,
        "convergence": conv,
        "cross_device": cross_device(cases[-1] if len(cases) == 1 else
                                     next((c for c in cases if c["H"] == 29), None),
                                     Path(args.baseline) if args.baseline else None),
        "scan_summary_status": summary.get("status"),
        "figures": [],
    }

    if not args.no_figures:
        fig_dir = Path(args.fig_dir) if args.fig_dir else scan_dir / "figures"
        try:
            evidence["figures"] = make_figures(cases, fig_dir)
        except Exception as exc:  # figures must never invalidate the evidence
            evidence["figure_error"] = f"{type(exc).__name__}: {exc}"

    out_json = Path(args.out_json) if args.out_json else scan_dir / "evidence.json"
    out_json.write_text(json.dumps(evidence, indent=2))

    print(json.dumps({k: evidence[k] for k in ("case", "verdict", "convergence")}, indent=2))
    for c in checks:
        print(f"H={c['H']:4d}: centre={c['metrics_pct']['centerline_err_max_over_umax_pct']:.3f}% "
              f"l2={c['metrics_pct']['profile_l2_max_pct']:.3f}% "
              f"ptp={c['metrics_pct']['profile_pointwise_max_over_umax_pct']:.3f}% "
              f"pass={c['pass']} compile={c['compile_status']} dev={c['device_info'].get('device')}")
    print(f"evidence -> {out_json}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())