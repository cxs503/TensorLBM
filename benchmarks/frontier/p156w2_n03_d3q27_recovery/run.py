#!/usr/bin/env python3
"""P156W2-N03 · D3Q27 recovery front: lattice audit, 3-D Poiseuille anchor,
Stokes sphere drag, long-run stability.

Every number written here comes from :mod:`tensorlbm.d3q27_recovery`; the
script only orchestrates runs, timings and JSON/PNG output (no hand-copied
numbers anywhere: the report generator reads the JSON emitted by this file).

Sub-commands
------------
audit       exact lattice-algebra audit + per-direction liveness
poiseuille  steady 3-D plane-Poiseuille ladder vs the analytic profile
stability   >= 1e5-step float32 stability gate
sphere      Stokes sphere drag in a periodic box (presets: re_sweep,
            resolution, box, smoke)
plot        figures from the emitted JSONs
merge       combine the partial JSONs into one report payload

Example
-------
    python run.py audit --out artifacts/p156w2_n03_d3q27_recovery
    python run.py poiseuille --out artifacts/p156w2_n03_d3q27_recovery
    python run.py sphere --preset re_sweep --out artifacts/p156w2_n03_d3q27_recovery
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tensorlbm.d3q27_recovery import (  # noqa: E402
    bgk_shear_viscosity,
    direction_liveness_audit,
    equilibrium_third_moment_residual,
    hasimoto_correction,
    moment_audit,
    oseen_correction,
    run_channel_poiseuille,
    run_long_stability,
    run_sphere_stokes,
    schiller_naumann_cd,
    sphere_mask,
)

REFS = {
    "stokes_cd": lambda re: 24.0 / re,  # zero-Re limit of C_D = 24/Re
    "oseen_cd": lambda re: 24.0 / re * oseen_correction(re),
    "schiller_naumann_cd": schiller_naumann_cd,
}


def _env() -> dict:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:  # pragma: no cover - git may be absent in a tarball
        sha, branch = "unknown", "unknown"
    return {
        "git_sha": sha,
        "git_branch": branch,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_threads": torch.get_num_threads(),
        "cpu_count": os.cpu_count(),
        "argv": sys.argv,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _dump(out_dir: Path, name: str, payload: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"[write] {path}", flush=True)
    return path


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------


def cmd_audit(args) -> None:
    t0 = time.time()
    audit = moment_audit()
    third = equilibrium_third_moment_residual()
    live = direction_liveness_audit()
    # the campaign quotes the third-moment identity with e_gamma on the RHS;
    # quantify how far that literal form is from the lattice moments.
    literal = {
        "note": (
            "sum_i w_i c_a c_b c_g == c_s^2 (d_ab c_g + d_bg c_a + d_ga c_b) "
            "can never hold for a centro-symmetric stencil: the left side is "
            "identically zero and the right side is not.  The physically "
            "meaningful identities are sum_i w_i c c c = 0 (checked) and the "
            "equilibrium third moment (checked separately)."
        ),
        "lhs_max_abs": audit.rank3_residual,
        "rhs_scale_for_cs2_form": 1.0 / 3.0,
    }
    payload = {
        "env": _env(),
        "moment_audit": audit.as_dict(),
        "third_moment": {"equilibrium": third, "campaign_literal_form": literal},
        "direction_liveness": {
            "n_directions": live["n_directions"],
            "failures": live["failures"],
            "all_live": live["all_live"],
            "note": live["note"],
        },
        "acceptance": {
            "weight_sum_one": abs(audit.weight_sum - 1.0) < 1e-12,
            "cs2_is_one_third": audit.cs2_error < 1e-12,
            "rank2_isotropic": audit.is_isotropic_rank2,
            "rank3_vanishes": audit.rank3_residual < 1e-12,
            "rank4_isotropic": audit.is_isotropic_rank4,
            "rank6_not_isotropic": not audit.is_isotropic_rank6,
            "all_27_directions_live": live["all_live"],
        },
        "elapsed_s": time.time() - t0,
    }
    _dump(Path(args.out), "lattice_audit.json", payload)


# ---------------------------------------------------------------------------
# viscosity gate
# ---------------------------------------------------------------------------


def cmd_viscosity(args) -> None:
    """BGK effective-viscosity gate: nu_eff(tau, k) from the linearised operator."""
    t0 = time.time()
    taus = [0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 6.5]
    boxs = [16, 24, 32, 48, 64]
    table = []
    for n in boxs:
        k = 2.0 * math.pi / n
        for tau in taus:
            nu_f = (tau - 0.5) / 3.0
            nu_e = bgk_shear_viscosity(tau, k)
            table.append(
                {
                    "box": n,
                    "k": k,
                    "tau": tau,
                    "nu_formula": nu_f,
                    "nu_effective": nu_e,
                    "deviation": nu_e / nu_f - 1.0,
                }
            )
    payload = {
        "env": _env(),
        "table": table,
        "gate": {
            "tau_1p5_max_deviation": max(
                abs(r["deviation"]) for r in table if abs(r["tau"] - 1.5) < 1e-12
            ),
            "tau_2p0_max_deviation": max(
                abs(r["deviation"]) for r in table if abs(r["tau"] - 2.0) < 1e-12
            ),
            "tau_6p5_max_deviation": max(
                abs(r["deviation"]) for r in table if abs(r["tau"] - 6.5) < 1e-12
            ),
        },
        "elapsed_s": time.time() - t0,
    }
    _dump(Path(args.out), "viscosity_gate.json", payload)
    print(f"[viscosity] gate={payload['gate']}", flush=True)


# ---------------------------------------------------------------------------
# poiseuille
# ---------------------------------------------------------------------------


DEFAULT_POISEUILLE_CASES = [
    {"H": 8, "tau": 0.8, "force_density": 1e-5, "n_steps": 8000},
    {"H": 16, "tau": 0.8, "force_density": 1e-5, "n_steps": 20000},
    {"H": 32, "tau": 0.8, "force_density": 1e-5, "n_steps": 40000},
    {"H": 16, "tau": 1.5, "force_density": 1e-5, "n_steps": 30000},
    {"H": 16, "tau": 3.0, "force_density": 1e-5, "n_steps": 30000},
]


def cmd_poiseuille(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cases = DEFAULT_POISEUILLE_CASES
    if args.cases:
        cases = json.loads(Path(args.cases).read_text())
    records = []
    for case in cases:
        t0 = time.time()
        res = run_channel_poiseuille(
            H=case["H"],
            nx=case.get("nx", 4),
            nz=case.get("nz", 4),
            tau=case["tau"],
            force_density=case["force_density"],
            n_steps=case["n_steps"],
            check_every=case.get("check_every", 100),
        )
        rec = res.as_dict()
        rec["elapsed_s"] = time.time() - t0
        nu_f = res.nu
        nu_e = bgk_shear_viscosity(case["tau"], math.pi / case["H"])
        rec["nu_effective"] = nu_e
        rec["nu_deviation"] = nu_e / nu_f - 1.0
        # the analytic profile scales as 1/nu, so the effective-viscosity
        # reference profile is the formula one times nu_f / nu_e
        scale = nu_f / nu_e
        prof_e = [v * scale for v in rec["profile_analytic"]]
        num = math.sqrt(sum((a - b) ** 2 for a, b in zip(rec["profile_lbm"], prof_e)))
        den = math.sqrt(sum(b * b for b in prof_e))
        rec["rel_l2_error_vs_effective_nu"] = num / den if den else float("inf")
        rec["u_max_analytic_effective_nu"] = max(prof_e)
        records.append(rec)
        print(
            f"[poiseuille] H={case['H']} tau={case['tau']} "
            f"l2={rec['rel_l2_error']:.3e} linf={rec['rel_linf_error']:.3e} "
            f"steps={rec['n_steps']} conv={rec['converged']} "
            f"balance={rec['force_balance_error']:.2e} "
            f"t={rec['elapsed_s']:.0f}s",
            flush=True,
        )
        _dump(out, "poiseuille.json", {"env": _env(), "cases": records})

    # observed order on the fixed-tau family (H = 8, 16, 32)
    ladder = [r for r in records if abs(r["tau"] - 0.8) < 1e-12]
    ladder.sort(key=lambda r: r["H"])
    orders = []
    for a, b in zip(ladder, ladder[1:]):
        ratio_h = b["H"] / a["H"]
        orders.append(
            {
                "H_from": a["H"],
                "H_to": b["H"],
                "order": math.log(a["rel_l2_error"] / b["rel_l2_error"])
                / math.log(ratio_h),
            }
        )
    # acceptance is judged on the fixed-tau (0.8) ladder, where the lattice
    # viscosity deviation is <=0.5 %; the large-tau rows exist to *expose* that
    # deviation, not to be judged by the same line.
    ladder08 = [r for r in records if abs(r["tau"] - 0.8) < 1e-12]
    checks = {
        "l2_below_1pct_all_H16plus_tau0.8": all(
            r["rel_l2_error"] < 0.01 for r in ladder08 if r["H"] >= 16
        ),
        "l2_below_1pct_all_H8plus_tau0.8": all(
            r["rel_l2_error"] < 0.01 for r in ladder08
        ),
        "wall_force_balance_below_1pct": all(
            r["force_balance_error"] < 0.01 for r in records
        ),
        "all_finite": all(r["finite"] for r in records),
        "observed_orders": orders,
    }
    _dump(out, "poiseuille.json", {"env": _env(), "cases": records, "checks": checks})


# ---------------------------------------------------------------------------
# stability
# ---------------------------------------------------------------------------


def cmd_stability(args) -> None:
    t0 = time.time()
    res = run_long_stability(
        H=args.H,
        nx=args.nx,
        nz=args.nz,
        tau=args.tau,
        n_steps=args.steps,
        force_density=args.force,
        dtype=torch.float32,
        probe_every=1000,
    )
    payload = {
        "env": _env(),
        "result": res.as_dict(),
        "acceptance": {
            "finite_through_run": res.finite,
            "no_divergence": not res.diverged,
            "density_bounded": 0.1 < res.rho_min and res.rho_max < 10.0,
            "peak_velocity_drift_below_5pct": res.drift_u_max < 0.05,
        },
        "elapsed_s": time.time() - t0,
    }
    _dump(Path(args.out), "stability.json", payload)
    print(f"[stability] {payload['result']}", flush=True)


# ---------------------------------------------------------------------------
# sphere
# ---------------------------------------------------------------------------


def _sigma_estimate(diameter: float, tau: float, n_solid: int, box: int, re_target: float) -> float:
    """Body-force density that hits ``re_target`` under the Hasimoto-corrected
    Stokes estimate (used only as a starting point; the achieved Re is what the
    report quotes)."""
    nu = (tau - 0.5) / 3.0
    a = 0.5 * diameter
    u_target = re_target * nu / (2.0 * a)
    k = hasimoto_correction(a / box)
    v_fluid = box**3 - n_solid
    return u_target * 6.0 * math.pi * nu * a * k / v_fluid


# tau = 1.5 (nu = 1/3) is used everywhere: the BGK effective viscosity deviates
# from (tau-1/2)/3 by <=1 % for the box wavenumbers used here (see the
# viscosity gate), whereas tau = 6.5 deviates by up to -49 % and silently
# corrupts any drag normalised by the formula viscosity.
TAU = 1.5

SPHERE_PRESETS = {
    "smoke": {
        "cases": [
            {"name": "D8_box3_re0.1", "diameter": 8, "box_factor": 3.0, "tau": TAU, "re_target": 0.1, "n_steps": 4000},
        ]
    },
    "resolution": {
        "cases": [
            {"name": "D8_box3_re0.1", "diameter": 8, "box_factor": 3.0, "tau": TAU, "re_target": 0.1, "n_steps": 6000},
            {"name": "D12_box3_re0.1", "diameter": 12, "box_factor": 3.0, "tau": TAU, "re_target": 0.1, "n_steps": 6000},
            {"name": "D16_box3_re0.1", "diameter": 16, "box_factor": 3.0, "tau": TAU, "re_target": 0.1, "n_steps": 5000},
        ]
    },
    "re_sweep": {
        "cases": [
            {"name": "D8_box3_re1", "diameter": 8, "box_factor": 3.0, "tau": TAU, "re_target": 1.0, "n_steps": 8000},
            {"name": "D12_box3_re1", "diameter": 12, "box_factor": 3.0, "tau": TAU, "re_target": 1.0, "n_steps": 8000},
        ]
    },
    "box": {
        "cases": [
            {"name": "D12_box3_re0.1", "diameter": 12, "box_factor": 3.0, "tau": TAU, "re_target": 0.1, "n_steps": 6000},
            {"name": "D8_box4_re0.1", "diameter": 8, "box_factor": 4.0, "tau": TAU, "re_target": 0.1, "n_steps": 6000},
        ]
    },
}


def cmd_sphere(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    preset = json.loads(Path(args.cases).read_text()) if args.cases else SPHERE_PRESETS[args.preset]
    cases = preset["cases"]
    if args.only:
        keep = set(args.only.split(","))
        cases = [c for c in cases if c["name"] in keep]
    fname = f"sphere_{args.preset}.json"
    records = []
    for case in cases:
        t0 = time.time()
        d = case["diameter"]
        bf = case["box_factor"]
        tau = case["tau"]
        n = int(round(bf * d))
        n_solid = int(sphere_mask(n, float(d)).sum().item())
        g = case.get("force_density") or _sigma_estimate(d, tau, n_solid, n, case["re_target"])
        res = run_sphere_stokes(
            diameter=d,
            box_factor=bf,
            tau=tau,
            force_density=g,
            n_steps=case.get("n_steps", args.steps),
            check_every=case.get("check_every", 200),
            dtype=torch.float32 if args.float32 else torch.float64,
            # a uniform initial stream cuts the box-scale transient; at Re >= 1
            # the start-up pulse of a full-strength stream destabilises BGK, so
            # those cases start from rest
            u_init=(
                case["re_target"] * (tau - 0.5) / 3.0 / d
                if case["re_target"] < 1.0
                else 0.0
            ),
            verbose=True,
        )
        rec = res.as_dict()
        rec.update(
            {
                "name": case["name"],
                "re_target": case["re_target"],
                "n_solid": n_solid,
                "elapsed_s": time.time() - t0,
                "references": {
                    "stokes_cd": REFS["stokes_cd"](res.re),
                    "oseen_cd": REFS["oseen_cd"](res.re),
                    "schiller_naumann_cd": REFS["schiller_naumann_cd"](res.re),
                },
                "cd_over_stokes": res.cd_measured / REFS["stokes_cd"](res.re),
                "oseen_cd_over_stokes": oseen_correction(res.re),
                "rel_error_vs_oseen": abs(res.cd_measured - REFS["oseen_cd"](res.re))
                / REFS["oseen_cd"](res.re),
                "rel_error_vs_schiller_naumann": abs(
                    res.cd_measured - REFS["schiller_naumann_cd"](res.re)
                )
                / REFS["schiller_naumann_cd"](res.re),
                "k_from_body_force": res.body_force_total
                / (6.0 * math.pi * res.nu * (0.5 * d) * max(res.mean_u, 1e-30)),
            }
        )
        rec["rel_error_k_from_body_force_vs_hasimoto"] = (
            abs(rec["k_from_body_force"] - res.k_hasimoto) / res.k_hasimoto
        )
        # effective (lattice BGK) viscosity at the box wavenumber: the drag is
        # proportional to mu, so this is the honest normalisation.
        nu_e = bgk_shear_viscosity(tau, 2.0 * math.pi / n)
        rec["nu_effective"] = nu_e
        rec["nu_deviation"] = nu_e / res.nu - 1.0
        k_eff = res.drag_force / (6.0 * math.pi * nu_e * (0.5 * d) * max(res.mean_u, 1e-30))
        rec["k_measured_effective_nu"] = k_eff
        rec["rel_error_vs_hasimoto_effective_nu"] = abs(k_eff - res.k_hasimoto) / res.k_hasimoto
        rec["k_from_body_force_effective_nu"] = res.body_force_total / (
            6.0 * math.pi * nu_e * (0.5 * d) * max(res.mean_u, 1e-30)
        )
        rec["rel_error_vs_hasimoto_body_force_effective_nu"] = (
            abs(rec["k_from_body_force_effective_nu"] - res.k_hasimoto) / res.k_hasimoto
        )
        records.append(rec)
        print(
            f"[sphere] {case['name']} Re={rec['re']:.3f} "
            f"K={rec['k_measured']:.4f} Khas={rec['k_hasimoto']:.4f} "
            f"err={rec['rel_error_vs_hasimoto']*100:.2f}% "
            f"balance={rec['momentum_balance_error']*100:.2f}% "
            f"steps={rec['n_steps']} conv={rec['converged']} t={rec['elapsed_s']:.0f}s",
            flush=True,
        )
        _dump(out, fname, {"env": _env(), "preset": args.preset, "cases": records})


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------


def cmd_plot(args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    data = Path(args.data)
    fig_dir = Path(args.out)
    fig_dir.mkdir(parents=True, exist_ok=True)
    made = []

    # 1. Poiseuille profiles
    pj = json.loads((data / "poiseuille.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for case in pj["cases"]:
        y = np.array(case["profile_y"])
        axes[0].plot(
            np.array(case["profile_lbm"]),
            y,
            "o",
            ms=2.5,
            label=f"H={case['H']:.0f}, τ={case['tau']:.2f}",
        )
    ref = max(pj["cases"], key=lambda c: c["H"])
    axes[0].plot(
        np.array(ref["profile_analytic"]),
        np.array(ref["profile_y"]),
        "k-",
        lw=1.0,
        label="analytic $F/(2\\rho\\nu)\\,y(H-y)$",
    )
    axes[0].set_xlabel("$u_x$ (lattice units)")
    axes[0].set_ylabel("$y$ (distance from lower wall)")
    axes[0].set_title("D3Q27 plane Poiseuille: LBM vs exact")
    axes[0].legend(fontsize=7)
    hs = [c["H"] for c in pj["cases"] if abs(c["tau"] - 0.8) < 1e-12]
    errs = [c["rel_l2_error"] for c in pj["cases"] if abs(c["tau"] - 0.8) < 1e-12]
    axes[1].loglog(hs, errs, "s-", label="relative $L_2$")
    if len(hs) >= 2:
        hh = np.array(hs, dtype=float)
        axes[1].loglog(hh, errs[0] * (hh / hh[0]) ** -2.0, "k--", label="slope $-2$")
    axes[1].axhline(1e-2, color="r", ls=":", label="1 % acceptance line")
    axes[1].set_xlabel("channel gap $H$ (cells)")
    axes[1].set_ylabel("relative $L_2$ error")
    axes[1].set_title("convergence of the analytic anchor")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    f1 = fig_dir / "p156w2_n03_poiseuille.png"
    fig.savefig(f1, dpi=170)
    plt.close(fig)
    made.append(str(f1))

    # 2. sphere: K vs box size and C_D vs Re
    frames = {}
    for name in ("sphere_re_sweep.json", "sphere_resolution.json", "sphere_box.json"):
        p = data / name
        if p.exists():
            frames[name] = json.loads(p.read_text())
    if frames:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        if "sphere_box.json" in frames:
            bx = frames["sphere_box.json"]["cases"]
            x = [c["a_over_l"] for c in bx]
            axes[0].plot(x, [c["k_measured"] for c in bx], "o-", label="LBM (momentum exchange)")
            axes[0].plot(
                x, [c["k_hasimoto"] for c in bx], "s--", label="Hasimoto periodic-array"
            )
            axes[0].axhline(1.0, color="r", ls=":", label="unbounded Stokes ($K=1$)")
            axes[0].set_xlabel("$a/L$ (blockage)")
            axes[0].set_ylabel("$K = F/(6\\pi\\mu a U)$")
            axes[0].set_title("confinement: measured vs Hasimoto")
            axes[0].legend(fontsize=7)
        if "sphere_re_sweep.json" in frames:
            sw = sorted(frames["sphere_re_sweep.json"]["cases"], key=lambda c: c["re"])
            re = np.array([c["re"] for c in sw])
            axes[1].loglog(re, [c["cd_measured"] for c in sw], "o-", label="LBM $C_D$")
            rr = np.logspace(np.log10(re.min()), np.log10(re.max()), 100)
            axes[1].loglog(rr, 24.0 / rr, "k--", label="$24/Re$ (Stokes)")
            axes[1].loglog(rr, 24.0 / rr * (1 + 3 * rr / 16), "b:", label="Oseen")
            axes[1].set_xlabel("Re")
            axes[1].set_ylabel("$C_D$")
            axes[1].set_title("sphere drag vs Reynolds number")
            axes[1].legend(fontsize=7)
        fig.tight_layout()
        f2 = fig_dir / "p156w2_n03_stokes_sphere.png"
        fig.savefig(f2, dpi=170)
        plt.close(fig)
        made.append(str(f2))

        if "sphere_resolution.json" in frames and "sphere_re_sweep.json" in frames:
            rs = frames["sphere_resolution.json"]["cases"]
            fig, ax = plt.subplots(figsize=(5.6, 4.0))
            ax.semilogx(
                [c["diameter"] for c in rs],
                [c["rel_error_vs_hasimoto"] * 100 for c in rs],
                "o-",
                label="vs Hasimoto (periodic)",
            )
            ax.semilogx(
                [c["diameter"] for c in rs],
                [c["rel_error_vs_stokes"] * 100 for c in rs],
                "s--",
                label="vs unbounded Stokes",
            )
            ax.axhline(5.0, color="r", ls=":", label="5 % acceptance")
            ax.set_xlabel("sphere resolution $D$ (cells)")
            ax.set_ylabel("relative drag error (%)")
            ax.set_title("staircase-sphere resolution study (Re≈0.1)")
            ax.legend(fontsize=7)
            fig.tight_layout()
            f3 = fig_dir / "p156w2_n03_sphere_resolution.png"
            fig.savefig(f3, dpi=170)
            plt.close(fig)
            made.append(str(f3))
    for m in made:
        print(f"[figure] {m}", flush=True)


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------


def cmd_merge(args) -> None:
    data = Path(args.data)
    keys = [
        "lattice_audit.json",
        "viscosity_gate.json",
        "poiseuille.json",
        "stability.json",
        "sphere_smoke.json",
        "sphere_resolution.json",
        "sphere_re_sweep.json",
        "sphere_box.json",
    ]
    merged = {"env": _env(), "sources": {}}
    for k in keys:
        p = data / k
        if p.exists():
            merged["sources"][k] = json.loads(p.read_text())
    _dump(data, "report_data.json", merged)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("audit")
    a.add_argument("--out", required=True)
    a.set_defaults(func=cmd_audit)

    v = sub.add_parser("viscosity")
    v.add_argument("--out", required=True)
    v.set_defaults(func=cmd_viscosity)

    p = sub.add_parser("poiseuille")
    p.add_argument("--out", required=True)
    p.add_argument("--cases", default=None, help="JSON file with a list of cases")
    p.set_defaults(func=cmd_poiseuille)

    s = sub.add_parser("stability")
    s.add_argument("--out", required=True)
    s.add_argument("--steps", type=int, default=100_000)
    s.add_argument("--H", type=int, default=16)
    s.add_argument("--nx", type=int, default=4)
    s.add_argument("--nz", type=int, default=4)
    s.add_argument("--tau", type=float, default=0.8)
    s.add_argument("--force", type=float, default=2e-5)
    s.set_defaults(func=cmd_stability)

    sp = sub.add_parser("sphere")
    sp.add_argument("--out", required=True)
    sp.add_argument("--preset", default="smoke", choices=sorted(SPHERE_PRESETS))
    sp.add_argument("--cases", default=None)
    sp.add_argument("--only", default=None)
    sp.add_argument("--steps", type=int, default=6000)
    sp.add_argument("--float32", action="store_true")
    sp.set_defaults(func=cmd_sphere)

    pl = sub.add_parser("plot")
    pl.add_argument("--data", required=True)
    pl.add_argument("--out", required=True)
    pl.set_defaults(func=cmd_plot)

    m = sub.add_parser("merge")
    m.add_argument("--data", required=True)
    m.set_defaults(func=cmd_merge)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()