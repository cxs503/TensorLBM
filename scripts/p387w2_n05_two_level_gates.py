#!/usr/bin/env python3
"""Run the p387w2-n05 two-level static-refinement capability gate.

Usage
-----
    python scripts/p387w2_n05_two_level_gates.py --out artifacts/p387w2_n05_gates.json
    python scripts/p387w2_n05_two_level_gates.py --cylinder-only --out artifacts/p387w2_n05_cylinder.json

The script writes a JSON artifact (all gate numbers, programmatically
injected, no hand copying) and, unless ``--no-figures`` is given, the figures
consumed by the mission report.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.two_level_gates import (  # noqa: E402
    gate_conservation,
    gate_cylinder,
    gate_negative_controls,
    gate_order,
    gate_poiseuille,
    gate_reflection,
)


def _print(name: str, payload: dict, elapsed: float) -> None:
    keys = [
        k
        for k in (
            "passed",
            "mass_residual_relative",
            "momentum_residual_relative",
            "profile_error_two_level",
            "profile_error_uniform_coarse",
            "reflection_coefficient",
            "worst_reflection_coefficient",
            "observed_order",
            "c_d_error_two_level",
            "c_d_error_baseline",
            "error_reduction",
        )
        if k in payload
    ]
    summary = {k: payload[k] for k in keys}
    print(f"[{name}] {elapsed:7.1f}s {json.dumps(summary, default=float)}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/p387w2_n05_gates.json")
    parser.add_argument("--figures", default="figures/p387w2_n05")
    parser.add_argument("--threads", type=int, default=int(os.environ.get("TP_TORCH_THREADS", "4")))
    parser.add_argument("--conservation-steps", type=int, default=200)
    parser.add_argument("--poiseuille-steps", type=int, default=600)
    parser.add_argument("--reflection-steps", type=int, default=300)
    parser.add_argument("--order-resolutions", default="32,64,128")
    parser.add_argument("--cylinder-steps", type=int, default=12000)
    parser.add_argument("--cylinder-only", action="store_true")
    parser.add_argument("--with-cylinder", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(max(args.threads, 1))
    resolutions = tuple(int(v) for v in args.order_resolutions.split(","))

    gates: dict[str, dict] = {}
    timings: dict[str, float] = {}

    if not args.cylinder_only:
        t0 = time.time()
        gates["conservation"] = gate_conservation(steps=args.conservation_steps)
        timings["conservation"] = time.time() - t0
        _print("conservation", gates["conservation"], timings["conservation"])

        t0 = time.time()
        gates["poiseuille"] = gate_poiseuille(steps=args.poiseuille_steps)
        timings["poiseuille"] = time.time() - t0
        _print("poiseuille", gates["poiseuille"], timings["poiseuille"])

        t0 = time.time()
        gates["reflection"] = gate_reflection(steps=args.reflection_steps)
        timings["reflection"] = time.time() - t0
        _print("reflection", gates["reflection"], timings["reflection"])

        t0 = time.time()
        gates["order"] = gate_order(resolutions=resolutions)
        timings["order"] = time.time() - t0
        _print("order", gates["order"], timings["order"])

        t0 = time.time()
        gates["negative_control"] = gate_negative_controls()
        timings["negative_control"] = time.time() - t0
        _print("negative_control", gates["negative_control"], timings["negative_control"])

    if args.cylinder_only or args.with_cylinder:
        t0 = time.time()
        gates["cylinder"] = gate_cylinder(steps=args.cylinder_steps)
        timings["cylinder"] = time.time() - t0
        _print("cylinder", gates["cylinder"], timings["cylinder"])

    passed = {name: bool(cell.get("passed")) for name, cell in gates.items()}
    artifact = {
        "suite": "p387w2-n05 two-level static refinement gate",
        "module": "src/tensorlbm/two_level_amr2d.py",
        "gates": gates,
        "passed": passed,
        "all_passed": all(passed.values()),
        "timings_seconds": timings,
        "torch_threads": torch.get_num_threads(),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=1, default=float))
    print(f"wrote {out}  all_passed={artifact['all_passed']}", flush=True)

    if not args.no_figures and not args.cylinder_only:
        make_figures(artifact, Path(args.figures))
    return 0


def make_figures(artifact: dict, outdir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    gates = artifact["gates"]

    # 1 --- Poiseuille profiles
    if "poiseuille" in gates:
        p = gates["poiseuille"]
        fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
        y = list(range(len(p["profile_analytic"])))
        ax[0].plot(p["profile_analytic"], y, "k-", label="analytic")
        ax[0].plot(p["profile_uniform_coarse"], y, "o", ms=4, label="uniform coarse")
        ax[0].plot(p["profile_two_level"], y, "s", ms=4, label="two-level (refined band)")
        ax[0].set_xlabel("$u_x/u_{max}$")
        ax[0].set_ylabel("wall-normal cell index")
        ax[0].set_title("Poiseuille profile through the refined band")
        ax[0].legend(fontsize=8)
        ax[0].grid(alpha=0.3)
        err_two = [
            abs(a - b) / max(abs(c) for c in p["profile_analytic"])
            for a, b in zip(p["profile_two_level"], p["profile_analytic"])
        ]
        err_uni = [
            abs(a - b) / max(abs(c) for c in p["profile_analytic"])
            for a, b in zip(p["profile_uniform_coarse"], p["profile_analytic"])
        ]
        ax[1].semilogy(y, [max(v, 1e-12) for v in err_two], "s-", label="two-level")
        ax[1].semilogy(y, [max(v, 1e-12) for v in err_uni], "o-", label="uniform coarse")
        ax[1].axhline(p["tolerance"], color="r", ls="--", label="1% gate")
        ax[1].set_xlabel("wall-normal cell index")
        ax[1].set_ylabel("relative profile error")
        ax[1].legend(fontsize=8)
        ax[1].grid(alpha=0.3, which="both")
        fig.tight_layout()
        fig.savefig(outdir / "poiseuille_profile.png", dpi=150)
        plt.close(fig)

    # 2 --- acoustic reflection
    if "reflection" in gates:
        r = gates["reflection"]
        tr = r.get("traces_sigma4") or {}
        fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
        if tr:
            ax[0].plot(tr["incident"], label="incident (uniform reference)")
            ax[0].plot(tr["total"], label="two-level probe signal")
            ax[0].plot(tr["reflected"], label="difference = spurious reflection")
            ax[0].set_xlabel("time step")
            ax[0].set_ylabel(r"$\Delta\rho$ at probe")
            ax[0].set_title(r"Pulse $\sigma=4$, normal incidence on the 2:1 interface")
            ax[0].legend(fontsize=8)
            ax[0].grid(alpha=0.3)
        sigmas = [row["pulse_sigma"] for row in r["per_sigma"].values()]
        coeffs = [100.0 * row["reflection_coefficient"] for row in r["per_sigma"].values()]
        ax[1].semilogy(sigmas, coeffs, "o-")
        ax[1].axhline(100.0 * r["tolerance"], color="r", ls="--", label="1% gate")
        ax[1].set_xlabel(r"pulse width $\sigma$ [coarse cells]")
        ax[1].set_ylabel("spurious reflection R [%]")
        ax[1].legend(fontsize=8)
        ax[1].grid(alpha=0.3, which="both")
        fig.tight_layout()
        fig.savefig(outdir / "acoustic_reflection.png", dpi=150)
        plt.close(fig)

    # 3 --- order of accuracy
    if "order" in gates:
        o = gates["order"]
        rows = o.get("rows") or o.get("measured_rows") or []
        n = [row["resolution"] for row in rows]
        e = [row["l2_error"] for row in rows]
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        ax.loglog(n, e, "o-", label="two-level composite")
        ref = [e[0] * (n[0] / x) ** 2 for x in n]
        observed = o.get("observed_order", float("nan"))
        ax.loglog(n, ref, "k--", label="slope 2 reference")
        ax.set_xlabel("coarse resolution N")
        ax.set_ylabel(r"$L_2$ velocity error / $U_0$")
        ax.set_title(f"Taylor-Green decay: observed order {observed:.2f} (baseline unstable)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        fig.savefig(outdir / "order_convergence.png", dpi=150)
        plt.close(fig)

    # 4 --- conservation history
    if "conservation" in gates:
        c = gates["conservation"]
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        names = ["mass", "momentum"]
        vals = [
            max(c["mass_residual_relative"], 1e-18),
            max(c["momentum_residual_relative"], 1e-18),
        ]
        tols = [c["mass_tolerance"], c["momentum_tolerance"]]
        ax.bar(names, vals, color=["#4477aa", "#ee6677"])
        ax.bar(names, tols, color="none", edgecolor="k", linestyle="--", label="gate tolerance")
        ax.set_yscale("log")
        ax.set_ylabel("worst relative residual")
        ax.set_title("Composite-grid conservation over %d steps" % c["steps"])
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(outdir / "conservation.png", dpi=150)
        plt.close(fig)
    print(f"figures written to {outdir}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())