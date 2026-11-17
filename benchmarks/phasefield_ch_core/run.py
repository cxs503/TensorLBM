"""CPU/float64 CH-core acceptance; writes evidence before exiting nonzero.

PYTHONPATH=src python benchmarks/phasefield_ch_core/run.py --output artifacts/ch_core/report.json --plot
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
from pathlib import Path

import torch

from tensorlbm.phasefield.ch_benchmarks import run_linear_mode, run_planar_interface

ROOT = Path(__file__).resolve().parents[2]


def provenance():
    def git(*args):
        result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

    paths = [
        "src/tensorlbm/phasefield/ch_equilibrium.py",
        "src/tensorlbm/phasefield/ch_benchmarks.py",
        "src/tensorlbm/multiphase.py",
        "src/tensorlbm/multiphase3d.py",
        "src/tensorlbm/d2q9.py",
        "src/tensorlbm/d3q19.py",
        "src/tensorlbm/solver.py",
        "src/tensorlbm/solver3d.py",
        "src/tensorlbm/phasefield/operators.py",
        "src/tensorlbm/phasefield/free_energy.py",
        "benchmarks/phasefield_ch_core/run.py",
    ]
    return {
        "base_commit": git("rev-parse", "HEAD"),
        "working_tree_dirty": bool(git("status", "--porcelain")),
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths},
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "device": "cpu",
        "dtype": "float64",
        "threads": torch.get_num_threads(),
    }


def run_suite(convention="lee_lin"):
    cases, convergence = [], []
    for dim in (2, 3):
        for bg in (0.0, 1.0):
            group = [
                run_linear_mode(
                    dimension=dim,
                    n=n,
                    background=bg,
                    steps=100 * (n // 16) ** 2,
                    geq_convention=convention,
                )
                for n in (16, 32, 64)
            ]
            cases.extend(group)
            errors = [c["relative_rate_error"] for c in group]
            orders = [
                math.log2(a / b) if a and b and a > 0 and b > 0 else None
                for a, b in zip(errors, errors[1:])
            ]
            convergence.append(
                {
                    "dimension": dim,
                    "background": bg,
                    "relative_errors": errors,
                    "observed_orders": orders,
                    "passed": all(p is not None and 1.8 < p < 2.2 for p in orders),
                }
            )
        cases.append(run_linear_mode(dimension=dim, Gamma=0.0, geq_convention=convention))
        for tau, gamma in ((0.8, 0.2), (1.2, 0.4)):
            cases.append(
                run_linear_mode(dimension=dim, tau_g=tau, Gamma=gamma, geq_convention=convention)
            )
    for xi in (3.0, 4.5, 6.0):
        cases.append(
            run_planar_interface(
                xi=xi, steps=round(1000 * (xi / 3) ** 4), geq_convention=convention
            )
        )
    planar_errors = [
        c["sigma_relative_error"] for c in cases if c["case"] == "evolved_planar_interface"
    ]
    planar_refinement = all(b < a for a, b in zip(planar_errors, planar_errors[1:]))
    # Controls are reported as failed physics, never counted as accepted cases.
    controls = [run_linear_mode(dimension=d, Gamma=0.0, geq_convention="swift") for d in (2, 3)]
    controls.append(run_planar_interface(geq_convention="swift"))
    controls_rejected = all(not c["passed"] for c in controls)
    # Preserve the discovered unstable stronger-capillarity regime. A passed
    # low-capillarity envelope must not conceal failures at ten times kappa.
    stress_cases = [
        run_planar_interface(
            xi=xi, steps=round(1000 * (xi / 3) ** 4), A_base=0.04, geq_convention=convention
        )
        for xi in (4.5, 6.0)
    ]
    passed = (
        all(c["passed"] for c in cases)
        and all(c["passed"] for c in convergence)
        and planar_refinement
        and controls_rejected
    )
    return {
        "schema_version": 1,
        "status": "passed_scoped_ch_core" if passed else "failed_ch_core",
        "passed": passed,
        "physical_validation_passed": False,
        "evidence_origin": "executed_lbm",
        "method": "free_energy_ch",
        "geq_convention": convention,
        "scope": [
            "linear_CH_growth_and_decay_with_prescribed_zero_physical_velocity",
            "coupled_D2Q9_planar_interface_gradient_stress_integral",
        ],
        "not_validated": [
            "curved_interface_Laplace_pressure",
            "capillary_wave_frequency_and_damping",
            "impact_and_splash",
            "high_density_ratio",
            "GPU",
            "LES",
        ],
        "admitted_planar_envelope": {
            "A_base": 0.004,
            "kappa": 0.018,
            "xi": [3.0, 4.5, 6.0],
            "description": "Only these sampled configurations; not a stability guarantee between or beyond them.",
        },
        "thresholds": {
            "mode_relative_rate_error": 0.05,
            "fit_log_residual": 1e-4,
            "zero_mobility_relative_amplitude_change": 1e-8,
            "phase_mass_drift": 1e-10,
            "mode_observed_order": [1.8, 2.2],
            "planar_sigma_relative_error": 0.1,
            "planar_profile_linf_change": 0.1,
            "planar_max_physical_speed": 1e-3,
        },
        "cases": cases,
        "mode_refinement": convergence,
        "planar_error_decreases": planar_refinement,
        "negative_controls": controls,
        "negative_controls_rejected": controls_rejected,
        "outside_admitted_envelope": stress_cases,
        "provenance": provenance(),
    }


def plot_report(report, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for group in report["mode_refinement"]:
        axes[0].loglog(
            [16, 32, 64],
            group["relative_errors"],
            "o-",
            label=f"D{group['dimension']}, phi={group['background']:g}",
        )
    axes[0].set(
        title="CH rate error vs continuum", xlabel="Cells per wavelength", ylabel="Relative error"
    )
    axes[0].legend(fontsize=7)
    for case in report["cases"] + report["negative_controls"]:
        cfg = case["config"]
        if case["case"] == "evolved_planar_interface":
            axes[1].plot(
                [r["step"] / cfg["steps"] for r in case["history"]],
                [r["sigma_gradient_integral"] / case["sigma_analytic"] for r in case["history"]],
                label=f"{cfg['geq_convention']}, xi={cfg['xi']:g}",
            )
        elif cfg["Gamma"] == 0 and cfg["dimension"] == 2:
            axes[2].plot(
                [r["step"] for r in case["history"]],
                [r["amplitude"] / cfg["epsilon"] for r in case["history"]],
                label=cfg["geq_convention"],
            )
    axes[1].axhline(1, color="black", linestyle="--", linewidth=0.8)
    axes[1].set(
        title="Freely evolved planar interface",
        xlabel="Fraction of run",
        ylabel="Sigma integral / analytic",
    )
    axes[1].legend(fontsize=7)
    axes[2].set(
        title="Zero mobility: extra diffusion control", xlabel="Step", ylabel="Amplitude / initial"
    )
    axes[2].legend(fontsize=7)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/ch_core/report.json"))
    parser.add_argument("--geq-convention", choices=("lee_lin", "swift"), default="lee_lin")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    report = run_suite(args.geq_convention)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Preserve failed evidence even if a finite state overflowed a diagnostic.
    # Null is missing/invalid evidence, never a zero error or a passing gate.
    def strict(value):
        if isinstance(value, dict):
            return {key: strict(item) for key, item in value.items()}
        if isinstance(value, list):
            return [strict(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    report = strict(report)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.plot:
        plot_report(report, args.output.with_suffix(".png"))
    print(
        json.dumps(
            {
                "status": report["status"],
                "cases": len(report["cases"]),
                "accepted": sum(c["passed"] for c in report["cases"]),
                "report": str(args.output),
            }
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
