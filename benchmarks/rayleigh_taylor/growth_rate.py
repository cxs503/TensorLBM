"""Rayleigh--Taylor linear growth-rate benchmark (single-mode, Boussinesq D2Q9).

Runs the Boussinesq two-layer Rayleigh--Taylor instability for three lattice
wavelengths ``lam in {32, 64, 128}`` (density ratio ``rho1/rho2 = 2``,
``A = 1/3``, initial amplitude ``eta0 = lam/100``) and compares the measured
exponential growth rate with the inviscid analytic value

    sigma_analytic = sqrt(A g k),          k = 2 pi / lam

and with the viscous linear result ``sqrt(Agk + nu^2 k^4) - nu k^2``.
A grid-refinement study on the normalised growth rate ``sigma/sigma_analytic``
gives the observed order of accuracy and the Grid Convergence Index (GCI,
Roache 1994).

Usage::

    PYTHONPATH=src python benchmarks/rayleigh_taylor/growth_rate.py
    PYTHONPATH=src python benchmarks/rayleigh_taylor/growth_rate.py --fast
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parents[1]
SRC_ROOT = REPO_ROOT / "src"
for _p in (str(SRC_ROOT), str(REPO_ROOT)):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from tensorlbm.rayleigh_taylor import (  # noqa: E402
    RTConfig,
    analytic_growth_rate,
    run_rayleigh_taylor,
    viscous_growth_rate,
)

LAMBDAS = (32, 64, 128)
GATE_RATIO_PCT = 3.0
GATE_ORDER = 1.5


def _observed_order(e_coarse: float, e_fine: float, r: float = 2.0) -> float:
    """Observed order of accuracy from two error levels."""
    if e_coarse <= 0.0 or e_fine <= 0.0 or e_fine >= e_coarse:
        return float("nan")
    return math.log(e_coarse / e_fine) / math.log(r)


def _gci(e_fine: float, p: float, r: float = 2.0) -> float:
    """Roache's Grid Convergence Index (safety factor 1.25)."""
    if not math.isfinite(p) or p <= 0.0:
        return float("nan")
    return 1.25 * e_fine / (r**p - 1.0)


def run_growth_rate_suite(
    lambdas: tuple[int, ...] = LAMBDAS,
    fast: bool = False,
    output_root: Path | None = None,
    result_path: Path | None = None,
) -> dict:
    """Run the RT growth-rate suite; return the machine-readable report."""
    out_root = Path(output_root) if output_root is not None else BENCH_DIR / "runs"
    out_root.mkdir(parents=True, exist_ok=True)

    cases: list[dict] = []
    for lam in lambdas:
        cfg = RTConfig(
            lam=lam,
            n_steps=400 if fast else 1500,
            sample_every=2 if not fast else 5,
            layers=3.0 if not fast else 2.0,
            stop_a_frac=0.14,
        )
        case = run_rayleigh_taylor(cfg, output_dir=out_root / f"lam{lam}")
        cases.append(case)
        print(
            f"[RT] lam={lam:4d}  sigma_sim={case['sigma_sim']:.6f}  "
            f"analytic={case['sigma_analytic']:.6f}  "
            f"ratio={case['ratio_sim_over_analytic']:.4f}  "
            f"err={case['rel_error_pct']:+.2f}%  "
            f"visc_theory={case['sigma_viscous_theory']:.6f}  "
            f"steps={case['steps_run']}",
            flush=True,
        )

    # ---- grid-refinement study on the normalised growth rate
    ratios = [c["ratio_sim_over_analytic"] for c in cases]
    errs = [abs(r - 1.0) for r in ratios]
    orders: list[float] = []
    gcis: list[float] = []
    for idx in range(len(errs) - 1):
        p = _observed_order(errs[idx], errs[idx + 1])
        orders.append(p)
        gcis.append(_gci(errs[idx + 1], p))
    p_mean = (
        sum(o for o in orders if math.isfinite(o)) / max(1, sum(1 for o in orders if math.isfinite(o)))
        if any(math.isfinite(o) for o in orders)
        else float("nan")
    )
    # Richardson extrapolation of the normalised ratio towards lam -> infinity
    ratio_inf = float("nan")
    if len(ratios) >= 3 and math.isfinite(orders[-1]) and orders[-1] > 0.0:
        r_fac = 2.0 ** orders[-1]
        ratio_inf = ratios[-1] + (ratios[-1] - ratios[-2]) / (r_fac - 1.0)

    per_case_pass = {
        str(c["lam"]): bool(abs(c["rel_error_pct"]) <= GATE_RATIO_PCT) for c in cases
    }
    report = {
        "benchmark": "rayleigh_taylor_growth_rate",
        "title": "Rayleigh-Taylor linear growth rate vs sigma = sqrt(A g k)",
        "physics": {
            "atwood": cases[0]["atwood"],
            "density_ratio": 2.0,
            "gravity": cases[0]["g"],
            "initial_amplitude_over_lambda": cases[0]["eta0"] / cases[0]["lam"],
            "model": "Boussinesq two-layer D2Q9, transported phase field, no surface tension",
        },
        "gates": {
            "growth_rate_rel_error_le_pct": GATE_RATIO_PCT,
            "observed_order_ge": GATE_ORDER,
        },
        "cases": [
            {
                key: c[key]
                for key in (
                    "lam", "k", "sigma_analytic", "sigma_viscous_theory", "sigma_sim",
                    "ratio_sim_over_analytic", "rel_error_pct", "ratio_sim_over_viscous",
                    "rel_error_vs_viscous_pct", "fit_r2", "fit_points", "fit_a_over_lam",
                    "steps_run", "nx", "ny", "nu", "tau", "tau_phi", "diffusivity",
                    "interface_width", "phase_mass_drift_pct", "sigma_sim_colmoment",
                )
            }
            for c in cases
        ],
        "grid_convergence": {
            "refinement_ratio": 2.0,
            "normalised_ratio": dict(zip([str(c["lam"]) for c in cases], ratios)),
            "abs_error": dict(zip([str(c["lam"]) for c in cases], errs)),
            "observed_orders": dict(
                zip([f"{lambdas[i]}->{lambdas[i + 1]}" for i in range(len(orders))], orders)
            ),
            "observed_order_mean": p_mean,
            "gci": dict(zip([f"gci_{lambdas[i + 1]}" for i in range(len(gcis))], gcis)),
            "richardson_ratio_inf": ratio_inf,
        },
        "verdict": {
            "per_lambda_error_gate": per_case_pass,
            "all_lambdas_within_gate": all(per_case_pass.values()),
            "observed_order_ge_gate": bool(math.isfinite(p_mean) and p_mean >= GATE_ORDER),
            "status": "PASS"
            if all(per_case_pass.values())
            and math.isfinite(p_mean)
            and p_mean >= GATE_ORDER
            else "FAIL",
        },
    }
    (Path(result_path) if result_path is not None else BENCH_DIR / "result.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="short smoke run")
    parser.add_argument("--lambdas", type=int, nargs="*", default=list(LAMBDAS))
    args = parser.parse_args(argv)
    report = run_growth_rate_suite(tuple(args.lambdas), fast=args.fast)
    print(json.dumps(report["verdict"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())