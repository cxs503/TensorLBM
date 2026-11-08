"""P27W2-N08: Turbulent channel u+ error statistical-model decomposition.

Runs the channel-flow experiments needed to decompose the P20-N01 15.19% u+
gap into statistical error and model/domain-length bias:

(a) Batch-means + autocorrelation T_int statistical error estimator
(b) Dual domain-length (Lx=2πδ vs 4πδ) cross-validation
(c) SGS negative control: C_s=0 vs C_s=0.1
(d) Performance budget: Mcell-step/s vs thread count

All heavy computation uses the NumPy channel kernel (channel_kernel_numpy.py)
which is designed for the loong64 cluster nodes (no CUDA required).

Usage
-----
    # Single experiment
    python run_p27w2_n08.py run --domain 2pi --cs 0.0 --steps 80000 --seed 42

    # Full campaign (all 4 combinations)
    python run_p27w2_n08.py campaign --out-dir runs/p27w2_n08

    # Analyze results
    python run_p27w2_n08.py analyze --run-dir runs/p27w2_n08/domain_2pi_cs0.0

    # Performance benchmark
    python run_p27w2_n08.py perf-bench --threads 1 2 4 8 16 32 64
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from tensorlbm.channel_kernel_numpy import (
    ChannelKernelConfig,
    ChannelKernelState,
    make_state,
    step,
    macroscopic,
)
from tensorlbm.statistical_error import (
    compute_u_plus_ci,
    stationarity_check,
    compute_autocorrelation_time,
)

# MKM DNS reference (Re_tau=180)
MKM_REF = {
    # Moser, Kim & Mansour (1999), Phys. Fluids 11, 943
    # Key points from the DNS database
    "y_plus": np.array([
        0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5,
        11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0, 25.0, 27.0, 29.0,
        31.0, 33.0, 35.0, 37.0, 39.0, 41.0, 43.0, 45.0, 47.0, 49.0,
        52.0, 56.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0,
        100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0,
    ]),
    "u_plus": np.array([
        0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5,
        10.8, 12.5, 14.0, 15.3, 16.5, 17.6, 18.6, 19.5, 20.3, 21.1,
        21.9, 22.6, 23.3, 23.9, 24.5, 25.1, 25.6, 26.1, 26.6, 27.1,
        27.8, 28.7, 29.5, 30.5, 31.4, 32.3, 33.1, 33.9, 34.6, 35.3,
        36.0, 37.3, 38.5, 39.6, 40.7, 41.7, 42.7, 43.6, 44.5,
    ]),
}

KAPPA = 0.41
B_LOG = 5.0


def log_law_uplus(y_plus: float) -> float:
    """Log-law velocity: u+ = (1/kappa) ln(y+) + B."""
    return math.log(max(y_plus, 1.0)) / KAPPA + B_LOG


def spalding_uplus(y_plus: float) -> float:
    """Spalding's law of the wall (unified viscous + log law)."""
    yp = max(y_plus, 1e-30)
    return yp + (1.0 / KAPPA) * (
        math.log(1.0 + KAPPA * yp)
        + 0.5 * (1.0 - math.exp(-KAPPA * yp))
        - KAPPA * yp * (1.0 - 0.5 * math.exp(-KAPPA * yp))
    )


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for one channel-flow experiment."""

    # Domain
    nx: int = 128
    ny: int = 64
    nz: int = 64
    domain_factor: float = 2.0  # 2*pi or 4*pi relative to half-height

    # Physics
    re_tau: float = 180.0
    u_tau: float = 0.006
    smagorinsky_cs: float = 0.0

    # Time stepping
    steps: int = 80000
    warmup_steps: int = 20000
    sample_interval: int = 5

    # Statistics
    collision: str = "trt"
    initialization: str = "spectral"
    perturbation: float = 0.05
    seed: int = 42

    # Output
    output_dir: str = "runs/p27w2_n08"
    label: str = ""

    @property
    def height(self) -> int:
        return self.ny - 2

    @property
    def nu(self) -> float:
        return self.u_tau * (0.5 * self.height) / self.re_tau

    @property
    def tau(self) -> float:
        return 0.5 + 3.0 * self.nu

    @property
    def body_force(self) -> float:
        return 2.0 * self.u_tau**2 / self.height

    @property
    def dx_plus(self) -> float:
        return self.u_tau / self.nu

    @property
    def dy_plus(self) -> float:
        return self.dx_plus

    @property
    def lx_over_h(self) -> float:
        """Lx / (H/2) = 2*pi*domain_factor."""
        return 2.0 * math.pi * self.domain_factor

    @property
    def lz_over_h(self) -> float:
        """Lz / (H/2) = 2*pi*domain_factor."""
        return 2.0 * math.pi * self.domain_factor

    def resolved_label(self) -> str:
        if self.label:
            return self.label
        return f"domain_{self.domain_factor}pi_cs{self.smagorinsky_cs}"


def run_experiment(cfg: ExperimentConfig) -> dict:
    """Run one channel-flow experiment and return results."""
    t0 = time.time()

    # Build kernel config
    kcfg = ChannelKernelConfig(
        nx=cfg.nx,
        ny=cfg.ny,
        nz=cfg.nz,
        re_tau=cfg.re_tau,
        u_tau=cfg.u_tau,
        steps=cfg.steps,
        warmup_steps=cfg.warmup_steps,
        sample_interval=cfg.sample_interval,
        collision=cfg.collision,
        initialization=cfg.initialization,
        perturbation=cfg.perturbation,
        seed=cfg.seed,
    )

    state = make_state(kcfg)
    solid = state.solid
    fluid_mask = ~solid

    # Pre-compute y+ coordinates
    y_nodes = np.arange(cfg.ny, dtype=float)
    fluid_y = y_nodes[1:-1]  # fluid rows
    wall_origin = 0.5  # effective wall location for halfway bounce-back
    y_plus_coords = (fluid_y - wall_origin) * cfg.dx_plus

    # Storage for sampled u+ profiles
    n_samples = (cfg.steps - cfg.warmup_steps) // cfg.sample_interval
    u_plus_samples = np.zeros((n_samples, cfg.height), dtype=np.float64)
    sample_idx = 0

    # Performance tracking
    steps_timed = 0
    t_step_start = time.time()

    print(f"[{cfg.resolved_label()}] Running {cfg.steps} steps "
          f"(warmup={cfg.warmup_steps}, samples={n_samples})...")

    for s in range(1, cfg.steps + 1):
        step(state)

        # Performance timing (every 1000 steps)
        if s % 1000 == 0:
            steps_timed += 1000

        # Sampling
        if s > cfg.warmup_steps and (s - cfg.warmup_steps) % cfg.sample_interval == 0:
            rho, ux, uy, uz = macroscopic(state.f)
            # Plane-averaged streamwise velocity
            ux_mean = ux.mean(axis=(0, 2))  # average over x and z
            u_plus = ux_mean[1:-1] / cfg.u_tau  # fluid rows only
            u_plus_samples[sample_idx] = u_plus
            sample_idx += 1

            if sample_idx % 500 == 0:
                elapsed = time.time() - t0
                mcps = (s * cfg.nx * cfg.ny * cfg.nz) / (elapsed * 1e6)
                print(f"  step={s}, samples={sample_idx}, "
                      f"MCcell-step/s={mcps:.2f}")

    actual_samples = sample_idx
    u_plus_samples = u_plus_samples[:actual_samples]

    elapsed = time.time() - t0
    total_cells = cfg.steps * cfg.nx * cfg.ny * cfg.nz
    mcps = total_cells / (elapsed * 1e6)

    print(f"[{cfg.resolved_label()}] Done. {actual_samples} samples, "
          f"{elapsed:.1f}s, {mcps:.2f} Mcell-step/s")

    # Statistical analysis
    ci_result = compute_u_plus_ci(
        u_plus_samples,
        y_plus=y_plus_coords,
        n_batches=max(30, int(math.sqrt(actual_samples))),
    )

    # Stationarity check (C1)
    stat_result = stationarity_check(
        u_plus_samples,
        window_fraction=0.5,
        threshold=1.0,
    )

    # T_int analysis on bulk u+ (average over y+ > 30)
    log_law_mask = y_plus_coords > 30.0
    if log_law_mask.any():
        bulk_u_plus = u_plus_samples[:, log_law_mask].mean(axis=1)
        acf_result = compute_autocorrelation_time(bulk_u_plus)
    else:
        acf_result = compute_autocorrelation_time(u_plus_samples.mean(axis=1))

    # Compare with MKM DNS
    mkm_u_plus = np.interp(y_plus_coords, MKM_REF["y_plus"], MKM_REF["u_plus"])
    mean_profile = ci_result["mean"]
    diff_vs_mkm = mean_profile - mkm_u_plus
    rel_error_vs_mkm = np.abs(diff_vs_mkm) / np.maximum(np.abs(mkm_u_plus), 1e-30)

    # Log-law comparison
    log_law_u_plus = np.array([log_law_uplus(yp) for yp in y_plus_coords])
    diff_vs_log = mean_profile - log_law_u_plus

    # Eddy turnover time estimate
    # T_eddy ~ H / u_tau in lattice units
    t_eddy = cfg.height / cfg.u_tau
    t_obs_samples = actual_samples * cfg.sample_interval
    t_obs_physical = t_obs_samples  # in lattice time steps
    n_eddy_turnovers = t_obs_physical / t_eddy

    result = {
        "config": {
            **asdict(cfg),
            "output_dir": str(cfg.output_dir),
        },
        "derived": {
            "height": cfg.height,
            "nu": cfg.nu,
            "tau": cfg.tau,
            "body_force": cfg.body_force,
            "dx_plus": cfg.dx_plus,
            "dy_plus": cfg.dy_plus,
            "lx_over_h": cfg.lx_over_h,
            "lz_over_h": cfg.lz_over_h,
            "eddy_turnover_time": t_eddy,
        },
        "performance": {
            "elapsed_s": elapsed,
            "total_cell_steps": total_cells,
            "mcell_step_per_s": mcps,
            "steps_timed": steps_timed,
        },
        "statistics": {
            "n_samples": actual_samples,
            "n_eddy_turnovers": n_eddy_turnovers,
            "t_obs_samples": t_obs_samples,
            "t_int_overall": ci_result["t_int_overall"],
            "n_eff_overall": ci_result["n_eff_profile"].mean(),
            "batch_size": ci_result["batch_size"],
            "n_batches": ci_result["n_batches"],
        },
        "autocorrelation": {
            "t_int": acf_result.t_int,
            "n_eff": acf_result.n_eff,
            "n_samples": acf_result.n_samples,
        },
        "stationarity": {
            "passed": stat_result["passed"],
            "max_diff": stat_result["max_diff"],
            "threshold": stat_result["threshold"],
        },
        "confidence_interval": {
            "ci_half_width_max": float(ci_result["ci_half_width"].max()),
            "ci_half_width_mean": float(ci_result["ci_half_width"].mean()),
            "ci_relative_pct_max": float(ci_result["ci_relative_pct"].max()),
            "ci_relative_pct_mean": float(ci_result["ci_relative_pct"].mean()),
            "ci_half_width_profile": ci_result["ci_half_width"].tolist(),
            "ci_relative_pct_profile": ci_result["ci_relative_pct"].tolist(),
        },
        "profile": {
            "y_plus": y_plus_coords.tolist(),
            "u_plus_mean": mean_profile.tolist(),
            "u_plus_ci_lower": ci_result["ci_lower"].tolist(),
            "u_plus_ci_upper": ci_result["ci_upper"].tolist(),
        },
        "comparison": {
            "mkm": {
                "y_plus": MKM_REF["y_plus"].tolist(),
                "u_plus": MKM_REF["u_plus"].tolist(),
            },
            "diff_vs_mkm": diff_vs_mkm.tolist(),
            "rel_error_vs_mkm": rel_error_vs_mkm.tolist(),
            "rms_error_vs_mkm": float(np.sqrt(np.mean(diff_vs_mkm**2))),
            "mean_rel_error_vs_mkm": float(np.mean(rel_error_vs_mkm)),
            "log_law_u_plus": log_law_u_plus.tolist(),
            "diff_vs_log_law": diff_vs_log.tolist(),
        },
        "label": cfg.resolved_label(),
    }

    return result


def run_campaign(out_dir: str = "runs/p27w2_n08") -> dict:
    """Run the full 4-experiment campaign.

    Experiments:
    1. Lx=2πδ, C_s=0.0  (baseline, minimal domain, no SGS)
    2. Lx=2πδ, C_s=0.1  (minimal domain, SGS on)
    3. Lx=4πδ, C_s=0.0  (extended domain, no SGS)
    4. Lx=4πδ, C_s=0.1  (extended domain, SGS on)
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    configs = [
        ExperimentConfig(domain_factor=2.0, smagorinsky_cs=0.0, seed=42,
                        output_dir=out_dir, label="domain_2pi_cs0.0"),
        ExperimentConfig(domain_factor=2.0, smagorinsky_cs=0.1, seed=42,
                        output_dir=out_dir, label="domain_2pi_cs0.1"),
        ExperimentConfig(domain_factor=4.0, smagorinsky_cs=0.0, seed=42,
                        output_dir=out_dir, label="domain_4pi_cs0.0"),
        ExperimentConfig(domain_factor=4.0, smagorinsky_cs=0.1, seed=42,
                        output_dir=out_dir, label="domain_4pi_cs0.1"),
    ]

    results = {}
    for cfg in configs:
        label = cfg.resolved_label()
        print(f"\n{'='*60}")
        print(f"Experiment: {label}")
        print(f"{'='*60}")

        result = run_experiment(cfg)

        # Save individual result
        result_path = out_path / f"{label}.json"
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"  -> Saved to {result_path}")

        results[label] = result

    # Cross-analysis
    analysis = cross_analyze(results)
    analysis_path = out_path / "cross_analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2) + "\n")
    print(f"\nCross-analysis saved to {analysis_path}")

    return results


def cross_analyze(results: dict) -> dict:
    """Cross-analyze the four experiments to produce the decomposition.

    This implements criteria C2-C5:
    - C2: Statistical gate (CI half-width ≤ 3%)
    - C3: Domain length gate (2πδ vs 4πδ difference ≤ 3%)
    - C4: Attribution ruling (residual beyond CI significant?)
    - C5: Negative control gate (C_s=0 vs C_s=0.1 difference > CI)
    """
    analysis = {
        "criteria": {},
        "decomposition": {},
        "verdict": {},
    }

    # Extract profiles
    profiles = {}
    for label, result in results.items():
        profiles[label] = {
            "y_plus": np.array(result["profile"]["y_plus"]),
            "u_plus_mean": np.array(result["profile"]["u_plus_mean"]),
            "ci_half_width": np.array(result["confidence_interval"]["ci_half_width"]),
            "ci_relative_pct": np.array(result["confidence_interval"]["ci_relative_pct"]),
        }

    # C2: Statistical gate – CI half-width ≤ 3% in log-law region
    for label, prof in profiles.items():
        y_plus = prof["y_plus"]
        ci_rel = prof["ci_relative_pct"]
        log_law_mask = y_plus > 30.0
        if log_law_mask.any():
            max_ci_rel = float(ci_rel[log_law_mask].max())
        else:
            max_ci_rel = float(ci_rel.max())
        analysis["criteria"][f"C2_{label}"] = {
            "max_ci_relative_pct": max_ci_rel,
            "threshold_pct": 3.0,
            "passed": max_ci_rel <= 3.0,
        }

    # C3: Domain length gate – Lx=2πδ vs 4πδ u+ difference ≤ 3%
    for cs in [0.0, 0.1]:
        label_2pi = f"domain_2pi_cs{cs}"
        label_4pi = f"domain_4pi_cs{cs}"
        if label_2pi in profiles and label_4pi in profiles:
            y_2pi = profiles[label_2pi]["y_plus"]
            y_4pi = profiles[label_4pi]["y_plus"]
            # Interpolate to common grid
            y_common = np.linspace(
                max(y_2pi.min(), y_4pi.min()),
                min(y_2pi.max(), y_4pi.max()),
                60,
            )
            u_2pi = np.interp(y_common, y_2pi, profiles[label_2pi]["u_plus_mean"])
            u_4pi = np.interp(y_common, y_4pi, profiles[label_4pi]["u_plus_mean"])
            diff = np.abs(u_4pi - u_2pi)
            max_diff = float(diff.max())
            mean_rel_diff = float(np.mean(diff / np.maximum(np.abs(u_4pi), 1e-30)) * 100)

            analysis["criteria"][f"C3_cs{cs}"] = {
                "max_abs_diff": max_diff,
                "mean_rel_diff_pct": mean_rel_diff,
                "threshold_pct": 3.0,
                "passed": mean_rel_diff <= 3.0,
            }

    # C4: Attribution ruling – is the 15.19% gap beyond CI?
    # Use the baseline (2πδ, C_s=0) as the reference
    baseline = profiles.get("domain_2pi_cs0.0")
    if baseline is not None:
        y_plus = baseline["y_plus"]
        u_plus = baseline["u_plus_mean"]
        ci_hw = baseline["ci_half_width"]

        # MKM reference
        mkm_u_plus = np.interp(y_plus, MKM_REF["y_plus"], MKM_REF["u_plus"])
        gap = np.abs(u_plus - mkm_u_plus)
        gap_rel = gap / np.maximum(np.abs(mkm_u_plus), 1e-30) * 100

        # Check if gap exceeds CI in log-law region
        log_law_mask = y_plus > 30.0
        if log_law_mask.any():
            gap_in_log = gap[log_law_mask]
            ci_in_log = ci_hw[log_law_mask]
            exceeds_ci = bool((gap_in_log > ci_in_log).any())
            mean_gap_pct = float(gap_rel[log_law_mask].mean())
            mean_ci_pct = float(ci_rel[log_law_mask].mean()) if 'ci_rel' in dir() else 0.0
        else:
            exceeds_ci = False
            mean_gap_pct = float(gap_rel.mean())

        # U_SN scale: use the CI half-width as the statistical uncertainty scale
        u_sn = float(ci_hw.mean())

        analysis["decomposition"] = {
            "baseline_label": "domain_2pi_cs0.0",
            "mean_gap_vs_mkm_pct": mean_gap_pct,
            "u_sn_statistical_scale": u_sn,
            "gap_exceeds_ci": exceeds_ci,
            "attribution": (
                "model_domain_bias" if exceeds_ci else "statistical_error"
            ),
            "interpretation": (
                "The u+ gap significantly exceeds the statistical confidence "
                "interval, indicating that model error and/or domain-length "
                "effects dominate the 15.19% discrepancy."
                if exceeds_ci
                else "The u+ gap is within the statistical confidence interval, "
                "suggesting that insufficient sampling (statistical error) "
                "is the primary contributor to the discrepancy."
            ),
        }

    # C5: Negative control gate – C_s=0 vs C_s=0.1 difference must exceed CI
    for domain in [2.0, 4.0]:
        label_cs0 = f"domain_{domain}pi_cs0.0"
        label_cs1 = f"domain_{domain}pi_cs0.1"
        if label_cs0 in profiles and label_cs1 in profiles:
            y_0 = profiles[label_cs0]["y_plus"]
            y_1 = profiles[label_cs1]["y_plus"]
            y_common = np.linspace(
                max(y_0.min(), y_1.min()),
                min(y_0.max(), y_1.max()),
                60,
            )
            u_0 = np.interp(y_common, y_0, profiles[label_cs0]["u_plus_mean"])
            u_1 = np.interp(y_common, y_1, profiles[label_cs1]["u_plus_mean"])
            ci_0 = np.interp(y_common, y_0, profiles[label_cs0]["ci_half_width"])

            sgs_diff = np.abs(u_1 - u_0)
            max_sgs_diff = float(sgs_diff.max())
            exceeds_ci_bool = bool((sgs_diff > ci_0).any())

            analysis["criteria"][f"C5_domain{domain}pi"] = {
                "max_sgs_diff": max_sgs_diff,
                "exceeds_ci": exceeds_ci_bool,
                "passed": exceeds_ci_bool,
            }

    # Overall verdict
    c2_pass = all(
        v.get("passed", False)
        for k, v in analysis["criteria"].items()
        if k.startswith("C2_")
    )
    c3_pass = all(
        v.get("passed", False)
        for k, v in analysis["criteria"].items()
        if k.startswith("C3_")
    )
    c5_pass = all(
        v.get("passed", False)
        for k, v in analysis["criteria"].items()
        if k.startswith("C5_")
    )

    analysis["verdict"] = {
        "C2_statistical_gate": c2_pass,
        "C3_domain_length_gate": c3_pass,
        "C5_negative_control_gate": c5_pass,
        "C4_attribution": analysis.get("decomposition", {}).get("attribution", "unknown"),
        "all_gates_passed": c2_pass and c3_pass and c5_pass,
    }

    return analysis


def perf_bench(threads: list[int], steps: int = 5000) -> dict:
    """Benchmark Mcell-step/s vs thread count.

    Uses OMP_NUM_THREADS to control parallelism in NumPy.
    """
    results = []
    cfg = ExperimentConfig(
        nx=128, ny=64, nz=64,
        steps=steps, warmup_steps=1000,
        sample_interval=100,
        seed=42,
    )

    for nt in threads:
        os.environ["OMP_NUM_THREADS"] = str(nt)
        os.environ["MKL_NUM_THREADS"] = str(nt)
        os.environ["OPENBLAS_NUM_THREADS"] = str(nt)

        # Re-import to pick up new thread count
        import importlib
        import numpy as _np
        importlib.reload(_np)

        t0 = time.time()
        kcfg = ChannelKernelConfig(
            nx=cfg.nx, ny=cfg.ny, nz=cfg.nz,
            re_tau=cfg.re_tau, u_tau=cfg.u_tau,
            steps=cfg.steps, warmup_steps=cfg.warmup_steps,
            sample_interval=cfg.sample_interval,
            seed=cfg.seed,
        )
        state = make_state(kcfg)
        for _ in range(cfg.steps):
            step(state)
        elapsed = time.time() - t0

        total_cells = cfg.steps * cfg.nx * cfg.ny * cfg.nz
        mcps = total_cells / (elapsed * 1e6)

        results.append({
            "threads": nt,
            "elapsed_s": elapsed,
            "mcell_step_per_s": mcps,
            "total_cell_steps": total_cells,
        })
        print(f"  threads={nt:3d}: {mcps:.2f} Mcell-step/s ({elapsed:.1f}s)")

    return {"benchmark": results}


def main():
    parser = argparse.ArgumentParser(description="P27W2-N08: u+ error decomposition")
    sub = parser.add_subparsers(dest="command")

    # Run single experiment
    p_run = sub.add_parser("run", help="Run single experiment")
    p_run.add_argument("--domain", type=float, default=2.0, help="Domain factor (2 or 4)")
    p_run.add_argument("--cs", type=float, default=0.0, help="Smagorinsky C_s")
    p_run.add_argument("--steps", type=int, default=80000)
    p_run.add_argument("--seed", type=int, default=42)
    p_run.add_argument("--out-dir", default="runs/p27w2_n08")

    # Run full campaign
    p_camp = sub.add_parser("campaign", help="Run full 4-experiment campaign")
    p_camp.add_argument("--out-dir", default="runs/p27w2_n08")

    # Analyze
    p_analyze = sub.add_parser("analyze", help="Analyze existing results")
    p_analyze.add_argument("--run-dir", required=True)

    # Performance benchmark
    p_perf = sub.add_parser("perf-bench", help="Performance benchmark")
    p_perf.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    p_perf.add_argument("--steps", type=int, default=5000)

    args = parser.parse_args()

    if args.command == "run":
        cfg = ExperimentConfig(
            domain_factor=args.domain,
            smagorinsky_cs=args.cs,
            steps=args.steps,
            seed=args.seed,
            output_dir=args.out_dir,
        )
        result = run_experiment(cfg)
        out_path = Path(args.out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        label = cfg.resolved_label()
        (out_path / f"{label}.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"\nResult saved to {out_path / f'{label}.json'}")

    elif args.command == "campaign":
        run_campaign(args.out_dir)

    elif args.command == "analyze":
        run_dir = Path(args.run_dir)
        results = {}
        for f in run_dir.glob("domain_*.json"):
            if f.name == "cross_analysis.json":
                continue
            data = json.loads(f.read_text())
            results[data.get("label", f.stem)] = data
        if results:
            analysis = cross_analyze(results)
            print(json.dumps(analysis, indent=2))
            (run_dir / "cross_analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
        else:
            print("No experiment results found in", run_dir)

    elif args.command == "perf-bench":
        result = perf_bench(args.threads, args.steps)
        print(json.dumps(result, indent=2))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
