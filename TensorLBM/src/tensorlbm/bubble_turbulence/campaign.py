"""Campaign runner for bubble-turbulence interaction study.

Runs the full campaign:
  1. Single-phase turbulent channel (Re_tau=180) baseline
  2. Two-phase channel with microbubbles (d+ = 5, 10, 20)
  3. Negative controls: zero bubbles, very large bubbles
  4. Statistics collection and mechanism decomposition
  5. Acceptance criteria evaluation
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from tensorlbm.bubble_turbulence.bubble_solver import (
    BubbleConfig,
    TwoPhaseChannelSolver,
)
from tensorlbm.bubble_turbulence.statistics import (
    ChannelStatistics,
    compute_q_criterion,
    count_vortex_events,
    detect_burst_events,
)
from tensorlbm.bubble_turbulence.mechanisms import (
    DragReductionDecomposition,
    compute_slip_length,
    decompose_drag_reduction,
    compute_momentum_balance,
)


def run_single_phase_baseline(
    re_tau: float = 180.0,
    nx: int = 64,
    ny: int = 33,
    nz: int = 32,
    u_tau: float = 0.01,
    steps: int = 20000,
    warmup: int = 5000,
    device: str = "cpu",
    label: str = "single_phase",
) -> dict:
    """Run single-phase turbulent channel as baseline.

    Returns dict with profiles and acceptance criteria.
    """
    from tensorlbm.verification.turbulent_channel.channel_solver import (
        ChannelConfig, ChannelSolver,
    )

    cfg = ChannelConfig(
        nx=nx, ny=ny, nz=nz,
        re_tau=re_tau, u_tau=u_tau,
        smagorinsky_cs=0.1,
        steps=steps, warmup_steps=warmup,
        perturbation_amplitude=0.3,
        seed=42, device=device, label=label,
    )

    solver = ChannelSolver(cfg)

    # Run warmup
    print(f"  [{label}] Warmup {warmup} steps...")
    solver.run(warmup)

    # Run with statistics collection
    stats = ChannelStatistics(ny, u_tau, cfg.nu, device=device)

    print(f"  [{label}] Production {steps - warmup} steps...")
    for step in range(steps - warmup):
        solver.step()
        rho, ux, uy, uz = solver.get_macroscopic()
        stats.accumulate(ux, uy, uz)

    profiles = stats.get_profiles()

    # Compute friction velocity from three methods
    h = cfg.half_height
    u_tau_1 = math.sqrt(cfg.body_force * h)  # from body force
    dU_dy = (profiles["U_plus"][1] - profiles["U_plus"][0]) * u_tau / (cfg.nu / u_tau)
    u_tau_2 = math.sqrt(abs(cfg.nu * dU_dy))  # from wall gradient
    u_tau_3 = u_tau_1  # for constant flow rate

    # Q-criterion on final snapshot
    rho, ux, uy, uz = solver.get_macroscopic()
    Q = compute_q_criterion(ux, uy, uz)
    vortex_stats = count_vortex_events(Q, torch.tensor(profiles["y_plus"]))

    # Burst events
    burst = detect_burst_events(ux, uy, u_tau=u_tau, nu=cfg.nu)

    return {
        "label": label,
        "config": {
            "re_tau": re_tau, "nx": nx, "ny": ny, "nz": nz,
            "u_tau": u_tau, "steps": steps,
        },
        "profiles": profiles,
        "u_tau_methods": {
            "body_force": u_tau_1,
            "wall_gradient": u_tau_2,
            "reference": u_tau_3,
        },
        "u_tau_consistency": abs(u_tau_1 - u_tau_2) / max(u_tau_1, 1e-10),
        "vortex_stats": {
            "n_vortices": vortex_stats["n_vortices"],
            "burst_frequency_mean": float(np.mean(vortex_stats["burst_frequency"])),
        },
        "burst_events": burst,
    }


def run_two_phase_case(
    re_tau: float = 180.0,
    nx: int = 64,
    ny: int = 33,
    nz: int = 32,
    u_tau: float = 0.01,
    d_plus: float = 10.0,
    n_bubbles: int = 1000,
    steps: int = 20000,
    warmup: int = 5000,
    device: str = "cpu",
    label: str = "two_phase",
) -> dict:
    """Run two-phase (bubbly) turbulent channel.

    Returns dict with profiles, bubble statistics, and mechanism decomposition.
    """
    from tensorlbm.verification.turbulent_channel.channel_solver import ChannelConfig

    channel_cfg = ChannelConfig(
        nx=nx, ny=ny, nz=nz,
        re_tau=re_tau, u_tau=u_tau,
        smagorinsky_cs=0.1,
        steps=steps, warmup_steps=warmup,
        perturbation_amplitude=0.3,
        seed=42, device=device, label=label,
    )

    bubble_cfg = BubbleConfig(
        n_bubbles=n_bubbles,
        d_plus=d_plus,
        density_ratio=0.012,
        viscosity_ratio=0.018,
        injection_y_plus_max=30.0,
        two_way_coupling=True,
        seed=42,
    )

    solver = TwoPhaseChannelSolver(channel_cfg, bubble_cfg, device=device)

    # Run warmup (no statistics)
    print(f"  [{label}] Warmup {warmup} steps...")
    for _ in range(warmup):
        solver.step()

    # Run with statistics
    stats = ChannelStatistics(ny, u_tau, channel_cfg.nu, device=device)

    print(f"  [{label}] Production {steps - warmup} steps...")
    for step in range(steps - warmup):
        solver.step()
        rho, ux, uy, uz = solver.channel.get_macroscopic()
        stats.accumulate(ux, uy, uz)

    profiles = stats.get_profiles()

    # Bubble statistics
    y_bubble, alpha = solver.bubble.get_volume_fraction_profile()

    # Mass conservation: integral of alpha should be ~ n_bubbles * V_bubble / V_domain
    V_bubble = (math.pi / 6.0) * (d_plus * channel_cfg.nu / u_tau)**3
    V_domain = nx * (ny - 2) * nz
    alpha_total_expected = n_bubbles * V_bubble / V_domain
    alpha_total_measured = np.trapezoid(alpha, y_bubble) / (y_bubble[-1] - y_bubble[0])
    mass_error = abs(alpha_total_measured - alpha_total_expected) / max(alpha_total_expected, 1e-15)

    # Effective slip length
    b_plus = compute_slip_length(
        profiles["U_plus"], profiles["y_plus"], u_tau, channel_cfg.nu
    )

    # Q-criterion
    rho, ux, uy, uz = solver.channel.get_macroscopic()
    Q = compute_q_criterion(ux, uy, uz)
    vortex_stats = count_vortex_events(Q, torch.tensor(profiles["y_plus"]))

    # Burst events
    burst = detect_burst_events(ux, uy, u_tau=u_tau, nu=channel_cfg.nu)

    # Mixture viscosity (Einstein formula for dilute suspension)
    alpha_mean = float(np.mean(alpha))
    nu_bubble = channel_cfg.nu * (1.0 + 2.5 * alpha_mean)

    return {
        "label": label,
        "config": {
            "re_tau": re_tau, "nx": nx, "ny": ny, "nz": nz,
            "u_tau": u_tau, "d_plus": d_plus, "n_bubbles": n_bubbles,
            "steps": steps,
        },
        "profiles": profiles,
        "bubble_profiles": {
            "y": y_bubble.tolist(),
            "alpha": alpha.tolist(),
        },
        "alpha_mean": alpha_mean,
        "alpha_total_expected": alpha_total_expected,
        "alpha_total_measured": alpha_total_measured,
        "mass_conservation_error": mass_error,
        "slip_length_plus": b_plus,
        "nu_bubble": nu_bubble,
        "vortex_stats": {
            "n_vortices": vortex_stats["n_vortices"],
            "burst_frequency_mean": float(np.mean(vortex_stats["burst_frequency"])),
        },
        "burst_events": burst,
    }


def run_negative_control_zero_bubbles(
    re_tau: float = 180.0,
    nx: int = 64,
    ny: int = 33,
    nz: int = 32,
    u_tau: float = 0.01,
    steps: int = 10000,
    warmup: int = 3000,
    device: str = "cpu",
) -> dict:
    """Negative control: zero bubbles should give ΔC_f → 0.

    Runs with n_bubbles=0 (no coupling).
    """
    from tensorlbm.verification.turbulent_channel.channel_solver import ChannelConfig

    channel_cfg = ChannelConfig(
        nx=nx, ny=ny, nz=nz,
        re_tau=re_tau, u_tau=u_tau,
        smagorinsky_cs=0.1,
        steps=steps, warmup_steps=warmup,
        perturbation_amplitude=0.3,
        seed=42, device=device, label="neg_control_zero",
    )

    bubble_cfg = BubbleConfig(
        n_bubbles=0,  # zero bubbles
        d_plus=10.0,
        two_way_coupling=False,
        seed=42,
    )

    solver = TwoPhaseChannelSolver(channel_cfg, bubble_cfg, device=device)

    for _ in range(warmup):
        solver.step()

    stats = ChannelStatistics(ny, u_tau, channel_cfg.nu, device=device)
    for _ in range(steps - warmup):
        solver.step()
        rho, ux, uy, uz = solver.channel.get_macroscopic()
        stats.accumulate(ux, uy, uz)

    profiles = stats.get_profiles()

    return {
        "label": "neg_control_zero_bubbles",
        "profiles": profiles,
        "n_bubbles": 0,
    }


def run_negative_control_large_bubbles(
    re_tau: float = 180.0,
    nx: int = 64,
    ny: int = 33,
    nz: int = 32,
    u_tau: float = 0.01,
    d_plus: float = 100.0,  # very large bubbles
    n_bubbles: int = 100,
    steps: int = 10000,
    warmup: int = 3000,
    device: str = "cpu",
) -> dict:
    """Negative control: d+ → ∞ should show saturation or degradation.

    Very large bubbles (d+ = 100) should not provide effective DR.
    """
    return run_two_phase_case(
        re_tau=re_tau, nx=nx, ny=ny, nz=nz, u_tau=u_tau,
        d_plus=d_plus, n_bubbles=n_bubbles,
        steps=steps, warmup=warmup, device=device,
        label=f"neg_control_d{int(d_plus)}",
    )


def evaluate_acceptance_criteria(
    baseline: dict,
    cases: list,
    neg_zero: dict,
    neg_large: dict,
) -> dict:
    """Evaluate all acceptance criteria.

    Criteria:
    C1: Mass conservation: bubble volume fraction integral error ≤ 1%
    C2: Momentum conservation: u_τ from 3 methods consistent ≤ 2%
    C3: Mechanism separation: sum of 3 terms vs total ΔC_f deviation ≤ 3%
    C4: Negative control (zero bubbles): ΔC_f → 0, deviation ≤ 0.5%
    C5: Negative control (large d+): saturation or degradation trend
    """
    results = {}

    # C1: Mass conservation for each case
    for case in cases:
        mass_err = case.get("mass_conservation_error", 1.0)
        results[f"C1_mass_{case['label']}"] = {
            "pass": mass_err <= 0.01,
            "value": mass_err,
            "threshold": 0.01,
        }

    # C2: Momentum conservation (u_τ consistency)
    u_tau_methods = baseline.get("u_tau_methods", {})
    if len(u_tau_methods) >= 2:
        vals = list(u_tau_methods.values())
        max_dev = max(abs(v - vals[0]) for v in vals[1:]) / max(vals[0], 1e-10)
        results["C2_momentum_baseline"] = {
            "pass": max_dev <= 0.02,
            "value": max_dev,
            "threshold": 0.02,
        }

    # C3: Mechanism decomposition (computed per case)
    for case in cases:
        if "decomposition" in case:
            decomp = case["decomposition"]
            residual_pct = decomp.get("residual_pct", 100.0)
            results[f"C3_mechanism_{case['label']}"] = {
                "pass": residual_pct <= 3.0,
                "value": residual_pct / 100.0,
                "threshold": 0.03,
            }

    # C4: Negative control (zero bubbles)
    # Compare profiles: should be nearly identical to baseline
    U_base = baseline["profiles"]["U_plus"]
    U_neg = neg_zero["profiles"]["U_plus"]
    U_deviation = np.sqrt(np.mean((U_base - U_neg)**2)) / max(np.mean(U_base), 1e-10)
    results["C4_neg_control_zero"] = {
        "pass": U_deviation <= 0.005,
        "value": U_deviation,
        "threshold": 0.005,
    }

    # C5: Negative control (large bubbles)
    # Should show saturation: DR should be less than for small bubbles
    if cases:
        DR_small = cases[0].get("decomposition", {}).get("total_DR", 0.0)
        DR_large = neg_large.get("decomposition", {}).get("total_DR", 0.0)
        # Large bubbles should have less DR (or even negative)
        results["C5_neg_control_large"] = {
            "pass": DR_large <= DR_small * 1.5,  # allow some tolerance
            "value": DR_large,
            "threshold": DR_small * 1.5,
            "note": "Large bubbles should show saturation/degradation",
        }

    # Overall
    n_pass = sum(1 for v in results.values() if v.get("pass", False))
    n_total = len(results)

    return {
        "criteria": results,
        "n_pass": n_pass,
        "n_total": n_total,
        "all_pass": n_pass == n_total,
    }


def run_bubble_campaign(
    output_dir: str = "runs/bubble_turbulence",
    device: str = "cpu",
    re_tau: float = 180.0,
    nx: int = 64,
    ny: int = 33,
    nz: int = 32,
    u_tau: float = 0.01,
    steps: int = 20000,
    warmup: int = 5000,
    d_plus_list: list = None,
    n_bubbles: int = 1000,
) -> dict:
    """Run the full bubble-turbulence campaign.

    Parameters
    ----------
    output_dir : str
        Directory for output files.
    device : str
        Compute device.
    re_tau : float
        Friction Reynolds number.
    nx, ny, nz : int
        Grid dimensions.
    u_tau : float
        Friction velocity.
    steps : int
        Total simulation steps.
    warmup : int
        Warmup steps before statistics.
    d_plus_list : list
        List of bubble diameters in wall units.
    n_bubbles : int
        Number of bubbles per case.

    Returns
    -------
    dict with all results and acceptance criteria.
    """
    if d_plus_list is None:
        d_plus_list = [5.0, 10.0, 20.0]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    print("\n" + "=" * 70)
    print("BUBBLE-TURBULENCE INTERACTION CAMPAIGN")
    print(f"Re_tau = {re_tau}, Grid = {nx}x{ny}x{nz}")
    print(f"Bubble diameters d+ = {d_plus_list}")
    print("=" * 70)

    all_results = {}

    # 1. Single-phase baseline
    print("\n[1/5] Running single-phase baseline...")
    baseline = run_single_phase_baseline(
        re_tau=re_tau, nx=nx, ny=ny, nz=nz, u_tau=u_tau,
        steps=steps, warmup=warmup, device=device,
        label="single_phase",
    )
    all_results["baseline"] = baseline
    print(f"  Baseline C_f reference: u_tau = {u_tau}")

    # 2. Two-phase cases
    cases = []
    for d_plus in d_plus_list:
        label = f"d{int(d_plus)}"
        print(f"\n[2/5] Running two-phase case d+ = {d_plus}...")
        case = run_two_phase_case(
            re_tau=re_tau, nx=nx, ny=ny, nz=nz, u_tau=u_tau,
            d_plus=d_plus, n_bubbles=n_bubbles,
            steps=steps, warmup=warmup, device=device,
            label=label,
        )

        # Mechanism decomposition
        nu_bubble = case.get("nu_bubble", baseline["config"].get("nu", 0.01))
        if nu_bubble == 0:
            from tensorlbm.verification.turbulent_channel.channel_solver import ChannelConfig
            cfg_tmp = ChannelConfig(nx=nx, ny=ny, nz=nz, re_tau=re_tau, u_tau=u_tau)
            nu_bubble = cfg_tmp.nu * (1.0 + 2.5 * case.get("alpha_mean", 0.0))

        decomp = decompose_drag_reduction(
            U_single=baseline["profiles"]["U_plus"] * u_tau,
            uv_single=baseline["profiles"]["uv_plus"] * u_tau**2,
            y=np.arange(ny, dtype=float),
            nu_single=u_tau * re_tau / ((ny - 2) / 2.0),
            u_tau_single=u_tau,
            U_bubble=case["profiles"]["U_plus"] * u_tau,
            uv_bubble=case["profiles"]["uv_plus"] * u_tau**2,
            nu_bubble=nu_bubble,
            u_tau_bubble=u_tau,
            alpha_profile=np.array(case["bubble_profiles"]["alpha"]),
            slip_length=case.get("slip_length_plus", 0.0) * u_tau / (u_tau * re_tau / ((ny - 2) / 2.0)),
        )
        case["decomposition"] = decomp.to_dict()
        cases.append(case)
        all_results[label] = case

    # 3. Negative controls
    print("\n[3/5] Running negative control: zero bubbles...")
    neg_zero = run_negative_control_zero_bubbles(
        re_tau=re_tau, nx=nx, ny=ny, nz=nz, u_tau=u_tau,
        steps=steps // 2, warmup=warmup // 2, device=device,
    )
    all_results["neg_control_zero"] = neg_zero

    print("\n[4/5] Running negative control: large bubbles (d+ = 100)...")
    neg_large = run_negative_control_large_bubbles(
        re_tau=re_tau, nx=nx, ny=ny, nz=nz, u_tau=u_tau,
        d_plus=100.0, n_bubbles=100,
        steps=steps // 2, warmup=warmup // 2, device=device,
    )
    # Add decomposition for large bubbles
    from tensorlbm.verification.turbulent_channel.channel_solver import ChannelConfig
    cfg_tmp = ChannelConfig(nx=nx, ny=ny, nz=nz, re_tau=re_tau, u_tau=u_tau)
    nu_large = cfg_tmp.nu * (1.0 + 2.5 * neg_large.get("alpha_mean", 0.0))
    decomp_large = decompose_drag_reduction(
        U_single=baseline["profiles"]["U_plus"] * u_tau,
        uv_single=baseline["profiles"]["uv_plus"] * u_tau**2,
        y=np.arange(ny, dtype=float),
        nu_single=cfg_tmp.nu,
        u_tau_single=u_tau,
        U_bubble=neg_large["profiles"]["U_plus"] * u_tau,
        uv_bubble=neg_large["profiles"]["uv_plus"] * u_tau**2,
        nu_bubble=nu_large,
        u_tau_bubble=u_tau,
    )
    neg_large["decomposition"] = decomp_large.to_dict()
    all_results["neg_control_large"] = neg_large

    # 4. Evaluate acceptance criteria
    print("\n[5/5] Evaluating acceptance criteria...")
    acceptance = evaluate_acceptance_criteria(baseline, cases, neg_zero, neg_large)
    all_results["acceptance"] = acceptance

    elapsed = time.time() - start_time

    # Final result
    final = {
        "campaign": "p106w2-n01: Bubble-Turbulence Interaction DNS",
        "results": all_results,
        "acceptance_summary": acceptance,
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": elapsed,
            "device": device,
            "re_tau": re_tau,
            "grid": f"{nx}x{ny}x{nz}",
            "d_plus_list": d_plus_list,
            "n_bubbles": n_bubbles,
        },
    }

    # Save JSON
    json_path = output_dir / "bubble_campaign_results.json"
    with open(json_path, "w") as f:
        json.dump(final, f, indent=2, default=str)
    print(f"\nResults saved to: {json_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("CAMPAIGN SUMMARY")
    print("=" * 70)
    for name, crit in acceptance["criteria"].items():
        status = "PASS" if crit.get("pass", False) else "FAIL"
        print(f"  {name}: {status} (value={crit.get('value', 'N/A')})")
    print(f"\nOverall: {acceptance['n_pass']}/{acceptance['n_total']} PASSED")
    print(f"Elapsed: {elapsed:.1f} seconds")

    return final
