"""KBC collision diagnostic for sphere flow: gamma, H, population, force tracking.

Runs a small D3Q19 sphere flow with KBC collision and BGK for comparison,
tracking per-step:
  - gamma distribution (min/max/mean) over all fluid cells
  - discrete entropy H before/after collision
  - population min/max (positivity check)
  - drag force fx and Cd
  - number of cells with negative populations

This script does NOT modify the solver hot path.  It instruments the
existing collide_kbc_d3q19 by wrapping it to capture intermediate state.

Usage:
    python -m tensorlbm.kbc_diagnostic
    python -m tensorlbm.kbc_diagnostic --steps 20 --nx 16
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from .boundaries3d import (
    apply_simple_channel_boundaries_3d,
    make_channel_wall_mask_3d,
    sphere_mask,
)
from .d3q19 import C, W, equilibrium3d, macroscopic3d
from .entropic_kbc import (
    _kbc_decompose,
    _lattice_constants,
    discrete_entropy,
    kbc_decompose_d3q19,
    solve_gamma_entropy,
)
from .obstacles import compute_obstacle_forces_3d
from .solver3d import collide_bgk3d, stream3d


# ---------------------------------------------------------------------------
# Instrumented KBC collision (captures intermediate state)
# ---------------------------------------------------------------------------

@dataclass
class KBCStepDiagnostic:
    """Per-step diagnostic snapshot."""
    step: int
    gamma_min: float
    gamma_max: float
    gamma_mean: float
    gamma_std: float
    H_before: float
    H_after: float
    H_violation_count: int
    H_violation_max: float
    f_min_before: float
    f_max_before: float
    f_min_after: float
    f_max_after: float
    neg_count_after: int
    neg_min_after: float
    fx: float
    Cd: float | None
    rho_min: float
    rho_max: float
    ux_max: float
    # Decomposition norms
    s_norm: float
    k_norm: float
    h_norm: float
    # Admissibility domain vs gamma
    gamma_below_admissibility: int
    gamma_above_admissibility: int


def _instrumented_kbc_collision(
    f: torch.Tensor,
    tau: float,
    w: torch.Tensor,
    p: dict[str, torch.Tensor],
    *,
    max_iter: int = 28,
    tol: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Run KBC collision and return (f_star, diagnostics_dict)."""
    rho, ux, uy, uz = macroscopic3d(f)
    feq = equilibrium3d(rho, ux, uy, uz)
    f_neq = f - feq

    s, k, h = _kbc_decompose(f_neq, p)

    gamma_init = torch.full(rho.shape, 1.0 - 1.0 / tau, device=f.device, dtype=f.dtype)
    gamma = solve_gamma_entropy(feq, s, h, w, gamma_init, max_iter=max_iter, tol=tol)

    f_star = feq + gamma.unsqueeze(0) * s + h

    # Admissibility domain (natural bounds)
    f_base = feq + h
    eps_s = 1e-30
    s_safe = torch.where(s.abs() > eps_s, s, torch.full_like(s, eps_s))
    ratio = -f_base / s_safe
    neg_inf = torch.full_like(gamma_init, -1e6)
    pos_inf = torch.full_like(gamma_init, 1e6)
    pos_mask = s > eps_s
    ratio_pos = torch.where(pos_mask, ratio, neg_inf.unsqueeze(0).expand_as(ratio))
    gamma_lower_natural = ratio_pos.amax(dim=0)
    neg_mask = s < -eps_s
    ratio_neg = torch.where(neg_mask, ratio, pos_inf.unsqueeze(0).expand_as(ratio))
    gamma_upper_natural = ratio_neg.amin(dim=0)

    diag = {
        "gamma": gamma,
        "s": s,
        "k": k,
        "h": h,
        "feq": feq,
        "f_neq": f_neq,
        "gamma_lower_natural": gamma_lower_natural,
        "gamma_upper_natural": gamma_upper_natural,
    }
    return f_star, diag


# ---------------------------------------------------------------------------
# Main diagnostic runner
# ---------------------------------------------------------------------------

@dataclass
class KBCDiagnosticConfig:
    nx: int = 16
    ny: int = 16
    nz: int = 16
    steps: int = 20
    u_in: float = 0.05
    re: float = 100.0
    device: str = "cpu"


@dataclass
class KBCDiagnosticReport:
    config: dict[str, Any]
    reference_Cd: float
    kbc_steps: list[dict[str, Any]]
    bgk_steps: list[dict[str, Any]]
    kbc_final_Cd: float | None
    bgk_final_Cd: float | None
    findings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _schiller_naumann(re: float) -> float:
    if re < 1e-6:
        return 100.0
    return 24.0 / re * (1.0 + 0.15 * re ** 0.687)


def run_kbc_diagnostic(config: KBCDiagnosticConfig | None = None) -> KBCDiagnosticReport:
    """Run KBC vs BGK sphere flow diagnostic."""
    if config is None:
        config = KBCDiagnosticConfig()

    dev = torch.device(config.device)
    nx, ny, nz = config.nx, config.ny, config.nz
    radius = max(4.0, nx * 0.08)
    u_in = config.u_in
    re = config.re
    nu = u_in * 2.0 * radius / re
    tau = 3.0 * nu + 0.5

    mask = sphere_mask(nx, ny, nz, nx * 0.5, ny * 0.5, nz * 0.5, radius, device=dev)
    wall_mask = make_channel_wall_mask_3d(nz, ny, nx, mask, device=dev)

    w = W.view(19, 1, 1, 1).to(dev)
    p = _lattice_constants(C, W, dev, torch.float32)

    ref_cd = _schiller_naumann(re)
    area = math.pi * radius ** 2

    # --- KBC run ---
    rho0 = torch.ones(nz, ny, nx, device=dev)
    ux0 = torch.full((nz, ny, nx), u_in, device=dev)
    uy0 = torch.zeros(nz, ny, nx, device=dev)
    uz0 = torch.zeros(nz, ny, nx, device=dev)
    f_kbc = equilibrium3d(rho0, ux0, uy0, uz0, device=dev)

    # --- BGK run (for comparison) ---
    f_bgk = equilibrium3d(rho0, ux0, uy0, uz0, device=dev)

    kbc_steps_data: list[dict[str, Any]] = []
    bgk_steps_data: list[dict[str, Any]] = []

    for step in range(1, config.steps + 1):
        # ===== KBC =====
        H_before_kbc = discrete_entropy(f_kbc, w)
        f_min_before = f_kbc.min().item()
        f_max_before = f_kbc.max().item()

        f_kbc, kbc_diag = _instrumented_kbc_collision(f_kbc, tau, w, p)

        H_after_kbc = discrete_entropy(f_kbc, w)
        f_min_after = f_kbc.min().item()
        f_max_after = f_kbc.max().item()
        neg_mask_kbc = f_kbc < 0
        neg_count_kbc = int(neg_mask_kbc.sum().item())

        # Stream
        f_kbc = stream3d(f_kbc)

        # Force (before bounce-back)
        fx_kbc, _, _ = compute_obstacle_forces_3d(f_kbc, mask)
        fx_kbc_val = float(fx_kbc.item())
        cd_kbc = fx_kbc_val / (0.5 * u_in ** 2 * area) if u_in > 0 else None

        # Boundaries
        f_kbc = apply_simple_channel_boundaries_3d(f_kbc, u_in=u_in, wall_mask=wall_mask, obstacle_mask=mask)

        # Macros
        rho_kbc, ux_kbc, _, _ = macroscopic3d(f_kbc)

        # Gamma stats
        gamma = kbc_diag["gamma"]
        gamma_lower = kbc_diag["gamma_lower_natural"]
        gamma_upper = kbc_diag["gamma_upper_natural"]

        # H-theorem check
        H_violation = H_after_kbc > H_before_kbc + 1e-10
        H_violation_count = int(H_violation.sum().item())
        H_violation_max = float((H_after_kbc - H_before_kbc).max().item()) if H_violation_count > 0 else 0.0

        # Admissibility check
        below = int((gamma < gamma_lower - 1e-8).sum().item())
        above = int((gamma > gamma_upper + 1e-8).sum().item())

        step_diag = KBCStepDiagnostic(
            step=step,
            gamma_min=float(gamma.min().item()),
            gamma_max=float(gamma.max().item()),
            gamma_mean=float(gamma.mean().item()),
            gamma_std=float(gamma.std().item()),
            H_before=float(H_before_kbc.mean().item()),
            H_after=float(H_after_kbc.mean().item()),
            H_violation_count=H_violation_count,
            H_violation_max=H_violation_max,
            f_min_before=f_min_before,
            f_max_before=f_max_before,
            f_min_after=f_min_after,
            f_max_after=f_max_after,
            neg_count_after=neg_count_kbc,
            neg_min_after=float(f_kbc[neg_mask_kbc].min().item()) if neg_count_kbc > 0 else 0.0,
            fx=fx_kbc_val,
            Cd=cd_kbc,
            rho_min=float(rho_kbc.min().item()),
            rho_max=float(rho_kbc.max().item()),
            ux_max=float(ux_kbc.abs().max().item()),
            s_norm=float(kbc_diag["s"].abs().max().item()),
            k_norm=float(kbc_diag["k"].abs().max().item()),
            h_norm=float(kbc_diag["h"].abs().max().item()),
            gamma_below_admissibility=below,
            gamma_above_admissibility=above,
        )
        kbc_steps_data.append(asdict(step_diag))

        # ===== BGK (for comparison) =====
        f_bgk = collide_bgk3d(f_bgk, tau=tau)
        f_bgk = stream3d(f_bgk)
        fx_bgk, _, _ = compute_obstacle_forces_3d(f_bgk, mask)
        fx_bgk_val = float(fx_bgk.item())
        cd_bgk = fx_bgk_val / (0.5 * u_in ** 2 * area) if u_in > 0 else None
        f_bgk = apply_simple_channel_boundaries_3d(f_bgk, u_in=u_in, wall_mask=wall_mask, obstacle_mask=mask)

        bgk_steps_data.append({
            "step": step,
            "fx": fx_bgk_val,
            "Cd": cd_bgk,
            "f_min": float(f_bgk.min().item()),
            "f_max": float(f_bgk.max().item()),
            "neg_count": int((f_bgk < 0).sum().item()),
        })

        if not torch.isfinite(f_kbc).all():
            break
        if not torch.isfinite(f_bgk).all():
            break

    # Final Cd (average of last half)
    kbc_final_cd = None
    if kbc_steps_data:
        late = [s["Cd"] for s in kbc_steps_data[len(kbc_steps_data)//2:] if s["Cd"] is not None]
        kbc_final_cd = sum(late) / len(late) if late else kbc_steps_data[-1]["Cd"]

    bgk_final_cd = None
    if bgk_steps_data:
        late = [s["Cd"] for s in bgk_steps_data[len(bgk_steps_data)//2:] if s["Cd"] is not None]
        bgk_final_cd = sum(late) / len(late) if late else bgk_steps_data[-1]["Cd"]

    # --- Findings ---
    findings: list[str] = []
    if kbc_steps_data:
        last = kbc_steps_data[-1]
        if last["neg_count_after"] > 0:
            findings.append(
                f"NEGATIVE POPULATIONS: {last['neg_count_after']} cells have f*<0 "
                f"(min={last['neg_min_after']:.6e}) at step {last['step']}"
            )
        if last["H_violation_count"] > 0:
            findings.append(
                f"H-THEOREM VIOLATION: {last['H_violation_count']} cells have H(f*)>H(f) "
                f"(max excess={last['H_violation_max']:.6e})"
            )
        if last["gamma_below_admissibility"] > 0 or last["gamma_above_admissibility"] > 0:
            findings.append(
                f"GAMMA OUTSIDE ADMISSIBILITY: {last['gamma_below_admissibility']} below, "
                f"{last['gamma_above_admissibility']} above natural bounds"
            )
        if kbc_final_cd is not None and bgk_final_cd is not None:
            ratio = kbc_final_cd / bgk_final_cd if bgk_final_cd != 0 else float('inf')
            findings.append(
                f"Cd RATIO KBC/BGK = {ratio:.2f} (KBC={kbc_final_cd:.2f}, BGK={bgk_final_cd:.2f}, ref={ref_cd:.2f})"
            )
        # Check h retention: h should not decay to near-zero (it's fully retained)
        h_peak = max(s["h_norm"] for s in kbc_steps_data)
        h_last = kbc_steps_data[-1]["h_norm"]
        if h_peak > 1e-10 and h_last < 0.1 * h_peak:
            findings.append(
                f"H-MODE DECAYED: h_norm peaked at {h_peak:.6e} then decayed to {h_last:.6e} "
                f"(ratio={h_last/h_peak:.2f}) — h is being relaxed, contradicting f*=feq+γ·s+h"
            )
        elif h_peak > 1e-10:
            findings.append(
                f"H-MODE RETAINED: h_norm peaked at {h_peak:.6e}, last={h_last:.6e} "
                f"(ratio={h_last/h_peak:.2f}) — h is NOT relaxed by KBC collision"
            )

    return KBCDiagnosticReport(
        config=asdict(config),
        reference_Cd=ref_cd,
        kbc_steps=kbc_steps_data,
        bgk_steps=bgk_steps_data,
        kbc_final_Cd=kbc_final_cd,
        bgk_final_Cd=bgk_final_cd,
        findings=findings,
    )


def write_kbc_diagnostic_report(report: KBCDiagnosticReport, path: str | Path) -> Path:
    """Write the diagnostic report as JSON."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


def main():
    parser = argparse.ArgumentParser(description="KBC collision diagnostic for sphere flow")
    parser.add_argument("--nx", type=int, default=16, help="Grid size x")
    parser.add_argument("--ny", type=int, default=16, help="Grid size y")
    parser.add_argument("--nz", type=int, default=16, help="Grid size z")
    parser.add_argument("--steps", type=int, default=20, help="Number of time steps")
    parser.add_argument("--u-in", type=float, default=0.05, help="Inlet velocity")
    parser.add_argument("--re", type=float, default=100.0, help="Reynolds number")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--output", type=str, default="kbc_diagnostic_report.json", help="Output JSON path")
    args = parser.parse_args()

    config = KBCDiagnosticConfig(
        nx=args.nx, ny=args.ny, nz=args.nz,
        steps=args.steps, u_in=args.u_in, re=args.re, device=args.device,
    )
    report = run_kbc_diagnostic(config)

    # Print summary
    print("=" * 72)
    print("KBC Collision Diagnostic — Sphere Flow")
    print("=" * 72)
    print(f"Grid: {config.nx}³, Steps: {config.steps}, Re: {config.re}, u_in: {config.u_in}")
    print(f"Reference Cd (Schiller-Naumann): {report.reference_Cd:.4f}")
    print(f"KBC final Cd: {report.kbc_final_Cd}")
    print(f"BGK final Cd: {report.bgk_final_Cd}")
    print()
    print("Per-step KBC summary:")
    print(f"{'Step':>4} {'gamma_min':>10} {'gamma_max':>10} {'gamma_mean':>10} "
          f"{'f_min':>12} {'f_max':>10} {'neg#':>6} {'H_bef':>10} {'H_aft':>10} "
          f"{'H_viol':>6} {'fx':>10} {'Cd':>8}")
    for s in report.kbc_steps:
        print(f"{s['step']:4d} {s['gamma_min']:10.4f} {s['gamma_max']:10.4f} {s['gamma_mean']:10.4f} "
              f"{s['f_min_after']:12.6e} {s['f_max_after']:10.4f} {s['neg_count_after']:6d} "
              f"{s['H_before']:10.4e} {s['H_after']:10.4e} {s['H_violation_count']:6d} "
              f"{s['fx']:10.4f} {str(s['Cd']):>8s}")

    print()
    print("Per-step BGK summary:")
    print(f"{'Step':>4} {'f_min':>12} {'f_max':>10} {'neg#':>6} {'fx':>10} {'Cd':>8}")
    for s in report.bgk_steps:
        print(f"{s['step']:4d} {s['f_min']:12.6e} {s['f_max']:10.4f} {s['neg_count']:6d} "
              f"{s['fx']:10.4f} {str(s['Cd']):>8s}")

    print()
    print("Decomposition norms (KBC):")
    print(f"{'Step':>4} {'s_norm':>12} {'k_norm':>12} {'h_norm':>12} "
          f"{'below_adm':>10} {'above_adm':>10}")
    for s in report.kbc_steps:
        print(f"{s['step']:4d} {s['s_norm']:12.6e} {s['k_norm']:12.6e} {s['h_norm']:12.6e} "
              f"{s['gamma_below_admissibility']:10d} {s['gamma_above_admissibility']:10d}")

    print()
    print("Findings:")
    for f in report.findings:
        print(f"  • {f}")

    out = write_kbc_diagnostic_report(report, args.output)
    print(f"\nReport written to: {out}")


if __name__ == "__main__":
    main()
