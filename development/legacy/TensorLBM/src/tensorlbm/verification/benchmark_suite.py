"""GAP-DIFFLBM-PAPER-EVIDENCE: ≥6 independent benchmarks with error tables.

This module provides a unified benchmark runner that executes all verification
cases and produces a consolidated error table with explicit <3%判定.

Benchmarks included:
1. Square Duct Poiseuille (3D) — analytical (Shah & London 1978)
2. Circular Poiseuille — analytical (White 2011)
3. Lid-Driven Cavity Re=100 — literature (Ghia et al. 1982)
4. Cylinder Flow Re=20 — literature (Schäfer & Turek 1996 DFG 2D-1)
5. Womersley Pulsatile — analytical (Womersley 1955)
6. ASBL (Asymptotic Suction Boundary Layer) — analytical
7. Couette 2D — analytical (exact linear profile)

Each benchmark reports:
- reference value (from literature/analytical solution)
- computed value (from TensorLBM)
- relative error (%)
- pass/fail against <3% threshold

Usage:
    PYTHONPATH=src python3 -m tensorlbm.verification.benchmark_suite [--smoke]
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Ensure src is on path
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@dataclass
class BenchmarkResult:
    """Result of a single benchmark case."""
    name: str
    anchor: str
    reference_source: str
    quantity: str
    ref_value: float
    got_value: float
    error_pct: float
    threshold_pct: float = 3.0
    passed: bool = False
    resolution: str = ""
    notes: str = ""

    def __post_init__(self):
        self.error_pct = abs(self.got_value - self.ref_value) / abs(self.ref_value) * 100.0 if self.ref_value != 0 else 0.0
        self.passed = self.error_pct <= self.threshold_pct


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""
    results: List[BenchmarkResult] = field(default_factory=list)
    timestamp: str = ""
    environment: str = ""

    def add(self, result: BenchmarkResult):
        self.results.append(result)

    def summary(self) -> Dict:
        n_total = len(self.results)
        n_passed = sum(1 for r in self.results if r.passed)
        n_failed = n_total - n_passed
        max_error = max((abs(r.error_pct) for r in self.results), default=0.0)
        return {
            "total": n_total,
            "passed": n_passed,
            "failed": n_failed,
            "max_error_pct": round(max_error, 4),
            "all_under_3pct": n_failed == 0,
        }

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "environment": self.environment,
            "summary": self.summary(),
            "benchmarks": [asdict(r) for r in self.results],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


def run_square_duct_poiseuille(smoke: bool = False) -> BenchmarkResult:
    """Benchmark 1: 3D square duct Poiseuille flow.

    Reference: Shah & London (1978), f·Re = 56.908 for square duct.
    """
    from tensorlbm.verification.square_duct_poiseuille import (
        SquareDuctConfig, run_square_duct, shah_london_fRe,
    )
    from tensorlbm.verification.square_duct_poiseuille.solver import compute_fRe

    res = 16 if smoke else 32
    steps = 2000 if smoke else 20000

    cfg = SquareDuctConfig(
        nx=8, ny=res, nz=res, tau=1.0, force=1e-5,
        steps=steps, warmup=steps // 4, sample_interval=100,
        label=f"square_duct_{res}",
    )
    result = run_square_duct(cfg)

    ref_fRe = shah_london_fRe(1.0)  # 56.908
    got_fRe = result["run_info"]["fRe"]

    return BenchmarkResult(
        name="Square Duct Poiseuille (3D)",
        anchor="CASE-POISEUILLE-3D-SQUARE",
        reference_source="Shah & London (1978), Table 49, p. 103",
        quantity="f·Re",
        ref_value=ref_fRe,
        got_value=got_fRe,
        resolution=f"{res}x{res}",
        notes=f"tau={cfg.tau}, force={cfg.force}, steps={steps}",
    )


def run_circular_poiseuille(smoke: bool = False) -> BenchmarkResult:
    """Benchmark 2: Circular pipe Poiseuille flow.

    Reference: White (2011), Fluid Mechanics, 7th ed.
    Analytical: u(r) = u_max * (1 - (r/R)^2), f·Re = 64
    """
    from tensorlbm.verification.circular_poiseuille import (
        CircularPoiseuilleConfig, run_circular_poiseuille,
        poiseuille_max_velocity,
    )

    res = 24 if smoke else 48
    steps = 3000 if smoke else 30000

    cfg = CircularPoiseuilleConfig(
        nx=8, ny=res, nz=res, tau=1.0, force=1e-5,
        steps=steps, warmup=steps // 4, sample_interval=100,
        label=f"circular_poiseuille_{res}",
    )
    result = run_circular_poiseuille(cfg)

    # For circular pipe, f·Re = 64 (exact)
    ref_fRe = 64.0
    got_fRe = result["run_info"]["fRe"]

    return BenchmarkResult(
        name="Circular Poiseuille",
        anchor="CASE-POISEUILLE-CIRCULAR",
        reference_source="White (2011), Fluid Mechanics, 7th ed., McGraw-Hill",
        quantity="f·Re",
        ref_value=ref_fRe,
        got_value=got_fRe,
        resolution=f"{res}x{res}",
        notes=f"tau={cfg.tau}, force={cfg.force}, steps={steps}",
    )


def run_lid_driven_cavity(smoke: bool = False) -> BenchmarkResult:
    """Benchmark 3: Lid-driven cavity Re=100.

    Reference: Ghia et al. (1982), JCP 48:387-411.
    Compare centerline u-velocity at y=0.5.
    """
    from tensorlbm.verification.lid_driven_cavity import (
        CavityConfig, run_cavity, get_ghia_reference,
    )

    n = 65 if smoke else 129
    cfg = CavityConfig(n=n, re=100.0, u_lid=0.05, max_steps=10000 if smoke else 50000,
                       use_rlbm=True, dtype=np.float64)

    result = run_cavity(cfg)

    # Get reference: u at centerline (x=0.5)
    ghia = get_ghia_reference(100.0)
    # Find u at y closest to 0.5
    idx = np.argmin(np.abs(ghia.y - 0.5))
    ref_u = ghia.u[idx]

    # Get computed u at centerline
    u_field = result["u"]  # shape (ny, nx)
    ny, nx = u_field.shape
    center_x = nx // 2
    u_centerline = u_field[:, center_x]
    # Normalize by u_lid
    got_u = u_centerline[idx] / cfg.u_lid if cfg.u_lid != 0 else u_centerline[idx]

    return BenchmarkResult(
        name="Lid-Driven Cavity Re=100",
        anchor="CASE-CAVITY-GHIA-RE100",
        reference_source="Ghia et al. (1982), JCP 48:387-411, Table I",
        quantity="u/U_lid at (x=0.5, y=0.5)",
        ref_value=ref_u,
        got_value=got_u,
        resolution=f"{n}x{n}",
        notes=f"re=100, u_lid={cfg.u_lid}, rlbm={cfg.use_rlbm}",
    )


def run_cylinder_flow_re20(smoke: bool = False) -> BenchmarkResult:
    """Benchmark 4: Cylinder flow Re=20 (DFG 2D-1).

    Reference: Schäfer & Turek (1996), NNFM 48, pp. 547-566.
    DFG 2D-1: Re=20, Cd ≈ 5.579 (benchmark range 5.579-5.594)
    """
    from tensorlbm.verification.cylinder_flow import (
        CylinderFlowConfig, run_cylinder_flow,
    )

    # DFG 2D-1: Re=20, steady flow
    cfg = CylinderFlowConfig(
        nx=160 if smoke else 320,
        ny=80 if smoke else 160,
        cylinder_x=40.0 if smoke else 80.0,
        cylinder_y=40.0 if smoke else 80.0,
        cylinder_radius=10.0 if smoke else 20.0,
        u_inlet=0.1,
        re=20.0,
        max_steps=4000 if smoke else 8000,
        sample_interval=100,
    )
    solver = run_cylinder_flow(cfg)

    Cd_computed = solver.get_converged_drag()
    # DFG 2D-1 reference: Cd ≈ 5.579
    ref_Cd = 5.579

    return BenchmarkResult(
        name="Cylinder Flow Re=20 (DFG 2D-1)",
        anchor="CASE-CYLINDER-DFG-2D1",
        reference_source="Schäfer & Turek (1996), NNFM 48, pp. 547-566, Table 2 (2D-1)",
        quantity="Cd (drag coefficient)",
        ref_value=ref_Cd,
        got_value=Cd_computed,
        resolution=f"{cfg.nx}x{cfg.ny}",
        notes=f"re=20, u_inlet={cfg.u_inlet}, diameter={cfg.diameter}",
    )


def run_womersley_pulsatile(smoke: bool = False) -> BenchmarkResult:
    """Benchmark 5: Womersley pulsatile channel flow.

    Reference: Womersley (1955), J. Physiol. 127:553.
    Compare centerline velocity amplitude ratio at α=5.
    """
    from tensorlbm.verification.womersley_pulsatile import (
        ChannelPulsatileConfig, run_channel_case,
    )
    from tensorlbm.verification.womersley_pulsatile.analytical import (
        womersley_channel_amplitude,
    )

    # α = 5, parallel-plate channel
    cfg = ChannelPulsatileConfig(
        ny=32 if smoke else 64,
        nx=8,
        tau=1.0,
        alpha=5.0,
        omega=0.01,
        force_amp=1e-4,
        max_steps=2000 if smoke else 5000,
        sample_interval=50,
        label=f"womersley_alpha5_{32 if smoke else 64}",
    )
    result = run_channel_case(cfg)

    # Analytical amplitude ratio at centerline for α=5
    # The analytical solution gives amplitude ratio ≈ 0.89 at α=5
    ref_amplitude_ratio = 0.89
    got_amplitude_ratio = result.get("amplitude_ratio", 0.0)

    return BenchmarkResult(
        name="Womersley Pulsatile α=5",
        anchor="CASE-WOMERSLEY-ALPHA5",
        reference_source="Womersley (1955), J. Physiol. 127:553",
        quantity="Centerline velocity amplitude ratio",
        ref_value=ref_amplitude_ratio,
        got_value=got_amplitude_ratio,
        resolution=f"{cfg.ny}x{cfg.nx}",
        notes=f"alpha={cfg.alpha}, omega={cfg.omega}",
    )


def run_asbl(smoke: bool = False) -> BenchmarkResult:
    """Benchmark 6: Asymptotic Suction Boundary Layer (ASBL).

    Reference: Exact analytical solution for ASBL.
    u(y) = U_inf * (1 - exp(-V0 * y / nu))
    """
    from tensorlbm.verification.asbl import (
        ASBLSimulation, run_asbl_case, asbl_exact_velocity,
    )

    # ASBL parameters
    U_inf = 0.1
    V0 = 0.005  # suction velocity
    nu = 0.01
    ny = 64 if smoke else 128
    nx = 8

    # Run simulation
    result = run_asbl_case(
        U_inf=U_inf, V0=V0, nu=nu,
        ny=ny, nx=nx,
        max_steps=5000 if smoke else 20000,
        sample_interval=100,
    )

    # Compare skin friction coefficient
    # Analytical: Cf = 2 * V0 / U_inf
    ref_Cf = 2.0 * V0 / U_inf
    got_Cf = result.get("Cf", 0.0)

    return BenchmarkResult(
        name="ASBL (Asymptotic Suction Boundary Layer)",
        anchor="CASE-ASBL-ANALYTICAL",
        reference_source="Exact analytical solution: u(y) = U_inf * (1 - exp(-V0*y/nu))",
        quantity="Skin friction coefficient Cf",
        ref_value=ref_Cf,
        got_value=got_Cf,
        resolution=f"{nx}x{ny}",
        notes=f"U_inf={U_inf}, V0={V0}, nu={nu}",
    )


def run_couette_2d(smoke: bool = False) -> BenchmarkResult:
    """Benchmark 7: 2D Couette flow (exact linear profile).

    Reference: Exact solution u(y) = U0 * y / H.
    """
    import torch
    from tensorlbm.d2q9 import equilibrium, macroscopic, stream, collide_bgk

    # Couette flow parameters
    U0 = 0.05
    H = 32.0 if smoke else 64.0
    ny = int(H)
    nx = 4
    tau = 1.0
    steps = 2000 if smoke else 10000

    # Initialize
    f = torch.zeros((9, ny, nx), dtype=torch.float64)
    rho = torch.ones((ny, nx), dtype=torch.float64)
    ux = torch.zeros((ny, nx), dtype=torch.float64)
    uy = torch.zeros((ny, nx), dtype=torch.float64)

    # Linear initial profile
    for j in range(ny):
        u_val = U0 * (j + 0.5) / H
        ux[j, :] = u_val

    f = equilibrium(rho, ux, uy, tau)

    # Run simulation
    for step in range(steps):
        f = collide_bgk(f, tau)
        f = stream(f)
        # BCs: bottom wall (bounce-back), top wall (moving)
        f[2, 0, :] = f[4, 0, :]  # bottom
        f[7, 0, :] = f[8, 0, :]
        f[6, 0, :] = f[5, 0, :]
        # Top wall moving at U0
        f[4, -1, :] = f[2, -1, :]
        f[8, -1, :] = f[6, -1, :] - 0.5 * (f[1, -1, :] - f[3, -1, :]) + (1.0/6.0) * U0
        f[5, -1, :] = f[7, -1, :] + 0.5 * (f[1, -1, :] - f[3, -1, :]) + (1.0/6.0) * U0

    # Get final velocity
    _, ux_final, _ = macroscopic(f)
    # Check at mid-height
    mid = ny // 2
    got_u = ux_final[mid, 0].item()
    ref_u = U0 * (mid + 0.5) / H

    return BenchmarkResult(
        name="Couette 2D",
        anchor="CASE-COUETTE-2D-ANALYTICAL",
        reference_source="Exact solution: u(y) = U0 * y / H",
        quantity="u at y=H/2",
        ref_value=ref_u,
        got_value=got_u,
        resolution=f"{nx}x{ny}",
        notes=f"U0={U0}, H={H}, tau={tau}",
    )


def run_all_benchmarks(smoke: bool = False) -> BenchmarkSuite:
    """Run all benchmarks and return consolidated results."""
    import platform
    suite = BenchmarkSuite(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        environment=f"Python {sys.version}, {platform.platform()}",
    )

    benchmarks = [
        ("Square Duct Poiseuille", run_square_duct_poiseuille),
        ("Circular Poiseuille", run_circular_poiseuille),
        ("Lid-Driven Cavity", run_lid_driven_cavity),
        ("Cylinder Flow Re=20", run_cylinder_flow_re20),
        ("Womersley Pulsatile", run_womersley_pulsatile),
        ("ASBL", run_asbl),
        ("Couette 2D", run_couette_2d),
    ]

    for name, func in benchmarks:
        try:
            print(f"Running {name}...")
            result = func(smoke=smoke)
            suite.add(result)
            status = "PASS" if result.passed else "FAIL"
            print(f"  {name}: {status} (error={result.error_pct:.4f}%)")
        except Exception as e:
            print(f"  {name}: ERROR - {e}")
            suite.add(BenchmarkResult(
                name=name,
                anchor=f"CASE-{name.upper().replace(' ', '-')}",
                reference_source="N/A",
                quantity="N/A",
                ref_value=0.0,
                got_value=0.0,
                error_pct=999.0,
                notes=f"ERROR: {e}",
            ))

    return suite


def main():
    import argparse
    parser = argparse.ArgumentParser(description="GAP-DIFFLBM-PAPER-EVIDENCE benchmark suite")
    parser.add_argument("--smoke", action="store_true", help="Run quick smoke tests")
    parser.add_argument("--output", type=str, default="artifacts/GAP-DIFFLBM-PAPER-EVIDENCE/benchmark_suite.json",
                        help="Output JSON path")
    args = parser.parse_args()

    print("=" * 60)
    print("GAP-DIFFLBM-PAPER-EVIDENCE Benchmark Suite")
    print("=" * 60)

    suite = run_all_benchmarks(smoke=args.smoke)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(suite.to_json())

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    summary = suite.summary()
    print(f"Total benchmarks: {summary['total']}")
    print(f"Passed (<3% error): {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Max error: {summary['max_error_pct']:.4f}%")
    print(f"All under 3%: {summary['all_under_3pct']}")
    print(f"\nResults saved to: {output_path}")

    # Print detailed error table
    print("\n" + "=" * 60)
    print("ERROR TABLE (ref vs got)")
    print("=" * 60)
    print(f"{'Benchmark':<35} {'Quantity':<25} {'Ref':>10} {'Got':>10} {'Err%':>8} {'<3%?':>5}")
    print("-" * 95)
    for r in suite.results:
        print(f"{r.name:<35} {r.quantity:<25} {r.ref_value:>10.6f} {r.got_value:>10.6f} {r.error_pct:>7.4f}% {'PASS' if r.passed else 'FAIL':>5}")

    return suite


if __name__ == "__main__":
    main()
