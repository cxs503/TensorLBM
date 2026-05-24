"""Porous-media gas-water displacement example.

Demonstrates all four porous-media benchmarks on small domains so that
the script runs quickly.  For production-quality results increase the
domain sizes and step counts in the configuration objects.

Usage
-----
    PYTHONPATH=src python examples/porous_drainage.py

"""
from __future__ import annotations

from pathlib import Path

from tensorlbm import (
    CapillaryInvasionConfig,
    LaplaceTestConfig,
    PorousDrainageConfig,
    TwoPhasePoiseuilleConfig,
    run_capillary_invasion,
    run_laplace_test,
    run_porous_drainage,
    run_two_phase_poiseuille,
)

OUTPUT_ROOT = Path("outputs/porous_media_demo")


def demo_laplace() -> None:
    """Run Laplace pressure benchmark and print surface tension estimate."""
    cfg = LaplaceTestConfig(
        nx=80,
        ny=80,
        bubble_radius=15.0,
        G_12=0.9,
        tau1=1.0,
        tau2=1.0,
        rho_water=0.7,
        rho_gas=0.3,
        n_steps=3000,
        output_interval=1000,
        output_root=OUTPUT_ROOT / "laplace",
        overwrite=True,
    )
    result = run_laplace_test(cfg)
    print(
        f"\n[Laplace] ΔP = {result['final_delta_p']:.6f}  "
        f"σ_eff = ΔP·R = {result['sigma_eff']:.4f}\n"
    )


def demo_capillary_invasion() -> None:
    """Run capillary invasion (Washburn) benchmark."""
    cfg = CapillaryInvasionConfig(
        nx=200,
        ny=30,
        tube_width=20,
        G_12=0.9,
        G_ads_water=0.3,
        G_ads_gas=0.0,
        tau_water=1.0,
        tau_gas=1.0,
        rho_water=0.7,
        rho_gas=0.3,
        n_steps=3000,
        output_interval=500,
        output_root=OUTPUT_ROOT / "capillary",
        overwrite=True,
    )
    result = run_capillary_invasion(cfg)
    print(
        f"\n[Capillary] Washburn exponent ≈ {result['washburn_exponent']:.3f}  "
        f"(theory ≈ 0.5)\n"
    )


def demo_two_phase_poiseuille() -> None:
    """Run two-phase Poiseuille benchmark."""
    cfg = TwoPhasePoiseuilleConfig(
        nx=6,
        ny=40,
        tau_water=1.0,
        tau_gas=0.7,
        rho_water=0.7,
        rho_gas=0.3,
        G_x=5e-5,
        G_12=0.9,
        n_steps=6000,
        output_interval=2000,
        output_root=OUTPUT_ROOT / "poiseuille",
        overwrite=True,
    )
    result = run_two_phase_poiseuille(cfg)
    print(
        f"\n[Two-phase Poiseuille] Relative L2 error = {result['l2_error_rel']:.4f}\n"
    )


def demo_porous_drainage() -> None:
    """Run 2-D porous medium primary drainage benchmark."""
    # SC model with random cylinders
    cfg_sc = PorousDrainageConfig(
        nx=200,
        ny=80,
        geometry="random_cylinders",
        n_cylinders=15,
        r_min=4.0,
        r_max=8.0,
        seed=42,
        model="sc",
        G_12=0.9,
        G_ads_water=0.3,
        G_ads_gas=0.0,
        tau_water=1.0,
        tau_gas=1.0,
        rho_water=0.7,
        rho_gas=0.3,
        n_steps=4000,
        output_interval=1000,
        output_root=OUTPUT_ROOT / "drainage_sc",
        overwrite=True,
    )
    result_sc = run_porous_drainage(cfg_sc)
    bt_sc = result_sc["breakthrough_step"]
    bt_sc_str = f"step {bt_sc}" if bt_sc else "not reached"
    print(
        f"\n[SC drainage] Porosity={result_sc['porosity']:.3f}  "
        f"Breakthrough={bt_sc_str}\n"
    )

    # CG model with tube array
    cfg_cg = PorousDrainageConfig(
        nx=200,
        ny=80,
        geometry="tube_array",
        n_tubes=3,
        tube_width=12,
        model="cg",
        G_12=0.9,
        tau_water=1.0,
        tau_gas=1.0,
        rho_water=0.7,
        rho_gas=0.3,
        n_steps=4000,
        output_interval=1000,
        output_root=OUTPUT_ROOT / "drainage_cg",
        overwrite=True,
    )
    result_cg = run_porous_drainage(cfg_cg)
    bt_cg = result_cg["breakthrough_step"]
    bt_cg_str = f"step {bt_cg}" if bt_cg else "not reached"
    print(
        f"\n[CG drainage] Porosity={result_cg['porosity']:.3f}  "
        f"Breakthrough={bt_cg_str}\n"
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Porous-media gas-water displacement benchmarks")
    print("=" * 60)
    demo_laplace()
    demo_capillary_invasion()
    demo_two_phase_poiseuille()
    demo_porous_drainage()
    print("Done.  Results in:", OUTPUT_ROOT.resolve())
