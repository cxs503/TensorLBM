"""Validation case: Lid-Driven Cavity (Ghia et al. 1982).

This script runs D2Q9 LBM simulations for Re=100 and Re=1000 and compares
the centreline velocity profiles against the reference data from:
    Ghia, U., Ghia, K. N., & Shin, C. T. (1982). High-Re solutions for
    incompressible flow using the Navier-Stokes equations and a multigrid
    method. Journal of Computational Physics, 48(3), 387-411.

Usage
-----
    PYTHONPATH=src python validation/cavity_ghia/run_cavity_ghia.py

Outputs are written to ``outputs/validation/cavity_ghia/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

from tensorlbm.lid_driven_cavity import (
    LidDrivenCavityConfig,
    run_lid_driven_cavity,
    GHIA_RE100,
    GHIA_RE1000,
    compare_ghia,
)
from tensorlbm.d2q9 import macroscopic
from tensorlbm.logging_config import configure_logging, logger

matplotlib.use("Agg")


def run_single_case(
    re: float,
    nx: int,
    u_lid: float,
    n_steps: int,
    device: str,
    output_root: Path,
) -> tuple[Path, dict]:
    """Run a single lid-driven cavity case and return run_dir and config."""
    config = LidDrivenCavityConfig(
        nx=nx,
        u_lid=u_lid,
        re=re,
        n_steps=n_steps,
        output_interval=n_steps,  # Only save final state
        output_root=output_root,
        run_name=f"re{int(re)}_nx{nx}_ulid{u_lid:.3f}",
        seed=0,
        device=device,
        overwrite=True,
        use_compile=False,
    )
    config.validate()
    run_dir = run_lid_driven_cavity(config)
    
    return run_dir, config


def generate_comparison_plots(
    run_dir_re100: Path,
    run_dir_re1000: Path | None,
    output_dir: Path,
) -> list[dict]:
    """Generate comparison plots between LBM and Ghia reference."""
    
    # Load CSV data for Re=100
    csv_100 = np.genfromtxt(run_dir_re100 / "ghia_comparison.csv", 
                            delimiter=',', skip_header=1)
    
    # Extract data
    y_100 = csv_100[:, 0]
    u_100 = csv_100[:, 1]
    x_100 = csv_100[:, 2]
    v_100 = csv_100[:, 3]
    
    # Ghia reference data
    ghia_y = np.array(GHIA_RE100["y"])
    ghia_u = np.array(GHIA_RE100["u"])
    ghia_x = np.array(GHIA_RE100["x"])
    ghia_v = np.array(GHIA_RE100["v"])
    
    figures = []
    
    # Plot 1: Vertical centreline u-velocity for Re=100
    fig1, ax1 = plt.subplots(1, 1, figsize=(6, 5))
    
    ax1.plot(ghia_u, ghia_y, 'ko-', markersize=5, label='Ghia et al. (1982)', linewidth=1.5)
    ax1.plot(u_100, y_100, 'b-', linewidth=2, label='LBM D2Q9 (nx=33)')
    ax1.set_xlabel('u/u_lid', fontsize=12)
    ax1.set_ylabel('y/H', fontsize=12)
    ax1.set_title('Re=100, Vertical Centreline (x=0.5H)', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-0.5, 1.1])
    
    plt.tight_layout()
    fig1.savefig(output_dir / "cavity_ghia_re100_vertical_u.png", dpi=150, bbox_inches='tight')
    plt.close(fig1)
    
    figures.append({
        "name": "cavity_ghia_re100_vertical_u",
        "caption": "Vertical centreline u-velocity comparison: LBM D2Q9 vs Ghia et al. (1982) for Re=100.",
    })
    
    # Plot 2: Horizontal centreline v-velocity for Re=100
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 5))
    
    ax2.plot(ghia_x, ghia_v, 'ko-', markersize=5, label='Ghia et al. (1982)', linewidth=1.5)
    ax2.plot(x_100, v_100, 'b-', linewidth=2, label='LBM D2Q9 (nx=33)')
    ax2.set_xlabel('x/W', fontsize=12)
    ax2.set_ylabel('v/u_lid', fontsize=12)
    ax2.set_title('Re=100, Horizontal Centreline (y=0.5H)', fontsize=12)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    
    plt.tight_layout()
    fig2.savefig(output_dir / "cavity_ghia_re100_horizontal_v.png", dpi=150, bbox_inches='tight')
    plt.close(fig2)
    
    figures.append({
        "name": "cavity_ghia_re100_horizontal_v",
        "caption": "Horizontal centreline v-velocity comparison: LBM D2Q9 vs Ghia et al. (1982) for Re=100.",
    })
    
    # Plot for Re=1000 if available
    if run_dir_re1000 is not None and (run_dir_re1000 / "ghia_comparison.csv").exists():
        csv_1000 = np.genfromtxt(run_dir_re1000 / "ghia_comparison.csv", 
                                delimiter=',', skip_header=1)
        
        y_1000 = csv_1000[:, 0]
        u_1000 = csv_1000[:, 1]
        x_1000 = csv_1000[:, 2]
        v_1000 = csv_1000[:, 3]
        
        ghia_y_1000 = np.array(GHIA_RE1000["y"])
        ghia_u_1000 = np.array(GHIA_RE1000["u"])
        ghia_x_1000 = np.array(GHIA_RE1000["x"])
        ghia_v_1000 = np.array(GHIA_RE1000["v"])
        
        # Plot 3: Vertical centreline u-velocity for Re=1000
        fig3, ax3 = plt.subplots(1, 1, figsize=(6, 5))
        
        ax3.plot(ghia_u_1000, ghia_y_1000, 'ko-', markersize=5, label='Ghia et al. (1982)', linewidth=1.5)
        ax3.plot(u_1000, y_1000, 'r-', linewidth=2, label='LBM D2Q9')
        ax3.set_xlabel('u/u_lid', fontsize=12)
        ax3.set_ylabel('y/H', fontsize=12)
        ax3.set_title('Re=1000, Vertical Centreline (x=0.5H)', fontsize=12)
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim([-0.5, 1.1])
        
        plt.tight_layout()
        fig3.savefig(output_dir / "cavity_ghia_re1000_vertical_u.png", dpi=150, bbox_inches='tight')
        plt.close(fig3)
        
        figures.append({
            "name": "cavity_ghia_re1000_vertical_u",
            "caption": "Vertical centreline u-velocity comparison: LBM D2Q9 vs Ghia et al. (1982) for Re=1000.",
        })
    
    return figures


def main() -> None:
    configure_logging()
    
    output_root = Path("outputs/validation/cavity_ghia")
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Run both cases
    logger.info("=" * 70)
    logger.info("Lid-Driven Cavity Validation: Ghia et al. (1982)")
    logger.info("=" * 70)
    
    # Re=100 case (stable and accurate)
    nx_100 = 33
    u_lid_100 = 0.1
    n_steps_100 = 2000
    
    logger.info("\nRunning Re=100 case (nx=%d, u_lid=%.3f, steps=%d)...", 
                nx_100, u_lid_100, n_steps_100)
    run_dir_100, config_100 = run_single_case(
        re=100, nx=nx_100, u_lid=u_lid_100, n_steps=n_steps_100,
        device='cpu', output_root=output_root
    )
    
    # Re=1000 case - use lower u_lid for stability
    # For Re=1000, nx=65, u_lid=0.05: tau = 3*(0.05*65/1000) + 0.5 = 0.50975
    # This is still close to 0.5, so we use MRT collision
    nx_1000 = 65
    u_lid_1000 = 0.05
    n_steps_1000 = 5000
    
    logger.info("\nRunning Re=1000 case (nx=%d, u_lid=%.3f, steps=%d)...", 
                nx_1000, u_lid_1000, n_steps_1000)
    try:
        run_dir_1000, config_1000 = run_single_case(
            re=1000, nx=nx_1000, u_lid=u_lid_1000, n_steps=n_steps_1000,
            device='cpu', output_root=output_root
        )
        re1000_success = True
    except Exception as e:
        logger.warning("Re=1000 case failed: %s", e)
        run_dir_1000 = None
        re1000_success = False
    
    # Load final fields and compute L2 errors
    logger.info("\n" + "=" * 70)
    logger.info("Computing L2 errors vs Ghia reference")
    logger.info("=" * 70)
    
    # Load metadata to get Ghia errors computed by the solver
    meta_100 = json.loads((run_dir_100 / "run_metadata.json").read_text())
    errors_100 = meta_100.get("ghia_errors", {})
    
    errors_1000 = {}
    if re1000_success and run_dir_1000 is not None:
        meta_1000 = json.loads((run_dir_1000 / "run_metadata.json").read_text())
        errors_1000 = meta_1000.get("ghia_errors", {})
    
    # Generate comparison plots
    logger.info("\nGenerating comparison plots...")
    figures = generate_comparison_plots(run_dir_100, run_dir_1000 if re1000_success else None, output_root)
    
    # Create error table
    error_table = {
        "re_100": {
            "rmse_u": errors_100.get("rmse_u", 0.0),
            "rmse_v": errors_100.get("rmse_v", 0.0),
            "criterion": "<=5%",
            "pass": errors_100.get("rmse_u", 1.0) <= 0.05,
        },
    }
    
    if re1000_success and not np.isnan(errors_1000.get("rmse_u", float('nan'))):
        error_table["re_1000"] = {
            "rmse_u": errors_1000.get("rmse_u", 0.0),
            "rmse_v": errors_1000.get("rmse_v", 0.0),
            "criterion": "<=8%",
            "pass": errors_1000.get("rmse_u", 1.0) <= 0.08,
        }
    else:
        error_table["re_1000"] = {
            "rmse_u": float('nan'),
            "rmse_v": float('nan'),
            "criterion": "<=8%",
            "pass": False,
            "note": "Simulation diverged due to low tau (close to 0.5 stability limit)",
        }
    
    # Save validation results
    validation_results = {
        "config": {
            "re100": {"nx": nx_100, "u_lid": u_lid_100, "n_steps": n_steps_100},
            "re1000": {"nx": nx_1000, "u_lid": u_lid_1000, "n_steps": n_steps_1000},
            "device": "cpu",
        },
        "errors": error_table,
        "figures": figures,
        "run_dirs": {
            "re100": str(run_dir_100),
            "re1000": str(run_dir_1000) if run_dir_1000 else None,
        },
    }
    
    validation_path = output_root / "validation_results.json"
    validation_path.write_text(json.dumps(validation_results, indent=2, default=str))
    logger.info("\nValidation results saved to: %s", validation_path)
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 70)
    logger.info("Re=100:  RMSE(u) = %.4f (%.2f%%)  [criterion: <=5%%]  %s",
                error_table["re_100"]["rmse_u"],
                error_table["re_100"]["rmse_u"] * 100,
                "PASS ✓" if error_table["re_100"]["pass"] else "FAIL ✗")
    logger.info("Re=100:  RMSE(v) = %.4f (%.2f%%)",
                error_table["re_100"]["rmse_v"],
                error_table["re_100"]["rmse_v"] * 100)
    
    if not np.isnan(error_table["re_1000"]["rmse_u"]):
        logger.info("Re=1000: RMSE(u) = %.4f (%.2f%%)  [criterion: <=8%%]  %s",
                    error_table["re_1000"]["rmse_u"],
                    error_table["re_1000"]["rmse_u"] * 100,
                    "PASS ✓" if error_table["re_1000"]["pass"] else "FAIL ✗")
        logger.info("Re=1000: RMSE(v) = %.4f (%.2f%%)",
                    error_table["re_1000"]["rmse_v"],
                    error_table["re_1000"]["rmse_v"] * 100)
    else:
        logger.info("Re=1000: FAILED - Simulation diverged (tau too close to 0.5)")
        logger.info("         Note: High-Re LBM requires special treatment (finer grid,")
        logger.info("         lower Mach number, or enhanced stability methods)")
    
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
