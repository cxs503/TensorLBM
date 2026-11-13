"""Lid-driven cavity example — structured version.

Replaces the monolithic ``examples/lid_driven_cavity.py`` script with a
registered example that can be run:

- Programmatically: ``registry.run('lid_driven_cavity', n=65, re=100)``
- Via unified CLI: ``python -m tensorlbm.examples.runner lid_driven_cavity --n 65``
- As a standalone script (backward compat): ``python examples/lid_driven_cavity.py``

Validates against Ghia et al. (1982) reference data.

Note: LBM stability requires ``tau > 0.5``.  For Re=100, use ``n >= 65``
or reduce ``u_lid``.  The defaults below are chosen for stability.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from ..verification.lid_driven_cavity import (
    CavityConfig,
    CavitySolver,
    get_ghia_reference,
)
from ..verification.lid_driven_cavity.reference_data import interpolate_ghia
from .base import Example, ExampleResult
from . import registry


class LidDrivenCavityExample(Example):
    """Lid-driven cavity benchmark (D2Q9 LBM, Ghia 1982 validation)."""

    @property
    def name(self) -> str:
        return "lid_driven_cavity"

    @property
    def description(self) -> str:
        return "2D lid-driven cavity, D2Q9 LBM, validated vs Ghia et al. (1982)"

    def configure(self) -> Dict[str, Any]:
        # Defaults chosen for stability: n=65, re=100 → tau≈0.69
        return {
            "n": 65,
            "u_lid": 0.1,
            "re": 100.0,
            "max_steps": 5000,
            "output_root": "outputs",
            "run_name": None,
            "device": "cpu",
            "overwrite": False,
            "use_rlbm": True,
            "ghia_tolerance": 0.05,
        }

    def run(self, params: Dict[str, Any]) -> ExampleResult:
        t0 = time.time()
        try:
            n = int(params["n"])
            u_lid = float(params["u_lid"])
            re = float(params["re"])
            max_steps = int(params["max_steps"])
            output_root = Path(params.get("output_root", "outputs"))
            run_name = params.get("run_name") or f"ldc_re{int(re)}_n{n}"
            device = params.get("device", "cpu")
            overwrite = bool(params.get("overwrite", False))
            use_rlbm = bool(params.get("use_rlbm", True))
            ghia_tol = float(params.get("ghia_tolerance", 0.05))

            cfg = CavityConfig(
                n=n,
                re=re,
                u_lid=u_lid,
                max_steps=max_steps,
                use_rlbm=use_rlbm,
                device=device,
            )
            solver = CavitySolver(cfg)

            # Run simulation with NaN guard
            for step in range(max_steps):
                solver.step()
                if step > 0 and step % 500 == 0:
                    rho, _, _ = solver._macroscopic(solver.f)
                    if torch.isnan(rho).any():
                        raise RuntimeError(f"Simulation diverged at step {step}")

            # Extract profiles
            y_norm, u_norm, x_norm, v_norm = solver.get_profiles()

            # Compare with Ghia
            metrics = {}
            ghia_passed = False
            try:
                ref = get_ghia_reference(re)
                u_ref = interpolate_ghia(ref, y_norm, axis='u')
                v_ref = interpolate_ghia(ref, x_norm, axis='v')

                l2_u = float(np.sqrt(np.mean((u_norm - u_ref) ** 2)))
                l2_v = float(np.sqrt(np.mean((v_norm - v_ref) ** 2)))
                max_u = float(np.max(np.abs(u_norm - u_ref)))
                max_v = float(np.max(np.abs(v_norm - v_ref)))

                metrics = {
                    "l2_u": round(l2_u, 6),
                    "l2_v": round(l2_v, 6),
                    "max_u": round(max_u, 6),
                    "max_v": round(max_v, 6),
                    "re": re,
                    "n": n,
                }
                ghia_passed = l2_u < ghia_tol and l2_v < ghia_tol
            except KeyError:
                metrics = {"re": re, "n": n, "ghia_available": False}

            # Save output
            run_dir = output_root / run_name
            if run_dir.exists() and overwrite:
                shutil.rmtree(run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)

            # Save profiles
            np.savetxt(run_dir / "u_profile.csv", np.column_stack([y_norm, u_norm]),
                       header="y,u", comments="", delimiter=",")
            np.savetxt(run_dir / "v_profile.csv", np.column_stack([x_norm, v_norm]),
                       header="x,v", comments="", delimiter=",")

            # Save metadata
            meta = {
                "example": self.name,
                "params": {k: v for k, v in params.items() if not callable(v)},
                "metrics": metrics,
                "ghia_passed": ghia_passed,
                "elapsed_s": round(time.time() - t0, 2),
            }
            (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

            elapsed = time.time() - t0
            return ExampleResult(
                name=self.name,
                params=params,
                elapsed_s=round(elapsed, 2),
                success=True,
                artifacts={"run_dir": str(run_dir)},
                metrics=metrics,
            )

        except Exception as e:
            elapsed = time.time() - t0
            return ExampleResult(
                name=self.name,
                params=params,
                elapsed_s=round(elapsed, 2),
                success=False,
                error=str(e),
            )


# Auto-register on import
registry.register(LidDrivenCavityExample())
