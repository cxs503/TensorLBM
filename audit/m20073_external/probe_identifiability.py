"""Audit probe: is the (cy, r) -> (drag, cy_wake) observable map injective?

Re-executes the forward model of m20073's examples/inverse_shape_multiobs.py on a
fixed (cy, r) grid (no optimizer involved) and reports the two observables, so the
identifiability of the inverse problem can be judged from raw numbers instead of
the optimizer's trajectory.

Usage (from repo root):  PYTHONPATH=src python audit/m20073_external/probe_identifiability.py
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def load_module():
    spec = importlib.util.spec_from_file_location(
        "inverse_shape_multiobs", HERE / "inverse_shape_multiobs.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    mod = load_module()
    dtype, device = torch.float64, torch.device("cpu")

    def obs(cy: float, r: float) -> tuple[float, float]:
        with torch.no_grad():
            drag, cy_wake, _ = mod.run_forward(
                cy, r, mod.U_IN, mod.STEPS, mod.PROBE_START, mod.EPS_SOFT, dtype, device
            )
        return float(drag), float(cy_wake)

    drag_star, cyw_star = obs(mod.CY_TRUE, mod.R_TRUE)
    print(f"reference (truth cy={mod.CY_TRUE}, r={mod.R_TRUE}): "
          f"drag*={drag_star:.6e}  cy_wake*={cyw_star:.6f}")

    rows = []
    for dcy in (-1.0, -0.5, 0.0, 0.5, 1.0):
        for dr in (-0.3, 0.0, 0.3):
            cy = mod.CY_TRUE + dcy
            r = mod.R_TRUE + dr
            drag, cyw = obs(cy, r)
            rel_drag = (drag - drag_star) / abs(drag_star)
            rel_cyw = (cyw - cyw_star) / abs(cyw_star)
            loss = rel_drag ** 2 + rel_cyw ** 2
            rows.append({
                "cy": cy, "r": r,
                "drag": drag, "cy_wake": cyw,
                "rel_drag": rel_drag, "rel_cy_wake": rel_cyw,
                "loss": loss,
            })
            print(f"  cy={cy:5.2f} r={r:5.2f}  drag={drag:.6e} (rel {rel_drag:+.3e})  "
                  f"cy_wake={cyw:.4f} (rel {rel_cyw:+.3e})  loss={loss:.3e}")

    # reported 20-iteration optimizer end point (independently reproduced)
    cy_opt, r_opt = 9.3124, 2.4171
    drag_o, cyw_o = obs(cy_opt, r_opt)
    rel_drag_o = (drag_o - drag_star) / abs(drag_star)
    rel_cyw_o = (cyw_o - cyw_star) / abs(cyw_star)
    loss_o = rel_drag_o ** 2 + rel_cyw_o ** 2
    print(f"\noptimizer end point (iter 20, reproduced): cy={cy_opt} r={r_opt}")
    print(f"  drag={drag_o:.6e} (rel {rel_drag_o:+.3e})  cy_wake={cyw_o:.4f} "
          f"(rel {rel_cyw_o:+.3e})  loss={loss_o:.3e}")
    print(f"  parametric error: cy {abs(cy_opt-mod.CY_TRUE)/mod.CY_TRUE*100:.2f}%  "
          f"r {abs(r_opt-mod.R_TRUE)/mod.R_TRUE*100:.2f}%")

    out = {
        "reference": {"cy": mod.CY_TRUE, "r": mod.R_TRUE,
                      "drag": drag_star, "cy_wake": cyw_star},
        "grid": rows,
        "optimizer_endpoint": {"cy": cy_opt, "r": r_opt, "drag": drag_o,
                              "cy_wake": cyw_o, "loss": loss_o,
                              "rel_err_cy": abs(cy_opt - mod.CY_TRUE) / mod.CY_TRUE,
                              "rel_err_r": abs(r_opt - mod.R_TRUE) / mod.R_TRUE},
        "config": {"NZ": mod.NZ, "NY": mod.NY, "NX": mod.NX, "steps": mod.STEPS,
                   "eps_soft": mod.EPS_SOFT, "x_probe": mod.X_PROBE,
                   "tau": mod.TAU, "u_in": mod.U_IN, "dtype": "float64"},
    }
    outdir = HERE / "probe"
    outdir.mkdir(exist_ok=True)
    (outdir / "identifiability.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {outdir / 'identifiability.json'}")


if __name__ == "__main__":
    main()