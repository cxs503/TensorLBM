#!/usr/bin/env python3
"""Integration caller: the production stepper drives the model registry.

This is not a new benchmark -- it is the *integration* proof for
:mod:`tensorlbm.model_matrix`:

1. A plain channel flow is advanced with
   :func:`tensorlbm.lbm_step_correct.lbm_step_correct`, the shared verified
   stepper used by the 3-D solvers, passing the collision operator **as a
   model-registry key string** (``"d3q19/bgk"``, ``"d3q19/mrt"``,
   ``"d3q19/trt"``) instead of a raw function.
2. For every key the run is compared with the legacy call that passes the raw
   function: the two must be **bit-identical**, otherwise the registry binding
   is not faithful and the whole matrix would certify a different operator
   than the solver runs.
3. Physical invariants of the stepper are asserted per key (finite field, mass
   conservation at round-off level, flow develops).

Artifacts: ``artifacts/model_matrix_ci/integration.json`` (+ printed table).

Usage::

    PYTHONPATH=src python scripts/model_matrix_integration.py \
        --out artifacts/model_matrix_ci/integration.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d  # noqa: E402
from tensorlbm.lbm_step_correct import lbm_step_correct  # noqa: E402
from tensorlbm.model_matrix.registry import get_model, resolve_collide_fn  # noqa: E402

MODELS = ("d3q19/bgk", "d3q19/mrt", "d3q19/trt")


def _setup(nx: int, ny: int, nz: int, u_in: float):
    solid = torch.zeros((nz, ny, nx), dtype=torch.bool)
    solid[:, 0, :] = True
    solid[:, -1, :] = True
    wall = torch.ones((nz, ny, nx), dtype=torch.bool)
    del wall  # all faces periodic in this integration run: no inlet/outlet BC
    def far_field_fn(f: torch.Tensor, *args, **kwargs) -> torch.Tensor:  # noqa: ARG001
        """No-op far-field BC: the integration run is fully periodic."""
        return f
    rho0 = torch.ones((nz, ny, nx))
    ux0 = torch.full((nz, ny, nx), u_in)
    ux0[solid] = 0.0
    zero = torch.zeros_like(ux0)
    f = equilibrium3d(rho0, ux0, zero, zero)
    return solid, far_field_fn, f


def run_case(model: str, *, steps: int, nx: int, ny: int, nz: int, tau: float, u_in: float) -> dict:
    spec = get_model(model)
    solid, far_field_fn, f_key = _setup(nx, ny, nz, u_in)
    _, _, f_raw = _setup(nx, ny, nz, u_in)
    m0 = float(macroscopic3d(f_key)[0].sum())

    for step in range(steps):
        f_key = lbm_step_correct(f_key, model, tau, solid, u_in, far_field_fn, step=step)
        f_raw = lbm_step_correct(
            f_raw, resolve_collide_fn(model, tau), tau, solid, u_in, far_field_fn, step=step
        )
    rho, ux, uy, uz = macroscopic3d(f_key)
    m1 = float(rho.sum())
    umax = float(torch.sqrt(ux**2 + uy**2 + uz**2).max())
    identical = bool(torch.equal(f_key, f_raw))
    finite = bool(torch.isfinite(f_key).all())
    mass_drift = abs(m1 - m0) / m0
    profile = ux[0, ny // 2, :].tolist()
    return {
        "model": model,
        "implementation": spec.source,
        "tau_arg": spec.tau_arg,
        "params": spec.params,
        "steps": steps,
        "grid": [nz, ny, nx],
        "tau": tau,
        "u_in": u_in,
        "mass_drift_rel": mass_drift,
        "u_max": umax,
        "finite": finite,
        "bit_identical_to_legacy_raw_fn": identical,
        "centerline_ux": profile,
        "pass": bool(finite and identical and mass_drift < 1e-5 and umax > 0.5 * u_in),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/model_matrix_ci/integration.json")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--nx", type=int, default=24)
    ap.add_argument("--ny", type=int, default=12)
    ap.add_argument("--nz", type=int, default=1)
    ap.add_argument("--tau", type=float, default=0.6)
    ap.add_argument("--u-in", type=float, default=0.05)
    args = ap.parse_args(argv)

    rows = [
        run_case(
            m, steps=args.steps, nx=args.nx, ny=args.ny, nz=args.nz, tau=args.tau, u_in=args.u_in
        )
        for m in MODELS
    ]
    payload = {
        "caller": "tensorlbm.lbm_step_correct.lbm_step_correct",
        "trigger": "collide_fn passed as a model_matrix registry key (str)",
        "torch": torch.__version__,
        "config": {
            "steps": args.steps,
            "grid": [args.nz, args.ny, args.nx],
            "tau": args.tau,
            "u_in": args.u_in,
        },
        "results": rows,
        "verdict": "PASS" if all(r["pass"] for r in rows) else "FAIL",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    for r in rows:
        print(
            f"[integration] {r['model']:12s} bit_identical={r['bit_identical_to_legacy_raw_fn']} "
            f"mass_drift={r['mass_drift_rel']:.2e} u_max={r['u_max']:.4f} pass={r['pass']}"
        )
    print(f"[integration] verdict={payload['verdict']} -> {out}")
    return 0 if payload["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())