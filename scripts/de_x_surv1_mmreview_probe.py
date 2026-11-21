"""DE-X-SURV1-MMREVIEW: first-hand probe of multiphase families in TensorLBM.

Purpose
-------
The survey card (BB/X/DE-X-SURV1-MMREVIEW.md) must quote *measured* numbers for
"reachable density ratio" and "parasitic-current level" per multiphase family.
This script measures them with the repository's own production kernels, on CPU,
at small size so that it stays inside the compute budget.

Nothing here modifies the solver: every probe calls the shipped benchmark /
verification entry points.

Outputs a JSON blob to ``artifacts/de_x_surv1_mmreview/probe_results.json``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

# The cluster checkout is not pip-installed: make the src-layout package
# importable without relying on PYTHONPATH being set by the caller.
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUT = _REPO_ROOT / "artifacts" / "de_x_surv1_mmreview"
OUT.mkdir(parents=True, exist_ok=True)

results: dict[str, object] = {
    "torch": torch.__version__,
    "dtype": "float64",
    "device": "cpu",
    "repo_root": str(_REPO_ROOT),
}


def _timed(name, fn):
    t0 = time.time()
    try:
        out = fn()
    except Exception as exc:  # noqa: BLE001 - probe must report, not crash
        out = {"error": f"{type(exc).__name__}: {exc}"}
    out["wall_time_s"] = round(time.time() - t0, 2)
    results[name] = out
    print(f"[probe] {name}: {json.dumps(out, default=str)[:400]}")
    return out


# ── A. Cahn--Hilliard / free-energy phase field (lee_lin vs swift) ─────────
def _ch_planar(convention: str):
    from tensorlbm.phasefield.ch_benchmarks import run_planar_interface

    r = run_planar_interface(xi=3.0, steps=1000, geq_convention=convention)
    hist = r.get("history", [])
    tail = hist[-1] if hist else {}
    return {
        "checks": r.get("checks"),
        "max_physical_speed": r.get("max_physical_speed", tail.get("max_physical_speed")),
        "phase_mass_drift": r.get("phase_mass_drift", tail.get("phase_mass_drift")),
        "sigma_error": r.get("sigma_error"),
        "config": r.get("config"),
        "history_tail": hist[-3:],
        "raw_keys": sorted(r.keys()),
    }


# ── B. Two-component Shan--Chen (MCMP): coexistence density ratio ──────────
def _mcmp_coex(g12: float):
    from tensorlbm.mcmp_verification import estimate_coexistence

    r = estimate_coexistence(G_12=g12, n=64, geometry="planar", steps=3000, device="cpu")
    return {
        "G_12": g12,
        "density_ratio": r.get("density_ratio"),
        "rho1_liq": r.get("rho1_liq"),
        "rho2_liq": r.get("rho2_liq"),
        "rho1_gas": r.get("rho1_gas"),
        "rho2_gas": r.get("rho2_gas"),
        "flat_interface_dp": r.get("flat_interface_dp"),
        "u_max": r.get("u_max"),
        "converged": r.get("converged"),
    }


# ── C. MCMP droplet at rest: parasitic current level ───────────────────────
def _mcmp_spurious(g12: float):
    from tensorlbm.mcmp_verification import spurious_current_benchmark

    r = spurious_current_benchmark(
        [{"n": 64, "G_12": g12, "radius": 12.0}], steps=3000, device="cpu"
    )
    return r


# ── D. Color gradient + SCMC static droplet (2-D) ──────────────────────────
def _static_droplet():
    from tensorlbm.multiphase_benchmarks import StaticDropletConfig, run_static_droplet

    cfg = StaticDropletConfig(
        nx=64,
        ny=64,
        radii=(10.0,),
        n_steps=2000,
        output_interval=1000,
        output_root=OUT,
        run_name="mmreview_static_droplet",
        device="cpu",
        overwrite=True,
    )
    r = run_static_droplet(cfg)
    return r


# ── E. Single-component pseudopotential EOS: analytic density ratio ───────
def _pseudo_ratio():
    from tensorlbm.multiphase import psi_carnahan_starling, psi_exp
    from tensorlbm.pseudopotential_edm3d import maxwell_coexistence

    out = {}
    for g in (0.5, 1.0, 1.5, 2.0, 3.0):
        cs = maxwell_coexistence(g, 1.0, psi_carnahan_starling)
        ex = maxwell_coexistence(g, 1.0, psi_exp)
        out[f"G={g}"] = {
            "CS_ratio": cs.get("density_ratio"),
            "CS_converged": cs.get("converged"),
            "CS_rho_v": cs.get("rho_v"),
            "CS_rho_l": cs.get("rho_l"),
            "exp_ratio": ex.get("density_ratio"),
            "exp_converged": ex.get("converged"),
        }
    return out


def main() -> None:
    _timed("ch_planar_lee_lin", lambda: _ch_planar("lee_lin"))
    _timed("ch_planar_swift", lambda: _ch_planar("swift"))
    _timed("mcmp_coex_G-0.9", lambda: _mcmp_coex(-0.9))
    _timed("mcmp_coex_G-1.5", lambda: _mcmp_coex(-1.5))
    _timed("mcmp_coex_G-2.5", lambda: _mcmp_coex(-2.5))
    _timed("mcmp_spurious_G-1.5", lambda: _mcmp_spurious(-1.5))
    _timed("pseudo_coexistence_ratio", _pseudo_ratio)
    _timed("static_droplet_2d", _static_droplet)

    p = OUT / "probe_results.json"
    p.write_text(json.dumps(results, indent=1, default=str))
    print(f"[probe] wrote {p}")


if __name__ == "__main__":
    main()
