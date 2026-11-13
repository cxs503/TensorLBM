#!/usr/bin/env python3
"""XVAL: TensorLBM Shan-Chen MCMP (barycentric velocity + Guo forcing) vs
``lssfau/lbmpy`` ``shan_chen/test_shan_chen_two_component.py``.

Anchor: ``XVAL-LBM-SHAN-CHEN-MCMP-BARYCENTRIC-GUO``

Reference (zero hand-copy)
--------------------------
The OSS reference is the 64-point density profile the OSS test asserts after
1000 steps of a two-component Shan-Chen run on a 64x64 periodic box.  It is
fetched (source text hashed, numbers parsed) by ``fetch_oss_ref.py`` into
``artifacts/xval_lbm_shan_chen_mcmp/oss_ref_shan_chen_two_component.json``.

Arms (one factor changed at a time)
-----------------------------------
``lib_barycentric_guo_exp``  primary: the library path
                            :func:`tensorlbm.multiphase.collide_sc_two_component`
                            with ``u_eq="mixture"`` (barycentric equilibrium
                            velocity), ``force_scheme="guo"`` (Guo 2002 source
                            term with ``1 - 1/(2 tau)``) and the exponential
                            pseudopotential ``psi_exp``.
``oss_faithful``            driver-level replication of the OSS expression set on
                            top of this library's primitives (``equilibrium``,
                            ``guo_force_term``, ``stream``): identical to the
                            primary arm except that the *barycentric* velocity
                            also enters the Guo source term (the OSS convention).
``lib_self_guo_exp``        equilibrium velocity = per-component velocity
                            (``u_eq="self"``) instead of barycentric.
``lib_barycentric_shift_exp`` historical ``tau F/rho`` velocity shift instead of
                            the Guo source term.
``lib_barycentric_guo_edm`` exact-difference (well-balanced) split of the pair
                            force instead of the standard stencil force.
``lib_barycentric_guo_linear``  linear pseudopotential ``psi(rho)=rho`` (the
                            library default) instead of ``psi_exp``.

Sensitivity studies (each is a *predeclared* one-dimensional sweep)
-------------------------------------------------------------------
* step count: 500, 900, 1000, 1100, 1500 (time alignment / transient stage);
* grid: N = 64 (reference grid) and N = 128 (block-averaged back to 64 points);
* relaxation time: tau = 0.8, 1.0, 1.2 (reference is tied to tau = 1).

Everything is CPU, single-threaded, deterministic (no RNG anywhere).

Usage::

    OMP_NUM_THREADS=1 PYTHONPATH=src python \
        xval/XVAL-LBM-SHAN-CHEN-MCMP-BARYCENTRIC-GUO/run_xval.py [--stage all|arms|sensitivity]
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm import __version__ as TENSORLBM_VERSION  # noqa: E402
from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.multiphase import (  # noqa: E402
    collide_sc_two_component,
    guo_force_term,
    psi_exp,
    sc_two_component_force,
)
from tensorlbm.solver import stream  # noqa: E402

ART = ROOT / "artifacts" / "xval_lbm_shan_chen_mcmp"
REF_FILE = ART / "oss_ref_shan_chen_two_component.json"
RESULTS_FILE = ART / "results.json"
PROFILES_FILE = ART / "profiles.json"

THRESHOLD_REL_L2 = 0.15  # owner accuracy standard: < 15 % is acceptable


# ---------------------------------------------------------------------------
# Reference + initial condition
# ---------------------------------------------------------------------------


def load_ref() -> dict[str, Any]:
    if not REF_FILE.exists():
        raise SystemExit(
            f"missing reference artifact {REF_FILE}; run fetch_oss_ref.py first"
        )
    return json.loads(REF_FILE.read_text())


def init_state(n: int, dtype: torch.dtype = torch.float64) -> tuple[torch.Tensor, torch.Tensor]:
    """OSS initial condition: step in x, rho_a = 0.1 (left) / 0.9 (right), rho_b mirrored."""
    x = torch.arange(n).view(1, n)
    half = n // 2
    rho_a = torch.where(x < half, 0.1, 0.9).expand(n, n).contiguous().to(dtype)
    rho_b = torch.where(x < half, 0.9, 0.1).expand(n, n).contiguous().to(dtype)
    zero = torch.zeros_like(rho_a)
    return equilibrium(rho_a, zero, zero), equilibrium(rho_b, zero, zero)


# ---------------------------------------------------------------------------
# Time steppers
# ---------------------------------------------------------------------------


def step_library(
    f1: torch.Tensor,
    f2: torch.Tensor,
    *,
    G_12: float = -6.0,
    tau: float = 1.0,
    u_eq: str = "mixture",
    force_scheme: str = "guo",
    sc_form: str = "standard",
    use_psi: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One step of the library path (force recomputed from the current state)."""
    f1, f2 = collide_sc_two_component(
        f1,
        f2,
        G_12=G_12,
        tau1=tau,
        tau2=tau,
        u_eq=u_eq,
        force_scheme=force_scheme,
        sc_form=sc_form,
        psi=psi_exp if use_psi else None,
    )
    return stream(f1), stream(f2)


def step_oss_faithful(
    f1: torch.Tensor,
    f2: torch.Tensor,
    *,
    G_12: float = -6.0,
    tau: float = 1.0,
    variant: str = "barycentric_guo",
) -> tuple[torch.Tensor, torch.Tensor]:
    """OSS-/lbmpy-convention variants re-assembled from this library's primitives.

    The OSS test fixes the *barycentric* velocity as ``velocity_input`` and the
    Guo force model, but the published lbmpy Guo documentation says the
    equilibrium velocity is additionally shifted by ``F/(2 rho)`` while the
    source term is evaluated at the (unshifted) input velocity.  Reading the two
    statements together leaves three defensible velocity couplings, so each is
    run as its own arm instead of guessing:

    ``barycentric_guo``  equilibrium on the barycentric velocity, Guo source also
                         at the barycentric velocity (the literal reading of the
                         OSS code: u_bary built once, used for both);
    ``lbmpy_doc``        equilibrium on ``u_bary + F/(2 rho)`` (lbmpy's
                         ``equilibrium_velocity_shift``), Guo source at ``u_bary``;
    ``self_corrected``   equilibrium *and* Guo source at the component-wise
                         force-corrected velocity ``u_s + F_s/(2 rho_s)``
                         (identical to the library's ``u_eq="self", guo``).

    All three share the OSS force law (``psi_exp``, per-neighbour sum) and the
    OSS SRT collision, so they differ from each other only in the velocity
    coupling -- one factor at a time.
    """
    rho1, ux1, uy1 = macroscopic(f1)
    rho2, ux2, uy2 = macroscopic(f2)
    Fx1, Fy1, Fx2, Fy2 = sc_two_component_force(rho1, rho2, G_12, psi=psi_exp)

    r1 = torch.clamp(rho1, min=1e-12)
    r2 = torch.clamp(rho2, min=1e-12)
    tot = r1 + r2
    ubx = (rho1 * ux1 + rho2 * ux2) / tot
    uby = (rho1 * uy1 + rho2 * uy2) / tot

    if variant == "barycentric_guo":
        e1 = equilibrium(rho1, ubx, uby)
        e2 = equilibrium(rho2, ubx, uby)
        g1 = (ubx, uby)
        g2 = (ubx, uby)
    elif variant == "lbmpy_doc":
        e1 = equilibrium(rho1, ubx + 0.5 * Fx1 / r1, uby + 0.5 * Fy1 / r1)
        e2 = equilibrium(rho2, ubx + 0.5 * Fx2 / r2, uby + 0.5 * Fy2 / r2)
        g1 = (ubx, uby)
        g2 = (ubx, uby)
    elif variant == "self_corrected":
        s1x, s1y = ux1 + 0.5 * Fx1 / r1, uy1 + 0.5 * Fy1 / r1
        s2x, s2y = ux2 + 0.5 * Fx2 / r2, uy2 + 0.5 * Fy2 / r2
        e1 = equilibrium(rho1, s1x, s1y)
        e2 = equilibrium(rho2, s2x, s2y)
        g1 = (s1x, s1y)
        g2 = (s2x, s2y)
    else:
        raise ValueError(f"unknown variant {variant!r}")

    f1o = f1 - (f1 - e1) / tau + guo_force_term(rho1, g1[0], g1[1], Fx1, Fy1, tau)
    f2o = f2 - (f2 - e2) / tau + guo_force_term(rho2, g2[0], g2[1], Fx2, Fy2, tau)
    return stream(f1o), stream(f2o)


def run(
    stepper: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    *,
    n: int = 64,
    steps: int = 1000,
    dtype: torch.dtype = torch.float64,
    **kwargs: Any,
) -> dict[str, Any]:
    """Step the model and return the fields plus divergence diagnostics.

    The run is *not* aborted on blow-up: the first step at which a non-finite
    value appears (or the density leaves the physical range) is recorded, both
    components are then held at their last finite state, and the caller sees
    ``diverged_at_step``.  That is what makes an unstable arm report a *step*
    instead of a bare ``nan``.
    """
    f1, f2 = init_state(n, dtype=dtype)
    t0 = time.perf_counter()
    diverged_at: int | None = None
    min_rho = math.inf
    max_u = 0.0
    for step in range(1, steps + 1):
        f1, f2 = stepper(f1, f2, **kwargs)
        if step % 10 == 0 or step == 1:
            rho1, ux1, uy1 = macroscopic(f1)
            rho2, ux2, uy2 = macroscopic(f2)
            if not (torch.isfinite(rho1).all() and torch.isfinite(rho2).all()):
                diverged_at = step
                break
            min_rho = min(min_rho, float(rho1.min()), float(rho2.min()))
            max_u = max(max_u, float(torch.sqrt(ux1**2 + uy1**2).max()),
                        float(torch.sqrt(ux2**2 + uy2**2).max()))
    wall = time.perf_counter() - t0
    rho1, ux1, uy1 = macroscopic(f1)
    rho2, ux2, uy2 = macroscopic(f2)
    return {
        "n": n,
        "steps": steps,
        "dtype": str(dtype).replace("torch.", ""),
        "steps_completed": steps if diverged_at is None else diverged_at,
        "diverged_at_step": diverged_at,
        "min_density": None if not math.isfinite(min_rho) else round(min_rho, 6),
        "max_speed": round(max_u, 6),
        "wall_seconds": round(wall, 3),
        "profile_a": rho1[0].tolist(),
        "profile_b": rho2[0].tolist(),
        "field_a": rho1.tolist(),
        "field_b": rho2.tolist(),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _crossing(profile: np.ndarray, level: float, rising: bool) -> float | None:
    """Linear-interpolated index where ``profile`` crosses ``level``."""
    idx = range(len(profile) - 1) if rising else range(len(profile) - 1, 0, -1)
    for i in idx:
        j = i + 1 if rising else i - 1
        a, b = profile[i], profile[j]
        if (rising and a <= level <= b) or (not rising and a <= level <= b):
            if b == a:
                return float(i)
            t = (level - a) / (b - a)
            return float(i + t * (j - i))
    return None


def _width_10_90(profile: np.ndarray, lo: float, hi: float) -> float | None:
    """Interface thickness measured between the 10 % and 90 % levels of one interface."""
    l10 = lo + 0.1 * (hi - lo)
    l90 = lo + 0.9 * (hi - lo)
    x10 = _crossing(profile, l10, rising=True)
    x90 = _crossing(profile, l90, rising=True)
    if x10 is None or x90 is None:
        return None
    return abs(x90 - x10)


def profile_metrics(got: np.ndarray, ref: np.ndarray) -> dict[str, Any]:
    got = np.asarray(got, dtype=float)
    ref = np.asarray(ref, dtype=float)
    delta = got - ref
    n_pl = len(ref)
    plateau = list(range(4, 14)) + list(range(n_pl - 14, n_pl - 4))
    lo = float(ref[plateau].min())
    hi = float(ref[plateau].max())
    return {
        "rel_l2": float(np.linalg.norm(delta) / np.linalg.norm(ref)),
        "max_abs_err": float(np.abs(delta).max()),
        "rel_linf": float(np.abs(delta).max() / np.abs(ref).max()),
        "rms_abs_err": float(np.sqrt(np.mean(delta**2))),
        "plateau_mean_got": float(got[plateau].mean()),
        "plateau_mean_ref": float(ref[plateau].mean()),
        "plateau_rel_err": float(
            abs(got[plateau].mean() - ref[plateau].mean()) / abs(ref[plateau].mean())
        ),
        "interface_crossings_got": [
            _crossing(got, 0.5 * (lo + hi), True),
            _crossing(got, 0.5 * (lo + hi), False),
        ],
        "interface_crossings_ref": [
            _crossing(ref, 0.5 * (lo + hi), True),
            _crossing(ref, 0.5 * (lo + hi), False),
        ],
        "width_10_90_got": _width_10_90(got, lo, hi),
        "width_10_90_ref": _width_10_90(ref, lo, hi),
    }


def evaluate(run_out: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    ref_a = np.asarray(ref["ref_a"])
    ref_b = np.asarray(ref["ref_b"])
    m_a = profile_metrics(run_out["profile_a"], ref_a)
    m_b = profile_metrics(run_out["profile_b"], ref_b)
    worst = max(m_a["rel_l2"], m_b["rel_l2"])
    finite = math.isfinite(worst)
    return {
        "component_a": m_a,
        "component_b": m_b,
        "worst_rel_l2": worst if finite else None,
        "passes_15pct": bool(finite and worst < THRESHOLD_REL_L2),
        "n": run_out["n"],
        "steps": run_out["steps"],
        "dtype": run_out.get("dtype"),
        "diverged_at_step": run_out.get("diverged_at_step"),
        "min_density": run_out.get("min_density"),
        "max_speed": run_out.get("max_speed"),
        "wall_seconds": run_out["wall_seconds"],
    }


def block_average(profile: list[float], factor: int) -> list[float]:
    arr = np.asarray(profile, dtype=float)
    return arr.reshape(-1, factor).mean(axis=1).tolist()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


ARM_SPECS: dict[str, dict[str, Any]] = {
    "lib_barycentric_guo_exp": {"stepper": "library", "kwargs": {}},
    "oss_literal_barycentric": {"stepper": "oss", "kwargs": {"variant": "barycentric_guo"}},
    "oss_lbmpy_doc_shift": {"stepper": "oss", "kwargs": {"variant": "lbmpy_doc"}},
    "oss_self_corrected": {"stepper": "oss", "kwargs": {"variant": "self_corrected"}},
    "lib_self_guo_exp": {"stepper": "library", "kwargs": {"u_eq": "self"}},
    "lib_barycentric_shift_exp": {"stepper": "library", "kwargs": {"force_scheme": "shift"}},
    "lib_barycentric_guo_edm": {"stepper": "library", "kwargs": {"sc_form": "edm"}},
    "lib_barycentric_guo_linear": {"stepper": "library", "kwargs": {"use_psi": False}},
}

STEP_SWEEP = [500, 900, 1000, 1100, 1500]
TAU_SWEEP = [0.8, 1.0, 1.2]
GRID_SWEEP = [64, 128]


def _stepper(name: str) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    return step_library if name == "library" else step_oss_faithful


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "arms", "sensitivity"])
    ap.add_argument("--arms", default="", help="comma-separated arm subset (default: all)")
    ap.add_argument("--sens", default="", help="comma-separated sensitivity subset of steps,grid,dtype,tau")
    args = ap.parse_args(argv)

    torch.set_num_threads(1)
    ref = load_ref()
    ART.mkdir(parents=True, exist_ok=True)

    out: dict[str, Any] = {}
    if RESULTS_FILE.exists():
        out = json.loads(RESULTS_FILE.read_text())
    out.setdefault("arms", {})
    out.setdefault("sensitivity", {})
    out.update({
        "schema": "xval-result/1.0",
        "anchor": "XVAL-LBM-SHAN-CHEN-MCMP-BARYCENTRIC-GUO",
        "threshold_rel_l2": THRESHOLD_REL_L2,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "tensorlbm": TENSORLBM_VERSION,
            "platform": platform.platform(),
            "threads": torch.get_num_threads(),
        },
        "reference": {
            "file": str(REF_FILE.relative_to(ROOT)),
            "repo": ref["source"]["repo"],
            "commit": ref["source"]["commit"],
            "path": ref["source"]["path"],
            "source_sha256": ref["source"]["source_sha256"],
            "ref_a_sha256": ref["ref_a_sha256"],
        },
    })

    arm_sel = [a for a in args.arms.split(",") if a] or list(ARM_SPECS)
    sens_sel = [t for t in args.sens.split(",") if t] or ["steps", "grid", "dtype", "tau"]

    if args.stage in ("all", "arms"):
        for name, spec in ARM_SPECS.items():
            if name not in arm_sel:
                continue
            res = run(_stepper(spec["stepper"]), n=ref["parameters"]["n"],
                      steps=ref["parameters"]["steps"], **spec["kwargs"])
            ev = evaluate(res, ref)
            ev["spec"] = spec
            out["arms"][name] = ev
            out["arms"][name]["profiles"] = {
                "a": res["profile_a"],
                "b": res["profile_b"],
            }
            print(
                f"[arm] {name:30s} relL2(a)={ev['component_a']['rel_l2']:.6f} "
                f"maxabs={ev['component_a']['max_abs_err']:.3e} "
                f"diverged={ev['diverged_at_step']} minrho={ev['min_density']} "
                f"pass={ev['passes_15pct']} ({res['wall_seconds']}s)",
                flush=True,
            )

    if args.stage in ("all", "sensitivity"):
        n = ref["parameters"]["n"]
        if "steps" in sens_sel:
             # (1) step-count sweep on the primary arm
          sweep = {}
          for steps in STEP_SWEEP:
              res = run(step_library, n=n, steps=steps)
              sweep[str(steps)] = evaluate(res, ref)
              print(
                  f"[steps] {steps:5d} relL2(a)={sweep[str(steps)]['component_a']['rel_l2']:.6f} "
                  f"maxabs={sweep[str(steps)]['component_a']['max_abs_err']:.3e}",
                  flush=True,
              )
          out["sensitivity"]["steps"] = sweep

        if "grid" in sens_sel:
             # (2) grid sweep: N = 128 block-averaged back onto the 64 reference points
          grid = {}
          for nn in GRID_SWEEP:
              res = run(step_library, n=nn, steps=ref["parameters"]["steps"])
              if nn == n:
                  ev = evaluate(res, ref)
              else:
                  factor = nn // n
                  coarse = {
                      "profile_a": block_average(res["profile_a"], factor),
                      "profile_b": block_average(res["profile_b"], factor),
                      "n": n,
                      "steps": res["steps"],
                      "wall_seconds": res["wall_seconds"],
                  }
                  ev = evaluate(coarse, ref)
                  ev["block_average_factor"] = factor
              grid[str(nn)] = ev
              print(
                  f"[grid] N={nn:4d} relL2(a)={ev['component_a']['rel_l2']:.6f} "
                  f"maxabs={ev['component_a']['max_abs_err']:.3e} ({res['wall_seconds']}s)",
                  flush=True,
              )
          out["sensitivity"]["grid"] = grid

        if "dtype" in sens_sel:
             # (3) arithmetic-precision sweep (the OSS reference is float64 text)
          prec = {}
          for dt in (torch.float32, torch.float64):
              res = run(step_library, n=n, steps=ref["parameters"]["steps"], dtype=dt)
              prec[str(dt).replace("torch.", "")] = evaluate(res, ref)
              print(
                  f"[dtype] {str(dt):15s} relL2(a)={prec[str(dt).replace('torch.', '')]['component_a']['rel_l2']:.3e} "
                  f"maxabs={prec[str(dt).replace('torch.', '')]['component_a']['max_abs_err']:.3e}",
                  flush=True,
              )
          out["sensitivity"]["dtype"] = prec

        if "tau" in sens_sel:
             # (4) relaxation-time sweep (the reference is tied to tau = 1)
          tau_sweep = {}
          for tau in TAU_SWEEP:
              res = run(step_library, n=n, steps=ref["parameters"]["steps"], tau=tau)
              tau_sweep[f"{tau:.2f}"] = evaluate(res, ref)
              print(
                  f"[tau] {tau:.2f} relL2(a)={tau_sweep[f'{tau:.2f}']['component_a']['rel_l2']:.6f}",
                  flush=True,
              )
          out["sensitivity"]["tau"] = tau_sweep

    RESULTS_FILE.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {RESULTS_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())