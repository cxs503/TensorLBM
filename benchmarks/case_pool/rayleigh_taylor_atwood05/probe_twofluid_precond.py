"""Staged start-up probe for the finite-density-ratio RT arm (case CASE-RT-INSTABILITY-ATWOOD).

Background
----------
``run_twofluid.py`` (mission m21799) started the SCMP RT box with a hand-picked
``tanh(width=2)`` interface and gravity on at step 1.  Both runs aborted with
``stop_reason = "mach"`` at ``steps_run = 5`` (``Ma`` 0.26 / 0.30 against the 0.20
guard), so the anchor quantity -- the *spike* growth rate at ``A = 1/2`` -- was
never measured (``docs/CASE-RT-TWOFLUID.md`` sec. 4, honest INCOMPLETE).

What this script measures (all numbers written to ``artifacts_twofluid_precond/probe.json``)
---------------------------------------------------------------------------------------------
P1  ``startup``    legacy start-up: Mach of the first sampled steps.  Isolates the
                   force impulse ``|F|/rho`` of a non-equilibrium interface profile.
P2  ``floor``      spurious-current floor of a *flat* relaxed interface (``g = 0``)
                   as a function of ``tau``: the sensitivity limit of any front
                   estimator built on this model.
P3  ``null``       null-dynamics control: seeded mode, gravity **off**, staged
                   start-up.  The measured front drift is the estimator noise
                   floor -- a mode that grows at less than this cannot be resolved.
P4  ``seed``       seed read-back: the gradient-estimator amplitude at ``t = 0``
                   and its evolution over the first steps.  ``a(t=0)`` must read
                   the planned ``a0`` for the gate to mean anything.

Usage::

    python3 probe_twofluid_precond.py            # ~3 min on 6 threads
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

torch.set_num_threads(int(os.environ.get("RT_THREADS", "6")))

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (str(REPO / "src"), str(REPO)):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

from tensorlbm.rayleigh_taylor_twofluid import (  # noqa: E402
    SCMPRTConfig,
    column_front_state,
    precondition_scmp_rt,
    run_scmp_rt,
    scmp_coexistence,
)

SC_G = 2.8            # coupling whose measured A is 0.505 (calibration.json)
LAM = 64
LAYERS = 4.0
RHO_H = 1.172169369791218
RHO_L = 0.38598072297323593
OUT = HERE / "artifacts_twofluid_precond" / "probe.json"


def _cfg(tau: float, gravity: float = 2.0e-4, **kw) -> SCMPRTConfig:
    return SCMPRTConfig(lam=LAM, layers=LAYERS, sc_g=SC_G, gravity=gravity,
                        tau_relax=tau, a0_over_lam=0.02, interface_width=2.0, **kw)


def p1_startup() -> dict:
    """Legacy start-up (tanh seed + gravity from step 1): the m21799 failure."""
    cfg = _cfg(1.0, steps=20, sample_every=1, ma_limit=1e9)
    res = run_scmp_rt(cfg, rho_high=RHO_H, rho_low=RHO_L)
    return {
        "note": "legacy path (precondition_steps=0): tanh(width=2) + gravity at step 1",
        "ma_first_samples": [(h["step"], round(h["ma"], 4)) for h in res["history"][:5]],
        "ma_peak": res["ma_peak"],
        "ma_limit_default": 0.20,
        "aborts_under_default_guard": res["ma_peak"] > 0.20,
    }


def p2_floor() -> list[dict]:
    """Steady spurious-current floor of a flat relaxed interface vs tau."""
    out = []
    for tau in (0.8, 1.0, 1.2, 1.5):
        cfg = _cfg(tau)
        t0 = time.time()
        _, info = precondition_scmp_rt(cfg, rho_high=RHO_H, rho_low=RHO_L,
                                       relax_steps=600, sample_every=100)
        tr = info["relax_trace"]
        out.append({
            "tau": tau,
            "nu": cfg.nu,
            "ma_trace": [(r["step"], round(r["ma_all"], 4)) for r in tr],
            "ma_floor": tr[-1]["ma_all"],
            "ma_floor_iface": tr[-1]["ma_iface"],
            "u_spurious": tr[-1]["ma_all"] * math.sqrt(1.0 / 3.0),
            "rho_min_relaxed": tr[-1]["rho_min"],
            "rho_max_relaxed": tr[-1]["rho_max"],
            "mass_drift_pct": tr[-1]["mass_drift_pct"],
            "wall_time_s": round(time.time() - t0, 1),
        })
    return out


def p3_null() -> dict:
    """Null-dynamics control: seeded mode, gravity off, staged start-up.

    Uses the gradient estimator (not crossings) to measure the noise floor.
    """
    cfg = _cfg(1.0, steps=600, sample_every=100, ma_limit=1e9)
    res = run_scmp_rt(cfg, rho_high=RHO_H, rho_low=RHO_L, precondition_steps=600,
                      ramp_steps=0, ma_guard="bulk", gravity_scale=0.0,
                      front_estimator="gradient")
    hs = [(h["t"], round(h.get("amplitude_lam", float("nan")), 5),
           round(h.get("spike_q10_lam", float("nan")), 4),
           round(h.get("bubble_q90_lam", float("nan")), 4))
          for h in res["history"]]
    return {
        "note": "gravity_scale = 0: any measured front motion is spurious",
        "estimator": "gradient",
        "history (t, amp, spike_q10, bubble_q90)": hs,
        "spurious_ma_floor": res["precondition"]["spurious_ma_floor"],
        "stop_reason": res["stop_reason"],
        "front_drift_lam": max(abs(r[1]) for r in hs if not math.isnan(r[1])) if hs else float("nan"),
    }


def p4_seed() -> dict:
    """Seed read-back: does the gradient-estimator amplitude see the planned a0?

    Uses ``front_estimator="gradient"`` (not the contaminated ``crossings``).
    The gradient estimator is calibrated in
    ``tests/test_rayleigh_taylor_twofluid.py::test_gradient_estimator_reads_back_a_known_amplitude``
    to read back a0 with <5% error for a0 >= 0.02 lam.
    """
    cfg = _cfg(1.0, steps=300, sample_every=25, ma_limit=1e9)
    res = run_scmp_rt(cfg, rho_high=RHO_H, rho_low=RHO_L, precondition_steps=600,
                      ramp_steps=200, ma_guard="bulk",
                      front_estimator="gradient")
    rows = [(h["t"], round(h.get("amplitude_lam", float("nan")), 5),
             round(h.get("spike_q10_lam", float("nan")), 4),
             round(h.get("ma_bulk", float("nan")), 4))
            for h in res["history"]]
    a0 = cfg.a0_over_lam
    valid_rows = [r for r in rows if not math.isnan(r[1])]
    readback = valid_rows[0][1] if valid_rows else float("nan")
    return {
        "note": "front_estimator=gradient (not the contaminated crossings)",
        "planned_a0_over_lam": a0,
        "readback_a0_over_lam": readback,
        "readback_ratio": (readback / a0) if readback != float("nan") else float("nan"),
        "history (t, amp, spike_q10, ma_bulk)": rows,
        "sigma_amp": res["sigma_amp"],
        "sigma_theory_visc": res["sigma_theory_visc"],
        "err_sigma_pct": res["err_sigma_pct"],
        "ma_peak_bulk": res["ma_peak_bulk"],
        "stop_reason": res["stop_reason"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(OUT))
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    co = scmp_coexistence(SC_G, steps=1500, ny=48, tau=1.0)
    payload: dict = {
        "sc_g": SC_G,
        "lam": LAM,
        "layers": LAYERS,
        "coexistence": co.as_dict(),
        "atwood": co.atwood,
    }
    stages = {
        "startup": p1_startup,
        "floor": p2_floor,
        "null": p3_null,
        "seed": p4_seed,
    }
    only = {s for s in args.only.split(",") if s}
    for name, fn in stages.items():
        if only and name not in only:
            continue
        print(f"[{name}] ...", flush=True)
        payload[name] = fn()
        print(f"[{name}] done", flush=True)

    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.json}")
    if "floor" in payload:
        print("\nspurious-current floor vs tau (flat interface, g=0):")
        for row in payload["floor"]:
            print(f"  tau={row['tau']:.2f} nu={row['nu']:.3f} ma_floor={row['ma_floor']:.4f} "
                  f"u_spurious={row['u_spurious']:.4f}")
    if "seed" in payload:
        s = payload["seed"]
        est = s.get("estimator", "crossings")
        print(f"\nseed read-back (estimator={est}): "
              f"planned a0/lam={s['planned_a0_over_lam']:.4f} "
              f"read back {s['readback_a0_over_lam']:.5f} "
              f"(ratio {s['readback_ratio']:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
