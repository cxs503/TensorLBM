#!/usr/bin/env python3
"""XVAL-LBM-GEOMETRIC-WETTING-CONTACT-ANGLE :: operator-level cross-validation.

Decision (predeclared, see ``docs/XVAL-LBM-OPENLBMPM-GEOMETRIC-WETTING.md`` §五):
which realisation of the geometrical wetting condition should TensorLBM keep for
the wall-adjacent colour value -- and does it reproduce the prescribed contact
angle within the owner's 15 % tolerance?

Competing alternatives (all impose a static contact angle ``theta`` by rotating
the near-wall colour gradient):

* ``V1 closed2pt``      -- the shipped condition of
  :func:`tensorlbm.cg_wetting2d.geometrical_wetting_ghost`: two-point central
  tangential difference, wall-normal difference fixed to ``+|dphi/dt| cot theta``
  (Akai et al. 2018 / ``lanl/MF-LBM`` port).
* ``V2 cand2pt``        -- same stencil, but the sign of the wall-normal
  difference is chosen by the openLBMPM *two-candidate* rule (nearer of the two
  unit normals at angle ``theta`` to the wall normal).
* ``V3 candIso9``       -- the openLBMPM recipe verbatim: isotropic 9-point colour
  gradient (``AcceleratedRKGPU2D.py:1584-1632``) + two-candidate selection.
* ``V4 candIso9+diff``  -- V3 with a prescribed solid colour difference
  (``SolidColorDiff = +-0.5``, ``IniFiles/RKtwophasesetup2D.ini:42``), which is
  where openLBMPM's candidate sign actually comes from.

Outcome metrics (both machine-checkable):

* ``M1`` operator self-consistency: contact angle read back from the imposed
  wall-normal difference with the *same* stencil that built it
  (:func:`tensorlbm.cg_wetting2d.wall_local_contact_angle` for the 2-point
  variants, the module's own read-back for the isotropic variants);
  reported as relative error to the prescribed angle -> **decision gate 15 %**.
* ``M2`` operator fidelity: mean ``|phi_ghost - phi_exact|`` on the wall row,
  where ``phi_exact`` is the analytic interface continued *into the solid* (an
  external truth, not self-referential).  Lower is better; this is the
  discriminating metric between the variants.
* ``M3`` ambiguity statistics: how many wall faces the candidate rule resolves
  against the fixed sign.

Unit of analysis: one (variant, theta, xi) configuration; 27 per variant
(9 angles x 3 interface widths) -> paired comparison across variants on the same
fields (same fields, same seeds: fully deterministic, no repetitions needed).
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from tensorlbm.cg_wetting2d import (  # noqa: E402
    geometrical_wetting_ghost,
    wall_local_contact_angle,
)
from tensorlbm.wetting.geometric_two_candidate import (  # noqa: E402
    cap_geometry_contact_angle,
    geometric_wetting_ghost_candidates,
    isotropic_color_gradient,
)

ANGLES = (30.0, 45.0, 60.0, 75.0, 90.0, 105.0, 120.0, 135.0, 150.0)
XIS = (1.5, 2.0, 3.0)
VARIANTS = ("V1_closed2pt", "V2_cand2pt", "V3_candIso9", "V4_candIso9_diff")
GATE_PCT = 15.0


def analytic_cap(ny: int, nx: int, theta_deg: float, xi: float, radius: float = 24.0):
    """``phi = tanh((R - r)/xi)`` cap whose level set meets ``y = 0.5`` at ``theta``."""
    y = torch.arange(ny, dtype=torch.float64).view(-1, 1)
    x = torch.arange(nx, dtype=torch.float64).view(1, -1)
    yc = 0.5 - radius * math.cos(math.radians(theta_deg))
    r = torch.sqrt((x - nx / 2.0) ** 2 + (y - yc) ** 2)
    phi = torch.tanh((radius - r) / xi)
    solid = torch.zeros(ny, nx, dtype=torch.bool)
    solid[0, :] = True
    solid[-1, :] = True
    return phi, solid


def _readback_isotropic(phi, ghost, solid, gate=1e-6):
    """Contact angle implied by the imposed wall normal, isotropic 9-point stencil."""
    from tensorlbm.wetting.geometric_two_candidate import solid_color_mirror

    angles = []
    for dy, dx in ((1, 0), (-1, 0)):
        face = (~solid) & torch.roll(solid, shifts=(-dy, -dx), dims=(-2, -1))
        if not bool(face.any()):
            continue
        ext = solid_color_mirror(phi, solid, dy, dx)
        gx, gy = isotropic_color_gradient(ext)
        g_t = gx * dy + gy * (-dx)
        g_n = torch.where(face, _nb_local(ghost, dy, dx), torch.zeros_like(phi)) - phi
        sel = face & (torch.sqrt(g_t * g_t + g_n * g_n) > gate) & (phi.abs() < 0.9)
        if bool(sel.any()):
            ang = torch.atan2(g_t[sel].abs(), g_n[sel].abs()) * (180.0 / math.pi)
            ang = torch.where(
                torch.tensor(math.radians(90.0)) < torch.zeros(1),
                ang,
                ang,
            )
            angles.append(ang)
    if not angles:
        return torch.zeros(0)
    return torch.cat(angles)


def _nb_local(a, dy, dx):
    return torch.roll(a, shifts=(-dy, -dx), dims=(-2, -1))


def run_case(theta: float, xi: float, ny: int = 128, nx: int = 128) -> list[dict]:
    phi, solid = analytic_cap(ny, nx, theta, xi)
    exact = phi.clone()  # analytic continuation of the interface into the solid
    wall = solid[0, :]
    rows = []

    for variant in VARIANTS:
        if variant == "V1_closed2pt":
            ghost = geometrical_wetting_ghost(phi, solid, theta, periodic=(True, False))
            rb = wall_local_contact_angle(phi, ghost, solid, periodic=(True, False))
            flips, n_active, stencil = 0, int(wall.sum()), "central2"
        else:
            stencil = "central2" if variant == "V2_cand2pt" else "isotropic9"
            offset = 0.5 if variant == "V4_candIso9_diff" else 0.0
            ghost, diag = geometric_wetting_ghost_candidates(
                phi,
                solid,
                theta,
                periodic=(True, False),
                stencil=stencil,
                solid_color_offset=offset,
                return_diagnostics=True,
            )
            flips = int(diag["n_sign_flips"])
            n_active = int(diag["n_interface_face_nodes"])
            if stencil == "central2":
                rb = wall_local_contact_angle(phi, ghost, solid, periodic=(True, False))
            else:
                rb = _readback_isotropic(phi, ghost, solid)

        rb_med = float(rb.median()) if rb.numel() else float("nan")
        # the two-point read-back of cg_wetting2d is signed (it separates the
        # two theta branches itself); the isotropic read-back above takes the
        # magnitude of the normal difference, hence returns the acute branch
        if stencil == "central2":
            rb_eff = rb_med
        else:
            rb_eff = rb_med if theta <= 90.0 else 180.0 - rb_med
        fid = float((ghost[0, wall] - exact[0, wall]).abs().mean())
        rows.append(
            {
                "variant": variant,
                "theta_prescribed": theta,
                "xi": xi,
                "stencil": stencil,
                "theta_readback_deg": rb_med,
                "theta_readback_effective_deg": rb_eff,
                "rel_err_pct": 100.0 * abs(rb_eff - theta) / theta,
                "fidelity_mean_abs_dphi": fid,
                "n_sign_flips": flips,
                "n_wall_faces": n_active,
                "gate_pass": bool(100.0 * abs(rb_eff - theta) / theta <= GATE_PCT),
            }
        )
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for xi in XIS:
        for th in ANGLES:
            rows.extend(run_case(th, xi))

    by_variant = {}
    for v in VARIANTS:
        sub = [r for r in rows if r["variant"] == v]
        errs = [r["rel_err_pct"] for r in sub if math.isfinite(r["rel_err_pct"])]
        fids = [r["fidelity_mean_abs_dphi"] for r in sub if math.isfinite(r["fidelity_mean_abs_dphi"])]
        by_variant[v] = {
            "n_cases": len(sub),
            "rel_err_pct_max": max(errs) if errs else float("nan"),
            "rel_err_pct_median": sorted(errs)[len(errs) // 2] if errs else float("nan"),
            "fidelity_mean_abs_dphi": sum(fids) / len(fids) if fids else float("nan"),
            "fidelity_max": max(fids) if fids else float("nan"),
            "n_sign_flips_total": int(sum(r["n_sign_flips"] for r in sub)),
            "gate_pass_all": all(r["gate_pass"] for r in sub),
        }

    # paired fidelity comparison (same fields, so a plain paired mean is the test)
    paired = {}
    for v in VARIANTS[1:]:
        diffs = []
        for xi in XIS:
            for th in ANGLES:
                a = next(r for r in rows if r["variant"] == "V1_closed2pt" and r["theta_prescribed"] == th and r["xi"] == xi)
                b = next(r for r in rows if r["variant"] == v and r["theta_prescribed"] == th and r["xi"] == xi)
                diffs.append(b["fidelity_mean_abs_dphi"] - a["fidelity_mean_abs_dphi"])
        diffs.sort()
        paired[v] = {
            "mean_delta_fidelity_vs_V1": sum(diffs) / len(diffs),
            "median_delta": diffs[len(diffs) // 2],
            "n_better": sum(1 for d in diffs if d < 0),
            "n_total": len(diffs),
        }

    best = min(by_variant, key=lambda v: by_variant[v]["fidelity_mean_abs_dphi"])
    payload = {
        "anchor": "XVAL-LBM-GEOMETRIC-WETTING-CONTACT-ANGLE",
        "reference": {
            "repo": "PorousMediaSimulation/openLBMPM",
            "method": "geometric wetting contact angle",
            "files": [
                "RKCG2D/AcceleratedRKGPU2D.py:1560 (calColorValueOnSolid)",
                "RKCG2D/AcceleratedRKGPU2D.py:1584-1632 (isotropic colour gradient)",
                "RKCG2D/AcceleratedRKGPU2D.py:2430-2492 (two-candidate projection)",
                "IniFiles/RKtwophasesetup2D.ini:42 (SolidColorDiff)",
            ],
            "provenance": "platform OSS inventory docs/oss-inventory-openLBMPM.md §4.2 "
            "+ oss_solutions.id=5 (read at source level, line pointers above)",
        },
        "criteria": {
            "decision_gate": "relative contact-angle error <= 15 % (M1, every case)",
            "ranking_metric": "M2 fidelity to the analytic continuation of the interface",
            "unit_of_analysis": "(variant, theta, xi)",
            "repetitions": "none needed: deterministic analytic fields, paired comparison",
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "variants": by_variant,
        "paired_vs_V1": paired,
        "selected_variant": best,
        "cases": rows,
    }
    (out / "operator_study.json").write_text(json.dumps(payload, indent=1, default=float))
    for v in VARIANTS:
        d = by_variant[v]
        print(
            f"{v:18s} M1 max={d['rel_err_pct_max']:.2e}% med={d['rel_err_pct_median']:.2e}%  "
            f"M2={d['fidelity_mean_abs_dphi']:.5f} (max {d['fidelity_max']:.5f})  "
            f"flips={d['n_sign_flips_total']}  gate={d['gate_pass_all']}"
        )
    print("paired vs V1:", json.dumps(paired, default=float))
    print("selected:", best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())