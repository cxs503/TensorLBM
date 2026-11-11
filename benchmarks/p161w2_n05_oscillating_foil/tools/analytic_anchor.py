#!/usr/bin/env python
"""Analytic anchor tables for the oscillating-foil verification face.

Produces (JSON on stdout or to ``--out``) every closed-form number used by the
acceptance criteria:

* Theodorsen function ``C(k) = F + iG`` with the exact limits ``C(0)=1``,
  ``C(inf)=1/2`` and a two-path (Hankel / series) cross-check;
* the pure-heave Garrick result against its independent algebraic form
  ``C_T = 4 pi k^2 (h0/c)^2 |C|^2``, ``C_P = 4 pi k^2 (h0/c)^2 F``,
  ``eta = |C|^2/F`` (derived in the report);
* the ``k -> 0`` quasi-steady limit ``C_T -> pi (h0/b)^2 k^2``;
* phasor vs time-quadrature agreement of the full Garrick coefficients;
* ``eta(k)`` curves for the three motion modes;
* the thrust-reversal line ``C_T(St) = 0`` -> ``St_crit(k)``.

Usage::

    python tools/analytic_anchor.py --out artifacts/analytic.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "src")
sys.path.insert(0, SRC)

from tensorlbm.flapping_propulsion import (  # noqa: E402
    FlappingKinematics,
    garrick_coefficients,
    theodorsen_C,
    theodorsen_C_bessel,
    thrust_reversal_heave,
)

U = 1.0
C_CHORD = 1.0
A_AXIS = -0.5


def theodorsen_table(ks):
    rows = []
    for k in ks:
        C = theodorsen_C(k)
        Cb = theodorsen_C_bessel(k)
        rows.append(
            {
                "k": k,
                "F": C.real,
                "G": C.imag,
                "absC": abs(C),
                "two_path_abs_diff": abs(C - Cb),
            }
        )
    return rows


def pure_heave_check(k_list, h0_over_c=(0.02, 0.10, 0.25)):
    """Independent algebraic form of the pure-heave Garrick coefficients."""
    rows = []
    for k in k_list:
        C = theodorsen_C(k)
        for h0c in h0_over_c:
            kin = FlappingKinematics(U=U, c=C_CHORD, h0=h0c * C_CHORD, alpha0=0.0, phi=0.0, a=A_AXIS)
            g = garrick_coefficients(kin, k)
            ct = 4.0 * math.pi * k**2 * h0c**2 * abs(C) ** 2
            cp = 4.0 * math.pi * k**2 * h0c**2 * C.real
            eta = abs(C) ** 2 / C.real
            rows.append(
                {
                    "k": k,
                    "h0_over_c": h0c,
                    "C_T": g.C_T,
                    "C_T_closed_form": ct,
                    "C_P": g.C_P,
                    "C_P_closed_form": cp,
                    "eta": g.eta,
                    "eta_closed_form": eta,
                    "rel_err_C_T": abs(g.C_T - ct) / abs(ct) if ct else float("nan"),
                    "rel_err_C_P": abs(g.C_P - cp) / abs(cp) if cp else float("nan"),
                }
            )
    return rows


def quasi_steady_check(ks=(1e-4, 1e-3, 1e-2), h0_over_c=(0.02, 0.10)):
    rows = []
    for k in ks:
        for h0c in h0_over_c:
            kin = FlappingKinematics(U=U, c=C_CHORD, h0=h0c * C_CHORD, alpha0=0.0, phi=0.0, a=A_AXIS)
            g = garrick_coefficients(kin, k)
            # C_T -> pi (h0/b)^2 k^2, with b = c/2
            lim = math.pi * (h0c * C_CHORD / (0.5 * C_CHORD)) ** 2 * k**2
            rows.append(
                {
                    "k": k,
                    "h0_over_c": h0c,
                    "C_T": g.C_T,
                    "C_T_quasi_steady": lim,
                    "rel_err": abs(g.C_T - lim) / lim,
                }
            )
    return rows


def quadrature_check(k_list, h0_over_c=0.25, alpha0_deg=10.0, phi_deg=90.0, n=720):
    rows = []
    for k in k_list:
        kin = FlappingKinematics(
            U=U,
            c=C_CHORD,
            h0=h0_over_c * C_CHORD,
            alpha0=math.radians(alpha0_deg),
            phi=math.radians(phi_deg),
            a=A_AXIS,
        )
        gp = garrick_coefficients(kin, k)
        gq = garrick_coefficients(kin, k, n_samples=n)
        rows.append(
            {
                "k": k,
                "C_T_phasor": gp.C_T,
                "C_T_quadrature": gq.C_T,
                "C_P_phasor": gp.C_P,
                "C_P_quadrature": gq.C_P,
                "abs_diff_C_T": abs(gp.C_T - gq.C_T),
                "abs_diff_C_P": abs(gp.C_P - gq.C_P),
            }
        )
    return rows


def eta_curves(ks, h0_over_c=0.25, alpha0_deg=10.0, phi_deg=90.0):
    out = {"k": list(ks), "heave": [], "pitch": [], "coupled": []}
    for k in ks:
        for key, h0, a0 in (
            ("heave", h0_over_c, 0.0),
            ("pitch", 0.0, alpha0_deg),
            ("coupled", h0_over_c, alpha0_deg),
        ):
            kin = FlappingKinematics(
                U=U,
                c=C_CHORD,
                h0=h0 * C_CHORD,
                alpha0=math.radians(a0),
                phi=math.radians(phi_deg),
                a=A_AXIS,
            )
            g = garrick_coefficients(kin, k)
            out[key].append(
                {
                    "k": k,
                    "C_T": g.C_T,
                    "C_P": g.C_P,
                    "eta": g.eta,
                    "St": kin.strouhal(k),
                }
            )
    return out


def reversal_line(ks, alpha0_deg=10.0, phi_deg=90.0, cd0_values=(0.0, 0.05)):
    out = {}
    for cd0 in cd0_values:
        out[f"{cd0:.2f}"] = [
            thrust_reversal_heave(
                k, math.radians(alpha0_deg), math.radians(phi_deg), A_AXIS, cd0=cd0
            )
            for k in ks
        ]
    return out


def build():
    ks = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
    payload = {
        "node": "p161w2-arb-n05",
        "pitch_axis_a": A_AXIS,
        "theodorsen": theodorsen_table(ks),
        "theodorsen_limits": {
            "C_zero_minus_one": abs(theodorsen_C(1e-9) - 1.0),
            "C_inf_minus_half": abs(theodorsen_C(1e5) - 0.5),
        },
        "pure_heave_closed_form": pure_heave_check(ks),
        "quasi_steady": quasi_steady_check(),
        "quadrature_vs_phasor": quadrature_check([0.5, 1.0, 2.0]),
        "eta_curves": eta_curves(ks),
        "thrust_reversal": reversal_line([0.25, 0.5, 1.0, 1.5, 2.0, 3.0]),
    }
    # summary numbers used by the criteria
    # the k -> 0 check is taken where the leading O(k) correction is <= 0.4 %;
    # the full table (k up to 1e-2) documents the linear-in-k approach
    qs = max(r["rel_err"] for r in payload["quasi_steady"] if r["k"] <= 1e-3)
    qs_k1e2 = max(r["rel_err"] for r in payload["quasi_steady"] if r["k"] >= 1e-2)
    ph = max(r["rel_err_C_T"] for r in payload["pure_heave_closed_form"])
    payload["summary"] = {
        "quasi_steady_rel_error": qs,
        "quasi_steady_rel_error_k1e2": qs_k1e2,
        "pure_heave_rel_error": ph,
        "C_at_zero_minus_one": payload["theodorsen_limits"]["C_zero_minus_one"],
        "C_at_inf_minus_half": payload["theodorsen_limits"]["C_inf_minus_half"],
        "quadrature_max_abs_diff_C_T": max(
            r["abs_diff_C_T"] for r in payload["quadrature_vs_phasor"]
        ),
    }
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    payload = build()
    text = json.dumps(payload, indent=1)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write(text)
    else:
        print(text)
    s = payload["summary"]
    print(
        f"quasi-steady rel err={s['quasi_steady_rel_error']:.3e}  "
        f"pure-heave closed form rel err={s['pure_heave_rel_error']:.3e}  "
        f"|C(0)-1|={s['C_at_zero_minus_one']:.3e}  |C(inf)-1/2|={s['C_at_inf_minus_half']:.3e}  "
        f"quadrature vs phasor={s['quadrature_max_abs_diff_C_T']:.3e}",
        flush=True,
    )
    for r in payload["thrust_reversal"]["0.05"]:
        print(
            f"  k={r['k']:.2f}  h0/c_crit={r['h0_over_c_crit']:.4f}  St_crit={r['St_crit']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())