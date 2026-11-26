#!/usr/bin/env python3
"""XVAL-LBM-MCMP-CROSS-FORCE-OPERATOR-XVAL-SYMBOL-SP -- four-arm cross-validation
of the two-component (MCMP) Shan-Chen cross-force operator.

Arms
----
S  symbol   : sympy derivation of the D2Q9 moment tensors, of the forward/backward
              gather identity, and of the *closed form* transfer function
              (stencil gain) of the discrete gradient, including its leading
              truncation term.
P  spectral : independent FFT reference on periodic pure modes; the measured
              lattice gain is compared both with the symbolic closed form
              (machine precision) and with the continuum value cs^2 (the
              error band that matters for real fields).
O  oss      : the operator transcription of the OSS code (lbmpy 2.0 test suite)
              against the library kernel, on identical fields.
F  physics  : two-component Shan-Chen simulation run to mechanical equilibrium
              and checked against *analytic* invariants -- mass conservation,
              Newton's third law of the cross force, the mixture equation of
              state p = cs^2 (rho1+rho2) - G cs^2 rho1 rho2 (uniform across a
              flat interface) and the discrete Boltzmann relation
              ln rho1 - G psi2 = const -- plus the operator error on the
              *measured* equilibrium interface (lattice vs spectral force).

All numbers are produced programmatically; nothing is transcribed by hand.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))), "src"))

from tensorlbm.d2q9 import C, W, equilibrium, macroscopic  # noqa: E402
from tensorlbm.multiphase import (  # noqa: E402
    psi_exp,
    psi_linear,
    sc_two_component_force,
)

CS2 = 1.0 / 3.0
DT = torch.float64
PSI = {"linear": psi_linear, "exp": psi_exp}


# ---------------------------------------------------------------------------
# arm S: symbolic
# ---------------------------------------------------------------------------

def arm_symbol() -> dict:
    """Exact D2Q9 algebra with sympy: moments, gather identity, transfer function."""
    import sympy as sp

    # D2Q9 weight / velocity table (identical to tensorlbm.d2q9 for verification)
    table = [(0, 0, sp.Rational(4, 9)), (1, 0, sp.Rational(1, 9)), (-1, 0, sp.Rational(1, 9)),
             (0, 1, sp.Rational(1, 9)), (0, -1, sp.Rational(1, 9)),
             (1, 1, sp.Rational(1, 36)), (-1, -1, sp.Rational(1, 36)),
             (1, -1, sp.Rational(1, 36)), (-1, 1, sp.Rational(1, 36))]
    # library table must agree (guards against a stale copy in this file)
    lib_c = np.asarray(C, dtype=float)
    lib_w = np.asarray(W, dtype=float)
    table_match = True
    for (cx, cy, w) in table:
        hit = [j for j in range(9)
               if lib_c[j, 0] == cx and lib_c[j, 1] == cy and abs(lib_w[j] - float(w)) < 1e-6]
        table_match &= len(hit) == 1

    cs2 = sp.Rational(1, 3)

    def M(rank):
        idx = [(a, b) for a in range(2) for b in range(rank)]
        out = {}
        for combo in _combs(rank):
            s = 0
            for (cx, cy, w) in table:
                c = (cx, cy)
                p = sp.Integer(1)
                for a in combo:
                    p *= c[a]
                s += w * p
            out[combo] = sp.nsimplify(s)
        return out

    def _combs(rank):
        from itertools import product
        return list(product(range(2), repeat=rank))

    M0 = sum(w for (_, _, w) in table)
    M2 = M(2)
    M3 = M(3)
    M4 = M(4)

    # second moment must be cs^2 * delta
    M2_ok = (M2[(0, 0)] == cs2 and M2[(1, 1)] == cs2 and M2[(0, 1)] == 0 and M2[(1, 0)] == 0)
    # third moment must vanish (this is why the stencil is a *second-order* gradient)
    M3_zero = all(v == 0 for v in M3.values())
    # fourth moment must be cs^4 (delta delta + 2 perms)
    cs4 = cs2 ** 2
    M4_ok = True
    for combo, v in M4.items():
        theory = sp.Integer(0)
        a, b, c_, d = combo
        pairs = [((a, b), (c_, d)), ((a, c_), (b, d)), ((a, d), (b, c_))]
        for (p1, p2) in pairs:
            theory += cs4 * (sp.Integer(1) if p1[0] == p1[1] else 0) * (sp.Integer(1) if p2[0] == p2[1] else 0)
        M4_ok &= (sp.nsimplify(v) == theory)

    # exact transfer function:  Ghat_alpha(k) = -sum_i w_i c_i,alpha exp(-i k.c_i)
    # (the "gradient operator" implicit in F = -G psi_a sum_i w_i c_i psi_b(x-c_i))
    kx, ky = sp.symbols("kx ky", real=True)
    Sx = sp.Integer(0)
    Sy = sp.Integer(0)
    for (cx, cy, w) in table:
        ph = sp.exp(-sp.I * (kx * cx + ky * cy))
        Sx += w * cx * ph
        Sy += w * cy * ph
    Sx = sp.simplify(Sx)
    # closed form we claim: Sx = -2i sin(kx) [1/9 + cos(ky)/18]
    Sx_claim = -2 * sp.I * sp.sin(kx) * (sp.Rational(1, 9) + sp.cos(ky) / 18)
    Sy_claim = -2 * sp.I * sp.sin(ky) * (sp.Rational(1, 9) + sp.cos(kx) / 18)
    Sx_ok = sp.simplify(sp.expand_complex(Sx - Sx_claim)) == 0
    Sy_ok = sp.simplify(sp.expand_complex(Sy - Sy_claim)) == 0

    # small-k expansion of the gain (numeric, from the closed form): ratio to i*k
    k = sp.symbols("k", positive=True)
    gain_x = sp.simplify(-Sx_claim.subs({kx: k, ky: 0}) / (sp.I * k))
    ser = sp.series(gain_x, k, 0, 6).removeO()
    # leading coefficients
    c0 = sp.simplify(ser.subs(k, 0))
    c2 = sp.simplify(sp.diff(ser, k, 2).subs(k, 0) / 2)
    # continuum value cs^2 and truncation coefficient -cs^2/2 per unit k^2
    trunc_ok = bool(c0 == cs2 and sp.simplify(c2 + cs2 / 6) == 0)

    return {
        "table_matches_library_d2q9": bool(table_match),
        "M0": str(M0),
        "M2": {str(k_): str(v) for k_, v in M2.items()},
        "M2_equals_cs2_delta": bool(M2_ok),
        "M3_all_zero": bool(M3_zero),
        "M4_equals_cs4_delta_delta": bool(M4_ok),
        "stencil_closed_form_x": str(Sx_claim),
        "stencil_closed_form_x_verified": bool(Sx_ok and Sy_ok),
        "gain_small_k_expansion": str(ser),
        "gain_leading_cs2": str(c0),
        "gain_leading_cs2_float": float(c0),
        "gain_k2_coefficient": str(c2),
        "gain_k2_coefficient_float": float(c2),
        "gain_equals_cs2_times_1_minus_k2_over_6": bool(trunc_ok),
        "cs2": float(cs2),
        # wavenumber at which the gain error reaches a given tolerance:
        # err(k) = k^2/6  ->  k = sqrt(6*err)
        "k_at_3pct_gain_error": math.sqrt(6 * 0.03),
        "k_at_15pct_gain_error": math.sqrt(6 * 0.15),
    }


# ---------------------------------------------------------------------------
# arm P: spectral (FFT) reference
# ---------------------------------------------------------------------------

def stencil_gain_exact(kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
    """Closed form |Ghat/(i k)| for the x-component, from arm S."""
    return 2 * np.sin(kx) / kx * (1.0 / 9.0 + np.cos(ky) / 18.0)


def arm_spectral(n: int = 64, m_max: int = 20) -> dict:
    """FFT/spectral reference for the discrete gradient inside the cross-force operator.

    Two independent references are used:

    * the *closed form* of arm S, ``Ghat/(i k) = cs2 (1 - k^2/6 + ...)``, and
    * a genuine FFT derivative ``ik * FFT`` of the same field (only meaningful
      for wavevectors that are harmonics of the box, i.e. integer multiples of
      ``2*pi/n`` -- using an arbitrary k makes the FFT derivative of a
      non-periodic mode meaningless, which is why the mode ladder is harmonic).
    """
    rows = []
    worst_closed = 0.0
    worst_fft = 0.0
    # only reciprocal-lattice wavevectors are eigenmodes of the periodic
    # convolution, so the mode ladder is built from integer multiples of 2*pi/n
    for m in range(1, m_max + 1):
        for tag, (mx, my) in (("axis", (m, 0)), ("diagonal", (m, m))):
            kx, ky = 2 * math.pi * mx / n, 2 * math.pi * my / n
            field = _mode(n, kx, ky)
            t = torch.as_tensor(field, dtype=DT)
            sx, sy = _raw_sum(t)
            got, ref = (-sx.numpy(), _mode_grad(field, kx, ky, 0)) if mx else \
                       (-sy.numpy(), _mode_grad(field, kx, ky, 1))
            fft_axis = 1 if mx else 0   # numpy axis of the measured component
            k_meas = kx if mx else ky
            g_lat = _projection(got, ref)
            g_fft = _projection(_fft_grad(field, axis=fft_axis), ref)
            g_closed = float(2 * math.sin(k_meas) / k_meas *
                             (1 / 9 + math.cos(ky if mx else kx) / 18))
            worst_closed = max(worst_closed, abs(g_lat / g_closed - 1))
            worst_fft = max(worst_fft, abs(g_fft - 1.0))
            rows.append({"m": m, "dir": tag, "k": float(math.hypot(kx, ky)),
                         "k_measured_axis": float(k_meas),
                         "gain_lattice": g_lat, "gain_closed_form": g_closed,
                         "gain_over_cs2": g_lat / CS2,
                         "rel_err_vs_continuum": abs(g_lat / CS2 - 1),
                         "cells_per_wavelength": float(2 * math.pi / math.hypot(kx, ky))})

    # anisotropy of the closed form on a fixed-|k| ring (equally spaced angles)
    aniso = []
    for kmag in (0.3, 0.6, 1.0, 1.5, 2.0):
        gs = []
        for j in range(32):
            ang = math.pi * j / 64  # first octant is enough by symmetry
            gx = 2 * np.sin(kmag * math.cos(ang)) / max(kmag * math.cos(ang), 1e-30) * \
                (1 / 9 + np.cos(kmag * math.sin(ang)) / 18)
            gs.append(float(gx))
        aniso.append({"k": kmag, "spread_rel": (max(gs) - min(gs)) / (sum(gs) / len(gs))})

    # grid-refinement convergence at fixed physical wavenumber (3 waves per box)
    conv = []
    for n_ in (32, 64, 128, 256):
        kk = 2 * math.pi * 3.0 / n_
        field = _mode(n_, kk, kk)
        t = torch.as_tensor(field, dtype=DT)
        sx, _ = _stencil_gradient(t)
        ref = _mode_grad(field, kk, kk, axis=0)
        g_lat = _projection(-sx.numpy(), ref)
        conv.append({"n": n_, "k": float(kk),
                     "rel_err_vs_continuum": abs(g_lat / CS2 - 1)})
    slopes = []
    for a, b in zip(conv, conv[1:]):
        if a["rel_err_vs_continuum"] > 0 and b["rel_err_vs_continuum"] > 0 and b["n"] > a["n"]:
            slopes.append(float(np.log(a["rel_err_vs_continuum"] / b["rel_err_vs_continuum"]) /
                                np.log(b["n"] / a["n"])))
    return {
        "rows": rows,
        "max_rel_diff_lattice_vs_closed_form": float(worst_closed),
        "max_rel_diff_fft_gain_vs_unity": float(worst_fft),
        "anisotropy_closed_form": aniso,
        "refinement": conv,
        "refinement_order": slopes,
        "gain_error_vs_k": [{"k": float(k), "rel_err_vs_continuum": float(k * k / 6.0),
                             "cells_per_wavelength": float(2 * math.pi / k)}
                            for k in (0.3, 0.6, 0.957, 1.5, 2.0)],
    }


def _mode(n, kx, ky):
    x = np.arange(n)
    X, Y = np.meshgrid(x, x, indexing="xy")
    return np.cos(kx * X + ky * Y)


def _stencil_gradient(t: torch.Tensor):
    """sum_i w_i c_i f(x-c_i) -- the operator inside sc_two_component_force."""
    return _raw_sum(t)


def _raw_sum(t: torch.Tensor):
    fx = torch.zeros_like(t)
    fy = torch.zeros_like(t)
    c = C.to(t.device)
    w = W.to(t.device)
    for q in range(9):
        shifted = torch.roll(t, shifts=(int(c[q, 1]), int(c[q, 0])), dims=(-2, -1))
        fx = fx + w[q] * c[q, 0] * shifted
        fy = fy + w[q] * c[q, 1] * shifted
    return fx, fy


def _fft_grad(field: np.ndarray, axis: int):
    n = field.shape[axis]
    k = 2 * np.pi * np.fft.fftfreq(n)
    shape = [1, 1]
    shape[axis] = n
    kv = k.reshape(shape)
    return np.real(np.fft.ifft(np.fft.fft(field, axis=axis) * 1j * kv, axis=axis))


def _mode_grad(field: np.ndarray, kx, ky, axis: int) -> np.ndarray:
    """Analytic derivative of the pure mode cos(kx X + ky Y) along one axis."""
    k = kx if axis == 0 else ky
    return -k * np.sin(kx * np.arange(field.shape[1])[None, :] +
                       ky * np.arange(field.shape[0])[:, None])


def _projection(got: np.ndarray, ref: np.ndarray) -> float:
    """Scalar gain for a pure mode: <got, ref>/<ref, ref>."""
    return float(np.sum(got * ref) / np.sum(ref * ref))


# ---------------------------------------------------------------------------
# arm O: OSS (lbmpy 2.0) transcription vs library kernel
# ---------------------------------------------------------------------------

OSS_SOURCE = {
    "package": "lbmpy 2.0 (upstream sdist file tests/shan_chen/test_shan_chen_two_component.py:55-59,73-81)",
    "operator": "F_a = -g_ab * psi(rho_a) * sum_d w_d d psi(rho_b(x+d))   [forward gather]",
    "convention_bridge": "G_12 (library, <0 repulsive) = -g_ab (OSS, >0 repulsive)",
}


def oss_force(rho_a: torch.Tensor, rho_b: torch.Tensor, g_ab: float,
              psi_a=psi_linear, psi_b=psi_linear):
    """Independent transcription of the OSS operator (forward gather, +d)."""
    pa = psi_a(rho_a)
    pb = psi_b(rho_b)
    # forward gather: sum_d w_d d psi_b(x+d) = -sum_d w_d d psi_b(x-d) on a
    # symmetric D2Q9 stencil; implemented literally, forward.
    n = pb.shape[-1]
    acc_x = torch.zeros_like(pb)
    acc_y = torch.zeros_like(pb)
    c = C.to(pb.device)
    w = W.to(pb.device)
    for q in range(9):
        shifted = torch.roll(pb, shifts=(-int(c[q, 1]), -int(c[q, 0])), dims=(-2, -1))
        acc_x = acc_x + w[q] * c[q, 0] * shifted
        acc_y = acc_y + w[q] * c[q, 1] * shifted
    return -g_ab * pa * acc_x, -g_ab * pa * acc_y


def arm_oss(n: int = 48, seed: int = 7, frozen_ref: str | None = None) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    out = {"oss_source": OSS_SOURCE, "n": n}
    cases = {}
    for name, psi_a, psi_b, G in (
        ("random_linear", psi_linear, psi_linear, -1.3),
        ("smooth_linear", psi_linear, psi_linear, -1.3),
        ("random_exp", psi_exp, psi_exp, +1.3),   # OSS convention g_ab>0 repulsive
    ):
        if name.startswith("random"):
            r1 = torch.as_tensor(1.0 + 0.4 * rng.standard_normal((n, n)), dtype=DT)
            r2 = torch.as_tensor(1.0 + 0.4 * rng.standard_normal((n, n)), dtype=DT)
        else:
            x = np.arange(n)
            X, Y = np.meshgrid(x, x, indexing="xy")
            r1 = torch.as_tensor(1.0 + 0.3 * np.cos(2 * np.pi * X / n), dtype=DT)
            r2 = torch.as_tensor(1.0 + 0.3 * np.cos(2 * np.pi * X / n + 0.7), dtype=DT)
        # convention bridge: library G_12 = -g_ab
        fx1, fy1, _, _ = sc_two_component_force(r1, r2, -G, psi=psi_b)
        ox, oy = oss_force(r1, r2, G, psi_a=psi_a, psi_b=psi_b)
        scale = float(torch.sqrt((ox ** 2 + oy ** 2).mean()))
        d = float(torch.sqrt(((fx1 - ox) ** 2 + (fy1 - oy) ** 2).mean()) / scale)
        cases[name] = {
            "G_12_library": -G, "g_ab_oss": G,
            "rel_rms_diff": d,
            "max_abs_diff": float(max((fx1 - ox).abs().max(), (fy1 - oy).abs().max())),
            "rms_force": scale,
        }
    out["cases"] = cases
    out["max_rel_rms_diff"] = max(c["rel_rms_diff"] for c in cases.values())
    if frozen_ref and os.path.exists(frozen_ref):
        out["frozen_oss_transcription_sha256"] = hashlib.sha256(
            open(frozen_ref, "rb").read()).hexdigest()
    return out


# ---------------------------------------------------------------------------
# arm F: two-component Shan-Chen to mechanical equilibrium + analytic invariants
# ---------------------------------------------------------------------------

def _mcmp_pressure(rho1: torch.Tensor, rho2: torch.Tensor, G: float,
                   psi=psi_linear, convention: str = "derived") -> torch.Tensor:
    """Mixture pressure of the cross-force-only model, two conventions.

    ``derived`` -- from the momentum balance of the *discrete* kernel used here:
        F_s = -G psi_s sum_i w_i c_i psi_s'(x - c_i) = +G cs^2 psi_s grad psi_s'
    so F_1 + F_2 = G cs^2 grad(psi_1 psi_2) = grad(G cs^2 psi_1 psi_2) and the
    total momentum flux is
        P = cs^2 (rho_1 + rho_2) I - G cs^2 psi_1 psi_2 I.
    With G < 0 (repulsive) the mixed region is at *higher* pressure than either
    bulk phase, which is what drives separation.

    ``library`` -- the convention documented in
    ``tensorlbm.mcmp_verification.sc_mcmp_pressure``:
        p = cs^2 (rho_1 + rho_2) + G cs^2 psi_1 psi_2.
    For a flat interface both differ only by the sign of the interaction term,
    so mechanical equilibrium decides the question: only the correct
    convention gives a spatially uniform pressure across the interface.
    """
    p1, p2 = psi(rho1), psi(rho2)
    ideal = CS2 * (rho1 + rho2)
    inter = CS2 * p1 * p2
    return ideal - G * inter if convention == "derived" else ideal + G * inter


def arm_physics(n: int = 96, G_values=(-2.5, -2.6), steps: int = 12000,
                psi_name: str = "linear", tau: float = 1.0, u_eq: str = "self") -> dict:
    """Flat-interface mechanical equilibrium + analytic invariants."""
    from tensorlbm.mcmp_verification import (  # proven in-repo relaxation harness
        init_planar_slab, sc_mcmp_pressure, simulate_planar_slab,
    )

    psi = PSI[psi_name]
    out = {"cases": []}
    for G in G_values:
        run = simulate_planar_slab(G_12=G, n=n, steps=steps, tau1=tau, tau2=tau,
                                   u_eq=u_eq)
        r1, r2 = run["rho1"], run["rho2"]
        diag = _diagnose_equilibrium(r1, r2, G, psi=psi)
        p_lib_fn = sc_mcmp_pressure(r1, r2, G)
        p_lib = (p_lib_fn.max() - p_lib_fn.min()).abs() / p_lib_fn.abs().mean()
        diag.update({
            "G": G, "status": "ok", "n": n, "steps_requested": steps,
            "steps_run": run.get("steps_run", run.get("steps")),
            "converged": bool(run.get("converged", False)),
            "u_max": float(run.get("u_max", float("nan"))),
            "p_uniformity_library_convention": float(p_lib),
            "psi": psi_name, "tau": tau, "u_eq": u_eq,
        })
        # mass conservation over a short integration of the same state
        diag["mass_drift_rel"] = _mass_drift(G, n, tau, u_eq, psi, steps=200)
        out["cases"].append(diag)
        out.setdefault("profiles", {})[f"G{G}"] = {
            "rho1": r1.mean(dim=0).tolist(), "rho2": r2.mean(dim=0).tolist()}
        out.setdefault("fields", {})[f"G{G}"] = {"rho1": r1.numpy(), "rho2": r2.numpy()}
    return out


def _mass_drift(G, n, tau, u_eq, psi, steps=200) -> float:
    from tensorlbm.mcmp_verification import init_planar_slab
    from tensorlbm.multiphase import collide_sc_two_component
    f1, f2 = init_planar_slab(n, 0.92, 0.08, 0.14, 0.86, half_thickness=0.23 * n,
                              interface_width=2.0)
    m1 = float(f1.sum()); m2 = float(f2.sum())
    for _ in range(steps):
        f1, f2 = collide_sc_two_component(f1, f2, G_12=G, tau1=tau, tau2=tau, u_eq=u_eq)
        f1 = _stream(f1); f2 = _stream(f2)
    return max(abs(float(f1.sum()) - m1) / m1, abs(float(f2.sum()) - m2) / m2)


def _stream(f: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(f)
    c = C.to(f.device)
    for q in range(9):
        out[q] = torch.roll(f[q], shifts=(int(c[q, 1]), int(c[q, 0])), dims=(-2, -1))
    return out


def _detect_normal(r1_2d: torch.Tensor, r2_2d: torch.Tensor) -> int:
    """Which axis carries the interface (0 = x, 1 = y)?

    The slab harness may lay the layers along either axis; averaging over the
    *homogeneous* direction is the only way to get a meaningful profile.
    """
    s0 = float((r1_2d.std(dim=0).mean() + r2_2d.std(dim=0).mean()))
    s1 = float((r1_2d.std(dim=1).mean() + r2_2d.std(dim=1).mean()))
    return 1 if s1 > s0 else 0   # returns the axis the density varies along


def _profile(field: torch.Tensor, normal: int) -> torch.Tensor:
    """Average out the homogeneous axis, keep the profile along ``normal``."""
    return field.mean(dim=1) if normal == 0 else field.mean(dim=0)


def _diagnose_equilibrium(r1_2d, r2_2d, G, psi) -> dict:
    """All invariants from one relaxed state.

    ``normal`` is the direction of the density variation (found automatically);
    the other direction is homogeneous and is averaged out.
    """
    normal = _detect_normal(r1_2d, r2_2d)
    r1 = _profile(r1_2d, normal)
    r2 = _profile(r2_2d, normal)
    n = int(r1.shape[0])
    p1, p2 = psi(r1), psi(r2)

    p_der = _mcmp_pressure(r1, r2, G, psi, "derived")
    p_uniform = float((p_der.max() - p_der.min()).abs() / p_der.abs().mean())

    # discrete Boltzmann relation: ln rho_1 - G psi_2 = const
    lnr1 = torch.log(r1)
    resid = lnr1 - G * p2
    span = float(lnr1.max() - lnr1.min())
    boltz = float((resid.max() - resid.min()).abs() / span) if span > 1e-12 else float("nan")

    # operator error on the *measured* interface: lattice vs spectral force.
    # The profile is embedded in a 2-D field that is homogeneous along the
    # transverse axis, so the lattice sum reduces to the 1-D stencil along the
    # normal; the reference is the exact (FFT) derivative of the same profile.
    a1, a2 = psi(r1), psi(r2)
    if normal == 0:      # profile along dim 0 (rows) -> force component y
        T1 = a1.reshape(n, 1).repeat(1, 8)
        T2 = a2.reshape(n, 1).repeat(1, 8)
        _, fy = _raw_sum(T2)
        f_lat = (-G * T1 * fy)[:, 0]
        psi2 = T2[:, 0].numpy()
    else:                # profile along dim 1 (columns) -> force component x
        T1 = a1.reshape(1, n).repeat(8, 1)
        T2 = a2.reshape(1, n).repeat(8, 1)
        fx, _ = _raw_sum(T2)
        f_lat = (-G * T1 * fx)[0]
        psi2 = T2[0].numpy()
    k = 2 * np.pi * np.fft.fftfreq(n)
    grad_n = np.real(np.fft.ifft(np.fft.fft(psi2) * 1j * k))
    f_spec = torch.as_tensor(G * CS2 * a1.numpy() * grad_n)
    denom = float(torch.sqrt((f_spec ** 2).sum()))
    op_err = float(torch.sqrt(((f_lat - f_spec) ** 2).sum()) / denom) if denom > 0 else float("nan")

    spec = np.abs(np.fft.rfft(psi2 - psi2.mean())) ** 2
    kk = 2 * np.pi * np.fft.rfftfreq(n)
    hi = float(spec[kk > 0.6].sum() / spec.sum()) if spec.sum() > 0 else 0.0

    # Newton's third law of the cross force: sum_x (F_1 + F_2) = 0 exactly
    F = sc_two_component_force(r1_2d, r2_2d, G, psi=psi)
    scale = max(float(F[j].abs().max()) for j in range(4))
    integ = max(abs(float((F[0] + F[2]).sum())), abs(float((F[1] + F[3]).sum())))
    third_law = integ / (scale * r1_2d.numel()) if scale > 0 else float("nan")

    # density contrast over the whole box (a real interface has both phases)
    contrast = float((r1.max() - r1.min()) / (r1.max() + r1.min()))

    return {
        "normal_axis": int(normal),
        "interface_contrast": contrast,
        "p_uniformity_rel": p_uniform,
        "boltzmann_relation_residual_rel": boltz,
        "operator_error_on_interface_rel": op_err,
        "interface_spectrum_frac_above_k06": hi,
        "newton_third_law_residual_rel": third_law,
        "rho1_min": float(r1.min()), "rho1_max": float(r1.max()),
        "rho2_min": float(r2.min()), "rho2_max": float(r2.max()),
        "p_mean": float(p_der.mean()),
    }


if __name__ == "__main__":
    torch.set_num_threads(8)
    print(json.dumps(arm_symbol(), indent=1)[:2000])
    print(json.dumps(arm_oss(), indent=1)[:600])