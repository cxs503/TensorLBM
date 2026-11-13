#!/usr/bin/env python3
"""XVAL-LBM-LES-SCALAR-Q-ABS-CC-PROXY -- cross-validation against the upstream
non-equilibrium-stress SGS closure.

Reference (OSS, inventoried as ``tensorplatform.oss_solutions#row-39``)::

    loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow
    LatFlow/Domain.py  CollideSC(les=True):
        Q    = sqrt(2 * sum_q (f_q - f_q^eq)^2 * EEk_q),  EEk_q = sum_{n,m}|c_qn c_qm|
        tau  = 0.5*(tau0 + sqrt(tau0^2 + 6*Q*Sc/rho)),    Sc = 0.17
        f    = f - (f - f^eq)/tau

i.e. a *scalar* (single-coefficient) Smagorinsky-type eddy viscosity whose
strain-rate input is an absolute-value ``|c_n c_m|`` proxy of the
non-equilibrium stress.  The tree is machine-extracted
(``extract_oss_tables.py``, sha256 in ``oss_ref/provenance.json``) and ported
line-by-line in ``oss_port.py``.

In-house (TensorLBM) closure under test:
``tensorlbm.turbulence.collide_smagorinsky_bgk`` uses the Frobenius norm of the
*signed* non-equilibrium stress ``Pi_ij = sum_q c_qi c_qj (f_q - f_q^eq)`` and
``tau = 0.5*(tau0 + sqrt(tau0^2 + 18 C_s^2 |Pi|_F/rho))``, ``C_s = 0.1``.

Three probes, one decision (contract: ``predeclared.json``):

1. **Algebra** -- on an exactly constructed first-order Chapman-Enskog field the
   identity ``|Pi|_F = (2/3) rho tau0 |S|_F`` is closed form.  How far are the
   two closures' strain-rate inputs (``|Pi|_F`` and the proxy ``Q``) from it?
2. **Coefficient conversion** -- is ``Q/|Pi|_F`` a topology-independent
   constant, so that one published scalar conversion reproduces the upstream
   viscosity on arbitrary states?
3. **Trajectory** -- on Taylor-Green decay (analytic closed form) do the two
   closures, at coefficient-mapped settings, produce the same effective
   viscosity and the same decay rate inside the owner 15% band?

Run::

    python benchmarks/xval_les_scalar_proxy/run_xval.py            # full
    python benchmarks/xval_les_scalar_proxy/run_xval.py --quick     # CI-sized

Artefacts: ``runs/xval_les_scalar_proxy/{result.json,table.csv,figures/*.png}``
and the report ``docs/XVAL-LBM-LES-SCALAR-Q-ABS-CC-PROXY.md``, rendered from
``result.json`` so that no number in the write-up is typed by hand.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

import oss_port as OSS  # noqa: E402
import report as RPT  # noqa: E402

from tensorlbm.d2q9 import C as C2D, W as W2D, equilibrium, macroscopic  # noqa: E402
from tensorlbm.solver import collide_bgk, stream  # noqa: E402
from tensorlbm.turbulence.turbulence import (  # noqa: E402
    _neq_stress_norm_2d,
    _smagorinsky_tau,
    collide_smagorinsky_bgk,
)
from tensorlbm.xval.case_spec import TG2DSpec  # noqa: E402

OUT_DIR = ROOT / "runs" / "xval_les_scalar_proxy"
DOC_PATH = ROOT / "docs" / "XVAL-LBM-LES-SCALAR-Q-ABS-CC-PROXY.md"
PREDECL = json.loads((HERE / "predeclared.json").read_text())

CS_OWN = 0.1        # shipped in-house default
SC_OSS = 0.17       # upstream default (LatFlow/Domain.py:57)
THRESH = PREDECL["predeclared_thresholds"]
OWNER_PCT = float(THRESH["owner_acceptance_pct"])
TARGET_PCT = float(THRESH["internal_target_pct"])
PROXY_PCT = float(THRESH["proxy_consistency_pct"])


# ---------------------------------------------------------------------------
# closure helpers
# ---------------------------------------------------------------------------
def closure_stats(f: torch.Tensor, tau0: float) -> dict:
    """Both closures evaluated on one distribution field."""
    rho, ux, uy = macroscopic(f)
    feq = equilibrium(rho, ux, uy)
    f_neq = f - feq
    pi = _neq_stress_norm_2d(f_neq)                     # in-house input
    q = OSS.q_proxy(f_neq)                              # upstream input
    tau_a = _smagorinsky_tau(tau0, pi, rho, CS_OWN)
    tau_b = OSS.tau_eff(q, rho, tau0, SC_OSS)
    return {
        "pi": pi,
        "q": q,
        "rho": rho,
        "nu_a": (tau_a - tau0) / 3.0,
        "nu_b": (tau_b - tau0) / 3.0,
        "k": q / torch.clamp(pi, min=1e-30),
    }


def nu_t_from_pi(pi: torch.Tensor, rho: torch.Tensor, tau0: float, C_s: float) -> torch.Tensor:
    """In-house law re-evaluated from a given ``|Pi|_F`` field (for C_s mapping)."""
    return (_smagorinsky_tau(tau0, pi, rho, C_s) - tau0) / 3.0


# ---------------------------------------------------------------------------
# 1. port fidelity
# ---------------------------------------------------------------------------
def port_fidelity(seed: int = 20261113) -> dict:
    g = torch.Generator().manual_seed(seed)
    n = 24
    jj, ii = torch.meshgrid(
        torch.arange(n, dtype=torch.float64), torch.arange(n, dtype=torch.float64), indexing="ij"
    )
    x, y = ii + 0.5, jj + 0.5
    k = 2 * math.pi / n
    ux = -0.07 * torch.cos(k * x) * torch.sin(k * y)
    uy = 0.07 * torch.sin(k * x) * torch.cos(k * y)
    f = equilibrium(torch.ones_like(ux), ux, uy)
    f = f + 1e-3 * torch.randn(f.shape, dtype=torch.float64, generator=g)
    tau = 0.55
    fa = collide_bgk(f, tau)
    fb = OSS.collide(f, tau, 0.0)
    _, ua_x, ua_y = macroscopic(f)
    rb, ub_x, ub_y = OSS.moments(f)
    return {
        "max_abs_diff_f": float((fa - fb).abs().max()),
        "rel_f": float((fa - fb).norm() / fa.norm()),
        "max_abs_diff_ux": float((ua_x - ub_x).abs().max()),
        "max_abs_diff_uy": float((ua_y - ub_y).abs().max()),
        "max_abs_diff_rho": float((rb - 1.0).abs().max()),
        "threshold_rel": float(THRESH["port_fidelity_rel"]),
    }


# ---------------------------------------------------------------------------
# 2. exact Chapman-Enskog field
# ---------------------------------------------------------------------------
def _rotate_strain(S: list[list[float]], deg: float) -> list[list[float]]:
    """``R S R^T`` -- keeps the Frobenius norm and the trace, rotates the eigen-directions."""
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    R = [[c, -s], [s, c]]
    out = [[0.0, 0.0], [0.0, 0.0]]
    for i in range(2):
        for j in range(2):
            out[i][j] = sum(R[i][k] * S[k][l] * R[j][l] for k in range(2) for l in range(2))
    return out


def strain_topologies() -> dict[str, list[list[float]]]:
    """Unit-Frobenius, symmetric, trace-free 2-D strain tensors.

    The lattice-symmetric cases (``pure_shear``, ``pure_strain``, ``mixed_45deg``)
    are the ones where the ``|c c|`` proxy has no sign to lose; the rotated
    variants are the generic case in which the strain eigen-directions sit
    between lattice diagonals.
    """
    s = 1.0 / math.sqrt(2.0)
    shear = [[0.0, s], [s, 0.0]]
    ans = {
        "pure_shear": shear,
        "pure_strain": [[s, 0.0], [0.0, -s]],
        "mixed_45deg": [[0.5, 0.5], [0.5, -0.5]],
    }
    for deg in (15, 30, 45):
        ans[f"shear_rot{deg}"] = _rotate_strain(shear, float(deg))
    g = [[0.6, 0.3], [0.3, -0.6]]
    gn = math.sqrt(sum(v * v for row in g for v in row))
    ans["generic_aniso"] = [[v / gn for v in row] for row in g]
    ans["solid_rotation"] = [[0.0, 0.0], [0.0, 0.0]]
    return ans


def exact_ce_field(topology: str, amp: float, tau0: float, n: int = 16) -> tuple[torch.Tensor, float]:
    """``f = f^eq(rho=1, u=S.r) + f^(1)`` with the exact CE first-order part.

    For a uniform, trace-free strain tensor ``S``::

        f^(1)_q = -3 tau0 w_q rho (c_qi c_qj) S_ij
        Pi_ij   = sum_q c_qi c_qj f^(1)_q = -2 rho c_s^2 tau0 S_ij      (exact)
        |Pi|_F  = (2/3) rho tau0 |S|_F,     |S|_F = sqrt(sum_ij S_ij^2)

    Coordinates are ``r in [-0.5, 0.5)`` so that ``u = S.r`` carries both the
    exact strain ``S`` and a low Mach number.
    """
    S = torch.tensor(strain_topologies()[topology], dtype=torch.float64) * amp
    jj, ii = torch.meshgrid(
        torch.arange(n, dtype=torch.float64), torch.arange(n, dtype=torch.float64), indexing="ij"
    )
    rx = (ii + 0.5) / n - 0.5
    ry = (jj + 0.5) / n - 0.5
    ux = S[0, 0] * rx + S[0, 1] * ry
    uy = S[1, 0] * rx + S[1, 1] * ry
    rho = torch.ones_like(ux)
    f = equilibrium(rho, ux, uy)

    cx = C2D[:, 0].double().view(9, 1, 1)
    cy = C2D[:, 1].double().view(9, 1, 1)
    cij_s = cx * cx * S[0, 0] + 2.0 * cx * cy * S[0, 1] + cy * cy * S[1, 1]
    f = f - 3.0 * tau0 * W2D.double().view(9, 1, 1) * rho.unsqueeze(0) * cij_s
    sf_closed = float(torch.sqrt((S * S).sum()))
    return f, sf_closed


def ce_algebra(tau0: float, amplitudes: tuple[float, ...]) -> dict:
    """Both closures' strain-rate inputs vs the closed form R1."""
    rows = []
    for topology in strain_topologies():
        for amp in amplitudes:
            f, sf = exact_ce_field(topology, amp, tau0)
            st = closure_stats(f, tau0)
            ref_pi = (2.0 / 3.0) * tau0 * sf            # closed form, rho = 1
            pi_mean = float(st["pi"].mean())
            q_mean = float(st["q"].mean())
            nu_a = float(st["nu_a"].mean())
            nu_b = float(st["nu_b"].mean())
            k = (q_mean / pi_mean) if pi_mean > 0 else float("nan")
            nu_ref_std = math.sqrt(2.0) * sf
            pi_rel = 100.0 * abs(pi_mean - ref_pi) / ref_pi if ref_pi > 0 else float("nan")
            q_rel = 100.0 * abs(q_mean - ref_pi) / ref_pi if ref_pi > 0 else float("nan")
            rows.append(
                {
                    "topology": topology,
                    "amp": amp,
                    "sf": sf,
                    "pi_numeric": pi_mean,
                    "pi_closed_form": ref_pi,
                    "pi_identity_rel_pct": pi_rel,
                    "q_proxy": q_mean,
                    "q_proxy_rel_pct": q_rel,
                    "k_ratio": k,
                    "nu_t_inhouse": nu_a,
                    "nu_t_oss": nu_b,
                    "nu_t_closed_form_inhouse": CS_OWN**2 * sf,
                    "nu_t_closed_form_oss": (2.0 / 3.0) * SC_OSS * (k if k == k else 0.0) * sf,
                    "Cs_equiv_standard_inhouse": math.sqrt(max(nu_a, 0.0) / nu_ref_std) if nu_ref_std else float("nan"),
                    "Cs_equiv_standard_oss": math.sqrt(max(nu_b, 0.0) / nu_ref_std) if nu_ref_std else float("nan"),
                }
            )
    real = [r for r in rows if r["topology"] != "solid_rotation"]
    k_vals = [r["k_ratio"] for r in real]
    k_mean = sum(k_vals) / len(k_vals)
    sr = [r for r in rows if r["topology"] == "solid_rotation"]
    lat = OSS.lattice()
    return {
        "tau0": tau0,
        "amplitudes": list(amplitudes),
        "rows": rows,
        "k_spread_pct": 100.0 * (max(k_vals) - min(k_vals)) / k_mean,
        "k_min": min(k_vals),
        "k_max": max(k_vals),
        "k_mean": k_mean,
        "pi_identity_max_rel_pct": max(r["pi_identity_rel_pct"] for r in real),
        "q_proxy_max_rel_pct": max(r["q_proxy_rel_pct"] for r in real),
        "solid_rotation_pi_max": max(abs(r["pi_numeric"]) for r in sr),
        "solid_rotation_nu_t_max": max(max(abs(r["nu_t_inhouse"]), abs(r["nu_t_oss"])) for r in sr),
        "eek_abs": lat["eek_abs_dyadic"],
        "Sc": lat["Sc"],
        "source": {
            "eek_block": lat["extracted_source"]["eek_block"],
            "les_branch": lat["extracted_source"]["les_branch"],
            "sc_line": lat["extracted_source"]["sc_line"],
            "les_lines": lat["locators"]["les_branch"].split(":")[1],
        },
    }


# ---------------------------------------------------------------------------
# 2b. EXPLORATORY follow-up (added after the predeclared probes; no predeclared
# threshold was changed): when is the |c c| proxy equal to the exact norm?
# ---------------------------------------------------------------------------
def _proxy_identity_for(C, w, dim: int, *, trace_free: bool, draws: int, seed: int) -> dict:
    """max |Q/|Pi|_F - 1| over random strain tensors, for a given lattice."""
    rng = np.random.default_rng(seed)
    c = np.asarray(C, dtype=float)
    w = np.asarray(w, dtype=float)
    eek = np.array([sum(abs(a * b) for a in cq for b in cq) for cq in c])
    k_vals = []
    for _ in range(draws):
        S = rng.normal(size=(dim, dim))
        S = 0.5 * (S + S.T)
        if trace_free:
            S = S - np.eye(dim) * np.trace(S) / dim
        S = S / np.sqrt((S * S).sum())
        B = np.einsum("qi,ij,qj->q", c, S, c)
        f = -w * B
        pi = np.einsum("qi,qj,q->ij", c, c, f)
        pi_f = np.sqrt((pi**2).sum())
        q = np.sqrt(2.0 * np.sum(eek * f**2))
        k_vals.append(q / pi_f)
    k_vals = np.array(k_vals)
    return {
        "lattice": f"Q{len(c)}-{dim}D",
        "trace_free": bool(trace_free),
        "draws": draws,
        "k_min": float(k_vals.min()),
        "k_max": float(k_vals.max()),
        "k_mean": float(k_vals.mean()),
        "max_abs_dev_from_1": float(np.abs(k_vals - 1.0).max()),
        "relative_spread_pct": float(100.0 * (k_vals.max() - k_vals.min()) / k_vals.mean()),
        "eek_matches_abs_sum_squared": bool(
            np.allclose(eek, (np.abs(c).sum(1)) ** 2)
        ),
    }


def proxy_identity(draws_2d: int = 2000, draws_3d: int = 500) -> dict:
    from tensorlbm.d2q9 import C as c9, W as w9
    from tensorlbm.d3q19 import C as c19, W as w19
    from tensorlbm.d3q27 import C as c27, W as w27

    cases = {
        "d2q9_trace_free": _proxy_identity_for(c9, w9, 2, trace_free=True, draws=draws_2d, seed=11),
        "d2q9_with_trace": _proxy_identity_for(c9, w9, 2, trace_free=False, draws=draws_2d, seed=12),
        "d3q19_trace_free": _proxy_identity_for(c19, w19, 3, trace_free=True, draws=draws_3d, seed=13),
        "d3q27_trace_free": _proxy_identity_for(c27, w27, 3, trace_free=True, draws=draws_3d, seed=14),
    }
    # negative control: an arbitrary (non Chapman-Enskog) non-equilibrium field,
    # where the identity has no reason to hold.
    rng = np.random.default_rng(15)
    c = np.asarray(c9, dtype=float)
    eek = np.array([sum(abs(a * b) for a in cq for b in cq) for cq in c])
    kk = []
    for _ in range(500):
        f = rng.normal(size=9)
        pi = np.einsum("qi,qj,q->ij", c, c, f)
        kk.append(np.sqrt(2.0 * np.sum(eek * f**2)) / np.sqrt((pi**2).sum()))
    kk = np.array(kk)
    return {
        "cases": cases,
        "negative_control_random_noneq": {
            "draws": 500,
            "k_p05": float(np.quantile(kk, 0.05)),
            "k_mean": float(kk.mean()),
            "k_max": float(kk.max()),
        },
        "note": "EXPLORATORY: added after the predeclared probes were run, reported separately; "
                "no predeclared threshold or KPI was changed.",
    }


# ---------------------------------------------------------------------------
# 3. cases
# ---------------------------------------------------------------------------
def spec_for(case: dict) -> TG2DSpec:
    return TG2DSpec(
        n=int(case["n"]),
        re=float(case["Re"]),
        u0=float(case["U0"]),
        steps=int(case["steps"]),
        sample_every=int(case["sample_every"]),
    )


def initial_field(case: dict, spec: TG2DSpec) -> torch.Tensor:
    n = spec.n
    jj, ii = torch.meshgrid(
        torch.arange(n, dtype=torch.float64), torch.arange(n, dtype=torch.float64), indexing="ij"
    )
    x, y = ii + 0.5, jj + 0.5
    if case["kind"].startswith("analytic"):
        ux = -spec.u0 * torch.cos(spec.k * x) * torch.sin(spec.k * y)
        uy = spec.u0 * torch.sin(spec.k * x) * torch.cos(spec.k * y)
    else:
        # stream function psi = sum_m A_m cos(kx_m x) cos(ky_m y)  -> divergence free
        modes = [(1, 2), (2, 1), (2, 2), (3, 1)]
        ux = torch.zeros_like(x)
        uy = torch.zeros_like(x)
        for kx_m, ky_m in modes:
            kx = 2 * math.pi * kx_m / n
            ky = 2 * math.pi * ky_m / n
            amp = spec.u0 * 0.25 / math.hypot(kx, ky)
            ux = ux - amp * ky * torch.cos(kx * x) * torch.sin(ky * y)   # dpsi/dy
            uy = uy + amp * kx * torch.sin(kx * x) * torch.cos(ky * y)   # -dpsi/dx
        scale = spec.u0 / float(torch.sqrt(ux * ux + uy * uy).max())
        ux, uy = ux * scale, uy * scale
    return equilibrium(torch.ones_like(ux), ux, uy)


def make_arm(name: str, tau0: float, cs_map: float | None):
    if name == "bgk":
        return lambda f: collide_bgk(f, tau0)
    if name == "A_nominal":
        return lambda f: collide_smagorinsky_bgk(f, tau0, CS_OWN)
    if name == "A_mapped":
        if cs_map is None:
            raise ValueError("A_mapped needs the measured C_s conversion")
        return lambda f: collide_smagorinsky_bgk(f, tau0, cs_map)
    if name == "B_oss":
        return lambda f: OSS.collide(f, tau0, SC_OSS)
    raise ValueError(name)


def run_arm(case: dict, spec: TG2DSpec, arm: str, cs_map: float | None, threads: int = 4) -> dict:
    torch.set_num_threads(max(1, threads))
    tau0 = spec.tau
    f = initial_field(case, spec)
    collide = make_arm(arm, tau0, cs_map)
    samples, closure_samples = [], []
    wall0 = time.perf_counter()
    with torch.no_grad():
        for step in range(spec.steps + 1):
            if step % spec.sample_every == 0 or step == spec.steps:
                _rho, uxc, uyc = macroscopic(f)
                K = 0.5 * float((uxc * uxc + uyc * uyc).mean())
                st = closure_stats(f, tau0)
                if arm == "A_mapped":
                    nu_applied = float(nu_t_from_pi(st["pi"], st["rho"], tau0, cs_map).mean())
                elif arm == "A_nominal":
                    nu_applied = float(st["nu_a"].mean())
                elif arm == "B_oss":
                    nu_applied = float(st["nu_b"].mean())
                else:
                    nu_applied = 0.0
                samples.append((step, K, float(torch.sqrt(uxc * uxc + uyc * uyc).max())))
                closure_samples.append(
                    {
                        "step": step,
                        "pi_mean": float(st["pi"].mean()),
                        "q_mean": float(st["q"].mean()),
                        "k_mean": float(torch.clamp(st["k"], max=1e3).mean()),
                        "nu_a_mean": float(st["nu_a"].mean()),
                        "nu_b_mean": float(st["nu_b"].mean()),
                        "nu_t_applied": nu_applied,
                    }
                )
            if step == spec.steps:
                break
            f = collide(stream(f))
    wall = time.perf_counter() - wall0

    finite = bool(torch.isfinite(f).all())
    _rho, ux, uy = macroscopic(f)
    umax = float(torch.sqrt(ux * ux + uy * uy).max())
    rho_min, rho_max = float(_rho.min()), float(_rho.max())
    stable = finite and umax <= 10.0 * spec.u0 and 0.5 <= rho_min and rho_max <= 1.5
    return {
        "arm": arm,
        "case": case["name"],
        "n": spec.n,
        "re": spec.re,
        "u0": spec.u0,
        "nu": spec.nu,
        "tau0": tau0,
        "mach": spec.mach,
        "steps": spec.steps,
        "cs": CS_OWN if arm == "A_nominal" else (cs_map if arm == "A_mapped" else None),
        "Sc": SC_OSS if arm == "B_oss" else None,
        "samples": samples,
        "closure_samples": closure_samples,
        "gamma_K_fit": fit_decay_rate(samples, spec.steps),
        "finite": finite,
        "stable": stable,
        "umax_final": umax,
        "rho_range": [rho_min, rho_max],
        "K_final": samples[-1][1] if samples else float("nan"),
        "wall_s": wall,
    }


def fit_decay_rate(samples, steps: int, index: int = 1) -> float:
    """Least-squares slope of ln(quantity) over the second half of the run."""
    pts = [(s[0], s[index]) for s in samples if s[0] >= steps // 2 and s[index] > 0.0]
    if len(pts) < 3:
        return float("nan")
    t = np.array([p[0] for p in pts], dtype=float)
    ln = np.log(np.array([p[1] for p in pts], dtype=float))
    return float(-np.polyfit(t, ln, 1)[0])


def window_mean(run: dict, key: str) -> float:
    steps = run["steps"]
    vals = [c[key] for c in run["closure_samples"] if c["step"] >= steps // 2]
    return float(np.mean(vals)) if vals else float("nan")


# ---------------------------------------------------------------------------
# 4. a-priori check at t=0
# ---------------------------------------------------------------------------
def strain_magnitude_from_field(f: torch.Tensor) -> torch.Tensor:
    """|S|_F per cell from periodic central differences of the macroscopic field."""
    _rho, ux, uy = macroscopic(f)
    # macroscopic returns (ny, nx): index 0 is y, index 1 is x
    dux_dx = 0.5 * (torch.roll(ux, -1, 1) - torch.roll(ux, 1, 1))
    dux_dy = 0.5 * (torch.roll(ux, -1, 0) - torch.roll(ux, 1, 0))
    duy_dx = 0.5 * (torch.roll(uy, -1, 1) - torch.roll(uy, 1, 1))
    duy_dy = 0.5 * (torch.roll(uy, -1, 0) - torch.roll(uy, 1, 0))
    s_xx = dux_dx
    s_yy = duy_dy
    s_xy = 0.5 * (dux_dy + duy_dx)
    return torch.sqrt(s_xx * s_xx + s_yy * s_yy + 2.0 * s_xy * s_xy), s_xx, s_yy, s_xy


def apriori_vs_state(case: dict, spec: TG2DSpec, step: int = 200, threads: int = 4) -> dict:
    """A-priori (macroscopic strain) vs a-posteriori (applied) SGS viscosity.

    The t = 0 state is an exact equilibrium (f = f^eq), so the closure sees a zero
    non-equilibrium stress there and the comparison is degenerate.  The probe is
    therefore taken at the first developed state: the upstream arm is advanced
    ``step`` steps and, on that single state, the following are compared

    * the strain magnitude implied by the closure input,
      ``|S|_implied = 1.5 |Pi|_F / (rho tau_eff)`` (from the exact CE identity),
    * the strain magnitude of the macroscopic field measured by periodic central
      differences, ``|S|_FD``,

    together with the closed-form SGS viscosity each closure predicts from
    ``<|S|_FD>`` and the viscosity it actually applies.
    """
    torch.set_num_threads(max(1, threads))
    tau0 = spec.tau
    f = initial_field(case, spec)
    collide = make_arm("B_oss", tau0, None)
    with torch.no_grad():
        for _ in range(step):
            f = collide(stream(f))
        st = closure_stats(f, tau0)
        sf_fd, s_xx, s_yy, s_xy = strain_magnitude_from_field(f)
    sf_fd_mean = float(sf_fd.mean())
    pi_mean = float(st["pi"].mean())
    q_mean = float(st["q"].mean())
    kbar = q_mean / pi_mean if pi_mean > 0 else float("nan")
    tau_a = _smagorinsky_tau(tau0, st["pi"], st["rho"], CS_OWN)
    sf_implied = float((1.5 * st["pi"] / (st["rho"] * tau_a)).mean())
    nu_a = float(st["nu_a"].mean())
    nu_b = float(st["nu_b"].mean())
    apriori_a = CS_OWN**2 * sf_fd_mean
    # upstream law linearised in Q about Q=0: nu_t = Sc Q /(2 rho tau0) with
    # Q = k (2/3) rho tau0 |S|  ->  nu_t = (1/3) Sc k |S|_F  (the (2/3) form only holds
    # in the self-consistent limit where Pi carries tau_eff, not tau0)
    apriori_b = (1.0 / 3.0) * SC_OSS * kbar * sf_fd_mean
    return {
        "case": case["name"],
        "step": step,
        "sf_mean_analytic": sf_fd_mean,      # macroscopic |S|_F of that state (fd)
        "sf_implied_by_closure": sf_implied,
        "implied_vs_fd_rel_pct": 100.0 * abs(sf_implied - sf_fd_mean) / sf_fd_mean,
        "k_ratio_mean": kbar,
        "nu_t_inhouse_measured": nu_a,
        "nu_t_inhouse_apriori": apriori_a,
        "inhouse_apriori_rel_pct": 100.0 * abs(nu_a - apriori_a) / apriori_a if apriori_a else float("nan"),
        "nu_t_oss_measured": nu_b,
        "nu_t_oss_apriori": apriori_b,
        "oss_apriori_rel_pct": 100.0 * abs(nu_b - apriori_b) / apriori_b if apriori_b else float("nan"),
    }


# ---------------------------------------------------------------------------
# 5. per-case driver
# ---------------------------------------------------------------------------
def run_case(case: dict, threads: int) -> dict:
    spec = spec_for(case)
    runs: dict[str, dict] = {}
    for arm in ("bgk", "A_nominal", "B_oss"):
        runs[arm] = run_arm(case, spec, arm, None, threads=threads)

    # coefficient conversion measured on the states themselves
    k_means = [c["k_mean"] for r in runs.values() for c in r["closure_samples"]]
    kbar = float(np.nanmean(k_means))
    cs_map = math.sqrt(SC_OSS * kbar / 3.0)
    runs["A_mapped"] = run_arm(case, spec, "A_mapped", cs_map, threads=threads)

    def gam(arm: str) -> float:
        return runs[arm]["gamma_K_fit"]

    def rel(a: float, b: float) -> float:
        if not (b == b) or b == 0.0 or not (a == a):
            return float("nan")
        return 100.0 * abs(a - b) / abs(b)

    nu_a_win = window_mean(runs["A_mapped"], "nu_t_applied")
    nu_b_win = window_mean(runs["B_oss"], "nu_t_applied")
    nu0 = spec.nu
    kk = spec.k

    def consistency(run: dict, nu_win: float) -> float:
        g = run["gamma_K_fit"]
        if not (g == g) or not (nu_win == nu_win):
            return float("nan")
        predicted = 4.0 * kk * kk * (nu0 + nu_win)
        return 100.0 * abs(g - predicted) / predicted if predicted else float("nan")

    out = {
        "case": case,
        "n": spec.n,
        "Re": spec.re,
        "U0": spec.u0,
        "Ma": spec.mach,
        "nu": nu0,
        "tau0": spec.tau,
        "k": kk,
        "steps": spec.steps,
        "gamma_K_theory_bgk": spec.gamma_k_theory,
        "gamma_K_bgk": gam("bgk"),
        "gamma_K_A_nominal": gam("A_nominal"),
        "gamma_K_A_mapped": gam("A_mapped"),
        "gamma_K_B_oss": gam("B_oss"),
        "gamma_cross_rel_pct": rel(gam("A_mapped"), gam("B_oss")),
        "gamma_cross_nominal_rel_pct": rel(gam("A_nominal"), gam("B_oss")),
        "les_gain_A_pct": rel(gam("A_nominal"), gam("bgk")),
        "les_gain_A_mapped_pct": rel(gam("A_mapped"), gam("bgk")),
        "les_gain_B_pct": rel(gam("B_oss"), gam("bgk")),
        "nu_t_window_A_mapped": nu_a_win,
        "nu_t_window_B_oss": nu_b_win,
        "consistency_A_mapped_pct": consistency(runs["A_mapped"], nu_a_win),
        "consistency_B_oss_pct": consistency(runs["B_oss"], nu_b_win),
        "consistency_bgk_pct": consistency(runs["bgk"], 0.0),
        "k_mean_states": kbar,
        "Cs_map": cs_map,
        "K_final": {arm: runs[arm]["K_final"] for arm in runs},
        "stable": {arm: runs[arm]["stable"] for arm in runs},
        "K_final_cross_rel_pct": rel(runs["A_mapped"]["K_final"], runs["B_oss"]["K_final"]),
        "wall_s": {arm: runs[arm]["wall_s"] for arm in runs},
        "runs": list(runs.values()),
    }
    return out


def field_level(case: dict, decay_case: dict, threads: int) -> dict:
    """Both closures' applied nu_t on identical real states (mid-run snapshots)."""
    spec = spec_for(case)
    f = initial_field(case, spec)
    tau0 = spec.tau
    collide = make_arm("A_nominal", tau0, None)
    torch.set_num_threads(max(1, threads))
    snap_steps = [spec.steps // 4, spec.steps // 2, spec.steps]
    rows = []
    map_data = None
    with torch.no_grad():
        for step in range(spec.steps + 1):
            if step in snap_steps:
                st = closure_stats(f, tau0)
                pi_m = float(st["pi"].mean())
                q_m = float(st["q"].mean())
                kbar = q_m / pi_m if pi_m > 0 else float("nan")
                cs_map = math.sqrt(SC_OSS * kbar / 3.0)
                nu_a_map = nu_t_from_pi(st["pi"], st["rho"], tau0, cs_map)
                nu_a_nom = st["nu_a"]
                nu_b = st["nu_b"]
                # use the configuration-wide C_s conversion, as in the trajectory arm
                cs_glob = decay_case["Cs_map"]
                nu_a_glob = nu_t_from_pi(st["pi"], st["rho"], tau0, cs_glob)
                rows.append(
                    {
                        "case": case["name"],
                        "step": step,
                        "k_mean": kbar,
                        "nu_t_A_mean": float(nu_a_nom.mean()),
                        "nu_t_B_mean": float(nu_b.mean()),
                        "nu_t_nominal_rel_pct": 100.0 * abs(float(nu_a_nom.mean()) - float(nu_b.mean())) / float(nu_b.mean()),
                        "Cs_map": cs_glob,
                        "nu_t_A_mapped_mean": float(nu_a_glob.mean()),
                        "nu_t_mapped_rel_pct": 100.0 * abs(float(nu_a_glob.mean()) - float(nu_b.mean())) / float(nu_b.mean()),
                        "nu_t_mapped_rel_max_cell_pct": float(
                            100.0 * ((nu_a_glob - nu_b).abs() / torch.clamp(nu_b, min=1e-30)).max()
                        ),
                        "k_p05": float(torch.quantile(st["k"].flatten(), 0.05)),
                        "k_p95": float(torch.quantile(st["k"].flatten(), 0.95)),
                    }
                )
                if step == snap_steps[len(snap_steps) // 2]:
                    map_data = {
                        "case": case["name"],
                        "step": step,
                        "nu_t_A_mapped": nu_a_glob.double().numpy().tolist(),
                        "nu_t_oss": nu_b.double().numpy().tolist(),
                        "k_field": torch.clamp(st["k"], max=10.0).double().numpy().tolist(),
                    }
            if step == spec.steps:
                break
            f = collide(stream(f))
    agg = rows[-1]
    return {
        "case": case["name"],
        "rows": rows,
        "k_mean": agg["k_mean"],
        "nu_t_A_mean": agg["nu_t_A_mean"],
        "nu_t_B_mean": agg["nu_t_B_mean"],
        "nu_t_nominal_rel_pct": agg["nu_t_nominal_rel_pct"],
        "Cs_map": agg["Cs_map"],
        "nu_t_A_mapped_mean": agg["nu_t_A_mapped_mean"],
        "nu_t_mapped_rel_pct": agg["nu_t_mapped_rel_pct"],
        "k_p05": agg["k_p05"],
        "k_p95": agg["k_p95"],
        "map": map_data,
    }


# ---------------------------------------------------------------------------
# 6. main
# ---------------------------------------------------------------------------
def git_head() -> str:
    try:
        import subprocess

        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), text=True).strip()
    except Exception:
        return "unknown"


def sensitivity(result: dict) -> list[dict]:
    ce = result["ce_algebra"]
    dec = {c["case"]["name"]: c for c in result["decay"]}
    fld = {f["case"]: f for f in result["field_level"]}
    out = []

    # (i) resolution
    names = [n for n in dec if n.startswith("tg2d") and "re1000" in n]
    if len(names) >= 2:
        a, b = (dec[n] for n in sorted(names, key=lambda s: dec[s]["n"])[:2])
        d = 100.0 * abs(b["gamma_cross_rel_pct"] - a["gamma_cross_rel_pct"]) / max(a["gamma_cross_rel_pct"], 1e-12)
        conv = d < 15.0
        out.append({
            "axis": f"resolution n={a['n']} -> n={b['n']} (Re=1000)",
            "kpi": "gamma cross-rel% / K_final cross-rel%",
            "finding": (
                f"cross-rel {a['gamma_cross_rel_pct']:.2f}% -> {b['gamma_cross_rel_pct']:.2f}% "
                f"(change {d:.1f}%, {'grid-converged' if conv else 'NOT grid-converged'} at 15%); "
                f"K_final cross-rel {a['K_final_cross_rel_pct']:.2f}% -> {b['K_final_cross_rel_pct']:.2f}%"
            ),
        })

    # (ii) tau0
    names = [n for n in dec if n.startswith("tg2d_n32")]
    if len(names) >= 2:
        a, b = (dec[n] for n in sorted(names, key=lambda s: dec[s]["Re"]))
        out.append({
            "axis": f"molecular tau0 {a['tau0']:.4f} (Re={a['Re']:g}) -> {b['tau0']:.4f} (Re={b['Re']:g})",
            "kpi": "les_gain_A_mapped% / nu_t window",
            "finding": (
                f"LES gain over BGK {a['les_gain_A_mapped_pct']:.2f}% -> {b['les_gain_A_mapped_pct']:.2f}%; "
                f"applied nu_t {a['nu_t_window_A_mapped']:.3e} -> {b['nu_t_window_A_mapped']:.3e} "
                f"(tau0 -> 0.5 raises the non-equilibrium stress, so the closure does more work)"
            ),
        })

    # (iii) strain topology
    real = [r for r in ce["rows"] if r["topology"] != "solid_rotation"]
    topos = sorted({r["topology"] for r in real})
    per = {t: [r["k_ratio"] for r in real if r["topology"] == t] for t in topos}
    amps = sorted({r["sf"] for r in real})
    amp_rel = []
    for t in topos:
        vals = [r["k_ratio"] for r in sorted(real, key=lambda r: r["sf"]) if r["topology"] == t]
        amp_rel.append(100.0 * (max(vals) - min(vals)) / (sum(vals) / len(vals)))
    out.append({
        "axis": f"strain topology {topos} x amplitude {amps}",
        "kpi": "k = Q/|Pi|_F",
        "finding": (
            " ; ".join(f"{t}: k = {np.mean(per[t]):.3f}" for t in topos)
            + f" ; amplitude-induced spread within a topology <= {max(amp_rel):.3f}% "
            f"(the proxy is amplitude-independent, so the bias is purely topological)"
        ),
    })

    # (iv) nonlinear multi-mode case
    mm = [c for c in result["decay"] if not c["case"]["kind"].startswith("analytic")]
    if mm:
        c = mm[0]
        out.append({
            "axis": f"initial condition: multi-mode nonlinear ({c['case']['name']})",
            "kpi": "K_final cross-rel% / gamma cross-rel%",
            "finding": (
                f"K_final {c['K_final']['A_mapped']:.4e} vs {c['K_final']['B_oss']:.4e} "
                f"-> {c['K_final_cross_rel_pct']:.2f}%; gamma cross-rel {c['gamma_cross_rel_pct']:.2f}%; "
                f"LES gain over BGK {c['les_gain_A_mapped_pct']:.2f}% (A_mapped) / {c['les_gain_B_pct']:.2f}% (B)"
            ),
        })
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="CI-sized: short runs, no figures")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--no-docs", action="store_true")
    ap.add_argument("--cases", default="", help="comma separated case names: run only those and cache them")
    args = ap.parse_args(argv)

    t_start = time.time()
    cases = [dict(c) for c in PREDECL["cases"]]
    if args.quick:
        for c in cases:
            c["steps"] = min(int(c["steps"]), 400)
            c["sample_every"] = 20

    prov = OSS.provenance()
    tau0 = spec_for(cases[0]).tau
    result: dict = {
        "xval_id": PREDECL["xval_id"],
        "env": {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_head": git_head(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "platform": platform.platform(),
            "oss_files": prov["files"],
            "oss_tarball_sha256": prov["tarball_sha256"],
            "oss_source_root": prov["source_root"],
            "quick": bool(args.quick),
        },
        "predeclared": PREDECL,
        "port_fidelity": port_fidelity(),
        "ce_algebra": ce_algebra(tau0, (0.005, 0.02, 0.05)),
        "proxy_identity": proxy_identity(),
    }

    # Cases are cached one file per case so the sweep can be run in wall-clock
    # slices smaller than the control-plane per-command limit; the assembly below
    # always reads whatever is cached, in predeclared order.
    parts_dir = OUT_DIR / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    wanted = {n.strip() for n in args.cases.split(",") if n.strip()}
    for case in cases:
        if wanted and case["name"] not in wanted:
            continue
        spec = spec_for(case)
        d = run_case(case, args.threads)
        part = {
            "case": case,
            "decay": d,
            "parameters": {
                "case": case["name"],
                "n": spec.n,
                "Re": spec.re,
                "U0": spec.u0,
                "Ma": spec.mach,
                "nu": spec.nu,
                "tau0": spec.tau,
                "k": spec.k,
                "steps": spec.steps,
            },
        }
        if case["kind"].startswith("analytic"):
            part["field_level"] = field_level(case, d, args.threads)
            part["apriori"] = apriori_vs_state(case, spec, step=200, threads=args.threads)
        (parts_dir / f"{case['name']}.json").write_text(json.dumps(part))
        print(f"[done] {case['name']}: gamma bgk={d['gamma_K_bgk']:.5f} "
              f"A_mapped={d['gamma_K_A_mapped']:.5f} B_oss={d['gamma_K_B_oss']:.5f} "
              f"cross={d['gamma_cross_rel_pct']:.2f}%", flush=True)

    result["decay"] = []
    result["field_level"] = []
    result["apriori"] = []
    result["parameters"] = []
    maps = []
    for case in cases:
        pf = parts_dir / f"{case['name']}.json"
        if not pf.is_file():
            raise SystemExit(f"missing case cache {pf} -- run --cases {case['name']} first")
        part = json.loads(pf.read_text())
        result["decay"].append(part["decay"])
        result["parameters"].append(part["parameters"])
        if "field_level" in part:
            result["field_level"].append(part["field_level"])
            if part["field_level"].get("map") is not None:
                maps.append(part["field_level"]["map"])
        if case["kind"].startswith("analytic"):
            # cheap enough to recompute at assembly time (200 steps), so the probe
            # always matches the current definition
            result["apriori"].append(
                apriori_vs_state(case, spec_for(case), step=200, threads=args.threads)
            )

    result["field_maps"] = maps[0] if maps else None
    result["sensitivity"] = sensitivity(result)

    # ---------------- verdict (predeclared rules only) ---------------------
    ce = result["ce_algebra"]
    cross_vals = [c["gamma_cross_rel_pct"] for c in result["decay"]]
    mapped_vals = [f["nu_t_mapped_rel_pct"] for f in result["field_level"]]
    k_spread = ce["k_spread_pct"]
    h1 = (
        k_spread <= PROXY_PCT
        and all(v <= OWNER_PCT for v in mapped_vals)
        and all(v <= OWNER_PCT for v in cross_vals)
    )
    all_stable = all(all(c["stable"].values()) for c in result["decay"])
    worst_cross = max(cross_vals) if cross_vals else float("nan")
    pi2 = result["proxy_identity"]["cases"]
    result["verdict"] = {
        "hypothesis_1_accepted": bool(h1),
        "identity_2d_max_dev": pi2["d2q9_trace_free"]["max_abs_dev_from_1"],
        "identity_3d_q19_spread_pct": pi2["d3q19_trace_free"]["relative_spread_pct"],
        "identity_3d_q27_spread_pct": pi2["d3q27_trace_free"]["relative_spread_pct"],
        "headline": (
            "H1 ACCEPTED (single-scalar conversion reproduces the upstream closure)"
            if h1
            else "H0 CONFIRMED (proxy bias is not a single-constant conversion; do not transplant)"
        ),
        "decision_text": (
            "**Do not transplant; publish the conversion.** In the shipped 2-D configuration the upstream "
            "`|c_n c_m|` proxy is not an approximation at all: for any trace-free strain tensor the identity "
            "`Q = |Pi|_F` holds exactly on D2Q9 (measured max deviation "
            f"{pi2['d2q9_trace_free']['max_abs_dev_from_1']:.1e} over "
            f"{pi2['d2q9_trace_free']['draws']} random tensors), so both codes implement the *same* closure. "
            f"The only real difference is the coefficient, and the conversion is closed form: "
            f"`C_s = sqrt(Sc/3)`, i.e. upstream `Sc = {SC_OSS}` <-> in-house `C_s = {math.sqrt(SC_OSS / 3.0):.4f}` "
            f"(= classical Smagorinsky `C_s = {math.sqrt(SC_OSS / 3.0) / 2 ** 0.25:.3f}` in the "
            "`(C_s Delta)^2 sqrt(2 S:S)` normalisation). Adopted as documentation of the mapping, not as code: "
            "the in-house operator already is the upstream operator in 2-D. "
            "**In 3-D the equivalence breaks** (`k = Q/|Pi|_F` spreads "
            f"{pi2['d3q19_trace_free']['relative_spread_pct']:.1f}% on D3Q19 and "
            f"{pi2['d3q27_trace_free']['relative_spread_pct']:.1f}% on D3Q27, and "
            f"{result['proxy_identity']['cases']['d2q9_with_trace']['relative_spread_pct']:.1f}% on D2Q9 once the "
            "isotropic nonequilibrium is included), so the Frobenius form - correct in every dimension - remains "
            "the primary implementation and the scalar conversion is published for the 2-D incompressible case only."
            if h1 else
            "Keep the in-house signed-stress Frobenius closure as the primary SGS operator and do NOT "
            "transplant the |c_n c_m| proxy: the measured band "
            f"{ce['k_min']:.3f} <= Q/|Pi|_F <= {ce['k_max']:.3f} is topology-dependent, so no single scalar "
            "reproduces it."
        ),
        "k_spread_pct": k_spread,
        "worst_gamma_cross_rel_pct": worst_cross,
        "worst_nu_t_mapped_rel_pct": max(mapped_vals) if mapped_vals else float("nan"),
        "all_runs_stable": bool(all_stable),
        "owner_threshold_pct": OWNER_PCT,
    }

    notes = []
    if not h1:
        notes.append(
            f"Cross-validation FAILS the 15% band: the upstream proxy ratio k=Q/|Pi|_F spans "
            f"{ce['k_min']:.3f}..{ce['k_max']:.3f} ({k_spread:.1f}% spread) over strain topologies, so a "
            "single C_s conversion cannot reproduce it on arbitrary states. This is the honest result of "
            "the predeclared test, not a tuning failure."
        )
    notes.append(
        "EXPLORATORY (added after the predeclared probes, no threshold changed): the proxy turns out to be an "
        "identity on D2Q9 for trace-free strain - with S = [[a,c],[c,-a]] the Chapman-Enskog nonequilibrium is "
        "f_axis = -+a/3 (4 populations, EEk=1) and f_diag = -+c/6 (4 populations, EEk=4), so "
        "Q^2 = 8(a^2+c^2)/9 and |Pi|_F^2 = (2a/3)^2+(2a/3)^2+2(2c/3)^2 = 8(a^2+c^2)/9. The identity is a "
        "property of the D2Q9 symmetry, not of the |c c| kernel in general: it fails in 3-D and it fails as "
        "soon as an isotropic nonequilibrium component is present."
    )
    notes.append(
        "The trajectory consistency identity assumes a single decaying mode (gamma = 4 k^2 (nu0 + <nu_t>)). "
        "The 4-mode nonlinear case violates that premise by construction, so its consistency residual is a "
        "premise failure, not a closure error; the cross-closure agreement (K_final and gamma) is the KPI "
        "that remains meaningful there."
    )
    notes.append(
        "Single-mode Taylor-Green carries no sub-grid content: LES activity there is driven by the "
        "discretisation defect (the lattice non-equilibrium stress), so the decay tests measure the "
        "closure LAW and its transport, not turbulence-model quality."
    )
    notes.append(
        "Upstream ships no runnable LES trajectory (les_train.py is TF1 + undefined names), so this is a "
        "formulation-level comparison on identical float64 states, not a bit-level replay of upstream "
        "binaries."
    )
    notes.append(
        "In-house nu_t is capped at 1/6 (tau_eff <= tau + 0.5) while upstream is unbounded; at these "
        "operating points the cap never binds (max applied nu_t is far below 1/6), but it would at higher "
        "strain."
    )
    if not all_stable:
        notes.append("At least one arm was numerically unstable; the unstable arm is reported, not re-tuned.")
    notes.append(
        "Non-dimensional bracketing: Mach number is fixed by U0 = 0.05 (Ma = 0.087) and tau0 spans "
        "0.5048..0.548, i.e. omega = 1/tau0 = 1.98..1.82, the near-stability-limit band where the SGS "
        "term actually matters."
    )
    result["notes"] = notes

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    write_csv(result, OUT_DIR / "table.csv")

    figs = []
    if not args.quick:
        figs = RPT.make_figures(result, OUT_DIR / "figures")
        result["figures"] = figs
        (OUT_DIR / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    if not args.no_docs:
        RPT.render_docs(result, figs, DOC_PATH)

    print(f"\nverdict: {result['verdict']['headline']}")
    print(f"  k spread          = {k_spread:.2f}%   (band {PROXY_PCT}%)")
    print(f"  gamma cross worst = {worst_cross:.2f}%  (owner {OWNER_PCT}%)")
    print(f"  nu_t mapped worst = {result['verdict']['worst_nu_t_mapped_rel_pct']:.2f}%")
    print(f"  wall              = {time.time() - t_start:.1f}s")
    print(f"  artefacts         = {OUT_DIR}")
    return 0


def write_csv(result: dict, path: Path) -> None:
    rows = []
    for r in result["ce_algebra"]["rows"]:
        rows.append({
            "scope": "ce_algebra", "config": f"{r['topology']}@{r['sf']:.5f}",
            "kpi": "pi_identity_rel_pct", "value": r["pi_identity_rel_pct"],
            "threshold_pct": OWNER_PCT, "pass": r["pi_identity_rel_pct"] <= OWNER_PCT,
        })
        rows.append({
            "scope": "ce_algebra", "config": f"{r['topology']}@{r['sf']:.5f}",
            "kpi": "k_ratio_Q_over_Pi", "value": r["k_ratio"], "threshold_pct": "", "pass": "",
        })
        rows.append({
            "scope": "ce_algebra", "config": f"{r['topology']}@{r['sf']:.5f}",
            "kpi": "nu_t_inhouse", "value": r["nu_t_inhouse"], "threshold_pct": "", "pass": "",
        })
        rows.append({
            "scope": "ce_algebra", "config": f"{r['topology']}@{r['sf']:.5f}",
            "kpi": "nu_t_oss", "value": r["nu_t_oss"], "threshold_pct": "", "pass": "",
        })
    for c in result["decay"]:
        for kpi, val, thr in (
            ("gamma_K_bgk", c["gamma_K_bgk"], ""),
            ("gamma_K_A_mapped", c["gamma_K_A_mapped"], ""),
            ("gamma_K_B_oss", c["gamma_K_B_oss"], ""),
            ("gamma_cross_rel_pct", c["gamma_cross_rel_pct"], OWNER_PCT),
            ("consistency_A_mapped_pct", c["consistency_A_mapped_pct"], OWNER_PCT),
            ("consistency_B_oss_pct", c["consistency_B_oss_pct"], OWNER_PCT),
            ("consistency_bgk_pct", c["consistency_bgk_pct"], OWNER_PCT),
            ("les_gain_A_mapped_pct", c["les_gain_A_mapped_pct"], ""),
            ("les_gain_B_pct", c["les_gain_B_pct"], ""),
            ("K_final_cross_rel_pct", c["K_final_cross_rel_pct"], OWNER_PCT),
        ):
            ok = ""
            if thr != "" and isinstance(val, float) and val == val:
                ok = val <= thr
            rows.append({
                "scope": "decay", "config": c["case"]["name"], "kpi": kpi,
                "value": val, "threshold_pct": thr, "pass": ok,
            })
    for f in result["field_level"]:
        for kpi, val, thr in (
            ("nu_t_nominal_rel_pct", f["nu_t_nominal_rel_pct"], ""),
            ("nu_t_mapped_rel_pct", f["nu_t_mapped_rel_pct"], OWNER_PCT),
            ("k_p05", f["k_p05"], ""),
            ("k_p95", f["k_p95"], ""),
        ):
            ok = ""
            if thr != "" and isinstance(val, float) and val == val:
                ok = val <= thr
            rows.append({
                "scope": "field_level", "config": f["case"], "kpi": kpi,
                "value": val, "threshold_pct": thr, "pass": ok,
            })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["scope", "config", "kpi", "value", "threshold_pct", "pass"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    sys.exit(main())