#!/usr/bin/env python3
"""Restored experiment harness for XVAL-LBM-TRT-LAMBDA-MAGIC-TABLE-DRIVEN (R1, m18452).

Why this file exists
--------------------
R1 (m18452) merged its report and its reference data but **not** the driver named in
that report (`xval_trt_lambda.py`), nor `predeclared.json`, nor the vendored
`oss_src/`.  `tests/test_xval_trt_lambda_magic_table_driven.py` therefore could not
run at all (FileNotFoundError on this module).  Round 2 (m23150) restores an
equivalent harness that needs **no OSS source and no g++**: the OSS reference
numbers it compares against are the `.dat` fields already committed under
``gaussian_1d_magic{12,6}_hi/`` (the ``precision(17)`` instrumented build recorded in
the R1 report).  Against those committed fields the reconstruction reproduces the
OSS binary to **4.44e-16** (see ``results.json: k3``), which is what licenses the
claim that the constants below are the OSS case parameters rather than a guess.

The OSS C++ sources were downloaded from GitHub but never committed; family-level
OSS inventory lives in ``docs/oss-inventory/REPORT_lbm-principles-practice-code.md``.
The case constants are transcribed from the R1 report (its parameter table, section 3)
and are *validated*, not assumed: ``k3`` re-derives the fields from them.

Gates (thresholds transcribed from the R1 report section 2 into ``predeclared.json``):

  K1  magic relation identity, three OSS writings vs this repo     <= 1e-15
  K2  discrete table identity (weights / complement maps)          == 1.0
  K3  field identity vs the committed OSS reference fields         <= 1e-13
  K4  analytic accuracy of both arms (periodic heat kernel)        <= 0.15
  K5  Lambda scan on the ADE arm                                   report only
  K6  refinement order                                             report only
  K7  the opt-in u^2 equilibrium term at u = 0.5                   == 0.15 band

Usage:  PYTHONPATH=src python3 xval_trt_lambda.py
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

from tensorlbm import scalar_transport as st  # noqa: E402
from tensorlbm.d2q9 import OPPOSITE as OPP_D2Q9  # noqa: E402
from tensorlbm.d2q9 import W as W_D2Q9  # noqa: E402

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# OSS case constants (transcribed from the R1 report parameter table; the K3
# identity below is what certifies them against the committed reference fields)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OssCase:
    label: str
    oss_file: str
    nx: int
    n_steps: int
    sigma0: float
    ux: float
    magic: float
    omega_hand: float          # the *hand-set* relaxation rate in the OSS source
    hand_branch: str           # which TRT branch that hand-set rate relaxes
    equilibrium: str           # "linear" (u only) or "oss" (u + u^2 term)

    @property
    def n_frames(self) -> int:
        return self.n_steps + 1

    @property
    def tau_hand(self) -> float:
        return 1.0 / self.omega_hand

    @property
    def tau_odd(self) -> float:
        """tau of the branch that sets D: the odd (anti-symmetric) one, both arms.

        Verified, not assumed: the alternative attribution (even branch sets D)
        misses the committed reference fields by ~31% (``k3_d_branch_reversal``).
        """
        if self.hand_branch == "odd":
            return self.tau_hand
        return st.tau_symmetric_from_magic(self.tau_hand, self.magic)

    @property
    def tau_sym(self) -> float:
        if self.hand_branch == "even":
            return self.tau_hand
        return st.tau_symmetric_from_magic(self.tau_odd, self.magic)

    @property
    def derived_rate(self) -> float:
        """The OSS ``derived`` relaxation rate = 1/tau of the D-setting (odd) branch."""
        return 1.0 / self.tau_odd

    @property
    def diffusivity(self) -> float:
        """c_s^2 (tau_odd - 1/2) with c_s^2 = 1/3 on D2Q5."""
        return (self.tau_odd - 0.5) / 3.0


OSS_CASES: dict[str, OssCase] = {
    "magic12": OssCase("magic12", "chapter8/gaussian_1d_magic12.cpp", 50, 20, 3.0, 0.0,
                       1.0 / 12.0, 0.5, "odd", "linear"),
    "magic6": OssCase("magic6", "chapter8/gaussian_1d_magic6.cpp", 50, 10, 3.0, 0.5,
                      1.0 / 6.0, 0.3, "even", "oss"),
}

REFERENCE_DIRS = {"magic12": "gaussian_1d_magic12_hi", "magic6": "gaussian_1d_magic6_hi"}
REFERENCE_STEMS = {"magic12": "gaussian_1d_magic12", "magic6": "gaussian_1d_magic6"}


def parse_oss_1d(name: str, label: str) -> OssCase:
    """The case record for an OSS 1D demo. ``name`` is the .cpp basename (API compat)."""
    if label not in OSS_CASES:
        raise KeyError(f"unknown OSS demo {label!r}")
    case = OSS_CASES[label]
    if name != case.oss_file.rsplit("/", 1)[-1]:
        raise KeyError(f"{label!r} is {case.oss_file}, not {name!r}")
    return case


def load_reference(label: str) -> np.ndarray:
    """Committed OSS reference fields (precision(17) build), shape (n_frames, nx)."""
    d = HERE / REFERENCE_DIRS[label]
    stem = REFERENCE_STEMS[label]
    files = sorted(d.glob(f"{stem}*.dat"))
    if not files:
        raise FileNotFoundError(f"no committed reference fields under {d}")
    return np.vstack([[float(t) for t in f.read_text().split()] for f in files])


def initial_condition(case: OssCase) -> np.ndarray:
    x = np.arange(case.nx)
    c0 = np.exp(-0.5 * (x - case.nx // 2) ** 2 / case.sigma0 ** 2)
    c0[np.abs(x - case.nx // 2) > int(3 * case.sigma0)] = 0.0
    return c0


def run_tb(case: OssCase, diffusivity: float | None = None, magic: float | None = None,
           equilibrium: str | None = None) -> np.ndarray:
    """Run the repo's D2Q5 TRT scalar solver on the OSS case; returns (n_frames, nx)."""
    diffusivity = case.diffusivity if diffusivity is None else diffusivity
    magic = case.magic if magic is None else magic
    equilibrium = case.equilibrium if equilibrium is None else equilibrium
    tau_odd = 0.5 + 3.0 * diffusivity
    tau_sym = st.tau_symmetric_from_magic(tau_odd, magic)
    nx = case.nx
    c = torch.tensor(initial_condition(case), dtype=torch.float64).view(1, 1, nx)
    ux = torch.full((1, nx), case.ux, dtype=torch.float64)
    uy = torch.zeros((1, nx), dtype=torch.float64)
    second_order = equilibrium == "oss"
    g = st.equilibrium_scalar_d2q5(c, ux, uy, second_order=second_order)
    frames = [st.scalar_field(g).numpy().ravel()]
    for _ in range(case.n_steps):
        g = st.collide_scalar_trt(g, c, ux, uy, tau_odd, tau_sym, second_order)
        g, _ = st.stream_scalar(g, "periodic", "periodic")
        c = st.scalar_field(g)
        frames.append(c.numpy().ravel())
    return np.vstack(frames)


def exact_periodic(case: OssCase, t: float, diffusivity: float | None = None) -> np.ndarray:
    """Exact solution of the periodic ADE for the *truncated* Gaussian initial data.

    c(x,t) = sum_n int_{-3 sigma0}^{3 sigma0} G_sigma0(xi) G_sqrt(2Dt)(x - u t - xi + n nx) dxi

    Each term is the product of two Gaussians evaluated in the numerically stable
    "completed square" form (no exp overflow at small 2Dt):

        prod = Normal(y; 0, s0^2 + st^2) * Normal(xi; xi*, seff^2),
        seff^-2 = s0^-2 + st^-2,  xi* = y s0^2 / (s0^2 + st^2),

    so the integral over the truncation window is one erf difference.  Mirrors
    n = -3..3 cover the periodic wrap of a 50-node ring.
    """
    D = case.diffusivity if diffusivity is None else diffusivity
    x = np.arange(case.nx, dtype=float)
    if t == 0:
        return initial_condition(case)
    s0sq = case.sigma0 ** 2
    stsq = 2.0 * D * t
    S = s0sq + stsq
    seff = math.sqrt(s0sq * stsq / S)
    centre = case.nx // 2
    shift = case.ux * t
    L = 3.0 * case.sigma0
    out = np.zeros_like(x)
    for xv in x:
        acc = 0.0
        for n in range(-3, 4):
            y = xv - shift + n * case.nx - centre
            xi_star = y * s0sq / S
            # the initial data is exp(-xi^2/2s0^2), NOT a unit-normalised density:
            # the seff/st prefactor carries its (2 pi s0^2)^(1/2) mass.
            acc += (seff / math.sqrt(stsq)) * math.exp(-y * y / (2.0 * S)) * 0.5 * (
                math.erf((L - xi_star) / (math.sqrt(2.0) * seff))
                + math.erf((L + xi_star) / (math.sqrt(2.0) * seff)))
        out[int(xv)] = acc
    return out


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# --------------------------------------------------------------------------- #
# gates
# --------------------------------------------------------------------------- #
def k1_magic_relation() -> dict:
    """K1: the three OSS writings of the magic relation vs the repo relation."""
    lambdas = (1.0 / 12.0, 1.0 / 6.0, 3.0 / 16.0, 0.25, 0.375)
    omegas = np.linspace(0.51, 1.99, 75)
    diffs = {"oss_1d_solve_minus": 0.0, "oss_1d_solve_plus": 0.0,
             "oss_2d_solve_plus": 0.0}
    rt = 0.0
    for lam in lambdas:
        for w_hand in omegas:
            # OSS 1D magic6: hand-set omega is the PLUS rate, minus is solved
            om_minus_1d = 1.0 / (lam / (1.0 / w_hand - 0.5) + 0.5)
            ref_minus = 1.0 / st.tau_symmetric_from_magic(1.0 / w_hand, lam)
            diffs["oss_1d_solve_minus"] = max(diffs["oss_1d_solve_minus"],
                                              abs(om_minus_1d - ref_minus))
            # OSS 1D magic12: hand-set omega is the MINUS rate, plus is solved
            om_plus_1d = 1.0 / (lam / (1.0 / w_hand - 0.5) + 0.5)
            ref_plus = 1.0 / st.tau_symmetric_from_magic(1.0 / w_hand, lam)
            diffs["oss_1d_solve_plus"] = max(diffs["oss_1d_solve_plus"],
                                             abs(om_plus_1d - ref_plus))
            # OSS 2D gaussian_2d_trt: omegaplus = 1/(0.5 + lambda*omegaminus/(1-0.5*omegaminus))
            om_minus = w_hand
            om_plus_2d = 1.0 / (0.5 + lam * om_minus / (1.0 - 0.5 * om_minus))
            ref_plus2 = 1.0 / st.tau_symmetric_from_magic(1.0 / om_minus, lam)
            diffs["oss_2d_solve_plus"] = max(diffs["oss_2d_solve_plus"],
                                             abs(om_plus_2d - ref_plus2))
            # round trip Lambda -> tau -> Lambda
            tau_s = st.tau_symmetric_from_magic(1.0 / w_hand, lam)
            rt = max(rt, abs(st.magic_from_taus(tau_s, 1.0 / w_hand) - lam))
    return {"max_abs_omega_diff": diffs, "max_abs_roundtrip_lambda_diff": rt,
            "n_points": len(lambdas) * len(omegas)}


def k2_tables() -> dict:
    """K2: discrete tables (weights / complement maps) are the same objects."""
    # 1D rest weight: OSS D1Q3 rest weight 2/3 vs D2Q5 reduced to one row
    w = st.W_D2Q5 if hasattr(st, "W_D2Q5") else None
    if w is None:  # pragma: no cover - fall back to the literal D2Q5 weights
        w = torch.tensor([1.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    rest = float(w[0])
    perp = float(w[1] + w[2] + w[3] + w[4])           # the y-pair freezes out (ny=1)
    w_identity = 1.0 if abs((rest + perp) - (2.0 / 3.0 + 1.0 / 3.0)) < 1e-15 else 0.0
    # value at a y-frozen row: rest 1/3 + x-pair 1/6+1/6 = 2/3 == OSS D1Q3 rest weight
    # OPP_D2Q5 == (0,3,4,1,2) => ordering (rest, +x, +y, -x, -y); the x-pair is (1,3)
    w_identity_1d = 1.0 if abs((rest + float(w[1] + w[3])) - 2.0 / 3.0) < 1e-15 else 0.0
    opp_1d = tuple(int(i) for i in st.OPP_D2Q5)
    opp_map_1d = 1.0 if opp_1d == (0, 3, 4, 1, 2) else 0.0
    opp_map_2d = 1.0 if tuple(int(i) for i in OPP_D2Q9) == (0, 3, 4, 1, 2, 7, 8, 5, 6) else 0.0
    d2q9_weights_ok = 1.0 if abs(float(W_D2Q9.sum()) - 1.0) < 1e-15 else 0.0
    o9 = np.asarray(OPP_D2Q9)
    d2q9_inv = 1.0 if np.array_equal(o9[o9], np.arange(9)) else 0.0
    try:
        from tensorlbm.d3q27 import OPPOSITE as OPP_D3Q27
        o27 = np.asarray(OPP_D3Q27)
        d3q27_inv = 1.0 if np.array_equal(o27[o27], np.arange(27)) else 0.0
    except Exception:  # pragma: no cover
        d3q27_inv = 0.0
    return {"w_identity_1d": w_identity_1d, "w_identity_total": w_identity,
            "opp_map_1d": opp_map_1d, "opp_map_2d_d2q9": opp_map_2d,
            "d2q9_weights_ok": d2q9_weights_ok,
            "d2q9_opposite_involutive": d2q9_inv,
            "d3q27_opposite_involutive": d3q27_inv}


def k3_identity() -> dict:
    """K3: repo field vs the committed OSS reference fields, both arms."""
    out = {}
    for label in ("magic12", "magic6"):
        case = OSS_CASES[label]
        got = run_tb(case)
        want = load_reference(label)
        out[label] = {
            "max_abs": float(np.abs(got - want).max()),
            "rel_l2": rel_l2(got.ravel(), want.ravel()),
            "shape": list(got.shape),
            "equilibrium": case.equilibrium,
        }
    # counterfactual: attribute the diffusivity to the even branch instead
    case = OSS_CASES["magic12"]
    alt = run_tb(case, diffusivity=(case.tau_sym - 0.5) / 3.0)
    out["k3_d_branch_reversal"] = {
        "max_abs": float(np.abs(alt - load_reference("magic12")).max()),
        "rel_l2": rel_l2(alt.ravel(), load_reference("magic12").ravel()),
        "note": "wrong branch attribution misses by ~31% -> the odd branch sets D",
    }
    return out


def k4_analytic() -> dict:
    """K4: relative L2 vs the exact periodic heat-kernel solution, max over t."""
    out = {}
    for label in ("magic12", "magic6"):
        case = OSS_CASES[label]
        got = run_tb(case)
        errs = []
        for t in range(1, case.n_frames):
            exact = exact_periodic(case, float(t), case.diffusivity)
            errs.append(rel_l2(got[t], exact))
        out[label] = {"max_over_t": max(errs), "final": errs[-1], "n_steps": case.n_steps}
    return out


def k5_lambda_scan() -> dict:
    """K5 (report only): ADE field error vs lambda on the magic12 arm."""
    case = OSS_CASES["magic12"]
    rows = {}
    for lam in (1.0 / 12.0, 1.0 / 6.0, 3.0 / 16.0, 0.25, 0.375):
        got = run_tb(case, magic=lam)
        exact = exact_periodic(case, float(case.n_steps), case.diffusivity)
        rows[f"{lam:.8f}"] = {"rel_l2_final": rel_l2(got[-1], exact),
                              "max_over_t": max(rel_l2(got[t], exact_periodic(case, float(t)))
                                                for t in range(1, case.n_frames))}
    rows["bgk_single_rate"] = {
        "rel_l2_final": rel_l2(_run_bgk(case)[-1],
                               exact_periodic(case, float(case.n_steps), case.diffusivity))}
    return rows


def _run_bgk(case: OssCase) -> np.ndarray:
    nx = case.nx
    tau = case.tau_odd
    c = torch.tensor(initial_condition(case), dtype=torch.float64).view(1, 1, nx)
    ux = torch.full((1, nx), case.ux, dtype=torch.float64)
    uy = torch.zeros((1, nx), dtype=torch.float64)
    g = st.equilibrium_scalar_d2q5(c, ux, uy)
    frames = [st.scalar_field(g).numpy().ravel()]
    for _ in range(case.n_steps):
        g = st.collide_scalar_trt(g, c, ux, uy, tau, tau)
        g, _ = st.stream_scalar(g, "periodic", "periodic")
        c = st.scalar_field(g)
        frames.append(c.numpy().ravel())
    return np.vstack(frames)


def k6_refinement() -> dict:
    """K6 (report only): diffusive-scaling refinement of the ADE arm, u = 0."""
    D_phys, T_phys, L_phys, sig = 0.5, 1.0, 50.0, 3.0
    rows = []
    for factor in (1, 2, 4):
        nx = 50 * factor
        dx = L_phys / nx
        dt = dx ** 2 * D_phys / D_phys       # keep lattice D == D_phys
        n_steps = int(round(T_phys / dt))
        xx = np.arange(nx) * dx
        c0 = np.exp(-0.5 * (xx - L_phys / 2) ** 2 / sig ** 2)
        c0[np.abs(xx - L_phys / 2) > 3 * sig] = 0.0
        c = torch.tensor(c0, dtype=torch.float64).view(1, 1, nx)
        ux = torch.zeros((1, nx), dtype=torch.float64)
        uy = torch.zeros((1, nx), dtype=torch.float64)
        tau_odd = 0.5 + 3.0 * D_phys
        tau_sym = st.tau_symmetric_from_magic(tau_odd, 1.0 / 12.0)
        g = st.equilibrium_scalar_d2q5(c, ux, uy)
        for _ in range(n_steps):
            g = st.collide_scalar_trt(g, c, ux, uy, tau_odd, tau_sym)
            g, _ = st.stream_scalar(g, "periodic", "periodic")
            c = st.scalar_field(g)
        # exact on the same physical domain at t = T_phys (periodic mirrors)
        exact = _exact_periodic_generic(xx, sig, T_phys, D_phys, L_phys)
        rows.append({"nx": nx, "n_steps": n_steps,
                     "rel_l2_pct": rel_l2(c.numpy().ravel(), exact) * 100.0})
    orders = []
    for a, b in zip(rows[:-1], rows[1:]):
        orders.append(math.log(a["rel_l2_pct"] / b["rel_l2_pct"]) / math.log(b["nx"] / a["nx"]))
    return {"rows": rows, "observed_order": orders}


def _exact_periodic_generic(xx, sig, t, D, L):
    """Same closed form as :func:`exact_periodic`, for an arbitrary physical grid."""
    s0sq = sig ** 2
    stsq = 2.0 * D * t
    S = s0sq + stsq
    seff = math.sqrt(s0sq * stsq / S)
    out = np.zeros_like(np.asarray(xx, dtype=float))
    for i, xv in enumerate(np.asarray(xx, dtype=float)):
        acc = 0.0
        for n in range(-4, 5):
            y = xv + n * L - L / 2.0
            xi_star = y * s0sq / S
            acc += (seff / math.sqrt(stsq)) * math.exp(-y * y / (2.0 * S)) * 0.5 * (
                math.erf((3 * sig - xi_star) / (math.sqrt(2.0) * seff))
                + math.erf((3 * sig + xi_star) / (math.sqrt(2.0) * seff)))
        out[i] = acc
    return out


def k7_u2_term() -> dict:
    """K7: the opt-in u^2 equilibrium term on the advected (magic6) arm at u = 0.5."""
    case = OSS_CASES["magic6"]
    want = load_reference("magic6")
    got = run_tb(case, equilibrium="oss")
    lin = run_tb(case, equilibrium="linear")
    exact = exact_periodic(case, float(case.n_steps), case.diffusivity)
    return {
        "max_abs_vs_oss_second_order": float(np.abs(got - want).max()),
        "max_abs_oss_vs_linear": float(np.abs(want - lin).max()),
        "rel_l2_final_linear": rel_l2(lin[-1], exact),
        "rel_l2_final_second_order": rel_l2(got[-1], exact),
    }


def main() -> int:
    results = {
        "case": "xval_trt_lambda_r1_restored",
        "anchor": "XVAL-LBM-TRT-LAMBDA-MAGIC-TABLE-DRIVEN",
        "restored_by": "mission 23150 (round 2)",
        "reference_source": "committed OSS fields (precision(17) build) under "
                            "gaussian_1d_magic{12,6}_hi/; OSS C++ sources absent from the tree",
        "k1": k1_magic_relation(),
        "k2": k2_tables(),
        "k3": k3_identity(),
        "k4": k4_analytic(),
        "k5": k5_lambda_scan(),
        "k6": k6_refinement(),
        "k7": k7_u2_term(),
    }
    (HERE / "results.json").write_text(json.dumps(results, indent=1))
    print(json.dumps({k: results[k] for k in ("k3", "k4", "k7")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())