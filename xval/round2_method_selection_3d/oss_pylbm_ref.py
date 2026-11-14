#!/usr/bin/env python3
"""OSS reference generator for the 3-D method-selection XVAL.

Anchor: ``XVAL-LBM-ROUND2-METHOD-SELECTION-3D``.

Why this file exists
--------------------
The round-2 audit of m19633 rejected the previous delivery because every number
was hand-copied or never landed in git.  Here the reference is **generated from
the OSS object itself**, not transcribed:

* ``/root/oss-cache/pylbm`` (pylbm, symbolic lattice-Boltzmann scheme builder,
  see ``PROVENANCE`` below for the exact commit / file hashes) builds the
  19x19 moment matrix ``M`` and its inverse ``M^-1`` from a *polynomial* moment
  basis with sympy -- the structure the owner asked us to learn from
  (``Scheme._create_moments_matrices`` in ``pylbm/scheme.py``).
* The quadrature weights are **not** read out of any code base: they are solved
  from the Gaussian moment conditions of the basis
  (``sum_i w_i P_r(v_i) = E_G[P_r]``, ``G = N(0, c_s^2 I)``, ``c_s^2 = 1/3``),
  which is exactly the reason those weights exist.  Solving the 19x19 rational
  system gives them in closed form.

Both products are then compared numerically against TensorLBM in
``kernel_xval_3d.py``.  Nothing in this file is hand-typed except the moment
polynomials, which *are* the method definition (d'Humieres' D3Q19 moment basis)
and are cited in the doc/report.

Usage::

    PYTHONPATH=tools/xval_pylbm_stub:/root/oss-cache/pylbm \
        python3 xval/round2_method_selection_3d/oss_pylbm_ref.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PYLBM_ROOT = Path("/root/oss-cache/pylbm")
OUT_DIR = REPO / "xval" / "round2_method_selection_3d" / "runs"


# --------------------------------------------------------------------------- #
# provenance of the OSS anchor
# --------------------------------------------------------------------------- #
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def oss_provenance() -> dict:
    files = ["pylbm/scheme.py", "pylbm/stencil.py", "pylbm/validator.py"]
    prov = {
        "root": str(PYLBM_ROOT),
        "commit": None,
        "describe": None,
        "files_sha256": {},
    }
    try:
        prov["commit"] = subprocess.run(
            ["git", "-C", str(PYLBM_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        prov["commit_date"] = subprocess.run(
            ["git", "-C", str(PYLBM_ROOT), "log", "-1", "--format=%cI"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        prov["commit"] = "unavailable"
    for f in files:
        p = PYLBM_ROOT / f
        prov["files_sha256"][f] = _sha256(p) if p.exists() else "missing"
    return prov


# --------------------------------------------------------------------------- #
# moment basis (d'Humieres 1992 D3Q19 ordering used by TensorLBM solver3d)
# --------------------------------------------------------------------------- #
def moment_polynomials(sp, X, Y, Z):
    """The 19 moment polynomials, in TensorLBM ``_build_d3q19_mrt_matrices`` order.

    Rows 0-2 energy family, 3-8 momentum/heat-flux family (d'Humieres' ghost
    modes), 9-13 the five deviatoric stresses, 14-18 the third-order ghost
    modes.  Reference: d'Humieres (1992), *Prog. Astronaut. Aeronaut.* 159, 450;
    Lallemand & Luo (2000), *Phys. Rev. E* 61, 6546 (D3Q19 stable basis).
    """
    e2 = X**2 + Y**2 + Z**2
    e4 = e2**2
    return [sp.sympify(x) for x in [
        1,
        19 * e2 - 30,
        sp.Rational(21, 2) * e4 - sp.Rational(53, 2) * e2 + 12,
        X,
        X * (5 * e2 - 9) / 2,
        Y,
        Y * (5 * e2 - 9) / 2,
        Z,
        Z * (5 * e2 - 9) / 2,
        3 * X**2 - e2,
        X**2 - Y**2,
        X * Y,
        X * Z,
        Y * Z,
        (3 * e2 - 5) * (3 * X**2 - e2) / 2,
        (3 * e2 - 5) * (X**2 - Y**2) / 2,
        X**2 * Y,
        X**2 * Z,
        Y**2 * X,
    ]]


MOMENT_LABELS = [
    "rho", "e", "eps", "jx", "qx", "jy", "qy", "jz", "qz",
    "Pxx", "Pww", "Pxy", "Pxz", "Pyz", "Txx", "Tww", "m3xy", "m3xz", "m3yx",
]


def pylbm_matrices() -> dict:
    """Build the pylbm scheme and return velocities / M / M^-1 / parity classes."""
    import numpy as np
    import sympy as sp

    sys.path.insert(0, str(PYLBM_ROOT))
    import pylbm  # noqa: E402

    X, Y, Z = sp.symbols("X Y Z")
    polys = moment_polynomials(sp, X, Y, Z)
    rho, qx, qy, qz = sp.symbols("rho, qx, qy, qz")
    dico = {
        "scheme_velocity": 1,          # LA = 1 -> P(v) == P(c)
        "space_step": 1,
        "dim": 3,
        "schemes": [
            {
                "velocities": list(range(19)),   # pylbm's standard D3Q19 numbering
                "conserved_moments": [rho, qx, qy, qz],
                "polynomials": [rho] + polys[1:],
                "equilibrium": [rho, qx, qy, qz] + [0.0] * 15,
                # relaxation / equilibrium are required by pylbm's validator but
                # are irrelevant here: only the symbolic moment matrix is used.
                "relaxation_parameters": [1.0] * 19,
            }
        ],
    }
    scheme = pylbm.Scheme(dico)
    stencil = scheme.stencil
    vel = np.array([tuple(int(c) for c in v.v) for v in stencil.v[0]], dtype=int)

    # ``M_no_swap`` keeps the *user* polynomial order (pylbm internally permutes
    # the conserved moments to the front -- that permutation is a pylbm
    # convention, not part of the method, so the no-swap pair is the one to align).
    free = set(scheme.M_no_swap.free_symbols)
    subs = {sym: 1 for sym in free}      # scheme_velocity == 1 -> LA = 1
    M_sym = sp.Matrix(scheme.M_no_swap).subs(subs)
    invM_sym = sp.Matrix(scheme.invM_no_swap).subs(subs)
    M = np.array([[float(sp.nsimplify(x)) for x in row] for row in M_sym.tolist()], dtype=np.float64)
    invM = np.array([[float(sp.nsimplify(x)) for x in row] for row in invM_sym.tolist()], dtype=np.float64)
    swapped_rows = [int(i) for i in range(len(polys))]

    # parity of each row under v -> -v: +1 symmetric, -1 antisymmetric, 0 conserved (odd low order)
    parity = []
    for p in polys:
        if p == 1:                      # density row: symmetric by definition
            parity.append(1)
            continue
        pm = sp.expand(p.subs({X: -X, Y: -Y, Z: -Z}, simultaneous=True))
        parity.append(int(sp.simplify(pm - p) == 0) - int(sp.simplify(pm + p) == 0))
    return {
        "velocities": vel,
        "polynomials_str": [str(p) for p in polys],
        "M": M,
        "invM": invM,
        "parity": parity,
        "nschemes": int(scheme.nschemes),
        "free_symbols_in_M": sorted(str(x) for x in free),
    }


# --------------------------------------------------------------------------- #
# quadrature weights solved from the Gaussian moment conditions
# --------------------------------------------------------------------------- #
def gaussian_moment(poly, sp, X, Y, Z) -> sp.Expr:
    """E_G[P(v)] for G = N(0, (1/3) I) -- Wick's theorem, exact rational."""
    expr = sp.expand(sp.Poly(sp.expand(poly), X, Y, Z).as_expr())
    total = sp.Integer(0)
    for monom, coeff in sp.Poly(expr, X, Y, Z).terms():
        a, b, c = monom
        # E[z^n] for N(0, s^2), s^2 = 1/3: (n-1)!! s^n for even n, else 0
        def one(n):
            if n % 2:
                return sp.Integer(0)
            val = sp.Integer(1)
            for k in range(1, n, 2):
                val *= k
            return val * sp.Rational(1, 3) ** (n // 2)
        total += coeff * one(a) * one(b) * one(c)
    return sp.nsimplify(total)


def solve_weights(sp, polys, vel, X, Y, Z) -> list:
    """Solve sum_i w_i P_r(v_i) = E_G[P_r] for the 19 weights (exact rational)."""
    A = sp.Matrix([[sp.Integer(0)] * 19 for _ in range(19)])
    b = sp.zeros(19, 1)
    for r, p in enumerate(polys):
        for i in range(19):
            A[r, i] = sp.expand(p.subs({X: int(vel[i, 0]), Y: int(vel[i, 1]), Z: int(vel[i, 2])}))
        b[r] = gaussian_moment(p, sp, X, Y, Z)
    sol = A.solve(b)
    return [Fraction(int(sp.numer(s)), int(sp.denom(s))) for s in sol]



def build(out_dir: Path = OUT_DIR) -> dict:
    import numpy as np
    import sympy as sp

    ref = pylbm_matrices()
    Xs, Ys, Zs = sp.symbols("X Y Z")
    polys = moment_polynomials(sp, Xs, Ys, Zs)
    w = solve_weights(sp, polys, ref["velocities"], Xs, Ys, Zs)

    payload = {
        "schema": "xval-round2-method-selection-3d/oss-ref/v1",
        "anchor": "XVAL-LBM-ROUND2-METHOD-SELECTION-3D",
        "oss": oss_provenance(),
        "stencil": "D3Q19",
        "scheme_velocity": 1.0,
        "space_step": 1.0,
        "velocities": ref["velocities"].tolist(),
        "polynomials": ref["polynomials_str"],
        "moment_labels": MOMENT_LABELS,
        "parity": ref["parity"],
        "M": ref["M"].tolist(),
        "invM": ref["invM"].tolist(),
        "weights": [float(x) for x in w],
        "weights_exact": [f"{x.numerator}/{x.denominator}" for x in w],
        "checks": {},
    }

    # residual of the weight solve, recomputed numerically from the JSON payload
    M = np.array(payload["M"])
    v = np.array(payload["velocities"])
    wn = np.array(payload["weights"])
    payload["checks"]["weights_sum"] = float(wn.sum())
    payload["checks"]["second_moment_xx"] = float((wn * v[:, 0] ** 2).sum())
    payload["checks"]["fourth_moment_xxyy"] = float((wn * v[:, 0] ** 2 * v[:, 1] ** 2).sum())
    payload["checks"]["M_times_invM_minus_I"] = float(np.abs(M @ np.array(payload["invM"]) - np.eye(19)).max())
    payload["checks"]["M_condition_number"] = float(np.linalg.cond(M))
    payload["checks"]["nschemes"] = ref["nschemes"]
    payload["checks"]["free_symbols_in_M"] = ref["free_symbols_in_M"]

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "oss_pylbm_ref_3d.json"
    out.write_text(json.dumps(payload, indent=1))
    print(f"wrote {out}")
    print("  oss commit           :", payload["oss"]["commit"])
    print("  weights (exact)      :", payload["weights_exact"][:6], "...")
    print("  |M M^-1 - I|_max     : %.3e" % payload["checks"]["M_times_invM_minus_I"])
    print("  cond(M)              : %.3e" % payload["checks"]["M_condition_number"])
    print("  sum w                : %.17g" % payload["checks"]["weights_sum"])
    return payload


if __name__ == "__main__":
    build()