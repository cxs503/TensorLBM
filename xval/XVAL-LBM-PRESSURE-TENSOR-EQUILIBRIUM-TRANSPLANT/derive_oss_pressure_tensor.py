#!/usr/bin/env python3
"""Derive the discrete pressure tensor realised by the OSS (sailfish) free-energy
equilibrium, symbolically, straight from the OSS source text.

Provenance (zero hand-copy)
---------------------------
* ``ref/sailfish_src/sailfish/sym_equilibrium.py:16-49`` --
  ``free_energy_equilibrium_fluid``: the D2Q9/D3Q19 equilibrium whose second
  moment is the pressure tensor.  The expression transcribed below is a literal
  reading of that function; the code block is printed with file:line from the
  vendored copy at run time so the transcription cannot silently drift.
* ``ref/sailfish_src/sailfish/lb_binary.py:224-259`` -- the D2Q9 ``wi/wxx/wyy/wxy``
  coefficient tables (``if x.dot(x) == 1: wi=1/3 ... else: wi=1/12 ...``).
* ``ref/sailfish_src/sailfish/templates/finite_difference_optimized.mako:46-48``
  -- the nine-point isotropic ``grad``/``lap`` operators.

The script does *not* assume the continuum pressure tensor.  It computes

    P_ab = sum_i f_i^eq(rho=?, u=0) c_ia c_ib   and   M_ab = sum_i f_i^eq c_ia c_ib - rho u_a u_b

and then asks sympy ``simplify``/``Poly`` whether the result is proportional to
the continuum free-energy pressure tensor

    P^cont_ab = [pb - kappa phi lap(phi)] delta_ab + kappa (d_a phi d_b phi - 1/2 |grad phi|^2 delta_ab)

with ``pb = rho/3 + A(-phi^2/2 + 3 phi^4/4)``.  Any mismatch is reported as an
exact coefficient pair ``(c_xx, c_xy)`` -- that is the number the transplant has
to reproduce, and it is what the XVAL report quotes.

Run: ``python xval/XVAL-LBM-PRESSURE-TENSOR-EQUILIBRIUM-TRANSPLANT/derive_oss_pressure_tensor.py``
"""
from __future__ import annotations

import json
import os
import re

import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "ref", "sailfish_src")

# D2Q9 basis in sailfish order (sym.D2Q9.basis): (0,0) then axes then diagonals.
BASIS = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)]


def oss_d2q9_coefficients():
    """D2Q9 ``wi/wxx/wyy/wxy`` exactly as ``lb_binary.py:224-259`` builds them."""
    wi, wxx, wyy = [], [], []
    for (x, y) in BASIS[1:]:
        if x * x + y * y == 1:
            wi.append(sp.Rational(1, 3))
            wxx.append(sp.Rational(1, 3) if abs(x) == 1 else -sp.Rational(1, 6))
            wyy.append(sp.Rational(1, 3) if abs(y) == 1 else -sp.Rational(1, 6))
        else:
            wi.append(sp.Rational(1, 12))
            wxx.append(-sp.Rational(1, 24))
            wyy.append(-sp.Rational(1, 24))
    wxy = [sp.Rational(x * y, 4) for (x, y) in BASIS[1:]]
    return wi, wxx, wyy, wxy


def oss_equilibrium(rho, ux, uy, phi, gx, gy, lap, A, kappa, wi, wxx, wyy, wxy):
    """Literal transcription of ``sym_equilibrium.py:16-49`` for D2Q9, lambda=0."""
    # lambda_ = S.visc * (1 - cssq*3) == 0 on D2Q9  -> grad0 terms drop out.
    pb = rho / 3 + A * (-(phi**2) / 2 + sp.Rational(3, 4) * phi**4)
    out = []
    t_sum = 0
    for i, (ex, ey) in enumerate(BASIS[1:]):
        eu = ex * ux + ey * uy
        t = (wi[i] * (pb - kappa * phi * lap + rho * eu)
             + sp.Rational(3, 2) * (wi[i] * rho * eu**2
                                    - sp.Rational(1, 3) * wi[i] * rho * (ux * ux + uy * uy))
             + kappa * (wxx[i] * gx**2 + wyy[i] * gy**2 + wxy[i] * gx * gy))
        t_sum += t
        out.append(sp.expand(t))
    return [sp.expand(rho - t_sum)] + out  # f_0 chosen so that rho is conserved


def main() -> int:
    src = open(os.path.join(REF, "sailfish", "sym_equilibrium.py")).read().splitlines()
    fn = next(i for i, l in enumerate(src) if l.startswith("def free_energy_equilibrium_fluid"))
    print("OSS source excerpt (sym_equilibrium.py:%d-%d):" % (fn + 1, fn + 35))
    for i in range(fn, fn + 35):
        print("  %4d| %s" % (i + 1, src[i]))
    print()

    rho, ux, uy, phi, gx, gy, lap = sp.symbols("rho ux uy phi gx gy lap")
    A, kappa = sp.symbols("A kappa", positive=True)
    wi, wxx, wyy, wxy = oss_d2q9_coefficients()

    # --- A. second moment with u = 0 : the pressure tensor -------------------
    f0 = oss_equilibrium(rho, sp.Integer(0), sp.Integer(0), phi, gx, gy, lap, A, kappa,
                         wi, wxx, wyy, wxy)
    Pxx = sp.simplify(sum(f0[i] * BASIS[i][0] ** 2 for i in range(9)))
    Pyy = sp.simplify(sum(f0[i] * BASIS[i][1] ** 2 for i in range(9)))
    Pxy = sp.simplify(sum(f0[i] * BASIS[i][0] * BASIS[i][1] for i in range(9)))

    pb = rho / 3 + A * (-(phi**2) / 2 + sp.Rational(3, 4) * phi**4)
    cont_xx = pb - kappa * phi * lap + kappa * (gx**2 - (gx**2 + gy**2) / 2)
    cont_yy = pb - kappa * phi * lap + kappa * (gy**2 - (gx**2 + gy**2) / 2)
    cont_xy = kappa * gx * gy

    print("discrete  Pxx =", sp.expand(Pxx))
    print("continuum Pxx =", sp.expand(cont_xx))
    print("discrete  Pxy =", sp.expand(Pxy))
    print("continuum Pxy =", sp.expand(cont_xy))

    def collect_derivative_coeffs(expr, name):
        # coefficients of gx^2, gy^2, gx*gy, phi*lap, rho, phi^2, phi^4 ...
        return {str(s): sp.simplify(expr.coeff(s)) for s in (gx**2, gy**2, gx * gy, phi * lap, rho)}

    res = {
        "pxx": {k: str(v) for k, v in collect_derivative_coeffs(Pxx, "Pxx").items()},
        "pxy": {k: str(v) for k, v in collect_derivative_coeffs(Pxy, "Pxy").items()},
        "continuum_pxx": {k: str(v) for k, v in collect_derivative_coeffs(cont_xx, "cxx").items()},
        "continuum_pxy": {k: str(v) for k, v in collect_derivative_coeffs(cont_xy, "cxy").items()},
    }
    # exact residuals against the continuum tensor
    res["residual_pxx"] = str(sp.simplify(Pxx - cont_xx))
    res["residual_pyy"] = str(sp.simplify(Pyy - cont_yy))
    res["residual_pxy"] = str(sp.simplify(Pxy - cont_xy))
    res["trace_isotropic"] = str(sp.simplify(Pxx - Pyy))
    print("\nresidual Pxx-cont =", res["residual_pxx"])
    print("residual Pyy-cont =", res["residual_pyy"])
    print("residual Pxy-cont =", res["residual_pxy"])
    print("Pxx-Pyy          =", res["trace_isotropic"])

    # --- B. velocity part: full second moment minus rho u_a u_b -------------
    fv = oss_equilibrium(rho, ux, uy, phi, gx, gy, lap, A, kappa, wi, wxx, wyy, wxy)
    Mxx = sp.simplify(sum(fv[i] * BASIS[i][0] ** 2 for i in range(9)))
    Mxy = sp.simplify(sum(fv[i] * BASIS[i][0] * BASIS[i][1] for i in range(9)))
    res["momentum_xx_residual"] = str(sp.simplify(Mxx - Pxx - rho * ux * ux))
    res["momentum_xy_residual"] = str(sp.simplify(Mxy - Pxy - rho * ux * uy))
    print("\nMxx - Pxx - rho ux^2 =", res["momentum_xx_residual"])
    print("Mxy - Pxy - rho ux uy =", res["momentum_xy_residual"])

    # --- C. order-parameter equilibrium (sym_equilibrium.py:51-83) ----------
    mu, Gamma = sp.symbols("mu Gamma")
    g = []
    t_sum = 0
    for i, (ex, ey) in enumerate(BASIS[1:]):
        eu = ex * ux + ey * uy
        t = wi[i] * (Gamma * mu + eu * phi
                     + sp.Rational(3, 2) * phi * (-sp.Rational(1, 3) * (ux * ux + uy * uy) + eu**2))
        t_sum += t
        g.append(sp.expand(t))
    g = [sp.expand(phi - t_sum)] + g
    G0 = sp.simplify(sum(g))
    Gx = sp.simplify(sum(g[i] * BASIS[i][0] for i in range(9)))
    res["order_param_closure"] = {"sum_g": str(G0), "sum_g_cx": str(Gx)}
    print("\nsum_i g_i =", G0, " sum_i g_i c_ix =", Gx)

    out = os.path.join(HERE, "ref", "oss_discrete_pressure_tensor.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2, sort_keys=True)
    print("\nwrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())