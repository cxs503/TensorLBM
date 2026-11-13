"""Exact linear (unsteady Stokes) solution for a circular cylinder oscillating in a
quiescent viscous fluid: complex force coefficient as a function of beta.

Derivation (in-code, no borrowed formula):
  linearised vorticity equation  d(omega)/dt = nu lap(omega),
  omega = W(r) cos(theta) exp(-i w t)  ->  W = A K_1(k r),  k^2 = -i w / nu
  stream function psi = f(r) sin(theta) exp(-i w t),  L f = -W,
      L = d2/dr2 + (1/r) d/dr - 1/r2   ->  f = C2/r - (A/k^2) K_1(k r)
  no-slip on r=a (no rotation): f(a) = U a, f'(a) = U
      =>  A = 2 U k / K_0(k a),      C2 = U a^2 + (A a/k^2) K_1(k a)
  pressure mode p = P(r) cos(theta) exp(-i w t) from the radial momentum equation,
  integrated outwards from P(inf) = 0:
      P'(r) = i w rho V_r + mu [V_r'' + V_r'/r - V_r/r^2 - 2 V_theta/r^2]
  force per unit depth  F_x = pi a [ (-P + 2 mu V_r') + mu (V_r/a - V_theta' + V_theta/a) ]_{r=a}
"""
from __future__ import annotations

import math
from typing import Any

import mpmath as mp

__all__ = ["oscillating_cylinder_force", "added_mass_from_force", "beta_from"]


def beta_from(D: float, nu: float, T: float) -> float:
    """Stokes number beta = D^2/(nu T)."""
    return D * D / (nu * T)


def _fields(U: float, a: float, omega: float, nu: float, r: float, dps: int = 40):
    """Complex V_r, V_theta and their r-derivatives at radius r."""
    mp.mp.dps = dps
    w = mp.mpf(omega)
    nu = mp.mpf(nu)
    a = mp.mpf(a)
    rr = mp.mpf(r)
    k = mp.sqrt(-1j * w / nu)
    ka = k * a
    U = mp.mpf(U)
    A = 2 * U * k / mp.besselk(0, ka)
    C2 = U * a**2 + (A * a / k**2) * mp.besselk(1, ka)

    def K1(x):
        return mp.besselk(1, k * x)

    def K1p(x):  # d/dr K_1(k r) = k K_1'(k r) = -k (K_0(kr) + K_1(kr)/(kr))
        return -k * (mp.besselk(0, k * x) + mp.besselk(1, k * x) / (k * x))

    def f(x):
        return C2 / x - (A / k**2) * K1(x)

    def fp(x):
        return -C2 / x**2 - (A / k**2) * K1p(x)

    def fpp(x):
        # g = K_1(k x):  g'' = k^2 g - g'/x + g/x^2
        g = K1(x)
        gp = K1p(x)
        gpp = k**2 * g - gp / x + g / x**2
        return 2 * C2 / x**3 - (A / k**2) * gpp

    Vr = f(rr) / rr
    Vt = -fp(rr)
    Vrp = fp(rr) / rr - f(rr) / rr**2
    Vrpp = fpp(rr) / rr - 2 * fp(rr) / rr**2 + 2 * f(rr) / rr**3
    Vtp = -fpp(rr)
    return dict(Vr=Vr, Vt=Vt, Vrp=Vrp, Vrpp=Vrpp, Vtp=Vtp, k=k, A=A, C2=C2)


def _pressure_at(U: float, a: float, omega: float, nu: float, rho: float = 1.0,
                 dps: int = 25, path: float = 8.0, n: int = 48) -> Any:
    """P(a) from integrating P'(r) outward (P(inf)=0) with a decaying tail cut-off."""
    mp.mp.dps = dps
    w = mp.mpf(omega)
    mu = mp.mpf(rho) * mp.mpf(nu)
    R0 = mp.mpf(a)
    R1 = mp.mpf(a) + mp.mpf(path) * (mp.mpf(a) + 19 * mp.sqrt(2 * mp.mpf(nu) / w))
    h = (R1 - R0) / n
    integral = mp.mpf(0)
    for i in range(n):
        for x, wgt in ((R0 + i * h, mp.mpf(1)), (R0 + (i + 1) * h, mp.mpf(1))):
            fl = _fields(U, a, omega, nu, x, dps)
            Pp = 1j * w * mp.mpf(rho) * fl["Vr"] + mu * (
                fl["Vrpp"] + fl["Vrp"] / x - fl["Vr"] / x**2 - 2 * fl["Vt"] / x**2
            )
            integral += 0.5 * h * wgt * Pp
    return -integral


def oscillating_cylinder_force(a: float, omega: float, nu: float, U: float = 1.0,
                               rho: float = 1.0, dps: int = 25, n_int: int = 48) -> dict[str, Any]:
    """Complex force per unit depth for u_body = U exp(-i w t), fluid at rest at infinity."""
    mp.mp.dps = dps
    fl = _fields(U, a, omega, nu, a, dps)
    P = _pressure_at(U, a, omega, nu, rho, dps, n=n_int)
    mu = mp.mpf(rho) * mp.mpf(nu)
    bracket = (-P + 2 * mu * fl["Vrp"]) + mu * (
        fl["Vr"] / mp.mpf(a) - fl["Vtp"] + fl["Vt"] / mp.mpf(a)
    )
    Z = mp.pi * mp.mpf(a) * bracket
    beta = float(mp.mpf(a) ** 2 * 2 * mp.pi / (mp.mpf(nu) * (2 * mp.pi / mp.mpf(omega))))
    b = 4.0 * a * a * omega / (2 * nu)  # beta = D^2/(nu T) with D=2a, T=2pi/omega -> 4a^2/ (nu T)
    beta = (2 * a) ** 2 / (float(nu) * 2 * math.pi / float(omega))
    m_a = complex(float(-Z.imag) / float(omega))
    c_d = complex(Z.real)
    vol = math.pi * a * a
    return {
        "Z": complex(complex(Z)),
        "C_a": float(Z.imag and m_a.real) / (rho * vol),          # added-mass coefficient
        "damping": float(c_d.real) / (rho * vol * omega),  # normalised
        "beta": beta,
        "Re_ref": beta,  # beta
        "U": U,
        "a": a,
    }


if __name__ == "__main__":
    print("beta     C_a      damping/(rho V w)")
    for beta in (5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 4000.0):
        a = 8.0
        T = 1.0
        nu = (2 * a) ** 2 / (beta * T)
        omega = 2 * math.pi / T
        r = oscillating_cylinder_force(a, omega, nu, U=1.0, rho=1.0)
        print(f"{beta:7.1f} {r['C_a']:8.5f} {r['damping']:10.5f}  (nu={nu:.3e})")