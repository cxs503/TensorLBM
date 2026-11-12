"""Flapping-foil propulsion: closed-form analytic anchors (P162W1-N07).

Analytic half of the oscillating-wing (flapping-foil) verification face.  No
LBM dependency; pure numpy/scipy.

Contents
--------
* harmonic kinematics (heave+pitch), reduced frequency ``k`` and Strouhal ``St``;
* Theodorsen lift-deficiency function ``C(k) = F + iG`` in closed form
  (Hankel route + independent modified-Bessel route + exact inviscid limits);
* Theodorsen unsteady lift/moment in phasor form for arbitrary heave+pitch;
* Garrick (1936) thrust / input-power / efficiency closed forms for combined
  heave+pitch, including the leading-edge suction thrust **and** the lift-tilt
  term, with the cycle mean evaluated in the phasor domain (no time stepping)
  and a quadrature cross-check;
* thrust-reversal line ``C_T(St) = 0``;
* Roache grid-convergence index (GCI) helper;
* an energy-budget ledger for the LBM side.

Sign conventions (documented once, used everywhere)
--------------------------------------------------
* Foil chord ``c = 2b``, span 1, upstream flow at speed ``U`` in ``+x``.
* Heave ``h`` is positive **downward** (``+y`` is up), so ``hdot > 0`` means the
  foil descends and the effective angle of attack increases.
* Lift ``L`` is positive **upward** (Leishman/Fung form, consistent with the
  heave sign above): the quasi-steady limit is ``L = 2*pi*rho*U*b*(hdot+U*alpha)``.
* Pitch ``alpha`` positive nose-up, moment ``M`` positive nose-up, pitch axis at
  ``a*b`` from the mid-chord (``a=0`` mid-chord, ``a=-1`` leading edge).
* The generalized force conjugate to ``h`` is ``-L`` (since ``h`` points down),
  hence the input power ``P_in = mean(L*hdot) - mean(M*alphadot) > 0`` for a
  thrust-producing foil.

Thrust (Garrick 1936)
---------------------
Leading-edge suction plus the tilt of the normal force into the flight
direction, both at second order::

    S(t) = 2*pi*rho*b*|C(k)|^2 * w(t)^2,     w = hdot + U*alpha + b*(1/2-a)*alphadot
    T(t) = S(t) - L(t)*alpha(t)

The tilt term is what makes the quasi-steady limit exact: for slow pure
pitching ``w = U*alpha`` and ``L = 2*pi*rho*U*b*w``, so ``T -> 0`` (a steadily
pitched flat plate produces no net thrust).  Dropping it would violate energy
conservation (``eta > 1``).

References
----------
Theodorsen, T. (1935) NACA TR 496.  Garrick, I. E. (1936) NACA TR 567.
Fung, Y. C. (1969) *An Introduction to the Theory of Aeroelasticity*.
Leishman, J. G. (2006) *Principles of Helicopter Aerodynamics*, ch. 8.
Roache, P. J. (1994) ASME J. Fluids Eng. 116, 405-413 (GCI).
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "FlappingKinematics",
    "theodorsen_C",
    "theodorsen_C_bessel",
    "theodorsen_table",
    "TheodorsenResponse",
    "theodorsen_response",
    "GarrickResult",
    "garrick_coefficients",
    "kinematics_diagnostics",
    "reduced_frequency",
    "strouhal",
    "k_from_strouhal",
    "thrust_reversal_heave",
    "thrust_reversal_line",
    "GCIResult",
    "gci",
    "EnergyLedger",
]


# ---------------------------------------------------------------------------
# kinematics
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FlappingKinematics:
    """Harmonic heave + pitch kinematics of a 2D foil.

    ``h(t) = h0*sin(omega*t)`` (positive downward),
    ``alpha(t) = alpha0*sin(omega*t + phi)`` (positive nose-up).
    """

    U: float
    c: float
    h0: float = 0.0
    alpha0: float = 0.0      # [rad]
    phi: float = 0.0         # [rad] pitch-heave phase lead
    a: float = 0.0           # pitch axis, semi-chords from mid-chord

    @property
    def b(self) -> float:
        return 0.5 * self.c

    def omega(self, k: float) -> float:
        return 2.0 * k * self.U / self.c

    def period(self, k: float) -> float:
        return 2.0 * math.pi / self.omega(k)

    def strouhal(self, k: float) -> float:
        """``St = f*A/U`` with ``A = 2*h0`` (peak-to-peak heave excursion)."""
        return strouhal(f=1.0 / self.period(k), A=2.0 * self.h0, U=self.U)

    # explicit time-domain kinematics (used by the LBM driver)
    def heave(self, t, k: float):
        return self.h0 * np.sin(self.omega(k) * np.asarray(t, dtype=float))

    def heave_rate(self, t, k: float):
        w = self.omega(k)
        return self.h0 * w * np.cos(w * np.asarray(t, dtype=float))

    def heave_accel(self, t, k: float):
        w = self.omega(k)
        return -self.h0 * w * w * np.sin(w * np.asarray(t, dtype=float))

    def pitch(self, t, k: float):
        w = self.omega(k)
        return self.alpha0 * np.sin(w * np.asarray(t, dtype=float) + self.phi)

    def pitch_rate(self, t, k: float):
        w = self.omega(k)
        return self.alpha0 * w * np.cos(w * np.asarray(t, dtype=float) + self.phi)

    def pitch_accel(self, t, k: float):
        w = self.omega(k)
        return -self.alpha0 * w * w * np.sin(w * np.asarray(t, dtype=float) + self.phi)


def reduced_frequency(f: float, c: float, U: float) -> float:
    """``k = 2*pi*f*c/(2*U)``."""
    return math.pi * f * c / U


def strouhal(f: float, A: float, U: float) -> float:
    """``St = f*A/U``."""
    return f * A / U


def k_from_strouhal(St: float, h0_over_c: float) -> float:
    """``k`` of a pure-heave foil for a target Strouhal number.

    ``St = 2*k*(h0/c)/pi``  =>  ``k = pi*St/(2*h0/c)``.
    """
    if h0_over_c <= 0.0:
        raise ValueError("h0_over_c must be positive")
    return math.pi * St / (2.0 * h0_over_c)


# ---------------------------------------------------------------------------
# Theodorsen function
# ---------------------------------------------------------------------------
def theodorsen_C(k: float) -> complex:
    """``C(k) = F(k) + i*G(k)`` through Hankel functions (2nd kind).

    ``C(k) = H1^(2)(k) / [H1^(2)(k) + i*H0^(2)(k)]``.

    Exact inviscid limits used as self-checks: ``C(0) = 1`` (quasi-steady) and
    ``C(k->inf) = 1/2`` (the circulation cannot follow a fast oscillation).
    """
    if k < 0.0:
        raise ValueError("k must be non-negative")
    if k == 0.0:
        return complex(1.0, 0.0)
    from scipy.special import hankel2

    h1 = hankel2(1, k)
    h0 = hankel2(0, k)
    return complex(h1 / (h1 + 1j * h0))


def theodorsen_C_bessel(k: float) -> complex:
    """Independent code path: ``C(k) = K1(ik) / [K1(ik) + K0(ik)]``.

    ``K_nu`` = modified Bessel function of the second kind (AMOS/ZBK rational
    approximations).  Agreement with :func:`theodorsen_C` checks that the Hankel
    branch cut / sign convention is right.
    """
    if k < 0.0:
        raise ValueError("k must be non-negative")
    if k == 0.0:
        return complex(1.0, 0.0)
    from scipy.special import kv

    z = 1j * k
    k0 = kv(0, z)
    k1 = kv(1, z)
    return complex(k1 / (k1 + k0))


def theodorsen_table(ks: Iterable[float]) -> list[dict]:
    rows = []
    for k in ks:
        c_h = theodorsen_C(k)
        c_b = theodorsen_C_bessel(k)
        rows.append(
            {
                "k": float(k),
                "F": c_h.real,
                "G": c_h.imag,
                "absC": abs(c_h),
                "absC2": abs(c_h) ** 2,
                "bessel_re": c_b.real,
                "bessel_im": c_b.imag,
                "hankel_minus_bessel": abs(c_h - c_b),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Theodorsen airloads (phasor form)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TheodorsenResponse:
    """Phasor amplitudes of the unsteady airloads.

    ``L(t) = Re(L_hat*exp(i*omega*t))`` (lift, positive up), likewise ``M``.
    ``phi_h = -pi/2`` is used internally because ``h = h0*sin`` is the
    imaginary part of ``h0*exp(i*omega*t)``.
    """

    k: float
    C: complex
    w_hat: complex
    L_hat: complex
    M_hat: complex
    lift_amplitude: float
    lift_phase: float
    moment_amplitude: float
    moment_phase: float


def _phasors(kin: FlappingKinematics, k: float, omega: float) -> tuple[complex, complex]:
    """``H`` (heave, positive down) and ``A`` (pitch) phasors.

    ``h = h0*sin(omega t) = Re((-i*h0) e^{i omega t})``
    ``alpha = alpha0*sin(omega t+phi) = Re((-i*alpha0*e^{i phi}) e^{i omega t})``
    """
    H = complex(0.0, -kin.h0)
    A = -1j * kin.alpha0 * cmath.exp(1j * kin.phi)
    return H, A


def theodorsen_response(kin: FlappingKinematics, k: float) -> TheodorsenResponse:
    """Theodorsen lift/moment phasors for ``kin`` at reduced frequency ``k``.

    ``L    = pi*rho*b^2*(hddot + U*alphadot - b*a*alphaddot)
            + 2*pi*rho*U*b*C(k)*w``
    ``M    = pi*rho*b^2*(b*a*hddot - U*b*(1/2-a)*alphadot - b^2*(1/8+a^2)*alphaddot)
            + 2*pi*rho*U*b^2*(a+1/2)*C(k)*w``
    ``w    = hddot... -> w = hdot + U*alpha + b*(1/2-a)*alphadot``

    (``rho = 1``; multiply by ``rho`` for a general density.)
    """
    U, c, a = kin.U, kin.c, kin.a
    b = 0.5 * c
    w_om = 2.0 * k * U / c
    C = theodorsen_C(k)
    H, A = _phasors(kin, k, w_om)

    hdot = 1j * w_om * H
    hddot = -(w_om**2) * H
    adot = 1j * w_om * A
    addot = -(w_om**2) * A

    w = hdot + U * A + b * (0.5 - a) * adot
    L = math.pi * b**2 * (hddot + U * adot - b * a * addot) + 2.0 * math.pi * U * b * C * w
    M = math.pi * b**2 * (
        b * a * hddot - U * b * (0.5 - a) * adot - b**2 * (0.125 + a * a) * addot
    ) + 2.0 * math.pi * U * b**2 * (a + 0.5) * C * w

    return TheodorsenResponse(
        k=k,
        C=C,
        w_hat=w,
        L_hat=L,
        M_hat=M,
        lift_amplitude=abs(L),
        lift_phase=cmath.phase(L),
        moment_amplitude=abs(M),
        moment_phase=cmath.phase(M),
    )


# ---------------------------------------------------------------------------
# Garrick coefficients
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GarrickResult:
    """Garrick (1936) propulsion coefficients (per unit span, ``rho`` folded in).

    ``C_T``  net inviscid thrust coefficient, ``T/(0.5*rho*U^2*c)``
    ``C_S``  leading-edge suction part of ``C_T``
    ``C_T_tilt``  lift-tilt part of ``C_T`` (negative for nose-up bias)
    ``C_P``  input power coefficient, ``P_in/(0.5*rho*U^3*c)``
    ``eta``  ``C_T/C_P = C_T/(C_T + C_P_loss)`` (Garrick efficiency)
    """

    k: float
    C: complex
    C_T: float
    C_S: float
    C_T_tilt: float
    C_P: float
    C_P_loss: float
    eta: float
    strouhal: float
    w_rms_over_U: float
    terms: dict = field(default_factory=dict)


def garrick_coefficients(
    kin: FlappingKinematics,
    k: float,
    n_samples: int = 0,
    rho: float = 1.0,
) -> GarrickResult:
    """Garrick thrust / power / efficiency for harmonic heave+pitch.

    Phasor domain (default) or explicit time quadrature (``n_samples > 0``, an
    independent cross-check of the phasor algebra).

    ``C_T   = [2*pi*b*|C|^2*mean(w^2) - mean(L*alpha)] / (0.5*U^2*c)``
    ``C_P   = [mean(L*hdot) - mean(M*alphadot)] / (0.5*U^3*c)``
    """
    if k <= 0.0:
        raise ValueError("k must be positive")
    U, c, a = kin.U, kin.c, kin.a
    b = 0.5 * c
    w_om = 2.0 * k * U / c
    C = theodorsen_C(k)
    q = 0.5 * rho * U * U * c
    qp = 0.5 * rho * U**3 * c

    if n_samples > 0:
        t = np.linspace(0.0, 2.0 * math.pi / w_om, n_samples, endpoint=False)
        hdot = kin.heave_rate(t, k)
        alpha = kin.pitch(t, k)
        adot = kin.pitch_rate(t, k)
        H, A = _phasors(kin, k, w_om)
        # exact harmonic fields via phasors (no quadrature error in L, M)
        ph = np.exp(1j * w_om * t)
        L = np.real(
            (
                math.pi * b**2 * (-(w_om**2) * H + U * 1j * w_om * A - b * a * (-(w_om**2) * A))
                + 2.0 * math.pi * U * b * C * (1j * w_om * H + U * A + b * (0.5 - a) * 1j * w_om * A)
            )
            * ph
        )
        M = np.real(
            (
                math.pi
                * b**2
                * (
                    b * a * (-(w_om**2) * H)
                    - U * b * (0.5 - a) * 1j * w_om * A
                    - b**2 * (0.125 + a * a) * (-(w_om**2) * A)
                )
                + 2.0 * math.pi * U * b**2 * (a + 0.5) * C
                * (1j * w_om * H + U * A + b * (0.5 - a) * 1j * w_om * A)
            )
            * ph
        )
        w = hdot + U * alpha + b * (0.5 - a) * adot
        S = 2.0 * math.pi * rho * b * abs(C) ** 2 * w**2
        T = S - L * alpha
        C_S = float(np.mean(S)) / q
        C_T_tilt = -float(np.mean(L * alpha)) / q
        C_T = float(np.mean(T)) / q
        P_in = float(np.mean(L * hdot - M * adot))
        C_P = P_in / qp
        mean_w2 = float(np.mean(w**2))
    else:
        resp = theodorsen_response(kin, k)
        H, A = _phasors(kin, k, w_om)
        w_hat = resp.w_hat
        L_hat, M_hat = resp.L_hat, resp.M_hat
        hdot_hat = 1j * w_om * H
        adot_hat = 1j * w_om * A
        mean_w2 = 0.5 * abs(w_hat) ** 2
        S_mean = 2.0 * math.pi * rho * b * abs(C) ** 2 * mean_w2
        mean_Lalpha = 0.5 * (L_hat * np.conj(A)).real
        C_S = S_mean / q
        C_T_tilt = -mean_Lalpha / q
        C_T = (S_mean - mean_Lalpha) / q
        # P_in = mean(L*hdot) - mean(M*alphadot)
        P_in = 0.5 * (L_hat * np.conj(hdot_hat)).real - 0.5 * (M_hat * np.conj(adot_hat)).real
        C_P = P_in / qp

    C_P_loss = C_P - C_T
    eta = C_T / C_P if C_P != 0.0 else float("nan")

    return GarrickResult(
        k=k,
        C=C,
        C_T=C_T,
        C_S=C_S,
        C_T_tilt=C_T_tilt,
        C_P=C_P,
        C_P_loss=C_P_loss,
        eta=eta,
        strouhal=kin.strouhal(k),
        w_rms_over_U=math.sqrt(mean_w2) / U,
        terms={
            "C_S": C_S,
            "C_T_tilt": C_T_tilt,
            "C_T": C_T,
            "C_P": C_P,
            "C_P_loss": C_P_loss,
            "eta": eta,
        },
    )


def kinematics_diagnostics(kin: FlappingKinematics, k: float) -> dict:
    """Reduced frequency, period, Strouhal number, amplitude ratios."""
    T = kin.period(k)
    return {
        "k": k,
        "omega": kin.omega(k),
        "period": T,
        "frequency": 1.0 / T,
        "strouhal": kin.strouhal(k),
        "h0_over_c": kin.h0 / kin.c,
        "alpha0_deg": math.degrees(kin.alpha0),
        "phi_deg": math.degrees(kin.phi),
        "pitch_axis_a": kin.a,
        "amplitude_ratio_h0_over_b": kin.h0 / (0.5 * kin.c),
    }


# ---------------------------------------------------------------------------
# thrust reversal
# ---------------------------------------------------------------------------
def thrust_reversal_heave(
    k: float,
    alpha0: float,
    phi: float,
    a: float,
    cd0: float = 0.0,
    h0_over_c_hi: float = 2.0,
) -> dict:
    """Critical heave amplitude at which ``C_T(h0/c) = cd0`` (thrust reversal).

    ``C_T`` is monotone increasing in ``h0/c`` at fixed ``k`` (it is a positive
    quadratic form in the amplitudes), so bisection is safe.
    """
    U, c = 1.0, 1.0

    def net(h0c: float) -> float:
        kin = FlappingKinematics(U=U, c=c, h0=h0c * c, alpha0=alpha0, phi=phi, a=a)
        return garrick_coefficients(kin, k).C_T - cd0

    if net(h0_over_c_hi) <= 0.0:
        return {
            "k": k,
            "exists": False,
            "h0_over_c_crit": float("nan"),
            "St_crit": float("nan"),
            "C_T_at_crit": float("nan"),
        }
    lo, hi = 1e-8, h0_over_c_hi
    if net(lo) > 0.0:
        h0c = lo
    else:
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if net(mid) > 0.0:
                hi = mid
            else:
                lo = mid
            if hi - lo < 1e-15:
                break
        h0c = 0.5 * (lo + hi)
    kin = FlappingKinematics(U=U, c=c, h0=h0c * c, alpha0=alpha0, phi=phi, a=a)
    return {
        "k": k,
        "exists": True,
        "h0_over_c_crit": h0c,
        "St_crit": kin.strouhal(k),
        "C_T_at_crit": garrick_coefficients(kin, k).C_T,
    }


def thrust_reversal_line(
    ks: Sequence[float],
    alpha0: float,
    phi: float,
    a: float,
    cd0: float = 0.0,
) -> list[dict]:
    return [thrust_reversal_heave(k, alpha0, phi, a, cd0) for k in ks]


# ---------------------------------------------------------------------------
# GCI (Roache 1994)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GCIResult:
    f_fine: float
    f_medium: float
    f_coarse: float
    r: float
    p_observed: float
    f_extrapolated: float
    gci_fine: float
    error_ratio: float


def gci(f_fine: float, f_medium: float, f_coarse: float, r: float = 2.0) -> GCIResult:
    """Roache GCI on a triplet with refinement ratio ``r``.

    ``p = ln|e32/e21| / ln r`` (observed order), Richardson extrapolation,
    ``GCI_fine = 1.25*|e21|/|f_fine| / (r^p - 1)`` (Roache 1994).

    History (2026-11-12, ``docs/GCI_CONVENTION_AUDIT.md``): the first version
    of this function omitted the ``1/(r^p - 1)`` factor, so it reported the
    *relative fine-grid change* times the safety factor - a quantity that is
    ``r^p - 1`` (3x at ``p = 2, r = 2``) larger than Roache's index and
    independent of the observed order.  The change is conservative in
    direction (GCI was overstated) but made this site non-comparable with
    every other GCI in the repository.
    ``error_ratio = |e32|/|e21|`` -- this is the quantity compared against the
    theoretical ``r^p`` (``= 4`` for ``p = 2``, ``r = 2``).
    """
    e21 = f_medium - f_fine
    e32 = f_coarse - f_medium
    if abs(e21) <= 0.0 or abs(e32) <= 0.0:
        p = float("nan")
        f_ex = f_fine
    else:
        p = math.log(abs(e32) / abs(e21)) / math.log(r)
        if not math.isfinite(p) or abs(r**p - 1.0) < 1e-14:
            f_ex = f_fine
        else:
            f_ex = f_fine + (f_fine - f_medium) / (r**p - 1.0)
    return GCIResult(
        f_fine=f_fine,
        f_medium=f_medium,
        f_coarse=f_coarse,
        r=r,
        p_observed=p,
        f_extrapolated=f_ex,
        gci_fine=(
            1.25 * abs(e21) / max(abs(f_fine), 1e-300) / (r**p - 1.0)
            if (math.isfinite(p) and abs(r**p - 1.0) > 1e-14)
            else float("nan")
        ),
        error_ratio=float("nan") if abs(e21) == 0.0 else abs(e32) / abs(e21),
    )


# ---------------------------------------------------------------------------
# energy ledger
# ---------------------------------------------------------------------------
class EnergyLedger:
    """Cycle-resolved energy budget of a forced IBM/LBM run.

    Discrete budget (lattice units, ``dt = 1``)::

        W_body = dKE + Phi_visc + F_boundary

    ``W_body`` = work done by the body on the fluid ``sum(F_E . u_E)``,
    ``dKE``    = change of domain kinetic energy over the window,
    ``Phi_visc`` = measured viscous dissipation,
    ``F_boundary`` = energy flux through open boundaries.
    """

    def __init__(self) -> None:
        self.records: list[dict] = []

    def add(self, **kw) -> None:
        self.records.append(dict(kw))

    def residual(self, window: slice | None = None) -> dict:
        recs = self.records[window] if window is not None else self.records
        if not recs:
            return {"W": 0.0, "dKE": 0.0, "Phi": 0.0, "flux": 0.0, "residual": float("nan"),
                    "relative_residual": float("nan")}
        W = sum(r.get("W", 0.0) for r in recs)
        dKE = sum(r.get("dKE", 0.0) for r in recs)
        Phi = sum(r.get("Phi", 0.0) for r in recs)
        flux = sum(r.get("flux", 0.0) for r in recs)
        denom = abs(W) if abs(W) > 0 else 1.0
        return {
            "W": W,
            "dKE": dKE,
            "Phi": Phi,
            "flux": flux,
            "residual": W - dKE - Phi - flux,
            "relative_residual": abs(W - dKE - Phi - flux) / denom,
        }

    def to_dict(self) -> dict:
        return {"records": self.records, "summary": self.residual()}