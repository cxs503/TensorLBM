"""Wave probe inversion methods.

Three inversion methods for estimating wave parameters from probe measurements:

1. TwoPointInversion: Classic two-probe method (requires known k).
2. ThreePointGodaInversion: Goda & Suzuki (1976) three-point method.
3. MultiProbeLeastSquares: Simultaneous estimation of R, k_r, α using
   nonlinear least squares over multiple probes.

All methods operate on harmonic fits (complex amplitudes) extracted from
probe time series.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import least_squares

from .synthetic_tank import SyntheticWaveConfig


@dataclass
class InversionResult:
    """Result of wave parameter inversion.

    Attributes
    ----------
    R_hat : float
        Estimated reflection coefficient |B|/|A|.
    k_r_hat : float
        Estimated real wavenumber.
    alpha_hat : float
        Estimated viscous attenuation coefficient.
    c_hat : float
        Estimated phase speed ω/k_r_hat.
    omega : float
        Angular frequency (input).
    a_incident : float
        Estimated incident wave amplitude.
    a_reflected : float
        Estimated reflected wave amplitude.
    residual_rms : float
        RMS residual of the fit (for multi-probe method).
    method : str
        Name of the inversion method used.
    """
    R_hat: float
    k_r_hat: float
    alpha_hat: float
    c_hat: float
    omega: float
    a_incident: float
    a_reflected: float
    residual_rms: float = 0.0
    method: str = ""

    def to_dict(self) -> dict:
        return {
            "R_hat": self.R_hat,
            "k_r_hat": self.k_r_hat,
            "alpha_hat": self.alpha_hat,
            "c_hat": self.c_hat,
            "omega": self.omega,
            "a_incident": self.a_incident,
            "a_reflected": self.a_reflected,
            "residual_rms": self.residual_rms,
            "method": self.method,
        }


def fit_harmonic(
    t: np.ndarray,
    eta: np.ndarray,
    omega: float,
) -> complex:
    """Extract complex amplitude at frequency ω from time series.

    Uses least-squares fit: η(t) ≈ mean + Re[c * e^{iωt}]
    where c = a * e^{-iφ} for η = a * cos(ωt - φ).

    Returns the complex amplitude c.
    """
    t = np.asarray(t, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)

    # Build design matrix: [1, cos(ωt), sin(ωt)]
    A = np.column_stack([
        np.ones_like(t),
        np.cos(omega * t),
        np.sin(omega * t),
    ])

    # Least squares fit
    coef, *_ = np.linalg.lstsq(A, eta, rcond=None)

    # Complex amplitude: c = coef_cos - i * coef_sin
    # (since η = a*cos(ωt - φ) = a*cos(φ)*cos(ωt) + a*sin(φ)*sin(ωt))
    c = coef[1] - 1j * coef[2]
    return c


class TwoPointInversion:
    """Two-probe inversion method.

    Given two probes at x1, x2 with complex amplitudes c1, c2,
    and assuming known wavenumber k and no reflection (R=0):
        c(x) = A * e^{-ikx} * e^{-αx}
        c2/c1 = e^{-ik(x2-x1)} * e^{-α(x2-x1)}

    This method requires known k and cannot separate R from α.
    It's included as a baseline/negative control.
    """

    def __init__(self, omega: float, k_known: float):
        self.omega = omega
        self.k_known = k_known

    def invert(
        self,
        x1: float,
        x2: float,
        c1: complex,
        c2: complex,
    ) -> InversionResult:
        """Invert from two probe measurements.

        Parameters
        ----------
        x1, x2 : float
            Probe positions.
        c1, c2 : complex
            Complex amplitudes at probes.

        Returns
        -------
        InversionResult
        """
        dx = x2 - x1
        if abs(dx) < 1e-12:
            raise ValueError("Probe separation must be non-zero")

        # Phase difference gives k
        ratio = c2 / c1
        dphi = np.angle(ratio)

        # Unwrap to nearest branch of -k*dx
        target = -self.k_known * dx
        dphi = dphi + 2.0 * math.pi * round((target - dphi) / (2.0 * math.pi))

        k_hat = -dphi / dx

        # Amplitude ratio gives α
        amp_ratio = abs(ratio)
        alpha_hat = -math.log(amp_ratio) / dx if amp_ratio > 0 else 0.0

        # Phase speed
        c_hat = self.omega / k_hat if abs(k_hat) > 1e-15 else float('inf')

        # Cannot estimate R with only 2 probes (assume R=0)
        a_inc = abs(c1) * math.exp(alpha_hat * x1)
        a_ref = 0.0

        return InversionResult(
            R_hat=0.0,
            k_r_hat=k_hat,
            alpha_hat=alpha_hat,
            c_hat=c_hat,
            omega=self.omega,
            a_incident=a_inc,
            a_reflected=a_ref,
            method="two_point",
        )


class ThreePointGodaInversion:
    """Three-point Goda & Suzuki (1976) inversion method.

    Given three probes at x1, x2, x3 with complex amplitudes c1, c2, c3,
    solve for incident amplitude A, reflected amplitude B, and wavenumber k.

    The system is:
        c_j = A * e^{-ikx_j} + B * e^{+ikx_j},  j = 1, 2, 3

    This is a nonlinear system in k. Goda's method uses the fact that
    for equally spaced probes (dx = x2-x1 = x3-x2), the system can be
    reduced to a single equation in k.

    For non-equally spaced probes, we use a direct nonlinear solve.
    """

    def __init__(self, omega: float, k_guess: Optional[float] = None):
        self.omega = omega
        self.k_guess = k_guess

    def invert(
        self,
        x1: float,
        x2: float,
        x3: float,
        c1: complex,
        c2: complex,
        c3: complex,
    ) -> InversionResult:
        """Invert from three probe measurements.

        Parameters
        ----------
        x1, x2, x3 : float
            Probe positions (must be distinct).
        c1, c2, c3 : complex
            Complex amplitudes at probes.

        Returns
        -------
        InversionResult
        """
        # Check for degenerate spacing
        dx12 = x2 - x1
        dx23 = x3 - x2
        if abs(dx12) < 1e-12 or abs(dx23) < 1e-12:
            raise ValueError("Probe separations must be non-zero")

        # Initial guess for k
        if self.k_guess is not None:
            k0 = self.k_guess
        else:
            # Estimate from phase difference
            dphi = np.angle(c2 / c1)
            k0 = -dphi / dx12
            if k0 < 0:
                k0 = -k0

        # Solve for k using the Goda method
        # For three probes, we have:
        #   c1 = A*e^{-ikx1} + B*e^{ikx1}
        #   c2 = A*e^{-ikx2} + B*e^{ikx2}
        #   c3 = A*e^{-ikx3} + B*e^{ikx3}
        #
        # Eliminating A and B:
        #   c1*sin(k(x3-x2)) - c2*sin(k(x3-x1)) + c3*sin(k(x2-x1)) = 0
        #
        # This is the Goda equation. We solve for k.

        def goda_residual(k):
            """Goda's equation residual (complex)."""
            term1 = c1 * np.sin(k * (x3 - x2))
            term2 = c2 * np.sin(k * (x3 - x1))
            term3 = c3 * np.sin(k * (x2 - x1))
            return term1 - term2 + term3

        # Solve using Newton's method
        k_hat = k0
        for _ in range(50):
            f = goda_residual(k_hat)
            # Numerical derivative
            dk = 1e-8 * abs(k_hat) + 1e-12
            df = (goda_residual(k_hat + dk) - f) / dk
            if abs(df) < 1e-20:
                break
            step = f / df
            k_hat -= step
            if abs(step) < 1e-12 * abs(k_hat) + 1e-14:
                break

        # Now solve for A and B given k_hat
        # c1 = A*e^{-ikx1} + B*e^{ikx1}
        # c2 = A*e^{-ikx2} + B*e^{ikx2}
        e1m = np.exp(-1j * k_hat * x1)
        e1p = np.exp(1j * k_hat * x1)
        e2m = np.exp(-1j * k_hat * x2)
        e2p = np.exp(1j * k_hat * x2)

        det = e1m * e2p - e1p * e2m
        if abs(det) < 1e-15:
            # Degenerate case
            A_hat = 0.5 * (abs(c1) + abs(c2))
            B_hat = 0.0
        else:
            A_hat = (c1 * e2p - e1p * c2) / det
            B_hat = (e1m * c2 - c1 * e2m) / det

        a_inc = abs(A_hat)
        a_ref = abs(B_hat)
        R_hat = a_ref / a_inc if a_inc > 1e-15 else 0.0

        # For Goda method, α is not directly estimated (assume α=0)
        alpha_hat = 0.0

        # Phase speed
        c_hat = self.omega / k_hat if abs(k_hat) > 1e-15 else float('inf')

        return InversionResult(
            R_hat=R_hat,
            k_r_hat=k_hat.real if isinstance(k_hat, complex) else k_hat,
            alpha_hat=alpha_hat,
            c_hat=c_hat,
            omega=self.omega,
            a_incident=a_inc,
            a_reflected=a_ref,
            method="three_point_goda",
        )


class MultiProbeLeastSquares:
    """Multi-probe least squares inversion.

    Simultaneously estimates R, k_r, α, and incident amplitude A
    from N ≥ 3 probe measurements using nonlinear least squares.

    The model is:
        c(x) = A * e^{-αx} * e^{-ik_r*x} + R*A * e^{-αx} * e^{+ik_r*x}
             = A * e^{-αx} * [e^{-ik_r*x} + R*e^{+ik_r*x}]

    Parameters to estimate: [A_real, A_imag, k_r, α, R_real, R_imag]
    where A and R are complex amplitudes.
    """

    def __init__(
        self,
        omega: float,
        k_guess: Optional[float] = None,
        alpha_guess: float = 0.0,
    ):
        self.omega = omega
        self.k_guess = k_guess
        self.alpha_guess = alpha_guess

    def invert(
        self,
        x_probes: np.ndarray,
        c_probes: np.ndarray,
    ) -> InversionResult:
        """Invert from multiple probe measurements.

        Parameters
        ----------
        x_probes : ndarray
            Probe positions, shape (N,).
        c_probes : ndarray
            Complex amplitudes at probes, shape (N,).

        Returns
        -------
        InversionResult
        """
        x = np.asarray(x_probes, dtype=np.float64)
        c = np.asarray(c_probes, dtype=np.complex128)
        N = len(x)
        if N < 3:
            raise ValueError("Need at least 3 probes for multi-probe inversion")

        # Initial guesses
        if self.k_guess is not None:
            k0 = self.k_guess
        else:
            # Estimate from first two probes
            dphi = np.angle(c[1] / c[0])
            dx = x[1] - x[0]
            k0 = abs(dphi / dx)

        alpha0 = self.alpha_guess

        # Estimate A from average amplitude (assuming small R and α)
        amp = np.abs(c)
        A0 = np.mean(amp)

        # Estimate R from amplitude variation
        # |c(x)|² ≈ |A|² * e^{-2αx} * (1 + R² + 2R*cos(2kx + φ))
        # For small R, the variation is ~2R|A|²
        amp_var = np.std(amp) / np.mean(amp) if np.mean(amp) > 0 else 0
        R0 = min(0.5, amp_var)  # initial guess for |R|

        # Parameters: [A_real, A_imag, k_r, alpha, R_real, R_imag]
        # where A = A_real + i*A_imag, R = R_real + i*R_imag
        p0 = np.array([
            A0,  # A_real
            0.0,  # A_imag
            k0,  # k_r
            alpha0,  # alpha
            R0,  # R_real
            0.0,  # R_imag
        ])

        def model(p):
            """Compute model complex amplitudes."""
            A_real, A_imag, k_r, alpha, R_real, R_imag = p
            A = A_real + 1j * A_imag
            R = R_real + 1j * R_imag
            decay = np.exp(-alpha * x)
            incident = A * decay * np.exp(-1j * k_r * x)
            reflected = R * A * decay * np.exp(1j * k_r * x)
            return incident + reflected

        def residuals(p):
            """Residual vector (real and imaginary parts)."""
            c_model = model(p)
            diff = c - c_model
            return np.concatenate([diff.real, diff.imag])

        # Solve with bounds to prevent overflow
        # Bounds: A_real, A_imag unbounded; k_r > 0; alpha >= 0; |R| < 1
        bounds = (
            [-np.inf, -np.inf, 0.5*k0, 0.0, -0.99, -0.99],  # lower
            [np.inf, np.inf, 2.0*k0, 0.1, 0.99, 0.99],  # upper
        )

        try:
            result = least_squares(
                residuals, p0, method='trf', bounds=bounds, max_nfev=2000
            )
        except Exception as e:
            # Fall back to unconstrained
            result = least_squares(residuals, p0, method='lm', max_nfev=1000)

        # Extract parameters
        A_real, A_imag, k_r_hat, alpha_hat, R_real, R_imag = result.x
        A_hat = A_real + 1j * A_imag
        R_hat_complex = R_real + 1j * R_imag

        R_mag = abs(R_hat_complex)
        a_inc = abs(A_hat)
        a_ref = R_mag * a_inc

        # Phase speed
        c_hat = self.omega / k_r_hat if abs(k_r_hat) > 1e-15 else float('inf')

        # Residual
        residual_rms = math.sqrt(np.mean(result.fun ** 2))

        return InversionResult(
            R_hat=R_mag,
            k_r_hat=k_r_hat,
            alpha_hat=alpha_hat,
            c_hat=c_hat,
            omega=self.omega,
            a_incident=a_inc,
            a_reflected=a_ref,
            residual_rms=residual_rms,
            method="multi_probe_ls",
        )
