"""Exact time-Taylor-series reference for the genuine (z-dependent) 3-D Taylor–Green vortex.

Motivation (the honest gap this module closes)
----------------------------------------------
The Taylor–Green case in :mod:`tensorlbm.d3q27_validation` is the *z-invariant*
mode ``u = A(sin kx cos ky, −cos kx sin ky, 0)``.  Its convective term
``(u·∇)u`` is a pure gradient, so it drops out of the *projected*
Navier–Stokes equations: the flow decays as one Fourier mode,
``exp(−ν|k|²t)``.  That is an exact reference, but it never exercises the 3-D
nonlinearity — no vortex stretching, no enstrophy production, no energy
transfer across scales.  Every earlier round of ``CASE-TAYLOR-GREEN-3D`` hence
had to declare ``exact_solution: null`` for the classic z-dependent field.

The classic Taylor–Green initial condition (Taylor & Green 1937,
*Proc. R. Soc. A* 158:499)

    u = A sin x cos y cos z,   v = −A cos x sin y cos z,   w = 0            (1)

has no closed-form Navier–Stokes solution, but it does admit an **exact
time-Taylor series**.  Write ``u(x,t) = Σ_{n≥0} tⁿ u⁽ⁿ⁾(x)``, insert into the
Helmholtz-projected Navier–Stokes equations
``∂_t u = ν Δu − P[(u·∇)u]`` and equate powers of ``t``:

    u⁽ⁿ⁺¹⁾ = ( ν Δu⁽ⁿ⁾ − P Σ_{j=0}^{n} (u⁽ʲ⁾·∇)u⁽ⁿ⁻ʲ⁾ ) / (n+1).           (2)

Every ``u⁽ⁿ⁾`` is a finite Fourier series on **integer** wavevectors (the data
(1) have ``|k|_∞ = 1``; a convolution of modes with ``|k|_∞ ≤ a`` and ``≤ b``
has ``|k|_∞ ≤ a+b``, so at order ``n`` the support is ``|k|_∞ ≤ n+1``).  The
recursion is therefore carried out in floating point with no time
discretisation and no spatial truncation other than a dealiasing band that is
verified to be inactive (:meth:`TG3DSeries.dealias_margin`).  For any ``t``
inside the radius of convergence the velocity field, the kinetic energy and the
enstrophy are polynomials in ``t`` whose coefficients are machine-precision
accurate.  This is the classical early-time construction used for the same flow
by Morf, Orszag & Frisch (1980, *Phys. Rev. A* 23:267) and Brachet et al.
(1983, *J. Fluid Mech.* 130:411); here it serves as a *verification reference*
inside the convergence radius, not for singularity hunting.

What the series can and cannot do
---------------------------------
* It is exact for the *continuum* incompressible Navier–Stokes equations with
  the data (1) — a different approximation from LBM's.
* It converges only for ``t < t*`` (a few eddy turnovers ``T = A k t``).  Use
  it inside the radius only, and report the truncation level via
  :meth:`TG3DSeries.relative_tail` — the error is a reported quantity, not a
  hope.
* It says nothing about walls, forcing or compressibility.

Independent cross-check
-----------------------
:func:`spectral_dns_reference` integrates the *same* data with a dealiased
Fourier pseudo-spectral method (spectral derivatives, integrating-factor RK4).
Series vs DNS checks the series algebra; LBM vs series is the physics claim.
The two references share no code path.

Usage
-----
    from tensorlbm.tg3d_series import TG3DSeries, taylor_green_classic

    ser = TG3DSeries(taylor_green_classic(0.02), box=32.0, nu=0.01, order=12)
    e0, e1 = ser.energy(0.0), ser.energy(600.0)
    u, v, w = ser.velocity_field(600.0, n=32)      # onto the 32³ LBM lattice
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

__all__ = [
    "abc_flow",
    "convective_acceleration",
    "default_nfft",
    "spectral_dns_reference",
    "taylor_green_classic",
    "taylor_green_extruded",
    "TG3DSeries",
    "vorticity_spectral",
]

Field3D = tuple[np.ndarray, np.ndarray, np.ndarray]
ICField = Callable[[np.ndarray, np.ndarray, np.ndarray], Field3D]


# ---------------------------------------------------------------------------
# Initial conditions.  Arguments are *radian-like* coordinates
# ξ = 2π x / box, so one period of each trigonometric factor spans the periodic
# box exactly regardless of the box length (lattice units, box = N).
# ---------------------------------------------------------------------------


def taylor_green_classic(amplitude: float = 1.0) -> ICField:
    """Classic z-dependent 3-D Taylor–Green vortex, eq. (1)."""

    def ic(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Field3D:
        u = amplitude * np.sin(x) * np.cos(y) * np.cos(z)
        v = -amplitude * np.cos(x) * np.sin(y) * np.cos(z)
        w = np.zeros_like(u)
        return u, v, w

    return ic


def taylor_green_extruded(amplitude: float = 1.0) -> ICField:
    """z-invariant (extruded 2-D) Taylor–Green mode — the round-1/2 reference."""

    def ic(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Field3D:
        u = amplitude * np.sin(x) * np.cos(y)
        v = -amplitude * np.cos(x) * np.sin(y)
        w = np.zeros_like(u + y + z)
        return u, v, w

    return ic


def abc_flow(amplitude: float = 1.0) -> ICField:
    """ABC (Beltrami) flow — exact, z-dependent, but *linear* (∇×u = k u)."""

    def ic(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Field3D:
        u = amplitude * (np.sin(z) + np.cos(y))
        v = amplitude * (np.sin(x) + np.cos(z))
        w = amplitude * (np.sin(y) + np.cos(x))
        return u, v, w

    return ic


# ---------------------------------------------------------------------------
# Spectral helpers
# ---------------------------------------------------------------------------


def _next_fast_len(n: int) -> int:
    """Smallest 2/3/5-smooth even integer ≥ n (cheap FFT length)."""
    n = max(n, 8)
    if n % 2:
        n += 1
    best = None
    for p2 in range(0, 24):
        for p3 in range(0, 16):
            for p5 in range(0, 12):
                m = (2**p2) * (3**p3) * (5**p5)
                if m >= n and (best is None or m < best):
                    best = m
    return int(best if best is not None else n)


def default_nfft(order: int) -> int:
    """Dealiasing-safe grid for a series truncated at ``order``.

    Support at order ``n`` is ``|k|_∞ ≤ n+1``; the 2/3 rule on an ``nfft``
    grid keeps ``|k|_∞ ≤ nfft/3``, so ``nfft ≥ 3(order+2)`` is required.  A 4×
    band is used so that the reported dealiasing margin stays strictly
    positive (``TG3DSeries.dealias_margin``) instead of landing exactly on the
    cut-off.
    """
    return _next_fast_len(4 * (int(order) + 2))


def _freqs(n: int) -> np.ndarray:
    """Integer FFT frequencies in numpy's unshifted ordering."""
    return np.fft.fftfreq(n, d=1.0 / n)


def _axes(nfft: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Radian-like coordinates ξ for a uniform grid of ``nfft`` points on 2π."""
    xi = 2.0 * math.pi * np.arange(nfft) / nfft
    zz, yy, xx = np.meshgrid(xi, xi, xi, indexing="ij")
    return zz, yy, xx


def _spec(real: np.ndarray) -> np.ndarray:
    """Real field (3, n, n, n) → normalised Fourier coefficients."""
    n = real.shape[-1]
    return np.fft.fftn(real, axes=(1, 2, 3)) / (n**3)


def _real(spec: np.ndarray) -> np.ndarray:
    """Normalised Fourier coefficients → real field (3, n, n, n)."""
    n = spec.shape[-1]
    return np.fft.ifftn(spec, axes=(1, 2, 3)).real * (n**3)


def _dealias_mask(nfft: int) -> np.ndarray:
    """2/3-rule mask, shape (1, n, n, n), broadcasting over components."""
    m = _freqs(nfft)
    keep = np.abs(m) <= nfft / 3.0 + 1e-12
    return keep[:, None, None] & keep[None, :, None] & keep[None, None, :]


def vorticity_spectral(u: np.ndarray, v: np.ndarray, w: np.ndarray, box: float) -> Field3D:
    """Curl of a periodic velocity field by spectral differentiation.

    ``u, v, w`` are arrays of shape ``(n, n, n)`` on a uniform periodic grid of
    side ``box``; index order is ``(z, y, x)``.
    """
    n = u.shape[0]
    k = 2.0 * math.pi / box * _freqs(n)
    kz = k[:, None, None]
    ky = k[None, :, None]
    kx = k[None, None, :]
    U = np.fft.fftn(u)
    V = np.fft.fftn(v)
    W = np.fft.fftn(w)
    wx = np.fft.ifftn(1j * ky * W - 1j * kz * V).real
    wy = np.fft.ifftn(1j * kz * U - 1j * kx * W).real
    wz = np.fft.ifftn(1j * kx * V - 1j * ky * U).real
    return wx, wy, wz


def convective_acceleration(ic: ICField, *, box: float, n: int = 32) -> dict:
    """Witness of genuine nonlinearity: ``∇×[(u·∇)u]``.

    The z-invariant Taylor–Green mode has ``(u·∇)u = ∇(A²/2)(sin²x cos²y + …)``,
    i.e. its convective acceleration is a pure gradient, which the Helmholtz
    projector annihilates: such a flow is *linear* despite having vorticity.
    The classic z-dependent mode has a rotational convective acceleration — the
    vortex-stretching source.  This function measures the ratio
    ``‖∇×[(u·∇)u]‖ / ‖(u·∇)u)‖`` for an initial-condition callable.
    """
    zz, yy, xx = _axes(n)
    u, v, w = ic(xx, yy, zz)
    field = np.stack([u, v, w]).astype(np.float64)
    spec = _spec(field)
    k = 2.0 * math.pi / box * _freqs(n)
    kz, ky, kx = k[:, None, None], k[None, :, None], k[None, None, :]
    grad = np.empty((3, 3, n, n, n))
    for l, kv in enumerate((kx, ky, kz)):
        grad[l] = _real(1j * kv * spec)
    adv = np.zeros_like(field)
    for l in range(3):
        adv += field[l][None, :, :, :] * grad[l]
    a_spec = _spec(adv)
    curl = np.stack(
        [
            np.fft.ifftn(1j * ky * a_spec[2] - 1j * kz * a_spec[1]).real * n**3,
            np.fft.ifftn(1j * kz * a_spec[0] - 1j * kx * a_spec[2]).real * n**3,
            np.fft.ifftn(1j * kx * a_spec[1] - 1j * ky * a_spec[0]).real * n**3,
        ]
    )
    adv_rms = float(np.sqrt(np.mean(adv**2)))
    curl_rms = float(np.sqrt(np.mean(curl**2)))
    return {
        "adv_rms": adv_rms,
        "curl_adv_rms": curl_rms,
        "curl_over_adv": curl_rms / adv_rms if adv_rms else float("nan"),
        "u_rms": float(np.sqrt(np.mean(field**2))),
    }


# ---------------------------------------------------------------------------
# The series
# ---------------------------------------------------------------------------


class TG3DSeries:
    """Exact time-Taylor series of the (projected) Navier–Stokes equations.

    Parameters
    ----------
    ic:
        Initial-condition callable ``ic(x, y, z) -> (u, v, w)`` in radian-like
        coordinates (see :func:`taylor_green_classic`).
    box:
        Period of the box in the working units (lattice units ⇒ ``box = N``).
    nu:
        Kinematic viscosity in the same units (LBM: ``ν = (τ−½)/3``).
    order:
        Highest retained power of ``t``.
    nfft:
        Working grid.  Defaults to :func:`default_nfft` (dealiasing-safe).
    """

    def __init__(
        self,
        ic: ICField,
        *,
        box: float,
        nu: float,
        order: int = 12,
        nfft: int | None = None,
        dtype: type = np.float64,
    ) -> None:
        self.ic = ic
        self.box = float(box)
        self.nu = float(nu)
        self.order = int(order)
        self.nfft = int(nfft) if nfft is not None else default_nfft(order)
        self.dtype = dtype
        k = 2.0 * math.pi / self.box * _freqs(self.nfft)
        self.kz = k[:, None, None]
        self.ky = k[None, :, None]
        self.kx = k[None, None, :]
        self.k2 = self.kx**2 + self.ky**2 + self.kz**2
        self._mask = _dealias_mask(self.nfft)
        self.coef: list[np.ndarray] = []
        self._build()

    # -- construction ----------------------------------------------------

    def _initial_coef(self) -> np.ndarray:
        zz, yy, xx = _axes(self.nfft)
        u, v, w = self.ic(xx, yy, zz)
        field = np.stack([u, v, w]).astype(self.dtype)
        return _spec(field) * self._mask

    def _project(self, spec: np.ndarray) -> np.ndarray:
        """Helmholtz projection P = I − k k^T/|k|² (k = 0 row zeroed)."""
        kx, ky, kz = self.kx, self.ky, self.kz
        k2 = self.k2.copy()
        inv = np.where(k2 > 0, 1.0 / np.where(k2 > 0, k2, 1.0), 0.0)
        kdot = kx * spec[0] + ky * spec[1] + kz * spec[2]
        out = spec.copy()
        out[0] -= kx * kdot * inv
        out[1] -= ky * kdot * inv
        out[2] -= kz * kdot * inv
        return out * self._mask

    def _build(self) -> None:
        self.coef = [self._initial_coef()]
        # real-space fields and their gradients, per retained order
        real = [_real(self.coef[0])]
        grads = [self._grad(real[0])]
        for n in range(self.order):
            adv = np.zeros_like(real[0])
            for j in range(n + 1):
                uj = real[j]          # (3, n, n, n)
                gj = grads[n - j]     # (3, 3, n, n, n): gj[l, i] = ∂_l u_i
                for l in range(3):
                    adv += uj[l][None, :, :, :] * gj[l]
            rhs = self._project(_spec(adv))
            nxt = (-self.nu * self.k2[None, :, :, :] * self.coef[n] - rhs) / (n + 1)
            nxt = nxt * self._mask
            self.coef.append(nxt)
            real.append(_real(nxt))
            grads.append(self._grad(real[-1]))
        self._real = real
        self._grads = grads

    def _grad(self, field: np.ndarray) -> np.ndarray:
        """Gradient of a real field: returns (3, 3, n, n, n), ``[l, i] = ∂_l u_i``."""
        spec = _spec(field)
        kvec = (self.kx, self.ky, self.kz)
        out = np.empty((3, 3) + field.shape[1:], dtype=field.dtype)
        for l in range(3):
            out[l] = _real(1j * kvec[l] * spec)
        return out

    # -- diagnostics -----------------------------------------------------

    def support_peak(self, tol: float = 1e-13) -> float:
        """Largest |k|_∞ (in integer mode units) with a non-negligible amplitude."""
        peak = 0.0
        for c in self.coef:
            mag = np.abs(c).max(axis=0)
            if mag.max() <= 0:
                continue
            live = mag > tol * mag.max()
            idx = np.argwhere(live)
            if idx.size:
                m = _freqs(self.nfft)
                peak = max(peak, float(np.max(np.abs(m[idx]))))
        return peak

    def dealias_margin(self) -> dict[str, float]:
        """How far the true spectral support stays below the 2/3 cut-off."""
        peak = self.support_peak()
        return {
            "nfft": self.nfft,
            "support_peak_index": peak,
            "cutoff_index": self.nfft / 3.0,
            "margin": self.nfft / 3.0 - peak,
        }

    def _inner_product_series(self, weight2: np.ndarray | None = None) -> np.ndarray:
        """Σ_{a+b=n} ⟨u⁽ᵃ⁾, u⁽ᵇ⁾⟩ coefficients, optionally weighted by |k|²."""
        n_ord = self.order
        out = np.zeros(n_ord + 1, dtype=np.float64)
        for n in range(n_ord + 1):
            acc = 0.0 + 0.0j
            for a in range(n + 1):
                ca, cb = self.coef[a], self.coef[n - a]
                prod = np.conj(ca) * cb
                if weight2 is not None:
                    prod = prod * weight2[None, :, :, :]
                acc += prod.sum()
            out[n] = float(np.real(acc))
        return out

    def energy_series(self) -> np.ndarray:
        """Coefficients ``E_n`` with ``E(t) = ½ Σ_n E_n tⁿ`` (kinetic energy)."""
        return 0.5 * self._inner_product_series()

    def enstrophy_series(self) -> np.ndarray:
        """Coefficients ``Ω_n`` with ``Ω(t) = ½ Σ_n Ω_n tⁿ`` (enstrophy)."""
        return 0.5 * self._inner_product_series(weight2=self.k2)

    def energy(self, t: float, *, n: int | None = None) -> float:
        c = self.energy_series()
        n = self.order if n is None else int(n)
        return float(np.polyval(np.asarray(c[: n + 1])[::-1], t))

    def enstrophy(self, t: float, *, n: int | None = None) -> float:
        c = self.enstrophy_series()
        n = self.order if n is None else int(n)
        return float(np.polyval(np.asarray(c[: n + 1])[::-1], t))

    def relative_tail(self, t: float) -> dict[str, float]:
        """Relative contribution of the last retained order (truncation proxy)."""
        ce, co = self.energy_series(), self.enstrophy_series()
        e_full, o_full = self.energy(t), self.enstrophy(t)
        e_prev = self.energy(t, n=self.order - 1)
        o_prev = self.enstrophy(t, n=self.order - 1)
        a = abs(t**self.order * ce[self.order])
        b = abs(t**self.order * co[self.order])
        return {
            "energy_last_order_rel": float(a / abs(e_full)) if e_full else float("nan"),
            "enstrophy_last_order_rel": float(b / abs(o_full)) if o_full else float("nan"),
            "energy_step_rel": float(abs(e_full - e_prev) / abs(e_full)) if e_full else float("nan"),
            "enstrophy_step_rel": float(abs(o_full - o_prev) / abs(o_full)) if o_full else float("nan"),
            "energy": e_full,
            "enstrophy": o_full,
        }

    # -- field evaluation ------------------------------------------------

    def _to_grid(self, coef: np.ndarray, n: int) -> np.ndarray:
        if n == self.nfft:
            return np.fft.ifftn(coef, axes=(1, 2, 3)).real * (n**3)
        centered = np.fft.fftshift(coef, axes=(1, 2, 3))
        cs, ct = self.nfft // 2, n // 2
        span = min(cs, ct)
        out = np.zeros((3, n, n, n), dtype=np.complex128)
        out[:, ct - span : ct + span, ct - span : ct + span, ct - span : ct + span] = centered[
            :, cs - span : cs + span, cs - span : cs + span, cs - span : cs + span
        ]
        out = np.fft.ifftshift(out, axes=(1, 2, 3))
        return np.fft.ifftn(out, axes=(1, 2, 3)).real * (n**3)

    def velocity_field(self, t: float, *, n: int | None = None, order: int | None = None,
                       band_limit: bool = False) -> Field3D:
        """Velocity field at time ``t`` on an ``n³`` grid over the box.

        The truncation order is automatically lowered (and reported via
        :meth:`velocity_field_order`) until the retained spectrum fits below
        Nyquist on the requested grid — the spectral support of the classic
        Taylor–Green series grows by ≈2 per order (vortex stretching doubles
        the wavenumber), much faster than the ``n+1`` bound.
        """
        n = self.nfft if n is None else int(n)
        top = self.order if order is None else min(int(order), self.order)
        if band_limit:
            # Keep the full retained order and let ``_to_grid`` restriction-map
            # the spectrum onto the representable band |k_i| < n/2.  This is the
            # exact band-limited projection of the series field; the default
            # path instead *lowers the truncation order* until the support fits,
            # which discards in-band content of the high orders and makes the
            # returned field depend on the working grid (see the case-pool case
            # CASE-TAYLOR-GREEN-3D, quantity Q11).
            spec = self._series_spectrum(t, top)
            self.last_field_order = top
            self.last_field_support = self._spectrum_peak(spec)
            field = self._to_grid(spec, n)
            return field[0], field[1], field[2]
        while top >= 0:
            spec = self._series_spectrum(t, top)
            peak = self._spectrum_peak(spec)
            if peak < n / 2.0:
                break
            top -= 1
        if top < 0:
            raise ValueError(f"no truncation order of this series fits an {n}³ evaluation grid")
        self.last_field_order = top
        self.last_field_support = peak
        field = self._to_grid(spec, n)
        return field[0], field[1], field[2]

    def _series_spectrum(self, t: float, top: int) -> np.ndarray:
        spec = np.zeros_like(self.coef[0])
        tp = 1.0
        for a in range(top + 1):
            if a:
                tp *= t
            spec += tp * self.coef[a]
        return spec

    @staticmethod
    def _spectrum_peak(spec: np.ndarray, tol: float = 1e-13) -> float:
        mag = np.abs(spec).max(axis=0)
        if mag.max() <= 0:
            return 0.0
        idx = np.argwhere(mag > tol * mag.max())
        n = spec.shape[-1]
        m = _freqs(n)
        return float(np.max(np.abs(m[idx]))) if idx.size else 0.0

    def vorticity_field(self, t: float, *, n: int | None = None) -> Field3D:
        box = self.box
        u, v, w = self.velocity_field(t, n=n)
        return vorticity_spectral(u, v, w, box)

    def enstrophy_field(self, t: float, *, n: int | None = None) -> np.ndarray:
        wx, wy, wz = self.vorticity_field(t, n=n)
        return wx * wx + wy * wy + wz * wz


# ---------------------------------------------------------------------------
# Independent reference: Fourier pseudo-spectral DNS (integrating-factor RK4)
# ---------------------------------------------------------------------------


def spectral_dns_reference(
    ic: ICField,
    *,
    box: float,
    nu: float,
    steps: int,
    dt: float = 1.0,
    n: int | None = None,
    sample_every: int = 1,
) -> dict:
    """Dealiased pseudo-spectral integration of the same problem.

    Uses spectral (exact) derivatives, a 2/3 dealiasing rule and
    integrating-factor RK4 for the viscous term.  Returns a history with
    ``energy``, ``enstrophy`` and the final velocity field — the same
    quantities the series provides, computed by an independent code path.
    """
    n = int(n) if n is not None else 32
    zz, yy, xx = _axes(n)
    u, v, w = ic(xx, yy, zz)
    U = _spec(np.stack([u, v, w]).astype(np.float64))
    k = 2.0 * math.pi / box * _freqs(n)
    kz, ky, kx = k[:, None, None], k[None, :, None], k[None, None, :]
    k2 = kx**2 + ky**2 + kz**2
    mask = _dealias_mask(n)
    k2m = k2 * mask
    inv = np.where(k2m > 0, 1.0 / np.where(k2m > 0, k2m, 1.0), 0.0)

    def project(spec: np.ndarray) -> np.ndarray:
        kdot = kx * spec[0] + ky * spec[1] + kz * spec[2]
        out = spec.copy()
        out[0] -= kx * kdot * inv
        out[1] -= ky * kdot * inv
        out[2] -= kz * kdot * inv
        return out * mask

    def nonlinear(spec: np.ndarray) -> np.ndarray:
        uu = _real(spec)
        grad = np.empty((3, 3, n, n, n))
        for l, kv in enumerate((kx, ky, kz)):
            grad[l] = _real(1j * kv * spec)
        adv = np.zeros_like(uu)
        for l in range(3):
            adv += uu[l][None, :, :, :] * grad[l]
        return -project(_spec(adv))

    E = np.exp(-nu * k2m * dt)
    Ehalf = np.exp(-nu * k2m * dt / 2.0)

    def energy_of(spec: np.ndarray) -> float:
        return 0.5 * float(np.real(np.sum(np.abs(spec) ** 2)))

    def enstrophy_of(spec: np.ndarray) -> float:
        return 0.5 * float(np.real(np.sum(k2m * np.abs(spec) ** 2)))

    history = [
        {"step": 0, "time": 0.0, "energy": energy_of(U), "enstrophy": enstrophy_of(U)}
    ]
    for s in range(1, steps + 1):
        k1 = nonlinear(U)
        a = Ehalf * (U + 0.5 * dt * k1)
        k2v = nonlinear(a)
        b = Ehalf * U + 0.5 * dt * k2v
        k3 = nonlinear(b)
        c = E * U + dt * Ehalf * k3
        k4 = nonlinear(c)
        U = E * U + dt / 6.0 * (E * k1 + 2.0 * Ehalf * k2v + 2.0 * Ehalf * k3 + k4)
        U = U * mask
        if s % sample_every == 0 or s == steps:
            history.append(
                {"step": s, "time": s * dt, "energy": energy_of(U), "enstrophy": enstrophy_of(U)}
            )
    field = _real(U)
    return {
        "n": n,
        "box": box,
        "nu": nu,
        "steps": steps,
        "dt": dt,
        "history": history,
        "u": field[0],
        "v": field[1],
        "w": field[2],
        "energy": history[-1]["energy"],
        "enstrophy": history[-1]["enstrophy"],
    }