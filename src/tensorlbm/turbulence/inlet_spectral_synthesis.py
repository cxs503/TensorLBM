"""Spectrum- and moment-calibrated inlet turbulence synthesis (DFM primary, DSRFG auxiliary).

Campaign node ``p234w2-n05-inlet-turbulence-synthesis``.

What this module adds on top of :mod:`tensorlbm.turbulence.dfsem_inflow`
-----------------------------------------------------------------------
``dfsem_inflow`` answers *"is the inlet divergence-free and does it carry the
prescribed Reynolds stresses?"*.  It has no notion of a **target spectrum**: its
digital filter is a separable Gaussian of prescribed width (Klein 2003) and its
temporal content is whatever the eddy recycling produces.  This module adds the
missing leg -- a *spectral* inlet layer:

1. **Target spectra** (analytic, normalized in code, no hand-copied constants):

   * :func:`von_karman_e3d` / :func:`von_karman_tensor_spectrum` -- 3-D
     isotropic von Kármán energy spectrum and its Reynolds-stress spectral
     tensor ``Φ_ij(k) = E(k)/(4πk²) (δ_ij − k_i k_j/k²)``;
   * :func:`von_karman_e11` -- 1-D longitudinal spectrum ``E11(k1)`` (the
     quantity a single time series can actually be compared against);
   * :func:`kaimal_su` -- Kaimal 1-D spectrum (second target family);
   * :func:`von_karman_plane_spectrum` -- the *inlet-plane* spectral density
     ``Ψ_ii(k_y,k_z) = ∫ Φ_ii dk_x``, i.e. the correct 2-D target for a plane
     inlet (integrating out the normal direction is what makes the plane
     statistics consistent with the 3-D tensor).

2. **Generation by digital filtering with an explicit Turin decomposition**:
   a stationary process with target correlation ``R(τ)`` is realized as white
   noise convolved with ``h = IFFT(√S)``, where ``S = FFT(R)`` is the target
   spectral density (Turin 1960 spectral factorization ``S(ω) = |H(ω)|²``).
   The kernel is obtained analytically from the target spectrum and applied
   along time (temporal correlation, exponential/OU limit) and on the inlet
   plane (spatial correlation).

3. **DSRFG** (Smirnov 2001; Huang *et al.* 2010) as an independent second
   generator, used to cross-check the digital-filter statistics and the moment
   calibration.

4. **In-plane divergence control that matches the measured operator**: the
   transverse pair is projected with the *central-difference symbol*
   ``sin(k)`` rather than the continuous symbol ``k``, so the divergence the
   sampler measures (and an LBM stencil would feel) vanishes to round-off, not
   just to ``O(k²Δx²)``.

5. **Statistical sampler + machine gates**: one-sided spectrum ``E(f)``,
   1/3-octave band error against the target over ``[0.2, 5] f_peak``, Reynolds
   stresses ``R_ij``, integral scale ``L_ux = U∫R_uu dτ``, exponential-fit
   ``R²`` of ``R_uu(τ)``, divergence metric, ``sha256`` reproducibility and the
   ``L_ux = 0`` negative control -- see :func:`evaluate_gates`.

Units: lattice units (length = cells, time = steps, velocity = c_s·Mach).
``U`` is the mean convection velocity used both for Taylor's hypothesis
``f = U k1 / 2π`` and for the integral-scale conversion ``L_ux = U T_u``.

References
----------
* von Kármán T. (1948) *Proc. Natl. Acad. Sci.* 34 530.
* Kaimal J.C. *et al.* (1972) *J. Fluid Mech.* 53 647.
* Turin G.L. (1960) *IRE Trans. Inf. Theory* 6 311 (spectral factorization).
* Klein M. *et al.* (2003) *J. Comput. Phys.* 186 652 (digital filter method).
* Smirnov A. *et al.* (2001) *J. Turbul.* 2 1; Huang S.H. *et al.* (2010)
  *J. Wind Eng. Ind. Aerodyn.* 98 334 (DSRFG).
* Lund T.S. *et al.* (1998) *J. Comput. Phys.* 140 233 (moment rescaling).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field, replace

import numpy as np

__all__ = [
    "DSRFGInlet",
    "InletStatistics",
    "SpectralInletSynthesis",
    "SynthesisConfig",
    "autocorrelation",
    "evaluate_gates",
    "exponential_fit",
    "inlet_statistics",
    "inplane_divergence",
    "kaimal_su",
    "periodogram",
    "project_inplane_divergence_free",
    "third_octave_bands",
    "turin_filter_amplitudes",
    "von_karman_acf_time",
    "von_karman_e11",
    "von_karman_e3d",
    "von_karman_f_peak",
    "von_karman_plane_spectrum",
    "von_karman_tensor_spectrum",
]


# ---------------------------------------------------------------------------
# 1. Target spectra
# ---------------------------------------------------------------------------

#: ``1.339`` is not a free constant: it is fixed by ``∫ E11(k1) dk1 = u_rms²``
#: together with the von Kármán 5/6 exponent.  The test-suite re-derives the
#: normalization numerically (``test_von_karman_e11_normalisation``), so a typo
#: here cannot pass silently.
_VK_1D_SHAPE = 1.339

#: Constant of the 3-D von Kármán spectrum, derived from the Beta function so
#: that ``∫₀^∞ E(k) dk = 3/2 u_rms²`` (isotropic trace of ``R_ij``).
_VK_3D_CONST = 1.5 / (
    0.5 * math.exp(math.lgamma(2.5) + math.lgamma(1.0 / 3.0) - math.lgamma(17.0 / 6.0))
)


def von_karman_e11(k1, u_rms: float = 1.0, L: float = 1.0) -> np.ndarray:
    """1-D longitudinal von Kármán spectrum ``E11(k1)`` (defined for ``k1 >= 0``).

    ``E11(k1) = (2 u_rms² L/π) (1 + (1.339 L k1)²)^(-5/6)`` with the one-sided
    normalization ``∫₀^∞ E11 dk1 = u_rms²`` (the standard convention: the value
    at negative ``k1`` is the mirror image, ``E11`` is even).  The longitudinal
    integral scale of the associated correlation is ``L`` and ``k1·E11`` peaks
    at ``k1 L = 0.9146``.  The 1.339 constant is re-derived numerically in the
    test-suite, so the normalization cannot drift silently.
    """
    k = np.asarray(k1, dtype=np.float64)
    return (2.0 * u_rms**2 * L / np.pi) / (1.0 + (_VK_1D_SHAPE * L * k) ** 2) ** (5.0 / 6.0)


def von_karman_peak_wavenumber(L: float = 1.0) -> float:
    """Wavenumber of the maximum of ``k1·E11(k1)``."""
    return math.sqrt(1.5) / (_VK_1D_SHAPE * max(L, 1e-300))


def von_karman_f_peak(u_rms: float, L: float, U: float) -> float:
    """Frequency of the peak of ``f·E11(f)`` under Taylor's hypothesis."""
    return von_karman_peak_wavenumber(L) * U / (2.0 * math.pi)


_KAIMAL_NORM: dict[tuple[float, float], float] = {}


def kaimal_su(f, u_rms: float = 1.0, L: float = 1.0, U: float = 1.0) -> np.ndarray:
    """One-sided Kaimal spectrum ``S_u(f)`` normalized to ``∫ S_u df = u_rms²``.

    Shape: ``f S_u / u_rms² = 6.868 x / (1 + 10.302 x)^(5/3)`` with
    ``x = f L / U``.  The published form is only approximately normalized, so the
    amplitude is fixed by numerical integration on a log grid (constant computed
    in code, never copied); the returned spectrum integrates to ``u_rms²``.
    """
    f = np.asarray(f, dtype=np.float64)
    absf = np.maximum(np.abs(f), 1e-300)
    x = absf * L / U
    shape = 6.868 * x / (1.0 + 10.302 * x) ** (5.0 / 3.0) / absf
    key = (float(L), float(U))
    norm = _KAIMAL_NORM.get(key)
    if norm is None:
        g = np.logspace(-6.0, 4.0, 20001) * U / L
        xg = g * L / U
        s_raw = 6.868 * xg / (1.0 + 10.302 * xg) ** (5.0 / 3.0) / g
        norm = float(np.trapezoid(s_raw, g))
        _KAIMAL_NORM[key] = norm
    return shape * u_rms**2 / norm


def von_karman_e3d(k, u_rms: float = 1.0, L: float = 1.0) -> np.ndarray:
    """3-D isotropic von Kármán energy spectrum ``E(k)``, ``∫₀^∞ E dk = 3/2 u_rms²``."""
    k = np.asarray(k, dtype=np.float64)
    x = L * k
    return _VK_3D_CONST * u_rms**2 * L**5 * k**4 / (1.0 + x**2) ** (17.0 / 6.0)


def von_karman_tensor_spectrum(kvec, u_rms: float = 1.0, L: float = 1.0) -> np.ndarray:
    """Reynolds-stress spectral tensor ``Φ_ij(k)`` of isotropic von Kármán turbulence.

    Args:
        kvec: wavenumber vectors of shape ``(..., 3)``.
        u_rms: one-component rms; isotropic, so ``⟨u_i'u_j'⟩ = δ_ij u_rms²``.
        L: longitudinal integral scale.

    Returns:
        ``(..., 3, 3)`` spectral tensor with ``∫ Φ_ii dk = u_rms²`` per component
        (no summation) and ``∫ trace Φ dk = 3 u_rms²``.
    """
    kvec = np.asarray(kvec, dtype=np.float64)
    if kvec.shape[-1] != 3:
        raise ValueError("kvec must have last dimension 3")
    kmag = np.sqrt((kvec**2).sum(axis=-1))
    E = von_karman_e3d(kmag, u_rms, L)
    eps = 1e-300
    ksq = np.maximum(kmag**2, eps)
    shape = np.eye(3) - kvec[..., :, None] * kvec[..., None, :] / ksq[..., None, None]
    weight = np.where(kmag > 0.0, E / (4.0 * np.pi * ksq), 0.0)
    return weight[..., None, None] * shape


def von_karman_plane_spectrum(
    ky,
    kz,
    u_rms: float = 1.0,
    L: float = 1.0,
    n_kx: int = 201,
    kx_max_factor: float = 10.0,
) -> np.ndarray:
    """Inlet-plane spectral density ``Ψ_ii(k_y, k_z) = ∫ Φ_ii dk_x`` (i = 1,2,3).

    Integrating the 3-D tensor over the direction normal to the inlet plane is
    what makes the plane field statistically consistent with the 3-D target:
    ``∫∫ Ψ_ii dk_y dk_z = u_rms²``.  The ``k_x`` integral uses the midpoint rule on
    a grid of half-width ``kx_max_factor / L`` (beyond that the von Kármán
    integrand has decayed by many orders of magnitude).

    Returns:
        ``(3, len(ky), len(kz))`` array; the ``(k_y,k_z) = (0,0)`` mode is zeroed
        (the synthesis must be zero-mean on the plane as well as in time).
    """
    ky = np.asarray(ky, dtype=np.float64)
    kz = np.asarray(kz, dtype=np.float64)
    kx = np.linspace(-kx_max_factor / L, kx_max_factor / L, n_kx)
    dkx = kx[1] - kx[0]
    KY, KZ = np.meshgrid(ky, kz, indexing="ij")
    out = np.zeros((3,) + KY.shape)
    for kxv in kx:
        k2 = np.maximum(KY**2 + KZ**2 + kxv**2, 1e-300)
        base = von_karman_e3d(np.sqrt(k2), 1.0, L) / (4.0 * np.pi * k2)
        for i, comp in enumerate((kxv, KY, KZ)):
            out[i] += base * (1.0 - comp**2 / k2) * dkx
    out[:, 0, 0] = 0.0
    if ky.size > 1 and kz.size > 1:
        pass
    return out * u_rms**2


# ---------------------------------------------------------------------------
# 2. Turin decomposition / digital filters
# ---------------------------------------------------------------------------


def turin_filter_amplitudes(psd_one_sided, dt: float) -> np.ndarray:
    """Turin (1960) spectral-factorization amplitudes for a digital filter.

    White noise of unit variance sampled with step ``dt`` has two-sided PSD
    ``dt``; the filter ``H`` turning it into a process with one-sided PSD
    ``S1(f)`` must satisfy ``|H|² = S1 / (2 dt)``.  The impulse response
    ``h = IFFT(H)`` is the Turin kernel; convolving white noise with it realizes
    the target correlation exactly in expectation.

    Args:
        psd_one_sided: one-sided PSD on the ``rfft`` frequency grid.
        dt: sampling step.

    Returns:
        Non-negative real amplitudes, zeroed at ``f = 0`` (zero-mean output).
    """
    S1 = np.asarray(psd_one_sided, dtype=np.float64)
    H = np.sqrt(np.maximum(S1, 0.0) / (2.0 * dt))
    H[0] = 0.0
    return H


def _spectrum_parameter(target: str, L_int: float, U: float) -> float:
    """Map the *integral scale* ``L_int = U T_u`` to the spectrum's length parameter.

    For a spectrum finite at ``f = 0`` the integral time scale is exactly
    ``T_u = ∫₀^∞R dτ/R(0) = S2(0)/(2σ²) = S1(0)/(4σ²)``: the two-sided density at
    zero frequency ``S2(0) = S1(0)/2`` carries the whole integral (because
    ``∫₀^∞cos(2πfτ)dτ = δ(f)/2``, and the one-sided ``S1`` is twice the
    two-sided).  Since ``S1(0) ∝ σ²L_p/U``, ``T_u = κL_p/(4U)`` with ``κ`` a pure
    number of the spectral family (``κ = 4`` von Kármán, ``6.868`` Kaimal).  The
    helper measures ``κ`` from the family itself, so ``L_ux`` in
    :class:`SynthesisConfig` is *always* the realized integral scale -- no
    hand-derived factor can drift out of sync.

    (Getting this factor wrong by 2 is exactly the kind of silent error the
    ``L_ux`` gate exists to catch: the measured ``L_ux`` came out at half the
    prescribed value until the factor was corrected -- see the node report.)
    """
    probe = _psd_raw(np.array([1e-12]), target, 1.0, 1.0, U)[0]
    kappa = probe * U  # S1(0) / (u_rms² L_p / U)  == 4 (vK) / 6.868 (Kaimal)
    return 4.0 * L_int / max(kappa, 1e-30)


def _psd_raw(f: np.ndarray, target: str, u_rms: float, L_param: float, U: float) -> np.ndarray:
    """One-sided target PSD for a given *length parameter* (no integral-scale mapping)."""
    if target == "von_karman":
        k1 = 2.0 * np.pi * f / U
        return von_karman_e11(k1, u_rms=u_rms, L=L_param) * (2.0 * np.pi / U)
    if target == "kaimal":
        return kaimal_su(f, u_rms=u_rms, L=L_param, U=U)
    raise ValueError(f"unknown temporal spectrum target {target!r}")


def _one_sided_target_psd(
    f: np.ndarray, target: str, u_rms: float, L_int: float, U: float
) -> np.ndarray:
    """One-sided target PSD in Hz (``∫ S1 df = u_rms²``).

    ``L_int`` is the *integral scale* ``L_ux = U·T_u``; it is mapped onto the
    spectrum's own length parameter by :func:`_spectrum_parameter`, so the target
    curve and the realized ``L_ux`` refer to the same quantity.
    """
    return _psd_raw(f, target, u_rms, _spectrum_parameter(target, L_int, U), U)


def _time_kernel(
    n_time: int, dt: float, target: str, u_rms: float, L: float, U: float
) -> np.ndarray:
    """Turin filter amplitudes on the ``rfft`` grid of an ``n_time`` record.

    A discrete-time process cannot carry the spectral tail above its Nyquist
    frequency ``f_N = 1/(2dt)``: for the von Kármán ``5/6`` spectrum that tail is
    ``1 - ∫₀^{f_N}S1 df / u_rms² ≈ 1.43 (a f_N)^{-2/3}`` (``a = 1.339·2πL/U``), i.e.
    21 % at ``f_N = 13.7 f_peak`` and 10 % at ``f_N = 33 f_peak``.  The kernel is
    therefore built from the target **band-limited and renormalized** to
    ``u_rms²`` over ``[0, f_N]``: the prescribed variance is carried by the
    frequencies the inlet can actually represent, and the in-band *shape* stays
    the analytic von Kármán shape.  :func:`inlet_statistics` reports the tail
    fraction and the error against the *continuous* (un-renormalized) target as
    well, so the renormalisation can never hide a mismatch.
    """
    f = np.fft.rfftfreq(n_time, d=dt)
    S1 = _one_sided_target_psd(f, target, u_rms, L, U)
    resolved = float(np.trapezoid(S1, f)) if f.size > 1 else 0.0
    if resolved > 0.0:
        S1 = S1 * (u_rms**2 / resolved)
    return turin_filter_amplitudes(S1, dt)


def von_karman_acf_time(
    n_time: int, dt: float, u_rms: float = 1.0, L: float = 1.0, U: float = 1.0
) -> np.ndarray:
    """Target temporal autocorrelation ``R_uu(τ)`` implied by ``E11`` (Taylor's hypothesis).

    Built from the *same* Turin kernel the synthesis uses, so the analytic target
    and the realized statistics are guaranteed to refer to the same process: the
    two-sided PSD ``S2 = S1/2`` is transformed back by ``ifft`` (Wiener-Khinchin),
    which gives ``R_uu(τ) = ∫ S1(f) cos(2πfτ) df``.

    Returns:
        ``R_uu(τ_n)`` for ``τ_n = n·dt``, normalized to ``R(0) = u_rms²``.
    """
    S1 = _one_sided_target_psd(np.fft.rfftfreq(n_time, d=dt), "von_karman", u_rms, L, U)
    two_sided = np.concatenate([S1, S1[-2:0:-1]]) * 0.5
    acf = np.fft.ifft(two_sided).real / dt
    return acf * (u_rms**2 / max(acf[0], 1e-300))


# ---------------------------------------------------------------------------
# 3. Divergence control
# ---------------------------------------------------------------------------


def _central_symbol(n: int) -> np.ndarray:
    """Symbol ``i·sin(2πk/n)`` of the 2nd-order central difference on ``n`` points.

    This is the operator an LBM stencil (and :func:`inlet_statistics`) actually
    applies.  Projecting with *this* symbol instead of the continuous ``k``
    makes the *measured* divergence vanish to round-off, not to ``O(k²Δx²)``.
    """
    k = 2.0 * np.pi * np.fft.fftfreq(n)
    return np.sin(k)


def project_inplane_divergence_free(field: np.ndarray) -> np.ndarray:
    """Project the transverse pair ``(v', w')`` onto the discretely solenoidal space.

    ``(v', w') ← (v', w') − ∇_h φ`` with ``φ̂ = div_h / (sin²k_y + sin²k_z)`` and
    ``div_h = i sin(k_y) v̂ + i sin(k_z) ŵ`` the central-difference divergence.
    The streamwise component is untouched (under frozen advection it carries no
    in-plane divergence).

    Args:
        field: ``(..., 3, ny, nz)`` velocity field; leading axes are batched.

    Returns:
        Projected field with the same shape/dtype.
    """
    work = np.asarray(field, dtype=np.float64)
    ny, nz = work.shape[-2], work.shape[-1]
    sy = _central_symbol(ny).reshape(-1, 1)
    sz = _central_symbol(nz).reshape(1, -1)
    denom = sy**2 + sz**2
    vh = np.fft.fft2(work[..., 1, :, :])
    wh = np.fft.fft2(work[..., 2, :, :])
    div = 1j * sy * vh + 1j * sz * wh
    phi = np.where(denom > 0.0, div / np.where(denom > 0.0, denom, 1.0), 0.0)
    out = work.copy()
    # (v,w) - grad_h(phi) with grad_h having symbol i·sin(k): the '+' below is the
    # '-grad' of the derivation, i.e. (i·sin)(-i·sin) = +sin², which cancels div.
    out[..., 1, :, :] = np.fft.ifft2(vh + 1j * sy * phi).real
    out[..., 2, :, :] = np.fft.ifft2(wh + 1j * sz * phi).real
    return out


def inplane_divergence(field: np.ndarray) -> tuple[float, float]:
    """Central-difference in-plane divergence diagnostics, scaled by ``Δx/u_rms``.

    Returns:
        ``(max|∇·u'| Δx / u_rms, rms|∇·u'| Δx / u_rms)`` with ``Δx = 1``.
    """
    work = np.asarray(field, dtype=np.float64)
    v, w = work[..., 1, :, :], work[..., 2, :, :]
    dvdy = 0.5 * (np.roll(v, -1, axis=-2) - np.roll(v, 1, axis=-2))
    dwdz = 0.5 * (np.roll(w, -1, axis=-1) - np.roll(w, 1, axis=-1))
    div = dvdy + dwdz
    urms = float(np.sqrt(np.mean(work[..., 0, :, :] ** 2)))
    scale = 1.0 / max(urms, 1e-300)
    return float(np.abs(div).max() * scale), float(np.sqrt(np.mean(div**2)) * scale)


# ---------------------------------------------------------------------------
# 4. Digital-filter synthesis (primary method)
# ---------------------------------------------------------------------------


@dataclass
class SynthesisConfig:
    """Configuration of the spectrum/moment-calibrated inlet synthesis.

    Attributes:
        ny, nz: inlet-plane resolution (``nz = 1`` gives a 2-D plane).
        n_time: number of time samples (acceptance requires >= 200).
        dt: sampling step (lattice steps per sample).
        u_rms: target rms of each velocity component (isotropic default).
        L_ux: target streamwise integral scale ``L_ux = U·T_u`` in cells.
            ``L_ux = 0`` degenerates to white noise in time and is the documented
            negative control.
        U: mean convection velocity (Taylor hypothesis).
        spectrum: temporal target, ``"von_karman"`` (default) or ``"kaimal"``.
        L_spatial: integral scale of the plane field (defaults to ``L_ux``).
        seed: RNG seed; identical seeds give bit-identical fields.
        divergence_free: apply the discrete in-plane Helmholtz projection.
        stress_target: optional 3x3 target Reynolds stress (overrides ``u_rms``).
    """

    ny: int = 64
    nz: int = 64
    n_time: int = 2048
    dt: float = 0.15
    u_rms: float = 0.1
    L_ux: float = 2.0
    U: float = 1.0
    spectrum: str = "von_karman"
    L_spatial: float | None = None
    seed: int = 0
    divergence_free: bool = True
    stress_target: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.ny < 8 or self.nz < 1:
            raise ValueError("ny >= 8 and nz >= 1 required")
        if self.spectrum not in ("von_karman", "kaimal"):
            raise ValueError(f"unknown spectrum {self.spectrum!r}")
        if self.L_ux < 0.0:
            raise ValueError("L_ux must be >= 0")
        if self.dt <= 0.0:
            raise ValueError("dt must be > 0")
        if self.u_rms <= 0.0:
            raise ValueError("u_rms must be > 0")
        if self.stress_target is None:
            self.stress_target = np.eye(3) * self.u_rms**2

    @property
    def target_stress(self) -> np.ndarray:
        assert self.stress_target is not None
        return np.asarray(self.stress_target, dtype=np.float64)

    @property
    def L_eff(self) -> float:
        """Length scale used for the plane field (``L_spatial`` or ``L_ux``)."""
        return self.L_spatial if self.L_spatial is not None else max(self.L_ux, 1e-6)


def _component_variance(field: np.ndarray) -> np.ndarray:
    """Variance of each of the 3 velocity components, shape ``(3,)``.

    ``field`` is ``(n_time, 3, ny, nz)``, so a plain ``reshape(3, -1)`` would
    reinterpret the flat buffer and mix components with time -- the classic
    layout trap.  The transpose makes the component axis leading first.
    """
    return field.transpose(1, 0, 2, 3).reshape(3, -1).var(axis=1)


def _match_moments(field: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rescale ``field`` onto the target ``R_ij`` diagonal *without* breaking solenoidality.

    The in-plane divergence only involves ``(v', w')``, so:

    * the streamwise component is rescaled by its own factor (harmless);
    * the transverse pair is rescaled by **one common factor** derived from the
      mean of the two target rms values.

    The common transverse factor leaves the ratio ``⟨v'v'⟩/⟨w'w'⟩`` untouched --
    which is exactly what the Helmholtz projection preserves -- so a projected
    field stays divergence-free to round-off while its transverse stresses match
    the target.  An anisotropic ``(vv, ww)`` target that is *not* in the ratio the
    projection produces cannot be honoured without re-introducing divergence: the
    residual shows up in :func:`inlet_statistics` (``stress_rel_error`` /
    ``divergence_max``), which is the documented trade-off.
    """
    target_sd = np.sqrt(np.maximum(np.diag(target), 0.0))
    cur = np.sqrt(np.maximum(_component_variance(field), 1e-300))
    out = field.copy()
    out[:, 0] *= target_sd[0] / cur[0]
    out[:, 1:] *= float(np.sqrt(np.mean(target_sd[1:] ** 2)) / np.sqrt(np.mean(cur[1:] ** 2)))
    return out


class SpectralInletSynthesis:
    """Digital-filter (Turin) inlet synthesis with spectrum and moment calibration.

    Pipeline (:meth:`synthesize`):

    1. white Gaussian noise ``w(t, i, y, z)`` (unit variance, seeded);
    2. **temporal** digital filter from the Turin factorization of the target
       spectrum ``E11`` (or Kaimal) under Taylor's hypothesis -- this fixes
       ``E(f)``, ``R_uu(τ)`` and ``L_ux = U·T_u``;
    3. **spatial** digital filter from the plane target
       ``Ψ_ii(k_y,k_z) = ∫Φ_ii dk_x`` -- this fixes the plane length scales and
       the 3-D-consistent spatial correlation;
    4. discrete in-plane Helmholtz projection (``divergence_free=True``);
    5. per-component rescaling onto the target ``R_ij`` diagonal (Lund 1998
       moment matching).  The energy the projection removes is reported by
       :func:`inlet_statistics`, not hidden.
    """

    def __init__(self, config: SynthesisConfig | None = None, **kwargs) -> None:
        if config is not None:
            self.config = replace(config, **kwargs) if kwargs else config
        else:
            self.config = SynthesisConfig(**kwargs)
        self._field: np.ndarray | None = None

    # -- filters ---------------------------------------------------------

    def _temporal_amplitudes(self) -> np.ndarray:
        cfg = self.config
        if cfg.L_ux <= 0.0:  # negative control: white noise in time
            H = np.ones(cfg.n_time // 2 + 1)
            H[0] = 0.0
            return H
        return _time_kernel(cfg.n_time, cfg.dt, cfg.spectrum, cfg.u_rms, cfg.L_ux, cfg.U)

    def _spatial_amplitudes(self) -> np.ndarray:
        cfg = self.config
        ky = 2.0 * np.pi * np.fft.fftfreq(cfg.ny)
        kz = 2.0 * np.pi * np.fft.fftfreq(cfg.nz) if cfg.nz > 1 else np.zeros(1)
        psi = von_karman_plane_spectrum(ky, kz, u_rms=1.0, L=max(cfg.L_eff, 1e-6))
        # With unit grid spacing, |G|² = (2π)² Ψ makes the filtered unit-variance
        # white noise carry ∫∫Ψ dk_y dk_z = u_rms².
        return 2.0 * np.pi * np.sqrt(np.maximum(psi, 0.0))

    # -- synthesis -------------------------------------------------------

    def synthesize(self) -> np.ndarray:
        """Generate the inlet field with shape ``(n_time, 3, ny, nz)``."""
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        noise = rng.standard_normal((cfg.n_time, 3, cfg.ny, cfg.nz))

        H = self._temporal_amplitudes()
        field = np.fft.irfft(
            np.fft.rfft(noise, axis=0) * H[:, None, None, None], n=cfg.n_time, axis=0
        )

        G = self._spatial_amplitudes()
        field = np.fft.ifft2(
            np.fft.fft2(field, axes=(-2, -1)) * G[None, :, :, :], axes=(-2, -1)
        ).real
        field = np.ascontiguousarray(field)

        if cfg.divergence_free:
            field = project_inplane_divergence_free(field)

        field = _match_moments(field, cfg.target_stress)
        self._field = field
        return field

    def sha256(self) -> str:
        """``sha256`` of the generated field (reproducibility stamp)."""
        if self._field is None:
            self.synthesize()
        assert self._field is not None
        return hashlib.sha256(
            np.ascontiguousarray(self._field, dtype=np.float64).tobytes()
        ).hexdigest()


# ---------------------------------------------------------------------------
# 5. DSRFG (auxiliary generator)
# ---------------------------------------------------------------------------


@dataclass
class DSRFGInlet:
    """Discretizing-and-Synthesizing-Random-Flow-Generation inlet (auxiliary).

    Smirnov *et al.* (2001) / Huang *et al.* (2010): the fluctuation is a sum of
    ``n_modes`` Fourier modes whose wavenumber magnitudes are drawn from the von
    Kármán spectral density and whose directions/phases are random; each
    snapshot is then rescaled onto the target ``R_ij`` (Lund whitening).  Here
    the snapshot sequence is additionally passed through the *same* Turin
    temporal filter as the digital-filter branch, so both generators can be
    judged against the identical spectral acceptance criteria.

    Consequence (documented limitation): the temporal evolution is filter-driven
    rather than frozen-convection-driven, so the Taylor space-time coherence is
    only reproduced statistically.  This is why the branch is the auxiliary one.
    """

    config: SynthesisConfig = field(default_factory=SynthesisConfig)
    n_modes: int = 512

    def synthesize(self) -> np.ndarray:
        cfg = self.config
        rng = np.random.default_rng(cfg.seed + 1009)
        ny, nz = cfg.ny, cfg.nz
        Lsp = max(cfg.L_eff, 1e-6)

        # Wavenumber magnitudes drawn from the von Kármán 3-D spectral density.
        k_lo, k_hi = 1e-2 / Lsp, 8.0 / Lsp
        k_grid = np.logspace(np.log10(k_lo), np.log10(k_hi), 4000)
        pdf = von_karman_e3d(k_grid, 1.0, Lsp) * k_grid**2
        cdf = np.cumsum(pdf)
        cdf /= cdf[-1]
        kmag = np.interp(rng.random(self.n_modes), cdf, k_grid)
        dk = k_hi / 4000.0
        amp = np.sqrt(np.maximum(von_karman_e3d(kmag, 1.0, Lsp) * dk / self.n_modes, 0.0))

        cos_t = 2.0 * rng.random(self.n_modes) - 1.0
        sin_t = np.sqrt(np.maximum(1.0 - cos_t**2, 0.0))
        phi = 2.0 * np.pi * rng.random(self.n_modes)
        kvec = kmag[:, None] * np.stack([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t], axis=1)

        y = np.arange(ny)[None, :]
        z = np.arange(nz)[None, :]
        # Separable form: cos(a_m + b_m + ph) = cosA cos(B+ph) - sinA sin(B+ph),
        # so each snapshot is two small matrix products instead of a mode loop.
        cosA = np.cos(kvec[:, 1][None, :] * y.T)  # (ny, M)
        sinA = np.sin(kvec[:, 1][None, :] * y.T)
        kzcol = kvec[:, 2][:, None]  # (M, 1)
        kx_conv = kvec[:, 0]

        field = np.empty((cfg.n_time, 3, ny, nz), dtype=np.float64)
        for it in range(cfg.n_time):
            t = it * cfg.dt
            for comp in range(3):
                ph = 2.0 * np.pi * rng.random(self.n_modes) + kx_conv * cfg.U * t
                argB = kzcol * z + ph[:, None]
                field[it, comp] = cosA @ (amp[:, None] * np.cos(argB)) - sinA @ (
                    amp[:, None] * np.sin(argB)
                )

        # Lund (1998) whitening + target colouring onto R_ij.
        flat = field.reshape(cfg.n_time, 3, -1)
        R = self._sample_stress(field)
        evals, evecs = np.linalg.eigh(0.5 * (R + R.T))
        whitening = (evecs / np.sqrt(np.clip(evals, 1e-300, None))) @ evecs.T
        target = cfg.target_stress
        ev_t, evec_t = np.linalg.eigh(0.5 * (target + target.T))
        colour = (evec_t * np.sqrt(np.clip(ev_t, 0.0, None))) @ evec_t.T
        field = np.einsum("ij,njk->nik", colour @ whitening, flat).reshape(cfg.n_time, 3, ny, nz)

        if cfg.L_ux > 0.0:
            H = _time_kernel(cfg.n_time, cfg.dt, cfg.spectrum, cfg.u_rms, cfg.L_ux, cfg.U)
            field = np.fft.irfft(
                np.fft.rfft(field, axis=0) * H[:, None, None, None], n=cfg.n_time, axis=0
            )
        if cfg.divergence_free:
            field = project_inplane_divergence_free(field)
        return _match_moments(field, target)

    @staticmethod
    def _sample_stress(field: np.ndarray) -> np.ndarray:
        flat = field.reshape(field.shape[0], 3, -1).transpose(1, 0, 2).reshape(3, -1)
        flat = flat - flat.mean(axis=1, keepdims=True)
        return (flat @ flat.T) / flat.shape[1]


# ---------------------------------------------------------------------------
# 6. Statistical sampler and machine gates
# ---------------------------------------------------------------------------


def periodogram(series: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """One-sided PSD ``E(f)`` in ``u²/Hz``, averaged over all leading axes.

    ``E(f) = 2 dt |X_k|² / n`` on the ``rfft`` grid, with the DC and (for even
    ``n``) Nyquist bins half-weighted; ``∫ E df`` then equals the variance of the
    record.  Averaging over points of the inlet plane is the standard
    homogeneous-turbulence estimator: each plane point is one realization of the
    same stationary process.
    """
    x = np.asarray(series, dtype=np.float64)
    n = x.shape[-1]
    x = x - x.mean(axis=-1, keepdims=True)
    spec = np.fft.rfft(x, axis=-1)
    psd = 2.0 * np.abs(spec) ** 2 * dt / n
    psd[..., 0] *= 0.5
    if n % 2 == 0:
        psd[..., -1] *= 0.5
    f = np.fft.rfftfreq(n, d=dt)
    axes = tuple(range(x.ndim - 1))
    return f, psd.mean(axis=axes)


def autocorrelation(series: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    """Normalized autocorrelation of the last axis, averaged over leading axes.

    Uses zero-padded FFTs so that the estimate is unbiased for lags up to
    ``max_lag`` (no circular wrap-around).
    """
    x = np.asarray(series, dtype=np.float64)
    n = x.shape[-1]
    x = x - x.mean(axis=-1, keepdims=True)
    nfft = 2 * n
    spec = np.fft.rfft(x, n=nfft, axis=-1)
    acf = np.fft.irfft(spec * np.conj(spec), n=nfft, axis=-1)[..., :n]
    axes = tuple(range(x.ndim - 1))
    acf = acf.mean(axis=axes)
    acf /= max(acf[0], 1e-300)
    if max_lag is not None:
        acf = acf[: max_lag + 1]
    return acf


def exponential_fit(tau: np.ndarray, acf: np.ndarray) -> tuple[float, float]:
    """Least-squares fit of ``A·exp(-τ/T)`` to ``acf`` (pure numpy).

    ``T`` is found on a log grid (the model is linear in ``A`` given ``T``), then
    refined by a golden-section-free fine grid; ``R²`` is reported on the fitted
    range.

    Returns:
        ``(R², T)``.
    """
    tau = np.asarray(tau, dtype=np.float64)
    y = np.asarray(acf, dtype=np.float64)
    if y[0] < 0.0:
        y = -y
    scale = tau[-1] if tau[-1] > 0 else 1.0

    def r2_for(T: float) -> tuple[float, float]:
        basis = np.exp(-tau / max(T, 1e-12))
        denom = float(np.dot(basis, basis))
        if denom <= 0.0:  # pragma: no cover
            return -np.inf, 0.0
        A = float(np.dot(basis, y) / denom)
        resid = y - A * basis
        ss_res = float(np.dot(resid, resid))
        ss_tot = float(np.dot(y - y.mean(), y - y.mean()))
        return 1.0 - ss_res / max(ss_tot, 1e-300), A

    best = (-np.inf, 0.0)
    for T in np.logspace(-3.0, 2.0, 400) * scale:
        r2, A = r2_for(T)
        if r2 > best[0]:
            best = (r2, T)
    # local refinement
    T0 = best[1]
    for T in np.linspace(max(T0 * 0.5, 1e-9), T0 * 1.5, 401):
        r2, A = r2_for(T)
        if r2 > best[0]:
            best = (r2, T)
    return best[0], best[1]


def third_octave_bands(f_peak: float, low: float = 0.2, high: float = 5.0) -> np.ndarray:
    """Band ``(f_lo, f_hi)`` edges of the 1/3-octave bands inside ``[low, high]·f_peak``."""
    edges = []
    n = -30
    while n <= 30:
        fc = f_peak * 2.0 ** (n / 3.0)
        lo, hi = fc * 2.0 ** (-1.0 / 6.0), fc * 2.0 ** (1.0 / 6.0)
        if lo >= low * f_peak and hi <= high * f_peak:
            edges.append((lo, hi))
        n += 1
    return np.array(edges)


def _band_energy(f: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> float:
    mask = (f >= lo) & (f <= hi)
    if mask.sum() < 2:
        # Single-bin (or empty) band: use the nearest bin value times the width.
        idx = int(np.argmin(np.abs(f - 0.5 * (lo + hi))))
        return float(psd[idx] * (hi - lo))
    return float(np.trapezoid(psd[mask], f[mask]))


@dataclass
class InletStatistics:
    """Sampled inlet statistics and their distance from the target."""

    f: np.ndarray
    psd: np.ndarray
    target_psd: np.ndarray
    band_edges: np.ndarray
    band_error: np.ndarray
    band_centres: np.ndarray
    band_error_continuous: np.ndarray
    max_band_error: float
    max_band_error_continuous: float
    tail_fraction: float
    stress_sample: np.ndarray
    stress_target: np.ndarray
    stress_rel_error: float
    acf: np.ndarray
    acf_tau: np.ndarray
    acf_exp_r2: float
    acf_exp_time: float
    L_ux_sample: float
    L_ux_target: float
    divergence_max: float
    divergence_rms: float

    def as_dict(self) -> dict:
        return {
            "max_band_error": self.max_band_error,
            "band_centres": self.band_centres.tolist(),
            "band_edges": self.band_edges.tolist(),
            "band_error": self.band_error.tolist(),
            "band_error_continuous": self.band_error_continuous.tolist(),
            "max_band_error_continuous": self.max_band_error_continuous,
            "tail_fraction": self.tail_fraction,
            "stress_sample": self.stress_sample.tolist(),
            "stress_target": self.stress_target.tolist(),
            "stress_rel_error": self.stress_rel_error,
            "acf_exp_r2": self.acf_exp_r2,
            "acf_exp_time": self.acf_exp_time,
            "L_ux_sample": self.L_ux_sample,
            "L_ux_target": self.L_ux_target,
            "divergence_max": self.divergence_max,
            "divergence_rms": self.divergence_rms,
        }


def inlet_statistics(
    field: np.ndarray,
    config: SynthesisConfig,
    *,
    fit_factor: float = 1.5,
) -> InletStatistics:
    """Sample ``E(f)``, ``R_ij``, ``L_ux``, ``R_uu(τ)`` and the divergence from a field.

    Args:
        field: ``(n_time, 3, ny, nz)`` synthesized inlet field.
        config: the synthesis configuration (targets, ``dt``, ``U``, ``L_ux``).
        fit_factor: exponential-fit window in units of the *target* integral
            time scale ``T_u = L_ux/U`` (pre-registered, not tuned on the data).

    Returns:
        :class:`InletStatistics`.
    """
    field = np.asarray(field, dtype=np.float64)
    n_time = field.shape[0]
    u = field[:, 0]  # (n_time, ny, nz)

    # Reynolds stresses: time-mean removed per (component, plane point), then the
    # time average of the outer product (layout-safe transposition).
    flat = field.transpose(1, 0, 2, 3).reshape(3, -1)
    flat = flat - flat.mean(axis=1, keepdims=True)
    R = (flat @ flat.T) / flat.shape[1]

    # Spectrum / correlation: the samplers expect the *series* axis last, so the
    # plane axes are flattened first and time is moved to the end.
    u_series = u.transpose(1, 2, 0).reshape(-1, n_time)
    f, psd = periodogram(u_series, config.dt)
    target_psd = _one_sided_target_psd(
        f, config.spectrum, config.u_rms, max(config.L_ux, 1e-9), config.U
    )

    f_peak = von_karman_f_peak(config.u_rms, max(config.L_ux, 1e-9), config.U)
    edges = third_octave_bands(f_peak)
    resolved = float(np.trapezoid(target_psd, f))
    tail_fraction = max(0.0, 1.0 - resolved / config.u_rms**2)
    target_resolved = target_psd * (config.u_rms**2 / resolved) if resolved > 0 else target_psd
    errors, errors_cont, centres = [], [], []
    for lo, hi in edges:
        e_s = _band_energy(f, psd, lo, hi)
        e_t = _band_energy(f, target_resolved, lo, hi)
        e_c = _band_energy(f, target_psd, lo, hi)
        if e_t > 0.0:
            errors.append(abs(e_s - e_t) / e_t)
            centres.append(math.sqrt(lo * hi))
        if e_c > 0.0:
            errors_cont.append(abs(e_s - e_c) / e_c)

    band_error = np.array(errors) if errors else np.array([np.nan])
    band_error_continuous = np.array(errors_cont) if errors_cont else np.array([np.nan])
    band_centres = np.array(centres) if centres else np.array([np.nan])

    # Temporal correlation and integral scale.
    T_u_target = config.L_ux / config.U if config.L_ux > 0 else 0.0
    n_lag = n_time // 2
    acf = autocorrelation(u_series, max_lag=n_lag - 1)
    tau = np.arange(n_lag) * config.dt
    if config.L_ux > 0:
        fit_hi = max(2, int(round(fit_factor * T_u_target / config.dt)))
        fit_hi = min(fit_hi, n_lag - 1)
        r2, T_fit = exponential_fit(tau[: fit_hi + 1], acf[: fit_hi + 1])
        # Integral scale: integrate the ACF over a *pre-registered* window of
        # 4 T_u (not to the end of the record, where the estimator is
        # noise-dominated and the von Kármán ACF has already decayed by >99 %).
        n_int = min(n_lag - 1, max(2, int(round(4.0 * T_u_target / config.dt))))
        L_ux_sample = config.U * float(np.trapezoid(acf[: n_int + 1], tau[: n_int + 1]))
    else:
        r2, T_fit, L_ux_sample = float("nan"), float("nan"), 0.0

    div_max, div_rms = inplane_divergence(field)
    return InletStatistics(
        f=f,
        psd=psd,
        target_psd=target_psd,
        band_edges=edges,
        band_error=band_error,
        band_error_continuous=band_error_continuous,
        max_band_error_continuous=float(np.nanmax(band_error_continuous)),
        tail_fraction=float(tail_fraction),
        band_centres=band_centres,
        max_band_error=float(np.nanmax(band_error)),
        stress_sample=R,
        stress_target=config.target_stress,
        stress_rel_error=float(
            np.max(
                np.abs(np.diag(R) - np.diag(config.target_stress)) / np.diag(config.target_stress)
            )
        ),
        acf=acf,
        acf_tau=tau,
        acf_exp_r2=float(r2),
        acf_exp_time=float(T_fit),
        L_ux_sample=float(L_ux_sample),
        L_ux_target=float(config.L_ux),
        divergence_max=div_max,
        divergence_rms=div_rms,
    )


# Acceptance thresholds of node p234w2-n05 (pre-registered, do not tune).
GATES: dict[str, float] = {
    "spectrum_band_error": 0.15,  # (a) 1/3-octave band error vs von Kármán
    "stress_rel_error": 0.10,  # (b) u'u'/v'v'/w'w' relative error
    "integral_scale_error": 0.20,  # (c) L_ux vs target
    "acf_exp_r2": 0.95,  # (d) R_uu(τ) exponential fit (lower bound)
    "divergence_max": 0.05,  # (e) max|∇·u'|Δx/u_rms
    "negative_control_min": 0.50,  # (g) L_ux = 0 ⇒ band error must exceed this
    "route_consistency": 0.02,  # (3) two application routes agree within 2 %
}


def evaluate_gates(
    stats: InletStatistics, *, sha_a: str | None = None, sha_b: str | None = None
) -> dict:
    """Turn :class:`InletStatistics` into the node's machine verdicts.

    Returns:
        Dict with every gate's measured value and boolean, plus ``all_pass``.
    """
    gates = GATES
    integral_err = (
        abs(stats.L_ux_sample - stats.L_ux_target) / stats.L_ux_target
        if stats.L_ux_target > 0
        else float("nan")
    )
    verdict = {
        "spectrum_band_error": stats.max_band_error,
        "spectrum_band_pass": bool(stats.max_band_error < gates["spectrum_band_error"]),
        "stress_rel_error": stats.stress_rel_error,
        "stress_pass": bool(stats.stress_rel_error < gates["stress_rel_error"]),
        "integral_scale_rel_error": integral_err,
        "integral_scale_pass": bool(integral_err < gates["integral_scale_error"]),
        "acf_exp_r2": stats.acf_exp_r2,
        "acf_pass": bool(stats.acf_exp_r2 > gates["acf_exp_r2"]),
        "divergence_max": stats.divergence_max,
        "divergence_pass": bool(stats.divergence_max < gates["divergence_max"]),
    }
    if sha_a is not None and sha_b is not None:
        verdict["sha256_a"] = sha_a
        verdict["sha256_b"] = sha_b
        verdict["reproducible_pass"] = bool(sha_a == sha_b)
    key = [
        "spectrum_band_pass",
        "stress_pass",
        "integral_scale_pass",
        "acf_pass",
        "divergence_pass",
    ]
    if "reproducible_pass" in verdict:
        key.append("reproducible_pass")
    verdict["all_pass"] = all(verdict[k] for k in key)
    return verdict
