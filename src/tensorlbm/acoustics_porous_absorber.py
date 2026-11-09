r"""Delany-Bazley porous absorbers, multilayer transfer matrices (TMM) and a
locally reacting impedance boundary for the lattice Boltzmann solver.

Physical problem
----------------
A compartment / engine-room bulkhead is covered by a porous sound-absorbing
lining.  The engineering questions are: (i) what are the absorption
coefficient ``alpha(f) = 1 - |R(f)|^2`` and the insertion loss
``IL = 10 log10(1/(1-alpha))`` in the target band, (ii) where does the acoustic
energy go (reflection vs dissipation), and (iii) under a shipboard areal-mass
constraint, which (thickness, flow resistivity) pair is optimal?

Model chain
-----------
1. **Delany-Bazley (1970) empirical characteristic impedance / wavenumber**
   (:func:`delany_bazley`) as a function of ``X = rho0 * f / sigma``
   (``sigma`` = static flow resistivity, Pa*s/m^2, correlation valid for
   roughly ``0.01 < X < 1``).
2. **Transfer-matrix method (TMM)** (:func:`layer_transfer_matrix`,
   :func:`surface_impedance`): every layer of thickness ``d`` is represented by

   .. math::

       \begin{bmatrix} p(0) \\ v(0) \end{bmatrix} =
       \begin{bmatrix} \cos k_c d & -i Z_c \sin k_c d \\
       -i \sin k_c d / Z_c & \cos k_c d \end{bmatrix}
       \begin{bmatrix} p(d) \\ v(d) \end{bmatrix},

   layers are chained by matrix multiplication and a rigid / anechoic backing
   closes the recursion.  The surface impedance gives
   ``R = (Z_s - rho0 c0) / (Z_s + rho0 c0)`` and ``alpha = 1 - |R|^2``.
3. **Locally reacting LBM impedance boundary** (:func:`fit_impedance_filter`,
   :class:`DuctLBM`): the frequency dependent surface impedance is fitted by a
   real rational function, discretised with the bilinear transform into an IIR
   digital filter, and imposed at the wall node through the pressure/velocity
   relation ``p' = Z * v_n'`` solved implicitly for the unknown incoming
   population.  The boundary only sees the local normal velocity, hence it is
   *locally reacting* -- exactly the hypothesis under which a TMM surface
   impedance is defined.
4. **Verification**: a 1-D duct LBM forced by a broadband pulse yields the
   reflectance spectrum, compared against the TMM prediction.  The analytic
   single-layer closed form and the exact layer-splitting invariance provide
   machine-precision anchors.

Sign / time convention (stated once, used everywhere)
-----------------------------------------------------
All phasors use ``exp(-i omega t)``.  A wave travelling in ``+x`` is
``exp(i(kx - omega t))``, so **dissipation requires Im(k) > 0** and a
mass-like reactance requires ``Im(Z) > 0``.  Delany-Bazley is implemented as

.. math::

    Z_c = \rho_0 c_0 (1 + 0.0571 X^{-0.754} + i\,0.0870 X^{-0.732}), \quad
    k_c = \frac{2\pi f}{c_0}(1 + 0.0978 X^{-0.700} + i\,0.189 X^{-0.595}).

The frequently tabulated ``-j`` form of the same correlation belongs to the
opposite (``exp(+i omega t)``) convention; ``alpha`` and ``|R|`` are invariant
under the conjugation (see ``conjugate=True`` and the corresponding test).

Scope / honesty notes
---------------------
* The Delany-Bazley correlation is *empirical*: it carries its own 10-20 %
  scatter against fibrous-material measurements, so no claim about agreement
  with any specific specimen's measured curve is made here.  What is verified
  numerically is (a) the TMM algebra (machine precision against closed forms),
  (b) model-to-model agreement (Delany-Bazley vs Miki 1990), (c) the
  time-domain LBM impedance boundary against the TMM of the *same* surface
  impedance (two independent numerical paths, no shared quadrature), and
  (d) passivity / energy bookkeeping / asymptotic limits.
* The LBM duct is 1-D (D1Q3) linear acoustics: it validates the boundary
  condition and the energy budget, not a 3-D room-acoustic field.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "DEFAULT_RHO0",
    "DEFAULT_C0",
    "DB_COEFFICIENTS",
    "delany_bazley",
    "db_validity_ratio",
    "Layer",
    "Lining",
    "layer_transfer_matrix",
    "transfer_matrix",
    "surface_impedance",
    "reflection_coefficient",
    "absorption_coefficient",
    "insertion_loss",
    "band_statistics",
    "optimize_lining",
    "DuctLBM",
    "ImpedanceWall",
    "RigidWall",
    "measure_reflection",
    "sweep_absorption",
]

#: Default air properties (20 C, 1 atm).
DEFAULT_RHO0 = 1.204            # kg/m^3
DEFAULT_C0 = 343.0              # m/s

#: Delany & Bazley (1970), Applied Acoustics 3, 105-116, eqs. (3)-(4).
DB_COEFFICIENTS: Dict[str, float] = {
    "zr": 0.0571, "zr_exp": -0.754,
    "zi": 0.0870, "zi_exp": -0.732,
    "kr": 0.0978, "kr_exp": -0.700,
    "ki": 0.1890, "ki_exp": -0.595,
}

# ---------------------------------------------------------------------------
# 1. Semi-empirical porous-material models
# ---------------------------------------------------------------------------
def _as_float_array(f: Sequence[float] | float) -> np.ndarray:
    """Return ``f`` as a float array, preserving the input shape (0-d stays 0-d)."""
    return np.asarray(f, dtype=float)


def delany_bazley(
    f: Sequence[float] | float,
    sigma: float,
    rho0: float = DEFAULT_RHO0,
    c0: float = DEFAULT_C0,
    *,
    conjugate: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Delany-Bazley characteristic impedance and complex wavenumber.

    Parameters
    ----------
    f : array_like
        Frequency in Hz (may be any shape; returned arrays match it).
    sigma : float
        Static flow resistivity in Pa*s/m^2 (N*s/m^4).
    rho0, c0 : float
        Ambient density and speed of sound.
    conjugate : bool
        Return the complex conjugate pair (the ``exp(+i omega t)`` tabulation).
        ``|R|`` and ``alpha`` are invariant under this switch.

    Returns
    -------
    Zc : ndarray of complex
        Characteristic impedance of the material.
    kc : ndarray of complex
        Complex wavenumber, ``Im(kc) > 0`` meaning attenuation along ``+x``.
    """
    f_arr = _as_float_array(f)
    if sigma <= 0.0:
        raise ValueError("flow resistivity sigma must be positive")
    if np.any(f_arr <= 0.0):
        raise ValueError("frequency must be positive")
    X = rho0 * f_arr / sigma
    sgn = -1.0 if conjugate else 1.0
    c = DB_COEFFICIENTS
    Zc = rho0 * c0 * (
        1.0
        + c["zr"] * X ** c["zr_exp"]
        + sgn * 1j * c["zi"] * X ** c["zi_exp"]
    )
    kc = (2.0 * math.pi * f_arr / c0) * (
        1.0
        + c["kr"] * X ** c["kr_exp"]
        + sgn * 1j * c["ki"] * X ** c["ki_exp"]
    )
    return Zc, kc


def db_validity_ratio(f: Sequence[float] | float, sigma: float, rho0: float = DEFAULT_RHO0) -> np.ndarray:
    """Dimensionless Delany-Bazley argument ``X = rho0 f / sigma``.

    The correlation was fitted on ``0.01 < X < 1``; values outside that window
    are extrapolation and must be flagged in any report using them.
    """
    return rho0 * _as_float_array(f) / float(sigma)


# ---------------------------------------------------------------------------
# 2. Multilayer transfer-matrix method
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Layer:
    """One homogeneous layer of a locally reacting multilayer lining.

    ``model`` is ``"delany_bazley"`` (``sigma`` required) or ``"air"``
    (a plain cavity, optionally with a small loss factor).
    """

    thickness: float
    sigma: float = 0.0
    model: str = "delany_bazley"
    loss_factor: float = 0.0      # air only: Im(k)/Re(k), default lossless

    def __post_init__(self) -> None:
        if self.thickness <= 0.0:
            raise ValueError("layer thickness must be positive")
        if self.model not in ("delany_bazley", "air"):
            raise ValueError(f"unknown layer model {self.model!r}")

    def wave_properties(
        self, f: Sequence[float] | float,
        rho0: float = DEFAULT_RHO0, c0: float = DEFAULT_C0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(Zc, kc)`` of the bulk material of this layer."""
        if self.model == "delany_bazley":
            return delany_bazley(f, self.sigma, rho0, c0)
        f_arr = _as_float_array(f)
        kc = (1.0 + 1j * self.loss_factor) * 2.0 * math.pi * f_arr / c0
        return np.full(f_arr.shape, rho0 * c0, dtype=complex), kc.astype(complex)

    @property
    def areal_mass(self) -> float:
        """Areal mass proxy ``rho_bulk * d`` (kg/m^2) for fibre bulk 40 kg/m^3.

        Used only for the shipboard weight constraint in
        :func:`optimize_lining`; it is *not* a structural claim.  An air cavity
        carries no fibre mass.
        """
        if self.model == "air":
            return 0.0
        return 40.0 * self.thickness


@dataclass(frozen=True)
class Lining:
    """A sequence of layers, front (facing the sound field) first."""

    layers: Tuple[Layer, ...]

    def __init__(self, layers: Sequence[Layer]):
        object.__setattr__(self, "layers", tuple(layers))
        if not self.layers:
            raise ValueError("a lining needs at least one layer")

    @property
    def total_thickness(self) -> float:
        return float(sum(layer.thickness for layer in self.layers))

    @property
    def areal_mass(self) -> float:
        return float(sum(layer.areal_mass for layer in self.layers))

    def surface_impedance(
        self, f: Sequence[float] | float, rho0: float = DEFAULT_RHO0,
        c0: float = DEFAULT_C0, backing: str | complex = "rigid",
    ) -> np.ndarray:
        return surface_impedance(self.layers, f, rho0, c0, backing=backing)

    def reflection(
        self, f: Sequence[float] | float, rho0: float = DEFAULT_RHO0,
        c0: float = DEFAULT_C0, backing: str | complex = "rigid",
    ) -> np.ndarray:
        return reflection_coefficient(
            self.surface_impedance(f, rho0, c0, backing), rho0, c0
        )

    def absorption(
        self, f: Sequence[float] | float, rho0: float = DEFAULT_RHO0,
        c0: float = DEFAULT_C0, backing: str | complex = "rigid",
    ) -> np.ndarray:
        return absorption_coefficient(self.reflection(f, rho0, c0, backing))


def layer_transfer_matrix(
    layer: Layer, f: Sequence[float] | float,
    rho0: float = DEFAULT_RHO0, c0: float = DEFAULT_C0,
) -> np.ndarray:
    """Transfer matrix of one layer, shape ``(nf, 2, 2)`` (complex).

    Maps ``[p, v]`` at the far (back) face to ``[p, v]`` at the near face for
    the ``exp(-i omega t)`` convention (see module docstring).
    """
    Zc, kc = layer.wave_properties(f, rho0, c0)
    kd = np.asarray(kc) * layer.thickness
    cos, sin = np.cos(kd), np.sin(kd)
    T = np.empty(kd.shape + (2, 2), dtype=complex)
    T[..., 0, 0] = cos
    T[..., 0, 1] = -1j * Zc * sin
    T[..., 1, 0] = -1j * sin / Zc
    T[..., 1, 1] = cos
    return T


def transfer_matrix(
    layers: Sequence[Layer], f: Sequence[float] | float,
    rho0: float = DEFAULT_RHO0, c0: float = DEFAULT_C0,
) -> np.ndarray:
    """Total transfer matrix of a layer stack (near face <- back face)."""
    mats = [layer_transfer_matrix(layer, f, rho0, c0) for layer in layers]
    total = mats[0]
    for mat in mats[1:]:
        total = total @ mat
    return total


def surface_impedance(
    layers: Sequence[Layer] | Lining, f: Sequence[float] | float,
    rho0: float = DEFAULT_RHO0, c0: float = DEFAULT_C0,
    backing: str | complex = "rigid",
) -> np.ndarray:
    """Plane-wave surface impedance seen by the incident field.

    ``backing`` is ``"rigid"`` (``v = 0``, hard wall), ``"anechoic"``
    (``Z = rho0 c0``) or an explicit complex impedance.
    """
    layers = layers.layers if isinstance(layers, Lining) else tuple(layers)
    T = transfer_matrix(layers, f, rho0, c0)
    if isinstance(backing, str):
        if backing == "rigid":
            return T[..., 0, 0] / T[..., 1, 0]
        if backing == "anechoic":
            z_back = np.full(T.shape[:-2], rho0 * c0, dtype=complex)
        else:
            raise ValueError(f"unknown backing {backing!r}")
    else:
        f_arr = _as_float_array(f)
        z_back = np.broadcast_to(np.asarray(backing, dtype=complex), f_arr.shape).astype(complex)
    return (T[..., 0, 0] * z_back + T[..., 0, 1]) / (T[..., 1, 0] * z_back + T[..., 1, 1])


def reflection_coefficient(
    z_surface: np.ndarray, rho0: float = DEFAULT_RHO0, c0: float = DEFAULT_C0
) -> np.ndarray:
    """Plane-wave pressure reflection coefficient ``R = (Zs - Z0)/(Zs + Z0)``."""
    z0 = rho0 * c0
    zs = np.asarray(z_surface, dtype=complex)
    return (zs - z0) / (zs + z0)


def absorption_coefficient(R: np.ndarray) -> np.ndarray:
    """Absorption coefficient ``alpha = 1 - |R|^2`` (clipped at 0 for passivity)."""
    R = np.asarray(R, dtype=complex)
    return np.clip(1.0 - np.abs(R) ** 2, 0.0, None)


def insertion_loss(alpha: np.ndarray) -> np.ndarray:
    """Insertion loss ``IL = 10 log10(1/(1-alpha))`` in dB (energy form)."""
    a = np.clip(np.asarray(alpha, dtype=float), 0.0, 1.0 - 1e-15)
    return -10.0 * np.log10(1.0 - a)


def band_statistics(
    f: Sequence[float], alpha: np.ndarray, band: Tuple[float, float] = (100.0, 5000.0)
) -> Dict[str, float]:
    """Band-averaged metrics plus the peak location of ``alpha(f)``."""
    f = _as_float_array(f)
    a = np.asarray(alpha, dtype=float)
    mask = (f >= band[0]) & (f <= band[1])
    if not np.any(mask):
        raise ValueError("empty band")
    idx = int(np.argmax(a[mask]))
    return {
        "f_lo": float(band[0]),
        "f_hi": float(band[1]),
        "alpha_mean": float(np.trapezoid(a[mask], f[mask]) / (band[1] - band[0])),
        "alpha_min": float(a[mask].min()),
        "alpha_max": float(a[mask].max()),
        "f_peak": float(f[mask][idx]),
        "alpha_peak": float(a[mask][idx]),
    }


# ---------------------------------------------------------------------------
# 3. Engineering optimisation under a mass constraint
# ---------------------------------------------------------------------------
def optimize_lining(
    band: Tuple[float, float] = (250.0, 2000.0),
    *,
    d_max: float = 0.10,
    d_min: float = 0.02,
    sigma_range: Tuple[float, float] = (5000.0, 50000.0),
    mass_limit: Optional[float] = None,
    fibre_bulk_density: float = 40.0,
    n_freq: int = 61,
    n_d: int = 25,
    n_sigma: int = 21,
    rho0: float = DEFAULT_RHO0,
    c0: float = DEFAULT_C0,
) -> Dict[str, object]:
    """Grid search for the (thickness, flow resistivity) pair maximising the
    band-averaged absorption under an optional areal-mass limit.

    The areal mass is the fibre bulk density times thickness
    (``fibre_bulk_density * d``), i.e. a laminar-board weight proxy such as a
    shipyard would use for a bulkhead treatment.  Returns a report dict with
    the optimum, the full objective grid (for a Pareto plot) and the feasible
    thickness range.
    """
    f = np.linspace(band[0], band[1], n_freq)
    ds = np.linspace(d_min, d_max, n_d)
    sigmas = np.geomspace(sigma_range[0], sigma_range[1], n_sigma)
    objective = np.zeros((n_d, n_sigma))
    alpha_min = np.zeros((n_d, n_sigma))
    for i, d in enumerate(ds):
        for j, sig in enumerate(sigmas):
            lay = Lining([Layer(d, sig)])
            a = lay.absorption(f, rho0, c0)
            objective[i, j] = float(np.trapezoid(a, f) / (band[1] - band[0]))
            alpha_min[i, j] = float(a.min())
    masses = fibre_bulk_density * ds
    feasible = np.ones_like(ds, dtype=bool)
    if mass_limit is not None:
        feasible = masses <= mass_limit * (1.0 + 1e-12)
    obj_masked = np.where(feasible[:, None], objective, -np.inf)
    if not np.any(np.isfinite(obj_masked)):
        raise ValueError("mass constraint leaves no feasible thickness")
    i_best, j_best = np.unravel_index(int(np.argmax(obj_masked)), obj_masked.shape)
    # Thinnest design that still reaches 95 % of the best achievable mean alpha:
    # the engineering trade the owner actually cares about (weight vs damping).
    target = 0.95 * objective[i_best, j_best]
    ok = feasible[:, None] & (objective >= target)
    thin = np.where(ok)[0].min() if np.any(ok) else i_best
    thin_sigmas = np.where(ok[thin])[0]
    j_thin = int(thin_sigmas[np.argmax(objective[thin, thin_sigmas])]) if thin_sigmas.size else j_best
    return {
        "band": [float(band[0]), float(band[1])],
        "d_grid": ds.tolist(),
        "sigma_grid": sigmas.tolist(),
        "alpha_mean_grid": objective.tolist(),
        "alpha_min_grid": alpha_min.tolist(),
        "areal_mass_grid": masses.tolist(),
        "mass_limit": None if mass_limit is None else float(mass_limit),
        "best": {
            "d": float(ds[i_best]),
            "sigma": float(sigmas[j_best]),
            "alpha_mean": float(objective[i_best, j_best]),
            "alpha_min": float(alpha_min[i_best, j_best]),
            "areal_mass": float(masses[i_best]),
            "il_mean_dB": float(insertion_loss(objective[i_best, j_best])),
        },
        "thinnest_near_optimal": {
            "d": float(ds[thin]),
            "sigma": float(sigmas[j_thin]),
            "alpha_mean": float(objective[thin, j_thin]),
            "alpha_min": float(alpha_min[thin, j_thin]),
            "areal_mass": float(masses[thin]),
            "fraction_of_best": float(objective[thin, j_thin] / objective[i_best, j_best]),
        },
        "fibre_bulk_density": float(fibre_bulk_density),
    }


# ---------------------------------------------------------------------------
# 4. 1-D (D1Q2) acoustic duct with a locally reacting impedance boundary
# ---------------------------------------------------------------------------
#: D1Q2 lattice: two populations, weights 1/2, lattice speed 1, ``c_s = 1``.
#: Density perturbation *is* pressure (``p' = rho - rho0``) and the
#: characteristic impedance is ``rho0 c_s = 1`` in lattice units, so the
#: normalised impedance ``Z/(rho0 c0)`` maps one-to-one onto the lattice.
_W2 = 0.5


@dataclass
class DuctLBM:
    """One-dimensional D1Q2 linear-acoustic duct on a uniform lattice.

    Physical mapping for a duct of length ``length`` discretised by ``n``
    nodes: ``dx = length/(n-1)``, ``dt = dx/c0`` (the lattice speed is one
    node per step), and ``Z/(rho0 c0)`` equals the lattice impedance.

    The D1Q2 model is used because it has no rest/entropy population: the two
    characteristic variables are exactly the two populations, so a locally
    reacting impedance condition is a one-line relation with no spurious
    thermal mode (a D1Q3 variant leaks energy into the rest population and
    diverges -- see the notes in the module report).
    """

    n: int
    length: float
    c0: float = DEFAULT_C0
    rho0: float = DEFAULT_RHO0
    tau: float = 0.55
    dx: float = field(init=False)
    dt: float = field(init=False)
    f: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n < 32:
            raise ValueError("duct needs at least 32 nodes")
        self.dx = self.length / (self.n - 1)
        self.dt = self.dx / self.c0
        self.f = np.full((self.n, 2), _W2)
        if self.tau <= 0.5:
            raise ValueError("tau must exceed 0.5")

    # -- primary variables ------------------------------------------------
    def rho(self) -> np.ndarray:
        return self.f.sum(axis=1)

    def velocity(self) -> np.ndarray:
        return (self.f[:, 0] - self.f[:, 1]) / self.rho()

    def pressure(self) -> np.ndarray:
        """Pressure perturbation ``p'/(rho0 c_s^2) = rho - rho0`` (lattice)."""
        return self.f.sum(axis=1) - 1.0

    def add_soft_source(self, node: int, amplitude: float) -> None:
        """Add a density perturbation (soft source; negligible reflection)."""
        self.f[node, :] += 0.5 * amplitude

    def step(self, boundary: "ImpedanceWall", source_node: int = -1,
             source_amplitude: float = 0.0) -> None:
        # collide (BGK)
        rho = self.f.sum(axis=1)
        u = (self.f[:, 0] - self.f[:, 1]) / rho
        feq = np.stack([0.5 * rho * (1.0 + u), 0.5 * rho * (1.0 - u)], axis=1)
        self.f += (feq - self.f) / self.tau
        # stream
        f_new = np.empty_like(self.f)
        f_new[1:, 0] = self.f[:-1, 0]
        f_new[:-1, 1] = self.f[1:, 1]
        self.f = f_new
        # boundaries
        self.f[0, 0] = _W2                        # matched inlet, a = 0
        boundary.apply(self, node=self.n - 1)
        if source_node >= 0 and source_amplitude != 0.0:
            self.f[source_node, :] += 0.5 * source_amplitude


class ImpedanceWall:
    """Locally reacting impedance boundary at the last duct node.

    With the two D1Q2 populations split into fluctuations
    ``a = f+ - 1/2`` (incoming to the wall) and ``c = f- - 1/2`` (reflected),
    the reflection coefficient is simply ``c = R a`` for a harmonic signal:
    a single-frequency complex ``R`` becomes an amplitude gain ``|R|`` plus a
    delay ``tau = (arg R + 2 pi)/omega`` (a full-period shift is invisible in
    steady state and keeps ``tau >= T/2`` so the update is explicit).  The
    fractional delay is realised by third-order Lagrange interpolation on the
    stored incident history.

    Because ``|R| <= 1`` the map is non-amplifying, which is what makes it
    stable for a passive lining (a rational broadband realisation would need a
    causal fit of the Delany-Bazley surface impedance; that fit is not used
    here, see the honest-limitations section of the report).
    """

    def __init__(self, R: complex, freq: float, dt: float, order: int = 3):
        if not np.isfinite(abs(R)):
            raise ValueError("R must be finite")
        if freq <= 0.0:
            raise ValueError("freq must be positive")
        if abs(R) > 1.0 + 1e-9:
            raise ValueError("|R| > 1 is not a passive locally reacting wall")
        self.R = complex(R)
        self.freq = float(freq)
        self.dt = float(dt)
        self.order = int(order)
        omega = 2.0 * math.pi * self.freq
        period = 1.0 / (self.freq * self.dt)
        # a delay by tau multiplies a phasor by exp(+i omega tau) under the
        # exp(-i omega t) convention, so omega tau = arg R (mod 2 pi)
        tau = (np.angle(self.R) + 2.0 * math.pi) / omega / self.dt
        if tau < 2.0:
            tau += period
        self.tau_samples = float(tau)
        self.m = int(math.floor(tau))
        self.frac = float(tau - self.m)
        self._hist: List[float] = [0.0] * (self.m + self.order + 2)
        self.wall_pressure: List[float] = []
        self.wall_velocity: List[float] = []

    def reset(self) -> None:
        self._hist = [0.0] * len(self._hist)
        self.wall_pressure = []
        self.wall_velocity = []

    def _delayed(self) -> float:
        """Incident fluctuation delayed by ``m + frac`` samples (Lagrange)."""
        idx = self.m
        t = self.frac
        h = self._hist
        if self.order == 1:
            return (1.0 - t) * h[idx] + t * h[idx + 1]
        if self.order == 2:
            return (
                (t - 1.0) * (t - 2.0) / 2.0 * h[idx]
                + (-t) * (t - 2.0) * h[idx + 1]
                + t * (t - 1.0) / 2.0 * h[idx + 2]
            )
        return (
            -(t - 1.0) * (t - 2.0) * (t - 3.0) / 6.0 * h[idx]
            + t * (t - 2.0) * (t - 3.0) / 2.0 * h[idx + 1]
            - t * (t - 1.0) * (t - 3.0) / 2.0 * h[idx + 2]
            + t * (t - 1.0) * (t - 2.0) / 6.0 * h[idx + 3]
        )

    def apply(self, duct: DuctLBM, node: int) -> None:
        a = duct.f[node, 0] - _W2
        c = abs(self.R) * self._delayed()
        duct.f[node, 1] = _W2 + c
        self._hist = [a] + self._hist[:-1]
        self.wall_pressure.append(duct.f[node, :].sum() - 1.0)
        self.wall_velocity.append((duct.f[node, 0] - duct.f[node, 1]) / duct.f[node, :].sum())


class RigidWall:
    """Rigid (``v_n = 0``) end: bounce-back of the normal population."""

    def apply(self, duct: DuctLBM, node: int) -> None:
        duct.f[node, 1] = duct.f[node, 0]


# ---------------------------------------------------------------------------
# 5. Harmonic (steady-state) reflection measurement
# ---------------------------------------------------------------------------
def _complex_amplitude(signal: np.ndarray, freq: float, dt: float) -> complex:
    """Complex phasor ``X`` with ``x(t) = Re[X exp(-i omega t)]``."""
    n = signal.size
    t = np.arange(n) * dt
    return 2.0 / n * np.sum(signal * np.exp(1j * 2.0 * math.pi * freq * t))


def _single_frequency_run(
    duct: DuctLBM,
    freq: float,
    boundary: "ImpedanceWall | RigidWall",
    probes: Sequence[int],
    source_node: int,
    amplitude: float = 1e-4,
    settle_periods: float = 12.0,
    measure_periods: float = 6.0,
) -> Dict[str, object]:
    """Drive one harmonic source and return the recorded steady-state window."""
    period = 1.0 / (freq * duct.dt)
    # the record must start after (i) the wave has travelled from the source to
    # the wall, (ii) the boundary delay line has filled, (iii) transients of the
    # duct/lining resonance have decayed -> measured in periods
    travel = float(duct.n - 1 - source_node)     # lattice speed = 1 node/step
    # a *fixed* margin of two periods (not the boundary's own delay) keeps the
    # total record length identical for every run at this frequency, so the
    # calibration and the lined record are time-aligned sample by sample
    n_settle = int(round(settle_periods * period + travel + 2.0 * period))
    n_measure = int(round(measure_periods * period))
    if isinstance(boundary, ImpedanceWall):
        boundary.reset()
    total = n_settle + n_measure
    probes = list(probes)
    rec = np.zeros((n_measure, len(probes)))
    wall_p: List[float] = []
    wall_u: List[float] = []
    for step in range(total):
        if step >= n_settle:
            rec[step - n_settle] = duct.pressure()[probes]
            if isinstance(boundary, ImpedanceWall):
                wall_p.append(duct.f[-1, :].sum() - 1.0)
                wall_u.append((duct.f[-1, 0] - duct.f[-1, 1]) / duct.f[-1, :].sum())
        duct.add_soft_source(source_node, amplitude * math.sin(2.0 * math.pi * freq * step * duct.dt))
        duct.step(boundary)
    return {
        "signals": rec,
        "wall_pressure": np.asarray(wall_p),
        "wall_velocity": np.asarray(wall_u),
        "n_settle": n_settle,
        "n_measure": n_measure,
        "probes": probes,
    }


def measure_reflection(
    lining: Optional[Lining],
    freq: float,
    *,
    n: int = 420,
    length: float = 0.9,
    probes: Optional[Sequence[int]] = None,
    source_node: int = 40,
    wall_offset: float = 0.0,
    settle_periods: float = 12.0,
    measure_periods: float = 6.0,
    amplitude: float = 1e-4,
    tau: float = 0.55,
    c0: float = DEFAULT_C0,
    rho0: float = DEFAULT_RHO0,
) -> Dict[str, object]:
    """Measure the wall reflectance of a lining at one frequency with the LBM.

    Two runs (anechoic calibration + lined) share lattice, source and inlet.
    The calibration gives the numerical wavenumber ``k`` of the lattice at this
    frequency (so dispersion is not assumed); the lined run gives the total
    field.  A two-probe decomposition then yields ``R`` at the wall plane.

    Returns a dict with the complex reflectance, the TMM prediction, the
    measured/expected absorption, the wall energy budget and the raw data.
    """
    duct = DuctLBM(n=n, length=length, c0=c0, rho0=rho0, tau=tau)
    if probes is None:
        # two probes 12 / 20 nodes away from the wall: close enough that the
        # de-rotation to the wall plane stays small, far enough that lattice
        # artefacts at the boundary node do not reach them
        probes = (n - 12, n - 20)
    probes = tuple(int(p) for p in probes)
    calib = _single_frequency_run(
        DuctLBM(n=n, length=length, c0=c0, rho0=rho0, tau=tau), freq,
        ImpedanceWall(0.0 + 0j, freq, duct.dt), list(probes), source_node,
        amplitude, settle_periods, measure_periods,
    )
    x0, x1 = probes
    p0c = np.array([_complex_amplitude(calib["signals"][:, i], freq, duct.dt) for i in range(2)])
    # p ~ exp(i k x) with x in nodes  =>  k = -i ln(p0/p1) / (x0 - x1);
    # the complex k also carries the lattice's own attenuation (Im k < 0 in
    # this convention means decay along +x).
    k = -1j * np.log(p0c[0] / p0c[1]) / (x0 - x1)

    if lining is None:
        R_target = 0.0 + 0j
        z_surf = None
    elif isinstance(lining, (int, float, complex, np.complexfloating, np.floating)):
        # direct reflectance (used by the unit tests and negative controls)
        R_target = complex(lining)
        z_surf = None
    else:
        z_surf = complex(lining.surface_impedance(freq, rho0, c0))
        R_target = complex(reflection_coefficient(z_surf, rho0, c0))
    lined = _single_frequency_run(
        duct, freq, ImpedanceWall(R_target, freq, duct.dt), list(probes), source_node,
        amplitude, settle_periods, measure_periods,
    )
    p0l = np.array([_complex_amplitude(lined["signals"][:, i], freq, duct.dt) for i in range(2)])
    # Superposition: the lined and the calibration run share source, inlet and
    # lattice, so the reflected field is exactly their difference at a probe.
    # This is far better conditioned than a two-probe 2x2 solve when the probe
    # spacing is a small fraction of a wavelength (low frequency end).
    a_ref0 = p0l[0] - p0c[0]
    a_inc0 = p0c[0]
    wall = float(n - 1) + float(wall_offset)
    # phase de-rotation to the wall plane; the magnitude is unaffected by k
    R_meas = a_ref0 / a_inc0 * np.exp(-2j * k * (wall - x0))
    alpha_meas = 1.0 - abs(R_meas) ** 2
    # incident amplitude at the wall plane (for the energy budget)
    pi_wall = a_inc0 * np.exp(1j * k * (wall - x0))

    # --- energy budget at the wall node (independent of the decomposition) ---
    p_w = lined["wall_pressure"]
    u_w = lined["wall_velocity"]
    P_w = _complex_amplitude(p_w, freq, duct.dt)
    U_w = _complex_amplitude(u_w, freq, duct.dt)
    intensity = 0.5 * (P_w * np.conj(U_w)).real      # +x = into the wall
    # incident intensity at the wall plane from the decomposition
    inc_intensity = 0.5 * abs(pi_wall) ** 2  # Z0 = 1 in D1Q2 lattice units
    alpha_energy = intensity / inc_intensity if inc_intensity > 0 else np.nan

    return {
        "freq": float(freq),
        "R_lbm": complex(R_meas),
        "R_tmm": complex(R_target),
        "alpha_lbm": float(alpha_meas),
        "alpha_tmm": float(1.0 - abs(R_target) ** 2),
        "k_num": complex(k),
        "alpha_energy": float(alpha_energy),
        "intensity": float(intensity),
        "incident_intensity": float(inc_intensity),
        "duct": duct,
        "calibration": calib,
        "lined": lined,
        "wall_offset": float(wall_offset),
        "z_surface_tmm": None if z_surf is None else complex(z_surf),
    }


def sweep_absorption(
    lining: Optional[Lining],
    freqs: Sequence[float],
    *,
    wall_offset: float = 0.0,
    **kwargs,
) -> Dict[str, np.ndarray]:
    """Frequency sweep of :func:`measure_reflection` (steady-state LBM).

    Returns arrays of the LBM and TMM reflectance / absorption at each
    frequency, plus the wall energy budget.
    """
    Rs, Rt, alphas, alphat, aenergy = [], [], [], [], []
    for f in np.asarray(freqs, dtype=float):
        res = measure_reflection(lining, float(f), wall_offset=wall_offset, **kwargs)
        Rs.append(res["R_lbm"])
        Rt.append(res["R_tmm"])
        alphas.append(res["alpha_lbm"])
        alphat.append(res["alpha_tmm"])
        aenergy.append(res["alpha_energy"])
    return {
        "freqs": np.asarray(freqs, dtype=float),
        "R_lbm": np.asarray(Rs, dtype=complex),
        "R_tmm": np.asarray(Rt, dtype=complex),
        "alpha_lbm": np.asarray(alphas),
        "alpha_tmm": np.asarray(alphat),
        "alpha_energy": np.asarray(aenergy),
    }
