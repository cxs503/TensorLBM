"""Spectral toolkit for free-decaying isotropic turbulence (DIT) + CBC1971 reference.

This module supplies the analysis side of the CBC DIT (grid-turbulence) LES
validation surface:

* the digitised **Comte-Bellot & Corrsin (1971)** longitudinal spectra
  :math:`E_{11}(k_1)` at the three measuring stations
  :math:`tU_0/M = 42, 98, 172` (cgs units as in the source table, see
  :data:`CBC1971_DATA`);
* derived reference quantities of that experiment (integral of the measured
  spectrum, decay exponent, dissipation, Kolmogorov scale, Taylor microscale);
* the von Karman model spectrum
  :math:`E(k) = C k^4 / (k^2+k_0^2)^{17/6}` and its forward projection to the
  one-dimensional longitudinal spectrum
  :math:`E_{11}(k_1)=\\int_{k_1}^\\infty (E(k)/k)(1-k_1^2/k^2)\\,dk`
  (Pope 2000, eq. 6.118), plus a least-squares fit of :math:`(C,k_0)` to the
  measured :math:`E_{11}`;
* a divergence-free random-field generator whose 3-D spectrum follows the von
  Karman model (lattice initial condition);
* spectra estimators for the LBM velocity field (3-D shell spectrum
  :math:`E(k)` and one-dimensional longitudinal spectrum :math:`E_{11}(k_1)`,
  both normalised so that :math:`\\int E_{11}dk_1 = u_1'^2` exactly) and the
  fitting / pile-up diagnostics used by the acceptance gates.

Reference
---------
Comte-Bellot, G. & Corrsin, S. (1971) *Simple Eulerian time correlation of
full- and narrow-band velocity signals in grid-generated "isotropic"
turbulence*, J. Fluid Mech. **48**(2), 273-337.

The tabulated values shipped here are the widely circulated digitisation of
the three measured spectra (provenance and checksums in
:data:`CBC1971_DATA["provenance"]`); the source table carries three
significant digits, which is the honest precision of the comparison.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

__all__ = [
    "CBC1971_DATA",
    "CBC_PHYSICAL",
    "cbc_spectra",
    "cbc_station_moments",
    "cbc_derived",
    "von_karman_E3d",
    "von_karman_E11",
    "fit_von_karman",
    "make_von_karman_ic",
    "shell_spectrum_3d",
    "longitudinal_spectrum",
    "fit_loglog_slope",
    "fit_decay_exponent",
    "pileup_indicator",
    "spectrum_integral_error",
    "spectrum_energy",
    "sig6",
]

_DATA_PATH = Path(__file__).with_name("data") / "cbc1971_dit_spectra.json"

with _DATA_PATH.open("r", encoding="utf-8") as _fh:
    CBC1971_DATA: dict = json.load(_fh)

#: Physical conditions of the CBC1971 grid-turbulence experiment (SI).
#: ``M`` is the grid mesh length, ``U0`` the free-stream velocity, ``nu`` the
#: kinematic viscosity of air at the reported temperature.
CBC_PHYSICAL: dict = {
    "M": 0.0508,  # m, square-bar grid mesh
    "U0": 10.0,  # m/s, free-stream
    "nu": 1.5e-5,  # m^2/s, air
    "stations_tU0_over_M": (42.0, 98.0, 172.0),
}


# ---------------------------------------------------------------------------
# CBC1971 reference spectra
# ---------------------------------------------------------------------------
def cbc_spectra(station: float, si: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Measured longitudinal spectrum at ``station`` (= t U0 / M).

    Parameters
    ----------
    station : 42, 98 or 172 (the third station is labelled 171 in the source).
    si : if True return wavenumber in 1/m and ``E11`` in m^3/s^2, else the raw
        cgs table (k1 in 1/cm, E11 in cm^3/s^2).

    Returns
    -------
    (k1, E11) arrays, ``k1`` increasing.  With ``si=True`` the pair satisfies
    ``int E11 dk1 = u1'^2`` in m^2/s^2.
    """
    cols = CBC1971_DATA["columns"]
    key = {42.0: "42", 98.0: "98", 172.0: "172", 171.0: "172"}[float(station)]
    k1 = np.asarray(cols[f"k_{key}"], dtype=np.float64)
    e11 = np.asarray(cols[f"E11_{key}"], dtype=np.float64)
    if si:
        # 1/cm -> 1/m ; cm^3/s^2 -> m^3/s^2 (dimension of E11 is u^2/k)
        return k1 * 100.0, e11 * 1e-6
    return k1, e11


def _power_law_tail(k: np.ndarray, e: np.ndarray, npts: int = 4) -> tuple[float, float]:
    """log-log slope/intercept fitted on the last ``npts`` points."""
    slope, intercept = np.polyfit(np.log(k[-npts:]), np.log(e[-npts:]), 1)
    return float(slope), float(intercept)


def cbc_station_moments(station: float) -> dict:
    """Integrals of the measured spectrum: ``u1'^2`` and ``int k^2 E11 dk``.

    Low-k tail: ``E11 ~ a k^4`` fitted on the first point and integrated to
    zero.  High-k tail: power law fitted on the last four points and integrated
    to infinity (the exponent is steeper than -5, so the tail is small).  The
    energy fraction carried by the two tails is reported as an honest
    uncertainty indicator of the digitisation.
    """
    k, e = cbc_spectra(station)  # SI
    m_hi, _ = _power_law_tail(k, e)
    if m_hi >= -1.0:  # pragma: no cover - defensive
        m_hi = -1.0000001
    tail_hi = e[-1] * k[-1] / (-m_hi - 1.0)
    a_lo = e[0] / k[0] ** 4
    tail_lo = a_lo * k[0] ** 5 / 5.0
    core = float(np.trapezoid(e, k))
    u1sq = core + tail_hi + tail_lo
    # dissipation integral int k^2 E11 dk with the same tails
    k2e = k**2 * e
    core_d = float(np.trapezoid(k2e, k))
    if m_hi < -3.0:
        tail_hi_d = e[-1] * k[-1] ** 3 / (-m_hi - 3.0)
    else:  # pragma: no cover - defensive (measured slopes are ~ -5)
        tail_hi_d = k2e[-1] * k[-1] / 2.0
    tail_lo_d = a_lo * k[0] ** 7 / 7.0
    return {
        "station": float(station),
        "k_min": float(k[0]),
        "k_max": float(k[-1]),
        "u1_sq": float(u1sq),
        "u1_rms": float(np.sqrt(u1sq)),
        "u1_sq_core": core,
        "tail_lo_frac": float(tail_lo / u1sq),
        "tail_hi_frac": float(tail_hi / u1sq),
        "high_k_slope": float(m_hi),
        "int_k2E11": float(core_d + tail_hi_d + tail_lo_d),
        "n_points": int(k.size),
    }


def cbc_derived() -> dict:
    """Experiment-level derived reference quantities (SI units).

    * ``u1_rms`` per station from ``int E11 dk1``;
    * decay exponent ``n`` of ``u1'^2 ~ (t U0/M)^-n`` fitted on the three
      stations (the acceptance band 1.15-1.30 of the mission refers to this
      quantity);
    * dissipation from the dissipation spectrum, ``eps_spec = 15 nu int k^2 E11
      dk`` (isotropic relation, Pope 2000 eq. 6.166) -- *primary*;
    * dissipation from the measured decay, ``eps_decay = -d(1.5 u'^2)/dt`` --
      reported as a cross-check (for grid turbulence it is an upper bound
      because part of the streamwise change is transport, not decay);
    * ``eta``, ``lambda_taylor``, ``Re_lambda`` from each of the two.
    """
    phys = CBC_PHYSICAL
    M, U0, nu = phys["M"], phys["U0"], phys["nu"]
    st = np.asarray(phys["stations_tU0_over_M"], dtype=np.float64)
    mom = [cbc_station_moments(s) for s in st]
    u1sq = np.asarray([m["u1_sq"] for m in mom])
    t = st * M / U0  # physical time in the frame moving with U0

    a, b = np.polyfit(np.log(t), np.log(u1sq), 1)
    n_decay = float(-a)
    resid = np.log(u1sq) - (b + a * np.log(t))
    n_12 = float(-np.log(u1sq[1] / u1sq[0]) / np.log(t[1] / t[0]))
    n_23 = float(-np.log(u1sq[2] / u1sq[1]) / np.log(t[2] / t[1]))

    per_station = []
    for m, tt, uu in zip(mom, t, u1sq):
        eps_spec = 15.0 * nu * m["int_k2E11"]
        eps_decay = 1.5 * n_decay * uu / tt
        lam_spec = np.sqrt(15.0 * nu * uu / eps_spec)
        lam_dec = np.sqrt(15.0 * nu * uu / eps_decay)
        per_station.append(
            {
                "station": m["station"],
                "u1_rms": float(np.sqrt(uu)),
                "eps_spec": float(eps_spec),
                "eps_decay": float(eps_decay),
                "eta_spec": float((nu**3 / eps_spec) ** 0.25),
                "eta_decay": float((nu**3 / eps_decay) ** 0.25),
                "lambda_spec": float(lam_spec),
                "lambda_decay": float(lam_dec),
                "Re_lambda_spec": float(np.sqrt(uu) * lam_spec / nu),
                "Re_lambda_decay": float(np.sqrt(uu) * lam_dec / nu),
                "tail_hi_frac": m["tail_hi_frac"],
                "tail_lo_frac": m["tail_lo_frac"],
            }
        )
    return {
        "stations_tU0_over_M": st.tolist(),
        "t_seconds": t.tolist(),
        "u1_sq": u1sq.tolist(),
        "u1_rms": np.sqrt(u1sq).tolist(),
        "n_decay_3station": n_decay,
        "n_decay_station12": n_12,
        "n_decay_station23": n_23,
        "decay_fit_log_rms_resid": float(np.sqrt(np.mean(resid**2))),
        "per_station": per_station,
        "note": (
            "eps_spec (dissipation spectrum, tail extrapolated) is the primary "
            "scale estimate; eta_decay from -d(1.5u'^2)/dt is an upper-bound "
            "cross-check (larger eta => wider k*eta window). Both are "
            "propagated to the acceptance window and reported."
        ),
    }


# ---------------------------------------------------------------------------
# von Karman model spectrum
# ---------------------------------------------------------------------------
def von_karman_E3d(k: np.ndarray, C: float, k0: float) -> np.ndarray:
    """von Karman 3-D energy spectrum ``E(k) = C k^4 / (k^2+k0^2)^(17/6)``."""
    k = np.asarray(k, dtype=np.float64)
    out = np.zeros_like(k)
    pos = k > 0
    out[pos] = C * k[pos] ** 4 / (k[pos] ** 2 + k0**2) ** (17.0 / 6.0)
    return out


def von_karman_E11(k1: np.ndarray, C: float, k0: float, kmax: float | None = None) -> np.ndarray:
    """Forward projection of the 3-D model spectrum to ``E11(k1)`` (Pope 6.118)."""
    k1 = np.atleast_1d(np.asarray(k1, dtype=np.float64))
    if kmax is None:
        kmax = max(2000.0 * k0, 50.0 * float(k1.max()))
    out = np.zeros_like(k1)
    for i, kk in enumerate(k1):
        lo = max(kk * 1.0000001, 1e-12)
        k = np.geomspace(lo, kmax, 3000)
        f = von_karman_E3d(k, C, k0) / k * np.maximum(1.0 - (kk / k) ** 2, 0.0)
        out[i] = np.trapezoid(f, k)
    return out


def fit_von_karman(k1_data: np.ndarray, e11_data: np.ndarray) -> dict:
    """Least-squares fit of ``(C, k0)`` to measured ``E11`` in log space."""
    from scipy.optimize import least_squares

    k1_data = np.asarray(k1_data, dtype=np.float64)
    e11_data = np.asarray(e11_data, dtype=np.float64)
    log_e = np.log(e11_data)

    def resid(p):
        model = np.maximum(von_karman_E11(k1_data, np.exp(p[0]), np.exp(p[1])), 1e-300)
        return np.log(model) - log_e

    sol = least_squares(resid, x0=[np.log(1e-6), np.log(20.0)], bounds=(-80.0, 40.0))
    r = resid(sol.x)
    return {
        "C": float(np.exp(sol.x[0])),
        "k0": float(np.exp(sol.x[1])),
        "log_rms_error": float(np.sqrt(np.mean(r**2))),
        "log_max_abs_error": float(np.max(np.abs(r))),
        "max_rel_error": float(np.max(np.abs(np.exp(r) - 1.0))),
        "n_points": int(k1_data.size),
        "cost": float(sol.cost),
    }


def spectrum_energy(C: float, k0: float) -> float:
    """``int_0^inf E(k) dk`` for the von Karman model (numerical, dense grid)."""
    k = np.geomspace(k0 * 1e-4, k0 * 1e4, 4000)
    return float(np.trapezoid(von_karman_E3d(k, C, k0), k))


# ---------------------------------------------------------------------------
# spectral initial condition (solenoidal random field)
# ---------------------------------------------------------------------------
def make_von_karman_ic(
    n: int,
    dx: float,
    k0: float,
    u_rms: float,
    seed: int = 20261108,
    dtype=np.float64,
) -> tuple[np.ndarray, dict]:
    """Divergence-free random velocity field with a von Karman 3-D spectrum.

    The 3-D energy spectrum of the generated field follows
    ``C k^4/(k^2+k0^2)^(17/6)`` (``k0`` from the CBC fit); the amplitude is
    then rescaled so that the *per-component* rms velocity equals ``u_rms``
    exactly (so that ``int E11 dk1 = u_rms^2``, the CBC convention).

    Random phases are taken from a real white-noise field, so the Hermitian
    symmetry of the spectrum is exact *before* the solenoidal projection -- this
    is what keeps the generated field divergence free to round-off.
    """
    rng = np.random.default_rng(seed)
    kn = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    KZ, KY, KX = np.meshgrid(kn, kn, kn, indexing="ij")
    K2 = KX**2 + KY**2 + KZ**2
    K = np.sqrt(K2)
    noise = rng.standard_normal((3, n, n, n))
    uh = np.fft.fftn(noise, axes=(1, 2, 3))
    with np.errstate(divide="ignore", invalid="ignore"):
        # isotropic per-mode amplitude: <|u_i(k)|^2> = E(K)/(4 pi K^2) (shape only)
        amp = np.sqrt(np.where(K > 0, von_karman_E3d(K, 1.0, k0) / (4.0 * np.pi * K2), 0.0))
    uh *= amp
    kdotu = KX * uh[0] + KY * uh[1] + KZ * uh[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        fac = np.where(K2 > 0, kdotu / K2, 0.0)
    for i, Kc in enumerate((KX, KY, KZ)):
        uh[i] -= Kc * fac
    u = np.fft.ifftn(uh, axes=(1, 2, 3)).real
    u -= u.mean(axis=(1, 2, 3), keepdims=True)
    rms_comp = float(np.sqrt((u**2).mean()))
    u *= u_rms / rms_comp
    urms_now = float(np.sqrt((u**2).mean()))
    uhat = np.fft.fftn(u, axes=(1, 2, 3))
    div_hat = KX * uhat[0] + KY * uhat[1] + KZ * uhat[2]
    div_rms = float(np.sqrt((np.abs(div_hat) ** 2).mean()))
    uhat_rms = float(np.sqrt((np.abs(uhat) ** 2).mean()))
    # energy spectrum of the generated field: int E dk should be 1.5 u_rms^2
    k3, e3 = shell_spectrum_3d(u, dx)
    info = {
        "n": int(n),
        "dx": float(dx),
        "k0": float(k0),
        "u_rms_target": float(u_rms),
        "u_rms_achieved": urms_now,
        "u_rms_vector": float(np.sqrt(3.0) * urms_now),
        "seed": int(seed),
        "div_hat_over_uhat": div_rms / uhat_rms,
        "k_min": float(2.0 * np.pi / (n * dx)),
        "k_max": float(np.pi / dx),
        "tke_from_spectrum": float(np.trapezoid(e3, k3)),
        "tke_expected": float(1.5 * urms_now**2),
        "mean_removed": True,
        "dtype": np.dtype(dtype).name,
    }
    return u.astype(dtype), info


# ---------------------------------------------------------------------------
# spectra estimators
# ---------------------------------------------------------------------------
def shell_spectrum_3d(u: np.ndarray, dx: float) -> tuple[np.ndarray, np.ndarray]:
    """3-D shell-averaged spectrum density ``E(k)`` of a velocity field.

    ``E(k)`` is normalised so that ``int E dk = 0.5 <u_i u_i>``.
    """
    from scipy import fft as sfft

    u = np.asarray(u, dtype=np.float32)
    n = u.shape[1]
    uhat = sfft.fftn(u, axes=(1, 2, 3), workers=-1, norm="forward")
    k1 = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    KZ, KY, KX = np.meshgrid(k1, k1, k1, indexing="ij")
    K = np.sqrt(KX**2 + KY**2 + KZ**2)
    energy = 0.5 * (
        np.abs(uhat[0]) ** 2 + np.abs(uhat[1]) ** 2 + np.abs(uhat[2]) ** 2
    )
    return _bin_isotropic(K, energy, n, dx, mode="radial")


def _bin_isotropic(K: np.ndarray, energy: np.ndarray, n: int, dx: float, mode: str):
    """Bin ``energy`` (per-mode energy) into wavenumber shells; return density."""
    dk = 2.0 * np.pi / (n * dx)
    ib = np.rint(K / dk).astype(np.int64)
    # |k| reaches sqrt(3)*pi/dx on the cube corners, so the bin count exceeds n/2
    nb = int(ib.max()) + 1
    sums = np.bincount(ib.ravel(), weights=energy.ravel().astype(np.float64), minlength=nb)
    counts = np.bincount(ib.ravel(), minlength=nb).astype(np.float64)
    # shell width: number of modes with |k| in [ (j-0.5)dk, (j+0.5)dk ] spans dk
    kmax = np.pi / dx
    kk = np.arange(nb) * dk
    counts[0] = max(counts[0], 1.0)
    dens = sums / dk
    ok = (kk > 0) & (kk <= kmax) & (counts > 0)
    return kk[ok], dens[ok]


def longitudinal_spectrum(u: np.ndarray, dx: float) -> tuple[np.ndarray, np.ndarray]:
    """1-D longitudinal spectrum ``E11(k1)`` along axis 0 (x direction).

    ``E11`` is normalised so that ``int_0^inf E11 dk1 = u_1'^2``.
    """
    from scipy import fft as sfft

    u = np.asarray(u, dtype=np.float32)
    n = u.shape[1]
    uhat = sfft.fftn(u, axes=(1, 2, 3), workers=-1, norm="forward")
    k1 = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    dk = 2.0 * np.pi / (n * dx)
    # mode axis (0) is the x axis of the array (u[s, i, j, k] -> kx along i)
    energy = 0.5 * np.abs(uhat[0]) ** 2
    kx = k1
    ib = np.rint(np.abs(kx) / dk).astype(np.int64)
    nb = int(ib.max()) + 1
    profile = energy.sum(axis=(1, 2))  # sum over ky, kz for each kx mode
    sums = np.bincount(ib, weights=profile, minlength=nb)
    kmax = np.pi / dx
    kk = np.arange(nb) * dk
    dens = sums / dk
    ok = (kk > 0) & (kk <= kmax)
    return kk[ok], dens[ok]


def fit_loglog_slope(
    k: np.ndarray, e: np.ndarray, k_lo: float, k_hi: float
) -> dict:
    """Power-law fit ``E = A k^m`` on ``k_lo <= k <= k_hi`` (log-log LSQ)."""
    k = np.asarray(k, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    sel = (k >= k_lo) & (k <= k_hi) & (e > 0)
    npts = int(sel.sum())
    if npts < 3:
        return {
            "slope": float("nan"),
            "slope_se": float("nan"),
            "r2": float("nan"),
            "n_points": npts,
            "k_lo": float(k_lo),
            "k_hi": float(k_hi),
        }
    lx = np.log(k[sel])
    ly = np.log(e[sel])
    a, b = np.polyfit(lx, ly, 1)
    res = ly - (b + a * lx)
    dof = max(npts - 2, 1)
    s2 = float(res @ res) / dof
    sxx = float(((lx - lx.mean()) ** 2).sum())
    se = float(np.sqrt(s2 / sxx)) if sxx > 0 else float("nan")
    ss_tot = float(((ly - ly.mean()) ** 2).sum())
    r2 = 1.0 - float(res @ res) / ss_tot if ss_tot > 0 else float("nan")
    return {
        "slope": float(a),
        "intercept": float(b),
        "slope_se": se,
        "r2": float(r2),
        "n_points": npts,
        "k_lo": float(k_lo),
        "k_hi": float(k_hi),
    }


def fit_decay_exponent(t: np.ndarray, e: np.ndarray, t_lo: float | None = None, t_hi: float | None = None) -> dict:
    """Decay exponent ``E(t) = A (t+t0)^-n`` via log-log LSQ (+ virtual origin)."""
    t = np.asarray(t, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    sel = np.ones_like(t, dtype=bool)
    if t_lo is not None:
        sel &= t >= t_lo
    if t_hi is not None:
        sel &= t <= t_hi
    tt, ee = t[sel], e[sel]
    if tt.size < 4:
        return {"n": float("nan"), "r2": float("nan"), "n_points": int(tt.size)}
    lx, ly = np.log(tt), np.log(ee)
    a, b = np.polyfit(lx, ly, 1)
    res = ly - (b + a * lx)
    ss_tot = float(((ly - ly.mean()) ** 2).sum())
    r2 = 1.0 - float(res @ res) / ss_tot if ss_tot > 0 else float("nan")
    # virtual origin: maximise R^2 over t0
    best = (-np.inf, float("nan"), 0.0)
    for t0 in np.linspace(-0.9 * float(tt[0]) + 1e-9, 3.0 * float(tt[-1]), 601):
        lx0 = np.log(tt + t0)
        a0, b0 = np.polyfit(lx0, ly, 1)
        rr = ly - (b0 + a0 * lx0)
        sst = float(((ly - ly.mean()) ** 2).sum())
        r20 = 1.0 - float(rr @ rr) / sst if sst > 0 else -np.inf
        if r20 > best[0]:
            best = (r20, -float(a0), float(t0))
    return {
        "n": float(-a),
        "amplitude": float(np.exp(b)),
        "r2": float(r2),
        "n_virtual_origin": best[1],
        "t0_best": best[2],
        "r2_virtual_origin": float(best[0]),
        "n_points": int(tt.size),
        "t_lo": float(tt[0]),
        "t_hi": float(tt[-1]),
    }


def pileup_indicator(k: np.ndarray, e: np.ndarray, k_lo: float, k_hi: float, probe_frac: float = 0.85) -> dict:
    """Energy pile-up diagnostic at the spectral cut-off.

    A power law is fitted in ``[k_lo, k_hi]`` (inertial window) and
    extrapolated to ``k_probe = probe_frac * k_max``; the ratio
    ``E_measured(k_probe) / E_extrapolated(k_probe)`` is returned.  Values
    ``>> 1`` mean energy has piled up above the inertial power law, i.e. the
    sub-grid closure is not draining the cut-off scales (under-dissipation).
    """
    k = np.asarray(k, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    fit = fit_loglog_slope(k, e, k_lo, k_hi)
    k_probe = probe_frac * float(k.max())
    if not np.isfinite(fit["slope"]):
        return {"pileup_ratio": float("nan"), "k_probe": k_probe, "fit": fit}
    e_extrap = np.exp(fit["intercept"]) * k_probe ** fit["slope"]
    idx = int(np.argmin(np.abs(k - k_probe)))
    e_meas = float(e[idx])
    # local log-slope over the last available octave below the cut-off
    k_hi2 = float(k[max(idx - 4, 0) :].max())
    k_lo2 = float(k[max(idx - 4, 0) :].min())
    local = fit_loglog_slope(k, e, k_lo2, k_hi2)
    return {
        "pileup_ratio": float(e_meas / e_extrap) if e_extrap > 0 else float("nan"),
        "k_probe": float(k[idx]),
        "E_probe": e_meas,
        "E_extrap": float(e_extrap),
        "local_slope": local["slope"],
        "fit": fit,
    }


def spectrum_integral_error(k_a, e_a, k_b, e_b, k_lo: float, k_hi: float) -> dict:
    """Relative L1 difference of two spectra over ``[k_lo, k_hi]``.

    Both spectra are interpolated onto a common grid and integrated, so the
    metric answers "how much energy (in the window) does grid B miss with
    respect to grid A".  ``rel_l1`` = ``int|Ea-Eb|dk / int Ea dk``.
    """
    k_lo = max(k_lo, float(np.min(k_a)), float(np.min(k_b)))
    k_hi = min(k_hi, float(np.max(k_a)), float(np.max(k_b)))
    if k_hi <= k_lo:
        return {"rel_l1": float("nan"), "k_lo": k_lo, "k_hi": k_hi}
    grid = np.geomspace(k_lo, k_hi, 200)
    ea = np.interp(grid, k_a, e_a)
    eb = np.interp(grid, k_b, e_b)
    num = float(np.trapezoid(np.abs(ea - eb), grid))
    den = float(np.trapezoid(np.abs(ea), grid))
    return {
        "rel_l1": num / den if den > 0 else float("nan"),
        "int_a": den,
        "int_b": float(np.trapezoid(np.abs(eb), grid)),
        "k_lo": float(k_lo),
        "k_hi": float(k_hi),
        "n_grid": int(grid.size),
    }


def sig6(x):
    """Round a float (or nested container of floats) to 6 significant digits."""
    import math

    if isinstance(x, dict):
        return {k: sig6(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [sig6(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        if x == 0.0 or not math.isfinite(x):
            return x
        return float(f"{x:.6g}")
    return x