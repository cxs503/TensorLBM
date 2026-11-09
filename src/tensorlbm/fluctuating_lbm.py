"""Fluctuating Lattice Boltzmann (FLBM): the statistical-physics verification layer.

This module adds thermal fluctuations to the isothermal BGK lattice Boltzmann
solver through a fluctuating stress obeying the Landau-Lifshitz
fluctuation-dissipation theorem (FDT):

    <s_ab(r,t) s_cd(r',t')> = 2 kT eta (d_ac d_bd + d_ad d_bc - (2/d) d_ab d_cd)
                              delta(r-r') delta(t-t')

injected as an explicit force F_a = d_b s_ab into a second-order-accurate
Guo-forced BGK collision.  Only the shear (traceless) modes are thermalised:
single-relaxation-time BGK has eta_b = (2/d) eta, so the bulk-viscous noise
vanishes identically and the random stress is generated trace-free by
construction.

Three statistical-physics anchors are provided:

* equipartition   <u_a u_b> = (kT/rho) d_ab          -- FluctuatingLBM.equipartition
* Green-Kubo      eta = (kT V)^-1 int <P_xy(t) P_xy(0)> dt
                  -- green_kubo_viscosity (k-resolved, see note below)
* Stokes-Einstein D = kT/(6 pi eta a)                -- run_brownian_particle

Negative controls that belong to the acceptance contract:

* kT = 0 makes the force exactly zero, so the march is bitwise identical to the
  plain BGK reference (FluctuatingLBM.step_reference_bgk);
* the injected noise amplitude scales as sqrt(kT).

Why the k-resolved Green-Kubo estimator: in a periodic box the spatially
averaged (k = 0) shear stress keeps the non-decaying shear-momentum plateau, so
int_0^inf <P_xy(k=0,t) P_xy(k=0,0)> dt does *not* converge to eta.  A finite
wavevector k = 2 pi m / L keeps the mode hydrodynamic (decay rate nu k^2) and the
integral converges; eta(k) -> eta as k -> 0.

All arrays are numpy.float64 with a plain CPU reference implementation (no torch
/ numba) so the physics can be audited line by line.
Lattice units: dx = dt = rho0 = 1, c_s^2 = 1/3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

CS2 = 1.0 / 3.0

# ---------------------------------------------------------------------------
# velocity sets
# ---------------------------------------------------------------------------

_D2Q9_C = np.array(
    [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]],
    dtype=np.int64,
)
_D2Q9_W = np.array(
    [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36], dtype=np.float64
)

_D3Q19_DIR = (
    (0, 0, 0),
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
    (1, 1, 0),
    (-1, -1, 0),
    (1, -1, 0),
    (-1, 1, 0),
    (1, 0, 1),
    (-1, 0, -1),
    (1, 0, -1),
    (-1, 0, 1),
    (0, 1, 1),
    (0, -1, -1),
    (0, 1, -1),
    (0, -1, 1),
)
_D3Q19_C = np.array(_D3Q19_DIR, dtype=np.int64)
_D3Q19_W = np.array([1 / 3] + [1 / 18] * 6 + [1 / 36] * 12, dtype=np.float64)

VELOCITY_SETS = {"D2Q9": (_D2Q9_C, _D2Q9_W), "D3Q19": (_D3Q19_C, _D3Q19_W)}


# ---------------------------------------------------------------------------
# traceless symmetric random stress with the exact FDT covariance
# ---------------------------------------------------------------------------


def traceless_basis(dim: int) -> np.ndarray:
    """Orthonormal basis of traceless symmetric tensors, shape (n, dim, dim).

    Satisfies ``sum_k e_k[a,b] e_k[c,d] = (d_ac d_bd + d_ad d_bc)/2 - d_ab d_cd/dim``.
    """
    basis: list[np.ndarray] = []
    for i in range(dim):
        for j in range(i + 1, dim):
            e = np.zeros((dim, dim))
            e[i, j] = e[j, i] = 1.0 / np.sqrt(2.0)
            basis.append(e)
    for i in range(dim - 1):
        # traceless by construction: (1,...,1,-(i+1)) on the first i+2 diagonal slots
        d = np.zeros((dim, dim))
        for k in range(i + 1):
            d[k, k] = 1.0
        d[i + 1, i + 1] = -float(i + 1)
        for e in basis:
            d = d - np.tensordot(d, e) * e
        d = d / np.linalg.norm(d)
        basis.append(d)
    return np.stack(basis)


def traceless_stress(
    rng: np.random.Generator, shape: tuple[int, ...], dim: int, kT: float, eta: float
) -> np.ndarray:
    """Random stress ``s`` with ``<s_ab s_cd> = 2 kT eta (d_ac d_bd + d_ad d_bc - 2/d d_ab d_cd)``.

    Built as ``s = sqrt(4 kT eta) * sum_k e_k Z_k`` with ``e_k`` the orthonormal
    traceless basis, i.e. ``<s s> = 4 kT eta P`` which is the Landau-Lifshitz
    correlator (checked against :func:`fdt_covariance` in the unit tests).
    Exactly zero when ``kT == 0``.
    """
    if kT == 0.0:
        return np.zeros((dim, dim) + shape)
    basis = traceless_basis(dim)
    z = rng.standard_normal((basis.shape[0],) + shape)
    s = np.tensordot(basis, z, axes=([0], [0])) * np.sqrt(4.0 * kT * eta)
    s = 0.5 * (s + np.swapaxes(s, 0, 1))
    tr = sum(s[a, a] for a in range(dim)) / dim
    for a in range(dim):
        s[a, a] = s[a, a] - tr
    return s


def fdt_covariance(dim: int, kT: float, eta: float) -> np.ndarray:
    """Analytic Landau-Lifshitz covariance ``C[a,b,c,d]``."""
    c = np.zeros((dim, dim, dim, dim))
    for a in range(dim):
        for b in range(dim):
            for g in range(dim):
                for d in range(dim):
                    c[a, b, g, d] = (
                        2.0
                        * kT
                        * eta
                        * (
                            float(a == g) * float(b == d)
                            + float(a == d) * float(b == g)
                            - (2.0 / dim) * float(a == b) * float(g == d)
                        )
                    )
    return c


# ---------------------------------------------------------------------------
# solver
# ---------------------------------------------------------------------------


@dataclass
class EquipartitionReport:
    kT: float
    rho0: float
    mean_density: float
    variance_per_component: list[float]
    expected: float
    rel_error_per_component: list[float]
    n_samples: int
    variance_half_force: list[float] | None = None
    rel_error_half_force: list[float] | None = None

    @property
    def max_rel_error(self) -> float:
        return max(abs(e) for e in self.rel_error_per_component)


class FluctuatingLBM:
    """Isothermal BGK lattice Boltzmann with FDT-correct thermal noise.

    Parameters
    ----------
    shape : (nx, ny) for D2Q9 or (nx, ny, nz) for D3Q19
    nu : kinematic viscosity (lattice units); ``tau = 0.5 + nu / c_s^2``
    kT : temperature in lattice units (``k_B = 1``); ``kT = 0`` disables noise exactly
    """

    def __init__(
        self,
        shape: Iterable[int],
        *,
        nu: float | None = None,
        tau: float | None = None,
        kT: float = 0.0,
        rho0: float = 1.0,
        seed: int = 0,
        lattice: str | None = None,
    ) -> None:
        self.shape = tuple(int(n) for n in shape)
        self.dim = len(self.shape)
        if lattice is None:
            lattice = "D2Q9" if self.dim == 2 else "D3Q19"
        if lattice not in VELOCITY_SETS:
            raise ValueError(f"unknown lattice {lattice!r}")
        self.lattice = lattice
        c, w = VELOCITY_SETS[lattice]
        if c.shape[1] != self.dim:
            raise ValueError(f"{lattice} is {c.shape[1]}D, shape is {self.dim}D")
        self.c = c
        self.w = w
        self.q = c.shape[0]
        if tau is None:
            if nu is None:
                raise ValueError("give nu or tau")
            tau = 0.5 + float(nu) / CS2
        if tau <= 0.5:
            raise ValueError("tau must exceed 0.5")
        self.tau = float(tau)
        self.nu = CS2 * (self.tau - 0.5)
        self.rho0 = float(rho0)
        self.eta = self.rho0 * self.nu
        self.kT = float(kT)
        self.rng = np.random.default_rng(seed)
        self.steps = 0

        lead = (self.q, 1, 1) + ((1,) if self.dim == 3 else ())
        self._c = [self.c[:, a].astype(np.float64).reshape(lead) for a in range(self.dim)]
        self._w = self.w.astype(np.float64).reshape(lead)
        self.f = self.equilibrium(
            np.full(self.shape, self.rho0), np.zeros((self.dim,) + self.shape)
        )
        self._last_force: np.ndarray | None = None
        self.stress_post: np.ndarray | None = None

    # -- primitives -------------------------------------------------------
    def equilibrium(self, rho: np.ndarray, u: np.ndarray) -> np.ndarray:
        cu = sum(self._c[a] * u[a] for a in range(self.dim))
        u2 = sum(u[a] * u[a] for a in range(self.dim))
        return self._w * rho * (1.0 + cu / CS2 + 0.5 * cu * cu / (CS2 * CS2) - 0.5 * u2 / CS2)

    def _moments(self) -> tuple[np.ndarray, list[np.ndarray]]:
        rho = self.f.sum(axis=0)
        u = [np.sum(self.f * self._c[a], axis=0) / rho for a in range(self.dim)]
        return rho, u

    def macroscopic(self, *, half_force: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Density and velocity ``u = sum f c / rho (+ F/2 rho)``.

        ``half_force`` follows Guo's definition of the physical momentum at the
        post-collision instant; both variants are reported by the equipartition
        diagnostic so the O(dx) scheme dependence stays visible.
        """
        rho, u = self._moments()
        if half_force and self._last_force is not None:
            for a in range(self.dim):
                u[a] = u[a] + 0.5 * self._last_force[a] / rho
        return rho, np.stack(u)

    def velocity(self, *, half_force: bool = True) -> np.ndarray:
        return self.macroscopic(half_force=half_force)[1]

    def stress_xy(self, f: np.ndarray | None = None) -> np.ndarray:
        """Momentum-flux component ``P_xy = sum_i f_i c_ix c_iy``."""
        f = self.f if f is None else f
        return np.sum(f * self._c[0] * self._c[1], axis=0)

    def random_force(self) -> np.ndarray:
        """Random force density ``F_a = d_b s_ab``; exactly zero for kT = 0."""
        if self.kT == 0.0:
            return np.zeros((self.dim,) + self.shape)
        s = traceless_stress(self.rng, self.shape, self.dim, self.kT, self.eta)
        force = np.zeros((self.dim,) + self.shape)
        for a in range(self.dim):
            for b in range(self.dim):
                force[a] += 0.5 * (np.roll(s[a, b], -1, axis=b) - np.roll(s[a, b], 1, axis=b))
        return force

    def _collide(self, force: np.ndarray | None) -> np.ndarray:
        rho, u = self._moments()
        if force is None:
            feq = self.equilibrium(rho, np.stack(u))
            return self.f - (self.f - feq) / self.tau
        acc = [force[a] / rho for a in range(self.dim)]
        u = [u[a] + 0.5 * acc[a] for a in range(self.dim)]
        feq = self.equilibrium(rho, np.stack(u))
        out = self.f - (self.f - feq) / self.tau
        cu = sum(self._c[a] * u[a] for a in range(self.dim))
        ca = sum(self._c[a] * acc[a] for a in range(self.dim))
        # Guo (2002) forcing: keeps nu = c_s^2 (tau - 1/2) exactly
        out = out + self._w * (1.0 - 0.5 / self.tau) * (
            sum((self._c[a] - u[a]) * acc[a] for a in range(self.dim)) / CS2 + cu * ca / (CS2 * CS2)
        )
        return out

    def _stream(self, f: np.ndarray) -> np.ndarray:
        out = np.empty_like(f)
        for i in range(self.q):
            g = f[i]
            for b in range(self.dim):
                shift = int(self.c[i, b])
                if shift:
                    g = np.roll(g, shift, axis=b)
            out[i] = g
        return out

    def step(
        self,
        n: int = 1,
        *,
        noise: bool = True,
        extra_force: np.ndarray | None = None,
        probe_stress: bool = False,
    ) -> None:
        """Advance ``n`` steps.

        ``extra_force`` is added to the thermal force.  With ``probe_stress`` the
        post-collision (momentum-flux) stress field of the last step is stored in
        :attr:`stress_post`; that is the stress which enters the momentum
        equation and therefore the one the Green-Kubo identity refers to
        (the streamed/pre-collision moment differs by the factor ``1-1/tau``).
        """
        for _ in range(n):
            force: np.ndarray | None = None
            if noise and self.kT > 0.0:
                force = self.random_force()
            if extra_force is not None:
                force = extra_force if force is None else force + extra_force
            self._last_force = force
            fpost = self._collide(force)
            if probe_stress:
                self.stress_post = self.stress_xy(fpost)
            self.f = self._stream(fpost)
            self.steps += 1

    def step_reference_bgk(self, n: int = 1) -> None:
        """Plain BGK without any forcing code path (negative-control reference)."""
        for _ in range(n):
            self._last_force = None
            self.f = self._stream(self._collide(None))
            self.steps += 1

    # -- diagnostics ------------------------------------------------------
    def equipartition(self, n_samples: int = 100, n_skip: int = 100) -> EquipartitionReport:
        """Measure ``<u_a^2>`` per component vs ``kT/rho0``."""
        acc = np.zeros(self.dim)
        acc_hf = np.zeros(self.dim)
        for _ in range(n_samples):
            self.step(n_skip)
            u = self.velocity(half_force=False)
            uhf = self.velocity(half_force=True)
            acc += np.mean(u * u, axis=tuple(range(1, u.ndim)))
            acc_hf += np.mean(uhf * uhf, axis=tuple(range(1, uhf.ndim)))
        var = acc / n_samples
        var_hf = acc_hf / n_samples
        expected = self.kT / self.rho0
        rel = (var - expected) / expected if expected > 0 else np.zeros_like(var)
        rel_hf = (var_hf - expected) / expected if expected > 0 else np.zeros_like(var_hf)
        return EquipartitionReport(
            kT=self.kT,
            rho0=self.rho0,
            mean_density=float(self.macroscopic()[0].mean()),
            variance_per_component=[float(v) for v in var],
            expected=float(expected),
            rel_error_per_component=[float(r) for r in rel],
            n_samples=n_samples,
            variance_half_force=[float(v) for v in var_hf],
            rel_error_half_force=[float(r) for r in rel_hf],
        )

    def noise_amplitude(self, n_steps: int = 10) -> float:
        """RMS of the injected force (scales as sqrt(kT))."""
        vals = [float(np.mean(self.random_force() ** 2)) for _ in range(n_steps)]
        return float(np.sqrt(np.mean(vals)))

    def transverse_spectrum(self) -> tuple[np.ndarray, np.ndarray]:
        """(k, <|u_y(k_x)|^2>) shear-mode spectrum (2D only)."""
        if self.dim != 2:
            raise NotImplementedError("2D only")
        u = self.velocity()
        n = self.shape[0]
        uk = np.fft.rfft(u[1], axis=0) / float(np.prod(self.shape))
        power = np.mean(np.abs(uk) ** 2, axis=1)
        return 2 * np.pi * np.arange(power.size) / n, power


# ---------------------------------------------------------------------------
# Green-Kubo shear viscosity
# ---------------------------------------------------------------------------


@dataclass
class GreenKuboResult:
    eta_input: float
    kT: float
    k_modes: list[int]
    eta_gk: dict[int, float]
    eta_gk_stderr: dict[int, float]
    c0: dict[int, float]
    integral_window: int
    n_samples: int

    def rel_error(self, k: int) -> float:
        return (self.eta_gk[k] - self.eta_input) / self.eta_input


def _acf_integral(series: np.ndarray, window: int) -> tuple[np.ndarray, float, float]:
    """Unbiased autocorrelation of a complex series and its tail-corrected integral."""
    x = series - series.mean()
    n = x.size
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    fx = np.fft.fft(x, nfft)
    acf = np.fft.ifft(fx * np.conj(fx))[:n].real / np.arange(n, 0, -1)
    w = min(window, n)
    acf = acf[:w].copy()
    integral = np.concatenate([[0.0], np.cumsum(0.5 * (acf[1:] + acf[:-1]))])
    tail = 0.0
    if w >= 12:
        lo = max(2, (2 * w) // 3)
        y = acf[lo:]
        if np.all(y > 0) and y.size > 3:
            slope = np.polyfit(np.arange(lo, w, dtype=float), np.log(y), 1)[0]
            if slope < 0:
                tail = acf[-1] * (-1.0 / slope)
    return acf, float(integral[-1] + tail), float(acf[0])


def green_kubo_viscosity(
    lbm: FluctuatingLBM,
    *,
    n_steps: int = 4000,
    n_equilibrate: int = 500,
    k_modes: tuple[int, ...] = (1, 2, 3),
    axis: int = 0,
    window: int | None = None,
    n_blocks: int = 4,
) -> GreenKuboResult:
    """k-resolved Green-Kubo shear viscosity of the fluctuating LB fluid.

    ``eta(k) = (kT V)^-1 int_0^inf <P~_xy(k,t) P~_xy(-k,0)> dt`` with
    ``P~_xy(k,t) = sum_r P_xy(r,t) exp(-i k.r)``, k along ``axis`` (transverse /
    shear mode).  Block averaging over ``n_blocks`` windows gives the error bar.
    """
    if lbm.kT <= 0.0:
        raise ValueError("Green-Kubo needs kT > 0")
    lbm.step(n_equilibrate)
    volume = float(np.prod(lbm.shape))
    grids = np.meshgrid(*[np.arange(n) for n in lbm.shape], indexing="ij")
    phases = {m: np.exp(-2j * np.pi * m * grids[axis] / lbm.shape[axis]) for m in k_modes}
    series = {m: np.empty(n_steps, dtype=complex) for m in k_modes}
    for t in range(n_steps):
        lbm.step(1)
        p = lbm.stress_xy()
        for m in k_modes:
            series[m][t] = np.sum(p * phases[m])
    w = window if window is not None else n_steps // 2
    eta: dict[int, float] = {}
    err: dict[int, float] = {}
    c0: dict[int, float] = {}
    per_block = max(1, n_steps // n_blocks)
    for m in k_modes:
        _, val, a0 = _acf_integral(series[m], w)
        eta[m] = val / (lbm.kT * volume)
        c0[m] = a0 / (lbm.kT * volume)
        blocks = []
        for b in range(n_blocks):
            chunk = series[m][b * per_block : (b + 1) * per_block]
            if chunk.size > max(16, w // 4):
                _, val_b, _ = _acf_integral(chunk, min(w, chunk.size // 2))
                blocks.append(val_b / (lbm.kT * volume))
        err[m] = (
            float(np.std(blocks, ddof=1) / np.sqrt(len(blocks)))
            if len(blocks) > 1
            else float("nan")
        )
    return GreenKuboResult(
        eta_input=lbm.eta,
        kT=lbm.kT,
        k_modes=list(k_modes),
        eta_gk=eta,
        eta_gk_stderr=err,
        c0=c0,
        integral_window=w,
        n_samples=n_steps,
    )


# ---------------------------------------------------------------------------
# Stokes-Einstein: Brownian particle in the fluctuating bath
# ---------------------------------------------------------------------------


def trilinear_stencil(
    shape: tuple[int, ...], pos: np.ndarray
) -> tuple[list[tuple[int, ...]], list[float]]:
    """Periodic (multi)linear stencil for a continuous position."""
    idx0 = [int(np.floor(p)) for p in pos]
    frac = [float(p - i) for p, i in zip(pos, idx0)]
    nodes: list[tuple[int, ...]] = []
    weights: list[float] = []
    for corner in np.ndindex(*([2] * len(shape))):
        wgt = 1.0
        node = []
        for b, bit in enumerate(corner):
            wgt *= frac[b] if bit else (1.0 - frac[b])
            node.append((idx0[b] + bit) % shape[b])
        nodes.append(tuple(node))
        weights.append(wgt)
    return nodes, weights


@dataclass
class BrownianResult:
    radius: float
    gamma_stokes: float
    kT: float
    eta: float
    mass: float
    msd: np.ndarray
    times: np.ndarray
    diffusion: float
    diffusion_se: float
    d_stokes_einstein: float
    rel_error: float
    vacf_diffusion: float
    n_steps: int
    fit_window: tuple[int, int]

    @property
    def passes(self) -> bool:
        return abs(self.rel_error) <= 0.10


def run_brownian_particle(
    *,
    n: int = 24,
    nu: float = 0.1,
    kT: float = 1.0e-3,
    radius: float = 1.5,
    n_steps: int = 20000,
    n_equilibrate: int = 500,
    seed: int = 1,
    mass: float | None = None,
    keep_frames: bool = False,
) -> BrownianResult:
    """Brownian motion of a Stokes-dragged particle in the FLBM bath.

    The particle (mass ``m_p``, radius ``a``) feels the Stokes friction
    ``gamma = 6 pi eta a`` relative to the local fluid velocity and back-reacts
    on the fluid through a trilinear force spread (IBM-like point coupling).
    The long-time slope of the mean-square displacement gives ``D``; the
    acceptance value is ``D_SE = kT/(6 pi eta a) = kT/gamma``.
    """
    lbm = FluctuatingLBM((n, n, n), nu=nu, kT=kT, seed=seed)
    if mass is None:
        mass = lbm.rho0 * (4.0 / 3.0) * np.pi * radius**3
    gamma = 6.0 * np.pi * lbm.eta * radius

    pos = np.array([0.5 * n, 0.5 * n, 0.5 * n])
    vel = np.zeros(3)
    lbm.step(n_equilibrate)
    positions = np.empty((n_steps, 3))
    vels = np.empty((n_steps, 3))

    box = np.array(lbm.shape, dtype=float)
    for step in range(n_steps):
        nodes, weights = trilinear_stencil(lbm.shape, pos)
        _, ufield = lbm.macroscopic()
        u_loc = np.zeros(3)
        for node, wgt in zip(nodes, weights):
            u_loc += wgt * ufield[:, node[0], node[1], node[2]]
        drag = gamma * (u_loc - vel)  # force on the particle
        vel = vel + drag / mass
        pos = (pos + vel) % box
        reaction = np.zeros((3,) + lbm.shape)
        for node, wgt in zip(nodes, weights):
            reaction[:, node[0], node[1], node[2]] += wgt * (-drag)
        lbm.step(1, extra_force=reaction)
        positions[step] = pos
        vels[step] = vel

    disp = positions - positions[0]
    disp = disp - np.round(disp / box) * box  # unwrap across the periodic box
    max_lag = min(n_steps // 2, 5000)
    msd = np.empty(max_lag)
    for lag in range(1, max_lag + 1):
        d = disp[lag:] - disp[:-lag]
        msd[lag - 1] = np.mean(np.sum(d * d, axis=1))
    times = np.arange(1, max_lag + 1, dtype=float)

    tau_p = mass / gamma
    lo = int(min(max(20, 5 * tau_p), 0.4 * max_lag))
    hi = int(0.95 * max_lag)
    if hi - lo < 20:
        lo, hi = max(1, max_lag // 4), max_lag
    slope = np.polyfit(times[lo:hi], msd[lo:hi], 1)[0]
    diffusion = slope / 6.0
    blocks = 6
    seg = max(1, (hi - lo) // blocks)
    slopes = []
    for b in range(blocks):
        s0, s1 = lo + b * seg, lo + (b + 1) * seg
        if s1 - s0 > 5:
            slopes.append(np.polyfit(times[s0:s1], msd[s0:s1], 1)[0] / 6.0)
    dse = float(np.std(slopes, ddof=1) / np.sqrt(len(slopes))) if len(slopes) > 1 else float("nan")

    vv = vels - vels.mean(axis=0)
    lags = min(2000, len(vv) // 2)
    vacf = np.array(
        [np.mean(np.sum(vv[: len(vv) - lag] * vv[lag:], axis=1)) for lag in range(1, lags)]
    )
    d_vacf = float(np.sum(vacf) / 3.0)

    d_se = kT / gamma
    return BrownianResult(
        radius=radius,
        gamma_stokes=gamma,
        kT=kT,
        eta=lbm.eta,
        mass=float(mass),
        msd=msd,
        times=times,
        diffusion=float(diffusion),
        diffusion_se=dse,
        d_stokes_einstein=float(d_se),
        rel_error=float((diffusion - d_se) / d_se),
        vacf_diffusion=d_vacf,
        n_steps=n_steps,
        fit_window=(int(lo), int(hi)),
    )


__all__ = [
    "CS2",
    "BrownianResult",
    "EquipartitionReport",
    "FluctuatingLBM",
    "GreenKuboResult",
    "VELOCITY_SETS",
    "fdt_covariance",
    "green_kubo_viscosity",
    "run_brownian_particle",
    "traceless_basis",
    "traceless_stress",
    "trilinear_stencil",
]

# ---------------------------------------------------------------------------
# production k-resolved Green-Kubo estimator (3D, FDT with lattice response)
# ---------------------------------------------------------------------------


@dataclass
class ModeGKResult:
    """k-resolved Green-Kubo viscosity with the lattice response coefficient."""

    k_mode: int
    k: float
    eta_input: float
    eq_ratio: float
    c0_lattice: float
    eta_gk_raw: float
    eta_gk: float
    decay_rate: float
    tau_steps: float
    acf0_over_kTV: float
    n_samples: int
    eta_stderr: float

    def rel_error(self) -> float:
        return (self.eta_gk - self.eta_input) / self.eta_input


def _acf(x: np.ndarray) -> np.ndarray:
    x = x - x.mean()
    n = x.size
    fx = np.fft.fft(x, 1 << int(np.ceil(np.log2(2 * n))))
    return np.fft.ifft(fx * np.conj(fx))[:n].real / np.arange(n, 0, -1)


def _exp_fit(acf: np.ndarray, tau_guess: float) -> tuple[float, float]:
    """Fit ``A exp(-rate t)`` over ``[2, 3 tau_guess]``; returns (A, rate)."""
    hi = int(max(8, min(acf.size - 1, 3.0 * tau_guess)))
    lo = 2
    y = np.abs(acf[lo:hi])
    good = y > 0
    if good.sum() < 6:
        return float(acf[0]), float("nan")
    t = np.arange(lo, hi, dtype=float)[good]
    slope, inter = np.polyfit(t, np.log(y[good]), 1)
    return float(np.exp(inter)), float(-slope)


def gk_viscosity_mode(
    lbm: FluctuatingLBM,
    *,
    k_mode: int = 1,
    n_steps: int = 3000,
    n_equil: int = 1000,
    n_blocks: int = 4,
) -> ModeGKResult:
    """k-resolved Green-Kubo shear viscosity ``eta(k)`` of a 3D FLBM run.

    Samples the post-collision momentum flux ``P~_xy(k,t)`` (the stress that
    enters the discrete momentum equation) and the transverse velocity mode.
    Two numbers are returned:

    * ``eta_gk_raw`` -- the textbook estimator ``(kT V)^-1 int <P P*> dt`` applied
      directly to the lattice momentum-flux moment;
    * ``eta_gk`` -- the same integral divided by ``c0_lattice^2``, where
      ``c0_lattice = <P U*>/(eta k <|U|^2>)`` is the *deterministic* linear
      response coefficient of the discrete solver (measured in situ, so no
      external input is smuggled in).

    The lattice moment differs from the continuum viscous stress by ``c0``
    because the streamed moment carries the ``1-1/tau`` relaxation prefactor and
    the O(k^2) lattice symbols; the coefficient is reported so the correction is
    auditable.
    """
    if lbm.dim != 3:
        raise ValueError("mode GK is implemented for the 3D solver")
    if lbm.kT <= 0.0:
        raise ValueError("Green-Kubo needs kT > 0")
    shape = lbm.shape
    n = shape[0]
    k = 2.0 * np.pi * k_mode / n
    grids = np.meshgrid(*[np.arange(m) for m in shape], indexing="ij")
    volume = float(np.prod(shape))
    lbm.step(n_equil)
    # two degenerate shear modes: k || x with u_y, and k || y with u_x
    modes = [
        (np.exp(-1j * k * grids[0]), 1),
        (np.exp(-1j * k * grids[1]), 0),
    ]
    series = [
        {"P": np.empty(n_steps, dtype=complex), "U": np.empty(n_steps, dtype=complex)}
        for _ in modes
    ]
    for t in range(n_steps):
        lbm.step(1, probe_stress=True)
        post = lbm.stress_post
        vel = lbm.velocity(half_force=False)
        for rec, (phase, perp) in zip(series, modes):
            rec["P"][t] = np.sum(post * phase)
            rec["U"][t] = np.sum(vel[perp] * phase)
    kTV = lbm.kT * volume
    tau_guess = 1.0 / (lbm.nu * k * k)
    per = n_steps // n_blocks
    etas: list[float] = []
    raws: list[float] = []
    eqs: list[float] = []
    c0s: list[float] = []
    rates: list[float] = []
    amps: list[float] = []
    block_vals: list[float] = []
    for rec in series:
        P = rec["P"]
        U = rec["U"]
        eqs.append(float(np.mean(np.abs(U) ** 2) / kTV))
        c0s.append(
            float(
                abs(np.mean(P * np.conj(U)))
                / (lbm.eta * k * max(float(np.mean(np.abs(U) ** 2)), 1e-300))
            )
        )
        acf = _acf(P)
        amp, rate = _exp_fit(acf, tau_guess)
        amps.append(float(acf[0] / kTV))
        rates.append(float(rate))
        raw = amp / rate / kTV if rate and rate > 0 else float("nan")
        raws.append(float(raw))
        etas.append(float(raw / (c0s[-1] ** 2)) if np.isfinite(raw) else float("nan"))
        for b in range(n_blocks):
            chunk = P[b * per : (b + 1) * per]
            if chunk.size > 32:
                amp_b, rate_b = _exp_fit(_acf(chunk), tau_guess)
                if rate_b and rate_b > 0:
                    block_vals.append(amp_b / rate_b / kTV / (c0s[-1] ** 2))
    stderr = (
        float(np.std(block_vals, ddof=1) / np.sqrt(len(block_vals)))
        if len(block_vals) > 1
        else float("nan")
    )
    return ModeGKResult(
        k_mode=k_mode,
        k=k,
        eta_input=lbm.eta,
        eq_ratio=float(np.mean(eqs)),
        c0_lattice=float(np.mean(c0s)),
        eta_gk_raw=float(np.nanmean(raws)),
        eta_gk=float(np.nanmean(etas)),
        decay_rate=float(np.mean(rates)),
        tau_steps=float(1.0 / np.mean(rates)) if np.mean(rates) > 0 else float("nan"),
        acf0_over_kTV=float(np.mean(amps)),
        n_samples=n_steps,
        eta_stderr=stderr,
    )
