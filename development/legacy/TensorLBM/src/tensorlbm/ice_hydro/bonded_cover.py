"""Bonded-particle (lattice DEM) idealisation of the ice cover.

The cover is discretised into ``N`` particles (nodes) at spacing ``a = L/(N-1)``.
Consecutive bonds carry

* a **rotational (bending) spring** of stiffness ``k_r = D / a`` -- the classical
  bonded-particle beam identity: the nodal moment of such a spring chain is
  ``M_i = k_r (theta_{i+1/2} - theta_{i-1/2}) = D (w_{i+1} - 2 w_i + w_{i-1}) / a^2``,
  i.e. exactly the Euler-Bernoulli curvature relation (verified in the test-suite
  against the analytical cantilever / simply-supported solutions);
* an **axial spring chain** that transmits the membrane load ``N`` [N/m] applied to
  the cover (used for the momentum-flux-compression loading model);
* **foundation springs** ``k_f = rho_w g`` per unit length (buoyancy, Winkler
  foundation) -- the elastic-foundation restoring term of the prescribed criterion;
* **breakable bonds**: a bond fails when the transmitted moment reaches the
  flexural capacity ``M_f = sigma_f h_i^2 / 6``.  A failed bond becomes an internal
  hinge (zero moment release) and stays failed: the crack is irreversible, which
  yields a *crack sequence* (positions + times) rather than a single onset
  indicator.

Loading models (selectable):
* ``hydro``   : distributed cover pressure from the LBM-measured law
  ``p_hat = gain * rho_w U^2 k coth(k h) w_hat`` applied as a **non-local
  spectral operator** (each wavenumber carries its own gain);
* ``momentum``: membrane compression ``N = rho_w U^2`` corresponding to the
  prescribed k^2 criterion.

Time integration is explicit central difference with mass-proportional damping
(dynamic relaxation), the standard DEM stepping scheme.

Known approximation (documented in the report): a failed bond is implemented by
replacing its row with the zero-curvature (hinge) condition instead of a full
contact/penalty formulation.  This keeps the linear algebra symmetric and is
adequate for locating the crack sequence; it slightly over-stiffens the post-crack
structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .analytical import CoverProperties


@dataclass
class CrackEvent:
    time: float
    position: float
    node: int
    moment: float


@dataclass
class BondedCover:
    props: CoverProperties
    n_nodes: int = 129
    damping_ratio: float = 0.08
    breakable: bool = True

    def __post_init__(self) -> None:
        self.N = self.n_nodes
        self.x = np.linspace(0.0, self.props.L, self.N)
        self.a = self.props.L / (self.N - 1)
        self.kf = self.props.rho_w * self.props.g
        self.hinges: set[int] = set()
        self.cracks: list[CrackEvent] = []
        self.mass = self.props.mass_per_area          # kg/m^2 (plate only)
        self._Pcache: np.ndarray | None = None

    # ------------------------------------------------------------------ operators
    def stiffness_matrix(
        self,
        membrane: float = 0.0,
        feedback: float = 0.0,
        hinges: Iterable[int] | None = None,
    ) -> np.ndarray:
        """Assemble the (N x N) lattice stiffness matrix.

        membrane : axial compression N [N/m] (destabilising, enters as +N w'')
        feedback : *local* pressure gain C [Pa/m] with p = C w  (destabilising);
                   used only for the closed-form cross-check, the spectral
                   operator :meth:`feedback_operator` is used for U-sweeps.
        """
        N, a, D, kf = self.N, self.a, self.props.D, self.kf
        hs = set(self.hinges if hinges is None else hinges)
        K = np.zeros((N, N))
        for i in range(1, N - 1):
            if i in hs:                      # released bond -> hinge (M_i = 0)
                K[i, i - 1] += 1.0 / a**2
                K[i, i] += -2.0 / a**2
                K[i, i + 1] += 1.0 / a**2
                continue
            stencil = {i - 2: 1.0, i - 1: -4.0, i: 6.0, i + 1: -4.0, i + 2: 1.0}
            for j, c in stencil.items():
                sign = 1.0
                jj = j
                if j == -1:                  # ghost: simply supported end (w''=0)
                    jj, sign = 1, -1.0
                elif j == N:                 # ghost at the far end
                    jj, sign = N - 2, -1.0
                if jj in (0, N - 1):         # pinned: w = 0
                    continue
                K[i, jj] += sign * c * D / a**4
            if membrane:
                for j, c in ((i - 1, 1.0), (i, -2.0), (i + 1, 1.0)):
                    if j in (0, N - 1):
                        continue
                    K[i, j] += membrane * c / a**2
            K[i, i] += kf
            if feedback:
                K[i, i] -= feedback
        K[0, 0] = 1.0
        K[N - 1, N - 1] = 1.0
        return K

    def feedback_operator(self, h: float, gain: float = 1.0) -> np.ndarray:
        """Non-local hydrodynamic feedback operator L with p = rho_w U^2 L[w].

        In Fourier space ``L(k) = gain * k / tanh(k h)`` (limit ``1/h`` at k=0),
        which is the linear potential-flow wall pressure over a deflected cover.
        Built column by column (N FFTs) so that it can be used with any solver.
        """
        if self._Pcache is not None and gain == 1.0:
            return self._Pcache
        N, a = self.N, self.a
        k = 2.0 * np.pi * np.fft.fftfreq(N, d=a)
        kh = k * h
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.where(np.abs(kh) < 1e-9, 1.0 / h, k / np.tanh(np.where(np.abs(kh) < 1e-9, 1.0, kh)))
        P = np.empty((N, N))
        e = np.eye(N)
        P[:] = np.fft.ifft(g[:, None] * np.fft.fft(e, axis=0), axis=0).real
        P = gain * P
        if gain == 1.0:
            self._Pcache = P
        return P

    @property
    def _interior(self) -> slice:
        return slice(1, self.N - 1)

    def min_eigenvalue(self, membrane: float = 0.0, feedback: float = 0.0,
                       hinges: Iterable[int] | None = None) -> float:
        K = self.stiffness_matrix(membrane, feedback, hinges)[self._interior, self._interior]
        return float(np.linalg.eigvalsh(K)[0])

    def _hydro_matrices(self, gain: float) -> tuple[np.ndarray, np.ndarray]:
        K = self.stiffness_matrix()[self._interior, self._interior]
        P = self.feedback_operator(self.props.h, gain)[self._interior, self._interior]
        return K, P

    def onset_velocity(self, load_model: str, gain: float = 1.0,
                       k: float | None = None) -> float:
        """Smallest U at which the lattice stiffness loses positive definiteness.

        load_model = "momentum": N = rho_w U^2               (prescribed k^2 criterion)
        load_model = "hydro"   : spectral feedback, p = gain rho_w U^2 L[w]
        """
        from scipy.optimize import brentq

        p = self.props
        if load_model == "momentum":
            def lam(u: float) -> float:
                return self.min_eigenvalue(membrane=p.rho_w * u * u)
        elif load_model == "hydro":
            K, P = self._hydro_matrices(gain)

            def lam(u: float) -> float:
                return float(np.linalg.eigvalsh(K - p.rho_w * u * u * P)[0])
        else:
            raise ValueError(load_model)

        lo, hi = 1e-3, 10.0
        while lam(hi) > 0.0 and hi < 1.0e5:
            hi *= 2.0
        if lam(lo) <= 0.0:
            return 0.0
        return float(brentq(lam, lo, hi, xtol=1e-6, rtol=1e-10))

    # ------------------------------------------------------------------ statics
    def static_deflection(self, pressure: np.ndarray, membrane: float = 0.0,
                          feedback: float = 0.0) -> np.ndarray:
        K = self.stiffness_matrix(membrane, feedback)
        w = np.linalg.solve(K, pressure)
        w[0] = 0.0
        w[-1] = 0.0
        return w

    def bond_moments(self, w: np.ndarray) -> np.ndarray:
        M = np.zeros(self.N)
        M[1:-1] = self.props.D * (w[2:] - 2.0 * w[1:-1] + w[:-2]) / self.a**2
        return M

    # ------------------------------------------------------------------ dynamics
    def transient(
        self,
        load_model: str,
        U: float,
        gain: float = 1.0,
        t_end: float = 30.0,
        seed_mode: int = 1,
        seed_moment_fraction: float = 0.05,
        max_cracks: int = 8,
        dt_factor: float = 0.5,
        include_added_mass: bool = True,
    ) -> dict:
        """Dynamic relaxation with irreversible bond breaking -> crack sequence.

        The cover starts from rest with an initial displacement in mode
        ``seed_mode`` whose peak bond moment equals ``seed_moment_fraction * M_f``.
        Below the critical velocity the oscillation decays (damping) and no bond
        fails; above it the mode grows exponentially and the first bond to reach
        ``M_f`` is recorded (time + position); the structure is then re-solved with
        that bond released, producing the crack *sequence*.
        """
        p = self.props
        N = self.N
        x = self.x
        self.hinges = set()
        self.cracks = []

        K = self.stiffness_matrix()
        Ki = K[self._interior, self._interior]

        k1 = np.pi / p.L
        added = p.rho_w / (k1 * np.tanh(k1 * p.h)) if include_added_mass else 0.0
        mass = p.mass_per_area + added
        self.mass = mass
        lam0 = float(np.linalg.eigvalsh(Ki)[0])
        omega1 = np.sqrt(max(lam0 / mass, 1e-12))
        c_damp = 2.0 * self.damping_ratio * omega1
        dt = dt_factor * self.a**2 * np.sqrt(mass / p.D) / 4.0
        n_steps = int(t_end / dt)

        shape = np.sin(seed_mode * np.pi * x / p.L)
        M_shape = self.bond_moments(shape)
        w_amp = seed_moment_fraction * p.M_f / max(np.max(np.abs(M_shape)), 1e-30)
        w = w_amp * shape
        v = np.zeros(N)

        if load_model == "momentum":
            Kfull = self.stiffness_matrix(membrane=p.rho_w * U * U)
        elif load_model == "hydro":
            Kfull = _pad_interior(Ki - p.rho_w * U * U
                                  * self.feedback_operator(p.h, gain)[self._interior,
                                                                     self._interior], N)
        else:
            raise ValueError(load_model)
        M_int = Kfull[self._interior, self._interior]
        t = 0.0
        M_max_hist: list[float] = []
        for step in range(n_steps):
            t = step * dt
            f_int = M_int @ w[self._interior]
            acc = -f_int / mass - c_damp * v[self._interior]
            v[self._interior] += dt * acc
            w[self._interior] += dt * v[self._interior]
            if step % 10 == 0:
                M = self.bond_moments(w)
                M_max_hist.append(float(np.max(np.abs(M))))
                if self.breakable:
                    cracked = False
                    for _ in range(4):
                        cand = [i for i in range(1, N - 1) if i not in self.hinges]
                        idx = cand[int(np.argmax(np.abs(M[cand])))]
                        if abs(M[idx]) <= p.M_f:
                            break
                        self.cracks.append(CrackEvent(time=t, position=float(x[idx]),
                                                      node=idx, moment=float(M[idx])))
                        self.hinges.add(idx)
                        cracked = True
                        if len(self.cracks) >= max_cracks:
                            break
                    if cracked:
                        Kfull = self.stiffness_matrix(
                            membrane=(p.rho_w * U * U if load_model == "momentum" else 0.0))
                        if load_model == "hydro":
                            Kfull = _pad_interior(
                                Kfull[self._interior, self._interior]
                                - p.rho_w * U * U * self.feedback_operator(
                                    p.h, gain)[self._interior, self._interior], N)
                        M_int = Kfull[self._interior, self._interior]
                        M = self.bond_moments(w)
                if len(self.cracks) >= max_cracks:
                    break
        return {
            "load_model": load_model,
            "U": U,
            "seed_mode": seed_mode,
            "seed_moment_fraction": seed_moment_fraction,
            "dt": dt,
            "t_end": t_end,
            "n_steps": n_steps,
            "cracks": [c.__dict__ for c in self.cracks],
            "n_cracks": len(self.cracks),
            "w_max": float(np.max(np.abs(w))),
            "M_max": float(np.max(np.abs(self.bond_moments(w)))),
            "M_f": p.M_f,
            "M_max_history": M_max_hist,
            "w_final": w.tolist(),
        }


def _pad_interior(Kint: np.ndarray, N: int) -> np.ndarray:
    """Embed an interior-block matrix back into the full (pinned) system."""
    K = np.zeros((N, N))
    K[1:-1, 1:-1] = Kint
    K[0, 0] = 1.0
    K[N - 1, N - 1] = 1.0
    return K
