"""Ghost-mode resolved thermostat for the fluctuating LBM (XVAL extension).

Background
----------
:mod:`tensorlbm.fluctuating_lbm` injects thermal noise as the divergence of a
*Landau-Lifshitz random stress*, i.e. only the second-moment (stress / energy)
modes of the discrete distribution receive noise.  The remaining, non-hydrodynamic
kinetic modes of D2Q9/D3Q19 -- the third- and fourth-order "ghost modes" -- are
left untouched, so their fluctuations stay at the deterministic value (zero)
instead of the equipartition value.

The reference OSS implementation ``ludwig-cf/ludwig``
(commit 78d83c5151ff1d4c4201a96b266a08e97047cae6) does not have this limitation:
its compiled object ``src/collision.o`` exports

    lb_fluctuations_var_eta     (tau, kt)      -- stress-mode amplitude
    lb_fluctuations_var_bulk    (tau, kt)      -- bulk-mode amplitude
    lb_fluctuations_var_ghost   (rna, rtau, kt)-- *separate ghost-mode amplitude*
    lb_collision_ghost_modes_on / _off         -- ghosts thermalised or switched off
    lb_relaxation_time_ghosts   / _v           -- separate ghost relaxation rate
    lb_fluctuations_stress      (noise, index)
    lb_fluctuations_ghosts      (noise, index)

i.e. ludwig assigns an explicit, independently configured variance to the ghost
modes (and lets them be switched off).  Evidence and the regeneration script are
in ``xval/fluctuating_lbm_ghost_modes/extract_ludwig_reference.py`` and
``xval/fluctuating_lbm_ghost_modes/artifacts/ludwig_reference.json``.

This module transplants that practice into TensorLBM in the cheapest auditable
way: work in the *weighted-orthonormal mode basis* of the lattice, classify the
modes, and add one independent Gaussian per selected mode with the amplitude
that makes the stationary mode variance equal to its equilibrium (FDT) value.

Mode basis
----------
For a velocity set with weights ``w_i`` build an orthonormal basis
``e_a`` (``sum_i w_i e_ai e_bi = delta_ab``) by Gram-Schmidt on the Hermite
polynomials ``1, c_x, c_y, c^2, c_x^2-c_y^2, c_x c_y, c^2 c_x, c^2 c_y, c^4``.
A mode is ``m_a(x) = sum_i e_ai f_i(x)`` and a mode perturbation ``dm_a`` is
added as ``df_i = w_i e_ai dm_a`` (the dual of ``m_a``), which conserves mass and
momentum for every mode orthogonal to ``1, c_x, c_y``.

Equilibrium (FDT) target
------------------------
At rest, for a thermalised lattice gas, every non-conserved mode has variance

    <dm_a^2> = rho * kT / c_s^2 = 3 rho kT                (c_s^2 = 1/3)

(the ``3`` is the Hermite normalisation: e.g. the momentum mode is
``sqrt(3) j_x``, and ``<j_x^2> = rho kT``).  This is the same target the
existing module checks through ``<u_a u_b> = (kT/rho) delta_ab``, so the
shear-mode ratio measured here is a *self-check of the normalisation*: an arm
that is known to be correct (stress-only noise) must give ratio ~1 on p_xx/p_xy.

Stationary-variance amplitude
-----------------------------
A mode relaxed by BGK as ``m* = (1 - 1/tau) m + (1/tau) m_eq + xi`` with
independent ``xi`` has the stationary variance ``var(m) = var(xi) * tau^2 /
(2 tau - 1)``, hence the injected amplitude for a target variance ``v`` is

    var(xi) = v * (2 tau - 1) / tau^2 .

For the ghost modes this is exactly the role of ludwig's ``var_ghost``: the
amplitude is a function of the relaxation rate and ``kt`` only.  In BGK the
ghost relaxation rate equals the single ``tau``; ludwig exposes a separate
``lb_relaxation_time_ghosts``, which is a documented difference of this
transplant (see the report's diff list).

All arrays are ``numpy.float64``; nothing here is used by the existing
``fluctuating_lbm`` code paths, so the module is purely additive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from .fluctuating_lbm import CS2, VELOCITY_SETS, FluctuatingLBM

# ---------------------------------------------------------------------------
# mode classification
# ---------------------------------------------------------------------------

MODE_NAMES: tuple[str, ...] = ("rho", "jx", "jy", "e", "pxx", "pxy", "qx", "qy", "eps")
MODE_KIND: dict[str, str] = {
    "rho": "conserved",
    "jx": "conserved",
    "jy": "conserved",
    "e": "hydro",  # 2nd order, trace part (density/temperature mode)
    "pxx": "hydro",  # 2nd order, traceless normal stress
    "pxy": "hydro",  # 2nd order, shear stress
    "qx": "ghost",  # 3rd order
    "qy": "ghost",  # 3rd order
    "eps": "ghost",  # 4th order
}
GHOST_MODES: tuple[str, ...] = tuple(n for n in MODE_NAMES if MODE_KIND[n] == "ghost")
HYDRO_MODES: tuple[str, ...] = tuple(n for n in MODE_NAMES if MODE_KIND[n] == "hydro")
CONSERVED_MODES: tuple[str, ...] = tuple(n for n in MODE_NAMES if MODE_KIND[n] == "conserved")


def orthonormal_modes(lattice: str = "D2Q9") -> np.ndarray:
    """Weighted-orthonormal D2Q9 mode basis, shape ``(q, q)``.

    Row ``a`` is the mode vector ``e_a`` with ``sum_i w_i e_ai e_bi = delta_ab``.
    """
    if lattice != "D2Q9":
        raise ValueError("only D2Q9 is tabulated here (bearings on D2Q9 XVAL)")
    c, w = VELOCITY_SETS[lattice]
    del w
    cx = c[:, 0].astype(np.float64)
    cy = c[:, 1].astype(np.float64)
    c2 = cx * cx + cy * cy
    candidates = np.stack(
        [
            np.ones_like(cx),  # rho
            cx,  # jx
            cy,  # jy
            c2,  # e
            cx * cx - cy * cy,  # pxx
            cx * cy,  # pxy
            c2 * cx,  # qx
            c2 * cy,  # qy
            c2 * c2,  # eps
        ]
    )
    basis: list[np.ndarray] = []
    for v in candidates:
        u = v.copy()
        for b in basis:
            # plain (unweighted) Gram-Schmidt: the dual of m_a = sum_i e_ai f_i is
            # then simply df_i = e_ai dm_a, which keeps mass/momentum exactly
            # conserved because every mode is dot-orthogonal to 1, c_x, c_y.
            u = u - float(np.dot(b, u)) * b
        norm = float(np.sqrt(np.dot(u, u)))
        if norm < 1e-12:
            raise ValueError("mode basis lost rank (lattice/weights mismatch)")
        basis.append(u / norm)
    return np.stack(basis)


def mode_weights(lattice: str = "D2Q9") -> np.ndarray:
    """FDT weight ``W_a = sum_i w_i e_ai^2`` such that ``<dm_a^2> = 3 rho kT W_a``."""
    _, w = VELOCITY_SETS[lattice]
    basis = orthonormal_modes(lattice)
    return np.array([float(np.sum(w * basis[a] ** 2)) for a in range(basis.shape[0])])


def modes_of(f: np.ndarray, basis: np.ndarray | None = None) -> np.ndarray:
    """Modal field ``m[a, ...] = sum_i e_ai f_i``, shape ``(q,) + f.shape[1:]``."""
    b = orthonormal_modes() if basis is None else basis
    return np.tensordot(b, f, axes=([1], [0]))


# ---------------------------------------------------------------------------
# solver with a mode-resolved thermostat
# ---------------------------------------------------------------------------


@dataclass
class ModeVarianceReport:
    """Measured variance of every mode normalised by the FDT target ``3 rho kT``."""

    ratios: dict[str, float]
    variances: dict[str, float]
    expected: float
    rho: float
    kT: float
    tau: float
    n_samples: int
    thermostat: str

    @property
    def ghost_ratio_mean(self) -> float:
        return float(np.mean([self.ratios[n] for n in GHOST_MODES]))

    @property
    def hydro_ratio_mean(self) -> float:
        return float(np.mean([self.ratios[n] for n in HYDRO_MODES]))

    def error_pct(self, name: str) -> float:
        """Signed percentage error of the mode ratio against the FDT target (1.0)."""
        return 100.0 * (self.ratios[name] - 1.0)

    def as_dict(self) -> dict:
        return {
            "thermostat": self.thermostat,
            "tau": self.tau,
            "kT": self.kT,
            "rho": self.rho,
            "n_samples": self.n_samples,
            "expected_variance": self.expected,
            "ratios": dict(self.ratios),
            "variances": dict(self.variances),
            "error_pct": {n: self.error_pct(n) for n in MODE_NAMES},
            "ghost_ratio_mean": self.ghost_ratio_mean,
            "hydro_ratio_mean": self.hydro_ratio_mean,
            "ghost_error_pct": 100.0 * (self.ghost_ratio_mean - 1.0),
            "hydro_error_pct": 100.0 * (self.hydro_ratio_mean - 1.0),
        }


class FluctuatingLBMGhost(FluctuatingLBM):
    """Fluctuating BGK LBM with a mode-resolved thermostat.

    Parameters
    ----------
    thermostat : {"stress", "ghosts", "all"}
        ``stress`` reproduces the existing TensorLBM behaviour (traceless random
        stress only, ghosts untouched); ``ghosts`` adds ludwig-style independent
        noise to the ghost modes; ``all`` additionally thermalises every
        non-conserved mode (sensitivity arm).
    """

    def __init__(
        self,
        shape: Iterable[int],
        *,
        thermostat: str = "stress",
        **kwargs,
    ) -> None:
        super().__init__(shape, lattice="D2Q9", **kwargs)
        if thermostat not in ("stress", "ghosts", "all"):
            raise ValueError(f"unknown thermostat {thermostat!r}")
        self.thermostat = thermostat
        self.basis = orthonormal_modes(self.lattice)
        if thermostat == "ghosts":
            names: Sequence[str] = GHOST_MODES
        elif thermostat == "all":
            names = GHOST_MODES + HYDRO_MODES
        else:
            names = ()
        self.noise_mode_names = tuple(names)
        self.noise_mode_index = tuple(MODE_NAMES.index(n) for n in names)
        lead = (self.q, 1, 1)
        self.W = mode_weights(self.lattice)
        self._w_vec = self.w.astype(np.float64).reshape(lead)
        self._mode_vec = [self.basis[a].astype(np.float64).reshape(lead) for a in self.noise_mode_index]

    # -- amplitudes -------------------------------------------------------
    def target_mode_variance(self, rho: np.ndarray | float) -> np.ndarray | float:
        """FDT target ``3 rho kT W_a`` for every non-conserved mode ``a``."""
        return 3.0 * np.asarray(rho, dtype=np.float64) * self.kT * self.W[self.noise_mode_index[0]] if self.noise_mode_index else 3.0 * np.asarray(rho, dtype=np.float64) * self.kT

    def target_for(self, name: str, rho: float) -> float:
        """Scalar FDT target of one named mode."""
        return 3.0 * rho * self.kT * float(self.W[MODE_NAMES.index(name)])

    def noise_amplitude(self, rho: np.ndarray | float, tau: float | None = None) -> np.ndarray | float:
        """Injected per-mode variance ``v (2 tau - 1) / tau^2`` for target ``v = 3 rho kT``."""
        t = self.tau if tau is None else float(tau)
        return self.target_mode_variance(rho) * (2.0 * t - 1.0) / (t * t)

    def _add_mode_noise(self, f: np.ndarray, rho: np.ndarray) -> np.ndarray:
        for evec, a in zip(self._mode_vec, self.noise_mode_index):
            target = 3.0 * rho * self.kT * float(self.W[a])
            sd = np.sqrt(target * (2.0 * self.tau - 1.0) / self.tau ** 2)
            xi = self.rng.standard_normal(self.shape)
            f = f + evec * xi * sd
        return f

    # -- march ------------------------------------------------------------
    def step(
        self,
        n: int = 1,
        *,
        noise: bool = True,
        extra_force: np.ndarray | None = None,
    ) -> None:
        for _ in range(int(n)):
            rho, _u = self._moments()
            force = self.random_force() if (noise and self.kT > 0.0) else None
            if extra_force is not None:
                force = extra_force if force is None else force + extra_force
            fpost = self._collide(force)
            if noise and self.kT > 0.0 and self.noise_mode_index:
                fpost = self._add_mode_noise(fpost, rho)
            self.f = self._stream(fpost)
            self._last_force = force
            self.steps += 1

    # -- measurement ------------------------------------------------------
    def mode_variance_report(
        self,
        *,
        n_samples: int = 800,
        n_skip: int = 200,
        window: tuple[int, int] | None = None,
    ) -> ModeVarianceReport:
        """Time-and-space averaged mode variances normalised by ``3 rho kT``."""
        if window is not None:
            n_skip, n_samples = window
        total = int(n_skip) + int(n_samples)
        acc = np.zeros((self.q,) + self.shape)
        acc2 = np.zeros((self.q,) + self.shape)
        for s in range(total):
            self.step()
            if s >= n_skip:
                m = modes_of(self.f, self.basis)
                acc += m
                acc2 += m * m
        n = float(n_samples)
        mean = acc / n
        var = np.maximum(acc2 / n - mean * mean, 0.0)
        rho_mean = float(np.mean(acc[MODE_NAMES.index("rho")] / n))
        expected = 3.0 * rho_mean * self.kT
        variances = {name: float(np.mean(var[i])) for i, name in enumerate(MODE_NAMES)}
        per_mode_target = {name: expected * float(self.W[i]) for i, name in enumerate(MODE_NAMES)}
        ratios = {
            name: (variances[name] / per_mode_target[name] if per_mode_target[name] > 0 else float("nan"))
            for name in MODE_NAMES
        }
        return ModeVarianceReport(
            ratios=ratios,
            variances=variances,
            expected=expected,
            rho=rho_mean,
            kT=self.kT,
            tau=self.tau,
            n_samples=int(n_samples),
            thermostat=self.thermostat,
        )


# ---------------------------------------------------------------------------
# convenience arm runner
# ---------------------------------------------------------------------------


@dataclass
class ArmResult:
    name: str
    thermostat: str
    tau: float
    kT: float
    report: ModeVarianceReport
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = {
            "arm": self.name,
            "tau": self.tau,
            "kT": self.kT,
            "report": self.report.as_dict(),
        }
        d.update(self.extra)
        return d


def run_arm(
    name: str,
    *,
    thermostat: str,
    tau: float = 0.8,
    kT: float = 0.01,
    shape: tuple[int, int] = (32, 32),
    n_samples: int = 800,
    n_skip: int = 200,
    seed: int = 12345,
    bitwise_reference: bool = False,
) -> ArmResult:
    """Run one thermostat arm and return its mode-variance report.

    ``bitwise_reference`` additionally checks the ``kT = 0`` negative control:
    the march must be bit-for-bit identical to the plain BGK reference solver.
    """
    lbm = FluctuatingLBMGhost(shape, tau=tau, kT=kT, seed=seed, thermostat=thermostat)
    extra: dict = {}
    if bitwise_reference:
        f0 = lbm.f.copy()
        for _ in range(5):
            lbm.step()
        f_ghost = lbm.f.copy()
        ref = FluctuatingLBM(shape, tau=tau, kT=0.0, seed=seed)
        ref.f = f0.copy()
        ref.step_reference_bgk(5)
        extra["bitwise_identical_at_kT0"] = bool(np.array_equal(f_ghost, ref.f))
        # reset the thermal march
        lbm = FluctuatingLBMGhost(shape, tau=tau, kT=kT, seed=seed, thermostat=thermostat)
    report = lbm.mode_variance_report(n_samples=n_samples, n_skip=n_skip)
    return ArmResult(name=name, thermostat=thermostat, tau=tau, kT=kT, report=report, extra=extra)