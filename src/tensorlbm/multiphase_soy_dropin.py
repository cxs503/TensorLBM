"""SOY free-energy D2Q9 LBM with drop-in array backends (NumPy **and** PyTorch).

Why this module exists
----------------------
Cross-validation (XVAL) against the open-source implementation
``ebinan92/Fingering_dynamics`` (``lattice_boltzmann/fingering_periodic.py`` and the
GPU sibling ``fingering_periodic_gpu.py``).  That project's GPU strategy is the
"**cupy drop-in, no kernel fusion**" recipe: ``fingering_periodic_gpu.py`` is a
line-for-line copy of the NumPy script with ``import numpy as np`` replaced by
``import cupy as cp`` -- no custom CUDA kernels, no fused push/pull, just the
array-API swap.

This module ports that *practice* to TensorLBM: one algorithm implementation
written against a thin array-namespace shim, executable with two backends

* ``backend="numpy"``  -- CPU reference / drop-in for the OSS NumPy script,
* ``backend="torch"``  -- CPU or CUDA tensors (``torch.Tensor`` is itself a
  drop-in array API: same ops, no hand-written kernels).

The physics is the Swift-Osborn-Yeomans (SOY) free-energy binary-fluid model used
by the OSS project:

* two D2Q9 populations: ``f`` (momentum / total density) and ``g`` (order
  parameter ``psi in [-1, +1]``, ``+1`` = wetting/injected fluid);
* chemical potential ``mu = a*psi*(1 - psi^2) - kappa*lap(psi)`` with
  ``kappa = 0.75*sigma*xi`` and ``a = -2*kappa/xi^2``;
* free-energy pressure.  The OSS code uses ``p = rho/3 + psi*mu``
  (``fingering_periodic.py:123-124``, ``getP``), which is *not* the naive
  ``rho/3 + psi*mu/3`` sometimes quoted for SOY.  ``pressure_form`` selects
  between them so the difference can be quantified instead of asserted;
* Guo "large force" source with the half-force correction
  ``(1 - 1/(2*tau_mix))`` acting on the capillary force ``mu*grad(psi)``;
* equilibrium populations ``feq`` / ``geq`` with the OSS ``A0/A1_8`` and
  ``B0/B1_8`` groupings;
* isotropic D2Q9 stencils for ``grad(psi)`` and ``lap(psi)``;
* non-Newtonian-style variable relaxation time ``tau_mix`` from the OSS
  harmonic two-viscosity mixing law (with ``M = 1`` it degenerates to the
  Newtonian ``tau_f``).

Deliberate scope limits (documented, not hidden)
-----------------------------------------------
* Periodic domains only.  The OSS project couples the same scheme to Zou-He
  inlet/outlet and half-way bounce-back walls; those boundary conditions are
  not ported here (the XVAL cases are static droplets in a periodic box).
* ``torch`` backend is eager-mode; no kernel fusion, no ``torch.compile``,
  no custom CUDA, exactly the "no kernel fusion" premise being cross-validated.

References
----------
* OSS source (read-only): ``ebinan92/Fingering_dynamics`` @ ``150a573``,
  ``lattice_boltzmann/fingering_periodic.py`` / ``fingering_periodic_gpu.py``.
* Laplace law validation: ``Delta p = sigma / R`` for a static droplet.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# D2Q9 lattice (index order identical to the OSS scripts)
# ---------------------------------------------------------------------------
E = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
              [1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=np.int64)
W = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])

# Streaming shifts used by the OSS ``stream(f, g)`` helper:
# direction i is advected by (+e_y, +e_x) on (axis0, axis1).
STREAM_SHIFT = [(int(E[i][1]), int(E[i][0])) for i in range(9)]


# ---------------------------------------------------------------------------
# Array-namespace shim ("drop-in" layer).  Only ops that NumPy/CuPy/PyTorch all
# provide are used, so the same step() body runs on every backend.
# ---------------------------------------------------------------------------
class NumpyBackend:
    """CPU array namespace backed by NumPy (double precision by default)."""

    name = "numpy"

    def __init__(self, dtype: str = "float64") -> None:
        self.dtype = np.dtype(dtype)
        self.ops = 0

    # -- construction --
    def zeros(self, shape):
        self.ops += 1
        return np.zeros(shape, dtype=self.dtype)

    def const(self, value, shape):
        self.ops += 1
        return np.full(shape, value, dtype=self.dtype)

    def from_numpy(self, a):
        self.ops += 1
        return np.array(a, dtype=self.dtype, copy=True)

    def to_numpy(self, a):
        return np.asarray(a, dtype=np.float64)

    # -- elementary ops --
    def roll(self, a, shift, axis):
        self.ops += 1
        return np.roll(a, shift, axis=axis)

    def stack(self, arrays):
        self.ops += 1
        return np.stack(tuple(arrays))

    def sum_pop(self, arrays):
        self.ops += 1
        return np.sum(np.stack(tuple(arrays)), axis=0)

    def power(self, a, p):
        self.ops += 1
        return np.power(a, p)

    def maximum(self, a, b):
        self.ops += 1
        return np.maximum(a, b)

    def mean(self, a):
        self.ops += 1
        return float(np.mean(a))

    def maxabs(self, a):
        self.ops += 1
        return float(np.max(np.abs(a)))


class TorchBackend:
    """PyTorch array namespace: eager tensor ops, no custom/fused kernels."""

    name = "torch"

    def __init__(self, dtype: str = "float64", device: str = "cpu") -> None:
        import torch  # local import: NumPy-only users pay no cost

        self.torch = torch
        self.dtype = getattr(torch, dtype)
        self.device = torch.device(device)
        self.ops = 0

    def zeros(self, shape):
        self.ops += 1
        return self.torch.zeros(shape, dtype=self.dtype, device=self.device)

    def const(self, value, shape):
        self.ops += 1
        return self.torch.full(tuple(shape), float(value), dtype=self.dtype, device=self.device)

    def from_numpy(self, a):
        self.ops += 1
        return self.torch.as_tensor(np.asarray(a), dtype=self.dtype, device=self.device).clone()

    def to_numpy(self, a):
        return a.detach().to("cpu").numpy().astype(np.float64, copy=False)

    def roll(self, a, shift, axis):
        self.ops += 1
        return self.torch.roll(a, shifts=int(shift), dims=int(axis))

    def stack(self, arrays):
        self.ops += 1
        return self.torch.stack(tuple(arrays))

    def sum_pop(self, arrays):
        self.ops += 1
        return self.torch.stack(tuple(arrays)).sum(dim=0)

    def power(self, a, p):
        self.ops += 1
        return a ** p

    def maximum(self, a, b):
        self.ops += 1
        return self.torch.maximum(a, b)

    def mean(self, a):
        self.ops += 1
        return float(a.mean().item())

    def maxabs(self, a):
        self.ops += 1
        return float(a.abs().max().item())


def make_backend(name: str, dtype: str = "float64", device: str = "cpu"):
    if name == "numpy":
        return NumpyBackend(dtype=dtype)
    if name == "torch":
        return TorchBackend(dtype=dtype, device=device)
    raise ValueError(f"unknown backend {name!r} (expected 'numpy' or 'torch')")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class SOYConfig:
    """Parameters of one SOY free-energy run (lattice units)."""

    nx: int = 60
    ny: int = 60
    radius: float = 15.0            # initial droplet radius (lattice units)
    init_profile: str = "equilibrium"  # "equilibrium" (tanh(r/xi)) or "sharp" (tanh(2r/xi))
    sigma: float = 0.01             # target surface tension
    xi: float = 3.0                 # interface width
    rho0: float = 1.0               # bulk density (both phases: matched density)
    tau_g: float = 1.0 / (3.0 - math.sqrt(3.0))   # OSS value 0.7887
    gamma: float = 0.1              # mobility (OSS derives it from u0/Pe; static case -> fixed)
    eta_n: float = 1.0 / 6.0        # base kinematic viscosity -> Newtonian tau_f = 1.0
    viscosity_ratio: float = 1.0    # OSS ``M``: tau_mix mixing ratio (M=1 -> Newtonian)
    pressure_form: str = "oss"      # "oss" (rho/3 + psi*mu) or "psi_mu_over_3"
    backend: str = "numpy"
    dtype: str = "float64"
    device: str = "cpu"

    # derived free-energy constants (identical grouping to the OSS parameter block)
    @property
    def kappa(self) -> float:
        return 0.75 * self.sigma * self.xi

    @property
    def a(self) -> float:
        return -2.0 * self.kappa / (self.xi ** 2)

    def as_dict(self) -> dict[str, Any]:
        return {
            "nx": self.nx, "ny": self.ny, "radius": self.radius,
            "init_profile": self.init_profile, "sigma": self.sigma,
            "xi": self.xi, "kappa": self.kappa, "a": self.a, "rho0": self.rho0,
            "tau_g": self.tau_g, "gamma": self.gamma, "eta_n": self.eta_n,
            "viscosity_ratio": self.viscosity_ratio, "pressure_form": self.pressure_form,
            "backend": self.backend, "dtype": self.dtype, "device": self.device,
        }


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
class SOYFreeEnergyD2Q9:
    """Periodic-domain SOY free-energy D2Q9 solver with pluggable array backend.

    The update order reproduces the OSS driver loop
    (``fingering_periodic.py:436-460``):

    1. Guo source + ``feq``/``geq`` from the *old* state, then collide ``f``
       (variable ``tau_mix``) and ``g`` (fixed ``tau_g``);
    2. roll-based streaming over the 9 populations;
    3. macroscopic recovery: ``rho = sum f``, ``psi = sum g``;
    4. isotropic ``grad psi`` / ``lap psi`` -> ``mu`` -> ``u`` -> ``p`` -> ``tau_mix``.
    """

    def __init__(self, config: SOYConfig | None = None, **overrides: Any) -> None:
        cfg = config or SOYConfig()
        if overrides:
            cfg = SOYConfig(**{**cfg.as_dict(), **overrides})
        self.cfg = cfg
        self.b = make_backend(cfg.backend, dtype=cfg.dtype, device=cfg.device)
        b = self.b
        self.shape = (cfg.ny, cfg.nx)

        self.psi = b.const(-1.0, self.shape)
        self.rho = b.const(cfg.rho0, self.shape)
        self.ux = b.zeros(self.shape)
        self.uy = b.zeros(self.shape)
        self.p = b.zeros(self.shape)
        self.mu = b.zeros(self.shape)
        self.mix_tau = b.const(self.tau_f, self.shape)
        self.grad_x = b.zeros(self.shape)
        self.grad_y = b.zeros(self.shape)
        self.lap = b.zeros(self.shape)

        self.f = [b.zeros(self.shape) for _ in range(9)]
        self.g = [b.zeros(self.shape) for _ in range(9)]

        self.steps = 0
        self.wall_time = 0.0
        self.ops_at_init = b.ops

    # -- derived scalars -------------------------------------------------
    @property
    def tau_f(self) -> float:
        """Newtonian momentum relaxation time implied by ``eta_n``."""
        return 3.0 * self.cfg.eta_n + 0.5

    # -- initialisation ---------------------------------------------------
    def init_droplet(self, radius: float | None = None) -> None:
        """tanh droplet in a periodic box, then equilibrium populations."""
        cfg, b = self.cfg, self.b
        R = float(cfg.radius if radius is None else radius)
        yy, xx = np.meshgrid(np.arange(cfg.ny), np.arange(cfg.nx), indexing="ij")
        cx, cy = cfg.nx / 2.0, cfg.ny / 2.0
        r = np.hypot(xx - cx, yy - cy)
        # Analytic SOY equilibrium profile: psi = tanh(n/xi) solves
        # kappa*psi'' = a*psi*(1-psi^2) with a = -2*kappa/xi^2, so the width
        # parameter is xi itself.  "sharp" initialises twice as steep and is kept
        # only for the initial-condition sensitivity study (it relaxes very slowly).
        width = cfg.xi if cfg.init_profile == "equilibrium" else cfg.xi / 2.0
        psi = -np.tanh((r - R) / width)                # +1 inside, -1 outside
        self.psi = b.from_numpy(psi)
        self.rho = b.const(cfg.rho0, self.shape)
        self.ux = b.zeros(self.shape)
        self.uy = b.zeros(self.shape)
        self._gradients()
        self.mu = self._chemical_potential()
        self.p = self._pressure()
        self.mix_tau = self._mix_tau()
        for i in range(9):
            self.f[i] = self._feq(i)
            self.g[i] = self._geq(i)
        self.steps = 0

    # -- stencils ---------------------------------------------------------
    def _grad(self, field, axis: int):
        b = self.b
        if axis == 1:   # d/dx
            return (4.0 * (b.roll(field, -1, 1) - b.roll(field, 1, 1))
                    + b.roll(b.roll(field, -1, 1), -1, 0)
                    - b.roll(b.roll(field, 1, 1), -1, 0)
                    + b.roll(b.roll(field, -1, 1), 1, 0)
                    - b.roll(b.roll(field, 1, 1), 1, 0)) / 12.0
        return (4.0 * (b.roll(field, -1, 0) - b.roll(field, 1, 0))
                + b.roll(b.roll(field, -1, 1), -1, 0)
                + b.roll(b.roll(field, 1, 1), -1, 0)
                - b.roll(b.roll(field, 1, 1), 1, 0)
                - b.roll(b.roll(field, -1, 1), 1, 0)) / 12.0

    def _laplacian(self, field):
        b = self.b
        return (-20.0 * field
                + 4.0 * (b.roll(field, -1, 0) + b.roll(field, -1, 1)
                         + b.roll(field, 1, 1) + b.roll(field, 1, 0))
                + b.roll(b.roll(field, -1, 1), -1, 0)
                + b.roll(b.roll(field, 1, 1), -1, 0)
                + b.roll(b.roll(field, 1, 1), 1, 0)
                + b.roll(b.roll(field, -1, 1), 1, 0)) / 6.0

    def _gradients(self) -> None:
        self.grad_x = self._grad(self.psi, 1)
        self.grad_y = self._grad(self.psi, 0)
        self.lap = self._laplacian(self.psi)

    # -- constitutive -----------------------------------------------------
    def _chemical_potential(self):
        cfg = self.cfg
        return (cfg.a * self.psi * (1.0 - self.psi * self.psi)
                - cfg.kappa * self.lap)

    def _pressure(self):
        cfg = self.cfg
        if cfg.pressure_form == "oss":
            return self.rho / 3.0 + self.psi * self.mu
        if cfg.pressure_form == "psi_mu_over_3":
            return self.rho / 3.0 + self.psi * self.mu / 3.0
        raise ValueError(f"unknown pressure_form {cfg.pressure_form!r}")

    def _mix_tau(self):
        """OSS harmonic mixing of two viscosities (``getMix_tau``)."""
        cfg = self.cfg
        v1 = cfg.eta_n / self.rho
        v2 = cfg.eta_n * cfg.viscosity_ratio / self.rho
        mix_v = 2.0 * v1 * v2 / (v1 * (1.0 - self.psi) + v2 * (1.0 + self.psi))
        return 3.0 * mix_v + 0.5

    # -- equilibria (OSS groupings A0/A1_8, B0/B1_8) ----------------------
    def _feq(self, i: int):
        b, cfg = self.b, self.cfg
        eu = float(E[i][0]) * self.ux + float(E[i][1]) * self.uy
        u2 = self.ux * self.ux + self.uy * self.uy
        base = 3.0 * eu + 4.5 * eu * eu - 1.5 * u2
        if i == 0:
            a0 = (self.rho - 3.0 * (1.0 - W[0]) * self.p) / W[0]
            out = a0 + self.rho * base
        else:
            out = 3.0 * self.p + self.rho * base
        return out * W[i]

    def _geq(self, i: int):
        cfg = self.cfg
        eu = float(E[i][0]) * self.ux + float(E[i][1]) * self.uy
        u2 = self.ux * self.ux + self.uy * self.uy
        base = 3.0 * eu + 4.5 * eu * eu - 1.5 * u2
        if i == 0:
            b0 = (self.psi - 3.0 * (1.0 - W[0]) * cfg.gamma * self.mu) / W[0]
            out = b0 + self.psi * base
        else:
            out = 3.0 * cfg.gamma * self.mu + self.psi * base
        return out * W[i]

    def _guo_source(self, i: int):
        """OSS ``getLarge_F``: Guo source for the capillary force mu*grad(psi)."""
        eu = float(E[i][0]) * self.ux + float(E[i][1]) * self.uy
        cx, cy = float(E[i][0]), float(E[i][1])
        factor = self.mu * W[i] * (1.0 - 1.0 / (2.0 * self.mix_tau))
        return factor * ((cx - self.ux) * 3.0 + cx * eu * 9.0) * self.grad_x + factor * (
            (cy - self.uy) * 3.0 + cy * eu * 9.0) * self.grad_y

    # -- one time step ----------------------------------------------------
    def step(self) -> None:
        b, cfg = self.b, self.cfg
        t0 = time.perf_counter()
        feq = [self._feq(i) for i in range(9)]
        geq = [self._geq(i) for i in range(9)]
        src = [self._guo_source(i) for i in range(9)]
        f_new, g_new = [], []
        for i in range(9):
            f_new.append(self.f[i] - (self.f[i] - feq[i]) / self.mix_tau + src[i])
            g_new.append(self.g[i] - (self.g[i] - geq[i]) / cfg.tau_g)
        # stream (same per-direction shifts as the OSS ``stream`` helper)
        self.f = [b.roll(f_new[i], STREAM_SHIFT[i][0], 0) for i in range(9)]
        self.g = [b.roll(g_new[i], STREAM_SHIFT[i][0], 0) for i in range(9)]
        for i in range(9):
            sy, sx = STREAM_SHIFT[i]
            if sx:
                self.f[i] = b.roll(self.f[i], sx, 1)
                self.g[i] = b.roll(self.g[i], sx, 1)
        # macroscopic recovery
        self.rho = b.sum_pop(self.f)
        self.psi = b.sum_pop(self.g)
        self._gradients()
        self.mu = self._chemical_potential()
        jx = b.sum_pop([self.f[i] * float(E[i][0]) for i in range(9)])
        jy = b.sum_pop([self.f[i] * float(E[i][1]) for i in range(9)])
        self.ux = (jx + self.mu * self.grad_x / 2.0) / self.rho
        self.uy = (jy + self.mu * self.grad_y / 2.0) / self.rho
        self.p = self._pressure()
        self.mix_tau = self._mix_tau()
        self.steps += 1
        self.wall_time += time.perf_counter() - t0

    def run(self, n_steps: int) -> None:
        for _ in range(n_steps):
            self.step()

    # -- diagnostics ------------------------------------------------------
    def measure(self, radius: float | None = None) -> dict[str, float]:
        """Laplace pressure jump and residual flow metrics (NumPy, read-only)."""
        cfg = self.cfg
        R = float(cfg.radius if radius is None else radius)
        psi = self.b.to_numpy(self.psi)
        p = self.b.to_numpy(self.p)
        ux = self.b.to_numpy(self.ux)
        uy = self.b.to_numpy(self.uy)
        gx = self.b.to_numpy(self.grad_x)
        gy = self.b.to_numpy(self.grad_y)
        yy, xx = np.meshgrid(np.arange(cfg.ny), np.arange(cfg.nx), indexing="ij")
        r = np.hypot(xx - cfg.nx / 2.0, yy - cfg.ny / 2.0)
        inside = r < R - cfg.xi
        outside = r > R + cfg.xi
        dp = float(p[inside].mean() - p[outside].mean())
        sigma_cap = float(np.sum(cfg.kappa * (gx * gx + gy * gy)) / (2.0 * math.pi * R))
        return {
            "dp": dp,
            "dp_analytical": cfg.sigma / R,
            "p_inside": float(p[inside].mean()),
            "p_outside": float(p[outside].mean()),
            "u_max": float(np.max(np.hypot(ux, uy))),
            "psi_mean": float(psi.mean()),
            "psi_sum": float(psi.sum()),
            "psi_abs_sum": float(np.abs(psi).sum()),
            "rho_mean": float(self.b.to_numpy(self.rho).mean()),
            "sigma_from_capillary_energy": sigma_cap,
            "sigma_rel_err_pct": 100.0 * (sigma_cap - cfg.sigma) / cfg.sigma,
            "steps": self.steps,
            "backend": cfg.backend,
            "pressure_form": cfg.pressure_form,
        }

    def ops_per_step(self) -> float:
        if self.steps == 0:
            return 0.0
        return (self.b.ops - self.ops_at_init) / float(self.steps)


def laplace_error_pct(dp: float, sigma: float, radius: float) -> float:
    """Signed relative error of a measured pressure jump vs ``sigma/R`` (percent)."""
    ref = sigma / radius
    return 100.0 * (dp - ref) / ref