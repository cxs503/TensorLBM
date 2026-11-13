"""Adapter that *executes* the OSS reference implementation (no source vendoring).

Cross-validation needs the reference arm to be the OSS project's own code, not a
re-implementation of it.  ``ebinan92/Fingering_dynamics`` ships no licence
(all-rights-reserved, see the TensorPlatform OSS inventory for that repo), so its
source is **not** copied into TensorLBM.  Instead this module reads the file from
a read-only checkout at run time, extracts the two top-level definitions that
carry the algorithm -- ``class Compute`` and ``def stream`` -- and ``exec``s them
inside a parameter namespace that we provide.

That gives a genuine external reference arm:

* the discrete operators, Guo source, equilibria and collision are exactly the
  OSS text (unmodified), executed on NumPy;
* only the *driver* is ours: a periodic static droplet instead of the OSS
  channel-with-circles setup, so that the Laplace law ``dp = sigma/R`` has a
  closed-form reference;
* the OSS ``getP`` (``p = rho/3 + psi*mu``) is used verbatim for the pressure
  jump, which is also what :class:`tensorlbm.multiphase_soy_dropin` implements.

If no checkout is available the arm reports ``available=False`` with a reason
(honest NOT_RUN, never a fabricated number).

Environment: ``OSS_FINGERING_ROOT`` overrides the default checkout path.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_ROOT = os.environ.get("OSS_FINGERING_ROOT", "/tmp/Fingering_dynamics")
OSS_FILE = "lattice_boltzmann/fingering_periodic.py"
OSS_COMMIT = "150a573"


def _extract_defs(path: Path) -> str:
    """Return the source slice holding ``class Compute`` + ``def stream``."""
    text = path.read_text(encoding="utf-8", errors="replace")
    start = text.index("class Compute:")
    stream_at = text.index("\ndef stream(")
    # ``Compute`` ends where its last method ends: everything up to the module
    # level helper that follows the class body.
    end = text.index("\ndef ", stream_at + 1)
    return text[start:stream_at] + "\n" + text[stream_at:end]


class OSSReference:
    """Drive the OSS NumPy implementation on a periodic static droplet."""

    def __init__(self, *, nx: int = 60, ny: int = 60, radius: float = 15.0,
                 sigma: float = 0.01, xi: float = 3.0, rho0: float = 1.0,
                 eta_n: float = 1.0 / 6.0, viscosity_ratio: float = 1.0,
                 tau_g: float = 1.0 / (3.0 - math.sqrt(3.0)),
                 gamma: float = 0.1, root: str | None = None) -> None:
        self.root = Path(root or DEFAULT_ROOT)
        self.path = self.root / OSS_FILE
        self.nx, self.ny, self.radius = nx, ny, radius
        self.sigma, self.xi, self.rho0 = sigma, xi, rho0
        self.eta_n, self.M = eta_n, viscosity_ratio
        self.tau, self.gamma = tau_g, gamma
        self.kappa = 0.75 * sigma * xi
        self.a = -2.0 * self.kappa / (xi ** 2)
        self.wall_time = 0.0
        self.steps = 0
        self._available = self.path.is_file()
        self._reason = "" if self._available else f"OSS checkout missing: {self.path}"

    # -- introspection ----------------------------------------------------
    @property
    def available(self) -> bool:
        return self._available

    @property
    def reason(self) -> str:
        return self._reason

    def provenance(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "oss_repo": "ebinan92/Fingering_dynamics",
            "oss_commit": OSS_COMMIT,
            "oss_file": str(self.path),
            "available": self._available,
            "reason": self._reason,
            "executed": False,
        }
        if self._available:
            import hashlib
            src = self.path.read_bytes()
            info["oss_sha256"] = hashlib.sha256(src).hexdigest()
            info["oss_bytes"] = len(src)
        return info

    # -- setup ------------------------------------------------------------
    def _namespace(self) -> dict[str, Any]:
        ns: dict[str, Any] = {
            "np": np, "copy": __import__("copy"),
            "H": self.ny, "W": self.nx,
            "a": self.a, "kappa": self.kappa, "gamma": self.gamma,
            "tau": self.tau, "Eta_n": self.eta_n, "M": self.M,
            "rho0": self.rho0, "psi_wall": -1.0,
        }
        src = _extract_defs(self.path)
        exec(compile(src, str(self.path), "exec"), ns)  # noqa: S102 - executed reference arm
        return ns

    def setup(self) -> None:
        ns = self._namespace()
        mask = np.ones((self.ny, self.nx), dtype=bool)
        cm = ns["Compute"](mask)
        yy, xx = np.meshgrid(np.arange(self.ny), np.arange(self.nx), indexing="ij")
        r = np.hypot(xx - self.nx / 2.0, yy - self.ny / 2.0)
        cm.psi = -np.tanh((r - self.radius) / self.xi)
        # OSS hard-codes ghost columns for its Zou-He inlet/outlet
        # (``self.left_wall = +1``, ``self.right_wall = -1``).  We do not use
        # those boundaries, so the ghost columns are set to the far-field value
        # (-1) instead: otherwise a spurious +1/-1 "interface" is forced at
        # x = 0 / x = N-1 every step and the run diverges.
        cm.left_wall = np.full((self.ny, 1), -1.0)
        cm.right_wall = np.full((self.ny, 1), -1.0)
        cm.rho = cm.rho * 0.0 + self.rho0
        cm.ux = np.zeros_like(cm.rho)
        cm.uy = np.zeros_like(cm.rho)
        cm.nabla_psix = cm.getNabla_psix()
        cm.nabla_psiy = cm.getNabla_psiy()
        cm.nabla_psi2 = cm.getNabla_psi2()
        cm.mu = cm.getMu()
        cm.ux = cm.getUx()
        cm.uy = cm.getUy()
        cm.p = cm.getP()
        cm.mix_tau = cm.getMix_tau()
        for i in range(9):
            cm.f[i][cm.mask] = cm.getfeq(i)
            cm.g[i][cm.mask] = cm.getgeq(i)
        self.cm = cm
        self.mask = mask
        self._stream = ns["stream"]

    # -- time loop (mirrors fingering_periodic.py:436-460) -----------------
    def run(self, n_steps: int) -> None:
        cm, mask, stream = self.cm, self.mask, self._stream
        t0 = time.perf_counter()
        for _ in range(int(n_steps)):
            for j in range(9):
                cm.F[j] = cm.getLarge_F(j)
                cm.feq[j] = cm.getfeq(j)
                cm.geq[j] = cm.getgeq(j)
                cm.f[j][mask] = cm.getF(j)
                cm.g[j][mask] = cm.getG(j)
            stream(cm.f, cm.g)
            cm.rho = cm.getRho()
            cm.udpatePsi()
            cm.nabla_psix = cm.getNabla_psix()
            cm.nabla_psiy = cm.getNabla_psiy()
            cm.nabla_psi2 = cm.getNabla_psi2()
            cm.mu = cm.getMu()
            cm.ux = cm.getUx()
            cm.uy = cm.getUy()
            cm.p = cm.getP()
            cm.mix_tau = cm.getMix_tau()
        self.steps += int(n_steps)
        self.wall_time += time.perf_counter() - t0

    # -- diagnostics ------------------------------------------------------
    def measure(self) -> dict[str, float]:
        cm = self.cm
        p = np.asarray(cm.p).reshape(self.ny, self.nx)
        yy, xx = np.meshgrid(np.arange(self.ny), np.arange(self.nx), indexing="ij")
        r = np.hypot(xx - self.nx / 2.0, yy - self.ny / 2.0)
        inside = r < self.radius - self.xi
        outside = r > self.radius + self.xi
        psi = np.asarray(cm.psi)
        return {
            "dp": float(p[inside].mean() - p[outside].mean()),
            "dp_analytical": self.sigma / self.radius,
            "p_inside": float(p[inside].mean()),
            "p_outside": float(p[outside].mean()),
            "u_max": float(np.max(np.hypot(np.asarray(cm.ux), np.asarray(cm.uy)))),
            "psi_mean": float(psi.mean()),
            "psi_abs_sum": float(np.abs(psi).sum()),
            "rho_mean": float(np.asarray(cm.rho).mean()),
            "steps": self.steps,
            "pressure_form": "oss (p = rho/3 + psi*mu, OSS getP)",
        }

    def run_and_measure(self, n_steps: int, probe_every: int = 0) -> dict[str, Any]:
        out: dict[str, Any] = {"history": []}
        remaining = int(n_steps)
        while remaining > 0:
            chunk = min(probe_every, remaining) if probe_every else remaining
            self.run(chunk)
            remaining -= chunk
            if probe_every:
                m = self.measure()
                out["history"].append({"step": self.steps, "dp": m["dp"],
                                       "u_max": m["u_max"]})
        out.update(self.measure())
        out["wall_time_s"] = self.wall_time
        return out