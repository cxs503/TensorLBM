#!/usr/bin/env python3
"""Faithful torch port of the upstream LES closure (arm B).

Upstream source (machine-extracted into ``oss_ref/``; see
``extract_oss_tables.py`` for the locators and sha256):

``LatFlow/Domain.py:90-96``  (``Domain.__init__``) -- the ``|c_n c_m|`` kernel::

    self.EEk = tf.zeros(self.Dim*[1] + [self.Nneigh])
    for n in xrange(3):
      for m in xrange(3):
        if self.Dim == 2:
          self.EEk = self.EEk + tf.abs(self.C[:,:,:,n] * self.C[:,:,:,m])

``LatFlow/Domain.py:118-128`` (``Domain.CollideSC``) -- the collision, with the
``les`` branch::

    NonEq = f - Feq
    if self.les:
      Q = tf.expand_dims(tf.reduce_sum(NonEq*NonEq*self.EEk, axis=self.Dim+1), axis=self.Dim+1)
      Q = tf.sqrt(2.0*Q)
      tau = 0.5*(self.tau[0]+tf.sqrt(self.tau[0]*self.tau[0] + 6.0*Q*self.Sc/rho))
    else:
      tau = self.tau[0]
    f = f - NonEq/tau

``LatFlow/Domain.py:57`` -- ``self.Sc = 0.17``.

Porting decisions (each one is a *deviation* and is listed in the report):

* the upstream ``if not graph_unroll`` branch and the solid-mask bounce
  convolution are dropped: every configuration here is fully periodic fluid,
  and the benchmark of the transport/wall handling already exists
  (XVAL-LBM-CONV-AS-STREAMING-FIXED-KERNEL).  The closure algebra is untouched.
* upstream reads ``self.Rho`` / ``self.Vel`` (stored by the previous stream)
  and divides Q by that rho; the port recomputes rho from ``f`` at the
  collision instant.  Inside upstream's own ``Solve`` loop (stream -> collide)
  the stored value *is* the instantaneous one, so this is not a change of
  operator.
* upstream has no clamp on ``tau``; the port keeps it that way (the in-house
  closure clamps, see ``turbulence.turbulence._tau_eff_max``).  The difference
  is reported, not silently harmonised.
* float64 instead of float32, torch instead of TensorFlow: formulation-level
  comparison only.

Self-check: ``collide_oss(..., Sc=0.0)`` must reproduce
``tensorlbm.solver.collide_bgk`` to round-off (predeclared ``port_fidelity_rel``).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import torch

REF_DIR = Path(__file__).resolve().parent / "oss_ref"


@lru_cache(maxsize=1)
def lattice() -> dict:
    """The machine-extracted upstream D2Q9 table (weights, velocities, EEk, Sc)."""
    path = REF_DIR / "oss_lattice.json"
    if not path.is_file():
        raise RuntimeError(
            f"{path} missing -- run extract_oss_tables.py first (it needs a checkout "
            "of loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow)"
        )
    return json.loads(path.read_text())


@lru_cache(maxsize=1)
def provenance() -> dict:
    path = REF_DIR / "provenance.json"
    if not path.is_file():
        raise RuntimeError(f"{path} missing -- run extract_oss_tables.py first")
    return json.loads(path.read_text())


def _tables(dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lat = lattice()
    w = torch.tensor(lat["weights"], dtype=dtype, device=device)
    c = torch.tensor(lat["velocities_2d"], dtype=dtype, device=device)  # (9, 2)
    eek = torch.tensor(lat["eek_abs_dyadic"], dtype=dtype, device=device)
    return w, c, eek


def moments(f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """rho, ux, uy from the population tensor ``(9, ny, nx)`` using upstream's table."""
    w, c, _ = _tables(f.dtype, f.device)
    rho = f.sum(0)
    ux = (c[:, 0].view(9, 1, 1) * f).sum(0) / rho
    uy = (c[:, 1].view(9, 1, 1) * f).sum(0) / rho
    return rho, ux, uy


def equilibrium(rho: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    """Upstream ``Feq`` (``Domain.CollideSC``): Cs = dx/dt = 1.

    ``W*rho*(1 + 3*(c.u) + 4.5*(c.u)^2 - 1.5*(u.u))``
    """
    w, c, _ = _tables(rho.dtype, rho.device)
    cx = c[:, 0].view(9, 1, 1)
    cy = c[:, 1].view(9, 1, 1)
    cu = cx * ux.unsqueeze(0) + cy * uy.unsqueeze(0)
    uu = ux * ux + uy * uy
    return w.view(9, 1, 1) * rho.unsqueeze(0) * (
        1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * uu.unsqueeze(0)
    )


def q_proxy(f_neq: torch.Tensor) -> torch.Tensor:
    """``Q = sqrt(2 * sum_q f_neq_q^2 * EEk_q)`` -- the upstream |c_n c_m| proxy."""
    _, _, eek = _tables(f_neq.dtype, f_neq.device)
    q = (f_neq * f_neq * eek.view(9, 1, 1)).sum(0)
    return torch.sqrt(2.0 * q)


def tau_eff(q: torch.Tensor, rho: torch.Tensor, tau0: float, Sc: float) -> torch.Tensor:
    """``tau = 0.5*(tau0 + sqrt(tau0^2 + 6*Q*Sc/rho))`` (no clamp, as upstream)."""
    rho_safe = torch.clamp(rho, min=1e-30)
    return 0.5 * (tau0 + torch.sqrt(torch.clamp(tau0 * tau0 + 6.0 * q * Sc / rho_safe, min=0.0)))


def collide(f: torch.Tensor, tau0: float, Sc: float | None = None) -> torch.Tensor:
    """Upstream ``CollideSC`` for a fully periodic fluid domain.

    ``Sc = 0`` (or ``None``) is upstream's ``les=False`` branch, i.e. plain BGK.
    """
    if Sc is None:
        Sc = float(lattice()["Sc"])
    rho, ux, uy = moments(f)
    feq = equilibrium(rho, ux, uy)
    f_neq = f - feq
    if Sc == 0.0:
        return f - f_neq / tau0
    q = q_proxy(f_neq)
    return f - f_neq / tau_eff(q, rho, tau0, Sc).unsqueeze(0)


def nu_t(q: torch.Tensor, rho: torch.Tensor, tau0: float, Sc: float) -> torch.Tensor:
    """SGS viscosity the upstream closure applies: ``nu_t = (tau_eff - tau0)/3``."""
    return (tau_eff(q, rho, tau0, Sc) - tau0) / 3.0