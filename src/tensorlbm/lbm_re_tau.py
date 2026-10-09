"""Re ↔ tau (BGK relaxation) conversion helpers — unified diameter convention.

Reynolds-number / relaxation-time conversion is repeated across many
validation and benchmark scripts.  Historically each script rolled its own
formula and some used the *radius* ``R`` as the characteristic length while
others used the *diameter* ``2R``, silently producing a factor-2 mismatch in
the effective Reynolds number (e.g. the Re=200 bug in
``octree_distributed_validate.py``).

This module fixes the convention once, here:

**Convention (D 口径): the characteristic length ``L_ref`` is the DIAMETER
``L_ref = 2R``** — matching the classic cylinder/sphere definition
``Re = u·D/ν = u·2R/ν``.  Callers must pass ``L_ref = 2 * radius``.

Formulas (D3Q27/D3Q19 BGK, lattice speed of sound :math:`c_s = 1/\\sqrt{3}`,
so :math:`\\nu = (\\tau - 0.5)/3`):

* ``tau = 0.5 + 3·(u·L_ref/Re)``
* ``Re  = 3·u·L_ref/(tau - 0.5)``
* ``nu  = u·L_ref/Re``
* ``nu  = (tau - 0.5)/3``

All functions accept plain floats or 0-dim tensors and are pure arithmetic
(no torch import needed).
"""

from __future__ import annotations

__all__ = ["tau_from_re", "re_from_tau", "nu_from_re", "nu_from_tau"]


def _check_positive(name: str, value) -> None:
    if value is not None and value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


def tau_from_re(u, L_ref, Re):
    """BGK relaxation time for a target Reynolds number.

    ``tau = 0.5 + 3·(u·L_ref/Re)``

    Parameters
    ----------
    u : float
        Characteristic (inflow) lattice velocity, e.g. ``u_in``.
    L_ref : float
        Characteristic length in lattice units.  **Project convention:
        diameter, i.e. ``L_ref = 2 * radius``** (D 口径, ``Re = u·2R/ν``).
        Do NOT pass the bare radius — that halves the Reynolds number.
    Re : float
        Target Reynolds number (``Re = u·L_ref/ν``).

    Returns
    -------
    float
        BGK relaxation time ``tau`` (``> 0.5`` for physical ``Re > 0``).

    Examples
    --------
    >>> tau_from_re(0.06, 2 * 6, 100)    # R=6  → 0.5216
    0.5216
    >>> tau_from_re(0.06, 2 * 10, 100)   # R=10 → 0.536
    0.536
    """
    _check_positive("L_ref", L_ref)
    _check_positive("Re", Re)
    return 0.5 + 3.0 * (u * L_ref / Re)


def re_from_tau(u, L_ref, tau):
    """Reynolds number implied by a BGK relaxation time.

    ``Re = 3·u·L_ref/(tau - 0.5)``

    Parameters
    ----------
    u : float
        Characteristic (inflow) lattice velocity.
    L_ref : float
        Characteristic length — **diameter ``2R``** (see :func:`tau_from_re`).
    tau : float
        BGK relaxation time, must be ``> 0.5`` (positive viscosity).

    Returns
    -------
    float
        Reynolds number ``Re``.
    """
    _check_positive("L_ref", L_ref)
    if tau <= 0.5:
        raise ValueError(f"tau must be > 0.5, got {tau}")
    return 3.0 * u * L_ref / (tau - 0.5)


def nu_from_re(u, L_ref, Re):
    """Kinematic lattice viscosity for a target Reynolds number.

    ``nu = u·L_ref/Re``

    Parameters
    ----------
    u : float
        Characteristic (inflow) lattice velocity.
    L_ref : float
        Characteristic length — **diameter ``2R``** (see :func:`tau_from_re`).
    Re : float
        Target Reynolds number.

    Returns
    -------
    float
        Kinematic viscosity ``nu`` in lattice units.
    """
    _check_positive("L_ref", L_ref)
    _check_positive("Re", Re)
    return u * L_ref / Re


def nu_from_tau(tau):
    """Kinematic lattice viscosity from a BGK relaxation time.

    ``nu = (tau - 0.5)/3``  (lattice ``c_s = 1/sqrt(3)``).

    Parameters
    ----------
    tau : float
        BGK relaxation time, must be ``> 0.5``.

    Returns
    -------
    float
        Kinematic viscosity ``nu`` in lattice units.
    """
    if tau <= 0.5:
        raise ValueError(f"tau must be > 0.5, got {tau}")
    return (tau - 0.5) / 3.0
