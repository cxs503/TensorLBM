"""Selector-only extraction of the OSS ``Maarten-vd-Sande/lbm`` mass-exchange
(cell-conversion) rules — anchor ``XVAL-LBM-MASS-EXCHANGE-SELECTOR``.

Why a separate module
---------------------
The full OSS free-surface step is already ported and fidelity-checked in
``xval/oracle_minimal_d2q9_fs/oss_oracle_port.py`` (that port reproduces the
unmodified upstream code to float32 round-off).  This XVAL is about *one*
sub-model — the **mass-exchange selector**: which INTERFACE cell is allowed to
become FLUID / forced to GAS, and where its surplus/defect mass is placed.
Extracting it keeps the differential test single-factor.

Zero-hand-copy provenance
-------------------------
Every rule and constant below is a transcription of the inventoried snapshot
``/root/oss-cache/Maarten-vd-Sande_lbm`` at the cited line range.  The
arithmetic primitives (``pulli``/``pushi`` lattice slicing, ``get_normal``,
``get_epsilon``, ``CT`` bit flags) are **imported** from the fidelity-checked
numpy port, so no lattice coordinate convention is re-derived by hand here.

=================================  =========================================
OSS location                       rule reproduced in this module
=================================  =========================================
``lbm/update_types.py:12``         ``fill_offset = 0.003``
``lbm/update_types.py:13``         ``lonely_tresh = 0.1``
``lbm/update_types.py:18-27``      conversion marking (``mark``), incl. the
                                   ``NO_EMPTY_NEIGH`` guard on promotion and
                                   the unconditional ``NO_IFACE_NEIGH |
                                   NO_FLUID_NEIGH`` demotion
``lbm/update_types.py:34-70``      envelope promotion / demotion of the
                                   neighbours of a converting cell
                                   (``envelope``), used only to expose the
                                   receiver set of the distribution
``lbm/update_types.py:72-116``     excess-mass distribution along the surface
                                   normal ``eta = max(c.v n, 0)`` with a uniform
                                   fallback over interface neighbours
                                   (``redistribute``)
``lbm/update_types.py:118-136``    neighbourhood-flag sweep ``neigh_flags``
``lbm/streamcollide.py``           ``get_epsilon`` / ``get_normal`` (imported)
=================================  =========================================

Coordinates: interior arrays shaped ``(nx, ny)`` (the port's convention, axis 0
= x).  The caller is responsible for stripping the one-cell OBSTACLE padding.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_PORT_DIR = Path(__file__).resolve().parent.parent / "oracle_minimal_d2q9_fs"
if str(_PORT_DIR) not in sys.path:
    sys.path.insert(0, str(_PORT_DIR))

import oss_oracle_port as P  # noqa: E402  (fidelity-checked upstream port)

FLUID = P.FLUID
INTERFACE = P.INTERFACE
GAS = P.GAS
OBSTACLE = P.OBSTACLE
NO_FLUID_NEIGH = P.NO_FLUID_NEIGH
NO_EMPTY_NEIGH = P.NO_EMPTY_NEIGH
NO_IFACE_NEIGH = P.NO_IFACE_NEIGH
TO_FLUID = P.TO_FLUID
TO_GAS = P.TO_GAS
_MOVING = P._MOVING
V = P.V

FILL_OFFSET = P.FILL_OFFSET      # update_types.py:12  -> 0.003
LONELY_TRESH = P.LONELY_TRESH    # update_types.py:13  -> 0.1


# ---------------------------------------------------------------------------
def neigh_flags(ct):
    """``update_types.py:118-136`` — refresh the ``NO_*_NEIGH`` bits.

    Same construction as the port's sweep 5 (pull sampling with ``False``
    outside the domain, ``OBSTACLE`` cells excluded), so the bit pattern is
    byte-identical to what the upstream code holds at the start of a step.
    """
    ct = np.asarray(ct, dtype=np.int32)
    not_obs = (ct & OBSTACLE) == 0
    has_f = np.zeros(ct.shape, dtype=bool)
    has_g = np.zeros(ct.shape, dtype=bool)
    has_i = np.zeros(ct.shape, dtype=bool)
    for k in _MOVING:
        has_f |= P.pulli((ct & FLUID) != 0, k, False)
        has_g |= P.pulli((ct & GAS) != 0, k, False)
        has_i |= P.pulli((ct & INTERFACE) != 0, k, False)
    base = NO_FLUID_NEIGH | NO_EMPTY_NEIGH | NO_IFACE_NEIGH
    out = np.where(not_obs, ct | base, ct).astype(np.int32)
    out = np.where(not_obs & has_f, out & ~NO_FLUID_NEIGH, out).astype(np.int32)
    out = np.where(not_obs & has_g, out & ~NO_EMPTY_NEIGH, out).astype(np.int32)
    out = np.where(not_obs & has_i, out & ~NO_IFACE_NEIGH, out).astype(np.int32)
    return out


def mark(ct, rho, mass, fill_offset=FILL_OFFSET, lonely_tresh=LONELY_TRESH,
         extra_demotion=True):
    """``update_types.py:18-27`` — the selector itself (sweep 1).

    ``extra_demotion`` switches the unconditional ``NO_IFACE_NEIGH |
    NO_FLUID_NEIGH`` clause of the upstream ``elif`` (line 26).  ``True`` is
    the shipped OSS behaviour; ``False`` is the factorial-design cell that
    isolates the term from the ``fill_offset`` magnitude.
    """
    ct = np.asarray(ct, dtype=np.int32)
    rho = np.asarray(rho, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64)
    iface = (ct & INTERFACE) != 0
    no_empty = (ct & NO_EMPTY_NEIGH) != 0
    no_fluid = (ct & NO_FLUID_NEIGH) != 0
    no_iface = (ct & NO_IFACE_NEIGH) != 0

    to_fluid = iface & ((mass > (1.0 + fill_offset) * rho)
                        | ((mass >= (1.0 - lonely_tresh) * rho) & no_empty))
    to_gas = iface & (~to_fluid) & (
        (mass < -fill_offset * rho)
        | ((mass <= lonely_tresh * rho) & no_fluid)
        | ((no_iface | no_fluid) if extra_demotion
           else np.zeros(iface.shape, dtype=bool)))
    return to_fluid, to_gas


def envelope(ct, to_fluid, to_gas):
    """``update_types.py:18-70`` — mark the conversions, then the neighbour
    envelope of every converting cell.

    Two upstream semantics that matter for the receiver set and are easy to
    get wrong:

    * ``cell_type[ix, iy] = CT.TO_FLUID`` (line 22) is an **assignment**, not a
      bit-or: a converting cell loses its ``INTERFACE`` bit, so it can never be
      a receiver of excess mass in the same sweep (line 78 tests
      ``cell_type & CT.INTERFACE``).  The repo reaches the same effect by
      masking ``recv_iface = iface & ~to_liq & ~to_gas``.
    * line 30 clears the three ``NO_*_NEIGH`` bits on *every* cell.

    Returns ``(ct_now, promote, demote)``: GAS neighbours of a ``TO_FLUID``
    cell are promoted to INTERFACE (``average_surround``), FLUID neighbours of
    a ``TO_GAS`` cell are demoted to INTERFACE.  Only the classification is
    needed to know the receiver set; the refill (density/velocity) is outside
    this XVAL.
    """
    ct = np.asarray(ct, dtype=np.int32)
    ct = np.where(to_fluid, np.int32(TO_FLUID), ct).astype(np.int32)
    ct = np.where(to_gas, np.int32(TO_GAS), ct).astype(np.int32)
    ct = (ct & ~np.int32(NO_FLUID_NEIGH + NO_EMPTY_NEIGH + NO_IFACE_NEIGH)).astype(np.int32)
    tf = (ct & TO_FLUID) != 0
    tg = (ct & TO_GAS) != 0
    promote = np.zeros(ct.shape, dtype=bool)
    demote = np.zeros(ct.shape, dtype=bool)
    for k in _MOVING:
        promote |= P.pulli(tf, k, False) & ((ct & GAS) != 0)
        demote |= P.pulli(tg, k, False) & ((ct & FLUID) != 0)
    ct = np.where(promote, INTERFACE, ct).astype(np.int32)
    ct = np.where(demote & ~promote, INTERFACE, ct).astype(np.int32)
    return ct, promote, demote


def redistribute(ct_prev, ct_now, rho_prev, mass_prev, to_fluid, to_gas,
                 mass, rho):
    """``update_types.py:72-116`` — spread the surplus/defect mass.

    Mirrors the port's sweep 4 (lines 424-456 of ``oss_oracle_port.py``): the
    surface normal comes from the *pre-conversion* field, the receivers are the
    INTERFACE cells of the *post-envelope* field, weights are
    ``eta = max(c_k . n, 0)`` with a uniform fallback when every ``eta``
    vanishes.  Returns ``(mass_new, placed)`` where ``placed`` is the 9-way
    per-direction increment (its sum is what each receiver actually gets).
    """
    mass = np.asarray(mass, dtype=np.float64)
    nrm_x, nrm_y = P.get_normal(np.asarray(ct_prev, dtype=np.int32),
                               np.asarray(rho_prev, dtype=np.float64),
                               np.asarray(mass_prev, dtype=np.float64))
    nrm_x = np.where(to_gas, -nrm_x, nrm_x)
    nrm_y = np.where(to_gas, -nrm_y, nrm_y)
    mex = np.zeros(mass.shape, dtype=np.float64)
    mex = np.where(to_fluid, mass - rho, mex)
    mex = np.where(to_gas, mass, mex)
    mass = np.where(to_fluid, rho, mass)
    mass = np.where(to_gas, 0.0, mass)

    eta = np.zeros((9,) + mass.shape, dtype=np.float64)
    is_if = np.zeros((9,) + mass.shape, dtype=bool)
    for k in _MOVING:
        nb_if = P.pushi((ct_now & INTERFACE) != 0, k, False)
        eta[k] = np.where(nb_if, np.maximum(V[k, 0] * nrm_x + V[k, 1] * nrm_y,
                                           0.0), 0.0)
        is_if[k] = nb_if
    eta_total = eta.sum(axis=0)
    if_total = is_if.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ok = eta_total > 0
        fdist = np.where(ok, eta * mex / np.where(ok, eta_total, 1.0), 0.0)
        fb = (~ok) & (if_total > 0)
        per = np.where(if_total > 0, mex / np.where(if_total > 0, if_total, 1.0), 0.0)
        fdist = np.where(fb, np.where(is_if, per, 0.0), fdist)
    placed = np.zeros((9,) + mass.shape, dtype=np.float64)
    add = np.zeros(mass.shape, dtype=np.float64)
    for k in _MOVING:
        placed[k] = P.pushi(fdist[k], k, 0.0)
        add += placed[k]
    mass_new = np.where((ct_now & INTERFACE) != 0, mass + add, mass)
    return mass_new, placed, mex


def select(ct, rho, mass, ct_prev=None, rho_prev=None, mass_prev=None,
           fill_offset=FILL_OFFSET, lonely_tresh=LONELY_TRESH,
           extra_demotion=True):
    """One full OSS selector pass on an already-refreshed flag field.

    ``ct`` must carry the ``NO_*_NEIGH`` bits (``neigh_flags``); ``*_prev``
    default to the current fields (what the upstream code holds when nothing
    changed between two ``update_types`` calls).
    """
    to_fluid, to_gas = mark(ct, rho, mass, fill_offset, lonely_tresh,
                            extra_demotion)
    ct_now, promote, demote = envelope(ct, to_fluid, to_gas)
    ct_prev = ct if ct_prev is None else np.asarray(ct_prev, dtype=np.int32)
    rho_prev = rho if rho_prev is None else rho_prev
    mass_prev = mass if mass_prev is None else mass_prev
    mass_new, placed, mex = redistribute(ct_prev, ct_now, rho_prev, mass_prev,
                                         to_fluid, to_gas, mass, rho)
    return {
        "to_fluid": to_fluid,
        "to_gas": to_gas,
        "mex": mex,
        "placed": placed,
        "mass_new": mass_new,
        "ct_now": ct_now,
        "promote": promote,
        "demote": demote,
    }