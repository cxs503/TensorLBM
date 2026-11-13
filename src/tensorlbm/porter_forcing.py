"""Porter et al. (2012) explicit forcing for two-component (MCMP) lattice Boltzmann.

Zero code copy: the algorithm below is implemented from the public literature and
from the *documented* reading of the reference implementation recorded in the
platform OSS inventory (``docs/oss-inventory-openLBMPM.md`` §II.2 and §4.3,
``oss_solutions.tag = porter-explicit-forcing-4th-order``).  No line of the
reference implementation was copied (that tree carries no licence).

The four pieces of the scheme (inventory §II.2, file:line pointers
``ExplicitD2Q9GPU.py:255/271-276/278-287/294-304/311-332/340-363``):

1. subtract a half force before collision      ``f <- f - 0.5 F``
2. collision                                  ``f <- f + (f_eq - f - 0.5 F)/tau + F``
3. physical velocity carries the half force   ``u = [sum_k sum_i f_ik c_k + 0.5 (F_0+F_1)]/rho_tot``
4. viscosity-weighted equilibrium velocity
   ``u_eq = sum_i[(sum_k f_ik c_k + 0.5 F_i)/tau_i] / sum_i (rho_i/tau_i)``

with the discrete force PDF

    F_i = (F . (c_i - u_eq)) f_i^eq / (rho c_s^2)                     (Porter 2012, Eq. 20)

Piece 4 is the one that makes large viscosity ratios usable: the equilibrium
velocity is weighted by ``rho_i/tau_i`` rather than by ``rho_i``, so a stiff
(high-viscosity) component does not drag the light one out of the Stokes
solution.

The module also provides the isotropic gradient stencils used for the
interaction (pseudo-potential) force.  The reference implementation exposes
"isotropy 4/8/10" stencils; its weight literals are not recoverable without the
source, so the stencils here are *reconstructed* from their defining invariants
(``sum_k w_k c_kx^2 = 1`` and vanishing of the anisotropic moments up to the
stated order) and the achieved isotropy constants are reported by
:func:`isotropy_moments`.  For the 4th-order set the reconstruction reproduces
the documented ``sum w = 5/3`` and ``A4 = 1, A6 = 0.6, A8 = 3/7`` exactly.
"""

from __future__ import annotations

import numpy as np

CS2 = 1.0 / 3.0

# D2Q9, the numbering used throughout TensorLBM (E, N, W, S, NE, NW, SW, SE)
C = np.array(
    [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]],
    dtype=float,
)
W = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36], float)
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
CX = C[:, 0]
CY = C[:, 1]


# ---------------------------------------------------------------------------
# basic D2Q9 kernels
# ---------------------------------------------------------------------------


def equilibrium(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """Second-order equilibrium ``f_i^eq(rho, u)`` for D2Q9 (shape ``(9, ny, nx)``)."""
    cu = CX[:, None, None] * ux[None] + CY[:, None, None] * uy[None]
    usq = ux * ux + uy * uy
    return rho[None] * (W[:, None, None] * (1.0 + cu / CS2 + (cu * cu - CS2 * usq[None]) / (2 * CS2**2)))


def momentum(f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Raw lattice momentum ``(sum_k f_k c_kx, sum_k f_k c_ky)`` (no half force)."""
    return np.tensordot(CX, f, axes=(0, 0)), np.tensordot(CY, f, axes=(0, 0))


def density(f: np.ndarray) -> np.ndarray:
    return f.sum(axis=0)


def porter_force_pdf(
    fx: np.ndarray,
    fy: np.ndarray,
    feq: np.ndarray,
    rho: np.ndarray,
    ux_eq: np.ndarray,
    uy_eq: np.ndarray,
) -> np.ndarray:
    """Discrete force PDF ``F_i = (F.(c_i - u_eq)) f_i^eq / (rho c_s^2)``.

    The ratio ``f_i^eq/rho`` is evaluated analytically (``w_i(1 + cu/cs^2 +
    (cu^2-u^2)/(2cs^4))``) instead of dividing the built equilibrium by the
    density: in a two-component run a cell can hold ~1e-3 of the absent
    component, and the naive quotient turns the rounding of that residue into a
    spuriously large force PDF (measured: blow-up in 3 steps).
    """
    cu = CX[:, None, None] * ux_eq[None] + CY[:, None, None] * uy_eq[None]
    usq = ux_eq * ux_eq + uy_eq * uy_eq
    shape = W[:, None, None] * (1.0 + cu / CS2 + (cu * cu - CS2 * usq[None]) / (2 * CS2**2))
    cum = (CX[:, None, None] - ux_eq[None]) * fx[None] + (CY[:, None, None] - uy_eq[None]) * fy[None]
    return cum * shape / CS2


def porter_collide(
    f: np.ndarray,
    feq: np.ndarray,
    fpdf: np.ndarray,
    tau: float,
    variant: str = "doc",
) -> np.ndarray:
    """Collision step of the explicit forcing scheme.

    ``variant="doc"`` is the documented reference sequence (pieces 1-2);
    ``"guo"`` and ``"plain"`` are kept as sensitivity arms (they differ only in
    the bookkeeping of the half force, which is exactly the quantity this module
    is validating).
    """
    if variant in ("doc", "doc_literal"):
        # documented sequence read literally: the -0.5F inside the relaxation is a
        # second subtraction on top of the pre-collision one
        half = 0.5 * fpdf
        f_star = f - half
        return f_star + (feq - f_star - half) / tau + fpdf
    if variant == "doc_seq":
        # documented sequence read as "half force before collision, full force after"
        f_star = f - 0.5 * fpdf
        return f_star + (feq - f_star) / tau + fpdf
    if variant == "guo":
        return f + (feq - f) / tau + (1.0 - 0.5 / tau) * fpdf
    if variant == "plain":
        return f + (feq - f) / tau + fpdf
    raise ValueError(f"unknown variant {variant!r}")


def physical_velocity(
    jx: np.ndarray, jy: np.ndarray, fx_tot: np.ndarray, fy_tot: np.ndarray, rho_tot: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Piece 3: ``u = [sum_k sum_i f_ik c_k + 0.5 sum_i F_i] / rho_tot``."""
    safe = np.maximum(rho_tot, 1e-30)
    return (jx + 0.5 * fx_tot) / safe, (jy + 0.5 * fy_tot) / safe


def viscosity_weighted_ueq(
    jxs: list[np.ndarray],
    jys: list[np.ndarray],
    fxs: list[np.ndarray],
    fys: list[np.ndarray],
    rhos: list[np.ndarray],
    taus: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Piece 4: ``u_eq = sum_i[(J_i + 0.5 F_i)/tau_i] / sum_i (rho_i/tau_i)``."""
    num_x = np.zeros_like(rhos[0])
    num_y = np.zeros_like(rhos[0])
    den = np.zeros_like(rhos[0])
    for jx, jy, fx, fy, rho, tau in zip(jxs, jys, fxs, fys, rhos, taus):
        num_x += (jx + 0.5 * fx) / tau
        num_y += (jy + 0.5 * fy) / tau
        den += rho / tau
    safe = np.maximum(den, 1e-30)
    return num_x / safe, num_y / safe


# ---------------------------------------------------------------------------
# streaming with half-way bounce-back walls
# ---------------------------------------------------------------------------


def stream_halfway_bb(f: np.ndarray, solid: np.ndarray) -> np.ndarray:
    """Pull streaming, periodic in ``x``, half-way bounce-back on solid nodes.

    Solid rows are expected at ``j = 0`` and ``j = NY-1`` (channel walls); a
    population whose upstream node is solid is reflected from the *same* node,
    which places the wall at half a lattice spacing outside the last fluid cell.
    """
    ny, nx = f.shape[1], f.shape[2]
    jj = np.arange(ny)[:, None]
    ii = np.arange(nx)[None, :]
    out = np.zeros_like(f)
    for k in range(9):
        dy, dx = int(CY[k]), int(CX[k])
        sj = jj - dy
        si = (ii - dx) % nx  # broadcast (ny, 1) x (1, nx) -> (ny, nx)
        inside = (sj >= 0) & (sj < ny)
        sjc = np.clip(sj, 0, ny - 1)
        src = f[k][sjc, si]
        src_solid = solid[sjc, si]
        reflected = f[OPP[k]]
        out[k] = np.where(inside, np.where(src_solid, reflected, src), 0.0)
    out[:, solid] = 0.0
    return out


# ---------------------------------------------------------------------------
# isotropic gradient stencils (interaction force)
# ---------------------------------------------------------------------------


def isotropic_gradient_weights(order: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Isotropic gradient stencil with ``sum_k w_k c_kx^2 = 1``.

    ``order=2``  D2Q9 eight-point weights (as used by the classic force kernel);
    ``order=4``  axis weight ``1/3`` + diagonal weight ``1/12``  (``sum w = 5/3``);
    ``order=6``  adds the four ``(+-2, 0)``/``(0, +-2)`` neighbours: axis ``4/15``,
                 diagonal ``1/10``, second neighbour ``1/120``.

    The 4th-order set is *derived* here from the two constraints
    ``sum w c_kx^2 = 1`` and ``sum w c_kx^4 = 3 sum w c_kx^2 c_ky^2``; it
    reproduces the reference implementation's documented ``sum w = 5/3``.
    """
    if order == 2:
        return C.copy(), W.copy()
    if order == 4:
        offsets = np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]], float)
        weights = np.array([1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 12, 1 / 12, 1 / 12, 1 / 12])
        return offsets, weights
    if order == 6:
        offsets = np.array(
            [
                [1, 0], [0, 1], [-1, 0], [0, -1],
                [1, 1], [-1, 1], [-1, -1], [1, -1],
                [2, 0], [0, 2], [-2, 0], [0, -2],
            ],
            float,
        )
        weights = np.array(
            [4 / 15, 4 / 15, 4 / 15, 4 / 15, 1 / 10, 1 / 10, 1 / 10, 1 / 10, 1 / 120, 1 / 120, 1 / 120, 1 / 120]
        )
        return offsets, weights
    raise ValueError(f"unsupported isotropy order {order!r} (supported: 2, 4, 6)")


def isotropy_moments(offsets: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    """Measured moments of a gradient stencil.

    The isotropy constants follow the normalisation implied by the reference
    inventory table (denominators 3/5/7, theory value 1.0):

        ``A4 = sum w cx^4 / (3 sum w cx^2 cy^2)``
        ``A6 = sum w cx^6 / (5 sum w cx^4 cy^2)``
        ``A8 = sum w cx^8 / (7 sum w cx^6 cy^2)``
    """
    cx = offsets[:, 0]
    cy = offsets[:, 1]
    m2 = float(np.sum(weights * cx**2))
    m4a = float(np.sum(weights * cx**4))
    m4b = float(np.sum(weights * cx**2 * cy**2))
    m6a = float(np.sum(weights * cx**6))
    m6b = float(np.sum(weights * cx**4 * cy**2))
    m8a = float(np.sum(weights * cx**8))
    m8b = float(np.sum(weights * cx**6 * cy**2))

    def _ratio(a: float, b: float) -> float:
        return float("inf") if b == 0.0 else a / b

    return {
        "sum_w": float(np.sum(weights)),
        "sum_w_cx2": m2,
        "A4": _ratio(m4a, 3.0 * m4b),
        "A6": _ratio(m6a, 5.0 * m6b),
        "A8": _ratio(m8a, 7.0 * m8b),
    }


def gradient_field(field: np.ndarray, offsets: np.ndarray, weights: np.ndarray, periodic: bool = True):
    """Discrete gradient ``grad f = sum_k w_k f(x+c_k) c_k``.

    Periodic wrap in both directions (the reaction/validation cases here are
    either periodic in ``x`` only and flat in ``y``, where the ``y`` wrap is
    irrelevant, or fully periodic).
    """
    gx = np.zeros_like(field)
    gy = np.zeros_like(field)
    ny, nx = field.shape
    for (dx, dy), wk in zip(offsets.astype(int), weights):
        if periodic:
            shifted = np.roll(np.roll(field, -dy, axis=0), -dx, axis=1)
        else:  # zero padding
            shifted = np.zeros_like(field)
            j0, j1 = max(0, dy), ny + min(0, dy)
            i0, i1 = max(0, dx), nx + min(0, dx)
            shifted[j0:j1, i0:i1] = field[j0 - dy : j1 - dy, i0 - dx : i1 - dx]
        gx += wk * dx * shifted
        gy += wk * dy * shifted
    return gx, gy


def shan_chen_force(
    psi_self: np.ndarray,
    psi_other: np.ndarray,
    G: float,
    order: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-component pseudo-potential force with an isotropic gradient.

    ``F_s = G c_s^2 psi_s grad psi_s'``  (``G < 0`` phase-separates, the sign
    convention recorded in the platform inventory and in
    ``mcmp_verification.sc_mcmp_pressure``).
    """
    offsets, weights = isotropic_gradient_weights(order)
    gx, gy = gradient_field(psi_other, offsets, weights)
    return G * CS2 * psi_self * gx, G * CS2 * psi_self * gy