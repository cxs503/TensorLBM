"""The two cross-force operators under cross-validation.

Repo operator (as shipped)
--------------------------
``tensorlbm.multiphase.sc_two_component_force(rho1, rho2, G_12)`` with the
library's linear pseudopotential psi = rho (Palabos ``PsiIsRho`` convention):

    F1 = -G_12 * rho1 * sum_i w_i c_i rho2(x - c_i)   (backward gather)
    F2 = -G_12 * rho2 * sum_i w_i c_i rho1(x - c_i)

G_12 < 0 is repulsive (phase separation).  The gather is the vectorised
``_sc_neighbor_weighted_sum`` (advanced-index gather, float32 weight table).

OSS operator (lbmpy 2.0)
------------------------
Transcribed from the lbmpy sdist test
``tests/shan_chen/test_shan_chen_two_component.py:55-59,73-81`` (the sdist
itself does not ship tests/; the transcription was made in m18126 from the
vendored copy, sha256 7e88e89825a34a56e79723fa670a1114926d6282766bf98b918849577fa63415):

    F_a = -g_ab * psi(rho_a) * sum_d w_d d psi(rho_b(x + d))   (forward gather)
    F_b = -g_ab * psi(rho_b) * sum_d w_d d psi(rho_a(x + d))

g_ab > 0 is repulsive.  Convention bridge:  G_12 = -g_ab.

The D2Q9 stencil is symmetric ({-c_i} = {c_i}, equal weights), so the forward
and the backward gather are algebraically identical:
    sum_i w_i c_i psi(x - c_i) = - sum_i w_i c_i psi(x + c_i)
which makes the two operators comparable on the same field.
"""

from __future__ import annotations

import torch

from tensorlbm.d2q9 import C, W
from tensorlbm.multiphase import (
    _sc_neighbor_weighted_sum,
    sc_two_component_force,
)

DT = torch.float64


# ---------------------------------------------------------------------------
# pseudopotentials
# ---------------------------------------------------------------------------

def psi_exp(rho: torch.Tensor) -> torch.Tensor:
    """lbmpy upstream ``:53-54``: rho0*(1 - exp(-rho/rho0)), rho0 = 1."""
    return 1.0 - torch.exp(-rho)


def psi_lin(rho: torch.Tensor) -> torch.Tensor:
    """Palabos ``PsiIsRho`` == TensorLBM MCMP as-shipped convention."""
    return rho


PSI = {"exp": psi_exp, "linear": psi_lin}


# ---------------------------------------------------------------------------
# repo operator (the library kernel, called directly — integration caller)
# ---------------------------------------------------------------------------

def repo_cross_force(rho_a: torch.Tensor, rho_b: torch.Tensor,
                     G_12: float) -> tuple[torch.Tensor, ...]:
    """Library cross force, psi = rho (the only form the library supports).

    Returns ``(Fx1, Fy1, Fx2, Fy2)`` in the repo (y, x) component order.
    """
    return sc_two_component_force(rho_a, rho_b, G_12)


def repo_cross_force_psi(rho_a: torch.Tensor, rho_b: torch.Tensor,
                        G_12: float, psi_mode: str = "linear") -> tuple[torch.Tensor, ...]:
    """Cross force with a selectable pseudopotential.

    ``linear`` -> the library kernel as shipped.  ``exp`` -> the *same*
    library gather operator (``_sc_neighbor_weighted_sum``) applied to the
    OSS pseudopotential; this is a diagnostic composition, **not** a library
    code path (the library two-component collision hard-codes psi = rho).
    """
    if psi_mode == "linear":
        return sc_two_component_force(rho_a, rho_b, G_12)
    p1, p2 = PSI[psi_mode](rho_a), PSI[psi_mode](rho_b)
    sx2, sy2 = _sc_neighbor_weighted_sum(p2)
    sx1, sy1 = _sc_neighbor_weighted_sum(p1)
    return (-G_12 * p1 * sx2, -G_12 * p1 * sy2,
            -G_12 * p2 * sx1, -G_12 * p2 * sy1)


# ---------------------------------------------------------------------------
# OSS operator (transcribed from the lbmpy sdist source, forward gather)
# ---------------------------------------------------------------------------

def oss_cross_force(rho_a: torch.Tensor, rho_b: torch.Tensor, g_ab: float,
                    psi_mode: str = "linear") -> tuple[torch.Tensor, ...]:
    """OSS forward-gather cross force (lbmpy sdist ``:55-59``/``:73-81``).

    ``[+d]`` is the *forward* stencil neighbour (pystencils field offset
    ``rho[d]``).  Returns ``(Fx_a, Fy_a, Fx_b, Fy_b)`` in the repo (y, x)
    component order.
    """
    c = C.to(torch.long)
    w = W.to(DT)
    ny, nx = rho_a.shape

    def fwd(field: torch.Tensor):
        fx = torch.zeros_like(field)
        fy = torch.zeros_like(field)
        for i in range(9):
            cx, cy = int(c[i, 0]), int(c[i, 1])
            if cx == 0 and cy == 0:
                continue
            fx = fx + w[i] * cx * torch.roll(field, shifts=(-cy, -cx), dims=(0, 1))
            fy = fy + w[i] * cy * torch.roll(field, shifts=(-cy, -cx), dims=(0, 1))
        return fx, fy

    pa, pb = PSI[psi_mode](rho_a), PSI[psi_mode](rho_b)
    sa_x, sa_y = fwd(pa)
    sb_x, sb_y = fwd(pb)
    # g_aa = g_bb = 0: g_ab multiplies the *other* component's sum
    return (-g_ab * pa * sb_x, -g_ab * pa * sb_y,
            -g_ab * pb * sa_x, -g_ab * pb * sa_y)


# ---------------------------------------------------------------------------
# streaming (shared by the repo arm and the tests)
# ---------------------------------------------------------------------------

def stream_periodic(f: torch.Tensor) -> torch.Tensor:
    """Periodic pull streaming, D2Q9 (repo cell order: dim 0 = y, dim 1 = x)."""
    out = torch.empty_like(f)
    for i in range(9):
        cx, cy = int(C[i, 0]), int(C[i, 1])
        out[i] = torch.roll(f[i], shifts=(-cy, -cx), dims=(0, 1))
    return out
