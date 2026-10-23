"""Differentiable solids from analytic signed distance functions.

`tensorlbm.autograd_path` treats the obstacle as a *static boolean mask*:
collision is skipped with ``torch.where``, streaming is followed by a full-way
bounce-back that swaps populations on solid nodes, and the momentum-exchange
force sums over solid cells.  That operator is differentiable with respect to
everything *except the geometry itself* — the mask is a discrete object, so no
gradient can reach a radius, an axis length or a centre position.

This module closes that gap with a **soft solid**: an analytic signed distance
function ``phi(x; params)`` (negative inside the solid) blurred by a
temperature ``epsilon`` into the fluid weight field

.. math:: w(x) = \\sigma\\big(\\phi(x)/\\epsilon\\big) \\in (0, 1),

with :math:`\\sigma` the logistic function: ``w -> 1`` deep in the fluid,
``w -> 0`` deep in the solid, and a smooth transition band of width
:math:`O(\\epsilon)` (in lattice units) across the surface.  Beyond
:math:`|\\phi/\\epsilon| = 30` the weight is clipped to *exactly* 0/1 so the
operators degenerate bit-for-bit to the hard-mask chain wherever the nodes
sit more than :math:`30\\epsilon` from the surface (the logistic tail there
is ~1e-13 — no physics change; see :meth:`SoftGeometry.fluid_weight`).
Plugged into the
step chain through ``differentiable_step(..., soft=geom)`` (opt-in; the
default ``soft=None`` chain is bit-for-bit unchanged), every geometric
parameter — centre, radius, semi-axes, half-extents — becomes a 0-dim tensor
the loss can be differentiated against: **inverse design through the solver**
(``examples/inverse_design.py``).

The three operators, each the convex homotopy of its hard counterpart
(:math:`s = 1 - w` is the solid weight):

* soft collision skip (soft NoDynamics; ``w`` is the *fluid* weight — the
  fluid collides, the solid is frozen)

  .. math:: f_{\\text{coll}} = w\\,\\mathrm{collide}(f) + (1 - w)\\,f;

* soft full-way bounce-back

  .. math:: f_{\\text{post}} = w\\,f_{\\text{str}} + (1 - w)\\,f_{\\text{str}}[\\bar q],

  the continuous blend of ``where(mask, f_str[opp], f_str)`` — full-way
  bounce-back reflects the populations *on the wet node itself*, so no
  link-wise mask shift is needed (unlike half-way link bounce-back, whose
  weight would have to be sampled at ``x + c_q``);

* soft momentum-exchange force (the continuous Ladd wet-node form)

  .. math:: F_\\alpha = 2 \\sum_x (1 - w(x)) \\sum_q c_{q,\\alpha}\\,
      f_{\\text{probe}}[q, x].

**Derivation of the soft force** (the assumption is stated first): the solid
weight :math:`s(x) = 1 - w(x)` is interpreted as the *solid occupancy* of node
``x`` — the fraction of the node's momentum exchange attributed to the body.
Per node, the soft bounce-back changes the field momentum by

.. math:: \\sum_q c_q \\big(f_{\\text{post}} - f_{\\text{str}}\\big)
    = s\\,\\Big(\\sum_q c_q f_{\\text{str}}[\\bar q] - \\sum_q c_q f_{\\text{str}}\\Big)
    = -2 s \\sum_q c_q f_{\\text{str}}[q],

(the first sum relabels :math:`q \\to \\bar q` with :math:`c_{\\bar q} = -c_q`),
so the momentum handed to the body is exactly
:math:`2 s \\sum_q c_q f_{\\text{probe}}` — action equals reaction *per node*,
and summing over nodes gives the force above.  At :math:`s \\in \\{0, 1\\}`
(black/white occupancy) every operator degenerates **bit-for-bit** to the
existing hard-mask path: the blends collapse onto the ``torch.where``
selections and the force onto ``autograd_path.obstacle_force`` — which is why
that function deliberately accepts either a boolean mask or a float solid
weight as its second argument.

The temperature trades gradient quality against hard-limit fidelity: the
transition band is :math:`\\sim 6\\epsilon` wide (sigmoid tails), so small
``epsilon`` converges to the hard-mask geometry (measured: C_D of a soft
sphere matches the hard-mask sphere to <1% at :math:`\\epsilon \\lesssim 0.1`
lattice units) while shrinking :math:`\\partial w/\\partial\\phi =
w(1-w)/\\epsilon` to a thin ring — the classic diffuse-interface compromise.
``epsilon`` of a few tenths of a lattice spacing is the practical sweet spot
for optimisation.

Shapes (``kind``): ``"sphere"`` (exact SDF), ``"ellipsoid"`` (normalised-radius
surrogate, see :meth:`SoftGeometry.sdf`) and ``"box"`` (exact axis-aligned
SDF).  All size/centre parameters accept Python floats or graph-connected
0-dim tensors.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["SoftGeometry"]

_KINDS = ("sphere", "ellipsoid", "box")

# Sigmoid saturation band: |phi/epsilon| beyond this is clipped to exactly
# 0/1.  The logistic tail at 30 is ~1e-13 (no physics change), but the exact
# black/white values matter: a blend weight of 1e-28 is *not* zero, and the
# convex homotopies would collide/bounce deep-solid cells at O(1) instead of
# freezing them like the hard mask (the hard-limit contract needs w in {0, 1}
# exactly, which plain sigmoid only reaches by fp underflow, at |phi/eps|>745
# in fp64 and >89 in fp32).
_SATURATION = 30.0

# Floor for the sqrt arguments of the sphere/ellipsoid SDFs: the exact-centre
# node has zero argument, where d(sqrt)/dx is infinite — inf upstream times a
# zero local factor (e.g. d((x-c)/a)^2/da at x = c) produces NaN gradients.
# Clamping the argument to this floor keeps values unchanged to ~1e-15 and
# zeroes the (meaningless) gradient within 1e-15 lattice of the centre.
_SQRT_FLOOR = 1e-30


def _scalar(value: float | torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Coerce a Python float or 0-dim tensor to *dtype/device* (graph kept)."""
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.tensor(value, dtype=dtype, device=device)


@dataclass(frozen=True)
class SoftGeometry:
    """Soft solid: analytic SDF ``phi`` (negative inside) + temperature.

    The geometry is evaluated lazily on a grid of node coordinates
    ``(z, y, x) = (0..nz-1, 0..ny-1, 0..nx-1)`` — the same convention as
    ``tensorlbm.boundaries3d.sphere_mask``, so ``hard_mask`` of the same
    parameters reproduces that mask exactly (up to boundary ties).

    Args:
        kind: ``"sphere"`` — exact Euclidean SDF ``|p - c| - r``;
            ``"ellipsoid"`` — triaxial ellipsoid with semi-axes ``(a, b, c)``;
            ``"box"`` — axis-aligned box with half-extents ``(hx, hy, hz)``:
            the standard box SDF, whose ``max(q, 0)`` construction keeps the
            distance continuous across face/corner regions (no discontinuous
            jumps of a naive inside/outside indicator).
        center: ``(cx, cy, cz)`` node coordinates of the shape centre.
        size: ``(r,)`` for a sphere; ``(a, b, c)`` semi-axes for an ellipsoid;
            ``(hx, hy, hz)`` half-extents for a box.  Every entry accepts a
            float or a graph-connected 0-dim tensor (the learnable parameter
            of the inverse-design demo).
        epsilon: SDF temperature in lattice units (must be > 0); a 0-dim
            tensor keeps it in the autograd graph (e.g. annealing schedules).
            The fluid weight is exactly 0/1 more than ``30 * epsilon`` from
            the surface (saturated logistic — see :meth:`fluid_weight`).

    All ``float | torch.Tensor`` fields keep 0-dim tensors connected to the
    autograd graph, exactly like the boundary specs of
    ``tensorlbm.autograd_path``.
    """

    kind: str = "sphere"
    center: tuple[float | torch.Tensor, float | torch.Tensor, float | torch.Tensor] = (
        0.0,
        0.0,
        0.0,
    )
    size: tuple[float | torch.Tensor, ...] = (1.0,)
    epsilon: float | torch.Tensor = 0.5

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f"soft geometry kind must be one of {_KINDS}, got {self.kind!r}")
        if len(self.center) != 3:
            raise ValueError(f"center must be (cx, cy, cz), got {len(self.center)} entries")
        expected = 1 if self.kind == "sphere" else 3
        if len(self.size) != expected:
            raise ValueError(
                f"{self.kind} needs {expected} size entr{'y' if expected == 1 else 'ies'} "
                f"({'radius' if expected == 1 else 'rx, ry, rz'}), got {len(self.size)}"
            )
        eps = (
            self.epsilon.detach().item() if isinstance(self.epsilon, torch.Tensor) else self.epsilon
        )
        if not eps > 0.0:
            raise ValueError(f"epsilon must be > 0, got {eps!r}")

    # -- field constructors ------------------------------------------------

    def _grid(
        self, nz: int, ny: int, nx: int, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Node coordinate grids ``(zz, yy, xx)`` of shape ``(nz, ny, nx)``."""
        zz, yy, xx = torch.meshgrid(
            torch.arange(nz, dtype=dtype, device=device),
            torch.arange(ny, dtype=dtype, device=device),
            torch.arange(nx, dtype=dtype, device=device),
            indexing="ij",
        )
        return zz, yy, xx

    def sdf(
        self,
        nz: int,
        ny: int,
        nx: int,
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Signed distance function ``phi`` on the grid, negative inside the solid.

        Shapes:

        * ``"sphere"`` — ``phi = |p - c| - r`` (exact distance; the sqrt is
          clamped at 0 so the degenerate centre node keeps finite values).
        * ``"ellipsoid"`` — the normalised-radius surrogate
          ``phi = (t - 1) * min(a, b, c)`` with
          ``t = sqrt(sum_i ((p_i - c_i) / a_i)^2)``.  ``t = 1`` is exactly the
          ellipsoid surface, but ``t - 1`` measures distance in units of
          *normalised radius*, not Euclidean length; rescaling by the smallest
          semi-axis makes ``epsilon`` lattice-unit meaningful (exact along the
          shortest axis, a lower bound elsewhere).  The exact ellipsoid SDF
          needs an iterative closest-point projection and is deliberately not
          reproduced here.
        * ``"box"`` — the standard exact axis-aligned box SDF
          ``phi = length(max(q, 0)) + min(max(q_x, q_y, q_z), 0)`` with
          ``q = |p - c| - h`` componentwise: Euclidean distance outside,
          signed distance to the nearest face inside.
        """
        dev = torch.device("cpu") if device is None else device
        zz, yy, xx = self._grid(nz, ny, nx, dtype, dev)
        cx = _scalar(self.center[0], dtype, dev)
        cy = _scalar(self.center[1], dtype, dev)
        cz = _scalar(self.center[2], dtype, dev)

        if self.kind == "sphere":
            radius = _scalar(self.size[0], dtype, dev)
            dist2 = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2
            return torch.sqrt(torch.clamp(dist2, min=_SQRT_FLOOR)) - radius

        rx = _scalar(self.size[0], dtype, dev)
        ry = _scalar(self.size[1], dtype, dev)
        rz = _scalar(self.size[2], dtype, dev)

        if self.kind == "ellipsoid":
            t2 = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 + ((zz - cz) / rz) ** 2
            t = torch.sqrt(torch.clamp(t2, min=_SQRT_FLOOR))
            shortest = torch.minimum(rx, torch.minimum(ry, rz))
            return (t - 1.0) * shortest

        # "box": q = |p - c| - h, exact SDF (outside + inside terms)
        qx = (xx - cx).abs() - rx
        qy = (yy - cy).abs() - ry
        qz = (zz - cz).abs() - rz
        outside = torch.sqrt(
            torch.clamp(
                qx.clamp_min(0.0) ** 2 + qy.clamp_min(0.0) ** 2 + qz.clamp_min(0.0) ** 2,
                min=torch.finfo(dtype).tiny,
            )
        )
        inside = torch.clamp(torch.maximum(qx, torch.maximum(qy, qz)), max=0.0)
        return outside + inside

    def fluid_weight(
        self,
        nz: int,
        ny: int,
        nx: int,
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Fluid weight ``w = sigmoid(phi / epsilon)`` on the grid.

        ``w -> 1`` in the fluid, ``-> 0`` in the solid; the transition band is
        ``~6 * epsilon`` wide.  Beyond ``|phi/epsilon| = 30`` the weight is
        clipped to *exactly* 0/1 (the logistic tail there is ~1e-13): a blend
        weight of e.g. 1e-28 is numerically-but-not-exactly zero, and the
        convex homotopies in ``autograd_path`` only degenerate to the
        hard-mask operators at exact black/white weights.  The solid weight
        used by the soft bounce-back and the soft force is exactly
        ``1 - w`` (same tensor arithmetic, no independent sigmoid).
        """
        phi = self.sdf(nz, ny, nx, dtype=dtype, device=device)
        eps = _scalar(self.epsilon, dtype, phi.device)
        ratio = phi / eps
        w = torch.sigmoid(ratio)
        one = torch.ones_like(w)
        zero = torch.zeros_like(w)
        return torch.where(ratio > _SATURATION, one, torch.where(ratio < -_SATURATION, zero, w))

    def hard_mask(
        self,
        nz: int,
        ny: int,
        nx: int,
        *,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Boolean inside-test ``phi <= 0`` (the ``epsilon -> 0`` limit mask)."""
        return self.sdf(nz, ny, nx, dtype=torch.float64, device=device) <= 0.0

    def solid_weight(
        self,
        nz: int,
        ny: int,
        nx: int,
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Solid weight ``1 - w`` — the mask argument of the soft force.

        Passing this field to ``autograd_path.obstacle_force`` realises the
        continuous momentum-exchange force derived in the module docstring;
        a boolean mask gives the hard Ladd wet-node force back.
        """
        return 1.0 - self.fluid_weight(nz, ny, nx, dtype=dtype, device=device)
