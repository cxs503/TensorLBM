"""Curved-surface immersed-boundary force integration (node ``P399W2-N04``).

The immersed-boundary kernels in :mod:`tensorlbm.ibm_force_transfer` provide the
marker↔node sparse operators, the two-pass direct forcing and the Eulerian
body-force field.  What they do *not* provide is the step that turns a set of
Lagrangian marker forces into a **surface integral**: their marker force is a
bare velocity deficit (``F_k = ρ (U_b − u_k)``) with no surface measure, so

* the Lagrangian budget ``Σ_k F_k`` is not a quadrature of the traction field
  and therefore carries no information about *where* on the surface the load
  sits, and
* on a non-uniform surface discretisation (equal-angle markers on an ellipse,
  ``(θ, φ)`` markers on a sphere, adaptive surface remeshing) the sum is biased
  towards the densely sampled regions.

This module adds the missing pieces, and keeps them auditable:

1. **Curved-surface discretisation with an exact measure** —
   :class:`SurfaceQuadrature` bundles marker positions, outward normals and
   per-marker measure elements ``Δs`` (2-D, arc length) or ``ΔA`` (3-D, area).
   The measure elements are produced by Gauss–Legendre integration of the
   parametric speed ``|x'(t)|`` per marker cell (:func:`curve_quadrature`), so
   ``Σ w_k`` reproduces the exact length / area of the curve / surface to
   round-off, independently of the marker count (:func:`circle_quadrature`,
   :func:`ellipse_quadrature`, :func:`sphere_quadrature`).  An equal-arc-length
   layout (:func:`resample_uniform_arc`) is available for moving boundaries,
   where marker spacing must not degrade over time.
2. **Force → traction → force round trip** — :meth:`SurfaceQuadrature.traction`
   converts marker forces to a traction field ``τ_k = F_k / Δs_k`` and
   :meth:`SurfaceQuadrature.forces` converts back.  The integral ``∮ τ dS`` is
   computed by quadrature and the *unweighted* marker sum (the legacy
   behaviour) is returned alongside it, so that the quadrature bias is measured
   instead of assumed (:meth:`SurfaceQuadrature.integrate`).
3. **Measure-consistent direct forcing** — :func:`curved_surface_forcing`
   implements Uhlmann's direct forcing *with* the Lagrangian volume element
   (``F_k = τ_k Δs_k``), optional Guo half-force correctors, and an optional
   **normal-projection pass** (``n_normal_passes``) that removes the residual
   wall-normal slip after the forcing.  That pass is the mass-conservation
   closure: it reduces the boundary mass flux ``∮ ρ (u − u_b)·n dS`` without
   touching the tangential (friction) traction.
4. **Momentum / mass ledgers** — :func:`momentum_budget` (Lagrangian vs
   Eulerian momentum injection), :func:`conservative_closure` (exact
   redistribution of the round-off residual of the spreading step) and
   :func:`surface_mass_flux` (dimensionless leakage rate) make the two
   conservation statements machine-checkable rather than "by construction".

Conventions
-----------
Lattice units throughout: ``dt = 1``, ``dx = 1``, ``ρ₀ = 1``.  A marker force
``F_k`` is a momentum change per time step, a traction ``τ_k = F_k / Δs_k`` has
units of momentum per time step per unit length (2-D) or area (3-D), and the
reported ``force_total`` is therefore ``Σ_k F_k`` — the same sign convention as
:func:`tensorlbm.ibm_force_transfer.direct_forcing`, i.e. ``−force_total`` is
the force *on* the body.

References
----------
Peskin, C. S. (2002). The immersed boundary method. *Acta Numerica* 11, 479.
Uhlmann, M. (2005). An immersed boundary method with direct forcing for the
    simulation of particulate flows. *J. Comput. Phys.* 209, 448.
Guo, Z., Zheng, C. & Shi, B. (2002). Discrete lattice effects on the forcing
    term in the lattice Boltzmann method. *Phys. Rev. E* 65, 046308.
Breugem, W.-P. (2012). A second-order accurate immersed boundary method for
    fully resolved simulations of particle-laden flows. *J. Comput. Phys.*
    231, 4469.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from .ibm_force_transfer import IBMStencil

__all__ = [
    "CYLINDER_CD_LITERATURE",
    "CurvedForcingResult",
    "SurfaceQuadrature",
    "circle_quadrature",
    "conservative_closure",
    "curve_quadrature",
    "curved_surface_forcing",
    "drag_coefficient",
    "ellipse_quadrature",
    "momentum_budget",
    "polyline_arc_weights",
    "resample_uniform_arc",
    "sphere_quadrature",
    "stokes_cylinder_cd",
    "surface_mass_flux",
]

_GL_X = torch.tensor(
    [
        -0.9602898564975363,
        -0.7966664774136267,
        -0.5255324099163290,
        -0.1834346424956498,
        0.1834346424956498,
        0.5255324099163290,
        0.7966664774136267,
        0.9602898564975363,
    ],
    dtype=torch.float64,
)
_GL_W = torch.tensor(
    [
        0.1012285362903763,
        0.2223810344533745,
        0.3137066458778873,
        0.3626837833783620,
        0.3626837833783620,
        0.3137066458778873,
        0.2223810344533745,
        0.1012285362903763,
    ],
    dtype=torch.float64,
)


# --------------------------------------------------------------------------- #
# Surface quadrature container
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SurfaceQuadrature:
    """Markers on a curved surface plus their measure elements and normals.

    Attributes:
        points: ``(M, d)`` marker positions in lattice units.
        normals: ``(M, d)`` unit outward normals.
        weights: ``(M,)`` surface measure per marker (``Δs`` in 2-D, ``ΔA`` in
            3-D), all strictly positive.
        name: human-readable label (case name / discretisation).
        reference_measure: exact measure of the underlying curve/surface when
            known analytically (perimeter ``2πR``, area ``4πR²``, …).
    """

    points: torch.Tensor
    normals: torch.Tensor
    weights: torch.Tensor
    name: str = ""
    reference_measure: float | None = None

    def __post_init__(self) -> None:
        pts = torch.as_tensor(self.points, dtype=torch.float64)
        nrm = torch.as_tensor(self.normals, dtype=torch.float64)
        w = torch.as_tensor(self.weights, dtype=torch.float64).reshape(-1)
        if pts.dim() != 2 or nrm.shape != pts.shape or w.numel() != pts.shape[0]:
            raise ValueError("points/normals must be (M, d) and weights (M,)")
        if not bool(torch.all(w > 0.0)):
            raise ValueError("surface measure elements must be strictly positive")
        object.__setattr__(self, "points", pts)
        object.__setattr__(self, "normals", nrm)
        object.__setattr__(self, "weights", w)

    # -- basic geometry ----------------------------------------------------- #
    @property
    def n_markers(self) -> int:
        return int(self.points.shape[0])

    @property
    def dim(self) -> int:
        return int(self.points.shape[1])

    @property
    def measure(self) -> float:
        """``Σ_k w_k`` — the discretised length (2-D) or area (3-D)."""
        return float(self.weights.sum().item())

    @property
    def measure_rel_error(self) -> float | None:
        """Relative measure error against the analytic reference (if known)."""
        if self.reference_measure is None:
            return None
        return abs(self.measure - self.reference_measure) / abs(self.reference_measure)

    @property
    def spacing(self) -> tuple[float, float]:
        """Marker spacing ``(min, max)`` — uniformity diagnostic for remeshing."""
        w = self.weights
        return float(w.min().item()), float(w.max().item())

    def with_points(
        self, points: torch.Tensor, normals: torch.Tensor
    ) -> "SurfaceQuadrature":
        """Same measure elements, new positions (rigid motion / deformation)."""
        return SurfaceQuadrature(
            points=points,
            normals=normals,
            weights=self.weights,
            name=self.name,
            reference_measure=self.reference_measure,
        )

    # -- force <-> traction ------------------------------------------------- #
    def traction(self, marker_force: torch.Tensor) -> torch.Tensor:
        """Traction ``τ_k = F_k / w_k`` from marker forces ``(M, d)``."""
        F = torch.as_tensor(marker_force, dtype=torch.float64)
        return F / self.weights.unsqueeze(-1)

    def forces(self, traction: torch.Tensor) -> torch.Tensor:
        """Marker forces ``F_k = τ_k w_k`` from a traction field ``(M, d)``."""
        tau = torch.as_tensor(traction, dtype=torch.float64)
        return tau * self.weights.unsqueeze(-1)

    def integrate(self, traction: torch.Tensor, *, weighted: bool = True) -> torch.Tensor:
        """``∮ τ n dS`` — quadrature over the surface (or the unweighted sum)."""
        tau = torch.as_tensor(traction, dtype=torch.float64)
        if not weighted:
            return tau.sum(dim=0)
        return (tau * self.weights.unsqueeze(-1)).sum(dim=0)

    def torque(
        self, marker_force: torch.Tensor, about: Sequence[float] | None = None
    ) -> torch.Tensor:
        """Surface torque ``∮ (x − x_c) × t dS`` from marker forces."""
        F = torch.as_tensor(marker_force, dtype=torch.float64)
        c = (
            torch.zeros(self.dim, dtype=torch.float64)
            if about is None
            else torch.as_tensor(about, dtype=torch.float64)
        )
        rel = self.points - c
        if self.dim == 2:
            return torch.tensor(
                (rel[:, 0] * F[:, 1] - rel[:, 1] * F[:, 0]).sum().item(),
                dtype=torch.float64,
            )
        return torch.cross(rel, F, dim=1).sum(dim=0)

    def normal_tangential_split(self, marker_force: torch.Tensor) -> dict:
        """Split the surface load into wall-normal and tangential parts.

        For an IBM direct-forcing constraint the wall-normal part of the marker
        force is the discrete counterpart of the pressure load and the
        tangential part the counterpart of the viscous (friction) load — the
        classical pressure-vs-friction decomposition of the drag.
        """
        F = torch.as_tensor(marker_force, dtype=torch.float64)
        F_n = (F * self.normals).sum(dim=1, keepdim=True) * self.normals
        F_t = F - F_n
        total = float(F.sum().item())
        return {
            "normal": [float(v) for v in F_n.sum(dim=0).tolist()],
            "tangential": [float(v) for v in F_t.sum(dim=0).tolist()],
            "normal_fraction": (
                float(F_n.sum().item()) / total if abs(total) > 0.0 else 0.0
            ),
        }

    def audit(self, marker_force: torch.Tensor) -> dict:
        """Measure-consistency audit of a force field on this quadrature.

        ``quadrature`` is ``Σ_k F_k`` (the exact momentum statement used by the
        solver) and ``traction_integral`` is ``∮ (F_k/w_k) dS`` evaluated with
        the quadrature weights — the two coincide by construction.
        ``unweighted_sum`` is the legacy estimate that treats every marker as
        carrying the *mean* measure, i.e. the biased quantity this module
        exists to replace.
        """
        F = torch.as_tensor(marker_force, dtype=torch.float64)
        quadrature = F.sum(dim=0)
        traction_integral = self.integrate(self.traction(F))
        unweighted = self.integrate(self.traction(F), weighted=False) * float(
            self.weights.mean().item()
        )
        return {
            "quadrature": [float(v) for v in quadrature.tolist()],
            "traction_integral": [float(v) for v in traction_integral.tolist()],
            "unweighted_sum": [float(v) for v in unweighted.tolist()],
            "quadrature_vs_traction_rel": _rel(traction_integral, quadrature),
            "unweighted_bias_rel": _rel(unweighted, quadrature),
            "spacing_min": self.spacing[0],
            "spacing_max": self.spacing[1],
            "spacing_ratio": self.spacing[1] / self.spacing[0],
            "measure": self.measure,
            "measure_rel_error": self.measure_rel_error,
        }


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    num = float((a - b).abs().max().item())
    den = float(b.abs().max().item())
    return num / den if den > 0.0 else num


# --------------------------------------------------------------------------- #
# Surface discretisation
# --------------------------------------------------------------------------- #


def curve_quadrature(
    curve: Callable[[torch.Tensor], torch.Tensor],
    n_markers: int,
    *,
    t0: float = 0.0,
    t1: float = 2.0 * math.pi,
    name: str = "curve",
    reference_measure: float | None = None,
) -> SurfaceQuadrature:
    """Midpoint markers + Gauss–Legendre measure on a closed parametric curve.

    Markers sit at the parameter midpoints ``t_k = t0 + (k + ½) Δt`` and each
    marker owns the *exact* arc length of its cell,

        ``Δs_k = ∫_{t_k − Δt/2}^{t_k + Δt/2} |x'(t)| dt``,

    evaluated with an 8-point Gauss–Legendre rule.  ``Σ_k Δs_k`` therefore
    equals the exact length of the curve to round-off for any ``n_markers``
    (for a circle the rule is exact because ``|x'| = R`` is constant), while
    the marker positions are second-order accurate.  The measure — the quantity
    that enters every force statement — never inherits the discretisation
    error of the marker positions.

    Args:
        curve: callable ``t (K,) -> (K, 2)`` parametric curve.
        n_markers: number of markers.
        t0, t1: parameter range; the curve is assumed closed.
        name: label stored on the result.
        reference_measure: analytic length for the measure audit, when known.
    """
    if n_markers < 4:
        raise ValueError("need at least 4 markers")
    dt = (t1 - t0) / n_markers
    dt_t = torch.tensor(dt, dtype=torch.float64)
    centres = t0 + (torch.arange(n_markers, dtype=torch.float64) + 0.5) * dt
    pts = curve(centres)

    # Measure: GL-8 per cell with t = centre + (Δt/2) s , s in [-1, 1].
    s = _GL_X.unsqueeze(0)
    w_gl = _GL_W.unsqueeze(0)
    t_nodes = centres.unsqueeze(1) + 0.5 * dt_t * s  # (M, 8)
    flat_t = t_nodes.reshape(-1)
    h = 1e-6 * max(abs(dt), 1e-6)
    dpdt = (curve(flat_t + h) - curve(flat_t - h)) / (2.0 * h)
    speed = torch.linalg.norm(dpdt, dim=-1).reshape(n_markers, -1)  # (M, 8)
    weights = (speed * w_gl).sum(dim=1) * (0.5 * dt_t)

    # Normals from the tangent at the marker positions.
    dp = (curve(centres + h) - curve(centres - h)) / (2.0 * h)
    tang = dp / torch.linalg.norm(dp, dim=-1, keepdim=True)
    # outward normal = tangent rotated by −90° for a counter-clockwise curve
    normals = torch.stack([tang[:, 1], -tang[:, 0]], dim=1)
    return SurfaceQuadrature(
        points=pts,
        normals=normals,
        weights=weights,
        name=name,
        reference_measure=reference_measure,
    )


def circle_quadrature(
    center: Sequence[float],
    radius: float,
    n_markers: int,
    *,
    layout: str = "equal_arc",
    name: str | None = None,
) -> SurfaceQuadrature:
    """Markers on a circle with the exact arc-length measure.

    ``layout="equal_arc"`` is the analytic uniform discretisation
    (``Δs_k = 2πR/M`` exactly, ``Σ Δs = 2πR`` to the last bit).
    ``layout="equal_angle"`` is identical for a circle and kept for symmetry
    with :func:`ellipse_quadrature`; ``layout="gauss"`` shifts each marker to
    the first Gauss node of its panel while keeping the exact panel length —
    the discretisation that breaks the "markers are the quadrature points"
    assumption an unweighted sum relies on.
    """
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    c = torch.as_tensor(center, dtype=torch.float64)
    dt = 2.0 * math.pi / n_markers
    base = (torch.arange(n_markers, dtype=torch.float64) + 0.5) * dt
    if layout in ("equal_arc", "equal_angle"):
        th = base
        w = torch.full((n_markers,), dt * radius, dtype=torch.float64)
    elif layout == "gauss":
        th = base + 0.5 * dt * float(_GL_X[0].item())
        w = torch.full((n_markers,), dt * radius, dtype=torch.float64)
    else:
        raise ValueError(f"unknown layout {layout!r}")
    pts = torch.stack([c[0] + radius * torch.cos(th), c[1] + radius * torch.sin(th)], dim=1)
    nrm = torch.stack([torch.cos(th), torch.sin(th)], dim=1)
    return SurfaceQuadrature(
        points=pts,
        normals=nrm,
        weights=w,
        name=name or f"circle_R{radius:g}_M{n_markers}_{layout}",
        reference_measure=2.0 * math.pi * radius,
    )


def ellipse_quadrature(
    center: Sequence[float],
    a: float,
    b: float,
    n_markers: int,
    *,
    layout: str = "equal_angle",
    name: str | None = None,
) -> SurfaceQuadrature:
    """Markers on an ellipse (non-uniform curvature, strongly non-uniform ``Δs``).

    ``layout="equal_angle"`` samples uniformly in the eccentric anomaly — the
    natural output of equal-step surface remeshing — which gives arc elements
    ``Δs ∝ sqrt(a² sin²θ + b² cos²θ)`` varying by the aspect ratio along the
    surface.  ``layout="equal_arc"`` redistributes the markers at constant
    ``Δs`` while keeping the exact measure.
    """
    if a <= 0.0 or b <= 0.0:
        raise ValueError("semi-axes must be positive")
    c = torch.as_tensor(center, dtype=torch.float64)
    ref = _ellipse_perimeter(a, b)

    def curve(t: torch.Tensor) -> torch.Tensor:
        return torch.stack([c[0] + a * torch.cos(t), c[1] + b * torch.sin(t)], dim=1)

    quad = curve_quadrature(
        curve,
        n_markers,
        name=name or f"ellipse_{a:g}x{b:g}_M{n_markers}_{layout}",
        reference_measure=ref,
    )
    if layout == "equal_angle":
        return quad
    if layout != "equal_arc":
        raise ValueError(f"unknown layout {layout!r}")
    arc = _cumulative_arc(curve, 0.0, 2.0 * math.pi, 8000)
    targets = (torch.arange(n_markers, dtype=torch.float64) + 0.5) * (
        quad.measure / n_markers
    )
    t_new = _invert_arclength(arc, targets)
    pts = curve(t_new)
    h = 1e-6
    dp = (curve(t_new + h) - curve(t_new - h)) / (2.0 * h)
    tang = dp / torch.linalg.norm(dp, dim=-1, keepdim=True)
    nrm = torch.stack([tang[:, 1], -tang[:, 0]], dim=1)
    return SurfaceQuadrature(
        points=pts,
        normals=nrm,
        weights=torch.full((n_markers,), quad.measure / n_markers, dtype=torch.float64),
        name=name or f"ellipse_{a:g}x{b:g}_M{n_markers}_equal_arc",
        reference_measure=ref,
    )


def sphere_quadrature(
    center: Sequence[float],
    radius: float,
    n_theta: int,
    n_phi: int,
    *,
    name: str | None = None,
) -> SurfaceQuadrature:
    """Markers on a sphere with the exact area measure ``ΔA = R² sinθ Δθ Δφ``.

    The ``sin θ`` factor is what an unweighted marker sum is missing: at fixed
    ``(θ, φ)`` resolution the polar rings hold as many markers as the
    equatorial ones while carrying a vanishingly small area.  Exact
    integration of ``sin θ`` per polar cell gives ``Σ ΔA = 4πR²`` to round-off
    for any resolution.
    """
    if radius <= 0.0 or n_theta < 2 or n_phi < 4:
        raise ValueError("invalid sphere resolution")
    c = torch.as_tensor(center, dtype=torch.float64)
    dth = math.pi / n_theta
    dph = 2.0 * math.pi / n_phi
    th_c = (torch.arange(n_theta, dtype=torch.float64) + 0.5) * dth
    lo = torch.arange(n_theta, dtype=torch.float64) * dth
    dsin = torch.cos(lo) - torch.cos(lo + dth)  # exact ∫_cell sinθ dθ
    th = th_c.unsqueeze(1).expand(n_theta, n_phi).reshape(-1)
    ph = (
        (torch.arange(n_phi, dtype=torch.float64) + 0.5)
        .unsqueeze(0)
        .expand(n_theta, n_phi)
        .reshape(-1)
        * dph
    )
    pts = torch.stack(
        [
            c[0] + radius * torch.sin(th) * torch.cos(ph),
            c[1] + radius * torch.sin(th) * torch.sin(ph),
            c[2] + radius * torch.cos(th),
        ],
        dim=1,
    )
    nrm = torch.stack(
        [torch.sin(th) * torch.cos(ph), torch.sin(th) * torch.sin(ph), torch.cos(th)],
        dim=1,
    )
    w = radius**2 * dsin.unsqueeze(1).expand(n_theta, n_phi).reshape(-1) * dph
    return SurfaceQuadrature(
        points=pts,
        normals=nrm,
        weights=w,
        name=name or f"sphere_R{radius:g}_{n_theta}x{n_phi}",
        reference_measure=4.0 * math.pi * radius**2,
    )


def polyline_arc_weights(points: torch.Tensor, *, closed: bool = True) -> torch.Tensor:
    """Chord-trapezoid arc elements for a marker polyline.

    ``Δs_k = ½ (|x_k − x_{k−1}| + |x_{k+1} − x_k|)`` — the only measure
    available when the marker positions are supplied directly (e.g. from an
    STL remesh).  It *underestimates* the true length by ``O((Δθ)²/24)`` per
    element for a circle, so it should be used only after
    :func:`resample_uniform_arc` or with a fine marker set;
    :func:`curve_quadrature` does not have this bias.
    """
    p = torch.as_tensor(points, dtype=torch.float64)
    if p.dim() != 2:
        raise ValueError("points must be (M, d)")
    if closed:
        nxt = torch.roll(p, shifts=-1, dims=0)
        prv = torch.roll(p, shifts=1, dims=0)
    else:
        nxt = torch.cat([p[1:], p[-1:]], dim=0)
        prv = torch.cat([p[:1], p[:-1]], dim=0)
    return 0.5 * (torch.linalg.norm(nxt - p, dim=1) + torch.linalg.norm(p - prv, dim=1))


def resample_uniform_arc(points: torch.Tensor, n_markers: int) -> torch.Tensor:
    """Redistribute markers at constant arc length along a closed 2-D polyline."""
    p = torch.as_tensor(points, dtype=torch.float64)
    if p.dim() != 2 or p.shape[1] != 2:
        raise ValueError("resample_uniform_arc handles (M, 2) polylines")
    if n_markers < 4:
        raise ValueError("need at least 4 markers")
    closed = torch.cat([p, p[:1]], dim=0)
    seg = torch.linalg.norm(closed[1:] - closed[:-1], dim=1)
    s = torch.cat([torch.zeros(1, dtype=p.dtype), torch.cumsum(seg, dim=0)])
    total = float(s[-1].item())
    targets = torch.arange(n_markers, dtype=p.dtype) * (total / n_markers)
    idx = torch.searchsorted(s, targets, right=True).clamp(1, s.numel() - 1)
    frac = (targets - s[idx - 1]) / (s[idx] - s[idx - 1])
    return closed[idx - 1] + frac.unsqueeze(1) * (closed[idx] - closed[idx - 1])


def _cumulative_arc(
    curve: Callable[[torch.Tensor], torch.Tensor], t0: float, t1: float, n: int
) -> tuple[torch.Tensor, torch.Tensor]:
    t = torch.linspace(t0, t1, n, dtype=torch.float64)
    p = curve(t)
    seg = torch.linalg.norm(p[1:] - p[:-1], dim=1)
    return t, torch.cat([torch.zeros(1, dtype=p.dtype), torch.cumsum(seg, dim=0)])


def _invert_arclength(
    arc: tuple[torch.Tensor, torch.Tensor], targets: torch.Tensor
) -> torch.Tensor:
    t, s = arc
    idx = torch.searchsorted(s, targets, right=True).clamp(1, s.numel() - 1)
    frac = (targets - s[idx - 1]) / (s[idx] - s[idx - 1])
    return t[idx - 1] + frac * (t[idx] - t[idx - 1])


def _ellipe(m: float) -> float:
    """Complete elliptic integral of the second kind ``E(m)`` (AGM, m = k²)."""
    if not 0.0 <= m < 1.0:
        raise ValueError("m must be in [0, 1)")
    a, b = 1.0, math.sqrt(1.0 - m)
    c = math.sqrt(m)
    s = 0.5 * c * c  # n = 0 term: 2^{-1} c_0^2
    p = 1.0
    for _ in range(60):
        c = 0.5 * (a - b)  # c_n from the *previous* a, b (A&S 17.6.4)
        a_next = 0.5 * (a + b)
        b = math.sqrt(a * b)
        a = a_next
        s += p * c * c
        p *= 2.0
        if abs(c) < 1e-18:
            break
    return (math.pi / (2.0 * a)) * (1.0 - s)


def _ellipse_perimeter(a: float, b: float) -> float:
    """Exact perimeter of an ellipse, ``4 a E(e)`` with ``e² = 1 − b²/a²``."""
    if a >= b:
        m = 1.0 - (b / a) ** 2
        return 4.0 * a * _ellipe(min(m, 1.0 - 1e-15))
    m = 1.0 - (a / b) ** 2
    return 4.0 * b * _ellipe(min(m, 1.0 - 1e-15))


# --------------------------------------------------------------------------- #
# Measure-consistent direct forcing
# --------------------------------------------------------------------------- #


@dataclass
class CurvedForcingResult:
    """Output of :func:`curved_surface_forcing` with its conservation audits."""

    marker_force: torch.Tensor  # (M, d)  Σ_k F_k = momentum injected per step
    traction: torch.Tensor  # (M, d)  F_k / w_k
    force_grid: torch.Tensor  # (d, *shape) Eulerian forcing actually applied
    force_total: torch.Tensor  # (d,)  Σ_k F_k (Lagrangian budget)
    force_euler: torch.Tensor  # (d,)  Σ_x f_grid (Eulerian budget)
    closure_rel: float  # |F_euler − F_total| / |F_total|
    normal_slip_rel: float  # boundary mass flux / (ρ_ref |u_ref| measure)
    tangential_slip_rel: float
    n_correctors: int
    n_normal_passes: int
    scheme: str
    audit: dict

    def as_dict(self) -> dict:
        return {
            "force_total": [float(v) for v in self.force_total.tolist()],
            "force_euler": [float(v) for v in self.force_euler.tolist()],
            "closure_rel": self.closure_rel,
            "normal_slip_rel": self.normal_slip_rel,
            "tangential_slip_rel": self.tangential_slip_rel,
            "n_correctors": self.n_correctors,
            "n_normal_passes": self.n_normal_passes,
            "scheme": self.scheme,
            "audit": self.audit,
        }


def curved_surface_forcing(
    stencil: IBMStencil,
    u_grid: torch.Tensor,
    u_target: torch.Tensor,
    surface: SurfaceQuadrature,
    *,
    rho_grid: torch.Tensor | None = None,
    n_correctors: int = 2,
    n_normal_passes: int = 0,
    scheme: str = "quadrature",
    velocity_scale: float | None = None,
) -> CurvedForcingResult:
    """Direct forcing on a curved surface with measure-consistent marker forces.

    Extends :func:`tensorlbm.ibm_force_transfer.direct_forcing` with the
    Lagrangian measure element of Uhlmann (2005),

        ``τ_k = ρ_k (U_b,k − ũ_k)``,   ``F_k = τ_k Δs_k``,
        ``f(x) = Σ_k F_k δ_h(x − X_k)``,   ``Σ_x f(x) = Σ_k F_k``,

    so that the traction ``τ_k``, not the marker force, is the resolution
    independent quantity and ``∮ τ dS`` is a genuine surface integral.  The
    legacy convention (no measure element, ``F_k = τ_k``) is available as
    ``scheme="raw"`` for baseline comparison; it reproduces
    :func:`~tensorlbm.ibm_force_transfer.direct_forcing` exactly.

    Args:
        stencil: marker stencil built from ``surface.points``.
        u_grid: ``(d, *shape)`` Eulerian velocity (Guo-corrected).
        u_target: ``(M, d)`` prescribed boundary velocity ``U_b``.
        surface: surface quadrature supplying ``Δs`` and the outward normals.
        rho_grid: optional ``(*shape,)`` density; defaults to ``ρ = 1``.
        n_correctors: Guo half-force corrector passes (as in ``direct_forcing``).
        n_normal_passes: normal-projection passes (Breugem 2012 style): the
            residual wall-normal slip is cancelled in the traction before the
            force is spread — the mass-conservation closure.
        scheme: ``"quadrature"`` (measure-consistent) or ``"raw"`` (legacy).
        velocity_scale: reference velocity for the normalised slip report
            (defaults to ``max|U_b|``).

    Returns:
        :class:`CurvedForcingResult` with marker forces, traction, the Eulerian
        forcing field and the conservation audits.
    """
    if scheme not in ("quadrature", "raw"):
        raise ValueError("scheme must be 'quadrature' or 'raw'")
    ub = torch.as_tensor(u_target, dtype=torch.float64).reshape(
        surface.n_markers, surface.dim
    )
    w = surface.weights if scheme == "quadrature" else torch.ones_like(surface.weights)
    rho_m = (
        torch.ones(surface.n_markers, dtype=torch.float64)
        if rho_grid is None
        else stencil.interpolate(rho_grid).reshape(-1)
    )
    u_m = stencil.interpolate(u_grid)  # (d, M)
    tau = ub.T - u_m  # (d, M), dt = 1

    for _ in range(max(0, n_correctors)):
        u_phys = u_m + 0.5 * tau / rho_m.unsqueeze(0)
        tau = tau + (ub.T - u_phys)

    for _ in range(max(0, n_normal_passes)):
        u_phys = u_m + 0.5 * tau / rho_m.unsqueeze(0)
        slip_n = ((u_phys - ub.T) * surface.normals.T).sum(dim=0)  # (M,)
        tau = tau - slip_n.unsqueeze(0) * surface.normals.T

    u_phys = u_m + 0.5 * tau / rho_m.unsqueeze(0)
    slip = u_phys - ub.T
    slip_n = (slip * surface.normals.T).sum(dim=0)
    slip_t = torch.linalg.norm(slip - slip_n.unsqueeze(0) * surface.normals.T, dim=0)
    u_scale = (
        float(ub.abs().max().item())
        if velocity_scale is None
        else float(velocity_scale)
    ) or 1.0
    rho_scale = float(rho_m.mean().item()) or 1.0
    measure = surface.measure

    marker_force = tau.T * w.unsqueeze(-1)  # (M, d)
    force_grid = stencil.spread(marker_force)
    force_total = marker_force.sum(dim=0)
    force_euler = force_grid.reshape(force_grid.shape[0], -1).sum(dim=1)
    closure_rel = _rel(force_euler, force_total)
    traction = marker_force / w.unsqueeze(-1)

    audit = {
        "n_markers": surface.n_markers,
        "measure": measure,
        "measure_rel_error": surface.measure_rel_error,
        "spacing_min": surface.spacing[0],
        "spacing_max": surface.spacing[1],
        "partition_error": abs(stencil.peak_normalization() - 1.0),
        "normal_slip_abs": float(torch.abs(w * rho_m * slip_n).sum().item()),
        "scheme_weight_mean": float(w.mean().item()),
        "rho_mean": rho_scale,
    }
    return CurvedForcingResult(
        marker_force=marker_force,
        traction=traction,
        force_grid=force_grid,
        force_total=force_total,
        force_euler=force_euler,
        closure_rel=closure_rel,
        normal_slip_rel=float(torch.abs(w * rho_m * slip_n).sum().item())
        / (rho_scale * u_scale * measure),
        tangential_slip_rel=float(torch.abs(w * rho_m * slip_t).sum().item())
        / (rho_scale * u_scale * measure),
        n_correctors=int(n_correctors),
        n_normal_passes=int(n_normal_passes),
        scheme=scheme,
        audit=audit,
    )


# --------------------------------------------------------------------------- #
# Conservation ledgers
# --------------------------------------------------------------------------- #


def momentum_budget(
    result: CurvedForcingResult, *, target: torch.Tensor | None = None
) -> dict:
    """Lagrangian vs Eulerian momentum injection per time step.

    The spreading operator has partition of unity, so ``Σ_x f(x) = Σ_k F_k`` is
    an *identity* up to round-off; ``closure_rel`` therefore measures
    floating-point loss only.  Passing ``target`` (e.g. the drive force
    ``ρ g N`` of a periodic channel) additionally measures the physical
    momentum balance between the body load and the imposed forcing.
    """
    out = {
        "force_total": [float(v) for v in result.force_total.tolist()],
        "force_euler": [float(v) for v in result.force_euler.tolist()],
        "closure_abs": float((result.force_euler - result.force_total).abs().max().item()),
        "closure_rel": result.closure_rel,
    }
    if target is not None:
        tgt = torch.as_tensor(target, dtype=torch.float64)
        out["target"] = [float(v) for v in tgt.tolist()]
        out["vs_target_rel"] = _rel(result.force_total, tgt)
    return out


def conservative_closure(
    force_grid: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, dict]:
    """Redistribute the spreading residual so the grid budget is exact.

    ``force_grid`` is ``(d, *shape)`` and ``target`` is ``(d,)``.  The residual
    ``r = target − Σ_x force_grid`` is distributed over the nodes that already
    carry forcing, proportionally to their own magnitude ``|f|`` (per
    component), so the correction is *localised on the body* and leaves the
    far field untouched.  The residual is O(1e-16) |F| in practice — this is a
    bookkeeping closure, not a physical correction, and the function reports
    both the before/after numbers so that the claim can be checked.

    Returns:
        ``(corrected, audit)`` with ``audit["residual_before_rel"]``,
        ``audit["residual_after_rel"]`` and the number of nodes touched.
    """
    f = torch.as_tensor(force_grid, dtype=torch.float64).clone()
    d = f.shape[0]
    tgt = torch.as_tensor(target, dtype=torch.float64).reshape(d)
    flat = f.reshape(d, -1)
    before = flat.sum(dim=1)
    scale = float(tgt.abs().max().item()) or 1.0
    residual_before = float((before - tgt).abs().max().item()) / scale
    touched = 0
    for c in range(d):
        r = float(tgt[c].item() - before[c].item())
        if r == 0.0:
            continue
        mag = flat[c].abs()
        support = mag > 0.0
        n_sup = int(support.sum().item())
        if n_sup == 0:
            # no support: dump everything on the maximum-velocity node
            idx = int(flat[c].abs().argmax().item())
            flat[c, idx] += r
            touched += 1
            continue
        weights = torch.where(support, mag, torch.zeros_like(mag))
        weights = weights / weights.sum()
        idx = int(mag.argmax().item())
        flat[c] += r * weights
        # absorb the float remainder on the largest node
        flat[c, idx] += r - float((r * weights).sum().item())
        touched += n_sup
    after = flat.sum(dim=1)
    residual_after = float((after - tgt).abs().max().item()) / scale
    audit = {
        "target": [float(v) for v in tgt.tolist()],
        "sum_before": [float(v) for v in before.tolist()],
        "sum_after": [float(v) for v in after.tolist()],
        "residual_before_abs": float((before - tgt).abs().max().item()),
        "residual_after_abs": float((after - tgt).abs().max().item()),
        "residual_before_rel": residual_before,
        "residual_after_rel": residual_after,
        "nodes_touched": touched,
    }
    return f, audit


def surface_mass_flux(
    stencil: IBMStencil,
    surface: SurfaceQuadrature,
    u_grid: torch.Tensor,
    u_target: torch.Tensor,
    rho_grid: torch.Tensor | None = None,
    *,
    reference: float | None = None,
) -> dict:
    """Boundary mass flux ``∮ ρ (u − u_b)·n dS`` — the leakage audit.

    This is the direct statement of impermeability: with the exact surface
    measure, the flux is a quadrature of the normal velocity error on the
    immersed boundary.  It is normalised by ``reference`` (typically
    ``ρ₀ U₀ D``) to give the dimensionless leakage rate; the unweighted marker
    average is reported alongside so that the measure effect is visible.

    Returns:
        ``flux`` (lattice units), ``leak_rate`` (normalised, ``None`` without a
        reference), ``leak_rate_unweighted`` and the signed normal-velocity
        statistics.
    """
    ub = torch.as_tensor(u_target, dtype=torch.float64).reshape(
        surface.n_markers, surface.dim
    )
    u_m = stencil.interpolate(u_grid).T  # (M, d)
    rho_m = (
        torch.ones(surface.n_markers, dtype=torch.float64)
        if rho_grid is None
        else stencil.interpolate(rho_grid).reshape(-1)
    )
    un = ((u_m - ub) * surface.normals).sum(dim=1)
    weighted = rho_m * un * surface.weights
    flux = float(weighted.sum().item())
    abs_flux = float(torch.abs(weighted).sum().item())
    unweighted = float((rho_m * un).mean().item()) * surface.measure
    out = {
        "flux": flux,
        "abs_flux": abs_flux,
        "flux_unweighted": unweighted,
        "leak_rate": abs(flux) / reference if reference else None,
        "leak_rate_abs": abs_flux / reference if reference else None,
        "leak_rate_unweighted": abs(unweighted) / reference if reference else None,
        "un_normal_max": float(torch.abs(un).max().item()),
        "un_normal_rms": float(un.pow(2).mean().sqrt().item()),
        "measure": surface.measure,
    }
    return out


# --------------------------------------------------------------------------- #
# Reference drag values
# --------------------------------------------------------------------------- #

CYLINDER_CD_LITERATURE: dict[str, dict[str, float | str]] = {
    "re100_rajani2009": {
        "cd": 1.34,
        "re": 100.0,
        "source": "Rajani, Kandasamy & Majumdar (2009), Appl. Math. Modelling 33, "
        "suppl. — LBM/BGK, unconfined cylinder, mean Cd over shedding cycle",
    },
    "re100_braza1986": {
        "cd": 1.36,
        "re": 100.0,
        "source": "Braza, Chassaing & Ha Minh (1986), J. Fluid Mech. 165, 79 — "
        "finite-volume, unconfined",
    },
    "re100_tritton1959": {
        "cd": 1.26,
        "re": 100.0,
        "source": "Tritton (1959), J. Fluid Mech. 6, 547 — experiment",
    },
}


def stokes_cylinder_cd(reynolds: float) -> dict:
    """Low-Reynolds drag of a circular cylinder (asymptotic analytic reference).

    The steady 2-D Stokes problem past a cylinder has no solution decaying at
    infinity (Stokes paradox), so the analytic anchor is the Oseen /
    Kaplun–Lagerstrom matched expansion

        ``Cd = 8π / (Re [2.002 − ln Re])``,

    whose ``O(Re² ln Re)`` relative error is a few percent for
    ``Re ≲ 0.5`` and grows quickly above ``Re ≈ 1``.
    """
    if reynolds <= 0.0:
        raise ValueError("Re must be positive")
    denom = reynolds * (2.002 - math.log(reynolds))
    cd = 8.0 * math.pi / denom
    return {
        "re": float(reynolds),
        "cd": float(cd),
        "formula": "Cd = 8*pi / (Re * (2.002 - ln Re))",
        "validity": "matched Oseen/Kaplun-Lagerstrom expansion, O(Re^2 ln Re) error",
    }


def drag_coefficient(force: torch.Tensor | Sequence[float], rho: float, u_ref: float, length: float) -> float:
    """``Cd = 2 F / (ρ U² D)`` (2-D, per unit span) from a total force vector."""
    f = torch.as_tensor(force, dtype=torch.float64)
    fx = float(f[0].item())
    return 2.0 * fx / (rho * u_ref * u_ref * length)