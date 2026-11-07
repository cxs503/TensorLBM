"""Galilean-invariant kinetic-flux control-surface (GCS) force estimator.

Why this module exists
----------------------
TensorLBM has three historical routes to the hydrodynamic force on a curved
(staircase) body, and two of them are recorded as *open failures* in the gap
ledger (``benchmarks/gaps.md``):

* **G12** — pressure + friction integration on the *voxelised body surface*
  (``drag_pressure``): the near-wall cell-centre pressure misses the
  stagnation rise, giving ``cd_p = -49%`` on a D40 sphere at Re=100.
* **G15** — link-wise momentum exchange (``momentum_exchange``): the
  ``standard`` Ladd sum pairs populations from two different time levels
  (``f_q(x_f)`` + ``f_opp(x_s)``) and therefore leaves a large *additive*
  free-stream background uncancelled on a curved wall (+264.6% on the same
  case); the "background-subtracted" repair was measured to be a no-op
  (identical digits) and the supposed "Galilean" variant is a pure scalar
  rescale ``(1 + 1/(2τ))·standard`` (m2553).

This module is a **third, methodologically independent** route.  It does not
touch the body surface at all: it integrates the *conserved kinetic momentum
flux* over a closed, **analytically defined spherical control surface** that
merely *encloses* the body.

The estimator
-------------
For a steady, incompressible flow the momentum balance is

    ∂_t (ρ u_i) + ∂_j Π_ij = f_i ,
    Π_ij = ρ u_i u_j + p δ_ij + Π^neq_ij ,

so the divergence of the flux tensor is *exactly* the momentum equation (no
inertia approximation is made).  Integrating over the volume between the body
and a closed control surface S and using the divergence theorem, together
with ``u = 0`` on the body surface, gives

    F_body = -∮_S Π_ij n_j dA  -  d/dt ∫_V ρ u_i dV .           (1)

The key observation is that ``Π_ij`` is the **raw second moment** of the
populations::

    Π_ij = Σ_q f_q c_qi c_qj

with no reconstruction, no equilibrium split and no background model.  That
single line is what makes the estimator background-proof:

* A **spatially uniform** population state — the configuration in which the
  hydrostatic solution is a uniform stream — yields a *constant* Π, and a
  constant tensor contracted with the closed-surface normal integrates to
  zero because the quadrature carries the exact null space ``∮ n dA = 0``
  (enforced, see :func:`sphere_quadrature`).  Any free-stream / moving-frame
  background is therefore annihilated *by construction*, to machine
  precision, instead of being estimated and subtracted (the G15 failure).
* The surface is an **analytic sphere**, not the voxel wall: its normals and
  areas carry no staircase error, and its radius is a free parameter
  (``r_s``).  Equation (1) is *radius independent* in the exact solution, so
  ``r_s`` gives a built-in self-consistency check that no point-wise wall
  method has (the G12 failure is exactly a surface-geometry/one-sided-
  difference error).

Discretisation: ``Π`` is evaluated on the lattice (cell centres), trilinearly
interpolated to the quadrature points (exact for linear fields, second-order
otherwise) and integrated with a tensor-product Gauss-Legendre (polar) ×
uniform (azimuthal) quadrature that is spectrally accurate for smooth
integrands.  The quadrature point set is symmetric under ``x → -x``,
``y → -y``, ``z → -z`` *separately*, which is what makes the null space
exact rather than approximate.

Known limits (see ``docs/sphere_settling_gcs_20261107.md``)
-----------------------------------------------------------
* The control surface must enclose the body with at least ~1.5 cells of
  clearance so that the trilinear stencil stays in fluid; it must stay
  inside the domain (or be treated as periodic — see ``periodic``).
* The estimator measures the *momentum flux*; at finite Re the flux over a
  finite surface equals the body force only up to the (exactly zero)
  accumulation term, i.e. it is steady-state only.  The unsteady term is
  provided separately (:func:`control_volume_momentum`) for diagnostics.
* The *geometric representation* of the body is still the staircase (the
  flow field is what it is), so the measured drag inherits the effective-
  radius error of the voxel body.  That error is a property of the body
  representation, not of this instrument; it is quantified by grid
  refinement in the benchmark that accompanies this module.

References
----------
Ladd, A. J. C. (1994), *J. Fluid Mech.* 271 — link-wise momentum exchange.
Hasimoto, H. (1959), *J. Fluid Mech.* 5 — periodic-sphere drag correction.
Faxén, H. (1922/1927) — wall corrections for a sphere in a cylinder.
Guo, Z.; Zheng, C.; Shi, B. (2002), *Phys. Rev. E* 65 — forcing scheme.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .d3q19 import C

__all__ = [
    "Quadrature",
    "ControlSurfaceForce",
    "ControlSurfaceForceTracker",
    "gauss_legendre",
    "sphere_quadrature",
    "momentum_flux_tensor",
    "trilinear_sample",
    "control_surface_force",
    "control_volume_momentum",
    "stokes_drag",
    "stokes_terminal_velocity",
    "hasimoto_factor",
]

_CS2 = 1.0 / 3.0
#: (i, j) index pairs of the six independent components of a symmetric tensor.
_TENSOR_PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


# ---------------------------------------------------------------------------
# 1. Quadrature on the sphere
# ---------------------------------------------------------------------------
def gauss_legendre(n: int, dtype: torch.dtype = torch.float64) -> tuple[torch.Tensor, torch.Tensor]:
    """Gauss-Legendre nodes/weights on ``[-1, 1]`` (Newton iteration, torch only).

    Nodes are returned in ascending order and mirrored **exactly**
    (``x[n-1-i] == -x[i]``) for even ``n``; the mirroring is what gives the
    spherical quadrature of :func:`sphere_quadrature` an exact null space.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    half = (n + 1) // 2
    xs: list[float] = []
    ws: list[float] = []
    for i in range(1, half + 1):
        x = math.cos(math.pi * (i - 0.25) / (n + 0.5))
        for _ in range(100):
            p0, p1 = 1.0, x
            for k in range(2, n + 1):
                p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
            dp = n * (x * p1 - p0) / (x * x - 1.0)
            dx = -p1 / dp
            x += dx
            if abs(dx) < 1e-16:
                break
        p0, p1 = 1.0, x
        for k in range(2, n + 1):
            p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
        dp = n * (x * p1 - p0) / (x * x - 1.0)
        xs.append(x)
        ws.append(2.0 / ((1.0 - x * x) * dp * dp))
    # ascending order with exact mirror symmetry about 0
    pos = xs[::-1]
    wpos = ws[::-1]
    if n % 2 == 0:
        nodes = [-p for p in pos[::-1]] + pos
        weights = wpos[::-1] + wpos
    else:
        nodes = [-p for p in pos[1:][::-1]] + [0.0] + pos[1:]
        weights = wpos[1:][::-1] + [wpos[0]] + wpos[1:]
    return (
        torch.tensor(nodes, dtype=dtype),
        torch.tensor(weights, dtype=dtype),
    )


@dataclass(frozen=True)
class Quadrature:
    """Points, outward normals and area weights of a spherical control surface.

    ``points`` are in **lattice index coordinates** (cell ``(k, j, i)`` of a
    ``(nz, ny, nx)`` array sits at ``(x, y, z) = (i, j, k)``), ``normals`` are
    unit vectors and ``weights`` are areas, so ``Σ weights * normals = 0`` and
    ``Σ weights = 4 π r_s²``.
    """

    points: torch.Tensor
    normals: torch.Tensor
    weights: torch.Tensor
    radius: float
    n_theta: int
    n_phi: int

    @property
    def n_points(self) -> int:
        return int(self.normals.shape[0])

    def null_space_residual(self) -> dict[str, float]:
        """Relative residuals of the exact null-space identities (diagnostics)."""
        w = self.weights
        n = self.normals
        area = 4.0 * math.pi * self.radius**2
        nn = torch.einsum("p,pi,pj->ij", w, n, n)
        # ∮ x_i n_j dA = V δ_ij  (V = 4/3 π r³) — check of the moments
        xn = torch.einsum("p,pi,pj->ij", w, self.points, n)
        vol = 4.0 / 3.0 * math.pi * self.radius**3
        return {
            "sum_n_over_area": float((n * w[:, None]).sum().abs()) / area,
            "sum_outer_n_rel": float((nn - nn.trace() / 3.0 * torch.eye(3, dtype=nn.dtype)).abs().max())
            / (area / 3.0),
            "sum_xn_rel": float((xn - vol * torch.eye(3, dtype=xn.dtype)).abs().max())
            / (4.0 * math.pi * self.radius**3),
        }


def sphere_quadrature(
    radius: float,
    *,
    centre: tuple[float, float, float] = (0.0, 0.0, 0.0),
    n_theta: int = 16,
    n_phi: int = 32,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> Quadrature:
    """Tensor-product quadrature on a sphere of radius ``radius``.

    ``mu = cos θ`` uses ``n_theta`` Gauss-Legendre nodes (with ``n_theta``
    even, so the set is exactly symmetric under ``mu → -mu``), ``φ`` uses
    ``n_phi`` equispaced nodes (exact symmetry under ``φ → φ + π`` and
    ``φ → -φ``).  The combined point set is therefore invariant under
    ``x → -x``, ``y → -y`` **and** ``z → -z`` separately, so any constant
    flux tensor integrates to zero to machine precision — the property that
    removes the free-stream background from the force (the G15 root cause).

    Accuracy is spectral for analytic integrands (Gauss-Legendre in ``mu``,
    trapezoidal in ``φ``); ``n_theta * n_phi = 16 x 32 = 512`` points is
    enough for ``< 1e-4`` relative error on smooth fields.
    """
    if radius <= 0:
        raise ValueError("radius must be positive")
    if n_theta % 2 or n_phi % 2:
        raise ValueError("n_theta and n_phi must be even (exact mirror symmetry)")
    mu, w_mu = gauss_legendre(n_theta, dtype=dtype)  # ∫_{-1}^{1}
    phi = 2.0 * math.pi * torch.arange(n_phi, dtype=dtype) / n_phi
    w_phi = torch.full((n_phi,), 2.0 * math.pi / n_phi, dtype=dtype)

    st = torch.sqrt(torch.clamp(1.0 - mu * mu, min=0.0))
    # (n_theta, n_phi, 3)
    nx = st[:, None] * torch.cos(phi)[None, :]
    ny = st[:, None] * torch.sin(phi)[None, :]
    nz = mu[:, None].expand(-1, n_phi)
    normals = torch.stack([nx, ny, nz], dim=-1).reshape(-1, 3)
    weights = (w_mu[:, None] * w_phi[None, :]).reshape(-1) * radius**2

    if device is not None:
        normals = normals.to(device)
        weights = weights.to(device)
    centre_t = torch.as_tensor(centre, dtype=dtype, device=normals.device)
    points = normals * radius + centre_t
    return Quadrature(
        points=points,
        normals=normals,
        weights=weights,
        radius=float(radius),
        n_theta=int(n_theta),
        n_phi=int(n_phi),
    )


# ---------------------------------------------------------------------------
# 2. Kinetic momentum-flux tensor and interpolation
# ---------------------------------------------------------------------------
def momentum_flux_tensor(
    f: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Raw second moment ``Π_ij = Σ_q f_q c_qi c_qj`` for D3Q19.

    Returns a ``(6, nz, ny, nx)`` tensor with components in the order
    ``(xx, yy, zz, xy, xz, yz)``.  This is the full momentum flux: it already
    contains the advective (``ρ u u``), pressure (``ρ c_s² δ``) and viscous
    (``Π^neq``) parts, so no macroscopic reconstruction is needed.
    """
    if f.ndim != 4 or f.shape[0] != 19:
        raise ValueError("populations must have shape (19, nz, ny, nx)")
    c = C.to(device=f.device, dtype=dtype)
    out = [
        (f * (c[:, i] * c[:, j]).view(19, 1, 1, 1)).sum(dim=0)
        for (i, j) in _TENSOR_PAIRS
    ]
    return torch.stack(out)


def _flat_index(
    k: torch.Tensor, j: torch.Tensor, i: torch.Tensor, nz: int, ny: int, nx: int, periodic: bool
) -> torch.Tensor:
    if periodic:
        k = k % nz
        j = j % ny
        i = i % nx
    else:
        if (
            bool((k < 0).any())
            or bool((k >= nz).any())
            or bool((j < 0).any())
            or bool((j >= ny).any())
            or bool((i < 0).any())
            or bool((i >= nx).any())
        ):
            raise ValueError("quadrature point outside the sampling grid")
    return (k * ny + j) * nx + i


def trilinear_sample(
    field: torch.Tensor,
    points: torch.Tensor,
    *,
    periodic: bool = True,
) -> torch.Tensor:
    """Trilinearly interpolate ``field`` (``(nz, ny, nx)``) at ``points``.

    ``points`` are in lattice index coordinates, ``field[k, j, i]`` sits at
    ``(x, y, z) = (i, j, k)``.  Accumulation is carried out in float64, so a
    spatially constant field is reproduced to ~1e-16 relative even when the
    field itself is float32 — the property the null-space test relies on.
    With ``periodic=True`` the corners wrap around (a periodic box); with
    ``periodic=False`` points must lie at least one cell inside the grid.
    """
    if field.ndim != 3:
        raise ValueError("field must be (nz, ny, nx)")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be (N, 3)")
    nz, ny, nx = field.shape
    x = points[:, 0].to(torch.float64)
    y = points[:, 1].to(torch.float64)
    z = points[:, 2].to(torch.float64)
    i0 = torch.floor(x)
    j0 = torch.floor(y)
    k0 = torch.floor(z)
    tx = x - i0
    ty = y - j0
    tz = z - k0
    i0 = i0.to(torch.long)
    j0 = j0.to(torch.long)
    k0 = k0.to(torch.long)
    flat = field.reshape(-1)
    acc = torch.zeros(points.shape[0], dtype=torch.float64, device=points.device)
    for dk in (0, 1):
        wk = tz if dk else 1.0 - tz
        for dj in (0, 1):
            wj = ty if dj else 1.0 - ty
            for di in (0, 1):
                wi = tx if di else 1.0 - tx
                idx = _flat_index(k0 + dk, j0 + dj, i0 + di, nz, ny, nx, periodic)
                acc = acc + flat[idx].to(torch.float64) * (wk * wj * wi)
    return acc


# ---------------------------------------------------------------------------
# 3. The force itself
# ---------------------------------------------------------------------------
@dataclass
class ControlSurfaceForce:
    """Force estimate from one control-surface evaluation.

    Components are the **force on the body** in the drag-positive convention
    (a positive ``fx`` resists the +x flow).  ``*_pressure``, ``*_viscous``
    and ``*_advective`` split the same integral into the ``ρ c_s² δ``,
    ``Π^neq`` and ``ρ u u`` parts of the flux — a diagnostic decomposition,
    not three different methods.
    """

    fx: float
    fy: float
    fz: float
    fx_pressure: float = 0.0
    fy_pressure: float = 0.0
    fz_pressure: float = 0.0
    fx_viscous: float = 0.0
    fy_viscous: float = 0.0
    fz_viscous: float = 0.0
    fx_advective: float = 0.0
    fy_advective: float = 0.0
    fz_advective: float = 0.0
    radius: float = 0.0
    n_points: int = 0

    def as_dict(self) -> dict[str, float]:
        d = {"fx": self.fx, "fy": self.fy, "fz": self.fz, "radius": self.radius}
        for comp in ("pressure", "viscous", "advective"):
            for axis in "xyz":
                d[f"f{axis}_{comp}"] = getattr(self, f"f{axis}_{comp}")
        return d


def control_surface_force(
    f: torch.Tensor,
    centre: tuple[float, float, float],
    radius: float,
    *,
    n_theta: int = 16,
    n_phi: int = 32,
    periodic: bool = True,
    decompose: bool = True,
    body_force: tuple[float, float, float] | None = None,
) -> ControlSurfaceForce:
    """Force on the body enclosed by a spherical control surface.

    Implements equation (1) with ``d/dt`` omitted (steady state):

        F = -∮_S Π_ij n_j dA ,  Π_ij = Σ_q f_q c_qi c_qj .

    Args:
        f:        Post-streaming populations ``(19, nz, ny, nx)``.
        centre:   Control-surface centre in lattice index coordinates.
        radius:   Control-surface radius (cells).  Must enclose the body with
                  ≳1.5 cells clearance; the result is radius independent in
                  the exact solution, so varying it is a self-check.
        n_theta:  Gauss-Legendre nodes in ``cos θ`` (even).
        n_phi:    Azimuthal nodes (even).
        periodic: Wrap interpolation across the domain faces (periodic box).
        decompose: Also return the pressure / viscous / advective split.
        body_force: Optional uniform force *density* ``f`` (force per unit
                 volume) acting on the fluid, e.g. gravity in a periodic
                 settling cell.  When given, the half-force momentum-flux
                 correction ``½(f_i u_j + u_i f_j)`` of the Guo forcing scheme
                 is added to ``Π`` so that the sampled flux is the one that
                 appears in the discrete momentum balance.

    Returns:
        :class:`ControlSurfaceForce`.
    """
    quad = sphere_quadrature(
        radius,
        centre=centre,
        n_theta=n_theta,
        n_phi=n_phi,
        dtype=torch.float64,
        device=f.device,
    )
    pi = momentum_flux_tensor(f, dtype=torch.float32)
    # (6, N) samples at the quadrature points
    samp = torch.stack(
        [trilinear_sample(pi[k], quad.points, periodic=periodic) for k in range(6)]
    )
    w = quad.weights
    n = quad.normals
    need_macro = decompose or body_force is not None
    c = C.to(device=f.device, dtype=torch.float64)
    if need_macro:
        rho = trilinear_sample(f.sum(dim=0), quad.points, periodic=periodic)
        ux = torch.stack(
            [trilinear_sample(f[q], quad.points, periodic=periodic) * float(c[q, 0]) for q in range(19)]
        ).sum(dim=0) / rho
        uy = torch.stack(
            [trilinear_sample(f[q], quad.points, periodic=periodic) * float(c[q, 1]) for q in range(19)]
        ).sum(dim=0) / rho
        uz = torch.stack(
            [trilinear_sample(f[q], quad.points, periodic=periodic) * float(c[q, 2]) for q in range(19)]
        ).sum(dim=0) / rho
        if body_force is not None:
            fx_b, fy_b, fz_b = (float(v) for v in body_force)
            samp = samp + torch.stack(
                [
                    0.5 * 2.0 * fx_b * ux,
                    fy_b * ux + fx_b * uy,
                    fz_b * ux + fx_b * uz,
                    0.5 * 2.0 * fy_b * uy,
                    fz_b * uy + fy_b * uz,
                    0.5 * 2.0 * fz_b * uz,
                ]
            )
    else:
        rho = ux = uy = uz = None
    pxx, pyy, pzz, pxy, pxz, pyz = samp
    # Π · n  (three components)
    flux_x = pxx * n[:, 0] + pxy * n[:, 1] + pxz * n[:, 2]
    flux_y = pxy * n[:, 0] + pyy * n[:, 1] + pyz * n[:, 2]
    flux_z = pxz * n[:, 0] + pyz * n[:, 1] + pzz * n[:, 2]
    out = ControlSurfaceForce(
        fx=-float((w * flux_x).sum()),
        fy=-float((w * flux_y).sum()),
        fz=-float((w * flux_z).sum()),
        radius=float(radius),
        n_points=quad.n_points,
    )
    if not decompose:
        return out

    p = rho * _CS2
    adv = torch.stack(
        [
            rho * ux * ux,
            rho * ux * uy,
            rho * ux * uz,
            rho * uy * uy,
            rho * uy * uz,
            rho * uz * uz,
        ]
    )
    for name, base in (("pressure", p), ("advective", adv)):
        if name == "pressure":
            bxx = byy = bzz = base
            bxy = bxz = byz = torch.zeros_like(base)
        else:
            bxx, bxy, bxz, byy, byz, bzz = base
        bx = bxx * n[:, 0] + bxy * n[:, 1] + bxz * n[:, 2]
        by = bxy * n[:, 0] + byy * n[:, 1] + byz * n[:, 2]
        bz = bxz * n[:, 0] + byz * n[:, 1] + bzz * n[:, 2]
        setattr(out, f"fx_{name}", -float((w * bx).sum()))
        setattr(out, f"fy_{name}", -float((w * by).sum()))
        setattr(out, f"fz_{name}", -float((w * bz).sum()))
    out.fx_viscous = out.fx - out.fx_pressure - out.fx_advective
    out.fy_viscous = out.fy - out.fy_pressure - out.fy_advective
    out.fz_viscous = out.fz - out.fz_pressure - out.fz_advective
    return out


def control_volume_momentum(
    f: torch.Tensor,
    centre: tuple[float, float, float],
    radius: float,
) -> torch.Tensor:
    """Total fluid momentum ``∫_V ρ u dV`` inside the control sphere.

    Used for the ``d/dt`` term of equation (1): differencing this value
    between samples measures the (transient) momentum accumulation that the
    steady-state estimator drops.  In the body-fixed frame of a settling
    sphere at terminal velocity it must vanish — a quantitative steadiness
    diagnostic.
    """
    nz, ny, nx = f.shape[1:]
    z, y, x = torch.meshgrid(
        torch.arange(nz, dtype=torch.float64, device=f.device),
        torch.arange(ny, dtype=torch.float64, device=f.device),
        torch.arange(nx, dtype=torch.float64, device=f.device),
        indexing="ij",
    )
    mask = (
        (x - centre[0]) ** 2 + (y - centre[1]) ** 2 + (z - centre[2]) ** 2
    ) <= radius**2
    rho = f.sum(dim=0)
    c = C.to(device=f.device, dtype=torch.float32)
    mom = [
        (f * c[:, i].view(19, 1, 1, 1)).sum(dim=0) for i in range(3)
    ]
    return torch.stack(
        [(m * mask).sum().to(torch.float64) for m in mom]
    )


# ---------------------------------------------------------------------------
# 4. Time-series tracker (mean force, oscillation amplitude, steadiness)
# ---------------------------------------------------------------------------
@dataclass
class ControlSurfaceForceTracker:
    """Accumulate control-surface force samples and report oscillation stats.

    The instantaneous force in a laminar, steady flow is *not* constant: it
    carries a small lattice oscillation whose amplitude is a useful quality
    indicator (it is the noise floor of any convergence claim).  This tracker
    reports the tail-window mean, the RMS deviation and the peak-to-peak
    amplitude, all normalised by the mean, plus the ``d/dt`` momentum drift.
    """

    centre: tuple[float, float, float]
    radius: float
    n_theta: int = 16
    n_phi: int = 32
    periodic: bool = True
    history: list[ControlSurfaceForce] = field(default_factory=list)
    momentum_history: list[torch.Tensor] = field(default_factory=list)

    def sample(self, f: torch.Tensor, *, with_momentum: bool = True) -> ControlSurfaceForce:
        res = control_surface_force(
            f,
            self.centre,
            self.radius,
            n_theta=self.n_theta,
            n_phi=self.n_phi,
            periodic=self.periodic,
        )
        self.history.append(res)
        if with_momentum:
            self.momentum_history.append(control_volume_momentum(f, self.centre, self.radius))
        return res

    # -- reporting -----------------------------------------------------
    def series(self, axis: str = "x") -> list[float]:
        return [float(getattr(r, f"f{axis}")) for r in self.history]

    def window(self, axis: str = "x", last: int = 50) -> list[float]:
        return self.series(axis)[-last:] if self.history else []

    def summary(self, axis: str = "x", last: int = 50) -> dict[str, float]:
        vals = self.window(axis, last)
        if not vals:
            return {}
        n = len(vals)
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / n
        return {
            "mean": mean,
            "rms": math.sqrt(var),
            "ptp": max(vals) - min(vals),
            "rms_over_mean": math.sqrt(var) / abs(mean) if mean else float("nan"),
            "ptp_over_mean": (max(vals) - min(vals)) / abs(mean) if mean else float("nan"),
            "n_samples": n,
        }

    def momentum_drift(self, dt_steps: int, last: int = 50) -> list[float]:
        """``d/dt ∫ρu dV`` between the first and last of the tail window."""
        if len(self.momentum_history) < 2:
            return [0.0, 0.0, 0.0]
        k = min(last, len(self.momentum_history))
        d = self.momentum_history[-1] - self.momentum_history[-k]
        return [float(v) / max(1, dt_steps * (k - 1)) for v in d]


# ---------------------------------------------------------------------------
# 5. Analytic references (Stokes settling)
# ---------------------------------------------------------------------------
def stokes_drag(radius: float, mu: float, velocity: float) -> float:
    """Stokes drag ``6 π μ a U`` (unbounded creeping flow past a sphere)."""
    return 6.0 * math.pi * mu * radius * velocity


def stokes_terminal_velocity(radius: float, mu: float, delta_rho: float, gravity: float) -> float:
    """Stokes terminal settling velocity ``U_t = 2 a² (ρ_s − ρ_f) g / (9 μ)``."""
    return 2.0 * radius**2 * delta_rho * gravity / (9.0 * mu)


def hasimoto_factor(lam: float) -> float:
    """Drag enhancement of a periodic simple-cubic array of spheres (Hasimoto 1959).

    ``F_array / F_isolated = 1 / (1 - 2.10444 λ + 2.08877 λ³ - 0.94813 λ⁵
    - 1.372 λ⁶ + 3.87 λ⁸ - 4.19 λ¹⁰)`` with ``λ = a / L`` (sphere radius over
    periodic-cell side).  Used only as an *a-posteriori* comparison against
    our own ``a/L`` extrapolation, never as a correction baked into a claim.
    """
    return 1.0 / (
        1.0
        - 2.10444 * lam
        + 2.08877 * lam**3
        - 0.94813 * lam**5
        - 1.372 * lam**6
        + 3.87 * lam**8
        - 4.19 * lam**10
    )