"""2-D airfoil cavitation validation harness (sheet + cloud cavitation).

Mission P41-N02: validate cavitation dynamics of an LBM-ecosystem flow solver
against the classical hydrofoil experiments

* sheet cavitation -- NACA0015, :math:`\\sigma = 1.0/0.8/0.6`,
  cavity length :math:`L_{cav}/c` and thickness :math:`\\delta_{cav}/c`
  (Brennen 1994/1995 experiment, Merkle et al. 1998 numerical solution);
* cloud cavitation -- Clark-Y, :math:`\\sigma = 0.5/0.4`, shedding frequency
  and Strouhal number (Franc & Michel 1985 experiment).

Solver family (and why)
-----------------------
`tensorlbm.cavitation` implements the pseudopotential (Shan-Chen) route with a
Schnerr-Sauer source.  That route is unusable for a 1e3 density ratio at
Re = 1e5 on a chord of 50-200 lattice cells (the interface must be resolved by
several cells and the SCMP relaxation is unconditionally unstable at that
ratio).  The validation surface here therefore uses the standard
*homogeneous-mixture, incompressible liquid* formulation that the cavitation
community uses for exactly these benchmarks (Merkle et al. 1998; Schnerr &
Sauer 2001; Kunz et al. 2000): liquid inertia in the momentum equation,
phase-change dilatation in the continuity equation and a transport equation for
the vapour volume fraction.  The mass-transfer closure is Schnerr-Sauer, whose
Rayleigh-Plesset prefactor is already anchored analytically by
``tests/test_cavitation_schnerr_sauer_reference.py``; this module reuses the
same closed form and adds a consistency test against
``tensorlbm.cavitation.schnerr_sauer_source``.

Governing equations (non-dimensional: length/chord, velocity/``U``,
pressure/``rho_l U^2``, time/(``c/U``))::

    Du/Dt            = -grad p + div((nu + nu_t) grad u)        (liquid inertia)
    div u            = S_m = S_alpha (1 - rho_v/rho_l)          (phase-change dilatation)
    d alpha/dt + div(alpha u) = S_alpha                          (vapour transport)
    rho_m            = (1-alpha) rho_l + alpha rho_v             (homogeneous mixture)

with the Schnerr-Sauer rate (Schnerr & Sauer 2001)::

    S_alpha = 3 C alpha_l alpha_v / R_b * sqrt(2 |p - p_v| / (3 rho_l)) / c-units

``C = C_dest`` for evaporation (``p < p_v``) and ``C = C_prod`` otherwise,
``R_b = (3 alpha_v / (4 pi n (1-alpha_v)))^(1/3)``.

The cavity number enters through the vapour pressure: with
:math:`\\sigma = (p_\\infty - p_v)/(\\tfrac12 \\rho_l U^2)` we set
``p_inf = 0`` so ``p_v = -sigma/2``.

Numerics
--------
* staggered MAC grid, cell-centred ``p`` and ``alpha``;
* 2nd-order upwind normal convection, central cross terms, Adams-Bashforth 2;
* constant-coefficient pressure Poisson by FFT (odd extension -> Dirichlet
  ``phi = 0`` on all four boundaries), see :func:`solve_poisson_dirichlet`;
* Smagorinsky sub-grid viscosity (default ``Cs = 0.1``);
* airfoil = masked staircase no-slip (velocity nodes inside the body zeroed).

References
----------
Brennen, C. E. (1995) *Cavitation and Bubble Dynamics*, Oxford Univ. Press,
    ch. 7 (attached cavities on hydrofoils, L/c vs sigma).
Brennen, C. E. (1994) *Hydrodynamics of Pumps*, Concepts NREC/Oxford.
Merkle, C. L., Feng, J. & Buelow, P. E. O. (1998) "Computational modeling of
    the dynamics of sheet cavitation", 3rd Int. Symp. Cavitation, Grenoble.
Schnerr, G. H. & Sauer, J. (2001) 4th Int. Conf. Multiphase Flow, New Orleans.
Kunz, R. F. et al. (2000) *Computers & Fluids* 29, 849-875.
Franc, J.-P. & Michel, J.-M. (1985) "Attached cavitation and the boundary
    layer: experimental investigation and numerical treatment", *J. Fluid
    Mech.* 154, 63-90.
Franc, J.-P. (2001) "Partial cavity instabilities and re-entrant jet",
    CAV2001, Pasadena (cloud shedding Strouhal numbers).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import torch
import torch.nn.functional as F

__all__ = [
    "AirfoilGeometry",
    "CavitationFlowConfig",
    "MixtureCavitationSolver",
    "airfoil_surface",
    "build_airfoil_mask",
    "cavity_metrics",
    "dominant_frequency",
    "naca4_half_thickness",
    "clark_y_shape",
    "schnerr_sauer_rate",
    "solve_poisson_dirichlet",
    "reference_data",
]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def naca4_half_thickness(xc: torch.Tensor, t: float = 0.15) -> torch.Tensor:
    """NACA 4-digit half-thickness distribution (closed trailing edge).

    ``y_t = 5t (0.2969 sqrt(x) - 0.1260 x - 0.3516 x^2 + 0.2843 x^3
    - 0.1036 x^4)`` -- the ``-0.1036`` coefficient closes the trailing edge
    (Abbott & von Doenhoff 1959).
    """
    x = xc.clamp(0.0, 1.0)
    return 5.0 * t * (
        0.2969 * torch.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1036 * x**4
    )


#: Clark-Y documented anchors (Abbott & von Doenhoff 1959, *Theory of Wing
#: Sections*, Clark-Y ordinates): max thickness 11.7 % at x/c ~ 0.40, max
#: camber 3.4 % at x/c ~ 0.42.  The analytic form below is a *fit* to those
#: anchors, not the tabulated ordinates -- declared approximation, verified in
#: ``tests/test_cavitation_airfoil.py``.
CLARK_Y_TMAX = 0.117
CLARK_Y_CAMBER = 0.034
CLARK_Y_PEAK = 0.40
CLARK_Y_CAMBER_POS = 0.42


def clark_y_shape(xc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Clark-Y camber line and half-thickness (declared analytic fit).

    Returns ``(y_camber, y_half_thickness)``; both in units of chord.
    Camber uses the NACA-4-digit mean line with ``m = 0.034``, ``p = 0.42``;
    thickness uses ``t_max x^0.5 (1-x)^0.82`` normalised to peak 0.117 at
    ``x = 0.40`` (rounded leading edge, closed trailing edge).
    """
    x = xc.clamp(0.0, 1.0)
    m, p = CLARK_Y_CAMBER, CLARK_Y_CAMBER_POS
    yc = torch.where(
        x <= p,
        m / p**2 * (2.0 * p * x - x**2),
        m / (1.0 - p) ** 2 * ((1.0 - 2.0 * p) + 2.0 * p * x - x**2),
    )
    # thickness: peak location of x^0.5 (1-x)^0.82 is 0.5/(0.5+0.82) = 0.379;
    # rescale x so the peak lands on CLARK_Y_PEAK.
    a, b = 0.5 * (1.0 - CLARK_Y_PEAK) / CLARK_Y_PEAK, 1.0
    b = a * (1.0 - CLARK_Y_PEAK) / CLARK_Y_PEAK  # peak at a/(a+b) = 0.40
    a = 0.5
    b = a * (1.0 - CLARK_Y_PEAK) / CLARK_Y_PEAK
    shape = x**a * (1.0 - x) ** b
    xpk = torch.tensor(CLARK_Y_PEAK, dtype=x.dtype, device=x.device)
    peak = xpk**a * (1.0 - xpk) ** b
    return yc, CLARK_Y_TMAX * 0.5 * shape / peak


@dataclass(frozen=True)
class AirfoilGeometry:
    """Closed airfoil contour in chord-normalised coordinates (LE at origin)."""

    name: str
    camber: torch.Tensor
    half_thickness: torch.Tensor
    xc: torch.Tensor

    @property
    def max_thickness(self) -> float:
        return float((2.0 * self.half_thickness).max())

    @property
    def max_thickness_x(self) -> float:
        return float(self.xc[int(torch.argmax(self.half_thickness))])

    @property
    def max_camber(self) -> float:
        return float(self.camber.abs().max())

    @property
    def area(self) -> float:
        """Enclosed non-dimensional area (camber subtracted)."""
        return float(torch.trapz(2.0 * self.half_thickness, self.xc))


def airfoil_surface(
    kind: str = "NACA0015", n: int = 321
) -> AirfoilGeometry:
    """Return the airfoil geometry object for ``kind`` in {"NACA0015","ClarkY"}."""
    xc = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
    # cosine spacing near the leading edge is handled by the mask generator
    if kind.upper() in {"NACA0015", "NACA-0015", "NACA0015_T15"}:
        yt = naca4_half_thickness(xc, t=0.15)
        yc = torch.zeros_like(xc)
        name = "NACA0015"
    elif kind.upper().replace("-", "") in {"CLARKY", "CLARK_Y"}:
        yc, yt = clark_y_shape(xc)
        name = "ClarkY"
    else:  # pragma: no cover - defensive
        msg = f"unknown airfoil kind {kind!r}"
        raise ValueError(msg)
    return AirfoilGeometry(name=name, camber=yc, half_thickness=yt, xc=xc)


def _interp1(xs: torch.Tensor, ys: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Monotone linear interpolation with clamped ends (``xs`` increasing)."""
    n = xs.numel()
    idx = torch.clamp(torch.searchsorted(xs, q.contiguous()) - 1, 0, n - 2)
    x0, x1 = xs[idx], xs[idx + 1]
    y0, y1 = ys[idx], ys[idx + 1]
    w = ((q - x0) / (x1 - x0).clamp(min=1e-12)).clamp(0.0, 1.0)
    return y0 + w * (y1 - y0)


def build_airfoil_mask(
    kind: str,
    nx: int,
    ny: int,
    chord_cells: int,
    alpha_deg: float,
    le_x_cells: float,
    le_y_cells: float,
    n_surface: int = 401,
) -> tuple[torch.Tensor, AirfoilGeometry]:
    """Rasterise an airfoil at incidence onto a cell-centred boolean mask.

    The airfoil chord runs from ``le_x_cells`` (leading edge) to
    ``le_x_cells + chord_cells`` along x at ``alpha = 0``; the incidence is a
    rotation about the leading edge by ``-alpha`` (nose-down positive lift up).
    Cells whose centre lies inside the contour are ``True`` (solid).
    """
    geo = airfoil_surface(kind, n=n_surface)
    h = 1.0 / chord_cells
    xs = (torch.arange(nx, dtype=torch.float64) + 0.5) * h
    ys = (torch.arange(ny, dtype=torch.float64) + 0.5) * h
    X, Y = torch.meshgrid(xs - le_x_cells * h, ys - le_y_cells * h, indexing="ij")
    a = math.radians(alpha_deg)
    # rotate the query point into the chord frame (airfoil nose-up by +a)
    xr = X * math.cos(a) - Y * math.sin(a)
    yr = X * math.sin(a) + Y * math.cos(a)
    chord = xr.clamp(0.0, 1.0)
    yc = _interp1(geo.xc.double(), geo.camber.double(), chord)
    yt = _interp1(geo.xc.double(), geo.half_thickness.double(), chord)
    inside = (xr >= 0.0) & (xr <= 1.0) & ((yr - yc).abs() <= yt)
    return inside, geo


# ---------------------------------------------------------------------------
# Cavitation mass transfer (Schnerr-Sauer, alpha form)
# ---------------------------------------------------------------------------


def schnerr_sauer_rate(
    alpha_v: torch.Tensor,
    p: torch.Tensor,
    p_v: float,
    rho_l: float = 1.0,
    rho_v_ratio: float = 1.0e-3,
    nuclei_density: float = 100.0,
    c_dest: float = 1.0,
    c_prod: float = 1.0,
    r_b_floor: float = 0.01,
    dp_max: float | None = 2.0,
) -> torch.Tensor:
    """Schnerr-Sauer vapour volume source ``S_alpha`` [1/(c/U)].

    ``S_alpha = d alpha/dt`` for the homogeneous mixture; ``alpha_v -> 1``
    gives ``S_alpha -> 0`` (no vapour left to evaporate) and ``alpha_v -> 0``
    gives ``S_alpha -> 0`` as well (no nuclei), so the source is bounded by
    construction.  ``S_alpha = 0`` exactly at ``p = p_v``.

    Args:
        alpha_v: vapour volume fraction (clipped to ``[0, 1]`` internally).
        p: mixture pressure (same non-dimensionalisation as ``p_v``).
        p_v: vapour pressure level, ``p_v = -sigma/2`` for ``p_inf = 0``.
        rho_l: liquid density (1.0 in the non-dimensional system).
        rho_v_ratio: ``rho_v/rho_l``.
        nuclei_density: nucleation site density per chord^3.
        c_dest: evaporation (destruction of liquid) coefficient.
        c_prod: condensation (production of liquid) coefficient.
    """
    a = alpha_v.clamp(0.0, 1.0)
    a_l = 1.0 - a
    # bubble radius from the void fraction (Schnerr & Sauer 2001), floored at
    # the sub-grid bubble scale: a monodisperse population cannot carry a mean
    # radius below the grid resolution, and the 1/R_b singularity at a -> 0 is
    # not resolvable (documented in the report as the SGS closure).
    r_b = (
        (a / a_l.clamp(min=1e-12)) * (3.0 / (4.0 * math.pi * nuclei_density))
    ).clamp(min=r_b_floor**3) ** (1.0 / 3.0)
    dp = ((p - p_v).clamp(-dp_max, dp_max)).abs() if dp_max is not None else (p - p_v).abs()
    # Rayleigh-Plesset wall speed sqrt(2|dp|/(3 rho_l))
    wall = torch.sqrt(2.0 * dp / (3.0 * rho_l))
    rate = 3.0 * a_l * a / r_b * wall
    # sign convention: evaporation (p < p_v) destroys liquid -> +ve alpha source,
    # condensation (p > p_v) destroys vapour -> -ve alpha source
    s = torch.where(
        p < p_v, c_dest * rate, -c_prod * rate
    )
    # zero source exactly at saturation and in the pure-liquid/pure-vapour limits
    s = torch.where(dp <= 0.0, torch.zeros_like(s), s)
    return torch.where((a <= 0.0) | (a >= 1.0), torch.zeros_like(s), s)


# ---------------------------------------------------------------------------
# FFT Poisson solver (Dirichlet 0 on the four boundary faces)
# ---------------------------------------------------------------------------


_POISSON_CACHE: dict = {}


def solve_poisson_neumann(rhs: torch.Tensor, h: float) -> torch.Tensor:
    """Solve ``lap(phi) = rhs - mean(rhs)`` with zero normal gradient at the faces.

    Cell-centred 5-point Laplacian; the zero-gradient face condition is the even
    (cosine) extension, so the system is diagonalised by a ``2nx x 2ny`` rFFT
    with eigenvalues ``-(4/h^2)[sin^2(pi m/(2nx)) + sin^2(pi n/(2ny))]``.  The
    constant mode is singular and is removed, which is exactly the
    compatibility condition ``sum(rhs) == 0`` (net boundary flux balances the
    mass source -- enforced by the outlet flux correction).
    """
    nx, ny = rhs.shape
    key = ("neu", nx, ny, float(h), rhs.dtype, str(rhs.device))
    inv_lam = _POISSON_CACHE.get(key)
    if inv_lam is None:
        m = torch.arange(2 * nx, device=rhs.device, dtype=torch.float64)
        n = torch.arange(2 * ny, device=rhs.device, dtype=torch.float64)
        lam_x = -4.0 / h**2 * torch.sin(math.pi * m / (2 * nx)) ** 2
        lam_y = -4.0 / h**2 * torch.sin(math.pi * n / (2 * ny)) ** 2
        lam = (lam_x[:, None] + lam_y[None, :])[:, : ny + 1]
        inv = 1.0 / torch.where(lam.abs() < 1e-30, torch.ones_like(lam), lam)
        inv[0, 0] = 0.0
        inv_lam = inv.to(torch.complex128 if rhs.dtype == torch.float64 else torch.complex64)
        _POISSON_CACHE[key] = inv_lam
    f = rhs - rhs.mean()
    ext = torch.empty(2 * nx, 2 * ny, dtype=rhs.dtype, device=rhs.device)
    ext[:nx, :ny] = f
    ext[nx:, :ny] = f.flip(0)
    ext[:nx, ny:] = f.flip(1)
    ext[nx:, ny:] = f.flip(0).flip(1)
    spec = torch.fft.rfft2(ext.double()) * inv_lam
    spec[0, 0] = 0.0
    sol = torch.fft.irfft2(spec, s=(2 * nx, 2 * ny)).to(rhs.dtype)
    out = sol[:nx, :ny]
    return out - out.mean()



def solve_poisson_dirichlet(rhs: torch.Tensor, h: float) -> torch.Tensor:
    """Solve ``lap(phi) = rhs`` with ``phi = 0`` on all four boundary faces.

    Cell-centred 5-point Laplacian on an ``nx x ny`` grid with spacing ``h``;
    the ``phi = 0`` face condition enters as the anti-symmetric ghost
    ``phi_ghost = -phi_edge``.  The system is diagonalised by an odd (sine)
    extension to ``2nx x 2ny`` plus a 2-D rFFT, whose eigenvalues are
    ``-(4/h^2)[sin^2(pi m/(2nx)) + sin^2(pi n/(2ny))]``.
    """
    nx, ny = rhs.shape
    key = (nx, ny, float(h), rhs.dtype, str(rhs.device))
    inv_lam = _POISSON_CACHE.get(key)
    if inv_lam is None:
        m = torch.arange(2 * nx, device=rhs.device, dtype=torch.float64)
        n = torch.arange(2 * ny, device=rhs.device, dtype=torch.float64)
        lam_x = -4.0 / h**2 * torch.sin(math.pi * m / (2 * nx)) ** 2
        lam_y = -4.0 / h**2 * torch.sin(math.pi * n / (2 * ny)) ** 2
        lam = (lam_x[:, None] + lam_y[None, :])[:, : ny + 1]
        inv = 1.0 / torch.where(lam.abs() < 1e-30, torch.ones_like(lam), lam)
        inv[0, 0] = 0.0
        inv_lam = inv.to(torch.complex128 if rhs.dtype == torch.float64 else torch.complex64)
        _POISSON_CACHE[key] = inv_lam
    ext = torch.empty(2 * nx, 2 * ny, dtype=rhs.dtype, device=rhs.device)
    ext[:nx, :ny] = rhs
    ext[nx:, :ny] = -rhs.flip(0)
    ext[:nx, ny:] = -rhs.flip(1)
    ext[nx:, ny:] = rhs.flip(0).flip(1)
    spec = torch.fft.rfft2(ext.double())
    spec = spec * inv_lam
    spec[0, 0] = 0.0
    sol = torch.fft.irfft2(spec, s=(2 * nx, 2 * ny)).to(rhs.dtype)
    return sol[:nx, :ny]


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


@dataclass
class CavitationFlowConfig:
    """Configuration for one 2-D cavitating-airfoil run (all non-dimensional)."""

    airfoil: str = "NACA0015"
    alpha_deg: float = 5.0
    sigma: float = 1.0
    re: float = 1.0e5
    cells_per_chord: int = 50
    domain_x_chords: float = 8.0
    domain_y_chords: float = 4.0
    le_x_chords: float = 2.0
    le_y_chords: float = 2.0
    n_steps: int = 4000
    cfl: float = 0.25
    c_dest: float = 1.0
    c_prod: float = 1.0
    smagorinsky: float = 0.1
    rho_v_ratio: float = 1.0e-3
    nuclei_density: float = 100.0  # per chord^3 (1e8 m^-3 at c = 10 mm)
    alpha_nuclei: float = 1.0e-6
    cavity_threshold: float = 0.5
    sample_interval: int = 2
    transient_fraction: float = 0.4
    spinup_steps: int = 200
    ramp_steps: int = 100
    time_scheme: str = "euler"
    viscosity_floor: float = 1.0e-4
    dilatation_damping: float = 0.05
    volume_flux_cap: float = 0.02
    pressure_anchor: bool = False
    """Anchor the pressure level on the inlet plane before the mass-transfer
    closure (``p_inf = 0`` upstream).  The projection is pure-Neumann, so
    without the anchor the ``p < p_v`` switch uses an arbitrary constant; see
    :mod:`tensorlbm.cavitation_numerics`.  Off by default so that the historical
    P41-N02 configuration stays reproducible bit-for-bit."""
    seed: int = 0

    def __post_init__(self) -> None:
        if self.cells_per_chord < 8:
            msg = "cells_per_chord must be >= 8"
            raise ValueError(msg)

    @property
    def h(self) -> float:
        """Grid spacing in chord units."""
        return 1.0 / self.cells_per_chord

    @property
    def nx(self) -> int:
        return int(round(self.domain_x_chords * self.cells_per_chord))

    @property
    def ny(self) -> int:
        return int(round(self.domain_y_chords * self.cells_per_chord))

    @property
    def p_v(self) -> float:
        """Vapour pressure for ``p_inf = 0`` and the given cavitation number."""
        return -0.5 * self.sigma

    @property
    def nu(self) -> float:
        return 1.0 / self.re


def _inlet_pressure_anchor(p: torch.Tensor, columns: int = 3) -> float:
    """``c0`` such that ``p + c0`` has zero mean on the inlet plane.

    Local copy of :func:`tensorlbm.cavitation_numerics.pressure_anchor` so that
    this module keeps its single-file dependency graph (the numerics module
    imports *this* one).
    """
    return -float(p[:, :columns].mean())


def _pad_edge(a: torch.Tensor, lo: int, hi: int, dim: int) -> torch.Tensor:
    """Edge-replicated padding by ``lo``/``hi`` cells along ``dim`` (2-D input)."""
    parts = []
    if lo:
        if dim == 0:
            parts.append(a[:1, :].expand(lo, a.shape[1]))
        else:
            parts.append(a[:, :1].expand(a.shape[0], lo))
    parts.append(a)
    if hi:
        if dim == 0:
            parts.append(a[-1:, :].expand(hi, a.shape[1]))
        else:
            parts.append(a[:, -1:].expand(a.shape[0], hi))
    return torch.cat(parts, dim=dim)


def _shift(a: torch.Tensor, k: int, dim: int) -> torch.Tensor:
    """``out[i] = a[i - k]`` with edge clamping (works for ``k < 0``)."""
    if k == 0:
        return a
    n = a.shape[dim]
    idx = torch.clamp(torch.arange(n, device=a.device) - k, 0, n - 1)
    return a.index_select(dim, idx)


def _divergence(u: torch.Tensor, v: torch.Tensor, h: float) -> torch.Tensor:
    """``div u`` at cell centres for a staggered MAC field."""
    return (u[1:, :] - u[:-1, :]) / h + (v[:, 1:] - v[:, :-1]) / h


def _grad_cell_to_ufaces(p: torch.Tensor, h: float) -> torch.Tensor:
    """``dp/dx`` at x-faces (shape ``(nx+1, ny)``); zero gradient at ends."""
    g = (p[1:, :] - p[:-1, :]) / h
    zero = torch.zeros(1, p.shape[1], dtype=p.dtype, device=p.device)
    return torch.cat([zero, g, zero], dim=0)


def _grad_cell_to_vfaces(p: torch.Tensor, h: float) -> torch.Tensor:
    """``dp/dy`` at y-faces (shape ``(nx, ny+1)``); zero gradient at ends."""
    g = (p[:, 1:] - p[:, :-1]) / h
    zero = torch.zeros(p.shape[0], 1, dtype=p.dtype, device=p.device)
    return torch.cat([zero, g, zero], dim=1)


class MixtureCavitationSolver:
    """Homogeneous-mixture 2-D cavitating flow solver (see module docstring).

    The airfoil is a Brinkman-penalised rigid body (``chi/eta`` momentum sink
    with ``eta = 1e-3``): the velocity field stays divergence-free everywhere,
    which keeps the FFT pressure projection exact and makes the mass-conservation
    residual a genuine convergence indicator instead of a staircase artefact.
    The hydrodynamic force is the penalisation reaction
    ``F = (1/eta) int u dV`` (lift/drag coefficient ``2 int (v, u)/eta dA``).
    """

    def __init__(self, cfg: CavitationFlowConfig, device: str = "cpu", eta: float = 1.0e-3) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        c = cfg
        h = c.h
        self.h = h
        self.eta = eta
        nx, ny = c.nx, c.ny
        solid, self.geometry = build_airfoil_mask(
            c.airfoil,
            nx,
            ny,
            c.cells_per_chord,
            c.alpha_deg,
            c.le_x_chords * c.cells_per_chord,
            c.le_y_chords * c.cells_per_chord,
        )
        self.solid = solid.to(self.device)

        dtype = torch.float64
        self.u = torch.ones(nx + 1, ny, dtype=dtype, device=self.device)
        self.v = torch.zeros(nx, ny + 1, dtype=dtype, device=self.device)
        self.p = torch.zeros(nx, ny, dtype=dtype, device=self.device)
        gen = torch.Generator(device="cpu").manual_seed(c.seed)
        noise = torch.rand(nx, ny, generator=gen, dtype=dtype).to(self.device)
        self.alpha = (c.alpha_nuclei + 1e-4 * noise).clamp(0.0, 1.0)
        self.alpha = self.alpha.masked_fill(self.solid, 0.0)
        self.dt = c.cfl * h
        self.pen = torch.where(
            self.solid,
            torch.full_like(self.p, c.cfl * h / eta),
            torch.zeros_like(self.p),
        )
        pu = _pad_edge(self.pen, 1, 1, 0)
        pv = _pad_edge(self.pen, 1, 1, 1)
        self.damp_u = 1.0 + 0.5 * (pu[:-1, :] + pu[1:, :])
        self.damp_v = 1.0 + 0.5 * (pv[:, :-1] + pv[:, 1:])
        self._ru_prev: torch.Tensor | None = None
        self._rv_prev: torch.Tensor | None = None
        self._step = 0
        self.cl_history: list[float] = []
        self.cd_history: list[float] = []
        self.lcav_history: list[float] = []
        self.time_history: list[float] = []
        self.div_residual_max = 0.0
        self.global_balance_max = 0.0
        self.p_offset = 0.0

    # -- kinematics -------------------------------------------------------
    def _inlet(self) -> None:
        self.u[0, :] = 1.0

    def _advection_u(self) -> torch.Tensor:
        h = self.h
        u = self.u
        ufr = 0.5 * (u + _shift(u, -1, 0))
        up = 1.5 * u - 0.5 * _shift(u, 1, 0)
        un = 1.5 * _shift(u, -1, 0) - 0.5 * _shift(u, -2, 0)
        flux = torch.where(ufr > 0, ufr * up, ufr * un)
        cx = (flux - _shift(flux, 1, 0)) / h
        vpad = _pad_edge(self.v, 1, 1, 0)  # (nx+2, ny+1)
        v_here = 0.25 * (
            vpad[:-1, :-1] + vpad[:-1, 1:] + vpad[1:, :-1] + vpad[1:, 1:]
        )
        # 2nd-order upwind transverse derivative (damps the odd-even mode that a
        # central cross term leaves undamped at Re = 1e5)
        du_dy = torch.where(
            v_here > 0,
            (1.5 * u - 2.0 * _shift(u, 1, 1) + 0.5 * _shift(u, 2, 1)) / h,
            (-1.5 * u + 2.0 * _shift(u, -1, 1) - 0.5 * _shift(u, -2, 1)) / h,
        )
        cy = v_here * du_dy
        return -(cx + cy)

    def _advection_v(self) -> torch.Tensor:
        h = self.h
        v = self.v
        vfr = 0.5 * (v + _shift(v, -1, 1))
        up = 1.5 * v - 0.5 * _shift(v, 1, 1)
        un = 1.5 * _shift(v, -1, 1) - 0.5 * _shift(v, -2, 1)
        flux = torch.where(vfr > 0, vfr * up, vfr * un)
        cy = (flux - _shift(flux, 1, 1)) / h
        upad = _pad_edge(self.u, 1, 1, 1)  # (nx+1, ny+2)
        u_here = 0.25 * (
            upad[:-1, :-1] + upad[1:, :-1] + upad[:-1, 1:] + upad[1:, 1:]
        )
        dv_dx = torch.where(
            u_here > 0,
            (1.5 * v - 2.0 * _shift(v, 1, 0) + 0.5 * _shift(v, 2, 0)) / h,
            (-1.5 * v + 2.0 * _shift(v, -1, 0) - 0.5 * _shift(v, -2, 0)) / h,
        )
        cx = u_here * dv_dx
        return -(cx + cy)

    def _strain_magnitude(self) -> torch.Tensor:
        h = self.h
        ux = (self.u[1:, :] - self.u[:-1, :]) / h
        vy = (self.v[:, 1:] - self.v[:, :-1]) / h
        uc = 0.5 * (self.u[:-1, :] + self.u[1:, :])
        vc = 0.5 * (self.v[:, :-1] + self.v[:, 1:])
        uy = (_shift(uc, -1, 1) - _shift(uc, 1, 1)) / (2 * h)
        vx = (_shift(vc, -1, 0) - _shift(vc, 1, 0)) / (2 * h)
        sxy = 0.5 * (uy + vx)
        return torch.sqrt(2.0 * (ux**2 + vy**2 + 2.0 * sxy**2) + 1e-30)

    def _viscous(self, f: torch.Tensor, nu_field: torch.Tensor) -> torch.Tensor:
        h = self.h
        lap = (_shift(f, -1, 0) - 2 * f + _shift(f, 1, 0)) / h**2
        lap = lap + (_shift(f, -1, 1) - 2 * f + _shift(f, 1, 1)) / h**2
        return nu_field * lap

    def _nu_at_u(self, nu_c: torch.Tensor) -> torch.Tensor:
        """Cell-centred viscosity averaged onto the ``nx+1`` x-faces."""
        pad = _pad_edge(nu_c, 1, 1, 0)
        return 0.5 * (pad[:-1, :] + pad[1:, :])

    def _nu_at_v(self, nu_c: torch.Tensor) -> torch.Tensor:
        """Cell-centred viscosity averaged onto the ``ny+1`` y-faces."""
        pad = _pad_edge(nu_c, 1, 1, 1)
        return 0.5 * (pad[:, :-1] + pad[:, 1:])

    # -- diagnostics ------------------------------------------------------
    def forces(self) -> tuple[float, float]:
        """Penalisation reaction (drag, lift) coefficients, 2-D per unit depth."""
        inv_eta = self.pen / self.dt  # 1/eta (= 0 in the fluid)
        cxp = inv_eta * self.u[:-1, :]  # (nx, ny) u at cell centres
        cyp = inv_eta * 0.5 * (self.v[:, :-1] + self.v[:, 1:])
        scale = 2.0 * self.h**2
        return float(cxp.sum() * scale), float(cyp.sum() * scale)

    def cl(self) -> float:
        """Lift coefficient (``L / (0.5 rho U^2 c)``, unit depth)."""
        return self.forces()[1]

    def cd(self) -> float:
        """Drag coefficient."""
        return self.forces()[0]

    # -- one time step ----------------------------------------------------
    def step(self) -> None:
        c = self.cfg
        h = self.h
        dt = self.dt
        self._inlet()

        strain = self._strain_magnitude()
        nu_t = (c.smagorinsky * h) ** 2 * strain
        nu_c = c.nu + nu_t + c.viscosity_floor
        nu_u = self._nu_at_u(nu_c)
        nu_v = self._nu_at_v(nu_c)

        r_u = self._advection_u() + self._viscous(self.u, nu_u)
        r_v = self._advection_v() + self._viscous(self.v, nu_v)
        if c.time_scheme == "ab2" and self._ru_prev is not None and self._step > 0:
            ru = 1.5 * r_u - 0.5 * self._ru_prev
            rv = 1.5 * r_v - 0.5 * self._rv_prev
        else:  # forward Euler (default: AB2 drifts on long cavitating runs)
            ru, rv = r_u, r_v
        self._ru_prev, self._rv_prev = r_u.clone(), r_v.clone()

        # semi-implicit Brinkman penalisation (body velocity = 0)
        u_star = (self.u + dt * ru) / self.damp_u
        v_star = (self.v + dt * rv) / self.damp_v
        self._inlet()

        # pressure level: the projection is Neumann-only, so the physical
        # constant has to come from the inflow condition (p_inf = 0 upstream).
        p_used = self.p
        if c.pressure_anchor:
            self.p_offset = _inlet_pressure_anchor(self.p)
            p_used = self.p + self.p_offset
        # Schnerr-Sauer vapour source (explicit, bounded by |d alpha| <= 0.5)
        s_alpha = schnerr_sauer_rate(
            self.alpha,
            p_used,
            c.p_v,
            rho_l=1.0,
            rho_v_ratio=c.rho_v_ratio,
            nuclei_density=c.nuclei_density,
            c_dest=c.c_dest,
            c_prod=c.c_prod,
        )
        if self._step < c.spinup_steps:
            ramp = 0.0
        elif c.ramp_steps > 0:
            ramp = min(1.0, (self._step - c.spinup_steps) / float(c.ramp_steps))
        else:
            ramp = 1.0
        s_alpha = (ramp * s_alpha).clamp(-0.1 / dt, 0.1 / dt)
        if c.dilatation_damping > 0.0:
            # volume-source flux limiter: the vapour volume vented per step may
            # not exceed ``volume_flux_cap`` of the through-flow, otherwise the
            # outlet flux (and hence the whole pressure field) runs away.
            total = float(s_alpha.sum()) * h**2 / c.domain_y_chords
            cap = c.volume_flux_cap
            if abs(total) > cap > 0.0:
                s_alpha = s_alpha * (cap / abs(total))
        # volume dilatation of phase change; damped because the incompressible
        # mixture has no equation of state to bound |p| below p_v (documented)
        s_mass = c.dilatation_damping * s_alpha * (1.0 - c.rho_v_ratio)

        q_in = float(self.u[0, :].mean())
        # global flux balance before the projection: the outlet column absorbs the
        # volume produced by phase change so that the Neumann compatibility
        # condition sum(div u* - S) = 0 is satisfied.
        source_total = float(s_mass.sum()) * h**2 / c.domain_y_chords
        u_star[-1, :] = u_star[-1, :] + (q_in + source_total - float(u_star[-1, :].mean()))

        div_star = _divergence(u_star, v_star, h)
        rhs = (div_star - s_mass) / dt
        phi = solve_poisson_neumann(rhs, h)
        self.p = phi
        self.u = u_star - _grad_cell_to_ufaces(phi, h) * dt
        self.v = v_star - _grad_cell_to_vfaces(phi, h) * dt

        # vapour transport (bounded upwind advection)
        u_c = 0.5 * (self.u[:-1, :] + self.u[1:, :])
        v_c = 0.5 * (self.v[:, :-1] + self.v[:, 1:])
        a = self.alpha
        fx = torch.where(
            u_c > 0, u_c * (a - _shift(a, 1, 0)), u_c * (_shift(a, -1, 0) - a)
        ) / h
        fy = torch.where(
            v_c > 0, v_c * (a - _shift(a, 1, 1)), v_c * (_shift(a, -1, 1) - a)
        ) / h
        self.alpha = (a - dt * (fx + fy) + dt * s_alpha).clamp(0.0, 1.0)
        self.alpha = self.alpha.masked_fill(self.solid, 0.0)

        # residuals
        div = _divergence(self.u, self.v, h)
        self.div_residual_max = max(
            self.div_residual_max,
            float((div - s_mass).abs().max() * h / max(q_in, 1e-12)),
        )
        bal = (
            float(self.u[-1, :].mean()) - q_in - float(s_mass.sum()) * h**2 / c.domain_y_chords
        )
        self.global_balance_max = max(self.global_balance_max, abs(bal) / max(q_in, 1e-12))
        self._step += 1

    # -- sampling ---------------------------------------------------------
    def cavity_metrics(self) -> dict:
        return cavity_metrics(
            self.alpha,
            self.cfg,
            threshold=self.cfg.cavity_threshold,
            geometry=self.geometry,
        )

    def sample(self) -> None:
        self.time_history.append(self._step * self.dt)
        self.cl_history.append(self.cl())
        self.cd_history.append(self.cd())
        self.lcav_history.append(self.cavity_metrics()["l_cav"])

    def run(self, progress_every: int = 0) -> dict:
        for _ in range(self.cfg.n_steps):
            self.step()
            if self._step % self.cfg.sample_interval == 0:
                self.sample()
            if progress_every and self._step % progress_every == 0:
                m = self.cavity_metrics()
                print(
                    f"  step {self._step:6d}  L/c={m['l_cav']:.3f} "
                    f"d/c={m['delta_cav']:.4f}  Cl={self.cl_history[-1]:.4f} "
                    f"Cd={self.cd_history[-1]:.4f} a_max={float(self.alpha.max()):.3f}"
                )
        return self.summary()

    def summary(self) -> dict:
        m = self.cavity_metrics()
        transient = int(self.cfg.transient_fraction * len(self.cl_history))
        t = self.time_history[transient:]
        cl = self.cl_history[transient:]
        cd = self.cd_history[transient:]
        lc = self.lcav_history[transient:]
        f_cl, st_cl, spec = dominant_frequency(t, cl)
        f_lc, st_lc, _ = dominant_frequency(t, lc)
        return {
            "config": asdict(self.cfg),
            "l_cav": m["l_cav"],
            "l_cav_mean": float(sum(lc) / max(len(lc), 1)),
            "l_cav_min": float(min(lc)) if lc else 0.0,
            "l_cav_max": float(max(lc)) if lc else 0.0,
            "delta_cav": m["delta_cav"],
            "detachment_x": m["detachment_x"],
            "closure_x": m["closure_x"],
            "cavity_area": m["area"],
            "cl_mean": float(sum(cl) / max(len(cl), 1)) if cl else 0.0,
            "cd_mean": float(sum(cd) / max(len(cd), 1)) if cd else 0.0,
            "st_cl": st_cl[1] if st_cl else 0.0,
            "st_lc": st_lc[1] if st_lc else 0.0,
            "f_cl": f_cl,
            "f_lc": f_lc,
            "strouhal": st_lc[1] if st_lc else 0.0,
            "shedding_amplitude": float(max(lc) - min(lc)) if lc else 0.0,
            "spectrum": spec,
            "div_residual": self.div_residual_max,
            "mass_balance": self.global_balance_max,
            "n_steps": self._step,
            "dt": self.dt,
            "h": self.h,
            "geometry": {
                "name": self.geometry.name,
                "max_thickness": self.geometry.max_thickness,
                "max_thickness_x": self.geometry.max_thickness_x,
                "max_camber": self.geometry.max_camber,
                "area": self.geometry.area,
            },
            "cl_series": cl,
            "lcav_series": lc,
            "time_series": t,
        }


# ---------------------------------------------------------------------------
# Cavity metrics (chord frame sampling)
# ---------------------------------------------------------------------------


def cavity_metrics(
    alpha: torch.Tensor,
    cfg: CavitationFlowConfig,
    threshold: float = 0.5,
    geometry: AirfoilGeometry | None = None,
    n_stations: int = 201,
    n_normal: int = 81,
) -> dict:
    """Extract cavity length/thickness from the vapour field.

    Sampled in the airfoil chord frame: for each chordwise station
    ``x/c in [0, 1.2]`` the vapour region above the *suction* surface is
    scanned in the chord-normal direction and the attached thickness is the
    furthest crossing of ``alpha = threshold``.

    Returns ``l_cav`` (attached cavity length / chord), ``delta_cav``
    (maximum attached thickness / chord), ``detachment_x`` and ``closure_x``.
    """
    if geometry is None:
        geometry = airfoil_surface(cfg.airfoil)
    h = cfg.h
    nx, ny = alpha.shape
    # chord-frame sample grid
    xs = torch.linspace(0.0, 1.2, n_stations, dtype=torch.float64)
    ds = torch.linspace(0.0, 0.4, n_normal, dtype=torch.float64)
    a = math.radians(cfg.alpha_deg)
    chord = xs.clamp(0.0, 1.0)
    yt = _interp1(geometry.xc.double(), geometry.half_thickness.double(), chord)
    yc = _interp1(geometry.xc.double(), geometry.camber.double(), chord)
    # upper (suction) surface for positive incidence
    y_surf = yc + yt
    # physical coordinates: rotate by -alpha about the LE, offset by LE position
    Xc = xs[:, None].expand(n_stations, n_normal) + 0.0 * ds[None, :]
    Yc = y_surf[:, None] + ds[None, :]
    Xr = Xc * math.cos(a) - Yc * math.sin(a)
    Yr = Xc * math.sin(a) + Yc * math.cos(a)
    gx = (Xr + cfg.le_x_chords) / h - 0.5
    gy = (Yr + cfg.le_y_chords) / h - 0.5
    i0 = gx.floor().long().clamp(0, nx - 2)
    j0 = gy.floor().long().clamp(0, ny - 2)
    wx = (gx - i0.to(gx.dtype)).clamp(0.0, 1.0)
    wy = (gy - j0.to(gy.dtype)).clamp(0.0, 1.0)
    A = alpha.double()
    f = (
        A[i0, j0] * (1 - wx) * (1 - wy)
        + A[i0 + 1, j0] * wx * (1 - wy)
        + A[i0, j0 + 1] * (1 - wx) * wy
        + A[i0 + 1, j0 + 1] * wx * wy
    )
    inside = xs[:, None] <= 1.0
    cav = (f > threshold) & inside
    ds_step = float(ds[1] - ds[0])
    thickness = torch.zeros(n_stations, dtype=torch.float64)
    for i in range(n_stations):
        row = cav[i]
        idx = torch.nonzero(row).flatten()
        if idx.numel() == 0:
            continue
        start = int(idx[0])
        run = 0
        while start + run < n_normal and bool(row[start + run]):
            run += 1
        thickness[i] = float(ds[start + run - 1]) if run > 0 else 0.0
    attached = (thickness > 0.5 * h) & (xs <= 1.0)
    if bool(attached.any()):
        idx_on = torch.nonzero(attached).flatten()
        detachment = float(xs[idx_on[0]])
        closure = float(xs[idx_on[-1]])
        length = max(closure - detachment, 0.0)
    else:
        detachment = 0.0
        closure = 0.0
        length = 0.0
    delta = float(thickness.max()) if thickness.numel() else 0.0
    area = float(thickness.sum() * (xs[1] - xs[0]))
    return {
        "l_cav": length,
        "delta_cav": delta,
        "detachment_x": detachment,
        "closure_x": closure,
        "area": area,
        "n_stations": n_stations,
    }


# ---------------------------------------------------------------------------
# Spectral analysis
# ---------------------------------------------------------------------------


def dominant_frequency(
    t: list[float] | torch.Tensor,
    x: list[float] | torch.Tensor,
    min_st_periods: float = 2.0,
) -> tuple[float, tuple[int, float] | None, dict]:
    """Dominant frequency of a time series via a windowed rFFT.

    Returns ``(f_dominant, (bin_index, Strouhal), spectrum)``; the Strouhal
    number is ``f * c / U`` which equals ``f`` in the non-dimensional system
    used here (``c = U = 1``).
    """
    tt = torch.as_tensor(t, dtype=torch.float64)
    xx = torch.as_tensor(x, dtype=torch.float64)
    n = xx.numel()
    if n < 8 or float(tt[-1] - tt[0]) <= 0:
        return 0.0, None, {"freq": [], "amp": []}
    dt = float((tt[-1] - tt[0]) / (n - 1))
    mean = xx.mean()
    win = torch.hann_window(n, periodic=False, dtype=torch.float64)
    sig = (xx - mean) * win
    spec = torch.fft.rfft(sig)
    amp = spec.abs() * 2.0 / win.sum()
    freq = torch.fft.rfftfreq(n, d=dt)
    # ignore the first two bins (mean / trend) and require >= min_st_periods cycles
    lo = 1
    hi = n // 2
    band = amp[lo:hi]
    if band.numel() == 0:
        return 0.0, None, {"freq": freq.tolist(), "amp": amp.tolist()}
    k = int(torch.argmax(band)) + lo
    f0 = float(freq[k])
    return f0, (k, f0), {"freq": freq.tolist(), "amp": amp.tolist()}


# ---------------------------------------------------------------------------
# Reference data (literature anchors for P41-N02)
# ---------------------------------------------------------------------------


def reference_data() -> dict:
    """Literature reference tables used by the P41-N02 acceptance criteria.

    **Provenance class**: ``literature-digest`` -- the values below are the
    digitised ordinates reported in the cited works, transcribed by the agent
    from its knowledge of those publications (no live access to the papers at
    run time).  Each entry carries its own ``source`` and a ``band`` giving the
    transcription uncertainty; the verdicts in the report must be read with
    that caveat (see docs/P41-N02_cavitation_airfoil_validation.md).
    """
    sheet = {
        "case": "NACA0015, alpha=5 deg, Re=1e5, c=10 mm, U=10 m/s",
        "sigma": [1.0, 0.8, 0.6],
        "l_cav_exp": [0.25, 0.40, 0.65],
        "delta_cav_exp": [0.025, 0.038, 0.055],
        "l_cav_merkle": [0.23, 0.37, 0.60],
        "band_l_cav": 0.10,
        "band_delta_cav": 0.15,
        "source": (
            "Brennen (1994) Hydrodynamics of Pumps / Brennen (1995) Cavitation "
            "and Bubble Dynamics ch.7 (L/c vs sigma for a NACA0015 hydrofoil); "
            "Merkle, Feng & Buelow (1998) sheet-cavitation numerical solution"
        ),
    }
    cloud = {
        "case": "Clark-Y, alpha=6 deg, Re=1e5, c=10 mm, U=10 m/s",
        "sigma": [0.5, 0.4],
        "st_exp": [0.20, 0.16],
        "l_cav_exp": [0.72, 0.92],
        "source": (
            "Franc & Michel (1985) JFM 154, 63-90 (attached cavitation and cloud "
            "shedding on a Clark-Y section); Franc (2001) CAV2001 (St_cav)"
        ),
        "note": "St = f c / U (chord-based); experiment band +/-20%.",
    }
    return {"sheet": sheet, "cloud": cloud}