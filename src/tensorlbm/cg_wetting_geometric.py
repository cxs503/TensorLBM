"""Geometrical wetting for the D2Q9 colour-gradient solver: gradient-rotation form.

This is a **faithful port** of the geometrical wetting boundary condition used
by ``PorousMediaSimulation/openLBMPM`` (``RKCG2D``).  Unlike the ghost-value
form in :mod:`tensorlbm.cg_wetting2d`, the wetting-consistent gradient here is
built by *rotating* the colour-gradient vector at wall-adjacent fluid nodes so
that it makes the prescribed contact angle with the wall normal; the magnitude
of the gradient is preserved.  That is exactly what the reference does and it
is what the surface-tension force is computed from, so no extra wall phase is
injected into the density field.

Source of each piece (``openLBMPM``, master; ``file:line`` from the checkout at
``/tmp/openLBMPM`` used for this port, sha256 recorded in the run artifacts):

=============================================  ==================================
piece                                          reference location
=============================================  ==================================
wall-normal table (ISO8 weighted, r <= 2)      ``RKCG2D/RKD2Q9.py:808-890``
phase value on solid nodes (weighted average)  ``RKCG2D/AcceleratedRKGPU2D.py:1557-1578``
isotropic colour gradient (8 dirs, x3)         ``RKCG2D/AcceleratedRKGPU2D.py:1585-1633``
gradient rotation + nearest-candidate choice   ``RKCG2D/AcceleratedRKGPU2D.py:2426-2492``
default parameters (``SurfaceTension``, ...)   ``IniFiles/RKtwophasesetup2D.ini``
=============================================  ==================================

The rotation, written out
-------------------------
Let ``n_s`` be the unit wall normal at a wall-adjacent fluid node, pointing
*from the fluid into the solid*, and let ``g = grad(phi)`` with
``phi > 0`` inside the wetting phase.  The reference defines the unit gradient
with a **minus sign**,

    g_hat = -g / |g|                                        (1)

and builds the two unit vectors that make the prescribed angle ``theta`` with
``n_s`` while staying in the plane of ``n_s`` and ``g_hat``:

    a  = g_hat . n_s,   Delta = |g_hat - a n_s| = sin(angle(g_hat, n_s))
    p  = (g_hat - a n_s) / Delta
    n1 = cos(theta) n_s + sin(theta) p
    n2 = cos(theta) n_s - sin(theta) p                      (2)

and keeps the candidate closer to ``g_hat`` (strict ``<``/``>``; a tie leaves
the gradient untouched), then writes back

    g := -|g| n_chosen                                       (3)

Because ``s = sin(theta) > 0`` and ``Delta >= 0``,

    |n1 - g_hat|^2 = (c-a)^2 + (s-Delta)^2
    |n2 - g_hat|^2 = (c-a)^2 + (s+Delta)^2

so candidate 1 is *always* the nearer one for ``0 < theta < 180`` and
``Delta > 0``: the "two-candidate nearest" branch of the reference reduces to
a signed rotation (rotate ``g_hat`` towards ``n_s`` by ``theta``) whose sign is
inherited from the tangential component of ``g_hat``.  A unit test pins this
equivalence; it also means the port cannot silently differ from the reference
on the branch choice.

Degenerate cases (identical to the reference): ``|g| <= 1e-8`` (no interface)
and ``Delta <= 1e-9`` (gradient parallel to the wall normal, i.e. an interface
parallel to the wall) leave the gradient unchanged -- the reference ends up in
its ``dist1 == dist2`` no-op branch there.

Sign convention check (``theta`` is measured inside the ``phi > 0`` phase):
with ``g_hat = -g/|g|``, a consistent static interface satisfies
``g_hat . n_s = cos(theta)``, equivalently ``n_int . (-n_s) = cos(theta)`` for
``n_int = g/|g|``: the interface normal makes the contact angle with the wall
normal that points into the fluid, which is the usual definition.

Deviation from the reference (documented, not hidden)
-----------------------------------------------------
* The host solver's surface-tension force is the anisotropic
  ``A``-perturbation of :mod:`tensorlbm.color_gradient2d`, not the reference's
  CSF force ``-0.5 sigma kappa grad(phi)``.  Only the *gradient* is rotated, so
  the wetting condition itself is transportable; the force form is unchanged.
* No MRT collision operator is used (host is BGK/TRT per colour).
* The wall-normal table is computed on the same mask the host solver uses
  (one solid row for a flat plate) with the reference's exact ISO8 weights; the
  reference additionally uses an image-based pore geometry.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import torch

__all__ = [
    "ISO8_OFFSETS",
    "geometric_reference_sha256",
    "wall_normal_field",
    "solid_color_values",
    "isotropic_color_gradient",
    "rotate_wall_color_gradient",
    "readback_wall_angle",
    "reference_rotation_scalar",
    "sessile_drop_initial",
    "run_sessile_drop_contact_angle",
]

#: Reference checkout used for the port (contains the files listed in the
#: module docstring).  The digest is recorded in every artifact so the exact
#: source revision behind a number is always recoverable.
REFERENCE_ROOT_CANDIDATES = (
    "/tmp/openLBMPM",
    "/root/oss-cache/openLBMPM",
)
REFERENCE_FILES = (
    "RKCG2D/AcceleratedRKGPU2D.py",
    "RKCG2D/RKD2Q9.py",
    "IniFiles/RKtwophasesetup2D.ini",
)


def geometric_reference_sha256() -> dict[str, str]:
    """sha256 of the openLBMPM source files the port was read from.

    Returns an empty dict when the checkout is not present (the run then
    records ``{}`` instead of inventing a digest).
    """
    import hashlib
    from pathlib import Path

    out: dict[str, str] = {}
    for root in REFERENCE_ROOT_CANDIDATES:
        base = Path(root)
        if not (base / "RKCG2D").is_dir():
            continue
        for rel in REFERENCE_FILES:
            f = base / rel
            if f.is_file():
                out[f"{root}/{rel}"] = hashlib.sha256(f.read_bytes()).hexdigest()
        if out:
            break
    return out


# ---------------------------------------------------------------------------
# Geometry tables (reference RKD2Q9.py:808-890)
# ---------------------------------------------------------------------------

#: Weighted offsets used by the reference to estimate the unit wall normal from
#: a binary solid mask.  ``(dy, dx, weight)``; the reference sums
#: ``weight * (dy, dx)`` over solid cells and normalises.  Weights are the
#: isotropy (ISO8) coefficients for radii 1, sqrt2, 2, sqrt5, sqrt8.
ISO8_OFFSETS: tuple[tuple[int, int, float], ...] = tuple(
    (dy, dx, w)
    for dy, dx, w in (
        # radius 1
        (0, 1, 4.0 / 21.0), (1, 0, 4.0 / 21.0), (0, -1, 4.0 / 21.0), (-1, 0, 4.0 / 21.0),
        # radius sqrt(2)
        (1, 1, 4.0 / 45.0), (1, -1, 4.0 / 45.0), (-1, -1, 4.0 / 45.0), (-1, 1, 4.0 / 45.0),
        # radius 2
        (0, 2, 1.0 / 60.0), (2, 0, 1.0 / 60.0), (0, -2, 1.0 / 60.0), (-2, 0, 1.0 / 60.0),
        # radius sqrt(5)
        (1, 2, 2.0 / 315.0), (2, 1, 2.0 / 315.0), (2, -1, 2.0 / 315.0), (1, -2, 2.0 / 315.0),
        (-1, -2, 2.0 / 315.0), (-2, -1, 2.0 / 315.0), (-2, 1, 2.0 / 315.0), (-1, 2, 2.0 / 315.0),
        # radius sqrt(8)
        (2, 2, 1.0 / 5040.0), (2, -2, 1.0 / 5040.0), (-2, -2, 1.0 / 5040.0), (-2, 2, 1.0 / 5040.0),
    )
)

#: D2Q9 weights in the reference's ``weightsCoeff`` ordering (RKD2Q9.py:302-303).
_D2Q9_WEIGHTS = (4.0 / 9.0, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36)
#: D2Q9 lattice directions in the reference's ordering (0, E, N, W, S, NE, NW, SW, SE).
_D2Q9_C = ((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1))


def _roll(a: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """``out[..., i, j] = a[..., i + dy, j + dx]`` (periodic in both axes)."""
    return torch.roll(a, shifts=(-dy, -dx), dims=(-2, -1))


def wall_normal_field(
    solid: torch.Tensor,
    periodic: tuple[bool, bool] = (True, False),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unit wall normals on the fluid nodes that see a solid within radius 2.

    Port of ``RKD2Q9.py:808-890``: for every fluid node, sum
    ``weight * (dy, dx)`` over solid cells inside the ISO8 stencil and
    normalise.  The normal points **from the fluid into the solid**.

    Returns ``(wet_mask, nsx, nsy)`` with ``(ny, nx)`` fields; ``nsx/nsy`` are
    zero where ``wet_mask`` is false.
    """
    if solid.ndim != 2:
        raise ValueError(f"solid must be 2-D, got shape {tuple(solid.shape)}")
    ny, nx = solid.shape
    float_mask = solid.to(torch.float64)
    acc_x = torch.zeros((ny, nx), dtype=torch.float64, device=solid.device)
    acc_y = torch.zeros_like(acc_x)
    seen = torch.zeros_like(acc_x)
    for dy, dx, w in ISO8_OFFSETS:
        nb_solid = _roll(float_mask, dy, dx)
        if not periodic[1]:
            if dy > 0:
                nb_solid[-dy:, :] = 0.0
            elif dy < 0:
                nb_solid[:-dy or None, :] = 0.0
        if not periodic[0]:
            if dx > 0:
                nb_solid[:, -dx:] = 0.0
            elif dx < 0:
                nb_solid[:, :-dx or None] = 0.0
        acc_x += w * dx * nb_solid
        acc_y += w * dy * nb_solid
        seen += w * nb_solid
    wet_mask = (~solid) & (seen > 0.0)
    norm = torch.sqrt(acc_x * acc_x + acc_y * acc_y)
    good = wet_mask & (norm > 1e-12)
    nsx = torch.where(good, acc_x / norm.clamp(min=1e-12), torch.zeros_like(acc_x))
    nsy = torch.where(good, acc_y / norm.clamp(min=1e-12), torch.zeros_like(acc_y))
    return good, nsx, nsy


def solid_color_values(
    phi: torch.Tensor,
    solid: torch.Tensor,
    periodic: tuple[bool, bool] = (True, False),
) -> torch.Tensor:
    """Phase value on solid nodes: D2Q9-weighted average of fluid neighbours.

    Port of ``AcceleratedRKGPU2D.py:1557-1578`` (``calColorValueOnSolid``).
    Solid nodes with no fluid neighbour get ``0`` (neutral); they never enter a
    gradient stencil of a fluid node that carries a wetting constraint.
    """
    if phi.shape != solid.shape:
        raise ValueError(f"shape mismatch: phi {tuple(phi.shape)} vs solid {tuple(solid.shape)}")
    ny, nx = phi.shape
    phi64 = phi.to(torch.float64)
    num = torch.zeros_like(phi64)
    den = torch.zeros_like(phi64)
    for (dy, dx), w in zip(_D2Q9_C[1:], _D2Q9_WEIGHTS[1:]):
        nb = _roll(phi64, dy, dx)
        nb_fluid = ~_roll(solid, dy, dx).bool()
        if not periodic[1]:
            if dy > 0:
                nb_fluid[-dy:, :] = False
            elif dy < 0:
                nb_fluid[:-dy or None, :] = False
        if not periodic[0]:
            if dx > 0:
                nb_fluid[:, -dx:] = False
            elif dx < 0:
                nb_fluid[:, :-dx or None] = False
        num += w * nb * nb_fluid.to(phi64)
        den += w * nb_fluid.to(phi64)
    val = torch.where(den > 0.0, num / den.clamp(min=1e-30), torch.zeros_like(num))
    return torch.where(solid, val, phi64)


def isotropic_color_gradient(
    phi: torch.Tensor,
    solid: torch.Tensor | None = None,
    periodic: tuple[bool, bool] = (True, False),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Isotropic 8-direction phase gradient, ``3 * sum_i w_i phi_i c_i``.

    Port of ``AcceleratedRKGPU2D.py:1585-1633`` (``calRKInitialGradient``).
    Solid neighbours are sampled from :func:`solid_color_values`; the factor 3
    is the reference's ``1/cs^2``.
    """
    if solid is not None:
        phi = solid_color_values(phi, solid, periodic=periodic)
    phi64 = phi.to(torch.float64)
    gx = torch.zeros_like(phi64)
    gy = torch.zeros_like(phi64)
    for (dy, dx), w in zip(_D2Q9_C[1:], _D2Q9_WEIGHTS[1:]):
        nb = _roll(phi64, dy, dx)
        gx += 3.0 * w * dx * nb
        gy += 3.0 * w * dy * nb
    return gx, gy


# ---------------------------------------------------------------------------
# The rotation kernel
# ---------------------------------------------------------------------------


def reference_rotation_scalar(
    gx: float,
    gy: float,
    nsx: float,
    nsy: float,
    cos_theta: float,
    sin_theta: float,
) -> tuple[float, float]:
    """Literal scalar transcription of ``updateColorGradientOnWettingNew``.

    ``AcceleratedRKGPU2D.py:2426-2492``, one node.  Kept as the executable
    reference for the equivalence test: the vectorised kernel must reproduce it
    to round-off.  Ties (``dist1 == dist2``) and the two degenerate guards
    leave the gradient unchanged, as in the source.
    """
    norm = math.sqrt(gx * gx + gy * gy)
    if norm > 1.0e-8:
        ugx, ugy = -gx / norm, -gy / norm
    else:
        ugx, ugy = 0.0, 0.0
    a = ugx * nsx + ugy * nsy
    a = min(max(a, -1.0), 1.0)
    theta_gs = math.acos(a)
    c1 = c2 = c3 = c4 = 0.0
    if abs(math.sin(theta_gs)) > 1.0e-9:
        c1 = sin_theta * math.cos(theta_gs) / math.sin(theta_gs)
        c2 = sin_theta / math.sin(theta_gs)
        c3 = -sin_theta * math.cos(theta_gs) / math.sin(theta_gs)
        c4 = -sin_theta / math.sin(theta_gs)
    n1x = (cos_theta - c1) * nsx + c2 * ugx
    n1y = (cos_theta - c1) * nsy + c2 * ugy
    n2x = (cos_theta - c3) * nsx + c4 * ugx
    n2y = (cos_theta - c3) * nsy + c4 * ugy
    d1 = math.sqrt((n1x - ugx) ** 2 + (n1y - ugy) ** 2)
    d2 = math.sqrt((n2x - ugx) ** 2 + (n2y - ugy) ** 2)
    if d1 < d2:
        return -norm * n1x, -norm * n1y
    if d1 > d2:
        return -norm * n2x, -norm * n2y
    return gx, gy


def rotate_wall_color_gradient(
    gx: torch.Tensor,
    gy: torch.Tensor,
    solid: torch.Tensor,
    theta_deg: float | Iterable[float],
    *,
    periodic: tuple[bool, bool] = (True, False),
    theta_by_direction: dict[tuple[int, int], float] | None = None,
    tol_norm: float = 1e-8,
    tol_delta: float = 1e-9,
    return_diagnostics: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Rotate the colour gradient at wall-adjacent fluid nodes (geometrical wetting).

    Args:
        gx, gy:    phase-gradient components ``(ny, nx)`` (``phi > 0`` wetting).
        solid:     boolean solid mask ``(ny, nx)``.
        theta_deg: prescribed static contact angle in degrees, inside the
                   ``phi > 0`` phase.  Either one value for all walls or a map
                   through ``theta_by_direction`` (keyed by the wall-normal
                   lattice direction, e.g. ``{(0, -1): 60.0, (0, 1): 90.0}``).
        periodic:  ``(x, y)`` periodicity of the domain.
        theta_by_direction: per-face override, e.g. a neutral top wall.

    Returns:
        ``(gx_out, gy_out)``; with ``return_diagnostics=True`` also a dict with
        the number of constrained nodes and the read-back angle statistics.
    """
    if gx.shape != gy.shape or gx.shape != solid.shape:
        raise ValueError(
            f"shape mismatch: gx {tuple(gx.shape)} gy {tuple(gy.shape)} solid {tuple(solid.shape)}"
        )
    wet_mask, nsx, nsy = wall_normal_field(solid, periodic=periodic)
    dtype = gx.dtype
    gx64 = gx.to(torch.float64)
    gy64 = gy.to(torch.float64)
    mag = torch.sqrt(gx64 * gx64 + gy64 * gy64)
    active = wet_mask & (mag > tol_norm)

    # per-node cos/sin of the prescribed angle
    base = float(theta_deg) if isinstance(theta_deg, (int, float)) else 0.0
    cos_t = torch.full_like(gx64, math.cos(math.radians(base)))
    sin_t = torch.full_like(gx64, math.sin(math.radians(base)))
    if theta_by_direction:
        for (dy, dx), th in theta_by_direction.items():
            dot = nsx * float(dx) + nsy * float(dy)
            face = wet_mask & (dot > 0.9)
            c = math.cos(math.radians(float(th)))
            s_ = math.sin(math.radians(float(th)))
            cos_t = torch.where(face, torch.full_like(cos_t, c), cos_t)
            sin_t = torch.where(face, torch.full_like(sin_t, s_), sin_t)

    # (1) unit gradient with the reference's minus sign
    inv = torch.where(active, 1.0 / mag.clamp(min=1e-30), torch.zeros_like(mag))
    ugx = -gx64 * inv
    ugy = -gy64 * inv

    # (2) two candidate unit vectors, nearest to g_hat
    a = (ugx * nsx + ugy * nsy).clamp(min=-1.0, max=1.0)
    delta = torch.sqrt(torch.clamp(1.0 - a * a, min=0.0))
    nonzero = delta > tol_delta
    inv_delta = torch.where(nonzero, 1.0 / delta.clamp(min=1e-30), torch.zeros_like(delta))
    px = (ugx - a * nsx) * inv_delta
    py = (ugy - a * nsy) * inv_delta
    n1x = cos_t * nsx + sin_t * px
    n1y = cos_t * nsy + sin_t * py
    n2x = cos_t * nsx - sin_t * px
    n2y = cos_t * nsy - sin_t * py
    d1 = torch.sqrt((n1x - ugx) ** 2 + (n1y - ugy) ** 2)
    d2 = torch.sqrt((n2x - ugx) ** 2 + (n2y - ugy) ** 2)
    use1 = (d1 < d2) & nonzero
    use2 = (d1 > d2) & nonzero
    chx = torch.where(use1, n1x, torch.where(use2, n2x, ugx))
    chy = torch.where(use1, n1y, torch.where(use2, n2y, ugy))

    # (3) write back with the preserved magnitude
    out_x = torch.where(active, -mag * chx, gx64)
    out_y = torch.where(active, -mag * chy, gy64)

    if not return_diagnostics:
        return out_x.to(dtype), out_y.to(dtype)

    with torch.no_grad():
        ang = readback_wall_angle(out_x, out_y, solid, periodic=periodic)
    diag = {
        "theta_deg": float(theta_deg) if isinstance(theta_deg, (int, float)) else None,
        "n_wetting_nodes": float(int(wet_mask.sum())),
        "n_active_nodes": float(int(active.sum())),
        "n_candidate1": float(int(use1.sum())),
        "n_candidate2": float(int(use2.sum())),
        "n_degenerate_delta": float(int((active & ~nonzero).sum())),
        "readback_median_deg": float(ang.median()) if ang.numel() else float("nan"),
        "readback_n": int(ang.numel()),
    }
    return out_x.to(dtype), out_y.to(dtype), diag


def readback_wall_angle(
    gx: torch.Tensor,
    gy: torch.Tensor,
    solid: torch.Tensor,
    *,
    periodic: tuple[bool, bool] = (True, False),
    tol_norm: float = 1e-8,
) -> torch.Tensor:
    """Angle implied by the gradient at wetting nodes, in degrees.

    ``theta = acos(g_hat . n_s)`` with ``g_hat = -g/|g|`` -- the reference's
    own ``theta_GS`` (``AcceleratedRKGPU2D.py:2453``).  After rotation this
    must equal the prescribed angle; before rotation it is the angle the
    interface actually has.
    """
    wet_mask, nsx, nsy = wall_normal_field(solid, periodic=periodic)
    gx64 = gx.to(torch.float64)
    gy64 = gy.to(torch.float64)
    mag = torch.sqrt(gx64 * gx64 + gy64 * gy64)
    active = wet_mask & (mag > tol_norm)
    if not bool(active.any()):
        return torch.zeros(0)
    a = (-gx64[active] * nsx[active] - gy64[active] * nsy[active]) / mag[active]
    a = a.clamp(min=-1.0, max=1.0)
    return torch.acos(a) * (180.0 / math.pi)


# ---------------------------------------------------------------------------
# Sessile-drop case (shared by the benchmark entry and the tests)
# ---------------------------------------------------------------------------


def sessile_drop_initial(
    ny: int,
    nx: int,
    radius: float,
    theta_deg: float,
    *,
    y_wall: float = 0.5,
    xi: float = 2.0,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Circular-cap drop that *exactly* satisfies the prescribed angle.

    ``phi = tanh((R - r)/xi)`` around a centre ``y_c = y_wall - R cos(theta)``
    (below the plate for ``theta < 90``), so the cap meets ``y = y_wall`` under
    ``theta``.  Returns rest distributions ``(f_r, f_b)`` with total density 1.
    """
    th = math.radians(theta_deg)
    yc = y_wall - radius * math.cos(th)
    y = torch.arange(ny, dtype=dtype).view(-1, 1)
    x = torch.arange(nx, dtype=dtype).view(1, -1)
    r = torch.sqrt((x - nx / 2.0) ** 2 + (y - yc) ** 2)
    phi = torch.tanh((radius - r) / xi)
    rho = torch.ones_like(phi)
    rho_r = 0.5 * rho * (1.0 + phi)
    rho_b = 0.5 * rho * (1.0 - phi)
    w = torch.tensor(_D2Q9_WEIGHTS, dtype=dtype).view(9, 1, 1)
    return w * rho_r.unsqueeze(0), w * rho_b.unsqueeze(0)


def run_sessile_drop_contact_angle(
    *,
    theta_deg: float,
    radius: float = 20.0,
    ny: int = 80,
    nx: int = 96,
    steps: int = 6000,
    A: float = 0.04,
    beta: float = 0.7,
    tau: float = 1.0,
    xi: float = 2.0,
    sample_every: int = 500,
    wetting: bool = True,
    top_theta_deg: float | None = 90.0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    verbose: bool = False,
) -> dict[str, Any]:
    """Relax a sessile drop and measure the contact angle (solver-level driver).

    This is the integration point used by the cross-validation entry
    ``benchmarks/xval_openlbmpm_wetting/run.py``: it drives the *host* solver
    :func:`tensorlbm.color_gradient2d.color_gradient_two_tau_step` with
    ``wall_contact_angle=theta_deg`` (the wetting condition ported here) and
    returns measured angles, so nothing is measured outside the real solver.

    ``wetting=False`` disables the boundary condition (neutral 90 deg
    gradient) while keeping the same initial cap -- the control arm.
    """
    from .cg_contact_angle import measure_contact_angle
    from .color_gradient2d import color_gradient_two_tau_step
    from .solver import stream

    # local imports keep this module importable on its own (no cycle)
    OPP = (0, 3, 2, 1, 4, 7, 8, 5, 6)

    solid = torch.zeros(ny, nx, dtype=torch.bool, device=device)
    solid[0, :] = True
    solid[ny - 1, :] = True
    solid3 = solid.unsqueeze(0)

    f_r, f_b = sessile_drop_initial(ny, nx, radius, theta_deg, xi=xi, dtype=dtype)
    f_r = f_r.to(device)
    f_b = f_b.to(device)

    history: list[dict[str, Any]] = []
    mass0 = None
    prev_phi = None
    t0 = __import__("time").time()

    def _phi() -> torch.Tensor:
        rr = f_r.sum(dim=0)
        rb = f_b.sum(dim=0)
        rho = (rr + rb).clamp(min=1e-12)
        return (rr - rb) / rho

    for it in range(1, steps + 1):
        if wetting:
            theta_map = None
            if top_theta_deg is not None:
                theta_map = {(0, 1): float(top_theta_deg)}
            step_kw = {"wall_contact_angle": float(theta_deg)}
            if theta_map:
                step_kw["wall_contact_angle_by_direction"] = theta_map
            f_r, f_b = color_gradient_two_tau_step(
                f_r, f_b, tau_r=tau, tau_b=tau, A=A, beta=beta,
                solid_mask=solid, visc_mode="mixture", **step_kw,
            )
        else:
            f_r, f_b = color_gradient_two_tau_step(
                f_r, f_b, tau_r=tau, tau_b=tau, A=A, beta=beta,
                solid_mask=solid, visc_mode="mixture",
            )
        f_r = stream(f_r)
        f_b = stream(f_b)
        f_r = torch.where(solid3, f_r[list(OPP)], f_r)
        f_b = torch.where(solid3, f_b[list(OPP)], f_b)

        if it % sample_every == 0 or it == 1:
            rr = f_r.sum(dim=0)
            rb = f_b.sum(dim=0)
            rho = rr + rb
            phi = ((rr - rb) / rho.clamp(min=1e-12)).detach().cpu()
            jx = ((f_r + f_b)[1] - (f_r + f_b)[3]).sum(dim=0).cpu()
            jy = ((f_r + f_b)[2] - (f_r + f_b)[4]).sum(dim=0).cpu()
            u_mag = torch.sqrt((jx / rho.cpu()) ** 2 + (jy / rho.cpu()) ** 2)
            umax = float(u_mag[~solid.cpu()].max())
            if mass0 is None:
                mass0 = (float(rr.sum()), float(rb.sum()))
            dphi = float((phi - prev_phi).abs().max()) if prev_phi is not None else float("nan")
            prev_phi = phi
            m = measure_contact_angle(phi.numpy(), solid.cpu().numpy(), 0.5)
            history.append(
                {
                    "step": it,
                    "theta_circle": m["theta_circle"],
                    "theta_local": m.get("theta_local", float("nan")),
                    "theta_cap": m["theta_cap"],
                    "u_max": umax,
                    "dphi_max": dphi,
                    "mass_drift_r": (float(rr.sum()) - mass0[0]) / mass0[0],
                    "mass_drift_b": (float(rb.sum()) - mass0[1]) / mass0[1],
                }
            )
            if verbose:
                print(
                    f"  theta={theta_deg:5.1f} wetting={int(wetting)} step={it:6d} "
                    f"circle={m['theta_circle']:7.2f} local={m.get('theta_local', float('nan')):7.2f} "
                    f"cap={m['theta_cap']:7.2f} |u|max={umax:.2e} dphi={dphi:.2e}",
                    flush=True,
                )

    phi_final = _phi().detach().cpu().numpy()
    final = measure_contact_angle(phi_final, solid.cpu().numpy(), 0.5)
    tail = [h for h in history if h["step"] > 0.7 * history[-1]["step"]] or history[-1:]
    tail_vals = [h["theta_circle"] for h in tail if math.isfinite(h["theta_circle"])]
    local_vals = [h["theta_local"] for h in tail if math.isfinite(h.get("theta_local", float("nan")))]
    return {
        "theta_target": float(theta_deg),
        "wetting": bool(wetting),
        "params": {
            "radius": radius, "ny": ny, "nx": nx, "steps": steps, "A": A, "beta": beta,
            "tau": tau, "xi": xi, "top_theta_deg": top_theta_deg, "dtype": str(dtype),
        },
        "theta_circle": final["theta_circle"],
        "theta_local": final.get("theta_local", float("nan")),
        "theta_cap": final["theta_cap"],
        "circle": final.get("circle"),
        "local": final.get("local"),
        "cap": final.get("cap"),
        "circle_error": final.get("circle_error"),
        "cap_error": final.get("cap_error"),
        "theta_tail_mean": float(sum(tail_vals) / len(tail_vals)) if tail_vals else float("nan"),
        "theta_local_tail_mean": float(sum(local_vals) / len(local_vals)) if local_vals else float("nan"),
        "err_pct_circle": 100.0 * (final["theta_circle"] - theta_deg) / theta_deg
        if math.isfinite(final["theta_circle"]) else float("nan"),
        "err_pct_local": 100.0 * (final.get("theta_local", float("nan")) - theta_deg) / theta_deg,
        "err_pct_cap": 100.0 * (final["theta_cap"] - theta_deg) / theta_deg
        if math.isfinite(final["theta_cap"]) else float("nan"),
        "u_max": history[-1]["u_max"],
        "mass_drift_r": history[-1]["mass_drift_r"],
        "mass_drift_b": history[-1]["mass_drift_b"],
        "dphi_max": history[-1]["dphi_max"],
        "wall_time_s": __import__("time").time() - t0,
        "history": history,
    }