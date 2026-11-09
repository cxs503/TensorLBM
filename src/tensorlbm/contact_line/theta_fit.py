"""Contact-angle measurement from a diffuse-interface density field.

Physics
-------
In a Shan-Chen (pseudopotential) simulation the interface is a diffuse layer of
finite width; the "contact line" is therefore not a line but a region.  The
operational definition used here is the **mid-density level set**

    rho(x) = rho_m = (rho_l + rho_v) / 2,

and the contact angle is the angle, measured *inside the liquid*, between that
level set and the wall plane.  Two estimators are implemented:

``cap``
    A single least-squares circle through the whole level set of a sessile
    droplet.  For a circle of radius ``R`` and centre ``(x_c, y_c)`` the tangent
    at the contact point ``(x_c +/- a, y_wall)`` satisfies

        cos(theta) = (y_c - y_wall) / R,   a = sqrt(R^2 - (y_c - y_wall)^2)

    hence ``theta = atan2(a, y_c - y_wall)``.  This is the classical
    "circle-fit" method; it is insensitive to point ordering and is the most
    accurate estimator when the droplet is close to a spherical cap.

``window``
    A local circle fit restricted to the interface band
    ``y_wall + offset <= y <= y_wall + window`` on one side of the drop
    (``front`` = advancing, ``rear`` = receding).  Required for a *driven*
    droplet because gravity/viscous bending makes the front and rear arcs
    differ; the whole-cap fit would return their average.

Because the interface is diffuse, the level set is displaced by a small,
roughly constant distance from the sharp-interface position; this biases the
*contact width* but not the *angle* (a uniform normal offset changes ``R`` and
``a`` together), which is why angles are the robust observable of this node.

Uncertainty
-----------
Three independent contributions are combined in quadrature
(:func:`angle_estimate`):

* **bootstrap** — resampling the extracted level-set points (statistical);
* **window** — the spread of the estimator over a set of fit-window heights
  (systematic, catches curvature/truncation bias);
* **wall-offset** — the shift produced by moving the assumed wall plane by
  +/-0.5 lattice unit (half-way bounce-back position uncertainty).

Reporting the three separately is what makes criterion C8 (angle uncertainty)
machine-checkable instead of a hand-waved error bar.

References
----------
Kasa (1976) IEEE Trans. Instrum. Meas. 25, 8 (algebraic circle fit).
Cox (1986) J. Fluid Mech. 168, 169 (contact angle as the boundary condition).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import torch

__all__ = [
    "AngleEstimate",
    "CircleFit",
    "TangentAngle",
    "local_tangent_angle",
    "wall_contact_angle",
    "angle_estimate",
    "circle_fit",
    "contact_angle_from_circle",
    "interface_points",
    "level_fraction",
]


@dataclass(frozen=True)
class CircleFit:
    """Algebraic least-squares circle through a point cloud (``x`` = column)."""

    xc: float
    yc: float
    r: float
    rms: float
    n: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class AngleEstimate:
    """Contact angle estimate with its uncertainty budget (degrees)."""

    theta_deg: float
    unc_deg: float
    side: str
    y_wall: float
    n_points: int
    rms: float
    radius_fit: float
    contact_half_width: float
    boot_std_deg: float = 0.0
    window_std_deg: float = 0.0
    wall_std_deg: float = 0.0
    theta_by_window: dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        d = asdict(self)
        d["theta_by_window"] = {int(k): v for k, v in self.theta_by_window.items()}
        return d


# ---------------------------------------------------------------------------
# Level-set extraction
# ---------------------------------------------------------------------------


def level_fraction(rho_l: float, rho_v: float) -> float:
    """Mid-density level used to define the interface (``(rho_l+rho_v)/2``)."""
    return 0.5 * (float(rho_l) + float(rho_v))


def interface_points(
    rho: torch.Tensor,
    level: float,
    solid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Extract the ``rho = level`` isoline as an unordered point cloud.

    Linear (marching-squares) interpolation on all grid edges; edges touching a
    solid cell are skipped so the wall layer cannot contaminate the fit.

    Args:
        rho:   Density field, shape ``(ny, nx)``.
        level: Isoline value (usually :func:`level_fraction`).
        solid: Optional boolean mask ``(ny, nx)`` of solid cells.

    Returns:
        Float tensor of shape ``(n, 2)`` with ``(x, y)`` in *cell-centre*
        coordinates (``x = j``, ``y = i``).
    """
    if rho.ndim != 2:
        raise ValueError(f"rho must be 2-D, got shape {tuple(rho.shape)}")
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []

    # Horizontal edges: between (i, j) and (i, j+1)
    a = rho[:, :-1]
    b = rho[:, 1:]
    cross = (a - level) * (b - level) < 0.0
    if solid is not None:
        cross = cross & ~(solid[:, :-1] | solid[:, 1:])
    if bool(cross.any()):
        ii, jj = cross.nonzero(as_tuple=True)
        av = a[ii, jj]
        bv = b[ii, jj]
        t = (level - av) / (bv - av)
        xs.append(jj.to(torch.float32) + t)
        ys.append(ii.to(torch.float32))

    # Vertical edges: between (i, j) and (i+1, j)
    a = rho[:-1, :]
    b = rho[1:, :]
    cross = (a - level) * (b - level) < 0.0
    if solid is not None:
        cross = cross & ~(solid[:-1, :] | solid[1:, :])
    if bool(cross.any()):
        ii, jj = cross.nonzero(as_tuple=True)
        av = a[ii, jj]
        bv = b[ii, jj]
        t = (level - av) / (bv - av)
        xs.append(jj.to(torch.float32))
        ys.append(ii.to(torch.float32) + t)

    if not xs:
        return torch.zeros((0, 2), dtype=torch.float32, device=rho.device)
    return torch.stack([torch.cat(xs), torch.cat(ys)], dim=1)


# ---------------------------------------------------------------------------
# Circle fitting
# ---------------------------------------------------------------------------


def circle_fit(points: torch.Tensor, refine: int = 3) -> CircleFit:
    """Algebraic (Kasa) circle fit refined by Gauss-Newton on the radius residual.

    Args:
        points: ``(n, 2)`` point cloud.
        refine: Number of Gauss-Newton refinements (0 = pure algebraic fit).

    Returns:
        :class:`CircleFit` with the RMS radial residual.

    Raises:
        ValueError: if fewer than 3 points are supplied.
    """
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    n = int(points.shape[0])
    if n < 3:
        raise ValueError(f"need at least 3 points for a circle fit, got {n}")

    # Work in a translated/scaled frame for conditioning.
    mean = points.mean(dim=0)
    q = points - mean
    scale = float(q.abs().max())
    if scale <= 0.0:
        raise ValueError("degenerate point cloud")
    q = q / scale

    x = q[:, 0].to(torch.float64)
    y = q[:, 1].to(torch.float64)
    A = torch.stack([x, y, torch.ones_like(x)], dim=1)
    rhs = -(x * x + y * y)
    sol = torch.linalg.lstsq(A, rhs.unsqueeze(1)).solution.squeeze(1)
    xc, yc = -0.5 * sol[0], -0.5 * sol[1]
    r2 = xc * xc + yc * yc - sol[2]
    if float(r2) <= 0.0:
        raise ValueError("algebraic circle fit produced a negative radius")
    r = torch.sqrt(r2)

    # Gauss-Newton on the geometric residual |p - c| - r
    for _ in range(int(refine)):
        dx = x - xc
        dy = y - yc
        dist = torch.sqrt(dx * dx + dy * dy).clamp_min(1e-12)
        resid = dist - r
        jac = torch.stack([-dx / dist, -dy / dist, -torch.ones_like(dist)], dim=1)
        step = torch.linalg.lstsq(jac, -resid.unsqueeze(1)).solution.squeeze(1)
        xc = xc + step[0]
        yc = yc + step[1]
        r = r + step[2]

    dist = torch.sqrt((x - xc) ** 2 + (y - yc) ** 2)
    rms = float((dist - r).pow(2).mean().sqrt()) * scale
    return CircleFit(
        xc=float(xc) * scale + float(mean[0]),
        yc=float(yc) * scale + float(mean[1]),
        r=float(r) * scale,
        rms=rms,
        n=n,
    )


def contact_angle_from_circle(fit: CircleFit, y_wall: float) -> tuple[float, float]:
    """Contact angle (degrees, inside the liquid) of a fitted circle at a wall.

    Args:
        fit:    Circle fitted to the interface.
        y_wall: Height of the wall plane.

    Returns:
        ``(theta_deg, contact_half_width)``.  ``contact_half_width`` is
        ``NaN`` when the circle does not reach the wall plane.
    """
    hc = fit.yc - float(y_wall)
    disc = fit.r * fit.r - hc * hc
    if disc < 0.0:
        return float("nan"), float("nan")
    a = math.sqrt(disc)
    theta = math.degrees(math.atan2(a, hc))
    if theta < 0.0:
        theta += 180.0
    return theta, a


# ---------------------------------------------------------------------------
# Putting it together
# ---------------------------------------------------------------------------


def _select(
    points: torch.Tensor,
    y_wall: float,
    side: str,
    centre_x: float,
    window: float,
    offset: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Band + side selection of interface points.

    Returns ``(selected, all_side)`` where ``all_side`` is the side selection
    without the height band (used to measure the full cap when possible).
    """
    if points.numel() == 0:
        return points, points
    x = points[:, 0]
    y = points[:, 1]
    if side == "front":
        side_mask = x >= centre_x
    elif side == "rear":
        side_mask = x <= centre_x
    elif side in ("cap", "all"):
        side_mask = torch.ones_like(x, dtype=torch.bool)
    else:
        raise ValueError(f"side must be 'front', 'rear' or 'cap', got {side!r}")
    band = (y >= y_wall + offset) & (y <= y_wall + window)
    return points[side_mask & band], points[side_mask]


def angle_estimate(
    points: torch.Tensor,
    y_wall: float,
    *,
    side: str = "cap",
    centre_x: float | None = None,
    windows: tuple[float, ...] = (8.0, 10.0, 12.0, 14.0),
    offset: float = 2.0,
    bootstrap: int = 64,
    seed: int = 0,
) -> AngleEstimate:
    """Contact angle from an interface point cloud, with an uncertainty budget.

    Args:
        points:    ``(n, 2)`` interface point cloud (see :func:`interface_points`).
        y_wall:    Wall plane height.
        side:      ``"cap"`` (whole droplet), ``"front"`` (advancing) or
                   ``"rear"`` (receding).
        centre_x:  Streamwise centre used to split front/rear (mandatory for
                   those sides; defaults to the point-cloud median).
        windows:   Fit-band heights (above the wall) swept for the systematic
                   window uncertainty.
        offset:    Points closer than this to the wall are excluded — the wall
                   adsorption layer (a density peak one cell thick) is not part
                   of the interface.
        bootstrap: Number of bootstrap resamples (0 disables).
        seed:      RNG seed for the bootstrap.

    Returns:
        :class:`AngleEstimate`.  ``theta_deg`` is the mean over ``windows`` and
        ``unc_deg`` the quadrature sum of the bootstrap, window and wall-offset
        contributions.
    """
    if points.numel() == 0:
        raise ValueError("no interface points")
    if centre_x is None:
        if side in ("front", "rear"):
            raise ValueError("centre_x is required for side='front'/'rear'")
        centre_x = float(points[:, 0].median())

    per_window: dict[int, float] = {}
    per_window_fit: dict[int, CircleFit] = {}
    for w in windows:
        sel, _ = _select(points, y_wall, side, centre_x, float(w), offset)
        if sel.shape[0] < 5:
            continue
        fit = circle_fit(sel)
        theta, _ = contact_angle_from_circle(fit, y_wall)
        if math.isnan(theta):
            continue
        per_window[int(w)] = theta
        per_window_fit[int(w)] = fit
    if not per_window:
        raise ValueError("no fit window produced a valid contact angle")

    ws = sorted(per_window)
    thetas = torch.tensor([per_window[w] for w in ws], dtype=torch.float64)
    theta_mean = float(thetas.mean())
    window_std = float(thetas.std(unbiased=True)) if len(ws) > 1 else 0.0

    # Reference fit: median window (best compromise between truncation bias and
    # noise amplification).
    ref_w = ws[len(ws) // 2]
    ref_fit = per_window_fit[ref_w]
    ref_theta, ref_a = contact_angle_from_circle(ref_fit, y_wall)

    boot_std = 0.0
    if bootstrap > 0:
        sel, _ = _select(points, y_wall, side, centre_x, float(ref_w), offset)
        g = torch.Generator().manual_seed(int(seed))
        idx = torch.randint(0, sel.shape[0], (int(bootstrap), sel.shape[0]), generator=g)
        vals = []
        for b in range(int(bootstrap)):
            sample = sel[idx[b]]
            try:
                fit = circle_fit(sample, refine=2)
            except ValueError:
                continue
            th, _ = contact_angle_from_circle(fit, y_wall)
            if not math.isnan(th):
                vals.append(th)
        if len(vals) > 2:
            boot_std = float(torch.tensor(vals, dtype=torch.float64).std(unbiased=True))

    wall_std = 0.0
    sel, _ = _select(points, y_wall, side, centre_x, float(ref_w), offset)
    if sel.shape[0] >= 5:
        try:
            fit_p = circle_fit(sel)
            th_p, _ = contact_angle_from_circle(fit_p, y_wall + 0.5)
            th_m, _ = contact_angle_from_circle(fit_p, y_wall - 0.5)
            if not (math.isnan(th_p) or math.isnan(th_m)):
                wall_std = 0.5 * abs(th_p - th_m)
        except ValueError:
            wall_std = 0.0

    unc = math.sqrt(boot_std**2 + window_std**2 + wall_std**2)
    return AngleEstimate(
        theta_deg=ref_theta if not math.isnan(ref_theta) else theta_mean,
        unc_deg=unc,
        side=side,
        y_wall=float(y_wall),
        n_points=int(sel.shape[0]),
        rms=float(ref_fit.rms),
        radius_fit=float(ref_fit.r),
        contact_half_width=float(ref_a),
        boot_std_deg=boot_std,
        window_std_deg=window_std,
        wall_std_deg=wall_std,
        theta_by_window=per_window,
    )


def angle_from_density(
    rho: torch.Tensor,
    level: float,
    y_wall: float,
    *,
    solid: torch.Tensor | None = None,
    side: str = "cap",
    centre_x: float | None = None,
    **kwargs: object,
) -> AngleEstimate:
    """Convenience wrapper: density field -> interface points -> angle estimate."""
    pts = interface_points(rho, level, solid)
    return angle_estimate(pts, y_wall, side=side, centre_x=centre_x, **kwargs)  # type: ignore[arg-type]

# ---------------------------------------------------------------------------
# Local-tangent estimator (robust for a deformed, driven droplet)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TangentAngle:
    """Contact angle from the local interface tangent, with a window sweep."""

    theta_deg: float
    unc_deg: float
    side: str
    y_wall: float
    n_points: int
    slope_dxdy: float
    rms: float
    theta_by_window: dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        d = asdict(self)
        d["theta_by_window"] = {int(k): v for k, v in self.theta_by_window.items()}
        return d


def local_tangent_angle(
    points: torch.Tensor,
    y_wall: float,
    *,
    side: str,
    centre_x: float = 0.0,
    windows: tuple[float, ...] = (6.0, 8.0, 10.0, 12.0),
    offset: float = 2.0,
    order: int = 2,
) -> TangentAngle:
    """Contact angle from the tangent of a local ``x(y)`` fit at the wall.

    The interface is treated as a single-valued function ``x(y)`` (true for any
    monotone contact-line profile), fitted on a band above the wall and
    extrapolated to ``y = y_wall``.  The angle measured inside the liquid is

        front (liquid on the left):   theta = atan2(1, +dx/dy)
        rear  (liquid on the right):  theta = atan2(1, -dx/dy)

    which gives ``90 deg`` for a vertical interface, ``-> 0`` for a flat
    interface and ``-> 180 deg`` for an overhanging one.  Unlike the circle fit,
    this estimator makes no assumption that the droplet is a circular cap, so it
    stays meaningful for the asymmetric shapes of a strongly driven droplet
    (where the front is compressed and the rear stretched).

    Args:
        points:    Interface point cloud with ``x`` re-centred on the droplet.
        y_wall:    Wall plane height.
        side:      ``"front"`` or ``"rear"``.
        centre_x:  Streamwise centre (usually 0 after re-centring).
        windows:   Fit-band heights swept for the systematic uncertainty.
        offset:    Wall-layer exclusion distance.
        order:     Polynomial order of the ``x(y)`` fit (2 or 3).

    Returns:
        :class:`TangentAngle`.
    """
    if side not in ("front", "rear"):
        raise ValueError("side must be 'front' or 'rear'")
    per_window: dict[int, float] = {}
    rms_ref = float("nan")
    slope_ref = float("nan")
    n_ref = 0
    for w in windows:
        band = (points[:, 1] >= y_wall + offset) & (points[:, 1] <= y_wall + float(w))
        sel = points[band]
        if side == "front":
            sel = sel[sel[:, 0] >= centre_x]
        else:
            sel = sel[sel[:, 0] <= centre_x]
        if sel.shape[0] < order + 2:
            continue
        # One x per row (mean), then fit x(t) with t = y - y_wall.  Averaging by
        # row removes the duplicate marching-squares points on horizontal edges.
        y = sel[:, 1]
        x = sel[:, 0]
        rows = torch.unique(y)
        if rows.numel() < order + 2:
            continue
        xs = torch.stack([x[y == r].mean() for r in rows])
        t = (rows - float(y_wall)).to(torch.float64)
        xs = xs.to(torch.float64)
        # Polynomial fit in a shifted/scaled variable for conditioning.
        tmax = float(t.max())
        tn = t / tmax
        V = torch.stack([tn**k for k in range(order + 1)], dim=1)
        coeffs = torch.linalg.lstsq(V, xs.unsqueeze(1)).solution.squeeze(1)
        # dx/dy at the wall: t = 0 -> tn = 0, dtn/dt = 1/tmax
        slope = float(coeffs[1]) / tmax
        resid = xs - (V @ coeffs)
        if w == windows[len(windows) // 2]:
            rms_ref = float(resid.pow(2).mean().sqrt())
            slope_ref = slope
            n_ref = int(sel.shape[0])
        # For a circular cap of contact angle theta the front contact has
        # dx/dy = cot(theta) (the profile bulges outward for theta < 90 deg and
        # leans inward for theta > 90 deg); the rear edge is its mirror image.
        if side == "front":
            theta = math.degrees(math.atan2(1.0, slope))
        else:
            theta = math.degrees(math.atan2(1.0, -slope))
        per_window[int(w)] = theta
    if not per_window:
        raise ValueError("no fit window produced a valid tangent angle")
    vals = torch.tensor(list(per_window.values()), dtype=torch.float64)
    unc = float(vals.std(unbiased=True)) if vals.numel() > 1 else 0.0
    ref_w = windows[len(windows) // 2]
    theta_ref = per_window.get(int(ref_w), float(vals.mean()))
    return TangentAngle(
        theta_deg=float(theta_ref),
        unc_deg=unc,
        side=side,
        y_wall=float(y_wall),
        n_points=n_ref,
        slope_dxdy=float(slope_ref),
        rms=rms_ref,
        theta_by_window=per_window,
    )


def wall_contact_angle(
    points: torch.Tensor,
    y_ref: float,
    *,
    below: bool = True,
    windows: tuple[float, ...] = (6.0, 8.0, 10.0, 12.0),
    offset: float = 2.0,
    order: int = 2,
    branch: str | None = None,
) -> TangentAngle:
    """Contact angle at a horizontal wall from the local ``x(y)`` profile.

    Same estimator as :func:`local_tangent_angle` but for a wall that spans the
    whole domain (channel meniscus) instead of a droplet contact: the fit band is
    taken on the fluid side of the wall and the liquid is assumed to lie on the
    ``-x`` side of the interface.

    Args:
        points: Interface point cloud.
        y_ref:  Wall plane height.
        below:  ``True`` if the wall is *below* the fluid (bottom wall),
                ``False`` for the top wall.
        windows: Fit-band heights swept for the systematic uncertainty.
        offset: Wall-layer exclusion distance.
        order:  Polynomial order of the ``x(y)`` fit.
        branch: How to reduce several crossings in one row to a single ``x``:
                ``None`` (default) takes the mean, which is correct for a
                single-valued interface such as a channel meniscus;
                ``"max"``/``"min"`` take the right/left crossing and are needed
                when two interfaces are present (e.g. a wedge with independent
                front and rear angles).

    Returns:
        :class:`TangentAngle` with ``side`` set to ``"bottom"``/``"top"``.
    """
    per_window: dict[int, float] = {}
    slope_ref = float("nan")
    rms_ref = float("nan")
    n_ref = 0
    mid = windows[len(windows) // 2]
    for w in windows:
        if below:
            band = (points[:, 1] >= y_ref + offset) & (points[:, 1] <= y_ref + float(w))
        else:
            band = (points[:, 1] <= y_ref - offset) & (points[:, 1] >= y_ref - float(w))
        sel = points[band]
        if sel.shape[0] < order + 2:
            continue
        y = sel[:, 1]
        x = sel[:, 0]
        rows = torch.unique(y)
        if rows.numel() < order + 2:
            continue
        if branch == "max":
            xs = torch.stack([x[y == r].max() for r in rows])
        elif branch == "min":
            xs = torch.stack([x[y == r].min() for r in rows])
        else:
            xs = torch.stack([x[y == r].mean() for r in rows])
        t = (rows - float(y_ref)).to(torch.float64)
        xs = xs.to(torch.float64)
        tmax = float(t.abs().max())
        if tmax <= 0.0:  # pragma: no cover
            continue
        tn = t / tmax
        V = torch.stack([tn**k for k in range(order + 1)], dim=1)
        coeffs = torch.linalg.lstsq(V, xs.unsqueeze(1)).solution.squeeze(1)
        slope = float(coeffs[1]) / tmax  # dx/dy
        if w == mid:
            slope_ref = slope
            rms_ref = float((xs - (V @ coeffs)).pow(2).mean().sqrt())
            n_ref = int(sel.shape[0])
        theta = math.degrees(math.atan2(1.0, slope if below else -slope))
        per_window[int(w)] = theta
    if not per_window:
        raise ValueError("no fit window produced a valid wall angle")
    vals = torch.tensor(list(per_window.values()), dtype=torch.float64)
    unc = float(vals.std(unbiased=True)) if vals.numel() > 1 else 0.0
    return TangentAngle(
        theta_deg=float(per_window.get(int(mid), float(vals.mean()))),
        unc_deg=unc,
        side="bottom" if below else "top",
        y_wall=float(y_ref),
        n_points=n_ref,
        slope_dxdy=float(slope_ref),
        rms=float(rms_ref),
        theta_by_window=per_window,
    )
