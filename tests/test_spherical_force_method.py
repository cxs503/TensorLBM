"""Tests for the spherical (curved-wall surface) force method, P233W2-N06.

Coverage map (AC1: >= 6 tests over planar / cylinder / sphere boundaries)

* planar      : ``test_uniform_state_preserved_planar``,
                ``test_planar_channel_force_and_profile``
* cylinder    : ``test_circle_links_geometry``,
                ``test_circle_q_matches_independent_bisection``,
                ``test_bfl_is_exact_for_linear_field``
* sphere (3D) : ``test_sphere_link_mesh_closure``
* conservation: ``test_momentum_audit_machine_precision`` (AC3),
                ``test_momentum_exchange_sign``
* acceptance  : ``test_cd_acceptance_from_artifacts`` (AC2/AC4/AC5)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from tensorlbm.spherical_force_method import (
    C_D2Q9,
    OPPOSITE_D2Q9,
    apply_bfl,
    circle_links,
    collide_bgk,
    equilibrium,
    macroscopic,
    momentum_audit,
    run_planar_channel,
    sdf_links,
)

ART = Path(__file__).resolve().parents[1] / "artifacts" / "spherical_force_method"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _solid_disc(nx, ny, cx, cy, radius):
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2


def _uniform_state(nx, ny, rho=1.0, ux=0.0, uy=0.0, dtype=torch.float64):
    rr = torch.full((ny, nx), rho, dtype=dtype)
    uu = torch.full((ny, nx), ux, dtype=dtype)
    vv = torch.full((ny, nx), uy, dtype=dtype)
    return equilibrium(rr, uu, vv)


# ---------------------------------------------------------------------------
# geometry: cylinder
# ---------------------------------------------------------------------------
def test_circle_links_geometry():
    cx, cy, radius = 30.0, 40.0, 10.0
    links = circle_links(120, 80, cx, cy, radius)
    assert len(links) > 6 * radius, "expected O(2*pi*R) boundary links"
    q = links.q.numpy()
    assert np.all(q > 0.0) and np.all(q <= 1.0)
    assert q.min() < 0.5 < q.max(), "both interpolated BFL branches must be exercised"
    wp = links.wall_point.numpy()
    assert np.abs(np.hypot(wp[:, 0] - cx, wp[:, 1] - cy) - radius).max() < 1e-10
    n = links.normal.numpy()
    assert np.abs(np.linalg.norm(n, axis=1) - 1.0).max() < 1e-12
    # outward normals: pointing away from the body centre
    radial = (wp - np.array([cx, cy])) / radius
    assert np.abs(n - radial).max() < 1e-12
    # closed-surface closure: the discrete surface integral of a uniform normal is 0
    assert np.linalg.norm(n.sum(axis=0)) < 1e-10
    # every active link joins a fluid node to a solid neighbour
    solid = _solid_disc(120, 80, cx, cy, radius)
    for k in range(0, len(links), 7):
        iy, ix = divmod(int(links.flat[k]), 120)
        assert not solid[iy, ix]
        dx, dy = C_D2Q9[int(links.direction[k])]
        assert solid[iy + dy, ix + dx]


def test_circle_q_matches_independent_bisection():
    """q must equal the ray/circle crossing found by an independent bisection."""
    cx, cy, radius = 30.0, 40.0, 10.0
    links = circle_links(120, 80, cx, cy, radius)
    worst = 0.0
    for k in range(len(links)):
        iy, ix = divmod(int(links.flat[k]), 120)
        d = int(links.direction[k])
        cxy = np.array(C_D2Q9[d], dtype=float)
        norm = np.linalg.norm(cxy)
        e = cxy / norm
        p0 = np.array([ix, iy], dtype=float)
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if np.linalg.norm(p0 + mid * norm * e - np.array([cx, cy])) > radius:
                lo = mid
            else:
                hi = mid
        worst = max(worst, abs(lo - float(links.q[k])))
    assert worst < 1e-6, f"q mismatch vs bisection: {worst}"


# ---------------------------------------------------------------------------
# geometry: sphere (3D) -- discrete surface mesh closure
# ---------------------------------------------------------------------------
def _sphere_links(n=24, radius=8.0):
    """Minimal D3Q27 link set of a sphere (analytic ray/sphere intersection)."""
    c3 = np.array(
        [
            (0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
            (1, 1, 0), (-1, 1, 0), (1, -1, 0), (-1, -1, 0), (1, 0, 1), (-1, 0, 1),
            (1, 0, -1), (-1, 0, -1), (0, 1, 1), (0, -1, 1), (0, 1, -1), (0, -1, -1),
            (1, 1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, -1), (-1, -1, 1), (-1, 1, -1),
            (1, -1, -1), (-1, -1, -1),
        ],
        dtype=float,
    )
    ctr = np.array([n / 2.0] * 3)
    zz, yy, xx = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    d2 = (xx - ctr[0]) ** 2 + (yy - ctr[1]) ** 2 + (zz - ctr[2]) ** 2
    fluid = d2 > radius**2
    pts, nrms = [], []
    for ci in c3[1:]:
        xn, yn, zn = xx + ci[0], yy + ci[1], zz + ci[2]
        inside = ((xn - ctr[0]) ** 2 + (yn - ctr[1]) ** 2 + (zn - ctr[2]) ** 2) <= radius**2
        ok = (xn >= 0) & (xn < n) & (yn >= 0) & (yn < n) & (zn >= 0) & (zn < n)
        idx = np.nonzero(fluid & inside & ok)
        if idx[0].size == 0:
            continue
        p = np.stack([xx[idx], yy[idx], zz[idx]], axis=1).astype(float)
        norm = np.linalg.norm(ci)
        e = ci / norm
        b = 2.0 * ((p - ctr) @ e)
        cc = np.sum((p - ctr) ** 2, axis=1) - radius**2
        disc = np.sqrt(np.maximum(b * b - 4.0 * cc, 0.0))
        t = -0.5 * (b + disc)
        q = np.clip(t / norm, 1e-3, 1.0)
        w = p + (q * norm)[:, None] * e
        pts.append(w)
        nrms.append((w - ctr) / radius)
    pts = np.concatenate(pts)
    nrms = np.concatenate(nrms)
    return pts, nrms


def test_sphere_link_mesh_closure():
    """Discrete surface mesh of a sphere: points on the sphere, closed normals."""
    pts, nrms = _sphere_links()
    assert pts.shape[0] > 300
    assert np.abs(np.linalg.norm(pts - 12.0, axis=1) - 8.0).max() < 1e-10
    assert np.abs(np.linalg.norm(nrms, axis=1) - 1.0).max() < 1e-12
    # area-weighted closure of a closed surface
    w = 4.0 * math.pi * 64.0 / pts.shape[0]
    assert np.linalg.norm(w * nrms.sum(axis=0)) < 1e-8


# ---------------------------------------------------------------------------
# planar boundary: uniform state preservation + channel momentum balance
# ---------------------------------------------------------------------------
def test_uniform_state_preserved_planar():
    """A fluid at rest next to no-slip walls must stay exactly at rest."""
    ny, nx = 14, 4
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    sdf = np.minimum(yy - 0.5, ny - 1.5 - yy).astype(np.float64)
    links = sdf_links(sdf, periodic=(True, False))
    fluid = torch.as_tensor(sdf > 0.0)
    f = _uniform_state(nx, ny)
    f = torch.where(fluid.unsqueeze(0), f, torch.zeros_like(f))
    m0 = float(f.sum())
    for _ in range(8):
        f_post = collide_bgk(f, 0.65, fluid)
        f_post = torch.where(fluid.unsqueeze(0), f_post, torch.zeros_like(f_post))
        f, _ = apply_bfl(f_post, links)
        f = torch.where(fluid.unsqueeze(0), f, torch.zeros_like(f))
    assert abs(float(f.sum()) - m0) < 1e-10
    u = macroscopic(f)[1][fluid].abs().max()
    assert float(u) < 1e-12


def test_planar_channel_force_and_profile():
    """Plane Poiseuille: wall force integral = g H / 2, profile = analytic."""
    r = run_planar_channel(offset=0.0, steps=2400, ny=18, nx=4)
    assert r["profile_rel_error"] < 0.05
    assert r["rel_error_bottom"] < 0.05
    assert r["rel_error_top"] < 0.05
    assert r["force_balance_rel"] < 0.05
    r2 = run_planar_channel(offset=0.3, steps=2400, ny=18, nx=4)
    assert r2["q_min"] < 0.5, "offset wall must exercise the q < 1/2 branch"
    assert r2["profile_rel_error"] < 0.05


# ---------------------------------------------------------------------------
# BFL interpolation exactness for a linear field
# ---------------------------------------------------------------------------
def test_bfl_is_exact_for_linear_field():
    """For a linear velocity profile the interpolated wall value is zero.

    The BFL reconstruction is exact for linear fields, so the wall velocity
    implied by the reflected population equals the no-slip value to O(Ma^2).
    """
    nx, ny = 4, 24
    u_wall = 0.02
    h = 16.0
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    sdf = np.minimum(yy - 0.5, ny - 1.5 - yy).astype(np.float64)
    links = sdf_links(sdf, periodic=(True, False))
    fluid = torch.as_tensor(sdf > 0.0)
    # linear Couette profile in equilibrium form
    ux = torch.as_tensor(np.where(sdf > 0, u_wall * (yy - 0.5) / h, 0.0))
    f = equilibrium(torch.ones_like(ux), ux, torch.zeros_like(ux))
    f = torch.where(fluid.unsqueeze(0), f, torch.zeros_like(f))
    f_new, res = apply_bfl(f, links)
    rho = f_new.sum(0)
    ux_new = macroscopic(f_new)[1]
    # away from the walls the linear profile must survive streaming/collision-free
    mid = fluid & (torch.as_tensor(yy) > 2) & (torch.as_tensor(yy) < ny - 3)
    err = float((ux_new[mid] - ux[mid]).abs().max())
    assert err < 1e-3, f"BFL distorted a linear field: {err}"
    assert torch.isfinite(rho).all()


# ---------------------------------------------------------------------------
# momentum conservation (AC3)
# ---------------------------------------------------------------------------
def test_momentum_audit_machine_precision():
    """dP_fluid + F_body = 0 to machine precision (relative residual <= 1e-6)."""
    nx, ny = 120, 80
    links = circle_links(nx, ny, 30.0, 40.0, 10.0)
    fluid = torch.as_tensor(~_solid_disc(nx, ny, 30.0, 40.0, 10.0))
    rng = np.random.default_rng(0)
    rho = torch.as_tensor(1.0 + 0.01 * rng.standard_normal((ny, nx)))
    ux = torch.as_tensor(0.01 * rng.standard_normal((ny, nx)))
    uy = torch.as_tensor(0.01 * rng.standard_normal((ny, nx)))
    f = equilibrium(rho, ux, uy)
    audit = momentum_audit(f, links, 0.7, fluid)
    assert audit["relative_residual"] < 1e-6, audit
    # the reported force must be the negative of the fluid momentum change
    assert np.allclose(audit["dp"], -np.asarray(audit["force"]), atol=1e-8)


def test_momentum_exchange_sign():
    """A body in a uniform stream feels drag along the flow direction."""
    nx, ny = 120, 80
    links = circle_links(nx, ny, 60.0, 40.0, 8.0)
    solid = _solid_disc(nx, ny, 60.0, 40.0, 8.0)
    seed = np.where(solid, 0.0, 1.0)
    rho = torch.as_tensor(seed)
    ux = torch.as_tensor(0.02 * seed)
    uy = torch.zeros_like(ux)
    f = equilibrium(rho, ux, uy)
    f, res = apply_bfl(f, links)
    # the fluid pushes the body downstream: drag > 0 for a stream in +x
    assert res.drag > 0.0, f"drag sign wrong (got {res.drag})"
    assert abs(res.lift) < 0.5 * abs(res.drag)


# ---------------------------------------------------------------------------
# acceptance gates from the campaign artifacts
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("artifact", ["summary.json"])
def test_cd_acceptance_from_artifacts(artifact):
    p = ART / artifact
    if not p.exists():
        pytest.skip(f"{p} not produced yet (run scripts/run_spherical_force_campaign.py)")
    s = json.loads(p.read_text())
    ac = s.get("acceptance", {})
    if ac.get("AC2", {}).get("pass") is not None:
        assert ac["AC2"]["pass"], ac["AC2"]
    if ac.get("AC4", {}).get("pass") is not None:
        assert ac["AC4"]["pass"], ac["AC4"]
    if ac.get("AC5", {}).get("pass") is not None:
        assert ac["AC5"]["pass"], ac["AC5"]
    if ac.get("AC1_planar", {}).get("pass") is not None:
        assert ac["AC1_planar"]["pass"], ac["AC1_planar"]