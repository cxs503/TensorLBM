"""Round 2 of ``GAP-LBM-T5-ICE-COUPLING`` — G5-07: a **real** 3-D LBM field
driving the load-exchange interface.

Round 1 (:mod:`tensorlbm.three_way.interface3d`) closed the *algebraic* side of
the 3-D hand-off: given a traction field it proved exact resultant / moment /
work conservation.  The open gap it left is G5-07 — *"interface not yet driven
by a real 3-D LBM field (only analytic tractions)"*.  An analytic traction is
affine, and affine fields are precisely the ones a Q1 consistent transfer
reproduces exactly, so the round-1 gate cannot distinguish a good transfer from
a lucky one on a real (non-affine, noisy, surface-restricted) load.

This module supplies the missing edge and nothing else:

1. :func:`per_link_force` — the per-link momentum ledger of the Bouzidi–Firdaouss–
   Lallemand interpolated bounce-back wall, i.e. the *same* arithmetic that
   :func:`tensorlbm.bfl_common.bfl_force_ledger_sparse` sums into a single total
   force, kept un-summed so that every link carries its own vector.  The sum of
   the distribution is gated against the library total (:func:`ledger_closure`),
   so the interface is demonstrably fed by the calibrated force caliber of
   ``benchmarks/verified/sphere_re100`` and not by a look-alike.
2. :func:`link_wall_points` — where on the body each link's force acts
   (``x_cell + q c_d``), in the same lattice frame as the geometry.
3. :func:`bin_to_surface` — the discrete traction: link forces are assigned to
   the nearest facet of a :class:`~tensorlbm.three_way.interface3d.Surface3D`
   sphere quadrature and returned as a facet traction ``t = F_facet / dA_facet``.
   Assignment conserves the resultant *exactly*; what it does **not** conserve
   exactly is the moment (a link force is moved from its wall point to a facet
   centroid), and that displacement error is measured by
   :func:`resultant_and_moment` instead of being assumed away.
4. :func:`run_sphere_field` — a D3Q19 + MRT + BFL sphere at Re=100 on the
   calibrated ``sphere_re100`` protocol (domain padding 1.25D/2.25D, tau=0.56,
   u=0.05), returning the drag history *and* the final-step traction field.

Honest scope (stated here, repeated in the report)
--------------------------------------------------
The sphere is **fixed**: the fluid leg is one-way (the structural response does
not move the wall).  The load path is real, the feedback is not; a two-way
moving-sphere run needs the per-step sub-cell geometry refresh (``q`` depends on
the body position) and is left to round 3.  Nothing in this module claims an
engineering-grade ice load — the ice leg (DEM) is untouched (G5-06).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch

from tensorlbm.bfl_common import (
    _OPP19_LIST,
    _OPP27_LIST,
    bfl_bounce_back_sparse,
    bfl_boundary_link_indices,
    bfl_force_ledger_sparse,
)
from tensorlbm.boundaries3d import far_field_bc_3d
from tensorlbm.d3q19 import C as _C19
from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.interpolated_bc import compute_q_sphere
from tensorlbm.solver3d import collide_mrt3d_low_memory, correct_mass3d, stream3d_roll

from .interface3d import DTYPE, HexGrid3D, Surface3D, sphere_surface

# ---------------------------------------------------------------------------
# 1. Per-link force distribution of the BFL wall
# ---------------------------------------------------------------------------


def _c_table(Q: int, device, dtype) -> torch.Tensor:
    c = _C19 if Q == 19 else torch.tensor(
        [[int(v) for v in row] for row in _C27_ROWS], dtype=torch.int64
    )
    return c.to(device=device, dtype=dtype)


_C27_ROWS = [  # pragma: no cover - only reachable for D3Q27 links
    [0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
    [1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0], [1, 0, 1], [-1, 0, -1], [1, 0, -1],
    [-1, 0, 1], [0, 1, 1], [0, -1, -1], [0, 1, -1], [0, -1, 1], [1, 1, 1], [-1, -1, -1],
    [1, 1, -1], [-1, -1, 1], [1, -1, 1], [-1, 1, -1], [-1, 1, 1], [1, -1, -1],
]


def per_link_force(
    f: torch.Tensor,
    f_prev: torch.Tensor,
    links,
    *,
    fraction: float = 1.0,
    force_frame: str = "laboratory",
) -> torch.Tensor:
    """Per-link momentum-exchange force, ``(n_links, 3)``, laboratory or wall frame.

    Reproduces the arithmetic of :func:`tensorlbm.bfl_common.bfl_force_ledger_sparse`
    per link instead of accumulating over links, so that

    ``per_link_force(f, f_prev, links).sum(0)``

    is the library's total force (gated to floating-point round-off by
    :func:`ledger_closure`).  ``force_frame="wall"`` removes the transparent-wall
    background flux ``c_d (f_d^* + f_od)`` exactly as the library does.
    """
    if force_frame not in {"laboratory", "wall"}:
        raise ValueError(f"force_frame must be 'laboratory' or 'wall', got {force_frame!r}")
    Q = links.Q
    nz, ny, nx = links.grid_shape
    if tuple(f.shape) != (Q, nz, ny, nx) or tuple(f_prev.shape) != (Q, nz, ny, nx):
        raise ValueError(
            f"f/f_prev must be ({Q}, {nz}, {ny}, {nx}); got {tuple(f.shape)} / {tuple(f_prev.shape)}"
        )
    if not (f.is_contiguous() and f_prev.is_contiguous()):
        raise ValueError("f and f_prev must be contiguous")

    opp = _OPP19_LIST if Q == 19 else _OPP27_LIST
    c_lat = _c_table(Q, f.device, f.dtype)
    f_flat = f.reshape(Q, -1)
    fp_flat = f_prev.reshape(Q, -1)
    out = torch.zeros((links.link_idx.numel(), 3), dtype=f.dtype, device=f.device)

    for d in range(1, Q):
        sl = links.direction_slice(d)
        if sl.start == sl.stop:
            continue
        od = opp[d]
        idx = links.link_idx[sl]
        up = links.link_up_idx[sl]
        q_cell = links.link_q[sl].to(dtype=f.dtype)
        lin = q_cell < 0.5
        fp_d = fp_flat[d, idx]
        fp_od = fp_flat[od, idx]
        fp_up = fp_flat[d, up]
        q_safe = torch.where(lin, torch.ones_like(q_cell), q_cell)
        inv_2q = 1.0 / (2.0 * q_safe)
        f_bc_stat = torch.where(
            lin,
            2.0 * q_cell * fp_d + (1.0 - 2.0 * q_cell) * fp_up,
            fp_d * inv_2q + (2.0 * q_safe - 1.0) * inv_2q * fp_od,
        )
        f_streamed_unk = f_flat[od, idx]
        stat_link = fp_d + (1.0 - float(fraction)) * f_streamed_unk + float(fraction) * f_bc_stat
        if force_frame == "wall":
            stat_link = stat_link - (fp_d + f_streamed_unk)
        out[sl] = c_lat[d].view(1, 3) * stat_link.view(-1, 1)

    return out


def ledger_closure(
    f: torch.Tensor,
    f_prev: torch.Tensor,
    links,
    *,
    fraction: float = 1.0,
    force_frame: str = "laboratory",
) -> dict:
    """Cross-check the distribution against the library's own total ledger.

    Returns the two totals and their relative difference; this is the gate that
    says the traction the interface consumes *is* the calibrated caliber of
    ``benchmarks/verified/sphere_re100`` rather than an independent look-alike.
    """
    dist = per_link_force(f, f_prev, links, fraction=fraction, force_frame=force_frame)
    ref, n_active = bfl_force_ledger_sparse(
        f, f_prev, links, fraction=fraction, force_frame=force_frame
    )
    ref_vec = torch.stack([ref[0], ref[1], ref[2]]).to(dtype=torch.float64)
    dist_vec = dist.to(dtype=torch.float64).sum(0)
    denom = max(float(torch.linalg.norm(ref_vec)), 1e-30)
    rel = float(torch.linalg.norm(dist_vec - ref_vec)) / denom
    return {
        "ledger_total": [float(x) for x in ref_vec],
        "distribution_total": [float(x) for x in dist_vec],
        "rel_diff": rel,
        "n_active_links": int(n_active),
        "n_links": int(links.link_idx.numel()),
    }


# ---------------------------------------------------------------------------
# 2. Where the links act, and the discrete traction
# ---------------------------------------------------------------------------


def link_wall_points(links, *, cell_centred: bool = True) -> torch.Tensor:
    """Wall point of every link in lattice units: ``x_cell + q c_d``.

    ``cell_centred=True`` puts the lattice cell centre at ``i + 0.5`` (the
    convention of :func:`tensorlbm.interpolated_bc.compute_q_sphere`).
    """
    nz, ny, nx = links.grid_shape
    idx = links.link_idx
    iz = torch.div(idx, ny * nx, rounding_mode="floor")
    rem = idx - iz * ny * nx
    iy = torch.div(rem, nx, rounding_mode="floor")
    ix = rem - iy * nx
    base = torch.stack([ix, iy, iz], dim=1).to(dtype=DTYPE)
    if cell_centred:
        base = base + 0.5
    c_lat = _c_table(links.Q, idx.device, DTYPE)
    return base + links.link_q.to(dtype=DTYPE).view(-1, 1) * c_lat[links.link_dir]


def bin_directions(positions: torch.Tensor, centre: Sequence[float]) -> torch.Tensor:
    """Unit direction of each point as seen from ``centre`` (robust to an offset)."""
    c = torch.as_tensor(list(centre), dtype=positions.dtype)
    v = positions - c.view(1, 3)
    return v / torch.linalg.norm(v, dim=1, keepdim=True).clamp_min(1e-30)


@dataclass
class SurfaceLoad:
    """A traction field on a :class:`Surface3D` plus the provenance it came from."""

    surface: Surface3D
    traction: torch.Tensor            # (n_facets, 3)
    forces: torch.Tensor              # (n_facets, 3) = traction * dA
    links_per_facet: torch.Tensor     # (n_facets,) int
    n_links: int
    n_covered_facets: int
    moment_binning_error: torch.Tensor  # (3,) facet moment minus link moment
    moment_link: torch.Tensor           # (3,)
    moment_facet: torch.Tensor          # (3,)
    sum_abs_link_force: float
    facet_size: float
    radius_mean: float

    @property
    def coverage(self) -> float:
        return self.n_covered_facets / float(len(self.surface))

    def resultant(self) -> torch.Tensor:
        return self.forces.sum(0)

    def moment(self, origin: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.surface.moment(self.traction, origin)


def bin_to_surface(
    surface: Surface3D,
    positions: torch.Tensor,
    forces: torch.Tensor,
    centre: Sequence[float],
    *,
    chunk: int = 4096,
) -> SurfaceLoad:
    """Assign each link force to the nearest facet (by direction from ``centre``).

    The assignment is a pure partition: every link force is counted once, so the
    resultant is conserved exactly.  The moment is *not* — each force is moved
    from its wall point to a facet centroid — and the resulting displacement
    moment is returned in ``moment_binning_error`` and gated in the driver.
    """
    c = torch.as_tensor(list(centre), dtype=DTYPE)
    d_links = bin_directions(positions, centre)                       # (n,3)
    d_facets = bin_directions(surface.positions.to(dtype=DTYPE), centre)  # (m,3)

    assign = torch.empty(d_links.shape[0], dtype=torch.long)
    for s in range(0, d_links.shape[0], chunk):
        block = d_links[s:s + chunk]
        sim = block @ d_facets.t()
        assign[s:s + chunk] = torch.argmax(sim, dim=1)

    n_facets = len(surface)
    facet_forces = torch.zeros((n_facets, 3), dtype=DTYPE)
    facet_forces.index_add_(0, assign, forces.to(dtype=DTYPE))
    counts = torch.zeros(n_facets, dtype=torch.long)
    counts.index_add_(0, assign, torch.ones_like(assign))

    dA = surface.dA.to(dtype=DTYPE).clamp_min(1e-30)
    traction = facet_forces / dA.view(-1, 1)

    pos_d = positions.to(dtype=DTYPE)
    frc_d = forces.to(dtype=DTYPE)
    moment_link = torch.linalg.cross(pos_d - c.view(1, 3), frc_d).sum(0)
    moment_facet = surface.moment(traction, origin=c)
    return SurfaceLoad(
        surface=surface,
        traction=traction,
        forces=facet_forces,
        links_per_facet=counts,
        n_links=int(positions.shape[0]),
        n_covered_facets=int((counts > 0).sum()),
        moment_binning_error=moment_facet - moment_link,
        moment_link=moment_link,
        moment_facet=moment_facet,
        sum_abs_link_force=float(torch.linalg.norm(frc_d, dim=1).sum()),
        facet_size=float(torch.sqrt(surface.dA.to(dtype=DTYPE).mean())),
        radius_mean=float(torch.linalg.norm(pos_d - c.view(1, 3), dim=1).mean()),
    )


# ---------------------------------------------------------------------------
# 3. Real field run (calibrated sphere_re100 protocol)
# ---------------------------------------------------------------------------


@dataclass
class SphereFieldRun:
    """Result of :func:`run_sphere_field` — drag history plus the load field."""

    D: int
    steps: int
    tau: float
    u_lb: float
    nu_lb: float
    re: float
    domain_lu: tuple
    centre: tuple
    radius: float
    dpS: float
    solid_cells: int
    n_links: int
    q_range: tuple
    history: list = field(default_factory=list)
    load: Optional[SurfaceLoad] = None
    loads: dict = field(default_factory=dict)   # label -> SurfaceLoad (surface refinement study)
    closure: dict = field(default_factory=dict)
    finite: bool = True
    wall_s: float = 0.0

    @property
    def cd_final(self) -> float:
        return self.history[-1]["cd"] if self.history else float("nan")

    @property
    def cd_mean_tail(self) -> float:
        tail = self.history[max(0, len(self.history) - 5):]
        return sum(e["cd"] for e in tail) / max(len(tail), 1)

    def transient(self) -> tuple:
        """``(t_star, cd)`` of the sampled drag history (``t* = u t / D``)."""
        t = [e["step"] * self.u_lb / self.D for e in self.history]
        return t, [e["cd"] for e in self.history]


def _sphere_solid(nx, ny, nz, centre, radius) -> torch.Tensor:
    ix = torch.arange(nx, dtype=DTYPE) + 0.5
    iy = torch.arange(ny, dtype=DTYPE) + 0.5
    iz = torch.arange(nz, dtype=DTYPE) + 0.5
    Z, Y, X = torch.meshgrid(iz, iy, ix, indexing="ij")
    r2 = (X - centre[0]) ** 2 + (Y - centre[1]) ** 2 + (Z - centre[2]) ** 2
    return r2 <= radius**2


def run_sphere_field(
    D: int = 16,
    steps: int = 1500,
    *,
    tau: float = 0.56,
    u_lb: float = 0.05,
    lateral_pad: float = 1.25,
    upstream_pad: float = 1.25,
    downstream_pad: float = 2.25,
    sample: int = 25,
    n_theta: int = 16,
    n_phi: int = 32,
    surface_levels: Optional[Sequence[Sequence[int]]] = None,
    device: str = "cpu",
    correct_every: int = 200,
    dtype=torch.float32,
) -> SphereFieldRun:
    """Fixed-sphere D3Q19 + MRT + BFL run on the calibrated ``sphere_re100`` grid.

    Padding is expressed in diameters, exactly as in
    ``benchmarks/verified/sphere_re100/run.py`` (``lat=1.25, up=1.25, down=2.25``
    for D=40/i.e. 50/50/90 lattice units), so a run here is the same case at a
    coarser resolution and its ``Cd`` is directly comparable with the stored
    D=40/D=60 artefacts.
    """
    dev = torch.device(device)
    R = D / 2.0
    nx = int(round((lateral_pad + downstream_pad + 1.0) * D))
    ny = int(round((2.0 * lateral_pad + 1.0) * D))
    nz = int(round((2.0 * lateral_pad + 1.0) * D))
    centre = (upstream_pad * D + R, ny / 2.0, nz / 2.0)

    solid = _sphere_solid(nx, ny, nz, centre, R).to(dev)
    bfl_mask, bfl_q = compute_q_sphere(nx, ny, nz, centre[0], centre[1], centre[2], R, dev)
    links = bfl_boundary_link_indices(bfl_mask, bfl_q, lattice="D3Q19")
    del bfl_mask, bfl_q

    rho = torch.ones((nz, ny, nx), dtype=dtype, device=dev)
    ux = torch.full_like(rho, u_lb)
    uy = torch.zeros_like(rho)
    uz = torch.zeros_like(rho)
    f = equilibrium3d(rho, ux, uy, uz, dev).to(dtype=dtype)
    target_mass = float(f.sum().item())
    sm = solid.unsqueeze(0).expand_as(f)

    bc_config = {"far_field_faces": ["y-", "y+", "z-", "z+"], "periodic_faces": []}
    nu_lb = (tau - 0.5) / 3.0
    dpS = 0.5 * u_lb**2 * math.pi * R**2
    levels = [tuple(int(v) for v in lv) for lv in (surface_levels or [(n_theta, n_phi)])]

    import time as _time

    t0 = _time.time()
    hist = []
    load = None
    loads = {}
    closure = {}
    finite = True
    f_pre_stream = f.clone()
    for step in range(1, steps + 1):
        f_pre_collide = f.clone()
        f = collide_mrt3d_low_memory(f, tau=tau)
        f = torch.where(sm, f_pre_collide, f)
        del f_pre_collide
        f_pre_stream = f.clone()
        f = stream3d_roll(f)
        f = far_field_bc_3d(f, u_lb, bc_config=bc_config)
        if step == steps:
            # Capture the traction *before* the wall is applied: the ledger wants
            # (post-stream, pre-stream) populations, which is exactly the pair the
            # library caliber consumes.
            dist = per_link_force(f, f_pre_stream, links)
            closure = ledger_closure(f, f_pre_stream, links)
            positions = link_wall_points(links)
            for nt, np_ in levels:
                surf = sphere_surface(centre=centre, radius=R, n_theta=nt, n_phi=np_)
                loads[f"theta{nt}_phi{np_}"] = bin_to_surface(surf, positions, dist, centre)
            load = loads[f"theta{levels[0][0]}_phi{levels[0][1]}"]
            f = bfl_bounce_back_sparse(f, f_pre_stream, links)
        else:
            f = bfl_bounce_back_sparse(f, f_pre_stream, links)
            if step % sample == 0:
                force3, _ = bfl_force_ledger_sparse(f, f_pre_stream, links)
                hist.append(
                    {
                        "step": step,
                        "cd": float(force3[0].item()) / dpS,
                        "cl": float(force3[1].item()) / dpS,
                        "cs": float(force3[2].item()) / dpS,
                    }
                )
        del f_pre_stream
        if correct_every and step % correct_every == 0:
            f = correct_mass3d(f, target_mass)
        if step % 500 == 0 and not bool(torch.isfinite(f).all()):
            finite = False
            break

    q = links.link_q
    return SphereFieldRun(
        D=D,
        steps=steps,
        tau=tau,
        u_lb=u_lb,
        nu_lb=nu_lb,
        re=u_lb * D / nu_lb,
        domain_lu=(nz, ny, nx),
        centre=centre,
        radius=R,
        dpS=dpS,
        solid_cells=int(solid.sum().item()),
        n_links=int(links.n_links),
        q_range=(float(q.min()), float(q.max())),
        history=hist,
        load=load,
        loads=loads,
        closure=closure,
        finite=finite,
        wall_s=_time.time() - t0,
    )


# ---------------------------------------------------------------------------
# 4. Real-traction invariants through the structural transfer
# ---------------------------------------------------------------------------


def transfer_gates(
    load: SurfaceLoad,
    grid: HexGrid3D,
    *,
    origin: Optional[Sequence[float]] = None,
    moment_origin: Optional[Sequence[float]] = None,
    velocity=None,
) -> dict:
    """Resultant / moment / work checks of the Q1 transfer on the *real* traction.

    Parameters
    ----------
    origin:
        Load resultant reference (defaults to the block centre).
    moment_origin:
        Reference point of the moment integral.  It must **not** be the sphere
        centre: for a drag-dominated load the moment about the centre nearly
        cancels (``|M| -> 0``), so a relative error taken there divides by noise.
        The driver uses an external reference (a "mooring point" offset by
        several radii), which is also the physically relevant lever arm for a
        hull-girder hand-off.  Errors are reported *relative to the moment scale*
        ``|F| * |x_ref - x_centroid|`` as well as absolutely.
    velocity:
        callable ``r (n,3) -> u (n,3)`` giving the structural velocity field whose
        trace the work check uses.  It is pulled through the hex grid with
        :meth:`HexGrid3D.gather_surface_velocity` (the transpose operator readings,
        i.e. the discrete transpose of the load operator), so a non-affine nodal
        field is exercised as well — the round-1 gate only used affine fields,
        where exactness is trivial.
    """
    if origin is None:
        origin = [float(grid.origin[d]) + 0.5 * float(grid.extent[d]) for d in range(3)]
    if moment_origin is None:
        moment_origin = origin
    c = torch.as_tensor(list(origin), dtype=DTYPE)
    m_ref = torch.as_tensor(list(moment_origin), dtype=DTYPE)

    nodal = grid.distribute_surface(load.surface, load.traction)
    f_surface = load.forces
    f_total = f_surface.sum(0)
    centroid = load.surface.centroid().to(dtype=DTYPE)

    res_err = torch.linalg.norm(nodal.sum(0) - f_total) / max(float(torch.linalg.norm(f_total)), 1e-30)

    moment_surface = load.surface.moment(load.traction, origin=m_ref)
    moment_nodal = torch.linalg.cross(grid.nodes - m_ref.view(1, 3), nodal).sum(0)
    moment_scale = float(torch.linalg.norm(f_total)) * max(
        float(torch.linalg.norm(centroid - m_ref)), 1e-30
    )
    mom_err = float(torch.linalg.norm(moment_nodal - moment_surface)) / max(moment_scale, 1e-30)

    out = {
        "moment_scale": moment_scale,
        "resultant_surface": [float(x) for x in f_total],
        "resultant_nodal": [float(x) for x in nodal.sum(0)],
        "resultant_rel_err": float(res_err),
        "moment_surface": [float(x) for x in moment_surface],
        "moment_nodal": [float(x) for x in moment_nodal],
        "moment_rel_err": mom_err,
        "moment_abs_err": float(torch.linalg.norm(moment_nodal - moment_surface)),
        "moment_binning_error": [float(x) for x in load.moment_binning_error],
        # Two honest scales for the binning (surface-quadrature) moment error.
        # (a) the *resultant* lever scale |F| R -- the engineering-relevant one;
        # (b) the pointwise scale sum|F_link| h_facet -- the error scale of moving
        # each link force to a facet centroid.  (|M_link| itself is useless as a
        # denominator: for a drag load the moment about the body centre cancels.)
        "moment_binning_rel_resultant": float(torch.linalg.norm(load.moment_binning_error))
        / max(float(torch.linalg.norm(f_total)) * load.radius_mean, 1e-30),
        "moment_binning_rel_pointwise": float(torch.linalg.norm(load.moment_binning_error))
        / max(load.sum_abs_link_force * load.facet_size, 1e-30),
        "sum_abs_link_force": load.sum_abs_link_force,
        "facet_size": load.facet_size,
        "radius_mean": load.radius_mean,
    }

    if velocity is not None:
        u_nodes = torch.as_tensor(velocity(grid.nodes), dtype=DTYPE)
        u_surface = grid.gather_surface_velocity(load.surface, u_nodes)
        w_surface = float((load.traction * load.surface.dA.view(-1, 1) * u_surface).sum())
        w_nodal = float((nodal * u_nodes).sum())
        out["work_surface"] = w_surface
        out["work_nodal"] = w_nodal
        out["work_rel_err"] = abs(w_nodal - w_surface) / max(abs(w_surface), 1e-30)
    return out


# ---------------------------------------------------------------------------
# 5. Driven structural response and its work ledger
# ---------------------------------------------------------------------------


def march_driven(
    mu: float,
    omega_n_dt: float,
    force_series: Sequence[float],
    *,
    x0: float = 0.0,
    v0: float = 0.0,
) -> dict:
    """Velocity-Verlet 1-DOF march driven by a *sampled* fluid force history.

    Dimensionless units (``k = 1``, ``m_s = 1``, length scale ``L = 1``), state
    ``(x, v)`` at integer steps with the added-mass reaction lagged by one step
    (explicit Dirichlet–Neumann staggering, the same structure as
    :func:`tensorlbm.three_way.interface3d.march_one_dof`)::

        g_n      = f_n - mu * a_{n-1}                       # external + lagged added mass
        v_{n+1/2}= v_n + (dt/2) a_n,  a_n = g_n - x_n
        x_{n+1}  = x_n + dt v_{n+1/2}
        a_{n+1}  = g_{n+1} - x_{n+1}
        v_{n+1}  = v_{n+1/2} + (dt/2) a_{n+1}

    The work ledger is closed as ``dE = W_ext + W_am`` with ``E = (x^2+v^2)/2``,
    ``W_ext`` and ``W_am`` the trapezoid work of the external force and of the
    (negative) added-mass reaction.  The returned ``closure`` residual is the
    integrator error; it is *measured* and its order is gated (G7), so a broken
    bookkeeping cannot hide behind a loose tolerance.
    """
    f = [float(v) for v in force_series]
    n = len(f)
    if n < 3:
        raise ValueError("force_series needs at least 3 samples")
    dt = float(omega_n_dt)

    x = float(x0)
    v = float(v0)
    a_prev = 0.0                    # a_{-1} = 0: fluid at rest
    a = f[0] - mu * a_prev - x
    xs, vs, energy = [x], [v], [0.5 * (x * x + v * v)]
    w_ext = 0.0
    w_am = 0.0
    e0 = energy[0]
    max_resid = 0.0
    w_scale = 0.0
    mu_a_prev = mu * a_prev
    for s in range(n - 1):
        g = f[s] - mu_a_prev
        v_half = vs[-1] + 0.5 * dt * (g - xs[-1])
        x_new = xs[-1] + dt * v_half
        mu_a_new = mu * (g - xs[-1])
        g_next = f[s + 1] - mu_a_new
        a_next = g_next - x_new
        v_new = v_half + 0.5 * dt * a_next
        dx = x_new - xs[-1]
        w_ext += 0.5 * (f[s] + f[s + 1]) * dx
        w_am += 0.5 * (-mu_a_prev - mu_a_new) * dx
        xs.append(x_new)
        vs.append(v_new)
        e_n = 0.5 * (x_new * x_new + v_new * v_new)
        energy.append(e_n)
        # running ledger residual, worst case over the whole march (not just the
        # final sample, whose energy is polluted by the undamped oscillation)
        w_scale = max(w_scale, abs(w_ext) + abs(w_am))
        max_resid = max(max_resid, abs((e_n - e0) - (w_ext + w_am)))
        mu_a_prev = mu_a_new
    delta_e = energy[-1] - e0
    denom = max(w_scale, 1e-30)
    return {
        "max_residual": max_resid,
        "closure_max": max_resid / denom,
        "x": xs,
        "v": vs,
        "energy": energy,
        "w_ext": w_ext,
        "w_added_mass": w_am,
        "delta_energy": delta_e,
        "closure": abs(delta_e - (w_ext + w_am)) / denom,
        "w_scale": w_scale,
        "x_peak": max(abs(v) for v in xs),
        "x_final": xs[-1],
    }


def growth_rate(history: torch.Tensor) -> float:
    """Measured per-step growth rate ``ln|x_n/x_{n-1}|`` averaged over the tail."""
    x = history[0].abs().to(dtype=torch.float64)
    tail = x[max(len(x) // 2, 1):]
    tail = tail[tail > 1e-300]
    if len(tail) < 3:
        return float("nan")
    ratios = (tail[1:] / tail[:-1]).log()
    return float(ratios.mean())
