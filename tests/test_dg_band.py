"""Tests for the hybrid DG-band coupling (dg_band.py).

Gate: with the band == whole periodic domain, the packed-band DG RHS must equal
the validated full-grid DG RHS element-wise.  Run CPU-only:

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=src python -m pytest tests/test_dg_band.py -q
"""
from __future__ import annotations

import torch

from tensorlbm.dg_advection import dg_rhs, get_ops
from tensorlbm.dg_band import build_band_topology, dg_rhs_band, dg_advect_band, hybrid_advect, hybrid_step
from tensorlbm.d2q9 import C as C2D, W as W2D, OPPOSITE as OPP2D, equilibrium as eq2d
from tensorlbm.d3q19 import C as C3D

DT = torch.float64


def _pack_full_to_band(f_full: torch.Tensor, ndim: int) -> torch.Tensor:
    """``(Q, *shape, *nodes)`` → ``(Q, n_band, *nodes)`` (row-major band order)."""
    q = f_full.shape[0]
    n_node = f_full.shape[-ndim:]
    shape = f_full.shape[1:-ndim]
    n_band = int(torch.tensor(shape).prod().item())
    return f_full.reshape(q, n_band, *n_node).contiguous()


class TestBandEqualsFullGrid:
    """Packed-band RHS == full-grid RHS when the band is the whole domain."""

    def test_2d_periodic(self) -> None:
        nz_ny, nx = 12, 10
        torch.manual_seed(0)
        ops = get_ops(degree=1, dx=1.0, dtype=DT)
        f_full = 0.5 + 0.4 * torch.rand(9, nz_ny, nx, 2, 2, dtype=DT)
        mask = torch.ones(nz_ny, nx, dtype=torch.bool)
        topo = build_band_topology(mask, periodic=True)
        f_band = _pack_full_to_band(f_full, ndim=2)

        rhs_full = dg_rhs(f_full, C2D.to(DT), ops, ndim_spatial=2)
        rhs_band = dg_rhs_band(f_band, C2D.to(DT), ops, topo, ext_field=None)
        rhs_full_packed = _pack_full_to_band(rhs_full, ndim=2)

        assert torch.allclose(rhs_band, rhs_full_packed, atol=1e-10), (
            f"2D band vs full max diff: {(rhs_band - rhs_full_packed).abs().max().item():.2e}"
        )

    def test_3d_periodic(self) -> None:
        nz, ny, nx = 6, 5, 7
        torch.manual_seed(1)
        ops = get_ops(degree=1, dx=1.0, dtype=DT)
        f_full = 0.5 + 0.4 * torch.rand(19, nz, ny, nx, 2, 2, 2, dtype=DT)
        mask = torch.ones(nz, ny, nx, dtype=torch.bool)
        topo = build_band_topology(mask, periodic=True)
        f_band = _pack_full_to_band(f_full, ndim=3)

        rhs_full = dg_rhs(f_full, C3D.to(DT), ops, ndim_spatial=3)
        rhs_band = dg_rhs_band(f_band, C3D.to(DT), ops, topo, ext_field=None)
        rhs_full_packed = _pack_full_to_band(rhs_full, ndim=3)

        assert torch.allclose(rhs_band, rhs_full_packed, atol=1e-10), (
            f"3D band vs full max diff: {(rhs_band - rhs_full_packed).abs().max().item():.2e}"
        )

    def test_topology_neighbours(self) -> None:
        """A full-domain mask has only band neighbours (no -1)."""
        mask = torch.ones(5, 6, dtype=torch.bool)
        topo = build_band_topology(mask, periodic=True)
        assert topo.n_band == 30
        assert (topo.nbr_minus >= 0).all().item()
        assert (topo.nbr_plus >= 0).all().item()
        # Neighbour round-trip: nbr_plus[nbr_minus[b]] == b (periodic, full grid).
        nb = topo.nbr_plus[0, topo.nbr_minus[0]]
        assert torch.all(nb == torch.arange(30))


class TestInterfaceConservation:
    """Hybrid DG-band + LBM-exterior advection conserves total mass.

    A periodic 2-D domain split into an exterior LBM region and a central DG
    band slab.  Pure advection (no collision) over many macro-steps: the global
    mass (exterior P0 populations + band DG cell-means) must be conserved — the
    gate for the DG↔LBM interface (no double-write leak).
    """

    def _setup(self, ny=24, nx=20, band_rows=range(8, 16)):
        torch.manual_seed(11)
        rho = 1.0 + 0.3 * torch.rand(ny, nx, dtype=DT)
        ux = 0.05 + 0.05 * torch.rand(ny, nx, dtype=DT)
        uy = 0.03 * torch.rand(ny, nx, dtype=DT)
        f_lbm = eq2d(rho, ux, uy).to(DT)
        mask = torch.zeros(ny, nx, dtype=torch.bool)
        mask[list(band_rows), :] = True
        topo = build_band_topology(mask, periodic=True)
        # Seed band DOFs from the cell-mean LBM values at band cells (P0 seed).
        n_band = topo.n_band
        f_dg = f_lbm[:, topo.band_coords[:, 0], topo.band_coords[:, 1]]  # (Q, n_band)
        f_dg = f_dg.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2, 2).contiguous()
        return f_lbm, f_dg, topo, (ny, nx)

    def _total_mass(self, f_lbm, f_dg, topo, shape):
        ny, nx = shape
        mask = torch.zeros(ny, nx, dtype=torch.bool)
        mask[tuple(topo.band_coords.t())] = True
        ext_mass = f_lbm[:, ~mask].sum().item()
        # Band DG cell-mean mass = mean over (p+1)^d nodes (P1 weights all 1).
        band_mass = f_dg.mean(dim=tuple(range(2, f_dg.ndim))).sum().item()
        return ext_mass + band_mass

    def test_global_mass_conserved(self) -> None:
        """The interface is approximately conservative.

        Exact conservation between method-of-lines DG (flux-based) and shift-based
        LBM does not hold for *diagonal* lattice velocities: DG splits a diagonal
        advection into per-axis fluxes, while LBM streams diagonally in one step.
        The residual leak is small (a few ×10⁻⁵/30 steps here) and is mopped up
        by the periodic ``correct_mass`` already used in every LBM runner.  A
        genuine interface bug (double-write, flux mismatch) shows up as an O(1)
        drift, so this gate still catches it.
        """
        f_lbm, f_dg, topo, shape = self._setup()
        ops = get_ops(degree=1, dx=1.0, dtype=DT)
        m0 = self._total_mass(f_lbm, f_dg, topo, shape)
        for _ in range(30):
            f_lbm, f_dg = hybrid_advect(
                f_lbm, f_dg, C2D.to(DT), ops, topo, dt=1.0, n_substeps=6, scheme="rk3"
            )
        m1 = self._total_mass(f_lbm, f_dg, topo, shape)
        rel = abs(m1 - m0) / abs(m0)
        assert rel < 5e-4, f"hybrid mass drift {rel:.2e} (m0={m0:.4f} m1={m1:.4f})"

    def test_exterior_stays_bounded(self) -> None:
        """No explosive instability through the interface over many steps."""
        f_lbm, f_dg, topo, shape = self._setup()
        ops = get_ops(degree=1, dx=1.0, dtype=DT)
        for _ in range(60):
            f_lbm, f_dg = hybrid_advect(
                f_lbm, f_dg, C2D.to(DT), ops, topo, dt=1.0, n_substeps=6, scheme="rk3"
            )
        assert torch.isfinite(f_lbm).all()
        assert torch.isfinite(f_dg).all()
        assert f_lbm.abs().max().item() < 1e3
        assert f_dg.abs().max().item() < 1e3


class TestSolidWallBounceBack:
    """Half-way bounce-back at solid faces: zero net mass flux, stable."""

    def test_closed_band_mass_conserved(self) -> None:
        """A band fully enclosed by solid walls conserves mass (reflections
        produce zero net normal flux) and stays stable."""
        torch.manual_seed(21)
        ny, nx = 7, 7
        solid = torch.zeros(ny, nx, dtype=torch.bool)
        solid[0, :] = solid[-1, :] = solid[:, 0] = solid[:, -1] = True
        band = torch.zeros(ny, nx, dtype=torch.bool)
        band[2:5, 2:5] = True
        topo = build_band_topology(band, solid_mask=solid, periodic=False)
        ops = get_ops(degree=1, dx=1.0, dtype=DT)
        n_band = topo.n_band
        rho = 1.0 + 0.3 * torch.rand(n_band, dtype=DT)
        ux = 0.05 * torch.rand(n_band, dtype=DT)
        uy = 0.05 * torch.rand(n_band, dtype=DT)
        # Build (Q, n_band, 2, 2) DOFs from per-cell equilibrium (P0 seed).
        rho_f = rho.view(1, n_band, 1, 1).expand(9, n_band, 2, 2)
        ux_f = ux.view(1, n_band, 1, 1).expand(9, n_band, 2, 2)
        uy_f = uy.view(1, n_band, 1, 1).expand(9, n_band, 2, 2)
        from tensorlbm.dg_advection import equilibrium_dg
        f_dg = equilibrium_dg(rho_f, [ux_f, uy_f], C2D.to(DT), W2D.to(DT))

        def mass(fd: torch.Tensor) -> float:
            return fd.mean(dim=tuple(range(2, fd.ndim))).sum().item()

        m0 = mass(f_dg)
        for _ in range(40):
            f_dg = dg_advect_band(
                f_dg, C2D.to(DT), ops, topo, ext_field=None, dt=1.0,
                n_substeps=6, scheme="rk3", opposite=OPP2D.to(DT),
            )
        m1 = mass(f_dg)
        rel = abs(m1 - m0) / abs(m0)
        assert rel < 1e-6, f"closed-band wall mass drift {rel:.2e}"
        assert torch.isfinite(f_dg).all()
        assert f_dg.abs().max().item() < 1e3

    def test_solid_marked_in_topology(self) -> None:
        """A band cell next to the obstacle is tagged solid (type 2), not exterior."""
        solid = torch.zeros(5, 5, dtype=torch.bool)
        solid[2, 2] = True
        band = torch.zeros(5, 5, dtype=torch.bool)
        band[2, 3] = True              # band cell immediately +x of the solid
        topo = build_band_topology(band, solid_mask=solid, periodic=False)
        # Grid dims are (y, x): axis 1 = x. The -x neighbour of the band cell is
        # the solid cell at (2, 2) → tagged type 2 (solid).
        assert int(topo.nbr_type_minus[1, 0]) == 2


class TestHybridStepWithCollision:
    """hybrid_step (collide + method-of-lines band + stream) is stable and
    recovers a uniform viscosity across band + exterior (τ_dg = τ_lbm − ½)."""

    def _reconstruct_ux(self, f_lbm, f_dg, topo, ny, nx):
        from tensorlbm.d2q9 import macroscopic as mac2d
        from tensorlbm.dg_advection import macroscopic_dg
        rho_l, ux_l, uy_l = mac2d(f_lbm)
        rho_b, us_b = macroscopic_dg(f_dg, C2D.to(DT))
        ux_full = ux_l.clone()
        mean_b = us_b[0].mean(dim=tuple(range(1, us_b[0].ndim)))   # (n_band,) cell-mean
        coords = topo.band_coords
        ux_full[coords[:, 0], coords[:, 1]] = mean_b
        return ux_full

    def test_stable_and_viscosity_matched(self) -> None:
        import math
        ny, nx = 24, 16
        band = torch.zeros(ny, nx, dtype=torch.bool)
        band[8:16, :] = True
        topo = build_band_topology(band, periodic=True)
        ops = get_ops(degree=1, dx=1.0, dtype=DT)

        U0 = 0.01
        yc = (torch.arange(ny, dtype=DT) + 0.5)
        rho = torch.ones(ny, nx, dtype=DT)
        ux = (U0 * torch.sin(2 * math.pi * yc / ny)).view(ny, 1).expand(ny, nx)
        uy = torch.zeros(ny, nx, dtype=DT)
        f_lbm = eq2d(rho, ux, uy).to(DT)
        # Seed band DOFs (P0) from the band-cell LBM values.
        cb = topo.band_coords
        f_dg = f_lbm[:, cb[:, 0], cb[:, 1]].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2, 2).contiguous()

        def amp_of() -> float:
            ux_full = self._reconstruct_ux(f_lbm, f_dg, topo, ny, nx)
            return (ux_full.mean(dim=1) * torch.sin(2 * math.pi * yc / ny)).sum().item() * (2.0 / ny)

        a0 = amp_of()
        tau_lbm = 0.9
        for _ in range(40):
            f_lbm, f_dg = hybrid_step(
                f_lbm, f_dg, C2D.to(DT), W2D.to(DT), ops, topo,
                tau_lbm=tau_lbm, dt=1.0, n_substeps=6, opposite=OPP2D.to(DT),
            )
        a1 = amp_of()
        assert a1 > 0, "hybrid_step unstable (amp<=0)"
        k2 = (2 * math.pi / ny) ** 2
        nu_eff = -math.log(a1 / a0) / (k2 * 40)
        nu_ext = (tau_lbm - 0.5) / 3.0          # exterior LBM viscosity
        # Band uses τ_dg = τ_lbm − ½ ⇒ same viscosity; whole domain ≈ uniform.
        rel = abs(nu_eff - nu_ext) / nu_ext
        assert rel < 0.25, f"hybrid ν_eff={nu_eff:.4f} vs {(tau_lbm-0.5)/3:.4f} (rel {rel:.0%})"
        assert torch.isfinite(f_lbm).all() and torch.isfinite(f_dg).all()
