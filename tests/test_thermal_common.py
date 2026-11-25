"""Contract tests for the thermal common module.

Verifies the composable DDF thermal LBM step (D3Q7 + D3Q19/D3Q27),
buoyancy coupling, and conjugate heat transfer.

The operator tests check shape, finiteness, conservation, and equilibrium.
Wall diagnostics are also checked against manufactured fields and transient slab cooling.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.d3q27 import equilibrium27
from tensorlbm.thermal_common import (
    C_D3Q7,
    W_D3Q7,
    apply_buoyancy_3d,
    conjugate_ht_step,
    thermal_collide_bgk_3d,
    thermal_equilibrium_3d,
    thermal_macroscopic_3d,
    thermal_step,
    thermal_stream_3d,
)

TAU_T = 0.8  # > 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _f3d19(nz=4, ny=6, nx=8, u_mag=0.04) -> torch.Tensor:
    rho = torch.rand((nz, ny, nx)) + 0.5
    ux = torch.rand_like(rho) * u_mag
    uy = torch.rand_like(rho) * u_mag
    uz = torch.rand_like(rho) * u_mag
    return equilibrium3d(rho, ux, uy, uz)


def _f3d27(nz=4, ny=6, nx=8, u_mag=0.04) -> torch.Tensor:
    rho = torch.rand((nz, ny, nx)) + 0.5
    ux = torch.rand_like(rho) * u_mag
    uy = torch.rand_like(rho) * u_mag
    uz = torch.rand_like(rho) * u_mag
    return equilibrium27(rho, ux, uy, uz)


def _g_thermal(nz=4, ny=6, nx=8, T_mag=1.0, u_mag=0.04) -> torch.Tensor:
    T = torch.rand((nz, ny, nx)) * T_mag + 0.5
    ux = torch.rand_like(T) * u_mag
    uy = torch.rand_like(T) * u_mag
    uz = torch.rand_like(T) * u_mag
    return thermal_equilibrium_3d(T, ux, uy, uz)


# ---------------------------------------------------------------------------
# D3Q7 lattice constants
# ---------------------------------------------------------------------------


class TestD3Q7Constants:
    def test_weights_sum_to_one(self) -> None:
        assert abs(float(W_D3Q7.sum().item()) - 1.0) < 1e-6

    def test_has_7_directions(self) -> None:
        assert C_D3Q7.shape == (7, 3)
        assert W_D3Q7.shape == (7,)

    def test_rest_direction_is_first(self) -> None:
        assert C_D3Q7[0, 0].item() == 0
        assert C_D3Q7[0, 1].item() == 0
        assert C_D3Q7[0, 2].item() == 0


# ---------------------------------------------------------------------------
# Equilibrium
# ---------------------------------------------------------------------------


class TestThermalEquilibrium:
    def test_shape(self) -> None:
        nz, ny, nx = 4, 6, 8
        T = torch.ones((nz, ny, nx))
        ux = torch.zeros_like(T)
        uy = torch.zeros_like(T)
        uz = torch.zeros_like(T)
        geq = thermal_equilibrium_3d(T, ux, uy, uz)
        assert geq.shape == (7, nz, ny, nx)

    def test_sums_to_temperature(self) -> None:
        nz, ny, nx = 4, 6, 8
        T = torch.rand((nz, ny, nx)) + 0.5
        ux = torch.full_like(T, 0.03)
        uy = torch.full_like(T, -0.02)
        uz = torch.full_like(T, 0.01)
        geq = thermal_equilibrium_3d(T, ux, uy, uz)
        T_out = geq.sum(dim=0)
        assert torch.allclose(T_out, T, atol=1e-5)

    def test_finite(self) -> None:
        nz, ny, nx = 4, 6, 8
        T = torch.rand((nz, ny, nx)) * 2.0 + 0.1
        ux = torch.rand_like(T) * 0.05
        uy = torch.rand_like(T) * 0.04
        uz = torch.rand_like(T) * 0.03
        geq = thermal_equilibrium_3d(T, ux, uy, uz)
        assert torch.isfinite(geq).all()


# ---------------------------------------------------------------------------
# Collision
# ---------------------------------------------------------------------------


class TestThermalCollision:
    def test_preserves_shape(self) -> None:
        g = _g_thermal()
        T = thermal_macroscopic_3d(g)
        ux = torch.zeros_like(T)
        uy = torch.zeros_like(T)
        uz = torch.zeros_like(T)
        g_out = thermal_collide_bgk_3d(g, T, ux, uy, uz, tau_T=TAU_T)
        assert g_out.shape == g.shape

    def test_conserves_temperature(self) -> None:
        g = _g_thermal()
        T = thermal_macroscopic_3d(g)
        ux = torch.full_like(T, 0.03)
        uy = torch.full_like(T, -0.01)
        uz = torch.zeros_like(T)
        # Perturb to create non-equilibrium
        g = g + 0.001 * torch.rand_like(g)
        T_orig = g.sum(dim=0)
        g_out = thermal_collide_bgk_3d(g, T_orig, ux, uy, uz, tau_T=TAU_T)
        T_out = g_out.sum(dim=0)
        assert torch.allclose(T_out, T_orig, atol=1e-5)

    def test_identity_at_equilibrium(self) -> None:
        nz, ny, nx = 4, 6, 8
        T = torch.ones((nz, ny, nx))
        ux = torch.full_like(T, 0.04)
        uy = torch.zeros_like(T)
        uz = torch.full_like(T, -0.01)
        geq = thermal_equilibrium_3d(T, ux, uy, uz)
        g_out = thermal_collide_bgk_3d(geq, T, ux, uy, uz, tau_T=TAU_T)
        assert torch.allclose(g_out, geq, atol=1e-5)

    def test_finite_output(self) -> None:
        g = _g_thermal()
        T = thermal_macroscopic_3d(g)
        ux = torch.rand_like(T) * 0.04
        uy = torch.rand_like(T) * 0.03
        uz = torch.rand_like(T) * 0.02
        g_out = thermal_collide_bgk_3d(g, T, ux, uy, uz, tau_T=0.6)
        assert torch.isfinite(g_out).all()


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class TestThermalStreaming:
    def test_preserves_shape(self) -> None:
        g = _g_thermal()
        g_out = thermal_stream_3d(g)
        assert g_out.shape == g.shape

    def test_conserves_total_temperature(self) -> None:
        g = _g_thermal(nz=8, ny=8, nx=8)
        total_before = g.sum()
        g_out = thermal_stream_3d(g)
        total_after = g_out.sum()
        assert torch.allclose(total_after, total_before, atol=1e-5)

    def test_finite_output(self) -> None:
        g = _g_thermal()
        g_out = thermal_stream_3d(g)
        assert torch.isfinite(g_out).all()


# ---------------------------------------------------------------------------
# Macroscopic recovery
# ---------------------------------------------------------------------------


class TestThermalMacroscopic:
    def test_recovers_temperature(self) -> None:
        nz, ny, nx = 4, 6, 8
        T_in = torch.rand((nz, ny, nx)) * 2.0 + 0.3
        ux = torch.zeros_like(T_in)
        uy = torch.zeros_like(T_in)
        uz = torch.zeros_like(T_in)
        g = thermal_equilibrium_3d(T_in, ux, uy, uz)
        T_out = thermal_macroscopic_3d(g)
        assert torch.allclose(T_out, T_in, atol=1e-5)


# ---------------------------------------------------------------------------
# Buoyancy force
# ---------------------------------------------------------------------------


class TestBuoyancyForce:
    @pytest.mark.parametrize("lattice", ["D3Q19", "D3Q27"])
    def test_preserves_shape(self, lattice: str) -> None:
        f = _f3d19() if lattice == "D3Q19" else _f3d27()
        T = torch.ones(f.shape[1:])
        f_out = apply_buoyancy_3d(f, T, T_ref=1.0, beta=0.001, lattice=lattice)
        assert f_out.shape == f.shape

    @pytest.mark.parametrize("lattice", ["D3Q19", "D3Q27"])
    def test_zero_force_at_reference_temperature(self, lattice: str) -> None:
        f = _f3d19() if lattice == "D3Q19" else _f3d27()
        T_ref = 1.0
        T = torch.full_like(f[0], T_ref)
        f_out = apply_buoyancy_3d(f, T, T_ref=T_ref, beta=0.01, lattice=lattice)
        assert torch.allclose(f_out, f, atol=1e-6)

    @pytest.mark.parametrize("lattice", ["D3Q19", "D3Q27"])
    def test_finite_output(self, lattice: str) -> None:
        f = _f3d19() if lattice == "D3Q19" else _f3d27()
        T = torch.rand(f.shape[1:]) * 0.5 + 0.75
        f_out = apply_buoyancy_3d(f, T, T_ref=1.0, beta=0.001, lattice=lattice)
        assert torch.isfinite(f_out).all()

    def test_rejects_unknown_lattice(self) -> None:
        f = _f3d19()
        T = torch.ones(f.shape[1:])
        with pytest.raises(ValueError, match="lattice"):
            apply_buoyancy_3d(f, T, T_ref=1.0, beta=0.001, lattice="D2Q9")


# ---------------------------------------------------------------------------
# Combined thermal_step
# ---------------------------------------------------------------------------


class TestThermalStep:
    @pytest.mark.parametrize("lattice", ["D3Q19", "D3Q27"])
    def test_shape(self, lattice: str) -> None:
        f = _f3d19() if lattice == "D3Q19" else _f3d27()
        g = _g_thermal()
        f_out, g_out, T_out = thermal_step(f, g, tau_T=TAU_T, lattice=lattice)
        assert f_out.shape == f.shape
        assert g_out.shape == g.shape
        assert T_out.shape == f.shape[1:]

    @pytest.mark.parametrize("lattice", ["D3Q19", "D3Q27"])
    def test_finite(self, lattice: str) -> None:
        f = _f3d19() if lattice == "D3Q19" else _f3d27()
        g = _g_thermal()
        f_out, g_out, T_out = thermal_step(f, g, tau_T=TAU_T, lattice=lattice)
        assert torch.isfinite(f_out).all()
        assert torch.isfinite(g_out).all()
        assert torch.isfinite(T_out).all()

    @pytest.mark.parametrize("lattice", ["D3Q19", "D3Q27"])
    def test_no_buoyancy_preserves_mass(self, lattice: str) -> None:
        """With beta=0, f should be unchanged (no buoyancy coupling)."""
        f = _f3d19() if lattice == "D3Q19" else _f3d27()
        g = _g_thermal()
        f_out, _, _ = thermal_step(f, g, tau_T=TAU_T, lattice=lattice, beta=0.0)
        assert torch.allclose(f_out, f, atol=1e-7)

    @pytest.mark.parametrize("lattice", ["D3Q19", "D3Q27"])
    def test_temperature_conservation(self, lattice: str) -> None:
        """Thermal step should conserve total temperature (periodic)."""
        f = _f3d19() if lattice == "D3Q19" else _f3d27()
        g = _g_thermal()
        T_before = thermal_macroscopic_3d(g)
        _, _, T_after = thermal_step(f, g, tau_T=TAU_T, lattice=lattice)
        assert torch.allclose(T_after.sum(), T_before.sum(), atol=1e-4)

    @pytest.mark.parametrize("lattice", ["D3Q19", "D3Q27"])
    def test_buoyancy_modifies_f(self, lattice: str) -> None:
        """With beta>0 and T≠T_ref, f should be modified."""
        f = _f3d19() if lattice == "D3Q19" else _f3d27()
        g = _g_thermal()
        # Set T away from T_ref
        T_field = torch.full(f.shape[1:], 2.0)
        g = thermal_equilibrium_3d(
            T_field, torch.zeros_like(T_field), torch.zeros_like(T_field), torch.zeros_like(T_field)
        )
        f_out, _, _ = thermal_step(f, g, tau_T=TAU_T, lattice=lattice, beta=0.01, T_ref=1.0)
        assert not torch.allclose(f_out, f, atol=1e-7)

    def test_rejects_unknown_lattice(self) -> None:
        f = _f3d19()
        g = _g_thermal()
        with pytest.raises(ValueError, match="lattice"):
            thermal_step(f, g, tau_T=TAU_T, lattice="D2Q9")


# ---------------------------------------------------------------------------
# Conjugate heat transfer
# ---------------------------------------------------------------------------


class TestConjugateHT:
    def test_shape_2d(self) -> None:
        ny, nx = 8, 12
        T_f = torch.ones((ny, nx))
        T_s = torch.ones((ny, nx))
        mask = torch.zeros((ny, nx), dtype=torch.bool)
        T_f_out, T_s_out = conjugate_ht_step(T_f, T_s, mask, alpha_s=0.1)
        assert T_f_out.shape == T_f.shape
        assert T_s_out.shape == T_s.shape

    def test_shape_3d(self) -> None:
        nz, ny, nx = 4, 6, 8
        T_f = torch.ones((nz, ny, nx))
        T_s = torch.ones((nz, ny, nx))
        mask = torch.zeros((nz, ny, nx), dtype=torch.bool)
        T_f_out, T_s_out = conjugate_ht_step(T_f, T_s, mask, alpha_s=0.1)
        assert T_f_out.shape == T_f.shape
        assert T_s_out.shape == T_s.shape

    def test_no_solid_no_change_to_fluid(self) -> None:
        """With no solid cells, fluid temperature should be unchanged."""
        nz, ny, nx = 4, 6, 8
        T_f = torch.rand((nz, ny, nx))
        T_s = torch.rand((nz, ny, nx))
        mask = torch.zeros((nz, ny, nx), dtype=torch.bool)
        T_f_out, _ = conjugate_ht_step(T_f, T_s, mask, alpha_s=0.1)
        assert torch.allclose(T_f_out, T_f, atol=1e-7)

    def test_solid_diffusion_decays(self) -> None:
        """A sinusoidal solid temperature mode should decay."""
        nz, ny, nx = 1, 1, 32
        mask = torch.ones((nz, ny, nx), dtype=torch.bool)
        k = 2.0 * math.pi / nx
        xs = torch.arange(nx, dtype=torch.float32)
        T_s = 0.1 * torch.sin(k * xs).unsqueeze(0).unsqueeze(0) + 1.0
        T_f = torch.ones_like(T_s)
        alpha_s = 0.1
        n_steps = 50
        for _ in range(n_steps):
            T_f, T_s = conjugate_ht_step(T_f, T_s, mask, alpha_s=alpha_s)
        # Amplitude should decay
        T_mean = float(T_s.mean().item())
        amp = float((T_s - T_mean).abs().max().item())
        assert amp < 0.1, f"Solid temperature did not decay: amp={amp}"

    def test_interface_coupling(self) -> None:
        """Interface cells should have averaged temperature."""
        nz, ny, nx = 1, 1, 8
        mask = torch.zeros((nz, ny, nx), dtype=torch.bool)
        mask[..., 4:] = True  # right half is solid
        T_f = torch.ones((nz, ny, nx)) * 2.0
        T_s = torch.ones((nz, ny, nx)) * 1.0
        T_f_out, T_s_out = conjugate_ht_step(T_f, T_s, mask, alpha_s=0.0, k_ratio=1.0)
        # At interface (k_ratio=1), T_int = 0.5*T_f + 0.5*T_s = 1.5
        # Fluid cell at index 3 (adjacent to solid at 4)
        assert abs(float(T_f_out[..., 3].item()) - 1.5) < 1e-5

    def test_finite_output(self) -> None:
        nz, ny, nx = 4, 6, 8
        T_f = torch.rand((nz, ny, nx))
        T_s = torch.rand((nz, ny, nx))
        mask = torch.zeros((nz, ny, nx), dtype=torch.bool)
        mask[..., 0] = True
        T_f_out, T_s_out = conjugate_ht_step(T_f, T_s, mask, alpha_s=0.1, k_ratio=5.0)
        assert torch.isfinite(T_f_out).all()
        assert torch.isfinite(T_s_out).all()


# ---------------------------------------------------------------------------
# Composability: thermal_step with different collision outputs
# ---------------------------------------------------------------------------


class TestThermalComposability:
    @pytest.mark.parametrize("lattice", ["D3Q19", "D3Q27"])
    def test_compose_with_bgk(self, lattice: str) -> None:
        """thermal_step should accept BGK-collided f without error."""
        from tensorlbm.d3q27 import collide_bgk27, stream27
        from tensorlbm.solver3d import collide_bgk3d, stream3d

        f = _f3d19() if lattice == "D3Q19" else _f3d27()
        g = _g_thermal()
        T = thermal_macroscopic_3d(g)
        tau = 0.7
        for _ in range(3):
            if lattice == "D3Q19":
                f = collide_bgk3d(f, tau)
                f = stream3d(f)
            else:
                f = collide_bgk27(f, tau)
                f = stream27(f)
            f, g, T = thermal_step(f, g, tau_T=TAU_T, lattice=lattice)
        assert torch.isfinite(f).all()
        assert torch.isfinite(g).all()
        assert torch.isfinite(T).all()


@pytest.mark.parametrize("axis,letter", [(0, "z"), (1, "y"), (2, "x")])
def test_hot_wall_nusselt_on_rectangular_lattice(axis, letter):
    from tensorlbm.thermal_common import nusselt_hot_wall_3d

    shape = (4, 7, 9)
    dims = [1, 1, 1]
    dims[axis] = shape[axis]
    t = torch.linspace(1., 0., shape[axis], dtype=torch.float64).reshape(dims).expand(shape)
    assert nusselt_hot_wall_3d(t, 1., 0., letter + "-") == pytest.approx(1.)
    assert nusselt_hot_wall_3d(1-t, 1., 0., letter + "+") == pytest.approx(1.)


@pytest.mark.parametrize("axis,letter", [(0, "z"), (1, "y"), (2, "x")])
@pytest.mark.parametrize("side", ["-", "+"])
def test_wall_gradient_convergence_on_cubic_field(axis, letter, side):
    from tensorlbm.thermal_common import nusselt_hot_wall_3d

    errors = []
    for intervals in (8, 16, 32):
        s = torch.linspace(0, 1, intervals + 1, dtype=torch.float64)
        profile = (1 - s)**3 if side == "-" else s**3
        t = profile.expand(3, 5, -1).movedim(-1, axis)
        nu = nusselt_hot_wall_3d(t, 1., 0., letter + side, order=2)
        errors.append(abs(nu - 3.))
    assert errors[0] / errors[1] == pytest.approx(4.)
    assert errors[1] / errors[2] == pytest.approx(4.)
    assert errors[-1] == pytest.approx(2 / 32**2)


def test_second_order_wall_gradient_requires_three_nodes():
    from tensorlbm.thermal_common import nusselt_hot_wall_3d

    with pytest.raises(ValueError, match="3 nodes"):
        nusselt_hot_wall_3d(torch.ones(4, 4, 2), 1., 0., order=2)


def test_wall_heat_rate_on_six_rectangular_faces(record_property):
    from tensorlbm.thermal_common import wall_heat_rate_3d

    lengths = (2., 1.5, .75)  # x, y, z in meters
    counts = (9, 6, 7)
    axes = [torch.linspace(0., L, n, dtype=torch.float64) for L,n in zip(lengths,counts)]
    z,y,x = torch.meshgrid(*reversed(axes), indexing="ij")
    spacing = tuple(L/(n-1) for L,n in zip(reversed(lengths),reversed(counts)))
    k = 2.75
    a, b = (.7,-.4,.9), (1.2,-.6,.3)
    field = (a[0]*x*x + a[1]*y*y + a[2]*z*z + b[0]*x+b[1]*y+b[2]*z + .8*x*y*z)
    total = 0.
    for i, label in enumerate("xyz"):
        area = lengths[(i+1)%3]*lengths[(i+2)%3]
        for sign, side in ((-1,"-"), (1,"+")):
            wall = label+side
            rate = wall_heat_rate_3d(field,k,wall,spacing_zyx=spacing)
            position = 0. if sign < 0 else lengths[i]
            exact = -sign*k*area*(2*a[i]*position+b[i]+.8*area/4)
            record_property(f"absolute_heat_rate_error_{wall}_W",abs(rate-exact))
            assert abs(rate-exact) < 2e-13
            total += rate
    volume = lengths[0]*lengths[1]*lengths[2]
    residual = abs(total+2*k*sum(a)*volume)
    record_property("divergence_balance_error_W",residual)
    assert residual < 3e-13


def test_wall_heat_rate_weights_face_edges_by_area(record_property):
    from tensorlbm.thermal_common import wall_heat_rate_3d

    errors = []
    for intervals in (4,8,16):
        y = torch.linspace(0., 1., intervals+1, dtype=torch.float64)
        x = torch.tensor([0., .5, 1.],dtype=torch.float64)
        field = (y[:,None]**2*x).expand(3,-1,-1)
        rate = wall_heat_rate_3d(field,1.,"x+",spacing_zyx=(.5,1/intervals,.5),order=1)
        errors.append(abs(rate+1/3))
        record_property(f"wall_quadrature_error_{intervals}_W",errors[-1])
        assert errors[-1] == pytest.approx(1/(6*intervals**2))


def test_d3q7_slab_cooling_wall_heat_and_storage(record_property):
    from tensorlbm.thermal_common import wall_heat_rate_3d

    # Odd periodic extension puts stationary zero-temperature planes at x=0,1.
    # The second sine mode yields unequal wall heat while adding zero net heat.
    # Physical k=C=1. At tau=1 D3Q7 has lattice diffusivity 1/8.
    end = 1/64
    decay = math.exp(-math.pi**2*end)
    first_heat = (1-decay)/math.pi
    second_heat = .1*(1-decay**4)/math.pi
    exact_walls = torch.tensor([first_heat+second_heat, first_heat-second_heat],
                               dtype=torch.float64)
    errors = []
    for n in (8,16,32):
        dx = 1/n
        dt = dx*dx/8
        x = torch.arange(2*n, dtype=torch.float64)*dx
        initial = (torch.sin(math.pi*x)+.2*torch.sin(2*math.pi*x)).expand(3,3,-1)
        zero = torch.zeros_like(initial)
        g = thermal_equilibrium_3d(initial,zero,zero,zero)
        powers = []
        steps = round(end/dt)
        for step in range(steps+1):
            field = thermal_macroscopic_3d(g)
            slab = field[:,:,:n+1]
            powers.append([wall_heat_rate_3d(slab,1.,wall,spacing_zyx=(.5,.5,dx))
                           for wall in ("x-","x+")])
            if step < steps:
                g = thermal_stream_3d(thermal_collide_bgk_3d(g,field,zero,zero,zero,1.))

        heat = torch.trapezoid(torch.tensor(powers,dtype=torch.float64),dx=dt,dim=0)
        # All rows are identical, and transverse cross-sectional area is 1 m².
        stored = torch.trapezoid(slab[1,1],dx=dx).item()
        initial_heat = torch.trapezoid(initial[1,1,:n+1],dx=dx).item()
        wall_error = (heat-exact_walls).abs().max().item()
        storage_error = abs(stored-2*decay/math.pi)
        balance = abs(initial_heat-stored-heat.sum().item())
        record_property(f"max_cumulative_wall_heat_error_{n}_J",wall_error)
        record_property(f"final_storage_error_{n}_J",storage_error)
        record_property(f"heat_balance_residual_{n}_J",balance)
        errors.append((wall_error,storage_error,balance))
    for coarse,fine in zip(errors,errors[1:]):
        assert all(a/b > 3.4 for a,b in zip(coarse,fine))
    assert max(errors[-1]) < 4.5e-4


@pytest.mark.parametrize(("direct_walls", "tau"), [(False,1.), (True,1.), (True,.75), (True,1.5)])
def test_d3q7_cuboid_cooling_six_face_energy_budget(direct_walls, tau, record_property):
    from tensorlbm.thermal_common import thermal_dirichlet_wall_3d, wall_heat_rate_3d

    lengths = (.75, 1.25, 1.)  # z, y, x in meters
    modes = ((1., (1,1,1)), (.08, (2,1,1)), (-.12, (1,2,1)), (.16, (1,1,2)))
    end = 1/64
    faces = ("z-", "z+", "y-", "y+", "x-", "x+")
    exact_heat = dict.fromkeys(faces, 0.)
    exact_storage = 0.
    for amplitude, orders in modes:
        waves = [m*math.pi/L for m,L in zip(orders,lengths)]
        decay_rate = sum(w*w for w in waves)
        # The integral of a sine factor along its side vanishes for even modes.
        integrals = [(1-(-1)**m)/w for m,w in zip(orders,waves)]
        exact_storage += amplitude*math.exp(-end*decay_rate)*math.prod(integrals)
        time_integral = -math.expm1(-end*decay_rate)/decay_rate
        for axis,w in enumerate(waves):
            area_integral = math.prod(integrals[i] for i in range(3) if i != axis)
            heat = amplitude*time_integral*w*area_integral
            exact_heat[faces[2*axis]] += heat
            exact_heat[faces[2*axis+1]] -= (-1)**orders[axis]*heat

    errors = []
    for n in (8,16,24):
        dx = 1/n
        dt = (tau-.5)*dx**2/4  # physical diffusivity k/C=1
        intervals = [round(L*n) for L in lengths]
        coordinates = [torch.arange(m+1 if direct_walls else 2*m,dtype=torch.float64)*dx for m in intervals]
        z,y,x = torch.meshgrid(*coordinates,indexing="ij")
        initial = torch.zeros_like(x)
        for amplitude, orders in modes:
            kz,ky,kx = [m*math.pi/L for m,L in zip(orders,lengths)]
            initial += amplitude*torch.sin(kz*z)*torch.sin(ky*y)*torch.sin(kx*x)
        zero = torch.zeros_like(initial)
        g = thermal_equilibrium_3d(initial,zero,zero,zero)
        box = tuple(slice(0,m+1) for m in intervals)
        powers = []
        steps = round(end/dt)
        for step in range(steps+1):
            field = thermal_macroscopic_3d(g)
            powers.append([wall_heat_rate_3d(field[box],1.,face,spacing_zyx=(dx,dx,dx))
                           for face in faces])
            if step < steps:
                g = thermal_stream_3d(thermal_collide_bgk_3d(g,field,zero,zero,zero,tau))
                if direct_walls:
                    for face in faces:
                        g = thermal_dirichlet_wall_3d(g, 0., face, method="odd_reflection")
        heat = torch.trapezoid(torch.tensor(powers,dtype=torch.float64),dx=dt,dim=0)
        def storage(values):
            for _ in range(3):
                values = torch.trapezoid(values,dx=dx,dim=-1)
            return values.item()
        final = storage(field[box])
        wall_errors = [abs(q.item()-exact_heat[face]) for face,q in zip(faces,heat)]
        for face,error in zip(faces,wall_errors):
            record_property(f"wall_heat_error_{face}_{n}_J",error)
        energy_error = abs(final-exact_storage)
        balance = abs(final+heat.sum().item()-storage(initial[box]))
        record_property(f"storage_error_{n}_J",energy_error)
        record_property(f"balance_error_{n}_J",balance)
        errors.append((max(wall_errors),energy_error,balance))
    for a,b,c in zip(*errors):
        assert a/b > (2.5 if tau == .75 else 3.) and b/c > 2


@pytest.mark.parametrize("tau", [.75, 1.5])
def test_direct_wall_populations_match_odd_extension(tau,record_property):
    from tensorlbm.thermal_common import thermal_dirichlet_wall_3d

    n = 8
    wall_temp = 2.5
    s = torch.arange(2*n,dtype=torch.float64)*math.pi/n
    z,y,x = torch.meshgrid(s,s,s,indexing="ij")
    initial = wall_temp + torch.sin(z)*torch.sin(2*y)*torch.sin(x)
    full_zero = torch.zeros_like(initial)
    full = thermal_equilibrium_3d(initial,full_zero,full_zero,full_zero)
    box = (slice(None),) + (slice(0,n+1),)*3
    direct = full[box].clone()
    zero = torch.zeros_like(direct[0])
    worst = 0.
    for _ in range(12):
        full = thermal_stream_3d(thermal_collide_bgk_3d(
            full,thermal_macroscopic_3d(full),full_zero,full_zero,full_zero,tau))
        direct = thermal_stream_3d(thermal_collide_bgk_3d(
            direct,thermal_macroscopic_3d(direct),zero,zero,zero,tau))
        for face in ("x-","x+","y-","y+","z-","z+"):
            direct = thermal_dirichlet_wall_3d(direct,wall_temp,face,method="odd_reflection")
        worst = max(worst,(direct-full[box]).abs().max().item())
    record_property("maximum_population_difference",worst)
    assert worst < 3e-14


@pytest.mark.parametrize("tau", [.75, 1.5])
def test_insulated_box_even_reflection_and_cosine_decay(tau,record_property):
    from tensorlbm.thermal_common import thermal_adiabatic_walls_3d

    faces = ["x-", "x+", "y-", "y+", "z-", "z+"]
    errors = []
    for n in (8,16):
        dx = 1/n
        dt = (tau-.5)*dx**2/4
        end = 1/64
        s = torch.arange(2*n,dtype=torch.float64)*math.pi/n
        z,y,x = torch.meshgrid(s,s,s,indexing="ij")
        mode = torch.cos(z)*torch.cos(y)*torch.cos(2*x)
        initial = 1.25+mode
        zero = torch.zeros_like(initial)
        extended = thermal_equilibrium_3d(initial,zero,zero,zero)
        box = (slice(0,n+1),)*3
        direct = extended[(slice(None),)+box].clone()
        box_zero = zero[box]
        discrepancy = drift = normal_flux = 0.
        for _ in range(round(end/dt)):
            extended = thermal_stream_3d(thermal_collide_bgk_3d(
                extended,thermal_macroscopic_3d(extended),zero,zero,zero,tau))
            direct = thermal_stream_3d(thermal_collide_bgk_3d(
                direct,thermal_macroscopic_3d(direct),box_zero,box_zero,box_zero,tau))
            direct = thermal_adiabatic_walls_3d(direct,faces,method="even_reflection")
            discrepancy = max(discrepancy,(direct-extended[(slice(None),)+box]).abs().max().item())
            for axis in range(3):
                current = direct[1+2*axis]-direct[2+2*axis]
                for boundary in (0,-1):
                    normal_flux = max(normal_flux,current.select(2-axis,boundary).abs().max().item())
            heat = thermal_macroscopic_3d(direct)
            for _ in range(3):
                heat = torch.trapezoid(heat,dx=dx,dim=-1)
            drift = max(drift,abs(heat.item()-1.25))
        exact = 1.25 + math.exp(-6*math.pi**2*end)*mode[box]
        error = (thermal_macroscopic_3d(direct)-exact).abs().max().item()
        errors.append(error)
        record_property(f"max_temperature_error_{n}_K",error)
        record_property(f"max_population_discrepancy_{n}",discrepancy)
        record_property(f"max_normal_population_flux_{n}",normal_flux)
        record_property(f"max_heat_drift_{n}_J",drift)
        assert normal_flux == 0 and discrepancy < 2e-14 and drift < 2e-14
    assert errors[0]/errors[1] > 3.5


@pytest.mark.parametrize("tau", [.75, 1.5])
def test_mixed_wall_intersections_against_extended_mode(tau, record_property):
    from tensorlbm.thermal_common import thermal_adiabatic_walls_3d, thermal_dirichlet_wall_3d

    # On a unit cube this mode is odd about x=0 and y=1 (fixed T),
    # even about x=1, y=0 and both z faces (insulating). Opposite
    # boundary types on each of x and y yield a four-length period.
    fixed = ("x-", "y+")
    insulated = ["x+", "y-", "z-", "z+"]
    wall = 1.75
    end = 1/64
    errors = []
    for n in (8,16):
        dx = 1/n
        dt = (tau-.5)*dx**2/4
        z,y,x = torch.meshgrid(*(torch.arange(period*n,dtype=torch.float64)*math.pi*dx
                                  for period in (2,4,4)), indexing="ij")
        mode = torch.sin(x/2)*torch.cos(y/2)*torch.cos(z)
        full_zero = torch.zeros_like(mode)
        full = thermal_equilibrium_3d(wall+mode,full_zero,full_zero,full_zero)
        box = (slice(0,n+1),)*3
        direct = full[(slice(None),)+box].clone()
        zero = full_zero[box]
        population_error = order_error = wall_error = 0.
        for _ in range(round(end/dt)):
            full = thermal_stream_3d(thermal_collide_bgk_3d(
                full,thermal_macroscopic_3d(full),full_zero,full_zero,full_zero,tau))
            streamed = thermal_stream_3d(thermal_collide_bgk_3d(
                direct,thermal_macroscopic_3d(direct),zero,zero,zero,tau))
            direct = thermal_adiabatic_walls_3d(streamed,insulated,method="even_reflection")
            other = streamed
            for face in fixed:
                direct = thermal_dirichlet_wall_3d(direct,wall,face,method="odd_reflection")
                other = thermal_dirichlet_wall_3d(other,wall,face,method="odd_reflection")
            other = thermal_adiabatic_walls_3d(other,insulated,method="even_reflection")
            order_error = max(order_error, (direct-other).abs().max().item())
            population_error = max(population_error, (direct-full[(slice(None),)+box]).abs().max().item())
            temperature = thermal_macroscopic_3d(direct)
            wall_error = max(wall_error, (temperature[:,:,0]-wall).abs().max().item(),
                             (temperature[:,-1,:]-wall).abs().max().item())
        exact = wall + math.exp(-1.5*math.pi**2*end)*mode[box]
        errors.append((temperature-exact).abs().max().item())
        record_property(f"field_error_{n}_K", errors[-1])
        record_property(f"population_error_{n}", population_error)
        record_property(f"boundary_order_error_{n}", order_error)
        record_property(f"fixed_wall_error_{n}_K", wall_error)
        assert max(population_error,order_error,wall_error) < 3e-14
    assert errors[0]/errors[1] > 3.5


@pytest.mark.parametrize("axis", ["x","y","z"])
def test_halfway_prescribed_flux_slab_against_transient_solution(axis,record_property):
    from tensorlbm.thermal_common import thermal_flux_wall_pair_3d

    # C=2, k=2, length and cross-sectional area both one. The quadratic
    # profile accepts unequal signed wall fluxes; a cosine adds transient decay.
    capacity, end = 2., 1/32
    inward = (1.4,-.4)
    tau = 1.
    errors = []
    for n in (8,16,32):
        dx = 1/n
        dt = dx**2/8
        x = (torch.arange(n,dtype=torch.float64)+.5)*dx
        steady_part = (.5*sum(inward)*x*x-inward[0]*x)/capacity
        mode = .25*torch.cos(math.pi*x)
        dim = 2-"xyz".index(axis)
        initial = (steady_part+mode).expand(3,2,n).movedim(-1,dim)
        zero = torch.zeros_like(initial)
        g = thermal_equilibrium_3d(initial,zero,zero,zero)
        worst_balance = 0.
        for step in range(1,round(end/dt)+1):
            g = thermal_stream_3d(thermal_collide_bgk_3d(
                g,thermal_macroscopic_3d(g),zero,zero,zero,tau))
            g = thermal_flux_wall_pair_3d(g,axis,inward,dt=dt,spacing=dx,
                                          volumetric_heat_capacity=capacity)
            field = thermal_macroscopic_3d(g)
            energy_change = capacity*(field-initial).mean().item()
            worst_balance = max(worst_balance,abs(energy_change-step*dt*sum(inward)))
        exact = steady_part+end*sum(inward)/capacity+math.exp(-math.pi**2*end)*mode
        error = (field.movedim(dim,-1)-exact).abs().max().item()
        errors.append(error)
        record_property(f"temperature_error_{n}_K",error)
        record_property(f"max_heat_budget_error_{n}_J",worst_balance)
        assert worst_balance < 5e-14
    assert all(a/b > 3.8 for a,b in zip(errors,errors[1:]))


@pytest.mark.parametrize("tau", [0.75, 1.5])
def test_opposed_fluxes_preserve_steady_fourier_field(tau, record_property):
    from tensorlbm.thermal_common import thermal_flux_wall_pair_3d

    spacing, capacity, conductivity = 0.125, 3.0, 2.0
    flux = (1.25, -0.5, 0.75)
    dt = (tau - 0.5) * spacing**2 * capacity / (4 * conductivity)
    z, y, x = torch.meshgrid(
        *((torch.arange(n, dtype=torch.float64) + 0.5) * spacing for n in (4, 6, 8)),
        indexing="ij",
    )
    exact = 2.0 - sum(q * pos for q, pos in zip(flux, (x, y, z))) / conductivity
    zero = torch.zeros_like(exact)
    g = thermal_equilibrium_3d(exact, zero, zero, zero)
    # For linear T, streaming shifts equilibrium by exactly one lattice
    # gradient. The steady pre-collision offset is -tau*w_i*c_i.grad(T)*dx.
    for i, velocity in enumerate(C_D3Q7):
        projected_flux = sum(c * q for c, q in zip(velocity, flux))
        g[i] += tau * W_D3Q7[i] * spacing * projected_flux / conductivity

    max_field_error = max_flux_error = 0.0
    for _ in range(64):
        g = thermal_stream_3d(thermal_collide_bgk_3d(
            g, thermal_macroscopic_3d(g), zero, zero, zero, tau
        ))
        for axis, q in zip("xyz", flux):
            g = thermal_flux_wall_pair_3d(
                g, axis, (q, -q), dt=dt, spacing=spacing, volumetric_heat_capacity=capacity
            )
        field = thermal_macroscopic_3d(g)
        max_field_error = max(max_field_error, (field - exact).abs().max().item())
        for dim, q in zip((2, 1, 0), flux):
            measured = -conductivity * field.diff(dim=dim) / spacing
            max_flux_error = max(max_flux_error, (measured - q).abs().max().item())

    record_property("max_temperature_error_K", max_field_error)
    record_property("max_fourier_flux_error_W_m2", max_flux_error)
    assert max_field_error < 5e-14
    assert max_flux_error < 5e-13


def test_face_flux_maps_preserve_harmonic_polynomial(record_property):
    from tensorlbm.thermal_common import thermal_flux_wall_pair_3d

    dx, capacity, conductivity = 0.1, 1.8, 2.5
    dt = dx**2 * capacity / (8 * conductivity)
    coordinates = [(torch.arange(n, dtype=torch.float64) + 0.5) * dx for n in (5, 7, 9)]
    z, y, x = torch.meshgrid(*coordinates, indexing="ij")
    exact = 1 + x*x - y*y + x*y*z
    zero = torch.zeros_like(exact)
    g = thermal_equilibrium_3d(exact, zero, zero, zero)
    # Inward flux is -k*dT/dx at x=0 and +k*dT/dx at x=L (likewise y,z).
    yz = (y*z)[:, :, 0]
    xz = (x*z)[:, 0, :]
    xy = (x*y)[0, :, :]
    faces = [
        (-conductivity*yz, conductivity*(2*x.shape[2]*dx + yz)),
        (-conductivity*xz, conductivity*(-2*y.shape[1]*dx + xz)),
        (-conductivity*xy, conductivity*xy),
    ]
    worst = 0.0
    for _ in range(64):
        g = thermal_stream_3d(thermal_collide_bgk_3d(
            g, thermal_macroscopic_3d(g), zero, zero, zero, 1.0
        ))
        for axis, pair in zip("xyz", faces):
            g = thermal_flux_wall_pair_3d(
                g, axis, pair, dt=dt, spacing=dx, volumetric_heat_capacity=capacity
            )
        worst = max(worst, (thermal_macroscopic_3d(g) - exact).abs().max().item())
    record_property("max_temperature_error_K", worst)
    assert worst < 5e-14


@pytest.mark.parametrize("flux", [torch.zeros(5, 1), torch.full((5, 7), float("nan"))])
def test_invalid_face_flux_map(flux):
    from tensorlbm.thermal_common import thermal_flux_wall_pair_3d

    with pytest.raises(ValueError):
        thermal_flux_wall_pair_3d(torch.zeros(7, 5, 7, 9), "x", (flux, 0.0))


def test_reversing_heat_flux_against_travelling_thermal_wave(record_property):
    import cmath

    from tensorlbm.thermal_common import thermal_flux_wall_pair_3d

    # Unit diffusivity, conductivity, heat capacity, length and cross-section.
    # The real part of exp(8it - (2+2i)x) solves the heat equation.
    decay = 2 + 2j
    slab_integral = (1 - cmath.exp(-decay)) / decay
    errors = []
    for n in (8, 16, 32):
        dx = 1 / n
        dt = dx**2 / 8
        x = (torch.arange(n, dtype=torch.float64) + 0.5) * dx
        mode = torch.exp(-decay * x).reshape(1, 1, n)
        initial = mode.real
        zero = torch.zeros_like(initial)
        g = thermal_equilibrium_3d(initial, zero, zero, zero)
        worst_field = worst_balance = 0.0
        for step in range(1, round(0.5 / dt) + 1):
            phase = cmath.exp(8j * step * dt)
            average_phase = cmath.exp(8j * (step - 0.5) * dt) * math.sin(4 * dt) / (4 * dt)
            inward = ((decay * average_phase).real,
                      -(decay * average_phase * cmath.exp(-decay)).real)
            g = thermal_stream_3d(thermal_collide_bgk_3d(
                g, thermal_macroscopic_3d(g), zero, zero, zero, 1.0
            ))
            g = thermal_flux_wall_pair_3d(g, "x", inward, dt=dt, spacing=dx)
            field = thermal_macroscopic_3d(g)
            worst_field = max(worst_field, (field - (mode * phase).real).abs().max().item())
            exact_storage_change = ((phase - 1) * slab_integral).real
            worst_balance = max(worst_balance, abs((field - initial).mean().item() - exact_storage_change))
        record_property(f"max_temperature_error_{n}_K", worst_field)
        record_property(f"max_storage_error_{n}_J", worst_balance)
        errors.append(worst_field)
        assert worst_balance < 1e-13
    assert all(coarse / fine > 3.7 for coarse, fine in zip(errors, errors[1:]))


def test_internal_link_flux_is_conservative_for_arbitrary_populations():
    from tensorlbm.thermal_common import thermal_link_fluxes_3d, thermal_flux_wall_pair_3d

    post = torch.randn((7, 3, 1, 4), generator=torch.Generator().manual_seed(41), dtype=torch.float64)
    units = dict(spacing=.35, dt=.012, volumetric_heat_capacity=2.1)
    fluxes = thermal_link_fluxes_3d(post, **units)
    streamed = thermal_stream_3d(post)
    loss = torch.zeros_like(post[0])
    for axis, flux in enumerate(fluxes):
        dim = 2-axis
        low, high = .21+axis, -.12*axis
        streamed = thermal_flux_wall_pair_3d(streamed, "xyz"[axis], (low, high), **units)
        edge = loss.narrow(dim, 0, 1)
        faces = torch.cat((torch.full_like(edge,low), flux, torch.full_like(edge,-high)), dim=dim)
        loss += faces.diff(dim=dim)
    torch.testing.assert_close(streamed.sum(0)-post.sum(0), -loss*.012/(2.1*.35), atol=2e-14, rtol=0.)


def test_periodic_link_flux_includes_seam_and_single_cell_directions():
    from tensorlbm.thermal_common import thermal_link_fluxes_3d

    post = torch.randn((7, 1, 3, 4), generator=torch.Generator().manual_seed(73), dtype=torch.float64)
    all_faces = thermal_link_fluxes_3d(post, periodic=True)
    divergence = torch.zeros_like(post[0])
    for axis, (flux, internal) in enumerate(zip(all_faces, thermal_link_fluxes_3d(post))):
        dim = 2-axis
        torch.testing.assert_close(flux.narrow(dim, 0, flux.shape[dim]-1), internal)
        divergence += flux-flux.roll(1, dims=dim)
    torch.testing.assert_close(thermal_stream_3d(post).sum(0)-post.sum(0), -divergence)


@pytest.mark.parametrize("axes", ["x", "y", "z", "xy", "zx", "yz"])
def test_link_flux_mixed_periodic_axes(axes):
    from tensorlbm.thermal_common import thermal_link_fluxes_3d

    post = torch.arange(7*3*2, dtype=torch.float64).reshape(7, 3, 2, 1)
    internal = thermal_link_fluxes_3d(post)
    wrapped = thermal_link_fluxes_3d(post, periodic=True)
    actual = thermal_link_fluxes_3d(post, periodic=axes)
    for axis, name in enumerate("xyz"):
        torch.testing.assert_close(actual[axis], wrapped[axis] if name in axes else internal[axis])
    with pytest.raises(ValueError, match="distinct"):
        thermal_link_fluxes_3d(post, periodic="xx")


@pytest.mark.parametrize("tau", [.65, 1., 1.4])
def test_initialized_linear_gradient_gives_fourier_flux_on_first_stream(tau):
    from tensorlbm.thermal_common import thermal_gradient_initialization_3d, thermal_link_fluxes_3d

    dx, capacity, alpha = .13, 1.7, .08
    dt = (tau-.5)*dx**2/(4*alpha)
    z,y,x = torch.meshgrid(*(torch.arange(n, dtype=torch.float64)*dx for n in (3,4,5)), indexing="ij")
    t = .7+.63*x-.31*y+.18*z
    zero = torch.zeros_like(t)
    grad = torch.stack([torch.full_like(t, v) for v in (.63,-.31,.18)])
    g = thermal_gradient_initialization_3d(t, zero, zero, zero, grad, tau_T=tau, spacing=dx)
    torch.testing.assert_close(g.sum(0), t, atol=1e-14, rtol=0)
    post = thermal_collide_bgk_3d(g,t,zero,zero,zero,tau)
    fluxes = thermal_link_fluxes_3d(post,dt=dt,spacing=dx,volumetric_heat_capacity=capacity)
    for flux, slope in zip(fluxes, (.63,-.31,.18)):
        torch.testing.assert_close(flux, torch.full_like(flux, -capacity*alpha*slope), atol=1e-13, rtol=0)


def test_periodic_gradient_start_matches_discrete_fourier_derivative():
    from tensorlbm.thermal_common import thermal_gradient_initialization_3d

    nz, ny, nx = 5, 6, 9
    spacing = .12
    z, y, x = torch.meshgrid(*(torch.arange(n, dtype=torch.float64) for n in (nz, ny, nx)), indexing="ij")
    theta = torch.tensor([4*torch.pi/nx, -2*torch.pi/ny, 2*torch.pi/nz], dtype=x.dtype)
    phase = theta[0]*x + theta[1]*y + theta[2]*z
    field = .7 + phase.cos()
    grad = -theta.sin()[:, None, None, None]*phase.sin()/spacing
    zero = torch.zeros_like(field)
    actual = thermal_gradient_initialization_3d(field, zero, zero, zero, spacing=spacing, tau_T=.8)
    expected = thermal_gradient_initialization_3d(field, zero, zero, zero, grad, spacing=spacing, tau_T=.8)
    torch.testing.assert_close(actual, expected, atol=2e-15, rtol=0.)
    torch.testing.assert_close(actual.sum(0), field, atol=2e-15, rtol=0.)


@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("length", [2, 5])
def test_sampled_wall_gradient_uses_one_sided_derivatives(dim, length):
    from tensorlbm.thermal_common import thermal_gradient_initialization_3d

    shape = [3, 3, 3]
    shape[dim] = length
    h = .23
    coord = (torch.arange(length, dtype=torch.float64)+.5)*h
    view = [1, 1, 1]
    view[dim] = length
    x = coord.reshape(view).expand(shape)
    field = x*x+2*x if length > 2 else 2*x
    grad = field.new_zeros((3, *shape))
    grad[2-dim] = 2*x+2 if length > 2 else 2.
    zero = torch.zeros_like(field)
    opts = dict(spacing=h, tau_T=1.4)
    actual = thermal_gradient_initialization_3d(field, zero, zero, zero,
        periodic="xyz".replace("xyz"[2-dim], ""), **opts)
    expected = thermal_gradient_initialization_3d(field, zero, zero, zero, grad, **opts)
    torch.testing.assert_close(actual, expected, atol=2e-15, rtol=0.)


@pytest.mark.parametrize("axis", list("xyz"))
def test_halfway_temperature_pair_tracks_local_wall_heat(axis):
    from tensorlbm.thermal_common import thermal_stream_3d, thermal_temperature_wall_pair_3d

    post = torch.randn(7, 3, 5, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(421))
    streamed = thermal_stream_3d(post)
    dim = 2-"xyz".index(axis)
    pos = 1+2*"xyz".index(axis)
    shape = post[0].select(dim,0).shape
    wall = torch.arange(math.prod(shape), dtype=post.dtype).reshape(shape)/9
    updated, flux = thermal_temperature_wall_pair_3d(streamed, axis, (wall,.31), dt=.2, spacing=.3, volumetric_heat_capacity=2.)
    for i, (incoming, outgoing, index, temp) in enumerate(((pos,pos+1,0,wall),(pos+1,pos,-1,.31))):
        escaped = post[outgoing].select(dim,index)
        returned = updated[incoming].select(dim,index)
        torch.testing.assert_close(returned, .25*temp-escaped)
        torch.testing.assert_close(flux[i], 3*(returned-escaped))
    torch.testing.assert_close(updated.sum()-post.sum(), (flux[0].sum()+flux[1].sum())/3)
