"""Analytic verification of the FWH thickness (monopole) term.

The thickness term (Farassat 1A, surface Mach number ``M = 0``) is validated
against the classical pulsating-sphere solution:

* compact monopole::

      p'(r, t) = -(rho0 a^2 U omega / r) sin(omega (t - r/c0))

* exact exterior solution with the radius velocity prescribed at ``r = a``::

      p = Re[A e^{i(kr - omega t)}/r],  |A| = omega rho0 a^2 U / sqrt(1 + (ka)^2)

Tolerances are stated per test.  The ``ka -> 0`` limit is where agreement with
the exact solution is expected; the ``(ka)^2/2`` residual is a property of any
compact-source far-field analogy and is documented in
``docs/fwh_thickness_monopole_verification.md``.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.acoustics import AcousticObserver, FWHSurface, compute_fwh_far_field

C0 = 343.0
RHO0 = 1.225
A_SPHERE = 0.05
U_SURF = 0.1


def _fibonacci_sphere(n: int, radius: float) -> tuple[torch.Tensor, torch.Tensor]:
    i = torch.arange(n, dtype=torch.float64) + 0.5
    phi = torch.acos(1.0 - 2.0 * i / n)
    theta = math.pi * (1.0 + 5.0**0.5) * i
    normals = torch.stack(
        [torch.sin(phi) * torch.cos(theta), torch.sin(phi) * torch.sin(theta), torch.cos(phi)], dim=-1
    )
    return radius * normals, normals


def _pulsating_surface(
    n_src: int = 128,
    f: float = 109.4,
    r_obs: float = 5.0,
    u_amp: float = U_SURF,
    rho0: float = RHO0,
) -> tuple[FWHSurface, torch.Tensor, float]:
    """Breathing sphere: uniform outward normal velocity v_n = u_amp cos(omega t)."""
    dt = (1.0 / f) / 256.0
    arrival = r_obs / C0
    n_steps = int(arrival / dt) + 1200
    t = torch.arange(n_steps, dtype=torch.float64) * dt
    omega = 2.0 * math.pi * f

    pos, nrm = _fibonacci_sphere(n_src, A_SPHERE)
    surf = FWHSurface(
        positions=pos,
        normals=nrm,
        areas=torch.full((n_src,), 4.0 * math.pi * A_SPHERE**2 / n_src, dtype=torch.float64),
        pressure=torch.zeros((n_src, n_steps), dtype=torch.float64),
        dt=dt,
        c0=C0,
        rho0=rho0,
        normal_velocity=u_amp * torch.cos(omega * t).unsqueeze(0).expand(n_src, n_steps),
    )
    return surf, t, arrival


def _fit_amp_phase(p: torch.Tensor, t: torch.Tensor, f: float) -> tuple[float, float]:
    omega = 2.0 * math.pi * f
    basis = torch.stack([torch.cos(omega * t), torch.sin(omega * t)], dim=1)
    coef, *_ = torch.linalg.lstsq(basis, p.unsqueeze(1))
    a_coef, b_coef = float(coef[0, 0]), float(coef[1, 0])
    return math.hypot(a_coef, b_coef), math.atan2(-b_coef, a_coef)


def _steady(t: torch.Tensor, arrival: float, f: float) -> torch.Tensor:
    return t > arrival + 3.0 / f


def test_pulsating_sphere_matches_compact_monopole() -> None:
    """ka = 0.10: amplitude within 1% and phase within 1 deg of the compact monopole."""
    f = 109.4
    ka = 2.0 * math.pi * f * A_SPHERE / C0
    assert ka < 0.11
    surf, t, arrival = _pulsating_surface(f=f)
    p, _ = compute_fwh_far_field(surf, [AcousticObserver(x=5.0, y=0.0, z=0.0, label="o")])
    sel = _steady(t, arrival, f)

    omega = 2.0 * math.pi * f
    p_ana = -(RHO0 * A_SPHERE**2 * U_SURF * omega / 5.0) * torch.sin(omega * (t - 5.0 / C0))
    amp_fwh, phase_fwh = _fit_amp_phase(p[0][sel], t[sel], f)
    amp_ana, phase_ana = _fit_amp_phase(p_ana[sel], t[sel], f)

    assert amp_fwh == pytest.approx(amp_ana, rel=0.01)
    phase_err_deg = math.degrees((phase_fwh - phase_ana + math.pi) % (2 * math.pi) - math.pi)
    assert abs(phase_err_deg) < 1.0


def test_pulsating_sphere_within_3pct_of_exact_solution() -> None:
    """ka = 0.183: relative RMS error against the exact exterior solution < 3%."""
    f = 200.0
    ka = 2.0 * math.pi * f * A_SPHERE / C0
    assert ka < 0.19
    surf, t, arrival = _pulsating_surface(f=f, n_src=256)
    p, _ = compute_fwh_far_field(surf, [AcousticObserver(x=5.0, y=0.0, z=0.0, label="o")])
    sel = _steady(t, arrival, f)

    omega = 2.0 * math.pi * f
    k = omega / C0
    amp = -1j * omega * RHO0 * A_SPHERE**2 * U_SURF * complex(math.cos(-k * A_SPHERE), math.sin(-k * A_SPHERE))
    amp /= complex(1.0, -k * A_SPHERE)
    vals = amp * complex(math.cos(k * 5.0), math.sin(k * 5.0)) / 5.0
    p_ex = torch.tensor(
        [
            (vals * complex(math.cos(-omega * ti), math.sin(-omega * ti))).real
            for ti in t.tolist()
        ],
        dtype=torch.float64,
    )
    rel_rms = float((p[0][sel] - p_ex[sel]).norm() / p_ex[sel].norm())
    assert rel_rms < 0.03


def test_monopole_is_isotropic() -> None:
    """A monopole has no directivity: amplitudes at 5 observer directions agree within 0.5%."""
    f = 109.4
    surf, t, arrival = _pulsating_surface(f=f)
    r = 5.0
    dirs = [(r, 0.0, 0.0), (0.0, r, 0.0), (0.0, 0.0, r), (-r, 0.0, 0.0), (r / math.sqrt(3.0),) * 3]
    obs = [AcousticObserver(x=x, y=y, z=z, label=f"o{k}") for k, (x, y, z) in enumerate(dirs)]
    p, _ = compute_fwh_far_field(surf, obs)
    sel = _steady(t, arrival, f)
    amps = [_fit_amp_phase(p[k][sel], t[sel], f)[0] for k in range(len(dirs))]
    spread = (max(amps) - min(amps)) / (sum(amps) / len(amps))
    assert spread < 0.005


def test_thickness_term_decays_as_inverse_distance() -> None:
    """Far-field monopole: p * r is invariant across r = 5, 10, 20 m (within 1%)."""
    f = 109.4
    products = []
    for r in (5.0, 10.0, 20.0):
        surf, t, arrival = _pulsating_surface(f=f, r_obs=r)
        p, _ = compute_fwh_far_field(surf, [AcousticObserver(x=r, y=0.0, z=0.0, label="o")])
        sel = _steady(t, arrival, f)
        products.append(_fit_amp_phase(p[0][sel], t[sel], f)[0] * r)
    for value in products:
        assert value == pytest.approx(products[0], rel=0.01)


def test_rigid_body_thickness_term_is_exactly_zero() -> None:
    """A non-deforming body has v_n = 0: the thickness term contributes exactly nothing."""
    n_src, n_steps = 64, 256
    pos, nrm = _fibonacci_sphere(n_src, A_SPHERE)
    surf = FWHSurface(
        positions=pos,
        normals=nrm,
        areas=torch.full((n_src,), 4.0 * math.pi * A_SPHERE**2 / n_src, dtype=torch.float64),
        pressure=torch.zeros((n_src, n_steps), dtype=torch.float64),
        dt=1.0e-5,
        c0=C0,
        normal_velocity=torch.zeros((n_src, n_steps), dtype=torch.float64),
    )
    p, _ = compute_fwh_far_field(surf, [AcousticObserver(x=5.0, y=0.0, z=0.0, label="o")])
    assert float(p.abs().max()) == 0.0


def test_regression_loading_only_path_unchanged() -> None:
    """normal_velocity=None reproduces the historical loading-only result exactly."""
    n_src, n_steps = 32, 512
    pos, nrm = _fibonacci_sphere(n_src, A_SPHERE)
    # the observer is ~1 m away, so the record must span the propagation delay
    t = torch.arange(n_steps, dtype=torch.float64) * 5.0e-5
    # dipole surface pressure p_s = P0 cos(theta) sin(omega t): a closed sphere with a
    # uniform pressure would radiate exactly nothing (net force zero), so use a dipole.
    pressure = (
        1.0e-3
        * nrm[:, 2:3]
        * torch.sin(2.0 * math.pi * 500.0 * t).unsqueeze(0).expand(n_src, n_steps)
    )
    base = dict(
        positions=pos,
        normals=nrm,
        areas=torch.full((n_src,), 4.0 * math.pi * A_SPHERE**2 / n_src, dtype=torch.float64),
        pressure=pressure,
        dt=5.0e-5,
        c0=C0,
    )
    obs = [AcousticObserver(x=1.0, y=0.3, z=0.2, label="o")]

    p_none, _ = compute_fwh_far_field(FWHSurface(**base), obs)
    p_zero, _ = compute_fwh_far_field(
        FWHSurface(**base, normal_velocity=torch.zeros((n_src, n_steps), dtype=torch.float64)), obs
    )
    assert torch.equal(p_none, p_zero)
    # a dipole loading term is non-zero, so this is not a trivial equality
    assert float(p_none.abs().max()) > 0.0


def test_zero_net_volume_flux_radiates_weakly() -> None:
    """v_n = U cos(theta) has zero net flux; the compact residual stays below 10%."""
    f = 200.0
    n_src = 128
    surf_pu, _, _ = _pulsating_surface(f=f, n_src=n_src)
    pos, nrm = _fibonacci_sphere(n_src, A_SPHERE)
    n_steps = surf_pu.normal_velocity.shape[1]
    t = torch.arange(n_steps, dtype=torch.float64) * surf_pu.dt
    omega = 2.0 * math.pi * f
    surf_tr = FWHSurface(
        positions=pos,
        normals=nrm,
        areas=surf_pu.areas,
        pressure=torch.zeros((n_src, n_steps), dtype=torch.float64),
        dt=surf_pu.dt,
        c0=C0,
        rho0=RHO0,
        normal_velocity=U_SURF * nrm[:, 2:3] * torch.cos(omega * t).unsqueeze(0),
    )
    obs = [AcousticObserver(x=5.0, y=0.0, z=0.0, label="o")]
    p_tr, _ = compute_fwh_far_field(surf_tr, obs)
    p_pu, _ = compute_fwh_far_field(surf_pu, obs)
    assert float(p_tr.abs().max()) < 0.10 * float(p_pu.abs().max())


def test_amplitude_scales_with_rho0() -> None:
    """The thickness term is proportional to ambient density rho0."""
    f = 109.4
    amps = {}
    for rho0 in (RHO0, 2.0 * RHO0):
        surf, t, arrival = _pulsating_surface(f=f, rho0=rho0)
        p, _ = compute_fwh_far_field(surf, [AcousticObserver(x=5.0, y=0.0, z=0.0, label="o")])
        sel = _steady(t, arrival, f)
        amps[rho0] = _fit_amp_phase(p[0][sel], t[sel], f)[0]
    assert amps[2.0 * RHO0] == pytest.approx(2.0 * amps[RHO0], rel=1e-6)


def test_normal_velocity_shape_is_validated() -> None:
    """Mismatched normal_velocity shape is rejected before any integration."""
    n_src, n_steps = 8, 16
    pos, nrm = _fibonacci_sphere(n_src, A_SPHERE)
    surf = FWHSurface(
        positions=pos,
        normals=nrm,
        areas=torch.full((n_src,), 4.0 * math.pi * A_SPHERE**2 / n_src, dtype=torch.float64),
        pressure=torch.zeros((n_src, n_steps), dtype=torch.float64),
        dt=1.0e-5,
        c0=C0,
        normal_velocity=torch.zeros((n_src, n_steps - 1), dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="normal_velocity must have shape"):
        compute_fwh_far_field(surf, [AcousticObserver(x=1.0, y=0.0, z=0.0, label="o")])


def test_nonpositive_rho0_is_rejected_with_thickness_term() -> None:
    """rho0 <= 0 is a nonsense ambient density when the thickness term is active."""
    n_src, n_steps = 8, 16
    pos, nrm = _fibonacci_sphere(n_src, A_SPHERE)
    surf = FWHSurface(
        positions=pos,
        normals=nrm,
        areas=torch.full((n_src,), 1.0, dtype=torch.float64),
        pressure=torch.zeros((n_src, n_steps), dtype=torch.float64),
        dt=1.0e-5,
        c0=C0,
        rho0=0.0,
        normal_velocity=torch.zeros((n_src, n_steps), dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="rho0 must be positive"):
        compute_fwh_far_field(surf, [AcousticObserver(x=1.0, y=0.0, z=0.0, label="o")])