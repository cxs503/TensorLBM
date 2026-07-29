"""Post-processing utilities for TensorLBM simulation data.

Provides:
- :func:`extract_velocity_profile`       – velocity slice at a fixed x or y position.
- :func:`extract_wake_profile`           – cross-stream velocity profile at a given x-index.
- :func:`compute_recirculation_length`   – x-extent of the reverse-flow region.
- :func:`compute_pressure_coefficient`   – pressure coefficient Cp field.
- :func:`compute_q_criterion`            – Q-criterion for 3-D vortex identification.
- :func:`compute_lambda2_criterion`      – λ₂ criterion for 3-D vortex identification.
- :func:`compute_vorticity_2d`           – z-vorticity scalar field for 2-D flows.
- :func:`compute_vorticity_3d`           – vorticity vector field for 3-D flows.
- :func:`compute_velocity_magnitude`     – velocity magnitude |u| for 2-D/3-D flows.
- :func:`compute_kinetic_energy`         – kinetic energy ½|u|² per cell for 2-D/3-D flows.
- :func:`compute_enstrophy_2d`           – enstrophy ½ωz² for 2-D flows.
- :func:`compute_divergence`             – velocity divergence ∇·u for 2-D/3-D flows.
- :func:`compute_drag_lift_coefficients` – drag and lift coefficients from force data.
- :class:`RunningStats`                  – online accumulator for time-averaged statistics.
"""
from __future__ import annotations

import torch


def extract_velocity_profile(
    ux: torch.Tensor,
    uy: torch.Tensor,
    axis: str = "x",
    index: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract a 1-D velocity profile by slicing the 2-D velocity fields.

    Args:
        ux: x-velocity field, shape ``(ny, nx)``.
        uy: y-velocity field, shape ``(ny, nx)``.
        axis: ``"x"`` to slice at a constant *x* (returns a profile along y),
              ``"y"`` to slice at a constant *y* (returns a profile along x).
        index: Grid index along the chosen axis.

    Returns:
        Tuple ``(ux_profile, uy_profile)`` — 1-D tensors of length ``ny``
        (when *axis* = ``"x"``) or ``nx`` (when *axis* = ``"y"``).
    """
    if axis == "x":
        return ux[:, index], uy[:, index]
    if axis == "y":
        return ux[index, :], uy[index, :]
    raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")


def extract_wake_profile(
    ux: torch.Tensor,
    x_wake: int,
) -> torch.Tensor:
    """Extract the streamwise velocity profile at a given x-index (2-D or 3-D).

    For a 2-D field ``(ny, nx)`` returns a 1-D profile of length ``ny``.
    For a 3-D field ``(nz, ny, nx)`` returns the mid-z slice as a 1-D
    profile of length ``ny``.

    Args:
        ux: Streamwise (x) velocity field, shape ``(ny, nx)`` or
            ``(nz, ny, nx)``.
        x_wake: x-index of the wake cross-section.

    Returns:
        1-D streamwise velocity profile of length ``ny``.
    """
    if ux.ndim == 2:
        return ux[:, x_wake]
    if ux.ndim == 3:
        mid_z = ux.shape[0] // 2
        return ux[mid_z, :, x_wake]
    raise ValueError(f"ux must be 2-D or 3-D, got {ux.ndim}-D")


def compute_recirculation_length(
    ux: torch.Tensor,
    obstacle_mask: torch.Tensor,
) -> float:
    """Compute the x-extent of the reverse-flow (recirculation) region.

    Identifies the longest contiguous run of grid columns downstream of the
    obstacle in which the centreline streamwise velocity ``ux`` is negative.

    For 2-D inputs the centreline is the mid-y row; for 3-D inputs it is the
    mid-z, mid-y line.

    Args:
        ux: Streamwise velocity field, shape ``(ny, nx)`` or ``(nz, ny, nx)``.
        obstacle_mask: Boolean solid-cell mask, same shape as *ux*.

    Returns:
        Length of the recirculation zone in lattice units (0.0 if none found).
    """
    if ux.ndim == 2:
        ny, nx = ux.shape
        mid_y = ny // 2
        centreline = ux[mid_y, :]         # (nx,)
        obs_line = obstacle_mask[mid_y, :]
    elif ux.ndim == 3:
        nz, ny, nx = ux.shape
        mid_z, mid_y = nz // 2, ny // 2
        centreline = ux[mid_z, mid_y, :]  # (nx,)
        obs_line = obstacle_mask[mid_z, mid_y, :]
    else:
        raise ValueError(f"ux must be 2-D or 3-D, got {ux.ndim}-D")

    # Find the last solid column (obstacle trailing edge)
    solid_cols = obs_line.nonzero(as_tuple=True)[0]
    start_col = 0 if solid_cols.numel() == 0 else int(solid_cols.max().item()) + 1

    # Count consecutive columns with ux < 0 starting from the trailing edge
    recirculation_len = 0.0
    for xi in range(start_col, nx):
        if float(centreline[xi].item()) < 0.0:
            recirculation_len += 1.0
        else:
            break
    return recirculation_len


def compute_pressure_coefficient(
    rho: torch.Tensor,
    u_in: float,
    rho_ref: float = 1.0,
    cs2: float = 1.0 / 3.0,
) -> torch.Tensor:
    """Compute the pressure coefficient field Cp.

    In LBM the equation of state is :math:`p = c_s^2 \\rho`, so the pressure
    fluctuation relative to the reference state is:

    .. math::

        C_p = \\frac{p - p_{ref}}{\\tfrac{1}{2} \\rho_{ref} U^2}
            = \\frac{c_s^2 (\\rho - \\rho_{ref})}{\\tfrac{1}{2} \\rho_{ref} U^2}

    Args:
        rho: Density field of shape ``(ny, nx)`` or ``(nz, ny, nx)``.
        u_in: Reference inlet velocity :math:`U`.
        rho_ref: Reference density (default 1.0).
        cs2: Lattice speed of sound squared (default 1/3).

    Returns:
        Cp field of the same shape as *rho*.
    """
    dyn_pressure = 0.5 * rho_ref * u_in**2
    if dyn_pressure == 0.0:
        return torch.zeros_like(rho)
    p = cs2 * rho
    p_ref = cs2 * rho_ref
    return (p - p_ref) / dyn_pressure


def _grad3d(field: torch.Tensor, dim: int) -> torch.Tensor:
    """Central-difference gradient of a 3-D field along *dim* with edge padding.

    Args:
        field: 3-D tensor of shape ``(nz, ny, nx)``.
        dim: Dimension to differentiate: 0 → z, 1 → y, 2 → x.

    Returns:
        Gradient tensor with the same shape as *field*.
    """
    g = torch.zeros_like(field)
    if dim == 0:
        g[1:-1] = 0.5 * (field[2:] - field[:-2])
        g[0] = field[1] - field[0]
        g[-1] = field[-1] - field[-2]
    elif dim == 1:
        g[:, 1:-1] = 0.5 * (field[:, 2:] - field[:, :-2])
        g[:, 0] = field[:, 1] - field[:, 0]
        g[:, -1] = field[:, -1] - field[:, -2]
    else:
        g[:, :, 1:-1] = 0.5 * (field[:, :, 2:] - field[:, :, :-2])
        g[:, :, 0] = field[:, :, 1] - field[:, :, 0]
        g[:, :, -1] = field[:, :, -1] - field[:, :, -2]
    return g


def _grad2d(field: torch.Tensor, dim: int) -> torch.Tensor:
    """Central-difference gradient of a 2-D field along *dim* with edge padding.

    Args:
        field: 2-D tensor of shape ``(ny, nx)``.
        dim: Dimension to differentiate: 0 → y, 1 → x.

    Returns:
        Gradient tensor with the same shape as *field*.
    """
    g = torch.zeros_like(field)
    if dim == 0:
        g[1:-1, :] = 0.5 * (field[2:, :] - field[:-2, :])
        g[0, :] = field[1, :] - field[0, :]
        g[-1, :] = field[-1, :] - field[-2, :]
    else:
        g[:, 1:-1] = 0.5 * (field[:, 2:] - field[:, :-2])
        g[:, 0] = field[:, 1] - field[:, 0]
        g[:, -1] = field[:, -1] - field[:, -2]
    return g


def compute_q_criterion(
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
) -> torch.Tensor:
    """Compute the Q-criterion for 3-D vortex identification.

    The Q-criterion is defined as:

    .. math::

        Q = \\tfrac{1}{2}\\left(\\|\\boldsymbol{\\Omega}\\|_F^2
            - \\|\\mathbf{S}\\|_F^2\\right)

    where :math:`\\boldsymbol{\\Omega}` is the antisymmetric (rotation) part
    and :math:`\\mathbf{S}` is the symmetric (strain-rate) part of the
    velocity gradient tensor. Vortex cores are regions where *Q* > 0.

    Uses second-order central differences for interior cells; boundary rows
    use forward/backward differences.

    Args:
        ux: x-velocity, shape ``(nz, ny, nx)``.
        uy: y-velocity, shape ``(nz, ny, nx)``.
        uz: z-velocity, shape ``(nz, ny, nx)``.

    Returns:
        Q-criterion field of shape ``(nz, ny, nx)``.
    """
    dudx, dudy, dudz = _grad3d(ux, 2), _grad3d(ux, 1), _grad3d(ux, 0)
    dvdx, dvdy, dvdz = _grad3d(uy, 2), _grad3d(uy, 1), _grad3d(uy, 0)
    dwdx, dwdy, dwdz = _grad3d(uz, 2), _grad3d(uz, 1), _grad3d(uz, 0)

    s_xx = dudx
    s_yy = dvdy
    s_zz = dwdz
    s_xy = 0.5 * (dudy + dvdx)
    s_xz = 0.5 * (dudz + dwdx)
    s_yz = 0.5 * (dvdz + dwdy)
    s_sq = s_xx**2 + s_yy**2 + s_zz**2 + 2.0 * (s_xy**2 + s_xz**2 + s_yz**2)

    w_xy = 0.5 * (dudy - dvdx)
    w_xz = 0.5 * (dudz - dwdx)
    w_yz = 0.5 * (dvdz - dwdy)
    omega_sq = 2.0 * (w_xy**2 + w_xz**2 + w_yz**2)

    return 0.5 * (omega_sq - s_sq)


def compute_lambda2_criterion(
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
) -> torch.Tensor:
    """Compute the λ₂ criterion for 3-D vortex identification.

    The λ₂ criterion identifies vortex cores as regions where the second
    (middle) eigenvalue of the symmetric tensor :math:`\\mathbf{S}^2 +
    \\boldsymbol{\\Omega}^2` is negative, where :math:`\\mathbf{S}` and
    :math:`\\boldsymbol{\\Omega}` are the symmetric and antisymmetric parts
    of the velocity gradient tensor respectively.

    Unlike the Q-criterion, λ₂ < 0 always indicates a vortex core even in
    the presence of strong unsteady irrotational straining.

    Uses second-order central differences with first-order boundary stencils.
    Eigenvalues are computed via :func:`torch.linalg.eigvalsh` on batched
    3 × 3 symmetric matrices.

    Args:
        ux: x-velocity, shape ``(nz, ny, nx)``.
        uy: y-velocity, shape ``(nz, ny, nx)``.
        uz: z-velocity, shape ``(nz, ny, nx)``.

    Returns:
        λ₂ field of shape ``(nz, ny, nx)``.  Vortex cores are where λ₂ < 0.
    """
    dudx, dudy, dudz = _grad3d(ux, 2), _grad3d(ux, 1), _grad3d(ux, 0)
    dvdx, dvdy, dvdz = _grad3d(uy, 2), _grad3d(uy, 1), _grad3d(uy, 0)
    dwdx, dwdy, dwdz = _grad3d(uz, 2), _grad3d(uz, 1), _grad3d(uz, 0)

    # Symmetric strain-rate components S_ij = (∂_j u_i + ∂_i u_j) / 2
    s_xx = dudx
    s_yy = dvdy
    s_zz = dwdz
    s_xy = 0.5 * (dudy + dvdx)
    s_xz = 0.5 * (dudz + dwdx)
    s_yz = 0.5 * (dvdz + dwdy)

    # Antisymmetric rotation-rate components Ω_ij = (∂_j u_i − ∂_i u_j) / 2
    w_xy = 0.5 * (dudy - dvdx)
    w_xz = 0.5 * (dudz - dwdx)
    w_yz = 0.5 * (dvdz - dwdy)

    # S² + Ω² (symmetric): component-wise product of S and Ω matrices
    # M_ij = Σ_k (S_ik S_kj + Ω_ik Ω_kj)
    m_xx = (s_xx * s_xx + s_xy * s_xy + s_xz * s_xz
            - w_xy * w_xy - w_xz * w_xz)
    m_yy = (s_xy * s_xy + s_yy * s_yy + s_yz * s_yz
            - w_xy * w_xy - w_yz * w_yz)
    m_zz = (s_xz * s_xz + s_yz * s_yz + s_zz * s_zz
            - w_xz * w_xz - w_yz * w_yz)
    m_xy = (s_xx * s_xy + s_xy * s_yy + s_xz * s_yz
            + w_xy * s_xx - w_xy * s_yy - w_xz * w_yz)
    m_xz = (s_xx * s_xz + s_xy * s_yz + s_xz * s_zz
            + w_xz * s_xx - w_xy * w_yz - w_xz * s_zz)
    m_yz = (s_xy * s_xz + s_yy * s_yz + s_yz * s_zz
            + w_yz * s_yy - w_xy * w_xz - w_yz * s_zz)

    shape = ux.shape
    n = ux.numel()

    # Build batched (n, 3, 3) symmetric matrix for eigvalsh
    M = torch.stack(
        [
            m_xx.reshape(n), m_xy.reshape(n), m_xz.reshape(n),
            m_xy.reshape(n), m_yy.reshape(n), m_yz.reshape(n),
            m_xz.reshape(n), m_yz.reshape(n), m_zz.reshape(n),
        ],
        dim=1,
    ).reshape(n, 3, 3)

    # eigvalsh returns eigenvalues in ascending order → λ₂ is index 1
    eigs = torch.linalg.eigvalsh(M.float())
    return eigs[:, 1].reshape(shape)


def compute_vorticity_2d(
    ux: torch.Tensor,
    uy: torch.Tensor,
) -> torch.Tensor:
    """Compute the z-component of vorticity for a 2-D flow.

    .. math::

        \\omega_z = \\frac{\\partial u_y}{\\partial x}
                  - \\frac{\\partial u_x}{\\partial y}

    Uses second-order central differences for interior cells; boundary rows
    use first-order forward/backward differences.

    Args:
        ux: x-velocity, shape ``(ny, nx)``.
        uy: y-velocity, shape ``(ny, nx)``.

    Returns:
        ωz scalar field of shape ``(ny, nx)``.
    """
    if ux.ndim != 2:
        raise ValueError(f"ux must be 2-D, got {ux.ndim}-D")
    duy_dx = _grad2d(uy, 1)
    dux_dy = _grad2d(ux, 0)
    return duy_dx - dux_dy


def compute_vorticity_3d(
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the vorticity vector field for a 3-D flow.

    Returns the three vorticity components:

    .. math::

        \\omega_x = \\frac{\\partial u_z}{\\partial y} - \\frac{\\partial u_y}{\\partial z}

        \\omega_y = \\frac{\\partial u_x}{\\partial z} - \\frac{\\partial u_z}{\\partial x}

        \\omega_z = \\frac{\\partial u_y}{\\partial x} - \\frac{\\partial u_x}{\\partial y}

    Uses second-order central differences for interior cells; boundary rows
    use first-order forward/backward differences.

    Args:
        ux: x-velocity, shape ``(nz, ny, nx)``.
        uy: y-velocity, shape ``(nz, ny, nx)``.
        uz: z-velocity, shape ``(nz, ny, nx)``.

    Returns:
        Tuple ``(omega_x, omega_y, omega_z)`` each of shape ``(nz, ny, nx)``.
    """
    duz_dy = _grad3d(uz, 1)
    duy_dz = _grad3d(uy, 0)
    dux_dz = _grad3d(ux, 0)
    duz_dx = _grad3d(uz, 2)
    duy_dx = _grad3d(uy, 2)
    dux_dy = _grad3d(ux, 1)

    omega_x = duz_dy - duy_dz
    omega_y = dux_dz - duz_dx
    omega_z = duy_dx - dux_dy

    return omega_x, omega_y, omega_z


def compute_velocity_magnitude(
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the velocity magnitude field |u|.

    Supports 2-D (``ux``, ``uy``) and 3-D (``ux``, ``uy``, ``uz``) flows.

    Args:
        ux: x-velocity field, shape ``(ny, nx)`` or ``(nz, ny, nx)``.
        uy: y-velocity field, same shape as *ux*.
        uz: z-velocity field (optional), same shape as *ux*; pass ``None``
            for 2-D flows.

    Returns:
        Velocity magnitude field of the same shape as *ux*.
    """
    mag_sq = ux**2 + uy**2
    if uz is not None:
        mag_sq = mag_sq + uz**2
    return torch.sqrt(mag_sq)


def compute_kinetic_energy(
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the specific kinetic energy field ½|u|² per lattice cell.

    Supports 2-D and 3-D flows.

    Args:
        ux: x-velocity field, shape ``(ny, nx)`` or ``(nz, ny, nx)``.
        uy: y-velocity field, same shape as *ux*.
        uz: z-velocity field (optional), same shape as *ux*; pass ``None``
            for 2-D flows.

    Returns:
        Kinetic energy field of the same shape as *ux*.
    """
    ke = 0.5 * (ux**2 + uy**2)
    if uz is not None:
        ke = ke + 0.5 * uz**2
    return ke


def compute_enstrophy_2d(
    ux: torch.Tensor,
    uy: torch.Tensor,
) -> torch.Tensor:
    """Compute the enstrophy field for a 2-D flow.

    Enstrophy is defined as half the squared vorticity magnitude:

    .. math::

        \\mathcal{E} = \\tfrac{1}{2}\\,\\omega_z^2

    Args:
        ux: x-velocity, shape ``(ny, nx)``.
        uy: y-velocity, shape ``(ny, nx)``.

    Returns:
        Enstrophy field of shape ``(ny, nx)``.
    """
    omega_z = compute_vorticity_2d(ux, uy)
    return 0.5 * omega_z**2


def compute_divergence(
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the velocity divergence ∇·u.

    For an incompressible LBM flow the divergence should be approximately
    zero everywhere; this function is useful as a diagnostic to verify
    mass conservation or detect numerical artefacts.

    Supports 2-D (``ux``, ``uy``) and 3-D (``ux``, ``uy``, ``uz``) flows.
    Uses second-order central differences for interior cells and first-order
    forward/backward differences at boundaries.

    Args:
        ux: x-velocity field, shape ``(ny, nx)`` or ``(nz, ny, nx)``.
        uy: y-velocity field, same shape as *ux*.
        uz: z-velocity field (optional), same shape as *ux*; pass ``None``
            for 2-D flows.

    Returns:
        Divergence scalar field of the same shape as *ux*.
    """
    if ux.ndim == 2:
        if uz is not None:
            raise ValueError("uz must be None for 2-D divergence")
        return _grad2d(ux, 1) + _grad2d(uy, 0)
    if ux.ndim == 3:
        div = _grad3d(ux, 2) + _grad3d(uy, 1)
        if uz is not None:
            div = div + _grad3d(uz, 0)
        return div
    raise ValueError(f"ux must be 2-D or 3-D, got {ux.ndim}-D")


def compute_drag_lift_coefficients(
    fx: float | torch.Tensor,
    fy: float | torch.Tensor,
    u_in: float,
    rho_ref: float = 1.0,
    area: float = 1.0,
) -> tuple[float, float]:
    """Compute drag and lift coefficients from force data.

    .. math::

        C_D = \\frac{F_x}{\\tfrac{1}{2}\\rho_{ref}\\,U^2\\,A}, \\qquad
        C_L = \\frac{F_y}{\\tfrac{1}{2}\\rho_{ref}\\,U^2\\,A}

    Args:
        fx: Streamwise (drag) force in lattice units.  May be a scalar or a
            single-element :class:`torch.Tensor`.
        fy: Cross-stream (lift) force in lattice units.  Same convention
            as *fx*.
        u_in: Reference inlet velocity :math:`U`.
        rho_ref: Reference fluid density (default 1.0).
        area: Reference area (chord length in 2-D, projected area in 3-D;
              default 1.0).

    Returns:
        Tuple ``(Cd, Cl)`` — drag and lift coefficients as Python floats.
        Returns ``(0.0, 0.0)`` when *u_in* is zero.
    """
    dyn_pressure = 0.5 * rho_ref * u_in**2 * area
    if dyn_pressure == 0.0:
        return 0.0, 0.0
    fx_val = float(fx.item()) if isinstance(fx, torch.Tensor) else float(fx)
    fy_val = float(fy.item()) if isinstance(fy, torch.Tensor) else float(fy)
    return fx_val / dyn_pressure, fy_val / dyn_pressure


class RunningStats:
    """Online accumulator for time-averaged field statistics.

    Uses Welford's numerically stable one-pass algorithm to track the
    running mean and variance of a sequence of identically-shaped
    :class:`torch.Tensor` fields without storing the full time history.

    Typical usage::

        stats = RunningStats()
        for step in range(n_steps):
            ux, uy, _ = macroscopic(f)
            stats.update(ux)

        mean_ux = stats.mean
        rms_ux  = stats.variance.sqrt()
        fluct   = stats.fluctuation(ux)

    All internal state is stored as ``float32`` tensors on the same device
    as the first field passed to :meth:`update`.
    """

    def __init__(self) -> None:
        self._count: int = 0
        self._mean: torch.Tensor | None = None
        self._m2: torch.Tensor | None = None  # sum of squared deviations

    @property
    def count(self) -> int:
        """Number of samples accumulated so far."""
        return self._count

    def update(self, field: torch.Tensor) -> None:
        """Incorporate a new field snapshot into the running statistics.

        Args:
            field: Any-shape tensor.  All subsequent calls must use the same
                shape and device.
        """
        f = field.float()
        self._count += 1
        if self._mean is None:
            self._mean = f.clone()
            self._m2 = torch.zeros_like(f)
        else:
            delta = f - self._mean
            self._mean.add_(delta / self._count)
            delta2 = f - self._mean
            self._m2.add_(delta * delta2)

    @property
    def mean(self) -> torch.Tensor:
        """Time-averaged field.

        Raises:
            RuntimeError: If no data has been accumulated yet.
        """
        if self._mean is None:
            raise RuntimeError("RunningStats has no data yet; call update() first.")
        return self._mean

    @property
    def variance(self) -> torch.Tensor:
        """Sample variance field (Bessel-corrected, ddof=1).

        Returns a zero tensor when fewer than two samples have been added.

        Raises:
            RuntimeError: If no data has been accumulated yet.
        """
        if self._m2 is None:
            raise RuntimeError("RunningStats has no data yet; call update() first.")
        if self._count < 2:
            return torch.zeros_like(self._m2)
        return self._m2 / (self._count - 1)

    def fluctuation(self, field: torch.Tensor) -> torch.Tensor:
        """Return the fluctuation of *field* about the running mean.

        .. math::

            u'(t) = u(t) - \\langle u \\rangle

        Args:
            field: Snapshot tensor with the same shape as the accumulated
                fields.

        Returns:
            Fluctuation tensor of the same shape.

        Raises:
            RuntimeError: If no data has been accumulated yet.
        """
        return field.float() - self.mean

    def reset(self) -> None:
        """Reset all accumulated statistics."""
        self._count = 0
        self._mean = None
        self._m2 = None


def compute_strouhal_fft(
    fy_signal: torch.Tensor,
    sample_rate: float = 1.0,
    u_ref: float = 1.0,
    length_ref: float = 1.0,
) -> float:
    """Estimate the Strouhal number from a force signal using the FFT.

    Args:
        fy_signal: Force-history tensor of shape ``(N,)``.
        sample_rate: Samples per time unit.
        u_ref: Reference velocity.
        length_ref: Reference length.

    Returns:
        Estimated Strouhal number.
    """
    if fy_signal.numel() < 4 or u_ref == 0.0:
        return 0.0
    start = fy_signal.numel() // 4
    signal = fy_signal[start:].float()
    signal = signal - signal.mean()
    window = torch.hann_window(signal.numel(), device=signal.device, dtype=signal.dtype)
    spectrum = torch.fft.rfft(signal * window)
    amplitude = spectrum.abs()
    if amplitude.numel() <= 1:
        return 0.0
    peak_index = int(torch.argmax(amplitude[1:]).item()) + 1
    freq = torch.fft.rfftfreq(signal.numel(), d=1.0 / sample_rate, device=signal.device)[peak_index]
    return float((freq * length_ref / u_ref).item())


def compute_added_mass_2d(
    fx_history: torch.Tensor,
    fy_history: torch.Tensor,
    motion_history: torch.Tensor,
    omega: float,
    rho_ref: float = 1.0,
    area: float = 1.0,
) -> tuple[float, float]:
    """Estimate 2D added mass and damping from forced-oscillation data.

    Args:
        fx_history: In-line force history.
        fy_history: Cross-flow force history.
        motion_history: Prescribed displacement history.
        omega: Oscillation angular frequency.
        rho_ref: Reference density for optional normalization.
        area: Reference area for optional normalization.

    Returns:
        Tuple ``(added_mass, damping_coeff)``.
    """
    del fy_history
    x = motion_history.float()
    xdot = torch.gradient(x, spacing=1.0)[0]
    design = torch.stack([-omega**2 * x, -omega * xdot], dim=1)
    solution = torch.linalg.lstsq(design, fx_history.float().unsqueeze(1)).solution.squeeze(1)
    scale = rho_ref * area if rho_ref * area != 0.0 else 1.0
    return float(solution[0].item() / scale), float(solution[1].item() / scale)


def compute_added_mass_3d(
    fx_history: torch.Tensor,
    fy_history: torch.Tensor,
    fz_history: torch.Tensor,
    motion_x_history: torch.Tensor,
    omega: float,
    rho_ref: float = 1.0,
    volume: float = 1.0,
) -> tuple[float, float]:
    """Estimate 3D added mass and damping from forced-oscillation data.

    Args:
        fx_history: In-line force history.
        fy_history: Lateral force history.
        fz_history: Vertical force history.
        motion_x_history: Prescribed x-displacement history.
        omega: Oscillation angular frequency.
        rho_ref: Reference density for optional normalization.
        volume: Reference volume for optional normalization.

    Returns:
        Tuple ``(added_mass_x, damping_x)``.
    """
    del fy_history, fz_history
    x = motion_x_history.float()
    xdot = torch.gradient(x, spacing=1.0)[0]
    design = torch.stack([-omega**2 * x, -omega * xdot], dim=1)
    solution = torch.linalg.lstsq(design, fx_history.float().unsqueeze(1)).solution.squeeze(1)
    scale = rho_ref * volume if rho_ref * volume != 0.0 else 1.0
    return float(solution[0].item() / scale), float(solution[1].item() / scale)


__all__ = [
    "extract_velocity_profile",
    "extract_wake_profile",
    "compute_recirculation_length",
    "compute_pressure_coefficient",
    "compute_q_criterion",
    "compute_lambda2_criterion",
    "compute_vorticity_2d",
    "compute_vorticity_3d",
    "compute_velocity_magnitude",
    "compute_kinetic_energy",
    "compute_enstrophy_2d",
    "compute_divergence",
    "compute_drag_lift_coefficients",
    "RunningStats",
    "compute_strouhal_fft",
    "compute_added_mass_2d",
    "compute_added_mass_3d",
]
