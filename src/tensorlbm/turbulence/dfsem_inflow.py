"""Divergence-free synthetic inflow generators with machine-checkable quality gates.

Inlet-quality layer for the development-section and isotropic-decay experiments:

* :class:`DFSEMInlet` — Divergence-Free Synthetic Eddy Method.  Compactly
  supported eddies carry a stream function ``ψ``; the transverse velocity is its
  curl (``v' = ∂_z ψ``, ``w' = -∂_y ψ``, evaluated with central differences so
  the *discrete* in-plane divergence vanishes to round-off) and the streamwise
  component is an independent-sign eddy field.  A 3×3 linear map imposes the
  target Reynolds-stress tensor ``R_ij`` (including the shear components), and an
  a-posteriori spectral projection removes the compressive part this mixing
  reintroduces.  The residual divergence and the residual stress error are both
  measured; the trade-off between them is reported, not hidden.
* :func:`project_divergence_free` — a-posteriori Helmholtz projection on the
  periodic inlet plane.
* :class:`DigitalFilterInlet` — Klein et al. (2003) digital-filter method with a
  Lund et al. (1998) transformation layer that rescales the signal to ``R_ij``.

``projection=False`` on :class:`DFSEMInlet` is the documented negative control:
same target stresses, same eddy statistics, residual divergence left in.  The
downstream consequence (spurious pressure waves) is measured in
:mod:`tensorlbm.turbulence.inlet_development`.

References
----------
* Jarrin N. *et al.* (2006) Int. J. Heat Fluid Flow 27 585.
* Poletto R. *et al.* (2013) Flow Turbul. Combust. 91 519 (DFSEM).
* Klein M. *et al.* (2003) J. Comput. Phys. 186 652 (digital filter).
* Lund T.S. *et al.* (1998) J. Comput. Phys. 140 233 (rescaling).
* Comte-Bellot G., Corrsin S. (1971) J. Fluid Mech. 48 273 (decay exponent).

Relation to :mod:`tensorlbm.turbulence.synthetic_inflow`
-------------------------------------------------------
That module is the older, production-path generator exported at package level as
``tensorlbm.DFSEMInlet`` (uniform mean profile, normal stresses only, no
projection).  This module is the *inlet-quality* layer: full ``R_ij`` including
shears, the a-posteriori projection, the divergence diagnostics and the
digital-filter/Lund control.  It is imported by module path
(``from tensorlbm.turbulence.dfsem_inflow import ...``) so that the package-level
names keep their historical meaning.

Units: lengths in lattice cells, velocities in lattice units, Reynolds stresses
in lattice units squared.  The algebra is done in ``float64`` internally.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

__all__ = [
    "DFSEMInlet",
    "DigitalFilterInlet",
    "divergence_ratio",
    "divergence_rms",
    "discrete_divergence_rms",
    "lund_transform",
    "project_divergence_free",
    "sample_stress",
    "stress_cholesky",
    "stress_tensor",
]


# ---------------------------------------------------------------------------
# Reynolds-stress algebra
# ---------------------------------------------------------------------------


def stress_tensor(
    uu: float,
    vv: float,
    ww: float,
    uv: float = 0.0,
    uw: float = 0.0,
    vw: float = 0.0,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build the symmetric 3×3 Reynolds-stress tensor ``R_ij = ⟨u_i' u_j'⟩``."""
    R = torch.tensor([[uu, uv, uw], [uv, vv, vw], [uw, vw, ww]], dtype=dtype, device=device)
    return 0.5 * (R + R.transpose(0, 1))


def stress_cholesky(R: torch.Tensor, *, reg: float = 1e-14) -> torch.Tensor:
    """Lower Cholesky factor ``L`` with ``L Lᵀ = R`` for a positive-definite ``R``.

    A physically realisable Reynolds-stress tensor is positive semi-definite; a
    generator needs it positive *definite* to be invertible.  Marginally PD
    tensors are regularised by adding ``shift * I``; the shift actually used is
    recorded in the module-level dict ``STRESS_CHOLESKY_AUDIT``.

    Raises:
        ValueError: if ``R`` is not a realisable Reynolds-stress tensor.
    """
    R = 0.5 * (R + R.transpose(0, 1))
    if R.shape != (3, 3):
        raise ValueError(f"R must be 3x3, got {tuple(R.shape)}")
    dtype, device = R.dtype, R.device
    evals = torch.linalg.eigvalsh(R.to(torch.float64))
    scale = float(torch.clamp(torch.abs(evals).max(), min=1e-30))
    if float(evals.min()) < -1e-10 * scale:
        raise ValueError(
            "Reynolds-stress tensor is not positive semi-definite "
            f"(min eigenvalue {float(evals.min()):.3e}, scale {scale:.3e})"
        )
    shift = 0.0
    for _ in range(20):
        try:
            L = torch.linalg.cholesky(
                R.to(torch.float64) + shift * torch.eye(3, dtype=torch.float64, device=device)
            )
            STRESS_CHOLESKY_AUDIT["last_regularisation"] = shift
            return L.to(dtype)
        except RuntimeError:  # pragma: no cover - degenerate input only
            shift = max(reg, (shift or reg) * 10.0)
    raise ValueError("Reynolds-stress tensor is numerically singular")  # pragma: no cover


STRESS_CHOLESKY_AUDIT: dict[str, float] = {"last_regularisation": 0.0}


def _inv_sqrt_psd(C: torch.Tensor, *, floor: float = 1e-300) -> torch.Tensor:
    """Symmetric ``C^{-1/2}`` of a positive semi-definite 3×3 matrix."""
    C = 0.5 * (C + C.transpose(0, 1))
    evals, evecs = torch.linalg.eigh(C.to(torch.float64))
    evals = torch.clamp(evals, min=floor)
    return ((evecs * evals.rsqrt()) @ evecs.transpose(0, 1)).to(C.dtype)


def sample_stress(u: torch.Tensor, *, mean_subtract: bool = True) -> torch.Tensor:
    """Sample Reynolds-stress tensor of a velocity field ``(3, ...)``."""
    if u.ndim < 2 or u.shape[0] != 3:
        raise ValueError(f"u must have shape (3, ...), got {tuple(u.shape)}")
    flat = u.reshape(3, -1)
    if mean_subtract:
        flat = flat - flat.mean(dim=1, keepdim=True)
    return (flat @ flat.transpose(0, 1)) / float(flat.shape[1])


# ---------------------------------------------------------------------------
# Divergence diagnostics and a-posteriori projection
# ---------------------------------------------------------------------------


def _wavenumbers(n: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Spectral wavenumbers ``2π k / N`` with the transverse Nyquist mode de-aliased.

    The ``k = N/2`` mode is ``(-1)^j``: it is *real* on the lattice, so the
    transverse derivative that represents ``∂/∂y`` there (central difference,
    symbol ``sin k``) vanishes, and ``k = +N/2`` cannot be distinguished from
    ``-N/2``.  Keeping the raw value ``π`` makes the Helmholtz correction
    non-Hermitian on that plane (the cross term ``k_y k_z`` flips sign), which
    leaves an ``O(10 %)`` residual divergence that no projection can remove.
    Setting the Nyquist wavenumber to zero -- the usual de-aliasing convention --
    makes the symbol consistent with the lattice and the projection exact with
    respect to the divergence that is actually measured.
    """
    k = 2.0 * math.pi * torch.fft.fftfreq(n, d=1.0, device=device).to(dtype)
    if n % 2 == 0 and n > 2:
        k[n // 2] = 0.0
    return k


def _fft_setup(u: torch.Tensor) -> tuple[torch.dtype, torch.Tensor, torch.Tensor]:
    dtype = u.dtype if u.dtype in (torch.float32, torch.float64) else torch.float64
    ny, nz = u.shape[-2], u.shape[-1]
    ky = _wavenumbers(ny, device=u.device, dtype=dtype).view(-1, 1)
    kz = _wavenumbers(nz, device=u.device, dtype=dtype).view(1, -1)
    return dtype, ky, kz[:, : nz // 2 + 1]


def divergence_rms(u: torch.Tensor) -> torch.Tensor:
    """RMS of the spectral in-plane divergence ``∂v'/∂y + ∂w'/∂z``.

    This is the quantity that acts as the acoustic source term of the pressure
    equation, so it is the right measure of inlet "compressibility" even though
    LBM evaluates derivatives on a stencil.
    """
    dtype, ky, kz = _fft_setup(u)
    work = u.to(dtype)
    ny, nz = work.shape[-2], work.shape[-1]
    vh = torch.fft.rfft2(work[..., 1, :, :])
    wh = torch.fft.rfft2(work[..., 2, :, :])
    div = torch.fft.irfft2(ky * vh + kz * wh, s=(ny, nz))
    return torch.sqrt((div**2).mean())


def discrete_divergence_rms(u: torch.Tensor) -> torch.Tensor:
    """RMS of the central-difference in-plane divergence (LBM-stencil form)."""
    work = u.to(torch.float64)
    v, w = work[..., 1, :, :], work[..., 2, :, :]
    dv = 0.5 * (torch.roll(v, -1, dims=-2) - torch.roll(v, 1, dims=-2))
    dw = 0.5 * (torch.roll(w, -1, dims=-1) - torch.roll(w, 1, dims=-1))
    return torch.sqrt(((dv + dw) ** 2).mean())


def _gradient_rms(u: torch.Tensor) -> torch.Tensor:
    """RMS of the diagonal strain ``[(∂v'/∂y)² + (∂w'/∂z)²]^{1/2}``."""
    dtype, ky, kz = _fft_setup(u)
    work = u.to(dtype)
    ny, nz = work.shape[-2], work.shape[-1]
    vy = torch.fft.irfft2(ky * torch.fft.rfft2(work[..., 1, :, :]), s=(ny, nz))
    wz = torch.fft.irfft2(kz * torch.fft.rfft2(work[..., 2, :, :]), s=(ny, nz))
    return torch.sqrt((vy**2 + wz**2).mean())


def divergence_ratio(u: torch.Tensor) -> torch.Tensor:
    """Scale-free divergence diagnostic ``‖∇·u'‖_rms / ‖∂u_i/∂x_i‖_rms``.

    ``≈ 0`` for a divergence-free inlet, ``O(1)`` for an inlet whose transverse
    fluctuations carry a compressive part.  Because the amplitude cancels, a
    projected and an unprojected inlet carrying the same Reynolds stresses are
    directly comparable.
    """
    denom = _gradient_rms(u)
    if float(denom) <= 0.0:  # pragma: no cover - degenerate input
        return torch.zeros((), dtype=torch.float64, device=u.device)
    return (divergence_rms(u) / denom).to(torch.float64)


def project_divergence_free(u: torch.Tensor) -> torch.Tensor:
    """A-posteriori transverse Helmholtz projection on the inlet plane.

    Removes the compressive part of ``(v', w')`` with the exact spectral symbol

    .. math::

        \\hat{u}_t \\leftarrow \\hat{u}_t
            - \\hat{k}_t \\frac{\\hat{k}_t \\cdot \\hat{u}_t}{|k|^2},

    leaving the streamwise component untouched (it does not enter the in-plane
    divergence under frozen advection).  Works plane-by-plane, so batched arrays
    with leading axes are supported.

    A *full* complex transform (``fft2``/``ifft2``) is used on purpose: with a
    one-sided ``rfft2`` the self-conjugate part of the transverse Nyquist column
    (``k_z = π``) is silently discarded by ``irfft2``, so the correction cannot
    be represented there and the projected white-noise field keeps
    ``‖∇·u'‖_rms / ‖∂_t u_t‖_rms ≈ 0.10`` instead of round-off.  The full
    transform keeps the residual at the round-off level of the working dtype.

    Returns:
        Projected field with the same shape and dtype.
    """
    dtype = u.dtype if u.dtype in (torch.float32, torch.float64) else torch.float64
    work = u.to(dtype)
    ny, nz = work.shape[-2], work.shape[-1]
    ky = _wavenumbers(ny, device=work.device, dtype=dtype).view(-1, 1)
    kz = _wavenumbers(nz, device=work.device, dtype=dtype).view(1, -1)
    kk = ky**2 + kz**2
    safe = torch.where(kk > 0.0, kk, torch.ones_like(kk))
    trans = torch.stack([work[..., 1, :, :], work[..., 2, :, :]], dim=-3)
    tw_h = torch.fft.fft2(trans)
    div = ky * tw_h[..., 0, :, :] + kz * tw_h[..., 1, :, :]
    corr = torch.where(kk > 0.0, div / safe, torch.zeros_like(div))
    tw_new = torch.stack(
        [tw_h[..., 0, :, :] - ky * corr, tw_h[..., 1, :, :] - kz * corr], dim=-3
    )
    new = torch.fft.ifft2(tw_new).real
    out = work.clone()
    out[..., 1, :, :] = new[..., 0, :, :]
    out[..., 2, :, :] = new[..., 1, :, :]
    return out.to(u.dtype)


# ---------------------------------------------------------------------------
# Divergence-Free Synthetic Eddy Method
# ---------------------------------------------------------------------------


@dataclass
class DFSEMInlet:
    """DFSEM inlet: divergence-free eddy kernels + a-posteriori divergence projection.

    Per-snapshot pipeline:

    1. **Eddy superposition.** ``n_eddies`` compactly supported eddies with
       random signs ``σ_k`` define a stream function ``ψ = Σ_k σ_k φ_k`` and a
       streamwise field ``U = Σ_k τ_k φ_k`` (independent signs ``τ_k``); the
       fluctuation field is ``(U, ∂_z ψ, -∂_y ψ)``.  ``φ_k`` is the C² bump
       ``(1 - r²)³`` on ``|r| < σ``.  The transverse pair uses the *discrete
       central-difference* curl, so ``D_y v' + D_z w'`` vanishes to round-off
       (spurious pressure is driven by the divergence, not by the discretisation).
       Positions are periodic in ``(y, z)`` and advected streamwise at ``u_conv``
       through a virtual box of length ``2σ``: that is what makes successive
       snapshots temporally correlated, with correlation time ``≈ σ/u_conv``.
    2. **Stress imposition.** ``A = L_R C^{-1/2}`` with ``L_R = chol(R)`` and
       ``C`` the raw-field covariance gives ``⟨u_i u_j⟩ = R_ij`` for the full
       tensor including shears.  ``C`` is estimated on a calibration window
       (``stress_correction="calibration"``) or per snapshot
       (``"per_snapshot"``, a Lund-type layer).
    3. **A-posteriori projection.** Step 2 mixes the streamwise component into
       the transverse ones, which reintroduces a compressive part;
       :func:`project_divergence_free` removes it.  The projection also removes
       transverse energy, so a stress-restoration map follows and the pair
       (project → restore) is iterated ``n_stress_iterations`` times, the last
       step being a projection.  Both residuals are recorded in
       :attr:`diagnostics`, quantifying the **Pareto front** between
       ``‖∇·u'‖`` and the stress error: a homogeneous in-plane field with
       reflection symmetry has a rank-one transverse covariance per wavenumber,
       so an arbitrary ``(R_vv, R_ww, R_vw)`` and *exactly* zero divergence
       cannot both be realised — reporting both is the honest statement.

    ``projection=False`` disables step 3 (negative control).

    Args:
        ny, nz: Inlet plane size in cells.
        stress: Target ``R_ij`` — 3×3 tensor or ``(uu, vv, ww, uv, uw, vw)``.
        length_scale: Eddy length scale ``σ`` in lattice units.
        n_eddies: Number of eddies.
        seed: RNG seed.
        projection: Run the a-posteriori divergence projection.
        divergence_free: ``True`` builds the transverse pair as the curl of a
            stream function (DFSEM).  ``False`` builds the classical synthetic
            eddy method (SEM, Jarrin 2006): independent scalar eddy kernels per
            velocity component with amplitudes drawn from ``N(0, R)``.  SEM
            reproduces ``R_ij`` but its fluctuations are *not* divergence-free —
            this is the negative control for the spurious-pressure comparison.
        stress_correction: ``"per_snapshot"`` or ``"calibration"``.
        n_stress_iterations: Project → restore iterations.
        u_conv: Streamwise eddy advection velocity.
        n_calibration: Snapshots used for the calibration covariance.
        device: Torch device.
    """

    ny: int
    nz: int
    stress: torch.Tensor | tuple[float, float, float, float, float, float]
    length_scale: float = 6.0
    n_eddies: int = 600
    seed: int = 0
    projection: bool = True
    divergence_free: bool = True
    stress_correction: str = "per_snapshot"
    n_stress_iterations: int = 1
    u_conv: float = 0.05
    n_calibration: int = 24
    device: torch.device | str = "cpu"
    dtype: torch.dtype = torch.float32

    _R: torch.Tensor = field(init=False, repr=False)
    _L: torch.Tensor = field(init=False, repr=False)
    _A: torch.Tensor = field(init=False, repr=False)
    _pos: torch.Tensor = field(init=False, repr=False)
    _sign: torch.Tensor = field(init=False, repr=False)
    _stream_sign: torch.Tensor = field(init=False, repr=False)
    _sem_amp: torch.Tensor = field(init=False, repr=False)
    _rng: torch.Generator = field(init=False, repr=False)
    diagnostics: list[dict] = field(init=False, default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if self.ny < 4 or self.nz < 4:
            raise ValueError("inlet plane must be at least 4x4 cells")
        if self.length_scale <= 1.0:
            raise ValueError("length_scale must exceed one lattice cell")
        if self.stress_correction not in ("per_snapshot", "calibration"):
            raise ValueError("stress_correction must be 'per_snapshot' or 'calibration'")
        if int(self.n_stress_iterations) < 0:
            raise ValueError("n_stress_iterations must be non-negative")
        self.device = torch.device(self.device)
        if isinstance(self.stress, (tuple, list)):
            if len(self.stress) != 6:
                raise ValueError("stress tuple must be (uu, vv, ww, uv, uw, vw)")
            self._R = stress_tensor(*self.stress, dtype=torch.float64, device=self.device)
        else:
            self._R = self.stress.to(device=self.device, dtype=torch.float64)
        self._L = stress_cholesky(self._R)
        self._ygrid = torch.arange(self.ny, device=self.device, dtype=torch.float64).view(-1, 1)
        self._zgrid = torch.arange(self.nz, device=self.device, dtype=torch.float64).view(1, -1)
        self._rng = torch.Generator(device=self.device)
        self._seed_eddies(int(self.seed))
        self._A = self._calibrate()

    # -- eddy bookkeeping --------------------------------------------------

    def _seed_eddies(self, seed: int) -> None:
        self._rng.manual_seed(int(seed))
        n = int(self.n_eddies)
        sigma = float(self.length_scale)
        self._pos = torch.stack(
            [
                (torch.rand(n, generator=self._rng, device=self.device, dtype=torch.float64) * 2 - 1)
                * sigma,
                torch.rand(n, generator=self._rng, device=self.device, dtype=torch.float64) * self.ny,
                torch.rand(n, generator=self._rng, device=self.device, dtype=torch.float64) * self.nz,
            ]
        )
        self._sign = (
            torch.randint(0, 2, (n,), generator=self._rng, device=self.device) * 2 - 1
        ).to(torch.float64)
        self._stream_sign = (
            torch.randint(0, 2, (n,), generator=self._rng, device=self.device) * 2 - 1
        ).to(torch.float64)
        eps = torch.randn((n, 3), generator=self._rng, device=self.device, dtype=torch.float64)
        # fixed per-eddy amplitude vectors with covariance R (classical SEM)
        self._sem_amp = (eps @ self._L.to(torch.float64).transpose(0, 1)).to(torch.float64)

    def _advance(self, dt: float = 1.0) -> None:
        """Advect eddies streamwise and recycle them periodically."""
        sigma = float(self.length_scale)
        self._pos[0] -= float(self.u_conv) * dt
        self._pos[0] = torch.remainder(self._pos[0] + sigma, 2.0 * sigma) - sigma

    # -- raw eddy field ----------------------------------------------------

    def _raw_field(self) -> torch.Tensor:
        """Raw eddy field ``(3, ny, nz)``: divergence-free (DFSEM) or SEM."""
        if not self.divergence_free:
            return self._raw_field_sem()
        sigma = float(self.length_scale)
        dx = self._pos[0].view(-1, 1, 1)
        dy_raw = self._ygrid.unsqueeze(0) - self._pos[1].view(-1, 1, 1)
        dz_raw = self._zgrid.unsqueeze(0) - self._pos[2].view(-1, 1, 1)
        dy = dy_raw - self.ny * torch.round(dy_raw / self.ny)
        dz = dz_raw - self.nz * torch.round(dz_raw / self.nz)
        r2 = ((dx / sigma) ** 2 + (dy / sigma) ** 2 + (dz / sigma) ** 2).clamp(max=1.0)
        env = (1.0 - r2) ** 3  # C2 compactly supported bump
        psi = (self._sign.view(-1, 1, 1) * env).sum(0)
        u_field = (self._stream_sign.view(-1, 1, 1) * env).sum(0)
        v_field = 0.5 * (torch.roll(psi, -1, dims=1) - torch.roll(psi, 1, dims=1))
        w_field = -0.5 * (torch.roll(psi, -1, dims=0) - torch.roll(psi, 1, dims=0))
        return torch.stack([u_field, v_field, w_field])

    def _raw_field_sem(self) -> torch.Tensor:
        """Classical synthetic eddy method: independent scalar kernels per component.

        ``u_i(x) = Σ_k a_{k,i} φ_k(x)`` with ``a_k ~ N(0, R)`` fixed per eddy.
        The Reynolds stresses are reproduced, but ``∂v'/∂y + ∂w'/∂z`` does not
        vanish: neighbouring eddies do not share a stream function.  This is the
        documented negative control for the spurious-pressure experiment.
        """
        sigma = float(self.length_scale)
        dx = self._pos[0].view(-1, 1, 1)
        dy_raw = self._ygrid.unsqueeze(0) - self._pos[1].view(-1, 1, 1)
        dz_raw = self._zgrid.unsqueeze(0) - self._pos[2].view(-1, 1, 1)
        dy = dy_raw - self.ny * torch.round(dy_raw / self.ny)
        dz = dz_raw - self.nz * torch.round(dz_raw / self.nz)
        r2 = ((dx / sigma) ** 2 + (dy / sigma) ** 2 + (dz / sigma) ** 2).clamp(max=1.0)
        env = (1.0 - r2) ** 3  # (N, ny, nz)
        flat = env.reshape(env.shape[0], -1)  # (N, ny*nz)
        u = torch.empty((3, self.ny, self.nz), dtype=torch.float64, device=self.device)
        for i in range(3):
            u[i] = (self._sem_amp[:, i].view(-1, 1) * flat).sum(0).reshape(self.ny, self.nz)
        return u

    # -- stress imposition -------------------------------------------------

    def _target_map(self, field: torch.Tensor) -> torch.Tensor:
        """``L_R C^{-1/2}``: map the raw field onto the target covariance."""
        C = sample_stress(field)
        return (self._L @ _inv_sqrt_psd(C)).to(field.dtype)

    def _calibrate(self) -> torch.Tensor:
        """Time-averaged stress map estimated on a short calibration window."""
        saved = self._pos.clone()
        snaps = []
        for _ in range(max(2, int(self.n_calibration))):
            self._advance()
            snaps.append(self._raw_field())
        self._pos = saved
        stacked = torch.cat(snaps, dim=2)
        return self._target_map(stacked)

    # -- snapshot ----------------------------------------------------------

    def _snapshot(self) -> tuple[torch.Tensor, dict]:
        q = self._raw_field()
        A = self._target_map(q) if self.stress_correction == "per_snapshot" else self._A
        u = (A.to(torch.float64) @ q.reshape(3, -1)).reshape(3, self.ny, self.nz)
        info: dict[str, float] = {"div_ratio_raw": float(divergence_ratio(u.float()))}
        if self.projection:
            for _ in range(int(self.n_stress_iterations)):
                u = project_divergence_free(u.float()).to(torch.float64)
                u = (self._target_map(u) @ u.reshape(3, -1)).reshape(3, self.ny, self.nz)
            u = project_divergence_free(u.float()).to(torch.float64)
            info["div_ratio_discrete"] = float(discrete_divergence_rms(u.float()))
        info["div_ratio"] = float(divergence_ratio(u.float()))
        info["stress_error"] = float(
            ((sample_stress(u) - self._R) / self._R[0, 0]).abs().amax()
        )
        if len(self.diagnostics) < 64:
            self.diagnostics.append(info)
        return u.to(self.dtype), info

    # -- public API --------------------------------------------------------

    def sample(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance one step; return ``(u', v', w')`` each of shape ``(ny, nz)``."""
        self._advance()
        u, _ = self._snapshot()
        return u[0], u[1], u[2]

    def sequence(self, n_steps: int, *, dt: float = 1.0) -> torch.Tensor:
        """Return ``n_steps`` snapshots with shape ``(n_steps, 3, ny, nz)``."""
        out = torch.empty((n_steps, 3, self.ny, self.nz), dtype=self.dtype, device=self.device)
        for i in range(n_steps):
            self._advance(dt)
            u, _ = self._snapshot()
            out[i] = u
        return out

    def reset(self, seed: int | None = None) -> None:
        """Reset eddy positions (and optionally the seed)."""
        if seed is not None:
            self.seed = int(seed)
        self._seed_eddies(self.seed)
        self.diagnostics.clear()


# ---------------------------------------------------------------------------
# Digital-filter method (Klein) with Lund transformation
# ---------------------------------------------------------------------------


def _gaussian_filter_1d(x: torch.Tensor, sigma: float, axis: int) -> torch.Tensor:
    """Separable Gaussian filter of width ``sigma`` along ``axis`` (replicate pad)."""
    half = max(1, int(math.ceil(2.0 * sigma)))
    r = torch.arange(-half, half + 1, dtype=x.dtype, device=x.device)
    g = torch.exp(-math.pi * (r / sigma) ** 2)
    g = g / g.sum()
    xm = x.movedim(axis, -1)
    shape = xm.shape
    flat = xm.reshape(-1, 1, shape[-1])
    padded = torch.nn.functional.pad(flat, (half, half), mode="replicate")
    out = torch.nn.functional.conv1d(padded, g.view(1, 1, -1))
    return out.reshape(shape).movedim(-1, axis)


def lund_transform(
    u: torch.Tensor, stress: torch.Tensor, *, mean_subtract: bool = True
) -> torch.Tensor:
    """Lund et al. (1998) scaling layer: ``u ← chol(R) chol(C)^{-1/2} u``.

    ``C`` is the sample covariance over all trailing axes, so the transformed
    sample covariance equals ``R`` to round-off; the sample mean is removed when
    requested (Lund's transformation also covers the mean profile, which is
    irrelevant for a fluctuation field).
    """
    flat = u.reshape(3, -1)
    if mean_subtract:
        flat = flat - flat.mean(dim=1, keepdim=True)
    C = (flat @ flat.transpose(0, 1)) / float(flat.shape[1])
    A = (stress_cholesky(stress) @ _inv_sqrt_psd(C)).to(u.dtype)
    return (A @ flat).reshape(u.shape)


@dataclass
class DigitalFilterInlet:
    """Klein et al. (2003) digital-filter inlet with a Lund transformation layer.

    Independent Gaussian noise is filtered along the wall-normal, spanwise and
    time axes with a separable Gaussian kernel; the filter width sets the
    integral scale (``L_int = σ/√2`` for this kernel).  A :func:`lund_transform`
    layer then imposes the target Reynolds stresses and removes the sample mean.

    The block is generated offline (how the classical DFM is used); a long block
    avoids the start-up transient of recursive formulations.

    Args:
        ny, nz: Inlet plane size.
        stress: Target ``R_ij`` (3×3 tensor or 6-tuple).
        length_scale: Spatial filter width in cells.
        time_scale: Temporal filter width in samples.
        n_steps: Default block length.
        apply_lund: Apply the Lund layer (``False`` = raw DFM, negative control).
        seed, device, dtype: As for :class:`DFSEMInlet`.
    """

    ny: int
    nz: int
    stress: torch.Tensor | tuple[float, float, float, float, float, float]
    length_scale: float = 6.0
    time_scale: float = 8.0
    n_steps: int = 512
    apply_lund: bool = True
    seed: int = 0
    device: torch.device | str = "cpu"
    dtype: torch.dtype = torch.float32

    _R: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        if isinstance(self.stress, (tuple, list)):
            self._R = stress_tensor(*self.stress, dtype=torch.float64, device=self.device)
        else:
            self._R = self.stress.to(device=self.device, dtype=torch.float64)

    def sequence(self, n_steps: int | None = None) -> torch.Tensor:
        """Generate a block of snapshots with shape ``(n_steps, 3, ny, nz)``.

        The Lund transformation is applied to the ``(3, T, ny, nz)`` block (where
        the trailing axes are the samples); the returned tensor is transposed to
        the generator-wide ``(time, component, y, z)`` layout.
        """
        n_steps = int(self.n_steps if n_steps is None else n_steps)
        gen = torch.Generator(device=self.device)
        gen.manual_seed(int(self.seed))
        half_s = max(1, int(math.ceil(2.0 * self.length_scale)))
        half_t = max(1, int(math.ceil(2.0 * self.time_scale)))
        raw = torch.randn(
            (3, n_steps + 2 * half_t, self.ny + 2 * half_s, self.nz + 2 * half_s),
            generator=gen,
            device=self.device,
            dtype=self.dtype,
        )
        f = _gaussian_filter_1d(raw, float(self.time_scale), axis=1)
        f = _gaussian_filter_1d(f, float(self.length_scale), axis=2)
        f = _gaussian_filter_1d(f, float(self.length_scale), axis=3)
        f = f[:, half_t : half_t + n_steps, half_s : half_s + self.ny, half_s : half_s + self.nz]
        if self.apply_lund:
            f = lund_transform(f, self._R.to(self.dtype))
        # (3, T, ny, nz) -> (T, 3, ny, nz): same layout as DFSEMInlet.sequence
        return f.transpose(0, 1).contiguous()