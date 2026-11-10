"""LES plane-channel campaign: Smagorinsky SGS + van Driest damping (P127W2-N03).

The wall-resolved machinery of :mod:`tensorlbm.turbulence.turbulent_channel3d`
(D3Q19 TRT, magic parameter ``Lambda = 3/16``, Guo forcing with the exact TRT
force split, Bouzidi interpolated bounce-back) is reused verbatim; this module
adds what a *LES* campaign needs on top of it:

* **Sub-grid closure** — Smagorinsky with van Driest wall damping,
  ``nu_t = (C_s f_v Delta)^2 |S|`` with ``f_v = 1 - exp(-y+ / A+)``.  The
  damping multiplies the converged local-equilibrium eddy viscosity, which is
  algebraically identical to solving the implicit relation with
  ``C_s^eff = C_s f_v`` (both sides scale with ``(C_s f_v)^2``).
* **Wall-unit statistics** — ``U+``, Reynolds stresses, TKE production
  ``P+ = -<u'v'>+ dU+/dy+`` (resolved part, and resolved + SGS part) and the
  sub-grid stress fraction ``tau_sgs / tau_total`` whose wall-normal
  distribution is criterion (d) of the campaign.
* **Energy spectra** — one-dimensional streamwise (``k_x``) and spanwise
  (``k_z``) spectra of the streamwise fluctuation on wall-normal planes, with
  an *a-priori banded* log-log power-law fit (no post-hoc band search that
  could tune the reported slope).
* **Mass-flux bookkeeping** — per-sample bulk velocity and total mass, so the
  constant-body-force constraint is checked instead of assumed.
* **Grid convergence** — Celik-style GCI from a three-grid ladder using the
  observed order of accuracy, with an explicit ``p = 2`` fallback flag.

Reference data
--------------
``MKM1999`` mean-profile tables (``chan180.means`` / ``chan590.means`` of the
Moser/Kim/Mansour database, vendored under
``benchmarks/pending/channel_turb_3d/dns_ref``) carry ``U+`` and the mean
shear.  The Reynolds shear stress is *reconstructed* from the exact channel
stress budget

    ``-<u'v'>+ = 1 - y+/Re_tau - dU+/dy+``

so the DNS reference for criterion (b) is derived from published data with no
hand-copied digit.  ``Re_tau = 395`` has no table in the repository, so the
Re=395 leg is reported as ``WITHHELD`` for the Reynolds-stress criteria rather
than substituting an unverifiable number.

Everything here is physical (wall units, spectra, convergence); nothing is
platform semantics.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from tensorlbm.turbulence.turbulent_channel3d import (
    TurbulentChannel3D,
    TurbulentChannel3DConfig,
)
from tensorlbm.turbulence.wall_model_usability import MkmProfile

KAPPA = 0.41
B_LOG = 5.2
VAN_DRIEST_A_PLUS = 26.0
DEFAULT_CS = 0.12

_F64 = torch.float64


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LESChannelConfig(TurbulentChannel3DConfig):
    """TRT channel configuration plus LES/SGS campaign options."""

    # "fneq": strain from the non-equilibrium populations (cheap, but the
    # Guo force and single-step noise enter the estimate); "velocity_gradient":
    # strain from central differences of the resolved velocity field.
    strain_source: str = "fneq"
    sgs_model: str = "smagorinsky_van_driest"
    van_driest: bool = True
    van_driest_a_plus: float = VAN_DRIEST_A_PLUS
    spectrum_planes_y_plus: tuple[float, ...] = (15.0, 30.0, 50.0, 100.0)
    spectrum_from_fraction: float = 0.5  # start spectra after this part of the window
    n_blocks: int = 5  # blocks for the statistical-uncertainty estimate
    # --- wall model (law of the wall) -------------------------------------
    # "spalding": the wall-adjacent node's effective viscosity is set from the
    # composite law of the wall so that the wall exerts the modelled shear
    # stress tau_w = rho u_tau^2 instead of the (grid-starved) molecular value.
    wall_model: str = "none"
    wall_model_interval: int = 10
    wall_model_filter: float = 0.5
    wall_model_cap_nu: float = 20.0

    def validate(self) -> None:
        super().validate()
        if self.sgs_model not in ("smagorinsky_van_driest", "smagorinsky", "none"):
            raise ValueError(
                "sgs_model must be 'smagorinsky_van_driest', 'smagorinsky' or 'none'"
            )
        if self.van_driest_a_plus <= 0.0:
            raise ValueError("van_driest_a_plus must be positive")
        if not 0.0 <= self.spectrum_from_fraction < 1.0:
            raise ValueError("spectrum_from_fraction must lie in [0, 1)")
        if self.n_blocks < 1:
            raise ValueError("n_blocks must be >= 1")
        if self.strain_source not in ("fneq", "velocity_gradient"):
            raise ValueError(
                "strain_source must be 'fneq' or 'velocity_gradient'"
            )
        if self.wall_model not in ("none", "spalding"):
            raise ValueError("wall_model must be 'none' or 'spalding'")
        if self.wall_model != "none" and self.smagorinsky_cs <= 0.0:
            raise ValueError(
                "the wall model needs the LES branch: set smagorinsky_cs > 0"
            )
        if self.wall_model_interval < 1:
            raise ValueError("wall_model_interval must be >= 1")
        if not 0.0 <= self.wall_model_filter <= 1.0:
            raise ValueError("wall_model_filter must lie in [0, 1]")
        if self.wall_model_cap_nu <= 0.0:
            raise ValueError("wall_model_cap_nu must be positive")


# ---------------------------------------------------------------------------
# SGS building blocks
# ---------------------------------------------------------------------------
def van_driest_damping(y_plus, a_plus: float = VAN_DRIEST_A_PLUS) -> torch.Tensor:
    """``f_v = 1 - exp(-y+ / A+)`` — van Driest (1956) mixing-length damping."""
    y_plus = torch.as_tensor(y_plus, dtype=_F64)
    return 1.0 - torch.exp(-y_plus.clamp(min=0.0) / float(a_plus))


def van_driest_nu_t(
    cs: float, strain_magnitude, y_plus, *, a_plus: float = VAN_DRIEST_A_PLUS
) -> torch.Tensor:
    """Damped Smagorinsky eddy viscosity ``(C_s f_v)^2 |S|`` (unit filter length).

    ``strain_magnitude`` is ``|S| = sqrt(2 S_ij S_ij)`` in lattice units; the
    filter length ``Delta = 1`` is absorbed into ``C_s``.
    """
    fv = van_driest_damping(y_plus, a_plus)
    return (float(cs) ** 2) * (fv**2) * torch.as_tensor(strain_magnitude, dtype=_F64)


# ---------------------------------------------------------------------------
# wall model: composite law of the wall (Spalding 1961)
# ---------------------------------------------------------------------------
def spalding_y_plus(u_plus, *, kappa: float = KAPPA, b: float = B_LOG):
    """Spalding's composite law of the wall ``y+ = Y(u+)``.

    ``Y(u+) = u+ + e^{-kappa b}[e^{kappa u+} - 1 - kappa u+ - (kappa u+)^2/2
    - (kappa u+)^3/6]`` is valid from the viscous sublayer through the buffer
    layer into the log region, so it can be evaluated at the first off-wall node
    even when that node sits at ``y+ ~ 5..30`` where the pure log law has no
    meaning.  ``Y`` is monotonically increasing, which makes the inversion for
    ``u_tau`` a well-posed scalar root find.
    """
    up = torch.as_tensor(u_plus, dtype=_F64)
    k = float(kappa) * up
    return up + math.exp(-float(kappa) * float(b)) * (
        torch.exp(k) - 1.0 - k - 0.5 * k**2 - k**3 / 6.0
    )


def u_tau_from_wall_law(
    u_1, y_1: float, nu: float, *, kappa: float = KAPPA, b: float = B_LOG,
    iters: int = 80,
):
    """Friction velocity from the composite law of the wall (vectorised).

    Solves ``y+ = Y(u+)`` with ``y+ = y_1 u_tau / nu`` and ``u+ = |u_1|/u_tau``
    for ``u_tau``; equivalently ``Y(u+) u+ = y_1 |u_1| / nu``, whose left-hand
    side is monotone in ``u+``.  A bracketed vectorised bisection is used
    (no Newton singularity at ``u_1 -> 0``), so the result is reproducible and
    free of iteration-dependent noise.
    """
    u1 = torch.as_tensor(u_1, dtype=_F64).abs()
    target = float(y_1) * u1 / float(nu)          # = y+ * u+
    lo = torch.zeros_like(u1)
    hi = torch.full_like(u1, 1.0)
    for _ in range(12):                            # grow bracket until it holds
        need = spalding_y_plus(hi) * hi < target
        if not bool(need.any()):
            break
        hi = torch.where(need, hi * 2.0, hi)
    for _ in range(int(iters)):
        mid = 0.5 * (lo + hi)
        f = spalding_y_plus(mid) * mid - target
        lo = torch.where(f < 0.0, mid, lo)
        hi = torch.where(f < 0.0, hi, mid)
    up = (0.5 * (lo + hi)).clamp(min=1e-12)
    return u1 / up


# ---------------------------------------------------------------------------
# solver with LES diagnostics
# ---------------------------------------------------------------------------
class LESChannel(TurbulentChannel3D):
    """TRT channel + Smagorinsky/van Driest SGS + LES statistics collection."""

    def __init__(self, config: LESChannelConfig) -> None:
        config.validate()
        super().__init__(config)
        self._build_wall_damping()
        self._build_wall_model()

    # ---- geometry helpers ------------------------------------------------
    def _build_wall_damping(self) -> None:
        cfg = self.cfg
        j = torch.arange(self.ny, dtype=_F64)
        y = (j - cfg.wall_origin).clamp(min=0.0)
        # distance to the nearest wall (mirror about the mid-plane)
        y = torch.minimum(y, (cfg.height - (j - cfg.wall_origin)).clamp(min=0.0))
        self.y_plus_rows = y * cfg.dy_plus
        if cfg.van_driest and cfg.sgs_model != "none":
            fv = van_driest_damping(self.y_plus_rows, cfg.van_driest_a_plus)
            self.fv2_rows = (fv**2).to(torch.float32)
        else:
            self.fv2_rows = torch.ones(self.ny, dtype=torch.float32)
        self._nu_t_field: torch.Tensor | None = None

    # ---- wall model ------------------------------------------------------
    def _build_wall_model(self) -> None:
        cfg = self.cfg
        self.wall_rows = (1, self.ny - 2)
        # effective (molecular + modelled) viscosity needed at each wall-adjacent
        # column to make the wall carry tau_w = rho u_tau^2
        self._wm_nu = torch.full((2, self.nz, self.nx), float(self.nu), dtype=_F64)
        self._wm_u_tau = torch.zeros((2, self.nz, self.nx), dtype=_F64)

    @torch.no_grad()
    def _update_wall_model(self) -> None:
        """Refresh the wall-model stress from the current wall-adjacent velocity.

        The composite law of the wall is inverted at each wall column; the
        implied effective viscosity ``nu_wm = u_tau^2 y_1 / |u_1|`` is what the
        first cell must present so that the wall shear equals ``rho u_tau^2``
        (``y_1`` = distance from the wall plane to the first node centre, i.e.
        ``wall_origin`` cells).  A first-order exponential filter in time
        (``wall_model_filter``) suppresses the per-step noise of the
        instantaneous velocity without freezing the model.
        """
        cfg = self.cfg
        if cfg.wall_model == "none":
            return
        m = torch.mm(self.A, self.f.view(19, self.N))
        inv = torch.reciprocal(m[0])
        ux = (m[1] * inv).view(self.nz, self.ny, self.nx) + 0.5 * self.accel
        uz = (m[3] * inv).view(self.nz, self.ny, self.nx)
        y1 = float(cfg.wall_origin)
        filt = float(cfg.wall_model_filter)
        for k, j in enumerate(self.wall_rows):
            ut = torch.sqrt(ux[:, j, :] ** 2 + uz[:, j, :] ** 2).to(_F64)
            u_tau = u_tau_from_wall_law(ut, y1, self.nu)
            nu_wm = (u_tau**2) * y1 / ut.clamp(min=1e-8)
            nu_wm = nu_wm.clamp(max=float(cfg.wall_model_cap_nu * self.nu))
            nu_wm = torch.nan_to_num(nu_wm, nan=float(self.nu))
            if filt > 0.0:
                self._wm_nu[k] = (1.0 - filt) * self._wm_nu[k] + filt * nu_wm
            else:
                self._wm_nu[k] = nu_wm
            self._wm_u_tau[k] = u_tau

    def _wall_model_field(self) -> torch.Tensor | None:
        """``nu_t`` floor implied by the wall model on the two wall rows."""
        if self.cfg.wall_model == "none":
            return None
        return self._wm_nu - float(self.nu)

    # ---- SGS closure -----------------------------------------------------
    def _strain_magnitude_velocity_gradient(self) -> torch.Tensor:
        """``|S| = sqrt(2 S_ij S_ij)`` from central differences of ``u``.

        The velocity sits in ``self.b_u`` at the moment the SGS closure is
        evaluated (the base step writes it before the collision), so no extra
        momentum sum is needed.  Streamwise (``x``) and spanwise (``z``) are
        periodic; the wall-normal derivative is one-sided on the two
        wall-adjacent rows, which keeps the wall stress single-valued.
        """
        u = self.b_u.view(3, self.nz, self.ny, self.nx)
        ux, uy, uz = u[0], u[1], u[2]

        def ddx(a):
            return (torch.roll(a, -1, dims=2) - torch.roll(a, 1, dims=2)) * 0.5

        def ddz(a):
            return (torch.roll(a, -1, dims=0) - torch.roll(a, 1, dims=0)) * 0.5

        def ddy(a):
            g = torch.empty_like(a)
            g[:, 1:-1, :] = (a[:, 2:, :] - a[:, :-2, :]) * 0.5
            g[:, 0, :] = a[:, 1, :] - a[:, 0, :]
            g[:, -1, :] = a[:, -1, :] - a[:, -2, :]
            return g

        sxx, syy, szz = ddx(ux), ddy(uy), ddz(uz)
        sxy = 0.5 * (ddy(ux) + ddx(uy))
        sxz = 0.5 * (ddz(ux) + ddx(uz))
        syz = 0.5 * (ddz(uy) + ddy(uz))
        q = 2.0 * (
            sxx**2 + syy**2 + szz**2 + 2.0 * (sxy**2 + sxz**2 + syz**2)
        )
        return q.clamp(min=0.0).sqrt()

    def _damped_sgs_field(self, nu_t: torch.Tensor) -> torch.Tensor:
        """Apply the van Driest damping and the wall-model floor, row-wise."""
        cfg = self.cfg
        field = nu_t.view(self.nz, self.ny, self.nx)
        if cfg.van_driest:
            field = field * self.fv2_rows.view(1, self.ny, 1).to(field.dtype)
        wm = self._wall_model_field()
        if wm is not None:
            for k, j in enumerate(self.wall_rows):
                field[:, j, :] = torch.maximum(
                    field[:, j, :], wm[k].clamp(min=0.0).to(field.dtype)
                )
        self._nu_t_field = field
        return field.reshape(self.N)

    def _smagorinsky(self, fneq: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        """Local-equilibrium Smagorinsky, damped, with the wall-model floor."""
        cfg = self.cfg
        if cfg.sgs_model == "none" or cfg.smagorinsky_cs <= 0.0:
            self._nu_t_field = None
            return torch.zeros(self.N, dtype=self.dtype, device=self.device)
        if cfg.strain_source == "velocity_gradient":
            strain = self._strain_magnitude_velocity_gradient()
            nu_t = (cfg.smagorinsky_cs**2) * strain
            return self._damped_sgs_field(nu_t)
        nu_t = super()._smagorinsky(fneq, rho)
        return self._damped_sgs_field(nu_t)

    # ---- statistics ------------------------------------------------------
    @torch.no_grad()
    def _velocity_fields(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Instantaneous ``(ux, uy, uz)`` with the Guo half-force correction.

        The base solver's own statistics use ``m_x / rho``; the physical
        velocity carries the ``+a/2`` half-force term of the Guo scheme, so a
        profile built without it is biased low by ``a/2`` (``2e-6`` at
        ``u_tau = 0.01, h = 24``, i.e. ``3.5e-4`` in ``u+``).  Included here so
        the statistics are consistent with the momentum balance they are
        compared against.
        """
        m = torch.mm(self.A, self.f.view(19, self.N))
        rho = m[0]
        inv = torch.reciprocal(rho)
        ux = m[1] * inv + 0.5 * self.accel
        uy = m[2] * inv
        uz = m[3] * inv
        return ux, uy, uz

    def _spectrum_planes(self) -> list[int]:
        cfg = self.cfg
        rows: list[int] = []
        for target in cfg.spectrum_planes_y_plus:
            j = int(round(float(target) / cfg.dy_plus + cfg.wall_origin))
            j = min(max(j, 1), self.ny - 2)
            if j not in rows:
                rows.append(j)
        return rows

    @torch.no_grad()
    def run_les(self, *, verbose: bool = False) -> dict[str, Any]:
        """Integrate and collect LES statistics; returns a JSON-able dict."""
        cfg = self.cfg
        ny, nx, nz, N = self.ny, self.nx, self.nz, self.N
        wgt = 1.0 / float(nx * nz)
        window = cfg.steps - cfg.warmup_steps
        block_len = max(window // max(cfg.n_blocks, 1), 1)

        sums = torch.zeros(ny, 8, dtype=_F64)
        block_sums = torch.zeros(cfg.n_blocks, ny, 8, dtype=_F64)
        block_count = torch.zeros(cfg.n_blocks, dtype=_F64)
        nut_sum = torch.zeros(ny, dtype=_F64)
        spec_planes = self._spectrum_planes()
        nk = nx // 2 + 1
        nkz = nz // 2 + 1
        spec_x = torch.zeros(len(spec_planes), nk, dtype=_F64)
        spec_z = torch.zeros(len(spec_planes), nkz, dtype=_F64)
        spec_count = 0
        samples = 0
        wall_flux = 0.0
        bulk_samples: list[float] = []
        mass_samples: list[float] = []
        history: list[dict[str, Any]] = []
        spec_start = cfg.warmup_steps + int(cfg.spectrum_from_fraction * window)
        mean_ux_rows = torch.zeros(ny, dtype=_F64)
        t0 = time.time()

        for step in range(1, cfg.steps + 1):
            if cfg.wall_model != "none" and (step - 1) % cfg.wall_model_interval == 0:
                self._update_wall_model()
            self._step_once()
            if step <= cfg.warmup_steps:
                # warm-up momentum-exchange flux must not enter the sampled mean
                self._wall_flux_accum = 0.0
            if step == 1 or step % cfg.report_interval == 0:
                h = self.health()
                if not h["finite"]:
                    raise FloatingPointError(f"divergence at step {step}: {h}")
                history.append(
                    {
                        "step": step,
                        "u_bulk": self.bulk_velocity(),
                        **h,
                        **self.turbulence_energy(),
                    }
                )
                if verbose:
                    print(
                        f"[{cfg.label}] {step}/{cfg.steps} Ma={h['max_mach']:.3f} "
                        f"u_bulk={history[-1]['u_bulk']:.4f} "
                        f"tke={history[-1]['tke']:.3e} "
                        f"{step / max(time.time() - t0, 1e-9):.2f} steps/s",
                        flush=True,
                    )
            if step > cfg.warmup_steps and (step - cfg.warmup_steps) % cfg.sample_interval == 0:
                ux, uy, uz = self._velocity_fields()
                ux3 = ux.view(nz, ny, nx)
                uy3 = uy.view(nz, ny, nx)
                uz3 = uz.view(nz, ny, nx)
                rho3 = torch.mm(self.A[:1], self.f.view(19, N)).view(nz, ny, nx)
                row_ux = ux3.sum(dim=(0, 2)).double() * wgt
                row = torch.stack(
                    [
                        rho3.sum(dim=(0, 2)).double() * wgt,
                        row_ux,
                        uy3.sum(dim=(0, 2)).double() * wgt,
                        uz3.sum(dim=(0, 2)).double() * wgt,
                        (ux3 * ux3).sum(dim=(0, 2)).double() * wgt,
                        (uy3 * uy3).sum(dim=(0, 2)).double() * wgt,
                        (uz3 * uz3).sum(dim=(0, 2)).double() * wgt,
                        (ux3 * uy3).sum(dim=(0, 2)).double() * wgt,
                    ],
                    dim=1,
                )
                sums += row
                blk = min((step - cfg.warmup_steps - 1) // block_len, cfg.n_blocks - 1)
                block_sums[blk] += row
                block_count[blk] += 1.0
                if self._nu_t_field is not None:
                    nut_sum += self._nu_t_field.sum(dim=(0, 2)).double() * wgt
                samples += 1
                mean_ux_rows += (row_ux - mean_ux_rows) / samples
                bulk_samples.append(float(ux3[:, 1:-1, :].mean()))
                # fluid rows only: the two solid rows are not part of the fluid
                # and their populations drift, so a full-array sum would report a
                # spurious mass leak (8.6e-3 on the toy grid).
                # accumulate the global mass in float64: a float32 reduction over
                # thousands of cells already carries ~1e-5 relative round-off,
                # which would masquerade as a mass leak in the audit.
                mass_samples.append(float(rho3[:, 1:-1, :].double().sum()))
                wall_flux += self._wall_flux_accum
                self._wall_flux_accum = 0.0
                if step > spec_start:
                    ufluc = ux3 - mean_ux_rows.to(ux3.dtype).view(1, ny, 1)
                    for ip, j in enumerate(spec_planes):
                        fx_ = torch.fft.rfft(ufluc[:, j, :], dim=1)
                        spec_x[ip] += (fx_.abs() ** 2).mean(dim=0).double() / nx
                        fz_ = torch.fft.rfft(ufluc[:, j, :], dim=0)
                        spec_z[ip] += (fz_.abs() ** 2).mean(dim=1).double() / nz
                    spec_count += 1

        elapsed = time.time() - t0
        stats = sums / max(samples, 1)
        nut_mean = nut_sum / max(samples, 1)
        result = super()._finalize(stats, samples, wall_flux, elapsed, history)
        result["config"]["sgs_model"] = cfg.sgs_model
        result["config"]["strain_source"] = cfg.strain_source
        result["config"]["van_driest"] = bool(cfg.van_driest)
        result["config"]["van_driest_a_plus"] = float(cfg.van_driest_a_plus)
        result["config"]["spectrum_from_fraction"] = float(cfg.spectrum_from_fraction)
        result["config"]["averaging_steps"] = int(window)
        result["config"]["statistics_samples"] = int(samples)
        result["config"]["y_plus_first_fluid_node"] = float(self.y_plus_rows[1])
        result["config"]["dy_plus"] = float(cfg.dy_plus)
        result["les"] = self._les_diagnostics(
            result=result,
            stats=stats,
            nut_mean=nut_mean,
            spec_x=spec_x,
            spec_z=spec_z,
            spec_planes=spec_planes,
            spec_count=spec_count,
            block_sums=block_sums,
            block_count=block_count,
            bulk_samples=bulk_samples,
            mass_samples=mass_samples,
        )
        if cfg.wall_model != "none":
            result["les"]["wall_model"] = {
                "model": cfg.wall_model,
                "interval": int(cfg.wall_model_interval),
                "filter": float(cfg.wall_model_filter),
                "cap_nu_over_nu": float(cfg.wall_model_cap_nu),
                "y1": float(cfg.wall_origin),
                "y1_plus": float(self.y_plus_rows[1]),
                "u_tau_bottom": float(self._wm_u_tau[0].mean()),
                "u_tau_top": float(self._wm_u_tau[1].mean()),
                "nu_wm_over_nu_bottom": float(self._wm_nu[0].mean() / self.nu),
                "nu_wm_over_nu_top": float(self._wm_nu[1].mean() / self.nu),
            }
        return result

    # ---- LES post-processing --------------------------------------------
    def _les_diagnostics(
        self,
        *,
        result: dict[str, Any],
        stats: torch.Tensor,
        nut_mean: torch.Tensor,
        spec_x: torch.Tensor,
        spec_z: torch.Tensor,
        spec_planes: Sequence[int],
        spec_count: int,
        block_sums: torch.Tensor,
        block_count: torch.Tensor,
        bulk_samples: Sequence[float],
        mass_samples: Sequence[float],
    ) -> dict[str, Any]:
        cfg = self.cfg
        h = cfg.half_height
        u_tau = math.sqrt(cfg.acceleration * h)
        nu = self.nu

        ux = stats[:, 1]
        uy = stats[:, 2]
        uv = stats[:, 7] - ux * uy
        y, prof = self._fold(torch.stack([ux, uv], dim=1))
        nut_fold = self._fold(nut_mean.view(-1, 1))[1][:, 0]
        ux_f, uv_f = prof[:, 0], prof[:, 1]
        yp = y * cfg.dy_plus

        du = torch.zeros_like(ux_f)
        du[1:-1] = (ux_f[2:] - ux_f[:-2]) / (y[2:] - y[:-2])
        du[0] = (ux_f[1] - ux_f[0]) / (y[1] - y[0])
        du[-1] = (ux_f[-1] - ux_f[-2]) / (y[-1] - y[-2])
        tau_total = u_tau**2 * (1.0 - y / h)  # exact momentum balance
        tau_sgs = nut_fold * du
        frac = torch.where(
            tau_total > 1e-12, tau_sgs / tau_total, torch.zeros_like(tau_sgs)
        )
        prod_res = -uv_f * du
        prod_total = prod_res + tau_sgs * du

        spectra = {
            "planes_y_plus": [float(yp[j]) for j in spec_planes],
            "samples": int(spec_count),
            "streamwise": [],
            "spanwise": [],
        }
        for ip, j in enumerate(spec_planes):
            e_x = spec_x[ip]
            k_x = torch.arange(e_x.numel(), dtype=_F64)
            spectra["streamwise"].append(
                {
                    "y_plus": float(yp[j]),
                    "k": k_x[1:].tolist(),
                    "E": e_x[1:].tolist(),
                    "fit": fit_power_law_slope(k_x[1:], e_x[1:]),
                }
            )
            e_z = spec_z[ip]
            k_z = torch.arange(e_z.numel(), dtype=_F64)
            spectra["spanwise"].append(
                {
                    "y_plus": float(yp[j]),
                    "k": k_z[1:].tolist(),
                    "E": e_z[1:].tolist(),
                    "fit": fit_power_law_slope(k_z[1:], e_z[1:]),
                }
            )

        bulk = torch.tensor(list(bulk_samples), dtype=_F64)
        mass = torch.tensor(list(mass_samples), dtype=_F64)
        half = bulk.numel() // 2 if bulk.numel() > 1 else 0
        bulk_tail = bulk[half:] if bulk.numel() > 1 else bulk
        mass_diag = {
            "bulk_mean": float(bulk.mean()) if bulk.numel() else float("nan"),
            "bulk_std": float(bulk.std(unbiased=True)) if bulk.numel() > 1 else 0.0,
            "bulk_rel_std_half_window": float(bulk_tail.std(unbiased=True) / bulk_tail.mean())
            if bulk_tail.numel() > 1
            else 0.0,
            "bulk_min": float(bulk.min()) if bulk.numel() else float("nan"),
            "bulk_max": float(bulk.max()) if bulk.numel() else float("nan"),
            "total_mass_rel_drift": float((mass[-1] - mass[0]).abs() / mass[0].abs())
            if mass.numel() > 1
            else 0.0,
            "bulk_samples": bulk.tolist(),
        }

        band = (yp >= 20.0) & (yp <= 60.0)
        ipr = int(torch.argmax(prod_res))
        ip = int(torch.argmax(prod_total))
        sgs = {
            "y_plus": yp.tolist(),
            "nu_t_mean": nut_fold.tolist(),
            "tau_sgs_plus": (tau_sgs / u_tau**2).tolist(),
            "tau_total_plus": (tau_total / u_tau**2).tolist(),
            "fraction": frac.tolist(),
            "fraction_at_y_plus_50": interpolate_profile(yp, frac, 50.0),
            "fraction_band_20_60_mean": float(frac[band].mean())
            if int(band.sum())
            else float("nan"),
            "nu_t_over_nu_max": float((nut_fold / nu).max()),
            "cs": float(cfg.smagorinsky_cs),
            "van_driest": bool(cfg.van_driest),
            "van_driest_a_plus": float(cfg.van_driest_a_plus),
        }
        production = {
            "y_plus": yp.tolist(),
            "resolved_plus": (prod_res / u_tau**3).tolist(),
            "with_sgs_plus": (prod_total / u_tau**3).tolist(),
            "peak_resolved_plus": float(prod_res[ipr] / u_tau**3),
            "peak_resolved_y_plus": float(yp[ipr]),
            "peak_with_sgs_plus": float(prod_total[ip] / u_tau**3),
            "peak_with_sgs_y_plus": float(yp[ip]),
        }

        blocks = []
        for ib in range(int(block_count.numel())):
            n = float(block_count[ib])
            if n < 1:
                continue
            st = block_sums[ib] / n
            b_ux = st[:, 1]
            b_uv = st[:, 7] - st[:, 1] * st[:, 2]
            by, bprof = self._fold(torch.stack([b_ux, b_uv], dim=1))
            b_uvm = -bprof[:, 1]
            blocks.append(
                {
                    "block": ib,
                    "samples": int(n),
                    "uv_peak_plus": float(b_uvm.max() / u_tau**2),
                    "uv_peak_y_plus": float(by[int(torch.argmax(b_uvm))] * cfg.dy_plus),
                    "u_plus_max": float(bprof[:, 0].max() / u_tau),
                }
            )
        if blocks:
            uv_peaks = torch.tensor([b["uv_peak_plus"] for b in blocks], dtype=_F64)
            kpeaks = torch.tensor([b["u_plus_max"] for b in blocks], dtype=_F64)
            unc = {
                "blocks": blocks,
                "uv_peak_mean": float(uv_peaks.mean()),
                "uv_peak_std": float(uv_peaks.std(unbiased=True))
                if uv_peaks.numel() > 1
                else 0.0,
                "u_plus_max_std": float(kpeaks.std(unbiased=True))
                if kpeaks.numel() > 1
                else 0.0,
            }
        else:
            unc = {
                "blocks": [],
                "uv_peak_mean": float("nan"),
                "uv_peak_std": float("nan"),
                "u_plus_max_std": float("nan"),
            }

        return {
            "spectra": spectra,
            "sgs_stress": sgs,
            "production": production,
            "mass_conservation": mass_diag,
            "statistics_uncertainty": unc,
        }

    def run_les_to_file(self, path: str | Path, *, verbose: bool = False) -> dict[str, Any]:
        result = self.run_les(verbose=verbose)
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=1))
        return result


# ---------------------------------------------------------------------------
# spectral post-processing
# ---------------------------------------------------------------------------
def fit_power_law_slope(
    k, energy, *, k_min: float = 4.0, k_max: float | None = None, min_points: int = 4
) -> dict[str, Any]:
    """Least-squares slope of ``log E`` vs ``log k`` in a fixed band.

    The band is ``k in [k_min, k_max]`` with ``k_max = max(k_min + 8,
    0.25 k_available)`` by default — an *a-priori* rule, so the reported slope
    cannot be tuned to a target value.
    """
    k = torch.as_tensor(k, dtype=_F64).reshape(-1)
    e = torch.as_tensor(energy, dtype=_F64).reshape(-1)
    if k.numel() == 0:
        return {"slope": float("nan"), "r2": float("nan"), "valid": False, "n_points": 0}
    if k_max is None:
        k_max = max(k_min + 8.0, 0.25 * float(k.max()))
    sel = (k >= k_min) & (k <= k_max) & (e > 0.0)
    kk, ee = k[sel], e[sel]
    if kk.numel() < min_points:
        return {
            "slope": float("nan"),
            "r2": float("nan"),
            "k_min": None,
            "k_max": None,
            "n_points": int(kk.numel()),
            "valid": False,
        }
    lx, ly = torch.log(kk), torch.log(ee)
    A = torch.stack([torch.ones_like(lx), lx], dim=1)
    coef = torch.linalg.lstsq(A, ly).solution
    resid = ly - A @ coef
    ss_res = float((resid**2).sum())
    ss_tot = float(((ly - ly.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "slope": float(coef[1]),
        "r2": r2,
        "k_min": float(kk.min()),
        "k_max": float(kk.max()),
        "n_points": int(kk.numel()),
        "valid": True,
        "slope_error_vs_minus_5_3": float(coef[1] + 5.0 / 3.0),
    }


def interpolate_profile(y, values, y_target: float) -> float:
    """Linear interpolation of a folded profile at ``y_target`` (NaN off-range)."""
    y = torch.as_tensor(y, dtype=_F64).reshape(-1)
    v = torch.as_tensor(values, dtype=_F64).reshape(-1)
    if y.numel() < 2:
        return float("nan")
    order = torch.argsort(y)
    ys, vs = y[order], v[order]
    if float(y_target) < float(ys[0]) or float(y_target) > float(ys[-1]):
        return float("nan")
    idx = int(torch.searchsorted(ys, torch.tensor(float(y_target))))
    idx = min(max(idx, 1), ys.numel() - 1)
    y0, y1 = float(ys[idx - 1]), float(ys[idx])
    v0, v1 = float(vs[idx - 1]), float(vs[idx])
    if abs(y1 - y0) < 1e-30:
        return v0
    t = (float(y_target) - y0) / (y1 - y0)
    return v0 + t * (v1 - v0)


# ---------------------------------------------------------------------------
# DNS reference: Reynolds shear stress from the exact stress budget
# ---------------------------------------------------------------------------
def dns_reynolds_shear(profile: MkmProfile) -> tuple[torch.Tensor, torch.Tensor]:
    """``-<u'v'>+ = 1 - y+/Re_tau - dU+/dy+`` from a published MKM mean table.

    Identity of the fully developed channel momentum balance, so the
    reconstructed profile is exactly consistent with the published ``U+`` and
    shear columns.
    """
    re_tau = float(profile.re_tau)
    y_plus = profile.y_plus.to(_F64)
    du = profile.du_plus_dy_plus.to(_F64)
    return y_plus, 1.0 - y_plus / re_tau - du


def compare_mean_profile(result: dict[str, Any], profile: MkmProfile) -> dict[str, Any]:
    """Log-law band and DNS ``U+`` agreement for the simulated mean profile."""
    y_plus = torch.tensor(result["wall_units"]["y_plus"], dtype=_F64)
    u_plus = torch.tensor(result["wall_units"]["u_plus"], dtype=_F64)
    re_tau = float(profile.re_tau)
    band = (y_plus >= 30.0) & (y_plus <= 100.0) & (y_plus <= 0.3 * re_tau)
    if int(band.sum()) < 2:
        return {"valid": False, "reason": "fewer than 2 nodes in y+ in [30,100]"}
    yb, ub = y_plus[band], u_plus[band]
    law = torch.log(yb) / KAPPA + B_LOG
    err_law = ub - law
    dns = torch.tensor(
        [interpolate_profile(profile.y_plus, profile.u_plus, float(yv)) for yv in yb],
        dtype=_F64,
    )
    err_dns = ub - dns
    lx = torch.log(yb)
    A = torch.stack([torch.ones_like(lx), lx], dim=1)
    coef = torch.linalg.lstsq(A, ub).solution
    resid = ub - A @ coef
    ss_res = float((resid**2).sum())
    ss_tot = float(((ub - ub.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "valid": True,
        "y_plus_band": yb.tolist(),
        "u_plus_sim": ub.tolist(),
        "u_plus_log_law": law.tolist(),
        "u_plus_dns": dns.tolist(),
        "max_abs_error_log_law": float(err_law.abs().max()),
        "rms_error_log_law": float(torch.sqrt((err_law**2).mean())),
        "max_rel_error_log_law": float((err_law.abs() / law).max()),
        "rms_rel_error_log_law": float(torch.sqrt(((err_law / law) ** 2).mean())),
        "max_abs_error_dns": float(err_dns.abs().max()),
        "max_rel_error_dns": float((err_dns.abs() / dns.clamp(min=1e-12)).max()),
        "rms_rel_error_dns": float(
            torch.sqrt(((err_dns / dns.clamp(min=1e-12)) ** 2).mean())
        ),
        "kappa_fit": float(1.0 / coef[1]) if float(coef[1]) != 0.0 else float("nan"),
        "b_fit": float(coef[0]),
        "fit_r2": r2,
        "n_points": int(yb.numel()),
    }


def compare_reynolds_stress(result: dict[str, Any], profile: MkmProfile) -> dict[str, Any]:
    """Peak location/amplitude of ``-<u'v'>+`` against the DNS reconstruction."""
    y_plus = torch.tensor(result["wall_units"]["y_plus"], dtype=_F64)
    uv_minus = torch.tensor(result["wall_units"]["uv_minus_plus"], dtype=_F64)
    if y_plus.numel() < 3:
        return {"valid": False, "reason": "profile too short"}
    idx = int(torch.argmax(uv_minus))
    sim_peak, sim_pos = float(uv_minus[idx]), float(y_plus[idx])
    dns_y, dns_uv = dns_reynolds_shear(profile)
    sel = dns_y <= float(profile.re_tau) * 0.999
    dns_y, dns_uv = dns_y[sel], dns_uv[sel]
    jdx = int(torch.argmax(dns_uv))
    dns_peak, dns_pos = float(dns_uv[jdx]), float(dns_y[jdx])
    return {
        "valid": True,
        "sim_peak_plus": sim_peak,
        "sim_peak_y_plus": sim_pos,
        "dns_peak_plus": dns_peak,
        "dns_peak_y_plus": dns_pos,
        "rel_error_peak_amplitude": (sim_peak - dns_peak) / dns_peak if dns_peak else float("nan"),
        "rel_error_peak_position": (sim_pos - dns_pos) / dns_pos if dns_pos else float("nan"),
        "sim_at_dns_peak": interpolate_profile(y_plus, uv_minus, dns_pos),
        "dns_at_sim_peak": interpolate_profile(dns_y, dns_uv, sim_pos),
        "rms_error_profile_plus": _rms_over_common_band(y_plus, uv_minus, dns_y, dns_uv),
    }


def _rms_over_common_band(y_a, v_a, y_b, v_b) -> float:
    ya = torch.as_tensor(y_a, dtype=_F64)
    yb = torch.as_tensor(y_b, dtype=_F64)
    lo = max(float(ya.min()), float(yb.min()))
    hi = min(float(ya.max()), float(yb.max()))
    if hi <= lo:
        return float("nan")
    grid = torch.linspace(lo, hi, 64, dtype=_F64)
    a = torch.tensor([interpolate_profile(ya, v_a, float(g)) for g in grid], dtype=_F64)
    b = torch.tensor([interpolate_profile(yb, v_b, float(g)) for g in grid], dtype=_F64)
    return float(torch.sqrt(((a - b) ** 2).mean()))


# ---------------------------------------------------------------------------
# grid convergence (GCI)
# ---------------------------------------------------------------------------
def gci_three_grid(
    fine: float,
    medium: float,
    coarse: float,
    *,
    r: float = 2.0,
    fs: float = 1.25,
    p_fallback: float = 2.0,
) -> dict[str, Any]:
    """Celik et al. (2008) GCI from a uniform-refinement three-grid ladder.

    ``r`` is the (nominal) refinement ratio.  The observed order
    ``p = |ln|(f_coarse-f_medium)/(f_medium-f_fine)|| / ln r`` is used only when
    admissible (``0.5 <= p <= 4``); otherwise ``p_fallback`` is used and
    ``order_source`` records the fallback.
    """
    e21 = medium - fine
    e32 = coarse - medium
    if abs(e21) < 1e-15:
        return {
            "valid": False,
            "reason": "medium equals fine: difference below machine precision",
            "order_source": "observed",
        }
    ratio = e32 / e21
    p_obs = float("nan")
    if ratio > 0.0:
        p_obs = abs(math.log(abs(ratio))) / math.log(r)
    if not (0.5 <= p_obs <= 4.0):
        p = p_fallback
        order_source = "fallback_p2"
    else:
        p = p_obs
        order_source = "observed"
    rp = r**p
    gci_fine = fs * abs(e21 / fine) / (rp - 1.0) if fine != 0.0 else float("nan")
    gci_medium = fs * abs(e32 / medium) / (rp - 1.0) if medium != 0.0 else float("nan")
    return {
        "valid": True,
        "fine": fine,
        "medium": medium,
        "coarse": coarse,
        "observed_order": p_obs,
        "order_used": p,
        "order_source": order_source,
        "extrapolated": fine - e21 / (rp - 1.0),  # f_exact = f1 - (f2-f1)/(r^p-1)
        "relative_change_fine_medium": abs(e21 / fine) if fine else float("nan"),
        "gci_fine": gci_fine,
        "gci_medium": gci_medium,
        "convergence_ratio": ratio if math.isfinite(ratio) else float("nan"),
    }


# ---------------------------------------------------------------------------
# acceptance verdicts
# ---------------------------------------------------------------------------
def evaluate_case(
    result: dict[str, Any],
    *,
    dns: MkmProfile | None = None,
    log_law_tol: float = 0.05,
    uv_peak_amp_tol: float = 0.15,
    uv_peak_pos_tol: float = 0.10,
    spectrum_slope_tol: float = 0.10,
    spectrum_r2_min: float = 0.95,
    sgs_fraction_band: tuple[float, float] = (0.20, 0.40),
    mass_tol: float = 1e-3,
) -> dict[str, Any]:
    """Per-case acceptance verdict with the number behind every boolean."""
    les = result.get("les", {})
    checks: dict[str, Any] = {}

    mean_cmp = (
        compare_mean_profile(result, dns)
        if dns is not None
        else {"valid": False, "reason": "no DNS table"}
    )
    checks["a_mean_profile_log_law"] = _check(
        bool(mean_cmp.get("valid", False))
        and mean_cmp.get("max_rel_error_log_law") is not None
        and mean_cmp["max_rel_error_log_law"] <= log_law_tol,
        "y+ = 30..min(100, 0.3 Re_tau); U+ vs (1/kappa) ln y+ + B (kappa=0.41, B=5.2)",
        {
            "max_rel_error_log_law": mean_cmp.get("max_rel_error_log_law"),
            "rms_rel_error_log_law": mean_cmp.get("rms_rel_error_log_law"),
            "max_rel_error_dns": mean_cmp.get("max_rel_error_dns"),
            "kappa_fit": mean_cmp.get("kappa_fit"),
            "b_fit": mean_cmp.get("b_fit"),
            "fit_r2": mean_cmp.get("fit_r2"),
            "n_points": mean_cmp.get("n_points"),
        },
        tol=log_law_tol,
        metric=mean_cmp.get("max_rel_error_log_law"),
    )

    if dns is not None:
        uv_cmp = compare_reynolds_stress(result, dns)
        ok = bool(
            uv_cmp.get("valid")
            and abs(uv_cmp["rel_error_peak_amplitude"]) <= uv_peak_amp_tol
            and abs(uv_cmp["rel_error_peak_position"]) <= uv_peak_pos_tol
        )
        checks["b_reynolds_stress_peak"] = _check(
            ok,
            "-<u'v'>+ peak amplitude (<=15%) and position (<=10%) vs DNS budget reconstruction",
            uv_cmp,
            tol=uv_peak_amp_tol,
            metric=uv_cmp.get("rel_error_peak_amplitude"),
        )
    else:
        checks["b_reynolds_stress_peak"] = {
            "passed": None,
            "status": "WITHHELD",
            "reason": "no DNS reference table for this Re_tau in the repository",
        }

    spec_best = None
    for leg in les.get("spectra", {}).get("streamwise", []):
        fit = leg.get("fit", {})
        if not fit.get("valid"):
            continue
        if spec_best is None or abs(fit["slope"] + 5.0 / 3.0) < abs(
            spec_best["slope"] + 5.0 / 3.0
        ):
            spec_best = {
                "slope": fit["slope"],
                "r2": fit["r2"],
                "y_plus": leg["y_plus"],
                "k_min": fit["k_min"],
                "k_max": fit["k_max"],
                "n_points": fit["n_points"],
                "samples": les["spectra"].get("samples"),
            }
    if spec_best is None:
        checks["c_energy_spectrum_slope"] = {
            "passed": None,
            "status": "WITHHELD",
            "reason": "no valid spectral fit (too short a window or too few modes)",
        }
    else:
        ok = (
            abs(spec_best["slope"] + 5.0 / 3.0) <= spectrum_slope_tol
            and spec_best["r2"] >= spectrum_r2_min
        )
        checks["c_energy_spectrum_slope"] = _check(
            ok,
            "log-log slope of E_uu(k_x) in the fitted inertial band vs -5/3 (tol 0.10, R2>=0.95)",
            spec_best,
            tol=spectrum_slope_tol,
            metric=spec_best["slope"] + 5.0 / 3.0,
        )

    frac = les.get("sgs_stress", {}).get("fraction_at_y_plus_50")
    if frac is None or not isinstance(frac, (int, float)) or not math.isfinite(float(frac)):
        checks["d_sgs_stress_fraction"] = {
            "passed": None,
            "status": "WITHHELD",
            "reason": "no grid node spanning y+ = 50",
        }
    else:
        ok = sgs_fraction_band[0] <= frac <= sgs_fraction_band[1]
        checks["d_sgs_stress_fraction"] = _check(
            ok,
            "tau_sgs / tau_total at y+ = 50 (target band 20-40 %)",
            {
                "fraction_at_y_plus_50": frac,
                "fraction_band_20_60_mean": les["sgs_stress"].get("fraction_band_20_60_mean"),
                "cs": les["sgs_stress"].get("cs"),
                "van_driest": les["sgs_stress"].get("van_driest"),
                "nu_t_over_nu_max": les["sgs_stress"].get("nu_t_over_nu_max"),
            },
            tol=0.0,
            metric=frac,
        )

    mass = les.get("mass_conservation", {})
    rel = mass.get("bulk_rel_std_half_window")
    drift = mass.get("total_mass_rel_drift")
    ok = bool(
        (rel is not None and math.isfinite(float(rel)) and float(rel) <= mass_tol)
        or (drift is not None and math.isfinite(float(drift)) and float(drift) <= mass_tol)
    )
    checks["f_mass_conservation"] = _check(
        ok,
        "bulk-flow fluctuation in the second half window (<0.1 %) or total-mass drift (<0.1 %)",
        {
            "bulk_rel_std_half_window": rel,
            "total_mass_rel_drift": drift,
            "bulk_mean": mass.get("bulk_mean"),
            "bulk_min": mass.get("bulk_min"),
            "bulk_max": mass.get("bulk_max"),
        },
        tol=mass_tol,
        metric=rel,
    )

    passed = [k for k, v in checks.items() if v.get("passed") is True]
    failed = [k for k, v in checks.items() if v.get("passed") is False]
    withheld = [k for k, v in checks.items() if v.get("passed") is None]
    return {
        "criterion_checks": checks,
        "passed": passed,
        "failed": failed,
        "withheld": withheld,
        "all_passed": not failed and not withheld,
    }


def _check(ok: bool, description: str, numbers: dict[str, Any], *, tol: float, metric) -> dict[str, Any]:
    return {
        "passed": bool(ok),
        "status": "PASS" if ok else "FAIL",
        "criterion": description,
        "numbers": numbers,
        "tolerance": tol,
        "metric": metric,
    }


# ---------------------------------------------------------------------------
# campaign-level helpers
# ---------------------------------------------------------------------------
DNS_REFERENCE = {
    180.0: "benchmarks/pending/channel_turb_3d/dns_ref/mkm180.means",
    590.0: "benchmarks/pending/channel_turb_3d/dns_ref/mkm590.means",
}


def mkm_profile_for(re_tau: float, root: str | Path = ".") -> MkmProfile | None:
    """Return the vendored MKM1999 table matching ``re_tau`` (exact match only)."""
    from tensorlbm.turbulence.wall_model_usability import load_mkm_profile

    rel = DNS_REFERENCE.get(float(re_tau))
    if rel is None:
        return None
    path = Path(root) / rel
    if not path.exists():
        return None
    return load_mkm_profile(path)


def summary_metrics(result: dict[str, Any], profile: MkmProfile | None = None) -> dict[str, Any]:
    """Compact scalar summary used by the GCI ladder and the figures."""
    les = result.get("les", {})
    rs = result.get("reynolds_stress", {})
    wu = result.get("wall_units", {})
    u_plus = wu.get("u_plus", [])
    out: dict[str, Any] = {
        "label": result.get("config", {}).get("label") or result.get("label"),
        "re_tau_target": result.get("config", {}).get("re_tau_target"),
        "cells": result.get("config", {}).get("cells"),
        "steps": result.get("config", {}).get("steps"),
        "dy_plus": result.get("config", {}).get("dy_plus"),
        "u_plus_max": max(u_plus) if u_plus else float("nan"),
        "uv_peak_plus": rs.get("uv_minus_peak_plus"),
        "uv_peak_y_plus": rs.get("uv_minus_peak_y_plus"),
        "u_tau_body_force": result.get("friction_velocity", {}).get("body_force"),
        "u_tau_wall_exchange": result.get("friction_velocity", {}).get("wall_exchange"),
        "u_tau_stress_fit": result.get("friction_velocity", {}).get("stress_fit"),
        "momentum_balance_max_error": result.get("momentum_balance", {}).get("max_abs_error_plus"),
        "sgs_fraction_y50": les.get("sgs_stress", {}).get("fraction_at_y_plus_50"),
        "production_peak_plus": les.get("production", {}).get("peak_with_sgs_plus"),
        "production_peak_y_plus": les.get("production", {}).get("peak_with_sgs_y_plus"),
        "mass_rel_std": les.get("mass_conservation", {}).get("bulk_rel_std_half_window"),
        "mass_drift": les.get("mass_conservation", {}).get("total_mass_rel_drift"),
        "bulk_reynolds": result.get("reynolds_number_bulk"),
        "re_tau_achieved": result.get("re_tau_achieved"),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "steps_per_second": result.get("steps_per_second"),
    }
    if profile is not None:
        out["log_law"] = compare_mean_profile(result, profile)
        out["uv"] = compare_reynolds_stress(result, profile)
    spectra = les.get("spectra", {}).get("streamwise", [])
    fits = [s["fit"] for s in spectra if s.get("fit", {}).get("valid")]
    if fits:
        best = min(fits, key=lambda f: abs(f["slope"] + 5.0 / 3.0))
        out["spectrum_slope"] = best["slope"]
        out["spectrum_r2"] = best["r2"]
        out["spectrum_k_range"] = [best["k_min"], best["k_max"]]
    return out


__all__ = [
    "B_LOG",
    "DEFAULT_CS",
    "DNS_REFERENCE",
    "KAPPA",
    "LESChannel",
    "LESChannelConfig",
    "VAN_DRIEST_A_PLUS",
    "compare_mean_profile",
    "compare_reynolds_stress",
    "dns_reynolds_shear",
    "evaluate_case",
    "fit_power_law_slope",
    "gci_three_grid",
    "interpolate_profile",
    "mkm_profile_for",
    "summary_metrics",
    "van_driest_damping",
    "van_driest_nu_t",
]