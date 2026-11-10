"""Wall-resolved 3D turbulent plane-channel solver (D3Q19 TRT + Guo forcing).

Physics
-------
Pressure-driven plane channel of half-height ``h``, streamwise/spanwise
periodic, no-slip walls.  The driving pressure gradient is represented by a
constant body force ``a`` (Guo forcing), so the streamwise momentum balance of
the statistically stationary flow reads exactly

    tau_w = rho * a * h,        u_tau = sqrt(a * h),        a = 2 u_tau^2 / H

with ``H = 2h`` the channel height in lattice units.

Discretisation
--------------
* lattice D3Q19 (``tensorlbm.d3q19`` velocities / weights / opposite map);
* collision: two-relaxation-time (TRT) with the magic parameter
  ``Lambda = 3/16``, i.e. ``(tau+ - 1/2)(tau- - 1/2) = 3/16``; with the magic
  parameter the bounce-back wall is second-order accurate and the scheme stays
  stable for ``tau+ -> 1/2`` (needed at Re_tau = 395);
* forcing: Guo (1993) with the symmetric/antisymmetric split consistent with
  TRT, so the discrete Poiseuille solution is second order;
* walls: Bouzidi interpolated bounce-back.  ``wall_origin`` is the position of
  the no-slip plane measured from the centre of the innermost solid node row, in
  lattice units; ``0.5`` is classical half-way bounce-back.  The *realised*
  wall origin is measured back from laminar twin runs
  (:func:`run_laminar_poiseuille`), never assumed.

Wall units
----------
For fluid row ``j`` (``1 <= j <= ny-2``) the distance to the lower wall is
``y = j - wall_origin`` (symmetric for the upper wall), ``dy+ = u_tau/nu`` and
``y+ = y u_tau / nu``.  A wrong wall origin shifts every ``y+`` by
``delta * dy+``; this is a pure post-processing error, hence the campaign
insists on a *measured* ``delta``.

Three independent friction-velocity routes
------------------------------------------
``body_force``    ``sqrt(a H/2)`` -- the driving force, exact by construction.
``wall_exchange`` momentum-exchange (Ladd) wall flux accumulated over the
                  averaging window; uses populations only, no knowledge of ``a``.
``stress_fit``    least-squares fit of ``rho (nu dU/dy - <u'v'>) =
                  u_tau^2 (1 - y/h)`` extrapolated to the wall (Reynolds-stress
                  integral route).

Their mutual agreement is the self-consistency gate of the campaign.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from tensorlbm.d3q19 import C, OPPOSITE, W

__all__ = [
    "TurbulentChannel3DConfig",
    "TurbulentChannel3D",
    "run_turbulent_channel3d",
    "run_laminar_poiseuille",
    "log_law_u_plus",
    "reichardt_u_plus",
    "poiseuille_exact",
    "TRT_MAGIC",
]

TRT_MAGIC = 3.0 / 16.0


# ---------------------------------------------------------------------------
# analytic references
# ---------------------------------------------------------------------------
def log_law_u_plus(y_plus, kappa: float = 0.41, b: float = 5.2):
    """von Karman log law (outer reference; invalid in the viscous sublayer)."""
    y_plus = torch.as_tensor(y_plus, dtype=torch.float64)
    return torch.log(y_plus) / kappa + b


def reichardt_u_plus(y_plus, kappa: float = 0.41):
    """Reichardt (1951) composite law.

    ``u+ = ln(1 + kappa y+)/kappa + 7.8 (1 - exp(-y+/11) - (y+/11) exp(-y+/3))``

    Empirical correlation that reproduces channel DNS ``u+`` to about 1 % for
    ``y+ > 5``; used only where no DNS table exists (Re_tau = 395).
    """
    y_plus = torch.as_tensor(y_plus, dtype=torch.float64)
    return torch.log1p(kappa * y_plus) / kappa + 7.8 * (
        1.0 - torch.exp(-y_plus / 11.0) - (y_plus / 11.0) * torch.exp(-y_plus / 3.0)
    )


def poiseuille_exact(y, height: float, accel: float, nu: float):
    """Exact plane-Poiseuille profile ``u(y) = a y (H - y) / (2 nu)`` (y from wall)."""
    y = torch.as_tensor(y, dtype=torch.float64)
    return accel * y * (height - y) / (2.0 * nu)


@dataclass(frozen=True)
class TurbulentChannel3DConfig:
    """Configuration of the D3Q19 TRT channel."""

    nx: int = 128
    nz: int = 64
    ny: int = 64  # includes the two solid node rows -> H = ny - 2
    re_tau: float = 180.0
    u_tau: float = 0.01
    wall_origin: float = 0.5
    steps: int = 30000
    warmup_steps: int = 9000
    sample_interval: int = 5
    report_interval: int = 2000
    perturbation_amplitude: float = 0.35  # relative to u_tau
    perturbation_modes: int = 6
    seed: int = 20261107
    device: str = "cpu"
    dtype: str = "float32"
    smagorinsky_cs: float = 0.0  # 0.0 = no model (pure molecular)
    body_force: float | None = None
    wall_scheme: str = "interpolated"  # "interpolated" (Bouzidi) | "halfway"
    # Guo-force weighting in the TRT step.  "trt_split" weights the symmetric
    # and the antisymmetric force sector by (1 - s+/2) and (1 - s-/2); only this
    # split reproduces the exact momentum source (measured dU/dt / a = 1.0000
    # for tau+ in [0.505, 1.1]).  "guo_sp" reuses (1 - s+/2) for both sectors and
    # therefore injects only ~(2.11 - 1.045 s+) a: 2.4 % of the drive at the
    # campaign tau+ = 0.505.  It survives only as an audited negative variant
    # (tests/test_turbulent_channel3d.py::test_guo_sp_force_split_underdrives).
    force_scheme: str = "trt_split"
    label: str = "channel3d"
    output: Path | None = None

    def __post_init__(self) -> None:
        if self.output is not None:
            object.__setattr__(self, "output", Path(self.output))

    # ---- derived physical parameters -------------------------------------
    @property
    def height(self) -> int:
        return self.ny - 2

    @property
    def half_height(self) -> float:
        return 0.5 * self.height

    @property
    def nu(self) -> float:
        return self.u_tau * self.half_height / self.re_tau

    @property
    def tau_plus(self) -> float:
        return 0.5 + 3.0 * self.nu

    @property
    def tau_minus(self) -> float:
        return 0.5 + TRT_MAGIC / (self.tau_plus - 0.5)

    @property
    def acceleration(self) -> float:
        if self.body_force is not None:
            return float(self.body_force)
        return 2.0 * self.u_tau**2 / self.height

    @property
    def dy_plus(self) -> float:
        return self.u_tau / self.nu

    @property
    def eddy_turnover_steps(self) -> float:
        return self.half_height / self.u_tau

    def validate(self) -> None:
        if self.nx < 8 or self.nz < 8 or self.ny < 10:
            raise ValueError("channel dimensions are too small")
        if self.re_tau <= 0.0 or self.u_tau <= 0.0:
            raise ValueError("re_tau and u_tau must be positive")
        if not 0.25 <= float(self.wall_origin) < 1.0:
            raise ValueError("wall_origin must satisfy 0.25 <= wall_origin < 1")
        if not 0 <= self.warmup_steps < self.steps:
            raise ValueError("require 0 <= warmup_steps < steps")
        if self.tau_plus - 0.5 <= 1e-6:
            raise ValueError("tau+ too close to 1/2: nu is not resolved")
        if self.wall_scheme not in ("interpolated", "halfway"):
            raise ValueError("wall_scheme must be 'interpolated' or 'halfway'")
        if self.force_scheme not in ("trt_split", "guo_sp"):
            raise ValueError(
                "force_scheme must be 'trt_split' (exact momentum source) or "
                "'guo_sp' (audited negative variant)"
            )

# ---------------------------------------------------------------------------
# solver
# ---------------------------------------------------------------------------
class TurbulentChannel3D:
    """D3Q19 TRT plane channel; populations stored as ``(19, nz, ny, nx)``."""

    def __init__(self, config: TurbulentChannel3DConfig) -> None:
        config.validate()
        self.cfg = config
        self.device = torch.device(config.device)
        self.dtype = getattr(torch, config.dtype)
        self.gen = torch.Generator(device="cpu").manual_seed(config.seed)

        cfg = config
        self.nx, self.ny, self.nz = cfg.nx, cfg.ny, cfg.nz
        self.N = self.nx * self.ny * self.nz
        self.c = C.to(self.device)
        self.w = W.to(self.device).to(self.dtype)
        self.opp = OPPOSITE.to(self.device)
        self.cx = self.c[:, 0].to(self.dtype)
        self.cy = self.c[:, 1].to(self.dtype)
        self.cz = self.c[:, 2].to(self.dtype)
        # moment matrix A (4,19) = [1; cx; cy; cz] -> one GEMM per step for rho,u
        self.A = torch.stack(
            [torch.ones(19, dtype=self.dtype), self.cx, self.cy, self.cz]
        ).to(self.device)
        self.Ct = self.c.to(self.dtype).contiguous().to(self.device)  # (19,3)
        self.dcol = (self.w * self.cx).view(19, 1)  # w_i c_ix  (force geometry)
        self.wcol = self.w.view(19, 1)

        self.sp = 1.0 / cfg.tau_plus
        self.sm = 1.0 / cfg.tau_minus
        self.accel = cfg.acceleration
        self.nu = cfg.nu

        up = torch.nonzero(self.cy > 0, as_tuple=False).flatten()
        dn = torch.nonzero(self.cy < 0, as_tuple=False).flatten()
        self.idx_up, self.idx_dn = up.to(self.device), dn.to(self.device)
        if not torch.equal(OPPOSITE[up.cpu()], dn.cpu()):
            raise RuntimeError("D3Q19 opposite map is inconsistent")

        self.q = 1.0 - cfg.wall_origin if cfg.wall_scheme == "interpolated" else 0.5
        self._stream_plan = self._build_stream_plan()
        self._allocate_buffers()
        self.f = self._initial_populations()
        self.step_count = 0
        self._wall_flux_accum = 0.0

    # ---- streaming plan --------------------------------------------------
    def _build_stream_plan(self) -> list:
        """For every lattice direction: list of (destination slice, source slice).

        ``y`` is wrapped as well; this is harmless because the two solid rows
        are never read back and the unknown boundary populations are overwritten
        by the bounce-back.
        """
        plan = []
        dims = (self.nz, self.ny, self.nx)
        for i in range(19):
            cx, cy, cz = (int(v) for v in C[i])
            axes = []
            for axis, s in ((0, cz), (1, cy), (2, cx)):
                n = dims[axis]
                shift = s  # f_new[j] = f_post[j - c] (torch.roll convention)
                if shift == 0:
                    axes.append([(slice(None), slice(None))])
                else:
                    p = shift % n
                    axes.append(
                        [
                            (slice(p, None), slice(None, n - p)),
                            (slice(None, p), slice(n - p, None)),
                        ]
                    )
            combos = [()]
            for options in axes:
                combos = [c0 + (opt,) for c0 in combos for opt in options]
            pieces = []
            for combo in combos:
                dst = tuple(combo[a][0] for a in range(3))
                src = tuple(combo[a][1] for a in range(3))
                pieces.append((dst, src))
            plan.append(pieces)
        return plan

    # ---- initial condition ----------------------------------------------
    def _initial_populations(self) -> torch.Tensor:
        cfg = self.cfg
        shape = (self.nz, self.ny, self.nx)
        rho = torch.ones(shape, dtype=self.dtype, device=self.device)
        ux = torch.zeros(shape, dtype=self.dtype, device=self.device)
        uy = torch.zeros_like(ux)
        uz = torch.zeros_like(ux)
        if cfg.perturbation_amplitude > 0.0:
            px, py, pz = self._spectral_perturbation()
            ux, uy, uz = ux + px, uy + py, uz + pz
            # log-law-like mean guess: a starting condition only (it decays away)
            h = cfg.half_height
            jj = torch.arange(self.ny, dtype=torch.float64)
            yl = jj - cfg.wall_origin
            yl = torch.minimum(yl.clamp(min=0.0), 2 * h - yl.clamp(min=0.0))
            yp = yl * cfg.dy_plus
            up = torch.where(yp < 5.0, yp, log_law_u_plus(torch.clamp(yp, min=1e-3)))
            ux = ux + (up * cfg.u_tau).to(self.dtype).view(1, self.ny, 1)
            ux[0] = 0.0
            ux[-1] = 0.0
        u = torch.stack((ux, uy, uz)).reshape(3, self.N)
        rho = rho.reshape(1, self.N)
        return self._equilibrium(rho, u).reshape(19, self.nz, self.ny, self.nx)

    def _spectral_perturbation(self):
        """Divergence-free random perturbation from a smooth vector potential."""
        cfg = self.cfg
        amp = cfg.perturbation_amplitude * cfg.u_tau
        kmax = max(1, cfg.perturbation_modes)
        h = cfg.half_height
        nx, ny, nz = self.nx, self.ny, self.nz
        xf = torch.arange(nx, dtype=torch.float64) * (2 * math.pi / nx)
        zf = torch.arange(nz, dtype=torch.float64) * (2 * math.pi / nz)
        jj = torch.arange(ny, dtype=torch.float64)
        yl = (jj - cfg.wall_origin).clamp(min=0.0)
        env = torch.sin(math.pi * yl / (2 * h))  # zero at both walls
        denv = (math.pi / (2 * h)) * torch.cos(math.pi * yl / (2 * h))
        env3 = env.view(1, -1, 1)
        denv3 = denv.view(1, -1, 1)
        ux = torch.zeros(nz, ny, nx, dtype=torch.float64)
        uy = torch.zeros(nz, ny, nx, dtype=torch.float64)
        uz = torch.zeros(nz, ny, nx, dtype=torch.float64)
        for _ in range(max(8, 3 * kmax * kmax)):
            kx = float(torch.randint(-kmax, kmax + 1, (1,), generator=self.gen))
            kz = float(torch.randint(-kmax, kmax + 1, (1,), generator=self.gen))
            if kx == 0 and kz == 0:
                continue
            ph = torch.rand(3, generator=self.gen, dtype=torch.float64) * 2 * math.pi
            kx_, kz_ = kx * 2 * math.pi / nx, kz * 2 * math.pi / nz
            arg = kx_ * xf.view(1, -1) + kz_ * zf.view(-1, 1)

            def cos_of(p):
                return torch.cos(arg + float(p)).view(nz, 1, nx)

            def sin_of(p):
                return torch.sin(arg + float(p)).view(nz, 1, nx)

            iax, iay, iaz = cos_of(ph[0]), cos_of(ph[1]), cos_of(ph[2])
            sax, say, saz = sin_of(ph[0]), sin_of(ph[1]), sin_of(ph[2])
            # A = env(y) * [cos(kx x + kz z + phi_i)]_i ; u = curl A (divergence free)
            # d/dy A_i = denv * cos_i ; d/dx -> -kx sin_i ; d/dz -> -kz sin_i
            ux += denv3 * iaz - kz_ * env3 * say
            uy += kx_ * env3 * saz - kz_ * env3 * sax
            uz += -kx_ * env3 * say - denv3 * iax
        norm = math.sqrt(float((ux**2 + uy**2 + uz**2).mean()))
        scale = amp / max(norm, 1e-30)
        ux = ux * scale * env3
        uy = uy * scale * env3
        uz = uz * scale * env3
        return ux.to(self.dtype), uy.to(self.dtype), uz.to(self.dtype)

    # ---- kernels ---------------------------------------------------------
    def _equilibrium(self, rho: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """``rho`` (1,N), ``u`` (3,N) -> ``feq`` (19,N)."""
        cu = torch.mm(self.Ct, u)  # (19,N)
        usq = (u * u).sum(0)
        t = cu.mul_(3.0).add_(4.5 * cu * cu)
        t.sub_((1.5 * usq).expand_as(t)).add_(1.0)
        return (self.wcol * rho).mul_(t)

    def step(self, n: int = 1) -> None:
        """Advance ``n`` time steps."""
        for _ in range(n):
            self._step_once()

    def _allocate_buffers(self) -> None:
        """Pre-allocate every scratch array: per-step malloc churn dominated the cost."""
        dt, dev, N = self.dtype, self.device, self.N
        self.b_m = torch.zeros(4, N, dtype=dt, device=dev)
        self.b_u = torch.zeros(3, N, dtype=dt, device=dev)
        self.b_inv = torch.zeros(1, N, dtype=dt, device=dev)
        self.b_usq = torch.zeros(N, dtype=dt, device=dev)
        self.b_cu = torch.zeros(19, N, dtype=dt, device=dev)
        self.b_t = torch.zeros(19, N, dtype=dt, device=dev)
        self.b_tn = torch.zeros(N, dtype=dt, device=dev)
        self.b_feq = torch.zeros(19, N, dtype=dt, device=dev)
        self.b_fneq = torch.zeros(19, N, dtype=dt, device=dev)
        self.b_fopp = torch.zeros(19, N, dtype=dt, device=dev)
        self.b_fpost = torch.zeros(19, N, dtype=dt, device=dev)

    def _step_once(self) -> None:
        cfg = self.cfg
        f2 = self.f.view(19, self.N)
        m, u = self.b_m, self.b_u
        torch.mm(self.A, f2, out=m)               # rho, mom_x, mom_y, mom_z
        rho = m[0]
        torch.reciprocal(rho, out=self.b_inv[0])
        inv = self.b_inv[0]
        torch.mul(m[1], inv, out=u[0])
        u[0].add_(0.5 * self.accel)
        torch.mul(m[2], inv, out=u[1])
        torch.mul(m[3], inv, out=u[2])

        cu, t = self.b_cu, self.b_t
        torch.mm(self.Ct, u, out=cu)
        usq = self.b_usq
        torch.mul(u[0], u[0], out=usq)
        usq.addcmul_(u[1], u[1]).addcmul_(u[2], u[2])
        torch.mul(cu, 3.0, out=t)
        t.addcmul_(cu, cu, value=4.5)              # t = 3cu + 4.5 cu^2
        t.sub_(usq, alpha=1.5).add_(1.0)
        feq = self.b_feq
        torch.mul(self.wcol, rho, out=feq)
        feq.mul_(t)

        fneq, fopp, fpost = self.b_fneq, self.b_fopp, self.b_fpost
        torch.sub(f2, feq, out=fneq)
        torch.index_select(fneq, 0, self.opp, out=fopp)

        if cfg.smagorinsky_cs > 0.0:
            nu_t = self._smagorinsky(fneq, rho)
            # flat (N,) so the downstream force block can reuse pre-allocated
            # scratch tensors without a shape-changing ``out=`` (see b_tn)
            sp = 1.0 / (0.5 + 3.0 * (self.nu + nu_t))
            sm = 1.0 / (0.5 + TRT_MAGIC / (sp - 0.5))
        else:
            sp, sm = self.sp, self.sm
        ab = 1.0 - 0.5 * sp          # symmetric force weight
        amb = ab if cfg.force_scheme == "guo_sp" else 1.0 - 0.5 * sm

        fpost.copy_(f2)
        if isinstance(sp, torch.Tensor):
            # Variable-viscosity (LES) branch: torch's scalar ``alpha`` only
            # accepts numbers, so the SGS weights must be applied element-wise.
            # ``sp``/``sm`` are per-cell relaxation rates there.
            fpost.addcmul_(fneq, sp + sm, value=-0.5)
            fpost.addcmul_(fopp, sm - sp, value=0.5)  # -0.5(sp - sm)
        else:
            fpost.add_(fneq, alpha=-0.5 * (sp + sm))
            fpost.add_(fopp, alpha=0.5 * (sm - sp))   # -0.5(sp - sm)

        # ---- Guo forcing, closed form over (i, opposite-i) pairs ----
        #   w fx [ 3 cx (1 - sm/2) + 9 cu cx (1 - sp/2) - 3 ux (1 - sp/2) ]
        #   sum_i (i, opposite-i):  w fx [ 3 amb c_ix + 9 amb cu c_ix - 3 ab ux ]
        fx = self.accel * rho
        coef9 = (9.0 * amb) * fx
        cu.add_(1.0 / 3.0)
        cu.mul_(self.dcol)
        cu.mul_(coef9.view(1, self.N))
        torch.mul(u[0], 3.0 * ab, out=self.b_tn).mul_(fx)
        cu.sub_(self.wcol * self.b_tn)
        fpost.add_(cu)

        # ---- streaming ----
        # streaming: the collision output lives in a scratch buffer, so the new
        # populations can be written straight into the (now dead) current array
        fpost3 = fpost.view(19, self.nz, self.ny, self.nx)
        out = self.f
        for i in range(19):
            dst = out[i]
            src = fpost3[i]
            for d_sl, s_sl in self._stream_plan[i]:
                dst[d_sl] = src[s_sl]
        self._bounce_back(fpost3)
        self.step_count += 1

    def _smagorinsky(self, fneq: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        """Local-equilibrium Smagorinsky viscosity switch (model on/off control)."""
        cfg = self.cfg
        fneq3 = fneq.view(19, self.nz, self.ny, self.nx)
        r3 = rho.view(self.nz, self.ny, self.nx)
        cx, cy, cz = self.cx, self.cy, self.cz
        pixy = (fneq3 * (cx * cy).view(-1, 1, 1, 1)).sum(0) / r3
        pixx = (fneq3 * (cx * cx).view(-1, 1, 1, 1)).sum(0) / r3
        piyy = (fneq3 * (cy * cy).view(-1, 1, 1, 1)).sum(0) / r3
        pixz = (fneq3 * (cx * cz).view(-1, 1, 1, 1)).sum(0) / r3
        piyz = (fneq3 * (cy * cz).view(-1, 1, 1, 1)).sum(0) / r3
        pizz = (fneq3 * (cz * cz).view(-1, 1, 1, 1)).sum(0) / r3
        q = torch.sqrt(2.0 * (pixy**2 + pixz**2 + piyz**2) + (pixx**2 + piyy**2 + pizz**2))
        nu_t = torch.zeros_like(q)
        for _ in range(3):
            nu_t = (cfg.smagorinsky_cs**2) * q / (2.0 * (self.nu + nu_t)).clamp(min=1e-12)
        return nu_t.clamp(max=50.0 * self.nu).reshape(self.N)

    def _bounce_back(self, fpost: torch.Tensor) -> None:
        """Bouzidi interpolated bounce-back + wall momentum-exchange flux."""
        q = self.q
        f = self.f
        up, dn = self.idx_up, self.idx_dn
        ny = self.ny
        if q < 0.5:
            a, b = 2.0 * q, 1.0 - 2.0 * q
            f[up, :, 1, :] = a * fpost[dn, :, 1, :] + b * fpost[dn, :, 2, :]
            f[dn, :, ny - 2, :] = a * fpost[up, :, ny - 2, :] + b * fpost[up, :, ny - 3, :]
        else:
            a = 1.0 / (2.0 * q)
            b = (2.0 * q - 1.0) / (2.0 * q)
            f[up, :, 1, :] = a * fpost[dn, :, 1, :] + b * fpost[up, :, 1, :]
            f[dn, :, ny - 2, :] = a * fpost[up, :, ny - 2, :] + b * fpost[dn, :, ny - 2, :]
        # momentum-exchange flux (exact for q = 0.5); >0 means the fluid gains
        # streamwise momentum from the wall
        self._wall_flux_accum += float(
            (self.cx[up].view(-1, 1, 1) * (f[up, :, 1, :] + fpost[dn, :, 1, :])).sum()
        ) + float(
            (
                self.cx[dn].view(-1, 1, 1)
                * (f[dn, :, ny - 2, :] + fpost[up, :, ny - 2, :])
            ).sum()
        )

    # ---- diagnostics -----------------------------------------------------
    def health(self) -> dict:
        f2 = self.f.view(19, self.N)
        m = torch.mm(self.A, f2)
        rho = m[0]
        spd = torch.sqrt((m[1:] ** 2).sum(0)) / rho
        return {
            "step": self.step_count,
            "finite": bool(torch.isfinite(f2).all().item()),
            "rho_min": float(rho.min()),
            "rho_max": float(rho.max()),
            "max_speed": float(spd.max()),
            "max_mach": float(spd.max() / math.sqrt(1.0 / 3.0)),
        }

    def turbulence_energy(self) -> dict:
        """Domain-averaged Reynolds stress and TKE about the horizontal mean."""
        f2 = self.f.view(19, self.N)
        m = torch.mm(self.A, f2)
        rho = m[0]
        ux = (m[1] / rho).view(self.nz, self.ny, self.nx)
        uy = (m[2] / rho).view(self.nz, self.ny, self.nx)
        uz = (m[3] / rho).view(self.nz, self.ny, self.nx)
        n2 = float(self.nx * self.nz)
        mx = ux.sum(dim=(0, 2)) / n2
        my = uy.sum(dim=(0, 2)) / n2
        mz = uz.sum(dim=(0, 2)) / n2
        fl = slice(1, self.ny - 1)
        fx = ux - mx.view(1, -1, 1)
        fy = uy - my.view(1, -1, 1)
        fz = uz - mz.view(1, -1, 1)
        tke = 0.5 * float((fx * fx + fy * fy + fz * fz)[:, fl, :].mean())
        uv = float((fx * fy)[:, fl, :].mean())
        return {"tke": tke, "uv_mean": uv}

    def bulk_velocity(self) -> float:
        m = torch.mm(self.A, self.f.view(19, self.N))
        return float((m[1] / m[0]).mean())

    # ---- production run --------------------------------------------------
    def run(self) -> dict:
        cfg = self.cfg
        ny, nx, nz, N = self.ny, self.nx, self.nz, self.N
        sums = torch.zeros(ny, 8, dtype=torch.float64)  # rho ux uy uz ux2 uy2 uz2 uxuy
        samples = 0
        wall_flux = 0.0
        history = []
        t0 = time.time()
        reports = []
        for step in range(1, cfg.steps + 1):
            self._step_once()
            if step <= cfg.warmup_steps:
                # The averaging window opens after warmup.  The momentum-exchange
                # flux of a warmup step must not reach the sampled average: it is
                # divided by ``sample_interval`` steps, so a book kept from step 1
                # inflates tau_w by (warmup + interval) / interval (e.g. x51 at
                # warmup=5000, interval=100).
                self._wall_flux_accum = 0.0
            if step % cfg.report_interval == 0 or step == 1:
                h = self.health()
                reports.append(h)
                if not h["finite"]:
                    raise FloatingPointError(f"divergence at step {step}: {h}")
                history.append({"step": step, "u_bulk": self.bulk_velocity(), **h, **self.turbulence_energy()})
                if step == 1 or step % (5 * cfg.report_interval) == 0:
                    print(
                        f"[{cfg.label}] {step}/{cfg.steps} Ma={h['max_mach']:.3f} "
                        f"rho=[{h['rho_min']:.4f},{h['rho_max']:.4f}] "
                        f"u_bulk={history[-1]['u_bulk']:.4f} "
                        f"{step / (time.time() - t0):.2f} steps/s",
                        flush=True,
                    )
            if step > cfg.warmup_steps and (step - cfg.warmup_steps) % cfg.sample_interval == 0:
                f2 = self.f.view(19, N)
                m = torch.mm(self.A, f2)
                rho = m[0]
                ux = m[1] / rho
                uy = m[2] / rho
                uz = m[3] / rho
                wgt = 1.0 / (nx * nz)
                sums[:, 0] += rho.view(nz, ny, nx).sum(dim=(0, 2)).double() * wgt
                sums[:, 1] += ux.view(nz, ny, nx).sum(dim=(0, 2)).double() * wgt
                sums[:, 2] += uy.view(nz, ny, nx).sum(dim=(0, 2)).double() * wgt
                sums[:, 3] += uz.view(nz, ny, nx).sum(dim=(0, 2)).double() * wgt
                sums[:, 4] += (ux * ux).view(nz, ny, nx).sum(dim=(0, 2)).double() * wgt
                sums[:, 5] += (uy * uy).view(nz, ny, nx).sum(dim=(0, 2)).double() * wgt
                sums[:, 6] += (uz * uz).view(nz, ny, nx).sum(dim=(0, 2)).double() * wgt
                sums[:, 7] += (ux * uy).view(nz, ny, nx).sum(dim=(0, 2)).double() * wgt
                samples += 1
                wall_flux += self._wall_flux_accum
                self._wall_flux_accum = 0.0
        elapsed = time.time() - t0
        stats = sums / max(samples, 1)
        return self._finalize(stats, samples, wall_flux, elapsed, history)

    # ---- post-processing -------------------------------------------------
    def _fold(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Fold the channel about mid-plane (statistical symmetry) -> (y, value)."""
        cfg = self.cfg
        ny = self.ny
        rows = list(range(1, ny - 1))
        y_raw = torch.tensor([j - cfg.wall_origin for j in rows], dtype=torch.float64)
        upper = torch.tensor([(ny - 1) - cfg.wall_origin - j for j in rows], dtype=torch.float64)
        y = torch.minimum(y_raw, upper)
        order = torch.argsort(y)
        vals = values[1:-1][order]
        ys = y[order]
        # average duplicated (mirror) rows
        out_y, out_v = [], []
        i = 0
        n = ys.numel()
        while i < n:
            j = i + 1
            key = ys[i]
            acc = vals[i].clone()
            cnt = 1.0
            while j < n and abs(float(ys[j] - key)) < 1e-9:
                acc += vals[j]
                cnt += 1.0
                j += 1
            out_y.append(float(key))
            out_v.append(acc / cnt)
            i = j
        return torch.tensor(out_y, dtype=torch.float64), torch.stack(out_v)

    def _finalize(self, stats, samples, wall_flux, elapsed, history) -> dict:
        cfg = self.cfg
        h = cfg.half_height
        H = cfg.height
        u_tau_bf = math.sqrt(cfg.acceleration * h)
        nu = self.nu
        ux = stats[:, 1]
        uy = stats[:, 2]
        uz = stats[:, 3]
        uu = (stats[:, 4] - ux**2).clamp(min=0.0)
        vv = (stats[:, 5] - uy**2).clamp(min=0.0)
        ww = (stats[:, 6] - uz**2).clamp(min=0.0)
        uv = stats[:, 7] - ux * uy

        y, prof = self._fold(torch.stack([ux, uu, vv, ww, uv], dim=1))
        ux_f, uu_f, vv_f, ww_f, uv_f = prof[:, 0], prof[:, 1], prof[:, 2], prof[:, 3], prof[:, 4]
        yp = y * cfg.dy_plus

        du = torch.zeros_like(ux_f)
        du[1:-1] = (ux_f[2:] - ux_f[:-2]) / (y[2:] - y[:-2])
        du[0] = (ux_f[1] - ux_f[0]) / (y[1] - y[0])
        du[-1] = (ux_f[-1] - ux_f[-2]) / (y[-1] - y[-2])
        total = nu * du - uv_f

        sel = (y > 0.08 * H) & (y < 0.70 * H)
        if int(sel.sum()) >= 3:
            A = torch.stack([torch.ones(int(sel.sum()), dtype=torch.float64), y[sel]], dim=1)
            coef = torch.linalg.lstsq(A, total[sel]).solution
            u_tau_fit = math.sqrt(max(float(coef[0]), 0.0))
        else:
            coef = torch.zeros(2, dtype=torch.float64)
            u_tau_fit = float("nan")

        area = float(self.nx * self.nz)
        n_flux_steps = max(samples, 1) * max(cfg.sample_interval, 1)
        tau_w_exchange = -wall_flux / (2.0 * area * n_flux_steps)  # two walls, drag on the fluid
        u_tau_exchange = math.sqrt(max(tau_w_exchange, 0.0))

        bal = (total / u_tau_bf**2) - (1.0 - y / h)
        bal_sel = (yp > 10.0) & (y < 0.8 * h)
        balance_err = float(bal[bal_sel].abs().max()) if int(bal_sel.sum()) else float("nan")

        uv_minus = -uv_f
        ipk = int(torch.argmax(uv_minus))
        k_plus = 0.5 * (uu_f + vv_f + ww_f) / u_tau_bf**2
        ik = int(torch.argmax(k_plus))

        return {
            "label": cfg.label,
            "config": {
                "nx": cfg.nx,
                "ny": cfg.ny,
                "nz": cfg.nz,
                "height": H,
                "half_height": h,
                "re_tau_target": cfg.re_tau,
                "u_tau_target": cfg.u_tau,
                "nu": nu,
                "tau_plus": cfg.tau_plus,
                "tau_minus": cfg.tau_minus,
                "acceleration": cfg.acceleration,
                "wall_origin": cfg.wall_origin,
                "wall_scheme": cfg.wall_scheme,
                "dy_plus": cfg.dy_plus,
                "steps": cfg.steps,
                "warmup_steps": cfg.warmup_steps,
                "sample_interval": cfg.sample_interval,
                "samples": samples,
                "eddy_turnovers_averaged": (cfg.steps - cfg.warmup_steps) / cfg.eddy_turnover_steps,
                "domain_lx_over_h": cfg.nx / h,
                "domain_lz_over_h": cfg.nz / h,
                "smagorinsky_cs": cfg.smagorinsky_cs,
                "dtype": cfg.dtype,
                "cells": cfg.nx * cfg.ny * cfg.nz,
                "seed": cfg.seed,
            },
            "wall_units": {
                "y": y.tolist(),
                "y_plus": yp.tolist(),
                "u_plus": (ux_f / u_tau_bf).tolist(),
                "u_mean": ux_f.tolist(),
                "uu_plus": (uu_f / u_tau_bf**2).tolist(),
                "vv_plus": (vv_f / u_tau_bf**2).tolist(),
                "ww_plus": (ww_f / u_tau_bf**2).tolist(),
                "uv_plus": (uv_f / u_tau_bf**2).tolist(),
                "uv_minus_plus": (uv_minus / u_tau_bf**2).tolist(),
                "k_plus": k_plus.tolist(),
                "total_stress_plus": (total / u_tau_bf**2).tolist(),
                "dudy": du.tolist(),
                "u_tau_used": u_tau_bf,
            },
            "reynolds_stress": {
                "uv_minus_peak_plus": float(uv_minus[ipk] / u_tau_bf**2),
                "uv_minus_peak_y_plus": float(yp[ipk]),
                "k_peak_plus": float(k_plus[ik]),
                "k_peak_y_plus": float(yp[ik]),
            },
            "momentum_balance": {
                "max_abs_error_plus": balance_err,
                "reference": "total_stress_plus = 1 - y/h",
            },
            "friction_velocity": {
                "body_force": u_tau_bf,
                "wall_exchange": u_tau_exchange,
                "stress_fit": u_tau_fit,
                "target": cfg.u_tau,
                "wall_exchange_tau_w": tau_w_exchange,
                "body_force_tau_w": cfg.acceleration * h,
            },
            "stress_fit": {"intercept": float(coef[0]), "slope": float(coef[1])},
            "mean_velocity_bulk": float(ux[1:-1].mean()),
            "reynolds_number_bulk": float(ux[1:-1].mean() * H / nu),
            "re_tau_achieved": float(u_tau_bf * h / nu),
            "elapsed_seconds": elapsed,
            "steps_per_second": cfg.steps / max(elapsed, 1e-9),
            "history": history,
        }


def run_turbulent_channel3d(config: TurbulentChannel3DConfig) -> dict:
    """Build the solver, integrate, and (optionally) dump the diagnostics to JSON."""
    sim = TurbulentChannel3D(config)
    result = sim.run()
    if config.output is not None:
        config.output.parent.mkdir(parents=True, exist_ok=True)
        config.output.write_text(json.dumps(result, indent=1))
    return result


def run_laminar_poiseuille(
    config: TurbulentChannel3DConfig,
    steps: int | None = None,
    u_max_target: float | None = 0.05,
    nu_target: float = 0.05,
) -> dict:
    """Laminar twin case: exact Poiseuille steady state, no perturbation.

    Two purposes: (i) measure the *realised* wall origin of the bounce-back
    scheme by fitting the analytic parabola with a free offset, and (ii) serve
    as the negative control ("no model" -> laminar solution recovered).

    The wall origin is a *geometric* property of the boundary scheme, so the twin
    does not have to -- and must not -- run at the campaign Reynolds number: with
    the campaign's own ``u_tau``/``re_tau`` the laminar peak velocity is
    ``u_max = u_tau re_tau / 2`` (0.9 at Re_tau = 180, i.e. Ma = 1.56), which
    diverges to NaN (see
    ``tests/test_turbulent_channel3d.py::test_legacy_twin_parameterisation_is_hypersonic``).
    ``u_max_target`` therefore re-derives a subsonic twin,
    ``re_tau_twin = sqrt(2 u_max h / nu_target)`` and
    ``u_tau_twin = 2 u_max / re_tau_twin``, while ``ny``, ``wall_origin``, ``q``
    and ``wall_scheme`` of the campaign are kept bit-identical.  Pass
    ``u_max_target=None`` to use ``config`` verbatim (legacy behaviour; only
    stable for ``u_tau re_tau / 2 <~ 0.1``).
    """
    cfg = config
    if u_max_target is None:
        twin = cfg
    else:
        h_twin = cfg.half_height
        re_tau_twin = math.sqrt(2.0 * u_max_target * h_twin / nu_target)
        u_tau_twin = 2.0 * u_max_target / re_tau_twin
        twin = TurbulentChannel3DConfig(
            nx=cfg.nx,
            nz=cfg.nz,
            ny=cfg.ny,
            re_tau=re_tau_twin,
            u_tau=u_tau_twin,
            wall_origin=cfg.wall_origin,
            wall_scheme=cfg.wall_scheme,
            steps=cfg.steps,
            warmup_steps=0,
            sample_interval=1,
            perturbation_amplitude=0.0,
            dtype=cfg.dtype,
            device=cfg.device,
            force_scheme=cfg.force_scheme,
            smagorinsky_cs=0.0,
            label=cfg.label + "_laminar",
            output=None,
        )
    ny, nx, nz = twin.ny, twin.nx, twin.nz
    h = twin.half_height
    y = torch.arange(ny, dtype=torch.float64) - twin.wall_origin
    y = torch.minimum(y, twin.height - y)
    u = poiseuille_exact(y.clamp(min=0.0), twin.height, twin.acceleration, twin.nu)
    u[0] = 0.0
    u[-1] = 0.0
    nsteps = twin.steps if steps is None else steps
    sim = TurbulentChannel3D(
        TurbulentChannel3DConfig(
            nx=nx,
            nz=nz,
            ny=ny,
            re_tau=twin.re_tau,
            u_tau=twin.u_tau,
            wall_origin=twin.wall_origin,
            wall_scheme=twin.wall_scheme,
            steps=nsteps,
            warmup_steps=0,
            sample_interval=1,
            perturbation_amplitude=0.0,
            dtype=twin.dtype,
            device=twin.device,
            force_scheme=twin.force_scheme,
            smagorinsky_cs=0.0,
            label=twin.label,
            output=None,
        )
    )
    rho = torch.ones(nz, ny, nx, dtype=sim.dtype)
    ux = u.to(sim.dtype).view(1, ny, 1).expand(nz, ny, nx).contiguous()
    zeros = torch.zeros_like(rho)
    uvec = torch.stack((ux, zeros, zeros)).reshape(3, sim.N)
    sim.f = sim._equilibrium(rho.reshape(1, sim.N), uvec).reshape(19, nz, ny, nx)

    samp = torch.zeros(ny, dtype=torch.float64)
    count = 0
    flux = 0.0
    flux_steps = 0
    last_flux_step = nsteps // 2
    t0 = time.time()
    for step in range(1, nsteps + 1):
        sim._step_once()
        if step == nsteps // 2:
            # Same bookkeeping rule as ``TurbulentChannel3D.run``: the flux
            # accumulated during spin-up must not be divided by the (shorter)
            # averaging window.  Keeping it inflated tau_w by nsteps//(window),
            # i.e. exactly x2 for the standard half/whole twin split (u_tau x
            # sqrt(2) = 42 %), which is the artefact reported in
            # docs/turbulent_channel3d_verification_r1.md section 5.1.
            sim._wall_flux_accum = 0.0
        if step > nsteps // 2 and step % 25 == 0:
            m = torch.mm(sim.A, sim.f.view(19, sim.N))
            ux_t = (m[1] / m[0]).view(nz, ny, nx)
            samp += ux_t.sum(dim=(0, 2)).double() / (nx * nz)
            count += 1
            flux += sim._wall_flux_accum
            sim._wall_flux_accum = 0.0
            flux_steps += step - last_flux_step
            last_flux_step = step
    u_mean = (samp / max(count, 1))[1:-1]
    yf = y[1:-1]
    analytic = poiseuille_exact(yf, twin.height, twin.acceleration, twin.nu)
    best = (float("inf"), 0.0)
    for delta in torch.linspace(-0.45, 0.45, 361, dtype=torch.float64):
        yy = (yf + delta).clamp(min=0.0)
        yy = torch.minimum(yy, twin.height - yy)
        cand = poiseuille_exact(yy, twin.height, twin.acceleration, twin.nu)
        err = float(((cand - u_mean) ** 2).sum())
        if err < best[0]:
            best = (err, float(delta))
    rel_l2 = float(torch.sqrt(((u_mean - analytic) ** 2).mean()) / analytic.abs().max())
    tau_w = twin.acceleration * h
    # Numerator and denominator must cover the *same* steps: use the measured
    # window length instead of assuming count*25 (they differ whenever the
    # last block is partial).
    n_flux_steps = max(flux_steps, 1)
    tau_w_exchange = -flux / (2.0 * float(nx * nz) * n_flux_steps)
    return {
        "label": twin.label,
        "config": {
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "height": twin.height,
            "u_tau": twin.u_tau,
            "re_tau": twin.re_tau,
            "nu": twin.nu,
            "tau_plus": twin.tau_plus,
            "acceleration": twin.acceleration,
            "imposed_wall_origin": twin.wall_origin,
            "wall_scheme": twin.wall_scheme,
            "force_scheme": twin.force_scheme,
            "steps": nsteps,
            "u_max_target": u_max_target,
        },
        "y": yf.tolist(),
        "u_mean": u_mean.tolist(),
        "u_analytic": analytic.tolist(),
        "u_max_mean": float(u_mean.max()),
        "u_max_analytic": float(analytic.max()),
        "relative_l2_error": rel_l2,
        "wall_origin_measured": twin.wall_origin + best[1],
        "wall_origin_shift": best[1],
        "wall_origin_fit_residual": math.sqrt(best[0] / max(len(yf), 1)),
        "u_tau_body_force": math.sqrt(tau_w),
        "u_tau_wall_exchange": math.sqrt(max(tau_w_exchange, 0.0)),
        "elapsed_seconds": time.time() - t0,
    }
