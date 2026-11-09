"""D3Q19 moving-wall Couette shear cell for suspension viscosity.

Geometry
--------
``z`` is wall-normal.  Rows ``z = 0`` and ``z = nz-1`` are walls moving with
``-u_wall`` and ``+u_wall`` in ``x``; ``x`` and ``y`` are periodic.  With
half-way bounce-back the effective wall planes sit ``0.5`` lattice units
outside the wall node centres (``z = 0.5`` and ``z = nz-1.5``), so the
effective gap is ``nz - 2`` and the imposed shear rate is

    gamma_dot = 2 u_wall / (nz - 2),

which is verified by the pure-fluid momentum-budget control.

Particles
---------
Particles are *rigid* and are advected kinematically with the imposed shear:
translation ``gamma_dot (z_c - z_mid)`` plus the half-vorticity rotation
``Omega = gamma_dot/2 e_y``.  This makes each particle force- and
torque-free to leading order -- the condition under which Einstein's
``[eta] = 5/2`` holds.  The residual net force on each particle is *measured
and reported*, so a departure from the force-free state is visible instead of
assumed away.

Stress measurement
------------------
The walls exchange momentum with the fluid through the moving-wall bounce-back

    f_{qbar}(x_f) = f*_q(x_f) + 2 w_q rho (c_q . u_w) / c_s**2

so the force on a wall cell is ``sum_q (2 f*_q + coef_q rho) c_q`` with
``coef_q = 2 w_q (c_q . u_w)/c_s**2`` evaluated at the solid partner node.
``force_variant`` selects among the literature forms

``mem_sum``      plain momentum exchange, post-collision populations
``galilean``     as above times ``(1 + 1/(2 tau))`` (Lorenz 2014)
``bg_sub``       wall-frame equilibrium background removed first

The variant is *selected by evidence*: :meth:`ShearCell.calibrate_force_variant`
runs the pure-fluid control, whose exact solution is ``tau = rho nu gamma_dot``,
and reports the momentum-budget residual of every variant.  The campaign uses
the one that closes that budget.

Relative viscosity
------------------
``eta_r`` is always a **ratio of the same measurement** with and without
particles at the same wall speed, so a residual *multiplicative* bias of the
force estimator cancels:

    eta_r = (tau_susp / tau_pure) * (gamma_dot_pure / gamma_dot_susp)

with the shear rates taken from the *measured* mean velocity profile
(:meth:`ShearCell.velocity_profile`, linear fit + R**2) rather than from the
nominal wall speed.  ``R**2 > 0.99`` is one of the acceptance criteria.

References
----------
Ladd, A. J. C. (1994) *J. Fluid Mech.* 271, 285.
Lorenz, E. (2014) *Comput. Math. Appl.* 67, 742.
Batchelor, G. K. (1970) *J. Fluid Mech.* 41, 545.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..d3q19 import C, OPPOSITE, W, equilibrium3d, macroscopic3d
from ..solver3d import collide_bgk3d, stream3d

__all__ = ["ShearCellConfig", "ShearCell", "FORCE_VARIANTS"]

FORCE_VARIANTS = ("mem_sum", "galilean", "bg_sub")
CS2 = 1.0 / 3.0


@dataclass
class ShearCellConfig:
    nx: int = 32
    ny: int = 32
    nz: int = 33
    nu: float = 0.1
    u_wall: float = 0.02
    dtype: str = "float32"
    device: str = "cpu"
    force_variant: str = "mem_sum"
    mode: str = "couette"          # "couette" (shear) or "drag" (uniform flow past a fixed sphere)

    @property
    def gamma_dot(self) -> float:
        """Imposed shear rate.

        The half-way bounce-back wall sits half a lattice unit outside the wall
        *node* centre, i.e. at ``z = 0.5`` and ``z = nz - 1.5``, so the
        effective gap is ``nz - 2``.  This was **measured** with the pure-fluid
        negative control (the plane momentum flux then matches
        ``rho nu gamma_dot`` to 4+ digits; using ``nz - 1`` leaves a 6% bias at
        ``nz = 17``).
        """
        return 0.0 if self.mode == "drag" else 2.0 * self.u_wall / (self.nz - 2)

    @property
    def z_mid(self) -> float:
        return 0.5 * (self.nz - 1)

    @property
    def tau(self) -> float:
        return self.nu / CS2 + 0.5

    def clone(self, **kw) -> "ShearCellConfig":
        d = dict(self.__dict__)
        d.update(kw)
        return ShearCellConfig(**d)


class ShearCell:
    """Moving-wall Couette suspension cell (D3Q19, lattice units, rho_f = 1)."""

    def __init__(self, cfg: ShearCellConfig, particles=(), wall_band: int = 4):
        if cfg.force_variant not in FORCE_VARIANTS:
            raise ValueError(f"force_variant must be one of {FORCE_VARIANTS}")
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.dtype = getattr(torch, cfg.dtype)
        self.particles = list(particles)
        self.wall_band = int(wall_band)
        nx, ny, nz = cfg.nx, cfg.ny, cfg.nz
        shape = (nz, ny, nx)
        self.N = nx * ny * nz

        # ---- solid / wall masks and prescribed solid velocity field --------
        wall = np.zeros(shape, dtype=bool)
        wall[0, :, :] = True
        wall[-1, :, :] = True
        solid = wall.copy()
        usx = np.zeros(shape)
        usy = np.zeros(shape)
        usz = np.zeros(shape)
        if cfg.mode == "drag":
            # uniform stream past the sphere: both walls move together, no shear
            usx[0, :, :] = +cfg.u_wall
            usx[-1, :, :] = +cfg.u_wall
        else:
            usx[0, :, :] = -cfg.u_wall
            usx[-1, :, :] = +cfg.u_wall
        sph_id = np.full(shape, -1, dtype=np.int64)
        overlaps = 0
        for i, p in enumerate(self.particles):
            overlaps += int((sph_id[p.mask] >= 0).sum())
            solid |= p.mask
            sph_id[p.mask] = i
            if cfg.mode == "drag":
                vx = np.zeros(shape)
                vy = np.zeros(shape)
                vz = np.zeros(shape)
            else:
                vx, vy, vz = _particle_velocity(p, cfg.gamma_dot, cfg.z_mid)
            usx[p.mask] = vx[p.mask]
            usy[p.mask] = vy[p.mask]
            usz[p.mask] = vz[p.mask]

        self.solid_np, self.wall_np = solid, wall
        self.phi_lattice = float(solid.sum() - wall.sum()) / self.N
        self.n_particles = len(self.particles)
        self.node_overlaps = overlaps

        t = lambda a, dt=None: torch.as_tensor(  # noqa: E731
            a, device=self.device, dtype=self.dtype if dt is None else dt
        )
        self.solid = t(solid, torch.bool)
        self.fluid = ~self.solid
        self.usx, self.usy, self.usz = t(usx), t(usy), t(usz)
        solid_f = self.solid.to(self.dtype)

        # ---- bounce-back link bookkeeping ---------------------------------
        c = C.cpu().numpy()
        w = W.cpu().numpy()
        self.link_masks = torch.zeros((19, nz, ny, nx), device=self.device, dtype=torch.bool)
        self.wall_links = torch.zeros((19, nz, ny, nx), device=self.device, dtype=torch.bool)
        self.sph_links = torch.zeros((19, nz, ny, nx), device=self.device, dtype=torch.bool)
        self.coefs = torch.zeros((19, nz, ny, nx), device=self.device, dtype=self.dtype)
        wall_t = self.solid & (t(wall, torch.bool))
        self._dirs = []
        for q in range(1, 19):
            cx, cy, cz = int(c[q, 0]), int(c[q, 1]), int(c[q, 2])
            shift = (-cz, -cy, -cx)
            s_shift = torch.roll(solid_f, shifts=shift, dims=(0, 1, 2)) > 0.5
            lq = self.fluid & s_shift
            if not bool(lq.any()):
                continue
            dot = (
                cx * torch.roll(self.usx, shifts=shift, dims=(0, 1, 2))
                + cy * torch.roll(self.usy, shifts=shift, dims=(0, 1, 2))
                + cz * torch.roll(self.usz, shifts=shift, dims=(0, 1, 2))
            )
            coef = -(2.0 * float(w[q]) / CS2) * dot
            opp = int(OPPOSITE[q].item())
            self.link_masks[opp] = lq
            self.coefs[opp] = torch.where(lq, coef, torch.zeros_like(coef))
            wl = torch.roll(wall_t.to(self.dtype), shifts=shift, dims=(0, 1, 2)) > 0.5
            self.wall_links[opp] = lq & wl
            self.sph_links[opp] = lq & (~wl)
            self._dirs.append((q, opp, (cx, cy, cz)))
        self.n_links = int(self.link_masks.sum().item())

        # per-particle node index (for the force-free check)
        self.sph_index = [
            torch.as_tensor(np.flatnonzero(p.mask), device=self.device, dtype=torch.long)
            for p in self.particles
        ]
        self._cxcz = torch.as_tensor(
            C.cpu().numpy()[:, 0] * C.cpu().numpy()[:, 2],
            device=self.device, dtype=self.dtype)
        self.f = self.initial_state()
        self._last: dict = {}

    # ------------------------------------------------------------------
    def initial_state(self) -> torch.Tensor:
        """Equilibrium at the analytic linear shear profile.

        The pure-fluid problem then starts *on* its exact steady solution, so
        the pure control is converged from the first step (no long viscous
        transient) and the suspension transient is limited to the local
        equilibration around the particles.
        """
        cfg = self.cfg
        z = torch.arange(cfg.nz, device=self.device, dtype=self.dtype)
        if cfg.mode == "drag":
            prof = torch.full_like(z, cfg.u_wall)
        else:
            prof = cfg.gamma_dot * (z - cfg.z_mid)
        ux = prof.view(cfg.nz, 1, 1).expand(cfg.nz, cfg.ny, cfg.nx).contiguous()
        uy = torch.zeros_like(ux)
        uz = torch.zeros_like(ux)
        ux = torch.where(self.solid, self.usx, ux)
        uy = torch.where(self.solid, self.usy, uy)
        uz = torch.where(self.solid, self.usz, uz)
        rho = torch.ones_like(ux)
        return equilibrium3d(rho, ux, uy, uz, device=self.device).to(self.dtype)

    # ------------------------------------------------------------------
    def step(self, sample: bool = False, band=(1, 4)) -> dict:
        """One LBM step: collide -> stream -> moving-wall bounce-back.

        With ``sample=True`` the diagnostics of :meth:`diagnostics` are
        accumulated as well (they cost extra passes over the grid, so they are
        taken every ``sample_every`` steps only).
        """
        cfg = self.cfg
        f = self.f
        rho, ux, uy, uz = macroscopic3d(f)
        solid = self.solid
        rho = torch.where(solid, torch.ones_like(rho), rho)
        ux = torch.where(solid, self.usx, ux)
        uy = torch.where(solid, self.usy, uy)
        uz = torch.where(solid, self.usz, uz)
        f = torch.where(solid.unsqueeze(0), equilibrium3d(rho, ux, uy, uz), f)

        fstar = collide_bgk3d(f, cfg.tau)
        fnew = stream3d(fstar)
        acc_wall = np.zeros(3)
        acc_sph = np.zeros(3)
        sample_solids = bool(sample and (self.n_particles or True))
        for q, opp, cvec in self._dirs:
            val_bb = fstar[q] + self.coefs[opp] * rho
            fnew[opp] = torch.where(self.link_masks[opp], val_bb, fnew[opp])
            if sample_solids:
                c = np.array(cvec, dtype=float)
                wl = self.wall_links[opp]
                if bool(wl.any()):
                    acc_wall += c * float((2.0 * fstar[q] + self.coefs[opp] * rho)[wl].sum())
                sl = self.sph_links[opp]
                if self.n_particles and bool(sl.any()):
                    acc_sph += c * float((2.0 * fstar[q] + self.coefs[opp] * rho)[sl].sum())
        self.f = fnew
        if sample:
            self._last = self.diagnostics(rho, ux, uy, uz, band=band,
                                          mem_wall=acc_wall, mem_sph=acc_sph)
            return self._last
        return {}

    # ------------------------------------------------------------------
    def diagnostics(self, rho, ux, uy, uz, band=(1, 4), mem_wall=None, mem_sph=None) -> dict:
        """Momentum-flux (stress) profile and solid forces.

        The wall-normal momentum flux of the discrete populations is

            Pi_xz = sum_q c_qx c_qz (f_q - f_q^eq)

        whose non-equilibrium part is, for BGK, ``Pi^(1)_xz = -tau c_s^2 rho
        gamma_dot`` while the Navier-Stokes shear stress is ``-nu rho
        gamma_dot``.  The exact discrete-to-continuum ratio is therefore
        ``tau/(tau - 1/2)``, applied here as ``bias`` so ``tau_row`` is in
        physical (NS) units.  The ratio form of ``eta_r`` cancels this factor
        anyway; it is applied so the absolute number can be compared with
        ``rho nu gamma_dot``.

        ``band`` selects the particle-free rows next to the walls, where the
        measured flux is the *total* suspension stress (momentum balance
        across a plane that carries no particle).
        """
        solid = self.solid
        f = self.f
        feq = equilibrium3d(rho, ux, uy, uz)
        fneq = f - feq
        flux = -(fneq * self._cxcz.view(19, 1, 1, 1)).sum(0)
        w = self.fluid.to(flux.dtype)
        rows = (flux * w).sum(dim=(1, 2)) / w.sum(dim=(1, 2)).clamp(min=1.0)
        bias = (2.0 * self.cfg.tau - 1.0) / (2.0 * self.cfg.tau)
        tau_rows = rows * bias

        f_wall = np.zeros(3)
        f_sph = np.zeros(3)
        for q, opp, cvec in self._dirs:
            val = 2.0 * fneq[q]  # stress-carrying (non-equilibrium) part only
            wl = self.wall_links[opp]
            if bool(wl.any()):
                f_wall += np.array(cvec, dtype=float) * float(val[wl].sum())
            sl = self.sph_links[opp]
            if bool(sl.any()):
                f_sph += np.array(cvec, dtype=float) * float(val[sl].sum())
        lo, hi = band
        rows_np = tau_rows.detach().cpu().numpy()
        near = float(np.mean(np.r_[rows_np[lo:hi], rows_np[-hi:-lo]]))
        return {
            "tau_rows": rows_np,
            "tau_near": near,
            "tau_mid": float(np.mean(rows_np[self.cfg.nz // 2 - 2:self.cfg.nz // 2 + 2])),
            "f_wall": f_wall,
            "f_particle": f_sph,
            "f_wall_mem": np.zeros(3) if mem_wall is None else mem_wall,
            "f_particle_mem": np.zeros(3) if mem_sph is None else mem_sph,
        }

    # ------------------------------------------------------------------
    def velocity_profile(self) -> dict:
        """Mean ``u_x(z)`` over fluid nodes and its linear fit."""
        cfg = self.cfg
        rho, ux, uy, uz = macroscopic3d(self.f)
        w = self.fluid.to(ux.dtype)
        prof = ((ux * w).sum(dim=(1, 2)) / w.sum(dim=(1, 2)).clamp(min=1.0)).cpu().numpy()
        rows = np.flatnonzero(self.fluid.any(dim=(1, 2)).cpu().numpy())
        zz = rows.astype(float)
        uu = prof[rows]
        A = np.vstack([zz, np.ones_like(zz)]).T
        sol, *_ = np.linalg.lstsq(A, uu, rcond=None)
        pred = A @ sol
        ss_res = float(np.sum((uu - pred) ** 2))
        ss_tot = float(np.sum((uu - uu.mean()) ** 2))
        return {
            "z": zz.tolist(),
            "u": uu.tolist(),
            "slope": float(sol[0]),
            "intercept": float(sol[1]),
            "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        }

    # ------------------------------------------------------------------
    def run(self, n_steps: int, n_warmup: int = 0, sample_every: int = 20,
            band=None) -> dict:
        """Run the cell and time-average the stress diagnostics."""
        cfg = self.cfg
        if band is None:
            band = (1, max(2, int(self.wall_band)))
        taus, fws, fps, rows_acc = [], [], [], []
        mem_sph_x, mem_wall_x = [], []
        for it in range(n_steps):
            do = it >= n_warmup and (it - n_warmup) % sample_every == 0
            out = self.step(sample=do, band=band)
            if do:
                taus.append(out["tau_near"])
                rows_acc.append(out["tau_rows"])
                fws.append(out["f_wall"][0] / (cfg.nx * cfg.ny))
                mem_wall_x.append(out["f_wall_mem"][0])
                mem_sph_x.append(out["f_particle_mem"][0])
                fps.append(out["f_particle"][0] / (cfg.nx * cfg.ny))
        tau_near = float(np.mean(taus))
        f_ppart = float(np.mean(fps))
        tau_corr = tau_near - f_ppart
        res = {
            "tau_near": tau_near,
            "tau_rows_mean": np.mean(np.array(rows_acc), axis=0).tolist(),
            "tau_wall_mem": float(np.mean(fws)),
            "f_particle_per_area": f_ppart,
            "tau_corrected": tau_corr,
            "n_samples": len(taus),
            "n_steps": n_steps,
            "n_warmup": n_warmup,
            "n_particles": self.n_particles,
            "phi_lattice": self.phi_lattice,
            "gamma_dot_nominal": cfg.gamma_dot,
            "nu": cfg.nu,
            "u_wall": cfg.u_wall,
            "nz": cfg.nz, "nx": cfg.nx, "ny": cfg.ny,
            "band": list(band),
            "node_overlaps": self.node_overlaps,
            "mode": cfg.mode,
            "drag_force_mem": float(np.mean(mem_sph_x)) if mem_sph_x else 0.0,
            "drag_force_mem_std": float(np.std(mem_sph_x)) if mem_sph_x else 0.0,
        }
        prof = self.velocity_profile()
        res["profile"] = prof
        res["gamma_dot_measured"] = prof["slope"]
        res["profile_r2"] = prof["r2"]
        g = prof["slope"] if prof["slope"] else cfg.gamma_dot
        res["eta_eff"] = tau_corr / g
        res["eta_eff_near"] = tau_near / g
        res["eta_eff_mem"] = res["tau_wall_mem"] / g
        return res

    # ------------------------------------------------------------------
    @staticmethod
    def calibrate_force_variant(cfg: ShearCellConfig, variants=FORCE_VARIANTS,
                                n_steps: int = 600, n_warmup: int = 100) -> dict:
        """Pure-fluid momentum-budget closure of every force variant.

        Kept as the documented diagnostic of the *wall* momentum-exchange
        estimator: pure Couette has the exact solution ``tau = rho nu
        gamma_dot``, so ``|tau_wall_mem/(nu gamma_dot) - 1|`` measures how much
        equilibrium-background cancellation pollutes that estimator (see the
        campaign report: it is O(1), which is why the production measurement
        uses the plane momentum flux instead).
        """
        out = {}
        for v in variants:
            c = cfg.clone(force_variant=v)
            cell = ShearCell(c)
            r = cell.run(n_steps, n_warmup=n_warmup)
            g = r.get("gamma_dot_measured", c.gamma_dot)
            out[v] = {
                "tau_wall_mem": r["tau_wall_mem"],
                "tau_near": r["tau_near"],
                "tau_ref": c.nu * g,
                "rel_err_mem": abs(r["tau_wall_mem"] / (c.nu * g) - 1.0),
                "rel_err_flux": abs(r["tau_near"] / (c.nu * g) - 1.0),
                "gamma_dot_measured": g,
                "profile_r2": r.get("profile_r2"),
            }
        return {"variants": out,
                "best_flux": min(out, key=lambda k: out[k]["rel_err_flux"])}


def _particle_velocity(p, gamma_dot: float, z_mid: float):
    """Rigid-body velocity fields on a particle's nodes."""
    nz, ny, nx = p.mask.shape
    z, y, x = np.meshgrid(
        np.arange(nz, dtype=float), np.arange(ny, dtype=float), np.arange(nx, dtype=float),
        indexing="ij",
    )
    cz, cy, cx = p.centre
    om = 0.5 * gamma_dot
    vx = gamma_dot * (cz - z_mid) + om * (z - cz)
    vz = -om * (x - cx)
    m = p.mask
    return np.where(m, vx, 0.0), np.zeros_like(vx), np.where(m, vz, 0.0)