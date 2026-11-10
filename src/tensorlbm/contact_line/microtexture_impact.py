"""Droplet impact on a micro-textured (pillar-array) wall.

Companion of :mod:`tensorlbm.contact_line.microtexture`.  A drop of radius ``R``
impacts the composite surface with velocity ``U``; the run measures the
inertia-capillary scaling quantities of the super-hydrophobic / anti-icing
literature:

* ``beta_max = D_max / D0`` — maximum spread (lamella width in the band just
  above the pillar tops),
* ``tau_contact`` — residence time, from first solid contact to detachment,
* rebound / adhesion / break-up classification (break-up = more than one liquid
  cluster at any instant),
* the liquid volume left in the grooves after detachment — the anti-icing proxy
  for surface "wetting memory" (a rebound that leaves the texture dry is the
  ice-phobic ideal),

plus a mechanical-energy audit

    KE + E_surf ,   E_surf = sigma * integral |grad phi| dA

with the viscous dissipation ``integral 2 mu S:S dt`` accumulated from the
lattice strain rate (``mu = rho (tau - 1/2)/3``), which closes the impact energy
budget to the reported residual.

References
----------
Clanet, Beguin, Richard & Quere (2004) J. Fluid Mech. 517, 199 (D_max scaling).
Richard, Clanet & Quere (2002) Nature 417, 811 (contact time ~ sqrt(rho R^3/sigma)).
Bird, Tsai & Stone (2013) NJP 15, 063007 (contact time on ridges).
"""

from __future__ import annotations

import math

import numpy as np
import torch

from ..d2q9 import C, equilibrium, macroscopic
from .engine import RHO_L_DEFAULT, RHO_V_DEFAULT, SIGMA_DEFAULT, tanh_profile

__all__ = [
    "ImpactRunner",
    "run_impact",
    "scaling_fits",
]


def _neq_dissipation(f, rho, ux, uy, tau: float, fluid: torch.Tensor) -> float:
    """Viscous dissipation rate from the *non-equilibrium stress* (BGK identity).

    ``Pi^neq_ab = sum_i c_ia c_ib (f_i - f_i^eq)`` and, for BGK with relaxation
    time ``tau``, the viscous stress is ``sigma_ab = -(1 - 1/(2 tau)) Pi^neq_ab``,
    so the dissipation rate per cell is ``-sigma:grad u``.  This avoids dividing
    by the (vanishing) density across the diffuse interface, which makes a
    ``mu * S:S`` estimate blow up at the interface.
    """
    feq = equilibrium(rho, ux, uy)
    df = f - feq
    c = C.to(f.device).float()
    cx = c[:, 0].view(9, 1, 1)
    cy = c[:, 1].view(9, 1, 1)
    pxx = (cx * cx * df).sum(0)
    pyy = (cy * cy * df).sum(0)
    pxy = (cx * cy * df).sum(0)
    dux_dx = 0.5 * (torch.roll(ux, -1, dims=-1) - torch.roll(ux, 1, dims=-1))
    dux_dy = 0.5 * (torch.roll(ux, -1, dims=-2) - torch.roll(ux, 1, dims=-2))
    duy_dx = 0.5 * (torch.roll(uy, -1, dims=-1) - torch.roll(uy, 1, dims=-1))
    duy_dy = 0.5 * (torch.roll(uy, -1, dims=-2) - torch.roll(uy, 1, dims=-2))
    work = pxx * dux_dx + pyy * duy_dy + pxy * (dux_dy + duy_dx)
    return float(((1.0 - 1.0 / (2.0 * tau)) * work)[fluid].sum())


def _grad_magnitude_sum(phi: torch.Tensor, fluid: torch.Tensor) -> float:
    """``integral |grad phi| dA`` (central differences) over fluid cells."""
    gx = 0.5 * (torch.roll(phi, -1, dims=-1) - torch.roll(phi, 1, dims=-1))
    gy = 0.5 * (torch.roll(phi, -1, dims=-2) - torch.roll(phi, 1, dims=-2))
    mag = torch.sqrt(gx * gx + gy * gy)
    return float(mag[fluid].sum())


def _liquid_clusters(phi: torch.Tensor, min_cells: int = 40) -> int:
    """Number of connected liquid clusters with at least ``min_cells`` cells."""
    try:
        from scipy import ndimage
    except Exception:  # pragma: no cover
        return -1
    lab, n = ndimage.label((phi > 0.5).cpu().numpy())
    if n == 0:
        return 0
    sizes = np.bincount(lab.ravel())[1:]
    return int((sizes >= min_cells).sum())


class ImpactRunner:
    """One impact event on a textured wall (SCMP kernel, D2Q9)."""

    def __init__(
        self,
        channel,
        *,
        radius: float,
        we: float,
        sigma: float = SIGMA_DEFAULT,
        rho_l: float = RHO_L_DEFAULT,
        rho_v: float = RHO_V_DEFAULT,
        G: float = 5.0,
        tau: float = 1.0,
        forcing: str = "guo",
        scheme: str = "standard",
        gap: float = 3.0,
        width: float = 1.5,
        fill_grooves: bool = False,
    ) -> None:
        from .engine import WallStepper

        self.channel = channel
        self.radius = float(radius)
        self.we = float(we)
        self.sigma = float(sigma)
        self.rho_l = float(rho_l)
        self.rho_v = float(rho_v)
        self.tau = float(tau)
        self.D0 = 2.0 * self.radius
        self.U0 = math.sqrt(self.we * self.sigma / (self.rho_l * self.D0))
        self.gap = float(gap)
        self.width = float(width)
        self.fill_grooves = bool(fill_grooves)
        self.solid = channel.solid_mask()
        self.fluid = ~self.solid
        self.groove = channel.groove_mask()
        self.stepper = WallStepper(
            channel,
            G=G,
            tau=tau,
            rho_l=rho_l,
            rho_v=rho_v,
            wall_fraction=channel.texture.wetting_fraction,
            forcing=forcing,
            scheme=scheme,
        )
        self.mu = self.rho_l * (self.tau - 0.5) / 3.0
        self.forcing = forcing
        self.scheme = scheme

        # 4-neighbour "adjacent to solid" stencil
        near = torch.zeros_like(self.solid)
        near[1:, :] |= self.solid[:-1, :]
        near[:-1, :] |= self.solid[1:, :]
        near[:, 1:] |= self.solid[:, :-1]
        near[:, :-1] |= self.solid[:, 1:]
        self.near_solid = near

    # -- state --------------------------------------------------------------
    def initial_state(self) -> torch.Tensor:
        dev = torch.device(self.channel.device)
        xx, yy = self.channel.grid(dev)
        cx = self.channel.nx / 2.0
        cy = self.channel.y_wall() + self.radius + self.gap
        dist = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) - self.radius
        rho = tanh_profile(self.channel.shape, dist, self.rho_l, self.rho_v, self.width)
        if self.fill_grooves:
            rho = torch.where(self.groove, torch.full_like(rho, self.rho_l), rho)
        rho = torch.where(self.solid, torch.full_like(rho, self.rho_l), rho)
        phi = torch.clamp((rho - self.rho_v) / (self.rho_l - self.rho_v), 0.0, 1.0)
        zero = torch.zeros_like(rho)
        # only the liquid travels: a uniform velocity field would give the vapour
        # column an unphysical downward momentum that slams into the texture
        return equilibrium(rho, zero, -self.U0 * phi)

    def observables(self, f: torch.Tensor) -> dict:
        rho, ux, uy = macroscopic(f)
        phi = torch.clamp((rho - self.rho_v) / (self.rho_l - self.rho_v), 0.0, 1.0) * self.fluid
        mass = float(phi.sum())
        y_wall = self.channel.y_wall()
        yy = torch.arange(self.channel.ny, dtype=torch.float32, device=rho.device).view(-1, 1)
        xx = torch.arange(self.channel.nx, dtype=torch.float32, device=rho.device).view(1, -1)
        wet = phi > 0.5

        contact_cells = int((wet & self.near_solid).sum())
        # contact *on the composite surface*: liquid pressed on a solid face above
        # the groove floor, i.e. what detaches when the drop leaves the texture
        above_plane = (yy > y_wall).expand_as(self.solid)
        contact_top = int((wet & self.near_solid & above_plane).sum())
        band = (yy >= y_wall + 1.0) & (yy <= y_wall + 5.0) & wet
        if bool(band.any()):
            xs = xx.expand_as(rho)[band]
            footprint = float(xs.max() - xs.min() + 1.0)
        else:
            footprint = 0.0

        if bool(wet.any()):
            yy_e = yy.expand_as(rho)
            y_top = float(yy_e[wet].max())
            y_bot = float(yy_e[wet].min())
        else:
            y_top = y_bot = float("nan")
        y_cm = float((phi * yy.expand_as(phi)).sum()) / mass if mass > 0 else float("nan")

        mass_total = float(rho[self.fluid].sum())
        ke = float((0.5 * rho * (ux * ux + uy * uy))[self.fluid].sum())
        e_surf = self.sigma * _grad_magnitude_sum(phi, self.fluid)
        diss = _neq_dissipation(f, rho, ux, uy, self.tau, self.fluid)
        groove_mass = float(phi[self.groove].sum())

        return {
            "mass": mass,
            "mass_total": mass_total,
            "contact_cells": contact_cells,
            "contact_top": contact_top,
            "footprint": footprint,
            "y_top": y_top,
            "y_bot": y_bot,
            "y_cm": y_cm,
            "ke": ke,
            "e_surf": e_surf,
            "diss_rate": diss,
            "groove_mass": groove_mass,
            "_phi": phi,
        }

    # -- driver -------------------------------------------------------------
    def run(
        self,
        *,
        steps: int | None = None,
        every: int = 20,
        contact_gap: int = 4,
        clusters_every: int = 100,
        keep_series: bool = True,
        snapshot_path: str | None = None,
        snapshots: tuple[float, ...] = (0.25, 0.5, 1.0),
    ) -> dict:
        """Advance the impact and reduce the observables.

        ``contact`` is detected from the number of liquid cells adjacent to a
        solid cell; the run is stopped ``steps`` after detachment so that
        rebound is observable.
        """
        tau_c = math.sqrt(self.rho_l * self.radius**3 / self.sigma)
        if steps is None:
            steps = int(2.6 * tau_c)
        steps = int(steps)

        snap_targets = {int(round(fr * tau_c)): fr for fr in snapshots} if snapshot_path else {}
        frames: dict[int, np.ndarray] = {}
        f = self.initial_state()
        obs = self.observables(f)
        series = []
        t_attach: int | None = None
        t_detach: int | None = None
        beta_max = 0.0
        clusters_max = 1
        contact_streak = 0

        def record(t: int, o: dict, clusters: int | None = None) -> None:
            if keep_series:
                row = {k: v for k, v in o.items() if not k.startswith("_")}
                row["t"] = t
                row["beta"] = o["footprint"] / self.D0
                if clusters is not None:
                    row["clusters"] = clusters
                series.append(row)

        record(0, obs, _liquid_clusters(obs["_phi"]) if clusters_every else None)
        e0 = obs["ke"] + obs["e_surf"]
        diss_int = 0.0
        t = 0
        stop_at = steps
        for t in range(1, steps + 1):
            f = self.stepper.step(f)
            if t in snap_targets:
                frames[t] = self.observables(f)["_phi"].cpu().numpy().astype(np.float16)
            if t % every == 0 or t == 1:
                obs = self.observables(f)
                diss_int += obs["diss_rate"] * every
                beta = obs["footprint"] / self.D0
                beta_max = max(beta_max, beta)
                in_contact = obs["contact_top"] > 0
                if in_contact and t_attach is None:
                    t_attach = t
                if t_attach is not None:
                    if in_contact:
                        contact_streak = 0
                    else:
                        contact_streak += every
                        if contact_streak >= contact_gap * every and t_detach is None:
                            t_detach = t - contact_streak + every
                            stop_at = min(steps, t + int(2.0 * tau_c))
                cl = None
                if clusters_every and t % clusters_every == 0:
                    cl = _liquid_clusters(obs["_phi"])
                    clusters_max = max(clusters_max, cl)
                record(t, obs, cl)
            if t >= stop_at > 0 and t_attach is not None and t_detach is not None:
                if t >= stop_at:
                    break

        final = self.observables(f)
        diss_int += final["diss_rate"] * every * 0.5
        e1 = final["ke"] + final["e_surf"]

        # rebound: detached, and the centre of mass rises afterwards
        rebound = False
        lift = float("nan")
        if t_detach is not None:
            at_detach = [r for r in series if r["t"] >= t_detach]
            if at_detach:
                y0 = at_detach[0]["y_cm"]
                lift = max(r["y_cm"] for r in at_detach) - y0
                rebound = bool(lift > 0.15 * self.radius and final["contact_top"] == 0)

        if t_detach is not None and t_attach is not None:
            tau_contact = float(t_detach - t_attach)
        elif t_attach is not None:
            tau_contact = float("nan")  # never detached within the run
        else:
            tau_contact = 0.0

        mass0 = series[0]["mass"]
        mass_tot0 = series[0]["mass_total"]
        residual_groove = final["groove_mass"] / mass0 if mass0 > 0 else float("nan")

        if clusters_max > 1:
            regime = "breakup"
        elif rebound:
            regime = "rebound"
        else:
            regime = "adhesion"

        if clusters_max > 1:
            regime = "breakup"
        if not math.isfinite(tau_contact):
            regime = "adhesion"

        return {
            "R": self.radius,
            "D0": self.D0,
            "U0": self.U0,
            "We": self.we,
            "Oh": self.mu / math.sqrt(self.rho_l * self.sigma * self.radius),
            "Re": self.rho_l * self.U0 * self.D0 / self.mu,
            "tau_c": tau_c,
            "beta_max": beta_max,
            "tau_contact": tau_contact,
            "tau_contact_over_tau_c": (
                tau_contact / tau_c if math.isfinite(tau_contact) else float("nan")
            ),
            "rebound": rebound,
            "lift": lift,
            "regime": regime,
            "clusters_max": clusters_max,
            "residual_groove_fraction": residual_groove,
            "t_attach": t_attach,
            "t_detach": t_detach,
            "steps_run": t,
            "mass_initial": mass0,
            "mass_final": final["mass"],
            "mass_drift": (final["mass"] - mass0) / mass0 if mass0 else float("nan"),
            "mass_total_initial": mass_tot0,
            "mass_total_final": final["mass_total"],
            "mass_total_drift": (
                (final["mass_total"] - mass_tot0) / mass_tot0 if mass_tot0 else float("nan")
            ),
            "energy": {
                "e0": e0,
                "e1": e1,
                "dissipated": diss_int,
                "residual": e1 - e0 + diss_int,
                "residual_over_e0": (e1 - e0 + diss_int) / e0 if e0 else float("nan"),
                "ke0": series[0]["ke"],
                "ke_max": max(r["ke"] for r in series),
                "e_surf0": series[0]["e_surf"],
            },
            "series": series,
            "snapshot_frames": {str(k): v for k, v in frames.items()},
            "forcing": self.forcing,
            "scheme": self.scheme,
            "tau": self.tau,
        }


def run_impact(
    channel,
    *,
    radius: float,
    we: float,
    sigma: float = SIGMA_DEFAULT,
    rho_l: float = RHO_L_DEFAULT,
    rho_v: float = RHO_V_DEFAULT,
    G: float = 5.0,
    tau: float = 1.0,
    forcing: str = "guo",
    scheme: str = "standard",
    steps: int | None = None,
    every: int = 20,
    clusters_every: int = 100,
    width: float = 1.5,
    fill_grooves: bool = False,
) -> dict:
    """Convenience wrapper around :class:`ImpactRunner`."""
    runner = ImpactRunner(
        channel,
        radius=radius,
        we=we,
        sigma=sigma,
        rho_l=rho_l,
        rho_v=rho_v,
        G=G,
        tau=tau,
        forcing=forcing,
        scheme=scheme,
        width=width,
        fill_grooves=fill_grooves,
    )
    return runner.run(steps=steps, every=every, clusters_every=clusters_every)


def scaling_fits(points: list[dict]) -> dict:
    """Power-law fits of ``beta_max(We)`` and ``tau_contact(rho R^3 / sigma)``.

    ``points`` are :func:`run_impact` results.  Returns

    * ``beta``: ``beta_max = a We^b`` (log-log least squares) together with the
      relative deviation of every point from the inertia-capillary closed form
      ``beta = sqrt(We/3)`` and from the Clanet et al. (2004) fit
      ``beta = 0.9 We^{1/4}``;
    * ``tau``: ``tau_contact = C (rho R^3/sigma)^p`` fitted against the group
      ``X = rho R^3/sigma`` itself (``X`` has units of time^2); the acceptance
      window is ``p = 0.5 +- 0.07``, equivalently ``tau ~ R^{3/2}``
      (``equiv_R_exponent = 3p``).
    """
    out: dict = {}
    use = [p for p in points if p.get("beta_max", 0.0) > 0.0 and p.get("We")]
    if len(use) >= 2:
        we = np.array([p["We"] for p in use], dtype=float)
        beta = np.array([p["beta_max"] for p in use], dtype=float)
        b, loga = np.polyfit(np.log(we), np.log(beta), 1)
        beta_inv = np.sqrt(we / 3.0)
        beta_clanet = 0.9 * we**0.25
        out["beta"] = {
            "a": float(math.exp(loga)),
            "b": float(b),
            "n_points": int(len(use)),
            "we": we.tolist(),
            "beta_max": beta.tolist(),
            "dev_vs_inviscid_sqrt_We_over_3": (np.abs(beta - beta_inv) / beta_inv).tolist(),
            "max_dev_vs_inviscid": float((np.abs(beta - beta_inv) / beta_inv).max()),
            "dev_vs_clanet_0p9_We_quarter": (np.abs(beta - beta_clanet) / beta_clanet).tolist(),
            "max_dev_vs_clanet": float((np.abs(beta - beta_clanet) / beta_clanet).max()),
        }

    tau_pts = [
        p
        for p in points
        if p.get("tau_contact") and math.isfinite(p["tau_contact"]) and p["tau_contact"] > 0
    ]
    radii = sorted({round(p["R"], 6) for p in tau_pts})
    if len(radii) >= 2:
        # one representative (largest We with a finite contact time) per radius
        reps = []
        for r in radii:
            cand = [p for p in tau_pts if round(p["R"], 6) == r]
            reps.append(max(cand, key=lambda p: p["We"]))
        # abscissa is the inertial-capillary group (rho R^3 / sigma) [time^2]; the
        # accepted law tau = C (rho R^3/sigma)^0.5 has exponent 1/2 (tau ~ R^{3/2})
        x = np.array([1.957 * p["R"] ** 3 / 0.05611 for p in reps])
        y = np.array([p["tau_contact"] for p in reps])
        p_exp, logc = np.polyfit(np.log(x), np.log(y), 1)
        resid = y - np.exp(logc) * x**p_exp
        out["tau"] = {
            "exponent": float(p_exp),
            "prefactor": float(math.exp(logc)),
            "radii": [float(p["R"]) for p in reps],
            "tau_contact": [float(p["tau_contact"]) for p in reps],
            "x_group": x.tolist(),
            "tau_over_sqrt_rhoR3_sigma": [
                float(p["tau_contact"] / math.sqrt(1.957 * p["R"] ** 3 / 0.05611)) for p in reps
            ],
            "equiv_R_exponent": float(3.0 * p_exp),
            "rel_residual": (resid / y).tolist(),
            "in_window": bool(abs(p_exp - 0.5) <= 0.07),
        }
    return out