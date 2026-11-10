"""Case definitions and drivers for the three turbulence-validation cases.

Each driver returns a plain dict that is JSON-serialisable and holds

* the case parameters actually used (from the config dataclass),
* the measured quantities (Strouhal number, reattachment length, wall-unit
  profile versus DNS),
* the error against the declared anchor (:mod:`...reference`) and a verdict
  computed by the same code that writes the report - no hand-copied numbers.

Geometry conventions
--------------------
cylinder
    Uniform inflow, no-slip top/bottom walls far from the body, NEEM pressure
    outlet; the body is a staircase disk of diameter ``diameter`` lattice units.
backward-facing step
    Armaly et al. (1983) geometry with expansion ratio ``(H1+h)/H1``; parabolic
    inflow, NEEM pressure outlet, no-slip walls and step block.
channel
    Periodic in x and z, halfway bounce-back walls, constant body force; the
    target friction Reynolds number fixes ``g = (Re_tau nu / h)^2 / h``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import metrics, native, reference

__all__ = [
    "CylinderConfig",
    "StepConfig",
    "ChannelConfig",
    "cylinder_masks",
    "step_masks",
    "run_cylinder",
    "run_step",
    "run_channel",
]


# ---------------------------------------------------------------------------
# circular cylinder
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CylinderConfig:
    re: int = 100
    diameter: int = 24
    length_diameters: float = 40.0
    height_diameters: float = 20.0
    cx_diameters: float = 10.0
    u_in: float = 0.1
    steps: int = 60000
    warmup_fraction: float = 0.25
    sample_every: int = 20
    cs: float = 0.05
    threads: int | None = None

    @property
    def nx(self) -> int:
        return int(round(self.length_diameters * self.diameter))

    @property
    def ny(self) -> int:
        return int(round(self.height_diameters * self.diameter)) + 2

    @property
    def cx(self) -> int:
        return int(round(self.cx_diameters * self.diameter))

    @property
    def nu(self) -> float:
        return self.u_in * self.diameter / self.re

    @property
    def tau(self) -> float:
        return 0.5 + 3.0 * self.nu

    def validate(self) -> None:
        if self.tau <= 0.502:
            raise ValueError(f"tau={self.tau:.4f} too close to 1/2 for a stable run")
        if self.cx < 3 * self.diameter or self.nx - self.cx < 20 * self.diameter:
            raise ValueError("cylinder needs >=3D upstream and >=20D downstream")


def cylinder_masks(cfg: CylinderConfig) -> tuple[np.ndarray, np.ndarray]:
    nx, ny = cfg.nx, cfg.ny
    yy, xx = np.mgrid[0:ny, 0:nx]
    solid = np.zeros((ny, nx), dtype=np.int32)
    solid[0, :] = 1
    solid[-1, :] = 1
    body = ((xx - cfg.cx) ** 2 + (yy - ny // 2) ** 2) <= (cfg.diameter / 2.0) ** 2
    solid[body] = 1
    return solid, body


def run_cylinder(cfg: CylinderConfig, out_dir: str | Path, *, build_dir=None) -> dict:
    cfg.validate()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    solid, body = cylinder_masks(cfg)
    prefix = out_dir / f"cylinder_re{cfg.re}"
    warmup = int(cfg.steps * cfg.warmup_fraction)
    run = native.run_d2q9(
        out_prefix=prefix,
        nx=cfg.nx, ny=cfg.ny, solid=solid, body=body,
        steps=cfg.steps, warmup=warmup, u_in=cfg.u_in, tau=cfg.tau, cs=cfg.cs,
        sample_every=cfg.sample_every, threads=cfg.threads, build_dir=build_dir,
    )
    forces = native.read_force_series(prefix)
    st = metrics.strouhal_from_lift(
        forces["step"], forces["fy"], u_in=cfg.u_in, diameter=cfg.diameter,
        discard=cfg.warmup_fraction,
    )
    mean = native.read_mean_field(prefix, cfg.nx, cfg.ny)
    cd = metrics.pressure_drag_from_field(mean, u_in=cfg.u_in, diameter=cfg.diameter)
    ref_exp = reference.cylinder_reference(cfg.re, "experiment")
    ref_dns = reference.cylinder_reference(cfg.re, "dns_2d")
    st_best = st["st_crossings"] if np.isfinite(st["st_crossings"]) else st["st_fft"]
    err_exp = (st_best - ref_exp.value) / ref_exp.value
    err_dns = (st_best - ref_dns.value) / ref_dns.value
    return {
        "case": "cylinder",
        "config": asdict(cfg) | {"nu": cfg.nu, "tau": cfg.tau, "nx": cfg.nx, "ny": cfg.ny},
        "measured": {"strouhal": st_best, **st, **cd},
        "reference": {
            "experiment": {"value": ref_exp.value, "source": ref_exp.source,
                           "rel_error": float(err_exp),
                           "abs_rel_error": float(abs(err_exp)),
                           "passed": bool(abs(err_exp) <= ref_exp.tolerance)},
            "dns_2d": {"value": ref_dns.value, "source": ref_dns.source,
                       "rel_error": float(err_dns),
                       "abs_rel_error": float(abs(err_dns)),
                       "passed": bool(abs(err_dns) <= ref_dns.tolerance)},
        },
        "run": run,
    }


# ---------------------------------------------------------------------------
# backward-facing step
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StepConfig:
    re: int = 100
    step_h: int = 49
    upstream_h: int = 52          # Armaly: 5.2 mm step 4.9 mm -> ER = 1.9423
    u_mean: float = 0.1
    entrance_steps: int = 8       # upstream length in upstream channel heights
    downstream_steps: int = 26    # downstream length in step heights
    steps: int = 50000
    warmup_fraction: float = 0.5
    sample_every: int = 20
    threads: int | None = None

    @property
    def expansion_ratio(self) -> float:
        return (self.upstream_h + self.step_h) / self.upstream_h

    @property
    def ny(self) -> int:
        return self.upstream_h + self.step_h + 2

    @property
    def x_step(self) -> int:
        return self.entrance_steps * self.upstream_h

    @property
    def nx(self) -> int:
        return self.x_step + self.downstream_steps * self.step_h

    @property
    def nu(self) -> float:
        return self.u_mean * 2.0 * self.upstream_h / self.re

    @property
    def tau(self) -> float:
        return 0.5 + 3.0 * self.nu

    def validate(self) -> None:
        if self.tau >= 1.9:
            raise ValueError(f"tau={self.tau:.3f}: inlet velocity too high for stability")
        if abs(self.expansion_ratio - 1.9423) > 0.01:
            raise ValueError(
                f"expansion ratio {self.expansion_ratio:.4f} does not match Armaly 1.9423"
            )


def step_masks(cfg: StepConfig) -> tuple[np.ndarray, np.ndarray, int]:
    nx, ny = cfg.nx, cfg.ny
    solid = np.zeros((ny, nx), dtype=np.int32)
    solid[0, :] = 1
    solid[-1, :] = 1
    solid[1 : cfg.step_h + 1, : cfg.x_step] = 1  # step block + upstream floor
    body = solid.copy()
    body[0, :] = body[-1, :] = 0
    return solid, body, cfg.x_step


def run_step(cfg: StepConfig, out_dir: str | Path, *, build_dir=None) -> dict:
    cfg.validate()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    solid, body, x_step = step_masks(cfg)
    prefix = out_dir / f"step_re{cfg.re}"
    warmup = int(cfg.steps * cfg.warmup_fraction)
    run = native.run_d2q9(
        out_prefix=prefix,
        nx=cfg.nx, ny=cfg.ny, solid=solid, body=body,
        steps=cfg.steps, warmup=warmup, u_in=cfg.u_mean, tau=cfg.tau, parabolic=True,
        inlet_y0=cfg.step_h + 0.5, inlet_y1=cfg.step_h + 0.5 + cfg.upstream_h,
        sample_every=cfg.sample_every, threads=cfg.threads, build_dir=build_dir,
    )
    mean = native.read_mean_field(prefix, cfg.nx, cfg.ny)
    meas = metrics.reattachment_length(
        mean, x_step=x_step, step_h=float(cfg.step_h), u_in=cfg.u_mean
    )
    ref = reference.bfs_reference(cfg.re)
    err = (meas["x_r_over_h"] - ref.value) / ref.value
    return {
        "case": "backward_facing_step",
        "config": asdict(cfg) | {
            "nu": cfg.nu, "tau": cfg.tau, "nx": cfg.nx, "ny": cfg.ny,
            "x_step": x_step, "expansion_ratio": cfg.expansion_ratio,
        },
        "measured": meas,
        "reference": {
            "experiment": {"value": ref.value, "source": ref.source,
                           "rel_error": float(err), "abs_rel_error": float(abs(err)),
                           "passed": bool(abs(err) <= ref.tolerance)},
        },
        "run": run,
    }


# ---------------------------------------------------------------------------
# turbulent channel
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChannelConfig:
    re_tau: int = 180
    ny: int = 182
    length_wall_units: float = 400.0
    span_wall_units: float = 160.0
    nu: float = 0.004
    cs: float = 0.1
    wall_mode: int = 0
    steps: int = 100000
    warmup_fraction: float = 0.2
    sample_every: int = 20
    threads: int | None = None

    @property
    def h(self) -> float:
        return (self.ny - 2) / 2.0

    @property
    def u_tau(self) -> float:
        return self.re_tau * self.nu / self.h

    @property
    def force(self) -> float:
        return self.u_tau**2 / self.h

    @property
    def du(self) -> float:
        """Wall-unit length of one lattice cell (``Re_tau / h``)."""
        return self.re_tau / self.h

    @property
    def nx(self) -> int:
        return int(round(self.length_wall_units / self.du))

    @property
    def nz(self) -> int:
        return int(round(self.span_wall_units / self.du))

    def validate(self) -> None:
        if 0.5 + 3.0 * self.nu <= 0.503:
            raise ValueError(
                f"tau={0.5 + 3 * self.nu:.5f} too close to 1/2: increase nu or lower Re_tau"
            )
        if not 4 <= self.nx <= 512 or not 4 <= self.nz <= 256:
            raise ValueError(f"unreasonable periodic box nx={self.nx} nz={self.nz}")


def run_channel(cfg: ChannelConfig, out_dir: str | Path, *, build_dir=None) -> dict:
    cfg.validate()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / f"channel_retau{cfg.re_tau}_wall{cfg.wall_mode}"
    warmup = int(cfg.steps * cfg.warmup_fraction)
    run = native.run_channel3d(
        out_prefix=prefix, nx=cfg.nx, ny=cfg.ny, nz=cfg.nz,
        steps=cfg.steps, warmup=warmup, tau=0.5 + 3.0 * cfg.nu, force=cfg.force,
        cs=cfg.cs, wall_mode=cfg.wall_mode, sample_every=cfg.sample_every,
        threads=cfg.threads, build_dir=build_dir,
    )
    profile = native.read_profile(prefix)
    diag = native.read_diag(prefix)
    from tensorlbm.validation import mkm1999

    dns = mkm1999.load_profile(cfg.re_tau)
    wu = metrics.wall_units(profile, ny=cfg.ny, nu=cfg.nu, force=cfg.force)
    cmp_resolved = metrics.compare_u_plus(wu, dns, y_plus_range=(5.0, None))
    cmp_all = metrics.compare_u_plus(wu, dns, y_plus_range=(1.0, None))
    # fluctuation check: a true turbulent channel has u_rms+ > 0.3 somewhere
    with np.errstate(invalid="ignore"):
        peak_rms = float(np.nanmax(wu["u_rms_plus"])) if np.isfinite(
            wu["u_rms_plus"]).any() else float("nan")
    utau_wall = float(np.nanmean(diag["utau_wall"][len(diag["utau_wall"]) // 2:]))
    passed = bool(
        cmp_resolved.get("status") == "ok"
        and (cmp_resolved.get("max_rel_error") or 1e9) <= 0.10
        and np.isfinite(peak_rms) and peak_rms > 0.3
    )
    return {
        "case": "turbulent_channel",
        "config": asdict(cfg) | {
            "nu": cfg.nu, "tau": 0.5 + 3.0 * cfg.nu, "force": cfg.force,
            "u_tau": cfg.u_tau, "h": cfg.h, "nx": cfg.nx, "nz": cfg.nz,
        },
        "measured": {
            "y_plus": wu["y_plus"].tolist(),
            "u_plus": wu["u_plus"].tolist(),
            "u_rms_plus": wu["u_rms_plus"].tolist(),
            "re_tau_force": wu["re_tau_force"],
            "re_tau_gradient": wu["re_tau_gradient"],
            "u_tau_force": wu["u_tau_force"],
            "u_tau_gradient": wu["u_tau_gradient"],
            "u_tau_wall_model": utau_wall,
            "peak_u_rms_plus": peak_rms,
            "kinetic_energy_final": float(diag["ke"][-1]),
            "u_plus_centre": float(wu["u_plus"][len(wu["u_plus"]) // 2]),
        },
        "reference": {
            "dns": {
                "source": "Moser, Kim & Mansour (1999) Phys. Fluids 11:943 "
                          "(shipped table, SHA-256 verified)",
                "nominal_re_tau": int(dns.nominal_re_tau),
                "re_tau": float(dns.re_tau),
                "bulk_u_plus": float(dns.bulk_velocity_plus()),
                **{k: v for k, v in cmp_resolved.items()},
                "all_y_plus": {k: v for k, v in cmp_all.items()},
                "passed": bool(passed),
            }
        },
        "run": run,
    }

# ---------------------------------------------------------------------------
# cylinder blockage study
# ---------------------------------------------------------------------------

def run_cylinder_sweep(re_primary: int = 100,
                       blockage_height_diameters: tuple[float, ...] = (40.0, 30.0, 20.0, 12.0),
                       secondary_re: tuple[int, ...] = (200, 300),
                       diameter: int = 20, length_diameters: float = 30.0,
                       cx_diameters: float = 8.0, steps: int = 50000,
                       threads: int | None = None, build_dir=None,
                       out_dir: str | Path = ".") -> dict:
    """Strouhal number versus blockage ratio, plus the two higher-Re cases.

    The criterion is stated against the *unbounded* experiment, while any finite
    domain has a finite blockage ``B = D / H``.  Instead of hiding the resulting
    bias, the campaign runs a blockage sweep at ``Re = 100`` and reports both the
    raw value at the working blockage and the linear ``B -> 0`` extrapolation.
    """
    runs = []
    for height in blockage_height_diameters:
        cfg = CylinderConfig(
            re=re_primary, diameter=diameter, length_diameters=length_diameters,
            height_diameters=height, cx_diameters=cx_diameters, steps=steps,
            threads=threads,
        )
        res = run_cylinder(cfg, Path(out_dir) / f"cylinder_B{1.0 / height:.4f}", build_dir=build_dir)
        ref = reference.cylinder_reference(re_primary, "experiment")
        st = res["measured"]["strouhal"]
        runs.append({
            "blockage": 1.0 / height,
            "height_diameters": height,
            "re": re_primary,
            "strouhal": st,
            "st_fft": res["measured"].get("st_fft"),
            "rel_error_vs_experiment": (st - ref.value) / ref.value if st else None,
            "case": res,
        })
    finite = [(r["blockage"], r["strouhal"]) for r in runs
              if r["strouhal"] is not None and np.isfinite(r["strouhal"])]
    fit = None
    if len(finite) >= 2:
        b = np.asarray([f[0] for f in finite])
        st = np.asarray([f[1] for f in finite])
        slope, intercept = np.polyfit(b, st, 1)
        resid = st - (slope * b + intercept)
        fit = {
            "slope_per_unit_blockage": float(slope),
            "st_unbounded": float(intercept),
            "rms_residual": float(np.sqrt(np.mean(resid**2))),
            "n_points": int(len(finite)),
        }
    secondary = []
    for re in secondary_re:
        cfg = CylinderConfig(
            re=re, diameter=diameter, length_diameters=length_diameters,
            height_diameters=max(blockage_height_diameters), cx_diameters=cx_diameters,
            steps=steps, threads=threads,
        )
        res = run_cylinder(cfg, Path(out_dir) / f"cylinder_re{re}", build_dir=build_dir)
        ref = reference.cylinder_reference(re, "experiment")
        ref2 = reference.cylinder_reference(re, "dns_2d")
        st = res["measured"]["strouhal"]
        secondary.append({
            "re": re,
            "blockage": 1.0 / max(blockage_height_diameters),
            "strouhal": st,
            "rel_error_vs_experiment": (st - ref.value) / ref.value if st else None,
            "rel_error_vs_2d_dns": (st - ref2.value) / ref2.value if st else None,
            "case": res,
        })
    return {"primary_re": re_primary, "blockage_sweep": runs, "blockage_fit": fit,
            "secondary": secondary}
