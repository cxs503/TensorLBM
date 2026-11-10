"""Rebuild a campaign payload from raw kernel artefacts (no re-running).

Used when a long campaign is interrupted before it can write its own
``results.json``: every metric is recomputed from the artefacts on disk, using
the same functions the campaign uses.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from tensorlbm.turbulence_closure import cases, metrics, native, reference  # noqa: E402
from tensorlbm.turbulence_closure.campaign import _json_safe, _verdicts  # noqa: E402
from tensorlbm.validation import mkm1999  # noqa: E402

ROOT = Path("runs")
payload = {"campaign": "P152W1-N02 turbulence validation face",
           "profile": "compact (rebuilt from raw artefacts)",
           "cases": {}}


def rebuild_channel(prefix: Path, cfg: cases.ChannelConfig) -> dict:
    prof = native.read_profile(str(prefix).replace("_profile.txt", ""))
    diag = native.read_diag(str(prefix).replace("_profile.txt", ""))
    dns = mkm1999.load_profile(cfg.re_tau)
    wu = metrics.wall_units(prof, ny=cfg.ny, nu=cfg.nu, force=cfg.force)
    cmp_resolved = metrics.compare_u_plus(wu, dns, y_plus_range=(5.0, None))
    cmp_all = metrics.compare_u_plus(wu, dns, y_plus_range=(1.0, None))
    peak_rms = float(np.nanmax(wu["u_rms_plus"]))
    passed = bool(cmp_resolved.get("status") == "ok"
                  and (cmp_resolved.get("max_rel_error") or 1e9) <= 0.10
                  and peak_rms > 0.3)
    return {
        "case": "turbulent_channel",
        "config": {"re_tau": cfg.re_tau, "ny": cfg.ny, "nx": cfg.nx, "nz": cfg.nz,
                   "nu": cfg.nu, "tau": 0.5 + 3 * cfg.nu, "force": cfg.force,
                   "u_tau": cfg.u_tau, "h": cfg.h, "cs": cfg.cs,
                   "steps": cfg.steps, "wall_mode": cfg.wall_mode},
        "measured": {
            "y_plus": wu["y_plus"].tolist(), "u_plus": wu["u_plus"].tolist(),
            "u_rms_plus": wu["u_rms_plus"].tolist(),
            "re_tau_force": wu["re_tau_force"], "re_tau_gradient": wu["re_tau_gradient"],
            "u_tau_force": wu["u_tau_force"], "u_tau_gradient": wu["u_tau_gradient"],
            "peak_u_rms_plus": peak_rms,
            "kinetic_energy_final": float(diag["ke"][-1]),
            "u_plus_centre": float(wu["u_plus"][len(wu["u_plus"]) // 2]),
        },
        "reference": {"dns": {"source": "Moser, Kim & Mansour (1999) Phys. Fluids 11:943",
                              "nominal_re_tau": int(dns.nominal_re_tau),
                              "re_tau": float(dns.re_tau),
                              **cmp_resolved, "all_y_plus": cmp_all,
                              "passed": passed}},
        "run": {"artefacts": {}},
    }


def rebuild_step(prefix: Path, cfg: cases.StepConfig) -> dict:
    mean = native.read_mean_field(str(prefix).replace("_mean.txt", ""), cfg.nx, cfg.ny)
    meas = metrics.reattachment_length(mean, x_step=cfg.x_step, step_h=float(cfg.step_h),
                                       u_in=cfg.u_mean)
    ref = reference.bfs_reference(cfg.re)
    err = (meas["x_r_over_h"] - ref.value) / ref.value
    return {
        "case": "backward_facing_step",
        "config": {"re": cfg.re, "nx": cfg.nx, "ny": cfg.ny, "nu": cfg.nu,
                   "tau": cfg.tau, "x_step": cfg.x_step, "step_h": cfg.step_h,
                   "u_mean": cfg.u_mean, "expansion_ratio": cfg.expansion_ratio,
                   "steps": cfg.steps},
        "measured": meas,
        "reference": {"experiment": {"value": ref.value, "source": ref.source,
                                     "rel_error": float(err),
                                     "abs_rel_error": float(abs(err)),
                                     "passed": bool(abs(err) <= ref.tolerance)}},
        "run": {"artefacts": {}},
    }


def rebuild_cylinder(prefix: Path, cfg: cases.CylinderConfig) -> dict:
    forces = native.read_force_series(str(prefix).replace("_force.txt", ""))
    st = metrics.strouhal_from_lift(forces["step"], forces["fy"], u_in=cfg.u_in,
                                    diameter=cfg.diameter, discard=cfg.warmup_fraction)
    try:
        mean = native.read_mean_field(str(prefix).replace("_mean.txt", ""), cfg.nx, cfg.ny)
        cd = metrics.pressure_drag_from_field(mean, u_in=cfg.u_in, diameter=cfg.diameter)
    except Exception:
        cd = {"cd_note": "mean field not available (run truncated before the mean dump)"}
    exp = reference.cylinder_reference(cfg.re, "experiment")
    dns = reference.cylinder_reference(cfg.re, "dns_2d")
    st_best = st["st_crossings"] if np.isfinite(st["st_crossings"]) else st["st_fft"]
    return {
        "case": "cylinder",
        "config": {"re": cfg.re, "nx": cfg.nx, "ny": cfg.ny, "nu": cfg.nu,
                   "tau": cfg.tau, "u_in": cfg.u_in, "diameter": cfg.diameter,
                   "steps": cfg.steps, "blockage": cfg.diameter / cfg.ny},
        "measured": {"strouhal": st_best, **st, **cd},
        "reference": {
            "experiment": {"value": exp.value, "source": exp.source,
                           "rel_error": float((st_best - exp.value) / exp.value),
                           "abs_rel_error": float(abs((st_best - exp.value) / exp.value)),
                           "passed": bool(abs((st_best - exp.value) / exp.value) <= exp.tolerance)},
            "dns_2d": {"value": dns.value, "source": dns.source,
                       "rel_error": float((st_best - dns.value) / dns.value),
                       "abs_rel_error": float(abs((st_best - dns.value) / dns.value)),
                       "passed": bool(abs((st_best - dns.value) / dns.value) <= dns.tolerance)},
        },
        "run": {"artefacts": {}},
    }


ch = []
for prefix in sorted(ROOT.glob("tvf_ch*/channel/*_profile.txt")):
    meta = native.read_meta(str(prefix).replace("_profile.txt", ""))
    ny_ = int(meta["ny"])
    h_ = (ny_ - 2) / 2.0
    u_tau_ = float(np.sqrt(float(meta["force"]) * h_))
    nu_ = float(meta["nu"])
    re_tau_ = u_tau_ * h_ / nu_
    du_ = re_tau_ / h_
    cfg = cases.ChannelConfig(
        re_tau=int(round(re_tau_)), ny=ny_, nu=nu_, cs=float(meta["cs"]),
        steps=int(meta["steps"]),
        length_wall_units=int(meta["nx"]) * du_, span_wall_units=int(meta["nz"]) * du_)
    ch.append(rebuild_channel(prefix, cfg))
if ch:
    payload["cases"]["channel"] = ch

st_cases = []
for prefix in sorted(ROOT.glob("tvf_planar/step/*_mean.txt")):
    p = str(prefix).replace("_mean.txt", "")
    meta = native.read_meta(p)
    nx, ny = int(meta["nx"]), int(meta["ny"])
    re = int(p.rsplit("re", 1)[1])
    # geometry from the mask: x_step is the first column where the bed is fluid
    solid = np.loadtxt(p + ".mask.txt", dtype=int).reshape(ny, nx)
    x_step = int(np.argmax(solid[1, :] == 0))
    h = int(np.argmax(solid[:, 0] == 0)) - 1
    cfg = cases.StepConfig(re=re, step_h=h, upstream_h=ny - h - 2,
                           entrance_steps=int(round(x_step / (ny - h - 2))),
                           downstream_steps=int(round((nx - x_step) / h)),
                           steps=int(meta["steps"]), u_mean=float(meta["u_in"]))
    st_cases.append(rebuild_step(prefix, cfg))
if st_cases:
    payload["cases"]["backward_facing_step"] = st_cases

cyl = []
cyl_prefixes = sorted(ROOT.glob("tvf_planar/cylinder/*/*_force.txt"))
for prefix in cyl_prefixes:
    prefix = Path(str(prefix).replace("_force.txt", "_mean.txt"))
    p = str(prefix).replace("_mean.txt", "")
    try:
        meta = native.read_meta(p)
    except FileNotFoundError:
        continue
    nx, ny = int(meta["nx"]), int(meta["ny"])
    re = int(p.rsplit("re", 1)[1])
    solid = np.loadtxt(p + ".mask.txt", dtype=int).reshape(ny, nx)
    body = np.loadtxt(p + ".body.txt", dtype=int).reshape(ny, nx).astype(bool)
    d = 2.0 * np.sqrt(body.sum() / np.pi)
    cfg = cases.CylinderConfig(re=re, diameter=int(round(d)), length_diameters=nx / d,
                               height_diameters=ny / d, u_in=float(meta["u_in"]),
                               steps=int(meta["steps"]))
    cyl.append(rebuild_cylinder(prefix, cfg))
if cyl:
    payload["cases"]["cylinder"] = cyl

payload["verdicts"] = _verdicts(payload)
out = ROOT / "tvf_rebuilt"
out.mkdir(parents=True, exist_ok=True)
(out / "results.json").write_text(
    json.dumps(_json_safe(payload), indent=2, allow_nan=False, default=str), encoding="utf-8")
print(json.dumps(payload["verdicts"], indent=2))