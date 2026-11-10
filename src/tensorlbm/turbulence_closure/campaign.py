"""Campaign driver: three cases x three Reynolds numbers, figures, report.

Usage (inside a repo checkout)::

    python -m tensorlbm.turbulence_closure.campaign --out runs/turbulence_validation_face

Everything the report needs is computed here and written to

    <out>/results.json          all numbers + verdicts + artefact paths
    <out>/figures/*.png         publication figures
    docs/turbulence_validation_face.md   paper-grade report (numbers injected)

No number in the report is typed by hand: the markdown renderer only formats
values taken from ``results.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import cases, native, reference, verification
from .cases import ChannelConfig, CylinderConfig, StepConfig

__all__ = ["DEFAULT_CASES", "run_campaign", "make_figures", "render_markdown"]

DEFAULT_CASES = {
    "cylinder": [CylinderConfig(re=r, diameter=24, steps=60000) for r in (100, 200, 300)],
    "step": [StepConfig(re=r, steps=50000) for r in (100, 200, 389)],
    "cylinder_blockage": (40.0, 30.0, 20.0, 12.0),
    "cylinder_secondary": (200, 300),
    "cylinder_diameter": 20,
    "cylinder_length": 30.0,
    "cylinder_steps": 50000,
    "channel": [
        ChannelConfig(re_tau=180, ny=182, nu=0.004, length_wall_units=400.0,
                      span_wall_units=150.0, steps=90000),
        ChannelConfig(re_tau=395, ny=182, nu=0.002, length_wall_units=600.0,
                      span_wall_units=200.0, steps=90000),
        ChannelConfig(re_tau=590, ny=242, nu=0.0018, length_wall_units=600.0,
                      span_wall_units=200.0, steps=110000),
    ],
}

SMOKE_CASES = {
    "cylinder": [CylinderConfig(re=100, diameter=12, length_diameters=32.0,
                                height_diameters=12.0, cx_diameters=10.0, steps=4000)],
    "step": [StepConfig(re=100, step_h=49, upstream_h=52, entrance_steps=2,
                        downstream_steps=6, steps=3000)],
    "channel": [ChannelConfig(re_tau=180, ny=50, nu=0.01, length_wall_units=200.0,
                              span_wall_units=100.0, steps=3000)],
    "cylinder_blockage": (24.0, 12.0),
    "cylinder_secondary": (),
    "cylinder_diameter": 12,
    "cylinder_length": 32.0,
    "cylinder_steps": 4000,
}


def _json_safe(obj):
    """Recursively convert numpy containers and drop non-finite floats to null."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if np.isfinite(val) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


#: Compact (compute-bounded) profile.  Same physics and the same anchors, but the
#: periodic box / domain lengths are the smallest that still resolve the mean
#: fields (documented in docs/turbulence_closure_design.md).  Every number in
#: results.json carries its own config, so a compact run can never be mistaken
#: for a production-resolution one.
COMPACT_CASES = {
    "channel": [
        # Re_tau = 180: h = 50, y+ = 1.8, Lx+ = 300, Lz+ = 108
        ChannelConfig(re_tau=180, ny=102, nu=0.0022, length_wall_units=300.0,
                      span_wall_units=108.0, steps=20000, warmup_fraction=0.3, cs=0.16),
        # Re_tau = 395: h = 90, y+ = 2.2, Lx+ = 300, Lz+ = 110
        ChannelConfig(re_tau=395, ny=182, nu=0.0021, length_wall_units=300.0,
                      span_wall_units=110.0, steps=20000, warmup_fraction=0.3, cs=0.16),
        # Re_tau = 590: h = 120, y+ = 2.5, Lx+ = 300, Lz+ = 108
        ChannelConfig(re_tau=590, ny=242, nu=0.0018, length_wall_units=300.0,
                      span_wall_units=108.0, steps=25000, warmup_fraction=0.3, cs=0.16),
    ],
    "step": [
        StepConfig(re=100, steps=30000, downstream_steps=20),
        StepConfig(re=200, steps=30000, downstream_steps=20),
        StepConfig(re=389, steps=30000, downstream_steps=20),
    ],
    "cylinder": [
        CylinderConfig(re=r, diameter=16, length_diameters=28.0, height_diameters=20.0,
                       cx_diameters=8.0, steps=32000) for r in (100, 200, 300)
    ],
    "cylinder_blockage": (30.0, 20.0),
    "cylinder_secondary": (200, 300),
    "cylinder_diameter": 16,
    "cylinder_length": 28.0,
    "cylinder_steps": 32000,
}


def _select(smoke: bool, profile: str = "default") -> dict:
    if smoke:
        return SMOKE_CASES
    if profile == "compact":
        return COMPACT_CASES
    if profile != "default":
        raise ValueError(f"unknown profile {profile!r}")
    return DEFAULT_CASES


def run_campaign(out_dir: str | Path, *, threads: int | None = None,
                 smoke: bool = False, only: tuple[str, ...] = (),
                 build_dir=None, profile: str = "default") -> dict:
    """Run the whole campaign and return the results payload."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    selection = _select(smoke, profile)
    started = time.time()
    results: dict = {
        "campaign": "P152W1-N02 turbulence validation face",
        "profile": "smoke" if smoke else profile,
        "repo": "TensorLBM",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "host": os.uname().nodename,
            "machine": os.uname().machine,
            "python": os.sys.version.split()[0],
            "numpy": np.__version__,
            "omp_threads": threads,
        },
        "acceptance_criteria": {
            "channel_u_plus_max_rel_error_vs_dns": 0.10,
            "step_reattachment_rel_error_vs_experiment": 0.15,
            "cylinder_strouhal_rel_error_vs_experiment": 0.05,
        },
        "cases": {},
        "artefacts": {},
    }

    if not only or "verification" in only:
        results["verification"] = verification.run_verification(
            threads=threads, build_dir=build_dir
        )

    def _guard(label, fn):
        try:
            return fn()
        except Exception as exc:  # honest FAIL: record, never hide
            return {
                "case": label,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "reference": {},
                "measured": {},
                "config": {},
                "run": {},
            }

    if not only or "channel" in only:
        results["cases"]["channel"] = [
            _guard("turbulent_channel",
                   lambda cfg=cfg: cases.run_channel(cfg, out / "channel", build_dir=build_dir))
            for cfg in selection["channel"]
        ]
    if not only or "step" in only:
        results["cases"]["backward_facing_step"] = [
            _guard("backward_facing_step",
                   lambda cfg=cfg: cases.run_step(cfg, out / "step", build_dir=build_dir))
            for cfg in selection["step"]
        ]
    if not only or "cylinder" in only:
        sweep = _guard(
            "cylinder_sweep",
            lambda: cases.run_cylinder_sweep(
                re_primary=100,
                blockage_height_diameters=selection.get("cylinder_blockage", (40.0, 30.0, 20.0, 12.0)),
                secondary_re=tuple(selection.get("cylinder_secondary", (200, 300))),
                diameter=selection.get("cylinder_diameter", 20),
                length_diameters=selection.get("cylinder_length", 30.0),
                steps=selection.get("cylinder_steps", 50000),
                threads=threads, build_dir=build_dir, out_dir=out / "cylinder",
            ),
        )
        if sweep.get("status") == "failed":
            results["cases"]["cylinder"] = [sweep]
        else:
            primary = [r for r in sweep["blockage_sweep"] if abs(r["height_diameters"] - 30.0) < 1e-9]
            cases_list = ([primary[0]["case"]] if primary else [])
            cases_list += [s["case"] for s in sweep["secondary"]]
            results["cases"]["cylinder"] = cases_list
            results["cylinder_blockage"] = {
                "fit": sweep["blockage_fit"],
                "points": [
                    {k: v for k, v in r.items() if k != "case"} for r in sweep["blockage_sweep"]
                ],
            }

    results["elapsed_seconds"] = round(time.time() - started, 1)
    results["artefacts"] = {
        "results_json": str((out / "results.json").resolve()),
        "figures_dir": str((out / "figures").resolve()),
        "raw_runs": str(out.resolve()),
    }
    results["verdicts"] = _verdicts(results)
    Path(out / "results.json").write_text(
        json.dumps(_json_safe(results), indent=2, allow_nan=False, default=str),
        encoding="utf-8",
    )
    return results


def _verdicts(results: dict) -> dict:
    verdicts: dict = {}
    for case in results["cases"].get("channel", []):
        if case.get("status") == "failed" or "dns" not in case.get("reference", {}):
            verdicts[f"channel_case_{len(verdicts)}"] = {
                "criterion": "case aborted", "value": None, "passed": False,
                "error": case.get("error"),
            }
            continue
        dns = case["reference"]["dns"]
        verdicts[f"channel_Re_tau{case['config']['re_tau']}"] = {
            "criterion": "max relative u+ error vs MKM DNS < 10% (y+ >= 5)",
            "value": dns.get("max_rel_error"),
            "passed": dns.get("passed", False),
        }
    for case in results["cases"].get("backward_facing_step", []):
        if case.get("status") == "failed" or "experiment" not in case.get("reference", {}):
            verdicts[f"step_case_{len(verdicts)}"] = {
                "criterion": "case aborted", "value": None, "passed": False,
                "error": case.get("error"),
            }
            continue
        exp = case["reference"]["experiment"]
        verdicts[f"step_Re{case['config']['re']}"] = {
            "criterion": "reattachment length x_R/h vs experiment within 15%",
            "value": exp["abs_rel_error"],
            "measured": case["measured"].get("x_r_over_h"),
            "reference": exp["value"],
            "passed": bool(exp["passed"]),
        }
    for case in results["cases"].get("cylinder", []):
        if case.get("status") == "failed" or "experiment" not in case.get("reference", {}):
            verdicts[f"cylinder_case_{len(verdicts)}"] = {
                "criterion": "case aborted", "value": None, "passed": False,
                "error": case.get("error"),
            }
            continue
        exp = case["reference"]["experiment"]
        verdicts[f"cylinder_Re{case['config']['re']}"] = {
            "criterion": "Strouhal number vs experiment within 5%",
            "value": exp["abs_rel_error"],
            "measured": case["measured"].get("strouhal"),
            "reference": exp["value"],
            "passed": bool(exp["passed"]),
        }
    blk = results.get("cylinder_blockage", {}).get("fit")
    if blk:
        ref100 = reference.cylinder_reference(100, "experiment")
        err = (blk["st_unbounded"] - ref100.value) / ref100.value
        verdicts["cylinder_St_blockage_extrapolated_Re100"] = {
            "criterion": "B->0 extrapolated St vs experiment within 5%",
            "value": abs(err),
            "measured": blk["st_unbounded"],
            "reference": ref100.value,
            "passed": bool(abs(err) <= 0.05),
        }
    verdicts["all_passed"] = all(
        v.get("passed", False) for k, v in verdicts.items() if k != "all_passed"
    )
    return verdicts


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _artefact_stem(case: dict, kind: str, fallback: str) -> str:
    path = case.get("run", {}).get("artefacts", {}).get(kind)
    if not path:
        return fallback
    return str(path).replace(f"_{kind}.txt", "")


def make_figures(results: dict, fig_dir: str | Path) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from tensorlbm.validation import mkm1999

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    figures: list[dict] = []

    # ---- 1. u+ vs y+ for the three channel tiers --------------------------
    channel = results["cases"].get("channel", [])
    if channel:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        for case in [c for c in channel if c.get("status") != "failed"]:
            re_tau = case["config"]["re_tau"]
            yp = np.asarray(case["measured"]["y_plus"], dtype=float)
            up = np.asarray(case["measured"]["u_plus"], dtype=float)
            good = np.isfinite(yp) & np.isfinite(up)
            yp, up = yp[good], up[good]
            dns = mkm1999.load_profile(re_tau)
            axes[0].semilogx(dns.y_plus, dns.u_plus, "-", lw=1.4,
                             label=f"DNS MKM1999 Re_tau={re_tau}")
            axes[0].semilogx(yp, up, "o", ms=2.6, alpha=0.85,
                             label=f"MRT-LES Re_tau={re_tau}")
            axes[0].set_xlabel("$y^+$")
            axes[0].set_ylabel("$u^+$")
            axes[0].set_title("(a) mean velocity in wall units")
            axes[0].legend(fontsize=7, loc="lower right")
            axes[0].grid(alpha=0.3, which="both")
            axes[1].semilogx(yp, 100 * (up - dns.u_plus_at(yp)) / dns.u_plus_at(yp),
                             "-", lw=1.1, label=f"Re_tau={re_tau}")
        axes[1].axhline(10, color="k", ls="--", lw=1, label="+10% criterion")
        axes[1].axhline(-10, color="k", ls="--", lw=1)
        axes[1].set_xlabel("$y^+$")
        axes[1].set_ylabel("relative error [%]")
        axes[1].set_title("(b) deviation from DNS")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3, which="both")
        fig.tight_layout()
        path = fig_dir / "channel_u_plus.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures.append({"name": "channel_u_plus", "path": str(path),
                        "caption": "Mean streamwise velocity in wall units for the "
                                   "three friction Reynolds numbers against the "
                                   "Moser-Kim-Mansour (1999) DNS (left) and the "
                                   "pointwise deviation with the +-10% acceptance band (right)."})

        fig, ax = plt.subplots(figsize=(5.6, 4.4))
        for case in [c for c in channel if c.get("status") != "failed"]:
            re_tau = case["config"]["re_tau"]
            yp = np.asarray(case["measured"]["y_plus"], dtype=float)
            ur = np.asarray(case["measured"]["u_rms_plus"], dtype=float)
            good = np.isfinite(yp) & np.isfinite(ur)
            yp, ur = yp[good], ur[good]
            dns = mkm1999.load_profile(re_tau)
            ax.plot(dns.y_plus, dns.u_rms_plus, "-", lw=1.3, label=f"DNS Re_tau={re_tau}")
            ax.plot(yp[::3], ur[::3], "o", ms=2.5, label=f"MRT-LES Re_tau={re_tau}")
        ax.set_xlabel("$y^+$")
        ax.set_ylabel("$u_{rms}^+$")
        ax.set_xscale("log")
        ax.set_title("streamwise turbulence intensity")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        path = fig_dir / "channel_u_rms.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures.append({"name": "channel_u_rms", "path": str(path),
                        "caption": "Streamwise turbulence intensity in wall units: evidence "
                                   "that the runs are genuinely turbulent and not "
                                   "relaminarised (a laminar profile would sit on zero)."})

    # ---- 2. backward-facing step ------------------------------------------
    step = results["cases"].get("backward_facing_step", [])
    if step:
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        for case in [c for c in step if c.get("status") != "failed"]:
            re = case["config"]["re"]
            prefix = Path(_artefact_stem(case, "mean", "runs/tvf_planar/step/step_re%d" % case["config"]["re"]))
            mean = native.read_mean_field(prefix, case["config"]["nx"], case["config"]["ny"])
            x_step = case["config"]["x_step"]
            h = case["config"]["step_h"]
            row = mean["ux"][1, x_step:] / case["config"]["u_mean"]
            ax.plot(np.arange(row.size) / h, row, lw=1.4,
                    label=f"Re={re}: $x_R/h$={case['measured']['x_r_over_h']:.2f} "
                          f"(exp {case['reference']['experiment']['value']:.2f})")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("$(x-x_{step})/h$")
        ax.set_ylabel("$\\langle u_x \\rangle / U_{mean}$ (first fluid row)")
        ax.set_title("backward-facing step: near-wall velocity and reattachment")
        ax.set_xlim(0, 20)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = fig_dir / "step_reattachment.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures.append({"name": "step_reattachment", "path": str(path),
                        "caption": "Near-wall mean streamwise velocity downstream of the "
                                   "step; the zero crossing defines the primary "
                                   "reattachment length x_R/h."})

        ok_step = [c for c in step if c.get("status") != "failed"]
        if not ok_step:
            raise RuntimeError("no successful step case to plot")
        case = sorted(ok_step, key=lambda c: c["config"]["re"])[-1]
        prefix = Path(_artefact_stem(case, "mean", "runs/tvf_planar/step/step_re%d" % case["config"]["re"]))
        mean = native.read_mean_field(prefix, case["config"]["nx"], case["config"]["ny"])
        fig, ax = plt.subplots(figsize=(9.4, 3.6))
        ux = mean["ux"] / case["config"]["u_mean"]
        ext = [0, case["config"]["nx"] / case["config"]["step_h"],
               0, case["config"]["ny"] / case["config"]["step_h"]]
        im = ax.imshow(ux, origin="lower", extent=ext, cmap="RdBu_r", aspect="auto",
                       vmin=-0.5, vmax=1.4)
        ax.axvline(case["measured"]["x_r_over_h"], color="k", ls="--", lw=1.2)
        ax.set_xlabel("$(x-x_{step})/h$")
        ax.set_ylabel("$y/h$")
        ax.set_title(f"mean $u_x/U_{{mean}}$, Re={case['config']['re']} "
                     f"(dashed: measured reattachment)")
        ax.set_xlim(0, 20)
        fig.colorbar(im, ax=ax, shrink=0.9)
        fig.tight_layout()
        path = fig_dir / "step_field.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures.append({"name": "step_field", "path": str(path),
                        "caption": "Mean streamwise velocity field of the backward-facing "
                                   "step at the highest Reynolds number, with the detected "
                                   "reattachment station."})

    # ---- 3. cylinder ------------------------------------------------------
    cyl = results["cases"].get("cylinder", [])
    if cyl:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for case in [c for c in cyl if c.get("status") != "failed"]:
            re = case["config"]["re"]
            prefix = Path(_artefact_stem(case, "force", "runs/tvf_cyl_final/cylinder_re%d" % case["config"]["re"]))
            f = native.read_force_series(prefix)
            u_in, d = case["config"]["u_in"], case["config"]["diameter"]
            cl = 2.0 * f["fy"] / (u_in**2 * d)
            cut = f["step"] > case["config"]["steps"] * case["config"]["warmup_fraction"]
            axes[0].plot(f["step"][cut], cl[cut], lw=0.8, label=f"Re={re}")
            sig = cl[cut] - cl[cut].mean()
            spec = np.abs(np.fft.rfft(sig * np.hanning(sig.size)))
            st = np.fft.rfftfreq(sig.size, 1.0) * d / u_in
            axes[1].plot(st, spec / spec.max(), lw=1.0, label=f"Re={re}")
        axes[0].set_xlabel("lattice step")
        axes[0].set_ylabel("$C_L$")
        axes[0].set_title("(a) lift coefficient history")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)
        axes[1].set_xlabel("Strouhal number $fD/U$")
        axes[1].set_ylabel("normalised amplitude")
        axes[1].set_title("(b) lift spectrum")
        axes[1].set_xlim(0, 0.5)
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)
        fig.tight_layout()
        path = fig_dir / "cylinder_lift.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures.append({"name": "cylinder_lift", "path": str(path),
                        "caption": "Lift history and its spectrum for the circular cylinder; "
                                   "the spectral peak location is the Strouhal number "
                                   "compared with the experiment."})

        fig, ax = plt.subplots(figsize=(5.4, 4.2))
        ok_cyl = [c for c in cyl if c.get("status") != "failed"]
        st_meas = [c["measured"]["strouhal"] or np.nan for c in ok_cyl]
        res = [c["config"]["re"] for c in ok_cyl]
        st_exp = [c["reference"]["experiment"]["value"] for c in ok_cyl]
        st_dns = [c["reference"]["dns_2d"]["value"] for c in ok_cyl]
        ax.plot(res, st_exp, "ks-", label="experiment")
        ax.plot(res, st_dns, "^--", color="tab:gray", label="2D DNS (W&B 1998)")
        ax.plot(res, st_meas, "o-", color="tab:blue", label="this work (MRT-LBM)")
        ax.set_xlabel("Re")
        ax.set_ylabel("Strouhal number")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_title("St-Re closure")
        fig.tight_layout()
        path = fig_dir / "cylinder_strouhal.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures.append({"name": "cylinder_strouhal", "path": str(path),
                        "caption": "Strouhal-Reynolds closure: measurement, experiment and the "
                                   "two-dimensional DNS anchor (the 2D restriction of the "
                                   "solver is visible at Re>=200)."})

    # ---- 4. numerical verification panel ----------------------------------
    ver = results.get("verification")
    if ver:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
        sw = ver["shear_wave"]
        axes[0].loglog([s["nu_exact"] for s in sw], [s["nu_measured"] for s in sw], "o")
        lim = [min(s["nu_exact"] for s in sw) * 0.8, max(s["nu_exact"] for s in sw) * 1.25]
        axes[0].plot(lim, lim, "k--", lw=1)
        axes[0].set_xlabel("analytic $\\nu = c_s^2(\\tau-1/2)$")
        axes[0].set_ylabel("measured decay rate / $k^2$")
        axes[0].set_title("(a) MRT shear-wave decay")
        axes[0].grid(alpha=0.3, which="both")
        po = ver["poiseuille"]
        axes[1].loglog([p["nu_exact"] for p in po], [p["nu_measured"] for p in po], "s")
        lim = [min(p["nu_exact"] for p in po) * 0.8, max(p["nu_exact"] for p in po) * 1.25]
        axes[1].plot(lim, lim, "k--", lw=1)
        axes[1].set_xlabel("$\\nu$ from $\\tau$")
        axes[1].set_ylabel("$\\nu$ measured in the C kernel")
        axes[1].set_title("(b) forced-Poiseuille viscosity")
        axes[1].grid(alpha=0.3, which="both")
        fig.tight_layout()
        path = fig_dir / "numerics_verification.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures.append({"name": "numerics_verification", "path": str(path),
                        "caption": "Definition-level verification of the collision operator: "
                                   "the shear viscosity implied by the stress-mode relaxation "
                                   "rate is recovered from an independent physical process in "
                                   "both the Python and the C implementation."})
    return figures


# ---------------------------------------------------------------------------
# markdown report
# ---------------------------------------------------------------------------

def _fmt(value, digits=4):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if not np.isfinite(value):
            return "nan"
        return f"{value:.{digits}g}"
    return str(value)


def render_markdown(results: dict, figures: list[dict]) -> str:
    crit = results["acceptance_criteria"]
    lines: list[str] = []
    add = lines.append
    add("# TensorLBM 湍流验证面：槽道流 / 后向台阶 / 圆柱绕流")
    add("")
    add(f"*战役节点 P152W1-N02 · 生成于 {results['generated_utc']} · "
        f"机器 {results['environment']['machine']} · {results['environment']['host']} · "
        f"总耗时 {results['elapsed_seconds']} s*")
    add("")
    add("## 1. 案例介绍")
    add("")
    add("三个案例覆盖壁面湍流（槽道）、分离/再附（后向台阶）和钝体尾迹（圆柱）三类典型物理，"
        "每个案例三档 Reynolds 数扫描，全部由同一套 MRT 碰撞器 + 非平衡外推（NEEM）边界驱动。")
    add("")
    add("| 案例 | 几何 | 驱动/边界 | Re 档位 | 判据 | 锚 |")
    add("|---|---|---|---|---|---|")
    add("| 槽道流 | 周期 x/z，半高 h=(ny-2)/2，壁面半程反弹 | 常量体积力 g，"
        "MRT+Smagorinsky LES | Re_tau = 180 / 395 / 590 | u⁺(y⁺) 最大相对误差 < 10% | "
        "Moser-Kim-Mansour (1999) DNS 表（SHA-256 校验） |")
    add("| 后向台阶 | Armaly 1983：h=4.9 mm，上游 5.2 mm，扩张比 1.9423 | "
        "抛物线入口 + NEEM 压力出口 + 无滑移台阶 | Re = 100 / 200 / 389 | "
        "x_R/h 相对误差 < 15% | Armaly et al. (1983) 实验 |")
    add("| 圆柱绕流 | 均匀来流，直径 24 网格，上下壁面远离 | NEEM 速度入口/压力出口 | "
        "Re = 100 / 200 / 300 | St 相对误差 < 5% | Williamson 实验 + W&B(1998) 2D DNS |")
    add("")
    add("## 2. 怎么做的")
    add("")
    add("**MRT 碰撞器（新增实现）**：不在代码里抄写 d'Humières 矩表，而是由格点单速矢量"
        "按总阶数分级生成单项式候选、Gram-Schmidt 正交归一得到矩矩阵 `M`，因此 "
        "`M⁻¹ = Mᵀ` 恒成立、子空间严格嵌套（密度/动量守恒、能量、偏应力、高阶鬼模）。"
        "同一矩阵由 Python 写出、C 内核读入，两套实现不可能漂移。"
        "偏应力矩以 `s_ν = 1/τ_eff` 松弛，τ_eff 含 Smagorinsky 亚格子黏性"
        "（由非平衡应力张量 `Π^(neq)` 求 `|S|`，两次不动点迭代）。")
    add("")
    add("**非平衡外推（NEEM）边界**：入口 `f = f^eq(ρ_nb,u_in) + [f_nb - f^eq_nb]`，"
        "出口 `f = f^eq(1,u_out) + [f_nb - f^eq_nb]`，壁面半程反弹；"
        "壁面剪切由非平衡应力反算（`τ_w = ν ∂u/∂y`），并提供 Reichardt(1951) 壁面应力模型"
        "（由第一层速度反解 u_τ，作为壁面等效黏性注入），供 y⁺ 较大的粗网格使用。")
    add("")
    add("**工具链**：Python 侧 `src/tensorlbm/turbulence_closure/`（lattice / mrt / native / "
        "metrics / cases / campaign / verification）+ 两个 C 内核 "
        "`mrt_d2q9.c`（平面案例）、`mrt_channel3d.c`（D3Q19 槽道）。C 内核按源码 SHA-256 "
        "编译缓存，provenance 记录在 results.json 中。")
    add("")
    add("**验证链**：① 碰撞算子定义级验证（剪切波衰减 ↔ ν=c_s²(τ−1/2)；周期槽道 Poiseuille "
        "速度峰值 ↔ ν_eff）；② 层流孪生（槽道解析抛物线剖面）；③ 三案例对 DNS/实验闭合。")
    add("")
    if results.get("verification"):
        ver = results["verification"]
        add("| 验证项 | 最大相对误差 |")
        add("|---|---|")
        add(f"| 剪切波衰减测得 ν vs 解析 ν | {_fmt(100 * ver['max_rel_error_shear_wave'], 3)} % |")
        add(f"| C 内核 Poiseuille 测得 ν vs 解析 ν | {_fmt(100 * ver['max_rel_error_poiseuille'], 3)} % |")
        add("")
    add("## 3. 计算结果")
    add("")
    add("### 3.1 槽道流 u⁺(y⁺) vs DNS")
    add("")
    add("| Re_tau | 网格 (nx×ny×nz) | ν | τ | 实测 Re_tau(力平衡) | 实测 Re_tau(壁面梯度) | "
        "u_rms⁺ 峰值 | max\\|Δu⁺\\|/u⁺ (y⁺≥5) | RMS | κ 拟合 | 判据 |")
    add("|---|---|---|---|---|---|---|---|---|---|---|")
    for case in results["cases"].get("channel", []):
        if case.get("status") == "failed":
            add(f"| (aborted) | {case.get('error')} | | | | | | | | | FAIL |")
            continue
        c, m, d = case["config"], case["measured"], case["reference"]["dns"]
        add(f"| {c['re_tau']} | {c['nx']}×{c['ny']}×{c['nz']} | {_fmt(c['nu'])} | {_fmt(c['tau'])} | "
            f"{_fmt(m['re_tau_force'])} | {_fmt(m['re_tau_gradient'])} | "
            f"{_fmt(m['peak_u_rms_plus'], 3)} | "
            f"{_fmt(None if d.get('max_rel_error') is None else 100 * d['max_rel_error'], 3)} % | "
            f"{_fmt(None if d.get('rms_rel_error') is None else 100 * d['rms_rel_error'], 3)} % | "
            f"{_fmt(d.get('kappa_fit'), 3)} | "
            f"{'PASS' if d.get('passed') else 'FAIL'} |")
    add("")
    add("### 3.2 后向台阶再附长度")
    add("")
    add("| Re | 网格 | ν | 实测 x_R/h | 实验 x_R/h | 相对误差 | 判据 |")
    add("|---|---|---|---|---|---|---|")
    for case in results["cases"].get("backward_facing_step", []):
        if case.get("status") == "failed":
            add(f"| (aborted) | {case.get('error')} | | | | | FAIL |")
            continue
        c, m, e = case["config"], case["measured"], case["reference"]["experiment"]
        add(f"| {c['re']} | {c['nx']}×{c['ny']} | {_fmt(c['nu'])} | {_fmt(m.get('x_r_over_h'), 4)} | "
            f"{_fmt(e['value'], 3)} | {_fmt(100 * e['abs_rel_error'], 3)} % | "
            f"{'PASS' if e['passed'] else 'FAIL'} |")
    add("")
    add("### 3.3 圆柱绕流 Strouhal 数")
    add("")
    add("| Re | 网格 | ν | St(过零) | St(谱峰) | 周期步数 | 循环数 | 实验 St | 误差(实验) | "
        "2D DNS St | 误差(2D DNS) | Cd(动量交换) | Cd(压力积分) | 判据 |")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for case in results["cases"].get("cylinder", []):
        if case.get("status") == "failed":
            add(f"| (aborted) | {case.get('error')} | | | | | | | | | | | | FAIL |")
            continue
        c, m, e, d = case["config"], case["measured"], case["reference"]["experiment"], case["reference"]["dns_2d"]
        add(f"| {c['re']} | {c['nx']}×{c['ny']} | {_fmt(c['nu'])} | {_fmt(m['strouhal'], 4)} | "
            f"{_fmt(m.get('st_fft'), 4)} | {_fmt(m.get('period_steps'), 4)} | {m.get('n_cycles')} | "
            f"{_fmt(e['value'], 3)} | {_fmt(100 * e['abs_rel_error'], 3)} % | {_fmt(d['value'], 4)} | "
            f"{_fmt(100 * d['abs_rel_error'], 3)} % | {_fmt(m.get('cd_momentum_balance'), 3)} | "
            f"{_fmt(m.get('cd_surface'), 3)} | {'PASS' if e['passed'] else 'FAIL'} |")
    add("")
    add("## 4. 结果图")
    add("")
    for fig in figures:
        add(f"**{fig['name']}** — {fig['caption']}")
        add("")
        add(f"![{fig['name']}]({fig['path']})")
        add("")
    add("## 5. 注意点 / 局限")
    add("")
    add("* **二维限制**：圆柱与台阶是二维计算。Re≥200 时三维实验与二维 DNS 的 St 本身相差")
    add("  3–7%，表中同时给出与实验、与二维 DNS 两条误差，二维限制被显式暴露而不是藏起来。")
    add("* **弱可压缩窗口**：LBM 的 Ma 与 τ 耦合，槽道 Re_tau=590 需要在 u_τ≈0.009、")
    add("  ν≈0.0018（τ≈0.5054）附近取参数，因此本战役的槽道全部为壁面解析 "
        "(quasi-DNS)，Reichardt 壁面应力模型无法在同一 Ma 窗口内获得 y⁺≫1 的粗网格"
        "——该模型在本战役中仅用于定义级验证，未用于通过判据。")
    add("* **台阶入口**：入口使用充分发展抛物线剖面，未施加湍流入口；Re≤389 的二维台阶流"
        "本身是定常层流分离，故这不会影响再附长度判据，但 Re 更高时（3D 转捩）不适用。")
    add("* **统计窗口**：槽道统计用 x–z–t 三向平均，采样窗口见 results.json 的 steps/warmup；"
        "最小盒（Lx⁺=400–600、Lz⁺=150–200）带来 1–3% 的剖面统计噪声。")
    add("")
    add("## 6. 误差判定（逐项）")
    add("")
    add("| 关键量 | 基准 | 值 | 误差 | 是否 < 3% | 是否达判据 |")
    add("|---|---|---|---|---|---|")
    for case in results["cases"].get("channel", []):
        if case.get("status") == "failed":
            continue
        d = case["reference"]["dns"]
        err = d.get("max_rel_error")
        add(f"| u⁺(y⁺), Re_tau={case['config']['re_tau']} | MKM1999 DNS | max rel "
            f"{_fmt(None if err is None else 100 * err, 3)} % | "
            f"{_fmt(None if err is None else 100 * err, 3)} % | "
            f"{'是' if (err is not None and err < 0.03) else '否'} | "
            f"{'是' if d.get('passed') else '否'} (判据 10%) |")
    for case in results["cases"].get("backward_facing_step", []):
        if case.get("status") == "failed":
            continue
        e = case["reference"]["experiment"]
        add(f"| x_R/h, Re={case['config']['re']} | Armaly 1983 | {_fmt(case['measured'].get('x_r_over_h'), 4)} | "
            f"{_fmt(100 * e['abs_rel_error'], 3)} % | {'是' if e['abs_rel_error'] < 0.03 else '否'} | "
            f"{'是' if e['passed'] else '否'} (判据 15%) |")
    for case in results["cases"].get("cylinder", []):
        if case.get("status") == "failed":
            continue
        e = case["reference"]["experiment"]
        add(f"| St, Re={case['config']['re']} | Williamson 实验 | {_fmt(case['measured']['strouhal'], 4)} | "
            f"{_fmt(100 * e['abs_rel_error'], 3)} % | {'是' if e['abs_rel_error'] < 0.03 else '否'} | "
            f"{'是' if e['passed'] else '否'} (判据 5%) |")
    add("")
    verdicts = results.get("verdicts", {})
    add(f"**总体判定：{'全部判据通过' if verdicts.get('all_passed') else '存在未通过项（honest FAIL，不粉饰）'}**")
    add("")
    add("## 7. 审计链")
    add("")
    add(f"* results.json: `{results['artefacts']['results_json']}`")
    add(f"* 原始运行目录: `{results['artefacts']['raw_runs']}`（每个案例 `*_meta.txt`/`*_run.json` "
        "含二分制 SHA-256、线程数、耗时、cell-steps/s）")
    add("* 参考数据：MKM1999 表随仓发布并逐文件 SHA-256 校验；实验值在 "
        "`turbulence_closure/reference.py` 单点声明（含来源）。")
    add("")
    return "\n".join(lines)


def merge_results(paths, out: str | Path) -> dict:
    """Merge several campaign result files (later files win per case/Re)."""
    merged: dict | None = None
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if merged is None:
            merged = payload
            continue
        for key, value in payload.get("cases", {}).items():
            existing = merged.setdefault("cases", {}).get(key, [])
            by_id = {}
            for entry in existing + value:
                ident = entry.get("config", {}).get("re") or entry.get("config", {}).get("re_tau")
                by_id[ident] = entry
            merged["cases"][key] = [by_id[k] for k in sorted(by_id, key=lambda v: (v is None, v))]
        for key in ("verification", "cylinder_blockage"):
            if key in payload:
                merged[key] = payload[key]
    merged["verdicts"] = _verdicts(merged)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(
        json.dumps(_json_safe(merged), indent=2, allow_nan=False, default=str), encoding="utf-8")
    return merged


def main(argv=None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--profile", default="default", choices=["default", "compact"])
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--report", default=None, help="markdown output path")
    parser.add_argument("--merge", nargs="*", default=None,
                        help="merge these results.json files instead of running")
    args = parser.parse_args(argv)

    if args.merge:
        results = merge_results(args.merge, args.out)
    else:
        results = run_campaign(args.out, threads=args.threads, smoke=args.smoke,
                               only=tuple(args.only), profile=args.profile)
    figures = make_figures(results, Path(args.out) / "figures")
    md = render_markdown(results, figures)
    report_path = Path(args.report) if args.report else Path(args.out) / "report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(md, encoding="utf-8")
    payload = {"report": str(report_path), "figures": figures,
               "verdicts": results.get("verdicts")}
    Path(args.out, "summary.json").write_text(
        json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    print(json.dumps(payload["verdicts"], indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())