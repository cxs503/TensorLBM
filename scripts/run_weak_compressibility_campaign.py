#!/usr/bin/env python3
"""Campaign driver for the IN130-N03 weak-compressibility verification surface.

The campaign is split into per-Mach-number tasks so that it can be sharded over
the compute cluster.  Every task writes one JSON file into ``--out``; the
``aggregate`` task collects them, applies the power-law fits and emits the
verification record (criteria + provenance) consumed by the report.

Examples
--------
Per-Mach shards (cluster)::

    python scripts/run_weak_compressibility_campaign.py \
        --task cylinder --machs 0.01 --out artifacts/IN130-N03/raw
    python scripts/run_weak_compressibility_campaign.py \
        --task aggregate --out artifacts/IN130-N03

Local smoke run::

    python scripts/run_weak_compressibility_campaign.py --task all --quick
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

# Pin BLAS to one thread *before* numpy is imported.  The D2Q9 kernels here are
# tiny-matrix reductions; letting OpenBLAS spawn one thread per core on a
# 128-core node makes ``macroscopic`` ~45x slower through thread contention
# (measured: 20.7 ms vs 0.46 ms per call on an 81x210 grid).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from tensorlbm import weak_compressibility as wc
from tensorlbm.weak_compressibility import fit_power_law

DEFAULT_MACHS = (0.01, 0.05, 0.1, 0.2, 0.3)
# The Re=100 cylinder sweep cannot include Ma=0.01: at fixed D=20 lattice cells,
# nu = u_max D / Re gives tau = 0.5035 there, i.e. the BGK/TRT limit (measured:
# both diverge within ~2e3 steps).  The low-Ma corner is probed separately by
# ``--task lowma_probe`` and reported as a documented scope limit.
CYLINDER_MACHS = (0.05, 0.1, 0.2, 0.3)


def _json_safe(obj):
    """Recursively convert numpy containers/scalars to plain Python types."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def task_poiseuille(args) -> list[Path]:
    """Body-force, periodic-streamwise plane Poiseuille (one file per Ma).

    With ``--hold-viscosity`` (default) the kinematic viscosity -- hence ``tau``
    and the lattice wall slip -- is held fixed while ``Ma`` varies, so that the
    Ma-dependence of the error is not contaminated by a tau-dependence of the
    discretisation offset.  The contrast with the pressure-driven channel is
    then like-for-like.
    """
    out = []
    for ma in args.machs:
        t0 = time.time()
        re_i = (
            ma * wc.CS * float(args.ny - 1) / args.nu_fixed
            if (args.hold_viscosity and args.poiseuille_hold_viscosity)
            else args.re
        )
        setup = wc.PoiseuilleSetup(
            mach=ma,
            re=re_i,
            ny=args.ny,
            nx=args.nx,
            collision=args.collision,
            max_steps=args.max_steps,
        )
        rec = wc.run_poiseuille(setup)
        rec["wall_seconds"] = time.time() - t0
        rec["task"] = "poiseuille"
        rec["host"] = platform.node()
        p = args.out / "raw" / f"poiseuille_ma{ma:.3f}.json"
        _dump(rec, p)
        out.append(p)
        print(f"[poiseuille] Ma={ma:.3f} err_inf={rec['err_inf']:.4e} "
              f"steps={rec['steps']} {rec['wall_seconds']:.1f}s -> {p}", flush=True)
    return out


def task_pressure(args) -> list[Path]:
    """Velocity-inlet / pressure-outlet channel (one file per Ma)."""
    out = []
    for ma in args.machs:
        t0 = time.time()
        re_i = (
            ma * wc.CS * float(args.ny - 1) / args.nu_fixed
            if args.hold_viscosity
            else args.re
        )
        setup = wc.PressureChannelSetup(
            mach=ma, re=re_i, ny=args.ny, nx=args.nx, steps=args.pressure_steps
        )
        rec = wc.run_channel_pressure_driven(setup)
        rec["wall_seconds"] = time.time() - t0
        rec["task"] = "pressure_channel"
        rec["host"] = platform.node()
        p = args.out / "raw" / f"pressure_ma{ma:.3f}.json"
        _dump(rec, p)
        out.append(p)
        print(f"[pressure] Ma={ma:.3f} err_inf={rec['err_inf']:.4e} "
              f"drho={rec['drho_over_rho0']:.3e} steps={rec['steps']} "
              f"{rec['wall_seconds']:.1f}s -> {p}", flush=True)
    return out


def task_lowma_probe(args) -> list[Path]:
    """Probe the low-Ma corner of the cylinder: is ``tau -> 1/2`` accessible?

    At fixed ``Re`` and fixed diameter in lattice units, ``nu = u_max D / Re``
    shrinks linearly with ``Ma``, i.e. ``tau -> 1/2``: the low-Mach corner of a
    fixed-resolution LBM is *stiff*, and this task documents where the BGK
    solver stops being stable.  It writes ``lowma_cylinder_ma*.json`` (a name
    that deliberately does not match the ``cylinder_ma*`` shard glob, so the
    probe never contaminates the sweep).
    """
    out: list[Path] = []
    for ma in args.lowma_probe_machs:
        t0 = time.time()
        setup = wc.CylinderSetup(
            mach=float(ma), re=args.re_cyl, nx=args.cyl_nx, ny=args.cyl_ny,
            diameter=args.cyl_d, collision=args.collision,
            transient_tu=0.0, max_tu=args.lowma_max_tu, min_periods=1,
            sample_interval=50,
        )
        rec = wc.run_cylinder(setup)
        rec["wall_seconds"] = time.time() - t0
        rec["task"] = "lowma_probe"
        rec["host"] = platform.node()
        p = args.out / "raw" / f"lowma_cylinder_ma{float(ma):.3f}.json"
        _dump(rec, p)
        out.append(p)
        print(f"[lowma] Ma={float(ma):.3f} tau={rec['tau']:.4f} "
              f"blowup_step={rec['blowup_step']} steps={rec['steps']} "
              f"drho={rec['drho_over_rho0']:.3e} -> {p}", flush=True)
    return out


def task_acoustic(args) -> list[Path]:
    """Acoustic-filter experiment at the lowest Mach number (Ma = 0.01)."""
    t0 = time.time()
    ma = min(args.machs) if args.machs else 0.01
    rec = wc.acoustic_filter_check(
        mach=ma,
        nu=args.acoustic_nu,
        ny=args.ny,
        nx=args.nx,
        steps=args.acoustic_steps,
        alpha=args.acoustic_alpha,
        pulse=args.acoustic_pulse,
    )
    rec["wall_seconds"] = time.time() - t0
    rec["task"] = "acoustic"
    rec["host"] = platform.node()
    p = args.out / "raw" / f"acoustic_ma{ma:.3f}.json"
    _dump(rec, p)
    print(
        f"[acoustic] Ma={ma:.3f} raw_ac={rec['standard']['drho_acoustic_amplitude']:.3e} "
        f"filtered_ac={rec['filtered']['drho_acoustic_amplitude']:.3e} "
        f"pass_band={rec['pass_acoustic_band']} {rec['wall_seconds']:.1f}s -> {p}",
        flush=True,
    )
    return [p]


def task_cylinder(args) -> list[Path]:
    """Re=100 circular cylinder (one file per Ma)."""
    out = []
    for ma in args.machs:
        t0 = time.time()
        setup = wc.CylinderSetup(
            mach=ma,
            re=args.re_cyl,
            nx=args.cyl_nx,
            ny=args.cyl_ny,
            diameter=args.cyl_d,
            collision=args.collision,
            transient_tu=args.transient_tu,
            min_periods=args.min_periods,
            max_tu=args.max_tu,
            sample_interval=args.sample_interval,
        )
        rec = wc.run_cylinder(setup)
        rec["wall_seconds"] = time.time() - t0
        rec["task"] = "cylinder"
        rec["host"] = platform.node()
        p = args.out / "raw" / f"cylinder_ma{ma:.3f}.json"
        _dump(rec, p)
        out.append(p)
        print(
            f"[cylinder] Ma={ma:.3f} St={rec['st']:.5f} CD={rec['cd_mean']:.4f} "
            f"CL_amp={rec['cl_amplitude']:.4f} steps={rec['steps']} "
            f"{rec['wall_seconds']:.1f}s -> {p}",
            flush=True,
        )
    return out


def _dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(obj), indent=1))


def _load_all(raw: Path, prefix: str) -> list[dict]:
    files = sorted(raw.glob(f"{prefix}_ma*.json"))
    return [json.loads(p.read_text()) for p in files]


def _safe_max(a):
    """``max|a|`` that maps an all-NaN array to NaN instead of raising."""
    a = np.asarray(a, dtype=float)
    finite = np.isfinite(a)
    if not finite.any():
        return float("nan")
    return float(np.max(np.abs(a[finite])))


def task_aggregate(args) -> list[Path]:
    """Collect per-Ma shards, fit the power laws, emit the verification record."""
    raw = args.out / "raw"
    pois = _load_all(raw, "poiseuille")
    pres_fixed_nu = _load_all(raw, "pressure")
    raw_re = raw.parent / "raw_re"
    pres_re = _load_all(raw_re, "pressure") if raw_re.exists() else []
    # the *fixed-Re* channel sweep is the primary protocol (it is the only one in
    # which |rho-rho_0| ~ O(Ma^2): at fixed nu the pressure drop, hence the
    # density variation, scales like Ma^1 by construction)
    pres = pres_re or pres_fixed_nu
    cyls = _load_all(raw, "cylinder")
    acs = _load_all(raw, "acoustic")
    if not pois or not pres or not cyls:
        raise SystemExit(
            f"missing shards under {raw}: poiseuille={len(pois)} pressure={len(pres)} "
            f"cylinder={len(cyls)}"
        )

    def _sweep(runs: list[dict], case: str) -> dict:
        mas = [r["mach_measured"] for r in runs]
        errs = [r["err_inf"] for r in runs]
        drhos = [r["drho_over_rho0"] for r in runs]
        return {
            "case": case,
            "runs": runs,
            "machs": mas,
            "err_inf": errs,
            "err_l2": [r["err_l2"] for r in runs],
            "drho_over_rho0": drhos,
            "nu": [r["nu"] for r in runs],
            "tau": [r["tau"] for r in runs],
            "fit_error_raw": wc.fit_power_law(mas, errs),
            "fit_error_excess": wc.fit_power_law(mas, errs, subtract_offset=True),
            "fit_drho": wc.fit_power_law(mas, drhos),
        }

    poiseuille = _sweep(pois, "poiseuille_mach_sweep")
    pressure = _sweep(pres, "pressure_channel_mach_sweep")

    mas = np.array([r["mach_measured"] for r in cyls], dtype=float)
    def _num(v):
        return float("nan") if v is None else float(v)

    st = np.array([_num(r["st"]) for r in cyls], dtype=float)
    cd = np.array([_num(r["cd_mean"]) for r in cyls], dtype=float)
    cl = np.array([_num(r["cl_amplitude"]) for r in cyls], dtype=float)
    order = np.argsort(mas)
    mas, st, cd, cl = mas[order], st[order], cd[order], cl[order]
    # nan-safe: a diverged shard stores NaN (serialised as null), and the
    # cylinder sub-face must then be reported as not verified rather than crash
    nan = float("nan")
    st_ref = float(st[0]) if np.isfinite(st[0]) else nan
    cd_ref = float(cd[0]) if np.isfinite(cd[0]) else nan
    cl_ref = float(cl[0]) if np.isfinite(cl[0]) else nan
    with np.errstate(invalid="ignore", divide="ignore"):
        rel_st = (st - st_ref) / st_ref
        rel_cd = (cd - cd_ref) / abs(cd_ref)
        rel_cl = (cl - cl_ref) / abs(cl_ref)
    low = mas <= 0.1 + 1e-12
    cylinder = {
        "case": "cylinder_mach_sweep",
        "runs": cyls,
        "machs": mas.tolist(),
        "st": st.tolist(),
        "st_fft": [r["st_fft"] for r in cyls],
        "cd_mean": cd.tolist(),
        "cl_amplitude": cl.tolist(),
        "cd_rms": [r["cd_rms"] for r in cyls],
        "drho_over_rho0": [r["drho_over_rho0"] for r in cyls],
        "st_ref": st_ref,
        "cd_ref": cd_ref,
        "cl_ref": cl_ref,
        "rel_st": rel_st.tolist(),
        "rel_cd": rel_cd.tolist(),
        "rel_cl": rel_cl.tolist(),
        "max_rel_st_low_ma": _safe_max(rel_st[low]),
        "max_rel_cd_low_ma": _safe_max(rel_cd[low]),
        "max_rel_cl_low_ma": _safe_max(rel_cl[low]),
        "fit_st_extrapolation": wc.fit_power_law(
            mas[:3].tolist(), np.maximum(st[:3], 1e-12).tolist(), subtract_offset=True
        ),
    }

    fixed_nu_block = None
    if pres_re and pres_fixed_nu:
        fixed_nu_block = {
            "note": (
                "same channel with the viscosity held fixed (nu=0.03, tau=0.59) "
                "instead of Re: the driving acceleration a = 8 nu u_max/H^2 then "
                "scales like Ma^1, so |rho-rho_0| ~ Ma^1 by construction -- the "
                "reason the fixed-Re protocol is the primary one"
            ),
            "machs": [r["mach_measured"] for r in pres_fixed_nu],
            "err_inf": [r["err_inf"] for r in pres_fixed_nu],
            "drho_over_rho0": [r["drho_over_rho0"] for r in pres_fixed_nu],
            "fit_drho": fit_power_law([r["mach_measured"] for r in pres_fixed_nu],
                                      [r["drho_over_rho0"] for r in pres_fixed_nu]),
            "fit_error_excess": fit_power_law([r["mach_measured"] for r in pres_fixed_nu],
                                             [r["err_inf"] for r in pres_fixed_nu],
                                             subtract_offset=True),
        }
    conv_raw = raw.parent / "conv" / "raw"
    conv_check = None
    if conv_raw.exists():
        conv_runs = _load_all(conv_raw, "pressure")
        if conv_runs:
            cs = _sweep(conv_runs, "pressure_channel_conv_check")
            conv_check = {
                "steps": [r["steps"] for r in conv_runs],
                "exponent_excess": cs["fit_error_excess"].get("exponent_excess"),
                "offset": cs["fit_error_excess"].get("offset"),
                "err_inf": cs["err_inf"],
                "drho_over_rho0": cs["drho_over_rho0"],
            }

    acoustic = acs[0] if acs else None
    record = wc.build_verification_record(
        poiseuille=poiseuille,
        pressure_channel=pressure,
        cylinder=cylinder,
        acoustic_filter=acoustic,
    )
    record["provenance"] = {
        "raw_dir": str(raw),
        "raw_files": {
            p.name: _sha256(p) for p in sorted(raw.glob("*.json"))
        },
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if conv_check is not None:
        record["convergence_check"] = conv_check
    if fixed_nu_block is not None:
        record["pressure_channel_fixed_nu"] = fixed_nu_block
    _dump(record, args.out / "verification.json")
    summary = {
        k: {kk: vv for kk, vv in v.items() if kk in ("pass", "exponent_raw", "exponent_excess",
                                                     "max_rel_st_low_ma", "max_rel_cd_low_ma",
                                                     "max_rel_cl_low_ma", "cylinder_exponent",
                                                     "pressure_channel_exponent", "limit")}
        for k, v in record["criteria"].items()
    }
    _dump(summary, args.out / "criteria_summary.json")
    print(json.dumps(_json_safe(summary), indent=1), flush=True)
    return [args.out / "verification.json", args.out / "criteria_summary.json"]


def task_all(args) -> list[Path]:
    outs: list[Path] = []
    outs += task_poiseuille(args)
    outs += task_pressure(args)
    outs += task_cylinder(args)
    outs += task_acoustic(args)
    outs += task_aggregate(args)
    return outs


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="all",
                    choices=["poiseuille", "pressure", "cylinder", "acoustic",
                             "lowma_probe", "aggregate", "all"])
    ap.add_argument("--out", type=Path, default=Path("artifacts/IN130-N03"))
    ap.add_argument("--machs", type=float, nargs="+", default=list(DEFAULT_MACHS))
    ap.add_argument("--quick", action="store_true", help="small grids / short runs")
    ap.add_argument("--collision", default="bgk", choices=["bgk", "trt"])
    # Poiseuille (body force)
    ap.add_argument("--re", type=float, default=20.0)
    ap.add_argument("--ny", type=int, default=33)
    ap.add_argument("--nx", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=60000)
    # pressure-driven channel
    ap.add_argument("--hold-viscosity", action="store_true", default=True)
    ap.add_argument("--no-hold-viscosity", dest="hold_viscosity", action="store_false")
    ap.add_argument("--poiseuille-hold-viscosity", action="store_true", default=True)
    ap.add_argument("--nu-fixed", type=float, default=0.03)
    ap.add_argument("--pressure-steps", type=int, default=0)
    # cylinder
    ap.add_argument("--re-cyl", type=float, default=100.0)
    ap.add_argument("--cyl-nx", type=int, default=420)
    ap.add_argument("--cyl-ny", type=int, default=161)
    ap.add_argument("--cyl-d", type=float, default=20.0)
    ap.add_argument("--transient-tu", type=float, default=12.0)
    ap.add_argument("--max-tu", type=float, default=90.0)
    ap.add_argument("--min-periods", type=int, default=5)
    ap.add_argument("--sample-interval", type=int, default=4)
    # acoustic filter experiment
    ap.add_argument("--acoustic-nu", type=float, default=0.03)
    ap.add_argument("--lowma-probe-machs", type=float, nargs="+", default=[0.01])
    ap.add_argument("--lowma-max-tu", type=float, default=3.0)
    ap.add_argument("--acoustic-steps", type=int, default=20000)
    ap.add_argument("--acoustic-alpha", type=float, default=0.05)
    ap.add_argument("--acoustic-pulse", type=float, default=1e-3)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.quick:
        args.ny = 17
        args.nx = 6
        args.max_steps = 20000
        args.pressure_steps = 6000
        args.cyl_nx, args.cyl_ny, args.cyl_d = 168, 65, 8.0
        args.transient_tu, args.max_tu, args.min_periods = 20.0, 30.0, 3
        args.sample_interval = 2
        args.acoustic_steps = 2000
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "raw").mkdir(parents=True, exist_ok=True)
    tasks = {
        "poiseuille": task_poiseuille,
        "pressure": task_pressure,
        "cylinder": task_cylinder,
        "acoustic": task_acoustic,
        "lowma_probe": task_lowma_probe,
        "aggregate": task_aggregate,
        "all": task_all,
    }
    tasks[args.task](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())