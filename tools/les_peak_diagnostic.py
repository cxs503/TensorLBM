#!/usr/bin/env python3
"""Reynolds-stress peak diagnostic for WMLES flow fields.

Extracts the wall-normal Reynolds-shear-stress profile from an LBM channel
run, locates its peak sub-sample (cubic spline + independent parabola),
compares it with a DNS reference, and — with three grids — splits the error
into a grid part (Richardson/GCI) and a model part.

The library implementation lives in
:mod:`tensorlbm.wall_model_les.peak_diagnostic`; this script is the operator
entry point that reads run artifacts and emits JSON that the acceptance record
consumes.

Usage
-----
    # one grid
    python tools/les_peak_diagnostic.py --profiles artifacts/wall_model_les_peak/profiles_fine.npz

    # DNS reference table (MKM .means, mean profile only)
    python tools/les_peak_diagnostic.py --dns-means benchmarks/pending/channel_turb_3d/dns_ref/mkm590.means

    # three grids + DNS reference -> Richardson/GCI + correction map
    python tools/les_peak_diagnostic.py --compare fine,medium,coarse \
        --dir artifacts/wall_model_les_peak --json artifacts/wall_model_les_peak/peak_diagnostic.json

    # analytic self-test (peak known in closed form)
    python tools/les_peak_diagnostic.py --selftest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tensorlbm.wall_model_les.peak_diagnostic import (  # noqa: E402
    analytic_stress_profile,
    find_stress_peak,
    fit_offset_correction,
    load_channel_means_table,
    reynolds_stress_from_mean_profile,
    richardson_gci,
    spacing_for_tolerance,
)


def profile_from_npz(path: Path, *, u_tau: float | None = None) -> dict:
    """Load a channel run profile and return it in wall units.

    ``uv`` in the run artifact is ``<u'v'>`` in lattice units; the wall-unit
    stress is ``-<u'v'>/u_tau^2``.  The normalising ``u_tau`` defaults to the
    value the run used for its own wall units (stored in the artifact), so the
    wall-unit conversion is self-consistent with ``y_plus``.
    """
    data = np.load(path)
    y_plus = np.asarray(data["y_plus"], dtype=float)
    uv = np.asarray(data["uv"], dtype=float)
    ut = float(data["u_tau"]) if "u_tau" in data else float(u_tau or 1.0)
    return {
        "source": str(path),
        "y_plus": y_plus,
        "uv_plus": -uv / (ut**2),
        "u_tau": ut,
        "nu": float(data["nu"]) if "nu" in data else float("nan"),
        "re_tau_nominal": float(data["re_tau_nominal"]) if "re_tau_nominal" in data else float("nan"),
    }


def diagnose_profile(profile: dict, *, reference: dict | None = None) -> dict:
    """Peak estimate for one profile, optionally against a reference peak."""
    y_cap = 0.5 * float(np.nanmax(profile["y_plus"]))
    peak = find_stress_peak(profile["y_plus"], profile["uv_plus"], y_plus_max=y_cap)
    out = {
        "source": profile["source"],
        "u_tau": profile["u_tau"],
        "nu": profile["nu"],
        "n_samples_in_half": peak.n_samples,
        "spacing_plus": peak.spacing,
        "peak": peak.as_dict(),
    }
    if reference is not None:
        dy = peak.y_peak - reference["peak_y_plus"]
        out["offset_abs"] = dy
        out["offset_rel"] = dy / reference["peak_y_plus"]
        out["value_rel_error"] = (peak.value_peak - reference["peak_value"]) / reference["peak_value"]
    return out


def dns_reference(path: Path) -> dict:
    table = load_channel_means_table(path)
    stress = reynolds_stress_from_mean_profile(table)
    peak = find_stress_peak(stress["y_plus"], stress["uv_plus"], y_plus_max=0.5 * table["re_tau"])
    return {
        "source": str(path),
        "re_tau_tabulated": table["re_tau"],
        "gates": stress["gates"],
        "peak_y_plus": peak.y_peak,
        "peak_value": peak.value_peak,
        "peak": peak.as_dict(),
    }


def compare_grids(names: list[str], directory: Path, *, reference: dict | None) -> dict:
    rows = {}
    peaks = {}
    for name in names:
        path = directory / f"profiles_{name}.npz"
        if not path.exists():
            raise FileNotFoundError(f"missing profile {path}")
        prof = profile_from_npz(path)
        diag = diagnose_profile(prof, reference=reference)
        rows[name] = diag
        peaks[name] = prof

    spacings = [rows[n]["spacing_plus"] for n in names]
    y_peak = [rows[n]["peak"]["y_peak"] for n in names]
    v_peak = [rows[n]["peak"]["value_peak"] for n in names]
    out = {"grids": rows, "order": names}
    out["richardson_y_peak"] = richardson_gci(
        y_peak, spacings, reference=None if reference is None else reference["peak_y_plus"]
    ).as_dict()
    out["richardson_value_peak"] = richardson_gci(
        v_peak, spacings, reference=None if reference is None else reference["peak_value"]
    ).as_dict()
    offsets = [rows[n].get("offset_rel") for n in names]
    if all(o is not None for o in offsets):
        corr = fit_offset_correction(spacings, offsets)
        out["offset_correction"] = corr
        out["spacing_for_15pct"] = spacing_for_tolerance(corr, 0.15)
        y_ext = out["richardson_y_peak"]["value_extrapolated"]
        out["verdict"] = {
            "extrapolated_offset_rel": None
            if reference is None
            else (y_ext - reference["peak_y_plus"]) / reference["peak_y_plus"],
            "observed_order": out["richardson_y_peak"]["observed_order"],
            "converged": out["richardson_y_peak"]["converged"],
            "gci_fine_rel": out["richardson_y_peak"]["gci_fine"] / y_peak[0] if y_peak[0] else None,
        }
    return out


def _selftest() -> dict:
    """Analytic check: the profile peak sits exactly at ``a_plus``."""
    y = np.linspace(0.5, 200.0, 60)
    exact = 32.0
    vals = analytic_stress_profile(y, a_plus=exact, n=1.0)
    est = find_stress_peak(y, vals)
    return {
        "exact_peak": exact,
        "spline_peak": est.y_peak,
        "parabola_peak": est.y_peak_parabola,
        "spline_error": est.y_peak - exact,
        "error_over_spacing": (est.y_peak - exact) / est.spacing,
        "spacing": est.spacing,
    }


def _fmt(value, nd=4):
    try:
        return f"{float(value):.{nd}f}"
    except (TypeError, ValueError):
        return str(value)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profiles", help="channel run profile .npz (needs y_plus, uv, u_tau)")
    ap.add_argument("--dns-means", help="MKM-style .means table used as the reference")
    ap.add_argument("--compare", help="comma-separated grid names, ordered fine -> coarse")
    ap.add_argument("--dir", default=str(REPO_ROOT / "artifacts" / "wall_model_les_peak"),
                    help="directory holding profiles_<name>.npz")
    ap.add_argument("--json", help="write the diagnostic JSON here")
    ap.add_argument("--selftest", action="store_true", help="analytic peak round-trip check")
    args = ap.parse_args(argv)

    if args.selftest:
        result = _selftest()
        print(json.dumps(result, indent=2))
        if args.json:
            Path(args.json).write_text(json.dumps(result, indent=2))
        return 0

    reference = None
    payload: dict = {}
    if args.dns_means:
        reference = dns_reference(Path(args.dns_means))
        payload["reference"] = reference
        print(f"DNS reference: y+_peak={_fmt(reference['peak_y_plus'],2)} "
              f"-<u'v'>+_peak={_fmt(reference['peak_value'])} (Re_tau={_fmt(reference['re_tau_tabulated'],1)})")

    if args.profiles:
        prof = profile_from_npz(Path(args.profiles))
        payload["profile"] = diagnose_profile(prof, reference=reference)
        p = payload["profile"]["peak"]
        print(f"model: y+_peak={_fmt(p['y_peak'],2)} (discrete {_fmt(p['y_peak_discrete'],2)}, "
              f"parabola {_fmt(p['y_peak_parabola'],2)}) value={_fmt(p['value_peak'])} "
              f"spacing={_fmt(p['spacing'],2)}")
        if reference is not None:
            print(f"offset: {_fmt(payload['profile']['offset_abs'],2)} wall units "
                  f"({100*payload['profile']['offset_rel']:.1f} %), value error "
                  f"{100*payload['profile']['value_rel_error']:.1f} %")

    if args.compare:
        names = [n.strip() for n in args.compare.split(",") if n.strip()]
        payload["comparison"] = compare_grids(names, Path(args.dir), reference=reference)
        rc = payload["comparison"]["richardson_y_peak"]
        print(f"Richardson y_peak: p={_fmt(rc['observed_order'],2)} "
              f"extrapolated={_fmt(rc['value_extrapolated'],2)} GCI_fine={_fmt(rc['gci_fine'])} "
              f"converged={rc['converged']}")
        if "verdict" in payload["comparison"]:
            v = payload["comparison"]["verdict"]
            print(f"extrapolated offset: {100*v['extrapolated_offset_rel']:.1f} %")

    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2, default=str))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())