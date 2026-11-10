#!/usr/bin/env python3
"""Three-grid WMLES peak-position study at Re_tau = 590 (P372W2-N01).

The point of the study is *two indicators at once*:

* **stability** -- does the ``Re_tau = 590`` wall-modelled channel run to
  statistical stationarity without a non-finite state (the historical failure
  mode of this module);
* **peak position** -- where the resolved ``-<u'v'>+`` maximum sits, how far it
  is from the DNS value, and how much of that distance is grid quantisation
  (Richardson/GCI on three grids) versus wall-model error.

Grids are geometrically similar in wall units: the box is kept at
``Lx+ = Lz+ ~ 240`` and the wall-normal spacing is halved, so
``dy+ = 7.5 / 15 / 30`` corresponds to ``(ny, nx, nz) = (158, 32, 32) /
(79, 16, 16) / (40, 8, 8)``.  Refining all three directions together is what
makes the Richardson estimate meaningful; refining ``y`` alone would change
the resolved turbulence while pretending the box were the same.

Usage
-----
    python benchmarks/wall_model_les/peak_offset_study.py --grid fine --steps 4500
    python benchmarks/wall_model_les/peak_offset_study.py --grid all
    python benchmarks/wall_model_les/peak_offset_study.py --analyse
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tensorlbm.wall_model_les.channel import (  # noqa: E402
    ChannelWMLESConfig,
    run_channel_wmles,
)
from tensorlbm.wall_model_les.peak_diagnostic import (  # noqa: E402
    find_stress_peak,
    fit_offset_correction,
    load_channel_means_table,
    reynolds_stress_from_mean_profile,
    richardson_gci,
    spacing_for_tolerance,
)

DEFAULT_OUT = REPO_ROOT / "artifacts" / "wall_model_les_peak"
DNS_MEANS = REPO_ROOT / "benchmarks" / "pending" / "channel_turb_3d" / "dns_ref" / "mkm590.means"
DNS_MEANS_SHA256 = "878bc951794516b6136912f8f098cd6f2f96c4bbe7f21a8cea657ffb270d419d"

# (name, ny, nx, nz): geometric refinement in all three directions, box held at
# Lx+ = Lz+ ~ 240 and H+ = 2 Re_tau.
GRIDS = {
    "fine": (158, 16, 16),      # dy+ = 7.5,  Lx+ = Lz+ = 120
    "medium": (79, 16, 16),     # dy+ = 15,   Lx+ = Lz+ = 240
    "coarse": (40, 16, 16),     # dy+ = 30,   Lx+ = Lz+ = 480
    # control: Re_tau = 180, dy+ = 7.5 -- the configuration this solver is
    # verified against DNS (MKM-180), used as the diagnostic's own baseline.
    "control180": (49, 16, 16),
}
# The refinement family is wall-normal only: the lateral resolution is held at
# 16 x 16 lattice cells, so the lateral box *in wall units* grows by 4x from the
# fine to the coarse grid. Co-refining the lateral directions (32/16/8) would
# hold Lx+ fixed but puts the coarse grid at 8 x 8 cells, which cannot hold
# resolved turbulence at dy+ = 30 at all. The confound is therefore deliberate
# and is reported: the peak-position error is dominated by the wall-normal
# spacing, but box-size effects are not excluded by this family.


def run_grid(
    name: str,
    *,
    steps: int,
    averaging_start: int,
    re_tau: float = 590.0,
    u_tau: float = 0.02,
    cs: float = 0.1,
    exchange_offset: int = 1,
    wall_coupling: str = "assimilate",
    out_dir: Path = DEFAULT_OUT,
    seed: int = 0,
) -> dict:
    ny, nx, nz = GRIDS[name]
    config = ChannelWMLESConfig(
        nx=nx,
        nz=nz,
        ny=ny,
        re_tau=float(re_tau),
        u_tau=float(u_tau),
        smagorinsky_cs=float(cs),
        steps=int(steps),
        averaging_start=int(averaging_start),
        stat_interval=10,
        log_interval=max(steps // 20, 1),
        exchange_offset=int(exchange_offset),
        wall_coupling=wall_coupling,
        seed=int(seed),
        diagnostics=True,
        run_name=f"re{re_tau:g}_{name}_dyplus{re_tau / (0.5 * (ny - 1)):.2f}",
    )
    started = time.time()
    error = None
    stats = None
    try:
        stats = run_channel_wmles(config)
    except Exception as exc:  # noqa: BLE001 - a non-converged run is a result
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.time() - started

    history = (stats or {}).get("history") or []
    first_nonfinite = None
    for row in history:
        vals = [v for k, v in row.items() if isinstance(v, float) and k != "step"]
        if any(not math.isfinite(v) for v in vals):
            first_nonfinite = row.get("step")
            break

    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "grid": name,
        "params": {
            "re_tau": float(re_tau),
            "u_tau": float(u_tau),
            "cs": float(cs),
            "ny": ny,
            "nx": nx,
            "nz": nz,
            "steps": int(steps),
            "averaging_start": int(averaging_start),
            "exchange_offset": int(exchange_offset),
            "wall_coupling": wall_coupling,
            "seed": int(seed),
        },
        "derived": {
            "nu": config.nu,
            "tau_molecular": config.tau,
            "tau_minus_half_molecular": config.tau - 0.5,
            "delta_y_plus": config.delta_y_plus,
            "H": config.H,
            "h": config.h,
            "Lx_plus": nx * config.delta_y_plus,
            "Lz_plus": nz * config.delta_y_plus,
            "y_plus_exchange": config.y_of_node(config.j_exchange) * config.u_tau / config.nu,
            "y_plus_first_node": config.y_of_node(config.j_wall_node) * config.u_tau / config.nu,
        },
        "changed": {
            "re_tau_measured": (history[-1].get("re_tau_measured") if history else None),
            "u_tau_model_mean": (stats or {}).get("u_tau_model_mean"),
            "u_tau_model_over_balance": (stats or {}).get("u_tau_model_over_balance"),
            "u_bulk": ((stats or {}).get("profiles") or {}).get("u_bulk"),
            "n_samples": (stats or {}).get("n_samples"),
        },
        "stable": bool(stats is not None and error is None and first_nonfinite is None),
        "first_nonfinite_step": first_nonfinite,
        "error": error,
        "elapsed_s": round(elapsed, 2),
        "history": history,
    }

    if stats is not None:
        prof = stats["profiles"]
        arrays = {}
        for key, value in prof.items():
            if isinstance(value, np.ndarray):
                arrays[key] = value
            elif hasattr(value, "detach"):
                arrays[key] = value.detach().cpu().numpy()
            elif isinstance(value, (int, float, bool)):
                # scalars (nu, u_tau, Re_tau) must travel with the profile:
                # wall-unit re-normalisation downstream depends on them
                arrays[key] = np.asarray(float(value))
        np.savez(out_dir / f"profiles_{name}.npz", **arrays)
        record["profiles_file"] = str(out_dir / f"profiles_{name}.npz")

    (out_dir / f"run_{name}.json").write_text(json.dumps(record, indent=2, default=str))
    print(
        f"[{name}] stable={record['stable']} dy+={record['derived']['delta_y_plus']:.2f} "
        f"Re_tau_meas={record['changed']['re_tau_measured']} "
        f"u_tau/bal={record['changed']['u_tau_model_over_balance']} "
        f"{record['elapsed_s']:.0f}s {error or ''}",
        flush=True,
    )
    return record


def _load_profile(name: str, out_dir: Path) -> dict:
    """Load a grid profile in *measured* wall units.

    The profile stored by the solver is in lattice units and its ``y_plus``
    column uses the *imposed* ``u_tau``.  Wall units are only meaningful with
    the friction velocity the run actually produced, so the profile is
    re-normalised with the run's mean modelled ``u_tau`` (read from
    ``run_<grid>.json``) and the ratio ``u_tau_model / u_tau_imposed`` is
    carried along as the quality gate: a ratio far from 1 means the run is not
    at the nominal ``Re_tau`` and any comparison with the DNS at that nominal
    ``Re_tau`` must be read with that caveat.
    """
    path = out_dir / f"profiles_{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run the grid first")
    data = np.load(path)
    y_plus_imposed = np.asarray(data["y_plus"], dtype=float)
    uv = np.asarray(data["uv"], dtype=float)
    nu = float(data["nu"]) if "nu" in data else float("nan")
    u_tau_imposed = float(data["u_tau"]) if "u_tau" in data else float("nan")
    # rows at y+ <= 0 sit inside the wall (row 0 of the bounce-back layout): they
    # carry no resolved stress and are excluded from the peak search
    keep = y_plus_imposed > 0.0
    y_plus_imposed, uv = y_plus_imposed[keep], uv[keep]
    record_path = out_dir / f"run_{name}.json"
    u_tau_model = float("nan")
    re_tau_nominal = float(data["re_tau_nominal"]) if "re_tau_nominal" in data else float("nan")
    if record_path.exists():
        rec = json.loads(record_path.read_text())
        value = (rec.get("changed") or {}).get("u_tau_model_mean")
        if value is not None and math.isfinite(float(value)):
            u_tau_model = float(value)
        if not math.isfinite(u_tau_imposed):
            u_tau_imposed = float((rec.get("params") or {}).get("u_tau", float("nan")))
        if not math.isfinite(re_tau_nominal):
            re_tau_nominal = float((rec.get("params") or {}).get("re_tau", float("nan")))
    have_both = math.isfinite(u_tau_model) and math.isfinite(u_tau_imposed) and u_tau_imposed > 0.0
    ratio = u_tau_model / u_tau_imposed if have_both else 1.0
    y_plus = y_plus_imposed * ratio
    u_tau_used = u_tau_model if math.isfinite(u_tau_model) else u_tau_imposed
    return {
        "name": name,
        "y_plus": y_plus,
        "uv_plus": -uv / (u_tau_used**2),
        "uv_raw": uv,
        "u_tau": u_tau_used,
        "u_tau_imposed": u_tau_imposed,
        "u_tau_ratio": ratio,
        "re_tau_measured": ratio * re_tau_nominal if math.isfinite(re_tau_nominal) else float("nan"),
        "nu": nu,
        "delta_y_plus": float(np.median(np.diff(y_plus[:5]))) if y_plus.size > 5 else float("nan"),
    }


def analyse(out_dir: Path = DEFAULT_OUT) -> dict:
    """Fold the three grids + DNS reference into the acceptance record."""
    import hashlib

    sha = hashlib.sha256(DNS_MEANS.read_bytes()).hexdigest()
    table = load_channel_means_table(DNS_MEANS)
    dns_stress = reynolds_stress_from_mean_profile(table)
    half = float(table["re_tau"]) / 2.0
    dns_peak = find_stress_peak(dns_stress["y_plus"], dns_stress["uv_plus"], y_plus_max=half)

    reference = {
        "source": str(DNS_MEANS.relative_to(REPO_ROOT)),
        "sha256": sha,
        "sha256_matches_provenance": sha == DNS_MEANS_SHA256,
        "re_tau_tabulated": float(table["re_tau"]),
        "re_tau_nominal": 590.0,
        "peak_y_plus": dns_peak.y_peak,
        "peak_value": dns_peak.value_peak,
        "peak_estimate": dns_peak.as_dict(),
        "stress_identity_gates": dns_stress["gates"],
        "method": "-<u'v'>+ = 1 - y/h - dU+/dy+ from the tabulated mean profile (exact channel balance)",
    }

    profiles = {}
    peaks = {}
    for name in GRIDS:
        try:
            prof = _load_profile(name, out_dir)
        except FileNotFoundError:
            continue
        profiles[name] = prof
        # The Reynolds-shear-stress maximum of a channel lives in the inner
        # layer. Searching the whole half-channel lets a noise-dominated
        # short-window profile put its maximum at the centreline (measured: a
        # seed-to-seed jump from y+ = 23 to y+ = 345 in a 600-step window), so
        # the search band is the inner 25 % of the channel and the unrestricted
        # maximum is kept alongside as a diagnostic.
        # the profile spans the whole channel, so the wall-normal station of the
        # centreline (the channel half-height in wall units) is half the maximum
        h_plus = 0.5 * float(np.nanmax(prof["y_plus"]))
        y_cap = 0.25 * h_plus
        peaks[name] = find_stress_peak(prof["y_plus"], prof["uv_plus"], y_plus_max=y_cap)
        peaks[f"{name}__full"] = find_stress_peak(
            prof["y_plus"], prof["uv_plus"], y_plus_max=0.5 * h_plus
        )

    result = {"reference": reference, "grids": {}, "peaks": {}}
    order = [g for g in ("fine", "medium", "coarse") if g in peaks]
    for name in order:
        prof = profiles[name]
        peak = peaks[name]
        offset_abs = peak.y_peak - dns_peak.y_peak
        # equilibrium gates: a wall-modelled channel profile is only a
        # measurement of the wall model if the run reached force balance
        # (tau_w = rho a h  <=>  u_tau_model ~ u_tau) and if the resolved stress
        # stays inside the total-stress envelope 1 - y+/h+.
        h_plus = 0.5 * float(np.nanmax(prof["y_plus"]))
        envelope = 1.0 - min(peak.y_peak, h_plus) / h_plus
        result["grids"][name] = {
            "u_tau_ratio": prof["u_tau_ratio"],
            "equilibrium_gates": {
                "u_tau_ratio": prof["u_tau_ratio"],
                "u_tau_ratio_ok": bool(abs(prof["u_tau_ratio"] - 1.0) <= 0.05),
                "peak_over_stress_envelope": peak.value_peak / envelope if envelope > 0 else float("nan"),
                "stress_envelope_ok": bool(envelope > 0 and peak.value_peak <= 1.05 * envelope),
                "in_equilibrium": bool(
                    abs(prof["u_tau_ratio"] - 1.0) <= 0.05 and envelope > 0 and peak.value_peak <= 1.05 * envelope
                ),
            },
            "re_tau_measured": prof["re_tau_measured"],
            "delta_y_plus": float(np.median(np.diff(prof["y_plus"][:6]))),
            "y_peak": peak.y_peak,
            "value_peak": peak.value_peak,
            "y_peak_discrete": peak.y_peak_discrete,
            "value_peak_discrete": peak.value_peak_discrete,
            "uncertainty": peak.uncertainty,
            "bracket": list(peak.bracket),
            "n_samples_in_half": peak.n_samples,
            "offset_abs": offset_abs,
            "offset_rel": offset_abs / dns_peak.y_peak,
            "value_rel_error": (peak.value_peak - dns_peak.value_peak) / dns_peak.value_peak,
        }
        result["peaks"][name] = peak.as_dict()
        result["peaks"][f"{name}__full"] = peaks[f"{name}__full"].as_dict()

    # control tier: Re_tau = 180 against its own DNS reference
    if "control180" in peaks:
        mkm180 = REPO_ROOT / "benchmarks" / "pending" / "channel_turb_3d" / "dns_ref" / "mkm180.csv"
        if mkm180.exists():
            raw = np.genfromtxt(mkm180, delimiter=",", names=True)
            y_p = np.asarray(raw["y_plus"], dtype=float)
            u_p = np.asarray(raw["u_plus"], dtype=float)
            re180 = 180.0
            du = np.gradient(u_p, y_p, edge_order=2)
            uv180 = 1.0 - (y_p / y_p[-1]) - du
            ref180 = find_stress_peak(y_p, uv180, y_plus_max=re180 * 0.25)
            peak180 = peaks["control180"]
            result["control"] = {
                "reference": {
                    "source": str(mkm180.relative_to(REPO_ROOT)),
                    "peak_y_plus": ref180.y_peak,
                    "peak_value": ref180.value_peak,
                },
                "measured": {
                    "y_peak": peak180.y_peak,
                    "value_peak": peak180.value_peak,
                    "n_samples_in_half": peak180.n_samples,
                },
                "peak_offset_rel": (peak180.y_peak - ref180.y_peak) / ref180.y_peak,
                "value_rel_error": (peak180.value_peak - ref180.value_peak) / ref180.value_peak,
            }

    # --- Re-matched comparison (supplementary) -----------------------------
    # A run whose measured friction is not the nominal u_tau is not at
    # Re_tau = 590, so comparing its peak position with the Re_tau = 590 DNS is
    # not apples-to-apples.  The repository holds two DNS tiers (180 and 590);
    # their peak positions are interpolated in ln(Re_tau) to give each grid a
    # reference at its own measured Re_tau.  The assumption (the peak position
    # depends on Re_tau only through that two-point log fit) is stated in the
    # report; the primary metric stays the nominal-Re comparison.
    tiers = {"180": 180.0, "590": 590.0}
    tier_peaks = {}
    for label, value in tiers.items():
        path = REPO_ROOT / "benchmarks" / "pending" / "channel_turb_3d" / "dns_ref" / (
            f"mkm{label}.csv" if label == "180" else f"mkm{label}.means"
        )
        if not path.exists():
            continue
        if path.suffix == ".csv":
            raw = np.genfromtxt(path, delimiter=",", names=True)
            y_p = np.asarray(raw["y_plus"], dtype=float)
            u_p = np.asarray(raw["u_plus"], dtype=float)
            du = np.gradient(u_p, y_p, edge_order=2)
            uv = 1.0 - (y_p / y_p[-1]) - du
        else:
            tbl = load_channel_means_table(path)
            stress = reynolds_stress_from_mean_profile(tbl)
            y_p, uv = stress["y_plus"], stress["uv_plus"]
        tier_peaks[value] = find_stress_peak(y_p, uv, y_plus_max=0.25 * value)
    re_matched = {}
    if len(tier_peaks) == 2:
        re_lo, re_hi = sorted(tier_peaks)
        y_lo, y_hi = tier_peaks[re_lo].y_peak, tier_peaks[re_hi].y_peak
        slope = (y_hi - y_lo) / (math.log(re_hi) - math.log(re_lo))
        for name in order:
            re_meas = result["grids"][name].get("re_tau_measured")
            if re_meas is None or not math.isfinite(float(re_meas)):
                continue
            y_ref = y_lo + slope * (math.log(float(re_meas)) - math.log(re_lo))
            y_meas = result["grids"][name]["y_peak"]
            re_matched[name] = {
                "re_tau_measured": float(re_meas),
                "dns_peak_at_measured_re_tau": float(y_ref),
                "offset_abs": float(y_meas - y_ref),
                "offset_rel": float((y_meas - y_ref) / y_ref),
                "within_15pct": bool(abs((y_meas - y_ref) / y_ref) <= 0.15),
            }
    result["re_matched"] = {
        "tier_peaks": {str(k): {"y_peak": v.y_peak, "value_peak": v.value_peak} for k, v in tier_peaks.items()},
        "per_grid": re_matched,
        "model": "y_peak+(Re_tau) interpolated linearly in ln(Re_tau) between the DNS tiers",
    }

    if len(order) == 3:
        spacings = [result["grids"][g]["delta_y_plus"] for g in order]
        y_vals = [result["grids"][g]["y_peak"] for g in order]
        v_vals = [result["grids"][g]["value_peak"] for g in order]
        gci_y = richardson_gci(y_vals, spacings, reference=dns_peak.y_peak)
        gci_v = richardson_gci(v_vals, spacings, reference=dns_peak.value_peak)
        result["grid_convergence"] = {"y_peak": gci_y.as_dict(), "value_peak": gci_v.as_dict()}

        offsets = [result["grids"][g]["offset_rel"] for g in order]
        corr = fit_offset_correction(spacings, offsets)
        result["wall_function_correction"] = {
            "offset_map": corr,
            "spacing_for_15pct": spacing_for_tolerance(corr, 0.15),
            "note": (
                "offset(dy+) is the relative peak-position error of the model; the map is a "
                "correction target for the wall-function sampling height, and "
                "spacing_for_15pct is the wall-normal spacing at which the fitted map "
                "reaches the 15 % acceptance tolerance"
            ),
        }
        # grid-independent verdict: Richardson-extrapolated offset
        y_ext = gci_y.value_extrapolated
        result["verdict"] = {
            # explicit name of the node's primary quantity
            "delta_y_plus_peak_offset_rel_fine": offsets[0],
            "delta_y_plus_peak_offset_rel_extrapolated": (y_ext - dns_peak.y_peak) / dns_peak.y_peak,
            "fine_offset_rel": offsets[0],
            "extrapolated_offset_rel": (y_ext - dns_peak.y_peak) / dns_peak.y_peak,
            "extrapolated_within_15pct": bool(abs((y_ext - dns_peak.y_peak) / dns_peak.y_peak) <= 0.15),
            "grid_independent": bool(gci_y.converged and abs(gci_y.grid_error_fine) / dns_peak.y_peak <= 0.15),
            "observed_order": gci_y.observed_order,
            "gci_fine_rel": gci_y.gci_fine,
        }
    else:
        result["verdict"] = {"note": f"only {len(order)} of 3 grids available: {order}"}

    (out_dir / "peak_offset_study.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", default=None, choices=["fine", "medium", "coarse", "control180", "all"],
                    help="grid to run; default = all three, or none when --analyse is given alone")
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--steps", type=int, default=4500)
    ap.add_argument("--averaging-start", type=int, default=2000)
    ap.add_argument("--re-tau", type=float, default=590.0)
    ap.add_argument("--cs", type=float, default=0.1)
    ap.add_argument("--u-tau", type=float, default=0.02)
    ap.add_argument("--exchange-offset", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    if args.grid is None:
        names = [] if args.analyse else ["fine", "medium", "coarse"]
    elif args.grid == "all":
        names = ["fine", "medium", "coarse"]
    else:
        names = [args.grid]
    for name in names:
        run_grid(
            name,
            steps=args.steps,
            averaging_start=args.averaging_start,
            re_tau=args.re_tau,
            u_tau=args.u_tau,
            cs=args.cs,
            exchange_offset=args.exchange_offset,
            seed=args.seed,
            out_dir=out_dir,
        )
    if args.analyse or args.grid == "all":
        result = analyse(out_dir)
        print(json.dumps(result.get("verdict", {}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())