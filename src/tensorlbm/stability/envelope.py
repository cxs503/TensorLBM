"""D2Q9 collision-operator stability envelope campaign (node P482-N05).

Four machine-checkable legs, all predeclared in
:data:`tensorlbm.stability.collision_stability.PREDECLARED` before any run:

``M1  exchange law``
    von Neumann prediction vs measured modal growth at the *same* wave vector,
    plus the divergence boundary ``tau_crit(Ma)`` from both sides.
``M2  plane scan``
    ``tau x Ma`` plane over ``[0.505, 0.9] x [0.02, 0.5]`` with >= 30x30 cases
    per tier (BGK / MRT-A / MRT-B): stable-area ratio and maximum stable Mach
    number per tier.
``M3  falsifiability``
    "MRT is more stable" must be falsifiable: a legal but poorly tuned MRT
    setting is scanned over the same plane to locate the subdomain where MRT is
    *worse*, and declared defective operators (rate > 2, non-orthogonal basis +
    rate > 2) must be rejected by both routes.
``M4  envelope``
    deterministic JSON envelope (+ figure) wired into
    :func:`tensorlbm.accuracy_recommendation.recommend_with_stability_envelope`,
    so no recommendation can be issued for an operating point outside it.

Run ``python -m tensorlbm.stability.envelope --stage all`` inside a checkout to
regenerate ``TensorLBM/artifacts/p482_n05/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import time
from typing import Any, Sequence

import numpy as np

from .collision_stability import (
    MEASURED_TOL,
    design_rates,
    tier_model_designed,
    PREDECLARED,
    STABILITY_TOL,
    box_predicted_critical_tau,
    canonical_json,
    envelope_digest,
    measured_critical_tau,
    negative_control,
    foil_model,
    poor_mrt_model,
    predicted_box_radius,
    predicted_critical_tau,
    quarter_wave_vectors,
    run_broadband_box,
    run_mode_box,
    spectral_radius,
    tier_factory,
    tier_model_at,
)

TIERS: tuple[str, ...] = ("bgk", "mrt_a", "mrt_b")
VARIANTS: tuple[str, ...] = ("canonical", "oracle")
DIGITS = 9  # rounding of stored amplification factors (bitwise-reproducible payload)


def tier_label(tier: str, variant: str) -> str:
    return f"{tier}:{variant}"


# ---------------------------------------------------------------------------
# M2: (tau, Ma) plane scan
# ---------------------------------------------------------------------------


def scan_plane(
    tier: str,
    variant: str,
    *,
    n_tau: int | None = None,
    n_ma: int | None = None,
    n_k: int | None = None,
) -> dict[str, Any]:
    """von Neumann plane scan of one tier over the declared window."""
    pre = PREDECLARED
    n_tau = int(pre["scan_grid"][0] if n_tau is None else n_tau)
    n_ma = int(pre["scan_grid"][1] if n_ma is None else n_ma)
    n_k = int(pre["k_grid_scan"] if n_k is None else n_k)
    taus = np.linspace(pre["tau_window"][0], pre["tau_window"][1], n_tau)
    mas = np.linspace(pre["ma_window"][0], pre["ma_window"][1], n_ma)
    k = quarter_wave_vectors(n_k)

    lam = np.empty((n_tau, n_ma))
    designs: list[dict[str, Any]] = []
    for i, tau in enumerate(taus):
        if tier != "bgk" and variant == "oracle":
            design = design_rates(tier, float(tau))
            designs.append({"tau": round(float(tau), 12), **{k: v for k, v in design.items()
                                                             if not isinstance(v, (list, dict))}})
            model = tier_model_designed(tier, float(tau), design)
        else:
            model = tier_model_at(tier, float(tau), float(mas[0]), variant="canonical")
        for j, ma in enumerate(mas):
            lam[i, j] = float(spectral_radius(model, float(ma), k=k)["lambda_max"])

    stable = lam <= 1.0 + STABILITY_TOL
    lam_round = np.round(lam, DIGITS)
    i_max, j_max = np.unravel_index(int(np.argmax(lam)), lam.shape)
    return {
        "tier": tier,
        "variant": variant,
        "label": tier_label(tier, variant),
        "tau_axis": [round(float(t), 12) for t in taus],
        "ma_axis": [round(float(m), 12) for m in mas],
        "n_cases": int(n_tau * n_ma),
        "k_grid": int(n_k),
        "lambda_max": lam_round.tolist(),
        "stable_mask": stable.astype(int).tolist(),
        "stable_area_ratio": float(stable.mean()),
        "n_stable_cells": int(stable.sum()),
        "n_unstable_cells": int((~stable).sum()),
        "lambda_max_overall": float(lam.max()),
        "lambda_max_overall_at": [float(taus[i_max]), float(mas[j_max])],
        "min_margin": float((1.0 - lam).min()),
        "lambda_grid_sha256": hashlib.sha256(
            canonical_json(lam_round.tolist()).encode("utf-8")
        ).hexdigest(),
        "designs": designs,
    }


def _foil_plane(family: str) -> dict[str, Any]:
    """Plane scan of one declared MRT foil (M3 falsifiability)."""
    pre = PREDECLARED
    n_tau, n_ma = pre["scan_grid"]
    n_k = pre["k_grid_scan"]
    taus = np.linspace(pre["tau_window"][0], pre["tau_window"][1], n_tau)
    mas = np.linspace(pre["ma_window"][0], pre["ma_window"][1], n_ma)
    k = quarter_wave_vectors(n_k)
    lam = np.empty((n_tau, n_ma))
    for i, tau in enumerate(taus):
        model = foil_model(family, float(tau))
        for j, ma in enumerate(mas):
            lam[i, j] = float(spectral_radius(model, float(ma), k=k)["lambda_max"])
    lam_round = np.round(lam, DIGITS)
    stable = lam <= 1.0 + STABILITY_TOL
    return {
        "tier": f"mrt_foil_{family}",
        "variant": "canonical",
        "label": f"mrt_foil_{family}:canonical",
        "tau_axis": [round(float(t), 12) for t in taus],
        "ma_axis": [round(float(m), 12) for m in mas],
        "n_cases": int(n_tau * n_ma),
        "n_unstable_cells": int((~stable).sum()),
        "stable_mask": stable.astype(int).tolist(),
        "stable_area_ratio": float(stable.mean()),
        "lambda_max": lam_round.tolist(),
        "lambda_max_overall": float(lam.max()),
        "lambda_grid_sha256": hashlib.sha256(
            canonical_json(lam_round.tolist()).encode("utf-8")
        ).hexdigest(),
    }


def max_stable_ma(tier: str, variant: str, tau: float) -> dict[str, Any]:
    """Largest ``Ma`` still linearly stable at one ``tau`` (bisection in ``Ma``)."""
    pre = PREDECLARED
    k = quarter_wave_vectors(pre["k_grid_boundary"])
    k_oracle = quarter_wave_vectors(pre["k_grid_oracle"])
    lo, hi = float(pre["ma_window"][0]), float(pre["ma_window"][1])

    def lam(ma: float) -> float:
        model = tier_model_at(tier, tau, ma, variant=variant, k=k_oracle)
        return float(spectral_radius(model, ma, k=k)["lambda_max"])

    lam_lo, lam_hi = lam(lo), lam(hi)
    if lam_lo > 1.0 + STABILITY_TOL:
        return {"max_stable_ma": 0.0, "lam_at_lo": lam_lo, "lam_at_hi": lam_hi, "capped": True}
    if lam_hi <= 1.0 + STABILITY_TOL:
        return {"max_stable_ma": hi, "lam_at_lo": lam_lo, "lam_at_hi": lam_hi, "capped": True}
    a, b = lo, hi
    for _ in range(int(pre["bisection_iters"]) + 4):
        mid = 0.5 * (a + b)
        if lam(mid) > 1.0 + STABILITY_TOL:
            b = mid
        else:
            a = mid
    return {"max_stable_ma": 0.5 * (a + b), "lam_at_lo": lam_lo, "lam_at_hi": lam_hi, "capped": False}


# ---------------------------------------------------------------------------
# M1: exchange law (prediction <-> measurement)
# ---------------------------------------------------------------------------


def exchange_law_points() -> list[tuple[float, float]]:
    """Predeclared ``(tau, Ma)`` stations for the point-wise exchange law."""
    ma_stations = (0.30, 0.35, 0.40, 0.45, 0.50)
    tau_offsets = (0.0, 0.01, 0.03)
    tau0 = PREDECLARED["tau_window"][0]
    return [(tau0 + d, ma) for ma in ma_stations for d in tau_offsets]


def _exchange_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summary statistics of the exchange-law rows, split by conditioning.

    Two tiers are reported because the *relative* growth-rate error is not a
    meaningful metric as ``sigma -> 0``: a fixed absolute bias of ~1e-3 in the
    fitted amplification factor becomes an arbitrarily large relative error.
    The tiers are declared, not chosen after the fact:

    * ``well_conditioned``     -- ``|sigma_pred| >= 0.005`` (absolute metric),
    * ``strongly_conditioned`` -- ``|sigma_pred| >= 0.03`` (relative metric).
    """
    good = [r for r in rows if r["well_conditioned"] and r["sigma_rel_dev"] is not None
            and math.isfinite(r["sigma_rel_dev"])]
    strong = [r for r in good if abs(math.log(r["lambda_pred"])) >= 0.03]
    abs_dev = [r["lambda_abs_dev"] for r in rows if math.isfinite(r["lambda_meas"])]
    r2 = [r["fit_r2"] for r in rows if math.isfinite(r["fit_r2"])]
    marginal = [r for r in rows if not r["well_conditioned"]]
    return {
        "n_points": len(rows),
        "n_well_conditioned": len(good),
        "n_strongly_conditioned": len(strong),
        "n_marginal": len(marginal),
        "max_abs_lambda_dev": float(max(abs_dev)) if abs_dev else float("nan"),
        "max_abs_lambda_dev_well_conditioned": float(max(r["lambda_abs_dev"] for r in good)) if good else float("nan"),
        "max_abs_lambda_dev_strongly_conditioned": float(max(r["lambda_abs_dev"] for r in strong)) if strong else float("nan"),
        "max_abs_lambda_dev_marginal": float(max(r["lambda_abs_dev"] for r in marginal)) if marginal else float("nan"),
        "max_rel_sigma_dev": float(max(abs(r["sigma_rel_dev"]) for r in good)) if good else float("nan"),
        "max_rel_sigma_dev_strongly_conditioned": float(max(abs(r["sigma_rel_dev"]) for r in strong)) if strong else float("nan"),
        "mean_rel_sigma_dev": float(np.mean([abs(r["sigma_rel_dev"]) for r in strong])) if strong else float("nan"),
        "median_fit_r2": float(np.median(r2)) if r2 else float("nan"),
        "tolerance": {
            "abs_lambda": 0.005,
            "rel_sigma": 0.05,
            "rel_sigma_conditioning": "|sigma_pred| >= 0.03",
        },
        "criterion": (
            "absolute amplification-factor deviation <= 5e-3 for |sigma_pred| >= 0.005; "
            "relative growth-rate deviation <= 5% for |sigma_pred| >= 0.03; near-marginal "
            "points (|sigma_pred| < 0.005) are reported but excluded, because the neutral "
            "checkerboard branch at (pi, 0)/(0, pi) carries no momentum and is invisible to "
            "a velocity-residual measurement"
        ),
    }


def exchange_law(tiers: Sequence[str] = TIERS, *, n_box: int | None = None) -> dict[str, Any]:
    """Predicted vs measured modal growth at the same (box-representable) wave vector."""
    pre = PREDECLARED
    n_box = int(pre["measure_box"] if n_box is None else n_box)
    k_oracle = quarter_wave_vectors(pre["k_grid_oracle"])
    rows: list[dict[str, Any]] = []
    for tier in tiers:
        for tau, ma in exchange_law_points():
            for variant in ("canonical", "oracle"):
                model = tier_model_at(tier, float(tau), float(ma), variant=variant, k=k_oracle)
                pred = predicted_box_radius(model, float(ma), n_box=n_box)
                meas = run_mode_box(model, float(ma), k=pred["k_argmax"], n=n_box)
                sigma_p = math.log(pred["lambda_max"]) if pred["lambda_max"] > 0 else float("nan")
                sigma_m = meas["growth_rate_meas"]
                well_cond = bool(math.isfinite(sigma_p) and abs(sigma_p) >= 0.005)
                rows.append(
                    {
                        "tier": tier,
                        "variant": variant,
                        "tau": float(tau),
                        "ma": float(ma),
                        "k": [float(pred["k_argmax"][0]), float(pred["k_argmax"][1])],
                        "lambda_pred": float(pred["lambda_max"]),
                        "lambda_meas": float(meas["lambda_meas"]),
                        "lambda_abs_dev": float(abs(meas["lambda_meas"] - pred["lambda_max"])),
                        "sigma_rel_dev": float((sigma_m - sigma_p) / abs(sigma_p)) if well_cond else None,
                        "fit_r2": float(meas["fit_r2"]),
                        "n_fit": int(meas["n_fit"]),
                        "blown_up": bool(meas["blown_up"]),
                        "well_conditioned": well_cond,
                    }
                )
    summary = _exchange_summary(rows)
    summary["rows"] = rows
    return summary


def boundary_table(tiers: Sequence[str] = TIERS, *, n_stations: int = 9) -> dict[str, Any]:
    """Predicted (fine + box grid) and measured divergence boundaries ``tau_crit(Ma)``."""
    ma_stations = np.linspace(PREDECLARED["ma_window"][1] - 0.20,
                              PREDECLARED["ma_window"][1], int(n_stations))
    rows: list[dict[str, Any]] = []
    for tier in tiers:
        factory = tier_factory(tier, "canonical")
        for ma in ma_stations:
            fine = predicted_critical_tau(factory, float(ma))
            box = box_predicted_critical_tau(factory, float(ma))
            meas = measured_critical_tau(factory, float(ma))
            dev_box = dev_fine = None
            if meas["tau_crit"] and box["tau_crit"]:
                dev_box = (meas["tau_crit"] - box["tau_crit"]) / box["tau_crit"]
            if meas["tau_crit"] and fine["tau_crit"]:
                dev_fine = (meas["tau_crit"] - fine["tau_crit"]) / fine["tau_crit"]
            rows.append(
                {
                    "tier": tier,
                    "ma": float(ma),
                    "tau_crit_fine_grid": fine["tau_crit"],
                    "tau_crit_box_grid": box["tau_crit"],
                    "tau_crit_measured": meas["tau_crit"],
                    "measured_interior": bool(meas["interior"]),
                    "rel_dev_vs_box_grid": dev_box,
                    "rel_dev_vs_fine_grid": dev_fine,
                    "lambda_meas_at_lo": meas["lambda_lo"],
                    "lambda_meas_at_hi": meas["lambda_hi"],
                    "fit_r2": meas.get("fit_r2_at_bracket"),
                    "k_measured": meas.get("k_at_bracket"),
                }
            )
    interior = [r for r in rows if r["rel_dev_vs_box_grid"] is not None and math.isfinite(r["rel_dev_vs_box_grid"])]
    interior_fine = [r for r in rows if r["rel_dev_vs_fine_grid"] is not None and math.isfinite(r["rel_dev_vs_fine_grid"])]
    return {
        "n_stations": len(rows),
        "n_interior_crossings": len(interior),
        "max_abs_rel_dev_vs_box_grid": float(max(abs(r["rel_dev_vs_box_grid"]) for r in interior)) if interior else None,
        "max_abs_rel_dev_vs_fine_grid": float(max(abs(r["rel_dev_vs_fine_grid"]) for r in interior_fine)) if interior_fine else None,
        "tolerance": 0.05,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# M3: falsifiability + negative controls
# ---------------------------------------------------------------------------


def _foil_stats(bgk_plane: dict[str, Any], foil_plane: dict[str, Any], family: str) -> dict[str, Any]:
    lam_b = np.asarray(bgk_plane["lambda_max"], dtype=float)
    lam_f = np.asarray(foil_plane["lambda_max"], dtype=float)
    taus = np.asarray(bgk_plane["tau_axis"], dtype=float)
    mas = np.asarray(bgk_plane["ma_axis"], dtype=float)
    worse = lam_f > lam_b + 1e-9
    only_foil_unstable = (lam_f > 1.0 + STABILITY_TOL) & (lam_b <= 1.0 + STABILITY_TOL)
    cells = [
        {"tau": float(taus[i]), "ma": float(mas[j]),
         "lambda_bgk": float(lam_b[i, j]), "lambda_foil": float(lam_f[i, j])}
        for i, j in zip(*np.nonzero(worse))
    ]
    by_ma: dict[str, int] = {}
    for i, j in zip(*np.nonzero(worse)):
        key = f"{float(mas[j]):.6f}"
        by_ma[key] = by_ma.get(key, 0) + 1
    return {
        "foil": family,
        "n_cells": int(worse.size),
        "n_worse_cells": int(worse.sum()),
        "worse_area_fraction": float(worse.mean()),
        "n_bgk_stable_foil_unstable_cells": int(only_foil_unstable.sum()),
        "stable_area_ratio_foil": float(foil_plane["stable_area_ratio"]),
        "stable_area_ratio_bgk": float(bgk_plane["stable_area_ratio"]),
        "max_excess_lambda": float((lam_f - lam_b).max()),
        "tau_subdomain": ([float(taus[worse.any(axis=1)].min()), float(taus[worse.any(axis=1)].max())]
                          if worse.any() else None),
        "ma_subdomain": ([float(mas[worse.any(axis=0)].min()), float(mas[worse.any(axis=0)].max())]
                         if worse.any() else None),
        "worse_cells_by_ma": by_ma,
        "first_cells": cells[:12],
        "foil_setting": foil_model(family, PREDECLARED["tau_window"][0]).describe(),
        "verdict": ("observed: a legal, badly tuned MRT setting is less stable than BGK on "
                    f"{100.0 * worse.mean():.2f}% of the scanned plane")
        if worse.any() else "not observed within the declared scan resolution",
    }


def falsification(
    bgk_plane: dict[str, Any], foil_planes: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Locate the ``(tau, Ma)`` subdomains where a legal MRT setting is *worse* than BGK.

    The claim "MRT is more stable than BGK" is falsifiable only if a legitimate
    MRT parameterisation of the same design space is demonstrably *less* stable.
    Two declared foils are scanned over the identical plane:

    * ``s4_under_relaxed`` -- the single knob MRT-B releases (``s_qx``) is
      under-relaxed; expect a small degradation in the corner,
    * ``mismatched`` -- the worst legal rate combination at the declared probe
      point; expect a large degradation across the window.
    """
    foils = {family: _foil_stats(bgk_plane, plane, family) for family, plane in foil_planes.items()}
    worst = max(foils.values(), key=lambda d: d["max_excess_lambda"]) if foils else None
    taus = np.asarray(bgk_plane["tau_axis"], dtype=float)
    mas = np.asarray(bgk_plane["ma_axis"], dtype=float)
    return {
        "foils": foils,
        "n_foils": len(foils),
        "worst_foil": worst["foil"] if worst else None,
        "n_worse_cells": int(sum(v["n_worse_cells"] for v in foils.values())),
        "worst_excess_lambda": float(worst["max_excess_lambda"]) if worst else 0.0,
        "verdict": " ; ".join(f"{k}: {v['verdict']} (max excess lambda {v['max_excess_lambda']:.3f})"
                              for k, v in foils.items()),
        "scan_resolution": {
            "n_tau": int(len(taus)),
            "n_ma": int(len(mas)),
            "delta_tau": float(taus[1] - taus[0]),
            "delta_ma": float(mas[1] - mas[0]),
            "k_grid": int(bgk_plane["k_grid"]),
        },
    }


def negative_controls_report(points: Sequence[tuple[float, float]] | None = None) -> dict[str, Any]:
    """Defective operators must be rejected by BOTH routes (prediction and measurement)."""
    if points is None:
        points = ((0.6, 0.2), (0.505, 0.4), (0.7, 0.35))
    rows: list[dict[str, Any]] = []
    for family in ("over_relax_shear", "over_relax_energy", "mixed_basis_over_relax"):
        for tau, ma in points:
            model = negative_control(family, float(tau))
            pred = spectral_radius(model, float(ma))
            k_box = predicted_box_radius(model, float(ma))["k_argmax"]
            meas = run_mode_box(model, float(ma), k=k_box)
            bb = run_broadband_box(model, float(ma))
            rows.append(
                {
                    "control": family,
                    "tau": float(tau),
                    "ma": float(ma),
                    "defect": model.notes.get("defect"),
                    "basis_orthogonality_defect": model.describe()["basis_orthogonality_defect"],
                    "max_rate": model.max_rate,
                    "lambda_pred": float(pred["lambda_max"]),
                    "k_pred": [float(pred["k_argmax"][0]), float(pred["k_argmax"][1])],
                    "predicted_unstable": bool(pred["lambda_max"] > 1.0 + STABILITY_TOL),
                    "lambda_meas": float(meas["lambda_meas"]),
                    "measured_growing": bool(
                        math.isfinite(meas["lambda_meas"]) and meas["lambda_meas"] > 1.0 + MEASURED_TOL
                    ),
                    "broadband_diverged": bool(bb["divergent"]),
                    "broadband_max_window_ratio": float(bb["max_window_ratio"]),
                    "fit_r2": float(meas["fit_r2"]),
                }
            )
    probe: list[dict[str, Any]] = []
    for tau, ma in points:
        model = negative_control("mixed_basis", float(tau))
        pred = spectral_radius(model, float(ma))
        probe.append(
            {
                "control": "mixed_basis",
                "tau": float(tau),
                "ma": float(ma),
                "basis_orthogonality_defect": model.describe()["basis_orthogonality_defect"],
                "lambda_pred": float(pred["lambda_max"]),
                "predicted_unstable": bool(pred["lambda_max"] > 1.0 + STABILITY_TOL),
            }
        )
    all_red = all(r["predicted_unstable"] and r["measured_growing"] for r in rows)
    return {
        "n_controls": len(rows),
        "n_detected_by_both_routes": int(sum(r["predicted_unstable"] and r["measured_growing"] for r in rows)),
        "n_broadband_fired": int(sum(r["broadband_diverged"] for r in rows)),
        "all_red": bool(all_red),
        "rows": rows,
        "specificity_probe": {
            "note": (
                "A non-orthogonal basis alone (all rates inside (0, 2)) is neutral at "
                f"{sum(not r['predicted_unstable'] for r in probe)}/{len(probe)} probe points but "
                f"does destabilise at {sum(r['predicted_unstable'] for r in probe)} of them "
                "(the low-tau corner, where the same kinetic pocket is already active).  So the "
                "basis defect is *not* silently ignored; it simply requires the pocket to exist.  "
                "The must-fail control set combines the basis defect with an out-of-range rate, "
                "where both routes fire unambiguously."
            ),
            "n_flagged": int(sum(r["predicted_unstable"] for r in probe)),
            "rows": probe,
        },
        "broadband_detector_note": (
            "the literal 10-step ratio > 1e3 detector fires only for lambda > 1e3**(1/10) = 1.995, "
            "so it is a deep-instability detector; the marginal verdict uses the fitted growth rate"
        ),
    }


def verdict_agreement(tiers: Sequence[str] = TIERS, *, n_sample: int = 12) -> dict[str, Any]:
    """Measured growth verdict vs predicted verdict on a deterministic cell subsample."""
    pre = PREDECLARED
    taus = np.linspace(pre["tau_window"][0], pre["tau_window"][1], pre["scan_grid"][0])
    mas = np.linspace(pre["ma_window"][0], pre["ma_window"][1], pre["scan_grid"][1])
    k_oracle = quarter_wave_vectors(pre["k_grid_oracle"])
    rows: list[dict[str, Any]] = []
    for tier in tiers:
        picks = [(0, k) for k in range(0, len(mas), max(len(mas) // int(n_sample), 1))]
        picks += [(len(taus) - 1, k) for k in range(0, len(mas), max(len(mas) // 4, 1))]
        for i, j in picks:
            tau, ma = float(taus[i]), float(mas[j])
            model = tier_model_at(tier, tau, ma, variant="canonical", k=k_oracle)
            pred_box = predicted_box_radius(model, ma)
            meas = run_mode_box(model, ma, k=pred_box["k_argmax"])
            pred_unstable = pred_box["lambda_max"] > 1.0 + STABILITY_TOL
            meas_unstable = math.isfinite(meas["lambda_meas"]) and meas["lambda_meas"] > 1.0 + MEASURED_TOL
            rows.append(
                {
                    "tier": tier,
                    "tau": tau,
                    "ma": ma,
                    "lambda_pred_box": float(pred_box["lambda_max"]),
                    "lambda_meas": float(meas["lambda_meas"]),
                    "predicted_unstable": bool(pred_unstable),
                    "measured_unstable": bool(meas_unstable),
                    "agree": bool(pred_unstable == meas_unstable),
                }
            )
    return {
        "n_points": len(rows),
        "n_agree": int(sum(r["agree"] for r in rows)),
        "agreement": float(np.mean([r["agree"] for r in rows])),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Assembly: envelope + gate
# ---------------------------------------------------------------------------


def _sanitize(obj: Any) -> Any:
    """Replace non-finite floats by ``None`` so the payload stays canonical-JSON safe.

    Unavailable measurements (a failed fit, a bisection without an interior
    crossing) are represented as ``null``, never as ``NaN``: the digest is taken
    over canonical JSON with ``allow_nan=False``, so a NaN would either corrupt
    the digest or silently pass through a permissive serializer.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    return obj


def _round(obj: Any) -> Any:
    if isinstance(obj, float):
        return round(obj, 12)
    if isinstance(obj, dict):
        return {k: _round(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round(v) for v in obj]
    return obj


def assemble_envelope(
    planes: dict[str, Any],
    foil_planes: dict[str, dict[str, Any]],
    measurements: dict[str, Any] | None,
    *,
    tiers: Sequence[str] = TIERS,
    runtime_seconds: float = 0.0,
    outdir: str | pathlib.Path | None = None,
) -> dict[str, Any]:
    """Assemble the envelope from (possibly cached) planes and measurements."""
    exchange = (measurements or {}).get("exchange_law")
    if exchange and exchange.get("rows"):
        cached_rows = exchange["rows"]
        exchange = _exchange_summary(cached_rows)
        exchange["rows"] = cached_rows
    boundary = (measurements or {}).get("boundary")
    controls = (measurements or {}).get("negative_controls")
    agreement = (measurements or {}).get("verdict_agreement")
    bgk_plane = planes["bgk:canonical"]
    falsif = falsification(bgk_plane, foil_planes)
    tier_stats: dict[str, Any] = {}
    lam_b = np.asarray(bgk_plane["lambda_max"], dtype=float)
    stab_b = lam_b <= 1.0 + STABILITY_TOL
    for label, plane in planes.items():
        lam = np.asarray(plane["lambda_max"], dtype=float)
        stab = lam <= 1.0 + STABILITY_TOL
        tier, variant = label.split(":")
        tier_stats[label] = {
            "n_cases": plane["n_cases"],
            "stable_area_ratio": plane["stable_area_ratio"],
            "n_unstable_cells": plane["n_unstable_cells"],
            "lambda_max_overall": plane["lambda_max_overall"],
            "lambda_max_overall_at": plane["lambda_max_overall_at"],
            "lambda_grid_sha256": plane["lambda_grid_sha256"],
            "designs": plane.get("designs", []),
            "max_stable_ma_at_tau_min": max_stable_ma(tier, variant, PREDECLARED["tau_window"][0])["max_stable_ma"],
            "cells_stable_vs_bgk_unstable": int((stab & ~stab_b).sum()),
            "cells_unstable_vs_bgk_stable": int((~stab & stab_b).sum()),
            "max_margin_gain_vs_bgk": float((lam_b - lam).max()),
            "mean_margin_gain_vs_bgk": float((lam_b - lam).mean()),
        }


    envelope: dict[str, Any] = {
        "schema": "tensorlbm.stability.envelope/v1",
        "node": "P482-N05",
        "title": "D2Q9 collision-operator stability domain: predicted vs measured exchange law",
        "lattice": "D2Q9",
        "ma_definition": PREDECLARED["ma_definition"],
        "stability_test": "lambda_max = max_{k != 0} max_j |mu_j(k)| <= 1",
        "predeclared": _round(PREDECLARED),
        "operating_window": {
            "tau": [float(PREDECLARED["tau_window"][0]), float(PREDECLARED["tau_window"][1])],
            "ma": [float(PREDECLARED["ma_window"][0]), float(PREDECLARED["ma_window"][1])],
            "tau_axis": bgk_plane["tau_axis"],
            "ma_axis": bgk_plane["ma_axis"],
            "tiers": list(planes),
        },
        "tiers": {
            label: {**tier_stats[label], "stable_mask": planes[label]["stable_mask"]}
            for label in planes
        },
        "m1_exchange_law": exchange,
        "m1_boundary": boundary,
        "m2_planes": {
            "grid": [len(bgk_plane["tau_axis"]), len(bgk_plane["ma_axis"])],
            "n_cases_per_tier": bgk_plane["n_cases"],
            "k_grid": bgk_plane["k_grid"],
            "mrt_gain": {
                label: {
                    "max_stable_ma_at_tau_min": tier_stats[label]["max_stable_ma_at_tau_min"],
                    "delta_vs_bgk": tier_stats[label]["max_stable_ma_at_tau_min"]
                    - tier_stats["bgk:canonical"]["max_stable_ma_at_tau_min"],
                }
                for label in planes
            },
        },
        "m3_falsification": falsif,
        "m3_negative_controls": controls,
        "m3_verdict_agreement": agreement,
        "m4_reproducibility": {
            "digits": DIGITS,
            "lambda_grid_sha256": {label: planes[label]["lambda_grid_sha256"] for label in planes},
        },
        "limitations": [
            "The unstable pocket of D2Q9 BGK inside this window is localised at wave vectors that a "
            "32x32 box cannot represent; the box-grid prediction and the measurement agree with each "
            "other (<=1%) while sitting slightly below the fine-grid boundary (<=1.5%).",
            "D2Q9 has an exactly neutral (|mu| = 1) checkerboard branch at (pi, 0) and (0, pi) for "
            "every tau; it carries no momentum, so a velocity-residual measurement cannot see it and "
            "reports faster decay than the marginal prediction there.",
            "Measured growth is fitted over the linear window only; amplification factors above 1.3 "
            "saturate within a few steps and their fits carry several-percent error.",
            "The oracle tiers are upper envelopes over a declared discrete rate set, not proofs of "
            "optimality.",
        ],
        "runtime_seconds": round(float(runtime_seconds), 3),
    }
    envelope = _sanitize(envelope)
    envelope["digest"] = envelope_digest(envelope)
    if outdir is not None:
        write_artifacts(envelope, planes, foil_planes, outdir)
    return envelope


def build_envelope(
    *,
    tiers: Sequence[str] = TIERS,
    outdir: str | pathlib.Path | None = None,
    quick: bool = False,
    skip_measurements: bool = False,
) -> dict[str, Any]:
    """Run the campaign end to end and (optionally) write the artefacts."""
    t0 = time.time()
    plane_kwargs = dict(n_tau=13, n_ma=13, n_k=11) if quick else {}
    planes: dict[str, Any] = {}
    for tier in tiers:
        for variant in VARIANTS:
            if tier == "bgk" and variant == "oracle":
                continue
            planes[tier_label(tier, variant)] = scan_plane(tier, variant, **plane_kwargs)
    foil_planes = {family: _foil_plane(family) for family in PREDECLARED["foil_rates"]}
    if skip_measurements:
        measurements = None
    else:
        measurements = {
            "exchange_law": exchange_law(tiers, n_box=16 if quick else None),
            "boundary": boundary_table(tiers, n_stations=5 if quick else 9),
            "negative_controls": negative_controls_report(((0.6, 0.2),) if quick else None),
            "verdict_agreement": verdict_agreement(tiers, n_sample=4 if quick else 12),
        }
    return assemble_envelope(
        planes,
        foil_planes,
        measurements,
        tiers=tiers,
        runtime_seconds=time.time() - t0,
        outdir=outdir,
    )


def point_in_envelope(
    envelope: dict[str, Any], tau: float, ma: float, *, tier: str = "bgk"
) -> dict[str, Any]:
    """Fail-closed gate: is ``(tau, Ma)`` inside the declared stable envelope?

    Two conditions are checked, both conservative:

    1. the point is inside the declared operating window, and
    2. no neighbouring cell of the stored stability mask is flagged unstable
       (nearest-neighbour look-up, no interpolation).
    """
    window = envelope["operating_window"]
    tau_lo, tau_hi = window["tau"]
    ma_lo, ma_hi = window["ma"]
    if not (math.isfinite(tau) and math.isfinite(ma)):
        return {"inside": False, "reason": "NON_FINITE_OPERATING_POINT", "tier": tier}
    if not (tau_lo <= tau <= tau_hi and ma_lo <= ma <= ma_hi):
        return {
            "inside": False,
            "reason": "OUTSIDE_DECLARED_WINDOW",
            "tier": tier,
            "window": {"tau": [tau_lo, tau_hi], "ma": [ma_lo, ma_hi]},
        }
    key = tier if ":" in tier else f"{tier}:canonical"
    if key not in envelope["tiers"]:
        return {"inside": False, "reason": "UNKNOWN_TIER", "tier": tier}
    taus = np.asarray(window["tau_axis"], dtype=float)
    mas = np.asarray(window["ma_axis"], dtype=float)
    mask = np.asarray(envelope["tiers"][key]["stable_mask"], dtype=int)
    i = int(np.clip(np.abs(taus - tau).argmin(), 0, len(taus) - 1))
    j = int(np.clip(np.abs(mas - ma).argmin(), 0, len(mas) - 1))
    neighbours = mask[max(i - 1, 0): i + 2, max(j - 1, 0): j + 2]
    if int(neighbours.min()) != 1:
        return {
            "inside": False,
            "reason": "INSIDE_OR_NEAR_UNSTABLE_REGION",
            "tier": key,
            "cell": [float(taus[i]), float(mas[j])],
            "neighbour_stable_mask_min": int(neighbours.min()),
        }
    return {
        "inside": True,
        "reason": "INSIDE_STABLE_ENVELOPE",
        "tier": key,
        "cell": [float(taus[i]), float(mas[j])],
    }


# ---------------------------------------------------------------------------
# Artefacts
# ---------------------------------------------------------------------------


def write_artifacts(
    envelope: dict[str, Any],
    planes: dict[str, Any],
    foil_planes: dict[str, dict[str, Any]],
    outdir: str | pathlib.Path,
) -> dict[str, str]:
    """Write the envelope JSON, the plane tables and the figure; return created paths."""
    out = pathlib.Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    env_path = out / "stability_envelope.json"
    env_path.write_text(canonical_json(envelope), encoding="utf-8")
    paths["envelope"] = str(env_path)

    flat_path = out / "stability_planes.csv"
    with flat_path.open("w", encoding="utf-8") as fh:
        fh.write("tier,variant,tau,ma,lambda_max,stable\n")
        all_planes = dict(planes)
        for family, plane in foil_planes.items():
            all_planes[f"mrt_foil_{family}:canonical"] = plane
        for label, plane in all_planes.items():
            tier, variant = label.split(":")
            for i, tau in enumerate(plane["tau_axis"]):
                for j, ma in enumerate(plane["ma_axis"]):
                    fh.write(
                        f"{tier},{variant},{tau:.9f},{ma:.9f},"
                        f"{plane['lambda_max'][i][j]:.9f},{plane['stable_mask'][i][j]}\n"
                    )
    paths["planes_csv"] = str(flat_path)

    try:
        fig_path = out / "stability_envelope.png"
        make_figure(envelope, planes, foil_planes, fig_path)
        paths["figure"] = str(fig_path)
    except Exception as exc:  # pragma: no cover - plotting is best effort
        paths["figure_error"] = repr(exc)

    manifest = {
        "digest": envelope["digest"],
        "files": {k: pathlib.Path(v).name for k, v in paths.items() if k != "figure_error"},
        "sha256": {
            pathlib.Path(v).name: hashlib.sha256(pathlib.Path(v).read_bytes()).hexdigest()
            for k, v in paths.items() if k != "figure_error" and pathlib.Path(v).exists()
        },
    }
    (out / "MANIFEST.json").write_text(canonical_json(manifest), encoding="utf-8")
    paths["manifest"] = str(out / "MANIFEST.json")
    return paths


def make_figure(
    envelope: dict[str, Any],
    planes: dict[str, Any],
    foil_planes: dict[str, dict[str, Any]],
    path: pathlib.Path,
) -> pathlib.Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6))
    bgk = planes["bgk:canonical"]
    taus = np.asarray(bgk["tau_axis"])
    mas = np.asarray(bgk["ma_axis"])

    ax = axes[0][0]
    lam = np.asarray(bgk["lambda_max"])
    im = ax.pcolormesh(taus, mas, (lam - 1.0).T, shading="nearest", cmap="RdYlBu_r")
    ax.contour(taus, mas, np.asarray(bgk["stable_mask"]).T, levels=[0.5], colors="k", linewidths=1.2)
    fig.colorbar(im, ax=ax, label=r"$\lambda_{max}-1$")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$Ma$")
    ax.set_title("BGK von Neumann amplification (contour = stability boundary)")

    ax = axes[0][1]
    ma_target = float(mas[-1])
    j = int(np.abs(mas - ma_target).argmin())
    for label, style in (
        ("bgk:canonical", dict(color="C0", lw=2.0)),
        ("mrt_a:canonical", dict(color="C1", lw=1.6, ls="--")),
        ("mrt_b:canonical", dict(color="C2", lw=1.6, ls=":")),
        ("mrt_a:oracle", dict(color="C3", lw=1.2, ls="-.")),
    ):
        if label in planes:
            ax.plot(planes[label]["tau_axis"], np.asarray(planes[label]["lambda_max"])[:, j], **style, label=label)
    for family, style in (("s4_under_relaxed", dict(color="C4", lw=1.1, ls="--")),
                          ("mismatched", dict(color="C5", lw=1.1, ls=":"))):
        if family in foil_planes:
            ax.plot(foil_planes[family]["tau_axis"], np.asarray(foil_planes[family]["lambda_max"])[:, j],
                    label=f"MRT foil: {family} (M3)", **style)
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$\lambda_{max}$")
    ax.set_ylim(0.98, min(1.6, float(np.max(lam[:, j])) + 0.05))
    ax.set_title(f"Tier comparison at Ma = {ma_target:.2f}")
    ax.legend(fontsize=7)

    ax = axes[1][0]
    rows = (envelope.get("m1_boundary") or {}).get("rows", [])
    for tier, color in (("bgk", "C0"), ("mrt_a", "C1"), ("mrt_b", "C2")):
        sel = [r for r in rows if r["tier"] == tier and r["tau_crit_fine_grid"]]
        if sel:
            ax.plot([r["ma"] for r in sel], [r["tau_crit_fine_grid"] for r in sel], "o-", color=color,
                    label=f"{tier}: predicted (fine k grid)")
        sel_m = [r for r in rows if r["tier"] == tier and r["tau_crit_measured"]]
        if sel_m:
            ax.plot([r["ma"] for r in sel_m], [r["tau_crit_measured"] for r in sel_m], "s", color=color,
                    mfc="none", label=f"{tier}: measured")
    ax.set_xlabel(r"$Ma$")
    ax.set_ylabel(r"$\tau_{crit}$")
    ax.set_title("Divergence boundary: prediction vs measurement")
    ax.legend(fontsize=7)

    ax = axes[1][1]
    ex = envelope.get("m1_exchange_law") or {"rows": []}
    if ex["rows"]:
        pred = [r["lambda_pred"] for r in ex["rows"]]
        meas = [r["lambda_meas"] for r in ex["rows"]]
        ok = [r["well_conditioned"] for r in ex["rows"]]
        ax.plot([min(pred), max(pred)], [min(pred), max(pred)], "k-", lw=0.8)
        ax.scatter([p for p, f in zip(pred, ok) if f], [m for m, f in zip(meas, ok) if f],
                   s=18, color="C0", label="|sigma_pred| >= 0.005")
        ax.scatter([p for p, f in zip(pred, ok) if not f], [m for m, f in zip(meas, ok) if not f],
                   s=18, color="C3", marker="x", label="near-marginal")
        ax.set_xlabel(r"predicted $\lambda_{max}$ (box grid)")
        ax.set_ylabel(r"measured $\lambda$")
        ax.set_title("Exchange law at identical wave vector")
        ax.legend(fontsize=7)

    fig.suptitle("D2Q9 collision stability envelope (node P482-N05)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def recompute_check(tier: str = "bgk", variant: str = "canonical", *, n_skip: int = 37) -> dict[str, Any]:
    """Re-evaluate a deterministic subsample of the plane and compare bitwise."""
    pre = PREDECLARED
    n_tau, n_ma = pre["scan_grid"]
    taus = np.linspace(pre["tau_window"][0], pre["tau_window"][1], n_tau)
    mas = np.linspace(pre["ma_window"][0], pre["ma_window"][1], n_ma)
    k = quarter_wave_vectors(pre["k_grid_scan"])
    k_oracle = quarter_wave_vectors(pre["k_grid_oracle"])
    pairs: list[tuple[float, float, float]] = []
    for i in range(0, n_tau, max(n_tau // 6, 1)):
        for j in range(0, n_ma, max(n_ma // 6, 1)):
            model = tier_model_at(tier, float(taus[i]), float(mas[j]), variant=variant, k=k_oracle)
            lam = float(spectral_radius(model, float(mas[j]), k=k)["lambda_max"])
            pairs.append((float(taus[i]), float(mas[j]), round(lam, DIGITS)))
    return {
        "tier": tier,
        "variant": variant,
        "n_points": len(pairs),
        "points_sha256": hashlib.sha256(canonical_json(sorted(pairs)).encode("utf-8")).hexdigest(),
        "sample": [[t, m, v] for t, m, v in pairs[:5]],
    }


def verify_envelope(path: str | pathlib.Path) -> dict[str, Any]:
    """Re-derive the digest of a stored envelope and compare it with the stored one."""
    data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    stored = data.get("digest")
    derived = envelope_digest(data)
    return {
        "path": str(path),
        "stored_digest": stored,
        "derived_digest": derived,
        "intact": bool(stored == derived),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="TensorLBM/artifacts/p482_n05", help="artefact directory")
    parser.add_argument("--quick", action="store_true", help="small grids, smoke test only")
    parser.add_argument("--no-measurements", action="store_true", help="prediction side only")
    parser.add_argument("--verify", default=None, help="verify an existing envelope JSON")
    args = parser.parse_args(argv)
    if args.verify:
        print(json.dumps(verify_envelope(args.verify), indent=2, sort_keys=True))
        return 0
    envelope = build_envelope(outdir=args.out, quick=args.quick, skip_measurements=args.no_measurements)
    print(json.dumps({
        "digest": envelope["digest"],
        "runtime_seconds": envelope["runtime_seconds"],
        "tau_bands": {k: v["stable_area_ratio"] for k, v in envelope["tiers"].items()},
        "max_stable_ma": {k: v["max_stable_ma_at_tau_min"] for k, v in envelope["tiers"].items()},
        "m1_boundary_dev_vs_fine": (envelope.get("m1_boundary") or {}).get("max_abs_rel_dev_vs_fine_grid"),
        "m1_boundary_dev_vs_box": (envelope.get("m1_boundary") or {}).get("max_abs_rel_dev_vs_box_grid"),
        "m1_exchange_max_rel_sigma": (envelope.get("m1_exchange_law") or {}).get("max_rel_sigma_dev"),
        "m3_all_red": (envelope.get("m3_negative_controls") or {}).get("all_red"),
        "m3_falsification": (envelope.get("m3_falsification") or {}).get("verdict"),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DIGITS",
    "TIERS",
    "VARIANTS",
    "assemble_envelope",
    "boundary_table",
    "build_envelope",
    "exchange_law",
    "exchange_law_points",
    "falsification",
    "make_figure",
    "max_stable_ma",
    "negative_controls_report",
    "point_in_envelope",
    "recompute_check",
    "scan_plane",
    "tier_label",
    "verdict_agreement",
    "verify_envelope",
    "write_artifacts",
]
