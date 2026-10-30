#!/usr/bin/env python
"""Analysis + figures for the CPU bstep_3d reattachment diagnosis (mission m469).

Consumes ``artifacts/cpu_repro_*.json`` + ``artifacts/field_*.npz`` produced by
``diag_cpu_repro.py``, writes ``artifacts/summary.json`` and the figure set in
``artifacts/figures/``.  All numbers in the report are taken from these files -
nothing is transcribed by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
FIG = ART / "figures"

XR_REF = 3.0  # Armaly et al. (1983), Re=100 (U_max x h / nu)
ER_REF = 1.94


def _load(name: str) -> dict | None:
    p = ART / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _steady(hist: list[dict], key: str, tail: int = 3) -> float | None:
    vals = [r[key] for r in hist if r.get(key) is not None]
    if not vals:
        return None
    return float(np.mean(vals[-tail:]))


def main() -> None:
    figures = FIG
    figures.mkdir(parents=True, exist_ok=True)

    ladder = [_load("cpu_repro_H16.json"), _load("cpu_repro_H20.json")]
    ladder = sorted((r for r in ladder if r is not None), key=lambda r: -r["grid"]["step_h"])
    tau_cross = _load("cpu_repro_H16_tau0524.json")

    if not ladder:
        raise SystemExit(f"no cpu_repro results in {ART}")

    summary: dict = {
        "mission": "m469",
        "case": "bstep_3d",
        "reference": {
            "source": "Armaly, Durst, Pereira & Schoenung, JFM 127:473-496 (1983)",
            "xr_h_ref_re100": XR_REF,
            "re_definition": "Re = U_max * h / nu",
            "er_ref": ER_REF,
        },
        "ladder": [],
        "tau_cross_check": None,
    }

    for r in ladder:
        hist = r["history"]
        m = r["measurement_new"]
        xr_steady_new = _steady(hist, "xr_h_new")
        xr_steady_leg = _steady(hist, "xr_h_legacy_meanfield")
        xr_planes_leg = _steady(hist, "xr_h_legacy_planes")
        entry = {
            "tag": r["tag"],
            "grid": r["grid"],
            "physics": r["physics"],
            "run": r["run"],
            "er_dev_pct": (r["grid"]["expansion_ratio"] - ER_REF) / ER_REF * 100.0,
            "xr_h_new_steady": xr_steady_new,
            "xr_h_new_final": m["xr_h"],
            "xr_status_final": m["xr_status"],
            "xr_planes_valid": m["xr_planes_valid"],
            "xr_planes_invalid": m["xr_planes_invalid"],
            "xr_planes_std": m["xr_planes_std"],
            "min_rel_backflow": m["xr_min_rel"],
            "backflow_span_h": [
                m["xr_backflow_start_h"],
                m["xr_backflow_end_h"],
            ],
            "xr_h_legacy_steady_meanfield": xr_steady_leg,
            "xr_h_legacy_steady_planes": xr_planes_leg,
            "err_pct_new": (
                None if xr_steady_new is None else (xr_steady_new - XR_REF) / XR_REF * 100.0
            ),
            "err_pct_legacy": (
                None if xr_steady_leg is None else (xr_steady_leg - XR_REF) / XR_REF * 100.0
            ),
            "estimator_delta_h": (
                None
                if (xr_steady_new is None or xr_steady_leg is None)
                else xr_steady_new - xr_steady_leg
            ),
        }
        summary["ladder"].append(entry)

    if tau_cross is not None:
        hist = tau_cross["history"]
        m = tau_cross["measurement_new"]
        xr_steady = _steady(hist, "xr_h_new")
        summary["tau_cross_check"] = {
            "tag": tau_cross["tag"],
            "grid": tau_cross["grid"],
            "physics": tau_cross["physics"],
            "run": tau_cross["run"],
            "xr_h_new_steady": xr_steady,
            "xr_status_final": m["xr_status"],
            "err_pct_new": None if xr_steady is None else (xr_steady - XR_REF) / XR_REF * 100.0,
            "note": (
                "same Re and grid as the H16 ladder rung but tau=0.524 / U_max=0.05 instead of "
                "tau=0.56 / U_max=0.125: isolates the (tau-0.5) slip + compressibility "
                "confound that the tau-matched protocol removes"
            ),
        }

    # estimator robustness: same field, two wall-normal nodes
    import sys as _sys

    if str(HERE) not in _sys.path:
        _sys.path.insert(0, str(HERE))
    import reattach as _rt  # noqa: E402

    for entry in summary["ladder"] + ([summary["tau_cross_check"]] if summary["tau_cross_check"] else []):
        f = ART / f"field_{entry['tag']}.npz"
        if not f.exists():
            continue
        ux = np.load(f)["ux"]
        vals = {}
        for y in (1, 2):
            mrow = _rt.measure_planes(
                ux, entry["grid"]["x_step"], entry["grid"]["step_h"], y=y,
                rel_thresh=1e-3, u_scale=entry["physics"]["u_max"],
            )
            vals[f"y{y}"] = mrow["xr_h"]
        entry["node_sensitivity"] = {
            "xr_h_at_y1": vals.get("y1"),
            "xr_h_at_y2": vals.get("y2"),
            "delta": (
                None
                if vals.get("y1") is None or vals.get("y2") is None
                else vals["y2"] - vals["y1"]
            ),
        }

    # grid refinement metric: ladder is sorted coarse (largest h) first
    if len(summary["ladder"]) >= 2:
        a, b = summary["ladder"][0], summary["ladder"][1]
        if a["xr_h_new_steady"] and b["xr_h_new_steady"]:
            summary["refinement"] = {
                "h_coarse": a["grid"]["step_h"],
                "h_fine": b["grid"]["step_h"],
                "d_xr_h": b["xr_h_new_steady"] - a["xr_h_new_steady"],
                "monotone_toward_reference": bool(
                    abs(b["xr_h_new_steady"] - XR_REF) < abs(a["xr_h_new_steady"] - XR_REF)
                ),
            }
    # O(h) extrapolation onto the legacy H=40 tau-matched rung (same tau=0.56, U_max=0.05)
    audit_path = ART / "record_audit.json"
    pts = [
        (e["grid"]["step_h"], e["xr_h_new_steady"])
        for e in summary["ladder"]
        if e["xr_h_new_steady"] is not None
    ]
    if audit_path.exists():
        rec = json.loads(audit_path.read_text(encoding="utf-8"))["records"]["result_H40.json"]
        if rec["tau"] == 0.56 and rec["xr_h"] is not None:
            pts.append((rec["step_h"], rec["xr_h"]))
    if len(pts) >= 2:
        hs = np.array([p[0] for p in pts], dtype=float)
        xs = np.array([p[1] for p in pts], dtype=float)
        b1, b0 = np.polyfit(hs, xs, 1)
        summary["extrapolation"] = {
            "model": "X_r/h = a + b*h  (linear in grid spacing, pts = CPU ladder + legacy H=40)",
            "points": [[float(h), float(x)] for h, x in pts],
            "slope_per_cell": float(b1),
            "h0_limit": float(b0),
            "h0_err_pct": float((b0 - XR_REF) / XR_REF * 100.0),
            "rms_residual": float(np.sqrt(np.mean((np.polyval([b1, b0], hs) - xs) ** 2))),
        }

    summary["gate"] = {
        "err_tol_pct": 3.0,
        "all_within_3pct": all(
            e["err_pct_new"] is not None and abs(e["err_pct_new"]) <= 3.0
            for e in summary["ladder"]
        ),
        "note": "campaign gate is <=3% on the direct observable plus a >=2-grid monotone ladder",
    }

    (ART / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ----------------------------- figures -----------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True})
    metas: list[dict] = []

    # Fig 1: near-wall tangential velocity rows, both grids, estimator markers
    fig, axes = plt.subplots(1, len(ladder), figsize=(5.2 * len(ladder), 4.0), squeeze=False)
    for ax, r in zip(axes[0], ladder):
        h = r["grid"]["step_h"]
        row = np.asarray(r["final_wall_row"], dtype=float) / r["physics"]["u_max"]
        x = (np.arange(row.size) + 0.5) / h
        ax.plot(x, row, "o-", ms=2.5, lw=1.0, label="u(y=1)/U_max (spanwise mean)")
        ax.axhline(0.0, color="k", lw=0.8)
        m = r["measurement_new"]
        ax.axvline(m["xr_h"], color="tab:red", ls="--", label=f"X_r/h = {m['xr_h']:.3f} (robust)")
        leg = r["history"][-1]["xr_h_legacy_meanfield"]
        if leg is not None:
            ax.axvline(leg, color="tab:orange", ls=":", label=f"X_r/h = {leg:.3f} (legacy)")
        ax.axvspan(0.0, max(m["xr_backflow_end_h"], 1e-3), color="tab:blue", alpha=0.08,
                   label="separated wall")
        ax.set_xlabel("x / h  (from the step face)")
        ax.set_ylabel("u_t / U_max at wall-adjacent node")
        ax.set_title(f"H={h}, ER={r['grid']['expansion_ratio']:.4f}, Re=100")
        ax.legend(loc="lower right", fontsize=7)
    fig.suptitle("bstep_3d (CPU repro): wall-tangential velocity and reattachment closure")
    fig.tight_layout()
    p = figures / "fig1_wall_profiles.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    metas.append(
        {
            "name": p.name,
            "caption": "Near-wall tangential velocity along the stepped wall for both "
            "tau-matched ladder rungs; dashed/red = robust closure, dotted/orange = pre-fix "
            "estimator, shaded = detected backflow region.",
        }
    )

    # Fig 2: convergence history, both estimators
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    colors = ["tab:blue", "tab:green", "tab:purple"]
    for c, r in zip(colors, ladder):
        steps = [d["step"] for d in r["history"]]
        new = [np.nan if d["xr_h_new"] is None else d["xr_h_new"] for d in r["history"]]
        leg = [
            np.nan if d["xr_h_legacy_meanfield"] is None else d["xr_h_legacy_meanfield"]
            for d in r["history"]
        ]
        h = r["grid"]["step_h"]
        ax.plot(steps, new, "o-", color=c, lw=1.4, label=f"robust, H={h}")
        ax.plot(steps, leg, "x--", color=c, lw=0.9, alpha=0.7, label=f"pre-fix, H={h}")
    ax.axhline(XR_REF, color="k", lw=1.0, label="Armaly 1983, Re=100: 3.0")
    ax.axhspan(XR_REF * 0.97, XR_REF * 1.03, color="k", alpha=0.08, label="+/-3% band")
    ax.set_xlabel("lattice step")
    ax.set_ylabel("X_r / h")
    ax.set_title("Reattachment-length convergence (CPU repro, tau-matched, Re=100)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    p = figures / "fig2_convergence.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    metas.append(
        {
            "name": p.name,
            "caption": "X_r/h versus lattice step for the robust (solid) and pre-fix (dashed) "
            "estimators on identical fields; black band = Armaly Re=100 reference +/-3%.",
        }
    )

    # Fig 3: grid trend
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    hs = [e["grid"]["step_h"] for e in summary["ladder"]]
    xrs = [e["xr_h_new_steady"] for e in summary["ladder"]]
    ax.plot(hs, xrs, "o-", color="tab:red", label="robust estimator (tau-matched)")
    if tau_cross is not None:
        ax.plot([tau_cross["grid"]["step_h"]], [summary["tau_cross_check"]["xr_h_new_steady"]],
                "s", color="tab:blue", ms=8,
                label=f"tau=0.524 cross-check (H={tau_cross['grid']['step_h']})")
    ax.axhline(XR_REF, color="k", lw=1.0, label="Armaly 1983 (3.0)")
    ax.axhspan(XR_REF * 0.97, XR_REF * 1.03, color="k", alpha=0.10, label="+/-3% band")
    for hh, xx in zip(hs, xrs):
        if xx is not None:
            ax.annotate(f"{xx:.3f}\n({(xx - XR_REF) / XR_REF * 100:+.1f}%)",
                        (hh, xx), textcoords="offset points", xytext=(8, 6), fontsize=7)
    ax.set_xlabel("step height h [lattice cells]")
    ax.set_ylabel("X_r / h")
    ax.set_title("Grid ladder at Re=100 (tau = 0.56 fixed, U_max = 2/h)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = figures / "fig3_grid_trend.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    metas.append(
        {
            "name": p.name,
            "caption": "X_r/h versus grid resolution for the tau-matched Re=100 ladder, with the "
            "Armaly reference and the +/-3% acceptance band; square = tau/Ma cross-check.",
        }
    )

    # Fig 4: field contour + reattachment line
    fields = sorted(ART.glob("field_re100_H*_nz16_umax*.npz"))
    if fields:
        best = [f for f in fields if "umax0.1" in f.name] or fields
        nz_i = len(best) // 2
        data = np.load(best[nz_i])
        ux, solid = data["ux"], data["solid"]
        h = int(best[nz_i].name.split("_H")[1].split("_")[0])
        zmid = ux.shape[0] // 2
        u = ux[zmid].copy()
        u[solid[zmid]] = np.nan
        x_step = 2 * h
        xx = (np.arange(u.shape[1]) - (x_step - 0.5)) / h
        yy = np.arange(u.shape[0]) / h
        fig, ax = plt.subplots(figsize=(7.0, 3.2))
        vmax = np.nanmax(np.abs(u))
        im = ax.pcolormesh(xx, yy, u, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
        fig.colorbar(im, ax=ax, label="u_x / U_max")
        rec = [e for e in summary["ladder"] if e["grid"]["step_h"] == h]
        if rec and rec[0]["xr_h_new_steady"] is not None:
            ax.axvline(rec[0]["xr_h_new_steady"], color="k", ls="--", lw=1.2,
                       label=f"X_r/h = {rec[0]['xr_h_new_steady']:.3f}")
        ax.plot(0.0, 0.0, "ks", ms=3)
        ax.set_xlim(-1.5, max(xx) * 0.85)
        ax.set_ylim(0, yy[-1] * 0.9)
        ax.set_xlabel("x / h")
        ax.set_ylabel("y / h")
        ax.set_title(f"ux field, mid-span plane (H={h}, Re=100): recirculation and closure")
        ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout()
        p = figures / "fig4_field_contour.png"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        metas.append(
            {
                "name": p.name,
                "caption": "Spanwise mid-plane streamwise velocity field of the re-run case; the "
                "dashed line marks the measured reattachment closure X_r/h.",
            }
        )

    (FIG / "figures.json").write_text(json.dumps(metas, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False)[:4000])
    print("figures:", [m["name"] for m in metas])


if __name__ == "__main__":
    main()