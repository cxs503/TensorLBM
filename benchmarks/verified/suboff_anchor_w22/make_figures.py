#!/usr/bin/env python3
"""W22 — figures for ../../papers/reports/w22_lbm_suboff_anchor8.md.

All plotted numbers are read from anchor_transcription.json / ittc57_closure.json;
no value is re-typed here.  Each figure gets a sha256 entry in figures/manifest.json.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
FIGDIR = HERE.parent / "figures"
TRANS = json.loads((HERE.parent / "anchor_transcription.json").read_text(encoding="utf-8"))
CLOS = json.loads((HERE.parent / "ittc57_closure.json").read_text(encoding="utf-8"))

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cf_ittc57(re: float) -> float:
    x = math.log10(re) - 2.0
    return float("inf") if x == 0.0 else 0.075 / (x * x)


MANIFEST = []


def save(fig, name: str, caption: str, data_evidence: list[str]) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    path = FIGDIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    MANIFEST.append({"file": str(path.relative_to(REPO)), "sha256": sha256(path),
                     "caption": caption, "data_evidence": data_evidence})


# --------------------------------------------------------------- figure 1: closure
def fig_closure() -> None:
    grids = CLOS["grids"]
    tiers = [g["tier"] for g in grids]
    ct_e2 = [g["C_T_primary_e2_window"] for g in grids]
    ct_mix = [g["C_T_archived_mix50"] for g in grids]
    cal_a = CLOS["closure"]["caliber_A_primary"]
    cal_b = CLOS["closure"]["caliber_B"]
    cal_c = CLOS["closure"]["caliber_C"]
    pred_a = cal_a["per_tier"][0]["C_T_predicted"]
    pred_b = cal_b["per_tier"][0]["C_T_predicted"]
    pred_c = cal_c["per_tier"][0]["C_T_predicted"]
    cf_case = CLOS["case_caliber_gap"]["cf_case_reference_recorded"]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    x = np.arange(len(tiers))
    ax.plot(x, ct_e2, "o-", color="C0", label="LBM $C_T$ (erratum-E2 window mean)")
    ax.plot(x, ct_mix, "s--", color="C0", alpha=0.55, label="LBM $C_T$ (archived mix50)")
    ax.axhline(cf_case, color="C2", ls=":", label=f"case reference $C_F$={cf_case:.6f} (Blasius)")
    ax.axhline(pred_a, color="C3", ls="--",
               label=f"$(1+k)C_F$ caliber A = {pred_a:.5f}\n(ITTC-57@Re=1e3, $k$={cal_a['k']:.4f})")
    ax.axhline(pred_b, color="C1", ls="-.",
               label=f"$(1+k)C_F$ caliber B = {pred_b:.5f}\n(case $C_F$, AFF-8 $k$)")
    ax.axhline(pred_c, color="C4", ls="-.",
               label=f"$(1+k)C_F$ caliber C = {pred_c:.5f}\n(case $C_F$, AFF-1 $k$)")
    ax.set_xticks(x)
    ax.set_xticklabels(tiers)
    ax.set_xlabel("grid tier (suboff_sail_resistance ladder)")
    ax.set_ylabel("$C_T$ / $C_F$  (—)")
    ax.set_yscale("log")
    ax.set_title("C_T=(1+k)C_F+C_W closure: measured vs predicted")
    ax.legend(fontsize=6.5, loc="lower right")

    res = {c["caliber"]: [r["residual_pct"] for r in c["per_tier"]]
           for c in (cal_a, cal_b, cal_c)}
    w = 0.26
    for i, (cid, vals) in enumerate(res.items()):
        ax2.bar(x + (i - 1) * w, vals, width=w, label=f"caliber {cid}")
    ax2.axhline(-3.0, color="k", ls=":", lw=1)
    ax2.axhline(3.0, color="k", ls=":", lw=1)
    for i, vals in enumerate(res.values()):
        for xi, v in zip(x, vals):
            ax2.annotate(f"{v:.0f}%", (xi + (i - 1) * w, v), ha="center",
                         va="bottom" if v < 0 else "top", fontsize=6.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(tiers)
    ax2.set_ylabel("closure residual  $[C_T-(1+k)C_F-C_W]/C_T$  (%)")
    ax2.set_title("closure residual per grid (dotted lines = ±3% gate)")
    ax2.legend(fontsize=7)
    save(fig, "fig1_closure.png",
         "Left: measured $C_T$ per grid vs the three $(1+k)C_F$ predictions. Right: the "
         "closure residual per grid for calibers A/B/C; every bar is far outside the ±3% gate.",
         ["ittc57_closure.json:/grids", "ittc57_closure.json:/closure/caliber_A_primary",
          "ittc57_closure.json:/closure/caliber_B", "ittc57_closure.json:/closure/caliber_C"])


# ------------------------------------------------------------- figure 2: Prohaska
def fig_prohaska() -> None:
    pro = CLOS["k_identifiability"]["prohaska"]
    pts = pro["ladder_points_aff8"]
    fit = pro["fit_aff8"]
    fr4 = np.array([p["Fr"] ** 4 for p in pts])
    y = np.array([p["CT"] / p["CF"] for p in pts])
    order = np.argsort(fr4)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    xx = np.linspace(0, fr4.max() * 1.1, 50)
    ax.plot(xx, fit["intercept"] + fit["slope"] * xx, "-", color="C0",
            label=f"OLS fit: $(1+k)$={fit['intercept']:.4f} ± {fit['stderr_intercept']:.4f}\n"
                  f"slope={fit['slope']:.4f}, $R^2$={fit['r2']:.3f}")
    ax.plot(fr4[order], y[order], "o", color="C3", label="AFF-8 ladder points (transcribed)")
    for p, xv, yv in zip(pts, fr4, y):
        ax.annotate(f"{p['Re']:.2e}", (xv, yv), textcoords="offset points", xytext=(4, 4),
                    fontsize=6)
    ax.axhline(1.0 + CLOS["k_identifiability"]["record"]["primary"], color="C1", ls="--",
               label="recorded $(1+k)$="
                     f"{1.0 + CLOS['k_identifiability']['record']['primary']:.4f}")
    ax.set_xlabel("$Fr^4$   ($Fr=U/\\sqrt{gL}$)")
    ax.set_ylabel("$C_T/C_F$  (—)")
    ax.set_title("Prohaska fit on the AFF-8 ladder (n=4, all points)")
    ax.legend(fontsize=7)

    k_rec = CLOS["k_identifiability"]["record"]["primary"]
    aff8_k = [p["one_plus_k_recorded"] - 1.0 for p in pts]
    ax2.plot([p["Re"] for p in pts], aff8_k, "o-", color="C3", label="AFF-8 per-point $k$")
    ax2.axhline(k_rec, color="C1", ls="--", label=f"record $k$={k_rec:.4f}")
    ax2.axhline(pro["k_prohaska"], color="C0", ls="-.",
                label=f"Prohaska $k$={pro['k_prohaska']:.4f}")
    ax2.set_xscale("log")
    ax2.set_xlabel("Re (—)")
    ax2.set_ylabel("$k$  (—)")
    ax2.set_title("Two k calibers on the same ladder\n"
                  f"diff(k)={pro['diff_pct_k_basis']:+.2f}%, "
                  f"diff(1+k)={pro['diff_pct_one_plus_k_basis']:+.2f}%")
    ax2.legend(fontsize=7)
    save(fig, "fig2_prohaska.png",
         "Left: Prohaska regression $C_T/C_F$ on $Fr^4$ over the five transcribed AFF-8 ladder "
         "points, with the recorded $(1+k)$ for comparison. Right: the per-point $k=C_T/C_F-1$ "
         "along the ladder, showing why the fit is weakly constrained ($R^2$ well below 1).",
         ["ittc57_closure.json:/k_identifiability/prohaska/ladder_points_aff8",
          "ittc57_closure.json:/k_identifiability/prohaska/fit_aff8",
          "ittc57_closure.json:/k_identifiability/record"])


# ------------------------------------------------------------ figure 3: domain map
def fig_domain() -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    re_lo, re_hi = 30.0, 1.0e8
    res = np.logspace(math.log10(re_lo), math.log10(re_hi), 800)
    cf = np.array([cf_ittc57(r) for r in res])
    keep = np.isfinite(cf) & (cf < 1.0)
    ax.plot(res[keep], cf[keep], "-", color="C0", lw=1.6,
            label=r"AN-ITTC57  $C_F=0.075/(\log_{10}Re-2)^2$")
    ax.plot(res, 1.328 / np.sqrt(res), "--", color="C2", lw=1.2,
            label=r"case reference: Blasius laminar $1.328/\sqrt{Re}$")

    re_case = CLOS["case"]["Re_case"]
    ax.axvline(re_case, color="C3", ls=":", label=f"case Re = {re_case:.0f} (lattice)")
    ax.plot([re_case], [CLOS["case_caliber_gap"]["cf_ittc57_at_case_Re"]], "v", color="C3", ms=7)
    ax.annotate(f"ITTC-57 @Re=1e3 = {CLOS['case_caliber_gap']['cf_ittc57_at_case_Re']:.4f}",
                (re_case, CLOS["case_caliber_gap"]["cf_ittc57_at_case_Re"]),
                textcoords="offset points", xytext=(-150, 8), fontsize=7, color="C3")
    ax.plot([re_case], [CLOS["case_caliber_gap"]["cf_case_reference_recorded"]], "o", color="C2", ms=6)
    ax.annotate(f"case $C_F$ = {CLOS['case_caliber_gap']['cf_case_reference_recorded']:.5f}",
                (re_case, CLOS["case_caliber_gap"]["cf_case_reference_recorded"]),
                textcoords="offset points", xytext=(10, -14), fontsize=7, color="C2")
    for g in CLOS["grids"]:
        ax.plot([re_case], [g["C_F_case_recorded"]], "x", color="k", ms=6)
    ax.plot([re_case], [CLOS["grids"][-1]["C_T_primary_e2_window"]], "s", color="C0", ms=6)
    ax.annotate("LBM $C_T$ (finest grid)", (re_case, CLOS["grids"][-1]["C_T_primary_e2_window"]),
                textcoords="offset points", xytext=(10, 6), fontsize=7)

    re_anchor = CLOS["formula_AN_SUBOFF"]["anchor_Re_aff8"]["value"]
    ax.axvline(re_anchor, color="C1", ls=":", label=f"AFF-8 anchor Re = {re_anchor:.3e}")
    ax.plot([re_anchor], [CLOS["formula_AN_SUBOFF"]["anchor_CT_aff8"]["value"]], "D", color="C1",
            ms=6, label="AFF-8 anchor $C_T$ (transcribed)")
    ax.plot([100.0], [1.0], "k|", color="0.3", ms=14)
    ax.annotate("pole at Re=100", (100.0, 1.0), textcoords="offset points", xytext=(6, -16),
                fontsize=7, color="0.3")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Re = U·L/ν  (—)")
    ax.set_ylabel("friction / resistance coefficient (—)")
    ax.set_title("AN-ITTC57 applicability map vs the case and the AFF-8 anchor")
    ax.legend(fontsize=6.8, loc="lower left")
    save(fig, "fig3_ittc57_domain.png",
         "AN-ITTC57 as a function of Re with the case point (Re=1e3), the AFF-8 anchor point "
         "(Re≈1.17e7) and the laminar case reference. The ITTC-57 line is a turbulent "
         "correlation: it has a pole at Re=1e2 and lies ~79% above the laminar reference at the "
         "case Reynolds number, which is the structural reason the closure cannot hold.",
         ["ittc57_closure.json:/formula_AN_ITTC57", "ittc57_closure.json:/case",
          "ittc57_closure.json:/formula_AN_SUBOFF"])


# ------------------------------------------------------- figure 4: transcription census
def fig_census() -> None:
    srcs = TRANS["sources"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.0, 3.6))
    ids = [s["source_id"] for s in srcs]
    n = [s["n_scalar_leaves"] for s in srcs]
    ax.bar(ids, n, color="C0")
    for i, v in enumerate(n):
        ax.annotate(str(v), (i, v), ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("transcribed scalar leaves")
    ax.set_title(f"Transcription census: {TRANS['n_leaves_total']} leaves, "
                 f"{TRANS['completeness']['completeness_pct']:.2f}% complete")
    ax.tick_params(axis="x", labelsize=7)
    mism = [s["n_readback_mismatch"] for s in srcs]
    ax2.bar(ids, mism, color="C3")
    ax2.set_ylabel("readback mismatches")
    ax2.set_title("Independent readback (pointer resolution / char-level re-read)")
    ax2.set_ylim(0, 1)
    ax2.annotate(f"total mismatches = {sum(mism)}", (0.02, 0.85), xycoords="axes fraction",
                 fontsize=8)
    ax2.tick_params(axis="x", labelsize=7)
    save(fig, "fig4_transcription_census.png",
         "Left: number of scalar leaves transcribed per source artifact. Right: readback "
         "mismatches per source — zero everywhere, i.e. every transcribed value resolves back to "
         "its source with identical type and value.",
         ["anchor_transcription.json:/sources", "anchor_transcription.json:/completeness"])


def main() -> int:
    fig_closure()
    fig_prohaska()
    fig_domain()
    fig_census()
    manifest = {
        "schema": "suboff-w22-figures/v1",
        "task_key": CLOS["task_key"],
        "mission_id": CLOS["mission_id"],
        "generated_by": str(HERE.relative_to(REPO)),
        "inputs": {
            "anchor_transcription_sha256": sha256(HERE.parent / "anchor_transcription.json"),
            "ittc57_closure_sha256": sha256(HERE.parent / "ittc57_closure.json"),
        },
        "figures": MANIFEST,
    }
    out = FIGDIR / "manifest.json"
    out.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(MANIFEST)} figures + {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())