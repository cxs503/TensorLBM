"""KPIs, tables and figures of XVAL-LBM-SCMP-TWO-COMPONENT-PLANAR-INTERFACE.

Reads the arm JSONs written by ``run_all.py`` and produces
``kpi.json`` + ``KPI_TABLE.md`` + the figures under ``figures/``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import oss_arm  # noqa: E402
import tl_arm  # noqa: E402

OUT = ROOT / "artifacts" / "xval_scmp_two_component_planar_interface"
FIG = OUT / "figures"
CS2 = 1.0 / 3.0


def profile(rho: np.ndarray) -> np.ndarray:
    """Transverse profile <rho>(x) of a (ny, nx) field."""
    return rho.mean(axis=1)


def bulk(rho: np.ndarray) -> tuple[float, float]:
    """(A-rich region mean, A-poor region mean) using the pre-registered windows."""
    ny = rho.shape[0]
    q = ny // 8
    poor = rho[q: ny // 2 - q, :].mean()      # rows just below the interface: component-0 poor half
    rich = rho[ny // 2 + q: ny - q, :].mean()  # rows just above the interface: component-0 rich half
    return float(rich), float(poor)


def interface_width(prof: np.ndarray) -> float:
    """Mean 10-90 % interface width over the two (periodic) interfaces."""
    lo, hi = prof.min(), prof.max()
    t = 0.1 + 0.8 * (prof - lo) / (hi - lo)
    inside = (t > 0.1) & (t < 0.9)
    return float(inside.sum() / 2.0)  # two interfaces per period


def pressure_oss(f: dict, g_ab: float, g_self=(0.0, 0.0)) -> np.ndarray:
    """OSS computePressure(): p = sum_s rho_s/3 + sum_{s,ss} g_ss psi_s psi_ss/6."""
    p1 = 1.0 - np.exp(-f["density_comp0"])
    p2 = 1.0 - np.exp(-f["density_comp1"])
    tot = (f["density_comp0"] + f["density_comp1"]) / 3.0
    cross = (g_ab * p1 * p2 + g_ab * p2 * p1) / 6.0
    self_ = (g_self[0] * p1 * p1 + g_self[1] * p2 * p2) / 6.0
    return tot + cross + self_


def pressure_balance(f: dict, g_ab: float) -> np.ndarray:
    """Momentum-balance form p = cs2 sum_s rho_s + G_12 cs2 psi_1 psi_2, G_12 = -g_AB."""
    p1 = 1.0 - np.exp(-f["density_comp0"])
    p2 = 1.0 - np.exp(-f["density_comp1"])
    return CS2 * (f["density_comp0"] + f["density_comp1"]) - g_ab * CS2 * p1 * p2


def flatness(p: np.ndarray) -> dict:
    prof = p.mean(axis=1)
    return {
        "mean": float(prof.mean()),
        "ptp": float(prof.max() - prof.min()),
        "ptp_rel": float((prof.max() - prof.min()) / abs(prof.mean())),
    }


def compare(ref: dict, got: dict) -> dict:
    """KPI set of one TL arm against its OSS reference arm."""
    fr, fg = ref["fields"], got["fields"]
    out: dict = {}
    for comp in ("density_comp0", "density_comp1"):
        br, bg = bulk(fr[comp]), bulk(fg[comp])
        out[f"{comp}_rich"] = {"ref": br[0], "got": bg[0],
                               "err_pct": 100.0 * abs(bg[0] - br[0]) / abs(br[0])}
        out[f"{comp}_poor"] = {"ref": br[1], "got": bg[1],
                               "err_pct": 100.0 * abs(bg[1] - br[1]) / abs(br[1])}
    pr, pg = profile(fr["density_comp0"]), profile(fg["density_comp0"])
    span = float(pr.max() - pr.min())
    out["eps_profile_pct"] = 100.0 * float(np.abs(pg - pr).max()) / span
    out["profile_span_ref"] = span
    wr, wg = interface_width(pr), interface_width(pg)
    out["width"] = {"ref": wr, "got": wg, "err_pct": 100.0 * abs(wg - wr) / wr}
    ur = float(np.hypot(fr["velx"], fr["vely"]).max())
    ug = float(np.hypot(fg["velx"], fg["vely"]).max())
    out["u_max"] = {"ref": ur, "got": ug}
    g_ab = ref["params"]["gAB"]
    g_self = (-float(ref["params"].get("gA", 0.0)), -float(ref["params"].get("gB", 0.0)))
    out["pressure"] = {
        "oss_formula_on_ref": flatness(pressure_oss(fr, g_ab, g_self)),
        "oss_formula_on_tl": flatness(pressure_oss(fg, g_ab, g_self)),
        "balance_form_on_ref": flatness(pressure_balance(fr, g_ab)),
        "balance_form_on_tl": flatness(pressure_balance(fg, g_ab)),
    }
    out["eps_bulk_max_pct"] = max(
        out[f"{c}_{r}"]["err_pct"] for c in ("density_comp0", "density_comp1") for r in ("rich", "poor")
    )
    return out


def main() -> int:
    pairs = [
        ("tl_primary", "oss_primary"),
        ("tl_g4", "oss_g4"),
        ("tl_n32", "oss_n32"),
        ("tl_t8000", "oss_t8000"),
        ("tl_native", "oss_primary"),
        ("tl_linear", "oss_primary"),
    ]
    kpi: dict = {"pairs": {}, "thresholds": json.loads((HERE / "predeclared.json").read_text()
                                                      if False else "{}")}
    kpi["thresholds"] = {"owner_acceptance_pct": 15.0, "target_pct": 3.0}
    for tl_label, oss_label in pairs:
        try:
            got = tl_arm.load(OUT, tl_label)
            ref = oss_arm.load(OUT, oss_label)
        except FileNotFoundError:
            continue
        if not got.get("finite", True):
            kpi["pairs"][f"{tl_label}_vs_{oss_label}"] = {"fail_stage": "divergence",
                                                          "failure_reason": "TensorLBM arm non-finite"}
            continue
        if not ref.get("finite", True):
            kpi["pairs"][f"{tl_label}_vs_{oss_label}"] = {"fail_stage": "divergence",
                                                         "failure_reason": "OSS arm non-finite"}
            continue
        kpi["pairs"][f"{tl_label}_vs_{oss_label}"] = compare(ref, got)

    primary = kpi["pairs"].get("tl_primary_vs_oss_primary", {})
    if primary:
        kpi["primary_pass_15pct"] = bool(primary["eps_bulk_max_pct"] < 15.0
                                        and primary["eps_profile_pct"] < 15.0)
        kpi["primary_within_3pct"] = bool(primary["eps_bulk_max_pct"] < 3.0
                                          and primary["eps_profile_pct"] < 3.0)
    kpi["divergence_arms"] = {
        "oss": (oss_arm.load(OUT, "oss_divergence")["finite"]
                if (OUT / "oss_oss_divergence.json").exists() else None),
        "tl": (tl_arm.load(OUT, "tl_divergence").get("finite")
               if (OUT / "tl_tl_divergence.json").exists() else None),
    }
    with open(OUT / "kpi.json", "w") as fh:
        json.dump(kpi, fh, indent=1)

    # ------------------------------------------------------------- markdown
    lines = ["# KPI table — TL vs OSS (SCMP two-component planar interface)", "",
             "| pair | eps_bulk max | eps_profile | width | u_max ref | u_max TL | p_flat(OSS form, ref) | p_flat(balance form, ref) |",
             "|---|---|---|---|---|---|---|---|"]
    for name, v in kpi["pairs"].items():
        if "fail_stage" in v:
            lines.append(f"| {name} | FAIL: {v['failure_reason']} | | | | | | |")
            continue
        lines.append(
            f"| {name} | {v['eps_bulk_max_pct']:.2f} % | {v['eps_profile_pct']:.2f} % | "
            f"{v['width']['err_pct']:.2f} % | {v['u_max']['ref']:.2e} | {v['u_max']['got']:.2e} | "
            f"{v['pressure']['oss_formula_on_ref']['ptp_rel']:.3e} | "
            f"{v['pressure']['balance_form_on_ref']['ptp_rel']:.3e} |"
        )
    lines += ["", f"primary PASS (<15 %): {kpi.get('primary_pass_15pct')}",
              f"primary within 3 %: {kpi.get('primary_within_3pct')}", ""]
    (OUT / "KPI_TABLE.md").write_text("\n".join(lines) + "\n")

    # -------------------------------------------------------------- figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print("matplotlib unavailable:", exc)
        print("\n".join(lines))
        return 0

    FIG.mkdir(parents=True, exist_ok=True)
    ref = oss_arm.load(OUT, "oss_primary")

    def _first_available(*labels):
        for lab in labels:
            try:
                return tl_arm.load(OUT, lab)
            except FileNotFoundError:
                continue
        raise FileNotFoundError(labels)

    got = _first_available("tl_primary", "tl_n32")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    y = np.arange(ref["fields"]["density_comp0"].shape[0])
    for comp, lbl in (("density_comp0", "component A"), ("density_comp1", "component B")):
        ax[0].plot(y, profile(ref["fields"][comp]), "o", ms=3, mfc="none",
                   label=f"OSS {lbl}")
        ax[0].plot(y, profile(got["fields"][comp]), "-", lw=1.4, label=f"TensorLBM {lbl}")
    ax[0].set_xlabel("y (lattice node)")
    ax[0].set_ylabel(r"$\langle\rho\rangle_x$")
    ax[0].set_title("planar interface, g_AB=6, N=64, 20000 steps")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    for label, style in (("tl_primary", "-"), ("tl_n32", "-"), ("tl_native", "--"), ("tl_linear", ":")):
        try:
            r = tl_arm.load(OUT, label)
        except FileNotFoundError:
            continue
        d = profile(r["fields"]["density_comp0"]) - profile(ref["fields"]["density_comp0"])
        ax[1].plot(y, d, style, lw=1.3, label=f"{label} - OSS")
    ax[1].axhline(0.0, color="k", lw=0.6)
    ax[1].set_xlabel("y (lattice node)")
    ax[1].set_ylabel(r"$\Delta\langle\rho_A\rangle_x$")
    ax[1].set_title("deviation from the reference profile")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_profiles_and_deviation.png", dpi=150)

    fig2, ax2 = plt.subplots(1, 2, figsize=(11, 4.2))
    g_ab = ref["params"]["gAB"]
    for k, lbl, sty in (("oss_formula_on_ref", "OSS formula (ref fields)", "o-"),
                        ("balance_form_on_ref", "balance form (ref fields)", "s-")):
        ax2[0].plot(y, flatness.__self__ if False else profile(pressure_oss(ref["fields"], g_ab)
                    if k.startswith("oss") else pressure_balance(ref["fields"], g_ab)),
                    sty, ms=3, lw=1.2, label=lbl)
    ax2[0].set_xlabel("y (lattice node)")
    ax2[0].set_ylabel("p (lattice units)")
    ax2[0].set_title("pressure profiles on the OSS fields")
    ax2[0].legend(fontsize=8)
    ax2[0].grid(alpha=0.3)

    try:
        hist = tl_arm.load(OUT, "tl_primary")["history"]
        if hist:
            s = [h["step"] for h in hist]
            ax2[1].plot(s, [h["rho1_max"] for h in hist], "o-", ms=3, label=r"max $\rho_A$ (TL)")
            ax2[1].plot(s, [h["rho1_min"] for h in hist], "s-", ms=3, label=r"min $\rho_A$ (TL)")
            ax2[1].set_xlabel("step")
            ax2[1].set_ylabel(r"$\rho_A$ extremes")
            ax2[1].set_title("time stationarity of the TensorLBM arm")
            ax2[1].legend(fontsize=8)
            ax2[1].grid(alpha=0.3)
    except FileNotFoundError:
        pass
    fig2.tight_layout()
    fig2.savefig(FIG / "fig2_pressure_and_stationarity.png", dpi=150)

    print("\n".join(lines))
    print("figures ->", FIG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())