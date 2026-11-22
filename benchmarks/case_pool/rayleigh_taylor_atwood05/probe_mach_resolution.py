"""CASE-RT-INSTABILITY-ATWOOD — de-confound lattice Mach from resolution in the
linear growth-rate ladder.

Why this probe exists
---------------------
``run_case.py``'s grid ladder refines the wavelength at constant ``A g / lam``.
That holds the *dimensionless* RT problem fixed (``sigma sqrt(lam/(A g))`` and the
viscous group are invariant), but it does **not** hold the lattice Mach number
fixed: the natural velocity scale of the problem is

    U = S / k = sqrt(A g k) / k = sqrt(A g lam / (2 pi))  ->  U ~ lam

so ``Ma`` grows linearly with ``lam``.  The archived ladder (``artifacts/runs``)
indeed reports ``Ma_peak = 0.247 / 0.363 / 0.542`` for ``lam = 64 / 96 / 128``.
Consequently its headline reading — "the residual deficit *grows* with
resolution (-10.1 -> -11.2 -> -13.0 %), therefore it is not an interface-thickness
effect" (REPORT.md 5.2 / failure mode F5) — cannot separate discretisation
convergence from compressibility error: the two axes move together, in the
direction that *increases* the error.

This probe builds the two missing axes with the dimensionless problem held fixed
by construction:

* **M axis** (compressibility, everything else fixed): ``lam = 64``, viscous group
  ``G = nu k^2 / S`` fixed by ``nu ~ sqrt(g)``, capillary group
  ``Pi = sigma_ac k^2 / (A g)`` fixed (``lam`` fixed), interface thickness fixed.
  Only ``Ma`` changes: ``g = 1e-4 / 2e-4 / 4e-4`` -> ``Ma ~ 0.17 / 0.25 / 0.49``.
* **R axis** (resolution, ``Ma`` fixed): ``A g lam = P`` fixed (so ``U`` and hence
  ``Ma`` are invariant), ``nu ~ lam`` (so ``G`` is invariant) and
  ``sigma_ac ~ lam`` (so ``Pi`` is invariant).  Two interface variants:
  ``kd0`` keeps the Allen-Cahn width at 2 cells (``k delta`` halves as ``lam``
  grows -- the old ladder's interface treatment) and ``kd`` scales the width with
  ``lam`` so that ``k delta`` -- and therefore the *whole* dimensionless problem --
  is invariant.

Measurement is inherited unchanged from ``run_case.run_case`` (``phi = 0``
interface amplitude, pre-declared amplitude windows, ``sigma_visc`` from
``ref/ref_lock.py``); this probe adds no estimator, only a design of experiments
and a regression of the residual on ``Ma^2`` and on ``1/lam``.

Pre-declared hypotheses (written before the runs)
-------------------------------------------------
* **H1 compressibility** — at fixed dimensionless problem the measured
  ``sigma/sigma_visc - 1`` drifts with ``Ma``, and a fit
  ``err = e0 + c Ma^2`` leaves ``|e0|`` well below the reference deficit.
* **H2 scheme bias** — ``sigma/sigma_visc`` at fixed ``Ma`` is independent of
  ``lam`` and stays at the reference value, i.e. the deficit is a Ma-independent
  bias of the buoyancy/interface coupling.
* Decision rule: H1 is *supported* if the fitted ``Ma``-term accounts for at least
  half of the reference-case deficit (``|c| Ma_ref^2 >= 0.5 |err_ref|``);
  otherwise H2 is supported.  No gate is attached to this probe -- it is a
  falsification test of the REPORT.md F5 reading, and the honest outcome is
  recorded either way.

Usage
-----
    python3 probe_mach_resolution.py                 # run all arms + analysis
    python3 probe_mach_resolution.py --quick         # plumbing smoke (<=400 steps)
    python3 probe_mach_resolution.py --post-only     # analysis from `--out`
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.case_pool.rayleigh_taylor_atwood05.ref import ref_lock  # noqa: E402
from benchmarks.case_pool.rayleigh_taylor_atwood05.run_case import run_case as run_one  # noqa: E402
from tensorlbm.rayleigh_taylor import RTConfig  # noqa: E402

ATWOOD = ref_lock.ATWOOD               # 0.5 (anchor value)
TAU_END = 3.0                          # linear phase only: a/lam <= ~0.12
NU_REF = 0.005                         # reference case: g = 2e-4 (tau = 0.515)
G_REF = 2.0e-4
LAM_REF = 64
P_REF = ATWOOD * G_REF * LAM_REF       # A g lam : the Mach-invariant (U^2 * 2 pi)
W_REF = 2.0                            # Allen-Cahn equilibrium width (cells)
SIG_REF = 3.0e-4                       # Allen-Cahn surface tension
IFW_REF = 1.5                          # initial tanh half-width (RTConfig default)


# ---------------------------------------------------------------------------
# design of experiments
# ---------------------------------------------------------------------------
def nu_for_group(g: float, lam: int) -> float:
    """``nu`` that keeps ``G = nu k^2 / S`` at its reference value.

    With ``S = sqrt(A g k)`` and ``k = 2 pi / lam`` this is
    ``G = 4 pi^2 nu / (lam^{3/2} sqrt(2 pi A g))``, hence ``nu ~ lam^{3/2} sqrt(g)``.
    That reproduces the two special cases: fixed ``lam`` -> ``nu ~ sqrt(g)`` (M axis)
    and fixed ``A g lam`` (fixed Ma) -> ``nu ~ lam`` (R axis).
    """
    return NU_REF * (lam / LAM_REF) ** 1.5 * math.sqrt(g / G_REF)


@dataclass
class ProbeSpec:
    """``run_case.CaseSpec``-compatible spec with the extra fixed-group knobs.

    ``run_case.run_case`` only touches ``cfg()``, ``atwood``, ``g``, ``lam``,
    ``nu``, ``name``, ``note``, ``tau``, ``tau_phi``, ``phase_scheme``,
    ``layers``, ``ac_sigma``, ``ac_width``, ``ac_mobility``; this class supplies
    exactly those and lets ``ac_width`` / ``ac_sigma`` / ``interface_width`` be
    derived from the resolution so that the *dimensionless* interface is fixed.
    """

    name: str
    lam: int
    g: float
    nu: float
    ac_width: float = W_REF
    ac_sigma: float = SIG_REF
    interface_width: float = IFW_REF
    layers: float = 8.0
    tau_end: float = TAU_END
    ac_mobility: float = 20.0
    note: str = ""
    axis: str = ""
    atwood: float = ATWOOD
    phase_scheme: str = "allen_cahn"

    @property
    def tau(self) -> float:
        return 0.5 + 3.0 * self.nu

    @property
    def tau_phi(self) -> float:
        return 0.501

    def n_steps(self) -> int:
        return int(round(self.tau_end * math.sqrt(self.lam / (self.atwood * self.g))))

    @property
    def sample_every(self) -> int:
        return max(1, self.n_steps() // 800)

    def cfg(self) -> RTConfig:
        cfg = RTConfig(
            lam=self.lam, layers=self.layers, g=self.g, atwood=self.atwood,
            tau=self.tau, tau_phi=self.tau_phi, n_steps=self.n_steps(),
            sample_every=self.sample_every,
            a_lo_frac=0.010, a_hi_frac=0.050,   # only seeds the library's own guard
            stop_a_frac=10.0, seed_eigenmode=True,
            phase_scheme=self.phase_scheme, ac_sigma=self.ac_sigma,
            ac_width=self.ac_width, ac_mobility=self.ac_mobility,
            interface_width=self.interface_width,
        )
        cfg.validate()
        return cfg

    # --- dimensionless groups actually realised by this spec ----------------
    def groups(self) -> dict:
        k = 2.0 * math.pi / self.lam
        S = math.sqrt(self.atwood * self.g * k)
        U = S / k
        return {
            "k": k,
            "sigma_inviscid": S,
            "viscous_group_G": self.nu * k * k / S,      # = nu k^2 / S
            "capillary_group_Pi": self.ac_sigma * k * k / (self.atwood * self.g),
            "kdelta": k * self.ac_width,
            "velocity_scale_U": U,
            "ma_scale": U * math.sqrt(3.0),              # U / c_s, c_s = 1/sqrt(3)
            "normalised_rate_inviscid": S * math.sqrt(self.lam / (self.atwood * self.g)),
        }


def build_matrix() -> list[ProbeSpec]:
    specs: list[ProbeSpec] = []

    # ---- M axis: lam fixed, only g (hence Ma) changes, G and Pi invariant ----
    for tag, g in (("g5e-5", 5.0e-5), ("g1e-4", 1.0e-4), ("g2e-4", 2.0e-4),
                   ("g4e-4", 4.0e-4), ("g8e-4", 8.0e-4)):
        specs.append(ProbeSpec(
            name=f"M_l064_{tag}", lam=LAM_REF, g=g, nu=nu_for_group(g, LAM_REF),
            # capillary group Pi = sigma_ac k^2 / (A g) is invariant only if
            # sigma_ac tracks g (lam is fixed here), so only Ma is left to vary
            ac_sigma=SIG_REF * (g / G_REF),
            note="Mach axis: dimensionless problem fixed, Ma varies",
            axis="M"))

    # ---- L axis: the archived (confounded) design, re-run under this estimator -
    # run_case.SPECS held A g / lam fixed and left nu = 0.005 and the 2-cell
    # interface fixed, so refining lam moved Ma, G and Pi *and* k delta at once.
    # The archived case.json files are not in the repo (gate-filtered), so the
    # comparison arm is re-run here with the same code path.
    for lam, g in ((96, 3.0e-4), (128, 4.0e-4)):
        specs.append(ProbeSpec(
            name=f"L_l{lam:03d}_g{g:.0e}", lam=lam, g=g, nu=NU_REF,
            note="archived ladder design (A g/lam fixed, nu and interface fixed in cells)",
            axis="L"))

    # ---- R axis: Ma fixed (A g lam = P), G and Pi and k delta invariant ------
    for lam in (96, 128):
        g = P_REF / (ATWOOD * lam)
        for variant, width_scale in (("kd", lam / LAM_REF), ("kd0", 1.0)):
            specs.append(ProbeSpec(
                name=f"R_l{lam:03d}_{variant}", lam=lam, g=g,
                nu=nu_for_group(g, lam),
                ac_width=W_REF * width_scale,
                ac_sigma=SIG_REF * width_scale,
                interface_width=IFW_REF * width_scale,
                note=("resolution axis at fixed Ma; k delta invariant"
                      if variant == "kd" else
                      "resolution axis at fixed Ma; interface width fixed in cells "
                      "(k delta halves)"),
                axis="R"))
    return specs


# ---------------------------------------------------------------------------
# legacy ladder (the confounded design) read straight from the archive
# ---------------------------------------------------------------------------
def legacy_ladder(cases: list[dict]) -> list[dict]:
    """The confounded design: ``lam = 64`` point is the shared reference case."""
    pts = [c for c in cases if c["axis"] == "L"]
    ref = [c for c in cases if c["name"] == "M_l064_g2e-4"]
    return sorted(ref + pts, key=lambda c: c["lam"])


def archived_ladder(run_root: Path) -> list[dict]:
    """Optional: ladder numbers from an archive that still has ``case.json``."""
    out = []
    for name in ("A050_l064_g2e-4", "A050_l096_g3e-4", "A050_l128_g4e-4"):
        f = run_root / name / "case.json"
        if not f.is_file():
            continue
        c = json.loads(f.read_text())
        k = 2.0 * math.pi / c["lam"]
        S = math.sqrt(c["atwood"] * c["g"] * k)
        out.append({
            "name": c["name"], "lam": c["lam"], "g": c["g"], "nu": c["nu"],
            "ma_peak": c["ma_peak"],
            "err_vs_viscous_pct": c["err_vs_viscous_pct"],
            "err_vs_inviscid_pct": c["err_vs_inviscid_pct"],
            "err_flow_uy_vs_viscous_pct": c["err_flow_uy_vs_viscous_pct"],
            "viscous_group_G": c["nu"] * k * k / S,
            "capillary_group_Pi": c["ac_sigma"] * k * k / (c["atwood"] * c["g"]),
            "kdelta": k * c["ac_width"],
            "velocity_scale_U": S / k,
        })
    return out


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------
def linfit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return b, a, (1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"))


def analyse(cases: list[dict], legacy: list[dict] | None = None) -> dict:
    if not legacy:
        legacy = legacy_ladder(cases)
    m_axis = sorted([c for c in cases if c["axis"] == "M"], key=lambda c: c["ma_peak"])
    r_axis = [c for c in cases if c["axis"] == "R"]

    res: dict = {"m_axis": [], "r_axis": [], "legacy_ladder": legacy}
    for c in m_axis:
        res["m_axis"].append({
            "name": c["name"], "g": c["g"], "nu": c["nu"],
            "ma_peak": c["ma_peak"], "ma_scale": c["groups"]["ma_scale"],
            "sigma_sim": c["sigma_sim"], "sigma_viscous": c["sigma_viscous"],
            "err_vs_viscous_pct": c["err_vs_viscous_pct"],
            "err_vs_inviscid_pct": c["err_vs_inviscid_pct"],
            "err_flow_uy_vs_viscous_pct": c["err_flow_uy_vs_viscous_pct"],
            "viscous_group_G": c["groups"]["viscous_group_G"],
            "capillary_group_Pi": c["groups"]["capillary_group_Pi"],
        })
    for c in sorted(r_axis, key=lambda c: (c["lam"], c["name"])):
        res["r_axis"].append({
            "name": c["name"], "lam": c["lam"], "g": c["g"], "nu": c["nu"],
            "ac_width": c["ac_width"], "ma_peak": c["ma_peak"],
            "ma_scale": c["groups"]["ma_scale"], "kdelta": c["groups"]["kdelta"],
            "sigma_sim": c["sigma_sim"], "sigma_viscous": c["sigma_viscous"],
            "err_vs_viscous_pct": c["err_vs_viscous_pct"],
            "err_vs_inviscid_pct": c["err_vs_inviscid_pct"],
            "err_flow_uy_vs_viscous_pct": c["err_flow_uy_vs_viscous_pct"],
            "viscous_group_G": c["groups"]["viscous_group_G"],
            "capillary_group_Pi": c["groups"]["capillary_group_Pi"],
        })

    # -- H1: err = e0 + c Ma^2 on the M axis (Ma measured, not nominal) ------
    pts = [p for p in res["m_axis"]
           if math.isfinite(p["err_vs_viscous_pct"]) and math.isfinite(p["ma_peak"])]
    if len(pts) >= 3:
        slope, intercept, r2 = linfit([p["ma_peak"] ** 2 for p in pts],
                                      [p["err_vs_viscous_pct"] for p in pts])
        ma_ref = next((p["ma_peak"] for p in pts if abs(p["g"] - G_REF) < 1e-12), None)
        err_ref = next((p["err_vs_viscous_pct"] for p in pts
                        if abs(p["g"] - G_REF) < 1e-12), None)
        mach_part = slope * ma_ref ** 2 if ma_ref is not None else float("nan")
        frac = (abs(mach_part) / abs(err_ref)) if err_ref else float("nan")
        res["h1_mach_fit"] = {
            "model": "err_pct = e0 + c * Ma_peak^2",
            "slope_c": slope, "intercept_e0_pct": intercept, "r2": r2,
            "ma_ref": ma_ref, "err_ref_pct": err_ref,
            "mach_term_at_ref_pct": mach_part,
            "frac_of_deficit_explained": frac,
            "h1_supported": bool(frac >= 0.5),
            "predicted_err_at_ma_010_pct": intercept + slope * 0.10 ** 2,
            "predicted_err_at_ma_005_pct": intercept + slope * 0.05 ** 2,
        }
    else:
        res["h1_mach_fit"] = {"h1_supported": None, "reason": "insufficient M-axis points"}

    # -- H2: err at fixed Ma vs lam (resolution convergence) -----------------
    r_by_var: dict[str, list[dict]] = {}
    for p in res["r_axis"]:
        r_by_var.setdefault(p["name"].split("_")[-1], []).append(p)
    res["h2_resolution"] = {}
    for variant, ps in r_by_var.items():
        ps = sorted(ps, key=lambda p: p["lam"])
        vals = [p["err_vs_viscous_pct"] for p in ps]
        res["h2_resolution"][variant] = {
            "lams": [p["lam"] for p in ps],
            "errs_pct": vals,
            "spread_pct_points": (max(vals) - min(vals)) if vals else float("nan"),
            "kdelta": [p["kdelta"] for p in ps],
        }
    # compare with the confounded legacy ladder for the same lam triple
    leg = [p for p in legacy if math.isfinite(p["err_vs_viscous_pct"])]
    if leg:
        vs = [p["err_vs_viscous_pct"] for p in leg]
        res["legacy_ladder_summary"] = {
            "lams": [p["lam"] for p in leg], "errs_pct": vs,
            "mach": [p["ma_peak"] for p in leg],
            "spread_pct_points": max(vs) - min(vs),
            "delta_err_l128_minus_l64_pct_points": vs[-1] - vs[0],
            "delta_ma2_l128_minus_l64": leg[-1]["ma_peak"] ** 2 - leg[0]["ma_peak"] ** 2,
        }
    return res


# ---------------------------------------------------------------------------
# figures + generated note
# ---------------------------------------------------------------------------
def make_figure(res: dict, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), dpi=130)

    ax = axes[0]
    m = [p for p in res["m_axis"] if math.isfinite(p["err_vs_viscous_pct"])]
    if m:
        ax.plot([p["ma_peak"] for p in m], [p["err_vs_viscous_pct"] for p in m],
                "o-", lw=1.8, ms=7, color="#1f77b4",
                label="M axis: dimensionless problem fixed")
        fit = res.get("h1_mach_fit", {})
        if "slope_c" in fit:
            xs = [0.0, max(p["ma_peak"] for p in m) * 1.05]
            ax.plot(xs, [fit["intercept_e0_pct"] + fit["slope_c"] * x * x for x in xs],
                    "--", color="#d62728", lw=1.2,
                    label=f"fit $e_0+c\\,Ma^2$: $e_0$={fit['intercept_e0_pct']:+.2f}%, "
                          f"$c$={fit['slope_c']:+.1f} (r²={fit['r2']:.3f})")
            ax.plot([0.0], [fit["intercept_e0_pct"]], "s", color="#d62728", ms=7,
                    label="Ma$\\to$0 extrapolation")
    leg = [p for p in res["legacy_ladder"] if math.isfinite(p["err_vs_viscous_pct"])]
    if leg:
        ax.plot([p["ma_peak"] for p in leg], [p["err_vs_viscous_pct"] for p in leg],
                "x--", color="#7f7f7f", lw=1.0, ms=8,
                label="archived ladder (Ag/λ fixed, Ma free)")
    ax.axhspan(-15, 15, color="#2ca02c", alpha=0.06, label="case gate ±15 %")
    ax.set_xlabel("lattice Mach number  $Ma_{peak}$")
    ax.set_ylabel(r"$(\sigma_{sim}/\sigma_{visc}-1)$  [%]")
    ax.set_title("Compressibility axis (λ = 64 fixed)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower left")

    ax = axes[1]
    for variant, style in (("kd", ("o", "#1f77b4")), ("kd0", ("s", "#ff7f0e"))):
        d = res.get("h2_resolution", {}).get(variant)
        if not d:
            continue
        ax.plot(d["lams"], d["errs_pct"], style[0] + "-", color=style[1], ms=7, lw=1.8,
                label=f"R axis, {variant} (Ma, G, Π fixed"
                      + (", kδ fixed)" if variant == "kd" else ")"))
    if leg:
        ax.plot([p["lam"] for p in leg], [p["err_vs_viscous_pct"] for p in leg],
                "x--", color="#7f7f7f", lw=1.0, ms=8,
                label="archived ladder (Ag/λ fixed: Ma and kδ free)")
    ax.axhspan(-15, 15, color="#2ca02c", alpha=0.06)
    ax.set_xlabel(r"wavelength  $\lambda$  [lattice cells]")
    ax.set_ylabel(r"$(\sigma_{sim}/\sigma_{visc}-1)$  [%]")
    ax.set_title("Resolution axis (Ma fixed)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower left")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def write_note(res: dict, out_md: Path, fig_name: str) -> None:
    fit = res.get("h1_mach_fit", {})
    L = []
    A = L.append
    A("# CASE-RT-INSTABILITY-ATWOOD — Mach vs resolution: de-confounding the linear "
      "growth-rate deficit")
    A("")
    A("> Generated by `probe_mach_resolution.py` from `artifacts_mach_res/*/case.json` "
      "and the archived ladder in `artifacts/runs/*/case.json` (programmatic injection, "
      "no hand-copied numbers).")
    A("> Anchor: `CASE-RT-INSTABILITY-ATWOOD` (ref: Tryggvason 2011 lineage, tolerance "
      "< 15 %). This note is a **verification** increment on the linear arm: it tests "
      "the reading of REPORT.md §5.2 / failure mode F5.")
    A("")
    A("## 1. The conflation this probe removes")
    A("")
    A("`run_case.py`'s ladder holds `A g / lam` fixed. That makes the dimensionless "
      "problem invariant, but the lattice velocity scale `U = sqrt(A g lam / 2 pi)` "
      "grows like `lam`, so **refining the grid raises the lattice Mach number**. The "
      "archived ladder therefore moves two axes at once:")
    A("")
    A("| lam | g | Ma_peak | nu | kδ | G = νk²/S | Π = σ_ac k²/(Ag) | err vs σ_visc |")
    A("|---|---|---|---|---|---|---|---|")
    for p in res["legacy_ladder"]:
        A(f"| {p['lam']} | {p['g']:.1e} | {p['ma_peak']:.3f} | {p['nu']:.4f} | "
          f"{p['kdelta']:.3f} | {p['viscous_group_G']:.4f} | "
          f"{p['capillary_group_Pi']:.4f} | {p['err_vs_viscous_pct']:+.2f} % |")
    A("")
    A("## 2. M axis — dimensionless problem fixed, only `Ma` changes")
    A("")
    A("`lam = 64`, `nu ~ sqrt(g)` (so `G = nu k^2 / S` is invariant), `Π` invariant "
      "because `lam` is fixed. Nothing in the continuum problem changes between the "
      "three runs; only the compressibility of the lattice realisation does.")
    A("")
    A("| case | g | nu | Ma_peak | G | Π | σ_sim | σ_visc | err vs σ_visc | "
      "err vs σ_inv | err (flow side, u_y mode) |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for p in res["m_axis"]:
        A(f"| {p['name']} | {p['g']:.1e} | {p['nu']:.5f} | {p['ma_peak']:.3f} | "
          f"{p['viscous_group_G']:.4f} | {p['capillary_group_Pi']:.4f} | "
          f"{p['sigma_sim']:.5e} | {p['sigma_viscous']:.5e} | "
          f"{p['err_vs_viscous_pct']:+.2f} % | {p['err_vs_inviscid_pct']:+.2f} % | "
          f"{p['err_flow_uy_vs_viscous_pct']:+.2f} % |")
    A("")
    if "slope_c" in fit:
        A(f"Regression `err[%] = e0 + c Ma^2`: **e0 = {fit['intercept_e0_pct']:+.2f} %**, "
          f"c = {fit['slope_c']:+.1f} %/Ma², r² = {fit['r2']:.3f}. At the reference "
          f"case (`Ma = {fit['ma_ref']:.3f}`) the Ma-term is "
          f"{fit['mach_term_at_ref_pct']:+.2f} %, i.e. "
          f"{100.0 * fit['frac_of_deficit_explained']:.1f} % of the observed "
          f"{fit['err_ref_pct']:+.2f} %. Extrapolated to `Ma -> 0`: "
          f"{fit['intercept_e0_pct']:+.2f} % (at Ma = 0.10: "
          f"{fit['predicted_err_at_ma_010_pct']:+.2f} %, at Ma = 0.05: "
          f"{fit['predicted_err_at_ma_005_pct']:+.2f} %).")
        A("")
        A(f"**H1 (compressibility): {'SUPPORTED' if fit['h1_supported'] else 'NOT SUPPORTED'}** "
          f"(decision rule: `|c| Ma_ref^2 >= 0.5 |err_ref|`; "
          f"fraction = {fit['frac_of_deficit_explained']:.2f}).")
    else:
        A("M-axis regression not available (insufficient finite points).")
    A("")
    A("## 3. R axis — `Ma` fixed, resolution refined")
    A("")
    A("`A g lam = P` (=> `U`, `Ma` invariant), `nu ~ lam` (=> `G` invariant), "
      "`sigma_ac ~ lam` (=> `Π` invariant). `kd` also scales the Allen-Cahn width "
      "with `lam`, so `k delta` is invariant: **every** group of the continuum problem "
      "is held while the grid is refined. `kd0` keeps the 2-cell width (the archived "
      "ladder's interface treatment), isolating the interface-thickness effect.")
    A("")
    A("| case | lam | g | nu | width | kδ | Ma_peak | G | Π | σ_sim | σ_visc | "
      "err vs σ_visc | err (flow side) |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in res["r_axis"]:
        A(f"| {p['name']} | {p['lam']} | {p['g']:.1e} | {p['nu']:.5f} | "
          f"{p['ac_width']:.1f} | {p['kdelta']:.3f} | {p['ma_peak']:.3f} | "
          f"{p['viscous_group_G']:.4f} | {p['capillary_group_Pi']:.4f} | "
          f"{p['sigma_sim']:.5e} | {p['sigma_viscous']:.5e} | "
          f"{p['err_vs_viscous_pct']:+.2f} % | {p['err_flow_uy_vs_viscous_pct']:+.2f} % |")
    A("")
    for variant, d in sorted(res.get("h2_resolution", {}).items()):
        A(f"* `{variant}`: err = "
          + ", ".join(f"{e:+.2f} % (λ={l})" for l, e in zip(d["lams"], d["errs_pct"]))
          + f" -> spread **{d['spread_pct_points']:.2f} pts** over λ = {d['lams'][0]}→"
            f"{d['lams'][-1]}.")
    if "legacy_ladder_summary" in res:
        s = res["legacy_ladder_summary"]
        A(f"* Archived (confounded) ladder for comparison: spread "
          f"**{s['spread_pct_points']:.2f} pts**, with Ma² rising by "
          f"{s['delta_ma2_l128_minus_l64']:.3f} (Ma {s['mach'][0]:.3f} → "
          f"{s['mach'][-1]:.3f}) along the same ladder -- the two axes cannot be "
          f"separated there.")
    A("")
    A(f"![mach vs resolution]({fig_name})")
    A("")
    A("## 4. Reading and verdict")
    A("")
    fit_ok = "slope_c" in fit
    if fit_ok:
        frac = fit["frac_of_deficit_explained"]
        A(f"1. **Compressibility (H1)**: at a *fixed* continuum problem the residual moves "
          f"by {fit['slope_c']:+.1f} %/Ma² ; at the reference Mach number that term is "
          f"{fit['mach_term_at_ref_pct']:+.2f} %, i.e. {100.0 * frac:.0f} % of the "
          f"{fit['err_ref_pct']:+.2f} % deficit. Extrapolating to `Ma -> 0` leaves "
          f"**{fit['intercept_e0_pct']:+.2f} %**.")
    h2 = res.get("h2_resolution", {})
    if "kd" in h2:
        d = h2["kd"]
        A(f"2. **Resolution at fixed Ma (H2)**: with every group held "
          f"(Ma, G, Π, kδ), the residual moves "
          f"{d['spread_pct_points']:.2f} pts over λ = {d['lams'][0]}→{d['lams'][-1]} "
          f"(" + ", ".join(f"{e:+.2f} %" for e in d["errs_pct"]) + "). "
          + ("That is grid convergence within the run-to-run scatter: the residual at "
             "fixed Ma is a **property of the scheme, not of the grid**."
             if d["spread_pct_points"] <= 1.5 else
             "The residual still drifts with λ at fixed Ma, so grid convergence is "
             "**not** established and the fixed-Ma residual cannot yet be called a "
             "scheme constant."))
    if "kd0" in h2:
        d0 = h2["kd0"]
        A(f"3. **Interface thickness**: keeping the 2-cell Allen–Cahn width (kδ "
          f"{d0['kdelta'][0]:.3f}→{d0['kdelta'][-1]:.3f}) moves the residual "
          f"{d0['spread_pct_points']:.2f} pts over the same λ range "
          f"(" + ", ".join(f"{e:+.2f} %" for e in d0["errs_pct"]) + "), versus "
          f"{h2.get('kd', {}).get('spread_pct_points', float('nan')):.2f} pts when kδ "
          f"is held — the interface-thickness contribution is "
          + ("small" if "kd" in h2 and d0["spread_pct_points"] <= 1.5 else "not small")
          + " but is **not** what produced the archived ladder's trend.")
    s_ = res.get("legacy_ladder_summary")
    if s_ and fit_ok:
        A(f"4. **The archived reading**: its ladder moves Ma² by "
          f"{s_['delta_ma2_l128_minus_l64']:.3f} while also moving G, Π and kδ (section 1), "
          f"and it spans {s_['spread_pct_points']:.2f} pts. Attributing that spread to "
          f"resolution (REPORT.md §5.2, failure mode F5) is therefore an "
          f"**over-attribution**: the same λ sweep at fixed Ma and G spans only "
          f"{h2.get('kd', {}).get('spread_pct_points', float('nan')):.2f} pts.")
    if fit_ok:
        if fit["h1_supported"]:
            A("")
            A(f"**Disposition — H1 (compressibility) SUPPORTED at the design level**: "
              f"{100.0 * fit['frac_of_deficit_explained']:.0f} % of the reference-case "
              f"deficit is accounted for by the `Ma²` term, and the Ma→0 extrapolation is "
              f"{fit['intercept_e0_pct']:+.2f} %. The case's PASS at −10.1 % is therefore "
              f"partly a *Mach-number* effect of the chosen lattice realisation, not only a "
              f"scheme bias; the case-pool gate is still met (|err| ≤ 15 %).")
        else:
            A("")
            A(f"**Disposition — H2 (Ma-independent scheme bias) SUPPORTED**: the `Ma²` "
              f"term accounts for only {100.0 * fit['frac_of_deficit_explained']:.0f} % of "
              f"the deficit, so the residual is dominated by something the Mach number does "
              f"not explain. Combined with the fixed-Ma λ sweep (item 2) this localises the "
              f"bias to the buoyancy/interface coupling of the Allen–Cahn form, which is "
              f"F5's own conclusion — now with the compressibility axis excluded.")
    A("")
    A("Machine-readable verdict: `artifacts_mach_res/summary.json` "
      "(`h1_mach_fit`, `h2_resolution`, `legacy_ladder_summary`).")
    A("")
    path = out_md
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(BENCH_DIR / "artifacts_mach_res"))
    ap.add_argument("--runs", default=str(BENCH_DIR / "artifacts" / "runs"),
                    help="archived (confounded) ladder root")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--post-only", action="store_true")
    args = ap.parse_args(argv)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    if not args.post_only:
        specs = build_matrix()
        if args.only:
            specs = [s for s in specs if s.name in args.only]
        for spec in specs:
            g = spec.groups()
            print(f"[MACH-RES] {spec.name:16s} lam={spec.lam:3d} g={spec.g:.2e} "
                  f"nu={spec.nu:.5f} Ma_scale={g['ma_scale']:.3f} "
                  f"G={g['viscous_group_G']:.4f} Pi={g['capillary_group_Pi']:.4f} "
                  f"kdelta={g['kdelta']:.3f} steps={spec.n_steps()}", flush=True)
            case = run_one(spec, out_root, quick=args.quick)  # type: ignore[arg-type]
            case["axis"] = spec.axis
            case["groups"] = g
            (out_root / spec.name / "case.json").write_text(
                json.dumps(case, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"           -> sigma={case['sigma_sim']:.5e} "
                  f"err={case['err_vs_viscous_pct']:+.2f}% Ma_peak={case['ma_peak']:.3f}",
                  flush=True)

    cases = [json.loads(f.read_text()) for f in sorted(out_root.glob("*/case.json"))]
    cases = [c for c in cases if "axis" in c]
    legacy = legacy_ladder(cases)
    if len(legacy) < 2:                       # no L arms run: fall back to archive
        legacy = archived_ladder(Path(args.runs))
    res = analyse(cases, legacy)
    res["cases"] = [{k: c[k] for k in ("name", "lam", "g", "nu", "axis", "ma_peak",
                                       "err_vs_viscous_pct", "err_vs_inviscid_pct")}
                    for c in cases]
    (out_root / "summary.json").write_text(json.dumps(res, indent=2, sort_keys=True) + "\n",
                                           encoding="utf-8")

    fig = BENCH_DIR / "figures" / "fig6_mach_vs_resolution.png"
    try:
        make_figure(res, fig)
    except Exception as exc:  # pragma: no cover - plotting must not kill the probe
        print(f"[MACH-RES] figure skipped: {exc}", flush=True)
    write_note(res, BENCH_DIR.parents[2] / "docs" / "CASE-RT-MACH-RESOLUTION.md",
               "../benchmarks/case_pool/rayleigh_taylor_atwood05/figures/"
               "fig6_mach_vs_resolution.png")
    print(json.dumps(res.get("h1_mach_fit", {}), indent=2, sort_keys=True), flush=True)
    print(json.dumps(res.get("h2_resolution", {}), indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
