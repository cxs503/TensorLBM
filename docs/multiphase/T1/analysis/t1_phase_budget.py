"""T1 multiphase phase-budget analysis (GAP-LBM-T1-MULTIPHASE, round 1, research only).

This script turns the qualitative five-model audit (repo records
``benchmarks/pending/hadamard_rybczynski_droplet/terminal_verdict_m155.md`` and
``docs/multiphase/MULTIPHASE_CAPABILITY_AUDIT_2026-10-18.md``) into pre-registered
numbers: for each model family it computes the parameter ceilings that decide whether
a Hadamard-Rybczynski (HR) buoyant droplet can be represented at all, and how large a
spurious current the discretisation is allowed to have.

Nothing here runs a solver.  Every number is either
  * an algebraic/thermodynamic identity evaluated on a grid (the equations are quoted in
    the T1 docs), or
  * a *reduction of already-committed repo measurements* loaded from repository JSON
    artifacts (never retyped by hand; paths are printed in the provenance block).

Outputs (all inside the worktree):
  artifacts/T1_multiphase/t1_phase_budget.json
  docs/multiphase/T1/T1_NUMERIC_APPENDIX.md   (tables only, fully generated)
  docs/multiphase/T1/figs/fig*.png

Run:  python3 docs/multiphase/T1/analysis/t1_phase_budget.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
DOCS = REPO / "docs" / "multiphase" / "T1"
FIGS = DOCS / "figs"
ART = REPO / "artifacts" / "T1_multiphase"

CS2 = 1.0 / 3.0  # c_s^2 for every D2Q9/D3Q19/D3Q27 velocity set in the repo

# --------------------------------------------------------------------------------------
# 1. Shan-Chen pseudopotential: coexistence from the Carnahan-Starling equation of state
#       p(rho) = rho RT (1+eta+eta^2-eta^3)/(1-eta)^3 - a rho^2,   eta = b rho / 4
#
#    PARAMETER WARNING (source-confirmed, not a hypothesis): the legacy
#    ``tensorlbm.multiphase.psi_carnahan_starling`` hard-codes a = 0.5, b = 4, RT = 1/3.
#    With those values dp/drho > 0 on the whole admissible density interval, i.e. the EOS
#    has NO van der Waals loop, NO spinodal and therefore NO two-phase region at all
#    (checked in this script: ``legacy_loop_scan`` below).  The module's own docstring
#    already records this ("no van der Waals loop ... the model cannot phase-separate at
#    all").  The usable builder is ``make_psi_carnahan_starling(a=5.0, b=4.0, RT=1/3)``;
#    this script therefore runs the coexistence construction on a = 5.0.
# --------------------------------------------------------------------------------------
A_CS, B_CS, RT_CS = 5.0, 4.0, 1.0 / 3.0
A_CS_LEGACY, B_CS_LEGACY = 0.5, 4.0
RHO_MAX = 0.999 * 4.0 / B_CS


def p_cs(rho: float, rt: float = RT_CS) -> float:
    eta = B_CS * rho / 4.0
    if eta >= 1.0:
        return 1e6
    return rho * rt * (1.0 + eta + eta**2 - eta**3) / (1.0 - eta) ** 3 - A_CS * rho**2


def has_vdw_loop(a: float, rt: float, b: float = B_CS, n: int = 3000) -> bool:
    """True iff dp/drho < 0 somewhere: i.e. the EOS admits a two-phase region."""
    grid = np.linspace(2e-3, 0.999 * 4.0 / b - 1e-5, n)
    pp = np.array([_p_with(a, b, rt, r) for r in grid])
    return bool((np.diff(pp) < 0).any())


def _p_with(a: float, b: float, rt: float, rho: float) -> float:
    eta = b * rho / 4.0
    if eta >= 1.0:
        return 1e6
    return rho * rt * (1.0 + eta + eta**2 - eta**3) / (1.0 - eta) ** 3 - a * rho**2


def _bisect(f, lo: float, hi: float, tol: float = 1e-14, n: int = 200) -> float:
    flo = f(lo)
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if flo * fm <= 0.0:
            hi = mid
        else:
            lo, flo = mid, fm
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def cs_spinodal(rt: float = RT_CS):
    """Gas/liquid spinodal densities: dp/drho = 0 (numeric scan + bisection)."""
    def dp(rho):
        h = 1e-7
        a, b = rho + h, rho - h
        if b <= 0:
            b = rho
        return (p_cs(a, rt) - p_cs(b, rt)) / (a - b)

    grid = np.linspace(1e-3, RHO_MAX - 1e-5, 3000)
    d = np.array([dp(r) for r in grid])
    sgn = np.sign(d)
    idx = list(np.where(np.diff(sgn) != 0)[0])
    if len(idx) < 2:
        raise RuntimeError("no spinodal found: EOS parameters outside the two-phase range")
    return float(grid[idx[0]]), float(grid[idx[1] + 1])


def cs_coexistence(rt: float = RT_CS) -> dict:
    """Maxwell equal-area coexistence (equal pressure + equal Gibbs)."""
    rg_s, rl_s = cs_spinodal(rt)

    def rho_l_for(rg: float, p_sat: float) -> float:
        return _bisect(lambda r: p_cs(r, rt) - p_sat, rl_s, RHO_MAX - 1e-6)

    def residual(rg: float) -> float:
        p_sat = p_cs(rg, rt)
        rl = rho_l_for(rg, p_sat)
        rr = np.linspace(rg, rl, 4001)
        integ = np.trapezoid(np.array([p_cs(r, rt) for r in rr]) / rr**2, rr)
        return integ - p_sat * (1.0 / rg - 1.0 / rl)

    lo, hi = 1e-4, rg_s * (1.0 - 1e-9)
    rg = _bisect(residual, lo, hi)
    p_sat = p_cs(rg, rt)
    rl = rho_l_for(rg, p_sat)
    return {"rho_g": rg, "rho_l": rl, "p_sat": p_sat, "ratio": rl / rg, "T_r": rt / RT_CS}


def sc_section() -> dict:
    co = cs_coexistence(RT_CS)
    sweep = []
    for T_r in (0.60, 0.70, 0.80, 0.85, 0.90, 0.95):
        try:
            c = cs_coexistence(rt=T_r * RT_CS)
            sweep.append({"T_r": T_r, "rho_g": c["rho_g"], "rho_l": c["rho_l"],
                          "ratio": c["ratio"], "drho_over_rhol": 1.0 - 1.0 / c["ratio"]})
        except Exception as exc:  # no coexistence at that reduced temperature
            sweep.append({"T_r": T_r, "error": str(exc)[:60]})
    legacy = {
        "a": A_CS_LEGACY, "b": B_CS_LEGACY, "RT": RT_CS,
        "has_vdw_loop": has_vdw_loop(A_CS_LEGACY, RT_CS),
        "usable_a": A_CS, "usable_b": B_CS, "usable_RT": RT_CS,
        "usable_has_vdw_loop": has_vdw_loop(A_CS, RT_CS),
        "source": "src/tensorlbm/multiphase.py::psi_carnahan_starling (hard-coded a=0.5) "
                  "and make_psi_carnahan_starling (a=5.0)",
    }
    lock = [{"rho_ratio": r, "lambda_uniform_tau": 1.0, "lambda_tau_of_rho": r}
            for r in (2.0, 3.0, 9.0)]
    return {"coexistence": co, "sweep": sweep, "viscosity_lock": lock, "legacy_params": legacy}


# --------------------------------------------------------------------------------------
# 2. Colour-gradient: total-density contrast the isothermal EOS can support.
#    CG advects rho = rho_r + rho_b and takes p = rho c_s^2.  Static mechanical
#    equilibrium needs  p_in - p_out = (dim-1) sigma / R, hence
#        d rho_EOS,max = (dim-1) sigma / (R c_s^2)
#    Any larger initial contrast is vented acoustically (that is the measured collapse
#    in probe/p0_static.json).
# --------------------------------------------------------------------------------------
def cg_ceiling(sigma, R, dim: int = 3):
    return (dim - 1) * np.asarray(sigma) / (np.asarray(R) * CS2)


def cg_sigma_inference(cg_cases: dict) -> list:
    """Invert the measured CG plateau contrast into the effective surface tension.

        d rho_plateau = 2 sigma / (R c_s^2)   =>   sigma_implied = d rho R c_s^2 / 2

    Both probe cases (r_in = 0.5 and r_in = 2.0) collapse to the SAME plateau, so the
    two independent inversions are a consistency check of the acoustic-ceiling reading.
    """
    rows = []
    for name, c in (cg_cases or {}).items():
        if not c.get("ratio_final") or not c.get("R_eq_final"):
            continue
        drho = c["ratio_final"] - 1.0
        rows.append({"case": name, "ratio_final": c["ratio_final"],
                     "R_eq_final": c["R_eq_final"],
                     "drho_plateau": drho,
                     "sigma_implied": drho * c["R_eq_final"] * CS2 / 2.0})
    return rows


def cg_section() -> dict:
    Rs = [6.0, 9.0, 12.0, 16.0, 24.0]
    sigmas = [1.0e-3, 1.0e-2, 5.0e-2]
    rows = []
    for s in sigmas:
        for R in Rs:
            dr = float(cg_ceiling(s, R))
            rows.append({"sigma": s, "R": R, "drho_max": dr, "ratio_max": 1.0 + dr})
    tgt = []
    for R in Rs:
        for target_ratio in (1.05, 1.5, 2.0, 3.0):
            drho = target_ratio - 1.0
            tgt.append({"R": R, "target_ratio": target_ratio,
                        "sigma_required": drho * CS2 * R / 2.0})
    return {"ceiling_rows": rows, "required_sigma_rows": tgt}


# --------------------------------------------------------------------------------------
# 3. Allen-Cahn (non-conservative, Model A): a dispersed droplet must shrink.
#       v_n = -Mtilde sigma kappa,  kappa = 2/R  =>  dR/dt = -2 Mtilde sigma / R
#       =>  R^2 = R0^2 - 4 Mtilde sigma t,  t* = R0^2 / (4 Mtilde sigma)
#    The collapse time is a property of the model, not of the grid: refinement cannot
#    save it.  Model B (Cahn-Hilliard / conservative AC with interface-sharpening flux)
#    conserves the order-parameter integral and does not predict collapse.
# --------------------------------------------------------------------------------------
def ac_collapse(R0: float, Msigma: float) -> float:
    return R0**2 / (4.0 * Msigma)


def ac_section() -> dict:
    rows = []
    for Msigma in (0.02, 0.05, 0.16, 0.5):
        for R0 in (4.0, 8.0, 11.4, 16.0):
            rows.append({"Mtilde_sigma": Msigma, "R0": R0,
                         "t_collapse": ac_collapse(R0, Msigma)})
    return {"grid": rows}


# --------------------------------------------------------------------------------------
# 4. HR budget: the dimensionless groups that decide feasibility.
#       U   = (2/3) drho g R^2 (1+lam) / (mu_o (2+3 lam))     [Hadamard 1911; Rybczynski 1911]
#       Bo  = drho g (2R)^2 / sigma = 4 drho g R^2 / sigma
#       Ca  = mu_o U / sigma
#       Re  = 2 rho_o U R / mu_o
#    Eliminating the buoyancy term gives the exact identity
#       Ca = U mu_o / sigma = Bo (1+lam) / (6 (2+3 lam))
#    i.e. for a HR drop Bo and Ca are NOT independent.  The spurious-current requirement
#    for a 10% terminal-velocity error is  S := u_sp mu_o / sigma < 0.1 Ca.
# --------------------------------------------------------------------------------------
def hr_identity(bo: float, lam: float) -> float:
    return bo * (1.0 + lam) / (6.0 * (2.0 + 3.0 * lam))


def hr_section() -> dict:
    rows = []
    for bo in (0.001, 0.005, 0.01, 0.05, 0.2):
        for lam in (0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0):
            ca = hr_identity(bo, lam)
            rows.append({"Bo": bo, "lambda": lam, "Ca": ca,
                         "S_required_for_10pct": 0.1 * ca,
                         "S_required_for_3pct": 0.03 * ca})
    return {"rows": rows}


def hr_case_design(R: float, ratio_rho: float, lam: float, nu_o: float,
                   Re_t: float, Bo_t: float) -> dict:
    """Solve the (g, sigma) pair placing the HR benchmark at a chosen (Re, Bo).

    Units dx = dt = 1, rho_o = 1 (so mu_o = nu_o).
    """
    rho_o = 1.0
    mu_o = rho_o * nu_o
    drho = (ratio_rho - 1.0) * rho_o
    rho_i = rho_o + drho
    nu_i = lam * nu_o * rho_o / rho_i          # mu_i = lam mu_o
    U = Re_t * mu_o / (2.0 * rho_o * R)        # from Re = 2 rho_o U R / mu_o
    g = U / ((2.0 / 3.0) * drho * R**2 * (1.0 + lam) / (mu_o * (2.0 + 3.0 * lam)))
    sigma = 4.0 * drho * g * R**2 / Bo_t       # from Bo = 4 drho g R^2 / sigma
    ca = mu_o * U / sigma
    return {
        "R": R, "rho_ratio": ratio_rho, "lambda": lam, "nu_o": nu_o, "nu_i": nu_i,
        "tau_o": 0.5 + 3.0 * nu_o, "tau_i": 0.5 + 3.0 * nu_i,
        "Re_target": Re_t, "Bo_target": Bo_t,
        "U_HR": U, "g": g, "sigma": sigma, "Ca": ca,
        "Bo_check": 4.0 * drho * g * R**2 / sigma,
        "Re_check": 2.0 * rho_o * U * R / mu_o,
        "S_budget_10pct": 0.1 * ca,
        "u_sp_budget_10pct": 0.1 * U,
        "t_diffusive": R**2 / nu_o,
        "steps_to_terminal_est": 3.0 * R**2 / nu_o,
        "gravity_head_over_cs2": rho_o * g * 4.0 * R / CS2,
    }


# --------------------------------------------------------------------------------------
# 5. Reduce committed repo measurements (never retyped) into the spurious number S.
# --------------------------------------------------------------------------------------
def load_measurements() -> dict:
    paths = {
        "p0_static": REPO / "benchmarks/pending/hadamard_rybczynski_droplet/probe/p0_static.json",
        "p0b_envelope": REPO / "benchmarks/pending/hadamard_rybczynski_droplet/probe/p0b_envelope.json",
        "p1_ac": REPO / "benchmarks/pending/hadamard_rybczynski_droplet/probe/p1_ac.json",
        "evidence": REPO / "benchmarks/pending/hadamard_rybczynski_droplet/out/evidence.json",
    }
    out = {"provenance": {}, "cg_cases": {}, "scmcmp_envelope": None, "ac_probe": None}

    for k, p in paths.items():
        out["provenance"][k] = str(p.relative_to(REPO)) if p.exists() else "MISSING"

    p = paths["p0_static"]
    if p.exists():
        d = json.load(open(p))
        for case in d:
            if not isinstance(d[case], dict) or "trace" not in d[case]:
                continue
            tr = d[case]["trace"]
            out["cg_cases"][case] = {
                "case": case, "r_in": d[case].get("r_in"),
                "ratio_t1": tr[0].get("ratio"), "step_t1": tr[0].get("step"),
                "ratio_final": tr[-1].get("ratio"), "step_final": tr[-1].get("step"),
                "R_eq_t1": tr[0].get("R_eq"), "R_eq_final": tr[-1].get("R_eq"),
                "max_u_t1": tr[0].get("max_u"), "max_u_final": tr[-1].get("max_u"),
            }

    p = paths["p0b_envelope"]
    if p.exists():
        d = json.load(open(p))
        rows = []
        for case in d:
            hist = case.get("hist") or []
            if not hist:
                continue
            last = hist[-1]
            rows.append({"G": case.get("G"), "c1": case.get("c1"), "c2": case.get("c2"),
                         "maxu_last": last.get("maxu"), "ratio_last": last.get("ratio"),
                         "Req_last": last.get("Req"), "nan": case.get("nan")})
        out["scmcmp_envelope"] = {"n_cases": len(rows), "rows": rows}

    p = paths["p1_ac"]
    if p.exists():
        out["ac_probe_path"] = str(p.relative_to(REPO))
    return out


def sc_spurious_number() -> dict:
    """S = u_sp mu / sigma for the one SC-family run reporting BOTH u_max and sigma.

    Source row `wb_psi2` of docs/3D_MCMP_SPURIOUS_SCALING_STATUS.md
    (n=32, G=-5.0, psi2 force, EDM forcing).  tau = 1 -> nu = c_s^2(tau-1/2) = 1/6,
    and the run sits at rho ~ 1, so mu ~ 1/6.  Values are parsed from the doc text so
    nothing is retyped.
    """
    doc = REPO / "docs/3D_MCMP_SPURIOUS_SCALING_STATUS.md"
    if not doc.exists():
        return {"status": "doc_missing"}
    txt = doc.read_text(encoding="utf-8", errors="replace")
    txt = txt.replace("\u2212", "-")  # the doc uses the Unicode minus sign
    row = None
    for line in txt.splitlines():
        if "wb_psi2" in line and "1.52" in line:
            row = line
            break
    if row is None:
        return {"status": "row_not_found"}
    # cell order in that table row: | case | n | G | drho | form | u_max | note |
    # and inside the note: "(dp = ..., sigma = ...)".  Parse by label where the label
    # exists, else take the first exponent-form number of the row.
    def _lab(pattern):
        m = re.search(pattern, row)
        return float(m.group(1)) if m else None

    dp = _lab(r"dp\s*=\s*([-+]?\d\.\d+e[-+]\d+)")
    sigma = _lab(r"[\u03c3s]{1,3}\s*=\s*([-+]?\d\.\d+e[-+]\d+)\)")
    nums = [float(x) for x in re.findall(r"[-+]?\d\.\d+e[-+]\d+", row)]
    umax = nums[0]
    assert sigma is not None and dp is not None and umax > 0, row
    mu = 1.0 / 6.0
    return {"source": str(doc.relative_to(REPO)), "n_cells": 32, "G": -5.0,
            "u_max": umax, "sigma": sigma, "dp": dp, "mu_assumed": mu,
            "S_measured": umax * mu / sigma,
            "sigma_over_mu": sigma / mu}


# --------------------------------------------------------------------------------------
# 6. Figures
# --------------------------------------------------------------------------------------
def make_figures(budget: dict) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGS.mkdir(parents=True, exist_ok=True)
    made = {}

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    Rs = np.linspace(4, 40, 200)
    for s, style in zip((1e-3, 1e-2, 5e-2), ("-", "--", "-.")):
        ax.loglog(Rs, cg_ceiling(s, Rs), style, label="sigma = %g" % s)
    ax.axhline(0.5, color="k", lw=1, ls=":")
    ax.text(4.4, 0.56, "HR buoyant drop needs dRho/Rho ~ 0.5", fontsize=8)
    ax.set_xlabel("drop radius R [lattice]")
    ax.set_ylabel("sustainable dRho/Rho from p = rho cs^2")
    ax.set_title("Colour-gradient acoustic ceiling (3D): dRho_max = 2 sigma/(R cs^2)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    p = FIGS / "fig1_cg_density_ceiling.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made["fig1"] = str(p.relative_to(REPO))

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    lams = np.logspace(-2, 2, 200)
    for bo in (0.001, 0.005, 0.01, 0.05):
        ax.loglog(lams, 0.1 * bo * (1 + lams) / (6 * (2 + 3 * lams)), "-",
                  label="Bo=%g (10%% budget)" % bo)
    sm = budget["sc_spurious"]
    if "S_measured" in sm:
        ax.axhline(sm["S_measured"], color="crimson", lw=1.6,
                   label="SC measured S = %.2f" % sm["S_measured"])
    ax.axhspan(1e-6, 1e-4, color="green", alpha=0.10)
    ax.text(0.011, 2.4e-5, "phase-field target band", color="green", fontsize=8)
    ax.set_xlabel("viscosity ratio lambda = mu_d/mu_c")
    ax.set_ylabel("S = u_sp mu_o / sigma_o")
    ax.set_title("Spurious budget: S < 0.1 Ca,  Ca = Bo(1+lam)/(6(2+3lam))")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="lower left")
    p = FIGS / "fig2_spurious_budget.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made["fig2"] = str(p.relative_to(REPO))

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    sw = [r for r in budget["sc"]["sweep"] if "ratio" in r]
    if sw:
        ax.semilogy([r["T_r"] for r in sw], [r["ratio"] for r in sw], "o-",
                    label="CS-EOS coexistence rho_l/rho_g")
    ax.axhline(2.0, color="k", ls=":", lw=1)
    ax.text(0.62, 2.2, "ratio needed for dRho/Rho_o = 1", fontsize=8)
    ax.set_xlabel("reduced temperature T_r")
    ax.set_ylabel("rho_l / rho_g")
    ax.set_title("Shan-Chen: the coexistence ratio is an EOS property, not a G knob")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    p = FIGS / "fig3_sc_coexistence.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made["fig3"] = str(p.relative_to(REPO))

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    bos = np.logspace(-2.5, 0.5, 200)
    for lam in (0.1, 1.0, 10.0):
        ax.loglog(bos, [hr_identity(b, lam) for b in bos], "-", label="Ca at lambda=%g" % lam)
    ax.axhline(0.1, color="k", ls="--", lw=1)
    ax.text(0.004, 0.13, "deformation limit Ca ~ 0.1", fontsize=8)
    ax.axvline(0.01, color="k", ls=":", lw=1)
    ax.text(0.0115, 4e-4, "spherical limit Bo < 0.01", fontsize=8, rotation=90)
    ax.set_xlabel("Bond number Bo")
    ax.set_ylabel("Capillary number Ca")
    ax.set_title("HR validity window is one line: Ca = Bo(1+lam)/(6(2+3lam))")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    p = FIGS / "fig4_hr_window.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made["fig4"] = str(p.relative_to(REPO))

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    R0 = np.linspace(2, 20, 200)
    for ms in (0.02, 0.05, 0.16, 0.5):
        ax.plot(R0, [ac_collapse(r, ms) for r in R0], "-", label="M sigma = %g" % ms)
    ax.axhline(200, color="crimson", ls=":", lw=1)
    ax.text(2.2, 240, "observed dissolution by step ~200 (repo probe)", fontsize=8,
            color="crimson")
    ax.set_yscale("log")
    ax.set_xlabel("initial radius R0 [lattice]")
    ax.set_ylabel("finite collapse time t* = R0^2/(4 M sigma) [steps]")
    ax.set_title("Non-conserved Allen-Cahn (Model A): t* ~ R0^2, refinement cannot save it")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    p = FIGS / "fig5_ac_collapse.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made["fig5"] = str(p.relative_to(REPO))
    return made


# --------------------------------------------------------------------------------------
# 7. Generated markdown appendix (tables only)
# --------------------------------------------------------------------------------------
def fmt(x, nd: int = 4) -> str:
    if x is None:
        return "-"
    if isinstance(x, str):
        return x
    if isinstance(x, bool):
        return "true" if x else "false"
    if x == 0:
        return "0"
    ax = abs(x)
    if ax < 1e-3 or ax >= 1e5:
        return "%.*e" % (nd, x)
    return "%.*f" % (nd, x)


def md_table(rows, cols, header=None) -> str:
    header = header or cols
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |")
    return "\n".join(out)


def write_appendix(budget: dict) -> Path:
    L = []
    L.append("# T1 数值附录（程序生成，请勿手工编辑）\n")
    L.append("生成器：`docs/multiphase/T1/analysis/t1_phase_budget.py`；"
             "机器可读全量：`artifacts/T1_multiphase/t1_phase_budget.json`。\n")
    L.append("文档正文只引用本附录中的数字；本文件由脚本一次写出。\n")

    L.append("\n## A1. Shan–Chen（CS 状态方程）共存与两个锁\n")
    lp = budget["sc"]["legacy_params"]
    L.append("历史参数退化检查（`psi_carnahan_starling` 硬编码 a=0.5, b=4, RT=1/3）：\n")
    L.append(md_table([lp], ["a", "b", "RT", "has_vdw_loop", "usable_a", "usable_b",
                             "usable_RT", "usable_has_vdw_loop", "source"]))
    L.append("\n合法参数（a=5.0）的 Maxwell 共存解：\n")
    c = budget["sc"]["coexistence"]
    L.append(md_table([{"q": "rho_g", "v": c["rho_g"]}, {"q": "rho_l", "v": c["rho_l"]},
                       {"q": "p_sat", "v": c["p_sat"]}, {"q": "rho_l/rho_g", "v": c["ratio"]}],
                      ["q", "v"], ["量（a=5.0, b=4, RT=1/3，Maxwell 等面积）", "值"]))
    L.append("\n约化温度扫描（G 只沿共存曲线移动工作点，不解除上限）：\n")
    L.append(md_table(budget["sc"]["sweep"],
                      ["T_r", "rho_g", "rho_l", "ratio", "drho_over_rhol", "error"]))
    L.append("\n黏度比锁（BGK: μ = ρ c_s²(τ−½)）：\n")
    L.append(md_table(budget["sc"]["viscosity_lock"],
                      ["rho_ratio", "lambda_uniform_tau", "lambda_tau_of_rho"]))

    L.append("\n## A2. Color-Gradient 声学密度对比上限 Δρ_max = (d−1)σ/(R c_s²)\n")
    L.append(md_table(budget["cg"]["ceiling_rows"], ["sigma", "R", "drho_max", "ratio_max"]))
    L.append("\n由实测平台反解有效表面张力 σ_implied = Δρ_plateau·R_eq·c_s²/2"
             "（两例独立塌缩到同一平台，是声学上限读法的自洽性检验）：\n")
    L.append(md_table(budget["cg"].get("sigma_inference") or [],
                      ["case", "ratio_final", "R_eq_final", "drho_plateau", "sigma_implied"]))
    L.append("\n反解：维持给定密度比所需的 σ（σ 在此是被约束量，不是自由旋钮）：\n")
    L.append(md_table(budget["cg"]["required_sigma_rows"],
                      ["R", "target_ratio", "sigma_required"]))

    L.append("\n## A3. Allen–Cahn（非守恒，Model A）有限崩塌时间 t* = R0²/(4 M̃σ)\n")
    L.append(md_table(budget["ac"]["grid"], ["Mtilde_sigma", "R0", "t_collapse"]))

    L.append("\n## A4. HR 无量纲恒等式 Ca = Bo(1+λ)/(6(2+3λ)) 与伪流预算 S < 0.1·Ca\n")
    L.append(md_table(budget["hr"]["rows"],
                      ["Bo", "lambda", "Ca", "S_required_for_10pct", "S_required_for_3pct"]))

    L.append("\n## A5. 候选 HR 格点参数设计（由 (Re, Bo) 反解 g 与 σ）\n")
    L.append(md_table(budget["hr_cases"],
                      ["R", "rho_ratio", "lambda", "nu_o", "nu_i", "tau_o", "tau_i",
                       "Re_target", "Bo_target", "U_HR", "g", "sigma", "Ca",
                       "S_budget_10pct", "u_sp_budget_10pct", "steps_to_terminal_est",
                       "gravity_head_over_cs2"]))

    L.append("\n## A6. 仓库既有工件归约（只读，不重打）\n")
    L.append("来源路径：\n")
    L.append(md_table([{"k": k, "v": v} for k, v in budget["measurements"]["provenance"].items()],
                      ["k", "v"]))
    L.append("\nColor-Gradient 总密度对比塌缩（`probe/p0_static.json` 逐例）：\n")
    rows = list(budget["measurements"]["cg_cases"].values())
    if rows:
        L.append(md_table(rows, ["case", "r_in", "ratio_t1", "step_t1", "ratio_final",
                                 "step_final", "R_eq_t1", "R_eq_final", "max_u_t1",
                                 "max_u_final"]))
    L.append("\nSC-MCMP 包络（`probe/p0b_envelope.json`，末帧）：\n")
    env = budget["measurements"]["scmcmp_envelope"]
    if env:
        L.append(md_table(env["rows"], ["G", "c1", "c2", "maxu_last", "ratio_last",
                                        "Req_last", "nan"]))

    L.append("\n## A7. 伪流数 S = u_sp μ/σ（仓库实测归约）\n")
    sm = budget["sc_spurious"]
    if "S_measured" in sm:
        L.append(md_table([sm], ["source", "n_cells", "G", "u_max", "sigma", "dp",
                                 "mu_assumed", "S_measured"]))

    L.append("\n## A8. 图\n")
    L.append(md_table([{"name": k, "path": v} for k, v in budget["figures"].items()],
                      ["name", "path"]))
    L.append("")
    p = DOCS / "T1_NUMERIC_APPENDIX.md"
    p.write_text("\n".join(L), encoding="utf-8")
    return p


def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)

    budget = {
        "meta": {
            "anchor": "GAP-LBM-T1-MULTIPHASE",
            "round": 1,
            "kind": "method research only (no solver code changed)",
            "cs2": CS2,
            "units": "lattice units, dx = dt = 1, rho_o = 1",
        },
        "sc": sc_section(),
        "cg": cg_section(),
        "ac": ac_section(),
        "hr": hr_section(),
        "hr_cases": [
            hr_case_design(R=10.0, ratio_rho=2.0, lam=1.0, nu_o=0.1, Re_t=0.05, Bo_t=0.005),
            hr_case_design(R=12.0, ratio_rho=2.0, lam=1.0, nu_o=0.1, Re_t=0.05, Bo_t=0.005),
            hr_case_design(R=12.0, ratio_rho=2.0, lam=5.0, nu_o=0.1, Re_t=0.05, Bo_t=0.005),
            hr_case_design(R=16.0, ratio_rho=3.0, lam=0.1, nu_o=0.1, Re_t=0.05, Bo_t=0.005),
            hr_case_design(R=12.0, ratio_rho=2.0, lam=1.0, nu_o=0.1, Re_t=0.05, Bo_t=0.05),
        ],
        "measurements": load_measurements(),
        "sc_spurious": sc_spurious_number(),
    }
    budget["cg"]["sigma_inference"] = cg_sigma_inference(
        budget["measurements"]["cg_cases"])
    budget["figures"] = make_figures(budget)
    app = write_appendix(budget)
    out = ART / "t1_phase_budget.json"
    out.write_text(json.dumps(budget, indent=2, default=str), encoding="utf-8")
    print("wrote", out)
    print("wrote", app)
    for k, v in budget["figures"].items():
        print("fig", k, v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())