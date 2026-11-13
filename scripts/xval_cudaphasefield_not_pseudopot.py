#!/usr/bin/env python3
"""XVAL driver: TensorLBM vs AminZar69/CudaPhaseField -- phase field, NOT pseudo-potential.

Anchor: ``XVAL-LBM-NOT-APPLICABLE-PHASE-FIELD-NOT-PSEUDOPOT``
OSS entry: oss_solutions id 155 ("No contribution. The code has no Shan-Chen
pseudo-potential, no interaction strength G, no colour-gradient, no
component-resolved pressure boundary, no Poiseuille/layered-flow setup; its
multiphase physics is a ternary Allen-Cahn phase field with algebraic third
phase.  Mapping it onto the MCMP Poiseuille gap would be a category error.")

The anchor is a *negative* cross-validation: the OSS cannot serve as a reference
for the MCMP pseudo-potential line.  A negative claim still has to be paid for
with numbers, so this driver measures three things:

A. **Semantics identity** -- every OSS constant/formula is re-evaluated from the
   vendored source (``oss_src/``) and compared with the independently derived
   analytic value (lambda combination rule, harmonic coupling, over-relaxation
   factor, 10-90 width of the documented interface).
B. **Re-execution** -- the transplanted ternary conservative Allen-Cahn solver
   (:mod:`tensorlbm.phasefield.ternary_ac_oss`) is run on the interface problem
   and measured against analytic references: equilibrium profile width, tension
   closure ``sigma = lambda/2``, phase-mass closure, and the density-independence
   of the phase equation.  Sensitivity arms: interface thickness ``w`` and
   mobility ``m`` (two knobs, as the mission requires).
C. **Non-isomorphism** -- the pseudo-potential side is exercised as a *model*
   (Shan-Chen mechanical pressure + Maxwell construction): its coexistence
   density ratio is a function of the single interaction strength ``G``, whereas
   the OSS model's phase equation is independent of the mass-density triple.
   Two measured facts, no assertion.

Outputs (all under ``--out``):
  * ``xval_cudaphasefield_not_pseudopot.json`` -- machine-readable results
  * ``xval_cudaphasefield_not_pseudopot.log``  -- the printed table
  * ``fig_interface_profile.png``, ``fig_tension_and_width.png``

Usage
-----
    python scripts/xval_cudaphasefield_not_pseudopot.py --out artifacts/xval/m18180
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.phasefield.ternary_ac_oss import (  # noqa: E402
    OSS_PHASE_MASS_DENSITIES,
    OSS_SHIPPED_SIGMA,
    TernaryAC2D,
    analytic_interface_width,
    flat_interface_state,
    interface_width_10_90,
    lambda_from_sigma_matrix,
    oss_laplacian_is_stencil_identity,
    oss_relaxation_factor,
    rho_of_phases,
    tension_from_lambda,
    tension_from_profile,
)

ANCHOR = "XVAL-LBM-NOT-APPLICABLE-PHASE-FIELD-NOT-PSEUDOPOT"
OWNER_TOL = 0.15  # owner precision band: >15% is unacceptable

OSS_SRC = REPO / "xval" / ANCHOR / "oss_src"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def analytic_profile(nx: int, w: float, x0: float) -> torch.Tensor:
    """The equilibrium profile the OSS documents (single interface, sampled)."""
    x = torch.arange(nx, dtype=torch.float64)
    return 0.5 * (1.0 + torch.tanh(2.0 * (x - x0) / w))


def git_commit(path: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=20, check=False,
        )
        return out.stdout.strip() or None
    except Exception:  # pragma: no cover - provenance is best effort
        return None


def grep_count(pattern: str, files: list[Path]) -> int:
    rx = re.compile(pattern, re.IGNORECASE)
    n = 0
    for f in files:
        n += sum(1 for line in f.read_text(errors="ignore").splitlines()
                 if rx.search(line) and not line.lstrip().startswith("!"))
    return n


def grep_lines(pattern: str, files: list[Path], limit: int = 6) -> list[str]:
    rx = re.compile(pattern, re.IGNORECASE)
    hits: list[str] = []
    for f in files:
        for i, line in enumerate(f.read_text(errors="ignore").splitlines(), 1):
            if rx.search(line) and not line.lstrip().startswith("!"):
                hits.append(f"{f.name}:{i}:{line.strip()[:120]}")
                if len(hits) >= limit:
                    return hits
    return hits


# ---------------------------------------------------------------------------
# C. pseudo-potential side: Shan-Chen coexistence as a function of G only
# ---------------------------------------------------------------------------
def sc_coexistence(G: float, rho0: float = 1.0, cs2: float = 1.0 / 3.0) -> dict:
    """Maxwell construction for the Shan-Chen mechanical pressure.

    ``p(rho) = cs2*rho - (G*cs2/2) psi^2``, ``psi = rho0 (1 - exp(-rho/rho0))``.
    Equal pressure plus equal-area ``integral (p - p_sat) / rho^2 d rho = 0``
    gives the coexistence pair, i.e. the density ratio the pseudo-potential can
    produce at interaction strength ``G``.

    Only ``G`` is varied -- that is the point: the pseudo-potential has no
    interface-thickness, mobility or tension-matrix knob.
    """
    rho_max, n = 12.0 * rho0, 6000
    rho = torch.linspace(1e-6, rho_max, n, dtype=torch.float64)
    psi = rho0 * (1.0 - torch.exp(-rho / rho0))
    p = cs2 * rho - 0.5 * G * cs2 * psi ** 2
    dp = cs2 * (1.0 - G * psi * torch.exp(-rho / rho0))

    # spinodal: sign changes of dp/drho
    sign = torch.sign(dp)
    idx = torch.nonzero(sign[1:] * sign[:-1] < 0).reshape(-1)
    if idx.numel() < 2:
        return {"G": G, "status": "no_spinodal"}
    i_lo, i_hi = int(idx[0].item()), int(idx[-1].item()) + 1
    p_min, p_max = float(p[i_lo].item()), float(p[i_hi].item())

    def outer_roots(p_sat: float) -> tuple[float, float] | None:
        s = torch.sign(p - p_sat)
        j = torch.nonzero(s[1:] * s[:-1] < 0).reshape(-1)
        j = [int(v) for v in j]
        if len(j) < 3:
            return None

        def root(k: int) -> float:
            r0, r1 = float(rho[k].item()), float(rho[k + 1].item())
            p0, p1 = float(p[k].item()), float(p[k + 1].item())
            return r0 + (p_sat - p0) * (r1 - r0) / (p1 - p0)

        return root(j[0]), root(j[-1])

    def area(p_sat: float) -> float | None:
        roots = outer_roots(p_sat)
        if roots is None:
            return None
        rv, rl = roots
        m = (rho >= rv) & (rho <= rl)
        integrand = (p[m] - p_sat) / rho[m] ** 2
        return float(torch.trapezoid(integrand, rho[m]).item())

    span = p_max - p_min
    lo, hi = p_min + 1e-6 * span, p_max - 1e-6 * span
    a_lo, a_hi = area(lo), area(hi)
    if a_lo is None or a_hi is None:
        return {"G": G, "status": "no_area_bracket"}
    if a_lo * a_hi > 0:
        return {"G": G, "status": "no_sign_change", "area_lo": a_lo, "area_hi": a_hi}
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        a_mid = area(mid)
        if a_mid is None:
            return {"G": G, "status": "lost_bracket"}
        if a_lo * a_mid <= 0:
            hi, a_hi = mid, a_mid
        else:
            lo, a_lo = mid, a_mid
        if hi - lo < 1e-12 * max(1.0, abs(mid)):
            break
    p_sat = 0.5 * (lo + hi)
    rv, rl = outer_roots(p_sat)  # type: ignore[misc]
    return {
        "G": G,
        "status": "ok",
        "rho_v": rv,
        "rho_l": rl,
        "density_ratio": rl / rv,
        "p_sat": p_sat,
        "area_residual": area(p_sat),
        "spinodal_rho": [float(rho[i_lo].item()), float(rho[i_hi].item())],
    }


# ---------------------------------------------------------------------------
# experiment driver
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    results: dict = {
        "anchor": ANCHOR,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "owner_tolerance": OWNER_TOL,
        "repo_commit": git_commit(REPO),
        "torch": torch.__version__,
        "dtype": "float64",
        "device": "cpu",
    }

    # ------------------------------------------------------------------
    # provenance / source audit (A6)
    # ------------------------------------------------------------------
    oss_files = sorted(OSS_SRC.glob("*.f90"))
    audit = {
        "vendored_files": {f.name: sha256(f) for f in oss_files},
        "oss_commit": (OSS_SRC / "PROVENANCE.txt").read_text().strip() if (OSS_SRC / "PROVENANCE.txt").exists() else None,
        "pseudopotential_markers": {
            "interaction_strength_G": grep_count(r"\bG\s*=|pseudo", oss_files),
            "psi_effective_mass": grep_count(r"\bpsi\b", oss_files),
            "shan_chen": grep_count(r"shan", oss_files),
            "colour_gradient": grep_count(r"colour|color gradient", oss_files),
            "component_pressure_bc": grep_count(r"pressure_boundary|p_?bc", oss_files),
            "poiseuille": grep_count(r"poiseuille", oss_files),
        },
        "phase_field_markers": {
            "chemical_potential": grep_count(r"chemical_potential", oss_files),
            "mu1_assignments": grep_count(r"mu1\(x,y\)\s*=", oss_files),
            "phase_relaxation_source": grep_count(r"hlph", oss_files),
            "over_relaxation": grep_count(r"w_c", oss_files),
            "lambda_combination": grep_count(r"lambdat", oss_files),
        },
        "pseudopotential_evidence_lines": grep_lines(r"\bG\s*=\s*-|psi\s*=|shan", oss_files, 5),
    }
    results["source_audit"] = audit

    # ------------------------------------------------------------------
    # A. semantics identity (constants and formulas)
    # ------------------------------------------------------------------
    lam, lambdat = lambda_from_sigma_matrix(
        [[0.0, OSS_SHIPPED_SIGMA, OSS_SHIPPED_SIGMA],
         [OSS_SHIPPED_SIGMA, 0.0, OSS_SHIPPED_SIGMA],
         [OSS_SHIPPED_SIGMA, OSS_SHIPPED_SIGMA, 0.0]]
    )
    w_c = oss_relaxation_factor(0.01)
    # a non-degenerate matrix: triangle-inequality-admissible triple
    lam_nd, lambdat_nd = lambda_from_sigma_matrix(
        [[0.0, 0.20, 0.12], [0.20, 0.0, 0.15], [0.12, 0.15, 0.0]]
    )
    # an inadmissible triple (sigma_12 > sigma_31 + sigma_23) -> negative lambda
    lam_bad, _ = lambda_from_sigma_matrix(
        [[0.0, 0.40, 0.10], [0.40, 0.0, 0.12], [0.10, 0.12, 0.0]]
    )
    semantics = {
        "lambda_degenerate_shipped": {
            "lambda": list(lam), "lambdat": lambdat,
            "ref_formula": "lambda_k = s_jk + s_ki - s_ij; all 0.1 -> 0.1; lambdat = harmonic -> 0.1",
            "abs_err": max(abs(v - 0.1) for v in lam) + abs(lambdat - 0.1),
        },
        "lambda_nondegenerate": {
            "sigma": {"s12": 0.20, "s31": 0.12, "s23": 0.15},
            "lambda": list(lam_nd), "lambdat": lambdat_nd,
            "ref_lambda": [0.17, 0.23, 0.07],
            "abs_err": max(abs(lam_nd[0] - 0.17), abs(lam_nd[1] - 0.23), abs(lam_nd[2] - 0.07)),
        },
        "lambda_inadmissible": {
            "sigma": {"s12": 0.40, "s31": 0.10, "s23": 0.12},
            "lambda": list(lam_bad),
            "note": "negative lambda for an inadmissible tension triple; the OSS rule is not a "
                    "triangle-inequality projection and ships no phase-diagram guard",
        },
        "over_relaxation": {
            "m": 0.01, "w_c": w_c, "ref": 1.0 / (0.5 + 3 * 0.01),
            "abs_err": abs(w_c - 1.0 / 0.53),
            "amplification_abs": abs(1.0 - w_c),
            "tau_phi": 1.0 / w_c,
            "doc_tau_phi": 0.53,
            "tau_err": abs(1.0 / w_c - 0.53),
        },
        "interface_width_analytic": {
            "w": 4.0,
            "width_10_90": analytic_interface_width(4.0),
            "doc_width": 4.394,
            "rel_err": abs(analytic_interface_width(4.0) - 4.394) / 4.394,
        },
        "laplacian_stencil_identity": {
            "max_abs_diff_vs_repo_iso_laplacian": oss_laplacian_is_stencil_identity(),
            "note": "difference is the float32 rounding of tensorlbm.d2q9.W (sum = 1 + 7.45e-9); "
                    "the transplanted solver uses exact double literals instead",
        },
    }
    results["semantics"] = semantics
    print("[A] semantics: lambda", semantics["lambda_degenerate_shipped"]["lambda"],
          "w_c", round(w_c, 6), "width_ref", round(semantics["interface_width_analytic"]["width_10_90"], 4))

    # ------------------------------------------------------------------
    # B1. interface equilibrium from three initial widths (A1) + mass closure (A3)
    # ------------------------------------------------------------------
    nx, ny, w = 256, 4, 4.0
    steps = args.steps
    profile_arms = []
    converged_fields: dict[float, torch.Tensor] = {}
    for w0 in (2.0, 4.0, 8.0):
        sim = TernaryAC2D(shape=(ny, nx), w=w, m=0.01)
        sim.initialize(flat_interface_state((ny, nx), w=w, initial_width=w0))
        m0 = sim.phase_mass().clone()
        t0 = time.time()
        sim.run(steps)
        field = sim.fields()[0]
        prof = field[ny // 2]
        ref = analytic_profile(nx, w, 0.25 * nx)
        ref_est = interface_width_10_90(ref)  # estimator applied to the analytic profile
        width = interface_width_10_90(prof)
        # amplitude and profile L-inf against the analytic profile, aligned on the crossing
        arm = {
            "initial_width": w0,
            "steps": steps,
            "wall_s": round(time.time() - t0, 2),
            "width_10_90": width,
            "width_10_90_analytic_sampled": ref_est,
            "width_10_90_analytic_continuum": analytic_interface_width(w),
            "width_rel_err_vs_analytic_sampled": (width - ref_est) / ref_est,
            "amplitude_max": float(prof.max().item()),
            "amplitude_min": float(prof.min().item()),
            "mass_drift_abs": float((sim.phase_mass() - m0).abs().max().item()),
            "mass_drift_rel": float(((sim.phase_mass() - m0).abs().max() / m0.abs().max()).item()),
            "sigma_measured": tension_from_profile(prof, w, lam[0], half_width=3.0 * w),
            "phase_sum_max_dev": float((sim.fields().sum(0) - 1.0).abs().max().item()),
        }
        profile_arms.append(arm)
        converged_fields[w0] = prof.clone()
        print(f"[B1] w0={w0}: width={width:.3f} ref_est={ref_est:.3f} "
              f"err={(arm['width_rel_err_vs_analytic_sampled']*100):+.2f}% "
              f"mass_drift={arm['mass_drift_abs']:.2e} sigma={arm['sigma_measured']:.5f} "
              f"({arm['wall_s']}s)")
    results["interface_equilibrium"] = profile_arms

    # ------------------------------------------------------------------
    # B2. tension closure: sigma = lambda/2 (analytic) vs shipped sigma matrix
    # ------------------------------------------------------------------
    prof = converged_fields[4.0]
    sigma_got = tension_from_profile(prof, w, lam[0], half_width=3.0 * w)
    sigma_from_lam = tension_from_lambda(lam[0])
    # independent route: discrete gradient-energy integral of the analytic profile
    sigma_analytic_profile = tension_from_profile(
        analytic_profile(nx, w, 0.25 * nx), w, lam[0], half_width=3.0 * w
    )
    tension = {
        "lambda_shipped": lam[0],
        "lambda_coupling": lambdat,
        "sigma_analytic": sigma_from_lam,
        "sigma_analytic_from_sampled_profile": sigma_analytic_profile,
        "sigma_measured": sigma_got,
        "rel_err_vs_lambda_over_2": (sigma_got - sigma_from_lam) / sigma_from_lam,
        "rel_err_vs_shipped_sigma_matrix": (sigma_got - OSS_SHIPPED_SIGMA) / OSS_SHIPPED_SIGMA,
        "conclusion": "the OSS pair (12/w, (3/4)w*lambda) is the standard CAC free energy with "
                      "lambda = 2 sigma; the shipped sigma(3,3)=0.1 therefore injects a true lattice "
                      "tension of 0.05 and a literal transplant into a sigma-parameterised solver "
                      "would be 100% off",
    }
    print(f"[B2] sigma measured={sigma_got:.5f} lambda/2={sigma_from_lam:.5f} "
          f"shipped={OSS_SHIPPED_SIGMA} -> err vs lambda/2 {tension['rel_err_vs_lambda_over_2']*100:+.2f}%, "
          f"vs shipped {tension['rel_err_vs_shipped_sigma_matrix']*100:+.1f}%")
    results["tension_closure"] = tension

    # ------------------------------------------------------------------
    # B3. sensitivity: interface thickness w and mobility m (two knobs)
    # ------------------------------------------------------------------
    sensitivity = []
    for w_arm in (3.0, 4.0, 6.0):
        sim = TernaryAC2D(shape=(ny, nx), w=w_arm, m=0.01)
        sim.initialize(flat_interface_state((ny, nx), w=w_arm, initial_width=w_arm))
        t0 = time.time()
        sim.run(args.steps // 2)
        p = sim.fields()[0][ny // 2]
        got = interface_width_10_90(p)
        ref_est = interface_width_10_90(analytic_profile(nx, w_arm, 0.25 * nx))
        sensitivity.append({
            "knob": "w", "value": w_arm, "steps": args.steps // 2,
            "arm": "width_preservation_from_analytic_state",
            "width_10_90": got, "analytic_sampled": ref_est,
            "rel_err": (got - ref_est) / ref_est,
            "sigma_measured": tension_from_profile(p, w_arm, lam[0], half_width=3.0 * w_arm),
            "sigma_expected": tension_from_lambda(lam[0]),
            "wall_s": round(time.time() - t0, 2),
        })
        print(f"[B3] w={w_arm}: width={got:.3f} ref={ref_est:.3f} err={sensitivity[-1]['rel_err']*100:+.2f}%")
    for m_arm in (0.01, 0.05):
        sim = TernaryAC2D(shape=(ny, nx), w=w, m=m_arm)
        sim.initialize(flat_interface_state((ny, nx), w=w, initial_width=2.0 * w))
        m0 = sim.phase_mass().clone()
        t0 = time.time()
        sim.run(args.steps // 2)
        p = sim.fields()[0][ny // 2]
        got = interface_width_10_90(p)
        ref_est = interface_width_10_90(analytic_profile(nx, w, 0.25 * nx))
        sensitivity.append({
            "knob": "m", "value": m_arm, "steps": args.steps // 2,
            "tau_phi": 1.0 / oss_relaxation_factor(m_arm),
            "width_10_90": got, "analytic_sampled": ref_est,
            "rel_err": (got - ref_est) / ref_est,
            "mass_drift_abs": float((sim.phase_mass() - m0).abs().max().item()),
            "wall_s": round(time.time() - t0, 2),
        })
        print(f"[B3] m={m_arm}: width={got:.3f} err={sensitivity[-1]['rel_err']*100:+.2f}% "
              f"mass_drift={sensitivity[-1]['mass_drift_abs']:.2e}")
    results["sensitivity"] = sensitivity

    # ------------------------------------------------------------------
    # B4. density independence of the phase equation
    # ------------------------------------------------------------------
    field_a = converged_fields[4.0]
    sim_b = TernaryAC2D(shape=(ny, nx), w=w, m=0.01, density=(1.0, 1.0, 1.0))
    sim_b.initialize(flat_interface_state((ny, nx), w=w, initial_width=4.0))
    sim_b.run(steps)
    field_b = sim_b.fields()[0][ny // 2]
    rho_a = rho_of_phases(TernaryAC2D(shape=(ny, nx), w=w).fields() * 0 + torch.stack(
        [sim_b.fields()[0], sim_b.fields()[1], sim_b.fields()[2]]))
    results["density_independence"] = {
        "density_arm_a": list(OSS_PHASE_MASS_DENSITIES),
        "density_arm_b": [1.0, 1.0, 1.0],
        "max_abs_field_difference": float((field_a - field_b).abs().max().item()),
        "rho_range_arm_a": [float(rho_a.min().item()), float(rho_a.max().item())],
        "note": "the density triple enters only rho = sum_k phi_k rho_k (inertial mass, OSS "
                "global_subroutines.f90:47); the phase equation is bit-identical, so the "
                "triple is NOT a thermodynamic density ratio -- this falsifies any reading of "
                "(1000,500,100) as a 10:1 high-density-ratio capability",
    }
    print("[B4] density arm diff =", results["density_independence"]["max_abs_field_difference"])

    # ------------------------------------------------------------------
    # C. pseudo-potential side: coexistence is a function of G only
    # ------------------------------------------------------------------
    sc = [sc_coexistence(G) for G in (4.5, 5.0, 6.0, 8.0)]
    results["pseudopotential_coexistence"] = sc
    for row in sc:
        if row["status"] == "ok":
            print(f"[C] G={row['G']}: rho_v={row['rho_v']:.4f} rho_l={row['rho_l']:.4f} "
                  f"ratio={row['density_ratio']:.2f} p_sat={row['p_sat']:.5f}")
        else:
            print(f"[C] G={row['G']}: {row['status']}")

    # ------------------------------------------------------------------
    # acceptance
    # ------------------------------------------------------------------
    checks = {
        "C1_lambda_identity": semantics["lambda_degenerate_shipped"]["abs_err"] < 1e-15,
        "C2_lambda_nondegenerate": semantics["lambda_nondegenerate"]["abs_err"] < 1e-15,
        "C3_over_relaxation": semantics["over_relaxation"]["abs_err"] < 1e-15,
        "C4_width_analytic_vs_doc": semantics["interface_width_analytic"]["rel_err"] < OWNER_TOL,
        "C5_width_from_three_initial_states": all(
            abs(a["width_rel_err_vs_analytic_sampled"]) < OWNER_TOL for a in profile_arms
        ),
        "C6_width_initial_state_independence": (
            max(a["width_10_90"] for a in profile_arms) - min(a["width_10_90"] for a in profile_arms)
        ) / (sum(a["width_10_90"] for a in profile_arms) / len(profile_arms)) < OWNER_TOL,
        "C7_mass_closure_machine_precision": all(a["mass_drift_rel"] < 1e-12 for a in profile_arms),
        "C8_tension_lambda_over_2": abs(tension["rel_err_vs_lambda_over_2"]) < OWNER_TOL,
        "C9_sensitivity_w_band": all(
            abs(row["rel_err"]) < OWNER_TOL for row in sensitivity if row["knob"] == "w"
        ),
        "C10_sensitivity_m_band": all(
            abs(row["rel_err"]) < OWNER_TOL for row in sensitivity if row["knob"] == "m"
        ),
        "C11_density_independence_exact": results["density_independence"]["max_abs_field_difference"] < 1e-12,
        "C12_oss_has_no_pseudopotential": (
            audit["pseudopotential_markers"]["interaction_strength_G"] == 0
            and audit["pseudopotential_markers"]["shan_chen"] == 0
            and audit["pseudopotential_markers"]["colour_gradient"] == 0
            and audit["pseudopotential_markers"]["poiseuille"] == 0
        ),
        "C13_oss_has_phase_field": (
            audit["phase_field_markers"]["chemical_potential"] > 0
            and audit["phase_field_markers"]["over_relaxation"] > 0
            and audit["phase_field_markers"]["lambda_combination"] > 0
        ),
        "C14_sc_coexistence_solved": sum(1 for r in sc if r["status"] == "ok") >= 1,
    }
    results["acceptance"] = checks
    results["acceptance_pass"] = all(checks.values())
    results["wall_s_total"] = round(time.time() - t_start, 2)
    results["repro_cmd"] = (
        "python scripts/xval_cudaphasefield_not_pseudopot.py --out artifacts/xval/m18180"
    )
    results["environment"] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cwd": str(REPO),
        "platform": sys.platform,
        "grid": [ny, nx],
        "steps": steps,
    }

    # ------------------------------------------------------------------
    # persist
    # ------------------------------------------------------------------
    json_path = out / "xval_cudaphasefield_not_pseudopot.json"
    json_path.write_text(json.dumps(results, indent=2))
    log_lines = [
        f"anchor={ANCHOR}",
        f"generated={results['generated_utc']} wall_s={results['wall_s_total']}",
        f"acceptance_pass={results['acceptance_pass']}",
    ]
    for k, v in checks.items():
        log_lines.append(f"  {k}: {'PASS' if v else 'FAIL'}")
    log_lines.append("")
    log_lines.append("interface_equilibrium:")
    for a in profile_arms:
        log_lines.append(
            f"  w0={a['initial_width']}: width={a['width_10_90']:.4f} ref={a['width_10_90_analytic_sampled']:.4f} "
            f"err={a['width_rel_err_vs_analytic_sampled']*100:+.2f}% mass_drift={a['mass_drift_abs']:.2e} "
            f"sigma={a['sigma_measured']:.5f}"
        )
    log_lines.append("tension_closure:")
    log_lines.append(
        f"  sigma_measured={tension['sigma_measured']:.6f} lambda/2={tension['sigma_analytic']:.6f} "
        f"shipped_sigma={OSS_SHIPPED_SIGMA} err_vs_lambda/2={tension['rel_err_vs_lambda_over_2']*100:+.2f}% "
        f"err_vs_shipped={tension['rel_err_vs_shipped_sigma_matrix']*100:+.2f}%"
    )
    log_lines.append("pseudopotential_coexistence (G -> density ratio):")
    for row in sc:
        log_lines.append(f"  {row}")
    (out / "xval_cudaphasefield_not_pseudopot.log").write_text("\n".join(log_lines) + "\n")
    print(f"\nwrote {json_path}  pass={results['acceptance_pass']}  wall={results['wall_s_total']}s")

    # ------------------------------------------------------------------
    # figures
    # ------------------------------------------------------------------
    try:
        make_figures(out, results, converged_fields, w, nx, lam[0])
    except Exception as exc:  # pragma: no cover - figures are optional for the verdict
        print(f"[fig] skipped: {exc}")
    return results


def make_figures(out: Path, results: dict, fields: dict, w: float, nx: int, lam0: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = torch.arange(nx).numpy()
    plt.figure(figsize=(7.2, 4.2))
    ref = analytic_profile(nx, w, 0.25 * nx).numpy()
    plt.plot(x, ref, "k-", lw=1.6, label="analytic tanh ($W=4$)")
    for w0, prof in fields.items():
        plt.plot(x, prof.numpy(), "--", lw=1.1, label=f"LBM, init width {w0:g}W")
    plt.xlim(0.1 * nx, 0.4 * nx)
    plt.xlabel("x [lattice units]")
    plt.ylabel(r"$\phi_1$")
    plt.title("Ternary AC interface equilibrium (OSS semantics) vs analytic profile")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "fig_interface_profile.png", dpi=140)
    plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.8))
    ws = [row["value"] for row in results["sensitivity"] if row["knob"] == "w"]
    got = [row["width_10_90"] for row in results["sensitivity"] if row["knob"] == "w"]
    refw = [row["analytic_sampled"] for row in results["sensitivity"] if row["knob"] == "w"]
    ax[0].plot(ws, refw, "k-o", label="analytic (sampled)")
    ax[0].plot(ws, got, "r--s", label="LBM measured")
    ax[0].set_xlabel("interface thickness $w$")
    ax[0].set_ylabel("10-90 width")
    ax[0].set_title("width scaling")
    ax[0].legend(fontsize=8)

    labels = [f"$w$={row['value']:g}" for row in results["sensitivity"] if row["knob"] == "w"] + [
        f"$m$={row['value']:g}" for row in results["sensitivity"] if row["knob"] == "m"
    ]
    errs = [100 * row["rel_err"] for row in results["sensitivity"]]
    ax[1].bar(range(len(errs)), errs, color=["#4477aa"] * 3 + ["#cc6677"] * 2)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].axhline(100 * (-0.15), color="r", ls=":", lw=1)
    ax[1].axhline(100 * 0.15, color="r", ls=":", lw=1)
    ax[1].set_xticks(range(len(errs)), labels, fontsize=8)
    ax[1].set_ylabel("width error vs analytic [%]")
    ax[1].set_title("sensitivity arms (15% band)")
    fig.tight_layout()
    fig.savefig(out / "fig_tension_and_width.png", dpi=140)
    plt.close(fig)
    print(f"[fig] wrote {out/'fig_interface_profile.png'} and {out/'fig_tension_and_width.png'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(REPO / "artifacts" / "xval" / "m18180"))
    ap.add_argument("--steps", type=int, default=900)
    args = ap.parse_args()
    res = run(args)
    return 0 if res["acceptance_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())