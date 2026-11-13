#!/usr/bin/env python3
"""Board audit m17966 of mission m17843 (XVAL-LBM-GRAPH-UNROLL-BPTT-LES-PARAMETER).

Independent re-derivation of the delivered numbers.  Nothing here imports the
delivered *verification machinery* for the primary checks: the D2Q9 + LES
closure chain, the eddy-viscosity probe, the reference solution and the
finite-difference gradient are re-implemented in NumPy from the equations.
The delivered torch implementation is imported only where the claim itself is
"my NumPy chain and the shipped graph chain are the same solver" (equality
check A0) and for reading its stored results (results.json).

Usage:
    PYTHONPATH=<worktree>/src python3 docs/board_audit/m17966/audit_m17843.py \
        --impl /root/tp-agent/work/m17843 --out artifacts/m17966/audit_m17843.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Independent D2Q9 / Smagorinsky implementation (NumPy, no delivered code)
# --------------------------------------------------------------------------- #
C = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]], float)
W = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36], float)
EX = C[:, 0].reshape(9, 1, 1)
EY = C[:, 1].reshape(9, 1, 1)
WV = W.reshape(9, 1, 1)
# upstream EEk table: sum_{n,m} |c_in c_im| per direction
EEK = np.array([sum(abs(int(a) * int(b)) for a in (cx, cy) for b in (cx, cy))
                for cx, cy in C.tolist()], float).reshape(9, 1, 1)


def feq(rho, ux, uy):
    eu = EX * ux + EY * uy
    return WV * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * (ux * ux + uy * uy))


def macro(f):
    rho = f.sum(axis=0)
    return rho, (f * EX).sum(axis=0) / rho, (f * EY).sum(axis=0) / rho


def tau_field(f, fe, rho, tau0, cs, closure="smagorinsky"):
    fn = f - fe
    if closure == "smagorinsky":
        pxx = (fn * EX * EX).sum(axis=0)
        pxy = (fn * EX * EY).sum(axis=0)
        pyy = (fn * EY * EY).sum(axis=0)
        q = np.sqrt(pxx * pxx + 2.0 * pxy * pxy + pyy * pyy + 1e-300)
        c = 18.0 * math.sqrt(2.0)
    else:  # upstream scalar proxy
        q = np.sqrt(2.0 * (fn * fn * EEK).sum(axis=0) + 1e-300)
        c = 6.0
    return 0.5 * (tau0 + np.sqrt(tau0 * tau0 + c * cs * cs * q / rho))


def step(f, tau0=0.6, cs=0.0, closure="smagorinsky"):
    rho, ux, uy = macro(f)
    fe = feq(rho, ux, uy)
    tau = tau_field(f, fe, rho, tau0, cs, closure)
    fp = f - (f - fe) / tau
    return np.stack([np.roll(fp[i], shift=(int(C[i, 0]), int(C[i, 1])), axis=(0, 1))
                     for i in range(9)])


def run(f0, n, tau0, cs, closure="smagorinsky", snaps=()):
    f = np.array(f0, float, copy=True)
    out = {}
    want = set(int(s) for s in snaps)
    for k in range(1, int(n) + 1):
        f = step(f, tau0, cs, closure)
        if k in want:
            _, ux, uy = macro(f)
            out[k] = (ux.copy(), uy.copy())
    return f, out


def tau_for_re(nx, u_ref, re, length=1.0):
    nu_phys = u_ref * length / re
    dx = 2.0 * math.pi / nx
    return 0.5 + 3.0 * (nu_phys / (dx * dx))


def grid(nx):
    x = np.arange(nx) * (2.0 * math.pi / nx)
    return np.broadcast_to(x.reshape(nx, 1), (nx, nx)), np.broadcast_to(x.reshape(1, nx), (nx, nx))


def tgv_ic(nx, u0=0.1):
    X, Y = grid(nx)
    ux, uy = u0 * np.sin(X) * np.cos(Y), -u0 * np.cos(X) * np.sin(Y)
    rho = np.ones_like(ux)
    return feq(rho, ux, uy), ux, uy


FOURIER_MODES = ((1, 1, 0.55, 0.20), (2, -1, 0.42, 1.10), (-1, 2, 0.38, 2.35),
                 (2, 2, 0.30, 3.05), (3, -2, 0.24, 0.75), (-3, 1, 0.20, 4.20))


def fourier_ic(nx, u0=0.1):
    X, Y = grid(nx)
    ux = np.zeros((nx, nx))
    uy = np.zeros((nx, nx))
    for kx, ky, amp, ph0 in FOURIER_MODES:
        ph = amp / math.sqrt(kx * kx + ky * ky) * np.cos(kx * X + ky * Y + ph0)
        ux += ky * ph
        uy -= kx * ph
    rms = math.sqrt(float(np.mean(ux * ux + uy * uy)))
    ux, uy = ux * u0 / rms, uy * u0 / rms
    rho = np.ones_like(ux)
    return feq(rho, ux, uy), ux, uy


def box_filter(field, ratio):
    nx = field.shape[0]
    n = nx // ratio
    return field.reshape(n, ratio, n, ratio).mean(axis=(1, 3))


def rel_l2(ux, uy, rx, ry):
    return math.sqrt(float((((ux - rx) ** 2 + (uy - ry) ** 2).sum())
                           / ((rx * rx + ry * ry).sum())))


def tally(probe, cs, tau0=0.6, amp=0.05, nx=8, ny=32, ky=1, steps=6, closure="smagorinsky"):
    """Realised viscosity of the closure from a decaying shear mode (lattice units)."""
    k = 2.0 * math.pi * ky / ny
    yy = np.arange(ny) * (2.0 * math.pi / ny)

    def decay(c):
        ux = amp * np.sin(ky * yy).reshape(1, ny) * np.ones((nx, 1))
        f = feq(np.ones_like(ux), ux, np.zeros_like(ux))
        a = []
        for _ in range(steps):
            f = step(f, tau0, c, closure)
            _, uxa, _ = macro(f)
            a.append(float((uxa * np.sin(ky * yy).reshape(1, ny)).mean()) * 2.0)
        t = np.arange(1, steps + 1, dtype=float)
        la = np.log(np.maximum(np.array(a), 1e-300))
        slope = float(np.polyfit(t, la, 1)[0])
        return -slope / (k * k)

    nu0, nucs = decay(0.0), decay(cs)
    return {"cs": float(cs), "nu_t_measured": nucs - nu0,
            "nu_t_analytic": cs * cs * amp * k, "closure": closure}


def err_pct(got, want):
    return abs(got - want) / abs(want) * 100.0 if want else float("nan")


# --------------------------------------------------------------------------- #
# Audit steps
# --------------------------------------------------------------------------- #
def a0_equivalence_with_delivered_torch(impl: Path):
    """cs=0 graph chain (shipped code) vs independent NumPy BGK: must agree."""
    import torch
    from tensorlbm.les_graph_unroll.solver2d import macroscopic, unroll
    f0, _, _ = tgv_ic(32, 0.1)
    f = torch.as_tensor(f0, dtype=torch.float64)
    with torch.no_grad():
        ff, _ = unroll(f, 5, tau_for_re(32, 0.1, 500.0), 0.0)
        rho_t, ux_t, uy_t = macroscopic(ff)
        _, _, _ = rho_t, ux_t, uy_t
        f_torch = ff.numpy().copy()
    f_np, _ = run(f0, 5, tau_for_re(32, 0.1, 500.0), 0.0)
    return {"cs0_5step_max_abs_diff": float(np.max(np.abs(f_torch - f_np))),
            "note": "shipped torch chain vs auditor NumPy BGK, 32^2, 5 steps, tau=%.6f"
                    % tau_for_re(32, 0.1, 500.0)}


def a1_closure(delivered: dict):
    rows = []
    for tag, got in delivered["closure_check"].items():
        cs = got["cs"]
        closure = got["closure"]
        mine = tally(None, cs, closure=closure)
        rows.append({
            "tag": tag, "closure": closure, "cs": cs,
            "delivered_err_pct": got["err_pct"],
            "auditor_err_pct": err_pct(mine["nu_t_measured"], mine["nu_t_analytic"]),
            "delivered_nu_t": got["nu_t_measured"],
            "auditor_nu_t": mine["nu_t_measured"],
            "nu_t_analytic": mine["nu_t_analytic"],
            "nu_t_rel_diff_pct": err_pct(mine["nu_t_measured"], got["nu_t_measured"]),
        })
    lit = [r for r in rows if r["closure"] == "smagorinsky" and abs(r["cs"] - 0.17) < 1e-12][0]
    oss = [r for r in rows if r["closure"] == "oss_proxy"][0]
    ratio = lit["auditor_nu_t"] / oss["auditor_nu_t"]
    expected = 18.0 * math.sqrt(2.0) / 6.0
    return {"rows": rows,
            "oss_over_literature_ratio_measured": lit["auditor_nu_t"] / oss["auditor_nu_t"],
            "literature_over_oss_ratio_measured": ratio,
            "expected_prefactor_ratio_18sqrt2_over_6": expected,
            "ratio_err_pct": err_pct(ratio, expected)}


def a2_twin_gradient_and_recovery(delivered: dict, snapshots=(50, 100, 150, 200), n_steps=200):
    """Independent twin problem: same-solver reference at planted Cs, FD gradient."""
    tau0 = tau_for_re(32, 0.1, 500.0)
    out = {}
    for kind, ic in (("tgv", tgv_ic), ("fourier", fourier_ic)):
        truths = (0.17, 0.30) if kind == "tgv" else (0.17,)
        for truth in truths:
            f0, _, _ = ic(32, 0.1)
            _, refs = run(f0, n_steps, tau0, truth, snaps=snapshots)

            def loss(cs):
                _, s = run(f0, n_steps, tau0, float(cs), snaps=snapshots)
                return sum(rel_l2(*s[k], *refs[k]) ** 2 for k in snapshots) / len(snapshots)

            block = delivered["stage1_twin"][kind][f"cs_truth_{truth:g}"]
            grad_rows = []
            for g in block["x1_gradient"]:
                cs, h = float(g["cs"]), 1e-3
                fd = (loss(cs + h) - loss(cs - h)) / (2.0 * h)
                grad_rows.append({
                    "cs": cs,
                    "delivered_autograd": g["autograd"],
                    "delivered_fd": g["finite_difference"],
                    "auditor_fd": fd,
                    "auditor_vs_delivered_autograd_rel_pct": err_pct(fd, g["autograd"]),
                    "delivered_vs_auditor_fd_rel_pct": err_pct(g["finite_difference"], fd),
                })
            golden = block["golden"]["cs"]
            out[f"{kind}_truth{truth}"] = {
                "gradient": grad_rows,
                "delivered_golden_cs": golden,
                "auditor_loss_at_delivered_golden": loss(golden),
                "delivered_bptt_cs": block["bptt"]["cs"],
                "planted_truth": truth,
                "delivered_bptt_recovery_err_pct": err_pct(block["bptt"]["cs"], truth),
            }
    return out


def a3_physical_reference(delivered: dict, n_steps=200, snaps=(50, 100, 150, 200)):
    """Re-derive the Stage-2 error floor with an independent 128^2 reference."""
    tau0 = tau_for_re(32, 0.1, 500.0)
    tau_ref = tau_for_re(128, 0.1, 500.0)
    out = {}
    for kind, ic in (("tgv", tgv_ic), ("fourier", fourier_ic)):
        f0r, _, _ = ic(128, 0.1)
        _, refs = run(f0r, n_steps, tau_ref, 0.0, snaps=snaps)
        refs = {k: (box_filter(v[0], 4), box_filter(v[1], 4)) for k, v in refs.items()}
        f0c, _, _ = ic(32, 0.1)
        arms = {"no_model": 0.0,
                "upstream_0.17": 0.17,
                "bptt": delivered["stage2_physical"][kind]["bptt"]["cs"],
                "golden": delivered["stage2_physical"][kind]["golden"]["cs"]}
        row = {"tau_ref": tau_ref}
        for name, cs in arms.items():
            _, s = run(f0c, n_steps, tau0, cs, snaps=snaps)
            row[name] = 100.0 * rel_l2(*s[snaps[-1]], *refs[snaps[-1]])
            row[name + "_delivered"] = delivered["stage2_physical"][kind].get(
                "terminal_field_err_pct", {}).get(name)
        out[kind] = row
    return out


def a4_claim_vs_results(predeclared_doc_claims: dict):
    """Every number quoted in the delivery's own §6 verdict table."""
    out = {}
    for k, (got, ref) in predeclared_doc_claims.items():
        out[k] = {"got": got, "ref": ref, "err_pct": err_pct(got, ref)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default="/root/tp-agent/work/m17843")
    ap.add_argument("--out", default="artifacts/m17966/audit_m17843.json")
    args = ap.parse_args()
    impl = Path(args.impl)
    res_path = impl / "artifacts/xval_graph_unroll_bptt_les/results.json"
    delivered = json.loads(res_path.read_text())

    report = {
        "audit_mission": 17966,
        "audited_mission": 17843,
        "anchor": delivered["anchor"],
        "delivered_results_sha256": None,
        "delivered_git_head": delivered["environment"].get("git_head"),
    }
    import hashlib
    report["delivered_results_sha256"] = hashlib.sha256(res_path.read_bytes()).hexdigest()

    report["A0_equivalence"] = a0_equivalence_with_delivered_torch(impl)
    report["A1_closure"] = a1_closure(delivered)
    report["A2_twin"] = a2_twin_gradient_and_recovery(delivered)
    report["A3_physical"] = a3_physical_reference(delivered)

    # doc §6 claims, restated as (got, reference) pairs for independent re-arithmetic
    d = delivered
    tgv = d["stage2_physical"]["tgv"]
    fou = d["stage2_physical"]["fourier"]
    report["A4_doc_claim_arithmetic"] = a4_claim_vs_results({
        "X2_phys_tgv_bptt_vs_golden": (tgv["bptt"]["cs"], tgv["golden"]["cs"]),
        "X2_phys_fourier_bptt_vs_golden": (fou["bptt"]["cs"], fou["golden"]["cs"]),
        "X3_twin_tgv017": (d["stage1_twin"]["tgv"]["cs_truth_0.17"]["bptt"]["cs"], 0.17),
        "X3_twin_tgv030": (d["stage1_twin"]["tgv"]["cs_truth_0.3"]["bptt"]["cs"], 0.30),
        "X3_twin_fourier017": (d["stage1_twin"]["fourier"]["cs_truth_0.17"]["bptt"]["cs"], 0.17),
        "identifiability_loss_gain": (
            (d["stage2_physical"]["tgv"]["terminal_field_err_pct"]["no_model"]
             - d["stage2_physical"]["tgv"]["terminal_field_err_pct"]["bptt"]),
            d["stage2_physical"]["tgv"]["terminal_field_err_pct"]["no_model"]),
    })
    report["A5_convergence"] = {
        "tgv_ke_128_to_256_pct": d["sensitivity"]["tgv"]["reference_convergence"]["ke_rel_err_pct_at_T"],
        "fourier_ke_128_to_256_pct": d["sensitivity"]["fourier"]["reference_convergence"]["ke_rel_err_pct_at_T"],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps(report, indent=2, default=float)[:4000])
    print(f"\n[audit] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())