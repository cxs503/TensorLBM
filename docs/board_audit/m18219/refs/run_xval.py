#!/usr/bin/env python3
"""XVAL driver — TensorLBM MCMP cross-force operator vs the live OSS lbmpy MCMP arm.

Anchor: XVAL-LBM-MCMP-CROSS-FORCE-OPERATOR (round 1, open-source cross-benchmark).
Criterion (owner): reference error < 15 %.

Arms (all numbers land in ``artifacts/summary.json``; the report is rendered from it):

  A  operator identity   : repo ``sc_two_component_force`` (G_12) vs the OSS D2Q9
                           stencil force transcribed from the lbmpy sdist source
                           (``tests/shan_chen/test_shan_chen_two_component.py:55-59,73-81``),
                           same pseudopotential (psi = rho), convention g_oss = -G_12.
  B  matched model       : repo collision/streaming kernels + OSS pseudopotential
                           (psi = 1-exp(-rho)) vs the executed OSS run, same case.
  C  as-shipped          : repo ``collide_sc_two_component`` (psi = rho) vs OSS psi = exp.
  D  resolution/g sweep  : N = 64/128 and g = 5/6 -> convergence + sensitivity.
  E  faithfulness        : (i) OSS arm reproduces the upstream paper profile;
                           (ii) the driver's replica collision reproduces the library
                           ``collide_sc_two_component`` bit-for-bit with psi = rho.

Run:  LBMPY_SRC=<lbmpy sdist>/src python run_xval.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

import tensorlbm
from tensorlbm.d2q9 import C, W, equilibrium, macroscopic  # noqa: E402


def stream(f):
    """Periodic pull streaming, D2Q9 (repo cell order: dim 0 = y, dim 1 = x)."""
    out = torch.empty_like(f)
    for i in range(9):
        cx, cy = int(C[i, 0]), int(C[i, 1])
        out[i] = torch.roll(f[i], shifts=(-cy, -cx), dims=(0, 1))
    return out
from tensorlbm.multiphase import (  # noqa: E402
    _sc_neighbor_weighted_sum, collide_sc_two_component, guo_force_term,
    sc_two_component_force)

torch.set_grad_enabled(False)
DT = torch.float64

# ---------------------------------------------------------------------------
# reference: OSS stencil force, transcribed from the lbmpy sdist source
# ---------------------------------------------------------------------------


def psi_oss(rho: torch.Tensor) -> torch.Tensor:
    """lbmpy sdist ``tests/shan_chen/test_shan_chen_two_component.py:53-54`` (rho0 = 1)."""
    return 1.0 - torch.exp(-rho)


def psi_lin(rho: torch.Tensor) -> torch.Tensor:
    """Palabos ``PsiIsRho`` == TensorLBM MCMP convention."""
    return rho


def oss_force(rho_a, rho_b, g_ab, psi=psi_lin):
    """OSS forward-gather cross force.

    lbmpy sdist ``:55-59``/``:73-81``:
        F_a = -sum_b g_ab psi(rho_a) * sum_d w_d psi(rho_b[+d]) d
    with ``[+d]`` the *forward* stencil neighbour (pystencils field offset ``rho[d]``).
    Returns (Fx_a, Fy_a, Fx_b, Fy_b) with the repo's (y, x) component order.
    """
    c = C.to(torch.long)
    w = W.to(torch.float64)
    ny, nx = rho_a.shape

    def fwd(field):
        fx = torch.zeros_like(field)
        fy = torch.zeros_like(field)
        for i in range(9):
            cx, cy = int(c[i, 0]), int(c[i, 1])
            if cx == 0 and cy == 0:
                continue
            fx = fx + w[i] * cx * torch.roll(field, shifts=(-cy, -cx), dims=(0, 1))
            fy = fy + w[i] * cy * torch.roll(field, shifts=(-cy, -cx), dims=(0, 1))
        return fx, fy

    pa, pb = psi(rho_a), psi(rho_b)
    sa_x, sa_y = fwd(pa)
    sb_x, sb_y = fwd(pb)
    # g_ab multiplies the *other* component's sum (g_aa = g_bb = 0)
    return (-g_ab * pa * sb_x, -g_ab * pa * sb_y,
            -g_ab * pb * sa_x, -g_ab * pb * sa_y)


# ---------------------------------------------------------------------------
# repo arms
# ---------------------------------------------------------------------------


def tl_forces(rho1, rho2, G_12, psi_mode):
    """Cross force. 'linear' -> the library kernel as shipped; 'exp' -> same gather
    operator (library helper ``_sc_neighbor_weighted_sum``) on the OSS pseudopotential."""
    if psi_mode == "linear":
        Fx1, Fy1, Fx2, Fy2 = sc_two_component_force(rho1, rho2, G_12)
        return Fx1, Fy1, Fx2, Fy2
    p1, p2 = psi_oss(rho1), psi_oss(rho2)
    sx2, sy2 = _sc_neighbor_weighted_sum(p2)   # reverse gather, repo convention
    sx1, sy1 = _sc_neighbor_weighted_sum(p1)
    return (-G_12 * p1 * sx2, -G_12 * p1 * sy2, -G_12 * p2 * sx1, -G_12 * p2 * sy1)


def collide_replica(f1, f2, Fx1, Fy1, Fx2, Fy2, tau=1.0):
    """Replica of ``collide_sc_two_component(force_scheme='guo', u_eq='mixture')``
    with externally supplied forces (needed for the psi = exp arm)."""
    rho1, ux1, uy1 = macroscopic(f1)
    rho2, ux2, uy2 = macroscopic(f2)
    r1 = torch.clamp(rho1, min=1e-12)
    r2 = torch.clamp(rho2, min=1e-12)
    ux1, uy1 = ux1 + 0.5 * Fx1 / r1, uy1 + 0.5 * Fy1 / r1
    ux2, uy2 = ux2 + 0.5 * Fx2 / r2, uy2 + 0.5 * Fy2 / r2
    rt = r1 + r2
    mx = (r1 * ux1 + r2 * ux2) / rt
    my = (r1 * uy1 + r2 * uy2) / rt
    feq1 = equilibrium(rho1, mx, my)
    feq2 = equilibrium(rho2, mx, my)
    out1 = f1 - (f1 - feq1) / tau + guo_force_term(rho1, ux1, uy1, Fx1, Fy1, tau)
    out2 = f2 - (f2 - feq2) / tau + guo_force_term(rho2, ux2, uy2, Fx2, Fy2, tau)
    return out1, out2


def init_fields(N):
    """OSS initial condition (upstream ``:122-126``): two slabs along axis 0."""
    rho1 = torch.zeros(N, N, dtype=DT)
    rho2 = torch.zeros(N, N, dtype=DT)
    rho1[:, : N // 2] = 0.1
    rho1[:, N // 2:] = 0.9
    rho2[:, : N // 2] = 0.9
    rho2[:, N // 2:] = 0.1
    f1 = equilibrium(rho1, torch.zeros_like(rho1), torch.zeros_like(rho1))
    f2 = equilibrium(rho2, torch.zeros_like(rho2), torch.zeros_like(rho2))
    return f1, f2


def tl_run(N, g_ab, psi_mode, steps=1000, tau=1.0, use_library_collision=False):
    f1, f2 = init_fields(N)
    G_12 = -float(g_ab)              # repo: G_12 < 0 == repulsive;  OSS: g_ab > 0 == repulsive
    for _ in range(steps):
        rho1, _, _ = macroscopic(f1)
        rho2, _, _ = macroscopic(f2)
        Fx1, Fy1, Fx2, Fy2 = tl_forces(rho1, rho2, G_12, psi_mode)
        if use_library_collision:
            c1, c2 = collide_sc_two_component(f1, f2, G_12=G_12, tau1=tau, tau2=tau,
                                              force_scheme="guo", u_eq="mixture")
            # the library recomputes the force internally; keep it only for the
            # faithfulness check, where the external force equals the internal one
            _ = (Fx1, Fy1, Fx2, Fy2)
        else:
            c1, c2 = collide_replica(f1, f2, Fx1, Fy1, Fx2, Fy2, tau=tau)
        f1, f2 = stream(c1), stream(c2)
    rho1, ux1, uy1 = macroscopic(f1)
    rho2, ux2, uy2 = macroscopic(f2)
    umix = torch.maximum(torch.sqrt(ux1 ** 2 + uy1 ** 2), torch.sqrt(ux2 ** 2 + uy2 ** 2))
    return dict(prof_a=rho1[0, :].numpy(), prof_b=rho2[0, :].numpy(),
                rho_a=rho1.numpy(), rho_b=rho2.numpy(), umax=float(umix.max()),
                rho_tot_drift=float(abs((rho1 + rho2).mean() - 1.0)))


def rel_l2(got, ref):
    got = np.asarray(got, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    span = ref.max() - ref.min()
    return float(np.linalg.norm(got - ref) / np.linalg.norm(ref - ref.mean())), \
        float(np.max(np.abs(got - ref)) / span)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    os.makedirs(ART, exist_ok=True)
    lbmpy_src = os.environ.get("LBMPY_SRC", "/tmp/lbmpy_src/src")
    cases = [dict(N=64, g=6.0), dict(N=128, g=6.0), dict(N=64, g=5.0)]

    # ---- E(i): run the OSS arm (live lbmpy) -------------------------------
    oss_npz = os.path.join(ART, "oss_cases.npz")
    oss_cases = [dict(N=64, g_ab=6.0, psi_mode="exp"), dict(N=128, g_ab=6.0, psi_mode="exp"),
                 dict(N=64, g_ab=5.0, psi_mode="exp"), dict(N=64, g_ab=6.0, psi_mode="linear")]
    env = dict(os.environ, PYTHONPATH=lbmpy_src)
    cmd = [sys.executable, os.path.join(HERE, "oss_arm.py"), "--out", oss_npz,
           "--cases", json.dumps(oss_cases), "--steps", "1000"]
    print("running OSS arm:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, env=env, cwd=HERE, capture_output=True, text=True)
    oss_log = (proc.stdout or "") + (proc.stderr or "")
    oss_ok = proc.returncode == 0 and os.path.exists(oss_npz)
    summary = dict(anchor="XVAL-LBM-MCMP-CROSS-FORCE-OPERATOR", oss_ok=oss_ok,
                   oss_returncode=proc.returncode, oss_log_tail=oss_log[-2500:],
                   cases=cases, criterion_pct=15.0)
    if not oss_ok:
        print(oss_log[-3000:])
        with open(os.path.join(ART, "summary.json"), "w") as fh:
            json.dump(summary, fh, indent=1)
        return

    oss = np.load(oss_npz)
    with open(os.path.join(ART, "oss_cases_meta.json")) as fh:
        oss_meta = json.load(fh)

    def key(N, g, m, what="prof_a"):
        return "N%d_g%g_%s_%s" % (N, g, m, what)

    # ---- E(ii): driver faithfulness (replica == library, psi = rho) -------
    f1, f2 = init_fields(32)
    rho1, _, _ = macroscopic(f1)
    rho2, _, _ = macroscopic(f2)
    rho2 = rho2 * 0.5 + 0.7 * torch.rand_like(rho2)       # non-trivial field
    f1 = equilibrium(rho1, torch.zeros_like(rho1), torch.zeros_like(rho1))
    f2 = equilibrium(rho2, torch.zeros_like(rho2), torch.zeros_like(rho2))
    F = tl_forces(*(macroscopic(f1)[0], macroscopic(f2)[0]), -6.0, "linear")
    a1, a2 = collide_replica(f1, f2, *F, tau=1.0)
    b1, b2 = collide_sc_two_component(f1, f2, G_12=-6.0, tau1=1.0, tau2=1.0,
                                      force_scheme="guo", u_eq="mixture")
    faith = float(max((a1 - b1).abs().max().item(), (a2 - b2).abs().max().item()))
    summary["driver_faithfulness_maxdiff"] = faith

    # ---- A: operator identity on a planted smooth field -------------------
    N = 64
    yy, xx = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    ra = torch.tensor(0.5 + 0.35 * np.sin(2 * np.pi * xx / N) * np.cos(2 * np.pi * yy / N), dtype=DT)
    rb = torch.tensor(0.5 - 0.30 * np.sin(2 * np.pi * xx / N) * np.cos(2 * np.pi * yy / N), dtype=DT)
    G12 = -6.0
    repo_F = sc_two_component_force(ra, rb, G12)
    oss_F = oss_force(ra, rb, -G12, psi=psi_lin)
    id_err = {}
    for name, got, ref in zip(("Fx1", "Fy1", "Fx2", "Fy2"), repo_F, oss_F):
        id_err[name] = float((got - ref).abs().max().item() / ref.abs().max().item())
    summary["operator_identity"] = {"max_rel": max(id_err.values()), "per_component": id_err,
                                   "threshold": 1e-6,
                                   "note": "float32 weight table in the library -> 1e-6 floor"}
    uniform = sc_two_component_force(torch.full((32, 32), 0.7, dtype=DT),
                                     torch.full((32, 32), 0.4, dtype=DT), G12)
    summary["uniform_zero_force"] = float(max(t.abs().max().item() for t in uniform))

    # ---- B/C: physics arms ------------------------------------------------
    table = []
    for c in cases:
        N, g = c["N"], c["g"]
        for psi_mode in ("exp", "linear"):
            tag = "N%d_g%g_%s" % (N, g, psi_mode)
            tl = tl_run(N, g, psi_mode)
            ref_a = oss[key(N, g, psi_mode)]
            ref_full = oss[key(N, g, psi_mode, "rho_a")]
            e_l2, e_max = rel_l2(tl["prof_a"], ref_a)
            row = dict(N=N, g=g, psi_mode=psi_mode, arm="B_matched" if psi_mode == "exp" else "C_as_shipped",
                       err_rel_l2=e_l2, err_max=e_max,
                       tl_umax=tl["umax"], oss_umax=[m["umax"] for m in oss_meta["cases"]
                                                     if m["N"] == N and m["g_ab"] == g
                                                     and m["psi_mode"] == psi_mode][0],
                       tl_rho_tot_drift=tl["rho_tot_drift"],
                       tl_rho_range=[float(tl["rho_a"].min()), float(tl["rho_a"].max())],
                       oss_rho_range=[float(ref_full.min()), float(ref_full.max())],
                       tl_steps=1000)
            # full-field comparison too (profile only samples one row)
            e_full, _ = rel_l2(tl["rho_a"], ref_full)
            row["err_full_rel_l2"] = e_full
            table.append(row)
            np.savez_compressed(os.path.join(ART, "arm_%s.npz" % tag),
                                prof_a=tl["prof_a"], rho_a=tl["rho_a"], rho_b=tl["rho_b"])
    summary["physics_arms"] = table
    # store every profile for the report figure


    matched = [r for r in table if r["psi_mode"] == "exp"]
    shipped = [r for r in table if r["psi_mode"] == "linear"]
    summary["matched_model_max_err_pct"] = 100 * max(r["err_rel_l2"] for r in matched)
    summary["as_shipped_max_err_pct"] = 100 * max(r["err_rel_l2"] for r in shipped)
    summary["gate_pass_matched"] = summary["matched_model_max_err_pct"] <= 15.0
    conv = {str(r["N"]): r["err_rel_l2"] for r in matched if r["g"] == 6.0}
    if len(conv) == 2:
        n0, n1 = sorted(conv, key=int)
        summary["convergence"] = {"N": [int(n0), int(n1)], "err": [conv[n0], conv[n1]],
                                  "ratio": conv[n1] / conv[n0] if conv[n0] else None}
    sens = {str(r["g"]): r["err_rel_l2"] for r in matched if r["N"] == 64}
    summary["g_sensitivity"] = sens
    summary["oss_reference_faithfulness"] = [m.get("ref_profile_rel") for m in oss_meta["cases"]
                                             if "ref_profile_rel" in m]

    with open(os.path.join(ART, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps({k: summary[k] for k in
                      ("operator_identity", "matched_model_max_err_pct", "as_shipped_max_err_pct",
                       "convergence", "g_sensitivity", "driver_faithfulness_maxdiff",
                       "oss_reference_faithfulness")}, indent=1))


if __name__ == "__main__":
    main()