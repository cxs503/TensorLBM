#!/usr/bin/env python
"""W11-B Phase 1 production runner — creeping-flow single-sphere periodic-domain drag.

Executes the frozen prereg (phase0/prereg.md, md5 pinned below) incl. Amendment A1
and Amendment A2 (附录 D: U-init feq(u=U_sup_target) + per-tier A2 steps table):

  * 12 main runs: phi in {0.005, 0.01, 0.02} x R in {8, 12, 16, 24}, tau = 1.0
  * 2 tau-diagnostics: tau in {0.8, 1.2} at (phi = 0.01, R = 16), no gate
  * D3Q19 BGK, fp64, fully tri-periodic box, single fixed analytic sphere,
    BFL interpolated bounce-back with exact ray-sphere q (sparse kernels)
  * Guo (2002) body force, proper collision velocity
        u* = u_raw + F/(2 rho)  with F = rho * g_x
        f' = feq(u*) + (1 - 1/tau) (f - feq(u*)) + (1 - 1/(2 tau)) df_g(u*)
    (naive collide-at-raw-u halves the force at tau = 1 and is NOT used)
  * primary estimator  F_ledger = g_x * N_fluid (body-force ledger, exact)
  * cross estimator    F_ME     = bfl_force_ledger_sparse (laboratory frame),
    computed BEFORE bfl_bounce_back_sparse, both on (f, f_pre_stream)
  * K_sim = mean over tail-20% samples of K(t) = F_ledger / (6 pi mu a U_sup(t));
    U_sup = whole-cell mean of u_hydro,x with solid nodes u = 0

grep rule honored: collide/stream/equilibrium/bounce-back kernels are library
calls only. The one-line BGK relaxation is composed from library
macroscopic3d/equilibrium3d because no library collider relaxes toward
feq(u*) (the Guo-correct collision state); this is algebra, not a kernel.

All run parameters are loaded from phase0/ref_table.json (machine chain,
zero hand-copying). Judgment numbers are machine-written to result JSONs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

# In-repo copy: resolve the repository root relative to this file and import
# the bundled tensorlbm (the formal campaign pinned its frozen worktree; that
# pin's provenance is recorded in result.json:provenance).
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
WORKTREE_SRC = os.path.join(_REPO_ROOT, "src")
if WORKTREE_SRC not in sys.path:
    sys.path.insert(0, WORKTREE_SRC)

import torch  # noqa: E402

import tensorlbm  # noqa: E402
from tensorlbm.bfl_common import (  # noqa: E402
    bfl_bounce_back_sparse,
    bfl_boundary_link_indices,
    bfl_force_ledger_sparse,
    compute_q_sphere_common,
)
from tensorlbm.boundaries3d import bounce_back_cells_3d  # noqa: E402
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d  # noqa: E402
from tensorlbm.free_surface_common import guo_force_delta_3d  # noqa: E402
from tensorlbm.solver3d import stream3d_roll  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
PHASE0 = _HERE  # reference_table.json ships beside this runner (md5-locked)
PHASE1 = os.environ.get("STOKES_SPHERE_DG_OUT", "results_stokes_sphere_dg")
REF_TABLE_PATH = os.path.join(PHASE0, "reference_table.json")
PREREG_PATH = os.path.join(PHASE0, "prereg.md")
MD5_REF_TABLE = "af32757f9398e3f85b2bd0022c7a8efd"
MD5_PREREG = "db0ffc4342ecc1db42c73a66f38cb925"  # post-附录D (Amendment A2) chain

STEPS_BY_R = {
    8: 20000,
    12: 40000,
    16: 70000,
    24: 150000,
}  # prereg par.3 (superseded by A2, kept for record)
# Amendment A2 (附录 D): steps = ceil(6.5 * tau_spindown_exact * tau_fit_over_pred);
# tau_fit_over_pred = 1.0425957616218076 exact-read from phase1/diagnosis_halt/spindown_fit.json
# doc1. Literals machine-generated from phase0/steps_table_a2.json by phase1/patch_run_a2.py.
STEPS_BY_TIER = {
    (0.005, 8): 79491,
    (0.005, 12): 181606,
    (0.005, 16): 325290,
    (0.005, 24): 726417,
    (0.01, 8): 36421,
    (0.01, 12): 81943,
    (0.01, 16): 145665,
    (0.01, 24): 327738,
    (0.02, 8): 15973,
    (0.02, 12): 34054,
    (0.02, 16): 61350,
    (0.02, 24): 139922,
}
SAMPLE = 100  # monitoring cadence (prereg par.4)
TAU_MAIN = 1.0
TAU_DIAG = (0.8, 1.2)
DIAG_PHI, DIAG_R = 0.01, 16
MA_HALT = 0.025  # prereg par.4 stop line
RE_A_HALT = 0.06  # Amendment A1 C.5 stop line
S0_HALT = 1e-6  # mass drift stop line
S1_LIM = 1e-3
S2_LIM = 1e-3
S3_LIM = 1e-2
ERR_LIM = 0.03  # main gate 3%
EXTENSION_NOTE = (
    "prereg par.4 extension clause suspended by Phase-1 release directive: "
    "S-gate violation = FAIL, no re-run"
)


class CampaignHalt(RuntimeError):
    """Pre-registered stop line tripped — halt the batch, preserve data."""


def md5_of(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_inputs() -> dict:
    rt_md5 = md5_of(REF_TABLE_PATH)
    pg_md5 = md5_of(PREREG_PATH)
    if rt_md5 != MD5_REF_TABLE:
        raise SystemExit(f"ref_table.json md5 mismatch: {rt_md5} != {MD5_REF_TABLE}")
    if pg_md5 != MD5_PREREG:
        raise SystemExit(f"prereg.md md5 mismatch: {pg_md5} != {MD5_PREREG}")
    with open(REF_TABLE_PATH) as fh:
        rows = json.load(fh)
    if len(rows) != 12:
        raise SystemExit(f"ref_table must have 12 rows, got {len(rows)}")
    return {
        "ref_table_md5": rt_md5,
        "prereg_md5": pg_md5,
        "prereg_mtime": time.strftime(
            "%Y-%m-%dT%H:%M:%S%z", time.localtime(os.stat(PREREG_PATH).st_mtime)
        ),
        "rows": rows,
    }


def tail_slice(n_samples: int, frac: float) -> slice:
    k = max(1, int(math.ceil(frac * n_samples)))
    return slice(n_samples - k, n_samples)


def sliding_window_swing(samples: list[float], steps: int, mean_ref: float) -> dict:
    """S1: max relative swing of 5000-step sliding-window means inside the tail.

    Tail length W = min(20000, 20% steps), widened to at least one full
    5000-step window; window = 5000 steps, stride = 1000 steps.
    """
    win = 5000 // SAMPLE
    stride = 1000 // SAMPLE
    W_steps = min(20000, int(0.2 * steps))
    W_eff = max(W_steps, 5000)
    m = int(math.ceil(W_eff / SAMPLE))
    n = len(samples)
    start0 = max(0, n - m)
    if start0 + win > n:  # guarantee at least one window
        start0 = max(0, n - win)
    means = []
    i = start0
    while i + win <= n:
        w = samples[i : i + win]
        means.append(sum(w) / len(w))
        i += stride
    if not means:
        w = samples[start0:]
        means = [sum(w) / len(w)]
    swing = (max(means) - min(means)) / abs(mean_ref)
    return {
        "W_steps_formula": W_steps,
        "W_steps_effective": W_eff,
        "window_steps": win * SAMPLE,
        "stride_steps": stride * SAMPLE,
        "n_windows": len(means),
        "window_means": means,
        "swing_rel": swing,
        "pass": swing <= S1_LIM,
    }


def run_one(
    row: dict, tau: float, steps: int, dev, out_dir: str, meta: dict, smoke: bool = False
) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    L = int(row["L_lat"])
    R = int(row["R"])
    gx = float(row["g_x"])
    K_ref = float(row["K_ref_bie_P5"])
    U_target = float(row["U_sup_target"])
    phi_gate = float(row["phi_gate_exact"])

    # --- geometry (continuous-coordinate sphere centred at box centre) ---
    c = L / 2.0
    zz, yy, xx = torch.meshgrid(
        torch.arange(L, device=dev, dtype=torch.float64),
        torch.arange(L, device=dev, dtype=torch.float64),
        torch.arange(L, device=dev, dtype=torch.float64),
        indexing="ij",
    )
    solid = (xx - c) ** 2 + (yy - c) ** 2 + (zz - c) ** 2 <= R**2
    N_solid = int(solid.sum().item())
    N_total = L**3
    N_fluid = N_total - N_solid
    phi_discrete = N_solid / N_total

    bfl_mask, bfl_q = compute_q_sphere_common(L, L, L, c, c, c, float(R), dev)
    links = bfl_boundary_link_indices(bfl_mask, bfl_q, lattice="D3Q19")
    n_links_dense = int(bfl_mask[1:].sum().item())
    q_min = float(links.link_q.min().item())
    q_max = float(links.link_q.max().item())
    del bfl_mask, bfl_q
    torch.cuda.empty_cache()
    if links.n_wrapped != 0:
        raise CampaignHalt(f"wraparound links detected: {links.n_wrapped} (prereg par.2 R4)")
    if links.n_links != n_links_dense:
        raise CampaignHalt(f"link precompute mismatch {links.n_links} vs {n_links_dense}")

    # --- init: feq(rho=1, u=(U_sup_target,0,0)), fp64 (Amendment A2, 附录 D;
    #     solid interior overwritten by NoDynamics/bounce-back as usual) ---
    ones = torch.ones(L, L, L, dtype=torch.float64, device=dev)
    zeros = torch.zeros_like(ones)
    u0x = torch.full_like(ones, U_target)
    f = equilibrium3d(ones, u0x, zeros, zeros).to(torch.float64)
    del ones, zeros, u0x
    mass0 = float(f.sum().item())

    F_ledger = gx * N_fluid  # exact body-force ledger
    mu = (1.0 / 3.0) * (tau - 0.5)  # lattice units, rho ~ 1
    denom = 6.0 * math.pi * mu * R  # 6 pi mu a, a = R
    om = 1.0 - 1.0 / tau  # BGK coefficient
    half_g = gx / 2.0  # u* = u + F/(2 rho) = u + g/2
    Fy_z = torch.zeros(L, L, L, dtype=torch.float64, device=dev)

    solid_idx = solid.reshape(-1).nonzero(as_tuple=True)[0]
    u_hyd_x = torch.empty_like(solid, dtype=torch.float64)

    series = []
    t0 = time.time()
    status = "completed"
    halt_reason = None
    sample_set = set(range(SAMPLE, steps + 1, SAMPLE))
    flush_set = set(range(5000, steps + 1, 5000))
    t_step_acc = 0.0

    for step in range(1, steps + 1):
        ts = time.time()
        # NoDynamics bookkeeping: gather pre-collision f at solid nodes
        f_solid_pre = f.reshape(19, -1)[:, solid_idx]

        # Guo-correct BGK collision at feq(u*), u* = u + F/(2 rho)
        rho, ux, uy, uz = macroscopic3d(f)
        uxs = ux + half_g
        feqs = equilibrium3d(rho, uxs, uy, uz)
        f = feqs + om * (f - feqs)
        del feqs
        f = f + guo_force_delta_3d(rho * gx, Fy_z, Fy_z, uxs, uy, uz, tau, dev)
        del rho, ux, uy, uz, uxs

        # NoDynamics restore (wipes collision + force inside solid), halfway BB
        f.reshape(19, -1)[:, solid_idx] = f_solid_pre
        del f_solid_pre
        f = bounce_back_cells_3d(f, solid)

        f = f.contiguous()
        f_pre_stream = f.clone()
        f = stream3d_roll(f)  # fully periodic
        f = f.contiguous()
        force3, _ = bfl_force_ledger_sparse(f, f_pre_stream, links)
        f = bfl_bounce_back_sparse(f, f_pre_stream, links)
        del f_pre_stream
        t_step_acc += time.time() - ts

        if step in sample_set:
            rho_m, ux_m, _, _ = macroscopic3d(f)
            # u_hydro = u_raw + F/(2 rho); F = rho g_x -> correction is g_x/2.
            # Solid nodes: NoDynamics keeps u_raw = 0 there, physical u = 0.
            u_hyd_x = torch.where(solid, torch.zeros_like(ux_m), ux_m + half_g)
            U_sup = float(u_hyd_x.mean().item())
            mass = float(f.sum().item())
            F_ME = float(force3[0].item())
            K_t = F_ledger / (denom * U_sup)
            Ma = U_sup * math.sqrt(3.0)
            Re_a = 6.0 * U_sup * R
            finite = bool(torch.isfinite(f).all().item())
            drift = abs(mass / mass0 - 1.0)
            series.append(
                {
                    "step": step,
                    "U_sup": U_sup,
                    "K": K_t,
                    "F_ME": F_ME,
                    "rho_mean": float(rho_m.mean().item()),
                    "mass": mass,
                    "Ma": Ma,
                    "Re_a": Re_a,
                    "mass_drift": drift,
                }
            )
            del rho_m, ux_m
            if not smoke:
                # pre-registered stop lines (campaign-halting)
                if not finite or not math.isfinite(U_sup) or not math.isfinite(F_ME):
                    status, halt_reason = "halted", "non-finite state (NaN/divergence)"
                    break
                if Ma > MA_HALT:
                    status, halt_reason = "halted", f"Ma={Ma:.4f} > {MA_HALT}"
                    break
                if Re_a > RE_A_HALT:
                    status, halt_reason = "halted", f"Re_a={Re_a:.5f} > {RE_A_HALT}"
                    break
                if drift > S0_HALT:
                    status, halt_reason = "halted", f"mass drift {drift:.3e} > {S0_HALT}"
                    break
            if step % 1000 == 0 or step == SAMPLE:
                print(
                    f"  step {step}/{steps} U_sup={U_sup:.6e} K={K_t:.6f} "
                    f"F_ME/F_ledger={F_ME / F_ledger:.6f} drift={drift:.2e}",
                    flush=True,
                )
        if step in flush_set and not smoke:
            with open(os.path.join(out_dir, "series_partial.json"), "w") as fh:
                json.dump(series, fh)

    wall = time.time() - t0
    n_samples = len(series)
    U = [s["U_sup"] for s in series]
    Ks = [s["K"] for s in series]

    # --- verdicts (prereg par.4) ---
    verdict: dict = {}
    if n_samples >= 5:
        t20 = tail_slice(n_samples, 0.2)
        t40 = tail_slice(n_samples, 0.4)
        U20 = sum(U[t20]) / len(U[t20])
        U40 = sum(U[t40]) / len(U[t40])
        K_sim = sum(Ks[t20]) / len(Ks[t20])
        rel = [abs(s["F_ME"] - F_ledger) / F_ledger for s in series[t20]]
        S0_max = max(s["mass_drift"] for s in series)
        s1 = sliding_window_swing(U, steps, U20)
        s2_val = abs(U20 - U40) / abs(U20)
        s3_mean = sum(rel) / len(rel)
        s3_max = max(rel)
        err = K_sim / K_ref - 1.0
        verdict = {
            "S0_max_mass_drift": S0_max,
            "S0_pass": S0_max <= S0_HALT,
            "S1": s1,
            "S2_rel": s2_val,
            "S2_pass": s2_val <= S2_LIM,
            "S3_mean_rel": s3_mean,
            "S3_max_rel": s3_max,
            "S3_pass": s3_mean <= S3_LIM,
            "U_sup_tail20": U20,
            "U_sup_tail40": U40,
            "U_sup_vs_target_rel": U20 / U_target - 1.0,
            "K_sim": K_sim,
            "K_ref": K_ref,
            "err": err,
            "err_pct": err * 100.0,
            "main_gate_pass": abs(err) <= ERR_LIM,
            "Ma_tail20": U20 * math.sqrt(3.0),
            "Re_a_tail20": 6.0 * U20 * R,
            "s_gates_all_pass": bool(
                s1["pass"] and s2_val <= S2_LIM and s3_mean <= S3_LIM and S0_max <= S0_HALT
            ),
            "extension_policy": EXTENSION_NOTE,
        }
    else:
        verdict = {"incomplete": True, "n_samples": n_samples}

    result = {
        "run_id": os.path.basename(out_dir),
        "status": status,
        "halt_reason": halt_reason,
        "smoke": smoke,
        "role": "diagnostic" if tau != TAU_MAIN else "main",
        "tau": tau,
        "steps": steps,
        "sample_interval": SAMPLE,
        "inputs": row,
        "provenance": meta,
        "geometry": {
            "L_lat": L,
            "R": R,
            "centre": [c, c, c],
            "N_total": N_total,
            "N_solid": N_solid,
            "N_fluid": N_fluid,
            "phi_discrete": phi_discrete,
            "phi_gate_exact": phi_gate,
            "n_links": int(links.n_links),
            "n_wrapped": int(links.n_wrapped),
            "q_min": q_min,
            "q_max": q_max,
        },
        "estimators": {
            "F_ledger": F_ledger,
            "mu": mu,
            "a": R,
            "denominator_6pi_mu_a": denom,
        },
        "wall_s": wall,
        "steps_per_s": steps / max(t_step_acc, 1e-9),
        "series_n": n_samples,
        "verdict": verdict,
        "series": series,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t0)),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with open(os.path.join(out_dir, "result.json"), "w") as fh:
        json.dump(result, fh, indent=1)
    del f, solid, xx, yy, zz, Fy_z, links, solid_idx
    try:
        del u_hyd_x
    except NameError:
        pass
    torch.cuda.empty_cache()
    return result


def aggregate(results: list[dict], out_path: str) -> dict:
    main = [
        r
        for r in results
        if r["role"] == "main" and not r["smoke"] and "err" in r.get("verdict", {})
    ]
    by_tier: dict[str, dict[int, dict]] = {}
    for r in main:
        phi = r["inputs"]["phi_nominal"]
        R = int(r["inputs"]["R"])
        by_tier.setdefault(repr(phi), {})[R] = r["verdict"]["err"]
    tiers = {}
    for phi_key, rd in sorted(by_tier.items()):
        seq = [rd[R] for R in sorted(rd)]
        Rs = sorted(rd)
        mono_signed = all(seq[i] > seq[i + 1] for i in range(len(seq) - 1))
        mono_abs = all(abs(seq[i]) > abs(seq[i + 1]) for i in range(len(seq) - 1))
        tiers[phi_key] = {
            "R_ladder": Rs,
            "err_seq": seq,
            "err_seq_pct": [e * 100.0 for e in seq],
            "abs_err_seq_pct": [abs(e) * 100.0 for e in seq],
            "n_points": len(seq),
            "monotone_signed_strict": mono_signed,
            "monotone_abs_strict": mono_abs,
            "finest_abs_err_pct": abs(seq[-1]) * 100.0,
            "finest_le_3pct": abs(seq[-1]) <= ERR_LIM,
            "complete_4_points": len(seq) == 4,
            "convergence_gate_pass": bool(len(seq) == 4 and mono_abs and abs(seq[-1]) <= ERR_LIM),
            "gate_reading_note": (
                "gate read as |err| strictly decreasing (convergence intent); "
                "signed sequence recorded alongside; conflict flagged if different"
            ),
            "monotone_reading_conflict": mono_signed != mono_abs,
        }
    all_pass_main = all(r["verdict"]["main_gate_pass"] for r in main) and len(main) == 12
    all_sgates = all(
        r["verdict"]["s_gates_all_pass"]
        for r in results
        if not r["smoke"] and "s_gates_all_pass" in r.get("verdict", {})
    )
    conv_all = (
        bool(tiers) and all(t["convergence_gate_pass"] for t in tiers.values()) and len(tiers) == 3
    )
    diag = {}
    for r in results:
        if r["role"] == "diagnostic":
            diag[f"tau_{r['tau']}"] = r.get("verdict", {}).get("K_sim")
    base = next(
        (
            r
            for r in results
            if r["role"] == "main"
            and not r["smoke"]
            and float(r["inputs"]["phi_nominal"]) == DIAG_PHI
            and int(r["inputs"]["R"]) == DIAG_R
        ),
        None,
    )
    diag_rel = {}
    if base and "K_sim" in base.get("verdict", {}):
        k0 = base["verdict"]["K_sim"]
        diag_rel = {k: (v / k0 - 1.0) if v is not None else None for k, v in diag.items()}
    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "campaign": "W11-B stokes_sphere_dg remake, Phase 1 (frozen prereg + Amendment A1)",
        "n_runs_completed": len(results),
        "n_main_completed": len(main),
        "halted": any(r["status"] == "halted" for r in results),
        "halt_reasons": [r["halt_reason"] for r in results if r["halt_reason"]],
        "s_gates_all_pass": all_sgates,
        "main_gate_all_12_pass": all_pass_main,
        "per_run": {
            r["run_id"]: {
                "status": r["status"],
                "err": r.get("verdict", {}).get("err"),
                "err_pct": r.get("verdict", {}).get("err_pct"),
                "main_gate_pass": r.get("verdict", {}).get("main_gate_pass"),
                "s_gates_all_pass": r.get("verdict", {}).get("s_gates_all_pass"),
            }
            for r in results
            if not r["smoke"]
        },
        "tiers": tiers,
        "convergence_gate_all_tiers_pass": conv_all,
        "tau_diagnostics_K": diag,
        "tau_diagnostics_rel_vs_tau1": diag_rel,
        "verified": bool(
            all_pass_main
            and conv_all
            and all_sgates
            and not any(r["status"] == "halted" for r in results)
            and len(main) == 12
        ),
    }
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=1)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--smoke", action="store_true", help="single (phi=0.01, R=8) run, 2000 steps, no gates"
    )
    ap.add_argument(
        "--only", default=None, help="comma-separated run keys phi:R[:tau], e.g. 0.01:16:0.8"
    )
    ap.add_argument(
        "--probe",
        default=None,
        metavar="phi:R",
        help="timing probe: run this tier for --probe-steps, no gates",
    )
    ap.add_argument("--probe-steps", type=int, default=200)
    args = ap.parse_args()

    omp = os.environ.get("OMP_NUM_THREADS")
    if omp is not None and int(omp) > 16:
        raise SystemExit(f"OMP_NUM_THREADS={omp} violates release directive (<=16)")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda":
        raise SystemExit("formal runs require CUDA (set CUDA_VISIBLE_DEVICES)")
    inputs = verify_inputs()
    rows = inputs["rows"]
    meta = {
        "ref_table_md5": inputs["ref_table_md5"],
        "prereg_md5": inputs["prereg_md5"],
        "prereg_mtime": inputs["prereg_mtime"],
        "worktree_src": WORKTREE_SRC,
        "tensorlbm_file": tensorlbm.__file__,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "dtype": "float64",
        "OMP_NUM_THREADS": omp,
        "run_py_md5": md5_of(os.path.abspath(__file__)),
    }
    if not str(meta["tensorlbm_file"]).startswith(WORKTREE_SRC):
        raise SystemExit(
            f"tensorlbm imported from {meta['tensorlbm_file']}, "
            f"not this repository's src ({WORKTREE_SRC})"
        )
    print("provenance:", json.dumps({k: v for k, v in meta.items()}, default=str), flush=True)

    results_dir = os.path.join(PHASE1, "results")
    os.makedirs(results_dir, exist_ok=True)
    results: list[dict] = []

    if args.smoke:
        row = next(r for r in rows if r["phi_nominal"] == 0.01 and r["R"] == 8)
        out = os.path.join(PHASE1, "smoke")
        r = run_one(row, TAU_MAIN, 2000, dev, out, meta, smoke=True)
        print("SMOKE:", json.dumps(r["verdict"], indent=1), flush=True)
        return

    if args.probe:
        phi, R = (float(x) for x in args.probe.split(":"))
        row = next(r for r in rows if r["phi_nominal"] == phi and r["R"] == int(R))
        out = os.path.join(PHASE1, f"probe_phi{str(phi).replace('.', 'p')}_R{int(R)}")
        r = run_one(row, TAU_MAIN, args.probe_steps, dev, out, meta, smoke=True)
        print(
            "PROBE:",
            json.dumps(
                {
                    "L": row["L_lat"],
                    "steps": args.probe_steps,
                    "wall_s": r["wall_s"],
                    "steps_per_s": r["steps_per_s"],
                    "peak_mem_GiB": None,
                }
            ),
            flush=True,
        )
        return

    keys = None
    if args.only:
        keys = set(args.only.split(","))
    jobs = []
    for R in (8, 12, 16, 24):  # coarse-to-fine: cheapest validation first
        for phi in (0.005, 0.01, 0.02):
            jobs.append((phi, R, TAU_MAIN))
    for tau in TAU_DIAG:
        jobs.append((DIAG_PHI, DIAG_R, tau))

    for phi, R, tau in jobs:
        key = f"{phi}:{R}" + (f":{tau}" if tau != TAU_MAIN else "")
        if keys is not None and key not in keys:
            continue
        row = next(r for r in rows if r["phi_nominal"] == phi and r["R"] == R)
        steps = STEPS_BY_TIER[(phi, R)]
        run_id = f"phi{phi}_R{R}" + ("_tau1.0" if tau == TAU_MAIN else f"_tau{tau}")
        run_id = run_id.replace(".", "p")
        out_dir = os.path.join(results_dir, run_id)
        print(f"=== RUN {run_id} (tau={tau}, steps={steps}, L={row['L_lat']}) ===", flush=True)
        try:
            res = run_one(row, tau, steps, dev, out_dir, meta)
        except CampaignHalt as exc:
            print(f"=== STRUCTURAL HALT in {run_id}: {exc} ===", flush=True)
            aggregate(results, os.path.join(PHASE1, "results_summary.json"))
            raise
        results.append(res)
        v = res.get("verdict", {})
        print(
            f"=== DONE {run_id}: status={res['status']} "
            f"err={v.get('err_pct', float('nan')):.4f}% "
            f"S-gates={v.get('s_gates_all_pass')} ===",
            flush=True,
        )
        if res["status"] == "halted":
            break
        if "s_gates_all_pass" in v and not v["s_gates_all_pass"]:
            print("S-gate violation -> campaign halted (no extension, no re-run)", flush=True)
            break
        # aggregate incrementally so a later halt never loses the summary
        aggregate(results, os.path.join(PHASE1, "results_summary.json"))

    summary = aggregate(results, os.path.join(PHASE1, "results_summary.json"))
    print(
        "SUMMARY:",
        json.dumps(
            {k: v for k, v in summary.items() if k not in ("per_run", "tiers")}, default=str
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
