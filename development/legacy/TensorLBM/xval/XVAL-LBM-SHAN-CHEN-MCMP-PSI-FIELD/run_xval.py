#!/usr/bin/env python3
"""XVAL-LBM-SHAN-CHEN-MCMP-PSI-FIELD.

Cross-validation of the *psi-field* multicomponent Shan-Chen formulation
(Palabos ``shanChenMultiComponentProcessor3D`` convention: one pseudopotential
field per component, self-interaction excluded, ``G > 0`` repulsive) against

  G1  the exact discrete identities of the force operator   (machine precision)
  G2  the analytic equation of state / acoustic speed       (target <= 15 %)
  G3  the Young-Laplace law for a static drop               (target <= 15 %)
  G4  the production kernel of this repository              (exact algebra)
  G5  ablation arms: common-velocity coupling, self-coupling, psi-field choice

Everything is written to ``artifacts/results.json`` (numbers) and
``artifacts/*.png`` (figures).  Honest FAILs are recorded as-is.

Acceptance: <15% on every quantitative gauge (owner precision standard).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tensorlbm.multiphase import mcmp_psi as M  # noqa: E402

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

OUT = {"meta": {}, "exact": {}, "acoustic": {}, "laplace": {}, "ablations": {}}


# --------------------------------------------------------------------------- #
# G1 + G4: exact identities and comparison with the repository kernel
# --------------------------------------------------------------------------- #
def exact_checks(n: int = 32) -> dict:
    g = torch.Generator().manual_seed(20250101)
    rho_a = 0.4 + 0.6 * torch.rand(n, n, generator=g, dtype=M.DTYPE)
    rho_b = 0.3 + 0.7 * torch.rand(n, n, generator=g, dtype=M.DTYPE)
    uni = torch.full((n, n), 0.7, dtype=M.DTYPE)
    zero = torch.zeros_like(uni)

    res = {}
    F = M.mcmp_psi_force(uni, uni, G=2.5)
    res["uniform_force_max"] = float(max(F[k].abs().max() for k in F))
    F = M.mcmp_psi_force(M.psi_of(rho_a), zero, G=2.5)
    res["single_component_force_max"] = float(max(F[k].abs().max() for k in F))
    F = M.mcmp_psi_force(M.psi_of(rho_a), M.psi_of(rho_b), G=2.5)
    res["momentum_sum"] = [float((F["fx_a"] + F["fx_b"]).sum()), float((F["fy_a"] + F["fy_b"]).sum())]
    sxa, sya = M.neighbor_vec_sum(rho_a)
    sxb, syb = M.neighbor_vec_sum_scatter(rho_a)
    res["gather_vs_scatter"] = [float((sxa - sxb).abs().max()), float((sya - syb).abs().max())]
    F2 = M.mcmp_psi_force(3.0 * M.psi_of(rho_a), M.psi_of(rho_b), G=2.5)
    res["psi_linearity"] = [float((F2["fx_a"] - 3.0 * F["fx_a"]).abs().max()),
                            float((F2["fx_b"] - 3.0 * F["fx_b"]).abs().max())]

    # self-coupling ablation on the same random field
    Fs = M.mcmp_psi_force(M.psi_of(rho_a), M.psi_of(rho_b), G=2.5, self_coupling=2.5)
    res["self_coupling_extra_force_max"] = float(
        max((Fs[k] - F[k]).abs().max() for k in ("fx_a", "fy_a", "fx_b", "fy_b"))
    )

    # G4: repository production kernel (sc_two_component_force, psi = rho)
    try:
        from tensorlbm.multiphase import sc_two_component_force

        fx_a, fy_a, fx_b, fy_b = sc_two_component_force(
            rho_a, rho_b, G=2.5, psi_fn=lambda r: r
        )
        d = [
            float((fx_a - F["fx_a"]).abs().max()),
            float((fy_a - F["fy_a"]).abs().max()),
            float((fx_b - F["fx_b"]).abs().max()),
            float((fy_b - F["fy_b"]).abs().max()),
        ]
        res["production_kernel_max_abs_diff"] = max(d)
        res["production_kernel_diff"] = d
        # the production kernel mirrors psi at the domain edges (wall_mode
        # "mirror") instead of applying periodicity -> compare the interior
        inter = (slice(3, -3), slice(3, -3))
        res["production_kernel_interior_max_abs_diff"] = max(
            float((fx_a - F["fx_a"])[inter].abs().max()),
            float((fy_a - F["fy_a"])[inter].abs().max()),
            float((fx_b - F["fx_b"])[inter].abs().max()),
            float((fy_b - F["fy_b"])[inter].abs().max()),
        )
        res["production_kernel_boundary_max_abs_diff"] = max(d)
        ratio = (fx_a / F["fx_a"])[inter]
        res["production_kernel_interior_ratio"] = {
            "mean": float(ratio.mean()), "std": float(ratio.std())
        }
        res["production_kernel_force_scale"] = float(
            max(F["fx_a"].abs().max(), F["fy_a"].abs().max(), F["fx_b"].abs().max(), F["fy_b"].abs().max())
        )
    except Exception as exc:  # pragma: no cover - defensive
        res["production_kernel_error"] = repr(exc)
    return res


# --------------------------------------------------------------------------- #
# G2: acoustic (total-density) mode speed vs the analytic EOS
# --------------------------------------------------------------------------- #
def acoustic_arms(n_x: int = 128, n_steps: int = 2400) -> dict:
    arms = [
        {"name": "G=0 control", "G": 0.0, "kind": "rho", "rho_ref": 1.0},
        {"name": "G=+1 psi=rho", "G": 1.0, "kind": "rho", "rho_ref": 1.0},
        {"name": "G=-1 psi=rho", "G": -1.0, "kind": "rho", "rho_ref": 1.0},
        {"name": "G=-1 psi=rho-1", "G": -1.0, "kind": "rho_minus_ref", "rho_ref": 1.0},
    ]
    out = []
    for a in arms:
        t0 = time.time()
        r = M.acoustic_mode_speed(
            G=a["G"], rho0=0.2, kind=a["kind"], rho_ref=a["rho_ref"],
            n_x=n_x, n_steps=n_steps,
        )
        r.update({"name": a["name"], "G": a["G"], "psi_kind": a["kind"], "elapsed_s": time.time() - t0})
        r["pass_15pct"] = bool(abs(r["rel_err_pct"]) < 15.0)
        out.append(r)
        print(f"  acoustic {a['name']:18s} c_meas={r['c_measured']:.4f} c_pred={r['c_predicted']:.4f} "
              f"err={r['rel_err_pct']:+.2f}%", flush=True)
    return {"arms": out}


# --------------------------------------------------------------------------- #
# G3: Young-Laplace law for a static drop
# --------------------------------------------------------------------------- #
def laplace_arm(
    radii, n=64, G=2.0, rho_in=0.99, rho_out=0.07, steps=4000, width=2.0,
    velocity_mode="self", self_coupling=0.0, psi_kind="rho", rho_ref=1.0, tag="primary",
) -> dict:
    runs = []
    for R in radii:
        rho_a, rho_b = M.drop_rho_fields(n, R, rho_in, rho_out, width=width)
        cfg = M.MCMPConfig(G=G, psi_kind=psi_kind, rho_ref=rho_ref,
                           velocity_mode=velocity_mode, self_coupling=self_coupling)
        solver = M.MCMPPsiSolver(rho_a, rho_b, cfg)
        t0 = time.time()
        history = []
        finite = True
        for k in range(steps):
            solver.step()
            if (k + 1) % 1000 == 0:
                if not torch.isfinite(solver.f_a).all() or not torch.isfinite(solver.f_b).all():
                    finite = False
                    break
                m = M.measure_drop(solver, rho_in, rho_out)
                history.append({"step": k + 1, "R_eq": m["R_eq"], "dp": m["dp"], "u_max": m["u_max"]})
        elapsed = time.time() - t0
        if not finite:
            runs.append({"R_init": R, "nan": True, "elapsed_s": elapsed})
            print(f"  laplace[{tag}] R={R}: NaN", flush=True)
            continue
        m = M.measure_drop(solver, rho_in, rho_out)
        m.update({"R_init": R, "elapsed_s": elapsed, "history": history, "nan": False})
        m["sigma_per_radius"] = m["dp"] * m["R_eq"]
        runs.append(m)
        print(f"  laplace[{tag}] R_init={R:5.1f} -> R_eq={m['R_eq']:6.2f} dp={m['dp']:+.5f} "
              f"sigma=dR={m['sigma_per_radius']:.4f} u_max={m['u_max']:.4f} rho_in={m['rho_a_in']:.3f} "
              f"rho_out={m['rho_a_out']:.3f} ({elapsed:.0f}s)", flush=True)

    good = [r for r in runs if not r.get("nan")]
    fit = {}
    if len(good) >= 3:
        x = np.array([1.0 / r["R_eq"] for r in good])
        y = np.array([r["dp"] for r in good])
        coef = np.polyfit(x, y, 1)
        yhat = np.polyval(coef, x)
        ss_res = float(((y - yhat) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        sigma_i = np.array([r["sigma_per_radius"] for r in good])
        fit = {
            "sigma": float(coef[0]),
            "intercept": float(coef[1]),
            "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
            "sigma_per_radius_mean": float(sigma_i.mean()),
            "sigma_per_radius_spread_pct": float(100.0 * (sigma_i.max() - sigma_i.min()) / abs(sigma_i.mean())),
            "fit_rel_err_vs_per_radius_mean_pct": float(
                100.0 * (coef[0] - sigma_i.mean()) / abs(sigma_i.mean())
            ),
        }
        fit["pass_r2"] = bool(fit["r2"] >= 0.95)
        fit["pass_spread_15pct"] = bool(fit["sigma_per_radius_spread_pct"] < 15.0)
    return {"tag": tag, "G": G, "rho_in": rho_in, "rho_out": rho_out, "n": n, "steps": steps,
            "velocity_mode": velocity_mode, "psi_kind": psi_kind, "self_coupling": self_coupling,
            "runs": runs, "fit": fit}


def radial_profile(n=48, G=2.0, rho_in=0.99, rho_out=0.07, steps=2000, R=8.0):
    rho_a, rho_b = M.drop_rho_fields(n, R, rho_in, rho_out, width=2.0)
    solver = M.MCMPPsiSolver(rho_a, rho_b, M.MCMPConfig(G=G))
    solver.run(steps)
    F = solver.fields(need_forces=True)
    j = torch.arange(n, dtype=M.DTYPE).view(-1, 1).expand(n, n)
    i = torch.arange(n, dtype=M.DTYPE).view(1, -1).expand(n, n)
    r = torch.sqrt((i - n / 2) ** 2 + (j - n / 2) ** 2)
    bins = np.arange(0, n // 2, 1.0)
    prof = {"r": [], "rho_a": [], "rho_b": [], "p": [], "u": []}
    u = torch.sqrt(F["ux"] ** 2 + F["uy"] ** 2)
    for lo in bins:
        m = (r >= lo) & (r < lo + 1.0)
        if int(m.sum()) == 0:
            continue
        prof["r"].append(float(lo + 0.5))
        for key, src in (("rho_a", F["rho_a"]), ("rho_b", F["rho_b"]), ("p", F["p"]), ("u", u)):
            prof[key].append(float(src[m].mean()))
    prof["R_eq"] = float(math.sqrt(float((F["rho_a"] > 0.5 * (rho_in + rho_out)).sum()) / math.pi))
    return prof


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def fig_laplace(primary: dict) -> str:
    good = [r for r in primary["runs"] if not r.get("nan")]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    x = np.array([1.0 / r["R_eq"] for r in good])
    y = np.array([r["dp"] for r in good])
    ax.plot(x, y, "o", ms=7, label="psi-field MCMP (measured)")
    if primary["fit"]:
        f = primary["fit"]
        xs = np.linspace(0, x.max() * 1.15, 50)
        ax.plot(xs, f["sigma"] * xs + f["intercept"], "-",
                label=rf"fit: $\sigma$={f['sigma']:.4f}, $R^2$={f['r2']:.3f}")
    ax.set_xlabel(r"$1/R_{eq}$  [lattice units$^{-1}$]")
    ax.set_ylabel(r"$\Delta p = p_{in}-p_{out}$  [lattice units]")
    ax.set_title(r"Young-Laplace law, $\psi$-field MCMP (G=%.2f)" % primary["G"])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = ART / "laplace.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p.name


def fig_acoustic(ac: dict) -> str:
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    cm = np.array([a["c_measured"] for a in ac["arms"]])
    cp = np.array([a["c_predicted"] for a in ac["arms"]])
    ax.plot([0.4, 0.95], [0.4, 0.95], "k--", lw=1, label="exact match")
    ax.plot(cp, cm, "o", ms=8)
    for a in ac["arms"]:
        ax.annotate(a["name"], (a["c_predicted"], a["c_measured"]),
                    textcoords="offset points", xytext=(6, -10), fontsize=8)
    ax.fill_between([0.4, 0.95], [0.4 * 0.85, 0.95 * 0.85], [0.4 * 1.15, 0.95 * 1.15],
                    color="green", alpha=0.12, label="±15%")
    ax.set_xlabel(r"analytic $c_{eff}=\sqrt{c_s^2(1+G\,\psi(\rho_0)\psi'(\rho_0))}$")
    ax.set_ylabel("measured acoustic speed (lattice units)")
    ax.set_title("Equation-of-state cross-check of the psi-field force")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = ART / "acoustic.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p.name


def fig_profile(prof: dict) -> str:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.0, 3.8))
    a1.plot(prof["r"], prof["rho_a"], label=r"$\rho_A$")
    a1.plot(prof["r"], prof["rho_b"], label=r"$\rho_B$")
    a1.axvline(prof["R_eq"], color="k", ls=":", lw=1, label=r"$R_{eq}$")
    a1.set_xlabel("r [lattice units]")
    a1.set_ylabel("density")
    a1.set_title("equilibrium drop profile (psi-field MCMP)")
    a1.grid(alpha=0.3)
    a1.legend(fontsize=8)
    a2.plot(prof["r"], prof["p"], label="p (model EOS)")
    a2.set_xlabel("r [lattice units]")
    a2.set_ylabel("pressure")
    a2b = a2.twinx()
    a2b.plot(prof["r"], prof["u"], color="tab:red", lw=0.9, label="|u| (spurious)")
    a2b.set_ylabel("|u|", color="tab:red")
    a2.set_title("pressure / spurious current across the interface")
    a2.grid(alpha=0.3)
    fig.tight_layout()
    p = ART / "drop_profile.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p.name


def fig_fields(n=48, G=2.0, R=8.0, steps=2000):
    rho_a, rho_b = M.drop_rho_fields(n, R, 0.99, 0.07, width=2.0)
    solver = M.MCMPPsiSolver(rho_a, rho_b, M.MCMPConfig(G=G))
    solver.run(steps)
    F = solver.fields()
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6))
    for ax, key, cmap in zip(axes, ("rho_a", "rho_b", "p"), ("viridis", "viridis", "coolwarm")):
        im = ax.imshow(F[key].numpy(), cmap=cmap)
        ax.set_title(key)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("psi-field MCMP drop: component densities and model pressure", fontsize=10)
    fig.tight_layout()
    p = ART / "fields.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p.name


# --------------------------------------------------------------------------- #
# chunked driver: a single stage per process (keeps every local run < 10 min)
# --------------------------------------------------------------------------- #
def _chunk(name: str, payload: dict) -> None:
    (ART / f"chunk_{name}.json").write_text(json.dumps(payload, indent=1))


def _load_chunks() -> dict:
    out = {"exact": {}, "acoustic": {}, "laplace": {"runs": []}, "ablations": {}}
    f = ART / "chunk_exact.json"
    if f.exists():
        out["exact"] = json.loads(f.read_text())
    f = ART / "chunk_acoustic.json"
    if f.exists():
        out["acoustic"] = json.loads(f.read_text())
    runs = {}
    for f in sorted(ART.glob("chunk_laplace_*.json")):
        r = json.loads(f.read_text())
        tag = r["tag"]
        runs.setdefault(tag, {"meta": r, "runs": []})["runs"].append(r["run"])
    for tag, blob in runs.items():
        meta = blob["meta"]
        good = [r for r in blob["runs"] if not r.get("nan")]
        fit = _fit_laplace(good) if len(good) >= 3 else {}
        arm = {k: meta.get(k) for k in ("tag", "G", "rho_in", "rho_out", "n", "steps",
                                        "velocity_mode", "psi_kind", "self_coupling")}
        arm.update({"runs": blob["runs"], "fit": fit})
        if tag == "primary":
            out["laplace"] = arm
        else:
            out["ablations"][tag] = arm
    return out


def _fit_laplace(good):
    x = np.array([1.0 / r["R_eq"] for r in good])
    y = np.array([r["dp"] for r in good])
    coef = np.polyfit(x, y, 1)
    yhat = np.polyval(coef, x)
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    sigma_i = np.array([r["sigma_per_radius"] for r in good])
    fit = {
        "sigma": float(coef[0]),
        "intercept": float(coef[1]),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "sigma_per_radius_mean": float(sigma_i.mean()),
        "sigma_per_radius_spread_pct": float(100.0 * (sigma_i.max() - sigma_i.min()) / abs(sigma_i.mean())),
        "fit_rel_err_vs_per_radius_mean_pct": float(100.0 * (coef[0] - sigma_i.mean()) / abs(sigma_i.mean())),
    }
    fit["pass_r2"] = bool(fit["r2"] >= 0.95)
    fit["pass_spread_15pct"] = bool(fit["sigma_per_radius_spread_pct"] < 15.0)
    return fit


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all")
    ap.add_argument("--radius", type=str, default=None)
    ap.add_argument("--steps", type=int, default=6000)
    args = ap.parse_args()
    stage = args.only
    t_start = time.time()

    if stage == "exact":
        _chunk("exact", exact_checks())
        print("exact done", flush=True)
        return
    if stage == "acoustic":
        _chunk("acoustic", acoustic_arms())
        return
    if stage in ("laplace_primary", "common_velocity", "self_coupling"):
        tag = {"laplace_primary": "primary", "common_velocity": "common_velocity",
               "self_coupling": "self_coupling"}[stage]
        kw = dict(tag=tag, steps=args.steps)
        if tag == "common_velocity":
            kw["velocity_mode"] = "common"
        if tag == "self_coupling":
            kw["self_coupling"] = 2.0
        radii = ([float(v) for v in args.radius.split(",")] if args.radius
                 else [6.0, 8.0, 10.0])
        for R in radii:
            res = laplace_arm([R], **kw)
            meta = {k: res[k] for k in ("tag", "G", "rho_in", "rho_out", "n", "steps",
                                        "velocity_mode", "psi_kind", "self_coupling")}
            _chunk(f"laplace_{tag}_R{R:g}", {"run": res["runs"][0], **meta})
        return
    if stage == "merge":
        data = _load_chunks()
        data["meta"] = {"card": "XVAL-LBM-SHAN-CHEN-MCMP-PSI-FIELD",
                        "anchor": "omalaspinas/palabos shan-chen-mcmp-psi-field",
                        "convention": "F_k = -G psi_k(x) sum_i w_i psi_k'(x+c_i) c_i, k'!=k, G>0 repulsive",
                        "acceptance": "<15%",
                        "merged": time.strftime("%Y-%m-%dT%H:%M:%S")}
        primary = data["laplace"]
        figs = {}
        if primary.get("fit"):
            figs["laplace"] = fig_laplace(primary)
        if data["acoustic"].get("arms"):
            figs["acoustic"] = fig_acoustic(data["acoustic"])
        prof = radial_profile()
        data["profile"] = prof
        figs["drop_profile"] = fig_profile(prof)
        figs["fields"] = fig_fields()
        data["figures"] = figs
        data["meta"]["elapsed_s"] = time.time() - t_start
        (ART / "results.json").write_text(json.dumps(data, indent=1))
        print(json.dumps({k: data[k] for k in ("exact",)}, indent=1)[:800])
        print("acoustic:", [(a["name"], round(a["rel_err_pct"], 2)) for a in data["acoustic"].get("arms", [])])
        print("laplace fit:", primary.get("fit"))
        print("ablations:", {k: v.get("fit") for k, v in data["ablations"].items()})
        return

    raise SystemExit(f"unknown stage {stage}")


def _unused_main():  # kept for reference: single-process full sweep
    t_start = time.time()
    OUT["meta"] = {
        "card": "XVAL-LBM-SHAN-CHEN-MCMP-PSI-FIELD",
        "anchor": "omalaspinas/palabos shan-chen-mcmp-psi-field",
        "convention": "F_k = -G psi_k(x) sum_i w_i psi_k'(x+c_i) c_i, k' != k, G>0 repulsive",
        "acceptance": "<15%",
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    print("[G1/G4] exact identities", flush=True)
    OUT["exact"] = exact_checks()
    print("  ", {k: v for k, v in OUT["exact"].items() if k != "production_kernel_diff"}, flush=True)

    print("[G2] acoustic EOS cross-check", flush=True)
    OUT["acoustic"] = acoustic_arms()

    print("[G3] Young-Laplace, primary arm", flush=True)
    primary = laplace_arm([8.0, 10.0, 12.0, 14.0], tag="primary")
    OUT["laplace"]["primary"] = primary

    print("[G5] ablations", flush=True)
    OUT["ablations"]["common_velocity"] = laplace_arm(
        [12.0], tag="common_velocity", velocity_mode="common"
    )
    OUT["ablations"]["self_coupling"] = laplace_arm(
        [12.0], tag="self_coupling", self_coupling=2.0
    )

    print("[figures]", flush=True)
    OUT["figures"] = {}
    OUT["figures"]["laplace"] = fig_laplace(primary)
    OUT["figures"]["acoustic"] = fig_acoustic(OUT["acoustic"])
    prof = radial_profile()
    OUT["profile"] = prof
    OUT["figures"]["drop_profile"] = fig_profile(prof)
    OUT["figures"]["fields"] = fig_fields()

    OUT["meta"]["elapsed_s"] = time.time() - t_start
    (ART / "results.json").write_text(json.dumps(OUT, indent=1))
    print("done in %.1f s -> %s" % (OUT["meta"]["elapsed_s"], ART / "results.json"), flush=True)

    # headline verdict
    f = primary["fit"]
    print("\nHEADLINE")
    print("  exact invariants: uniform=%s single=%s momentum=%s gather/scatter=%s" % (
        OUT["exact"]["uniform_force_max"], OUT["exact"]["single_component_force_max"],
        OUT["exact"]["momentum_sum"], OUT["exact"]["gather_vs_scatter"]))
    print("  production-kernel max diff: %s" % OUT["exact"].get("production_kernel_max_abs_diff"))
    for a in OUT["acoustic"]["arms"]:
        print("  acoustic %-18s err=%+.2f%% pass=%s" % (a["name"], a["rel_err_pct"], a["pass_15pct"]))
    if f:
        print("  laplace sigma=%.5f r2=%.4f spread=%.2f%% intercept=%.5f" % (
            f["sigma"], f["r2"], f["sigma_per_radius_spread_pct"], f["intercept"]))


if __name__ == "__main__":
    main()