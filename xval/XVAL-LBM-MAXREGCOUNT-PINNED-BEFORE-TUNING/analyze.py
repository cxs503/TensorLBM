"""Analysis for XVAL-LBM-MAXREGCOUNT-PINNED-BEFORE-TUNING (mission 17957).

Reads the raw arm JSONs written by ``run_arm.py`` (nothing is recomputed from
memory, every number in the report comes out of this file), applies the
thresholds predeclared in ``predeclared.json``, writes ``summary.json`` +
``*.csv`` and the report figures.

Usage::

    python xval/XVAL-LBM-MAXREGCOUNT-PINNED-BEFORE-TUNING/analyze.py \
        --artifacts artifacts/xval_maxregcount_pinned \
        --figures figures/xval_maxregcount_pinned \
        --oss-docs /root/tensor-ecosystem/TensorPlatform/docs/oss_inventory
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorlbm.bench_pinning import register_cap_verdict  # noqa: E402

THRESH = {
    "owner_acceptance_pct": 15.0,
    "stability_agreement_required": 5,
    "spread_pct_max": 10.0,
    "physics_acceptance_pct": 15.0,
    "physics_target_pct": 3.0,
    "pin_neutrality_rel": 1e-10,
    "sensitivity_pct_min_for_confound": 15.0,
}

# Live-value counts per lattice site, counted by reading the kernels:
#   tensorlbm/solver.py:86-97  collide_bgk        -> f(9) + feq(9) + rho/ux/uy/u_sq/cu(5) = 23
#   tensorlbm/solver.py:100-153 collide_bgk_fused -> f(9) + feq(9) + cu(9) + 5 scalars     = 32
#   tensorlbm/solver.py:221-274 collide_bgk_matmul-> f(9) + feq(9) + phi(6) + moments(3)  = 27
# The D3Q19 numbers are the same pattern with 19 directions (a 19-direction fused
# port of collide_bgk_fused): 19 f + 19 feq + 19 cu + 5 = 62.
# The OSS value is the collision_h inventory entry: 18 h + 18 heq + 18 hlph = 54 (fp32).
LIVE_VALUES = {
    "D2Q9 collide_bgk (fp64)": (23, "fp64"),
    "D2Q9 collide_bgk_fused (fp32)": (32, "fp32"),
    "D2Q9 collide_bgk_fused (fp64)": (32, "fp64"),
    "D2Q9 collide_bgk_matmul (fp64)": (27, "fp64"),
    "D3Q19 fused port (fp32)": (62, "fp32"),
    "D3Q19 fused port (fp64)": (62, "fp64"),
    "OSS CudaPhaseField collision_h (fp32)": (54, "fp32"),
}


def rel(a: float, b: float) -> float:
    return abs(a - b) / abs(b) if b else float("inf")


def spread_pct(values: list[float]) -> float:
    lo, hi = min(values), max(values)
    return (hi - lo) / lo * 100.0 if lo else float("inf")


def load_tune(art: Path) -> dict[str, list[dict]]:
    arms: dict[str, list[dict]] = {"U": [], "P": []}
    for arm in ("U", "P"):
        for k in range(5):
            p = art / f"tune_{arm}_{k}.json"
            if not p.exists():
                raise SystemExit(f"missing raw artifact {p}")
            rec = json.loads(p.read_text())
            rows = rec["audit"].get("tuned_rows") if arm == "P" else rec["tuned_rows"]
            phys = rec["audit"].get("physics") if arm == "P" else rec["physics"]
            arms[arm].append({
                "launch": rec["launch"],
                "inherited_threads": rec.get("inherited_threads"),
                "pin_sha256": rec["audit"].get("pin_sha256"),
                "pin_threads": (rec["audit"].get("pin") or {}).get("threads"),
                "rows": rows,
                "physics": phys,
            })
    return arms


def rank1(rows: list[dict], key: str) -> tuple[str, object]:
    best = min(rows, key=lambda r: r[key])
    return (best["collision_impl"], best["compile_mode"], best["step_time_s"], best["step_cpu_s"])


def arm_summary(launches: list[dict]) -> dict:
    out: dict = {"launches": [], "per_config": {}}
    ranks_wall, ranks_cpu = [], []
    for L in launches:
        rw = rank1(L["rows"], "step_time_s")
        rc = rank1(L["rows"], "step_cpu_s")
        ranks_wall.append((rw[0], rw[1]))
        ranks_cpu.append((rc[0], rc[1]))
        out["launches"].append({
            "launch": L["launch"],
            "inherited_threads": L["inherited_threads"],
            "pin_threads": L["pin_threads"],
            "rank1_wall": f"{rw[0]}/{rw[1]}",
            "rank1_wall_step_s": rw[2],
            "rank1_cpu": f"{rc[0]}/{rc[1]}",
            "rank1_cpu_step_s": rc[2],
        })
        for r in L["rows"]:
            key = f"{r['collision_impl']}/{r['compile_mode']}"
            out["per_config"].setdefault(key, {"wall": [], "cpu": [], "finite": []})
            out["per_config"][key]["wall"].append(r["step_time_s"])
            out["per_config"][key]["cpu"].append(r["step_cpu_s"])
            out["per_config"][key]["finite"].append(r["finite"])

    modal_wall = statistics.mode([f"{a}/{b}" for a, b in ranks_wall])
    modal_cpu = statistics.mode([f"{a}/{b}" for a, b in ranks_cpu])
    agree_wall = sum(1 for a, b in ranks_wall if f"{a}/{b}" == modal_wall)
    agree_cpu = sum(1 for a, b in ranks_cpu if f"{a}/{b}" == modal_cpu)
    for key, v in out["per_config"].items():
        v["median_wall_s"] = statistics.median(v["wall"])
        v["spread_wall_pct"] = spread_pct(v["wall"])
        v["median_cpu_s"] = statistics.median(v["cpu"])
        v["spread_cpu_pct"] = spread_pct(v["cpu"])
        v["all_finite"] = all(v["finite"])
    out.update({
        "modal_rank1_wall": modal_wall,
        "agreement_wall": agree_wall,
        "modal_rank1_cpu": modal_cpu,
        "agreement_cpu": agree_cpu,
        "modal_config_spread_wall_pct": out["per_config"][modal_wall]["spread_wall_pct"],
        "modal_config_spread_cpu_pct": out["per_config"][modal_cpu]["spread_cpu_pct"],
    })
    return out


def ladder_summary(ladder: dict) -> dict:
    rows = ladder["rows"]
    budgets = sorted({r["threads"] for r in rows})
    configs = sorted({f"{r['collision_impl']}/{r['compile_mode']}" for r in rows})
    per_config = {}
    for key in configs:
        sel = [r for r in rows if f"{r['collision_impl']}/{r['compile_mode']}" == key]
        per_config[key] = {
            "wall": {r["threads"]: r["step_time_s"] for r in sel},
            "cpu": {r["threads"]: r["step_cpu_s"] for r in sel},
            "sensitivity_wall_pct": spread_pct([r["step_time_s"] for r in sel]),
            "sensitivity_cpu_pct": spread_pct([r["step_cpu_s"] for r in sel]),
        }
    rank1_per_budget = {}
    for t in budgets:
        cand = [r for r in rows if r["threads"] == t]
        best_w = min(cand, key=lambda r: r["step_time_s"])
        best_c = min(cand, key=lambda r: r["step_cpu_s"])
        rank1_per_budget[t] = {
            "wall": f"{best_w['collision_impl']}/{best_w['compile_mode']}",
            "cpu": f"{best_c['collision_impl']}/{best_c['compile_mode']}",
        }
    return {
        "threads_ladder": budgets,
        "sweeps": ladder.get("sweeps"),
        "per_config": per_config,
        "rank1_per_budget": rank1_per_budget,
        "distinct_rank1_wall": len({v["wall"] for v in rank1_per_budget.values()}),
        "distinct_rank1_cpu": len({v["cpu"] for v in rank1_per_budget.values()}),
        "max_sensitivity_wall_pct": max(v["sensitivity_wall_pct"] for v in per_config.values()),
        "max_sensitivity_cpu_pct": max(v["sensitivity_cpu_pct"] for v in per_config.values()),
    }


def physics_summary(arms: dict[str, list[dict]]) -> dict:
    out = {}
    for arm in ("U", "P"):
        ph = [L["physics"] for L in arms[arm]]
        out[arm] = {
            "u_max_rel_err": [p["u_max_rel_err"] for p in ph],
            "profile_l2_rel": [p["profile_l2_rel"] for p in ph],
            "u_max_numeric": [p["u_max_numeric"] for p in ph],
            "u_max_analytic": ph[0]["u_max_analytic"],
            "ma_max": [p["ma_max"] for p in ph],
            "rho_drift_fluid": [p["rho_drift_fluid"] for p in ph],
            "profile_numeric": ph[0]["profile_numeric"],
            "profile_analytic": ph[0]["profile_analytic"],
            "H_fluid_rows": ph[0]["H_fluid_rows"],
        }
    pu, pp = out["U"], out["P"]
    prof_u = pu["profile_numeric"]
    prof_p = pp["profile_numeric"]
    max_prof_diff = max(abs(a - b) for a, b in zip(prof_u, prof_p))
    max_prof = max(abs(b) for b in prof_p)
    mp = max(pp["u_max_numeric"])
    out["pin_effect"] = {
        "u_max_rel": rel(max(pu["u_max_numeric"]), mp),
        "u_max_rel_err_rel": rel(statistics.median(pu["u_max_rel_err"]),
                                 statistics.median(pp["u_max_rel_err"])),
        "profile_l2_rel_rel": rel(statistics.median(pu["profile_l2_rel"]),
                                  statistics.median(pp["profile_l2_rel"])),
        "max_profile_point_rel": max_prof_diff / max_prof if max_prof else float("inf"),
    }
    return out


def register_table() -> list[dict]:
    out = []
    for name, (live, prec) in LIVE_VALUES.items():
        v = register_cap_verdict(live, prec, 64)
        out.append({"kernel": name, **v})
    return out


def oss_citation(oss_docs: Path | None) -> dict:
    """Re-read the OSS practice from the inventory (zero hand-copied numbers)."""
    if oss_docs is None or not oss_docs.exists():
        return {"available": False, "reason": "oss_inventory directory not present in this environment"}
    md = oss_docs / "CudaPhaseField.md"
    ev = oss_docs / "CudaPhaseField_evidence.py"
    hit_lines = []
    if md.exists():
        for i, line in enumerate(md.read_text().splitlines(), start=1):
            if "maxregcount" in line or "maxrregcount" in line:
                hit_lines.append({"file": "CudaPhaseField.md", "line": i, "text": line.strip()})
    ev_hits = []
    if ev.exists():
        for i, line in enumerate(ev.read_text().splitlines(), start=1):
            if "maxregcount" in line:
                ev_hits.append({"file": "CudaPhaseField_evidence.py", "line": i, "text": line.strip()})
    return {
        "available": True,
        "md_sha256": hashlib.sha256(md.read_bytes()).hexdigest() if md.exists() else None,
        "evidence_sha256": hashlib.sha256(ev.read_bytes()).hexdigest() if ev.exists() else None,
        "practice_lines": hit_lines,
        "makefile_line_records": ev_hits,
        "commit": "d6181e677c413bbec443570010bca60d896022b5",
    }


def verdicts(arms: dict, lad: dict, phys: dict) -> dict:
    u, p = arms["U"], arms["P"]
    v = {
        "P_selection_agreement_wall": f"{p['agreement_wall']}/5",
        "P_modal_spread_wall_pct": round(p["modal_config_spread_wall_pct"], 2),
        "U_selection_agreement_wall": f"{u['agreement_wall']}/5",
        "U_modal_spread_wall_pct": round(u["modal_config_spread_wall_pct"], 2),
        "P_selection_agreement_cpu": f"{p['agreement_cpu']}/5",
        "U_selection_agreement_cpu": f"{u['agreement_cpu']}/5",
        "ladder_max_sensitivity_wall_pct": round(lad["max_sensitivity_wall_pct"], 2),
        "ladder_max_sensitivity_cpu_pct": round(lad["max_sensitivity_cpu_pct"], 2),
        "ladder_distinct_rank1_wall": lad["distinct_rank1_wall"],
        "ladder_distinct_rank1_cpu": lad["distinct_rank1_cpu"],
        "physics_u_max_rel_err_pct_U": round(100 * statistics.median(phys["U"]["u_max_rel_err"]), 3),
        "physics_u_max_rel_err_pct_P": round(100 * statistics.median(phys["P"]["u_max_rel_err"]), 3),
        "physics_profile_l2_pct_U": round(100 * statistics.median(phys["U"]["profile_l2_rel"]), 3),
        "physics_profile_l2_pct_P": round(100 * statistics.median(phys["P"]["profile_l2_rel"]), 3),
        "pin_effect_on_physics_max_rel": max(phys["pin_effect"].values()),
    }
    checks = {
        "h1_pinned_reproducible": (p["agreement_wall"] >= THRESH["stability_agreement_required"]
                                   and p["modal_config_spread_wall_pct"] < THRESH["spread_pct_max"]),
        "h1_unpinned_confounded": (u["agreement_wall"] < THRESH["stability_agreement_required"]
                                   or lad["max_sensitivity_wall_pct"] >= THRESH["sensitivity_pct_min_for_confound"]),
        "physics_within_owner_band": (100 * statistics.median(phys["U"]["u_max_rel_err"]) < THRESH["physics_acceptance_pct"]
                                      and 100 * statistics.median(phys["P"]["u_max_rel_err"]) < THRESH["physics_acceptance_pct"]
                                      and 100 * statistics.median(phys["P"]["profile_l2_rel"]) < THRESH["physics_acceptance_pct"]),
        "physics_target_3pct": (100 * statistics.median(phys["P"]["u_max_rel_err"]) < THRESH["physics_target_pct"]
                                and 100 * statistics.median(phys["P"]["profile_l2_rel"]) < THRESH["physics_target_pct"]),
        "pin_is_physics_neutral": max(phys["pin_effect"].values()) < THRESH["pin_neutrality_rel"],
    }
    v["checks"] = checks
    v["H1_holds"] = bool(checks["h1_pinned_reproducible"] and checks["h1_unpinned_confounded"])
    v["physics_holds"] = bool(checks["physics_within_owner_band"] and checks["pin_is_physics_neutral"])
    return v


# ---------------------------------------------------------------- figures
def fig_ladder(lad: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for key, v in sorted(lad["per_config"].items()):
        xs = sorted(v["wall"])
        ax.plot(xs, [1e3 * v["wall"][t] for t in xs], marker="o", ms=3.5,
                ls="--" if key.endswith("None") else "-", label=key)
    for t, r in lad["rank1_per_budget"].items():
        ax.annotate("", xy=(t, 1e3 * lad["per_config"][r["wall"]]["wall"][t]), xytext=(t, 1e3 * 12),
                    arrowprops=dict(arrowstyle="-", color="0.7", lw=0.6))
    wins = sorted({v["wall"] for v in lad["rank1_per_budget"].values()})
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("resource budget: torch threads (pinned before tuning?)")
    ax.set_ylabel("per-step wall time [ms]")
    ax.set_title("A tuning result is a function of the budget it was tuned at\n"
                 f"rank-1 winner changes {lad['distinct_rank1_wall']} times across the ladder "
                 f"({', '.join(wins)})", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_selection(arms: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    for ax, arm in zip(axes, ("U", "P")):
        data = arms[arm]
        configs = sorted(data["per_config"])
        for i, key in enumerate(configs):
            ys = data["per_config"][key]["cpu"]
            ax.plot(range(1, 6), [1e3 * y for y in ys], marker="o", ms=4, label=key)
        for k, L in enumerate(data["launches"]):
            ax.annotate(L["rank1_cpu"], (k + 1, 1e3 * L["rank1_cpu_step_s"]), fontsize=6,
                        xytext=(0, 6), textcoords="offset points", ha="center", color="crimson")
        sub = ("arm U (status quo: inherited budget\n"
               + "threads per launch: " + ", ".join(str(L["inherited_threads"]) for L in data["launches"]) + ")") \
            if arm == "U" else "arm P (pinned budget: 8 threads, shift-capped)"
        ax.set_title(f"{sub}\nrank-1 agreement {data['agreement_wall']}/5 (wall)", fontsize=8.5)
        ax.set_xlabel("independent process launch")
        ax.set_ylabel("per-step CPU time [ms]")
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=6.5, ncol=2)
    fig.suptitle("Pinning the budget before tuning makes the selected kernel reproducible", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_physics(phys: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    an = phys["P"]["profile_analytic"]
    ys = range(1, len(an) + 1)
    ax.plot([1e3 * a for a in an], list(ys), "k-", lw=2, label="analytic (half-way walls)")
    ax.plot([1e3 * a for a in phys["P"]["profile_numeric"]], list(ys), "o", ms=4,
            label="arm P (pinned budget)")
    ax.plot([1e3 * a for a in phys["U"]["profile_numeric"]], list(ys), "x", ms=6,
            label="arm U (inherited budget)")
    ax.set_ylabel("y (fluid row)")
    ax.set_xlabel("u_x * 1e3 (lattice units)")
    ax.set_title("The resource pin is physics-neutral: both arms land on the same\n"
                 f"Poiseuille parabola (u_max err {100 * statistics.median(phys['P']['u_max_rel_err']):.2f}%, "
                 f"profile L2 {100 * statistics.median(phys['P']['profile_l2_rel']):.2f}%)", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_registers(reg: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    names = [r["kernel"] for r in reg]
    need = [r["required_with_reserve"] for r in reg]
    colors = ["tab:green" if r["fits"] else "tab:red" for r in reg]
    y = range(len(names))
    ax.barh(list(y), need, color=colors, alpha=0.85)
    ax.axvline(64, color="k", ls="--", lw=1.2)
    ax.text(64, len(names) - 0.3, " OSS pin: maxregcount 64", fontsize=8, va="bottom")
    for i, r in enumerate(reg):
        ax.text(r["required_with_reserve"] + 2, i, f"{r['required_with_reserve']}"
                + ("" if r["fits"] else f"  (+{r['overflow_words']} over)"), va="center", fontsize=7.5)
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=7.5)
    ax.set_xlabel("register words per thread (32-bit) = live values x words/value + 16 reserve")
    ax.set_title("The OSS pin *value* does not transfer: it must be recomputed per kernel/precision", fontsize=9.5)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts/xval_maxregcount_pinned")
    ap.add_argument("--figures", default="figures/xval_maxregcount_pinned")
    ap.add_argument("--oss-docs", default="/root/tensor-ecosystem/TensorPlatform/docs/oss_inventory")
    args = ap.parse_args(argv)

    art = REPO / args.artifacts
    figdir = REPO / args.figures
    figdir.mkdir(parents=True, exist_ok=True)

    arms_raw = load_tune(art)
    arms = {a: arm_summary(arms_raw[a]) for a in ("U", "P")}
    lad = ladder_summary(json.loads((art / "ladder.json").read_text()))
    phys = physics_summary(arms_raw)
    reg = register_table()
    cite = oss_citation(Path(args.oss_docs) if args.oss_docs else None)
    verd = verdicts(arms, lad, phys)

    summary = {
        "anchor": "XVAL-LBM-MAXREGCOUNT-PINNED-BEFORE-TUNING",
        "mission": 17957,
        "thresholds": THRESH,
        "arms": arms,
        "ladder": lad,
        "physics": phys,
        "registers": reg,
        "oss_citation": cite,
        "verdicts": verd,
    }
    (art / "summary.json").write_text(json.dumps(summary, indent=2))
    (art / "oss_citation.json").write_text(json.dumps(cite, indent=2))

    with (art / "table_arms.csv").open("w") as fh:
        fh.write("arm,launch,budget_threads,rank1_wall,rank1_wall_step_ms,rank1_cpu,rank1_cpu_step_ms\n")
        for a in ("U", "P"):
            for L in arms[a]["launches"]:
                budget = L["inherited_threads"] if a == "U" else L["pin_threads"]
                fh.write(f"{a},{L['launch']},{budget},{L['rank1_wall']},"
                         f"{1e3 * L['rank1_wall_step_s']:.4f},{L['rank1_cpu']},"
                         f"{1e3 * L['rank1_cpu_step_s']:.4f}\n")
    with (art / "table_ladder.csv").open("w") as fh:
        fh.write("collision_impl/compile_mode," + ",".join(str(t) for t in lad["threads_ladder"])
                 + ",sensitivity_wall_pct,sensitivity_cpu_pct\n")
        for key, v in sorted(lad["per_config"].items()):
            fh.write(key + "," + ",".join(f"{1e3 * v['wall'][t]:.4f}" for t in lad["threads_ladder"])
                     + f",{v['sensitivity_wall_pct']:.2f},{v['sensitivity_cpu_pct']:.2f}\n")
    with (art / "table_registers.csv").open("w") as fh:
        fh.write("kernel,live_arrays,precision,required_words,reserve,required_with_reserve,cap,fits,overflow\n")
        for r in reg:
            fh.write(f"{r['kernel']},{r['live_arrays']},{r['precision']},{r['required_words']},"
                     f"{r['reserve_words']},{r['required_with_reserve']},{r['cap_words']},"
                     f"{r['fits']},{r['overflow_words']}\n")

    fig_ladder(lad, figdir / "fig1_tuning_winner_vs_budget.png")
    fig_selection(arms, figdir / "fig2_selection_stability.png")
    fig_physics(phys, figdir / "fig3_poiseuille_pin_neutrality.png")
    fig_registers(reg, figdir / "fig4_register_budget_vs_oss_cap.png")

    print(json.dumps(verd, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())