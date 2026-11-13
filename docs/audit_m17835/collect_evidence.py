#!/usr/bin/env python3
"""m17835 交差核验：机器取证脚本（零手抄，所有数字由本脚本重算）。

审计对象：m17835 XVAL-LBM-CONV-AS-STREAMING-FIXED-KERNEL (TensorLBM)
被审树  : /root/tp-agent/work/m17835

产出
----
artifacts/m17835_audit/audit_m17835.json   机器读数
docs/audit_m17835/fig_m17835_evidence.png  证据图（3 panel）
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import torch

AUDITED = os.environ.get("AUDITED_TREE", "/root/tp-agent/work/m17835")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                       "artifacts", "m17835_audit")
DOC_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.join(AUDITED, "src"))

from tensorlbm.conv_streaming_fixed_kernel import (  # noqa: E402
    build_stream_kernels, stream_conv, stream_roll,
    run_poiseuille, PoiseuilleConfig,
    run_lid_driven_cavity, LidDrivenCavityConfig,
)


def sh(cmd, cwd=AUDITED):
    r = subprocess.run(cmd, cwd=cwd, shell=True, capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()


def main() -> dict:
    out: dict = {"audited_tree": AUDITED, "auditor_tree": os.getcwd()}

    # ---- 1. 交付面：git 归属 ----
    out["git"] = {
        "head": sh("git log --oneline -1"),
        "branch": sh("git branch --show-current"),
        "status_short": sh("git status --short").splitlines(),
        "claimed_files_tracked": sh(
            "git ls-files src/tensorlbm/conv_streaming_fixed_kernel.py "
            "examples/conv_streaming_xval.py runs/conv_streaming_poiseuille.json"),
        "commits_ahead_of_base": int(sh("git rev-list --count be78a87b..HEAD") or 0),
        "runs_dir_gitignored": sh("git check-ignore -v runs/conv_streaming_poiseuille.json"),
        "prior_round_commit_in_history": bool(
            sh("git merge-base --is-ancestor 725c1992 HEAD && echo YES || echo NO") == "YES"),
        "tests_referencing_module": sh(
            "grep -rln conv_streaming_fixed_kernel tests/ 2>/dev/null | wc -l"),
        "production_callers": sh(
            "grep -rln conv_streaming_fixed_kernel --include=*.py src/ benchmarks/ app/ 2>/dev/null | wc -l"),
    }

    # ---- 2. 工件：交付的 run_artifacts ----
    art_path = os.path.join(AUDITED, "runs", "conv_streaming_poiseuille.json")
    art = json.load(open(art_path)) if os.path.exists(art_path) else {}
    p = art.get("poiseuille", {})
    num, ana = p.get("numerical_ux", []), p.get("analytical_ux", [])
    per_node = []
    for j, (n, a) in enumerate(zip(num, ana)):
        per_node.append({
            "j": j,
            "got": n,
            "ref": a,
            "err_pct": (abs(n - a) / abs(a) * 100.0) if abs(a) > 1e-15 else 0.0,
        })
    interior = [r for r in per_node if 0 < r["j"] < len(per_node) - 1]
    out["artifact"] = {
        "path": art_path,
        "exists": bool(art),
        "top_level_keys": sorted(art.keys()),
        "poiseuille_keys": sorted(p.keys()),
        "shipped": {k: p.get(k) for k in
                    ("max_error_pct", "l2_error_pct", "converged",
                     "numerical_max_ux", "analytical_max_ux")},
        "shipped_schema_matches_entry": sorted(art.get("poiseuille", {}).keys()) == sorted(
            ["config", "converged", "l2_error_pct", "max_error_pct", "numerical_max_ux",
             "analytical_max_ux", "numerical_ux", "analytical_ux", "wall_clock_s"]),
        "shipped_has_ldc_block": "lid_driven_cavity" in art,
        "recomputed_max_err_pct": max(r["err_pct"] for r in interior),
        "recomputed_max_err_at_j": max(interior, key=lambda r: r["err_pct"])["j"],
        "recomputed_l2_err_pct": (sum(r["err_pct"] ** 2 for r in interior) / len(interior)) ** 0.5,
        "nodes_over_15pct": sum(1 for r in interior if r["err_pct"] > 15.0),
        "nodes_over_3pct": sum(1 for r in interior if r["err_pct"] > 3.0),
        "interior_nodes": len(interior),
        "wall_err_pct": {"j1_bottom": per_node[1]["err_pct"], "j_last_top": per_node[-1]["err_pct"]},
        "peak_deficit_pct": abs(max(num) - max(ana)) / max(ana) * 100.0,
        "profile_asymmetry_pct": abs(per_node[1]["err_pct"] - per_node[62]["err_pct"]),
    }

    # ---- 3. 独立复跑（repro_cmd 可执行性 + 数值一致性）----
    rp = run_poiseuille(PoiseuilleConfig(ny=64, nx=8, tau=1.0, body_force=1e-5, n_steps=5000))
    out["rerun"] = {
        "cmd": ("cd /root/tp-agent/work/m17835 && PYTHONPATH=src python3 -c "
                "'from tensorlbm.conv_streaming_fixed_kernel import run_poiseuille, PoiseuilleConfig;"
                "print(run_poiseuille(PoiseuilleConfig(ny=64,nx=8,tau=1.0,body_force=1e-5,n_steps=5000)))'"),
        "max_error_pct": rp.max_error_pct,
        "l2_error_pct": rp.l2_error_pct,
        "converged": rp.converged,
        "numerical_max_ux": max(rp.numerical_ux),
        "analytical_max_ux": max(rp.analytical_ux),
        "bitwise_equal_to_artifact": (rp.max_error_pct == p.get("max_error_pct")
                                      and rp.l2_error_pct == p.get("l2_error_pct")),
    }

    # ---- 4. conv 路径 vs roll 路径：等价性/质量守恒（核心主张探针）----
    torch.manual_seed(0)
    # 正定种群场（质量守恒探针需非负物理量；randn 有正负会掩盖符号）
    f = 1.0 / 9.0 + 0.01 * torch.rand(9, 8, 6, dtype=torch.float64)
    k = build_stream_kernels()
    a_conv, b_roll = stream_conv(f, k), stream_roll(f)
    out["conv_vs_roll"] = {
        "total_maxdiff": (a_conv - b_roll).abs().max().item(),
        "interior_maxdiff": (a_conv[:, 1:-1, 1:-1] - b_roll[:, 1:-1, 1:-1]).abs().max().item(),
        "mass_in": f.sum().item(),
        "mass_conv": a_conv.sum().item(),
        "mass_roll": b_roll.sum().item(),
        "conv_mass_loss_pct": (f.sum().item() - a_conv.sum().item()) / f.sum().item() * 100.0,
        "roll_mass_loss_pct": (f.sum().item() - b_roll.sum().item()) / f.sum().item() * 100.0,
        "doc_claim": "docs/xval/conv_streaming_fixed_kernel.md §2.1/§5: stream_conv = 'exact TF parity', "
                     "stream_roll 'mathematically identical'",
    }

    # ---- 5. LDC 交付面：文档称 pending，实跑一个可执行壳确认 ----
    ldc = run_lid_driven_cavity(LidDrivenCavityConfig(nx=32, ny=32, tau=1.0, u_lid=0.1, n_steps=300))
    out["ldc"] = {
        "shipped_artifact_for_ldc": out["artifact"]["shipped_has_ldc_block"],
        "doc_claim": "docs/xval/..._kernel.md §4.2 'Results pending from cluster job cj1700'",
        "auditor_smoke_32x32x300": {
            "center_vortex_ux": ldc.center_vortex_ux,
            "center_vortex_uy": ldc.center_vortex_uy,
            "final_max_ux": ldc.final_max_ux,
            "converged": ldc.converged,
        },
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "audit_m17835.json"), "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)

    # ---- 6. 证据图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
        jj = [r["j"] for r in per_node]
        ax[0].plot([r["got"] for r in per_node], jj, "o-", ms=3, label="got (artifact)")
        ax[0].plot([r["ref"] for r in per_node], jj, "--", label="ref analytic u(y)")
        ax[0].set_xlabel("u_x [lattice]"); ax[0].set_ylabel("y index")
        ax[0].set_title("Poiseuille: got vs ref (m17835)")
        ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

        ax[1].plot([r["err_pct"] for r in per_node], jj, "o-", ms=3, color="crimson")
        ax[1].axvline(15.0, color="k", ls="--", lw=1, label="anchor criterion 15%")
        ax[1].axvline(3.0, color="gray", ls=":", lw=1, label="internal target 3%")
        ax[1].set_xlabel("relative error [%]"); ax[1].set_ylabel("y index")
        ax[1].set_title("error grows with y (asymmetric walls)")
        ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)

        names = ["mass loss\nconv2d", "mass loss\nroll", "|conv-roll|\ninterior", "|conv-roll|\nboundary"]
        vals = [out["conv_vs_roll"]["conv_mass_loss_pct"], out["conv_vs_roll"]["roll_mass_loss_pct"],
                out["conv_vs_roll"]["interior_maxdiff"], out["conv_vs_roll"]["total_maxdiff"]]
        bars = ax[2].bar(names, vals, color=["crimson", "seagreen", "gray", "orange"])
        ax[2].set_yscale("symlog", linthresh=1e-6)
        ax[2].set_ylabel("percent / abs diff (symlog)")
        ax[2].set_title("conv-as-streaming: zero-pad breaks BC parity")
        for b, v in zip(bars, vals):
            ax[2].text(b.get_x() + b.get_width() / 2, v, f"{v:.2e}", ha="center", va="bottom", fontsize=7)
        ax[2].grid(alpha=.3, axis="y")

        fig.suptitle("m17835 audit — XVAL-LBM-CONV-AS-STREAMING-FIXED-KERNEL", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(DOC_DIR, "fig_m17835_evidence.png"), dpi=130)
        out["figure"] = os.path.join(DOC_DIR, "fig_m17835_evidence.png")
    except Exception as exc:  # pragma: no cover
        out["figure_error"] = repr(exc)

    with open(os.path.join(OUT_DIR, "audit_m17835.json"), "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return out


if __name__ == "__main__":
    main()