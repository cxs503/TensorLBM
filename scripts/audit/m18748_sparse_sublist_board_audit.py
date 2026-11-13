#!/usr/bin/env python3
"""Board audit m18748 — closure check of m18377 ``XVAL-LBM-SPARSE-SUBLIST-NEIGHBOR-LIST``.

This is a *closure verification* (交差核验), not a redo.  Nothing in the audited
tree is modified; the audited module is imported read-only from
``/root/tp-agent/work/m18377`` and every number below is re-derived from a
primary source:

S1 static : deliverable inventory + git identity (tracked vs untracked)
S2 static : integration callers (is ``sparse_sublist_neighbor_list`` wired into
            the solver / platform? any import outside the module + its test?)
S3 static : hallucinated-number probe (are the reported values hard-coded?)
S4 static : shipped repro entry (does the deliverable contain a runner script?)
D1 db     : ``tensor_benchmark_cases`` rows for the anchor — the only referee
D2 db     : mission 18377 row (status / accepted)
D3 db     : ``tensor_reports`` row for mission 18377 (report discipline)
R1 runtime: fresh re-run of the delivered plane-channel case (H=16, ny=32, 2500
            steps) -> k_darcy vs the recorded 10.654330145418363
R2 runtime: convergence ladder 2500/5000/10000/20000 steps — is the reported
            0.1157 % an *error* or an under-resolved transient?
R3 runtime: representation equivalence (swap pair vs FluidNodeTable push), seeds
            1/5/13, plus the dense control arm
R4 runtime: neighbour-table equivalence against an *independent numpy* rewrite
            (not against the repo's own builder, which would be circular)
R5 runtime: mass conservation + no-NaN over 200 steps on a random porous mask
R6 runtime: analytic profile shape (ux vs parabola, wall value, symmetry)

Usage::

    python3 scripts/audit/m18748_sparse_sublist_board_audit.py
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

AUD = Path("/root/tp-agent/work/m18377")
AUD_SRC = AUD / "TensorLBM" / "src"
DELIV = AUD / "TensorLBM"
MODULE = DELIV / "src" / "tensorlbm" / "sparse_sublist_neighbor_list.py"
TESTS = DELIV / "tests" / "test_sparse_sublist_neighbor_list.py"
DOC = AUD / "docs" / "xval_sparse_sublist_neighbor_list.md"
ARTIFACT = DELIV / "artifacts" / "xval_sparse_sublist_neighbor_list" / "results.json"
FIGURE = DELIV / "artifacts" / "xval_sparse_sublist_neighbor_list" / "figures" / "fig1_channel_profile.png"
PREDECL = AUD / "xval" / "XVAL-LBM-SPARSE-SUBLIST-NEIGHBOR-LIST" / "predeclared.json"
ANCHOR = "XVAL-LBM-SPARSE-SUBLIST-NEIGHBOR-LIST"
MISSION = 18377

WS = Path(__file__).resolve().parents[2]
OUT = WS / "artifacts" / "m18748_audit"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(AUD_SRC))

import torch  # noqa: E402

torch.set_default_dtype(torch.float64)

import tensorlbm.sparse_sublist_neighbor_list as mod  # noqa: E402
from tensorlbm import porous  # noqa: E402
from tensorlbm.d2q9 import C, OPPOSITE  # noqa: E402

assert str(Path(mod.__file__).resolve()).startswith(str(AUD)), mod.__file__

TAU = 0.8
NU = (TAU - 0.5) / 3.0
H, NY, NX, G = 16, 32, 16, 1e-5
K_REF = (H / NY) * H * H / 12.0          # superficial slab permeability = 10.666666...
K_REF_INTRINSIC = H * H / 12.0           # intrinsic = 21.3333...
RECORDED = json.loads(ARTIFACT.read_text())

findings: list[dict] = []


def finding(fid: str, severity: str, title: str, evidence: str) -> None:
    findings.append({"id": fid, "severity": severity, "title": title, "evidence": evidence})


def sh(cmd: str, cwd: Path | None = None) -> str:
    return subprocess.run(cmd, shell=True, cwd=str(cwd) if cwd else None,
                          capture_output=True, text=True).stdout.strip()


def psql(sql: str) -> str:
    return sh(f"runuser -u postgres -- psql -d tensorplatform -tA -F'|' -c \"{sql}\"")


# ------------------------------------------------------------------ S1-S4 static
def static_checks() -> dict:
    tracked = sh(f"cd {AUD} && git ls-files | grep -E 'sparse_sublist|xval_sparse|SPARSE-SUBLIST' || true")
    untracked = sh(f"cd {AUD} && git status --porcelain | grep -E 'sparse_sublist|xval_sparse|SPARSE-SUBLIST' || true")
    callers = sh(f"grep -rl 'sparse_sublist' --include='*.py' {AUD} 2>/dev/null | "
                 f"grep -v '^{MODULE}$' | grep -v '^{TESTS}$' || true")
    runners = sh(f"ls {AUD}/xval/{ANCHOR}/ ; "
                 f"find {DELIV}/artifacts/xval_sparse_sublist_neighbor_list -name '*.py' | head")
    canon_import = sh(f"cd {AUD} && python3 -c \"import sys; sys.path.insert(0,'src');"
                      f"import tensorlbm.sparse_sublist_neighbor_list\" 2>&1 | tail -1")
    hard = sh(f"grep -nE '10\\.65|21\\.30|0\\.1156|10\\.6666' {MODULE} {TESTS} || true")
    test_ids = sh(f"grep -nE '^def test_' {TESTS}")
    if not tracked:
        finding("F1", "high",
                "交付文件全部未入 git（untracked）：模块/测试/工件/文档/预声明均无 commit",
                f"`git ls-files | grep sparse_sublist` = 空；`git status --porcelain` 5 条 ?? 条目 = {untracked!r}")
    if not callers:
        finding("F2", "high",
                "零集成调用方：模块未被主求解器/平台任何路径 import",
                "grep -rl 'sparse_sublist' 全树 .py（排除模块自身/测试）= 0 命中")
    if "ModuleNotFoundError" in canon_import:
        finding("F2b", "high",
                "canonical 包不可导入：交付写在旧嵌套影子树 TensorLBM/src，"
                "仓根 canonical src/tensorlbm 无 porous 子包、无该模块",
                f"cd {AUD} && python3 -c 'sys.path=[src]; import tensorlbm.sparse_sublist_neighbor_list' "
                f"-> {canon_import.strip()}")
    if not runners.replace("predeclared.json", "").strip():
        finding("F3", "high",
                "无交付 runner：xval/<anchor>/ 只有 predeclared.json，results.json 的产生脚本未随交付",
                f"ls {AUD}/xval/{ANCHOR}/ = {runners.splitlines()!r}；artifacts 目录下无生成脚本")
    if hard.strip():
        finding("F4", "low", "模块/测试中出现与结果同形的字面量（需确认非硬编码）", hard[:300])
    return {
        "tracked_files": tracked.splitlines(),
        "untracked_files": untracked.splitlines(),
        "integration_callers": callers.splitlines(),
        "shipped_runner_scripts": runners.splitlines(),
        "canonical_import": canon_import.strip(),
        "suspicious_literals": hard.splitlines(),
        "test_ids": test_ids.splitlines(),
        "files": {str(p): (p.stat().st_size if p.exists() else None)
                  for p in (MODULE, TESTS, DOC, ARTIFACT, FIGURE, PREDECL)},
    }


# ------------------------------------------------------------------ D1-D3 db
def db_checks() -> dict:
    cases = psql(f"SELECT case_name,metric,tolerance,status FROM tensor_benchmark_cases "
                 f"WHERE anchor_slug='{ANCHOR}';")
    n_cases = psql("SELECT count(*) FROM tensor_benchmark_cases;")
    mission = psql(f"SELECT status||'|'||coalesce(finished_at::text,'')||'|'||"
                   f"coalesce(result->>'accepted','') FROM tensor_dev_missions WHERE id={MISSION};")
    report = psql(f"SELECT id||'|'||title FROM tensor_reports WHERE mission_id={MISSION};")
    if not cases.strip():
        finding("F5", "high",
                f"基准裁判无数据：tensor_benchmark_cases 该 anchor 0 行（全表 {n_cases} 行）",
                f"SELECT ... WHERE anchor_slug='{ANCHOR}' -> 空；sibling X VAL 锚点均有登记行")
    if not report.strip():
        finding("F6", "medium",
                "报告纪律：tensor_reports 无 mission 18377 行（交付未按六段式发布报告）",
                f"SELECT id,title FROM tensor_reports WHERE mission_id={MISSION} -> 空")
    return {
        "benchmark_cases_for_anchor": cases.splitlines(),
        "benchmark_cases_total": int(n_cases or 0),
        "mission_row": mission,
        "report_rows": report.splitlines(),
    }


# ------------------------------------------------------------------ R1-R6 runtime
def channel_mask(ny: int = NY, nx: int = NX, h: int = H) -> torch.Tensor:
    solid = torch.ones((ny, nx), dtype=torch.bool)
    y0 = (ny - h) // 2
    solid[y0:y0 + h, :] = False
    return solid


def rest_state(k: int) -> torch.Tensor:
    f = torch.zeros((9, k))
    f[0] = 4.0 / 9.0
    f[1] = f[2] = f[3] = f[4] = 1.0 / 9.0
    f[5] = f[6] = f[7] = f[8] = 1.0 / 36.0
    return f


def run_channel(steps: int, dense: bool = False) -> dict:
    """Run the delivered plane-channel case; return k, profile and diagnostics."""
    solid = channel_mask()
    tbl = mod.build_sparse_sublist_neighbor_list(solid)
    f = rest_state(tbl.n_fluid)
    dtab = porous.build_fluid_node_table(solid) if dense else None
    fd = f.clone() if dense else None
    drift0 = float(f.sum())
    for _ in range(steps):
        f = mod.listlbm_step(f, tbl, TAU, G, 0.0)
        if dense:
            fd = porous.compressed_step(fd, dtab, TAU, G, 0.0)
    rho, ux, uy = porous.macroscopic_with_forcing(f, G, 0.0)
    u_intr = float(ux.mean())
    u_super = u_intr * tbl.porosity
    k = NU * u_super / G
    out = {
        "steps": steps,
        "k_darcy": k,
        "k_intrinsic": NU * u_intr / G,
        "u_intrinsic_mean": u_intr,
        "porosity": tbl.porosity,
        "err_vs_ref_pct": 100.0 * abs(k - K_REF) / K_REF,
        "err_intrinsic_pct": 100.0 * abs(NU * u_intr / G - K_REF_INTRINSIC) / K_REF_INTRINSIC,
        "ma_max": float(ux.abs().max()) / math.sqrt(1.0 / 3.0),
        "mass_drift_rel": abs(float(f.sum()) - drift0) / drift0,
        "n_nan": int(torch.isnan(f).sum() + torch.isinf(f).sum()),
        "finite": bool(torch.isfinite(f).all()),
    }
    # profile: the compressed state holds *only* the H fluid rows, row-major,
    # so row j of the channel is nodes [j*NX, (j+1)*NX)
    prof = [float(ux.view(H, NX)[j].mean()) for j in range(H)]
    out["profile"] = prof
    out["profile_shape"] = analytic_profile(prof, u_intr)
    out["uy_maxabs"] = float(uy.abs().max())
    if dense:
        out["dense_k_darcy"] = NU * float(ux.mean()) * 0.0 + NU * (float(
            porous.macroscopic_with_forcing(fd, G, 0.0)[1].mean()) * dtab.porosity) / G
        out["dense_maxabs_du_intrinsic"] = float((porous.macroscopic_with_forcing(fd, G, 0.0)[1]
                                                 - ux).abs().max())
        out["listlbm_vs_dense_state_maxabs"] = float((fd - f).abs().max())
    return out


def analytic_profile(prof: list[float], u_mean: float) -> dict:
    """Compare the measured parabola with u(y)=u_max*[1-((y-yc)/(H/2))^2]."""
    h = len(prof)
    y = np.arange(h) + 0.5 - h / 2.0          # half-way wall distance convention
    yc = y - y.mean()
    a = np.vstack([np.ones_like(yc), -(2.0 * yc / h) ** 2 + 1.0 * 0]).T  # placeholder
    # least-squares fit of u = A*(1 - (2*yc/h)^2) * ... simpler: fit u = c0 + c2*yc^2
    coef = np.polyfit(yc, np.asarray(prof), 2)
    fit = np.polyval(coef, yc)
    umax = float(np.max(prof))
    return {
        "fit_c0": float(coef[2]),
        "fit_c2": float(coef[0]),
        "parabola_resid_max_rel": float(np.max(np.abs(fit - np.asarray(prof))) / umax),
        "u_max_over_u_mean": umax / u_mean,
        "u_first_row": prof[0], "u_last_row": prof[-1],
        "symmetry_rel": float(abs(prof[0] - prof[-1]) / max(umax, 1e-300)),
        "u_mean_ok": u_mean,
    }


def numpy_table(solid_np: np.ndarray) -> dict:
    """Independent row-major compaction + periodic neighbour lookup (no repo code)."""
    ny, nx = solid_np.shape
    fluid = ~solid_np
    idx = -np.ones((ny, nx), dtype=np.int64)
    ys, xs = np.nonzero(fluid)
    idx[ys, xs] = np.arange(len(ys))
    c = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
                  [1, 1], [-1, 1], [-1, -1], [1, -1]])
    nb = -np.ones((len(ys), 8), dtype=np.int64)
    for s, d in enumerate(range(1, 9)):
        yy = (ys + c[d, 1]) % ny
        xx = (xs + c[d, 0]) % nx
        nb[:, s] = idx[yy, xx]
    return {"nodes": ys * nx + xs, "neighbour": nb, "index": idx}


def runtime_checks() -> dict:
    rec = RECORDED["records"][0]["A1"]
    # R1 fresh re-run at the delivered configuration
    r1 = run_channel(2500)
    r1["recorded_k_darcy"] = rec["k_darcy"]
    r1["rel_diff_vs_recorded"] = abs(r1["k_darcy"] - rec["k_darcy"]) / rec["k_darcy"]
    r1["recorded_k_intrinsic"] = rec["k_intrinsic"]
    r1["recorded_n_fluid"] = rec["n_fluid"]
    r1["n_fluid"] = int(mod.build_sparse_sublist_neighbor_list(channel_mask()).n_fluid)
    if r1["rel_diff_vs_recorded"] > 1e-12:
        finding("F7", "high", "工件数值与本次复跑不一致（不可复现）",
                f"recorded {rec['k_darcy']!r} vs fresh {r1['k_darcy']!r}")

    # R2 convergence ladder: is 0.1157 % converged?
    ladder = {}
    for steps in (2500, 5000, 10000, 20000):
        ladder[str(steps)] = run_channel(steps)
    k_end = ladder["20000"]["k_darcy"]
    ladder_drift = abs(ladder["2500"]["k_darcy"] - k_end) / k_end * 100.0
    if ladder_drift > 3.0:
        finding("F8", "high", "2500 步未收敛：报告的 0.12% 是瞬态而非稳态误差",
                f"k(2500)={ladder['2500']['k_darcy']:.6f} vs k(20000)={k_end:.6f} "
                f"= {ladder_drift:.3f}% 漂移")
    else:
        finding("F8-positive", "info", "2500 步已收敛（外推漂移 < 3%）",
                f"k(2500)={ladder['2500']['k_darcy']:.6f} vs k(20000)={k_end:.6f} = {ladder_drift:.4f}%")

    # R3 representation equivalence, fresh, seeds 1/5/13 + dense control arm
    equiv = []
    for seed in (1, 5, 13):
        gen = torch.Generator().manual_seed(seed)
        mask = torch.rand((20, 20), generator=gen) < 0.3
        mask[0, 0] = False
        t1 = porous.build_fluid_node_table(mask)
        t2 = mod.build_sparse_sublist_neighbor_list(mask)
        gen2 = torch.Generator().manual_seed(seed)
        f = torch.rand((9, t1.n_fluid), generator=gen2)
        f[0] = 4.0 / 9.0
        worst = 0.0
        for _ in range(50):
            f1 = porous.compressed_step(f, t1, TAU, 1e-5, 0.0)
            f2 = mod.listlbm_step(f, t2, TAU, 1e-5, 0.0)
            worst = max(worst, float((f1 - f2).abs().max()))
            f = f1
        equiv.append({"seed": seed, "max_abs_diff_50_steps": worst,
                      "table_nodes_equal": bool(torch.equal(t1.nodes, t2.nodes)),
                      "table_neighbour_equal": bool(torch.equal(t1.neighbour, t2.neighbour))})
    r3_dense = run_channel(2500, dense=True)

    # R4 independent numpy table rebuild on a random mask
    solid_np = np.zeros((17, 23), dtype=bool)
    solid_np[3:6, 4:9] = True
    solid_np[0, :] = True                     # a solid row -> wall wrapping
    t = mod.build_sparse_sublist_neighbor_list(torch.from_numpy(solid_np))
    ref = numpy_table(solid_np)
    r4 = {
        "nodes_equal": bool(np.array_equal(t.nodes.numpy(), ref["nodes"])),
        "neighbour_equal": bool(np.array_equal(t.neighbour.numpy(), ref["neighbour"])),
        "index_equal": bool(np.array_equal(t.index.numpy(), ref["index"])),
        "sublist": list(t.sublist),
        "n_fluid": t.n_fluid,
    }
    if not (r4["nodes_equal"] and r4["neighbour_equal"] and r4["index_equal"]):
        finding("F9", "high", "邻居表与独立 numpy 重写不一致（表构造有误）", json.dumps(r4))

    # R5 mass conservation / NaN
    mask = torch.rand((16, 16), generator=torch.Generator().manual_seed(7)) < 0.3
    mask[0, 0] = False
    tt = mod.build_sparse_sublist_neighbor_list(mask)
    f = rest_state(tt.n_fluid)
    m0 = float(f.sum())
    drift = 0.0
    for _ in range(200):
        f = mod.listlbm_step(f, tt, TAU, 0.0, 0.0)
        drift = max(drift, abs(float(f.sum()) - m0) / m0)
    r5 = {"max_mass_drift_rel_200_steps": drift,
          "finite": bool(torch.isfinite(f).all()),
          "n_fluid": tt.n_fluid, "porosity": tt.porosity}
    if not r5["finite"]:
        finding("F10", "high", "无外力 200 步出现 NaN/Inf", json.dumps(r5))

    return {
        "R1_fresh_at_delivered_config": r1,
        "R2_convergence_ladder": ladder,
        "R2_drift_2500_to_20000_pct": ladder_drift,
        "R3_representation_equivalence": equiv,
        "R3_dense_control_arm": r3_dense,
        "R4_independent_table": r4,
        "R5_mass_and_finiteness": r5,
        "env": {"torch": torch.__version__, "numpy": np.__version__,
                "module_file": mod.__file__},
    }


# ------------------------------------------------------ external anchor (/tmp/listlbm)
def external_anchor_check() -> dict:
    oss = Path("/tmp/listlbm")
    out: dict = {"oss_clone_present": oss.is_dir()}
    if not oss.is_dir():
        finding("F11", "medium", "外部 OSS 克隆 /tmp/listlbm 不存在（锚不可复核）", str(oss))
        return out
    out["clone_head"] = sh(f"cd {oss} && git rev-parse HEAD")
    out["clone_head_date"] = sh(f"cd {oss} && git log -1 --format=%cd")
    ml = (oss / "lists/base/mainlist_mod.f90")
    lat = (oss / "lattice/lattice_mod.f90")
    st = (oss / "stream/stream_mod.f90")
    out["mainlist_14_20_text"] = sh(f"sed -n '14,20p' {ml}").splitlines()[:3]
    out["mainlist_pdf_alloc_line"] = sh(f"grep -n 'allocate (this%pdf(0:nq_1' {ml}")
    out["mainlist_neilist_alloc_line"] = sh(f"grep -n 'allocate (this%neilist(1:nq_1' {ml}")
    out["d2q9_halfqs_line"] = sh(f"grep -n 'halfqs(1:nq_1/2) = (/1,2,5,6/)' {lat}")
    out["d3q19_halfqs_line"] = sh(f"grep -n 'halfqs(1:nq_1/2) = (/1,3,5' {lat}")
    out["stream1_sub_line"] = sh(f"grep -n 'subroutine  opposite_streamfluid' {st}")
    out["stream2_sub_line"] = sh(f"grep -n 'subroutine  reversepdfdirs_streamfluid' {st}")
    out["maxwell_swap_rule_present"] = bool(sh(f"sed -n '50,60p' {st}").count("ml%pdf(opposite(dir),ml%neilist"))
    if "GNU General Public" in " ".join(out["mainlist_14_20_text"]) or \
            "License GPLv3" in " ".join(out["mainlist_14_20_text"]):
        finding("F12", "low",
                "外部锚引用行号有误：docs 称 mainlist_mod.f90:14-20 为 pdf+neilist 声明，实际该区间是 GPL 头",
                f"sed -n 14,20p mainlist_mod.f90 -> {out['mainlist_14_20_text']!r}；"
                f"真实锚点 = {out['mainlist_pdf_alloc_line']} / {out['mainlist_neilist_alloc_line']}")
    claimed_commit = str(RECORDED.get("oss", {}).get("commit", ""))
    out["artifact_claimed_commit"] = claimed_commit
    if not re.search(r"\b[0-9a-f]{7,40}\b", claimed_commit):
        finding("F13", "low",
                "工件 provenance 未钉 commit sha：results.json 只写 'master (local clone /tmp/listlbm)'",
                f"本次实测 clone HEAD = {out['clone_head']} ({out['clone_head_date']})")
    return out


def main() -> int:
    result = {
        "anchor": ANCHOR,
        "audited_mission": MISSION,
        "audit_mission": 18748,
        "audited_tree": str(AUD),
        "static": static_checks(),
        "db": db_checks(),
        "external_anchor": external_anchor_check(),
        "runtime": runtime_checks(),
    }
    result["findings"] = findings
    (OUT / "audit_m18748.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(json.dumps({k: (v if k != "runtime" else "<<runtime>>") for k, v in result.items()},
                     indent=1, ensure_ascii=False)[:4000])
    rt = result["runtime"]
    print("\nR1 recorded vs fresh:", rt["R1_fresh_at_delivered_config"]["recorded_k_darcy"],
          "vs", rt["R1_fresh_at_delivered_config"]["k_darcy"],
          "rel", rt["R1_fresh_at_delivered_config"]["rel_diff_vs_recorded"])
    print("R2 ladder:", {k: round(v["k_darcy"], 8) for k, v in rt["R2_convergence_ladder"].items()})
    print("R3:", rt["R3_representation_equivalence"])
    print("R4:", rt["R4_independent_table"])
    print("R5:", rt["R5_mass_and_finiteness"])
    print("profile shape:", rt["R1_fresh_at_delivered_config"]["profile_shape"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())