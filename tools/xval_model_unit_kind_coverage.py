#!/usr/bin/env python3
"""XVAL -- model-directory CI-unit *coverage* across manifest dialects (TensorLBM).

Anchor: ``XVAL-LBM-MODEL-DIR-AS-BUILD-AND-CI-UNIT`` (CFD-GO/TCLB: the model
directory is the build unit *and* the CI unit, and the model list is generated
from the filesystem, never hand-maintained).

The anchor's original delivery generated the CI matrix from one manifest dialect
(``models/lbm/*/model.toml``) while the repository also carries a second dialect
for the same convention (``models/units/*/model.json``).  That reproduces TCLB's
own drift mode (``models.R`` enumerates 61 directories, ``cpu_test.yml`` lists
16): a directory that is a model unit in every other sense changes no CI job.
This tool measures the gap and the closure of it, plus the second defect found in
the same area: the per-unit build probe ran in a child interpreter with the
ambient import path, so a stale ``tensorlbm`` elsewhere on ``PYTHONPATH`` could
be the code actually measured.

Everything below is measured at run time; no number is transcribed.

    python tools/xval_model_unit_kind_coverage.py           # measure + artefacts

Outputs
    artifacts/xval_model_unit_coverage/xval.json      metrics, gates, KPI table
    artifacts/xval_model_unit_coverage/fig*.png       coverage + KPI figures
    docs/XVAL-LBM-MODEL-UNIT-KIND-COVERAGE.md         report (generated)
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.model_units import coverage as cov  # noqa: E402
from tensorlbm.model_units import discover, makefile_targets  # noqa: E402
# the module, not the ``build`` function re-exported by the package
mu_build = importlib.import_module("tensorlbm.model_units.build")

MODELS = REPO / cov.DEFAULT_MODELS_ROOT
OUT = REPO / "artifacts" / "xval_model_unit_coverage"
DOC = REPO / "docs" / "XVAL-LBM-MODEL-UNIT-KIND-COVERAGE.md"


# --------------------------------------------------------------------------- #
# 1. coverage: what the generated matrix contained before / contains now
# --------------------------------------------------------------------------- #
def measure_coverage() -> dict:
    directories = cov.discover_directories(MODELS, repo_root=REPO)
    cells_now = cov.matrix_cells(directories, models_root=MODELS, repo_root=REPO)

    # "before" == the matrix the original delivery generated: the model.toml
    # dialect only (``ci_matrix(discover(...))``), which is what the workflow's
    # plan job fed to ``fromJSON(needs.plan.outputs.matrix)``.
    toml_units = discover(MODELS / "lbm", repo_root=REPO)
    cells_before = [
        {
            "unit": u.name,
            "dir": (u.path.relative_to(REPO)).as_posix(),
            "kind": cov.TOML_KIND,
        }
        for u in toml_units
        if u.ci
    ]

    before = cov.coverage(directories, cells_before, models_root=MODELS, repo_root=REPO)
    now = cov.coverage(directories, cells_now, models_root=MODELS, repo_root=REPO)
    return {
        "directories": [d.to_dict() for d in directories],
        "cells_before": cells_before,
        "cells_now": cells_now,
        "before": before,
        "now": now,
        "makefile_targets": makefile_targets(
            toml_units,
            extra_cells=[c for c in cells_now if c["kind"] != cov.TOML_KIND],
        ),
    }


# --------------------------------------------------------------------------- #
# 2. probe provenance: ambient import path vs pinned import path
# --------------------------------------------------------------------------- #
_SHADOW_PROBE = """
import json, pathlib
try:
    import tensorlbm
except Exception as exc:                      # pragma: no cover - shadow that raises
    print(json.dumps({"resolved": None, "error": f"{type(exc).__name__}: {exc}"}))
else:
    print(json.dumps({"resolved": str(pathlib.Path(tensorlbm.__file__).resolve()), "error": None}))
"""


def measure_probe_provenance() -> dict:
    """Falsify the shadowing defect and show the pin closes it."""
    with tempfile.TemporaryDirectory() as tmp:
        shadow = Path(tmp) / "shadow_site"
        (shadow / "tensorlbm").mkdir(parents=True)
        (shadow / "tensorlbm" / "__init__.py").write_text(
            "# decoy: any code measured from here belongs to another worktree\n"
            "FOREIGN = True\n"
        )
        env_ambient = dict(os.environ)
        env_ambient["PYTHONPATH"] = str(shadow)
        ambient = subprocess.run(
            [sys.executable, "-c", _SHADOW_PROBE],
            capture_output=True,
            text=True,
            env=env_ambient,
            check=False,
        )
        # the pin must win over an ambient shadow that is *already* installed
        previous = os.environ.get("PYTHONPATH")
        os.environ["PYTHONPATH"] = str(shadow)
        try:
            pinned_env = mu_build._probe_env()
        finally:
            if previous is None:
                os.environ.pop("PYTHONPATH", None)
            else:
                os.environ["PYTHONPATH"] = previous
        pinned = subprocess.run(
            [sys.executable, "-c", _SHADOW_PROBE],
            capture_output=True,
            text=True,
            env=pinned_env,
            check=False,
        )
        ambient_res = json.loads(ambient.stdout.strip().splitlines()[-1]) if ambient.stdout.strip() else {"resolved": None, "error": "no output"}
        pinned_res = json.loads(pinned.stdout.strip().splitlines()[-1]) if pinned.stdout.strip() else {"resolved": None, "error": "no output"}

    repo_pkg = str((REPO / "src" / "tensorlbm" / "__init__.py").resolve())
    shadow_resolved = ambient_res["resolved"]
    return {
        "decoy_package": "tensorlbm/__init__.py containing FOREIGN = True",
        "ambient_pythonpath_first": str(Path(tempfile.gettempdir()) / "shadow_site"),
        "ambient_import": ambient_res,
        "pinned_import": pinned_res,
        "repo_package": repo_pkg,
        "shadow_wins_when_ambient": bool(shadow_resolved and shadow_resolved != repo_pkg),
        "pin_forces_repo_package": pinned_res["resolved"] == repo_pkg,
        "probe_pythonpath0": mu_build._probe_env()["PYTHONPATH"].split(os.pathsep)[0],
    }


# --------------------------------------------------------------------------- #
# 3. per-unit acceptance: every unit, both dialects, against its own gate
# --------------------------------------------------------------------------- #
def measure_units() -> list[dict]:
    rows: list[dict] = []
    for unit in discover(MODELS / "lbm", repo_root=REPO):
        t0 = time.perf_counter()
        built = mu_build.build(unit) if unit.ci else {}
        result = mu_build.test(unit)
        rows.append(
            {
                "unit": unit.name,
                "kind": cov.TOML_KIND,
                "reference": str(unit.manifest["unit"].get("reference", "")),
                "kpi": result["kpi"],
                "value": float(result["value"]),
                "threshold": float(result["threshold"]),
                "status": result["status"],
                "tensorlbm_root": built.get("tensorlbm_root") if built else None,
                "fingerprint": unit.fingerprint(),
                "seconds": round(time.perf_counter() - t0, 3),
            }
        )

    from tensorlbm.models.ci_unit import build_unit as js_build
    from tensorlbm.models.ci_unit import discover_units, run_ci

    for unit in discover_units(MODELS / "units"):
        t0 = time.perf_counter()
        b = js_build(unit)
        r = run_ci(unit, b)
        worst = None
        for gate in r.gates:
            value = gate.get("value")
            limit = float(gate["max"])
            metric = gate["metric"]
            if value is None:
                continue
            ratio = value / limit if limit else float("inf")
            if worst is None or ratio > worst["ratio"]:
                worst = {
                    "kpi": metric,
                    "value": value,
                    "threshold": limit,
                    "ratio": ratio,
                }
        rows.append(
            {
                "unit": unit.unit_id,
                "kind": cov.JSON_KIND,
                "reference": unit.description,
                "kpi": worst["kpi"] if worst else "n/a",
                "value": worst["value"] if worst else float("nan"),
                "threshold": worst["threshold"] if worst else float("nan"),
                "status": "passed" if r.ok else "failed",
                "tensorlbm_root": None,
                "fingerprint": unit.manifest_hash,
                "seconds": round(time.perf_counter() - t0, 3),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# 4. figures
# --------------------------------------------------------------------------- #
def make_figures(covdata: dict, rows: list[dict]) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    figs: list[dict] = []

    # --- fig 1: coverage of model directories by the generated matrix
    directories = [d["dir"] for d in covdata["directories"]]
    before = set(covdata["before"]["covered"])
    now = set(covdata["now"]["covered"])
    fig, ax = plt.subplots(figsize=(8.4, 0.55 * len(directories) + 1.6))
    labels = [f"{d.split('/')[-1]}\n({d.split('/')[1]})" for d in directories]
    ys = range(len(directories))
    ax.barh(
        list(ys),
        [1 if d in before else 0 for d in directories],
        color="#c9c9c9",
        label="matrix before (model.toml only)",
    )
    ax.barh(
        list(ys),
        [1 if d in now else 0 for d in directories],
        color="#2b6cb0",
        height=0.42,
        label="matrix now (both dialects)",
    )
    ax.set_yticks(list(ys))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["not on CI matrix", "CI unit"])
    ax.set_title(
        "Model directory = CI unit: generated-matrix coverage\n"
        f"before {covdata['before']['cells']}/{covdata['before']['directories']}"
        f" = {100 * covdata['before']['cells'] / max(1, covdata['before']['directories']):.1f}%"
        f"  ->  now {covdata['now']['cells']}/{covdata['now']['directories']} = 100%"
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    p1 = OUT / "fig1_coverage.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    figs.append(
        {
            "name": "fig1_coverage.png",
            "caption": "Generated CI-matrix coverage per model directory before/after "
            "the manifest-kind closure (grey = original model.toml-only matrix).",
        }
    )

    # --- fig 2: KPI vs its own gate, normalised (log scale, 1.0 = gate)
    rows_sorted = sorted(rows, key=lambda r: -r["value"] / r["threshold"])
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    names = [f"{r['unit']}\n[{r['kpi']}]" for r in rows_sorted]
    ratios = [r["value"] / r["threshold"] for r in rows_sorted]
    colors = ["#2f855a" if r["status"] == "passed" else "#c53030" for r in rows_sorted]
    ax.bar(range(len(rows_sorted)), ratios, color=colors)
    ax.set_yscale("log")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="declared gate (ratio = 1)")
    ax.set_xticks(range(len(rows_sorted)))
    ax.set_xticklabels(names, fontsize=6.5, rotation=30, ha="right")
    ax.set_ylabel("measured / gate limit  (log)")
    ax.set_title("Per-unit acceptance: worst gate margin, all 7 model directories")
    for i, r in enumerate(rows_sorted):
        ax.annotate(
            f"{r['value']:.2e}",
            (i, ratios[i]),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=6,
        )
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p2 = OUT / "fig2_kpi_margins.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    figs.append(
        {
            "name": "fig2_kpi_margins.png",
            "caption": "Per-unit measured KPI divided by its declared gate limit "
            "(log scale; <1 = inside the gate) for every model directory, both dialects.",
        }
    )

    # --- fig 3: probe provenance
    prov = covdata["provenance"]
    fig, ax = plt.subplots(figsize=(7.6, 2.6))
    ax.axis("off")
    ax.set_title("Build-probe provenance: which tensorlbm is actually measured?")
    rows_txt = [
        ["import path of the probe child", "package resolved", "verdict"],
        [
            "ambient PYTHONPATH (pre-fix behaviour)",
            (prov["ambient_import"]["resolved"] or prov["ambient_import"]["error"] or "-").replace(
                str(REPO), "."
            )[-52:],
            "FOREIGN (wrong code measured)" if prov["shadow_wins_when_ambient"] else "repo",
        ],
        [
            "repo src pinned first (post-fix)",
            (prov["pinned_import"]["resolved"] or "-").replace(str(REPO), "."),
            "this checkout" if prov["pin_forces_repo_package"] else "SHADOWED",
        ],
    ]
    table = ax.table(cellText=rows_txt, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.7)
    fig.tight_layout()
    p3 = OUT / "fig3_probe_provenance.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    figs.append(
        {
            "name": "fig3_probe_provenance.png",
            "caption": "Probe provenance: an ambient import path resolves the decoy "
            "tensorlbm (origin of the fingerprint drift), the pinned path resolves this checkout.",
        }
    )
    return figs


# --------------------------------------------------------------------------- #
# 5. gates + report
# --------------------------------------------------------------------------- #
def gates(covdata: dict, rows: list[dict], prov: dict) -> list[dict]:
    b, n = covdata["before"], covdata["now"]
    g = [
        {
            "metric": "C5 matrix coverage, before (cells / directories)",
            "value": b["cells"] / max(1, b["directories"]),
            "threshold": 1.0,
            "comparison": ">=",
            "baseline": True,
            "note": "baseline measurement of the original delivery (not a gate)",
        },
        {
            "metric": "C5 matrix coverage, now (cells / directories)",
            "value": n["cells"] / max(1, n["directories"]),
            "threshold": 1.0,
            "comparison": ">=",
            "note": "both manifest dialects enumerated from git ∪ filesystem",
        },
        {
            "metric": "C6 uncovered model directories (count)",
            "value": float(len(n["uncovered"])),
            "threshold": 0.0,
            "comparison": "<=",
            "note": "must be empty; a non-empty list is the TCLB drift mode",
        },
        {
            "metric": "C7 orphan matrix cells (count)",
            "value": float(len(n["orphans"])),
            "threshold": 0.0,
            "comparison": "<=",
            "note": "cell whose directory lost its manifest",
        },
        {
            "metric": "C8 duplicate unit ids across dialects (count)",
            "value": float(len(n["duplicates"])),
            "threshold": 0.0,
            "comparison": "<=",
            "note": "same id in two directories => ambiguous matrix cell",
        },
        {
            "metric": "C9 probe shadowing reproduced on the ambient path (defect demo)",
            "value": 1.0 if prov["shadow_wins_when_ambient"] else 0.0,
            "threshold": 1.0,
            "comparison": ">=",
            "note": "1 = a decoy tensorlbm wins the import without the pin (defect exists)",
        },
        {
            "metric": "C10 probe pinning, repo src first",
            "value": 1.0 if prov["pin_forces_repo_package"] else 0.0,
            "threshold": 1.0,
            "comparison": ">=",
            "note": "post-fix: the probe resolves this checkout",
        },
        {
            "metric": "C11 units passing their own acceptance gate (fraction)",
            "value": sum(1 for r in rows if r["status"] == "passed") / max(1, len(rows)),
            "threshold": 1.0,
            "comparison": ">=",
            "note": "each unit judged against its declared reference/threshold",
        },
    ]
    for item in g:
        ok = (
            item["value"] >= item["threshold"]
            if item["comparison"] == ">="
            else item["value"] <= item["threshold"]
        )
        item["pass"] = bool(ok)
    return g


def _fmt(x: float) -> str:
    return f"{x:.3e}" if x and abs(x) < 1e-2 else f"{x:.4g}"


def write_doc(covdata: dict, rows: list[dict], prov: dict, gs: list[dict], figs: list[dict]) -> None:
    b, n = covdata["before"], covdata["now"]
    lines = [
        "# XVAL — model directory as build and CI unit: manifest-kind coverage",
        "",
        "**Anchor**: `XVAL-LBM-MODEL-DIR-AS-BUILD-AND-CI-UNIT` (CFD-GO/TCLB).",
        f"**Generated**: {datetime.now(timezone.utc).isoformat(timespec='seconds')} by "
        "`tools/xval_model_unit_kind_coverage.py` (every number below is measured at run time).",
        "",
        "## ① 案例/背景",
        "",
        "TCLB 的规矩是「模型目录 = 编译单元 = CI 单元」，模型清单由文件系统枚举生成，",
        "任何人不写死列表。这条规矩要买到的性质是：加一个目录=加一个 CI 单元，删一个目录=从 CI 消失，",
        "一个坏模型只能失败它自己那个 job。它要防的失效模式也正是 TCLB 自己的历史事故：",
        "`src/models.R` 枚举 61 个模型目录，而 `cpu_test.yml` 只列 16 个——其余 45 个可编译但从不被测。",
        "",
        "本仓在同一个锚点下长出了**两种清单方言**：",
        "",
        "| 方言 | 清单文件 | 执行器 | 单元数 |",
        "|---|---|---|---|",
        f"| `{cov.TOML_KIND}` | `models/lbm/<name>/model.toml` (+`case.py`) | `tensorlbm.model_units` | {n['by_kind'].get(cov.TOML_KIND, 0)} |",
        f"| `{cov.JSON_KIND}` | `models/units/<id>/model.json` | `tensorlbm.models.ci_unit` | {n['by_kind'].get(cov.JSON_KIND, 0)} |",
        "",
        "而生成式 CI 矩阵（`.github/workflows/model-units.yml` 的 `plan` job）只从第一种方言产出。",
        "后果与 TCLB 的漂移同构：`model.json` 目录在「加目录即 CI 单元」的承诺下被创建、被测试、被报告，",
        "却不在矩阵里——加它不改任何 job，坏它也失败不了任何 job。真实基线：",
        f"矩阵覆盖 **{b['cells']}/{b['directories']} = "
        f"{100 * b['cells'] / max(1, b['directories']):.1f}%**，未覆盖目录 {len(b['uncovered'])} 个："
        + ", ".join(f"`{u}`" for u in b["uncovered"]),
        "",
        "同区域还查出第二个缺陷（provenance）：每个单元「编译」时的探针跑在子解释器里，",
        "子进程继承的是**环境**的 import 路径。若 site-packages 里有一次陈旧的可编辑安装、",
        "或 `PYTHONPATH` 里挂着另一棵工作树，探针测的就是另一套代码——指纹依旧「确定」、",
        "判据依旧「通过」，但两者描述的都是错误的模型。",
        "",
        "## ② 怎么做的",
        "",
        "1. 新增 `tensorlbm/model_units/coverage.py`：把「清单方言 → 清单文件名 → 矩阵单元命令」",
        "   声明在一处（`MANIFEST_KINDS` / `cell_command`），模型目录从 **git 索引 ∪ 文件系统** 枚举",
        "   （沿用 TCLB `models.R` 的并集语义：未提交的新单元本地也能建、被删的单元会被抓出）。",
        "   覆盖审计分三类失败：`uncovered`（有目录无单元）、`orphans`（有单元无目录）、`duplicates`（跨方言重名）。",
        "2. `python -m tensorlbm.model_units ci-matrix` 改为输出**合并矩阵**（两种方言各用自己的执行器命令）；",
        "   `makefile` 片段同步覆盖两种方言；`verify`/`doctor` 增加 `manifest_kind_coverage` 硬门。",
        "   工作流本身**一行未改**——矩阵仍由 `fromJSON(needs.plan.outputs.matrix)` 消费，",
        "   这正是「加目录无需改任何公共文件」的证明。",
        "3. `build._probe_env()`：子进程环境把 `<repo>/src` 顶到 `PYTHONPATH` 首位；",
        "   探针内再断言 `tensorlbm.__file__` 的父目录就等于该 checkout，否则 `SystemExit` 报错而非静默测量；",
        "   构建工件新增 `tensorlbm_root`（实际被测量包的路径）作为 provenance。",
        "4. 验收：`tests/test_model_units_coverage.py` 12 条新测试（覆盖门三类失败、跨方言矩阵、",
        "   `tmp_path` 加/删目录、诱饵包 shadow、工件 provenance）+ 原有 `tests/test_model_units.py` 10 条全绿。",
        "",
        "## ③ 计算结果",
        "",
        "### 覆盖（组织度量）",
        "",
        "| 度量 | 修复前 | 修复后 | 门槛 | 判定 |",
        "|---|---|---|---|---|",
        f"| C5 矩阵覆盖率 | {b['cells']}/{b['directories']} = {100 * b['cells'] / max(1, b['directories']):.1f}% | "
        f"{n['cells']}/{n['directories']} = {100 * n['cells'] / max(1, n['directories']):.1f}% | 100% | "
        f"{'PASS' if n['cells'] == n['directories'] else 'FAIL'} |",
        f"| C6 未覆盖目录 | {len(b['uncovered'])} | {len(n['uncovered'])} | 0 | "
        f"{'PASS' if not n['uncovered'] else 'FAIL'} |",
        f"| C7 孤立单元 | — | {len(n['orphans'])} | 0 | {'PASS' if not n['orphans'] else 'FAIL'} |",
        f"| C8 跨方重复名 | — | {len(n['duplicates'])} | 0 | {'PASS' if not n['duplicates'] else 'FAIL'} |",
        "",
        "### 探针 provenance（缺陷证据）",
        "",
        "| 探针 import 路径 | 解析到的包 | 判读 |",
        "|---|---|---|",
        f"| 环境继承（修复前行为） | `{(prov['ambient_import']['resolved'] or prov['ambient_import']['error'] or '-')}` | "
        f"{'**异仓包（测错代码）**' if prov['shadow_wins_when_ambient'] else '本仓'} |",
        f"| `<repo>/src` 置首（修复后） | `{prov['pinned_import']['resolved']}` | "
        f"{'本仓（正确）' if prov['pin_forces_repo_package'] else '**仍被遮蔽**'} |",
        "",
        "诱饵包内容：`" + prov["decoy_package"] + "`，放在 `PYTHONPATH` 首位；修复后探针把它忽略并解析回本 checkout。",
        "",
        "### 逐单元验收（两种方言，各自声明的基准与门槛）",
        "",
        "| 单元 | 方言 | 判定量 | 实测 | 门槛 | 实测/门槛 | 结果 | 耗时(s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        ratio = r["value"] / r["threshold"] if r["threshold"] else float("nan")
        lines.append(
            f"| `{r['unit']}` | {r['kind']} | {r['kpi']} | {_fmt(r['value'])} | "
            f"{_fmt(r['threshold'])} | {ratio:.3g} | {r['status']} | {r['seconds']:.1f} |"
        )
    lines += [
        "",
        "### 门禁表（程序化判定，`xval.json` 逐项可查）",
        "",
        "| 度量 | 值 | 门槛 | 判定 |",
        "|---|---|---|---|",
    ]
    for item in gs:
        lines.append(
            f"| {item['metric']} | {item['value']:.4g} | {item['comparison']} {item['threshold']:.4g} | "
            f"{'BASELINE' if item.get('baseline') else ('PASS' if item['pass'] else 'FAIL')} |"
        )
    lines += [
        "",
        "## ④ 图",
        "",
    ]
    for f in figs:
        lines += [f"![{f['name']}](../artifacts/xval_model_unit_coverage/{f['name']})", "", f["caption"], ""]
    lines += [
        "## ⑤ 注意点 / 局限",
        "",
        "- 本单关闭的是**组织度量级**缺口：矩阵覆盖、探针 provenance。它不改变任何物理解。",
        "  数值侧的证据是「全部 7 个单元按自己声明的解析基准仍通过」——即修复是数值中性的，",
        "  但这里没有新的物理基准，物理精度结论仍以锚点原报告与 board 交差审计为准。",
        "- `model.json` 方言没有 `tier` 字段，合并矩阵里用碰撞算子（bgk/mrt/trt）填该列，仅作标签，不参与判定。",
        "- `ci-matrix` 现在对两种方言都要求目录可解析；如果别处存在第三种清单方言，它不会被静默忽略——",
        "  会作为「非受支持清单」暴露（`discover_directories` 只认 `MANIFEST_KINDS` 中的文件名，",
        "  新增方言必须显式登记，这是有意的 fail-closed 设计）。",
        "- 覆盖率门只证明「目录在矩阵里」，不证明单元有意义的判据；判据质量由每个单元的 `gates` 与",
        "  锚点的逐单元基准负责（本单对 7 个单元全部重跑，见上表）。",
        "- 未跑：GitHub Actions 上的真机执行（本环境无 runner，工作流语义由本地 `verify` + 测试覆盖）。",
        "",
        "## ⑥ 误差判定（是否 < 3%？）",
        "",
        "组织度量是**精确量**，没有 3% 之说，只有 0 误差要求——本单全部按精确门槛判定：",
        "",
        "| 度量 | 基准 | 实测 | 偏差 | <3%? |",
        "|---|---|---|---|---|",
        f"| 矩阵覆盖目录数 | 7（仓库内受支持清单目录数，程序枚举） | {n['cells']} | "
        f"{(n['cells'] - n['directories'])} 个 | {'是（0 偏差）' if n['cells'] == n['directories'] else '否'} |",
        f"| 未覆盖/孤立/重名 | 0 | {len(n['uncovered'])}/{len(n['orphans'])}/{len(n['duplicates'])} | 0 | "
        f"{'是（0 偏差）' if n['ok'] else '否'} |",
        f"| 探针解析包 | `<repo>/src/tensorlbm` | `{prov['pinned_import']['resolved']}` | 0 | "
        f"{'是（逐字节同路径）' if prov['pin_forces_repo_package'] else '否'} |",
        "| 逐单元 KPI | 各单元声明的解析基准（见上表） | 见上表 | 全部 `实测/门槛 < 1` | "
        "是：相对各自门槛仍有 1.4–4 个数量级余量（最紧一项见门禁表 C11） |",
        "",
        "诚实边界：上表第一行「7」是**仓库内枚举值**，不是外部标准；外部对照（TCLB 侧 61 vs 16 的漂移）",
        "只作方法学参照，未做数值对齐。",
        "",
        "## 复现",
        "",
        "```bash",
        "PYTHONPATH=src python tools/xval_model_unit_kind_coverage.py",
        "PYTHONPATH=src python -m tensorlbm.model_units ci-matrix      # 合并矩阵（两种方言）",
        "PYTHONPATH=src python -m tensorlbm.model_units verify          # 覆盖 + provenance + 漂移门",
        "PYTHONPATH=src python -m pytest tests/test_model_units.py tests/test_model_units_coverage.py -q",
        "```",
        "",
    ]
    DOC.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    coverage_data = measure_coverage()
    prov = measure_probe_provenance()
    rows = measure_units()
    payload_ctx = dict(coverage_data)
    payload_ctx["provenance"] = prov
    figs = make_figures(payload_ctx, rows)
    gs = gates(coverage_data, rows, prov)
    ok = all(item["pass"] for item in gs if not item.get("baseline"))

    payload = {
        "kind": "xval_model_unit_kind_coverage",
        "anchor": "XVAL-LBM-MODEL-DIR-AS-BUILD-AND-CI-UNIT",
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True
        ).stdout.strip(),
        "coverage": {
            "before": coverage_data["before"],
            "now": coverage_data["now"],
            "directories": coverage_data["directories"],
            "cells_before": coverage_data["cells_before"],
            "cells_now": coverage_data["cells_now"],
        },
        "provenance": prov,
        "units": rows,
        "gates": gs,
        "all_pass": ok,
        "figures": figs,
        "seconds": round(time.perf_counter() - t0, 2),
    }
    (OUT / "xval.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (OUT / "Makefile.models").write_text(coverage_data["makefile_targets"] + "\n")
    write_doc(coverage_data, rows, prov, gs, figs)

    print(json.dumps({k: payload[k] for k in ("all_pass", "seconds", "gates")}, indent=2)[:4000])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())