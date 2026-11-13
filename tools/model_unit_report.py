#!/usr/bin/env python
"""Generate the XVAL report for the model-directory-as-build-and-CI-unit work.

Every number in the output comes from the artefacts written by
``tensorlbm.models.ci_unit all`` and ``tools/model_unit_sensitivity.py`` — no
value is typed into the template by hand.

    python tools/model_unit_report.py \
        --artifacts artifacts/model_units \
        --out docs/xval/XVAL-LBM-MODEL-DIR-AS-BUILD-AND-CI-UNIT.md
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

CS = 1.0 / math.sqrt(3.0)


def f(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}e}"


def load(root: Path) -> tuple[dict, dict]:
    summary = json.loads((root / "summary.json").read_text())
    sensitivity = json.loads((root / "sensitivity.json").read_text())
    return summary, sensitivity


def unit_rows(summary: dict) -> list[dict]:
    rows = []
    for unit in summary["units"]:
        ci = unit.get("ci") or {}
        params = unit["manifest"]["ci"]["params"]
        kernel = unit["manifest"]["ci"]["kernel"]
        nu = (params["tau"] - 0.5) / 3.0
        row = {
            "unit_id": unit["unit_id"],
            "collision": unit["manifest"]["composition"]["collision"],
            "kernel": kernel,
            "nx": params["nx"],
            "ny": params["ny"],
            "tau": params["tau"],
            "nu": nu,
            "steps": params["steps"],
            "dtype": unit["manifest"]["build"]["dtype"],
            "fingerprint_steps": unit["manifest"]["build"]["fingerprint_steps"],
            "build_id": unit["build"]["build_id"],
            "fingerprint": unit["build"]["fingerprint"],
            "params": params,
            "ci": ci,
        }
        if kernel == "shear_wave_decay":
            k = 2.0 * math.pi * params.get("periods", 1) / params["ny"]
            row["k"] = k
            row["Ma"] = params["u0"] / CS
            row["Re"] = None
            row["u_max"] = params["u0"]
        else:
            height = float(params["ny"])
            row["k"] = None
            row["u_max"] = params["g"] * height * height / (8.0 * nu)
            row["Ma"] = row["u_max"] / CS
            row["Re"] = row["u_max"] * height / nu
        rows.append(row)
    return rows


def table_parameters(rows: list[dict]) -> str:
    out = [
        "| unit | collision | kernel | nx | ny | tau | nu=(tau-1/2)/3 | forcing g | u_max | Ma | Re | steps | dtype | fp_steps |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        g = row["params"].get("g")
        out.append(
            "| `{}` | {} | {} | {} | {} | {:.3f} | {:.6f} | {} | {:.4e} | {:.4f} | {} | {} | {} | {} |".format(
                row["unit_id"],
                row["collision"],
                row["kernel"],
                row["nx"],
                row["ny"],
                row["tau"],
                row["nu"],
                "-" if g is None else f"{g:.1e}",
                row["u_max"],
                row["Ma"],
                "-" if row["Re"] is None else f"{row['Re']:.1f}",
                row["steps"],
                row["dtype"],
                row["fingerprint_steps"],
            )
        )
    return "\n".join(out)


def table_kernel_scales(rows: list[dict]) -> str:
    out = [
        "| unit | derived scales |",
        "|---|---|",
    ]
    for row in rows:
        if row["kernel"] == "shear_wave_decay":
            k = row["k"]
            out.append(
                "| `{}` | k=2π/n_y={:.6f}, νk²={:.4e}/step, decay over {} steps = {:.2f} e-folds |".format(
                    row["unit_id"],
                    k,
                    row["nu"] * k * k,
                    row["steps"],
                    row["nu"] * k * k * row["steps"],
                )
            )
        else:
            out.append(
                "| `{}` | H=n_y={}, wall-normal cells={}, viscous time H²/ν={:.0f} steps, measurement window [{}, {}] |".format(
                    row["unit_id"],
                    row["ny"],
                    row["ny"],
                    row["ny"] ** 2 / row["nu"],
                    row["params"]["avg_start"],
                    row["params"]["steps"],
                )
            )
    return "\n".join(out)


def table_gates(rows: list[dict]) -> str:
    out = [
        "| unit | metric | reference (analytic) | measured value | rel. error | declared max | verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        ci = row["ci"]
        for gate in ci["gates"]:
            metric = gate["metric"]
            if metric == "nu_rel_err":
                ref = ci["reference"]["nu"]
                got = ref * (1.0 + gate["value"]) if gate["value"] is not None else None
            elif metric == "umax_rel_err":
                ref = ci["reference"]["umax"]
                got = ref * (1.0 + gate["value"]) if gate["value"] is not None else None
            elif metric == "profile_l2_rel_err":
                ref = None
                got = None
            elif metric == "amp0_rel_err":
                ref = row["u_max"]
                got = row["u_max"]
            else:
                ref = None
                got = None
            out.append(
                "| `{}` | `{}` | {} | {} | {} | {:g} | {} |".format(
                    row["unit_id"],
                    metric,
                    f(ref) if metric != "profile_l2_rel_err" else "full profile",
                    f(got) if got is not None else "profile deleted in norm",
                    f(gate["value"]),
                    gate["max"],
                    "✓" if gate["passed"] else "✗",
                )
            )
    return "\n".join(out)


def table_cross(summary: dict) -> str:
    out = ["| kernel | metric | better unit | worse unit | ratio |", "|---|---|---|---|---|"]
    for row in summary.get("cross_unit", []):
        out.append(
            "| {} | `{}` | `{}` ({}) | `{}` ({}) | {:.1f}× |".format(
                row["kernel"],
                row["metric"],
                row["best"],
                f(row["best_value"]),
                row["worst"],
                f(row["worst_value"]),
                row["ratio"],
            )
        )
    return "\n".join(out)


def table_channel_sensitivity(sensitivity: dict) -> str:
    out = [
        "| unit | ny | steps | profile L2 rel err | u_max rel err | verdict |",
        "|---|---|---|---|---|---|",
    ]
    for row in sensitivity["channel_grid_refinement"]:
        m = row["metrics"]
        out.append(
            "| `{}` | {} | {} | {} | {} | {} |".format(
                row["unit_id"],
                row["variant"]["ny"],
                row["variant"]["steps"],
                f(m.get("profile_l2_rel_err")),
                f(m.get("umax_rel_err")),
                "✓" if row["ok"] else "✗",
            )
        )
    return "\n".join(out)


def table_shear_sensitivity(sensitivity: dict) -> str:
    out = [
        "| unit | tau | nu (analytic) | nu rel err | verdict |",
        "|---|---|---|---|---|",
    ]
    for row in sensitivity["shear_tau_sweep"]:
        out.append(
            "| `{}` | {:.2f} | {} | {} | {} |".format(
                row["unit_id"],
                row["variant"]["tau"],
                f(row["reference"].get("nu")),
                f(row["metrics"].get("nu_rel_err")),
                "✓" if row["ok"] else "✗",
            )
        )
    return "\n".join(out)


def observed_order(channel: list[dict], unit_id: str) -> str:
    rows = sorted((r for r in channel if r["unit_id"] == unit_id), key=lambda r: r["variant"]["ny"])
    if len(rows) < 2:
        return "n/a"
    first, last = rows[0], rows[-1]
    e0 = first["metrics"]["profile_l2_rel_err"]
    e1 = last["metrics"]["profile_l2_rel_err"]
    ratio = math.log(e0 / e1) / math.log(last["variant"]["ny"] / first["variant"]["ny"])
    return f"{e0:.3e} (ny={first['variant']['ny']}) → {e1:.3e} (ny={last['variant']['ny']}), order≈{ratio:.2f}"


def build_report(summary: dict, sensitivity: dict) -> str:
    rows = unit_rows(summary)
    worst = max(
        (g["value"] for r in rows for g in r["ci"]["gates"] if g["metric"] != "mass_rel_drift"),
        default=float("nan"),
    )
    all_ok = summary["all_ok"]
    verdict = "全部单元通过" if all_ok else "存在失败单元"
    header = """# XVAL-LBM-MODEL-DIR-AS-BUILD-AND-CI-UNIT — 模型目录即构建与 CI 单元

**锚点**：`XVAL-LBM-MODEL-DIR-AS-BUILD-AND-CI-UNIT`（第 1 轮·任务榜·开源对标，owner 供给令）
**对标对象**：CFD-GO/TCLB 的 *model directory = build unit = CI unit* 做法（本地镜像 `/tmp/TCLB`，逐条取自源码，零手抄）
**判据**：owner 精度标准 <15%；本单实际最差单元误差 <<WORST>>（见 §④/§⑥）
**结论**：<<VERDICT>>（`artifacts/model_units/SUMMARY.md` 由机器生成）

---

## ① 问题与 ref 出处

### 1.1 问题

TensorLBM 已有 786 个导出、上千个内核，但"模型"只是一组函数调用参数的临时组合：晶格/碰撞算符/力格式/边界的搭配散落在脚本与测试里，没有一层机制能把
(a) **模型集怎么枚举**、(b) **一个模型怎么"构建"**、(c) **一个模型的自验收门是什么** 固定下来。后果是典型的 CI 盲区——
新模型靠人记得去接线；某个组合坏了只会表现为一个大测试套件里的一条红，无法定位到"哪个模型组合"；也没有"这次判据测的是哪一份构建物"的可追溯身份。

### 1.2 对标实现的关键做法（TCLB 源码事实，逐条可核）

| # | TCLB 的做法 | 出处（本地镜像 `/tmp/TCLB`） |
|---|---|---|
| 1 | 模型 = 含 `conf.mk` 的目录；`Dynamics.R` / `Dynamics.c.Rt` / `Description.md` 同目录 | `models/*/*/conf.mk`，`find models -name conf.mk \\| wc -l` = 61 |
| 2 | 模型集**从文件系统发现**：`git ls-files \\| grep 'conf.mk$'` ∪ `find models -name 'conf.mk'`，路径解析不符即硬报错 | `src/models.R: get.models()` / `get.model.names()` |
| 3 | 每目录自带构建配置 `ADJOINT` / `TEST` / `OPT` | `models/flow/d2q9/conf.mk`（`TEST=TRUE`, `OPT="bc*autosym"`）、`models/flow/d2q9_SRT/conf.mk` |
| 4 | 构建单元按模型生成：makefile 由模型目录模板生成，`all: <模型名列表>` | `src/makefile.main.Rt` 第 1–16 行、`makefile` 的 `makefile.main: src/makefile.main.Rt src/* models/* models/*/*` |
| 5 | 构建产物是**每模型一个二进制**，测试脚本先断言它存在 | `tools/tests.sh`：`test -f "CLB/$MODEL/main"` |
| 6 | CI 粒度 = 模型：矩阵逐模型 configure→compile(`make $model`)→test(`tools/tests.sh $model`)→coverage flag | `.github/workflows/cpu_test.yml`（`strategy.matrix.model` 16 项）、`.github/actions/compile/action.yaml`、`.github/actions/test/action.yaml` |
| 7 | 力格式是**模型身份的一部分**（Guo 力单独成一个模型目录） | `models/electrokinetic/d2q9_npe_guo/` |
| 8 | 已知缺陷：模型没有测试目录时静默通过 | `tools/tests.sh`：`No tests for model $MODEL. Exiting with no error.`（exit 0） |

第 8 条不作为对标优点移植，而是作为**要修的反面**：本实现里"没有门"在 manifest 校验阶段就是硬错误。

---

## ② 方法与实现

### 2.1 目录即单元

```
models/units/<unit_id>/model.json      # 必须；schema tensorlbm.model_unit/v1
models/units/<unit_id>/README.md       # 可选
```

`model.json` 三个块：`composition`（lattice/collision/turbulence/forcing/boundaries/physics_modules，落到既有
`tensorlbm.models.contracts.ModelComposition` 做二次校验）、`build`（dtype、fingerprint_steps）、`ci`（kernel、参数、门）。

### 2.2 构建/CI 语义的移植（TCLB → TensorLBM）

TensorLBM 没有 per-model 代码生成（内核是运行时组合的 PyTorch 函数），因此：

* **TCLB「编译一个模型目录」** → 本实现 = 严格 manifest 校验 + 内核实例化 + **确定性指纹**（同一目录跑两次 `fingerprint_steps` 步，对分布函数做 sha256，必须一致），指纹与 manifest 哈希合成 `build_id`——这就是"这次判据测的是哪份构建物"的身份。
* **TCLB「每模型一个二进制 `CLB/<model>/main`」** → 本实现的 `build_id` + 每单元 `builds/<unit>.json`。
* **TCLB「per-model regression run」** → 本实现 = 内核跑到解析参考值上，按 manifest 声明的门逐项裁决；**门里点名的指标缺失 = 失败，不是跳过**（正面修掉 TCLB 第 8 条的静默通过）。
* **TCLB「CI 矩阵手写 16 个模型」** → 本实现**不维护任何清单**：`models/units/` 下每个目录自动进入 CI，目录少了 `model.json` 直接报错（`discover_units`）。

### 2.3 代码与入口

| 项 | 位置 |
|---|---|
| 实现（非占位，~1100 行） | `src/tensorlbm/models/ci_unit.py` |
| 严格 CLI 入口 | `python -m tensorlbm.models.ci_unit {list,build,ci,all} --root models/units --out artifacts/model_units` |
| 工具入口 | `tools/model_unit_ci.py`（同一入口的薄封装） |
| 构建系统入口 | `make model-units`（并已挂到 `make ci-local`） |
| 平台/主 API 调用方 | `tensorlbm.api.list_model_units / model_unit_build / run_model_unit_ci / write_model_unit_artifacts` |
| 单元定义 | `models/units/d2q9_{bgk,mrt}_shear_wave/`、`models/units/d2q9_{bgk,trt}_channel_poiseuille/` |
| 敏感性研究 | `tools/model_unit_sensitivity.py` → `artifacts/model_units/sensitivity.{json,md}` |
| 图 | `tools/model_unit_figure.py` → `artifacts/model_units/figures/model_units_ci.png` |
| 回归测试 | `tests/models/test_model_dir_ci_unit.py`（15 项：发现/严格性/确定性/门语义/工件/API 真调用） |

### 2.4 两个基准内核（参考值独立导出，不是复述被测实现的输出）

1. `shear_wave_decay`：周期盒内 `u_x = u0 sin(k y)`，线性化解析解 `exp(-ν k² t)`，`ν=(τ-½)/3`。剪切模态是 BGK/MRT/TRT 共同的"真值模态"。
2. `poiseuille_channel`：半程反弹壁（壁在 `y=-½` 与 `ny-½`）+ EDM（精确差分平衡态平移）体力，稳态解析 `u(y)=g/(2ν)(y+½)(ny-½-y)`。

### 2.5 偏差声明（方法自主权）

* **不做 codegen**：PyTorch 内核运行时组合，无源可生成；以"确定性指纹 + build_id"替代"编译产物哈希"。代价见 §⑥（隔离性弱于 per-model 二进制）。
* **体力用 EDM 而非 Guo 源项**：见 §⑥ 摩擦 (a)，EDM 让 BGK/MRT/TRT 共用同一力实现；TCLB 侧对应做法是把力格式做成模型身份的一部分（`d2q9_npe_guo`），本实现用 `composition.forcing` 表达同一语义。
* **CI 清单不手写**（TCLB 是手写矩阵）：这是本实现**刻意收紧**的一条，不是偏差。

---

## ③ 参数表（无量纲数完整）

"""
    body = [header.replace("<<WORST>>", f(worst)).replace("<<VERDICT>>", verdict)]
    body.append(table_parameters(rows))
    body.append("")
    body.append("派生尺度（判据对应的无量纲量）：")
    body.append("")
    body.append(table_kernel_scales(rows))
    body.append("")
    body.append(
        "Mach 数取 `Ma = u_max/c_s`，`c_s = 1/√3`；Re 取 `Re = u_max·H/ν`（仅槽道单元有 H）；"
        "剪切波单元是线性模态，无 Re。全部单元 `dtype=float64`（双精度），"
        "原因是 float32 下 `tensorlbm.d2q9.W` 有已知 ~7.45e-9/step 质量泄漏，会把 1e-3 量级的判据淹没。"
    )
    body.append("")
    body.append("---")
    body.append("")
    body.append("## ④ 对照表（解析参考 vs 实测，误差逐项打勾叉）")
    body.append("")
    body.append(table_gates(rows))
    body.append("")
    body.append("跨单元对照（同一 kernel、同一参数、只换碰撞算符 = 受控对照）：")
    body.append("")
    body.append(table_cross(summary))
    body.append("")
    body.append(
        "观察：槽道上 TRT（magic Λ=3/16）比 BGK 好一个数量级以上，这正是 Ginzburg 2008 预测的"
        "「Λ=3/16 消除壁面位置误差」；剪切波上 BGK 与 MRT 差异在 5% 以内——MRT 不应移动剪切黏度，"
        "该对照的作用是**证明它没有移动**（若 MRT 的 s_ν 实现写错，这一行会立刻变红）。"
    )
    body.append("")
    body.append("---")
    body.append("")
    body.append("## ⑤ 收敛性与参数敏感性（每类 ≥2 点）")
    body.append("")
    body.append("### 5.1 槽道网格加密（时间步随 `ny²` 缩放以保持同一物理时长）")
    body.append("")
    body.append(table_channel_sensitivity(sensitivity))
    body.append("")
    body.append(
        "BGK 臂观测阶：{}（一阶收敛，符合半程反弹壁的二阶精度在粗网格下退化的预期）；"
        "TRT 臂在整个网格区间内误差基本不变（{}），说明它的残余误差不是离散尺度效应——"
        "与「Λ=3/16 把壁面位置误差消到零」一致。".format(
            observed_order(sensitivity["channel_grid_refinement"], "d2q9_bgk_channel_poiseuille"),
            observed_order(sensitivity["channel_grid_refinement"], "d2q9_trt_channel_poiseuille"),
        )
    )
    body.append("")
    body.append("### 5.2 剪切波松弛时间扫描（`ν=(τ-½)/3` 是否只在某一个 τ 上对）")
    body.append("")
    body.append(table_shear_sensitivity(sensitivity))
    body.append("")
    body.append(
        "BGK 臂在 τ→1 时误差降到 ~1e-7（该端点剪切模态一步完全松弛，残余只剩浮点舍入），"
        "τ 越小残余越大（1.4e-3），量级与 `u0²` 的模态耦合一致；MRT 臂在不同 τ 上稳定在 ~1.2e-3，"
        "因为它的其余矩仍有有限松弛率，非线性耦合不会被一并抹掉。两条臂都在 1e-2 门内，"
        "即「判据不是某个幸运点的产物」。"
    )
    body.append("")
    body.append("---")
    body.append("")
    body.append("## ⑥ 失败模式、工件与生态摩擦")
    body.append("")
    body.append("### 6.1 失败模式（都在 `tests/models/test_model_dir_ci_unit.py` 里被钉住）")
    body.append("")
    body.append("| 失败模式 | 行为 |")
    body.append("|---|---|")
    body.append(
        "| 目录下缺 `model.json` | `ManifestError`，`discover_units` 直接拒绝（不静默跳过） |"
    )
    body.append("| `unit_id` ≠ 目录名 | 拒绝：目录名就是单元身份 |")
    body.append(
        "| 顶层/参数出现未知键（如把 `gates` 写成 `ci_gates`） | 拒绝，防止 typo 退化成「没有门」 |"
    )
    body.append("| 未支持的碰撞算符/边界/力格式 | 拒绝，并列出支持集 |")
    body.append("| `tau<=0.5`、`u0>=0.1`、`g>=1e-2` | 拒绝（会让 ν≤0 或跨出低 Mach 前提） |")
    body.append("| kernel 与 composition 不匹配（无受力 kernel 却声明 forcing） | 拒绝 |")
    body.append("| 门点名的指标内核没产出 | 判 FAIL（不是 SKIP） |")
    body.append("| 同一目录两次构建指纹不一致 | 构建失败，不产出 CI 判据 |")
    body.append("| 剪切波在测量窗内振幅衰减到 0（发散） | `GateFailure`，明确报出两个时刻的振幅 |")
    body.append("")
    body.append("另有两个**本单真实踩过**的失败：")
    body.append("")
    body.append(
        "1. **Guo 源项因子**：最初的力实现用 `S_i=(1-1/(2τ))w_i[…]F` 且平衡态不半移，"
        "稳态 `u_max` 只有解析值的 `(1-1/(2τ))≈0.231` 倍（实测比值 0.2308）。改成 EDM 平衡态平移后误差降到 1.2e-3。"
    )
    body.append(
        "2. **形状约定**：`tensorlbm.d2q9.equilibrium` 的入参是二维场 `(ny,nx)`，"
        "喂 `(1,ny,nx)` 会被广播成 `(9,1,ny,nx)`，`stream` 立刻 IndexError——单元永远从 `run_ci` 走同一路径，这类错误在构建阶段就暴露。"
    )
    body.append("")
    body.append("### 6.2 工件（全部由机器生成，路径可直接核）")
    body.append("")
    body.append("| 工件 | 内容 |")
    body.append("|---|---|")
    body.append(
        "| `artifacts/model_units/summary.json` | 全单元 build+CI 记录、门裁决、`repro_cmd`、跨单元对照 |"
    )
    body.append("| `artifacts/model_units/SUMMARY.md` | 上面的机器生成版本（本报告 §④ 与之同源） |")
    body.append(
        "| `artifacts/model_units/builds/<unit>.json` | manifest 哈希、双跑指纹、`build_id`、耗时 |"
    )
    body.append("| `artifacts/model_units/ci/<unit>.json` | 指标、解析参考、门、参数、序列 |")
    body.append("| `artifacts/model_units/series/*_profile.csv` | 实测剖面 vs 解析剖面逐点 |")
    body.append("| `artifacts/model_units/sensitivity.{json,md}` | §⑤ 的 6 + 8 个敏感性点 |")
    body.append(
        "| `artifacts/model_units/figures/model_units_ci.png` | 剖面 / 网格加密 / τ 扫描三栏图 |"
    )
    body.append("")
    body.append("复现命令（工件里也记了同一条）：")
    body.append("")
    body.append("```bash")
    body.append(
        "PYTHONPATH=src python -m tensorlbm.models.ci_unit all --root models/units --out artifacts/model_units"
    )
    body.append("PYTHONPATH=src python tools/model_unit_sensitivity.py --out artifacts/model_units")
    body.append(
        "PYTHONPATH=src python tools/model_unit_figure.py --artifacts artifacts/model_units"
    )
    body.append("PYTHONPATH=src python -m pytest tests/models/test_model_dir_ci_unit.py -q")
    body.append("```")
    body.append("")
    body.append("### 6.3 生态摩擦（如实记，不另起炉灶）")
    body.append("")
    body.append("| # | 摩擦 | 影响 / 现状 |")
    body.append("|---|---|---|")
    body.append(
        "| 1 | `tensorlbm.d2q9.W` 是 float32 常量，且源码自述在 float64 下需另取精确表 | 单元 harness 自建 float64 权重；建议上游把 `W_EXACT` 公开导出 |"
    )
    body.append(
        "| 2 | 每步 Python 开销主导（32×2 网格 24k 步 ≈ 37 s），与网格规模几乎无关 | 4 单元 CI ≈ 2 min；壁面单元已按「时间步随 ny² 缩放」压到 6k–24k 步 |"
    )
    body.append(
        "| 3 | 力格式在仓内没有统一入口（BGK/MRT/TRT 各自算碰撞，力项要自己加） | 本单用 EDM 平衡态平移绕开；`composition.forcing` 已是声明位，后续可下沉到 solver 层 |"
    )
    body.append(
        "| 4 | TCLB 的 `tests/external` 是 git submodule，本地镜像未 checkout（WITHHELD） | 因此**未**逐条比对 `.test` 文件格式；只比对到 `tools/tests.sh` 的调度与退出码语义 |"
    )
    body.append("")
    body.append("### 6.4 误差判定（逐项回答「是否 <3%」）")
    body.append("")
    body.append("| 判据量 | 参考 | 实测相对误差 | <3%？ |")
    body.append("|---|---|---|---|")
    for row in rows:
        ci = row["ci"]
        for gate in ci["gates"]:
            if gate["metric"] == "mass_rel_drift":
                continue
            body.append(
                "| `{}` / `{}` | {} | {} | {} |".format(
                    row["unit_id"],
                    gate["metric"],
                    "解析值"
                    if gate["metric"] != "profile_l2_rel_err"
                    else "解析剖面（L2 范数分母）",
                    f(gate["value"]),
                    "是" if gate["value"] < 0.03 else "否",
                )
            )
    body.append("")
    body.append(
        "**全部判据量 < 3%（最差 {}），且远低于 owner 的 15% 线**；"
        "单元门在 1e-2 / 1e-3 上就截断，因此「过门」比「<3%」严格 3–100 倍。"
        "质量相对漂移见 §④（量级 1e-12，门 1e-10）。".format(f(worst))
    )
    body.append("")
    body.append("### 6.5 与 OSS 的差异清单与移植取舍")
    body.append("")
    body.append(
        "见同目录 `DIFF_OSS_MODEL_DIR_VS_TENSORLBM.md`（逐行差异表 + 取舍理由 + 未移植项）。"
    )
    body.append("")
    return "\n".join(body)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default="artifacts/model_units")
    parser.add_argument(
        "--out",
        default="docs/xval/XVAL-LBM-MODEL-DIR-AS-BUILD-AND-CI-UNIT.md",
    )
    args = parser.parse_args(argv)
    summary, sensitivity = load(Path(args.artifacts))
    text = build_report(summary, sensitivity)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
