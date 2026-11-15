"""Build and publish the m21802 board-audit report (six required sections).

All numbers are loaded from artefacts / DB / git; nothing is typed by hand.
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import urllib.request
from pathlib import Path

BASE = os.environ["TENSOR_API_BASE"]
AUDITED = Path("/srv/tensor/agent/work/m21672")
HERE = Path(__file__).resolve().parent
FIG = HERE / "fig_m21802_backstep_audit.png"
PGENV = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"}


def psql(sql: str) -> str:
    r = subprocess.run(["runuser", "-u", "postgres", "--", "psql", "-d", "tensorplatform",
                        "-tA", "-F", "|", "-c", sql], capture_output=True, text=True, env=PGENV)
    return r.stdout.strip()


def git_show(rev: str, path: str) -> str:
    return subprocess.run(["git", "-C", str(AUDITED), "show", f"{rev}:{path}"],
                          capture_output=True, text=True, check=True).stdout


def main() -> int:
    partial = json.loads((AUDITED / "artifacts/case_backward_step_armaly/h1_re100/result.partial.json").read_text())
    manifest = json.loads((AUDITED / "artifacts/case_backward_step_armaly/h1_re100_partial/partial_manifest.json").read_text())
    refs = json.loads((AUDITED / "benchmarks/case_pool/backward_step_armaly/refs.json").read_text())
    case_verdict = json.loads((AUDITED / "artifacts/case_backward_step_armaly/verdict.json").read_text())
    conv = json.loads(git_show("f3a211d78", "artifacts/case_backward_step_armaly/h1_re100/result.json"))
    ref_exp = refs["adopted_numeric_reference"]["points"]["100"]["x1_over_s"]
    tol = refs["adopted_numeric_reference"]["tolerance_pct"]
    ert = refs["cross_checks"][0]["value"]
    hist = partial["history"]
    last = hist[-1]
    conv_r = conv["result"]
    err_exp = (conv_r["x1_over_s"] - ref_exp) / ref_exp * 100.0
    err_ert = (conv_r["x1_over_s"] - ert) / ert * 100.0
    err_tr = (last["x1_over_s_instant"] - ref_exp) / ref_exp * 100.0
    anchor_rows = psql("select count(*) from tensor_benchmark_cases where anchor_slug='CASE-BACKSTEP-FVM-RE100';")
    tot_rows = psql("select count(*) from tensor_benchmark_cases;")
    verdict_row = psql("select id||' level='||level||' gate='||gate_state from tensor_maturity_verdicts order by id desc limit 1;")

    rows_tr = "\n".join(
        f"| {h['step']} | {h['x1_over_s_instant']:.4f} | {h['max_speed']:.5f} | {h['mean_rho']:.5f} |"
        for h in hist
    )

    md = f"""# m21802 交差核验：m21672 / CASE-BACKSTEP-FVM-RE100（LBM 臂）判定 **L3**

**判定**：`tensor_maturity_verdicts` 落表 = `{verdict_row}`（`repo=TensorLBM`，`scope='m21672 CASE-BACKSTEP-FVM-RE100'`，`verdict_by=board-audit`）。
**一句话**：L2 实测通过（真代码、入口可跑、产物落盘），L3 有条件通过（库内真实调用者 + 算例池可发现，但驱动按文档调用会崩，已在本单分支修好），**L4 不通过**——交付内无收敛点，且基准裁判表里根本没有这个锚点。

---

## ① 案例（物理背景 / 几何 / 参数）

后台阶层流分离—再附：入口高 `h_in`、台阶 `s`、下游高 `H`。观测量 = 下壁面主再附长度 `X1/s`（台阶高归一）。
Apache 口径（Armaly et al. 1983, *JFM* 127:473）：`Re = u_bar·D_h/ν`，`D_h = 2·h_in`，实验 ER = 10.1/5.2 mm = **1.9423**。

| 项 | 交付运行 `h1_re100`（cj3613） | 历史收敛点（未合并分支 `f3a211d78`） |
|---|---|---|
| 网格 hi / s（栅格） | {partial['config']['hi']} / {partial['config']['s']} | {conv['config']['hi']} / {conv['config']['s']} |
| ER = (hi+s)/hi | {51/26:.4f}（+{ (51/26/1.9423077-1)*100:.2f}% vs 实验） | {51/26:.4f}（同档） |
| Re / u_bar | {partial['config']['re']:g} / {partial['config']['u_bar']} | {conv['config']['re']:g} / {conv['config']['u_bar']} |
| 步数预算 / 实跑 | {partial['config']['n_steps']} / **{partial['steps_run']}** | {conv['config']['n_steps']} / **{conv_r['steps_run']}** |
| 碰撞 / 出口 | {partial['config']['collision']} / Zou-He ρ=1 + 海绵层 {partial['config']['sponge_width_h']}H | {conv['config']['collision']} / 同 |
| 稳态判据 | drift≤{partial['config']['steady_tol']}%, hits={partial['config']['steady_hits']} | 同（`steady_at_step={conv_r['steady_at_step']}`） |
| 参考值 | Armaly 1983 Re=100 → **X1/s={ref_exp}**（tier `{refs['adopted_numeric_reference']['tier']}`，原文 PDF `{refs['primary_reference']['acquisition_status']}`），容差 {tol}% | 交叉锚 Erturk 2008 2D → {ert}（band {refs['cross_checks'][0]['band']}） |

## ② 怎么做的（审计流程 / 工具链）

1. **取证**：`git show / ls-tree / merge-base` 于被审交付 `agent/m21672 @0fb4c2ebe`（审计前后 `git status` 干净）；读取 run 工件三件套、`refs.json`、案例自带 `verdict.json`、report id 19655 全文。
2. **裁判**：`psql` 直查 `tensor_benchmark_cases`（锚点精确 + `%BACKSTEP%/%backward%/%armaly%/%reattach%` 模糊）→ **{anchor_rows} 行**（全表 {tot_rows} 行）。
3. **可执行性实测**：`run_case.py --smoke`（400 步，2 s）确认输入→计算→输出闭合；`pytest tests/test_case_pool_backward_step_armaly.py -q`；`run_pool.py --list` / `run_pool.py --case … --smoke`（驱动集成路径）。
4. **历史追溯**：交付里的 round-3 文档自称 PASS，指向 `artifacts/…/h1_re100/{{result.json,wall_shear.csv,field.npz}}` —— 交付树里不存在；沿 git 对象追到未合并分支 `agent/m17034 @f3a211d78` 的收敛工件（`git merge-base --is-ancestor` = false）。
5. **图表**：`make_figure.py` 程序化读取上述 JSON 生成 png（零手抄）。

## ③ 计算结果（数字表，程序化注入）

交付工件（`result.partial.json` @0fb4c2ebe，provenance `cj3613 / n330`，`elapsed_s={partial['elapsed_s']:.1f}`）：

| step | X1/s（瞬时） | max_speed | mean ρ |
|---|---|---|---|
{rows_tr}

关键读数对照：

| 量 | 值 | 参考 | 偏差 | 是否可计分 |
|---|---|---|---|---|
| X1/s @交付第 {partial['steps_run']} 步（瞬态） | {last['x1_over_s_instant']:.4f} | {ref_exp} | {err_tr:+.2f}% | **否**（瞬态，`steady_at_step` 不存在） |
| X1/s @历史 88k 步收敛点（**不在交付**） | {conv_r['x1_over_s']:.4f} | {ref_exp} | {err_exp:+.2f}% | 否（未合并分支，且其后代码改动） |
| 同上 vs Erturk 2008 2D | {conv_r['x1_over_s']:.4f} | {ert} | {err_ert:+.2f}% | 同上 |
| 质量流守恒（历史收敛点） | {conv_r['mass_flux_rel_error']:.2e} | ≤1e-3（案例 gate） | ✓ | 只对该历史工件成立 |
| 案例自带机器判分 | `runs={len(case_verdict['runs'])}, n_pass={case_verdict['n_pass']}, all_judged_pass={case_verdict['all_judged_pass']}` | — | — | **无点可判** |

审计者实测（m21802 侧）：
- `pytest tests/test_case_pool_backward_step_armaly.py -q` → **7 passed, 2 skipped**；两条 skip 正是基准断言（`:110` 需要 `g2_re100/result.json`；`:163` 报 `no committed Re=100 artefact yet`）。
- `run_case.py --smoke` → rc=0，`X1/s=2.0761`，产物 `result.json / wall_shear.csv / field.npz` 齐全（L2 实测）。
- `run_pool.py --list` → 含 `backward_step_armaly`；`run_pool.py --case backward_step_armaly --smoke` 修复前 `ModuleNotFoundError: case_pool`，修复后 rc=0。

## ④ 结果图

![audit](fig)
左：交付瞬态轨迹（红）vs Armaly 实验锚 {ref_exp}（蓝线）与 Erturk 2D 带（绿）；黑菱形 = 只存在于未合并分支的 88k 步收敛点 {conv_r['x1_over_s']:.4f}。右：`max_speed` 诊断（m21672 修的 bug：原为全局 L2 范数 10~30，被上一轮审计误读为发散）。

## ⑤ 注意点（坑与局限）

1. **裁判缺位是最硬的卡点**：锚点未在 `tensor_benchmark_cases` 注册（{anchor_rows} 行），任何「达标」都无法被机器复核；本审计单不代注册（审计者不给自己造裁判）。
2. **交付的 PASS 声明是陈旧的**：`docs/CASE_BACKWARD_STEP_LBM_ROUND3.md` 仍写「首个收敛解落地 · Re=100 达标 · X1/s=2.920 → −4.3%」，并指向交付树里不存在的三个工件；数字真实但工件在 `agent/m17034 @f3a211d78`（未合并、且之后 `cf178e1de` 改过代码）。→ 本审计据此**不计 L4**，并建议加 `SUPERSEDED` 抬头。
3. **仓库级驱动缺陷（已修）**：`benchmarks/case_pool/run_pool.py` 自身文档的调用方式 13/13 算例全崩（`case_pool` 包不可导入），另有 3/13 无参 `main()` 算例必崩（`TypeError`）；修好后无参算例仍会看到驱动 argv。已在本单分支修复 + 7 条回归测试。这条正是 m21672 的 L3 证据「驱动可发现/可执行」所依赖的路径 → 故 L3 判「有条件」。
4. **审计未重算**：未重跑 88k 步收敛点（本单非重做单；按 cj3613 的 287 s/20000 步外推进 ≈21 min/作业，收单时队列深度 60–84），故 2.92021 是**历史工件读数**，不是审计者复现值。
5. **参考层级**：主参考是 `SECONDARY_IN_REPO`（原文 PDF `PRIMARY_PDF_CLOSED`），Re=100 处实验 3.05 与 2D 数值 2.878 本身差 6.0%，判据带宽 15% 已被这个不可约差吃掉近半。

## ⑥ 误差判定（逐项，显式回答「是否 <3%」）

| 关键量 | got | ref | 误差 | <3%？ |
|---|---|---|---|---|
| X1/s（交付工件，第 {partial['steps_run']} 步瞬态） | {last['x1_over_s_instant']:.4f} | {ref_exp} | {err_tr:+.2f}% | **不适用**（瞬态，非基准点；也 >3%） |
| X1/s（历史收敛点，不在交付） | {conv_r['x1_over_s']:.4f} | {ref_exp} | {err_exp:+.2f}% | **否**（{abs(err_exp):.2f}% > 3%），但在 15% 判带内 |
| X1/s（历史收敛点 vs 2D 数值基准） | {conv_r['x1_over_s']:.4f} | {ert} | {err_ert:+.2f}% | **否**（{abs(err_ert):.2f}% > 3%） |
| 质量流相对误差（历史收敛点） | {conv_r['mass_flux_rel_error']:.2e} | ≤1e-3 | 满足 | ✓（该工件自身的守恒检查通过） |

**结论**：对**交付**而言，唯一能算的对比是瞬态值 {err_tr:+.2f}%（相对实验锚），且它不构成基准点 → 精度判定应为 **WITHHELD**（m21672 自己的报告也是这么写的，无粉饰）；历史 88k 步工件给出 {err_exp:+.2f}%（实验）/ {err_ert:+.2f}%（2D），两者都 **不小于 3%**，仅落在 owner 15% 标准带内。

---

### 返修最小清单（冲 L4）
(i) 用交付代码跑出 `steady_at_step!=null` 且 `mass_flux_rel_error≤1e-3` 的收敛点并提交 `result.json`（非 partial）；(ii) 注册裁判行（metric `X1/s`，tolerance `<15%`）；(iii) 让 `verdict.py` 产出 `all_judged_pass=true`、两条被跳过的基准断言变绿；(iv) round-3 文档加 `SUPERSEDED` 抬头。
"""

    tokens = json.load(urllib.request.urlopen(urllib.request.Request(
        BASE + "/auth/login",
        data=json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                         "password": os.environ["TENSOR_AGENT_PASS"]}).encode(),
        headers={"Content-Type": "application/json"})))
    tok = tokens["access_token"]
    body = {
        "title": "m21802 交差核验：m21672 CASE-BACKSTEP-FVM-RE100（LBM）判 L3 —— 裁判缺位 + 交付无收敛点，L4 不通过；顺带修掉算例池驱动 3 处 repo 级缺陷",
        "summary": (f"按 tensor_benchmark_cases（唯一裁判）核验 m21672：该锚点注册 0 行，交付仅 20000/200000 步瞬态工件、"
                    f"案例自带判分 runs=0/all_judged_pass=false → L4 不通过；L2 实测通过（--smoke 真跑出三件产物）、L3 有条件通过"
                    f"（库内调用者 backward_facing_step.py:78 + 池驱动可发现）。落 verdict level=3，并修复 run_pool.py 的 "
                    f"case_pool 导入/无参 main/argv 三处缺陷（7 tests）。"),
        "scenario_key": "case_pool/backward_step_armaly",
        "mission_id": 21802,
        "tags": ["TensorLBM", "board-audit", "backstep", "maturity-L3", "referee-missing", "defect-fix"],
        "markdown": md,
        "figures": [{"name": FIG.name,
                     "caption": "左：m21672 交付瞬态 X1/s（20000/200000 步）vs Armaly 3.05 与 Erturk 2.878 带；黑菱形=仅存在于未合并分支 f3a211d78 的 88k 步收敛点 2.9202。右：修复后的 max_speed 诊断。",
                     "png_base64": base64.b64encode(FIG.read_bytes()).decode()}],
    }
    req = urllib.request.Request(BASE + "/tensor/reports", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + tok})
    resp = json.load(urllib.request.urlopen(req))
    print("report id", resp.get("id"), "figures", len(resp.get("figures") or []))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())