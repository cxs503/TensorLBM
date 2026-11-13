#!/usr/bin/env python3
"""Publish the m17880 board-audit report (six sections, figures embedded, numbers from artifacts)."""
import base64
import json
import pathlib
import subprocess
import urllib.request

HERE = pathlib.Path("/root/tp-agent/work/m17880/docs/board_audit/m17880")
DATA = json.loads((HERE / "artifacts/ref_vs_got.json").read_text())
REPRO = json.loads((HERE / "artifacts/repro_compare.json").read_text())
LAND = json.loads((HERE / "artifacts/landing_checks.json").read_text())
rows = DATA["dispersion_rows"]
s = DATA["dispersion_summary"]


def pct(x):
    return f"{100*x:+.2f}%"


table = "\n".join(
    f"| {r['m']} | {r['k']:.4f} | {r['ref_analytic_omega']:.4e} | {r['got_oss_omega']:.4e} | "
    f"{r['got_tl_lee_lin_omega']:.4e} | {pct(r['err_tl_vs_analytic_rel'])} | "
    f"{pct(r['err_oss_vs_analytic_rel'])} | {r['abs_dev_tl_vs_analytic']:.2e} | "
    f"{'yes' if r['pass_15pct_tl_vs_analytic'] else '**NO**'} |" for r in rows)

md = f"""# 交差审计 m17880 — m17762 `XVAL-LBM-FREE-ENERGY-CAHN-HILLIARD-CORE`（TensorLBM）

**裁定 L3**｜审计席 board-audit @ m17880｜verdict #{663}｜gate 机器预审 = confirm L3（judge 返回 (3, [])）

## ① 案例介绍

自由能（Cahn–Hilliard）双分布 LBM：`f` 走 NS、`g` 走序参量 ψ，迁移率 `M=Γ(τ_g−1/2)`。被审单（m17762，deepseek-v4-flash，succeeded）声称 TensorLBM 历史 `swift` 序参量平衡态带一个与 Γ 无关的虚假扩散 `D=c_s²(τ_g−1/2)`（因 `Σ_i w_i c_ia c_ib ψ = c_s²ψδ_ab`），故它解的不是 CH 方程；把参考实现（ψ 由静止分量承载 + `λ_i=(3/2)(3|c|²−3)`）移植为 `geq_convention="lee_lin"` 后应无伪项。

- 外部锚：`ebinan92/Fingering_dynamics@150a5739`（`lattice_boltzmann/fingering_periodic.py` sha256 `d3789463…`，本审计独立复核 HEAD 与 sha256 双向一致，OSS 臂经 `importlib` **真跑**上游算子）
- 几何：z 不变 `(1,64,64)` 全周期，动量冻结 u=0（只跑序参量块）
- 参数：σ=0.05、ξ=1.0、A=−a=0.075、κ=0.0375（k_c=√(A/κ)=1.4142）、Γ=1.0（另扫 0.5/2.0）、τ_g=0.788675、ε=0.01（另扫 0.005/0.02）
- 模：m=3/5/8/11/14 → k=0.2945…1.3744
- 判据：owner 15%；**审计前该 anchor 在 tensor_benchmark_cases 为 0 行**（本审计补登 2 行：等价性 passed / 解析色散 failed）

## ② 怎么做的

被审交付：`src/tensorlbm/multiphase3d.py:775 free_energy_step_3d(..., geq_convention=)`（+41/−2，默认 swift 不变）＋`scripts/xval_m17762_free_energy_ch_core.py`（A 算子/B 参数反演/C 色散/D 力形式差/E swift 衰减扫/F 作图）＋6 测试＋`artifacts/xval_m17762/`。

审计动作（全部程序化、零手抄，`build_evidence.py`）：重跑 repro_cmd；`git cat-file -e HEAD:<path>` 落地探针；交付 JSON 逐行重算裁判表；重跑单测与既有回归；`py_compile` 渲染脚本。

**复现性**：JSON payload 逐键相同，3 张 PNG sha256 **逐位相同**（`a494ff2d/ae733aee/9a2c9396`）→ `repro_cmd` 真可执行且确定性。
**落地**：4/4 交付件 `git cat-file -e HEAD` = ABSENT，`HEAD:src/tensorlbm/multiphase3d.py` 中 `geq_convention` 计数 **0**，`git status` 全 `??/M`。

## ③ 计算结果

| m | k | ref 解析 ω | got OSS ω | got TL(lee_lin) ω | TL vs 解析 | OSS vs 解析 | 绝对偏差 | <15% |
|---|---|---|---|---|---|---|---|---|
{table}

移植等价性 TL vs OSS 逐模：0.0522%/0.0007%/0.0009%/0.0083%/0.0062%（全部 ≪15%）。
其他：σ 反演相对误差 1.4e-16；μ 算子 TL-vs-OSS 1.9e-13；M_eff vs 预测 1.94%/2.00%/2.12%；剖面法 σ −10.40%（两臂同值）；swift 伪扩散衰减 vs 模型 0.51–9.76%；算子阶 1.985–1.997。

## ④ 结果图

左：解析 CH 色散曲线 + TL/OSS 实测点 + swift 臂模型（负，全 k 衰减）+ k_c；右：相对误差与 ±15% 带，m=11（23.69%）、m=14（301.75%）越界。图由 `make_figures.py` 只读 `ref_vs_got.json` 生成。

## ⑤ 注意点

- **F1 落地缺陷（阻断）**：整单 0 个文件入库，能力只活在将被回收的工作树；本审计已把字节+patch 快照进 `docs/board_audit/m17880/refs/`（sha256 对齐），凭 `git apply` 即可补录。
- **F2 裁判缺失（阻断）**：审计前 benchmark 表该 anchor 0 行、无预声明容差 → 15% 只能事后套 owner 通标；已补登 2 行。
- **F3 报告缺陷（阻断）**：代码/测试 docstring 引用的 `docs/XVAL_M17762_...md` 不存在，渲染脚本 `py_compile` 失败（30 处 `%%`，首错 line 48），`tensor_reports` 该单 0 行。
- **F4 断言挑模**：测试只断言 m=6/N=48（7.26%）这一个最有利模，自报的 23.69%/301.75% 从未进断言。
- **F5 读数**：m=14 的 k/k_c=0.972，解析 ω 近零（2.27e-3），相对误差病态，应看绝对偏差 6.84e-3；**m=11 才是干净的健康模失败点**。
- 范围：仅线性单模 z 不变；3D 非线性 fingering 分支未验，不能外推为"CH 求解器已通"。

## ⑥ 误差判定（显式回答是否 <3%）

- **<3% 成立**：σ 参数反演（1.4e-16）、μ 算子（1.9e-13）、M_eff（1.94%/2.00%/2.12%）、m=5 色散（2.00%）。
- **不成立**：色散对解析解 —— m=3 4.56%、m=8 7.26%、m=11 **23.69%**、m=14 **301.75%**（5 模中 3 模 >3%、2 模 >15%）；swift 伪扩散模型最差 9.76%；剖面法 σ −10.40%。
- 移植等价性 0.052% **成立**（but code-to-code，非物理真值）。

**判级 L3**（L2/L3 机器可核；L4 拒绝：裁判 5 模中 2 模越 15%、无预声明、断言挑模、0 落地；L5 拒绝：外部锚为同级实现，无实验/文献真值）。honest-report：被审单"复现解析 CH 生长率"的表述被其自身工件表否定（仅 k≤0.785 成立）。
"""

figs = []
for name, cap in [("fig_audit_dispersion.png", "审计重算：解析 CH 色散 vs OSS/TL 实测；右图为相对误差与 15% 判据带"),
                  ("fig1_ch_dispersion.png", "被审单自带图（与审计重跑逐位一致）：swift 约定全 k 衰减、lee_lin 与 OSS/解析重合")]:
    p = (HERE / "artifacts" / name) if name.startswith("fig_audit") else (HERE / "refs/artifacts" / name)
    figs.append({"name": name, "caption": cap,
                 "png_base64": base64.b64encode(p.read_bytes()).decode()})

report = {
    "title": "交差审计 m17880：m17762 XVAL-LBM-FREE-ENERGY-CAHN-HILLIARD-CORE 判级 L3（移植等价 0.052% 成立；解析色散 5 模 2 模越 15%；整单 0 落地）",
    "summary": "board-audit 复核 m17762：真代码真可复现（PNG 逐位相同），移植等价性 TL-vs-OSS ≤0.052% 成立；但对解析 CH 色散 5 模中 m=11 23.69%、m=14 301.75% 超 15%，测试只断言最有利单模；benchmark 表原 0 行（已补登 2 行），交付件 0 个入库，doc 缺失且渲染脚本语法错 → L3，L4/L5 拒绝。",
    "scenario_key": "XVAL-LBM-FREE-ENERGY-CAHN-HILLIARD-CORE",
    "mission_id": 17880,
    "tags": ["board-audit", "TensorLBM", "XVAL", "free-energy", "Cahn-Hilliard", "maturity-L3", "m17762", "honest-FAIL"],
    "markdown": md,
    "figures": figs,
}
pathlib.Path("/tmp/report.json").write_text(json.dumps(report, ensure_ascii=False))

API = "http://127.0.0.1:8010/api/v1"
env = dict(l.strip().split("=", 1) for l in open("/root/tp-agent/agent.env") if "=" in l and not l.startswith("#"))
req = urllib.request.Request(API + "/auth/login", method="POST",
                            data=json.dumps({"username": env["TENSOR_AGENT_USER"],
                                             "password": env["TENSOR_AGENT_PASS"]}).encode(),
                            headers={"Content-Type": "application/json"})
tok = json.loads(urllib.request.urlopen(req).read())["access_token"]
req = urllib.request.Request(API + "/tensor/reports", method="POST",
                             data=pathlib.Path("/tmp/report.json").read_bytes(),
                             headers={"Content-Type": "application/json", "Authorization": "Bearer " + tok})
try:
    r = json.loads(urllib.request.urlopen(req).read())
    print("published:", json.dumps(r)[:400])
except Exception as e:  # pragma: no cover
    print("publish failed:", e, getattr(e, "read", lambda: b"")()[:500])