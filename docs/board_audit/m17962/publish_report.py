#!/usr/bin/env python3
"""Publish the m17962 board-audit report to the platform (six-element paper-grade)."""
import base64, json, os, pathlib, urllib.request, urllib.error

ENV = dict(l.strip().split("=", 1) for l in open("/root/tp-agent/agent.env")
           if l.strip() and not l.startswith("#") and "=" in l)
BASE = ENV["TENSOR_API_BASE"].rstrip("/")


def api(method, path, payload=None, token=None):
    req = urllib.request.Request(BASE + path, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    data = json.dumps(payload).encode() if payload is not None else None
    try:
        with urllib.request.urlopen(req, data=data, timeout=120) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


st, d = api("POST", "/auth/login", {"username": ENV["TENSOR_AGENT_USER"],
                                    "password": ENV["TENSOR_AGENT_PASS"]})
tok = d.get("access_token")
print("login", st, bool(tok))

WT = pathlib.Path("/root/tp-agent/work/m17962")
AUD = WT / "docs/board_audit/m17962"
P = AUD / "preserved_m17894/runs_xval_edt_two_segment"


def fig(path, name, caption):
    return {"name": name, "caption": caption,
            "png_base64": base64.b64encode(pathlib.Path(path).read_bytes()).decode()}


figures = [
    fig(AUD / "fig_audit_m17894_overview.png", "fig1_audit_overview",
        "左：m17894 交付的 24 条相对判据误差（对数轴，3%/15% 参考线；22 条 ≤3%，2 条 3-15%，0 条 >15%）"
        "＋ 3 条绝对偏差注记（1.123/1.262/0.434 cells vs 1.5 cells 门槛）。右：L1-L5 判据状态阶梯，裁定 L2。"),
    fig(P / "figures/fig2_distance_field.png", "fig2_distance_field",
        "被审交付图复核：中平面深度场——解析胶囊距离 vs 点阵 EDT vs 差场（差 ≤1.26 cells，源 runs/xval_edt_two_segment/figures/fig2_distance_field.png）。"),
    fig(P / "figures/fig3_convergence.png", "fig3_convergence",
        "被审交付图复核：体素胶囊排出体积误差 vs 1/h（log-log），斜率 1.81（≈2 阶，本单复现同值）。"),
]

markdown = """# 交差审计 m17894：TensorLBM EDT 两段几何标签压缩（XVAL vs MPLBM-UT）→ 裁定 **L2**

## ① 案例介绍
被审对象 m17894（succeeded，平台报告 16822，commit `dd38f652`，分支 `agent/m17894`）。锚点
`XVAL-LBM-EDT-TWO-SEGMENT-GEOMETRY-TAG-COMPRESSION`：复刻 je-santos/MPLBM-UT @`a9fc4966` 的岩心几何预处理链
（`rock/3` → `erase_regions` → scipy 精确 EDT → `0<d<2` boundary / `d>=2` interior 二段分割 → Palabos int16 码
2608/2609/2610/2611），并提出扩展：独立 k-d tree EDT、2 bit/体素无损标签打包、per-grain id 侧信道、
JSON sidecar、逐颗粒 `G_wall` 壁面力（混合润湿）。交付：`src/tensorlbm/edt_geometry_tags.py`(711 行)、
`preprocess_geo.py:908 palabos_geometry_tags`、`multiphase3d.py:1324 wall_adhesion_force_3d`、
`tools/xval_edt_two_segment.py`(888 行)、35 用例、25 个运行工件（1.8 MB）、219 行六段报告。

## ② 怎么做的（审计流程，判据先于动手）
1) 三件套取证：工件实存+大小+入 commit；`table.csv` 27 行 ref/got/误差/门槛/通过位逐行核。
2) 复现：`PYTHONPATH=src python tools/xval_edt_two_segment.py --out /tmp/m17962_repro --no-figures`
   → 6/6 算例逐字段与交付 `summary.json` 相等（零 DIFF）。
3) 测试复跑：`pytest tests/test_edt_geometry_tags.py tests/test_xval_edt_two_segment_runner.py -q` → **35 passed / 5.69 s**。
4) 裁判检查：`tensor_benchmark_cases` 该 anchor **0 行**；`tensor_case_pool` linked_anchor **0 行**。
5) 独立性对抗：追问 agreement=1.0 的 ref 是否可执行上游代码；追问 32× 是否落盘。
6) 耐久性：`git status` 干净、`git log --all`、生态克隆 main HEAD `be78a87b` 是否含模块。
7) gate 干跑：调 `maturity_gate.judge(2, evidence, 'TensorLBM', 17962)` → `gate_level=2, reasons=[]`（L2 指针全过）；
   若有人claim L3/L4，机器把级别压回 2（`integration_callers(grep 无引用)`），与本审计同判。

## ③ 计算结果（程序化注入，零手抄）
| 项 | 结果 |
| --- | --- |
| 三件套 | 工件 25 文件实存入 commit；`table.csv` 27 数据行；`repro_cmd` 实跑成功 |
| 复现一致性 | 6/6 算例 `tag_agreement`、不符体素 0、EDT 互检 0.0、`packed_bytes` 逐字段相等；`all_checks_pass=True`；conv slope 1.81 |
| 测试 | 35 passed / 5.69 s（单跑 test_edt_geometry_tags.py 32 passed / 7.43 s）|
| 27 判据误差分布 | 相对 24 行：≤3% **22**，3-15% **2**（V 形近界面 9.522%、胶囊深部 3.597%），>15% **0**；绝对偏差 3 行 1.123/1.262/0.434 cells（门槛 1.5）全过 |
| 裁判 | `tensor_benchmark_cases` 0 行、`tensor_case_pool` 0 行 ⇒ 无登记容差，15% 系 runner 自设+平台底线 |
| 交付状态 | worktree 干净、`dd38f652` 在 `agent/m17894`；**main `be78a87b` 无该模块** |
| 集成调用方 | grep 仅命中 tests + docstring，无求解器/算例消费者 |

## ④ 结果图
图1 审计总览（误差分布 + 级别阶梯，本单生成，数据源 `table.csv`）；图2/图3 为被审交付场图与收敛图的复核内嵌。

## ⑤ 注意点（发现，按严重度）
- **F1 阻塞(耐久)**：交付未合入 main，只在分支 `agent/m17894`；worktree 回收前须推送/合并，否则生态内零可达。
- **F2 阻塞(L4)**：唯一裁判缺位（两表 0 行），故即使误差表干净也不予 L4。
- **F3 中**：`tag agreement=1.0` 的 ref 是 runner 内**自撰转写**（非执行上游代码），度量的是转写保真度；
  真独立证据是 k-d tree vs scipy（0.0 cells）、解析几何分类一致率 0.9953/0.9929/0.9724、解析体积 2.60%/1.94%、收敛率 1.81。
- **F4 低(措辞)**：32× 是编解码器级属性；落盘 `.dat` 仍 int16（4×），2-bit 类图以 hex 存 JSON sidecar；报告 4b 两列并陈（诚实），headline 易误读。
- **F5 低(流程)**：同锚点重复派单 m17728→m17829（L2 但 gate reopened，因 evidence 指针占位/缺失）→m17894；应走 reopen 返修而非新单重做。
- **F6 低(记录)**：skimage 与 numpy 2.4 ABI 不兼容 → `erase_regions` 用 scipy 等价实现替代（差异已登记、测试固化）；无 vendor 快照。

## ⑥ 误差判定（逐项回答是否 <3%）
| 关键量 | 误差 | <3%？ |
| --- | --- | --- |
| Palabos 标签 vs OSS 转写（6 算例） | 0 体素不符 | 是（但为同规则转写保真度，F3） |
| k-d tree EDT vs scipy EDT | max diff 0.0 cells | 是 |
| 二段分类 vs 解析几何 | 0.474% / 0.710% / 2.756% | 是 |
| 深部（d>4）相对深度误差 | 0.367% / 2.640% | 是 |
| 胶囊 / 三球堆体积 vs 解析 | 2.597% / 1.943% | 是 |
| 平板固相分数 | 0.000% | 是 |
| V 形近界面（d≈2）相对深度 | 9.522% | **否**（已显式登记，绝对偏差 1.262 cells，15% 判据内 PASS，非实现缺陷） |
| 物理层 Cd/St | 未测 | — 本锚点为几何预处理，无流动解，不得引作水动力精度证据 |

## 裁定
**L2**（impl_files/entry/run_artifacts/tests 全实、复现逐字段相等）；L3 不予（无集成调用方、main 无模块）、
L4 不予（裁判 0 行）、L5 不予（外部锚为自撰转写、无 vendor 快照）。Disposition：approve with notes。
落表 `tensor_maturity_verdicts` id=670（repo=TensorLBM, mission_id=17962, level=2, verdict_by=board-audit, gate_state=pending）。
被审交付已按字节保全：`docs/board_audit/m17962/preserved_m17894/`（MANIFEST.sha256 32/32 OK）。
"""

payload = {
    "title": "交差审计 m17894：TensorLBM EDT 两段几何标签压缩（vs MPLBM-UT）→ L2（三件套实、复现逐字段相等；L3/L4 驳回）",
    "summary": ("board audit of m17894: 27 判据/6 算例复跑逐字段相等（35 tests passed），工件与 commit 实存；"
                "tensor_benchmark_cases 与 tensor_case_pool 对该 anchor 均 0 行 ⇒ L4 不予；适配层无任何非测试调用方且 "
                "main be78a87b 无模块 ⇒ L3 不予；裁定 L2，verdict id=670。"),
    "scenario_key": "xval_edt_two_segment_geometry_tag_compression_audit_m17894",
    "mission_id": 17962,
    "tags": ["board-audit", "TensorLBM", "XVAL", "EDT", "geometry-tags", "MPLBM-UT", "maturity-L2", "m17894"],
    "markdown": markdown,
    "figures": figures,
}
st, d = api("POST", "/tensor/reports", payload, tok)
print("report post", st, json.dumps(d)[:300])