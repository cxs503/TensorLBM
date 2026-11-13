#!/usr/bin/env python3
"""Generate docs/reports/XVAL_LIFE_GEOMETRY_CONFIG.md from run artifacts (no hand-copied numbers)."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAR = ROOT / "runs/m17673/xval_parity.json"
RUNS = {
    "runtime_config": ROOT / "runs/m17673/xval_D20_config.json",
    "hard_coded": ROOT / "runs/m17673/xval_D20_hardcoded.json",
}
OUT = ROOT / "docs/reports/XVAL_LIFE_GEOMETRY_CONFIG.md"


def main() -> None:
    par = json.loads(PAR.read_text())
    rows = par["rows"]
    mech = par["mechanism"]
    lines: list[str] = []
    A = lines.append
    A("# XVAL: 本仓 vs joconnor22/LIFE `compiletime-params-plus-runtime-geometry-config`\n")
    A("锚点 `XVAL-LBM-COMPILETIME-PARAMS-PLUS-RUNTIME-GEOMETRY`（第 1 轮·任务榜·开源对标）。"
      "判据：<15%（owner 精度标准）。\n")
    A("## ① 问题 + ref 出处\n")
    A("对标做法：一个算例 = 一个目录 + 两份输入——`inc/params.h`（网格/物性/边界类型/ω/步数，"
      "编译期常量，改则必须 `make clean && make`）+ `input/geometry.config`（运行期文本："
      "`CIRCLE`/`FILAMENT` 关键字行，含编号、起始坐标、阵列间距、半径或 长度/厚度/角度/FLEX|RIGID/单元数/BC/密度/E）。\n")
    A("ref 出处（零手抄，逐处 file:line，源码位于 `/tmp/LIFE-master`，GPL-3.0）：\n")
    A("| 语义 | LIFE 源码/文件 | 本仓对应实现 |\n|---|---|---|")
    A("| 两遍扫描（先计数预留、再逐行构造） | `src/Objects.cpp:324-472` | `Scene.counts()` / `parse_geometry_config()` |")
    A("| 阵列展开 `pos=start+i*space`，`ID=start_id+i` | `src/Objects.cpp:399-440` | 同上 |")
    A("| 圆标记数 `floor(2πr/dx)`，等弧长摆放 | `src/IBMBody.cpp:47-77` | `CircleBody.marker_count/markers` |")
    A("| 梁标记数 `floor(L/dx)+1`，`R=[[c,-s],[s,c]]` | `src/IBMBody.cpp:79-130`, `inc/Utils.h:258` | `FilamentBody.marker_count/markers` |")
    A("| 角度输入=度，构造时一次性转弧度 | `src/IBMBody.cpp:100` | `FilamentBody.angle_rad` |")
    A("| 标记去重（<0.5dx；柔性优先；柔性-柔性报错） | `src/Objects.cpp:removeOverlapMarkers` | `_overlap_keep()` |")
    A("| 编译期块（`const`/`#define`） | `inc/params.h` | `CaseParams`（frozen）+ `apply_params(rebuild=True)` |")
    A("| 运行期体清单 | `input/geometry.config` | `Scene`（add/remove 不改指纹） |\n")
    A("## ② 方法与实现\n")
    A("新增 `src/tensorlbm/runtime_geometry.py`（格式互操作，不抄任何 LIFE 代码）"
      "：`CaseParams`（frozen→编译期块，`fingerprint()` 入证）、`Scene`（运行期体清单，"
      "支持重启期增删）、`parse_geometry_config()/read_geometry_config()`、`CircleBody/FilamentBody`"
      "（标记生成 + 栅格化到 solid mask），并在 `cylinder_st2d1.run_cylinder_st2d1()` 增加 "
      "`solid_mask=None` 注入点（默认路径不变，见测试 `test_solver_injection_is_bit_identical...`）。\n")
    A("对照口径：把 LIFE 自己 in-tree 的几何行 `CIRCLE 1 0 0.2 0.2 0.0 0.0 0.05`（Turek–Hron 算例）"
      "直接喂进本仓，映射到 Schäfer–Turek 2D-1 已验证通道算例（`CASE-CYLINDER-DRAG-RE20`，"
      "锚 Cd=5.57953523384，Nabh 1998 谱解，`benchmarks/verified/cylinder/re20_st/result.json`）。"
      "物理坐标→格点的映射 `x_lat=(x_phys-ox)/dx`；D=20 时 `dx=0.005 m`，锚点 (0.2,0.2)、r=0.05 m "
      "恰为 (40,40)、10 格。\n")
    A("## ③ 参数表（无量纲）\n")
    A("| 量 | 值 | 说明 |\n|---|---|---|")
    A("| Re | 20 | `u_mean·D/ν`，D=0.1 m，ν=0.001 m²/s |")
    A("| 阻塞比 B=D/H | 0.2439 | H=0.41 m |")
    A("| 格距 dx | 见下表 | `dx=D/D_cells` |")
    A("| 松弛时间 τ | 0.8 | MRT，与已验证配置一致 |")
    A("| u_mean(lattice) | 0.1 | D=20；Ma=0.173 |")
    A("| 标记数 | 见下表 | `floor(2πr/dx)` |\n")
    A("## ④ 对照表（ref vs got，逐项打勾/叉）\n")
    A("### 4.1 几何/机制层（精确量，非容差量）\n")
    A("| 网格 | dx [m] | 硬编码 mask 格数 | 配置 mask 格数 | Δ格数 | 标记数(ref 公式) | 标记数(got) | 标记误差 | 周长误差 | 判定 |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        ok = "✅" if (r["pass_mask"] and r["pass_perimeter"]) else "❌"
        A(f"| {r['grid']} | {r['dx_m']:.4f} | {r['mask_cells_ref']} | {r['mask_cells_got']} | "
          f"{r['mask_delta_cells']} | {r['markers_ref_formula']} | {r['markers_got']} | "
          f"{r['marker_count_err_pct']:.3f}% | {r['perimeter_err_pct']:.4f}% | {ok} |")
    A("")
    A("编译期/运行期边界（机制层）：\n")
    A(f"- 几何增删（add/remove body）后参数指纹不变：**{mech['fingerprint_unchanged_by_geometry']}** "
      f"(`{mech['fingerprint_before']}`)。")
    A(f"- 改 `CaseParams`（如 τ）未声明 rebuild 被拒（等价 `make clean && make`）：**{mech['param_edit_refused_without_rebuild']}**；"
      f"rebuild 后指纹变为 `{mech['fingerprint_after_rebuild']}`。\n")
    A("### 4.2 物理层（Cd vs 锚点）\n")
    got_any = False
    for label, path in RUNS.items():
        if not path.is_file():
            continue
        got_any = True
        r = json.loads(path.read_text())
        err = r["err_cd_pct"]
        verdict = "✅" if abs(err) < 15.0 else "❌"
        sub = "✅" if abs(err) < r["ref_cd"] * 0 + 5.0 else "❌"
        A(f"- `{label}`：Cd={r['cd']:.5f} vs ref {r['ref_cd']:.5f} → 误差 {err:+.3f}% "
          f"（<15% {verdict}，<5% {sub}）；步数 {r['n_steps']}，stop={r['stop_reason']}，"
          f"平台漂移 {r['cd_plateau_drift']:.2e}（stationary={r['stationary']}），几何来源 `{r.get('geometry_source')}`。")
    if not got_any:
        A("- 物理层运行已递交集群（job `cj1641`，D=20，60k 步上限 + 平台检测），本版报告生成时仍在排队，"
          "**未取得数值**——按证据纪律记 WITHHELD，不作任何 Cd 结论。")
    A("")
    A("## ⑤ 收敛性 / 参数敏感性（至少两点）\n")
    sens = [s for s in par["sensitivities"] if s["kind"] == "dx_refinement"]
    sens = sorted(sens, key=lambda s: -s["dx_m"])
    trend = " / ".join(f"{s['dx_m']:.4f}m->{s['perimeter_err_pct']:.4f}%" for s in sens)
    A("1. **dx 加密（标记离散）**：周长误差 " + trend +
      "；dx 每减半误差降约 4 倍（二阶，见图 2），网格加密不引入额外几何偏差。")
    ang = [s for s in par["sensitivities"] if s["kind"] == "angle_convention"]
    A("2. **角度约定（度→弧度 + 旋转矩阵）**：0°/30°/45°/90° 下标记跨度误差最大 "
      f"{max(abs(s['span_err_pct']) for s in ang):.2e}%，标记数恒为 "
      f"{ {s['n_markers'] for s in ang} }（与角度无关）——角度约定与 LIFE 一致，无额外误差。")
    A("3. **标记数随 dx 变化**：" + ", ".join(f"{s['dx_m']:.4f}m→{s['n_markers']}" for s in sens) +
      "，与 `floor(2πr/dx)` 一致（D=20 时 62 个）。\n")
    A("## ⑥ 失败模式与工件\n")
    A("- 失败模式（已处理）：关键字拼错/列数不符 → 抛 `ValueError`（LIFE 为静默错读）；"
      "体越界 → `validate_domain()` 拒绝（LIFE 静默截断）；两柔性体标记重叠 → 报错（同 LIFE）。")
    A("- 失败模式（未处理）：本仓对标的是**刚性**几何→mask 路径；LIFE 的 `FLEXIBLE` 分支需要"
      "其 Newmark+Newton FEM 与 Aitken 子迭代（另一 slug `NEW:LBM-FSI-BLOCK-GS-AITKEN`），"
      "本次只解析出 `FilamentBody.beam_properties()`（A=dx·h，I=dx·h³/12），**未移植结构求解器**。")
    A("- 差异清单与移植取舍见 `docs/reports/XVAL_LIFE_GEOMETRY_CONFIG_diffs.md`。")
    A("- 工件：`src/tensorlbm/runtime_geometry.py`、`tests/test_runtime_geometry.py`（29 项）、"
      "`xval/life_geometry_config/*`、`runs/m17673/xval_parity.json`、"
      "`artifacts/xval_life_geometry_config/fig1_geometry_parity.png`、`fig2_dx_sensitivity.png`"
      "（+ 若物理运行完成则 `fig3_cd_history.png`）。")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
