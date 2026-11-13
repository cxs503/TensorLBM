#!/usr/bin/env python3
"""Publish the m18098 audit report (numbers injected programmatically from the audit JSON)."""
from __future__ import annotations
import base64, json, os, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
res = json.load(open(HERE / "verify_m17955.json"))
perc = json.load(open(HERE / "verify_perc_oss.json"))["percolation"]
c1 = json.load(open(HERE / "verify_case1_oss.json"))
rep = json.load(open(HERE / "evidence_snapshot" / "results.json"))
lad = rep["case2_channel"]["ladder"]
seq = [r["H"] for r in lad]; deld = [abs(r["err_vs_analytic"]) for r in lad]
aud = [abs(r["err_int"]) for r in res["channel"]]
enc = json.load(open(HERE.parent / "audit_m18098" / "evidence_snapshot" / "SHA256SUMS.txt")) if False else None

rows = "\n".join(
    f"| {r['H']} | {row['k_intrinsic']:.5f} | {100*row['err_vs_analytic']:+.3f} % | **{r['k_intrinsic']:.5f}** | "
    f"**{100*r['err_int']:+.4f} %** | {100*r['err_dense_int']:+.4f} % | {r['sparse_vs_dense_u_rel']:.1e} |"
    for r, row in zip(res["channel"], lad))

md = f"""# 交差审计 m17955 · XVAL-LBM-LIST-BASED-SPARSE-LBM-PLUS-PERMEABILITY（TensorLBM）

## ① 案例介绍

被审单 m17955（status=succeeded）把成熟开源实现 `lssfau/walberla` 的 **list-based 稀疏 LBM**
（lbmpy `SparseLbMapper.create_index_array` + `stream_pull_collide`，waLBerla 稀疏代码路径的上游本体）
移植进 TensorLBM，并以 Darcy 反算渗透率作判据。审计单 m18098 是交差核验：三件套真伪 + 基准裁判 + 判级。

物理算例：①24×32 封闭箱（四壁 + 两块障碍，100 步，g=1e-4）三臂核等价；②周期平面槽道
（流体行 H∈{{8,16,32}}，一行 wall → 半程壁面在 y=0.5，x 周期，ν=0.1，τ=0.8），解析 `k=H²/12`；
③随机逾渗基体 24²/32²（φ≈0.592，6000 步，g=2e-5）；④τ∈{{0.6,0.8,1.0,1.5}} 敏感性。
精度判据：owner 标准 <15 %（对标误差 >15 % 不可接受），另设二级判据（地址表等价、核一致、表示中立性）。

## ② 怎么做的

1. **取证**：`tensor_benchmark_cases` 按锚点查询 → **0 行**（全表 84 行 / 71 锚），机器裁判缺席；
   `tensor_reports` 里该 mission **无报告行**；交付文件在被审工作树内 **未提交**（`git status` 四个 `??`），
   审计工作树 master `fc86f508` 内不存在该模块。
2. **独立复算**：不读交付 json 的键，直接 import 交付的 `build_index_table / sparse_step /
   macroscopic_sparse`，用**单一显式约定** `k_int = ν⟨u⟩_fluid/g`、`k_super = φ·k_int` 重算槽道与逾渗，
   并与同几何稠密控制臂对拍（`verify_m17955.py`, `verify_perc_oss.py`）。
3. **三臂核等价独立复跑**：自建 `oss_ref.OssSparseStepper`（lbmpy 2.0 + pystencils 编译 C）跑 case-1，
   枚举状态约定配对（`verify_case1_oss.py`）。
4. **证据冻结**：交付 9 文件逐字节 `evidence_snapshot/` + `SHA256SUMS.txt`（被审工作树会被回收）。
5. **判级 + 落表**：`tensor_maturity_verdicts` id=686，level=2，verdict_by=board-audit，
   `maturity_gate.judge()` 机器复核 → `gate_level=2`，L3 报 `integration_callers(grep 无引用)`。

## ③ 计算结果（数字由审计 JSON 程序化注入）

槽道渗透率（裁判 = 解析解 `k=H²/12`）：

| H | 交付自报 k_int | 交付自报误差 | **审计复算 k_int** | **审计复算误差** | 稠密控制臂误差 | 稀疏 vs 稠密 ⟨u⟩ |
|---|---|---|---|---|---|---|
{rows}

误差比 {aud[0]/aud[1]:.2f} / {aud[1]/aud[2]:.2f} → **二阶收敛 O(H⁻²)**；交付 docs ⑤.1 的 "O(1/H) 半程壁面离散"
解释被证伪（那序列正是 `1−φ = 1/(H+1)` 叠加真实 −0.44 %）。

逾渗 24²（seed 11，φ={perc['phi']:.6f}，6000 步）：稀疏 k_int={perc['k_int_sparse']:.6e} vs 稠密
{perc['k_int_dense']:.6e}，相对差 **{perc['rel_diff_intrinsic']:.2e}**（交付自报 68.915 % = `1−φ²` 假数）；
k_super 双臂 {perc['k_super_sparse']:.6e}，max|Δf|={perc['max_abs_pdf_diff']:.3e}。

case-1 核等价（独立复跑）：稀疏 vs **OSS 生成核** max|Δf| = **{c1['max_sparse_vs_oss_postcollision']:.6e}**
（pdf 量级 {c1['scale_max_pdf']:.4f}，相对 {c1['max_sparse_vs_oss_postcollision']/c1['scale_max_pdf']:.1e}）
→ 交付自设判据 1e-4 **实际 PASS**，而交付 `results.json` 报 5.8957e-3 FAIL；
稀疏 vs 稠密 {c1['max_sparse_vs_dense']:.4e}（状态约定差，非物理）；
地址表 5152 条目 100 % 坐标级一致（对 lbmpy `SparseLbMapper`）。

## ④ 结果图

左：槽道误差 vs H（交付自报 vs 审计复算 vs 斜率 −1/−2 参考线，3 % 闸门）；右：三项差异量的
交付自报 vs 审计复算（对数轴，含交付自设 1e-4 闸门）。图见 `fig_audit_m17955.png`。

## ⑤ 注意点 / 局限

* **交付台账是坏的，功能是好的**：`make_report.py:111-130 fix_darcy()` 二次施加 φ（`:116`），
  H=8 报 4.72 = φ·5.310 → −11.50 %；逾渗把 φ²k_int 与 φk_int 相比 → 68.9 %；case-1 的 c1/c3 与实跑结果正好互换。
  三者方向都是"把好结果报成坏结果"，不是造假，但**报告六段里的 ④⑤ 结论按错数字写成**。
* `profile_max_rel_err` ≈ {100*lad[0]['profile_max_rel_err_vs_parabola']:.1f}–{100*lad[-1]['profile_max_rel_err_vs_parabola']:.1f} %
  以 ref=0 记 PASS：交付的解析抛物线峰值只有真峰值 2/3，~50 % 是归一化假象；同归一后剖面误差
  0.72 % / 0.19 % / 0.049 %。
* H=32 臂在 `max_steps=9000` 截断（自设 tol 1e-10 未达）：交付 85.2957 vs 复算 16662 步 85.3100（差 0.017 %）。
* 新模块**零 pytest**、**零集成调用者**、交付**未入库** → L3 不达。
* OSS 臂周期域不可用（lbmpy `mapping.py:131` uint32 下溢，`IndexError: 4294967295`）→ 周期域只有 S+D 两臂；
  waLBerla C++ 侧（`Permeability.h`/`Percolation.cpp`）本机无 C++ 树 → **WITHHELD**，不作断言。
* 3D（D3Q19）稀疏收益：交付未测，本审计亦未测 → 未知。

## ⑥ 误差判定（逐项）

| 关键量 | ref | got（审计复算） | 误差 | <3 % ? |
|---|---|---|---|---|
| 槽道 k_int H=8 | 5.33333 | {res['channel'][0]['k_intrinsic']:.5f} | {100*res['channel'][0]['err_int']:+.4f} % | **是** |
| 槽道 k_int H=16 | 21.33333 | {res['channel'][1]['k_intrinsic']:.5f} | {100*res['channel'][1]['err_int']:+.4f} % | **是** |
| 槽道 k_int H=32 | 85.33333 | {res['channel'][2]['k_intrinsic']:.5f} | {100*res['channel'][2]['err_int']:+.4f} % | **是** |
| 逾渗 k_int 稀疏 vs 稠密 | 同约定 | {perc['k_int_sparse']:.6e} vs {perc['k_int_dense']:.6e} | {perc['rel_diff_intrinsic']:.2e} | **是** |
| case-1 稀疏 vs OSS 核 | 0（同态逐点） | {c1['max_sparse_vs_oss_postcollision']:.6e} | 相对 {c1['max_sparse_vs_oss_postcollision']/c1['scale_max_pdf']:.1e} | **是** |
| 地址表等价 | 100 % | 100 %（5152 条目） | 0 | **是** |

结论：**功能实质达标（L2）但台账不达标**——机器裁判缺席（无 `tensor_benchmark_cases` 行）+ 交付证据链不可复现
（c1/c3 反号、c4/c5 为 φ 假数）+ 零集成/零测试/未入库 → 判 **L2**，L3/L4 不达；
返修路径见 `docs/audit_m18098/AUDIT_M17955.md` §6。
"""

fig = (HERE / "fig_audit_m17955.png").read_bytes()
payload = {
    "title": "交差审计 m17955：list-based 稀疏 LBM（TensorLBM）功能成立但台账/判据失真的 φ 记账缺陷（L2）",
    "summary": "独立复算槽道 k_int 误差 0.4375/0.1094/0.0274 %（H=8/16/32，解析 H²/12）、"
               "逾渗稀疏-稠密 8.1e-11、lbmpy 生成核 3.35e-6、地址表 100 %：功能真实；"
               "交付自报 -11.5/-5.985/-3.073 %、68.9 % 与 all_pass=false 均为 make_report.py 二次 φ 与状态约定错配造成的假数。判 L2。",
    "scenario_key": "maturity-audit",
    "mission_id": 18098,
    "tags": ["board-audit", "TensorLBM", "XVAL", "sparse-LBM", "permeability", "maturity-L2"],
    "markdown": md,
    "figures": [{"name": "fig_audit_m17955.png",
                 "caption": "左：槽道渗透率误差 vs H（交付自报 / 审计复算 / 斜率 −1、−2 参考线 / 3 % 闸门）；"
                            "右：case-1 OSS 核差、稀疏-稠密差、逾渗相对差的自报 vs 复算（对数轴，含交付 1e-4 闸门）",
                 "png_base64": base64.b64encode(fig).decode()}],
}
api = os.environ["TENSOR_API_BASE"]
tok = json.load(urllib.request.urlopen(urllib.request.Request(
    api + "/auth/login", data=json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                                          "password": os.environ["TENSOR_AGENT_PASS"]}).encode(),
    headers={"Content-Type": "application/json"})))["access_token"]
r = urllib.request.Request(api + "/tensor/reports", data=json.dumps(payload).encode(),
                           headers={"Content-Type": "application/json", "Authorization": "Bearer " + tok})
resp = json.load(urllib.request.urlopen(r))
print("report id:", resp.get("id"), "len(markdown):", len(md))
