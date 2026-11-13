"""Publish the round-2 XVAL report to the platform (numbers injected from results.json)."""

import json, base64, io, urllib.request, os
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)
r = json.loads(Path('results.json').read_text())
H = r['arms']

from PIL import Image
im = Image.open('fig_xval_dem.png').convert('RGB')
w = 900
im = im.resize((w, int(im.height * w / im.width)))
buf = io.BytesIO()
im.convert('P', palette=Image.ADAPTIVE, colors=64).save(buf, 'PNG', optimize=True)
png = buf.getvalue()
print('png bytes', len(png))


def pct(x):
    return f"{x*100:+.2f}%"


L = []
L.append("## ① 问题设定 + 基准出处\n")
L.append("OSS 台账 `ls -1 /root/oss-cache` = 3 仓（`Maarten-vd-Sande_lbm@573a01c`, `pylbm@f9f5d16`, `hschwane/GraSPH@ae8c6ff`），"
         "`grep -ril -E \"discrete element|spring.?dashpot|granular|contact model|restitution\"` **无输出** ⇒ 无 DEM 孪生，"
         "第 1 轮找孪生的死法作废。改锚：闭式解（Cundall&Strack 1979 / Di Renzo&Di Maio 2004）+ 文献曲线"
         "（Schiller&Naumann 1935 / Clift-Grace-Weber 1978 / ten Cate 2002 的无界极限）。判据预声明 `predeclared.json`："
         "<15%（owner），接触臂更严 2%。\n")
L.append("## ② 方法与实现\n")
L.append("驱动 `xval/XVAL-LBM-NO-DEM-REPO-IN-OSS-INVENTORY/run_xval.py`，全程调用本仓真实模块 "
         "`tensorlbm.dem.DEMSolver/LinearSpringDashpot/MomentumExchange`；参考式在驱动内独立写出（不从被测代码导入）。"
         "A/B 接触、C Stokes 自明（预先声明 H3，不计物理验收）、D Schiller-Naumann vs Clift-Gauvin、"
         "E 流体侧 LBM Cd（不设门）、F 动量交换账本。关键控制：`clip=False` 对照模型只去掉 `max(0, kδ−ηv_n)` 限幅。\n")
L.append("## ③ 参数表\n")
L.append("| 量 | 值 |\n|---|---|\n| k / η / ζ / ω_n | 1e5 N/m / 0.2479 N·s/m / 0.2 / 1.6142e5 rad/s |\n"
         "| Δt/t_c | 1/40, 1/80, 1/160 |\n| 碰撞 v0 / R | 0.05 m/s / 1e-3 m |\n"
         "| ρp/ρf/μ/g | 917 / 1000 kg/m³, 1e-3 Pa·s, 9.81 m/s² |\n| D | 1e-4, 2e-4, 5e-4 m |\n"
         "| Δt/τ_p | 1/25, 1/100 |\n| Re_measured | 0.0444, 0.3378, 4.0598 |\n")
L.append("## ④ 对照表（ref vs got）\n")
L.append("**接触臂**（解析 e=%.6f, t_c=%.5e s）\n\n| t_c/Δt | e_sim | e 误差 | t_c 误差 | δmax 误差 | 15pct | 2pct |\n"
         "|---|---|---|---|---|---|---|\n" % (H['contact'][0]['e_analytic'], H['contact'][0]['t_c_analytic_s']))
for x in H['contact']:
    L.append(f"| {x['steps_per_contact']} | {x['e_sim']:.6f} | {pct(x['e_rel_err'])} | {pct(x['t_c_rel_err'])} | "
             f"{pct(x['delta_max_rel_err'])} | {'✓' if abs(x['e_rel_err']) < 0.15 else '✗'} | ✗ |\n")
L.append("\n**控制臂（去掉无张力限幅）**\n\n| t_c/Δt | e_sim | e 误差 |\n|---|---|---|\n")
for x in H['contact_no_clip_control']:
    L.append(f"| {x['steps_per_contact']} | {x['e_sim']:.6f} | {pct(x['e_rel_err'])} |\n")
L.append("\n**沈降臂（DEMSolver 真实积分，尾窗漂移 ≤1.2e-7）**\n\n| D [m] | 臂 | u_sim | 参考 | 相对误差 |\n|---|---|---|---|---|\n")
for key, lab in (('stokes_settling', 'C Stokes'), ('schiller_naumann_settling', 'D SN vs Clift-Gauvin')):
    for x in H[key]:
        if x['dt_div'] != 25:
            continue
        L.append(f"| {x['diameter_m']:.0e} | {lab} | {x['u_sim_m_s']:.6e} | "
                 f"{x['u_reference_m_s']:.6e} | {x['rel_err_vs_reference']:+.2e} |\n")
L.append(f"\n**F 动量账本**：relative_residual = {H['ledger']['relative_residual']}"
         f"（n_events={H['ledger']['n_events']}）✓\n")
L.append("\n**E 流体侧 LBM 体检（不设门）**\n\n| Re | Cd(LBM) | Cd(Clift-Gauvin) | 误差 |\n|---|---|---|---|\n")
for x in H['lbm_leg']['rows']:
    if x['variant'] != 'raw_run_lambda':
        continue
    L.append(f"| {x['Re']:.2f} | {x['cd_lbm']:.3f} | {x['cd_clift_gauvin']:.3f} | {pct(x['rel_err'])} |\n")
L.append("\n## ⑤ 收敛性与敏感性\n")
L.append("Δt 细化 ×4：t_c 误差 −2.50→−1.25→−0.63%（一阶），δmax +0.68→+0.32→+0.15%，e 10.00→9.25→8.90%（指向非零平台）。"
         "e 的极限外推：无限幅 0.5280（+0.26% vs 解析）、带限幅 0.5716（+8.5%）⇒ 偏置是**模型性**（max(0,·) 限幅）而非离散性；"
         "160 步接触里 20 步触发限幅。沈降两档 Δt/τ_p 终速 8 位有效数字不变（稳态为不动点）。\n")
L.append("## ⑥ 误差判定（逐项 vs 基准，<3% 显式回答）\n")
L.append("* 接触 t_c：−0.63% → **<3% ✓**；峰值重叠：+0.15% → **<3% ✓**；\n"
         "* 恢复系数：+8.90%（限幅臂）/ +0.43%（无限幅控制臂）→ 限幅臂 **不 <3%**（也不 <2%），但 <15% 达标；\n"
         "* 沈降 vs Clift-Gauvin：≤5.1e-6 相对 → **<3% ✓**（Stokes 臂同式自明，不计物理验收）；\n"
         "* 动量账本：残差 0.0 → **<3% ✓**；\n"
         "* 流体侧 LBM Cd：4/6 在 15pct 内，最差 **−28.5%（Re=50）**、Re=1/2 为 +16.5%/+20.3% → **不达标**，"
         "根因单域尺寸无 λ-外推。\n")
L.append("\n**诚实不达标 3 条**：① 恢复系数 +8.90%（无张力限幅偏置，修法已明，本轮只验证不改本构）；"
         "② 流体侧 LBM 在 Re≤2 与 Re=50 超 15%；③ 未做 ten Cate 原始实验表对标/体标定/Hertz–Mindlin/λ-外推。"
         "`oss_solutions` 表本会话无写权限，选型记录以 `oss_solutions.jsonl` 落工件，不谎称已入库。\n"
         "\n复现：`cd xval/XVAL-LBM-NO-DEM-REPO-IN-OSS-INVENTORY && python3 run_xval.py && python3 make_figures.py`；"
         "回归：`python3 -m pytest tests/test_xval_dem_literature_anchors.py -q` → 6 passed, 1 xfailed。\n")

md = ("# XVAL-LBM-NO-DEM-REPO-IN-OSS-INVENTORY 第 2 轮：无 OSS DEM 孪生 ⇒ 闭式解+文献锚分层验收\n\n"
      + "".join(L))

base = os.environ['TENSOR_API_BASE']


def call(path, obj, tok=None):
    req = urllib.request.Request(base + path, data=json.dumps(obj).encode(),
                                 headers={'Content-Type': 'application/json',
                                          **({'Authorization': 'Bearer ' + tok} if tok else {})})
    return json.load(urllib.request.urlopen(req, timeout=120))


tok = call('/auth/login', {'username': os.environ['TENSOR_AGENT_USER'],
                           'password': os.environ['TENSOR_AGENT_PASS']})
token = tok.get('access_token') or tok.get('token')
payload = {
    "title": "XVAL-LBM-NO-DEM-REPO-IN-OSS-INVENTORY R2：OSS 无 DEM 孪生 → 闭式解+文献锚分层验收 DEM/LBM 耦合",
    "summary": ("证得 OSS 缓存 3 仓无 DEM 实现；改用闭式解+文献锚：接触 t_c −0.63%、δmax +0.15%、e +8.90%"
                "（限幅偏置，控制臂 +0.43%）、SN 沈降 vs Clift-Gauvin ≤5.1e-6、动量账本残差 0.0；"
                "流体侧 LBM Cd 4/6 在 15% 内，最差 −28.5%。"),
    "scenario_key": "xval_dem_no_oss_twin",
    "mission_id": 18215,
    "tags": ["XVAL", "TensorLBM", "DEM", "LBM-DEM", "literature-anchor", "honest-partial"],
    "markdown": md,
    "figures": [{"name": "fig_xval_dem.png",
                 "caption": ("(a) 恢复系数随 Δt/t_c 收敛（红线带限幅、蓝线去除限幅、虚线解析）"
                             "(b) 冰球沈降终速 vs Clift-Gauvin (c) 流体侧 LBM Cd vs Clift-Gauvin 及误差"),
                 "png_base64": base64.b64encode(png).decode()}],
}
res = call('/tensor/reports', payload, token)
print('posted id=', res.get('id'), '| md bytes=', len(md))
chk = json.load(urllib.request.urlopen(urllib.request.Request(
    base + '/tensor/reports?mission_id=18215', headers={'Authorization': 'Bearer ' + token}), timeout=60))
items = chk.get('items', chk) if isinstance(chk, dict) else chk
print('readback n=', len(items) if isinstance(items, list) else '?')
if isinstance(items, list) and items:
    print('readback title:', items[0].get('title', '')[:70])