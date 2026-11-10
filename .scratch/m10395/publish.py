import base64, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api import call, login

W = "/root/tp-agent/work/m10395"
camp = json.load(open(f"{W}/artifacts/p88w1/campaign.json"))
crit = camp["criteria"]
md = open(f"{W}/docs/P88W1_N01_sloshing_forced.md", encoding="utf-8").read()

figs = []
caps = {
    "fig1_resonance_curve.png": "图1 受迫扫描共振曲线（实测一阶谐波幅值 vs 有限时间线性参考）与奇偶性诊断",
    "fig2_free_decay_sensitivity.png": "图2 自由衰减时历与频率误差（5% 门限）",
    "fig3_pressure_and_waveform.png": "图3 舱壁压力时历（线性支/冲击支）、自由面波形与闭式冲击界",
    "fig4_grid_convergence.png": "图4 网格收敛：H/32 vs H/64 的压力幅值与壁面爬高",
}
for name, cap in caps.items():
    p = f"{W}/artifacts/p88w1/figures/{name}"
    if os.path.exists(p):
        figs.append({"name": name, "caption": cap, "png_base64": base64.b64encode(open(p, "rb").read()).decode()})

summary = (
    f"LNG 矩形舱(L1×B0.5×H0.4m, 50%充装)受迫晃荡验证面: "
    f"f1={crit['sweep']['f1_theory_hz']:.4f}Hz 实测误差 {crit['a_f1_decay']['value_pct']:+.2f}%(衰减)/"
    f"{crit['a2_f1_sweep']['value_pct']:+.2f}%(受迫扫描峰), f2 {crit['b_f2_decay']['value_pct']:+.2f}%; "
    f"壁面压力 vs 线性锚 {crit['b_pressure_linear']['value_pct']:+.2f}%, 冲击支 vs ρg·η {crit['b_pressure_impact']['value_pct']:+.2f}%; "
    f"2×网格加密压力幅值变化 {crit['d_grid_convergence']['value_pct']:.2f}%; "
    f"奇偶性 a2/a1≤{crit['parity']['value']:.4f}; 21 pytest 全绿; 弱可压敏感度支 honest FAIL(账本破裂)如实登记。"
)

body = {
    "title": "P88W1-N01 LNG液舱晃荡验证面：部分充装矩形舱的受迫共振与舱壁冲击压力",
    "summary": summary,
    "scenario_key": "lng_sloshing",
    "mission_id": 10395,
    "tags": ["sloshing", "LNG", "free-surface-LBM", "resonance", "impact-pressure", "GCI", "P88W1-N01", "TensorLBM"],
    "markdown": md,
    "figures": figs,
}
st, tok, _ = login()
print("login", st)
st, d = call("tensor/reports", "POST", body, token=tok)
print(st, json.dumps(d, ensure_ascii=False)[:600] if isinstance(d, dict) else str(d)[:600])