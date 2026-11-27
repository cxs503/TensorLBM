"""Build the platform progress report payload for P493-N02 (figures inlined)."""
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "invariance" / "P493-N02"
DOC = ROOT / "docs" / "P493_N02_INVARIANCE_MATURITY.md"

report = json.loads((ART / "report.json").read_text())
cr = {c["id"]: c for c in report["criteria"]}
meas = report["measured"]

def fig(name, caption):
    b = (ART / "figures" / name).read_bytes()
    return {"name": name, "caption": caption, "png_base64": base64.b64encode(b).decode()}

markdown = DOC.read_text()

# prepend a compact summary block (the document itself is the paper-grade body)
head = f"""# P493-N02 · TensorLBM 不变性群验证面成熟化

**结论：7/11 判据通过（honest FAIL 4 项，未覆盖）。**

| 判据 | 判定 | 关键数字 |
|---|---|---|
| C1 周期槽道（壁面在格点中点）观测阶 | PASS | p̄ = {cr['C1']['per_case']['interpolated']['mean_observed_order']:.2f}（目标 2±0.3） |
| C1b 斜槽 Poiseuille 观测阶 | FAIL | θ=0/15/30/45 → {', '.join(f"{v['mean_observed_order']:.2f}" for v in cr['C1b']['per_theta'].values())} |
| C9 有效壁面位置取向不变性 | FAIL | δ = {', '.join(f"{v['delta_finest']:+.3f}" for v in cr['C9']['per_theta'].values())} 格（阈值 0.1） |
| C2 Cd 取向散布 | FAIL | Re=100: {cr['C2']['by_re']['100']['max_spread']:.1%}；Re=200: {cr['C2']['by_re']['200']['max_spread']:.1%} |
| C3 St 取向散布 | FAIL | Re=100: {cr['C3']['by_re']['100']['max_spread']:.1%}；Re=200: {cr['C3']['by_re']['200']['max_spread']:.1%} |
| C4 伽利略叠加缺陷 | PASS | max |u(U)−u(0)−U|/U = {cr['C4']['max_defect']:.2e}，随 Ma 单调 |
| C5 Ma² 偏离律 | PASS | 双对数斜率 {cr['C5']['slope']:.3f}，R²={cr['C5']['r2']:.4f} |
| C6 负控（半步反弹，亚格点壁面） | PASS | 半步 p̄={cr['C6']['halfway_mean_order']:.2f}，误差 {cr['C6']['error_ratio_finest']:.1f} 倍 |
| C7 逐位可复现 | PASS | {cr['C7']['n_unique']} 个唯一哈希 |
| C8 压缩性门禁 | PASS | max|Δρ|={cr['C8']['max_seen']:.2e} |
| C10 体相（无壁面）取向不变性 | PASS | max|λ/λ_th−1|={cr['C10']['max_rel_deviation']:.2e} |

## 关键结论（三句话）
1. **无壁面体相**取向不变（1.4e-3）+ 伽利略 Ma² 律（斜率 2.000）成立 → 体相求解器可发合格证。
2. **有壁面**时插值反弹只提供 O(h) 级壁面定位（|δ| 0.01–0.26 格），斜槽速度场拿不到二阶；同一现象在**完全周期、无进出流**对照中复现 → 不是边界条件 bug，是 bounce-back 家族上限。
3. **圆柱阻力**在双周期盒内未验证（Cd 比文献高 6–10 倍、取向散布 74–103%，而 St 只差 18–28%）→ 判 FAIL 并给出整改路径。

---

"""
markdown = head + markdown.split("\n", 1)[1]

payload = {
    "title": "P493-N02 TensorLBM 不变性群验证面成熟化：7/11 判据通过（体相取向不变 + Ma² 律成立；壁面定位与圆柱阻力 honest FAIL）",
    "summary": (
        "给 TensorLBM 的不变性验证面升级并复检：新增零压差（体力驱动）斜槽算例、"
        "任意亚格点壁面的周期槽道分辨率算例、共格波矢体相取向扫描、两段式驱动的圆柱算例，"
        "新增壁面滑移/有效壁面位移/压缩性门禁诊断，重定义伽利略残差为方案级叠加缺陷。"
        "结果 7/11 判据通过：体相取向不变性 1.4e-3、伽利略 Ma² 律斜率 2.000（R²=1.0000）、"
        "周期槽道 p̄=1.96、负控确认半步反弹严格一阶（误差 14.5 倍）、逐位可复现。"
        "honest FAIL：斜槽速度场非二阶（壁面定位误差 O(h)，已在完全周期无进出流对照中复现），"
        "亚格点壁面有效位置 δ 最高 0.26 格，圆柱阻力在双周期盒内被盒尺度非定常污染"
        "（Cd 高 6–10 倍、散布 74–103%）。"
    ),
    "scenario_key": "invariance_certificate",
    "mission_id": 9483,
    "tags": ["TensorLBM", "invariance", "Galilean", "mesh-orientation",
             "bounce-back", "verification", "P493-N02"],
    "markdown": markdown,
    "figures": [
        fig("fig1_resolution.png",
            "图1 分辨率收敛：左=周期槽道（q=0.5 二阶；q=0.25 半步反弹严格一阶，负控）；"
            "右=斜槽四取向误差随分辨率。"),
        fig("fig2_wall_offset.png",
            "图2 由壁面滑移反演的有效壁面位移 δ(θ,H)：不随分辨率消失=壁面定位误差。"),
        fig("fig3_galilean.png",
            "图3 左=衰减率相对偏离随 Ma 的幂律（斜率 2.000）；右=伽利略叠加缺陷在 Ma≤0.05 上为常数平台。"),
        fig("fig4_cylinder.png",
            "图4 圆柱 Cd、St 随格子取向：Cd 散布 74–103%（阻力测量未验证），St 18–41%。"),
        fig("fig5_error_bands.png",
            "图5 斜槽误差沿壁距分布：近壁带与通道中部共同构成，中部来自质量守恒对近壁滑移的重分配。"),
    ],
}
Path(ROOT / ".scratch" / "report_p493.json").write_text(json.dumps(payload))
print("payload bytes:", len(json.dumps(payload)), "figures:", len(payload["figures"]))