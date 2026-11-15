import json, base64, io, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(".")
A = REPO/"artifacts/xval_ac_csf_force_chain_m22329"
data = json.loads((A/"local_3pt.json").read_text())
runs = data["runs"]; static = data["static_chain_grading"]
def lab(r): return f"{r['density_ratio']:g}:1 {'+'.join(r['terms'])} ({r['formulation']})"

# ---- figures -------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for r in runs:
    s = r["samples"]
    if not s: continue
    steps = [q["step"] for q in s]
    ax[0].plot(steps, [q["dp"]/(2*0.01/r["radius"]) for q in s], label=lab(r), lw=1.6)
    ax[1].semilogy(steps, [q["max_abs_u"] for q in s], label=lab(r), lw=1.6)
ax[0].axhline(1.0, color="k", ls="--", lw=1, label="Young-Laplace $2\\sigma/R$")
ax[0].fill_between([0, 300], 0.85, 1.15, color="green", alpha=0.12, label="$\\pm$15% acceptance")
ax[0].set(xlabel="step", ylabel="$\\Delta p\\,(\\sigma/R)^{-1}$", title="Laplace jump vs analytic (sphere, $2\\sigma/R$)")
ax[0].set_ylim(0.5, 1.6); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
ax[1].set(xlabel="step", ylabel="$\\max|u|$ (lattice)", title="Spurious current (static droplet)")
ax[1].legend(fontsize=7); ax[1].grid(alpha=.3, which="both")
fig.tight_layout(); buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=130); png1 = buf.getvalue()
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
names = ["seven_point", "isotropic"]
w = 0.35
ax[0].bar([0, 1], [static["n48_R14_W3.5"][n]["max_abs_force"] for n in names], width=.5, color=["#b23","#27a"])
ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["7-point (legacy)", "isotropic D3Q19"])
ax[0].set(ylabel="$\\max|F_{st}|$", title="Korteweg force magnitude (48$^3$, R=14, W=3.5)")
ax[1].bar([0, 1], [static["n48_R14_W3.5"][n]["tangential_share"] for n in names], width=.5, color=["#b23","#27a"])
ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["7-point (legacy)", "isotropic D3Q19"])
ax[1].set(ylabel="spurious tangential share", title="Tangential (non-physical) force fraction")
fig.tight_layout(); buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=130); png2 = buf.getvalue()
plt.close(fig)
(A/"fig_laplace.png").write_bytes(png1); (A/"fig_static_chain.png").write_bytes(png2)

def row(r):
    st = r.get("steady", {})
    blow = r["trace"][0].get("step") if r.get("trace") else "-"
    return ("| {r:g}:1 | {t} | {f} | {fin} | {s} | {u} | {dp} | {ref:.3e} | {e} |").format(
        r=r["density_ratio"], t="+".join(r["terms"]), f=r["formulation"],
        fin="yes" if r["finite"] else f"no (step {blow})",
        s=len(r["samples"]) * 25, u=f"{st.get('max_abs_u_mean', float('nan')):.2e}",
        dp=f"{st.get('dp_mean', float('nan')):.3e}", ref=2*0.01/r["radius"],
        e=("**{:.2f}%**".format(100*st["dp_rel_error_mean"]) if st else "—"))

table = "\n".join(row(r) for r in runs)
passing = [r for r in runs if r.get("steady") and r["steady"]["dp_rel_error_mean"] < 0.15]
best = min(passing, key=lambda r: r["steady"]["dp_rel_error_mean"]) if passing else None
stat48 = static["n48_R14_W3.5"]

md = f"""# XVAL m22329 — Allen–Cahn / CSF 力链（第二轮：动态闭合）

锚点：feature ledger `XVAL-LBM-ALLEN-CAHN-CSF-FORCE-CHAIN`
（lssfau/lbmpy 的 allen-cahn-csf-force-chain 做法），验收 `<15%`。
仓：TensorLBM　分支：agent/m22329　日期：2026-11-09

## ① 案例与问题

第一轮（`docs/XVAL_M17410_ALLEN_CAHN_CSF_FORCE_CHAIN.md`）只对标了**力的求值**
（`F_st = mu grad(phi)` 与 lbmpy 逐位一致），并把**动态力链**（力如何进入流体步）明确留空；
`allen_cahn_lbm.allen_cahn_step` 因此整体 WITHHELD（静态液滴 5 步发散）。
本单做的是这个留空处：**力链进入流体步后，静态液滴能否守住 Young–Laplace 定律。**

算例：周期盒内的静止球滴，界面 `phi = tanh((R-r)/W)`，ρ_H=1、ρ_L=1/ratio，
ν_H=0.1、ν_L=0.02（τ = 1/2 + 3ν），σ = 1e-2，M = 5e-3，网格 20³，R=7，W=2.5
（R/W = 2.8，故意取粗以在单机秒级完成）。解析基准：**球滴 Δp = 2σ/R**（两个主曲率等值）。

## ② 根因（本单最重要的结论）

旧流体步（`full_density`）把平衡态的密度**钉死**在 ρ(φ) 上，因此
`p = cs²ρ(φ)` 也被钉死：界面处 `∇p = cs²Δρ∇φ` 是 O(0.086/cell) 的固定驱动力，
而 Young–Laplace 需要的压力跳变 σ/R 只有 O(1e-3)，两者无法平衡。
实测（10:1 球滴，48³，R=14，W=3.5）：`max|u|` 每步线性累加 ≈8.3e-2，
第 5 步 0.35、第 10 步 0.56、第 22 步非有限；重相密度 0.98→0.787、轻相 0.106→0.143
——质量从重相漏进轻相，正是"压力被钉死、力无处可去"的直接后果。

修法：改用 lbmpy 链所用的**密度偏差（压力演化）格式**——`f` 输运的是
`rho_dev = Σf`（即 p/cs²），
`f_i^eq = w_i[rho_dev + rho_0(3c·u + 4.5(c·u)² - 1.5u²)]`，`rho_0 = 1`；
压力重新自由， capillarity 才能自建。这一步把 10:1 从"22 步发散"变成本单表格中的有界稳定。

## ③ 力链消融（三项力，程序化注入）

lbmpy `hydrodynamic_force` 返回 `F = F_st(mu∇φ) + F_p(-⅓rho_dev Δρ ∇φ) + F_v((½-τ)Δρ Π^neq·∇φ)`，
本仓旧实现只有 `F_st`。三项在密度偏差格式下的实测（时间平均取后半程）：

| 密度比 | 力项 | 格式 | 稳定 | 采样步 | max\\|u\\| | Δp | 2σ/R | 误差 |
|---|---|---|---|---|---|---|---|---|
{table}

![](fig_laplace.png)

**读法**
1. `F_st` 单独 + 密度偏差格式：**误差 {best['steady']['dp_rel_error_mean']*100:.2f}% < 15%，验收达标**，
   且 max|u| ≈ 8.7e-4（毛细数 Ca ≈ 1e-3，寄生流很小）。
2. **加上 lbmpy 的 F_p/F_v 后误差反而升到 32%**（同一算例、同一初始场）。
   1:1 与 10:1 在"仅 F_st"下给出逐位相同的数，说明此时密度比根本不进入力链
   （`momentum_density="reference"` 下 ρ(φ) 只出现在 τ 场，而 ν_H/ν_L 是给定常数）
   ——即 F_p、F_v 是**唯一**携带密度比信息的通道，而按 lbmpy 公式直译后它们使 Δp 偏大 3.4 倍。
   本单不下"系数错误"的结论，只记录：**该翻译在本文格式下未通过验收**（见 ⑤）。
3. `full_density`（旧格式）在第 1 个采样点之前就非有限，与②的根因一致。

## ④ 静态力链复核（第一轮指标的复现）

48³、R=14、W=3.5 初始场上的 `F_st`：

| 算子族 | max\\|F_st\\| | 净力 \\|ΣF\\| | 伪切向占比 |
|---|---|---|---|
| 七点（旧默认） | {stat48['seven_point']['max_abs_force']:.3e} | {stat48['seven_point']['net_force']:.2e} | {stat48['seven_point']['tangential_share']:.3e} |
| 各向同性 D3Q19 | {stat48['isotropic']['max_abs_force']:.3e} | {stat48['isotropic']['net_force']:.2e} | {stat48['isotropic']['tangential_share']:.3e} |

![](fig_static_chain.png)

与第一轮结论一致：净力到机器零，七点族的伪切向占比高一个量级——本单的动态算例全部用各向同性算子。

## ⑤ 注意点与局限（不粉饰）

- **验收只在一个粗算例上达成**（20³、R/W=2.8、300 步、时间平均）。更细的矩阵
  （32³/48³/64³ × ratio 1/10/100 × 3 种力项组合，2000 步）已递交集群 `cj4046`，
  递交后长时间排队（22 min 仍在 queued），**本报告不引用其结果**。
- **F_p/F_v 翻译未通过**：偏大 3.4 倍。可能原因（未验证）：lbmpy 的速度表达式把
  `F/ρ(φ)` 代进的是 `(Σcf + F/2)/ρ_cqc` 且带 2 次子迭代，本文格式的 `rho_0 = 1` 归一
  与该式不等价；需要 lbmpy 示例脚本的 `LBMConfig(incompressible=?)` 才能定标，
  而该脚本不在已安装的 `lbmpy 2.0` 包内（`phasefield_allen_cahn/__init__.py` 为空，
  无 `create_lbm`）。
- `momentum_density="physical"`（`u = (Σcf + F/2)/ρ(φ)`）在密度偏差格式下 10–13 步发散，
  已写成回归测试 `test_physical_velocity_normalisation_is_unstable` 作为否定证据。
- 旧的 `laplace_pressure_jump` 在 ρ 比 ≠ 1 时把**体密度差**（10:1 时 0.3）
  当成毛细跳变报告，误差 ~3e4%；且球滴用 σ/R 而非 2σ/R（2 倍）。
  新助手 `laplace_pressure_jump_2phase` 修掉这两点，并由
  `test_laplace_helper_does_not_report_the_bulk_density_jump` 锁死。
- 相场仍用有限差分 AC（非 LB-AC），`interface_tracking_force` 不适用；未对标接触角边界。

## ⑥ 误差判定

| 关键量 | 基准 | 误差 | 是否 <15% |
|---|---|---|---|
| Δp（10:1，F_st 链） | 2σ/R = {2*0.01/runs[1]['radius']:.3e} | {runs[1]['steady']['dp_rel_error_mean']*100:.2f}% | **是** |
| Δp（10:1，F_st+F_p+F_v） | 2σ/R = {2*0.01/runs[0]['radius']:.3e} | {runs[0]['steady']['dp_rel_error_mean']*100:.2f}% | 否 |
| max\\|u\\|（寄生流，10:1） | 0（静止） | 8.7e-4（拉丁无量纲） | 是（Ca≈1e-3） |
| 净力 \\|ΣF\\|/max\\|F\\|（周期盒） | 0 | {stat48['isotropic']['net_force']/stat48['isotropic']['max_abs_force']:.1e} | 是 |
| 稳定性（10:1，旧格式） | 有限 | 22 步内非有限 | 否（已修） |

**结论：验收"部分达成"**——密度偏差格式 + `F_st` 链在 10:1 上 Δp 误差 <15%（9.53%），
但完整 lbmpy 三项链（含 F_p/F_v）为 32%，未达标，且未在细网格/更长积分上复核（集群排队）。
"""

(REPO/"docs/XVAL_M22329_ALLEN_CAHN_CSF_FORCE_CHAIN.md").write_text(md)

report = {
  "title": "Allen–Cahn/CSF 力链第二轮：密度偏差格式修复静态液滴发散，Δp 误差 9.5%（F_st 链）/ 32%（完整三项链）",
  "summary": "定位 allen_cahn_step 发散的根因是 p=cs²ρ(φ) 被钉死使毛细压力无法建立；移植 lbmpy 的密度偏差(压力演化)格式与 F_st/F_p/F_v 三项力链并做消融：10:1 球滴上 F_st 链 Δp 对 2σ/R 误差 9.5%(<15%)，加 F_p/F_v 后 32%(未达标)；旧格式 22 步内发散。15 个单元/回归测试全绿。",
  "scenario_key": "lbm_phasefield_force_chain_xval",
  "mission_id": 22329,
  "tags": ["TensorLBM","Allen-Cahn","CSF","force-chain","Laplace-law","verification","lbmpy","xval"],
  "markdown": md,
  "figures": [
    {"name":"fig_laplace.png","caption":"左：Δp/(2σ/R) 随时间；绿带为 ±15% 验收带。右：寄生流 max|u|。dev 格式稳定，full_density 在第 1 个采样点前非有限；加 F_p/F_v 使 Δp 偏大。","png_base64": base64.b64encode(png1).decode()},
    {"name":"fig_static_chain.png","caption":"48³ 球滴初始场上的 F_st：七点族 vs 各向同性 D3Q19 的幅值与伪切向占比（复现第一轮结论）。","png_base64": base64.b64encode(png2).decode()},
  ],
}
Path("/tmp/report.json").write_text(json.dumps(report))
print("doc+report ok; passing run:", best["density_ratio"], best["terms"], f"{best['steady']['dp_rel_error_mean']*100:.2f}%")
