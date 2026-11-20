原 60 行 → 压缩 54 行
# DE-A4 前役经验卡 · 液滴入水多相流战役史
**仓库**: TensorLBM · **战役**: DE · **臂**: A4 · **日期**: 2026-11-19
**来源**: 5 份共享镜像（MP_DAMBREAK_S0_VERDICT/AUDIT、phasefield_static_droplet_r1、superhydrophobic_slip_report、XVAL_LBM_CUPY_DROPIN_NO_KERNEL_FUSION）

## 1) 溃坝战役判据与教训
### 已验证结论（高证据级）
- **S1 判据三件套**：①直接观测量 ≤3%（≥2 档网格）；②≥2 档单调收敛（|ΔE|≤3% 档间差）；③质量机器精度守恒（drift ≤ ~1e-5）。锚点 M&M 1952：T=1→X=1.326, T=2→X=2.360, T=2.96→X=3.670。
- **路线 A（2D 自由面 LBM FS2D）为 S1 首选**：最佳 +3.34/-1.62/-3.13，T=2/T=2.96 已过 3%，仅 T=1 超 0.34 点；质量守恒 PASS（drift ≤ 2.4e-5）。
- **waLBerla 论文原码对照**：论文自身 +5.9~+13.1 点全面 FAIL，本库比论文近 3-8 点——门禁严于文献自身水平。
- **五层根因链（前 2 层已修）**：①重力双重计力（g_eff=1.5g@τ=1，velocity_shift+Guo 叠加）→ 已修 opt-in Guo；②壁面镜面反射缺陷 → 已修；③refilling 方案（EQ→AVG 改善 8.6 点）；④τ 窗口闭合（T=1 需 τ∈[0.70,0.75]，T=2.96 需 τ∈[0.55,0.70]，交集为空）；⑤receiver gate 结构性核心（前锋推进与账本闭合同一开关）。
- **dam_break_3d_mm（m91）TERMINAL**：12/12 反收敛，质量漂移 -49.46%/-62.19%，缺陷随分辨率放大——3D 留 S2，勿在 3D 原版继续投入。
- **dam_break_sc（m163）TERMINAL**：SC94 psi_exp 密度比仅 12.3 vs 需 ~800；反向 G 依赖（降 G 修晚期炸早期）为结构性障碍。

### 未验证观察
- 高密度比 EOS（PR@0.55 密度比 400-557）是否解反向 G 依赖——**未实测**。
- MCMP 路线溃坝适用性——**零实测**，从共存标定开始，物理不确定性最高。

## 2) 相场静态液滴校准
### 已验证结论
- **D3Q19 CH 静滴初始化**：`tanh((R-r)/(sqrt(2)*interface_width))`，r 为 minimum-image 周期距离；球心/界面可跨周期面；radius/interface_width/shape/center 均验证。
- **Laplace-style 诊断链明确 WITHHELD**：无真实热力学 pressure field，不虚构 pressure；`status="withheld"`、`physical_acceptance=False`——**不是** Laplace PASS/表面张力标定/静滴平衡证明。

### 未验证观察
- 该路径不构成生产 D3Q19/CH solver 的物理接受准则——需另建真实压力场验证链。

## 3) 表面张力/Laplace 标定既有数字
### 已验证结论
- **SOY 自由能静滴（R=15, xi=3.0, σ=0.01, κ=0.0225）**：Laplace 压差 dp=6.640176e-04 vs 解析 σ/R=6.666667e-04，**误差 -0.3974%（PASS <15%）**；NumPy vs PyTorch 后端差 8.36e-12%（PASS）；TensorLBM vs OSS 实跑差 0.0174%。
- **独立表面张力路线**：σ_cap=Σ(κ|∇ψ|²)/(2πR)=9.711077e-03 vs 模型 σ=0.01，偏差 **-2.889%**——自由能泛函与机械压力路线一致，Laplace 匹配非单公式伪影。
- **SOY 参数制式**：κ=0.75·σ·xi；a=-2κ/xi²；τ_g=1/(3-√3)=0.788675；p=ρ/3+ψμ。
- **Philip(1972) 超疏水锚**：b_eff=a/π·ln(sec(πφ/2))；φ=0.5→b_eff/a=ln(2)/(2π)=0.110318（机器精度）；LBM 显式 vs Philip O(1) 比 <10×；Navier 等效 <3×。
- **M4 高密度比 Laplace（Wave-10）**：PR@0.55 σ=0.32023/0.32649/0.32942（L=128/192/256），残差 0.045-0.100%，密度比 400-557——**PASS（定案）**。

### 未验证观察
- 静滴 Laplace 通过 ≠ 溃坝/大变形/冲击工况适用（物理差异大，m163 评估）。
- 伪电流 u_max 高于良调文献锚 5-25×——可能污染前沿测量，需监控。

## 4) 可复用验证方法论
### 已验证结论
- **判据设计原则**：直接观测量（前沿位置/压差/深度）必须对文献锚点 ≤3%；≥2 档网格单调收敛（误差严格递减）；质量/体积漂移 ≤ 机器精度量级（~1e-5）；参考值先冻结（chmod a-w）再跑。
- **已知的坑**：
  ① 时间轴口径错误——旧口径 `T=t·√(g/a)` 缺 √2，须用 `T_MM=t·√(2g/a)`（m163 已修正）；
  ② 参考表映射错误——把 M&M T=3 值 2.7 映到 T=2（run.py 旧口径）；
  ③ 力方案双重计力——velocity_shift+Guo 叠加致 g_eff=1.5g；
  ④ 收敛非单调——SOY 静滴瞬态在解析值附近振荡（-36.9%→+0.85%），须报末样本+带宽；
  ⑤ R/xi 收敛阶假说被证伪——|err| 随 R 增大（-0.25%@R=10→-0.60%@R=20），残差由采样瞬态主导而非曲率项；
  ⑥ 界面胞膨胀——interface 计数随细化恶化（976→2006），须监控。
- **验证方法论**：三臂交叉（本库 NumPy / 本库 Torch / OSS 实跑 exec）+ 独立物理路线（σ_cap 无压力场）+ 参数敏感性扫描（xi/gamma/visco）+ 预声明阈值（predeclared.json 先于跑）。

### 未验证观察
- 高密度比 EOS 在溃坝大变形工况的适用性——需 ~10 GPU·h 参数扫描，无保证过门。

---
**出处索引**：①MP_DAMBREAK_S0_VERDICT.md（m206 裁决）；②MP_DAMBREAK_S0_AUDIT.md（m206 审计）；③phasefield_static_droplet_r1.md；④superhydrophobic_slip_report.md（m7455）；⑤XVAL_LBM_CUPY_DROPIN_NO_KERNEL_FUSION.md（m17979）。
