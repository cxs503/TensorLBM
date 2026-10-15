# suboff_highre_resistance — SUBOFF 高 Re 全附体阻力（待验证 ⛔ 网格不可达）

**状态**：⛔ 精度在单卡 32GB 可达网格（400×200×200）下系统性失败——正式五锚档
M1 0/5（误差 +1295.5..+2962.7%）、M2 非单调、overall FAIL；根因裁定=分辨率饥饿
（y+_mean 1878..4660 全程超壁定律可信域）。壁模型架构本身成立（摩擦塌缩已修、
周期-2 振荡已定罪并部分压制）。**晋级动作=真超算（多卡/HBM，~10⁸ 胞级网格）按
§晋级路径复跑；在 5090 级硬件上参数面不可行。**

## 物理问题

AFF-8 全附体 SUBOFF 模型（L_max=4.356 m，WSA=5.998 m²，参考面积=裸壳解析面积）
五速拖曳阻力 CT 对标 EFD：CT=RT/(0.5·ρ·V²·S)，ρ=1000 kg/m³，ν=1.139e-6 m²/s，
Re 以 L_max 计（1.17e7..3.15e7）。主参考=CFD Letters 16(10) 2024 Table 3 EFD 列
（本体 [24]=ADA359226）；交叉验证=Flow360 DDES 误差表反推，幅值差 ≤2.02%。
幂律拟合 CT(Re)=0.0169331·Re^−0.093465（5 点 log-log LSQ）。

## 参考锁（先锁后跑，2026-10-10 10:29）

| 文件 | md5 |
|---|---|
| `refs_lock.md`（本目录） | `4bb2dc0562378ab833de0dad3c3cf29f` |
| `refs_lock.json`（本目录，机器规范源） | `d27b28ac0974b1be12198d667048500c` |
| refs_lock.py（生成器，留 staging） | `7b1e905686bb4bf7e9b429b77e1810d6` |

五锚（`refs_lock.json` `anchors[]`）：V=3.050/5.144/6.100/7.160/8.230 m/s，
RT=102.30/283.80/389.20/526.60/675.60 N，CT=0.003667→0.003326（严格降）。
预注册 `prereg.md`（冻结 a72fcd30 @2026-10-11T02:21:00，append 后 c91b987a）：
判据 M1=|CT_sim/CT_EFD−1|≤5% 于 ≥4/5 锚；M2=CT_sim 沿锚非增；**判决只读
formal r1–r5**，p0–p38 全部探针不参与判决（防选购）。

## 计算结果现状

### 正式五锚档（判决层，`formal/formal_verdict.json`）

配置=prereg §3 冻结（bgk_smag C_s=0.05、双黏度 collision_nu=1.2e-4
（resolved_Re=1e5，τ=0.50036）+ 逐锚 wall_nu、BFL halfway 链
`wall_model_slip`+`reichardt`、y+ 诊断带 [30,5000]、400×200×200、
16000 步 warmup 8000、sponge 24/0.2、五锚各独立全新跑）：

| 锚 | Re | CT_EFD | CT_sim | 误差 | Ct_fric | Ct_fric/ITTC | y+_mean | 奇偶 rel | 观察者差 | M1 |
|---|---|---|---|---|---|---|---|---|---|---|
| r1 | 1.166e7 | 0.003667 | 0.051173 | **+1295.5%** | 7.37e-3 | 2.522 | 1878 | 0.326 | 59.0% | ✗ |
| r2 | 1.967e7 | 0.003576 | 0.075394 | **+2008.1%** | 6.73e-3 | 2.515 | 3031 | 0.234 | 73.2% | ✗ |
| r3 | 2.333e7 | 0.003488 | 0.061713 | **+1669.5%** | 6.48e-3 | 2.491 | 3519 | 0.696⚠ | 65.7% | ✗ |
| r4 | 2.738e7 | 0.003425 | 0.075225 | **+2096.3%** | 6.35e-3 | 2.502 | 4080 | 0.604⚠ | 72.0% | ✗ |
| r5 | 3.147e7 | 0.003326 | 0.101863 | **+2962.7%** | 6.26e-3 | 2.525 | 4660 | 0.511⚠ | 82.4% | ✗ |

⚠=模式卫生披露位 >0.5（prereg §2 披露条款，非门）。prereg §4 诚实预测
"五锚全 FAIL M1"命中。**恒定比读法**：Ct_fric/ITTC(锚 Re)≈2.49-2.53 五锚近常数、
Ct_fric/ITTC(resolved 1e5)≈0.75-0.88——壁链交付的是**分辨率 Re（1e5）**的摩擦，
水平误差为壁链-分辨率结构性，非锚间漂移。

### 已确立的架构与诊断结论（探针层 p0–p38，全部机器工件）

1. 旧 wall_function 路径物理 FAIL（τ≈0.5 时 Ct_fric 1.8e-3→5e-5 单调塌缩、
   Ct_pres 恒 −2e-3；根因=Guo 体力正反馈+近壁动量无补充→自由滑移）→ 已弃。
2. 修复架构=flat_plate_wall_model 双黏度（`src/tensorlbm/suboff_wallmodel_les_runner.py`，
   campaign 树 md5 `78c12af0fe33240971ebd076aec8518a`，本库入档字节一致）：
   摩擦塌缩已修复（小网格 Ct_fric 稳定 2.65e-3）。
3. 周期-2 振荡定罪（p11-p22 开关矩阵）：唯一灭者=纯 BGK 同 τ（Mx 交替 10 步内
   死灭；壁模型/Guo/sponge/双 outer/positivity/cumulant/TRT/RLKB/MRT 全照旧）。
4. 振荡压制部分成立：reichardt WM 臂 mode_rel_ct_cv≈0.092（formal r1 同 0.33），
   但 stationary 无 WM 臂把周期-2 模用无滑移壁泵回（奇偶不对称）。
5. 探针括号（anchor 1）：[无 WM：Ct_fric 1.9e-4 塌缩、+801.7%、振荡在] vs
   [WM reichardt：Ct_fric 7.27e-3 过预测、+1689.2%、振荡压住] vs [WM spalding：
   Ct_wall_total=8.06 量级崩坏、+2283.4%、观察者差 10163.1%]——真值被塌缩−/过预测+
   夹住，壁模型在 y+≈1900 外推失控。

## 不达标清单

1. M1：五锚误差 +1295.5..+2962.7%，0/5 过 5% 门（差 ~26-60 倍）。
2. M2：CT_sim 沿锚非单调（0.0512→0.0754→0.0617→0.0752→0.1019）。
3. y+_mean 1878..4660 超壁定律可信内插域（prereg 诊断带 [30,5000] 形式过、
   物理域远低于此）。
4. 480×240×240 全网格 32GB 单卡 OOM（Q19 全域张量 ~37GiB）；内存三件套
   （guo 分块/低显存宏观/del collided）后仍需降网格。

## 根因/诊断

**分辨率饥饿（参数面不可行），非模型定论性缺陷**：恒定 Ct_fric/ITTC 比读法
（上表）+探针括号表明壁链在 400 网格下工作在分辨率 Re 而非锚 Re；y+ 全程超域
使壁定律外推失控。误差两头（无 WM 塌缩 −/WM 过预测 +）都不含真值邻域。

## 晋级路径（HPC 复跑协议）

1. 网格 ≥480×240×240 起步（消 OOM）；**过门预估需 400 网格 ~3× 线加密
   （~10⁸ 胞级，≈1200×600×600，张量足迹 ~350-400GiB @Q19）→ 多卡/HBM**。
   目标 y+ 带入 [30,500]。
2. 顺序：先 anchor-1 单速全新初场 16k 步；门=Ct_cv≤5%、mode_rel_ct_cv≤0.15、
   观察者差≤10%、正性 limited fraction≤2%、CT vs 0.003667 误差 ≤±10%
   （或预注册披露带）；过门后铺满五速出 CT(U) 对 `refs_lock.json` 全锚。
3. 必带小子集：resolved_Re {1e5,3e5,1e6}×壁定律 {reichardt, spalding 修复版}
   ——spalding 崩坏（Ct_wall_total=8.06/观察者差 10163%）需在密网格复判是否
   y+ 外推伪影。
4. 入口：`src/tensorlbm/suboff_wallmodel_les_runner.py`（本库入档与 campaign 树
   字节一致，md5 `78c12af0fe33240971ebd076aec8518a`）；逐探针真值=staging 各
   p*.json 内嵌 config。
5. 已知坑：驱动脚本 `hull_type` kwarg 与 runner config 签名断裂（复跑前先修）；
   480 OOM→内存三件套；**短探针/续跑结论铺量前必须全新初场长跑复核**；
   跨跑逐位门禁用（DG0 规），对拍用统计口径；判据数字零手抄、prereg 冻结先于
   计算、append-only。

## 文件

- `run.py`（同 `src/tensorlbm/suboff_wallmodel_les_runner.py`，字节一致入档）
- `refs_lock.md` / `refs_lock.json` — 参考锁
- `prereg.md` — 预注册（append-only 链）
- `probe/p37_bracket_summary.json` — 探针层括号摘要（reichardt WM vs stationary）
- `formal/formal_verdict.json` — **判决层**五锚机器判决
- staging 完整工件（p0–p38、r1–r5 含 `_fields.pt`、refs/ 文献）：
  `/nfs/wangxi/runs/suboff_highre_20261010/`（入口文档
  `ARCHIVE_PENDING_VERIFICATION.md`）

<!-- PROVENANCE
[
 {"v": 0, "f": "formal/formal_verdict.json", "k": "M1_pass_count"},
 {"v": 1295.5503183094308, "f": "formal/formal_verdict.json", "k": "table[0].error_pct"},
 {"v": 2008.1486846731239, "f": "formal/formal_verdict.json", "k": "table[1].error_pct"},
 {"v": 1669.468406207324, "f": "formal/formal_verdict.json", "k": "table[2].error_pct"},
 {"v": 2096.263939490076, "f": "formal/formal_verdict.json", "k": "table[3].error_pct"},
 {"v": 2962.6791299912215, "f": "formal/formal_verdict.json", "k": "table[4].error_pct"},
 {"v": 0.051173484438575216, "f": "formal/formal_verdict.json", "k": "table[0].CT_sim"},
 {"v": 0.10186265785873225, "f": "formal/formal_verdict.json", "k": "table[4].CT_sim"},
 {"v": 0.007367567728682241, "f": "formal/formal_verdict.json", "k": "table[0].Ct_fric"},
 {"v": 2.5219784208719593, "f": "formal/formal_verdict.json", "k": "table[0].fric_over_ITTC"},
 {"v": 2.4914699568829377, "f": "formal/formal_verdict.json", "k": "table[2].fric_over_ITTC"},
 {"v": 2.524742006740258, "f": "formal/formal_verdict.json", "k": "table[4].fric_over_ITTC"},
 {"v": 0.8841081274418688, "f": "formal/formal_verdict.json", "k": "table[0].fric_over_ITTC_resolved1e5"},
 {"v": 0.7517193586218571, "f": "formal/formal_verdict.json", "k": "table[4].fric_over_ITTC_resolved1e5"},
 {"v": 1878.2914085388184, "f": "formal/formal_verdict.json", "k": "table[0].y_plus_mean"},
 {"v": 4660.108557128906, "f": "formal/formal_verdict.json", "k": "table[4].y_plus_mean"},
 {"v": 0.6964504901915316, "f": "formal/formal_verdict.json", "k": "table[2].odd_even_rel_ct_cv"},
 {"v": 0.0036669035696661055, "f": "refs_lock.json", "k": "anchors[0].CT"},
 {"v": 0.0033259330649836705, "f": "refs_lock.json", "k": "anchors[4].CT"},
 {"v": 0.016933131677714163, "f": "refs_lock.json", "k": "powerlaw_fit.A"},
 {"v": -0.09346548072606699, "f": "refs_lock.json", "k": "powerlaw_fit.m"},
 {"v": 1689.2444510550388, "f": "probe/p37_bracket_summary.json", "k": "[0].reference_error_pct"},
 {"v": 801.7024863376314, "f": "probe/p37_bracket_summary.json", "k": "[1].reference_error_pct"},
 {"v": 0.007270038630671284, "f": "probe/p37_bracket_summary.json", "k": "[0].Ct_fric"},
 {"v": 0.00018994255261229205, "f": "probe/p37_bracket_summary.json", "k": "[1].Ct_fric"},
 {"v": 0.09220189681903314, "f": "probe/p37_bracket_summary.json", "k": "[0].mode_rel_ct_cv"},
 {"v": 3.1732447140037428, "f": "probe/p37_bracket_summary.json", "k": "[1].mode_rel_ct_cv"},
 {"v": 2283.3580904595165, "f": "/nfs/wangxi/runs/suboff_highre_20261010/probe/p35s_bgksmag_spalding_a1.json", "k": "result.reference_error_pct"},
 {"v": 10163.123355836495, "f": "/nfs/wangxi/runs/suboff_highre_20261010/probe/p35s_bgksmag_spalding_a1.json", "k": "result.observer_difference_pct"},
 {"v": 8.05691303795431, "f": "/nfs/wangxi/runs/suboff_highre_20261010/probe/p35s_bgksmag_spalding_a1.json", "k": "result.Ct_wall_total"},
 {"v": 45.272155571978104, "f": "/nfs/wangxi/runs/suboff_highre_20261010/probe/p35s_bgksmag_spalding_a1.json", "k": "result.drag_stationarity.linear_trend_pct"}
]
-->
