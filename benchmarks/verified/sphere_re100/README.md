# sphere_re100 — 球体绕流阻力 Re = 100  ✅ VERIFIED

**状态：✅ 已入库（2026-10-05）** — 两档网格（D=40 / D=60）|err| ≤ 3% 且
随分辨率**单调下降**（+2.771% → +2.540%，跨度 0.231%）。

判定口径 = **BFL 逐链动量账本（interpolated bouzidi 边界，实验室系），末 20%
采样窗均值 Cd**；这是本案预注册（`NOTES.md` §7）指定的判定观测式，不是事后挑选。

- 参考：**Schiller–Naumann Cd = 24/Re·(1+0.15·Re^0.687) = 1.0917311**（多源核对见 `REFERENCE_AUDIT.md`）
- D=40：Cd = **1.121980** → **+2.771%**
- D=60：Cd = **1.119462** → **+2.540%**
- 稳态：末 20% vs 前一 20% 窗漂移 −0.145% / −0.042%（门 <0.3%）
- 独立 CV 动量账本对账（窗均）：0.015% / 0.008%
- 无发散；质量漂移 ≤ 0.08 ppm；Cl ≈ 1e-4（对称）

---

## 1. 物理问题

均匀来流绕球（直径 D，Re = U·D/ν = 100）。低于涡脱落下临界；定常轴对称尾流。
量 = Cd = Fx / (½ρU²·πR²)。

## 2. 参考口径（见 `REFERENCE_AUDIT.md`）

主锚（多源核对）：`Cd_ref = 24/100·(1 + 0.15·100^0.687) = 1.0917311`（Schiller–Naumann）。
交叉参考：S-N(0.681) = 1.06852、Clift–Gauvin = 1.10923。三相关式带 [1.069, 1.109]
（跨度 3.7%），落在 3% 门内的判定即落在相关式带内。门：两档 |err| ≤ 3% 且单调下降。

## 3. 力法口径（本案的关键）

### 3.1 判定路线：BFL 逐链动量账本（interpolated wall）

壁面用 **Bouzidi–Firdaouss–Lallemand 插值 bounce-back**（`compute_q_sphere` 给出每个
边界链的亚格距 q，`bfl_boundary_link_indices` 预计算边界链）表示**光滑球面**；力取
**逐边界链动量账本** `bfl_force_ledger_sparse`（实验室系，与 dense 核逐位一致），
即 `F = Σ_links Δ(动量)` 只对**壁面相邻链**求和 —— 天然是 surface 口径，但壁面是
**亚格插值**而非阶梯。

**为什么这条路线能过、而阶梯 surface-MEM 过不了**：球是**双曲率**体，阶梯体素化在
每个方向都畸变界面法向，阶梯壁的界面动量通量带一个一致性偏高的偏差（见 §5）；
插值壁（BFL）把界面还原到亚格光滑位置，消掉了这个几何偏差。库侧由 **PR #316**
（`bfl_common` 稀疏边界链核 + `general_sim` 低显存初始化）解锁大域 + 大 D 的显存。

### 3.2 替代口径（负结果，见 `staircase_surface_mem/`）

阶梯球 + 半程 bounce-back + **surface-only Ladd MEM**（`Σ_solid = Σ_surface +
Σ_interior`，只取 surface 格）在本 BGK 链上收敛到 **+9%**（D=12 1.1916 =
+9.15%，网格收敛到「错的值」，跨度 0.23%）。横向域/阻塞是**死杠杆**（lateral 16
vs 32 逐位相同），独立 CV 仪器给出同一 +10% —— 是**双曲率阶梯几何偏差**，
不是口径 bug、也不是分辨率不足。这与本案判定路线（BFL 光滑壁）的方向完全一致：
**只有光滑/插值边界表示能消掉球的这个偏差**。详见 §5。

## 4. 两档结果与收敛

| 档 | D | 域 (nz×ny×nx) | 堵塞 | 边界链 | 步数 | Cd（末 20% 窗） | err vs 1.0917311 | 稳态漂移 | CV 窗闭合 |
|---|---|---|---|---|---|---|---|---|---|
| T1 | 40 | 280×280×320 | 1.60% | 28 722 | 12 000 | **1.121980** | **+2.771%** | −0.145% | 0.015% |
| T2 | 60 | 420×420×480 | 1.60% | 64 794 | 24 000 | **1.119462** | **+2.540%** | −0.042% | 0.008% |

- 两档均 ≤3% ✅；跨度 |2.771−2.540| = **0.231% ≤3%** ✅；|err| **单调下降** ✅
- τ 校验：τ = 3·u_lb·D/100 + 0.5（D40 τ=0.56，D60 τ=0.59，Re_eff = 100.000）
- T2 = 5.7 渡时，全程质量漂移 0.0 ppm（采样分辨率内）

### 4.1 域形收敛签名（W7，为什么大域是必要的）

BFL 路线误差由**上游入口距离**主导（`far_field_bc_3d` 在入口面钳制 Dirichlet：

| up/D（lat 2.0） | Cd | err |
|---|---|---|
| 1.25 | 1.214927 | +11.28% |
| 2.00 | 1.155335 | +5.83% |
| 2.75 | 1.137950 | +4.23% |
| 3.00 | 1.134710 | +3.94% |
| 4.00 | 1.127859 | +3.31% |
| **大域 lat3/up3/down4** | **1.121980** | **+2.77%** |

五点签名 err ≈ E₀ + C·(a/x)³（R² = 0.9998）→ 入口平面 Dirichlet 钳制是主因；域形
（lat 2→3，up 1.25→3，down 2.25→4）把误差压进 3% 门。**不是**事后调参：W5-B/W7 的
预注册协议（`NOTES.md`）在跑之前已把判定观测式与收敛判据锁定。

### 4.2 位一致与独立复核

- **T1（稀疏核）对 W7 密集核存档 `h2_big.json`：240 采样 ×（cd/cl/cs/mass）
  逐位相同**（`w7_dense_h2_big.json`）。稀疏核 = 密集核位等（PR #316 断言）。
- 驱动链位等：W7 dense 驱动 == W8-A dense 驱动（D20，6 步 × 4 量逐位相同）。
- 独立 **控制体积动量平衡**（`cv_instrument.py`，壁外 6 格盒的上/下游精确链账本）
  与逐链账本窗均一致到 0.008–0.015% → 力测量不是瓶颈。

## 5. 阶梯 surface-only Ladd MEM 为何不达标（负结果，见 `staircase_surface_mem/`）

阶梯球 + 半程 BB + surface-only Ladd MEM（cylinder_3d 采纳口径）：

| 口径 | Cd（D=12 收敛） | vs SN |
|---|---|---|
| `cd_mem_surface`（壁面相邻，ghost 口径） | 1.1916 | **+9.15%** |
| `cd_mem_all`（全部 solid） | 1.2301 | +12.68% |
| `cd_mem_interior`（内部伪项） | 0.0385 | — |

- **网格收敛到错的值**：D=12 +9.15% / D=18 +8.92%，跨度 0.23% ≤3%。
- **横向域/阻塞是死杠杆**：D=10 lateral 16 vs 32 的 step-1000 值逐位相同
  （1.3177 vs 1.3178）。
- **独立仪器一致**：控制体积动量平衡 D=16 1.1985（+9.8%）/ D=20 1.2130（+11.1%），
  与 surface-MEM 同量级 → 不是口径 bug。
- **与 cylinder_3d 的差异**：cylinder 只在 xy 截面弯曲、z 向不变（**单曲率**），
  阶梯表面壳的界面通量偏差小，surface-only 口径把它是拉进参考簇（+1.62%/−0.99%）；
  球是**双曲率**，interior/total = **0.62（D12）/ 0.71（D18）**，几乎全部固体格都在
  「法向错误的 surface 壳」上，于是同一口径只把球从 +12.7% 修到 +9.15%，过不了 3%。
- 达标需**光滑边界表示**（插值/octree/BFL）—— 本案判定路线正是 BFL 插值壁。

## 6. 复现

```bash
# 判定路线（BFL 稀疏核大域阶梯）；提示：本机 SDAA 卡 ~15 GiB，D60 需稀疏核。
# 设备由 W8A_DEV 环境变量选择（默认 CUDA 上 cuda:0，本机自动回退 sdaa:0）。
cd benchmarks/verified/sphere_re100
export PYTHONPATH=/root/TensorLBM_feat2/src:/root/TensorLBM_feat2/benchmarks
W8A_DEV=sdaa:0 TMPDIR=<tmp> PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run.py --D 40 --steps 12000 --lat 3.0 --up 3.0 --down 4.0 --tau 0.56 \
    --sample 50 --cv_tail 2500            # T1（CUDA 上 ~20 min；SDAA 视卡而定）
W8A_DEV=sdaa:0 TMPDIR=<tmp> PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run.py --D 60 --steps 24000 --lat 3.0 --up 3.0 --down 4.0 --tau 0.59 \
    --sample 50 --cv_tail 2500            # T2（CUDA 上 ~2.1 h；需稀疏核，84.7M 胞）
python verify.py            # 从原始力史独立重算（result.json 为其 --write 输出）

# 替代口径（阶梯 surface-MEM，负结果）
python staircase_surface_mem/run_mem_surface.py single 12 --steps 7000 --lateral 16 \
  --up 4 --down 12 --u-in 0.06 --device sdaa:0 --compile-mode eager
```

## 7. 工件

- `run.py` —— 判定路线驱动（库全入口；BFL 稀疏/密集核）
- `verify.py` —— 判据独立重算（`result.json` 为其 `--write` 输出）
- `t1_d40big_12k.json` / `t2_d60big_24k.json` —— 两档正式力史（采样 Cd/Cl/Cs/mass + CV 尾账本）
- `w7_dense_h2_big.json` —— W7 密集核存档（T1 逐位复现）
- `chain_w7diag_D20.json` / `loop_dense_D20.json` —— 驱动链位等证据
- `cv_instrument.py` —— 独立控制体积动量平衡
- `NOTES.md` / `NOTES_sha256_chain.txt` / `formal_*.json` —— W5-B/W7 预注册协议与历史正式档
- `run_w5b_formal.py` / `verify_w5b_formal.py` / `README_w5b.md` —— W5-B 阶段（含历史 −16.6% 压力积分 / +264% MEM 失败诊断）
- `w7/` —— Wave-7 域形修复诊断档（含 h2_big 等）
- `w8a/` —— Wave-8A 破门档（稀疏核显存解锁证据、机器档案）
- `staircase_surface_mem/` —— 阶梯 surface-only Ladd MEM 负结果（`run_mem_surface.py` + 三口径 + 网格/阻塞/CV 证据）
- `REFERENCE_AUDIT.md` —— 参考口径裁定

## 8. 判定

真实模拟、无外推；判定观测式预注册（`NOTES.md` §7）；两档 |err| ≤ 3%、跨度
0.231% ≤ 3%、随 D 单调下降；位一致 + 独立 CV 对账。**→ verified ✅**

历史演化：+264.6% MEM（全 solid 求和）/ −16.6% 压力积分（B1）→ 力法修复后
+11.28%/+11.21%（B52，入口钳制地板）→ 域形修复 +4.20%/+4.01%（B58，显存天花
板挡住 D60 孪生）→ **W8-A 稀疏核解锁大域阶梯 +2.77%/+2.54%（本案）**。