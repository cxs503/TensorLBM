# W8-A：BFL 稀疏核解锁 + 球 Re=100 大域阶梯（D40/D60）

W7 破门路径（BFL 核显存优化）的执行记录。库侧改动 = PR #316
（`bfl_common` 稀疏边界链核 + `general_sim` 低显存初始化，纯增量、
位一致证据在该 PR）。本目录记录其解锁的判定档。

## 设置

- 达标形 lat3.0/up3.0/down4.0（W7 h2_big 家族），u_lb=0.05，
  MRT `collide_mrt3d_low_memory`，硬 far field，每 200 步质量校正。
- T1 = D40（280×280×320，28 722 链，τ=0.56，12 000 步）
  = W7 h2_big 逐位复跑。
- T2 = D60（420×420×480 = 84.7M 胞，64 794 链，τ=0.59，24 000 步
  = 5.7 渡时）。两档堵塞 1.60%。
- 主观测：BFL 逐链动量账本（实验室系）**末 20% 采样窗**均值 Cd
  （W5-B/W7 同口径；稳态门 = 末 20% vs 前一 20% 窗漂移 <0.3%）。

## 参考

    Cd_ref(Re=100) = 24/100·(1 + 0.15·100^0.687) = 1.0917311（Schiller–Naumann）
    交叉：S-N(0.681) = 1.06852、Clift–Gauvin = 1.10923
    门：两档 |err| ≤ 3% 且 |err| 随 D 单调降

## 判定表（verify.py 从原始力史独立重算）

| 档 | Cd（末 20% 窗） | err | 稳态漂移 | CV 对账（窗均） | 链接数 |
|---|---|---|---|---|---|
| T1 D40 | 1.121980 | **+2.771%** | −0.145% | 0.015%（n=50） | 28 722 |
| T2 D60 | 1.119462 | **+2.540%** | −0.042% | 0.008%（n=50） | 64 794 |

**阶梯判决：PASS** — 两档 |err| ≤ 3% 且随 D 单调降（2.771% → 2.540%）。
球 Re=100 直接力法（W5-B FAIL → W7 域形定源 → W8-A 破门），已于 **2026-10-05 晋级
`benchmarks/verified/sphere_re100/`**（本目录为 W8-A 档原始存档，已随案目录移入）。

## 位一致与复现

- T1（稀疏核）对 W7 存档 `../w7/diag/h2_big.json`（密集核）：
  240 采样 × cd/cl/cs/mass **逐位相同**（含 mass_drift −0.0797 ppm 末位）。
- τ 校验过（τ = 3·u_lb·D/100 + 0.5）；re_eff = 100.0；无发散；
  T2 全程质量漂移 0.0 ppm（采样分辨率内）。

## 显存与速度（解锁证据，机器档案）

| 项 | 值 |
|---|---|
| D60 大域稀疏路线 600 步峰值 | 26.67 GiB alloc / 30.20 reserved / 30 658 MiB（nvidia-smi） |
| 同域密集核对照 | 第 1 步 OOM（再申请 5.99 GiB，余 2.55 GiB / 31.40 GiB 卡） |
| setup 第二 OOM 点 | 全量 `equilibrium3d` 表达式树 ~33 GiB → 换 `equilibrium3d_low_memory`（位同等，PR #316 断言） |
| 核速度 sparse/dense 中位比 | 0.41（1.2M 胞）/ 0.14（9.6M 胞）——更快 |

## 文件

- run.py — 驱动（库全入口；铁律自查零命中；需 PR #316 的稀疏入口）
- verify.py — 判据独立重算（`result.json` 为其输出）
- t1_d40big_12k.json / t2_d60big_24k.json — 正式档（采样力史 + CV 尾账本）
- mem_sparse_D60big.json / mem_dense_D60big.oom.json / speed_probe.json /
  bit_check.json — 解锁证据机器档案
- result.json — 判定汇总

## 复现

```bash
cd <repo>/benchmarks/pending/sphere_re100_mem/w8a
CUDA_VISIBLE_DEVICES=5 TMPDIR=<tmp> PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run.py --D 60 --steps 24000 --lat 3.0 --up 3.0 --down 4.0 --tau 0.59 \
    --sample-every 50 --cv-tail 2500   # T2；T1 同参数 --D 40 --tau 0.56 --steps 12000
python verify.py
```
