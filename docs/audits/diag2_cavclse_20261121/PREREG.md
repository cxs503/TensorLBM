# DE-X-DIAG2-CAVCLSE · 预登记（跑前锁定，跑后禁改）

> 战役 DE · 单滴入深水池 Santini case-3 · 2026-11-21 · TensorLBM agent/m26887
> 锚点=NEW:DE-X-DIAG2-CAVCLSE-R1（m26883 死树抢救续建；131 行 diff 自 BB/tmp/DIAG2-CAVCLSE.kernel.patch 应用，禁重写已建部分）

## 1. 观测与物理出处

- 观测：ctrl 峰深 1.435220 D @ 11.0906 ms（abs）vs 实验 1.857434 D @ 10.39446 ms（post-impact），欠深 22.73%。
- K1-DIAG-B 定谳：生长段（step 250–1900）全程 d_cav_gap ≡ 0、p_cav ≡ 1.0、气穴连通块数 = 1 ⇒ 空穴开放单连通=**弹坑（crater）非 Rayleigh 闭气泡**；欠深是真实物理欠深（H1_under_penetration），非测量口径截断。
- 本单问题：弹坑闭合点（最深开放气-液交点/缩颈点）的运动学——逐时步位置 (x,z) 与速度，闭合时刻 t_close 与闭合速度 U_close。

## 2. 探针定义（量测口径）

- 闭合点 = 测量圆柱（半径 1.5n，与基准 open_depth 同 ROI）内、与大气（headspace，自顶播种）连通的气→液 floor 交点中**最深**者（轴 0 指向上：小索引=深，取 crossing 最小值）。
- 亚胞位置：z = k + (φ_k − 0.5)/(φ_k − φ_{k+1})，x/y 为交点胞索引；depth = (surf − z)/n（surf = level−1，D 单位）。
- t_close = 深度峰（depth 极大步）之后**第一个**闭合点开始上移（z 增大）的步；U_close = 该步闭合点垂向速度（向上为正，m/s）。
- 探针只读 fill 场，不写任何求解器状态（read-only）。

## 3. 预登记预测（跑前写死）

| 项 | 预登记内容 |
|---|---|
| t_close 预测区间 | **t_close ∈ [12, 20] ms（post-contact）**，闭合点自底向上移 |
| 证伪条件 | 闭合点先上后下（空穴重开）或 t_close 越界（<12 或 >20 ms） |
| 臂设计 | ctrl（探针关）+ cavclse（探针开），cells=8，SDAA 满窗，各 1 臂，不重跑取最好 |
| 旗标 | 与 CAVSWEEP ctrl 锚同配置：cells=8 gap 0.18 speed 0.06 rs 1.53 pd 3.0 ul 0.15 span 4.0 steps 10000 |
| 零行为改变判据 | cavclse 与 ctrl 的 history 逐字段相同；ctrl 必须逐位复现战役锚 |
| 位级锚四项 | 峰深 1.4352204748 D @ 11.0905714286 ms；健康窗 24.0052 ms（post-contact）；mach_max 0.2598076211；contact_step 45 |
| 停止规则 | window_complete（post-contact 24.005 ms）即止；失败臂照登不复活 |

## 4. 抢救补丁缺陷修复记录（实现修复，非预测修改）

m26883 抢救 diff 含两处实现缺陷，本单最小修复（各 1 行级），预测/判据/臂设计一字未改：

1. **门控**：cav_closure 块原嵌于 `if (diag_cavity or diag_budget)` 内，单独开 `--cav-closure` 永不触发 → 外层条件加 `or cav_closure`。
2. **方向**：`_closure_summary` 原判 `z < prev_z`（=下移）并以此定 t_close，与轴约定（小索引=深，上移=z 增大）相反 → 改 `z > prev_z`，速度符号同步 `(z − prev_z)`。

## 5. 交付

- 闭合点轨迹 (x, z vs t) 数字表 + t_close + U_close（程序化提取，零手抄）。
- 卡落共享正典位 /root/tp-agent/drop_entry/BB/X/DIAG2-CAVCLSE.md + 树内双写。
