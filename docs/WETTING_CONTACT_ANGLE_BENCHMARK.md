# 静态接触角基准工装（SCMP 润湿壁面力）— 指针

**位置**：`tests/benchmarks/wetting_contact_angle/`（工装 + 结果 + 图 + 报告）
**锚点**：GAP-LBM-MCMP-POISEUILLE 的润湿判据
**生成报告**：`tests/benchmarks/wetting_contact_angle/results/REPORT.md`（数字全部由
`results/contact_angle_sweep.json` 程序化注入，本文件不复制任何测量数字，避免手抄）

## 一句话结论

在库现有 SCMP 壁面伪势（``psi_w = f_w·psi(rho_l)``）上，预注册的
`|θ_measured − θ_prescribed| < 5°` 判据 **两条处方律都没通过**（honest FAIL，未放宽判据、
未丢案例）：Young 关系与实测角系统性偏离（单一符号）；而用粗网格实测标定曲线外推到细网格
（hold-out）仍超差，因为同一 ``f_w`` 下测角随分辨率漂移、且静止角本身对初始形状有历史依赖。
报告逐项给出根因证据：解析帽对照（估计器偏差 <1.2°，证明漂移不是测量方法造成的）、
同 ``f_w`` 跨网格对照、同 ``f_w`` 双初始帽（亚稳劈裂）、以及强润湿端退化为膜导致
``NO_INTERFACE`` 的量程限制。

## 对下游（GAP-LBM-MCMP-POISEUILLE）的可操作含义

1. **不要**把 ``f_w`` 当 Young 角设定器使用；若需指定接触角，必须用**同分辨率**的标定曲线
   反解（本工装 `wetting_bench.calibration_curve` / `invert_calibration` 可直接复用）。
2. 报告接触角时必须声明初始帽角与网格；否则数值不可比（亚稳劈裂量级见报告表）。
3. MCMP（双组分）路径**没有**润湿壁面力：`src/tensorlbm/multiphase.py:495`
   `sc_two_component_force` 只有 `solid_mask`（把固体密度清零 = 干壁），无壁密度/壁 ψ 参数，
   故锚点的 MCMP 半边仍是缺口（NOT_SUPPORTED），需新增壁密度旋钮后再套同一工装。

## 复现

见 `tests/benchmarks/wetting_contact_angle/README.md` §2（stage A / B1 / B2 / 亚稳探针 /
assemble / make_report / pytest 一条链），单机数分钟量级，案例间彼此独立可并行。