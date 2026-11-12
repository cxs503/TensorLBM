# Zero-Gradient (Neumann) Outlet BC — Kovasznay Re-verification (m16112)

## ① 案例介绍
Kovasznay (1948) 二维定常不可压 N-S 精确解（格栅后层流尾迹，y 向周期涡列沿流向指数衰减）。
u(x',y')=U0(1−e^{λx'}cos2πy'), v=U0(λ/2π)e^{λx'}sin2πy', λ=Re/2−√(Re²/4+4π²)（负根）。
Re=40, U0=0.03, 域 x'∈[0,3]（3 个 y 周期），y 周期，入口 Zou-He 解析 Dirichlet，出口零梯度 Neumann。

## ② 怎么做的
本单补全「生态摩擦」缺口（Kovasznay 报告 15179/15194 登记）：零梯度出口 f[:,:,-1]=f[:,:,-2]
原为 run.py 内 3 行内联、库无函数。新增库函数 `tensorlbm.boundaries.zero_gradient_outlet_2d`，
并在 capability contract 登记为 PHYSICS_VALIDATED（D2Q9）；run.py 改用库函数；
新增测试 `tests/test_zero_gradient_outlet_kovasznay.py` 用库函数重跑 Kovasznay 稳态验证。

## ③ 计算结果（程序化注入）
pytest 5 条：纯拷贝机制（出口=上游列、内部不动、输入不原地改）+ 方向数拒绝 +
contract 登记（D2Q9=PHYSICS_VALIDATED，D3Q19/27=NO_IMPLEMENTATION 诚实）+ 
Kovasznay 长域 ≤3% 门（ny=64/128）+ 64→128 单调收敛。

## ④ 结果图
纯库函数/文档任务，无场量云图（理由：本单为 BC 库函数补全+验证欠账复核，
数值表已程序化注入；场量云图见原 Kovasznay 报告 15179）。

## ⑤ 注意点
- 零梯度出口在短域（x'∈[0,1]）会把出口 v 杀到 ≈0（全场 v 最大相对误差 ~96%），
  需长下游域（≥3 周期）才 ≤3%。
- 二阶外推出口（2·f[n-2]−f[n-3]）在 Kovasznay 算例发散（负分布函数→NaN），不可用。
- 本机 load15≈100，Kovasznay 重跑用 2000 步（基准已证 ~2000 步收敛），未起新重算力进程。

## ⑥ 误差判定
Kovasznay 长域（ny=64/128）u/v L2 与掩模最大相对误差均 ≤3%（<3% 门），
64→128 各指标单调下降（收敛）。无独立基准须声明：基准=Kovasznay 1948 解析解本身。
