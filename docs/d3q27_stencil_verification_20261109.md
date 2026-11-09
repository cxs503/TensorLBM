# D3Q27 Stencil Contract Verification

**Date**: 2026-11-09  
**Mission**: m7554 (tp-qa-2 self-start verifier)  
**Category**: 验证欠账 — D3Q27 张量模板契约测试补全

## 背景

TensorLBM 的 `tests/core/` 目录仅有 3 个测试文件：
- `test_collision_contracts.py` — 碰撞核契约
- `test_d3q19_stencil.py` — D3Q19 张量模板契约（6 方向面 + 12 方向边 = 18 移动方向）
- `test_lattice_contracts.py` — 格子契约

**缺口**：D3Q27（26 移动方向：6 面 + 12 边 + 8 角）无对应模板契约测试。D3Q27 是高精度碰撞（cumulant、KBC 等）的基础格子，其张量坐标转换（C 矩阵 (x,y,z) 序 ↔ 场 (z,y,x) 序）的正确性对求解器至关重要。

## 方法

仿照 `test_d3q19_stencil.py` 的独立预言机模式，编写 `test_d3q27_stencil.py`：

1. **独立预言机**：直接从 `tensorlbm.d3q27.C` 读取速度描述符，不依赖被测模块
2. **坐标转换验证**：C 矩阵为 (x,y,z) 序，场索引为 (z,y,x) 序，预言机做正确转换
3. **66 项测试**：
   - 移动方向完整性（q=1..26）
   - 张量位移与预言机逐方向比对（26 参数化）
   - 非法 q 值拒绝（7 参数化）
   - Pull roll 与预言机逐方向比对（26 参数化）
   - Push roll 与预言机逐方向比对（26 参数化）
   - 邻居掩码覆盖 + 相位守卫
   - 维度约束（2D/4D 场拒绝）
   - 非张量输入拒绝
   - D3Q27 方向分类验证（6 面 + 12 边 + 8 角）
   - 反向方向对验证（每方向必有 c_opp = -c_q）
   - Pull-Push 往返恒等验证

## 结果

```
66 passed, 1 warning in 5.92s
```

全部 66 项测试通过，证明：
- D3Q27 的 26 个移动方向的张量位移全部正确
- Pull/Push roll 操作与独立预言机逐位一致
- 邻居掩码生成覆盖全部 26 方向
- 相位守卫正确检测直接相链
- 输入验证（维度、类型、q 值范围）全部生效
- 方向分类（面/边/角）与理论一致
- 反向方向对完整性验证通过
- Pull-Push 往返恒等性验证通过

## 文件变更

- **新增**: `tests/core/test_d3q27_stencil.py` (5906 bytes, 66 tests)

## 局限

- 仅验证张量模板层（坐标转换/roll/掩码），不验证碰撞核本身
- D3Q27 碰撞核（cumulant、KBC）的数值精度需另案验证
- 未覆盖 GPU 设备路径（测试在 CPU 上运行，但 roll 操作是设备无关的）
