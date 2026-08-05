# 通用能力矩阵 R1

`tensorlbm.general_capability_matrix` 是给未来前端或通用计算平台消费的**纯 Python**能力判定契约。它不启动 FastAPI、不导入求解器、不改变任何 solver hot path，也不把同名的 legacy 函数误作已经组合验证的产品能力。

## API

```python
from tensorlbm.general_capability_matrix import (
    CapabilityRequest,
    assess_capability,
    capability_matrix,
)

result = assess_capability(CapabilityRequest(lattice="D3Q19", collision="MRT"))
assert result.status.value == "supported"
payload = result.to_dict()  # JSON-ready dict
matrix = capability_matrix()  # audited component evidence only
```

候选字段为：`lattice`、`collision`、`turbulence`、`multiphase`、`boundary`、`geometry`、`wall_treatment`、`refinement`、`backend`、`outputs`。也可以传入具有这些字段的 mapping。未知字段会报 `ValueError`，以防适配器静默丢失请求语义。

返回 `CapabilityAssessment`：

- `status`: `supported`、`withheld` 或 `not_supported`；
- `reasons`: 稳定的机器可读 `code`、`field`、`message`；
- `evidence_tier`: 证据强度；
- `capability_hash`: R1 声明/证据注册表的 SHA-256；
- `config_hash`: 规范化候选配置的 SHA-256；
- `normalized_request`: 实际参与判断的规范化配置。

## R1 的刻意范围（fail closed）

唯一返回 `supported` 的完整组合是：**D3Q19 / MRT / single-phase / static-wall / static solid mask / bounce-back / no AMR / Torch / `rho, velocity` 输出**。这是可执行的共同契约，不是精度声明。

`advanced_collision_contract` 已证明 D3Q19 与 D3Q27 的 MRT 内核可执行；但是 R1 只将前述 D3Q19 完整组合标为 `supported`。D3Q27/MRT 仅有组件级证据，缺少该完整平台组合证据，故返回 `withheld`。

CM/cascaded 和 KBC/entropic KBC 均遵循 `advanced_collision_contract` 的显式 withholding：不能被 legacy 函数名提升为可用能力。壁面函数、AMR、湍流、多相、IBM/动态几何、不同边界或输出组合同样不会因为仓库中存在模块而被自动拼接；对**输入 schema 中已知但尚未完整验证**的值，R1 返回 `withheld` 和 `WITHHELD_UNVERIFIED_COMPOSITION` 原因。所有字段（包括 `outputs` 列表的每个元素）中**未知**的值都在组合证据评估之前返回 `not_supported`，并给出 `UNKNOWN_VALUE` 及对应 `field`。

后续壁面函数、AMR、组合验证及精度推荐应先新增完整组合的可复现实证和测试，再收窄/升级此注册表；不能仅通过模块存在性修改状态。
