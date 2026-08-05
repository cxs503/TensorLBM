# 壁面—加密组合门（fail-closed）

`tensorlbm.wall_refinement_combination_gate` 是能力声明门，不是数值求解器或
自动配置器。它只根据一个显式的 `WallRefinementCombination` 返回 `ALLOWED` 或
`WITHHELD`；不能从已有函数名称推断组合能力。

## 当前允许矩阵

仅下列最小基线允许：

* `D3Q19` 或 `D3Q27` 的、advanced collision contract 中可用的 `MRT`；
* 单相、平面静态几何、`SINGLE_LEVEL`；
* `wall_treatment` 为 `none` 或 `standard_static`；
* `refinement` 为 `none`。

`CM` 与 `KBC` 的可用性直接服从 `advanced_collision_contract`；该 contract
已 WITHHOLD 的 collision 不能因其余字段是基线而被放行。

## 明确保留（WITHHELD）

* 任意 wall function + static local refinement、surface shell 或 dynamic AMR；
* 任意 D3Q27 wall function；
* 加密 + multiphase、IBM 或 curved static wall；
* 未在上述基线中列出的任何组合。

即使调用者填写了 evidence，这些组合仍是 `WITHHELD`。填写 evidence 只会使
缺失项报告更完整，绝不构成验证证明或放行依据。

## 未来要新增组合行时的证据

跨级壁面处理至少需要记录：

* `wall_distance_dy`；
* `y_plus`；
* `level_link_owner`（哪一级/接口拥有 level link）；
* `wall_geometry_owner`；
* `interface_transfer_proof`。

新增允许组合必须同时新增显式矩阵行与接受测试；不得修改 gate 以通过“存在
wall_model/refinement 函数”这一事实推导支持。