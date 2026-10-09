"""D2Q5 权重 fp32 精确和判别测试（thermal / passive_scalar）。

裸 fp32 构造的 D2Q5 权重 [1/3, 1/6×4] 五元精确和为 1 + 2⁻²⁵
（= 1 + 2.98e-8；fp64 归一化的相对修正 ~1e-16 远小于 fp32 舍入间隔，
量化回 fp32 后残差同号保留），长跑中表现为每步 ~3.7e-8 的标量重标
漂移（W3-A 记录）。修复 = fp64 构造 → 归一化 → fp32 量化 → 残差
吸收进静止权重（thermal.W5 与 passive_scalar._W5 两处独立定义；
thermal.W_D2Q5 是 W5 的别名，随修复受益）。

判别条款：
1. 每个被修权重 .double().sum() 恰为 1.0（修复前 = 1 + 2.98e-8）；
2. 逐元素与裸 fp32 构造差 ≤ 1 ulp（舍入级修复）；
3. 四个方向权重逐位不变（D2Q5 交叉方向对称性保持）；
4. 反向对照：裸构造的精确和确实 ≠ 1（证明上两条并非恒真）。
"""

from __future__ import annotations

import torch

from tensorlbm.passive_scalar import _W5 as _PS_W5
from tensorlbm.thermal import W5, W_D2Q5

# 修复前的裸 fp32 构造（判别基准）
_BARE_D2Q5 = torch.tensor(
    [1.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0], dtype=torch.float32
)
# 裸构造的精确和（Fraction 级验证过：恰为 1 + 2**-25）
_BARE_SUM = 1.0 + 2.9802322387695312e-08

_ALL_WEIGHTS = {
    "thermal.W5": W5,
    "thermal.W_D2Q5 (alias)": W_D2Q5,
    "passive_scalar._W5": _PS_W5,
}


def test_d2q5_weights_exact_unit_sum_in_double():
    for name, w in _ALL_WEIGHTS.items():
        assert w.dtype == torch.float32, name
        exact_sum = w.double().sum().item()
        assert exact_sum == 1.0, (name, exact_sum)
        assert abs(exact_sum - 1.0) < 1e-16, name


def test_d2q5_weights_change_is_rounding_level():
    for name, w in _ALL_WEIGHTS.items():
        ulp_delta = (w.view(torch.int32) - _BARE_D2Q5.view(torch.int32)).abs()
        assert int(ulp_delta.max()) <= 1, (name, ulp_delta.tolist())
        # 方向权重逐位不变（残差只吸收进静止权重）
        assert torch.equal(w[1:], _BARE_D2Q5[1:]), name


def test_d3q5_alias_identity():
    assert W_D2Q5 is W5


def test_bare_construction_exhibits_the_debt():
    """反向对照：裸构造确实带 +2**-25 残差，条款 1/2 有判别力。"""
    assert _BARE_D2Q5.double().sum().item() == _BARE_SUM
    assert _BARE_D2Q5.double().sum().item() != 1.0
