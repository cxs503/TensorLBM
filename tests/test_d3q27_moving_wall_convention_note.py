"""d3q27.moving_wall_linkwise_me_force_torque 约定标注的契约测试。

该原语的力矩是理想（镜面）反射模型口径：仅当固体格不参与碰撞
（反弹即碰撞）时精确；对 collide 全格 + post-stream 反弹的方案，
回场种群实为 BGK 弛豫过的镜像，力矩被系统性高估约 2 倍
（verified taylor_couette 披露 5：内壁 T/M_ref = 2.06/2.14/2.14
@ τ=0.65，τ=0.74 时 1.69）。约定无关口径 = 界面通量记账。
本测试钉住 docstring 标注不被无意删除；数值行为由
test_d3q27_moving_wall_momentum_exchange.py 回归覆盖。
"""

from __future__ import annotations

from tensorlbm.d3q27 import moving_wall_linkwise_me_force_torque

_DOC = moving_wall_linkwise_me_force_torque.__doc__ or ""


def test_docstring_documents_ideal_reflection_convention():
    assert "ideal" in _DOC
    assert "bounce-back replaces collision" in _DOC


def test_docstring_documents_two_x_overestimate_with_numbers():
    assert "2x" in _DOC
    assert "2.06/2.14/2.14" in _DOC
    assert "1.69" in _DOC  # τ=0.74 辅助工况（倍数的 τ 依赖）


def test_docstring_points_to_interface_flux_bookkeeping():
    assert "interface flux" in _DOC
    assert "0.73%" in _DOC
    assert "taylor_couette" in _DOC
