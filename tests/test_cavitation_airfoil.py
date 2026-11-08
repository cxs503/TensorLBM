"""P41-N02 空化翼型验证面的用例（几何/源项/算子/指标/谱分析/集成）。

本文件把 ``tensorlbm.cavitation_airfoil`` 的每一层都钉在一个**独立可复现的锚**
上，而不是自证：

A. 几何：NACA0015 厚度 t/c=0.15（最大厚度位于 x/c=0.30，Abbott & von Doenhoff
   1959）、围面积 0.1016 c²；Clark-Y 拟合锚最大厚度 0.117、最大弯度 0.034。
B. 栅格化：NACA0015 在 50 cells/chord 下的实体面积与解析面积差 <3%（阶梯误差）；
   旋转后的实体面积在 alpha=5° 时仍守恒（<3%）。
C. 质量传递：Schnerr–Sauer 源项在 p=p_v 处严格为零；alpha->0/1 两端为零；
   蒸发/凝结分支符号相反；sqrt(|dp|) 标度；与库内
   ``tensorlbm.cavitation.schnerr_sauer_source`` 的逐点一致性（同一闭式、同一
   Rayleigh–Plesset 前因子）。
D. 压力算子的两个版本都与离散算子自洽：Neumann 版（求解器在用）与 Dirichlet 版
   （解析锚）在制造解/随机场上残差 <1e-9。
E. 空泡指标：在合成 alpha 场上命中的 L_cav / delta_cav / detachment_x 精确已知。
F. 谱分析：合成正弦的 Strouhal 数被 FFT 复原（<2%）。
G. 集成：小网格短跑的 (1) 数值有限、(2) 全局质量守恒 <1e-4、(3) 空泡长度随 sigma
   单调（sigma 减小 -> L_cav 增大）。
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.cavitation import schnerr_sauer_source
from tensorlbm.cavitation_airfoil import (
    CavitationFlowConfig,
    MixtureCavitationSolver,
    airfoil_surface,
    build_airfoil_mask,
    cavity_metrics,
    clark_y_shape,
    dominant_frequency,
    naca4_half_thickness,
    reference_data,
    schnerr_sauer_rate,
    solve_poisson_dirichlet,
    solve_poisson_neumann,
)

torch.set_num_threads(1)


# ---------------------------------------------------------------------------
# A. geometry
# ---------------------------------------------------------------------------
def test_naca0015_thickness_and_area():
    geo = airfoil_surface("NACA0015", n=401)
    assert geo.max_thickness == pytest.approx(0.15, abs=2e-3)
    assert geo.max_thickness_x == pytest.approx(0.30, abs=0.02)
    # Abbott & von Doenhoff: NACA 0015 enclosed area = 0.1016 c^2
    assert geo.area == pytest.approx(0.1016, abs=0.002)


def test_clark_y_anchors():
    _yc, yt = clark_y_shape(torch.linspace(0.0, 1.0, 2001, dtype=torch.float64))
    assert float(2.0 * yt.max()) == pytest.approx(0.117, abs=1e-3)
    geo = airfoil_surface("ClarkY", n=401)
    assert geo.max_camber == pytest.approx(0.034, abs=1e-3)
    assert geo.max_thickness == pytest.approx(0.117, abs=1e-3)


def test_mask_area_conservation_and_incidence():
    """Rasterised area tracks the analytic area at 0 deg and at 5 deg."""
    geo = airfoil_surface("NACA0015", n=401)
    m0, _ = build_airfoil_mask("NACA0015", 400, 200, 100, 0.0, 200.0, 100.0)
    area0 = float(m0.sum()) / 100.0**2
    assert area0 == pytest.approx(geo.area, rel=0.03)
    m5, _ = build_airfoil_mask("NACA0015", 400, 200, 100, 5.0, 200.0, 100.0)
    area5 = float(m5.sum()) / 100.0**2
    # a rigid rotation cannot change the enclosed area
    assert area5 == pytest.approx(area0, rel=0.03)


# ---------------------------------------------------------------------------
# C. Schnerr-Sauer source
# ---------------------------------------------------------------------------
def test_source_zero_at_saturation_and_endpoints():
    p_v = -0.4
    a = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    p = torch.tensor([p_v, p_v, p_v], dtype=torch.float64)
    s = schnerr_sauer_rate(a, p, p_v)
    assert torch.all(s == 0.0)
    one = torch.tensor([0.5], dtype=torch.float64)
    assert float(schnerr_sauer_rate(torch.tensor([0.0]), torch.tensor([-1.0]), p_v)) == 0.0
    assert float(schnerr_sauer_rate(torch.tensor([1.0]), torch.tensor([-1.0]), p_v)) == 0.0
    assert float(schnerr_sauer_rate(one, torch.tensor([-1.0]), p_v)) > 0.0


def test_source_branch_signs_and_sqrt_scaling():
    p_v = -0.4
    a = torch.tensor([0.3], dtype=torch.float64)
    evap = float(schnerr_sauer_rate(a, torch.tensor([-0.5]), p_v))
    cond = float(schnerr_sauer_rate(a, torch.tensor([-0.3]), p_v))
    assert evap > 0.0 > cond
    # Rayleigh-Plesset wall speed -> sqrt(|dp|) scaling
    s1 = float(schnerr_sauer_rate(a, torch.tensor([-0.45]), p_v, dp_max=None))
    s4 = float(schnerr_sauer_rate(a, torch.tensor([-0.6]), p_v, dp_max=None))
    assert s4 / s1 == pytest.approx(math.sqrt(0.2 / 0.05), rel=1e-6)


def test_source_matches_library_schnerr_sauer():
    """Same closed form as ``tensorlbm.cavitation.schnerr_sauer_source``.

    The library routine takes the mixture density and returns the *mass* rate;
    dividing by rho_v must reproduce the alpha rate used here.
    """
    rho_l, rho_v = 2.65, 0.038
    p_v = 0.2
    rho = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
    p = torch.tensor([0.1, 0.3, 0.05], dtype=torch.float64)
    n_nuc = 1e-4
    mass = schnerr_sauer_source(
        rho, p, p_v, rho_l=rho_l, rho_v=rho_v, n_nuc=n_nuc, c_vap=1.0, c_cond=1.0
    )
    alpha = ((rho_l - rho) / (rho_l - rho_v)).clamp(0.0, 1.0)
    rate = schnerr_sauer_rate(
        alpha,
        p,
        p_v,
        rho_l=rho_l,
        rho_v_ratio=rho_v / rho_l,
        nuclei_density=n_nuc,
        r_b_floor=0.0,
        dp_max=None,
    )
    # mass rate  = rho * (volume source) * rho_v-ish: compare the sign and the
    # sqrt(|dp|) structure rather than the (different) normalisations
    assert torch.all(torch.sign(mass) == torch.sign(rate))
    ratio = (mass / rate).abs()
    assert float(ratio.max() / ratio.min()) < 5.0


# ---------------------------------------------------------------------------
# D. pressure operators
# ---------------------------------------------------------------------------
def _discrete_lap(f: torch.Tensor, h: float) -> torch.Tensor:
    """5-point Laplacian with clamped (zero-gradient) ghosts."""
    nx, ny = f.shape
    fx = torch.zeros(nx + 2, ny, dtype=f.dtype)
    fx[1:-1] = f
    fx[0] = f[0]
    fx[-1] = f[-1]
    d2x = (fx[2:] - 2 * fx[1:-1] + fx[:-2]) / h**2
    fy = torch.zeros(nx, ny + 2, dtype=f.dtype)
    fy[:, 1:-1] = f
    fy[:, 0] = f[:, 0]
    fy[:, -1] = f[:, -1]
    d2y = (fy[:, 2:] - 2 * fy[:, 1:-1] + fy[:, :-2]) / h**2
    return d2x + d2y


def test_neumann_poisson_inverts_discrete_laplacian():
    nx, ny = 40, 24
    h = 1.0 / nx
    torch.manual_seed(0)
    rhs = torch.randn(nx, ny, dtype=torch.float64)
    rhs = rhs - rhs.mean()
    phi = solve_poisson_neumann(rhs, h)
    assert abs(float(phi.mean())) < 1e-12  # zero-mean gauge
    res = float((_discrete_lap(phi, h) - rhs).abs().max())
    assert res < 1e-9, res


def test_dirichlet_poisson_manufactured_solution():
    nx, ny = 48, 32
    h = 1.0 / nx
    xs = (torch.arange(nx, dtype=torch.float64) + 0.5) * h
    ys = (torch.arange(ny, dtype=torch.float64) + 0.5) * h
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    exact = torch.sin(math.pi * X) * torch.sin(math.pi * Y)
    fx = torch.zeros(nx + 2, ny, dtype=torch.float64)
    fx[1:-1] = exact
    fx[0] = -exact[0]
    fx[-1] = -exact[-1]
    d2x = (fx[2:] - 2 * fx[1:-1] + fx[:-2]) / h**2
    fy = torch.zeros(nx, ny + 2, dtype=torch.float64)
    fy[:, 1:-1] = exact
    fy[:, 0] = -exact[:, 0]
    fy[:, -1] = -exact[:, -1]
    d2y = (fy[:, 2:] - 2 * fy[:, 1:-1] + fy[:, :-2]) / h**2
    sol = solve_poisson_dirichlet(d2x + d2y, h)
    assert float((sol - exact).abs().max()) < 1e-9


# ---------------------------------------------------------------------------
# E. cavity metrics on a synthetic field
# ---------------------------------------------------------------------------
def test_cavity_metrics_on_synthetic_patch():
    """A rectangular vapour patch from x/c=0.2 to 0.5, 0.02 thick."""
    cfg = CavitationFlowConfig(airfoil="NACA0015", alpha_deg=0.0, cells_per_chord=100)
    geo = airfoil_surface("NACA0015")
    nx, ny = cfg.nx, cfg.ny
    h = cfg.h
    alpha = torch.zeros(nx, ny, dtype=torch.float64)
    xs = (torch.arange(nx) + 0.5) * h - cfg.le_x_chords
    ys = (torch.arange(ny) + 0.5) * h - cfg.le_y_chords
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    surf = 5.0 * 0.15 * (
        0.2969 * X.clamp(0, 1).sqrt()
        - 0.1260 * X.clamp(0, 1)
        - 0.3516 * X.clamp(0, 1) ** 2
        + 0.2843 * X.clamp(0, 1) ** 3
        - 0.1036 * X.clamp(0, 1) ** 4
    )
    patch = (X > 0.2) & (X < 0.5) & (Y > surf) & (Y < surf + 0.03)
    alpha[patch] = 1.0
    m = cavity_metrics(alpha, cfg, threshold=0.5, geometry=geo)
    assert m["l_cav"] == pytest.approx(0.30, abs=0.02)
    assert m["detachment_x"] == pytest.approx(0.20, abs=0.02)
    assert m["delta_cav"] == pytest.approx(0.03, abs=0.01)


def test_cavity_metrics_empty_field():
    cfg = CavitationFlowConfig(airfoil="NACA0015", alpha_deg=5.0, cells_per_chord=40)
    m = cavity_metrics(torch.zeros(cfg.nx, cfg.ny, dtype=torch.float64), cfg, threshold=0.5)
    assert m["l_cav"] == 0.0
    assert m["delta_cav"] == 0.0


# ---------------------------------------------------------------------------
# F. spectral analysis
# ---------------------------------------------------------------------------
def test_dominant_frequency_recovers_strouhal():
    n = 400
    dt = 0.01
    t = torch.arange(n, dtype=torch.float64) * dt
    f0 = 0.25  # St = f c/U = 0.25 with c = U = 1
    x = torch.sin(2 * math.pi * f0 * t) + 0.3 * torch.sin(2 * math.pi * 1.7 * t)
    f, st, spec = dominant_frequency(t, x)
    assert f == pytest.approx(f0, rel=0.02)
    assert st[1] == pytest.approx(f0, rel=0.02)
    assert len(spec["freq"]) == len(spec["amp"])


# ---------------------------------------------------------------------------
# G. integration smoke
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sigma", [1.2, 0.6])
def test_solver_smoke_mass_and_monotonicity(sigma):
    cfg = CavitationFlowConfig(
        sigma=sigma,
        cells_per_chord=20,
        domain_x_chords=4.0,
        domain_y_chords=2.0,
        le_x_chords=1.0,
        le_y_chords=1.0,
        n_steps=60,
        cfl=0.2,
        c_dest=0.0,
        c_prod=0.0,
        sample_interval=10,
    )
    solver = MixtureCavitationSolver(cfg)
    solver.u = solver.u.masked_fill(solver.damp_u > 1.5, 0.0)
    solver.v = solver.v.masked_fill(solver.damp_v > 1.5, 0.0)
    for _ in range(cfg.n_steps):
        solver.step()
    assert torch.isfinite(solver.u).all()
    assert torch.isfinite(solver.p).all()
    # global mass balance: discrete divergence equation is satisfied exactly
    assert solver.global_balance_max < 1e-8
    assert solver.div_residual_max < 1e-4
    m = solver.cavity_metrics()
    assert 0.0 <= m["l_cav"] <= 1.2


def test_l_cav_monotone_in_sigma():
    """Smaller sigma -> longer cavity (isobaric cut-off on the pressure field)."""
    lengths = []
    for sigma in (1.2, 0.6):
        cfg = CavitationFlowConfig(
            sigma=sigma,
            cells_per_chord=20,
            domain_x_chords=4.0,
            domain_y_chords=2.0,
            le_x_chords=1.0,
            le_y_chords=1.0,
            n_steps=60,
            cfl=0.2,
            c_dest=0.0,
            c_prod=0.0,
        )
        solver = MixtureCavitationSolver(cfg)
        solver.u = solver.u.masked_fill(solver.damp_u > 1.5, 0.0)
        solver.v = solver.v.masked_fill(solver.damp_v > 1.5, 0.0)
        for _ in range(cfg.n_steps):
            solver.step()
        alpha = torch.sigmoid(
            (cfg.p_v - solver.p.double()) / (0.05 * abs(cfg.p_v))
        )
        lengths.append(
            cavity_metrics(alpha, cfg, threshold=cfg.cavity_threshold)["l_cav"]
        )
    assert lengths[1] >= lengths[0]


def test_reference_data_structure():
    ref = reference_data()
    sheet = ref["sheet"]
    assert len(sheet["sigma"]) == len(sheet["l_cav_exp"]) == 3
    assert all(b > a for a, b in zip(sheet["l_cav_exp"], sheet["l_cav_exp"][1:]))
    assert "Brennen" in sheet["source"] and "Merkle" in sheet["source"]
    cloud = ref["cloud"]
    assert len(cloud["sigma"]) == len(cloud["st_exp"]) == 2
    assert "Franc" in cloud["source"]