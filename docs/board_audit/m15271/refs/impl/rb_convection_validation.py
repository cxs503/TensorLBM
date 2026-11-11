"""Rayleigh–Bénard / 差分加热腔的**定量**验证面（战役 P96W3-N04）。

背景（为什么需要本模块）
------------------------
``thermal.py`` / ``thermal3d.py`` 的热对流此前只有"能跑"级检查。P96W2-N04
建立过一版验证面，但审计发现三处口径问题（详见
``docs/reports/P96W3-N04_thermal_convection_validation.md`` §2）：

1. C1 的 Ra_c 是在**连续谱特征值解算器**（Orr–Sommerfeld）上验证的，与
   1707.76 对照等于"解析解 vs 文献常数"，并未度量 LBM 自身起稳阈值；
2. C2 的 2D 周期 RB 参考 Nu 是手写区间"2.6–2.8"（无文献锚），判据还从
   10% 放宽到 20%；
3. 定常判定缺失（5000 步远未定常），且 ``simulate_rayleigh_benard_2d``
   以 ``H = ny`` 定 Ra 而 ``nusselt_rb`` 以 ``H = ny−1`` 归一 ⇒ Ra 有效值
   偏离 ((ny−1)/ny)³（ny=32 时 −9%）。

本模块因此**从零**建立口径自洽的验证面：几何、边界、Nu 估计、无量纲换算
集中在同一处定义，所有可观测量均由时间推进的 LBM 直接给出。

离散口径（唯一真源）
--------------------
* 网格 ``(ny, nx)`` 节点，y 节点 ``j=0..ny−1``，半程壁面 ``y=−1/2`` 与
  ``y=ny−1/2`` ⇒ **H = ny**（格子单位）。
* 速度：D2Q9 BGK + Guo 力格式（``thermal.collide_bgk_force``），
  pre-streaming 半程反弹（``thermal.pre_streaming_bounce_back``）给静止
  no-slip 壁（壁面亦在半程）。
* 温度：D2Q5 BGK（α=(τ_T−1/2)/3）；壁面 Dirichlet 用标准
  **anti-bounce-back**（壁面精确落在半程，二阶精度）``g_i = −g_ī^* + 2 w_i T_w``；
  绝热壁用零通量反弹 ``g_i = g_ī^*``。
* Nu：壁面三点二次插值求导（纯导热线性剖面时**逐位**给 1）。
* 无量纲：ν=(τ_f−1/2)/3，α=(τ_T−1/2)/3，Pr=ν/α，
  Ra = g·β·ΔT·H³/(ν·α) ⇒ g_beta = Ra·ν·α/(H³·ΔT)。
* 压缩性硬约束：``Ma = u_max/c_s``，``Ma ≈ α·√(Ra·Pr)/(H·c_s)``，
  c_s=1/√3 ⇒ 高 Ra 必须小 α / 大 H。

文献锚（唯一真源）
------------------
* Ra_c=1707.762、a_c=3.11632：Chandrasekhar (1961) *Hydrodynamic and
  Hydromagnetic Stability*；Drazin & Reid (1981) *Hydrodynamic Stability*。
* 差分加热方腔平均 Nu：de Vahl Davis (1983) *Int. J. Numer. Methods
  Fluids* 3:249–264（Pr=0.71，刚-刚方腔）。
* 2D 周期 RB 定常辊：Clever & Busse (1974) *J. Fluid Mech.* 65:625–645
  （Pr=0.71；本模块以文献带 ``RB_ROLL_BAND`` 形式使用）。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch

from .d2q9 import equilibrium, macroscopic
from .solver import stream
from .thermal import (
    buoyancy_force,
    collide_bgk_force,
    collide_thermal_bgk,
    pre_streaming_bounce_back,
    temperature_stream,
)

__all__ = [
    "RA_CRIT",
    "A_CRIT",
    "DE_VAHL_DAVIS_NU",
    "RB_ROLL_BAND",
    "Convection2D",
    "make_convection",
    "wall_flux_nusselt",
    "gci",
    "richardson",
    "measure_growth_rate",
    "critical_rayleigh_measured",
    "critical_wavenumber_measured",
    "run_to_steady",
    "dimensionless_selfconsistency",
    "no_buoyancy_control",
    "adiabatic_enthalpy_drift",
    "diffusion_decay_alpha",
]

# ── 文献锚 ──────────────────────────────────────────────────────────────────
RA_CRIT = 1707.762       # 刚-刚 RB 临界 Rayleigh 数（Chandrasekhar 1961）
A_CRIT = 3.11632         # 临界波数 a_c = 2π/λ_c（Drazin & Reid 1981）

#: de Vahl Davis (1983)：差分加热方腔 Pr=0.71 平均 Nu
DE_VAHL_DAVIS_NU: dict[float, float] = {
    1e3: 1.118,
    1e4: 2.243,
    1e5: 4.519,
    1e6: 8.800,
}

#: Clever & Busse (1974) 2D 定常辊（Pr=0.71）Nu 文献带 (lo, hi)
RB_ROLL_BAND: dict[float, tuple[float, float]] = {
    1e4: (2.10, 2.30),
    1e5: (4.20, 4.80),
    1e6: (7.60, 8.80),
}

CS = math.sqrt(1.0 / 3.0)
_W5 = (1.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0)


# ── 工具 ────────────────────────────────────────────────────────────────────


def wall_flux_nusselt(
    T: torch.Tensor,
    *,
    axis: int,
    side: str,
    t_wall: float,
    dT: float,
    H: float,
) -> torch.Tensor:
    """壁面逐位置 Nu（二阶；壁面在半程，节点间距 1）。

    * ``side="lo"``（壁面在 −1/2）  dT/dn|wall = (9·T[0] − T[1] − 8·T_w)/3
    * ``side="hi"``（壁面在 N−1/2） dT/dn|wall = (8·T_w − 9·T[-1] + T[-2])/3

    返回沿壁面的一维张量（调用方平均）。
    """
    if axis == 0:
        t0, t1, tm1, tm2 = T[0, :], T[1, :], T[-1, :], T[-2, :]
    elif axis == 1:
        t0, t1, tm1, tm2 = T[:, 0], T[:, 1], T[:, -1], T[:, -2]
    else:  # pragma: no cover
        raise ValueError("axis must be 0 or 1")
    if side == "lo":
        grad = (9.0 * t0 - t1 - 8.0 * t_wall) / 3.0
    elif side == "hi":
        grad = (8.0 * t_wall - 9.0 * tm1 + tm2) / 3.0
    else:  # pragma: no cover
        raise ValueError("side must be 'lo' or 'hi'")
    return -grad * H / dT


def gci(fine: float, medium: float, coarse: float, r: float = 2.0) -> dict[str, float]:
    """Roache 三网格 GCI：观测阶 p、Richardson 外推值、细网格 GCI(%)。"""
    eps21 = abs(medium - fine)
    eps32 = abs(coarse - medium)
    out: dict[str, float] = {
        "fine": fine, "medium": medium, "coarse": coarse, "r": r,
        "eps21": eps21, "eps32": eps32,
        "p": float("nan"), "extrapolated": float("nan"), "gci_fine_pct": float("nan"),
    }
    if eps21 == 0.0 and eps32 == 0.0:
        out.update({"extrapolated": fine, "gci_fine_pct": 0.0})
        return out
    if eps21 == 0.0 or eps32 == 0.0:
        return out
    ratio = eps32 / eps21
    if ratio > 0:
        out["p"] = math.log(ratio) / math.log(r) if r != 1 else float("nan")
    p = out["p"]
    if math.isfinite(p) and abs(r**p - 1.0) > 1e-12:
        ext = fine + (fine - medium) / (r**p - 1.0)
        out["extrapolated"] = ext
        if ext != 0.0:
            out["gci_fine_pct"] = 100.0 * 1.25 * abs(fine - ext) / abs(ext)
    return out


def richardson(fine: float, medium: float, coarse: float, r: float = 2.0) -> float:
    return gci(fine, medium, coarse, r)["extrapolated"]


# ── 主求解器 ────────────────────────────────────────────────────────────────


@dataclass
class Convection2D:
    """2D 浮力驱动热对流双分布 LBM（float64；几何/边界口径唯一真源）。

    ``case="rb"``：x 周期 + 上下刚性等温板；``case="dhc"``：左右等温竖壁 +
    上下绝热方腔。输运系数给 ``alpha``（热扩散率）或 ``tau_f``（黏性松弛）。
    """

    ny: int
    nx: int
    ra: float
    pr: float = 0.71
    case: str = "rb"
    dT: float = 1.0
    t_ref: float = 0.0
    alpha: float | None = None
    tau_f: float | None = None
    enable_buoyancy: bool = True
    gap_flow: float | None = None
    wall_bc: str = "pre"
    dtype: torch.dtype = torch.float64
    device: str = "cpu"
    f: torch.Tensor | None = field(default=None, repr=False)
    g: torch.Tensor | None = field(default=None, repr=False)
    wall: torch.Tensor | None = field(default=None, repr=False)
    step_count: int = 0
    nu_lat: float = 0.0
    alpha_lat: float = 0.0
    tau_T: float = 0.0
    tau_f_eff: float = 0.0
    g_beta: float = 0.0
    H: float = 0.0
    t_hot: float = 0.0
    t_cold: float = 0.0

    def __post_init__(self) -> None:
        if self.case not in ("rb", "dhc"):
            raise ValueError("case must be 'rb' or 'dhc'")
        if self.case == "dhc" and self.nx != self.ny:
            raise ValueError("dhc requires nx == ny")
        if self.nx % 2 or self.ny % 2:
            raise ValueError("nx, ny must be even")
        self.H = float(self.ny)
        self.t_hot, self.t_cold = self.t_ref + self.dT, self.t_ref
        if self.alpha is not None:
            self.alpha_lat = float(self.alpha)
            self.nu_lat = self.pr * self.alpha_lat
            self.tau_f_eff = 3.0 * self.nu_lat + 0.5
            if self.tau_f is not None and abs(self.tau_f - self.tau_f_eff) > 1e-9:
                raise ValueError("alpha and tau_f are inconsistent")
        elif self.tau_f is not None:
            self.tau_f_eff = float(self.tau_f)
            self.nu_lat = (self.tau_f_eff - 0.5) / 3.0
            self.alpha_lat = self.nu_lat / self.pr
        else:
            raise ValueError("give alpha= or tau_f=")
        if not (0.5 < self.tau_f_eff < 3.0):
            raise ValueError(f"tau_f={self.tau_f_eff} out of range")
        self.tau_T = 3.0 * self.alpha_lat + 0.5
        if self.tau_T <= 0.5:
            raise ValueError("tau_T <= 0.5 (non-diffusive)")
        # Ra 的几何长度用**流动间隙** d_v（pre-streaming 半程反弹把 no-slip
        # 面放在壁面行内侧一格外，实测 d_v = ny−2；见 calibrate_gaps 与
        # rb_discrete_stability 的同一约定）。温度间隙 d_T 由 ABB 决定（实测
        # d_T = 0.999·ny），两者差一格/侧，属库内壁面配方的已知几何残差。
        self.gap_flow = float(self.gap_flow if self.gap_flow is not None else self.ny - 2)
        self.g_beta = (
            self.ra * self.nu_lat * self.alpha_lat / (self.gap_flow**3 * self.dT)
            if self.enable_buoyancy else 0.0
        )
        self._init_state(torch.device(self.device))

    # -- 初始化 ----------------------------------------------------------
    def _init_state(self, dev: torch.device) -> None:
        ny, nx = self.ny, self.nx
        ones = torch.ones((ny, nx), dtype=self.dtype, device=dev)
        zeros = torch.zeros((ny, nx), dtype=self.dtype, device=dev)
        self.f = equilibrium(ones, zeros, zeros)
        self.g = self._equilibrium_T(self.base_profile(), zeros, zeros)
        mask = torch.zeros((ny, nx), dtype=torch.bool, device=dev)
        mask[0, :] = mask[-1, :] = True
        if self.case == "dhc":
            mask[:, 0] = mask[:, -1] = True
        self.wall = mask

    def base_profile(self) -> torch.Tensor:
        """导热基态（两半程壁面之间的线性剖面），形状 (ny, nx)。"""
        assert self.f is not None
        dev = self.f.device
        ny, nx = self.ny, self.nx
        if self.case == "rb":
            y = (torch.arange(ny, dtype=self.dtype, device=dev) + 0.5) / self.H
            prof = self.t_hot + (self.t_cold - self.t_hot) * y
            return prof.unsqueeze(1).expand(ny, nx).contiguous()
        x = (torch.arange(nx, dtype=self.dtype, device=dev) + 0.5) / float(nx)
        prof = self.t_hot + (self.t_cold - self.t_hot) * x
        return prof.unsqueeze(0).expand(ny, nx).contiguous()

    def _w5(self) -> torch.Tensor:
        assert self.f is not None
        return torch.tensor(_W5, dtype=self.dtype, device=self.f.device)

    def _cx5(self) -> torch.Tensor:
        assert self.f is not None
        return torch.tensor([0.0, 1.0, -1.0, 0.0, 0.0], dtype=self.dtype, device=self.f.device)

    def _cy5(self) -> torch.Tensor:
        assert self.f is not None
        return torch.tensor([0.0, 0.0, 0.0, 1.0, -1.0], dtype=self.dtype, device=self.f.device)

    def _equilibrium_T(self, T, ux, uy):
        w = self._w5().view(5, 1, 1)
        cu = self._cx5().view(5, 1, 1) * ux + self._cy5().view(5, 1, 1) * uy
        return w * T.unsqueeze(0) * (1.0 + 3.0 * cu)

    def set_temperature(self, T: torch.Tensor) -> None:
        zeros = torch.zeros_like(T)
        self.g = self._equilibrium_T(T, zeros, zeros)

    # -- 温度边界 --------------------------------------------------------
    def _thermal_bc(self, g_post: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """post-streaming 温度边界（Dirichlet=ABB 半程；绝热=零通量反弹）。"""
        w = self._w5()
        g = g.clone()
        if self.case == "rb":
            g[3, 0, :] = -g_post[4, 0, :] + 2.0 * w[3] * self.t_hot
            g[4, -1, :] = -g_post[3, -1, :] + 2.0 * w[4] * self.t_cold
        else:
            g[3, 0, :] = g_post[4, 0, :]          # 下壁绝热
            g[4, -1, :] = g_post[3, -1, :]        # 上壁绝热
            g[1, :, 0] = -g_post[2, :, 0] + 2.0 * w[1] * self.t_hot
            g[2, :, -1] = -g_post[1, :, -1] + 2.0 * w[2] * self.t_cold
        return g

    # -- 推进 ------------------------------------------------------------
    def step(self, n: int = 1) -> None:
        assert self.f is not None and self.g is not None
        zeros = None
        for _ in range(n):
            rho, ux, uy = macroscopic(self.f)
            T = self.g.sum(0)
            if self.enable_buoyancy:
                F = buoyancy_force(rho, T, self.g_beta, t_ref=self.t_ref)
            else:
                if zeros is None:
                    zeros = torch.zeros((2, self.ny, self.nx), dtype=self.dtype,
                                        device=self.f.device)
                F = zeros
            # 壁面行按 no-slip 口径取 u=0：ABB 的 2w·T_w 项本就假定 u_wall=0，
            # 若把半程反弹在壁面行留下的 O(u) 伪速度喂进温度平衡，会逐步注热
            # （实测 dt=0.02 时 100 步内总焓漂 3.8%，Nu 反号）。
            ux_T = torch.where(self.wall, torch.zeros_like(ux), ux)
            uy_T = torch.where(self.wall, torch.zeros_like(uy), uy)
            g_post = collide_thermal_bgk(self.g, T, ux_T, uy_T, self.tau_T)
            self.g = self._thermal_bc(g_post, temperature_stream(g_post))
            f_pre = self.f
            f_post = collide_bgk_force(self.f, self.tau_f_eff, F)
            if self.wall_bc == "pre":
                f_post = pre_streaming_bounce_back(f_pre, f_post, self.wall)
                self.f = stream(f_post)
            else:  # "post"：标准半程 bounce-back（壁面落在 y=∓1/2 / x=∓1/2）
                f_new = stream(f_post).clone()
                f_new[3, 0, :] = f_post[4, 0, :]
                f_new[4, -1, :] = f_post[3, -1, :]
                if self.case == "dhc":
                    f_new[1, :, 0] = f_post[2, :, 0]
                    f_new[2, :, -1] = f_post[1, :, -1]
                self.f = f_new
            self.step_count += 1

    # -- 诊断 ------------------------------------------------------------
    def fields(self):
        assert self.f is not None and self.g is not None
        rho, ux, uy = macroscopic(self.f)
        return rho, ux, uy, self.g.sum(0)

    def u_max(self) -> float:
        _, ux, uy, _ = self.fields()
        return float(torch.sqrt(ux * ux + uy * uy).max().item())

    def mach(self) -> float:
        return self.u_max() / CS

    def nusselt(self, T: torch.Tensor | None = None) -> dict[str, float]:
        if T is None:
            T = self.fields()[3]
        if self.case == "rb":
            a = float(wall_flux_nusselt(T, axis=0, side="lo", t_wall=self.t_hot,
                                        dT=self.dT, H=self.H).mean().item())
            b = float(wall_flux_nusselt(T, axis=0, side="hi", t_wall=self.t_cold,
                                        dT=self.dT, H=self.H).mean().item())
        else:
            a = float(wall_flux_nusselt(T, axis=1, side="lo", t_wall=self.t_hot,
                                        dT=self.dT, H=self.H).mean().item())
            b = float(wall_flux_nusselt(T, axis=1, side="hi", t_wall=self.t_cold,
                                        dT=self.dT, H=self.H).mean().item())
        m = 0.5 * (a + b)
        return {"nu_hot": a, "nu_cold": b, "nu": m,
                "nu_imbalance": abs(a - b) / max(abs(m), 1e-30)}

    def nu_midplane(self, T: torch.Tensor | None = None) -> float:
        """中截面**导热**分量的 Nu（仅作参考：定常态下不是守恒量）。"""
        if T is None:
            T = self.fields()[3]
        if self.case == "rb":
            j = self.ny // 2
            q = -(T[j + 1, :] - T[j - 1, :]) / 2.0
        else:
            i = self.nx // 2
            q = -(T[:, i + 1] - T[:, i - 1]) / 2.0
        return float(q.mean().item()) * self.H / self.dT

    def nu_energy(self, T: torch.Tensor | None = None) -> dict[str, float]:
        """能量平衡 Nu（第二**独立**估计）：定常时总热流沿壁法向守恒。

        RB（对标量作 x 平均，u 为竖向速度分量）::

            Nu(y) = ( <u_y T> − α·d<T>/dy ) · H / (α·ΔT)

        DHC（对 y 平均，u 为横向分量，H 取方腔边长 L）::

            Nu(x) = ( <u_x T> − α·d<T>/dx ) · L / (α·ΔT)

        纯导热时 u=0 且 d<T>/dn=−ΔT/H ⇒ Nu≡1（与壁面通量估计同一极限）。
        返回逐截面值及其内部区标准差——定常态下该值应为常数，故标准差本身
        就是定常与守恒性的诊断。
        """
        if T is None:
            _, ux, uy, T = self.fields()
        else:
            _, ux, uy = macroscopic(self.f)
        # 归一化因子：导热通量 α·ΔT/H
        norm = self.alpha_lat * self.dT / self.H
        if self.case == "rb":
            Tm = T.mean(dim=1)                      # <T>(y)
            uyT = (uy * T).mean(dim=1)              # <u_y T>(y)
            dT = torch.zeros_like(Tm)
            dT[1:-1] = (Tm[2:] - Tm[:-2]) / 2.0
            flux = uyT - self.alpha_lat * dT        # 总热流（向上为正）
            prof = flux / norm
            inner = prof[2:-2]
        else:
            Tm = T.mean(dim=0)                      # <T>(x)
            uxT = (ux * T).mean(dim=0)
            dT = torch.zeros_like(Tm)
            dT[1:-1] = (Tm[2:] - Tm[:-2]) / 2.0
            flux = uxT - self.alpha_lat * dT
            prof = flux / norm
            inner = prof[2:-2]
        return {
            "nu_energy": float(inner.mean().item()),
            "nu_energy_std": float(inner.std(unbiased=False).item()),
            "nu_energy_min": float(inner.min().item()),
            "nu_energy_max": float(inner.max().item()),
        }

    def enthalpy(self) -> float:
        assert self.g is not None
        return float(self.g.sum().item())

    def mode_amplitude(self, m: int = 1, T: torch.Tensor | None = None) -> float:
        """θ=T−T_base 在 cos(kx)·sin(π(y+½)/H) 上的有符号投影幅值。"""
        if T is None:
            T = self.fields()[3]
        theta = T - self.base_profile()
        y = torch.arange(self.ny, dtype=self.dtype, device=T.device) + 0.5
        shape = torch.sin(math.pi * y / self.H)
        x = torch.arange(self.nx, dtype=self.dtype, device=T.device)
        ck = torch.cos(2.0 * math.pi * m / self.nx * x)
        num = (theta * shape.unsqueeze(1) * ck.unsqueeze(0)).sum()
        den = (shape * shape).sum() * self.nx / 2.0
        return float((num / den).item())

    def info(self) -> dict[str, Any]:
        return {
            "case": self.case, "ny": self.ny, "nx": self.nx, "ra": self.ra,
            "pr": self.pr, "H": self.H, "nu_lat": self.nu_lat,
            "alpha_lat": self.alpha_lat, "tau_f": self.tau_f_eff,
            "tau_T": self.tau_T, "g_beta": self.g_beta, "dT": self.dT,
            "gap_flow": self.gap_flow, "wall_bc": self.wall_bc,
            "t_hot": self.t_hot, "t_cold": self.t_cold,
            "enable_buoyancy": self.enable_buoyancy,
            "ma_estimate": self.alpha_lat * math.sqrt(self.ra * self.pr) / (self.H * CS),
            "steps": self.step_count,
        }


def make_convection(
    *,
    ny: int,
    nx: int | None = None,
    ra: float,
    pr: float = 0.71,
    case: str = "rb",
    dT: float = 1.0,
    t_ref: float = 0.0,
    ma_max: float | None = 0.10,
    alpha: float | None = None,
    tau_f: float | None = None,
    enable_buoyancy: bool = True,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
) -> Convection2D:
    """构造求解器；未显式给输运系数时按 Ma 约束自动选 α。

    ``Ma ≈ α·√(Ra·Pr)/(H·c_s)``（由 U=√(gβΔT·H) 与 Ra 定义联立），故取
    ``α = min(0.10, ma_max·H·c_s/√(Ra·Pr))``，并保证 τ_T ≥ 0.52。
    """
    if alpha is None and tau_f is None:
        if ma_max is None:
            alpha = 0.05
        else:
            alpha = ma_max * float(ny) * CS / math.sqrt(max(ra * pr, 1e-30))
            alpha = min(alpha, 0.10)
            alpha = max(alpha, (0.52 - 0.5) / 3.0)
    return Convection2D(
        ny=ny, nx=nx if nx is not None else ny, ra=ra, pr=pr, case=case, dT=dT,
        t_ref=t_ref, alpha=alpha, tau_f=tau_f, enable_buoyancy=enable_buoyancy,
        dtype=dtype, device=device,
    )


# ── C1：线性起稳 σ(Ra)、σ(k) ───────────────────────────────────────────────


def _fit_log_slope(t: Sequence[float], a: Sequence[float]) -> tuple[float, float, int]:
    """log|A|~t 最小二乘斜率（保留符号投影，不整流；返回 slope, R², n）。"""
    import numpy as np

    tt = np.asarray(t, dtype=float)
    aa = np.asarray(a, dtype=float)
    if aa.size == 0:
        return float("nan"), float("nan"), 0
    sign = 1.0 if aa[0] >= 0 else -1.0
    aa = sign * aa
    mask = aa > 0
    if mask.sum() < 3:
        return float("nan"), float("nan"), int(mask.sum())
    tt, ll = tt[mask], np.log(aa[mask])
    A = np.vstack([tt, np.ones_like(tt)]).T
    sol, *_ = np.linalg.lstsq(A, ll, rcond=None)
    resid = ll - A @ sol
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((ll - ll.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(sol[0]), r2, int(mask.sum())


def measure_growth_rate(
    *,
    ny: int = 32,
    nx: int | None = None,
    ra: float,
    pr: float = 0.71,
    mode: int = 1,
    alpha: float | None = 1.0 / 6.0,
    amp: float = 1e-4,
    steps: int | None = None,
    sample_every: int | None = None,
    window: tuple[float, float] = (0.15, 0.85),
    amp_floor: float = 1e-10,
    amp_ceiling: float = 0.05,
    device: str = "cpu",
) -> dict[str, Any]:
    """RB 小扰动起稳：时间推进测 σ，换算无量纲 σ* = σ·H²/α。

    种子 = 导热基态 + ``amp·cos(kx)·sin(π(y+½)/H)`` 温度扰动（无速度扰动，
    避免声学瞬态）。观测 = 同一模态的有符号投影 A(t)，在 ``window`` 内对
    ``log|A|`` 做线性 LSQ。
    """
    nx = nx if nx is not None else 2 * ny
    sim = make_convection(ny=ny, nx=nx, ra=ra, pr=pr, case="rb", alpha=alpha,
                          device=device)
    T0 = sim.base_profile()
    y = (torch.arange(ny, dtype=sim.dtype, device=T0.device) + 0.5) / sim.H
    x = torch.arange(nx, dtype=sim.dtype, device=T0.device)
    k = 2.0 * math.pi * mode / nx
    T0 = T0 + amp * torch.sin(math.pi * y).unsqueeze(1) * torch.cos(k * x).unsqueeze(0)
    sim.set_temperature(T0)
    if steps is None:
        steps = int(min(max(4000, 0.35 * sim.H ** 2 / sim.alpha_lat), 400000))
    if sample_every is None:
        sample_every = max(1, steps // 300)
    ts: list[float] = [0.0]
    amps: list[float] = [sim.mode_amplitude(mode)]
    while sim.step_count < steps:
        sim.step(min(sample_every, steps - sim.step_count))
        ts.append(float(sim.step_count))
        amps.append(sim.mode_amplitude(mode))
    # 自适应拟合窗：先把幅值限制在"可测且线性"的区间
    #   |A| ∈ [amp_floor, amp_ceiling]（默认 1e-10 / 0.05）
    # 再在该连续段内取 window 分数段。否则强衰减臂会掉到 float64 舍入底
    # （1e-16），强增长臂会进入非线性饱和，两种情形拟合出的 σ 都是垃圾。
    import numpy as _np

    aa = _np.abs(_np.asarray(amps, dtype=float))
    ok = (aa >= amp_floor) & (aa <= amp_ceiling)
    best = (0, 0)
    i = 0
    while i < len(ok):
        if not ok[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(ok) and ok[j + 1]:
            j += 1
        if j - i > best[1] - best[0]:
            best = (i, j)
        i = j + 1
    i0, i1 = best
    span = i1 - i0
    if span >= 6:
        a0 = i0 + int(window[0] * span)
        a1 = max(a0 + 3, i0 + int(window[1] * span))
    else:  # 落不出可用段：退回整段（结果由 fit_r2/valid 标出）
        a0, a1 = 0, len(ts) - 1
    slope, r2, npts = _fit_log_slope(ts[a0:a1], amps[a0:a1])
    return {
        "ra": ra, "ra_over_rac": ra / RA_CRIT, "ny": ny, "nx": nx, "mode": mode,
        "k": 2.0 * math.pi * mode / nx, "kH": 2.0 * math.pi * mode / nx * sim.H,
        "alpha": sim.alpha_lat, "steps": sim.step_count,
        "sigma_lattice": slope, "sigma_star": slope * sim.H ** 2 / sim.alpha_lat,
        "fit_r2": r2, "fit_points": npts,
        "fit_window": (float(ts[a0]), float(ts[a1 - 1])) if len(ts) > 1 else (0.0, 0.0),
        "valid": bool(span >= 6),
        "amp_final": amps[-1], "amp_initial": amps[0], "ma": sim.mach(),
        "t": ts, "amp": amps, "params": sim.info(),
    }


def critical_rayleigh_measured(
    *,
    ny: int = 32,
    nx: int | None = None,
    pr: float = 0.71,
    ratios: Sequence[float] = (0.90, 0.95, 1.00, 1.05, 1.10),
    mode: int = 1,
    **kw: Any,
) -> dict[str, Any]:
    """时间推进扫 Ra（固定 k）→ σ*(Ra) 零点插值 / 线性回归给 Ra_c。"""
    import numpy as np

    arms = []
    for r in ratios:
        out = measure_growth_rate(ny=ny, nx=nx, ra=r * RA_CRIT, pr=pr, mode=mode, **kw)
        out.pop("t", None)
        out.pop("amp", None)
        arms.append(out)
    crossing = None
    for a, b in zip(arms, arms[1:]):
        ya, yb = a["sigma_star"], b["sigma_star"]
        if ya * yb <= 0 and ya != yb:
            crossing = a["ra"] + (0.0 - ya) * (b["ra"] - a["ra"]) / (yb - ya)
            break
    xs = np.array([a["ra"] for a in arms], dtype=float)
    ys = np.array([a["sigma_star"] for a in arms], dtype=float)
    A = np.vstack([xs, np.ones_like(xs)]).T
    sol, *_ = np.linalg.lstsq(A, ys, rcond=None)
    ra_c_reg = float(-sol[1] / sol[0]) if sol[0] != 0 else float("nan")
    ra_c = float(crossing) if crossing is not None else ra_c_reg
    return {
        "arms": arms, "ra_c_interp": crossing, "ra_c_regression": ra_c_reg,
        "ra_c": ra_c, "ra_c_ref": RA_CRIT,
        "ra_c_error_pct": 100.0 * abs(ra_c - RA_CRIT) / RA_CRIT,
        "k": arms[0]["k"], "kH": arms[0]["kH"], "ny": ny, "nx": arms[0]["nx"],
    }


def critical_wavenumber_measured(
    *,
    ny: int = 32,
    pr: float = 0.71,
    ra: float = 1.2 * RA_CRIT,
    boxes: Sequence[tuple[int, int]] | None = None,
    **kw: Any,
) -> dict[str, Any]:
    """固定超临界 Ra，扫（盒宽 nx, 模数 m）→ σ*(a) 抛物顶点给最不稳定波数。

    无量纲波数 a = k·d，k = 2πm/nx（格子单位），d 为**流动间隙**
    （``gap_flow``，pre-半程反弹实测 ny−2）。盒宽按 Γ=nx/ny ∈ [1.6, 2.5]
    取，使 a 覆盖 2.4–3.7（跨过 a_c=3.11632）。
    """
    import numpy as np

    if boxes is None:
        nxs = [int(round(g * ny / 2)) * 2 for g in (1.6, 1.8, 2.0, 2.2, 2.5)]
        boxes = [(nx, 1) for nx in nxs]
    arms = []
    for nx, m in boxes:
        out = measure_growth_rate(ny=ny, nx=nx, ra=ra, pr=pr, mode=m, **kw)
        out.pop("t", None)
        out.pop("amp", None)
        d = float(out["params"].get("gap_flow", ny - 2))
        out["a"] = 2.0 * math.pi * m / nx * d
        out["box"] = (nx, m)
        arms.append(out)
    a = np.array([x["a"] for x in arms], dtype=float)
    sg = np.array([x["sigma_star"] for x in arms], dtype=float)
    A = np.vstack([a ** 2, a, np.ones_like(a)]).T
    sol, *_ = np.linalg.lstsq(A, sg, rcond=None)
    qa, qb = float(sol[0]), float(sol[1])
    a_c = -qb / (2.0 * qa) if qa != 0 else float("nan")
    return {
        "arms": arms, "ra": ra, "k_c_measured": a_c, "k_c_ref": A_CRIT,
        "k_c_error_pct": 100.0 * abs(a_c - A_CRIT) / A_CRIT,
        "k_c_argmax_arm": float(a[int(np.argmax(sg))]),
        "parabola": {"a": qa, "b": qb, "c": float(sol[2])},
        "fit_rms": float(np.sqrt(((sg - A @ sol) ** 2).mean())),
    }


# ── C2/C3：定常 Nu + GCI ───────────────────────────────────────────────────


def run_to_steady(
    *,
    case: str = "rb",
    ny: int = 64,
    nx: int | None = None,
    ra: float = 1e4,
    pr: float = 0.71,
    dT: float = 1.0,
    t_ref: float = 0.0,
    alpha: float | None = None,
    ma_max: float | None = 0.10,
    turnover_target: float = 150.0,
    min_steps: int = 20000,
    max_steps: int = 2_000_000,
    window_frac: float = 0.25,
    report_every: int = 2000,
    perturb: float = 1e-3,
    device: str = "cpu",
    verbose: bool = False,
    dtype: torch.dtype = torch.float64,
    early_stop_tol: float | None = 2e-3,
    min_frac: float = 0.5,
    max_frac: float = 1.0,
    return_fields: bool = False,
) -> dict[str, Any]:
    """推进到定常/统计定常，返回时间平均 Nu 与收敛证据。

    ``return_fields=True`` 时额外返回末态场（``T_field``/``u_field``/
    ``v_field``，torch 张量，非 JSON 可序列化）——供等温线/云图后处理；
    默认关闭以保持返回值可直接落 JSON。

    ``early_stop_tol``：块平均 Nu 的相对漂移连续两块小于该阈值（且已跑过
    ``min_frac`` 预算）即停；``max_frac`` 为预算上限系数（相对 H²/α 或
    ``turnover_target``）。
    """
    sim = make_convection(ny=ny, nx=nx or ny, ra=ra, pr=pr, case=case, dT=dT,
                          t_ref=t_ref, alpha=alpha, ma_max=ma_max, device=device,
                          dtype=dtype)
    T0 = sim.base_profile()
    if perturb:
        torch.manual_seed(12345)
        T0 = T0 + perturb * sim.dT * (torch.rand_like(T0) - 0.5)
    sim.set_temperature(T0)
    u_ff = math.sqrt(max(sim.g_beta * sim.dT * sim.H, 1e-30))
    t_turn = sim.H / u_ff
    diff_steps = sim.H**2 / sim.alpha_lat
    budget = min(max(min_steps, turnover_target * t_turn), diff_steps)
    steps = int(min(max_steps, budget * (max_frac if early_stop_tol is not None else 1.0)))
    nu_hist: list[float] = []
    t0 = time.time()
    block_means: list[float] = []
    while sim.step_count < steps:
        sim.step(min(report_every, steps - sim.step_count))
        nu_hist.append(sim.nusselt()["nu"])
        if verbose and sim.step_count % (report_every * 10) == 0:
            print(f"[{case} Ra={ra:g} ny={ny}] step={sim.step_count} "
                  f"Nu={nu_hist[-1]:.5f} umax={sim.u_max():.4f} "
                  f"Ma={sim.mach():.3f}", flush=True)
        if early_stop_tol is not None and sim.step_count % max(report_every, 1) == 0:
            block_means.append(sum(nu_hist[-5:]) / len(nu_hist[-5:]))
            done = sim.step_count >= min_frac * budget
            if done and len(block_means) >= 3:
                m1, m2 = block_means[-2], block_means[-3]
                if abs(m1 - m2) / max(abs(m1), 1e-30) < early_stop_tol:
                    break
    n_win = max(3, int(window_frac * len(nu_hist)))
    tail = nu_hist[-n_win:]
    prev = nu_hist[-2 * n_win:-n_win] or tail
    nu_mean = sum(tail) / len(tail)
    nu_std = (sum((x - nu_mean) ** 2 for x in tail) / len(tail)) ** 0.5
    nu_prev = sum(prev) / len(prev)
    diag = sim.nusselt()
    _, ux_f, uy_f, T = sim.fields()
    out = {
        **sim.info(), "steps": sim.step_count,
        "elapsed_s": round(time.time() - t0, 2),
        "diff_steps": diff_steps, "budget_steps": int(budget),
        "final_report_every": report_every,
        "max_frac": max_frac,
        "early_stop_tol": early_stop_tol,
        "min_frac": min_frac,
        "t_turn": t_turn, "turnovers": sim.step_count / t_turn,
        "nu": nu_mean, "nu_std": nu_std,
        "nu_rel_std": nu_std / abs(nu_mean) if nu_mean else float("nan"),
        "nu_last": diag["nu"], "nu_hot": diag["nu_hot"], "nu_cold": diag["nu_cold"],
        "nu_imbalance": diag["nu_imbalance"],
        "nu_midplane": sim.nu_midplane(T),
        **sim.nu_energy(T),
        "nu_drift": abs(nu_mean - nu_prev) / max(abs(nu_mean), 1e-30),
        "u_max": sim.u_max(), "ma": sim.mach(),
        "t_min": float(T.min().item()), "t_max": float(T.max().item()),
        "nu_history": nu_hist,
    }
    if return_fields:
        out["T_field"] = T.detach().cpu()
        out["u_field"] = ux_f.detach().cpu()
        out["v_field"] = uy_f.detach().cpu()
    return out


def dimensionless_selfconsistency(
    *,
    case: str = "rb",
    ny: int = 48,
    ra: float = 1e4,
    pr: float = 0.71,
    dTs: Sequence[float] = (0.5, 1.0, 2.0),
    steps: int = 40000,
    **kw: Any,
) -> dict[str, Any]:
    """C4 无量纲自洽：固定 (Ra,Pr)，改 ΔT（≡β 与 g 的反向缩放）后 Nu 必须不变。

    Boussinesq 下只有乘积 g·β·ΔT 进入动量方程，温度方程对 T 线性，故
    ``ΔT→sΔT`` 且 ``gβ→gβ/s``（本模块由 ``g_beta=Ra·να/(H³ΔT)`` 自动满足）
    在离散层面等价，差异只能来自浮点结合律。若实现里把 ΔT 或 T_ref 写死，
    该检验会立刻炸开（这正是本条要防的回归）。
    """
    rows = []
    for dT in dTs:
        out = run_to_steady(case=case, ny=ny, ra=ra, pr=pr, dT=dT,
                            turnover_target=1e9, min_steps=steps, max_steps=steps,
                            perturb=1e-3, **kw)
        rows.append({"dT": dT, "nu": out["nu"], "nu_std": out["nu_std"],
                     "ma": out["ma"], "steps": out["steps"],
                     "u_max": out["u_max"]})
    nus = [r["nu"] for r in rows]
    mean = sum(nus) / len(nus)
    spread = (max(nus) - min(nus)) / abs(mean) if mean else float("nan")
    return {"rows": rows, "mean": mean, "spread": spread,
            "spread_pct": 100.0 * spread, "ra": ra, "ny": ny, "case": case}


# ── C5：负控与极限 ─────────────────────────────────────────────────────────


def no_buoyancy_control(
    *, ny: int = 32, ra: float = 1e4, steps: int = 20000, case: str = "rb",
    alpha: float | None = 1.0 / 6.0,
) -> dict[str, Any]:
    """关浮力 → 纯导热：Nu 必须 = 1，温度剖面 = 解析线性解。"""
    sim = make_convection(ny=ny, nx=ny, ra=ra, case=case, alpha=alpha,
                          enable_buoyancy=False)
    T0 = sim.base_profile()
    torch.manual_seed(7)
    T0 = T0 + 0.01 * sim.dT * (torch.rand_like(T0) - 0.5)
    sim.set_temperature(T0)
    sim.step(steps)
    T = sim.fields()[3]
    nu = sim.nusselt(T)
    lin = float((T - sim.base_profile()).abs().max().item())
    return {
        "case": case, "steps": steps, "nu": nu["nu"],
        "nu_error": abs(nu["nu"] - 1.0), "nu_hot": nu["nu_hot"],
        "nu_cold": nu["nu_cold"], "u_max": sim.u_max(),
        "profile_linf": lin, "profile_linf_over_dT": lin / sim.dT,
        "ma": sim.mach(),
    }


def _adiabatic_bc_factory() -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    def _bc(g_post: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        g = g.clone()
        g[3, 0, :] = g_post[4, 0, :]
        g[4, -1, :] = g_post[3, -1, :]
        return g

    return _bc


def adiabatic_enthalpy_drift(
    *, ny: int = 32, nx: int = 64, steps: int = 10000,
    alpha: float | None = 1.0 / 6.0,
) -> dict[str, Any]:
    """双绝热壁 + 无浮力：总焓相对漂移（D2Q5 通量守恒性）。

    上下两壁改为零通量反弹（绝热），左右仍周期，故总焓是守恒量；漂移只应
    来自浮点舍入。这是纯导热/绝热壁离散守恒性的直接检验。
    """
    sim = make_convection(ny=ny, nx=nx, ra=0.0, case="rb", alpha=alpha,
                          enable_buoyancy=False)
    torch.manual_seed(3)
    T0 = 0.5 + 0.2 * (torch.rand((ny, nx), dtype=sim.dtype) - 0.5)
    sim.set_temperature(T0)
    sim._thermal_bc = _adiabatic_bc_factory()  # type: ignore[method-assign]
    h0 = sim.enthalpy()
    sim.step(steps)
    h1 = sim.enthalpy()
    T = sim.fields()[3]
    return {
        "steps": steps, "enthalpy_initial": h0, "enthalpy_final": h1,
        "drift_rel": abs(h1 - h0) / abs(h0),
        "t_min": float(T.min().item()), "t_max": float(T.max().item()),
    }


# ── C6：正弦温度波衰减反演 α_eff 与精度阶 ──────────────────────────────────


def diffusion_decay_alpha(
    *, n: int = 64, alpha: float = 1.0 / 6.0, decay: float = 0.5,
    dtype: torch.dtype = torch.float64,
) -> dict[str, Any]:
    """正弦温度波衰减 → 有效热扩散率 α_eff，及其对解析 α 的相对误差。

    周期域 n×n，无流动，初值 T=cos(2πx/n)（一个波长覆盖域）。D2Q5 BGK
    扩散的解析衰减率：A(t)=A0·exp(−α_eff k² t)，k=2π/n。反演
    α_eff = −ln(A/A0)/(k²·t)。空间截断误差 O((kh)²) ⇒ 网格加倍误差比 → 4。
    """
    from .thermal import collide_thermal_bgk as _coll, temperature_stream as _ts

    w5 = torch.tensor(_W5, dtype=dtype)
    c5x = torch.tensor([0.0, 1.0, -1.0, 0.0, 0.0], dtype=dtype)
    x = torch.arange(n, dtype=dtype)
    k = 2.0 * math.pi / n
    zeros = torch.zeros((n, n), dtype=dtype)
    ck = torch.cos(k * x)
    T0 = torch.cos(k * x).unsqueeze(0).expand(n, n).contiguous()
    g = w5.view(5, 1, 1) * T0.unsqueeze(0) * (1.0 + 3.0 * c5x.view(5, 1, 1) * zeros)
    tau_T = 3.0 * alpha + 0.5
    amp0 = float((T0.sum(0) * ck).sum().item() / (n / 2.0))
    t_total = int(max(20, round(decay / (alpha * k * k))))
    for _ in range(t_total):
        T = g.sum(0)
        g = _ts(_coll(g, T, zeros, zeros, tau_T))
    T = g.sum(0)
    amp1 = float((T.sum(0) * ck).sum().item() / (n / 2.0))
    alpha_eff = -math.log(amp1 / amp0) / (k * k * t_total)
    return {
        "n": n, "k": k, "kh": k, "alpha_theory": alpha, "alpha_eff": alpha_eff,
        "alpha_rel_error": (alpha_eff - alpha) / alpha,
        "alpha_rel_error_abs": abs(alpha_eff - alpha) / alpha,
        "steps": t_total, "amp0": amp0, "amp1": amp1,
        "tau_T": tau_T,
    }


def diffusion_decay_order(
    *, ns: Sequence[int] = (16, 32, 64, 128), alpha: float = 1.0 / 6.0,
    decay: float = 0.5,
) -> dict[str, Any]:
    """网格加倍误差比 → 观测精度阶（期望 4±20%）。"""
    rows = [diffusion_decay_alpha(n=n, alpha=alpha, decay=decay) for n in ns]
    ratios = []
    for a, b in zip(rows, rows[1:]):
        ea = abs(a["alpha_rel_error"])
        eb = abs(b["alpha_rel_error"])
        ratios.append({
            "n_pair": (a["n"], b["n"]),
            "err_coarse": ea, "err_fine": eb,
            "ratio": (ea / eb) if eb > 0 else float("nan"),
        })
    return {"rows": rows, "ratios": ratios}

# ── 几何口径标定（速度/温度有效间隙）───────────────────────────────────────


def _decay_rate(vals: Sequence[float], t: Sequence[float]) -> tuple[float, float]:
    """log|A| ~ t 的斜率（返回 slope, R²）。"""
    import numpy as np

    aa = np.abs(np.asarray(vals, dtype=float))
    ok = aa > (aa.max() * 1e-10 if aa.size else 1.0)
    if ok.sum() < 4:
        return float("nan"), float("nan")
    tt = np.asarray(t, dtype=float)[ok]
    ll = np.log(aa[ok])
    A = np.vstack([tt, np.ones_like(tt)]).T
    sol, *_ = np.linalg.lstsq(A, ll, rcond=None)
    resid = ll - A @ sol
    ss_tot = float(((ll - ll.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    return float(sol[0]), r2


def calibrate_gaps(
    *, ny: int = 32, nx: int | None = None, alpha: float = 1.0 / 6.0,
    pr: float = 0.71, steps: int = 4000, sample_every: int = 20,
    wall_bc: str = "post", amp: float = 1e-3,
) -> dict[str, Any]:
    """直接测量温度/速度的**有效间隙**（几何口径标定，替代"信 H=ny"）。

    温度：RB 几何、无浮力，初值 = 导热基态 + A·sin(π(y+½)/ny)；其最慢竖模
    以 λ_T = α(π/d_T)² 衰减 ⇒ d_T = π√(α/λ_T)。
    速度：同几何，初值给 u_x = A·sin(π(y+½)/ny)（无散剪切模，作为 Stokes
    问题第一模态），以 λ_v = ν(π/d_v)² 衰减 ⇒ d_v = π√(ν/λ_v)。

    纯导热时两式都是精确的连续极限；实测 d 与 ny 的偏差即边界处理的几何偏移
    （半程壁 ⇒ d≈ny；壁落在节点行 ⇒ d≈ny−1 或 ny−2）。
    """
    nx = nx or 2 * ny
    out: dict[str, Any] = {"ny": ny, "nx": nx, "alpha": alpha, "wall_bc": wall_bc}
    # 温度
    sim = make_convection(ny=ny, nx=nx, ra=0.0, case="rb", alpha=alpha,
                          enable_buoyancy=False)
    sim.wall_bc = wall_bc
    y = (torch.arange(ny, dtype=sim.dtype, device=sim.f.device) + 0.5) / sim.H
    T0 = sim.base_profile() + amp * torch.sin(math.pi * y).unsqueeze(1).expand(ny, nx)
    sim.set_temperature(T0)
    shape = torch.sin(math.pi * y)
    base = sim.base_profile()
    ts, amps = [0.0], [float(((T0 - base) * shape.unsqueeze(1)).sum().item())]
    while sim.step_count < steps:
        sim.step(sample_every)
        T = sim.fields()[3]
        ts.append(float(sim.step_count))
        amps.append(float(((T - base) * shape.unsqueeze(1)).sum().item()))
    h = len(ts)
    lam_T, r2_T = _decay_rate(amps[int(0.25 * h):], ts[int(0.25 * h):])
    d_T = math.pi * math.sqrt(-alpha / lam_T) if lam_T < 0 else float("nan")
    out["thermal"] = {"lambda": lam_T, "r2": r2_T, "d_eff": d_T,
                      "d_over_ny": d_T / ny}
    # 速度（剪切模）
    sim2 = make_convection(ny=ny, nx=nx, ra=0.0, case="rb", alpha=alpha,
                           enable_buoyancy=False)
    sim2.wall_bc = wall_bc
    zeros = torch.zeros((ny, nx), dtype=sim2.dtype)
    ux0 = amp * torch.sin(math.pi * y).unsqueeze(1).expand(ny, nx).contiguous()
    sim2.f = equilibrium(torch.ones_like(zeros), ux0, zeros)
    ts, amps = [0.0], [float((ux0 * shape.unsqueeze(1)).sum().item())]
    while sim2.step_count < steps:
        sim2.step(sample_every)
        _, ux, _, _ = sim2.fields()
        ts.append(float(sim2.step_count))
        amps.append(float((ux * shape.unsqueeze(1)).sum().item()))
    h = len(ts)
    lam_v, r2_v = _decay_rate(amps[int(0.25 * h):], ts[int(0.25 * h):])
    d_v = math.pi * math.sqrt(-sim2.nu_lat / lam_v) if lam_v < 0 else float("nan")
    out["velocity"] = {"lambda": lam_v, "r2": r2_v, "d_eff": d_v,
                       "d_over_ny": d_v / ny, "nu_nominal": sim2.nu_lat}
    out["consistent"] = bool(
        math.isfinite(d_T) and math.isfinite(d_v)
        and abs(d_T - d_v) / max(d_T, d_v) < 0.03)
    return out
