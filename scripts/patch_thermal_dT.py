"""Apply the ΔT fix + Ra_effective audit field to thermal.py / thermal_verification.py."""
import re
import pathlib

root = pathlib.Path("src/tensorlbm")
tp = root / "thermal.py"
src = tp.read_text()

old_params = '''def thermal_params(nx: int, ra: float, pr: float, tau: float) -> dict[str, float]:
    """由 Ra、Pr、τ 反解格子参数。

    方腔物理长度 L = nx（half-way 壁面位于 x=∓0.5，间距 = nx 个格子单位，
    与 verified/cavity_re100 的 H = nx 约定一致），ν = (τ − 1/2)/3，
    α = ν/Pr，τ_T = 3α + 1/2，g·β = Ra·ν·α/(ΔT·L³)（ΔT = 1）。
    """
    nu = (tau - 0.5) / 3.0
    alpha = nu / pr
    tau_T = 3.0 * alpha + 0.5
    H = float(nx)
    g_beta = ra * nu * alpha / (H**3)  # ΔT = 1
    return {"nu": nu, "alpha": alpha, "tau_T": tau_T, "H": H, "g_beta": g_beta}
'''

new_params = '''def thermal_params(
    nx: int, ra: float, pr: float, tau: float, dT: float = 1.0
) -> dict[str, float]:
    """由 Ra、Pr、τ、ΔT 反解格子参数。

    方腔物理长度 L = nx（half-way 壁面位于 x=∓0.5，间距 = nx 个格子单位，
    与 verified/cavity_re100 的 H = nx 约定一致），ν = (τ − 1/2)/3，
    α = ν/Pr，τ_T = 3α + 1/2。

    浮力系数的定义是 **g·β = Ra·ν·α/(ΔT·L³)**：Ra 的定义里含 ΔT
    （Ra = g·β·ΔT·L³/(ν·α)），因此 ΔT ≠ 1 时必须除以 ΔT，否则同一条
    物理工况（同一 Ra）在 ΔT 翻倍时会被算成 Ra 翻倍。
    *dT* 缺省 1.0 与旧行为兼容；dT ≤ 0 直接 ValueError（静默 NaN 更难查）。
    """
    if not (dT > 0.0):
        raise ValueError(f"thermal_params: dT must be > 0, got {dT!r}")
    nu = (tau - 0.5) / 3.0
    alpha = nu / pr
    tau_T = 3.0 * alpha + 0.5
    H = float(nx)
    g_beta = ra * nu * alpha / (dT * H**3)
    return {"nu": nu, "alpha": alpha, "tau_T": tau_T, "H": H, "g_beta": g_beta}
'''
assert old_params in src
src = src.replace(old_params, new_params)

old_use = '''    p = thermal_params(nx, ra, pr, tau)
    nu_lat, alpha_lat, tau_T, H, g_beta = (p["nu"], p["alpha"], p["tau_T"], p["H"], p["g_beta"])
    dT = t_hot - t_cold
'''
new_use = '''    dT = t_hot - t_cold
    p = thermal_params(nx, ra, pr, tau, dT=dT)
    nu_lat, alpha_lat, tau_T, H, g_beta = (p["nu"], p["alpha"], p["tau_T"], p["H"], p["g_beta"])
'''
assert old_use in src
src = src.replace(old_use, new_use)
tp.write_text(src)
print("thermal.py patched")

tv = root / "thermal_verification.py"
src = tv.read_text()

old_rb = '''    H = float(ny - 1)
    dT = t_hot - t_cold

    # Lattice parameters
    nu_lat = (tau - 0.5) / 3.0
    alpha_lat = nu_lat / pr
    tau_T = 3.0 * alpha_lat + 0.5
    g_beta = ra * nu_lat * alpha_lat / (H**3) if enable_buoyancy else 0.0
'''
new_rb = '''    H = float(ny - 1)
    dT = t_hot - t_cold
    if not (dT > 0.0):
        raise ValueError(f"simulate_rayleigh_benard_2d: dT = t_hot - t_cold must be > 0, got {dT!r}")

    # Lattice parameters.  Ra = g_beta * dT * H^3 / (nu * alpha), so the
    # buoyancy coefficient carries a 1/dT: the same Ra must give the same flow
    # whatever the imposed temperature difference (dimensionless invariance).
    nu_lat = (tau - 0.5) / 3.0
    alpha_lat = nu_lat / pr
    tau_T = 3.0 * alpha_lat + 0.5
    g_beta = ra * nu_lat * alpha_lat / (dT * H**3) if enable_buoyancy else 0.0
'''
assert old_rb in src
src = src.replace(old_rb, new_rb)

# RBResult: add ra_effective field
old_rbf = '''    g_beta: float
    H: float
    steps: int
    elapsed_s: float
    nu: float
    nu_bot: float'''
new_rbf = '''    g_beta: float
    H: float
    dT: float
    ra_effective: float
    steps: int
    elapsed_s: float
    nu: float
    nu_bot: float'''
assert old_rbf in src
src = src.replace(old_rbf, new_rbf)

old_rbr = '''    return RBResult(
        nx=nx, ny=ny, ra=ra, pr=pr, tau=tau, tau_T=tau_T,
        nu_lat=nu_lat, alpha_lat=alpha_lat, g_beta=g_beta, H=H,
        steps=steps, elapsed_s=round(elapsed, 1),'''
new_rbr = '''    ra_effective = (
        g_beta * dT * H**3 / (nu_lat * alpha_lat) if nu_lat * alpha_lat > 0.0 else 0.0
    )

    return RBResult(
        nx=nx, ny=ny, ra=ra, pr=pr, tau=tau, tau_T=tau_T,
        nu_lat=nu_lat, alpha_lat=alpha_lat, g_beta=g_beta, H=H,
        dT=dT, ra_effective=ra_effective,
        steps=steps, elapsed_s=round(elapsed, 1),'''
assert old_rbr in src
src = src.replace(old_rbr, new_rbr)

# DHCResult fields
old_dhf = '''    g_beta: float
    H: float
    steps: int
    elapsed_s: float
    nu_grad2: float'''
new_dhf = '''    g_beta: float
    H: float
    dT: float
    ra_effective: float
    steps: int
    elapsed_s: float
    nu_grad2: float'''
assert old_dhf in src
src = src.replace(old_dhf, new_dhf)

old_dhr = '''    return DHCResult(
        nx=nx, ra=ra, pr=pr, tau=tau,
        tau_T=res["tau_T"], nu_lat=res["nu"], alpha_lat=res["alpha"],
        g_beta=res["g_beta"], H=res["H"],
        steps=res["steps"], elapsed_s=res["elapsed_s"],'''
new_dhr = '''    _dT = t_hot - t_cold
    _ra_eff = (
        res["g_beta"] * _dT * res["H"] ** 3 / (res["nu"] * res["alpha"])
        if res["nu"] * res["alpha"] > 0.0
        else 0.0
    )

    return DHCResult(
        nx=nx, ra=ra, pr=pr, tau=tau,
        tau_T=res["tau_T"], nu_lat=res["nu"], alpha_lat=res["alpha"],
        g_beta=res["g_beta"], H=res["H"],
        dT=_dT, ra_effective=_ra_eff,
        steps=res["steps"], elapsed_s=res["elapsed_s"],'''
assert old_dhr in src
src = src.replace(old_dhr, new_dhr)
tv.write_text(src)
print("thermal_verification.py patched")