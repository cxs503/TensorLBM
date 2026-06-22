"""Engineering simulation templates for the TensorLBM platform.

Provides pre-configured scenario templates that mirror the workflow-centric
approach of professional LBM simulation tools:

  - External aerodynamics
  - Ship/marine resistance
  - Multiphase free-surface
  - Internal duct flow
  - Rotating machinery
  - Porous media
  - Thermal convection
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

router = APIRouter()

# ---------------------------------------------------------------------------
# Template catalogue
# ---------------------------------------------------------------------------

_TEMPLATES: list[dict[str, Any]] = [
    # ------------------------------------------------------------------ #
    # Category: External Aerodynamics / Hydrodynamics
    # ------------------------------------------------------------------ #
    {
        "id": "ext_aero_cylinder",
        "category": "external_flow",
        "title": "2D Cylinder – External Flow",
        "title_zh": "二维绕圆柱外流",
        "icon": "bi-circle",
        "description": (
            "Classic 2-D cylinder vortex-shedding benchmark. "
            "Suitable for VIV onset studies, wake characterisation, and "
            "solver validation against Williamson (1988) Strouhal data."
        ),
        "description_zh": (
            "经典二维圆柱绕流涡脱基准。适用于涡激振动（VIV）起始研究、"
            "尾流特征分析以及与 Williamson (1988) Strouhal 数据的求解器验证。"
        ),
        "difficulty": "beginner",
        "solver_type": "cylinder_flow",
        "default_config": {
            "nx": 320, "ny": 100, "u_in": 0.08, "re": 100.0,
            "radius": 12.0, "n_steps": 5000, "output_interval": 500,
            "device": "cpu",
        },
        "parameter_hints": {
            "re": "Reynolds number Re = U·D/ν.  Vortex shedding occurs for Re > 47.",
            "n_steps": "Recommend ≥ 5 000 steps so at least 4–5 shedding cycles are captured.",
        },
        "references": ["Williamson (1988) J. Fluid Mech.", "Zdravkovich (1997)"],
    },
    {
        "id": "ext_aero_ship_hull",
        "category": "external_flow",
        "title": "3D Ship Hull – Resistance",
        "title_zh": "三维船体阻力",
        "icon": "bi-tsunami",
        "description": (
            "3-D Wigley / Series-60 / KCS hull resistance computation using "
            "Smagorinsky MRT-LBM.  Produces drag / lift forces, Cb, "
            "and wake visualisations.  Compare with ITTC-1957 friction line."
        ),
        "description_zh": (
            "使用 Smagorinsky MRT-LBM 进行三维 Wigley / Series-60 / KCS 船体阻力计算。"
            "输出阻力/升力、方形系数 Cb 及尾流可视化，并与 ITTC-1957 摩擦线比较。"
        ),
        "difficulty": "intermediate",
        "solver_type": "ship_hull_flow",
        "default_config": {
            "hull_type": "wigley",
            "nx": 120, "ny": 50, "nz": 40,
            "hull_length": 60, "hull_beam": 10, "hull_draft": 12,
            "re": 200.0, "n_steps": 4000, "output_interval": 500,
            "device": "cpu",
        },
        "parameter_hints": {
            "hull_type": "wigley | series60 | kcs",
            "re": "Physical Re for ship. LBM Re = U_lbm · L_lbm / ν_lbm.",
        },
        "references": ["ITTC-1957 friction line", "Larsson & Raven (2010)"],
    },
    {
        "id": "ext_aero_suboff",
        "category": "external_flow",
        "title": "SUBOFF Submarine – Resistance",
        "title_zh": "SUBOFF 潜艇阻力",
        "icon": "bi-submarine",
        "description": (
            "DARPA SUBOFF submarine resistance benchmark (bare hull, with sail, "
            "or full-appendage variants).  Validated against DTMB model tests."
        ),
        "description_zh": (
            "DARPA SUBOFF 潜艇阻力基准（裸体、带帆罩或全附件型式）。"
            "与 DTMB 模型试验数据对标。"
        ),
        "difficulty": "advanced",
        "solver_type": "suboff",
        "default_config": {
            "hull_type": "bare_hull",
            "nx": 200, "ny": 60, "nz": 60,
            "re": 150.0, "n_steps": 5000, "output_interval": 500,
            "device": "cpu",
        },
        "parameter_hints": {
            "hull_type": "bare_hull | with_sail | full",
        },
        "references": ["Groves et al. (1998) DTMB Technical Report"],
    },
    # ------------------------------------------------------------------ #
    # Category: Internal / Benchmark flows
    # ------------------------------------------------------------------ #
    {
        "id": "internal_lid_cavity",
        "category": "internal_flow",
        "title": "Lid-Driven Cavity",
        "title_zh": "顶盖驱动方腔",
        "icon": "bi-square",
        "description": (
            "2-D lid-driven cavity (Re = 100 … 1 000). "
            "Gold-standard validation against Ghia et al. (1982) "
            "centreline velocity profiles."
        ),
        "description_zh": (
            "二维顶盖驱动方腔（Re = 100 … 1 000）。"
            "与 Ghia 等（1982）中心线速度剖面经典数据对标。"
        ),
        "difficulty": "beginner",
        "solver_type": "lid_driven_cavity",
        "default_config": {
            "nx": 100, "ny": 100, "u_lid": 0.1, "re": 100.0,
            "n_steps": 20000, "output_interval": 2000,
            "device": "cpu",
        },
        "references": ["Ghia et al. (1982) J. Comput. Phys."],
    },
    {
        "id": "internal_backward_step",
        "category": "internal_flow",
        "title": "Backward-Facing Step",
        "title_zh": "倒台阶流动",
        "icon": "bi-layout-text-sidebar",
        "description": (
            "2-D backward-facing step reattachment benchmark. "
            "Reattachment length comparison against Armaly et al. (1983)."
        ),
        "description_zh": (
            "二维倒台阶再附流动基准。"
            "与 Armaly 等（1983）实验再附着长度对比。"
        ),
        "difficulty": "beginner",
        "solver_type": "backward_facing_step",
        "default_config": {
            "nx": 300, "ny": 60, "u_in": 0.05, "re": 100.0,
            "n_steps": 10000, "output_interval": 1000,
            "device": "cpu",
        },
        "references": ["Armaly et al. (1983) J. Fluid Mech."],
    },
    {
        "id": "internal_turbulent_channel",
        "category": "internal_flow",
        "title": "Turbulent Channel – Smagorinsky LES",
        "title_zh": "湍流槽道（Smagorinsky LES）",
        "icon": "bi-align-center",
        "description": (
            "Body-force driven turbulent channel flow with Smagorinsky LES. "
            "Log-law comparison against Moser et al. DNS (Re_τ = 180)."
        ),
        "description_zh": (
            "体力驱动湍流槽道，采用 Smagorinsky LES。"
            "与 Moser 等 DNS（Re_τ = 180）的对数律进行比较。"
        ),
        "difficulty": "intermediate",
        "solver_type": "turbulent_channel",
        "default_config": {
            "nx": 240, "ny": 120, "re_tau": 180.0,
            "n_steps": 30000, "output_interval": 2000,
            "device": "cpu",
        },
        "references": ["Moser, Kim & Mansour (1999) Phys. Fluids"],
    },
    # ------------------------------------------------------------------ #
    # Category: Multiphase / Free surface
    # ------------------------------------------------------------------ #
    {
        "id": "multiphase_dam_break",
        "category": "multiphase",
        "title": "Dam Break – Free Surface",
        "title_zh": "溃坝自由液面",
        "icon": "bi-water",
        "description": (
            "2-D dam-break free-surface collapse using Shan-Chen or "
            "Color-Gradient LBM.  Validation against Martin & Moyce (1952) "
            "surge-front position."
        ),
        "description_zh": (
            "使用 Shan-Chen 或颜色梯度 LBM 模拟二维溃坝自由液面演化。"
            "与 Martin & Moyce（1952）浪前位置实验对比。"
        ),
        "difficulty": "beginner",
        "solver_type": "dam_break",
        "default_config": {
            "nx": 200, "ny": 100, "multiphase_model": "cg",
            "n_steps": 5000, "output_interval": 500,
            "device": "cpu",
        },
        "references": ["Martin & Moyce (1952)", "Shan & Chen (1993)"],
    },
    {
        "id": "multiphase_sloshing",
        "category": "multiphase",
        "title": "Sloshing Tank – Marine",
        "title_zh": "液舱晃动（船舶）",
        "icon": "bi-align-bottom",
        "description": (
            "Partially filled rectangular tank under harmonic horizontal "
            "excitation. Validated against Faltinsen (1978) natural frequency formula."
        ),
        "description_zh": (
            "矩形液舱在水平谐波激励下的晃动响应。"
            "与 Faltinsen（1978）固有频率解析公式对标。"
        ),
        "difficulty": "intermediate",
        "solver_type": "sloshing_tank",
        "default_config": {
            "nx": 200, "ny": 100,
            "fill_ratio": 0.5, "exc_amplitude": 0.02, "exc_frequency": 0.5,
            "n_steps": 8000, "output_interval": 500,
            "device": "cpu",
        },
        "references": ["Faltinsen (1978) J. Ship Res."],
    },
    {
        "id": "multiphase_porous",
        "category": "multiphase",
        "title": "Porous Media Drainage",
        "title_zh": "多孔介质排水",
        "icon": "bi-grid-1x2",
        "description": (
            "Capillary drainage through a random porous medium. "
            "Two-phase Shan-Chen or Color-Gradient model. "
            "Validates Young-Laplace law and Washburn equation."
        ),
        "description_zh": (
            "随机多孔介质中的毛细管排水。使用两相 Shan-Chen 或颜色梯度模型。"
            "验证 Young-Laplace 定律和 Washburn 方程。"
        ),
        "difficulty": "intermediate",
        "solver_type": "porous_drainage",
        "default_config": {
            "nx": 150, "ny": 150, "porosity": 0.6,
            "multiphase_model": "sc", "n_steps": 10000, "output_interval": 1000,
            "device": "cpu",
        },
        "references": ["Pan et al. (2004) Phys. Rev. E"],
    },
    # ------------------------------------------------------------------ #
    # Category: Rotating machinery / Ocean engineering
    # ------------------------------------------------------------------ #
    {
        "id": "ocean_pipeline",
        "category": "ocean_engineering",
        "title": "Near-Bed Pipeline – VIV",
        "title_zh": "近床管道涡激振动",
        "icon": "bi-arrows-vertical",
        "description": (
            "2-D near-seabed pipeline flow with gap-ratio study. "
            "Validated against Bearman & Zdravkovich (1978) Strouhal data."
        ),
        "description_zh": (
            "近海床管道二维绕流，包含间隙比研究。"
            "与 Bearman & Zdravkovich（1978）Strouhal 数据对标。"
        ),
        "difficulty": "intermediate",
        "solver_type": "pipeline_flow",
        "default_config": {
            "nx": 320, "ny": 120, "re": 200.0, "gap_ratio": 0.5,
            "n_steps": 8000, "output_interval": 500,
            "device": "cpu",
        },
        "references": ["Bearman & Zdravkovich (1978) J. Fluid Mech."],
    },
    # ------------------------------------------------------------------
    # Advanced 3-D templates
    # ------------------------------------------------------------------
    {
        "id": "sphere_flow_d3q27",
        "category": "external_flow",
        "title": "3-D Sphere Flow (D3Q27)",
        "title_zh": "三维绕球流动（D3Q27）",
        "icon": "bi-circle",
        "description": (
            "3-D channel flow past a sphere using the D3Q27 lattice (27-velocity, "
            "4th-order isotropy). Validates Cd and wake structure against "
            "Johnson & Patel (1999) and Tomboulides (2000) DNS data."
        ),
        "description_zh": (
            "使用 D3Q27（27速度，四阶各向同性）格子的三维绕球通道流。"
            "与 Johnson & Patel (1999) 及 Tomboulides (2000) DNS 数据对比验证 Cd 和尾迹结构。"
        ),
        "difficulty": "advanced",
        "solver_type": "sphere_flow_d3q27",
        "default_config": {
            "nx": 120, "ny": 60, "nz": 60,
            "u_in": 0.06, "re": 50.0, "radius": 8.0,
            "n_steps": 1000, "output_interval": 200,
            "device": "cpu",
        },
        "parameter_hints": {
            "re": "Reynolds number Re = u_in·2r/ν. Re < 300 for steady wake.",
            "radius": "Sphere radius in lattice units. Keep > 6 for accuracy.",
        },
        "references": [
            "Johnson & Patel (1999) J. Fluid Mech.",
            "Tomboulides & Orszag (2000) J. Fluid Mech.",
        ],
    },
    {
        "id": "thermal_cavity_3d",
        "category": "internal_flow",
        "title": "3-D Differentially Heated Cavity",
        "title_zh": "三维差分加热方腔",
        "icon": "bi-thermometer-half",
        "description": (
            "3-D differentially heated square cavity benchmark (hot/cold vertical walls). "
            "Couples D3Q19 velocity with D3Q7 temperature solver via Boussinesq. "
            "Validated Nusselt numbers against De Vahl Davis (1983) data."
        ),
        "description_zh": (
            "三维差分加热方腔对流基准（冷热垂直壁）。"
            "D3Q19 流速 + D3Q7 温度 Boussinesq 耦合。"
            "与 De Vahl Davis（1983）Nusselt 数据对标。"
        ),
        "difficulty": "intermediate",
        "solver_type": "thermal_cavity_3d",
        "default_config": {
            "nx": 32, "ny": 32, "nz": 32,
            "ra": 1e4, "pr": 0.71,
            "n_steps": 500, "device": "cpu",
        },
        "parameter_hints": {
            "ra": "Rayleigh number Ra = gβΔT·L³/(να). 10³–10⁶ range well-validated.",
            "pr": "Prandtl number. Air ≈ 0.71, water ≈ 7.",
        },
        "references": ["De Vahl Davis (1983) Int. J. Num. Methods Fluids"],
    },
    {
        "id": "hull_free_surface",
        "category": "ocean_engineering",
        "title": "Hull Free-Surface Wave Resistance",
        "title_zh": "船体自由液面波浪阻力",
        "icon": "bi-water",
        "description": (
            "3-D two-phase Color-Gradient LBM simulation of a ship hull (Wigley / "
            "Series-60 / KCS) moving through calm water. Reports wave-making "
            "resistance (drag) and free-surface elevation."
        ),
        "description_zh": (
            "基于彩色梯度双相 LBM 的三维船体（Wigley / Series-60 / KCS）"
            "在静水中的波浪阻力仿真。输出兴波阻力（Cd）和自由液面高度。"
        ),
        "difficulty": "advanced",
        "solver_type": "hull_free_surface",
        "default_config": {
            "nx": 80, "ny": 32, "nz": 32,
            "hull_type": "wigley", "fill_fraction": 0.5,
            "re": 100.0, "u_in": 0.05,
            "n_steps": 200, "output_interval": 50,
            "device": "cpu",
        },
        "parameter_hints": {
            "hull_type": "'wigley' (parabolic), 'series60' (transom), or 'kcs' (container ship).",
            "fill_fraction": "Fraction of nz cells filled with water (undisturbed).",
        },
        "references": [
            "Hochbaum & Vogt (2002) Proc. 24th ONR Symp.",
            "Kim et al. (2001) J. Ship Res.",
        ],
    },
]

# Build lookup dict
_TEMPLATE_MAP: dict[str, dict] = {t["id"]: t for t in _TEMPLATES}

# Available category labels
_CATEGORIES: dict[str, str] = {
    "external_flow": "External Flow / Aerodynamics & Hydrodynamics",
    "internal_flow": "Internal Flow / Duct & Cavity",
    "multiphase": "Multiphase / Free Surface",
    "ocean_engineering": "Ocean & Marine Engineering",
}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/")
async def list_templates(category: str | None = None) -> dict:
    """List all engineering simulation templates, optionally filtered by category."""
    templates = _TEMPLATES
    if category:
        templates = [t for t in templates if t["category"] == category]
    return {
        "categories": _CATEGORIES,
        "templates": templates,
        "total": len(templates),
    }


@router.get("/categories")
async def list_categories() -> dict:
    """Return the available scenario categories."""
    counts = {}
    for t in _TEMPLATES:
        counts[t["category"]] = counts.get(t["category"], 0) + 1
    return {
        "categories": [
            {"id": k, "label": v, "count": counts.get(k, 0)}
            for k, v in _CATEGORIES.items()
        ]
    }


@router.get("/{template_id}")
async def get_template(template_id: str) -> dict:
    """Return full detail for a single template."""
    tmpl = _TEMPLATE_MAP.get(template_id)
    if tmpl is None:
        raise HTTPException(status_code=404, detail=f"Template '{template_id}' not found")
    return tmpl
