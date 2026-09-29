# TensorLBM 平台现状分析与融合方案

## 一、当前平台现状

### 1.1 后端 (FastAPI, :8004)
- **23个router, 239个API端点**
- **solver.py**: 20个case-specific端点 (cylinder-flow, sphere-flow, ship-hull等)
  - 每个case有独立的Pydantic参数模型和调用代码
  - 不使用共性接口
- **simulations.py**: XFlow-style通用接口 (GeneralSimEngine)
  - 有general_sim.py框架, 但不使用9个共性模块
  - 自定义voxelize/BC/loop/force, 非共性
- **orchestration.py**: 工作流编排
- **postprocess.py**: 后处理 (场数据/力/St)
- **cad.py**: CAD几何导入

### 1.2 前端
- **Vue.js (船舶平台)**: 独立前端, 在ship-performance-platform
- **HTML/JS (TensorLBM)**: app/frontend/ 14个JS文件
  - app_core, app_solver, app_cad, app_postprocess, app_projects等
  - case-specific UI, 非通用

### 1.3 src/tensorlbm/ (200+模块)
- **共性模块 (9个, 已验证)**:
  1. drag_pressure.py — 法向(8种)+压力积分+摩擦积分
  2. stl_geometry.py — STL几何+体素化+法向
  3. lbm_step_correct.py — 主循环(NoDynamics+BB+streaming+BC)
  4. boundaries3d.py — BB(f_pre)+far_field_bc
  5. bfl_d3q19.py — BFL插值反弹
  6. wall_model.py — 壁面函数(对数律/梯度律)
  7. momentum_exchange.py — MEM(5种变体)
  8. postprocess.py — St检测(Hanning+带通+自相关)
  9. force_methods.py — 5种力方法
- **通用框架**: general_sim.py (存在但不用共性模块!)
- **专用模块**: cylinder_flow, sphere_flow, ship_flow, suboff_*等

### 1.4 nginx (:8000)
- /api/v1/tensorlbm/ → :8001 (船舶平台)
- /api/ → :8004 (TensorLBM, 239端点)
- /tensorlbm-app/ → :8004

## 二、核心问题

### 2.1 共性模块未融入平台
- 9个共性模块已验证(45Bug修复, 30+benchmark)
- 但平台API不调用它们!
- general_sim.py有自己的voxelize/BC/loop/force
- solver.py每个case有独立代码

### 2.2 无通用求解器
- solver.py是case-specific (20个端点)
- 每个case: 独立参数+独立几何+独立力计算
- 不能"导入任意几何→运行→出力"

### 2.3 前端非通用
- 每个case有独立UI页面
- 无通用几何上传+参数设置+运行+结果

## 三、对标PowerFLOW/xFlow

### 3.1 PowerFLOW工作流
1. 导入CAD (STL/STEP)
2. 自动网格生成 (体素化)
3. 设置BC (入口速度/出口压力/壁面)
4. 运行 (LBM)
5. 后处理 (Cd/Cl/场可视化)

### 3.2 xFlow工作流
1. 导入几何 (STL)
2. 设置参数 (Re/速度/粘度)
3. 运行 (LBM, 实时可视化)
4. 力/场/涡可视化

### 3.3 共同特点
- **任意几何**: STL导入, 无case-specific代码
- **自动网格**: 体素化, 自动域大小
- **通用BC**: far_field/wall/periodic
- **通用力**: 压力+摩擦积分
- **实时可视化**: 场数据流式传输

## 四、融合方案

### 4.1 重写general_sim.py用共性模块
```python
# 通用流程 (所有case共用)
from tensorlbm.drag_pressure import get_near_wall_3d, SurfaceMesh, drag_pressure_integration, drag_friction_integration
from tensorlbm.boundaries3d import bounce_back_cells_3d, far_field_bc_3d
from tensorlbm.lbm_step_correct import lbm_step_correct
from tensorlbm.postprocess import detect_strouhal
from tensorlbm.stl_geometry import read_stl, voxelize_stl, SurfaceMesh_from_stl
from tensorlbm.wall_model import wall_function_3d

def generic_run(geometry, physics, solver_config):
    # 1. 几何
    if geometry.source == 'stl':
        vertices, faces, normals = read_stl(geometry.path)
        solid = voxelize_stl(vertices, faces, grid_shape)
    else:
        solid = build_parametric(geometry)  # cylinder/sphere/suboff/naca
    
    # 2. 近壁+法向
    near = get_near_wall_3d(solid)
    mesh = SurfaceMesh.from_xxx(solid, near, ...)  # auto-select
    
    # 3. 主循环
    for step in range(n_steps):
        f = lbm_step_correct(f, solid, collide_fn, tau, ...)
        if step > warmup:
            Cd_p = drag_pressure_integration(f, mesh, dpS)
            Cd_f = drag_friction_integration(f, mesh, dpS, nu)
    
    # 4. 后处理
    St = detect_strouhal(cl_hist, dt, D, u_in)
    return Cd_p, Cd_f, Cd_tot, Cl, St, field_data
```

### 4.2 新增API端点
```
POST /api/simulations/generic-run
  Input: {
    geometry: {source: 'stl', path: '/path/to/file.stl'} 或 {source: 'parametric', shape: 'cylinder', D: 48},
    physics: {Re: 200, u_in: 0.08, density: 1.0},
    solver: {collision: 'mrt_smag', Cs: 0.05, steps: 5000, warmup: 1000},
    output: {fields: ['velocity', 'pressure'], forces: true, strouhal: true}
  }
  Output: {job_id, status, Cd_p, Cd_f, Cd_tot, Cl, St, field_url}
```

### 4.3 自动参数选择 (like PowerFLOW)
- 自动域大小: 几何包围盒×3 (阻塞比<10%)
- 自动碰撞模型: Re<1000 MRT, Re>1000 MRT+Smag
- 自动壁面处理: Re<10000 BB, Re>10000 WF
- 自动warmup: 域大小²×0.5
- 自动法向选择: 解析几何用from_xxx, STL用from_gradient

### 4.4 前端改造
- 通用页面: 几何上传(STL) + 参数设置 + 运行 + 结果
- 实时Cd/Cl/St曲线
- 场可视化 (速度/压力/涡量)
- 力分解 (压力+摩擦)

## 五、实施步骤

### Phase 1: 重写general_sim.py (1天)
- 用9个共性模块替换自定义代码
- 通用流程: solid→near→mesh→lbm_step→drag→St
- 支持STL+参数化几何

### Phase 2: 新增API端点 (1天)
- POST /api/simulations/generic-run
- 自动参数选择
- 实时力监控 (WebSocket)

### Phase 3: 前端通用页面 (2天)
- STL上传+预览
- 参数设置面板
- 运行+实时结果
- 场可视化

### Phase 4: 验证 (1天)
- 用共性接口benchmark验证
- 圆柱/球/SUBOFF/Wigley通过generic-run运行
- 对比case-specific结果

## 六、预期效果

| 对比项 | 当前 | 融合后 |
|--------|------|--------|
| 几何支持 | case-specific | 任意STL |
| API端点 | 20个case | 1个generic |
| 代码复用 | 低(每case独立) | 高(共性模块) |
| 新case开发 | 1-2天 | 0(自动) |
| 力计算 | case-specific | P+F共性 |
| 壁面处理 | case-specific | BB/WF自动 |
| 对标 | 无 | PowerFLOW/xFlow |
