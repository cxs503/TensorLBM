原 80 行 → 压缩 86 行
# DE-A2 代码地图 · 液滴入水 single_drop
**仓库**: TensorLBM · **目录**: `benchmarks/water_entry/single_drop/`
**文件**: `free_surface_probe.py`(300行) · `run.py`(340行) · `score.py`(274行) · `silhouette.py`(44行) · `protocol.json`

## 1) 可调开关
### 1a. free_surface_probe.py — `probe()` (L32-45) / CLI (L237-257)
| 参数 | 含义 | 默认 | 定义行 |
|---|---|---|---|
| `n`(cells) | 每D格点数 | 必填 | L34/L253 |
| `speed` | 晶格速度 | 0.04 | L35/L254 |
| `limit`(steps) | 最大步数 | 10000 | L36/L255 |
| `gap_fraction`(gap) | 间隙/D | 0.4 | L37/L257 |
| `capacity` | 容量再分配 | False | L38/L242 |
| `nearest` | 最近面速度 | False | L39/L249 |
| `pilot` | 试点材料 | False | L40/L250 |
| `span` | 池宽/D | 4.0 | L41/L251 |
| `pool_depth` | 池深/D | 3.0 | L42/L252 |
| `simple` | 简单质量交换 | False | L43/L243 |
| `release` | 释放速度m/s | None | L44/L245 |
| `planes_path` | 中平面输出 | None | L45/L237 |
| `silhouette_path` | 侧视轮廓输出 | None | L46/L241 |

### 1b. run.py — `run()` (L99-113) / CLI (L284-301)
| 参数 | 含义 | 默认 | 定义行 |
|---|---|---|---|
| `n`(cells) | 每D格点数 | 必填 | L100/L287 |
| `speed` | 晶格速度 | 0.04 | L101/L288 |
| `steps` | 最大步数 | None→自动 | L102/L300 |
| `width` | 界面宽度 | 4.0 | L103/L289 |
| `gap` | 间隙/D | 0.4 | L104/L290 |
| `gas_flow` | 外部势流 | False | L105/L291 |
| `odd_rate` | 水动力奇偶率 | None | L106/L299 |
| `at_rest` | 静止诊断 | False | L107/L295 |
| `uniform_drop` | 均匀液滴速度 | False | L108/L293 |
| `stencil_wall` | 模板壁面 | False | L109/L292 |
| `pilot` | 试点材料 | False | L110/L284 |
| `span` | 池宽/D | 4.0 | L111/L285 |
| `pool_depth` | 池深/D | 3.0 | L112/L286 |

### 1c. 内部常量
- `stop_ms=24` (run.py L180) — 接触后观察窗
- `sample_targets_ms=(1.39643,6.4,10.39446,13.4)` (free_surface_probe.py L97)
- `frame_interval_ms=1.0` (silhouette.py L25)

## 2) 输入输出契约
### 2a. run.py `main()` → `--out` JSON (L301-336)
**写**: `solver,case_id,span_over_D,pool_depth_over_D,cells_per_D,lattice_speed,shape_zyx,interface_width_cells,gap_over_D,exterior_potential_flow,hydro_odd_rate,scales,stop_reason,contact_step,impact_speed_m_s,at_rest_control,uniform_drop,stencil_wall,history[]`
**history**: `step,time_ms,mach_max,rho_min,min_phi,volume_drift,phi_at_mach_max,mach_liquid_max,[depth_over_D],[detached_downward_speed_m_s]`
**退出码**: 0 仅当 `stop_reason=="window_complete"` (L336)

### 2b. free_surface_probe.py `main()` → `--out` JSON (L292-300)
**额外**: `release_speed_m_s,prescribed_release_speed_m_s,side_view_measurement,capacity_redistribution,nearest_surface_velocity,simple_mass_exchange`
**history**: `step,time_ms,volume_drift,geometric_volume_drift,represented_mass_drift,mach_max,rho_min,[depth_over_D]`
**退出码**: 0 仅当 `stop_reason=="window_complete"` **且** `geometric_volume_drift<=tol` (L300)

### 2c. score.py `assess()` → `--output` JSON (L268-274)
**读**: 每个 run JSON(或.gz,L168-172)；校验 history 步序连续 (L31-33)
**判**:
- 健康过滤 (L38-52): `rho_min>0`,`0<=mach_max<=0.3`,`0<=volume_drift<=1e-4`,接触后`depth_over_D>=0`
- 插值 (L55-63): 对每个实验点 `t0` 在 `[t[0],t[-1]]` 内 `np.interp`
- `numeric_pass` (L120-128): 体积漂移≤1e-4+全覆盖+非at_rest+`camera_error<=0.03`+`max_error<=0.03`+`peak_error<=0.03`
- 收敛 (L212-235): 组内≥2网格(L215),严格递减误差(L228-233),末格`numeric_pass`
- `status` (L249-254): `eligible_for_review` 需 `converged`+`resolved`+无`unresolved_times`+`domain_and_material_match_confirmed`

## 3) 运行健康检查（终止条件）
| 条件 | 阈值 | 位置 |
|---|---|---|
| 非有限值 | 任意row/f,g非有限 | fsp L123; run L219 |
| `mach_max` | >0.3 | fsp L126; run L223 |
| `rho_min` | ≤0 | fsp L127; run L222 |
| `volume_drift` | >1e-4 | fsp L128; run L224 |
| 液滴丢失 | `DropTracker.update`抛ValueError | fsp L133-135; run L238-239 |
| 拓扑事务 | `TopologyTransactionError` | fsp L191 |
| 接触后窗口 | `(step-contact)*dt>=24ms`→`window_complete` | fsp L172; run L248-250 |
| 步数上限 | `step>limit`→`step_limit` | 默认 L103/L192 |

**CLI校验** (fsp L260-269): `span<3`,`pool_depth<2`,`cells<8`,`speed∉(0,0.15)`,`gap∉(0,0.8]`,`release_speed∉(0,2*1.4)`,`steps<0`
**CLI校验** (run L302-309): `span<3`,`pool_depth<2`,`cells<8`,`width<=0`,`gap∉[0.1,0.6]`,`speed∉(0,0.15)`,`steps<0`

## 4) 路径入口
| 路径 | 入口文件 | 入口行 | 求解器 |
|---|---|---|---|
| **双相(保守Allen-Cahn)** | `run.py` `run()` | L99 | `CACSim/D3Q19`(L129-131),`sim.step()`L257 |
| **自由面** | `free_surface_probe.py` `probe()` | L32 | `free_surface_step/D3Q19`(L183-189),常压 |

**score.py 区分**: `free_surface=run.get("solver","").startswith("free_surface_step")`(L103)—决定用`geometric_volume_drift`还是`volume_drift`(L104-111)。
