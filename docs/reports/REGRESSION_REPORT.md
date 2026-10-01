# 6DOF/FSI 回归等价性验证报告

**Worktree**: `regress-sixdof-fsi-r1`
**Base commit**: `cfb26c670b7a5778e219c60499526523593a29e7`
**Date**: 2026-07-17
**Status**: ✅ ALL 58 TESTS PASS (29 existing + 29 new regression)

---

## 1. 原始Bug辨识（带病上岗检测）

### BUG-1: `step_sixdof` 旋转DOF约束不一致

**文件**: `src/tensorlbm/sixdof.py`, line 261

**描述**: 平移DOF约束只零化力（保留已有速度），但旋转DOF约束在零化加速度后，
又对整个 `omega_new` 重新施加约束掩码，导致已有角速度被清零。

**代码**:
```python
# 平移约束 (line 238): 只约束力，速度保留
F_total = F_total * constraints_lin
vel_new = vel + (F_total / m) * dt   # 约束DOF: vel + 0 = vel (保留)

# 旋转约束 (line 258-261): 约束加速度后又约束速度
alpha_body = alpha_body * constraints_rot
omega_new = omega_body + alpha_body * dt  # 约束DOF: omega + 0 = omega (应保留)
omega_new = omega_new * constraints_rot   # BUG: 重新清零约束DOF的角速度
```

**影响**: 当 `fix_roll=True` 且初始 `omega_x=0.5` 时，`omega_x` 被错误清零为0.0，
而平移约束 `fix_surge=True` 时 `vel_x=0.5` 被正确保留。

**测试**: `TestOriginalBugRotationalDOFConstraint` (3 tests, all PASS)
- `test_translational_constraint_preserves_velocity`: 平移约束保留速度 ✓
- `test_rotational_constraint_zeroes_existing_omega`: 旋转约束清零角速度 (BUG) ✓
- `test_rotational_constraint_inconsistency_documented`: 不一致性已记录 ✓

**结论**: 原始 `step_sixdof` 带病上岗。`sixdof_common.rigid_body_step` 直接委托
`step_sixdof`，因此继承了此bug。共性模块未修复此bug，但等价性验证确认两者行为一致。

---

### BUG-2: `cummins_step` 文档声称RK4但实现是Euler

**文件**: `src/tensorlbm/rigid_body_6dof.py`, line 235 (docstring) vs lines 276-277

**描述**: docstring声称"4th-order Runge-Kutta"，但实际代码使用Symplectic Euler:
```python
# docstring (line 235): "Advance the Cummins equation by one time step (4th-order Runge-Kutta)"
# 实际代码 (line 275-277):
#   # Simple Euler integration (RK4 can be added for higher accuracy)
new_velocity = state.velocity + accel * dt
new_position = state.position + new_velocity * dt
```

**测试**: `TestOriginalBugCumminsDocstringVsImplementation` (1 test, PASS)
- `test_cummins_uses_euler_not_rk4`: 通过误差收敛率验证使用Euler (O(dt))而非RK4 (O(dt⁴))
  - dt=0.01, 0.005, 0.0025 的误差比 ≈ 2 (Euler特征)，而非 ≈ 16 (RK4特征)

**结论**: 文档与实现不一致。`cummins` 积分器在capability contract中标记为
`VERIFICATION_IMPLEMENTED_ONLY`（无contract test），且不通过 `rigid_body_step` 暴露。

---

### BUG-3: `cummins_time_integration` 卷积索引off-by-one

**文件**: `src/tensorlbm/rigid_body_6dof.py`, line 339

**描述**: Cummins方程的卷积积分需要 K(t_n - t_k) 对 k=0..n-1，
即 K_retard[n-k]。代码使用 `K_retard[step - N_hist : step].flip(0)`，
给出 K_retard[n-1-k] 而非 K_retard[n-k]。

**影响**:
- 最近速度被 K(0) 加权（过大），而非 K(dt)
- 最老速度被 K((n-1)*dt) 加权，而非 K(n*dt)
- 历史截断后索引错误更严重

**测试**: `TestOriginalBugCumminsConvolutionIndexing` (1 test, PASS)
- `test_convolution_uses_k0_for_recent_velocity`: 验证卷积使用K(0)而非K(dt)，
  确认状态有限（bug不导致崩溃）

**结论**: Cummins solver的卷积索引有off-by-one错误。此模块不通过共性接口暴露。

---

### 架构差异: `fsi.py` vs `fsi_common.py`

**描述**: 原始 `fsi.py` 是2D载荷提取 + 线性化结构响应（Euler-Bernoulli梁），
不是IBM+6DOF组合。`fsi_common.py` 是3D IBM直接力 + 6DOF刚体推进。

**测试**: `TestOriginalFSIArchitectureDifference` (2 tests, PASS)
- `test_original_fsi_is_2d_load_extraction`: 原始FSI产出FSILoads/FSIResponse (挠度/应力)
- `test_fsi_common_is_3d_ibm_plus_6dof`: 共性FSI产出FSIResult (刚体状态/f修正)

**结论**: 两者架构不同，无法数值等价比较。`fsi_common.py` 是新的组合实现，
不是 `fsi.py` 的抽取。

---

## 2. 等价性验证

### `sixdof_common.rigid_body_step` vs 原始 `step_sixdof`

**验证方法**: 相同输入 (state + force + dt) → 对比输出 allclose

**测试**: `TestEquivalenceRigidBodyStepVsStepSixdof` (4 tests, all PASS)

| 测试 | 场景 | 结果 |
|------|------|------|
| `test_same_inputs_produce_allclose_outputs` | 一般力+重力+非零omega | ✅ allclose(atol=1e-12) |
| `test_equivalence_with_nontrivial_quaternion` | 30°旋转四元数+角速度 | ✅ allclose(atol=1e-12) |
| `test_equivalence_with_dof_constraints` | fix_surge + fix_pitch | ✅ allclose(atol=1e-12) |
| `test_equivalence_multi_step` | 50步连续积分 | ✅ allclose(atol=1e-10) |

**结论**: `rigid_body_step` 是 `step_sixdof` 的薄包装（直接委托），
等价性由构造保证。测试验证了包括DOF约束在内的所有场景。

### 力强制转换一致性

**测试**: `TestEquivalenceForceCoercion` (2 tests, PASS)
- `(6,)` 张量 == `FluidForcesMoments` 相同值 → allclose(atol=1e-12) ✓
- `(3,)` 力 == `(6,)` 力(零力矩) → allclose(atol=1e-12) ✓

### 轨迹等价性

**测试**: `TestEquivalenceRunSimulationVsCommon::test_trajectory_equivalence` (PASS)
- `run_sixdof_simulation` (100步) vs 手动 `rigid_body_step` 循环 → 最终状态 allclose(atol=1e-10) ✓

---

## 3. FSI组合验证

### IBM力 → 6DOF更新 → f修正

**测试**: `TestFSICompositionIBMTo6DOF` (4 tests, all PASS)

| 测试 | 验证内容 | 结果 |
|------|----------|------|
| `test_fsi_body_advances_with_ibm_reaction_force` | 手动IBM→反力→6DOF == fsi_step | ✅ |
| `test_fsi_force_on_body_is_negative_of_fluid_force` | 牛顿第三定律: F_body = -ΣF_fluid | ✅ |
| `test_fsi_f_corrected_matches_ibm_output` | f_updated == ibm_direct_forcing的f_corrected | ✅ |
| `test_fsi_moment_resolution_about_centroid` | 力矩关于质心分解正确 | ✅ |

### 双向显式耦合

**测试**: `TestFSICompositionTwoWay` (2 tests, PASS)
- `test_two_way_reapplies_ibm_with_advanced_velocity`: 双向用推进后速度重新IBM → f_updated不同 ✓
- `test_two_way_force_recomputed_from_second_pass`: 力来自第二次IBM pass ✓

---

## 4. 组合测试: FSI + Collision完整循环

**测试**: `TestFSICollisionCombination` (4 tests, all PASS)

| 测试 | 场景 | 步数 | 结果 |
|------|------|------|------|
| `test_collision_fsi_loop_finite_and_consistent` | BGK碰撞→FSI(D3Q19) | 10 | ✅ 全有限, 物体移动 |
| `test_collision_fsi_loop_d3q27` | FSI(D3Q27) | 5 | ✅ 全有限 |
| `test_collision_fsi_loop_with_gravity` | 重力下落 | 10 | ✅ 物体下落(y<0) |
| `test_collision_fsi_two_way_loop` | 双向耦合 | 5 | ✅ 全有限 |

---

## 5. 边缘情况

**测试**: `TestEdgeCases` (5 tests, all PASS)
- dtype保持 (float64) ✓
- 输入不被修改 ✓
- 空mask → 零力 ✓
- 显式markers ✓
- Euler角提取 (90° z旋转 → yaw=π/2) ✓

---

## 总结

| 类别 | 测试数 | 通过 | 失败 |
|------|--------|------|------|
| 原始Bug辨识 | 7 | 7 | 0 |
| 等价性验证 | 7 | 7 | 0 |
| FSI组合验证 | 6 | 6 | 0 |
| 组合测试 | 4 | 4 | 0 |
| 边缘情况 | 5 | 5 | 0 |
| **总计(新)** | **29** | **29** | **0** |
| 现有测试 | 29 | 29 | 0 |
| **总计(全部)** | **58** | **58** | **0** |

### 辨识的原始Bug清单

1. **BUG-1** (sixdof.py): 旋转DOF约束清零已有角速度（平移约束保留速度，不一致）
2. **BUG-2** (rigid_body_6dof.py): cummins_step docstring声称RK4但实现是Euler
3. **BUG-3** (rigid_body_6dof.py): cummins卷积索引off-by-one (K_retard[n-1-k] vs K_retard[n-k])
4. **架构差异**: fsi.py(2D载荷提取) vs fsi_common.py(3D IBM+6DOF) — 非数值等价

### 等价性结论

- `rigid_body_step` ≡ `step_sixdof` (by construction, verified to atol=1e-12)
- `fsi_step` 正确组合 IBM → 反力 → 6DOF → f修正
- FSI + Collision循环在D3Q19/D3Q27、单向/双向、有/无重力下均产出有限结果
