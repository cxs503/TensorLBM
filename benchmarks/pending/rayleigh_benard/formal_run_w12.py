#!/usr/bin/env python3
"""
W12 正式运行脚本：γ_pred 重锁验证

使用格子算子特征值 γ_pred 作为参考，重算三级全臂相对误差。
基于 W11-C 的 formal_run.py，仅修改参考值来源。

输入：
  - /nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/ref/lattice_predictions.json
  - /nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/formal/results/f_ny{ny}_arm{frac}.json

输出：
  - /nfs/wangxi/runs/bm_close/rayleigh_benard/formal_w12/results/f_ny{ny}_arm{frac}_w12.json
"""

import json
import os
from pathlib import Path

# 路径配置
LATTICE_PREDICTIONS = "/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/ref/lattice_predictions.json"
W11_RESULTS_DIR = "/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/formal/results"
W12_RESULTS_DIR = "/nfs/wangxi/runs/bm_close/rayleigh_benard/formal_w12/results"

# 网格与臂配置
NYS = [64, 128, 256]
FRACS = [0.85, 0.95, 1.05, 1.15, 1.3]

def load_lattice_predictions():
    """加载格子算子特征值预测"""
    with open(LATTICE_PREDICTIONS, 'r') as f:
        data = json.load(f)
    
    # 构建查找表: {(ny, frac): gamma_pred}
    lookup = {}
    for level in data['levels']:
        ny = level['ny']
        for arm in level['arms']:
            frac = arm['frac']
            gamma_pred = arm['gamma_pred']
            lookup[(ny, frac)] = gamma_pred
    
    return lookup

def load_w11_result(ny, frac):
    """加载 W11-C 的模拟结果"""
    filename = f"f_ny{ny}_arm{frac}.json"
    filepath = os.path.join(W11_RESULTS_DIR, filename)
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return data

def compute_w12_result(ny, frac, gamma_pred, w11_data):
    """
    计算 W12 结果：使用 γ_pred 作为参考
    
    γ_sim 保持不变，仅替换 γ_ref
    """
    gamma_sim = w11_data['gamma_sim']
    health = w11_data.get('health', {})
    
    # 计算相对误差（百分比）
    rel_err_pct = (gamma_sim / gamma_pred - 1.0) * 100.0
    
    # 构建 W12 结果
    w12_data = {
        'ny': ny,
        'frac': frac,
        'gamma_sim': gamma_sim,
        'gamma_ref': gamma_pred,  # 使用 γ_pred 作为参考
        'rel_pct': rel_err_pct,
        'Ra': w11_data.get('Ra', None),
        'slope_spread_rel': w11_data.get('slope_spread_rel', None),
        'e_folds': health.get('e_folds', None),
        'mass_drift_rel': health.get('mass_drift_rel', None),
        'u_max': health.get('u_max', None),
        'theta_absmax': max(abs(health.get('theta_min', 0)), abs(health.get('theta_max', 0))),
        'wall_s': w11_data.get('wall_seconds', None),
        'steps': w11_data.get('steps', None),
        'reference_type': 'lattice_operator',  # 标记参考类型
    }
    
    return w12_data

def main():
    print("=" * 70)
    print("W12 γ_pred 重锁验证")
    print("=" * 70)
    
    # 加载格子算子预测
    print("\n加载格子算子特征值预测...")
    gamma_pred_lookup = load_lattice_predictions()
    print(f"  ✓ 加载 {len(gamma_pred_lookup)} 个 (ny, frac) 组合")
    
    # 创建输出目录
    os.makedirs(W12_RESULTS_DIR, exist_ok=True)
    print(f"\n输出目录: {W12_RESULTS_DIR}")
    
    # 处理所有组合
    print("\n处理三级全臂...")
    results_summary = []
    
    for ny in NYS:
        print(f"\n  ny={ny}:")
        for frac in FRACS:
            # 加载 W11 结果
            w11_data = load_w11_result(ny, frac)
            
            # 获取 γ_pred
            gamma_pred = gamma_pred_lookup[(ny, frac)]
            
            # 计算 W12 结果
            w12_data = compute_w12_result(ny, frac, gamma_pred, w11_data)
            
            # 保存结果
            output_filename = f"f_ny{ny}_arm{frac}_w12.json"
            output_filepath = os.path.join(W12_RESULTS_DIR, output_filename)
            with open(output_filepath, 'w') as f:
                json.dump(w12_data, f, indent=2)
            
            # 打印摘要
            rel_pct = w12_data['rel_pct']
            gamma_sim = w12_data['gamma_sim']
            status = "✓" if abs(rel_pct) <= 3.0 else "✗"
            print(f"    {status} frac={frac:.2f}: γ_sim={gamma_sim:+.6f}, "
                  f"γ_pred={gamma_pred:+.6f}, rel={rel_pct:+.4f}%")
            
            results_summary.append({
                'ny': ny,
                'frac': frac,
                'rel_pct': rel_pct,
                'pass': abs(rel_pct) <= 3.0
            })
    
    # 计算 E(ny)
    print("\n" + "=" * 70)
    print("收敛性分析")
    print("=" * 70)
    
    for ny in NYS:
        ny_results = [r for r in results_summary if r['ny'] == ny]
        rel_errs = [abs(r['rel_pct']) for r in ny_results]
        E_ny = (sum(e**2 for e in rel_errs) / len(rel_errs)) ** 0.5
        print(f"  E(ny={ny}) = {E_ny:.4f}%")
    
    # 检查单调性
    E_64 = (sum(abs(r['rel_pct'])**2 for r in results_summary if r['ny'] == 64) / 5) ** 0.5
    E_128 = (sum(abs(r['rel_pct'])**2 for r in results_summary if r['ny'] == 128) / 5) ** 0.5
    E_256 = (sum(abs(r['rel_pct'])**2 for r in results_summary if r['ny'] == 256) / 5) ** 0.5
    
    monotonic = E_64 > E_128 > E_256
    print(f"\n单调性检查: E(64)={E_64:.4f}% > E(128)={E_128:.4f}% > E(256)={E_256:.4f}%")
    print(f"  {'✓ 单调收敛' if monotonic else '✗ 非单调收敛'}")
    
    # 总结
    print("\n" + "=" * 70)
    print("W12 判决总结")
    print("=" * 70)
    
    pass_count = sum(1 for r in results_summary if r['pass'])
    total_count = len(results_summary)
    
    print(f"  通过门 G1 (rel≤3%): {pass_count}/{total_count} 臂")
    print(f"  通过门 G2 (单调): {'是' if monotonic else '否'}")
    print(f"  最终判决: {'PROMOTE' if (pass_count == total_count and monotonic) else 'TERMINAL'}")
    
    print("\n✓ W12 重锁验证完成")

if __name__ == "__main__":
    main()
