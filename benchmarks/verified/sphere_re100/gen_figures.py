"""Generate figures for sphere_re100 m475 re-verification report."""
import json
import os
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))

def load(path):
    with open(path) as f:
        return json.load(f)

def gen_cd_convergence():
    """Figure 1: Cd convergence history for T1 and T2."""
    t1 = load(os.path.join(BASE, "t1_d40big_12k.json"))
    t2 = load(os.path.join(BASE, "t2_d60big_24k.json"))
    
    cd_ref = 1.0917310910948732
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    steps1 = [h["step"] for h in t1["history"]]
    cd1 = [h["cd"] for h in t1["history"]]
    steps2 = [h["step"] for h in t2["history"]]
    cd2 = [h["cd"] for h in t2["history"]]
    
    ax.plot(steps1, cd1, 'b-', linewidth=1.5, label='T1: D=40 (280×280×320)', alpha=0.8)
    ax.plot(steps2, cd2, 'r-', linewidth=1.5, label='T2: D=60 (420×420×480)', alpha=0.8)
    ax.axhline(y=cd_ref, color='g', linestyle='--', linewidth=2, label=f'Reference (Schiller-Naumann): {cd_ref:.5f}')
    
    # 3% gate bands
    ax.axhline(y=cd_ref * 1.03, color='orange', linestyle=':', linewidth=1, alpha=0.6, label='±3% gate')
    ax.axhline(y=cd_ref * 0.97, color='orange', linestyle=':', linewidth=1, alpha=0.6)
    ax.fill_between([0, max(steps2)], cd_ref * 0.97, cd_ref * 1.03, alpha=0.08, color='orange')
    
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Drag Coefficient $C_d$', fontsize=12)
    ax.set_title('Sphere Re=100 — $C_d$ Convergence History (BFL Momentum Ledger)', fontsize=14)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(steps2))
    
    # Annotate final values
    ax.annotate(f'T1: {cd1[-1]:.5f}\n(+2.771%)', xy=(steps1[-1], cd1[-1]), 
                xytext=(steps1[-1]*0.6, cd1[-1]+0.015), fontsize=9,
                arrowprops=dict(arrowstyle='->', color='blue', alpha=0.6))
    ax.annotate(f'T2: {cd2[-1]:.5f}\n(+2.540%)', xy=(steps2[-1], cd2[-1]),
                xytext=(steps2[-1]*0.55, cd2[-1]+0.015), fontsize=9,
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.6))
    
    plt.tight_layout()
    path = os.path.join(BASE, "m475_cd_convergence.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    
    with open(path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    return path, b64

def gen_error_ladder():
    """Figure 2: Error ladder showing monotonic convergence."""
    cd_ref = 1.0917310910948732
    
    # Domain convergence data from result.json
    domain_points = {
        'up/D=1.25': 1.214927032369151,
        'up/D=2.00': 1.1553350990866116,
        'up/D=2.75': 1.1379502944831688,
        'up/D=3.00': 1.134709647988964,
        'up/D=4.00': 1.1278585129054965,
        'Big Domain\n(lat3/up3/down4)': 1.1219804015117896,
    }
    
    # Grid convergence
    grid_points = {
        'D=40\n(28722 links)': 1.1219804015117896,
        'D=60\n(64794 links)': 1.119462001144486,
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Domain convergence
    labels1 = list(domain_points.keys())
    vals1 = list(domain_points.values())
    errs1 = [(v - cd_ref) / cd_ref * 100 for v in vals1]
    
    colors1 = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(labels1)))
    bars1 = ax1.bar(range(len(labels1)), errs1, color=colors1, edgecolor='black', linewidth=0.5)
    ax1.axhline(y=3, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='3% gate')
    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.set_xticks(range(len(labels1)))
    ax1.set_xticklabels(labels1, fontsize=8, rotation=15)
    ax1.set_ylabel('Error vs Reference (%)', fontsize=11)
    ax1.set_title('Domain Convergence Signature\n(upstream distance lever)', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, err) in enumerate(zip(bars1, errs1)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{err:.2f}%', ha='center', fontsize=8, fontweight='bold')
    
    # Right: Grid convergence
    labels2 = list(grid_points.keys())
    vals2 = list(grid_points.values())
    errs2 = [(v - cd_ref) / cd_ref * 100 for v in vals2]
    
    colors2 = ['#4472C4', '#ED7D31']
    bars2 = ax2.bar(range(len(labels2)), errs2, color=colors2, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=3, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='3% gate')
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_xticks(range(len(labels2)))
    ax2.set_xticklabels(labels2, fontsize=9)
    ax2.set_ylabel('Error vs Reference (%)', fontsize=11)
    ax2.set_title('Grid Convergence Ladder\n(D=40 → D=60)', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 13)
    
    for i, (bar, err) in enumerate(zip(bars2, errs2)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{err:.2f}%', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    path = os.path.join(BASE, "m475_error_ladder.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    
    with open(path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    return path, b64

def gen_cv_closure():
    """Figure 3: Control volume closure comparison."""
    t1 = load(os.path.join(BASE, "t1_d40big_12k.json"))
    t2 = load(os.path.join(BASE, "t2_d60big_24k.json"))
    
    # Get CV data (last 50 samples have cv_mom_cd)
    cv_t1 = [(h["cd"], h["cv_mom_cd"]) for h in t1["history"] if "cv_mom_cd" in h]
    cv_t2 = [(h["cd"], h["cv_mom_cd"]) for h in t2["history"] if "cv_mom_cd" in h]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    if cv_t1:
        steps_t1 = [h["step"] for h in t1["history"] if "cv_mom_cd" in h]
        cd_t1 = [c[0] for c in cv_t1]
        cv_t1_vals = [c[1] for c in cv_t1]
        ax.plot(steps_t1, cd_t1, 'b-', linewidth=1.5, label='T1: BFL Ledger $C_d$', alpha=0.8)
        ax.plot(steps_t1, cv_t1_vals, 'b--', linewidth=1.2, label='T1: CV Balance $C_d$', alpha=0.6)
    
    if cv_t2:
        steps_t2 = [h["step"] for h in t2["history"] if "cv_mom_cd" in h]
        cd_t2 = [c[0] for c in cv_t2]
        cv_t2_vals = [c[1] for c in cv_t2]
        ax.plot(steps_t2, cd_t2, 'r-', linewidth=1.5, label='T2: BFL Ledger $C_d$', alpha=0.8)
        ax.plot(steps_t2, cv_t2_vals, 'r--', linewidth=1.2, label='T2: CV Balance $C_d$', alpha=0.6)
    
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('$C_d$', fontsize=12)
    ax.set_title('Control Volume Closure: BFL Ledger vs Independent CV Balance', fontsize=13)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(BASE, "m475_cv_closure.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    
    with open(path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    return path, b64

if __name__ == "__main__":
    p1, b1 = gen_cd_convergence()
    print(f"Figure 1: {p1}")
    p2, b2 = gen_error_ladder()
    print(f"Figure 2: {p2}")
    p3, b3 = gen_cv_closure()
    print(f"Figure 3: {p3}")
    print("All figures generated.")
