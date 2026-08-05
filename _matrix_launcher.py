"""Agent Matrix launcher — 8 parallel configs on SDAA 0-7.

Matrix:
  FLOW: Attached (SUBOFF 200³, Re=2e6, 1500 steps):
    1. D3Q19 MRT+Smag log-law       (SDAA:0) — baseline
    2. D3Q19 MRT+Smag gradient       (SDAA:1) — faster, check accuracy
    3. D3Q27 KBC+Smag log-law        (SDAA:2) — fastest D3Q27 variant

  FLOW: Bluff cylinder (D=24, 200×80×4, Re=200, 1500 steps):
    4. D3Q27 CASCADED log-law        (SDAA:3) — proven 2.93% breakthrough
    5. D3Q19 MRT+Smag log-law        (SDAA:4) — compare MRT vs CASCADED
    6. D3Q27 CUMULANT log-law        (SDAA:5) — compare with CASCADED

  FLOW: Bluff sphere (D=40, 200×100×100, Re=1000, 1500 steps):
    7. D3Q19 MRT+Smag log-law        (SDAA:6)
    8. D3Q27 CASCADED log-law        (SDAA:7)

ALL: farfield BC, sign-fixed code, sliding window=300.
"""
import json, os, subprocess, sys, time
from pathlib import Path


# ── Helper ───────────────────────────────────────────────────────
def _print_row(r):
    if "error" in r:
        print(f"C{r.get('config_id','?'):<3} {'ERROR':<7} {r.get('name','?')} {r['error']}")
        return
    cid  = r.get("config_id", "?")
    lt   = r.get("lattice", "?")
    col  = r.get("collision", "?")
    wl   = r.get("wall_law", "?")
    cs   = r.get("Cs", 0)
    cf   = r.get("Ct_fric_full", 0)
    cp   = r.get("Ct_pres_full", 0)
    ct   = r.get("Ct_total_full", 0)
    cts  = r.get("Ct_total_slide", 0)
    err  = r.get("error_pct", float("nan"))
    ok   = "✓" if r.get("finite") else "✗"
    et   = r.get("elapsed_s", 0)
    print(f"{cid:<4} {lt:<7} {col:<10} {wl:<9} {cs:<5.2f} {cf:<10.5f} {cp:<10.5f} {ct:<10.5f} {cts:<10.5f} "
          f"{err:<8.1f} {ok:<4} {et:<8.0f}s")


# ── Main ─────────────────────────────────────────────────────────

WORKER     = Path(__file__).parent / "_matrix_worker.py"
OUT_DIR    = Path("/tmp/matrix_logs")
RESULTS_JSON = Path("/tmp/matrix_results.json")

# Clean
OUT_DIR.mkdir(exist_ok=True)
for f in OUT_DIR.glob("*.json"):
    f.unlink()
for f in OUT_DIR.glob("*.log"):
    f.unlink()
RESULTS_JSON.unlink(missing_ok=True)

env = os.environ.copy()
env["PYTHONPATH"] = str(Path(__file__).parent / "src")

# ── 8-CONFIG MATRIX ──────────────────────────────────────────────
# (did, flow, lattice, collision, wall_law, Cs, nx, ny, nz, geom_param, n_steps, out_path)
CONFIGS = [
    # ── SUBOFF (Attached flow) ──
    (0, "suboff",   "D3Q19", "MRT",       "log",      0.05,  200, 200, 200, 200.0, 1500, "/tmp/matrix_logs/c0_suboff_q19_mrt_log.json"),
    (1, "suboff",   "D3Q19", "MRT",       "gradient", 0.05,  200, 200, 200, 200.0, 1500, "/tmp/matrix_logs/c1_suboff_q19_mrt_gradient.json"),
    (2, "suboff",   "D3Q27", "KBC",       "log",      0.05,  200, 200, 200, 200.0, 1500, "/tmp/matrix_logs/c2_suboff_q27_kbc_log.json"),
    # ── Cylinder (Bluff body, Re=200) ──
    (3, "cylinder", "D3Q27", "CASCADED",  "log",      0.0,   200, 80,  4,   24.0,  1500, "/tmp/matrix_logs/c3_cylinder_q27_cascaded_log.json"),
    (4, "cylinder", "D3Q19", "MRT",       "log",      0.05,  200, 80,  4,   24.0,  1500, "/tmp/matrix_logs/c4_cylinder_q19_mrt_log.json"),
    (5, "cylinder", "D3Q27", "CUMULANT",  "log",      0.0,   200, 80,  4,   24.0,  1500, "/tmp/matrix_logs/c5_cylinder_q27_cumulant_log.json"),
    # ── Sphere (Bluff body, Re=1000, D=40) ──
    (6, "sphere",   "D3Q19", "MRT",       "log",      0.05,  200, 100, 100, 40.0,  1500, "/tmp/matrix_logs/c6_sphere_q19_mrt_log.json"),
    (7, "sphere",   "D3Q27", "CASCADED",  "log",      0.0,   200, 100, 100, 40.0,  1500, "/tmp/matrix_logs/c7_sphere_q27_cascaded_log.json"),
]

print("=" * 110)
print("AGENT MATRIX — 8 configs (collision × wall-law × geometry)")
print("=" * 110)
print(f"SUBOFF: 200³ Re=2e6    | Cylinder: 200×80×4 D=24 Re=200 | Sphere: 200×100×100 D=40 Re=1000")
print(f"All: farfield BC, sliding window=300, 1500 steps each")
print(f"SDAA cards: 0-7")
print()

# ── Launch workers ───────────────────────────────────────────────
procs = []
for (did, flow, lattice, collision, wall_law, cs, nx, ny, nz, geom, ns, out_path) in CONFIGS:
    smag = f" Cs={cs}" if cs > 0 else ""
    name = f"[C{did}] {flow} {lattice} {collision}{smag} {wall_law}-law"
    log = OUT_DIR / f"worker_{did:02d}.log"
    cmd = [
        sys.executable, str(WORKER),
        str(did), flow, lattice, collision, wall_law,
        str(cs), str(nx), str(ny), str(nz), str(geom), str(ns), out_path,
    ]
    with open(log, "w") as lf:
        p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    procs.append((did, name, p, out_path, log))
    print(f"  {name} (PID={p.pid})", flush=True)

print(f"\nAll {len(procs)} workers launched. Logs: {OUT_DIR}", flush=True)
print()

# ── Wait for completion ──────────────────────────────────────────
t0 = time.time()
while True:
    time.sleep(30)
    done = sum(1 for _, _, p, _, _ in procs if p.poll() is not None)
    n = len(procs)
    print(f"[{time.time()-t0:.0f}s] {done}/{n} workers done", flush=True)
    if done == n:
        break

print(f"\nAll workers finished in {time.time()-t0:.0f}s\n", flush=True)

# ── Collect results ──────────────────────────────────────────────
results = []
for did, name, p, out_path, log_path in procs:
    rf = Path(out_path)
    if rf.exists():
        try:
            r = json.loads(rf.read_text())
            results.append(r)
        except Exception as e:
            results.append({"config_id": did, "name": name, "error": str(e)})
    else:
        tail = ""
        if log_path.exists():
            try:
                lines = log_path.read_text().strip().split("\n")
                tail = "\n".join(lines[-5:])
            except Exception:
                pass
        results.append({"config_id": did, "name": name, "error": "no result file", "log_tail": tail})

# ── PRINT RESULTS TABLE ──────────────────────────────────────────
print("=" * 125)
print("AGENT MATRIX — RESULTS SUMMARY")
print("=" * 125)

# Separators
print()
print("─── SUBOFF (Attached flow, 200³ Re=2e6, Ref Ct=0.00405) ───")
header = f"{'Cfg':<4} {'Lattice':<7} {'Collision':<10} {'Wall':<9} {'Cs':<5} {'Ct_fric':<10} {'Ct_pres':<10} {'Ct_total':<10} {'Ct_slide':<10} {'Err%':<8} {'OK':<4} {'Time':<8}"
print(header)
print("-" * 125)

for r in results:
    if r.get("flow") != "suboff":
        continue
    _print_row(r)

print()
print("─── Cylinder (Bluff, 200×80×4 D=24 Re=200, Ref Cd=1.30) ───")
print(header.replace("Ct_", "Cd_").replace("Ref Ct", "Ref Cd"))
print("-" * 125)

for r in results:
    if r.get("flow") != "cylinder":
        continue
    _print_row(r)

print()
print("─── Sphere (Bluff, 200×100×100 D=40 Re=1000, Ref Cd=0.47) ───")
print(header.replace("Ct_", "Cd_").replace("Ref Ct", "Ref Cd"))
print("-" * 125)

for r in results:
    if r.get("flow") != "sphere":
        continue
    _print_row(r)

# ── KEY QUESTIONS ────────────────────────────────────────────────
print()
print("=" * 125)
print("KEY QUESTIONS")
print("=" * 125)

suboff_base = next((r for r in results if r.get("config_id") == 0 and "error" not in r), None)
suboff_grad = next((r for r in results if r.get("config_id") == 1 and "error" not in r), None)
suboff_kbc  = next((r for r in results if r.get("config_id") == 2 and "error" not in r), None)
cyl_casc    = next((r for r in results if r.get("config_id") == 3 and "error" not in r), None)
cyl_mrt     = next((r for r in results if r.get("config_id") == 4 and "error" not in r), None)
cyl_cum     = next((r for r in results if r.get("config_id") == 5 and "error" not in r), None)
sph_mrt     = next((r for r in results if r.get("config_id") == 6 and "error" not in r), None)
sph_casc    = next((r for r in results if r.get("config_id") == 7 and "error" not in r), None)

# Q1: Does gradient wall-law match log-law accuracy?
if suboff_base and suboff_grad:
    d = abs(suboff_base["Ct_total_slide"] - suboff_grad["Ct_total_slide"])
    print(f"\nQ1: Gradient vs log-law on SUBOFF:")
    print(f"    log-law: Ct={suboff_base['Ct_total_slide']:.5f}, gradient: Ct={suboff_grad['Ct_total_slide']:.5f}")
    print(f"    ΔCt={d:.5f} — {'✓ Matches' if d < 0.001 else '✗ Diverges significantly'}")

# Q2: Does KBC beat MRT on SUBOFF?
if suboff_base and suboff_kbc:
    print(f"\nQ2: KBC vs MRT on SUBOFF:")
    print(f"    MRT+Smag: Ct={suboff_base['Ct_total_slide']:.5f} err={suboff_base['error_pct']:.1f}%")
    print(f"    KBC+Smag: Ct={suboff_kbc['Ct_total_slide']:.5f} err={suboff_kbc['error_pct']:.1f}%")
    if suboff_kbc.get("finite"):
        print(f"    → KBC is {abs(suboff_kbc['error_pct']):.1f}% from ref — "
              f"{'✓ Better' if abs(suboff_kbc['error_pct']) < abs(suboff_base['error_pct']) else '✗ Worse'} than MRT")

# Q3: Does CASCADED beat MRT on cylinder?
if cyl_casc and cyl_mrt:
    print(f"\nQ3: CASCADED vs MRT on cylinder:")
    print(f"    CASCADED: Cd={cyl_casc['Ct_total_slide']:.5f} err={cyl_casc['error_pct']:.1f}%")
    print(f"    MRT+Smag: Cd={cyl_mrt['Ct_total_slide']:.5f} err={cyl_mrt['error_pct']:.1f}%")

# Q4: Does CUMULANT still fail on cylinder?
if cyl_cum:
    print(f"\nQ4: CUMULANT on cylinder:")
    print(f"    CUMULANT: Cd={cyl_cum['Ct_total_slide']:.5f} err={cyl_cum['error_pct']:.1f}%")
    print(f"    Finite: {cyl_cum.get('finite')}")

# Q5: Can either operator handle sphere at D=40?
if sph_mrt and sph_casc:
    print(f"\nQ5: Sphere D=40 at Re=1000:")
    print(f"    MRT+Smag:  Cd={sph_mrt['Ct_total_slide']:.5f} err={sph_mrt['error_pct']:.1f}% OK={sph_mrt.get('finite')}")
    print(f"    CASCADED:  Cd={sph_casc['Ct_total_slide']:.5f} err={sph_casc['error_pct']:.1f}% OK={sph_casc.get('finite')}")

# ── FIND THE RULE ────────────────────────────────────────────────
print()
print("=" * 125)
print("INFERRED RULES")
print("=" * 125)

rules = []
if suboff_base:
    rules.append("SUBOFF: D3Q19 MRT+Smag log-law is the baseline (stable, well-validated)")
if suboff_grad and suboff_base:
    d = abs(suboff_grad.get("Ct_total_slide", 0) - suboff_base.get("Ct_total_slide", 0))
    rules.append(f"SUBOFF: gradient wall-law {'matches' if d < 0.001 else 'differs from'} log-law (ΔCt={d:.5f})")
if suboff_kbc and suboff_base:
    r = f"SUBOFF: KBC+Smag {'beats' if abs(suboff_kbc.get('error_pct', 99)) < abs(suboff_base.get('error_pct', 99)) else 'underperforms vs'} MRT+Smag"
    rules.append(r)
if cyl_casc and cyl_mrt:
    r = f"Cylinder: CASCADED {'beats' if abs(cyl_casc.get('error_pct', 99)) < abs(cyl_mrt.get('error_pct', 99)) else 'vs'} MRT"
    rules.append(r)
if cyl_cum:
    rules.append(f"Cylinder: CUMULANT {'stable' if cyl_cum.get('finite') else 'DIVERGED/FAILED'}")
if sph_mrt:
    rules.append(f"Sphere: MRT+Smag {'stable' if sph_mrt.get('finite') else 'DIVERGED'}")
if sph_casc:
    rules.append(f"Sphere: CASCADED {'stable' if sph_casc.get('finite') else 'DIVERGED'}")

for i, r in enumerate(rules, 1):
    print(f"  {i}. {r}")

# ── Save combined results ────────────────────────────────────────
combined = {
    "title": "Agent Matrix — 8-config collision×wall-law×geometry",
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "setup": {
        "suboff": {"grid": "200×200×200", "Re": 2e6, "ref_Ct": 0.00405, "hull_length": 200},
        "cylinder": {"grid": "200×80×4", "Re": 200, "ref_Cd": 1.30, "D": 24},
        "sphere": {"grid": "200×100×100", "Re": 1000, "ref_Cd": 0.47, "D": 40},
        "common": {"n_steps": 1500, "sliding_window": 300, "farfield_bc": True, "sign_fixed": True},
    },
    "key_questions": [
        "Does gradient wall-law match log-law accuracy on SUBOFF?",
        "Does KBC beat MRT on SUBOFF?",
        "Does CASCADED still beat MRT on cylinder?",
        "Does CUMULANT still fail on cylinder?",
        "Can either operator handle sphere at D=40?",
    ],
    "results": results,
    "rules": rules,
}
RESULTS_JSON.write_text(json.dumps(combined, indent=2))
print(f"\nSaved: {RESULTS_JSON}")
print(f"Total wall time: {time.time()-t0:.0f}s")
