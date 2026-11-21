"""Apply the R3 (m21017) changes to resolution_study.py: per-run persistence,
compact series recovery lines, interface-layer-corrected front metric, and a
per-arm selector.  Minimal string surgery on the frozen R2 harness."""
from pathlib import Path

p = Path("xval/oracle_minimal_d2q9_fs/resolution_study.py")
s = p.read_text()
orig = s

# ---- 1. DESIGN: round tag + declared offset metric -------------------------
s = s.replace('    "round": "R2",', '    "round": "R3",')
s = s.replace(
    '''    "front_metric": {"Z_cell": "rightmost LIQUID|INTERFACE column / a (R1)",
                     "Z_sub": "sub-cell toe, depth threshold = f*h0",
                     "f_fracs": list(THRESH_FRACS), "f_primary": PRIMARY_FRAC},''',
    '''    "front_metric": {"Z_cell": "rightmost LIQUID|INTERFACE column / a (R1, "
                              "counts the leading INTERFACE layer => +1 cell)",
                     "Z_cell0": "Z_cell - 1/a: same column index with the leading "
                                "INTERFACE layer removed (x_front = x_rightmost - 1); "
                                "= 1 exactly at t = 0 by construction",
                     "Z_sub": "sub-cell toe, depth threshold = f*h0",
                     "f_fracs": list(THRESH_FRACS), "f_primary": PRIMARY_FRAC},
    "interface_offset_cells": 1.0,''')

# ---- 2. snapshot_metrics: offset-corrected cell metric ---------------------
s = s.replace(
    '''def snapshot_metrics(fill: np.ndarray, a: float, h0: float) -> dict:
    depth = depth_profile(fill)
    out = {"Z_cell": front_cells(fill) / a}''',
    '''def snapshot_metrics(fill: np.ndarray, a: float, h0: float) -> dict:
    depth = depth_profile(fill)
    xr = front_cells(fill)
    out = {"Z_cell": xr / a,
           # the initial block spans x in [0, a+1): the leading cell is the
           # INTERFACE layer, so the liquid column ends at x_rightmost - 1 and
           # Z_cell0 = 1 at t = 0 (documented convention, not a fit).
           "Z_cell0": max(xr - DESIGN["interface_offset_cells"], 0.0) / a}''')

# ---- 3. eval_run / analyze metric key lists --------------------------------
s = s.replace('    keys = ["Z_cell"] + [f"Z_sub{f:g}" for f in THRESH_FRACS]\n    pts = []',
              '    keys = ["Z_cell", "Z_cell0"] + [f"Z_sub{f:g}" for f in THRESH_FRACS]\n    pts = []')
s = s.replace('    keys = ["Z_cell"] + [f"Z_sub{f:g}" for f in THRESH_FRACS]\n    for a, r in sorted(runs.items()):',
              '    keys = ["Z_cell", "Z_cell0"] + [f"Z_sub{f:g}" for f in THRESH_FRACS]\n'
              '    for a, r in sorted(runs.items()):')

# ---- 4. run_task: label every result (controls included) -------------------
s = s.replace(
    '''        return hydrostatic_control(**task)
    except Exception as exc:
        return {"kind": kind, "status": "error", **task,
                "error": f"{type(exc).__name__}: {exc}"}''',
    '''        out = hydrostatic_control(**task)
        out["kind"] = "control"
        out["label"] = f"control_{out.get('arm', task.get('arm', 'arm'))}"
        return out
    except Exception as exc:
        return {"kind": kind, "status": "error", **task,
                "error": f"{type(exc).__name__}: {exc}"}


def result_label(r: dict) -> str:
    """Stable per-run artifact name (arm -> arm_aA_tauT, control -> control_arm)."""
    if r.get("label"):
        return str(r["label"])
    if r.get("kind") == "control":
        return f"control_{r.get('arm', 'arm')}"
    return f"{r.get('arm', 'arm')}_a{r.get('a')}_tau{r.get('tau', 1.0):g}"


SER_KEYS = ("Z_cell", "Z_cell0", "Z_sub0.02", "Z_sub0.05", "Z_sub0.1")


def series_row(m: dict) -> list:
    """One [T, Z_cell, Z_cell0, Z_sub.02, Z_sub.05, Z_sub.10] row (log-safe)."""
    row = [round(float(m["T"]), 4)]
    for k in SER_KEYS:
        v = m.get(k)
        row.append(round(float(v), 4) if v is not None else None)
    return row


def series_from_rows(rows: list, steps_hint: int | None = None) -> list:
    """Inverse of ``series_row``: rebuild the ``series`` dicts ``eval_run`` needs."""
    out = []
    for i, row in enumerate(rows):
        m = {"T": row[0]}
        for k, v in zip(SER_KEYS, row[1:]):
            if v is not None:
                m[k] = v
        m["step"] = i if steps_hint is None else int(round(i * steps_hint / max(len(rows) - 1, 1)))
        out.append(m)
    return out''')

# ---- 5. main(): per-arm selector, per-run artifacts, compact R2SER lines ---
s = s.replace(
    '''    ap.add_argument("--tag", default="resolution_ladder.json")''',
    '''    ap.add_argument("--tag", default="resolution_ladder.json")
    ap.add_argument("--only", default="",
                    help="comma-separated label substrings (e.g. a16,a32); default all")''')

s = s.replace(
    '''        print(f"[r2] {len(tasks)} tasks, up to {args.workers} parallel", flush=True)''',
    '''        if args.only:
            sel = [x for x in args.only.split(",") if x]
            tasks = [t for t in tasks
                     if any(k in result_label(t) for k in sel)]
        if not tasks:
            print("[r3] no task selected", flush=True)
            return 2
        print(f"[r3] {len(tasks)} tasks, up to {args.workers} parallel", flush=True)''')

# payload must not carry the field snapshots (they go to npz)
s = s.replace(
    '''        payload = {"design": DESIGN,
                   "env": {"python": sys.version.split()[0], "numpy": np.__version__,
                           "workers": args.workers},
                   "runs": results, "wall_seconds": round(time.time() - t0, 1)}
        (ART / args.tag).write_text(json.dumps(payload, indent=1))''',
    '''        runs_dir = ART / "runs_r3"
        runs_dir.mkdir(parents=True, exist_ok=True)
        for r in results:
            lab = result_label(r)
            slim = {k: v for k, v in r.items()
                    if k not in ("final_fill", "final_umag")}
            slim["label"] = lab
            slim["artifact"] = f"xval/oracle_minimal_d2q9_fs/artifacts/runs_r3/{lab}.json"
            (runs_dir / f"{lab}.json").write_text(json.dumps(slim, indent=1))
            if "final_fill" in r:
                np.savez_compressed(runs_dir / f"{lab}_fields.npz",
                                    fill=np.asarray(r["final_fill"], dtype=np.float32),
                                    umag=np.asarray(r["final_umag"], dtype=np.float32))
        payload = {"design": DESIGN,
                   "env": {"python": sys.version.split()[0], "numpy": np.__version__,
                           "workers": args.workers},
                   "runs": [{k: v for k, v in r.items()
                             if k not in ("final_fill", "final_umag")} for r in results],
                   "run_artifacts": [f"artifacts/runs_r3/{result_label(r)}.json"
                                     for r in results],
                   "wall_seconds": round(time.time() - t0, 1)}
        (ART / args.tag).write_text(json.dumps(payload, indent=1))''')

# compact, log-safe per-run series lines (the cluster log keeps only the tail)
s = s.replace(
    '''        print("R2END:" + args.shard, flush=True)''',
    '''        for r in results:
            lab = result_label(r)
            rec = {"l": lab, "k": r.get("kind", "arm"), "a": r.get("a"),
                   "tau": r.get("tau"), "st": r.get("status"),
                   "steps": r.get("steps"), "wall": r.get("wall_seconds"),
                   "u": (round(float(r["u_max"]), 5) if r.get("u_max") is not None else None),
                   "art": f"artifacts/runs_r3/{lab}.json"}
            if r.get("series"):
                rec["S"] = [series_row(m) for m in r["series"]]
            if r.get("inventory"):
                rec["inv"] = round(max(abs(p["drift_pct"]) for p in r["inventory"]), 4)
            print("R2SER:" + json.dumps(rec, separators=(",", ":"), allow_nan=True),
                  flush=True)
        print("R2END:" + args.shard, flush=True)''')

# ---- 6. analyze: merge every persisted run (sharded jobs compose) ----------
s = s.replace(
    '''    if args.analyze:
        payload = json.loads((ART / args.tag).read_text())''',
    '''    if args.analyze:
        payload = (json.loads((ART / args.tag).read_text())
                   if (ART / args.tag).exists() else {"runs": []})
        seen = {result_label(r) for r in payload["runs"]}
        runs_dir = ART / "runs_r3"
        if runs_dir.exists():
            for f in sorted(runs_dir.glob("*.json")):
                r = json.loads(f.read_text())
                if result_label(r) not in seen:
                    payload["runs"].append(r)
        print(f"[r3] analysing {len(payload['runs'])} runs "
              f"({len(seen)} from {args.tag}, rest from artifacts/runs_r3/)", flush=True)''')

assert s != orig
p.write_text(s)
print("patched", p)