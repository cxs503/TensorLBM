"""m20976 audit: recover the m20893 evidence from the job logs it cites.

The R2 deliverable cites `r2_cj3032_maxima.json` with
``source = cluster/jobs/cj3032.log output_tail``. This script parses the raw
job payloads of cj3032 (ladder + extras, crashed on a KeyError after printing
the summary) and cj3062 (recomputed extras) and re-derives every KPI the report
claims, so the report numbers can be bound to raw evidence instead of prose.

Outputs audit_m20976/evidence/job_log_recovery.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "evidence"
OUT.mkdir(parents=True, exist_ok=True)
JOBS = Path("/srv/tensor/agent/cluster/jobs")

SUMMARY = re.compile(
    r"^\s*(?P<label>\S+)\s+Z_cell\s+(?P<zc>[\d.]+)%\s+Z_sub\.05\s+(?P<zs>[\d.]+)%\s+"
    r"u_max\s+(?P<um>[\d.]+)\s+inv\s+(?P<inv>[\d.]+)%\s+(?P<wall>[\d.]+)s")


def load(jid):
    raw = (JOBS / f"{jid}.log").read_text(errors="replace")
    rec = json.loads(raw)
    tail = rec.get("output_tail", "")
    return rec, tail


out = {"jobs": {}, "summary_lines": {}, "series_runs": {}, "recomputed": {}}

for jid in ("cj3032", "cj3062", "cj3085", "cj3087"):
    p = JOBS / f"{jid}.log"
    if not p.exists():
        out["jobs"][jid] = "MISSING (no log on disk)"
        continue
    rec, tail = load(jid)
    out["jobs"][jid] = {
        "bytes": p.stat().st_size,
        "mtime": p.stat().st_mtime,
        "rc": rec.get("rc"), "node": rec.get("node"), "head": rec.get("head"),
        "job_id_field": rec.get("job_id"),
        "has_R2RUN_lines": "R2RUN:" in tail,
        "tail_chars": len(tail),
    }
    rows = {}
    for line in tail.splitlines():
        m = SUMMARY.match(line)
        if m:
            rows[m["label"]] = {"Z_cell_pct": float(m["zc"]), "Z_sub05_pct": float(m["zs"]),
                                "u_max": float(m["um"]), "inv_drift_pct": float(m["inv"]),
                                "wall_s": float(m["wall"])}
    out["summary_lines"][jid] = rows

    # the payload may carry a JSON blob with full per-T series
    blob = None
    start = tail.find('{"runs"')
    if start >= 0:
        frag = tail[start:]
        # tolerate trailing text after the JSON object
        dec = json.JSONDecoder()
        try:
            blob, _ = dec.raw_decode(frag)
        except Exception as e:  # noqa: BLE001
            blob = None
            out["jobs"][jid]["payload_parse_error"] = str(e)
    if blob:
        runs = {}
        for r in blob.get("runs", []):
            key = r.get("label") or f"{r.get('arm')}_a{r.get('a')}_tau{r.get('tau')}_{r.get('kind')}"
            runs[key] = {
                "arm": r.get("arm"), "a": r.get("a"), "tau": r.get("tau"),
                "kind": r.get("kind"), "status": r.get("status"),
                "steps": r.get("steps"), "n_series": len(r.get("series") or []),
                "u_max": r.get("u_max") or r.get("u_max_lattice"),
                "T_last": (r.get("series") or [{}])[-1].get("T") if r.get("series") else None,
            }
            if r.get("series"):
                runs[key]["series"] = r["series"]
            if r.get("inventory"):
                runs[key]["inventory"] = r["inventory"]
        out["series_runs"][jid] = runs
    else:
        out["series_runs"][jid] = {}

# --- recompute the M&M / Ritter errors straight from the recovered series ---
MM = json.loads((HERE.parent / "src" / "artifacts" /
                 "martin_moyce_1952_dam_break.json").read_text())
MT = [d["t_star"] for d in MM["data"]]
MZ = [d["X_star"] for d in MM["data"]]


def interp(t):
    if t < MT[0] or t > MT[-1]:
        return None
    for i in range(len(MT) - 1):
        if MT[i] <= t <= MT[i + 1]:
            w = (t - MT[i]) / (MT[i + 1] - MT[i])
            return MZ[i] + w * (MZ[i + 1] - MZ[i])
    return None


for jid, runs in out["series_runs"].items():
    for key, r in runs.items():
        if not r.get("series") or r.get("arm") != "tensorlbm_koerner":
            continue
        for metric in ("Z_cell", "Z_sub0.05", "Z_sub0.1"):
            errs = []
            for s in r["series"]:
                T = s["T"]
                if not (0.5 - 1e-9 <= T <= 4.0 + 1e-9):
                    continue
                z = interp(T)
                if z is None or metric not in s:
                    continue
                errs.append(abs(100.0 * (s[metric] - z) / z))
            if errs:
                out["recomputed"].setdefault(key, {})[metric + "_max_abs_err_pct"] = round(max(errs), 3)
        inv = r.get("inventory") or []
        if inv:
            out["recomputed"].setdefault(key, {})["inv_drift_max_pct"] = round(
                max(abs(p["drift_pct"]) for p in inv), 4)

(OUT / "job_log_recovery.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: out["jobs"][k] for k in out["jobs"]}, indent=1))
print("SUMMARY:", json.dumps(out["summary_lines"], indent=1))
print("SERIES RUNS:", json.dumps({j: {k: {kk: vv for kk, vv in v.items() if kk != 'series' and kk != 'inventory'}
                                        for k, v in rs.items()} for j, rs in out["series_runs"].items()}, indent=1))
print("RECOMPUTED:", json.dumps(out["recomputed"], indent=1))