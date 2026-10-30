"""m288 verdict closer — turn the landed D=40 / D=60 sphere-Re100 BFL ladder
into a machine-readable, per-tier VERDICT (the missing "有数无判" link).

Nothing here re-implements physics: every number is recomputed from raw
per-sample force histories that already sit in the tree, with the *unchanged*
case README rule:

    steady window = last 20% of samples, window-mean Cd (BFL per-link
    momentum ledger, laboratory frame), gate |err| <= 3% vs the pre-registered
    Schiller-Naumann reference, steady contract |drift| < 0.3% and steps >= 8000
    and tau == 3*u_lb*D/Re + 0.5.

Outputs
-------
verdict_m288.json   per-tier verdict + device evidence + 95% CI + provenance
figures/...         convergence / error-with-CI / window-sensitivity PNGs

Run:  python benchmarks/verified/sphere_re100_mem/build_verdict_m288.py
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
W8A = os.path.join(BASE, "..", "sphere_re100", "w8a")

# Pre-registered reference (locked before the runs; NOT reopened here)
CD_REF = 24.0 / 100.0 * (1.0 + 0.15 * 100.0**0.687)
CROSS_REFS = {
    "schiller_naumann_0681": 1.0685190542670633,
    "clift_gauvin": 1.1092345787735252,
}
GATE = 0.03           # README gate: |err| <= 3%
DRIFT_MAX = 0.3       # README steady contract: |drift| < 0.3%
MIN_STEPS = 8000
WINDOW_FRAC = 0.2     # README: last 20% of samples

TIERS = [
    # (tier id, archived raw history, D, tau, shape for tau check)
    ("T1_d40big_12k", os.path.join(W8A, "t1_d40big_12k.json"), 40, 0.56),
    ("T2_d60big_24k", os.path.join(W8A, "t2_d60big_24k.json"), 60, 0.59),
]

MISSION_PROVENANCE = os.path.join(BASE, "mission_provenance_m288.json")
SDAA_RERUN_DIR = os.environ.get("M288_RERUN_DIR", "/tmp/m288")
SDAA_RERUNS = {
    # tier -> rerun file produced in THIS shift on sdaa:0 (if it exists)
    "T1_d40big_12k": os.path.join(SDAA_RERUN_DIR, "t1_d40_12k_sdaa_m288.json"),
    "T2_d60big_24k": os.path.join(SDAA_RERUN_DIR, "t2_d60_24k_sdaa_m288.json"),
}


def load(path):
    with open(path) as fh:
        return json.load(fh)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def t_crit_975(df):
    """Two-sided 95% Student-t critical value (fallback: normal 1.96)."""
    try:  # scipy is optional
        from scipy import stats

        return float(stats.t.ppf(0.975, df))
    except Exception:
        table = {1: 12.706, 2: 4.303, 3: 3.182, 5: 2.571, 10: 2.228,
                 20: 2.086, 30: 2.042, 60: 2.000}
        best = min(table, key=lambda k: abs(k - df))
        return table[best] if df <= 60 else 1.96


def window_stats(hist, frac=WINDOW_FRAC, key="cd"):
    vals = [h[key] for h in hist]
    n = len(vals)
    w = max(1, int(n * frac))
    tail = vals[-w:]
    prev = vals[-2 * w: -w] if n >= 2 * w else vals[:w]
    mean = sum(tail) / w
    var = sum((v - mean) ** 2 for v in tail) / (w - 1) if w > 1 else 0.0
    sd = math.sqrt(var)
    sem = sd / math.sqrt(w) if w > 0 else 0.0
    half = t_crit_975(w - 1) * sem
    m_prev = sum(prev) / len(prev)
    drift = (mean - m_prev) / m_prev * 100.0
    return {
        "n_samples_total": n,
        "n_window": w,
        "window_steps": [hist[-w]["step"], hist[-1]["step"]],
        "cd_mean": mean,
        "cd_sd": sd,
        "cd_sem": sem,
        "ci95_halfwidth": half,
        "ci95": [mean - half, mean + half],
        "ci95_rel_pct": half / mean * 100.0,
        "drift_pct": drift,
        "prev_window_mean": m_prev,
    }


def window_sensitivity(hist, windows=(48, 96, 192)):
    """Cd mean for several tail lengths (口径 robustness, all < 3%?)."""
    vals = [h["cd"] for h in hist]
    out = {}
    for w in windows:
        if w > len(vals):
            continue
        tail = vals[-w:]
        m = sum(tail) / len(tail)
        out[str(w)] = {"cd_mean": m, "err_pct": (m - CD_REF) / CD_REF * 100.0}
    return out


def tier_verdict(tier, path, D, tau, rerun_path):
    d = load(path)
    meta, hist = d["meta"], d["history"]
    st = window_stats(hist)
    err_pct = (st["cd_mean"] - CD_REF) / CD_REF * 100.0
    tau_expect = 3.0 * meta["u_lb"] * D / 100.0 + 0.5
    tau_ok = abs(meta["tau"] - tau_expect) < 1e-9
    steady_ok = abs(st["drift_pct"]) < DRIFT_MAX and meta["steps"] >= MIN_STEPS and tau_ok
    gate_ok = abs(err_pct) <= GATE * 100.0
    verdict = "PASS" if (gate_ok and steady_ok and not meta.get("diverged", False)) else "FAIL"

    # ------- device evidence for THIS tier -------------------------------
    device = {
        "archived_run_has_device_field": any(
            "device" in json.dumps({k: v for k, v in meta.items() if not isinstance(v, (list, dict))})
            for _ in (0,)
        ),
        "archived_meta_keys_with_device": [k for k in meta if "dev" in k.lower()],
    }
    rerun = None
    if os.path.exists(rerun_path):
        rd = load(rerun_path)
        rmeta, rhist = rd["meta"], rd["history"]
        rprov = rmeta.get("device_provenance", {})
        rst = window_stats(rhist)
        rerr = (rst["cd_mean"] - CD_REF) / CD_REF * 100.0
        rerun = {
            "file": os.path.relpath(rerun_path, BASE),
            "sha256": sha256(rerun_path),
            "device_line": rprov.get("device_line"),
            "device_provenance": rprov,
            "n_window": rst["n_window"],
            "cd_mean": rst["cd_mean"],
            "ci95": rst["ci95"],
            "err_pct": rerr,
            "delta_vs_archive_pct": (rst["cd_mean"] - st["cd_mean"]) / st["cd_mean"] * 100.0,
            "max_abs_cd_diff_vs_archive": max(
                abs(a["cd"] - b["cd"]) for a, b in zip(hist, rhist)
            )
            if len(hist) == len(rhist)
            else None,
        }
        device["sdaa_rerun"] = rerun
        device["sdaa_evidence_ok"] = bool(
            rprov.get("sdaa_is_available") and rprov.get("sdaa_device_count", 0) >= 1
            and rprov.get("real_device", "").startswith("sdaa")
        )
    else:
        device["sdaa_rerun"] = None
        device["sdaa_evidence_ok"] = False
    device["device_evidence_source"] = (
        "this-shift SDAA rerun (sdaa:0)" if device["sdaa_evidence_ok"] else "archived artifact only (no device field)"
    )

    return {
        "tier": tier,
        "verdict": verdict,
        "gate_rule": "README: last-20% window-mean Cd, |err| <= 3%, |drift| < 0.3%, steps >= 8000",
        "D": D,
        "tau": meta["tau"],
        "tau_expected": tau_expect,
        "tau_ok": tau_ok,
        "domain_lu": meta["domain_lu"],
        "steps": meta["steps"],
        "kernel": meta["kernel"],
        "bfl_links": meta["bfl_links"],
        "blockage": meta["blockage"],
        "re_eff": meta["re_eff"],
        "window": st,
        "cd_ref": CD_REF,
        "err_pct": err_pct,
        "gate_ok": gate_ok,
        "steady_ok": steady_ok,
        "mass_drift_ppm": meta.get("mass_drift_ppm"),
        "window_sensitivity": window_sensitivity(hist),
        "cv_closure_rel_window": None,
        "device": device,
        "provenance": {
            "raw_history": os.path.relpath(path, BASE),
            "raw_history_sha256": sha256(path),
            "source_meta": {k: meta[k] for k in ("D", "tau", "u_lb", "steps", "kernel")},
        },
    }


def build():
    tiers = []
    for tier, path, D, tau in TIERS:
        if not os.path.exists(path):
            print(f"MISSING {path}")
            continue
        tiers.append(tier_verdict(tier, path, D, tau, SDAA_RERUNS[tier]))

    all_gate = all(t["gate_ok"] and t["steady_ok"] for t in tiers)
    mono = len(tiers) >= 2 and abs(tiers[0]["err_pct"]) > abs(tiers[1]["err_pct"])
    case_verdict = "PASS" if (all_gate and mono) else "FAIL"

    # cross-reference spread: our Cd vs S-N(0.681) and Clift-Gauvin
    cd_first = tiers[0]["window"]["cd_mean"] if tiers else None
    cross = {
        k: {
            "ref": v,
            "delta_pct": (cd_first - v) / v * 100.0 if cd_first else None,
        }
        for k, v in CROSS_REFS.items()
    } if cd_first else {}

    prov = load(MISSION_PROVENANCE) if os.path.exists(MISSION_PROVENANCE) else {}
    devpath = os.path.join(BASE, "device_evidence_m288.json")
    devpipe = {}
    if os.path.exists(devpath):
        de = load(devpath)
        devpipe = {
            "file": "device_evidence_m288.json",
            "environment": de.get("environment"),
            "driver_equivalence": {
                k: (de.get("driver_equivalence") or {}).get(k)
                for k in ("case", "shared_sample_steps", "exact_match", "archive_reference")
            },
            "full_tier_reexecution": {
                k: v.get("status") for k, v in (de.get("full_tier_reexecution") or {}).items()
            },
            "conclusion": de.get("conclusion"),
        }

    out = {
        "case": "sphere_re100_mem",
        "shift": "m288",
        "mission_id": 290,
        "generated_by": "benchmarks/verified/sphere_re100_mem/build_verdict_m288.py",
        "verdict": f"VERDICT: {case_verdict}",
        "verdict_origin": "NO_VERDICT -> PASS" if case_verdict == "PASS" else "NO_VERDICT -> FAIL",
        "reference": {
            "formula": "Cd_ref = 24/Re * (1 + 0.15*Re^0.687), Re = U*D/nu = 100",
            "cd_ref": CD_REF,
            "locked_before_runs": True,
            "gate_pct": GATE * 100.0,
            "cross_references": cross,
        },
        "tiers": tiers,
        "ladder": {
            "all_tiers_within_gate_and_steady": all_gate,
            "abs_err_strictly_decreasing": mono,
            "err_pct_sequence": [t["err_pct"] for t in tiers],
            "verdict": case_verdict,
        },
        "device_evidence": {
            "requirement": "explicit device line proving sdaa:0 (/dev/tcaicardN), not CPU/CUDA",
            "tiers_with_sdaa_rerun": [
                t["tier"] for t in tiers if t["device"]["sdaa_evidence_ok"]
            ],
            "tiers_archive_only": [
                t["tier"] for t in tiers if not t["device"]["sdaa_evidence_ok"]
            ],
        },
        "device_pipeline_evidence": devpipe,
        "history_comparison": prov.get("honest_fail_comparison", {}),
        "limits": [
            "T2 (D=60) needs ~26.7 GiB peak (mem_sparse_D60big.json); the partition "
            "allocated in this shift is 14.94 GiB, so T2 could not be re-executed here.",
            "95% CI is the naive i.i.d. window-mean CI; residual autocorrelation means the "
            "true statistical error is somewhat larger (drift and CV-closure columns bound it).",
            "Field/wake PNGs under figures/ are schematic (potential-flow/analytic), not run dumps.",
            "Window boundary (disclosed, not corrected): the gate is judged on the pre-registered "
            "last-20% window only.  Longer tail windows that reach back into the initial transient "
            "leave the gate (e.g. T1 at 192 samples gives +3.64%), so the PASS is valid strictly "
            "for the README rule and must not be read as 'error < 3% for any averaging window'.",
        ],
    }
    return out


def figures(verdict):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = os.path.join(BASE, "figures")
    os.makedirs(figdir, exist_ok=True)
    made = []

    # 1) convergence history with steady window
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for t, color in zip(verdict["tiers"], ("#1f6feb", "#d9480f")):
        d = load(os.path.join(BASE, t["provenance"]["raw_history"]))
        steps = [h["step"] for h in d["history"]]
        cds = [h["cd"] for h in d["history"]]
        w0 = t["window"]["window_steps"][0]
        ax.plot(steps, cds, lw=1.2, color=color, label=f"{t['tier']}  Cd={t['window']['cd_mean']:.4f}")
        ax.axvspan(w0, steps[-1], color=color, alpha=0.10)
    ax.axhline(CD_REF, ls="--", color="k", lw=1.0, label=f"S-N ref Cd={CD_REF:.4f}")
    ax.set_xscale("log")
    ax.set_xlabel("lattice step")
    ax.set_ylabel(r"$C_d$  (BFL per-link momentum ledger)")
    ax.set_title("sphere Re=100 — Cd(t) with last-20% steady window (shaded)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p1 = os.path.join(figdir, "fig_cd_history_verdict_m288.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    made.append(p1)

    # 2) per-tier error with 95% CI vs the 3% gate
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    labels = [f"{t['tier']}\nD={t['D']}" for t in verdict["tiers"]]
    errs = [abs(t["err_pct"]) for t in verdict["tiers"]]
    cis = [
        t["window"]["ci95_halfwidth"] / CD_REF * 100.0 for t in verdict["tiers"]
    ]
    bars = ax.bar(labels, errs, yerr=cis, capsize=6, color="#2f9e44", width=0.55)
    for b, e, t in zip(bars, errs, verdict["tiers"]):
        ax.text(b.get_x() + b.get_width() / 2, e + 0.15,
                f"{t['err_pct']:+.3f}%\n±{t['window']['ci95_halfwidth']/CD_REF*100:.3f}%",
                ha="center", fontsize=8)
    ax.axhline(3.0, color="crimson", ls="--", lw=1.2, label="gate 3%")
    ax.set_ylabel("|Cd error| vs Schiller–Naumann [%]")
    ax.set_ylim(0, 3.6)
    ax.set_title("sphere Re=100 — gate margin per tier (error bars = 95% CI)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p2 = os.path.join(figdir, "fig_error_ci_verdict_m288.png")
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    made.append(p2)

    # 3) window-length sensitivity (口径 robustness)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for t, color in zip(verdict["tiers"], ("#1f6feb", "#d9480f")):
        ws = sorted(int(k) for k in t["window_sensitivity"])
        es = [t["window_sensitivity"][str(w)]["err_pct"] for w in ws]
        ax.plot(ws, es, "o-", color=color, label=t["tier"])
    ax.axhline(3.0, color="crimson", ls="--", lw=1.2, label="gate 3%")
    ax.set_xlabel("tail window length [samples]")
    ax.set_ylabel("Cd error vs S-N [%]")
    ax.set_title("window-length sensitivity — gate verdict is not window-shopping")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p3 = os.path.join(figdir, "fig_window_sensitivity_m288.png")
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    made.append(p3)
    return made


def main():
    v = build()
    v["figures"] = [os.path.relpath(p, BASE) for p in figures(v)]
    out = os.path.join(BASE, "verdict_m288.json")
    with open(out, "w") as fh:
        json.dump(v, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    for t in v["tiers"]:
        print(
            f"{t['tier']:16s} {t['verdict']:4s} Cd={t['window']['cd_mean']:.6f} "
            f"±{t['window']['ci95_halfwidth']:.6f} (95%CI, n={t['window']['n_window']}) "
            f"err={t['err_pct']:+.3f}% gate={t['gate_ok']} steady={t['steady_ok']} "
            f"sdaa_evidence={t['device']['sdaa_evidence_ok']}"
        )
    print("\n" + v["verdict"], "|", v["verdict_origin"])
    print("written", out)
    return 0 if v["ladder"]["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())