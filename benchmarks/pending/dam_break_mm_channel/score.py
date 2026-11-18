"""Score the Martin–Moyce channel experiments on the two published clocks."""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from zipfile import ZipFile

import numpy as np

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE / "experiment.csv"
REFERENCE_SHA256 = "8361490898d9a618b49673d8f5b48f7d14c1c24031a243c2a878c25d5a5809c3"
ANCHOR_T, ANCHOR_Z = 1.19, 1.44
ERROR_LIMIT_PCT = 3.0
DRIFT_LIMIT = 1e-4


def read_reference():
    data = EXPERIMENT.read_bytes()
    if hashlib.sha256(data).hexdigest() != REFERENCE_SHA256:
        raise ValueError("Experimental table checksum differs from the audited scan transcription")
    rows = csv.DictReader(io.StringIO(data.decode()))
    return [(r["observable"], float(r["mean_time"]), float(r["value"])) for r in rows]


def crossing(times, values, target):
    for t0, t1, y0, y1 in zip(times, times[1:], values, values[1:]):
        if y0 < target <= y1:
            return float(t0 + (t1 - t0) * (target - y0) / (y1 - y0))
    return None


def lookup(t, times, values):
    if t is None or not len(times) or t < times[0] or t > times[-1]:
        return None
    return float(np.interp(t, times, values))


def metrics(points):
    errors = [abs(p["error_pct"]) for p in points if p["error_pct"] is not None]
    return dict(
        required=len(points),
        covered=len(errors),
        within_3pct=sum(e <= ERROR_LIMIT_PCT for e in errors),
        max_abs_pct=max(errors, default=None),
    )


def score_run(meta, data, reference):
    if not len(data) or not np.array_equal(data["step"], np.arange(len(data))):
        raise ValueError("History must begin at release and contain contiguous steps")
    if meta["last_step"] != len(data) - 1 or not np.allclose(
        data["T"], data["step"] * (meta["c0"] / meta["n"]), rtol=0, atol=1e-12
    ):
        raise ValueError("Metadata and history clocks disagree")
    measures = np.isfinite(data["Z"]) & np.isfinite(data["H"])
    if not measures[0] or not measures[:-1].all():
        raise ValueError("Incomplete interior measurement")
    t, z, h = (data[k][measures] for k in ("T", "Z", "H"))
    first = crossing(t, z, ANCHOR_Z)
    shift = None if first is None else ANCHOR_T - first
    points = []
    for observable, clock, target in reference:
        if observable == "front":
            t_sim = None if shift is None else clock - shift
            elapsed = crossing(t, z, target)
            arrival = None if shift is None or elapsed is None else elapsed + shift
            pred = lookup(t_sim, t, z)
            secondary = lookup(clock, t, z)  # unshifted release clock
        else:
            arrival = crossing(t / math.sqrt(2), -h, -target)
            pred = lookup(clock * math.sqrt(2), t, h)
            # Sensitivity to the reported upper bound on start delay, in tau.
            secondary = lookup((clock - 0.16) * math.sqrt(2), t, h)
        points.append(
            dict(
                observable=observable,
                time=clock,
                reference=target,
                prediction=pred,
                secondary_prediction=secondary,
                first_arrival_time=arrival,
                arrival_error_pct=None if arrival is None else 100 * (arrival / clock - 1),
                anchor=observable == "front" and clock == ANCHOR_T,
                error_pct=None if pred is None else 100 * (pred / target - 1),
            )
        )
    curves = {
        k: metrics([p for p in points if p["observable"] == k and not p["anchor"]])
        for k in ("front", "crest")
    }
    peaks = {}
    for key in ("mass_drift", "volume_drift", "mach"):
        series = data[key]
        peaks[key] = float(np.abs(series).max()) if np.isfinite(series).all() else None
    health = (
        meta["stop"] == "window_complete"
        and peaks["mach"] is not None
        and peaks["mach"] <= 0.30
        and np.isfinite(data["rho_min"]).all()
        and (data["rho_min"] > 0).all()
    )
    conservation = all(
        peaks[k] is not None and peaks[k] <= DRIFT_LIMIT for k in ("mass_drift", "volume_drift")
    )
    return dict(
        config=meta,
        last_measured_T=float(t[-1]),
        first_T_at_Z144=first,
        front_time_shift=shift,
        peaks=peaks,
        health_pass=bool(health),
        conservation_pass=conservation,
        curves=curves,
        points=points,
    )


def ladder(runs, collision):
    pair = sorted(
        (
            r
            for r in runs.values()
            if r["config"]["collision"] == collision
            and r["config"]["span"] == "channel"
            and r["config"]["c0"] == 0.06
        ),
        key=lambda r: r["config"]["n"],
    )
    matched, improved = 0, 0
    if len(pair) >= 2:
        for rows in zip(*(r["points"] for r in pair)):
            if rows[0]["anchor"] or any(p["error_pct"] is None for p in rows):
                continue
            errors = [abs(p["error_pct"]) for p in rows]
            matched += 1
            improved += all(b < a for a, b in zip(errors, errors[1:]))
    expected = sum(m["required"] for m in pair[0]["curves"].values()) if pair else 41
    all_points_pass = bool(pair) and all(
        v["within_3pct"] == v["required"] for r in pair for v in r["curves"].values()
    )
    passed = (
        len({r["config"]["n"] for r in pair}) >= 2
        and matched == improved == expected
        and all_points_pass
        and all(r["health_pass"] and r["conservation_pass"] for r in pair)
    )
    return dict(
        grids=[r["config"]["n"] for r in pair],
        required=expected,
        covered_on_all_grids=matched,
        strictly_improved=improved,
        passed=passed,
    )


def read_histories(path):
    def read_case(meta, history):
        data = np.atleast_1d(np.genfromtxt(io.BytesIO(history), delimiter=",", names=True))
        return json.loads(meta), data

    if path.is_dir():
        return {
            p.parent.name: read_case(p.read_bytes(), (p.parent / "history.csv").read_bytes())
            for p in sorted(path.glob("*/run.json"))
        }
    with ZipFile(path) as arc:
        return {
            Path(p).parent.name: read_case(
                arc.read(p), arc.read(str(Path(p).with_name("history.csv")))
            )
            for p in sorted(arc.namelist())
            if p.endswith("/run.json")
        }


def plot(raw, runs, reference, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for obs, ax in zip(("front", "crest"), axs):
        for name, (meta, data) in raw.items():
            if meta["span"] != "channel" or meta["c0"] != 0.06:
                continue
            shift = runs[name]["front_time_shift"]
            if obs == "front" and shift is None:
                continue
            x = data["T"] + shift if obs == "front" else data["T"] / math.sqrt(2)
            y = data["Z" if obs == "front" else "H"]
            ax.plot(
                x,
                y,
                label=f"{meta['collision'].upper()}  N={meta['n']}",
                color={"bgk": "#21649a", "mrt": "#a5481c"}[meta["collision"]],
                ls="-" if meta["n"] == 12 else "--",
            )
        rows = [
            (t, v) for k, t, v in reference if k == obs and not (k == "front" and t == ANCHOR_T)
        ]
        xp, yp = map(np.asarray, zip(*rows))
        ax.fill_between(xp, yp * 0.97, yp * 1.03, color="#a7a7a7", alpha=0.35, label="±3%")
        ax.plot(xp, yp, "ko", ms=3, label="Experiment")
        if obs == "front":
            ax.plot(
                ANCHOR_T, ANCHOR_Z, marker="D", ms=6, mfc="white", mec="black", label="Clock anchor"
            )
        ax.set(
            xlabel="Aligned front time T" if obs == "front" else "Release time τ",
            ylabel="Front position Z = x/a" if obs == "front" else "Crest height H = h/(2a)",
            title="Surge front · Table 2" if obs == "front" else "Residual column · Table 6",
        )
        ax.grid(alpha=0.2)
    axs[0].legend(fontsize=8)
    fig.savefig(out, dpi=145)
    plt.close(fig)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("histories", type=Path, nargs="?", default=HERE / "histories.zip")
    cli.add_argument("--out", type=Path, default=HERE)
    cli.add_argument("--require-pass", action="store_true", help="Exit 1 when admission gates fail")
    args = cli.parse_args()
    reference = read_reference()
    raw = read_histories(args.histories)
    if not raw:
        cli.error("No completed run histories")
    runs = {name: score_run(meta, data, reference) for name, (meta, data) in raw.items()}
    ladders = {k: ladder(runs, k) for k in ("bgk", "mrt")}
    passed = any(row["passed"] for row in ladders.values())
    result = dict(
        case="dam_break_mm_channel",
        status="PASS" if passed else "FAIL",
        passed_3pct_and_converged=passed,
        reference_sha256=REFERENCE_SHA256,
        ladders=ladders,
        runs=runs,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    plot(raw, runs, reference, args.out / "comparison.png")
    if args.require_pass and not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
