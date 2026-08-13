"""Evaluate completed FSI histories against explicit physical KPI acceptance rules."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _load_history(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty history: {path}")
    return {name: np.asarray([float(row[name]) for row in rows]) for name in rows[0]}


def _dominant_frequency(signal: np.ndarray) -> float:
    signal = signal[np.isfinite(signal)]
    if signal.size < 2:
        return 0.0
    centered = signal - signal.mean()
    if np.max(np.abs(centered)) < 1e-15:
        return 0.0
    spectrum = np.abs(np.fft.rfft(centered, n=max(4096, signal.size * 2)))
    spectrum[0] = 0.0
    return float(np.fft.rfftfreq(spectrum.size * 2 - 2)[int(np.argmax(spectrum))])


def evaluate_case(case: str, history: dict[str, np.ndarray], rules: dict[str, Any]) -> dict[str, Any]:
    finite = all(np.isfinite(values).all() for values in history.values())
    n = len(history["step"])
    steady_tip = history["tip_y"][n // 2 :]
    amplitude = 0.5 * float(np.ptp(steady_tip)) if steady_tip.size else 0.0
    checks: dict[str, bool] = {
        "finite": finite,
        "minimum_samples": n >= rules["minimum_samples"],
    }
    metrics: dict[str, float | int] = {"samples": n, "amplitude": amplitude}
    if case == "flag_flapping":
        wake_rms = float(np.sqrt(np.mean(history["uy_probe"] ** 2)))
        metrics["wake_probe_rms"] = wake_rms
        checks["amplitude"] = amplitude > rules["minimum_amplitude"]
        checks["wake_signal"] = wake_rms > rules["minimum_wake_signal"]
    elif case == "turek_hron":
        low, high = rules["amplitude_range"]
        f_tip = _dominant_frequency(steady_tip)
        f_wake = _dominant_frequency(history["uy_probe"][n // 2 :])
        mismatch = math.inf if f_wake <= 0.0 else abs(f_tip / f_wake - 1.0)
        metrics.update({"f_tip": f_tip, "f_wake": f_wake, "frequency_mismatch": mismatch})
        checks["amplitude"] = low <= amplitude <= high
        checks["frequency_lock_in"] = mismatch <= rules["maximum_frequency_mismatch"]
    else:
        raise ValueError(f"unsupported case: {case}")
    return {"case": case, "metrics": metrics, "checks": checks, "pass": all(checks.values())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=Path("configs/fsi_kpi_baselines.json"))
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    rules = json.loads(args.baseline.read_text(encoding="utf-8"))["cases"]
    report: dict[str, Any] = {"cases": []}
    for case, case_rules in rules.items():
        report["cases"].append(evaluate_case(case, _load_history(args.artifacts / case_rules["csv"]), case_rules))
    report["pass"] = all(case["pass"] for case in report["cases"])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
