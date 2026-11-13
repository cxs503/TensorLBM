#!/bin/bash
# W26 / m17569 — one-shot finaliser: GCI/sign-surface -> artifact -> figures -> report -> tests.
# Re-runnable: every step regenerates its output from the raw case stores, so running it
# again after a late case lands simply refreshes the artifacts.
set -u
REPO=/root/tp-agent/work/m17569
ART="$REPO/papers/artifacts/w26/lbm-suboff-tau"
RUN="$REPO/.scratch/m17569"
cd "$REPO" || exit 1

# collect the ladder cases into one directory of symlink-free copies (build_gci globs one dir)
LADDER="$RUN/ladder_collected"
rm -rf "$LADDER"; mkdir -p "$LADDER"
python3 - "$RUN" "$LADDER" <<'PYEOF'
import glob, json, math, os, shutil, sys
run, out = sys.argv[1], sys.argv[2]
groups = {}
for d in sorted(glob.glob(os.path.join(run, "runs", "ladder*_N*"))):
    tag = os.path.basename(d)
    cj = os.path.join(d, f"case_{tag}.json")
    if not os.path.exists(cj):
        continue
    fam = tag.rsplit("_N", 1)[0]
    groups.setdefault(fam, []).append((int(json.load(open(cj))["N"]), d, tag))
best = None
for fam, items in groups.items():
    items.sort()
    if len(items) != 3:
        continue
    Ns = [i[0] for i in items]
    rs = [Ns[k + 1] / Ns[k] for k in range(2)]
    if abs(rs[0] - rs[1]) > 1e-9:
        continue
    if best is None or min(Ns) > best[0]:
        best = (min(Ns), fam, items)
if best is None:
    # no complete triplet: fall back to the finest *pair* (two-grid mode in build_gci),
    # never mix grids from different families into one Richardson analysis
    pairs = [(min(i[0] for i in v), fam, v) for fam, v in groups.items() if len(v) >= 2]
    if pairs:
        pairs.sort()
        best = pairs[-1]
        print(f"[finalise] no complete triplet; two-grid mode on {best[1]} N={[i[0] for i in best[2][:2]]}")
        items = best[2][:2]
    else:
        raise SystemExit("[finalise] no usable ladder at all")
else:
    print(f"[finalise] ladder family {best[1]} N={[i[0] for i in best[2]]}")
    items = best[2]
for n, d, tag in items:
    for f in glob.glob(os.path.join(d, "*")):
        if f.endswith((".json", ".npz")):
            shutil.copy2(f, out)
PYEOF
echo "[finalise] ladder cases: $(ls "$LADDER"/case_*.json 2>/dev/null | wc -l)"

# scan cases (stage 1 + stage 2) into one directory
SCAN="$RUN/scan_collected"
rm -rf "$SCAN"; mkdir -p "$SCAN"
for d in "$RUN"/scan/*/ "$RUN"/scan2/*/; do
  [ -d "$d" ] || continue
  cp "$d"/*.json "$d"/*.npz "$SCAN"/ 2>/dev/null
done
echo "[finalise] scan cases: $(ls "$SCAN"/case_*.json 2>/dev/null | wc -l)"

python3 "$ART/build_gci.py" --cases "$LADDER" --scan "$SCAN" --out "$ART/gci_sign_surface.json" || exit 2
python3 "$ART/build_artifact.py" || exit 3
python3 "$ART/make_figures.py" || exit 4
python3 "$ART/build_report.py" || exit 5
python3 -m pytest "$ART/test_artifact.py" -q 2>&1 | tail -n 12