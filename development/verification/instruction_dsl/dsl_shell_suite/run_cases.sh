#!/bin/sh
# Shell-driven conformance suite for the instruction DSL (POSIX sh + awk only).
#
# Modelled on CFD-GO/TCLB's `instruction-dsl-tests-sh` practice: the tests are
# *shell scripts* that drive the executable entry point and judge its process
# status and streams, so the suite needs no Python test framework, runs in any
# CI image that has the package installed, and fails with a shell exit status.
# Each case is a directory:
#
#     cases/<name>/program.lbm    the instruction text under test
#     cases/<name>/expect.conf    what the CLI must do with it
#
# expect.conf vocabulary (one `key=value` per line, `#` starts a comment):
#
#     exit=N                          required process status (0/2/3/4)
#     error_class=<ClassName>         required `ERROR <Class>: ...` on stderr
#     diagnostic=<substring>          required substring of that diagnostic
#     eq.<field>=<value>              field must equal value (string compare)
#     approx.<field>=<ref>:<reltol>   |got-ref|/|ref| <= reltol
#     abs.<field>=<ref>:<abstol>      |got-ref| <= abstol
#
# Fields are the flat key=value record printed by
# `python -m tensorlbm.instruction_dsl_cli <program> --kv`.
#
# Usage:
#     ./run_cases.sh                 # run every case, exit 0 only if all pass
#     DSL_CASES_DIR=dir ./run_cases.sh
#     DSL_OUT_DIR=dir ./run_cases.sh
#
# Artifacts: $DSL_OUT_DIR/results.tsv (one row per case), results.json
# (machine-readable summary), logs/<case>.out|.err (raw streams, kept verbatim
# so a failure is diagnosable without re-running).
set -u

ROOT=$(cd "$(dirname "$0")" && pwd)
# locate the checkout: the nearest ancestor that has src/tensorlbm
REPO=$ROOT
_depth=0
while [ ! -f "$REPO/src/tensorlbm/instruction_dsl.py" ] && [ "$_depth" -lt 6 ]; do
    REPO=$(cd "$REPO/.." && pwd)
    _depth=$((_depth + 1))
done
if [ ! -f "$REPO/src/tensorlbm/instruction_dsl.py" ]; then
    printf 'cannot locate the TensorLBM checkout above %s\n' "$ROOT" >&2
    exit 1
fi
CASES_DIR=${DSL_CASES_DIR:-$ROOT/cases}
OUT_DIR=${DSL_OUT_DIR:-$ROOT/artifacts}
LOGS="$OUT_DIR/logs"
PY=${PYTHON:-python3}

mkdir -p "$LOGS"
PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH

RESULTS="$OUT_DIR/results.tsv"
: > "$RESULTS"

note() { printf '%s\n' "$*"; }
fail_detail() { PROBLEM="$PROBLEM; $*"; }

# first occurrence of key= in the kv record
kv_field() {
    sed -n "s/^$1=//p" "$2" | head -n 1
}

# numeric comparison: check_numeric <mode:eq|approx|abs> <got> <ref> <tol>
check_numeric() {
    awk -v mode="$1" -v got="$2" -v ref="$3" -v tol="$4" 'BEGIN {
        g = got + 0; r = ref + 0; t = tol + 0
        if (mode == "approx") { d = (r == 0) ? g : (g - r) / r } else { d = g - r }
        if (d < 0) d = -d
        exit (d <= t) ? 0 : 1
    }'
}

total=0
passed=0
failed=0

for dir in "$CASES_DIR"/*/; do
    [ -d "$dir" ] || continue
    name=$(basename "$dir")
    total=$((total + 1))
    PROBLEM=""
    program="$dir/program.lbm"
    expect="$dir/expect.conf"
    out="$LOGS/$name.out"
    err="$LOGS/$name.err"

    if [ ! -f "$program" ] || [ ! -f "$expect" ]; then
        fail_detail "case is missing program.lbm or expect.conf"
        PROBLEM=${PROBLEM#; }
        printf '%s\tFAIL\t%s\n' "$name" "$PROBLEM" >> "$RESULTS"
        failed=$((failed + 1))
        note "FAIL $name -- $PROBLEM"
        continue
    fi

    "$PY" -m tensorlbm.instruction_dsl_cli "$program" --kv >"$out" 2>"$err"
    status=$?

    exp_exit=""
    exp_class=""
    exp_diag=""
    while IFS= read -r line; do
        case $line in
            ''|\#*) continue ;;
            exit=*) exp_exit=${line#exit=} ;;
            error_class=*) exp_class=${line#error_class=} ;;
            diagnostic=*) exp_diag=${line#diagnostic=} ;;
        esac
    done < "$expect"

    if [ -z "$exp_exit" ]; then
        fail_detail "expect.conf has no exit= line"
    elif [ "$status" != "$exp_exit" ]; then
        fail_detail "exit $status != $exp_exit"
    fi

    # stderr diagnostic contract
    err_class=$(sed -n '1s/^ERROR \([A-Za-z]*\): .*$/\1/p' "$err")
    err_msg=$(sed -n '1s/^ERROR [A-Za-z]*: //p' "$err")
    if [ -n "$exp_class" ] && [ "$err_class" != "$exp_class" ]; then
        fail_detail "error_class '$err_class' != '$exp_class'"
    fi
    if [ -n "$exp_diag" ]; then
        case $err_msg in
            *"$exp_diag"*) : ;;
            *) fail_detail "diagnostic lacks '$exp_diag' (got '$err_msg')" ;;
        esac
    fi

    # field assertions
    while IFS= read -r line; do
        case $line in
            ''|\#*) continue ;;
            eq.*)
                key=${line#eq.}
                field=${key%%=*}
                want=${key#*=}
                got=$(kv_field "$field" "$out")
                if [ "$got" != "$want" ]; then
                    fail_detail "$field '$got' != '$want'"
                fi
                ;;
            approx.*|abs.*)
                mode=${line%%.*}
                key=${line#*.}
                field=${key%%=*}
                rest=${key#*=}
                ref=${rest%%:*}
                tol=${rest#*:}
                got=$(kv_field "$field" "$out")
                if [ -z "$got" ]; then
                    fail_detail "field $field missing from the record"
                elif ! check_numeric "$mode" "$got" "$ref" "$tol"; then
                    fail_detail "$field $got vs $ref (tol $mode $tol)"
                fi
                ;;
        esac
    done < "$expect"

    if [ -z "$PROBLEM" ]; then
        passed=$((passed + 1))
        printf '%s\tPASS\t%s\n' "$name" "$(sed -n '1s/^# //p' "$expect")" >> "$RESULTS"
        note "PASS $name"
    else
        failed=$((failed + 1))
        detail=${PROBLEM#; }
        printf '%s\tFAIL\t%s\n' "$name" "$detail" >> "$RESULTS"
        note "FAIL $name -- $detail"
    fi
done

# machine-readable summary (the Python wrapper and the report read this file)
{
    printf '{\n'
    printf '  "cases_dir": "%s",\n' "$CASES_DIR"
    printf '  "total": %s,\n' "$total"
    printf '  "passed": %s,\n' "$passed"
    printf '  "failed": %s,\n' "$failed"
    printf '  "results": [\n'
    first=1
    while IFS='	' read -r cname cstatus cdetail; do
        [ -n "$cname" ] || continue
        # JSON-escape the detail (quotes/backslashes only matter for readability)
        esc=$(printf '%s' "$cdetail" | sed 's/\\/\\\\/g; s/"/\\"/g')
        if [ "$first" = 1 ]; then first=0; else printf ',\n'; fi
        printf '    {"case": "%s", "status": "%s", "detail": "%s"}' \
            "$cname" "$cstatus" "$esc"
    done < "$RESULTS"
    printf '\n  ]\n}\n'
} > "$OUT_DIR/results.json"

note ""
note "cases: $total  passed: $passed  failed: $failed"
[ "$total" -gt 0 ] || { note "no cases found under $CASES_DIR"; exit 1; }
[ "$failed" -eq 0 ] || exit 1
exit 0