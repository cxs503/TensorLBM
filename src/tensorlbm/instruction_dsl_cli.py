#!/usr/bin/env python3
"""Command-line / programmatic entry point for the instruction DSL.

The library module :mod:`tensorlbm.instruction_dsl` knows how to *parse*,
*lower* and *run* a declarative LBM program, but a capability is only
usable if something outside its own test drives it.  This module is that
entry point: it turns ``*.lbm`` instruction text into a JSON run record.

    python3 -m tensorlbm.instruction_dsl_cli program.lbm --out run.json
    python3 -m tensorlbm.instruction_dsl_cli --text "stencil ... "

and programmatically::

    from tensorlbm.instruction_dsl_cli import run_program_file
    record = run_program_file("program.lbm")

Exit-code contract (needed by the shell-driven conformance suite under
``development/verification/instruction_dsl``, which can only observe the process
status and its streams):

=====  ===================================================================
code   meaning
=====  ===================================================================
0      program parsed, lowered and ran; record emitted
2      :class:`DSLSyntaxError` -- the instruction text is not well formed
3      :class:`DSLSemanticError` -- well formed but lattice/physics illegal
4      usage error (bad arguments / unreadable program file)
=====  ===================================================================

A DSL rejection never dumps a traceback: the diagnostic is a single line
``ERROR <ErrorClass>: <message>`` on stderr, so a caller can assert the *kind*
of failure rather than merely that something failed.  ``--kv`` prints the
record as flat ``key=value`` lines (``awk``-friendly, one per line) which is
what the shell suite compares against its expectations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .instruction_dsl import DSLSemanticError, DSLSyntaxError, run, parse

#: process status per failure class (see module docstring)
EXIT_SYNTAX = 2
EXIT_SEMANTIC = 3
EXIT_USAGE = 4


def run_program(text: str, *, sample_every: int = 0) -> Dict[str, Any]:
    """Parse and execute instruction ``text``; return a JSON-serialisable record."""
    prog = parse(text)
    res = run(prog, sample_every=sample_every)
    return {
        "program_digest": prog.digest,
        "steps": res.steps,
        "converged": res.converged,
        "wall_time_s": res.wall_time_s,
        "nu": res.scheme.nu,
        "tau": res.scheme.tau,
        "cs2": res.scheme.cs2,
        "u_max": res.summary()["u_max"],
        "kinetic_energy": res.kinetic_energy,
        "domain": {"nx": prog.nx, "ny": prog.ny},
        "history_points": len(res.history),
        "summary": res.summary(),
    }


def run_program_file(path: str, *, out: Optional[str] = None,
                     sample_every: int = 0) -> Dict[str, Any]:
    """Run the instruction file at ``path``; optionally write the record to ``out``."""
    text = Path(path).read_text(encoding="utf-8")
    record = run_program(text, sample_every=sample_every)
    record["program_path"] = str(path)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def _kv_lines(record: Dict[str, Any]) -> str:
    """Flatten a run record to ``key=value`` lines (shell-friendly)."""
    flat = {
        "program_digest": record["program_digest"],
        "steps": record["steps"],
        "converged": record["converged"],
        "nu": repr(float(record["nu"])),
        "tau": repr(float(record["tau"])),
        "cs2": repr(float(record["cs2"])),
        "u_max": repr(float(record["u_max"])),
        "kinetic_energy": repr(float(record["kinetic_energy"])),
        "nx": record["domain"]["nx"],
        "ny": record["domain"]["ny"],
        "history_points": record["history_points"],
    }
    return "".join(f"{k}={v}\n" for k, v in flat.items())


def _exit_code_for(exc: BaseException) -> int:
    if isinstance(exc, DSLSyntaxError):
        return EXIT_SYNTAX
    if isinstance(exc, DSLSemanticError):
        return EXIT_SEMANTIC
    return 1


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("program", nargs="?", help="instruction file (*.lbm)")
    src.add_argument("--text", help="instruction text on the command line")
    ap.add_argument("--out", help="write the JSON run record here")
    ap.add_argument("--sample-every", type=int, default=0)
    ap.add_argument("--kv", action="store_true",
                    help="print the record as flat key=value lines")
    args = ap.parse_args(argv)

    try:
        if args.program:
            record = run_program_file(args.program, out=args.out,
                                      sample_every=args.sample_every)
        else:
            record = run_program(args.text, sample_every=args.sample_every)
            if args.out:
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                Path(args.out).write_text(
                    json.dumps(record, indent=2, sort_keys=True) + "\n"
                )
    except OSError as exc:  # unreadable program file
        print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except Exception as exc:  # noqa: BLE001 - DSL errors are the contract here
        code = _exit_code_for(exc)
        if code == 1:
            raise
        # single-line diagnostic, no traceback: the failure *class* is part of
        # the contract, and a caller (shell suite) asserts on it.
        msg = str(exc).replace("\n", " ")
        print(f"ERROR {type(exc).__name__}: {msg}", file=sys.stderr)
        return code

    if args.kv:
        sys.stdout.write(_kv_lines(record))
    else:
        print(json.dumps({k: record[k] for k in
                          ("program_digest", "steps", "nu", "u_max", "converged")},
                         indent=2, sort_keys=True))
        sys.stdout.flush()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())