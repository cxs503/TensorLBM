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
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .instruction_dsl import parse, run


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


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("program", nargs="?", help="instruction file (*.lbm)")
    src.add_argument("--text", help="instruction text on the command line")
    ap.add_argument("--out", help="write the JSON run record here")
    ap.add_argument("--sample-every", type=int, default=0)
    args = ap.parse_args(argv)
    if args.program:
        record = run_program_file(args.program, out=args.out,
                                  sample_every=args.sample_every)
    else:
        record = run_program(args.text, sample_every=args.sample_every)
        if args.out:
            Path(args.out).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: record[k] for k in
                      ("program_digest", "steps", "nu", "u_max", "converged")},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
