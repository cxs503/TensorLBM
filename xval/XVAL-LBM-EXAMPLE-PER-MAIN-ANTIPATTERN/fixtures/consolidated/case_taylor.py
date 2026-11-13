"""Control fixture: thin main -- selects a case, runs it, writes the record.

No lattice, no time loop, no post-processing here: that is the whole point.
"""
import json
import sys

from tensorlbm.example_harness import get_case, run_case


def main(argv):
    ny = int(argv[0]) if argv else 32
    record = run_case(get_case("taylor_green_2d", ny=ny, steps=400), keep_history=False)
    print(json.dumps({"case": record["case"], "kpis": record["kpis"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
